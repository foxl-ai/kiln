"""Compile farm: build a sweep's NEFFs on CPU instances before the device box needs them.

    # shapes and config of a checkpoint (local dir with safetensors, or a Hugging Face repo id)
    python tools/compile_farm.py shapes zai-org/GLM-5.3-Flash /opt/kiln/shapes/glm53
    # the load-time decisions that read weight values (kiln/capture.py "Load-time decisions"),
    # from the real checkpoint by range reads, into the shape dir; a capture that needs one and
    # does not find it fails (GLM-5.3-Flash with KILN_MOE_PREFILL_KERNEL=nki needs them)
    python tools/compile_farm.py decisions --model zai-org/GLM-5.3-Flash --shape-dir /opt/kiln/shapes/glm53 --tp 32

    # capture every graph a serve_sweep configuration runs (one process per TP rank, meta
    # device, no NeuronCore needed): graph.hlo + metadata + command.txt in the local cache
    python tools/compile_farm.py capture --shape-dir /opt/kiln/shapes/glm53 -- \
        --model zai-org/GLM-5.3-Flash --tp 32 --dp-attention 8 --piecewise ...
    # the same for tools/check_device.py or tools/check_ppl.py arguments (--plp: prompt logprobs,
    # which check_ppl runs); --skip-prefill for decode graphs only (tools/time_decode.py)
    python tools/compile_farm.py capture --tool check_ppl --plp --shape-dir ... -- --model ... --tp 32

    Everything the device run's keys depend on must match: the code, the environment (KILN_MOE_KERNEL,
    KILN_PIECEWISE_*, KILN_MOE_PREFILL_*, KILN_CC_ARGS) and the arguments.

    # entries (graph.hlo + metadata) and the NKI kernel binaries they reference, from S3
    python tools/compile_farm.py fetch s3://<your-bucket>/compile-farm/<set>/ [--keys k1,k2]

    # compile every entry under the local cache that has graph.hlo and no completion marker, N at
    # once, in LNL's layout, then push the finished entries (kiln/compile_cache.py)
    python tools/compile_farm.py compile --workers 6 [--push s3://.../compile-cache/sdk-2.32/]

    # fan-out: put captured entries on an S3 queue, then run `work` on any number of CPU hosts;
    # each graph is claimed by exactly one host and lands in the cache prefix device boxes pull
    python tools/compile_farm.py enqueue --queue s3://.../compile-farm/q/<job>/ \
        --cache s3://<your-bucket>/compile-cache/sdk-2.32/ [--keys-file capture/keys.json]
    python tools/compile_farm.py work --queue s3://.../compile-farm/q/<job>/ [--mem-budget-gb 680]

    # measurement: compile entries into scratch directories (never into the cache), R copies of
    # each at once, with extra flags; one JSON line per compile plus a host summary
    python tools/compile_farm.py bench --keys k1,k2 --repeat 4 --extra "-O1" --out bench.jsonl

The per-entry mechanics (arguments, kernel references, the marker) are in kiln/compile_farm.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kiln import compile_cache, compile_farm  # noqa: E402

BUCKET_REGION = "us-east-2"


def _s3(*args: str) -> None:
    subprocess.run(["aws", "--region", BUCKET_REGION, "s3", *args], check=True)


def cmd_fetch(a) -> None:
    root = a.root or compile_cache.local_root()
    os.makedirs(root, exist_ok=True)
    src = a.src.rstrip("/")
    keys = a.keys.split(",") if a.keys else None
    if keys:
        for k in keys:
            _s3("sync", "--quiet", f"{src}/cache/{k}/", os.path.join(root, k), "--exclude", "*.lock")
    else:
        _s3("sync", "--quiet", f"{src}/cache/", root, "--exclude", "*.lock", "--exclude", ".*")
    os.makedirs(compile_cache.NKI_KERNEL_DIR, exist_ok=True)
    _s3("sync", "--quiet", f"{src}/nki-kernels/", compile_cache.NKI_KERNEL_DIR)
    print(f"fetched {len(keys) if keys else 'all'} entries into {root}", flush=True)


class HostMonitor:
    """Peak used memory and mean CPU utilisation of the whole host while a batch runs."""

    def __init__(self, every: float = 5.0):
        import psutil

        self.psutil, self.every = psutil, every
        self.peak_used, self.cpu, self._stop = 0, [], threading.Event()
        psutil.cpu_percent()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.every):
            vm = self.psutil.virtual_memory()
            self.peak_used = max(self.peak_used, vm.total - vm.available)
            self.cpu.append(self.psutil.cpu_percent())

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def summary(self) -> dict:
        vm = self.psutil.virtual_memory()
        return {"host_mem_total_gb": round(vm.total / 2**30, 1),
                "host_peak_used_gb": round(self.peak_used / 2**30, 1),
                "host_cpu_mean_pct": round(sum(self.cpu) / max(1, len(self.cpu)), 1),
                "host_cpu_max_pct": max(self.cpu, default=0.0),
                "vcpus": os.cpu_count()}


def pending(root: str) -> list[str]:
    out = []
    for k in sorted(os.listdir(root)):
        d = os.path.join(root, k)
        if (os.path.isdir(d) and os.path.exists(os.path.join(d, compile_farm.HLO))
                and not os.path.exists(os.path.join(d, compile_farm.MARKER))):
            out.append(d)
    return out


def _print(res: compile_farm.Result, extra: dict | None = None, out: str | None = None) -> None:
    row = {**compile_farm.result_dict(res), **(extra or {})}
    print(json.dumps(row), flush=True)
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")


def cmd_compile(a) -> None:
    root = a.root or compile_cache.local_root()
    todo = pending(root)
    if a.keys:
        want = set(a.keys.split(","))
        todo = [d for d in todo if os.path.basename(d) in want]
    # Longest first (HLO size is the best cheap proxy), so the tail of the batch is short graphs.
    todo.sort(key=lambda d: -os.path.getsize(os.path.join(d, compile_farm.HLO)))
    print(f"{len(todo)} entries to compile with {a.workers} workers", flush=True)
    t0 = time.time()
    with HostMonitor() as mon, ThreadPoolExecutor(a.workers) as pool:
        results = list(pool.map(lambda d: compile_farm.run_compile(d, work_root=a.work_root), todo))
    for r in results:
        _print(r, out=a.out)
    bad = [r for r in results if not r.ok]
    print(json.dumps({"summary": True, "entries": len(results), "failed": len(bad), "workers": a.workers,
                      "wall_seconds": round(time.time() - t0, 1), **mon.summary()}), flush=True)
    if a.push:
        print("pushed", compile_cache.push(a.push), flush=True)
    if bad:
        sys.exit(1)


def cmd_bench(a) -> None:
    root = a.root or compile_cache.local_root()
    keys = a.keys.split(",")
    extra = shlex.split(a.extra) if a.extra else []
    jobs = [(k, i) for k in keys for i in range(a.repeat)]
    scratch = a.scratch
    os.makedirs(scratch, exist_ok=True)
    workers = a.workers or len(jobs)
    tag = a.tag or (" ".join(extra) or "default")

    def one(job):
        k, i = job
        out = os.path.join(scratch, f"{k}.{tag.replace(' ', '_')}.{i}")
        r = compile_farm.run_compile(os.path.join(root, k), out_dir=out, extra=extra, work_root=a.work_root)
        if not a.keep and r.ok:
            os.remove(os.path.join(out, compile_farm.neff_name(k)))
        return r

    t0 = time.time()
    with HostMonitor() as mon, ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, jobs))
    wall = time.time() - t0
    for r in results:
        _print(r, {"tag": tag, "concurrent": workers}, a.out)
    summary = {"summary": True, "tag": tag, "jobs": len(jobs), "concurrent": workers,
               "failed": sum(not r.ok for r in results), "wall_seconds": round(wall, 1),
               "graphs_per_hour": round(len(jobs) * 3600 / wall, 2), **mon.summary()}
    print(json.dumps(summary), flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(summary) + "\n")


def cmd_shapes(a) -> None:
    from kiln import capture

    if os.path.isdir(a.model):
        capture.write_shape_dir(a.model, a.dest)
    else:
        capture.write_shape_dir_hub(a.model, a.dest)
    with open(os.path.join(a.dest, capture.HEADERS)) as f:
        n = len(json.load(f))
    print(f"{a.dest}: config + {n} tensor headers", flush=True)


def cmd_decisions(a) -> None:
    from kiln import capture, platform

    fp8_max = a.fp8_max if a.fp8_max else platform.fp8_max(a.target)
    ranks = None if a.ranks == "all" else [int(x) for x in a.ranks.split(",")]
    rec = capture.compute_decisions(a.model, a.shape_dir, a.tp, fp8_max, ranks=ranks,
                                    prefixes=a.layers.split(",") if a.layers else None, procs=a.procs,
                                    revision=a.revision)
    mine = {k: v for k, v in rec["decisions"].items() if k.startswith(f"fp8_max={fp8_max:g} tp={a.tp} ")}
    vals = sorted({(v["moe_prefill_dq"], v["moe_prefill_down"]) for v in mine.values()})
    print(f"{a.shape_dir}/{capture.DECISIONS}: {len(mine)} (layer, rank) decisions at fp8_max={fp8_max:g} "
          f"tp={a.tp} from {rec['model']}@{rec['revision']} in {rec['seconds']} s; (dq, down) values {vals}; "
          f"experts read per decision: max {max(v['experts_read'] for v in mine.values())}", flush=True)


# The tools whose EngineConfig a capture can rebuild: module, directory, and whether engine_config
# takes the model path (the check tools resolve it in main()).
TOOLS = {"serve_sweep": ("bench", False), "check_device": ("tools", True), "check_ppl": ("tools", True)}


def _tool(name: str):
    import importlib

    sub, _ = TOOLS[name]
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), sub)
    if d not in sys.path:
        sys.path.insert(0, d)
    return importlib.import_module(name)


def _capture_rank(rank: int, sweep_argv: list[str], shape_dir: str, target: str, out: str,
                  skip_prefill: bool = False, tool: str = "serve_sweep", plp: bool = False) -> None:
    """One tensor-parallel rank's capture, in its own process (dynamo and the process group are
    process-global)."""
    from kiln import capture

    capture.configure_env(target)
    import dataclasses

    import torch

    mod = _tool(tool)
    args = mod.build_parser().parse_args(sweep_argv)
    cfg = mod.engine_config(args, shape_dir) if TOOLS[tool][1] else mod.engine_config(args)
    torch.set_num_threads(2)
    # The process group BEFORE libtorch_neuronx_lite, the order a device rank has (engine/tp.py
    # init_rank, then build_shard imports LNL): the other order traces torch.topk / argmax with
    # different spellings, and so different cache keys (kiln/capture.py init_fake_world).
    if cfg.tp > 1:
        capture.init_fake_world(rank, cfg.tp)
    # What build_shard does on a device before LNL: the LNC for the target (NEURON_LOGICAL_NC_CONFIG
    # on trn2 / trn3), which kiln.platform.nki_grid hands every NKI kernel traced into a graph.
    try:
        from kiln import platform as kplatform

        if hasattr(kplatform, "configure_runtime_env"):
            kplatform.configure_runtime_env()
    except ImportError:
        pass
    import libtorch_neuronx_lite  # noqa: F401

    from kiln.config import ModelConfig
    from kiln.engine.engine import build_shard, pool_pages

    capture.install_header_checkpoint()
    mcfg = ModelConfig.from_pretrained(shape_dir)
    if cfg.num_layers is not None:
        mcfg = mcfg.truncated(cfg.num_layers)
    t0 = time.time()
    num_pages = pool_pages(cfg, mcfg)
    _, runner = build_shard(dataclasses.replace(cfg, device="meta"), shape_dir, num_pages, rank)
    t1 = time.time()
    if skip_prefill:  # decode graphs only (tools/time_decode.py)
        runner.prefill_buckets = []
    w = runner.warmup(1 + cfg.spec_k if cfg.spec_method else None, plp=plp)
    rec = {"rank": rank, "build_seconds": round(t1 - t0, 1), "warmup_seconds": round(w["seconds"], 1),
           "graphs": w["graphs"], "num_pages": num_pages, "keys": capture.CAPTURED}
    with open(out, "w") as f:
        json.dump(rec, f)


def cmd_capture(a) -> None:
    import multiprocessing as mp

    sweep = _tool(a.tool).build_parser().parse_args(a.sweep)
    tp = sweep.tp
    ranks = list(range(tp)) if a.ranks == "all" else [int(r) for r in a.ranks.split(",")]
    os.makedirs(a.out_dir, exist_ok=True)
    ctx = mp.get_context("spawn")
    t0 = time.time()

    def run(batch):
        procs = []
        for r in batch:
            p = ctx.Process(target=_capture_rank, args=(r, a.sweep, a.shape_dir, a.target,
                                                       os.path.join(a.out_dir, f"rank{r}.json"), a.skip_prefill,
                                                       a.tool, a.plp))
            p.start()
            procs.append((r, p))
        bad = []
        for r, p in procs:
            p.join()
            if p.exitcode != 0:
                bad.append((r, p.exitcode))
        return bad

    # Rank 0 first, alone: it compiles the NKI kernels and writes every graph shared by all
    # ranks, so the others mostly find their entries already there.
    bad = run(ranks[:1])
    rest = ranks[1:]
    for i in range(0, len(rest), a.procs):
        bad += run(rest[i : i + a.procs])
    keys: dict[str, list[int]] = {}
    for r in ranks:
        p = os.path.join(a.out_dir, f"rank{r}.json")
        if os.path.exists(p):
            with open(p) as f:
                rec = json.load(f)
            print(json.dumps({k: v for k, v in rec.items() if k != "keys"}), flush=True)
            for k in rec["keys"]:
                keys.setdefault(k, []).append(r)
    with open(os.path.join(a.out_dir, "keys.json"), "w") as f:
        json.dump(keys, f, indent=1)
    # The exact command line and environment the keys belong to, for a device run to copy verbatim
    # (a sweep once ran 1024 / 256 prefill against graphs captured at 2048 / 512 and missed every key).
    import shlex

    env = {k: v for k, v in sorted(os.environ.items())
           if k.startswith("KILN_") and k not in ("KILN_HASH_DUMP",)
           or k in ("NEURON_LOGICAL_NC_CONFIG", "NEURON_PLATFORM_TARGET_OVERRIDE")}
    # Knobs whose default has changed, written out with the value this tree takes when they are unset,
    # so a config copied onto another tree still names the graphs it was built for:
    # KILN_LINEAR_ATTN_KERNEL defaulted to torch until engine-v0 77862e6 and to nki since.
    from kiln.models import linear_attn

    env.setdefault("KILN_LINEAR_ATTN_KERNEL", linear_attn.LINEAR_ATTN_KERNEL)
    env = dict(sorted(env.items()))
    from kiln import capture

    dec = capture.read_decisions(a.shape_dir)
    tool_path = {"serve_sweep": "bench/serve_sweep.py", "check_device": "tools/check_device.py",
                 "check_ppl": "tools/check_ppl.py"}[a.tool]
    with open(os.path.join(a.out_dir, "config.json"), "w") as f:
        json.dump({"target": a.target, "tool": a.tool, "argv": a.sweep, "env": env,
                   "command": " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()
                                       if k != "NEURON_PLATFORM_TARGET_OVERRIDE")
                              + f" python {tool_path} " + shlex.join(a.sweep),
                   "graphs": len(keys), "ranks": len(ranks),
                   # the checkpoint revision the load-time decisions came from: the device must load the same one
                   "decisions": {k: dec.get(k) for k in ("model", "revision", "written")}}, f, indent=1)
    print(json.dumps({"summary": True, "ranks": len(ranks), "failed_ranks": bad, "distinct_graphs": len(keys),
                      "wall_seconds": round(time.time() - t0, 1)}), flush=True)
    if bad:
        sys.exit(1)


def _manifest(queue: str) -> dict:
    """{"cache": uri, "keys": {key: graph.hlo bytes}}: manifest.json (the cache, and the keys of
    queues written before per-key records) merged with every keys/<key>__<bytes> record. One object
    per key, because enqueues running at once each rewrote the shared manifest and the last one won
    (measured: two of four parallel enqueues lost their keys, 2026-10-04)."""
    q = queue.rstrip("/")
    p = subprocess.run(["aws", "--region", BUCKET_REGION, "s3", "cp", "--quiet", q + "/manifest.json", "-"],
                       capture_output=True, text=True)
    man = json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip() else {}
    man.setdefault("keys", {})
    ls = subprocess.run(["aws", "--region", BUCKET_REGION, "s3", "ls", q + "/keys/"], capture_output=True, text=True)
    man["keys"].update(compile_cache.parse_key_records(ls.stdout))
    return man


def cmd_enqueue(a) -> None:
    root = a.root or compile_cache.local_root()
    q = a.queue.rstrip("/")
    if a.keys_file:
        with open(a.keys_file) as f:
            keys = list(json.load(f))
    else:
        keys = [os.path.basename(d) for d in pending(root)]
    # The manifest holds the cache uri; created once (a conditional write), never rewritten.
    if not compile_farm.s3_claim(f"{q}/manifest.json", json.dumps({"cache": a.cache, "keys": {}})):
        have = _manifest(q).get("cache")
        if have and have.rstrip("/") != a.cache.rstrip("/"):
            raise SystemExit(f"queue {q} pushes to {have}, not {a.cache}")
    compile_cache.push_nki(a.cache)  # kernel binaries: every compile host and device box pulls them
    n = 0
    for k in keys:
        d = os.path.join(root, k)
        compiled = compile_farm.s3_exists(f"{a.cache.rstrip('/')}/{k}/{compile_farm.MARKER}")
        if not compiled:  # else compiled already, by an earlier run or a device box: no entry to upload
            for fn in compile_farm.ENTRY_FILES:
                if os.path.exists(os.path.join(d, fn)):
                    _s3("cp", "--quiet", os.path.join(d, fn), f"{q}/entries/{k}/{fn}")
        # Last: a worker that sees the key finds its entry complete. Compiled keys are recorded too, so
        # the queue lists its whole configuration and a device's FarmWait fetches them rather than
        # compiling (a queue whose keys were all cached used to list none).
        compile_farm.s3_claim(f"{q}/keys/{k}__{os.path.getsize(os.path.join(d, compile_farm.HLO))}", "")
        n += not compiled
    reindex(q)
    print(json.dumps({"enqueued": n, "skipped_compiled": len(keys) - n, "queue": q, "cache": a.cache}), flush=True)


def reindex(q: str) -> int:
    """Rewrite manifest.json's keys as the union of every keys/ record, for readers that predate the
    per-key records (kiln/compile_cache.py FarmWait before 2026-10-04 read manifest.json only, so a
    device run saw an empty queue and compiled what the farm had). Idempotent: a rewrite that raced
    another enqueue is completed by that enqueue's own rewrite."""
    man = _manifest(q)
    path = f"/tmp/kiln-farm-manifest-{os.getpid()}.json"
    with open(path, "w") as f:
        json.dump({"cache": man["cache"], "keys": man["keys"]}, f)
    _s3("cp", "--quiet", path, f"{q}/manifest.json")
    os.unlink(path)
    return len(man["keys"])


def backfill(q: str) -> int:
    """Record every key of the queue's configs/*.keys.json that is complete in its cache but has no keys/
    record: the keys an enqueue skipped as already compiled before it recorded them. Returns how many."""
    man = _manifest(q)
    cache = man["cache"].rstrip("/")
    ls = subprocess.run(["aws", "--region", BUCKET_REGION, "s3", "ls", f"{q}/configs/"], capture_output=True, text=True)
    want: set[str] = set()
    for line in ls.stdout.splitlines():
        name = line.split()[-1] if line.strip() else ""
        if name.endswith(".keys.json"):
            p = subprocess.run(["aws", "--region", BUCKET_REGION, "s3", "cp", "--quiet", f"{q}/configs/{name}", "-"],
                               capture_output=True, text=True)
            want |= set(json.loads(p.stdout or "{}"))
    n = 0
    for k in sorted(want - set(man["keys"])):
        if not compile_farm.s3_exists(f"{cache}/{k}/{compile_farm.MARKER}"):
            continue
        p = subprocess.run(["aws", "--region", BUCKET_REGION, "s3", "ls", f"{cache}/{k}/{compile_farm.HLO}"],
                           capture_output=True, text=True)
        size = next((int(line.split()[2]) for line in p.stdout.splitlines() if line.strip().endswith(compile_farm.HLO)), 0)
        compile_farm.s3_claim(f"{q}/keys/{k}__{size}", "")
        n += 1
    return n


def cmd_reindex(a) -> None:
    for q in a.queue:
        q = q.rstrip("/")
        added = backfill(q) if a.backfill else 0
        print(json.dumps({"queue": q, "backfilled": added, "keys": reindex(q)}), flush=True)


def _known_peaks(q: str) -> dict[int, float]:
    out: dict[int, float] = {}
    p = subprocess.run(["aws", "--region", BUCKET_REGION, "s3", "cp", "--quiet", "--recursive", f"{q}/done/",
                        "/tmp/kiln-farm-done/"], capture_output=True, text=True)
    if p.returncode == 0 and os.path.isdir("/tmp/kiln-farm-done"):
        for fn in os.listdir("/tmp/kiln-farm-done"):
            try:
                with open(os.path.join("/tmp/kiln-farm-done", fn)) as f:
                    d = json.load(f)
                out[int(d["hlo_bytes"])] = max(out.get(int(d["hlo_bytes"]), 0.0), float(d["peak_tree_rss_gb"]))
            except (OSError, ValueError, KeyError):
                continue
    return out


def _oom_reservations(q: str) -> dict[str, float]:
    """Per KEY, the reservation a graph the OOM killer took needs next time ({q}/oom/<key>.json, written
    by work()). By key, not by size: one F0 mixed group of a size the G64 groups peak at 23-30 GB
    reached 107 GB before it was killed (kiln-cf-1, 2026-10-04), so a size match would inflate them all."""
    out: dict[str, float] = {}
    import hashlib

    d = f"/tmp/kiln-farm-oom-{hashlib.sha1(q.encode()).hexdigest()[:12]}"
    p = subprocess.run(["aws", "--region", BUCKET_REGION, "s3", "cp", "--quiet", "--recursive", f"{q}/oom/", d],
                       capture_output=True, text=True)
    if p.returncode == 0 and os.path.isdir(d):
        for fn in os.listdir(d):
            try:
                with open(os.path.join(d, fn)) as f:
                    out[fn.removesuffix(".json")] = float(json.load(f)["reserve_gb"])
            except (OSError, ValueError, KeyError):
                continue
    return out


def cmd_work(a) -> None:
    import socket

    import psutil

    import shutil

    if shutil.which("neuronx-cc") is None:  # before any claim: a worker that dies holding claims strands them
        raise SystemExit("neuronx-cc is not on PATH (activate the Neuron venv)")
    q = a.queue.rstrip("/")
    man = _manifest(q)
    t_wait = time.time()
    while "cache" not in man and time.time() - t_wait < a.linger:  # started before the first enqueue
        time.sleep(15)
        man = _manifest(q)
    if "cache" not in man:
        raise SystemExit(f"no manifest at {q}/manifest.json: nothing was enqueued")
    cache = man["cache"].rstrip("/")
    # LNL's default root, whatever NEURON_LIBTORCH_CACHE_ROOT says: a NEFF records its --output path
    # (info.json "name") and the runtime prints that path in load errors, so a farm NEFF built under
    # another root reads as a file the device never had (seen on kiln-g1-trn1, 2026-10-04).
    root = a.root or os.path.expanduser("~/.cache/neuron_libtorch/neuron/compile_cache")
    os.makedirs(root, exist_ok=True)
    compile_cache.pull_nki(cache)
    total_gb = psutil.virtual_memory().total / 2**30
    host_budget = 0.9 * total_gb  # shared by every worker on the host (compile_farm.HostLedger)
    budget = min(a.mem_budget_gb or host_budget, host_budget)
    ledger = compile_farm.HostLedger()
    host = f"{socket.gethostname()}:{os.getpid()}"
    # Peaks measured by every host on this queue so far (done/<key>.json), by graph.hlo size: the
    # size-only estimate is a weak proxy (a 0.57 MB prefill graph peaked at 23.6 GB, a 1.85 MB
    # decode graph at 12 GB, docs/neuron-notes.md "Compile farm").
    known = _known_peaks(q)
    oom_reserve = _oom_reservations(q)
    todo = sorted(man["keys"].items(), key=lambda kv: -kv[1])  # largest first: the long poles start early
    running: dict = {}
    results = []
    oom_retried: dict[str, bool] = {}
    kernel_lock = threading.Lock()
    too_big: set[str] = set()
    t0 = time.time()
    pool = ThreadPoolExecutor(a.workers)

    def done_or_taken(k: str) -> bool:
        return (compile_farm.s3_exists(f"{cache}/{k}/{compile_farm.MARKER}")
                or compile_farm.s3_exists(f"{q}/done/{k}.json"))

    def one(k: str):
        d = os.path.join(root, k)
        _s3("sync", "--quiet", f"{q}/entries/{k}/", d)
        if compile_farm.missing_kernels(os.path.join(d, compile_farm.HLO)):
            # A capture enqueued after this worker started compiled new NKI kernels: fetch again.
            with kernel_lock:
                compile_cache.pull_nki(cache)
        r = compile_farm.run_compile(d, work_root=a.work_root)
        if r.ok:
            compile_cache.push_entry(cache, d)
        body = json.dumps({**compile_farm.result_dict(r), "host": host, "hlo_bytes": man["keys"][k]})
        with open(f"/tmp/kiln-farm-{k}.json", "w") as f:
            f.write(body)
        _s3("cp", "--quiet", f"/tmp/kiln-farm-{k}.json", f"{q}/{'done' if r.ok else 'failed'}/{k}.json")
        return r

    seen = set(man["keys"])
    last_refresh = last_busy = time.time()
    try:
        with HostMonitor() as mon:
            while running or any(k not in too_big for k, _ in todo) or time.time() - last_busy < a.linger:
                if running or todo:
                    last_busy = time.time()
                if time.time() - last_refresh > (15 if not (running or todo) else 60):  # keys enqueued later
                    last_refresh = time.time()
                    fresh = _manifest(q)["keys"]
                    new = {k: v for k, v in fresh.items() if k not in seen}
                    if new:
                        seen |= set(new)
                        man["keys"].update(new)
                        todo = sorted([*todo, *new.items()], key=lambda kv: -kv[1])
                launched = False
                i = 0
                # Largest first; a graph that does not fit the memory left is skipped for now and a
                # smaller one behind it may start (backfill), so small graphs never queue behind a
                # big one that waits for memory.
                while i < len(todo) and len(running) < a.workers:
                    k, size = todo[i]
                    est = oom_reserve.get(k) or compile_farm.peak_gb_estimate(size, known)
                    used = sum(e for _, e in running.values())
                    free = psutil.virtual_memory().available / 2**30
                    if est > budget:  # never here: a host with more memory takes it
                        too_big.add(k)
                        i += 1
                        continue
                    mine = [x for x, _ in running.values()]
                    with ledger.locked():  # check and reserve as one step against the other workers
                        others, busy = ledger.others()
                        fits = k not in busy and compile_farm.admit(est, used, others, budget, host_budget, free)
                        if fits:
                            ledger.set(used + est, queue=q, keys=[*mine, k])
                    if not fits:  # a key another queue's worker compiles stays here: its NEFF lands in the cache
                        i += 1
                        continue
                    todo.pop(i)
                    if done_or_taken(k) or not compile_farm.s3_claim(
                            f"{q}/claims/{k}", json.dumps({"host": host, "time": time.time()})):
                        ledger.set(used, queue=q, keys=mine)
                        continue
                    print(json.dumps({"start": k, "hlo_bytes": size, "est_peak_gb": round(est, 1),
                                      "reserved_gb": round(used + est, 1),
                                      "host_reserved_gb": round(others + used + est, 1)}), flush=True)
                    running[pool.submit(one, k)] = (k, est)
                    launched = True
                finished = [f for f in running if f.done()]
                if finished:
                    left = [v for f, v in running.items() if f not in finished]
                    ledger.set(sum(e for _, e in left), queue=q, keys=[x for x, _ in left])
                for f in finished:
                    k, est = running.pop(f)
                    try:
                        r = f.result()
                    except Exception as e:  # noqa: BLE001 - never leave a claim nobody will finish
                        r = compile_farm.Result(k, False, 0.0, -1, error=f"farm worker: {type(e).__name__}: {e}")
                        b, key = compile_farm.split_uri(f"{q}/claims/{k}")
                        compile_farm.s3api("delete-object", "--bucket", b, "--key", key, check=False)
                    results.append(r)
                    if r.ok:
                        known[man["keys"][k]] = r.peak_tree_rss_gb
                    elif compile_farm.oom_killed(r) and not oom_retried.get(k):
                        # Killed by the OOM killer: the estimate was low. Release the claim and the
                        # failed record (a device's FarmWait stops waiting on a failed key) and retry it
                        # here once with twice the reservation (LNL backend.py reads -9 the same way).
                        oom_retried[k] = True
                        # The tree's RSS when it was killed is a floor on what it needs, and can be far
                        # above twice the estimate (107 GB against 27 for one F0 mixed group).
                        oom_reserve[k] = compile_farm.oom_reservation(est, r.peak_tree_rss_gb)
                        with open(f"/tmp/kiln-farm-oom-{k}.json", "w") as fo:
                            json.dump({"key": k, "reserve_gb": oom_reserve[k], "est_gb": est,
                                       "peak_tree_rss_gb_at_kill": r.peak_tree_rss_gb, "host": host}, fo)
                        _s3("cp", "--quiet", f"/tmp/kiln-farm-oom-{k}.json", f"{q}/oom/{k}.json")
                        for rec in (f"{q}/claims/{k}", f"{q}/failed/{k}.json"):
                            b, key = compile_farm.split_uri(rec)
                            compile_farm.s3api("delete-object", "--bucket", b, "--key", key, check=False)
                        todo.insert(0, (k, man["keys"][k]))
                    _print(r, {"host": host}, a.out)
                if not finished and not launched:
                    time.sleep(2)
    finally:
        ledger.close()
    pool.shutdown()
    bad = [r for r in results if not r.ok]
    print(json.dumps({"summary": True, "host": host, "compiled": len(results), "failed": len(bad),
                      "left_for_bigger_hosts": sorted(too_big),
                      "wall_seconds": round(time.time() - t0, 1), "mem_budget_gb": round(budget), **mon.summary()}),
          flush=True)
    if bad:
        sys.exit(1)


def cmd_status(a) -> None:
    q = a.queue.rstrip("/")
    man = _manifest(q)

    def ls(sub):
        p = subprocess.run(["aws", "--region", BUCKET_REGION, "s3", "ls", f"{q}/{sub}/"], capture_output=True, text=True)
        return {line.split()[-1].removesuffix(".json") for line in p.stdout.splitlines() if line.strip()}

    claims, done, failed = ls("claims"), ls("done"), ls("failed")
    print(json.dumps({"keys": len(man["keys"]), "claimed": len(claims), "done": len(done), "failed": len(failed),
                      "running": len(claims - done - failed), "unclaimed": len(set(man["keys"]) - claims)}))


def cmd_check(a) -> None:
    """Is every graph a capture produced complete in a cache prefix? Exit 1 if not."""
    with open(a.keys_file) as f:
        keys = json.load(f)
    world = max((max(r) for r in keys.values() if r), default=0) + 1
    partial = {k: r for k, r in keys.items() if len(r) != world}
    missing = [k for k in keys if not compile_farm.s3_exists(f"{a.cache.rstrip('/')}/{k}/{compile_farm.MARKER}")]
    print(json.dumps({"graphs": len(keys), "ranks": world, "keys_not_on_every_rank": partial,
                      "missing_in_cache": missing}))
    if missing:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sh = sub.add_parser("shapes")
    sh.add_argument("model")
    sh.add_argument("dest")
    dc = sub.add_parser("decisions")
    dc.add_argument("--model", required=True, help="Hugging Face repo id (range reads) or a local checkpoint dir")
    dc.add_argument("--shape-dir", required=True)
    dc.add_argument("--tp", type=int, required=True)
    dc.add_argument("--target", default="trn1", help="its fp8_max (kiln/platform.py) is what the loader refits to")
    dc.add_argument("--fp8-max", type=float, default=0.0, help="override the target's")
    dc.add_argument("--ranks", default="all")
    dc.add_argument("--layers", default=None, help="comma list of prefixes (default: every MoE layer)")
    dc.add_argument("--procs", type=int, default=16)
    dc.add_argument("--revision", default=None, help="hub revision (default: the repo's current sha, recorded)")
    cp = sub.add_parser("capture")
    cp.add_argument("--shape-dir", required=True)
    cp.add_argument("--ranks", default="all", help="all, or a comma list")
    cp.add_argument("--procs", type=int, default=32, help="rank processes at once after rank 0")
    cp.add_argument("--target", default="trn1")
    cp.add_argument("--out-dir", default="/opt/kiln/work/capture")
    cp.add_argument("--skip-prefill", action="store_true", help="decode graphs only")
    cp.add_argument("--tool", default="serve_sweep", choices=sorted(TOOLS),
                    help="whose arguments follow the --: the capture rebuilds that tool's EngineConfig")
    cp.add_argument("--plp", action="store_true",
                    help="also the prompt-logprobs prefill graphs (check_ppl runs prompt logprobs)")
    cp.add_argument("sweep", nargs=argparse.REMAINDER, help="-- then the tool's arguments (bench/serve_sweep.py by default)")
    f = sub.add_parser("fetch")
    f.add_argument("src")
    f.add_argument("--keys", default=None)
    f.add_argument("--root", default=None)
    c = sub.add_parser("compile")
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--keys", default=None)
    c.add_argument("--root", default=None)
    c.add_argument("--push", default=None)
    c.add_argument("--out", default=None)
    c.add_argument("--work-root", default="/opt/kiln/work")
    b = sub.add_parser("bench")
    b.add_argument("--keys", required=True)
    b.add_argument("--repeat", type=int, default=1)
    b.add_argument("--workers", type=int, default=0, help="concurrent compiles (default: all jobs at once)")
    b.add_argument("--extra", default="")
    b.add_argument("--tag", default="")
    b.add_argument("--root", default=None)
    b.add_argument("--scratch", default="/opt/kiln/work/bench")
    b.add_argument("--work-root", default="/opt/kiln/work")
    b.add_argument("--out", default=None)
    b.add_argument("--keep", action="store_true", help="keep the NEFFs")
    e = sub.add_parser("enqueue")
    e.add_argument("--queue", required=True)
    e.add_argument("--cache", required=True, help="s3:// compile-cache prefix the device boxes pull")
    e.add_argument("--keys-file", default=None, help="capture's keys.json (default: every local entry without a NEFF)")
    e.add_argument("--root", default=None)
    w = sub.add_parser("work")
    w.add_argument("--queue", required=True)
    w.add_argument("--workers", type=int, default=64, help="compiles at once, at most")
    w.add_argument("--mem-budget-gb", type=float, default=0.0, help="default 90%% of host memory")
    w.add_argument("--root", default=None)
    w.add_argument("--out", default=None)
    w.add_argument("--work-root", default="/opt/kiln/work")
    w.add_argument("--linger", type=float, default=0.0,
                   help="seconds to keep polling the manifest for new keys once idle (start workers before enqueue)")
    ri = sub.add_parser("reindex")
    ri.add_argument("queue", nargs="+")
    ri.add_argument("--backfill", action="store_true",
                    help="first record the configs' keys that are complete in the cache but not listed")
    ck = sub.add_parser("check")
    ck.add_argument("--keys-file", required=True)
    ck.add_argument("--cache", required=True)
    st = sub.add_parser("status")
    st.add_argument("--queue", required=True)
    a = ap.parse_args()
    if getattr(a, "sweep", None) and a.sweep[:1] == ["--"]:
        a.sweep = a.sweep[1:]
    {"fetch": cmd_fetch, "compile": cmd_compile, "bench": cmd_bench, "shapes": cmd_shapes, "decisions": cmd_decisions,
     "capture": cmd_capture, "enqueue": cmd_enqueue, "work": cmd_work, "status": cmd_status, "check": cmd_check,
     "reindex": cmd_reindex}[a.cmd](a)


if __name__ == "__main__":
    main()
