"""Capture every graph, then compile them in parallel, then start: vllm-neuron's engine-start order, on one host.

vllm-neuron 0.24 starts an engine in three phases (vllm_neuron/vllm/worker/neuron_worker.py:1207-1240 at
release-0.24.0.1.1.0): it extracts the HLO of every bucket's graph first, then compiles all of them at once
(`model_runner.parallel_compile`, neuron_model_runner.py:5094-5120, LNL's compile.parallel_compile), and only then
warms up, so every warmup graph is a cache hit. A Kiln engine instead compiles during its warmup, one graph at a
time behind LNL's compile lock. Measured on trn2.3xlarge (2026-10-07, bench/results/2026-10-07-trn2.3xlarge-qwen3-ttft.md):
Qwen3-8B TP=4 started in 519.2 s, 497 s of it twelve serial compiles of 10.7-141.6 s each.

With EngineConfig.precompile_workers > 0 (KILN_PRECOMPILE_WORKERS), LLMEngine starts this module's `run` in its own
process before anything else:

- capture: one process per tensor-parallel rank builds the shard on the meta device and runs the warmup the engine
  is about to run through kiln/capture.py's backend. That writes graph.hlo + metadata + command.txt under each
  graph's LNL cache key, with no NeuronCore and no compile. The EngineConfig is the engine's own, so the keys are
  the ones the device run computes (the compile farm's ASSERT_CACHE_HIT runs rest on the same identity).
- compile: every captured key that is not yet complete in the local cache is compiled with `precompile_workers`
  neuronx-cc processes at once, longest HLO first (kiln/compile_farm.py run_compile, completion marker last).

Meanwhile the engine spawns its ranks and loads the weights. It waits for this process at the end of __init__,
before any graph runs, so capture and compile overlap the weight load, which vllm-neuron does not do.

Keys a capture first meets while the shard is BUILT (before warmup) are left to the device run: it may execute
them before the precompile finishes, and two writers must never fill one cache entry. A failure raises in the
engine. Nothing falls back silently.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor


def _capture_rank(rank: int, cfg, shape_dir: str, target: str, num_pages: int, out: str) -> None:
    """One rank's capture in its own spawned process (dynamo and the process group are process-global); the same
    sequence as tools/compile_farm.py _capture_rank, from the engine's EngineConfig instead of a tool's argv."""
    from . import capture

    capture.configure_env(target)
    import torch

    torch.set_num_threads(2)
    # The process group before libtorch_neuronx_lite, the order a device rank has (kiln/capture.py init_fake_world).
    if cfg.tp > 1:
        capture.init_fake_world(rank, cfg.tp)
    from . import platform as kplatform

    kplatform.configure_runtime_env()
    import libtorch_neuronx_lite  # noqa: F401

    from .engine.engine import build_shard

    capture.install_header_checkpoint()
    t0 = time.time()
    _, runner = build_shard(dataclasses.replace(cfg, device="meta", precompile_workers=0), shape_dir, num_pages, rank)
    built = list(capture.CAPTURED)  # first met while building: the device run compiles these itself
    t1 = time.time()
    role = cfg.pd_role
    w = runner.warmup(1 + cfg.spec_k if cfg.spec_method else None, decode=role != "prefill",
                      prefill=role != "decode" or cfg.pd_bypass_prefill)
    warm = [k for k in capture.CAPTURED if k not in set(built)]
    rec = {"rank": rank, "build_seconds": round(t1 - t0, 1), "warmup_seconds": round(w["seconds"], 1),
           "graphs": w["graphs"], "keys": warm, "build_keys": built}
    with open(out, "w") as f:
        json.dump(rec, f)


def capture_all(cfg, model_path: str, target: str, num_pages: int, work: str, procs: int | None = None) -> dict:
    """Capture every rank's graphs. Rank 0 runs alone first: it compiles the NKI kernels and writes every graph shared
    by all ranks, so the other ranks mostly find their entries already there."""
    import multiprocessing as mp

    from . import capture

    shape_dir = capture.write_shape_dir(model_path, os.path.join(work, "shapes"))
    ctx = mp.get_context("spawn")
    procs = procs or max(1, min(cfg.tp, (os.cpu_count() or 2) // 2))

    def batch(ranks):
        ps = [(r, ctx.Process(target=_capture_rank,
                              args=(r, cfg, shape_dir, target, num_pages, os.path.join(work, f"rank{r}.json"))))
              for r in ranks]
        for _, p in ps:
            p.start()
        bad = []
        for r, p in ps:
            p.join()
            if p.exitcode != 0:
                bad.append((r, p.exitcode))
        return bad

    t0 = time.time()
    ranks = list(range(cfg.tp))
    bad = batch(ranks[:1])
    for i in range(1, len(ranks), procs):
        bad += batch(ranks[i : i + procs])
    if bad:
        raise RuntimeError(f"precompile: capture failed on ranks {bad} (logs above)")
    keys: dict[str, list[int]] = {}
    build: set[str] = set()
    for r in ranks:
        with open(os.path.join(work, f"rank{r}.json")) as f:
            rec = json.load(f)
        build.update(rec["build_keys"])
        for k in rec["keys"]:
            keys.setdefault(k, []).append(r)
    for k in build:
        keys.pop(k, None)
    return {"keys": keys, "build_keys": sorted(build), "seconds": round(time.time() - t0, 1)}


def compile_missing(keys, workers: int, root: str | None = None) -> dict:
    """Compile every key of `keys` that is captured (graph.hlo present) and not yet complete, `workers` at once,
    longest HLO first so the tail of the batch is short graphs."""
    from . import compile_cache, compile_farm

    root = root or compile_cache.local_root()
    todo = []
    for k in keys:
        d = os.path.join(root, k)
        if (os.path.exists(os.path.join(d, compile_farm.HLO))
                and not os.path.exists(os.path.join(d, compile_farm.MARKER))):
            todo.append(d)
    todo.sort(key=lambda d: -os.path.getsize(os.path.join(d, compile_farm.HLO)))
    t0 = time.time()
    with ThreadPoolExecutor(max(1, workers)) as pool:
        results = list(pool.map(compile_farm.run_compile, todo))
    bad = [r for r in results if not r.ok]
    if bad:
        raise RuntimeError("precompile: neuronx-cc failed on " + ", ".join(f"{r.key} (rc {r.returncode}): "
                                                                            f"{r.error[-300:]}" for r in bad))
    return {"compiled": len(results), "already": len(keys) - len(todo), "workers": workers,
            "seconds": round(time.time() - t0, 1),
            "per_graph": {r.key: round(r.seconds, 1) for r in results},
            "cpu_seconds": round(sum(r.cpu_seconds for r in results), 1)}


# Graph-relevant environment: Kiln's own switches and the Neuron runtime / compiler ones (the cache key hashes the
# compiler arguments, and several KILN_* knobs pick kernels or graph shapes).
_ENV_PREFIXES = ("KILN_", "NEURON_")
_ENV_IGNORED = ("KILN_PRECOMPILE_WORKERS", "NEURON_RT_VISIBLE_CORES", "NEURON_LIBTORCH_ASSERT_CACHE_HIT")


def fingerprint(cfg, model_path: str, target: str) -> str:
    """What decides the set of graph keys a warmup runs, hashed: the EngineConfig, the checkpoint's config and tensor
    headers, the graph-relevant environment, Kiln's own source, the LNL and compiler versions, the target. A warm
    start whose fingerprint has a manifest with every key complete skips the capture (Precompile). A stale manifest
    cannot give a wrong graph: a key it does not list is simply compiled by the device run, as without precompile."""
    import glob
    import hashlib
    from importlib import metadata

    from . import capture

    h = hashlib.sha256()
    h.update(json.dumps(dataclasses.asdict(cfg), default=str, sort_keys=True).encode())
    with open(os.path.join(model_path, "config.json"), "rb") as f:
        h.update(f.read())
    h.update(json.dumps(capture.header_index(model_path), sort_keys=True).encode())
    h.update(json.dumps(sorted((k, v) for k, v in os.environ.items()
                               if k.startswith(_ENV_PREFIXES) and k not in _ENV_IGNORED)).encode())
    root = os.path.dirname(os.path.abspath(__file__))
    for fn in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        with open(fn, "rb") as f:
            h.update(fn[len(root):].encode() + f.read())
    for pkg in ("libtorch-neuronx-lite", "neuronx-cc", "nki", "torch"):
        try:
            h.update(f"{pkg}={metadata.version(pkg)}".encode())
        except metadata.PackageNotFoundError:
            h.update(f"{pkg}=none".encode())
    h.update(target.encode())
    return h.hexdigest()[:32]


def manifest_path(fp: str, root: str | None = None) -> str:
    from . import compile_cache

    return os.path.join(root or compile_cache.local_root(), f".kiln-precompile-{fp}.json")


def manifest_complete(fp: str, root: str | None = None) -> list[str] | None:
    """The manifest's keys when every one of them is a complete cache entry, else None."""
    from . import compile_cache, compile_farm

    p = manifest_path(fp, root)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        keys = json.load(f)["keys"]
    base = root or compile_cache.local_root()
    if all(os.path.exists(os.path.join(base, k, compile_farm.MARKER)) for k in keys):
        return keys
    return None


def run(cfg, model_path: str, target: str, num_pages: int, result_path: str) -> None:
    """The precompile process: capture, then compile; the record (or the error) goes to result_path."""
    work = tempfile.mkdtemp(prefix="kiln-precompile-")
    rec: dict = {"ok": False, "workers": cfg.precompile_workers}
    t0 = time.time()
    try:
        cap = capture_all(cfg, model_path, target, num_pages, work)
        rec["capture_seconds"], rec["graphs"], rec["build_keys"] = cap["seconds"], len(cap["keys"]), cap["build_keys"]
        comp = compile_missing(cap["keys"], cfg.precompile_workers)
        rec.update({"compile": comp, "ok": True})
        fp = fingerprint(cfg, model_path, target)
        with open(manifest_path(fp) + ".tmp", "w") as f:
            json.dump({"keys": sorted(cap["keys"]), "written": time.time()}, f)
        os.replace(manifest_path(fp) + ".tmp", manifest_path(fp))
    except BaseException:
        rec["error"] = traceback.format_exc()[-4000:]
    rec["seconds"] = round(time.time() - t0, 1)
    with open(result_path + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(result_path + ".tmp", result_path)
    shutil.rmtree(work, ignore_errors=True)


class Precompile:
    """The engine's handle on a running precompile: start() before the shard is built, wait() before any graph runs."""

    def __init__(self, cfg, model_path: str, num_pages: int):
        from . import platform as kplatform

        target = kplatform.target()
        if target is None:
            raise RuntimeError("precompile: no Neuron target on this host (kiln/platform.py target)")
        self.t0 = time.time()
        self.proc = None
        self.fingerprint = fingerprint(cfg, model_path, target)
        keys = manifest_complete(self.fingerprint)
        if keys is not None:  # a warm start: every graph this configuration runs is already complete
            self.skipped = {"ok": True, "skipped": "manifest", "graphs": len(keys), "fingerprint": self.fingerprint,
                            "seconds": round(time.time() - self.t0, 2)}
            return
        self.skipped = None
        self.result_path = os.path.join(tempfile.mkdtemp(prefix="kiln-precompile-rec-"), "result.json")
        import multiprocessing as mp

        self.proc = mp.get_context("spawn").Process(target=run, args=(cfg, model_path, target, num_pages,
                                                                       self.result_path), daemon=False)
        self.proc.start()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def terminate(self) -> None:
        if self.alive():
            self.proc.terminate()

    def wait(self) -> dict:
        if self.skipped is not None:
            rec = dict(self.skipped, wall_seconds=round(time.time() - self.t0, 1))
            print(f"kiln precompile: skipped, the manifest lists {rec['graphs']} graphs, all cached "
                  f"(fingerprint {self.fingerprint}, checked in {rec['seconds']} s)", flush=True)
            return rec
        self.proc.join()
        if not os.path.exists(self.result_path):
            raise RuntimeError(f"precompile: process exited with code {self.proc.exitcode} and no record")
        with open(self.result_path) as f:
            rec = json.load(f)
        rec["wall_seconds"] = round(time.time() - self.t0, 1)
        if not rec.get("ok"):
            raise RuntimeError("precompile failed:\n" + rec.get("error", "(no error text)"))
        comp = rec["compile"]
        print(f"kiln precompile: {rec['graphs']} graphs captured in {rec['capture_seconds']} s, {comp['compiled']} "
              f"compiled with {comp['workers']} workers in {comp['seconds']} s ({comp['already']} already cached); "
              f"{rec['wall_seconds']} s wall from engine start", flush=True)
        return rec
