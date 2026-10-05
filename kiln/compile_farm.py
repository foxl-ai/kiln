"""Compile captured graphs (graph.hlo) into LNL compile-cache entries on CPU hosts.

A compile-cache entry is `<cache>/<hash>/` (kiln/compile_cache.py). Everything a compile needs is
already in it once the graph has been lowered: graph.hlo, the artifact metadata LNL writes beside
it (.artifact_metadata_v0.json: io_map, output count, unused inputs) and, for an entry LNL compiled
itself, command.txt with the exact neuronx-cc command line. neuronx-cc is a plain CLI on an HLO
file and runs on any host with the Neuron compiler installed (every Neuron DLAMI, also on CPU-only
instance types), so the NEFF can be built anywhere and the entry shipped through S3: LNL treats a
directory as a cache hit once `.compilation_complete` is present (libtorch_neuronx_lite compile/
cache.py, _is_cache_complete) and never recompiles it.

This module is the per-entry half of tools/compile_farm.py: it reads the compiler arguments of an
entry, runs neuronx-cc the way LNL does (compile/backend.py neuroncc_compile: `neuronx-cc compile
<graph.hlo> --framework XLA --target <t> --output <dir>/graph_<hash>.neff --logfile <dir>/
log-neuron-cc.txt <compiler_args>`), measures it, and writes the completion marker LAST in LNL's
format (compile/parallel_compile.py _do_compile: "completed:<time>\\nneff_size:<bytes>\\n").

Graphs that call NKI kernels reference the kernel binaries by absolute path inside the HLO (the
custom call's base64 backend_config, field klir_binary.binary, under
/var/tmp/nki-intermediate-cache); `kernel_refs` lists them so a farm host can check it holds them
before compiling (neuronx-cc otherwise fails NCC_EVRF059, docs/neuron-notes.md).
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field

MARKER = ".compilation_complete"  # libtorch_neuronx_lite compile/cache.py NEURON_COMPILE_COMPLETE_FILE
HLO = "graph.hlo"  # NEURON_COMPILE_GRAPH_HLO_FILE
COMMAND = "command.txt"  # written by compile/backend.py neuroncc_compile
LOG = "log-neuron-cc.txt"
METADATA = ".artifact_metadata_v0.json"  # compile/cache.py _get_artifact_metadata_filename(0)
ARGS = "compiler_args.json"  # Kiln's: the compiler_args a captured entry was hashed with


def neff_name(key: str) -> str:
    """compile/cache.py get_neff_filename: graph_<hash>.neff."""
    return f"graph_{key}.neff"


# Arguments neuroncc_compile always sets itself; everything else on the command line is the
# entry's compiler_args (what LNL's cache key hashed, after _apply_platform_compiler_args).
_FIXED = {"--framework", "--target", "--output", "--logfile"}


def parse_command(text: str) -> tuple[str, list[str]]:
    """(target, compiler_args) from an LNL command.txt."""
    argv = shlex.split(text)
    if len(argv) < 3 or argv[1] != "compile":
        raise ValueError(f"not a neuronx-cc compile command: {text[:120]!r}")
    target, rest, i = None, [], 3  # argv[2] is the HLO path
    while i < len(argv):
        a = argv[i]
        name = a.split("=", 1)[0]
        if name in _FIXED:
            value = a.split("=", 1)[1] if "=" in a else argv[i + 1]
            if name == "--target":
                target = value
            i += 1 if "=" in a else 2
            continue
        rest.append(a)
        i += 1
    if target is None:
        raise ValueError("command has no --target")
    return target, rest


def entry_args(entry: str) -> tuple[str, list[str]]:
    """The target and compiler_args to compile `entry` with: command.txt when LNL wrote one,
    else compiler_args.json from Kiln's capture (kiln/capture.py)."""
    p = os.path.join(entry, COMMAND)
    if os.path.exists(p):
        with open(p) as f:
            return parse_command(f.read())
    with open(os.path.join(entry, ARGS)) as f:
        d = json.load(f)
    return d["target"], list(d["compiler_args"])


# A base64 run long enough to be a backend_config (they are kilobytes); decoded and parsed lazily.
_B64 = re.compile(rb"[A-Za-z0-9+/]{64,}={0,2}")


def kernel_configs(hlo_path: str) -> list[dict]:
    """The decoded backend_config of every NKI custom call in an HLO (func_name, klir_binary, ...)."""
    with open(hlo_path, "rb") as f:
        data = f.read()
    out = []
    for m in _B64.finditer(data):
        blob = m.group(0)
        if not blob.startswith(b"eyJ"):  # base64 of '{"'
            continue
        try:
            cfg = json.loads(base64.b64decode(blob + b"=" * (-len(blob) % 4)))
        except (binascii.Error, ValueError):
            continue
        if isinstance(cfg, dict):
            out.append(cfg)
    return out


def kernel_refs(hlo_path: str) -> set[str]:
    """Absolute kernel-binary paths an HLO's NKI custom calls point at (klir_binary.binary)."""
    refs = set()
    for cfg in kernel_configs(hlo_path):
        klir = cfg.get("klir_binary")
        if isinstance(klir, dict) and isinstance(klir.get("binary"), str):
            refs.add(klir["binary"])
    return refs


def missing_kernels(hlo_path: str) -> list[str]:
    return sorted(p for p in kernel_refs(hlo_path) if not os.path.exists(p))


def compiler_command(neuronx_cc: str, hlo: str, target: str, neff: str, log: str, args: list[str]) -> list[str]:
    """The command neuroncc_compile runs (compile/backend.py, SDK 2.32)."""
    return [neuronx_cc, "compile", hlo, "--framework", "XLA", "--target", target, "--output", neff,
            "--logfile", log, *args]


@dataclass
class Result:
    key: str
    ok: bool
    seconds: float
    returncode: int
    cpu_seconds: float = 0.0  # user + system of neuronx-cc and every descendant it waited for
    max_rss_gb: float = 0.0  # largest single process (wait4 ru_maxrss)
    peak_tree_rss_gb: float = 0.0  # largest sum over the live process tree, sampled every second
    neff_bytes: int = 0
    args: list[str] = field(default_factory=list)
    error: str = ""


def _tree_rss(pid: int) -> int:
    import psutil

    try:
        p = psutil.Process(pid)
        procs = [p, *p.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for q in procs:
        try:
            total += q.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def run_compile(entry: str, *, out_dir: str | None = None, extra: list[str] = (), work_root: str | None = None,
                neuronx_cc: str | None = None, write_marker: bool = True, sample_seconds: float = 1.0) -> Result:
    """Compile `entry`'s graph.hlo with its own compiler arguments plus `extra`.

    out_dir None: the NEFF, log and command go into the entry itself and the completion marker is
    written last, making it an LNL cache hit. An experiment with different flags must pass out_dir
    (the cache key hashes the flags, so a NEFF built with other flags must never sit under this key).
    """
    key = os.path.basename(entry.rstrip("/"))
    target, args = entry_args(entry)
    args = [*args, *extra]
    if extra and out_dir is None:
        raise ValueError("extra compiler flags change the cache key: compile them into out_dir")
    dest = out_dir or entry
    os.makedirs(dest, exist_ok=True)
    hlo = os.path.join(entry, HLO)
    neff, log = os.path.join(dest, neff_name(key)), os.path.join(dest, LOG)
    cc = neuronx_cc or shutil.which("neuronx-cc")
    if cc is None:
        raise RuntimeError("neuronx-cc is not on PATH")
    cmd = compiler_command(cc, hlo, target, neff, log, args)
    gone = missing_kernels(hlo)
    if gone:
        return Result(key, False, 0.0, -1, args=args, error=f"{len(gone)} NKI kernel binaries missing, e.g. {gone[0]}")
    with open(os.path.join(dest, COMMAND), "w") as f:
        f.write(shlex.join(cmd))
    # neuronx-cc leaves its scratch tree (neuronxcc-<random>) in the cwd.
    cwd = tempfile.mkdtemp(prefix=f"ncc-{key[:8]}-", dir=work_root)
    peak = [0]
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    done = threading.Event()

    def sample():
        while not done.wait(sample_seconds):
            peak[0] = max(peak[0], _tree_rss(proc.pid))

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    _, status, ru = os.wait4(proc.pid, 0)
    seconds = time.time() - t0
    done.set()
    th.join()
    proc.returncode = os.waitstatus_to_exitcode(status)
    err = proc.stderr.read().decode(errors="replace")[-2000:] if proc.stderr else ""
    res = Result(key, proc.returncode == 0 and os.path.exists(neff), seconds, proc.returncode,
                 cpu_seconds=ru.ru_utime + ru.ru_stime, max_rss_gb=ru.ru_maxrss / 2**20,
                 peak_tree_rss_gb=peak[0] / 2**30, args=args, error="" if proc.returncode == 0 else err)
    if res.ok:
        res.neff_bytes = os.path.getsize(neff)
        shutil.rmtree(cwd, ignore_errors=True)
        if write_marker and out_dir is None:
            # Last: a reader that sees the marker sees a complete entry.
            with open(os.path.join(dest, MARKER), "w") as f:
                f.write(f"completed:{time.time()}\nneff_size:{res.neff_bytes}\n")
    return res


def oom_killed(r: Result) -> bool:
    """The OOM killer took neuronx-cc itself (-9, or 137 through a shell) or one of its subprocesses,
    which neuronx-cc reports as exit 70 and "[F137] neuronx-cc was forcibly killed - This most
    commonly occurs due to insufficient system memory" (measured: kiln-cf-1, 2026-10-04, a 4.1 MB
    trn2 prefill group at 35.7 GB while two workers shared the host)."""
    return r.returncode in (-9, 137) or (r.returncode == 70 and "[F137]" in r.error)


# -- device memory a set of NEFFs takes ----------------------------------------------------------
#
# The Neuron runtime's own accounting, from the table it prints on an allocation failure (kiln-g1-trn1,
# 2026-10-04 02:47 UTC, /tmp/neuron_mem_table_device_12_nc_1.log, its MB / GB being MiB / GiB). Every
# term was matched against the 10 NEFFs it lists, read from the trn1 compile cache:
# - Model Code: the instruction streams (sg<N>/<Engine><N>.bin), 59.87 MiB printed for 62.66 MB of
#   streams in NEFF 1002, 182.81 MiB for 190.998 MB in 1010.
# - per-NEFF Scratchpad reservation: the compiler's "Peak scratchpad usage: local" line (GiB), exactly:
#   0.276024 GiB = 282.648 MiB (1002), 0.251030 = 257.055 (1004), 0.361675 = 370.354 (1008). The
#   shared scratchpad is the largest over the loaded NEFFs, rounded up to 64 MiB (370.479 -> 384).
#   The DRAM_Allocator "spill space" line is a different quantity (0.8-2.1 GB on the same NEFFs).
# - DMA Rings Spill: about 32 B per run of the Spill DMA queues (310.7 MiB summed against 333.3 printed).
# - IO and collectives rings, runtime, profiler, constants: 0.13 GiB together.
# Tensors + those = 15.91 GiB, the table's total, at which the load failed (16 GiB per NeuronCore).
SCRATCHPAD_PAGE = 64 * 2**20
RUNTIME_FIXED_GIB = 0.13
_SCRATCH = re.compile(r"Peak scratchpad usage: local\s*\S\s*([\d.]+) GB")


def scratchpad_gib(log_text: str) -> float:
    """A NEFF's scratchpad reservation (GiB) from its compiler log, 0.0 if the log has no summary."""
    vals = [float(x) for x in _SCRATCH.findall(log_text)]
    return max(vals) if vals else 0.0


def shared_scratchpad_bytes(peaks_gib) -> int:
    """The runtime's shared scratchpad for NEFFs with these reservations: the largest, page-rounded."""
    top = max(peaks_gib, default=0.0) * 2**30
    return -(-int(round(top)) // SCRATCHPAD_PAGE) * SCRATCHPAD_PAGE


def result_dict(r: Result) -> dict:
    return asdict(r)


# -- the S3 work queue ------------------------------------------------------------------------
#
#   <queue>/manifest.json          {"cache": <cache uri>, "keys": {key: graph.hlo bytes}}
#   <queue>/entries/<key>/...      what a compile needs: graph.hlo, the metadata, command.txt
#   <queue>/claims/<key>           created with If-None-Match: * by the host that compiles it
#   <queue>/done/<key>.json        the Result; the NEFF itself goes to <cache>/<key>/
#
# A claim is an S3 conditional write: PutObject with If-None-Match "*" succeeds for exactly one
# writer and answers 412 PreconditionFailed to the others (S3 User Guide, "Conditional writes":
# https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html). So any number of
# hosts can run `tools/compile_farm.py work` on one queue and each graph compiles once.

ENTRY_FILES = (HLO, METADATA, COMMAND, "fxgraph.txt", "example_inputs.txt")


def peak_gb_estimate(hlo_bytes: int, known: dict[int, float] | None = None) -> float:
    """Host memory one neuronx-cc compile is expected to peak at, for admission control.

    `known` maps graph.hlo sizes already compiled on this queue to their measured peaks (work()
    seeds it from the queue's done records); the nearest one within 10% wins, with 15% headroom.
    Otherwise a size table, measured (process-tree RSS, neuronx-cc 2.27.5334) over the 80 compiles of
    the e902f99 / d25e855 / efa1da4 queues (GLM-5.3-Flash tp=32, `--model-type=transformer`, NKI MoE
    kernels, elementwise mHC, 2026-10-04), the largest peak per HLO size band with ~1.3x headroom:
    under 1 MB 10.6 GB, 1-2 MB 14.4, 2-3 MB 22.9, 3-5 MB 17.0, and the 6.3 MB P=12 prefill groups at
    512 rows 30.0. The table this replaces (200 GB at 2 MB and over) came from the earlier bmm-form
    graphs at default flags, whose 12-layer prefill groups at 2048 rows (2.93-3.10 MB) peaked at
    174-175 GB; with it a host compiled the new P=12 groups one at a time. A graph that still exceeds
    its estimate is killed by the OOM killer, and work() retries it once with twice the reservation
    after checking the host's free memory before every launch."""
    if known:
        size, peak = min(known.items(), key=lambda kv: abs(kv[0] - hlo_bytes))
        if abs(size - hlo_bytes) <= 0.1 * hlo_bytes:
            return peak * 1.15
    mb = hlo_bytes / 1e6
    if mb < 1:
        return 14.0
    if mb < 2:
        return 19.0
    if mb < 3.5:
        return 30.0
    return max(40.0, 6.5 * mb)


def oom_reservation(est_gb: float, peak_at_kill_gb: float) -> float:
    """What to reserve for a graph the OOM killer took: twice the estimate it ran under, or 1.5x the
    process tree's RSS when it was killed (a floor on its peak), whichever is larger. Measured on
    kiln-cf-1, 2026-10-04: an F0 mixed group estimated at 27 GB was killed at 107.3 GB."""
    return max(2.0 * est_gb, 1.5 * peak_at_kill_gb)


LEDGER_DIR = "/dev/shm/kiln-farm-ledger"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class HostLedger:
    """Memory reserved by every compile-farm worker on this host, one file per worker pid.

    Each worker budgets the whole host by default, so two workers on one host (one per queue) each
    admitted compiles up to 90% of its memory. Measured on kiln-cf-1 (371 GiB), 2026-10-04: the
    v1-trn1, mx-trn1 and pfab-trn1 workers held 330 + 294 + 133 GB of reservations at once and the
    OOM killer took 11 compiles between 16:16 and 16:22 UTC. The free-memory check at launch could
    not see it, because a compile reaches its peak minutes after it starts. Admission now counts the
    reservations of the other live workers too, under one lock (`admit` holds the rule)."""

    def __init__(self, path: str = LEDGER_DIR, pid: int | None = None):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.pid = pid or os.getpid()
        self.mine = os.path.join(path, f"{self.pid}.json")

    def locked(self):
        import contextlib
        import fcntl

        @contextlib.contextmanager
        def held():
            fd = os.open(os.path.join(self.path, "lock"), os.O_CREAT | os.O_RDWR, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

        return held()

    def others_gb(self) -> float:
        return self.others()[0]

    def others(self) -> tuple[float, set[str]]:
        """(GB reserved, keys being compiled) by the other live workers. Two queues can hold the same
        key (t2max-U1 and t2max-U0 share 14 decode graphs), and two compiles of one key on one host
        write the same LNL entry directory, so a worker leaves a key another one is compiling."""
        total, keys = 0.0, set()
        for fn in os.listdir(self.path):
            if not fn.endswith(".json") or not fn[:-5].isdigit() or int(fn[:-5]) == self.pid:
                continue
            p = os.path.join(self.path, fn)
            if not _pid_alive(int(fn[:-5])):  # a worker that died without closing
                try:
                    os.remove(p)
                except OSError:
                    pass
                continue
            try:
                with open(p) as f:
                    d = json.load(f)
                total += float(d["reserved_gb"])
                keys |= set(d.get("keys") or [])
            except (OSError, ValueError, KeyError):
                continue
        return total, keys

    def set(self, reserved_gb: float, **info) -> None:
        tmp = f"{self.mine}.tmp"
        with open(tmp, "w") as f:
            json.dump({"reserved_gb": reserved_gb, **info}, f)
        os.replace(tmp, self.mine)

    def close(self) -> None:
        try:
            os.remove(self.mine)
        except OSError:
            pass


def admit(est: float, used: float, others: float, budget: float, host_budget: float, free: float) -> bool:
    """May a compile expected to peak at `est` GB start? `used` is this worker's reservations,
    `others` the other workers' on the host (HostLedger), `budget` this worker's cap, `host_budget`
    the host's, `free` the memory available now (captures and other processes outside the ledger)."""
    return used + est <= budget and others + used + est <= host_budget and est <= free


def s3api(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["aws", "--region", "us-east-2", "s3api", *args], capture_output=True, text=True,
                          check=check)


def split_uri(uri: str) -> tuple[str, str]:
    assert uri.startswith("s3://"), uri
    bucket, _, key = uri[5:].partition("/")
    return bucket, key


def s3_exists(uri: str) -> bool:
    b, k = split_uri(uri)
    return s3api("head-object", "--bucket", b, "--key", k, check=False).returncode == 0


def s3_claim(uri: str, body: str) -> bool:
    """True if this call created `uri` (the claim is ours), False if it already existed."""
    b, k = split_uri(uri)
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(body)
    try:
        p = s3api("put-object", "--bucket", b, "--key", k, "--body", f.name, "--if-none-match", "*", check=False)
    finally:
        os.unlink(f.name)
    if p.returncode == 0:
        return True
    if "PreconditionFailed" in p.stderr or "(412)" in p.stderr or "ConditionalRequestConflict" in p.stderr:
        return False
    raise RuntimeError(f"claim {uri} failed: {p.stderr.strip()}")
