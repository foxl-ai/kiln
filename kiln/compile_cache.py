"""Share compiled graphs (NEFFs) across instances through S3.

Neuron compiles dominate start-up (minutes per bucket), and spot instances come and go, so
a graph compiled once should never be compiled again anywhere. libtorch_neuronx_lite keeps
one directory per graph hash under its local cache and treats a directory as usable only
once `.compilation_complete` is present (compile/cache.py, `_is_cache_complete`). Its own
remote cache expects a shared filesystem; this does the same over S3:

- pull(): download every remote entry missing locally (before any compile);
- push(): upload every complete local entry missing remotely, the marker file LAST, so a
  concurrent pull never sees a directory that looks complete but is not.

The hash already includes the compiler version and flags, so one prefix per SDK is enough.

Graphs that call NKI kernels need two more directories, or a cache hit on another host fails:
LNL's NKI compile results (`<cache>/nki/<key>.json`, libtorch_neuronx_lite/nki/nki_cache.py
_NKI_CACHE_SUBDIR; no completion marker, so the per-entry loop above skips it) and the kernel
binaries they point at, which the NKI compiler writes OUTSIDE that cache, under
/var/tmp/nki-intermediate-cache. Measured 2026-10-03 on a fresh trn1.32xlarge that had pulled
only the graph entries: neuronx-cc failed with NCC_EVRF059 "Kernel file
'/var/tmp/nki-intermediate-cache/.../kiln.kernels.moe_dedupe...colz' referenced by
AwsNeuronCustomNativeKernel instruction does not exist on the host", exit 70, and the dying lock
holder took every rank of the 32-rank engine down with it. Both are synced as plain trees.
"""

from __future__ import annotations

import os
import subprocess

MARKER = ".compilation_complete"
NKI_SUBDIR = "nki"  # libtorch_neuronx_lite/nki/nki_cache.py _NKI_CACHE_SUBDIR (SDK 2.32)
# Where the NKI compiler leaves kernel binaries (path from the NCC_EVRF059 message, SDK 2.32).
NKI_KERNEL_DIR = os.environ.get("KILN_NKI_KERNEL_DIR", "/var/tmp/nki-intermediate-cache")


def local_root() -> str:
    # libtorch_neuronx_lite/envs.py: <NEURON_LIBTORCH_CACHE_ROOT>/neuron/compile_cache, with
    # ~/.cache/neuron_libtorch as the root by default (observed on the SDK 2.32 DLAMI).
    root = os.environ.get("NEURON_LIBTORCH_CACHE_ROOT", os.path.expanduser("~/.cache/neuron_libtorch"))
    return os.path.join(root, "neuron", "compile_cache")


def _aws(*args: str) -> str:
    return subprocess.run(["aws", "s3", *args], check=True, capture_output=True, text=True).stdout


def _remote_keys(uri: str) -> set[str]:
    # `aws s3 ls` exits 1 with nothing on stderr for a prefix that holds no objects yet.
    p = subprocess.run(["aws", "s3", "ls", uri.rstrip("/") + "/"], capture_output=True, text=True)
    if p.returncode != 0:
        if p.returncode == 1 and not p.stderr.strip():
            return set()
        raise RuntimeError(f"aws s3 ls {uri} failed ({p.returncode}): {p.stderr.strip()}")
    out = p.stdout
    return {line.split()[-1].rstrip("/") for line in out.splitlines() if line.strip().startswith("PRE ")}


def pull(uri: str) -> int:
    """Returns the number of entries downloaded."""
    root = local_root()
    os.makedirs(root, exist_ok=True)
    have = {d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, MARKER))}
    missing = sorted(_remote_keys(uri) - have - {NKI_SUBDIR, "_nki_kernels"})
    for key in missing:
        _aws("cp", "--recursive", "--quiet", f"{uri.rstrip('/')}/{key}/", os.path.join(root, key))
    _sync_nki(uri, push=False)
    return len(missing)


def _sync_nki(uri: str, push: bool) -> None:
    """The NKI compile results and the kernel binaries they reference (module docstring)."""
    base = uri.rstrip("/")
    for local, remote in ((os.path.join(local_root(), NKI_SUBDIR), f"{base}/{NKI_SUBDIR}/"),
                          (NKI_KERNEL_DIR, f"{base}/_nki_kernels/")):
        if push:
            if os.path.isdir(local):
                _aws("sync", "--quiet", local, remote, "--exclude", "*.lock")
        else:
            os.makedirs(local, exist_ok=True)
            _aws("sync", "--quiet", remote, local)


def push_entry(uri: str, d: str) -> None:
    """Upload one complete entry, the marker LAST."""
    dest = f"{uri.rstrip('/')}/{os.path.basename(d.rstrip('/'))}/"
    _aws("cp", "--recursive", "--quiet", d, dest, "--exclude", MARKER, "--exclude", "*.lock")
    _aws("cp", "--quiet", os.path.join(d, MARKER), dest + MARKER)


def push_nki(uri: str) -> None:
    _sync_nki(uri, push=True)


def pull_nki(uri: str) -> None:
    """Only the NKI results and kernel binaries: what a compile-farm host needs to compile graphs
    that call NKI kernels (kiln/compile_farm.py)."""
    _sync_nki(uri, push=False)


def push(uri: str) -> int:
    """Returns the number of entries uploaded."""
    root = local_root()
    if not os.path.isdir(root):
        return 0
    # Kernels first: a graph entry whose marker is up must find its kernels already there.
    _sync_nki(uri, push=True)
    remote = _remote_keys(uri)
    n = 0
    for key in sorted(os.listdir(root)):
        d = os.path.join(root, key)
        if key in remote or key == NKI_SUBDIR or not os.path.exists(os.path.join(d, MARKER)):
            continue
        push_entry(uri, d)
        n += 1
    return n


# -- pull on miss: a device run that waits for the compile farm ----------------------------------

def cache_key(gm, example_inputs, options: dict) -> str:
    """LNL's cache key for a graph, computed as compile() computes it (libtorch_neuronx_lite
    compile/backend.py: preprocess_and_validate_inputs, _apply_platform_compiler_args,
    cache.create_cache_hash), on a copy, so the graph LNL then receives is untouched."""
    import copy

    from libtorch_neuronx_lite.compile import cache
    from libtorch_neuronx_lite.compile.backend import _apply_platform_compiler_args, preprocess_and_validate_inputs

    gm2, inputs = preprocess_and_validate_inputs(copy.deepcopy(gm), list(example_inputs), options)
    return cache.create_cache_hash(gm2, inputs, _apply_platform_compiler_args(options))


def _s3_exists(uri: str) -> bool:
    return subprocess.run(["aws", "s3", "ls", uri], capture_output=True, text=True).returncode == 0


def fetch_key(uri: str, key: str) -> bool:
    """Copy <uri>/<key>/ into the local cache as one complete entry if the remote one is complete.
    The ranks of one host share the local cache: one copies under a file lock, the others find it."""
    import fcntl
    import shutil

    root = local_root()
    d = os.path.join(root, key)
    if os.path.exists(os.path.join(d, MARKER)):
        return True
    if not _s3_exists(f"{uri.rstrip('/')}/{key}/{MARKER}"):
        return False
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, f".{key}.farm.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(os.path.join(d, MARKER)):
            return True
        tmp = f"{d}.farm.{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        _aws("cp", "--recursive", "--quiet", f"{uri.rstrip('/')}/{key}/", tmp)
        if os.path.exists(d):  # a partial directory from an earlier compile: set it aside, as LNL does
            os.rename(d, f"{d}.trash.{os.getpid()}")
        os.rename(tmp, d)
    return True


def parse_key_records(ls_output: str) -> dict[str, int]:
    """{key: graph.hlo bytes} from `aws s3 ls <queue>/keys/` (objects named <key>__<bytes>)."""
    out = {}
    for line in ls_output.splitlines():
        name = line.split()[-1] if line.strip() else ""
        if "__" in name:
            k, n = name.split("__", 1)
            if n.isdigit():
                out[k] = int(n)
    return out


class FarmWait:
    """KILN_COMPILE_FARM=<queue uri> (tools/compile_farm.py enqueue): before LNL compiles a graph
    the farm was given, wait for the farm's NEFF and load that instead. Graphs the farm does not
    hold compile locally as before, and so does one the farm failed (its error is printed)."""

    def __init__(self, queue: str, poll: float = 15.0, timeout: float = 4 * 3600):
        import json

        self.queue, self.poll, self.timeout = queue.rstrip("/"), poll, timeout
        p = subprocess.run(["aws", "s3", "cp", "--quiet", f"{self.queue}/manifest.json", "-"],
                           capture_output=True, text=True, check=True)
        man = json.loads(p.stdout)
        self.cache, self._legacy = man["cache"].rstrip("/"), set(man.get("keys") or {})
        self.keys = self._list_keys()
        self.waited: dict[str, float] = {}

    def _list_keys(self) -> set[str]:
        """The queue's keys: manifest.json's (queues written before per-key records) and one
        keys/<key>__<graph.hlo bytes> object per enqueued graph (tools/compile_farm.py enqueue)."""
        p = subprocess.run(["aws", "s3", "ls", f"{self.queue}/keys/"], capture_output=True, text=True)
        return set(self._legacy) | set(parse_key_records(p.stdout))

    def before_compile(self, key: str) -> str:
        """'local', 'fetched', 'failed' (the farm's compile failed), or 'not-farmed'."""
        import time

        if os.path.exists(os.path.join(local_root(), key, MARKER)):
            return "local"
        if key not in self.keys:
            self.keys = self._list_keys()  # enqueued after this run started
            if key not in self.keys:
                # A complete entry in the queue's cache is as good as a farmed one. Queues enqueued
                # before tools/compile_farm.py recorded already-compiled keys list none of those, and
                # a box whose local cache lacked them compiled them again (kiln-trn2-b, 2026-10-04:
                # five decode groups of q/t2max-EU, 4-5 min each, all complete in the cache).
                if fetch_key(self.cache, key):
                    self.waited[key] = 0.0
                    return "fetched"
                return "not-farmed"
        t0 = time.time()
        while True:
            if fetch_key(self.cache, key):
                self.waited[key] = time.time() - t0
                return "fetched"
            if _s3_exists(f"{self.queue}/failed/{key}.json"):
                p = subprocess.run(["aws", "s3", "cp", "--quiet", f"{self.queue}/failed/{key}.json", "-"],
                                   capture_output=True, text=True)
                print(f"kiln compile farm: {key} failed on the farm: {p.stdout[-600:]}", flush=True)
                return "failed"
            if time.time() - t0 > self.timeout:
                return "timeout"
            time.sleep(self.poll)
