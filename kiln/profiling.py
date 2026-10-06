"""Host-side instrumentation for the utilization report (tools/util_report.py). Both parts are off unless
their environment variable is set, and neither changes a graph, its inputs or its compile key.

KILN_TIMELINE=<path>: rank 0 records every graph call (ModelRunner._exec: its broadcast to the other
tensor-parallel ranks, the upload of its host arguments, the launch of its graphs), every read-back of a
call's output (LLMEngine._collect: how long the host waited for the device) and every step() call with
the work it launched, as perf_counter seconds; written as JSON lines when the process exits (or by
`flush()`). Each record is a list: [kind, t_start, t_end, *fields]. Kinds:
  exec   name, key       t_start = _exec entered, then t_sent (broadcast done), t_up (arguments on the
                         device), t_end (every graph of the call launched)
  read   kind            t_end - t_start = the host blocked in .cpu() for the call's output
  step   n_dec, n_pre_tokens, decode_calls, prefill_calls
  sched  -               scheduler.schedule()

KILN_CAPTURE_INPUTS=<dir> with KILN_CAPTURE_AT=<name>:<n>[,<name>:<n>...] (name: decode, prefill, verify,
mixed or mtp; n counts this rank's calls of that name from 1): every NEFF executed during the chosen calls
has its exact input tuple written as raw .npy files, through libtorch_neuronx_lite's pre_execute_hook
(libtorch_neuronx_lite/compile/execute_context.py: "runs immediately before executor.execute with the
exact input tuple execute receives (post dead-input filtering and RNG-seed append)"), which is the NEFF's
input0..inputN order (neff.json arg_nodes). A tensor already written in this capture (the same tensor
object: the weights, the KV and state pools) is referenced instead of written again.
<dir>/r<rank>/manifest.jsonl gets one line per execution: {"call", "seq", "neff_id", "artifact_dir",
"inputs": [file, ...]}. tools/util_report.py replay feeds them back to `neuron-explorer capture
--multi-input`, so a replayed graph sees the serving run's routing and indices instead of zeros. Writing
a capture synchronises the host with the device (each input is read back), so the captured calls are
slow; the other calls of the run are untouched.
"""

from __future__ import annotations

import atexit
import json
import os
import time
import weakref

import numpy as np

_TL_PATH = os.environ.get("KILN_TIMELINE")
TIMELINE: list | None = [] if _TL_PATH else None


def now() -> float:
    return time.perf_counter()


def record(kind: str, t0: float, t1: float, *fields) -> None:
    if TIMELINE is not None:
        TIMELINE.append([kind, t0, t1, *fields])


def flush(path: str | None = None) -> None:
    """Write the timeline (JSON lines) and clear it."""
    global TIMELINE
    path = path or _TL_PATH
    if TIMELINE is None or not path:
        return
    with open(path, "a") as f:
        for r in TIMELINE:
            f.write(json.dumps(r) + "\n")
    TIMELINE.clear()


if TIMELINE is not None:
    atexit.register(flush)


def parse_at(spec: str) -> dict[str, set[int]]:
    """'prefill:40,decode:300,decode:301' -> {'prefill': {40}, 'decode': {300, 301}}."""
    out: dict[str, set[int]] = {}
    for part in filter(None, (p.strip() for p in (spec or "").split(","))):
        name, n = part.split(":")
        out.setdefault(name, set()).add(int(n))
    return out


def raw(t) -> np.ndarray:
    """A CPU tensor's bytes as a numpy array of the same element size (bf16 / fp8 as integers), which is
    what `neuron-explorer capture` reads from an input .npy file (tools/prof_engines.py)."""
    import torch

    t = t.contiguous()
    if t.dtype == torch.bool:
        return t.to(torch.uint8).numpy()
    if t.dtype.is_floating_point:
        view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
        return t.view(view).numpy()
    return t.numpy()


class InputCapture:
    """Writes the inputs of every NEFF executed while armed (see the module docstring)."""

    def __init__(self, root: str, rank: int, at: dict[str, set[int]]):
        self.dir = os.path.join(root, f"r{rank}")
        os.makedirs(self.dir, exist_ok=True)
        self.at = at
        self.counts: dict[str, int] = {}
        self.armed: tuple[str, int] | None = None
        self.seen: dict[int, tuple] = {}  # id(tensor) -> (weak reference, file)
        self.seq = 0
        self.nfiles = 0

    def begin(self, name: str) -> None:
        n = self.counts.get(name, 0) + 1
        self.counts[name] = n
        self.armed = (name, n) if n in self.at.get(name, ()) else None

    def end(self) -> None:
        self.armed = None

    def hook(self, inputs: tuple, meta) -> None:
        if self.armed is None:
            return
        files = []
        for t in inputs:
            # The same tensor OBJECT as an earlier input (a weight, a KV or state pool) is written once. A
            # device tensor of libtorch_neuronx_lite has no host-visible storage pointer ("Attempted to access
            # the data pointer on an invalid python storage", measured 2026-10-05), so the identity is the
            # object, held by a weak reference so that a new tensor reusing a dead one's id() is written again.
            ref, f = self.seen.get(id(t), (None, None))
            if ref is None or ref() is not t:
                f = os.path.join(self.dir, f"t{self.nfiles:05d}.npy")
                np.save(f, raw(t.detach().cpu()))
                self.seen[id(t)] = (weakref.ref(t), f)
                self.nfiles += 1
            files.append(f)
        with open(os.path.join(self.dir, "manifest.jsonl"), "a") as m:
            m.write(json.dumps({"call": f"{self.armed[0]}:{self.armed[1]}", "seq": self.seq,
                                "neff_id": getattr(meta, "neff_id", None),
                                "artifact_dir": getattr(meta, "artifact_dir", None),
                                "device_id": getattr(meta, "device_id", None), "inputs": files}) + "\n")
        self.seq += 1


CAPTURE: InputCapture | None = None


def install_capture(rank: int) -> InputCapture | None:
    """KILN_CAPTURE_INPUTS: install the pre-execute hook in this rank's process (once)."""
    global CAPTURE
    root = os.environ.get("KILN_CAPTURE_INPUTS")
    if not root or CAPTURE is not None:
        return CAPTURE
    CAPTURE = InputCapture(root, rank, parse_at(os.environ.get("KILN_CAPTURE_AT", "")))
    from libtorch_neuronx_lite.compile.execute_context import ExecuteContext, set_execute_context

    set_execute_context(ExecuteContext(pre_execute_hook=CAPTURE.hook))
    return CAPTURE
