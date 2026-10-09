"""Device-to-device KV handoff over NIXL (KILN_PD_TRANSPORT=nixl, opt-in): a decode rank reads a handed-off request's
rows straight out of the prefill ranks' paged caches and state pools into its own, over EFA on Neuron (the LIBFABRIC
backend, device memory registered as VRAM), with no host copy on either side. The host path of engine/disagg.py
(device -> host, TCP, /dev/shm, host -> device) stays the default and is what every other engine speaks.

What makes Neuron memory registrable: libtorch_neuronx_lite's allocator (libtorchneuron.so,
neuron::NeuronAllocator::allocate, csrc/neuron_op/storage.cpp) stores nrt_tensor_get_va() as a tensor's data pointer
only when NEURON_RT_MAP_HBM is "true" or a positive integer when the process allocates its first device tensor;
otherwise data_ptr() is 0 and NIXL refuses the region ("Failed to retrieve placement for VA: (nil)"). vllm-neuron 0.24
sets it the same way (vllm_neuron/vllm/worker/neuron_worker.py: "Enable HBM mapping (required for RDMA in
disaggregated inference)"). engine.build_shard sets it on every rank when this transport is on, and Rank refuses a
tensor whose data pointer is 0. Measured (docs/neuron-notes.md "Device-to-device KV handoff over EFA"): trn1.32xlarge,
SDK 2.32, nixl 1.3.2, one NeuronCore pair 11.4 GB/s, verified byte for byte.

The protocol (pull, as vLLM's NixlConnector reads blocks):
- every rank of both engines registers its paged caches and state pools once (Rank) and writes an export (its agent
  metadata and each region's address, size and row bytes) for rank 0;
- the prefill engine's sender opens every connection with a "hello" frame holding all of its ranks' exports
  (disagg.Sender hello); the decode receiver keeps the latest one per prefill engine in its directory, where every
  decode rank reads it (disagg.Receiver);
- a handoff is a meta frame alone (transport "nixl"): the request's pages, page size, state row, the sending group's
  ranks, a hold deadline, the bytes the host path would have moved (counted against the receive buffer from arrival
  to release, so pd_buffer_gb still bounds the handoffs in flight and a full buffer still blocks the connection), and
  the address of the prefill engine's release listener;
- the prefill engine keeps the request's pages (radix lock and tail) and state row out of reuse (scheduler pd_pin)
  until the decode engine's release frame for that transfer arrives, or the deadline passes (logged);
- the decode engine admits the request as before; when the step that admits it is planned, every rank of its group
  reads its rows from the matching sender rank (segments: position runs consecutive on both sides, one descriptor
  each, one transfer per remote rank) and waits for completion before the step's calls are launched; after that
  step is read back on rank 0 (so every rank has finished reading) the engine releases the transfer and sends the
  release frame. A refused, aborted or expired handoff is released the same way.

Regrouping across attention TP degrees (a latency prefill engine): replicated caches are read whole from one sender
rank; the head-split state rows of several sender ranks are read into a registered host buffer and regrouped there
(disagg.regroup), because a narrow along a row's split axis is not one contiguous range.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time

import numpy as np
import torch

TRANSPORTS = ("host", "tcp", "nixl")  # "tcp" is the host path's other name


def transport() -> str:
    """KILN_PD_TRANSPORT: host (default; engine/disagg.py's device -> host -> TCP -> host -> device) or nixl."""
    t = os.environ.get("KILN_PD_TRANSPORT", "host")
    if t not in TRANSPORTS:
        raise ValueError(f"KILN_PD_TRANSPORT must be one of {TRANSPORTS}, not {t!r}")
    return t


def enabled() -> bool:
    return transport() == "nixl"


def backend(device_type: str) -> str:
    """LIBFABRIC on Neuron (EFA, the backend vllm-neuron's disaggregated inference uses); UCX elsewhere (loopback and
    TCP on any host, which is how the CPU tests run the same code path). KILN_PD_NIXL_BACKEND overrides."""
    return os.environ.get("KILN_PD_NIXL_BACKEND") or ("LIBFABRIC" if device_type == "neuron" else "UCX")


def hold_seconds() -> float:
    """How long a prefill engine keeps a handed-off request's pages for a decode engine that has not released them."""
    return float(os.environ.get("KILN_PD_NIXL_HOLD_S", "900"))


def pos_slots(pages, ps: int, pos: np.ndarray) -> np.ndarray:
    """Cache slots of positions pos (pages in position order, ps slots per page)."""
    return np.asarray(pages, np.int64)[pos // ps] * ps + pos % ps


def local_page_slots(pages, n_tok: int, ps: int, lps: int) -> np.ndarray:
    """A context-parallel rank's local slots of whole pages (ModelRunner._pd_local_page_runs, as an array)."""
    n_pages = -(-n_tok // ps)
    pg = np.asarray(pages[:n_pages], np.int64)
    return (pg[:, None] * lps + np.arange(lps)[None, :]).ravel()


def segments(remote: np.ndarray, local: np.ndarray) -> list[tuple[int, int, int]]:
    """(first remote slot, first local slot, rows) runs where both slot sequences go up by one: one descriptor each."""
    if len(remote) != len(local):
        raise ValueError(f"{len(remote)} remote slots against {len(local)} local ones")
    if not len(remote):
        return []
    cut = np.flatnonzero((np.diff(remote) != 1) | (np.diff(local) != 1)) + 1
    b = np.concatenate([[0], cut, [len(remote)]])
    return [(int(remote[x]), int(local[x]), int(y - x)) for x, y in zip(b[:-1], b[1:])]


class Rank:
    """One rank's NIXL agent with its regions (name -> tensor, each contiguous, rows along dim 0) registered."""

    def __init__(self, name: str, regions: list[tuple[str, torch.Tensor]], device: torch.device):
        from nixl._api import nixl_agent, nixl_agent_config

        self.mem = "VRAM" if device.type == "neuron" else "DRAM"
        self.backend = backend(device.type)
        self.name = name
        self.agent = nixl_agent(name, nixl_agent_config(backends=[self.backend]))
        self.table: dict[str, tuple[int, int, int]] = {}  # name -> (address, bytes, row bytes)
        descs = []
        for rname, t in regions:
            if not t.is_contiguous():
                raise ValueError(f"region {rname} is not contiguous")
            va, n = t.data_ptr(), t.numel() * t.element_size()
            if n == 0:
                continue
            if va == 0:
                raise RuntimeError(f"region {rname}: data_ptr() is 0. A Neuron tensor has its device address there only "
                                   "with NEURON_RT_MAP_HBM=1 set before the process allocates its first device tensor "
                                   "(engine.build_shard sets it under KILN_PD_TRANSPORT=nixl)")
            self.table[rname] = (va, n, n // t.shape[0])
            descs.append((va, n, 0, ""))
        t0 = time.perf_counter()
        self.agent.register_memory(self.agent.get_reg_descs(descs, self.mem), backends=[self.backend])
        self.register_seconds = time.perf_counter() - t0
        self._remotes: dict[str, str] = {}
        self._staging = None  # (host tensor, address, bytes): registered DRAM for regrouped state rows
        self.read_bytes = 0
        self.read_seconds = 0.0
        self.reads = 0
        self.read_times: list[float] = []  # per read() call (one handoff on this rank), the last 4096

    def export(self, **extra) -> dict:
        return {"agent": self.name, "meta": base64.b64encode(self.agent.get_agent_metadata()).decode(),
                "mem": self.mem, "regions": {k: list(v) for k, v in self.table.items()}, **extra}

    def _remote(self, exp: dict) -> str:
        r = self._remotes.get(exp["agent"])
        if r is None:
            r = self._remotes[exp["agent"]] = self.agent.add_remote_agent(base64.b64decode(exp["meta"]))
        return r

    def staging(self, n: int) -> tuple[torch.Tensor, int]:
        """A registered host buffer of at least n bytes (kept and grown as needed)."""
        if self._staging is None or self._staging[2] < n:
            if self._staging is not None:
                self.agent.deregister_memory(self._staging[3], backends=[self.backend])
            buf = torch.empty(max(n, 1 << 20), dtype=torch.uint8)
            va = buf.data_ptr()
            reg = self.agent.register_memory(self.agent.get_reg_descs([(va, buf.numel(), 0, "")], "DRAM"),
                                             backends=[self.backend])
            self._staging = (buf, va, buf.numel(), reg)
        return self._staging[0], self._staging[1]

    def read(self, plans: list[tuple[dict, list[tuple[int, int, int]], str]], what: str = "") -> None:
        """plans: (remote export, [(local address, remote address, bytes)], local memory type) per remote rank. All the
        transfers start at once and this returns when every one is done; an error raises (never a partial handoff)."""
        t0 = time.perf_counter()
        handles, total = [], 0
        for exp, segs, local_mem in plans:
            if not segs:
                continue
            ld = self.agent.get_xfer_descs([(la, n, 0) for la, _, n in segs], local_mem)
            rd = self.agent.get_xfer_descs([(ra, n, 0) for _, ra, n in segs], exp["mem"])
            h = self.agent.initialize_xfer("READ", ld, rd, self._remote(exp), b"")
            st = self.agent.transfer(h)
            if st == "ERR":
                raise RuntimeError(f"nixl read {what} from {exp['agent']}: transfer failed to start")
            handles.append((h, exp["agent"], st))
            total += sum(n for _, _, n in segs)
        pending = [(h, a) for h, a, st in handles if st != "DONE"]
        deadline = time.monotonic() + float(os.environ.get("KILN_PD_NIXL_READ_TIMEOUT_S", "120"))
        while pending:
            left = []
            for h, a in pending:
                st = self.agent.check_xfer_state(h)
                if st == "ERR":
                    raise RuntimeError(f"nixl read {what} from {a} failed")
                if st != "DONE":
                    left.append((h, a))
            pending = left
            if pending and time.monotonic() > deadline:
                raise TimeoutError(f"nixl read {what}: {len(pending)} transfers not done after "
                                   "KILN_PD_NIXL_READ_TIMEOUT_S")
        for h, _, _ in handles:
            self.agent.release_xfer_handle(h)
        dt = time.perf_counter() - t0
        self.reads += 1
        self.read_bytes += total
        self.read_seconds += dt
        self.read_times.append(dt)
        del self.read_times[:-4096]


def write_export(d: str, rank: int, exp: dict) -> None:
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".r{rank}.json.tmp")
    with open(tmp, "w") as f:
        json.dump(exp, f)
    os.replace(tmp, os.path.join(d, f"r{rank}.json"))


def read_exports(d: str, world: int, timeout: float = 600.0) -> list[dict]:
    """Every rank's export (write_export), waiting for ranks still registering."""
    end = time.monotonic() + timeout
    out = []
    for r in range(world):
        p = os.path.join(d, f"r{r}.json")
        while not os.path.exists(p):
            if time.monotonic() > end:
                raise TimeoutError(f"rank {r} wrote no nixl export into {d} within {timeout:g} s")
            time.sleep(0.05)
        with open(p) as f:
            out.append(json.load(f))
    return out


def log(msg: str) -> None:
    print(f"kiln pd nixl: {msg}", file=sys.stderr, flush=True)
