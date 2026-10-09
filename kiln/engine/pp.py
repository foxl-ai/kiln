"""One long prefill over several engines: a pipeline of layer ranges (EngineConfig.pp_stages > 1, opt-in).

Each stage engine (a process tree of its own, on its own box or cores) holds the same model and runs the same
scheduler on the same requests, so its chunking, block tables and slots are those of every other stage. A prefill
call on stage s runs only the layers of its range [lo, hi) (EngineConfig.pp_split: the first layer of every stage
after the first). Stage 0 embeds the chunk; every later stage takes the hidden stream ([rows, hc x hidden] of the
call, after stage s-1's last layer) from stage s-1 instead; every stage but the last sends its output to the next.
Every stage keeps the KV, the DSA history and the KDA states of its own layers exactly as one engine does (a layer's
caches never leave its stage), so the arithmetic of every layer is the one engine's, and the last stage's logits,
first token and prompt logprobs are one engine's. Stages before the last also run the post graph on their hidden
state (their tokens are discarded: a pipeline engine is a prefill engine, max_new_tokens 1).

Transport: one TCP connection per (stage boundary, rank): rank r of stage s connects to rank r of stage s + 1, which
listens on pp_listen's port + r. The frames are kiln/engine/disagg.py's (a JSON header and the raw bytes), one per
call: {"call": n, "dtype", "shape"} and the tensor's bytes; the receiver checks the call number, so a pipeline that
lost step with itself fails loudly instead of mixing chunks.

KILN_PP_ASYNC=1 (opt-in, host only: no graph, input or key changes): the TCP half of every hop leaves the stage's
own thread. A sender thread writes the frames in call order, and a receiver thread reads the next frame ahead into a
bounded queue while the stage computes, so a later stage's recv only waits when its upstream is actually behind. A
failed send or receive is re-raised on the stage's thread at its next send / recv, never dropped. Every DEVICE
operation (the output's copy to the host, the received frame's copy to the device) stays on the stage's own thread: a
copy to the host on the sender thread deadlocked against the main thread's launch of the post graph on trn2 (2026-10-07,
trn2.48xlarge LNC=2, two stages in one box: a stage-0 worker's sender sat in h.to("cpu") while its main thread
launched post_prefill, and rank 0 then waited forever in that graph's collective; py-spy, s3
logs/kiln-t2-cb2/pyspy-ppb.txt), and on trn1 it had only happened not to.

KILN_PP_OVERLAP=1 (with KILN_PP_ASYNC=1) lets a stage run the engine's overlap scheduling (LLMEngine._overlap_ok): chunk
n + 1 is scheduled and launched before chunk n's output is read, so the device does not sit idle through the host's
per-step work. For that the main thread must not wait for a call's own layers inside its launch, so the copy to the
host is DEFERRED by one call: send(h) of call n + 1 (its layers already queued) copies call n's output, which waits for
call n only, then hands the bytes to the sender thread; the engine flushes the last one when a step launches no
further prefill (LLMEngine._step_overlap, ModelRunner.pp_flush on every rank).
Following stages (EngineConfig.pp_follow): only stage 0 is given requests (by a server or a driver), and stages 1 ..
S - 1 follow it instead of running in lockstep with it. Stage 0's rank 0 puts its step's plan in the header of the
step's first frame ({"step", "events": the requests added or aborted since its last plan frame, with their sampling
parameters, handoff and wall-clock arrival, "plan": [rid, start, tokens] per prefill call, "ids": each new request's
token ids as int32 payload}); a following stage's engine (LLMEngine.follow) reads that header ahead of the frame
(StageLink.peek), applies the events, runs one step of its own (deterministic) scheduler, refuses loudly a plan that
differs from stage 0's, and passes the header on to the next stage. The run ends with a control frame ({"close": true})
that every stage passes on. Needs KILN_PP_ASYNC=1 (the frames are read ahead).

KILN_TIMELINE (rank 0 only): pp_d2h (the host copy of the output, which waits for the stage's layers), pp_send (the
TCP write, or the enqueue under KILN_PP_ASYNC), pp_d2h_thread / pp_send_thread (the same on the sender thread),
pp_wait (the host blocked until the upstream frame was there), pp_h2d (the received frame onto the device), each with
the call number and its bytes.

The layer split, by measured time (pp_split None): every layer weighs 1, a pooled DSA layer DSA_WEIGHT; the stages
get contiguous ranges of about equal weight (default_split). Measured weights: a CP DSA layer's token mixer ~108 ms
per 1024-row call at a 1M bucket against ~17 ms for a KDA + MoE layer (docs/neuron-notes.md "Long context (1M)", the
replay of the trn1 CP call).
"""

from __future__ import annotations

import os
import queue
import socket
import threading
import time

import numpy as np
import torch

from .. import profiling

ASYNC = os.environ.get("KILN_PP_ASYNC", "0") == "1"
OVERLAP = os.environ.get("KILN_PP_OVERLAP", "0") == "1"

DSA_WEIGHT = float(os.environ.get("KILN_PP_DSA_WEIGHT", "6.0"))


def default_split(kinds: list[bool], stages: int) -> tuple[int, ...]:
    """The first layer of every stage after the first: contiguous ranges of about equal weight, a layer weighing
    DSA_WEIGHT when kinds[i] (a pooled DSA layer) and 1 otherwise; every stage at least one layer."""
    n = len(kinds)
    if stages < 2:
        return ()
    if stages > n:
        raise ValueError(f"pipeline: {stages} stages for {n} layers")
    w = [DSA_WEIGHT if k else 1.0 for k in kinds]
    total = sum(w)
    out, acc, s = [], 0.0, 1
    for i in range(n):
        acc += w[i]
        need = n - (i + 1)  # layers left
        if s < stages and (acc >= total * s / stages or need == stages - s) and need >= stages - s:
            out.append(i + 1)
            s += 1
    while len(out) < stages - 1:  # (only for degenerate weights)
        out.append(out[-1] + 1 if out else 1)
    return tuple(out)


def heavy(layer) -> bool:
    """A layer default_split weighs DSA_WEIGHT: an attention layer with a DSA indexer or MLA (the long path's)."""
    return getattr(layer, "dsa", None) is not None or bool(getattr(getattr(layer, "spec", None), "mla", None))


def model_stage_range(layers, stages: int, stage: int, split=None) -> tuple[int, int]:
    """The layers [lo, hi) stage `stage` of `stages` runs and (by default) loads: pp_split when given, else
    default_split over the layers' weights. ModelRunner (the stage's plan) and the loader (the stage's weights) both
    call this, so they cannot disagree."""
    split = tuple(split) if split else default_split([heavy(l) for l in layers], stages)
    return stage_range(len(layers), split, stage)


# KILN_PP_ALL_LAYERS=1: every stage loads every layer's weights (before 2026-10-07 always so), not only its own.
ALL_LAYERS = os.environ.get("KILN_PP_ALL_LAYERS", "0") == "1"


def stage_range(n_layers: int, split: tuple[int, ...], stage: int) -> tuple[int, int]:
    bounds = (0, *split, n_layers)
    if not all(a < b for a, b in zip(bounds, bounds[1:])):
        raise ValueError(f"pipeline split {split} over {n_layers} layers")
    return bounds[stage], bounds[stage + 1]


def _addr(s: str) -> tuple[str, int]:
    host, port = s.rsplit(":", 1)
    return host, int(port)


class StageLink:
    """This rank's two ends of the pipeline: the upstream connection (stages after the first: accepted on
    pp_listen's port + rank) and the downstream one (stages before the last: to pp_next's port + rank). Opened on
    first use; the frames are disagg.py's."""

    def __init__(self, ecfg, rank: int):
        self.stage, self.stages, self.rank = ecfg.pp_stage, ecfg.pp_stages, rank
        self.listen = _addr(ecfg.pp_listen) if self.stage > 0 else None
        self.next = _addr(ecfg.pp_next) if self.stage < self.stages - 1 else None
        self._srv = None
        self._up = None
        self._down = None
        self.sent = self.recvd = 0
        self.bytes = 0
        self._failed: BaseException | None = None
        self._pending: tuple[int, torch.Tensor] | None = None  # (call, device output) whose copy is deferred
        self.defer = False  # set per step by the engine (ModelRunner.pp_set_defer): an overlapped step defers
        # Rank 0 under a following pipeline (LLMEngine "plan frames"): what the next send carries in its header beside
        # the call (stage 0's step plan and its new requests), and the frame read ahead by peek().
        self.next_meta: dict | None = None
        self.last_meta: dict | None = None
        self._stash = None
        self._sq = self._rq = None
        self._st = self._rt = None
        if self.listen is not None:  # listen at once, so the upstream stage can connect whenever it is up
            host, port = self.listen
            self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._srv.bind((host, port + rank))
            self._srv.listen(1)

    def _connect(self) -> socket.socket:
        host, port = self.next
        deadline = time.time() + float(os.environ.get("KILN_PP_CONNECT_S", "600"))
        while True:
            try:
                s = socket.create_connection((host, port + self.rank), timeout=30)
                s.settimeout(None)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                return s
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.5)

    def _rec(self, kind: str, t0: float, t1: float, *fields) -> None:
        if self.rank == 0:
            profiling.record(kind, t0, t1, *fields)

    def _raise_failed(self) -> None:
        if self._failed is not None:
            raise RuntimeError(f"pipeline stage {self.stage} rank {self.rank}: the link thread failed") from self._failed

    def _sender(self) -> None:
        from . import disagg

        try:
            while True:
                item = self._sq.get()
                if item is None:
                    return
                hdr, t, extra = item  # host tensors: no device operation on this thread
                t0 = time.perf_counter()
                self.bytes += disagg.send_frame(self._down, hdr, [disagg.to_bytes(t), *extra])
                self._rec("pp_send_thread", t0, time.perf_counter(), hdr["call"], t.numel() * t.element_size())
        except BaseException as e:  # noqa: BLE001 - re-raised on the stage's thread (_raise_failed)
            self._failed = e

    def _host(self, call: int, h: torch.Tensor, meta: dict | None = None) -> None:
        """The stage's own thread: call's output to the host, then to the sender thread (or the socket). meta (rank 0
        of a following pipeline): carried in the header; its "ids" (rid -> token id list) go as int32 payload after the
        tensor's bytes, so a 1M-token prompt costs 4 MB of bytes rather than a JSON list."""
        from . import disagg

        t0 = time.perf_counter()
        t = h.to("cpu").contiguous()
        t1 = time.perf_counter()
        hdr = {"call": call, "dtype": disagg._dtype_name(t.dtype), "shape": list(t.shape)}
        extra = []
        if meta is not None:
            meta = dict(meta)
            ids = meta.pop("ids", {})
            hdr["meta"] = meta
            hdr["ids"] = [[rid, len(v)] for rid, v in ids.items()]
            extra = [memoryview(np.asarray(v, np.int32)).cast("B") for v in ids.values()]
        nb = t.numel() * t.element_size()
        if ASYNC:
            self._sq.put((hdr, t, extra))
        else:
            self.bytes += disagg.send_frame(self._down, hdr, [disagg.to_bytes(t), *extra])
        self._rec("pp_d2h", t0, t1, call, nb)
        self._rec("pp_send", t1, time.perf_counter(), call, nb)

    def flush(self) -> None:
        """The deferred output (KILN_PP_OVERLAP), if any, goes out now: the last chunk of a burst of prefill calls."""
        self._raise_failed()
        if self._pending is not None:
            call, h, meta = self._pending
            self._pending = None
            self._host(call, h, meta)

    def send(self, h: torch.Tensor) -> None:
        self._raise_failed()
        if self._down is None:
            self._down = self._connect()
            if ASYNC:
                self._sq = queue.Queue(maxsize=4)
                self._st = threading.Thread(target=self._sender, name=f"pp-send-{self.rank}", daemon=True)
                self._st.start()
        meta, self.next_meta = self.next_meta, None
        if ASYNC and OVERLAP and self.defer:  # this call's layers are queued: copy the PREVIOUS call's output, keep it
            self.flush()
            self._pending = (self.sent, h.detach(), meta)
        else:
            self._host(self.sent, h.detach(), meta)
        self.sent += 1

    def send_control(self, meta: dict) -> None:
        """Rank 0 of a following pipeline: a frame with no tensor and no call number ({"ctl": ...}), e.g. the end of
        the run; the next stage's engine reads it with peek() / pop_control(), never its recv()."""
        from . import disagg

        self._raise_failed()
        self.flush()
        if self._down is None:
            self._down = self._connect()
        hdr = {"call": -1, "ctl": meta}
        if ASYNC:
            self._sq.put((hdr, torch.empty(0, dtype=torch.uint8), []))
        else:
            self.bytes += disagg.send_frame(self._down, hdr, [])

    def _read_frame(self):
        """One frame off the upstream socket: (header, host tensor), or None when the upstream closed."""
        from . import disagg

        got = disagg.recv_header(self._up)
        if got is None:
            return None
        hdr, n = got
        buf = disagg._recv_exact(self._up, n) if n else b""
        if buf is None:
            raise RuntimeError(f"pipeline stage {self.stage} rank {self.rank}: upstream closed mid-frame")
        if "ctl" in hdr:
            return hdr, None
        t = disagg.from_bytes(buf, 0, disagg._dtype(hdr["dtype"]), hdr["shape"])
        if hdr.get("ids"):  # a plan frame's request token ids, after the tensor's bytes
            off = t.numel() * t.element_size()
            ids = {}
            for rid, k in hdr["ids"]:
                ids[rid] = np.frombuffer(buf, dtype=np.int32, count=k, offset=off).tolist()
                off += 4 * k
            hdr["meta"]["ids"] = ids
        return hdr, t

    def _receiver(self) -> None:
        try:
            while True:
                f = self._read_frame()
                self._rq.put(f)
                if f is None:
                    return
        except BaseException as e:  # noqa: BLE001 - re-raised on the stage's thread (recv)
            self._failed = e
            self._rq.put(None)

    def recv(self, device) -> torch.Tensor:
        self._raise_failed()
        self._accept()
        t0 = time.perf_counter()
        if self._stash is not None:
            got, self._stash = self._stash, None
        else:
            got = self._rq.get() if ASYNC else self._read_frame()
        t1 = time.perf_counter()
        if got is None:
            self._raise_failed()
            raise RuntimeError(f"pipeline stage {self.stage} rank {self.rank}: upstream closed")
        hdr, t = got
        if "ctl" in hdr:
            raise RuntimeError(f"pipeline stage {self.stage} rank {self.rank}: a control frame {hdr['ctl']} where call "
                               f"{self.recvd} was expected")
        self.last_meta = hdr.get("meta")
        if hdr["call"] != self.recvd:
            raise RuntimeError(f"pipeline stage {self.stage} rank {self.rank}: got call {hdr['call']}, expected "
                               f"{self.recvd} (the stages lost step)")
        out = t.to(device)
        t2 = time.perf_counter()
        nb = t.numel() * t.element_size()
        self._rec("pp_wait", t0, t1, self.recvd, nb)
        self._rec("pp_h2d", t1, t2, self.recvd, nb)
        self.recvd += 1
        return out

    def _accept(self) -> None:
        if self._up is None:
            self._up, _ = self._srv.accept()
            self._up.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if ASYNC:
                self._rq = queue.Queue(maxsize=2)
                self._rt = threading.Thread(target=self._receiver, name=f"pp-recv-{self.rank}", daemon=True)
                self._rt.start()

    def peek(self, timeout: float | None = None) -> dict | None:
        """Rank 0 of a following stage (KILN_PP_ASYNC): the header of the next frame, read ahead and kept for recv(),
        or None when none arrived within timeout. A frame whose upstream closed raises."""
        self._raise_failed()
        if self._stash is None:
            if not ASYNC:
                raise RuntimeError("a following pipeline stage needs KILN_PP_ASYNC=1 (its frames are read ahead)")
            self._accept()
            try:
                got = self._rq.get(timeout=timeout)
            except queue.Empty:
                return None
            if got is None:
                self._raise_failed()
                raise RuntimeError(f"pipeline stage {self.stage} rank {self.rank}: upstream closed")
            self._stash = got
        return self._stash[0]

    def pop_control(self) -> dict:
        hdr, _ = self._stash
        self._stash = None
        return hdr["ctl"]

    def close(self) -> None:
        """Idempotent. Every frame still queued on the sender thread goes out before the socket closes: a stage that
        finished its last chunk exits while its sender may still be writing it (seen on trn1 with KILN_PP_ASYNC=1:
        the next stage read "upstream closed mid-frame" on the 1M request's last chunk, 2026-10-07)."""
        if self._pending is not None and self._down is not None:
            self.flush()
        if self._st is not None:
            self._sq.put(None)
            self._st.join(timeout=600)
            if self._st.is_alive():
                raise RuntimeError(f"pipeline stage {self.stage} rank {self.rank}: frames still unsent after 600 s")
            self._st = None
        self._raise_failed()
        for s in (self._up, self._down, self._srv):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass

