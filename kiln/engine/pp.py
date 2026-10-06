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

The layer split, by measured time (pp_split None): every layer weighs 1, a pooled DSA layer DSA_WEIGHT; the stages
get contiguous ranges of about equal weight (default_split). Measured weights: a CP DSA layer's token mixer ~108 ms
per 1024-row call at a 1M bucket against ~17 ms for a KDA + MoE layer (docs/neuron-notes.md "Long context (1M)", the
replay of the trn1 CP call).
"""

from __future__ import annotations

import os
import socket
import time

import torch

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

    def send(self, h: torch.Tensor) -> None:
        from . import disagg

        if self._down is None:
            self._down = self._connect()
        t = h.detach().to("cpu").contiguous()
        hdr = {"call": self.sent, "dtype": disagg._dtype_name(t.dtype), "shape": list(t.shape)}
        self.bytes += disagg.send_frame(self._down, hdr, [disagg.to_bytes(t)])
        self.sent += 1

    def recv(self, device) -> torch.Tensor:
        from . import disagg

        if self._up is None:
            self._up, _ = self._srv.accept()
            self._up.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        got = disagg.recv_header(self._up)
        if got is None:
            raise RuntimeError(f"pipeline stage {self.stage} rank {self.rank}: upstream closed")
        hdr, n = got
        if hdr["call"] != self.recvd:
            raise RuntimeError(f"pipeline stage {self.stage} rank {self.rank}: got call {hdr['call']}, expected "
                               f"{self.recvd} (the stages lost step)")
        buf = disagg._recv_exact(self._up, n)
        if buf is None:
            raise RuntimeError(f"pipeline stage {self.stage} rank {self.rank}: upstream closed mid-frame")
        self.recvd += 1
        t = disagg.from_bytes(buf, 0, disagg._dtype(hdr["dtype"]), hdr["shape"])
        return t.to(device)

    def close(self) -> None:
        for s in (self._up, self._down, self._srv):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
