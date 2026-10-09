"""Prefill / decode disaggregation (EngineConfig.pd_role): the handoff of a prefilled request from a
prefill engine to a decode engine on another process or box.

A prefill engine (pd_role "prefill") runs only prefill graphs. When a request's last prompt chunk has
sampled its first token, every rank of the request's DP-attention group copies its share of the
request's state to the host and sends it to the decode engine's receiver; the request then ends on the
prefill side (its pages stay in the prefix cache). A decode engine (pd_role "decode") runs only decode
graphs (plus, optionally, small prefill buckets for prompts the router does not disaggregate): a
complete handoff becomes a running request whose next step is a decode of the first token.

What a handoff carries
----------------------
Per request, everything a decode step reads that the prefill engine computed:

- the KV of the prompt positions in every paged cache: K and V of every attention layer (for MLA the
  latent and the rope key plus the DSA indexer key, whose rows also hold the indexer keys of the open,
  incomplete pool), the token-slot states beside them (the DSA pool-key cache) and the per-token aux
  caches. The rows are the prompt's slots in position order, i.e. the concatenation of the host KV
  tier's page copies (engine/hicache.py, ModelRunner._kv_save: ps rows of every paged cache per page),
  cut at the prompt's last position;
- the request's row of every recurrent-state pool (the KDA conv and recurrent state of every linear
  layer, and the model's aux state rows);
- the first sampled token with its logprobs, the prompt logprobs if asked for, the sampling params and
  the sampler's RNG state (numpy PCG64 state: the decode side continues the same random stream), and
  the request id, so that a seeded or greedy request decodes exactly as on one engine.

Layout and deduplication
------------------------
Each rank sends its own shard. Inside a DP-attention group of A ranks (attention TP A) the caches of an
MLA layer are identical on every rank (the latent is replicated, models/mla.py), so each rank sends
only its 1/A slice of the replicated caches' prompt slots (part "r<a>"), and its split data whole
(part "s<a>": the state rows, which are head-partitioned, and every cache that is not replicated). A
decode rank of attention rank a reads all r parts and its own s part. Both engines must have the same
layout (model, attention TP, KV dtype, cache and state shapes): the meta carries a signature of it and
the decode side refuses a handoff whose signature differs from its own.

Except for the attention TP when the model is regroupable (ModelRunner.pd_regroupable: only MLA
caches, which are the same rows at any degree, and head-split state pools): a prefill engine at DP
attention 1 (attention TP 32: one request per call over every rank, the latency prefill box) can hand
off to a decode engine at DP attention 4 (attention TP 8). The meta names the sender's degree; a
receiver rank reads every r part and the s parts of the sender ranks holding its heads
(sender_ranks), and rebuilds its state rows component by component (regroup).

Transport
---------
TCP, one connection per (sending process, receiver), frames of a JSON header and raw bytes. No pickle
crosses the wire: the header is JSON and the payload is the arrays' bytes, so a peer can send data but
not code. The receiver writes each part's payload to a file in a host-memory directory (/dev/shm), so
every decode rank (its own process) can read its parts at admission; rank 0 deletes them after the
first call that every rank has executed after the copy (engine.py). The receive side holds at most
pd_buffer_gb of payload: a part that does not fit waits on its connection (the sender's socket then
blocks, and so does its queue), counted in buffer_full_events / buffer_full_seconds and logged once per
episode, never dropped and never retried silently. The router keeps that from happening by
dispatching at most as many requests to a decode engine as it has room for (server/router.py).

KILN_PD_TRANSPORT=nixl (engine/nixl_kv.py, opt-in) keeps this framing for the meta and replaces the parts: the
sender opens every connection with a "hello" frame (its ranks' NIXL exports), a handoff is its meta alone, and each
decode rank reads its rows from the prefill ranks' device memory itself. The receiver still counts the handoff's bytes
against pd_buffer_gb from the meta's arrival to its release, so backpressure works as above.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import socket
import struct
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

MAGIC = b"KPD1"
_HDR = struct.Struct("!4sIQ")  # magic, header bytes, payload bytes


def _dtype_name(dt: torch.dtype) -> str:
    return str(dt).removeprefix("torch.")


def _dtype(name: str) -> torch.dtype:
    dt = getattr(torch, name, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"unknown dtype {name!r} in a handoff")
    return dt


def to_bytes(t: torch.Tensor) -> memoryview:
    """A host tensor's bytes, without a copy when it is contiguous (bf16 / fp8 have no numpy dtype)."""
    t = t.contiguous()
    return memoryview(t.view(torch.uint8).numpy()).cast("B") if t.numel() else memoryview(b"")


def from_bytes(buf, offset: int, dtype: torch.dtype, shape) -> torch.Tensor:
    n = int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size()
    if n == 0:
        return torch.empty(shape, dtype=dtype)
    return torch.frombuffer(buf, dtype=torch.uint8, count=n, offset=offset).view(dtype).reshape(shape)


def slot_runs(pages: list[int], start: int, end: int, ps: int) -> list[tuple[int, int]]:
    """Cache slots of positions start .. end - 1 (pages in position order) as (first slot, count) runs of
    consecutive slots: one host copy per run and cache."""
    if end <= start:
        return []
    pos = np.arange(start, end)
    slots = np.asarray(pages, np.int64)[pos // ps] * ps + pos % ps
    cut = np.flatnonzero(np.diff(slots) != 1) + 1
    bounds = np.concatenate([[0], cut, [len(slots)]])
    return [(int(slots[a]), int(b - a)) for a, b in zip(bounds[:-1], bounds[1:])]


def rep_slice(n: int, a: int, A: int) -> tuple[int, int]:
    """Attention rank a's share [lo, hi) of n rows of a replicated cache (part r<a>)."""
    return n * a // A, n * (a + 1) // A


def part_names(A: int) -> list[str]:
    return [f"s{a}" for a in range(A)] + [f"r{a}" for a in range(A)]


def regroup_ok(sender_a: int, a_total: int) -> bool:
    """Whether a handoff from attention TP sender_a can be split for attention TP a_total: one divides the other."""
    return sender_a >= 1 and a_total >= 1 and (sender_a % a_total == 0 or a_total % sender_a == 0)


def sender_ranks(sender_a: int, A: int, a: int) -> list[int]:
    """The sender attention ranks whose split part (s<b>) receiver attention rank a of A needs. The ranks split
    heads contiguously in rank order (models/loader.py _part), so with sender_a = m * A rank a's heads are those of
    sender ranks m * a .. m * a + m - 1, and with A = m * sender_a they are a 1/m share of sender rank a // m's."""
    if not regroup_ok(sender_a, A):
        raise ValueError(f"a handoff from attention TP {sender_a} cannot be split for attention TP {A}")
    if sender_a >= A:
        m = sender_a // A
        return list(range(m * a, m * a + m))
    return [a // (A // sender_a)]


def regroup(rows: dict[int, torch.Tensor], axis: int, widths: list[int], sender_a: int, A: int,
            a: int) -> torch.Tensor:
    """Receiver attention rank a's (of A) row of a head-split state pool, from the sender ranks' rows (rows: sender
    rank -> its row, sender_ranks()). axis: the row's split axis; widths: the receiver's component widths along it
    (a KDA conv state is q, then k, then v channels on every rank, so each component is regrouped on its own; a
    recurrent state is one component, its heads). The sender's widths are widths * A / sender_a."""
    src = sender_ranks(sender_a, A, a)
    if sender_a == A:
        return rows[src[0]]
    if sender_a > A:  # concatenate, component by component, the m sender ranks' pieces
        m = sender_a // A
        w_s = [w // m for w in widths]
        if any(w % m for w in widths):
            raise ValueError(f"state widths {widths} do not split over {m} sender ranks")
        for b in src:
            if rows[b].shape[axis] != sum(w_s):
                raise ValueError(f"sender rank {b}'s state row is {rows[b].shape[axis]} wide on axis {axis}, its "
                                 f"components {w_s} sum to {sum(w_s)}")
        pieces, off = [], 0
        for w in w_s:
            pieces += [rows[b].narrow(axis, off, w) for b in src]
            off += w
        return torch.cat(pieces, dim=axis)
    m = A // sender_a  # a 1/m share of one sender rank's row, component by component
    j = a % m
    row = rows[src[0]]
    w_s = [w * m for w in widths]
    if sum(w_s) != row.shape[axis]:
        raise ValueError(f"a sender's state row is {row.shape[axis]} wide on axis {axis}, its components {w_s} sum to "
                         f"{sum(w_s)}")
    pieces, off = [], 0
    for w, ws in zip(widths, w_s):
        pieces.append(row.narrow(axis, off + j * w, w))
        off += ws
    return torch.cat(pieces, dim=axis)


def signature(layout: dict) -> str:
    """A process-independent digest of an engine's handoff layout (sha256 of its sorted JSON; never
    Python's salted hash() of a str)."""
    return hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()[:16]


def default_store_dir() -> str:
    """Host-memory directory for received parts: KILN_PD_STORE, else /dev/shm, else the temp dir."""
    root = os.environ.get("KILN_PD_STORE") or ("/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir())
    return tempfile.mkdtemp(prefix=f"kiln-pd-{os.getpid()}-", dir=root)


# -- framing ----------------------------------------------------------------------------------------


def send_frame(sock: socket.socket, header: dict, buffers=()) -> int:
    h = json.dumps(header).encode()
    n = sum(len(b) for b in buffers)
    sock.sendall(_HDR.pack(MAGIC, len(h), n) + h)
    for b in buffers:
        if len(b):
            sock.sendall(b)
    return _HDR.size + len(h) + n


def _recv_exact(sock: socket.socket, n: int, into=None) -> bytearray | None:
    buf = into if into is not None else bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            return None
        got += k
    return buf


def _recv_file(sock: socket.socket, n: int, path: str) -> bool:
    """n bytes off the socket straight into the file at path (sized first, then mapped): the kernel copies them once,
    into the file's pages. Reading them into a bytearray and writing that out copied every byte twice more, and at 1M a
    pipeline's handoff is ~44 GB through one decode box (the host path's ~17 s, kiln-pd4-dec e2e4-r8). False when the
    peer closes inside the frame; then, and when the socket raises (a reset peer), the file is removed, as the old
    read-then-write path never created one."""
    import mmap

    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
    got = 0
    try:
        if n:
            os.ftruncate(fd, n)
            with mmap.mmap(fd, n) as mm:
                view = memoryview(mm)
                try:
                    while got < n:
                        k = sock.recv_into(view[got:], n - got)
                        if k == 0:
                            break
                        got += k
                finally:
                    view.release()  # before the map closes, also when recv_into raised (its traceback holds view)
    finally:
        os.close(fd)
        if got != n:
            os.unlink(path)
    return got == n


def recv_header(sock: socket.socket) -> tuple[dict, int] | None:
    raw = _recv_exact(sock, _HDR.size)
    if raw is None:
        return None
    magic, hn, pn = _HDR.unpack(bytes(raw))
    if magic != MAGIC:
        raise ConnectionError(f"not a Kiln handoff stream (magic {magic!r})")
    h = _recv_exact(sock, hn)
    if h is None:
        raise ConnectionError("connection closed inside a frame header")
    return json.loads(bytes(h)), pn


# -- sending ---------------------------------------------------------------------------------------


def _peer_closed(s: socket.socket) -> bool:
    """Whether the other end closed the connection (readable, and a peek returns EOF). A handoff receiver never writes
    to its peers, so anything readable on a sender's socket is the close."""
    import select

    try:
        r, _, _ = select.select([s], [], [], 0)
        if not r:
            return False
        return s.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT) == b""
    except (BlockingIOError, InterruptedError):
        return False
    except OSError:
        return True


@dataclass
class SendStats:
    frames: int = 0
    bytes: int = 0
    queued_bytes: int = 0
    blocked_seconds: float = 0.0  # time enqueue() waited for the queue to drain below max_bytes
    send_seconds: float = 0.0  # time the sending thread spent writing frames
    errors: int = 0


class Sender:
    """One background thread per process: a FIFO of frames, one persistent connection per destination
    ("host:port"). enqueue() returns at once unless more than max_bytes are queued, then it waits (and
    counts the wait), so a slow or full receiver slows the producer instead of growing host memory without
    bound. A send error is kept and raised by the next enqueue() / flush(): a handoff is never dropped
    silently."""

    def __init__(self, max_bytes: int = 8 << 30, hello=None):
        """hello: a callable returning (header, buffers) of a frame to write first on every new connection (the NIXL
        transport's exports, which a receiver that restarted needs again)."""
        self.max_bytes = max_bytes
        self.hello = hello
        self.stats = SendStats()
        self._q: queue.Queue = queue.Queue()
        self._cv = threading.Condition()
        self._socks: dict[str, socket.socket] = {}
        self._error: BaseException | None = None
        self._pending = 0  # frames enqueued and not yet written
        self._thread = threading.Thread(target=self._loop, name="kiln-pd-send", daemon=True)
        self._thread.start()

    def enqueue(self, dest: str, header: dict, buffers=()) -> None:
        n = sum(len(b) for b in buffers)
        t = time.perf_counter()
        with self._cv:
            self._raise()
            waited = False
            while self.stats.queued_bytes and self.stats.queued_bytes + n > self.max_bytes:
                waited = True
                self._cv.wait(1.0)
                self._raise()
            if waited:
                self.stats.blocked_seconds += time.perf_counter() - t
            self.stats.queued_bytes += n
            self._pending += 1
        self._q.put((dest, header, list(buffers), n))

    def _raise(self) -> None:
        """Raise a send error once (the frames it lost are logged), then carry on: a receiver that restarted is
        reached again on a fresh connection."""
        if self._error is not None:
            e, self._error = self._error, None
            raise RuntimeError(f"handoff sender failed: {e!r}") from e

    def flush(self, timeout: float = 600.0) -> None:
        """Wait until every queued frame is written (tests, shutdown)."""
        end = time.monotonic() + timeout
        with self._cv:
            while self._pending and self._error is None:
                if not self._cv.wait(max(0.0, end - time.monotonic())):
                    raise TimeoutError(f"{self._pending} handoff frames still queued after {timeout} s")
            self._raise()

    def _conn(self, dest: str) -> socket.socket:
        s = self._socks.get(dest)
        if s is not None and _peer_closed(s):
            # The receiver restarted and closed this connection: a frame written into it now would vanish (the kernel
            # accepts the first write after the peer's FIN and the RST only fails a later one), so start a fresh one.
            print(f"kiln pd: connection to {dest} was closed by the receiver; reconnecting", file=sys.stderr, flush=True)
            s.close()
            del self._socks[dest]
            s = None
        if s is None:
            host, port = dest.rsplit(":", 1)
            s = socket.create_connection((host, int(port)), timeout=60)
            s.settimeout(None)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            for opt in (socket.SO_SNDBUF,):
                try:
                    s.setsockopt(socket.SOL_SOCKET, opt, 16 << 20)
                except OSError:
                    pass
            if self.hello is not None:
                send_frame(s, *self.hello())
            self._socks[dest] = s
        return s

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            dest, header, buffers, n = item
            t = time.perf_counter()
            try:
                try:
                    sent = send_frame(self._conn(dest), header, buffers)
                except (BrokenPipeError, ConnectionResetError) as e:
                    # A connection the receiver closed (it restarted): one fresh connection, the whole frame again.
                    # The receiver never keeps a partial frame, so this cannot duplicate one.
                    print(f"kiln pd: connection to {dest} broke ({e!r}); reconnecting once for {header.get('kind')} "
                          f"{header.get('xfer')}", file=sys.stderr, flush=True)
                    old = self._socks.pop(dest, None)
                    if old is not None:
                        old.close()
                    sent = send_frame(self._conn(dest), header, buffers)
                with self._cv:
                    self.stats.frames += 1
                    self.stats.bytes += sent
            except BaseException as e:  # noqa: BLE001  kept and raised to the producer
                print(f"kiln pd: sending {header.get('kind')} {header.get('xfer')} to {dest} failed: {e!r}",
                      file=sys.stderr, flush=True)
                self._socks.pop(dest, None)
                with self._cv:
                    self._error = e
                    self.stats.errors += 1
            finally:
                with self._cv:
                    self.stats.send_seconds += time.perf_counter() - t
                    self.stats.queued_bytes -= n
                    self._pending -= 1
                    self._cv.notify_all()

    def close(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=30)
        for s in self._socks.values():
            try:
                s.close()
            except OSError:
                pass
        self._socks.clear()


# -- receiving -------------------------------------------------------------------------------------


@dataclass
class RecvStats:
    complete: int = 0
    parts: int = 0
    bytes: int = 0
    held_bytes: int = 0
    max_held_bytes: int = 0
    buffer_full_events: int = 0  # parts that waited for room in the receive buffer
    buffer_full_seconds: float = 0.0
    refused: int = 0  # handoffs with a layout signature other than this engine's
    nixl: int = 0  # handoffs whose rows the decode ranks read themselves (KILN_PD_TRANSPORT=nixl)
    nixl_bytes: int = 0  # the bytes those handoffs declared (held from arrival to release)
    hellos: int = 0
    transfer_seconds: list = field(default_factory=list)  # first frame -> complete, per handoff (last 4096)


@dataclass
class _Pending:
    first: float
    meta: dict | None = None
    parts: dict = field(default_factory=dict)  # name -> (file, array table, bytes)
    done: bool = False
    nixl_bytes: int = 0  # a NIXL handoff's declared bytes, held against the budget until release
    stages: dict = field(default_factory=dict)  # a pipeline's handoff: stage -> its meta (combine_stages)


def stage_part(stage: int, name: str) -> str:
    """The name a pipeline stage's part travels under (engine/pp.py: every stage hands off its own layers)."""
    return f"p{stage}.{name}"


def combine_stages(metas: dict[int, dict]) -> dict:
    """One handoff from the metas of a pipeline's stages (each stage hands off the caches and state rows of its own
    layers [lo, hi), meta["pp"]): the last stage's meta (its first token, logprobs, sampler state, params) with every
    stage's parts, the stages' layer ranges and, over NIXL, every stage's read plan (nixl["pp"], in stage order).
    The ranges must tile 0 .. the last stage's end with no gap and no overlap, and every stage must name the same split;
    otherwise the result carries an error (the decode engine refuses it, and releases every stage)."""
    order = sorted(metas)
    S = int(metas[order[0]]["pp"]["stages"])
    last = metas[S - 1]
    out = dict(last)
    ranges = [list(metas[t]["pp"]["layers"]) for t in order]
    splits = {tuple(map(tuple, metas[t]["pp"].get("ranges") or [])) for t in order}
    parts = []
    for t in order:
        parts += list(metas[t].get("parts") or [])
    out["parts"] = parts
    out["pp"] = {"stages": S, "ranges": ranges}
    errs = [metas[t]["error"] for t in order if metas[t].get("error")]
    if order != list(range(S)):
        errs.append(f"stages {order} of a {S}-stage pipeline")
    elif ranges[0][0] != 0 or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:])) or \
            any(a[0] >= a[1] for a in ranges):
        errs.append(f"stage layer ranges {ranges} do not tile the model's layers")
    elif len(splits) != 1:
        errs.append(f"the stages name different splits {sorted(splits)}")
    trans = {metas[t].get("transport", "host") for t in order if metas[t].get("done") is None}
    if len(trans) > 1:
        errs.append(f"the stages handed off over different transports {sorted(trans)}")
    if errs:
        out["error"] = "pipeline handoff: " + "; ".join(errs)
    if out.get("done") is None and trans == {"nixl"}:
        nx = [metas[t]["nixl"] for t in order]
        out["nixl"] = {"pp": nx, "bytes": sum(int(x["bytes"]) for x in nx),
                       "deadline": min(float(x["deadline"]) for x in nx)}
    out["handoff_time"] = min(float(metas[t].get("handoff_time", 0) or 0) for t in order) or last.get("handoff_time")
    return out


class Receiver:
    """Accepts handoff connections on `listen` ("host:port", port 0 = any), stores each part's payload as a
    file under store_dir and calls on_complete(meta) once a handoff's meta and every part it names have
    arrived (from the reader thread). meta["parts"] then maps part name -> {"file", "arrays"}; release(xfer)
    deletes the files and gives their bytes back to the buffer."""

    def __init__(self, listen: str, on_complete, budget_bytes: int, store_dir: str | None = None,
                 signature: str | None = None, in_memory: bool = False):
        """in_memory: keep each part's payload as a bytearray in meta["parts"][name]["buf"] instead of a file (a receiver
        whose consumer is this process alone, e.g. a pipeline stage's rank)."""
        host, port = listen.rsplit(":", 1)
        self.budget = budget_bytes
        self.in_memory = in_memory
        self.dir = store_dir or default_store_dir()
        os.makedirs(self.dir, exist_ok=True)
        self.signature = signature
        self.on_complete = on_complete
        self.stats = RecvStats()
        self._cv = threading.Condition()
        self._pending: dict[str, _Pending] = {}
        self._full_since: float | None = None
        self._srv = socket.create_server((host, int(port)), reuse_port=False, backlog=256)
        self.port = self._srv.getsockname()[1]
        self.host = host
        self._closed = False
        self._conns: list[socket.socket] = []
        self.hellos: dict[str, str] = {}  # prefill engine id -> the file holding its ranks' NIXL exports
        self.nixl_reply: dict[str, str] = {}  # xfer -> the prefill engine's release listener (nixl handoffs)
        self._thread = threading.Thread(target=self._accept, name="kiln-pd-accept", daemon=True)
        self._thread.start()

    def address(self, host: str | None = None) -> str:
        """The address peers connect to: `host` (this box's address as they see it) or the listen host."""
        if host and host.count(":") == 1:  # already "host:port"
            return host
        h = host or self.host
        if h in ("0.0.0.0", "", "::"):
            h = socket.gethostbyname(socket.gethostname())
        return f"{h}:{self.port}"

    def _accept(self) -> None:
        while not self._closed:
            try:
                c, _ = self._srv.accept()
            except OSError:
                return
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                c.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
            except OSError:
                pass
            self._conns.append(c)
            threading.Thread(target=self._read, args=(c,), name="kiln-pd-recv", daemon=True).start()

    def _room(self, n: int, what: str) -> None:
        """Wait until n more payload bytes fit the buffer (counted, logged once per episode)."""
        if n > self.budget:
            raise RuntimeError(f"handoff part {what} of {n} bytes exceeds the whole receive buffer "
                               f"({self.budget} bytes, pd_buffer_gb)")
        with self._cv:
            if self.stats.held_bytes + n <= self.budget:
                self.stats.held_bytes += n
                self.stats.max_held_bytes = max(self.stats.max_held_bytes, self.stats.held_bytes)
                return
            t = time.perf_counter()
            self.stats.buffer_full_events += 1
            if self._full_since is None:
                self._full_since = t
                print(f"kiln pd: receive buffer full ({self.stats.held_bytes} of {self.budget} bytes held); "
                      f"part {what} ({n} bytes) waits", file=sys.stderr, flush=True)
            while self.stats.held_bytes + n > self.budget and not self._closed:
                self._cv.wait(1.0)
            self.stats.buffer_full_seconds += time.perf_counter() - t
            self.stats.held_bytes += n
            self.stats.max_held_bytes = max(self.stats.max_held_bytes, self.stats.held_bytes)
            if self._full_since is not None:
                print(f"kiln pd: receive buffer has room again after {time.perf_counter() - self._full_since:.2f} s",
                      file=sys.stderr, flush=True)
                self._full_since = None

    def _unreserve(self, n: int) -> None:
        """Give back bytes _room reserved for a frame that never completed (its peer closed or reset inside it, or it was
        refused after the reservation): no pending handoff owns them, so no release() would."""
        with self._cv:
            self.stats.held_bytes -= n
            self._cv.notify_all()

    def _read(self, c: socket.socket) -> None:
        # Bytes _room reserved for the frame in hand that no pending handoff owns yet. They pass to the handoff at the
        # _add call (whose release() then gives them back) and are otherwise given back once, when this reader ends.
        reserved = 0
        try:
            while True:
                got = recv_header(c)
                if got is None:
                    return
                h, n = got
                xfer = h["xfer"]
                if h["kind"] == "meta":
                    raw = _recv_exact(c, n) if n else bytearray()
                    if raw is None:
                        raise ConnectionError("connection closed inside a meta frame")
                    meta = json.loads(bytes(raw)) if n else h["meta"]
                    nb = 0
                    if meta.get("transport") == "nixl" and meta.get("done") is None:
                        nb = int(meta["nixl"]["bytes"])
                        self._room(nb, f"{xfer}/nixl")
                        reserved = nb
                        eng = meta["nixl"]["engine"]
                        if eng not in self.hellos:
                            raise ConnectionError(f"a nixl handoff from engine {eng} before its hello frame")
                        meta["nixl"]["exports"] = self.hellos[eng]
                        with self._cv:
                            if meta.get("pp"):  # one release listener per stage (combine_stages)
                                self.nixl_reply.setdefault(xfer, []).append(meta["nixl"]["reply"])
                            else:
                                self.nixl_reply[xfer] = meta["nixl"]["reply"]
                    reserved = 0
                    self._add(xfer, meta=meta, nixl_bytes=nb)
                elif h["kind"] == "hello":
                    raw = _recv_exact(c, n)
                    if raw is None:
                        raise ConnectionError("connection closed inside a hello frame")
                    hello = json.loads(bytes(raw))
                    path = os.path.join(self.dir, f"nixl-{hello['engine']}.json")
                    with open(path + ".tmp", "wb") as f:
                        f.write(raw)
                    os.replace(path + ".tmp", path)
                    with self._cv:
                        self.hellos[hello["engine"]] = path
                        self.stats.hellos += 1
                elif h["kind"] == "part":
                    name = h["part"]
                    self._room(n, f"{xfer}/{name}")
                    reserved = n
                    path = os.path.join(self.dir, f"{xfer}.{name}")
                    if self.in_memory:
                        buf = _recv_exact(c, n) if n else bytearray()
                        if buf is None:
                            raise ConnectionError("connection closed inside a part frame")
                        path = buf
                    elif not _recv_file(c, n, path):
                        raise ConnectionError("connection closed inside a part frame")
                    reserved = 0
                    self._add(xfer, part=(name, path, h["arrays"], n))
                else:
                    raise ConnectionError(f"unknown handoff frame kind {h['kind']!r}")
        except BaseException as e:  # noqa: BLE001  a broken peer must not take the receiver down
            if not self._closed:
                print(f"kiln pd: handoff connection failed: {e!r}", file=sys.stderr, flush=True)
        finally:
            if reserved:
                self._unreserve(reserved)
            try:
                c.close()
            except OSError:
                pass

    def _add(self, xfer: str, meta: dict | None = None, part=None, nixl_bytes: int = 0) -> None:
        with self._cv:
            p = self._pending.get(xfer)
            if p is None:
                p = self._pending[xfer] = _Pending(time.perf_counter())
            # What the handoff now owns (its release() gives it back), recorded before anything below can raise.
            if nixl_bytes:  # every stage's declared bytes, all held until the one release
                p.nixl_bytes += nixl_bytes
                self.stats.nixl_bytes += nixl_bytes
            if part is not None:
                name, path, arrays, n = part
                p.parts[name] = (path, arrays, n)
                self.stats.parts += 1
                self.stats.bytes += n
            if meta is not None and meta.get("pp"):  # a pipeline stage's share: complete once every stage's is here
                p.stages[int(meta["pp"]["stage"])] = meta
                if len(p.stages) == int(meta["pp"]["stages"]):
                    p.meta = combine_stages(p.stages)
                    self.stats.nixl += p.meta.get("nixl") is not None
            elif meta is not None:
                p.meta = meta
                self.stats.nixl += bool(nixl_bytes)
            if p.done or p.meta is None or not set(p.meta.get("parts", ())) <= set(p.parts):
                return
            p.done = True
            meta = dict(p.meta)
            meta["parts"] = {k: ({"buf": v[0]} if self.in_memory else {"file": v[0]}) | {"arrays": v[1]}
                             for k, v in p.parts.items()}
            meta["recv_seconds"] = time.perf_counter() - p.first
            self.stats.complete += 1
            self.stats.transfer_seconds.append(meta["recv_seconds"])
            del self.stats.transfer_seconds[:-4096]
            sigs = {m.get("signature") for m in p.stages.values()} if p.stages else {meta.get("signature")}
            bad = sorted(str(x) for x in sigs if x not in (None, self.signature))
            if self.signature is not None and bad and meta.get("done") is None:
                self.stats.refused += 1
                meta["error"] = (f"handoff layout {', '.join(bad)} differs from this engine's {self.signature}: "
                                 "the prefill and decode engines must run the same model, attention TP, page size, "
                                 "KV dtype and cache shapes")
        self.on_complete(meta)

    def release(self, xfer: str) -> None:
        """Delete a handoff's files and give their bytes back, on a background thread: unlinking hundreds of MB of
        tmpfs pages took ~6% of the G1 decode engine's thread at 384 rows. The bytes count as held until deleted."""
        with self._cv:
            p = self._pending.pop(xfer, None)
        if p is None:
            return
        if getattr(self, "_reaper", None) is None:
            self._dead: queue.Queue = queue.Queue()
            self._reaper = threading.Thread(target=self._reap, name="kiln-pd-reap", daemon=True)
            self._reaper.start()
        with self._cv:
            self._reaping = getattr(self, "_reaping", 0) + 1
        self._dead.put(p)

    def _reap(self) -> None:
        while True:
            p = self._dead.get()
            if p is None:
                return
            n_all = p.nixl_bytes
            for path, _, n in p.parts.values():
                if isinstance(path, str):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
                n_all += n
            with self._cv:
                self.stats.held_bytes -= n_all
                self._reaping -= 1
                self._cv.notify_all()

    def pop_reply(self, xfer: str) -> str | list[str] | None:
        """The release address of a NIXL handoff (None for a host-path one; a pipeline's: one per stage), once."""
        with self._cv:
            return self.nixl_reply.pop(xfer, None)

    def drain(self, timeout: float = 30.0) -> None:
        """Wait until every released handoff's files are deleted (tests)."""
        end = time.monotonic() + timeout
        with self._cv:
            while getattr(self, "_reaping", 0) and time.monotonic() < end:
                self._cv.wait(0.05)

    def close(self) -> None:
        if getattr(self, "_reaper", None) is not None:
            self._dead.put(None)
            self._reaper.join(timeout=30)
        self._closed = True
        try:
            self._srv.shutdown(socket.SHUT_RDWR)  # wakes the accept() blocked in the accepting thread
        except OSError:
            pass
        try:
            self._srv.close()
        except OSError:
            pass
        self._thread.join(timeout=5)
        for c in self._conns:
            try:
                c.shutdown(socket.SHUT_RDWR)
                c.close()
            except OSError:
                pass
        with self._cv:
            self._cv.notify_all()
        shutil.rmtree(self.dir, ignore_errors=True)


# -- request meta ------------------------------------------------------------------------------------


def params_to_json(params) -> dict:
    d = asdict(params)
    if d.get("grammar") is not None:
        raise ValueError("a grammar-constrained request cannot be disaggregated (route it to a decode engine)")
    d["stop"], d["stop_token_ids"] = list(d["stop"]), list(d["stop_token_ids"])
    return d


def params_from_json(d: dict):
    from .request import SamplingParams

    d = dict(d)
    d["stop"], d["stop_token_ids"] = tuple(d["stop"]), tuple(d["stop_token_ids"])
    return SamplingParams(**d)


def unsupported(params) -> str | None:
    """Why a request cannot be handed off (None: it can). Each of these keeps host state that only the
    engine that started the request has."""
    if params.grammar is not None:
        return "grammar-constrained output"
    if params.thinking_token_budget is not None:
        return "a thinking token budget"
    return None


def read_parts(meta: dict, names: list[str]) -> dict[str, dict[str, torch.Tensor]]:
    """{part: {array name: host tensor}} of a complete handoff's parts (each file read once)."""
    out = {}
    for name in names:
        info = meta["parts"][name]
        if "buf" in info:  # an in-memory receiver's part (Receiver in_memory)
            buf = info["buf"]
        elif os.path.getsize(info["file"]) == 0:
            buf = bytearray()
        else:
            # Mapped, not read: a consumer that takes some rows (a CP decode rank, ModelRunner._pd_assemble) touches only
            # their pages. Copy-on-write, because torch.frombuffer wants a writable buffer; nothing writes it. The
            # mapping outlives the file's deletion at release (Receiver._reap) for as long as a tensor holds it.
            buf = np.memmap(info["file"], dtype=np.uint8, mode="c")
        arrays, off = {}, 0
        for a_name, dt, shape, n in info["arrays"]:
            arrays[a_name] = from_bytes(buf, off, _dtype(dt), shape)
            off += n
        out[name] = arrays
    return out


def pack_arrays(named: list[tuple[str, torch.Tensor]]) -> tuple[list, list]:
    """(array table, buffers) of host tensors for a part frame."""
    table, bufs = [], []
    for name, t in named:
        b = to_bytes(t)
        table.append([name, _dtype_name(t.dtype), list(t.shape), len(b)])
        bufs.append(b)
    return table, bufs
