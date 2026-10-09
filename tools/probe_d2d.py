"""NeuronCore HBM -> NeuronCore HBM over EFA with NIXL, on every core of two boxes at once, against the host path
(device -> host, TCP, host -> device) on the same cores and sizes.

    # box A (holds the data, like a prefill box):
    python tools/probe_d2d.py --role target --port 7600 --cores 0-31 --mb 28 49 784
    # box B (pulls it, like a decode box):
    python tools/probe_d2d.py --role initiator --peer <A's private ip>:7600 --cores 0-31 --mb 28 49 784 \
        --descs 1 256 --verify --host-path

Why NEURON_RT_MAP_HBM=1: libtorch_neuronx_lite's allocator (libtorchneuron.so, neuron::NeuronAllocator::allocate,
csrc/neuron_op/storage.cpp) puts nrt_tensor_get_va() into a tensor's data pointer only when that variable is "true" or
a positive integer, read once per process; otherwise data_ptr() is 0 and NIXL cannot register the tensor ("Failed to
retrieve placement for VA: (nil)"). vllm-neuron 0.24 sets it in neuron_worker.py before the runtime starts ("required
for RDMA in disaggregated inference"). This tool sets it for its children.

One child process per NeuronCore (NEURON_RT_VISIBLE_CORES=<core>), each with its own NIXL agent (LIBFABRIC) and one
device buffer of the largest size registered as VRAM. Target children fill their buffer with a per-core pattern and
publish their agent metadata and buffer address; initiator child i reads target child i's buffer into its own (RDMA
READ, the decode side pulling), all children of a level at once behind a barrier, --iters times per (size, descriptor
count). A level's aggregate GB/s is the bytes of all children over the slowest child's time. --descs K splits each
transfer into K equal descriptors (a handoff is one descriptor per cache per slot run). --verify reads the received
buffer back to the host and compares it with the target's pattern byte for byte. Each child also reports the EFA device
its buffer is attached to (nrt_get_attached_efa_bdf, nrt/nrt.h), which is the rail NIXL uses for that memory.

--desc-kb S1 S2 ... splits each transfer into descriptors of S KB (instead of --descs counts), and --stride F spaces
the REMOTE blocks F apart (the target buffer is F times larger): block j comes from remote offset j * S * F. That is a
context-parallel decode rank's read of a non-CP prefill engine's rows (its DSA pools are every F-th run of kp
positions), whose descriptors NIXL cannot merge. --gather (target side) times the device copy that would stage such
blocks contiguously first: every F-th S-byte block of an F x n buffer into an n-byte one, on each target core.

--host-path measures, on the initiator's cores at once: device -> host (.cpu() of the buffer's first n bytes), host ->
device (copy_ into it), and n bytes over one TCP connection per child from the target child of the same index (the
TCP leg of engine/disagg.py). --mem dram uses host buffers on both sides (the network alone).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import socket
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _send(s: socket.socket, obj, blob: bytes = b"") -> None:
    h = json.dumps(obj).encode()
    s.sendall(struct.pack("!II", len(h), len(blob)) + h + blob)


def _exact(s: socket.socket, n: int) -> bytes:
    b = bytearray(n)
    v, got = memoryview(b), 0
    while got < n:
        k = s.recv_into(v[got:], n - got)
        if not k:
            raise ConnectionError("peer closed")
        got += k
    return bytes(b)


def _recv(s: socket.socket):
    hn, bn = struct.unpack("!II", _exact(s, 8))
    return json.loads(_exact(s, hn)), _exact(s, bn)


def parse_cores(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def efa_bdf(va: int) -> str:
    """The BDF of the EFA device attached to the Neuron device holding va (nrt_get_attached_efa_bdf, nrt/nrt.h)."""
    lib = ctypes.CDLL("libnrt.so.1")
    fn = lib.nrt_get_attached_efa_bdf
    fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t)]
    fn.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(64)
    n = ctypes.c_size_t(64)
    st = fn(ctypes.c_void_p(va), buf, ctypes.byref(n))
    return buf.value.decode() if st == 0 else f"status {st}"


def pattern(n: int, core: int):
    """n bytes of a period-251 sequence that differs per core (no power-of-two period, so a misplaced block shows)."""
    import torch

    base = (torch.arange(251, dtype=torch.int32) * 7 + core * 131).remainder(251).to(torch.uint8)
    return base.repeat(n // 251 + 1)[:n].clone()


def child(role: str, core: int, idx: int, a, conn, barrier) -> None:
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
    if a.mem == "vram":
        os.environ["NEURON_RT_MAP_HBM"] = "1"
    import torch

    if a.mem == "vram":
        import libtorch_neuronx_lite  # noqa: F401

        dev = torch.device("neuron:0")
    else:
        dev = torch.device("cpu")
    from nixl._api import nixl_agent, nixl_agent_config

    n_max = max(a.mb) << 20
    F = a.stride if role == "target" else 1  # the target holds F x the bytes (strided reads)
    host = pattern(n_max * F, core) if role == "target" else torch.zeros(n_max, dtype=torch.uint8)
    buf = host.to(dev) if a.mem == "vram" else host.clone()
    if a.mem == "vram":
        torch.zeros(1, device=dev).cpu()  # the copy above has executed
    va = buf.data_ptr()
    if va == 0:
        raise SystemExit(f"core {core}: data_ptr() is 0 (NEURON_RT_MAP_HBM={os.environ.get('NEURON_RT_MAP_HBM')})")
    mt = "VRAM" if a.mem == "vram" else "DRAM"
    agent = nixl_agent(f"kiln-d2d-{role}-{core}-{os.getpid()}", nixl_agent_config(backends=["LIBFABRIC"]))
    t0 = time.perf_counter()
    agent.register_memory(agent.get_reg_descs([(va, n_max * F, 0, "")], mt), backends=["LIBFABRIC"])
    reg_s = time.perf_counter() - t0
    bdf = efa_bdf(va) if a.mem == "vram" else "-"
    info = {"core": core, "va": va, "n": n_max * F, "bdf": bdf, "reg_s": round(reg_s, 4)}
    if role == "target" and a.gather and a.stride > 1:  # the staging copy a prefill rank would make instead
        # LNL's eager copy_ refuses a strided device view ("Expected self.is_contiguous()"), so the gather is a
        # compiled graph (one per shape), as the engine would run it; int32 lanes, the first compile excluded.
        info["gather"] = []
        kb = min(a.desc_kb or [4])
        src32 = buf.view(torch.int32)
        for mb in a.mb:
            n = mb << 20
            piece = (kb << 10) // 4
            k = n // (kb << 10)
            fn = torch.compile(lambda x, k=k, piece=piece: x[: k * F * piece].view(k, F, piece)[:, 0, :].contiguous(),
                               backend="neuron_libtorch", fullgraph=True)
            ts = []
            for i in range(a.iters + 1):
                t = time.perf_counter()
                out = fn(src32)
                _ = out[:1].cpu()
                if i:
                    ts.append(time.perf_counter() - t)
                del out
            ts.sort()
            info["gather"].append({"mb": mb, "desc_kb": kb, "p50_ms": round(ts[len(ts) // 2] * 1e3, 3)})
    if role == "target":
        conn.send((info, agent.get_agent_metadata()))
        srv = None
        if a.host_path:  # serve the TCP leg: n bytes of the pattern per request
            srv = socket.create_server(("0.0.0.0", a.port + 1 + idx))
            payload = memoryview(host.numpy())
            c, _ = srv.accept()
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                try:
                    (n,) = struct.unpack("!Q", _exact(c, 8))
                except ConnectionError:
                    break
                c.sendall(payload[:n])
        conn.recv()  # stop
        return
    peer, meta = conn.recv()
    remote = agent.add_remote_agent(meta)
    results = []
    local_dev = 0
    F = int(peer.get("stride", 1))
    splits = [("kb", kb) for kb in a.desc_kb] if a.desc_kb else [("n", k) for k in a.descs]
    for mb in a.mb:
        n = mb << 20
        for how, v in splits:
            piece = (v << 10) if how == "kb" else n // v
            k = n // piece
            loc = [(va + j * piece, piece, local_dev) for j in range(k)]
            rem = [(peer["va"] + j * piece * F, piece, 0) for j in range(k)]
            times = []
            for i in range(a.iters + 1):
                barrier.wait()
                t = time.perf_counter()  # descriptor lists and the handle are per handoff, so they are timed too
                ld, rd = agent.get_xfer_descs(loc, mt), agent.get_xfer_descs(rem, mt)
                h = agent.initialize_xfer("READ", ld, rd, remote, b"")
                st = agent.transfer(h)
                while st not in ("DONE", "ERR"):
                    st = agent.check_xfer_state(h)
                dt = time.perf_counter() - t
                agent.release_xfer_handle(h)
                if st == "ERR":
                    raise RuntimeError(f"core {core}: transfer of {mb} MB in {k} descriptors failed")
                if i:
                    times.append(dt)
            ok = None
            if a.verify:
                got = buf[: piece * k].cpu() if a.mem == "vram" else buf[: piece * k]
                if F == 1:
                    ok = bool(torch.equal(got, pattern(n_max, peer["core"])[: piece * k]))
                else:  # block j is remote bytes j * piece * F ..: check the first, a middle and the last block
                    want = pattern(piece * F * k, peer["core"]).view(k, F, piece)[:, 0, :]
                    ok = all(bool(torch.equal(got.view(k, piece)[j], want[j])) for j in sorted({0, k // 2, k - 1}))
                if a.mem == "vram":
                    buf.zero_()  # the next level must not pass on what this one left
                    torch.zeros(1, device=dev).cpu()
                else:
                    buf.zero_()
            results.append({"kind": "nixl", "mb": mb, "descs": k, "desc_kb": round(piece / 1024, 2), "stride": F,
                            "times": times, "verified": ok})
            if idx == 0:  # progress, one line per level from the first child
                print(json.dumps({"progress": "nixl", "core": core, "mb": mb, "descs": k, "stride": F,
                                  "times_ms": [round(t * 1e3, 2) for t in times], "verified": ok}), flush=True)
    if a.host_path:
        s = socket.create_connection((a.peer.rsplit(":", 1)[0], a.port + 1 + idx), timeout=600)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for opt in (socket.SO_RCVBUF,):
            s.setsockopt(socket.SOL_SOCKET, opt, 16 << 20)
        rx = bytearray(n_max)
        for mb in a.mb:
            n = mb << 20
            src = host[:n].clone()
            legs: dict[str, list[float]] = {"d2h": [], "tcp": [], "h2d": []}
            for i in range(a.iters + 1):
                barrier.wait()
                t = time.perf_counter()
                if a.mem == "vram":
                    _ = buf[:n].cpu()
                d2h = time.perf_counter() - t
                barrier.wait()
                t = time.perf_counter()
                s.sendall(struct.pack("!Q", n))
                v, got = memoryview(rx), 0
                while got < n:
                    got += s.recv_into(v[got:n], n - got)
                tcp = time.perf_counter() - t
                barrier.wait()
                t = time.perf_counter()
                if a.mem == "vram":
                    buf[:n].copy_(src)
                    torch.zeros(1, device=dev).cpu()
                h2d = time.perf_counter() - t
                if i:
                    legs["d2h"].append(d2h)
                    legs["tcp"].append(tcp)
                    legs["h2d"].append(h2d)
            for leg, ts in legs.items():
                results.append({"kind": leg, "mb": mb, "descs": 1, "times": ts, "verified": None})
        s.close()
    conn.send((info, results))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["target", "initiator"])
    ap.add_argument("--port", type=int, default=7600)
    ap.add_argument("--peer", default=None, help="initiator: the target's host:port")
    ap.add_argument("--cores", default="0")
    ap.add_argument("--mb", type=int, nargs="+", default=[28, 49, 784])
    ap.add_argument("--descs", type=int, nargs="+", default=[1])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--mem", default="vram", choices=["vram", "dram"])
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--host-path", action="store_true")
    ap.add_argument("--desc-kb", type=int, nargs="+", default=None)
    ap.add_argument("--stride", type=int, default=1, help="target: hold F x the bytes; reads take every F-th block")
    ap.add_argument("--gather", action="store_true", help="target: time the device copy that stages strided blocks")
    ap.add_argument("--out", default=None, help="initiator: write every child's raw times as JSON")
    a = ap.parse_args()
    cores = parse_cores(a.cores)
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(cores))
    pipes, procs = [], []
    for i, c in enumerate(cores):
        mine, theirs = ctx.Pipe()
        p = ctx.Process(target=child, args=(a.role, c, i, a, theirs, barrier), daemon=True)
        p.start()
        pipes.append(mine)
        procs.append(p)
    if a.role == "target":
        infos = [p.recv() for p in pipes]
        for info, _ in infos:
            print(json.dumps(info), flush=True)
        srv = socket.create_server(("0.0.0.0", a.port))
        c, _ = srv.accept()
        blob = b"".join(m for _, m in infos)
        _send(c, {"children": [{**info, "stride": a.stride} for info, _ in infos],
                  "meta_len": [len(m) for _, m in infos], "host_path": a.host_path}, blob)
        msg, _ = _recv(c)
        print("initiator reported:", json.dumps(msg)[:4000], flush=True)
        for p in pipes:
            p.send("stop")
        for p in procs:
            p.join(timeout=60)
        return
    host, port = a.peer.rsplit(":", 1)
    s = socket.create_connection((host, int(port)), timeout=900)
    hdr, blob = _recv(s)
    if len(hdr["children"]) != len(cores):
        raise SystemExit(f"the target runs {len(hdr['children'])} children, this side {len(cores)}")
    off = 0
    for p, info, n in zip(pipes, hdr["children"], hdr["meta_len"]):
        p.send((info, blob[off : off + n]))
        off += n
    got = [p.recv() for p in pipes]
    for p in procs:
        p.join(timeout=60)
    rows = {}
    for (info, results), tinfo in zip(got, hdr["children"]):
        print(json.dumps({"core": info["core"], "bdf": info["bdf"], "peer_core": tinfo["core"],
                          "peer_bdf": tinfo["bdf"], "reg_s": info["reg_s"], "peer_gather": tinfo.get("gather")}),
              flush=True)
        for r in results:
            rows.setdefault((r["kind"], r["mb"], r["descs"], r.get("desc_kb"), r.get("stride", 1)), []).append(r)
    summary = []
    for (kind, mb, k, dkb, F), rs in sorted(rows.items(), key=lambda kv: (kv[0][0] != "nixl", kv[0][1], kv[0][2], kv[0][0])):
        it = len(rs[0]["times"])
        wall = sorted(max(r["times"][i] for r in rs) for i in range(it))  # each iteration: the slowest child
        per = sorted(t for r in rs for t in r["times"])
        p50w, p50 = wall[len(wall) // 2], per[len(per) // 2]
        row = {"kind": kind, "mb_per_core": mb, "descs": k, "desc_kb": dkb, "stride": F, "cores": len(rs),
               "p50_core_ms": round(p50 * 1e3, 3),
               "core_gb_s": round(mb / 1024 / p50, 2), "p50_level_ms": round(p50w * 1e3, 3),
               "aggregate_gb_s": round(mb * len(rs) / 1024 / p50w, 2),
               "verified": None if rs[0]["verified"] is None else all(r["verified"] for r in rs)}
        summary.append(row)
        print(json.dumps(row), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "children": [g[0] for g in got], "peers": hdr["children"],
                       "results": [g[1] for g in got], "summary": summary}, f)
    _send(s, {"summary": summary})
    s.close()


if __name__ == "__main__":
    main()
