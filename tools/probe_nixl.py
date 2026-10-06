"""A device-to-device handoff over EFA with NIXL (the DLAMI venv's nixl 1.3.2, LIBFABRIC backend), against the host
path (engine/disagg.py: device -> host -> TCP -> host -> device).

Why: the disaggregated deployment's handoff copies each rank's share of a request to the host and back (measured on
trn1 per NeuronCore: device -> host ~3 GB/s, host -> device ~9-12 GB/s). vllm-neuron 0.24's disaggregated inference
reads KV blocks straight out of the prefill server's device memory with a NIXL RDMA READ over EFA
(vllm_neuron/vllm/kv_connector/neuron_nixl_connector.py registers the KV caches by data_ptr as "VRAM"). This probe
asks whether the same works for a buffer Kiln allocates on a NeuronCore through libtorch_neuronx_lite, and how fast.

    # box A (the side that holds the data, like a prefill box):
    python tools/probe_nixl.py --role target --port 7510 --mem vram --mb 64 1024
    # box B (the side that pulls it, like a decode box):
    python tools/probe_nixl.py --role initiator --peer <A's private ip>:7510 --mem vram --mb 64 1024 --op read

The two sides exchange NIXL agent metadata and their buffers' descriptors over a plain TCP socket (JSON + bytes),
then the initiator runs --iters transfers of each size and prints GB/s and latency; with --verify the target fills its
buffer with a pattern and the initiator checks what arrived (a device buffer is read back to the host for that).
--mem dram uses host buffers (a baseline for the network itself). One process per NeuronCore pair (--core).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _send(s: socket.socket, obj: dict, blob: bytes = b"") -> None:
    h = json.dumps(obj).encode()
    s.sendall(struct.pack("!II", len(h), len(blob)) + h + blob)


def _recv(s: socket.socket) -> tuple[dict, bytes]:
    def exact(n):
        b = bytearray()
        while len(b) < n:
            k = s.recv(n - len(b))
            if not k:
                raise ConnectionError("peer closed")
            b += k
        return bytes(b)

    hn, bn = struct.unpack("!II", exact(8))
    return json.loads(exact(hn)), exact(bn)


def buffers(mem: str, sizes_mb: list[int], core: int, fill: bool):
    import torch

    if mem == "vram":
        os.environ.setdefault("NEURON_RT_VISIBLE_CORES", str(core))
        import libtorch_neuronx_lite  # noqa: F401

        dev = torch.device("neuron:0")
    else:
        dev = torch.device("cpu")
    out = {}
    for mb in sizes_mb:
        n = mb << 20
        t = torch.zeros(n, dtype=torch.uint8)
        if fill:  # the target's pattern, which the initiator checks after a read (its own buffers start at zero)
            t[:] = torch.arange(n, dtype=torch.int64).remainder(251).to(torch.uint8)
        out[mb] = t.to(dev) if mem == "vram" else t
    if mem == "vram":
        torch.zeros(1, device=dev).cpu()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["target", "initiator"])
    ap.add_argument("--port", type=int, default=7510)
    ap.add_argument("--peer", default=None, help="initiator: the target's host:port")
    ap.add_argument("--mem", default="vram", choices=["vram", "dram"])
    ap.add_argument("--mb", type=int, nargs="+", default=[64, 1024])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--op", default="read", choices=["read", "write"])
    ap.add_argument("--core", type=int, default=0)
    ap.add_argument("--backend", default="LIBFABRIC")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    from nixl._api import nixl_agent, nixl_agent_config

    bufs = buffers(a.mem, a.mb, a.core, fill=a.role == "target")
    agent = nixl_agent(f"kiln-{a.role}-{os.getpid()}", nixl_agent_config(backends=[a.backend]))
    print(f"nixl plugins {agent.get_plugin_list()}, {a.backend} mem types {agent.get_backend_mem_types(a.backend)}",
          flush=True)
    mt = "VRAM" if a.mem == "vram" else "DRAM"
    regs = {}
    for mb, t in bufs.items():
        dev = t.get_device() if a.mem == "vram" else 0
        if dev < 0:
            dev = 0
        descs = agent.get_reg_descs([(t.data_ptr(), t.numel(), dev, "")], mt)
        t0 = time.perf_counter()
        regs[mb] = agent.register_memory(descs, backends=[a.backend])
        print(f"registered {mb} MB {mt} at 0x{t.data_ptr():x} dev {dev} in {time.perf_counter() - t0:.3f} s", flush=True)
    meta = agent.get_agent_metadata()
    mine = {str(mb): [t.data_ptr(), t.numel(), max(t.get_device(), 0) if a.mem == "vram" else 0]
            for mb, t in bufs.items()}
    if a.role == "target":
        srv = socket.create_server(("0.0.0.0", a.port))
        c, _ = srv.accept()
        _send(c, {"bufs": mine, "mem": mt}, meta)
        msg, _ = _recv(c)  # the initiator's report when it is done
        print("initiator reported:", json.dumps(msg), flush=True)
        c.close()
        return
    host, port = a.peer.rsplit(":", 1)
    s = socket.create_connection((host, int(port)), timeout=600)
    peer, peer_meta = _recv(s)
    remote = agent.add_remote_agent(peer_meta)
    results = []
    for mb, t in bufs.items():
        raddr, rlen, rdev = peer["bufs"][str(mb)]
        local = agent.get_xfer_descs([(t.data_ptr(), t.numel(), max(t.get_device(), 0) if a.mem == "vram" else 0)], mt)
        rem = agent.get_xfer_descs([(raddr, rlen, rdev)], peer["mem"])
        times = []
        for i in range(a.iters + 1):
            h = agent.initialize_xfer(a.op.upper(), local, rem, remote, b"")
            t0 = time.perf_counter()
            st = agent.transfer(h)
            while st not in ("DONE", "ERR"):
                st = agent.check_xfer_state(h)
            dt = time.perf_counter() - t0
            agent.release_xfer_handle(h)
            if st == "ERR":
                raise RuntimeError(f"transfer of {mb} MB failed")
            if i:
                times.append(dt)
        times.sort()
        p50 = times[len(times) // 2]
        ok = None
        if a.verify and a.op == "read":
            import torch

            got = t.cpu() if a.mem == "vram" else t
            want = torch.arange(got.numel(), dtype=torch.int64).remainder(251).to(torch.uint8)
            ok = bool(torch.equal(got, want))
        r = {"mb": mb, "op": a.op, "mem": mt, "p50_s": round(p50, 5), "gb_s": round(mb / 1024 / p50, 2), "verified": ok}
        results.append(r)
        print(json.dumps(r), flush=True)
    _send(s, {"results": results})
    s.close()


if __name__ == "__main__":
    main()
