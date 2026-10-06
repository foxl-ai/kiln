"""What a prefill -> decode KV handoff costs through the host on one NeuronCore: a sequence's KV pages gathered out of a
paged pool on the device, copied to the host, and written into another pool's pages on the device.

Why: prefill / decode disaggregation hands every request's DSA cache and KDA state from a prefill engine to a decode
engine. On one box the two engines are separate processes on disjoint cores and share no device collective, so the
first path to measure is device -> host -> device, per rank, in parallel over the ranks of an engine.

    python tools/probe_kv_handoff.py [--mb 16 64 256 1024] [--page-kb 16] [--iters 5]

Per size: the contiguous device -> host copy (.cpu()), host -> device (.to(device)), and the paged forms: the pages of
one sequence (a random permutation of the pool's pages, page_kb each) gathered on the device then read back, and a
host buffer scattered into another pool's pages (index_copy_ on the device after .to). GB/s per NeuronCore.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _t(fn, iters: int) -> float:
    fn()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, nargs="+", default=[16, 64, 256, 1024])
    ap.add_argument("--page-kb", type=int, default=16)
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()
    from kiln import platform

    platform.configure_runtime_env()
    import libtorch_neuronx_lite  # noqa: F401

    from kiln.engine.model_runner import canonical_neuron_backend

    dev = torch.device("neuron:0")
    print("kiln platform", platform.target(), flush=True)
    W = a.page_kb * 1024 // 2  # bf16 values per page row
    gather = torch.compile(lambda pool, idx: pool.index_select(0, idx), backend=canonical_neuron_backend(),
                           fullgraph=True, dynamic=False)

    def scatter_fn(pool, idx, rows):
        pool.index_copy_(0, idx, rows)
        return idx.sum()

    scatter = torch.compile(scatter_fn, backend=canonical_neuron_backend(), fullgraph=True, dynamic=False)
    for mb in a.mb:
        n = mb * 1024 // a.page_kb  # pages of the sequence
        x = torch.randn(n, W).to(torch.bfloat16)
        d = x.to(dev)
        d.cpu()
        h2d = _t(lambda: x.to(dev), a.iters)  # host -> device (LNL's copy returns when the bytes are on the device)
        d2h = _t(lambda: d.cpu(), a.iters)
        pool = torch.randn(2 * n, W).to(torch.bfloat16).to(dev)
        idx = torch.randperm(2 * n)[:n].to(dev)
        pg = _t(lambda: gather(pool, idx).cpu(), a.iters)
        pool2 = torch.zeros(2 * n, W, dtype=torch.bfloat16).to(dev)
        ps = _t(lambda: scatter(pool2, idx, x.to(dev)).cpu(), a.iters)
        gb = mb / 1024
        print(f"{mb:5d} MB in {n} pages of {a.page_kb} KB: device->host {d2h * 1e3:8.2f} ms ({gb / d2h:5.2f} GB/s), "
              f"host->device {h2d * 1e3:8.2f} ms ({gb / h2d:5.2f} GB/s), paged gather + read {pg * 1e3:8.2f} ms "
              f"({gb / pg:5.2f} GB/s), write + paged scatter {ps * 1e3:8.2f} ms ({gb / ps:5.2f} GB/s)", flush=True)


if __name__ == "__main__":
    main()
