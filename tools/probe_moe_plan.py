"""Device time of the pieces of kiln/kernels/moe_dedupe.plan (the routing -> slot table computed
in the graph before the dedupe kernel), one graph per piece, on one NeuronCore.

    python tools/probe_moe_plan.py [--batch 32] [--lanes 4]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[32])
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--experts", type=int, default=256)
    args = ap.parse_args()
    import profile_layer as pl

    pl.setup_device()
    from kiln.kernels import moe_dedupe as mdd

    E, K, L = args.experts, 8, args.lanes
    i32 = torch.int32
    g = torch.Generator().manual_seed(1)
    pl.timed("null graph", lambda h: h + 1, (torch.zeros(4, 64).to(pl.DEV),))
    for T in args.batch:
        topi = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)]).to(pl.DEV)
        topv = (torch.rand(T, K, generator=g) + 0.1).bfloat16().to(pl.DEV)
        N = T * K
        S = mdd.n_slots(T, K, E, L)
        SL = S * L

        def onehot(i):
            e = i.reshape(N).to(i32)
            return (e.unsqueeze(1) == torch.arange(E, device=i.device, dtype=i32).unsqueeze(0)).to(i32).sum(0, dtype=i32)

        def rank(i):
            e = i.reshape(N).to(i32)
            ar = torch.arange(N, device=i.device, dtype=i32)
            return ((e.unsqueeze(1) == e.unsqueeze(0)) & (ar.unsqueeze(0) < ar.unsqueeze(1))).to(i32).sum(1, dtype=i32)

        def hit(i):
            lane = i.reshape(N).to(i32) * 3
            h = lane.unsqueeze(1) == torch.arange(SL, device=i.device, dtype=i32).unsqueeze(0)
            return h.to(i32).view(T, K, SL).sum(1, dtype=i32).to(torch.bfloat16)

        def wl(v, i):
            lane = i.reshape(N).to(i32) * 3
            h = lane.unsqueeze(1) == torch.arange(SL, device=i.device, dtype=i32).unsqueeze(0)
            return (h.to(torch.float32) * v.reshape(N, 1).to(torch.float32)).sum(0)

        print(f"T={T} N={N} S={S} SL={SL}", flush=True)
        pl.timed("  [N, E] one-hot + counts", onehot, (topi,))
        pl.timed("  [N, N] rank within expert", rank, (topi,))
        pl.timed("  [N, SL] lane one-hot -> G", hit, (topi,))
        pl.timed("  [N, SL] lane weights", wl, (topv, topi))
        pl.timed("  whole plan", lambda v, i: mdd.plan(v, i, E, L), (topv, topi))


if __name__ == "__main__":
    main()
