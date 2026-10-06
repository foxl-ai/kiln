"""Real slots of a moe_dedupe call at decode-sized batches under real routing: rows drawn from tools/ep_routing.py's
saved routing (one token from each of T random positions across its sequences, as a decode step's rows are one token of
each of T sequences), per MoE layer the slots the routing fills (sum over experts of ceil(pairs / lanes)) against the
static count n_slots, and the blocks of 128 lanes they span.

    python tools/dedupe_slot_stats.py ep_routing.pt [--rows 128 192 256] [--trials 20] [--kinds text random]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("routing")
    ap.add_argument("--rows", type=int, nargs="+", default=[128, 192, 256])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--kinds", nargs="+", default=["text", "random"])
    a = ap.parse_args()
    from kiln.kernels import moe_dedupe as mdd

    d = torch.load(a.routing)
    E, names, rec = d["experts"], d["names"], d["topi"]
    g = torch.Generator().manual_seed(0)
    for kind in a.kinds:
        si = [i for i, n in enumerate(names) if n.startswith(kind)]
        if not si:
            continue
        for T in a.rows:
            L = mdd.default_lanes(T)
            S, BL = mdd.n_slots(T, 8, E, L)
            spb = BL // L
            reals = []
            for layer in sorted(rec):
                per = rec[layer]  # per sequence [tokens, k]
                pool = torch.cat([per[i] for i in si if i < len(per)])
                for _ in range(a.trials):
                    rows = pool[torch.randint(0, pool.shape[0], (T,), generator=g)]
                    cnt = torch.bincount(rows.reshape(-1).long(), minlength=E)
                    reals.append(int(((cnt + L - 1) // L).sum()))
            r = torch.tensor(reals, dtype=torch.float)
            print(f"{kind:6s} T={T:3d} lanes {L}: static {S} slots ({S // spb} blocks), real mean {r.mean():.0f} p90 "
                  f"{r.quantile(0.9):.0f} max {r.max():.0f} slots -> {r.mean() / spb:.1f} / {r.quantile(0.9) / spb:.1f} "
                  f"blocks; always-real head {-(-(-(-T * 8 // L)) // spb)} blocks", flush=True)


if __name__ == "__main__":
    main()
