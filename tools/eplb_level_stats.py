"""Per-level expert-load balance from an engine's recorded counts (serve_sweep --eplb-rebalance --eplb-dump):
for each level's all-reduced counts [layer][E], the busiest rank's load under (a) the contiguous placement,
(b) the copies the level ran with, both as max / mean pairs (aggregate over the level) and as the EP kernel's
lane-model time of the level's AVERAGE batch (0.15 ms + 0.78 us per executed lane, eplb_sim.prefill_passes),
summed over the layers. The per-batch variance is not in a level's totals, so the lane-model number is a lower
bound of what the kernel paid; tools/probe_eplb_kernel.py times real batches.

    python tools/eplb_level_stats.py --level L1.pt:init.pt --level L2.pt:L1.pt [--rows 4096] [--ranks 32] [--slots 1]

Each --level is COUNTS:PLACEMENT, the placement given by the counts file it was computed from (eplb.replicas).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> None:
    from eplb_sim import prefill_passes

    from kiln.models import eplb

    ap = argparse.ArgumentParser()
    ap.add_argument("--level", action="append", required=True)
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--ranks", type=int, default=32)
    ap.add_argument("--slots", type=int, default=1)
    a = ap.parse_args()
    E, R, s = 288, a.ranks, a.slots
    El = E // R
    for spec in a.level:
        cpath, ppath = spec.split(":")
        C, P = eplb.load_file(cpath, E), eplb.load_file(ppath, E)
        tot = {"contiguous": [0.0, 0.0], "copies": [0.0, 0.0]}
        for L in sorted(C):
            c = C[L]
            batches = float(c.sum()) / (a.rows * a.k)  # prefill calls of `rows` rows the level ran
            avg = c / batches
            extra = eplb.replicas(P[L], R, s)
            n = torch.ones(E)
            for e in extra:
                n[e] += 1
            ranks_c = [avg[r * El:(r + 1) * El].tolist() for r in range(R)]
            ranks_e = [(avg[r * El:(r + 1) * El] / n[r * El:(r + 1) * El]).tolist()
                       + [float(avg[e] / n[e]) for e in extra[r * s:(r + 1) * s]] for r in range(R)]
            for name, rk in (("contiguous", ranks_c), ("copies", ranks_e)):
                loads = [sum(x) for x in rk]
                tot[name][0] += max(loads) / (sum(loads) / R)
                lanes = [256 * st + 512 * ov for st, ov in (prefill_passes([int(round(v)) for v in x]) for x in rk)]
                tot[name][1] += 0.15 + 0.78e-3 * max(lanes)
        nl = len(C)
        print(f"{cpath} (placement from {ppath}): {batches:.0f} prefill calls of {a.rows} rows; busiest/mean pairs "
              f"contiguous {tot['contiguous'][0] / nl:.2f}, with the copies {tot['copies'][0] / nl:.2f}; lane-model "
              f"busiest-rank kernel of the average batch summed over {nl} layers: contiguous "
              f"{tot['contiguous'][1]:.1f} ms, with the copies {tot['copies'][1]:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
