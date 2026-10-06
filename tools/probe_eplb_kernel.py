"""The busiest rank's expert-parallel prefill kernel (kernels/moe_ep.py) per MoE layer, on REAL GLM-5.3-Flash routing,
under the contiguous placement and under EPLB with redundant slots (models/eplb.py), on one NeuronCore:

    python tools/probe_eplb_kernel.py --routing-file ep_routing.pt --init eplb-init-random1.pt \
        [--sequences random0:0 random0:1024 random0:2048 random0:3072] [--layers 3 4 ... 44] [--slots 1] [--iters 5]

For each layer the 4096-row batch is the named sequences' token ranges (tools/ep_routing.py's saved top-k, the
routers on real hidden states); the placement of the copies comes from --init (a KILN_EPLB_INIT file, e.g. the
counts of OTHER sequences: out of sample) through eplb.replicas, the ids through eplb.remap (row classes as the
sequence-parallel routing gives them: row t of the gathered batch is class t mod RMAX). Each rank's pairs decide its
kernel time (the other ranks' pairs do not enter its call), so the busiest rank is found by the kernel's own cost
model (0.15 ms + 0.78 us per executed lane, the MoE-kernel agent's fit) and then TIMED, both placements, with random
expert weights in the loaded layout (timing does not depend on their values). p50 of --iters synchronous calls
minus the read-back reduction, as tools/probe_moe_ep.py full measures.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def lanes_of(per_slot: list[int]) -> int:
    from eplb_sim import prefill_passes

    st, ov = prefill_passes(per_slot)
    return int(256 * st + 512 * ov)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing-file", required=True)
    ap.add_argument("--init", required=True, help="KILN_EPLB_INIT file the copies are placed from")
    ap.add_argument("--sequences", nargs="+", default=["random0:0", "random0:1024", "random0:2048", "random0:3072"])
    ap.add_argument("--rows", type=int, default=4096)
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(3, 45)))
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--ranks", type=int, default=32)
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()

    import profile_layer as pl
    from probe_moe_ep import ep_experts

    pl.setup_device()
    from kiln.kernels import moe_ep
    from kiln.models import eplb

    H, E, K, R, s = 4096, 288, 8, a.ranks, a.slots
    El = E // R
    real = torch.load(a.routing_file)
    loads = eplb.load_file(a.init, E)
    names = real["names"]
    blobs = {}
    for n in (El, El + s):  # random experts in the loaded tile-scale layout, El and El + s slots
        ws = ep_experts(n, H, 2048, block=True)
        blob = moe_ep.pack(*ws, tiles=True)
        bn = [k for k in ("gu", "sgu", "dn", "sdn", "dsg", "dsd", "tsg", "tsd") if k in blob]
        blobs[n] = (bn, [blob[k].to(pl.DEV) for k in bn])
    g = torch.Generator().manual_seed(1)
    x = torch.randn(a.rows, H, generator=g).bfloat16()
    topv = (torch.rand(a.rows, K, generator=g) + 0.1).bfloat16()
    null = pl.timed("null graph", lambda h: h + 1, (torch.zeros(4, H, dtype=torch.bfloat16).to(pl.DEV),), a.iters)
    ts = pl.timed("read-back reduction", lambda t: t.float().sum(0), (x.to(pl.DEV),), a.iters)

    def run(topi, lmap, nslots):
        bn, bt = blobs[nslots]

        def call(x_, topv_, topi_, lmap_, *b):
            return moe_ep.moe_ep(x_, topv_, topi_, dict(zip(bn, b)), lmap_, 1, 10.0).float().sum(0)

        d = (x.to(pl.DEV), topv.to(pl.DEV), topi.to(torch.int32).to(pl.DEV), lmap.to(pl.DEV), *bt)
        t = pl.timed("kernel", call, d, a.iters)
        return (t - (ts - null)) * 1e3

    tot = [0.0, 0.0]
    print("layer | contiguous busiest rank: pairs, largest expert, ms | EPLB +%d busiest rank: pairs, largest, ms" % s,
          flush=True)
    for L in a.layers:
        per = a.rows // len(a.sequences)
        topi = torch.cat([real["topi"][L][names.index(n)][int(o or 0):int(o or 0) + per].long()
                          for n, _, o in (sp.partition(":") for sp in a.sequences)])
        # contiguous
        owner = torch.arange(E) // El
        cnt = torch.bincount(topi.flatten(), minlength=E)
        per_rank = [cnt[r * El:(r + 1) * El].tolist() for r in range(R)]
        r0 = max(range(R), key=lambda r: lanes_of(per_rank[r]))
        t0 = run(topi, moe_ep.local_map(owner, r0), El)
        # EPLB
        extra = eplb.replicas(loads[L], R, s)
        ids, mp = eplb.tables(extra, E, R, s)
        phys = eplb.remap(topi, ids, mp)
        pc = torch.bincount(phys.flatten(), minlength=E + R * s)
        per_rank_e = [pc[r * El:(r + 1) * El].tolist() + pc[E + r * s:E + (r + 1) * s].tolist() for r in range(R)]
        r1 = max(range(R), key=lambda r: lanes_of(per_rank_e[r]))
        t1 = run(phys, eplb.physical_lmap(E, R, s, r1), El + s)
        tot[0] += t0
        tot[1] += t1
        print(f"L{L} | rank {r0}: {sum(per_rank[r0])}, {max(per_rank[r0])}, {t0:.3f} | rank {r1}: {sum(per_rank_e[r1])}, "
              f"{max(per_rank_e[r1])}, {t1:.3f}", flush=True)
    print(f"sum over {len(a.layers)} layers: contiguous {tot[0]:.1f} ms, EPLB +{s} {tot[1]:.1f} ms "
          f"({tot[1] - tot[0]:+.1f} ms per {a.rows}-row prefill call)", flush=True)


if __name__ == "__main__":
    main()
