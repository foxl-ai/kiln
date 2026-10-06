"""Per-layer, per-rank, per-slot pair counts of 4096-row prefill batches under the contiguous placement and under
EPLB copies (models/eplb.py), from tools/ep_routing.py's saved routing: what each rank's EP kernel call holds, for
sizing kernel passes.

    python tools/eplb_dump_counts.py --load ep_routing.pt --init eplb-init-random.pt --out counts.pt
        [--sequences random0 random1] [--batches 20] [--slots 1]

Batches as bench/serve_sweep.py forms them (4 chunks of 1024 consecutive tokens, each from one of --sequences at a
1024-aligned offset); copies placed from --init by eplb.replicas, pairs mapped by eplb.remap (row class t mod 16).
Saved: {"contiguous": {layer: int32 [batches, ranks, E / ranks]}, "eplb": {layer: int32 [batches, ranks, E / ranks + s]},
"extra": {layer: [ranks s experts]}, "source": ...}; slot j < E / ranks of rank r is expert r E / ranks + j, slot
E / ranks + i is extra[layer][r s + i].
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> None:
    from kiln.models import eplb

    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sequences", nargs="+", default=["random0", "random1"])
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--ranks", type=int, default=32)
    ap.add_argument("--slots", type=int, default=1)
    a = ap.parse_args()
    d = torch.load(a.load)
    E, R, s = d["experts"], a.ranks, a.slots
    El = E // R
    names = d["names"]
    sel = [names.index(n) for n in a.sequences]
    loads = eplb.load_file(a.init, E)
    g = torch.Generator().manual_seed(0)
    T = d["topi"][sorted(d["topi"])[0]][sel[0]].shape[0]
    picks = [[(sel[int(torch.randint(len(sel), (1,), generator=g))], int(torch.randint(T // 1024, (1,), generator=g)) * 1024)
              for _ in range(4)] for _ in range(a.batches)]
    out = {"contiguous": {}, "eplb": {}, "extra": {},
           "source": f"tools/ep_routing.py routing {a.load} sequences {a.sequences}, copies from {a.init}, s={s}"}
    for L, seqs in sorted(d["topi"].items()):
        extra = eplb.replicas(loads[L], R, s)
        ids, mp = eplb.tables(extra, E, R, s)
        c = torch.zeros(a.batches, R, El, dtype=torch.int32)
        e = torch.zeros(a.batches, R, El + s, dtype=torch.int32)
        for b, pk in enumerate(picks):
            topi = torch.cat([seqs[i][o:o + 1024].long() for i, o in pk])
            cnt = torch.bincount(topi.flatten(), minlength=E)
            c[b] = cnt.view(R, El).to(torch.int32)
            pc = torch.bincount(eplb.remap(topi, ids, mp).flatten(), minlength=E + R * s)
            e[b, :, :El] = pc[:E].view(R, El).to(torch.int32)
            e[b, :, El:] = pc[E:].view(R, s).to(torch.int32)
        out["contiguous"][L], out["eplb"][L], out["extra"][L] = c, e, extra
    torch.save(out, a.out)
    print(f"saved {a.out}: {len(out['eplb'])} layers x {a.batches} batches x {R} ranks", flush=True)


if __name__ == "__main__":
    main()
