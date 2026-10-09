"""kernels/dsa_long_select_t.py (one selection tile under a run-time trip count) against kernels/dsa_long_select.py on
one NeuronCore: at trip 1 the pools and scores must be bit-identical, and each form's time; at trip 0 its time.

    python tools/probe_lc_select_t.py [--pools 8192 32768] [--keep 512 128] [--kinds pooled ties short ...]

Inputs are tools/probe_dsa_long.py make_inputs' (128 queries, 32 heads of 128).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", type=int, nargs="+", default=[8192, 32768])
    ap.add_argument("--keep", type=int, nargs="+", default=[512, 128])
    ap.add_argument("--kinds", nargs="+", default=["pooled", "ties", "zeros", "wide", "short"])
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    import profile_layer as pl
    import probe_dsa_long as pdl

    pl.setup_device(False)
    if pl.DEV is None or pl.DEV.type == "cpu":
        raise SystemExit("probe_lc_select_t needs a NeuronCore")
    from kiln.kernels import dsa_long_select as dl
    from kiln.kernels import dsa_long_select_t as dlt

    N, scale = 128, 128 ** -0.5
    for P in a.pools:
        for keep in a.keep:
            for ki, kind in enumerate(a.kinds):
                g = torch.Generator().manual_seed(N * 7 + P + len(kind) + keep)
                q, w, pk, npool = pdl.make_inputs(kind, N, P, keep, g)
                dev = tuple(x.to(pl.DEV) for x in (q, w, pk, npool))
                one, zero = (torch.tensor([v], dtype=torch.int32).to(pl.DEV) for v in (1, 0))

                def ref(q_, w_, pk_, np_):
                    p_, _, v_ = dl.select(q_, w_, pk_, np_, keep, scale)
                    return p_, v_

                def tsel(q_, w_, pk_, np_, tr):
                    p_, _, v_ = dlt.select(q_, w_, pk_, np_, keep, scale, tr)
                    return p_, v_

                name = f"N={N} P={P} keep={keep} {kind}"
                rp, rv = (x.cpu() for x in torch.compile(ref, **pl.OPTS)(*dev))
                gp, gv = (x.cpu() for x in torch.compile(tsel, **pl.OPTS)(*dev, one))
                same = torch.equal(rp, gp) and torch.equal(rv.view(torch.int32), gv.view(torch.int32))
                msg = (f"  {name}: trip 1 == dsa_long_select bit for bit: {same} "
                       f"({int((rp != gp).any(-1).sum())} rows differ in pools, "
                       f"{int((rv.view(torch.int32) != gv.view(torch.int32)).any(-1).sum())} in scores)")
                if ki == 0:
                    t0 = pl.timed(f"select       {name}", lambda *x: ref(*x)[0].sum(), dev, a.iters)
                    t1 = pl.timed(f"select_t 1   {name}", lambda *x: tsel(*x)[0].sum(), (*dev, one), a.iters)
                    tz = pl.timed(f"select_t 0   {name}", lambda *x: tsel(*x)[0].sum(), (*dev, zero), a.iters)
                    msg += f"; select {t0 * 1e3:.3f} ms, trip 1 {t1 * 1e3:.3f} ms, trip 0 {tz * 1e3:.3f} ms"
                pl.say(msg, flush=True)


if __name__ == "__main__":
    main()
