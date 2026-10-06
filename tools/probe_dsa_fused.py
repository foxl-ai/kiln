"""The fused pooled-DSA prefill kernel (kernels/dsa_fused.py: indexer scores, exact top-k selection and attention in one
kernel) on one NeuronCore against its CPU emulation, and timed against the two kernels it replaces back to back
(dsa_topk.score_select's kp 4 + tail mask, the visibility added, then kernels/dsa_prefill.py), at GLM-5.3-Flash's
attention-TP-8 rank shape (8 heads, latent 512, a 32 x 128 indexer, 8448 keys = 2112 pools, keep 512).

    python tools/probe_dsa_fused.py [--rows 1024] [--offset O ...] [--forms fused two]

The chunk sits at positions offset .. offset + C - 1 (default: the bucket's end). Reported: max |o - emulate()| / max |o|
and p50 of synchronous calls.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

H, R, HI, D, KEEP = 8, 512, 32, 128, 512


def case(C: int, L: int, off: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    P = L // 4
    qI = torch.randn(C, HI, D, generator=g).to(torch.bfloat16)
    w = torch.randn(C, HI, generator=g) * HI ** -0.5
    pk = torch.randn(P, D, generator=g).to(torch.bfloat16)
    kc = torch.randn(L, R, generator=g).clamp(-6, 6).to(torch.bfloat16)
    q_lat = (torch.randn(C, H, R, generator=g) * 0.05).to(torch.bfloat16)
    pos = (torch.arange(C) + off).to(torch.int32)
    return qI, w, pk, pos, q_lat, kc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1024])
    ap.add_argument("--keys", type=int, default=8448)
    ap.add_argument("--offset", type=int, nargs="+", default=[-1])
    ap.add_argument("--forms", nargs="+", default=["fused", "two"])
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    import profile_layer as pl

    pl.setup_device()
    from kiln.kernels import dsa_fused as df
    from kiln.kernels import dsa_prefill as dp
    from kiln.kernels import dsa_topk as dk

    si, sa = D ** -0.5, 256 ** -0.5
    L = a.keys
    NEG = -1e30
    for C in a.rows:
        for off in a.offset:
            off = L - C if off < 0 else off
            qI, w, pk, pos, q_lat, kc = case(C, L, off, C + off)
            ref = df.emulate(qI, w, pk, pos, q_lat, kc, KEEP, si, sa)
            pl.say(f"C={C} L={L} offset={off}:", flush=True)

            def fused(qI_, w_, pk_, pos_, ql, kc_):
                return df.attend(qI_, w_, pk_, pos_, ql, kc_, KEEP, si, sa)

            def two(qI_, w_, pk_, pos_, ql, kc_):  # the engine's path with both kernels: score_select, vis, attention
                P = pk_.shape[0]
                last = torch.arange(P, device=pos_.device, dtype=torch.int32) * 4 + 3
                cand = torch.where(last.view(1, P) <= pos_.view(-1, 1), 0.0, NEG)
                sel = dk.score_select(qI_, w_, pk_, cand, KEEP, si, 4, True)
                vis = torch.where(torch.arange(L, device=pos_.device, dtype=torch.int32).view(1, L) <= pos_.view(-1, 1),
                                  0.0, NEG)
                return dp.attend(ql, kc_, sel + vis, sa, pos_)

            if "dbg" in a.forms:  # the kernel's token mask and pool selection against the emulation's
                P = L // 4
                last = torch.arange(P) * 4 + 3
                cand = torch.where(last.view(1, P) <= pos.view(C, 1).long(), 0.0, NEG)
                index = dk.emulate_scores(qI, w, pk, cand, si)
                selp = dk.emulate(index, KEEP, True, 1, True)  # [C, P] pool level, tail included
                vis = torch.where(torch.arange(L).view(1, L) <= pos.view(C, 1).long(), 0.0, NEG)
                want_mask = (dk.emulate(index, KEEP, True, 4, True) + vis).clamp(min=NEG)
                f = torch.compile(lambda *x: df.attend(*x, KEEP, si, sa, dbg=1), **pl.OPTS)
                o, dmask, dsel, dsc = (t_.cpu() for t_ in f(*(x.to(pl.DEV) for x in (qI, w, pk, pos, q_lat, kc))))
                e_ = (dsc - index[:128]).abs()
                fin = index[:128] > -1e29
                pl.say(f"    dbg: tile 0 scores max|err| {float(e_[fin].max()):.3e} (|index| max {float(index[:128][fin].abs().max()):.3e}); "
                       f"exact {bool(torch.equal(dsc, index[:128]))}; masked agree {bool(((dsc < -1e29) == ~fin).all())}", flush=True)
                for r_ in range(2):
                    k_ = (e_[r_] * fin[r_]).argmax()
                    pl.say(f"     row {r_}: worst pool {int(k_)} got {float(dsc[r_, k_]):.5f} want {float(index[r_, k_]):.5f}; "
                           f"got/want ratio over row {float((dsc[r_][fin[r_]] / index[r_][fin[r_]]).median()):.4f}", flush=True)
                gs = dsel.float() > 0.5
                ws_ = selp == 0
                bad = (gs != ws_).any(-1)
                pl.say(f"    dbg: pool selection rows wrong {int(bad.sum())} of {C} (first {bad.nonzero().flatten()[:6].tolist()}); "
                       f"selected per row got {sorted(set(gs.sum(-1).tolist()))[:5]} want {sorted(set(ws_.sum(-1).tolist()))[:5]}",
                       flush=True)
                gm = dmask.float() > -1e29
                wm = want_mask > -1e29
                badm = (gm != wm).any(-1)
                pl.say(f"    dbg: token mask rows wrong {int(badm.sum())} of {C} (first {badm.nonzero().flatten()[:6].tolist()}); "
                       f"attended per row got {sorted(set(gm.sum(-1).tolist()))[:5]} want {sorted(set(wm.sum(-1).tolist()))[:5]}",
                       flush=True)
                if badm.any():
                    r = int(badm.nonzero()[0])
                    d = (gm[r] != wm[r]).nonzero().flatten()
                    pl.say(f"    row {r} (pos {int(pos[r])}): first wrong keys {d[:12].tolist()}", flush=True)
            for name in a.forms:
                if name == "dbg":
                    continue
                fn = fused if name == "fused" else two
                dev = tuple(x.to(pl.DEV) for x in (qI, w, pk, pos, q_lat, kc))
                t = pl.timed(f"{name} C={C} off={off}", lambda *x, fn=fn: fn(*x).sum(), dev, a.iters)
                if t != t:
                    continue
                o = torch.compile(fn, **pl.OPTS)(*dev).cpu()
                err = ((o - ref).abs().max() / ref.abs().max()).item()
                rows = ((o - ref).abs().amax((1, 2)) / ref.abs().max() > 1e-2).sum().item()
                pl.say(f"    {name}: o max|err| / max|o| {err:.2e}; rows off by more than 1e-2: {rows}; non-finite "
                       f"{int((~torch.isfinite(o)).sum())}", flush=True)


if __name__ == "__main__":
    main()
