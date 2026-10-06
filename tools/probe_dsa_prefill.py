"""The pooled-DSA prefill attention core on one NeuronCore: the XLA forms (models/mla.py _core, expand and absorb,
over every key of the bucket with the selection and the causal visibility as an additive mask) against
kernels/dsa_prefill.py (nki: every key block; nki-loop: the causal form, KILN_DSA_PREFILL_LOOP), all against the CPU
emulation, at GLM-5.3-Flash's attention-TP-8 rank shape (8 heads, latent
512, dn = dv = 256, a chunk of C queries over L = 8448 keys).

    python tools/probe_dsa_prefill.py [--rows 1024] [--keys 8448] [--offset O ...] [--forms xla xla-absorb nki]

The chunk sits at positions offset .. offset + C - 1 (default: the last C positions of the bucket); each query
selects 512 random visible complete pools of 4 tokens (all of them when fewer) plus its own incomplete pool, as the
pooled selection does. Reported: max |error| of o (the latent-space output, before W_UV) against emulate() relative to
its max, and p50 of synchronous calls (profile_layer.timed).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

H, R, DN, KP, KEEP = 8, 512, 256, 4, 512
NEG_INF = -1e30


def case(C: int, L: int, offset: int, seed: int):
    """q_lat [C, H, R] bf16, kc [L, R] bf16, mask [C, L] fp32 (selection + causal), w_uv [H, DN, R] bf16, and the
    expand form's q_nope [C, H, DN] / w_uk [H, DN, R] with q_lat = q_nope W_UK."""
    g = torch.Generator().manual_seed(seed)
    kc = torch.randn(L, R, generator=g).clamp(-6, 6).to(torch.bfloat16)
    q_nope = (torch.randn(C, H, DN, generator=g) * 0.25).to(torch.bfloat16)
    w_uk = (torch.randn(H, DN, R, generator=g) * DN ** -0.5).to(torch.bfloat16)
    w_uv = (torch.randn(H, DN, R, generator=g) * R ** -0.5).to(torch.bfloat16)
    q_lat = torch.einsum("chd,hdr->chr", q_nope.float(), w_uk.float()).to(torch.bfloat16)
    P = L // KP
    pos = torch.arange(C) + offset
    nvis = pos + 1
    score = torch.rand(C, P, generator=g)
    cand = torch.arange(P).view(1, P) < (nvis // KP).view(C, 1)  # complete visible pools
    score = torch.where(cand, score, torch.full_like(score, -1.0))
    top = torch.topk(score, KEEP, dim=-1).indices
    sel = torch.zeros(C, P, dtype=torch.bool).scatter(1, top, True) & cand
    tok = sel.repeat_interleave(KP, dim=1) | ~cand.repeat_interleave(KP, dim=1)  # + the tail (non-candidate pools)
    vis = torch.arange(L).view(1, L) <= pos.view(C, 1)
    mask = torch.where(tok & vis, 0.0, NEG_INF)
    return q_lat, kc, mask, q_nope, w_uk, w_uv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1024])
    ap.add_argument("--keys", type=int, default=8448)
    ap.add_argument("--offset", type=int, nargs="+", default=[-1], help="first query position (-1: the bucket's end)")
    ap.add_argument("--forms", nargs="+", default=["xla", "xla-absorb", "nki", "nki-loop"])
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    import profile_layer as pl

    pl.setup_device()
    from kiln.kernels import dsa_prefill as dp

    scale = (DN + 0) ** -0.5  # GLM-5.3-Flash: qk_head_dim 256 (NoPE)
    L = a.keys
    for C in a.rows:
        for off in a.offset:
            off = L - C if off < 0 else off
            q_lat, kc, mask, q_nope, w_uk, w_uv = case(C, L, off, C + off)
            ref = dp.emulate(q_lat, kc, mask, scale)
            pl.say(f"C={C} L={L} offset={off}: attended keys per query mean "
                   f"{float((mask == 0).float().sum(-1).mean()):.0f}", flush=True)

            def xla(qn, kc_, m_, wk):  # _core's expand branch (o in latent space for the comparison: W_UV after)
                k_nope = torch.einsum("lr,hdr->lhd", kc_, wk)
                s = torch.einsum("qhd,lhd->hql", qn, k_nope)
                p = torch.softmax(s.float() * scale + m_.unsqueeze(0), dim=-1).to(torch.bfloat16)
                return torch.einsum("hql,lr->qhr", p, kc_).float()

            def xla_absorb(ql, kc_, m_):
                s = torch.einsum("qhr,lr->hql", ql, kc_)
                p = torch.softmax(s.float() * scale + m_.unsqueeze(0), dim=-1).to(torch.bfloat16)
                return torch.einsum("hql,lr->qhr", p, kc_).float()

            def nki(ql, kc_, m_):
                return dp.attend(ql, kc_, m_, scale)

            def nki_loop(ql, kc_, m_, pos):
                dp.LOOP = True
                try:
                    return dp.attend(ql, kc_, m_, scale, pos)
                finally:
                    dp.LOOP = False

            for name in a.forms:
                if name == "xla":
                    fn, args = xla, (q_nope, kc, mask, w_uk)
                elif name == "xla-absorb":
                    fn, args = xla_absorb, (q_lat, kc, mask)
                elif name == "nki-loop":
                    fn, args = nki_loop, (q_lat, kc, mask, (torch.arange(C) + off).to(torch.int32))
                else:
                    fn, args = nki, (q_lat, kc, mask)
                dev = tuple(x.to(pl.DEV) for x in args)
                t = pl.timed(f"{name} C={C} off={off}", lambda *x, fn=fn: fn(*x).sum(), dev, a.iters)
                if t != t:
                    continue
                o = torch.compile(fn, **pl.OPTS)(*dev).cpu()
                err = ((o - ref).abs().max() / ref.abs().max()).item()
                bad = ~torch.isfinite(o).all(-1).all(-1)
                pl.say(f"    {name}: o max|err| / max|o| {err:.2e}; rows with a non-finite value {int(bad.sum())}; "
                       f"worst row {int((o - ref).abs().amax((1, 2)).argmax())}", flush=True)


if __name__ == "__main__":
    main()
