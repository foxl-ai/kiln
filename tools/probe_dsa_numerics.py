"""Which form of the pooled-DSA prefill attention core is closest to exact arithmetic: the XLA forms (models/mla.py
_core expand / absorb), kernels/dsa_prefill.py (the static kernel; kernels/dsa_fused.py runs the same attention
arithmetic after its selection) and its CPU emulation, each against an fp64 reference on the same bf16 inputs.

Why: on wikitext the fused and the static kernel move per-token logprobs against XLA by the same amount (mean |d|
0.042-0.043 in prefill, signed mean within one SE of 0; docs/neuron-notes.md "The final combined measurement"), and the
follow-up asked whether an fp32 P K accumulation in the kernels, or XLA's reduction order, closes it. That only helps if
the kernels are the less exact side. The XLA forms round the scores to bf16 (the einsum of two bf16 tensors returns
bf16) and the normalised p to bf16; the kernel keeps S in fp32 PSUM and rounds the unnormalised P = exp(S - m) to bf16
before P K, dividing by an fp32 row sum.

    python tools/probe_dsa_numerics.py [--rows 1024] [--keys 8448] [--qscale 1 4 16] [--forms xla xla-absorb nki emu xla-f32]

The inputs are tools/probe_dsa_prefill.py's case (512 random selected pools of 4 + the tail, causal, the chunk at the
bucket's end); --qscale multiplies q (sharper softmax). Reported per form: ||o - ref|| / ||ref|| over the chunk, the mean
and p99 of the per-row relative error, and max |o - ref| / max |ref|; o is the latent-space output before W_UV.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--keys", type=int, default=8448)
    ap.add_argument("--qscale", type=float, nargs="+", default=[1.0, 4.0, 16.0])
    ap.add_argument("--forms", nargs="+", default=["xla", "xla-absorb", "nki", "emu", "xla-f32"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import probe_dsa_prefill as pdp
    import profile_layer as pl

    pl.setup_device()
    from kiln.kernels import dsa_prefill as dp

    scale = pdp.DN ** -0.5
    C, L = a.rows, a.keys
    for qs in a.qscale:
        q_lat, kc, mask, q_nope, w_uk, w_uv = pdp.case(C, L, L - C, a.seed + C)
        q_nope = (q_nope.float() * qs).to(torch.bfloat16)
        q_lat = torch.einsum("chd,hdr->chr", q_nope.float(), w_uk.float()).to(torch.bfloat16)
        # fp64 on the bf16 inputs each form starts from: q_lat for the absorbed forms, q_nope W_UK for expand
        def exact(ql):
            s = torch.einsum("chr,lr->hcl", ql.double(), kc.double()) * scale + mask.double().unsqueeze(0)
            return torch.einsum("hcl,lr->chr", torch.softmax(s, dim=-1), kc.double())

        ref_abs = exact(q_lat)
        ref_exp = exact(torch.einsum("chd,hdr->chr", q_nope.double(), w_uk.double()))
        p = torch.softmax((torch.einsum("chr,lr->hcl", q_lat.double(), kc.double()) * scale
                           + mask.double().unsqueeze(0)), dim=-1)
        pl.say(f"qscale {qs}: max p per row mean {float(p.amax(-1).mean()):.3f}, keys with p > 1e-3 per row mean "
               f"{float((p > 1e-3).sum(-1).double().mean()):.0f}", flush=True)

        def xla(qn, kc_, m_, wk):  # _core's expand branch
            k_nope = torch.einsum("lr,hdr->lhd", kc_, wk)
            s = torch.einsum("qhd,lhd->hql", qn, k_nope)
            p_ = torch.softmax(s.float() * scale + m_.unsqueeze(0), dim=-1).to(torch.bfloat16)
            return torch.einsum("hql,lr->qhr", p_, kc_).float()

        def xla_absorb(ql, kc_, m_):
            s = torch.einsum("qhr,lr->hql", ql, kc_)
            p_ = torch.softmax(s.float() * scale + m_.unsqueeze(0), dim=-1).to(torch.bfloat16)
            return torch.einsum("hql,lr->qhr", p_, kc_).float()

        def xla_f32(ql, kc_, m_):  # absorbed, every operand and product in fp32
            s = torch.einsum("qhr,lr->hql", ql.float(), kc_.float())
            p_ = torch.softmax(s * scale + m_.unsqueeze(0), dim=-1)
            return torch.einsum("hql,lr->qhr", p_, kc_.float())

        def nki(ql, kc_, m_):
            return dp.attend(ql, kc_, m_, scale)

        for name in a.forms:
            if name == "emu":
                o, ref = dp.emulate(q_lat, kc, mask, scale).double(), ref_abs
            else:
                fn, args, ref = {"xla": (xla, (q_nope, kc, mask, w_uk), ref_exp),
                                 "xla-absorb": (xla_absorb, (q_lat, kc, mask), ref_abs),
                                 "xla-f32": (xla_f32, (q_lat, kc, mask), ref_abs),
                                 "nki": (nki, (q_lat, kc, mask), ref_abs)}[name]
                dev = tuple(x.to(pl.DEV) for x in args)
                o = torch.compile(fn, **pl.OPTS)(*dev).cpu().double()
            d = o - ref
            rows = d.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)  # [C, H]
            pl.say(f"  {name:<11} rel {float(d.norm() / ref.norm()):.3e}  row mean {float(rows.mean()):.3e} p99 "
                   f"{float(rows.flatten().quantile(0.99)):.3e}  max|d|/max|ref| {float(d.abs().max() / ref.abs().max()):.3e}",
                   flush=True)


if __name__ == "__main__":
    main()
