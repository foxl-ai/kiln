"""Does the NKI Library's MoE token-generation kernel run on trn1? nkilib.core.moe.moe_tkg.moe_tkg
(SDK 2.32 venv) in selective-expert mode with FP8 ROW weights, through LNL's wrap_nki, at one
tensor-parallel rank's MiMo-V2.6-Flash shapes on one NeuronCore.

    python tools/probe_nkilib_moe.py [--batch 4] [--dtype fp8|bf16]

ROW is the closest of its FP8 modes to Kiln's experts (one scale per output channel, [E, 2, I] and
[E, H]; Kiln's are per 32 input columns), so this answers whether the shipped kernel runs on gen2
at all, not whether it fits Kiln's layout (it does not; kiln/kernels/moe_decode.py). Prints the
first error line if it fails, else the max error against an fp32 host reference and the p50.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dtype", default="fp8", choices=["fp8", "bf16"])
    # bf16 affinities fail on trn2 in selective mode: "'nisa.tensor_scalar_arith' op 'operand0' must
    # be float32, got 'bf16'" (selective_expert_impl.py:348, the POST_SCALE multiply).
    ap.add_argument("--aff-dtype", default="fp32", choices=["fp32", "bf16"])
    args = ap.parse_args()
    import profile_layer as pl

    pl.setup_device()
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from nkilib.core.moe.moe_tkg.moe_tkg import moe_tkg
    from nkilib.core.utils.common_types import ActFnType, ExpertAffinityScaleMode

    E, H, I, K, T = 256, 4096, 64, 8, args.batch
    g = torch.Generator().manual_seed(0)
    if args.dtype == "fp8":
        fp8 = lambda *s: (torch.randint(0, 256, s, dtype=torch.uint8, generator=g) & 0xBF).view(torch.float8_e4m3fn)  # noqa: E731
        w_gu, w_dn = fp8(E, H, 2, I), fp8(E, I, H)
        s_gu = torch.rand(E, 2, I, generator=g) * 0.02 + 0.005
        s_dn = torch.rand(E, H, generator=g) * 0.02 + 0.005
        wg_f = w_gu.float() * s_gu.unsqueeze(1)
        wd_f = w_dn.float() * s_dn.unsqueeze(1)
    else:
        w_gu = (torch.randn(E, H, 2, I, generator=g) * 0.02).bfloat16()
        w_dn = (torch.randn(E, I, H, generator=g) * 0.02).bfloat16()
        s_gu = s_dn = None
        wg_f, wd_f = w_gu.float(), w_dn.float()
    x = torch.randn(T, H, generator=g).bfloat16()
    topi = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)]).to(torch.int32)
    aff = torch.zeros(T, E)
    aff.scatter_(1, topi.long(), torch.rand(T, K, generator=g) + 0.1)
    aff = aff.bfloat16() if args.aff_dtype == "bf16" else aff.float()
    ref = torch.zeros(T, H)
    for t in range(T):
        for e in topi[t].tolist():
            gu = x[t].float() @ wg_f[e].reshape(H, 2 * I)  # [2I]: gate then up
            ref[t] += aff[t, e].float() * ((torch.nn.functional.silu(gu[:I]) * gu[I:]) @ wd_f[e])
    d = lambda t: None if t is None else t.to(pl.DEV)  # noqa: E731
    kern = wrap_nki(moe_tkg)
    from kiln import platform

    grid = platform.nki_grid()  # the runtime LNC: 1 on trn1, 2 on trn2

    def f(x, w_gu, w_dn, aff, idx, s_gu, s_dn):
        return kern[grid](hidden_input=x, expert_gate_up_weights=w_gu, expert_down_weights=w_dn,
                       expert_affinities=aff, expert_index=idx, is_all_expert=False,
                       expert_gate_up_weights_scale=s_gu, expert_down_weights_scale=s_dn,
                       expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
                       activation_fn=ActFnType.SiLU)

    inputs = (d(x), d(w_gu), d(w_dn), d(aff), d(topi), d(s_gu), d(s_dn))
    print(f"nkilib moe_tkg selective, {args.dtype} weights, T={T} E={E} H={H} I={I} top-{K}, grid {grid}", flush=True)
    try:
        out = torch.compile(f, **pl.OPTS)(*inputs).cpu().float()
    except Exception as e:  # the answer is the error
        lines = [l for l in traceback.format_exception_only(type(e), e) if l.strip()]
        print("FAILED:", " | ".join(l.strip() for l in lines)[:1500], flush=True)
        return
    err = (out - ref).abs().max().item()
    print(f"ran: max abs error vs fp32 {err:.3e} (rel {err / ref.abs().max().item():.4f})", flush=True)
    pl.timed(f"T={T} nkilib moe_tkg", f, inputs)


if __name__ == "__main__":
    main()
