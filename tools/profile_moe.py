"""Compile time and latency of one MoE layer per formulation, at one tensor-parallel rank's
shape of a real model, on one NeuronCore.

    python tools/profile_moe.py [--experts 256 --hidden 4096 --inter 64 --topk 8] [--tokens 4 32]

Default dims are MiMo-V2.6-Flash at tp=32 (moe_intermediate_size 2048 / 32). Variants: expert
storage (packed MXFP4 or FP8 + scale) x path (gather: only the selected experts' weights,
dense: every expert over every token), at each token count. Random weights; the numbers are
compile seconds and steady-state milliseconds, not accuracy.
"""

from __future__ import annotations

import argparse
import time
import types

import torch

import libtorch_neuronx_lite  # noqa: F401

from kiln.engine.model_runner import neuronx_cc_args
from kiln.models.decoder import DecoderForCausalLM, _LayerView
from kiln.models.quant import FP8, dequant_mxfp4, mxfp4_byte_table

DEV = torch.device("neuron:0")


def parts(E, H, Im, k, T, g) -> None:
    """Where the gather path's time goes: the weight gather alone, the FP8 dequant alone, the
    MXFP4 decode alone, the batched matvec alone (each graph returns a small reduction)."""
    P = T * k
    idx = torch.randint(0, E, (P,), generator=g).to(DEV)
    w8 = (torch.randn(E, 2 * Im, H, generator=g) * 0.5).to(FP8).to(DEV)
    s8 = torch.full((E, 2 * Im, H // 128), 0.01).to(DEV)
    w4 = torch.randint(0, 256, (E, 2 * Im, H // 2), dtype=torch.uint8, generator=g).to(DEV)
    s4 = torch.full((E, 2 * Im, H // 32), 2.0 ** -6, dtype=torch.bfloat16).to(DEV)
    wb = (torch.randn(P, 2 * Im, H, generator=g) * 0.02).to(torch.bfloat16).to(DEV)
    x = (torch.randn(P, H, 1, generator=g)).to(torch.bfloat16).to(DEV)
    wd8 = (torch.randn(E, H, Im, generator=g) * 0.5).to(FP8).to(DEV)
    sd8 = torch.full((E, H, 1), 0.01).to(DEV)
    router = (torch.randn(E, H, generator=g) * 0.02).to(torch.bfloat16).to(DEV)
    wd8t = (torch.randn(E, Im, H, generator=g) * 0.5).to(FP8).to(DEV)  # down stored [E, in, out]
    sd8t = torch.full((E, 1, H), 0.01).to(DEV)  # one scale per output column (in = 64 < 128)
    s4h = torch.full((E, 2 * Im, H // 32), 2.0 ** -6, dtype=torch.bfloat16).to(DEV)
    from kiln.models.quant import dequant

    def dequant_t(w, s):
        return w.to(torch.bfloat16) * s.to(torch.bfloat16)

    def e2m1(c):
        sg = torch.floor(c * 0.125)
        m = c - sg * 8.0
        e = torch.floor(m * 0.5)
        f = m - e * 2.0
        return (e + torch.floor(e * 0.375) + f * (0.5 + e * (e - 1.0) * 0.25)) * (1.0 - sg * 2.0)

    def planar(w):  # low nibbles and high nibbles as separate halves: no interleave
        b = w.to(torch.bfloat16)
        hi = torch.floor(b * 0.0625)
        return torch.cat([e2m1(b - hi * 16.0), e2m1(hi)], dim=-1)

    def planar_scaled(w, s):  # each half's element j belongs to block j // 16
        sc = s.repeat_interleave(16, dim=-1)
        b = w.to(torch.bfloat16)
        hi = torch.floor(b * 0.0625)
        return torch.cat([e2m1(b - hi * 16.0) * sc, e2m1(hi) * sc], dim=-1)

    probes = {
        "gather fp8": (lambda w, i: w[i].to(torch.bfloat16).sum(dim=(1, 2)), (w8, idx), True),
        "gather u8": (lambda w, i: w[i].to(torch.bfloat16).sum(dim=(1, 2)), (w4, idx), False),
        "dequant fp8 (pre-gathered)": (lambda w, s: dequant(w, s, torch.bfloat16).sum(dim=(1, 2)),
                                       (w8[:P].contiguous(), s8[:P].contiguous()), True),
        "decode mxfp4 (pre-gathered)": (lambda w, s: dequant_mxfp4(w, s, torch.bfloat16).sum(dim=(1, 2)),
                                        (w4[:P].contiguous(), s4[:P].contiguous()), False),
        "bmm bf16 (pre-gathered)": (lambda w, v: torch.bmm(w, v).sum(dim=(1, 2)), (wb, x), False),
        "gather+dequant fp8 gate_up": (lambda w, s, i: dequant(w[i], s[i], torch.bfloat16).sum(dim=(1, 2)),
                                       (w8, s8, idx), True),
        "gather+dequant+bmm fp8 gate_up": (lambda w, s, i, v: torch.bmm(dequant(w[i], s[i], torch.bfloat16), v)
                                           .sum(dim=(1, 2)), (w8, s8, idx, x), True),
        "gather+dequant fp8 down": (lambda w, s, i: dequant(w[i], s[i], torch.bfloat16).sum(dim=(1, 2)),
                                    (wd8, sd8, idx), True),
        "down fp8 transposed [P,Im,H]": (lambda w, s, i: dequant_t(w[i], s[i]).sum(dim=(1, 2)), (wd8t, sd8t, idx), True),
        "mxfp4 planar, no scale": (lambda w, i: planar(w[i]).sum(dim=(1, 2)), (w4, idx), False),
        "mxfp4 planar, scale repeat": (lambda w, s, i: planar_scaled(w[i], s[i]).sum(dim=(1, 2)), (w4, s4h, idx), False),
        "router": (lambda r, h: torch.topk(torch.nn.functional.linear(h.float(), r.float()).sigmoid(), k, dim=-1)[0].sum(),
                   (router, x[:T, :, 0].contiguous()), False),
    }
    for name, (f, a, fp8) in probes.items():
        c = torch.compile(f, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                          options={"compiler_args": neuronx_cc_args(torch.bfloat16, fp8)})
        c(*a).cpu()
        ts = []
        for _ in range(20):
            t0 = time.perf_counter()
            c(*a).cpu()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        print(f"  part {name:<28} pairs={P:<4} p50 {ts[10] * 1e3:8.3f} ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--inter", type=int, default=64, help="per-rank expert intermediate size")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--tokens", type=int, nargs="+", default=[4, 32])
    ap.add_argument("--variants", nargs="+",
                    default=["packed-gather", "packed-dense", "fp8-gather", "fp8-dense", "table-gather", "table-dense"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--parts", action="store_true", help="time the gather path's pieces instead")
    args = ap.parse_args()
    E, H, Im, k = args.experts, args.hidden, args.inter, args.topk
    g = torch.Generator().manual_seed(0)
    if args.parts:
        for T in args.tokens:
            parts(E, H, Im, k, T, g)
        return

    cfg = types.SimpleNamespace(num_experts=E, num_experts_per_tok=k, router_scoring="sigmoid",
                                routed_scaling_factor=1.0, norm_topk_prob=True)
    weights = {
        "packed": dict(w_gu=torch.randint(0, 256, (E, 2 * Im, H // 2), dtype=torch.uint8, generator=g),
                       w_gu_scale=torch.full((E, 2 * Im, H // 32), 2.0 ** -6, dtype=torch.bfloat16),
                       w_down=torch.randint(0, 256, (E, H, Im // 2), dtype=torch.uint8, generator=g),
                       w_down_scale=torch.full((E, H, Im // 32), 2.0 ** -6, dtype=torch.bfloat16)),
        "bf16": dict(w_gu=(torch.randn(E, 2 * Im, H, generator=g) * 0.02).to(torch.bfloat16),
                     w_gu_scale=None,
                     w_down=(torch.randn(E, Im, H, generator=g) * 0.02).to(torch.bfloat16),  # [E, in, out]
                     w_down_scale=None),
        "fp8": dict(w_gu=(torch.randn(E, 2 * Im, H, generator=g) * 0.5).to(FP8),
                    w_gu_scale=torch.full((E, 2 * Im, H // 128), 0.01),
                    w_down=(torch.randn(E, Im, H, generator=g) * 0.5).to(FP8),  # [E, in, out]
                    w_down_scale=torch.full((E, 1, H), 0.01)),
    }
    router = (torch.randn(E, H, generator=g) * 0.02).to(torch.bfloat16).to(DEV)
    bias = torch.zeros(E).to(DEV)
    print(f"E={E} H={H} Im={Im} k={k}")
    for name in args.variants:
        store, path = name.split("-")
        ws = {n: (t.to(DEV) if t is not None else None)
              for n, t in weights["packed" if store == "table" else store].items()}
        names = ["router", "router_bias", *ws]
        for T in args.tokens:
            m = types.SimpleNamespace(cfg=cfg, moe_inter=Im, dtype=torch.bfloat16,
                                      MOE_GATHER_MAX_PAIRS=(1 << 20) if path == "gather" else 0)
            for f in ("_moe", "_route", "_experts"):
                setattr(m, f, types.MethodType(getattr(DecoderForCausalLM, f), m))

            if store == "table":  # decode packed MXFP4 by byte lookup instead of arithmetic
                table = mxfp4_byte_table(torch.bfloat16, DEV)

                def experts(layer, name, idx=None, _t=table):
                    w, sc = getattr(layer, name), getattr(layer, name + "_scale")
                    if idx is not None:
                        w, sc = w[idx], sc[idx]
                    return dequant_mxfp4(w, sc, torch.bfloat16, _t)

                m._experts = experts

            static = {"down_t": store in ("bf16", "fp8")}

            def fn(x, *ts):
                return m._moe(_LayerView(static, dict(zip(names, ts))), x)

            # The engine's compiler flags: FP8 graphs need the unsafe e4m3fn-as-e4m3 option on trn1/trn2.
            c = torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                              options={"compiler_args": neuronx_cc_args(torch.bfloat16, store == "fp8")})
            x = (torch.randn(T, H, generator=g) * 0.5).to(torch.bfloat16).to(DEV)
            t0 = time.perf_counter()
            try:
                c(x, router, bias, *ws.values()).cpu()
            except Exception as e:  # report and go on: one formulation failing must not hide the rest
                print(f"  {name:<14} T={T:<4} FAILED {type(e).__name__}: {str(e)[:160]}")
                continue
            compile_s = time.perf_counter() - t0
            ts = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                c(x, router, bias, *ws.values()).cpu()
                ts.append(time.perf_counter() - t0)
            ts.sort()
            print(f"  {name:<14} T={T:<4} pairs={T * k:<5} compile {compile_s:7.1f} s   "
                  f"p50 {ts[len(ts) // 2] * 1e3:8.3f} ms")


if __name__ == "__main__":
    main()
