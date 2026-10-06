"""Does an XLA dot with FP8 operands run faster than BF16 on this NeuronCore? (trn2: "FP8 double performance mode"
on NeuronCore-v3, nki/isa nc_matmul perf_mode double_row: "2x matmul throughput by packing two FP8 weight/ifmap
element pairs"; whether neuronx-cc uses it for an HLO dot of f8e4m3 operands is what this measures.)

    python tools/probe_fp8_matmul.py [--shapes 1024x4096x4096 4096x4096x4096] [--iters 20]

Per shape M x K x N: y = x @ w with x [M, K], w [K, N], in bf16, and with both operands cast to float8_e4m3fn
(per-tensor scale folded in, as a dynamic-quantized projection would be) and the product in bf16; p50 of synchronous
calls of a graph that holds R matmuls against R different weights (so the per-graph floor is amortised and no
common subexpression folds them), and the relative error
of the FP8 product against the bf16 one.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["1024x4096x4096", "4096x4096x4096", "1024x4096x16384"])
    ap.add_argument("--reps", type=int, default=8, help="matmuls per graph (chained through a cheap add)")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    import profile_layer as pl

    pl.setup_device()
    g = torch.Generator().manual_seed(0)
    for spec in args.shapes:
        M, K, N = (int(v) for v in spec.split("x"))
        x = torch.randn(M, K, generator=g).bfloat16()
        R = args.reps
        w = (torch.randn(R, K, N, generator=g) * K ** -0.5).bfloat16()  # a different weight per rep: no CSE
        sx = x.float().abs().amax() / 240.0
        sw = w.float().abs().amax() / 240.0
        x8 = (x.float() / sx).to(torch.float8_e4m3fn)
        w8 = (w.float() / sw).to(torch.float8_e4m3fn)

        def bf(x, w):
            acc = torch.zeros(M, N, dtype=torch.float32, device=x.device)
            for r in range(R):
                acc = acc + (x @ w[r]).float()
            return acc.sum(0)

        def f8(x8, w8, s):
            acc = torch.zeros(M, N, dtype=torch.float32, device=x8.device)
            for r in range(R):
                acc = acc + (x8.to(torch.bfloat16) @ w8[r].to(torch.bfloat16)).float() * (s[0] * s[1])
            return acc.sum(0)

        def f8dot(x8, w8, s):
            acc = torch.zeros(M, N, dtype=torch.float32, device=x8.device)
            for r in range(R):
                acc = acc + torch.matmul(x8, w8[r]).float() * (s[0] * s[1])
            return acc.sum(0)

        dev = pl.DEV
        s = torch.tensor([sx, sw], dtype=torch.float32)
        tb = pl.timed(f"bf16 {spec} x{R}", bf, (x.to(dev), w.to(dev)), args.iters)
        flop = 2.0 * M * K * N * R
        print(f"  -> bf16: {flop / tb / 1e12:.1f} TFLOP/s", flush=True)
        for name, fn in (("fp8 dot (f8 x f8 matmul in the graph)", f8dot), ("fp8 cast-to-bf16 then dot", f8)):
            try:
                t = pl.timed(f"{name} {spec} x{R}", fn, (x8.to(dev), w8.to(dev), s.to(dev)), args.iters)
                print(f"  -> {name}: {flop / t / 1e12:.1f} TFLOP/s ({tb / t:.2f}x bf16)", flush=True)
            except Exception as e:  # noqa: BLE001 - report what the compiler or runtime refused
                print(f"  -> {name}: failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
                continue
        ref = (x.float() @ w[0].float())
        y8 = (x8.float() @ w8[0].float()) * (sx * sw)
        print(f"  fp8 product rel error vs fp32: {((y8 - ref).abs().max() / ref.abs().max()).item():.4f}", flush=True)


if __name__ == "__main__":
    main()
