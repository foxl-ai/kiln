"""The linear-attention layer's short causal conv (models/linear_attn.py _causal_conv) alone on one NeuronCore: the default XLA
lowering and the NKI kernel (kernels/short_conv.py, KILN_LA_CONV_KERNEL=nki) against the host, bit for bit, and their time.

    NEURON_RT_VISIBLE_CORES=0 python tools/probe_short_conv.py [--rows 1024] [--channels 3072]

At LNC=2 (trn2) the kernel runs at grid 2, each physical core its half of the channel tiles.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--channels", type=int, default=3072)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    import profile_layer as pl

    pl.setup_device()
    from kiln.kernels import short_conv as sc
    from kiln.models import linear_attn as la

    g = torch.Generator().manual_seed(0)
    T, C, K = a.rows, a.channels, 4
    xe = (torch.randn(K - 1 + T, C, generator=g) * 0.5).bfloat16()
    w = (torch.randn(C, K, generator=g) * 0.3).bfloat16()
    want = la._causal_conv(xe, w, T)
    for name in ("xla", "nki"):
        sc.KERNEL = name  # read at trace time by short_conv.takes

        def f(x, ww):
            return la._causal_conv(x, ww, T)

        c = torch.compile(f, **pl.OPTS)
        args = (xe.to(pl.DEV), w.to(pl.DEV))
        got = c(*args).cpu()
        bad = ((got != want).any(-1)).nonzero().flatten().tolist()
        c(*args).cpu()
        t = time.perf_counter()
        for _ in range(a.iters):
            out = c(*args)
        out.cpu()
        ms = (time.perf_counter() - t) / a.iters * 1e3
        print(f"RESULT {name}: bit-equal to the host {torch.equal(got, want)}, rows differing {len(bad)} (first {bad[:8]}), "
              f"max |d| {(got - want).abs().max().item():.4g}; {ms:.3f} ms per call chained incl. transfers", flush=True)
        torch._dynamo.reset()


if __name__ == "__main__":
    main()
