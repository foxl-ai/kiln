"""The hyper-connection output mix alone (models/hybrid.py mix_out: stream n = post_n y + sum_m comb[m, n]
S_m) on one NeuronCore against the host, per KILN_MHC_FORM, at GLM-5.3-Flash's widths (4 streams of
4096): whether the device computes what the host computes for the same code.

    python tools/probe_mhc_output.py [--rows 256 1024] [--outlier 300]

Streams in bf16 with a few large channels (--outlier: the magnitude of 8 channels per stream, as trained
residual streams carry), y bf16, post in (0, 2), comb doubly stochastic by Sinkhorn on random logits.
Reported per form: max abs error of the device output against the host's output of the same form
(relative to the output's max) and against an fp64 evaluation of the formula, and, as a check for a
transposed read, against the formula with comb transposed.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def inputs(T: int, hc: int, H: int, outlier: float, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(T, hc, H, generator=g)
    idx = torch.randint(0, H, (hc, 8), generator=g)
    for m in range(hc):
        S[:, m, idx[m]] *= outlier
    y = torch.randn(T, H, generator=g) * 2
    post = 2 * torch.sigmoid(torch.randn(T, hc, generator=g))
    comb = torch.softmax(torch.randn(T, hc, hc, generator=g) * 3, dim=-1)
    for _ in range(20):
        comb = comb / comb.sum(-2, keepdim=True)
        comb = comb / comb.sum(-1, keepdim=True)
    return post, comb, y.bfloat16(), S.bfloat16()


def _bound(fn, form):
    def f(p, c, y, s):  # (no default arguments: dynamo rejected a lambda with one, DefaultsSource)
        return fn(p, c, y, s, form)
    return f


def _chained(fn, form):
    def f(p, c, y, s):
        return fn(p, c, y, s, form).float() * 1.5
    return f


def _unrounded(post, comb, y, S):
    """The fp32 output mix without its final rounding to bf16."""
    T, hc, H = S.shape
    outs = []
    for n in range(hc):
        o = post[:, n : n + 1] * y.float()
        for m in range(hc):
            o = o + comb[:, m, n : n + 1] * S[:, m].float()
        outs.append(o)
    return torch.cat(outs, dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--outlier", type=float, default=300.0)
    ap.add_argument("--hidden", type=int, default=4096)
    args = ap.parse_args()
    import profile_layer as pl

    pl.setup_device()
    from kiln.models.hybrid import MHC_FORMS, mix_out

    hc, H = 4, args.hidden
    for T in args.rows:
        post, comb, y, S = inputs(T, hc, H, args.outlier)
        exact = (post.double().unsqueeze(-1) * y.double().unsqueeze(1)
                 + comb.double().transpose(-1, -2) @ S.double()).reshape(T, hc * H)
        swapped = (post.double().unsqueeze(-1) * y.double().unsqueeze(1) + comb.double() @ S.double()).reshape(T, hc * H)
        scale = exact.abs().max().item()
        dev = tuple(x.to(pl.DEV) for x in (post, comb, y, S))
        for form in MHC_FORMS:
            host = mix_out(post, comb, y, S, form).double()
            f = torch.compile(_bound(mix_out, form), **pl.OPTS)
            got = f(*dev).cpu().double()
            # The same followed by an fp32 op, as the next block reads the streams: if the compiler drops
            # the bf16 rounding of the streams (XLA's allow-excess-precision folding of f32 -> bf16 -> f32),
            # the chained result is the unrounded fp32 one.
            g = torch.compile(_chained(mix_out, form), **pl.OPTS)
            chained = g(*dev).cpu().double()
            unrounded = _unrounded(post, comb, y, S).double()
            print(f"T={T} {form:<17} chained (x 1.5): vs host-rounded {(chained - 1.5 * host).abs().max().item() / scale:.2e}, "
                  f"vs unrounded fp32 {(chained - 1.5 * unrounded).abs().max().item() / scale:.2e}", flush=True)
            e_host = (got - host).abs().max().item() / scale
            e_exact = (got - exact).abs().max().item() / scale
            e_host_exact = (host - exact).abs().max().item() / scale
            e_swap = (got - swapped).abs().max().item() / scale
            print(f"T={T} {form:<17} device vs host {e_host:.2e}, device vs fp64 {e_exact:.2e} (host vs fp64 "
                  f"{e_host_exact:.2e}), device vs comb-transposed formula {e_swap:.2e}", flush=True)


if __name__ == "__main__":
    main()
