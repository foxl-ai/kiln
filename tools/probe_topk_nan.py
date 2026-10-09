"""Does a top-k over rows holding NaN give out-of-range indices on the device, and does the gather at those indices
fault? The MoE router's pattern in a GLM-5.3-Flash graph (models: scores = sigmoid(logits), top-8 of scores + bias,
the routing weights gathered from scores at the top-8 indices), on one logical core, rows 0 .. --nan-rows - 1 NaN.

    NEURON_RT_VISIBLE_CORES=0 python tools/probe_topk_nan.py [--rows 8] [--experts 288] [--nan-rows 2] [--inf]

Prints the indices of the NaN rows and whether the gather ran; a scatter/gather out-of-bound notification from the
runtime (nrta status 1006) means an index left [0, experts).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--experts", type=int, default=288)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--nan-rows", type=int, default=2)
    ap.add_argument("--inf", action="store_true", help="-inf rows instead of NaN")
    ap.add_argument("--zero", action="store_true", help="all-zero logits with a zero bias (every score tied)")
    ap.add_argument("--zero-logits", action="store_true", help="the NaN rows' logits zero instead, the bias kept")
    ap.add_argument("--no-gather", action="store_true", help="return the indices only (see what top-k gives)")
    ap.add_argument("--clamp", action="store_true", help="clamp the indices into [0, experts) before the gather")
    a = ap.parse_args()
    import profile_layer as pl

    pl.setup_device()
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(a.rows, a.experts, generator=g)
    bias = torch.randn(a.experts, generator=g) * 0.01
    if a.zero:
        logits.zero_()
        bias.zero_()
    bad = float("-inf") if a.inf else float("nan")
    logits[:a.nan_rows] = 0.0 if a.zero_logits else bad

    def router(lg, b):
        s = torch.sigmoid(lg.float())
        topi = torch.topk(s + b, a.k, dim=-1).indices
        if a.no_gather:
            return s[:, :a.k], topi
        if a.clamp:
            topi = topi.clamp(0, a.experts - 1)
        w = s.gather(1, topi)
        return w / (w.sum(-1, keepdim=True) + 1e-20), topi

    c = torch.compile(router, **pl.OPTS)
    try:
        w, topi = c(logits.to(pl.DEV), bias.to(pl.DEV))
        w, topi = w.cpu(), topi.cpu()
    except Exception as e:
        print(f"RESULT the router graph FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
        sys.exit(1)
    print(f"indices of the {'-inf' if a.inf else 'NaN'} rows: {topi[:a.nan_rows].tolist()}", flush=True)
    print(f"indices of row {a.nan_rows}: {topi[a.nan_rows].tolist()} (host: "
          f"{torch.topk(torch.sigmoid(logits[a.nan_rows]) + bias, a.k).indices.tolist()})", flush=True)
    out = int(((topi < 0) | (topi >= a.experts)).sum())
    print(f"RESULT ran; {out} indices out of range; weights finite in the clean rows: "
          f"{bool(torch.isfinite(w[a.nan_rows:]).all())}", flush=True)


if __name__ == "__main__":
    main()
