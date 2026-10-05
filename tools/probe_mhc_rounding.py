"""Whether the device keeps the bf16 rounding of the hyper-connection streams (models/hybrid.py _round, the
Veltkamp split of KILN_MHC_FORM=elementwise) at the row counts the sequence-parallel prefill streams run at
(models/decoder.py prefill_sp_enabled: R / tp rows per rank, 1 for a 32-token chunk at dp_attention 1 and
tp=32, 64 for the trn1 sweep's 2048-row prefill), against the replicated streams' R rows.

    python tools/probe_mhc_rounding.py [--rows 1 2 8 32 64 256 1024] [--chain 0]

Per row count, one device graph computes the collapse x = sum_n pre_n S_n in fp32 from bf16 streams S
[rows, 4, 4096] (as hybrid._mhc does), rounds it with _round(x, bf16, barrier=True) and returns the residual
z = round(x).float() - x in fp32, which an fp32 consumer of the rounded value would see. If the device keeps
the rounding, z is the bf16 rounding error, nonzero for nearly every element; if it folds the rounding away
(a plain f32 -> bf16 -> f32 pair may be folded inside a device graph, "Why elementwise mHC moved real-weight
ppl" in docs/neuron-notes.md, and an algebraic or fused evaluation of the split c - (c - x) does the same),
z is exactly 0. Printed: the share of zero residuals on the device and on the host, the device residual's
largest difference from the host's, and "rounded" / "FOLDED". With --chain N the probe also runs N
consecutive blocks (tools/probe_mhc_blocks.py) in one graph and reports their streams against the host run
with the streams kept bf16 and kept fp32 between blocks (informative only: after several blocks the device's
own transcendental functions differ from the host's as much as the rounding does).

Found 2026-10-04: GLM-5.3-Flash's 4-sentence ppl with sequence-parallel streams was -2.073 on trn1 and -1.807
on trn2 (Water boils -2.591, the signature of the rounding folded away); run this first on a trn2.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def streams_in(T: int, hc: int, H: int, outlier: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    S = torch.randn(T, hc, H, generator=g)
    idx = torch.randint(0, H, (hc, 8), generator=g)
    for m in range(hc):
        S[:, m, idx[m]] *= outlier  # trained residual streams carry a few large channels
    return S.bfloat16()


def residual(pre: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """round(x).float() - x for the mHC collapse x of streams S [T, hc, H] with weights pre [T, hc] fp32."""
    from kiln.models.hybrid import _round

    hc = S.shape[1]
    x = pre[:, 0:1] * S[:, 0].float()
    for n in range(1, hc):
        x = x + pre[:, n : n + 1] * S[:, n].float()
    return _round(x, torch.bfloat16, True).float() - x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 8, 32, 64, 256, 1024])
    ap.add_argument("--chain", type=int, default=0, help="also N consecutive blocks in one graph (informative)")
    ap.add_argument("--outlier", type=float, default=300.0)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--cpu", action="store_true", help="host only (check the script)")
    args = ap.parse_args()
    import profile_layer as pl

    pl.setup_device(args.cpu)
    from kiln.models import hybrid

    hybrid.MHC_FORM = "elementwise"  # the default form, the one with the rounding barrier
    hc, H = 4, args.hidden
    g = torch.Generator().manual_seed(1)
    for T in args.rows:
        S = streams_in(T, hc, H, args.outlier)
        pre = torch.sigmoid(torch.randn(T, hc, generator=g)) + 1e-6
        host = residual(pre, S)
        f = torch.compile(residual, **pl.OPTS)
        dev = f(pre.to(pl.DEV), S.to(pl.DEV)).cpu()
        z_dev = (dev == 0).float().mean().item()
        z_host = (host == 0).float().mean().item()
        d = (dev - host).abs().max().item()
        verdict = "FOLDED" if z_dev > 0.99 > z_host else ("rounded" if z_dev < 0.5 else "check")
        print(f"rows={T:5d} collapse residual round(x) - x: zero on the device {z_dev:.4f}, on the host {z_host:.4f}; "
              f"device vs host max |diff| {d:.3e} -> {verdict}", flush=True)
        if args.chain:
            import probe_mhc_blocks as pb

            model = pb.fake_model(hc, H)
            ps = pb.params(args.chain, hc, H)
            s2 = S.reshape(T, hc * H)
            rounded = pb.run(model, s2, pb.layers_of(ps)).double()
            folded = pb.run(model, s2.float(), pb.layers_of(ps)).double()  # fp32 streams: _round a no-op
            dev_layers = pb.layers_of([tuple(t.to(pl.DEV) for t in p) for p in ps])
            got = torch.compile(lambda x: pb.run(model, x, dev_layers), **pl.OPTS)(s2.to(pl.DEV)).cpu().double()
            scale = rounded.abs().max().item()
            print(f"rows={T:5d} {args.chain} blocks in one graph: device vs host-rounded "
                  f"{(got - rounded).abs().max().item() / scale:.2e}, vs host-fp32-streams "
                  f"{(got - folded).abs().max().item() / scale:.2e} (host runs differ by "
                  f"{(folded - rounded).abs().max().item() / scale:.2e})", flush=True)


if __name__ == "__main__":
    main()
