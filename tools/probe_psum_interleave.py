"""Minimal device check of PSUM accumulation chains with other matmul writes between their steps (trn1, one core).

Each case accumulates, for every column j of a [128, J] result, the products X_e^T 1 over E steps e (X_e [128, 128] bf16,
small integers, so every sum is exact), in one of four orders:
  same-outer  e outer, j inner, all J columns in ONE PSUM tensor (one bank): a column's chain is interleaved with the
              other columns' writes to the same bank, including their accumulate=False starts at e = 0
  same-inner  j outer, e inner, one PSUM tensor: each column's chain runs uninterleaved
  banks-outer e outer, j inner, every column its own PSUM tensor (own bank): interleaved across banks
  start-mid   same-inner, but an unrelated accumulate=False matmul into another column of the same tensor between two
              steps of a chain

    python tools/probe_psum_interleave.py [--simulate]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nki
import nki.isa as nisa
import nki.language as nl

E, J = 3, 4


@nki.jit
def psum_width(xs, W: int):
    """same-inner with W columns per chain: column block j of [128, J W] = sum_e xs[e][:, j-block]^T 1 (W ones)."""
    out = nl.ndarray((128, J * W), dtype=nl.float32, buffer=nl.shared_hbm)
    X = nl.ndarray((128, E, J * 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    for e in range(E):
        nisa.dma_copy(dst=X[:, e, :], src=xs[e])
    ones = nl.ndarray((128, W), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    p = nl.ndarray((128, J * W), dtype=nl.float32, buffer=nl.psum)
    for j in range(J):
        for e in range(E):
            nisa.nc_matmul(dst=p[:, j * W:(j + 1) * W], stationary=X[:, e, j * 128:(j + 1) * 128], moving=ones,
                           accumulate=e > 0)
    res = nl.ndarray((128, J * W), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=res, src=p)
    nisa.dma_copy(dst=out, src=res)
    return out


@nki.jit
def psum_case(xs, case: int):
    """xs bf16 [E, 128, J * 128] -> f32 [128, J]: column j = sum_e xs[e][:, j-block]^T 1 (per output partition)."""
    out = nl.ndarray((128, J), dtype=nl.float32, buffer=nl.shared_hbm)
    X = nl.ndarray((128, E, J * 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    for e in range(E):
        nisa.dma_copy(dst=X[:, e, :], src=xs[e])
    ones = nl.ndarray((128, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    res = nl.ndarray((128, J), dtype=nl.float32, buffer=nl.sbuf)
    if case == 2:  # banks-outer
        ps = []
        for j in range(J):
            ps.append(nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum))
        for e in range(E):
            for j in range(J):
                nisa.nc_matmul(dst=ps[j][:, 0:1], stationary=X[:, e, j * 128:(j + 1) * 128], moving=ones,
                               accumulate=e > 0)
        for j in range(J):
            nisa.tensor_copy(dst=res[:, j:j + 1], src=ps[j][:, 0:1])
    else:
        p = nl.ndarray((128, J), dtype=nl.float32, buffer=nl.psum)
        if case == 0:  # same-outer
            for e in range(E):
                for j in range(J):
                    nisa.nc_matmul(dst=p[:, j:j + 1], stationary=X[:, e, j * 128:(j + 1) * 128], moving=ones,
                                   accumulate=e > 0)
        else:  # same-inner (1) / start-mid (3)
            for j in range(J):
                for e in range(E):
                    nisa.nc_matmul(dst=p[:, j:j + 1], stationary=X[:, e, j * 128:(j + 1) * 128], moving=ones,
                                   accumulate=e > 0)
                    if case == 3 and j == 0 and e == 0:  # another column's chain starts in the middle of column 0's
                        nisa.nc_matmul(dst=p[:, J - 1:J], stationary=X[:, 0, (J - 1) * 128:J * 128], moving=ones,
                                       accumulate=False)
        nisa.tensor_copy(dst=res, src=p)
    nisa.dma_copy(dst=out, src=res)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    a = ap.parse_args()
    g = torch.Generator().manual_seed(0)
    xs = torch.randint(0, 4, (E, 128, J * 128), generator=g).to(torch.bfloat16)
    want = xs.float().view(E, 128, J, 128).sum(1).sum(0).t()  # [128 (out partition), J]
    names = ["same-outer", "same-inner", "banks-outer", "start-mid"]
    if not a.simulate:
        from kiln import platform

        platform.configure_runtime_env()
        import libtorch_neuronx_lite  # noqa: F401
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from kiln.engine.model_runner import canonical_neuron_backend, neuronx_cc_args
        dev = torch.device("neuron:0")
    for case in range(4):
        if a.simulate:
            got = torch.as_tensor(nki.simulate(psum_case)(xs=xs.float().numpy().astype(np.float32), case=case))
        else:
            f = lambda x, c=case: wrap_nki(psum_case)[1](xs=x, case=c)  # noqa: E731
            fc = torch.compile(f, backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                               options={"compiler_args": neuronx_cc_args(torch.bfloat16)})
            got = fc(xs.to(dev)).cpu()
        err = (got.float() - want).abs()
        print(f"{names[case]:12s}: max |d| {err.max().item():.1f} (values up to {want.max().item():.0f}); wrong columns "
              f"{sorted(set(err.nonzero()[:, 1].tolist()))}; partition 0 got {got[0].tolist()} want {want[0].tolist()}",
              flush=True)
    # per-step partials of partition 0, to read what a wrong column holds
    parts = xs.float().view(E, 128, J, 128).sum(1)[:, :, 0]  # [E, J] for output partition 0
    print("partition 0 per-step partials [e][j]:", parts.tolist(), flush=True)
    for W in (1, 2, 4, 8, 16):
        if a.simulate:
            got = torch.as_tensor(nki.simulate(psum_width)(xs=xs.float().numpy().astype(np.float32), W=W))
        else:
            f = lambda x, w=W: wrap_nki(psum_width)[1](xs=x, W=w)  # noqa: E731
            fc = torch.compile(f, backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                               options={"compiler_args": neuronx_cc_args(torch.bfloat16)})
            got = fc(xs.to(dev)).cpu()
        wantw = want.repeat_interleave(W, dim=1)
        err = (got.float() - wantw).abs()
        print(f"same-inner W={W:2d}: max |d| {err.max().item():.1f}; wrong chains "
              f"{sorted(set((err.nonzero()[:, 1] // W).tolist()))}", flush=True)


if __name__ == "__main__":
    main()
