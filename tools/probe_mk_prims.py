"""Device checks of primitives that could take dequantization work off the vector and scalar engines (trn1, one
NeuronCore, nki 0.6.0), each timed and checked against torch:

    python tools/probe_mk_prims.py [--cases gpsimd_ts dma_cast dq_alone dq_pe]

gpsimd_ts: tensor_scalar on GpSimd (engine=nisa.gpsimd_engine) multiplying bf16 [128, 128] tiles by a per-partition
    fp32 scale into bf16 (GpSimd refuses fp8 operands and tensor_tensor: docs/neuron-notes.md "Expert parallelism").
dma_cast: dma_copy from fp8 codes in HBM into a bf16 SBUF tile (the DMA engines convert through fp32: nki.isa.dma_copy
    docstring), its rate against a plain uint8 copy, and whether the values are exact.
dq_alone / dq_pe: N fp8 -> bf16 tile dequantizations (alternately vector and scalar, as kiln_moe_ep_kernel) alone, and
    while the tensor engine streams a matmul per tile and DMAs land in SBUF, to see whether the in-kernel cost
    (~240 ns vector / ~180 ns scalar per tile in kiln_moe_ep_kernel's profile against 120-150 ns standalone) is
    contention.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def k_gpsimd_ts(w, s, reps: int, eng: int):
        """reps x 16 tensor_scalar ops of a bf16 [128, 128] tile times a per-partition fp32 scale into bf16 slices of one
        tile (eng 0 vector, 1 GpSimd), stored at the end."""
        f32, bf16 = nl.float32, nl.bfloat16
        wt = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=w)
        st = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=st, src=s)
        big = nl.ndarray((128, 16, 128), dtype=bf16, buffer=nl.sbuf)
        for r in range(reps):
            for i in range(16):
                nisa.tensor_scalar(dst=big[:, i, :], data=wt, op0=nl.multiply, operand0=st,
                                   engine=nisa.gpsimd_engine if eng else nisa.vector_engine)
        out = nl.ndarray((128, 16, 128), dtype=bf16, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=big)
        return out

    @nki.jit
    def k_dma_cast(w, NCH: int, cast: int):
        """NCH chunks of w uint8 [NCH, 128, 8192] (fp8 codes) into SBUF, as bf16 (cast: the DMA converts) or uint8, through
        4 buffers; returns the last chunk as it landed."""
        u8, bf16 = nl.uint8, nl.bfloat16
        bufs = []
        for _ in range(4):
            bufs.append(nl.ndarray((128, 8192), dtype=bf16 if cast else u8, buffer=nl.sbuf))
        acc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        for k in range(NCH):
            b = bufs[k % 4]
            src = w[k].view(nl.float8_e4m3) if cast else w[k]
            nisa.dma_copy(dst=b, src=src)
            t = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=t, src=b[:, 0:1], engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=t, op=nl.add, engine=nisa.vector_engine)
        out = nl.ndarray((128, 8192), dtype=bf16 if cast else u8, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=bufs[(NCH - 1) % 4])
        return out

    @nki.jit
    def k_dq(w, s, x, N: int, pe: int):
        """N fp8 [128, 128] tile dequantizations of w uint8 [128, 2048] (16 tiles, reused round robin) into bf16 slices,
        alternately vector and scalar (kiln_moe_ep_kernel's _dq_tile); pe: meanwhile one matmul of 256 moving columns per tile
        (x [128, 256] bf16 moving, one per tile) stream on the tensor engine and a 256 KB DMA lands per 16 tiles."""
        f32, bf16, fp8, u8 = nl.float32, nl.bfloat16, nl.float8_e4m3, nl.uint8
        wt = nl.ndarray((128, 2048), dtype=u8, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=w)
        st = nl.ndarray((128, 16), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=st, src=s)
        xt = nl.ndarray((128, 512), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=x)
        dq = nl.ndarray((128, 16, 128), dtype=bf16, buffer=nl.sbuf)
        acc = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
        lb = nl.ndarray((128, 2048), dtype=u8, buffer=nl.sbuf)
        nisa.nc_matmul(dst=acc[:, 0:256], stationary=xt[:, 0:128], moving=xt[:, 0:256], accumulate=False)
        for n in range(N):
            j = n % 16
            if n % 2 == 0:
                nisa.tensor_scalar(dst=dq[:, j, :], data=wt[:, j * 128:(j + 1) * 128].view(fp8), op0=nl.multiply,
                                   operand0=st[:, j:j + 1], engine=nisa.vector_engine)
            else:
                nisa.activation(dst=dq[:, j, :], op=nl.copy, data=wt[:, j * 128:(j + 1) * 128].view(fp8),
                                scale=st[:, j:j + 1])
            if pe:
                nisa.nc_matmul(dst=acc[:, 0:256], stationary=xt[:, (n % 4) * 128:(n % 4 + 1) * 128], moving=xt[:, 0:256],
                               accumulate=True)
                if n % 16 == 0:
                    nisa.dma_copy(dst=lb, src=w)
        out = nl.ndarray((128, 16, 128), dtype=bf16, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=dq)
        o2 = nl.ndarray((128, 256), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o2, src=acc[:, 0:256], engine=nisa.vector_engine)
        out2 = nl.ndarray((128, 256), dtype=f32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out2, src=o2)
        return out, out2


def main() -> None:
    import profile_layer as pl
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from kiln.models.quant import FP8

    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["gpsimd_ts", "dma_cast", "dq_alone", "dq_pe"])
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    pl.setup_device()
    g = torch.Generator().manual_seed(0)
    null = pl.timed("null graph", lambda h: h + 1, (torch.zeros(4, 128).to(pl.DEV),), a.iters)
    if "gpsimd_ts" in a.cases:
        w = torch.randn(128, 128, generator=g).bfloat16()
        s = (torch.rand(128, 1, generator=g) + 0.5).float()
        ref = (w.float() * s).bfloat16()
        for eng in (0, 1):
            for reps in (4, 16):
                def f(w_, s_, reps=reps, eng=eng):
                    return wrap_nki(k_gpsimd_ts)[1](w=w_, s=s_, reps=reps, eng=eng)
                try:
                    out = torch.compile(f, **pl.OPTS)(w.to(pl.DEV), s.to(pl.DEV)).cpu()
                except Exception as e:
                    print(f"-> gpsimd_ts eng={eng}: FAILED {type(e).__name__}: {str(e)[:400]}", flush=True)
                    break
                t = pl.timed(f"tensor_scalar eng={eng} reps={reps}", f, (w.to(pl.DEV), s.to(pl.DEV)), a.iters)
                print(f"-> tensor_scalar bf16 [128, 128] x scale on {'GpSimd' if eng else 'vector'}: "
                      f"{(t - null) / (16 * reps) * 1e9:.0f} ns per tile; exact {bool(torch.equal(out[:, 0], ref))}",
                      flush=True)
    if "dma_cast" in a.cases:
        NCH = 72  # 72 x 1 MB
        codes = torch.randint(0, 256, (NCH, 128, 8192), dtype=torch.uint8, generator=g)
        codes[(codes & 0x7F) >= 0x78] = 0x10  # trn1 e4m3 has no values past 240 (fit_e4m3_max) and no NaN codes
        for cast in (0, 1):
            def f(w_, cast=cast):
                return wrap_nki(k_dma_cast)[1](w=w_, NCH=NCH, cast=cast)
            try:
                out = torch.compile(f, **pl.OPTS)(codes.to(pl.DEV)).cpu()
            except Exception as e:
                print(f"-> dma_cast cast={cast}: FAILED {type(e).__name__}: {str(e)[:400]}", flush=True)
                continue
            t = pl.timed(f"dma {'fp8 -> bf16' if cast else 'uint8'}", f, (codes.to(pl.DEV),), a.iters)
            exact = ""
            if cast:
                ref = codes[NCH - 1].view(FP8).float().bfloat16()
                exact = f"; exact {bool(torch.equal(out, ref))} ({int((out != ref).sum())} differ)"
            print(f"-> DMA of {NCH} MB fp8 codes {'cast to bf16 in SBUF' if cast else 'as uint8'}: "
                  f"{NCH * 128 * 8192 / (t - null) / 1e9:.0f} GB/s of source bytes{exact}", flush=True)
    for case in ("dq_alone", "dq_pe"):
        if case not in a.cases:
            continue
        w = torch.randint(0, 0x70, (128, 2048), dtype=torch.uint8, generator=g)
        s = (torch.rand(128, 16, generator=g) + 0.5).float()
        x = torch.randn(128, 512, generator=g).bfloat16()
        for N in (512, 2048):
            def f(w_, s_, x_, N=N, pe=int(case == "dq_pe")):
                return wrap_nki(k_dq)[1](w=w_, s=s_, x=x_, N=N, pe=pe)
            t = pl.timed(f"{case} N={N}", f, (w.to(pl.DEV), s.to(pl.DEV), x.to(pl.DEV)), a.iters)
            print(f"-> {case}: {N} dequantized tiles in {(t - null) * 1e6:.0f} us, {(t - null) / N * 1e9:.0f} ns per tile "
                  f"(both engines){'; with a 256-column matmul per tile and a 256 KB DMA per 16 tiles' if case == 'dq_pe' else ''}",
                  flush=True)


if __name__ == "__main__":
    main()
