"""Cost of the primitives a grouped-GEMM (prefill) MoE kernel is built from, on one NeuronCore-v2
(trn1), each as a small NKI kernel inside an LNL graph:

    python tools/probe_prefill_prims.py [--tiles 64] [--hidden 4096] [--only gather scatter ...]

- gather / scatter: 128 rows of [., H] bf16 per dma_copy with a per-partition index
  (vector_offset, software DGE on trn1), against the same bytes as contiguous static DMAs;
- scatter_add / gather_add: the same through dma_compute's read-modify-write;
- small_scatter: 128 rows of 32 bytes per DMA (the descriptor cost alone);
- transpose: [128, H] bf16 tiles transposed on the tensor engine, PSUM drained by the scalar or
  the vector engine;
- matmul: fp8 stationary [128, 128] x bf16 moving [128, N], 32 accumulating tiles per output,
  drained by the scalar engine;
- drain: PSUM [128, 512] fp32 -> SBUF bf16 by the scalar engine (activation with a per-partition
  scale) and by the vector engine (tensor_scalar);
- ngather: nc_n_gather (GpSimd, a free-dim gather inside each partition).

Each kernel runs its body `--tiles` times; the time per tile is (p50 - null graph) / tiles.
The gathers and scatters are checked against torch on the host.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from nki.isa.constants import oob_mode
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def p_gather(x, idx):
        """out[t] = x[idx[:, t]] rows, NT tiles of 128 rows; returns the last tile and a checksum."""
        C, H = x.shape
        NT = idx.shape[1]
        out = nl.ndarray((128, NT, H), dtype=x.dtype, buffer=nl.shared_hbm)
        i_sb = nl.ndarray((128, NT), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=i_sb, src=idx)
        for t in range(NT):
            b = nl.ndarray((128, H), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=b, src=x.ap(pattern=[[H, 128], [1, H]], offset=0,
                                          vector_offset=i_sb.ap(pattern=[[NT, 128], [1, 1]], offset=t),
                                          indirect_dim=0))
            nisa.dma_copy(dst=out[:, t, :], src=b)
        return out

    @nki.jit
    def p_gather_only(x, idx):
        """NT gathers into rotating SBUF buffers; one tiny output depends on every one."""
        C, H = x.shape
        NT = idx.shape[1]
        out = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        i_sb = nl.ndarray((128, NT), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=i_sb, src=idx)
        acc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        for t in range(NT):
            b = nl.ndarray((128, H), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=b, src=x.ap(pattern=[[H, 128], [1, H]], offset=0,
                                          vector_offset=i_sb.ap(pattern=[[NT, 128], [1, 1]], offset=t),
                                          indirect_dim=0))
            nisa.tensor_tensor(dst=acc, data1=acc, data2=b[:, H - 1:H], op=nl.add)
        nisa.dma_copy(dst=out, src=acc)
        return out

    @nki.jit
    def p_copy_only(x, nt: int):
        """The same bytes as p_gather_only with static (contiguous) row DMAs."""
        C, H = x.shape
        out = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        acc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        for t in range(nt):
            b = nl.ndarray((128, H), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=b, src=x[t * 128:(t + 1) * 128, :])
            nisa.tensor_tensor(dst=acc, data1=acc, data2=b[:, H - 1:H], op=nl.add)
        nisa.dma_copy(dst=out, src=acc)
        return out

    @nki.jit
    def p_scatter(x, idx):
        """out[idx[:, t]] = x[t * 128 + p] (static loads, indirect stores); idx a permutation."""
        C, H = x.shape
        NT = idx.shape[1]
        out = nl.ndarray((C, H), dtype=x.dtype, buffer=nl.shared_hbm)
        i_sb = nl.ndarray((128, NT), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=i_sb, src=idx)
        for t in range(NT):
            b = nl.ndarray((128, H), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=b, src=x[t * 128:(t + 1) * 128, :])
            nisa.dma_copy(dst=out.ap(pattern=[[H, 128], [1, H]], offset=0,
                                     vector_offset=i_sb.ap(pattern=[[NT, 128], [1, 1]], offset=t),
                                     indirect_dim=0), src=b)
        return out

    @nki.jit
    def p_scatter_add(x, base, idx):
        """out = base; out[idx[:, t]] += x[t * 128 + p] by dma_compute read-modify-write."""
        C, H = x.shape
        NT = idx.shape[1]
        out = nl.ndarray((C, H), dtype=base.dtype, buffer=nl.shared_hbm)
        for t in range(C // 128):
            b0 = nl.ndarray((128, H), dtype=base.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=b0, src=base[t * 128:(t + 1) * 128, :])
            nisa.dma_copy(dst=out[t * 128:(t + 1) * 128, :], src=b0)
        i_sb = nl.ndarray((128, NT), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=i_sb, src=idx)
        for t in range(NT):
            b = nl.ndarray((128, H), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=b, src=x[t * 128:(t + 1) * 128, :])
            dst = out.ap(pattern=[[H, 128], [1, H]], offset=0,
                         vector_offset=i_sb.ap(pattern=[[NT, 128], [1, 1]], offset=t), indirect_dim=0)
            nisa.dma_compute(dst=dst, srcs=[dst, b], reduce_op=nl.add)
        return out

    @nki.jit
    def p_gather_add(x, idx):
        """acc (SBUF fp32) += x[idx[:, t]] for every t, by dma_compute; returns acc."""
        C, H = x.shape
        NT = idx.shape[1]
        out = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.shared_hbm)
        i_sb = nl.ndarray((128, NT), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=i_sb, src=idx)
        acc = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        for t in range(NT):
            src = x.ap(pattern=[[H, 128], [1, H]], offset=0,
                       vector_offset=i_sb.ap(pattern=[[NT, 128], [1, 1]], offset=t), indirect_dim=0)
            nisa.dma_compute(dst=acc, srcs=[acc, src], reduce_op=nl.add)
        nisa.dma_copy(dst=out, src=acc)
        return out

    @nki.jit
    def p_store(x, nt: int, mode: int):
        """nt [128, H] SBUF tiles (loaded once) stored to HBM rows t * 128 + p: mode 0 into the
        kernel output, 1 into a private HBM scratch (read back at the end), 2 HBM -> HBM copies
        of x rows into the output, 3 like 0 with the tile reloaded each time (load + store)."""
        C, H = x.shape
        out = nl.ndarray((C, H), dtype=x.dtype, buffer=nl.shared_hbm)
        scratch = nl.ndarray((C, H), dtype=x.dtype, buffer=nl.private_hbm)
        b = nl.ndarray((128, H), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=b, src=x[0:128, :])
        for t in range(nt):
            r = (t % (C // 128)) * 128
            if mode == 0:
                nisa.dma_copy(dst=out[r:r + 128, :], src=b)
            elif mode == 1:
                nisa.dma_copy(dst=scratch[r:r + 128, :], src=b)
            elif mode == 2:
                nisa.dma_copy(dst=out[r:r + 128, :], src=x[r:r + 128, :])
            else:
                b2 = nl.ndarray((128, H), dtype=x.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=b2, src=x[r:r + 128, :])
                nisa.dma_copy(dst=out[r:r + 128, :], src=b2)
        if mode == 1:
            for t in range(C // 128):
                b3 = nl.ndarray((128, H), dtype=x.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=b3, src=scratch[t * 128:(t + 1) * 128, :])
                nisa.dma_copy(dst=out[t * 128:(t + 1) * 128, :], src=b3)
        return out

    @nki.jit
    def p_gather_sum(x, idx):
        """Every gathered tile consumed whole: acc[p, :] += row (vector engine), so no DMA can be
        narrowed to the columns read."""
        C, H = x.shape
        NT = idx.shape[1]
        out = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.shared_hbm)
        i_sb = nl.ndarray((128, NT), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=i_sb, src=idx)
        acc = nl.ndarray((128, H), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        for t in range(NT):
            b = nl.ndarray((128, H), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=b, src=x.ap(pattern=[[H, 128], [1, H]], offset=0,
                                          vector_offset=i_sb.ap(pattern=[[NT, 128], [1, 1]], offset=t),
                                          indirect_dim=0))
            nisa.tensor_tensor(dst=acc, data1=acc, data2=b, op=nl.add)
        nisa.dma_copy(dst=out, src=acc)
        return out

    @nki.jit
    def p_small_scatter(w, idx):
        """out[idx[:, t]] = w[t * 128 + p], rows of w.shape[1] elements (the descriptor cost)."""
        C, F = w.shape
        NT = idx.shape[1]
        out = nl.ndarray((C, F), dtype=w.dtype, buffer=nl.shared_hbm)
        i_sb = nl.ndarray((128, NT), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=i_sb, src=idx)
        wb = nl.ndarray((128, NT, F), dtype=w.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=wb, src=w.reshape((NT, 128, F)).ap(pattern=[[F, 128], [128 * F, NT], [1, F]], offset=0))
        for t in range(NT):
            nisa.dma_copy(dst=out.ap(pattern=[[F, 128], [1, F]], offset=0,
                                     vector_offset=i_sb.ap(pattern=[[NT, 128], [1, 1]], offset=t),
                                     indirect_dim=0), src=wb[:, t, :])
        return out

    @nki.jit
    def p_transpose(x, nt: int, eng: int):
        """nt tiles x[t*128:(t+1)*128, :] transposed to [128 h, c, 128 rows] on the tensor engine,
        4 per PSUM bank, drained by the scalar (eng 0) or vector (eng 1) engine."""
        C, H = x.shape
        CT = H // 128
        out = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        acc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        for t in range(nt):
            b = nl.ndarray((128, H), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=b, src=x[(t % (C // 128)) * 128:((t % (C // 128)) + 1) * 128, :])
            xt = nl.ndarray((128, CT, 128), dtype=x.dtype, buffer=nl.sbuf)
            for g in range(CT // 4):
                ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                for j in range(4):
                    c = g * 4 + j
                    nisa.nc_transpose(dst=ps[:, j * 128:(j + 1) * 128], data=b[:, c * 128:(c + 1) * 128],
                                      engine=nisa.tensor_engine)
                if eng == 0:
                    nisa.activation(dst=xt[:, g * 4:(g + 1) * 4, :], op=nl.copy, data=ps)
                else:
                    nisa.tensor_copy(dst=xt[:, g * 4:(g + 1) * 4, :], src=ps, engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=xt[:, CT - 1, 127:128], op=nl.add)
        nisa.dma_copy(dst=out, src=acc)
        return out

    @nki.jit
    def p_matmul(w8, xT, nt: int):
        """nt outputs psum[128, N] = sum_c w8[:, c]^T xT[:, c] (fp8 stationary, bf16 moving), 32
        tiles each, drained by the scalar engine."""
        _, F = w8.shape
        CT = F // 128
        _, _, N = xT.shape
        out = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        w_sb = nl.ndarray((128, F), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_sb, src=w8)
        wf = w_sb.view(nl.float8_e4m3)
        x_sb = nl.ndarray((128, CT, N), dtype=xT.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_sb, src=xT)
        acc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        for t in range(nt):
            ps = nl.ndarray((128, N), dtype=nl.float32, buffer=nl.psum)
            for c in range(CT):
                nisa.nc_matmul(dst=ps, stationary=wf[:, c * 128:(c + 1) * 128], moving=x_sb[:, c, :],
                               accumulate=(c > 0))
            o = nl.ndarray((128, N), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=o, op=nl.copy, data=ps)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=o[:, N - 1:N], op=nl.add)
        nisa.dma_copy(dst=out, src=acc)
        return out

    @nki.jit
    def p_matmul_k64(a, wd, nt: int, tiled: int):
        """Down-projection shape: psum[128 lanes, 512] = a[64, 128]^T wd[64, 512] per 512 columns of H,
        H/512 per output, nt outputs; tiled 1 runs pairs of K=64 matmuls in the two row tiles."""
        _, H = wd.shape
        out = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        a_sb = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_sb, src=a)
        w_sb = nl.ndarray((128, H), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_sb, src=wd)
        wf = w_sb.view(nl.float8_e4m3)
        acc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        for t in range(nt):
            o = nl.ndarray((128, H), dtype=nl.bfloat16, buffer=nl.sbuf)
            for g in range(H // 512):
                ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                if tiled:
                    half = g % 2
                    nisa.nc_matmul(dst=ps, stationary=a_sb[half * 64:(half + 1) * 64, :],
                                   moving=wf[half * 64:(half + 1) * 64, g * 512:(g + 1) * 512],
                                   tile_position=(half * 64, 0), tile_size=(64, 128))
                else:
                    nisa.nc_matmul(dst=ps, stationary=a_sb[0:64, :], moving=wf[0:64, g * 512:(g + 1) * 512])
                nisa.activation(dst=o[:, g * 512:(g + 1) * 512], op=nl.copy, data=ps)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=o[:, H - 1:H], op=nl.add)
        nisa.dma_copy(dst=out, src=acc)
        return out

    @nki.jit
    def p_drain(w, nt: int, eng: int):
        """nt PSUM [128, 512] fp32 tiles (each from one tiny matmul) -> SBUF bf16, by the scalar
        engine (eng 0: activation copy with a per-partition scale), the vector engine (eng 1:
        tensor_scalar), or both alternating (eng 2)."""
        out = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        w_sb = nl.ndarray((1, 128), dtype=w.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_sb, src=w)
        ones = nl.ndarray((1, 512), dtype=w.dtype, buffer=nl.sbuf)
        nisa.memset(dst=ones, value=1.0)
        sc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=sc, value=0.5)
        acc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps, stationary=w_sb, moving=ones)
        for t in range(nt):
            o = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            if eng == 0 or (eng == 2 and t % 2 == 0):
                nisa.activation(dst=o, op=nl.copy, data=ps, scale=sc)
            else:
                nisa.tensor_scalar(dst=o, data=ps, op0=nl.multiply, operand0=sc, engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=o[:, 511:512], op=nl.add)
        nisa.dma_copy(dst=out, src=acc)
        return out

    @nki.jit
    def p_dequant(w8, s, nt: int, eng: int, bk: int):
        """nt dequantizations of an fp8 [128, H] weight by per-(row, bk-column block) fp32 scales
        [128, H / bk], broadcast over each block by a stride-0 access pattern, into bf16, on the
        GpSimd (eng 0) or vector (eng 1) engine."""
        _, H = w8.shape
        nb = H // bk
        out = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        w_sb = nl.ndarray((128, H), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_sb, src=w8)
        s_sb = nl.ndarray((128, nb), dtype=s.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=s_sb, src=s)
        wf = w_sb.view(nl.float8_e4m3)
        acc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        for t in range(nt):
            o = nl.ndarray((128, nb, bk), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=o, data1=wf.reshape((128, nb, bk)),
                               data2=s_sb.ap(pattern=[[nb, 128], [1, nb], [0, bk]], offset=0), op=nl.multiply,
                               engine=nisa.gpsimd_engine if eng == 0 else nisa.vector_engine)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=o[:, nb - 1, bk - 1:bk], op=nl.add)
        o2 = nl.ndarray((128, nb, bk), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o2, data1=wf.reshape((128, nb, bk)),
                           data2=s_sb.ap(pattern=[[nb, 128], [1, nb], [0, bk]], offset=0), op=nl.multiply)
        res = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        full = nl.ndarray((128, H), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=full, src=o2.reshape((128, H)))
        nisa.dma_copy(dst=out, src=acc)
        return full

    @nki.jit
    def p_act_width(w, nt: int, width: int):
        """PSUM [128, 512] fp32 -> SBUF bf16 by the scalar engine with a per-partition scale, as
        512 / width instructions of `width` columns each (the per-instruction overhead)."""
        out = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        w_sb = nl.ndarray((1, 128), dtype=w.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_sb, src=w)
        ones = nl.ndarray((1, 512), dtype=w.dtype, buffer=nl.sbuf)
        nisa.memset(dst=ones, value=1.0)
        sc = nl.ndarray((128, 4), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=sc, value=0.5)
        acc = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps, stationary=w_sb, moving=ones)
        for t in range(nt):
            o = nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            for j in range(512 // width):
                nisa.activation(dst=o[:, j * width:(j + 1) * width], op=nl.copy,
                                data=ps[:, j * width:(j + 1) * width], scale=sc[:, j % 4:j % 4 + 1])
            nisa.tensor_tensor(dst=acc, data1=acc, data2=o[:, 0:1], op=nl.add)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=o[:, 511:512], op=nl.add)
        nisa.dma_copy(dst=out, src=acc)
        return out

    @nki.jit
    def p_ngather(data, ind, nt: int):
        """nt nc_n_gather calls: dst[p, j] = data[p, ind[p, j]] (GpSimd, inside each partition)."""
        _, F = data.shape
        _, n = ind.shape
        out = nl.ndarray((128, n), dtype=data.dtype, buffer=nl.shared_hbm)
        d_sb = nl.ndarray((128, F), dtype=data.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=d_sb, src=data)
        i_sb = nl.ndarray((128, n), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(dst=i_sb, src=ind)
        o = nl.ndarray((128, n), dtype=data.dtype, buffer=nl.sbuf)
        for t in range(nt):
            nisa.nc_n_gather(dst=o, data=d_sb, indices=i_sb)
        nisa.dma_copy(dst=out, src=o)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    import profile_layer as pl
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    pl.setup_device()
    dev = pl.DEV
    NT, H = args.tiles, args.hidden
    C = NT * 128
    g = torch.Generator().manual_seed(0)
    x = torch.randn(C, H, generator=g).bfloat16()
    perm = torch.randperm(C, generator=g)
    idx = perm.view(NT, 128).T.contiguous().to(torch.int32)  # [128, NT]: tile t, partition p -> perm[t * 128 + p]
    xd, idxd = x.to(dev), idx.to(dev)
    null = pl.timed("null graph", lambda h: h + 1, (torch.zeros(4, H, dtype=torch.bfloat16).to(dev),), args.iters)
    rows_mb = C * H * 2 / 2**20

    def run(name, kernel, kwargs, per, check=None):
        if args.only and name.split(" ")[0] not in args.only:
            return
        call = wrap_nki(kernel)[1]
        tensors = {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
        static = {k: v for k, v in kwargs.items() if not isinstance(v, torch.Tensor)}
        names = tuple(tensors)

        def f(*a):
            return call(**dict(zip(names, a)), **static)

        def f_small(*a):  # read back 128 elements, not the whole output (PCIe would dominate)
            return f(*a).reshape(-1)[:128] + 0

        t = pl.timed(name, f_small, tuple(tensors.values()), args.iters)
        if t == t:
            pl.say(f"    -> {(t - null) / per * 1e6:9.3f} us per unit ({per} units)", flush=True)
        if check is not None and t == t:
            got = torch.compile(f, **pl.OPTS)(*tensors.values()).cpu()
            pl.say(f"    check: {check(got)}", flush=True)

    rows = lambda: x[perm].view(NT, 128, H).permute(1, 0, 2)  # noqa: E731
    run(f"gather {NT} x 128 rows of {H} bf16 ({rows_mb:.0f} MB), to HBM", p_gather, dict(x=xd, idx=idxd), NT,
        lambda o: f"exact {torch.equal(o, rows())}")
    for mode, nm in ((0, "SBUF -> output"), (1, "SBUF -> private HBM"), (2, "HBM -> HBM"), (3, "load + store")):
        run(f"store {nm}", p_store, dict(x=xd, nt=NT, mode=mode), NT,
            (lambda o: f"exact {torch.equal(o, x)}") if mode >= 2 else None)
    run("gather_sum (whole tiles consumed)", p_gather_sum, dict(x=xd, idx=idxd), NT,
        lambda o: f"max abs err {(o - x[perm].view(NT, 128, H).float().sum(0)).abs().max().item():.3e}")
    run("gather_only (into SBUF)", p_gather_only, dict(x=xd, idx=idxd), NT)
    run("copy_only (static rows, same bytes)", p_copy_only, dict(x=xd, nt=NT), NT)
    want_sc = torch.empty_like(x)
    want_sc[perm] = x
    run("scatter (static load, indirect store)", p_scatter, dict(x=xd, idx=idxd), NT,
        lambda o: f"exact {torch.equal(o, want_sc)}")
    base = torch.randn(C, H, generator=g).bfloat16()
    want_add = base.float().clone()
    want_add[perm] += x.float()
    run("scatter_add (dma_compute RMW into HBM)", p_scatter_add, dict(x=xd, base=base.to(dev), idx=idxd), NT,
        lambda o: f"max abs err {(o.float() - want_add).abs().max().item():.3e} (bf16 store)")
    want_g = x[perm].view(NT, 128, H).float().sum(0)
    run("gather_add (dma_compute RMW into SBUF fp32)", p_gather_add, dict(x=xd, idx=idxd), NT,
        lambda o: f"max abs err {(o - want_g).abs().max().item():.3e}")
    w16 = torch.randn(C, 16, generator=g).bfloat16()
    want_w = torch.empty_like(w16)
    want_w[perm] = w16
    run("small_scatter (rows of 32 bytes)", p_small_scatter, dict(w=w16.to(dev), idx=idxd), NT,
        lambda o: f"exact {torch.equal(o, want_w)}")
    run("transpose (tensor engine, scalar drain)", p_transpose, dict(x=xd, nt=NT, eng=0), NT)
    run("transpose (tensor engine, vector drain)", p_transpose, dict(x=xd, nt=NT, eng=1), NT)
    w8 = (torch.randint(0, 256, (128, H), dtype=torch.uint8, generator=g) & 0xBF)
    for N in (128, 256, 512):
        xT = torch.randn(128, H // 128, N, generator=g).bfloat16()
        run(f"matmul fp8 x bf16, N={N}, {H // 128} tiles", p_matmul, dict(w8=w8.to(dev), xT=xT.to(dev), nt=NT),
            NT * (H // 128) * N)
    a = torch.randn(128, 128, generator=g).bfloat16()
    wd = (torch.randint(0, 256, (128, H), dtype=torch.uint8, generator=g) & 0xBF)
    run("matmul_k64 (down shape), plain", p_matmul_k64, dict(a=a.to(dev), wd=wd.to(dev), nt=NT, tiled=0),
        NT * H // 512)
    run("matmul_k64 (down shape), row tiles", p_matmul_k64, dict(a=a.to(dev), wd=wd.to(dev), nt=NT, tiled=1),
        NT * H // 512)
    wrow = torch.randn(1, 128, generator=g).bfloat16()
    for eng, nm in ((0, "scalar"), (1, "vector"), (2, "both")):
        run(f"drain PSUM [128, 512] ({nm})", p_drain, dict(w=wrow.to(dev), nt=NT * 8, eng=eng), NT * 8)
    for bk, sdt in ((128, torch.float32), (32, torch.bfloat16)):
        s_ = (torch.rand(128, H // bk, generator=g) * 0.02 + 0.005).to(sdt)
        want_dq = (w8.view(torch.float8_e4m3fn).float() * s_.float().repeat_interleave(bk, dim=1)).bfloat16()
        for eng, nm in ((0, "gpsimd"), (1, "vector")):
            run(f"dequant fp8 [128, {H}] x {sdt} block-{bk} scales ({nm})", p_dequant,
                dict(w8=w8.to(dev), s=s_.to(dev), nt=NT, eng=eng, bk=bk), NT,
                lambda o, want_dq=want_dq: f"exact {torch.equal(o, want_dq)} (vs torch e4m3fn bytes; max abs "
                                           f"{(o.float() - want_dq.float()).abs().max().item():.3e})")
    for width in (128, 256, 512):
        run(f"act_width {width} (scalar drain, per-partition scale)", p_act_width,
            dict(w=wrow.to(dev), nt=NT * 8, width=width), NT * 8)
    F = 32 * 512
    data = torch.randn(128, F, generator=g).bfloat16()
    ind = torch.randint(0, F, (128, 512), generator=g).to(torch.int32)
    run("ngather 512 of 16384 per partition", p_ngather, dict(data=data.to(dev), ind=ind.to(dev), nt=NT), NT * 512)


if __name__ == "__main__":
    main()
