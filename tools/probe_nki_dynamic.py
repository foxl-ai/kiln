"""Which NKI dynamic features run on trn1 (NeuronCore-v2) inside an LNL graph: device-side loops
whose trip count is a register loaded from a tensor (nl.dynamic_range), register-indexed SBUF
operands of compute instructions (select with a VirtualRegister), a register-indexed moving
operand of nc_matmul, a register-indexed HBM DMA source, and DMAs whose out-of-bounds index is
skipped (oob_mode.skip). Each case runs through wrap_nki in a torch.compile graph on the device
and is compared with the same computation on the host; then a dynamic loop is timed against
the same work unrolled.

    python tools/probe_nki_dynamic.py [--sim]   # --sim: nki.simulate on the host instead
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402


@nki.jit
def k_gather(x, idx, cnt):
    """out[:, i, :] = 2 x[:, idx[i], :] for i < cnt (a register), 0 beyond; acc[:, idx[i], :] +=
    x[:, i, :] (register-indexed destination). x fp32 [128, 8, 16], idx int32 [1, 8], cnt [1, 1]."""
    _, S, D = x.shape
    out = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.shared_hbm)
    acc_out = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.shared_hbm)
    x_sb = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x)
    i_sb = nl.ndarray((1, S), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=i_sb, src=idx)
    c_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_sb, src=cnt)
    o_sb = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=o_sb, value=0.0)
    a_sb = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=a_sb, value=0.0)
    n = nisa.register_alloc()
    nisa.register_load(n, c_sb)
    for i in nl.dynamic_range(n):
        t = nisa.register_alloc()
        nisa.register_load(t, i_sb.ap(pattern=[[S, 1], [1, 1]], scalar_offset=i, indirect_dim=1))
        nisa.tensor_scalar(dst=o_sb.select(1, i), data=x_sb.select(1, t), op0=nl.multiply, operand0=2.0)
        nisa.tensor_tensor(dst=a_sb.select(1, t), data1=a_sb.select(1, t), data2=x_sb.select(1, i), op=nl.add)
    nisa.dma_copy(dst=out, src=o_sb)
    nisa.dma_copy(dst=acc_out, src=a_sb)
    return out, acc_out


@nki.jit
def k_loopvar(x, cnt):
    """out[:, i, :] = 2 x[:, i, :] for i < cnt: only the loop register indexes (no register load
    from a dynamic address)."""
    _, S, D = x.shape
    out = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.shared_hbm)
    x_sb = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x)
    c_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_sb, src=cnt)
    o_sb = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=o_sb, value=0.0)
    n = nisa.register_alloc()
    nisa.register_load(n, c_sb)
    for i in nl.dynamic_range(n):
        nisa.tensor_scalar(dst=o_sb.select(1, i), data=x_sb.select(1, i), op0=nl.multiply, operand0=2.0)
    nisa.dma_copy(dst=out, src=o_sb)
    return out


@nki.jit
def k_gather2(x, idx, cnt):
    """k_gather, the index read through a copy to a fixed [1, 1] tile before register_load."""
    _, S, D = x.shape
    out = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.shared_hbm)
    acc_out = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.shared_hbm)
    x_sb = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x)
    i_sb = nl.ndarray((1, S), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=i_sb, src=idx)
    c_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_sb, src=cnt)
    o_sb = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=o_sb, value=0.0)
    a_sb = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=a_sb, value=0.0)
    tmp = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    n = nisa.register_alloc()
    nisa.register_load(n, c_sb)
    for i in nl.dynamic_range(n):
        nisa.tensor_copy(dst=tmp, src=i_sb.select(1, i), engine=nisa.engine.gpsimd)
        t = nisa.register_alloc()
        nisa.register_load(t, tmp)
        nisa.tensor_scalar(dst=o_sb.select(1, i), data=x_sb.select(1, t), op0=nl.multiply, operand0=2.0)
        nisa.tensor_tensor(dst=a_sb.select(1, t), data1=a_sb.select(1, t), data2=x_sb.select(1, i), op=nl.add)
    nisa.dma_copy(dst=out, src=o_sb)
    nisa.dma_copy(dst=acc_out, src=a_sb)
    return out, acc_out


@nki.jit
def k_matmul(w, x, idx, cnt):
    """out[:, i, :] = w^T @ x[:, idx[i], :] for i < cnt, the moving operand register-indexed.
    w bf16 [128, 128], x bf16 [128, 8, 4]."""
    _, S, D = x.shape
    out = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.shared_hbm)
    w_sb = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_sb, src=w)
    x_sb = nl.ndarray((128, S, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x)
    i_sb = nl.ndarray((1, S), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=i_sb, src=idx)
    c_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_sb, src=cnt)
    o_sb = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=o_sb, value=0.0)
    n = nisa.register_alloc()
    nisa.register_load(n, c_sb)
    for i in nl.dynamic_range(n):
        t = nisa.register_alloc()
        nisa.register_load(t, i_sb.ap(pattern=[[S, 1], [1, 1]], scalar_offset=i, indirect_dim=1))
        ps = nl.ndarray((128, D), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps, stationary=w_sb, moving=x_sb.select(1, t), accumulate=False)
        nisa.tensor_copy(dst=o_sb.select(1, i), src=ps)
    nisa.dma_copy(dst=out, src=o_sb)
    return out


@nki.jit
def k_dma(blob, idx, cnt):
    """Dynamic loop over cnt slots: buf <- blob[idx[u]] (register-indexed HBM source, oob_mode
    skip), out[:, u, :] = buf. A skipped DMA leaves buf as the previous slot left it.
    blob bf16 [E, 128, F], idx int32 [1, U]."""
    E, _, F = blob.shape
    U = idx.shape[1]
    out = nl.ndarray((128, U, F), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    i_sb = nl.ndarray((1, U), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=i_sb, src=idx)
    c_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_sb, src=cnt)
    o_sb = nl.ndarray((128, U, F), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=o_sb, value=-1.0)
    buf = nl.ndarray((128, F), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=buf, value=7.0)
    n = nisa.register_alloc()
    nisa.register_load(n, c_sb)
    for u in nl.dynamic_range(n):
        e = nisa.register_alloc()
        nisa.register_load(e, i_sb.ap(pattern=[[U, 1], [1, 1]], scalar_offset=u, indirect_dim=1))
        nisa.dma_copy(dst=buf, src=blob.select(0, e), oob_mode=nisa.oob_mode.skip)
        nisa.tensor_copy(dst=o_sb.select(1, u), src=buf)
    nisa.dma_copy(dst=out, src=o_sb)
    return out


@nki.jit
def k_dma_static(blob, idx):
    """Static loop, one DMA per slot of blob[idx[u]] where idx comes from SBUF (as the decode
    kernel), with oob_mode skip: an index >= E skips the transfer."""
    E, _, F = blob.shape
    U = idx.shape[1]
    out = nl.ndarray((128, U, F), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    i_sb = nl.ndarray((1, U), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=i_sb, src=idx)
    o_sb = nl.ndarray((128, U, F), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=o_sb, value=-1.0)
    for u in range(U):
        e = i_sb.ap(pattern=[[U, 1], [1, 1]], offset=u)
        nisa.dma_copy(dst=o_sb[:, u, :], src=blob.select(0, e), oob_mode=nisa.oob_mode.skip)
    nisa.dma_copy(dst=out, src=o_sb)
    return out


@nki.jit
def k_tsoff(x, idx):
    """Static loop; compute operands at an SBUF-tensor offset (no registers): out[:, i, :] =
    2 x[:, idx[i], :] and acc[:, idx[i], :] += x[:, i, :]."""
    _, S, D = x.shape
    out = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.shared_hbm)
    acc_out = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.shared_hbm)
    x_sb = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x)
    i_sb = nl.ndarray((1, S), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=i_sb, src=idx)
    o_sb = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.sbuf)
    a_sb = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=a_sb, value=0.0)
    for i in range(S):
        t = i_sb.ap(pattern=[[S, 1], [1, 1]], offset=i)
        nisa.tensor_scalar(dst=o_sb[:, i, :], data=x_sb.select(1, t), op0=nl.multiply, operand0=2.0)
    for i in range(S):
        t = i_sb.ap(pattern=[[S, 1], [1, 1]], offset=i)
        nisa.tensor_tensor(dst=a_sb.select(1, t), data1=a_sb.select(1, t), data2=x_sb[:, i, :], op=nl.add)
    nisa.dma_copy(dst=out, src=o_sb)
    nisa.dma_copy(dst=acc_out, src=a_sb)
    return out, acc_out


@nki.jit
def k_mmoff(w, x, idx):
    """Static loop; nc_matmul with the stationary and the moving operand at SBUF-tensor offsets:
    out[:, i, :] = w[:, idx[i], :]^T @ x[:, idx[i], :]. w bf16 [128, 8, 128], x bf16 [128, 8, 4]."""
    _, S, D = x.shape
    out = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.shared_hbm)
    w_sb = nl.ndarray((128, S, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=w_sb, src=w)
    x_sb = nl.ndarray((128, S, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x)
    i_sb = nl.ndarray((1, S), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=i_sb, src=idx)
    o_sb = nl.ndarray((128, S, D), dtype=nl.float32, buffer=nl.sbuf)
    for i in range(S):
        t = i_sb.ap(pattern=[[S, 1], [1, 1]], offset=i)
        ps = nl.ndarray((128, D), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps, stationary=w_sb.select(1, t), moving=x_sb.select(1, t), accumulate=False)
        nisa.tensor_copy(dst=o_sb[:, i, :], src=ps)
    nisa.dma_copy(dst=out, src=o_sb)
    return out


@nki.jit
def k_dma_time(blob, idx):
    """U whole-expert DMAs (uint8 [128, F] rows of blob [E, 128, F]) at SBUF-tensor indices,
    oob_mode skip, into 4 rotating buffers; out = a checksum-ish reduction of the last buffers
    so nothing is dead. Times the skip."""
    E, _, F = blob.shape
    U = idx.shape[1]
    out = nl.ndarray((128, 4), dtype=nl.float32, buffer=nl.shared_hbm)
    i_sb = nl.ndarray((1, U), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=i_sb, src=idx)
    bufs = nl.ndarray((128, 4, F), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.memset(dst=bufs, value=0)
    o_sb = nl.ndarray((128, 4), dtype=nl.float32, buffer=nl.sbuf)
    for u in range(U):
        e = i_sb.ap(pattern=[[U, 1], [1, 1]], offset=u)
        nisa.dma_copy(dst=bufs[:, u % 4, :], src=blob.select(0, e), oob_mode=nisa.oob_mode.skip)
    for b in range(4):
        nisa.tensor_reduce(dst=o_sb[:, b:b + 1], op=nl.add, data=bufs[:, b, 0:64].view(nl.bfloat16), axis=1)
    nisa.dma_copy(dst=out, src=o_sb)
    return out


@nki.jit
def k_nest(x, flags):
    """Nested device loops of trip count 0/1 (flags[0] outer, flags[1 + i] inner i), the inner
    loops writing slices of a tile allocated in the outer loop's body, read after them."""
    out = nl.ndarray((128, 4, 64), dtype=nl.float32, buffer=nl.shared_hbm)
    x_sb = nl.ndarray((128, 4, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x)
    f_sb = nl.ndarray((1, 5), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=f_sb, src=flags)
    o_sb = nl.ndarray((128, 4, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=o_sb, value=0.0)
    ro = nisa.register_alloc()
    nisa.register_load(ro, f_sb.ap(pattern=[[5, 1], [1, 1]], offset=0))
    for _ in nl.dynamic_range(ro):
        t = nl.ndarray((128, 4, 64), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=t, value=0.0)
        for i in range(4):
            ri = nisa.register_alloc()
            nisa.register_load(ri, f_sb.ap(pattern=[[5, 1], [1, 1]], offset=1 + i))
            for _ in nl.dynamic_range(ri):
                nisa.tensor_scalar(dst=t[:, i, :], data=x_sb[:, i, :], op0=nl.multiply, operand0=2.0)
        nisa.tensor_tensor(dst=o_sb, data1=t, data2=x_sb, op=nl.add)
    nisa.dma_copy(dst=out, src=o_sb)
    return out


@nki.jit
def k_seq(x, flags):
    """k_nest's inner loops at the top level, the tile allocated before all of them."""
    out = nl.ndarray((128, 4, 64), dtype=nl.float32, buffer=nl.shared_hbm)
    x_sb = nl.ndarray((128, 4, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x)
    f_sb = nl.ndarray((1, 5), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=f_sb, src=flags)
    o_sb = nl.ndarray((128, 4, 64), dtype=nl.float32, buffer=nl.sbuf)
    t = nl.ndarray((128, 4, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=t, value=0.0)
    for i in range(4):
        ri = nisa.register_alloc()
        nisa.register_load(ri, f_sb.ap(pattern=[[5, 1], [1, 1]], offset=1 + i))
        for _ in nl.dynamic_range(ri):
            w = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=w, data=x_sb[:, i, :], op0=nl.multiply, operand0=2.0)
            nisa.tensor_copy(dst=t[:, i, :], src=w)
    nisa.tensor_tensor(dst=o_sb, data1=t, data2=x_sb, op=nl.add)
    nisa.dma_copy(dst=out, src=o_sb)
    return out


@nki.jit
def k_chain(x, flags):
    """Sequential device loops (trip counts flags[0..2]) that hand a tile on: loop 0 memsets A and
    writes its first half, loop 1 writes A's second half, loop 2 reads A into B; B is read after."""
    out = nl.ndarray((128, 2, 64), dtype=nl.float32, buffer=nl.shared_hbm)
    x_sb = nl.ndarray((128, 2, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x)
    f_sb = nl.ndarray((1, 3), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=f_sb, src=flags)
    a = nl.ndarray((128, 2, 64), dtype=nl.float32, buffer=nl.sbuf)
    b = nl.ndarray((128, 2, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=b, value=-1.0)
    r0 = nisa.register_alloc()
    nisa.register_load(r0, f_sb.ap(pattern=[[3, 1], [1, 1]], offset=0))
    for _ in nl.dynamic_range(r0):
        nisa.memset(dst=a, value=0.0)
        nisa.tensor_scalar(dst=a[:, 0, :], data=x_sb[:, 0, :], op0=nl.multiply, operand0=2.0)
    r1 = nisa.register_alloc()
    nisa.register_load(r1, f_sb.ap(pattern=[[3, 1], [1, 1]], offset=1))
    for _ in nl.dynamic_range(r1):
        nisa.tensor_scalar(dst=a[:, 1, :], data=x_sb[:, 1, :], op0=nl.multiply, operand0=3.0)
    r2 = nisa.register_alloc()
    nisa.register_load(r2, f_sb.ap(pattern=[[3, 1], [1, 1]], offset=2))
    for _ in nl.dynamic_range(r2):
        nisa.tensor_tensor(dst=b, data1=a, data2=x_sb, op=nl.add)
    nisa.dma_copy(dst=out, src=b)
    return out


@nki.jit
def k_plan(e):
    """The primitives an in-kernel routing plan needs, on e fp32-valued int32 [1, 64]: iota, a K=1
    broadcast matmul, compare + reduce, a prefix-sum scan, scalar_tensor_tensor, int32 add + shift,
    an fp32 matmul. out[p, :] rows: 0 broadcast e, 1 [e == p], 2 cumsum of row 1, 3 (cs - 1) * oh,
    4 column sums (fp32 matmul) of row 3 over 128 partitions (every row equal), 5 [p] (iota),
    6 count of row 1 (reduce), 7 ceil(count / 4) through int32 add + shift."""
    N = e.shape[1]
    out = nl.ndarray((128, 8, N), dtype=nl.float32, buffer=nl.shared_hbm)
    o = nl.ndarray((128, 8, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=o, value=0.0)
    ei = nl.ndarray((1, N), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ei, src=e)
    eb1 = nl.ndarray((1, N), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=eb1, src=ei)
    one = nl.ndarray((1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=one, value=1.0)
    pb = nl.ndarray((128, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=pb, stationary=one, moving=eb1, accumulate=False)
    nisa.tensor_copy(dst=o[:, 0, :], src=pb)
    ipi = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=ipi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    ip = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ip, src=ipi)
    nisa.tensor_scalar(dst=o[:, 5, :], data=o[:, 0, :], op0=nl.multiply, operand0=0.0, op1=nl.add, operand1=ip)
    cnt = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar_reduce(dst=o[:, 1, :], data=o[:, 0, :], op0=nl.equal, operand0=ip, reduce_op=nl.add,
                              reduce_res=cnt)
    zero = nl.ndarray((128, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero, value=0.0)
    nisa.tensor_tensor_scan(dst=o[:, 2, :], data0=o[:, 1, :], data1=zero, initial=0.0, op0=nl.add, op1=nl.add)
    nisa.scalar_tensor_tensor(dst=o[:, 3, :], data=o[:, 2, :], op0=nl.subtract, operand0=1.0, op1=nl.multiply,
                              operand1=o[:, 1, :])
    ones = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    ps = nl.ndarray((128, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ps, stationary=ones, moving=o[:, 3, :], accumulate=False)
    nisa.tensor_copy(dst=o[:, 4, :], src=ps)
    nisa.tensor_scalar(dst=o[:, 6, :], data=o[:, 0, :], op0=nl.multiply, operand0=0.0, op1=nl.add, operand1=cnt)
    ci = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ci, src=cnt)
    c2 = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=c2, data=ci, op0=nl.add, operand0=3)
    c3 = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=c3, data=c2, op0=nl.right_shift, operand0=2)
    cf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=cf, src=c3)
    nisa.tensor_scalar(dst=o[:, 7, :], data=o[:, 0, :], op0=nl.multiply, operand0=0.0, op1=nl.add, operand1=cf)
    nisa.dma_copy(dst=out, src=o)
    return out


@nki.jit
def k_time_dyn(x, cnt, reps: int):
    """reps (static) x cnt (dynamic) iterations of a small vector op chain, to time the cost of
    a dynamic iteration. x fp32 [128, 64]."""
    out = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.shared_hbm)
    x_sb = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x)
    c_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=c_sb, src=cnt)
    n = nisa.register_alloc()
    nisa.register_load(n, c_sb)
    for _ in nl.dynamic_range(n):
        for _ in range(reps):
            nisa.tensor_scalar(dst=x_sb, data=x_sb, op0=nl.multiply, operand0=0.5, op1=nl.add, operand1=1.0)
    nisa.dma_copy(dst=out, src=x_sb)
    return out


@nki.jit
def k_time_static(x, iters: int):
    out = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.shared_hbm)
    x_sb = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_sb, src=x)
    for _ in range(iters):
        nisa.tensor_scalar(dst=x_sb, data=x_sb, op0=nl.multiply, operand0=0.5, op1=nl.add, operand1=1.0)
    nisa.dma_copy(dst=out, src=x_sb)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="nki.simulate on the host, no device")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    if args.sim:
        os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
        dev = torch.device("cpu")

        def run(k, *a, **kw):
            outs = nki.simulate(k)(*a, **kw)
            if isinstance(outs, tuple):
                return tuple(torch.as_tensor(o) for o in outs)
            return torch.as_tensor(outs)
    else:
        import profile_layer as pl

        pl.setup_device()
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        dev = pl.DEV

        def run(k, *a, **kw):
            import inspect

            names = [n for n in inspect.signature(k.func).parameters][: len(a)]
            f = torch.compile(lambda *t: wrap_nki(k)[1](**dict(zip(names, t)), **kw), **pl.OPTS)
            o = f(*[t.to(dev) for t in a])
            return tuple(x.cpu() for x in o) if isinstance(o, (tuple, list)) else o.cpu()

    g = torch.Generator().manual_seed(0)
    want = lambda name: args.only is None or name in args.only  # noqa: E731

    def case(name, fn):
        if not want(name):
            return
        try:
            print(f"{name}: {fn()}", flush=True)
        except Exception as e:  # report and keep going
            msg = str(e).replace("\n", " ")
            print(f"{name}: FAILED {type(e).__name__}: {msg[:600]}", flush=True)

    x = torch.randn(128, 8, 16, generator=g)
    idx = torch.tensor([[5, 0, 7, 5, 2, 1, 3, 6]], dtype=torch.int32)

    def gather():
        res = []
        for c in (3, 8, 0):
            o, a = run(k_gather, x, idx, torch.tensor([[c]], dtype=torch.int32))
            eo = torch.zeros_like(x)
            ea = torch.zeros_like(x)
            for i in range(c):
                eo[:, i] = 2 * x[:, int(idx[0, i])]
                ea[:, int(idx[0, i])] += x[:, i]
            res.append(f"cnt={c} out err {(o - eo).abs().max().item():.2e} acc err {(a - ea).abs().max().item():.2e}")
        return "; ".join(res)

    case("gather", gather)

    def loopvar():
        res = []
        for c in (3, 8, 0):
            o = run(k_loopvar, x, torch.tensor([[c]], dtype=torch.int32))
            e = torch.zeros_like(x)
            e[:, :c] = 2 * x[:, :c]
            res.append(f"cnt={c} err {(o - e).abs().max().item():.2e}")
        return "; ".join(res)

    case("loopvar", loopvar)

    def gather2():
        res = []
        for c in (3, 8):
            o, a = run(k_gather2, x, idx, torch.tensor([[c]], dtype=torch.int32))
            eo = torch.zeros_like(x)
            ea = torch.zeros_like(x)
            for i in range(c):
                eo[:, i] = 2 * x[:, int(idx[0, i])]
                ea[:, int(idx[0, i])] += x[:, i]
            res.append(f"cnt={c} out err {(o - eo).abs().max().item():.2e} acc err {(a - ea).abs().max().item():.2e}")
        return "; ".join(res)

    case("gather2", gather2)

    def matmul():
        w = torch.randn(128, 128, generator=g).bfloat16()
        xb = torch.randn(128, 8, 4, generator=g).bfloat16()
        res = []
        for c in (5, 8):
            o = run(k_matmul, w, xb, idx, torch.tensor([[c]], dtype=torch.int32))
            e = torch.zeros(128, 8, 4)
            for i in range(c):
                e[:, i] = w.float().T @ xb[:, int(idx[0, i])].float()
            res.append(f"cnt={c} err {(o - e).abs().max().item():.2e} (|ref| {e.abs().max().item():.1f})")
        return "; ".join(res)

    case("matmul", matmul)

    blob = torch.randn(6, 128, 64, generator=g).bfloat16()
    sidx = torch.tensor([[4, 1, 9, 2, 6]], dtype=torch.int32)  # 9 and 6 are out of bounds for E=6

    def dma():
        o = run(k_dma, blob, sidx, torch.tensor([[5]], dtype=torch.int32))
        rows = []
        prev = torch.full((128, 64), 7.0).bfloat16()
        for u in range(5):
            e = int(sidx[0, u])
            exp = blob[e] if e < 6 else prev
            prev = exp
            rows.append("ok" if torch.equal(o[:, u], exp) else f"MISMATCH (got {o[0, u, :3].tolist()})")
        return " ".join(f"slot{u}(e={int(sidx[0, u])}):{r}" for u, r in enumerate(rows))

    case("dma", dma)

    def dma_static():
        o = run(k_dma_static, blob, sidx)
        rows = []
        for u in range(5):
            e = int(sidx[0, u])
            exp = blob[e] if e < 6 else torch.full((128, 64), -1.0).bfloat16()
            rows.append("ok" if torch.equal(o[:, u], exp) else f"MISMATCH (got {o[0, u, :3].tolist()})")
        return " ".join(f"slot{u}(e={int(sidx[0, u])}):{r}" for u, r in enumerate(rows))

    case("dma_static", dma_static)

    def tsoff():
        o, a = run(k_tsoff, x, idx)
        eo = torch.zeros_like(x)
        ea = torch.zeros_like(x)
        for i in range(8):
            eo[:, i] = 2 * x[:, int(idx[0, i])]
            ea[:, int(idx[0, i])] += x[:, i]
        return f"out err {(o - eo).abs().max().item():.2e} acc err {(a - ea).abs().max().item():.2e}"

    case("tsoff", tsoff)

    def mmoff():
        w = torch.randn(128, 8, 128, generator=g).bfloat16()
        xb = torch.randn(128, 8, 4, generator=g).bfloat16()
        o = run(k_mmoff, w, xb, idx)
        e = torch.zeros(128, 8, 4)
        for i in range(8):
            j = int(idx[0, i])
            e[:, i] = w[:, j].float().T @ xb[:, j].float()
        return f"err {(o - e).abs().max().item():.2e} (|ref| {e.abs().max().item():.1f})"

    case("mmoff", mmoff)

    def nested(k):
        def f():
            xx = torch.randn(128, 4, 64, generator=g)
            res = []
            for fl in ([1, 1, 0, 1, 0], [0, 1, 1, 1, 1], [1, 0, 0, 0, 1]):
                o = run(k, xx, torch.tensor([fl], dtype=torch.int32))
                e = torch.zeros_like(xx)
                if k is k_seq or fl[0]:
                    e = xx.clone()
                    for i in range(4):
                        if fl[1 + i]:
                            e[:, i] = 3 * xx[:, i]
                res.append(f"{fl}: err {(o - e).abs().max().item():.2e}")
            return "; ".join(res)

        return f

    def chain():
        xx = torch.randn(128, 2, 64, generator=g)
        o = run(k_chain, xx, torch.tensor([[1, 1, 1]], dtype=torch.int32))
        e = xx.clone()
        e[:, 0] += 2 * xx[:, 0]
        e[:, 1] += 3 * xx[:, 1]
        o2 = run(k_chain, xx, torch.tensor([[1, 0, 1]], dtype=torch.int32))
        e2 = xx.clone()
        e2[:, 0] += 2 * xx[:, 0]
        return (f"[1,1,1] err {(o - e).abs().max().item():.2e} (first half {(o - e)[:, 0].abs().max().item():.2e}); "
                f"[1,0,1] err {(o2 - e2).abs().max().item():.2e}")

    case("chain", chain)
    def plan_ops():
        e = torch.randint(0, 130, (1, 64), generator=g).to(torch.int32)
        e[0, :8] = 3  # a repeated value
        o = run(k_plan, e)
        ef = e.float().view(64)
        p = torch.arange(128).float().view(128, 1)
        oh = (ef.view(1, 64) == p).float()
        cs = oh.cumsum(1)
        rk = (cs - 1) * oh
        cnt = oh.sum(1, keepdim=True)
        want = [ef.expand(128, 64), oh, cs, rk, rk.sum(0, keepdim=True).expand(128, 64), p.expand(128, 64),
                cnt.expand(128, 64), torch.ceil(cnt / 4).expand(128, 64)]
        return " ".join(f"row{i}:{'ok' if torch.equal(o[:, i], w) else 'BAD ' + str((o[:, i] - w).abs().max().item())}"
                        for i, w in enumerate(want))

    case("plan", plan_ops)
    case("nest", nested(k_nest))
    case("seq", nested(k_seq))

    if not args.sim and want("dmatime"):
        import profile_layer as pl

        E, F, U = 256, 6528, 256
        big = torch.randint(0, 256, (E, 128, F), dtype=torch.uint8, generator=g).to(dev)
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            ids = torch.randperm(E, generator=g)[:U].to(torch.int32)
            ids[: int(U * frac)] = E  # out of bounds: skipped
            ids = ids[torch.randperm(U, generator=g)].view(1, U).to(dev)
            pl.timed(f"{U} expert DMAs, {frac:.0%} skipped", lambda b, i: wrap_nki(k_dma_time)[1](blob=b, idx=i), (big, ids))

    if args.sim or not want("time"):
        return
    import profile_layer as pl

    xt = torch.randn(128, 64, generator=g).to(dev)
    for reps, n in ((1, 64), (1, 256), (8, 32), (8, 128)):
        c = torch.tensor([[n]], dtype=torch.int32).to(dev)
        pl.timed(f"dynamic {n} x {reps} ops", lambda a, b: wrap_nki(k_time_dyn)[1](x=a, cnt=b, reps=reps), (xt, c))
        pl.timed(f"static {n * reps} ops", lambda a: wrap_nki(k_time_static)[1](x=a, iters=n * reps), (xt,))
    c0 = torch.tensor([[0]], dtype=torch.int32).to(dev)
    pl.timed("dynamic 0 iterations", lambda a, b: wrap_nki(k_time_dyn)[1](x=a, cnt=b, reps=1), (xt, c0))


if __name__ == "__main__":
    main()
