"""A small decode call's whole MoE FFN as ONE NKI kernel (GLM-5.3-Flash, models/hybrid.py _mlp): the router, its top-k,
the routed experts and the shared expert, for calls under kernels/moe_dedupe.py's PAIRS_BELOW tokens.

Why one call: inside a decode graph an NKI kernel call costs ~0.21 ms beyond its own work, because neuronx-cc 2.27
overlaps nothing across the call (tools/probe_gemv.py, trn1: 8 chained projections stream 262.6 GB/s as one kernel, 99
GB/s as 8 calls; docs/neuron-notes.md "Decode at scale"), and at 1 row per DP group the FFN block moves its ~37 MB at
~130 GB/s: the XLA router (fp32 logits, sigmoid, top-k, gather, normalisation), the per-pair expert kernel
(moe_dedupe.kiln_moe_tiles_pairs_v1) and the XLA shared expert run one after another with a kernel boundary in the
middle. Here they are one program: the router's weight, the experts and the shared expert stream back to back.

What it computes (DecoderForCausalLM._route with router_scoring "sigmoid", then hybrid._moe_clamped and _mlp's shared
expert, the Glm5NextTextTopkRouter / Glm5NextTextExperts arithmetic):
- logits [T, E] = x W_r^T (bf16 x bf16 products, exact, summed in fp32 PSUM); scores = sigmoid(logits);
- the K experts of the largest scores + bias (max8 / nc_find_index8: values descending, equal values at ascending
  indices), weights = their scores / (sum + 1e-20) * scale, rounded to bf16 as the graph's topv is;
- with `shared`, one more pair per token: expert E of the blob (the shared expert, packed like the routed ones by
  moe_dedupe.pack), weight 1;
- every pair through kiln_moe_tiles_pairs_v1's arithmetic (one expert load per pair, the gate_up tile dot products,
  the tile scales, the clamped SwiGLU, the down product and its scales), summed per token in fp32.
emulate() is that arithmetic in torch (moe_dedupe.emulate_pairs for the expert part).
"""

from __future__ import annotations

import os

import torch

from . import moe_dedupe as mdd

P = 128
KTOP = 8  # max8 gives 8 values per partition: the kernel takes top-8 routing (GLM-5.3-Flash num_experts_per_tok)


def route(x: torch.Tensor, w_router: torch.Tensor, bias: torch.Tensor, K: int, scale: float):
    """The kernel's routing in torch: (topv bf16 [T, K], topi int64 [T, K])."""
    logits = x.float() @ w_router.float().t()
    sc = torch.sigmoid(logits)
    ch = sc + bias.float()
    _, order = torch.sort(ch, dim=-1, descending=True, stable=True)
    topi = order[:, :K]
    tv = torch.gather(sc, 1, topi)
    tv = tv / (tv.sum(-1, keepdim=True) + 1e-20) * scale
    return tv.to(torch.bfloat16), topi


def extend(topv: torch.Tensor, topi: torch.Tensor, E: int):
    """The routing with the shared expert as one more pair per token: expert E, weight 1."""
    T = topi.shape[0]
    return (torch.cat([topv, torch.ones(T, 1, dtype=topv.dtype, device=topv.device)], dim=1),
            torch.cat([topi, torch.full((T, 1), E, dtype=topi.dtype, device=topi.device)], dim=1))


def emulate(x, w_router, bias, blob, K: int, scale: float, shared: bool, act: int = 1, limit: float = 0.0):
    E = w_router.shape[0]
    topv, topi = route(x, w_router, bias, K, scale)
    if shared:
        topv, topi = extend(topv, topi, E)
    return mdd.emulate_pairs(x, topv, topi, blob, act, limit)


try:  # the Neuron venv
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def kiln_moe_ffn_pairs_v1(xT, rT, rbias, blob, K: int, shared: int, scale: float, group: int, ring: int,
                              act: int, limit: float, rev: int):
        """xT bf16 [128, T, C] (x[t, c 128 + i] at [i, t, c]); rT bf16 [128, C, E] (the router weight W_r [E, H] as
        rT[i, c, e] = W_r[e, c 128 + i]); rbias fp32 [1, E]; blob uint8 [E (+ 1 with shared), 128, F] (moe_dedupe's tile
        layout); K = 8; scale the routed scaling factor; rev: this module's kernel source revision. Returns bf16
        [128, T, C] with y[t, h(p, g)] at [p, t, g] (moe_dedupe.from_pairs)."""
        _, T, C = xT.shape
        E = rT.shape[2]
        assert K == KTOP and T <= 128
        K1 = K + shared
        N = T * K1
        F = blob.shape[2]
        H = C * 128
        DW = H // 2
        C2 = DW // 128
        Q = group
        NQ = -(-N // Q)
        NP = NQ * Q
        SB = (F - H - DW) // (2 * C)
        o_dw, o_sg, o_sd = H, H + DW, H + DW + C * SB
        sdt = nl.bfloat16 if SB == 2 else nl.float32
        NR = ring
        PD = NR - 2
        f32, bf16, i32, u32 = nl.float32, nl.bfloat16, nl.int32, nl.uint32
        out = nl.ndarray((128, T, C), dtype=bf16, buffer=nl.shared_hbm)

        x_sb = nl.ndarray((128, T, C), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_sb, src=xT)
        # --- the router: logits [T, E] in one PSUM bank (E fp32 <= 512 columns)
        r_sb = nl.ndarray((128, C, E), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=r_sb, src=rT)
        pl = nl.ndarray((128, E), dtype=f32, buffer=nl.psum)
        for c in range(C):
            nisa.nc_matmul(dst=pl[0:T, :], stationary=x_sb[:, :, c], moving=r_sb[:, c, :], accumulate=(c > 0))
        sc = nl.ndarray((128, E), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=sc[0:T, :], op=nl.sigmoid, data=pl[0:T, :])
        bb = nl.ndarray((128, E), dtype=f32, buffer=nl.sbuf)
        for t in range(T):  # the bias on every token's partition
            nisa.dma_copy(dst=bb[t:t + 1, :], src=rbias)
        ch = nl.ndarray((128, E), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ch[0:T, :], data1=sc[0:T, :], data2=bb[0:T, :], op=nl.add)
        v8 = nl.ndarray((128, KTOP), dtype=f32, buffer=nl.sbuf)
        nisa.max8(dst=v8[0:T, :], src=ch[0:T, :])
        i8 = nl.ndarray((128, KTOP), dtype=u32, buffer=nl.sbuf)
        nisa.nc_find_index8(dst=i8[0:T, :], data=ch[0:T, :], vals=v8[0:T, :])
        i8f = nl.ndarray((128, KTOP), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=i8f[0:T, :], src=i8[0:T, :])
        # the chosen experts' scores: sum over e of [e == i_j] sc[e]
        ioe_i = nl.ndarray((128, E), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ioe_i, pattern=[[1, E]], offset=0, channel_multiplier=0)
        ioe = nl.ndarray((128, E), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ioe, src=ioe_i)
        tv = nl.ndarray((128, KTOP), dtype=f32, buffer=nl.sbuf)
        mk = nl.ndarray((128, E), dtype=f32, buffer=nl.sbuf)
        for j in range(KTOP):
            nisa.tensor_scalar(dst=mk[0:T, :], data=ioe[0:T, :], op0=nl.equal, operand0=i8f[0:T, j:j + 1])
            nisa.tensor_tensor(dst=mk[0:T, :], data1=mk[0:T, :], data2=sc[0:T, :], op=nl.multiply)
            nisa.tensor_reduce(dst=tv[0:T, j:j + 1], op=nl.add, data=mk[0:T, :], axis=1)
        ssum = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ssum[0:T, :], op=nl.add, data=tv[0:T, :], axis=1)
        nisa.tensor_scalar(dst=ssum[0:T, :], data=ssum[0:T, :], op0=nl.add, operand0=1e-20)
        rs = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rs[0:T, :], data=ssum[0:T, :])
        # pairs [T, K1]: index (fp32, exact below 2^24) and weight (bf16, as the graph's topv)
        PI = nl.ndarray((128, K1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=PI[0:T, 0:K], src=i8f[0:T, :])
        PV = nl.ndarray((128, K1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=PV[0:T, 0:K], data=tv[0:T, :], op0=nl.multiply, operand0=rs[0:T, :], op1=nl.multiply,
                           operand1=scale)
        if shared:
            nisa.memset(dst=PI[0:T, K:K1], value=float(E))
            nisa.memset(dst=PV[0:T, K:K1], value=1.0)
        PVb = nl.ndarray((128, K1), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=PVb[0:T, :], src=PV[0:T, :])
        # rows on partition 0 / on every partition: M[t', t, j] = PI[t', j] [t' == t], summed over t' by a matmul
        ipt_i = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ipt_i, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        ipt = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ipt, src=ipt_i)
        jt_i = nl.ndarray((128, T), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=jt_i, pattern=[[1, T]], offset=0, channel_multiplier=0)
        eyeT = nl.ndarray((128, T), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eyeT, src=jt_i)
        nisa.tensor_scalar(dst=eyeT, data=eyeT, op0=nl.equal, operand0=ipt)
        MI = nl.ndarray((128, T, K1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=MI[0:T], data1=PI[0:T].expand_dim(1).broadcast(1, T),
                           data2=eyeT[0:T].expand_dim(2).broadcast(2, K1), op=nl.multiply)
        eyeTb = nl.ndarray((128, T), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eyeTb, src=eyeT)
        MV = nl.ndarray((128, T, K1), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=MV[0:T], data1=PVb[0:T].expand_dim(1).broadcast(1, T),
                           data2=eyeTb[0:T].expand_dim(2).broadcast(2, K1), op=nl.multiply)
        ones_c = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ones_c, value=1.0)
        pe = nl.ndarray((1, N), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pe, stationary=ones_c[0:T, :], moving=MI[0:T].flatten_dims(1, 2), accumulate=False)
        e_sb = nl.ndarray((1, NP), dtype=i32, buffer=nl.sbuf)
        nisa.memset(dst=e_sb, value=0)
        nisa.tensor_copy(dst=e_sb[:, 0:N], src=pe)
        ones_b = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=ones_b, value=1.0)
        pwb = nl.ndarray((128, N), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pwb, stationary=ones_b[0:T, :], moving=MV[0:T].flatten_dims(1, 2), accumulate=False)
        wb = nl.ndarray((128, NP), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=wb, value=0.0)
        nisa.tensor_copy(dst=wb[:, 0:N], src=pwb)

        # --- the pairs: kiln_moe_tiles_pairs_v1's loop with K1 pairs per token
        ipi = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ipi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        ip = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ip, src=ipi)
        jfi = nl.ndarray((128, 128), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=jfi, pattern=[[1, 128]], offset=0, channel_multiplier=0)
        dd = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dd, src=jfi)
        nisa.tensor_scalar(dst=dd, data=dd, op0=nl.subtract, operand0=ip)
        fd = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        f1 = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        f2 = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=fd, data=dd, op0=nl.equal, operand0=0.0)
        nisa.tensor_scalar(dst=f1, data=dd, op0=nl.equal, operand0=64.0)
        nisa.tensor_scalar(dst=f2, data=dd, op0=nl.equal, operand0=-64.0)
        nisa.tensor_tensor(dst=fd, data1=fd, data2=f1, op=nl.add)
        nisa.tensor_tensor(dst=fd, data1=fd, data2=f2, op=nl.add)
        mh = nl.ndarray((128, 2), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mh[:, 0:1], data=ip, op0=nl.less, operand0=64.0)
        nisa.tensor_scalar(dst=mh[:, 1:2], data=ip, op0=nl.greater_equal, operand0=64.0)
        acc = nl.ndarray((128, T, C), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)

        bufs = (nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf),
                nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf),
                nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf))[:NR]
        sg_r = (nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf))
        sd_r = (nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf))
        q_r = (nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf),
               nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf))
        g_r = (nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf),
               nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf))
        mm_r = (nl.ndarray((128, Q, 2), dtype=bf16, buffer=nl.sbuf), nl.ndarray((128, Q, 2), dtype=bf16, buffer=nl.sbuf),
                nl.ndarray((128, Q, 2), dtype=bf16, buffer=nl.sbuf))
        sl_r = (nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf))
        au_r = (nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf))
        a2_r = (nl.ndarray((128, Q, 2), dtype=bf16, buffer=nl.sbuf), nl.ndarray((128, Q, 2), dtype=bf16, buffer=nl.sbuf),
                nl.ndarray((128, Q, 2), dtype=bf16, buffer=nl.sbuf))
        y_r = (nl.ndarray((128, Q, C2, 2), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, Q, C2, 2), dtype=f32, buffer=nl.sbuf),
               nl.ndarray((128, Q, C2, 2), dtype=f32, buffer=nl.sbuf))
        pg_r = (nl.ndarray((128, Q, C), dtype=f32, buffer=nl.psum), nl.ndarray((128, Q, C), dtype=f32, buffer=nl.psum),
                nl.ndarray((128, Q, C), dtype=f32, buffer=nl.psum))
        pf_r = (nl.ndarray((128, Q, 2), dtype=f32, buffer=nl.psum), nl.ndarray((128, Q, 2), dtype=f32, buffer=nl.psum),
                nl.ndarray((128, Q, 2), dtype=f32, buffer=nl.psum))
        pd_r = (nl.ndarray((128, Q, C2, 2), dtype=f32, buffer=nl.psum), nl.ndarray((128, Q, C2, 2), dtype=f32, buffer=nl.psum),
                nl.ndarray((128, Q, C2, 2), dtype=f32, buffer=nl.psum))
        for it in range(-PD, NQ + 1):
            gl = it + PD
            if 0 <= gl < NQ:
                bq = bufs[gl % NR]
                for s in range(Q):
                    e = e_sb.ap(pattern=[[NP, 1], [1, 1]], offset=gl * Q + s)
                    nisa.dma_copy(dst=bq[:, s, :], src=blob.select(0, e))
            if 0 <= it < NQ:  # stage A: gate_up, scales
                gq = it
                bq = bufs[gq % NR]
                pg = pg_r[gq % 3]
                for s in range(Q):
                    t = min(T - 1, (gq * Q + s) // K1)
                    wg = bq[:, s, 0:H].view(nl.float8_e4m3)
                    for c in range(C):
                        nisa.nc_matmul(dst=pg[:, s, c:c + 1], stationary=wg[:, c * 128:(c + 1) * 128],
                                       moving=x_sb[:, t, c:c + 1], accumulate=False)
                nisa.activation(dst=sg_r[gq % 3], op=nl.copy, data=bq[:, :, o_sg:o_sd].view(sdt))
                nisa.activation(dst=sd_r[gq % 3], op=nl.copy, data=bq[:, :, o_sd:F].view(sdt))
                q = q_r[gq % 3]
                nisa.tensor_tensor(dst=q, data1=pg, data2=sg_r[gq % 3], op=nl.multiply)
                g = g_r[gq % 3]
                nisa.tensor_reduce(dst=g, op=nl.add, data=q, axis=2)
                nisa.tensor_tensor(dst=mm_r[gq % 3], data1=g.expand_dim(2).broadcast(2, 2),
                                   data2=mh.expand_dim(1).broadcast(1, Q), op=nl.multiply)
            if 1 <= it <= NQ:  # stages B and C of group it - 1
                gq = it - 1
                pf = pf_r[gq % 3]
                nisa.nc_matmul(dst=pf.flatten_dims(1, 2), stationary=fd, moving=mm_r[gq % 3].flatten_dims(1, 2),
                               accumulate=False)
                sl = sl_r[gq % 3]
                au = au_r[gq % 3]
                if act == 0:
                    nisa.activation(dst=sl, op=nl.silu, data=pf[:, :, 0])
                    nisa.tensor_tensor(dst=au, data1=sl, data2=pf[:, :, 1], op=nl.multiply)
                else:  # silu(min(gate, limit)) * clamp(up, -limit, limit) (ACTS 1)
                    gc = nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=gc, data=pf[:, :, 0], op0=nl.minimum, operand0=limit)
                    uc = nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=uc, data=pf[:, :, 1], op0=nl.maximum, operand0=-limit, op1=nl.minimum,
                                       operand1=limit)
                    nisa.activation(dst=sl, op=nl.silu, data=gc)
                    nisa.tensor_tensor(dst=au, data1=sl, data2=uc, op=nl.multiply)
                nisa.tensor_tensor(dst=a2_r[gq % 3], data1=au.expand_dim(2).broadcast(2, 2),
                                   data2=mh.expand_dim(1).broadcast(1, Q), op=nl.multiply)
                bq = bufs[gq % NR]
                pd = pd_r[gq % 3]
                for s in range(Q):
                    wd = bq[:, s, o_dw:o_sg].view(nl.float8_e4m3)
                    for c in range(C2):
                        nisa.nc_matmul(dst=pd[:, s, c, :], stationary=wd[:, c * 128:(c + 1) * 128],
                                       moving=a2_r[gq % 3][:, s, :], accumulate=False)
                y = y_r[gq % 3]
                nisa.tensor_tensor(dst=y.flatten_dims(2, 3), data1=pd.flatten_dims(2, 3), data2=sd_r[gq % 3],
                                   op=nl.multiply)
                for s in range(Q):
                    u = gq * Q + s
                    if u < N:
                        t = u // K1
                        nisa.scalar_tensor_tensor(dst=acc[:, t, :], data=y[:, s].flatten_dims(1, 2), op0=nl.multiply,
                                                  operand0=wb[:, u:u + 1], op1=nl.add, operand1=acc[:, t, :])
        o_sb = nl.ndarray((128, T, C), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_sb, src=acc)
        nisa.dma_copy(dst=out, src=o_sb)
        return out
else:
    kiln_moe_ffn_pairs_v1 = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source (LNL's compile-cache key does not include NKI kernel source: CLAUDE.md)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_moe_ffn_pairs_v1 = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev()


def kernel_inputs(x, rT, bias, blob, K: int, scale: float, shared: bool, act: int = 1, limit: float = 0.0):
    """The kernel's arguments for x [T, H], the router rT [128, H / 128, E] (router_t()) and one layer's blob."""
    T, H = x.shape
    return dict(xT=x.view(T, H // P, P).permute(2, 0, 1).contiguous().to(torch.bfloat16), rT=rT,
                rbias=bias.float().reshape(1, -1).contiguous(), blob=blob, K=int(K), shared=int(bool(shared)),
                scale=float(scale), group=mdd.group_size(1), ring=mdd.RING, act=int(act), limit=float(limit), rev=REV)


def router_t(w_router: torch.Tensor) -> torch.Tensor:
    """The router weight [E, H] in the kernel's layout, rT[i, c, e] = W_r[e, c 128 + i] (bf16), made once at load."""
    E, H = w_router.shape
    return w_router.to(torch.bfloat16).view(E, H // P, P).permute(2, 1, 0).contiguous()


def moe_ffn(x: torch.Tensor, rT: torch.Tensor, w_router: torch.Tensor, bias: torch.Tensor, blob: torch.Tensor, K: int,
            scale: float, shared: bool, act: int = 1, limit: float = 0.0) -> torch.Tensor:
    """The routed experts plus (shared) the shared expert of x [T, H] (T < moe_dedupe.PAIRS_BELOW): the kernel on a
    Neuron device, emulate() elsewhere. Returns [T, H] in x's dtype (before the block's all-reduce)."""
    if x.device.type == "cpu":
        return emulate(x, w_router, bias, blob, K, scale, shared, act, limit)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if kiln_moe_ffn_pairs_v1 is None:
        raise RuntimeError("the NKI MoE FFN kernel needs the nki package (the Neuron venv)")
    out = wrap_nki(kiln_moe_ffn_pairs_v1)[platform.nki_grid()](**kernel_inputs(x, rT, bias, blob, K, scale, shared, act,
                                                                               limit))
    return mdd.from_pairs(out, x.dtype)
