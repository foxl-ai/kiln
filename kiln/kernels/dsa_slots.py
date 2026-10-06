"""Absorbed-MLA attention of N queries, each over its OWN slots of pooled tokens, as one NKI kernel for NeuronCore-v2
(trn1): the long-context DSA path's attention (models/dsa_long.py slots / attend) for a prefill chunk's queries (N up
to 4096 per call) and for decode batches.

What it computes is kernels/dsa_decode.py's arithmetic: per row b and head h, over the row's NS slots of KP cache rows
each (slot j starting at cache row KP rows[b, j]), s[t] = scale q_lat[b, h] . K[t] + bias[b, t], p = softmax(s), o[b,
h] = sum_t p[t] K[t], absorbed MLA over the latent K [cache rows, R] (fp8 e4m3 or bf16); with lse, also the row's
log-sum-exp per head, lse[b, h] = max_t s[t] + log(sum_t exp(s[t] - max)), so partial attentions over disjoint key
sets (context-parallel ranks) combine as o = sum_r exp(lse_r - LSE) o_r. A row whose slots are all NEG_INF stays
finite: its scores all equal NEG_INF (|scale q . K| is far below NEG_INF's ulp), so p is uniform and lse is NEG_INF
(+ log NS KP, below fp32's resolution there).

Why a new kernel: dsa_decode unrolls its rows (`for b in range(B)`), which is fine for a decode batch and explodes
at a prefill chunk's 1024-4096 queries. Here the rows run in a device loop (nl.fori_loop), RPI rows per iteration
(their gathers in flight while the previous row computes), every per-row input and output moved by a DMA at the loop
register's offset into static SBUF tiles.

Per row, as dsa_decode: the NS / 128 gathers of 128 slots (one indirect DMA each, KP R bytes per slot); the bias row
broadcast to the H head partitions by one DMA (stride 0 over the partitions); q_lat^T by four PE transposes; per
128-token block K^T by four PE transposes against the identity (exact in the fp32 PSUM) and the block's scores
[H, 128] by four matmuls over R; softmax on the scalar engine (exp with the row's sum in the same instruction, P in
bf16); P's blocks transposed back and P K accumulated over the blocks into [H, R]; 1 / sum on the vector engine.
"""

from __future__ import annotations

import os

import torch

NEG_INF = -1e30  # models/decoder.NEG_INF
KP = 4  # tokens per pool (GLM-5.3-Flash index_kpool)
KTR = 4  # K^T tiles in flight
# Rows per device-loop iteration (their gathers in flight while the previous rows compute). trn1, 1024 rows, 640 slots,
# fp8 latent, H 8 / 64 (tools/probe_dsa_long.py slots, outputs reduced on the device): 1 row 45.9 / 48.3 us per row,
# 2 rows 41.0 / 43.6, 4 rows 39.0 / 39.5, 8 rows 40.3 / 39.2.
RPI = int(os.environ.get("KILN_DSA_SLOTS_RPI", 4))
QKW = int(os.environ.get("KILN_DSA_SLOTS_QKW", 1))  # scores 512 tokens per matmul (1) or 128 (0, dsa_decode's form)


def emulate(q_lat: torch.Tensor, kc: torch.Tensor, rows: torch.Tensor, bias: torch.Tensor, scale: float,
            lse: bool = False):
    """The kernel's arithmetic in torch: dsa_decode.emulate (q_lat [N, H, R], kc [rows, R] or [rows, 1, R], rows
    [N, NS] int, bias [N, NS, KP] fp32 -> o [N, H, R] fp32), plus with lse the log-sum-exp [N, H] fp32 of the scaled,
    biased scores."""
    from . import dsa_decode

    kc2 = kc.reshape(kc.shape[0], -1)
    o = dsa_decode.emulate(q_lat, kc2, rows, bias, scale)
    if not lse:
        return o
    N, H, R = q_lat.shape
    NS = rows.shape[1]
    tok = (rows.long().unsqueeze(-1) * KP + torch.arange(KP, device=rows.device)).reshape(N, NS * KP)
    rnd = (lambda t: t.to(torch.bfloat16).float()) if kc2.dtype != torch.float32 else (lambda t: t.float())
    s = torch.einsum("bhr,btr->bht", rnd(q_lat), rnd(kc2[tok])) * scale + bias.reshape(N, 1, NS * KP)
    m = s.amax(-1)
    return o, m + torch.log(torch.exp(s - m.unsqueeze(-1)).sum(-1))


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    F32, BF16, I32 = nl.float32, nl.bfloat16, nl.int32
    VE = nisa.vector_engine

    def _sb(shape, dt=None):
        return nl.ndarray(shape, dtype=dt or F32, buffer=nl.sbuf)

    def _pair(shape, dt):
        return [_sb(shape, dt), _sb(shape, dt)]

    def _row(X, it, k):
        """Row (it, k) of the device loop: row it RPI + k."""
        H, R, LC, NCH, T, rpi = X["H"], X["R"], X["LC"], X["NCH"], X["T"], X["rpi"]
        x = k % 2
        IB = X["IB"]
        PT, PS, PP, PO, PQ = X["psum"]
        # the row's slot indices, q_lat, bias (broadcast over its H head partitions)
        rs = X["RS"][x]
        nisa.dma_copy(dst=rs, src=X["rows_t"].ap(pattern=[[NCH, 128], [1, NCH]], offset=k * 128 * NCH,
                                                 scalar_offset=it, indirect_dim=0))
        Kv = X["K"][x]
        for ch in range(NCH):
            nisa.dma_copy(dst=Kv[ch], src=X["kc"].ap(pattern=[[KP * R, 128], [1, KP * R]], offset=0,
                                                       vector_offset=rs.ap(pattern=[[NCH, 128], [1, 1]], offset=ch),
                                                       indirect_dim=0))
        qr = X["QR"][x]
        nisa.dma_copy(dst=qr, src=X["q_lat"].ap(pattern=[[R, H], [1, R]], offset=k * H * R, scalar_offset=it,
                                                indirect_dim=0))
        S = X["S"][x]
        if X["qkw"]:
            # the row's bias on ONE partition; the scores' PSUM adds it to every head by a K = 1 fp32 matmul (a DMA
            # broadcasting it to the H head partitions wrote H x 10 KB per row)
            br = X["BR"][x]
            nisa.dma_copy(dst=br, src=X["bias"].ap(pattern=[[T, 1], [1, T]], offset=k * T, scalar_offset=it,
                                                   indirect_dim=0))
        else:
            # S starts as the row's bias, broadcast over its H head partitions (one DMA, stride 0 over the partitions)
            nisa.dma_copy(dst=S, src=X["bias"].ap(pattern=[[0, H], [1, T]], offset=k * T, scalar_offset=it,
                                                  indirect_dim=0))
        # q_lat^T [128 r, LC, H]
        for lc in range(LC):
            nisa.nc_matmul(dst=PQ[:, lc, :], stationary=qr[:, lc * 128:(lc + 1) * 128], moving=IB[0:H, 0:H],
                           accumulate=False)
        qt = X["QT"][x]
        nisa.tensor_copy(dst=qt, src=PQ, engine=VE)
        if X["qkw"]:
            # every block's K^T into the row's [128 r, LC, T] tile, then the scores 512 tokens per matmul (one
            # stationary load per (group, lc) instead of per (block, lc)), each group one group behind its transposes
            kta = X["KTA"][x]
            NG = T // 512
            for g in range(NG + 1):
                if g < NG:
                    for b in range(4):
                        i = g * 4 + b
                        ch = i // KP
                        t = i % KP
                        pt = PT[i % 2]
                        for lc in range(LC):
                            c0 = t * R + lc * 128
                            nisa.nc_matmul(dst=pt[:, lc, :], stationary=Kv[ch][:, c0:c0 + 128], moving=IB,
                                           accumulate=False)
                        if i % 2 == 0:
                            nisa.tensor_copy(dst=kta[:, :, i * 128:(i + 1) * 128], src=pt, engine=VE)
                        else:
                            nisa.activation(dst=kta[:, :, i * 128:(i + 1) * 128], op=nl.copy, data=pt)
                if g >= 1:
                    j = (g - 1) * 512
                    ps = PS[(g - 1) % 2]
                    for lc in range(LC):
                        nisa.nc_matmul(dst=ps, stationary=qt[:, lc, :], moving=kta[:, lc, j:j + 512],
                                       accumulate=True if lc > 0 else False)
                    # + bias / scale on every head (0 stays an exact 0; a masked token's NEG_INF / scale), then x scale
                    nisa.nc_matmul(dst=ps, stationary=X["RSC"][:, 0:H], moving=br[:, j:j + 512], accumulate=True)
                    nisa.activation(dst=S[:, j:j + 512], op=nl.copy, data=ps, scale=X["scale"])
        else:
            for ch in range(NCH):
                for t in range(KP):
                    i = ch * KP + t
                    pt = PT[i % 2]
                    for lc in range(LC):  # the block's latent, R on the partitions
                        c0 = t * R + lc * 128
                        nisa.nc_matmul(dst=pt[:, lc, :], stationary=Kv[ch][:, c0:c0 + 128], moving=IB,
                                       accumulate=False)
                    kt = X["KT"][i % KTR]
                    if i % 2 == 0:
                        nisa.tensor_copy(dst=kt, src=pt, engine=VE)
                    else:
                        nisa.activation(dst=kt, op=nl.copy, data=pt)
                    ps = PS[i % 2]
                    for lc in range(LC):
                        nisa.nc_matmul(dst=ps[:, 0:128], stationary=qt[:, lc, :], moving=kt[:, lc, :],
                                       accumulate=(lc > 0))
                    j = i * 128
                    nisa.scalar_tensor_tensor(dst=S[:, j:j + 128], data=ps[:, 0:128], op0=nl.multiply,
                                              operand0=X["scale"], op1=nl.add, operand1=S[:, j:j + 128])
        # softmax over the row's T tokens per head: P = exp(S - max) in bf16 and its fp32 sum in one ACT instruction
        mx, nmx, den, rden = X["MX"][x], X["NMX"][x], X["DEN"][x], X["RDEN"][x]
        nisa.tensor_reduce(dst=mx, op=nl.maximum, data=S, axis=1)
        nisa.tensor_scalar(dst=nmx, data=mx, op0=nl.multiply, operand0=-1.0, engine=VE)
        P = X["P"][x]
        nisa.activation(dst=P, op=nl.exp, data=S, bias=nmx, scale=1.0, reduce_op=nl.add, reduce_res=den,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        # o = P K: P's 128-token blocks transposed to [128, H] against the gathered latent as it is
        for ch in range(NCH):
            for t in range(KP):
                i = ch * KP + t
                j = i * 128
                g = 512 // H
                pp = PP[:, (i % g) * H:(i % g + 1) * H]
                nisa.nc_matmul(dst=pp, stationary=P[:, j:j + 128], moving=IB[0:H, 0:H], accumulate=False)
                pT = X["PTb"][i % 2]
                nisa.activation(dst=pT, op=nl.copy, data=pp)
                nisa.nc_matmul(dst=PO, stationary=pT, moving=Kv[ch][:, t * R:(t + 1) * R], accumulate=(i > 0))
        nisa.reciprocal(dst=rden, data=den)
        ob = X["OB"][x]
        nisa.tensor_scalar(dst=ob, data=PO, op0=nl.multiply, operand0=rden, engine=VE)
        nisa.dma_copy(dst=X["o"].ap(pattern=[[R, H], [1, R]], offset=k * H * R, scalar_offset=it, indirect_dim=0),
                      src=ob)
        if X["want_lse"]:
            ls = X["LS"][x]
            nisa.activation(dst=ls, op=nl.log, data=den)
            nisa.tensor_tensor(dst=ls, data1=ls, data2=mx, op=nl.add, engine=VE)
            nisa.dma_copy(dst=X["lse"].ap(pattern=[[1, H], [1, 1]], offset=k * H, scalar_offset=it, indirect_dim=0),
                          src=ls)

    def _psum(H: int, LC: int):
        """Every PSUM tile one 2 KB bank per partition: two K^T transpose banks, two score banks, the P^T bank, the
        P K accumulator, the q_lat^T bank."""
        PT = []
        for _ in range(2):
            PT.append(nl.ndarray((128, LC, 128), dtype=F32, buffer=nl.psum))
        PS = []
        for _ in range(2):
            PS.append(nl.ndarray((H, 512), dtype=F32, buffer=nl.psum))
        PP = nl.ndarray((128, 512), dtype=F32, buffer=nl.psum)
        PO = nl.ndarray((H, 512), dtype=F32, buffer=nl.psum)
        PQ = nl.ndarray((128, LC, H), dtype=F32, buffer=nl.psum)
        return PT, PS, PP, PO, PQ

    @nki.jit
    def kiln_dsa_slots_kernel(q_lat, kc, rows_t, bias, identb, scale: float, fp8: int, want_lse: int, rev: int,
                              qkw: int = 1):
        """q_lat bf16 [NI, RPI, H, R]; kc [cache rows / KP, KP R] the latent cache's pool rows (fp8 = 1: trn1's e4m3,
        see dsa_decode; else bf16); rows_t int32 [NI, RPI, 128, NCH] slot ch 128 + p of a row at [.., p, ch]; bias fp32
        [NI, RPI, NCH KP 128] (token t of slot ch 128 + p at (ch KP + t) 128 + p); identb bf16 [128, 128]. Returns o fp32
        [NI, RPI, H, R] (and lse fp32 [NI, RPI, H, 1])."""
        NI, rpi, H, R = q_lat.shape
        NCH = rows_t.shape[3]
        LC = R // 128
        T = NCH * KP * 128
        o = nl.ndarray((NI, rpi, H, R), dtype=F32, buffer=nl.shared_hbm)
        lse = None
        if want_lse:
            lse = nl.ndarray((NI, rpi, H, 1), dtype=F32, buffer=nl.shared_hbm)
        KD = kc.dtype
        if fp8:
            KD = nl.float8_e4m3
        X = dict(H=H, R=R, LC=LC, NCH=NCH, T=T, rpi=rpi, q_lat=q_lat, kc=kc, rows_t=rows_t, bias=bias, o=o, lse=lse,
                 want_lse=want_lse, qkw=qkw)
        IB = _sb((128, 128), BF16)
        nisa.dma_copy(dst=IB, src=identb)
        X["IB"] = IB
        X["scale"] = scale
        X["K"] = []
        for _ in range(2):
            kv = []
            for _ in range(NCH):
                kv.append(_sb((128, KP * R), KD))
            X["K"].append(kv)
        X["KT"] = []
        if qkw:
            X["KTA"] = _pair((128, LC, T), BF16)
            X["BR"] = _pair((1, T), F32)
            X["RSC"] = _sb((1, 128))
            nisa.memset(dst=X["RSC"], value=1.0 / scale)
        else:
            for _ in range(KTR):
                X["KT"].append(_sb((128, LC, 128), BF16))
        X["RS"] = _pair((128, NCH), I32)
        X["QR"] = _pair((H, R), BF16)
        X["QT"] = _pair((128, LC, H), BF16)
        X["S"] = _pair((H, T), F32)
        X["P"] = _pair((H, T), BF16)
        X["OB"] = _pair((H, R), F32)
        X["MX"] = _pair((H, 1), F32)
        X["NMX"] = _pair((H, 1), F32)
        X["DEN"] = _pair((H, 1), F32)
        X["RDEN"] = _pair((H, 1), F32)
        X["LS"] = _pair((H, 1), F32)
        X["PTb"] = _pair((128, H), BF16)
        cnt = _sb((1, 1), I32)
        nisa.iota(dst=cnt, pattern=[[0, 1]], offset=NI, channel_multiplier=0)
        rt = nisa.register_alloc()
        nisa.register_load(rt, cnt)

        def body(it):
            X["psum"] = _psum(H, LC)
            for k in range(rpi):
                _row(X, it, k)

        nl.fori_loop(0, rt, body)
        if want_lse:
            return o, lse
        return o
else:
    kiln_dsa_slots_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source, passed as the static argument `rev` (LNL's compile-cache key does not
    include NKI kernel source: CLAUDE.md)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_dsa_slots_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev()


def attend(q_lat: torch.Tensor, kc: torch.Tensor, rows: torch.Tensor, bias: torch.Tensor, scale: float,
           lse: bool = False):
    """o [N, H, R] fp32 (and with lse the log-sum-exp [N, H] fp32): the kernel on a Neuron device, emulate()
    elsewhere. q_lat [N, H, R], kc the latent cache [cache rows, 1, R] or [cache rows, R] (fp8 or bf16), rows int [N,
    NS] each slot's pool row (NS a multiple of 128), bias fp32 [N, NS, KP]."""
    kc2 = kc.reshape(kc.shape[0], -1)
    if q_lat.device.type == "cpu":
        return emulate(q_lat, kc2, rows, bias, scale, lse)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if kiln_dsa_slots_kernel is None:
        raise RuntimeError("the NKI DSA slots kernel needs the nki package (the Neuron venv)")
    N, H, R = q_lat.shape
    NS = rows.shape[1]
    if NS % 128:
        raise ValueError(f"dsa_slots: {NS} slots per row is not a multiple of 128")
    NCH = NS // 128
    rpi = RPI
    Np = -(-N // rpi) * rpi
    q = q_lat.to(torch.bfloat16)
    rw, bs = rows, bias.float()
    if Np != N:  # padded rows read pool row 0, every slot masked
        q = torch.cat([q, q.new_zeros(Np - N, H, R)])
        rw = torch.cat([rw, rw.new_zeros(Np - N, NS)])
        bs = torch.cat([bs, bs.new_full((Np - N, NS, KP), NEG_INF)])
    fp8 = kc2.dtype == torch.float8_e4m3fn
    kcp = kc2.reshape(kc2.shape[0] // KP, KP * R)  # pool rows (a pool never straddles a page)
    NI = Np // rpi
    rows_t = rw.to(torch.int32).reshape(NI, rpi, NCH, 128).permute(0, 1, 3, 2).contiguous()
    bias_t = bs.reshape(NI, rpi, NCH, 128, KP).permute(0, 1, 2, 4, 3).reshape(NI, rpi, NCH * KP * 128).contiguous()
    eye = torch.eye(128, device=q_lat.device).to(torch.bfloat16)
    out = wrap_nki(kiln_dsa_slots_kernel)[platform.nki_grid()](
        q_lat=q.reshape(NI, rpi, H, R).contiguous(), kc=kcp, rows_t=rows_t, bias=bias_t, identb=eye,
        scale=float(scale), fp8=int(fp8), want_lse=int(lse), rev=REV, qkw=QKW)
    if lse:
        o, ls = out
        return o.reshape(Np, H, R)[:N], ls.reshape(Np, H)[:N]
    return out.reshape(Np, H, R)[:N]
