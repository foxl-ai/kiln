"""The attention of a pooled DSA layer's decode step (GLM-5.3-Flash, models/glm5_next.py) over the SELECTED pools only,
as one NKI kernel for NeuronCore-v2 (trn1), behind KILN_DSA_DECODE_KERNEL=nki.

Why: the XLA form (models/mla.py _core with the selection as an additive mask) attends every key of the bucket,
8448 at page bucket 264: it gathers the whole fp8 latent cache of each row, converts it to bf16 and, inside a decode
layer-group graph, spilled 0.6-0.9 GB per DSA layer to HBM; the DSA attention segment grew 1.25 / 1.66 / 2.60 /
4.12 / 6.07 / 11.95 ms at 1 / 4 / 8 / 16 / 32 / 64 rows per DP group (~0.17 ms per row per layer; neuron-explorer
replays of the layers 12-23 decode graph on 32 cores, rank 0, trn1.32xlarge, SDK 2.32, 2026-10-04,
docs/neuron-notes.md "Where a GLM-5.3-Flash decode step goes, from a device profile"). The selection keeps
index_topk / index_kpool = 512 pools of 4 tokens (2048 tokens) plus the query's own incomplete pool, so this kernel
reads ~1/4 of those bytes and never materialises a bf16 copy.

What it computes, per sequence row b and head h (absorbed MLA, NoPE: the latent is both key and value):
s[t] = scale * q_lat[b, h] . K[t] + bias[b, t] over the gathered tokens t, p = softmax(s), o[b, h] = sum_t p[t] K[t].
The gathered tokens are NS = NCH * 128 pool slots of KP consecutive cache rows each (a pool never crosses a page:
page_size is a multiple of KP), slot j of row b starting at cache row rows[b, j]; bias is 0 for a token to attend
and NEG_INF for padding (unselected slots, the tail pool's invisible tokens). With NEG_INF on every non-selected key
the mask form's softmax is this one (exp underflows to 0), so the two differ only in rounding: here the scores stay
fp32 (the XLA form rounds them to bf16 before the softmax), p is bf16 and the products accumulate in fp32 as there.

Layout: per row, the NCH gathers put slot (ch, p) on partition p as KP x R fp8 bytes (one DMA of 128 descriptors
per chunk, vector_offset = the slots' row indices). The scores need R on the partitions: each 128-token block is
transposed on the tensor engine (against a bf16 identity, exact in the fp32 PSUM) and copied to bf16, then
q_lat^T (4 column blocks of the row's H heads, transposed once for all rows at the start) against it gives [H, 128]
scores of the block. p's blocks are transposed back to [128, H] (tokens on the partitions) and the latent itself,
fp8 as gathered, is the moving operand of the p K products, accumulated over the row's blocks into [H, R].
"""

from __future__ import annotations

import os

import torch

NEG_INF = -1e30  # models/decoder.NEG_INF
KTR = 4  # K^T tiles in flight
VCOPY = os.environ.get("KILN_DSA_DECODE_VCOPY", "0") == "1"  # debug
ACT_COPY = True  # the K^T copies alternate between the vector and scalar engines
NCH = 5  # 128-slot chunks per row: 512 selected pools + the tail pool, padded to 640 slots
KP = 4  # tokens per pool (GLM-5.3-Flash index_kpool)


def emulate(q_lat: torch.Tensor, kc: torch.Tensor, rows: torch.Tensor, bias: torch.Tensor, scale: float) -> torch.Tensor:
    """The kernel's arithmetic in torch (CPU): q_lat [B, H, R], kc [N, R] (the latent cache's token rows), rows [B, NS]
    int (each slot's pool row: cache rows KP rows .. KP rows + KP - 1), bias [B, NS, KP] fp32 -> o [B, H, R] fp32."""
    B, H, R = q_lat.shape
    NS = rows.shape[1]
    tok = (rows.long().unsqueeze(-1) * KP + torch.arange(KP, device=rows.device)).reshape(B, NS * KP)
    # the kernel's bf16 operands; an fp32 cache (CPU tests) stays fp32 throughout, the mask form's arithmetic there
    rnd = (lambda t: t.to(torch.bfloat16).float()) if kc.dtype != torch.float32 else (lambda t: t.float())
    K = rnd(kc[tok])  # [B, T, R]
    s = torch.einsum("bhr,btr->bht", rnd(q_lat), K) * scale + bias.reshape(B, 1, NS * KP)
    p = rnd(torch.softmax(s, dim=-1))
    return torch.einsum("bht,btr->bhr", p, K)


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    F32, BF16, I32 = nl.float32, nl.bfloat16, nl.int32

    @nki.jit
    def kiln_dsa_decode_kernel(q_lat, kc, rows_t, bias_t, identb, scale: float, fp8: int, rev: int, dbg: int = 0,
                               spl: int = 0):
        """q_lat bf16 [B, H, R]; kc [N / KP, KP R] the latent cache's token rows, KP to a row (pool rows), fp8 e4m3fn (fp8 = 1; the graph is compiled
        with --experimental-unsafe-fp8e4m3fn-as-fp8e4m3 and Kiln's FP8 KV values are clamped to 240, so it is trn1's
        e4m3, which nc_matmul takes) or bf16; rows_t int32 [128, B, NCH] the pool row of slot ch * 128 + p of row b
        at [p, b, ch]; bias_t fp32 [B, NCH, KP, 128] (token t of slot ch * 128 + p at
        [b, ch, t, p]); identb bf16 [128, 128] the identity; rev: this module's kernel source revision. Returns o
        fp32 [B, H, R]."""
        B, H, R = q_lat.shape
        LC = R // 128
        assert kc.shape[1] == KP * R
        # LNC split (spl 1, grid 2: trn2 at LNC=2; KILN_LNC_SPLIT names dsa_decode): each program (physical core)
        # attends its half of the rows, b_lo .. b_hi - 1 (rows are independent, o rows disjoint), and both barrier
        # on o before the kernel ends; q_lat^T is built by both. spl 0 or one program: the kernel as before.
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if spl == 1 and nl.program_ndim() != 0 else (1, 0)
        b_lo, b_hi = (0, B) if npg == 1 else ((0, (B + 1) // 2) if pid == 0 else ((B + 1) // 2, B))
        T = NCH * KP * 128
        o = nl.ndarray((B, H, R), dtype=F32, buffer=nl.shared_hbm)
        sd = nl.ndarray((B, H, NCH * KP * 128), dtype=F32, buffer=nl.shared_hbm)  # debug: the scores
        kd = nl.ndarray((B, NCH, 128, KP * R), dtype=kc.dtype, buffer=nl.shared_hbm)  # debug: the gathered latent
        ktd = nl.ndarray((B, NCH, KP, 128, LC, 128), dtype=BF16, buffer=nl.shared_hbm)  # debug: the transposed latent
        IB = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=IB, src=identb)
        rs = nl.ndarray((128, B * NCH), dtype=I32, buffer=nl.sbuf)
        nisa.dma_copy(dst=rs, src=rows_t.reshape((128, B * NCH)))
        kf = kc  # [N / KP, KP R]: one pool per indexed row (an indirect DMA reads within one row of its tensor)
        # Every tile allocated once, in rings (SBUF is 192 KB per partition: per-row tiles allocated inside the
        # unrolled row loop overran it and aliased, which gave a chunk's scores from the wrong data).
        KD = nl.float8_e4m3 if fp8 else kc.dtype
        Kring, KTr, PTr, PSr, PPr, PTb, BSr, Sr, Pr, POr, OBr = [], [], [], [], [], [], [], [], [], [], []
        K2ring = []
        for _ in range(2 * NCH):
            Kring.append(nl.ndarray((128, KP * R), dtype=KD, buffer=nl.sbuf))
            if VCOPY:
                K2ring.append(nl.ndarray((128, KP * R), dtype=KD, buffer=nl.sbuf))
        for _ in range(KTR):
            KTr.append(nl.ndarray((128, LC, 128), dtype=BF16, buffer=nl.sbuf))
        # PSUM: every tile exactly one 2 KB bank per partition, allocated first (a smaller tile allocated between
        # them would leave the next one straddling two banks)
        for _ in range(2):
            PTr.append(nl.ndarray((128, LC, 128), dtype=F32, buffer=nl.psum))
        for _ in range(2):
            PSr.append(nl.ndarray((H, 512), dtype=F32, buffer=nl.psum))
        PPb = nl.ndarray((128, 512), dtype=F32, buffer=nl.psum)  # p^T blocks, [128, H] each, in turn
        POr.append(nl.ndarray((H, R), dtype=F32, buffer=nl.psum))
        for _ in range(2):
            PTb.append(nl.ndarray((128, H), dtype=BF16, buffer=nl.sbuf))
            BSr.append(nl.ndarray((H, T), dtype=F32, buffer=nl.sbuf))
            Sr.append(nl.ndarray((H, T), dtype=F32, buffer=nl.sbuf))
            Pr.append(nl.ndarray((H, T), dtype=BF16, buffer=nl.sbuf))
            OBr.append(nl.ndarray((H, R), dtype=F32, buffer=nl.sbuf))
        mx = nl.ndarray((H, 1), dtype=F32, buffer=nl.sbuf)
        nmx = nl.ndarray((H, 1), dtype=F32, buffer=nl.sbuf)
        den = nl.ndarray((H, 1), dtype=F32, buffer=nl.sbuf)
        rden = nl.ndarray((H, 1), dtype=F32, buffer=nl.sbuf)
        # q_lat^T: [128 r, LC, B H] (column c = b H + h), by PE transposes of 128-row blocks of q_lat [B H, R].
        N = B * H
        QT = nl.ndarray((128, LC, N), dtype=BF16, buffer=nl.sbuf)
        q2 = q_lat.reshape((N, R))
        for t0 in range(0, N, 128):
            n = min(128, N - t0)
            qr = nl.ndarray((128, R), dtype=BF16, buffer=nl.sbuf)
            nisa.dma_copy(dst=qr[0:n, :], src=q2[t0:t0 + n, :])
            pq = PTr[(t0 // 128) % 2]
            for lc in range(LC):
                nisa.nc_matmul(dst=pq[:, lc, 0:n], stationary=qr[0:n, lc * 128:(lc + 1) * 128], moving=IB[0:n, 0:n],
                               accumulate=False)
            nisa.tensor_copy(dst=QT[:, :, t0:t0 + n], src=pq[:, :, 0:n], engine=nisa.vector_engine)
        bf = bias_t.reshape((B, T))
        for b in range(b_lo, b_hi):
            x = b % 2
            # the row's slots: NCH tiles of [128 p, KP R] (trn1's e4m3 for an fp8 cache: under
            # --experimental-unsafe-fp8e4m3fn-as-fp8e4m3 the cache input is that type too, and an e4m3fn tile beside
            # it fails NCC_EOCP001, "Mixed use of the two mutually-exclusive types")
            Kv = []
            for ch in range(NCH):
                kch = Kring[x * NCH + ch]
                nisa.dma_copy(dst=kch, src=kf.ap(pattern=[[KP * R, 128], [1, KP * R]], offset=0,
                                                vector_offset=rs.ap(pattern=[[B * NCH, 128], [1, 1]], offset=b * NCH + ch),
                                                indirect_dim=0))
                if VCOPY:  # debug: the latent through a vector-engine copy before the tensor engine reads it
                    k2 = K2ring[x * NCH + ch]
                    nisa.tensor_copy(dst=k2, src=kch, engine=nisa.vector_engine)
                    kch = k2
                Kv.append(kch)
                if dbg == 2:
                    nisa.dma_copy(dst=kd[b, ch], src=kch)
            bs = BSr[x]  # the row's bias on each head's partition
            for h in range(H):
                nisa.dma_copy(dst=bs[h:h + 1, :], src=bf[b:b + 1, :])
            S = Sr[x]
            for ch in range(NCH):
                for t in range(KP):
                    i = ch * KP + t
                    pt = PTr[i % 2]
                    for lc in range(LC):  # the block's latent, R on the partitions
                        c0 = t * R + lc * 128
                        nisa.nc_matmul(dst=pt[:, lc, :], stationary=Kv[ch][:, c0:c0 + 128], moving=IB,
                                       accumulate=False)
                    kt = KTr[i % KTR]
                    if i % 2 == 0 or not ACT_COPY:
                        nisa.tensor_copy(dst=kt, src=pt, engine=nisa.vector_engine)
                    else:
                        nisa.activation(dst=kt, op=nl.copy, data=pt)
                    if dbg == 3:
                        nisa.dma_copy(dst=ktd[b, ch, t], src=kt)
                    # One block's scores per PSUM tile, read right after (on trn1 a block accumulated into the second
                    # 128-column slice group of a PSUM tile shared with three others kept only its last term:
                    # tools/probe_nki_scores.py mode 3, device against nki.simulate, 2026-10-04)
                    ps = PSr[i % 2]
                    for lc in range(LC):
                        nisa.nc_matmul(dst=ps[:, 0:128], stationary=QT[:, lc, b * H:(b + 1) * H],
                                       moving=kt[:, lc, :], accumulate=(lc > 0))
                    j = i * 128
                    if dbg == 4:  # the raw scores of the block
                        nisa.tensor_copy(dst=S[:, j:j + 128], src=ps[:, 0:128], engine=nisa.vector_engine)
                        continue
                    # the block's scores, scaled, plus the bias
                    nisa.scalar_tensor_tensor(dst=S[:, j:j + 128], data=ps[:, 0:128], op0=nl.multiply, operand0=scale,
                                              op1=nl.add, operand1=bs[:, j:j + 128])
            if dbg:
                nisa.dma_copy(dst=sd[b], src=S)
            # softmax over the row's T tokens, per head (partition); exp in place
            nisa.tensor_reduce(dst=mx, op=nl.maximum, data=S, axis=1)
            nisa.tensor_scalar(dst=nmx, data=mx, op0=nl.multiply, operand0=-1.0, engine=nisa.vector_engine)
            nisa.activation(dst=S, op=nl.exp, data=S, bias=nmx, scale=1.0)
            nisa.tensor_reduce(dst=den, op=nl.add, data=S, axis=1)
            P = Pr[x]
            nisa.tensor_copy(dst=P, src=S, engine=nisa.vector_engine)
            # o = p K: p's 128-token blocks transposed to [128, H], against the gathered latent as it is
            po = POr[0]
            for ch in range(NCH):
                for t in range(KP):
                    i = ch * KP + t
                    j = i * 128
                    pp = PPb[:, (i % (512 // H)) * H:(i % (512 // H) + 1) * H]
                    nisa.nc_matmul(dst=pp, stationary=P[:, j:j + 128], moving=IB[0:H, 0:H], accumulate=False)
                    pT = PTb[i % 2]
                    nisa.activation(dst=pT, op=nl.copy, data=pp)
                    nisa.nc_matmul(dst=po, stationary=pT, moving=Kv[ch][:, t * R:(t + 1) * R], accumulate=(i > 0))
            nisa.reciprocal(dst=rden, data=den)
            ob = OBr[x]
            nisa.tensor_scalar(dst=ob, data=po, op0=nl.multiply, operand0=rden, engine=nisa.vector_engine)
            nisa.dma_copy(dst=o[b], src=ob)
        if npg > 1:  # LNC: both programs' rows written before either ends
            nisa.core_barrier(data=o, cores=(0, 1))
        if dbg == 2:
            return o, kd
        if dbg == 3:
            return o, ktd
        if dbg:
            return o, sd
        return o
else:
    kiln_dsa_decode_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source, passed as the static argument `rev` (LNL's compile-cache key does
    not include NKI kernel source: CLAUDE.md, kernels/delta_rule.py _kernel_rev)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_dsa_decode_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev()


def attend(q_lat: torch.Tensor, kc: torch.Tensor, rows: torch.Tensor, bias: torch.Tensor, scale: float, dbg: int = 0):
    """o [B, H, R] fp32 (the kernel on a Neuron device, emulate() elsewhere): q_lat [B, H, R], kc the latent cache
    [N, 1, R] or [N, R], rows int [B, NS] each slot's pool row (glm5_next.decode_slots), bias fp32 [B, NS, KP]."""
    kc2 = kc.reshape(kc.shape[0], -1)
    if q_lat.device.type == "cpu":
        return emulate(q_lat, kc2, rows, bias, scale)
    fp8 = kc2.dtype == torch.float8_e4m3fn
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if kiln_dsa_decode_kernel is None:
        raise RuntimeError("the NKI DSA decode kernel needs the nki package (the Neuron venv)")
    B, NS = rows.shape
    kc2 = kc2.reshape(kc2.shape[0] // KP, KP * kc2.shape[1])  # pool rows (a pool never straddles a page)
    rows_t = rows.to(torch.int32).reshape(B, NCH, 128).permute(2, 0, 1).contiguous()
    bias_t = bias.float().reshape(B, NCH, 128, KP).permute(0, 1, 3, 2).contiguous()
    eye = torch.eye(128, device=q_lat.device)
    spl = {"spl": 1} if platform.nki_grid() == 2 and platform.lnc_split("dsa_decode") else {}  # in the key only then
    return wrap_nki(kiln_dsa_decode_kernel)[platform.nki_grid()](
        q_lat=q_lat.to(torch.bfloat16).contiguous(), kc=kc2, rows_t=rows_t,
        bias_t=bias_t, identb=eye.to(torch.bfloat16), scale=float(scale), fp8=int(fp8), rev=REV, dbg=dbg, **spl)


DECODE_KERNEL_FAMILIES = ("trn1", "trn2")  # where the decode-path checks and A/Bs ran (kernels/kda_decode.py)


def _default_kernel() -> str:
    """KILN_DSA_DECODE_KERNEL unset: nki on a trn1 / trn2 target, xla elsewhere (inf2, trn3 and a host without a Neuron
    device)."""
    from .. import platform

    t = platform.target()
    return "nki" if t is not None and platform.family_of(t) in DECODE_KERNEL_FAMILIES else "xla"


KERNEL = os.environ.get("KILN_DSA_DECODE_KERNEL") or _default_kernel()
if KERNEL not in ("xla", "nki"):
    raise ValueError(f"KILN_DSA_DECODE_KERNEL must be xla or nki, not {KERNEL!r}")
