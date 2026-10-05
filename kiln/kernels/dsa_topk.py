"""Exact top-k selection of DSA index scores as one NKI kernel for NeuronCore-v2 (trn1), behind
KILN_DSA_SELECT=nki: GLM-5.3-Flash's pooled indexer (models/glm5_next.block_mask, keep 512 of 2112
pools at 8448 keys) and DeepSeek Sparse Attention's index_topk (models/dsa_select.topk_mask).

Why a kernel: the torch selections are a chain of ~32-48 dependent rounds of `count(s >= m)` over a
few KB of scores (glm5_next.block_mask "bisect", dsa_select "bisect"), and each round is several XLA
ops with engine hand-offs between them. On trn1 that chain cost ~1.5 ms of GLM-5.3-Flash's 3.66 ms
DSA block at 8448 keys (decode, 8 rows, attention TP 8; docs/neuron-notes.md "Where a GLM-5.3-Flash
decode step goes"), more than attending every key (2.14 ms). Inside one kernel a round is a handful of
instructions on [128, W] SBUF tiles.

What it selects (the tie rule, the same as dsa_select.reference_mask): per row of P scores, the
`keep` largest; every selected score >= every unselected one; among the scores equal to the
keep-th largest t, the lowest indices. With `vis_only`, only scores above VISIBLE (= NEG_INF / 2)
can be selected, so a row with fewer than `keep` such scores selects all of them and nothing else:
exactly glm5_next.block_mask's set, whose invisible candidates score index + NEG_INF = NEG_INF.
Comparisons are IEEE float32 (-0.0 == +0.0); scores must not be NaN. The device's vector engine
may flush subnormals to zero, so on the device a subnormal score compares like 0 (CPU emulate()
keeps them).

How (emulate() is the same arithmetic in torch):
1. Sign: f = count(s >= 0) < keep, i.e. t < 0. Then s2 = -s and k2 = P + 1 - keep (the keep-th
   largest of s is minus the k2-th largest of -s), else s2 = s, k2 = keep; either way the k2-th
   largest of s2 is t2 = |t| >= 0 (+0.0 when t is a zero).
2. Radix over the bit pattern of t2: for a non-negative float the int32 bit pattern orders like the
   value (+inf = 0x7f800000; patterns above it are NaN and compare false), so the largest pattern x
   with count(s2 >= float(x)) >= k2 is t2's, built from bit 30 down in 31 rounds: candidate =
   prefix | 2^b (an integer OR on the bit pattern), its count by one float compare-and-sum, kept if
   >= k2. The threshold is a bit pattern, so no float arithmetic ever rounds it: t = +-float(x)
   exactly. (dsa_select bisects the float ORDER with sqrt and means instead, because an in-graph
   bitcast does not lower through LNL; inside NKI a tile view is free.)
3. Ties: above = s > t, room = keep - count(above) (>= 1), and of the tied (s == t) the first `room`
   by index: the largest lim with count(tied & j < lim) < room, by binary lifting over the position
   (nbits rounds, 2^nbits > P), then tied & j <= lim.
4. Out: additive float32, 0 where selected and NEG_INF where not (models/decoder.NEG_INF), per
   score, or (kp > 1) per token of each score's pool of kp tokens: GLM-5.3-Flash's block_mask
   expands the pool selection to its tokens and, with `tail`, adds the query's own incomplete pool,
   i.e. the tokens of the non-candidate pools (under a prefix visibility those are exactly the
   tokens from kp floor(visible / kp) on, block_mask's tail; the invisible ones among them are
   masked by the visibility the attention adds anyway). Done in the kernel because the same
   expansion in torch ([B, Q, P] -> [B, Q, P, kp] -> [B, Q, L]) plus the tail cost ~1.1 ms inside
   the decode layer on trn1 against ~0.1 ms for the whole selection here (docs/neuron-notes.md).

Layout: a row's P scores are cut into g pieces of W = P / g, one per partition (g a power of two,
rows x g <= 128 when it can: 8 decode rows of 2112 pools fill 128 partitions with 132 each), so a
round's compare-and-sum reads W values per partition; a row's count is the sum of its g partials,
one fp32 matmul with a block-diagonal 0/1 matrix (exact: counts < 2^24) that leaves every piece its
row's total. More than 128 / g rows (a prefill chunk's queries) are processed in tiles of 128
partitions, TILES at a time with their instructions interleaved. The input is the scores [rows, P]
reshaped to [rows * g, W] (a view), the output [rows * g, W kp].
"""

from __future__ import annotations

import os

import torch

NEG_INF = -1e30  # models/decoder.NEG_INF
VISIBLE = -5e29  # scores at or below it are invisible candidates (index + NEG_INF) under vis_only
BIG = float(2**20)  # the position of an untied score in the tie search (above any limit, exact)
MAX_W = 4096  # scores per partition the kernel holds (SBUF: ~10 fp32 tiles of [128, W] live)
TILES = int(os.environ.get("KILN_DSA_TILES", 2))  # row tiles of 128 partitions processed together


def pieces(rows: int, n: int, group: bool = True) -> int:
    """g: pieces per row (a power of two dividing n) so that rows x g fills up to 128 partitions."""
    g = 1
    if group:
        while rows * g * 2 <= 128 and n % (g * 2) == 0:
            g *= 2
    return g


def _out(sel: torch.Tensor, vis: torch.Tensor, kp: int, tail: bool) -> torch.Tensor:
    if tail:
        sel = sel | ~vis
    out = torch.where(sel, 0.0, NEG_INF)
    return out.repeat_interleave(kp, dim=-1) if kp > 1 else out


def emulate(sc: torch.Tensor, keep: int, vis_only: bool = True, kp: int = 1, tail: bool = False) -> torch.Tensor:
    """The kernel's arithmetic in torch (CPU): sc [R, P] float32 -> additive float32 [R, P kp]."""
    if tail and not vis_only:
        raise ValueError("tail needs vis_only")
    s = sc.float()
    R, P = s.shape
    vis = s > VISIBLE
    if keep >= P:
        sel = vis if vis_only else torch.ones_like(vis)
        return _out(sel, vis, kp, tail)
    f = (s >= 0).float().sum(-1, keepdim=True) < keep
    sg = torch.where(f, -1.0, 1.0)
    s2 = s * sg
    k2 = torch.where(f, float(P + 1 - keep), float(keep))
    pre = torch.zeros(R, 1, dtype=torch.int32)
    for b in range(30, -1, -1):
        cand = pre | (1 << b)
        ok = (s2 >= cand.view(torch.float32)).float().sum(-1, keepdim=True) >= k2
        pre = pre | (ok.to(torch.int32) << b)
    th = pre.view(torch.float32) * sg
    above = s > th
    room = keep - above.float().sum(-1, keepdim=True)
    j = torch.arange(P, dtype=torch.float32)
    tj = torch.where(s == th, j, BIG)
    lim = torch.zeros(R, 1)
    for b in range(P.bit_length() - 1, -1, -1):
        ok = (tj < lim + 2.0**b).float().sum(-1, keepdim=True) < room
        lim = lim + ok.float() * 2.0**b
    sel = above | (tj <= lim)
    if vis_only:
        sel &= vis
    return _out(sel, vis, kp, tail)


# --- kernel -------------------------------------------------------------------------------------

try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    F32, I32, BF16 = nl.float32, nl.int32, nl.bfloat16
    VE = nisa.vector_engine

    def _sb(shape, dtype=None):
        return nl.ndarray(shape, dtype=dtype or F32, buffer=nl.sbuf)

    def _total(cp, n, g, G):
        """A row's count on every one of its g pieces: [n, 1] partials -> [n, 1] (PSUM)."""
        if g == 1:
            return cp
        ps = nl.ndarray((n, 1), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps, stationary=G[0:n, 0:n], moving=cp)
        return ps

    def _consts(W: int, g: int, lg: int):
        """Constants: position within the piece, w (every partition); with g > 1 the block-diagonal
        0/1 matrix G[k, m] = [k // g == m // g] (sums a row's pieces) and each partition's piece
        offset (p % g) W; 2^b as int32 bit patterns, bits[30 - b] (shifts of one iota: exact integer
        bit operations)."""
        wio_i = _sb((128, W), I32)
        nisa.iota(dst=wio_i, pattern=[[1, W]], offset=0, channel_multiplier=0)
        wio = _sb((128, W))
        nisa.tensor_copy(dst=wio, src=wio_i, engine=VE)
        po = _sb((128, 1))
        G = None
        if g > 1:
            ipi = _sb((128, 1), I32)
            nisa.iota(dst=ipi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
            kq_i = _sb((128, 1), I32)
            nisa.tensor_scalar(dst=kq_i, data=ipi, op0=nl.right_shift, operand0=lg, engine=VE)
            kq = _sb((128, 1))
            nisa.tensor_copy(dst=kq, src=kq_i, engine=VE)
            mq_i = _sb((128, 128), I32)
            nisa.iota(dst=mq_i, pattern=[[1, 128]], offset=0, channel_multiplier=0)
            nisa.tensor_scalar(dst=mq_i, data=mq_i, op0=nl.right_shift, operand0=lg, engine=VE)
            mq = _sb((128, 128))
            nisa.tensor_copy(dst=mq, src=mq_i, engine=VE)
            G = _sb((128, 128))
            nisa.tensor_scalar(dst=G, data=mq, op0=nl.equal, operand0=kq, engine=VE)
            po_i = _sb((128, 1), I32)
            nisa.tensor_scalar(dst=po_i, data=ipi, op0=nl.bitwise_and, operand0=g - 1, engine=VE)
            nisa.tensor_scalar(dst=po, data=po_i, op0=nl.multiply, operand0=float(W), engine=VE)
        else:
            nisa.memset(dst=po, value=0.0)
        top = _sb((128, 1), I32)
        nisa.iota(dst=top, pattern=[[0, 1]], offset=1 << 30, channel_multiplier=0)
        bits = [top]
        for b in range(29, -1, -1):
            bt = _sb((128, 1), I32)
            nisa.tensor_scalar(dst=bt, data=top, op0=nl.right_shift, operand0=30 - b, engine=VE)
            bits.append(bt)
        return wio, po, G, bits

    def _select(S, R0, NS, W: int, g: int, G, po, wio, bits, keep: int, nbits: int, vis_only: int, kp: int,
                tail: int, out):
        """Steps 1-4 of the module docstring for a group of row tiles whose scores S[i] [NS[i], W]
        are in SBUF (rows R0[i]..): their selection written to out [N, W kp]."""
        P = g * W
        nt = len(NS)
        S2 = []
        SG = []
        K2 = []
        PRE = []
        SCR = []
        for i in range(nt):
            SCR.append(_sb((NS[i], W)))
        # 1. sign
        for i in range(nt):
            r0 = R0[i]
            n = NS[i]
            cp = _sb((n, 1))
            nisa.tensor_scalar_reduce(dst=SCR[i], data=S[i], op0=nl.greater_equal, operand0=0.0,
                                      reduce_op=nl.add, reduce_res=cp)
            tot = _total(cp, n, g, G)
            f = _sb((n, 1))
            nisa.tensor_scalar(dst=f, data=tot, op0=nl.less, operand0=float(keep), engine=VE)
            sg = _sb((n, 1))
            nisa.tensor_scalar(dst=sg, data=f, op0=nl.multiply, operand0=-2.0, op1=nl.add, operand1=1.0,
                               engine=VE)
            SG.append(sg)
            k2 = _sb((n, 1))
            nisa.tensor_scalar(dst=k2, data=f, op0=nl.multiply, operand0=float(P + 1 - 2 * keep), op1=nl.add,
                               operand1=float(keep), engine=VE)
            K2.append(k2)
            s2 = _sb((n, W))
            nisa.tensor_scalar(dst=s2, data=S[i], op0=nl.multiply, operand0=sg, engine=VE)
            S2.append(s2)
            pre = _sb((n, 1), I32)
            nisa.memset(dst=pre, value=0)
            PRE.append(pre)
        # 2. radix over t2's bit pattern
        for b in range(30, -1, -1):
            for i in range(nt):
                r0 = R0[i]
                n = NS[i]
                cand = _sb((n, 1), I32)
                nisa.tensor_tensor(dst=cand, data1=PRE[i], data2=bits[30 - b][0:n], op=nl.bitwise_or, engine=VE)
                cp = _sb((n, 1))
                nisa.tensor_scalar_reduce(dst=SCR[i], data=S2[i], op0=nl.greater_equal, operand0=cand.view(F32),
                                          reduce_op=nl.add, reduce_res=cp)
                tot = _total(cp, n, g, G)
                inc = _sb((n, 1), I32)
                nisa.tensor_scalar(dst=inc, data=tot, op0=nl.greater_equal, operand0=K2[i], op1=nl.multiply,
                                   operand1=float(1 << b), engine=VE)
                nisa.tensor_tensor(dst=PRE[i], data1=PRE[i], data2=inc, op=nl.bitwise_or, engine=VE)
        # 3. above, room, ties
        AB = []
        ROOM = []
        TJ = []
        LIM = []
        for i in range(nt):
            r0 = R0[i]
            n = NS[i]
            th = _sb((n, 1))
            nisa.tensor_scalar(dst=th, data=PRE[i].view(F32), op0=nl.multiply, operand0=SG[i], engine=VE)
            ab = _sb((n, W))
            abp = _sb((n, 1))
            nisa.tensor_scalar_reduce(dst=ab, data=S[i], op0=nl.greater, operand0=th, reduce_op=nl.add,
                                      reduce_res=abp)
            AB.append(ab)
            abt = _total(abp, n, g, G)
            room = _sb((n, 1))
            nisa.tensor_scalar(dst=room, data=abt, op0=nl.multiply, operand0=-1.0, op1=nl.add,
                               operand1=float(keep), engine=VE)
            ROOM.append(room)
            tied = _sb((n, W))
            nisa.tensor_scalar(dst=tied, data=S[i], op0=nl.equal, operand0=th, engine=VE)
            tj = _sb((n, W))  # w where tied, w + BIG elsewhere
            nisa.scalar_tensor_tensor(dst=tj, data=tied, op0=nl.multiply, operand0=-BIG, op1=nl.add,
                                      operand1=wio[0:n])
            nisa.tensor_scalar(dst=tj, data=tj, op0=nl.add, operand0=BIG, engine=VE)
            TJ.append(tj)
            lim = _sb((n, 1))
            nisa.memset(dst=lim, value=0.0)
            LIM.append(lim)
        for b in range(nbits - 1, -1, -1):
            for i in range(nt):
                r0 = R0[i]
                n = NS[i]
                thr = _sb((n, 1))  # lim + 2^b in this piece's positions
                nisa.tensor_scalar(dst=thr, data=LIM[i], op0=nl.add, operand0=float(1 << b), op1=nl.subtract,
                                   operand1=po[0:n], engine=VE)
                cp = _sb((n, 1))
                nisa.tensor_scalar_reduce(dst=SCR[i], data=TJ[i], op0=nl.less, operand0=thr, reduce_op=nl.add,
                                          reduce_res=cp)
                tot = _total(cp, n, g, G)
                inc = _sb((n, 1))
                nisa.tensor_scalar(dst=inc, data=tot, op0=nl.less, operand0=ROOM[i], op1=nl.multiply,
                                   operand1=float(1 << b), engine=VE)
                nisa.tensor_tensor(dst=LIM[i], data1=LIM[i], data2=inc, op=nl.add, engine=VE)
        # 4. out
        for i in range(nt):
            r0 = R0[i]
            n = NS[i]
            lp = _sb((n, 1))
            nisa.tensor_tensor(dst=lp, data1=LIM[i], data2=po[0:n], op=nl.subtract, engine=VE)
            sel = _sb((n, W))
            nisa.scalar_tensor_tensor(dst=sel, data=TJ[i], op0=nl.less_equal, operand0=lp, op1=nl.add,
                                      operand1=AB[i])
            if vis_only:
                vm = _sb((n, W))
                nisa.tensor_scalar(dst=vm, data=S[i], op0=nl.greater, operand0=VISIBLE, engine=VE)
                nisa.tensor_tensor(dst=sel, data1=sel, data2=vm, op=nl.multiply, engine=VE)
                if tail:  # sel or not visible (disjoint): sel + 1 - vm
                    nisa.scalar_tensor_tensor(dst=sel, data=vm, op0=nl.subtract, operand0=1.0, op1=nl.subtract,
                                              operand1=sel)
                    nisa.tensor_scalar(dst=sel, data=sel, op0=nl.multiply, operand0=-1.0, engine=VE)
            o = _sb((n, W))
            nisa.tensor_scalar(dst=o, data=sel, op0=nl.subtract, operand0=1.0, op1=nl.multiply,
                               operand1=-NEG_INF, engine=VE)
            if kp > 1:  # each score's value on its kp tokens
                ot = _sb((n, W, kp))
                for e in range(kp):
                    nisa.tensor_copy(dst=ot[:, :, e], src=o, engine=VE)
                nisa.dma_copy(dst=out.ap(pattern=[[W * kp, n], [1, W * kp]], offset=r0 * W * kp),
                              src=ot.reshape((n, W * kp)))
            else:
                nisa.dma_copy(dst=out.ap(pattern=[[W, n], [1, W]], offset=r0 * W), src=o)

    def _programs(spl: int = 1):
        """(programs, this program) of the launch grid. On trn2 at LNC=2 (grid 2) the kernel is traced once
        per program, the two physical cores of the logical core (nki/_backends/mlir_tracer program_id:
        "kernel is traced LNC times with different program_id_value"), so both are Python ints; the rows
        are independent, so each program selects its own row tiles and writes only their output rows.
        Grid 1 (trn1): (1, 0), every row as before."""
        return (nl.num_programs(axes=0), nl.program_id(axis=0)) if spl and nl.program_ndim() != 0 else (1, 0)

    @nki.jit
    def kiln_dsa_topk_kernel(sc, keep: int, g: int, lg: int, nbits: int, vis_only: int, kp: int, tail: int,
                             tiles: int, rev: int, spl: int = 1):
        """See the module docstring. sc float32 [N, W] = scores [N / g, g W] as pieces; keep < g W;
        g = 2^lg; nbits with 2^nbits > g W; kp tokens per score in the output, tail (needs vis_only)
        also selects the tokens of non-candidate scores; rev: this module's kernel source revision
        (REV, see _kernel_rev). Returns float32 [N, W kp]: 0 selected, NEG_INF not."""
        N, W = sc.shape
        out = nl.ndarray((N, W * kp), dtype=F32, buffer=nl.shared_hbm)
        wio, po, G, bits = _consts(W, g, lg)
        NT = (N + 127) // 128
        npg, pid = _programs(spl)
        for t0 in range(0, NT, tiles):
            if (t0 // tiles) % npg != pid:  # LNC: groups of row tiles alternate between the programs
                continue
            R0 = []
            NS = []
            for t in range(t0, min(t0 + tiles, NT)):
                R0.append(t * 128)
                NS.append(min(128, N - t * 128))
            S = []
            for i in range(len(NS)):
                s = _sb((NS[i], W))
                nisa.dma_copy(dst=s, src=sc.ap(pattern=[[W, NS[i]], [1, W]], offset=R0[i] * W))
                S.append(s)
            _select(S, R0, NS, W, g, G, po, wio, bits, keep, nbits, vis_only, kp, tail, out)
        if npg > 1:  # LNC: every program's rows written before either ends
            nisa.core_barrier(data=out, cores=(0, 1))
        return out

    @nki.jit
    def kiln_dsa_score_topk_kernel(qT, w, pkT, cand, keep: int, scale: float, nbits: int, kp: int, tail: int,
                                   tiles: int, rev: int, spl: int = 1):
        """The pooled indexer's scores and their selection for C queries of ONE sequence (a prefill
        chunk): index[q, p] = sum_h w[q, h] relu(scale (q_h . pk_p)) + cand[q, p], then the
        selection of kiln_dsa_topk_kernel (one piece per row, vis_only). qT bf16 [Hi, D, C] (the
        indexer queries, head-major and transposed), w fp32 [C, Hi], pkT bf16 [D, P] (the pool keys,
        transposed), cand fp32 [C, P] (0 for a candidate pool, NEG_INF not). Each head's scores are a
        bf16 matmul with fp32 accumulation into PSUM (bf16 x bf16 products are exact in fp32), its
        relu and scale on the scalar engine, the weighted head sum in fp32 on the vector engine,
        head 0 first; nothing of the [C, Hi, P] score tensor leaves the chip. Returns fp32
        [C, P kp]."""
        Hi, D, C = qT.shape
        P = pkT.shape[1]
        out = nl.ndarray((C, P * kp), dtype=F32, buffer=nl.shared_hbm)
        wio, po, G, bits = _consts(P, 1, 0)
        kt = _sb((D, P), BF16)
        nisa.dma_copy(dst=kt, src=pkT)
        NT = (C + 127) // 128
        npg, pid = _programs(spl)
        for t0 in range(0, NT, tiles):
            if (t0 // tiles) % npg != pid:  # LNC: groups of row tiles alternate between the programs
                continue
            R0 = []
            NS = []
            for t in range(t0, min(t0 + tiles, NT)):
                R0.append(t * 128)
                NS.append(min(128, C - t * 128))
            S = []
            for i in range(len(NS)):
                r0 = R0[i]
                n = NS[i]
                qs = _sb((D, Hi, n), BF16)
                nisa.dma_copy(dst=qs, src=qT.ap(pattern=[[C, D], [D * C, Hi], [1, n]], offset=r0))
                ws = _sb((n, Hi))
                nisa.dma_copy(dst=ws, src=w.ap(pattern=[[Hi, n], [1, Hi]], offset=r0 * Hi))
                acc = _sb((n, P))
                nisa.dma_copy(dst=acc, src=cand.ap(pattern=[[P, n], [1, P]], offset=r0 * P))
                for h in range(Hi):
                    for c0 in range(0, P, 512):
                        cw = min(512, P - c0)
                        ps = nl.ndarray((n, 512), dtype=F32, buffer=nl.psum)
                        nisa.nc_matmul(dst=ps[:, 0:cw], stationary=qs[:, h, :], moving=kt[:, c0:c0 + cw])
                        r = _sb((n, 512))
                        nisa.activation(dst=r[:, 0:cw], op=nl.relu, data=ps[:, 0:cw], scale=scale)
                        nisa.scalar_tensor_tensor(dst=acc[:, c0:c0 + cw], data=r[:, 0:cw], op0=nl.multiply,
                                                  operand0=ws[:, h:h + 1], op1=nl.add, operand1=acc[:, c0:c0 + cw])
                S.append(acc)
            _select(S, R0, NS, P, 1, G, po, wio, bits, keep, nbits, 1, kp, tail, out)
        if npg > 1:  # LNC: every program's rows written before either ends
            nisa.core_barrier(data=out, cores=(0, 1))
        return out
else:
    kiln_dsa_topk_kernel = kiln_dsa_score_topk_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source (from the NKI import to the end of the kernel), passed
    as the static argument `rev`: LNL's graph cache key hashes each NKI call's name, operands, grid,
    static arguments and MAC count, not the kernel source (libtorch_neuronx_lite/compile/cache.py
    create_cache_hash, SDK 2.32; CLAUDE.md), so without it a kernel edit would silently run the old
    NEFF from a warm cache."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_dsa_topk_kernel = kiln_dsa_score_topk_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev()  # at import: the traced caller reads a constant


def kernel():
    if kiln_dsa_topk_kernel is None:
        raise RuntimeError("the NKI DSA top-k kernel needs the nki package (the Neuron venv)")
    return kiln_dsa_topk_kernel


def score_kernel():
    if kiln_dsa_score_topk_kernel is None:
        raise RuntimeError("the NKI DSA score + top-k kernel needs the nki package (the Neuron venv)")
    return kiln_dsa_score_topk_kernel


def emulate_scores(q: torch.Tensor, w: torch.Tensor, pk: torch.Tensor, cand: torch.Tensor, scale: float) -> torch.Tensor:
    """kiln_dsa_score_topk_kernel's index scores in torch: q [C, Hi, D], w [C, Hi] fp32, pk [P, D],
    cand [C, P] -> [C, P] fp32, cand + sum_h w_h relu(scale (q_h . pk)) accumulated head 0 first
    (the device's PSUM accumulation order over D is its own, so this equals the kernel up to the
    last bits of each dot product)."""
    s = torch.einsum("chd,pd->chp", q.float(), pk.float())
    r = torch.relu(s * scale)
    acc = cand.float().clone()
    for h in range(q.shape[1]):
        acc = r[:, h] * w[:, h : h + 1].float() + acc
    return acc


def score_supported(Q: int, Hi: int, D: int, P: int) -> bool:
    return D == 128 and P <= MAX_W and Hi * min(Q, 128) * 2 <= 64 * 1024


def score_select(q: torch.Tensor, w: torch.Tensor, pk: torch.Tensor, cand: torch.Tensor, keep: int, scale: float,
                 kp: int = 1, tail: bool = False) -> torch.Tensor:
    """One sequence's pooled-indexer scores and their selection in one kernel (a prefill chunk):
    q [C, Hi, D] (bf16 values), w [C, Hi] fp32, pk [P, D] (bf16 values), cand [C, P] 0 / NEG_INF ->
    additive fp32 [C, P kp] (select(..., vis_only=True, kp, tail) of the scores). On the host:
    emulate_scores, then emulate."""
    C, Hi, D = q.shape
    P = pk.shape[0]
    if keep >= P or q.device.type == "cpu":
        return emulate(emulate_scores(q, w, pk, cand, scale), keep, True, kp, tail)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    qT = q.to(torch.bfloat16).permute(1, 2, 0).contiguous()
    pkT = pk.to(torch.bfloat16).t().contiguous()
    return wrap_nki(score_kernel())[platform.nki_grid()](
        qT=qT, w=w.float().contiguous(), pkT=pkT, cand=cand.float().contiguous(), keep=int(keep), scale=float(scale),
        nbits=P.bit_length(), kp=int(kp), tail=int(tail), tiles=TILES, rev=REV, spl=int(_lnc_split()))


def _lnc_split() -> bool:
    from .. import platform

    return platform.lnc_split("dsa_topk")


def kernel_inputs(sc: torch.Tensor, keep: int, vis_only: bool = True, group: bool = True, kp: int = 1,
                  tail: bool = False) -> dict:
    """The kernel's arguments for scores [R, P] (keep < P)."""
    R, P = sc.shape
    g = pieces(R, P, group)
    if P // g > MAX_W:
        raise NotImplementedError(f"DSA top-k kernel: {P // g} scores per partition (more than {MAX_W})")
    if tail and not vis_only:
        raise ValueError("tail needs vis_only")
    return dict(sc=sc.float().reshape(R * g, P // g).contiguous(), keep=int(keep), g=g, lg=g.bit_length() - 1,
                nbits=P.bit_length(), vis_only=int(vis_only), kp=int(kp), tail=int(tail), tiles=TILES, rev=REV,
                spl=int(_lnc_split()))


def supported(rows: int, n: int) -> bool:
    return n // pieces(rows, n) <= MAX_W


def select(scores: torch.Tensor, keep: int, vis_only: bool = True, group: bool = True, kp: int = 1,
           tail: bool = False) -> torch.Tensor:
    """Additive float32 [..., P kp] (0 = selected, NEG_INF = not) of the module's selection over the
    last axis (kp > 1: per token of each score's pool; tail: plus the non-candidate pools' tokens),
    inside the caller's graph: the NKI kernel on a Neuron device, emulate() elsewhere (group=False:
    one piece per row, probes only)."""
    *lead, P = scores.shape
    R = 1
    for d in lead:
        R *= d
    s = scores.float().reshape(R, P)
    if keep >= P or scores.device.type == "cpu":
        return emulate(s, keep, vis_only, kp, tail).reshape(*lead, P * kp)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    out = wrap_nki(kernel())[platform.nki_grid()](**kernel_inputs(s, keep, vis_only, group, kp, tail))
    return out.reshape(*lead, P * kp)
