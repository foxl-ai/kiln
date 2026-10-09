"""Exact pooled-indexer selection over a long context as one NKI kernel for NeuronCore-v2 (trn1): GLM-5.3-Flash's
DSA layers at up to 262,144 pools (1,048,576 tokens), for the long-context path (models/dsa_long.py).

What it computes, for N queries that share ONE context's pool keys (a prefill chunk's queries):
  index[q, p] = cand[q, p] + sum_h w[q, h] relu(scale (qI[q, h] . pk[p])) in fp32, the even heads and the odd heads
  each summed in head order into an accumulator of their own, then added (emulate_scores),
  cand = +0 for a candidate pool p < npool[q] and NEG_INF otherwise;
  the keep largest candidates of each query, ties to the LOWEST pool index, all of them when fewer
  (dsa_long.select_reference), returned as pool indices [N, keep] in ASCENDING order with 0 past the count
  min(npool, keep). emulate() is the same arithmetic in torch.

How (the two-level algorithm of models/dsa_long.py, whose proof covers ties):
1. Scores. Queries on the partitions, 128 per tile; the pools in chunks of 512: the chunk's pool keys by DMA as
   4 x [128 pools, 128 d], transposed on the tensor engine to pk^T [128 d, 512] (exact: a bf16 matmul against the
   identity accumulates in fp32), then per head a bf16 matmul (stationary qI^T of the head [128 d, 128 q], moving
   pk^T) into PSUM, relu x scale in place on the scalar engine, and acc = r w_h + acc on the vector engine with
   one PSUM operand (full rate: docs/neuron-notes.md "Engine rates"). The chunk's scores go to an HBM scratch
   [128 q, P] and its sub-block maxima (sub pools each) into bm [128, P / sub] in SBUF.
2. Level 1 (only when P / sub > keep): the top-keep sub-blocks by maximum by 64 rounds of max8 (the 8 largest
   values of each partition, duplicates included), nc_find_index8 (the first position of each; duplicate values
   pair with ascending positions) and nc_match_replace8 (those positions replaced by -3e38), which extracts in
   (value descending, index ascending) order: exactly the tie rule (measured on trn1, tools/probe_lc_prims.py;
   nc_match_replace8's dst_idx form does not compile on trn1: "Unimplemented instruction ... MaxIndexAndMatchReplace").
   The selected sub-block indices are then sorted ascending (64 rounds of max8 / match_replace8 over -index).
3. Level 2: each query's selected sub-blocks' scores gathered back from the scratch by indirect DMA (one row of sub
   fp32 per partition per instruction), in ascending sub-block order, so candidate position order is pool order;
   the top-keep of those keep x sub candidates by the same extraction; candidate positions mapped to pool indices
   with nc_n_gather (GpSimd, within the partition). With P / sub <= keep the whole score row is the candidate list.
4. Out: the pool indices sorted ascending, invalid ones (score at or below VISIBLE: fewer candidates than keep) last
   and written as 0.
The query tiles run in a device loop (nl.fori_loop): every tile reuses the same SBUF tiles and the scratch.

Engine time per (query, pool) at the bound (vector engine: one weighted add per head, 32 heads, ~505 ns per
[128, 512]): ~0.25 ns, plus the selection (~4.4 ms per 128 queries at P = 262,144, sub = 32: 64 extraction rounds
over 8,192 maxima and over 16,384 candidates, tools/probe_lc_prims.py rates).
"""

from __future__ import annotations

import os

import torch

NEG_INF = -1e30  # models/decoder.NEG_INF
VISIBLE = -5e29  # kernels/dsa_topk.VISIBLE
REPL = -3.0e38  # an extracted element's replacement (below NEG_INF)
SUB = int(os.environ.get("KILN_DSA_LONG_SUB", 32))  # pools per sub-block (models/dsa_long.SUB)
CH = 512  # pools per score chunk (one PSUM bank of fp32)


def emulate(qI: torch.Tensor, w: torch.Tensor, pk: torch.Tensor, npool: torch.Tensor, keep: int, scale: float,
            vorder: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The kernel's result in torch: qI [N, Hi, D], w [N, Hi] fp32, pk [P, D], npool [N] -> (pools [N, keep] int64
    ascending, 0 past the count; count [N] int64; each selected pool's score [N, keep] fp32, NEG_INF past the
    count). vorder: the same pools in (score descending, pool ascending) order instead (value_order)."""
    from ..models import dsa_long

    N = qI.shape[0]
    P = pk.shape[0]
    npool = npool.long().clamp(0, P)
    cand = torch.where(torch.arange(P).view(1, P) < npool.view(N, 1), 0.0, NEG_INF)
    sc = emulate_scores(qI, w, pk, cand, scale)
    pools, cnt = dsa_long.select_reference(sc, keep)
    k = torch.arange(keep).view(1, keep)
    vals = torch.where(k < cnt.view(N, 1), sc.gather(1, pools), NEG_INF)
    if vorder:
        return value_order(pools, cnt, vals)
    return pools, cnt, vals


def value_order(pools: torch.Tensor, cnt: torch.Tensor, vals: torch.Tensor):
    """A selection (pools ascending, count, scores; 0 / NEG_INF past the count) reordered to the kernel's vorder form:
    the selected pools by score descending, equal scores by pool ascending, then 0 / NEG_INF."""
    N, keep = pools.shape
    k = torch.arange(keep).view(1, keep)
    live = k < cnt.view(N, 1)
    # a stable sort by score descending keeps equal scores in pool (ascending) order; the dead entries sort last
    key = torch.where(live, vals, torch.full_like(vals, -3.0e38))
    order = torch.sort(key, dim=-1, descending=True, stable=True).indices
    return (torch.where(live, pools.gather(1, order), 0), cnt, torch.where(live, vals.gather(1, order), NEG_INF))


def _extract_host(x: torch.Tensor, rounds: int) -> tuple[torch.Tensor, torch.Tensor]:
    """rounds of the kernel's max8 / nc_find_index8 / nc_match_replace8 over the rows of x [N, W] in torch: the
    8 rounds largest in (value descending, position ascending) order, as (values, positions)."""
    # a stable sort by value descending keeps equal values in position order: what the rounds extract
    order = torch.sort(x, dim=-1, descending=True, stable=True).indices[:, :8 * rounds]
    return x.gather(1, order), order


def emulate_algorithm(sc: torch.Tensor, keep: int, sub: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The kernel's selection steps on scores sc [N, P] (non-candidates at or below VISIBLE), step for step:
    sub-block maxima, the top-keep sub-blocks by extraction, their indices sorted ascending, their scores gathered,
    the top-keep candidates by extraction, positions mapped to pools, then pools ascending with the invalid ones
    (score at or below VISIBLE) last as pool 0 / NEG_INF. sub 0: one level over the whole row. Returns (pools
    [N, keep] int64, scores [N, keep] fp32); equal to (select_reference's pools, the scores there) by the proof in
    models/dsa_long.py."""
    N, P = sc.shape
    rounds = keep // 8
    Pp = -(-P // CH) * CH
    s = torch.cat([sc, sc.new_full((N, Pp - P), NEG_INF)], dim=1) if Pp != P else sc
    if sub:
        NB = Pp // sub
        bm = s.view(N, NB, sub).amax(-1)
        _, bi = _extract_host(bm, rounds)
        blk = torch.sort(bi, dim=-1).values  # ascending
        cand = s.view(N, NB, sub).gather(1, blk.unsqueeze(-1).expand(N, keep, sub)).reshape(N, keep * sub)
        v2, ci = _extract_host(cand, rounds)
        pool = blk.gather(1, ci // sub) * sub + ci % sub
    else:
        v2, pool = _extract_host(s, rounds)
    valid = v2 > VISIBLE
    key = torch.where(valid, -pool.float(), -float(1 << 23))
    kv, ki = _extract_host(key, rounds)
    srt = -kv
    ok = srt < float(1 << 23)
    return torch.where(ok, srt, 0.0).to(torch.int64), torch.where(ok, v2.gather(1, ki), NEG_INF)


def emulate_scores(qI: torch.Tensor, w: torch.Tensor, pk: torch.Tensor, cand: torch.Tensor, scale: float) -> torch.Tensor:
    """The kernel's index scores [N, P] fp32: r_h = relu(scale (qI_h . pk)) (fp32 dot products, whose summation order
    on the device's tensor engine is its own, so these equal the kernel up to the last bits of each dot product), then
    two accumulators, A0 = cand + r_0 w_0 + r_2 w_2 + ... and A1 = 0 + r_1 w_1 + r_3 w_3 + ..., each in head order, and
    the score A0 + A1 (kernels/dsa_topk.emulate_scores' sum with the heads split over two chains)."""
    s = torch.einsum("nhd,pd->nhp", qI.float(), pk.float())
    r = torch.relu(s * scale)
    a0 = cand.float().clone()
    a1 = torch.zeros_like(a0)
    for h in range(qI.shape[1]):
        if h % 2:
            a1 = r[:, h] * w[:, h:h + 1].float() + a1
        else:
            a0 = r[:, h] * w[:, h:h + 1].float() + a0
    return a0 + a1


def supported(Hi: int, D: int, P: int, keep: int, sub: int) -> bool:
    """Shapes the kernel takes: 128-dim keys, up to 32 heads, keep a multiple of 8, the level-2 candidates and the
    sub-block maxima within max8's 16,384 elements per partition, sub a power of two from 2 to 32 (0: one level)."""
    Pp = -(-P // CH) * CH
    base = D == 128 and Hi <= 32 and keep % 8 == 0 and 8 <= keep <= Pp
    if sub == 0:  # one level: the whole score row is the candidate list
        return base and Pp <= 16384
    nb = Pp // sub
    return base and sub & (sub - 1) == 0 and 2 <= sub <= 32 and keep < nb <= 16384 and keep * sub <= 16384


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    F32, BF16, I32, U32 = nl.float32, nl.bfloat16, nl.int32, nl.uint32
    VE = nisa.vector_engine

    def _sb(shape, dt=None):
        return nl.ndarray(shape, dtype=dt or F32, buffer=nl.sbuf)

    def _extract(V, I, cur, rounds: int):
        """rounds of the 8 largest of cur [128, W] (values into V [128, 8 rounds], positions into I, uint32), each
        round's elements replaced in cur: (value descending, position ascending) order."""
        for r in range(rounds):
            v = V[:, 8 * r:8 * r + 8]
            nisa.max8(dst=v, src=cur)
            if I is not None:
                nisa.nc_find_index8(dst=I[:, 8 * r:8 * r + 8], data=cur, vals=v)
            if r + 1 < rounds:
                nisa.nc_match_replace8(dst=cur, data=cur, vals=v, imm=REPL)

    def _at(X, tensor, pattern, t, stride: int):
        """tensor's access pattern at element offset t stride: t a Python int (unrolled tiles) or the device loop's
        register (X["dyn"]: a dynamic DMA address along the tensor's first dimension, whose stride is `stride`)."""
        if X["dyn"]:
            return tensor.ap(pattern=pattern, offset=0, scalar_offset=t, indirect_dim=0)
        return tensor.ap(pattern=pattern, offset=t * stride)

    def _const(v: float):
        t = _sb((128, 1))
        nisa.memset(dst=t, value=v)
        return t

    def _psum():
        TP = []
        for _ in range(2):
            TP.append(nl.ndarray((128, CH), dtype=F32, buffer=nl.psum))
        SP = []
        for _ in range(6):
            SP.append(nl.ndarray((128, CH), dtype=F32, buffer=nl.psum))
        return TP, SP

    def _sort_asc(dst, key, V, rounds: int):
        """dst [128, 8 rounds] fp32: the values -key in ascending order (key [128, 8 rounds] is destroyed)."""
        _extract(V, None, key, rounds)
        nisa.tensor_scalar(dst=dst, data=V, op0=nl.multiply, operand0=-1.0, engine=VE)

    def _prologue(X, TP, thr, c):
        """Chunk c's pool keys transposed to K^T, and its score accumulator holding the candidates: +0 where
        CH c + j < npool, NEG_INF elsewhere (0/1 on the vector engine, then the scalar engine's 1e30 x - 1e30:
        an exact +0, as emulate_scores' cand). Issued one chunk ahead of the chunk's heads, so no engine waits
        on another at a chunk boundary."""
        x = c % 2
        D = X["D"]
        pkr = X["PKr"][x]
        nisa.dma_copy(dst=pkr, src=X["pk"].ap(pattern=[[D, 128], [128 * D, 4], [1, D]], offset=c * CH * D))
        tp = TP[x]
        for i in range(4):
            nisa.nc_matmul(dst=tp[:, i * 128:(i + 1) * 128], stationary=pkr[:, i, :], moving=X["IB"],
                           accumulate=False)
        nisa.activation(dst=X["KT"][x], op=nl.copy, data=tp)  # (copy takes only an immediate bias: NCC_IBVF043)
        acc = X["ACC"][x]
        nisa.tensor_scalar(dst=acc, data=X["iota"], op0=nl.less, operand0=thr[:, c:c + 1], engine=VE)
        nisa.activation(dst=acc, op=nl.copy, data=acc, scale=-NEG_INF, bias=NEG_INF)
        nisa.memset(dst=X["ACC1"][x], value=0.0)

    def _tile(X, t):
        """Query tile t (rows 128 t .. 128 t + 127): scores, level 1, level 2, out."""
        Hi, D, Pp, NC, NB, keep, sub = X["Hi"], X["D"], X["Pp"], X["NC"], X["NB"], X["keep"], X["sub"]
        lsub = X["lsub"]
        qs, ws, npf, scr = X["qs"], X["ws"], X["npf"], X["scr"]
        nisa.dma_copy(dst=qs, src=_at(X, X["qT"], [[Hi * 128, D], [128, Hi], [1, 128]], t, D * Hi * 128))
        nisa.dma_copy(dst=ws, src=_at(X, X["w"], [[Hi, 128], [1, Hi]], t, 128 * Hi))
        nisa.dma_copy(dst=npf, src=_at(X, X["npool"], [[1, 128], [1, 1]], t, 128))
        if X["psum"] is None:  # PSUM: two transpose banks, four score banks (each a full 2 KB bank), per loop region
            TP, SP = _psum()
        else:
            TP, SP = X["psum"]
        bm = X["bm"]
        thr = X["thr"]  # [128, NC]: npool - CH c, chunk c's candidate bound in its own positions
        nisa.tensor_scalar(dst=thr, data=X["ciota"], op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=npf,
                           engine=VE)

        _prologue(X, TP, thr, 0)
        for c in range(NC):
            x = c % 2
            if c + 1 < NC:
                _prologue(X, TP, thr, c + 1)
            kt = X["KT"][x]
            acc = X["ACC"][x]
            acc1 = X["ACC1"][x]
            for h in range(Hi):
                sp = SP[(c * Hi + h) % len(SP)]
                nisa.nc_matmul(dst=sp, stationary=qs[:, h, :], moving=kt, accumulate=False)
                nisa.activation(dst=sp, op=nl.relu, data=sp, scale=X["sclt"], bias=X["zero"])
                # even heads into acc, odd ones into acc1 (two accumulators: each add no longer waits on the
                # previous one; 1324 -> 1121 ns per head and 512 pools, tools/probe_lc_score.py variants 0 / 11)
                a = acc1 if h % 2 else acc
                nisa.scalar_tensor_tensor(dst=a, data=sp, op0=nl.multiply, operand0=ws[:, h:h + 1], op1=nl.add,
                                          operand1=a)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=acc1, op=nl.add, engine=VE)
            nisa.tensor_reduce(dst=bm[:, c * (CH // sub):(c + 1) * (CH // sub)], op=nl.maximum,
                               data=acc.reshape((128, CH // sub, sub)), axis=2)
            nisa.dma_copy(dst=scr.ap(pattern=[[Pp, 128], [1, CH]], offset=c * CH), src=acc)
        cand = X["cand"]
        V = X["V"]
        I = X["I"]
        rounds = keep // 8
        if X["two"]:
            # level 1: the top-keep sub-blocks by maximum, then their indices ascending
            _extract(V, I, bm, rounds)
            key = X["key"]
            nisa.tensor_copy(dst=key, src=I.view(I32), engine=VE)  # uint32 -> fp32, exact (< 2^24)
            nisa.tensor_scalar(dst=key, data=key, op0=nl.multiply, operand0=-1.0, engine=VE)
            blk = X["blk"]
            _sort_asc(blk, key, V, rounds)
            # each selected sub-block's row of the scratch viewed [128 NB, sub]: q NB + block
            ro = X["roff"]
            nisa.tensor_scalar(dst=X["rowf"], data=blk, op0=nl.add, operand0=X["qnb"], engine=VE)
            nisa.tensor_copy(dst=ro, src=X["rowf"], engine=VE)
            for j in range(keep):
                nisa.dma_copy(dst=cand[:, j * sub:(j + 1) * sub],
                              src=X["scr2"].ap(pattern=[[sub, 128], [1, sub]], offset=0,
                                               vector_offset=ro.ap(pattern=[[keep, 128], [1, 1]], offset=j),
                                               indirect_dim=0))
            W2 = keep * sub
        else:
            nisa.dma_copy(dst=cand[:, 0:Pp], src=scr.ap(pattern=[[Pp, 128], [1, Pp]], offset=0))
            W2 = Pp
        # level 2: the top-keep candidates, positions -> pool indices
        V2 = X["V2"]
        _extract(V2, I, cand[:, 0:W2], rounds)
        pool = X["pool"]
        if X["two"]:
            hi = X["hi"]
            nisa.tensor_scalar(dst=hi, data=I.view(I32), op0=nl.right_shift, operand0=lsub, engine=VE)
            lo = X["lo"]
            nisa.tensor_scalar(dst=lo, data=I.view(I32), op0=nl.bitwise_and, operand0=sub - 1, engine=VE)
            g = X["g"]
            nisa.nc_n_gather(dst=g, data=blk, indices=hi.view(U32))
            lof = X["lof"]
            nisa.tensor_copy(dst=lof, src=lo, engine=VE)
            nisa.scalar_tensor_tensor(dst=pool, data=g, op0=nl.multiply, operand0=float(sub), op1=nl.add, operand1=lof)
        else:
            nisa.tensor_copy(dst=pool, src=I.view(I32), engine=VE)
        # out: valid (score above VISIBLE) pools ascending with their scores, the rest last as pool 0 / NEG_INF
        vm = X["vm"]
        nisa.tensor_scalar(dst=vm, data=V2, op0=nl.greater, operand0=VISIBLE, engine=VE)
        if X["vorder"]:  # the extraction's own order (value descending, pool ascending), no sort by pool
            srt = X["srt"]
            nisa.tensor_tensor(dst=srt, data1=pool, data2=vm, op=nl.multiply, engine=VE)
            vals = X["vals"]
            nisa.tensor_tensor(dst=vals, data1=V2, data2=vm, op=nl.multiply, engine=VE)
            nisa.tensor_scalar(dst=vm, data=vm, op0=nl.multiply, operand0=-NEG_INF, op1=nl.add, operand1=NEG_INF,
                               engine=VE)
            nisa.tensor_tensor(dst=vals, data1=vals, data2=vm, op=nl.add, engine=VE)
            oi = X["oi"]
            nisa.tensor_copy(dst=oi, src=srt, engine=VE)
            nisa.dma_copy(dst=_at(X, X["out"], [[keep, 128], [1, keep]], t, 128 * keep), src=oi)
            nisa.dma_copy(dst=_at(X, X["outv"], [[keep, 128], [1, keep]], t, 128 * keep), src=vals)
            return
        key = X["key"]
        # key = valid ? -pool : -2^23  ==  vm (2^23 - pool) - 2^23 (exact in fp32: pools < 2^23)
        nisa.tensor_scalar(dst=key, data=pool, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=float(1 << 23),
                           engine=VE)
        nisa.tensor_tensor(dst=key, data1=key, data2=vm, op=nl.multiply, engine=VE)
        nisa.tensor_scalar(dst=key, data=key, op0=nl.add, operand0=-float(1 << 23), engine=VE)
        _extract(V, I, key, rounds)  # V: the keys descending = pools ascending; I: each one's place in V2
        srt = X["srt"]
        nisa.tensor_scalar(dst=srt, data=V, op0=nl.multiply, operand0=-1.0, engine=VE)
        vals = X["vals"]
        nisa.nc_n_gather(dst=vals, data=V2, indices=I)
        # invalid ones sorted last hold 2^23 now: pool x -> x if x < 2^23 else 0, value -> NEG_INF there
        nisa.tensor_scalar(dst=vm, data=srt, op0=nl.less, operand0=float(1 << 23), engine=VE)
        nisa.tensor_tensor(dst=srt, data1=srt, data2=vm, op=nl.multiply, engine=VE)
        nisa.tensor_tensor(dst=vals, data1=vals, data2=vm, op=nl.multiply, engine=VE)
        nisa.tensor_scalar(dst=vm, data=vm, op0=nl.multiply, operand0=-NEG_INF, op1=nl.add, operand1=NEG_INF,
                           engine=VE)
        nisa.tensor_tensor(dst=vals, data1=vals, data2=vm, op=nl.add, engine=VE)
        oi = X["oi"]
        nisa.tensor_copy(dst=oi, src=srt, engine=VE)
        nisa.dma_copy(dst=_at(X, X["out"], [[keep, 128], [1, keep]], t, 128 * keep), src=oi)
        nisa.dma_copy(dst=_at(X, X["outv"], [[keep, 128], [1, keep]], t, 128 * keep), src=vals)
        if X["dbg"]:  # the scores of the last tile run (probes)
            nisa.dma_copy(dst=X["dscr"], src=scr.ap(pattern=[[Pp, 128], [1, Pp]], offset=0))

    @nki.jit
    def kiln_dsa_long_select_kernel(qT, w, npool, pk, identb, keep: int, scale: float, sub: int, lsub: int, one: int,
                                    rev: int, loop: int = 1, dbg: int = 0, vorder: int = 0):
        """qT bf16 [NT, D, Hi, 128] (tile t's queries transposed, head-major), w fp32 [NT, 128, Hi], npool fp32
        [NT, 128, 1], pk bf16 [Pp, D] (Pp a multiple of 512), identb bf16 [128, 128]; keep a multiple of 8; sub a
        power of two; rev: this module's kernel source revision; loop: the query tiles in a device loop (1) or
        unrolled (0). Returns pools int32 [NT, 128, keep] (module docstring)."""
        NT, D, Hi, Q = qT.shape
        Pp = pk.shape[0]
        NC = Pp // CH
        NB = Pp // sub
        out = nl.ndarray((NT, 128, keep), dtype=I32, buffer=nl.shared_hbm)
        outv = nl.ndarray((NT, 128, keep), dtype=F32, buffer=nl.shared_hbm)
        scr = nl.ndarray((128 * NB, sub), dtype=F32, buffer=nl.shared_hbm)  # the scores, q-major
        X = dict(Hi=Hi, D=D, Pp=Pp, NC=NC, NB=NB, two=1 - one, keep=keep, sub=sub, scale=scale, qT=qT, w=w, npool=npool, pk=pk,
                 out=out, outv=outv, scr=scr, scr2=scr, dbg=dbg, psum=None, lsub=lsub, dyn=0, vorder=vorder,
                 dscr=nl.ndarray((128, Pp), dtype=F32, buffer=nl.shared_hbm) if dbg else None)
        npg = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1  # lnc2
        if npg == 2 and nl.program_id(axis=0) == 1:  # lnc2: program 0 alone selects (see _lnc2_note)
            nisa.core_barrier(data=out, cores=(0, 1))  # lnc2
            return (out, outv, X["dscr"]) if dbg else (out, outv)  # lnc2
        IB = _sb((128, 128), BF16)
        nisa.dma_copy(dst=IB, src=identb)
        X["IB"] = IB
        io_i = _sb((128, CH), I32)
        nisa.iota(dst=io_i, pattern=[[1, CH]], offset=0, channel_multiplier=0)
        iota = _sb((128, CH))
        nisa.tensor_copy(dst=iota, src=io_i, engine=VE)
        X["iota"] = iota
        qi = _sb((128, 1), I32)
        nisa.iota(dst=qi, pattern=[[0, 1]], offset=0, channel_multiplier=NB)
        qnb = _sb((128, 1))
        nisa.tensor_copy(dst=qnb, src=qi, engine=VE)
        X["qnb"] = qnb
        X["qs"] = _sb((D, Hi, 128), BF16)
        X["ws"] = _sb((128, Hi))
        X["npf"] = _sb((128, 1))
        X["thr"] = _sb((128, NC))
        ci_i = _sb((128, NC), I32)
        nisa.iota(dst=ci_i, pattern=[[CH, NC]], offset=0, channel_multiplier=0)
        X["ciota"] = _sb((128, NC))
        nisa.tensor_copy(dst=X["ciota"], src=ci_i, engine=VE)
        # the scalar engine's scale and bias operands as tiles (immediates cost two vector-engine memsets per
        # activation: measured, 1.7 ms of a 23.5 ms call at 262,144 pools)
        X["sclt"] = _const(scale)
        X["zero"] = _const(0.0)
        X["one"] = _const(1.0)
        X["big"] = _const(-NEG_INF)
        X["negb"] = _const(NEG_INF)
        X["PKr"] = [_sb((128, 4, D), BF16), _sb((128, 4, D), BF16)]
        X["KT"] = [_sb((128, CH), BF16), _sb((128, CH), BF16)]
        X["ACC"] = [_sb((128, CH)), _sb((128, CH))]
        X["ACC1"] = [_sb((128, CH)), _sb((128, CH))]
        X["bm"] = _sb((128, NB))
        W2 = keep * sub if not one else Pp
        X["cand"] = _sb((128, W2))
        X["V"] = _sb((128, keep))
        X["V2"] = _sb((128, keep))
        X["I"] = _sb((128, keep), U32)
        X["key"] = _sb((128, keep))
        X["blk"] = _sb((128, keep))
        X["rowf"] = _sb((128, keep))
        X["roff"] = _sb((128, keep), I32)
        X["hi"] = _sb((128, keep), I32)
        X["lo"] = _sb((128, keep), I32)
        X["lof"] = _sb((128, keep))
        X["g"] = _sb((128, keep))
        X["pool"] = _sb((128, keep))
        X["vm"] = _sb((128, keep))
        X["srt"] = _sb((128, keep))
        X["vals"] = _sb((128, keep))
        X["oi"] = _sb((128, keep), I32)
        if loop and NT > 1:
            cnt = _sb((1, 1), I32)
            nisa.iota(dst=cnt, pattern=[[0, 1]], offset=NT, channel_multiplier=0)
            rt = nisa.register_alloc()
            nisa.register_load(rt, cnt)

            X["dyn"] = 1

            def body(t):
                _tile(X, t)

            nl.fori_loop(0, rt, body)
        else:
            X["psum"] = _psum()
            for t in range(NT):
                _tile(X, t)
        if npg == 2:  # lnc2
            nisa.core_barrier(data=out, cores=(0, 1))  # lnc2
        if dbg:
            return out, outv, X["dscr"]
        return out, outv
else:
    kiln_dsa_long_select_kernel = None


def _kernel_rev(lnc2: bool = False) -> int:
    """CRC-32 of this module's kernel source, passed as the static argument `rev` (LNL's compile-cache key does
    not include NKI kernel source: CLAUDE.md). Lines marked "# lnc2" act only at grid 2 (trn2, LNC=2) and are left
    out of the grid-1 revision, so trn1's graphs keep their keys."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_dsa_long_select_kernel = None", a)
    body = src[a:b]
    if not lnc2:
        body = "".join(x for x in body.splitlines(keepends=True) if "# lnc2" not in x)
    return zlib.crc32(body.encode())


# _lnc2_note: at grid 2 (trn2, LNC=2) the kernel used to run whole on both programs, each writing the same out, outv
# and HBM score scratch, one tensor shared by the two physical cores. Program 0 alone selects now; program 1 waits at a
# core barrier on out, which program 0 reaches after its last write, so out is complete on both cores when either
# returns. (Measured on trn2: the R8 long-context engine's 4096-page graphs answered the needle wrong at P=3 and faulted
# with a vector-DGE out-of-bound copy at P=6 / P=12; docs/neuron-notes.md "Long prompts on trn2".)
REV = _kernel_rev()
REV_LNC2 = _kernel_rev(lnc2=True)
LOOP = int(os.environ.get("KILN_DSA_LONG_LOOP", 1))


def kernel_inputs(qI: torch.Tensor, w: torch.Tensor, pk: torch.Tensor, npool: torch.Tensor) -> dict:
    """The kernel's tensor arguments: queries padded to whole 128-row tiles (padded rows have no candidate),
    pools to whole chunks of 512 (padded pools are past every npool)."""
    N, Hi, D = qI.shape
    P = pk.shape[0]
    NT = -(-N // 128)
    Np, Pp = NT * 128, -(-P // CH) * CH
    q = qI.to(torch.bfloat16)
    wf = w.float()
    npf = npool.float().clamp(0, P)
    if Np != N:
        q = torch.cat([q, q.new_zeros(Np - N, Hi, D)])
        wf = torch.cat([wf, wf.new_zeros(Np - N, Hi)])
        npf = torch.cat([npf, npf.new_zeros(Np - N)])
    pkb = pk.to(torch.bfloat16)
    if Pp != P:
        pkb = torch.cat([pkb, pkb.new_zeros(Pp - P, D)])
    qT = q.reshape(NT, 128, Hi, D).permute(0, 3, 2, 1).contiguous()  # [NT, D, Hi, 128]
    eye = torch.eye(128, device=qI.device).to(torch.bfloat16)
    return dict(qT=qT, w=wf.reshape(NT, 128, Hi).contiguous(), npool=npf.reshape(NT, 128, 1).contiguous(),
                pk=pkb.contiguous(), identb=eye)


def pick_sub(P: int, keep: int) -> int:
    """The sub-block size whose two levels extract over the fewest values (P / sub maxima + keep x sub candidates),
    or 0 for one level over the whole row when that is no wider. Powers of two from 2 to 32, within max8's 16,384
    elements per partition."""
    Pp = -(-P // CH) * CH
    best, width = 0, Pp
    for sub in (2, 4, 8, 16, 32):
        nb = Pp // sub
        if nb <= keep or nb > 16384 or keep * sub > 16384:
            continue
        if nb + keep * sub < width:
            best, width = sub, nb + keep * sub
    if best == 0 and Pp > 16384:
        raise NotImplementedError(f"long-context DSA selection: {P} pools need a sub-block size above 32")
    return best


def select(qI: torch.Tensor, w: torch.Tensor, pk: torch.Tensor, npool: torch.Tensor, keep: int, scale: float,
           sub: int | None = None, vorder: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(pools [N, keep] int64 ascending, 0 past the count; count [N] int64; scores [N, keep] fp32, NEG_INF past the
    count): the kernel on a Neuron device, emulate() on the host. qI [N, Hi, D] (bf16 values), w [N, Hi] fp32, pk
    [P, D] (bf16 values), npool [N] int. sub: the sub-block size (None: pick_sub; 0: one level). vorder: the pools
    in (score descending, pool ascending) order, the extraction's own, without the final sort by pool."""
    N = qI.shape[0]
    P = pk.shape[0]
    if qI.device.type == "cpu":
        return emulate(qI, w, pk, npool, keep, scale, vorder)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if kiln_dsa_long_select_kernel is None:
        raise RuntimeError("the NKI long-context DSA selection kernel needs the nki package (the Neuron venv)")
    sub = pick_sub(P, keep) if sub is None else sub
    if not supported(qI.shape[1], qI.shape[2], P, keep, sub):
        raise NotImplementedError(f"long-context DSA selection kernel: Hi {qI.shape[1]}, D {qI.shape[2]}, P {P}, "
                                  f"keep {keep}, sub {sub}")
    grid = platform.nki_grid()
    out, outv = wrap_nki(kiln_dsa_long_select_kernel)[grid](
        **kernel_inputs(qI, w, pk, npool), keep=int(keep), scale=float(scale), sub=int(sub or CH),
        lsub=int(sub or CH).bit_length() - 1, one=int(sub == 0), rev=REV_LNC2 if grid == 2 else REV,
        # at grid 2 program 1 skips the tiles, so several tiles go unrolled (a device loop on one core only: NCC_IXGM002)
        loop=0 if grid == 2 and N > 128 else LOOP,
        **({"vorder": 1} if vorder else {}))
    cnt = npool.to(torch.int64).clamp(0, P).clamp(max=keep)
    return out.reshape(-1, keep)[:N].to(torch.int64), cnt, outv.reshape(-1, keep)[:N]
