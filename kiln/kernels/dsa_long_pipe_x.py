"""kernels/dsa_long_pipe.py with each tile's score chunks bounded by the tile's largest npool (skip = 1): the same
pools, order and scores, bit for bit, as dsa_long_pipe and as one dsa_long_select call per tile.

Why: a prefill chunk's queries all sit near one position (1024 queries within ~1024 pools of each other), but the
kernel scores every pool of the page bucket, 64 chunks of 512 at 32,768 local pools, ~1.97 ms per tile of 128 queries
(trn1, kiln-lc2-k1, 2026-10-06). A chunk at 30% of its bucket has ~70% of those chunks wholly past every query's
candidates: their pools are all invisible (npool <= the chunk's first pool for every query), so every score there is
exactly NEG_INF (the visibility mask starts the accumulator at NEG_INF and a weighted head sum of bf16 scores moves
-1e30 by nothing: its ulp is 7.6e22) and so is every sub-block maximum.

skip = 1 runs chunk c of tile t only when c 512 < max npool of the tile (its own 0 / 1-trip device loop around the
chunk's static code, the flag a host-made int32 [NT, 1, NC]); the tile's sub-block maxima are set to NEG_INF first, so
a skipped chunk's read exactly as the computed ones would. Its HBM scratch is NOT written (the scratches are reused by
tile parity, so it holds another tile's scores or nothing): each selected sub-block at or past the tile's computed
sub-blocks (nbt [NT, 128, 1], host-made) is gathered from one extra scratch row of NEG_INF instead, the values the
computed chunk would have held. That is the case of fewer than keep sub-blocks with a candidate (npool below keep x
sub), where level 1 selects NEG_INF sub-blocks. One-level shapes (sub 0) take dsa_long_pipe's kernel.

pe = 1 (NOT exact, opt-in KILN_DSA_LONG_PE): the weighted head sum on the tensor engine instead of the vector engine.
Per head the scalar engine writes relu(|w_h| scale s_h) in bf16 to SBUF (the weight's magnitude folded into the relu's
per-query scale: |w| relu(x) = relu(|w| x)), and a second matmul against diag(sign w_h) [128, 128] (bf16 +-1)
accumulates sign(w_h) relu(|w_h| scale s_h) = w_h relu(scale s_h) over the heads in a PSUM bank; the vector engine
adds the visibility mask once per chunk. The deviation is the bf16 rounding of each head's term (and one fp32 chain
instead of two); the vector engine, which bounded the scores (its weighted adds 1.22 of 1.97 ms per tile), keeps only
the selection.

Measured (tools/probe_lc_pipe.py, 1024 queries, 32 heads, trn1 kiln-lc2-k1, 2026-10-07, p50 per call):

  pe, keep 128 (sub 16), 32,768 pools: 10.59 ms (1.32 ms per tile) against dsa_long_pipe's 18.03 and the per-tile
      form's 19.28; 8,192 pools 4.61 against 6.22. keep 512: 23.52 against 31.43 (32,768), 15.90 against 17.63
      (8,192). Against emulate_scores' fp32 scores at the selected pools, |d| / the row's largest |score| <= 6.6e-3
      (mean ~5e-4) on every kind; the selected sets differ from the exact ones by 0.74 of 128 entries per row on
      real text (GLM-5.3-Flash layer 3's indexer, 1M tokens, CP rank 0 of 8: tools/make_real_indexer_inputs.py),
      2.5 of 512 at keep 512, at most 7; one "ties" row in 1022 (four distinct scores) flips its whole set. Issuing
      each head's sum two or three heads after its score matmul changed nothing (15.99 / 23.73 ms), so it is not.
  skip, bit-identical to the per-tile form on every kind (tilemix, chunkNN, pooled, short, ties, zeros), but a
      device loop costs ~30-37 us of engine synchronisation here: one loop per chunk, 256 queries x 8,192 pools,
      5.18 ms against 4.73 (tilemix) and 4.67 against 4.74 (chunk30), and 8 tiles x 64 such loops at 32,768 pools
      had not compiled after 41 minutes; groups of 8 chunks (KILN_DSA_LONG_SKIP_GROUP) 4.73 against 4.75 and
      4.48 against 4.75, and at 1024 x 32,768 (no KTall: each chunk's pool keys a static DMA inside the loop body)
      neuronx-cc 2.27 failed after ~25 minutes with NCC_ISQI001 "Expect only Spill/Reload queues are used in loop
      body, but found qSPIO0 of type input". Not wired: infeasible at the 1M shapes on this compiler.

A separate module so that the sources of kernels/dsa_long_pipe.py and kernels/dsa_long_select.py, their REVs and the
keys of every graph that calls them, are unchanged; its REV is this module's kernel source and dsa_long_pipe.REV.
"""

from __future__ import annotations

import os

import torch

from . import dsa_long_pipe as dp
from . import dsa_long_select as dls

CH = dls.CH
NEG_INF = dls.NEG_INF
# skip: chunks per device loop. A device loop costs ~30-37 us of engine synchronisation in this kernel (2 tiles x 16
# chunks of 512 at 8192 pools, half of them skipped: 6.48 ms against dsa_long_pipe's 6.29, profiled, kiln-lc2-k1,
# 2026-10-07), more than a chunk's scores, and 8 tiles x 64 one-chunk loops at 32,768 pools had not compiled after 41
# minutes, so a loop holds a group of chunks.
GROUP = int(os.environ.get("KILN_DSA_LONG_SKIP_GROUP", 8))
DBG = int(os.environ.get("KILN_DSA_LONG_X_DBG", 0))  # 1: skip's redirect and memset, every chunk scored (bisection)

try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None and dp.kiln_dsa_long_pipe_kernel is not None:
    from .dsa_long_pipe import _dma, _keys, _level1_round, _level2, _prologue2, _round  # by name (the tracer)
    from .dsa_long_select import _at, _const, _psum, _sb

    F32, BF16, I32, U32 = nl.float32, nl.bfloat16, nl.int32, nl.uint32
    VE = nisa.vector_engine

    def _sort_gather_x(X, scr, cand, par: int, dep_from, nbt):
        """dsa_long_pipe._sort_gather (two levels); with skip, the selected sub-blocks at or past nbt [128, 1] (the
        tile's computed ones) gathered from the scratch's NEG_INF row."""
        keep, sub = X["keep"], X["sub"]
        V, I = X["V"], X["I"]
        rounds = keep // 8
        key = X["key"]
        nisa.tensor_copy(dst=key, src=I.view(I32), engine=VE)
        nisa.tensor_scalar(dst=key, data=key, op0=nl.multiply, operand0=-1.0, engine=VE)
        blk = X["blk2"][par]
        for r in range(rounds):
            _round(V, None, key, r, rounds)
        nisa.tensor_scalar(dst=blk, data=V, op0=nl.multiply, operand0=-1.0, engine=VE)
        rf = X["rowf"]
        nisa.tensor_scalar(dst=rf, data=blk, op0=nl.add, operand0=X["qnb"], engine=VE)
        dep = X["dep"]
        nisa.tensor_scalar(dst=dep, data=dep_from, op0=nl.multiply, operand0=0.0, engine=VE)
        nisa.tensor_scalar(dst=rf, data=rf, op0=nl.add, operand0=dep, engine=VE)
        if X["skip"]:
            nr = float(X["NR"])
            mk = X["mk"]
            nisa.tensor_scalar(dst=mk, data=blk, op0=nl.less, operand0=nbt, engine=VE)  # 1: computed
            nisa.scalar_tensor_tensor(dst=rf, data=rf, op0=nl.add, operand0=-nr, op1=nl.multiply, operand1=mk)
            nisa.tensor_scalar(dst=rf, data=rf, op0=nl.add, operand0=nr, engine=VE)  # exact: integers below 2^24
        ro = X["roff"]
        nisa.tensor_copy(dst=ro, src=rf, engine=VE)
        for j in range(keep):
            nisa.dma_copy(dst=cand[:, j * sub:(j + 1) * sub],
                          src=scr.ap(pattern=[[sub, 128], [1, sub]], offset=0,
                                     vector_offset=ro.ap(pattern=[[keep, 128], [1, 1]], offset=j),
                                     indirect_dim=0))

    def _prol(X, TP, c: int):
        """Chunk c's prologue: dsa_long_pipe._prologue2 (its K^T unless in KTall, its two accumulators), or for pe its
        K^T and the visibility mask (0 / NEG_INF) alone."""
        if not X["pe"]:
            _prologue2(X, TP, X["thr"], c)
            return
        x = c % 2
        if not X["ktall"]:
            _keys(X, TP, c, X["KT"][x])
        acc = X["ACC"][x]
        nisa.tensor_scalar(dst=acc, data=X["iota"], op0=nl.less, operand0=X["thr"][:, c:c + 1], engine=VE)
        nisa.activation(dst=acc, op=nl.copy, data=acc, scale=-NEG_INF, bias=NEG_INF)

    def _begin(X, t):
        """dsa_long_pipe._scores_begin without its chunk-0 prologue (tile t's queries, weights, npool and the chunks'
        thresholds); for pe also the tile's folded head weights, |w_h| scale per query (the relu's scale) and
        diag(sign w_h) bf16 per head (the head sum's stationary)."""
        Hi, D = X["Hi"], X["D"]
        qs, ws, npf = X["qs"], X["ws"], X["npf"]
        _dma(qs, _at(X, X["qT"], [[Hi * 128, D], [128, Hi], [1, 128]], t, D * Hi * 128))
        _dma(ws, _at(X, X["w"], [[Hi, 128], [1, Hi]], t, 128 * Hi))
        _dma(npf, _at(X, X["npool"], [[1, 128], [1, 1]], t, 128))
        nisa.tensor_scalar(dst=X["thr"], data=X["ciota"], op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=npf,
                           engine=VE)
        if not X["pe"]:
            return
        wa, sg = X["wabs"], X["wsg"]
        nisa.tensor_scalar(dst=sg, data=ws, op0=nl.multiply, operand0=-1.0, engine=VE)
        nisa.tensor_tensor(dst=wa, data1=ws, data2=sg, op=nl.maximum, engine=VE)
        nisa.tensor_scalar(dst=wa, data=wa, op0=nl.multiply, operand0=X["scale"], engine=VE)
        nisa.tensor_scalar(dst=sg, data=ws, op0=nl.greater_equal, operand0=0.0, op1=nl.multiply, operand1=2.0,
                           engine=VE)
        nisa.tensor_scalar(dst=sg, data=sg, op0=nl.add, operand0=-1.0, engine=VE)
        for h in range(Hi):
            nisa.tensor_scalar(dst=X["SG"][h], data=X["IB"], op0=nl.multiply, operand0=sg[:, h:h + 1], engine=VE)

    def _core(X, scr, bm, c: int):
        """Chunk c's scores after its prologue (dsa_long_pipe._scores_chunk without the next chunk's prologue: the
        same instructions; or for pe the head sum on the tensor engine), into bm and scr."""
        Hi, sub, Pp = X["Hi"], X["sub"], X["Pp"]
        qs, ws = X["qs"], X["ws"]
        TP, SP = X["psum"]
        x = c % 2
        kt = X["KTall"][:, c * CH:(c + 1) * CH] if X["ktall"] else X["KT"][x]
        acc = X["ACC"][x]
        if X["pe"]:
            hp = X["HP"][x]
            RB = X["RB"]
            for h in range(Hi):
                sp = SP[(c * Hi + h) % len(SP)]
                nisa.nc_matmul(dst=sp, stationary=qs[:, h, :], moving=kt, accumulate=False)
                rb = RB[(c * Hi + h) % len(RB)]
                nisa.activation(dst=rb, op=nl.relu, data=sp, scale=X["wabs"][:, h:h + 1], bias=X["zero"])
                nisa.nc_matmul(dst=hp, stationary=X["SG"][h], moving=rb, accumulate=True if h > 0 else False)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=hp, op=nl.add, engine=VE)
        else:
            acc1 = X["ACC1"][x]
            for h in range(Hi):
                sp = SP[(c * Hi + h) % len(SP)]
                nisa.nc_matmul(dst=sp, stationary=qs[:, h, :], moving=kt, accumulate=False)
                nisa.activation(dst=sp, op=nl.relu, data=sp, scale=X["sclt"], bias=X["zero"])
                a = acc1 if h % 2 else acc
                nisa.scalar_tensor_tensor(dst=a, data=sp, op0=nl.multiply, operand0=ws[:, h:h + 1], op1=nl.add,
                                          operand1=a)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=acc1, op=nl.add, engine=VE)
        nisa.tensor_reduce(dst=bm[:, c * (CH // sub):(c + 1) * (CH // sub)], op=nl.maximum,
                           data=acc.reshape((128, CH // sub, sub)), axis=2)
        _dma(scr.ap(pattern=[[Pp, 128], [1, CH]], offset=c * CH), acc)

    def _group_if(X, scr, bm, c0: int, c1: int, flag):
        """Chunks c0 .. c1 - 1 of the tile (each one's prologue hoisted into the chunk before it, as dsa_long_pipe's)
        when flag [1, 1] int32 is 1: a device loop of flag iterations around their static code. Its PSUM is allocated
        inside the loop body (as kernels/dsa_slots.py's): the tensor engine writing PSUM allocated outside a device
        loop fails neuronx-cc 2.27 (NCC_IBIR092 "Live-in/Live-out MemoryLocation ... not allocated to MemoryType:
        DRAM"; SBUF and HBM state across the loop, and a PSUM tile written by the scalar engine, compile:
        tools/probe_nki_loop_liveout.py, kiln-lc2-k1, 2026-10-07), so nothing outside the loops holds PSUM."""
        reg = nisa.register_alloc()
        nisa.register_load(reg, flag)

        def body(i):
            _psum_x(X)
            TP = X["psum"][0]
            _prol(X, TP, c0)
            for c in range(c0, c1):
                if c + 1 < c1:
                    _prol(X, TP, c + 1)
                _core(X, scr, bm, c)

        nl.fori_loop(0, reg, body)

    def _psum_x(X):
        """PSUM: dsa_long_select._psum's (two K^T banks, six score banks), or for pe two K^T banks, four score banks
        and two head-sum banks (8 of 8)."""
        if not X["pe"]:
            X["psum"] = _psum()
            return
        TP = [nl.ndarray((128, CH), dtype=F32, buffer=nl.psum), nl.ndarray((128, CH), dtype=F32, buffer=nl.psum)]
        SP = []
        for _ in range(4):
            SP.append(nl.ndarray((128, CH), dtype=F32, buffer=nl.psum))
        X["psum"] = (TP, SP)
        X["HP"] = [nl.ndarray((128, CH), dtype=F32, buffer=nl.psum), nl.ndarray((128, CH), dtype=F32, buffer=nl.psum)]

    @nki.jit
    def kiln_dsa_long_pipe_x_kernel(qT, w, npool, pk, identb, cflag, nbt, keep: int, scale: float, sub: int,
                                    lsub: int, one: int, rev: int, per_chunk: int, ktall: int, skip: int,
                                    pe: int = 0, vorder: int = 0, dbg: int = 0, group: int = 1):
        """kiln_dsa_long_pipe_kernel's arguments plus cflag int32 [NT, 1, NG] (chunk group g, chunks g group ..
        g group + group - 1, of tile t runs iff 1) and nbt fp32 [NT, 128, 1] (the tile's computed sub-blocks, on every
        partition), read when skip = 1; pe = 1 the head sum on the tensor engine."""
        NT, D, Hi, Q = qT.shape
        Pp = pk.shape[0]
        NC = Pp // CH
        NG = -(-NC // group)
        NB = Pp // sub
        out = nl.ndarray((NT, 128, keep), dtype=I32, buffer=nl.shared_hbm)
        outv = nl.ndarray((NT, 128, keep), dtype=F32, buffer=nl.shared_hbm)
        # the scores, q-major, by tile parity; row 128 NB a sub-block of NEG_INF (the skipped chunks' gathers)
        scrs = [nl.ndarray((128 * NB + 1, sub), dtype=F32, buffer=nl.shared_hbm),
                nl.ndarray((128 * NB + 1, sub), dtype=F32, buffer=nl.shared_hbm)]
        X = dict(Hi=Hi, D=D, Pp=Pp, NC=NC, NB=NB, two=1 - one, keep=keep, sub=sub, scale=scale, qT=qT, w=w,
                 npool=npool, pk=pk, out=out, outv=outv, lsub=lsub, dyn=0, vorder=vorder, ktall=ktall, skip=skip,
                 NR=128 * NB, pe=pe, dbg=dbg)
        IB = _sb((128, 128), BF16)
        _dma(IB, identb)
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
        X["sclt"] = _const(scale)
        X["zero"] = _const(0.0)
        X["PKr"] = [_sb((128, 4, D), BF16), _sb((128, 4, D), BF16)]
        X["KT"] = [_sb((128, CH), BF16), _sb((128, CH), BF16)]
        X["ACC"] = [_sb((128, CH)), _sb((128, CH))]
        X["ACC1"] = [_sb((128, CH)), _sb((128, CH))]
        bms = [_sb((128, NB)), _sb((128, NB))]
        W2 = keep * sub
        cands = [_sb((128, W2)), _sb((128, W2))]
        for name in ("V", "V2", "key", "rowf", "lof", "g", "pool", "vm", "srt", "vals", "mk"):
            X[name] = _sb((128, keep))
        X["blk2"] = [_sb((128, keep)), _sb((128, keep))]
        X["I"] = _sb((128, keep), U32)
        for name in ("roff", "hi", "lo", "oi"):
            X[name] = _sb((128, keep), I32)
        X["dep"] = _sb((128, 1))
        X["fence"] = _sb((128, 1))
        if not skip or dbg == 1:  # (with skip, each chunk's device loop allocates its own)
            _psum_x(X)
        if pe:
            X["RB"] = []
            for _ in range(4):
                X["RB"].append(_sb((128, CH), BF16))
            X["wabs"] = _sb((128, Hi))
            X["wsg"] = _sb((128, Hi))
            X["SG"] = []
            for _ in range(Hi):
                X["SG"].append(_sb((128, 128), BF16))
        fls = [_sb((1, NG), I32), _sb((1, NG), I32)]
        nbts = [_sb((128, 1)), _sb((128, 1))]
        if skip:
            ninf = _sb((1, sub))
            nisa.memset(dst=ninf, value=NEG_INF, engine=VE)
            _dma(scrs[0].ap(pattern=[[sub, 1], [1, sub]], offset=128 * NB * sub), ninf)
            _dma(scrs[1].ap(pattern=[[sub, 1], [1, sub]], offset=128 * NB * sub), ninf)
        if ktall:
            X["KTall"] = _sb((128, Pp), BF16)
            TPk = _psum()[0] if skip and dbg != 1 else X["psum"][0]  # (skip: PSUM used by nothing after the build)
            for c in range(NC):
                _keys(X, TPk, c, X["KTall"][:, c * CH:(c + 1) * CH])
        rounds = keep // 8
        for t in range(NT + 2):
            if t < NT:  # scores(t), with level 1 of tile t - 1 in its chunks
                bm = bms[t % 2]
                _begin(X, t)
                if not skip or dbg == 1:
                    _prol(X, X["psum"][0], 0)
                if skip:
                    fl = fls[t % 2]
                    _dma(fl, _at(X, cflag, [[NG, 1], [1, NG]], t, NG))
                    _dma(nbts[t % 2], _at(X, nbt, [[1, 128], [1, 1]], t, 128))
                    nisa.memset(dst=bm, value=NEG_INF, engine=VE)  # the skipped chunks' maxima
                for gi in range(NG):
                    c0 = gi * group
                    c1 = min(NC, c0 + group)
                    if skip and dbg != 1:
                        _group_if(X, scrs[t % 2], bm, c0, c1, fl[0:1, gi:gi + 1])
                    for c in range(c0, c1):
                        if not skip or dbg == 1:  # dsa_long_pipe's order: the next chunk's prologue, its scores
                            if c + 1 < NC:
                                _prol(X, X["psum"][0], c + 1)
                            _core(X, scrs[t % 2], bm, c)
                        if 1 <= t:  # level 1 of tile t - 1: between the groups (always run), else in each chunk
                            for k in range(per_chunk):
                                rr = c * per_chunk + k
                                if rr < rounds:
                                    _level1_round(X, bms[(t - 1) % 2], rr)
                if 1 <= t:
                    for rr in range(NC * per_chunk, rounds):
                        _level1_round(X, bms[(t - 1) % 2], rr)
                if skip:  # the gathers' fence: after every group on the vector engine (bm's last maximum may be the
                    # memset's, when the last group was skipped, and the gathers would go out under the scores)
                    nisa.tensor_scalar(dst=X["fence"], data=bm[:, NB - 1:NB], op0=nl.multiply, operand0=0.0, engine=VE)
            elif t == NT:  # the last tile's level 1
                for rr in range(rounds):
                    _level1_round(X, bms[(t - 1) % 2], rr)
            if 1 <= t <= NT:  # gathers(t - 1), after scores(t)
                src = bms[t % 2] if t < NT else bms[(t - 1) % 2]
                dep = X["fence"] if skip and t < NT else src[:, NB - 1:NB]
                _sort_gather_x(X, scrs[(t - 1) % 2], cands[(t - 1) % 2], (t - 1) % 2, dep, nbts[(t - 1) % 2])
            if 2 <= t:  # level2(t - 2) while the gathers are in flight
                _level2(X, cands[(t - 2) % 2], (t - 2) % 2, t - 2)
        return out, outv
else:
    kiln_dsa_long_pipe_x_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source, mixed with dsa_long_pipe.REV (the helpers it imports)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_dsa_long_pipe_x_kernel = None", a)
    return zlib.crc32(src[a:b].encode()) ^ dp.REV


REV = _kernel_rev()


def supported(n: int, Hi: int, D: int, P: int, keep: int, sub: int) -> bool:
    """dsa_long_pipe's shapes with two levels (sub > 0)."""
    return sub > 0 and dp.supported(n, Hi, D, P, keep, sub)


def chunk_bounds(npool: torch.Tensor, N: int, P: int, sub: int, group: int = 1):
    """(cflag int32 [NT, 1, NG], nbt fp32 [NT, 128, 1]): per 128-query tile, chunk group g (chunks g group ..) runs
    iff its first pool g group CH < the tile's largest npool (clamped to P), and the sub-blocks before its computed
    chunks' end."""
    NT = -(-N // 128)
    NC = -(-P // CH)
    NG = -(-NC // group)
    npf = npool.reshape(N).to(torch.float32).clamp(0, P)
    if NT * 128 != N:
        npf = torch.cat([npf, npf.new_zeros(NT * 128 - N)])
    top = npf.view(NT, 128).amax(-1)  # [NT]
    g0 = torch.arange(NG, device=npool.device, dtype=torch.float32).view(1, NG) * (group * CH)
    run = g0 < top.view(NT, 1)  # [NT, NG]: the group holds some query's candidate
    nct = (run.to(torch.float32).sum(-1) * group).clamp(max=NC)  # chunks computed (a prefix)
    nbt = (nct * (CH // sub)).view(NT, 1, 1).expand(NT, 128, 1).contiguous()
    return run.to(torch.int32).view(NT, 1, NG).contiguous(), nbt


def select(qI: torch.Tensor, w: torch.Tensor, pk: torch.Tensor, npool: torch.Tensor, keep: int, scale: float,
           sub: int | None = None, vorder: bool = False, skip: bool = True, pe: bool = False):
    """dsa_long_pipe.select (the same pools, count and scores) with skip: each tile's chunks past its largest
    npool not scored; with pe the head sum on the tensor engine (not exact: bf16 head terms)."""
    N = qI.shape[0]
    P = pk.shape[0]
    if qI.device.type == "cpu":
        return dls.emulate(qI, w, pk, npool, keep, scale, vorder)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if kiln_dsa_long_pipe_x_kernel is None:
        raise RuntimeError("the pipelined long-context DSA selection kernel needs the nki package (the Neuron venv)")
    sub = dls.pick_sub(P, keep) if sub is None else sub
    if not supported(N, qI.shape[1], qI.shape[2], P, keep, sub):
        raise NotImplementedError(f"pipelined long-context DSA selection (chunk skip): N {N}, Hi {qI.shape[1]}, "
                                  f"D {qI.shape[2]}, P {P}, keep {keep}, sub {sub}")
    cflag, nbt = chunk_bounds(npool, N, P, sub, GROUP)
    out, outv = wrap_nki(kiln_dsa_long_pipe_x_kernel)[platform.nki_grid()](
        **dls.kernel_inputs(qI, w, pk, npool), cflag=cflag, nbt=nbt, keep=int(keep), scale=float(scale),
        sub=int(sub), lsub=int(sub).bit_length() - 1, one=0, rev=REV, per_chunk=int(dp.PER_CHUNK),
        ktall=int(-(-P // CH) * CH <= dp.KTALL_MAX), skip=int(skip), **({"pe": 1} if pe else {}),
        **({"dbg": DBG} if DBG else {}), **({"group": GROUP} if skip and GROUP != 1 else {}),
        **({"vorder": 1} if vorder else {}))
    cnt = npool.to(torch.int64).clamp(0, P).clamp(max=keep)
    return out.reshape(-1, keep)[:N].to(torch.int64), cnt, outv.reshape(-1, keep)[:N]
