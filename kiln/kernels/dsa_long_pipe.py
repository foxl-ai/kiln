"""kernels/dsa_long_select.py over several 128-query tiles in ONE call, software-pipelined: the same instructions on the
same data (so the same pools, order and scores, bit for bit), issued so that a tile's selection overlaps the next
tile's scores.

Why (trn1, nki 0.6.0, one NeuronCore, 128 queries x 32,768 pools, keep 512, sub 8; device profile of
tools/prof_attn_kernels.py dsa_long, kiln-lc2-k1, 2026-10-06): the single-tile kernel's 4.4 ms is the scores (64 chunks
of 512 pools x 32 heads, 1.97 ms, the vector engine's weighted head sum 1.22 ms of it, the scalar engine's relu 1.04 ms)
and then the selection on the vector engine alone (2.42 ms: level 1 and level 2 extraction, 64 rounds each of max8 /
nc_find_index8 / nc_match_replace8 at ~3.9 us per instruction over 4,096 values, the sort of the selected sub-blocks)
with a 0.71 ms hole in the middle: the 512 indirect DMAs that gather the selected sub-blocks' scores back from HBM
(software DGE on the GpSimd engine, ~1.4 us each) while every engine waits. The serving path calls the kernel once
per tile (models/mla.py _select_tiles: the single-tile kernel's device loop was wrong past the first tile), so nothing
overlaps across tiles either.

Here the tiles are unrolled (static offsets, no device loop) and run as a three-stage pipeline. Iteration t emits

    scores(t)    its chunks, with level 1 of tile t - 1 (the sub-block extraction over its maxima) interleaved: the
                 vector engine waits on the scalar engine's relu inside a chunk, and the extraction fills that time;
    gathers(t-1) tile t - 1's sub-block sort and its 512 gathers, made to depend on the end of scores(t) (a +-0 from
                 its last sub-block maximum added to the gathers' row offsets): issued earlier, the chunks' own score
                 writes queued behind the gathers' 65,536 small descriptors and the chunks stalled ~0.58 ms;
    level2(t-2)  tile t - 2's candidate extraction and output, on the vector engine while the gathers are in flight.

Two of each buffer the stages hold across an iteration (sub-block maxima, candidates, selected sub-blocks) and two
score scratches in HBM; the rest are the single-tile kernel's buffers in its own program order per buffer.

A separate module so that kernels/dsa_long_select.py's source, its REV and the keys of every graph that calls it, are
unchanged; its REV is this module's kernel source and dsa_long_select.REV.
"""

from __future__ import annotations

import os

import torch

from . import dsa_long_select as dls

CH = dls.CH
NEG_INF = dls.NEG_INF
VISIBLE = dls.VISIBLE
# Level-1 rounds of the previous tile emitted after each score chunk (64 rounds over 64 chunks at 32,768 pools).
PER_CHUNK = int(os.environ.get("KILN_DSA_LONG_PIPE_ROUNDS", 1))
MAX_TILES = 16
# Every chunk's transposed pool keys held in SBUF for the whole call (bf16 [128, P]), up to KTALL_MAX pools (16 KiB
# per partition at 8,192); above it each tile transposes its chunks as dsa_long_select does.
KTALL_MAX = int(os.environ.get("KILN_DSA_LONG_PIPE_KTALL_MAX", 8192))

try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None and dls.nki is not None:
    from .dsa_long_select import REPL, _at, _const, _psum, _sb  # the tracer resolves names

    F32, BF16, I32, U32 = nl.float32, nl.bfloat16, nl.int32, nl.uint32
    VE = nisa.vector_engine

    def _round(V, I, cur, r: int, rounds: int):
        """Round r of dsa_long_select._extract (its instructions for that round, in its order)."""
        v = V[:, 8 * r:8 * r + 8]
        nisa.max8(dst=v, src=cur)
        if I is not None:
            nisa.nc_find_index8(dst=I[:, 8 * r:8 * r + 8], data=cur, vals=v)
        if r + 1 < rounds:
            nisa.nc_match_replace8(dst=cur, data=cur, vals=v, imm=REPL)

    def _dma(dst, src):
        """A DMA with every address static, its descriptors made before the kernel runs (dge_mode none): off the
        GpSimd engine's software-DGE stream, which issues the indirect gathers one at a time."""
        nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.none)

    def _keys(X, TP, c, dst):
        """Chunk c's pool keys transposed into dst [128 d, CH] (dsa_long_select._prologue's first half)."""
        x = c % 2
        D = X["D"]
        pkr = X["PKr"][x]
        _dma(pkr, X["pk"].ap(pattern=[[D, 128], [128 * D, 4], [1, D]], offset=c * CH * D))
        tp = TP[x]
        for i in range(4):
            nisa.nc_matmul(dst=tp[:, i * 128:(i + 1) * 128], stationary=pkr[:, i, :], moving=X["IB"],
                           accumulate=False)
        nisa.activation(dst=dst, op=nl.copy, data=tp)

    def _prologue2(X, TP, thr, c):
        """dsa_long_select._prologue (chunk c's K^T unless they are all in KTall, and its accumulators)."""
        x = c % 2
        if not X["ktall"]:
            _keys(X, TP, c, X["KT"][x])
        acc = X["ACC"][x]
        nisa.tensor_scalar(dst=acc, data=X["iota"], op0=nl.less, operand0=thr[:, c:c + 1], engine=VE)
        nisa.activation(dst=acc, op=nl.copy, data=acc, scale=-NEG_INF, bias=NEG_INF)
        nisa.memset(dst=X["ACC1"][x], value=0.0)

    def _scores_begin(X, t):
        """dsa_long_select._tile's score part for tile t (static offsets) up to its chunk loop."""
        Hi, D = X["Hi"], X["D"]
        qs, ws, npf = X["qs"], X["ws"], X["npf"]
        _dma(qs, _at(X, X["qT"], [[Hi * 128, D], [128, Hi], [1, 128]], t, D * Hi * 128))
        _dma(ws, _at(X, X["w"], [[Hi, 128], [1, Hi]], t, 128 * Hi))
        _dma(npf, _at(X, X["npool"], [[1, 128], [1, 1]], t, 128))
        TP, SP = X["psum"]
        thr = X["thr"]
        nisa.tensor_scalar(dst=thr, data=X["ciota"], op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=npf,
                           engine=VE)
        _prologue2(X, TP, thr, 0)

    def _scores_chunk(X, scr, bm, c: int):
        """Chunk c of dsa_long_select._tile's score loop, into the sub-block maxima bm and the scratch scr."""
        Hi, NC, sub, Pp = X["Hi"], X["NC"], X["sub"], X["Pp"]
        qs, ws = X["qs"], X["ws"]
        TP, SP = X["psum"]
        x = c % 2
        if c + 1 < NC:
            _prologue2(X, TP, X["thr"], c + 1)
        kt = X["KTall"][:, c * CH:(c + 1) * CH] if X["ktall"] else X["KT"][x]
        acc = X["ACC"][x]
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

    def _level1_round(X, bm, r: int):
        """Round r of dsa_long_select._tile's level-1 extraction over the sub-block maxima bm."""
        if X["two"]:
            _round(X["V"], X["I"], bm, r, X["keep"] // 8)

    def _sort_gather(X, scr, cand, par: int, dep_from):
        """dsa_long_select._tile's level 1 after its extraction (the selected sub-blocks ascending, their scratch
        rows) and the gathers into cand, or for one level the whole score row; the row offsets take +-0 from
        dep_from [128, 1] (finite), so the gathers wait for whatever produced it."""
        keep, sub, Pp = X["keep"], X["sub"], X["Pp"]
        V, I = X["V"], X["I"]
        rounds = keep // 8
        if not X["two"]:
            nisa.dma_copy(dst=cand[:, 0:Pp], src=scr.ap(pattern=[[Pp, 128], [1, Pp]], offset=0))
            return
        key = X["key"]
        nisa.tensor_copy(dst=key, src=I.view(I32), engine=VE)  # uint32 -> fp32, exact (< 2^24)
        nisa.tensor_scalar(dst=key, data=key, op0=nl.multiply, operand0=-1.0, engine=VE)
        blk = X["blk2"][par]
        for r in range(rounds):  # dsa_long_select._sort_asc
            _round(V, None, key, r, rounds)
        nisa.tensor_scalar(dst=blk, data=V, op0=nl.multiply, operand0=-1.0, engine=VE)
        rf = X["rowf"]
        nisa.tensor_scalar(dst=rf, data=blk, op0=nl.add, operand0=X["qnb"], engine=VE)
        dep = X["dep"]
        nisa.tensor_scalar(dst=dep, data=dep_from, op0=nl.multiply, operand0=0.0, engine=VE)
        nisa.tensor_scalar(dst=rf, data=rf, op0=nl.add, operand0=dep, engine=VE)
        ro = X["roff"]
        nisa.tensor_copy(dst=ro, src=rf, engine=VE)
        for j in range(keep):
            nisa.dma_copy(dst=cand[:, j * sub:(j + 1) * sub],
                          src=scr.ap(pattern=[[sub, 128], [1, sub]], offset=0,
                                     vector_offset=ro.ap(pattern=[[keep, 128], [1, 1]], offset=j),
                                     indirect_dim=0))

    def _level2(X, cand, par: int, t):
        """dsa_long_select._tile's level 2 over cand and the output of tile t (parity par's selected sub-blocks)."""
        keep, sub, lsub, Pp = X["keep"], X["sub"], X["lsub"], X["Pp"]
        I, V, V2 = X["I"], X["V"], X["V2"]
        rounds = keep // 8
        W2 = keep * sub if X["two"] else Pp
        for r in range(rounds):
            _round(V2, I, cand[:, 0:W2], r, rounds)
        pool = X["pool"]
        if X["two"]:
            blk = X["blk2"][par]
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
        vm = X["vm"]
        nisa.tensor_scalar(dst=vm, data=V2, op0=nl.greater, operand0=VISIBLE, engine=VE)
        srt = X["srt"]
        vals = X["vals"]
        oi = X["oi"]
        if X["vorder"]:
            nisa.tensor_tensor(dst=srt, data1=pool, data2=vm, op=nl.multiply, engine=VE)
            nisa.tensor_tensor(dst=vals, data1=V2, data2=vm, op=nl.multiply, engine=VE)
            nisa.tensor_scalar(dst=vm, data=vm, op0=nl.multiply, operand0=-NEG_INF, op1=nl.add, operand1=NEG_INF,
                               engine=VE)
            nisa.tensor_tensor(dst=vals, data1=vals, data2=vm, op=nl.add, engine=VE)
            nisa.tensor_copy(dst=oi, src=srt, engine=VE)
        else:
            key = X["key"]
            nisa.tensor_scalar(dst=key, data=pool, op0=nl.multiply, operand0=-1.0, op1=nl.add,
                               operand1=float(1 << 23), engine=VE)
            nisa.tensor_tensor(dst=key, data1=key, data2=vm, op=nl.multiply, engine=VE)
            nisa.tensor_scalar(dst=key, data=key, op0=nl.add, operand0=-float(1 << 23), engine=VE)
            for r in range(rounds):
                _round(V, I, key, r, rounds)
            nisa.tensor_scalar(dst=srt, data=V, op0=nl.multiply, operand0=-1.0, engine=VE)
            nisa.nc_n_gather(dst=vals, data=V2, indices=I)
            nisa.tensor_scalar(dst=vm, data=srt, op0=nl.less, operand0=float(1 << 23), engine=VE)
            nisa.tensor_tensor(dst=srt, data1=srt, data2=vm, op=nl.multiply, engine=VE)
            nisa.tensor_tensor(dst=vals, data1=vals, data2=vm, op=nl.multiply, engine=VE)
            nisa.tensor_scalar(dst=vm, data=vm, op0=nl.multiply, operand0=-NEG_INF, op1=nl.add, operand1=NEG_INF,
                               engine=VE)
            nisa.tensor_tensor(dst=vals, data1=vals, data2=vm, op=nl.add, engine=VE)
            nisa.tensor_copy(dst=oi, src=srt, engine=VE)
        _dma(_at(X, X["out"], [[keep, 128], [1, keep]], t, 128 * keep), oi)
        _dma(_at(X, X["outv"], [[keep, 128], [1, keep]], t, 128 * keep), vals)

    @nki.jit
    def kiln_dsa_long_pipe_kernel(qT, w, npool, pk, identb, keep: int, scale: float, sub: int, lsub: int, one: int,
                                  rev: int, per_chunk: int, ktall: int, vorder: int = 0):
        """kiln_dsa_long_select_kernel's arguments (every tile unrolled) plus per_chunk (level-1 rounds of the
        previous tile per score chunk) and ktall (every chunk's K^T in SBUF once for all tiles)."""
        NT, D, Hi, Q = qT.shape
        Pp = pk.shape[0]
        NC = Pp // CH
        NB = Pp // sub
        out = nl.ndarray((NT, 128, keep), dtype=I32, buffer=nl.shared_hbm)
        outv = nl.ndarray((NT, 128, keep), dtype=F32, buffer=nl.shared_hbm)
        scrs = [nl.ndarray((128 * NB, sub), dtype=F32, buffer=nl.shared_hbm),
                nl.ndarray((128 * NB, sub), dtype=F32, buffer=nl.shared_hbm)]  # the scores, q-major, by tile parity
        X = dict(Hi=Hi, D=D, Pp=Pp, NC=NC, NB=NB, two=1 - one, keep=keep, sub=sub, scale=scale, qT=qT, w=w,
                 npool=npool, pk=pk, out=out, outv=outv, lsub=lsub, dyn=0, vorder=vorder, ktall=ktall)
        npg = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1  # lnc2
        if npg == 2 and nl.program_id(axis=0) == 1:  # lnc2: program 0 alone selects (dsa_long_select._lnc2_note)
            nisa.core_barrier(data=out, cores=(0, 1))  # lnc2
            return out, outv  # lnc2
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
        W2 = keep * sub if not one else Pp
        cands = [_sb((128, W2)), _sb((128, W2))]
        for name in ("V", "V2", "key", "rowf", "lof", "g", "pool", "vm", "srt", "vals"):
            X[name] = _sb((128, keep))
        X["blk2"] = [_sb((128, keep)), _sb((128, keep))]
        X["I"] = _sb((128, keep), U32)
        for name in ("roff", "hi", "lo", "oi"):
            X[name] = _sb((128, keep), I32)
        X["dep"] = _sb((128, 1))
        X["psum"] = _psum()
        if ktall:
            X["KTall"] = _sb((128, Pp), BF16)
            for c in range(NC):
                _keys(X, X["psum"][0], c, X["KTall"][:, c * CH:(c + 1) * CH])
        rounds = keep // 8
        for t in range(NT + 2):
            if t < NT:  # scores(t), with level 1 of tile t - 1 in its chunks
                bm = bms[t % 2]
                _scores_begin(X, t)
                for c in range(NC):
                    _scores_chunk(X, scrs[t % 2], bm, c)
                    if 1 <= t:
                        for k in range(per_chunk):
                            rr = c * per_chunk + k
                            if rr < rounds:
                                _level1_round(X, bms[(t - 1) % 2], rr)
                if 1 <= t:
                    for rr in range(NC * per_chunk, rounds):
                        _level1_round(X, bms[(t - 1) % 2], rr)
            elif t == NT:  # the last tile's level 1
                for rr in range(rounds):
                    _level1_round(X, bms[(t - 1) % 2], rr)
            if 1 <= t <= NT:  # gathers(t - 1), after scores(t)
                src = bms[t % 2] if t < NT else bms[(t - 1) % 2]
                _sort_gather(X, scrs[(t - 1) % 2], cands[(t - 1) % 2], (t - 1) % 2, src[:, NB - 1:NB])
            if 2 <= t:  # level2(t - 2) while the gathers are in flight
                _level2(X, cands[(t - 2) % 2], (t - 2) % 2, t - 2)
        if npg == 2:  # lnc2
            nisa.core_barrier(data=out, cores=(0, 1))  # lnc2
        return out, outv
else:
    kiln_dsa_long_pipe_kernel = None


def _kernel_rev(lnc2: bool = False) -> int:
    """CRC-32 of this module's kernel source, mixed with dsa_long_select's REV (the helpers it imports). Lines marked
    "# lnc2" act only at grid 2 (trn2) and are left out of the grid-1 revision, as in dsa_long_select."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_dsa_long_pipe_kernel = None", a)
    body = src[a:b]
    if not lnc2:
        body = "".join(x for x in body.splitlines(keepends=True) if "# lnc2" not in x)
    return zlib.crc32(body.encode()) ^ (dls.REV_LNC2 if lnc2 else dls.REV)


REV = _kernel_rev()
REV_LNC2 = _kernel_rev(lnc2=True)


def supported(n: int, Hi: int, D: int, P: int, keep: int, sub: int) -> bool:
    """Shapes the pipelined kernel takes: dsa_long_select's, two to MAX_TILES whole tiles of 128 queries."""
    NT = -(-n // 128)
    return 2 <= NT <= MAX_TILES and dls.supported(Hi, D, P, keep, sub)


def select(qI: torch.Tensor, w: torch.Tensor, pk: torch.Tensor, npool: torch.Tensor, keep: int, scale: float,
           sub: int | None = None, vorder: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """dsa_long_select.select over every tile of qI in one pipelined call: the same (pools, count, scores) as one
    dsa_long_select.select call per 128 queries (emulate() on the host)."""
    N = qI.shape[0]
    P = pk.shape[0]
    if qI.device.type == "cpu":
        return dls.emulate(qI, w, pk, npool, keep, scale, vorder)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if kiln_dsa_long_pipe_kernel is None:
        raise RuntimeError("the pipelined long-context DSA selection kernel needs the nki package (the Neuron venv)")
    sub = dls.pick_sub(P, keep) if sub is None else sub
    if not supported(N, qI.shape[1], qI.shape[2], P, keep, sub):
        raise NotImplementedError(f"pipelined long-context DSA selection: N {N}, Hi {qI.shape[1]}, D {qI.shape[2]}, "
                                  f"P {P}, keep {keep}, sub {sub}")
    out, outv = wrap_nki(kiln_dsa_long_pipe_kernel)[platform.nki_grid()](
        **dls.kernel_inputs(qI, w, pk, npool), keep=int(keep), scale=float(scale), sub=int(sub or CH),
        lsub=int(sub or CH).bit_length() - 1, one=int(sub == 0), rev=REV_LNC2 if platform.nki_grid() == 2 else REV,
        per_chunk=int(PER_CHUNK),
        ktall=int(-(-P // CH) * CH <= KTALL_MAX), **({"vorder": 1} if vorder else {}))
    cnt = npool.to(torch.int64).clamp(0, P).clamp(max=keep)
    return out.reshape(-1, keep)[:N].to(torch.int64), cnt, outv.reshape(-1, keep)[:N]
