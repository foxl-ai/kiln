"""kernels/dsa_long_select.py's single query tile under a run-time trip count of 0 or 1: the same instructions on the same
data when trip is 1 (so the same pools, order and scores, bit for bit), nothing computed and nothing written when it
is 0 (the outputs then hold whatever their HBM held; the caller masks them).

Why: under KILN_DSA_CP_LOCAL_K (models/mla.py) the rows whose local top-K certificate fails take one keep-512
selection tile of 128 rows per DSA layer; on real text that set is normally empty, yet the tile costs a full call
(~4.4 ms at 32,768 local pools, trn1). Here the tile's body is a device loop of `trip` iterations (register_load +
nl.fori_loop, as kernels/dsa_slots_n.py's row count), trip computed on the device from the failed-row count, so an
empty set costs the loop's entry and nothing else. dsa_long_select's own device loop was wrong past its first tile
(models/mla.py _select_tiles); this one never runs a second. The tile's PSUM is allocated inside the loop body
(dsa_long_select._tile's per-loop-region form): the tensor engine writing PSUM allocated outside a device loop fails
neuronx-cc 2.27 (kernels/dsa_long_pipe_x.py, tools/probe_nki_loop_liveout.py).

Measured (tools/probe_lc_select_t.py, 128 queries, kiln-lc2-k1, 2026-10-07): at trip 1 bit-identical to
dsa_long_select on every probe kind at 8,192 / 32,768 pools x keep 512 / 128, and 4.686 against 4.644 ms (32,768,
keep 512), 2.619 / 2.566 (keep 128); at trip 0 0.24 ms (the call's input layout and launch).

A separate module so that kernels/dsa_long_select.py's source, its REV and the keys of every graph that calls it, are
unchanged; its REV is this module's kernel source and dsa_long_select.REV.
"""

from __future__ import annotations

import torch

from . import dsa_long_select as dls

CH = dls.CH

try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None and dls.nki is not None:
    from .dsa_long_select import _const, _sb, _tile  # the tracer resolves names

    F32, BF16, I32, U32 = nl.float32, nl.bfloat16, nl.int32, nl.uint32
    VE = nisa.vector_engine

    @nki.jit
    def kiln_dsa_long_select_t_kernel(qT, w, npool, pk, identb, trip, keep: int, scale: float, sub: int, lsub: int,
                                      one: int, rev: int, vorder: int = 0):
        """kiln_dsa_long_select_kernel's arguments for ONE tile (qT [1, D, Hi, 128]) plus trip int32 [1, 1] (0 or
        1): the tile's body runs trip times."""
        NT, D, Hi, Q = qT.shape
        Pp = pk.shape[0]
        NC = Pp // CH
        NB = Pp // sub
        out = nl.ndarray((NT, 128, keep), dtype=I32, buffer=nl.shared_hbm)
        outv = nl.ndarray((NT, 128, keep), dtype=F32, buffer=nl.shared_hbm)
        scr = nl.ndarray((128 * NB, sub), dtype=F32, buffer=nl.shared_hbm)  # the scores, q-major
        X = dict(Hi=Hi, D=D, Pp=Pp, NC=NC, NB=NB, two=1 - one, keep=keep, sub=sub, scale=scale, qT=qT, w=w,
                 npool=npool, pk=pk, out=out, outv=outv, scr=scr, scr2=scr, dbg=0, psum=None, lsub=lsub, dyn=0,
                 vorder=vorder, dscr=None)
        npg = nl.num_programs(axes=0) if nl.program_ndim() != 0 else 1  # lnc2
        idle = npg == 2 and nl.program_id(axis=0) == 1  # lnc2: program 0 alone selects (dsa_long_select._lnc2_note)
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
        X["sclt"] = _const(scale)
        X["zero"] = _const(0.0)
        X["one"] = _const(1.0)
        X["big"] = _const(-dls.NEG_INF)
        X["negb"] = _const(dls.NEG_INF)
        X["PKr"] = [_sb((128, 4, D), BF16), _sb((128, 4, D), BF16)]
        X["KT"] = [_sb((128, CH), BF16), _sb((128, CH), BF16)]
        X["ACC"] = [_sb((128, CH)), _sb((128, CH))]
        X["ACC1"] = [_sb((128, CH)), _sb((128, CH))]
        X["bm"] = _sb((128, NB))
        W2 = keep * sub if not one else Pp
        X["cand"] = _sb((128, W2))
        for name in ("V", "V2", "key", "blk", "rowf", "lof", "g", "pool", "vm", "srt", "vals"):
            X[name] = _sb((128, keep))
        X["I"] = _sb((128, keep), U32)
        for name in ("roff", "hi", "lo", "oi"):
            X[name] = _sb((128, keep), I32)
        cnt = _sb((1, 1), I32)
        nisa.dma_copy(dst=cnt, src=trip)
        if idle:  # lnc2: program 1 runs the same device loop zero times (the two cores' code must have the same basic
            nisa.iota(dst=cnt, pattern=[[0, 1]], offset=0, channel_multiplier=0)  # lnc2: blocks, NCC_IXGM002)
        rt = nisa.register_alloc()
        nisa.register_load(rt, cnt)

        def body(i):
            _tile(X, 0)

        nl.fori_loop(0, rt, body)
        if npg == 2:  # lnc2
            nisa.core_barrier(data=out, cores=(0, 1))  # lnc2
        return out, outv
else:
    kiln_dsa_long_select_t_kernel = None


def _kernel_rev(lnc2: bool = False) -> int:
    """CRC-32 of this module's kernel source, mixed with dsa_long_select's REV (the tile code it runs). Lines marked
    "# lnc2" act only at grid 2 (trn2) and are left out of the grid-1 revision, as in dsa_long_select."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_dsa_long_select_t_kernel = None", a)
    body = src[a:b]
    if not lnc2:
        body = "".join(x for x in body.splitlines(keepends=True) if "# lnc2" not in x)
    return zlib.crc32(body.encode()) ^ (dls.REV_LNC2 if lnc2 else dls.REV)


REV = _kernel_rev()
REV_LNC2 = _kernel_rev(lnc2=True)


def select(qI: torch.Tensor, w: torch.Tensor, pk: torch.Tensor, npool: torch.Tensor, keep: int, scale: float,
           trip: torch.Tensor, sub: int | None = None, vorder: bool = False):
    """dsa_long_select.select of at most 128 queries when trip (an int tensor of one element, 0 or 1) is 1; when it is
    0 the pools and scores are unwritten device memory (emulate() on the host either way)."""
    N = qI.shape[0]
    P = pk.shape[0]
    if qI.device.type == "cpu":
        return dls.emulate(qI, w, pk, npool, keep, scale, vorder)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if kiln_dsa_long_select_t_kernel is None:
        raise RuntimeError("the NKI long-context DSA selection kernel needs the nki package (the Neuron venv)")
    sub = dls.pick_sub(P, keep) if sub is None else sub
    if N > 128 or not dls.supported(qI.shape[1], qI.shape[2], P, keep, sub):
        raise NotImplementedError(f"long-context DSA selection (one tile, run-time trip): N {N}, Hi {qI.shape[1]}, "
                                  f"D {qI.shape[2]}, P {P}, keep {keep}, sub {sub}")
    out, outv = wrap_nki(kiln_dsa_long_select_t_kernel)[platform.nki_grid()](
        **dls.kernel_inputs(qI, w, pk, npool), trip=trip.reshape(1, 1).to(torch.int32), keep=int(keep),
        scale=float(scale), sub=int(sub or CH), lsub=int(sub or CH).bit_length() - 1, one=int(sub == 0),
        rev=REV_LNC2 if platform.nki_grid() == 2 else REV,
        **({"vorder": 1} if vorder else {}))
    cnt = npool.to(torch.int64).clamp(0, P).clamp(max=keep)
    return out.reshape(-1, keep)[:N].to(torch.int64), cnt, outv.reshape(-1, keep)[:N]
