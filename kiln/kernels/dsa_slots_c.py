"""kernels/dsa_slots.py with each row's live slots compacted IN the kernel: a row attends only the slots that hold a
visible token, gathered into one 128-slot chunk, instead of all NS slots of its fixed layout.

Why: under context parallelism (models/mla.py attention_cp, a decode batch) a rank's row lists the rank's local pools
(keep = 512 slots), the tail pool and padding: 640 slots, of which only the rank's share of the global top-keep is
selected (keep / A, ~64 at A = 8) plus the tail when the rank owns it. dsa_slots attends all 640 (39.5 us per row at 64
heads, fp8, trn1) while 128 slots cost 14.9 us (kernels/dsa_slots_n.py's measurement). The value-ordered lists of the
chunk path (dsa_long_select vorder, attention_cp's slot classes) need a sort and per-element gathers in XLA (~0.4 ms
each per decode layer on trn1), so here the compaction is the kernel's own work, on the engines, per row:

  1. the slot tables of an iteration's rows (RPI of them), one slot per (partition p, chunk ch) for slot ch 128 + p:
     its pool row as three bf16-exact digits of 8 bits (hi, mid, lo; rows < 2^24) and its live token count nv (0 .. KP;
     the visible tokens of a slot are a prefix of its pool: every token of a selected pool, the tail pool's first ones);
  2. live m = nv > 0; its inclusive prefix over the chunks of a partition (vector adds) and the exclusive prefix of the
     partitions' counts (one matmul against the strictly-upper ones, every row of the iteration at once): every live
     slot's place c in the compacted list (partition-major order), dead slots placed past it;
  3. per (row, chunk) the one-hot [p, c] = (c == place) (vector engine) and its partial of the compacted (hi, mid, lo,
     nv)[c] = one-hot^T (hi, mid, lo, nv) (tensor engine), each into a PSUM region of its own with accumulate=False (on
     trn1 several accumulation chains in one PSUM tensor lose partials), summed over the chunks on the vector engine
     (exactly one nonzero term per entry, so exact): the compacted pool rows and live counts on partitions c = 0 .. 127,
     empty entries row 0 with nv 0; then every row's gather of its 128 compacted pool rows is issued, before any row
     attends;
  4. nv^T of every row by one PE transpose (row k on partition k), and the bias rows of the 512 compacted tokens (token t
     of slot c at t 128 + c) as nv[c] > t ? 0 : -1, which the bias matmul scales to NEG_INF / scale and picks per row by
     a one-hot stationary;
  5. dsa_slots' row (scores, softmax, P K, lse) over those 512 tokens.

A row with more live slots than one chunk holds (> 128) is never dropped: the caller counts every row's live slots, and
when ANY row of the call exceeds 128 the kernel runs no iteration (n_c = 0) and kernels/dsa_slots_n.py attends every row
over all NS slots instead (its row count is N then, 0 otherwise), so the result is exact for any split. The fallback is
a second kernel call because a second device loop inside this kernel failed neuronx-cc in the decode graph (see the
kernel's docstring). The compacted form attends the same tokens with the same scores; only fp32 summation order differs
from dsa_slots (its 128-token blocks group other tokens), so the two agree to fp32 rounding, not bit for bit.

The kernel is a separate module so that kernels/dsa_slots.py's source, its REV and the keys of every graph that calls it
are unchanged; its REV is this module's kernel source and dsa_slots.REV.
"""

from __future__ import annotations

import os

import torch

from . import dsa_slots

NEG_INF = dsa_slots.NEG_INF
KP = dsa_slots.KP
C = 128  # compacted slots per row (one chunk)
# Rows per device-loop iteration. The compaction of an iteration's rows runs before any of them attends, so their K
# gathers are in flight together; more rows amortise the one exposed compaction per iteration. One trn1 core, 96 rows x
# 640 slots, 64 heads, fp8, 64 live slots per row and a tail on every other row (tools/probe_dsa_long.py slots_c,
# kiln-dc2-k, 2026-10-06), us per row, attend() with its dsa_slots_n call: RPI 8 17.91, 16 17.19 (the kernel alone 16.55
# / 15.10); dsa_slots 41.6.
RPI = int(os.environ.get("KILN_DSA_SLOTS_C_RPI", 16))


def live_tokens(bias: torch.Tensor) -> torch.Tensor:
    """[N, NS] fp32: the visible tokens of every slot (bias 0, not NEG_INF), 0 .. KP."""
    # a tensor, not a float literal: those lower to f64 (NCC_ESPP004, trn1)
    return (bias > torch.full_like(bias, NEG_INF / 2)).to(torch.float32).sum(-1)


def plan(rows: torch.Tensor, nv: torch.Tensor):
    """The kernel's compaction in torch (steps 2-3): (crow [N, C] int64, cnv [N, C] fp32, fits [N] bool). Slot
    s = ch 128 + p is placed in partition-major order (p, then ch): place = (live slots of the partitions before p) +
    (live slots of chunks before ch on partition p). Entries past a row's live count are row 0 with nv 0; a row with
    more than C live slots does not fit (its placed entries past C are dropped here, as the kernel's one-hot drops
    them, and the caller runs the full form for the whole call)."""
    N, NS = rows.shape
    NCH = -(-NS // 128)
    pad = NCH * 128 - NS
    r = torch.nn.functional.pad(rows.long(), (0, pad))
    v = torch.nn.functional.pad(nv.float(), (0, pad))
    m = (v > 0).view(N, NCH, 128).permute(0, 2, 1).float()  # [N, p, ch]
    incl = torch.cumsum(m, dim=2)
    cntp = incl[:, :, -1]  # [N, p]
    excl = torch.cumsum(cntp, dim=1) - cntp
    place = excl.unsqueeze(-1) + incl - m  # [N, p, ch]
    place = torch.where(m > 0, place, torch.full_like(place, 4096.0))
    flat = place.permute(0, 2, 1).reshape(N, NCH * 128)  # back to slot order
    crow = torch.zeros(N, C, dtype=torch.int64, device=rows.device)
    cnv = torch.zeros(N, C, dtype=torch.float32, device=rows.device)
    keep = flat < C
    b, s = keep.nonzero(as_tuple=True)
    c = flat[b, s].long()
    crow[b, c] = r[b, s]
    cnv[b, c] = v[b, s]
    fits = (m.sum((1, 2)) <= C)
    return crow, cnv, fits


def emulate(q_lat: torch.Tensor, kc: torch.Tensor, rows: torch.Tensor, bias: torch.Tensor, scale: float,
            lse: bool = False):
    """The kernel's result in torch: when every row's live slots fit C, dsa_slots.emulate over the compacted slots
    (plan(); the bias of compacted token t of slot c is 0 for t < nv[c], NEG_INF otherwise); else dsa_slots.emulate
    over all slots (attend's dsa_slots_n call)."""
    nv = live_tokens(bias)
    crow, cnv, fits = plan(rows, nv)
    if not bool(fits.all()):
        return dsa_slots.emulate(q_lat, kc, rows, bias, scale, lse)
    t = torch.arange(KP, device=rows.device).view(1, 1, KP)
    cb = torch.where(t < cnv.unsqueeze(-1), 0.0, NEG_INF).to(torch.float32)
    return dsa_slots.emulate(q_lat, kc, crow, cb, scale, lse)


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None and dsa_slots.nki is not None:
    from .dsa_slots import _pair, _psum, _sb  # the kernel tracer resolves names, not module attributes

    F32, BF16, I32 = nl.float32, nl.bfloat16, nl.int32
    VE = nisa.vector_engine

    def _compact(X, it):
        """Steps 1-4 for every row of iteration it at once, before any row attends: the compacted pool rows (RSI
        [128, rpi] int32), each row's K gather issued into KC[k], and every row's bias row (BRS [rpi, 512] bf16, row
        k on partition k; the bias matmul picks it with the stationary EB[:, k, :])."""
        NCH, rpi, R = X["NCH"], X["rpi"], X["R"]
        PX, PCM, PNV = X["psum_c"]
        W = rpi * NCH * 4
        # 1. the iteration's slot tables: (hi, mid, lo, nv) of row k's slot ch 128 + p at [p, k, ch, :]
        cp = X["CPT"]
        nisa.dma_copy(dst=cp, src=X["cparts"].ap(pattern=[[W, 128], [1, W]], offset=0, scalar_offset=it,
                                                 indirect_dim=0))
        # 2. live slots, their inclusive prefix over a partition's chunks, the partitions' exclusive prefix, places
        m = X["M"]
        nisa.tensor_scalar(dst=m, data=cp[:, :, :, 3], op0=nl.greater, operand0=0.0)
        incl = X["INCL"]
        nisa.tensor_copy(dst=incl[:, :, 0], src=m[:, :, 0])
        for ch in range(1, NCH):
            nisa.tensor_tensor(dst=incl[:, :, ch], data1=incl[:, :, ch - 1], data2=m[:, :, ch], op=nl.add)
        cnb = X["CNB"]
        nisa.tensor_copy(dst=cnb, src=incl[:, :, NCH - 1])
        nisa.nc_matmul(dst=PX, stationary=X["UP"], moving=cnb, accumulate=False)  # every row's in one matmul
        pxs = X["PXS"]
        nisa.tensor_copy(dst=pxs, src=PX)
        pl_ = X["PL"]  # place = excl + incl - m for a live slot, excl + incl + 4096 - 4097 m in general
        nisa.scalar_tensor_tensor(dst=pl_, data=m, op0=nl.multiply, operand0=-4097.0, op1=nl.add, operand1=incl)
        for k in range(rpi):
            nisa.tensor_scalar(dst=pl_[:, k, :], data=pl_[:, k, :], op0=nl.add, operand0=pxs[:, k:k + 1], op1=nl.add,
                               operand1=4096.0)
        # 3. per (row, chunk) the one-hot and its partial of the compacted (hi, mid, lo, nv), each into its own PSUM
        # region with accumulate=False (no accumulation chains: several chains in one PSUM tensor lose partials on
        # trn1), then the chunks summed on the vector engine (one nonzero term per entry: exact)
        oh = X["OH"]
        for k in range(rpi):
            for ch in range(NCH):
                j = k * NCH + ch
                o_ = oh[:, j % 4, :]
                nisa.tensor_scalar(dst=o_, data=X["IOTA"], op0=nl.equal, operand0=pl_[:, k, ch:ch + 1])
                nisa.nc_matmul(dst=PCM[:, k, ch, :], stationary=o_, moving=cp[:, k, ch, :], accumulate=False)
        cm = X["CM"]
        nisa.tensor_copy(dst=cm, src=PCM)
        cs = X["CS"]
        nisa.tensor_tensor(dst=cs, data1=cm[:, :, 0, :], data2=cm[:, :, 1, :], op=nl.add)
        for ch in range(2, NCH):
            nisa.tensor_tensor(dst=cs, data1=cs, data2=cm[:, :, ch, :], op=nl.add)
        crf = X["CRF"]
        nisa.scalar_tensor_tensor(dst=crf, data=cs[:, :, 0], op0=nl.multiply, operand0=256.0, op1=nl.add,
                                  operand1=cs[:, :, 1])
        nisa.scalar_tensor_tensor(dst=crf, data=crf, op0=nl.multiply, operand0=256.0, op1=nl.add, operand1=cs[:, :, 2])
        rs = X["RSI"]
        nisa.tensor_copy(dst=rs, src=crf)
        for k in range(rpi):  # every row's gather in flight before the first row attends
            _gather(X, k)
        # 4. nv^T of every row by one PE transpose, the bias rows (0 visible, -1 masked)
        nvb = X["NVB"]
        nisa.tensor_copy(dst=nvb, src=cs[:, :, 3])
        nisa.nc_matmul(dst=PNV, stationary=nvb, moving=X["IB"], accumulate=False)
        brs = X["BRS"]
        for t in range(KP):
            nisa.tensor_scalar(dst=brs[:, t * 128:(t + 1) * 128], data=PNV, op0=nl.greater, operand0=float(t),
                               op1=nl.subtract, operand1=1.0)

    def _gather(X, k):
        """Row k's compacted K rows into KC[k]: one indirect DMA of its 128 pool rows (the offsets in RSI's column k)."""
        R, rpi = X["R"], X["rpi"]
        nisa.dma_copy(dst=X["KC"][k], src=X["kc"].ap(pattern=[[KP * R, 128], [1, KP * R]], offset=0,
                                                     vector_offset=X["RSI"].ap(pattern=[[rpi, 128], [1, 1]], offset=k),
                                                     indirect_dim=0))

    def _attend_c(X, it, k):
        """Step 5 for row (it, k): dsa_slots' row (qkw form) over its compacted chunk KC[k]."""
        H, R, LC = X["H"], X["R"], X["LC"]
        x = k % 2
        IB = X["IB"]
        PT, PS, PP, PO, PQ = X["psum"]
        T = KP * C
        Kv = X["KC"][k]
        qr = X["QR"][x]
        nisa.dma_copy(dst=qr, src=X["q_lat"].ap(pattern=[[R, H], [1, R]], offset=k * H * R, scalar_offset=it,
                                                indirect_dim=0))
        for lc in range(LC):
            nisa.nc_matmul(dst=PQ[:, lc, :], stationary=qr[:, lc * 128:(lc + 1) * 128], moving=IB[0:H, 0:H],
                           accumulate=False)
        qt = X["QT"][x]
        nisa.tensor_copy(dst=qt, src=PQ, engine=VE)
        kta = X["KTAC"][x]
        for i in range(KP):  # block i = token i of every compacted slot
            pt = PT[i % 2]
            for lc in range(LC):
                c0 = i * R + lc * 128
                nisa.nc_matmul(dst=pt[:, lc, :], stationary=Kv[:, c0:c0 + 128], moving=IB, accumulate=False)
            if i % 2 == 0:
                nisa.tensor_copy(dst=kta[:, :, i * 128:(i + 1) * 128], src=pt, engine=VE)
            else:
                nisa.activation(dst=kta[:, :, i * 128:(i + 1) * 128], op=nl.copy, data=pt)
        ps = PS[x]
        for lc in range(LC):
            nisa.nc_matmul(dst=ps, stationary=qt[:, lc, :], moving=kta[:, lc, 0:T], accumulate=True if lc > 0 else False)
        nisa.nc_matmul(dst=ps, stationary=X["EB"][:, k, :], moving=X["BRS"], accumulate=True)  # + row k's bias / scale
        S = X["SC"][x]
        nisa.activation(dst=S, op=nl.copy, data=ps, scale=X["scale"])
        mx, nmx, den, rden = X["MX"][x], X["NMX"][x], X["DEN"][x], X["RDEN"][x]
        nisa.tensor_reduce(dst=mx, op=nl.maximum, data=S, axis=1)
        nisa.tensor_scalar(dst=nmx, data=mx, op0=nl.multiply, operand0=-1.0, engine=VE)
        P = X["PC"][x]
        nisa.activation(dst=P, op=nl.exp, data=S, bias=nmx, scale=1.0, reduce_op=nl.add, reduce_res=den,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        g = 512 // H
        for i in range(KP):
            j = i * 128
            pp = PP[:, (i % g) * H:(i % g + 1) * H]
            nisa.nc_matmul(dst=pp, stationary=P[:, j:j + 128], moving=IB[0:H, 0:H], accumulate=False)
            pT = X["PTb"][i % 2]
            nisa.activation(dst=pT, op=nl.copy, data=pp)
            nisa.nc_matmul(dst=PO, stationary=pT, moving=Kv[:, i * R:(i + 1) * R], accumulate=(i > 0))
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

    @nki.jit
    def kiln_dsa_slots_c_kernel(q_lat, kc, cparts, identb, n_c, scale: float, fp8: int, want_lse: int, rev: int):
        """q_lat bf16 [NI, RPI, H, R]; kc [cache rows / KP, KP R] the latent cache's pool rows (fp8 = 1: trn1's e4m3,
        see dsa_decode; else bf16); cparts bf16 [NI, 128, RPI, NCH, 4] (slot ch 128 + p of row k of iteration i at
        [i, p, k, ch, :]: its pool row's digits hi, mid, lo and its live tokens nv); identb bf16 [128, 128]; n_c int32
        [1, 1] the iterations to run (NI, or 0 when the caller's fallback takes the call). Returns o fp32 [NI, RPI, H,
        R] (and lse fp32 [NI, RPI, H, 1]); rows of iterations not run are unwritten.

        ONE device loop: the same kernel with a second loop (dsa_slots' row over all slots, for the over-capacity
        calls) failed neuronx-cc 2.27 inside the 45-layer CP-96 decode graph on every rank ([NCC_INLA001] "... is
        overlapping with must-pinned memloc DynamicDMAScratchLoc"), also with a static first loop and with the
        gathers' offsets DMA-written, while compiling in small graphs; without it the graph compiles (q/dc2-dbg,
        2026-10-06). So the fallback is the caller's (attend: kernels/dsa_slots_n.py, a second call)."""
        NI, rpi, H, R = q_lat.shape
        NCH = cparts.shape[3]
        LC = R // 128
        TC = KP * C
        o = nl.ndarray((NI, rpi, H, R), dtype=F32, buffer=nl.shared_hbm)
        lse = None
        if want_lse:
            lse = nl.ndarray((NI, rpi, H, 1), dtype=F32, buffer=nl.shared_hbm)
        KD = kc.dtype
        if fp8:
            KD = nl.float8_e4m3
        X = dict(H=H, R=R, LC=LC, NCH=NCH, rpi=rpi, q_lat=q_lat, kc=kc, o=o, lse=lse, want_lse=want_lse,
                 cparts=cparts)
        IB = _sb((128, 128), BF16)
        nisa.dma_copy(dst=IB, src=identb)
        X["IB"] = IB
        X["scale"] = scale
        X["KC"] = []  # each row's compacted K rows (no list comprehensions: the kernel tracer refuses them)
        for _ in range(rpi):
            X["KC"].append(_sb((128, KP * R), KD))
        X["KTAC"] = _pair((128, LC, TC), BF16)
        X["SC"] = _pair((H, TC), F32)
        X["PC"] = _pair((H, TC), BF16)
        X["QR"] = _pair((H, R), BF16)
        X["QT"] = _pair((128, LC, H), BF16)
        X["OB"] = _pair((H, R), F32)
        X["MX"] = _pair((H, 1), F32)
        X["NMX"] = _pair((H, 1), F32)
        X["DEN"] = _pair((H, 1), F32)
        X["RDEN"] = _pair((H, 1), F32)
        X["LS"] = _pair((H, 1), F32)
        X["PTb"] = _pair((128, H), BF16)
        X["BRS"] = _sb((rpi, TC), BF16)  # 0 / -1, exact
        X["CPT"] = _sb((128, rpi, NCH, 4), BF16)
        X["M"] = _sb((128, rpi, NCH))
        X["INCL"] = _sb((128, rpi, NCH))
        X["CNB"] = _sb((128, rpi), BF16)
        X["PXS"] = _sb((128, rpi))
        X["PL"] = _sb((128, rpi, NCH))
        X["OH"] = _sb((128, 4, 128), BF16)
        X["CM"] = _sb((128, rpi, NCH, 4))
        X["CS"] = _sb((128, rpi, 4))
        X["CRF"] = _sb((128, rpi))
        X["RSI"] = _sb((128, rpi), I32)
        X["NVB"] = _sb((128, rpi), BF16)
        ji = _sb((128, 128), I32)
        nisa.iota(dst=ji, pattern=[[1, 128]], offset=0, channel_multiplier=0)
        io = _sb((128, 128))  # io[p, c] = c
        nisa.tensor_copy(dst=io, src=ji)
        X["IOTA"] = io
        pi = _sb((128, 1), I32)
        nisa.iota(dst=pi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        pf = _sb((128, 1))
        nisa.tensor_copy(dst=pf, src=pi)
        dd = _sb((128, 128))  # c - p
        nisa.tensor_scalar(dst=dd, data=io, op0=nl.subtract, operand0=pf)
        up = _sb((128, 128), BF16)  # [p < c]: the exclusive prefix over the partitions
        nisa.tensor_scalar(dst=up, data=dd, op0=nl.greater, operand0=0.0)
        X["UP"] = up
        idf = _sb((128, 128))  # [p == c]
        nisa.tensor_scalar(dst=idf, data=dd, op0=nl.equal, operand0=0.0)
        oneh = _sb((rpi, H))
        nisa.memset(dst=oneh, value=1.0)
        eb = _sb((rpi, rpi, H), BF16)  # eb[p, k, h] = [p == k] 1e30 / scale: row k's bias row times NEG_INF / scale
        for k in range(rpi):
            nisa.tensor_scalar(dst=eb[:, k, :], data=oneh, op0=nl.multiply, operand0=idf[0:rpi, k:k + 1],
                               op1=nl.multiply, operand1=-NEG_INF / scale)
        X["EB"] = eb

        rc = nisa.register_alloc()
        cc = _sb((1, 1), I32)
        nisa.dma_copy(dst=cc, src=n_c)
        nisa.register_load(rc, cc)

        def body_c(it):
            X["psum"] = _psum(H, LC)
            X["psum_c"] = (nl.ndarray((128, rpi), dtype=F32, buffer=nl.psum),
                           nl.ndarray((128, rpi, NCH, 4), dtype=F32, buffer=nl.psum),
                           nl.ndarray((rpi, 128), dtype=F32, buffer=nl.psum))
            _compact(X, it)
            for k in range(rpi):
                _attend_c(X, it, k)

        nl.fori_loop(0, rc, body_c)
        if want_lse:
            return o, lse
        return o
else:
    kiln_dsa_slots_c_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source and dsa_slots.REV (its tile and PSUM helpers are dsa_slots')."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_dsa_slots_c_kernel = None", a)
    return zlib.crc32(src[a:b].encode() + str(dsa_slots.REV).encode())


REV = _kernel_rev()


def kernel_args(q_lat: torch.Tensor, kc: torch.Tensor, rows: torch.Tensor, bias: torch.Tensor, scale: float,
                lse: bool = False) -> tuple[dict, int, int, torch.Tensor]:
    """(the kernel's keyword arguments, N, Np, over): attend()'s inputs in the kernel's layout (rows padded to RPI);
    over fp32 [1, 1] is 1 when some row has more than C live slots (the kernel then runs no iteration: n_c = 0)."""
    kc2 = kc.reshape(kc.shape[0], -1)
    N, H, R = q_lat.shape
    NS = rows.shape[1]
    if NS % 128:
        raise ValueError(f"dsa_slots_c: {NS} slots per row is not a multiple of 128")
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
    kcp = kc2.reshape(kc2.shape[0] // KP, KP * R)
    NI = Np // rpi
    # the slot table: the pool row's 8-bit digits (exact in fp32 below 2^24, each exact in bf16) and the live tokens
    nv = live_tokens(bs)  # [Np, NS]
    rf = rw.to(torch.float32)
    hi = torch.floor(rf * (1.0 / 65536))
    rem = rf - hi * 65536
    mid = torch.floor(rem * (1.0 / 256))
    lo = rem - mid * 256
    cparts = torch.stack([hi, mid, lo, nv], dim=-1).to(torch.bfloat16)  # [Np, NS, 4]
    cparts = cparts.reshape(NI, rpi, NCH, 128, 4).permute(0, 3, 1, 2, 4).contiguous()  # [NI, 128, rpi, NCH, 4]
    # over: some row has more than C live slots (an exact fp32 count; a full amax lowers to the wrong shape)
    cnt = (nv > torch.zeros_like(nv)).to(torch.float32).sum(-1).view(1, Np)
    over = (cnt > torch.full_like(cnt, float(C))).to(torch.float32).amax(dim=1, keepdim=True)  # [1, 1]
    n_c = ((1.0 - over) * NI).to(torch.int32)
    eye = torch.eye(128, device=q_lat.device).to(torch.bfloat16)
    kw = dict(q_lat=q.reshape(NI, rpi, H, R).contiguous(), kc=kcp, cparts=cparts, identb=eye, n_c=n_c,
              scale=float(scale), fp8=int(fp8), want_lse=int(lse), rev=REV)
    return kw, N, Np, over


def attend(q_lat: torch.Tensor, kc: torch.Tensor, rows: torch.Tensor, bias: torch.Tensor, scale: float,
           lse: bool = False):
    """dsa_slots.attend's arguments and outputs (o [N, H, R] fp32, with lse [N, H]), each row over its live slots
    compacted in the kernel. A call with a row over C live slots runs kernels/dsa_slots_n.py over every row instead
    (its row count is N then, 0 otherwise, so one of the two kernels computes; the other leaves its output unwritten
    and is not selected). emulate() off the device."""
    kc2 = kc.reshape(kc.shape[0], -1)
    if q_lat.device.type == "cpu":
        return emulate(q_lat, kc2, rows, bias, scale, lse)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform
    from . import dsa_slots_n

    if kiln_dsa_slots_c_kernel is None:
        raise RuntimeError("the NKI DSA slots kernel needs the nki package (the Neuron venv)")
    if not dsa_slots.QKW:
        raise ValueError("dsa_slots_c: the compacted form needs KILN_DSA_SLOTS_QKW=1 (the default)")
    kw, N, Np, over = kernel_args(q_lat, kc2, rows, bias, scale, lse)
    H, R = q_lat.shape[1], q_lat.shape[2]
    out = wrap_nki(kiln_dsa_slots_c_kernel)[platform.nki_grid()](**kw)
    n_f = (over.view(1) * N).to(torch.int64)
    full = dsa_slots_n.attend(q_lat, kc2, rows, bias, scale, n_f, lse=lse)
    take = (over > torch.zeros_like(over)).view(1, 1, 1)
    if lse:
        o, ls = out
        o, ls = o.reshape(Np, H, R)[:N], ls.reshape(Np, H)[:N]
        return torch.where(take, full[0], o), torch.where(take.view(1, 1), full[1], ls)
    return torch.where(take, full, out.reshape(Np, H, R)[:N])
