"""kernels/dsa_fused.py with the work past the chunk's causal horizon skipped (KILN_DSA_FUSED_CAUSAL=1): the same
pooled indexer scores, exact top-k selection and attention over the selected pools, computed only over the key blocks
some query of the call can see.

Why: kernels/dsa_fused.py attends EVERY 1024-key block of the page bucket for every query tile and selects over every
pool of it, whatever the chunk's position: one trn1 core at GLM-5.3-Flash's attention-TP-8 rank shape (1024 rows, 8
heads, latent 512, 32 x 128 indexer, keep 512) takes 4.89 ms at 8,448 keys with the chunk at 0, 3,072 or 7,168, and 2.50
ms at 4,096 keys (docs/neuron-notes.md "Phase 2, lever 1 measured"). An 8k prompt's 1024-row chunks see on average
~4.6k of the 8.4k keys.

How. The call's last query position m (the max of pos) fixes how many key blocks of KU = 1024 keys anything in it can
see: nb = #{j : KU j <= m}. The kernel holds one copy of dsa_fused's whole static body per block count k of a ladder
(every count 1 .. NB by default; KILN_DSA_FUSED_CAUSAL_LADDER), each inside a device loop of trip count 0 or 1
(nl.fori_loop over a register; NKI has no conditional), and the caller passes a one-hot over the ladder: the smallest
k >= nb runs. Variant k is dsa_fused at L_k = min(L, KU k) keys and P_k = L_k / 4 pools, phase 0, the first tile's
selection and every tile's attention with the next tile's selection between its steps, all static inside the region,
so nothing of dsa_fused's software pipeline changes. One region per variant, not one per (tile, block): an NKI device
loop puts a full all-engine barrier at its entry and every iteration (~20-37 us measured on trn1, docs/neuron-notes.md
"NKI device loops put a full all-engine barrier"), and compile time grows with the region count.

Why it is exact (bit for bit against kernels/dsa_fused.py, not only within rounding):
- Attention: a key block past every query's position has every key masked (mask NEG_INF), so its block max is far
  below the running max m, alpha = exp(0) = 1, P = 0, and acc = 1 acc + 0, l = relu(1 l + 0) leave the state unchanged
  in fp32. The visible blocks are a prefix and run first, with the same instructions in the same order per (tile,
  head) unit.
- Selection: the pools from P_k on are all past every query (their first key KU k > m), so they are non-candidates
  scoring index + NEG_INF = NEG_INF exactly (|index| << ulp(1e30)). Over P_k > keep pools the radix finds the same
  keep-th largest t (the dropped entries are all NEG_INF, the smallest values; the sign step counts only scores >= 0,
  and k2 = P_k + 1 - keep names the same element of -s), the same `above` set, and the same tie limit (when t is NEG_INF
  the first `room` tied entries lie inside [0, P_k): P_k - c >= keep - c of them do). With P_k <= keep every candidate
  is selected and every non-candidate is the tail, so the selection is all ones on [0, P_k) and the variant writes it
  directly (the selection costs nothing then). Positions past a query are masked by the position test either way.
tests/test_dsa_fused_c.py checks both in nki.simulate against kernels/dsa_fused.py's kernel (torch.equal) and emulate().

The kernel is a separate module so that kernels/dsa_fused.py's source, its REV and the key of every graph that runs it
are unchanged; this module's REV is its kernel source and dsa_fused.REV (its helpers are dsa_fused's). trn1 only (grid
1): a grid-2 call (trn2's LNC split) keeps kernels/dsa_fused.py.
"""

from __future__ import annotations

import os

import torch

from . import dsa_fused

KU = dsa_fused.KU  # keys per attention unit and per block of the ladder
NEG_INF = dsa_fused.NEG_INF


# The default ON configuration (2026-10-08, owner's decision on the coordinator's bar): this module's causal kernel
# (ladder 2,4,6,8) for the trn2 G1 8192-row whole-prefill calls ONLY: trn2 at LNC=2 (grid 2), 2048 rows per call (one
# DP-attention group's share of an 8192-row call), the 264-page bucket (8448 keys), KILN_PREFILL_WHOLE=1. Gated there
# (trn2.48xlarge kiln-t2-cb2, SDK 2.32, GLM-5.3-Flash real weights, the E1B-p6PW config; docs/neuron-notes.md "The
# final trn2 round", s3 logs/kiln-t2-cb2/): bit for bit in the graph, check_mixed 32 / 32 equal, teacher-forced
# |dlogprob| 0.0000 over 2016 decode calls and 32 prefill chunks (dsc-mixed-compare-t2PW0-PWC.txt); wikitext with the
# flag set (1024 rows per group) -0.5518 = -0.5518, |dlogprob| 0.0000, greedy agreement 1.0000
# (dsc-ppl-compare-PPL0-PPLC.txt); conc 128 real text 303.9 -> 312.4 out tok/s (+2.8%), prefill call 0.517 -> 0.495 s
# (-22 ms; 20261008T041947Z-dsc-sweep-PWC.log against 20261008T042153Z-t2-ctl-dsa1.log, a control ten minutes earlier:
# its same-time rerun failed at start). Caveat: the speed-up is that one run pair; the exactness is the property. trn1,
# the long-context DSA path, other trn2 shapes and every configuration whose graphs were not compiled keep dsa_fused
# (their keys are unchanged). KILN_DSA_FUSED_CAUSAL=0 / 1 overrides it either way. kernels/dsa_split.py's split
# selection stays opt-in (KILN_DSA_SPLIT_SELECT=1): not bit for bit in the same graphs (check_mixed 28 / 32).
DEFAULT_ROWS, DEFAULT_KEYS = 2048, 8448
DEFAULT_LADDER_TRN2 = (2, 4, 6, 8)  # an 8k prompt's 2048-row chunks see exactly 2 / 4 / 6 / 8 key blocks


def _trn2_grid2() -> bool:
    from .. import platform

    try:
        t = platform.target()
        return platform.nki_grid() == 2 and t is not None and platform.family_of(t) == "trn2"
    except RuntimeError:  # a host whose runtime was never configured
        return False


def default_on(C, L) -> bool:
    """Whether a call of C query rows over L keys is the gated default configuration (above)."""
    return (C == DEFAULT_ROWS and L == DEFAULT_KEYS and os.environ.get("KILN_PREFILL_WHOLE", "0") == "1"
            and _trn2_grid2())


def ladder(NB: int) -> tuple[int, ...]:
    """The block counts the kernel holds a variant for: KILN_DSA_FUSED_CAUSAL_LADDER (comma-separated counts, values
    above NB dropped), else on trn2 at LNC=2 DEFAULT_LADDER_TRN2 (the gated ladder), else every count 1 .. NB; NB (the
    whole bucket) is always one of them."""
    env = os.environ.get("KILN_DSA_FUSED_CAUSAL_LADDER")
    if env:
        ks = {int(x) for x in env.split(",") if 0 < int(x) <= NB}
    elif _trn2_grid2():
        ks = {k for k in DEFAULT_LADDER_TRN2 if k <= NB}
    else:
        ks = set(range(1, NB + 1))
    return tuple(sorted(ks | {NB}))


def _ladder_bits(ks: tuple[int, ...]) -> int:
    """The ladder as the kernel's static argument: bit k - 1 set for each count k."""
    b = 0
    for k in ks:
        b |= 1 << (k - 1)
    return b


def _ladder_of(bits: int) -> list:
    out = []
    k = 1
    while (1 << (k - 1)) <= bits:
        if bits & (1 << (k - 1)):
            out.append(k)
        k += 1
    return out


def emulate(qI, w, pk, pos, q_lat, kc, keep: int, scale_i: float, scale_a: float, kp: int = 4) -> torch.Tensor:
    """The arithmetic in torch (CPU): kernels/dsa_fused.py's, which this kernel computes exactly."""
    return dsa_fused.emulate(qI, w, pk, pos, q_lat, kc, keep, scale_i, scale_a, kp)


def variant_onehot(pos: torch.Tensor, L: int, ks: tuple[int, ...]) -> torch.Tensor:
    """int32 [1, len(ks)]: 1 at the smallest ladder count k >= nb, nb = the key blocks of KU keys holding a key at or
    before the call's last position (>= 1: block 0 holds position 0). Comparisons only (in-graph integer division is
    inexact on trn1: docs/neuron-notes.md)."""
    NB = -(-L // KU)
    # reductions over a dim with keepdim: a whole-tensor amax() then .view(1) failed LNL's lowering on trn1 ("Check
    # failed: total_element_count == ... (1024 vs. 1)", 2026-10-07)
    starts = torch.arange(NB, device=pos.device, dtype=torch.int32).view(1, NB) * KU
    m = pos.to(torch.int32).view(1, -1).amax(-1, keepdim=True)  # [1, 1]
    nb = (starts <= m).to(torch.int32).sum(-1, keepdim=True)  # [1, 1]
    # one [1, 1] compare per ladder count against Python ints (a torch.tensor of the counts is not traceable on the
    # capture's meta device: "Please convert all Tensors to FakeTensors first")
    prev = (0,) + tuple(ks[:-1])
    return torch.cat([((nb <= k) & (nb > p)).to(torch.int32) for k, p in zip(ks, prev)], dim=1)


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None and dsa_fused.nki is not None:
    # the kernel tracer resolves names, not module attributes (docs/neuron-notes.md "What a device loop may hold")
    from .dsa_fused import KR, NR, _attend, _copy, _ps, _ring, _sb, _sel_begin, _sel_consts, _sel_nsteps, _sel_step

    F32, BF16, I32 = nl.float32, nl.bfloat16, nl.int32
    VE = nisa.vector_engine

    def _pos_begin(Y, t: int):
        """Query tile t's positions and pos - e (the masks' inputs): _sel_begin without the selection's inputs."""
        nisa.dma_copy(dst=Y["posS"], src=Y["posf"].ap(pattern=[[1, 128], [1, 1]], offset=t * 128))
        for e in range(4):
            nisa.tensor_scalar(dst=Y["posm"][:, e:e + 1], data=Y["posS"], op0=nl.add, operand0=float(-e), engine=VE)

    def _variant(X, Y, k: int, t_lo: int, t_hi: int):
        """dsa_fused's static body at k key blocks (L_k = min(L, KU k) keys, P_k = L_k / 4 pools), its PSUM allocated
        here (a PSUM tensor referenced by two device-loop regions fails the backend: NCC_IBIR092)."""
        L, R, RC, keep = X["L"], X["R"], X["RC"], Y["keep"]
        Lk = min(L, KU * k)
        Pk = Lk // 4
        sel = Pk > keep
        Sr, Tr, Or = [], [], []
        for _ in range(2):
            sb_ = []
            for _ in range(2):
                sb_.append(_ps())
            Sr.append(sb_)
        tb_ = []
        for _ in range(2):
            tb_.append(_ps())
        Tr.append(tb_)
        Or.append(_ps())
        # X and Y are updated in place (the kernel tracer cannot iterate a dict to copy it); every field a variant
        # changes is set here, the full-width tiles stay under their *_f names
        Xk = X
        Yk = Y
        Xk["Sr"] = Sr
        Xk["Tr"] = Tr
        Xk["Or"] = Or
        Yk["P"] = Pk
        Yk["NC"] = (Pk + 511) // 512
        Yk["acc"] = Y["acc_f"][:, 0:Pk]
        Yk["s2"] = Y["s2_f"][:, 0:Pk]
        Yk["scr"] = Y["scr_f"][:, 0:Pk]
        Yk["tj"] = Y["tj_f"][:, 0:Pk]
        Yk["wio"] = Y["wio_f"][:, 0:Pk]
        Yk["pio"] = Y["pio_f"][:, 0:Pk]
        s01 = []
        for s in range(2):
            s01.append(Y["sel01_f"][s][:, 0:Pk])
        Yk["sel01"] = s01
        if sel:
            Yk["sp"] = _ps()
        # phase 0: K^T of blocks 0 .. k - 1 into the block-major scratch (dsa_fused's phase 0 over k blocks)
        TO = []
        for x in range(2):
            TO.append(Sr[0][x])
        for x in range(2):
            TO.append(Sr[1][x])
        KTb, Kr, kc, kth = X["KTb"], X["Kr"], X["kc"], X["kth"]
        IB = X["IB"]
        for j in range(k):
            wj = min(KU, L - j * KU)
            kin = Kr[j % KR]
            kts = KTb[j % KR]
            nt = wj // 128
            nisa.dma_copy(dst=kin[:, 0:nt, :], src=kc.ap(pattern=[[R, 128], [128 * R, nt], [1, R]], offset=j * KU * R))
            for c in range(RC):
                for g4 in range((nt + 3) // 4):
                    m4 = min(4, nt - 4 * g4)
                    pt = TO[(j * RC * 2 + c * 2 + g4) % 4]
                    for i in range(m4):
                        nisa.nc_matmul(dst=pt[:, i * 128:(i + 1) * 128], stationary=kin[:, 4 * g4 + i, c * 128:(c + 1) * 128],
                                       moving=IB, accumulate=False)
                    _copy(kts[:, c, g4 * 512:g4 * 512 + m4 * 128], pt[:, 0:m4 * 128], c + g4)
            nisa.dma_copy(dst=kth.ap(pattern=[[KU, 128], [128 * KU, RC], [1, wj]], offset=j * RC * 128 * KU),
                          src=kts[:, :, 0:wj])
        # tile 0's selection alone (or, with P_k <= keep, all ones), then tile t's attention with tile t + 1's selection
        Yk["slot"] = t_lo % 2
        if sel:
            _sel_begin(Yk, t_lo)
            for i in range(_sel_nsteps(Yk)):
                _sel_step(Yk, i)
        else:
            _pos_begin(Yk, t_lo)
            nisa.memset(dst=s01[t_lo % 2], value=1.0)
        nisa.tensor_copy(dst=X["posmA"][t_lo % 2], src=Yk["posm"], engine=VE)
        for t in range(t_lo, t_hi):
            nxt = t + 1 if t + 1 < t_hi else -1
            if nxt >= 0:
                Yk["slot"] = nxt % 2
                if sel:
                    _sel_begin(Yk, nxt)
                else:
                    _pos_begin(Yk, nxt)
                    nisa.memset(dst=s01[nxt % 2], value=1.0)
            _attend(Xk, Yk, t, t % 2, k, nxt if sel else -1)
            if nxt >= 0:
                nisa.tensor_copy(dst=X["posmA"][nxt % 2], src=Yk["posm"], engine=VE)

    @nki.jit
    def kiln_dsa_fused_c_kernel(qT, w, pkT, posf, q_lat, kc, identb, vt, keep: int, nbits: int, scale_i: float,
                                scale_a: float, lad: int, rev: int, spl: int = 0):
        """kiln_dsa_fused_kernel's arguments (qT bf16 [Hi, D, C], w fp32 [C, Hi], pkT bf16 [D, P], posf fp32 [C],
        q_lat bf16 [C, H, R], kc bf16 [L = 4 P, R], identb bf16 [128, 128]; keep < P, 2^nbits > P, C and L multiples
        of 128) plus vt int32 [1, NV], the one-hot over the ladder (variant_onehot), and lad, the ladder's bits
        (_ladder_bits). Returns o fp32 [C, H, R], equal to kiln_dsa_fused_kernel's. spl 1 at LNC=2 (trn2, grid 2: the
        two physical cores), as dsa_fused's: each program takes half of the query tiles, both build the whole K^T
        scratch (identical bytes) and meet at a core barrier on o. Both programs read the same one-hot, so they run the
        same device loops with the same trip counts (the two cores' functions must have the same basic blocks:
        docs/neuron-notes.md, NCC_IXGM002)."""
        Hi, D, C = qT.shape
        P = pkT.shape[1]
        _, H, R = q_lat.shape
        L = kc.shape[0]
        RC = R // 128
        NB = (L + KU - 1) // KU
        o = nl.ndarray((C, H, R), dtype=F32, buffer=nl.shared_hbm)
        kth = nl.ndarray((NB, RC, 128, KU), dtype=BF16, buffer=nl.shared_hbm)  # K^T, block-major
        IB = _sb((128, 128), BF16)
        nisa.dma_copy(dst=IB, src=identb)
        X = dict(H=H, RC=RC, R=R, L=L, scale=scale_a, IB=IB, kth=kth, kc=kc, q_lat=q_lat, o=o, dbg=0, q0=0)
        X["KTb"] = _ring(KR, (128, RC, KU), BF16)
        X["Kr"] = _ring(KR, (128, KU // 128, R), BF16)
        X["Mk"] = _ring(KR, (128, KU), BF16)
        X["att01"] = _sb((128, KU))
        X["BM2"] = None
        X["QT"] = _sb((128, H, RC * 128), BF16)
        X["Qin"] = _sb((128, H, R), BF16)
        X["ACC"] = _ring(H, (128, R))
        X["Lr"] = _ring(H, (128, 1))
        X["M"] = _ring(2 * H, (128, 1))
        X["Pr"] = _ring(3, (128, KU), BF16)
        X["PT"] = _ring(3, (128, KU), BF16)
        X["BM"] = _ring(NR, (128, 2))
        X["AL"] = _ring(NR, (128, 1))
        X["NB"] = _ring(NR, (128, 1))
        X["RS"] = _ring(NR, (128, 1))
        X["OB"] = _ring(2, (128, R))
        X["RL"] = _sb((128, 1))
        Y = dict(C=C, Hi=Hi, D=D, P=P, NC=(P + 511) // 512, keep=keep, nbits=nbits, scale_i=scale_i, qT=qT, w=w,
                 posf=posf.reshape((C, 1)), dscore=None, first=0)
        pk_s = _sb((D, P), BF16)
        nisa.dma_copy(dst=pk_s, src=pkT)
        Y["pkT"] = pk_s
        Y["qs"] = _sb((D, Hi, 128), BF16)
        Y["ws"] = _sb((128, Hi))
        Y["posS"] = _sb((128, 1))
        Y["posm"] = _sb((128, 4))
        Y["acc"] = _sb((128, P))
        Y["s2"] = _sb((128, P))
        Y["scr"] = _sb((128, P))
        Y["tj"] = _sb((128, P))
        Y["wio"] = _sb((128, P))
        Y["pio"] = _sb((128, P))
        Y["sel01"] = _ring(2, (128, P), BF16)
        Y["c1"] = _sb((128, 1))
        Y["f"] = _sb((128, 1))
        Y["sg"] = _sb((128, 1))
        Y["k2"] = _sb((128, 1))
        Y["th"] = _sb((128, 1))
        Y["room"] = _sb((128, 1))
        Y["lim"] = _sb((128, 1))
        Y["thr"] = _sb((128, 1))
        Y["incf"] = _sb((128, 1))
        Y["pre"] = _sb((128, 1), I32)
        Y["cand"] = _sb((128, 1), I32)
        Y["inc"] = _sb((128, 1), I32)
        _sel_consts(Y)
        Y["acc_f"] = Y["acc"]
        Y["s2_f"] = Y["s2"]
        Y["scr_f"] = Y["scr"]
        Y["tj_f"] = Y["tj"]
        Y["wio_f"] = Y["wio"]
        Y["pio_f"] = Y["pio"]
        Y["sel01_f"] = Y["sel01"]
        X["posmA"] = _ring(2, (128, 4))
        NQ = C // 128
        # LNC split (spl): program p takes query tiles [p NQ / 2, (p + 1) NQ / 2) (dsa_fused's kernel)
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        sp = spl == 1 and npg == 2
        t_lo, t_hi = (pid * (NQ // 2), (pid + 1) * (NQ // 2)) if sp else (0, NQ)
        ks = []  # the ladder's counts (_ladder_of, inline: the tracer runs the kernel's own code)
        for k in range(1, 33):
            if (lad >> (k - 1)) & 1:
                ks.append(k)
        NV = len(ks)
        vts = _sb((1, NV), I32)
        nisa.dma_copy(dst=vts, src=vt)
        for v in range(NV):
            rv = nisa.register_alloc()
            nisa.register_load(rv, vts.ap(pattern=[[NV, 1], [1, 1]], offset=v))

            def body(_, k=ks[v]):
                _variant(X, Y, k, t_lo, t_hi)

            nl.fori_loop(0, rv, body)
        if npg == 2:  # whatever runs next on either physical core reads the whole o (nki/isa/_lnc.py core_barrier)
            nisa.core_barrier(data=o, cores=(0, 1))
        return o
else:
    kiln_dsa_fused_c_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source and dsa_fused.REV (LNL's compile-cache key does not include NKI kernel
    source: CLAUDE.md)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_dsa_fused_c_kernel = None", a)
    return zlib.crc32(src[a:b].encode() + str(dsa_fused.REV).encode())


REV = _kernel_rev()
_env = os.environ.get("KILN_DSA_FUSED_CAUSAL")
CAUSAL = None if _env in (None, "") else _env == "1"  # None: the gated default (default_on)


def takes(C=None, L=None) -> bool:
    """Whether dsa_fused.attend hands its call (C rows over L keys) to this kernel: KILN_DSA_FUSED_CAUSAL=1 (trn1 at
    grid 1; trn2 at LNC=2, grid 2, splitting the query tiles as dsa_fused does when KILN_LNC_SPLIT names dsa_fused),
    =0 never, unset the gated default configuration only (default_on). Read when a graph is traced. (Until 2026-10-07
    evening it took grid 1 only, so on trn2 the flag changed no graph.)"""
    if CAUSAL is None:
        return C is not None and default_on(C, L)
    return CAUSAL


def attend(qI, w, pk, pos, q_lat, kc, keep: int, scale_i: float, scale_a: float, simulate: bool = False,
           grid: int = 1):
    """dsa_fused.attend's contract (o [C, H, R] fp32; the kernel on a Neuron device, emulate() elsewhere), through the
    causal kernel. simulate: nki.simulate at `grid` (2: both programs of an LNC=2 core, the query tiles split)."""
    kc2 = kc.reshape(kc.shape[0], -1)
    if q_lat.device.type == "cpu" and not simulate:
        return emulate(qI, w, pk, pos, q_lat, kc2, keep, scale_i, scale_a)
    from .. import platform

    if kiln_dsa_fused_c_kernel is None:
        raise RuntimeError("the NKI causal fused DSA kernel needs the nki package (the Neuron venv)")
    H = q_lat.shape[1]
    if H < dsa_fused.MIN_HEADS:  # the latent ring's hazard below 3 heads (dsa_fused.MIN_HEADS)
        q_lat = torch.cat([q_lat, q_lat.new_zeros(q_lat.shape[0], dsa_fused.MIN_HEADS - H, q_lat.shape[2])], dim=1)
    C = q_lat.shape[0]
    P = pk.shape[0]
    L = kc2.shape[0]
    Cp = -(-C // 128) * 128
    if Cp != C:  # whole query tiles; a padded row sits at position 0 and its output is dropped
        pad = Cp - C
        qI = torch.cat([qI, qI.new_zeros(pad, *qI.shape[1:])])
        w = torch.cat([w, w.new_zeros(pad, *w.shape[1:])])
        q_lat = torch.cat([q_lat, q_lat.new_zeros(pad, *q_lat.shape[1:])])
        pos = torch.cat([pos, pos.new_zeros(pad)])
    ks = ladder(-(-L // KU))
    vt = variant_onehot(pos, L, ks)
    eye = torch.eye(128, device=q_lat.device).to(torch.bfloat16)
    args = dict(qT=qI.to(torch.bfloat16).permute(1, 2, 0).contiguous(), w=w.float().contiguous(),
                pkT=pk.to(torch.bfloat16).t().contiguous(), posf=pos.float().contiguous(),
                q_lat=q_lat.to(torch.bfloat16).contiguous(), kc=kc2.to(torch.bfloat16).contiguous(), identb=eye, vt=vt,
                keep=int(keep), nbits=P.bit_length(), scale_i=float(scale_i), scale_a=float(scale_a),
                lad=_ladder_bits(ks), rev=REV)
    if simulate:  # the same arguments as the device call below
        spl = int(grid == 2 and (Cp // 128) % 2 == 0)
        k_ = kiln_dsa_fused_c_kernel[grid] if grid > 1 else kiln_dsa_fused_c_kernel
        o = torch.as_tensor(nki.simulate(k_)(**args, **({"spl": 1} if spl else {})))
    else:
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        spl = dsa_fused.split(Cp)  # LNC=2 with KILN_LNC_SPLIT naming dsa_fused: half the query tiles per core
        o = wrap_nki(kiln_dsa_fused_c_kernel)[platform.nki_grid()](**args, **({"spl": 1} if spl else {}))
    o = o[:C] if Cp != C else o
    return o[:, :H] if H < dsa_fused.MIN_HEADS else o
