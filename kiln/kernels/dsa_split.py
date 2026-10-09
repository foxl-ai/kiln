"""The pooled-DSA prefill chunk with its selection split over the attention group (KILN_DSA_SPLIT_SELECT=1, trn1):
each rank selects pools for 1 / attn_tp of the chunk's queries, the group gathers the 0 / 1 pool selection, and every
rank attends all queries for its heads with an attention-only causal kernel.

Why: GLM-5.3-Flash's indexer has 32 heads in total (index_n_heads), so every rank of an attention group computed the
same scores and the same exact top-k for the same 1024 queries: kernels/dsa_fused.py's selection is ~1.5 ms of its
3.3 ms of vector-engine time at 8448 keys (head sum ~0.77, radix and tie rounds ~0.75; docs/neuron-notes.md "The
attention kernels against their floors"), done 8 times over per group. Split, a rank's selection is one 128-query
tile; the attention kernel is left with the online softmax's block max and rescale on the vector engine (~1.8 us per
(tile, head, 1024-key block) against ~4.3 us of tensor-engine work), so the tensor engine paces it.

The pieces (each exact against kernels/dsa_fused.py's arithmetic, so the whole is bit for bit dsa_fused's):
- select(): kiln_dsa_select_kernel, dsa_fused's selection micro-steps (_sel_begin / _sel_step, the same instructions)
  for the rank's rows over the first P_k pools, P_k from the CALL's last position (dsa_fused_c's variant ladder, every
  rank the same variant): bf16 [R, P], 1 where a query attends a pool (selected, or the tail), 0 elsewhere and from
  P_k on. P_k <= keep selects every pool (dsa_fused_c's argument), written as ones.
- the group gather of the [R, P] tiles (DecoderForCausalLM._sp_group_gather, a zero-padded all-reduce: exact).
- attend(): kiln_dsa_attend_kernel, dsa_fused_c's variant k without the selection: phase 0 over k blocks, then per
  query tile its selection rows and positions by DMA and _attend2, dsa_fused._attend's per-unit arithmetic with no
  selection steps between the attention steps, P K into a ring of two PSUM banks (the bank the selection held), the
  block max and the running max as one tensor_scalar_reduce per 512 keys, and the token mask built by one vector
  instruction (_mask_block2) instead of four strided ones. A block past every query is an exact no-op of the online
  softmax, the pools past P_k are past every query, and max is exact, so the variant's result is dsa_fused's.

The model calls attend_split() from models/mla.py's fused branch when split_takes() (the sequence-parallel group
block, whose ranks hold the group's rows; the chunk a whole number of 128-row tiles per rank). Its own module, so
dsa_fused.py's and dsa_fused_c.py's sources and REVs, and every key that runs them, are unchanged.
"""

from __future__ import annotations

import os

import torch

from . import dsa_fused, dsa_fused_c, dsa_prefill, dsa_topk

KU = dsa_fused.KU
NEG_INF = dsa_fused.NEG_INF
# Streamed key blocks in flight in the attention kernel (K^T, latent rows, mask): 3, so a tile's first block loads while
# the previous tile's last one is still read (dsa_fused's 2 left that load exposed at every tile start; SBUF has the
# 16 KB per partition once the selection's tiles are elsewhere). Also safe below dsa_fused.MIN_HEADS's 3 heads.
KR2 = 3
# Opt-in, off by default everywhere (also in dsa_fused_c's gated configuration): on trn2 G1's 8192-row whole-prefill
# graphs it is not bit for bit (check_mixed 28 / 32, the 4 that differ the 700-token prompts; wikitext |dlogprob| mean
# 0.0561), although each kernel is torch.equal to dsa_fused alone on one core, a 700-real-row chunk with padding included
# (docs/neuron-notes.md "The final trn2 round"). Held until that is found.
SPLIT = os.environ.get("KILN_DSA_SPLIT_SELECT", "0") == "1"
# KILN_DSA_SPLIT_PT: dma (default) builds P^T by an SBUF -> SBUF DMA transpose where the core has one (NeuronCore-v3+,
# trn2; nkilib attention_cte does the same with its exp tile), pe by tensor-engine transposes and a scalar-engine copy
# (the only way on trn1). Pure data movement either way, so the same values.
PT_DMA = int(os.environ.get("KILN_DSA_SPLIT_PT", "dma") == "dma")
# KILN_DSA_SPLIT_MASK: vector (default) adds the token mask on the vector engine, tensor by an identity matmul (as
# kernels/dsa_fused.py). Both give the same scores; which is faster is the device's to say (not yet measured).
MASK_DVE = int(os.environ.get("KILN_DSA_SPLIT_MASK", "vector") == "vector")


def sel_ladder(NB: int) -> tuple[int, ...]:
    """The selection kernel's block counts (KILN_DSA_SPLIT_SEL_LADDER, default 2, 4, 6 and NB): coarser than the
    attention's, because a skipped device-loop region costs ~10-30 us (docs/neuron-notes.md "Phase 2, lever 2
    measured") against a one-tile selection of ~0.1 ms, and selecting over more pools than a call can see is exact."""
    env = os.environ.get("KILN_DSA_SPLIT_SEL_LADDER", "2,4,6")
    return tuple(sorted({int(x) for x in env.split(",") if 0 < int(x) <= NB} | {NB}))


def emulate_select(qI, w, pk, pos, keep: int, scale_i: float) -> torch.Tensor:
    """bf16 [R, P] 0 / 1: the pools each of R queries attends (dsa_fused.emulate's pool selection, the tail included)."""
    P = pk.shape[0]
    R = qI.shape[0]
    last = torch.arange(P) * 4 + 3
    cand = torch.where(last.view(1, P) <= pos.view(R, 1).long(), 0.0, NEG_INF)
    sel = dsa_topk.emulate(dsa_topk.emulate_scores(qI, w, pk, cand, scale_i), keep, True, 1, True)  # [R, P] additive
    return (sel == 0).to(torch.bfloat16)


def emulate_attend(sel, pos, q_lat, kc, scale_a: float) -> torch.Tensor:
    """o [C, H, R] fp32 from a pool selection sel [C, P] (0 / 1): dsa_fused.emulate's attention."""
    C = q_lat.shape[0]
    L = kc.shape[0]
    tok = torch.where(sel.float().repeat_interleave(4, dim=-1) > 0.5, 0.0, NEG_INF)
    vis = torch.where(torch.arange(L).view(1, L) <= pos.view(C, 1).long(), 0.0, NEG_INF)
    return dsa_prefill.emulate(q_lat, kc, tok + vis, scale_a)


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None and dsa_fused.nki is not None:
    # the kernel tracer resolves names, not module attributes (docs/neuron-notes.md "What a device loop may hold")
    from .dsa_fused import KR, NR, _copy, _ps, _ring, _sb, _sel_begin, _sel_consts, _sel_nsteps, _sel_step

    F32, BF16, I32 = nl.float32, nl.bfloat16, nl.int32
    VE = nisa.vector_engine

    def _ladder_ks(lad: int):
        ks = []  # dsa_fused_c._ladder_of, inline: the tracer runs the kernel's own code
        for k in range(1, 33):
            if (lad >> (k - 1)) & 1:
                ks.append(k)
        return ks

    def _sel_state(Y, P: int):
        """The selection's SBUF state (dsa_fused's kernel's Y), full width; variants slice it."""
        Y["posS"] = _sb((128, 1))
        Y["posm"] = _sb((128, 4))
        Y["acc_f"] = _sb((128, P))
        Y["s2_f"] = _sb((128, P))
        Y["scr_f"] = _sb((128, P))
        Y["tj_f"] = _sb((128, P))
        Y["wio"] = _sb((128, P))
        Y["pio"] = _sb((128, P))
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
        Y["wio_f"] = Y["wio"]
        Y["pio_f"] = Y["pio"]

    def _select_variant(Y, k: int, t_lo: int, t_hi: int, out):
        """The selection of NQ query tiles over P_k = min(P, KU k / 4) pools, each tile's 0 / 1 row block DMAd to out."""
        P, keep = Y["Pf"], Y["keep"]
        Pk = min(P, (KU * k) // 4)
        sel = Pk > keep
        Y["P"] = Pk
        Y["NC"] = (Pk + 511) // 512
        Y["acc"] = Y["acc_f"][:, 0:Pk]
        Y["s2"] = Y["s2_f"][:, 0:Pk]
        Y["scr"] = Y["scr_f"][:, 0:Pk]
        Y["tj"] = Y["tj_f"][:, 0:Pk]
        Y["wio"] = Y["wio_f"][:, 0:Pk]
        Y["pio"] = Y["pio_f"][:, 0:Pk]
        s01 = []
        for s in range(2):
            s01.append(Y["sel01_f"][s][:, 0:Pk])
        Y["sel01"] = s01
        if sel:
            Y["sp"] = _ps()
        for t in range(t_lo, t_hi):
            Y["slot"] = t % 2
            if sel:
                _sel_begin(Y, t)
                for i in range(_sel_nsteps(Y)):
                    _sel_step(Y, i)
            else:
                nisa.memset(dst=s01[t % 2], value=1.0)
            # the whole row block: [P_k, P) holds the zeros the kernel set first
            nisa.dma_copy(dst=out[t * 128:(t + 1) * 128, :], src=Y["sel01_f"][t % 2])

    @nki.jit
    def kiln_dsa_select_kernel(qT, w, pkT, posf, vt, keep: int, nbits: int, scale_i: float, lad: int, rev: int,
                               spl: int = 0):
        """qT bf16 [Hi, D, R] (the indexer queries of the R rows this rank selects for, head-major, transposed), w fp32
        [R, Hi], pkT bf16 [D, P], posf fp32 [R] (their positions), vt int32 [1, NV] (dsa_fused_c.variant_onehot of the
        CALL's positions), keep < P, 2^nbits > P, R a multiple of 128. Returns sel bf16 [R, P], 1 where a row attends a
        pool (dsa_fused's selection and tail), 0 elsewhere and from the variant's P_k on. spl 1 at LNC=2 (grid 2, R / 128
        even): each physical core selects half of the tiles, then a core barrier on the output."""
        Hi, D, R = qT.shape
        P = pkT.shape[1]
        out = nl.ndarray((R, P), dtype=BF16, buffer=nl.shared_hbm)
        Y = dict(C=R, Hi=Hi, D=D, P=P, Pf=P, NC=(P + 511) // 512, keep=keep, nbits=nbits, scale_i=scale_i, qT=qT, w=w,
                 posf=posf.reshape((R, 1)), dscore=None, first=0)
        pk_s = _sb((D, P), BF16)
        nisa.dma_copy(dst=pk_s, src=pkT)
        Y["pkT"] = pk_s
        Y["qs"] = _sb((D, Hi, 128), BF16)
        Y["ws"] = _sb((128, Hi))
        _sel_state(Y, P)
        Y["sel01_f"] = _ring(2, (128, P), BF16)
        for s in range(2):
            nisa.memset(dst=Y["sel01_f"][s], value=0.0)
        NQ = R // 128
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        sp = spl == 1 and npg == 2  # both programs read the same one-hot: the same loops, the same trip counts
        t_lo, t_hi = (pid * (NQ // 2), (pid + 1) * (NQ // 2)) if sp else (0, NQ)
        ks = _ladder_ks(lad)
        NV = len(ks)
        vts = _sb((1, NV), I32)
        nisa.dma_copy(dst=vts, src=vt)
        for v in range(NV):
            rv = nisa.register_alloc()
            nisa.register_load(rv, vts.ap(pattern=[[NV, 1], [1, 1]], offset=v))

            def body(_, k=ks[v]):
                _select_variant(Y, k, t_lo, t_hi, out)

            nl.fori_loop(0, rv, body)
        if npg == 2:
            nisa.core_barrier(data=out, cores=(0, 1))
        return out

    def _mask_block2(X, Y, slot: int, j: int, w: int, ks: int):
        """Query tile's additive mask of key block j from its pool selection, as dsa_fused._mask_block computes it (key
        KU j + i attended iff its pool is selected and KU j + i <= pos), in one vector instruction: the key index i
        against pos - KU j, times the selection read with a stride-0 access pattern (each pool's value on its 4 keys),
        instead of 4 strided writes. Then 0 / 1 -> NEG_INF / 0 on the scalar engine."""
        p0, npl = j * (KU // 4), w // 4
        P = Y["Pf"]
        pj = X["pj"]
        nisa.tensor_scalar(dst=pj, data=X["posA"][slot], op0=nl.add, operand0=float(-KU * j), engine=VE)
        att = X["att01"].reshape((128, KU // 4, 4))
        nisa.scalar_tensor_tensor(dst=att[:, 0:npl, :], data=X["kio"].reshape((128, KU // 4, 4))[:, 0:npl, :],
                                  op0=nl.less_equal, operand0=pj, op1=nl.multiply,
                                  operand1=Y["sel01_f"][slot].ap(pattern=[[P, 128], [1, npl], [0, 4]], offset=p0))
        mk = X["Mk"][ks]
        nisa.activation(dst=mk[:, 0:w], op=nl.copy, data=X["att01"][:, 0:w], scale=-NEG_INF, bias=NEG_INF)
        return mk

    def _attend2(X, Y, t: int, slot: int, NB: int, mdv: int):
        """dsa_fused._attend without the selection micro-steps, with the same arithmetic per (block, head) unit and
        three changes to how it is issued: P K into a ring of 2 PSUM banks (the bank the selection held), so the next
        unit's P K does not wait for this unit's rescale; the block max and the running max as one
        tensor_scalar_reduce per 512-key half (max(m_old, S) reduced: max is exact, so m is the same); the mask by
        _mask_block2. mdv 1: the mask is added on the vector engine instead of by the identity matmul (the tensor
        engine's ~10% of a unit, 0.44 of ~4.3 us; the vector engine has the room once the selection is elsewhere): the
        same fp32 sum, so the same scores; the exp then reads them from SBUF."""
        H, RC, R, L, sc = X["H"], X["RC"], X["R"], X["L"], X["scale"]
        q0 = t * 128
        X["q0"] = q0
        Qin, qt = X["Qin"][t % 2], X["QT"]
        for h in range(H):
            pt = X["Tr"][0][h % 2]
            for c in range(RC):
                nisa.nc_matmul(dst=pt[:, c * 128:(c + 1) * 128], stationary=Qin[:, h, c * 128:(c + 1) * 128],
                               moving=X["IB"], accumulate=False)
            _copy(qt[:, h, :], pt[:, 0:RC * 128], h)
            nisa.memset(dst=X["ACC"][h], value=0.0)
            nisa.memset(dst=X["M"][2 * h + 1], value=NEG_INF)
            nisa.memset(dst=X["Lr"][h], value=0.0)
        if t + 1 < X["NQ"]:  # the next tile's queries, loading while this tile attends
            nisa.dma_copy(dst=X["Qin"][(t + 1) % 2], src=X["q_lat"][q0 + 128:q0 + 256, :, :])
        n = NB * H
        scr = X["SCR"]
        b0 = t * NB  # the rings' slots follow a block count over the whole call, so tile t + 1's first blocks do not
        # share a slot with tile t's last one
        for step in range(n + 4):
            if step < n:
                j, h = step // H, step % H
                w = min(KU, L - j * KU)
                if h == 0:  # the block in: K^T from the scratch, latent rows, the mask from the selection
                    nisa.dma_copy(dst=X["KTb"][(b0 + j) % KR2][:, :, 0:w],
                                  src=X["kth"].ap(pattern=[[KU, 128], [128 * KU, RC], [1, w]], offset=j * RC * 128 * KU))
                    nisa.dma_copy(dst=X["Kr"][(b0 + j) % KR2][:, 0:w // 128, :],
                                  src=X["kc"].ap(pattern=[[R, 128], [128 * R, w // 128], [1, R]], offset=j * KU * R))
                    _mask_block2(X, Y, slot, j, w, (b0 + j) % KR2)
                sb = X["Sr"][step % 2]
                for hb in range((w + 511) // 512):
                    h0 = hb * 512
                    wh = min(512, w - h0)
                    for c in range(RC):
                        nisa.nc_matmul(dst=sb[hb][:, 0:wh], stationary=qt[:, h, c * 128:(c + 1) * 128],
                                       moving=X["KTb"][(b0 + j) % KR2][:, c, h0:h0 + wh], accumulate=c > 0)
                    if not mdv:  # the mask added by one more accumulating matmul against the identity
                        nisa.nc_matmul(dst=sb[hb][:, 0:wh], stationary=X["IB"],
                                       moving=X["Mk"][(b0 + j) % KR2][:, h0:h0 + wh], accumulate=True)
            if 1 <= step <= n:
                u = step - 1
                j, h = u // H, u % H
                w = min(KU, L - j * KU)
                nh = (w + 511) // 512
                sb = X["Sr"][u % 2]
                i = u % NR
                mo, mn = X["M"][2 * h + (j + 1) % 2], X["M"][2 * h + j % 2]
                if mdv:  # the mask added on the vector engine (fl(S + mask), the matmul's last accumulation), into SBUF
                    ss = X["SS"][u % 2]
                    sbm = []
                    for hb in range(nh):
                        h0 = hb * 512
                        wh = min(512, w - h0)
                        nisa.tensor_tensor(dst=ss[:, h0:h0 + wh], data1=sb[hb][:, 0:wh],
                                           data2=X["Mk"][(b0 + j) % KR2][:, h0:h0 + wh], op=nl.add, engine=VE)
                        sbm.append(ss[:, h0:h0 + wh])
                    sb = sbm
                if nh == 1:
                    nisa.tensor_scalar_reduce(dst=scr[:, 0:w], data=sb[0][:, 0:w], op0=nl.maximum, operand0=mo,
                                              reduce_op=nl.maximum, reduce_res=mn)
                else:
                    bm = X["BM"][i]
                    nisa.tensor_scalar_reduce(dst=scr[:, 0:512], data=sb[0][:, 0:512], op0=nl.maximum, operand0=mo,
                                              reduce_op=nl.maximum, reduce_res=bm[:, 0:1])
                    nisa.tensor_scalar_reduce(dst=scr[:, 512:w], data=sb[1][:, 0:w - 512], op0=nl.maximum,
                                              operand0=bm[:, 0:1], reduce_op=nl.maximum, reduce_res=mn)
                nisa.activation(dst=X["NB"][i], op=nl.copy, data=mn, scale=-sc)
                nisa.activation(dst=X["AL"][i], op=nl.exp, data=mo, bias=X["NB"][i], scale=sc)
                pr = X["Pr"][u % 3]
                if nh == 1:
                    nisa.activation_reduce(dst=pr[:, 0:w], op=nl.exp, data=sb[0][:, 0:w], reduce_op=nl.add,
                                           reduce_res=X["RS"][i], bias=X["NB"][i], scale=sc)
                else:
                    nisa.activation(dst=pr[:, 0:512], op=nl.exp, data=sb[0][:, 0:512], bias=X["NB"][i], scale=sc,
                                    reduce_op=nl.add, reduce_cmd=nisa.reduce_cmd.reset_reduce)
                    nisa.activation(dst=pr[:, 512:w], op=nl.exp, data=sb[1][:, 0:w - 512], bias=X["NB"][i], scale=sc,
                                    reduce_op=nl.add, reduce_res=X["RS"][i], reduce_cmd=nisa.reduce_cmd.reduce)
            if 2 <= step <= n + 1:
                u = step - 2
                w = min(KU, L - (u // H) * KU)
                pr = X["Pr"][u % 3]
                if X["ptd"]:  # NeuronCore-v3: P^T by one SBUF -> SBUF DMA transpose (nkilib attention_cte's exp tile)
                    nb = w // 128
                    nisa.dma_transpose(dst=X["PT"][u % 3].ap([[KU, 128], [1, 1], [128, nb], [1, 128]]),
                                       src=pr.ap([[KU, 128], [1, 1], [128, nb], [1, 128]], offset=0))
                else:  # 128-key blocks by tensor-engine transposes into fp32 PSUM, copied out in bf16
                    tb = X["Tr"][0]
                    for kk in range(w // 128):
                        nisa.nc_matmul(dst=tb[kk // 4][:, (kk % 4) * 128:(kk % 4 + 1) * 128],
                                       stationary=pr[:, kk * 128:(kk + 1) * 128], moving=X["IB"], accumulate=False)
                    for hb in range((w + 511) // 512):
                        wh = min(512, w - hb * 512)
                        nisa.activation(dst=X["PT"][u % 3][:, hb * 512:hb * 512 + wh], op=nl.copy,
                                        data=tb[hb][:, 0:wh])
            if 3 <= step <= n + 2:
                u = step - 3
                j = u // H
                w = min(KU, L - j * KU)
                op = X["Or"][u % 2]
                pt = X["PT"][u % 3]
                for kk in range(w // 128):
                    nisa.nc_matmul(dst=op, stationary=pt[:, kk * 128:(kk + 1) * 128],
                                   moving=X["Kr"][(b0 + j) % KR2][:, kk, :], accumulate=kk > 0)
                h = u % H
                i = u % NR
                al = X["AL"][i]
                nisa.activation(dst=X["Lr"][h], op=nl.relu, data=X["Lr"][h], bias=X["RS"][i], scale=al)
                nisa.scalar_tensor_tensor(dst=X["ACC"][h], data=X["ACC"][h], op0=nl.multiply, operand0=al,
                                          op1=nl.add, operand1=op)
        for h in range(H):
            ob = X["OB"][h % 2]
            nisa.reciprocal(dst=X["RL"], data=X["Lr"][h])
            nisa.tensor_scalar(dst=ob, data=X["ACC"][h], op0=nl.multiply, operand0=X["RL"], engine=VE)
            nisa.dma_copy(dst=X["o"][q0:q0 + 128, h, :], src=ob)

    def _attend_variant(X, Y, k: int, t_lo: int, t_hi: int, mdv: int):
        """dsa_fused_c's variant k without the selection: the rows' pool selection comes from HBM (X["sel"])."""
        L, R, RC = X["L"], X["R"], X["RC"]
        Lk = min(L, KU * k)
        Pk = Lk // 4
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
        for _ in range(2):  # P K: a ring of 2 (the bank dsa_fused's selection held)
            Or.append(_ps())
        X["Sr"] = Sr
        X["Tr"] = Tr
        X["Or"] = Or
        s01 = []
        for s in range(2):
            s01.append(Y["sel01_f"][s][:, 0:Pk])
        TO = []
        for x in range(2):
            TO.append(Sr[0][x])
        for x in range(2):
            TO.append(Sr[1][x])
        KTb, Kr, kc, kth, sel = X["KTb"], X["Kr"], X["kc"], X["kth"], X["sel"]
        IB = X["IB"]
        # the first tile's selection rows, positions and queries first: they do not depend on phase 0
        f0 = t_lo * 128
        nisa.dma_copy(dst=s01[t_lo % 2], src=sel[f0:f0 + 128, 0:Pk])
        nisa.dma_copy(dst=X["posA"][t_lo % 2], src=Y["posf"].ap(pattern=[[1, 128], [1, 1]], offset=f0))
        nisa.dma_copy(dst=X["Qin"][t_lo % 2], src=X["q_lat"][f0:f0 + 128, :, :])
        X["NQ"] = t_hi
        for j in range(k):  # phase 0: K^T of blocks 0 .. k - 1 into the block-major scratch
            wj = min(KU, L - j * KU)
            kin = Kr[j % KR2]
            kts = KTb[j % KR2]
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
        for t in range(t_lo, t_hi):
            nxt = t + 1 if t + 1 < t_hi else -1
            if nxt >= 0:  # the next tile's selection rows and positions, ahead of this tile's attention
                nisa.dma_copy(dst=s01[nxt % 2], src=sel[nxt * 128:(nxt + 1) * 128, 0:Pk])
                nisa.dma_copy(dst=X["posA"][nxt % 2], src=Y["posf"].ap(pattern=[[1, 128], [1, 1]], offset=nxt * 128))
            _attend2(X, Y, t, t % 2, k, mdv)

    @nki.jit
    def kiln_dsa_attend_kernel(sel, posf, q_lat, kc, identb, vt, scale_a: float, lad: int, rev: int, mdv: int = 1,
                               spl: int = 0, ptd: int = 0):
        """sel bf16 [C, P] (the rows' pool selection, 0 / 1: select()'s tiles gathered), posf fp32 [C], q_lat bf16
        [C, H, R], kc bf16 [L = 4 P, R], identb bf16 [128, 128], vt int32 [1, NV] (dsa_fused_c.variant_onehot of pos).
        C, L multiples of 128, H >= dsa_fused.MIN_HEADS. Returns o fp32 [C, H, R]: dsa_fused's attention. spl 1 at LNC=2
        (grid 2, C / 128 even): half of the query tiles per physical core (both build the whole K^T scratch, identical
        bytes), then a core barrier on o."""
        C, P = sel.shape
        _, H, R = q_lat.shape
        L = kc.shape[0]
        RC = R // 128
        NB = (L + KU - 1) // KU
        o = nl.ndarray((C, H, R), dtype=F32, buffer=nl.shared_hbm)
        kth = nl.ndarray((NB, RC, 128, KU), dtype=BF16, buffer=nl.shared_hbm)  # K^T, block-major
        IB = _sb((128, 128), BF16)
        nisa.dma_copy(dst=IB, src=identb)
        X = dict(H=H, RC=RC, R=R, L=L, scale=scale_a, IB=IB, kth=kth, kc=kc, q_lat=q_lat, o=o, dbg=0, q0=0, sel=sel)
        X["ptd"] = int(ptd == 1 and not nisa.get_nc_version() <= nisa.nc_version.gen2)  # an SBUF DMA transpose: v3+ only
        X["KTb"] = _ring(KR2, (128, RC, KU), BF16)
        X["Kr"] = _ring(KR2, (128, KU // 128, R), BF16)
        X["Mk"] = _ring(KR2, (128, KU), BF16)
        X["att01"] = _sb((128, KU))
        X["BM2"] = None
        X["QT"] = _sb((128, H, RC * 128), BF16)
        X["Qin"] = _ring(2, (128, H, R), BF16)
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
        X["posA"] = _ring(2, (128, 1))  # each tile's positions, kept with its selection rows for the masks
        X["pj"] = _sb((128, 1))
        X["SCR"] = _sb((128, KU))  # tensor_scalar_reduce's elementwise output (unread)
        X["SS"] = _ring(2, (128, KU))  # mdv: the masked scores, fp32
        ik = _sb((128, KU), I32)
        nisa.iota(dst=ik, pattern=[[1, KU]], offset=0, channel_multiplier=0)
        X["kio"] = _sb((128, KU))  # a key's index in its block, every partition
        nisa.tensor_copy(dst=X["kio"], src=ik, engine=VE)
        # what _mask_block2 reads besides X: the tiles' pool selections and the positions in HBM
        Y = dict(C=C, P=P, Pf=P, posf=posf.reshape((C, 1)))
        Y["sel01_f"] = _ring(2, (128, P), BF16)
        NQ = C // 128
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        sp = spl == 1 and npg == 2
        t_lo, t_hi = (pid * (NQ // 2), (pid + 1) * (NQ // 2)) if sp else (0, NQ)
        ks = _ladder_ks(lad)
        NV = len(ks)
        vts = _sb((1, NV), I32)
        nisa.dma_copy(dst=vts, src=vt)
        for v in range(NV):
            rv = nisa.register_alloc()
            nisa.register_load(rv, vts.ap(pattern=[[NV, 1], [1, 1]], offset=v))

            def body(_, k=ks[v]):
                _attend_variant(X, Y, k, t_lo, t_hi, mdv)

            nl.fori_loop(0, rv, body)
        if npg == 2:
            nisa.core_barrier(data=o, cores=(0, 1))
        return o
else:
    kiln_dsa_select_kernel = kiln_dsa_attend_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source and the REVs of the modules whose helpers it runs."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_dsa_select_kernel = kiln_dsa_attend_kernel = None", a)
    return zlib.crc32(src[a:b].encode() + f"{dsa_fused.REV} {dsa_fused_c.REV}".encode())


REV = _kernel_rev()


def _grid() -> int:
    """The NKI grid (kiln.platform.nki_grid); 1 on a host whose runtime was never configured (the CPU emulations)."""
    from .. import platform

    try:
        return platform.nki_grid()
    except RuntimeError:
        return 1


def _spl(tiles: int, grid: int) -> int:
    """1 when a kernel splits its query tiles over the two physical cores (grid 2, an even tile count, KILN_LNC_SPLIT
    naming dsa_fused: the knob dsa_fused's own split reads)."""
    if grid != 2 or tiles % 2:
        return 0
    from .. import platform

    return int(platform.lnc_split("dsa_fused"))


def _launch(kernel, grid: int, simulate: bool, args: dict):
    if simulate:
        return torch.as_tensor(nki.simulate(kernel[grid] if grid > 1 else kernel)(**args))
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    return wrap_nki(kernel)[grid](**args)


def select(qI, w, pk, pos_rows, pos_call, L: int, keep: int, scale_i: float, simulate: bool = False,
           grid: int | None = None) -> torch.Tensor:
    """bf16 [R, P] 0 / 1: the pool selection of R query rows (qI [R, Hi, D], w [R, Hi], positions pos_rows [R]) over
    the pool keys pk [P, D], its variant set by the call's positions pos_call (every rank of the group the same).
    grid: the launch grid (default the runtime's; simulate: 1 unless given)."""
    if qI.device.type == "cpu" and not simulate:
        return emulate_select(qI, w, pk, pos_rows, keep, scale_i)
    if kiln_dsa_select_kernel is None:
        raise RuntimeError("the NKI split DSA selection kernel needs the nki package (the Neuron venv)")
    P = pk.shape[0]
    ks = sel_ladder(-(-L // KU))
    vt = dsa_fused_c.variant_onehot(pos_call, L, ks)
    args = dict(qT=qI.to(torch.bfloat16).permute(1, 2, 0).contiguous(), w=w.float().contiguous(),
                pkT=pk.to(torch.bfloat16).t().contiguous(), posf=pos_rows.float().contiguous(), vt=vt,
                keep=int(keep), nbits=P.bit_length(), scale_i=float(scale_i), lad=dsa_fused_c._ladder_bits(ks), rev=REV)
    g = grid or (1 if simulate else _grid())
    if _spl(qI.shape[0] // 128, g):
        args["spl"] = 1
    return _launch(kiln_dsa_select_kernel, g, simulate, args)


def attend(sel, pos, q_lat, kc, scale_a: float, simulate: bool = False, grid: int | None = None) -> torch.Tensor:
    """o [C, H, R] fp32 from the pool selection sel [C, P] (0 / 1) of the call's C query rows at positions pos [C]:
    dsa_fused's attention, past the call's last position skipped (dsa_fused_c's variants)."""
    kc2 = kc.reshape(kc.shape[0], -1)
    if q_lat.device.type == "cpu" and not simulate:
        return emulate_attend(sel, pos, q_lat, kc2, scale_a)
    if kiln_dsa_attend_kernel is None:
        raise RuntimeError("the NKI split DSA attention kernel needs the nki package (the Neuron venv)")
    H = q_lat.shape[1]
    if H < dsa_fused.MIN_HEADS:  # the latent ring's hazard below 3 heads (dsa_fused.MIN_HEADS)
        q_lat = torch.cat([q_lat, q_lat.new_zeros(q_lat.shape[0], dsa_fused.MIN_HEADS - H, q_lat.shape[2])], dim=1)
    L = kc2.shape[0]
    ks = dsa_fused_c.ladder(-(-L // KU))
    vt = dsa_fused_c.variant_onehot(pos, L, ks)
    eye = torch.eye(128, device=q_lat.device).to(torch.bfloat16)
    args = dict(sel=sel.to(torch.bfloat16).contiguous(), posf=pos.float().contiguous(),
                q_lat=q_lat.to(torch.bfloat16).contiguous(), kc=kc2.to(torch.bfloat16).contiguous(), identb=eye, vt=vt,
                scale_a=float(scale_a), lad=dsa_fused_c._ladder_bits(ks), rev=REV, mdv=MASK_DVE)
    if PT_DMA:
        args["ptd"] = 1
    g = grid or (1 if simulate else _grid())
    if _spl(q_lat.shape[0] // 128, g):
        args["spl"] = 1
    o = _launch(kiln_dsa_attend_kernel, g, simulate, args)
    return o[:, :H] if H < dsa_fused.MIN_HEADS else o


def split_takes(model, T: int) -> bool:
    """Whether models/mla.py's fused branch splits the selection over the attention group: KILN_DSA_SPLIT_SELECT=1,
    the block traced on the group's rows (DecoderForCausalLM._sp_grp_call: every rank of the group holds all T rows and
    the group gather is the exact zero-padded all-reduce), T a whole number of 128-row tiles per rank (trn1 at grid 1,
    trn2 at LNC=2 with grid 2; on the host the emulations). Read when a graph is traced."""
    if not SPLIT or not getattr(model, "_sp_grp", False) or getattr(model, "attn_tp", 1) <= 1:
        return False
    return hasattr(model, "sp_grp_index") and T % (128 * model.attn_tp) == 0


def attend_split(model, qI, w, pk, pos, q_lat, kc, keep: int, scale_i: float, scale_a: float) -> torch.Tensor:
    """dsa_fused.attend's result (o [C, H, R] fp32) with the selection split over the attention group: this rank
    selects for its block sp_grp_index of attn_tp row blocks, _sp_group_gather assembles every block in row order."""
    A = model.attn_tp

    def own(x):
        return x.reshape(A, -1, *x.shape[1:]).index_select(0, model.sp_grp_index)[0]

    L = kc.shape[0]
    sel_own = select(own(qI), own(w), pk, own(pos), pos, L, keep, scale_i)
    sel = model._sp_group_gather(sel_own.to(torch.bfloat16))
    return attend(sel, pos, q_lat, kc, scale_a)
