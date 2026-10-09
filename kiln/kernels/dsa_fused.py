"""A pooled DSA layer's prefill chunk as ONE NKI kernel for NeuronCore-v2 (trn1): the pooled indexer's scores, their exact
top-k selection (kernels/dsa_topk.py's arithmetic) and the attention over the selected pools (kernels/dsa_prefill.py's),
behind KILN_DSA_FUSED=1 (with KILN_DSA_PREFILL_KERNEL=nki).

Why: in the layer graph the selection kernel (vector-engine bound: ~1.9 ms of DVE, 0.4 of PE) and the attention kernel
(tensor-engine bound: ~2.3 ms of PE, 1.7 of DVE) ran back to back on the critical path, ~6 ms of a ~9 ms token mixer
(trn1, SDK 2.32, 2026-10-05, docs/neuron-notes.md "The attention kernels against their floors"). Their engines are
complementary, so here query tile t + 1's selection is issued interleaved with query tile t's attention, and the
selection's [C, 4P] additive mask never goes to HBM (the selection is kept per pool, the token mask built per block).

What it computes, for one sequence's C queries at positions pos over L = kp P cached keys (P pools of kp = 4 tokens):
  index[q, p] = sum_h w[q, h] relu(scale_i qI[q, h] . pk[p]) + cand[q, p], cand = 0 if the pool's last token
    kp p + kp - 1 <= pos[q] else NEG_INF (a pool is a candidate once complete and visible);
  sel[q, p] = the `keep` largest candidates (ties: lowest index; all of them when fewer), plus the non-candidate pools
    (the query's own incomplete pool and the invisible ones: kernels/dsa_topk.py's tail);
  attend(q, k) = sel[q, k // kp] and k <= pos[q] (the visibility the graph adds to the selection);
  o[q, h] = softmax_k(scale_a q_lat[q, h] . K[k] | attend) K, absorbed MLA over the latent K [L, R].
That is glm5_next.pooled_selection's score kernel path (dsa_topk.score_select with kp 4 and the tail) plus visibility,
then models/mla.py _core; emulate() composes the two modules' emulations.
"""

from __future__ import annotations

import os

import torch

from . import dsa_prefill, dsa_topk

NEG_INF = -1e30
VISIBLE = dsa_topk.VISIBLE
BIG = dsa_topk.BIG
KU = 1024  # keys per attention unit: scores and P^T two one-bank tiles each (a matmul's PSUM tile is one bank at most)
KR = 2  # streamed key blocks in flight
NR = 6


def emulate(qI, w, pk, pos, q_lat, kc, keep: int, scale_i: float, scale_a: float, kp: int = 4) -> torch.Tensor:
    """The kernel's arithmetic in torch (CPU): qI [C, Hi, D], w [C, Hi] fp32, pk [P, D], pos [C] int, q_lat [C, H, R],
    kc [L = kp P, R] -> o [C, H, R] fp32."""
    C = qI.shape[0]
    P = pk.shape[0]
    L = kc.shape[0]
    last = torch.arange(P) * kp + kp - 1
    cand = torch.where(last.view(1, P) <= pos.view(C, 1).long(), 0.0, NEG_INF)
    index = dsa_topk.emulate_scores(qI, w, pk, cand, scale_i)
    sel = dsa_topk.emulate(index, keep, True, kp, True)  # [C, kp P] additive, the tail included
    vis = torch.where(torch.arange(L).view(1, L) <= pos.view(C, 1).long(), 0.0, NEG_INF)
    return dsa_prefill.emulate(q_lat, kc, sel + vis, scale_a)


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

    def _ps():
        return nl.ndarray((128, 512), dtype=F32, buffer=nl.psum)

    def _ring(n: int, shape, dt=None):
        out = []
        for _ in range(n):
            out.append(_sb(shape, dt))
        return out

    def _copy(dst, src, i: int):
        if i % 2:
            nisa.activation(dst=dst, op=nl.copy, data=src)
        else:
            nisa.tensor_copy(dst=dst, src=src, engine=VE)

    # --- the selection of one query tile, as micro-steps that can be issued between attention steps ---------------

    def _sel_nsteps(Y) -> int:
        return Y["Hi"] * Y["NC"] + 1 + 31 + 1 + Y["nbits"] + 1

    def _sel_begin(Y, t: int):
        """Query tile t's indexer queries, weights and positions in, the candidates' 0 / NEG_INF as the score
        accumulator's start (the first micro-steps' inputs)."""
        C, Hi, D, P = Y["C"], Y["Hi"], Y["D"], Y["P"]
        q0 = t * 128
        nisa.dma_copy(dst=Y["qs"], src=Y["qT"].ap(pattern=[[C, D], [D * C, Hi], [1, 128]], offset=q0))
        nisa.dma_copy(dst=Y["ws"], src=Y["w"].ap(pattern=[[Hi, 128], [1, Hi]], offset=q0 * Hi))
        ps_ = Y["posS"]
        nisa.dma_copy(dst=ps_, src=Y["posf"].ap(pattern=[[1, 128], [1, 1]], offset=q0))
        pm = Y["posm"]
        for e in range(4):  # pos - e: token kp p + e is visible iff kp p <= pos - e
            nisa.tensor_scalar(dst=pm[:, e:e + 1], data=ps_, op0=nl.add, operand0=float(-e), engine=VE)
        # cand: NEG_INF unless kp p + kp - 1 <= pos (pio holds kp p)
        nisa.tensor_scalar(dst=Y["acc"], data=Y["pio"], op0=nl.greater, operand0=pm[:, 3:4], op1=nl.multiply,
                           operand1=NEG_INF, engine=VE)

    def _sel_step(Y, i: int):
        """Micro-step i of the selection of the tile _sel_begin started (dsa_topk._select's steps 1-4 for one tile of
        one piece per row, vis_only, with the tail, the result as 0 / 1 per pool into Y["sel01"][Y["slot"]])."""
        Hi, NC, P, keep, nbits = Y["Hi"], Y["NC"], Y["P"], Y["keep"], Y["nbits"]
        S, S2, SCR, TJ = Y["acc"], Y["s2"], Y["scr"], Y["tj"]
        nh = Hi * NC
        if i < nh:  # one (head, 512-pool chunk) of the scores
            h = i // NC
            c0 = (i % NC) * 512
            cw = min(512, P - c0)
            # accumulate=False explicitly: one PSUM tile is reused by every (head, chunk), and the default (None) lets
            # the compiler chain consecutive matmuls into one tile as an accumulation group (measured: garbage scores)
            nisa.nc_matmul(dst=Y["sp"][:, 0:cw], stationary=Y["qs"][:, h, :], moving=Y["pkT"][:, c0:c0 + cw],
                           accumulate=False)
            # relu(scale s) in place in the same PSUM bank: the vector engine's weighted add then reads one PSUM and one
            # SBUF operand (full rate; two SBUF operands run at half, tools/probe_engine_rates.py vars 23 / 24)
            nisa.activation(dst=Y["sp"][:, 0:cw], op=nl.relu, data=Y["sp"][:, 0:cw], scale=Y["scale_i"])
            nisa.scalar_tensor_tensor(dst=S[:, c0:c0 + cw], data=Y["sp"][:, 0:cw], op0=nl.multiply,
                                      operand0=Y["ws"][:, h:h + 1], op1=nl.add, operand1=S[:, c0:c0 + cw])
        elif i == nh:  # 1. sign
            if Y["dscore"] is not None and Y["slot"] == 0 and Y["first"]:
                nisa.dma_copy(dst=Y["dscore"], src=S)
            cp = Y["c1"]
            nisa.tensor_scalar_reduce(dst=SCR, data=S, op0=nl.greater_equal, operand0=0.0, reduce_op=nl.add,
                                      reduce_res=cp)
            nisa.tensor_scalar(dst=Y["f"], data=cp, op0=nl.less, operand0=float(keep), engine=VE)
            nisa.tensor_scalar(dst=Y["sg"], data=Y["f"], op0=nl.multiply, operand0=-2.0, op1=nl.add, operand1=1.0,
                               engine=VE)
            nisa.tensor_scalar(dst=Y["k2"], data=Y["f"], op0=nl.multiply, operand0=float(P + 1 - 2 * keep), op1=nl.add,
                               operand1=float(keep), engine=VE)
            nisa.tensor_scalar(dst=S2, data=S, op0=nl.multiply, operand0=Y["sg"], engine=VE)
            nisa.memset(dst=Y["pre"], value=0)
        elif i <= nh + 31:  # 2. radix over t2's bit pattern, bit 30 .. 0
            b = 30 - (i - nh - 1)
            cand = Y["cand"]
            nisa.tensor_tensor(dst=cand, data1=Y["pre"], data2=Y["bits"][30 - b], op=nl.bitwise_or, engine=VE)
            cp = Y["c1"]
            nisa.tensor_scalar_reduce(dst=SCR, data=S2, op0=nl.greater_equal, operand0=cand.view(F32), reduce_op=nl.add,
                                      reduce_res=cp)
            inc = Y["inc"]
            nisa.tensor_scalar(dst=inc, data=cp, op0=nl.greater_equal, operand0=Y["k2"], op1=nl.multiply,
                               operand1=float(1 << b), engine=VE)
            nisa.tensor_tensor(dst=Y["pre"], data1=Y["pre"], data2=inc, op=nl.bitwise_or, engine=VE)
        elif i == nh + 32:  # 3. above (into S2's buffer, dead now), room, ties
            th = Y["th"]
            nisa.tensor_scalar(dst=th, data=Y["pre"].view(F32), op0=nl.multiply, operand0=Y["sg"], engine=VE)
            abp = Y["c1"]
            nisa.tensor_scalar_reduce(dst=S2, data=S, op0=nl.greater, operand0=th, reduce_op=nl.add, reduce_res=abp)
            nisa.tensor_scalar(dst=Y["room"], data=abp, op0=nl.multiply, operand0=-1.0, op1=nl.add,
                               operand1=float(keep), engine=VE)
            nisa.tensor_scalar(dst=SCR, data=S, op0=nl.equal, operand0=th, engine=VE)  # tied
            nisa.scalar_tensor_tensor(dst=TJ, data=SCR, op0=nl.multiply, operand0=-BIG, op1=nl.add,
                                      operand1=Y["wio"])
            nisa.tensor_scalar(dst=TJ, data=TJ, op0=nl.add, operand0=BIG, engine=VE)
            nisa.memset(dst=Y["lim"], value=0.0)
        elif i <= nh + 32 + nbits:  # binary search for the position limit of the tied
            b = nbits - 1 - (i - nh - 33)
            thr = Y["thr"]
            nisa.tensor_scalar(dst=thr, data=Y["lim"], op0=nl.add, operand0=float(1 << b), engine=VE)
            cp = Y["c1"]
            nisa.tensor_scalar_reduce(dst=SCR, data=TJ, op0=nl.less, operand0=thr, reduce_op=nl.add, reduce_res=cp)
            inc = Y["incf"]
            nisa.tensor_scalar(dst=inc, data=cp, op0=nl.less, operand0=Y["room"], op1=nl.multiply,
                               operand1=float(1 << b), engine=VE)
            nisa.tensor_tensor(dst=Y["lim"], data1=Y["lim"], data2=inc, op=nl.add, engine=VE)
        else:  # 4. out: sel = (above or (tied and j <= lim)) and visible, plus the non-visible (tail), 0 / 1 per pool
            sel = Y["sel01"][Y["slot"]]
            nisa.scalar_tensor_tensor(dst=SCR, data=TJ, op0=nl.less_equal, operand0=Y["lim"], op1=nl.add, operand1=S2)
            vm = TJ  # dead now
            nisa.tensor_scalar(dst=vm, data=S, op0=nl.greater, operand0=VISIBLE, engine=VE)
            nisa.tensor_tensor(dst=SCR, data1=SCR, data2=vm, op=nl.multiply, engine=VE)
            nisa.scalar_tensor_tensor(dst=sel, data=vm, op0=nl.subtract, operand0=1.0, op1=nl.subtract, operand1=SCR)
            nisa.tensor_scalar(dst=sel, data=sel, op0=nl.multiply, operand0=-1.0, engine=VE)

    def _sel_consts(Y):
        """Constants of the selection: pio = kp p and wio = p (every partition), the radix's 2^b bit patterns."""
        P = Y["P"]
        io_i = _sb((128, P), I32)
        nisa.iota(dst=io_i, pattern=[[1, P]], offset=0, channel_multiplier=0)
        nisa.tensor_copy(dst=Y["wio"], src=io_i, engine=VE)
        nisa.tensor_scalar(dst=Y["pio"], data=Y["wio"], op0=nl.multiply, operand0=4.0, engine=VE)
        top = _sb((128, 1), I32)
        nisa.iota(dst=top, pattern=[[0, 1]], offset=1 << 30, channel_multiplier=0)
        bits = [top]
        for b in range(29, -1, -1):
            bt = _sb((128, 1), I32)
            nisa.tensor_scalar(dst=bt, data=top, op0=nl.right_shift, operand0=30 - b, engine=VE)
            bits.append(bt)
        Y["bits"] = bits

    # --- attention ----------------------------------------------------------------------------------------------

    def _mask_block(X, Y, slot: int, j: int, w: int):
        """Query tile's mask of key block j (keys KU j .. KU j + w - 1) from its pool selection: token kp p + e is
        attended iff its pool is selected and kp p <= pos - e; bf16 0 / NEG_INF."""
        p0, npl = j * (KU // 4), w // 4
        att = X["att01"]
        sel = Y["sel01"][slot]
        for e in range(4):
            nisa.scalar_tensor_tensor(dst=att.reshape((128, KU // 4, 4))[:, 0:npl, e], data=Y["pio"][:, p0:p0 + npl],
                                      op0=nl.less_equal, operand0=X["posmA"][slot][:, e:e + 1], op1=nl.multiply,
                                      operand1=sel[:, p0:p0 + npl])
        mk = X["Mk"][j % KR]
        nisa.activation(dst=mk[:, 0:w], op=nl.copy, data=att[:, 0:w], scale=-NEG_INF, bias=NEG_INF)  # 0/1 -> NEG_INF/0
        if X["dbg"]:
            nisa.dma_copy(dst=X["dmask"][X["q0"]:X["q0"] + 128, j * KU:j * KU + w], src=mk[:, 0:w])
        return mk

    def _attend(X, Y, t: int, slot: int, NB: int, sel_t: int):
        """Query tile t's attention over every key block of KU keys (its selection in sel01[slot]), with query tile
        sel_t's selection micro-steps issued between its steps (sel_t < 0: none). The online softmax's per-unit
        stages as kernels/dsa_prefill.py's, unit u issuing A(u), B(u - 1), C(u - 2), D(u - 3) and D's update of u - 4;
        l = alpha l + rowsum runs on the scalar engine as relu(alpha l + rowsum) (all three are non-negative, and the
        copy function takes no tensor bias: NCC_IBVF043), so the vector engine keeps the block max, the running max and
        acc = alpha acc + P K."""
        H, RC, R, L, sc = X["H"], X["RC"], X["R"], X["L"], X["scale"]
        q0 = t * 128
        X["q0"] = q0
        # q_lat^T of the tile, fresh statistics
        Qin, qt = X["Qin"], X["QT"]
        nisa.dma_copy(dst=Qin, src=X["q_lat"][q0:q0 + 128, :, :])
        for h in range(H):
            pt = X["Tr"][0][h % 2]
            for c in range(RC):
                nisa.nc_matmul(dst=pt[:, c * 128:(c + 1) * 128], stationary=Qin[:, h, c * 128:(c + 1) * 128],
                               moving=X["IB"], accumulate=False)
            _copy(qt[:, h, :], pt[:, 0:RC * 128], h)
            nisa.memset(dst=X["ACC"][h], value=0.0)
            nisa.memset(dst=X["M"][2 * h + 1], value=NEG_INF)
            nisa.memset(dst=X["Lr"][h], value=0.0)
        n = NB * H
        ns = 0
        if sel_t >= 0:
            ns = _sel_nsteps(Y)
        for step in range(n + 4):
            # the next tile's selection: its ns micro-steps spread evenly over the n + 4 attention steps
            for si in range((step * ns) // (n + 4), ((step + 1) * ns) // (n + 4)):
                _sel_step(Y, si)
            if step < n:
                j, h = step // H, step % H
                w = min(KU, L - j * KU)
                if h == 0:  # the block in: K^T from the scratch, latent rows, the mask from the selection
                    nisa.dma_copy(dst=X["KTb"][j % KR][:, :, 0:w],
                                  src=X["kth"].ap(pattern=[[KU, 128], [128 * KU, RC], [1, w]], offset=j * RC * 128 * KU))
                    nisa.dma_copy(dst=X["Kr"][j % KR][:, 0:w // 128, :],
                                  src=X["kc"].ap(pattern=[[R, 128], [128 * R, w // 128], [1, R]], offset=j * KU * R))
                    _mask_block(X, Y, slot, j, w)
                sb = X["Sr"][step % 2]
                for hb in range((w + 511) // 512):
                    h0 = hb * 512
                    wh = min(512, w - h0)
                    for c in range(RC):
                        nisa.nc_matmul(dst=sb[hb][:, 0:wh], stationary=qt[:, h, c * 128:(c + 1) * 128],
                                       moving=X["KTb"][j % KR][:, c, h0:h0 + wh], accumulate=c > 0)
                    nisa.nc_matmul(dst=sb[hb][:, 0:wh], stationary=X["IB"], moving=X["Mk"][j % KR][:, h0:h0 + wh],
                                   accumulate=True)
            if 1 <= step <= n:
                u = step - 1
                j, h = u // H, u % H
                w = min(KU, L - j * KU)
                nh = (w + 511) // 512
                sb = X["Sr"][u % 2]
                i = u % NR
                mo, mn = X["M"][2 * h + (j + 1) % 2], X["M"][2 * h + j % 2]
                bm = X["BM"][i]
                for hb in range(nh):
                    nisa.tensor_reduce(dst=bm[:, hb:hb + 1], op=nl.maximum, data=sb[hb][:, 0:min(512, w - hb * 512)],
                                       axis=1)
                nisa.tensor_tensor(dst=mn, data1=mo, data2=bm[:, 0:1], op=nl.maximum, engine=VE)
                if nh > 1:
                    nisa.tensor_tensor(dst=mn, data1=mn, data2=bm[:, 1:2], op=nl.maximum, engine=VE)
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
                tb = X["Tr"][u % 1]
                pr = X["Pr"][u % 3]
                for k in range(w // 128):
                    nisa.nc_matmul(dst=tb[k // 4][:, (k % 4) * 128:(k % 4 + 1) * 128],
                                   stationary=pr[:, k * 128:(k + 1) * 128], moving=X["IB"], accumulate=False)
                for hb in range((w + 511) // 512):
                    wh = min(512, w - hb * 512)
                    nisa.activation(dst=X["PT"][u % 3][:, hb * 512:hb * 512 + wh], op=nl.copy, data=tb[hb][:, 0:wh])
            if 3 <= step <= n + 2:
                u = step - 3
                j = u // H
                w = min(KU, L - j * KU)
                op = X["Or"][0]
                pt = X["PT"][u % 3]
                for k in range(w // 128):
                    nisa.nc_matmul(dst=op, stationary=pt[:, k * 128:(k + 1) * 128], moving=X["Kr"][j % KR][:, k, :],
                                   accumulate=k > 0)
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

    @nki.jit
    def kiln_dsa_fused_kernel(qT, w, pkT, posf, q_lat, kc, identb, keep: int, nbits: int, scale_i: float,
                              scale_a: float, rev: int, dbg: int = 0, spl: int = 0):
        """qT bf16 [Hi, D, C] (the indexer queries, head-major, transposed), w fp32 [C, Hi], pkT bf16 [D, P] (the pool
        keys, transposed), posf fp32 [C] (the queries' positions), q_lat bf16 [C, H, R], kc bf16 [L = 4 P, R] (the
        chunk's latent rows in position order), identb bf16 [128, 128]; keep < P; 2^nbits > P. C, L multiples of 128.
        Returns o fp32 [C, H, R]. spl 1 at LNC=2 (trn2, launched with grid 2: the two physical cores of the logical
        core): each program selects and attends half of the query tiles (C / 128 even, no debug outputs), every tile
        exactly as unsplit; both build the whole K^T scratch (identical bytes), and both wait for the whole o."""
        Hi, D, C = qT.shape
        P = pkT.shape[1]
        _, H, R = q_lat.shape
        L = kc.shape[0]
        RC = R // 128
        NB = (L + KU - 1) // KU
        NT = L // 128
        o = nl.ndarray((C, H, R), dtype=F32, buffer=nl.shared_hbm)
        kth = nl.ndarray((NB, RC, 128, KU), dtype=BF16, buffer=nl.shared_hbm)  # K^T, block-major
        IB = _sb((128, 128), BF16)
        nisa.dma_copy(dst=IB, src=identb)
        # PSUM, the 8 banks: attention scores 2 units x 2 banks, P^T 1 unit x 2 banks, P K 1, selection scores 1
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
        TO = []  # phase 0's transpose banks
        for x in range(2):
            TO.append(Sr[0][x])
        for x in range(2):
            TO.append(Sr[1][x])
        X = dict(H=H, RC=RC, R=R, L=L, scale=scale_a, IB=IB, Sr=Sr, Tr=Tr, Or=Or, kth=kth, kc=kc, q_lat=q_lat, o=o,
                 dbg=dbg, q0=0)
        if dbg:  # debug: every tile's token mask and its pool selection
            X["dmask"] = nl.ndarray((C, L), dtype=BF16, buffer=nl.shared_hbm)
        # phase 0: K^T by transposes of 4 key tiles at a time, written block-major (blocks of KU keys) to the scratch
        KTb = _ring(KR, (128, RC, KU), BF16)
        Kr = _ring(KR, (128, KU // 128, R), BF16)
        for j in range(NB):
            wj = min(KU, L - j * KU)  # (not `w`: that is the indexer weights)
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
        X["KTb"] = KTb
        X["Kr"] = Kr
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
        # the selection's state (one tile at a time; its 0 / 1 pool selection double-buffered for the attention)
        Y = dict(C=C, Hi=Hi, D=D, P=P, NC=(P + 511) // 512, keep=keep, nbits=nbits, scale_i=scale_i, qT=qT, w=w,
                 posf=posf.reshape((C, 1)), dscore=None, first=1)
        if dbg:
            Y["dscore"] = nl.ndarray((128, P), dtype=F32, buffer=nl.shared_hbm)
        pk_s = _sb((D, P), BF16)
        nisa.dma_copy(dst=pk_s, src=pkT)
        Y["pkT"] = pk_s
        Y["qs"] = _sb((D, Hi, 128), BF16)
        Y["ws"] = _sb((128, Hi))
        Y["posS"] = _sb((128, 1))
        Y["posm"] = _sb((128, 4))
        Y["sp"] = _ps()
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
        X["posmA"] = _ring(2, (128, 4))  # each selected tile's pos - e, kept with its selection for the masks
        NQ = C // 128
        # LNC split (spl): program p takes query tiles [p NQ / 2, (p + 1) NQ / 2); the programs' device code differs only
        # in addresses (every loop here is unrolled at trace time)
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        sp = spl == 1 and npg == 2
        t_lo, t_hi = (pid * (NQ // 2), (pid + 1) * (NQ // 2)) if sp else (0, NQ)
        # tile t_lo's selection alone, then tile t's attention with tile t + 1's selection between its steps
        Y["slot"] = t_lo % 2
        _sel_begin(Y, t_lo)
        for i in range(_sel_nsteps(Y)):
            _sel_step(Y, i)
        Y["first"] = 0
        nisa.tensor_copy(dst=X["posmA"][t_lo % 2], src=Y["posm"], engine=VE)
        if dbg:
            Y["dsel"] = nl.ndarray((C, P), dtype=BF16, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=Y["dsel"][0:128, :], src=Y["sel01"][0])
        for t in range(t_lo, t_hi):
            nxt = t + 1 if t + 1 < t_hi else -1
            if nxt >= 0:
                Y["slot"] = nxt % 2
                _sel_begin(Y, nxt)
            _attend(X, Y, t, t % 2, NB, nxt)
            if nxt >= 0:
                nisa.tensor_copy(dst=X["posmA"][nxt % 2], src=Y["posm"], engine=VE)
                if dbg:
                    nisa.dma_copy(dst=Y["dsel"][nxt * 128:(nxt + 1) * 128, :], src=Y["sel01"][nxt % 2])
        if npg == 2:  # whatever runs next on either physical core reads the whole o (nki/isa/_lnc.py core_barrier)
            nisa.core_barrier(data=o, cores=(0, 1))
        if dbg:
            return o, X["dmask"], Y["dsel"], Y["dscore"]
        return o
else:
    kiln_dsa_fused_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source (LNL's compile-cache key does not include NKI kernel source: CLAUDE.md)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_dsa_fused_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev()
# Where its A/Bs and the wikitext gate were measured: trn1 (docs/neuron-notes.md "The attention kernels against their floors")
# and trn2 with the query-tile LNC split (feat/trn2-fast, trn2.48xlarge, SDK 2.32, GLM-5.3-Flash tp=32 / DP attention 4,
# 2026-10-06: per engine at conc 32 / 64 / 128 101.2 / 110.8 / 117.6 -> 104.4 / 114.7 / 121.9 out tok/s, prefill call
# 0.922-0.937 -> 0.885-0.899 s; wikitext -0.5441 -> -0.5498, |dlogprob| mean 0.045, greedy 98.2%; check_mixed 28 / 32 equal,
# signed dlogprob -0.0020 on the prefill chunks, +0.0009 on the decode calls; docs/neuron-notes.md "trn2 on engine-v0 70ddc1b").
FUSED_FAMILIES = ("trn1", "trn2")


def _default_fused() -> str:
    """KILN_DSA_FUSED unset: on for a trn1 or trn2 target, off elsewhere (inf2 and a host without a Neuron device)."""
    from .. import platform

    t = platform.target()
    return "1" if t is not None and platform.family_of(t) in FUSED_FAMILIES else "0"


FUSED = (os.environ.get("KILN_DSA_FUSED") or _default_fused()) == "1"


def supported(C: int, P: int, L: int, R: int, Hi: int, D: int, kp: int) -> bool:
    return kp == 4 and L == 4 * P and L % 128 == 0 and D == 128 and R % 128 == 0 and R <= 512 and P <= 4096 and Hi <= 64


def split(C: int, dbg: int = 0) -> int:
    """1 when the kernel splits its query tiles over the two physical cores of an LNC=2 logical core (kiln/platform.py
    KILN_LNC_SPLIT names dsa_fused and the runtime's LNC is 2; C / 128 even, no debug outputs). Part of the graph key
    (the kernel's spl argument, passed only when set, so the unsplit key is as before)."""
    from .. import platform

    return int(platform.nki_grid() == 2 and platform.lnc_split("dsa_fused") and (C // 128) % 2 == 0 and not dbg)


# The kernel's latent ring Kr holds KR = 2 key blocks: block j is loaded at attention step j H (its first head) and read
# by stage 3 up to step j H + H + 2 (its last head), and block j + 2 is loaded into the same slot at step j H + 2 H, before
# stage 3 within a step. So fewer than 3 heads per call read a block that is already overwritten: at attention TP 32
# (GLM-5.3-Flash's 64 MLA heads, 2 per rank) head 1 attended the wrong keys (nki.simulate: H = 2 off by 2.1, H = 1 by
# 2.3, H = 3 and 8 within bf16 of emulate()), which made every DP-attention-1 prefill wrong on the device. attend() pads
# such calls to MIN_HEADS zero heads (their outputs are dropped) instead of growing the ring, which would change this
# module's kernel source and with it REV, the cache key of every configuration that runs the kernel.
MIN_HEADS = 3


def attend(qI, w, pk, pos, q_lat, kc, keep: int, scale_i: float, scale_a: float, dbg: int = 0,
           simulate: bool = False):
    """o [C, H, R] fp32: the fused kernel on a Neuron device, emulate() elsewhere. qI [C, Hi, D] (bf16 values),
    w [C, Hi] fp32, pk [P, D] (bf16 values), pos [C] int, q_lat [C, H, R], kc [L, R] (or [L, 1, R]). simulate: run the
    kernel in nki.simulate on host tensors, with every argument exactly as the device call takes it (tests).
    KILN_DSA_FUSED_CAUSAL=1: kernels/dsa_fused_c.py, the same result bit for bit without the work past the call's last
    position; unset, only the gated trn2 configuration (dsa_fused_c.default_on) takes it, and every other graph traces
    as before."""
    from . import dsa_fused_c

    if not dbg and not simulate and dsa_fused_c.takes(q_lat.shape[0], kc.shape[0]):
        return dsa_fused_c.attend(qI, w, pk, pos, q_lat, kc, keep, scale_i, scale_a)
    kc2 = kc.reshape(kc.shape[0], -1)
    if q_lat.device.type == "cpu" and not simulate:
        return emulate(qI, w, pk, pos, q_lat, kc2, keep, scale_i, scale_a)
    from .. import platform

    if kiln_dsa_fused_kernel is None:
        raise RuntimeError("the NKI fused DSA kernel needs the nki package (the Neuron venv)")
    H = q_lat.shape[1]
    if H < MIN_HEADS:
        q_lat = torch.cat([q_lat, q_lat.new_zeros(q_lat.shape[0], MIN_HEADS - H, q_lat.shape[2])], dim=1)
    C = q_lat.shape[0]
    P = pk.shape[0]
    Cp = -(-C // 128) * 128
    if Cp != C:  # whole query tiles; a padded row sits at position 0 and its output is dropped
        pad = Cp - C
        qI = torch.cat([qI, qI.new_zeros(pad, *qI.shape[1:])])
        w = torch.cat([w, w.new_zeros(pad, *w.shape[1:])])
        q_lat = torch.cat([q_lat, q_lat.new_zeros(pad, *q_lat.shape[1:])])
        pos = torch.cat([pos, pos.new_zeros(pad)])
    eye = torch.eye(128, device=q_lat.device).to(torch.bfloat16)
    if simulate:  # the same arguments as the device call below
        o = nki.simulate(kiln_dsa_fused_kernel)(
            qT=qI.to(torch.bfloat16).permute(1, 2, 0).contiguous(), w=w.float().contiguous(),
            pkT=pk.to(torch.bfloat16).t().contiguous(), posf=pos.float().contiguous(),
            q_lat=q_lat.to(torch.bfloat16).contiguous(), kc=kc2.to(torch.bfloat16).contiguous(), identb=eye,
            keep=int(keep), nbits=P.bit_length(), scale_i=float(scale_i), scale_a=float(scale_a), rev=REV, dbg=dbg)
        o = tuple(torch.as_tensor(x) for x in o) if dbg else torch.as_tensor(o)
    else:
        # Written exactly as before MIN_HEADS: the compile-cache key hashes the traced FX graph, node names included, and
        # passing these arguments through a dict renamed nodes (contiguous_21 -> value_21) and changed every key.
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        spl = split(Cp, dbg)
        o = wrap_nki(kiln_dsa_fused_kernel)[platform.nki_grid()](
            qT=qI.to(torch.bfloat16).permute(1, 2, 0).contiguous(), w=w.float().contiguous(),
            pkT=pk.to(torch.bfloat16).t().contiguous(), posf=pos.float().contiguous(),
            q_lat=q_lat.to(torch.bfloat16).contiguous(), kc=kc2.to(torch.bfloat16).contiguous(), identb=eye,
            keep=int(keep), nbits=P.bit_length(), scale_i=float(scale_i), scale_a=float(scale_a), rev=REV, dbg=dbg,
            **({"spl": 1} if spl else {}))
    if dbg:
        return o
    if H < MIN_HEADS:
        o = o[:C] if Cp != C else o
        return o[:, :H]
    return o[:C] if Cp != C else o
