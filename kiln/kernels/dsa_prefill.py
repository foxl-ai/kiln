"""The attention core of a pooled DSA layer's prefill chunk (GLM-5.3-Flash, models/glm5_next.py) as one NKI kernel
for NeuronCore-v2 (trn1), behind KILN_DSA_PREFILL_KERNEL=nki: flash attention over the latent cache with the
selection (and the causal visibility) as an additive mask.

Why: in the sweep's 4096-token prefill call the XLA form (models/mla.py _core, expand: the latent decompressed to
per-head keys and values, a [8, 1024, 8448] fp32 score tensor, a softmax over it) took 8.7 ms per DSA layer at
1024 queries over 8448 keys, 11 layers, ~15% of the call (trn1.32xlarge, SDK 2.32, 2026-10-05, docs/neuron-notes.md
"Where the 4096-token prefill call goes after the group collectives"); alone on one NeuronCore 9.49 ms, with 953,472
spill DMA runs (tools/prof_attn_kernels.py dsa_core). Its arithmetic is ~106 GFLOP (expand) or ~146 (absorbed): 1.3
/ 1.8 ms at trn1's measured 82 TFLOPS bf16 (0.43 ns per moving column of a [128, 128] x [128, 512] matmul,
tools/probe_engine_rates.py var 0), so the XLA form runs at ~13% of the tensor engine.

What it computes, per query q and head h (absorbed MLA, NoPE: the latent is both key and value):
s[k] = scale * (q_lat[q, h] . K[k] + mask[q, k]) over the chunk's L cached keys, p = softmax(s), o[q, h] = sum_k p[k]
K[k]. mask is 0 for a key to attend and NEG_INF (bf16) otherwise; every query row must have at least one 0 (its own
pool is always attended: the selection's tail, or a selected pool). The model applies W_UV after (models/mla.py).
That is _core's absorbed branch; the expand branch computes the same values (q_nope . W_UK c = (W_UK^T q_nope) . c).

Layout (one NeuronCore): the latent arrives as K [L, R] bf16, tokens on the partitions after a DMA. Phase 0 makes
K^T [R, L] once (R = 4 chunks of 128 on the partitions) by tensor-engine transposes against a bf16 identity and keeps
it in SBUF (4 x L x 2 bytes per partition: 67.6 KB at 8448 keys). Then per tile of 128 queries: q_lat^T [R, 128] per
head by transposes; per block of 512 keys (the last one may be shorter) and head, a four-stage software pipeline over
the (block, head) units so no engine waits on another's work of the same unit:
  A (tensor): S = q_lat^T . K^T block, 4 accumulating matmuls over R, plus the block's mask added by one more
    accumulating matmul against the identity (the mask block is the moving operand), in one PSUM bank.
  B (vector, scalar): the block's row max, the running max m, alpha = exp(scale (m_old - m)), P = exp(scale S - scale
    m) in bf16 with its row sum (one scalar-engine instruction), l = alpha l + rowsum.
  C (tensor, scalar): P^T [keys, queries] by 4 transposes into one PSUM bank, copied to SBUF in bf16.
  D (tensor, vector): P K (4 accumulating matmuls, P^T stationary, the block's latent rows [128, R] moving), then
    acc = alpha acc + P K (one vector instruction reading the PSUM) and l = alpha l + rowsum.
Unit u issues A(u), B(u - 1), C(u - 2), D's matmuls of u - 3 and D's vector update of u - 4 (with l = alpha l +
rowsum), so the vector engine never waits on the tensor engine's work of the same step. The latent rows for D are streamed per (query tile, block) from HBM
(512 KB each; keeping them resident as well would overrun SBUF). Out: acc / l per head, fp32 [C, H, R].
Masked blocks: a row whose block is all NEG_INF gets m = NEG_INF there and P = 1 on it until the first block with an
attended key, whose alpha = exp(scale (NEG_INF - m)) = 0 wipes acc and l; after that a masked block's P is 0. So the
result is exact for every row with an attended key, which every real row has.

PSUM (8 banks of [128, 512] fp32 per partition): S ring 3, P^T ring 2, P K ring 2. No accumulation group ever writes
part of a bank another group shares (the trn1 hazard in docs/neuron-notes.md "A PSUM accumulation hazard on trn1").
"""

from __future__ import annotations

import os

import torch

NEG_INF = -1e30  # models/decoder.NEG_INF
KB = 512  # keys per block (one PSUM bank of fp32 scores)
SR, TR, OR, KR = 3, 2, 2, 3  # rings: score banks, P^T banks, P K banks, streamed latent blocks
NR = 6  # ring of the per-unit [128, 1] statistics (alive from stage B to stage D2)


def emulate(q_lat: torch.Tensor, kc: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Tensor:
    """The kernel's arithmetic in torch (CPU), up to the order of its sums: q_lat [C, H, R], kc [L, R], mask [C, L]
    additive (0 / NEG_INF) -> o [C, H, R] fp32. bf16 operands as the kernel takes them (an fp32 input stays fp32),
    fp32 scores, p rounded to bf16 before p K, as the XLA form does."""
    rnd = (lambda t: t.to(torch.bfloat16).float()) if kc.dtype != torch.float32 else (lambda t: t.float())
    K = rnd(kc)
    s = (torch.einsum("chr,lr->hcl", rnd(q_lat), K) + rnd(mask).unsqueeze(0)) * scale
    p = rnd(torch.softmax(s, dim=-1))
    return torch.einsum("hcl,lr->chr", p, K)


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
        return nl.ndarray((128, KB), dtype=F32, buffer=nl.psum)

    def _ring(n: int, shape, dt=None):
        """n SBUF tiles, each a tensor of its own (the compiler tracks dependencies per tensor)."""
        out = []
        for _ in range(n):
            out.append(_sb(shape, dt))
        return out

    def _copy(dst, src, i: int):
        """PSUM -> SBUF, alternating the vector and scalar engines."""
        if i % 2:
            nisa.activation(dst=dst, op=nl.copy, data=src)
        else:
            nisa.tensor_copy(dst=dst, src=src, engine=VE)

    def _width(X, j: int) -> int:
        return min(KB, X["L"] - j * KB)

    def _stage_a(X, u: int):
        """S = q_lat^T K^T block + mask block (tensor engine); the block's mask and latent rows streamed in."""
        H, RC, R = X["H"], X["RC"], X["R"]
        j, h = u // H, u % H
        w = _width(X, j)
        q0 = X["q0"]
        mk, kr = X["Mk"][j % KR], X["Kr"][j % KR]
        if h == 0:
            nisa.dma_copy(dst=mk[:, 0:w], src=X["mask"][q0:q0 + 128, j * KB:j * KB + w])
            n4 = w // 128
            nisa.dma_copy(dst=kr[:, 0:n4, :],
                          src=X["kc"].ap(pattern=[[R, 128], [128 * R, n4], [1, R]], offset=j * KB * R))
        s = X["Sr"][u % SR]
        qt = X["qt"]
        for c in range(RC):
            nisa.nc_matmul(dst=s[:, 0:w], stationary=qt[:, h, c * 128:(c + 1) * 128],
                           moving=X["KT"][c][:, j * KB:j * KB + w], accumulate=c > 0)
        nisa.nc_matmul(dst=s[:, 0:w], stationary=X["IB"], moving=mk[:, 0:w], accumulate=True)

    def _stage_b(X, u: int):
        """The block's row max and the running max (vector engine); -scale m, alpha = exp(scale m_old - scale m) and
        P = exp(scale S - scale m) with its row sum (scalar engine, which has more room)."""
        H, scale = X["H"], X["scale"]
        j, h = u // H, u % H
        w = _width(X, j)
        s = X["Sr"][u % SR]
        mo, mn = X["M"][2 * h + (j + 1) % 2], X["M"][2 * h + j % 2]
        i = u % NR
        bm, al, nb, rs = X["BM"][i], X["AL"][i], X["NB"][i], X["RS"][i]
        nisa.tensor_reduce(dst=bm, op=nl.maximum, data=s[:, 0:w], axis=1)
        nisa.tensor_tensor(dst=mn, data1=mo, data2=bm, op=nl.maximum, engine=VE)
        nisa.activation(dst=nb, op=nl.copy, data=mn, scale=-scale)
        nisa.activation(dst=al, op=nl.exp, data=mo, bias=nb, scale=scale)
        nisa.activation_reduce(dst=X["Pr"][u % 3][:, 0:w], op=nl.exp, data=s[:, 0:w], reduce_op=nl.add, reduce_res=rs,
                               bias=nb, scale=scale)

    def _stage_c(X, u: int):
        """P^T [keys, queries] by transposes into one PSUM bank, copied to SBUF in bf16."""
        w = _width(X, u // X["H"])
        tp = X["Tr"][u % TR]
        pr = X["Pr"][u % 3]
        for i in range(w // 128):
            nisa.nc_matmul(dst=tp[:, i * 128:(i + 1) * 128], stationary=pr[:, i * 128:(i + 1) * 128], moving=X["IB"],
                           accumulate=False)
        nisa.activation(dst=X["PT"][u % 3][:, 0:w], op=nl.copy, data=tp[:, 0:w])

    def _stage_d1(X, u: int):
        """P K: P^T stationary, the block's latent rows moving."""
        j = u // X["H"]
        w = _width(X, j)
        op = X["Or"][u % OR]
        pt, kr = X["PT"][u % 3], X["Kr"][j % KR]
        for i in range(w // 128):
            nisa.nc_matmul(dst=op, stationary=pt[:, i * 128:(i + 1) * 128], moving=kr[:, i, :], accumulate=i > 0)

    def _stage_d2(X, u: int):
        """l = alpha l + rowsum, acc = alpha acc + P K (vector engine; the scalar engine's copy takes no tensor bias:
        NCC_IBVF043)."""
        h = u % X["H"]
        i = u % NR
        al = X["AL"][i]
        nisa.scalar_tensor_tensor(dst=X["Lr"][h], data=X["Lr"][h], op0=nl.multiply, operand0=al, op1=nl.add,
                                  operand1=X["RS"][i])
        nisa.scalar_tensor_tensor(dst=X["ACC"][h], data=X["ACC"][h], op0=nl.multiply, operand0=al, op1=nl.add,
                                  operand1=X["Or"][u % OR])

    @nki.jit
    def kiln_dsa_prefill_kernel(q_lat, kc, mask, identb, scale: float, rev: int):
        """q_lat bf16 [C, H, R] (C a multiple of 128, R a multiple of 128); kc bf16 [L, R] (L a multiple of 128) the
        chunk's latent cache rows in order; mask bf16 [C, L] additive (0 attend, NEG_INF not); identb bf16 [128, 128]
        the identity; rev: this module's kernel source revision. Returns o fp32 [C, H, R]."""
        C, H, R = q_lat.shape
        L = kc.shape[0]
        RC = R // 128
        NT = L // 128  # key tiles of 128
        NB = (L + KB - 1) // KB  # key blocks
        o = nl.ndarray((C, H, R), dtype=F32, buffer=nl.shared_hbm)
        IB = _sb((128, 128), BF16)
        nisa.dma_copy(dst=IB, src=identb)
        # PSUM rings, each tile exactly one 2 KB bank per partition, allocated first
        Sr = []
        for _ in range(SR):
            Sr.append(_ps())
        Tr = []
        for _ in range(TR):
            Tr.append(_ps())
        Or = []
        for _ in range(OR):
            Or.append(_ps())
        TO = []  # the transpose banks of the prologues: Tr then Or
        for x in range(TR):
            TO.append(Tr[x])
        for x in range(OR):
            TO.append(Or[x])
        # phase 0: K^T [R, L] in SBUF (RC chunks of 128 rows), by transposes of 4 key tiles at a time
        KT = _ring(RC, (128, L), BF16)
        Kin = _ring(2, (128, 4, R), BF16)
        for g in range(0, NT, 4):
            n4 = min(4, NT - g)
            kin = Kin[(g // 4) % 2]
            nisa.dma_copy(dst=kin[:, 0:n4, :], src=kc.ap(pattern=[[R, 128], [128 * R, n4], [1, R]], offset=g * 128 * R))
            for c in range(RC):
                pt = TO[((g // 4) * RC + c) % (TR + OR)]
                for i in range(n4):
                    nisa.nc_matmul(dst=pt[:, i * 128:(i + 1) * 128], stationary=kin[:, i, c * 128:(c + 1) * 128],
                                   moving=IB, accumulate=False)
                _copy(KT[c][:, g * 128:(g + n4) * 128], pt[:, 0:n4 * 128], c)
        QT = _ring(2, (128, H, RC * 128), BF16)
        Qin = _sb((128, H, R), BF16)
        X = dict(H=H, RC=RC, R=R, L=L, scale=scale, mask=mask, kc=kc, IB=IB, KT=KT, Sr=Sr, Tr=Tr, Or=Or,
                 ACC=_ring(H, (128, R)), M=_ring(2 * H, (128, 1)), Lr=_ring(H, (128, 1)), Kr=_ring(KR, (128, 4, R), BF16),
                 Mk=_ring(KR, (128, KB), BF16), Pr=_ring(3, (128, KB), BF16), PT=_ring(3, (128, KB), BF16),
                 BM=_ring(NR, (128, 1)), AL=_ring(NR, (128, 1)), NB=_ring(NR, (128, 1)), RS=_ring(NR, (128, 1)))
        OB = _ring(2, (128, R))
        RL = _sb((128, 1))
        NU = NB * H
        for t in range(C // 128):
            q0 = t * 128
            qt = QT[t % 2]
            X["q0"] = q0
            X["qt"] = qt
            nisa.dma_copy(dst=Qin, src=q_lat[q0:q0 + 128, :, :])
            for h in range(H):
                pt = TO[(t * H + h) % (TR + OR)]
                for c in range(RC):
                    nisa.nc_matmul(dst=pt[:, c * 128:(c + 1) * 128], stationary=Qin[:, h, c * 128:(c + 1) * 128],
                                   moving=IB, accumulate=False)
                _copy(qt[:, h, :], pt[:, 0:RC * 128], h)
                nisa.memset(dst=X["ACC"][h], value=0.0)
                nisa.memset(dst=X["M"][2 * h + 1], value=NEG_INF)
                nisa.memset(dst=X["Lr"][h], value=0.0)
            for u in range(NU + 4):
                if u < NU:
                    _stage_a(X, u)
                if u >= 1 and u - 1 < NU:
                    _stage_b(X, u - 1)
                if u >= 2 and u - 2 < NU:
                    _stage_c(X, u - 2)
                if u >= 3 and u - 3 < NU:
                    _stage_d1(X, u - 3)
                if u >= 4:
                    _stage_d2(X, u - 4)
            for h in range(H):
                ob = OB[h % 2]
                nisa.reciprocal(dst=RL, data=X["Lr"][h])
                nisa.tensor_scalar(dst=ob, data=X["ACC"][h], op0=nl.multiply, operand0=RL, engine=VE)
                nisa.dma_copy(dst=o[q0:q0 + 128, h, :], src=ob)
        return o

    # --- the causal form: a device loop over the visible pairs of 512-key blocks -------------------------------

    def _pipeline(P_, units):
        """Run units through stages A, B, C, D1, D2 with the skew of the static kernel. P_: this region's PSUM rings
        and rings of per-unit statistics; each unit a dict: qt (the query tile's q_lat^T), h, w, kt (RC SBUF slices
        [128, w] of K^T), mk ([128, w] mask), kr ([128, w / 128, R] latent rows), acc, lr, mo, mn."""
        n = len(units)
        sc = P_["scale"]
        for step in range(n + 4):
            if step < n:
                u, U = step, units[step]
                s = P_["Sr"][u % SR]
                w = U["w"]
                for c in range(len(U["kt"])):
                    nisa.nc_matmul(dst=s[:, 0:w], stationary=U["qt"][:, U["h"], c * 128:(c + 1) * 128], moving=U["kt"][c],
                                   accumulate=c > 0)
                nisa.nc_matmul(dst=s[:, 0:w], stationary=P_["IB"], moving=U["mk"], accumulate=True)
            if 1 <= step <= n:
                u = step - 1
                U = units[u]
                w = U["w"]
                s = P_["Sr"][u % SR]
                i = u % NR
                nisa.tensor_reduce(dst=P_["BM"][i], op=nl.maximum, data=s[:, 0:w], axis=1)
                nisa.tensor_tensor(dst=U["mn"], data1=U["mo"], data2=P_["BM"][i], op=nl.maximum, engine=VE)
                nisa.activation(dst=P_["NB"][i], op=nl.copy, data=U["mn"], scale=-sc)
                nisa.activation(dst=P_["AL"][i], op=nl.exp, data=U["mo"], bias=P_["NB"][i], scale=sc)
                nisa.activation_reduce(dst=P_["Pr"][u % 3][:, 0:w], op=nl.exp, data=s[:, 0:w], reduce_op=nl.add,
                                       reduce_res=P_["RS"][i], bias=P_["NB"][i], scale=sc)
            if 2 <= step <= n + 1:
                u = step - 2
                w = units[u]["w"]
                tp = P_["Tr"][u % TR]
                for k in range(w // 128):
                    nisa.nc_matmul(dst=tp[:, k * 128:(k + 1) * 128], stationary=P_["Pr"][u % 3][:, k * 128:(k + 1) * 128],
                                   moving=P_["IB"], accumulate=False)
                nisa.activation(dst=P_["PT"][u % 3][:, 0:w], op=nl.copy, data=tp[:, 0:w])
            if 3 <= step <= n + 2:
                u = step - 3
                U = units[u]
                op = P_["Or"][u % OR]
                for k in range(U["w"] // 128):
                    nisa.nc_matmul(dst=op, stationary=P_["PT"][u % 3][:, k * 128:(k + 1) * 128], moving=U["kr"][:, k, :],
                                   accumulate=k > 0)
            if step >= 4:
                u = step - 4
                U = units[u]
                i = u % NR
                nisa.scalar_tensor_tensor(dst=U["lr"], data=U["lr"], op0=nl.multiply, operand0=P_["AL"][i], op1=nl.add,
                                          operand1=P_["RS"][i])
                nisa.scalar_tensor_tensor(dst=U["acc"], data=U["acc"], op0=nl.multiply, operand0=P_["AL"][i],
                                          op1=nl.add, operand1=P_["Or"][u % OR])

    def _region(X):
        """A region's own PSUM rings and statistic rings (a PSUM tensor may not be referenced by two device-loop
        regions: NCC_IBIR092, docs/neuron-notes.md)."""
        P_ = dict(scale=X["scale"], IB=X["IB"], Sr=[], Tr=[], Or=[], Pr=X["Pr"], PT=X["PT"], BM=X["BM"], AL=X["AL"],
                  NB=X["NB"], RS=X["RS"])
        for _ in range(SR):
            P_["Sr"].append(_ps())
        for _ in range(TR):
            P_["Tr"].append(_ps())
        for _ in range(OR):
            P_["Or"].append(_ps())
        return P_

    def _pair_units(X, nq: int, ne: int, w: int, kts, krs, mks):
        """The units of a pass's ne blocks: (block, query tile, head) in that order. kts[e]: the RC K^T slices of block
        e, krs[e] its latent rows, mks[qi][e] query tile qi's mask of it."""
        H = X["H"]
        units = []
        for e in range(ne):
            for qi in range(nq):
                for h in range(H):
                    k = qi * H + h
                    units.append(dict(qt=X["QT"][qi], h=h, w=w, kt=kts[e], kr=krs[e], mk=mks[qi][e], acc=X["ACC"][k],
                                      lr=X["Lr"][k], mo=X["M"][2 * k + (e + 1) % 2], mn=X["M"][2 * k + e % 2]))
        return units

    def _pair_body(X, it, qtiles):
        """One device-loop iteration: pair `it` (keys 1024 it .. 1024 it + 1023) for the pass's query tiles. Its
        K^T (pair-major scratch, the loop register as the offset), latent rows and masks (at the pair's first key,
        read from a table through the register) come in by DMA."""
        R, RC, L = X["R"], X["RC"], X["L"]
        P_ = _region(X)
        ktb, krb = X["KTb"], X["KRb"]
        nisa.dma_copy(dst=ktb, src=X["kth"].ap(pattern=[[2 * KB, 128], [128 * 2 * KB, RC], [1, 2 * KB]], offset=0,
                                                scalar_offset=it, indirect_dim=0))
        off = X["offs"]  # [1, 2] int32: the pair's first key and its first latent element (1024 it, 1024 it R)
        nisa.dma_copy(dst=off, src=X["tab"].ap(pattern=[[2, 1], [1, 2]], offset=0, scalar_offset=it, indirect_dim=0))
        nisa.dma_copy(dst=krb, src=X["kcf"].ap(pattern=[[R, 128], [128 * R, 8], [1, R]], offset=0,
                                               scalar_offset=off[:, 1:2], indirect_dim=0))
        nq = len(qtiles)
        for qi in range(nq):
            q0 = qtiles[qi] * 128
            nisa.dma_copy(dst=X["MKb"][qi], src=X["maskf"].ap(pattern=[[L, 128], [1, 2 * KB]], offset=q0 * L,
                                                              scalar_offset=off[:, 0:1], indirect_dim=0))
        kts, krs, mks = [], [], []
        for e in range(2):
            ke = []
            for c in range(RC):
                ke.append(ktb[:, c, e * KB:(e + 1) * KB])
            kts.append(ke)
            krs.append(krb[:, 4 * e:4 * e + 4, :])
        for qi in range(nq):
            me = []
            for e in range(2):
                me.append(X["MKb"][qi][:, e * KB:(e + 1) * KB])
            mks.append(me)
        _pipeline(P_, _pair_units(X, nq, 2, KB, kts, krs, mks))

    def _t4():
        out = []
        for _ in range(4):
            out.append(_ps())
        return out

    @nki.jit
    def kiln_dsa_prefill_loop_kernel(q_lat, kc, mask, identb, npair, tab, scale: float, qpass: int, rev: int):
        """kiln_dsa_prefill_kernel's arithmetic, attending only the pairs of 512-key blocks a pass of query tiles can
        see: npair int32 [1, C / 128 / qpass] the pairs per pass (keys 0 .. 1024 npair - 1; the caller's causal
        extent), tab int32 [NP, 2] = (1024 i, 1024 i R) (pair i's first key and latent element), qpass query tiles per
        pass. Keys from 1024 NP = 1024 (L // 1024) on are a static tail block every pass attends. Returns o fp32
        [C, H, R]."""
        C, H, R = q_lat.shape
        L = kc.shape[0]
        RC = R // 128
        NT = L // 128
        NP = L // (2 * KB)  # full pairs
        LT = L - NP * 2 * KB  # tail keys (a multiple of 128)
        o = nl.ndarray((C, H, R), dtype=F32, buffer=nl.shared_hbm)
        # scratch: K^T of every full pair, pair-major, for the loop's DMAs
        kth = nl.ndarray((NP, RC, 128, 2 * KB), dtype=BF16, buffer=nl.shared_hbm)
        IB = _sb((128, 128), BF16)
        nisa.dma_copy(dst=IB, src=identb)
        T0 = _t4()  # phase 0's transpose banks (each prologue takes fresh ones: nothing in PSUM lives across a loop)
        KTt = _ring(RC, (128, 256), BF16)  # the tail's K^T (LT <= 256 here: 8448 = 8 x 1024 + 256)
        KRt = _sb((128, 2, R), BF16)  # the tail's latent rows
        # SBUF is tight: in a layer graph the dynamic DMAs' scratch is pinned in SBUF too (NCC_INLA001 "overlapping
        # with must-pinned memloc DynamicDMAScratchLoc" with 4 query tiles per pass), so phase 0 runs in the loop's
        # pair buffers
        KTb = _sb((128, RC, 2 * KB), BF16)
        KRb = _sb((128, 8, R), BF16)
        for g in range(NP):  # phase 0: per pair, its 8 key tiles, transposed
            kin, kts = KRb, KTb
            nisa.dma_copy(dst=kin, src=kc.ap(pattern=[[R, 128], [128 * R, 8], [1, R]], offset=g * 2 * KB * R))
            for c in range(RC):
                for hf in range(2):
                    pt = T0[(g * RC * 2 + c * 2 + hf) % 4]
                    for i in range(4):
                        nisa.nc_matmul(dst=pt[:, i * 128:(i + 1) * 128], stationary=kin[:, hf * 4 + i, c * 128:(c + 1) * 128],
                                       moving=IB, accumulate=False)
                    _copy(kts[:, c, hf * KB:(hf + 1) * KB], pt, c + hf)
            nisa.dma_copy(dst=kth.ap(pattern=[[2 * KB, 128], [128 * 2 * KB, RC], [1, 2 * KB]], offset=g * RC * 128 * 2 * KB),
                          src=kts)
        if LT:
            nt = LT // 128
            nisa.dma_copy(dst=KRt[:, 0:nt, :], src=kc.ap(pattern=[[R, 128], [128 * R, nt], [1, R]], offset=NP * 2 * KB * R))
            for c in range(RC):
                pt = T0[c % 4]
                for i in range(nt):
                    nisa.nc_matmul(dst=pt[:, i * 128:(i + 1) * 128], stationary=KRt[:, i, c * 128:(c + 1) * 128],
                                   moving=IB, accumulate=False)
                _copy(KTt[c][:, 0:LT], pt[:, 0:LT], c)
        nps = _sb((1, C // 128 // qpass), I32)
        nisa.dma_copy(dst=nps, src=npair)
        X = dict(H=H, RC=RC, R=R, L=L, scale=scale, IB=IB, kth=kth, tab=tab, maskf=mask.reshape((C * L, 1)),
                 kcf=kc.reshape((L * R, 1)),
                 QT=_ring(qpass, (128, H, RC * 128), BF16), ACC=_ring(qpass * H, (128, R)), Lr=_ring(qpass * H, (128, 1)),
                 M=_ring(2 * qpass * H, (128, 1)), KTb=KTb, KRb=KRb,
                 MKb=_ring(qpass, (128, 2 * KB), BF16), offs=_sb((1, 2), I32), Pr=_ring(3, (128, KB), BF16),
                 PT=_ring(3, (128, KB), BF16), BM=_ring(NR, (128, 1)), AL=_ring(NR, (128, 1)), NB=_ring(NR, (128, 1)),
                 RS=_ring(NR, (128, 1)))
        Qin = _sb((128, H, R), BF16)
        MKt = _ring(qpass, (128, 256), BF16)
        OB = _ring(2, (128, R))
        RL = _sb((128, 1))
        for ps_ in range(C // 128 // qpass):
            qtiles = []
            for qi in range(qpass):
                qtiles.append(ps_ * qpass + qi)
            T1 = _t4()
            for qi in range(qpass):  # the pass's q_lat^T and fresh statistics
                q0 = qtiles[qi] * 128
                nisa.dma_copy(dst=Qin, src=q_lat[q0:q0 + 128, :, :])
                for h in range(H):
                    pt = T1[(qi * H + h) % 4]
                    for c in range(RC):
                        nisa.nc_matmul(dst=pt[:, c * 128:(c + 1) * 128], stationary=Qin[:, h, c * 128:(c + 1) * 128],
                                       moving=IB, accumulate=False)
                    _copy(X["QT"][qi][:, h, :], pt[:, 0:RC * 128], h)
                    k = qi * H + h
                    nisa.memset(dst=X["ACC"][k], value=0.0)
                    nisa.memset(dst=X["M"][2 * k + 1], value=NEG_INF)
                    nisa.memset(dst=X["Lr"][k], value=0.0)
            rk = nisa.register_alloc()
            nisa.register_load(rk, nps.ap(pattern=[[C // 128 // qpass, 1], [1, 1]], offset=ps_))

            def body(it, qtiles=qtiles):
                _pair_body(X, it, qtiles)

            nl.fori_loop(0, rk, body)
            if LT:  # the tail block, static
                for qi in range(qpass):
                    q0 = qtiles[qi] * 128
                    nisa.dma_copy(dst=MKt[qi][:, 0:LT], src=mask[q0:q0 + 128, NP * 2 * KB:L])
                P_ = _region(X)
                ktl = []
                for c in range(RC):
                    ktl.append(KTt[c][:, 0:LT])
                mkl = []
                for qi in range(qpass):
                    mkl.append([MKt[qi][:, 0:LT]])
                _pipeline(P_, _pair_units(X, qpass, 1, LT, [ktl], [KRt], mkl))
            for qi in range(qpass):
                q0 = qtiles[qi] * 128
                for h in range(H):
                    k = qi * H + h
                    ob = OB[k % 2]
                    nisa.reciprocal(dst=RL, data=X["Lr"][k])
                    nisa.tensor_scalar(dst=ob, data=X["ACC"][k], op0=nl.multiply, operand0=RL, engine=VE)
                    nisa.dma_copy(dst=o[q0:q0 + 128, h, :], src=ob)
        return o
else:
    kiln_dsa_prefill_kernel = kiln_dsa_prefill_loop_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source, passed as the static argument `rev` (LNL's compile-cache key does
    not include NKI kernel source: CLAUDE.md, kernels/delta_rule.py _kernel_rev)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_dsa_prefill_kernel = kiln_dsa_prefill_loop_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev()


def supported(C: int, L: int, R: int) -> bool:
    """Shapes the kernel takes: any number of queries (padded to whole 128-row tiles), whole 128-key tiles, a latent
    of 128-multiples up to 512."""
    return C >= 1 and L % 128 == 0 and R % 128 == 0 and R <= 512


QPASS = int(os.environ.get("KILN_DSA_PREFILL_QPASS", 2))  # query tiles per pass of the causal form


def loop_args(pos: torch.Tensor, C: int, L: int, R: int):
    """The causal form's npair int32 [1, C / 128 / QPASS] (per pass of QPASS query tiles: the number of 1024-key pairs
    holding a key at or before its last query's position, at most L // 1024; counted by comparisons, as in-graph
    integer division is inexact on trn1: docs/neuron-notes.md) and tab int32 [L // 1024, 2] (each pair's first key and
    first latent element)."""
    NP = L // (2 * KB)
    npass = C // 128 // QPASS
    starts = torch.arange(NP, device=pos.device, dtype=torch.int32) * (2 * KB)
    last = pos.to(torch.int32).reshape(npass, QPASS * 128).amax(-1, keepdim=True)  # [npass, 1]
    npair = (starts.view(1, NP) <= last).to(torch.int32).sum(-1).view(1, npass).to(torch.int32)
    tab = torch.stack([starts, starts * R], dim=1).contiguous()
    return npair, tab


def attend(q_lat: torch.Tensor, kc: torch.Tensor, mask: torch.Tensor, scale: float, pos: torch.Tensor | None = None
           ) -> torch.Tensor:
    """o [C, H, R] fp32 (the kernel on a Neuron device, emulate() elsewhere): q_lat [C, H, R], kc the chunk's latent
    rows [L, R] (or [L, 1, R]), mask [C, L] additive (0 / NEG_INF). pos [C] (the queries' positions, keys in position
    order) and KILN_DSA_PREFILL_LOOP=1: the causal form, which skips the pairs of key blocks past each pass's last
    query."""
    kc2 = kc.reshape(kc.shape[0], -1)
    if q_lat.device.type == "cpu":
        return emulate(q_lat, kc2, mask, scale)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if kiln_dsa_prefill_kernel is None:
        raise RuntimeError("the NKI DSA prefill kernel needs the nki package (the Neuron venv)")
    C, L, R = q_lat.shape[0], kc2.shape[0], kc2.shape[1]
    loop = LOOP and pos is not None and loop_supported(L)
    # Query rows padded to whole tiles (whole passes for the causal form): a padded row attends nothing (its mask row
    # is all NEG_INF, so its P is 1 everywhere and its finite output is dropped) and sits at position 0.
    tile = 128 * (QPASS if loop else 1)
    Cp = -(-C // tile) * tile
    ql, mk = q_lat.to(torch.bfloat16), mask.to(torch.bfloat16)
    if Cp != C:
        ql = torch.cat([ql, ql.new_zeros(Cp - C, *ql.shape[1:])])
        mk = torch.cat([mk, torch.full((Cp - C, L), NEG_INF, dtype=mk.dtype, device=mk.device)])
        if pos is not None:
            pos = torch.cat([pos, pos.new_zeros(Cp - C)])
    eye = torch.eye(128, device=q_lat.device).to(torch.bfloat16)
    args = dict(q_lat=ql.contiguous(), kc=kc2.to(torch.bfloat16).contiguous(), mask=mk.contiguous(), identb=eye)
    if loop:
        npair, tab = loop_args(pos, Cp, L, R)
        o = wrap_nki(kiln_dsa_prefill_loop_kernel)[platform.nki_grid()](
            **args, npair=npair, tab=tab, scale=float(scale), qpass=QPASS, rev=REV)
    else:
        o = wrap_nki(kiln_dsa_prefill_kernel)[platform.nki_grid()](**args, scale=float(scale), rev=REV)
    return o[:C] if Cp != C else o


def loop_supported(L: int) -> bool:
    """The causal form's key shapes: at least one full pair of blocks, a tail of at most 256 keys."""
    return L >= 2 * KB and L % (2 * KB) <= 256


LOOP = os.environ.get("KILN_DSA_PREFILL_LOOP", "0") == "1"
# KILN_DSA_PREFILL_KERNEL: nki (the kernel) or xla (models/mla.py _core). Default: nki on trn1, where it was measured
# (GLM-5.3-Flash tp=32 / DP attention 4 on trn1.32xlarge, 2026-10-05: G64 122.9 -> 127.6 and F0 112.5 -> 116.5 out tok/s,
# wikitext -0.547 -> -0.548; docs/neuron-notes.md "The attention kernels against their floors"), xla elsewhere (trn2,
# inf2 and a host without a Neuron device: the CPU tests take the kernel path only when they ask for it).
# trn2 too since 2026-10-06, for the fused kernel (kernels/dsa_fused.py FUSED_FAMILIES), which needs this one on.
PREFILL_KERNEL_FAMILIES = ("trn1", "trn2")


def _default_kernel() -> str:
    from .. import platform

    t = platform.target()
    return "nki" if t is not None and platform.family_of(t) in PREFILL_KERNEL_FAMILIES else "xla"


KERNEL = os.environ.get("KILN_DSA_PREFILL_KERNEL") or _default_kernel()
if KERNEL not in ("xla", "nki"):
    raise ValueError(f"KILN_DSA_PREFILL_KERNEL must be xla or nki, not {KERNEL!r}")
