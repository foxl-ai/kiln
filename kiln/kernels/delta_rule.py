"""The chunked gated delta rule (prefill of Gated DeltaNet and Kimi Delta Attention layers) as one NKI
kernel for NeuronCore-v2 (trn1) and trn2, the default prefill path (KILN_LINEAR_ATTN_KERNEL=torch turns it off; models/linear_attn.py).

What it computes is linear_attn.chunk_scan's: for every v head, from its state S [Dk, Dv],
    S_t = diag(exp(g_t)) S_{t-1} + k_t (beta_t (v_t - (diag(exp(g_t)) S_{t-1})^T k_t))^T,  o_t = S_t^T q_t
with per-channel log decays g [T, H, Dk] (KDA) or one per head g [T, H] (GDN), returning o and the
final S. Every per-chunk quantity is the chunked form of flash-linear-attention (fla, MIT,
https://github.com/fla-org/flash-linear-attention at 3e52d5a): fla/ops/kda/chunk_fwd.py chunk_kda_fwd
(intra-chunk A = beta k k^T and Aqk = q k^T under the decay, chunk_kda_fwd_intra in
fla/ops/kda/chunk_intra.py; the UT transform T = (I + A)^-1 and w = T (beta k exp(G)), u = T (beta v),
fla/ops/kda/wy_fast.py recompute_w_u_fwd and fla/ops/utils/solve_tril.py; the state recurrence
fla/ops/common/chunk_delta_h.py chunk_gated_delta_rule_fwd_h; the output fla/ops/gla/chunk.py
chunk_gla_fwd_o_gk), which vLLM v0.30.0 (vllm/third_party/flash_linear_attention/ops/kda.py,
vllm/models/glm5next/nvidia/kda.py: chunk_kda_with_fused_gate(..., safe_gate=True)) and SGLang
v0.5.21 (python/sglang/kernels/ops/attention/fla/kda.py, chunk_intra.py, chunk_delta_h.py,
wy_fast.py; GDN: fla/chunk.py) carry as Triton kernels. What differs here is how each piece is laid
onto a 128 x 128 systolic array, vector and scalar engines:

- Chunks of L = 128 tokens (fla uses 64): every chunk matrix is one [128, 128] tile. All matmuls
  are float32 (measured exact to 1.6e-7 relative on trn1, tools/probe_delta_prims.py), a transpose is
  a matmul with the identity (exact), exp runs on the scalar engine (1.5e-5 relative, same probe).
- KDA's decayed products A_ij = sum_d (beta k_i)_d k_jd exp(G_id - G_jd) are matmuls over d after
  factoring exp(G_i - G_j) = exp(G_i - G_r) exp(G_r - G_j) through a reference row r. One reference
  per chunk overflows; as fla (chunk_kda_fwd_kernel_inter_solve_fused: b_gn = g at the first row of
  the row sub-chunk; chunk_kda_fwd_kernel_intra_sub_chunk under safe_gate: one matmul for the
  diagonal sub-chunk) the rows are cut into sub-chunks of R = 16 and sub-chunk I uses the first row
  r = 16 I for all of its columns j <= 16 I + 15. For j < r both factors are <= 1; inside the
  diagonal sub-chunk exp(G_r - G_j) <= exp(15 |lower_bound|), finite in fp32 only because the
  "safe gate" bounds every log decay below (GLM-5.3-Flash and Kimi K3: lower bound -5, so <= e^75).
  A KDA layer without a lower bound (Kimi-Linear-48B) keeps the torch path.
- GDN's decay is a scalar per (token, head): the products are one matmul each and the decay is
  applied after, as exp(min(G_i - G_j, mask)).
- The UT transform is computed transposed, Y = T^T = (I + A^T)^-1, as linear_attn._unit_lower_inverse
  does it: 8 x 8 diagonal blocks inverted by repeated squaring ((I + V)(I + V^2)(I + V^4), exact for
  nilpotent blocks of 8 and numerically safe there), then four merge levels Y += Y (V o Lo_s) Y
  (fla's solve_tril merge of [[A, 0], [C, D]]), with block masks as constants: 18 matmuls per chunk.
- The state recurrence takes the chunk as one affine map, S' = P S + Q with P = diag(exp(G_L))
  - (k exp(G_L - G))^T w and Q = (k exp(G_L - G))^T u, and o = (q exp(G) - Aqk w) S + Aqk u: the same
  algebra as chunk_delta_h's v_new = u - w S, S' = exp(G_L) S + k_decayed^T v_new, but every term that
  does not involve S is computed before S is known, so the sequential part per chunk is two matmuls
  (P S + Q, M S + Aqk u) instead of three plus elementwise steps.

Inputs are the torch path's (linear_attn.mixer, after the short conv, the l2 norms, the gate
activations and q's scale), fp32: q, k [T, Hk, Dk], v [T, Hv, Dv], g [T, Hv, Dk] or [T, Hv], beta
[T, Hv], S0 [Hv, Dk, Dv]; Dk = Dv = 128 and Hv a multiple of Hk (GDN: v head h reads k head
h // (Hv / Hk), the torch path's repeat_interleave). T is padded to whole chunks by the caller with
beta = g = 0, which leaves S unchanged. emulate() is this kernel's arithmetic in torch.
"""

from __future__ import annotations

import os

import numpy as np
import torch

L = 128  # tokens per chunk (one tile)
R = 16  # rows per reference sub-chunk (KDA)
B8 = 8  # diagonal blocks of the inverse inverted by squaring
LEVELS = (8, 16, 32, 64)  # merge levels of the inverse
KINDS = {"gdn": 0, "kda": 1}
# Constant tiles (consts()): index of each in the [128, NCST, 128] block.
C_I, C_U, C_SU, C_BL, C_BU, C_LO, C_ONES, C_MZ = 0, 1, 2, 3, 4, 5, 9, 10
NCST = 11
NEG = -1.0e30


def consts_np() -> np.ndarray:
    """[128, NCST, 128] fp32, tile c at [:, c, :] (partition p, column q):
    I; U [p <= q]; SU [p > q]; BL [same 8-block, p > q] (strict lower within the diagonal blocks of
    8, [i, j] layout); BU = BL^T; LO_s for s in LEVELS: [same 2s-block, p in its lower half, q in its
    upper half] ([i, j]); ONES; MZ: 0 where p <= q, -1e30 elsewhere."""
    r = np.arange(L)
    p, q = r[:, None], r[None, :]
    tiles = [p == q, p <= q, p > q, (p // B8 == q // B8) & (p > q), (p // B8 == q // B8) & (p < q)]
    for s in LEVELS:
        tiles.append((p // (2 * s) == q // (2 * s)) & (p // s % 2 == 1) & (q // s % 2 == 0))
    tiles.append(np.ones((L, L), dtype=bool))
    out = np.stack([t.astype(np.float32) for t in tiles] + [np.where(p <= q, 0.0, NEG).astype(np.float32)], axis=1)
    assert out.shape == (L, NCST, L)
    return np.ascontiguousarray(out)


def consts(device) -> torch.Tensor:
    """consts_np() as a tensor (inside a traced graph a host constant, as linear_attn's masks)."""
    return torch.tensor(consts_np(), device=device)


# --- the kernel's arithmetic in torch -------------------------------------------------------------


def _inverse_t(NT: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Y = (I - NT)^-1 for strictly upper NT [j, i] = -A^T (entries on or below the diagonal are
    ignored), the kernel's way: N = NT^T; the 8 x 8 diagonal blocks by Y = (I + Vb)(I + Vb^2)(I +
    Vb^4) with Vb = NT o BU; then for s in LEVELS: Y += Y (NT o LO_s^T) Y. c = consts as a tensor."""
    I, BL, BU = c[:, C_I], c[:, C_BL], c[:, C_BU]
    N = NT.transpose(-1, -2)
    Nb, NbT = N * BL, NT * BU
    P2, P4 = Nb @ Nb, (Nb @ Nb) @ (Nb @ Nb)
    Y = I + NbT
    Y = Y + P2.transpose(-1, -2) @ Y
    Y = Y + P4.transpose(-1, -2) @ Y
    for i, _s in enumerate(LEVELS):
        V = (N * c[:, C_LO + i]).transpose(-1, -2)
        Y = Y + Y @ (V @ Y)
    return Y


def emulate(q, k, v, g, beta, S, chunk: int = L):
    """The kernel's algorithm in float32 torch (padding included): returns o [T, Hv, Dv] and the
    final S [Hv, Dk, Dv]. Checks the factorisation (reference sub-chunks, the transposed blocked
    inverse, the affine chunk map) on the host; nki.simulate and the device check the kernel."""
    T, Hk, Dk = k.shape
    Hv = v.shape[1]
    per_channel = g.dim() == 3
    pad = -T % chunk
    if pad:
        q, k, v, g, beta = (torch.cat([x, x.new_zeros((pad, *x.shape[1:]))]) for x in (q, k, v, g, beta))
    c = torch.tensor(consts_np())
    I, U, SU = c[:, C_I], c[:, C_U], c[:, C_SU]
    rep = Hv // Hk
    S = S.clone().float()
    outs = torch.zeros(T + pad, Hv, v.shape[2])
    for ci in range((T + pad) // chunk):
        sl = slice(ci * chunk, (ci + 1) * chunk)
        for h in range(Hv):
            qt, kt = q[sl, h // rep].float(), k[sl, h // rep].float()
            vt, bt = v[sl, h].float(), beta[sl, h].float().unsqueeze(-1)
            kb = kt * bt
            if per_channel:
                gt = g[sl, h].float()
                GT = gt.T @ U  # [d, i] cumulative log decay
                Gtok, Rtok = U.T @ gt, SU.T @ gt  # [t, d]: G and G_L - G
                EQ = torch.empty(Dk, chunk)
                ZA, ZQ = torch.zeros(chunk, chunk), torch.zeros(chunk, chunk)
                for I_ in range(chunk // R):
                    r, cols = I_ * R, slice(I_ * R, (I_ + 1) * R)
                    EQ[:, cols] = torch.exp(GT[:, cols] - GT[:, r : r + 1])
                    kg = torch.zeros(Dk, chunk)
                    kg[:, : r + R] = kt.T[:, : r + R] * torch.exp(GT[:, r : r + 1] - GT[:, : r + R])
                    ZA[:, cols] = kg.T @ (kb.T[:, cols] * EQ[:, cols])
                    ZQ[:, cols] = kg.T @ (qt.T[:, cols] * EQ[:, cols])
                QkT, NT = ZQ * U, -ZA
                qGT = qt.T * torch.exp(GT)
                Gam, Dkd = torch.exp(Gtok), torch.exp(Rtok)
                GL = torch.exp(GT[:, -1:])  # [d, 1]
                kbG, kd = kb * Gam, kt * Dkd
            else:
                gc = g[sl, h].float().unsqueeze(-1)  # [t, 1]
                E = c[:, C_ONES].T @ (U * gc)  # [j, i] = G_i
                Gc = U.T @ gc  # [t, 1]
                DEC = torch.exp(torch.minimum(E - Gc, c[:, C_MZ]))
                Z = torch.cat([kt @ kb.T, kt @ qt.T], 1)  # [j, (i | i)]
                NT, QkT = -Z[:, :chunk] * DEC, Z[:, chunk:] * DEC
                GamB = torch.exp(E)
                qGT = qt.T * GamB
                GL = GamB[:, -1:]  # exp(G_L) on every partition
                kbG, kd = kb * torch.exp(Gc), kt * DEC[:, -1:]
            Y = _inverse_t(NT, c)
            UW = Y.T @ torch.cat([vt * bt, kbG], 1)
            u, w = UW[:, : v.shape[2]], UW[:, v.shape[2] :]
            NPT = I * GL - w.T @ kd
            MT = qGT - w.T @ QkT
            outs[sl, h] = QkT.T @ u + MT.T @ S[h]
            S[h] = kd.T @ u + NPT.T @ S[h]
    return outs[:T], S


# --- kernel -------------------------------------------------------------------------------------

try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    F32 = nl.float32

    def _sb(shape):
        return nl.ndarray(shape, dtype=F32, buffer=nl.sbuf)

    def _ps(shape):
        return nl.ndarray(shape, dtype=F32, buffer=nl.psum)

    def _mm(dst, st, mv, acc=False):
        nisa.nc_matmul(dst=dst, stationary=st, moving=mv, accumulate=acc)

    def _dve(dst, a, b, op):
        nisa.tensor_tensor(dst=dst, data1=a, data2=b, op=op, engine=nisa.vector_engine)

    def _act_copy(dst, src, scale=1.0):
        nisa.activation(dst=dst, op=nl.copy, data=src, scale=scale)

    def _inverse(NTs, C):
        """Y = T^T for each head's NT (see _inverse_t), each step issued for every head in turn so
        the engines have independent work between dependent steps."""
        n = len(NTs)
        Ns, Nb, NbT, NL = [], [], [], []
        for h in range(n):
            pn = _ps((128, 128))
            _mm(pn, NTs[h], C[:, C_I, :])  # N = NT^T
            ns = _sb((128, 128))
            _act_copy(ns, pn)
            Ns.append(ns)
        for h in range(n):
            nb = _sb((128, 128))
            _dve(nb, Ns[h], C[:, C_BL, :], nl.multiply)
            Nb.append(nb)
            nbt = _sb((128, 128))
            _dve(nbt, NTs[h], C[:, C_BU, :], nl.multiply)
            NbT.append(nbt)
            row = []
            for i in range(len(LEVELS)):
                nl_ = _sb((128, 128))
                _dve(nl_, Ns[h], C[:, C_LO + i, :], nl.multiply)
                row.append(nl_)
            NL.append(row)
        P2, P2T, P4, Y = [], [], [], []
        for h in range(n):
            pp = _ps((128, 2, 128))
            _mm(pp[:, 0, :], NbT[h], Nb[h])  # Nb @ Nb
            _mm(pp[:, 1, :], Nb[h], NbT[h])  # NbT @ NbT
            p2 = _sb((128, 128))
            nisa.tensor_copy(dst=p2, src=pp[:, 0, :], engine=nisa.vector_engine)
            p2t = _sb((128, 128))
            _act_copy(p2t, pp[:, 1, :])
            P2.append(p2)
            P2T.append(p2t)
            y1 = _sb((128, 128))
            _dve(y1, NbT[h], C[:, C_I, :], nl.add)  # (a GpSimd tensor_tensor failed NCC_IXCG965 on trn1)
            Y.append(y1)
        for h in range(n):
            p4p = _ps((128, 128))
            _mm(p4p, P2T[h], P2[h])  # P2 @ P2
            p4 = _sb((128, 128))
            _act_copy(p4, p4p)
            P4.append(p4)
            py = _ps((128, 128))
            _mm(py, P2[h], Y[h])  # P2^T Y
            y2 = _sb((128, 128))
            _dve(y2, Y[h], py, nl.add)
            Y[h] = y2
        for h in range(n):
            py = _ps((128, 128))
            _mm(py, P4[h], Y[h])  # P4^T Y
            y3 = _sb((128, 128))
            _dve(y3, Y[h], py, nl.add)
            Y[h] = y3
        for i in range(len(LEVELS)):
            Xs, Rs = [], []
            for h in range(n):
                px = _ps((128, 128))
                _mm(px, Y[h], C[:, C_I, :])  # X = Y^T
                pr = _ps((128, 128))
                _mm(pr, NL[h][i], Y[h])  # (N o LO)^T Y
                xs = _sb((128, 128))
                _act_copy(xs, px)
                rs = _sb((128, 128))
                nisa.tensor_copy(dst=rs, src=pr, engine=nisa.vector_engine)
                Xs.append(xs)
                Rs.append(rs)
            for h in range(n):
                pw = _ps((128, 128))
                _mm(pw, Xs[h], Rs[h])  # Y R
                yn = _sb((128, 128))
                _dve(yn, Y[h], pw, nl.add)
                Y[h] = yn
        return Y

    @nki.jit
    def kiln_delta_rule_kernel(q, k, v, g, beta, S0, cst, kind: int, ug: int, rev: int):
        """See the module docstring. kind 1: KDA (g [T, Hv, 128]), 0: GDN (g [T, Hv]); ug: units
        (chunk, v head) processed together (their instructions interleaved); rev: this module's source revision
        (REV, see _kernel_rev). Returns o fp32 [T, Hv, Dv] and the final state fp32 [Hv, Dk, Dv]."""
        T, Hk, D = q.shape
        Hv, Dv = v.shape[1], v.shape[2]
        rep = Hv // Hk
        NCH = T // 128
        o = nl.ndarray((T, Hv, Dv), dtype=F32, buffer=nl.shared_hbm)
        S_out = nl.ndarray((Hv, D, Dv), dtype=F32, buffer=nl.shared_hbm)
        C = _sb((128, NCST, 128))
        nisa.dma_copy(dst=C, src=cst)
        I = C[:, C_I, :]
        U = C[:, C_U, :]
        # LNC (trn2 at LNC=2, grid 2): traced once per program (nki/_backends/mlir_tracer program_id), so
        # npg / pid are Python ints; each program (physical core) runs its own v heads, h0 .. h0 + HP - 1,
        # and writes only their o columns and states (disjoint, no barrier). Grid 1: every head, as before.
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        HP = Hv // npg if Hv % npg == 0 else Hv
        h0 = pid * HP if Hv % npg == 0 else 0
        mine = Hv % npg == 0 or pid == 0  # heads that do not split evenly all run on program 0
        St = [None] * Hv  # a list (the NKI tracer takes only str keys in a dict)
        for h in range(h0, h0 + HP):
            if mine:
                s0 = _sb((128, Dv))
                nisa.dma_copy(dst=s0, src=S0.ap(pattern=[[Dv, 128], [1, Dv]], offset=h * D * Dv))
                St[h] = s0
        # Units (chunk c, v head h) in chunk-major order, ug at a time: every step below is issued for
        # each unit of a group in turn, so the in-order engine queues hold independent work between
        # dependent steps; the chain (o, S) runs in unit order, chunk c of a head before chunk c + 1.
        NU = NCH * HP if mine else 0
        for u0 in range(0, NU, ug):
            heads = []
            t0s = []
            for u in range(u0, min(u0 + ug, NU)):
                heads.append(h0 + u % HP)
                t0s.append((u // HP) * 128)
            n = len(heads)
            # Loads (token-major tiles: rows are tokens).
            qt, kt, vt, bt, gt = [], [], [], [], []
            for j in range(n):
                h = heads[j]
                hk = h // rep
                a = _sb((128, D))
                nisa.dma_copy(dst=a, src=q.ap(pattern=[[Hk * D, 128], [1, D]], offset=t0s[j] * Hk * D + hk * D))
                qt.append(a)
                b = _sb((128, D))
                nisa.dma_copy(dst=b, src=k.ap(pattern=[[Hk * D, 128], [1, D]], offset=t0s[j] * Hk * D + hk * D))
                kt.append(b)
                vv = _sb((128, Dv))
                nisa.dma_copy(dst=vv, src=v.ap(pattern=[[Hv * Dv, 128], [1, Dv]], offset=t0s[j] * Hv * Dv + h * Dv))
                vt.append(vv)
                bb = _sb((128, 1))
                nisa.dma_copy(dst=bb, src=beta.ap(pattern=[[Hv, 128], [1, 1]], offset=t0s[j] * Hv + h))
                bt.append(bb)
                if kind == 1:
                    gg = _sb((128, D))
                    nisa.dma_copy(dst=gg, src=g.ap(pattern=[[Hv * D, 128], [1, D]], offset=t0s[j] * Hv * D + h * D))
                else:
                    gg = _sb((128, 1))
                    nisa.dma_copy(dst=gg, src=g.ap(pattern=[[Hv, 128], [1, 1]], offset=t0s[j] * Hv + h))
                gt.append(gg)
            # Token-major scaled operands; RHS = [beta v | beta k exp(G)].
            kb, RHS = [], []
            for j in range(n):
                a = _sb((128, D))
                nisa.tensor_scalar(dst=a, data=kt[j], op0=nl.multiply, operand0=bt[j], engine=nisa.vector_engine)
                kb.append(a)
                rh = _sb((128, 2, 128))
                _act_copy(rh[:, 0, :], vt[j], scale=bt[j])
                RHS.append(rh)
            # d-major transposes (and KDA's cumulative decay G^T = g^T U) on the tensor engine.
            qT, kT, kbT, GT = [], [], [], []
            for j in range(n):
                p1 = _ps((128, 4, 128))
                _mm(p1[:, 0, :], qt[j], I)
                _mm(p1[:, 1, :], kt[j], I)
                _mm(p1[:, 2, :], kb[j], I)
                if kind == 1:
                    _mm(p1[:, 3, :], gt[j], U)
                a = _sb((128, 128))
                _act_copy(a, p1[:, 0, :])
                qT.append(a)
                b = _sb((128, 128))
                nisa.tensor_copy(dst=b, src=p1[:, 1, :], engine=nisa.vector_engine)
                kT.append(b)
                cc = _sb((128, 128))
                _act_copy(cc, p1[:, 2, :])
                kbT.append(cc)
                if kind == 1:
                    gg = _sb((128, 128))
                    nisa.tensor_copy(dst=gg, src=p1[:, 3, :], engine=nisa.vector_engine)
                    GT.append(gg)
            NT, QkT, qGT, kd, GL = [], [], [], [], []
            if kind == 1:
                for j in range(n):
                    # Reference rows: EQ[:, i] = exp(G_i - G_r(i)); kg_I = k^T exp(G_r - G_j), j <= r + 15.
                    ngr = _sb((128, 8))
                    nisa.tensor_scalar(dst=ngr, data=GT[j].ap(pattern=[[128, 128], [16, 8]], offset=0),
                                       op0=nl.multiply, operand0=-1.0, engine=nisa.vector_engine)
                    eq = _sb((128, 128))
                    for b_ in range(8):
                        nisa.activation(dst=eq[:, b_ * 16:(b_ + 1) * 16], op=nl.exp,
                                        data=GT[j][:, b_ * 16:(b_ + 1) * 16], bias=ngr[:, b_:b_ + 1], scale=1.0)
                    kq = _sb((128, 2, 128))  # [kb^T EQ | q^T EQ]
                    _dve(kq[:, 0, :], kbT[j], eq, nl.multiply)
                    _dve(kq[:, 1, :], qT[j], eq, nl.multiply)
                    z = _ps((128, 2, 128))
                    for b_ in range(8):
                        w_ = (b_ + 1) * 16
                        ek = _sb((128, 128))
                        nisa.activation(dst=ek[:, 0:w_], op=nl.exp, data=GT[j][:, 0:w_],
                                        bias=GT[j][:, b_ * 16:b_ * 16 + 1], scale=-1.0)
                        kg = _sb((128, 128))
                        if b_ < 7:
                            nisa.memset(dst=kg[:, w_:128], value=0.0, engine=nisa.gpsimd_engine)
                        _dve(kg[:, 0:w_], kT[j][:, 0:w_], ek[:, 0:w_], nl.multiply)
                        _mm(z[:, :, b_ * 16:(b_ + 1) * 16], kg, kq[:, :, b_ * 16:(b_ + 1) * 16])
                    qk = _sb((128, 128))
                    _dve(qk, z[:, 1, :], U, nl.multiply)
                    QkT.append(qk)
                    nt = _sb((128, 128))
                    _act_copy(nt, z[:, 0, :], scale=-1.0)
                    NT.append(nt)
                for j in range(n):
                    gam = _sb((128, 128))
                    nisa.activation(dst=gam, op=nl.exp, data=GT[j])
                    qg = _sb((128, 128))
                    _dve(qg, qT[j], gam, nl.multiply)
                    qGT.append(qg)
                    gl = _sb((128, 1))
                    nisa.activation(dst=gl, op=nl.exp, data=GT[j][:, 127:128])
                    GL.append(gl)
                    p2 = _ps((128, 2, 128))
                    _mm(p2[:, 0, :], U, gt[j])  # G token-major
                    _mm(p2[:, 1, :], C[:, C_SU, :], gt[j])  # G_L - G
                    eg = _sb((128, 2, 128))
                    nisa.activation(dst=eg, op=nl.exp, data=p2)
                    _dve(RHS[j][:, 1, :], kb[j], eg[:, 0, :], nl.multiply)
                    kdd = _sb((128, 128))
                    _dve(kdd, kt[j], eg[:, 1, :], nl.multiply)
                    kd.append(kdd)
            else:
                for j in range(n):
                    gu = _sb((128, 128))
                    nisa.tensor_scalar(dst=gu, data=U, op0=nl.multiply, operand0=gt[j], engine=nisa.vector_engine)
                    pe = _ps((128, 128))
                    _mm(pe, C[:, C_ONES, :], gu)  # [j, i] = G_i
                    pg = _ps((128, 1))
                    _mm(pg, U, gt[j])  # G_t
                    gc = _sb((128, 1))
                    nisa.tensor_copy(dst=gc, src=pg, engine=nisa.vector_engine)
                    em = _sb((128, 128))
                    nisa.scalar_tensor_tensor(dst=em, data=pe, op0=nl.subtract, operand0=gc, op1=nl.minimum,
                                              operand1=C[:, C_MZ, :])
                    dec = _sb((128, 128))
                    nisa.activation(dst=dec, op=nl.exp, data=em)
                    gb = _sb((128, 128))
                    nisa.activation(dst=gb, op=nl.exp, data=pe)
                    GL.append(gb[:, 127:128])
                    gm = _sb((128, 1))
                    nisa.activation(dst=gm, op=nl.exp, data=gc)
                    nisa.tensor_scalar(dst=RHS[j][:, 1, :], data=kb[j], op0=nl.multiply, operand0=gm,
                                       engine=nisa.vector_engine)
                    kdd = _sb((128, 128))
                    nisa.tensor_scalar(dst=kdd, data=kt[j], op0=nl.multiply, operand0=dec[:, 127:128],
                                       engine=nisa.vector_engine)
                    kd.append(kdd)
                    kq = _sb((128, 2, 128))
                    _act_copy(kq[:, 0, :], kbT[j])
                    _act_copy(kq[:, 1, :], qT[j])
                    z = _ps((128, 2, 128))
                    _mm(z, kT[j], kq)  # [k_j . beta k_i | k_j . q_i]
                    nt = _sb((128, 128))
                    nisa.scalar_tensor_tensor(dst=nt, data=z[:, 0, :], op0=nl.multiply, operand0=-1.0,
                                              op1=nl.multiply, operand1=dec)
                    NT.append(nt)
                    qk = _sb((128, 128))
                    _dve(qk, z[:, 1, :], dec, nl.multiply)
                    QkT.append(qk)
                    qg = _sb((128, 128))
                    _dve(qg, qT[j], gb, nl.multiply)
                    qGT.append(qg)
            Y = _inverse(NT, C)
            # u | w, then every term of the chunk's affine map that does not involve S.
            UW, NPT, MT = [], [], []
            for j in range(n):
                pu = _ps((128, 2, 128))
                _mm(pu, Y[j], RHS[j])  # T [beta v | beta k exp(G)]
                uw = _sb((128, 2, 128))
                _act_copy(uw, pu)
                UW.append(uw)
            for j in range(n):
                pn = _ps((128, 128))
                _mm(pn, UW[j][:, 1, :], kd[j])  # w^T kd [d', d]
                npt = _sb((128, 128))
                nisa.scalar_tensor_tensor(dst=npt, data=I, op0=nl.multiply, operand0=GL[j], op1=nl.subtract,
                                          operand1=pn)
                NPT.append(npt)
                pm = _ps((128, 128))
                _mm(pm, UW[j][:, 1, :], QkT[j])  # w^T Aqk^T [d, i]
                mt = _sb((128, 128))
                _dve(mt, qGT[j], pm, nl.subtract)
                MT.append(mt)
            # The sequential part: o = M S + Aqk u, S' = P S + Q.
            for j in range(n):
                h = heads[j]
                po = _ps((128, Dv))
                _mm(po, QkT[j], UW[j][:, 0, :])
                _mm(po, MT[j], St[h], acc=True)
                ps_ = _ps((128, Dv))
                _mm(ps_, kd[j], UW[j][:, 0, :])
                _mm(ps_, NPT[j], St[h], acc=True)
                ob = _sb((128, Dv))
                _act_copy(ob, po)
                nisa.dma_copy(dst=o.ap(pattern=[[Hv * Dv, 128], [1, Dv]], offset=t0s[j] * Hv * Dv + h * Dv), src=ob)
                sn = _sb((128, Dv))
                nisa.tensor_copy(dst=sn, src=ps_, engine=nisa.vector_engine)
                St[h] = sn
        for h in range(h0, h0 + HP):
            if mine:
                nisa.dma_copy(dst=S_out.ap(pattern=[[Dv, 128], [1, Dv]], offset=h * D * Dv), src=St[h])
        if npg > 1:  # LNC: both programs' heads written before either ends (whatever runs next reads all of them)
            nisa.core_barrier(data=o, cores=(0, 1))
            nisa.core_barrier(data=S_out, cores=(0, 1))
        return o, S_out

    @nki.jit
    def kiln_delta_rule_null_kernel(q, k, v, g, beta, S0):
        """Probe only (KILN_DELTA_RULE_NULL=1): the kernel's operands and results with no work, o = v
        and S = S0, to tell the cost of an NKI call in a layer graph from the cost of its body."""
        T, Hv, Dv = v.shape
        D = S0.shape[1]
        o = nl.ndarray((T, Hv, Dv), dtype=F32, buffer=nl.shared_hbm)
        S_out = nl.ndarray((Hv, D, Dv), dtype=F32, buffer=nl.shared_hbm)
        for c in range(T // 128):
            t = _sb((128, Hv * Dv))
            nisa.dma_copy(dst=t, src=v.ap(pattern=[[Hv * Dv, 128], [1, Hv * Dv]], offset=c * 128 * Hv * Dv))
            nisa.dma_copy(dst=o.ap(pattern=[[Hv * Dv, 128], [1, Hv * Dv]], offset=c * 128 * Hv * Dv), src=t)
        for h in range(Hv):
            t = _sb((128, Dv))
            nisa.dma_copy(dst=t, src=S0.ap(pattern=[[Dv, 128], [1, Dv]], offset=h * D * Dv))
            nisa.dma_copy(dst=S_out.ap(pattern=[[Dv, 128], [1, Dv]], offset=h * D * Dv), src=t)
        return o, S_out
else:
    kiln_delta_rule_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source (everything from the NKI import to the end of the
    kernel), passed as the static argument `rev`: LNL's graph cache key hashes the FX graph and each
    NKI call's name, operands, grid, static arguments and MAC count, not the kernel source
    (libtorch_neuronx_lite/compile/cache.py create_cache_hash, SDK 2.32; CLAUDE.md), so without it a
    kernel edit would silently run the old NEFF from a warm cache."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_delta_rule_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev()


def kernel():
    if kiln_delta_rule_kernel is None:
        raise RuntimeError("the NKI delta-rule kernel needs the nki package (the Neuron venv)")
    return kiln_delta_rule_kernel


# (chunk, v head) units interleaved. GLM-5.3-Flash KDA, 8 heads, trn1.2xlarge (tools/probe_delta_rule.py,
# 2026-10-04): C=8192 10.84 / 8.37 / 8.73 / 8.68 ms at 1 / 2 / 4 / 6 units (C=2048 2.85 / 2.24 / 2.32 / 2.28).
UNIT_GROUP = int(os.environ.get("KILN_DELTA_RULE_UNITS", 2))


def kernel_inputs(q, k, v, g, beta, S0, device=None):
    """The kernel's arguments; T must be a multiple of 128 (chunk() pads)."""
    T, Hk, Dk = k.shape
    Hv, Dv = v.shape[1:]
    if Dk != 128 or Dv != 128 or Hv % Hk:
        raise NotImplementedError(f"delta-rule kernel: Dk = Dv = 128 and Hv % Hk == 0 (got {Dk}, {Dv}, {Hk}, {Hv})")
    if T % L:
        raise ValueError(f"T = {T} is not a multiple of {L}")
    kind = KINDS["kda"] if g.dim() == 3 else KINDS["gdn"]
    return dict(q=q.float().contiguous(), k=k.float().contiguous(), v=v.float().contiguous(),
                g=g.float().contiguous(), beta=beta.float().contiguous(), S0=S0.float().contiguous(),
                cst=consts(device if device is not None else q.device), kind=kind, ug=UNIT_GROUP, rev=REV)


def chunk(q, k, v, g, beta, S0):
    """linear_attn.chunk_scan's contract on the device inside the caller's graph: o [T, Hv, Dv] and
    the final state, fp32. T is padded to whole chunks with beta = g = 0 (the state is unchanged
    by padding)."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    T = q.shape[0]
    pad = -T % L
    if pad:
        q, k, v, g, beta = (torch.cat([x, x.new_zeros((pad, *x.shape[1:]))]) for x in (q, k, v, g, beta))
    if os.environ.get("KILN_DELTA_RULE_NULL") == "1":  # probe only: no work, wrong results
        a = kernel_inputs(q, k, v, g, beta, S0)
        o, S = wrap_nki(kiln_delta_rule_null_kernel)[1](q=a["q"], k=a["k"], v=a["v"], g=a["g"], beta=a["beta"],
                                                        S0=a["S0"])
    else:
        from .. import platform

        # The runtime's LNC as the grid (platform.nki_grid: 2 on trn2, each physical core its own heads); grid 1
        # as before the split when KILN_LNC_SPLIT leaves this kernel out.
        grid = platform.nki_grid() if platform.lnc_split("delta_rule") else 1
        o, S = wrap_nki(kernel())[grid](**kernel_inputs(q, k, v, g, beta, S0))
    return (o[:T] if pad else o), S
