"""The chunked gated delta rule (kernels/delta_rule.py) with its engine work rebalanced: a measured record, not wired
into the model (2026-10-07: -0.026 / -0.005 ms per layer at the R8 / G64 rank shapes, not worth a graph-key change;
docs/neuron-notes.md). tools/probe_delta_rule_v2.py runs it (KILN_DELTA_RULE_V=2 there). The arithmetic is delta_rule's, operation for operation, so o and the final
state are bit-identical to it (tests/test_delta_rule_v2.py in the NKI simulator, tools/probe_delta_rule_v2.py on the
device: torch.equal at the R8 and G64 rank shapes and unit groups 2 and 6); only which engine does what differs:

- cp 1: every PSUM -> SBUF fp32 copy on the scalar engine (an exact activation copy, as delta_rule already does for
  half of them) instead of the vector engine. In the R8 rank shape (4096 rows, 2 v heads, unit group 6) the vector
  engine is the busiest (union 700 us of a 1048 us window, against 668 tensor and 316 scalar), and 8 of its ops per
  unit are such copies.
- yacc 1: each update Y += A^T B of the blocked inverse formed in PSUM (I^T Y, then A^T B accumulated) and copied
  out on the scalar engine, instead of a vector-engine add of the product to Y: 6 vector adds per unit become 6
  identity matmuls and 6 scalar copies. Bit-identical only if the PSUM accumulation adds the product to Y with the
  one rounding the vector add makes; the probe checks it.

delta_rule.py's source is untouched (its REV and every default graph key stay); this module has its own REV, which
also covers delta_rule's helpers it calls.
"""

from __future__ import annotations

import os

from . import delta_rule as dr
from .delta_rule import C_BL, C_BU, C_I, C_LO, C_MZ, C_ONES, C_SU, C_U, LEVELS, NCST  # noqa: F401

ENABLED = os.environ.get("KILN_DELTA_RULE_V", "1") == "2"
CP = int(os.environ.get("KILN_DELTA_RULE_CP", "1"))
YACC = int(os.environ.get("KILN_DELTA_RULE_YACC", "0"))

try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    F32 = nl.float32
    _sb, _ps, _mm, _dve, _act_copy = dr._sb, dr._ps, dr._mm, dr._dve, dr._act_copy

    def _cp(dst, src, cp: int):
        """A PSUM -> SBUF fp32 copy: on the scalar engine (cp 1) or, as delta_rule, the vector engine."""
        if cp:
            _act_copy(dst, src)
        else:
            nisa.tensor_copy(dst=dst, src=src, engine=nisa.vector_engine)

    def _inverse2(NTs, C, cp: int, yacc: int):
        """delta_rule._inverse with cp (the PSUM copies on the scalar engine instead of the vector engine) and yacc
        (each Y += A^T B formed in PSUM as I^T Y + A^T B, then copied, instead of a vector-engine add)."""
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
            _cp(p2, pp[:, 0, :], cp)
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
            y2 = _sb((128, 128))
            if yacc:
                _mm(py, C[:, C_I, :], Y[h])
                _mm(py, P2[h], Y[h], acc=True)  # Y + P2^T Y
                _act_copy(y2, py)
            else:
                _mm(py, P2[h], Y[h])  # P2^T Y
                _dve(y2, Y[h], py, nl.add)
            Y[h] = y2
        for h in range(n):
            py = _ps((128, 128))
            y3 = _sb((128, 128))
            if yacc:
                _mm(py, C[:, C_I, :], Y[h])
                _mm(py, P4[h], Y[h], acc=True)  # Y + P4^T Y
                _act_copy(y3, py)
            else:
                _mm(py, P4[h], Y[h])  # P4^T Y
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
                _cp(rs, pr, cp)
                Xs.append(xs)
                Rs.append(rs)
            for h in range(n):
                pw = _ps((128, 128))
                yn = _sb((128, 128))
                if yacc:
                    _mm(pw, C[:, C_I, :], Y[h])
                    _mm(pw, Xs[h], Rs[h], acc=True)  # Y + Y R
                    _act_copy(yn, pw)
                else:
                    _mm(pw, Xs[h], Rs[h])  # Y R
                    _dve(yn, Y[h], pw, nl.add)
                Y[h] = yn
        return Y

    @nki.jit
    def kiln_delta_rule_v2_kernel(q, k, v, g, beta, S0, cst, kind: int, ug: int, cp: int, yacc: int, rev: int):
        """kernels/delta_rule.py's kiln_delta_rule_kernel with cp and yacc (this module's docstring); rev: REV here."""
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
                _cp(b, p1[:, 1, :], cp)
                kT.append(b)
                cc = _sb((128, 128))
                _act_copy(cc, p1[:, 2, :])
                kbT.append(cc)
                if kind == 1:
                    gg = _sb((128, 128))
                    _cp(gg, p1[:, 3, :], cp)
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
            Y = _inverse2(NT, C, cp, yacc)
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
                _cp(sn, ps_, cp)
                St[h] = sn
        for h in range(h0, h0 + HP):
            if mine:
                nisa.dma_copy(dst=S_out.ap(pattern=[[Dv, 128], [1, Dv]], offset=h * D * Dv), src=St[h])
        if npg > 1:  # LNC: both programs' heads written before either ends (whatever runs next reads all of them)
            nisa.core_barrier(data=o, cores=(0, 1))
            nisa.core_barrier(data=S_out, cores=(0, 1))
        return o, S_out

else:
    kiln_delta_rule_v2_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source, combined with delta_rule's REV (its helpers are traced into this
    kernel), passed as the static argument `rev` (kernels/delta_rule.py _kernel_rev)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_delta_rule_v2_kernel = None", a)
    return zlib.crc32(src[a:b].encode()) ^ dr.REV


REV = _kernel_rev()


def kernel_inputs(q, k, v, g, beta, S0, device=None, cp: int | None = None, yacc: int | None = None):
    """delta_rule.kernel_inputs plus this kernel's static arguments."""
    a = dr.kernel_inputs(q, k, v, g, beta, S0, device=device)
    a.update(cp=CP if cp is None else cp, yacc=YACC if yacc is None else yacc, rev=REV)
    return a


def kernel():
    if kiln_delta_rule_v2_kernel is None:
        raise RuntimeError("the NKI delta-rule kernel needs the nki package (the Neuron venv)")
    return kiln_delta_rule_v2_kernel
