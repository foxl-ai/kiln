"""Device check of kiln_moe_dedupe_v10's routing pieces, one trn1 core: the lanes of the pairs (pair-major, from bf16-exact
parts), and for block 0 the route matrix R [lanes, T] and the lanes' tokens, against moe_dedupe.plan() on the host.

    python tools/probe_v10_plan.py [--rows 192] [--experts 288]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def v10_plan_probe(topi, topv, E: int, L: int, S: int, BL: int, bk: int):
    """Returns (lam f32 [128, NJ], R f32 [BL, T] of block bk, tok f32 [BL, 1])."""
    T, K = topi.shape
    N = T * K
    NJ = N // 128
    NE = (E + 127) // 128
    f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
    LOG2L = 1 if L == 2 else 2 if L == 4 else 3 if L == 8 else 4 if L == 16 else 0
    o_lam = nl.ndarray((128, NJ), dtype=f32, buffer=nl.shared_hbm)
    o_r = nl.ndarray((BL, T), dtype=f32, buffer=nl.shared_hbm)
    o_t = nl.ndarray((BL, 2), dtype=f32, buffer=nl.shared_hbm)
    ipi = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
    nisa.iota(dst=ipi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
    ip = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ip, src=ipi)
    jfi = nl.ndarray((128, 128), dtype=i32, buffer=nl.sbuf)
    nisa.iota(dst=jfi, pattern=[[1, 128]], offset=0, channel_multiplier=0)
    dd = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=dd, src=jfi)
    nisa.tensor_scalar(dst=dd, data=dd, op0=nl.subtract, operand0=ip)
    idn = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=idn, data=dd, op0=nl.equal, operand0=0.0)
    up = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=up, data=dd, op0=nl.greater, operand0=0.0)
    one_r = nl.ndarray((1, 128), dtype=f32, buffer=nl.sbuf)
    nisa.memset(dst=one_r, value=1.0)
    ones_b = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
    nisa.memset(dst=ones_b, value=1.0)
    zero_n = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
    nisa.memset(dst=zero_n, value=0.0)
    ti = nl.ndarray((1, N), dtype=i32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ti, src=topi.reshape((1, N)))
    tb = nl.ndarray((1, N), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tb, src=ti)
    eb = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
    for n0 in range(0, N, 512):
        n1 = min(N, n0 + 512)
        pb = nl.ndarray((128, n1 - n0), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pb, stationary=one_r, moving=tb[:, n0:n1], accumulate=False)
        nisa.tensor_copy(dst=eb[:, n0:n1], src=pb)
    wJ = nl.ndarray((NJ, 128), dtype=bf16, buffer=nl.sbuf)
    nisa.dma_copy(dst=wJ, src=topv.reshape((NJ, 128)))
    pwT = nl.ndarray((128, NJ), dtype=f32, buffer=nl.psum)
    nisa.nc_matmul(dst=pwT, stationary=wJ, moving=idn[0:NJ, 0:NJ], accumulate=False)
    wT = nl.ndarray((128, NJ), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=wT, src=pwT)
    p8i = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=p8i, data=ipi, op0=nl.right_shift, operand0=3)
    p8 = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=p8, src=p8i)
    t16i = nl.ndarray((128, 16), dtype=i32, buffer=nl.sbuf)
    nisa.iota(dst=t16i, pattern=[[1, 16]], offset=0, channel_multiplier=0)
    t16 = nl.ndarray((128, 16), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=t16, src=t16i)
    m16 = nl.ndarray((128, 16), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=m16, data=t16, op0=nl.equal, operand0=p8)
    Wp = nl.ndarray((128, NJ, 16), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=Wp, data1=m16.expand_dim(1).broadcast(1, NJ), data2=wT.expand_dim(2).broadcast(2, 16),
                       op=nl.multiply)
    tkp = nl.ndarray((128, NJ, 2), dtype=bf16, buffer=nl.sbuf)
    jfj = nl.ndarray((128, NJ), dtype=i32, buffer=nl.sbuf)
    nisa.iota(dst=jfj, pattern=[[1, NJ]], offset=0, channel_multiplier=0)
    nisa.tensor_copy(dst=tkp[:, :, 1], src=jfj)
    zj = nl.ndarray((128, NJ), dtype=f32, buffer=nl.sbuf)
    nisa.memset(dst=zj, value=0.0)
    nisa.tensor_scalar(dst=tkp[:, :, 0], data=zj, op0=nl.add, operand0=p8)
    ilf = nl.ndarray((128, BL), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ilf, src=jfi[:, 0:BL])
    ns_t, nb_t, pe_t, bs_t, lb_t = [], [], [], [], []
    for _ in range(NE):
        ns_t.append(nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf))
        nb_t.append(nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf))
        pe_t.append(nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf))
        bs_t.append(nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf))
        lb_t.append(nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf))
    for et in range(NE):
        nisa.tensor_scalar(dst=pe_t[et], data=ip, op0=nl.add, operand0=128.0 * et)
        cnt = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
        oh = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar_reduce(dst=oh, data=eb, op0=nl.equal, operand0=pe_t[et], reduce_op=nl.add, reduce_res=cnt)
        ci = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ci, src=cnt)
        nisa.tensor_scalar(dst=ci, data=ci, op0=nl.add, operand0=L - 1)
        nisa.tensor_scalar(dst=ci, data=ci, op0=nl.right_shift, operand0=LOG2L)
        nisa.tensor_copy(dst=ns_t[et], src=ci)
        nisa.tensor_copy(dst=nb_t[et], src=ns_t[et])
    for et in range(NE):
        pbs = nl.ndarray((128, 1), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pbs, stationary=up, moving=nb_t[et], accumulate=False)
        for t2 in range(et):
            nisa.nc_matmul(dst=pbs, stationary=ones_b, moving=nb_t[t2], accumulate=True)
        nisa.tensor_copy(dst=bs_t[et], src=pbs)
        nisa.tensor_scalar(dst=lb_t[et], data=bs_t[et], op0=nl.multiply, operand0=1.0 * L)
    vacc = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)  # every expert tile's vv summed: one nonzero per column
    for et in range(NE):
        oh = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=oh, data=eb, op0=nl.equal, operand0=pe_t[et])
        cs = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor_scan(dst=cs, data0=oh, data1=zero_n, initial=0.0, op0=nl.add, op1=nl.add)
        rk = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=rk, data=cs, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=oh)
        vv = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=vv, data=oh, op0=nl.multiply, operand0=lb_t[et], op1=nl.add, operand1=rk)
        if et == 0:
            nisa.tensor_copy(dst=vacc, src=vv)
        else:
            nisa.tensor_tensor(dst=vacc, data1=vacc, data2=vv, op=nl.add)
    # the lanes as two bf16-exact parts (lane = 64 hi + lo), each pair tile's column transposed by one matmul (no PSUM
    # accumulation across the expert tiles: interleaved accumulations into one PSUM tensor came out wrong on trn1)
    vi = nl.ndarray((128, N), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vi, src=vacc)
    hi_i = nl.ndarray((128, N), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=hi_i, data=vi, op0=nl.right_shift, operand0=6)
    vh = nl.ndarray((128, N), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vh, src=hi_i)
    lo_i = nl.ndarray((128, N), dtype=i32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lo_i, data=vi, op0=nl.bitwise_and, operand0=63)
    vl = nl.ndarray((128, N), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vl, src=lo_i)
    plh = nl.ndarray((128, NJ), dtype=f32, buffer=nl.psum)
    pll = nl.ndarray((128, NJ), dtype=f32, buffer=nl.psum)
    for j in range(NJ):
        nisa.nc_matmul(dst=plh[:, j:j + 1], stationary=vh[:, j * 128:(j + 1) * 128], moving=ones_b[:, 0:1],
                       accumulate=False)
    for j in range(NJ):
        nisa.nc_matmul(dst=pll[:, j:j + 1], stationary=vl[:, j * 128:(j + 1) * 128], moving=ones_b[:, 0:1],
                       accumulate=False)
    lam = nl.ndarray((128, NJ), dtype=f32, buffer=nl.sbuf)
    lhs = nl.ndarray((128, NJ), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lhs, src=plh)
    nisa.scalar_tensor_tensor(dst=lam, data=lhs, op0=nl.multiply, operand0=64.0, op1=nl.add, operand1=pll)
    nisa.dma_copy(dst=o_lam, src=lam)
    pr = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
    ptk = nl.ndarray((128, 2), dtype=f32, buffer=nl.psum)
    lamk = nl.ndarray((128, NJ), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lamk, data=lam, op0=nl.subtract, operand0=1.0 * bk * BL)
    ohs = []
    for j in range(NJ):
        ohj = nl.ndarray((128, BL), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ohj, data=ilf, op0=nl.equal, operand0=lamk[:, j:j + 1])
        nisa.nc_matmul(dst=pr[0:BL, j * 16:(j + 1) * 16], stationary=ohj, moving=Wp[:, j, :], accumulate=False)
        ohs.append(ohj)
    for j in range(NJ):  # the tokens' accumulation on its own, uninterleaved
        nisa.nc_matmul(dst=ptk[0:BL, :], stationary=ohs[j], moving=tkp[:, j, :], accumulate=j > 0)
    rr = nl.ndarray((128, T), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=rr[0:BL, :], src=pr[0:BL, 0:T])
    nisa.dma_copy(dst=o_r, src=rr[0:BL, :])
    tt2 = nl.ndarray((128, 2), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tt2[0:BL, :], src=ptk[0:BL, :])
    nisa.dma_copy(dst=o_t, src=tt2[0:BL, :])
    return o_lam, o_r, o_t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=192)
    ap.add_argument("--experts", type=int, default=288)
    ap.add_argument("--simulate", action="store_true")
    a = ap.parse_args()
    from kiln.kernels import moe_dedupe as mdd

    T, E, K = a.rows, a.experts, 8
    g = torch.Generator().manual_seed(1)
    topi = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)]).to(torch.int32)
    topv = (torch.rand(T, K, generator=g) + 0.1).bfloat16()
    L = mdd.default_lanes(T)
    S, BL = mdd.n_slots(T, K, E, L)
    # host reference of the lanes (plan()'s arithmetic)
    slot_e, G, R = mdd.plan(topv, topi, E, L)  # G [T, SL], R [BL, blocks, T]
    N = T * K
    e = topi.reshape(N).long()
    lane = torch.full((N,), -1, dtype=torch.long)
    hit = G.t().float()  # [SL, T]
    if a.simulate:
        fn = nki.simulate(v10_plan_probe)
        lam, rr, tk = fn(topi=topi.numpy(), topv=topv.float().numpy().astype("float32"), E=E, L=L, S=S, BL=BL, bk=0)
        lam, rr, tk = torch.as_tensor(lam), torch.as_tensor(rr), torch.as_tensor(tk)
    else:
        from kiln import platform

        platform.configure_runtime_env()
        import libtorch_neuronx_lite  # noqa: F401
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        dev = torch.device("neuron:0")
        f = lambda ti, tv: wrap_nki(v10_plan_probe)[1](topi=ti, topv=tv, E=E, L=L, S=S, BL=BL, bk=0)  # noqa: E731
        from kiln.engine.model_runner import canonical_neuron_backend, neuronx_cc_args

        fc = torch.compile(f, backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                           options={"compiler_args": neuronx_cc_args(torch.bfloat16, True)})
        lam, rr, tk = [t.cpu() for t in fc(topi.to(dev), topv.to(dev))]
    # lanes from the host plan: lane of pair n is where G puts it; recompute directly from plan()'s formula
    oh = (e.unsqueeze(1) == torch.arange(E).unsqueeze(0)).long()
    cnt = oh.sum(0)
    ar = torch.arange(N)
    r = ((e.unsqueeze(1) == e.unsqueeze(0)) & (ar.unsqueeze(0) < ar.unsqueeze(1))).long().sum(1)
    nsl = (cnt + L - 1) // L
    base = torch.cumsum(nsl, 0) - nsl
    lane_ref = base[e] * L + r  # [N]
    lam_ref = lane_ref.view(N // 128, 128).t().float()  # [p, j]
    print(f"T={T} E={E} L={L} S={S} BL={BL}: lanes max |d| {(lam.float() - lam_ref).abs().max().item():.1f} "
          f"(lane max {lane_ref.max().item()})", flush=True)
    bad = (lam.float() != lam_ref).nonzero()
    print("first bad lanes [p, j]:", bad[:5].tolist(), "got", [lam[p, j].item() for p, j in bad[:5].tolist()],
          "want", [lam_ref[p, j].item() for p, j in bad[:5].tolist()], flush=True)
    R0 = R[:, 0, :].float()  # block 0: [BL, T]
    print(f"R block 0 max |d| {(rr.float() - R0).abs().max().item():.4f}", flush=True)
    tok_ref = torch.zeros(BL)
    for n in range(N):
        if lane_ref[n] < BL:
            tok_ref[lane_ref[n]] = n // K
    tkv = 16 * tk[:, 1].float() + tk[:, 0].float()
    print(f"tokens of block 0 max |d| {(tkv - tok_ref).abs().max().item():.1f}; got {tkv[:8].tolist()} want "
          f"{tok_ref[:8].tolist()}", flush=True)


if __name__ == "__main__":
    main()
