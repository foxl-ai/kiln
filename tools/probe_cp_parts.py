"""The context-parallel DSA layer's XLA pieces after the CP gathers (models/mla.py attention_cp, the replay's ~20 ms per
layer of "segment B" beside the dsa_slots kernel), each timed alone on one NeuronCore at the served shape: a 1024-row
chunk, A = 8 ranks, keep = 512, 64 heads x 512 latent, 640 slots, the 4096-page bucket of page 256 (32,768 local pools).

    python tools/probe_cp_parts.py [--rows 1024] [--A 8] [--iters 10]

Pieces: cp_merge (the dsa_topk threshold and the 21-step tie search over [rows, A keep]); the merge's selection kernel
alone; the slot rows and bias (pool_row gathers, the broadcast build); the slot-class prep (compact of the local list,
two buffers' gathers); dsa_slots with its wrapper's permutes; and the einsum of q_lat. Inputs are random of the right
kind (scores with ties, a valid prefix), so the times are the ops' and not the routing's.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--A", type=int, default=8)
    ap.add_argument("--keep", type=int, default=512)
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    import profile_layer as pl

    pl.setup_device(False)
    from kiln.kernels import dsa_decode, dsa_slots
    from kiln.models import dsa_long, mla

    T, A, K, H, R, kp, nh, dn = a.rows, a.A, a.keep, 64, 512, 4, 64 // a.A, 256
    ps, pages = 256, 4096
    ppl = ps // (kp * A)
    Pl = pages * ppl
    NS = dsa_decode.NCH * 128
    g = torch.Generator().manual_seed(1)
    vals = torch.randint(0, 50, (T, A, K), generator=g).float()  # integer scores: ties at the threshold
    cpool = (torch.arange(K).view(1, 1, K) * A + torch.arange(A).view(1, A, 1)).float().expand(T, A, K).contiguous()
    lp = torch.sort(torch.randint(0, Pl, (T, K), generator=g), dim=1).values
    mine = torch.rand(T, K, generator=g) < (1.0 / A)
    table = torch.randperm(pages, generator=g) + 1
    positions = torch.arange(T) + 600000
    npool = dsa_long.npools(positions, kp)
    tail_own = torch.rand(T, generator=g) < (1.0 / A)
    kc = torch.randn(pages * ps // A + 64, 1, R, generator=g).clamp(-8, 8).to(torch.float8_e4m3fn)
    q_all = (torch.randn(T, H, R, generator=g) * 0.05).to(torch.bfloat16)
    q_nope = torch.randn(T, nh, dn, generator=g).to(torch.bfloat16)
    w_uk = torch.randn(nh * dn, R, generator=g).to(torch.bfloat16)
    D = pl.DEV
    vals_d, cpool_d, lp_d, mine_d, table_d, pos_d, npool_d, own_d, kc_d, q_d, qn_d, wuk_d = (
        x.to(D) for x in (vals, cpool, lp, mine, table, positions, npool, tail_own, kc, q_all, q_nope, w_uk))

    def merge(v, c):
        return dsa_long.cp_merge(v, c, K).to(torch.float32).sum()

    def merge_kernel(v):
        from kiln.kernels import dsa_topk

        return dsa_topk.select(v.reshape(T, A * K), K, vis_only=True).sum()

    def rows_bias(lp_, mine_, tb, npool_, own_, pos_):
        def pool_row(mm):
            pg = torch.floor(mm.to(torch.float32) * (1.0 / ppl)).to(torch.int64)
            tbe = tb.view(1, -1).expand(T, -1)
            return tbe.gather(1, pg.clamp(max=tbe.shape[1] - 1)) * ppl + (mm - pg * ppl)
        tail_m = torch.floor(npool_.to(torch.float32) * (1.0 / A)).to(torch.int64)
        rows_sel = pool_row(lp_)
        rows_tail = pool_row(tail_m.clamp(max=Pl - 1).view(T, 1))
        sl = torch.arange(NS, device=lp_.device).view(1, NS)
        cl = sl.clamp(max=K - 1).expand(T, NS)
        srows = torch.where(sl < K, torch.gather(rows_sel, 1, cl),
                            torch.where(sl == K, rows_tail, torch.zeros_like(rows_tail)))
        t4 = torch.arange(kp, device=lp_.device).view(1, 1, kp)
        sel_ok = ((sl < K) & torch.gather(mine_, 1, cl)).unsqueeze(-1)
        tail_ok = ((sl == K) & own_.view(T, 1)).unsqueeze(-1) & (npool_.view(T, 1, 1) * kp + t4 <= pos_.view(T, 1, 1))
        sbias = torch.where(sel_ok | tail_ok, 0.0, mla.NEG_INF).to(torch.float32)
        return srows.sum().float() + sbias.clamp(min=-1.0).sum()

    def classes_prep(lp_, mine_, q_):
        sidx, scnt = dsa_long.compact(mine_.to(torch.float32), K)
        crow = torch.gather(lp_, 1, sidx)
        fits = scnt <= 127
        idx, n = dsa_long.compact(fits.to(torch.float32).view(1, T), T)
        idx = idx.view(T)
        return crow[idx].sum().float() + q_[idx].float().sum() + n.float().sum()

    def compact_local(mine_):
        sidx, scnt = dsa_long.compact(mine_.to(torch.float32), K)
        return sidx.sum().float() + scnt.sum().float()

    def compact_local128(mine_):
        sidx, scnt = dsa_long.compact(mine_.to(torch.float32), 128)
        return sidx.sum().float() + scnt.sum().float()

    def compact_rows(mine_):
        fits = mine_.sum(-1) <= 127
        idx, n = dsa_long.compact(fits.to(torch.float32).view(1, T), T)
        return idx.sum().float() + n.sum().float()

    def gather_q(q_, mine_):
        idx = (mine_.sum(-1) * 7) % T
        return q_[idx].float().sum()

    def q_lat(qn, w):
        return torch.einsum("thd,hdr->thr", qn, w.view(nh, dn, R)).float().sum()

    srows = torch.randint(0, kc.shape[0] // kp, (T, NS), generator=g).to(D)
    sbias = torch.zeros(T, NS, kp)
    sbias[:, 513:] = mla.NEG_INF
    sbias = sbias.to(D)

    def slots(q_, kc_, r_, b_):
        o, ls = dsa_slots.attend(q_, kc_, r_, b_, R ** -0.5, lse=True)
        return o.sum() + ls.clamp(min=-1.0).sum()

    pl.say(f"T {T} A {A} keep {K} local pools {Pl} slots {NS}", flush=True)
    for name, fn, args in (("cp_merge", merge, (vals_d, cpool_d)), ("cp_merge's dsa_topk alone", merge_kernel, (vals_d,)),
                           ("slot rows + bias", rows_bias, (lp_d, mine_d, table_d, npool_d, own_d, pos_d)),
                           ("slot-class prep", classes_prep, (lp_d, mine_d, q_d)),
                           ("  compact of the local list (keep outputs)", compact_local, (mine_d,)),
                           ("  compact of the local list (128 outputs)", compact_local128, (mine_d,)),
                           ("  compact of the rows", compact_rows, (mine_d,)),
                           ("  q_all[idx] gather", gather_q, (q_d, mine_d)),
                           ("q_lat einsum", q_lat, (qn_d, wuk_d)),
                           ("dsa_slots + wrapper", slots, (q_d, kc_d, srows, sbias))):
        t = pl.timed(name, fn, args, a.iters)
        pl.say(f"    {name}: {t * 1e3:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
