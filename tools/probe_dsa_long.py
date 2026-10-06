"""Exactness and time of the long-context DSA kernels on one NeuronCore (kernels/dsa_long_select.py,
kernels/dsa_slots.py).

    python tools/probe_dsa_long.py select [--rows 8 128] [--pools 2112 8448 32768 131072 262144] [--kinds ...]
    python tools/probe_dsa_long.py slots [--rows 8 128 1024 4096] [--kv fp8 bf16]

select: the pooled indexer's scores and their exact top-keep (keep 512, 32 heads of 128, scale 128^-0.5) for
`rows` queries sharing one context of `pools` pool keys, each query with its own candidate count npool (cycling
over 0, keep - 1, keep, keep + 1, P / 2, P and random), against emulate() (kernels/dsa_long_select.emulate_scores, then
models/dsa_long.select_reference) on the host: the selected sets must be EQUAL. Input kinds (the score kinds of
tools/probe_dsa_select.py, induced through the inputs):
  pooled   random queries, keys, weights of both signs (exact zeros where every head's relu is 0)
  randn    positive weights
  ties     keys drawn from 4 distinct vectors: thousands of pools tied per query
  zeros    zero weights: every score 0, the lowest indices win
  equal    one key for every pool
  wide     keys scaled by 2^k, k in [-40, 40]: scores over ~24 orders of magnitude
  ulps     keys of one base vector with single bf16 ulp perturbations: scores a few ulps apart
  short    pooled, npool <= 300 for every query
Timings are p50 of synchronous calls (tools/profile_layer.timed) and ns per (query, pool).

slots: kernels/dsa_slots.attend against kernels/dsa_decode.emulate (q_lat [N, 8, 512], a latent cache of 4096 pages
of 32, 640 slots per row: 512 random selected pools, a tail pool, padding), max |o - emulate| / max |o|; time per row.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NEG_INF = -1e30
KINDS = ["pooled", "randn", "ties", "zeros", "equal", "wide", "ulps", "short"]


def make_inputs(kind: str, N: int, P: int, keep: int, g: torch.Generator, Hi: int = 32, D: int = 128):
    """(qI [N, Hi, D] bf16, w [N, Hi] fp32, pk [P, D] bf16, npool [N] int64)."""
    q = torch.randn(N, Hi, D, generator=g)
    w = torch.randn(N, Hi, generator=g) * Hi ** -0.5
    pk = torch.randn(P, D, generator=g)
    if kind == "randn":
        w = w.abs()
    elif kind == "ties":
        base = torch.randn(4, D, generator=g)
        pk = base[torch.randint(0, 4, (P,), generator=g)]
    elif kind == "zeros":
        w = torch.zeros(N, Hi)
    elif kind == "equal":
        pk = torch.randn(1, D, generator=g).expand(P, D).clone()
    elif kind == "wide":
        e = torch.randint(-40, 41, (P, 1), generator=g).float()
        pk = pk * torch.pow(2.0, e)
    elif kind == "ulps":
        base = torch.randn(1, D, generator=g).to(torch.bfloat16)
        pk = base.expand(P, D).clone()
        flip = torch.randint(0, D, (P,), generator=g)
        bits = pk.view(torch.int16)
        bits[torch.arange(P), flip] += torch.randint(-1, 2, (P,), generator=g).to(torch.int16)
        pk = bits.view(torch.bfloat16).float()
    cyc = [0, keep - 1, keep, keep + 1, P // 2, P]
    npool = torch.tensor([cyc[i % len(cyc)] if i < 2 * len(cyc) else int(torch.randint(0, P + 1, (1,), generator=g))
                          for i in range(N)])
    if kind == "short":
        npool = npool.clamp(max=300)
    return q.to(torch.bfloat16), w.float(), pk.to(torch.bfloat16), npool.clamp(0, P)


def select_cases(args, pl) -> None:
    from kiln.kernels import dsa_long_select as dl

    keep, scale = args.keep, 128 ** -0.5
    for P in args.pools:
        for N in args.rows:
            for kind in args.kinds:
                g = torch.Generator().manual_seed(N * 7 + P + len(kind))
                q, w, pk, npool = make_inputs(kind, N, P, keep, g)
                want_p, want_c, want_v = dl.emulate(q, w, pk, npool, keep, scale, args.vorder)
                dev = tuple(x.to(pl.DEV) for x in (q, w, pk, npool))

                def f(q_, w_, pk_, np_):
                    p_, _, v_ = dl.select(q_, w_, pk_, np_, keep, scale, vorder=args.vorder)
                    return p_, v_

                name = f"select N={N} P={P} {kind}"
                t = pl.timed(name, f, dev, args.iters) if args.time or kind == args.kinds[0] else 0.0
                if t != t:
                    continue
                got, gv = (x.cpu() for x in torch.compile(f, **pl.OPTS)(*dev))
                bad = (got != want_p).any(-1)
                msg = f"    exact: {not bad.any().item()} ({int(bad.sum())} of {N} rows differ)"
                vb = (gv != want_v).any(-1) & ~bad
                msg += f"; scores equal where the sets are: {not vb.any().item()}"
                if bad.any() and N <= 128 and args.diag:  # the device's own scores: is its selection exact on them?
                    from kiln.models import dsa_long

                    kin = dl.kernel_inputs(q, w, pk, npool)
                    sub = dl.pick_sub(P, keep)

                    def fd(qT, w_, np_, pk_, ib):
                        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

                        return wrap_nki(dl.kiln_dsa_long_select_kernel)[1](
                            qT=qT, w=w_, npool=np_, pk=pk_, identb=ib, keep=keep, scale=scale, sub=sub or dl.CH,
                            lsub=(sub or dl.CH).bit_length() - 1, one=int(sub == 0), rev=dl.REV, loop=1, dbg=1)

                    d_out = torch.compile(fd, **pl.OPTS)(*(kin[k].to(pl.DEV) for k in ("qT", "w", "npool", "pk", "identb")))
                    dsc = d_out[2].cpu()[:N, :P]
                    cand = torch.where(torch.arange(P).view(1, P) < npool.view(N, 1), 0.0, NEG_INF)
                    esc = dl.emulate_scores(q, w, pk, cand, scale)
                    own = dsa_long.select_reference(dsc, keep)[0]
                    sdiff = (dsc != esc) & (esc > -5e29)
                    msg += (f"; device scores differ from emulate_scores in {int(sdiff.sum())} of {int((esc > -5e29).sum())} "
                            f"(max rel {((dsc - esc).abs() / esc.abs().clamp(min=1e-30))[sdiff].max().item() if sdiff.any() else 0:.2e}); "
                            f"device selection == exact selection of the device's scores: {torch.equal(own, got)}")
                if t:
                    msg += f"; {t * 1e9 / (N * P):.3f} ns per (query, pool)"
                if bad.any():
                    r = int(bad.nonzero()[0])
                    gs, ws_ = set(got[r, :int(want_c[r])].tolist()), set(want_p[r, :int(want_c[r])].tolist())
                    msg += f"; row {r} npool {int(npool[r])}: {len(gs - ws_)} extra, {len(ws_ - gs)} missing"
                pl.say(msg, flush=True)


def slots_cases(args, pl) -> None:
    from kiln.kernels import dsa_decode, dsa_slots

    R, KP = 512, 4
    pages, ps = 4096, 32
    for kv, H in [(kv, H) for kv in args.kv for H in args.heads]:
        g = torch.Generator().manual_seed(3)
        kc = (torch.randn(pages * ps, 1, R, generator=g) * 2).clamp(-200, 200)
        kc = kc.to(torch.float8_e4m3fn if kv == "fp8" else torch.bfloat16)
        for N, NS in [(N, NS) for N in args.rows for NS in (args.slots or [dsa_decode.NCH * 128])]:
            q = (torch.randn(N, H, R, generator=g) * 0.05).to(torch.bfloat16)
            rows = torch.randint(0, pages * ps // KP, (N, NS), generator=g)
            bias = torch.zeros(N, NS, KP)
            bias[:, NS - 127:] = NEG_INF  # the long path's shape: keep slots, the tail's partial pool, padding
            bias[:, NS - 128, 2:] = NEG_INF
            if N >= 4:
                bias[N // 2] = NEG_INF  # one row with every slot masked: finite, uniform p, lse ~ NEG_INF
            parts = [dsa_slots.emulate(q[a:a + 128].float(), kc.reshape(-1, R), rows[a:a + 128], bias[a:a + 128],
                                       R ** -0.5, lse=True) for a in range(0, N, 128)]  # (host memory: 128 rows at once)
            want, want_l = torch.cat([p_[0] for p_ in parts]), torch.cat([p_[1] for p_ in parts])
            ref = dsa_decode.emulate(q[:8].float(), kc.reshape(-1, R), rows[:8], bias[:8], R ** -0.5)
            assert torch.equal(ref, want[:8])
            dev = tuple(x.to(pl.DEV) for x in (q, kc, rows, bias))

            def f(q_, kc_, r_, b_):
                return dsa_slots.attend(q_, kc_, r_, b_, R ** -0.5, lse=True)

            def ft(q_, kc_, r_, b_):  # timed: the outputs reduced on the device (no [N, H, R] read-back in the time)
                o_, l_ = dsa_slots.attend(q_, kc_, r_, b_, R ** -0.5, lse=True)
                return o_.sum() + l_.clamp(min=-1.0).sum()

            t = pl.timed(f"slots N={N} NS={NS} H={H} kv={kv}", ft, dev, args.iters)
            if t != t:
                continue
            got, got_l = (x.cpu() for x in torch.compile(f, **pl.OPTS)(*dev))
            err = (got - want).abs().max().item() / want.abs().max().item()
            fin = torch.isfinite(got).all().item() and torch.isfinite(got_l).all().item()
            real = want_l > -1e29
            lerr = (got_l - want_l)[real].abs().max().item() if real.any() else 0.0
            masked = (got_l[~real] < -1e29).all().item() if (~real).any() else True
            pl.say(f"    max |o - emulate| / max |o| = {err:.2e}; lse max abs err {lerr:.2e} (masked rows lse ~NEG_INF "
                   f"{masked}); finite {fin}; {t * 1e6 / N:.2f} us per row", flush=True)
            if N <= 128 and H == 8:  # the baseline: kernels/dsa_decode.py on the same rows (unrolled)
                def fd(q_, kc_, r_, b_):
                    return dsa_decode.attend(q_, kc_, r_, b_, R ** -0.5)

                td = pl.timed(f"dsa_decode N={N} H={H} kv={kv}", lambda *a: fd(*a).sum(), dev, args.iters)
                if td == td:
                    gd = torch.compile(fd, **pl.OPTS)(*dev).cpu()
                    pl.say(f"    dsa_decode max |o - emulate| / max |o| = "
                           f"{(gd - want).abs().max().item() / want.abs().max().item():.2e}; {td * 1e6 / N:.2f} us per row",
                           flush=True)


def slots_n_cases(args, pl) -> None:
    """kernels/dsa_slots_n.py: the first n rows equal kernels/dsa_slots.py's on the same inputs (n = 0, 1, 37, N), and
    its time at n = N against dsa_slots' (the run-time loop count costs nothing per row)."""
    from kiln.kernels import dsa_slots, dsa_slots_n

    R, KP = 512, 4
    pages, ps = 4096, 32
    for kv, H in [(kv, H) for kv in args.kv for H in args.heads]:
        g = torch.Generator().manual_seed(5)
        kc = (torch.randn(pages * ps, 1, R, generator=g) * 2).clamp(-200, 200)
        kc = kc.to(torch.float8_e4m3fn if kv == "fp8" else torch.bfloat16)
        for N, NS in [(N, NS) for N in args.rows for NS in (args.slots or [128, 640])]:
            q = (torch.randn(N, H, R, generator=g) * 0.05).to(torch.bfloat16)
            rows = torch.randint(0, pages * ps // KP, (N, NS), generator=g)
            bias = torch.zeros(N, NS, KP)
            bias[:, NS - 64:] = NEG_INF
            dev = tuple(x.to(pl.DEV) for x in (q, kc, rows, bias))

            def ref(q_, kc_, r_, b_):
                return dsa_slots.attend(q_, kc_, r_, b_, R ** -0.5, lse=True)

            want_o, want_l = (x.cpu() for x in torch.compile(ref, **pl.OPTS)(*dev))
            for n in sorted({0, 1, 37, N}):
                nt = torch.tensor([n], dtype=torch.int64).to(pl.DEV)

                def f(q_, kc_, r_, b_, n_):
                    return dsa_slots_n.attend(q_, kc_, r_, b_, R ** -0.5, n_, lse=True)

                got_o, got_l = (x.cpu() for x in torch.compile(f, **pl.OPTS)(*dev, nt))
                same = torch.equal(got_o[:n], want_o[:n]) and torch.equal(got_l[:n], want_l[:n])
                pl.say(f"  slots_n N={N} NS={NS} H={H} kv={kv} n={n}: first n rows equal dsa_slots' {same}", flush=True)
            nt = torch.tensor([N], dtype=torch.int64).to(pl.DEV)
            ta = pl.timed(f"slots   N={N} NS={NS} H={H} kv={kv}",
                          lambda *a: sum(x.sum() for x in ref(*a)), dev, args.iters)
            tb = pl.timed(f"slots_n N={N} NS={NS} H={H} kv={kv} n={N}",
                          lambda *a: sum(x.sum() for x in dsa_slots_n.attend(*a[:4], R ** -0.5, a[4], lse=True)),
                          (*dev, nt), args.iters)
            nh = torch.tensor([N // 2], dtype=torch.int64).to(pl.DEV)
            th = pl.timed(f"slots_n N={N} NS={NS} H={H} kv={kv} n={N // 2}",
                          lambda *a: sum(x.sum() for x in dsa_slots_n.attend(*a[:4], R ** -0.5, a[4], lse=True)),
                          (*dev, nh), args.iters)
            pl.say(f"    us per row: dsa_slots {ta * 1e6 / N:.2f}, dsa_slots_n at n = N {tb * 1e6 / N:.2f}, at n = N / 2 "
                   f"{th * 1e6 / N:.2f} (per buffer row)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["select", "slots", "slots_n"])
    ap.add_argument("--rows", type=int, nargs="+", default=[8, 128])
    ap.add_argument("--pools", type=int, nargs="+", default=[2112, 8448, 32768, 131072, 262144])
    ap.add_argument("--keep", type=int, default=512)
    ap.add_argument("--kinds", nargs="+", default=KINDS)
    ap.add_argument("--kv", nargs="+", default=["fp8", "bf16"])
    ap.add_argument("--heads", type=int, nargs="+", default=[8, 64])
    ap.add_argument("--slots", type=int, nargs="+", default=None,
                    help="slots: slots per row (multiples of 128; default the long path's dsa_decode.NCH x 128 = 640)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--time", action="store_true", help="time every kind (default: the first only)")
    ap.add_argument("--diag", action="store_true", help="on a mismatch, rerun with the device scores dumped")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--vorder", action="store_true", help="select: the kernel's value-order output (CP slot classes)")
    args = ap.parse_args()
    import profile_layer as pl

    pl.setup_device(args.cpu)
    {"select": select_cases, "slots": slots_cases, "slots_n": slots_n_cases}[args.what](args, pl)


if __name__ == "__main__":
    main()
