"""Kiln's NKI kernels at one tensor-parallel rank's GLM-5.3-Flash shapes on ONE logical NeuronCore: device
time and the outputs, saved, so that two trees (or two LNC settings) can be compared bit for bit.

    python tools/probe_lnc_split.py --out /opt/kiln/work/lnc-<tag>.pt [--kernels prefill dedupe kda dsa dsa_score]
    python tools/probe_lnc_split.py --compare a.pt b.pt

Why: on trn2 at LNC=2 a logical core is two physical cores, and a kernel launched with grid 2 runs once on
each (kiln/platform.py nki_grid). A kernel that does not split its work by nl.program_id runs it twice and
writes the same output twice, so it uses half of the logical core. The trn2 split (feat/trn2-max) gives each
physical core half of the work: kernels/moe_prefill.py lane tiles and combine token tiles, moe_dedupe.py
blocks of lanes (fp32 partials exchanged by nisa.sendrecv), delta_rule.py v heads, dsa_topk.py row tiles. On
trn1 (grid 1) every kernel must produce the same bits as before the split; on trn2 every kernel but the
dedupe one must too (the dedupe split sums two fp32 partials, a different summation order; it is compared
with its host emulation instead).

Cases (fixed seeds; GLM-5.3-Flash at tp=32 and DP attention 4, the sweep's per-rank shapes):
- prefill: kernels/moe_prefill.py on the experts as the loader leaves them (tools/probe_moe_prefill.py
  --format loaded: per-row path, per-column down factors), C = 1024 / 4096 rows, uniform routing, top-8 of 288.
- dedupe: kernels/moe_dedupe.py (the decode MoE) on the same experts, T = 16 / 32 / 64 / 128 rows.
- kda: kernels/delta_rule.py, 8 v heads (64 / attention TP 8), C = 1024 / 4096 (tests/test_delta_rule inputs).
- dsa: kernels/dsa_topk.select, pooled form (keep 512 of 2112, vis_only), rows 8 / 32 / 1024.
- dsa_score: kernels/dsa_topk.score_select, the prefill chunk's scores and selection, C = 1024 queries,
  32 indexer heads of 128, 2112 pools.
Per case: p50 / min of synchronous calls (the graph reads back a reduction, as the other probes), the output
on the host, and its distance from the host emulation where the module has one.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

KERNELS = ("prefill", "dedupe", "kda", "dsa", "dsa_score")


def timed(pl, fn, args, iters: int) -> tuple[float, float]:
    c = torch.compile(fn, **pl.OPTS)
    pl._sync(c(*args))
    pl._sync(c(*args))
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        pl._sync(c(*args))
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[iters // 2] * 1e3, ts[0] * 1e3


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    fin = torch.isfinite(b)
    return ((a - b).abs()[fin].max() / b.abs()[fin].max().clamp(min=1e-30)).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--kernels", nargs="+", default=list(KERNELS), choices=KERNELS)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--prefill-chunks", type=int, nargs="+", default=[1024, 4096], help="MoE prefill rows per call")
    ap.add_argument("--skew", action="store_true", help="MoE routing: every token's first expert from 8 hot ones")
    ap.add_argument("--hot", type=int, default=0, help="MoE routing: every token picks from these first N experts")
    ap.add_argument("--replay", default=None, help="tools/dump_moe_inputs.py output: its x and routing as one more case")
    ap.add_argument("--experts-file", default=None, help="real experts (tools/check_moe_prefill_layout.py --save)")
    ap.add_argument("--ballast-gb", type=float, default=0.0,
                    help="device memory held before any case (a served model's weights and KV sit below its kernels)")
    ap.add_argument("--chain", type=int, default=0, help="also N MoE prefill calls chained in one graph")
    ap.add_argument("--chain-chunks", type=int, nargs="+", default=[256])
    ap.add_argument("--race", action="store_true", help="kernel outputs read across rows by XLA ops in one graph")
    ap.add_argument("--race-chunks", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--kda-heads", type=int, default=8, help="v heads per rank (64 / attention TP)")
    ap.add_argument("--kda-chunks", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--dsa-rows", type=int, nargs="+", default=[8, 32, 1024])
    ap.add_argument("--score-chunks", type=int, nargs="+", default=[1024])
    args = ap.parse_args()
    if args.compare:
        a, b = (torch.load(p) for p in args.compare)
        for name in a["outputs"]:
            if name not in b["outputs"]:
                continue
            ta, tb = a["outputs"][name], b["outputs"][name]
            same = all(torch.equal(x, y) for x, y in zip(ta, tb))
            d = max(rel(x, y) for x, y in zip(ta, tb))
            print(f"{name:<24} {a['times'][name][0]:8.3f} -> {b['times'][name][0]:8.3f} ms p50 "
                  f"({a['times'][name][0] / b['times'][name][0]:.2f}x)   outputs "
                  f"{'BIT-IDENTICAL' if same else f'differ, max rel {d:.2e}'}", flush=True)
        return

    import profile_layer as pl

    pl.setup_device()
    from kiln import platform
    from kiln.kernels import delta_rule as dr
    from kiln.kernels import dsa_topk as dk
    from kiln.kernels import moe_dedupe as mdd
    from kiln.kernels import moe_prefill as mp

    print(f"platform {platform.target()} grid {platform.nki_grid()} REV prefill {mp.REV:#x} delta {dr.REV:#x} "
          f"dsa {dk.REV:#x}", flush=True)
    ballast = []
    left = int(args.ballast_gb * 2**30)
    while left > 0:  # in 1 GiB tensors
        n = min(left, 2**30)
        ballast.append(torch.empty(n, dtype=torch.uint8).to(pl.DEV))
        left -= n
    if ballast:
        print(f"holding {args.ballast_gb} GiB of device memory", flush=True)
    outputs, times = {}, {}

    def record(name, fn, dev_args, emu=None):
        p50, mn = timed(pl, lambda *a: [o.float().sum(0) for o in _tup(fn(*a))], dev_args, args.iters)
        got = [o.cpu() for o in _tup(torch.compile(fn, **pl.OPTS)(*dev_args))]
        outputs[name], times[name] = got, (p50, mn)
        e = "" if emu is None else f"   vs host emulation max rel {max(rel(g, w) for g, w in zip(got, _tup(emu))):.2e}"
        print(f"{name:<24} p50 {p50:8.3f} ms  min {mn:8.3f}{e}", flush=True)

    H, E, K = 4096, 288, 8
    if "prefill" in args.kernels or "dedupe" in args.kernels:
        import probe_moe_prefill as pmp

        if args.experts_file:  # a real layer's experts for one rank (tools/check_moe_prefill_layout.py --save)
            dd = torch.load(args.experts_file)
            ws = (dd["w_gu"], dd["w_gu_scale"], dd["w_down"], dd["w_down_scale"])
            E = ws[0].shape[0]
            print(f"experts from {args.experts_file}: {dd.get('model')} layer {dd.get('layer')} rank {dd.get('rank')}",
                  flush=True)
        else:
            ws = pmp.experts("loaded", E, H)
        blob = mdd.pack(*ws)
        dq = mp.check_blob(blob, H)
        down = mp.down_factors(blob, H)
        dblob = blob.to(pl.DEV)
        ddown = None if down is None else tuple(t.to(pl.DEV) for t in down)
        act, lim = mdd.ACTS["silu_clamp"], 10.0
        g = torch.Generator().manual_seed(7)
        cases = [("prefill", C) for C in (args.prefill_chunks if "prefill" in args.kernels else ())]
        cases += [("chain", C) for C in (args.chain_chunks if args.chain else ())]
        cases += [("race", C) for C in (args.race_chunks if args.race else ())]
        cases += [("dedupe", C) for C in ((16, 32, 64, 128) if "dedupe" in args.kernels else ())]
        if args.replay:
            rp = torch.load(args.replay)
            cases.append(("replay", rp["x"].shape[0]))
        for kind, C in cases:
            if kind == "replay":  # the real routing (and input) of a prompt, through the MoE prefill kernel
                x, topv, topi = rp["x"].bfloat16(), rp["topv"].bfloat16(), rp["topi"]
                d = tuple(t.to(pl.DEV) for t in (x, topv, topi))
                record(f"prefill replay C={C}", lambda x, v, i: mp.moe_prefill(x, v, i, dblob, act, lim, dq=dq,
                                                                                down=ddown), d,
                       mp.emulate(x, topv, topi, blob, act, lim, dq=dq))
                continue
            x = torch.randn(C, H, generator=g).bfloat16()
            topi = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(C)])
            if args.hot:
                topi = torch.stack([torch.randperm(args.hot, generator=g)[:K] for _ in range(C)])
            if args.skew:
                hot = torch.randint(0, 8, (C,), generator=g)
                for t in range(C):
                    rest = [e for e in torch.randperm(E, generator=g).tolist() if e != int(hot[t])][: K - 1]
                    topi[t] = torch.tensor([int(hot[t])] + rest)
            topv = (torch.rand(C, K, generator=g) + 0.1).bfloat16()
            d = tuple(t.to(pl.DEV) for t in (x, topv, topi))
            if kind == "race":  # XLA ops reading other rows between the kernel calls, in one graph
                def race(x, v, i):
                    for _ in range(4):
                        y = mp.moe_prefill(x, v, i, dblob, act, lim, dq=dq, down=ddown)
                        x = (y.flip(0) * 0.5 + x * 0.5).to(x.dtype)
                    return x

                record(f"prefill race x4 C={C}", race, d)
            elif kind == "chain":  # args.chain kernel calls in ONE graph, each on the previous one's output
                def chain(x, v, i):
                    for _ in range(args.chain):
                        x = mp.moe_prefill(x, v, i, dblob, act, lim, dq=dq, down=ddown)
                    return x

                record(f"prefill x{args.chain} in one graph C={C}", chain, d)
            elif kind == "prefill":
                record(f"prefill C={C}", lambda x, v, i: mp.moe_prefill(x, v, i, dblob, act, lim, dq=dq, down=ddown), d,
                       mp.emulate(x, topv, topi, blob, act, lim, dq=dq) if C <= 1024 else None)
            else:
                record(f"dedupe T={C}", lambda x, v, i: mdd.moe_dedupe(x, v, i, dblob, act=act, limit=lim), d,
                       mdd.emulate(x, topv, topi, blob, act, lim) if hasattr(mdd, "emulate") else None)
    if "kda" in args.kernels:
        from tests.test_delta_rule import inputs

        Hh = args.kda_heads
        if args.race:
            def kda_race(q, k, v, g, b, S):
                for _ in range(4):
                    o, S = dr.chunk(q, k, v, g, b, S)
                    v = o.flip(0).flip(1) * 0.5 + v * 0.5
                return v, S

            host = inputs(512, Hh, Hh, True, seed=5)
            record(f"kda race x4 C=512 {Hh} heads", kda_race, tuple(t.to(pl.DEV) for t in host))

            def kda_pool(q, k, v, g, b, pool, slot):  # the state pool round trip a layer graph does
                for _ in range(4):
                    o, S = dr.chunk(q, k, v, g, b, pool.index_select(0, slot)[0])
                    pool = pool.index_put((slot,), S.unsqueeze(0))
                    v = o * 0.5 + v * 0.5
                return v, pool

            pool = torch.randn(4, Hh, 128, 128) * 0.1
            dpool = tuple(t.to(pl.DEV) for t in (*host[:5], pool, torch.tensor([2])))
            record(f"kda pool race x4 C=512 {Hh} heads", kda_pool, dpool)

            def kda_s_add(q, k, v, g, b, pool, slot):  # the final state read by an XLA op
                o, S = dr.chunk(q, k, v, g, b, pool[2])
                return o, S + 0.0

            def kda_s_put(q, k, v, g, b, pool, slot):  # the final state scattered into the pool once
                o, S = dr.chunk(q, k, v, g, b, pool[2])
                return o, pool.index_put((slot,), S.unsqueeze(0))

            def kda_s0_take(q, k, v, g, b, pool, slot):  # the initial state gathered from the pool once
                return dr.chunk(q, k, v, g, b, pool.index_select(0, slot)[0])

            host_S = dr.emulate(*host[:5], pool[2])
            record(f"kda S + 0 C=512 {Hh} heads", kda_s_add, dpool, host_S)
            record(f"kda S into pool C=512 {Hh} heads", kda_s_put, dpool)
            record(f"kda S0 from pool C=512 {Hh} heads", kda_s0_take, dpool, host_S)
        for C in args.kda_chunks:
            host = inputs(C, Hh, Hh, True, seed=C)
            record(f"kda C={C} {Hh} heads", lambda *a: dr.chunk(*a), tuple(t.to(pl.DEV) for t in host),
                   dr.emulate(*host) if C <= 1024 else None)
    if "dsa" in args.kernels:
        g = torch.Generator().manual_seed(11)
        for R in args.dsa_rows:
            sc = torch.randn(R, 2112, generator=g)
            sc[:, 2000:] = dk.NEG_INF if hasattr(dk, "NEG_INF") else -1e30  # invisible tail candidates
            record(f"dsa select R={R}", lambda s: dk.select(s, 512, True), (sc.to(pl.DEV),),
                   dk.emulate(sc, 512, True))
    if "dsa_score" in args.kernels:
        g = torch.Generator().manual_seed(13)
        for C in args.score_chunks:
            Hi, D, P = 32, 128, 2112
            q = torch.randn(C, Hi, D, generator=g).bfloat16().float()
            w = torch.randn(C, Hi, generator=g)
            pk = torch.randn(P, D, generator=g).bfloat16().float()
            cand = torch.zeros(C, P)
            cand[:, 2000:] = -1e30
            record(f"dsa score+select C={C}", lambda *a: dk.score_select(*a, keep=512, scale=D ** -0.5),
                   tuple(t.to(pl.DEV) for t in (q, w, pk, cand)),
                   dk.emulate(dk.emulate_scores(q, w, pk, cand, D ** -0.5), 512, True))
    if args.out:
        torch.save({"outputs": outputs, "times": times, "target": platform.target(), "grid": platform.nki_grid()},
                   args.out)
        print(f"saved {args.out}", flush=True)


def _tup(x):
    return tuple(x) if isinstance(x, (tuple, list)) else (x,)


if __name__ == "__main__":
    main()
