"""The pipelined multi-tile selection kernel (kernels/dsa_long_pipe.py) against the serving form, one NeuronCore.

    python tools/probe_lc_pipe.py [--rows 1024] [--pools 8192 32768] [--kinds pooled ties ulps ...] [--vorder]

For each case: the serving form (models/mla.py _select_tiles: kernels/dsa_long_select.py once per 128 queries) and
the pipelined kernel (one call over every tile) on the same inputs (tools/probe_dsa_long.py make_inputs, keep 512, 32
heads of 128): pools, counts and scores must be bit-identical (same instructions on the same data), and the selected
sets equal emulate()'s except where tools/probe_dsa_long.py already documents the PE's dot-product order flipping a
near tie (the "ulps" kind). Then each form's p50 per call (tools/profile_layer.timed) and ns per (query, pool).

--skip adds kernels/dsa_long_pipe_x.py (each tile's chunks bounded by its largest npool, KILN_DSA_LONG_SKIP), which
must equal the per-tile form bit for bit too, and its time. Two npool kinds exercise it (the scores as "pooled"):
"chunkNN" a prefill chunk at NN% of the bucket (query i at npool NN% P - N + i), "real:<file>" a saved
tools/make_real_indexer_inputs.py case (GLM-5.3-Flash's indexer on real text), "tilemix" one shape per tile (every
query at 0; the cycle 0, keep - 1, keep, keep + 1; <= 300; < keep sub; just above keep sub; random to P / 2; random to
P), laid out so that a tile follows a same-parity tile that scored more chunks (its scratch holds that tile's scores).

--pe adds the tensor-engine head sum (kernels/dsa_long_pipe_x.py pe, KILN_DSA_LONG_PE; not exact): each selected
entry's score against emulate_scores' fp32 score of that pool (max and mean relative deviation where |score| is above
1e-3 of the row's largest, and max |d| over the row's largest), the selected sets against the exact per-tile form's
(rows that differ, entries that differ per row: mean and max over keep), and its time.
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
    ap.add_argument("--rows", type=int, nargs="+", default=[1024])
    ap.add_argument("--pools", type=int, nargs="+", default=[8192, 32768])
    ap.add_argument("--kinds", nargs="+", default=["pooled", "ties", "zeros", "wide", "ulps", "short"])
    ap.add_argument("--keep", type=int, default=512)
    ap.add_argument("--vorder", action="store_true")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--skip", action="store_true", help="also kernels/dsa_long_pipe_x.py with skip = 1")
    ap.add_argument("--time-all", action="store_true", help="time every kind (default: the first only)")
    ap.add_argument("--pe", action="store_true", help="also the tensor-engine head sum (pe = 1, not exact)")
    a = ap.parse_args()
    import profile_layer as pl
    import probe_dsa_long as pdl

    pl.setup_device(False)
    if pl.DEV is None or pl.DEV.type == "cpu":
        raise SystemExit("probe_lc_pipe needs a NeuronCore (both forms would be the host emulation)")

    from kiln.kernels import dsa_long_pipe as dp
    from kiln.kernels import dsa_long_pipe_x as dpx
    from kiln.kernels import dsa_long_select as dl

    def inputs(kind, N, P, keep, g):
        if kind.startswith("real:"):  # tools/make_real_indexer_inputs.py: --rows / --pools must be the file's
            d = torch.load(kind[5:])
            if (d["q"].shape[0], d["pk"].shape[0]) != (N, P):
                raise SystemExit(f"{kind}: {d['q'].shape[0]} queries x {d['pk'].shape[0]} pools, not {N} x {P}")
            return d["q"], d["w"], d["pk"], d["npool"]
        base = "pooled" if kind.startswith("chunk") or kind == "tilemix" else kind
        q, w, pk, npool = pdl.make_inputs(base, N, P, keep, g)
        if kind.startswith("chunk"):
            top = int(round(int(kind[5:]) / 100 * P))
            npool = (top - N + torch.arange(N)).clamp(0, P)
        elif kind == "tilemix":
            sub = dl.pick_sub(P, keep)
            shapes = [lambda n: torch.randint(0, P + 1, (n,), generator=g),
                      lambda n: torch.randint(0, P // 2 + 1, (n,), generator=g),
                      lambda n: torch.zeros(n, dtype=torch.int64),
                      lambda n: torch.tensor([0, keep - 1, keep, keep + 1] * (n // 4)),
                      lambda n: torch.randint(0, 301, (n,), generator=g),
                      lambda n: torch.randint(0, keep * sub - 100, (n,), generator=g),
                      lambda n: torch.randint(keep * sub - 8, keep * sub + 9, (n,), generator=g),
                      lambda n: torch.full((n,), P, dtype=torch.int64)]
            npool = torch.cat([shapes[(i // 128) % len(shapes)](min(128, N - i)) for i in range(0, N, 128)])
            npool = npool.clamp(0, P)
        return q, w, pk, npool

    keep, scale = a.keep, 128 ** -0.5
    for P in a.pools:
        for N in a.rows:
            for ki, kind in enumerate(a.kinds):
                g = torch.Generator().manual_seed(N * 7 + P + len(kind))
                q, w, pk, npool = inputs(kind, N, P, keep, g)
                want_p, want_c, want_v = dl.emulate(q, w, pk, npool, keep, scale, a.vorder)
                dev = tuple(x.to(pl.DEV) for x in (q, w, pk, npool))

                def tiles(q_, w_, pk_, np_):
                    parts = [dl.select(q_[i:i + 128], w_[i:i + 128], pk_, np_[i:i + 128], keep, scale,
                                       vorder=a.vorder) for i in range(0, q_.shape[0], 128)]
                    return torch.cat([p[0] for p in parts]), torch.cat([p[2] for p in parts])

                def pipe(q_, w_, pk_, np_):
                    p_, _, v_ = dp.select(q_, w_, pk_, np_, keep, scale, vorder=a.vorder)
                    return p_, v_

                def pipex(q_, w_, pk_, np_):
                    p_, _, v_ = dpx.select(q_, w_, pk_, np_, keep, scale, vorder=a.vorder)
                    return p_, v_

                name = f"N={N} P={P} {kind}"
                tm = ki == 0 or a.time_all
                t1 = pl.timed(f"tiles {name}", tiles, dev, a.iters) if tm else 0.0
                t2 = pl.timed(f"pipe  {name}", pipe, dev, a.iters) if tm else 0.0
                t3 = pl.timed(f"skip  {name}", pipex, dev, a.iters) if tm and a.skip else 0.0
                rp, rv = (x.cpu() for x in torch.compile(tiles, **pl.OPTS)(*dev))
                gp, gv = (x.cpu() for x in torch.compile(pipe, **pl.OPTS)(*dev))
                same = torch.equal(rp, gp) and torch.equal(rv.view(torch.int32), gv.view(torch.int32))
                bad = (gp != want_p).any(-1)
                msg = (f"  {name}: pipe == tiles bit for bit: {same} ({int((rp != gp).any(-1).sum())} rows differ in "
                       f"pools, {int((rv.view(torch.int32) != gv.view(torch.int32)).any(-1).sum())} in scores); pipe "
                       f"== emulate: {not bad.any().item()} ({int(bad.sum())} of {N} rows differ; tiles "
                       f"{int((rp != want_p).any(-1).sum())})")
                if t1 and t2 and t1 == t1 and t2 == t2:
                    msg += (f"; tiles {t1 * 1e3:.3f} ms ({t1 * 1e9 / (N * P):.3f} ns/pair), pipe {t2 * 1e3:.3f} ms "
                            f"({t2 * 1e9 / (N * P):.3f} ns/pair), {t2 / t1:.3f}x")
                pl.say(msg, flush=True)
                if a.pe:
                    def pipe_pe(q_, w_, pk_, np_):
                        p_, _, v_ = dpx.select(q_, w_, pk_, np_, keep, scale, vorder=a.vorder, skip=False, pe=True)
                        return p_, v_

                    t4 = pl.timed(f"pe    {name}", pipe_pe, dev, a.iters) if tm else 0.0
                    ep, ev = (x.cpu() for x in torch.compile(pipe_pe, **pl.OPTS)(*dev))
                    cand = torch.where(torch.arange(P).view(1, P) < npool.view(N, 1), 0.0, pdl.NEG_INF)
                    esc = torch.cat([dl.emulate_scores(q[i:i + 64], w[i:i + 64], pk, cand[i:i + 64], scale)
                                     for i in range(0, N, 64)])  # [N, P] fp32
                    cnt = npool.clamp(0, P).clamp(max=keep)
                    live = torch.arange(keep).view(1, keep) < cnt.view(N, 1)
                    at = esc.gather(1, ep.clamp(0, P - 1))
                    vis = esc.gather(1, ep.clamp(0, P - 1)) > -5e29
                    rowmax = torch.where(esc > -5e29, esc.abs(), 0.0).amax(-1, keepdim=True).clamp(min=1e-30)
                    ok = live & vis & (at.abs() > 1e-3 * rowmax)
                    rel = ((ev - at).abs() / at.abs().clamp(min=1e-30))[ok]
                    nd = (ev - at).abs() / rowmax
                    nrm = nd[live & vis]
                    wr, wj = divmod(int(torch.where(live & vis, nd, -1.0).flatten().argmax()), keep)
                    pl.say(f"    worst entry: row {wr} (npool {int(npool[wr])}) slot {wj} pool {int(ep[wr, wj])}: pe "
                           f"{ev[wr, wj].item():.6e}, fp32 {at[wr, wj].item():.6e}, row max |score| "
                           f"{rowmax[wr, 0].item():.4e}", flush=True)
                    diff = []
                    for r in range(N):
                        c = int(cnt[r])
                        if c:
                            diff.append(len(set(ep[r, :c].tolist()) - set(rp[r, :c].tolist())))
                    dt = torch.tensor(diff, dtype=torch.float32)
                    msg = (f"  {name}: pe vs fp32 emulate_scores at its selected pools: rel max {rel.max().item():.2e} "
                           f"mean {rel.mean().item():.2e} ({rel.numel()} entries), |d| / row max: max "
                           f"{nrm.max().item():.2e} mean {nrm.mean().item():.2e}; selected sets vs the exact per-tile "
                           f"form: {int((dt > 0).sum())} of {len(diff)} rows differ, entries differing per row mean "
                           f"{dt.mean().item():.2f} max {int(dt.max().item()) if len(diff) else 0} (of keep {keep})")
                    if t4 and t4 == t4 and t2 == t2 and t1 == t1:
                        msg += (f"; pe {t4 * 1e3:.3f} ms ({t4 * 1e3 / -(-N // 128):.3f} ms per tile), {t4 / t2:.3f}x "
                                f"pipe, {t4 / t1:.3f}x tiles")
                    pl.say(msg, flush=True)
                if a.skip:
                    xp, xv = (x.cpu() for x in torch.compile(pipex, **pl.OPTS)(*dev))
                    same = torch.equal(rp, xp) and torch.equal(rv.view(torch.int32), xv.view(torch.int32))
                    _, nbt = dpx.chunk_bounds(npool, N, P, dl.pick_sub(P, keep), dpx.GROUP)
                    msg = (f"  {name}: skip == tiles bit for bit: {same} ({int((rp != xp).any(-1).sum())} rows differ "
                           f"in pools, {int((rv.view(torch.int32) != xv.view(torch.int32)).any(-1).sum())} in scores); "
                           f"chunks scored per tile {[int(v) for v in (nbt[:, 0, 0] * dl.pick_sub(P, keep) / dl.CH)]}")
                    if t3 and t3 == t3 and t2 == t2 and t1 == t1:
                        msg += f"; skip {t3 * 1e3:.3f} ms, {t3 / t2:.3f}x pipe, {t3 / t1:.3f}x tiles"
                    pl.say(msg, flush=True)


if __name__ == "__main__":
    main()
