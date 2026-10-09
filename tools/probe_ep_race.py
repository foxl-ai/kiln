"""Repeat-execution probe of the expert-parallel MoE kernel (kiln/kernels/moe_ep.py) inside a multi-layer graph on N
ranks, the shape of an EP decode group graph at LNC=2 (trn2): each rank holds its 9 whole experts (288 / 32), every
layer calls moe_ep() on the same C rows and routing as every other rank, the ranks' outputs are summed by an XLA
all-reduce (the block's own reduction under EP), and the next layer's input is x + y. The graph is executed --execs
times; every execution's outputs must equal the first's bit for bit (the kernel is deterministic), layer 0's local
output must equal moe_ep.emulate, and nothing may hang (the serving engine's EP decode graphs hung on their first
execution at LNC=2, not deterministically: docs/neuron-notes.md "trn2 on engine-v0 70ddc1b").

    python tools/probe_ep_race.py --ranks 32 --rows 128 --layers 12 --execs 20 [--routing uniform|skew|hot]
        [--fit row|block] [--no-ar] [--between none|xla|split]

--between split puts another LNC-split NKI kernel (kernels/dsa_topk.py's prefill top-k, KILN_LNC_SPLIT naming
dsa_topk) between the layers, the way an EP decode graph interleaves the MoE kernel with the other split kernels.
Run with KILN_CORE_BASE=<first logical core>. Every rank prints only when something is wrong; rank 0 prints the
summary. Exit 0 = every execution completed and matched.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def routing(kind: str, C: int, K: int, E: int, El: int, layer: int):
    g = torch.Generator().manual_seed(1000 + layer)
    topi = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(C)])
    if kind == "skew":  # a third of the tokens' first expert is expert 1 (rank 0's), so it has > 16 pairs
        sel = torch.rand(C, generator=g) < 1 / 3
        for t in sel.nonzero().flatten().tolist():
            if 1 not in topi[t].tolist():
                topi[t, 0] = 1
    elif kind == "hot":  # every rank's first local expert takes a share of the tokens: every rank > 16 pairs
        for t in range(C):
            hot = (t % (E // El)) * El
            if hot not in topi[t].tolist():
                topi[t, 0] = hot
    topv = (torch.rand(C, K, generator=g) + 0.1).bfloat16()
    return topi, topv


def rank_main(rank: int, port: int, a) -> None:
    import profile_layer as pl
    from kiln.engine import tp

    tp.neuron_env(rank, port, int(os.environ.get("KILN_CORE_BASE", "0")))
    if a.ranks > 1:
        tp.init_rank(rank, a.ranks, port)
    pl.setup_device()
    import torch.distributed as dist
    import torch.distributed._functional_collectives as funcol

    from kiln.kernels import moe_ep
    from probe_moe_ep import ep_experts

    say = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)
    world_eff = a.world or a.ranks
    E, K, H, C, L = 288, 8, 4096, a.rows, a.layers
    El = E // world_eff
    my = rank % world_eff
    t0 = time.time()
    ws = ep_experts(El, H, 2048, seed=my, block=a.fit == "block")
    blob = moe_ep.pack(*ws, tiles=a.fit == "block")
    bn = [k for k in ("gu", "sgu", "dn", "sdn", "dsg", "dsd", "tsg", "tsd") if k in blob]
    owner = torch.arange(E) // El
    lmap = moe_ep.local_map(owner, my)
    routs = [routing(a.routing, C, K, E, El, l) for l in range(L)]
    x0 = torch.randn(C, H, generator=torch.Generator().manual_seed(7)).bfloat16()
    say(f"ranks {a.ranks} (experts as {world_eff}), C={C}, layers {L}, routing {a.routing}, fit {a.fit}, "
        f"LNC split {os.environ.get('KILN_LNC_SPLIT', '(default)')}, small rows {moe_ep.SMALL_ROWS} v{moe_ep.SMALL_V} "
        f"LW {moe_ep.SMALL_LW}, between {a.between}, all-reduce {not a.no_ar}; experts built in {time.time() - t0:.0f} s")
    loc = lmap.view(-1)[routs[0][0]]
    n = torch.bincount(loc.flatten(), minlength=El + 1)[:El]
    say(f"  layer 0 on rank 0: {int(n.sum())} local pairs, per expert {n.tolist()}")
    dev = [blob[k].to(pl.DEV) for k in bn]
    ti = torch.stack([r[0] for r in routs]).to(torch.int32).to(pl.DEV)
    tv = torch.stack([r[1] for r in routs]).to(pl.DEV)
    grp = dist.group.WORLD if a.ranks > 1 else None
    from kiln.kernels import dsa_topk

    def between(v, l):
        if a.between == "xla":  # an XLA op reading every row of the MoE output (a stand-in for the norm)
            return v * torch.rsqrt(v.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(v.dtype)
        if a.between == "split":  # another LNC-split NKI kernel (dsa_topk, split at LNC=2 by default) on the rows
            m = dsa_topk.select(v[:, :1024].float(), 64)  # [C, 1024] 0 / NEG_INF
            kept = torch.exp(m).sum(-1, keepdim=True)  # 64 per row (no float-literal compare: NCC_ESPP004 f64)
            return (v.float() + 1e-3 * (kept - 64.0)).to(v.dtype)
        return v

    def graph(x, ti_, tv_, lm, *bt):
        b = dict(zip(bn, bt))
        y0 = None
        for l in range(L):
            y = moe_ep.moe_ep(x, tv_[l], ti_[l], b, lm, 1, 10.0)
            if y0 is None:
                y0 = y
            if grp is not None and not a.no_ar:
                y = funcol.all_reduce(y, "sum", grp)
            x = (x.float() + 0.25 * y.float()).bfloat16()
            x = between(x, l)
        return x, y0

    c = torch.compile(graph, **pl.OPTS)
    args = (x0.to(pl.DEV), ti, tv, lmap.to(pl.DEV), *dev)
    t = time.perf_counter()
    try:
        first = [o.cpu() for o in c(*args)]
    except Exception as e:
        print(f"rank {rank}: FIRST EXECUTION FAILED {type(e).__name__}: {str(e)[:1500]}", flush=True)
        raise
    say(f"  compile + first execution {time.perf_counter() - t:.1f} s")
    emu = moe_ep.emulate(x0, routs[0][1], routs[0][0], lmap, *ws, 1, 10.0).float()
    d0 = (first[1].float() - emu).abs().max().item()
    sc = max(emu.abs().max().item(), 1e-30)
    bad = 0 if d0 / sc < 0.02 else 1
    if bad:
        print(f"rank {rank}: layer 0 vs emulation rel {d0 / sc:.4f}", flush=True)
    say(f"  layer 0 vs emulation (rank 0) rel {d0 / sc:.4f}, final x finite {bool(torch.isfinite(first[0].float()).all())}")
    diffs = 0
    ts = []
    for i in range(a.execs):
        t = time.perf_counter()
        try:
            got = [o.cpu() for o in c(*args)]
        except Exception as e:
            print(f"rank {rank}: EXECUTION {i + 2} FAILED {type(e).__name__}: {str(e)[:1500]}", flush=True)
            raise
        ts.append(time.perf_counter() - t)
        if not all(torch.equal(g_, f_) for g_, f_ in zip(got, first)):
            diffs += 1
            if diffs <= 3:
                dd = [(g_.float() - f_.float()).abs().max().item() for g_, f_ in zip(got, first)]
                print(f"rank {rank}: execution {i + 2} differs from the first: max |d| {dd}", flush=True)
    if grp is not None:
        tot = torch.tensor([float(diffs), float(bad)])
        dist.all_reduce(tot)
        diffs_all, bad_all = int(tot[0]), int(tot[1])
    else:
        diffs_all, bad_all = diffs, bad
    ts.sort()
    say(f"RESULT {a.execs + 1} executions completed on {a.ranks} ranks; executions differing from the first, summed "
        f"over ranks: {diffs_all}; ranks off the emulation: {bad_all}; p50 {ts[len(ts) // 2] * 1e3:.2f} ms" if ts else
        "RESULT one execution")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=32)
    ap.add_argument("--world", type=int, default=0, help="experts laid out as for this many ranks (default --ranks)")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--execs", type=int, default=20)
    ap.add_argument("--routing", default="uniform", choices=["uniform", "skew", "hot"])
    ap.add_argument("--fit", default="row", choices=["row", "block"])
    ap.add_argument("--no-ar", action="store_true")
    ap.add_argument("--between", default="none", choices=["none", "xla", "split"])
    a = ap.parse_args()
    if a.ranks == 1:
        rank_main(0, 0, a)
        return
    import multiprocessing as mp

    from kiln.engine import tp

    port = tp.free_port()
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=rank_main, args=(r, port, a)) for r in range(a.ranks)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    codes = sorted({p.exitcode for p in ps})
    print("exit codes:", codes, flush=True)
    sys.exit(0 if codes == [0] else 1)


if __name__ == "__main__":
    main()
