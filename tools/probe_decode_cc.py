"""The per-collective latency of a decode step's small all-reduces, inside one graph, at N ranks: XLA (the form the decode
graphs use) against all-reduces issued from inside an NKI kernel (nki.collectives), and against smaller groups.

Why: a GLM-5.3-Flash decode call at 1 row per DP group spends 37.4 of its 79 ms in 92 world all-reduces of 32 KB
(0.196 ms trigger-to-start wait + 0.215 ms transfer each; util_report replay with captured inputs, trn1.32xlarge,
2026-10-05, docs/neuron-notes.md "Decode at scale"): at small batch the collectives, not the compute, set the step's
fixed cost. A graph-level number mixes the graph's own launch cost in, so every case here is a CHAIN of n collectives
in one graph, each fed by the previous one's output, timed at n = 1 and n = --chain: the difference over n - 1 is the
cost of one more collective inside a graph.

    python tools/probe_decode_cc.py [--ranks 32] [--rows 4 16] [--hidden 4096] [--chain 16] [--cases xar,xgar,kar,kar1]

Cases (rows x hidden bf16 per rank, the decode all-reduce's shape: 4 rows per group at 1 row per DP group x 4 groups is
[4, 4096], 32 KB):
  xar   XLA all-reduce over the world (torch.distributed functional collectives, as the decode graphs issue them),
        with a multiply by 1 / ranks between links so values stay finite
  xgar  the same inside groups of --group consecutive ranks (an attention group: 8 ranks = 4 trn1 chips)
  kar   one NKI kernel holding the whole chain of n all_reduces (private HBM scratch in and out: a collective cannot read
        or write the kernel's IO tensors, neuronx-cc 2.27 checkCollective)
  kar1  n kernels of one all_reduce each, chained in one graph (the form a fused per-layer kernel would issue)
  xarc  xar with a fixed block of compute after every all-reduce (--mm matmuls of [rows, H] x [H, H] bf16, the same
        on every rank: a decode layer's projections stand-in), and xc the same compute without the all-reduces: their
        difference over n is what an all-reduce costs between real compute
Each case's NEFF is printed (compile-cache key), for `neuron-explorer capture -n <neff> -r <ranks> -i 0`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

REV = 1  # bump after editing a kernel below: LNL's cache key does not include NKI kernel source

try:  # the Neuron venv
    import nki
    import nki.collectives as ncc
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

if nki is not None:
    def _group(world: int):
        ranks = []  # the NKI tracer takes a list literal, not list(range(...))
        for i in range(world):
            ranks.append(i)
        return ncc.ReplicaGroup([ranks])

    def _hbm_copy(dst, src, R: int, H: int, dt):
        T = nl.ndarray((R, H), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=T, src=src)
        nisa.dma_copy(dst=dst, src=T)

    @nki.jit
    def kiln_cc_chain(y, world: int, n: int, rev: int):
        """n all_reduces of y [R <= 128, H] in a chain (each one's output the next one's input), private HBM scratch."""
        R, H = y.shape
        a = nl.ndarray((R, H), dtype=y.dtype, buffer=nl.private_hbm)
        b = nl.ndarray((R, H), dtype=y.dtype, buffer=nl.private_hbm)
        out = nl.ndarray((R, H), dtype=y.dtype, buffer=nl.shared_hbm)
        _hbm_copy(a, y, R, H, y.dtype)
        src = a
        dst = b
        for _ in range(n):  # (no tuple assignment: the device tracer wants simple variables)
            ncc.all_reduce(srcs=[src], dsts=[dst], replica_group=_group(world), op=nl.add)
            t = src
            src = dst
            dst = t
        _hbm_copy(out, src, R, H, y.dtype)
        return out


def rank_main(rank: int, port: int, world: int, rows: list[int], H: int, chain: int, group: int, cases: list[str],
              mm: int = 2):
    import profile_layer as pl
    from kiln.engine import tp

    tp.neuron_env(rank, port, int(os.environ.get("KILN_CORE_BASE", "0")))
    tp.init_rank(rank, world, port)
    pl.setup_device()
    import torch.distributed as dist
    import torch.distributed._functional_collectives as funcol
    from libtorch_neuronx_lite.compile.execute_context import ExecuteContext, set_execute_context
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    grp = dist.group.WORLD
    sub = None
    if "xgar" in cases:
        for g0 in range(0, world, group):
            ranks = list(range(g0, g0 + group))
            pg = dist.new_group(ranks)
            if rank in ranks:
                sub = pg
    last = []
    set_execute_context(ExecuteContext(pre_execute_hook=lambda ins, meta: last.append(meta.neff_id)))
    inv = 1.0 / world
    W = (torch.randn(H, H, generator=torch.Generator().manual_seed(3)) * H ** -0.5).to(torch.bfloat16).to(pl.DEV)

    def build(case: str, n: int):
        if case == "xar":
            def f(y):
                for _ in range(n):
                    y = funcol.all_reduce(y, "sum", grp) * inv
                return y.float().sum()
        elif case == "xgar":
            def f(y):
                for _ in range(n):
                    y = funcol.all_reduce(y, "sum", sub) * (1.0 / group)
                return y.float().sum()
        elif case in ("xarc", "xc"):
            def f(y):
                for _ in range(n):
                    if case == "xarc":
                        y = funcol.all_reduce(y, "sum", grp) * inv
                    for _ in range(mm):
                        y = y @ W
                return y.float().sum()
        elif case == "kar":
            def f(y):
                return wrap_nki(kiln_cc_chain)[1](y=y, world=world, n=n, rev=REV).float().sum()
        elif case == "kar1":
            def f(y):
                for _ in range(n):
                    y = wrap_nki(kiln_cc_chain)[1](y=y, world=world, n=1, rev=REV) * inv
                return y.float().sum()
        else:
            raise SystemExit(f"unknown case {case}")
        return torch.compile(f, **pl.OPTS)

    for R in rows:
        y = torch.full((R, H), 1.0, dtype=torch.bfloat16).to(pl.DEV)
        for case in cases:
            res = {}
            for n in (1, chain):
                c = build(case, n)
                try:
                    t = time.perf_counter()
                    v = float(c(y).cpu())
                    first = time.perf_counter() - t
                except Exception as e:  # a form that does not compile or run is a result too
                    if rank == 0:
                        print(f"  rows {R} {case} n={n}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                    res = None
                    break
                key = last[-1] if last else "?"
                ts = []
                for _ in range(30):
                    t = time.perf_counter()
                    c(y).cpu()
                    ts.append(time.perf_counter() - t)
                ts.sort()
                res[n] = ts[15]
                if rank == 0:
                    print(f"  rows {R} {case:5s} n={n:3d}: NEFF {key} first {first:5.1f} s, p50 {ts[15] * 1e3:8.3f} ms "
                          f"(value {v:.4g})", flush=True)
            if res and rank == 0:
                per = (res[chain] - res[1]) / (chain - 1) * 1e3
                print(f"rows {R} hidden {H} ({R * H * 2 // 1024} KB) {case}: {per:.3f} ms per collective in a graph "
                      f"({chain} vs 1)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=32)
    ap.add_argument("--rows", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--chain", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--cases", default="xar,xgar,kar,kar1")
    ap.add_argument("--mm", type=int, default=2)
    args = ap.parse_args()
    import multiprocessing as mp

    from kiln.engine import tp

    port = tp.free_port()
    ctx = mp.get_context("spawn")
    cases = args.cases.split(",")
    ps = [ctx.Process(target=rank_main, args=(r, port, args.ranks, args.rows, args.hidden, args.chain, args.group, cases,
                                                      args.mm))
          for r in range(args.ranks)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print(f"ranks {args.ranks}, rows {args.rows}, hidden {args.hidden}, chain {args.chain}; exit codes: "
          f"{sorted({p.exitcode for p in ps})}", flush=True)


if __name__ == "__main__":
    main()
