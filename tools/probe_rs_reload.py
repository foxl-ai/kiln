"""Whether a graph holding a world reduce-scatter (or an all-to-all used as a gather) survives being loaded
from the compile cache by a LATER process, as an all-gather graph does not ("A cached all-gather NEFF breaks in the next process",
docs/neuron-notes.md): run this twice; the second run must load the first run's NEFF and give the
same, correct sums. Also times the reduce-scatter against the all-reduce + own rows it would replace
(models/hybrid.py's sequence-parallel block outputs).

    python tools/probe_rs_reload.py [--ranks 32] [--rows 128] [--hidden 4096]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def rank_main(rank: int, port: int, world: int, r: int, H: int) -> None:
    import profile_layer as pl
    from kiln.engine import tp

    tp.neuron_env(rank, port, int(os.environ.get("KILN_CORE_BASE", "0")))
    tp.init_rank(rank, world, port)
    pl.setup_device()
    import torch.distributed as dist
    import torch.distributed._functional_collectives as funcol

    grp = dist.group.WORLD
    parts = [torch.randn(world * r, H, generator=torch.Generator().manual_seed(m)).to(torch.bfloat16) for m in range(world)]
    want = sum(p.float() for p in parts)[rank * r:(rank + 1) * r]
    x = parts[rank].to(pl.DEV)
    onehot = torch.zeros(world, dtype=torch.bfloat16)
    onehot[rank] = 1
    idx = torch.tensor([rank]).to(pl.DEV)

    def rs(y):
        return funcol.reduce_scatter_tensor(y, "sum", 0, grp)

    def ar_rows(y):  # the current form: all-reduce the whole batch, keep this rank's rows
        return funcol.all_reduce(y, "sum", grp).reshape(world, r, H).index_select(0, idx)[0]

    own = torch.randn(r, H, generator=torch.Generator().manual_seed(100 + rank)).to(torch.bfloat16)
    rows_all = torch.cat([torch.randn(r, H, generator=torch.Generator().manual_seed(100 + m)).to(torch.bfloat16)
                          for m in range(world)]).float()
    y_own = own.to(pl.DEV)

    def a2a_gather(y):  # every rank's rows from an all-to-all of this rank's rows sent to every rank
        return funcol.all_to_all_single(y.repeat(world, 1), None, None, grp)

    def ar_gather(y):  # the current form (DecoderForCausalLM._sp_gather): a zero-padded all-reduce
        return funcol.all_reduce((y.unsqueeze(0) * onehot.to(pl.DEV).view(world, 1, 1)).reshape(-1, H), "sum", grp)

    cases = [("reduce-scatter", rs, x, want), ("all-reduce + own rows", ar_rows, x, want),
             ("all-to-all gather", a2a_gather, y_own, rows_all), ("zero-padded all-reduce gather", ar_gather, y_own, rows_all)]
    for name, f, inp, ref in cases:
        c = torch.compile(f, **pl.OPTS)
        t = time.perf_counter()
        got = c(inp).cpu().float()
        first = time.perf_counter() - t
        err = (got - ref).abs().max().item()
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            c(inp).cpu()
            ts.append(time.perf_counter() - t)
        ts.sort()
        if rank == 0:
            print(f"  {name:<30} {tuple(inp.shape)} -> {tuple(got.shape)} bf16: first call {first:6.1f} s, p50 "
                  f"{ts[10] * 1e3:7.3f} ms, max |err| {err:.3e} (rank 0)", flush=True)
        if err > 1.0:  # a 32-term bf16 sum moves by a few of its ulps
            print(f"  rank {rank} {name}: WRONG, max |err| {err:.3e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=32)
    ap.add_argument("--rows", type=int, default=128, help="rows each rank keeps")
    ap.add_argument("--hidden", type=int, default=4096)
    args = ap.parse_args()
    import multiprocessing as mp

    from kiln.engine import tp

    port = tp.free_port()
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=rank_main, args=(r, port, args.ranks, args.rows, args.hidden)) for r in range(args.ranks)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print("exit codes:", sorted({p.exitcode for p in ps}), flush=True)


if __name__ == "__main__":
    main()
