"""What a collective MISMATCH between ranks does, with the Neuron runtime's per-execution barrier on or off.

NEURON_RT_DISABLE_EXECUTION_BARRIER=1 removes the ~5 ms fixed cost of every graph execution that holds a cross-chip
collective (docs/neuron-notes.md "The runtime's per-execution barrier"). nkipy's commit that turns it on by default
(aws-neuron/nkipy 1089b54) says the barrier is "the runtime's only mid-run detector of mismatched graphs across
ranks". This probe builds the mismatch on purpose and reports, per rank, whether the read-back of the mismatched call
returns, raises (and with what), or hangs, and after how long.

    python tools/probe_mismatch.py --case shape [--ranks 32] [--wait 300]

Cases (every rank first runs one correct all-reduce graph and a gloo barrier, so all graphs are loaded and warm):
  shape    rank 0 all-reduces [4, H], the other ranks [8, H]
  kind     rank 0 all-reduces [R, H], the other ranks reduce-scatter [R, H] (the same input bytes)
  group    rank 0 all-reduces over the world, the other ranks over their attention group of 8 (rank 0's group too)
  missing  rank 0 runs a graph with no collective, then the world all-reduce; the others run the all-reduce twice
  order    rank 0 runs all-reduce A ([4, H]) then B ([4, 2H]), the other ranks B then A
  none     no mismatch: the correct graph again (the control)
  race     no mismatch, a stress of what the barrier also orders: --iters launches cycling through three graphs
           with collectives (world all-reduce, world reduce-scatter, group-of-8 all-reduce) over four inputs each,
           queued without read-backs (one read-back 48 launches behind, as the engine bounds its queue), every
           rank sleeping a random 0-3 ms before a quarter of its launches so the ranks drift apart; every output
           (a position-weighted sum, so stale or misplaced rows change it) must equal its synchronous reference
Each rank prints `RANK <r> <case> <outcome> after <s> s`: outcome `ok <checksum>`, `raised <message>` or `hung` (the
read-back had not returned when --wait ran out; the rank then exits with code 3). A hang is detected from a watchdog
thread, because a read-back blocked inside the runtime cannot be interrupted from Python.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def rank_main(rank: int, port: int, world: int, case: str, wait: float, iters: int = 0) -> None:
    import profile_layer as pl
    from kiln.engine import tp

    tp.neuron_env(rank, port, int(os.environ.get("KILN_CORE_BASE", "0")))
    tp.init_rank(rank, world, port)
    pl.setup_device()
    import torch.distributed as dist
    import torch.distributed._functional_collectives as funcol

    grp = dist.group.WORLD

    def host_barrier():  # dist.barrier() asks torch for the accelerator, which the neuron device does not register
        dist.all_reduce(torch.zeros(1))
    eight = tp.attention_group(world, 8)
    H = 4096
    bf = torch.bfloat16

    def x_of(rows, cols=H):
        return (torch.randn(rows, cols, generator=torch.Generator().manual_seed(rank)) * 0.1).to(bf).to(pl.DEV)

    def ar(v):
        return funcol.all_reduce(v, "sum", grp) * 0.5

    def ar8(v):
        return funcol.all_reduce(v, "sum", eight) * 0.5

    def rs(v):
        return funcol.reduce_scatter_tensor(v, "sum", 0, grp) * 0.5

    def plain(v):
        return v * 0.5 + 1.0

    def compiled(f):  # a position-weighted sum: a stale, swapped or misplaced row changes it
        def g(v):
            o = f(v).float()
            w = torch.arange(o.shape[0], device=o.device, dtype=torch.float32).view(-1, 1) + 1.0
            return (o * w).sum()
        return torch.compile(g, **pl.OPTS)

    car, car8, crs, cplain = compiled(ar), compiled(ar8), compiled(rs), compiled(plain)
    x4, x8, x4w, x32 = x_of(4), x_of(8), x_of(4, 2 * H), x_of(world * 4)
    # Warm every graph this case can launch, all ranks together and in the same order, so the mismatch below is the
    # only thing that differs (a first load or a compile would line the ranks up behind it).
    for f, v in ((car, x4), (car, x8), (car, x4w), (car8, x4), (crs, x32), (car, x32), (cplain, x4)):
        f(v).cpu()
    host_barrier()

    if case == "race":
        import random

        rng = random.Random(1000 + rank)
        cag = compiled(lambda v: funcol.all_gather_tensor(v, 0, grp))
        xs = [(torch.randn(world * 4, H, generator=torch.Generator().manual_seed(100 * rank + k)) * 0.1).to(bf)
              .to(pl.DEV) for k in range(4)]
        named = {"ar": car, "rs": crs, "ar8": car8, "ag": cag, "plain": cplain}
        # which graphs the race cycles through, "," or "+" separated (tools/hv_mismatch.sh splits its env on ",")
        pick = os.environ.get("PROBE_RACE_GRAPHS", "ar,rs,ar8,ag").replace("+", ",").split(",")
        graphs = [named[n] for n in pick]
        p_sleep = float(os.environ.get("PROBE_RACE_SLEEP", "0.25"))  # share of launches preceded by a 0-3 ms sleep
        ref = {}
        for gi, f in enumerate(graphs):
            for k, v in enumerate(xs):
                ref[gi, k] = f(v).cpu().item()
        host_barrier()
        t0 = time.perf_counter()
        outs, bad = [], 0
        for i in range(iters):
            gi, k = i % len(graphs), (i // len(graphs)) % len(xs)
            if rng.random() < p_sleep:
                time.sleep(rng.uniform(0.0, 0.003))
            outs.append((gi, k, graphs[gi](xs[k])))
            if len(outs) > 48:
                outs[-49][2].cpu()
            if rank in (0, 8) and i % 250 == 0:
                print(f"RANK {rank} race at launch {i} after {time.perf_counter() - t0:.2f} s", flush=True)
        for gi, k, o in outs:
            if o.cpu().item() != ref[gi, k]:
                bad += 1
        print(f"RANK {rank} race {bad} mismatches of {iters} launches ({','.join(pick)}, sleep {p_sleep}) after "
              f"{time.perf_counter() - t0:.2f} s", flush=True)
        host_barrier()
        os._exit(1 if bad else 0)

    if case == "none":
        plan = [(car, x4)]
    elif case == "shape":
        plan = [(car, x4)] if rank == 0 else [(car, x8)]
    elif case == "kind":
        plan = [(car, x32)] if rank == 0 else [(crs, x32)]
    elif case == "group":
        plan = [(car, x4)] if rank == 0 else [(car8, x4)]
    elif case == "missing":
        plan = [(cplain, x4), (car, x4)] if rank == 0 else [(car, x4), (car, x4)]
    elif case == "order":
        plan = [(car, x4), (car, x4w)] if rank == 0 else [(car, x4w), (car, x4)]
    else:
        raise ValueError(case)

    done = threading.Event()
    t0 = time.perf_counter()

    def watchdog():
        if not done.wait(wait):
            print(f"RANK {rank} {case} hung after {time.perf_counter() - t0:.1f} s", flush=True)
            os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()
    # KILN_PROBE_WATCHDOG=<s>: the engine's exec watchdog (kiln/engine/watchdog.py) around the launches and the
    # read-back, as ModelRunner._exec and LLMEngine._collect use it; it must end a blocked rank with EXIT_HANG and
    # name the calls before the probe's own --wait does.
    from kiln.engine import watchdog as kwd

    wd = kwd.ExecWatchdog(rank, float(os.environ["KILN_PROBE_WATCHDOG"])) if os.environ.get("KILN_PROBE_WATCHDOG") \
        else None
    try:
        outs = []
        for i, (f, v) in enumerate(plan):
            if wd is not None:
                wd.launched(f"probe-{case}", (i, tuple(v.shape)))
            with kwd.watched(wd, f"launching probe-{case} call {i}"):
                outs.append(f(v))
        with kwd.watched(wd, f"reading back probe-{case}"):
            vals = [o.cpu().item() for o in outs]
        done.set()
        print(f"RANK {rank} {case} ok {' '.join(f'{v:.4f}' for v in vals)} after {time.perf_counter() - t0:.2f} s",
              flush=True)
    except Exception as e:  # noqa: BLE001 - the probe reports whatever the runtime raises
        done.set()
        msg = " ".join(str(e).split())[:400]
        print(f"RANK {rank} {case} raised {type(e).__name__}: {msg} after {time.perf_counter() - t0:.2f} s", flush=True)
        os._exit(2)
    # A rank that finished while others hang must not tear the group down under them (a vanished peer is a different
    # failure): stay until the others' watchdogs have fired.
    time.sleep(max(0.0, t0 + wait + 5 - time.perf_counter()))
    os._exit(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=["none", "shape", "kind", "group", "missing", "order", "race"])
    ap.add_argument("--ranks", type=int, default=32)
    ap.add_argument("--wait", type=float, default=300.0, help="seconds a read-back may take before it counts as hung")
    ap.add_argument("--iters", type=int, default=3000, help="launches in the race case")
    args = ap.parse_args()
    import multiprocessing as mp

    from kiln.engine import tp

    port = tp.free_port()
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=rank_main, args=(r, port, args.ranks, args.case, args.wait, args.iters)) for r in range(args.ranks)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(args.wait + 600)
    for p in ps:
        if p.is_alive():
            p.kill()
    codes = [p.exitcode for p in ps]
    print(f"case {args.case}, ranks {args.ranks}, barrier "
          f"{'off' if os.environ.get('NEURON_RT_DISABLE_EXECUTION_BARRIER') == '1' else 'on'}, "
          f"NEURON_RT_EXEC_TIMEOUT={os.environ.get('NEURON_RT_EXEC_TIMEOUT', 'unset')}; exit codes "
          f"{ {c: codes.count(c) for c in sorted(set(codes), key=str)} }", flush=True)


if __name__ == "__main__":
    main()
