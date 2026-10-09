"""Does neuronx-cc overlap an XLA collective with independent compute in one graph (trn2, LNC=2)? --ranks processes (one logical
core each, from KILN_CORE_BASE), x [R, H] bf16 per rank, w [H, H] bf16:

    python tools/probe_cc_overlap.py [--ranks 32] [--rows 8192] [--hidden 4096]

  mm      y = x w                                        (the compute alone)
  rs      reduce_scatter(x) over the ranks               (the collective alone; R x H bf16 in, R / ranks rows out)
  chain   reduce_scatter(x w)                            (one dependent chain: no overlap possible)
  two     reduce_scatter(x_a w), reduce_scatter(x_b w)   (two independent half-size chains: the RS of one can overlap the
                                                          matmul of the other)
  If two ~ chain, the compiler serialises them (no overlap); if two ~ max(mm, rs) + rs / 2, it overlaps. Each graph returns a
  scalar sum, chained calls, rank 0 prints p50 ms.
"""
import argparse, os, sys, time
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def run(rank, world, port, R, H, iters):
    from kiln.engine import tp
    tp.neuron_env(rank, port, int(os.environ.get("KILN_CORE_BASE", "0")))
    tp.init_rank(rank, world, port)
    import profile_layer as pl
    pl.setup_device()
    import torch.distributed as dist
    import torch.distributed._functional_collectives as funcol
    from kiln.engine.model_runner import IN_FLIGHT
    grp = dist.group.WORLD
    g = torch.Generator().manual_seed(rank)
    x = (torch.randn(R, H, generator=g) * 0.1).bfloat16().to(pl.DEV)
    w = (torch.randn(H, H, generator=torch.Generator().manual_seed(7)) * H ** -0.5).bfloat16().to(pl.DEV)
    rs = lambda t: funcol.reduce_scatter_tensor(t, "sum", 0, grp)  # noqa: E731
    cases = {
        "mm": lambda x, w: (x @ w).float().sum(),
        "rs": lambda x, w: rs(x).float().sum(),
        "chain": lambda x, w: rs(x @ w).float().sum(),
        "two": lambda x, w: rs(x[: R // 2] @ w).float().sum() + rs(x[R // 2:] @ w).float().sum(),
    }
    for name, fn in cases.items():
        c = torch.compile(fn, **pl.OPTS)
        c(x, w).cpu()
        c(x, w).cpu()
        ts = []
        for _ in range(3):
            pend = []
            t0 = time.perf_counter()
            for _ in range(iters):
                pend.append(c(x, w))
                if len(pend) > IN_FLIGHT:
                    pend.pop(0).cpu()
            for o in pend:
                o.cpu()
            ts.append((time.perf_counter() - t0) / iters * 1e3)
        if rank == 0:
            print(f"RESULT {name}: {sorted(ts)[1]:.3f} ms per call chained (R {R}, H {H}, {world} ranks)", flush=True)
    dist.barrier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=32)
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--port", type=int, default=29711)
    a = ap.parse_args()
    mp.spawn(run, args=(a.ranks, a.port, a.rows, a.hidden, a.iters), nprocs=a.ranks, join=True)


if __name__ == "__main__":
    main()
