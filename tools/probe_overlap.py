"""Does neuronx-cc run a cross-chip collective concurrently with independent compute in the same graph? The question
two-batch overlap (SGLang two_batch_overlap.py, vLLM dbo) rests on: split a prefill call's rows into two micro-batches
so that one's collectives (the MoE FFN block's world gather and reduce-scatter) hide behind the other's compute. On a
GPU that is two streams; in one Neuron graph it is whatever the compiler's scheduler does with two independent chains.

    python tools/probe_overlap.py [--ranks 32] [--rows 128] [--hidden 4096] [--ffn 2048] [--reps 4]

Cases, each one graph whose outputs are summed to one number (a 4-byte read-back), timed at 32 ranks on rank 0: p50
of 40 synchronous calls and the mean of 20 calls queued back to back; each case prints its NEFF (compile-cache key), so
`neuron-explorer capture -n <neff> -r 32 -i 0` can replay it for the device's own timeline:
  gather            the SP world gather as served (models/decoder.py DecoderForCausalLM._sp_gather): a zero-padded
                    all-reduce of [ranks x rows, H] bf16
  mlp               a SwiGLU MLP over [ranks x rows, H] repeated --reps times (stands for the EP kernel's work)
  gather -> mlp     the MLP on the gathered rows: the dependent chain one micro-batch is today
  gather || mlp     the gather of x and the MLP of an unrelated y in one graph: no dependency between them
  two micro-batches rows split in halves a, b: gather(a) -> mlp(a), gather(b) -> mlp(b), in one graph; the
                    compiler may run gather(b) under mlp(a); "gathers first" issues both gathers before either mlp
  mlp || gather     gather || mlp with the compute first in program order
  layer             gather -> mlp -> reduce-scatter, one FFN block's shape; "layer two mb" the same on two halves
                    with both gathers issued first and each half's reduce-scatter right after its compute
If 'gather || mlp' is near max(gather, mlp) the compiler overlaps them; near the sum, it serialises.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def rank_main(rank: int, port: int, world: int, r: int, H: int, F: int, reps: int) -> None:
    import profile_layer as pl
    from kiln.engine import tp

    tp.neuron_env(rank, port, int(os.environ.get("KILN_CORE_BASE", "0")))
    tp.init_rank(rank, world, port)
    pl.setup_device()
    import torch.distributed as dist
    import torch.distributed._functional_collectives as funcol

    grp = dist.group.WORLD
    g = torch.Generator().manual_seed(rank)
    bf = torch.bfloat16
    x = (torch.randn(r, H, generator=g) * 0.1).to(bf).to(pl.DEV)
    y = (torch.randn(world * r, H, generator=g) * 0.1).to(bf).to(pl.DEV)
    wg = (torch.randn(H, 2 * F, generator=torch.Generator().manual_seed(7)) * 0.02).to(bf).to(pl.DEV)
    wd = (torch.randn(F, H, generator=torch.Generator().manual_seed(8)) * 0.02).to(bf).to(pl.DEV)
    onehot = torch.zeros(world, dtype=bf)
    onehot[rank] = 1
    oh = onehot.to(pl.DEV)

    def gather(v):  # the zero-padded all-reduce gather of DecoderForCausalLM._sp_gather
        return funcol.all_reduce((v.unsqueeze(0) * oh.view(world, 1, 1)).reshape(-1, v.shape[-1]), "sum", grp)

    def mlp(h):
        for _ in range(reps):
            gu = h @ wg
            h = h + (torch.nn.functional.silu(gu[:, :F]) * gu[:, F:]) @ wd
        return h

    def c_gather(a, b):
        return gather(a)

    def c_mlp(a, b):
        return mlp(b)

    def c_dep(a, b):
        return mlp(gather(a))

    def c_indep(a, b):
        return gather(a), mlp(b)

    def c_two(a, b):
        h = r // 2
        return mlp(gather(a[:h])), mlp(gather(a[h:]))

    def scalar(f):  # every output reduced to one number, so the read-back is 4 bytes and nothing is dead code
        def g(a, b):
            out = f(a, b)
            return sum(o.float().sum() for o in (out if isinstance(out, tuple) else (out,)))
        return g

    from libtorch_neuronx_lite.compile.execute_context import ExecuteContext, set_execute_context

    last = []
    set_execute_context(ExecuteContext(pre_execute_hook=lambda ins, meta: last.append(meta.neff_id)))
    def c_indep_rev(a, b):  # the same as c_indep, the compute first in program order
        return mlp(b), gather(a)

    def c_two_cf(a, b):  # both halves' gathers issued before either half's compute
        h = r // 2
        ga, gb = gather(a[:h]), gather(a[h:])
        return mlp(ga), mlp(gb)

    def rs(v):
        return funcol.reduce_scatter_tensor(v, "sum", 0, grp)

    def c_layer(a, b):  # one FFN block as served: gather, compute, reduce-scatter
        return rs(mlp(gather(a)))

    def c_layer_tbo(a, b):  # two half-batches, each one's collectives issued around the other's compute
        h = r // 2
        ga = gather(a[:h])
        gb = gather(a[h:])
        ya = mlp(ga)
        ra = rs(ya)
        yb = mlp(gb)
        return ra, rs(yb)

    def rs_cols(v, n):  # the reduce-scatter of [R, H] as n reduce-scatters of [R, H / n] column blocks: the same rows
        return torch.cat([rs(c.contiguous()) for c in v.chunk(n, dim=1)], dim=1)

    def c_rs32(a, b):  # the FFN block's world reduce-scatter alone: [ranks x rows, H] bf16 (32 MiB at the served shape)
        return rs(b)

    def c_rs16x2(a, b):
        return rs_cols(b, 2)

    def c_rs8x4(a, b):
        return rs_cols(b, 4)

    def c_layer_cols2(a, b):  # c_layer with its reduce-scatter as two column halves
        return rs_cols(mlp(gather(a)), 2)

    def c_layer_cols4(a, b):
        return rs_cols(mlp(gather(a)), 4)

    cases = [("rs 32 MiB", c_rs32), ("rs 2 x 16 MiB columns", c_rs16x2), ("rs 4 x 8 MiB columns", c_rs8x4),
             ("layer rs columns 2", c_layer_cols2), ("layer rs columns 4", c_layer_cols4),
             ("gather", c_gather), ("mlp", c_mlp), ("gather -> mlp", c_dep), ("gather || mlp", c_indep),
             ("mlp || gather", c_indep_rev), ("two micro-batches", c_two), ("two mb gathers first", c_two_cf),
             ("layer", c_layer), ("layer two mb", c_layer_tbo)]
    only = os.environ.get("PROBE_CASES")
    if only:
        cases = [c for c in cases if c[0] in only.split(",")]
    for name, f in cases:
        c = torch.compile(scalar(f), **pl.OPTS)
        t = time.perf_counter()
        c(x, y).cpu()
        first = time.perf_counter() - t
        key = last[-1] if last else "?"
        ts = []
        for _ in range(40):
            t = time.perf_counter()
            c(x, y).cpu()
            ts.append(time.perf_counter() - t)
        t = time.perf_counter()
        outs = [c(x, y) for _ in range(20)]  # queued back to back, as serving runs them
        outs[-1].cpu()
        queued = (time.perf_counter() - t) / 20
        ts.sort()
        if rank == 0:
            print(f"  {name:<20} NEFF {key}: first call {first:6.1f} s, synchronous p50 {ts[20] * 1e3:8.3f} ms min "
                  f"{ts[0] * 1e3:8.3f} ms, 20 queued {queued * 1e3:8.3f} ms each", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=32)
    ap.add_argument("--rows", type=int, default=128, help="rows each rank holds (sequence-parallel streams)")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--ffn", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=4, help="MLPs in the compute chain")
    args = ap.parse_args()
    import multiprocessing as mp

    from kiln.engine import tp

    port = tp.free_port()
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=rank_main, args=(r, port, args.ranks, args.rows, args.hidden, args.ffn, args.reps))
          for r in range(args.ranks)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print(f"ranks {args.ranks}, rows per rank {args.rows}, hidden {args.hidden}, ffn {args.ffn}, reps {args.reps}; "
          f"exit codes: {sorted({p.exitcode for p in ps})}", flush=True)


if __name__ == "__main__":
    main()
