"""The NKI chunked delta rule (kiln/kernels/delta_rule.py) on one NeuronCore against the torch path
(models/linear_attn.chunk_scan, what a prefill chunk runs without KILN_LINEAR_ATTN_KERNEL=nki), at
one tensor-parallel rank's shapes, per prefill chunk length.

    python tools/probe_delta_rule.py --kind kda --heads 2 --chunks 512 2048 8192 [--torch-max 2048]
    python tools/probe_delta_rule.py --kind gdn --heads 4 --k-heads 2 --chunks 512 2048

Inputs as linear_attn.mixer builds them (l2-normalised q scaled by Dk^-0.5 and k, uniform beta, KDA
gates lower_bound * sigmoid(.) with lower bound -5 as GLM-5.3-Flash / Kimi K3, GDN gates
-exp(A) softplus(.)), a random initial state, Dk = Dv = 128. Reported per C: max abs error of o and of
the final state relative to their max, against the token-by-token recurrence in float64 on the host, for the kernel and
for the torch path on the device; p50 of synchronous calls of a graph returning o.sum(0) and S.sum(0)
(the same reduction of an input of o's shape is timed alone as the floor); the compile + first call.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="kda", choices=["kda", "gdn"])
    ap.add_argument("--heads", type=int, default=2, help="v heads on this rank")
    ap.add_argument("--k-heads", type=int, default=None, help="k heads (GDN; default = --heads)")
    ap.add_argument("--chunks", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("--torch-max", type=int, default=2048, help="largest C the torch path is compiled for")
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--correlated", action="store_true", help="nearly parallel keys, beta near 1, slow decay")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--save-inputs", default=None, help="torch.save the kernel graph's inputs (per chunk: "
                    "<path>.C<chunk>.pt) for tools/prof_engines.py, and print the graph's cache hash")
    args = ap.parse_args()

    import profile_layer as pl

    pl.setup_device()
    from kiln.kernels import delta_rule as dr
    from kiln.models import linear_attn as la
    from tests.test_delta_rule import inputs, recurrence

    kda = args.kind == "kda"
    Hv, Hk = args.heads, args.k_heads or args.heads
    sub = la.CHUNKS[args.kind]
    print(f"{args.kind}: {Hk} k heads, {Hv} v heads, Dk = Dv = 128, torch path sub-chunks of {sub}, "
          f"kernel chunks of {dr.L}, rev {dr.REV:#010x}", flush=True)

    def kern(q, k, v, g, b, S):
        o, S2 = dr.chunk(q, k, v, g, b, S)
        return o.sum(0), S2.sum(0)

    def kern_full(q, k, v, g, b, S):
        return dr.chunk(q, k, v, g, b, S)

    def tor(q, k, v, g, b, S):
        rep = Hv // Hk
        if rep > 1:
            q, k = q.repeat_interleave(rep, 1), k.repeat_interleave(rep, 1)
        o, S2 = la.chunk_scan(q, k, v, g, b, S, sub)
        return o.sum(0), S2.sum(0)

    def tor_full(q, k, v, g, b, S):
        rep = Hv // Hk
        if rep > 1:
            q, k = q.repeat_interleave(rep, 1), k.repeat_interleave(rep, 1)
        return la.chunk_scan(q, k, v, g, b, S, sub)

    for C in args.chunks:
        host = inputs(C, Hk, Hv, kda, seed=C, correlated=args.correlated)
        rep = Hv // Hk
        q, k = host[0], host[1]
        qe, ke = (q.repeat_interleave(rep, 1), k.repeat_interleave(rep, 1)) if rep > 1 else (q, k)
        want, S_want = recurrence(qe, ke, *host[2:])
        dev = tuple(x.to(pl.DEV) for x in host)
        rel = lambda got, ref: ((got.double() - ref).abs().max() / ref.abs().max()).item()  # noqa: E731
        print(f"C={C}:", flush=True)
        pl.timed(f"floor: o-shaped sum (C={C})", lambda o: o.sum(0), (torch.zeros(C, Hv, 128).to(pl.DEV),),
                 args.iters)
        t_wall = __import__("time").time()
        t_k = pl.timed(f"kernel (C={C})", kern, dev, args.iters)
        if args.save_inputs:
            import neff_instructions as ni

            torch.save(dict(zip(("q", "k", "v", "g", "b", "S"), host)), f"{args.save_inputs}.C{C}.pt")
            for e in ni.latest(1, since=t_wall):
                print(f"  kernel graph {os.path.basename(e)}: {ni.line(e)}", flush=True)
        o, S = torch.compile(kern_full, **pl.OPTS)(*dev)
        print(f"  kernel vs float64: o {rel(o.cpu(), want):.2e}, S {rel(S.cpu(), S_want):.2e}", flush=True)
        if not args.no_torch and C <= args.torch_max:
            t_t = pl.timed(f"torch path (C={C})", tor, dev, args.iters)
            o, S = torch.compile(tor_full, **pl.OPTS)(*dev)
            print(f"  torch path vs float64: o {rel(o.cpu(), want):.2e}, S {rel(S.cpu(), S_want):.2e}; "
                  f"kernel {t_t / t_k:.1f}x faster", flush=True)


if __name__ == "__main__":
    main()
