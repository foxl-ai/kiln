"""kernels/delta_rule_v2.py against kernels/delta_rule.py on one NeuronCore: time, and bit-identity of o and the final
state, per rank shape and variant. ONE variant per process, chosen by the environment as the serving path reads it
(KILN_DELTA_RULE_UNITS, and KILN_DELTA_RULE_V=2 with KILN_DELTA_RULE_CP / KILN_DELTA_RULE_YACC for v2), so
every variant is its own graph (one process reusing a compiled function across static-argument values is how a first
version of this probe compared a graph with itself).

    KILN_DELTA_RULE_UNITS=2 python tools/probe_delta_rule_v2.py run --tag v1u2 [--shapes 2x4096 8x1024]
    KILN_DELTA_RULE_V=2 KILN_DELTA_RULE_UNITS=6 python tools/probe_delta_rule_v2.py run --tag v2u6
    python tools/probe_delta_rule_v2.py compare v1u2 v2u6 ...      # each against the first, torch.equal

--shapes HvxT (v heads x rows: 2x4096 the 1M R8 rank, 8x1024 the 8K G64 rank; KDA, Dk = Dv = 128); inputs as
tests/test_delta_rule.py builds them (correlated). run prints p50 and saves o and S to <dir>/<tag>-<shape>.pt.
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
    ap.add_argument("mode", choices=["run", "compare"])
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--shapes", nargs="+", default=["2x4096", "8x1024"])
    ap.add_argument("--dir", default="/opt/kiln/work/dr2")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.dir, exist_ok=True)
    if args.mode == "compare":
        base = args.tags[0]
        for shape in args.shapes:
            o1, S1 = torch.load(os.path.join(args.dir, f"{base}-{shape}.pt"))
            for t in args.tags[1:]:
                o2, S2 = torch.load(os.path.join(args.dir, f"{t}-{shape}.pt"))
                print(f"{shape}: {t} vs {base}: o equal {torch.equal(o1, o2)}, S equal {torch.equal(S1, S2)}, "
                      f"max |do| {(o1 - o2).abs().max().item():.3e}, max |dS| {(S1 - S2).abs().max().item():.3e}",
                      flush=True)
        return

    import profile_layer as pl

    pl.setup_device()
    from kiln.kernels import delta_rule as dr
    from kiln.kernels import delta_rule_v2 as d2
    from tests.test_delta_rule import inputs

    print(f"{args.tag}: units {dr.UNIT_GROUP}, v2 {d2.ENABLED} (cp {d2.CP}, yacc {d2.YACC}), revs {dr.REV:#010x} / "
          f"{d2.REV:#010x}", flush=True)
    for shape in args.shapes:
        Hv, T = (int(x) for x in shape.split("x"))
        dev = tuple(t.to(pl.DEV) for t in inputs(T, Hv, Hv, True, seed=7, correlated=True))

        def full(q, k, v, g, b, S):  # delta_rule.chunk's call, the kernel by the env (T is a multiple of 128 here)
            from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

            if d2.ENABLED:
                return wrap_nki(d2.kernel())[1](**d2.kernel_inputs(q, k, v, g, b, S))
            return wrap_nki(dr.kernel())[1](**dr.kernel_inputs(q, k, v, g, b, S))

        def kern(q, k, v, g, b, S):
            o, S2 = full(q, k, v, g, b, S)
            return o.sum(0), S2.sum(0)

        pl.timed(f"{args.tag} {shape}", kern, dev, args.iters)
        o, S = torch.compile(full, **pl.OPTS)(*dev)
        torch.save((o.cpu(), S.cpu()), os.path.join(args.dir, f"{args.tag}-{shape}.pt"))


if __name__ == "__main__":
    main()
