"""Engine profile of one moe_dedupe graph (GLM-5.3-Flash's tp=32 rank: 288 FP8 experts, 128 x 128 block scales,
clamped SiLU) at T tokens: compiles f(x, topv, topi, b) = moe_dedupe(...), saves its inputs by placeholder name, and
prints the compile-cache key for tools/prof_engines.py.

    KILN_MOE_DEDUPE_MAX_TOKENS=256 python tools/probe_dedupe_prof.py --rows 192 --out /opt/kiln/prof/dd192.pt
    python tools/prof_engines.py <key> /opt/kiln/prof/dd192.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class _Keys(logging.Handler):
    def __init__(self):
        super().__init__()
        self.keys = []

    def emit(self, r):
        m = re.search(r"Compilation cache key: (\w+)", r.getMessage())
        if m:
            self.keys.append(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=192)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from kiln import platform

    platform.configure_runtime_env()
    import libtorch_neuronx_lite  # noqa: F401

    from kiln.engine.model_runner import canonical_neuron_backend, neuronx_cc_args
    from kiln.kernels import moe_dedupe as mdd
    from probe_moe_kernel import experts128

    h = _Keys()
    logging.getLogger().addHandler(h)
    logging.getLogger().setLevel(logging.INFO)
    E, H, K, T = 288, 4096, 8, a.rows
    dev = torch.device("neuron:0")
    blob = mdd.pack(*experts128(E, H))
    g = torch.Generator().manual_seed(1)
    x = torch.randn(T, H, generator=g).bfloat16()
    topi = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])
    topv = (torch.rand(T, K, generator=g) + 0.1).bfloat16()

    def f(x, topv, topi, b):
        return mdd.moe_dedupe(x, topv, topi, b, act=mdd.ACTS["silu_clamp"], limit=10.0)

    fc = torch.compile(f, backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                       options={"compiler_args": neuronx_cc_args(torch.bfloat16, True)})
    o = fc(x.to(dev), topv.to(dev), topi.to(dev), blob.to(dev)).cpu().float()
    want = mdd.emulate(x, topv, topi, blob, 1, 10.0).float()
    print(f"T={T}: rel err vs emulate {((o - want).abs().max() / want.abs().max()).item():.4f}", flush=True)
    torch.save(dict(x=x, topv=topv, topi=topi, b=blob), a.out)
    print(f"keys {sorted(set(h.keys))}", flush=True)


if __name__ == "__main__":
    main()
