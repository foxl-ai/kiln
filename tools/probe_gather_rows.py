"""Does an XLA row gather (x[idx]: one indirect DMA row per index, vector DGE) of many rows fault on trn2? The context-
parallel DSA path gathers every local pool key of a context this way (models/mla.py attention_cp: pool_key.view(-1, Di)
[prow], 32,768 rows of a 4096-page bucket at CP 8), and the long-prompt engine's 4096-page prefill graphs fault with
"scatter/gather (indirect memory copy via vector DGE) out-of-bound access" while the 1024-page ones (8,192 rows) run.

    NEURON_RT_VISIBLE_CORES=32 python tools/probe_gather_rows.py [--rows-table 200000] [--width 128] [--n 8192 16384 32768 65536]

Each N: one graph y = x[idx] (idx uniform in range, int64), checked against the host; prints ok / FAILED per N.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-table", type=int, default=200000)
    ap.add_argument("--width", type=int, default=128, help="elements per row")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--n", type=int, nargs="+", default=[4096, 8192, 16384, 32768, 65536])
    a = ap.parse_args()
    import profile_layer as pl

    pl.setup_device()
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    g = torch.Generator().manual_seed(0)
    x = torch.randn(a.rows_table, a.width, generator=g).to(dt)
    xd = x.to(pl.DEV)
    for n in a.n:
        idx = torch.randint(0, a.rows_table, (n,), generator=g)

        def f(t, i):
            return t[i]

        try:
            got = torch.compile(f, **pl.OPTS)(xd, idx.to(pl.DEV)).cpu()
            ok = torch.equal(got, x[idx])
            print(f"RESULT n={n}: ran, equal to the host: {ok}", flush=True)
        except Exception as e:
            print(f"RESULT n={n}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    main()
