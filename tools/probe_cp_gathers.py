"""The context-parallel DSA layer's XLA gathers whose sizes follow the page bucket (models/mla.py attention_cp), each in its
own graph on one NeuronCore, checked against the host. Measured on trn2 the R8 long-context engine answers wrong in its
4096-page bucket and right in its 1024-page one (docs/neuron-notes.md "Long prompts on trn2"); these are the only XLA
indirect ops of its DSA piece whose shapes differ between the two buckets (the piece HLO's gathers, by kind):

  table   tb[pos // ps] and tb[pos4 // ps]: an int64 block table [P] read at C and 4 C positions
  pk      the pool-key rows of the bucket: cache.view(-1, 128)[table x ppl + j], [P ppl, 128] bf16 from [rows, 128]
  pool    pool_row: table.view(1, -1).expand(C, -1).gather(1, pg) for pg [C, keep] (the selection's pool pages) and for
          the tail page [C, 1], an element gather from a broadcast table

    NEURON_RT_VISIBLE_CORES=0 python tools/probe_cp_gathers.py [--pages 1024 4096] [--rows 1024] [--keep 512]

Index values like the serving run's: a table of `--used` real pages (1 .. used) then the null page 0, selection pages
uniform below the used pages. Prints per op and bucket: ran, equal to the host, rows that differ.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--rows", type=int, default=1024, help="C: query rows of the chunk on this rank")
    ap.add_argument("--keep", type=int, default=512)
    ap.add_argument("--used", type=int, default=0, help="real pages in the table (0: half the bucket)")
    ap.add_argument("--cache-pages", type=int, default=5500)
    ap.add_argument("--ps", type=int, default=256)
    ap.add_argument("--ppl", type=int, default=8, help="local pools per page per rank (ps / (kp A))")
    a = ap.parse_args()
    import profile_layer as pl

    pl.setup_device()
    g = torch.Generator().manual_seed(0)
    C, K = a.rows, a.keep
    cache = torch.randn(a.cache_pages * a.ppl, 128, generator=g).bfloat16()

    def run(name, f, *args):
        want = f(*args)
        try:
            got = torch.compile(f, **pl.OPTS)(*[x.to(pl.DEV) for x in args]).cpu()
        except Exception as e:
            print(f"RESULT {name}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
            return
        ok = torch.equal(got, want)
        bad = 0 if ok else int((got != want).reshape(got.shape[0], -1).any(1).sum())
        print(f"RESULT {name}: ran, equal to the host: {ok}{'' if ok else f' ({bad} of {got.shape[0]} rows differ)'}",
              flush=True)

    for P in a.pages:
        used = a.used or P // 2
        table = torch.zeros(P, dtype=torch.int64)
        table[:used] = torch.arange(1, used + 1)
        pos = torch.randint(0, used * a.ps, (C,), generator=g)
        pos4 = torch.randint(0, used * a.ps, (4 * C,), generator=g)
        run(f"table P={P} C={C}", lambda t, p: t[torch.div(p, a.ps, rounding_mode="floor")], table, pos)
        run(f"table P={P} 4C={4 * C}", lambda t, p: t[torch.div(p, a.ps, rounding_mode="floor")], table, pos4)
        off = torch.arange(a.ppl, dtype=torch.int64)

        def pk(c, t, o):
            return c.view(-1, 128)[(t.unsqueeze(-1) * a.ppl + o).flatten()]

        run(f"pk P={P} rows={P * a.ppl}", pk, cache, table, off)
        pg = torch.randint(0, used, (C, K), generator=g)
        tail = torch.randint(0, used, (C, 1), generator=g)

        def pool(t, q):
            tb = t.view(1, -1).expand(C, -1)
            return tb.gather(1, q.clamp(max=tb.shape[1] - 1))

        run(f"pool_row P={P} [{C},{K}]", pool, table, pg)
        run(f"pool_row tail P={P} [{C},1]", pool, table, tail)


if __name__ == "__main__":
    main()
