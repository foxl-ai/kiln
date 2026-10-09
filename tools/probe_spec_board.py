"""engine/spec_async.py's board graphs on the device against the CPU, on hand-built boards (one slot whose last verify
accepted its first draft, then one that rejected it), per k and per row count R (one row: no DP attention, decode
bucket 1).

    NEURON_RT_VISIBLE_CORES=0 python tools/probe_spec_board.py [--k 1 2 3] [--rows 1 2]

Prints EQUAL or the outputs that differ. neuronx-cc 2.27 misread the board in one-row reader graphs at k 1 (spec_prep's
draft column and mtp_prep's acc and base came back 0; see the note above spec_prep), which spec_prep / mtp_prep now
avoid by running a one-row call on the row twice.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2])
    a = ap.parse_args()
    from kiln import platform

    platform.configure_runtime_env()
    import libtorch_neuronx_lite  # noqa: F401

    from kiln.engine import spec_async as sa
    from kiln.engine.model_runner import canonical_neuron_backend, neuronx_cc_args

    dev = torch.device("neuron:0")

    def comp(f):
        return torch.compile(f, backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                             options={"compiler_args": neuronx_cc_args(torch.float32)})

    bad = 0
    for k in a.k:
        Q, ps, S = 1 + k, 32, 4
        for R in a.rows:
            for accept_first in (1.0, 0.0):
                board = torch.zeros(S, sa.width(Q, k))
                board[1, 0] = 500.0  # newest token 500 at position 70, acc 0, state row 0, drafts 600 ..
                board[1, Q:Q + 4] = torch.tensor([0.0, 70.0, 0.0, 1.0])
                board[1, Q + 4:] = torch.arange(600, 600 + k, dtype=torch.float32)
                slot_idx = torch.tensor([1] + [S - 1] * (R - 1))  # padding rows: the last (scratch) slot
                valid = torch.tensor([1.0] + [0.0] * (R - 1))
                table = torch.tensor([[3, 4, 5, 6]] * R)
                rows = torch.zeros(R, Q, dtype=torch.int64)
                pad = torch.full((R, Q), -7, dtype=torch.int64)
                out = torch.zeros(R * Q, 46)
                out[:, 0] = torch.arange(700, 700 + R * Q, dtype=torch.float32)  # y
                out[0, 1] = accept_first  # the first draft's accept flag
                out[:, 2] = torch.arange(800, 800 + R * Q, dtype=torch.float32)  # replacements
                res = {}
                for where in ("cpu", "neuron"):
                    mv = (lambda t: t.to(dev)) if where == "neuron" else (lambda t: t)
                    f = comp if where == "neuron" else (lambda g: g)
                    bd = mv(board.clone())
                    prep = f(lambda bb, si, t, r, p, v: sa.spec_prep(bb, si, t, r, p, v, Q, k, ps, 1))
                    ids, draft, pos, slot, st = prep(bd, mv(slot_idx), mv(table), mv(rows), mv(pad), mv(valid))
                    post = f(lambda bb, si, o, i, ra, v: sa.spec_post(bb, si, o, i, ra, v, Q, k))
                    r = post(bd, mv(slot_idx), mv(out), ids, mv(rows.float()), mv(valid))
                    mprep = f(lambda bb, si, t, p, v: sa.mtp_prep(bb, si, t, p, v, Q, ps, 1, None, k))
                    mo = mprep(bd, mv(slot_idx), mv(table), mv(pad), mv(valid))
                    mpost = f(lambda bb, si, dd, v: sa.mtp_post(bb, si, dd, v, Q, k))
                    mpost(bd, mv(slot_idx), mv(torch.arange(900, 900 + R * k, dtype=torch.float32).view(R, k)),
                          mv(valid))
                    res[where] = {"spec_prep": [x.cpu().tolist() for x in (ids, draft, pos, slot, st)],
                                  "spec_post": r.cpu().tolist(), "mtp_prep": [x.cpu().tolist() for x in mo],
                                  "board": bd.cpu()[1].tolist()}
                same = res["cpu"] == res["neuron"]
                bad += not same
                print(f"k={k} R={R} accept_first={accept_first:.0f}: {'EQUAL' if same else 'DIFFER'}", flush=True)
                for key in res["cpu"]:
                    if res["cpu"][key] != res["neuron"][key]:
                        print(f"   {key}: cpu {res['cpu'][key]}\n   {' ' * len(key)}  neuron {res['neuron'][key]}",
                              flush=True)
    print(f"RESULT {'all equal' if not bad else f'{bad} differ'}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
