"""A KILN_EPLB_INIT file (models/eplb.py load_file: {"load": {layer: fp64 [E]}}) from tools/ep_routing.py's saved
routing: each MoE layer's pairs per expert summed over the named sequences.

    python tools/eplb_init_from_routing.py ep_routing.pt eplb-init-random.pt [--sequences random0 random1]

The file the EPLB measurements used (s3://<your-bucket>/logs/kiln-tq-cpu/eplb-init-random.pt, 2026-10-05) is
this over s3://<your-bucket>/logs/kiln-mimo-trn1/ep_routing.pt with random0 and random1: GLM-5.3-Flash's real
routers on the real hidden states of two 4096-token sequences drawn as bench/serve_sweep.py draws its prompts
(randrange(1000, 100000), seeds 0 and 1). A serving engine records the same counts itself (ep_stats) and can dump them
(bench/serve_sweep.py --eplb-rebalance --eplb-dump), which is the file to use for other traffic.
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("routing")
    ap.add_argument("out")
    ap.add_argument("--sequences", nargs="+", default=["random0", "random1"])
    a = ap.parse_args()
    d = torch.load(a.routing)
    E, names = d["experts"], d["names"]
    sel = [names.index(n) for n in a.sequences]
    load = {int(l): torch.bincount(torch.cat([seqs[i].long().flatten() for i in sel]), minlength=E).double()
            for l, seqs in d["topi"].items()}
    torch.save({"load": load, "source": f"tools/eplb_init_from_routing.py {a.routing} --sequences {' '.join(a.sequences)}"},
               a.out)
    print(f"{a.out}: {len(load)} MoE layers from {[names[i] for i in sel]}")


if __name__ == "__main__":
    main()
