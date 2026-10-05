"""Whether the experts of a real checkpoint re-base exactly on one power-of-two scale per tile, as
kiln/kernels/moe_dedupe.pack() needs: per 128-column tile (gate / up) or per output column over
the rank's 64 input rows (down), how far the MXFP4 block exponents spread, and how many tiles have
no exact tile exponent (moe_dedupe._window: pack() would refuse the checkpoint).

    python tools/check_expert_scales.py [--model XiaomiMiMo/MiMo-V2.6-Flash-RL] [--shards 1]
        [--tp 32] [--delete] [--worker I N]   # --shards 0: all; --delete: remove each shard after
                                              # reading it; --worker: every N-th shard from the I-th

Downloads the index and the shards that hold expert weights, converts each projection exactly as
the loader does (models/quant.mxfp4_unpack: E2M1 codes as FP8, E8M0 exponents), and runs
moe_dedupe's own window on it. Also counts the codes that would round with the tile exponent at
the block maximum, which is what the window improves on by shifting a tile's largest block up.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="XiaomiMiMo/MiMo-V2.6-Flash-RL")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--tp", type=int, default=32)
    ap.add_argument("--delete", action="store_true", help="delete each downloaded shard after reading it")
    ap.add_argument("--worker", type=int, nargs=2, default=(0, 1), metavar=("I", "N"),
                    help="read only every N-th shard from the I-th (to run N of these side by side)")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    from kiln.kernels.moe_dedupe import _window
    from kiln.models.quant import mxfp4_unpack

    idx = json.load(open(hf_hub_download(args.model, "model.safetensors.index.json")))
    files = collections.OrderedDict()
    for name, f in idx["weight_map"].items():
        if re.search(r"mlp\.experts\.\d+\.(gate|up|down)_proj\.weight_scale$", name):
            files.setdefault(f, []).append(name)
    todo = list(files)[: args.shards] if args.shards else list(files)
    todo = todo[args.worker[0]::args.worker[1]]
    print(f"{len(files)} shards hold expert scales; reading {len(todo)}", flush=True)
    hist = collections.Counter()
    tiles = collections.Counter()
    refused = collections.Counter()
    atmax = collections.Counter()
    for f in todo:
        path = hf_hub_download(args.model, f)
        with safe_open(path, "pt") as fh:
            for sname in files[f]:
                kind = re.search(r"(gate|up|down)_proj", sname).group(1)
                w, s = mxfp4_unpack(fh.get_tensor(sname.replace("weight_scale", "weight")), fh.get_tensor(sname))
                k = torch.round(torch.log2(s.double())).to(torch.int64)  # [rows, cols / 32]
                R, nb = k.shape
                group = 4 if kind != "down" else max(1, nb // args.tp)  # blocks per tile
                wt, kt = w.view(R, nb // group, group, 32), k.view(R, nb // group, group)
                lo, hi, K = _window(wt, kt)
                refused[kind] += int((lo > hi).sum())
                tiles[kind] += lo.numel()
                kmax = kt.amax(-1)
                atmax[kind] += int(((kmax > hi) | (kmax < lo)).sum())  # K = max would not be exact
                d = kmax.unsqueeze(-1) - kt
                for v, n in zip(*torch.unique(d, return_counts=True)):
                    hist[(kind, int(v))] += int(n)
        print(f"{f}: done (tiles refused so far: {sum(refused.values())}, inexact at the block maximum: "
              f"{sum(atmax.values())})", flush=True)
        if args.delete:
            os.remove(os.path.realpath(path))
    for kind in ("gate", "up", "down"):
        hs = sorted((v, n) for (kk, v), n in hist.items() if kk == kind)
        print(f"{kind}: tiles {tiles[kind]}, refused {refused[kind]}, inexact at the block maximum {atmax[kind]}; "
              "blocks below their tile's largest exponent by: " + ", ".join(f"{v}: {n}" for v, n in hs), flush=True)


if __name__ == "__main__":
    main()
