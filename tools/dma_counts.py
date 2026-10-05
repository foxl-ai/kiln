"""DMA packet counts of compiled graphs, read from a hardware profile (neuron-explorer, SDK 2.32).

    python tools/dma_counts.py --since 30          # every graph compiled in the last 30 minutes
    python tools/dma_counts.py <cache hash> ...

For each LNL compile-cache entry (/root/.cache/neuron_libtorch/neuron/compile_cache/<hash>) it
runs `neuron-explorer capture -n graph_<hash>.neff -s <ntff>` (the runtime's own inputs: the
NEFF runs alone on NeuronCore 0, so only single-rank graphs, without collectives, can be
captured) and `neuron-explorer view --output-format summary-text`, and prints the fields that
showed the MoE gather problem (docs/neuron-notes.md): total time, software dynamic DMA packets
(trn1 has no hardware DGE) and their bytes, static packets, spill reload bytes. The graph is
labelled by what its FX graph holds: NKI kernel calls and all-reduces. Raw outputs go to
/opt/kiln/prof (cap_<hash>.log, sum_<hash>.txt).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import time

CACHE = "/root/.cache/neuron_libtorch/neuron/compile_cache"
OUT = "/opt/kiln/prof"
FIELDS = ("total_time", "software_dynamic_dma_packet_count", "software_dynamic_dma_size",
          "static_dma_packet_count", "dma_transfer_average_bytes", "spill_reload_bytes")


def label(d: str) -> str:
    try:
        with open(os.path.join(d, "fxgraph.txt")) as f:
            fx = f.read()
    except OSError:
        return "?"
    return (f"{len(re.findall(r'nki_kernel_wrapper', fx)) // 2} nki calls, "
            f"{len(re.findall(r'all_reduce', fx)) // 2} all-reduces, {len(fx.splitlines())} fx lines")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hashes", nargs="*")
    ap.add_argument("--since", type=float, default=0, help="graphs compiled in the last N minutes")
    args = ap.parse_args()
    dirs = [os.path.join(CACHE, h) for h in args.hashes]
    if args.since:
        cut = time.time() - args.since * 60
        dirs += sorted((d for d in glob.glob(os.path.join(CACHE, "*")) if os.path.getmtime(d) > cut),
                       key=os.path.getmtime)
    os.makedirs(OUT, exist_ok=True)
    for d in dirs:
        h = os.path.basename(d)
        neffs = glob.glob(os.path.join(d, "*.neff"))
        if not neffs:
            continue
        ntff = os.path.join(OUT, f"{h}.ntff")
        cap = subprocess.run(["neuron-explorer", "capture", "-n", neffs[0], "-s", ntff],
                             capture_output=True, text=True)
        with open(os.path.join(OUT, f"cap_{h}.log"), "w") as f:
            f.write(cap.stdout + cap.stderr)
        if cap.returncode:
            print(f"{h}: capture failed ({label(d)}): {(cap.stdout + cap.stderr).strip().splitlines()[-1:]}", flush=True)
            continue
        view = subprocess.run(["neuron-explorer", "view", "-n", neffs[0], "-s", ntff, "--output-format",
                               "summary-text"], capture_output=True, text=True)
        with open(os.path.join(OUT, f"sum_{h}.txt"), "w") as f:
            f.write(view.stdout)
        vals = dict(re.findall(r"^\s*(\w+)\s+(\S+)\s*$", view.stdout, re.M))
        print(f"{h} ({label(d)}): " + ", ".join(f"{k}={vals.get(k, '?')}" for k in FIELDS), flush=True)


if __name__ == "__main__":
    main()
