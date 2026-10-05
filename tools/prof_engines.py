"""Per-engine time of ONE compiled graph from a hardware profile (neuron-explorer, SDK 2.32), run on
the inputs it was traced with: where each engine is busy, what it waits for, and DMA totals.

    python tools/prof_engines.py <compile-cache hash> <inputs.pt> [--window 0.5 0.51]

<inputs.pt> is a dict of the graph function's arguments by name (torch.save, CPU tensors), as
tools/probe_moe_prefill.py --save-inputs writes it; the graph's placeholders (fxgraph.txt,
L_<name>_) give the NEFF's input order, and each tensor is written as input<i>.npy of its raw
bytes. `neuron-explorer capture -n <neff> -s <ntff> input<i> <file> ...` runs the NEFF alone on
NeuronCore 0 with those inputs (data-dependent work such as which expert a DMA loads is then the
real one), and `view --output-format json` gives the instruction records (timestamp, duration,
evt_wait_time in ns, subgroup = engine queue, opcode). Adapted from the timeline reader on
feat/moe-kernel-v2 (tools/prof_timeline.py). Outputs go to /opt/kiln/prof/eng.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess

import numpy as np
import torch

CACHE = "/root/.cache/neuron_libtorch/neuron/compile_cache"
OUT = "/opt/kiln/prof/eng"


def raw(t: torch.Tensor) -> np.ndarray:
    """The tensor's bytes as a numpy array of the same element size."""
    t = t.contiguous()
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.view(view).numpy() if t.dtype.is_floating_point or t.dtype == torch.bool else t.numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hash")
    ap.add_argument("inputs")
    ap.add_argument("--window", type=float, nargs=2, default=None, help="fraction of the run to list")
    ap.add_argument("--max-lines", type=int, default=120)
    ap.add_argument("--recapture", action="store_true", help="capture again even if a profile exists")
    args = ap.parse_args()
    d = os.path.join(CACHE, args.hash)
    out = os.path.join(OUT, args.hash)
    os.makedirs(out, exist_ok=True)
    neff = glob.glob(os.path.join(d, "*.neff"))[0]
    names = re.findall(r"placeholder\[target=L_(\w+?)_\]", open(os.path.join(d, "fxgraph.txt")).read())
    vals = torch.load(args.inputs)
    ifm = []
    for i, n in enumerate(names):
        f = os.path.join(out, f"input{i}.npy")
        np.save(f, raw(vals[n]))
        ifm += [f"input{i}", f]
    ntff = os.path.join(out, "profile.ntff")
    js = os.path.join(out, "profile.json")
    env = dict(os.environ, HOME=os.environ.get("HOME", "/root"))
    if args.recapture or not os.path.exists(js):
        cap = subprocess.run(["neuron-explorer", "capture", "-n", neff, "-s", ntff, *ifm], env=env,
                             capture_output=True, text=True)
        if cap.returncode:
            print(cap.stdout[-2000:], cap.stderr[-2000:])
            raise SystemExit("capture failed")
        subprocess.run(["neuron-explorer", "view", "-n", neff, "-s", ntff, "--output-format", "json"], check=True,
                       env=env, cwd=out, capture_output=True)
        new = sorted((f for f in glob.glob(os.path.join(out, "*.json")) if f != js), key=os.path.getmtime)
        os.replace(new[-1], js)
    data = json.load(open(js))
    ins = data["instruction"]
    ts = sorted(i["timestamp"] for i in ins)
    t0, t1 = ts[len(ts) // 200], ts[-1]
    ins = [i for i in ins if i["timestamp"] >= t0]
    print(f"{args.hash}: inputs {names}", flush=True)
    print(f"{len(ins)} instructions in {(t1 - t0) / 1e3:.1f} us (from the 0.5th percentile on)", flush=True)
    busy, wait, n = collections.Counter(), collections.Counter(), collections.Counter()
    by = collections.defaultdict(collections.Counter)
    for i in ins:
        e = i.get("subgroup", "?")
        dur = i.get("duration", 0) or 0
        busy[e] += dur
        wait[e] += i.get("evt_wait_time", 0) or 0
        n[e] += 1
        by[e][i.get("opcode", "?")] += dur
    for e in sorted(busy, key=lambda k: -busy[k]):
        top = ", ".join(f"{k} {v / 1e3:.0f}us" for k, v in by[e].most_common(6))
        print(f"  {e:<14} {n[e]:7d} instr  busy {busy[e] / 1e3:9.1f} us  waited {wait[e] / 1e3:9.1f} us  | {top}")
    for k in ("summary",):
        if k in data:
            s = data[k][0] if isinstance(data[k], list) else data[k]
            keys = [x for x in s if re.search(r"dma|time|util|active", x)]
            print("  summary: " + ", ".join(f"{x}={s[x]}" for x in keys[:40]))
    if args.window:
        lo, hi = (t0 + (t1 - t0) * w for w in args.window)
        sl = sorted((i for i in ins if lo <= i["timestamp"] <= hi), key=lambda i: i["timestamp"])
        print(f"timeline {args.window[0]:.3f}..{args.window[1]:.3f} ({len(sl)} instructions):")
        for i in sl[: args.max_lines]:
            print(f"  {(i['timestamp'] - t0) / 1e3:9.3f} us  +{(i.get('duration') or 0):6d} ns  wait "
                  f"{i.get('evt_wait_time', 0):6}  {i.get('subgroup', '?'):<10} {i.get('opcode', '?'):<18} "
                  f"{str(i.get('operands', ''))[:70]}")


if __name__ == "__main__":
    main()
