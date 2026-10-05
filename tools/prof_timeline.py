"""Per-engine instruction timeline of one NEFF from a neuron-explorer profile (SDK 2.32):
where each engine spends its time, what it waits on, and a slice of the timeline.

    python tools/prof_timeline.py <compile-cache hash> [--window 0.50 0.52] [--capture]

Runs `neuron-explorer capture` (unless the .ntff exists) and `view --output-format json` into
/opt/kiln/prof/tl, then reads the JSON's `instruction` records (timestamp, duration,
evt_wait_time in ns, subgroup = engine queue, instruction_type). The capture runs the NEFF with
the runtime's own inputs, so data-dependent work (an expert index) is not the real one.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import subprocess

CACHE = "/root/.cache/neuron_libtorch/neuron/compile_cache"
OUT = "/opt/kiln/prof/tl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hash")
    ap.add_argument("--window", type=float, nargs=2, default=(0.50, 0.51), help="fraction of the run to list")
    ap.add_argument("--max-lines", type=int, default=150)
    ap.add_argument("--inputs", default=None, help="directory of input<i>.npy to run the NEFF on (else its own)")
    ap.add_argument("--tag", default="", help="suffix of the profile files (one per input set)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = os.path.join(CACHE, args.hash)
    neff = glob.glob(os.path.join(d, "*.neff"))[0]
    ntff = os.path.join(OUT, f"{args.hash}{args.tag}.ntff")
    env = dict(os.environ, HOME=os.environ.get("HOME", "/root"))
    if not os.path.exists(ntff):
        ifm = []
        if args.inputs:
            for f in sorted(glob.glob(os.path.join(args.inputs, "input*.npy")), key=lambda f: int(f.split("input")[-1][:-4])):
                ifm += [os.path.basename(f)[:-4], f]
        subprocess.run(["neuron-explorer", "capture", "-n", neff, "-s", ntff, *ifm], check=True, env=env,
                       capture_output=True)
    js = os.path.join(OUT, f"{args.hash}{args.tag}.json")
    if not os.path.exists(js):
        before = set(glob.glob(os.path.join(OUT, "*.json")))
        subprocess.run(["neuron-explorer", "view", "-n", neff, "-s", ntff, "--output-format", "json"], check=True,
                       env=env, cwd=OUT, capture_output=True)
        new = sorted(set(glob.glob(os.path.join(OUT, "*.json"))) - before, key=os.path.getmtime)
        os.rename(new[-1], js)
    data = json.load(open(js))
    ins = data["instruction"]
    ts = sorted(i["timestamp"] for i in ins)
    # The first instructions are the runtime's setup, a long way before the kernel proper.
    t0, t1 = ts[len(ts) // 100], ts[-1]
    ins = [i for i in ins if i["timestamp"] >= t0]
    print(f"{len(ins)} instructions in {(t1 - t0) / 1e3:.1f} us (from the 1st percentile on)", flush=True)
    by = collections.defaultdict(lambda: collections.Counter())
    busy = collections.Counter()
    wait = collections.Counter()
    n = collections.Counter()
    for i in ins:
        e = i.get("subgroup", "?")
        dur = i.get("duration", 0) or 0
        busy[e] += dur
        wait[e] += i.get("evt_wait_time", 0) or 0
        n[e] += 1
        by[e][i.get("opcode", "?")] += dur
    for e in sorted(busy, key=lambda k: -busy[k]):
        top = ", ".join(f"{k} {v / 1e3:.0f}us" for k, v in by[e].most_common(6))
        print(f"  {e:<12} {n[e]:6d} instr  busy {busy[e] / 1e3:8.1f} us  waited {wait[e] / 1e3:8.1f} us  | {top}")
    # Gaps in each engine's instruction stream (issue to issue) and what the instruction after
    # the gap waited on: the semaphore it names (S[k] (<engine>)>=n) in its operands.
    import re

    for e in ("Tensor", "Vector", "Scalar", "GpSimd"):
        seq = sorted((i for i in ins if i.get("subgroup") == e), key=lambda i: i["timestamp"])
        gaps = collections.Counter()
        cnt = collections.Counter()
        for a, b in zip(seq, seq[1:]):
            gap = b["timestamp"] - a["timestamp"] - (a.get("duration") or 0) if e != "Tensor" else \
                b["timestamp"] - a["timestamp"] - 40
            if gap > 100:
                m = re.findall(r"S\[\d+\] \((\w+)\)>=", str(b.get("operands", "")))
                key = "+".join(sorted(set(m))) or "none"
                gaps[key] += gap
                cnt[key] += 1
        tot = sum(gaps.values())
        print(f"  {e:<8} gaps > 100 ns: {tot / 1e3:8.1f} us  " + ", ".join(
            f"waiting {k}: {v / 1e3:.1f} us ({cnt[k]})" for k, v in gaps.most_common(5)))
    lo = t0 + (t1 - t0) * args.window[0]
    hi = t0 + (t1 - t0) * args.window[1]
    sl = sorted((i for i in ins if lo <= i["timestamp"] <= hi), key=lambda i: i["timestamp"])
    print(f"timeline {args.window[0]:.3f}..{args.window[1]:.3f} ({len(sl)} instructions):")
    for i in sl[: args.max_lines]:
        ops = str(i.get("operands", ""))[:70]
        print(f"  {(i['timestamp'] - t0) / 1e3:9.3f} us  +{(i.get('duration') or 0):6d} ns  wait {i.get('evt_wait_time', 0):6}  "
              f"{i.get('subgroup', '?'):<8} {i.get('opcode', '?'):<18} {ops}")


if __name__ == "__main__":
    main()
