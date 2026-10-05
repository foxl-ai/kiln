"""Busy fraction of each engine per time bin of a profile tools/prof_engines.py captured (its profile.json), so the
phases of a kernel (plan, passes, combine) show as rows of a coarse timeline.

    python tools/prof_bins.py <compile-cache hash> [--bin 100] [--ops]

--bin: microseconds per row. --ops: also the busiest opcode of each engine in the bin.
"""

from __future__ import annotations

import argparse
import collections
import json
import os

OUT = "/opt/kiln/prof/eng"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("hash")
    ap.add_argument("--bin", type=float, default=100.0)
    ap.add_argument("--ops", action="store_true")
    a = ap.parse_args()
    data = json.load(open(os.path.join(OUT, a.hash, "profile.json")))
    ins = data["instruction"]
    ts = sorted(i["timestamp"] for i in ins)
    t0 = ts[len(ts) // 200]
    B = a.bin * 1e3
    busy = collections.defaultdict(collections.Counter)
    ops = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    engines = set()
    for i in ins:
        if i["timestamp"] < t0:
            continue
        e = i.get("subgroup", "?")
        op = i.get("opcode", "?")
        if op.startswith("EVENT_SEMAPHORE"):
            continue  # waits, not work
        engines.add(e)
        s, d = i["timestamp"] - t0, i.get("duration", 0) or 0
        while d > 0:  # split an instruction over the bins it spans
            b = int(s // B)
            take = min(d, (b + 1) * B - s)
            busy[b][e] += take
            ops[b][e][op] += take
            s += take
            d -= take
    eng = sorted(engines)
    print("     us  " + "  ".join(f"{e[:8]:>8}" for e in eng))
    for b in sorted(busy):
        row = "  ".join(f"{min(busy[b][e] / B, 9.99):8.2f}" for e in eng)
        extra = ""
        if a.ops:
            extra = "  " + ", ".join(f"{e[:3]}:{ops[b][e].most_common(1)[0][0]}" for e in eng if ops[b][e])
        print(f"{b * a.bin:7.0f}  {row}{extra}")


if __name__ == "__main__":
    main()
