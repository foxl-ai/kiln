"""A window of one profile.json's timeline (tools/prof_engines.py writes it), every engine interleaved by start time:
runs of consecutive tensor-engine instructions folded into one line (first start, last end, count), every other
instruction with its duration and semaphore wait, and DMA queue activity summed per window slice.

    python tools/prof_window.py /opt/kiln/prof/eng/<key>/profile.json [--at 0.5] [--us 20]
"""

from __future__ import annotations

import argparse
import json
import re


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--at", type=float, default=0.5, help="window start as a fraction of the instruction list")
    ap.add_argument("--us", type=float, default=20.0)
    a = ap.parse_args()
    d = json.load(open(a.profile))
    ins = sorted(d["instruction"], key=lambda i: i["timestamp"])
    t0 = ins[int(len(ins) * a.at)]["timestamp"]
    w = [i for i in ins if t0 <= i["timestamp"] < t0 + a.us * 1e3]
    run = None

    def flush():
        if run:
            print(f"{(run[0] - t0) / 1e3:8.3f}-{(run[1] - t0) / 1e3:7.3f} us  Tensor  x{run[2]} (waits {run[3]} ns)")

    for i in w:
        e, dur = i.get("subgroup", "?"), i.get("duration") or 0
        if e == "Tensor":
            if run:
                run[1], run[2], run[3] = i["timestamp"] + dur, run[2] + 1, run[3] + (i.get("evt_wait_time") or 0)
            else:
                run = [i["timestamp"], i["timestamp"] + dur, 1, i.get("evt_wait_time") or 0]
            continue
        flush()
        run = None
        o = re.sub(r"0x[0-9a-f]+", "", str(i.get("operands", "")))[:80]
        print(f"{(i['timestamp'] - t0) / 1e3:8.3f} +{dur:5d} ns wait {i.get('evt_wait_time') or 0:6}  {e:<7} "
              f"{i.get('opcode', '?'):<16} {o}")
    flush()
    dm = [x for x in d.get("dma", []) if t0 <= x.get("timestamp", 0) < t0 + a.us * 1e3] if "dma" in d else []
    if dm:
        print(f"dma records in window: {len(dm)}, bytes {sum(x.get('size', 0) for x in dm)}")
    print("keys:", list(d.keys()))


if __name__ == "__main__":
    main()
