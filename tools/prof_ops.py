"""Busy time of one engine's instructions grouped by opcode, ALU op and operand shapes, from a profile.json that
tools/prof_engines.py wrote (neuron-explorer view --output-format json).

    python tools/prof_ops.py /opt/kiln/prof/eng/<key>/profile.json [--engine Vector] [--top 20]
"""

from __future__ import annotations

import argparse
import collections
import json
import re


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--engine", default="Vector")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    ins = json.load(open(a.profile))["instruction"]
    busy, cnt = collections.Counter(), collections.Counter()
    tot = 0
    for i in ins:
        if i.get("subgroup") != a.engine:
            continue
        o = re.sub(r"S\[\d+\] \(\w+\)\+\+@complete ", "", str(i.get("operands", "")))
        o = re.sub(r"0x[0-9a-f]+", "", o).replace("@", "")
        shapes = " ".join(f"{k}={t}{s}" for k, t, _, s in re.findall(r"(src0|src1|src|dst)=(\w+)(\[[^\]]*\])(\[[^\]]*\])", o))
        m = re.search(r"op=(\w+)", o)
        key = f"{i.get('opcode')} {m.group(1) if m else ''} {shapes}"
        d = i.get("duration", 0) or 0
        busy[key] += d
        cnt[key] += 1
        tot += d
    print(f"{a.engine}: {sum(cnt.values())} instructions, busy {tot / 1e3:.1f} us")
    for k, v in busy.most_common(a.top):
        print(f"  {v / 1e3:8.1f} us {cnt[k]:6d}x {v / cnt[k]:7.0f} ns  {k[:160]}")


if __name__ == "__main__":
    main()
