"""The whole-call map of one replayed prefill call: every piece's tools/prof_segments.py split (rank 0) summed into
segment classes, with the collectives' time from tools/util_report.py's report beside them, as ms per call and %.

    python tools/pc_map.py --report repp-G64-fnu-text.report.txt split-G64-fnu-text-*.txt

A segment is the time between two collectives; prof_segments names its type by the collective that ends it and the
NKI kernel with the most engine time inside it. A type whose NKI engine time is under a fifth of its engine time is
counted as XLA-only whatever kernel it names (a few stray kernel instructions in a mostly-XLA segment). Classes: KDA
(delta_rule / gated_norm), FFN (moe_ep), DSA long (dsa_long_*, dsa_slots*, dsa_topk), short-context DSA attention
(dsa_fused, dsa_prefill), other kernels, XLA-only (hyper-connection mixing, norms, routing glue, embedding / head).
Per class: the span, and the part of it with no engine busy (waiting inside the segment). Collectives: the report's
cc ms (time with a collective in flight on rank 0, including its wait for the slowest rank). The rest of the call
(graph starts and ends, the small pieces without a split) is printed as such.
"""

from __future__ import annotations

import argparse
import collections
import re

CLASSES = [
    ("KDA mixer", ("delta_rule", "gated_norm"), "prefill-compute"),
    ("FFN (EP kernel, shared expert)", ("moe_ep",), "prefill-compute; EP skew D2D/EPLB"),
    ("DSA long (1M select, slots, top-k)", ("dsa_long", "dsa_slots", "dsa_topk", "dsa_index"), "lc2"),
    ("short-context DSA / MLA attention", ("dsa_fused", "dsa_prefill"), "dense-prefill"),
]
XLA = "XLA-only (mHC mixing, norms, routing glue)"


def classify(kernel: str, nki: float, xla: float) -> tuple[str, str]:
    if kernel == "no NKI" or nki < 0.2 * (nki + xla):
        return XLA, "unowned"
    for name, keys, owner in CLASSES:
        if any(k in kernel for k in keys):
            return name, owner
    return f"other kernel ({kernel})", "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("splits", nargs="+")
    a = ap.parse_args()
    call = cc = None
    for line in open(a.report):
        m = re.match(r"^sum\s+([\d.]+)\s+.*?\s([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if m:
            call, cc = float(m.group(1)), float(m.group(2))
    seg = collections.defaultdict(float)
    idle = collections.defaultdict(float)
    nseg = collections.Counter()
    owner = {}
    span_splits = 0.0
    seg_re = re.compile(r"^\s+(.+?) \[(.+?)\]\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+) / ([\d.]+)\s+\(total span ([\d.]+) ms\)")
    for path in a.splits:
        for line in open(path):
            m = re.match(r"^\d+ collectives, \d+ instructions, span ([\d.]+) ms", line)
            if m:
                span_splits += float(m.group(1))
                continue
            m = seg_re.match(line)
            if m:
                n, span, busy = int(m.group(3)), float(m.group(4)), float(m.group(5))
                name, own = classify(m.group(2), float(m.group(6)), float(m.group(7)))
                seg[name] += float(m.group(8))
                idle[name] += n * max(span - busy, 0.0)
                nseg[name] += n
                owner[name] = own
    print(f"call {call:.2f} ms (report); the split pieces span {span_splits:.2f} ms")
    print(f"{'class':44s} {'n':>5s} {'ms':>8s} {'%':>6s} {'idle in it':>10s}  owner")
    for name, ms in sorted(seg.items(), key=lambda kv: -kv[1]):
        print(f"{name:44s} {nseg[name]:5d} {ms:8.2f} {100 * ms / call:5.1f}% {idle[name]:9.2f}   {owner[name]}")
    print(f"{'collectives in flight (report cc ms)':44s} {'':5s} {cc:8.2f} {100 * cc / call:5.1f}% {'':10s}  D2D")
    rest = call - sum(seg.values()) - cc
    print(f"{'rest (graph ends, small pieces, gaps)':44s} {'':5s} {rest:8.2f} {100 * rest / call:5.1f}%")


if __name__ == "__main__":
    main()
