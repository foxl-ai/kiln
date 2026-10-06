"""Where one graph's device time goes by HLO op, from a neuron-explorer profile JSON (SDK 2.32): each engine's busy
time (the union of its instruction intervals), the HLO ops with most instruction time per engine (an NKI kernel's
instructions carry no HLO name and show as "(kernel)"), the DMA bytes by queue kind, and the profile summary. For a
graph whose collectives lack prof_step.py's trigger fields (a 2-rank replay).

    neuron-explorer capture -n g.neff -s g.ntff -r 2 -i 0 --ignore-exec-errors
    neuron-explorer view -n g.neff -s g_rank_0.ntff --output-format json      # HOME must be set
    python tools/prof_hlo.py ntff.json [--top 12] [--compare other.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import re


def union(iv) -> int:
    tot, end = 0, -1
    for a, b in sorted(iv):
        if b <= end:
            continue
        tot += b - max(a, end)
        end = b
    return tot


def qkind(sub: str) -> str:
    if "Spill" in sub:
        return "spill"
    if "IO" in sub:
        return "io"
    return "other"


def load(path: str, top: int):
    d = json.load(open(path))
    eng = collections.defaultdict(list)
    hlo = collections.Counter()
    for i in d["instruction"]:
        t, du = i["timestamp"], i.get("duration") or 0
        e = i.get("subgroup", "?")
        eng[e].append((t, t + du))
        name = i.get("hlo_name") or ""
        base = name.split(" = ")[0].lstrip("%") if name else "(kernel)"
        hlo[(e, re.sub(r"\.\d+$", "", base))] += du
    dma = collections.Counter()
    for p in d.get("dma", []):
        dma[qkind(p.get("subgroup", ""))] += p.get("transfer_size", 0) or 0
    s = d["summary"][0]
    return eng, hlo, dma, s


def show(path: str, top: int) -> None:
    eng, hlo, dma, s = load(path, top)
    print(f"{path}:")
    print("  busy us: " + ", ".join(f"{e}={union(v) / 1e3:.0f}" for e, v in sorted(eng.items(), key=lambda kv: -union(kv[1]))))
    print("  dma MB: " + ", ".join(f"{k}={v / 1e6:.0f}" for k, v in dma.most_common()))
    keys = ("total_time", "hbm_read_bytes", "hbm_write_bytes", "spill_save_bytes", "spill_reload_bytes",
            "tensor_engine_active_time", "vector_engine_active_time", "scalar_engine_active_time", "dma_active_time",
            "cc_op_time")
    print("  summary: " + ", ".join(f"{k}={s[k]}" for k in keys if k in s))
    print("  top ops (engine:op instruction us): " + ", ".join(f"{e}:{n} {v / 1e3:.0f}" for (e, n), v in hlo.most_common(top)))


def timeline(path: str, lo: float, hi: float, n: int) -> None:
    """The instructions starting in [lo, hi) of the profile's span (fractions), in time order, every engine."""
    d = json.load(open(path))
    ins = sorted(d["instruction"], key=lambda i: i["timestamp"])
    t0, t1 = ins[0]["timestamp"], max(i["timestamp"] + (i.get("duration") or 0) for i in ins)
    a, b = t0 + (t1 - t0) * lo, t0 + (t1 - t0) * hi
    sel = [i for i in ins if a <= i["timestamp"] < b]
    print(f"timeline {lo:.3f}..{hi:.3f} of {(t1 - t0) / 1e3:.1f} us ({len(sel)} instructions):")
    for i in sel[:n]:
        print(f"  {(i['timestamp'] - t0) / 1e3:9.3f} us +{(i.get('duration') or 0):6d} ns wait {i.get('evt_wait_time', 0) or 0:6} "
              f"{i.get('subgroup', '?'):<8} {i.get('opcode', '?'):<24} {str(i.get('operands', ''))[:90]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="+")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--window", type=float, nargs=2, default=None, help="also list the instructions in this fraction "
                    "of the span (first JSON)")
    ap.add_argument("--lines", type=int, default=150)
    a = ap.parse_args()
    for p in a.json:
        show(p, a.top)
    if a.window:
        timeline(a.json[0], a.window[0], a.window[1], a.lines)


if __name__ == "__main__":
    main()
