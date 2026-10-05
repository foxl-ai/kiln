"""Where one compiled graph's time goes, segment by segment, from a neuron-explorer device profile (SDK 2.32).

    neuron-explorer capture -n g.neff -s g.ntff -r 32 -i 0 --ignore-exec-errors   # replay on 32 cores, profile rank 0
    neuron-explorer view -n g.neff -s g_rank_0.ntff --output-format json             # (HOME must be set) -> ntff.json
    python tools/prof_step.py ntff.json [--top 8] [--labels attn,ffn,...]

The graph's collectives cut its timeline into segments: segment i runs from the end of collective i - 1 (or the
graph's start) to the trigger of collective i, the last one from the last collective's end to the graph's end. In a
decode layer-group graph every layer holds two (the token mixer's output all-reduce, then the MLP's), so the
segments alternate token mixer (with the attention block's hyper-connection arithmetic) and FFN block (with the
mix-out of the mixer and the FFN's own hyper-connection arithmetic). Per segment: its wall time, each engine's busy
time (the union of its instructions' intervals), the HBM bytes its DMA packets moved by queue kind, and the HLO ops
with the most busy time (an NKI kernel's instructions carry no HLO name: they show as "(kernel)"). Per collective:
the trigger-to-start wait and the transfer. The replay runs every input as zeros unless inputs are given to
`capture`, so data-dependent work (MoE routing: which experts load) is not the real one.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import re


def union(iv: list[tuple[int, int]]) -> int:
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
    if sub.startswith("qSyncIO") or "IO" in sub:
        return "io"
    if "DGE" in sub or "Dyn" in sub or "Indirect" in sub:
        return "dynamic"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--labels", default="", help="comma list naming the segments in order")
    ap.add_argument("--layers", default="", help="the graph's layer kinds in order, one letter each (e.g. KKKD for "
                    "three KDA layers and a DSA one): segments are labelled attn-<kind>, ffn per layer, then tail")
    a = ap.parse_args()
    d = json.load(open(a.json))
    t_end = d["metadata"][0]["last_hw_timestamp"]
    cc = sorted(d["cc_ops"], key=lambda c: c["timestamp"])
    bounds = []  # (segment start, segment end)
    start = 0
    for c in cc:
        bounds.append((start, c["cc_trigger"]))
        start = c["timestamp"] + c["duration"]
    bounds.append((start, t_end))
    starts = [b[0] for b in bounds]

    def seg(t: int) -> int:
        return max(0, bisect.bisect_right(starts, t) - 1)

    eng = [collections.defaultdict(list) for _ in bounds]
    hlo = [collections.Counter() for _ in bounds]
    for i in d["instruction"]:
        t, du = i["timestamp"], i.get("duration") or 0
        s = seg(t)
        e = i.get("subgroup", "?")
        eng[s][e].append((t, t + du))
        name = i.get("hlo_name") or ""
        base = name.split(" = ")[0].lstrip("%") if name else "(kernel)"
        base = re.sub(r"\.\d+$", "", base)
        hlo[s][(e, base)] += du
    dma = [collections.Counter() for _ in bounds]
    for p in d["dma"]:
        s = seg(p["timestamp"])
        dma[s][qkind(p.get("subgroup", ""))] += p.get("transfer_size", 0) or 0
    labels = a.labels.split(",") if a.labels else []
    if a.layers:
        labels = [x for c in a.layers for x in (f"attn-{c}", "ffn")] + ["tail"]
    tot, cnt = collections.Counter(), collections.Counter()
    print(f"graph {t_end / 1e3:.1f} us, {len(cc)} collectives, {len(bounds)} segments")
    for k, (lo, hi) in enumerate(bounds):
        w = hi - lo
        lab = labels[k] if k < len(labels) else ""
        busy = {e: union(v) for e, v in eng[k].items()}
        tot[lab or "?"] += w
        cnt[lab or "?"] += 1
        es = " ".join(f"{e}={b / 1e3:.0f}" for e, b in sorted(busy.items(), key=lambda x: -x[1])[:5])
        ds = " ".join(f"{q}={v / 1e6:.1f}MB" for q, v in dma[k].most_common())
        print(f"seg {k:3d} {lab:10s} {lo / 1e3:9.1f}..{hi / 1e3:9.1f} us  {w / 1e3:8.1f} us | busy us {es} | dma {ds}")
        if a.top:
            top = ", ".join(f"{e}:{n} {v / 1e3:.0f}us" for (e, n), v in hlo[k].most_common(a.top))
            print(f"      top: {top}")
    for k, c in enumerate(cc):
        tot["collective wait"] += c["cc_trigger_start_delay"]
        tot["collective transfer"] += c["duration"]
    print("collectives: " + ", ".join(f"{c['operation']} {c['input_size'] // 1024}KB wait {c['cc_trigger_start_delay'] / 1e3:.2f} "
                                       f"+ {c['duration'] / 1e3:.3f} ms" for c in cc[:4]) + (" ..." if len(cc) > 4 else ""))
    print("by label (ms, total / count = mean): " + ", ".join(
        f"{k}={v / 1e6:.2f}" + (f"/{cnt[k]}={v / 1e6 / cnt[k]:.3f}" if cnt[k] else "") for k, v in tot.items()))
    s = d["summary"][0]
    print("summary: " + ", ".join(f"{k}={s[k]}" for k in ("total_time", "hbm_read_bytes", "hbm_write_bytes", "spill_save_bytes",
                                                         "spill_reload_bytes", "software_dynamic_dma_size", "static_dma_size",
                                                         "tensor_engine_active_time", "vector_engine_active_time",
                                                         "scalar_engine_active_time", "gpsimd_engine_active_time",
                                                         "dma_active_time", "cc_op_time") if k in s))


if __name__ == "__main__":
    main()
