"""Op-level reading of one replayed graph's profile: what a time window of the step spends, and which HLO op it is.

    python tools/prof_layer_ops.py cc       <view.json>                          # the collectives in order (gaps, sizes)
    python tools/prof_layer_ops.py window   <view.json> LO:HI[,LO:HI...] [names]  # per window: span, any-engine active, per engine
    python tools/prof_layer_ops.py timeline <view.json> LO HI [gap_us] [min_us]   # runs of instructions by HLO op (NKI = unnamed)
    python tools/prof_layer_ops.py named    <view.json> LO HI [min_us]            # the named ops (layer_summary) starting in LO..HI
    python tools/prof_layer_ops.py hlo      <graph.hlo> name[,name...] [depth]    # shapes and operands of HLO ops by name

<view.json> is `neuron-explorer view -n <neff> -s <ntff> --output-format json --output-file <view.json>
--ignore-dma-trace --ignore-nc-buf-usage` (1.7 GB for a 48 ms piece of the CP-64 decode call); LO / HI in ms from the
graph's start. <graph.hlo> is the HloModuleProto in the NEFF's compile-cache directory (no op metadata survives there, so
an op is recognised by its shapes: docs/neuron-notes.md "One DSA layer under CP, op by op" was read this way). NKI kernel
instructions carry no HLO name: they show as runs of unnamed instructions, and a kernel called N times as N equal runs.
"""

from __future__ import annotations

import collections
import json
import re
import sys


def _load(p):
    return json.load(open(p))


def cmd_cc(d):
    prev = 0
    for c in sorted(d["cc_ops"], key=lambda c: c["timestamp"]):
        t = c["timestamp"]
        print(f"{t / 1e6:8.3f} gap={(t - prev) / 1e6:6.3f} dur={c['duration'] / 1e6:6.3f} {c['operation']:14s} "
              f"{c['input_size']:>9} {c['output_size']:>9} {c['replica_group'][:30]:30s} trig={c['trigger_engine']} "
              f"wait={c.get('cc_trigger_start_delay', 0) / 1e6:.3f}")
        prev = t + c["duration"]


def cmd_window(d, wins, names):
    act = d["active_time"]
    for (lo, hi), nm in zip(wins, names):
        lo, hi = lo * 1e6, hi * 1e6
        per, iv = collections.Counter(), []
        for a in act:
            s, e = max(a["start_ts"], lo), min(a["end_ts"], hi)
            if e > s:
                per[a["engine"]] += e - s
                if a["engine"] != "cc_instruction":
                    iv.append((s, e))
        iv.sort()
        u, cs, ce = 0, None, None
        for s, e in iv:
            if ce is None or s > ce:
                if ce is not None:
                    u += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        if ce is not None:
            u += ce - cs
        print(f"{nm:14s} {lo / 1e6:7.3f}-{hi / 1e6:7.3f} span={(hi - lo) / 1e3:7.1f}us any={u / 1e3:7.1f}us "
              + " ".join(f"{k[:6]}={v / 1e3:.0f}" for k, v in sorted(per.items())))


def _op(i):
    n = i.get("hlo_name") or ""
    if not n:
        return "NKI/unnamed"
    return re.sub(r"[._]\d+$", "", n.split("=")[0].strip().lstrip("%"))


def cmd_timeline(d, lo, hi, gap, mn):
    ins = sorted((i for i in d["instruction"] if i.get("timestamp") is not None and lo <= int(i["timestamp"]) < hi),
                 key=lambda i: int(i["timestamp"]))
    blocks = []
    for i in ins:
        k, t = _op(i), int(i["timestamp"])
        dur = int(i.get("duration") or 0)
        if blocks and blocks[-1]["k"] == k and t - blocks[-1]["e"] < gap:
            b = blocks[-1]
            b["e"] = max(b["e"], t + dur)
            b["n"] += 1
            b["eng"][i.get("subgroup")] += dur
            b["ops"][i.get("opcode")] += 1
        else:
            blocks.append({"k": k, "s": t, "e": t + dur, "n": 1, "eng": collections.Counter({i.get("subgroup"): dur}),
                           "ops": collections.Counter({i.get("opcode"): 1})})
    for b in blocks:
        if b["e"] - b["s"] < mn:
            continue
        print(f"{b['s'] / 1e6:8.3f}-{b['e'] / 1e6:8.3f} {(b['e'] - b['s']) / 1e3:8.1f}us n={b['n']:6d} {b['k'][:40]:40s} "
              + " ".join(f"{k[:3]}={v / 1e3:.0f}" for k, v in b["eng"].most_common(4)) + "  "
              + ",".join(f"{k}:{v}" for k, v in b["ops"].most_common(3)))


def cmd_named(d, lo, hi, mn):
    full = {}
    for i in d["instruction"]:
        n = i.get("hlo_name")
        if n and lo <= int(i["timestamp"]) < hi:
            full.setdefault(n.split("=")[0].strip().lstrip("%"), n)
    for x in sorted((x for x in d["layer_summary"] if lo <= x["start"] < hi), key=lambda x: x["start"]):
        dur = x["end"] - x["start"]
        if dur >= mn:
            nm = x["name"].split("/")[-1].lstrip("_")
            print(f"{x['start'] / 1e6:8.3f} {dur / 1e3:8.1f}us {x['name'][:30]:30s} {full.get(nm, '')[:200]}")


ET = {1: "pred", 2: "s8", 3: "s16", 4: "s32", 5: "s64", 6: "u8", 8: "u32", 10: "f16", 11: "f32", 16: "bf16", 20: "f8e4m3fn"}


def cmd_hlo(path, names, depth):
    from libtorch_neuronx_lite.pyhlo.service import hlo_pb2  # the SDK venv's HLO proto

    m = hlo_pb2.HloModuleProto()
    m.ParseFromString(open(path, "rb").read())
    byid, byname = {}, {}
    for c in m.computations:
        for i in c.instructions:
            byid[i.id] = i
            byname[i.name] = i

    def sh(i):
        s = i.shape
        if s.tuple_shapes:
            return "(" + ",".join(ET.get(t.element_type, str(t.element_type)) + str(list(t.dimensions)) for t in s.tuple_shapes) + ")"
        return ET.get(s.element_type, str(s.element_type)) + str(list(s.dimensions))

    def desc(i, k):
        ex = f" param#{i.parameter_number}" if i.opcode == "parameter" else ""
        if i.opcode == "gather":
            g = i.gather_dimension_numbers
            ex = f" slice={list(i.gather_slice_sizes)} off={list(g.offset_dims)} coll={list(g.collapsed_slice_dims)}"
        out = f"{i.name} {i.opcode} {sh(i)}{ex}"
        if k > 0:
            out += " <- [" + "; ".join(desc(byid[o], k - 1) for o in i.operand_ids) + "]"
        return out

    for n in names:
        if n in byname:
            print(desc(byname[n], depth))


def main() -> None:
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "hlo":
        return cmd_hlo(a[0], a[1].split(","), int(a[2]) if len(a) > 2 else 1)
    d = _load(a[0])
    if cmd == "cc":
        cmd_cc(d)
    elif cmd == "window":
        wins = [tuple(map(float, w.split(":"))) for w in a[1].split(",")]
        cmd_window(d, wins, a[2].split(",") if len(a) > 2 else [str(i) for i in range(len(wins))])
    elif cmd == "timeline":
        cmd_timeline(d, float(a[1]) * 1e6, float(a[2]) * 1e6, float(a[3]) * 1e3 if len(a) > 3 else 20e3,
                     float(a[4]) * 1e3 if len(a) > 4 else 30e3)
    elif cmd == "named":
        cmd_named(d, float(a[1]) * 1e6, float(a[2]) * 1e6, float(a[3]) * 1e3 if len(a) > 3 else 20e3)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
