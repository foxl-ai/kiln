"""Per-segment op split of one replayed piece (a neuron-explorer JSON view: `neuron-explorer view -n <neff> -s <ntff>
--output-format json --output-file view.json --ignore-dma-trace --ignore-nc-buf-usage`, HOME set). A segment runs from the
end of one collective to the start of the next and is named by the collective that ends it (what tools/util_report.py
report bins by); per segment type: the mean span, the time any compute engine is busy, each engine's busy time, the
engine time in NKI kernel instructions (by the kernel file their nki_source_location names) against XLA ops (by HLO name,
else the compiler's debug source location), and the ops with the most engine time.

    python tools/prof_segments.py view.json [--top 8]

Engine times are sums over the four compute engines (an instruction on the tensor engine and one on the vector engine at
the same time both count), so a segment's NKI + XLA engine time can exceed its span."""

import argparse
import collections
import json
import re


def op(i):
    """(side, name): an NKI kernel's instruction by its kernel file (nki_source_location), an XLA op by its HLO name,
    else by the compiler's debug source location (Python file:line of the model code), else "unnamed"."""
    k = i.get("nki_source_location") or ""
    if k:
        return "NKI", "NKI " + k.rsplit("/", 1)[-1].split(":")[0]
    n = i.get("hlo_name") or ""
    if n:
        return "XLA", re.sub(r"[._]\d+$", "", n.split("=")[0].strip().lstrip("%"))
    b = i.get("bir_debug_info_source_location") or ""
    if b:
        return "XLA", "src " + b.rsplit("/", 1)[-1]
    return "XLA", "unnamed " + (i.get("opcode") or "")


def union(iv):
    iv.sort()
    u, cs, ce = 0, None, None
    for s, e in iv:
        if ce is None or s > ce:
            if ce is not None:
                u += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return u + (ce - cs if ce is not None else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("view")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--first-cc", type=int, default=16, help="collectives to list in order")
    ap.add_argument("--no-by-kernel", dest="by_kernel", action="store_false",
                    help="do not split the segment types by their busiest NKI kernel (split by default)")
    a = ap.parse_args()
    d = json.load(open(a.view))
    cc = sorted(d["cc_ops"], key=lambda c: c["timestamp"])
    ins = sorted((i for i in d["instruction"] if i.get("timestamp") is not None), key=lambda i: int(i["timestamp"]))
    t0 = int(ins[0]["timestamp"]) if ins else 0
    t1 = max(int(i["timestamp"]) + int(i.get("duration") or 0) for i in ins) if ins else 0
    print(f"{len(cc)} collectives, {len(ins)} instructions, span {(t1 - t0) / 1e6:.3f} ms")
    for c in cc[: a.first_cc]:
        print(f"  {(c['timestamp'] - t0) / 1e6:8.3f} dur={c['duration'] / 1e6:6.3f} {c['operation']:14s} "
              f"in={int(c['input_size']) / 2**20:.2f}MiB grp={str(c['replica_group'])[:24]}")
    bounds, prev = [], t0
    for c in cc:
        bounds.append((prev, int(c["timestamp"]), f"{c['operation']} {int(c['input_size']) / 2**20:.2f}MiB"))
        prev = int(c["timestamp"]) + int(c["duration"])
    bounds.append((prev, t1, "end of graph"))
    agg = collections.defaultdict(collections.Counter)
    eng = collections.defaultdict(collections.Counter)
    tops = collections.defaultdict(collections.Counter)
    j = 0
    for lo, hi, kind in bounds:
        while j < len(ins) and int(ins[j]["timestamp"]) < lo:
            j += 1
        k, iv = j, []
        busy = collections.Counter()
        segeng, segtop = collections.Counter(), collections.Counter()
        while k < len(ins) and int(ins[k]["timestamp"]) < hi:
            i = ins[k]
            t, du = int(i["timestamp"]), int(i.get("duration") or 0)
            side, o = op(i)
            busy[side] += du
            segeng[f"{i.get('subgroup')}/{side}"] += du
            segtop[o] += du
            iv.append((t, t + du))
            k += 1
        nk = [(v, o) for o, v in segtop.items() if o.startswith("NKI ")]
        if a.by_kernel and nk:  # the segment named by its busiest NKI kernel too
            kind = f"{kind} [{max(nk)[1][4:]}]"
            for e_, v_ in segeng.items():
                eng[kind][e_] += v_
            for o_, v_ in segtop.items():
                tops[kind][o_] += v_
        elif a.by_kernel:
            kind = f"{kind} [no NKI]"
            for e_, v_ in segeng.items():
                eng[kind][e_] += v_
            for o_, v_ in segtop.items():
                tops[kind][o_] += v_
        ag = agg[kind]
        ag["n"] += 1
        ag["span"] += hi - lo
        ag["any"] += union(iv)
        ag["nki"] += busy["NKI"]
        ag["xla"] += busy["XLA"]
        if not a.by_kernel:
            for e_, v_ in segeng.items():
                eng[kind][e_] += v_
            for o_, v_ in segtop.items():
                tops[kind][o_] += v_
    print("segment type (ends in)            n   span ms  any-busy ms  engine ms: NKI / XLA")
    for kind, ag in sorted(agg.items(), key=lambda kv: -kv[1]["span"]):
        n = ag["n"]
        print(f"  {kind:30s} {n:3d} {ag['span'] / n / 1e6:8.3f} {ag['any'] / n / 1e6:10.3f}   "
              f"{ag['nki'] / n / 1e6:.3f} / {ag['xla'] / n / 1e6:.3f}   (total span {ag['span'] / 1e6:.1f} ms)")
        print("      per engine (ms per segment): " + ", ".join(
            f"{e} {v / n / 1e6:.3f}" for e, v in sorted(eng[kind].items(), key=lambda kv: -kv[1])[:8]))
        print("      top ops (engine ms per segment): " + ", ".join(
            f"{o} {v / n / 1e6:.3f}" for o, v in tops[kind].most_common(a.top)))


if __name__ == "__main__":
    main()
