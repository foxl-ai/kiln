"""Where a replayed prefill call's time goes, from `tools/util_report.py bins` output (one <seq>-<key>.bins.json per
replayed graph, 100 us bins of each compute engine's occupancy, the segments between collectives and the collectives).

    python tools/prefill_split.py <prof dir> [seq ...]     # seq: graphs that run twice in the call (counted twice)

Per part (a segment, labelled by the collective that ends it and the one before, as the long-context agent's
classify.py does): its time and the bins inside it where every compute engine is idle with no collective in flight
(DMA-only work: the bins ignore the DMA trace). Per collective kind: trigger-to-start wait, transfer, and the bins of its
window where every engine is idle. Measured on the 1M R8 call (logs/kiln-lc2-32/prof/p200, 2026-10-06): of 150.2 ms
all-idle inside segments, 144.9 ms is in DSA segments A and B; the world RS 32 MiB is 1.381 ms per transfer.
CPU only, seconds.
"""
import json, sys, glob, os, collections
ENG = ['tensor', 'vector', 'scalar', 'gpsimd']

def label(op, prev, after_route, ms):
    if op == ("ReduceScatter", 128.0): return "DSA lse combine"
    if op == ("AllReduce", 2.0): return "DSA seg B"
    if op == ("AllReduce", 8.0): return "DSA seg A" if ms > 5 else "DSA CP merge piece / attn-group gather"
    if op == ("ReduceScatter", 32.0):
        if prev == ("ReduceScatter", 128.0): return "DSA W_UV + o_proj"
        return "FFN (EP MoE, shared)" if after_route else "token mixer (KDA, or attn)"
    if op == ("ReduceScatter", 8.0): return "token mixer (attn group RS)"
    if op in (("AllReduce", 0.25), ("AllGather", 0.01)): return "FFN routing"
    if op == ("AllGather", 1.0): return "block entry (before SP gather)"
    if op is None: return "end of graph"
    return f"other {op}"

def main(d, twice):
    part = collections.defaultdict(collections.Counter)
    ccs_k = collections.defaultdict(collections.Counter)
    span = 0.0
    for f in sorted(glob.glob(os.path.join(d, "*.bins.json"))):
        b = json.load(open(f))
        tag = os.path.basename(f).split(".")[0]
        reps = 2 if tag.split("-")[0] in twice else 1
        span += b["span_ms"] * reps
        if b["span_ms"] < 20:
            part["prep+post"]["ms"] += b["span_ms"] * reps
            part["prep+post"]["idle_cc"] += b["idle_cc_ms"] * reps
            continue
        B = b["bin_us"] / 1e3
        occ = b["occ"]; nb = b["bins"]
        idle = [all(occ[e][i] < 0.05 for e in ENG if e in occ) for i in range(nb)]
        segs, ccs = b["segments"], b["collectives"]
        prev, after_route = None, False
        for i, s in enumerate(segs):
            c = ccs[i] if i < len(ccs) else None
            op = (c["op"], round(c["bytes"] / 2**20, 2)) if c else None
            lab = label(op, prev, after_route, s["ms"])
            lo, hi = s["lo_us"] / 1e3, s["lo_us"] / 1e3 + s["ms"]
            nxt = segs[i + 1]["lo_us"] / 1e3 if i + 1 < len(segs) else hi
            idle_seg = sum(B for q in range(int(lo / B), min(nb, int(hi / B))) if idle[q])
            idle_cc = sum(B for q in range(int(hi / B), min(nb, int(nxt / B))) if idle[q])
            p = part[lab]
            p["n"] += reps; p["ms"] += s["ms"] * reps; p["idle_seg"] += idle_seg * reps
            for e in ENG: p[e] += s.get(e, 0) * reps
            if c:
                k = f"{c['op']} {c['bytes'] / 2**20:.2f} MiB"
                q = ccs_k[k]; q["n"] += reps; q["wait"] += c["wait_ms"] * reps; q["xfer"] += c["ms"] * reps
                q["window"] += max(0, nxt - hi) * reps; q["idle_cc"] += idle_cc * reps
                q["after:" + lab] += reps
            if op == ("AllReduce", 0.25) or op == ("AllGather", 0.01): after_route = True
            elif op == ("ReduceScatter", 32.0): after_route = False
            prev = op
    print(f"call span (replay, rank 0): {span:.1f} ms")
    print(f"  {'part (segment, labelled by the collective ending it)':52s} {'n':>4s} {'ms':>8s} {'all-idle in seg':>16s}  busy share t/v/s")
    for k, p in sorted(part.items(), key=lambda kv: -kv[1]["ms"]):
        busy = " ".join(f"{p[e] / p['ms']:.0%}" for e in ENG[:3]) if p["ms"] else ""
        print(f"  {k:52s} {p['n']:4.0f} {p['ms']:8.1f} {p['idle_seg']:16.1f}  {busy}")
    print(f"  {'collective':28s} {'n':>4s} {'wait':>7s} {'xfer':>7s} {'window':>7s} {'all-idle in window':>19s}")
    tw = tx = ti = 0
    for k, q in sorted(ccs_k.items(), key=lambda kv: -kv[1]["window"]):
        print(f"  {k:28s} {q['n']:4.0f} {q['wait']:7.1f} {q['xfer']:7.1f} {q['window']:7.1f} {q['idle_cc']:19.1f}")
        tw += q["wait"]; tx += q["xfer"]; ti += q["idle_cc"]
    print(f"  totals: wait {tw:.1f} + transfer {tx:.1f} ms; all-idle in collective windows {ti:.1f}; all-idle inside segments "
          f"{sum(p['idle_seg'] for p in part.values()):.1f}")

main(sys.argv[1], set(sys.argv[2:]))
