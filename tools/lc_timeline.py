"""Per-call times of a long request from a KILN_TIMELINE file (kiln/profiling.py) of a serve_sweep run.

    python tools/lc_timeline.py /opt/kiln/logs/lc-1M-TL.timeline.jsonl [--at 4096 262144 524288 786432 1040384]

A step() record carries the prefill tokens and decode rows it launched. With one request in flight, the prefill steps
run its chunks in order, so the running sum of prefill tokens is the chunk's context position. Under overlap a step()
call reads back the previous launch, so its wall time is the previous call's device time in steady state. Prints:
the step wall time at the chosen positions (mean of the 8 steps around each), a linear fit of prefill step time against
position (the part that grows with the context is the long path's), the decode step time at the end, and the longest
host-blocked sections (read-backs and steps), which is what the exec watchdog (kiln/engine/watchdog.py) would see.
"""

from __future__ import annotations

import argparse
import json


def load(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline")
    ap.add_argument("--at", type=int, nargs="+", default=[4096, 131072, 262144, 524288, 786432, 1040384])
    ap.add_argument("--skip", type=int, default=2, help="first prefill steps left out of the fit (graph loads)")
    a = ap.parse_args()
    recs = load(a.timeline)
    steps = [r for r in recs if r[0] == "step"]
    reads = [r for r in recs if r[0] == "read"]
    pos, pre, dec = 0, [], []
    for r in steps:
        _, t0, t1, n_dec, n_pre = r[:5]
        if n_pre:
            pos += n_pre
            pre.append((pos, t1 - t0, n_pre, n_dec))
        elif n_dec:
            dec.append((pos, t1 - t0, n_dec))
    print(f"{len(steps)} steps: {len(pre)} with prefill (tokens {pos}), {len(dec)} decode only; "
          f"prefill wall {sum(s for _, s, _, _ in pre):.1f} s, decode wall {sum(s for _, s, _ in dec):.1f} s")
    for p in a.at:
        near = sorted(pre, key=lambda x: abs(x[0] - p))[:8]
        if near:
            print(f"  prefill step at position ~{p:>8}: {sum(s for _, s, _, _ in near) / len(near):.3f} s "
                  f"(tokens per step {near[0][2]})")
    fit = pre[a.skip:]
    if len(fit) > 2:
        n = len(fit)
        mx = sum(p for p, *_ in fit) / n
        my = sum(s for _, s, *_ in fit) / n
        b = sum((p - mx) * (s - my) for p, s, *_ in fit) / max(sum((p - mx) ** 2 for p, *_ in fit), 1e-9)
        c = my - b * mx
        print(f"  fit: step = {c:.3f} s + {b * 1e6:.4f} us x position  (at 1M: {c + b * 2**20:.3f} s; "
              f"integral over the prompt {sum(s for _, s, *_ in fit):.1f} s, of which grows-with-context "
              f"{sum(b * p for p, *_ in fit):.1f} s)")
    if dec:
        tail = dec[-64:]
        print(f"  decode step at context ~{tail[-1][0]}: {sum(s for _, s, _ in tail) / len(tail) * 1e3:.1f} ms "
              f"(mean of the last {len(tail)}); first decode steps {[round(s * 1e3) for _, s, _ in dec[:4]]} ms")
    worst = sorted(reads, key=lambda r: r[2] - r[1], reverse=True)[:5]
    print("  longest read-backs:", [(r[3], round(r[2] - r[1], 2)) for r in worst])
    print("  longest steps:", [round(t1 - t0, 2) for _, t0, t1, *_ in sorted(steps, key=lambda r: r[2] - r[1],
                                                                                reverse=True)[:5]])


if __name__ == "__main__":
    main()
