"""The 1M NLL gate over several windows: per band of --band positions, the signed mean of A's logprob minus B's
(tools/check_long.py nll --out-json files, logprobs[i] = position i + 1), then over every band of every window the
signed mean over all positions and the mean and sd of the band means (docs/neuron-notes.md "It does not replicate":
R8 - CP8CE-m over two 1M windows, 16 bands of 131k, signed -0.00006, band sd 0.0063).

    python tools/nll_bands.py A1.json B1.json [A2.json B2.json ...] [--band 131072]
"""

from __future__ import annotations

import argparse
import json
import math


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--band", type=int, default=131072)
    a = ap.parse_args()
    if len(a.pairs) % 2:
        raise SystemExit("give A / B json pairs")
    means, tot, n_all = [], 0.0, 0
    for w in range(0, len(a.pairs), 2):
        la = json.load(open(a.pairs[w]))["logprobs"]
        lb = json.load(open(a.pairs[w + 1]))["logprobs"]
        n = min(len(la), len(lb))
        print(f"window {w // 2}: {a.pairs[w]} vs {a.pairs[w + 1]}, {n} positions")
        for lo in range(0, n, a.band):
            hi = min(lo + a.band, n)
            d = sum(la[i] - lb[i] for i in range(lo, hi))
            nll_a = -sum(la[lo:hi]) / (hi - lo)
            nll_b = -sum(lb[lo:hi]) / (hi - lo)
            means.append(d / (hi - lo))
            tot += d
            n_all += hi - lo
            print(f"  [{lo:>8}, {hi:>8}): nll A {nll_a:.4f} B {nll_b:.4f}  mean(A-B) {d / (hi - lo):+.5f}")
    m = sum(means) / len(means)
    sd = math.sqrt(sum((x - m) ** 2 for x in means) / (len(means) - 1)) if len(means) > 1 else float("nan")
    print(f"all {n_all} positions: signed mean(A-B) {tot / n_all:+.5f}; {len(means)} bands: mean {m:+.5f}, sd {sd:.4f}, "
          f"range {min(means):+.4f} .. {max(means):+.4f}")


if __name__ == "__main__":
    main()
