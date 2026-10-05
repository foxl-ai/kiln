"""Compare two tools/check_ppl.py --out-json runs over the same text (device against device, or device
against a CPU reference): mean prompt logprob of each, per 256-token chunk, mean and max |dlogprob|, and
greedy agreement (the share of positions whose top-1 token is the same in both).

    python tools/compare_ppl.py a.json b.json [--chunk 256]
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--chunk", type=int, default=256)
    args = ap.parse_args()
    with open(args.a) as f:
        A = json.load(f)
    with open(args.b) as f:
        B = json.load(f)
    for i, (x, y) in enumerate(zip(A, B)):
        if x["ids"] != y["ids"]:
            raise SystemExit(f"text {i}: the two runs scored different token ids")
        la, lb = x["logprobs"], y["logprobs"]
        n = len(la)
        d = [abs(p - q) for p, q in zip(la, lb)]
        agree = sum(p == q for p, q in zip(x["top1"], y["top1"])) / n
        c = args.chunk
        by = lambda l: " ".join(f"{sum(l[j:j + c]) / len(l[j:j + c]):.3f}" for j in range(0, n, c))  # noqa: E731
        print(f"text {i}: {n} scored tokens")
        print(f"  mean logprob  a {sum(la) / n:.4f}   b {sum(lb) / n:.4f}   difference {sum(la) / n - sum(lb) / n:+.4f}")
        print(f"  by chunk a: {by(la)}")
        print(f"  by chunk b: {by(lb)}")
        print(f"  |dlogprob| mean {sum(d) / n:.4f}, max {max(d):.3f}; greedy agreement {agree:.4f}")


if __name__ == "__main__":
    main()
