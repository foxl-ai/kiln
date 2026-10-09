"""Per-position logprob differences of two tools/check_long.py nll runs (their --out-json files), by band.

    python tools/nll_compare.py A.json B.json

For each band of positions: each run's NLL, mean(A - B) (the signed bias: A's logprob minus B's, so positive means A
assigns the text more probability), mean |A - B|, the share of positions with |A - B| > 0.01 and the max. logprobs[i]
in a run's json is position i + 1 (check_long.py run_nll). The long-context gate: no band with a signed bias beyond the
spread two numerically neutral engines show (docs/neuron-notes.md "NLL by band at 128k against CP8CE-m").
"""

import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
la, lb = a["logprobs"], b["logprobs"]
n = min(len(la), len(lb))
print(f"A {sys.argv[1]}: mean nll {a['mean_nll']:.4f} over {len(la)}; B {sys.argv[2]}: {b['mean_nll']:.4f} over {len(lb)}")
edges = [0, 1024, 4096, 16384, 32768, 65536, 131072, 262144, 524288, 1 << 30]
for lo, hi in zip(edges, edges[1:]):
    # logprobs[i] is position i + 1
    idx = [i for i in range(n) if lo <= i + 1 < hi]
    if not idx:
        continue
    d = [la[i] - lb[i] for i in idx]
    ad = [abs(x) for x in d]
    na = -sum(la[i] for i in idx) / len(idx); nb = -sum(lb[i] for i in idx) / len(idx)
    print(f"  [{lo:>6}, {min(hi, n + 1):>6}): n {len(idx):6d}  nll A {na:.4f} B {nb:.4f}  mean(A-B) {sum(d)/len(d):+.5f}  "
          f"mean|d| {sum(ad)/len(ad):.4f}  |d|>0.01 {sum(x > 0.01 for x in ad)/len(ad):.1%}  max|d| {max(ad):.3f}")
d = [abs(la[i] - lb[i]) for i in range(n)]
print(f"  all: mean|d| {sum(d)/n:.4f}  mean(A-B) {sum(la[i]-lb[i] for i in range(n))/n:+.5f}")
