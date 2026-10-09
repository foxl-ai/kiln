"""Two time_decode --real-kv --dump files of one bucket, same seed (the same token ids and KV fill every step), token for
token: the first launch (identical inputs and state, so only the configurations' arithmetic differs), then every step
(each step's KV row and recurrent state follow the step before, so differences compound).

    python tools/compare_decode_dumps.py <a>-B96.npy <b>-B96.npy

Columns per row: sampled token, its logprob, then the top-N ids and logprobs (kiln/engine/sampler.py sample)."""

from __future__ import annotations

import sys

import numpy as np


def main() -> None:
    a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    tok_eq = a[..., 0] == b[..., 0]
    dl = np.abs(a[..., 1] - b[..., 1])
    print(f"first launch: tokens equal {int(tok_eq[0].sum())} / {tok_eq.shape[1]}, |dlogprob| mean {dl[0].mean():.2e} "
          f"max {dl[0].max():.2e}, signed mean {(b[0, :, 1] - a[0, :, 1]).mean():+.2e}")
    both = tok_eq  # the logprob of the same sampled token
    print(f"all {n} steps: tokens equal {tok_eq.mean() * 100:.2f}% ({int(tok_eq.sum())} / {tok_eq.size}); |dlogprob| "
          f"where equal mean {dl[both].mean():.2e} p99 {np.percentile(dl[both], 99):.2e} max {dl[both].max():.2e}; "
          f"signed mean {(b[..., 1] - a[..., 1])[both].mean():+.2e}")
    per = tok_eq.mean(1)
    print("per step agreement: " + " ".join(f"{x:.3f}" for x in per[: min(n, 16)]) + (" ..." if n > 16 else ""))


if __name__ == "__main__":
    main()
