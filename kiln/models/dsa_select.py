"""Exact top-k selection without torch.topk, as a mask: DeepSeek Sparse Attention's index_topk
(models/mla.py: DeepSeek-V3.2, GLM-5.3) and GLM-5.3-Flash's pooled indexer (models/glm5_next.py).

`torch.topk(scores, keep)` over the context is what dominated a DSA layer on trn1 (3.2 ms of a
6.0 ms GLM-5.3 indexer layer at 4K keys, 6.0 of 9.8 ms at 8K, decode B=8), and turning its
indices into an attention mask with a scatter cost 6.7 ms more in a 128-token prefill chunk
(docs/neuron-notes.md, "MLA and DSA on trn1"). The attention only needs WHICH keys are selected,
so this module finds the keep-th largest score t by counting, never sorting, and returns the
selection as a boolean mask directly. KILN_DSA_SELECT picks how t is found:

- "bisect": a bisection over the ORDER of the fp32 values, not over their range.
  Invariant: count(s >= lo) >= keep > count(s >= hi). Each round splits (lo, hi) at 0 when it
  holds both signs, at the geometric mean when one end is more than twice the other (halving the
  exponent distance) and at the arithmetic mean otherwise (halving the floats of a binade), so
  after BISECT_ROUNDS = 48 rounds lo and hi are ADJACENT floats (worst case about 41: one sign
  split, 9 to bring any two magnitudes within a factor 2 of each other, 7 more down to the smallest
  normal when lo is 0, 24 for a binade), and t = lo exactly. A bisection of the values themselves
  (qwen4_exp's QSA selection) cannot separate two scores closer than range / 2^rounds: DSA scores
  range over ~1e30 with the invisible keys at NEG_INF, and the threshold often sits at an exact 0.
- "radix": the same threshold built bit by bit from each score's bit pattern (an order-preserving
  int32 key). Exact and simpler, but it needs an in-graph bitcast (Tensor.view(torch.int32)), which
  LNL rejects on trn1 (SDK 2.32, 2026-10-03: "common_device INTERNAL ASSERT FAILED" at the first use
  of the view, tools/probe_dsa_select.py): CPU only, as a second reference.
- "topk": torch.topk and a scatter (the old path).
- "nki" (default): kernels/dsa_topk.py, the same tie rule exactly (a radix search over the
  threshold's bit pattern and a binary search over the tied positions) as one NKI kernel on the
  device, its torch emulation on the host. On trn1 it was exact on every score kind of
  tools/probe_dsa_select.py, "bisect" not on wide-range ones (device sqrt), 2026-10-04.
  GLM-5.3-Flash's pooled selection (glm5_next.block_mask) reads the same variable, each value
  meaning the same algorithm there.

Tie rule (the definition the tests check): exactly `keep` positions are selected (every position
when a row has fewer); every selected score is >= every unselected one; among the scores equal to
the keep-th largest, the lowest indices are taken. torch.topk returns the same set whenever the
keep-th and (keep + 1)-th largest scores differ, and an arbitrary choice of the tied ones otherwise
(CPU torch.topk is not lowest-index-first among ties; docs/neuron-notes.md). -0.0 and +0.0 are
equal, as in every float comparison. Scores must not be NaN.

No comparison is written against a float literal (those can lower to f64 in neuronx-cc,
docs/neuron-notes.md); counts are float sums of 0 / 1 (exact below 2^24 positions), the form
neuronx-cc compiled for QSA. The tie fill (the lowest-index `room` of the tied positions,
KILN_DSA_TIES) is a count search over the position ("index": the largest J with fewer than `room`
tied positions below J, log2(N) + 1 more count rounds), a running count by a cumsum within blocks of
128 plus the blocks' prefix ("block") or one cumsum over the row ("cumsum"). "auto" (default) takes
"block" for a prefill chunk's scores ([1, C, L]) and "index" otherwise, as measured inside GLM-5.3's
DSA layer on trn1 (SDK 2.32, 2026-10-03, tools/profile_mla.py, docs/neuron-notes.md): a 128-token
chunk over 8K keys 7.1 ms with "block" against 10.8 ms with "index" (16K: 14.0 against 13.2),
while a decode layer took 9.6 ms with "block" at B=8 over 4K keys and 18.2 ms at B=32 over 8K against
5.7 and 13.9 ms with "index". One cumsum over each 16K-key row made the selection of 128 rows 12.5 ms.
"""

from __future__ import annotations

import os

import torch

# "nki" by default: exact on the device too (tools/probe_dsa_select.py, docs/neuron-notes.md), where
# "bisect" was not on wide-range scores; on the host it runs the kernel's emulation.
SELECT = os.environ.get("KILN_DSA_SELECT", "nki")
TIES = os.environ.get("KILN_DSA_TIES", "auto")
GROUP = os.environ.get("KILN_DSA_GROUP", "1") == "1"
BISECT_ROUNDS = int(os.environ.get("KILN_DSA_ROUNDS", 48))
INT_MIN = -(2**31)
FLT_MAX = float(torch.finfo(torch.float32).max)


def _count(x: torch.Tensor) -> torch.Tensor:
    """How many of each row of the boolean x are True, [..., 1] fp32."""
    return x.float().sum(-1, keepdim=True)


def _split(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """A float strictly inside (lo, hi) that roughly halves the floats between them (rows of
    [..., 1], lo < hi; lo == hi gives lo): 0 when the interval holds both signs; else, on the
    magnitudes 0 <= x < y of its ends, sqrt(x y) when y > 2 x, (x + y) / 2 when y <= 2 x, and
    when x is 0 sqrt(y) 2^-63 (the geometric mean of y and the smallest normal 2^-126) while that
    is below y / 2, y / 2 after."""
    zero = torch.zeros_like(lo)
    neg = hi <= zero  # within (-inf, 0]: mirror it
    pos = lo >= zero
    x = torch.where(neg, -hi, lo)
    y = torch.where(neg, -lo, hi)
    # An infinite end splits like the largest finite float (inf is the next float after it).
    xs, ys = torch.where(x > zero, x, zero), torch.where(y > zero, y, zero).clamp(max=FLT_MAX)
    ry = torch.sqrt(ys)
    tiny = ry * 2.0**-63
    m = torch.where(x > zero, torch.where(ys > x + x, torch.sqrt(xs) * ry, x * 0.5 + ys * 0.5),
                    torch.where(tiny + tiny < ys, tiny, ys * 0.5))
    return torch.where(pos | neg, torch.where(neg, -m, m), zero)


def _groups(rows: int, n: int, group: bool | None = None) -> int:
    """Pieces each row's scores are cut into for counting, so that rows x pieces fills the 128
    partitions of a NeuronCore: a [8, 8192] count ran on 8 of them (tools/probe_dsa_select.py)."""
    if not (GROUP if group is None else group):
        return 1
    g = max(1, 128 // max(rows, 1))
    while g > 1 and n % g:
        g //= 2
    return g


def kth_value(scores: torch.Tensor, keep: int, group: bool | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(lo, hi) [..., 1] bracketing the keep-th largest score t of each row: count(s >= lo) >=
    keep > count(s >= hi) and no float strictly between them, so t == lo; lo == hi == max when
    keep or more scores equal the max. Rows need at least keep scores. (Counting against several
    candidates per pass, to read the scores fewer times, was slower on trn1: 1.43 / 2.10 ms with 3 /
    7 candidates against 0.97 ms for 8 x 4096, tools/probe_dsa_select.py, 2026-10-03.)"""
    *lead, n = scores.shape
    rows = 1
    for d in lead:
        rows *= d
    g = _groups(rows, n, group)
    # [rows * g, n / g]: each piece a row of its own (on the partition axis), the threshold of its
    # row broadcast to it, the piece counts summed per row.
    s = scores.float().reshape(rows * g, n // g)

    def count(m):  # count(s >= m) per row, [rows, 1]
        c = _count(s >= (m if g == 1 else m.expand(rows, g).reshape(rows * g, 1)))
        return c if g == 1 else c.reshape(rows, g).sum(-1, keepdim=True)

    lo = s.amin(-1, keepdim=True).reshape(rows, g).amin(-1, keepdim=True)
    hi = s.amax(-1, keepdim=True).reshape(rows, g).amax(-1, keepdim=True)
    lo = torch.where(count(hi) >= keep, hi, lo)
    for _ in range(BISECT_ROUNDS):
        mid = _split(lo, hi)
        ok = count(mid) >= keep
        lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
    return lo.reshape(*lead, 1), hi.reshape(*lead, 1)


def order_key(scores: torch.Tensor) -> torch.Tensor:
    """int32 keys in the order of the fp32 scores, key(-0.0) == key(+0.0) == 0 (IEEE 754 binary32:
    a non-negative float's bits are ordered like its value, a negative one's are the sign bit plus
    its magnitude's). The "radix" path only (an in-graph bitcast)."""
    b = scores.float().contiguous().view(torch.int32)
    return torch.where(b < 0, INT_MIN - b, b)  # negative: -(magnitude bits), in (-2^31, 0]


def kth_key(key: torch.Tensor, keep: int) -> torch.Tensor:
    """The keep-th largest int32 key of each row [..., 1]: the largest t with count(key >= t) >=
    keep, built bit by bit from bit 31 down (a binary search over t - INT_MIN in [0, 2^32) without
    int32 overflow: the candidate for bit 31 is 0, later ones add 2^b to a prefix with those bits clear)."""
    t = torch.full_like(key[..., :1], INT_MIN)
    for b in range(31, -1, -1):
        cand = torch.zeros_like(t) if b == 31 else t + (1 << b)
        t = torch.where(_count(key >= cand) >= keep, cand, t)
    return t


def _first(tied: torch.Tensor, room: torch.Tensor, ties: str) -> torch.Tensor:
    """The first `room` positions (lowest index) of each row of the boolean `tied`."""
    if ties == "cumsum":
        return tied & (tied.float().cumsum(-1) <= room)
    if ties == "block":  # the running count as a cumsum within blocks of 128 plus the blocks' prefix
        *lead, n = tied.shape
        b = 128 if n >= 128 else n
        pad = -n % b
        t = tied.float()
        if pad:
            t = torch.cat([t, t.new_zeros(*lead, pad)], dim=-1)
        c = t.reshape(*lead, -1, b).cumsum(-1)  # [..., blocks, b]
        tot = c[..., -1]
        run = (c + (tot.cumsum(-1) - tot).unsqueeze(-1)).reshape(*lead, -1)[..., :n]
        return tied & (run <= room)
    N = tied.shape[-1]
    j = torch.arange(N, dtype=torch.int32, device=tied.device)
    lim = torch.zeros(*tied.shape[:-1], 1, dtype=torch.int32, device=tied.device)
    for b in range(N.bit_length() - 1, -1, -1):  # largest lim with count(tied & j < lim) < room
        cand = lim + (1 << b)
        lim = torch.where(_count(tied & (j < cand)) < room, cand, lim)
    return tied & (j <= lim)


def topk_mask(scores: torch.Tensor, keep: int, select: str | None = None, ties: str | None = None) -> torch.Tensor:
    """Boolean [..., N]: the `keep` largest scores of each row under the module's tie rule
    ("bisect", "radix"), or torch.topk's choice ("topk")."""
    select = select or SELECT
    ties = ties or TIES
    if ties == "auto":  # measured per batch form (module docstring)
        ties = "block" if scores.dim() == 3 and scores.shape[0] == 1 and scores.shape[1] > 1 else "index"
    if keep >= scores.shape[-1]:
        return torch.ones_like(scores, dtype=torch.bool)
    if select == "topk":
        top = torch.topk(scores, keep, dim=-1).indices
        return torch.zeros_like(scores, dtype=torch.bool).scatter(-1, top, True)
    if select == "nki":
        from ..kernels import dsa_topk

        return dsa_topk.select(scores, keep, vis_only=False) == 0
    if select == "bisect":
        s = scores.float()
        lo, hi = kth_value(s, keep)
        above = (s >= hi) & (hi > lo)  # lo == hi: the max itself is the threshold
        tied = (s >= lo) & ~above
    elif select == "radix":
        key = order_key(scores)
        t = kth_key(key, keep)
        above, tied = key > t, key == t
    else:
        raise ValueError(f"KILN_DSA_SELECT must be bisect, radix, topk or nki, not {select!r}")
    return above | _first(tied, keep - _count(above), ties)


def reference_mask(scores: torch.Tensor, keep: int) -> torch.Tensor:
    """The tie rule spelled out with a stable sort (tests): descending score, then ascending index."""
    if keep >= scores.shape[-1]:
        return torch.ones_like(scores, dtype=torch.bool)
    order = torch.sort(scores.float(), dim=-1, descending=True, stable=True).indices[..., :keep]
    return torch.zeros_like(scores, dtype=torch.bool).scatter(-1, order, True)
