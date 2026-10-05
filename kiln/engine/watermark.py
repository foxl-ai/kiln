"""Gumbel-max text watermarking with a keyed PRF, and its detector.

vLLM v0.30's `--watermark-config '{"algorithm": "gumbel", "key": ...}'` (docs/features/
watermarking.md, vllm-project/vllm#54053) after Aaronson's scheme: every candidate token j at
a position gets a pseudorandom r_j = PRF(key, previous context_width tokens, j) in (0, 1),
and the sample is argmax_j log p_j - log(-log r_j), which is an exact sample from p (the
Gumbel-max trick) whose r of the chosen token is biased towards 1. A detector with the key
scores sum_i -ln(1 - r_i): Gamma(n, 1) for text that was not watermarked, much larger for
text that was.

Kiln specifics, all stated where they differ:
- The PRF is BLAKE2b keyed with the key (8-byte little-endian ints for the context and the
  candidate); it is not vLLM's PRF, so only this detector reads Kiln's watermark.
- Positions with fewer than context_width generated tokens before them, and positions whose
  context already occurred earlier in the generation (vLLM's default deduplicate_contexts
  "single_turn"), are sampled ordinarily and skipped by the detector, so detection needs only
  the generated token ids, not the prompt.
- The candidates are the request's sampling distribution (temperature, top-k, top-p, min-p)
  restricted to the 20 highest logits the device returns with every step, applied on the
  host; mass outside them is dropped (the device sampler's own candidate set is 64).
- Greedy requests are not watermarked, as in vLLM; watermarked requests are host-bound (no
  overlap scheduling, no speculative drafts).
"""

from __future__ import annotations

import hashlib
import math
import struct


class Watermark:
    def __init__(self, key: int, context_width: int = 4, deduplicate_contexts: str = "single_turn"):
        if context_width < 1:
            raise ValueError("context_width must be at least 1")
        if deduplicate_contexts not in ("none", "single_turn"):
            raise ValueError('deduplicate_contexts must be "none" or "single_turn"')
        self.key = struct.pack("<q", int(key)).ljust(16, b"\0")
        self.w = context_width
        self.dedup = deduplicate_contexts == "single_turn"

    def prf(self, context: tuple[int, ...], token: int) -> float:
        h = hashlib.blake2b(struct.pack(f"<{len(context) + 1}q", *context, token), digest_size=8, key=self.key)
        return (int.from_bytes(h.digest(), "little") + 0.5) / 2.0 ** 64  # in (0, 1)

    def context(self, output: list[int], seen: set) -> tuple[int, ...] | None:
        """The context of the next position of `output`, or None if it is not watermarked."""
        if len(output) < self.w:
            return None
        ctx = tuple(output[-self.w :])
        if self.dedup and ctx in seen:
            return None
        return ctx

    def choose(self, ctx: tuple[int, ...], ids: list[int], logprobs: list[float], temperature: float,
               top_p: float = 1.0, top_k: int = 0, min_p: float = 0.0) -> int:
        """Gumbel-max over the request's distribution on candidates (ids, raw logprobs sorted
        descending), with the PRF's r as the uniform."""
        scaled = [lp / temperature for lp in logprobs]
        m = max(scaled)
        probs = [math.exp(x - m) for x in scaled]
        z = sum(probs)
        probs = [p / z for p in probs]
        keep, cum = [], 0.0
        for rank, p in enumerate(probs):
            if rank and (cum >= top_p or (0 < top_k <= rank) or p < min_p * probs[0]):
                break
            keep.append(rank)
            cum += p
        best, best_score = ids[0], -math.inf
        for rank in keep:
            r = self.prf(ctx, ids[rank])
            score = math.log(probs[rank]) - math.log(-math.log(r))
            if score > best_score:
                best, best_score = ids[rank], score
        return best

    def detect(self, tokens: list[int]) -> dict:
        """Score generated token ids. p_value is P(score >= observed | not watermarked)."""
        seen: set = set()
        score, n = 0.0, 0
        for i in range(len(tokens)):
            ctx = self.context(tokens[:i], seen)
            if ctx is None:
                continue
            seen.add(ctx)
            score += -math.log(1.0 - self.prf(ctx, tokens[i]))
            n += 1
        return {"num_scored": n, "score": score, "p_value": _gamma_sf(score, n) if n else 1.0}


def _gamma_sf(x: float, n: int) -> float:
    """P(X >= x) for X ~ Gamma(n, 1) with integer n: e^-x sum_{k<n} x^k / k!, in log space."""
    if x <= 0:
        return 1.0
    terms, t = [], -x
    for k in range(n):
        terms.append(t)
        t += math.log(x) - math.log(k + 1)
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(a - m) for a in terms))
