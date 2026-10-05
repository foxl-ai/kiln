"""N-gram (prompt lookup) drafts for speculative decoding.

vLLM's `ngram` method and SGLang's `NGRAM`: no draft model. Find the longest recent suffix
of the sequence, of length max_n down to min_n, that occurred earlier, and propose the k
tokens that followed its most recent earlier occurrence. Cheap enough to run on the host
every step, and strong on inputs that repeat themselves (code edits, extraction, RAG).
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class NgramProposer:
    def __init__(self, k: int, min_n: int = 2, max_n: int = 4, window: int = 4096):
        if not 1 <= min_n <= max_n:
            raise ValueError("need 1 <= min_n <= max_n")
        self.k, self.min_n, self.max_n, self.window = k, min_n, max_n, window

    def propose(self, tokens: list[int], k: int | None = None) -> list[int]:
        k = self.k if k is None else k
        if k <= 0:
            return []
        arr = np.asarray(tokens[-self.window:], dtype=np.int64)
        L = arr.shape[0]
        for n in range(min(self.max_n, L - 1), self.min_n - 1, -1):
            suffix = arr[L - n:]
            # Windows that end before the suffix itself, so a match is always followed by
            # at least one known token.
            wins = sliding_window_view(arr[: L - 1], n)
            hits = np.flatnonzero((wins == suffix).all(axis=1))
            hits = hits[hits + n < L]
            if hits.size:
                start = int(hits[-1]) + n
                return arr[start : start + k].tolist()
        return []
