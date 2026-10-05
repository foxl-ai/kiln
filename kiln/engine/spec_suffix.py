"""Suffix decoding drafts for speculative decoding.

vLLM's `suffix` method (docs/features/speculative_decoding/suffix.md, which wraps Snowflake's
Arctic Inference) after Oliaro et al., "SuffixDecoding" (https://arxiv.org/abs/2411.04975).
No draft model. Two count-annotated suffix tries of bounded depth:

- a per-request trie over the request's own prompt and output, and
- a global trie over the outputs of the last `max_cached_requests` finished requests,
  which is what helps agentic loops and RL rollouts that keep producing similar text.

To propose, the longest suffixes of the current context that occur in a trie are extended
greedily along the most frequent continuation while the running product of estimated
token probabilities stays at or above `min_token_prob`, for at most `max_spec_factor` times
the match length (and the trie depth). The higher-scoring of the two tries' proposals wins,
the score being the expected number of accepted tokens (the sum of the running products).
Unlike n-gram, the draft length adapts per request and per step. The parameter names are
vLLM's (suffix_decoding_max_tree_depth 24, _max_cached_requests, _max_spec_factor 1.0,
_min_token_prob 0.1); the cache default is 1000 rather than vLLM's 10000 because these tries
are Python objects (about 100 bytes per node, up to depth nodes per cached token) where
Arctic Inference's are C++.
"""

from __future__ import annotations

from collections import OrderedDict


class _Node:
    __slots__ = ("count", "children")

    def __init__(self) -> None:
        self.count = 0
        self.children: dict[int, _Node] = {}


class _Trie:
    """Every substring of length <= depth of the inserted sequences, with occurrence counts."""

    def __init__(self, depth: int):
        self.depth = depth
        self.root = _Node()

    def add(self, tokens: list[int], sign: int = 1) -> None:
        """Count (sign=1) or uncount (sign=-1) every bounded substring of `tokens`."""
        for i in range(len(tokens)):
            node = self.root
            for t in tokens[i : i + self.depth]:
                child = node.children.get(t)
                if child is None:
                    if sign < 0:
                        break
                    child = node.children[t] = _Node()
                child.count += sign
                if child.count == 0:
                    del node.children[t]
                    break
                node = child

    def walk(self, tokens) -> _Node | None:
        node = self.root
        for t in tokens:
            node = node.children.get(t)
            if node is None:
                return None
        return node


class _Live:
    """A trie that grows with one request: `active[m]` is the node of its last m + 1 tokens."""

    def __init__(self, depth: int):
        self.trie = _Trie(depth)
        self.active: list[_Node] = []
        self.n = 0  # tokens consumed

    def extend(self, tokens: list[int]) -> None:
        root, depth = self.trie.root, self.trie.depth
        for t in tokens[self.n :]:
            new = []
            for parent in [root] + self.active[: depth - 1]:
                child = parent.children.get(t)
                if child is None:
                    child = parent.children[t] = _Node()
                child.count += 1
                new.append(child)
            self.active = new
            self.n += 1


def _extend(node: _Node, budget: int, min_prob: float):
    """Greedy most-frequent path below `node`: (tokens, score). Probabilities are normalised
    by the continuations actually seen: in a live trie the context's own latest occurrence
    is counted in node.count but has no continuation yet."""
    out, score, prob = [], 0.0, 1.0
    while len(out) < budget and node.children:
        total = sum(c.count for c in node.children.values())
        tok, child = max(node.children.items(), key=lambda kv: kv[1].count)
        prob *= child.count / total
        if prob < min_prob:
            break
        out.append(tok)
        score += prob
        node = child
    return out, score


class SuffixProposer:
    def __init__(self, k: int, max_tree_depth: int = 24, max_cached_requests: int = 1000,
                 max_spec_factor: float = 1.0, min_token_prob: float = 0.1):
        if max_tree_depth < 2:
            raise ValueError("max_tree_depth must be at least 2")
        self.k, self.depth = k, max_tree_depth
        self.max_cached, self.factor, self.min_prob = max_cached_requests, max_spec_factor, min_token_prob
        self.global_trie = _Trie(max_tree_depth)
        self._cached: OrderedDict[str, list[int]] = OrderedDict()
        self._live: dict[str, _Live] = {}

    def propose_for(self, rid: str, tokens: list[int], k: int | None = None) -> list[int]:
        k = self.k if k is None else k
        if k <= 0 or not tokens:
            return []
        live = self._live.get(rid)
        if live is None or live.n > len(tokens):
            live = self._live[rid] = _Live(self.depth)
        live.extend(tokens)
        best, best_score = [], 0.0
        # Per-request trie: active[m] already is the node of the last m + 1 tokens.
        for m, node in enumerate(live.active):
            match = m + 1
            budget = min(k, int(self.factor * match), self.depth - match)
            if budget <= 0 or not node.children:
                continue
            draft, score = _extend(node, budget, self.min_prob)
            if score > best_score:
                best, best_score = draft, score
        if self.max_cached:
            for match in range(min(self.depth - 1, len(tokens)), 0, -1):
                node = self.global_trie.walk(tokens[-match:])
                if node is None or not node.children:
                    continue
                budget = min(k, int(self.factor * match), self.depth - match)
                if budget <= 0:
                    continue
                draft, score = _extend(node, budget, self.min_prob)
                if score > best_score:
                    best, best_score = draft, score
        return best

    def finish(self, rid: str, output: list[int]) -> None:
        """A request ended: drop its live trie and cache its output in the global trie."""
        self._live.pop(rid, None)
        if not self.max_cached or not output:
            return
        if rid in self._cached:
            self.global_trie.add(self._cached.pop(rid), -1)
        self._cached[rid] = list(output)
        self.global_trie.add(output)
        while len(self._cached) > self.max_cached:
            _, old = self._cached.popitem(last=False)
            self.global_trie.add(old, -1)
