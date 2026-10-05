"""Suffix decoding proposals (vLLM `suffix`, engine/spec_suffix.py)."""

from kiln.engine.spec_suffix import SuffixProposer, _Trie


def test_follows_the_longest_match_in_the_request_itself():
    p = SuffixProposer(k=4, max_cached_requests=0)
    assert p.propose_for("r", [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]) == [3, 4, 1, 2]
    # The match length caps the draft (max_spec_factor 1.0): a 1-token match drafts 1 token.
    assert p.propose_for("q", [7, 8, 9, 7]) == [8]
    assert p.propose_for("z", [1, 2, 3]) == []


def test_probability_floor_and_most_frequent_continuation():
    p = SuffixProposer(k=4, max_cached_requests=0, min_token_prob=0.5)
    assert p.propose_for("a", [5, 1, 5, 2, 5, 3, 5]) == []  # 1/3 each after 5
    p = SuffixProposer(k=4, max_cached_requests=0, min_token_prob=0.1)
    assert p.propose_for("b", [5, 1, 5, 2, 5, 1, 9, 5]) == [1]  # 1 follows 5 twice, 2 once


def test_live_trie_grows_incrementally_like_a_fresh_build():
    toks = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4, 3, 3, 8, 3, 2, 7]
    inc = SuffixProposer(k=6, max_cached_requests=0)
    for n in range(1, len(toks) + 1):
        got = inc.propose_for("r", toks[:n])
        fresh = SuffixProposer(k=6, max_cached_requests=0).propose_for("r", toks[:n])
        assert got == fresh, n


def test_global_cache_proposes_previous_outputs_and_evicts_oldest():
    p = SuffixProposer(k=8, max_cached_requests=1)
    answer = [10, 11, 12, 13, 14, 15, 16, 17]
    p.finish("a", answer)
    # A new request whose context reached the same text continues it from the cache.
    assert p.propose_for("b", [99, 10, 11, 12, 13]) == [14, 15, 16, 17]
    p.finish("b", [40, 41, 42])
    assert p.propose_for("c", [99, 10, 11, 12, 13]) == []  # "a" was evicted
    ref = _Trie(24)
    ref.add([40, 41, 42])

    def dump(n):
        return {t: (c.count, dump(c)) for t, c in n.children.items()}

    assert dump(p.global_trie.root) == dump(ref.root)  # uncounting left exactly "b"
