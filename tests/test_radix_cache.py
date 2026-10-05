from kiln.engine.kv_pool import NULL_PAGE, PagePool
from kiln.engine.radix_cache import RadixCache

PS = 4


def make(num_pages=64):
    pool = PagePool(num_pages)
    return pool, RadixCache(pool, PS)


def test_pool_never_hands_out_null_page_and_rejects_double_free():
    pool = PagePool(8)
    pages = pool.alloc(7)
    assert pages is not None and NULL_PAGE not in pages and len(set(pages)) == 7
    assert pool.alloc(1) is None
    pool.free(pages[:1])
    try:
        pool.free(pages[:1])
    except ValueError:
        pass
    else:
        raise AssertionError("double free accepted")


def test_match_is_page_aligned_and_splits_on_partial_edge():
    pool, rc = make()
    ids = list(range(12))  # 3 pages
    pages = pool.alloc(3)
    rc.insert(ids, pages)
    # Shares the first page and a half: only one FULL page matches.
    m = rc.match_prefix([0, 1, 2, 3, 4, 5, 99, 99, 99])
    assert m.pages == pages[:1]
    # The edge was split at the match, so the node ends exactly there.
    assert m.node.pages == pages[:1]
    assert rc.match_prefix(ids).pages == pages
    assert rc.total_pages() == 3


def test_insert_returns_tree_pages_for_an_existing_prefix():
    pool, rc = make()
    ids = list(range(8))
    first = pool.alloc(2)
    rc.insert(ids, first)
    dup = pool.alloc(3)
    res = rc.insert(ids + [8, 9, 10, 11], dup)
    assert res.pages[:2] == first and res.pages[2] == dup[2]


def test_max_pages_limits_the_match():
    pool, rc = make()
    ids = list(range(16))
    pages = pool.alloc(4)
    rc.insert(ids, pages)
    assert rc.match_prefix(ids, max_pages=3).pages == pages[:3]


def test_locked_nodes_survive_eviction_and_lru_goes_first():
    pool, rc = make(num_pages=16)
    a = pool.alloc(2)
    b = pool.alloc(2)
    rc.insert([1] * 8, a)
    rc.insert([2] * 8, b)
    locked = rc.match_prefix([1] * 8).node  # also makes [1]* most recently used
    rc.lock(locked)
    free_before = pool.num_free
    assert rc.evict(10) == 2  # only b is evictable
    assert pool.num_free == free_before + 2
    assert rc.match_prefix([2] * 8).pages == []
    rc.unlock(locked)
    assert rc.evict(10) == 2
    assert rc.total_pages() == 0 and pool.num_free == pool.num_usable


def test_eviction_cascades_to_parent_once_children_are_gone():
    pool, rc = make()
    base = list(range(8))
    p = pool.alloc(3)
    rc.insert(base + [50, 51, 52, 53], p)
    q = pool.alloc(1)
    rc.insert(base + [60, 61, 62, 63], [p[0], p[1], q[0]])
    assert rc.total_pages() == 4
    assert rc.evict(100) == 4
    assert pool.num_free == pool.num_usable


def test_lfu_keeps_the_most_hit_prefix():
    pool = PagePool(64)
    rc = RadixCache(pool, PS, eviction_policy="lfu")
    hot, cold = pool.alloc(2), pool.alloc(2)
    rc.insert([1] * 8, hot)
    rc.insert([2] * 8, cold)
    for _ in range(5):
        rc.touch(rc.match_prefix([1] * 8).node, 0)
    rc.match_prefix([2] * 8)  # most recent, but hit once
    assert rc.evict(2) == 2
    assert rc.match_prefix([1] * 8).pages == hot and rc.match_prefix([2] * 8).pages == []


def test_priority_eviction_drops_the_worst_priority():
    pool = PagePool(64)
    rc = RadixCache(pool, PS, eviction_policy="priority")
    a, b = pool.alloc(2), pool.alloc(2)
    rc.insert([1] * 8, a)
    rc.touch(rc.match_prefix([1] * 8).node, 0)
    rc.insert([2] * 8, b)
    rc.touch(rc.match_prefix([2] * 8).node, 7)
    assert rc.evict(2) == 2
    assert rc.match_prefix([1] * 8).pages == a


def _all_refs_zero(rc):
    stack = [rc.root]
    while stack:
        n = stack.pop()
        assert n.session_refs == 0 and not n.sessions, (n.key, n.session_refs, n.sessions)
        stack.extend(n.children.values())


def test_session_referenced_prefix_is_evicted_last_and_released_on_close():
    """SGLang #29173 semantics: unreferenced first (whatever the base policy says), referenced
    KV still evictable when nothing else is, and close drops references without freeing."""
    pool, rc = make()
    a, b = list(range(100, 108)), list(range(200, 208))  # two 2-page prefixes
    na = rc.insert(a, pool.alloc(2)).node  # older: plain LRU would take it first
    rc.insert(b, pool.alloc(2))
    gen = rc.session_open("s")
    assert rc.session_register("s", gen, na)
    assert rc.evict(2) == 2
    assert rc.match_prefix(a).num_pages == 2 and rc.match_prefix(b).num_pages == 0
    assert rc.evict(1) == 2  # soft protection: the referenced prefix goes when nothing else is left
    assert rc.total_pages() == 0
    _all_refs_zero(rc)

    na = rc.insert(a, pool.alloc(2)).node
    rc.insert(b, pool.alloc(2))
    assert rc.session_register("s", gen, na)
    assert rc.session_close("s") and rc.total_pages() == 4  # nothing freed by the close
    _all_refs_zero(rc)
    rc.match_prefix(b)  # b is now the more recent; a is first again under LRU
    rc.evict(2)
    assert rc.match_prefix(a).num_pages == 0 and rc.match_prefix(b).num_pages == 2


def test_session_registration_is_generation_checked_and_split_safe():
    pool, rc = make()
    ids = list(range(12))
    old = rc.session_open("s")
    rc.session_close("s")
    new = rc.session_open("s")  # reopened by a later request
    assert new == old + 1
    pages = pool.alloc(3)
    node = rc.insert(ids, pages).node
    assert not rc.session_register("s", old, node)  # in flight across the close: stale
    assert rc.session_register("s", new, node)
    # A later turn extends the prefix: one registration, at the deeper node.
    deeper = rc.insert(ids + list(range(12, 16)), pages + pool.alloc(1)).node
    assert rc.session_register("s", new, deeper)
    assert rc._session_nodes["s"] == [deeper] and node.session_refs == 1
    # Splitting a referenced edge keeps the count on both halves.
    m = rc.match_prefix(ids[:4] + [999] * 4)
    assert m.node.session_refs == 1 and m.node.sessions == set()
    # Evicting the referenced leaf moves the registration to what is left of the prefix.
    pool2_free = pool.num_free
    rc.evict(1)
    assert pool.num_free == pool2_free + 1 and rc._session_nodes["s"] == [node]
    assert node.sessions == {"s"} and node.session_refs == 1
    rc.session_close("s")
    _all_refs_zero(rc)


def test_slru_evicts_every_probationary_prefix_before_a_protected_one():
    pool = PagePool(64)
    rc = RadixCache(pool, PS, eviction_policy="slru", policy_config={"protected_threshold": 2})
    hot, warm, new = pool.alloc(1), pool.alloc(1), pool.alloc(1)
    rc.insert([1] * 4, hot)
    rc.insert([2] * 4, warm)
    for _ in range(2):  # promoted: 2 hits
        rc.touch(rc.match_prefix([1] * 4).node, 0)
    rc.touch(rc.match_prefix([2] * 4).node, 0)  # 1 hit: still probationary
    rc.insert([3] * 4, new)  # newest of all
    assert rc.evict(2) == 2  # both probationary ones, whatever their recency
    assert rc.match_prefix([1] * 4).pages == hot
    assert rc.match_prefix([2] * 4).pages == [] and rc.match_prefix([3] * 4).pages == []


def _cached(rc, ids):
    """Pages of `ids` in the tree, without touching recency (match_prefix would)."""
    node, n, i = rc.root, 0, 0
    while True:
        child = node.children.get(tuple(ids[i : i + PS]))
        if child is None or child.key != ids[i : i + len(child.key)]:
            return n
        n, i, node = n + len(child.pages), i + len(child.key), child


def test_tlru_drops_tel_safe_tails_first_and_rejects_bad_configs():
    import pytest

    pool = PagePool(64)
    # Budget: threshold 12 - next prompt 4 = 8 tokens (2 pages) of each sequence's tail.
    rc = RadixCache(pool, PS, eviction_policy="tlru", policy_config={"threshold": 12, "next_prompt_estimate": 4})
    old = list(range(500, 512))  # 3 pages, the OLDEST entry: plain LRU would evict it first
    rc.insert(old, pool.alloc(3))
    convo = list(range(100, 120))  # a newer 5-page conversation
    pages = pool.alloc(5)
    rc.insert(convo[:12], pages[:3])
    rc.insert(convo, pages)  # its newest turn: a 2-page leaf
    assert rc.evict(2) == 2  # the conversation's TEL-safe tail goes first, not the older prefix
    assert _cached(rc, convo) == 3 and _cached(rc, old) == 3
    assert rc.evict(1) == 3  # nothing TEL-safe is left: plain recency, so the older prefix goes
    assert _cached(rc, convo) == 3 and _cached(rc, old) == 0
    with pytest.raises(ValueError):
        RadixCache(pool, PS, eviction_policy="tlru", policy_config={"threshold": 4, "next_prompt_estimate": 4})
    with pytest.raises(TypeError):
        RadixCache(pool, PS, eviction_policy="slru", policy_config={"protected_treshold": 4})
