"""Radix tree over token ids that maps shared prefixes to KV cache pages.

The idea is SGLang's RadixAttention (https://arxiv.org/abs/2312.07104), at page
granularity: every edge key is a whole number of pages, so a node owns exactly
`len(key) // page_size` pages and only FULL pages are ever shared. A sequence's partial
last page stays private to it until it fills.

Ownership rules:
- A page referenced by a node belongs to the tree. It is freed only by eviction.
- A running request holds a lock on the deepest node of its cached prefix, which pins
  every page on the path from the root to that node.
- Eviction frees whole unlocked leaves in least-recently-used order.
- Session references (SGLang v0.5.17 "session-reference-aware radix cache", sgl-project/sglang
  #29173): a finished request that carries a session_id registers its prefix under that
  session; eviction takes unreferenced prefixes first, then fewer-referenced ones, each in the
  base policy's order. Referenced KV is soft-protected, not pinned, and closing the session
  drops its references without freeing anything.
- State checkpoints (models with linear-attention layers, engine/state_pool.py): such a model can
  resume a prefix only where the recurrent state after it was saved, so a node may own one
  checkpoint row (`ckpt`), the state after the tokens up to the END of its key. A hit is the
  deepest node on the matched path that has one (match_checkpoint; SGLang v0.5.21
  mem_cache/unified_cache/components/mamba.py consumes only best_match_node's state the same way,
  and vLLM v0.30.0 v1/core/single_type_kv_cache_manager.py MambaManager.find_longest_cache_hit
  takes the rightmost block with a cached state). A split leaves the checkpoint on the lower
  node (it ends where the state was taken; SGLang redistribute_on_node_split); evicting a node
  frees its checkpoint; checkpoints alone can be evicted under checkpoint-pool pressure
  (evict_checkpoints) while the KV stays, as SGLang's separate mamba LRU does.
"""

from __future__ import annotations

import heapq
import itertools
from collections import OrderedDict
from dataclasses import dataclass, field

from .kv_pool import PagePool


@dataclass(eq=False)
class RadixNode:
    key: list[int] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    parent: "RadixNode | None" = None
    children: dict[tuple[int, ...], "RadixNode"] = field(default_factory=dict)
    lock_ref: int = 0
    last_access: int = 0
    hit_count: int = 0
    # Best (lowest) priority value of any request that inserted or hit this node.
    priority: int = 1 << 30
    # Session registrations whose path runs through this node, and those ending here.
    session_refs: int = 0
    sessions: set = field(default_factory=set)
    # tlru: tokens already evicted below this node along its deepest evicted branch.
    tail_evicted: int = 0
    # Recurrent-state checkpoint row holding the state after this node's last token, if any.
    ckpt: int | None = None

    def is_leaf(self) -> bool:
        return not self.children


@dataclass
class MatchResult:
    pages: list[int]
    node: RadixNode

    @property
    def num_pages(self) -> int:
        return len(self.pages)


EVICTION_POLICIES = ("lru", "lfu", "slru", "priority", "tlru")
POLICY_PARAMS = {"slru": {"protected_threshold": 2}, "tlru": {"threshold": None, "next_prompt_estimate": None}}


class RadixCache:
    """`eviction_policy` follows SGLang's names (docs/advanced_features/radix_eviction_policy):
    lru evicts the prefix unused for longest; lfu the one with the fewest hits, then the
    least recent; slru everything still probationary (fewer than protected_threshold hits)
    before anything protected, least recent first within a segment; priority the one whose
    best request priority is worst, then the least recent; tlru (Tail-Optimized LRU, arXiv
    2510.15152) the "TEL-safe" tail of each cached sequence first: up to threshold -
    next_prompt_estimate tokens from its end, which the next turn can recompute and still
    prefill at most `threshold` uncached tokens; then plain recency. `policy_config` holds the
    policy's own keys (SGLang --radix-eviction-policy-config); unknown keys are errors. The
    ordering key is read at eviction time."""

    def __init__(self, pool: PagePool, page_size: int, eviction_policy: str = "lru",
                 policy_config: dict | None = None):
        if eviction_policy not in EVICTION_POLICIES:
            raise ValueError(f"eviction_policy must be one of {EVICTION_POLICIES}")
        params = dict(POLICY_PARAMS.get(eviction_policy, {}))
        for key, val in (policy_config or {}).items():
            if key not in params:
                raise TypeError(f"{eviction_policy} eviction got an unexpected keyword argument {key!r}")
            params[key] = val
        if eviction_policy == "tlru":
            th, est = params["threshold"], params["next_prompt_estimate"]
            if th is None or est is None or not 0 <= est < th:
                raise ValueError("tlru needs threshold > next_prompt_estimate >= 0 (tokens)")
            self.tel_budget = int(th) - int(est)
        self.policy_params = params
        self.pool = pool
        self.page_size = page_size
        self.eviction_policy = eviction_policy
        self.root = RadixNode()
        self.root.lock_ref = 1  # the root is never evicted
        self._clock = itertools.count(1)
        self.evictable_pages = 0
        self.protected_pages = 0
        # Called with (token path, index of the leaf's first page, its pages, its checkpoint row or
        # None) before an evicted leaf's pages are freed: the host KV tier (engine/hicache.py)
        # keeps them.
        self.offload = None
        # Recurrent-state checkpoints: free_ckpt(row) returns a row to the state pool;
        # offload_ckpt(token path, row) may keep its state elsewhere (the host tier) first.
        self.free_ckpt = None
        self.offload_ckpt = None
        self.num_ckpts = 0
        self._session_nodes: dict[str, list[RadixNode]] = {}
        self._session_open: dict[str, int] = {}  # session id -> generation
        self._session_closed: OrderedDict[str, int] = OrderedDict()  # bounded tombstones

    # -- lookup -------------------------------------------------------------------

    def _page_key(self, ids: list[int], i: int) -> tuple[int, ...]:
        return tuple(ids[i : i + self.page_size])

    def _key(self, n: RadixNode):
        refs = (n.session_refs > 0, n.session_refs)
        policy = self.eviction_policy
        if policy == "lfu":
            return (*refs, n.hit_count, n.last_access)
        if policy == "slru":
            return (*refs, n.hit_count >= self.policy_params["protected_threshold"], n.last_access)
        if policy == "priority":
            return (*refs, -n.priority, n.last_access)
        if policy == "tlru":
            safe = n.tail_evicted + len(n.key) <= self.tel_budget
            return (*refs, not safe, n.last_access)
        return (*refs, n.last_access)

    def touch(self, node: RadixNode, priority: int) -> None:
        """Record a hit on the path to `node` by a request of `priority`."""
        while node is not self.root:
            node.hit_count += 1
            node.priority = min(node.priority, priority)
            node = node.parent  # type: ignore[assignment]

    def match_prefix(self, ids: list[int], max_pages: int | None = None) -> MatchResult:
        """Longest page-aligned cached prefix of `ids`, at most `max_pages` pages.

        The node of a partial edge match is split so that the returned node ends exactly
        at the match, which is what a caller locks.
        """
        ps = self.page_size
        limit = len(ids) // ps if max_pages is None else min(max_pages, len(ids) // ps)
        node, pages, i = self.root, [], 0
        now = next(self._clock)
        node.last_access = now
        while len(pages) < limit:
            child = node.children.get(self._page_key(ids, i))
            if child is None:
                break
            n = 0
            child_pages = len(child.pages)
            while n < child_pages and len(pages) + n < limit:
                if child.key[n * ps : (n + 1) * ps] != ids[i + n * ps : i + (n + 1) * ps]:
                    break
                n += 1
            if n < child_pages:
                child = self._split(child, n)
            child.last_access = now
            pages.extend(child.pages)
            i += n * ps
            node = child
            if n < child_pages:
                break
        return MatchResult(pages=pages, node=node)

    def _split(self, child: RadixNode, n_pages: int) -> RadixNode:
        """Split `child` after its first `n_pages` pages; return the new upper node."""
        ps = self.page_size
        upper = RadixNode(
            key=child.key[: n_pages * ps],
            pages=child.pages[:n_pages],
            parent=child.parent,
            lock_ref=child.lock_ref,
            last_access=child.last_access,
            hit_count=child.hit_count,
            priority=child.priority,
            session_refs=child.session_refs,  # registrations stay on the deeper node
        )
        parent = child.parent
        assert parent is not None
        parent.children[tuple(upper.key[:ps])] = upper
        child.key = child.key[n_pages * ps :]
        child.pages = child.pages[n_pages:]
        child.parent = upper
        upper.children[tuple(child.key[:ps])] = child
        return upper

    # -- insert -------------------------------------------------------------------

    def insert(self, ids: list[int], pages: list[int]) -> MatchResult:
        """Insert a page-aligned prefix and the pages holding its KV.

        Returns the tree's pages for the WHOLE prefix and its deepest node. Where the tree
        already held a page for some position, the returned page is the tree's; the caller
        must free its own duplicate (any position where its page differs).
        """
        ps = self.page_size
        if len(ids) != len(pages) * ps:
            raise ValueError("insert needs a page-aligned key and one page per page_size tokens")
        match = self.match_prefix(ids)
        node = match.node
        tree_pages = list(match.pages)
        n = len(tree_pages)
        if n < len(pages):
            leaf = RadixNode(
                key=list(ids[n * ps :]),
                pages=list(pages[n:]),
                parent=node,
                last_access=next(self._clock),
            )
            node.children[tuple(leaf.key[:ps])] = leaf
            node.tail_evicted = 0  # the sequence grew again: its old evicted tail is moot
            self.evictable_pages += len(leaf.pages)
            tree_pages.extend(leaf.pages)
            node = leaf
        return MatchResult(pages=tree_pages, node=node)

    # -- locking ------------------------------------------------------------------

    def lock(self, node: RadixNode) -> None:
        while node is not self.root:
            if node.lock_ref == 0:
                self.evictable_pages -= len(node.pages)
                self.protected_pages += len(node.pages)
            node.lock_ref += 1
            node = node.parent  # type: ignore[assignment]

    def unlock(self, node: RadixNode) -> None:
        while node is not self.root:
            if node.lock_ref <= 0:
                raise RuntimeError("unlock of an unlocked radix node")
            node.lock_ref -= 1
            if node.lock_ref == 0:
                self.evictable_pages += len(node.pages)
                self.protected_pages -= len(node.pages)
            node = node.parent  # type: ignore[assignment]

    # -- sessions -----------------------------------------------------------------

    MAX_SESSION_NODES = 8  # registrations kept per session (fan-out branches), oldest dropped
    MAX_CLOSED_SESSIONS = 4096

    def session_open(self, sid: str) -> int:
        """The generation a request of session `sid` arriving now belongs to. A session closed
        and then used again is a new generation, so requests in flight across the close
        cannot re-attach references to it."""
        gen = self._session_open.get(sid)
        if gen is None:
            gen = self._session_closed.pop(sid, -1) + 1
            self._session_open[sid] = gen
        return gen

    def session_close(self, sid: str) -> bool:
        gen = self._session_open.pop(sid, None)
        if gen is None:
            return False
        self._session_closed[sid] = gen
        while len(self._session_closed) > self.MAX_CLOSED_SESSIONS:
            self._session_closed.popitem(last=False)
        for node in self._session_nodes.pop(sid, []):
            node.sessions.discard(sid)
            self._add_refs(node, -1)
        return True

    def session_register(self, sid: str, gen: int, node: RadixNode) -> bool:
        """Reference the prefix ending at `node` from session `sid` (generation `gen`)."""
        if self._session_open.get(sid) != gen or node is self.root:
            return False
        self._register(sid, node)
        return True

    def _register(self, sid: str, node: RadixNode) -> None:
        nodes = self._session_nodes.setdefault(sid, [])
        for other in list(nodes):
            if self._is_ancestor(node, other):  # already covered by a deeper registration
                return
            if self._is_ancestor(other, node):  # the new prefix extends this one
                self._unregister(sid, other)
        self._add_refs(node, 1)
        node.sessions.add(sid)
        nodes.append(node)
        if len(nodes) > self.MAX_SESSION_NODES:
            self._unregister(sid, nodes[0])

    def _unregister(self, sid: str, node: RadixNode) -> None:
        self._session_nodes[sid].remove(node)
        node.sessions.discard(sid)
        self._add_refs(node, -1)

    def _is_ancestor(self, a: RadixNode, b: RadixNode) -> bool:
        """a is b or on b's path to the root."""
        while b is not None:
            if b is a:
                return True
            b = b.parent
        return False

    def _add_refs(self, node: RadixNode, d: int) -> None:
        while node is not None and node is not self.root:
            node.session_refs += d
            node = node.parent

    def session_ids(self) -> list[str]:
        return list(self._session_open)

    # -- eviction -----------------------------------------------------------------

    def evict(self, num_pages: int) -> int:
        """Free at least `num_pages` pages from unlocked LRU leaves if possible."""
        heap = [(self._key(n), id(n), n) for n in self._leaves() if n.lock_ref == 0]
        heapq.heapify(heap)
        freed = 0
        while heap and freed < num_pages:
            _, _, leaf = heapq.heappop(heap)
            if leaf.lock_ref != 0 or not leaf.is_leaf() or leaf.parent is None:
                continue
            if self.offload is not None:
                keys, n = [], leaf.parent
                while n is not None and n is not self.root:
                    keys.append(n.key)
                    n = n.parent
                prefix = [t for k in reversed(keys) for t in k]
                self.offload(prefix + leaf.key, len(prefix) // self.page_size, leaf.pages, leaf.ckpt)
            if leaf.ckpt is not None:
                self._drop_ckpt(leaf, offload=False)
            self.pool.free(leaf.pages)
            freed += len(leaf.pages)
            self.evictable_pages -= len(leaf.pages)
            parent = leaf.parent
            # A referenced leaf's sessions keep referencing what is left of their prefix.
            for sid in list(leaf.sessions):
                self._unregister(sid, leaf)
                if parent is not self.root:
                    self._register(sid, parent)
            del parent.children[tuple(leaf.key[: self.page_size])]
            parent.tail_evicted = max(parent.tail_evicted, leaf.tail_evicted + len(leaf.key))
            leaf.parent = None
            if parent is not self.root and parent.is_leaf() and parent.lock_ref == 0:
                heapq.heappush(heap, (self._key(parent), id(parent), parent))
        return freed

    # -- recurrent-state checkpoints ----------------------------------------------------

    def match_checkpoint(self, ids: list[int], max_pages: int | None = None) -> tuple[MatchResult, int]:
        """The longest cached prefix of `ids` (at most max_pages pages) whose recurrent state was
        checkpointed, ending at a node with `ckpt` (or the root: position 0, zero state), and the
        page count of the plain KV match, which may run further (a junction no checkpoint covers
        yet)."""
        m = self.match_prefix(ids, max_pages)
        node, n = m.node, m.num_pages
        while node is not self.root and node.ckpt is None:
            n -= len(node.pages)
            node = node.parent  # type: ignore[assignment]
        return MatchResult(pages=m.pages[:n], node=node), m.num_pages

    def attach_checkpoint(self, ids: list[int], n_pages: int, row: int) -> bool:
        """Make `row` the checkpoint of the prefix ids[: n_pages * page_size], whose pages must be
        in the tree. False (the caller keeps the row) when they are not, or the node has one."""
        if n_pages <= 0:
            return False
        m = self.match_prefix(ids[: n_pages * self.page_size])  # splits so a node ends there
        if m.num_pages != n_pages or m.node.ckpt is not None:
            return False
        m.node.ckpt = row
        self.num_ckpts += 1
        return True

    def evict_checkpoints(self, n: int) -> int:
        """Free up to n checkpoints of unlocked nodes, the node KV left in place, in the eviction
        policy's order. Checkpoints on a running request's path are kept (its lock covers the path,
        and a running prefix is the likeliest one to be shared again)."""
        heap, stack = [], [self.root]
        while stack:
            node = stack.pop()
            stack.extend(node.children.values())
            if node.ckpt is not None and node.lock_ref == 0:
                heap.append((self._key(node), id(node), node))
        heapq.heapify(heap)
        freed = 0
        while heap and freed < n:
            self._drop_ckpt(heapq.heappop(heap)[2], offload=True)
            freed += 1
        return freed

    def _drop_ckpt(self, node: RadixNode, offload: bool) -> None:
        if offload and self.offload_ckpt is not None:
            keys, n = [], node
            while n is not None and n is not self.root:
                keys.append(n.key)
                n = n.parent
            self.offload_ckpt([t for k in reversed(keys) for t in k], node.ckpt)
        if self.free_ckpt is not None:
            self.free_ckpt(node.ckpt)
        node.ckpt = None
        self.num_ckpts -= 1

    def _leaves(self):
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf() and node is not self.root:
                yield node
            stack.extend(node.children.values())

    def total_pages(self) -> int:
        return self.evictable_pages + self.protected_pages
