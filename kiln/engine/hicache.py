"""Host-memory KV tier under the radix cache (SGLang HiCache's L2; vLLM's CPU KV offload).

When the radix cache evicts a page, its KV (every layer, this rank's heads) is copied to host
memory instead of being lost; a later request whose prefix continues past the device-resident
match into pages the host still holds gets them copied back into fresh device pages and
inserted into the tree, so it skips their prefill. Pages are identified by a chain hash over
page-aligned token blocks (vLLM prefix caching's block hash: a page's hash covers its own
tokens and its parent's hash), so a host entry matches only the exact prefix that produced it.

Write-through: a restored page keeps its host copy, so evicting it again costs nothing. The
host tier is LRU over its own capacity; pages being restored are pinned against it. Under
tensor parallelism rank 0 decides and every rank moves its own shard (ModelRunner.kv_save /
kv_load / kv_drop are broadcast like graph calls).

DP attention (engine/dp.py): every group has its own page pool and radix cache, so it gets its own
host tier (GroupMoves), whose copies run on that group's ranks only; the budget is per rank as without
DP attention (a rank holds its own group's pages).

Linear-attention models (engine/state_pool.py): a prefix is only resumable where its recurrent
state was checkpointed, so an evicted node's checkpoint goes to host memory too, keyed by the
chain hash of the page it ends on (SGLang v0.5.21 mem_cache/pool_host/mamba.py keeps host copies
of mamba states beside the KV the same way), and a restore brings back the pages up to the deepest
host checkpoint the prompt reaches, then that checkpoint (Scheduler._restore).
"""

from __future__ import annotations

from collections import OrderedDict


class GroupMoves:
    """One DP-attention group's view of the runner's host copies (ModelRunner.kv_save ... with the group)."""

    def __init__(self, runner, group: int):
        self.runner, self.group = runner, group

    def kv_save(self, slot: int, page: int) -> None:
        self.runner.kv_save(slot, page, self.group)

    def kv_load(self, slot: int, page: int) -> None:
        self.runner.kv_load(slot, page, self.group)

    def kv_drop(self, slot: int) -> None:
        self.runner.kv_drop(slot, self.group)

    def state_save(self, slot: int, row: int) -> None:
        self.runner.state_save(slot, row, self.group)

    def state_load(self, slot: int, row: int) -> None:
        self.runner.state_load(slot, row, self.group)

    def state_drop(self, slot: int) -> None:
        self.runner.state_drop(slot, self.group)


class HostTier:
    def __init__(self, runner, page_size: int, capacity_pages: int, capacity_ckpts: int = 0):
        if capacity_pages < 1:
            raise ValueError("the host KV tier needs room for at least one page")
        self.runner, self.ps, self.capacity = runner, page_size, capacity_pages
        self.entries: OrderedDict[int, int] = OrderedDict()  # page hash -> host slot
        self.free_slots = list(range(capacity_pages - 1, -1, -1))
        self.pinned: set[int] = set()
        self.saved = self.restored = self.dropped = 0
        # Recurrent-state checkpoints: hash of the page a checkpoint ends on -> host slot.
        self.ckpt_capacity = capacity_ckpts
        self.ckpts: OrderedDict[int, int] = OrderedDict()
        self.free_ckpt_slots = list(range(capacity_ckpts - 1, -1, -1))
        self.ckpts_saved = self.ckpts_restored = 0

    def hashes(self, ids: list[int], n_pages: int) -> list[int]:
        out, h, ps = [], 0, self.ps
        for j in range(n_pages):
            h = hash((h, tuple(ids[j * ps : (j + 1) * ps])))
            out.append(h)
        return out

    def offload(self, path: list[int], first_page: int, pages: list[int], ckpt: int | None = None) -> None:
        """Keep the KV of `pages`, which hold the path's pages first_page, first_page + 1, ...,
        and `ckpt`, the state checkpoint row after the whole path, if the node had one."""
        hs = self.hashes(path, first_page + len(pages))[first_page:]
        for h, page in zip(hs, pages):
            if h in self.entries:  # write-through: still held from an earlier eviction
                self.entries.move_to_end(h)
                continue
            slot = self._slot()
            if slot is None:
                return
            self.runner.kv_save(slot, page)
            self.entries[h] = slot
            self.saved += 1
        if ckpt is not None:
            self.offload_ckpt(path, ckpt)

    def offload_ckpt(self, path: list[int], row: int) -> None:
        """Keep the state checkpoint `row`, the state after the page-aligned `path`."""
        if not self.ckpt_capacity or not path:
            return
        h = self.hashes(path, len(path) // self.ps)[-1]
        if h in self.ckpts:
            self.ckpts.move_to_end(h)
            return
        if self.free_ckpt_slots:
            slot = self.free_ckpt_slots.pop()
        else:  # least recently used first
            _, slot = self.ckpts.popitem(last=False)
            self.runner.state_drop(slot)
        self.runner.state_save(slot, row)
        self.ckpts[h] = slot
        self.ckpts_saved += 1

    def find_ckpt(self, ids: list[int], lo: int, hi: int) -> tuple[int, int] | None:
        """The deepest page count j in [lo, hi] whose prefix ids[: j * page_size] has a host
        checkpoint, and its slot."""
        if hi < max(lo, 1):
            return None
        hs = self.hashes(ids, hi)
        for j in range(hi, max(lo, 1) - 1, -1):
            slot = self.ckpts.get(hs[j - 1])
            if slot is not None:
                self.ckpts.move_to_end(hs[j - 1])
                return j, slot
        return None

    def load_ckpt(self, slot: int, row: int) -> None:
        self.runner.state_load(slot, row)
        self.ckpts_restored += 1

    def _slot(self) -> int | None:
        if self.free_slots:
            return self.free_slots.pop()
        for h, slot in self.entries.items():  # least recently used first
            if slot not in self.pinned:
                del self.entries[h]
                self.runner.kv_drop(slot)
                self.dropped += 1
                return slot
        return None

    def lookup(self, ids: list[int], start_page: int, limit: int) -> list[int]:
        """Host slots of the pages start_page .. that continue a device match, up to page
        `limit` (exclusive), pinned until load() or unpin()."""
        if start_page >= limit:
            return []
        slots = []
        for h in self.hashes(ids, limit)[start_page:]:
            slot = self.entries.get(h)
            if slot is None:
                break
            self.entries.move_to_end(h)
            slots.append(slot)
        self.pinned.update(slots)
        return slots

    def peek(self, ids: list[int], start_page: int, limit: int) -> int:
        """How many pages from start_page on (up to `limit`, exclusive) the tier holds, without pinning
        or touching them (DP-attention placement, engine/dp.py DPScheduler.held)."""
        n = 0
        if start_page < limit:
            for h in self.hashes(ids, limit)[start_page:]:
                if h not in self.entries:
                    break
                n += 1
        return n

    def load(self, slots: list[int], pages: list[int]) -> None:
        for slot, page in zip(slots, pages):
            self.runner.kv_load(slot, page)
        self.restored += len(slots)
        self.unpin(slots)

    def clear(self) -> None:
        for slot in self.entries.values():
            self.runner.kv_drop(slot)
        self.entries.clear()
        self.pinned.clear()
        self.free_slots = list(range(self.capacity - 1, -1, -1))
        for slot in self.ckpts.values():
            self.runner.state_drop(slot)
        self.ckpts.clear()
        self.free_ckpt_slots = list(range(self.ckpt_capacity - 1, -1, -1))

    def unpin(self, slots: list[int]) -> None:
        self.pinned.difference_update(slots)
