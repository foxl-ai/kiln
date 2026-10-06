"""Free list over the pages of the paged KV cache.

A page holds `page_size` consecutive token positions of one sequence, for every layer.
Page 0 is reserved and never handed out: padded tokens in a static-shape step write their
KV there, so padding can never corrupt a live sequence. They write its slots 1 .. page_size - 1
and read slot 0, which nothing writes after start-up (ModelRunner.pad_slots: padded rows that
wrote the slot they read gave different results run to run on the device).
"""

from __future__ import annotations

NULL_PAGE = 0


class PagePool:
    def __init__(self, num_pages: int):
        if num_pages < 2:
            raise ValueError("need at least one usable page besides the null page")
        self.num_pages = num_pages
        # pop() hands out low page ids first, which keeps early allocations compact.
        self._free = list(range(num_pages - 1, NULL_PAGE, -1))
        self._is_free = bytearray(num_pages)
        for p in self._free:
            self._is_free[p] = 1
        # hold (a decode engine of a disaggregated deployment, engine.py _pd_setup): freed pages stay out of the
        # free list until release_held(), which the engine calls once the step in flight when they were freed has
        # been read back. A handoff writes its pages with eager host copies that do not wait for queued calls
        # (ModelRunner._settle), and an overlapped step may still write a finished request's freed tail page.
        self.hold = False
        self._held: list[int] = []
        # ascending (a disaggregated engine, engine.py _pd_setup): a request's pages are the lowest free ids in
        # ascending order, so its slots form few contiguous runs and a handoff copies each cache in a few host
        # copies (engine/disagg.py slot_runs). The plain LIFO order hands a whole prompt's pages out in DESCENDING
        # runs: measured on the G1 decode box, ~257 one-page runs x 33 caches of eager copies per handoff, 240 ms.
        self.ascending = False

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_held(self) -> int:
        return len(self._held)

    def release_held(self) -> None:
        self._free.extend(self._held)
        self._held = []

    @property
    def num_usable(self) -> int:
        return self.num_pages - 1

    def alloc(self, n: int) -> list[int] | None:
        """Return `n` pages, or None (and allocate nothing) if fewer are free."""
        if n > len(self._free):
            return None
        if self.ascending and n > 1:
            self._free.sort(reverse=True)
        pages = [self._free.pop() for _ in range(n)]
        for p in pages:
            self._is_free[p] = 0
        return pages

    def free(self, pages: list[int]) -> None:
        for p in pages:
            if p == NULL_PAGE:
                raise ValueError("the null page is never allocated and cannot be freed")
            if self._is_free[p]:
                raise ValueError(f"double free of page {p}")
            self._is_free[p] = 1
            (self._held if self.hold else self._free).append(p)
