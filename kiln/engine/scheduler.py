"""Continuous-batching scheduler over a paged KV pool with a radix prefix cache.

Every step runs all decoding sequences (one query token each) and, within a token budget,
chunks of prefill. Decodes never pause for a prefill, which bounds inter-token latency by
the chunk size; the managed Neuron scheduler instead runs one prefill per step and pauses
decodes during it (vllm-neuron docs, "neuron-scheduler", see DESIGN.md section 3).

Admission does not reserve `max_model_len` of KV per request. When the pool runs dry the
youngest running request is preempted; its computed prefix goes into the radix cache, so
the recompute is usually a cache hit.

Models with linear-attention layers (recurrent=True) resume a prefix only from a state
checkpoint (engine/state_pool.py, RadixNode.ckpt), so their prefill chunks END where a checkpoint
is to be taken: the radix junction past the last checkpoint of a request's matched path (another
request shares those tokens: SGLang v0.5.21 unified_cache/components/mamba.py
mamba_branching_seqlen, vLLM v0.30.0 Scheduler._mamba_block_aligned_split's shared-prefix
junction), optionally the last page boundary of the prompt (vLLM's "replay boundary") and every
ckpt_interval tokens (vLLM prefix_cache_retention_interval); decode saves one every ckpt_track
tokens (SGLang mamba_track_interval), so a finished turn can be resumed by the next.

Junctions are also taken AHEAD (ckpt_lookahead): a junction exists only once a second request has
matched the first one's KV, so that second request recomputes the whole shared prefix just to leave
the checkpoint, and only a third one hits. When the requests sharing a prefix are already queued or
running together (a burst sharing a system prompt, a closed-loop benchmark), the junction is known
before either computes it: the longest page-aligned prefix a request shares with another request of
its group, waiting or running, becomes a checkpoint target of the one computing it first (at its
admission, or, when the other arrives while it is still prefilling, added to its targets then). So
the second request already resumes from it.
"""

from __future__ import annotations

import bisect
import time
from collections import deque
from dataclasses import dataclass, field

from .kv_pool import PagePool
from .radix_cache import RadixCache
from .request import Request, Status


POLICIES = ("fcfs", "lpm", "spf", "priority")
PENDING = -1  # placeholder for a token sampled by a step whose results are not read yet
LPM_MAX_QUEUE = 128  # SGLang falls back to FCFS above this queue length; so do we


# A blind speculative draft (engine/spec_async.py): the drafts are on the device, not known to the host.
BLIND = -2


class NeedSync(Exception):
    """Scheduling needs a preemption while a step is in flight. A preempted request must be
    recomputed from REAL tokens, so the caller commits the in-flight step and retries."""


class QueueFull(Exception):
    """Admission control refused a request (vLLM's max_num_queued_reqs / _tokens)."""


@dataclass
class SchedulerConfig:
    page_size: int
    max_num_seqs: int
    max_prefill_tokens: int  # prefill tokens per step, across all chunks
    max_model_len: int
    eos_token_ids: tuple[int, ...] = ()
    # fcfs: arrival order. lpm: longest cached prefix first (SGLang). spf: shortest
    # uncached prefill first (SGLang 0.5.21). priority: lowest priority value first, and
    # preemption takes the worst priority first (vLLM).
    policy: str = "fcfs"
    max_num_queued_reqs: int | None = None
    max_num_queued_tokens: int | None = None
    # False: requests never reuse or publish cached pages; every page is freed when a request ends.
    prefix_cache: bool = True
    # Linear-attention models: a cached prefix is usable only up to its deepest state checkpoint.
    recurrent: bool = False
    ckpt_interval: int = 0  # tokens between periodic prefill checkpoints (0: none), a page multiple
    ckpt_track: int = 0  # tokens between decode checkpoints (0: none), a page multiple
    ckpt_prompt: bool = False  # also checkpoint every prefill's last page boundary
    # "reserve": admit a waiting request only if the free plus evictable pages cover its whole prompt
    # (and its own decode reserve) on top of what the running requests still need: the rest of their
    # prompts plus up to admission_decode_tokens of decode each (_still_needs). A KV-short pool then runs
    # fewer requests instead of admitting prompts it must later preempt (each preemption of a
    # linear-attention model recomputes from its last state checkpoint): measured on GLM-5.3-Flash at
    # 8448-token contexts, docs/neuron-notes.md "Admission". "eager": admit whenever the first chunk
    # fits (vLLM's v1 scheduler without a watermark; the default before 2026-10-04).
    admission: str = "reserve"
    admission_decode_tokens: int = 256
    # Recurrent models: checkpoint the prefix a request shares with another queued or running request
    # of its group as soon as either computes it (module docstring), instead of one request later.
    ckpt_lookahead: bool = True


def shared_pages(a: list[int], b: list[int], ps: int, limit: int) -> int:
    """Whole pages a and b share from position 0, at most `limit`."""
    n = min(limit, len(a) // ps, len(b) // ps)
    if n <= 0 or a[:ps] != b[:ps]:
        return 0
    if a[: n * ps] == b[: n * ps]:
        return n
    lo, hi = 1, n  # a[:lo pages] equal, a[:hi pages] not
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[lo * ps : mid * ps] == b[lo * ps : mid * ps]:
            lo = mid
        else:
            hi = mid
    return lo


@dataclass
class ScheduledSeq:
    req: Request
    start: int  # first position computed this step
    num_tokens: int
    sample: bool  # this chunk reaches the newest token, so it produces the next one
    # Speculative draft tokens verified after the newest token (decode only). The step then
    # covers positions start .. start + len(draft) and may emit up to len(draft) + 1 tokens.
    draft: list[int] = field(default_factory=list)
    # Recurrent-state checkpoint rows (engine/state_pool.py): copied into the request's state row
    # before this entry runs (a prefix hit), and receiving a copy of it after (the state at end).
    restore: int | None = None
    save: int | None = None

    @property
    def end(self) -> int:
        return self.start + self.num_tokens


@dataclass
class StepPlan:
    decodes: list[ScheduledSeq] = field(default_factory=list)
    prefills: list[ScheduledSeq] = field(default_factory=list)
    deferred: int = 0  # prefill chunks planned and then deferred to a later step (engine/dp.py packing)
    # Decode side of a disaggregated deployment: transfer ids whose parts this step's launch copied onto
    # the device (engine._pd_inject_plan); their files are deleted once the step's outputs are read back.
    pd_injected: list = field(default_factory=list)

    def seqs(self) -> list[ScheduledSeq]:
        return self.decodes + self.prefills

    def __bool__(self) -> bool:
        return bool(self.decodes or self.prefills)


class Scheduler:
    def __init__(self, cfg: SchedulerConfig, pool: PagePool, radix: RadixCache, draft_fn=None):
        self.host = None  # engine/hicache.HostTier, when a host KV tier is configured
        self.states = None  # engine/state_pool.StatePool of a recurrent model (checkpoint rows)
        self.dp_group = 0  # the DP-attention group this scheduler serves (engine/dp.py)
        self._new_pending: list = []  # (request, pending checkpoint) the current plan created
        self.cfg = cfg
        # draft_fn(req) -> draft token ids for a decoding request (speculative decoding).
        self.draft_fn = draft_fn
        self.pool = pool
        self.radix = radix
        # Set by an overlapping engine while a launched step's results are unread.
        self.in_flight = False
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []  # admission order, oldest first
        self.num_preemptions = 0
        # Prefill / decode disaggregation (engine/disagg.py). prefill_only (a prefill engine): requests end
        # at their first sampled token, so a request that has one is never scheduled again (under overlap
        # it is "decoding" until the commit that ends it). prefilled (a decode engine): requests handed off
        # by a prefill engine, waiting for pages; admitted ahead of every decode of the step.
        self.prefill_only = False
        # A prefill engine under KILN_PD_TRANSPORT=nixl (engine/nixl_kv.py): a finished handoff request keeps its tail
        # pages and its radix lock (req.pd_pin) until the decode engine has read them (pd_unpin).
        self.pd_pin = False
        self.pd_pinned = 0  # requests whose state row such a pin keeps: they hold a seat (max_num_seqs) until released
        self.prefilled: deque[Request] = deque()
        # A decode engine holds the pages and state rows of requests that ended in the last commit until the step in
        # flight then is read back (PagePool.hold): `cooling` counts those requests' slots, which admission may not
        # reuse yet (engine.LLMEngine._pd_cool).
        self.cooling = 0
        self.admit_ok = True  # the engine's admission gate for handed-off requests (engine.LLMEngine._pd_gate)
        self.num_prefilled_preemptions = 0

    # -- public API ---------------------------------------------------------------

    def close_session(self, sid: str) -> bool:
        return self.radix.session_close(sid)

    def add(self, req: Request) -> None:
        if req.num_tokens >= self.cfg.max_model_len:
            raise ValueError(f"prompt of {req.num_tokens} tokens does not fit max_model_len={self.cfg.max_model_len}")
        c = self.cfg
        if c.max_num_queued_reqs is not None and len(self.waiting) >= c.max_num_queued_reqs:
            raise QueueFull(f"{len(self.waiting)} requests already queued")
        if c.max_num_queued_tokens is not None:
            queued = sum(r.num_tokens for r in self.waiting)
            if queued + req.num_tokens > c.max_num_queued_tokens:
                raise QueueFull(f"{queued} tokens already queued")
        if req.session_id is not None:
            req.session_gen = self.radix.session_open(req.session_id)
        if self._lookahead():
            self._mark_running_junctions(req)
        self.waiting.append(req)

    def _lookahead(self) -> bool:
        c = self.cfg
        return c.recurrent and c.prefix_cache and c.ckpt_lookahead and self.states is not None

    @staticmethod
    def _prompt_pages(req: Request, ps: int) -> int:
        """Pages of the prompt a checkpoint may end on: those before the newest token."""
        return (req.num_tokens - 1) // ps

    MAX_AHEAD = 4  # junction checkpoints one admission may add ahead (each costs a row and a copy)

    def _shared_with(self, req: Request, other: Request) -> int:
        """Page-aligned prompt tokens req and other share (checkpointable: before either's newest)."""
        ps = self.cfg.page_size
        return shared_pages(req.token_ids, other.token_ids, ps,
                            min(self._prompt_pages(req, ps), self._prompt_pages(other, ps))) * ps

    def _mark_running_junctions(self, req: Request) -> None:
        """A request arriving while another of the group is still prefilling a prefix they share:
        make the end of the shared pages a checkpoint target of the one computing it, if it has not
        passed it yet, so `req` resumes from it."""
        for other in self.running:
            if other.is_decoding or other.status is not Status.RUNNING:
                continue
            j = self._shared_with(req, other)
            if j > other.num_computed and j not in other.ckpt_targets \
                    and all(p != j for p, _, _ in other.ckpt_pending):
                bisect.insort(other.ckpt_targets, j)

    def _prefix_in_progress(self, req: Request) -> bool:
        """Another running request of the group is computing a prefix `req` shares with it and will
        checkpoint a position of it past what `req` could resume from now (a junction target not yet
        reached, or a saved checkpoint not yet in the tree): admitting `req` now would compute those
        tokens a second time, so it waits until it can resume from them."""
        best = None
        for other in self.running:
            if other.status is not Status.RUNNING:
                continue
            soon = [p for p, _, _ in other.ckpt_pending]
            if not other.is_decoding:
                soon += [p for p in other.ckpt_targets if p > other.num_computed]
            if not soon:
                continue
            j = self._shared_with(req, other)
            soon = [p for p in soon if 0 < p <= j]
            if not soon:
                continue
            if best is None:  # what req resumes from now (its checkpoint match)
                best = self._match(req.token_ids, self._prompt_pages(req, self.cfg.page_size))[0].num_pages \
                    * self.cfg.page_size
            if max(soon) > best:
                return True
        return False

    def _junctions_ahead(self, req: Request, cached: int) -> list[int]:
        """The page-aligned prefixes `req` shares with other waiting or running requests of its group,
        past what it resumes from: distinct lengths, the shortest (most widely shared) MAX_AHEAD."""
        out = set()
        for other in list(self.waiting) + self.running:
            if other is not req:
                j = self._shared_with(req, other)
                if j > cached:
                    out.add(j)
        return sorted(out)[: self.MAX_AHEAD]

    def _order_waiting(self) -> None:
        policy = self.cfg.policy
        if policy == "fcfs" or len(self.waiting) < 2:
            return
        if policy == "priority":
            key = lambda r: (r.priority, r.arrival_time)  # noqa: E731
        elif policy == "lpm" and len(self.waiting) <= LPM_MAX_QUEUE:
            ps = self.cfg.page_size
            hits = {id(r): self._match(r.token_ids, (r.num_tokens - 1) // ps)[0].num_pages for r in self.waiting}
            key = lambda r: (-hits[id(r)], r.arrival_time)  # noqa: E731
        elif policy == "spf":
            key = lambda r: (r.num_tokens, r.arrival_time)  # noqa: E731
        else:
            return
        self.waiting = deque(sorted(self.waiting, key=key))

    def has_work(self) -> bool:
        return bool(self.waiting or self.running or self.prefilled)

    def add_prefilled(self, req: Request) -> None:
        """A request a prefill engine handed off (engine.LLMEngine.add_prefilled): its prompt's KV and
        state are in a complete handoff, its first token is the last of token_ids. Admitted by
        _admit_prefilled."""
        if req.num_tokens > self.cfg.max_model_len:
            raise ValueError(f"handed-off request of {req.num_tokens} tokens does not fit "
                             f"max_model_len={self.cfg.max_model_len}")
        self.prefilled.append(req)

    def _admit_prefilled(self) -> None:
        """Admit handed-off requests, oldest first, while there are slots and pages: pages for every known
        token (the prompt's KV arrives into them, the first token's is written by its first decode), and
        under "reserve" admission the whole reserve the running requests and this one may still need, as
        for a waiting request. A request admitted here is running and decoding, with num_computed at the
        prompt's end; its first decode is in this step (step 1 of _schedule)."""
        ps = self.cfg.page_size
        reserve = self.cfg.admission == "reserve"
        owed = sum(self._still_needs(r) for r in self.running) if reserve else 0
        while self.prefilled and len(self.running) + self.cooling < self.cfg.max_num_seqs:
            req = self.prefilled[0]
            need = -(-req.num_tokens // ps)
            if reserve and self.running:
                full = -(-self._target_tokens(req) // ps)
                if self.pool.num_free + self.radix.evictable_pages < full + owed:
                    return
            if self.pool.num_free < need:
                self.radix.evict(need - self.pool.num_free)
            pages = self.pool.alloc(need)
            if pages is None:
                return
            self.prefilled.popleft()
            req.pages = pages
            req.node = None
            req.num_computed = req.num_prompt
            req.status = Status.RUNNING
            self.running.append(req)
            if reserve:
                owed += self._still_needs(req)

    def schedule(self) -> StepPlan:
        """The next step's work. A NeedSync aborts the plan (rollback)."""
        self._new_pending: list = []
        try:
            return self._schedule()
        except NeedSync:
            self.rollback()
            raise

    def rollback(self) -> None:
        """Undo the checkpoint bookkeeping of the last plan, which will not run (a NeedSync here, or
        in a later DP-attention group, engine/dp.py): the rows taken for its checkpoint copies go
        back and a prefill's checkpoint target is wanted again. A request admitted from a checkpoint
        keeps req.ckpt_restore, so the retry's first chunk restores it."""
        for req, item in self._new_pending:
            if item in req.ckpt_pending:
                req.ckpt_pending.remove(item)
                self.states.free_ckpt(item[1], self.dp_group)
                if not item[2]:
                    bisect.insort(req.ckpt_targets, item[0])
        self._new_pending = []

    def defer(self, entries: list[ScheduledSeq]) -> None:
        """Take planned prefill entries back out of the step (DP-attention packing, engine/dp.py): each must
        be the last of its request's entries in the plan (a suffix of them is deferred together). Their
        requests stay running where they are and continue from their num_computed next step; a checkpoint
        an entry would have saved is wanted again and its row goes back; a restore not launched yet stays on
        the request (req.ckpt_restore), so its next first chunk does it."""
        for e in entries:
            if e.save is None:
                continue
            req = e.req
            item = next((x for x in req.ckpt_pending if x[1] == e.save and not x[2]), None)
            if item is not None:
                req.ckpt_pending.remove(item)
                self._new_pending = [(r, x) for r, x in self._new_pending if x is not item]
                self.states.free_ckpt(item[1], self.dp_group)
                bisect.insort(req.ckpt_targets, item[0])

    def _schedule(self) -> StepPlan:
        plan = StepPlan()
        ps = self.cfg.page_size
        if self.prefilled and self.admit_ok:
            self._admit_prefilled()

        # 1. Every decoding sequence gets its one token, preempting the youngest if KV
        #    runs out. Iterating oldest-first while preempting youngest-first means a
        #    victim is never a sequence already scheduled this step.
        for req in list(self.running):
            if req.status is not Status.RUNNING or not req.is_decoding or self.prefill_only:
                continue
            draft = self.draft_fn(req) if self.draft_fn is not None else []
            blind = bool(draft) and draft[0] == BLIND
            # A blind speculative row (engine/spec_async.py): the steps still in flight may each have advanced the
            # request by up to 1 + k positions, so its newest token is at most this far on, and its pages cover the
            # verify past that bound.
            start = req.num_computed + (1 + len(draft)) * req.spec_inflight if blind else req.num_computed
            if blind:  # the request finishes by max_model_len: pages past it are never needed (padding writes go
                start = min(start, self.cfg.max_model_len - 1)  # to the null page)
            while not self._reserve(req, min(start + 1 + len(draft), self.cfg.max_model_len) if blind
                                    else start + 1 + len(draft)):
                if draft and not blind:  # under pressure, drop the draft before preempting anyone
                    draft = []
                    continue
                if self.in_flight:
                    raise NeedSync()
                victim = self._youngest_running()
                self._preempt(victim)
                if victim is req:
                    break
            if req.status is Status.RUNNING:
                save = None
                if not draft and self.cfg.recurrent and self.cfg.ckpt_track and self.cfg.prefix_cache:
                    save = self.track(req, req.num_computed, 1)
                plan.decodes.append(ScheduledSeq(req, start, 1 + len(draft), True, draft, save=save))

        # 2. Continue partial prefills, oldest first.
        budget = self.cfg.max_prefill_tokens
        for req in self.running:
            if budget <= 0:
                break
            # A prompt whose chunks left exactly its last token "decodes" it; a prefill engine has no decode
            # graph and runs it as a one-token chunk (a request that has sampled has a token past the prompt).
            if req.is_decoding and not (self.prefill_only and req.num_tokens == req.num_prompt):
                continue
            n = min(req.remaining_prefill, budget)
            if not self._reserve(req, req.num_computed + n):
                break
            plan.prefills.extend(self._chunks(req, req.num_computed, n))
            budget -= n

        # 3. Admit waiting requests while budget and capacity remain.
        self._order_waiting()
        reserve = self.cfg.admission == "reserve"
        owed = sum(self._still_needs(r) for r in self.running) if reserve else 0
        deferred = []  # waiting for a running request to checkpoint the prefix they share
        lookahead = self._lookahead()
        while self.waiting and budget > 0 and len(self.running) + self.pd_pinned < self.cfg.max_num_seqs:
            req = self.waiting[0]
            if lookahead and self._prefix_in_progress(req):
                deferred.append(self.waiting.popleft())
                continue
            # Leave at least one token to compute: the step must produce logits. Prompt
            # logprobs need every row from the first one they score, so the cache may only
            # serve the pages before it (vLLM v1 skips the cache for these requests outright).
            limit = (req.num_tokens - 1) // ps if self.cfg.prefix_cache else 0
            for rows in (req.prompt_logprob_rows, req.forced_logprob_rows):
                if rows is not None:
                    limit = min(limit, rows[0] // ps)
            match, kv_pages = self._match(req.token_ids, limit)
            if self.host is not None:
                match, kv_pages = self._restore(req, match, kv_pages, limit)
            cached = match.num_pages * ps
            n = min(req.num_tokens - cached, budget)
            need = -(-(cached + n) // ps) - match.num_pages
            self.radix.lock(match.node)  # so eviction below cannot take this prefix
            if reserve and self.running:
                full = -(-self._target_tokens(req) // ps) - match.num_pages
                if self.pool.num_free + self.radix.evictable_pages < full + owed:
                    self.radix.unlock(match.node)
                    break
            if self.pool.num_free < need:
                self.radix.evict(need - self.pool.num_free)
            new_pages = self.pool.alloc(need)
            if new_pages is None:
                self.radix.unlock(match.node)
                break
            self.waiting.popleft()
            req.pages = list(match.pages) + new_pages
            req.node = match.node
            self.radix.touch(match.node, req.priority)
            req.num_computed = cached
            if req.num_cached_tokens < 0:
                req.num_cached_tokens = cached
            req.status = Status.RUNNING
            self.running.append(req)
            if self.cfg.recurrent:
                # None at the root: the state starts from zero at position 0. Kept on the request
                # until the engine launches the copy (a plan given up with NeedSync is rescheduled).
                req.ckpt_restore = match.node.ckpt
                if self.cfg.prefix_cache:
                    ahead = self._junctions_ahead(req, cached) if lookahead else []
                    req.ckpt_targets = self._ckpt_targets(req, cached, kv_pages * ps, ahead)
            plan.prefills.extend(self._chunks(req, cached, n))
            budget -= n
            if reserve:
                owed += self._still_needs(req)
        self.waiting.extendleft(reversed(deferred))
        return plan

    def _target_tokens(self, req: Request) -> int:
        """The tokens a request is reserved pages for: its known tokens (the prompt, and what it has
        generated) plus up to admission_decode_tokens of what it may still generate."""
        left = max(0, req.params.max_new_tokens - len(req.output_ids))
        return min(self.cfg.max_model_len, req.num_tokens + min(left, self.cfg.admission_decode_tokens))

    def _still_needs(self, req: Request) -> int:
        """Pages a running request will still allocate up to its _target_tokens."""
        return max(0, -(-self._target_tokens(req) // self.cfg.page_size) - len(req.pages))

    def _match(self, ids: list[int], limit: int):
        """(usable match, KV-matched pages): a recurrent model's hit stops at its deepest checkpoint."""
        if self.cfg.recurrent:
            return self.radix.match_checkpoint(ids, max_pages=limit)
        m = self.radix.match_prefix(ids, max_pages=limit)
        return m, m.num_pages

    # -- recurrent-state checkpoints ------------------------------------------------------

    def _ckpt_targets(self, req: Request, cached: int, kv_matched: int, ahead=()) -> list[int]:
        """Positions past `cached` where this prefill should leave a checkpoint (page multiples
        below the newest token): the radix junction (KV other requests share runs past the last
        checkpoint), the junctions with requests still queued or running (`ahead`, ckpt_lookahead),
        with ckpt_prompt the last page boundary, and every ckpt_interval tokens."""
        ps, c = self.cfg.page_size, self.cfg
        last = (req.num_tokens - 1) // ps * ps
        out = set()
        if kv_matched > cached:
            out.add(kv_matched)
        out.update(p for p in ahead if p > cached)
        if c.ckpt_prompt and last > cached:
            out.add(last)
        if c.ckpt_interval:
            out.update(range((cached // c.ckpt_interval + 1) * c.ckpt_interval, last + 1, c.ckpt_interval))
        return sorted(p for p in out if cached < p <= last)

    def _alloc_ckpt(self) -> int | None:
        row = self.states.alloc_ckpt(self.dp_group)
        if row is None and self.radix.evict_checkpoints(1):
            row = self.states.alloc_ckpt(self.dp_group)
        return row

    def _chunks(self, req: Request, start: int, n: int) -> list[ScheduledSeq]:
        """One step's prefill of positions start .. start + n as entries ending at the request's
        checkpoint targets, each such entry saving the state at its end into a fresh row (pending
        until its pages are in the tree, _cache). The step runs them back to back; the first one
        restores the request's checkpoint if it was admitted from one and that copy has not run."""
        end, out = start + n, []
        restore = req.ckpt_restore
        while start < end:
            stop, save = end, None
            while req.ckpt_targets and req.ckpt_targets[0] <= end:
                t = req.ckpt_targets.pop(0)
                if t > start and (save := self._alloc_ckpt()) is not None:  # no row: no reason to stop
                    stop = t
                    req.ckpt_pending.append((t, save, False))
                    self._new_pending.append((req, req.ckpt_pending[-1]))
                    break
            out.append(ScheduledSeq(req, start, stop - start, stop == req.num_tokens, restore=restore, save=save))
            restore, start = None, stop
        return out

    def track(self, req: Request, start: int, n: int) -> int | None:
        """Decode checkpoint: when positions start .. start + n - 1 cross a multiple b of ckpt_track,
        a fresh row for the state after b tokens (the caller copies it there), registered as
        pending at b. It supersedes the request's older decode checkpoint, which is freed: only the
        newest serves the next turn. Rows are never reused while pending, so a copy still in flight
        cannot overwrite a checkpoint the tree already holds."""
        t = self.cfg.ckpt_track
        b = (start + n) // t * t
        if b <= start:
            return None
        row = self._alloc_ckpt()
        if row is None:
            return None
        for item in [x for x in req.ckpt_pending if x[2]]:
            req.ckpt_pending.remove(item)
            self.states.free_ckpt(item[1], self.dp_group)
        req.ckpt_pending.append((b, row, True))
        self._new_pending.append((req, req.ckpt_pending[-1]))
        return row

    def _restore(self, req: Request, match, kv_pages: int, limit: int):
        """Extend a device match with pages the host tier holds: copy them into fresh device
        pages and insert them into the tree, then match again. A recurrent model resumes only from a
        checkpoint, so it restores pages up to the deepest host checkpoint past its device one
        (none: nothing) and that checkpoint into a fresh row of the state pool."""
        ids, ps = req.token_ids, self.cfg.page_size
        full = self.radix.match_prefix(ids, max_pages=kv_pages) if self.cfg.recurrent else match
        slots = self.host.lookup(ids, kv_pages, limit)
        ck = None
        if self.cfg.recurrent:
            ck = self.host.find_ckpt(ids, match.num_pages + 1, kv_pages + len(slots))
            keep = max(0, ck[0] - kv_pages) if ck is not None else 0
            self.host.unpin(slots[keep:])
            slots = slots[:keep]
            if ck is None:
                return match, kv_pages
        elif not slots:
            return match, kv_pages
        self.radix.lock(full.node)
        try:
            if slots:
                if self.pool.num_free < len(slots):
                    self.radix.evict(len(slots) - self.pool.num_free)
                pages = self.pool.alloc(len(slots))
                if pages is None:
                    self.host.unpin(slots)
                    return match, kv_pages
                self.host.load(slots, pages)
                n = kv_pages + len(slots)
                res = self.radix.insert(ids[: n * ps], list(full.pages) + pages)
                for mine, tree in zip(pages, res.pages[kv_pages:]):
                    if mine != tree:  # someone inserted the same pages meanwhile: keep the tree's
                        self.pool.free([mine])
            if ck is not None:
                row = self._alloc_ckpt()
                if row is not None:
                    self.host.load_ckpt(ck[1], row)
                    if not self.radix.attach_checkpoint(ids, ck[0], row):
                        self.states.free_ckpt(row, self.dp_group)
        finally:
            self.radix.unlock(full.node)
        return self._match(ids, limit)

    def update(self, plan: StepPlan, tokens: list) -> list[Request]:
        """Apply one step's results synchronously: advance() then commit()."""
        self.advance(plan)
        return self.commit(plan, tokens)

    def advance(self, plan: StepPlan) -> None:
        """Move every scheduled request past this step BEFORE its results are known, so the
        next step can be scheduled while this one runs (overlap scheduling). A sampling
        entry gets a PENDING placeholder where its new token will go."""
        for s in plan.seqs():
            req = s.req
            if s.draft and s.draft[0] == BLIND:  # its tokens and progress are on the device until commit
                req.spec_inflight += 1
                continue
            req.num_computed = s.end
            if s.sample:
                req.token_ids.append(PENDING)

    def commit(self, plan: StepPlan, tokens: list) -> list[Request]:
        """Fill in a step's results. `tokens[i]` is the sampled id for `plan.seqs()[i]`, or
        for a speculative entry the list it emitted (accepted drafts plus one). An entry
        whose request already finished in an earlier commit is ignored. Returns requests
        finished by this commit."""
        now = time.monotonic()
        finished = []
        for s, tok in zip(plan.seqs(), tokens):
            req = s.req
            if s.draft and s.draft[0] == BLIND:
                req.spec_inflight -= 1
                if req.status is not Status.RUNNING:
                    continue
                reason = self._commit_blind(req, tok, now)
                if reason is not None:
                    finished.append(req)
                continue
            if req.status is not Status.RUNNING:
                continue  # finished or aborted while this step was in flight
            if not s.sample:
                self._cache(req)  # share a prefill chunk's full pages right away
                continue
            at = s.start + 1 if s.draft else s.end  # index of the placeholder
            new = tok if isinstance(tok, list) else [tok]
            if s.draft:
                del req.token_ids[at:]  # placeholder -> the accepted run
            reason, n = None, 0
            for t in new:
                if s.draft:
                    req.token_ids.append(int(t))
                else:
                    req.token_ids[at] = int(t)
                n += 1
                reason = self._finish_reason(req, int(t), length=at + n)
                if reason is not None:
                    break
            if s.draft:
                # Valid KV: the newest token and the accepted drafts, one position per
                # emitted token. KV written for rejected drafts lies beyond this.
                req.num_computed = s.start + n
            if req.first_token_time is None:
                req.first_token_time = now
            if reason is not None:
                # Drop placeholders of steps launched after this one, and count only KV
                # that belongs to tokens the request keeps.
                del req.token_ids[at + n:]
                req.num_computed = min(req.num_computed, at + n - 1)
                req.finish_reason = reason
                req.finish_time = now
                finished.append(req)
                self._release(req, Status.FINISHED)
            elif s.num_tokens > 1 and not s.draft:
                self._cache(req)
        return finished

    def _commit_blind(self, req: Request, tok: list, now: float):
        """A blind speculative entry's emitted tokens (accepted drafts, then the replacement or the bonus), in
        order: appended as a verify's are, the newest one's position becoming num_computed (the KV of the newest
        token and of the accepted drafts is written; a rejected draft's lies beyond). Returns the finish reason."""
        reason = None
        for t in tok:
            req.token_ids.append(int(t))
            reason = self._finish_reason(req, int(t), length=len(req.token_ids))
            if reason is not None:
                break
        req.num_computed = len(req.token_ids) - 1
        if req.first_token_time is None:
            req.first_token_time = now
        if reason is not None:
            req.finish_reason = reason
            req.finish_time = now
            self._release(req, Status.FINISHED)
        return reason

    @staticmethod
    def shared_tokens(req: Request) -> int:
        """Leading positions of `req` whose KV pages belong to the radix tree: other requests
        may read them, so they are never rewritten in place."""
        n, node = 0, req.node
        while node is not None and node.parent is not None:
            n += len(node.key)
            node = node.parent
        return n

    def _drop_pending(self, req: Request) -> None:
        if PENDING in req.token_ids:
            del req.token_ids[req.token_ids.index(PENDING):]
            req.num_computed = min(req.num_computed, len(req.token_ids) - 1)

    def finish_now(self, req: Request, reason: str) -> None:
        """Finish a running request outside update(), e.g. on a stop string."""
        self._drop_pending(req)
        req.finish_reason = reason
        req.finish_time = time.monotonic()
        self._release(req, Status.FINISHED)

    def abort(self, req: Request) -> None:
        if req.status is Status.WAITING and req in self.prefilled:
            self.prefilled.remove(req)
            req.status = Status.FINISHED
        elif req.status is Status.WAITING:
            self.waiting.remove(req)
            req.status = Status.FINISHED
        elif req.status is Status.RUNNING:
            req.finish_reason = "abort"
            self._drop_pending(req)
            self._release(req, Status.FINISHED)
        req.finish_reason = "abort"

    # -- internals ----------------------------------------------------------------

    def _finish_reason(self, req: Request, tok: int, length: int) -> str | None:
        """`length`: the request's token count up to and including `tok` (later
        placeholders of in-flight steps do not count)."""
        p = req.params
        if not p.ignore_eos and tok in self.cfg.eos_token_ids:
            return "stop"
        if tok in p.stop_token_ids:
            return "stop"
        if length - req.num_prompt >= p.max_new_tokens:
            return "length"
        if length >= self.cfg.max_model_len:
            return "length"
        return None

    def _reserve(self, req: Request, total_tokens: int) -> bool:
        need = -(-total_tokens // self.cfg.page_size) - len(req.pages)
        if need <= 0:
            return True
        if self.pool.num_free < need:
            self.radix.evict(need - self.pool.num_free)
        pages = self.pool.alloc(need)
        if pages is None:
            return False
        req.pages.extend(pages)
        return True

    def _youngest_running(self) -> Request:
        """Preemption victim: the youngest, or under the priority policy the worst
        priority first (then the youngest)."""
        if self.cfg.policy == "priority":
            return max(self.running, key=lambda r: (r.priority, r.arrival_time))
        return max(self.running, key=lambda r: r.arrival_time)

    def _cache(self, req: Request) -> None:
        """Insert the request's full computed pages into the radix tree and move its
        lock to the new deepest node. Duplicate pages (the same prefix computed by
        another request first) are freed in favour of the tree's."""
        ps = self.cfg.page_size
        n_full = req.num_computed // ps
        if n_full == 0 or not self.cfg.prefix_cache:
            return
        res = self.radix.insert(req.token_ids[: n_full * ps], req.pages[:n_full])
        if res.node.priority > req.priority:
            self.radix.touch(res.node, req.priority)
        for i, page in enumerate(res.pages):
            if req.pages[i] != page:
                self.pool.free([req.pages[i]])
                req.pages[i] = page
        self.radix.lock(res.node)
        if req.node is not None:
            self.radix.unlock(req.node)
        req.node = res.node
        if req.ckpt_pending:  # checkpoints whose pages are now in the tree join their node
            keep = []
            for item in req.ckpt_pending:
                pos, row, _ = item
                if pos > n_full * ps:
                    keep.append(item)
                elif not self.radix.attach_checkpoint(req.token_ids, pos // ps, row):
                    self.states.free_ckpt(row, self.dp_group)  # another request's checkpoint got there first
            req.ckpt_pending = keep

    def _release(self, req: Request, status: Status) -> None:
        if self.pool.hold:
            self.cooling += 1
        self._cache(req)
        if req.handoff is not None:  # the pages its handoff reads (engine._pd_handoff), before any is reused
            req.handoff_pages = list(req.pages)
        for _, row, _ in req.ckpt_pending:  # past the KV the request keeps: never valid
            self.states.free_ckpt(row, self.dp_group)
        req.ckpt_pending, req.ckpt_targets, req.ckpt_restore = [], [], None
        if (status is Status.FINISHED and req.session_id is not None and req.node is not None
                and req.finish_reason != "abort"):
            self.radix.session_register(req.session_id, req.session_gen, req.node)
        n_full = req.num_computed // self.cfg.page_size if self.cfg.prefix_cache else 0
        tail = req.pages[n_full:]
        if self.pd_pin and req.handoff is not None and status is Status.FINISHED:
            req.pd_pin = (self, tail, req.node)  # engine._pd_handoff keeps it until the release, or frees it at once
        else:
            if tail:
                self.pool.free(tail)
            if req.node is not None:
                self.radix.unlock(req.node)
        req.pages = []
        req.node = None
        req.status = status
        self.running.remove(req)

    def pd_unpin(self, pin) -> None:
        """Free what _release kept for a nixl handoff (pd_pin): its tail pages and its radix lock."""
        _, tail, node = pin
        if tail:
            self.pool.free(tail)
        if node is not None:
            self.radix.unlock(node)

    def _preempt(self, req: Request) -> None:
        if req.pd_meta is not None:
            # A handed-off request recomputes its prompt and output with this engine's prefill graphs, which a
            # decode engine loads only with pd_bypass_prefill; "reserve" admission (the default) sizes the pool
            # so that it does not happen.
            self.num_prefilled_preemptions += 1
            print(f"kiln pd: preempting handed-off request {req.rid} ({req.num_tokens} tokens): it recomputes "
                  "with this engine's prefill graphs", flush=True)
        self._release(req, Status.WAITING)
        req.num_computed = 0
        req.spec_ready = False  # its recompute rebuilds the speculative board row (engine/spec_async.py)
        req.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(req)
