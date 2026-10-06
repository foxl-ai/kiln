"""DP attention: one scheduler per attention group behind the Scheduler interface.

Under DP attention (EngineConfig.dp_attention = N, models/decoder.py DecoderForCausalLM: DP
attention) the token mixers of each group of tp / N ranks serve their own requests, so each group
has its own KV page pool, radix prefix cache and recurrent-state rows, and a request lives in one
group from the moment it is queued. That is SGLang's arrangement: one Scheduler per DP rank, each
with its own memory pool and radix cache, fed by the DataParallelController
(v0.5.21 srt/managers/data_parallel_controller.py), and vLLM's (one engine core per data-parallel
rank). Here the N schedulers live in rank 0's process and every step merges their plans into one
set of graph calls, because the MLP / experts of all N groups run in the same graphs on every rank.

Placement is SGLang's TOTAL_TOKENS load balancing (DataParallelController.total_tokens_scheduler
and DPBudget.dispatch: the group with the fewest tokens, ties broken by the fewest requests),
counting the tokens a group would have to COMPUTE for the request: a group whose radix cache
already holds part of the prompt is charged only for the rest, which keeps prefix hits without
piling every request sharing a system prompt onto one group. A preempted request goes back to its
own group's queue, where its cached prefix is.

Cache-aware placement beyond the radix match (the sgl-router's cache-aware policy approximates the
same with a radix tree of the prompts it routed): a prefix a group does not hold yet but will compute
for a request already queued or running there counts as held, and a group's load is what its live
requests still cost (each one's charge when it was placed: its prompt minus what that group held), so
requests sharing a system prompt that arrive together are placed where it will be computed once. A
group whose live requests already fill max_num_seqs comes after every group with a free slot: a
request queued behind a full group waits longer than recomputing its prefix elsewhere costs.

Prefill packing (`pack`, EngineConfig.dp_prefill_pack): a step makes as many prefill calls as its busiest
group has chunks, and a call costs about the same whether one group or all of them carry a chunk (measured on
GLM-5.3-Flash at tp=32: ~1.0 s per 4096-row call). "trim" takes a group's 2nd, 3rd... chunk back out of the
step when the call it would add carries fewer than `pack_min` chunks (default every group): that chunk
runs next step instead, in the one call the step makes anyway, and the step's decodes are not held up behind
a nearly empty prefill call (tools/sim_dp_pack.py: with prompt lengths spread over 1024-8192 tokens at conc
64 a step had made 1.45x the packed minimum of calls). "hold" also defers a step's only prefill call when
it carries fewer than pack_min chunks, the step has decodes to run anyway and none of its requests has sat
out `hold_steps` steps in its life already, so lone chunks wait (at most hold_steps steps per request) to
share a call. A deferred chunk's
request keeps its place, pages and state; its tokens are unchanged.
"""

from __future__ import annotations

from .request import Request
from .scheduler import NeedSync, QueueFull, Scheduler, StepPlan, shared_pages


class DPScheduler:
    """N group schedulers (engine/scheduler.py, unchanged) with the one Scheduler's interface:
    schedule() returns one StepPlan holding every group's sequences (each request carries its
    req.dp_group), and advance / commit / finish_now / abort route each request to its group."""

    PACKS = ("off", "trim", "hold")

    def __init__(self, groups: list[Scheduler], max_num_queued_reqs: int | None = None,
                 max_num_queued_tokens: int | None = None, pack: str = "off", pack_min: int | None = None,
                 hold_steps: int = 1):
        if pack not in self.PACKS:
            raise ValueError(f"dp prefill pack must be one of {self.PACKS}, not {pack!r}")
        self.pack, self.hold_steps = pack, hold_steps
        self.pack_min = pack_min if pack_min is not None else len(groups)
        self.groups = groups
        self.cfg = groups[0].cfg
        self.max_num_queued_reqs = max_num_queued_reqs  # engine-wide, like a single scheduler's
        self.max_num_queued_tokens = max_num_queued_tokens
        self.host = None
        self._sessions: dict[str, int] = {}  # session id -> the group holding its prefix
        self._held: dict[int, int] = {}  # place()'s held() per group for the request being added
        self._in_flight = False

    # -- the Scheduler interface ------------------------------------------------------

    @property
    def running(self) -> list[Request]:
        return [r for g in self.groups for r in g.running]

    @property
    def waiting(self) -> list[Request]:
        return [r for g in self.groups for r in g.waiting]

    @property
    def prefilled(self) -> list[Request]:
        return [r for g in self.groups for r in g.prefilled]

    @property
    def num_preemptions(self) -> int:
        return sum(g.num_preemptions for g in self.groups)

    @property
    def num_prefilled_preemptions(self) -> int:
        return sum(g.num_prefilled_preemptions for g in self.groups)

    @property
    def prefill_only(self) -> bool:
        return self.groups[0].prefill_only

    @prefill_only.setter
    def prefill_only(self, v: bool) -> None:
        for g in self.groups:
            g.prefill_only = v

    def add_prefilled(self, req: Request) -> None:
        """A handed-off request (Scheduler.add_prefilled) goes to the group with a free slot and the fewest
        live requests: its KV is already computed, so there is no prefix to place it by."""
        best = None
        for i, g in enumerate(self.groups):
            live = len(g.running) + len(g.waiting) + len(g.prefilled)
            key = (live >= g.cfg.max_num_seqs, live, i)
            if best is None or key < best:
                best = key
        req.dp_group = best[2]
        self.groups[best[2]].add_prefilled(req)

    @property
    def in_flight(self) -> bool:
        return self._in_flight

    @in_flight.setter
    def in_flight(self, v: bool) -> None:
        self._in_flight = v
        for g in self.groups:
            g.in_flight = v

    def has_work(self) -> bool:
        return any(g.has_work() for g in self.groups)

    def add(self, req: Request) -> None:
        if self.max_num_queued_reqs is not None and len(self.waiting) >= self.max_num_queued_reqs:
            raise QueueFull(f"{len(self.waiting)} requests already queued")
        if self.max_num_queued_tokens is not None:
            queued = sum(r.num_tokens for r in self.waiting)
            if queued + req.num_tokens > self.max_num_queued_tokens:
                raise QueueFull(f"{queued} tokens already queued")
        g = self._sessions.get(req.session_id) if req.session_id is not None else None
        self._held = {}
        if g is None:
            g = self.place(req)
        if req.session_id is not None:
            self._sessions[req.session_id] = g
        req.dp_group = g
        held = self._held.get(g)
        req.dp_charge = req.num_tokens - (self.held(self.groups[g], req) if held is None else held)
        self.groups[g].add(req)

    def held(self, g: Scheduler, req: Request) -> int:
        """Page-aligned tokens of req's prompt group g holds: its radix cache's match (continued by the
        pages its host tier holds), or the prefix req shares with a request queued or running there
        (computed once for both)."""
        if not g.cfg.prefix_cache:
            return 0
        ps = self.cfg.page_size
        lim = (req.num_tokens - 1) // ps
        n = g.radix.match_prefix(req.token_ids, lim).num_pages
        if g.host is not None:
            n += g.host.peek(req.token_ids, n, lim)
        for r in list(g.running) + list(g.waiting):
            if n >= lim:
                break
            n = max(n, shared_pages(req.token_ids, r.token_ids, ps, lim))
        return n * ps

    def place(self, req: Request) -> int:
        """The group for a new request: a group with a free slot first, then fewest tokens to compute
        (its own not held there plus what the group's live requests were charged), then fewest
        requests, then the lowest index."""
        best = None
        for i, g in enumerate(self.groups):
            live = list(g.running) + list(g.waiting)
            load = sum(r.dp_charge if r.dp_charge is not None else r.num_tokens for r in live)
            held = self._held[i] = self.held(g, req)
            key = (len(live) >= g.cfg.max_num_seqs, load + req.num_tokens - held, len(live), i)
            if best is None or key < best:
                best = key
        return best[3]

    def schedule(self) -> StepPlan:
        """Every group's plan, decodes first then prefills (a group raising NeedSync stops the
        step; the groups before it only reserved pages or admitted prefills, which the retry
        schedules again)."""
        plan, parts = StepPlan(), []
        for i, g in enumerate(self.groups):
            try:
                parts.append(g.schedule())
            except NeedSync:
                for done in self.groups[:i]:  # their plans will not run either (state checkpoints)
                    done.rollback()
                raise
        if self.pack != "off":
            self._pack(parts)
        for p in parts:
            plan.decodes += p.decodes
            plan.prefills += p.prefills
            plan.deferred += p.deferred
            plan.pd_injected += p.pd_injected
        return plan

    def _pack(self, parts: list[StepPlan]) -> None:
        """Prefill packing (module docstring): trim each group's chunks that would add a call carrying fewer
        than pack_min chunks, and with "hold" defer a sparse single call (bounded per request)."""
        keep = [len(p.prefills) for p in parts]
        for c in range(max(keep, default=0) - 1, 0, -1):  # the calls past the first, the last first
            if sum(k > c for k in keep) >= self.pack_min:
                break  # this call is full enough, and so is every one before it
            keep = [min(k, c) for k in keep]
        if self.pack == "hold" and max(keep, default=0) == 1 and sum(keep) < self.pack_min \
                and any(p.decodes for p in parts):
            first = [p.prefills[0] for p, k in zip(parts, keep) if k]
            if all(e.req.prefill_holds < self.hold_steps for e in first):
                for e in first:
                    e.req.prefill_holds += 1
                keep = [0] * len(keep)
        for g, p, k in zip(self.groups, parts, keep):
            if k < len(p.prefills):
                g.defer(p.prefills[k:])
                p.deferred += len(p.prefills) - k
                p.prefills = p.prefills[:k]

    def _split(self, plan: StepPlan, tokens: list | None = None):
        parts = [(StepPlan(), []) for _ in self.groups]
        seqs = plan.seqs()
        for i, s in enumerate(seqs):
            p, toks = parts[s.req.dp_group]
            (p.decodes if i < len(plan.decodes) else p.prefills).append(s)
            if tokens is not None:
                toks.append(tokens[i])
        return parts

    def update(self, plan: StepPlan, tokens: list) -> list[Request]:
        self.advance(plan)
        return self.commit(plan, tokens)

    def advance(self, plan: StepPlan) -> None:
        for g, (p, _) in zip(self.groups, self._split(plan)):
            g.advance(p)

    def commit(self, plan: StepPlan, tokens: list) -> list[Request]:
        finished = []
        for g, (p, toks) in zip(self.groups, self._split(plan, tokens)):
            finished += g.commit(p, toks)
        return finished

    def finish_now(self, req: Request, reason: str) -> None:
        self.groups[req.dp_group].finish_now(req, reason)

    def abort(self, req: Request) -> None:
        self.groups[req.dp_group].abort(req)

    def close_session(self, sid: str) -> bool:
        g = self._sessions.pop(sid, None)
        return self.groups[g].close_session(sid) if g is not None else False

    def _reserve(self, req: Request, total_tokens: int) -> bool:
        return self.groups[req.dp_group]._reserve(req, total_tokens)

    def _finish_reason(self, req: Request, tok: int, length: int) -> str | None:
        return self.groups[req.dp_group]._finish_reason(req, tok, length)

    shared_tokens = staticmethod(Scheduler.shared_tokens)
