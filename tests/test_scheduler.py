"""Scheduler invariants under load, prefix sharing and preemption.

The "model" here is a pure function of the token history (next = hash of history), so
the expected output of every request is known exactly no matter how the scheduler
batched, chunked, cached or preempted it. That is the property under test: scheduling
must never change what a request generates.
"""

import random

from kiln.engine.kv_pool import PagePool
from kiln.engine.radix_cache import RadixCache
from kiln.engine.request import Request, SamplingParams, Status
from kiln.engine.scheduler import Scheduler, SchedulerConfig

PS = 4
VOCAB = 1000
EOS = 7


def next_token(history):
    h = 1469598103934665603
    for t in history:
        h = ((h ^ t) * 1099511628211) & 0xFFFFFFFFFFFF
    return h % VOCAB


def reference_output(prompt, max_new):
    hist = list(prompt)
    out = []
    for _ in range(max_new):
        t = next_token(hist)
        out.append(t)
        hist.append(t)
        if t == EOS:
            break
    return out


def check_invariants(sched: Scheduler):
    pool, radix = sched.pool, sched.radix
    tree_pages = set()
    stack = [radix.root]
    while stack:
        n = stack.pop()
        tree_pages.update(n.pages)
        stack.extend(n.children.values())
    private = []
    for req in sched.running:
        assert len(req.pages) * PS >= req.num_computed
        n_full = req.num_computed // PS
        # Full pages may be shared (tree) or private; the partial tail is always private.
        private.extend(p for p in req.pages if p not in tree_pages)
    assert len(private) == len(set(private)), "a private page is held by two requests"
    assert pool.num_free + len(tree_pages) + len(private) == pool.num_usable
    assert radix.total_pages() == len(tree_pages)
    for req in sched.waiting:
        assert req.pages == [] and req.node is None


def run(sched: Scheduler, reqs, max_steps=10_000, check_every=1):
    for r in reqs:
        sched.add(r)
    steps = 0
    while sched.has_work():
        plan = sched.schedule()
        assert plan, "scheduler made no progress"
        tokens = []
        for s in plan.seqs():
            # The step "computes" positions [start, end); the next token depends only
            # on the history up to end, exactly like a real model's last-position logits.
            tokens.append(next_token(s.req.token_ids[: s.end]) if s.sample else -1)
        sched.update(plan, tokens)
        steps += 1
        if steps % check_every == 0:
            check_invariants(sched)
        assert steps < max_steps
    return steps


def make(num_pages=256, max_seqs=8, budget=16, max_len=512, admission="reserve", decode_tokens=256):
    pool = PagePool(num_pages)
    radix = RadixCache(pool, PS)
    cfg = SchedulerConfig(page_size=PS, max_num_seqs=max_seqs, max_prefill_tokens=budget,
                          max_model_len=max_len, eos_token_ids=(EOS,), admission=admission,
                          admission_decode_tokens=decode_tokens)
    return Scheduler(cfg, pool, radix)


def test_outputs_match_reference_with_chunked_prefill():
    sched = make()
    rng = random.Random(0)
    reqs = [Request(f"r{i}", [rng.randrange(8, VOCAB) for _ in range(rng.randrange(1, 60))],
                    SamplingParams(max_new_tokens=rng.randrange(1, 40))) for i in range(20)]
    run(sched, reqs)
    for r in reqs:
        assert r.status is Status.FINISHED
        assert r.output_ids == reference_output(r.prompt_ids, r.params.max_new_tokens)
    assert sched.pool.num_free + sched.radix.total_pages() == sched.pool.num_usable


def test_shared_prefix_is_served_from_cache():
    sched = make()
    system = list(range(100, 140))  # 10 full pages
    a = Request("a", system + [9, 10], SamplingParams(max_new_tokens=3))
    run(sched, [a])
    b = Request("b", system + [11, 12, 13], SamplingParams(max_new_tokens=3))
    run(sched, [b])
    assert b.num_cached_tokens == len(system)
    assert b.output_ids == reference_output(b.prompt_ids, 3)


def test_concurrent_same_prefix_dedups_pages():
    sched = make(budget=64)
    system = list(range(200, 232))
    reqs = [Request(f"c{i}", system + [i + 10], SamplingParams(max_new_tokens=5)) for i in range(4)]
    run(sched, reqs)
    for r in reqs:
        assert r.output_ids == reference_output(r.prompt_ids, 5)
    # One copy of the shared 8 pages survives in the tree, not four.
    assert sched.radix.match_prefix(system).num_pages == len(system) // PS


def test_preemption_under_kv_pressure_keeps_outputs_exact():
    # 40 usable pages for 8 sequences that each grow to ~30 pages: with eager admission preemption is
    # forced (the reserve admission runs them one or two at a time instead: next test).
    sched = make(num_pages=41, max_seqs=8, budget=32, admission="eager")
    rng = random.Random(1)
    reqs = [Request(f"p{i}", [rng.randrange(8, VOCAB) for _ in range(20)],
                    SamplingParams(max_new_tokens=100, ignore_eos=True)) for i in range(8)]
    run(sched, reqs)
    assert sched.num_preemptions > 0
    for r in reqs:
        assert r.output_ids == reference_output_no_eos(r.prompt_ids, 100)


def _peak_running(sched, reqs):
    """run() that also records the most requests running at once."""
    peak = [0]
    real = sched.schedule

    def schedule():
        plan = real()
        peak[0] = max(peak[0], len(sched.running))
        return plan

    sched.schedule = schedule
    run(sched, reqs)
    return peak[0]


def test_reserve_admission_avoids_preemption():
    """The same KV-short pool (40 usable pages, 8 sequences growing to 30 pages each): the reserve
    admission (SchedulerConfig.admission) never preempts, runs as many as fit at full length, gives
    the same outputs, and eager admission on the same load preempts."""
    rng = random.Random(1)
    prompts = [[rng.randrange(8, VOCAB) for _ in range(20)] for _ in range(8)]
    sp = SamplingParams(max_new_tokens=100, ignore_eos=True)
    res = make(num_pages=41, max_seqs=8, budget=32)
    reqs = [Request(f"p{i}", p, sp) for i, p in enumerate(prompts)]
    peak = _peak_running(res, reqs)
    assert res.num_preemptions == 0 and peak == 1  # 30 + 30 > 40: one at a time
    for r in reqs:
        assert r.output_ids == reference_output_no_eos(r.prompt_ids, 100)
    eager = make(num_pages=41, max_seqs=8, budget=32, admission="eager")
    run(eager, [Request(f"e{i}", p, sp) for i, p in enumerate(prompts)])
    assert eager.num_preemptions > 0
    assert res.pool.num_free + res.radix.total_pages() == res.pool.num_usable


def test_reserve_admission_runs_what_fits():
    """121 usable pages for 8 sequences of 30 pages each: four at a time, no preemption; with a decode
    reserve of 8 tokens (2 pages) it admits on the prompts' 5 + 2 pages each, overcommits and falls
    back to preemption, still exact."""
    rng = random.Random(5)
    prompts = [[rng.randrange(8, VOCAB) for _ in range(20)] for _ in range(8)]
    sp = SamplingParams(max_new_tokens=100, ignore_eos=True)
    sched = make(num_pages=121, max_seqs=8, budget=32)
    reqs = [Request(f"f{i}", p, sp) for i, p in enumerate(prompts)]
    assert _peak_running(sched, reqs) == 4 and sched.num_preemptions == 0
    small = make(num_pages=121, max_seqs=8, budget=32, decode_tokens=8)
    reqs2 = [Request(f"s{i}", p, sp) for i, p in enumerate(prompts)]
    assert _peak_running(small, reqs2) == 8 and small.num_preemptions > 0
    for r in reqs + reqs2:
        assert r.output_ids == reference_output_no_eos(r.prompt_ids, 100)


def test_reserve_admission_overlapped_and_oversized():
    """Overlapped scheduling with the reserve admission: no preemption, exact; and a request whose
    reserve (prompt plus a decode reserve larger than its generation) takes nearly the whole pool is
    admitted when nothing else runs."""
    rng = random.Random(6)
    sp = SamplingParams(max_new_tokens=100, ignore_eos=True)
    sched = make(num_pages=41, max_seqs=8, budget=32)
    reqs = [Request(f"ov{i}", [rng.randrange(8, VOCAB) for _ in range(20)], sp) for i in range(6)]
    run_overlapped(sched, reqs)
    assert sched.num_preemptions == 0
    for r in reqs:
        assert r.output_ids == reference_output_no_eos(r.prompt_ids, 100)
    big = make(num_pages=41, max_seqs=8, budget=32, decode_tokens=1000)
    r = Request("big", [rng.randrange(8, VOCAB) for _ in range(20)], SamplingParams(max_new_tokens=130, ignore_eos=True))
    run(big, [r])  # target 150 tokens = 38 pages of the 40: fits alone
    assert r.output_ids == reference_output_no_eos(r.prompt_ids, 130)


def reference_output_no_eos(prompt, max_new):
    hist = list(prompt)
    for _ in range(max_new):
        hist.append(next_token(hist))
    return hist[len(prompt):]


def test_abort_releases_everything():
    sched = make()
    r = Request("x", list(range(10, 30)), SamplingParams(max_new_tokens=50))
    sched.add(r)
    plan = sched.schedule()
    sched.update(plan, [next_token(s.req.token_ids[: s.end]) if s.sample else -1 for s in plan.seqs()])
    sched.abort(r)
    assert not sched.has_work()
    assert sched.pool.num_free + sched.radix.total_pages() == sched.pool.num_usable


def admitted_order(sched, reqs):
    order = []
    for r in reqs:
        sched.add(r)
    while sched.has_work():
        plan = sched.schedule()
        for s in plan.prefills:
            if s.req.rid not in order:  # first prefill, cached prefix or not
                order.append(s.req.rid)
        sched.update(plan, [next_token(s.req.token_ids[: s.end]) if s.sample else -1 for s in plan.seqs()])
    return order


def make_policy(policy, **kw):
    sched = make(max_seqs=1, budget=64, **kw)
    sched.cfg.policy = policy
    return sched


def test_priority_policy_serves_lower_values_first():
    sched = make_policy("priority")
    reqs = [Request(f"p{p}", [10 + p] * 8, SamplingParams(max_new_tokens=2), priority=p) for p in (5, 1, 3)]
    assert admitted_order(sched, reqs) == ["p1", "p3", "p5"]


def test_spf_serves_shortest_prefill_first():
    sched = make_policy("spf")
    reqs = [Request(f"n{n}", [20] * n, SamplingParams(max_new_tokens=2)) for n in (30, 5, 12)]
    assert admitted_order(sched, reqs) == ["n5", "n12", "n30"]


def test_lpm_serves_longest_cached_prefix_first():
    sched = make_policy("lpm")
    shared = list(range(300, 332))
    run(sched, [Request("warm", shared + [1], SamplingParams(max_new_tokens=1))])
    reqs = [Request("cold", list(range(500, 540)), SamplingParams(max_new_tokens=2)),
            Request("hot", shared + [2, 3], SamplingParams(max_new_tokens=2))]
    assert admitted_order(sched, reqs) == ["hot", "cold"]


def test_admission_control_rejects_when_queue_is_full():
    from kiln.engine.scheduler import QueueFull

    sched = make()
    sched.cfg.max_num_queued_reqs = 2
    sched.add(Request("a", [1, 2], SamplingParams()))
    sched.add(Request("b", [1, 2], SamplingParams()))
    try:
        sched.add(Request("c", [1, 2], SamplingParams()))
    except QueueFull:
        pass
    else:
        raise AssertionError("third request admitted past max_num_queued_reqs=2")


def test_priority_preemption_keeps_outputs_exact():
    sched = make(num_pages=41, max_seqs=8, budget=32, admission="eager")
    sched.cfg.policy = "priority"
    rng = random.Random(2)
    reqs = [Request(f"q{i}", [rng.randrange(8, VOCAB) for _ in range(20)],
                    SamplingParams(max_new_tokens=100, ignore_eos=True), priority=i % 3) for i in range(8)]
    run(sched, reqs)
    assert sched.num_preemptions > 0
    for r in reqs:
        assert r.output_ids == reference_output_no_eos(r.prompt_ids, 100)


def run_overlapped(sched, reqs, max_steps=20_000):
    """Drive the scheduler the way an overlapping engine does: step N+1 is scheduled and
    'launched' before step N's tokens are committed. The 'device' resolves a pending input
    token from a board of each request's last sampled token, exactly like the real board."""
    from kiln.engine.scheduler import PENDING, NeedSync

    for r in reqs:
        sched.add(r)
    board = {}
    inflight = None
    steps = 0

    def launch(plan):
        out = []
        for s in plan.seqs():
            hist = [board[s.req.rid] if t == PENDING else t for t in s.req.token_ids[: s.end]]
            assert PENDING not in hist
            if s.sample:
                board[s.req.rid] = next_token(hist)
            out.append(board[s.req.rid] if s.sample else -1)
        return out

    while sched.has_work() or inflight:
        sched.in_flight = inflight is not None
        try:
            plan = sched.schedule()
        except NeedSync:
            sched.commit(*inflight)
            inflight = None
            check_invariants(sched)
            continue
        if not plan:
            if inflight:
                sched.commit(*inflight)
                inflight = None
            continue
        toks = launch(plan)
        sched.advance(plan)
        if inflight:
            sched.commit(*inflight)
        inflight = (plan, toks)
        steps += 1
        assert steps < max_steps
    return steps


def test_overlapped_scheduling_gives_identical_outputs():
    rng = random.Random(3)
    reqs = [Request(f"o{i}", [rng.randrange(8, VOCAB) for _ in range(rng.randrange(1, 50))],
                    SamplingParams(max_new_tokens=rng.randrange(1, 40))) for i in range(24)]
    run_overlapped(make(), reqs)
    for r in reqs:
        assert r.output_ids == reference_output(r.prompt_ids, r.params.max_new_tokens), r.rid


def test_overlapped_scheduling_under_preemption_pressure():
    sched = make(num_pages=41, max_seqs=8, budget=32, admission="eager")
    rng = random.Random(4)
    reqs = [Request(f"op{i}", [rng.randrange(8, VOCAB) for _ in range(20)],
                    SamplingParams(max_new_tokens=100, ignore_eos=True)) for i in range(8)]
    run_overlapped(sched, reqs)
    assert sched.num_preemptions > 0
    for r in reqs:
        assert r.output_ids == reference_output_no_eos(r.prompt_ids, 100)
    assert sched.pool.num_free + sched.radix.total_pages() == sched.pool.num_usable


def test_session_turns_survive_pressure_that_evicts_other_prefixes():
    """An agent session's previous turn stays cached while one-off traffic churns the pool;
    the same workload without the session id loses it (SGLang #29173's motivating case)."""

    def workload(session):
        sched = make(num_pages=40, max_seqs=2, budget=64)
        convo = list(range(100, 164))  # 16 pages of conversation so far
        run(sched, [Request("t1", convo, SamplingParams(max_new_tokens=4, ignore_eos=True),
                            session_id=session)])
        rng = random.Random(1)
        for i in range(6):  # unrelated requests filling the rest of the pool
            run(sched, [Request(f"x{i}", [rng.randrange(8, VOCAB) for _ in range(56)],
                                SamplingParams(max_new_tokens=4, ignore_eos=True))])
        t2 = Request("t2", convo + [9, 10, 11], SamplingParams(max_new_tokens=4, ignore_eos=True),
                     session_id=session)
        run(sched, [t2])
        assert t2.output_ids == reference_output_no_eos(t2.prompt_ids, 4)
        return t2.num_cached_tokens, sched

    cached, sched = workload("agent-1")
    assert cached == 64
    assert sched.radix.session_ids() == ["agent-1"] and sched.close_session("agent-1")
    assert workload(None)[0] < 64
