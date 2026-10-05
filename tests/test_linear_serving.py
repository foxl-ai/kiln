"""Serving features of linear-attention models on CPU in fp32: the prefix cache resumed from state
checkpoints (engine/state_pool.py, RadixNode.ckpt), speculative decoding whose verify keeps the
state after every drafted position (models/linear_attn.verify_scan), jump-forward, and the host
KV tier. Every check is greedy output equal to the same engine with the feature off (and, for
the tiny Qwen3.5-architecture model, to transformers): shared system prompts, multi-turn
sessions, hits at checkpoint boundaries, eviction under memory pressure, preemption, overlap,
piecewise graphs, tp=2, and partial acceptance of drafts (an oracle proposer that corrupts a
chosen draft position, so how many tokens a verify keeps is known in advance).

The tiny models come from tests/test_linear_attn.py (Qwen3.5 GDN + gated attention via
transformers, a KDA-only Kimi Linear checkpoint written directly). The real Qwen/Qwen3.5-0.8B
runs when KILN_TEST_LINEAR_MODEL is set.
"""

import itertools
import os

import pytest
import torch

from tests.test_linear_attn import build_kda, build_qwen3_5, engine, hf_greedy, prompts

VOCAB = 384


@pytest.fixture(scope="module")
def gdn(tmp_path_factory):
    path = tmp_path_factory.mktemp("qwen3_5_serving")
    return str(path), build_qwen3_5(str(path))


@pytest.fixture(scope="module")
def kda(tmp_path_factory):
    path = tmp_path_factory.mktemp("kda_serving")
    build_kda(str(path), full_rank=False, lower_bound=-5.0)
    return str(path)


def sp(n=12, **kw):
    from kiln.engine.request import SamplingParams

    return SamplingParams(max_new_tokens=n, ignore_eos=True, **kw)


def run(eng, groups, n=12):
    """Generate each group of prompts in turn (later groups can hit what earlier ones cached)."""
    out = []
    for g in groups:
        out += eng.generate(g, sp(n))
    return out


# -- the verify scan --------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["recurrent", "chunk"])
@pytest.mark.parametrize("per_channel", [False, True])
def test_verify_scan_states_match_the_recurrence(form, per_channel):
    """Every intermediate state and output of B sequences x Q tokens equals the token-by-token
    recurrence from each sequence's own non-zero state (GDN per-head and KDA per-channel decay)."""
    from kiln.models.linear_attn import recurrent_step, verify_scan

    g = torch.Generator().manual_seed(5)
    B, Q, H, Dk, Dv = 3, 5, 2, 8, 6
    q = torch.nn.functional.normalize(torch.randn(B, Q, H, Dk, generator=g), dim=-1) * Dk ** -0.5
    k = torch.nn.functional.normalize(torch.randn(B, Q, H, Dk, generator=g), dim=-1)
    v = torch.randn(B, Q, H, Dv, generator=g)
    gate = -torch.rand((B, Q, H, Dk) if per_channel else (B, Q, H), generator=g) * 3
    beta = torch.rand(B, Q, H, generator=g)
    S0 = torch.randn(B, H, Dk, Dv, generator=g)
    o, states = verify_scan(q, k, v, gate, beta, S0, form)
    S = S0
    for t in range(Q):
        want_o, S = recurrent_step(q[:, t], k[:, t], v[:, t], gate[:, t], beta[:, t], S)
        assert (o[:, t] - want_o).abs().max().item() < 1e-5
        assert (states[:, t] - S).abs().max().item() < 1e-5


# -- speculative decoding -------------------------------------------------------------------------


def set_draft_fn(eng, fn):
    """The engine's scheduler, or every DP-attention group's (engine/dp.py), drafts with fn."""
    for sch in getattr(eng.scheduler, "groups", [eng.scheduler]):
        sch.draft_fn = fn


def oracle(eng, want: dict, k: int, pattern):
    """A proposer drafting the reference continuation with the draft position pattern[i] (cycling
    per call) corrupted, so verify keeps exactly that many drafts (all of them when the position is
    past the draft). Installed in place of the engine's own (ngram / suffix) proposer."""
    calls = itertools.count()
    stats = {"full": 0, "partial": 0, "none": 0}

    def draft(req):
        L = len(req.output_ids)
        w = want.get(tuple(req.prompt_ids))
        if w is None:  # a prompt the oracle has no reference for: plain decoding
            return []
        d = list(w[L : min(L + k, len(w) - 1)])
        c = pattern[next(calls) % len(pattern)]
        if not d:
            return d
        if c < len(d):
            d[c] = (d[c] + 1) % VOCAB
            stats["partial" if c else "none"] += 1
        else:
            stats["full"] += 1
        return d

    set_draft_fn(eng, draft)
    return stats


@pytest.mark.parametrize("mode", ["graph", "piecewise"])
def test_spec_partial_acceptance_is_exact(gdn, mode):
    """Drafts accepted fully, partly (rejected at every position) and not at all: the output equals
    plain greedy decoding (and transformers), and the accepted counts are exactly the oracle's."""
    path, hf = gdn
    ps = prompts(11, (5, 19, 33))
    want = [hf_greedy(hf, p, 20) for p in ps]
    k = 3
    eng = engine(path, spec_method="ngram", spec_k=k, piecewise=mode == "piecewise", piecewise_group=2)
    stats = oracle(eng, {tuple(p): w for p, w in zip(ps, want)}, k, pattern=[3, 0, 1, 2, 5, 1])
    reqs = eng.generate(ps, sp(20))
    assert [r.output_ids for r in reqs] == want
    assert stats["full"] and stats["partial"] and stats["none"]
    assert 0 < eng.spec_accepted < eng.spec_proposed
    print(f"oracle drafts {stats}: accepted {eng.spec_accepted}/{eng.spec_proposed}")


@pytest.mark.parametrize("method", ["ngram", "suffix"])
def test_spec_methods_match_plain_decoding(gdn, method):
    """The engine's own proposers. Suffix decoding remembers earlier outputs, so a second pass over
    the same prompts drafts their continuations and has them accepted."""
    path, hf = gdn
    ps = prompts(12, (7, 26))
    want = [hf_greedy(hf, p, 24) for p in ps]
    eng = engine(path, spec_method=method, spec_k=4, spec_ngram_min=1)
    for _ in range(2):
        assert [r.output_ids for r in eng.generate(ps, sp(24))] == want
    if method == "suffix":
        assert eng.spec_accepted > 0
    print(f"{method}: accepted {eng.spec_accepted}/{eng.spec_proposed}")


def test_spec_k_per_batch_size(gdn):
    """vLLM's num_speculative_tokens_per_batch_size: drafts of up to 3 while one request runs, none
    while three do; the output is plain greedy decoding's either way. Suffix decoding drafts a
    prompt's continuation once it has seen it (its global tree of finished outputs)."""
    path, hf = gdn
    ps = prompts(14, (5, 9, 30))
    want = [hf_greedy(hf, p, 12) for p in ps]
    eng = engine(path, spec_method="suffix", spec_k=3, spec_k_per_batch_size=((1, 1, 3), (2, 64, 0)))
    sizes = []
    real = eng._propose

    def spy(req):
        d = real(req)
        sizes.append((len(eng.scheduler.running), len(d)))
        return d

    eng.scheduler.draft_fn = spy
    for p, w in zip(ps, want):  # one at a time, twice: the second run of each is drafted
        for _ in range(2):
            assert eng.generate([p], sp(12))[0].output_ids == w
    alone = list(sizes)
    sizes.clear()
    assert [r.output_ids for r in eng.generate(ps, sp(12))] == want
    assert any(d > 0 for _, d in alone)
    assert all(d == 0 for n, d in sizes if n > 1)


def test_spec_kda_and_tp2(kda):
    """KDA (per-channel decay) through verify, at tp=1 and tp=2, against plain decoding."""
    ps = prompts(13, (9, 30))
    plain = engine(kda, max_num_seqs=2)
    want = [r.output_ids for r in plain.generate(ps, sp(16))]
    w = {tuple(p): x for p, x in zip(ps, want)}
    eng = engine(kda, max_num_seqs=2, spec_method="ngram", spec_k=3)
    oracle(eng, w, 3, pattern=[1, 3, 0, 2])
    assert [r.output_ids for r in eng.generate(ps, sp(16))] == want
    assert 0 < eng.spec_accepted < eng.spec_proposed
    two = engine(kda, max_num_seqs=2, spec_method="ngram", spec_k=3, tp=2)
    try:
        oracle(two, w, 3, pattern=[2, 0, 3])
        assert [r.output_ids for r in two.generate(ps, sp(16))] == want
    finally:
        two.close()


# -- prefix caching ---------------------------------------------------------------------------------


def test_shared_system_prompt(gdn):
    """A system prompt shared by several users. The first request leaves a checkpoint at its own
    last page boundary only; the second finds the KV junction past it and checkpoints there; every
    later one resumes from the junction. Outputs equal the cache-off engine and transformers."""
    path, hf = gdn
    (system,) = prompts(20, (37,))
    users = prompts(21, (5, 9, 3, 12, 7))
    ps = [system + u for u in users]
    want = [hf_greedy(hf, p, 12) for p in ps]
    off = engine(path, prefix_caching=False)
    assert [r.output_ids for r in run(off, [[p] for p in ps])] == want
    eng = engine(path)
    reqs = run(eng, [[ps[0]], [ps[1]], ps[2:]])
    assert [r.output_ids for r in reqs] == want
    junction = len(system) // 4 * 4
    assert [r.num_cached_tokens for r in reqs] == [0, 0] + [junction] * 3
    assert eng.radix.num_ckpts >= 1  # the junction (prompt-boundary checkpoints are off by default)
    assert eng.pool.num_free + eng.radix.total_pages() == eng.pool.num_usable  # nothing leaked
    assert eng.runner.state.num_free_ckpts + eng.radix.num_ckpts == eng.runner.state.num_ckpt_rows


def test_burst_sharing_a_prefix_hits_from_the_second_request(gdn):
    """Lookahead junctions (scheduler.py "Junctions are also taken AHEAD"): users sharing a system
    prompt arrive together. The first one admitted checkpoints the junction it shares with the queued
    ones and the others wait for it, so every one after the first resumes from it. Without lookahead
    they are admitted together and compute the shared prefix again. Outputs equal transformers."""
    path, hf = gdn
    (system,) = prompts(60, (37,))
    ps = [system + u for u in prompts(61, (5, 9, 3, 12, 7))]
    want = [hf_greedy(hf, p, 12) for p in ps]
    j = len(system) // 4 * 4
    eng = engine(path, max_num_seqs=5, max_prefill_tokens=64)
    reqs = eng.generate(ps, sp(12))
    assert [r.output_ids for r in reqs] == want
    assert [r.num_cached_tokens for r in reqs] == [0] + [j] * 4
    assert eng.runner.state.num_free_ckpts + eng.radix.num_ckpts == eng.runner.state.num_ckpt_rows
    late = engine(path, max_num_seqs=5, max_prefill_tokens=64, state_checkpoint_lookahead=False)
    reqs = late.generate(ps, sp(12))
    assert [r.output_ids for r in reqs] == want
    assert sum(r.num_cached_tokens for r in reqs) < 4 * j


def test_request_arriving_mid_prefill_resumes_from_the_shared_prefix(gdn):
    """A request sharing a prefix with one still being prefilled: the running one gets the junction as
    a checkpoint target, the new one waits for it and resumes from it."""
    path, hf = gdn
    (system,) = prompts(62, (40,))
    a, b = [system + u for u in prompts(63, (6, 10))]
    for overlap in (False, True):
        eng = engine(path, overlap=overlap)  # 8 prefill tokens per step: a takes 6 steps
        ra = eng.add_request(a, sp(10))
        eng.step()
        rb = eng.add_request(b, sp(10))
        while eng.has_work():
            eng.step()
        assert ra.output_ids == hf_greedy(hf, a, 10) and rb.output_ids == hf_greedy(hf, b, 10)
        assert (ra.num_cached_tokens, rb.num_cached_tokens) == (0, 40), overlap


def test_identical_prompt_resumes_from_its_last_page(gdn):
    """state_checkpoint_prompt (vLLM's replay boundary): every prefill checkpoints its last page
    boundary, so the second request of a prompt already hits. Without it the second one checkpoints
    the junction and the third hits."""
    path, hf = gdn
    (p,) = prompts(22, (30,))
    off = engine(path)
    assert [r.num_cached_tokens for r in run(off, [[p], [p], [p]])] == [0, 0, 28]
    eng = engine(path, state_checkpoint_prompt=True)
    a, b = run(eng, [[p], [p]])
    assert a.output_ids == b.output_ids == hf_greedy(hf, p, 12)
    assert (a.num_cached_tokens, b.num_cached_tokens) == (0, 28)


def test_multi_turn_session_resumes_from_the_decode_checkpoint(gdn):
    """Turn 2 is turn 1's prompt and answer plus a new message. Decode checkpoints every 8 tokens:
    turn 2 resumes from the last one inside turn 1's answer, not from turn 1's prompt."""
    path, hf = gdn
    (p1,) = prompts(23, (13,))
    (u2, u3) = prompts(24, (6, 4))
    eng = engine(path, state_track_interval=8)
    off = engine(path, prefix_caching=False)
    t1 = eng.generate([p1], sp(20))[0]
    assert t1.output_ids == hf_greedy(hf, p1, 20)
    p2 = p1 + t1.output_ids + u2
    t2 = eng.generate([p2], sp(10))[0]
    assert t2.output_ids == off.generate([p2], sp(10))[0].output_ids == hf_greedy(hf, p2, 10)
    # turn 1 computed 13 + 19 positions: decode checkpoints at 16, 24, 32 (the newest kept)
    assert t2.num_cached_tokens == 32
    p3 = p2 + t2.output_ids + u3
    t3 = eng.generate([p3], sp(8))[0]
    assert t3.output_ids == hf_greedy(hf, p3, 8)
    assert t3.num_cached_tokens >= len(p2) // 8 * 8


def test_hits_land_on_checkpoint_boundaries(gdn):
    """Periodic checkpoints every 8 tokens: a prompt sharing 30 tokens with an earlier one resumes
    from the checkpoint at 24 (the KV match at 28 has none yet) and checkpoints the junction at 28,
    which the next sharer resumes from."""
    path, hf = gdn
    (a,) = prompts(25, (50,))
    tails = prompts(26, (9, 11))
    b, c = a[:30] + tails[0], a[:30] + tails[1]
    eng = engine(path, state_checkpoint_interval=8)
    reqs = run(eng, [[a], [b], [c]])
    assert [r.output_ids for r in reqs] == [hf_greedy(hf, p, 12) for p in (a, b, c)]
    assert [r.num_cached_tokens for r in reqs] == [0, 24, 28]


def test_eviction_under_memory_pressure(gdn):
    """A small page pool and three checkpoint rows, prompts drawn from a few shared prefixes in a
    shuffled order, several running at once: KV and checkpoints are evicted and reused, and every
    output still equals the cache-off engine's."""
    path, _ = gdn
    bases = prompts(27, (17, 26, 9))
    tails = prompts(28, (3, 6, 2, 8, 5, 4, 7, 3, 6))
    ps = [bases[i % 3] + t for i, t in enumerate(tails)]
    off = engine(path, prefix_caching=False, num_pages=48)
    want = [r.output_ids for r in run(off, [ps[i : i + 3] for i in range(0, 9, 3)])]
    eng = engine(path, num_pages=48, state_checkpoints=3, state_checkpoint_interval=8)
    for _ in range(2):
        reqs = run(eng, [ps[i : i + 3] for i in range(0, 9, 3)])
        assert [r.output_ids for r in reqs] == want
    assert sum(r.num_cached_tokens > 0 for r in reqs) >= 3
    assert eng.radix.num_ckpts <= 3
    assert eng.runner.state.num_free_ckpts + eng.radix.num_ckpts == 3


def test_preemption_resumes_from_a_checkpoint(gdn):
    """A pool too small for every request preempts; the victim's checkpoints (prompt boundary,
    decode) enter the tree when it is released, and its recompute resumes from one."""
    path, hf = gdn
    ps = prompts(29, (25, 30, 22))
    eng = engine(path, num_pages=20, max_prefill_tokens=16, state_track_interval=8, admission="eager")
    reqs = eng.generate(ps, sp(16))
    assert eng.scheduler.num_preemptions > 0
    assert [r.output_ids for r in reqs] == [hf_greedy(hf, p, 16) for p in ps]


def test_overlap_preemption_keeps_checkpoint_rows(gdn):
    """Overlap scheduling with a pool that forces preemptions (the scheduler gives up a plan with
    NeedSync and retries after draining) and decode checkpoints every 4 tokens: outputs equal the
    reference and no checkpoint row is lost or left pending."""
    path, hf = gdn
    ps = prompts(38, (25, 30, 22))
    eng = engine(path, num_pages=20, max_prefill_tokens=16, state_track_interval=4, overlap=True,
                 admission="eager")
    reqs = eng.generate(ps, sp(16))
    assert eng.scheduler.num_preemptions > 0
    assert [r.output_ids for r in reqs] == [hf_greedy(hf, p, 16) for p in ps]
    st = eng.runner.state
    assert st.num_free_ckpts + eng.radix.num_ckpts == st.num_ckpt_rows


def test_aborted_plan_returns_its_checkpoint_rows(gdn):
    """A plan given up with NeedSync (a preemption needed while a step is in flight) after it took a
    row for a decode checkpoint: the row goes back, nothing at that position stays pending, and the
    generation continues to the reference's tokens."""
    from kiln.engine.scheduler import NeedSync

    path, hf = gdn
    ps = prompts(39, (7, 9))
    eng = engine(path, state_track_interval=4)
    a, b = [eng.add_request(p, sp(24)) for p in ps]
    while not (a.is_decoding and b.is_decoding and (a.num_computed + 1) % 4 == 0):
        eng.step()
    sch, st = eng.scheduler, eng.runner.state
    real = sch._reserve
    sch._reserve = lambda req, n: real(req, n) if req is a else False
    sch.in_flight = True
    with pytest.raises(NeedSync):
        sch.schedule()
    sch._reserve, sch.in_flight = real, False
    assert all(pos != a.num_computed + 1 for pos, _, _ in a.ckpt_pending)
    held = sum(len(r.ckpt_pending) for r in (a, b))
    assert st.num_free_ckpts + eng.radix.num_ckpts + held == st.num_ckpt_rows
    while eng.has_work():
        eng.step()
    assert [a.output_ids, b.output_ids] == [hf_greedy(hf, p, 24) for p in ps]
    assert st.num_free_ckpts + eng.radix.num_ckpts == st.num_ckpt_rows


@pytest.mark.parametrize("mode", ["overlap", "piecewise"])
def test_prefix_cache_overlap_and_piecewise(gdn, mode):
    path, hf = gdn
    (system,) = prompts(30, (21,))
    ps = [system + u for u in prompts(31, (4, 9, 6, 2))]
    eng = engine(path, overlap=mode == "overlap", piecewise=mode == "piecewise", piecewise_group=2,
                 state_track_interval=8)
    reqs = run(eng, [ps[:1], ps[1:2], ps[2:]])
    assert [r.output_ids for r in reqs] == [hf_greedy(hf, p, 12) for p in ps]
    assert [r.num_cached_tokens for r in reqs][2:] == [20, 20]


def test_prefix_cache_with_speculation(gdn):
    """Both at once: a shared prefix resumed from its checkpoint, drafts partly accepted, decode
    checkpoints taken from the verify rows that hold them."""
    path, hf = gdn
    (system,) = prompts(32, (18,))
    ps = [system + u for u in prompts(33, (3, 7, 5))]
    want = [hf_greedy(hf, p, 20) for p in ps]
    eng = engine(path, spec_method="ngram", spec_k=3, state_track_interval=8)
    oracle(eng, {tuple(p): w for p, w in zip(ps, want)}, 3, pattern=[2, 0, 3, 1])
    reqs = run(eng, [ps[:1], ps[1:2], ps[2:]], n=20)
    assert [r.output_ids for r in reqs] == want
    assert reqs[2].num_cached_tokens == 16
    # a follow-up turn resumes inside the previous answer (its decode checkpoint came from a verify)
    p2 = ps[2] + reqs[2].output_ids + [5, 6, 7]
    t2 = eng.generate([p2], sp(6))[0]
    assert t2.output_ids == hf_greedy(hf, p2, 6)
    assert t2.num_cached_tokens >= (len(ps[2]) + 8) // 8 * 8


def test_prefix_cache_kda_tp2(kda):
    """KDA through checkpoints, with tp=2 (every rank copies its own state rows)."""
    (system,) = prompts(34, (23,))
    ps = [system + u for u in prompts(35, (5, 8, 3))]
    off = engine(kda, prefix_caching=False, max_num_seqs=2)
    want = [r.output_ids for r in run(off, [[p] for p in ps])]
    two = engine(kda, max_num_seqs=2, tp=2)
    try:
        reqs = run(two, [[p] for p in ps])
        assert [r.output_ids for r in reqs] == want
        assert reqs[2].num_cached_tokens == 20
    finally:
        two.close()


def test_dp_attention_checkpoints_and_speculation(gdn):
    """DP attention (engine/dp.py) at tp=2, two groups: each group's radix cache owns checkpoints in
    its own rows, restores and saves are copied on that group's ranks only, verify rows are per group.
    A shared system prompt (several users per group), decode checkpoints and partly accepted drafts:
    equal to tp=1 without the cache or drafts."""
    path, hf = gdn
    (system,) = prompts(44, (22,))
    ps = [system + u for u in prompts(45, (3, 6, 4, 5, 7, 2))]
    want = [hf_greedy(hf, p, 14) for p in ps]
    eng = engine(path, tp=2, dp_attention=2, max_num_seqs=4, spec_method="ngram", spec_k=3, state_track_interval=8)
    try:
        oracle(eng, {tuple(p): w for p, w in zip(ps, want)}, 3, pattern=[2, 0, 3, 1])
        reqs = run(eng, [ps[:2], ps[2:4], ps[4:]], n=14)
        assert [r.output_ids for r in reqs] == want
        assert sorted({r.dp_group for r in reqs}) == [0, 1]
        assert any(r.num_cached_tokens > 0 for r in reqs) and 0 < eng.spec_accepted < eng.spec_proposed
        st = eng.runner.state
        for g, rad in enumerate(eng.radixes):
            assert len(st.free_ckpt_rows(g)) + rad.num_ckpts == st.num_ckpt_rows
    finally:
        eng.close()


def test_dp_attention_rollback_keeps_the_restore(gdn):
    """DP attention schedules group by group; when a later group needs a preemption while a step is
    in flight (NeedSync), the earlier groups' plans are dropped too. A request group 0 had just
    admitted from a checkpoint must still restore it when its prefill runs in the retry, and no
    checkpoint row of the dropped plans may stay pending."""
    from kiln.engine.request import Status
    from kiln.engine.scheduler import NeedSync

    path, hf = gdn
    (p,) = prompts(46, (30,))
    (q,) = prompts(47, (9,))
    eng = engine(path, tp=2, dp_attention=2, max_num_seqs=4, state_checkpoint_prompt=True)
    try:
        dps = eng.scheduler
        dps.place = lambda req: 0
        eng.generate([p], sp(4))  # group 0 keeps a checkpoint at 28 (the prompt's last page boundary)
        dps.place = lambda req: 1
        c = eng.add_request(q, sp(30))
        while not c.is_decoding:
            eng.step()
        dps.place = lambda req: 0
        b = eng.add_request(p + [5, 6, 7], sp(8))
        g1 = dps.groups[1]
        real = g1._reserve
        g1._reserve = lambda req, n: False
        dps.in_flight = True
        with pytest.raises(NeedSync):
            dps.schedule()
        g1._reserve, dps.in_flight = real, False
        assert b.status is Status.RUNNING and b.ckpt_restore is not None and b.num_computed == 28
        while eng.has_work():
            eng.step()
        assert b.num_cached_tokens == 28 and b.output_ids == hf_greedy(hf, p + [5, 6, 7], 8)
        assert c.output_ids == hf_greedy(hf, q, 30)
        st = eng.runner.state
        for g, rad in enumerate(eng.radixes):
            assert len(st.free_ckpt_rows(g)) + rad.num_ckpts == st.num_ckpt_rows
    finally:
        eng.close()


def test_host_tier_restores_kv_and_checkpoints(gdn):
    """Evicted pages and the checkpoints at their ends go to host memory; a later prompt continuing
    them restores both and resumes from the checkpoint."""
    path, hf = gdn
    (system,) = prompts(36, (29,))
    a, b, c = [system + u for u in prompts(37, (6, 4, 5))]
    eng = engine(path, hicache_host_gb=0.01)
    reqs = run(eng, [[a], [b]])
    eng.radix.evict(eng.radix.total_pages())  # everything to the host tier
    assert eng.radix.total_pages() == 0 and eng.radix.num_ckpts == 0
    (r,) = eng.generate([c], sp(12))
    assert r.output_ids == hf_greedy(hf, c, 12)
    assert r.num_cached_tokens == 28
    assert [x.output_ids for x in reqs] == [hf_greedy(hf, p, 12) for p in (a, b)]


def test_dp_attention_host_tier(gdn):
    """DP attention with one host tier per group (engine/hicache.py GroupMoves): each group's evicted
    pages and checkpoints go to its own ranks' host memory, a later prompt is placed in the group whose
    tier holds its prefix and resumes from the restored checkpoint."""
    path, hf = gdn
    s0, s1 = prompts(70, (29, 33))
    a0, b0, c0 = [s0 + u for u in prompts(71, (6, 4, 5))]
    a1, b1, c1 = [s1 + u for u in prompts(72, (5, 7, 3))]
    eng = engine(path, tp=2, dp_attention=2, max_num_seqs=4, hicache_host_gb=0.01)
    try:
        reqs = run(eng, [[a0, a1], [b0, b1]])  # the b's checkpoint the junctions, in each group
        assert [r.dp_group for r in reqs] == [0, 1, 0, 1]
        for rad in eng.radixes:
            rad.evict(rad.total_pages())  # everything to the host tiers
            assert rad.total_pages() == 0 and rad.num_ckpts == 0
        out = eng.generate([c1, c0], sp(12))
        assert [r.output_ids for r in out] == [hf_greedy(hf, p, 12) for p in (c1, c0)]
        assert [r.dp_group for r in out] == [1, 0]
        assert [r.num_cached_tokens for r in out] == [32, 28]
        assert all(t.restored > 0 and t.ckpts_restored > 0 for t in eng.host_tiers)
        assert [r.output_ids for r in reqs] == [hf_greedy(hf, p, 12) for p in (a0, a1, b0, b1)]
    finally:
        eng.close()


@pytest.mark.parametrize("pack", ["trim", "hold"])
def test_dp_prefill_pack_keeps_every_token(gdn, pack):
    """DP prefill packing (engine/dp.py): uneven prompts over two groups, a shared prefix whose junction
    checkpoint is planned ahead and a chunk budget that splits a group's step over two requests, so chunks
    (some of them saving a checkpoint, some the first of a hit that restores one) are taken back out of a
    step and run in a later one. Every output equals transformers; the deferred chunks' checkpoint rows all
    come back; overlap scheduling and the synchronous path alike."""
    path, hf = gdn
    (system,) = prompts(80, (23,))
    ps = [system + u for u in prompts(81, (9, 3, 14, 6, 11))] + prompts(82, (17, 30, 5))
    want = [hf_greedy(hf, p, 10) for p in ps]
    for overlap in (False, True):
        eng = engine(path, tp=2, dp_attention=2, max_num_seqs=8, max_prefill_tokens=24, overlap=overlap,
                     dp_prefill_pack=pack)
        try:
            # The prompts sharing the system prompt in group 0 (junction checkpoints, hits), the rest in group 1:
            # the two groups' chunks then end at different places, so a step often has two in one group only.
            where = iter([0] * 5 + [1] * 3)
            eng.scheduler.place = lambda req: next(where)
            reqs = [eng.add_request(p, sp(10)) for p in ps]
            deferred = 0
            while eng.has_work():
                eng.step()
                deferred += eng.last_step.deferred_chunks
            assert [r.output_ids for r in reqs] == want, (pack, overlap)
            assert deferred > 0, (pack, overlap)
            assert any(r.num_cached_tokens > 0 for r in reqs)
            st = eng.runner.state
            for g, rad in enumerate(eng.radixes):
                assert len(st.free_ckpt_rows(g)) + rad.num_ckpts == st.num_ckpt_rows
        finally:
            eng.close()


def test_dp_prefill_pack_is_opt_in(monkeypatch):
    """Packing is opt-in: EngineConfig defaults to "off" (docs/neuron-notes.md "Packing prefill chunks across
    DP-attention groups": the signed-bias check that would promote "trim" is not done), KILN_DP_PREFILL_PACK sets it,
    and a DPScheduler built without one does not pack."""
    from kiln.config import EngineConfig
    from kiln.engine.dp import DPScheduler
    from kiln.engine.kv_pool import PagePool
    from kiln.engine.radix_cache import RadixCache
    from kiln.engine.scheduler import Scheduler, SchedulerConfig

    monkeypatch.delenv("KILN_DP_PREFILL_PACK", raising=False)
    assert EngineConfig(model_path="m").dp_prefill_pack == "off"
    monkeypatch.setenv("KILN_DP_PREFILL_PACK", "trim")
    assert EngineConfig(model_path="m").dp_prefill_pack == "trim"
    cfg = SchedulerConfig(page_size=4, max_num_seqs=4, max_prefill_tokens=8, max_model_len=64)
    pools = [PagePool(16) for _ in range(2)]
    assert DPScheduler([Scheduler(cfg, p, RadixCache(p, 4)) for p in pools]).pack == "off"


def test_dp_prefill_pack_trims_and_holds():
    """The packing rules on the DP scheduler alone (no model): a group's second chunk is taken out when its
    call would carry fewer than pack_min chunks, kept when every group has one; with "hold" a sparse single
    call sits out while decodes run, at most hold_steps times per request; nothing leaks."""
    from kiln.engine.dp import DPScheduler
    from kiln.engine.kv_pool import PagePool
    from kiln.engine.radix_cache import RadixCache
    from kiln.engine.request import Request, SamplingParams
    from kiln.engine.scheduler import Scheduler, SchedulerConfig

    def make(pack):
        cfg = SchedulerConfig(page_size=4, max_num_seqs=4, max_prefill_tokens=8, max_model_len=64)
        pools = [PagePool(64) for _ in range(2)]
        return DPScheduler([Scheduler(cfg, p, RadixCache(p, 4)) for p in pools], pack=pack, hold_steps=1)

    par = SamplingParams(max_new_tokens=3, ignore_eos=True)
    # group 0: a 10-token prompt (8 + 2) then a 6-token one; group 1: one 16-token prompt (8 + 8).
    lens = {"a": 10, "b": 16, "c": 6}
    for pack in ("off", "trim", "hold"):
        dps = make(pack)
        reqs = {k: Request(k, list(range(100 * i, 100 * i + n)), par) for i, (k, n) in enumerate(lens.items())}
        for k in "abc":
            dps.add(reqs[k])
        assert [reqs[k].dp_group for k in "abc"] == [0, 1, 0]
        steps = []
        while dps.has_work():
            plan = dps.schedule()
            steps.append(sorted((s.req.rid, s.start, s.end) for s in plan.prefills))
            dps.update(plan, [1] * len(plan.seqs()))
        if pack == "off":  # step 2: group 0 runs a's last 2 tokens and c's 6 in two calls, group 1 one chunk
            assert steps[1] == [("a", 8, 10), ("b", 8, 16), ("c", 0, 6)]
        elif pack == "trim":  # c waits a step instead of adding a call only group 0 fills
            assert steps[1] == [("a", 8, 10), ("b", 8, 16)] and steps[2] == [("c", 0, 6)]
        else:  # and its lone chunk sits out once more while a and b decode
            assert steps[2] == [] and steps[3] == [("c", 0, 6)]
            assert reqs["c"].prefill_holds == 1
        assert all(len(r.output_ids) == 3 for r in reqs.values())
        for g in dps.groups:
            assert g.pool.num_free + g.radix.total_pages() == g.pool.num_usable


# -- Qwen3.5 MTP drafts ---------------------------------------------------------------------------


def add_qwen3_5_mtp(path, seed=1):
    """Random MTP weights in Qwen3.5's names (mtp.fc, mtp.pre_fc_norm_*, mtp.norm, mtp.layers.0.*:
    a full-attention layer whose q_proj holds each head's query and output gate, with q / k norms
    and a dense MLP; norms zero-centred) and mtp_num_hidden_layers 1 in config.json."""
    import json

    from safetensors.torch import load_file, save_file

    cfg = json.load(open(os.path.join(path, "config.json")))
    H, I = cfg["hidden_size"], cfg["intermediate_size"]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=0.08: torch.randn(*s, generator=g) * sc  # noqa: E731
    m = "mtp.layers.0."
    w = {"mtp.fc.weight": r(H, 2 * H), "mtp.pre_fc_norm_embedding.weight": r(H, sc=0.3),
         "mtp.pre_fc_norm_hidden.weight": r(H, sc=0.3), "mtp.norm.weight": r(H, sc=0.3),
         m + "input_layernorm.weight": r(H, sc=0.3), m + "post_attention_layernorm.weight": r(H, sc=0.3),
         m + "self_attn.q_proj.weight": r(2 * nh * hd, H), m + "self_attn.k_proj.weight": r(nkv * hd, H),
         m + "self_attn.v_proj.weight": r(nkv * hd, H), m + "self_attn.o_proj.weight": r(H, nh * hd),
         m + "self_attn.q_norm.weight": r(hd, sc=0.3), m + "self_attn.k_norm.weight": r(hd, sc=0.3),
         m + "mlp.gate_proj.weight": r(I, H), m + "mlp.up_proj.weight": r(I, H), m + "mlp.down_proj.weight": r(H, I)}
    f = os.path.join(path, "model.safetensors")
    tensors = load_file(f)
    tensors.update(w)
    save_file(tensors, f, metadata={"format": "pt"})
    cfg["mtp_num_hidden_layers"] = 1
    json.dump(cfg, open(os.path.join(path, "config.json"), "w"))
    return w


def qwen3_5_mtp_block(hf, w):
    """vLLM's Qwen3_5MultiTokenPredictor layer + norm as transformers' own Qwen3.5 decoder: a one-layer
    full-attention Qwen3_5ForCausalLM holding mtp.layers.0, whose final norm is mtp.norm."""
    import copy

    import transformers as tf

    c = copy.deepcopy(hf.config)
    c.num_hidden_layers, c.layer_types = 1, ["full_attention"]
    blk = tf.Qwen3_5ForCausalLM(c).eval()
    sd = {("model.norm.weight" if n == "mtp.norm.weight" else "model.layers.0." + n[len("mtp.layers.0."):]): t
          for n, t in w.items() if n.startswith("mtp.layers.0.") or n == "mtp.norm.weight"}
    missing = blk.load_state_dict(sd, strict=False).missing_keys
    assert not [k for k in missing if k.startswith("model.layers.0") or k == "model.norm.weight"]
    return blk


def qwen3_5_reference_drafts(hf, blk, w, tokens, k):
    """MTP drafts after tokens[-1] (Qwen3_5MultiTokenPredictor.forward): position p takes the token at
    p + 1 and the target's normalised hidden at p, fc(cat(norm(embedding), norm(hidden))), the layer,
    the norm, the shared lm_head; then k - 1 recursive positions on the MTP hidden."""
    eps = hf.config.rms_norm_eps

    def norm(x, wt):  # Qwen3_5RMSNorm, zero-centred
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1 + wt)

    with torch.no_grad():
        H = hf.model(torch.tensor([tokens[:-1]])).last_hidden_state[0]
        ids, hid, drafts = list(tokens[1:]), list(H), []
        for _ in range(k):
            e = hf.model.embed_tokens(torch.tensor([ids]))[0]
            x = torch.cat([norm(e, w["mtp.pre_fc_norm_embedding.weight"]), norm(torch.stack(hid), w["mtp.pre_fc_norm_hidden.weight"])], -1)
            hm = blk.model(inputs_embeds=(x @ w["mtp.fc.weight"].T).unsqueeze(0)).last_hidden_state[0]
            d = int(hf.lm_head(hm[-1]).argmax())
            drafts.append(d)
            ids.append(d)
            hid.append(hm[-1])
    return drafts


def test_qwen3_5_mtp_drafts_and_speculation(tmp_path):
    """The Qwen3.5 MTP head: after every step (chunked prefill, then verifies of every acceptance
    length) each request's drafts equal the reference's, and the greedy output equals plain decoding.
    Qwen3.5 MTP is one full-attention layer, so drafting touches no recurrent state; the verify does."""
    path = str(tmp_path)
    hf = build_qwen3_5(path, seed=3)
    w = add_qwen3_5_mtp(path)
    blk = qwen3_5_mtp_block(hf, w)
    g = torch.Generator().manual_seed(4)
    ps = [torch.randint(0, VOCAB, (n,), generator=g).tolist() for n in (11, 23)]
    want = [r.output_ids for r in engine(path, max_num_seqs=2).generate(ps, sp(14))]
    eng = engine(path, max_num_seqs=2, spec_method="mtp", spec_k=2)
    reqs = [eng.add_request(p, sp(14)) for p in ps]
    checked = 0
    while eng.has_work():
        eng.step()
        for r in reqs:
            if r.mtp_draft and r.status.value == "running":
                assert r.mtp_draft == qwen3_5_reference_drafts(hf, blk, w, r.token_ids, 2), (r.rid, len(r.token_ids))
                checked += 1
    assert checked >= 8
    assert [r.output_ids for r in reqs] == want
    print(f"qwen3_5 mtp: {checked} draft checks, accepted {eng.spec_accepted}/{eng.spec_proposed}")


def test_mtp_warmup_covers_drafting_after_a_shrunk_batch(tmp_path):
    """After warmup, drafting compiles nothing new, also when fewer requests draft than the verify
    that produced the hidden states held (one request finishes early): the MTP graph keeps the
    producing call's batch bucket (ModelRunner.mtp_drafts batch)."""
    from kiln.engine.request import SamplingParams

    path = str(tmp_path)
    build_qwen3_5(path, seed=3)
    add_qwen3_5_mtp(path)
    eng = engine(path, max_num_seqs=2, max_model_len=64, spec_method="mtp", spec_k=2, decode_batch_buckets=(1, 2),
                 prefill_token_buckets=(8,), page_buckets=(4, 16), max_prefill_tokens=16)
    g = torch.Generator().manual_seed(5)
    a, b = (torch.randint(0, VOCAB, (n,), generator=g).tolist() for n in (7, 7))  # prefilled in one step
    # a stops on its 5th greedy token (one its earlier tokens do not contain), which a verify of both
    # requests emits: then b drafts alone from that verify's hidden states
    out = engine(path).generate([a], SamplingParams(max_new_tokens=8, ignore_eos=True))[0].output_ids
    stop = next(t for i, t in enumerate(out) if i >= 3 and t not in out[:i])
    eng.warmup()
    before = set(eng.runner.compile_seconds)
    ra, rb = eng.generate([a, b], [SamplingParams(max_new_tokens=20, stop_token_ids=(stop,)),
                                   SamplingParams(max_new_tokens=20, ignore_eos=True)])
    assert ra.finish_reason == "stop" and len(ra.output_ids) < len(rb.output_ids)
    assert set(eng.runner.compile_seconds) == before, set(eng.runner.compile_seconds) - before


# -- hyper-connection hybrids: GLM-5.3-Flash and Qwen3.8-Flash-Next (transformers >= 5.18) ---------


@pytest.fixture(scope="module", params=["glm5_next", "qwen4_exp"])
def hybrid(request, tmp_path_factory):
    """Random configs from the real config.json (tests/test_glm5_next.py, tests/test_qwen4_exp.py),
    in the sparse regime: GLM's pooled DSA keeps 16 keys, Qwen's QSA a budget of 16, so the shared
    prefixes below (past 16 tokens) select per query in prefill, decode and verify. Qwen3.8-Flash-Next
    adds the Per-Layer Embedding, whose conv history is a third kind of state row."""
    name = request.param
    pytest.importorskip(f"transformers.models.{name}.modeling_{name}")
    path = str(tmp_path_factory.mktemp(f"{name}_serving"))
    if name == "glm5_next":
        from tests.test_glm5_next import build, prompts as mk

        hf = build(path, seed=1, index_topk=16)
    else:
        from tests.test_qwen4_exp import build, prompts as mk

        hf = build(path, seed=1, indexer_budget=16, indexer_n_heads=16)
    return name, path, hf, mk


def test_hybrid_prefix_cache(hybrid):
    """KDA / GDN states (and the PLE history) resumed from checkpoints, with KV of MLA + pooled DSA
    or QSA (its raw indexer keys too) from the tree: equal to the cache-off engine."""
    name, path, _, mk = hybrid
    from tests.test_glm5_next import engine as eng_

    (system,) = mk(40, (26,))
    ps = [system + u for u in mk(41, (5, 7, 3))]
    off = eng_(path, prefix_caching=False)
    want = [r.output_ids for r in run(off, [[p] for p in ps], n=10)]
    on = eng_(path, state_track_interval=8)
    reqs = run(on, [[p] for p in ps], n=10)
    assert [r.output_ids for r in reqs] == want
    assert reqs[2].num_cached_tokens == 24
    p2 = ps[2] + want[2] + mk(42, (4,))[0]
    a, b = off.generate([p2], sp(6))[0], on.generate([p2], sp(6))[0]
    assert a.output_ids == b.output_ids and b.num_cached_tokens >= 32
    print(f"{name}: cached {[r.num_cached_tokens for r in reqs]}, turn 2 {b.num_cached_tokens}")


def test_hybrid_speculation(hybrid):
    """Verify through KDA / GDN, MLA with the pooled indexer or QSA with its block selection, the
    hyper-connection streams and the PLE n-gram rows of the drafts: drafts partly accepted, output
    equal to plain decoding and to transformers."""
    name, path, hf, mk = hybrid
    from tests.test_glm5_next import engine as eng_
    from tests.test_glm5_next import hf_greedy as hf_greedy_full

    ps = mk(43, (21, 9))
    want = [hf_greedy_full(hf, p, 12) for p in ps]
    plain = eng_(path, max_num_seqs=2)
    assert [r.output_ids for r in plain.generate(ps, sp(12))] == want
    spec = eng_(path, max_num_seqs=2, spec_method="ngram", spec_k=3)
    oracle(spec, {tuple(p): w for p, w in zip(ps, want)}, 3, pattern=[1, 3, 0, 2])
    assert [r.output_ids for r in spec.generate(ps, sp(12))] == want
    assert 0 < spec.spec_accepted < spec.spec_proposed


def test_hybrid_host_tier(hybrid):
    """Evicted pages go to host memory with every per-token cache beside K and V (GLM-5.3-Flash's
    pool-key pieces, Qwen3.8-Flash-Next's raw indexer keys) and the checkpoints; a later prompt
    continuing them restores them and equals transformers."""
    name, path, hf, mk = hybrid
    from tests.test_glm5_next import engine as eng_
    from tests.test_glm5_next import hf_greedy as hf_greedy_full

    (system,) = mk(44, (29,))
    a, b, c = [system + u for u in mk(45, (6, 4, 5))]
    eng = eng_(path, hicache_host_gb=0.01, state_track_interval=8)
    reqs = run(eng, [[a], [b]])
    eng.radix.evict(eng.radix.total_pages())  # everything to the host tier
    assert eng.radix.total_pages() == 0
    (r,) = eng.generate([c], sp(12))
    assert r.output_ids == hf_greedy_full(hf, c, 12)
    assert r.num_cached_tokens >= 24
    assert [x.output_ids for x in reqs] == [hf_greedy_full(hf, p, 12) for p in (a, b)]
    print(f"{name} host tier: cached {r.num_cached_tokens}")


# -- GLM-5.3-Flash MTP: KDA verify rows under MTP drafting -------------------------------------------


def build_glm5_next_mtp(path, seed=2):
    """A random GLM-5.3-Flash config (tests/test_glm5_next.py, sparse: index_topk 16) plus one MTP
    layer (add_glm5_next_mtp)."""
    from tests.test_glm5_next import build

    hf = build(path, seed=seed, index_topk=16)
    add_glm5_next_mtp(path, seed)
    return hf


def add_glm5_next_mtp(path, seed=2):
    """One MTP layer under GLM-5.3-Flash's names (model.language_model.layers.<n>: pooled-DSA MLA, MoE
    + shared expert, no hyper-connections; enorm, hnorm, eh_proj, shared_head.norm), its block weights
    copied from target layer 3 (DSA + MoE) without that layer's hyper-connection tensors."""
    import json

    from safetensors.torch import load_file, save_file

    f = os.path.join(path, "model.safetensors")
    t = load_file(f)
    cfg = json.load(open(os.path.join(path, "config.json")))
    tc = cfg["text_config"]
    n, H = tc["num_hidden_layers"], tc["hidden_size"]
    src, dst = "model.language_model.layers.3.", f"model.language_model.layers.{n}."
    for k in [k for k in t if k.startswith(src) and ".hc_" not in k]:
        t[dst + k[len(src):]] = t[k].clone()
    g = torch.Generator().manual_seed(seed + 50)
    t[dst + "enorm.weight"] = 1 + 0.1 * torch.randn(H, generator=g)
    t[dst + "hnorm.weight"] = 1 + 0.1 * torch.randn(H, generator=g)
    t[dst + "shared_head.norm.weight"] = 1 + 0.1 * torch.randn(H, generator=g)
    t[dst + "eh_proj.weight"] = torch.randn(H, 2 * H, generator=g) * (2 * H) ** -0.5
    save_file(t, f, metadata={"format": "pt"})
    tc["num_nextn_predict_layers"] = 1
    json.dump(cfg, open(os.path.join(path, "config.json"), "w"))


def glm5_next_reference_drafts(model, tokens, k):
    """MTP drafts after tokens[-1] from whole-sequence forms (each HF-checked for the target's own
    layers in tests/test_glm5_next.py): the target's hidden after its final norm (models/hybrid.py
    final), then per pass x = eh_proj(cat(enorm(embed(next tokens)), hnorm(hidden))), the MTP layer as a
    plain residual block (MLA in its sequence form, the hybrid's clamped MoE), shared_head.norm, the
    target's lm_head; a later pass is fed the normed MTP hidden (vLLM v0.30.0 glm5next/nvidia/mtp.py)."""
    from kiln.models import hybrid
    from kiln.models import mla as _mla
    from kiln.models.decoder import rms_norm

    eps = model.cfg.rms_norm_eps
    with torch.no_grad():
        ids = torch.tensor(tokens[:-1])
        T = ids.shape[0]
        pos = torch.arange(T)
        h, seq = hybrid.hidden_in(model, ids, None), {}
        for layer in model.layers:
            h = hybrid.layer(model, layer, h, pos, None, None, None, None, seq)
        hid = list(hybrid.final(model, h))
        nxt, drafts, L = list(tokens[1:]), [], model.mtp
        for _ in range(k):
            e = model._embed(torch.tensor(nxt))
            x = torch.cat([rms_norm(e, model.mtp_enorm, eps), rms_norm(torch.stack(hid), model.mtp_hnorm, eps)], -1)
            x = torch.nn.functional.linear(x, model.mtp_eh)
            p = torch.arange(x.shape[0])
            x = x + _mla.reference(model, L, rms_norm(x, L.in_norm, eps), p, {})
            x = x + hybrid._mlp(model, L, rms_norm(x, L.post_norm, eps))
            hm = rms_norm(x[-1:], model.mtp_norm, eps)
            d = int(model._head(hm)[0].argmax())
            drafts.append(d)
            nxt.append(d)
            hid.append(hm[0])
    return drafts


@pytest.fixture(scope="module")
def glm_mtp(tmp_path_factory):
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    path = str(tmp_path_factory.mktemp("glm5_next_mtp"))
    build_glm5_next_mtp(path)
    return path


def test_glm5_next_mtp_drafts_match_the_reference(glm_mtp):
    """After every step (chunked prefill, verifies of every acceptance length through the KDA layers'
    per-position state rows) each request's drafts equal the whole-sequence reference's."""
    from tests.test_glm5_next import engine as eng_, prompts as mk

    eng = eng_(glm_mtp, max_num_seqs=2, max_prefill_tokens=16, spec_method="mtp", spec_k=2)
    assert eng.model.mtp is not None and eng.model.mtp.plain and eng.model.mtp.moe
    reqs = [eng.add_request(p, sp(12)) for p in mk(48, (9, 21))]
    checked = 0
    while eng.has_work():
        eng.step()
        for r in reqs:
            if r.mtp_draft and r.status.value == "running":
                assert r.mtp_draft == glm5_next_reference_drafts(eng.model, r.token_ids, 2), (r.rid, len(r.token_ids))
                checked += 1
    assert checked >= 8
    print(f"glm5_next mtp: {checked} draft checks, accepted {eng.spec_accepted}/{eng.spec_proposed}")


def test_glm5_next_mtp_speculation_and_dp_attention(glm_mtp):
    """MTP speculation on GLM-5.3-Flash's architecture equals plain greedy decoding, with the prefix
    cache on, at tp=1 and under DP attention (tp=2, two groups: per-group state rows and checkpoints)."""
    from tests.test_glm5_next import engine as eng_, prompts as mk

    (system,) = mk(49, (18,))
    ps = [system + u for u in mk(50, (3, 6, 4))]
    want = [r.output_ids for r in eng_(glm_mtp, max_num_seqs=3, prefix_caching=False).generate(ps, sp(12))]
    one = eng_(glm_mtp, max_num_seqs=3, spec_method="mtp", spec_k=2)
    assert [r.output_ids for r in run(one, [ps[:1], ps[1:]], n=12)] == want and one.spec_proposed > 0
    dp = eng_(glm_mtp, max_num_seqs=4, spec_method="mtp", spec_k=2, tp=2, dp_attention=2)
    try:
        reqs = run(dp, [ps[:1], ps[1:]], n=12)
        assert [r.output_ids for r in reqs] == want and dp.spec_proposed > 0
    finally:
        dp.close()


def test_glm5_next_mtp_with_sequence_parallel_streams(glm_mtp, monkeypatch):
    """Sequence-parallel prefill streams (KILN_PREFILL_SP, models/decoder.py prefill_sp_enabled) stay on with
    an MTP head: the MTP graph after a prefill chunk takes this rank's rows of the chunk's last hidden state and
    gathers their final norms itself (models/decoder.py _mtp_pass sp_onehot). Under DP attention (tp=4, two
    groups, piecewise) and plain TP (tp=2): the drafts after every step equal those of the same engine with
    the streams replicated (KILN_PREFILL_SP=0), and the output equals plain greedy decoding. They were off
    with MTP before (a ~20% prefill loss on the trn1 sweep, docs/price-performance.md)."""
    from tests.test_glm5_next import engine as eng_, prompts as mk

    ps = mk(51, (13, 40, 27))
    want = [r.output_ids for r in eng_(glm_mtp, max_num_seqs=3, max_prefill_tokens=16).generate(ps, sp(10))]

    def go(flag, **kw):
        monkeypatch.setenv("KILN_PREFILL_SP", flag)
        eng = eng_(glm_mtp, max_num_seqs=4, max_prefill_tokens=16, spec_method="mtp", spec_k=2, **kw)
        try:
            on = eng.runner.model.prefill_sp
            reqs = [eng.add_request(p, sp(10)) for p in ps]
            drafts = []
            while eng.has_work():
                eng.step()
                drafts += [(i, len(r.token_ids), tuple(r.mtp_draft)) for i, r in enumerate(reqs) if r.mtp_draft]
            return on, [r.output_ids for r in reqs], drafts, (eng.spec_proposed, eng.spec_accepted)
        finally:
            eng.close()

    for kw in (dict(tp=4, dp_attention=2, piecewise=True, piecewise_group=2), dict(tp=2)):
        on, ids, drafts, acc = go("1", **kw)
        off, ids0, drafts0, acc0 = go("0", **kw)
        print(f"tp={kw['tp']} dp_attention={kw.get('dp_attention', 1)}: sequence-parallel {on} / {off}, "
              f"{len(drafts)} drafts, accepted {acc[1]}/{acc[0]} (replicated {acc0[1]}/{acc0[0]})")
        assert on and not off
        assert drafts == drafts0 and len(drafts) >= 6, kw
        assert ids == ids0 == want, kw


# -- the real checkpoint ----------------------------------------------------------------------------

REAL = os.environ.get("KILN_TEST_LINEAR_MODEL")  # e.g. Qwen/Qwen3.5-0.8B


@pytest.mark.skipif(not REAL, reason="set KILN_TEST_LINEAR_MODEL (e.g. Qwen/Qwen3.5-0.8B) to run")
def test_real_checkpoint_prefix_cache_speculation_and_jump_forward():
    """Qwen3.5-0.8B: a shared system prompt, a second turn, n-gram and suffix speculation, and
    jump-forward under a JSON schema (with logprobs: the newest token is never re-scored, so the
    state never steps back), all equal to the same engine without the feature. One engine at a time
    (host memory)."""
    import gc
    import json

    from kiln.engine.grammar import GrammarSpec
    from kiln.engine.request import SamplingParams
    from kiln.models.loader import resolve_model_path
    from tests.test_grammar import chat_ids, run_steps, schema

    path = resolve_model_path(REAL)
    kw = dict(max_prefill_tokens=64, max_num_seqs=4, num_pages=512, max_model_len=512, page_size=8)
    system = ("You are a careful assistant. Answer in one short sentence, and repeat the key word of the question "
              "at the end of your answer.\n")
    qs = ["What is the capital of France?", "Name a prime number larger than ten.", "What color is the sky?",
          "List: apple, banana, cherry, apple, banana, cherry, apple,"]
    p = SamplingParams(max_new_tokens=32, ignore_eos=True)
    sch = schema("city_name", "country_code", "population_estimate")
    gp = SamplingParams(max_new_tokens=60, logprobs=2, grammar=GrammarSpec("json_schema", json.dumps(sch)))

    off = engine(path, prefix_caching=False, **kw)
    tok = off.tokenizer
    ps = [tok(system + q)["input_ids"] for q in qs]
    want = [r.output_ids for r in off.generate(ps, p)]
    turn2 = ps[0] + want[0] + tok(" And of Germany?")["input_ids"]
    want2 = off.generate([turn2], p)[0].output_ids
    gprompt = chat_ids(off, "Describe Osaka as a JSON object.")
    (gref,), gref_steps = run_steps(off, [gprompt], [gp], jump=False)
    del off
    gc.collect()

    on = engine(path, state_track_interval=16, state_checkpoints=16, **kw)
    reqs = [on.generate([x], p)[0] for x in ps]
    assert [r.output_ids for r in reqs] == want
    assert all(r.num_cached_tokens > 0 for r in reqs[2:])
    b = on.generate([turn2], p)[0]
    assert b.output_ids == want2 and b.num_cached_tokens > len(ps[0])
    (jf,), jf_steps = run_steps(on, [gprompt], [gp], jump=True)
    assert jf.output_ids == gref.output_ids and jf.num_forced > 0 and jf_steps < gref_steps
    for (x, tx, _), (y, ty, _) in zip(jf.logprobs, gref.logprobs):
        assert abs(x - y) < 1e-3 and tx == ty
    print(f"{REAL}: cached {[r.num_cached_tokens for r in reqs]}, turn 2 {b.num_cached_tokens}; "
          f"jump-forward {gref_steps} -> {jf_steps} steps")
    del on
    gc.collect()
    for method in ("ngram", "suffix"):
        spec = engine(path, prefix_caching=False, spec_method=method, spec_k=4, **kw)
        got = [r.output_ids for r in spec.generate(ps, p)]
        assert got == want, method
        print(f"{REAL} {method}: accepted {spec.spec_accepted}/{spec.spec_proposed}")
        del spec
        gc.collect()
