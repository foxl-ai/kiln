"""Expert-parallel load balancing with redundant expert slots (models/eplb.py, KILN_EP_REDUNDANT): the placement,
the routing-id remap, and engines whose ranks also hold copies of other ranks' experts giving the tokens and
logprobs of the plain expert-parallel layout (CPU tensor parallelism over gloo), with the statistics recorder on
and across a rebalance."""

import pytest
import torch

from kiln.models import eplb

LOGPROB_TOL = 1e-4  # fp32; the copies hold the same weights, only the ranks' summation order moves


def test_replicas_take_the_hottest_experts_off_their_rank():
    E, tp, s = 288, 32, 1
    g = torch.Generator().manual_seed(0)
    load = torch.rand(E, generator=g) * 10
    load[[20, 21, 100, 287]] = torch.tensor([4000.0, 3900.0, 3800.0, 3700.0])
    extra = eplb.replicas(load, tp, s)
    assert extra == eplb.replicas(load.clone(), tp, s)  # deterministic
    assert len(extra) == tp * s
    El = E // tp
    for r in range(tp):
        for e in extra[r * s:(r + 1) * s]:
            assert not (r * El <= e < (r + 1) * El), (r, e)  # never on the primary's own rank
    for e in (20, 21, 100, 287):  # the four hot experts get most of the 32 copies
        assert extra.count(e) >= 6
    # no two copies of one expert on one rank
    for r in range(tp):
        assert len(set(extra[r * s:(r + 1) * s])) == s
    assert eplb.default_extra(E, tp, 2)[:4] == [9, 10, 18, 19]


def test_sticky_placement_keeps_copies_under_a_stationary_load():
    """A rebalance on counts from the same distribution keeps (almost) every copy in its slot (prev=), and a
    changed hot set still moves the copies to it."""
    E, tp, s = 288, 32, 1
    g = torch.Generator().manual_seed(3)
    base = torch.rand(E, generator=g) * 10
    base[[20, 21, 100, 287]] += torch.tensor([4000.0, 3900.0, 3800.0, 3700.0])
    first = eplb.replicas(base, tp, s)
    assert eplb.replicas(base, tp, s, prev=first) == first
    noisy = base * (1 + 0.05 * torch.randn(E, generator=g))
    fresh, sticky = eplb.replicas(noisy, tp, s), eplb.replicas(noisy, tp, s, prev=first)
    moved = lambda a, b: sum(x != y for x, y in zip(a, b))  # noqa: E731
    assert moved(first, sticky) <= moved(first, fresh)
    assert moved(first, sticky) <= 4, (moved(first, sticky), moved(first, fresh))
    assert eplb.busiest_load(noisy, sticky, tp, s) <= 1.05 * eplb.busiest_load(noisy, fresh, tp, s)
    shifted = base.clone()
    shifted[[20, 21, 100, 287]] = 5.0
    shifted[[40, 41]] = 8000.0
    new = eplb.replicas(shifted, tp, s, prev=first)
    assert new.count(40) + new.count(41) >= 24  # the copies follow the new hot experts


def test_remap_sends_every_pair_to_exactly_one_copy():
    E, tp, s = 288, 32, 2
    g = torch.Generator().manual_seed(1)
    load = torch.rand(E, generator=g)
    load[:5] += 100
    extra = eplb.replicas(load, tp, s)
    ids, mp = eplb.tables(extra, E, tp, s)
    topi = torch.randint(0, E, (300, 8), generator=g)
    topi[:, 0] = 3  # a replicated expert in every row
    phys = eplb.remap(topi, ids, mp)
    copies = {e: [e] + [E + i for i, x in enumerate(extra) if x == e] for e in range(E)}
    lmaps = [eplb.physical_lmap(E, tp, s, r).view(-1) for r in range(tp)]
    El = E // tp
    for t in range(topi.shape[0]):
        for j in range(topi.shape[1]):
            e, p = int(topi[t, j]), int(phys[t, j])
            assert p in copies[e]
            assert p == copies[e][(t % eplb.RMAX) % len(copies[e])]
            owners = [r for r in range(tp) if int(lmaps[r][p]) < El + s]
            assert len(owners) == 1  # exactly one rank computes the pair
            r = owners[0]
            slot = int(lmaps[r][p])
            held = list(range(r * El, (r + 1) * El)) + extra[r * s:(r + 1) * s]
            assert held[slot] == e  # and that slot holds the pair's expert
    # copies share expert 3's pairs evenly over the row classes
    n3 = len(copies[3])
    assert n3 > 1
    got = torch.bincount(torch.tensor([copies[3].index(int(p)) for p in phys[:, 0]]), minlength=n3)
    assert int(got.max() - got.min()) <= topi.shape[0] // eplb.RMAX + 1
    # spread=False: the identity
    ids0, mp0 = eplb.tables(extra, E, tp, s, spread=False)
    assert torch.equal(eplb.remap(topi, ids0, mp0), topi)
    # the padding expert E stays E (no rank holds it)
    assert int(eplb.remap(torch.full((4, 2), E), ids, mp).max()) == E
    assert all(int(m[E + tp * s]) == El + s for m in lmaps)


def test_counts_and_load_file(tmp_path):
    topi = torch.tensor([[0, 2], [2, 3], [2, 0]])
    assert eplb.counts(topi, 4).tolist() == [2.0, 0.0, 3.0, 1.0]
    p = tmp_path / "r.pt"
    torch.save({"topi": {3: [topi, topi[:1]]}, "names": ["a", "b"]}, p)
    assert eplb.load_file(str(p), 4)[3].tolist() == [3.0, 0.0, 4.0, 1.0]
    torch.save({"load": {5: [1.0, 2.0, 3.0, 4.0]}}, p)
    assert eplb.load_file(str(p), 4)[5].tolist() == [1.0, 2.0, 3.0, 4.0]


def _run(path, tp, monkeypatch, prompts, sp, redundant, init=None, record=False, rebalance_after=None, **kw):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    # Count the pairs remap sends to a redundant slot (physical id >= E) on rank 0, so the comparison below is known
    # to exercise the copies.
    real, hits = eplb.remap, []

    def counting(topi, ids, mp):
        out = real(topi, ids, mp)
        if int(ids.max()) >= 0:
            hits.append(int((out > topi).sum()))
        return out

    monkeypatch.setattr(eplb, "remap", counting)

    monkeypatch.setenv("KILN_MOE_EP", "1")
    monkeypatch.setenv("KILN_EP_REDUNDANT", str(redundant))
    monkeypatch.setenv("KILN_EPLB_RECORD", "1" if record else "0")
    if init:
        monkeypatch.setenv("KILN_EPLB_INIT", init)
    else:
        monkeypatch.delenv("KILN_EPLB_INIT", raising=False)
    base = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256, max_num_seqs=3,
                max_model_len=256, max_prefill_tokens=16, tp=tp)
    base.update(kw)
    eng = LLMEngine(EngineConfig(**base))
    try:
        model = eng.runner.model
        moe = model.ep_layers()
        assert moe and all(getattr(l, "ep_s", 0) == redundant for l in moe)
        if rebalance_after is not None:
            reqs = eng.generate(prompts[:rebalance_after], sp)
            before = {i: list(l.ep_extra_all) for i, l in enumerate(moe)}
            moved = eng.runner.eplb_rebalance()
            after = {i: list(l.ep_extra_all) for i, l in enumerate(moe)}
            reqs = reqs + eng.generate(prompts[rebalance_after:], sp)
            assert moved and before != after, "the recorded counts should move some copies"
        else:
            reqs = eng.generate(prompts, sp)
        stats = [l.ep_stats.clone() for l in moe] if record else None
        if redundant:
            assert sum(hits) > 0, "no pair went to a redundant slot on rank 0"
        return [r.output_ids for r in reqs], [[x[0] for x in r.logprobs] for r in reqs], stats
    finally:
        eng.close()


def _compare(name, a, b):
    (ids, lp, _), (ids0, lp0, _) = a, b
    err = max(abs(x - y) for p, q in zip(lp, lp0) for x, y in zip(p, q))
    print(f"{name}: tokens {'equal' if ids == ids0 else 'DIFFER'}, max |dlogprob| {err:.2e}")
    assert ids == ids0, name
    assert err < LOGPROB_TOL, (name, err)


def _skewed_init(tmp_path, layers, E):
    """A statistics file that makes the copies hold a skewed set (expert 1 and 5 hottest)."""
    load = {l: [1.0] * E for l in layers}
    for l in layers:
        load[l][1], load[l][5] = 50.0, 30.0
    p = tmp_path / "init.pt"
    torch.save({"load": load}, p)
    return str(p)


def test_glm5_next_redundant_slots_equal_plain_expert_parallel(tmp_path, monkeypatch):
    """GLM-5.3-Flash at tp 4 with DP attention 2 and sequence-parallel prefill streams (the routing is remapped
    on each rank's own rows before the gather) and at tp 2: one redundant slot per rank, its copies chosen from a
    statistics file, gives the tokens and logprobs of plain expert parallelism; the recorder counts every rank's
    prefill pairs."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from kiln.config import ModelConfig
    from kiln.engine.request import SamplingParams
    from tests.test_glm5_next import build, prompts

    build(str(tmp_path), index_topk=16)
    cfg = ModelConfig.from_pretrained(str(tmp_path))
    init = _skewed_init(tmp_path, cfg.moe_layers, cfg.num_experts)
    ps = prompts(9, (13, 40, 27))
    sp = SamplingParams(max_new_tokens=8, ignore_eos=True, logprobs=1)
    for kw in (dict(tp=4, dp_attention=2, piecewise=True, piecewise_group=2), dict(tp=2)):
        tp = kw.pop("tp")
        ref = _run(str(tmp_path), tp, monkeypatch, ps, sp, 0, **kw)
        got = _run(str(tmp_path), tp, monkeypatch, ps, sp, 1, init=init, record=True, **kw)
        _compare(f"glm5_next tp={tp} {kw} +1 slot", got, ref)
        _compare(f"glm5_next tp={tp} {kw} +1 slot, placeholder copies",
                 _run(str(tmp_path), tp, monkeypatch, ps, sp, 1, **kw), ref)
        stats = got[2]
        assert all(float(s.sum()) > 0 for s in stats)


def test_mimo_v2_redundant_slots_equal_plain_expert_parallel(tmp_path, monkeypatch):
    """A plain MoE (no sequence-parallel streams: the ids are remapped in _moe_ep over every row) at tp 2 and 4,
    one and two redundant slots, against plain expert parallelism."""
    from kiln.engine.request import SamplingParams
    from tests.test_mimo_v2 import build_reference

    build_reference(str(tmp_path))
    # Decode pairs on the copies too: with decode v2's default (primaries only) these short prefills send no pair to
    # a copy on rank 0, and the comparison would not exercise the redundant slots.
    monkeypatch.setenv("KILN_EPLB_DECODE", "1")
    ps = [[5, 9, 11, 200, 3, 77, 12, 13, 14, 15, 16], list(range(40, 70))]
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True, logprobs=1)
    kw = dict(num_pages=128, max_num_seqs=2, max_model_len=128, max_prefill_tokens=8)
    for tp in (2, 4):
        ref = _run(str(tmp_path), tp, monkeypatch, ps, sp, 0, **kw)
        for s in (1, 2):
            _compare(f"mimo_v2 tp={tp} +{s}", _run(str(tmp_path), tp, monkeypatch, ps, sp, s, **kw), ref)


def test_rebalance_moves_copies_and_keeps_the_output(tmp_path, monkeypatch):
    """With the recorder on, a rebalance between two batches reads every rank's counts, places the copies from
    them (models/eplb.py replicas), loads the new copies' weights from the checkpoint into the redundant slots and
    rewrites the remap tables: the second batch's output equals plain expert parallelism's."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from kiln.engine.request import SamplingParams
    from tests.test_glm5_next import build, prompts

    build(str(tmp_path), index_topk=16)
    ps = prompts(11, (40, 33, 21, 38))
    sp = SamplingParams(max_new_tokens=6, ignore_eos=True, logprobs=1)
    kw = dict(dp_attention=2, piecewise=True, piecewise_group=2)
    ref = _run(str(tmp_path), 4, monkeypatch, ps, sp, 0, **kw)
    got = _run(str(tmp_path), 4, monkeypatch, ps, sp, 1, record=True, rebalance_after=2, **kw)
    _compare("glm5_next tp=4 rebalance", got, ref)


def test_periodic_rebalance_in_the_engine_equals_plain_expert_parallel(tmp_path, monkeypatch):
    """KILN_EPLB_INTERVAL: the engine starts a rebalance after that many prefill calls (loading on host threads
    beside serving) and installs it at a later step; generation across it equals plain expert parallelism's."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams
    from tests.test_glm5_next import build, prompts

    build(str(tmp_path), index_topk=16)
    ps = prompts(13, (40, 33, 21, 38))
    sp = SamplingParams(max_new_tokens=6, ignore_eos=True, logprobs=1)
    kw = dict(dp_attention=2, piecewise=True, piecewise_group=2)
    ref = _run(str(tmp_path), 4, monkeypatch, ps, sp, 0, **kw)
    monkeypatch.setenv("KILN_MOE_EP", "1")
    monkeypatch.setenv("KILN_EP_REDUNDANT", "1")
    monkeypatch.setenv("KILN_EPLB_RECORD", "1")
    monkeypatch.setenv("KILN_EPLB_INTERVAL", "3")
    monkeypatch.setenv("KILN_EPLB_MAX_REBALANCES", "1")
    monkeypatch.delenv("KILN_EPLB_INIT", raising=False)
    eng = LLMEngine(EngineConfig(model_path=str(tmp_path), device="cpu", dtype=torch.float32, page_size=4,
                                 num_pages=256, max_num_seqs=3, max_model_len=256, max_prefill_tokens=16, tp=4, **kw))
    try:
        reqs = eng.generate(ps, sp)
        got = ([r.output_ids for r in reqs], [[x[0] for x in r.logprobs] for r in reqs], None)
        assert len(eng.eplb_log) == 1, eng.eplb_log
        print("eplb_log", eng.eplb_log)
    finally:
        eng.close()
    _compare("glm5_next tp=4 periodic rebalance", got, ref)


def test_decode_replicas_follows_the_small_lane_kernel_in_use(monkeypatch):
    # KILN_EPLB_DECODE unset: decode pairs stay on the primaries with decode v2 and later, spread with v1.
    # It must read the kernel's own SMALL_V: an env fallback of its own drifted from moe_ep's default.
    from kiln.kernels import moe_ep

    monkeypatch.delenv("KILN_EPLB_DECODE", raising=False)
    monkeypatch.delenv("KILN_MOE_EP_SMALL_V", raising=False)
    monkeypatch.setattr(moe_ep, "SMALL_V", moe_ep.SMALL_V_DEFAULT)
    assert moe_ep.SMALL_V_DEFAULT >= 2 and eplb.decode_replicas() is False
    monkeypatch.setattr(moe_ep, "SMALL_V", 1)
    assert eplb.decode_replicas() is True
    monkeypatch.setenv("KILN_EPLB_DECODE", "0")
    assert eplb.decode_replicas() is False
