"""DP attention (EngineConfig.dp_attention, engine/dp.py, models/decoder.py DecoderForCausalLM: DP
attention) on CPU over gloo: the token mixers of each group of tp / N ranks serve their own requests
from their own KV pool and state rows, the MLP / experts run over every group's tokens.

Every case runs the same prompts at tp=4 with dp_attention 2 and 4 (and tp=2 dp_attention 2) and
compares greedy tokens (exactly) and chosen-token logprobs (within fp32 reduction-order noise) with
tp=1. The prompt counts and lengths are uneven across groups, max_num_seqs leaves some groups idle
in some steps, chunked prefill pairs chunks of different lengths in one call, and the preemption
case runs a pool too small for its requests. Each model's cases between them take every forward:
decode, chunked prefill, verify, MTP drafts, piecewise layer groups and overlap.
"""

import json
import os

import pytest
import torch

from tests.test_attention_tp import _Repeat, prompts
from tests.test_architectures import build as build_arch
from tests.test_linear_attn import build_kda, build_qwen3_5
from tests.test_mimo_v2 import build_reference as build_mimo
from tests.test_mla import build as build_mla

LOGPROB_TOL = 1e-4  # fp32; measured differences are printed


def run(path, ps, sp, proposer=None, **kw):
    """(output ids, chosen logprobs, groups used, rank-0 KV / state shapes, (proposed, accepted),
    preemptions) of one engine run."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    base = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256, max_num_seqs=4,
                max_model_len=256, max_prefill_tokens=8)
    base.update(kw)
    eng = LLMEngine(EngineConfig(**base))
    try:
        if proposer is not None and kw.get("spec_method") not in (None, "mtp"):
            eng.proposer = proposer
        reqs = eng.generate(ps, sp)
        shapes = [tuple(k.shape[1:]) for k in eng.runner.k_caches]
        if eng.runner.state is not None:
            shapes += [tuple(s.shape[1:]) for s in eng.runner.state.rec]
        spec = (eng.spec_proposed, eng.spec_accepted) if kw.get("spec_method") else None
        return ([r.output_ids for r in reqs], [[x[0] for x in r.logprobs] for r in reqs],
                sorted({r.dp_group for r in reqs}), shapes, spec, eng.scheduler.num_preemptions)
    finally:
        eng.close()


def check(path, ps, variants, new_tokens=10, proposer=None, preempt=False, **common):
    """variants: (tp, dp_attention, engine kwargs), each compared with tp=1 under the same kwargs."""
    from kiln.engine.request import SamplingParams

    sp = SamplingParams(max_new_tokens=new_tokens, ignore_eos=True, logprobs=1)
    refs = {}
    for tp, dp, kw in variants:
        key = json.dumps(kw, sort_keys=True)
        if key not in refs:
            refs[key] = run(path, ps, sp, proposer() if proposer else None, **common, **kw)
        want_ids, want_lp, _, _, want_spec, _ = refs[key]
        ids, lp, groups, shapes, spec, pre = run(path, ps, sp, proposer() if proposer else None, tp=tp,
                                                 dp_attention=dp, **common, **kw)
        err = max(abs(a - b) for x, y in zip(lp, want_lp) for a, b in zip(x, y))
        print(f"{os.path.basename(path)} tp={tp} dp_attention={dp} {kw}: tokens {'equal' if ids == want_ids else 'DIFFER'}, "
              f"max |dlogprob| {err:.2e}, groups {groups}, preemptions {pre}, rank-0 cache / state shapes {shapes}")
        assert ids == want_ids, (tp, dp, kw)
        assert err < LOGPROB_TOL, (tp, dp, kw, err)
        assert groups == list(range(min(dp, len(ps)))), groups  # every group served requests
        if preempt:
            assert pre > 0
        if want_spec is not None:
            # Drafts were verified in both runs. The counts may differ: each group's prefill budget
            # is max_prefill_tokens // dp_attention, so chunks end elsewhere and a request whose last
            # prompt token is left over decodes (and drafts) one step earlier.
            assert spec[0] > 0 and want_spec[0] > 0, (spec, want_spec)


# -- configuration and placement --------------------------------------------------------------------


def test_degrees_and_refusals(tmp_path):
    from kiln.config import EngineConfig, ModelConfig
    from kiln.engine.engine import resolve_attention_tp

    build_arch("qwen3", str(tmp_path))
    m = ModelConfig.from_pretrained(str(tmp_path))
    ec = lambda **kw: EngineConfig(model_path=str(tmp_path), **kw)  # noqa: E731
    assert resolve_attention_tp(m, ec(tp=4, dp_attention=2)) == 2
    assert resolve_attention_tp(m, ec(tp=4, dp_attention=4)) == 1
    assert resolve_attention_tp(m, ec(tp=4, dp_attention=2, attention_tp=2)) == 2
    with pytest.raises(ValueError, match="does not divide"):
        resolve_attention_tp(m, ec(tp=4, dp_attention=3))
    with pytest.raises(ValueError, match="runs attention TP 2"):
        resolve_attention_tp(m, ec(tp=4, dp_attention=2, attention_tp=4))
    e = ec(tp=4, dp_attention=4, max_num_seqs=9)
    assert e.group_max_num_seqs == 3 and e.resolved_decode_batch_buckets() == (1, 2, 3)
    assert e.group_max_prefill_tokens == 128 and e.resolved_prefill_token_buckets() == (32, 64, 128)  # 512 // 4
    with pytest.raises(ValueError, match="below one token per DP-attention group"):
        ec(tp=4, dp_attention=4, max_prefill_tokens=3).group_max_prefill_tokens


def test_placement_balances_tokens_and_follows_prefixes():
    """SGLang TOTAL_TOKENS placement (fewest tokens, then requests), charged only for the tokens a
    group's radix cache does not already hold."""
    from kiln.engine.dp import DPScheduler
    from kiln.engine.kv_pool import PagePool
    from kiln.engine.radix_cache import RadixCache
    from kiln.engine.request import Request, SamplingParams
    from kiln.engine.scheduler import Scheduler, SchedulerConfig

    cfg = SchedulerConfig(page_size=4, max_num_seqs=4, max_prefill_tokens=64, max_model_len=256)
    pools = [PagePool(64) for _ in range(2)]
    dp = DPScheduler([Scheduler(cfg, p, RadixCache(p, 4)) for p in pools])
    sp = SamplingParams(max_new_tokens=4)
    a = Request("a", list(range(40)), sp)
    b = Request("b", list(range(100, 110)), sp)
    c = Request("c", list(range(200, 210)), sp)
    for r in (a, b, c):
        dp.add(r)
    assert (a.dp_group, b.dp_group, c.dp_group) == (0, 1, 1)  # c: group 1 holds 10 tokens, group 0 40
    # A prompt sharing a's first 32 tokens costs group 0 only its 9 new ones once they are cached.
    dp.groups[0].radix.insert(list(range(32)), pools[0].alloc(8))
    d = Request("d", list(range(32)) + [900] * 9, sp)
    dp.add(d)
    assert d.dp_group == 0
    plan = dp.schedule()
    assert {s.req.rid for s in plan.prefills} == {"a", "b", "c", "d"}
    assert dp.groups[0].running == [a, d] and dp.groups[1].running == [b, c]


def test_placement_follows_queued_prefixes_and_free_slots():
    """Cache-aware placement: a prefix a group will compute for a request queued there counts as held,
    a group's load is what its requests were charged, and a full group (max_num_seqs live) comes after
    every group with a free slot."""
    from kiln.engine.dp import DPScheduler
    from kiln.engine.kv_pool import PagePool
    from kiln.engine.radix_cache import RadixCache
    from kiln.engine.request import Request, SamplingParams
    from kiln.engine.scheduler import Scheduler, SchedulerConfig

    cfg = SchedulerConfig(page_size=4, max_num_seqs=3, max_prefill_tokens=64, max_model_len=256)
    pools = [PagePool(64) for _ in range(2)]
    dp = DPScheduler([Scheduler(cfg, p, RadixCache(p, 4)) for p in pools])
    sp = SamplingParams(max_new_tokens=4)
    system = list(range(1000, 1032))
    u = Request("u", list(range(2000, 2040)), sp)
    reqs = [Request(f"r{i}", system + [i] * 8, sp) for i in range(4)]
    for r in [u] + reqs:
        dp.add(r)
    # u -> 0 and r0 -> 1 (40 tokens each); r1 shares r0's 32 queued tokens: group 1 is charged 8 (48)
    # against 40 more in group 0 (80); r2 likewise (56 against 80), and group 1 is then full (3 live),
    # so r3 goes to group 0 although group 1 would have been charged less.
    assert [r.dp_group for r in [u] + reqs] == [0, 1, 1, 1, 0]
    assert [r.dp_charge for r in [u] + reqs] == [40, 40, 8, 8, 40]


# -- parity with tp=1 -------------------------------------------------------------------------------


def test_qwen3_dense(tmp_path):
    """4 query / 2 KV heads: dp_attention 2 at tp=4 is attention TP 2 (one KV head per rank, each
    group its own requests), dp_attention 4 attention TP 1. 5 prompts over 2 groups (3 / 2) and
    over 4 groups (2 / 1 / 1 / 1)."""
    build_arch("qwen3", str(tmp_path))
    ps = prompts(1, (5, 19, 30, 11)) + [[7, 8, 9] * 5]
    check(str(tmp_path), ps, [(4, 2, dict(piecewise=True, piecewise_group=1)), (4, 4, dict(overlap=True)),
                              (2, 2, {})], max_num_seqs=5)


def test_qwen3_dense_preemption(tmp_path):
    """Each group's pool (11 usable pages of 4 tokens) is too small for its requests: groups
    preempt their own youngest and recompute it from their own radix cache."""
    build_arch("qwen3", str(tmp_path))
    ps = prompts(9, (13, 17, 21, 9))
    check(str(tmp_path), ps, [(4, 2, {}), (4, 2, dict(overlap=True))], new_tokens=14, preempt=True,
          num_pages=12, admission="eager")


def test_qwen3_dense_speculative_verify(tmp_path):
    """Extend graphs (Q = k + 1 rows per sequence) with every group's sequences."""
    build_arch("qwen3", str(tmp_path))
    check(str(tmp_path), prompts(2, (9, 23, 14)), [(t, d, dict(spec_method="ngram", spec_k=3))
                                                   for t, d in ((4, 2), (4, 4), (2, 2))],
          new_tokens=14, proposer=_Repeat, max_num_seqs=3)


def test_mimo_v2(tmp_path):
    """Full attention (4 query / 2 KV heads) and sliding-window attention with sinks (4 / 4,
    window 8: window tables per group), Dk 24 != Dv 16, value scale, MoE over every group's tokens."""
    build_mimo(str(tmp_path), "fused_qkv")
    ps = prompts(3, (5, 19, 30))
    check(str(tmp_path), ps, [(4, 2, dict(piecewise=True, piecewise_group=2)), (4, 4, dict(overlap=True)),
                              (2, 2, {})], max_num_seqs=3)


def test_mimo_v2_mtp(tmp_path):
    """MTP drafts (a sliding-window layer of its own, its KV per group) and their verify."""
    from tests.test_mtp import add_mtp

    build_mimo(str(tmp_path), "split")
    add_mtp(str(tmp_path), "split")
    check(str(tmp_path), prompts(4, (11, 23, 7)), [(t, d, dict(spec_method="mtp", spec_k=2))
                                                   for t, d in ((4, 2), (4, 4), (2, 2))],
          new_tokens=12, max_num_seqs=3)


@pytest.mark.parametrize("name", ["deepseek_v3", "glm_moe_dsa"])
def test_mla(tmp_path, name):
    """MLA: each group's latent cache holds only its requests (the case DP attention exists for).
    GLM's index_topk 12 makes the DSA top-k and IndexShare bite, its scratch per group."""
    build_mla(name, str(tmp_path), **(dict(index_topk=12) if name == "glm_moe_dsa" else {}))
    ps = prompts(5, (9, 30, 41))
    check(str(tmp_path), ps, [(4, 2, dict(piecewise=True, piecewise_group=1)), (4, 4, dict(overlap=True)),
                              (2, 2, dict(spec_method="ngram", spec_k=3))], proposer=_Repeat, max_prefill_tokens=16,
          max_num_seqs=3)


def test_qwen3_5_gdn(tmp_path):
    """Gated DeltaNet (2 k / 4 v heads, so attention TP at most 2: tp=4 dp_attention 2 or 4) with
    gated full attention; per-group recurrent-state rows."""
    build_qwen3_5(str(tmp_path))
    ps = prompts(6, (5, 19, 61))
    check(str(tmp_path), ps, [(4, 2, dict(piecewise=True, piecewise_group=2)), (4, 4, dict(overlap=True)),
                              (2, 2, {})], max_num_seqs=3)


def test_kda(tmp_path):
    """Kimi Delta Attention, 4 heads; 3 prompts over 2 groups with one row each per group."""
    build_kda(str(tmp_path))
    ps = prompts(7, (7, 45, 20))
    check(str(tmp_path), ps, [(4, 2, dict(piecewise=True)), (4, 4, {}), (2, 2, dict(overlap=True))], max_num_seqs=2)


def test_glm5_next(tmp_path):
    """GLM-5.3-Flash truncated: KDA and pooled-DSA NoPE MLA layers, mHC streams (computed over
    every group's rows), clamped SwiGLU."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from tests.test_glm5_next import build

    build(str(tmp_path), index_topk=16)
    ps = prompts(8, (11, 41, 26))
    check(str(tmp_path), ps, [(4, 2, dict(piecewise=True, piecewise_group=2)), (4, 4, {}), (2, 2, {})],
          max_prefill_tokens=16, max_num_seqs=3)


def test_glm5_next_sequence_parallel_streams(tmp_path, monkeypatch):
    """Sequence-parallel prefill streams (models/decoder.py prefill_sp_enabled, KILN_PREFILL_SP): each
    rank keeps its own rows of the mHC streams between prefill layers. The per-row arithmetic is the
    replicated streams' and the gather adds only zeros, but a CPU fp32 matmul over a few rows can round
    differently from the same rows inside a bigger one (BLAS blocking): greedy tokens equal those of
    KILN_PREFILL_SP=0 and chosen-token logprobs agree within fp32 noise (measured 2026-10-04: about
    1e-6), at DP attention and at plain TP; and the runner keeps them on only where every prefill
    bucket's rows divide over tp. Both with each rank routing its own rows (KILN_SP_ROUTE=1, the default),
    block outputs reduce-scattered (KILN_SP_RS) and the token mixers' collectives inside their attention group
    (KILN_SP_GROUP, which the engine's attention groups enable under DP attention), and with all three off."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams
    from tests.test_glm5_next import build

    build(str(tmp_path), index_topk=16)
    ps = prompts(9, (13, 40, 27))
    sp = SamplingParams(max_new_tokens=8, ignore_eos=True, logprobs=1)

    def go(flag, route="1", rs="1", grp="1", **kw):
        monkeypatch.setenv("KILN_PREFILL_SP", flag)
        monkeypatch.setenv("KILN_SP_ROUTE", route)
        monkeypatch.setenv("KILN_SP_RS", rs)
        monkeypatch.setenv("KILN_SP_GROUP", grp)
        base = dict(model_path=str(tmp_path), device="cpu", dtype=torch.float32, page_size=4, num_pages=256,
                    max_num_seqs=3, max_model_len=256, max_prefill_tokens=16)
        base.update(kw)
        eng = LLMEngine(EngineConfig(**base))
        try:
            on = eng.runner.model.prefill_sp
            if flag == "1" and kw.get("dp_attention", 1) > 1:  # the token mixers' group collectives can run
                assert eng.runner.model._sp_grp_ok()
            reqs = eng.generate(ps, sp)
            return on, [r.output_ids for r in reqs], [[x[0] for x in r.logprobs] for r in reqs]
        finally:
            eng.close()

    for kw in (dict(tp=4, dp_attention=2, piecewise=True, piecewise_group=2), dict(tp=2)):
        on, ids, lp = go("1", **kw)
        off, ids0, lp0 = go("0", **kw)
        err = max(abs(a - b) for x, y in zip(lp, lp0) for a, b in zip(x, y))
        print(f"tp={kw['tp']} dp_attention={kw.get('dp_attention', 1)}: sequence-parallel {on} / {off}, tokens "
              f"{'equal' if ids == ids0 else 'DIFFER'}, max |dlogprob| {err:.2e}")
        assert on and not off
        assert ids == ids0, kw
        assert err < LOGPROB_TOL, (kw, err)
        # KILN_SP_ROUTE=0: every rank routes all gathered rows (models/hybrid.py _ffn) instead of its own;
        # KILN_SP_RS=0: block outputs all-reduced and this rank's rows taken (on the host the reduce-scatter
        # is exactly that, so this checks the plumbing; the device check is tools/probe_rs_reload.py);
        # KILN_SP_GROUP=0: the token mixers gather and reduce over the world instead of their attention group
        _, ids1, lp1 = go("1", route="0", rs="0", grp="0", **kw)
        err1 = max(abs(a - b) for x, y in zip(lp1, lp0) for a, b in zip(x, y))
        print(f"  KILN_SP_ROUTE=0 KILN_SP_RS=0 KILN_SP_GROUP=0: tokens {'equal' if ids1 == ids0 else 'DIFFER'}, max |dlogprob| {err1:.2e}")
        assert ids1 == ids0, kw
        assert err1 < LOGPROB_TOL, (kw, err1)
    on, _, _ = go("1", tp=4, dp_attention=2, prefill_token_buckets=(3,), max_prefill_tokens=6)  # 2 x 3 rows over 4 ranks
    assert not on


def test_glm5_next_sequence_parallel_decode_streams(tmp_path, monkeypatch):
    """Sequence-parallel decode streams (models/decoder.py decode_sp_enabled, KILN_DECODE_SP=1): each rank keeps its
    own N B / tp rows of a decode call's streams, each block gathers them and reduce-scatters its output, the post
    graph gathers the final hidden state: greedy tokens equal the replicated decode streams' (KILN_DECODE_SP=0) and
    chosen-token logprobs agree within fp32 noise, at DP attention and at plain TP, with prefill SP on in both; and
    the runner turns them off when a decode bucket's rows do not divide over tp."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams
    from tests.test_glm5_next import build

    build(str(tmp_path), index_topk=16)
    ps = prompts(9, (13, 40, 27))
    sp = SamplingParams(max_new_tokens=8, ignore_eos=True, logprobs=1)

    def go(flag, **kw):
        monkeypatch.setenv("KILN_PREFILL_SP", "1")
        monkeypatch.setenv("KILN_DECODE_SP", flag)
        base = dict(model_path=str(tmp_path), device="cpu", dtype=torch.float32, page_size=4, num_pages=256,
                    max_num_seqs=3, max_model_len=256, max_prefill_tokens=16)
        base.update(kw)
        eng = LLMEngine(EngineConfig(**base))
        try:
            on = eng.runner.model.decode_sp
            reqs = eng.generate(ps, sp)
            return on, [r.output_ids for r in reqs], [[x[0] for x in r.logprobs] for r in reqs]
        finally:
            eng.close()

    for kw in (dict(tp=4, dp_attention=2, piecewise=True, piecewise_group=2, decode_batch_buckets=(2,)),
               dict(tp=2, decode_batch_buckets=(4,))):
        on, ids, lp = go("1", **kw)
        off, ids0, lp0 = go("0", **kw)
        err = max(abs(a - b) for x, y in zip(lp, lp0) for a, b in zip(x, y))
        print(f"tp={kw['tp']} dp_attention={kw.get('dp_attention', 1)}: sequence-parallel decode {on} / {off}, tokens "
              f"{'equal' if ids == ids0 else 'DIFFER'}, max |dlogprob| {err:.2e}")
        assert on and not off
        assert ids == ids0, kw
        assert err < LOGPROB_TOL, (kw, err)
    on, _, _ = go("1", tp=4, dp_attention=2, decode_batch_buckets=(1, 2))  # 2 x 1 rows over 4 ranks
    assert not on


def test_sequence_parallel_streams_round_where_replicated_do():
    """The hyper-connection blocks (models/hybrid.py _block, KILN_MHC_FORM=elementwise: the collapse and every
    output stream rounded to bf16 through the Veltkamp split) on rows split as the sequence-parallel prefill
    streams split them (R / tp per rank) against the same rows in one batch, as the replicated streams run
    them: the rounding happens at the same points, so the bf16 streams agree bit for bit except where a CPU
    fp32 matmul over fewer rows rounds its logits differently and the blocks carry that on (measured
    2026-10-04: 142 of 65,536 values, at most 2.0e-3 of the streams' max), while the same blocks with the
    rounding left out (fp32 streams, what a device graph may fold a bare bf16 round trip into) differ in
    nearly every value. A device check of the same question: tools/probe_mhc_rounding.py."""
    import tools.probe_mhc_blocks as pb
    from kiln.models import hybrid

    old = hybrid.MHC_FORM
    hybrid.MHC_FORM = "elementwise"
    try:
        hc, H, R, tp, blocks = 4, 256, 64, 8, 4
        model = pb.fake_model(hc, H)
        layers = pb.layers_of(pb.params(blocks, hc, H))
        g = torch.Generator().manual_seed(3)
        s = (torch.randn(R, hc * H, generator=g) * 3).bfloat16()
        whole = pb.run(model, s, layers)
        split = torch.cat([pb.run(model, s[r:r + R // tp], layers) for r in range(0, R, R // tp)])
        fp32 = pb.run(model, s.float(), layers)
        assert whole.dtype == split.dtype == torch.bfloat16
        diff = (whole.float() != split.float())
        rel = (whole.float() - split.float()).abs().max().item() / whole.float().abs().max().item()
        print(f"split vs whole: {int(diff.sum())} of {diff.numel()} values differ, at most {rel:.2e} of the max; "
              f"fp32 streams vs whole: {int((fp32 != whole.float()).sum())} differ")
        assert int(diff.sum()) <= diff.numel() // 100 and rel <= 2.0 ** -7
        assert int((fp32 != whole.float()).sum()) > diff.numel() * 9 // 10  # the check sees a dropped rounding
    finally:
        hybrid.MHC_FORM = old



def test_sequence_parallel_streams_default_per_platform(monkeypatch):
    """KILN_PREFILL_SP unset: sequence-parallel prefill streams on the trn1 families and trn2, where real-weight
    ppl holds with them (trn2: wikitext-2 -0.549 with them, -0.552 without, 2026-10-04), off on trn3 and inf2
    (not measured); KILN_PREFILL_SP=1 / 0 overrides either way. The target as kiln/platform.py reads it,
    through LNL's own override variable."""
    from kiln.models import decoder

    monkeypatch.delenv("KILN_PREFILL_SP", raising=False)
    for target, want in (("trn1", True), ("trn1n", True), ("trn2", True), ("trn3", False), ("inf2", False)):
        monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", target)
        assert decoder.prefill_sp_enabled() is want, target
    monkeypatch.setenv("KILN_PREFILL_SP", "1")
    assert decoder.prefill_sp_enabled()  # inf2, forced on
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setenv("KILN_PREFILL_SP", "0")
    assert not decoder.prefill_sp_enabled()  # trn2, forced off


def test_sp_group_collectives_default_per_platform(monkeypatch):
    """KILN_SP_GROUP=auto (the default): the token mixers' group gather and group reduce-scatter on the trn1
    families and trn2, where they are measured (models/decoder.py SP_GROUP_FAMILIES), off on trn3 and inf2;
    KILN_SP_GROUP=1 / 0 force it either way."""
    from kiln.models import decoder as hybrid

    for v in (None, "auto"):
        if v is None:
            monkeypatch.delenv("KILN_SP_GROUP", raising=False)
        else:
            monkeypatch.setenv("KILN_SP_GROUP", v)
        for target, want in (("trn1", True), ("trn1n", True), ("trn2", True), ("trn3", False), ("inf2", False)):
            monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", target)
            assert hybrid.sp_group_enabled() is want, (v, target)
    monkeypatch.setenv("KILN_SP_GROUP", "1")
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn3")
    assert hybrid.sp_group_enabled()  # trn3, forced on
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    monkeypatch.setenv("KILN_SP_GROUP", "0")
    assert not hybrid.sp_group_enabled()
