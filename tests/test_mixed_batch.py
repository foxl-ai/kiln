"""Mixed batches (EngineConfig.mixed_batch, KILN_MIXED_BATCH=1; DecoderForCausalLM.forward_mixed,
ModelRunner.mixed, LLMEngine._launch_mixed): the decode tokens of running requests ride in the prefill
calls of the same step. Every prefill call is one graph over each DP-attention group's chunk rows
followed by its decode rows; the residual stream, the hyper-connections and the MLP / experts run over
both, every token mixer runs its chunk form on the chunk's rows and its decode form on the rest.

The scheduler is not changed, so a mixed engine runs exactly the steps an unmixed one runs, and each
case compares greedy tokens (exactly) and chosen-token logprobs (within fp32 noise: a CPU fp32 matmul
over a few rows can round differently from the same rows inside a bigger one) against the same engine
with mixed_batch off. Prompts of different lengths, more requests than running slots and chunks smaller
than the prompts make prefill chunks and decodes meet in most steps; the cases count that they did.
"""

import os

import pytest
import torch

from tests.test_attention_tp import prompts
from tests.test_architectures import build as build_arch
from tests.test_linear_attn import build_kda

LOGPROB_TOL = 1e-4


def run(path, ps, sp, **kw):
    """(output ids, chosen logprobs, prompt logprobs, graph calls by kind, decode rows that rode in mixed
    calls) of one engine run."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    base = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256, max_num_seqs=3,
                max_model_len=256, max_prefill_tokens=16)
    base.update(kw)
    eng = LLMEngine(EngineConfig(**base))
    rode = [0]
    if eng.runner.mixed_rows:
        real = eng.runner.mixed

        def count(chunks, decs):
            rode[0] += len(decs)
            return real(chunks, decs)

        eng.runner.mixed = count
    try:
        reqs = eng.generate(ps, sp)
        calls: dict = {}
        for key, n in eng.runner.calls.items():
            calls[key[0]] = calls.get(key[0], 0) + n
        return ([r.output_ids for r in reqs], [[x[0] for x in r.logprobs] for r in reqs],
                [dict(r.prompt_logprobs) for r in reqs], calls, rode[0])
    finally:
        eng.close()


def check(path, ps, variants, new_tokens=10, prompt_logprobs=None, **common):
    """Every variant (engine kwargs) with mixed_batch on against the same with it off."""
    from kiln.engine.request import SamplingParams

    sp = SamplingParams(max_new_tokens=new_tokens, ignore_eos=True, logprobs=1, prompt_logprobs=prompt_logprobs)
    for kw in variants:
        want_ids, want_lp, want_plp, want_calls, _ = run(path, ps, sp, **common, **kw)
        ids, lp, plp, calls, rode = run(path, ps, sp, mixed_batch=True, **common, **kw)
        err = max(abs(a - b) for x, y in zip(lp, want_lp) for a, b in zip(x, y))
        if prompt_logprobs is not None:
            err = max([err] + [abs(a[q][0] - b[q][0]) for a, b in zip(plp, want_plp) for q in b])
            assert [sorted(a) for a in plp] == [sorted(b) for b in want_plp]
        print(f"{os.path.basename(path)} {kw}: tokens {'equal' if ids == want_ids else 'DIFFER'}, max |dlogprob| "
              f"{err:.2e}, calls unmixed {want_calls} mixed {calls}, decode rows in mixed calls {rode}")
        assert ids == want_ids, kw
        assert err < LOGPROB_TOL, (kw, err)
        assert "prefill" not in calls and calls.get("mixed", 0) == want_calls["prefill"], (calls, want_calls)
        assert rode > 0, "no decode rode in a prefill call: the case does not test mixing"
        assert calls.get("decode", 0) < want_calls["decode"], (calls, want_calls)


@pytest.fixture(scope="module")
def glm(tmp_path_factory):
    """GLM-5.3-Flash truncated to 8 layers (3 dense KDA, a pooled-DSA MoE layer, 3 KDA MoE, a DSA MoE),
    index_topk 16: prompts past 16 tokens are in the sparse regime (tests/test_glm5_next.py)."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from tests.test_glm5_next import build

    path = tmp_path_factory.mktemp("glm5_next_mixed")
    build(str(path), seed=3, index_topk=16)
    return str(path)


GLM_PROMPTS = (13, 41, 5, 27, 60, 9)


def _mixers(monkeypatch, form):
    from kiln.models import decoder

    monkeypatch.setattr(decoder, "MIXED_MIXERS", form)
    monkeypatch.setenv("KILN_MIXED_MIXERS", form)  # tensor-parallel workers import it from the environment


@pytest.mark.parametrize("form", ["joint", "calls"])
def test_glm5_next_mixed_matches_unmixed(glm, monkeypatch, form):
    """KDA (chunked delta rule beside recurrent steps), pooled-DSA MLA (latent and pool-key writes,
    the selection), mHC streams and the clamped MoE over chunk and decode rows; graph and piecewise
    execution, overlap scheduling (decode rows reading their token from the board); both mixer forms
    (decoder.MIXED_MIXERS: projections once over all rows, or the two batch forms as two calls)."""
    _mixers(monkeypatch, form)
    ps = prompts(11, GLM_PROMPTS)
    check(glm, ps, [{}, dict(piecewise=True, piecewise_group=2), dict(overlap=True)])
    if form == "joint":
        check(glm, ps, [dict(tp=4, dp_attention=2, piecewise=True, piecewise_group=2, mixed_decode_rows=2)],
              max_num_seqs=4)


def _layout(monkeypatch, layout):
    from kiln.models import hybrid

    monkeypatch.setattr(hybrid, "MIXED_SP", layout)
    monkeypatch.setenv("KILN_MIXED_SP", layout)  # tensor-parallel workers import it from the environment


@pytest.mark.parametrize("layout", ["split", "rows"])
def test_glm5_next_mixed_tensor_parallel(glm, monkeypatch, layout):
    """DP attention 2 at tp=4 (each group's chunk and decode rows, the mixers on the group's rows) with
    sequence-parallel streams in both mixed layouts (models/hybrid.py MIXED_SP: each rank's share of the
    chunk rows and every decode row, or each rank's share of all 2 x (8 + 2) rows), and plain tp=2."""
    from kiln.models import hybrid

    _layout(monkeypatch, layout)
    seen = []
    real = hybrid._layer_mixed_split
    monkeypatch.setattr(hybrid, "_layer_mixed_split", lambda *a, **k: seen.append(1) or real(*a, **k))
    ps = prompts(12, GLM_PROMPTS)
    check(glm, ps, [dict(tp=4, dp_attention=2, piecewise=True, piecewise_group=2, mixed_decode_rows=2),
                    dict(tp=4, dp_attention=2, mixed_decode_rows=2, overlap=True), dict(tp=2, overlap=True)],
          max_num_seqs=4)
    assert bool(seen) == (layout == "split")  # rank 0 ran the layout it was given


@pytest.mark.parametrize("layout", ["split", "rows"])
def test_glm5_next_mixed_sequence_parallel_stays_on(glm, monkeypatch, layout):
    """The runner keeps sequence-parallel streams on only where every mixed call's sequence-parallel rows
    divide over tp: "rows" 2 x (8 + 2) rows over 4 ranks do, 2 x (8 + 3) do not; "split" only the 2 x 8
    chunk rows, which do either way."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    _layout(monkeypatch, layout)
    monkeypatch.setenv("KILN_PREFILL_SP", "1")
    for rows, on in ((2, True), (3, layout == "split")):
        eng = LLMEngine(EngineConfig(model_path=glm, device="cpu", dtype=torch.float32, page_size=4, num_pages=64,
                                     max_num_seqs=4, max_model_len=128, max_prefill_tokens=16, tp=4, dp_attention=2,
                                     mixed_batch=True, mixed_decode_rows=rows))
        try:
            assert eng.runner.model.prefill_sp is on, rows
        finally:
            eng.close()


def test_glm5_next_mixed_overflow_and_prompt_logprobs(glm):
    """One decode row per mixed call: the other decodes of a step go to a decode call before it; and
    prompt logprobs scored from the mixed call's rows."""
    ps = prompts(13, GLM_PROMPTS)
    check(glm, ps, [dict(mixed_decode_rows=1)])
    check(glm, ps[:4], [{}], new_tokens=4, prompt_logprobs=1)


@pytest.mark.parametrize("pool", ["separate", "off"])
def test_glm5_next_mixed_pool_key_forms(glm, monkeypatch, pool):
    """The joint mixers with the pooled indexer's other pool-key forms (models/mla.py pool_cache_mode; the
    CPU's fp32 cache takes "inplace" by default): a separate token-slot cache, or keys rebuilt per call."""
    from kiln.models import mla

    monkeypatch.setattr(mla, "POOL_CACHE", pool)
    monkeypatch.setenv("KILN_DSA_POOL_CACHE", pool)
    ps = prompts(17, GLM_PROMPTS)
    check(glm, ps, [{}, dict(tp=4, dp_attention=2, piecewise=True, piecewise_group=2, mixed_decode_rows=2)],
          max_num_seqs=4)


def test_glm5_next_mixed_decode_split(glm, monkeypatch):
    """KILN_MIXED_DECODE_SPLIT: the token mixers take the decode rows of a mixed call in slices (of 1 here)."""
    from kiln.models import decoder

    monkeypatch.setattr(decoder, "MIXED_DECODE_SPLIT", 1)
    monkeypatch.setenv("KILN_MIXED_DECODE_SPLIT", "1")  # tensor-parallel workers import it from the environment
    ps = prompts(16, GLM_PROMPTS)
    check(glm, ps, [{}, dict(tp=4, dp_attention=2, piecewise=True, piecewise_group=2, mixed_decode_rows=2)],
          max_num_seqs=4)


def test_qwen3_mixed(tmp_path):
    """Plain GQA attention layers (DecoderForCausalLM._gqa per batch form), plain and DP attention."""
    build_arch("qwen3", str(tmp_path))
    ps = prompts(14, (9, 33, 3, 20, 47))
    check(str(tmp_path), ps, [{}, dict(piecewise=True, piecewise_group=1),
                              dict(tp=4, dp_attention=2, overlap=True)])


@pytest.mark.parametrize("form", ["joint", "calls"])
def test_kda_mixed(tmp_path, monkeypatch, form):
    """A model of KDA layers only (linear_attn.mixer without hyper-connections), both mixer forms."""
    _mixers(monkeypatch, form)
    build_kda(str(tmp_path))
    ps = prompts(15, (7, 45, 20, 3))
    check(str(tmp_path), ps, [{}, dict(piecewise=True), dict(tp=2, dp_attention=2)], max_num_seqs=2)


def test_mixed_refusals(tmp_path):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    build_arch("qwen3", str(tmp_path))
    with pytest.raises(ValueError, match="speculative decoding"):
        LLMEngine(EngineConfig(model_path=str(tmp_path), device="cpu", dtype=torch.float32, num_pages=64,
                               mixed_batch=True, spec_method="ngram"))
