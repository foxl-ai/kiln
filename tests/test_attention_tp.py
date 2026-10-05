"""Attention TP smaller than the world TP (models/decoder.py DecoderForCausalLM: attn_tp), on CPU
over gloo. The token mixers (GQA attention, MiMo-V2's sliding-window attention with sinks, MLA /
DSA, Gated DeltaNet, KDA, Qwen Sparse Attention) split their heads attn_tp ways, replicated over
tp / attn_tp groups of consecutive ranks, and reduce over the group; the MLP and experts keep tp.

Every case runs the same prompts at tp=4 attn_tp=2, tp=4 attn_tp=1 and tp=2 attn_tp=1 and compares
greedy tokens (exactly) and the chosen tokens' logprobs (within fp32 reduction-order noise: the
o_proj / out_proj and MLP partial sums are added in a different order at each degree, so the runs
are not bit-identical) with tp=1. Each model's cases between them take every forward: decode,
chunked prefill, speculative verify (extend), MTP drafts, piecewise layer groups and overlap.
"""

import json
import os

import pytest
import torch

from tests.test_architectures import build as build_arch
from tests.test_linear_attn import build_kda, build_qwen3_5
from tests.test_mimo_v2 import build_reference as build_mimo
from tests.test_mla import build as build_mla

CASES = ((4, 2), (4, 1), (2, 1))
LOGPROB_TOL = 1e-4  # fp32; measured differences are printed


def run(path, prompts, sp, proposer=None, **kw):
    """(output ids, chosen logprobs, attn_tp, rank-0 KV / state shapes) of one engine run."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    base = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256, max_num_seqs=3,
                max_model_len=256, max_prefill_tokens=8)
    base.update(kw)
    eng = LLMEngine(EngineConfig(**base))
    try:
        if proposer is not None and kw.get("spec_method") not in (None, "mtp"):
            eng.proposer = proposer
        reqs = eng.generate(prompts, sp)
        shapes = [tuple(k.shape[1:]) for k in eng.runner.k_caches]
        if eng.runner.state is not None:
            shapes += [tuple(s.shape[1:]) for s in eng.runner.state.rec]
        spec = (eng.spec_proposed, eng.spec_accepted) if kw.get("spec_method") else None
        return [r.output_ids for r in reqs], [[x[0] for x in r.logprobs] for r in reqs], eng.model.attn_tp, shapes, spec
    finally:
        eng.close()


def check(path, prompts, variants, new_tokens=10, proposer=None, **common):
    """variants: (tp, attn_tp, engine kwargs) each compared with tp=1 under the same kwargs; attn_tp
    None leaves the degree to the engine (attention_tp's default)."""
    from kiln.config import ModelConfig
    from kiln.engine.request import SamplingParams
    from kiln.models.decoder import attention_tp

    sp = SamplingParams(max_new_tokens=new_tokens, ignore_eos=True, logprobs=1)
    refs = {}
    for tp, atp, kw in variants:
        key = json.dumps(kw, sort_keys=True)
        if key not in refs:
            refs[key] = run(path, prompts, sp, proposer() if proposer else None, **common, **kw)
        want_ids, want_lp, _, _, want_spec = refs[key]
        ids, lp, got_atp, shapes, spec = run(path, prompts, sp, proposer() if proposer else None, tp=tp,
                                             attention_tp=atp, **common, **kw)
        err = max(abs(a - b) for x, y in zip(lp, want_lp) for a, b in zip(x, y))
        print(f"{os.path.basename(path)} tp={tp} attn_tp={got_atp}{'' if atp else ' (default)'} {kw}: tokens {'equal' if ids == want_ids else 'DIFFER'}, "
              f"max |dlogprob| {err:.2e}, rank-0 cache / state shapes {shapes}")
        assert got_atp == (atp if atp is not None else attention_tp(ModelConfig.from_pretrained(path), tp))
        assert ids == want_ids, (tp, atp, kw)
        assert err < LOGPROB_TOL, (tp, atp, kw, err)
        if want_spec is not None:
            assert spec == want_spec and spec[0] > 0, (spec, want_spec)


def prompts(seed, lengths, vocab=384):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (n,), generator=g).tolist() for n in lengths]


class _Repeat:
    """Drafts by repeating the last 3 tokens: verify graphs (Q = k + 1) with drafts accepted and
    rejected, whatever the model generates."""

    def propose(self, tokens, k=None):
        return list(tokens[-3:])[: k or 3]


# -- which degree, and the shapes it gives --------------------------------------------------------


def _config(tmp_path, name):
    from kiln.config import ModelConfig

    with open(os.path.join(os.path.dirname(__file__), "reference", "hybrid_configs", name)) as f:
        c = json.load(f)
    d = tmp_path / name.replace(".json", "")
    d.mkdir()
    (d / "config.json").write_text(json.dumps(c))
    return ModelConfig.from_pretrained(str(d))


def test_default_degree_and_refusals(tmp_path):
    """Qwen3.8-Flash-Next (24 / 2 attention heads, 16 k / 48 v GDN heads) at the tp its 185 GB FP8
    weights need on trn1.32xlarge: attention TP 8, everything else 16 or 32. GLM-5.3-Flash's 64
    KDA / 64 MLA heads divide every tp up to 64, so it keeps plain TP."""
    from kiln.models.decoder import attention_tp

    q = _config(tmp_path, "Qwen3.8-Flash-Next.json")
    assert [attention_tp(q, t) for t in (1, 2, 4, 8, 16, 32)] == [1, 2, 4, 8, 8, 8]
    assert attention_tp(q, 16, 4) == 4
    for bad, why in ((16, "does not fit 24 query / 2 KV heads"), (3, "does not divide tp=16"),
                     (32, "does not divide tp=16")):
        with pytest.raises(ValueError, match=why):
            attention_tp(q, 16, bad)
    g = _config(tmp_path, "GLM-5.3-Flash.json")
    assert [attention_tp(g, t) for t in (2, 8, 16, 32)] == [2, 8, 16, 32]


def test_qwen3_8_flash_next_shards_at_tp16(tmp_path):
    """The real config builds at tp=16 (refused before attention TP): each rank holds 3 of the 24
    query heads and one KV head (replicated: 8 attention ranks over 2 KV heads), 2 k / 6 v GDN heads,
    and 1 / 16 of every expert (40 of 640 intermediate columns)."""
    from kiln.models.decoder import DecoderForCausalLM

    cfg = _config(tmp_path, "Qwen3.8-Flash-Next.json").truncated(4)
    with torch.device("meta"):
        m = DecoderForCausalLM(cfg, torch.bfloat16, 64, tp_rank=13, tp_size=16)
    assert (m.attn_tp, m.attn_rank) == (8, 5)
    gdn, qsa = m.layers[0], m.layers[3]
    assert (qsa.nh, qsa.nkv, qsa.kv_offset) == (3, 1, 1)
    assert (gdn.nk, gdn.nv) == (2, 6) and m.state_shapes()[0] == ((3, 2 * 2 * 128 + 6 * 128), (6, 128, 128))
    assert m.kv_shapes() == [((1, 256), (1, 256))]
    assert m.moe_inter == 40 and tuple(qsa.w_gu.shape) == (512, 80, 2560)


# -- parity with tp=1 ------------------------------------------------------------------------------


def test_qwen3_dense(tmp_path):
    """4 query / 2 KV heads: attn_tp 2 gives each rank one KV head, attn_tp 1 all of them."""
    build_arch("qwen3", str(tmp_path))
    ps = prompts(1, (5, 19, 30)) + [[7, 8, 9] * 5]
    check(str(tmp_path), ps, [(4, 2, dict(piecewise=True, piecewise_group=1)), (4, 1, dict(overlap=True)),
                              (2, 1, {})], max_num_seqs=4)


def test_qwen3_dense_speculative_verify(tmp_path):
    """Extend graphs (Q = k + 1 rows per sequence) under attention TP."""
    build_arch("qwen3", str(tmp_path))
    check(str(tmp_path), prompts(2, (9, 23)), [(t, a, dict(spec_method="ngram", spec_k=3)) for t, a in CASES],
          new_tokens=14, proposer=_Repeat, max_num_seqs=2)


def test_mimo_v2(tmp_path):
    """Full attention (4 query / 2 KV heads, no sinks) and sliding-window attention (4 / 4, window 8,
    sinks per query head), Dk 24 != Dv 16, value scale, MoE; fused grouped qkv_proj."""
    build_mimo(str(tmp_path), "fused_qkv")
    ps = prompts(3, (5, 19, 30))
    check(str(tmp_path), ps, [(4, 2, dict(piecewise=True, piecewise_group=2)), (4, 1, dict(overlap=True)),
                              (2, 1, {})])


def test_mimo_v2_mtp(tmp_path):
    """MTP drafts (a sliding-window layer of its own, at the attention TP) and their verify."""
    from tests.test_mtp import add_mtp

    build_mimo(str(tmp_path), "split")
    add_mtp(str(tmp_path), "split")
    check(str(tmp_path), prompts(4, (11, 23)), [(t, a, dict(spec_method="mtp", spec_k=2)) for t, a in CASES],
          new_tokens=12, max_num_seqs=2)


@pytest.mark.parametrize("name", ["deepseek_v3", "glm_moe_dsa"])
def test_mla(tmp_path, name):
    """4 MLA heads (q_b, kv_b and o_proj by attention rank; the latent cache, q_a / kv_a and the
    DSA indexer replicated). GLM's index_topk 12 makes the top-k and IndexShare bite."""
    build_mla(name, str(tmp_path), **(dict(index_topk=12) if name == "glm_moe_dsa" else {}))
    ps = prompts(5, (9, 30, 41))
    check(str(tmp_path), ps, [(4, 2, dict(piecewise=True, piecewise_group=1)), (4, 1, dict(overlap=True)),
                              (2, 1, dict(spec_method="ngram", spec_k=3))], proposer=_Repeat, max_prefill_tokens=16)


def test_qwen3_5_gdn(tmp_path):
    """Gated DeltaNet (2 k / 4 v heads) with gated full attention (4 / 2 heads): at tp=4 the GDN
    heads cap the default attention TP at 2."""
    build_qwen3_5(str(tmp_path))
    from kiln.config import ModelConfig
    from kiln.models.decoder import attention_tp

    assert attention_tp(ModelConfig.from_pretrained(str(tmp_path)), 4) == 2
    ps = prompts(6, (5, 19, 61))
    check(str(tmp_path), ps, [(4, None, dict(piecewise=True, piecewise_group=2)), (4, 1, dict(overlap=True)),
                              (2, 1, {})])


def test_kda(tmp_path):
    """Kimi Delta Attention, 4 heads (f_b / g_b / dt_bias by attention rank, f_a / g_a replicated)."""
    build_kda(str(tmp_path))
    ps = prompts(7, (7, 45))
    check(str(tmp_path), ps, [(4, 2, dict(piecewise=True)), (4, 1, {}), (2, 1, dict(overlap=True))], max_num_seqs=2)


# -- the hyper-connection hybrids (transformers >= 5.18) -------------------------------------------


def test_qwen3_8_flash_next_heads_not_dividing_tp(tmp_path):
    """Qwen3.8-Flash-Next's real config truncated (GDN, GDN + PLE, GDN, QSA, x2) at 6 query / 2 KV
    attention heads and 2 k / 6 v GDN heads, so tp=4 cannot split them (the real model's 24 heads
    at tp=16, scaled down): the default attention TP is 2, the experts, the shared expert and the
    n-gram tables' columns are split 4 ways. QSA with indexer_budget 16 (sparse past 16 tokens)."""
    pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    from tests.test_qwen4_exp import build
    from tests.test_qwen4_exp import prompts as q_prompts

    from kiln.config import ModelConfig
    from kiln.models.decoder import attention_tp

    build(str(tmp_path), seed=1, indexer_budget=16, indexer_n_heads=16, num_attention_heads=6,
          linear_num_value_heads=6)
    cfg = ModelConfig.from_pretrained(str(tmp_path))
    with pytest.raises(ValueError, match="does not fit 2 k / 6 v linear-attention heads"):
        attention_tp(cfg, 4, 4)
    assert attention_tp(cfg, 4) == 2
    ps = q_prompts(3, (11, 41, 23))
    check(str(tmp_path), ps, [(4, None, {}), (4, 1, dict(piecewise=True, piecewise_group=2)), (2, 1, {})],
          max_prefill_tokens=16)


def test_glm5_next(tmp_path):
    """GLM-5.3-Flash truncated: KDA and pooled-DSA NoPE MLA layers, mHC streams, clamped SwiGLU."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from tests.test_glm5_next import build

    build(str(tmp_path), index_topk=16)
    ps = prompts(8, (11, 41))
    check(str(tmp_path), ps, [(4, 2, dict(piecewise=True, piecewise_group=2)), (4, 1, {}), (2, 1, {})],
          max_prefill_tokens=16, max_num_seqs=2)
