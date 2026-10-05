"""Multi-head latent attention and DeepSeek Sparse Attention against Hugging Face transformers.

Every model is built by transformers itself (5.15: models/deepseek_v3, deepseek_v32,
glm_moe_dsa, youtu) from a REAL config.json, truncated: fewer layers, heads, experts, a
small hidden size and vocabulary, everything else (MLA head geometry, RoPE / YaRN, routing,
the DSA indexer and GLM-5.3's IndexShare pattern) as published. The configs are vendored in
tests/reference/mla_configs, fetched 2026-10-02 from
https://huggingface.co/<repo>/raw/main/config.json for deepseek-ai/DeepSeek-V3,
deepseek-ai/DeepSeek-V3.2, zai-org/GLM-5.3, moonshotai/Kimi-K2.7-Code (its text_config),
moonshotai/Kimi-K2-Instruct,
moonshotai/Moonlight-16B-A3B-Instruct (no q_lora) and tencent/Youtu-LLM-2B (dense MLA).

The reference for generation is a full forward per step without a cache (greedy), so the
paged MLA cache, chunked prefill, weight absorption and the DSA top-k are all checked
against the model's plain definition.
"""

import json
import os

import pytest
import torch

CONFIGS = os.path.join(os.path.dirname(__file__), "reference", "mla_configs")

# name -> (config file, overrides). Sizes are cut; the attention geometry is the real one. Every
# matmul input dim is a multiple of 128, as in the real models: quant.dequant infers an FP8 scale
# block as K / (number of scale columns), which a K like 160 (blocks of 128 + 32) breaks.
SMALL = dict(hidden_size=128, intermediate_size=256, moe_intermediate_size=128, num_attention_heads=4,
             num_key_value_heads=4, vocab_size=384, max_position_embeddings=1024, num_nextn_predict_layers=0,
             bos_token_id=0, eos_token_id=1, pad_token_id=None)
MODELS = {
    # YaRN x40 with mscale_all_dim 1, 8 expert groups of which 4 are kept, a shared expert.
    "deepseek_v3": ("DeepSeek-V3.json", dict(num_hidden_layers=4, first_k_dense_replace=1, n_routed_experts=16,
                                             num_experts_per_tok=4)),
    # Kimi K2.7 Code's text model: YaRN x64 from theta 5e4, no expert groups.
    "kimi_k2": ("Kimi-K2.7-Code.json", dict(num_hidden_layers=3, n_routed_experts=8, num_experts_per_tok=2)),
    # Kimi K2 Instruct: YaRN x32 with beta_fast = beta_slow = 1 (a degenerate ramp).
    "kimi_k2_instruct": ("Kimi-K2-Instruct.json", dict(num_hidden_layers=3, n_routed_experts=8, num_experts_per_tok=2)),
    # No q_lora_rank (q_proj straight from the hidden state).
    "moonlight": ("Moonlight-16B-A3B-Instruct.json", dict(num_hidden_layers=3, n_routed_experts=8,
                                                          num_experts_per_tok=2)),
    "youtu": ("Youtu-LLM-2B.json", dict(num_hidden_layers=3)),
    # DSA on every layer, indexer RoPE half-split.
    "deepseek_v32": ("DeepSeek-V3.2.json", dict(num_hidden_layers=3, first_k_dense_replace=1, n_routed_experts=16,
                                               num_experts_per_tok=4)),
    # IndexShare (F F F S S S F ...), interleaved indexer RoPE, rope theta 8e6.
    "glm_moe_dsa": ("GLM-5.3.json", dict(num_hidden_layers=5, first_k_dense_replace=1, n_routed_experts=8,
                                        num_experts_per_tok=2, mlp_layer_types=None, indexer_types=None)),
}


def hf_config(name: str, **extra):
    import transformers as tf

    fname, over = MODELS[name]
    with open(os.path.join(CONFIGS, fname)) as f:
        c = json.load(f)
    if "text_config" in c:  # Kimi K2.5 / K2.7: a DeepseekV3 text model inside a VL wrapper
        c = c["text_config"]
    for k in ("quantization_config", "auto_map", "architectures", "_name_or_path", "transformers_version"):
        c.pop(k, None)
    c.update(SMALL)
    c.update(over)
    c.update(extra)
    c = {k: v for k, v in c.items() if v is not None}  # None: the class default (GLM derives its patterns)
    cls = {"deepseek_v3": tf.DeepseekV3Config, "kimi_k2": tf.DeepseekV3Config, "deepseek_v32": tf.DeepseekV32Config,
           "glm_moe_dsa": tf.GlmMoeDsaConfig, "youtu": tf.YoutuConfig}[c.pop("model_type")]
    return cls(**c)


def build(name: str, path: str, seed: int = 0, **extra):
    import transformers as tf

    cfg = hf_config(name, **extra)
    cfg._attn_implementation = "eager"
    model = {"DeepseekV3Config": tf.DeepseekV3ForCausalLM, "DeepseekV32Config": tf.DeepseekV32ForCausalLM,
             "GlmMoeDsaConfig": tf.GlmMoeDsaForCausalLM, "YoutuConfig": tf.YoutuForCausalLM}[type(cfg).__name__](cfg)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():  # O(1) activations: unit-variance inputs to every matmul
        for n, p in [*model.named_parameters(), *model.named_buffers()]:
            if "inv_freq" in n:
                continue
            if n.endswith("norm.weight") or "layernorm" in n:
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            elif n.endswith("bias"):
                p.copy_((0.5 if "correction" in n else 0.1) * torch.randn(p.shape, generator=g))
            elif "embed_tokens" in n:
                p.copy_(torch.randn(p.shape, generator=g))
            else:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
    model.eval()
    model.save_pretrained(path, safe_serialization=True)
    return model


def hf_greedy(model, ids, n):
    out = list(ids)
    with torch.no_grad():
        for _ in range(n):
            out.append(int(model(torch.tensor([out]), use_cache=False).logits[0, -1].argmax()))
    return out[len(ids):]


def engine(path, **kw):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    base = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256, max_num_seqs=3,
                max_model_len=256, max_prefill_tokens=8)
    base.update(kw)
    return LLMEngine(EngineConfig(**base))


@pytest.fixture(scope="module", params=list(MODELS))
def built(request, tmp_path_factory):
    path = tmp_path_factory.mktemp(request.param)
    return request.param, str(path), build(request.param, str(path))


def test_config_is_parsed(built):
    from kiln.config import ModelConfig
    from kiln.models.decoder import DecoderForCausalLM

    name, path, hf = built
    cfg = ModelConfig.from_pretrained(path)
    hc = hf.config
    m = cfg.attn_layers[0].mla
    assert (m.kv_lora_rank, m.qk_nope_head_dim, m.qk_rope_head_dim, m.v_head_dim) == (
        hc.kv_lora_rank, hc.qk_nope_head_dim, hc.qk_rope_head_dim, hc.v_head_dim)
    assert abs(m.softmax_scale - hf.model.layers[0].self_attn.scaling) < 1e-9
    want_moe = [i for i, l in enumerate(hf.model.layers) if hasattr(l.mlp, "experts")]
    assert list(cfg.moe_layers) == want_moe
    # The cache holds the latent and the rope key (+ the indexer key on DSA layers with an indexer).
    model = DecoderForCausalLM(cfg, torch.float32, 64)
    for i, (k, v) in enumerate(model.kv_shapes()):
        idx = getattr(hf.model.layers[i].self_attn, "indexer", None)
        assert k == (1, hc.kv_lora_rank)
        assert v == (1, hc.qk_rope_head_dim + (hc.index_head_dim if idx is not None else 0))
    if name == "glm_moe_dsa":
        assert [s.mla.dsa.indexer for s in cfg.attn_layers] == [t == "full" for t in hc.indexer_types]
        assert hc.indexer_types[:5] == ["full", "full", "full", "shared", "shared"]


def test_logits_match(built):
    from kiln.config import ModelConfig
    from kiln.models.loader import load_model

    name, path, hf = built
    ours = load_model(path, ModelConfig.from_pretrained(path), torch.float32, torch.device("cpu"), 512)
    ids = torch.randint(0, 384, (37,), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = hf(ids.unsqueeze(0), use_cache=False).logits[0]
        got = ours.forward_logits(ids)
    assert (got - want).abs().max().item() < 1e-3 * want.abs().max().item(), name


def test_paged_chunked_greedy_matches(built):
    """Chunked prefill (chunks of 8 over pages of 4), absorbed decode, three sequences in one
    batch, against the plain model."""
    from kiln.engine.request import SamplingParams

    name, path, hf = built
    eng = engine(path)
    g = torch.Generator().manual_seed(2)
    prompts = [torch.randint(0, 384, (n,), generator=g).tolist() for n in (5, 19, 30)]
    reqs = eng.generate(prompts, SamplingParams(max_new_tokens=12, ignore_eos=True))
    for ids, r in zip(prompts, reqs):
        assert r.output_ids == hf_greedy(hf, ids, 12), name


def test_cache_holds_only_the_latent(tmp_path):
    """Per token and layer the pool holds kv_lora_rank + qk_rope_head_dim values (576 for
    DeepSeek-V3's real dims), one copy, whatever the head count."""
    build("deepseek_v3", str(tmp_path))
    eng = engine(str(tmp_path), dtype=torch.bfloat16)
    run = eng.runner
    assert [tuple(k.shape[1:]) for k in run.k_caches] == [(1, 512)] * 4
    assert [tuple(v.shape[1:]) for v in run.v_caches] == [(1, 64)] * 4
    assert run.kv_page_bytes() == 4 * 4 * 576 * 2  # page_size 4, 4 layers, bf16
    assert eng.mcfg.kv_bytes_per_token(torch.bfloat16) == 4 * 576 * 2


@pytest.mark.parametrize("name", ["deepseek_v3", "moonlight"])
def test_prefill_absorbed_equals_expanded(tmp_path, monkeypatch, name):
    """A prefill chunk may decompress its context (default) or attend over the latent; both
    are the same computation."""
    import kiln.models.mla as mla
    from kiln.engine.request import SamplingParams

    build(name, str(tmp_path))
    prompts = [list(range(10, 41)), [3, 1, 4, 1, 5, 9, 2, 6]]
    sp = SamplingParams(max_new_tokens=8, ignore_eos=True, logprobs=0, prompt_logprobs=1)
    runs = []
    for mode in ("expand", "absorb"):
        monkeypatch.setattr(mla, "PREFILL", mode)
        runs.append([(r.output_ids, r.logprobs, r.prompt_logprobs) for r in engine(str(tmp_path)).generate(prompts, sp)])
    (a, b) = runs
    for (oa, la, pa), (ob, lb, pb) in zip(a, b):
        assert oa == ob
        assert max(abs(x[0] - y[0]) for x, y in zip(la, lb)) < 1e-4
        assert max(abs(pa[q][0] - pb[q][0]) for q in pa) < 1e-4


# -- DeepSeek Sparse Attention -------------------------------------------------------


@pytest.fixture(scope="module", params=["deepseek_v32", "glm_moe_dsa"])
def sparse(request, tmp_path_factory):
    """A DSA model whose index_topk (12) is far below the context, so the top-k bites."""
    path = tmp_path_factory.mktemp(request.param + "_k12")
    return request.param, str(path), build(request.param, str(path), seed=3, index_topk=12)


def test_dsa_topk_logits_match(sparse):
    import dataclasses

    from kiln.config import ModelConfig
    from kiln.models.loader import load_model

    name, path, hf = sparse
    cfg = ModelConfig.from_pretrained(path)
    assert cfg.attn_layers[0].mla.dsa.topk == 12
    ours = load_model(path, cfg, torch.float32, torch.device("cpu"), 512)
    ids = torch.randint(0, 384, (61,), generator=torch.Generator().manual_seed(4))
    no_dsa = dataclasses.replace(cfg, attn_layers=tuple(
        dataclasses.replace(s, mla=dataclasses.replace(s.mla, dsa=None), v_head_dim=s.mla.qk_rope_head_dim)
        for s in cfg.attn_layers))
    with torch.no_grad():
        want = hf(ids.unsqueeze(0), use_cache=False).logits[0]
        got = ours.forward_logits(ids)
        dense = load_model(path, no_dsa, torch.float32, torch.device("cpu"), 512).forward_logits(ids)
    assert (got - want).abs().max().item() < 1e-3 * want.abs().max().item(), name
    assert (dense - want).abs().max().item() > 1e-2  # the top-k changed the result


@pytest.mark.parametrize("mode", ["mask", "gather"])
def test_dsa_topk_paged_greedy_matches(sparse, monkeypatch, mode):
    """Decode, chunked prefill and (GLM) shared-indexer layers reading the full layers' top-k
    through the scratch tensor, with contexts up to 65 tokens against index_topk 12."""
    import kiln.models.mla as mla
    from kiln.engine.request import SamplingParams

    name, path, hf = sparse
    monkeypatch.setattr(mla, "DSA_MODE", mode)
    eng = engine(path, max_prefill_tokens=16)
    g = torch.Generator().manual_seed(5)
    prompts = [torch.randint(0, 384, (n,), generator=g).tolist() for n in (9, 30, 50)]
    reqs = eng.generate(prompts, SamplingParams(max_new_tokens=15, ignore_eos=True))
    for ids, r in zip(prompts, reqs):
        assert r.output_ids == hf_greedy(hf, ids, 15), (name, mode)


class _Oracle:
    """Drafts the reference's own greedy continuation, with every other draft's last token
    corrupted, so speculative verify accepts some drafts in full and rejects inside others."""

    def __init__(self, ref: list[int], k: int):
        self.ref, self.k, self.calls = ref, k, 0

    def propose(self, tokens, k=None):
        k = self.k if k is None else k
        n = len(tokens)
        if tokens != self.ref[:n] or n >= len(self.ref):
            return []
        d = list(self.ref[n : n + k])
        self.calls += 1
        if self.calls % 2 == 0 and d:
            d[-1] = (d[-1] + 1) % 384
        return d


@pytest.mark.parametrize("spec", [True, False])
def test_dsa_speculative_verify_and_prefix_cache(sparse, spec):
    """Extend graphs (speculative verify, Q = k + 1 rows per sequence) over the latent cache,
    with drafts accepted in full and in part, and a second request that reuses the first
    one's prefix pages from the radix cache."""
    from kiln.engine.request import SamplingParams

    name, path, hf = sparse
    kw = dict(spec_method="ngram", spec_k=3) if spec else {}
    eng = engine(path, max_prefill_tokens=16, **kw)
    prompt = torch.randint(0, 384, (32,), generator=torch.Generator().manual_seed(6)).tolist()
    want = hf_greedy(hf, prompt, 16)
    if spec:
        eng.proposer = _Oracle(prompt + want, 3)
    (a,) = eng.generate([prompt], SamplingParams(max_new_tokens=16, ignore_eos=True))
    assert a.output_ids == want
    if spec:
        eng.proposer = _Oracle([], 3)
    (b,) = eng.generate([prompt + a.output_ids[:4]], SamplingParams(max_new_tokens=10, ignore_eos=True))
    assert b.output_ids == hf_greedy(hf, prompt + a.output_ids[:4], 10)
    assert b.num_cached_tokens >= 32
    if spec:
        assert eng.spec_proposed >= 9 and 0 < eng.spec_accepted < eng.spec_proposed, (eng.spec_accepted,
                                                                                       eng.spec_proposed)


def test_piecewise_matches_whole_model(sparse):
    """One graph per layer kind: GLM's shared layers then read the top-k a full layer wrote in
    an earlier graph."""
    from kiln.engine.request import SamplingParams

    name, path, _ = sparse
    g = torch.Generator().manual_seed(7)
    prompts = [torch.randint(0, 384, (n,), generator=g).tolist() for n in (11, 40)]
    sp = SamplingParams(max_new_tokens=14, ignore_eos=True, logprobs=1)
    runs = [[(r.output_ids, r.logprobs) for r in engine(path, piecewise=pw, piecewise_group=grp,
                                                        max_prefill_tokens=16).generate(prompts, sp)]
            for pw, grp in ((False, None), (True, 1), (True, 2))]
    assert runs[0] == runs[1] == runs[2]


# -- tensor parallelism, FP8 -----------------------------------------------------------


@pytest.mark.parametrize("name", ["deepseek_v3", "glm_moe_dsa"])
def test_tp2_matches_tp1(tmp_path, name):
    """Heads split across ranks (q_b, kv_b, o_proj), the latent cache and the indexer
    replicated."""
    from kiln.engine.request import SamplingParams

    build(name, str(tmp_path), index_topk=12) if name == "glm_moe_dsa" else build(name, str(tmp_path))
    prompts = [[5, 9, 11, 200, 3, 77, 12, 13, 14, 15, 16], list(range(40, 80))]
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True)
    want = [r.output_ids for r in engine(str(tmp_path), max_num_seqs=2).generate(prompts, sp)]
    two = engine(str(tmp_path), max_num_seqs=2, tp=2)
    try:
        got = [r.output_ids for r in two.generate(prompts, sp)]
    finally:
        two.close()
    assert got == want


def _fp8_checkpoint(src: str, dst: str, deq_dst: str) -> None:
    """The checkpoint at src stored the way DeepSeek-V3 / GLM-5.3 store theirs: every
    projection except the router, the indexer's weights_proj, embeddings and lm_head in FP8
    e4m3fn with 128 x 128 block scales (weight_scale_inv; zai-org/GLM-5.3
    model.safetensors.index.json), and deq_dst: the same weights dequantized, for the reference."""
    import shutil

    from safetensors.torch import load_file, save_file

    shutil.copytree(src, dst)
    shutil.copytree(src, deq_dst)
    t = load_file(os.path.join(src, "model.safetensors"))
    q8, deq = dict(t), dict(t)
    for name, w in t.items():
        if (w.dim() != 2 or not name.endswith(".weight") or "norm" in name or "embed" in name or "lm_head" in name
                or name.endswith("mlp.gate.weight") or "weights_proj" in name):
            continue
        N, K = w.shape
        bn, bk = -(-N // 128), -(-K // 128)
        blocks = torch.nn.functional.pad(w.float(), (0, bk * 128 - K, 0, bn * 128 - N)).view(bn, 128, bk, 128)
        s = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-12) / 448.0  # e4m3fn max: the checkpoints' range
        q = (blocks / s[:, None, :, None]).to(torch.float8_e4m3fn)
        q8[name] = q.view(bn * 128, bk * 128)[:N, :K].contiguous()
        q8[name[: -len("weight")] + "weight_scale_inv"] = s.contiguous()
        deq[name] = (q.float() * s[:, None, :, None]).view(bn * 128, bk * 128)[:N, :K].to(w.dtype).contiguous()
    save_file(q8, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})
    save_file(deq, os.path.join(deq_dst, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(dst, "config.json")) as fh:
        c = json.load(fh)
    c["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
                                "weight_block_size": [128, 128]}
    with open(os.path.join(dst, "config.json"), "w") as fh:
        json.dump(c, fh)


@pytest.mark.parametrize("name", ["deepseek_v3", "glm_moe_dsa"])
def test_fp8_block_scaled_weights_match_dequantized_reference(tmp_path, name):
    """FP8 weights kept FP8 (per-row block scales, interleaved-RoPE rows permuted with their
    scales, kv_b split into W_UK / W_UV, the indexer's wq_b / wk) give the tokens of the
    reference on the dequantized weights."""
    from transformers import AutoModelForCausalLM

    from kiln.engine.request import SamplingParams

    src, dst, deq = str(tmp_path / "bf"), str(tmp_path / "fp8"), str(tmp_path / "deq")
    build(name, src, index_topk=12)
    _fp8_checkpoint(src, dst, deq)
    hf = AutoModelForCausalLM.from_pretrained(deq, dtype=torch.float32).eval()
    eng = engine(dst)
    layer = eng.model.layers[0]
    assert layer.q_b.dtype == torch.float8_e4m3fn and layer.w_uk_scale is not None
    g = torch.Generator().manual_seed(8)
    prompts = [torch.randint(0, 384, (n,), generator=g).tolist() for n in (7, 33)]
    reqs = eng.generate(prompts, SamplingParams(max_new_tokens=12, ignore_eos=True))
    for ids, r in zip(prompts, reqs):
        assert r.output_ids == hf_greedy(hf, ids, 12), name


@pytest.mark.parametrize("name", ["deepseek_v3", "glm_moe_dsa"])
def test_dense_fp8_dequantized_at_load(tmp_path, name, monkeypatch):
    """KILN_DENSE_FP8=0 (decoder.DENSE_FP8 False): the checkpoint's FP8 attention projections, dense
    and shared-expert MLPs are dequantized once at load into the model dtype (no scale left), the
    routed experts stay FP8, and greedy tokens equal the reference on the dequantized weights."""
    from transformers import AutoModelForCausalLM

    from kiln.engine.request import SamplingParams
    from kiln.models import decoder

    monkeypatch.setattr(decoder, "DENSE_FP8", False)
    src, dst, deq = str(tmp_path / "bf"), str(tmp_path / "fp8"), str(tmp_path / "deq")
    build(name, src, index_topk=12)
    _fp8_checkpoint(src, dst, deq)
    hf = AutoModelForCausalLM.from_pretrained(deq, dtype=torch.float32).eval()
    eng = engine(dst)
    layers = eng.model.layers
    for n in ("w_a", "q_b", "w_uk", "w_uv", "o"):
        assert getattr(layers[0], n).dtype == torch.float32 and getattr(layers[0], n + "_scale") is None, n
    moe = [l for l in layers if l.moe]
    assert moe and all(l.w_gu.dtype == torch.float8_e4m3fn for l in moe)
    assert all(l.shared_gate_up is None or l.shared_gate_up_scale is None for l in moe)
    g = torch.Generator().manual_seed(8)
    prompts = [torch.randint(0, 384, (n,), generator=g).tolist() for n in (7, 33)]
    reqs = eng.generate(prompts, SamplingParams(max_new_tokens=12, ignore_eos=True))
    for ids, r in zip(prompts, reqs):
        assert r.output_ids == hf_greedy(hf, ids, 12), name


def test_fp8_kv_cache_stores_the_latent_in_e4m3(sparse):
    """kv_cache_dtype=fp8: the latent, rope key and indexer key are stored e4m3 (scale 1,
    clamped), i.e. the full-precision cache rounded to fp8. Greedy agreement is measured on a
    real checkpoint (tests/test_kv_fp8.py with KILN_TEST_MODEL); a random model is too
    chaotic for it (a top-12 DSA selection flips on fp8 keys)."""
    from kiln.engine.request import SamplingParams

    name, path, _ = sparse
    prompt = torch.randint(0, 384, (30,), generator=torch.Generator().manual_seed(9)).tolist()
    sp = SamplingParams(max_new_tokens=1, ignore_eos=True)
    ref, fp8 = engine(path), engine(path, kv_cache_dtype="fp8")
    (a,), (b,) = ref.generate([prompt], sp), fp8.generate([prompt], sp)
    pages = [p for p in range(1, 9)]  # the prompt's 8 pages, the first ones handed out
    rows = torch.tensor([p * 4 + j for p in pages for j in range(4)])[: len(prompt)]
    assert all(y.dtype == torch.float8_e4m3fn for y in [*fp8.runner.k_caches, *fp8.runner.v_caches])
    # Layer 0 sees identical inputs, so its cache is exactly the rounded one; later layers'
    # inputs already went through fp8 attention.
    for x, y in ((ref.runner.k_caches[0], fp8.runner.k_caches[0]), (ref.runner.v_caches[0], fp8.runner.v_caches[0])):
        assert torch.equal(y[rows].float(), x[rows].clamp(-448, 448).to(torch.float8_e4m3fn).float()), name


# -- real checkpoints (opt-in; each under 2B parameters) --------------------------------

REAL = os.environ.get("KILN_TEST_MLA_REAL")  # comma-separated local paths or HF repo ids
# Override a DSA checkpoint's index_topk (e.g. 8) so short prompts run the sparse path.
REAL_TOPK = os.environ.get("KILN_TEST_MLA_REAL_TOPK")


@pytest.mark.skipif(not REAL, reason="set KILN_TEST_MLA_REAL to run")
@pytest.mark.parametrize("repo", (REAL or "").split(",") if REAL else [])
def test_real_checkpoint_matches_transformers(repo, tmp_path):
    """e.g. KILN_TEST_MLA_REAL=inference-optimization/GLM-5.3-0.6B-A0.4B,BAAI/OpenSeek-Small-v1-SFT:
    fp32 logits over a real prompt and 16 greedy tokens against transformers, loaded one after
    the other to keep host memory to one copy of the weights at a time."""
    import gc

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kiln.config import ModelConfig
    from kiln.engine.request import SamplingParams
    from kiln.models.loader import resolve_model_path

    path = resolve_model_path(repo)
    if REAL_TOPK:  # the same weights under a config with a small index_topk
        for f in os.listdir(path):
            os.symlink(os.path.join(path, f), tmp_path / f)
        with open(os.path.join(path, "config.json")) as fh:
            c = json.load(fh)
        c["index_topk"] = int(REAL_TOPK)
        os.unlink(tmp_path / "config.json")
        with open(tmp_path / "config.json", "w") as fh:
            json.dump(c, fh)
        path = str(tmp_path)
    tok = AutoTokenizer.from_pretrained(path)
    prompts = [tok(p)["input_ids"] for p in ["The capital of France is", "def fibonacci(n):\n    "]]
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    with torch.no_grad():
        want_logits = hf(torch.tensor([prompts[0]]), use_cache=False).logits[0]
    want = [hf_greedy(hf, p, 16) for p in prompts]
    del hf
    gc.collect()
    cfg = ModelConfig.from_pretrained(path)
    eng = engine(path, page_size=8, num_pages=64, max_model_len=128, max_prefill_tokens=16, max_num_seqs=2)
    with torch.no_grad():
        got_logits = eng.model.forward_logits(torch.tensor(prompts[0]))
    err = (got_logits - want_logits).abs().max().item()
    got = [r.output_ids for r in eng.generate(prompts, SamplingParams(max_new_tokens=16, ignore_eos=True))]
    print(f"{repo}: {cfg.architecture} max |dlogit| {err:.2e} (max |logit| {want_logits.abs().max():.1f}); "
          f"greedy {got == want}: {[tok.decode(g) for g in got]}")
    assert err < 1e-3 * want_logits.abs().max().item()
    assert got == want
