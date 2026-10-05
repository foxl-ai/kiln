"""Every supported architecture against Hugging Face transformers on the SAME weights.

Tiny randomly initialised models are built by transformers itself and saved as a normal
checkpoint, so this compares Kiln's implementation with the reference implementation
without downloading anything.
"""

import pytest
import torch

ARCHS = ["llama", "llama3_rope", "mistral", "qwen2", "qwen3", "qwen3_moe"]


def build(arch, path, vocab_size=384, tie_word_embeddings=False):
    import transformers as tf

    common = dict(vocab_size=vocab_size, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                  rms_norm_eps=1e-6, tie_word_embeddings=tie_word_embeddings)
    if arch in ("llama", "llama3_rope"):
        extra = {}
        if arch == "llama3_rope":
            extra["rope_scaling"] = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                     "high_freq_factor": 4.0, "original_max_position_embeddings": 64}
        cfg = tf.LlamaConfig(**common, rope_theta=500000.0, **extra)
        model = tf.LlamaForCausalLM(cfg)
    elif arch == "mistral":
        cfg = tf.MistralConfig(**common, sliding_window=None)
        model = tf.MistralForCausalLM(cfg)
    elif arch == "qwen2":
        cfg = tf.Qwen2Config(**common)
        model = tf.Qwen2ForCausalLM(cfg)
    elif arch == "qwen3":
        cfg = tf.Qwen3Config(**common, head_dim=16)
        model = tf.Qwen3ForCausalLM(cfg)
    else:
        cfg = tf.Qwen3MoeConfig(**common, head_dim=16, num_experts=8, num_experts_per_tok=2,
                                moe_intermediate_size=32, norm_topk_prob=True, decoder_sparse_step=1,
                                mlp_only_layers=[1])
        model = tf.Qwen3MoeForCausalLM(cfg)
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.08)
    model.eval()
    model.save_pretrained(path, safe_serialization=True)
    return model


@pytest.fixture(scope="module", params=ARCHS)
def built(request, tmp_path_factory):
    path = tmp_path_factory.mktemp(request.param)
    return request.param, str(path), build(request.param, str(path))


def test_logits_match(built):
    from kiln.config import ModelConfig
    from kiln.models.loader import load_model

    arch, path, hf = built
    cfg = ModelConfig.from_pretrained(path)
    ours = load_model(path, cfg, torch.float32, torch.device("cpu"), 512)
    ids = torch.randint(0, 384, (37,))
    with torch.no_grad():
        ref = hf(ids.unsqueeze(0)).logits[0]
        got = ours.forward_logits(ids)
    assert (got - ref).abs().max().item() < 1e-4, arch


def test_paged_chunked_greedy_generation_matches(built):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    arch, path, hf = built
    eng = LLMEngine(EngineConfig(model_path=path, device="cpu", dtype=torch.float32, page_size=4,
                                 num_pages=256, max_num_seqs=3, max_model_len=256, max_prefill_tokens=8))
    g = torch.Generator().manual_seed(1)
    prompts = [torch.randint(0, 384, (n,), generator=g).tolist() for n in (5, 19, 30)]
    reqs = eng.generate(prompts, SamplingParams(max_new_tokens=12, ignore_eos=True))
    for ids, r in zip(prompts, reqs):
        with torch.no_grad():
            ref = hf.generate(torch.tensor([ids]), max_new_tokens=12, do_sample=False,
                              eos_token_id=None, pad_token_id=0)[0, len(ids):].tolist()
        assert r.output_ids == ref, arch


def test_moe_dense_and_gather_paths_agree(tmp_path):
    from kiln.config import ModelConfig
    from kiln.models.decoder import DecoderForCausalLM
    from kiln.models.loader import load_model

    build("qwen3_moe", str(tmp_path))
    cfg = ModelConfig.from_pretrained(str(tmp_path))
    m = load_model(str(tmp_path), cfg, torch.float32, torch.device("cpu"), 512)
    ids = torch.randint(0, 384, (40,))
    with torch.no_grad():
        gather = m.forward_logits(ids)  # 40 tokens x 2 experts = 80 pairs: gather path
        DecoderForCausalLM.MOE_GATHER_MAX_PAIRS = 0
        try:
            dense = m.forward_logits(ids)
        finally:
            DecoderForCausalLM.MOE_GATHER_MAX_PAIRS = 512
    assert (gather - dense).abs().max().item() < 1e-4
