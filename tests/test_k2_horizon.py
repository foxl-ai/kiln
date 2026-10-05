"""K2-Horizon (IFM/K2-Horizon-375B-A23B) against the model repository's own modeling code
(tests/reference/k2_horizon, Apache-2.0) on the same random weights: plain GQA with partial RoPE
over the first rope_head_dim / 2 dims of each head half, dense first layers, sigmoid routing with
a selection-only gate bias, renormalised and scaled top-k weights, and a shared expert. Also FP8
quantization at load (the checkpoint is BF16 only, 758 GB), bounded against the BF16 weights.

The config is the real one (https://huggingface.co/IFM/K2-Horizon-375B-A23B/raw/main/config.json,
read 2026-10-03) with fewer layers and experts and smaller dims.
"""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

REAL = {  # https://huggingface.co/IFM/K2-Horizon-375B-A23B/raw/main/config.json
    "architectures": ["K2HorizonForCausalLM"], "attention_bias": False, "attention_dropout": 0.0,
    "attention_gate_func": None, "bos_token_id": 0, "decoder_sparse_step": 1, "dtype": "bfloat16", "eos_token_id": 1,
    "head_dim": 128, "hidden_act": "silu", "hidden_size": 6144, "initializer_range": 0.01, "intermediate_size": 16384,
    "layernorm_num_groups": 1, "max_position_embeddings": 524288, "mlp_only_layers": [0, 1, 2],
    "model_type": "k2_horizon", "moe_gate_bias": True, "moe_intermediate_size": 1792, "mova_num_experts": 0,
    "mova_num_experts_per_tok": 0, "norm_topk_prob": True, "num_attention_heads": 48, "num_experts": 192,
    "num_experts_per_tok": 8, "num_hidden_layers": 61, "num_key_value_heads": 8, "num_shared_experts": 1,
    "output_router_logits": False, "pad_token_id": None, "query_key_norm": False, "rms_norm_eps": 1e-06,
    "rope_head_dim": 64, "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
    "router_aux_loss_coef": 0.0001, "router_scaling_factor": 2.5, "router_score_func": "sigmoid",
    "sliding_window": None, "tie_word_embeddings": False, "transformers_version": "5.13.0", "use_cache": True,
    "use_sliding_window": False, "vocab_size": 250624,
}
# Layer 3 (the first MoE layer), experts collapsed to expert 0, in
# https://huggingface.co/IFM/K2-Horizon-375B-A23B/raw/main/model.safetensors.index.json
LAYER3 = {"input_layernorm.weight", "mlp.experts.0.down_proj.weight", "mlp.experts.0.gate_proj.weight",
          "mlp.experts.0.up_proj.weight", "mlp.gate.bias", "mlp.gate.weight", "mlp.shared_experts.down_proj.weight",
          "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight", "post_attention_layernorm.weight",
          "self_attn.k_proj.weight", "self_attn.o_proj.weight", "self_attn.q_proj.weight", "self_attn.v_proj.weight"}

SMALL = dict(num_hidden_layers=5, hidden_size=64, intermediate_size=96, moe_intermediate_size=32, head_dim=16,
             rope_head_dim=8, num_attention_heads=4, num_key_value_heads=2, num_experts=16, num_experts_per_tok=4,
             vocab_size=384, eos_token_id=2)


def build(path, **dims):
    from tests.reference.k2_horizon.configuration_k2_horizon import K2HorizonConfig
    from tests.reference.k2_horizon.modeling_k2_horizon import K2HorizonForCausalLM

    c = {**REAL, **SMALL, **dims}
    hf_cfg = K2HorizonConfig(**{k: v for k, v in c.items() if k not in ("architectures", "model_type",
                                                                          "transformers_version", "dtype")})
    hf_cfg._attn_implementation = "eager"
    model = K2HorizonForCausalLM(hf_cfg)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            else:  # gate.bias too: the selection bias reorders the choice
                p.copy_(0.08 * torch.randn(p.shape, generator=g))
    model.eval()
    os.makedirs(path, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(path, "model.safetensors"),
              metadata={"format": "pt"})
    json.dump(c, open(os.path.join(path, "config.json"), "w"))
    return model


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("k2h"))
    return path, build(path)


def test_checkpoint_names_and_config(ref):
    from safetensors import safe_open

    from kiln.config import ModelConfig

    path, _ = ref
    with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
        names = {k[len("model.layers.3."):] for k in f.keys() if k.startswith("model.layers.3.")}
    assert {n for n in names if ".experts." not in n or ".experts.0." in n} == LAYER3
    cfg = ModelConfig.from_pretrained(path)
    assert cfg.moe_layers == (3, 4) and cfg.router_bias and cfg.routed_scaling_factor == 2.5
    assert cfg.attn_layers[0].rope_dim == 16 and cfg.attn_layers[0].rope_freqs == 4
    assert cfg.n_shared_experts == 1 and cfg.quant_block is None


def test_logits_match(ref):
    from kiln.config import ModelConfig
    from kiln.models.loader import load_model

    path, hf = ref
    m = load_model(path, ModelConfig.from_pretrained(path), torch.float32, torch.device("cpu"), 512)
    ids = torch.randint(0, 384, (37,), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        want = hf(ids.unsqueeze(0)).logits[0]
        assert (m.forward_logits(ids) - want).abs().max().item() < 1e-4
        m.MOE_GATHER_MAX_PAIRS = 0
        assert (m.forward_logits(ids) - want).abs().max().item() < 1e-4


def test_paged_chunked_greedy_matches_reference(ref):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, hf = ref
    eng = LLMEngine(EngineConfig(model_path=path, device="cpu", dtype=torch.float32, page_size=4,
                                 num_pages=256, max_num_seqs=3, max_model_len=256, max_prefill_tokens=8))
    g = torch.Generator().manual_seed(2)
    prompts = [torch.randint(0, 384, (n,), generator=g).tolist() for n in (5, 19, 30)]
    reqs = eng.generate(prompts, SamplingParams(max_new_tokens=12, ignore_eos=True))
    for ids, r in zip(prompts, reqs):
        with torch.no_grad():
            want = hf.generate(torch.tensor([ids]), max_new_tokens=12, do_sample=False,
                               eos_token_id=None, pad_token_id=0)[0, len(ids):].tolist()
        assert r.output_ids == want


def kiln_weights_as_reference(m, hf) -> dict:
    """The weights a Kiln model holds (FP8 ones dequantized), under the reference model's names."""
    from kiln.models.quant import dequant, dequant_t

    def w(layer, name):
        return dequant(getattr(layer, name), getattr(layer, name + "_scale"), torch.float32)

    out = {}
    for i, layer in enumerate(m.layers):
        p = f"model.layers.{i}."
        q, k, _ = layer.split
        qkv = w(layer, "qkv")
        out[p + "self_attn.q_proj.weight"], out[p + "self_attn.k_proj.weight"] = qkv[:q], qkv[q : q + k]
        out[p + "self_attn.v_proj.weight"], out[p + "self_attn.o_proj.weight"] = qkv[q + k :], w(layer, "o")
        if not layer.moe:
            gu = w(layer, "gate_up")
            out[p + "mlp.gate_proj.weight"], out[p + "mlp.up_proj.weight"] = gu.chunk(2)
            out[p + "mlp.down_proj.weight"] = w(layer, "down")
            continue
        gu = w(layer, "shared_gate_up")
        out[p + "mlp.shared_experts.gate_proj.weight"], out[p + "mlp.shared_experts.up_proj.weight"] = gu.chunk(2)
        out[p + "mlp.shared_experts.down_proj.weight"] = w(layer, "shared_down")
        egu = w(layer, "w_gu")
        edown = dequant_t(layer.w_down, layer.w_down_scale, torch.float32).transpose(1, 2)  # stored [E, Im, H]
        for e in range(egu.shape[0]):
            g, u = egu[e].chunk(2)
            out[p + f"mlp.experts.{e}.gate_proj.weight"], out[p + f"mlp.experts.{e}.up_proj.weight"] = g, u
            out[p + f"mlp.experts.{e}.down_proj.weight"] = edown[e]
    assert set(out) <= set(hf.state_dict())
    return out


@pytest.mark.parametrize("weight_dtype", ["fp8-experts", "fp8"])
def test_fp8_at_load(ref, weight_dtype):
    """BF16 weights quantized at load to e4m3 (max 240, one scale per row and 128 columns).
    Exactness: the reference model run on Kiln's dequantized FP8 weights gives Kiln's logits.
    Error: one quantized Gaussian weight matrix is off by 2.6% RMS (and so is its product with
    an input); through this random 5-layer model the logits move by 2.6% RMS with the experts
    quantized and 11.9% with every linear (2026-10-03), top-1 agreeing on 36 of 37 positions."""
    import copy

    from kiln.config import EngineConfig, ModelConfig
    from kiln.engine.engine import weight_config
    from kiln.models.loader import load_model

    path, hf = ref
    cfg = weight_config(EngineConfig(model_path=path, weight_dtype=weight_dtype), ModelConfig.from_pretrained(path))
    m = load_model(path, cfg, torch.float32, torch.device("cpu"), 512, keep_fp8=True)
    assert m.layers[3].w_gu.dtype == torch.float8_e4m3fn and m.layers[3].w_down.dtype == torch.float8_e4m3fn
    assert (m.layers[0].gate_up.dtype == torch.float8_e4m3fn) == (weight_dtype == "fp8")
    assert (m.layers[3].shared_gate_up.dtype == torch.float8_e4m3fn) == (weight_dtype == "fp8")
    assert m.layers[3].w_gu.float().abs().max().item() <= 240.0
    ids = torch.randint(0, 384, (37,), generator=torch.Generator().manual_seed(3))
    hq = copy.deepcopy(hf)
    hq.load_state_dict(kiln_weights_as_reference(m, hf), strict=False)
    with torch.no_grad():
        want, wantq, got = hf(ids.unsqueeze(0)).logits[0], hq(ids.unsqueeze(0)).logits[0], m.forward_logits(ids)
    assert (got - wantq).abs().max().item() < 1e-4
    rel = ((got - want).norm() / (want - want.mean()).norm()).item()
    top1 = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
    print(f"{weight_dtype}: relative RMS logit error {rel:.4f}, top-1 agreement {top1:.3f}")
    assert rel < (0.04 if weight_dtype == "fp8-experts" else 0.15) and top1 >= 0.95


def test_tp2_matches_tp1(ref):
    """4 query / 2 KV heads, routed and shared experts and the dense layers over two ranks."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, _ = ref
    kw = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=128, max_num_seqs=2,
              max_model_len=128, max_prefill_tokens=8)
    prompts = [[5, 9, 11, 200, 3, 77, 12, 13, 14, 15, 16], list(range(40, 70))]
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True)
    want = [r.output_ids for r in LLMEngine(EngineConfig(**kw)).generate(prompts, sp)]
    two = LLMEngine(EngineConfig(tp=2, **kw))
    try:
        assert [r.output_ids for r in two.generate(prompts, sp)] == want
    finally:
        two.close()



def test_heads_that_do_not_split_take_a_smaller_attention_tp(tmp_path):
    """K2-Horizon's 48 query heads do not split over 32 ranks: attention runs at the largest TP
    that fits (decoder.attention_tp: 16 for 48 / 8 heads at tp=32) and the MoE at tp. Here 6 query
    / 2 KV heads at tp=4: attention TP 2, greedy equal to tp=1 and the reference."""
    from kiln.config import EngineConfig, ModelConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams
    from kiln.models.decoder import attention_tp

    hf = build(str(tmp_path), num_attention_heads=6, num_key_value_heads=2)
    assert attention_tp(ModelConfig.from_pretrained(str(tmp_path)), 4) == 2
    real = ModelConfig.from_pretrained(str(tmp_path))
    import dataclasses

    k2 = dataclasses.replace(real, attn_layers=(dataclasses.replace(real.attn_layers[0], num_heads=48,
                                                                    num_kv_heads=8),) * real.num_layers)
    assert attention_tp(k2, 32) == 16
    kw = dict(model_path=str(tmp_path), device="cpu", dtype=torch.float32, page_size=4, num_pages=128,
              max_num_seqs=2, max_model_len=128, max_prefill_tokens=8)
    prompts = [[5, 9, 11, 200, 3, 77, 12, 13, 14, 15, 16], list(range(40, 70))]
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True)
    four = LLMEngine(EngineConfig(tp=4, **kw))
    try:
        got = four.generate(prompts, sp)
    finally:
        four.close()
    for ids, r in zip(prompts, got):
        with torch.no_grad():
            want = hf.generate(torch.tensor([ids]), max_new_tokens=10, do_sample=False,
                               eos_token_id=None, pad_token_id=0)[0, len(ids):].tolist()
        assert r.output_ids == want
