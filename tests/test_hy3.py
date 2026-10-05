"""Hy3 (tencent/Hy3, tencent/Hy3-FP8) against Hugging Face transformers' hy_v3 (5.15) on the same
random weights: GQA with per-head q/k RMSNorm, a dense first layer, sigmoid routing with a
selection-only expert bias, renormalised and scaled top-k weights, and a shared expert.

The config is the real one (https://huggingface.co/tencent/Hy3/raw/main/config.json, read
2026-10-03) with fewer layers and experts and smaller dims; the checkpoint carries the hub's
tensor names (https://huggingface.co/tencent/Hy3/raw/main/model.safetensors.index.json), which
transformers renames on load (conversion_mapping.py, "hy_v3"). The FP8 variant stores every
linear as e4m3 with a scalar BF16 weight_scale and an unused static input_scale, as Hy3-FP8's
shard headers do.
"""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

REAL = {  # https://huggingface.co/tencent/Hy3/raw/main/config.json
    "architectures": ["HYV3ForCausalLM"], "bos_token_id": 120000, "dtype": "bfloat16",
    "enable_attention_fp32_softmax": False, "enable_lm_head_fp32": True, "enable_moe_fp32_combine": False,
    "eod_token_id": 120026, "eos_token_id": 120025, "expert_hidden_dim": 1536, "moe_intermediate_size": 1536,
    "first_k_dense_replace": 1, "head_dim": 128, "hidden_act": "silu", "hidden_size": 4096,
    "initializer_range": 0.006, "intermediate_size": 13312, "max_position_embeddings": 262144, "model_type": "hy_v3",
    "moe_router_enable_expert_bias": True, "moe_router_use_sigmoid": True, "num_attention_heads": 64,
    "num_experts": 192, "num_experts_per_tok": 8, "num_hidden_layers": 80, "num_key_value_heads": 8,
    "num_shared_experts": 1, "output_router_logits": True, "pad_token_id": 120002, "qk_norm": True,
    "rms_norm_eps": 1e-05, "rope_parameters": {"rope_theta": 11158840.0, "rope_type": "default"}, "route_norm": True,
    "router_scaling_factor": 2.826, "sep_token_id": 120007, "tie_word_embeddings": False,
    "transformers_version": "5.6.0", "use_cache": True, "use_grouped_mm": False, "vocab_size": 120832,
    "num_nextn_predict_layers": 1,
}
# https://huggingface.co/tencent/Hy3-FP8/raw/main/config.json
FP8_QUANT = {"activation_scheme": "static", "ignored_layers": ["lm_head", "model.embed_tokens"], "quant_method": "fp8",
             "kv_cache_scheme": "static"}
# Layer 1's tensors (experts collapsed to expert 0) in Hy3's model.safetensors.index.json
LAYER1 = {"input_layernorm.weight", "mlp.expert_bias", "mlp.experts.0.down_proj.weight",
          "mlp.experts.0.gate_proj.weight", "mlp.experts.0.up_proj.weight", "mlp.router.gate.weight",
          "mlp.shared_mlp.down_proj.weight", "mlp.shared_mlp.gate_proj.weight", "mlp.shared_mlp.up_proj.weight",
          "post_attention_layernorm.weight", "self_attn.k_norm.weight", "self_attn.k_proj.weight",
          "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.q_proj.weight", "self_attn.v_proj.weight"}

SMALL = dict(num_hidden_layers=3, hidden_size=64, intermediate_size=96, moe_intermediate_size=32,
             expert_hidden_dim=32, head_dim=16, num_attention_heads=4, num_key_value_heads=2, num_experts=16,
             num_experts_per_tok=4, vocab_size=384, bos_token_id=0, eos_token_id=2, pad_token_id=1)


def hub_names(sd: dict, E: int, I: int) -> dict:
    """transformers' state dict -> the hub checkpoint's names and per-expert layout."""
    out = {}
    for name, t in sd.items():
        if name.endswith("mlp.experts.gate_up_proj"):  # [E, 2I, H], gate rows first
            pre = name[: -len("gate_up_proj")]
            for e in range(E):
                out[f"{pre}{e}.gate_proj.weight"], out[f"{pre}{e}.up_proj.weight"] = t[e, :I], t[e, I:]
        elif name.endswith("mlp.experts.down_proj"):  # [E, H, I]
            pre = name[: -len("down_proj")]
            for e in range(E):
                out[f"{pre}{e}.down_proj.weight"] = t[e]
        else:
            out[name.replace("mlp.gate.weight", "mlp.router.gate.weight")
                .replace("mlp.e_score_correction_bias", "mlp.expert_bias")
                .replace("mlp.shared_experts.", "mlp.shared_mlp.")] = t
    return {k: v.contiguous() for k, v in out.items()}


def per_tensor_fp8(w: torch.Tensor):
    """e4m3fn with one scale for the whole tensor (max 448, the checkpoint's format)."""
    s = (w.abs().max() / 448.0).clamp(min=1e-12).to(torch.bfloat16)
    q = (w.float() / s.float()).to(torch.float8_e4m3fn)
    return q, s


def build(path, fp8=False, **dims):
    from transformers import HYV3Config, HYV3ForCausalLM

    c = {**REAL, **SMALL, **dims}
    hf_cfg = HYV3Config.from_dict({k: v for k, v in c.items() if k != "quantization_config"})
    hf_cfg._attn_implementation = "eager"
    model = HYV3ForCausalLM(hf_cfg)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_((1 + 0.1 * torch.randn(p.shape, generator=g)) if name.endswith("norm.weight")
                    else 0.08 * torch.randn(p.shape, generator=g))
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):  # makes the selection differ from the weights' order
                b.copy_(0.1 * torch.randn(b.shape, generator=g))
    model.eval()
    sd = hub_names(model.state_dict(), c["num_experts"], c["moe_intermediate_size"])
    if fp8:
        c["quantization_config"] = FP8_QUANT
        ref = {}
        for name in list(sd):
            if name.endswith("proj.weight") and "embed" not in name:
                q, s = per_tensor_fp8(sd[name])
                sd[name], sd[name[: -len("weight")] + "weight_scale"] = q, s
                sd[name[: -len("weight")] + "input_scale"] = torch.ones(1)
                ref[name] = q.float() * s.float()
        model.load_state_dict(_from_hub(ref, model, c), strict=False)
    os.makedirs(path, exist_ok=True)
    save_file(sd, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    json.dump(c, open(os.path.join(path, "config.json"), "w"))
    return model


def _from_hub(ref: dict, model, c) -> dict:
    """Dequantized hub tensors -> transformers' state dict entries (the inverse of hub_names)."""
    out, E, I = {}, c["num_experts"], c["moe_intermediate_size"]
    sd = model.state_dict()
    for name, t in sd.items():
        if name.endswith("mlp.experts.gate_up_proj"):
            pre = name[: -len("gate_up_proj")]
            out[name] = torch.stack([torch.cat([ref[f"{pre}{e}.gate_proj.weight"], ref[f"{pre}{e}.up_proj.weight"]])
                                     for e in range(E)])
        elif name.endswith("mlp.experts.down_proj"):
            pre = name[: -len("down_proj")]
            out[name] = torch.stack([ref[f"{pre}{e}.down_proj.weight"] for e in range(E)])
        else:
            hub = name.replace("mlp.shared_experts.", "mlp.shared_mlp.")
            if hub in ref:
                out[name] = ref[hub]
    return out


@pytest.fixture(scope="module", params=["bf16", "fp8"])
def ref(request, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("hy3" + request.param))
    return request.param, path, build(path, fp8=request.param == "fp8")


def test_checkpoint_names_and_config(ref):
    from safetensors import safe_open

    from kiln.config import ModelConfig

    kind, path, _ = ref
    with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
        names = {k[len("model.layers.1."):] for k in f.keys() if k.startswith("model.layers.1.")}
    names = {n for n in names if ".experts." not in n or ".experts.0." in n}
    if kind == "fp8":
        names = {n for n in names if not n.endswith(("weight_scale", "input_scale"))}
    assert names == LAYER1
    cfg = ModelConfig.from_pretrained(path)
    assert cfg.moe_layers == (1, 2) and cfg.qk_norm and cfg.router_scoring == "sigmoid" and cfg.router_bias
    assert cfg.n_shared_experts == 1 and cfg.routed_scaling_factor == 2.826
    assert (cfg.quant_block == 128) == (kind == "fp8")


def test_logits_match(ref):
    from kiln.config import ModelConfig
    from kiln.models.loader import load_model

    kind, path, hf = ref
    cfg = ModelConfig.from_pretrained(path)
    ids = torch.randint(0, 384, (37,), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        want = hf(ids.unsqueeze(0)).logits[0]
        variants = [(False, 448.0)] + ([(True, 448.0), (True, 240.0)] if kind == "fp8" else [])
        for keep, fp8_max in variants:  # 240: trn1's e4m3 max, blocks past it are halved exactly
            m = load_model(path, cfg, torch.float32, torch.device("cpu"), 512, keep_fp8=keep, fp8_max=fp8_max)
            if keep:
                assert m.layers[1].w_gu.dtype == torch.float8_e4m3fn and m.layers[1].shared_down.dtype == torch.float8_e4m3fn
            assert (m.forward_logits(ids) - want).abs().max().item() < 1e-4, (keep, fp8_max)
            m.MOE_GATHER_MAX_PAIRS = 0
            assert (m.forward_logits(ids) - want).abs().max().item() < 1e-4, (keep, fp8_max)


def test_paged_chunked_greedy_matches_reference(ref):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    _, path, hf = ref
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


def test_tp2_matches_tp1(tmp_path):
    """8 query / 2 KV heads, routed and shared experts sharded over two ranks."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    build(str(tmp_path), fp8=True)
    kw = dict(model_path=str(tmp_path), device="cpu", dtype=torch.float32, page_size=4, num_pages=128,
              max_num_seqs=2, max_model_len=128, max_prefill_tokens=8)
    prompts = [[5, 9, 11, 200, 3, 77, 12, 13, 14, 15, 16], list(range(40, 70))]
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True)
    want = [r.output_ids for r in LLMEngine(EngineConfig(**kw)).generate(prompts, sp)]
    two = LLMEngine(EngineConfig(tp=2, **kw))
    try:
        got = [r.output_ids for r in two.generate(prompts, sp)]
    finally:
        two.close()
    assert got == want
