"""Inkling (thinkingmachines/Inkling-Small, text model) against Hugging Face transformers' inkling
(5.15, InklingForCausalLM) on the same random weights: no RoPE, per-head q / k RMSNorm with
1 / head_dim scaling, hidden-conditioned relative position logits, log scaling on global layers,
causal short convolutions on k, v, the attention output and the MLP output, local / global
layers, sigmoid routing that also weights two shared experts, global scales, embedding norm,
mup-divided logits over the unpadded vocabulary.

The config is the real text_config (https://huggingface.co/thinkingmachines/Inkling-Small/raw/main/config.json,
read 2026-10-03) at small dims; the checkpoint carries the hub's names and layouts (model.llm.*,
gate / up rows interleaved), which transformers renames on load (conversion_mapping.py,
"inkling_mm_model"). log_scaling_n_floor is lowered from 128000 to 8 so that the log scaling acts
inside a short test sequence.
"""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

REAL_TEXT = {  # text_config of https://huggingface.co/thinkingmachines/Inkling-Small/raw/main/config.json
    "model_max_length": 1048576, "torch_dtype": "bfloat16", "hidden_size": 4096, "num_hidden_layers": 42,
    "vocab_size": 201024, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "d_rel": 16,
    "rel_extent": 1024, "q_bias": False, "o_bias": False, "log_scaling_n_floor": 128000, "log_scaling_alpha": 0.1,
    "rms_norm_eps": 1e-06, "use_embed_norm": True,
    "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28,
                        30, 31, 32, 33, 34, 36, 37, 38, 39, 40],
    "dense_mlp_idx": 2, "use_sconv": True, "sconv_kernel_size": 4, "unpadded_vocab_size": 200058,
    "logits_mup_width_multiplier": 16.0, "final_logit_softcapping": None, "swa_head_dim": 128,
    "swa_num_attention_heads": 32, "swa_num_key_value_heads": 8, "sliding_window_size": 512,
    "n_routed_experts": 256, "num_experts_per_tok": 6, "n_shared_experts": 2, "shared_expert_sink": True,
    "dense_intermediate_size": 16384, "intermediate_size": 2048, "route_scale": 8.0, "use_gate_bias": True,
    "gate_activation": "sigmoid", "norm_after_topk": True, "use_global_scale": True,
}
SMALL = dict(hidden_size=64, num_hidden_layers=4, vocab_size=384, unpadded_vocab_size=380, num_attention_heads=4,
             num_key_value_heads=2, head_dim=16, swa_head_dim=16, swa_num_attention_heads=4,
             swa_num_key_value_heads=2, d_rel=4, rel_extent=12, sliding_window_size=8, local_layer_ids=[0, 1, 3],
             dense_mlp_idx=1, n_routed_experts=8, num_experts_per_tok=2, dense_intermediate_size=48,
             intermediate_size=16, log_scaling_n_floor=8)
# Layer 2 (MoE) of https://huggingface.co/thinkingmachines/Inkling-Small/raw/main/model.safetensors.index.json
LAYER2 = {"attn.k_norm.weight", "attn.k_sconv.weight", "attn.q_norm.weight", "attn.rel_logits_proj.proj",
          "attn.v_sconv.weight", "attn.wk_dv.weight", "attn.wo_ud.weight", "attn.wq_du.weight", "attn.wr_du.weight",
          "attn.wv_dv.weight", "attn_norm.weight", "attn_sconv.weight", "mlp.experts.w13_weight",
          "mlp.experts.w2_weight", "mlp.gate.bias", "mlp.gate.global_scale", "mlp.gate.weight",
          "mlp.shared_experts.shared_w13_weight", "mlp.shared_experts.shared_w2_weight", "mlp_norm.weight",
          "mlp_sconv.weight"}


def interleave(gate: torch.Tensor, up: torch.Tensor, dim: int) -> torch.Tensor:
    """[gate; up] halves -> rows alternating gate, up (the checkpoints' w13 layout)."""
    return torch.stack([gate, up], dim=dim + 1).flatten(dim, dim + 1)


RENAMES = [("shared_experts.down_proj", "shared_experts.shared_w2_weight"), ("experts.down_proj", "experts.w2_weight"),
           ("mlp.down_proj.weight", "mlp.w2_md.weight"), ("gate.e_score_correction_bias", "gate.bias"),
           ("self_attn.q_proj", "attn.wq_du"), ("self_attn.k_proj", "attn.wk_dv"), ("self_attn.v_proj", "attn.wv_dv"),
           ("self_attn.r_proj", "attn.wr_du"), ("self_attn.o_proj", "attn.wo_ud"),
           ("self_attn.k_sconv.conv1d", "attn.k_sconv"), ("self_attn.v_sconv.conv1d", "attn.v_sconv"),
           ("attn_sconv.conv1d", "attn_sconv"), ("mlp_sconv.conv1d", "mlp_sconv"), ("self_attn.", "attn."),
           ("input_layernorm", "attn_norm"), ("post_attention_layernorm", "mlp_norm"),
           ("model.layers", "model.llm.layers"), ("model.embed_tokens", "model.llm.embed"),
           ("model.embed_norm", "model.llm.embed_norm"), ("model.norm", "model.llm.norm"), ("lm_head", "model.llm.unembed")]


def hub_state(sd: dict) -> dict:
    """transformers' InklingForCausalLM state dict -> the hub checkpoint's names and layouts (the
    inverse of conversion_mapping.py "inkling_mm_model")."""
    def hub(n):
        for a, b in RENAMES:
            n = n.replace(a, b)
        return n

    out = {}
    for name, t in sd.items():
        if name.endswith("mlp.gate_proj.weight"):  # dense
            out[hub(name).replace("gate_proj.weight", "w13_dn.weight")] = interleave(t, sd[name.replace("gate", "up")], 0)
        elif name.endswith("shared_experts.gate_proj"):
            out[hub(name).replace("gate_proj", "shared_w13_weight")] = interleave(t, sd[name.replace("gate", "up")], 1)
        elif name.endswith("experts.gate_up_proj"):
            half = t.shape[1] // 2
            out[hub(name).replace("gate_up_proj", "w13_weight")] = interleave(t[:, :half], t[:, half:], 1)
        elif not name.endswith(("mlp.up_proj.weight", "shared_experts.up_proj")):
            out[hub(name)] = t
    return {k: v.contiguous() for k, v in out.items()}


def build(path, **dims):
    from transformers.models.inkling.configuration_inkling import InklingTextConfig
    from transformers.models.inkling.modeling_inkling import InklingForCausalLM

    t = {**REAL_TEXT, **SMALL, **dims}
    # transformers 5.15 keeps moe_intermediate_size at its default when dense_intermediate_size is
    # given (see kiln/config.py _inkling), so the reference is told the experts' size explicitly.
    hf_cfg = InklingTextConfig(**{k: v for k, v in t.items() if k not in ("torch_dtype",)},
                               moe_intermediate_size=t["intermediate_size"], eos_token_id=2, pad_token_id=1)
    hf_cfg._attn_implementation = "eager"
    model = InklingForCausalLM(hf_cfg)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            elif "global_scale" in name:
                p.copy_(1 + 0.2 * torch.randn(p.shape, generator=g))
            elif "sconv" in name:
                p.copy_(0.3 * torch.randn(p.shape, generator=g))
            elif "rel_logits_proj" in name or "e_score_correction_bias" in name:
                p.copy_(0.5 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.08 * torch.randn(p.shape, generator=g))
    model.eval()
    os.makedirs(path, exist_ok=True)
    save_file(hub_state(model.state_dict()), os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    json.dump({"architectures": ["InklingForConditionalGeneration"], "model_type": "inkling_mm_model",
               "eos_token_id": 2, "text_config": t}, open(os.path.join(path, "config.json"), "w"))
    return model


def greedy(hf, ids: list[int], n: int) -> list[int]:
    """Greedy continuation by full forwards of the reference. Its cached generate() is not used:
    with this config it left its own uncached forward at step 8 of the 19-token prompt below
    (transformers 5.15, 2026-10-03), while each full forward is the model's definition."""
    seq = list(ids)
    with torch.no_grad():
        for _ in range(n):
            seq.append(int(hf(torch.tensor([seq])).logits[0, -1].argmax()))
    return seq[len(ids):]


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("inkling"))
    return path, build(path)


def test_checkpoint_names_and_config(ref):
    from safetensors import safe_open

    from kiln.config import ModelConfig

    path, _ = ref
    with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
        keys = set(f.keys())
    assert {k[len("model.llm.layers.2."):] for k in keys if k.startswith("model.llm.layers.2.")} == LAYER2
    assert {k for k in keys if ".layers." not in k} == {"model.llm.embed.weight", "model.llm.embed_norm.weight",
                                                         "model.llm.norm.weight", "model.llm.unembed.weight"}
    cfg = ModelConfig.from_pretrained(path)
    assert cfg.moe_intermediate_size == 16 and cfg.intermediate_size == 48 and cfg.vocab_size == 380
    assert [(s.window, s.rel_extent, s.log_scaled) for s in cfg.attn_layers] == [
        (8, 8, False), (8, 8, False), (None, 12, True), (8, 8, False)]
    assert cfg.moe_layers == (1, 2, 3) and cfg.router_shared_rows == 2 and cfg.sconv_kernel == 4


def test_logits_match(ref):
    from kiln.config import ModelConfig
    from kiln.models.loader import load_model

    path, hf = ref
    m = load_model(path, ModelConfig.from_pretrained(path), torch.float32, torch.device("cpu"), 512)
    ids = torch.randint(0, 380, (37,), generator=torch.Generator().manual_seed(3))  # past window, rel extents, floor
    with torch.no_grad():
        want = hf(ids.unsqueeze(0)).logits[0]
        assert want.shape[-1] == 380
        assert (m.forward_logits(ids) - want).abs().max().item() < 1e-4
        m.MOE_GATHER_MAX_PAIRS = 0
        assert (m.forward_logits(ids) - want).abs().max().item() < 1e-4


@pytest.mark.parametrize("spec", [None, "ngram"])
def test_paged_chunked_greedy_matches_reference(ref, spec):
    """Chunked prefill, decode and (with n-gram drafts) speculative verify, all reading the
    convolution histories and relative logits through the paged cache."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, hf = ref
    kw = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256, max_num_seqs=3,
              max_model_len=256, max_prefill_tokens=8)
    if spec:
        kw.update(spec_method=spec, spec_k=3)
    eng = LLMEngine(EngineConfig(**kw))
    g = torch.Generator().manual_seed(2)
    prompts = [torch.randint(0, 380, (n,), generator=g).tolist() for n in (5, 19, 30)]
    prompts[2][15:30] = prompts[2][0:15]  # repeats give the n-gram drafter something to propose
    reqs = eng.generate(prompts, SamplingParams(max_new_tokens=12, ignore_eos=True))
    for ids, r in zip(prompts, reqs):
        assert r.output_ids == greedy(hf, ids, 12)


def test_prefix_cache_hit_reads_histories(ref):
    """A second request sharing a cached prefix starts past it: its first convolutions read the
    histories the first request left in the shared pages."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, hf = ref
    eng = LLMEngine(EngineConfig(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256,
                                 max_num_seqs=2, max_model_len=256, max_prefill_tokens=8))
    base = torch.randint(0, 380, (21,), generator=torch.Generator().manual_seed(7)).tolist()
    sp = SamplingParams(max_new_tokens=6, ignore_eos=True)
    eng.generate([base], sp)
    (r,) = eng.generate([base + [11, 12, 13]], sp)
    assert r.num_cached_tokens >= 16
    assert r.output_ids == greedy(hf, base + [11, 12, 13], 6)


def test_tp2_matches_tp1(ref):
    """4 query / 2 KV heads, r_proj and the k / v convolutions with their heads, the shared experts'
    units split over two ranks (one each)."""
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


def test_attention_tp_2_of_tp_4_matches_tp1(ref):
    """Attention (with r_proj and the k / v convolutions) over groups of 2 ranks, the experts
    and the shared experts over all 4."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, _ = ref
    kw = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=128, max_num_seqs=2,
              max_model_len=128, max_prefill_tokens=8)
    prompts = [[5, 9, 11, 200, 3, 77, 12, 13, 14, 15, 16], list(range(40, 70))]
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True)
    want = [r.output_ids for r in LLMEngine(EngineConfig(**kw)).generate(prompts, sp)]
    four = LLMEngine(EngineConfig(tp=4, attention_tp=2, **kw))
    try:
        assert [r.output_ids for r in four.generate(prompts, sp)] == want
    finally:
        four.close()
