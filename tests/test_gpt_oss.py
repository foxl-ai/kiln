"""gpt-oss against Hugging Face transformers' gpt_oss (5.15) on the same random weights:
alternating sliding-window / full attention with per-head sinks, q/k/v/o biases, the biased
top-k-softmax router, expert biases, the clamped GLU, YaRN RoPE, and MXFP4 experts in the
checkpoint's `*_blocks` / `*_scales` layout.

The config is the real one (https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json,
read 2026-10-02) with fewer layers and experts and smaller dims; the checkpoint carries the real
tensor names. Expert weights are quantized to MXFP4 here, and the reference model runs on what
transformers' own dequantizer (integrations/mxfp4.py convert_moe_packed_tensors) makes of the
blocks, so the layout (row order, nibble order, gate/up interleave) is checked against it.
"""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from tests.test_quant import mxfp4

REAL = {  # https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json (layer_types: 18 x [sliding, full])
    "architectures": ["GptOssForCausalLM"], "attention_bias": True, "attention_dropout": 0.0, "eos_token_id": 200002,
    "experts_per_token": 4, "head_dim": 64, "hidden_act": "silu", "hidden_size": 2880,
    "initial_context_length": 4096, "initializer_range": 0.02, "intermediate_size": 2880,
    "layer_types": ["sliding_attention", "full_attention"] * 18, "max_position_embeddings": 131072,
    "model_type": "gpt_oss", "num_attention_heads": 64, "num_experts_per_tok": 4, "num_hidden_layers": 36,
    "num_key_value_heads": 8, "num_local_experts": 128, "output_router_logits": False, "pad_token_id": 199999,
    "quantization_config": {"modules_to_not_convert": ["model.layers.*.self_attn", "model.layers.*.mlp.router",
                                                       "model.embed_tokens", "lm_head"], "quant_method": "mxfp4"},
    "rms_norm_eps": 1e-05,
    "rope_scaling": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 32.0, "original_max_position_embeddings": 4096,
                     "rope_type": "yarn", "truncate": False},
    "rope_theta": 150000, "router_aux_loss_coef": 0.9, "sliding_window": 128, "swiglu_limit": 7.0,
    "tie_word_embeddings": False, "transformers_version": "4.55.0.dev0", "use_cache": True, "vocab_size": 201088,
}
# Layer 0's tensors in https://huggingface.co/openai/gpt-oss-120b/raw/main/model.safetensors.index.json
LAYER0 = {"input_layernorm.weight", "mlp.experts.down_proj_bias", "mlp.experts.down_proj_blocks",
          "mlp.experts.down_proj_scales", "mlp.experts.gate_up_proj_bias", "mlp.experts.gate_up_proj_blocks",
          "mlp.experts.gate_up_proj_scales", "mlp.router.bias", "mlp.router.weight", "post_attention_layernorm.weight",
          "self_attn.k_proj.bias", "self_attn.k_proj.weight", "self_attn.o_proj.bias", "self_attn.o_proj.weight",
          "self_attn.q_proj.bias", "self_attn.q_proj.weight", "self_attn.sinks", "self_attn.v_proj.bias",
          "self_attn.v_proj.weight"}

SMALL = dict(num_hidden_layers=4, hidden_size=64, intermediate_size=96, head_dim=16, num_attention_heads=4,
             num_key_value_heads=2, num_local_experts=8, vocab_size=384, sliding_window=8)


def build(path, **dims):
    """Random gpt-oss with REAL's fields at `dims`; returns the transformers reference model."""
    from transformers import GptOssConfig, GptOssForCausalLM
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    c = {**REAL, "pad_token_id": 1, "eos_token_id": 2, **dims}  # inside the small vocabulary
    c["layer_types"] = REAL["layer_types"][: c["num_hidden_layers"]]
    hf_cfg = GptOssConfig.from_dict({k: v for k, v in c.items() if k != "quantization_config"})
    hf_cfg._attn_implementation = "eager"
    model = GptOssForCausalLM(hf_cfg)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("layernorm.weight") or name == "model.norm.weight":
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            elif "experts.gate_up_proj" in name and not name.endswith("bias"):
                p.copy_(torch.randn(p.shape, generator=g))  # pre-activations past +-7: the clamp bites
            elif "experts.down_proj" in name and not name.endswith("bias"):
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
            else:
                p.copy_((0.5 if "sinks" in name else 0.08) * torch.randn(p.shape, generator=g))
    model.eval()
    tensors = {}
    for name, t in model.state_dict().items():
        if name.endswith(("experts.gate_up_proj", "experts.down_proj")):
            # transformers holds [E, in, out]; the checkpoint stores [E, out, in] as 32-wide blocks.
            w = t.transpose(1, 2).float()
            E, N, K = w.shape
            packed, exps = zip(*(mxfp4(w[e]) for e in range(E)))
            blocks = torch.stack(packed).view(E, N, K // 32, 16)
            scales = torch.stack(exps)
            tensors[name + "_blocks"], tensors[name + "_scales"] = blocks, scales
            with torch.no_grad():
                t.copy_(convert_moe_packed_tensors(blocks, scales, dtype=torch.float32))
        else:
            tensors[name] = t.contiguous()
    os.makedirs(path, exist_ok=True)
    save_file(tensors, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    json.dump(c, open(os.path.join(path, "config.json"), "w"))
    return model


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("gpt_oss"))
    return path, build(path, **SMALL)


def test_checkpoint_names_and_config(ref):
    from safetensors import safe_open

    from kiln.config import ModelConfig

    path, _ = ref
    with safe_open(os.path.join(path, "model.safetensors"), "pt") as f:
        names = {k[len("model.layers.0."):] for k in f.keys() if k.startswith("model.layers.0.")}
    assert names == LAYER0
    cfg = ModelConfig.from_pretrained(path)
    assert [s.window for s in cfg.attn_layers] == [8, None, 8, None] and all(s.sink for s in cfg.attn_layers)
    assert cfg.quant_expert_mxfp4 and cfg.quant_block is None and cfg.moe_layers == (0, 1, 2, 3)
    assert dict(cfg.rope_scaling)["rope_type"] == "yarn" and cfg.router_scoring == "topk_softmax"


def test_swiglu_oai_matches_transformers():
    from transformers import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts

    from kiln.models.decoder import swiglu_oai

    gu = torch.randn(64, 2 * 48) * 10  # well past the +-7 limit
    want = GptOssExperts(GptOssConfig(intermediate_size=48, hidden_size=16, num_local_experts=2))._apply_gate(gu)
    assert torch.equal(swiglu_oai(gu[:, 0::2], gu[:, 1::2], 7.0), want)


@pytest.mark.parametrize("keep,packed", [(True, False), (True, True), (False, False)])
def test_logits_match(ref, keep, packed):
    """FP8 experts (MXFP4 converted losslessly), packed MXFP4, and BF16-dequantized-at-load, each
    on the gather (decode-sized) and the every-expert (prefill-sized) MoE paths."""
    from kiln.config import ModelConfig
    from kiln.models.loader import load_model

    path, hf = ref
    m = load_model(path, ModelConfig.from_pretrained(path), torch.float32, torch.device("cpu"), 512,
                   keep_fp8=keep, packed_mxfp4=packed)
    ids = torch.randint(0, 384, (41,), generator=torch.Generator().manual_seed(3))  # longer than the window
    with torch.no_grad():
        want = hf(ids.unsqueeze(0)).logits[0]
        got = m.forward_logits(ids)
        assert (got - want).abs().max().item() < 1e-4
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
    assert eng.runner.window == 8
    for ids, r in zip(prompts, reqs):
        with torch.no_grad():
            want = hf.generate(torch.tensor([ids]), max_new_tokens=12, do_sample=False,
                               eos_token_id=None, pad_token_id=0)[0, len(ids):].tolist()
        assert r.output_ids == want


def test_tp2_pads_experts_and_matches_tp1(ref):
    """intermediate 96 = 3 MXFP4 blocks: at tp=2 each rank takes 2 whole blocks (64 units) and
    rank 1 is half zero padding; the o_proj and down_proj biases are added once (rank 0)."""
    from kiln.config import EngineConfig, ModelConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams
    from kiln.models.decoder import moe_inter_per_rank

    path, _ = ref
    assert moe_inter_per_rank(ModelConfig.from_pretrained(path), 2, quantized=True) == 64
    kw = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=128,
              max_num_seqs=2, max_model_len=128, max_prefill_tokens=8)
    prompts = [[5, 9, 11, 200, 3, 77, 12, 13, 14, 15, 16], list(range(40, 70))]
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True, logprobs=1)
    one = LLMEngine(EngineConfig(**kw)).generate(prompts, sp)
    two = LLMEngine(EngineConfig(tp=2, **kw))
    try:
        got = two.generate(prompts, sp)
    finally:
        two.close()
    for a, b in zip(one, got):
        assert a.output_ids == b.output_ids
        assert max(abs(x[0] - y[0]) for x, y in zip(a.logprobs, b.logprobs)) < 1e-4


def test_attention_tp_2_of_tp_4_matches_tp1(ref):
    """Attention over groups of 2 ranks, experts over all 4: the o_proj bias is added once per
    attention group (its rank 0), the down bias once overall."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, _ = ref
    kw = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=128,
              max_num_seqs=2, max_model_len=128, max_prefill_tokens=8)
    prompts = [[5, 9, 11, 200, 3, 77, 12, 13, 14, 15, 16], list(range(40, 70))]
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True, logprobs=1)
    one = LLMEngine(EngineConfig(**kw)).generate(prompts, sp)
    four = LLMEngine(EngineConfig(tp=4, attention_tp=2, **kw))
    try:
        got = four.generate(prompts, sp)
    finally:
        four.close()
    for a, b in zip(one, got):
        assert a.output_ids == b.output_ids
        assert max(abs(x[0] - y[0]) for x, y in zip(a.logprobs, b.logprobs)) < 1e-4


def test_real_dims_logits_match(tmp_path):
    """The real hidden 2880 (90 MXFP4 blocks), head_dim 64, 64 / 8 heads and intermediate 2880,
    at 2 layers and 4 experts, top-2 (a smaller vocabulary keeps it near 6 GB of host memory)."""
    from kiln.config import ModelConfig
    from kiln.models.loader import load_model

    hf = build(str(tmp_path), num_hidden_layers=2, num_local_experts=4, num_experts_per_tok=2, experts_per_token=2,
               vocab_size=512)
    m = load_model(str(tmp_path), ModelConfig.from_pretrained(str(tmp_path)), torch.float32, torch.device("cpu"), 64,
                   keep_fp8=True)
    ids = torch.randint(0, 512, (20,), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        want = hf(ids.unsqueeze(0)).logits[0]
        got = m.forward_logits(ids)
    print(f"real dims: max |dlogit| {(got - want).abs().max().item():.2e} of max |logit| {want.abs().max().item():.2f}")
    assert (got - want).abs().max().item() < 1e-3 * want.abs().max().item()
