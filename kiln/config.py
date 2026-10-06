from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class AttnSpec:
    """One layer's attention shape. Uniform models repeat one; hybrids mix kinds."""

    num_heads: int
    num_kv_heads: int
    head_dim: int  # query / key
    v_head_dim: int
    rope_dim: int  # leading dims of q / k that RoPE rotates
    rope_theta: float
    window: int | None = None  # sliding window: a query sees the last `window` keys
    sink: bool = False  # per-query-head attention-sink logit in the softmax
    # Rotate only the first rope_freqs of the rope_dim / 2 frequency pairs (pair j is dims j and
    # j + rope_dim / 2); the other pairs pass through. None: all of them. K2-Horizon's
    # rope_head_dim 64 of head_dim 128 is rope_dim 128, rope_freqs 32 (see _k2_horizon).
    rope_freqs: int | None = None
    scale: float | None = None  # score scale (None: head_dim ** -0.5)
    # Hidden-conditioned relative position logits (Inkling): bias for key distances below
    # rel_extent, from a per-head d_rel projection of the layer input.
    rel_extent: int = 0
    log_scaled: bool = False  # queries and position logits scaled by ModelConfig.log_scaling
    # Multi-head latent attention (models/mla.py MLASpec). For an MLA layer num_kv_heads,
    # head_dim and v_head_dim describe its paged cache: one latent "head" whose K is the
    # kv_lora_rank latent and whose V is the rope key (plus the DSA indexer key).
    mla: object | None = None


@dataclass(frozen=True)
class LinearSpec:
    """One linear-attention layer (models/linear_attn.py): a short causal conv over q, k, v
    and a gated delta rule whose state is a fixed-size matrix per head instead of a KV cache.

    kind "gdn": Gated DeltaNet (Qwen3.5 / Qwen3.8 / Qwen3-Next), one decay per head and token,
    num_v_heads >= num_k_heads (each k head serves num_v_heads / num_k_heads v heads).
    kind "kda": Kimi Delta Attention (Kimi Linear / Kimi K3, GLM-5.3-Flash), one decay per key
    channel, q, k and v heads all equal."""

    kind: str
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel: int
    gate_act: str = "silu"  # activation of the output gate in the gated RMSNorm
    gate_rank: int = 0  # KDA output gate: 0 = full-rank g_proj, else g_a_proj / g_b_proj of this rank
    lower_bound: float | None = None  # KDA "safe gate": decay = lower_bound * sigmoid(...)


@dataclass(frozen=True)
class ModelConfig:
    """The subset of a Hugging Face config.json that Kiln's model code reads."""

    architecture: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    eos_token_ids: tuple[int, ...]
    qk_norm: bool = False  # per-head RMSNorm on q and k before RoPE (Qwen3)
    qkv_bias: bool = False  # bias on q/k/v projections (Qwen2, attention_bias=True Llamas)
    rope_scaling: tuple | None = None  # sorted (key, value) items, e.g. llama3 scaling
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    norm_topk_prob: bool = True
    moe_layers: tuple[int, ...] = ()  # indices of layers whose MLP is a sparse MoE
    # Per-layer token mixer (None = uniform attention). An entry's type is the layer's kind:
    # AttnSpec (softmax attention over the paged KV cache) or LinearSpec (recurrent state).
    attn_layers: tuple[AttnSpec | LinearSpec, ...] | None = None
    attn_output_gate: bool = False  # attention output * sigmoid(gate), gate rows in q_proj (Qwen3.5)
    norm_offset: bool = False  # RMSNorm weights are stored zero-centred, applied as (1 + w) (Qwen3.5)
    v_scale: float | None = None  # values scaled before attention (MiMo-V2)
    # "softmax" (Qwen3-MoE), "sigmoid" (DeepSeek-V3 style) or "topk_softmax" (gpt-oss: top-k of
    # the logits, softmax over just those k)
    router_scoring: str = "softmax"
    router_bias: bool = False  # e_score_correction_bias added for expert choice only
    routed_scaling_factor: float = 1.0
    n_shared_experts: int = 0  # DeepSeek-V3 style: a dense MLP of n x moe_intermediate_size beside the experts
    n_group: int = 1  # DeepSeek-V3 node-limited routing: experts in n_group groups, topk_group kept
    topk_group: int = 1
    fused_qkv_weights: bool = False  # checkpoint stores self_attn.qkv_proj (MiMo-V2)
    # Multi-token-prediction layers in the checkpoint (num_nextn_predict_layers) and the
    # attention they use; MiMo-V2's MTP layer uses the sliding-window configuration
    # (vLLM model_executor/models/mimo_v2_mtp.py, v0.30.0).
    mtp_layers: int = 0
    mtp_spec: AttnSpec | None = None
    # DeepSeek-V3 / GLM-5 MTP (models/mtp.py): the layer's checkpoint prefix
    # (model.layers.<num_hidden_layers>; None = MiMo's model.mtp.layers.0), whether its MLP is the
    # MoE, and index_share_for_mtp_iteration (later draft passes reuse the first pass's DSA selection).
    mtp_prefix: str | None = None
    mtp_moe: bool = False
    mtp_index_share: bool = False
    # Checkpoint names of the MTP head's own weights beside its layer: "mimo" / DeepSeek
    # (enorm, hnorm, eh_proj, the final norm; models/mtp.py names) or "qwen3_5" (mtp.{pre_fc_norm_embedding,
    # pre_fc_norm_hidden, fc, norm}: vLLM v0.30.0 model_executor/models/qwen3_5_mtp.py
    # Qwen3_5MultiTokenPredictor).
    mtp_names: str = "mimo"
    # Checkpoint weight quantization (quantization_config): FP8 with block_size scales for
    # dense weights, optionally MXFP4 experts; `quant_ignored` modules stay unquantized.
    quant_block: int | None = None
    quant_expert_block: int | None = None
    quant_expert_mxfp4: bool = False  # experts are stored MXFP4 (store_dtype: mxfp4)
    quant_ignored: tuple[str, ...] = ()
    # The checkpoint's scales cover blocks of input columns (FP8 weight_block_size, MXFP4), so a
    # tensor-parallel column shard must not split a block. Per-tensor scales (Hy3-FP8) and FP8
    # made at load (quantized_at_load, after sharding) have no such constraint.
    quant_blockwise: bool = False
    o_bias: bool = False  # bias on o_proj (gpt-oss attention_bias)
    # Routed experts: "silu" (SwiGLU) or "swiglu_oai" (gpt-oss's clamped GLU, see
    # decoder.swiglu_oai); expert_bias: gate_up / down biases per expert (gpt-oss).
    moe_act: str = "silu"
    swiglu_limit: float = 7.0
    expert_bias: bool = False
    router_logit_bias: bool = False  # the router linear has a bias added to its logits (gpt-oss)
    # Inkling (see _inkling): the router also scores router_shared_rows shared experts, whose
    # weights scale them; a scalar global scale on the router and on the dense MLPs; RMSNorm of
    # the embeddings; logits of h / logits_divisor; causal depthwise short convolutions of kernel
    # sconv_kernel (with a residual) on k, v, the attention output and the MLP output; relative
    # position logits of d_rel dims; log_scaling = (n_floor, alpha) for log_scaled layers.
    router_shared_rows: int = 0
    router_global_scale: bool = False
    dense_mlp_scale: bool = False
    embed_norm: bool = False
    logits_divisor: float = 1.0
    sconv_kernel: int = 0
    d_rel: int = 0
    log_scaling: tuple[float, float] | None = None
    # Hyper-connection hybrids (GLM-5.3-Flash, Qwen3.8-Flash-Next): their residual streams and
    # per-model extras, a models/hybrid.py HybridSpec.
    hybrid: object | None = None

    def truncated(self, n: int) -> "ModelConfig":
        """The first n layers only (debugging a large checkpoint layer by layer)."""
        import dataclasses

        return dataclasses.replace(
            self, num_layers=n, moe_layers=tuple(i for i in self.moe_layers if i < n),
            attn_layers=self.attn_layers[:n] if self.attn_layers is not None else None)

    def quantized_at_load(self, dense: bool, block: int = 128) -> "ModelConfig":
        """A BF16 checkpoint's experts (and, with dense, its other linears) quantized to FP8 by
        Kiln at load: per (row, block input columns) absmax scales onto e4m3 max 240
        (models/quant.py quantize_fp8_rows). Embeddings, lm_head, norms and routers stay."""
        import dataclasses

        if self.quant_block is not None or self.quant_expert_block is not None:
            raise ValueError("the checkpoint is already quantized; --weight-dtype fp8 is for BF16 checkpoints")
        return dataclasses.replace(self, quant_block=block if dense else None,
                                   quant_expert_block=block if self.num_experts else None)

    def is_quantized(self, module: str, expert: bool = False) -> bool:
        """`module` like "model.layers.3.self_attn.qkv_proj"."""
        if module in self.quant_ignored:
            return False
        return (self.quant_expert_block if expert else self.quant_block) is not None

    @classmethod
    def from_pretrained(cls, path: str) -> "ModelConfig":
        with open(os.path.join(path, "config.json")) as f:
            c = json.load(f)
        eos = c.get("eos_token_id")
        gen_path = os.path.join(path, "generation_config.json")
        if os.path.exists(gen_path):
            with open(gen_path) as f:
                eos = json.load(f).get("eos_token_id", eos)
        eos_ids = tuple(eos) if isinstance(eos, list) else ((eos,) if eos is not None else ())
        from .models import hybrid, mla

        arch = (c.get("architectures") or [None])[0]
        if arch in mla.ARCHITECTURES:
            return mla.model_config(cls, c, eos_ids)
        if arch in GQA_MOE:  # these parse their own RoPE (gpt-oss: YaRN)
            return GQA_MOE[arch](cls, c, eos_ids)
        if (c.get("architectures") or [None])[0] in hybrid.ARCHITECTURES:
            return hybrid.config_from_hf(cls, c, eos_ids)
        # transformers 5 writes rope settings under "rope_parameters"; older files use
        # a top-level "rope_theta".
        rope = dict(c.get("rope_parameters") or {})
        rope_theta = rope.get("rope_theta", c.get("rope_theta", 10000.0))
        scaling = dict(c.get("rope_scaling") or {})
        rope_type = scaling.get("rope_type", scaling.get("type", rope.get("rope_type", "default")))
        if rope_type not in ("default", "llama3"):
            raise NotImplementedError(f"rope scaling {rope_type!r} is not supported yet")
        if rope_type == "llama3":
            scaling = {**rope, **scaling, "rope_type": "llama3"}
        if arch == "MiMoV2ForCausalLM":
            return cls._mimo_v2(c, eos_ids)
        from .models import linear_attn

        if arch in linear_attn.ARCHITECTURES:  # linear-attention hybrids
            return linear_attn.config_from_hf(cls, arch, c, eos_ids)
        known = {
            "LlamaForCausalLM": (False, bool(c.get("attention_bias", False))),
            "MistralForCausalLM": (False, False),
            "Qwen2ForCausalLM": (False, True),
            "Qwen3ForCausalLM": (True, False),
            "Qwen3MoeForCausalLM": (True, False),
        }
        if arch not in known:
            raise NotImplementedError(f"architecture {arch} is not supported (have {sorted([*known, *GQA_MOE])} "
                                      "+ MiMoV2ForCausalLM)")
        qk_norm, qkv_bias = known[arch]
        if c.get("use_sliding_window") or (arch == "MistralForCausalLM" and c.get("sliding_window")):
            raise NotImplementedError("sliding-window attention is only wired for MiMo-V2 so far")
        n_layers = c["num_hidden_layers"]
        # Hub configs say num_experts; transformers 5 writes num_local_experts.
        experts = c.get("num_experts") or c.get("num_local_experts") or 0
        moe_layers = ()
        if arch == "Qwen3MoeForCausalLM" and experts:
            step, dense = c.get("decoder_sparse_step", 1), set(c.get("mlp_only_layers") or [])
            moe_layers = tuple(i for i in range(n_layers) if i not in dense and (i + 1) % step == 0)
        num_heads = c["num_attention_heads"]
        return cls(
            architecture=arch,
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_layers=n_layers,
            num_heads=num_heads,
            num_kv_heads=c.get("num_key_value_heads", num_heads),
            head_dim=c.get("head_dim") or c["hidden_size"] // num_heads,
            rms_norm_eps=c["rms_norm_eps"],
            rope_theta=float(rope_theta),
            max_position_embeddings=c["max_position_embeddings"],
            tie_word_embeddings=c.get("tie_word_embeddings", False),
            eos_token_ids=eos_ids,
            qk_norm=qk_norm,
            qkv_bias=qkv_bias,
            rope_scaling=tuple(sorted(scaling.items())) if rope_type == "llama3" else None,
            num_experts=experts,
            num_experts_per_tok=c.get("num_experts_per_tok", 0) or 0,
            moe_intermediate_size=c.get("moe_intermediate_size", 0) or 0,
            norm_topk_prob=c.get("norm_topk_prob", True),
            moe_layers=moe_layers,
            **cls._quant(c),
        )

    @staticmethod
    def _quant(c: dict) -> dict:
        q = c.get("quantization_config") or {}
        if not q:
            return {}
        if q.get("quant_method") == "mxfp4":
            # gpt-oss: experts MXFP4 (`*_blocks` / `*_scales`, one E8M0 exponent per 32), every
            # other module ("model.layers.*.self_attn", "model.layers.*.mlp.router",
            # "model.embed_tokens", "lm_head", modules_to_not_convert in
            # https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json) BF16.
            return dict(quant_expert_block=32, quant_expert_mxfp4=True, quant_blockwise=True)
        if q.get("quant_method") != "fp8" or q.get("fmt", "e4m3") != "e4m3":
            raise NotImplementedError(f"quantization {q.get('quant_method')}/{q.get('fmt')} is not supported")
        if not q.get("weight_block_size"):
            # Per-tensor FP8 (tencent/Hy3-FP8: quant_method fp8, no weight_block_size, a scalar
            # BF16 `weight_scale` per weight; https://huggingface.co/tencent/Hy3-FP8/raw/main/config.json
            # and its shard headers). Kiln keeps per-row scales in 128-column blocks, so that
            # rescaling onto trn1's e4m3 max 240 (fit_e4m3_max) touches only the blocks that need it.
            return dict(quant_block=128, quant_expert_block=128, quant_ignored=tuple(q.get("ignored_layers") or ()))
        bn, bk = q.get("weight_block_size")
        if bn is None or bn != bk:
            raise NotImplementedError("only square FP8 weight blocks are supported")
        expert = q.get("mxfp4_block_size", 32) if q.get("store_dtype") == "mxfp4" else bk
        # GLM-5.3 names the unquantized modules modules_to_not_convert
        # (https://huggingface.co/zai-org/GLM-5.3/raw/main/config.json), MiMo-V2 ignored_layers.
        # Vision-language checkpoints may spell them under model.language_model. (Qwen/Qwen3.8-
        # Flash-Next-FP8 config.json); Kiln names modules model.layers..., as the loader reads them.
        vl = "model.language_model."
        ignored = [m if not m.startswith(vl) else "model." + m[len(vl):]
                   for m in q.get("ignored_layers") or q.get("modules_to_not_convert") or ()]
        return dict(quant_block=int(bk), quant_expert_block=int(expert), quant_blockwise=True,
                    quant_expert_mxfp4=q.get("store_dtype") == "mxfp4", quant_ignored=tuple(ignored))

    @classmethod
    def _mimo_v2(cls, c: dict, eos_ids: tuple[int, ...]) -> "ModelConfig":
        """Xiaomi MiMo-V2 (modeling_mimo_v2.py in the model repo): hybrid full / sliding-window
        attention per hybrid_layer_pattern (1 = sliding), partial RoPE, value scale, attention
        sinks, and DeepSeek-V3-style sigmoid MoE routing with a correction bias."""
        if (c.get("n_group") or 1) != 1 or (c.get("topk_group") or 1) != 1:
            raise NotImplementedError("grouped expert routing (n_group > 1) is not supported yet")
        if c.get("scoring_func", "sigmoid") != "sigmoid" or c.get("topk_method", "noaux_tc") != "noaux_tc":
            raise NotImplementedError("only sigmoid / noaux_tc MiMo-V2 routing is supported")
        prf = float(c.get("partial_rotary_factor", (c.get("rope_parameters") or {}).get("partial_rotary_factor", 1.0)))
        heads, dk = c["num_attention_heads"], c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
        full = AttnSpec(heads, c["num_key_value_heads"], dk, c.get("v_head_dim", dk), int(dk * prf),
                        float(c.get("rope_theta", 10000.0)), None, bool(c.get("add_full_attention_sink_bias")))
        sdk = c.get("swa_head_dim", dk)
        swa = AttnSpec(c.get("swa_num_attention_heads", heads), c.get("swa_num_key_value_heads", c["num_key_value_heads"]),
                       sdk, c.get("swa_v_head_dim", c.get("v_head_dim", sdk)), int(sdk * prf),
                       float(c.get("swa_rope_theta", c.get("rope_theta", 10000.0))), c.get("sliding_window"),
                       bool(c.get("add_swa_attention_sink_bias")))
        n = c["num_hidden_layers"]
        pattern = c.get("hybrid_layer_pattern") or [0] * n
        specs = tuple(swa if pattern[i] == 1 else full for i in range(n))
        freq = c.get("moe_layer_freq") or [0] * n
        experts = c.get("n_routed_experts") or 0
        if c.get("n_shared_experts"):
            raise NotImplementedError("MiMo-V2 shared experts are not supported yet")
        return cls(
            architecture="MiMoV2ForCausalLM", vocab_size=c["vocab_size"], hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"], num_layers=n, num_heads=full.num_heads,
            num_kv_heads=full.num_kv_heads, head_dim=dk, rms_norm_eps=c.get("layernorm_epsilon", c.get("rms_norm_eps", 1e-6)),
            rope_theta=full.rope_theta, max_position_embeddings=c["max_position_embeddings"],
            tie_word_embeddings=c.get("tie_word_embeddings", False), eos_token_ids=eos_ids,
            qkv_bias=bool(c.get("attention_bias", False)), num_experts=experts,
            num_experts_per_tok=c.get("num_experts_per_tok", 0) or 0,
            moe_intermediate_size=c.get("moe_intermediate_size", 0) or 0,
            norm_topk_prob=c.get("norm_topk_prob", True),
            moe_layers=tuple(i for i in range(n) if experts and freq[i]),
            attn_layers=specs, v_scale=c.get("attention_value_scale"), router_scoring="sigmoid",
            router_bias=c.get("topk_method", "noaux_tc") == "noaux_tc",
            routed_scaling_factor=float(c.get("routed_scaling_factor") or 1.0),
            fused_qkv_weights=c.get("attention_projection_layout") == "fused_qkv",
            **cls._quant(c),
            mtp_layers=int(c.get("num_nextn_predict_layers") or 0), mtp_spec=swa
        )

    def kv_bytes_per_token(self, dtype: torch.dtype) -> int:
        """KV bytes per token for a cache stored in `dtype` (linear-attention layers hold none;
        an MLA layer holds its own K and V widths, models/mla.py)."""
        specs = self.attn_layers or ()
        if any(getattr(s, "mla", None) is not None for s in specs):
            return sum(s.num_kv_heads * (s.head_dim + s.v_head_dim) for s in specs
                       if not isinstance(s, LinearSpec)) * torch.finfo(dtype).bits // 8
        n = self.num_layers - sum(isinstance(s, LinearSpec) for s in specs)
        aux = sum(getattr(s, "aux_kv_width", 0) for s in specs)  # e.g. a QSA indexer key (models/qwen4_exp.py)
        return (2 * n * self.num_kv_heads * self.head_dim + aux) * torch.finfo(dtype).bits // 8


def _rope_theta(c: dict, default: float = 10000.0) -> float:
    return float((c.get("rope_parameters") or {}).get("rope_theta", c.get("rope_theta", default)))


def _gpt_oss(cls, c: dict, eos_ids: tuple[int, ...]) -> ModelConfig:
    """openai/gpt-oss-120b / -20b. Keys from https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json;
    semantics from transformers 5.15 models/gpt_oss/modeling_gpt_oss.py: layer_types alternates
    sliding_attention (window 128) and full_attention, every layer has per-head sinks, q/k/v/o
    carry biases (attention_bias), the router is a biased linear whose top-k logits are
    softmaxed, experts (num_local_experts, intermediate_size each) have gate_up and down biases
    and the clamped GLU (swiglu_limit), and RoPE is YaRN (rope_scaling)."""
    n, heads = c["num_hidden_layers"], c["num_attention_heads"]
    D = c.get("head_dim") or c["hidden_size"] // heads
    theta = _rope_theta(c)
    rs = dict(c.get("rope_scaling") or c.get("rope_parameters") or {})
    if rs.get("rope_type", rs.get("type")) != "yarn":
        raise NotImplementedError(f"gpt-oss with rope {rs.get('rope_type')!r} (expected yarn)")
    rs.pop("rope_theta", None)
    types = c.get("layer_types") or ["sliding_attention" if i % 2 == 0 else "full_attention" for i in range(n)]
    specs = tuple(AttnSpec(heads, c["num_key_value_heads"], D, D, D, theta,
                           c["sliding_window"] if t == "sliding_attention" else None, True) for t in types[:n])
    experts = c.get("num_local_experts") or c.get("num_experts")
    return cls(
        architecture="GptOssForCausalLM", vocab_size=c["vocab_size"], hidden_size=c["hidden_size"],
        intermediate_size=c["intermediate_size"], num_layers=n, num_heads=heads,
        num_kv_heads=c["num_key_value_heads"], head_dim=D, rms_norm_eps=c["rms_norm_eps"], rope_theta=theta,
        max_position_embeddings=c["max_position_embeddings"], tie_word_embeddings=c.get("tie_word_embeddings", False),
        eos_token_ids=eos_ids, qkv_bias=bool(c.get("attention_bias", True)), o_bias=bool(c.get("attention_bias", True)),
        rope_scaling=tuple(sorted({**rs, "rope_type": "yarn"}.items())), num_experts=experts,
        num_experts_per_tok=c.get("num_experts_per_tok") or c.get("experts_per_token"),
        moe_intermediate_size=c["intermediate_size"], moe_layers=tuple(range(n)), attn_layers=specs,
        router_scoring="topk_softmax", router_logit_bias=True, expert_bias=True, moe_act="swiglu_oai",
        swiglu_limit=float(c.get("swiglu_limit", 7.0)), **cls._quant(c))


def _hy_v3(cls, c: dict, eos_ids: tuple[int, ...]) -> ModelConfig:
    """tencent/Hy3 and Hy3-FP8. Keys from https://huggingface.co/tencent/Hy3/raw/main/config.json;
    semantics from transformers 5.15 models/hy_v3/modeling_hy_v3.py: GQA with per-head q/k RMSNorm
    before RoPE, the first layer(s) dense (mlp_layer_types, else first_k_dense_replace), the rest
    sigmoid-routed with a selection bias (expert_bias), weights renormalised (+1e-20) and scaled
    by router_scaling_factor, plus num_shared_experts x moe_intermediate_size of shared MLP. The
    MTP layer (num_nextn_predict_layers, stored as model.layers.{num_hidden_layers}) is not loaded."""
    if c.get("attention_bias") or c.get("mlp_bias"):
        raise NotImplementedError("Hy3 with attention_bias / mlp_bias")
    if not c.get("moe_router_use_sigmoid", True) or c.get("hidden_act", "silu") != "silu":
        raise NotImplementedError("Hy3 routing other than sigmoid, or an activation other than silu")
    n, heads = c["num_hidden_layers"], c["num_attention_heads"]
    kinds = c.get("mlp_layer_types") or ["dense"] * c.get("first_k_dense_replace", 1) + ["sparse"] * n
    experts = c.get("num_experts") or c.get("num_local_experts")
    return cls(
        architecture="HYV3ForCausalLM", vocab_size=c["vocab_size"], hidden_size=c["hidden_size"],
        intermediate_size=c["intermediate_size"], num_layers=n, num_heads=heads,
        num_kv_heads=c["num_key_value_heads"], head_dim=c.get("head_dim") or c["hidden_size"] // heads,
        rms_norm_eps=c["rms_norm_eps"], rope_theta=_rope_theta(c, 11158840.0),
        max_position_embeddings=c["max_position_embeddings"], tie_word_embeddings=c.get("tie_word_embeddings", False),
        eos_token_ids=eos_ids, qk_norm=bool(c.get("qk_norm", True)), num_experts=experts,
        num_experts_per_tok=c["num_experts_per_tok"], moe_intermediate_size=c["moe_intermediate_size"],
        norm_topk_prob=bool(c.get("route_norm", True)), moe_layers=tuple(i for i in range(n) if kinds[i] == "sparse"),
        router_scoring="sigmoid", router_bias=bool(c.get("moe_router_enable_expert_bias", True)),
        routed_scaling_factor=float(c.get("router_scaling_factor") or 1.0),
        n_shared_experts=c.get("num_shared_experts") or 0, **cls._quant(c))


def _k2_horizon(cls, c: dict, eos_ids: tuple[int, ...]) -> ModelConfig:
    """IFM/K2-Horizon-375B-A23B. Keys from https://huggingface.co/IFM/K2-Horizon-375B-A23B/raw/main/config.json;
    semantics from that repo's modeling_k2_horizon.py (Qwen3-MoE layout): mlp_only_layers dense,
    sigmoid routing whose gate bias (moe_gate_bias) only steers the top-k choice, renormalised and
    scaled by router_scaling_factor, num_shared_experts x moe_intermediate_size of shared MLP, and
    partial RoPE over rope_head_dim: the head is split into halves and RoPE rotates the first
    rope_head_dim / 2 dims of each half together (split_to_interleaved, then the first
    rope_head_dim interleaved dims), i.e. a full-width table whose upper frequencies are zero."""
    unsupported = {"attention_gate_func": c.get("attention_gate_func") is not None,
                   "mova_num_experts": bool(c.get("mova_num_experts")),
                   "layernorm_num_groups": (c.get("layernorm_num_groups") or 1) != 1,
                   "query_key_norm": bool(c.get("query_key_norm")), "attention_bias": bool(c.get("attention_bias")),
                   "use_sliding_window": bool(c.get("use_sliding_window"))}
    if any(unsupported.values()):
        raise NotImplementedError(f"K2-Horizon options not supported: {[k for k, v in unsupported.items() if v]}")
    if c.get("router_score_func", "softmax") != "sigmoid":
        raise NotImplementedError("K2-Horizon routing other than sigmoid")
    n, heads = c["num_hidden_layers"], c["num_attention_heads"]
    D = c.get("head_dim") or c["hidden_size"] // heads
    rd = c.get("rope_head_dim") or D
    theta = _rope_theta(c)
    spec = AttnSpec(heads, c["num_key_value_heads"], D, D, D, theta, None, False, rd // 2 if rd != D else None)
    step, dense = c.get("decoder_sparse_step", 1), set(c.get("mlp_only_layers") or [])
    experts = c.get("num_experts") or 0
    return cls(
        architecture="K2HorizonForCausalLM", vocab_size=c["vocab_size"], hidden_size=c["hidden_size"],
        intermediate_size=c["intermediate_size"], num_layers=n, num_heads=heads, num_kv_heads=c["num_key_value_heads"],
        head_dim=D, rms_norm_eps=c["rms_norm_eps"], rope_theta=theta,
        max_position_embeddings=c["max_position_embeddings"], tie_word_embeddings=c.get("tie_word_embeddings", False),
        eos_token_ids=eos_ids, num_experts=experts,
        num_experts_per_tok=c["num_experts_per_tok"], moe_intermediate_size=c["moe_intermediate_size"],
        norm_topk_prob=bool(c.get("norm_topk_prob", False)),
        moe_layers=tuple(i for i in range(n) if experts and i not in dense and (i + 1) % step == 0),
        attn_layers=(spec,) * n, router_scoring="sigmoid", router_bias=bool(c.get("moe_gate_bias")),
        routed_scaling_factor=float(c.get("router_scaling_factor") or 1.0),
        n_shared_experts=c.get("num_shared_experts") or 0, **cls._quant(c))


def _inkling(cls, c: dict, eos_ids: tuple[int, ...]) -> ModelConfig:
    """thinkingmachines/Inkling-Small and Inkling, text model only. Keys from text_config in
    https://huggingface.co/thinkingmachines/Inkling-Small/raw/main/config.json; semantics from
    transformers 5.15 models/inkling/modeling_inkling.py: no RoPE; per-head q / k RMSNorm with
    1 / head_dim score scaling; a relative position bias from r_proj(x) @ rel_logits_proj over
    key distances below rel_extent (sliding_window_size on local layers); log scaling of queries
    and that bias on global layers; causal depthwise short convolutions (kernel sconv_kernel_size,
    plus the input) on k and v before k_norm and on the attention and MLP outputs; layers
    local_layer_ids slide over sliding_window_size keys; the first dense_mlp_idx layers dense;
    sigmoid routing with a selection bias, where the top-k and the n_shared_experts shared
    experts' weights are sigmoid(logit) normalised over all of them, times route_scale and the
    gate's global_scale; embeddings RMS-normalised; logits of h / logits_mup_width_multiplier
    over the first unpadded_vocab_size rows. The MTP layers (mtp_config) are not loaded.

    The experts' intermediate size is the text config's intermediate_size when
    dense_intermediate_size is given: Inkling-Small's experts.w2_weight is [256, 4096, 2048]
    (shard headers), while transformers 5.15's InklingTextConfig keeps moe_intermediate_size at
    its default 3072 in that case (it matches the large Inkling's 3072 only by coincidence)."""
    t = c["text_config"]
    unsupported = {"q_bias": bool(t.get("q_bias")), "o_bias": bool(t.get("o_bias")),
                   "final_logit_softcapping": t.get("final_logit_softcapping") is not None,
                   "gate_activation": t.get("gate_activation", "sigmoid") != "sigmoid",
                   "norm_after_topk": not t.get("norm_after_topk", True)}
    if any(unsupported.values()):
        raise NotImplementedError(f"Inkling options not supported: {[k for k, v in unsupported.items() if v]}")
    n, H = t["num_hidden_layers"], t["hidden_size"]
    local = set(t.get("local_layer_ids") if t.get("local_layer_ids") is not None
                else [i for i in range(n) if (i + 1) % 6])
    log = None
    if t.get("log_scaling_n_floor"):
        log = (float(t["log_scaling_n_floor"]), float(t.get("log_scaling_alpha", 0.1)))
    w = t.get("sliding_window_size", 512)
    D, sD = t.get("head_dim", 128), t.get("swa_head_dim", t.get("head_dim", 128))
    glob = AttnSpec(t["num_attention_heads"], t["num_key_value_heads"], D, D, 0, 0.0, None, False, None, 1.0 / D,
                    t.get("rel_extent", 1024), log is not None)
    swa = AttnSpec(t.get("swa_num_attention_heads", t["num_attention_heads"]),
                   t.get("swa_num_key_value_heads", t["num_key_value_heads"]), sD, sD, 0, 0.0, w, False, None, 1.0 / sD,
                   w, False)
    dense_I = t.get("dense_intermediate_size")
    moe_I = t.get("moe_intermediate_size") or (t["intermediate_size"] if dense_I else 3072)
    dense_n = t.get("dense_mlp_idx", 0)
    shared = t.get("n_shared_experts") or 0
    experts = t.get("n_routed_experts") or t.get("num_local_experts")
    V = t.get("unpadded_vocab_size") or t["vocab_size"]
    eos = c.get("eos_token_id", t.get("eos_token_id"))
    return cls(
        architecture="InklingForConditionalGeneration", vocab_size=V, hidden_size=H,
        intermediate_size=dense_I or t["intermediate_size"], num_layers=n, num_heads=glob.num_heads,
        num_kv_heads=glob.num_kv_heads, head_dim=D, rms_norm_eps=t.get("rms_norm_eps", 1e-6), rope_theta=0.0,
        max_position_embeddings=t.get("model_max_length", t.get("max_position_embeddings", 131072)),
        tie_word_embeddings=False, eos_token_ids=eos_ids or ((eos,) if isinstance(eos, int) else tuple(eos or ())),
        qk_norm=True, num_experts=experts, num_experts_per_tok=t["num_experts_per_tok"], moe_intermediate_size=moe_I,
        moe_layers=tuple(range(dense_n, n)), attn_layers=tuple(swa if i in local else glob for i in range(n)),
        router_scoring="sigmoid_shared", router_bias=bool(t.get("use_gate_bias", True)),
        routed_scaling_factor=float(t.get("route_scale", 1.0)), n_shared_experts=shared,
        router_shared_rows=shared, router_global_scale=bool(t.get("use_global_scale", True)),
        dense_mlp_scale=bool(t.get("use_global_scale", True)), embed_norm=bool(t.get("use_embed_norm", False)),
        logits_divisor=float(t.get("logits_mup_width_multiplier") or 1.0),
        sconv_kernel=t.get("sconv_kernel_size", 4) if t.get("use_sconv", True) else 0, d_rel=t.get("d_rel", 16),
        log_scaling=log)


# GQA + MoE architectures beyond Qwen3-MoE and MiMo-V2, parsed by their own functions.
GQA_MOE = {"GptOssForCausalLM": _gpt_oss, "HYV3ForCausalLM": _hy_v3, "K2HorizonForCausalLM": _k2_horizon,
           "InklingForConditionalGeneration": _inkling}


def _geo_ladder(lo: int, hi: int, ratio: float = 1.5) -> tuple[int, ...]:
    """lo, ~lo*ratio, ... hi, as distinct ints. With ratio 1.5 a step reads at most 1.5x the
    KV its longest sequence needs (a power-of-two ladder allows 2x)."""
    out, b = [], float(lo)
    while round(b) < hi:
        out.append(int(round(b)))
        b *= ratio
    out.append(hi)
    return tuple(sorted(set(out)))


def _pow2_ladder(lo: int, hi: int) -> tuple[int, ...]:
    out, b = [], lo
    while b < hi:
        out.append(b)
        b *= 2
    out.append(hi)
    return tuple(sorted(set(out)))


@dataclass
class EngineConfig:
    model_path: str
    device: str = "cpu"  # "cpu" or "neuron"
    dtype: torch.dtype = torch.bfloat16
    # "auto" stores KV in the model dtype; "fp8" in float8 e4m3 (vLLM / SGLang
    # --kv-cache-dtype fp8), halving the KV bytes every decode step reads.
    kv_cache_dtype: str = "auto"
    # "auto": weights a quantized checkpoint stores as FP8 / MXFP4 stay FP8 on the device
    # (fp32 block scales, dequantized in-graph); "bf16": dequantize them at load. "fp8" /
    # "fp8-experts": quantize a BF16 checkpoint to FP8 at load (every linear but embeddings,
    # lm_head and routers / the routed experts only), per row in 128-column blocks.
    weight_dtype: str = "auto"
    page_size: int = 32
    num_pages: int | None = None  # derived from kv_cache_gb when None
    kv_cache_gb: float = 4.0
    max_num_seqs: int = 32
    max_model_len: int = 4096
    max_prefill_tokens: int = 512  # prefill tokens per step (chunk budget)
    sampling_candidates: int = 64  # top-k candidate set the on-device sampler draws from
    # Static-shape buckets. None derives power-of-two ladders from the limits above.
    decode_batch_buckets: tuple[int, ...] | None = None
    prefill_token_buckets: tuple[int, ...] | None = None
    page_buckets: tuple[int, ...] | None = None
    seed: int = 0
    schedule_policy: str = "fcfs"  # fcfs | lpm | spf | priority (see engine/scheduler.py)
    # Admission (engine/scheduler.py): "reserve" admits a request only when the free and evictable pages
    # cover its whole prompt and the pages the running requests still need (their prompts plus up to
    # admission_decode_tokens of decode each); "eager" admits whenever the first chunk fits.
    admission: str = os.environ.get("KILN_ADMISSION", "reserve")
    admission_decode_tokens: int = int(os.environ.get("KILN_ADMISSION_DECODE_TOKENS", 256))
    eviction_policy: str = "lru"  # lru | lfu | slru | priority | tlru (see engine/radix_cache.py)
    eviction_policy_config: dict | None = None  # SGLang --radix-eviction-policy-config
    # Host-memory KV tier under the radix cache, GB per rank (engine/hicache.py); 0 = off.
    hicache_host_gb: float = 0.0
    # Radix prefix cache (vLLM --enable-prefix-caching, SGLang --disable-radix-cache inverted).
    prefix_caching: bool = True
    # Linear-attention models resume a cached prefix from a state checkpoint (engine/state_pool.py,
    # engine/scheduler.py): rows of the state pool kept for them per DP group (None: 2 x
    # max_num_seqs), tokens between periodic prefill checkpoints (0: only at radix junctions, vLLM's
    # prefix_cache_retention_interval=0), tokens between decode checkpoints (SGLang
    # mamba_track_interval, default 256), all rounded up to a page multiple; and whether every
    # prefill also checkpoints its last page boundary (vLLM's replay boundary). That costs one more
    # prefill call per uncached prompt: Qwen3.5-0.8B, 1081 tokens, trn1, 167 -> 186 ms to first token
    # (tools/bench_linear_serving.py --what ttft, 2026-10-03), so it is off; junctions catch the same
    # sharing one request later.
    state_checkpoints: int | None = None
    state_checkpoint_interval: int = 0
    state_track_interval: int = 256
    state_checkpoint_prompt: bool = False
    # Also checkpoint the prefix a request shares with another queued or running request as soon as
    # the first of them computes it, and hold the second until it can resume from it (scheduler.py
    # "Junctions are also taken AHEAD"); False: a junction is checkpointed only by the second request.
    state_checkpoint_lookahead: bool = True
    # DP attention: prefill packing across groups (engine/dp.py "Prefill packing"), opt-in: "off" (the default),
    # "trim" (a group's extra chunk waits a step rather than add a call fewer than dp_prefill_pack_min groups fill:
    # GLM-5.3-Flash conc 64 with prompts of 1024-8192 tokens +14.3% out tok/s with TTFT p50 / p90 lower, 75% shared
    # warm +16.4%, uniform prompts unchanged) or "hold" (also a sparse lone call waits up to dp_prefill_hold_steps
    # steps per request while decodes run: +37.9% where requests finish out of step, TTFT p50 +34%); min None: every
    # group. No graph changes. Not the default because a deferred chunk runs on its request's natural chunk grid and
    # moves device tokens by chunking rounding (mean |dlogprob| 3x the rerun floor, no signed-bias number yet):
    # docs/neuron-notes.md "Packing prefill chunks across DP-attention groups" names the check that would promote it.
    dp_prefill_pack: str = field(default_factory=lambda: os.environ.get("KILN_DP_PREFILL_PACK", "off"))
    dp_prefill_pack_min: int | None = None
    dp_prefill_hold_steps: int = 1
    max_num_queued_reqs: int | None = None
    max_num_queued_tokens: int | None = None
    # s3://... prefix shared by every instance on the same SDK: compiled graphs are pulled
    # before start-up and pushed after warmup (kiln/compile_cache.py).
    compile_cache_uri: str | None = None
    tp: int = 1  # tensor-parallel ranks, one process and one NeuronCore each
    # Tensor parallelism of the token mixers (attention, MLA, Gated DeltaNet, KDA), a divisor of
    # tp: their heads split attention_tp ways, replicated over tp / attention_tp groups of
    # consecutive ranks, while the MLP and experts keep tp (models/decoder.py DecoderForCausalLM).
    # None: the largest divisor of tp every mixer's head counts allow (tp itself when they divide it).
    attention_tp: int | None = None
    # DP attention (SGLang --enable-dp-attention with --dp-size, v0.5.21 srt/layers/dp_attention.py;
    # vLLM data_parallel_size with expert parallelism): the token mixers run data-parallel over
    # dp_attention groups of tp / dp_attention consecutive ranks, each group serving DIFFERENT
    # requests with its own KV pages and recurrent-state rows, while the MLP / experts, the
    # embedding and the lm_head keep tp over every group's tokens (engine/dp.py, models/decoder.py
    # DecoderForCausalLM: DP attention). The attention TP is then tp / dp_attention. max_num_seqs
    # and max_prefill_tokens stay the engine's totals: each group runs up to ceil(max_num_seqs /
    # dp_attention) requests and max_prefill_tokens // dp_attention prefill tokens per step, so the
    # MLP / experts see the prefill batch they see without DP attention (SGLang divides
    # chunked_prefill_size by dp_size the same way: v0.5.21 srt/arg_groups/parallel_hook.py
    # _handle_data_parallelism). 1: plain (or replicated attention-TP) execution.
    dp_attention: int = 1
    # Shard the embedding and lm_head over tensor-parallel ranks by vocabulary rows (vLLM
    # VocabParallelEmbedding / ParallelLMHead): MiMo-V2.6-Flash replicated them at 2.5 GB per
    # rank of 16 GB. Costs one all-reduce (lookup) and one all-gather (logits) per step.
    vocab_parallel: bool = True
    # Keep MXFP4 experts 4-bit on the device (half of FP8's HBM, much slower decode; see
    # models/quant.py) instead of converting them losslessly to FP8.
    mxfp4_packed: bool = False
    tp_core_base: int = 0  # first NeuronCore of this engine (data-parallel replicas offset it)
    # Overlap scheduling (SGLang's zero-overhead scheduler): schedule and launch step N+1
    # before reading step N's tokens, which stay on the device.
    overlap: bool = False
    # Compile one graph per kind of layer plus small prep / post graphs, instead of one graph
    # per whole model (see DecoderForCausalLM.layer_fn): compile time is one layer's per kind,
    # at the cost of one graph launch per layer.
    piecewise: bool = False
    # Layers per piecewise graph (None: about 14 launches). MoE layers always get a graph each
    # (model_runner.piecewise_groups: two of them in one graph made neuronx-cc split the expert
    # gathers into tiny software-generated DMAs; MiMo-V2.6-Flash decode 550 -> 55 ms per step),
    # unless the NKI MoE kernel is on and piecewise_moe_group is set.
    piecewise_group: int | None = None
    # Selected-expert MoE: "xla" (the gather in DecoderForCausalLM._moe), "nki" (= "nki-dedupe",
    # kiln/kernels/moe_dedupe.py: FP8 MXFP4 experts repacked at load into one blob per expert with
    # one power-of-two scale per 128-column tile, each distinct expert of a call one DMA) or
    # "nki-pair" (kiln/kernels/moe_decode.py: block-32 scales, each (token, expert) pair one
    # whole-expert DMA); decoder.NKI_MOE_KERNELS. Off by default.
    moe_kernel: str = field(default_factory=lambda: os.environ.get("KILN_MOE_KERNEL", "xla"))
    # With an NKI moe_kernel: layers per piecewise graph, MoE layers included (None: one graph per
    # MoE layer, as with the XLA path).
    piecewise_moe_group: int | None = field(
        default_factory=lambda: int(os.environ["KILN_PIECEWISE_MOE_GROUP"]) if os.environ.get("KILN_PIECEWISE_MOE_GROUP") else None)
    # The same for prefill chunks (default: piecewise_moe_group); see model_runner._piecewise for
    # why a big chunk wants smaller groups than decode (neuronx-cc's 5M-instruction limit).
    piecewise_prefill_moe_group: int | None = field(
        default_factory=lambda: int(os.environ["KILN_PIECEWISE_PREFILL_MOE_GROUP"])
        if os.environ.get("KILN_PIECEWISE_PREFILL_MOE_GROUP") else None)
    # KILN_DECODE_WHOLE=1 (off by default): under piecewise, a decode call is ONE graph (the prep, every layer
    # and the post: forward_decode compiled whole) while prefill, verify and mixed calls keep their pieces.
    # Each graph execution holding a cross-chip collective has a fixed cost at 32 ranks (docs/neuron-notes.md
    # "Collectives across chips"), and a decode call is otherwise prep + ceil(layers / piecewise_moe_group)
    # groups + post graphs.
    decode_whole: bool = field(default_factory=lambda: os.environ.get("KILN_DECODE_WHOLE", "0") == "1")
    # Mixed batches (KILN_MIXED_BATCH=1, off by default): the decode tokens of running requests ride in the
    # prefill calls of the same step, as vLLM's chunked prefill does (vllm 0.24.0 vllm/v1/core/sched/
    # scheduler.py Scheduler.schedule: "There's no 'decoding phase' nor 'prefill phase' in the scheduler",
    # running requests' tokens first, then waiting ones, from one token_budget, all in one forward).
    # Every prefill call becomes one graph over each DP-attention group's
    # chunk rows followed by up to mixed_decode_rows decoding sequences (DecoderForCausalLM.forward_mixed;
    # None: the largest decode bucket), so a step with prefill work makes no decode call for them, and the
    # prefill-only graphs are not warmed. Decode-only steps run the decode graphs as before.
    mixed_batch: bool = field(default_factory=lambda: os.environ.get("KILN_MIXED_BATCH", "0") == "1")
    mixed_decode_rows: int | None = field(
        default_factory=lambda: int(os.environ["KILN_MIXED_DECODE_ROWS"]) if os.environ.get("KILN_MIXED_DECODE_ROWS")
        else None)
    # vLLM --watermark-config: {"algorithm": "gumbel", "key": int, "context_width": 4,
    # "deduplicate_contexts": "single_turn" | "none"} (engine/watermark.py).
    watermark: dict | None = None
    # Thinking budget boundaries (vLLM --reasoning-config): counting starts after
    # reasoning_start_str; at a request's thinking_token_budget, reasoning_end_str is forced.
    reasoning_start_str: str | None = None
    reasoning_end_str: str | None = None
    # Jump-forward decoding for structured outputs (SGLang v0.4.3.post2, engine._jump_forward):
    # when a request's grammar allows one continuation string, its tokens are appended at once
    # and computed as a prefill chunk. Off by default: SGLang declared it on but switched it off
    # whenever its overlap scheduler (the default) ran, and removed it in v0.4.4; and it emits
    # the tokenizer's split of a forced string where the model's own may differ, so outputs can
    # differ from token-by-token decoding (tests/test_grammar.py).
    jump_forward: bool = False
    # Speculative decoding: None, "ngram" (vLLM ngram / SGLang NGRAM) or "suffix" (vLLM).
    spec_method: str | None = None
    spec_k: int = 4
    # vLLM num_speculative_tokens_per_batch_size (v0.30.0 config/speculative.py): (lo, hi, k) ranges of
    # the running batch size, inclusive, each with the draft length used there (at most spec_k; 0 = no
    # drafts, a plain decode step). The verify graph stays 1 + spec_k wide.
    spec_k_per_batch_size: tuple | None = None
    # Sequences that have no draft in a speculative step (their last token, a draft dropped under KV
    # pressure, k = 0 by spec_k_per_batch_size): True runs them in the step's verify graph beside the drafted
    # ones (a row with no draft, padded like a short draft), so a step makes one verify launch per bucket and
    # the plain decode graphs are neither loaded nor launched; False gives them a decode launch of their own,
    # which with pinned buckets costs a whole decode step for one row. None: True for "mtp", where almost every
    # sequence drafts every step; GLM-5.3-Flash tp=32 at conc 64 also does not fit trn1's HBM with the decode
    # graphs beside the verify graphs (docs/neuron-notes.md "MTP at serving scale").
    spec_verify_plain: bool | None = None
    # Speculative steps under overlap scheduling (engine/spec_async.py): the accepted count, the newest token, the
    # next positions and the MTP drafts stay on the device and the next step is scheduled before this one is read.
    # MTP with spec_k 1 only. None: KILN_SPEC_ASYNC (default 0).
    spec_async: bool | None = None
    spec_ngram_min: int = 2
    spec_ngram_max: int = 4
    # "suffix" (vLLM suffix decoding): see engine/spec_suffix.py.
    suffix_max_tree_depth: int = 24
    suffix_max_cached_requests: int = 1000
    suffix_max_spec_factor: float = 1.0
    suffix_min_token_prob: float = 0.1
    num_layers: int | None = None  # debugging: run only the first n layers of the checkpoint
    # Prefill / decode disaggregation (engine/disagg.py): None (one engine does both), "prefill" (prefill
    # graphs only; every request is handed off at its first token) or "decode" (decode graphs only; takes
    # handed-off requests). A decode engine listens for handoffs on pd_listen ("host:port", port 0: any),
    # announces pd_advertise (this box's address as prefill boxes reach it; default the listen host, or the
    # host's own address for 0.0.0.0), and holds at most pd_buffer_gb of received parts on the host
    # (/dev/shm) before a part waits. pd_bypass_prefill: the decode engine also loads its prefill buckets
    # and serves whole requests (short prompts the router does not disaggregate, server/router.py).
    pd_role: str | None = None
    pd_listen: str | None = None
    pd_advertise: str | None = None
    pd_buffer_gb: float = 16.0
    pd_bypass_prefill: bool = False
    # One long prefill over several engines (engine/pp.py, opt-in): stage pp_stage of pp_stages runs the layers of its
    # range only (pp_split: the first layer of every stage after the first; None: by measured time,
    # pp.default_split); a stage after the first listens on pp_listen ("host:port", rank r on port + r) for the
    # previous stage's hidden stream, a stage before the last sends to pp_next. Prefill only, piecewise only.
    pp_stage: int = 0
    pp_stages: int = 1
    pp_split: tuple | None = None
    pp_listen: str | None = None
    pp_next: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def spec_overlap(self) -> bool:
        """Whether speculative steps run under overlap scheduling (spec_async, engine/spec_async.py)."""
        import os

        v = self.spec_async if self.spec_async is not None else os.environ.get("KILN_SPEC_ASYNC", "0") == "1"
        return bool(v) and self.spec_method == "mtp"

    @property
    def spec_merged(self) -> bool:
        """Whether draftless sequences of a speculative step run in its verify graph (spec_verify_plain)."""
        if not self.spec_method:
            return False
        return self.spec_method == "mtp" if self.spec_verify_plain is None else self.spec_verify_plain

    @property
    def max_pages_per_seq(self) -> int:
        return -(-self.max_model_len // self.page_size)

    @property
    def group_max_num_seqs(self) -> int:
        """Running requests per DP-attention group (all of them without DP attention)."""
        return -(-self.max_num_seqs // self.dp_attention)

    @property
    def group_max_prefill_tokens(self) -> int:
        """Prefill tokens per DP-attention group and step (all of them without DP attention)."""
        if self.max_prefill_tokens < self.dp_attention:
            raise ValueError(f"max_prefill_tokens={self.max_prefill_tokens} is below one token per "
                             f"DP-attention group (dp_attention={self.dp_attention})")
        return self.max_prefill_tokens // self.dp_attention

    def resolved_decode_batch_buckets(self) -> tuple[int, ...]:
        return self.decode_batch_buckets or _pow2_ladder(1, self.group_max_num_seqs)

    def resolved_prefill_token_buckets(self) -> tuple[int, ...]:
        n = self.group_max_prefill_tokens
        return self.prefill_token_buckets or _pow2_ladder(min(32, n), n)

    def resolved_page_buckets(self) -> tuple[int, ...]:
        return self.page_buckets or _geo_ladder(min(4, self.max_pages_per_seq), self.max_pages_per_seq)


def pick_bucket(n: int, ladder: tuple[int, ...]) -> int:
    for b in ladder:
        if b >= n:
            return b
    raise ValueError(f"{n} exceeds the largest bucket {ladder[-1]}")
