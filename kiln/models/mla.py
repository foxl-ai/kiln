"""Multi-head latent attention (MLA) and DeepSeek Sparse Attention (DSA) as a per-layer
attention kind of DecoderForCausalLM.

MLA (DeepSeek-V2/V3, Kimi K2, GLM-5, Youtu-LLM): per layer

    q      = q_b(rmsnorm(q_a(x)))            (or q_proj(x) without q_lora_rank)  [T, H, dn + dr]
    c, kr  = rmsnorm(kv_a(x)[:r]), kv_a(x)[r:]                                  latent [T, r], rope key [T, dr]
    k_h    = [W_UK[h] c | rope(kr)],  v_h = W_UV[h] c       (kv_b_proj rows of head h: dn rows of W_UK, dv of W_UV)

with RoPE on the last dr dims of q and on the single shared rope key. The paged cache holds
ONLY c and rope(kr) per token: kv_lora_rank + qk_rope_head_dim values (512 + 64 = 576 for
DeepSeek-V3), against 2 * H * head_dim for MHA. In the cache's terms an MLA layer has one
"KV head" whose K is the latent (k_cache [slots, 1, r]) and whose V is the rope key (v_cache
[slots, 1, dr], plus the DSA indexer key, below). Decode and speculative verify attend over
the latent directly by weight absorption (q_lat = W_UK^T q_nope; out = W_UV (p @ c)), so
the per-head keys and values are never formed; a prefill chunk decompresses its context
(KILN_MLA_PREFILL=expand, the default) or absorbs as well (=absorb).

Tensor parallelism is SGLang v0.5.21's and vLLM v0.30.0's layout (srt/models/deepseek_v2.py
DeepseekV2AttentionMLA; vllm/model_executor/models/deepseek_v2.py DeepseekV2MLAAttention):
q_a_proj and kv_a_proj_with_mqa are ReplicatedLinear, q_b_proj / q_proj and kv_b_proj are
ColumnParallelLinear over heads, o_proj is RowParallelLinear; so every rank computes the same
latent and holds a full copy of the latent cache, and only the heads are split, over the
attention TP (DecoderForCausalLM: a divisor of tp, replicated across its groups). (SGLang's DP
attention, where each rank owns all heads for its own requests, is not implemented.)

DSA (DeepSeek-V3.2 `deepseek_v32`, GLM-5.x `glm_moe_dsa`): a lightning indexer scores every
cached token for each query, index[t, l] = sum_h w[t, h] relu(q_I[t, h] . k_I[l] / sqrt(Di)),
with q_I = wq_b(q latent), k_I = LayerNorm(wk(x)) (both with RoPE on their first dr dims) and
w = weights_proj(x) / sqrt(Hi); the query then attends only to its top-k (index_topk, 2048)
tokens. The indexer key is cached per token beside the rope key (v_cache [slots, 1, dr + Di]).
Numerics follow transformers 5.15 (models/deepseek_v32 and models/glm_moe_dsa, modeling_*.py):
scores in fp32 and no Hadamard rotation or FP8 activation quantisation of q_I / k_I, which
DeepSeek's own inference/model.py and vLLM v0.30's Indexer add as precision optimisations
(transformers' DeepseekV32Indexer docstring: the Hadamard transform is orthogonal). The
indexer is replicated on every tensor-parallel rank, as vLLM's Indexer is ("no tensor
parallel, just replicated": wq_b ReplicatedLinear, wk_weights_proj disable_tp=True).

A query sees at most `L` tokens of its bucket's block table. When L <= index_topk (static per
bucket) DSA selects every visible token, exactly as transformers' topk(min(index_topk, T))
does, so the layer runs as dense MLA and the indexer only writes its keys. Above it, the
top-k is applied as an attention mask (KILN_DSA=mask, default) or by gathering the selected
latents per query (=gather). The selection is found without torch.topk (models/dsa_select.py,
KILN_DSA_SELECT, default "nki": kernels/dsa_topk.py's exact radix search over the threshold's bit
pattern in one NKI kernel, lowest index first among ties at the threshold): the mask path takes it
as a mask directly, the gather path turns it into indices. GLM-5.3's IndexShare: layers whose indexer_types entry is "shared" have no indexer and
reuse the selection of the previous "full" layer (GlmMoeDsaAttention.skip_topk); it crosses layer
graphs through one scratch tensor that full layers write and shared layers read, like a KV cache:
the additive mask itself (layer.dsa_mask [rows, keys], bf16 0 / -1e30) for the mask path, so a
shared layer reads a slice instead of rebuilding the mask from indices with a scatter, and the
indices (layer.dsa_topk [rows, index_topk]) for the gather path.

Interleaved RoPE (rope_interleave, DeepSeek's checkpoint layout: pairs (x0, x1), (x2, x3) ...
rotated by one frequency; transformers apply_rotary_pos_emb_interleave) is folded into the
weights at load time: the rows producing each rope block are permuted to [evens, odds], after
which the usual rotate-half RoPE produces exactly transformers' output, layout included.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..config import AttnSpec, ModelConfig
from . import dsa_long as _dsa_long
from . import dsa_select
from .decoder import NEG_INF, rms_norm, rotate_half
from .quant import dequant

# How a prefill chunk attends: "expand" decompresses the gathered latents into per-head keys
# and values (SGLang's MHA path for prefill), "absorb" uses the decode formulation.
PREFILL = os.environ.get("KILN_MLA_PREFILL", "expand")
# How DSA applies its top-k when the context exceeds it: "mask" (an additive mask over the
# whole context) or "gather" (attend over the selected latents only).
DSA_MODE = os.environ.get("KILN_DSA", "mask")
# A layer that runs its indexer writes its selection to the scratch and attends with what it reads
# back (attention below; KILN_DSA_STAGE=0: with the selection itself).
STAGE = os.environ.get("KILN_DSA_STAGE", "1") == "1"
# A pooled indexer (GLM-5.3-Flash) keeps every pool's key in the KV cache, written when the pool's
# tokens are written, read by the selection, instead of rebuilding all L / kpool pool keys from their
# tokens' keys and gates in every call. KILN_DSA_POOL_CACHE (pool_cache_mode):
#   inplace: the key replaces the indexer key in the cached row of the pool's LAST token (no extra
#            bytes; write_pool_keys_inplace);
#   separate: kpool pieces in a bf16 token-slot state beside K and V (pool_key_width; 704 bytes per
#            token on every rank for GLM-5.3-Flash, i.e. 4% fewer KV pages in bf16, 8% in FP8);
#   off (or 0): rebuilt in every call;
#   auto (default): inplace for a bf16 / fp32 KV cache, separate for an FP8 one (an FP8 row would
#            round the key to e4m3, so inplace is out there). Measured 2026-10-04 on trn1.32xlarge,
#            GLM-5.3-Flash conc 64, prefill 4096, KV 1.5 GB fp8 (KV fits: 5500 pages vs 4224 needed):
#            separate 91.2 out tok/s against off 88.3 on the same box. When the KV pool caps the
#            requests in flight (G64-2048 at KV 1.0 fp8) off was ~1% ahead (65.1 vs 64.0), so size
#            the KV pool for every sequence; docs/neuron-notes.md "The pool-key cache's bytes".
# 1 is the 2026-10-04 spelling of separate.
POOL_CACHE = {"1": "separate", "0": "off"}.get(os.environ.get("KILN_DSA_POOL_CACHE", "auto"),
                                               os.environ.get("KILN_DSA_POOL_CACHE", "auto"))
if POOL_CACHE not in ("auto", "inplace", "separate", "off"):
    raise ValueError(f"KILN_DSA_POOL_CACHE must be auto, inplace, separate or off, not {POOL_CACHE!r}")
# The paged cache of a pooled DSA layer (KILN_DSA_KV, models/dsa_long.py "Minimal KV"): "full" (default) keeps every
# token's indexer key and gate logits beside its latent (V = [key | gates], 2 Di wide) plus the pool-key cache of
# KILN_DSA_POOL_CACHE; "minimal" keeps the latent and the pool-key pieces only (V = Di / kpool wide, in the KV
# dtype: GLM-5.3-Flash under FP8 KV 512 + 32 bytes per token per DSA layer, 5,984 per token instead of 9,152), the
# indexer rows of each request's open pool living in a per-request state row (layer.open_pool).
# KILN_DSA_QSHARD=1 (opt-in): a long-context prefill chunk's queries are split over the attention group and each rank
# attends ITS rows with EVERY head (models/mla.py attention_long), the head-split attention otherwise. The slot
# attention costs ~40-48 us per row at 8 heads against ~51-85 us at 64 (kernels/dsa_slots.py, trn1, 2026-10-05), so
# per (query, head) the whole-head form is 4-5x cheaper. Each rank then also holds q_b, kv_b (W_UK, W_UV) and o_proj
# for all heads (GLM-5.3-Flash: ~109 M parameters per DSA layer, ~1.2 GB per rank in FP8).
QSHARD = os.environ.get("KILN_DSA_QSHARD", "0") == "1"
# Context-parallel slot classes (attention_cp, kernels/dsa_slots_n.py): a rank attends each row over a buffer sized by
# the row's real local share, 128 slots for the rows that fit (keep / A + the tail is ~65 at A = 8) and the full 640
# for the rest, each buffer's kernel call running its live rows only. Exact for any split; opt-in (new graphs).
CP_SLOT_CLASSES = os.environ.get("KILN_DSA_CP_SLOT_CLASSES", "0") == "1"
CP_SLOTS_SMALL = int(os.environ.get("KILN_DSA_CP_SLOTS_SMALL", "128"))
# Context-parallel decode batches whose local pools fit keep (opt-in, attention_cp's decode branch only; prefill chunks
# trace as before). KILN_DSA_CP_ALL_LOCAL=1: when a rank holds at most keep local pools (any context up to keep A kpool
# tokens: 16,384 at A = 8, so every 8K row), every visible one is a candidate and the visible ones are a prefix (local
# pool m < nloc), so the local list is the pools in order (_cp_local_all), with no selection, compact() or per-element
# gathers. The same list, values and slot rows as the selection path, exactly. At 64 rows per group on trn1 that path
# was 2.15 ms of a 7.05 ms DSA layer, ~1.6 ms of it engines idle on [1, 1]-slice gathers (CP-64 profile, 2026-10-06).
# KILN_DSA_CP_PAGE_KEYS=1: the local pool keys read as whole page rows (the cache viewed as [pages, ppl Di], the
# block table as the index: 2 KiB descriptors at GLM-5.3-Flash's ppl 8, Di 128, against 256 B rows), the same values.
CP_ALL_LOCAL = os.environ.get("KILN_DSA_CP_ALL_LOCAL", "0") == "1"
CP_PAGE_KEYS = os.environ.get("KILN_DSA_CP_PAGE_KEYS", "0") == "1"
KV_LAYOUT = os.environ.get("KILN_DSA_KV", "full")
if KV_LAYOUT not in ("full", "minimal"):
    raise ValueError(f"KILN_DSA_KV must be full or minimal, not {KV_LAYOUT!r}")


def pool_cache_mode(kv_fp8: bool) -> str:
    """The pool-key cache form for a KV cache stored in FP8 (kv_fp8) or not (POOL_CACHE)."""
    if POOL_CACHE == "auto":
        return "separate" if kv_fp8 else "inplace"
    return POOL_CACHE


@dataclass(frozen=True)
class DSASpec:
    """One layer's DeepSeek Sparse Attention indexer."""

    n_heads: int  # index_n_heads
    head_dim: int  # index_head_dim
    topk: int  # index_topk
    rope_interleave: bool  # glm_moe_dsa: indexer_rope_interleave (true); deepseek_v32: half-split
    indexer: bool = True  # False: GLM IndexShare "shared" layer, reuses the previous full layer's top-k
    # GLM-5.3-Flash (glm5_next): the indexer scores pools of `kpool` consecutive keys and selects
    # topk / kpool of them (plus, with kpool_tail, the current incomplete pool); models/glm5_next.py.
    kpool: int = 1
    kpool_tail: bool = False


@dataclass(frozen=True)
class MLASpec:
    q_lora_rank: int | None
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    softmax_scale: float
    rope_interleave: bool
    rope: bool = True  # False: NoPE (Kimi Linear mla_use_nope): the rope dims exist but are not rotated
    # q_a_layernorm / kv_a_layernorm: transformers builds them with RMSNorm's default eps 1e-6
    # whatever rms_norm_eps says (models/deepseek_v3/modeling_deepseek_v3.py DeepseekV3Attention).
    norm_eps: float = 1e-6
    dsa: DSASpec | None = None


def _pooled(spec) -> bool:
    m = getattr(spec, "mla", None)
    d = m.dsa if m is not None else None
    return d is not None and d.indexer and d.kpool > 1


def pool_key_width(spec, kv_fp8: bool = False) -> int:
    """Per-token width of the separate pool-key cache of an attention layer (0: none, also in the
    inplace and off forms, pool_cache_mode). A pooled indexer's
    pool key (head_dim values) is stored as kpool pieces of head_dim / kpool, piece i in the slot of
    the pool's i-th token, so the cache is paged with the tokens (prefix sharing, the host tier and
    the page gathers need nothing else) and a page gather of the context reads [L, head_dim /
    kpool] = [L / kpool, head_dim], the pool keys in order. Kept in the model dtype even under an
    FP8 KV cache (a token-slot state, DecoderForCausalLM.token_state_shapes): the pool keys are
    computed in bf16 from the cached rows exactly as the per-call form computes them."""
    if pool_cache_mode(kv_fp8) != "separate" or not _pooled(spec) or minimal_layout(spec.mla):
        return 0
    d = spec.mla.dsa
    if d.head_dim % d.kpool:
        raise ValueError(f"index_head_dim {d.head_dim} is not divisible by index_kpool {d.kpool}")
    return d.head_dim // d.kpool


def cache_widths(m: MLASpec) -> tuple[int, int]:
    """(K, V) widths of the paged cache: the latent, and the rope key plus the indexer key (and,
    for a pooled indexer, each token's pool-compression gate logits beside it)."""
    if minimal_layout(m):  # V holds the pool-key pieces (KV_LAYOUT "minimal")
        return m.kv_lora_rank, m.dsa.head_dim // m.dsa.kpool
    v = m.qk_rope_head_dim + (m.dsa.head_dim * (2 if m.dsa.kpool > 1 else 1) if m.dsa is not None and m.dsa.indexer
                              else 0)
    return m.kv_lora_rank, max(v, 1)  # a NoPE layer without an indexer keeps a 1-wide dummy V


def minimal_layout(m: MLASpec) -> bool:
    """Whether an MLA layer's cache is the minimal layout (KV_LAYOUT): a pooled indexer without RoPE, tail selected."""
    d = m.dsa
    return (KV_LAYOUT == "minimal" and d is not None and d.indexer and d.kpool > 1 and not m.qk_rope_head_dim
            and d.kpool_tail and d.head_dim % d.kpool == 0)


# -- configuration ------------------------------------------------------------------

# HF architecture -> family. Kimi K2 checkpoints are DeepseekV3ForCausalLM (model_type kimi_k2,
# https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json); Kimi K2.5 / K2.7
# wrap that config in KimiK25ForConditionalGeneration.text_config
# (https://huggingface.co/moonshotai/Kimi-K2.7-Code/raw/main/config.json).
ARCHITECTURES = {
    "DeepseekV3ForCausalLM": "deepseek_v3",
    "DeepseekV32ForCausalLM": "deepseek_v32",
    "GlmMoeDsaForCausalLM": "glm_moe_dsa",
    "YoutuForCausalLM": "youtu",
    "KimiK25ForConditionalGeneration": "deepseek_v3",
}


def rope_params(c: dict) -> dict:
    """One RoPE dict from either spelling: transformers 5's rope_parameters, or the older
    rope_theta + rope_scaling (DeepSeek-V3's config.json: rope_scaling {"type": "yarn", ...})."""
    rp = dict(c.get("rope_parameters") or {})
    rp.update(c.get("rope_scaling") or {})
    rp.setdefault("rope_theta", c.get("rope_theta", 10000.0))
    rp["rope_type"] = rp.get("rope_type", rp.get("type", "default"))
    rp.pop("type", None)
    return rp


def yarn_mscale(scale: float, mscale: float = 1.0) -> float:
    """transformers modeling_rope_utils._compute_yarn_parameters.get_mscale."""
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def yarn_inv_freq(dim: int, base: float, rs: dict, max_positions: int) -> tuple[torch.Tensor, float]:
    """YaRN inverse frequencies and the cos / sin scale (attention_factor), as
    transformers 5.15 modeling_rope_utils._compute_yarn_parameters computes them."""
    orig = rs["original_max_position_embeddings"]
    factor = rs.get("factor") or max_positions / orig
    af = rs.get("attention_factor")
    if af is None:
        ms, msa = rs.get("mscale"), rs.get("mscale_all_dim")
        af = yarn_mscale(factor, ms) / yarn_mscale(factor, msa) if ms and msa else yarn_mscale(factor)
    beta_fast, beta_slow = rs.get("beta_fast") or 32, rs.get("beta_slow") or 1

    def corr_dim(rot):
        return (dim * math.log(orig / (rot * 2 * math.pi))) / (2 * math.log(base))

    low, high = corr_dim(beta_fast), corr_dim(beta_slow)
    if rs.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    pos_freqs = base ** (torch.arange(0, dim, 2).to(torch.float) / dim)
    extrapolation = 1 - ramp
    inv_freq = (1.0 / (factor * pos_freqs)) * (1 - extrapolation) + (1.0 / pos_freqs) * extrapolation
    return inv_freq, float(af)


def _indexer_types(c: dict, n: int) -> list[str]:
    """transformers GlmMoeDsaConfig.__post_init__: explicit indexer_types, else
    index_topk_pattern ("F" / "S"), else "full" every index_topk_freq layers from
    index_skip_topk_offset (GLM-5.3: freq 4, offset 3 -> F F F S S S F S S S ...)."""
    if c.get("indexer_types"):
        return list(c["indexer_types"])
    pattern = c.get("index_topk_pattern")
    if pattern is not None:
        return [{"F": "full", "S": "shared"}[x] for x in pattern] if isinstance(pattern, str) else list(pattern)
    freq, offset = max(c.get("index_topk_freq", 1), 1), c.get("index_skip_topk_offset", 2)
    return ["full" if (max(i - offset + 1, 0) % freq) == 0 else "shared" for i in range(n)]


def model_config(cls, c: dict, eos_ids: tuple[int, ...]) -> ModelConfig:
    """ModelConfig for an MLA checkpoint. Keys: transformers 5.15 DeepseekV3Config,
    DeepseekV32Config, GlmMoeDsaConfig, YoutuConfig (models/*/configuration_*.py)."""
    if c["architectures"][0] == "KimiK25ForConditionalGeneration":
        c = c["text_config"]
    arch = c["architectures"][0]
    family = ARCHITECTURES[arch]
    rp = rope_params(c)
    if rp["rope_type"] not in ("default", "yarn"):
        raise NotImplementedError(f"rope scaling {rp['rope_type']!r} is not supported for MLA")
    n = c["num_hidden_layers"]
    heads = c["num_attention_heads"]
    dn, dr, dv = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
    scale = (dn + dr) ** -0.5
    if rp["rope_type"] != "default" and rp.get("mscale_all_dim"):  # transformers yarn_apply_mscale
        scale *= yarn_mscale(rp["factor"], rp["mscale_all_dim"]) ** 2
    # transformers applies interleaved RoPE unconditionally in DeepseekV32Attention and
    # GlmMoeDsaAttention, and per rope_interleave (default True) in DeepseekV3Attention.
    interleave = bool(c.get("rope_interleave", True)) if family in ("deepseek_v3", "youtu") else True
    dsa = None
    if family in ("deepseek_v32", "glm_moe_dsa"):
        dsa = DSASpec(c["index_n_heads"], c["index_head_dim"], c["index_topk"],
                      rope_interleave=family == "glm_moe_dsa" and bool(c.get("indexer_rope_interleave", True)))
    types = _indexer_types(c, n) if family == "glm_moe_dsa" else ["full"] * n
    if types and types[0] != "full" and dsa is not None:
        raise ValueError("the first DSA layer must run its own indexer (indexer_types[0] == 'full')")
    specs = []
    for i in range(n + 1):  # the last one: the MTP layer's, which runs its own indexer (models/mtp.py)
        d = dsa if dsa is None else DSASpec(dsa.n_heads, dsa.head_dim, dsa.topk, dsa.rope_interleave,
                                             indexer=i == n or types[i] == "full")
        m = MLASpec(c.get("q_lora_rank"), c["kv_lora_rank"], dn, dr, dv, scale, interleave, dsa=d)
        kw, vw = cache_widths(m)
        specs.append(AttnSpec(heads, 1, kw, vw, dr, float(rp["rope_theta"]), mla=m))
    specs, mtp_spec = specs[:n], specs[n]
    experts = c.get("n_routed_experts") or 0
    if family == "youtu":
        experts = 0
    if experts and c.get("scoring_func", "sigmoid") != "sigmoid":
        raise NotImplementedError(f"MLA MoE routing {c.get('scoring_func')!r} is not supported (sigmoid only)")
    if family == "glm_moe_dsa" and c.get("mlp_layer_types"):
        moe_layers = tuple(i for i in range(n) if c["mlp_layer_types"][i] == "sparse")
    else:  # transformers DeepseekV3DecoderLayer: MoE from first_k_dense_replace on
        moe_layers = tuple(range(c.get("first_k_dense_replace", 0), n)) if experts else ()
    return cls(
        architecture=arch, vocab_size=c["vocab_size"], hidden_size=c["hidden_size"],
        intermediate_size=c["intermediate_size"], num_layers=n, num_heads=heads, num_kv_heads=1,
        head_dim=specs[0].head_dim, rms_norm_eps=c["rms_norm_eps"], rope_theta=float(rp["rope_theta"]),
        max_position_embeddings=c["max_position_embeddings"], tie_word_embeddings=c.get("tie_word_embeddings", False),
        eos_token_ids=eos_ids, rope_scaling=tuple(sorted(rp.items())) if rp["rope_type"] == "yarn" else None,
        num_experts=experts, num_experts_per_tok=c.get("num_experts_per_tok", 0) or 0,
        moe_intermediate_size=c.get("moe_intermediate_size", 0) or 0, norm_topk_prob=c.get("norm_topk_prob", True),
        moe_layers=moe_layers, attn_layers=tuple(specs), router_scoring="sigmoid",
        router_bias=c.get("topk_method", "noaux_tc") == "noaux_tc" and bool(experts),
        routed_scaling_factor=float(c.get("routed_scaling_factor") or 1.0),
        n_shared_experts=(c.get("n_shared_experts") or 0) if experts else 0,
        n_group=c.get("n_group") or 1, topk_group=c.get("topk_group") or 1,
        **cls._quant(c), **mtp_fields(c, n, mtp_spec, bool(experts)),
    )


def mtp_fields(c: dict, n: int, spec: AttnSpec, moe: bool) -> dict:
    """ModelConfig's MTP fields for a DeepSeek-V3-style checkpoint: num_nextn_predict_layers
    layers stored as model.layers.<n>... (zai-org/GLM-5.3, deepseek-ai/DeepSeek-V3 and V3.2
    model.safetensors.index.json, read 2026-10-03: enorm, hnorm, eh_proj, shared_head.norm and a
    whole decoder layer, its MLP the MoE and, for DSA, its own indexer). Kiln applies the first one
    recursively, as vLLM v0.30.0 (deepseek_mtp.py: spec_step_idx % num_mtp_layers) and SGLang
    v0.5.21 (deepseek_nextn.py, one decoder) do."""
    k = int(c.get("num_nextn_predict_layers") or 0)
    if not k:
        return {}
    return dict(mtp_layers=k, mtp_spec=spec, mtp_prefix=f"model.layers.{n}", mtp_moe=moe,
                mtp_index_share=bool(c.get("index_share_for_mtp_iteration", False)))


# -- MoE routing (DeepSeek-V3 node-limited top-k) -------------------------------------


def group_limited(choice: torch.Tensor, n_group: int, topk_group: int) -> torch.Tensor:
    """Experts outside the topk_group groups with the largest sum of their two best scores are
    excluded (transformers DeepseekV3TopkRouter; DeepSeek-V3: 8 groups, 4 kept). The kept groups
    are found by comparing indices, not by scatter."""
    T, E = choice.shape
    per = E // n_group
    g = choice.view(T, n_group, per)
    # The sum of each group's two best scores, as max + the max of the rest (argmax drops one
    # copy of the max, so a tie counts twice, as topk(2) would): g.topk(2, dim=-1) itself came
    # back wrong on trn1 (2 experts per group, max error 1.54; SDK 2.32, 2026-10-03,
    # tools/probe_dsa.py --routing).
    first = g.argmax(dim=-1, keepdim=True)
    rest = torch.where(torch.arange(per, device=choice.device) == first, NEG_INF, g)
    best = g.amax(dim=-1) + rest.amax(dim=-1)  # [T, n_group]
    _, gi = torch.topk(best, topk_group, dim=-1)
    keep = (gi.unsqueeze(-1) == torch.arange(n_group, device=choice.device)).any(dim=1)  # [T, n_group]
    return torch.where(keep.unsqueeze(-1), g, NEG_INF).reshape(T, E)


# -- parameters ---------------------------------------------------------------------


def init_layer(layer, cfg: ModelConfig, spec: AttnSpec, p, lin, tp: int) -> None:
    """MLA attention parameters of one attention rank of `tp` (the attention TP,
    DecoderForCausalLM; DecoderLayer.__init__)."""
    m = spec.mla
    H, nh = cfg.hidden_size, spec.num_heads // tp
    dn, dr, dv, r = m.qk_nope_head_dim, m.qk_rope_head_dim, m.v_head_dim, m.kv_lora_rank
    layer.nh, layer.nkv, layer.kv_offset = nh, 1, 0
    if m.q_lora_rank:
        # q_a_proj and kv_a_proj_with_mqa read the same input and are replicated: one matmul
        # (vLLM's fused_qkv_a_proj).
        lin("w_a", "self_attn.q_a_proj", m.q_lora_rank + r + dr, H)
        layer.q_a_norm = p(m.q_lora_rank)
        lin("q_b", "self_attn.q_b_proj", nh * (dn + dr), m.q_lora_rank)
    else:
        lin("w_a", "self_attn.kv_a_proj_with_mqa", r + dr, H)
        layer.q_a_norm = None
        lin("q_b", "self_attn.q_proj", nh * (dn + dr), H)
    layer.kv_a_norm = p(r)
    lin("w_uk", "self_attn.kv_b_proj", nh * dn, r)  # W_UK of this rank's heads, [nh * dn, r]
    lin("w_uv", "self_attn.kv_b_proj", nh * dv, r)  # W_UV, [nh * dv, r]
    lin("o", "self_attn.o_proj", H, nh * dv)
    if QSHARD and tp > 1 and long_capable(spec) and m.q_lora_rank is not None:
        # every head (KILN_DSA_QSHARD): a long-context chunk's own rows attend with all of them (only set here, so
        # the default layers' static attributes are unchanged)
        layer.qshard = True
        Hh = spec.num_heads
        lin("q_b_all", "self_attn.q_b_proj", Hh * (dn + dr), m.q_lora_rank)
        lin("w_uk_all", "self_attn.kv_b_proj", Hh * dn, r)
        lin("w_uv_all", "self_attn.kv_b_proj", Hh * dv, r)
        lin("o_all", "self_attn.o_proj", H, Hh * dv)
    d = m.dsa
    if d is not None and d.indexer:
        # wq_b split by rows into the rope part of every head and the rest (see _indexer:
        # rotating the first dims of each head in place broke neuronx-cc).
        if dr:
            lin("idx_wq_r", "self_attn.indexer.wq_b", d.n_heads * dr, m.q_lora_rank)
        else:
            layer.idx_wq_r = layer.idx_wq_r_scale = None
        lin("idx_wq", "self_attn.indexer.wq_b", d.n_heads * (d.head_dim - dr), m.q_lora_rank)
        lin("idx_wk", "self_attn.indexer.wk", d.head_dim, H)
        layer.idx_knorm_w, layer.idx_knorm_b = p(d.head_dim), p(d.head_dim)
        # transformers keeps weights_proj in fp32 (GlmMoeDsaPreTrainedModel._keep_in_fp32_modules)
        layer.idx_wproj = nn.Parameter(torch.empty(d.n_heads, H, dtype=torch.float32), requires_grad=False)
        if d.kpool > 1:  # Glm5NextTextIndexer.index_kpool_compress_ape / _gate
            layer.idx_pool_ape, layer.idx_pool_gate = p(d.kpool, d.head_dim), p(d.head_dim, H)
        else:
            layer.idx_pool_ape = layer.idx_pool_gate = None
    else:
        for name in ("idx_wq_r", "idx_wq_r_scale", "idx_wq", "idx_wq_scale", "idx_wk", "idx_wk_scale", "idx_knorm_w",
                     "idx_knorm_b", "idx_wproj", "idx_pool_ape", "idx_pool_gate"):
            setattr(layer, name, None)
    layer.dsa_topk = layer.dsa_mask = None  # selection scratch shared by the DSA layers (bind_scratch)
    layer.pool_key = None  # [slots, head_dim / kpool] pool-key pieces, bound by the runner (pool_key_width)
    layer.pool_inplace = False  # pool keys in the pool's last cached row (bind_scratch, pool_cache_mode)
    layer.dsa_share = False  # this layer's selection is written for IndexShare layers after it
    layer.dsa_stage = False  # ... and read back for its own attention (bind_scratch)
    layer.qkv_bias = layer.q_norm = layer.k_norm = layer.sink = None


def bind_scratch(model, rows: int, device, keys: int | None = None) -> None:
    """The selection scratch of the DSA layers (see the module docstring): rows query rows (the
    most one call carries) by keys (the longest context a bucket reads). A layer that runs its
    indexer writes its selection there and attends with what it reads back (layer.dsa_stage,
    KILN_DSA_STAGE); IndexShare "shared" layers read it (layer.dsa_share: someone reads it)."""
    layers = [l for l in model.kv_layers() if l.spec.mla is not None and l.spec.mla.dsa is not None]
    if not layers:
        return
    share = any(not l.spec.mla.dsa.indexer for l in layers)
    if keys and all(long_capable(l.spec) for l in layers):
        # Buckets past dsa_long.LONG_KEYS run the long path, which stages nothing (attention_long).
        keys = min(keys, max(_dsa_long.LONG_KEYS, 1))
    # int32 with explicit casts on both sides: an int64 topk index written into an int64
    # buffer failed in LNL's tracer with "Check failed: self.scalar_type() ==
    # values.scalar_type()" (trn1, SDK 2.32, 2026-10-03).
    top = torch.zeros(max(rows, 1), layers[0].spec.mla.dsa.topk, dtype=torch.int32, device=device)
    mask = torch.zeros(max(rows, 2), keys, dtype=torch.bfloat16, device=device) if keys and (share or STAGE) else None
    mtp = getattr(model, "mtp", None)
    inplace = pool_cache_mode(getattr(model, "fp8_max", None) is not None) == "inplace"
    for l in layers:
        l.pool_inplace = inplace and _pooled(l.spec) and not minimal_layout(l.spec.mla)
        l.dsa_topk, l.dsa_mask = top, mask
        # Nothing reads the MTP layer's selection, and its graphs carry more rows than the scratch
        # (B x Q with Q up to a prefill bucket): it selects in the graph.
        l.dsa_share = share and l is not mtp
        l.dsa_stage = STAGE and mask is not None and l is not mtp


def rope_perm(dr: int) -> torch.Tensor:
    """Rows of an interleaved rope block in the order rotate-half RoPE expects: evens, odds."""
    return torch.cat([torch.arange(0, dr, 2), torch.arange(1, dr, 2)])


def _permute_rope_rows(w, s, block: int, first: int, dr: int, n_blocks: int = 1):
    """Permute rows [first, first + dr) of every `block` rows (n_blocks blocks) to rope_perm."""
    idx = torch.arange(w.shape[0])
    pr = rope_perm(dr)
    for b in range(n_blocks):
        o = b * block + first
        idx[o : o + dr] = o + pr
    return w[idx], (s[idx] if s is not None else None)


def load_layer(layer, ck, pre: str, cfg: ModelConfig, rank: int, dtype, assign, concat, put) -> None:
    """Fill an MLA layer's attention parameters from the checkpoint (loader._load_decoder), as
    attention rank `rank` (DecoderForCausalLM.attn_rank). assign(param, scale_param, w, s) and
    concat(parts) are the loader's _assign / _concat."""
    m = layer.spec.mla
    a = pre + "self_attn."
    blk = cfg.quant_block
    nh, dn, dr, dv, r = layer.nh, m.qk_nope_head_dim, m.qk_rope_head_dim, m.v_head_dim, m.kv_lora_rank
    il = m.rope_interleave and dr > 0
    parts = [ck.linear(a + "q_a_proj", block=blk)] if m.q_lora_rank else []
    w, s = ck.linear(a + "kv_a_proj_with_mqa", block=blk)
    if il:
        w, s = _permute_rope_rows(w, s, w.shape[0], r, dr)
    parts.append((w, s))
    assign(layer.w_a, layer.w_a_scale, *concat(parts))
    if m.q_lora_rank:
        put(layer.q_a_norm, a + "q_a_layernorm.weight")
    put(layer.kv_a_norm, a + "kv_a_layernorm.weight")
    qd = dn + dr
    w, s = ck.linear(a + ("q_b_proj" if m.q_lora_rank else "q_proj"), slice(rank * nh * qd, (rank + 1) * nh * qd),
                     block=blk)
    if il:
        w, s = _permute_rope_rows(w, s, qd, dn, dr, nh)
    assign(layer.q_b, layer.q_b_scale, w, s)
    kd = dn + dv
    w, s = ck.linear(a + "kv_b_proj", slice(rank * nh * kd, (rank + 1) * nh * kd), block=blk)
    w3 = w.view(nh, kd, -1)
    s3 = s.view(nh, kd, -1) if s is not None else None
    assign(layer.w_uk, layer.w_uk_scale, w3[:, :dn].reshape(nh * dn, -1),
           s3[:, :dn].reshape(nh * dn, -1) if s3 is not None else None)
    assign(layer.w_uv, layer.w_uv_scale, w3[:, dn:].reshape(nh * dv, -1),
           s3[:, dn:].reshape(nh * dv, -1) if s3 is not None else None)
    w, s = ck.linear(a + "o_proj", cols=slice(rank * nh * dv, (rank + 1) * nh * dv), block=blk)
    assign(layer.o, layer.o_scale, w, s)
    if getattr(layer, "qshard", False):  # every head's copies (KILN_DSA_QSHARD); NoPE, so no rope permutation
        Hh = layer.spec.num_heads
        w, s = ck.linear(a + "q_b_proj", block=blk)
        assign(layer.q_b_all, layer.q_b_all_scale, w, s)
        w, s = ck.linear(a + "kv_b_proj", block=blk)
        w3 = w.view(Hh, kd, -1)
        s3 = s.view(Hh, kd, -1) if s is not None else None
        assign(layer.w_uk_all, layer.w_uk_all_scale, w3[:, :dn].reshape(Hh * dn, -1),
               s3[:, :dn].reshape(Hh * dn, -1) if s3 is not None else None)
        assign(layer.w_uv_all, layer.w_uv_all_scale, w3[:, dn:].reshape(Hh * dv, -1),
               s3[:, dn:].reshape(Hh * dv, -1) if s3 is not None else None)
        w, s = ck.linear(a + "o_proj", block=blk)
        assign(layer.o_all, layer.o_all_scale, w, s)
    d = m.dsa
    if d is None or not d.indexer:
        return
    # Tensor names: zai-org/GLM-5.3 model.safetensors.index.json (self_attn.indexer.wq_b, .wk,
    # .k_norm.weight / .bias, .weights_proj), the same as transformers' DeepseekV32Indexer.
    ii = d.rope_interleave and dr > 0
    w, s = ck.linear(a + "indexer.wq_b", block=blk)
    if ii:
        w, s = _permute_rope_rows(w, s, d.head_dim, 0, dr, d.n_heads)
    heads = torch.arange(d.n_heads).unsqueeze(1) * d.head_dim
    rope_rows = (heads + torch.arange(dr)).reshape(-1)  # every head's first dr rows, then the rest
    pass_rows = (heads + torch.arange(dr, d.head_dim)).reshape(-1)
    if dr:
        assign(layer.idx_wq_r, layer.idx_wq_r_scale, w[rope_rows], s[rope_rows] if s is not None else None)
    assign(layer.idx_wq, layer.idx_wq_scale, w[pass_rows], s[pass_rows] if s is not None else None)
    w, s = ck.linear(a + "indexer.wk", block=blk)
    kn_w, kn_b = ck.get(a + "indexer.k_norm.weight"), ck.get(a + "indexer.k_norm.bias")
    if ii:
        w, s = _permute_rope_rows(w, s, d.head_dim, 0, dr)
        kn_w, _ = _permute_rope_rows(kn_w, None, d.head_dim, 0, dr)
        kn_b, _ = _permute_rope_rows(kn_b, None, d.head_dim, 0, dr)
    assign(layer.idx_wk, layer.idx_wk_scale, w, s)
    layer.idx_knorm_w.data.copy_(kn_w.to(dtype))
    layer.idx_knorm_b.data.copy_(kn_b.to(dtype))
    w, s = ck.linear(a + "indexer.weights_proj", block=blk)
    layer.idx_wproj.data.copy_(w.float() if s is None else dequant(w, s, torch.float32))
    if d.kpool > 1:  # zai-org/GLM-5.3-Flash model.safetensors.index.json names
        layer.idx_pool_ape.data.copy_(ck.get(a + "indexer.index_kpool_compress_ape").to(dtype))
        layer.idx_pool_gate.data.copy_(ck.get(a + "indexer.index_kpool_compress_gate").to(dtype))


# -- forward ------------------------------------------------------------------------


def _rope(model, layer, x, positions, dims: int, head_axis: bool):
    """Rotate-half RoPE on the first `dims` entries of x's last axis (all of it when equal)."""
    cos = getattr(model, f"rope_cos{layer.rope}")[positions]
    sin = getattr(model, f"rope_sin{layer.rope}")[positions]
    if head_axis:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    if x.shape[-1] == dims:
        return x * cos + rotate_half(x) * sin
    xr = x[..., :dims]
    return torch.cat([xr * cos + rotate_half(xr) * sin, x[..., dims:]], dim=-1)


def _project(model, layer, x, positions):
    """q_nope [T, nh, dn], rotated q_pe [T, nh, dr], latent c [T, r], rotated k_pe [T, dr] and
    the normalised q latent (the indexer's input; None without q_lora_rank)."""
    m = layer.spec.mla
    T = x.shape[0]
    dn, dr, r = m.qk_nope_head_dim, m.qk_rope_head_dim, m.kv_lora_rank
    a = F.linear(x, model._w(layer, "w_a"))
    # Explicit slices, not torch.split (LNL miscompiles an unequal split; decoder._qkv).
    if m.q_lora_rank:
        ql = m.q_lora_rank
        q_resid = rms_norm(a[:, :ql], layer.q_a_norm, m.norm_eps)
        kv = a[:, ql:]
        q = F.linear(q_resid, model._w(layer, "q_b"))
    else:
        q_resid, kv = None, a
        q = F.linear(x, model._w(layer, "q_b"))
    q = q.view(T, layer.nh, dn + dr)
    q_nope, q_pe = q[..., :dn], q[..., dn:]
    c = rms_norm(kv[:, :r], layer.kv_a_norm, m.norm_eps)
    k_pe = kv[:, r:]
    if m.rope and dr:
        q_pe = _rope(model, layer, q_pe, positions, dr, True)
        k_pe = _rope(model, layer, k_pe, positions, dr, False)
    return q_nope, q_pe, c, k_pe, q_resid


def _indexer(model, layer, x, q_resid, positions):
    """The lightning indexer's per-token query (rope part [T, Hi, dr], rest [T, Hi, Di - dr]),
    key [T, Di] (rope dims first, as transformers lays it out) and head weights [T, Hi]
    (transformers DeepseekV32Indexer.forward / GlmMoeDsaIndexer.forward, before scoring). A pooled
    indexer's key row also carries the token's pool gate logits: [T, 2 Di]
    (Glm5NextTextIndexer.forward: gate_scores = index_kpool_compress_gate(x)).

    The query's two parts come from two matmuls over row blocks of wq_b and are never
    concatenated: rotating the first dr dims of each head in place (slice, RoPE, cat) made
    neuronx-cc 2.27 fail with "BIR verification failed: Pattern accesses 64 (> 32) partitions
    starting at partition 32" on GLM-5.3-0.6B's decode layer (trn1, SDK 2.32, 2026-10-03;
    tools/probe_dsa.py --layer)."""
    d, dr = layer.spec.mla.dsa, layer.spec.mla.qk_rope_head_dim
    T = x.shape[0]
    q_p = F.linear(q_resid, model._w(layer, "idx_wq")).view(T, d.n_heads, d.head_dim - dr)
    q_r = None
    k = F.linear(x, model._w(layer, "idx_wk"))
    k = F.layer_norm(k.float(), (d.head_dim,), layer.idx_knorm_w.float(), layer.idx_knorm_b.float(), 1e-6).to(x.dtype)
    if dr:
        q_r = F.linear(q_resid, model._w(layer, "idx_wq_r")).view(T, d.n_heads, dr)
        q_r = _rope(model, layer, q_r, positions, dr, True)
        k = _rope(model, layer, k, positions, dr, False)
    w = F.linear(x.float(), layer.idx_wproj) * d.n_heads ** -0.5
    if d.kpool > 1:
        k = torch.cat([k, F.linear(x, layer.idx_pool_gate)], dim=-1)
    return (q_r, q_p), k, w


def _scores(d: DSASpec, q, w, kI, vis):
    """Index scores [B, Q, L] over L cached tokens, invisible ones at about NEG_INF.
    q: (rope part [B, Q, Hi, dr] or None, rest [B, Q, Hi, Di - dr]); w [B, Q, Hi] fp32;
    kI [B, L, Di] (rope dims first); vis [B, Q, L] (0 or NEG_INF)."""
    q_r, q_p = q
    dr = 0 if q_r is None else q_r.shape[-1]
    s = torch.einsum("bqhd,bld->bqhl", q_p.float(), kI[..., dr:].float())
    if q_r is not None:
        s = s + torch.einsum("bqhd,bld->bqhl", q_r.float(), kI[..., :dr].float())
    s = torch.relu(s * d.head_dim ** -0.5)
    return torch.einsum("bqh,bqhl->bql", w, s) + vis


def _topk(d: DSASpec, q, w, kI, vis):
    """Top-k token indices [B, Q, k] per query (torch.topk; tools)."""
    return torch.topk(_scores(d, q, w, kI, vis), d.topk, dim=-1).indices


def _select(d: DSASpec, q, w, kI, vis):
    """The selection in the form DSA_MODE uses: the additive mask [B, Q, L] (0 selected,
    NEG_INF not; "mask") or the selected token indices [B, Q, k] ("gather"). index_topk of the
    L tokens (dsa_select.topk_mask), the invisible ones scoring about NEG_INF so they are taken
    only when fewer are visible, exactly as torch.topk(min(index_topk, T)) would."""
    index = _scores(d, q, w, kI, vis)
    if dsa_select.SELECT == "topk" and DSA_MODE == "gather":
        return torch.topk(index, d.topk, dim=-1).indices
    sel = dsa_select.topk_mask(index, d.topk)
    if DSA_MODE != "gather":
        return torch.where(sel, 0.0, NEG_INF)
    return mask_indices(sel, d.topk)


def mask_indices(sel: torch.Tensor, k: int) -> torch.Tensor:
    """Positions [B, Q, k] of the k True entries of sel [B, Q, L] (exactly k per row), in order:
    each selected position is scattered to its rank, the rest to a dump column k."""
    L = sel.shape[-1]
    rank = torch.where(sel, sel.float().cumsum(-1).long() - 1, k)
    pos = torch.arange(L, device=sel.device).expand(sel.shape)
    return torch.zeros(*sel.shape[:-1], k + 1, dtype=torch.int64, device=sel.device).scatter(-1, rank, pos)[..., :k]


def _layer_selection(layer, d: DSASpec, index, vis, B: int, Q: int):
    """The selection of a layer that runs its indexer, in a sparse bucket: index (q, w, kI[, pk])
    as _core takes it (pk: a pooled indexer's pool keys from the cache). A pooled indexer's
    (glm5_next) is the additive mask whatever DSA_MODE says."""
    (q_r, q_p), wi, kI, *pk = index
    q4 = (None if q_r is None else q_r.reshape(B, Q, d.n_heads, -1), q_p.reshape(B, Q, d.n_heads, -1))
    if d.kpool > 1:
        from .glm5_next import pooled_selection

        return pooled_selection(layer, d, q4, wi.view(B, Q, d.n_heads), kI, vis, pk[0] if pk else None)
    return _select(d, q4, wi.view(B, Q, d.n_heads), kI, vis)


def _core(model, layer, q_nope, q_pe, kc, kpe, vis, B: int, Q: int, top=None, index=None, absorb=True, pos=None):
    """Attention of B x Q queries over L latents each: kc [B, L, r], kpe [B, L, dr], vis [B, Q, L]
    additive visibility. index: (q, w, kI) for a DSA layer that runs its indexer; top: a selection
    made elsewhere (the previous full layer's for a shared one, an earlier MTP pass's), in _select's
    form (a pooled indexer's: always the additive mask), which the layer then uses instead of its
    own. Returns (o_proj input [B * Q, nh * dv], the selection used or None)."""
    m = layer.spec.mla
    d = m.dsa
    nh, dn, dr, dv, r = layer.nh, m.qk_nope_head_dim, m.qk_rope_head_dim, m.v_head_dim, m.kv_lora_rank
    L = kc.shape[1]
    sparse = d is not None and L > d.topk  # static per bucket: below it DSA selects every visible token
    pooled = sparse and d.kpool > 1
    if sparse and index is not None and top is None:
        top = _layer_selection(layer, d, index, vis, B, Q)
    if top is not None and (pooled or not sparse or DSA_MODE != "gather"):
        # An additive mask (a dense bucket only gets one from an earlier MTP pass): part of the
        # visibility, and the attention below runs over every key.
        vis, sparse = vis + top, False
    w_uk = model._w(layer, "w_uk").view(nh, dn, r)
    w_uv = model._w(layer, "w_uv").view(nh, dv, r)
    qn = q_nope.reshape(B, Q, nh, dn)
    qp = q_pe.reshape(B, Q, nh, dr)
    if pooled and top is not None and pos is not None and prefill_kernel_takes(B, Q, L, r, dr):
        # KILN_DSA_PREFILL_KERNEL=nki: a pooled DSA layer's prefill chunk (one sequence; pos is given for the chunk
        # form only) as one flash-attention kernel over the latent, the selection and the causal visibility as its
        # mask (kernels/dsa_prefill.py).
        from ..kernels import dsa_prefill

        q_lat = torch.einsum("qhd,hdr->qhr", qn[0], w_uk)
        ol = dsa_prefill.attend(q_lat, kc[0], vis[0], m.softmax_scale, pos).to(model.dtype)
        return torch.einsum("qhr,hvr->qhv", ol, w_uv).reshape(B * Q, nh * dv), top
    if sparse and DSA_MODE == "gather":
        # Each query attends over its own k selected latents.
        k = d.topk
        flat = (top + (torch.arange(B, device=top.device) * L).view(B, 1, 1)).reshape(-1)
        kcs = kc.reshape(B * L, r)[flat].view(B, Q, k, r)
        vis_s = torch.gather(vis, -1, top)
        q_lat = torch.einsum("bqhd,hdr->bqhr", qn, w_uk)
        s = torch.einsum("bqhr,bqkr->bhqk", q_lat, kcs)
        if dr:
            s = s + torch.einsum("bqhd,bqkd->bhqk", qp, kpe.reshape(B * L, dr)[flat].view(B, Q, k, dr))
        p = torch.softmax(s.float() * m.softmax_scale + vis_s.unsqueeze(1), dim=-1).to(model.dtype)
        o = torch.einsum("bqhr,hvr->bqhv", torch.einsum("bhqk,bqkr->bqhr", p, kcs), w_uv)
        return o.reshape(B * Q, nh * dv), top
    if not absorb:  # decompress the context into per-head keys and values
        k_nope = torch.einsum("blr,hdr->blhd", kc, w_uk)
        v = torch.einsum("blr,hvr->blhv", kc, w_uv)
        s = torch.einsum("bqhd,blhd->bhql", qn, k_nope)
        if dr:
            s = s + torch.einsum("bqhd,bld->bhql", qp, kpe)
        p = torch.softmax(s.float() * m.softmax_scale + vis.unsqueeze(1), dim=-1).to(model.dtype)
        o = torch.einsum("bhql,blhv->bqhv", p, v)
    else:  # weight absorption: attend over the latent itself
        q_lat = torch.einsum("bqhd,hdr->bqhr", qn, w_uk)
        s = torch.einsum("bqhr,blr->bhql", q_lat, kc)
        if dr:
            s = s + torch.einsum("bqhd,bld->bhql", qp, kpe)
        p = torch.softmax(s.float() * m.softmax_scale + vis.unsqueeze(1), dim=-1).to(model.dtype)
        o = torch.einsum("bqhr,hvr->bqhv", torch.einsum("bhql,blr->bqhr", p, kc), w_uv)
    return o.reshape(B * Q, nh * dv), top


def attention(model, layer, x, positions, slot_mapping, table, bias, top=None, want_top: bool = False, mixed=None,
              state_slot=None):
    """The attention block of an MLA layer over the paged cache, for every batch form of
    DecoderForCausalLM._layer (decode, prefill chunk, extend); x is already normalised.
    Returns the o_proj output before the tensor-parallel all-reduce (and, with want_top, the DSA
    selection the layer used, None when its bucket is dense). top: a selection to use instead of
    the layer's own (_core), e.g. the first MTP pass's for the later ones
    (index_share_for_mtp_iteration); the layer still writes its keys.

    mixed (a mixed batch: a prefill chunk's rows, then D decode rows, DecoderForCausalLM._layer): the
    chunk form on the first rows, then the decode form on the rest, each exactly as its own call
    would run it (latent and indexer writes, pool keys, selection through the scratch, attention). The
    two touch different requests' pages: a decode row never reads the chunk's tokens or the other
    way round, so their order inside the graph does not matter."""
    if mixed is not None:
        from .decoder import MIXED_DECODE_SPLIT, MIXED_MIXERS, mixed_chunk_rows

        if want_top or top is not None:
            raise NotImplementedError("a mixed batch with a given DSA selection (MTP)")
        if minimal_layout(layer.spec.mla):
            raise NotImplementedError("a mixed batch with the minimal DSA cache (KILN_DSA_KV=minimal)")
        C = mixed_chunk_rows(x, mixed)
        if MIXED_MIXERS == "joint":
            return attention_joint(model, layer, x, positions, slot_mapping, table, bias, mixed[0], mixed[1], C)
        if MIXED_DECODE_SPLIT:
            from .decoder import mixed_decode_slices

            return torch.cat([attention(model, layer, x[:C], positions[:C], slot_mapping[:C], table, bias)]
                             + [attention(model, layer, x[C + a : C + b], positions[C + a : C + b],
                                          slot_mapping[C + a : C + b], mixed[0][a:b], mixed[1][a:b])
                                for a, b in mixed_decode_slices(mixed)])
        return torch.cat([attention(model, layer, x[:C], positions[:C], slot_mapping[:C], table, bias),
                          attention(model, layer, x[C:], positions[C:], slot_mapping[C:], mixed[0], mixed[1])])
    if minimal_layout(layer.spec.mla):
        if getattr(model, "cp", 1) > 1 and long_capable(layer.spec):  # context parallel over the minimal layout
            if want_top or top is not None or bias.dim() == 5:
                raise NotImplementedError("context-parallel DSA (models/dsa_long.py): MTP selections and verify batches")
            if state_slot is None or getattr(layer, "open_pool", None) is None:
                raise RuntimeError("the minimal DSA cache needs the request's state row (layer.open_pool, state_slot)")
            return attention_cp(model, layer, x, positions, slot_mapping, table, state_slot=state_slot)
        return attention_minimal(model, layer, x, positions, slot_mapping, table, bias, top, want_top, state_slot)
    if getattr(model, "cp", 1) > 1 and long_capable(layer.spec):
        if want_top or top is not None or bias.dim() == 5:
            raise NotImplementedError("context-parallel DSA (models/dsa_long.py): MTP selections and verify batches")
        return attention_cp(model, layer, x, positions, slot_mapping, table)
    m = layer.spec.mla
    d = m.dsa
    dr = m.qk_rope_head_dim
    T = x.shape[0]
    q_nope, q_pe, c, k_pe, q_resid = _project(model, layer, x, positions)
    index = None
    vrow = k_pe
    if d is not None and d.indexer:
        qi, ki, wi = _indexer(model, layer, x, q_resid, positions)
        vrow = torch.cat([k_pe, ki], dim=-1) if dr else ki  # NoPE (glm5_next): no zero-width cat
    model._store(layer.k_cache, slot_mapping, c.unsqueeze(1))
    if vrow.shape[-1]:
        model._store(layer.v_cache, slot_mapping, vrow.unsqueeze(1))
    pooled = layer.pool_key is not None
    if pooled:
        write_pool_keys(model, layer, d, positions, table, slot_mapping)
    elif layer.pool_inplace:
        write_pool_keys_inplace(model, layer, d, positions, table, slot_mapping, vrow)
    if long_capable(layer.spec) and _dsa_long.enabled(table.shape[-1] * model.page_size):
        if want_top or top is not None or bias.dim() == 5:
            raise NotImplementedError("long-context DSA (models/dsa_long.py): MTP selections and verify batches")
        return attention_long(model, layer, q_nope, qi, wi, positions, table, q_resid)
    kc = model._load(layer.k_cache, table).squeeze(-2)  # [(B,) L, r]
    kv = model._load(layer.v_cache, table).squeeze(-2)  # [(B,) L, dr (+ Di)]
    pk = model._gather(layer.pool_key.unsqueeze(1), table).squeeze(-2) if pooled else None  # [(B,) L, Di / kp]
    if table.dim() == 1:  # chunk: one sequence, C queries
        B, Q = 1, T
        kc, kv = kc.unsqueeze(0), kv.unsqueeze(0)
        pk = pk.unsqueeze(0) if pooled else None
    elif bias.dim() == 5:  # extend
        B, Q = table.shape[0], bias.shape[3]
    else:  # decode
        B, Q = T, 1
    L = kc.shape[1]
    vis = bias.reshape(B, Q, L)
    if d is not None and d.indexer:
        index = (qi, wi, kv[..., dr : dr + d.head_dim * (2 if d.kpool > 1 else 1)])
        if pooled:  # the pool keys in order: [B, L / kp, Di]
            index = (*index, pk.reshape(B, L // d.kpool, d.head_dim))
        elif layer.pool_inplace:  # the key part of each pool's last row: [B, L / kp, Di]
            index = (*index, _inplace_keys(model, layer, d, table).reshape(B, L // d.kpool, d.head_dim))
    if fused_kernel_takes(layer, d, table, L, index, top, want_top):
        # KILN_DSA_FUSED=1: a pooled DSA layer's prefill chunk as one kernel, the indexer scores, their selection and
        # the attention over the latent (kernels/dsa_fused.py); the selection never leaves the chip, and no scratch
        # staging (GLM-5.3-Flash has no IndexShare layer reading it).
        from ..kernels import dsa_fused

        (q_r, q_p), wi, _, pk = index
        nh, dn, dv, r = layer.nh, m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
        q_lat = torch.einsum("qhd,hdr->qhr", q_nope.reshape(T, nh, dn), model._w(layer, "w_uk").view(nh, dn, r))
        ol = dsa_fused.attend(q_p.reshape(T, d.n_heads, -1), wi.view(T, d.n_heads), pk[0], positions, q_lat, kc[0],
                              d.topk // d.kpool, d.head_dim ** -0.5, m.softmax_scale).to(model.dtype)
        out = torch.einsum("qhr,hvr->qhv", ol, model._w(layer, "w_uv").view(nh, dv, r)).reshape(T, nh * dv)
        out = F.linear(out, model._w(layer, "o"))
        return (out, None) if want_top else out
    if (decode_kernel_takes(layer, d, table, bias, L, index, top, want_top)):
        # KILN_DSA_DECODE_KERNEL=nki: the selected pools gathered and attended in one kernel (kernels/dsa_decode.py);
        # the full-bucket loads above are dead here and drop out of the graph.
        from ..kernels import dsa_decode
        from . import glm5_next

        (q_r, q_p), wi, _, pk = index
        Hi = d.n_heads
        q4 = (None if q_r is None else q_r.reshape(B, 1, Hi, -1), q_p.reshape(B, 1, Hi, -1))
        rows, sbias = glm5_next.decode_slots(d, q4, wi.view(B, 1, Hi), pk, vis, table, model.page_size,
                                             dsa_decode.NCH * 128)
        nh, dn, dv, r = layer.nh, m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
        q_lat = torch.einsum("bhd,hdr->bhr", q_nope.reshape(B, nh, dn), model._w(layer, "w_uk").view(nh, dn, r))
        ol = dsa_decode.attend(q_lat, layer.k_cache, rows, sbias, m.softmax_scale).to(model.dtype)
        out = torch.einsum("bhr,hvr->bhv", ol, model._w(layer, "w_uv").view(nh, dv, r)).reshape(B, nh * dv)
        return F.linear(out, model._w(layer, "o"))
    as_mask = d is not None and (DSA_MODE != "gather" or d.kpool > 1)
    if d is not None and not d.indexer and L > d.topk and top is None:
        top = _shared(layer, T, L, as_mask).view(B, Q, -1)
    staged = False
    if (d is not None and d.indexer and layer.dsa_stage and STAGE and L > d.topk and top is None
            and T <= layer.dsa_mask.shape[0] and L <= layer.dsa_mask.shape[1]):
        # Select, write it to the scratch and attend with what is read back: the selection is
        # materialised once (and left for any IndexShare layer after this one). Attending with
        # the selection itself made GLM-5.3's decode layer 10.0 ms at B=8 over 4K keys against
        # 4.4 ms this way (trn1, SDK 2.32, 2026-10-03, tools/profile_mla.py; docs/neuron-notes.md).
        _share(layer, _layer_selection(layer, d, index, vis, B, Q), T, as_mask)
        top, staged = _shared(layer, T, L, as_mask).view(B, Q, -1), True
    absorb = not (table.dim() == 1 and PREFILL == "expand")
    out, top = _core(model, layer, q_nope, q_pe, kc, kv[..., :dr], vis, B, Q, top, index, absorb,
                     positions if table.dim() == 1 else None)
    if d is not None and d.indexer and top is not None and layer.dsa_share and not staged:
        _share(layer, top, T, as_mask)
    out = F.linear(out, model._w(layer, "o"))
    return (out, top) if want_top else out


def write_pool_keys_minimal(model, layer, d: DSASpec, positions, table, slot_mapping, ki, state_slot,
                            slots=None) -> None:
    """The minimal layout's pool keys (KV_LAYOUT): every written token's pool key from its kpool tokens' indexer rows,
    taken from this call (ki [T, 2 Di], the tokens' rows in call order) where the call wrote them and from the
    request's open-pool row (layer.open_pool [rows, kpool, 2 Di], slot j for the pool's j-th token) for the earlier
    ones, written as kpool pieces into V (the pool's kpool slots, as write_pool_keys does; padded rows their null-page
    slot). Then the rows of each sequence's still-open pool go to its open-pool row, the others to its slot kpool - 1,
    which no read takes (a pool's last token is always in the call that completes it). A chunk (table [P]) is one
    sequence's consecutive positions from row 0; a decode batch (table [B, P]) one position per row."""
    kp, Di = d.kpool, d.head_dim
    T = positions.shape[0]
    chunk = table.dim() == 1
    real = slot_mapping >= model.page_size
    rows_idx = torch.arange(T, device=positions.device)
    i0 = torch.zeros_like(rows_idx) if chunk else rows_idx  # the first row of each row's sequence in this call
    srow = (state_slot.reshape(-1)[:1].expand(T) if chunk else state_slot.reshape(-1)).to(torch.int64)
    j = torch.arange(kp, device=positions.device, dtype=positions.dtype).view(1, kp)
    first = positions - positions % kp
    q = first.view(T, 1) + j  # [T, kp] the pool's positions
    delta = positions.view(T, 1) - q  # rows back from this row
    in_call = (delta >= 0) & (delta <= (rows_idx - i0).view(T, 1))
    src = (rows_idx.view(T, 1) - delta).clamp(min=0, max=T - 1)
    from_call = ki[src.reshape(-1)].view(T, kp, -1)
    from_buf = layer.open_pool[srow.view(T, 1).expand(T, kp).reshape(-1), j.expand(T, kp).reshape(-1)].view(T, kp, -1)
    rows = torch.where(in_call.unsqueeze(-1), from_call, from_buf.to(from_call.dtype))
    from .glm5_next import pool_keys

    keys = pool_keys(layer, d, rows[..., : 2 * Di])  # [T, Di]
    if slots is None:  # (context parallelism passes the owner's local slots, the dump slot elsewhere)
        slots = _pool_key_slots(model, d, positions, table, slot_mapping)  # [T * kp]
    model._store(layer.v_cache, slots, keys.reshape(T * kp, 1, Di // kp))
    # the open pool's rows: a chunk's trailing incomplete pool (positions from kp floor(end / kp) on), every decode row
    if chunk:
        end = torch.where(real, positions + 1, torch.zeros_like(positions)).amax()
        start = end - end % kp
        keep_row = real & (positions >= start)
    else:
        keep_row = real
    slot_j = torch.where(keep_row, positions % kp, torch.full_like(positions, kp - 1))
    layer.open_pool.index_put_((srow, slot_j.to(torch.int64)), ki.to(layer.open_pool.dtype))


def attention_minimal(model, layer, x, positions, slot_mapping, table, bias, top, want_top, state_slot):
    """attention() of a pooled DSA layer in the minimal layout (KV_LAYOUT): K the latent, V the pool-key pieces, the
    open pool's indexer rows in the request's state row. Then the long path (models/dsa_long.py) or the bucketed one
    with the pool keys read from V."""
    m = layer.spec.mla
    d = m.dsa
    T = x.shape[0]
    if top is not None or want_top or bias.dim() == 5 or getattr(model, "cp", 1) > 1:
        raise NotImplementedError("the minimal DSA cache (KILN_DSA_KV=minimal): MTP, verify and context parallelism")
    if state_slot is None or getattr(layer, "open_pool", None) is None:
        raise RuntimeError("the minimal DSA cache needs the request's state row (layer.open_pool, state_slot)")
    q_nope, q_pe, c, k_pe, q_resid = _project(model, layer, x, positions)
    qi, ki, wi = _indexer(model, layer, x, q_resid, positions)
    model._store(layer.k_cache, slot_mapping, c.unsqueeze(1))
    write_pool_keys_minimal(model, layer, d, positions, table, slot_mapping, ki, state_slot)
    if _dsa_long.enabled(table.shape[-1] * model.page_size) and model.long_dsa:
        return attention_long(model, layer, q_nope, qi, wi, positions, table, q_resid)
    kc = model._load(layer.k_cache, table).squeeze(-2)  # [(B,) L, r]
    pk = context_pool_keys(model, layer, d, table)  # [(B,) P, Di]
    if table.dim() == 1:
        B, Q = 1, T
        kc, pk = kc.unsqueeze(0), pk.unsqueeze(0)
    else:
        B, Q = T, 1
    L = kc.shape[1]
    vis = bias.reshape(B, Q, L)
    index = (qi, wi, None, pk)
    if fused_kernel_takes(layer, d, table, L, index, None, False):
        from ..kernels import dsa_fused

        (q_r, q_p), wi, _, pk = index
        nh, dn, dv, r = layer.nh, m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
        q_lat = torch.einsum("qhd,hdr->qhr", q_nope.reshape(T, nh, dn), model._w(layer, "w_uk").view(nh, dn, r))
        ol = dsa_fused.attend(q_p.reshape(T, d.n_heads, -1), wi.view(T, d.n_heads), pk[0], positions, q_lat, kc[0],
                              d.topk // d.kpool, d.head_dim ** -0.5, m.softmax_scale).to(model.dtype)
        out = torch.einsum("qhr,hvr->qhv", ol, model._w(layer, "w_uv").view(nh, dv, r)).reshape(T, nh * dv)
        return F.linear(out, model._w(layer, "o"))
    if decode_kernel_takes(layer, d, table, bias, L, index, None, False):
        from ..kernels import dsa_decode
        from . import glm5_next

        (q_r, q_p), wi, _, pk = index
        Hi = d.n_heads
        q4 = (None, q_p.reshape(B, 1, Hi, -1))
        rows, sbias = glm5_next.decode_slots(d, q4, wi.view(B, 1, Hi), pk, vis, table, model.page_size,
                                             dsa_decode.NCH * 128)
        nh, dn, dv, r = layer.nh, m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
        q_lat = torch.einsum("bhd,hdr->bhr", q_nope.reshape(B, nh, dn), model._w(layer, "w_uk").view(nh, dn, r))
        ol = dsa_decode.attend(q_lat, layer.k_cache, rows, sbias, m.softmax_scale).to(model.dtype)
        out = torch.einsum("bhr,hvr->bhv", ol, model._w(layer, "w_uv").view(nh, dv, r)).reshape(B, nh * dv)
        return F.linear(out, model._w(layer, "o"))
    absorb = not (table.dim() == 1 and PREFILL == "expand")
    out, _ = _core(model, layer, q_nope, q_pe, kc, kc[..., :0], vis, B, Q, None, index, absorb,
                   positions if table.dim() == 1 else None)
    return F.linear(out, model._w(layer, "o"))


def long_capable(spec) -> bool:
    """Whether an attention layer can run the long-context path (models/dsa_long.py): a pooled DSA indexer
    layer (GLM-5.3-Flash) without RoPE or a sliding window."""
    return (_pooled(spec) and not spec.mla.qk_rope_head_dim and getattr(spec, "window", None) is None
            and spec.mla.dsa.kpool_tail)


def context_pool_keys(model, layer, d: DSASpec, table: torch.Tensor) -> torch.Tensor:
    """[(B,) P, Di]: the pool keys of the context a block table addresses, P = L / kpool, from the pool-key cache
    (separate or in place) or, without one, rebuilt from the cached indexer rows."""
    kp, Di = d.kpool, d.head_dim
    if minimal_layout(layer.spec.mla):  # V is the pool-key pieces
        pk = model._load(layer.v_cache, table).squeeze(-2)  # [(B,) L, Di / kp]
        return pk.reshape(*pk.shape[:-2], -1, Di)
    if layer.pool_key is not None:
        pk = model._gather(layer.pool_key.unsqueeze(1), table).squeeze(-2)  # [(B,) L, Di / kp]
        return pk.reshape(*pk.shape[:-2], -1, Di)
    if layer.pool_inplace:
        return _inplace_keys(model, layer, d, table)
    from .glm5_next import pool_keys

    kv = model._load(layer.v_cache, table).squeeze(-2)  # [(B,) L, 2 Di]
    return pool_keys(layer, d, kv.reshape(*kv.shape[:-2], -1, kp, kv.shape[-1])[..., : 2 * Di])


def _qshard_ok(model, layer, T: int) -> bool:
    """Whether a long-context chunk of T rows runs query-sharded (KILN_DSA_QSHARD): whole-head weights loaded, an
    attention group of more than one rank, the rows dividing over it."""
    A = model.attn_tp
    return (getattr(layer, "qshard", False) and model.tp_size > 1 and A > 1 and T % A == 0
            and getattr(model, "long_grp_onehot", None) is not None)


def attention_long_qshard(model, layer, qi, wi, positions, table, q_resid):
    """attention_long of a chunk with its rows split over the attention group (KILN_DSA_QSHARD): this rank selects
    and attends its block of C / A rows with every head (whole-head q_b, W_UK, W_UV, o_proj), and returns the o_proj
    output of those rows in their place of a zero [C, H] (the attention-group reduction _attn_all_reduce then sums
    the blocks: each row is one rank's value plus zeros)."""
    m = layer.spec.mla
    d = m.dsa
    kp = d.kpool
    T = positions.shape[0]
    A = model.attn_tp
    n = T // A
    Hh, dn, dv, r = layer.spec.num_heads, m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
    keep = d.topk // kp
    idx = model.long_grp_index

    def mine(x):
        return x.reshape(A, n, *x.shape[1:]).index_select(0, idx)[0]

    pos = mine(positions)
    pk = context_pool_keys(model, layer, d, table)  # [P, Di]
    npool = _dsa_long.npools(pos, kp)
    pools, cnt = _select_rows(mine(qi[1].reshape(T, d.n_heads, -1)), mine(wi.view(T, d.n_heads)), pk, npool, keep,
                              d.head_dim ** -0.5)
    from ..kernels import dsa_decode

    cpu = pos.device.type == "cpu"
    n_slots = keep + 1 if cpu else dsa_decode.NCH * 128
    rows, sbias = _dsa_long.slots(pools, cnt, npool, pos, table, model.page_size, kp, n_slots)
    q = F.linear(mine(q_resid), model._w(layer, "q_b_all")).view(n, Hh, dn)
    q_lat = torch.einsum("thd,hdr->thr", q, model._w(layer, "w_uk_all").view(Hh, dn, r))
    if not cpu and _has_slots_kernel():
        from ..kernels import dsa_slots

        ol = dsa_slots.attend(q_lat, layer.k_cache, rows, sbias, m.softmax_scale)
    else:
        ol = _dsa_long.attend(q_lat, layer.k_cache, rows, sbias, m.softmax_scale)
    out = torch.einsum("thr,hvr->thv", ol.to(model.dtype), model._w(layer, "w_uv_all").view(Hh, dv, r))
    out = F.linear(out.reshape(n, Hh * dv), model._w(layer, "o_all"))  # [n, H]
    full = out.unsqueeze(0) * model.long_grp_onehot.to(out.dtype).view(A, 1, 1)
    return full.reshape(T, -1)


def attention_long(model, layer, q_nope, qi, wi, positions, table, q_resid=None):
    """A pooled DSA layer's chunk (table [P], positions [C]) or decode batch (table [B, P], positions [B]) over a
    long context, after its latent, indexer rows and pool keys were written (attention): the pooled indexer's
    scores of the complete candidate pools, their exact top index_topk / kpool, and absorbed MLA over the selected
    pools' tokens plus the query's tail (models/dsa_long.py). Nothing of the context's size per query is formed.
    Returns the o_proj output."""
    m = layer.spec.mla
    d = m.dsa
    kp = d.kpool
    T = q_nope.shape[0]
    if table.dim() == 1 and q_resid is not None and _qshard_ok(model, layer, T):
        return attention_long_qshard(model, layer, qi, wi, positions, table, q_resid)
    nh, dn, dv, r = layer.nh, m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
    keep = d.topk // kp
    scale = d.head_dim ** -0.5
    pk = context_pool_keys(model, layer, d, table)  # [P, Di] (chunk) or [B, P, Di]
    npool = _dsa_long.npools(positions, kp)
    q_p = qi[1].reshape(T, d.n_heads, -1)
    w2 = wi.view(T, d.n_heads)
    if table.dim() == 1:  # a chunk: its queries share one context
        pools, cnt = _long_prefill_select(model, q_p, w2, pk, npool, keep, scale)
    elif _index_scorer_takes(layer, table, q_nope):  # a decode batch through kernels/dsa_index.py
        from ..kernels import dsa_index

        D = d.head_dim
        ppp = model.page_size // kp
        pkc = _index_scorer_cache(layer).view(-1, ppp * D)  # [pages, ppp D]: the pool-key cache as page rows
        P = table.shape[1] * ppp
        cand = torch.where(torch.arange(P, device=npool.device).view(1, P) < npool.view(T, 1), 0.0, NEG_INF)
        sc = dsa_index.scores(q_p, w2 * scale, pkc, dsa_index.page_groups(table), cand.to(torch.float32))
        pools, cnt = _dsa_long.select_device(sc, keep)
    else:  # a decode batch: one context per row
        sc = _dsa_long.scores(q_p, w2, pk, npool, scale)
        pools, cnt = _dsa_long.select(sc, keep) if q_nope.device.type == "cpu" else _dsa_long.select_device(sc, keep)
    from ..kernels import dsa_decode

    cpu = q_nope.device.type == "cpu"
    n_slots = keep + 1 if cpu else dsa_decode.NCH * 128  # the attention kernels' fixed slot count
    if n_slots < keep + 1:
        raise NotImplementedError(f"long-context DSA: {keep} + 1 pools in the decode kernel's {dsa_decode.NCH * 128} slots")
    rows, sbias = _dsa_long.slots(pools, cnt, npool, positions, table, model.page_size, kp, n_slots)
    q_lat = torch.einsum("thd,hdr->thr", q_nope.reshape(T, nh, dn), model._w(layer, "w_uk").view(nh, dn, r))
    if table.dim() == 1 and not cpu and _has_slots_kernel():  # many query rows: the looped kernel
        from ..kernels import dsa_slots

        ol = dsa_slots.attend(q_lat, layer.k_cache, rows, sbias, m.softmax_scale)
    else:
        ol = _dsa_long.attend(q_lat, layer.k_cache, rows, sbias, m.softmax_scale)
    out = torch.einsum("thr,hvr->thv", ol.to(model.dtype), model._w(layer, "w_uv").view(nh, dv, r)).reshape(T, nh * dv)
    return F.linear(out, model._w(layer, "o"))


def _cp_gather(model, x: torch.Tensor) -> torch.Tensor:
    """[A, ...]: every attention-group member's x, in attention-rank order (a zero-padded group all-reduce: exact)."""
    A = model.cp
    full = x.unsqueeze(0) * model.long_grp_onehot.to(x.dtype).view(A, *[1] * x.dim())
    return model._all_reduce(full.contiguous(), model.attn_group)


def _cp_reduce_scatter(model, x: torch.Tensor) -> torch.Tensor:
    """Block attention-rank of A equal row blocks of x [A r, ...] summed over the attention group."""
    if x.device.type == "cpu":
        y = model._all_reduce(x, model.attn_group)
        return y.reshape(model.cp, -1, *x.shape[1:]).index_select(0, model.long_grp_index)[0]
    import torch.distributed._functional_collectives as funcol

    return funcol.reduce_scatter_tensor(x, "sum", 0, model.attn_group)


def _cp_local_all(sc: torch.Tensor, nloc: torch.Tensor, prow: torch.Tensor, keep: int):
    """(lp [T, keep] int64, lc [T] int64, lval [T, keep] fp32, rows_sel [T, keep] int64): attention_cp's local list of
    a decode batch whose Pl local pools fit keep (Pl <= keep), without a selection. sc [T, Pl] are the scores
    (dsa_long.scores: NEG_INF at local pools m >= nloc), nloc [T] the visible local pools, prow [T, Pl] their pool rows
    (dsa_long.cp_pool_rows). The keep best of at most keep candidates are all of them, and the candidates are the prefix
    m < nloc, so select_device + compact give entry k = k for k < lc = min(nloc, Pl) and 0 after it; the list's values
    are the scores there (NEG_INF after), and pool_row(m) of the block table is prow[:, m]. Equal to that path's tensors
    element for element (tests/test_dsa_long.py), with the padding past Pl as the masked entry 0 it gives."""
    T, Pl = sc.shape
    if Pl > keep:
        raise ValueError(f"_cp_local_all: {Pl} local pools exceed keep {keep}")
    k = torch.arange(keep, device=sc.device).view(1, keep)
    lc = nloc.to(torch.int64).clamp(max=Pl)
    on = k < lc.view(T, 1)
    lp = torch.where(on, k, torch.zeros_like(k))
    pr = prow.to(torch.int64)
    if Pl < keep:  # pad to keep (padding is masked: k >= Pl >= lc)
        sc = F.pad(sc, (0, keep - Pl))
        pr = F.pad(pr, (0, keep - Pl))
    lval = torch.where(on, sc, torch.full_like(sc, NEG_INF))
    rows = torch.where(on, pr, pr[:, :1])
    return lp, lc, lval, rows


def attention_cp(model, layer, x, positions, slot_mapping, table, state_slot=None):
    """attention() of a pooled DSA layer under context parallelism (models/dsa_long.py "context parallelism"): this
    rank holds the context's pools m A + rank only. A chunk (table [P], positions [C]) or a decode batch (table
    [B, P], positions [B]). Returns the o_proj output of this rank's heads (the head partial _attn_all_reduce sums).
    In the minimal layout (KV_LAYOUT) V holds the owner's pool-key pieces in the KV dtype and every rank keeps the
    request's open-pool row (state_slot) the same way (each rank has every token's indexer rows)."""
    m = layer.spec.mla
    d = m.dsa
    kp, Di = d.kpool, d.head_dim
    A = model.cp
    rank = model.long_grp_index
    T = x.shape[0]
    nh, dn, dv, r = layer.nh, m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
    keep = d.topk // kp
    scale = Di ** -0.5
    ps = model.page_size
    minimal = minimal_layout(m)
    if layer.pool_key is None and not minimal:
        raise NotImplementedError("context-parallel DSA needs the separate pool-key cache (KILN_DSA_POOL_CACHE)")
    q_nope, q_pe, c, k_pe, q_resid = _project(model, layer, x, positions)
    qi, ki, wi = _indexer(model, layer, x, q_resid, positions)

    # A padded row (slot_mapping in the null page, ModelRunner.pad_slots) sits at position 0 of a real sequence's table
    # in a chunk: it must not write that sequence's first pool (docs/neuron-notes.md "Padded rows wrote the slot they
    # read"), so it writes the null page.
    real = slot_mapping >= ps
    dump = _dsa_long.cp_dump_slot(ps, A)
    ls = _dsa_long.cp_local_slots(positions, table, ps, kp, A, rank)
    ls = torch.where(real, ls, torch.full_like(ls, dump))
    model._store(layer.k_cache, ls, c.unsqueeze(1))
    if minimal:
        # the pool keys from this call's indexer rows and the open-pool row, as kpool pieces into the owner's V slots
        # (the others, and padded rows, write the dump slot); the open-pool row is updated on every rank
        first = positions - positions % kp
        pos4 = (first.unsqueeze(-1) + torch.arange(kp, device=positions.device, dtype=positions.dtype)).reshape(-1)
        tb4 = table if table.dim() == 1 else table.repeat_interleave(kp, dim=0)
        slots4 = _dsa_long.cp_local_slots(pos4, tb4, ps, kp, A, rank)
        slots4 = torch.where(real.repeat_interleave(kp), slots4, torch.full_like(slots4, dump))
        write_pool_keys_minimal(model, layer, d, positions, table, slot_mapping, ki, state_slot, slots=slots4)
    else:
        model._store(layer.v_cache, ls, ki.unsqueeze(1))
        # every written token's pool, from its kpool cached rows (all on the pool's owner; the others write their null
        # page), as write_pool_keys does
        first = positions - positions % kp
        pos4 = (first.unsqueeze(-1) + torch.arange(kp, device=positions.device, dtype=positions.dtype)).reshape(-1)
        tb4 = table if table.dim() == 1 else table.repeat_interleave(kp, dim=0)
        slots4 = _dsa_long.cp_local_slots(pos4, tb4, ps, kp, A, rank)
        slots4 = torch.where(real.repeat_interleave(kp), slots4, torch.full_like(slots4, dump))
        rows = layer.v_cache[slots4]
        if model.fp8_max is not None:
            rows = rows.to(model.dtype)
        from .glm5_next import pool_keys

        keys = pool_keys(layer, d, rows.reshape(T, kp, -1)[..., : 2 * Di])
        layer.pool_key.index_put_((slots4,), keys.reshape(T * kp, Di // kp).to(layer.pool_key.dtype))
    # this rank's pools of the context(s): local pool m = context pool m A + rank
    prow = _dsa_long.cp_pool_rows(table, ps, kp, A)  # [(B,) Pl]
    if CP_PAGE_KEYS and table.dim() == 2:  # whole page rows: row page of [pages, ppl Di] is pool rows page ppl ..
        ppk = ps // (kp * A)
        pk = (layer.v_cache if minimal else layer.pool_key).view(-1, ppk * Di)[table].view(T, prow.shape[-1], Di)
    else:
        pk = (layer.v_cache if minimal else layer.pool_key).view(-1, Di)[prow]  # [(B,) Pl, Di]
    if model.fp8_max is not None and pk.dtype != model.dtype:
        pk = pk.to(model.dtype)
    npool = _dsa_long.npools(positions, kp)
    nloc = _dsa_long.cp_local_count(npool, A, rank)
    q_p = qi[1].reshape(T, d.n_heads, -1)
    w2 = wi.view(T, d.n_heads)
    # A rank may hold fewer local pools than keep (a short context's bucket: 33 pages x 8 local pools = 264 < 512 at
    # A = 8); all of them are then candidates. Those shapes take the scores and the decode form's selection (the kernel
    # needs keep <= its pools), with the local list padded to keep; every other shape traces as before.
    few = prow.shape[-1] < keep
    classes = CP_SLOT_CLASSES and table.dim() == 1  # value-ordered local lists (_cp_attend_classes)
    direct = CP_ALL_LOCAL and table.dim() == 2 and not prow.shape[-1] > keep
    if direct:  # a decode batch whose local pools fit keep: all visible ones, in order
        sc = _dsa_long.scores(q_p, w2, pk, nloc, scale)  # [T, Pl]
        lp, lc, lval, rows_dir = _cp_local_all(sc, nloc, prow, keep)
    elif table.dim() == 1 and q_nope.device.type != "cpu" and not few:  # a chunk: the selection kernel (and scores)
        from ..kernels import dsa_long_select

        lp, lc, lval = _select_tiles(q_p, w2, pk, nloc, keep, scale, vorder=classes)
    else:
        sc = _dsa_long.scores(q_p, w2, pk, nloc, scale)  # [T, Pl]
        if q_nope.device.type == "cpu":
            lp, lc = _dsa_long.select(sc, keep)
        else:
            lp, lc = _dsa_long.select_device(sc, keep)
        k = torch.arange(keep, device=lp.device).view(1, keep)
        fill = sc.new_full((T, keep), NEG_INF) if few else torch.full_like(sc[:, :keep], NEG_INF)
        lval = torch.where(k < lc.view(T, 1), sc.gather(1, lp), fill)
        if classes:  # the kernel's vorder form, on the host and on the shapes the kernel does not take
            from ..kernels import dsa_long_select

            lp, lc, lval = dsa_long_select.value_order(lp, lc, lval)
    cpool = lp * A + rank.view(())  # context pool indices of the local list
    every_val = _cp_gather(model, lval).permute(1, 0, 2)  # [T, A, keep]
    every_c = _cp_gather(model, cpool.to(torch.float32)).permute(1, 0, 2)
    # a decode batch above CP_MERGE_ROWS rows merges in pieces of it; a prefill chunk's merge is one piece
    sel = _dsa_long.cp_merge(every_val, every_c, keep, _dsa_long.CP_MERGE_ROWS if table.dim() == 2 else None,
                             prow.shape[-1] * A)  # [T, A, keep]
    mine = sel.permute(1, 0, 2).reshape(A, T * keep)
    mine = mine.to(torch.float32).index_select(0, rank).view(T, keep) > 0  # this rank's selected entries
    # slots: the local list (selected ones attended), then the tail pool if this rank owns it
    ppl = ps // (kp * A)
    tb = table.view(1, -1).expand(T, -1) if table.dim() == 1 else table
    def pool_row(mm):
        pg = torch.floor(mm.to(torch.float32) * (1.0 / ppl)).to(torch.int64)
        return tb.gather(1, pg.clamp(max=tb.shape[1] - 1)) * ppl + (mm - pg * ppl)
    tail_m = torch.floor(npool.to(torch.float32) * (1.0 / A)).to(torch.int64)
    tail_own = (npool - tail_m * A) == rank.view(())
    Pl = prow.shape[-1]
    rows_sel = rows_dir if direct else pool_row(lp)
    rows_tail = pool_row(tail_m.clamp(max=Pl - 1).view(T, 1))
    cpu = q_nope.device.type == "cpu"
    from ..kernels import dsa_decode

    n_slots = keep + 1 if cpu else dsa_decode.NCH * 128
    # (chunks only: a decode batch has a row or a few per group, and its graphs failed neuronx-cc 2.27 with the classes,
    # NCC_ILSM901 "LegalizeSundaMacro ... Cannot split" on the f32[1] floor of the tail pool, q/lc-trn1-cpc)
    if CP_SLOT_CLASSES and table.dim() == 1:
        q_lat = torch.einsum("thd,hdr->thr", q_nope.reshape(T, nh, dn), model._w(layer, "w_uk").view(nh, dn, r))
        q_all = _cp_gather(model, q_lat).permute(1, 0, 2, 3).reshape(T, A * nh, r)
        o_r, lse_r = _cp_attend_classes(q_all, layer.k_cache, rows_sel, rows_tail, mine, tail_own, npool, positions,
                                        m.softmax_scale, kp, CP_SLOTS_SMALL, n_slots)
        return _cp_combine(model, layer, o_r, lse_r, T, A, nh, r, dv)
    # by broadcasting, not concatenation (dsa_long.slots: NCC_IFML902 on a concatenate of these pieces)
    sl = torch.arange(n_slots, device=positions.device).view(1, n_slots)
    cl = sl.clamp(max=keep - 1).expand(T, n_slots)
    srows = torch.where(sl < keep, torch.gather(rows_sel, 1, cl),
                        torch.where(sl == keep, rows_tail, torch.zeros_like(rows_tail)))
    t4 = torch.arange(kp, device=positions.device).view(1, 1, kp)
    sel_ok = ((sl < keep) & torch.gather(mine, 1, cl)).unsqueeze(-1)
    tail_ok = ((sl == keep) & tail_own.view(T, 1)).unsqueeze(-1) & (npool.view(T, 1, 1) * kp + t4
                                                                     <= positions.view(T, 1, 1))
    sbias = torch.where(sel_ok | tail_ok, 0.0, NEG_INF).to(torch.float32)
    # every head's latent query (the heads are split over the group: gather them), partial attention, combine
    q_lat = torch.einsum("thd,hdr->thr", q_nope.reshape(T, nh, dn), model._w(layer, "w_uk").view(nh, dn, r))
    q_all = _cp_gather(model, q_lat).permute(1, 0, 2, 3).reshape(T, A * nh, r)
    o_r, lse_r = _dsa_long.cp_attend_partial(q_all, layer.k_cache, srows, sbias, m.softmax_scale)
    lse_all = _cp_gather(model, lse_r.float())  # [A, T, H]
    LSE = torch.logsumexp(lse_all, dim=0)  # [T, H]
    part = o_r * torch.exp(lse_r - LSE).unsqueeze(-1)  # [T, H, R]
    part = part.view(T, A, nh, r).permute(1, 0, 2, 3).reshape(A * T, nh, r).contiguous()
    o = _cp_reduce_scatter(model, part)  # [T, nh, R]: this rank's heads
    out = torch.einsum("thr,hvr->thv", o.to(model.dtype), model._w(layer, "w_uv").view(nh, dv, r)).reshape(T, nh * dv)
    return F.linear(out, model._w(layer, "o"))


def _cp_combine(model, layer, o_r, lse_r, T: int, A: int, nh: int, r: int, dv: int):
    """attention_cp's tail: the ranks' partial attentions combined by their log-sum-exps, this rank's heads, o_proj."""
    lse_all = _cp_gather(model, lse_r.float())  # [A, T, H]
    LSE = torch.logsumexp(lse_all, dim=0)  # [T, H]
    part = o_r * torch.exp(lse_r - LSE).unsqueeze(-1)  # [T, H, R]
    part = part.view(T, A, nh, r).permute(1, 0, 2, 3).reshape(A * T, nh, r).contiguous()
    o = _cp_reduce_scatter(model, part)  # [T, nh, R]: this rank's heads
    out = torch.einsum("thr,hvr->thv", o.to(model.dtype), model._w(layer, "w_uv").view(nh, dv, r)).reshape(T, nh * dv)
    return F.linear(out, model._w(layer, "o"))


def _cp_attend_classes(q_all, kc, rows_sel, rows_tail, mine, tail_own, npool, positions, scale: float, kp: int,
                       small: int, full: int):
    """(o [T, H, R] fp32, lse [T, H] fp32): this rank's partial attention of every row over its selected local pools
    and its tail, as attention_cp's fixed slots give it, with each row in a buffer sized by where its selected entries
    end. The local lists come in the selection kernel's value order (score descending, pool ascending:
    dsa_long_select vorder), where the selected entries of a row are its first ones (the merge takes every value above
    the threshold and the ties with the smallest pools), so a row whose last selected entry sits before `small` - 1
    takes the small buffer: the list's first `small` - 1 slots as they are and the tail pool last, the others masked by
    `mine` as in the fixed slots; every other row the full one (attention_cp's fixed layout, `full` >= keep + 1
    slots). Masking by `mine` keeps it exact whatever the order. Each buffer holds its rows first (compact over the
    rows) and kernels/dsa_slots_n.py runs only those: n_small c(small) + n_full c(full). No gather of the local list
    (a per-element gather of [1024, 512] cost ~60 ms on trn1: tools/probe_cp_parts.py)."""
    from ..kernels import dsa_slots_n

    T, keep = mine.shape
    dev = q_all.device
    S1 = min(small - 1, keep)  # list slots of the small buffer; the tail takes slot S1
    j = torch.arange(keep, device=dev).view(1, keep)
    bound = torch.where(mine, j + 1, torch.zeros_like(j)).amax(-1)  # [T] one past the last selected entry
    fits = bound <= S1
    t4 = torch.arange(kp, device=dev).view(1, 1, kp)
    tail_tok = tail_own.view(T, 1, 1) & (npool.view(T, 1, 1) * kp + t4 <= positions.view(T, 1, 1))  # [T, 1, kp]

    def build(C: int, L: int):
        """C slots: the local list's first L entries, the tail at slot L, the rest padding."""
        sl = torch.arange(C, device=dev).view(1, C)
        if C <= keep:  # the list's first C columns as a slice (no gather), the tail and padding by broadcasting
            rs, ms = rows_sel[:, :C], mine[:, :C]
        else:
            cl = sl.clamp(max=keep - 1).expand(T, C)
            rs, ms = torch.gather(rows_sel, 1, cl), torch.gather(mine, 1, cl)
        rows = torch.where(sl < L, rs, torch.where(sl == L, rows_tail, torch.zeros_like(rows_tail)))
        ok = ((sl < L) & ms).unsqueeze(-1) | ((sl == L).unsqueeze(-1) & tail_tok)
        return rows, torch.where(ok, 0.0, NEG_INF).to(torch.float32)

    out_o, out_l = [], []
    for want, (C, L) in ((fits, (small, S1)), (~fits, (full, keep))):
        idx, n = _dsa_long.compact(want.to(torch.float32).view(1, T), T)  # this buffer's rows first
        idx = idx.view(T)
        rows, bias = build(C, L)
        o, ls = dsa_slots_n.attend(q_all[idx], kc, rows[idx], bias[idx], scale, n.view(1), lse=True)
        # row b's place in its buffer: the rows of its class before it (an exact fp32 count)
        pos = (torch.cumsum(want.to(torch.float32), 0) - 1.0).clamp(min=0.0).to(torch.int64)
        out_o.append(o[pos])
        out_l.append(ls[pos])
    o = torch.where(fits.view(T, 1, 1), out_o[0], out_o[1])
    lse = torch.where(fits.view(T, 1), out_l[0], out_l[1])
    return o, lse


def _index_scorer_takes(layer, table, q) -> bool:
    """Whether a decode batch's scores come from kernels/dsa_index.py (dsa_long.SCORER "index"): the separate bf16
    pool-key cache, or the minimal layout's fp8 pool-key pieces in V (the kernel's fp8 form, REV8), a page bucket of
    whole 128-page groups, 128-dim keys, a Neuron device."""
    if not (_dsa_long.SCORER == "index" and q.device.type != "cpu" and table.dim() == 2 and table.shape[1] % 128 == 0
            and layer.spec.mla.dsa.head_dim == 128):
        return False
    if layer.pool_key is not None:
        return layer.pool_key.dtype == torch.bfloat16
    return minimal_layout(layer.spec.mla) and layer.v_cache.dtype == torch.float8_e4m3fn


def _index_scorer_cache(layer):
    """The pool-key cache kernels/dsa_index.py reads: the separate one, or (minimal layout) V's pieces."""
    return layer.pool_key if layer.pool_key is not None else layer.v_cache


def _has_slots_kernel() -> bool:
    """Whether kernels/dsa_slots.py (the looped many-row attention) is in this tree; without it a chunk's rows go
    through kernels/dsa_decode.py, which unrolls them (fine for small chunks only). A module constant: dynamo
    cannot trace importlib."""
    return _HAS_SLOTS_KERNEL


_HAS_SLOTS_KERNEL = os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                "kernels", "dsa_slots.py"))


def _select_rows(q_p, w, pk, npool, keep: int, scale: float):
    """(pools [n, keep], cnt [n]) of n queries sharing one context's pool keys pk [P, Di]: kernels/dsa_long_select.py on
    the device (query tiles of 128), the host selection of models/dsa_long.py on the CPU."""
    if q_p.device.type == "cpu":
        return _dsa_long.select(_dsa_long.scores(q_p, w, pk, npool, scale), keep)
    from ..kernels import dsa_long_select

    P = pk.shape[0]
    try:
        ok = dsa_long_select.supported(q_p.shape[1], q_p.shape[2], P, keep, dsa_long_select.pick_sub(P, keep))
    except NotImplementedError:
        ok = False
    if not ok:
        # shapes the kernel does not take (keep below 8, a test config): the decode form's selection
        return _dsa_long.select_device(_dsa_long.scores(q_p, w, pk, npool, scale), keep)
    pools, cnt, _ = _select_tiles(q_p, w, pk, npool, keep, scale)
    return pools, cnt


def _select_tiles(q_p, w, pk, npool, keep: int, scale: float, vorder: bool = False):
    """kernels/dsa_long_select.py's select, one kernel call per 128 queries. The kernel's own device loop over
    128-query tiles returned wrong selections past the first tile (trn1.2xlarge, nki 0.6.0, 2026-10-05:
    `tools/probe_dsa_long.py select --rows 1024 --pools 512 2048` 112 / 496 of 1024 rows differ, N = 128 exact;
    the context-parallel W1M run then failed with "Out of bounds access" in its prefill graphs), so a call never
    holds more than one tile. A single tile is the original call unchanged (the same trace and cache keys)."""
    from ..kernels import dsa_long_select

    n = q_p.shape[0]
    kw = {"vorder": True} if vorder else {}  # (vorder: the kernel's value order, CP slot classes)
    if n <= 128:
        return dsa_long_select.select(q_p, w, pk, npool, keep, scale, **kw)
    parts = [dsa_long_select.select(q_p[i:i + 128], w[i:i + 128], pk, npool[i:i + 128], keep, scale, **kw)
             for i in range(0, n, 128)]
    return tuple(torch.cat([p[j] for p in parts]) for j in range(3))


def _long_prefill_select(model, q_p, w, pk, npool, keep: int, scale: float):
    """The selection of a chunk's C queries. With an attention group of A > 1 ranks (which all hold this context and
    would all select the same), each rank selects its block of C / A queries and the blocks are gathered over the
    group by a zero-padded fp32 all-reduce (exact: indices below 2^24), so the indexer's work is not replicated."""
    C = q_p.shape[0]
    A = model.attn_tp if (model.tp_size > 1 and getattr(model, "long_grp_onehot", None) is not None) else 1
    if A == 1 or C % A:
        return _select_rows(q_p, w, pk, npool, keep, scale)
    n = C // A
    idx = model.long_grp_index

    def mine(x):
        return x.reshape(A, n, *x.shape[1:]).index_select(0, idx)[0]

    pools, cnt = _select_rows(mine(q_p), mine(w), pk, mine(npool), keep, scale)
    part = torch.cat([pools.to(torch.float32), cnt.to(torch.float32).unsqueeze(-1)], dim=-1)  # [n, keep + 1]
    full = part.unsqueeze(0) * model.long_grp_onehot.view(A, 1, 1)
    full = model._all_reduce(full.reshape(C, keep + 1), model.attn_group).to(torch.int64)
    return full[:, :keep], full[:, keep]


def attention_joint(model, layer, x, positions, slot_mapping, table, bias, dec_table, dec_bias, C: int):
    """attention() of a mixed batch's rows (decoder.MIXED_MIXERS "joint"): x [C + D] this group's chunk rows
    then decode rows. The projections and the indexer's query, key and weights run once over all rows, the
    latent, indexer and pool-key caches are written once for all of them (the inplace pool keys' second
    store of the indexer rows too, write_pool_keys_inplace), then each part reads its own
    context and selects and attends in its own batch form (the chunk's prefill form over table / bias, the
    decode rows' decode form over dec_table / dec_bias), and o_proj runs once. The selection is attended
    with directly, not through the scratch (KILN_DSA_STAGE): it would be written twice in one graph, and
    with the selection kernels it bought nothing (docs/neuron-notes.md, "Pool keys cached")."""
    m = layer.spec.mla
    d = m.dsa
    dr = m.qk_rope_head_dim
    T = x.shape[0]
    D = T - C
    q_nope, q_pe, c, k_pe, q_resid = _project(model, layer, x, positions)
    qi = ki = wi = None
    vrow = k_pe
    if d is not None and d.indexer:
        qi, ki, wi = _indexer(model, layer, x, q_resid, positions)
        vrow = torch.cat([k_pe, ki], dim=-1) if dr else ki
    model._store(layer.k_cache, slot_mapping, c.unsqueeze(1))
    if vrow.shape[-1]:
        model._store(layer.v_cache, slot_mapping, vrow.unsqueeze(1))
    pooled = layer.pool_key is not None
    if pooled:  # every written token's pool, the chunk's through its table and the decode rows' through theirs
        sc, kc_ = _pool_key_rows(model, layer, d, positions[:C], table, slot_mapping[:C])
        sd, kd_ = _pool_key_rows(model, layer, d, positions[C:], dec_table, slot_mapping[C:])
        layer.pool_key.index_put_((torch.cat([sc, sd]),), torch.cat([kc_, kd_]).to(layer.pool_key.dtype))
    elif layer.pool_inplace:  # write_pool_keys_inplace's second store, once for both parts
        rows_c = _inplace_rows(model, layer, d, positions[:C], table, vrow[:C])
        rows_d = _inplace_rows(model, layer, d, positions[C:], dec_table, vrow[C:])
        model._store(layer.v_cache, slot_mapping, torch.cat([rows_c, rows_d]).unsqueeze(1))
    outs = []
    for part, (tb, bs, B, Q, rows) in enumerate(((table, bias, 1, C, slice(0, C)), (dec_table, dec_bias, D, 1,
                                                                                      slice(C, T)))):
        kc = model._load(layer.k_cache, tb).squeeze(-2)
        kv = model._load(layer.v_cache, tb).squeeze(-2)
        pk = model._gather(layer.pool_key.unsqueeze(1), tb).squeeze(-2) if pooled else None
        if part == 0:
            kc, kv = kc.unsqueeze(0), kv.unsqueeze(0)
            pk = pk.unsqueeze(0) if pooled else None
        L = kc.shape[1]
        vis = bs.reshape(B, Q, L)
        index = None
        if d is not None and d.indexer:
            qp = (None if qi[0] is None else qi[0][rows], qi[1][rows])
            index = (qp, wi[rows], kv[..., dr : dr + d.head_dim * (2 if d.kpool > 1 else 1)])
            if pooled:
                index = (*index, pk.reshape(B, L // d.kpool, d.head_dim))
            elif layer.pool_inplace:
                index = (*index, _inplace_keys(model, layer, d, tb).reshape(B, L // d.kpool, d.head_dim))
        if d is not None and not d.indexer:
            raise NotImplementedError("a mixed batch with IndexShare layers (they read the selection scratch)")
        if MIXED_KERNELS and part == 0 and fused_kernel_takes(layer, d, tb, L, index, None, False):
            # The chunk through the fused selection-and-attention kernel, as attention() runs an unmixed chunk
            # (kernels/dsa_fused.py), instead of the separate selection and the static attention kernel.
            outs.append(_fused_rows(model, layer, q_nope[rows], index, positions[rows], kc))
            continue
        if MIXED_KERNELS and part == 1 and decode_kernel_takes(layer, d, tb, bs, L, index, None, False):
            # The decode rows through the DSA decode kernel, as attention() runs a decode call (kernels/dsa_decode.py),
            # instead of the mask form over every key of the bucket.
            outs.append(_decode_rows(model, layer, q_nope[rows], index, vis, tb, B))
            continue
        o, _ = _core(model, layer, q_nope[rows], q_pe[rows], kc, kv[..., :dr], vis, B, Q, None, index,
                     absorb=not (part == 0 and PREFILL == "expand"), pos=positions[rows] if part == 0 else None)
        outs.append(o)
    return F.linear(torch.cat(outs), model._w(layer, "o"))


# KILN_MIXED_KERNELS (default 1): a mixed batch's chunk rows take the fused DSA kernel and its decode rows the DSA decode
# kernel wherever an unmixed call of the same rows would (attention_joint). 0: the forms mixed graphs were first built with
# (the chunk through _core's selection and static attention kernel, the decode rows through the mask form), where mixed
# batches measured G64 152.1 against 156.2 unmixed on engine-v0 f70c14b (docs/neuron-notes.md "The final combined
# measurement"). Read when a graph is traced.
MIXED_KERNELS = os.environ.get("KILN_MIXED_KERNELS", "1") == "1"


def _fused_rows(model, layer, q_nope, index, positions, kc):
    """attention()'s fused-kernel branch on a chunk's rows, before o_proj: [C, nh * dv]."""
    from ..kernels import dsa_fused

    m = layer.spec.mla
    d = m.dsa
    T = q_nope.shape[0]
    (q_r, q_p), wi, _, pk = index
    nh, dn, dv, r = layer.nh, m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
    q_lat = torch.einsum("qhd,hdr->qhr", q_nope.reshape(T, nh, dn), model._w(layer, "w_uk").view(nh, dn, r))
    ol = dsa_fused.attend(q_p.reshape(T, d.n_heads, -1), wi.view(T, d.n_heads), pk[0], positions, q_lat, kc[0],
                          d.topk // d.kpool, d.head_dim ** -0.5, m.softmax_scale).to(model.dtype)
    return torch.einsum("qhr,hvr->qhv", ol, model._w(layer, "w_uv").view(nh, dv, r)).reshape(T, nh * dv)


def _decode_rows(model, layer, q_nope, index, vis, table, B: int):
    """attention()'s decode-kernel branch on B decode rows, before o_proj: [B, nh * dv]."""
    from ..kernels import dsa_decode
    from . import glm5_next

    m = layer.spec.mla
    d = m.dsa
    (q_r, q_p), wi, _, pk = index
    Hi = d.n_heads
    q4 = (None if q_r is None else q_r.reshape(B, 1, Hi, -1), q_p.reshape(B, 1, Hi, -1))
    rows, sbias = glm5_next.decode_slots(d, q4, wi.view(B, 1, Hi), pk, vis, table, model.page_size,
                                         dsa_decode.NCH * 128)
    nh, dn, dv, r = layer.nh, m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
    q_lat = torch.einsum("bhd,hdr->bhr", q_nope.reshape(B, nh, dn), model._w(layer, "w_uk").view(nh, dn, r))
    ol = dsa_decode.attend(q_lat, layer.k_cache, rows, sbias, m.softmax_scale).to(model.dtype)
    return torch.einsum("bhr,hvr->bhv", ol, model._w(layer, "w_uv").view(nh, dv, r)).reshape(B, nh * dv)


def _pool_key_slots(model, d: DSASpec, positions: torch.Tensor, table: torch.Tensor,
                    slot_mapping: torch.Tensor) -> torch.Tensor:
    """[T * kpool]: the slots of each written token's pool through the block table, except that a padded row
    (slot_mapping < page_size: the null page, ModelRunner.pad_slots) targets its own slot kpool times. A
    padded row sits at position 0, so through a real sequence's table (a chunk's padded tail) its pool is
    that sequence's pool 0: it used to rewrite those slots, beside the real rows that own them (docs/
    neuron-notes.md "Padded rows wrote the slot they read")."""
    kp = d.kpool
    T = positions.shape[0]
    first = positions - positions % kp  # kpool is a power of two here (4): exact
    pos = first.unsqueeze(-1) + torch.arange(kp, device=positions.device, dtype=positions.dtype)  # [T, kp]
    slots = model._slots(pos, table).reshape(T, kp)
    real = (slot_mapping >= model.page_size).unsqueeze(-1)
    return torch.where(real, slots, slot_mapping.unsqueeze(-1).expand(T, kp)).reshape(-1)


def _pool_key_rows(model, layer, d: DSASpec, positions: torch.Tensor, table: torch.Tensor,
                   slot_mapping: torch.Tensor):
    """write_pool_keys' slots [T * kpool] and pool-key pieces [T * kpool, Di / kpool], not written."""
    kp, Di = d.kpool, d.head_dim
    dr = layer.spec.mla.qk_rope_head_dim
    T = positions.shape[0]
    slots = _pool_key_slots(model, d, positions, table, slot_mapping)
    rows = layer.v_cache[slots]
    if model.fp8_max is not None:
        rows = rows.to(model.dtype)
    from .glm5_next import pool_keys

    keys = pool_keys(layer, d, rows.reshape(T, kp, -1)[..., dr : dr + 2 * Di])
    return slots, keys.reshape(T * kp, Di // kp)


def fused_kernel_takes(layer, d, table, L: int, index, top, want_top: bool) -> bool:
    """Whether a pooled DSA layer's prefill chunk runs kernels/dsa_fused.py (KILN_DSA_FUSED=1): the chunk form (one
    sequence), a sparse bucket (L > index_topk) of an indexer layer with pool keys of 4 tokens and the tail, NoPE, the
    nki selection, its own selection (none given or returned, no IndexShare layer reading it)."""
    from ..kernels import dsa_fused

    m = layer.spec.mla
    return (dsa_fused.FUSED and d is not None and d.indexer and d.kpool == 4 and d.kpool_tail and table.dim() == 1
            and L > d.topk and top is None and not want_top and index is not None and len(index) == 4
            and not m.qk_rope_head_dim and index[0][0] is None and not getattr(layer, "dsa_share", False)
            and dsa_select.SELECT == "nki"
            and dsa_fused.supported(1, L // 4, L, m.kv_lora_rank, d.n_heads, d.head_dim, d.kpool))


def prefill_kernel_takes(B: int, Q: int, L: int, r: int, dr: int) -> bool:
    """Whether a pooled DSA layer's sparse prefill chunk runs kernels/dsa_prefill.py (KILN_DSA_PREFILL_KERNEL=nki):
    one sequence (the chunk form; the queries are padded to whole tiles), whole 128-key tiles, NoPE, a latent of at
    most 512."""
    from ..kernels import dsa_prefill

    return dsa_prefill.KERNEL == "nki" and B == 1 and not dr and dsa_prefill.supported(Q, L, r)


def decode_kernel_takes(layer, d, table, bias, L: int, index, top, want_top: bool) -> bool:
    """Whether a pooled DSA layer's decode step runs kernels/dsa_decode.py (KILN_DSA_DECODE_KERNEL=nki): a decode
    batch (one query per row) of a sparse bucket (L > index_topk) of an indexer layer with pool keys (a pool-key
    cache, separate or in place), NoPE, pools of the kernel's size, its own selection (no shared or earlier-pass
    one, none returned to the caller, no IndexShare layer reading it)."""
    from ..kernels import dsa_decode

    m = layer.spec.mla
    return (dsa_decode.KERNEL == "nki" and d is not None and d.indexer and d.kpool == dsa_decode.KP
            and d.topk // d.kpool + 1 <= dsa_decode.NCH * 128 and L > d.topk and table.dim() == 2
            and bias.dim() != 5 and top is None and not want_top and index is not None and len(index) == 4
            and not m.qk_rope_head_dim and not getattr(layer, "dsa_share", False))


def write_pool_keys(model, layer, d: DSASpec, positions: torch.Tensor, table: torch.Tensor,
                    slot_mapping: torch.Tensor) -> None:
    """Recompute the pool key of every written token's pool from the cached indexer rows of its
    kpool tokens (after this call's rows were stored) and write it to the pool-key cache as kpool
    pieces, one per slot of the pool (pool_key_width). A pool's key is therefore rewritten whenever
    one of its tokens is, and holds the key of its current rows: once its last token is written it
    is the key the per-call form would compute, and the selection only scores complete pools (the
    query's own incomplete pool is the tail, attended whole). Rows of several tokens of one pool
    write the same values to the same slots. Padded rows write only their own null-page slot
    (_pool_key_slots): no slot a real row owns, and not the null page's slot 0 that padded rows read."""
    kp, Di = d.kpool, d.head_dim
    dr = layer.spec.mla.qk_rope_head_dim
    T = positions.shape[0]
    slots = _pool_key_slots(model, d, positions, table, slot_mapping)  # [T * kp]
    rows = layer.v_cache[slots]  # [T * kp, 1, dr + 2 Di]
    if model.fp8_max is not None:
        rows = rows.to(model.dtype)
    from .glm5_next import pool_keys

    keys = pool_keys(layer, d, rows.reshape(T, kp, -1)[..., dr : dr + 2 * Di])  # [T, Di]
    layer.pool_key.index_put_((slots,), keys.reshape(T * kp, Di // kp).to(layer.pool_key.dtype))


# How the inplace pool keys are read (KILN_DSA_INPLACE_READ): "rows" gathers the context's cached
# rows as the per-call form does and keeps each pool's last row; "pools" first takes the key part of
# every pool's last row of the whole cache (a strided view, [slots / kpool, Di]) and gathers that by
# the block table, so the context's gates and other keys are never read.
INPLACE_READ = os.environ.get("KILN_DSA_INPLACE_READ", "pools")


def _inplace_keys(model, layer, d: DSASpec, table: torch.Tensor) -> torch.Tensor:
    """[(B,) L / kpool, Di]: the pool keys of the context a block table addresses (inplace form)."""
    kp, Di = d.kpool, d.head_dim
    dr = layer.spec.mla.qk_rope_head_dim
    ps = model.page_size
    v = layer.v_cache
    if INPLACE_READ == "rows":
        rows = model._load(v, table).squeeze(-2)  # [(B,) L, W]
        return rows.reshape(*rows.shape[:-2], -1, kp, rows.shape[-1])[..., kp - 1, dr : dr + Di]
    keys = v.view(-1, kp, v.shape[-1])[:, kp - 1, dr : dr + Di]  # [slots / kp, Di]
    pages = keys.reshape(-1, ps // kp, Di)[table]  # [(B,) P, ps / kp, Di]
    pages = pages.to(model.dtype) if model.fp8_max is not None else pages
    return pages.flatten(-3, -2)


def _inplace_rows(model, layer, d: DSASpec, positions: torch.Tensor, table: torch.Tensor,
                  vrow: torch.Tensor) -> torch.Tensor:
    """The rows write_pool_keys_inplace stores [T, dr + 2 Di], not stored (attention_joint)."""
    kp, Di = d.kpool, d.head_dim
    dr = layer.spec.mla.qk_rope_head_dim
    T = positions.shape[0]
    first = positions - positions % kp
    pos = first.unsqueeze(-1) + torch.arange(kp, device=positions.device, dtype=positions.dtype)
    rows = layer.v_cache[model._slots(pos, table).reshape(-1)]
    if model.fp8_max is not None:
        rows = rows.to(model.dtype)
    from .glm5_next import pool_keys

    keys = pool_keys(layer, d, rows.reshape(T, kp, -1)[..., dr : dr + 2 * Di])
    last = (positions % kp == kp - 1).unsqueeze(-1)
    key = torch.where(last, keys.to(vrow.dtype), vrow[:, dr : dr + Di])
    return torch.cat(([vrow[:, :dr]] if dr else []) + [key, vrow[:, dr + Di :]], dim=-1)


def write_pool_keys_inplace(model, layer, d: DSASpec, positions: torch.Tensor, table: torch.Tensor,
                            slot_mapping: torch.Tensor, vrow: torch.Tensor) -> None:
    """The inplace pool-key cache (pool_cache_mode): after this call's indexer rows vrow [T, dr + 2 Di]
    were stored, every token whose position is the last of its pool (position % kpool == kpool - 1)
    rewrites its own row with the pool's key (glm5_next.pool_keys of the pool's kpool cached rows) in
    place of its indexer key; the other tokens rewrite their rows unchanged. So a complete pool's last
    row holds [key | that token's gate logits], its other rows their own key and gates, and the
    selection reads the key part of every pool's last row. Only the last row is ever overwritten, so
    a pool can always be recomputed from its first kpool - 1 rows plus a fresh last row: a rejected
    speculative token at a pool's end, or a later write of an incomplete pool's tokens, finds the rows
    it needs. Padded rows (position 0) are no pool's last token. No extra bytes: the gate logits and
    keys of a complete pool are dead once its key exists, and the last row's key is one of them."""
    kp, Di = d.kpool, d.head_dim
    dr = layer.spec.mla.qk_rope_head_dim
    T = positions.shape[0]
    first = positions - positions % kp  # kpool is a power of two here (4): exact
    pos = first.unsqueeze(-1) + torch.arange(kp, device=positions.device, dtype=positions.dtype)  # [T, kp]
    rows = layer.v_cache[model._slots(pos, table).reshape(-1)]  # [T * kp, 1, dr + 2 Di]
    if model.fp8_max is not None:
        rows = rows.to(model.dtype)
    from .glm5_next import pool_keys

    keys = pool_keys(layer, d, rows.reshape(T, kp, -1)[..., dr : dr + 2 * Di])  # [T, Di]
    last = (positions % kp == kp - 1).unsqueeze(-1)  # [T, 1]
    key = torch.where(last, keys.to(vrow.dtype), vrow[:, dr : dr + Di])
    parts = ([vrow[:, :dr]] if dr else []) + [key, vrow[:, dr + Di :]]
    model._store(layer.v_cache, slot_mapping, torch.cat(parts, dim=-1).unsqueeze(1))


def _shared(layer, T: int, L: int, as_mask: bool = True) -> torch.Tensor:
    """The selection the last full layer left for T query rows over L keys (_share)."""
    if not as_mask:
        return layer.dsa_topk[:T].long()
    return layer.dsa_mask[:T, :L].float()


def _share(layer, top, T: int, as_mask: bool = True) -> None:
    """Write a selection to the scratch: the indices [B, Q, k] (gather) or the additive mask
    [B, Q, L] as whole scratch rows (mask), a one-row write as two rows (on trn1 a ONE-row
    index_put_ of a computed value is the slow path, docs/neuron-notes.md)."""
    rows = torch.arange(T, device=top.device)
    if not as_mask:
        layer.dsa_topk.index_put_((rows,), top.reshape(T, -1).to(torch.int32))
        return
    val = top.reshape(T, -1)
    keys = layer.dsa_mask.shape[1]
    if val.shape[1] < keys:
        val = torch.cat([val, torch.full((T, keys - val.shape[1]), NEG_INF, dtype=val.dtype, device=val.device)], 1)
    from .linear_attn import _write_rows

    _write_rows(layer.dsa_mask, rows, val.to(layer.dsa_mask.dtype))


def reference(model, layer, x, positions, state: dict):
    """The attention block over a whole sequence without a cache (forward_logits, tests):
    decompressed keys and values, the indexer over every earlier token. state carries the
    last full DSA layer's top-k to the shared layers after it."""
    m = layer.spec.mla
    d = m.dsa
    T = x.shape[0]
    q_nope, q_pe, c, k_pe, q_resid = _project(model, layer, x, positions)
    j = positions.unsqueeze(0)
    vis = torch.where(j <= positions.unsqueeze(1), 0.0, NEG_INF).view(1, T, T)
    index = None
    if d is not None and d.indexer:
        qi, ki, wi = _indexer(model, layer, x, q_resid, positions)
        index = (qi, wi, ki.unsqueeze(0))
    out, top = _core(model, layer, q_nope, q_pe, c.unsqueeze(0), k_pe.unsqueeze(0), vis, 1, T,
                     state.get("top") if d is not None and not d.indexer else None, index, absorb=False)
    if d is not None and d.indexer:
        state["top"] = top
    return F.linear(out, model._w(layer, "o"))
