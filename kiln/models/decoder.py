"""Decoder-only transformers (Llama, Mistral, Qwen2, Qwen3, Qwen3-MoE, MiMo-V2, gpt-oss, Hy3,
K2-Horizon, Inkling, and the MLA / DSA families of models/mla.py) for static-shape execution over
a paged KV cache.

Numerics follow Hugging Face transformers' modeling code so the CPU path can be checked
against it directly: RMSNorm in fp32 then cast back, optional per-head QK-norm before RoPE
(Qwen3), optional QKV bias (Qwen2), rotate-half RoPE from an fp32 table (Llama-3 frequency
scaling, partial rotary dims), softmax in fp32, MoE routing in fp32.

Attention is described PER LAYER by an AttnSpec, because hybrid models mix layer kinds:
MiMo-V2 alternates full-attention layers (4 KV heads, RoPE theta 1e7) with sliding-window
layers (8 KV heads, window 128, theta 1e4, per-head attention-sink logits), all with query/
key head dim 192 and value head dim 128. Uniform models get one spec repeated.

Graph entry points, each compiled once per bucket:
- `forward_decode`: B sequences with one query token each, context up to P pages.
- `forward_prefill`: one sequence's chunk of C tokens, with up to P pages of context
  (cached prefix included), so a radix-cache hit skips straight to the uncached tail.
- `forward_extend`: B sequences x Q tokens with every position scored (speculative verify).
All write new KV into the paged cache in place and return packed sampler output, so a step
reads back a few floats per sequence instead of a vocab-sized row.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from ..config import AttnSpec, LinearSpec, ModelConfig
from ..engine.sampler import NUM_TOP_LOGPROBS, logsumexp_large, sample, score_rows, topk_large, verify_sample
from . import linear_attn
from .quant import FP8, dequant, dequant_mxfp4, dequant_t

NEG_INF = -1e30
# KV gather granularity: "auto", "page" or "token". Measured on trn1.2xlarge, B=6 decode
# (tools/profile_decode.py --gather, 2026-10-02): token 8.80 / 14.59 / 21.99 ms and page
# 10.65 / 21.57 / 17.39 ms at 128 / 512 / 2048 context. neuronx-cc lowers the two
# differently per shape, so "auto" uses pages only from PAGE_GATHER_MIN_TOKENS up.
GATHER = os.environ.get("KILN_GATHER", "auto")
PAGE_GATHER_MIN_TOKENS = 2048


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def llama3_inv_freq(inv_freq: torch.Tensor, rs: dict) -> torch.Tensor:
    """transformers modeling_rope_utils._compute_llama3_parameters."""
    factor, lo, hi = rs["factor"], rs["low_freq_factor"], rs["high_freq_factor"]
    old = rs["original_max_position_embeddings"]
    lo_wl, hi_wl = old / lo, old / hi
    wavelen = 2 * math.pi / inv_freq
    out = torch.where(wavelen > lo_wl, inv_freq / factor, inv_freq)
    smooth = (old / wavelen - lo) / (hi - lo)
    smoothed = (1 - smooth) * out / factor + smooth * out
    is_medium = ~(wavelen < hi_wl) & ~(wavelen > lo_wl)
    return torch.where(is_medium, smoothed, out)


# gpt-oss's expert activation, transformers 5.15 models/gpt_oss/modeling_gpt_oss.py
# GptOssExperts._apply_gate (alpha = 1.702, limit = config swiglu_limit): the gate is clamped from
# above, up to [-limit, limit], and the output is (up + 1) * gate * sigmoid(alpha * gate).
SWIGLU_OAI_ALPHA = 1.702


def swiglu_oai(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return (up + 1) * (gate * torch.sigmoid(gate * SWIGLU_OAI_ALPHA))


def moe_inter_per_rank(cfg: ModelConfig, tp: int, quantized: bool) -> int:
    """Expert intermediate size on one tensor-parallel rank. A quantized expert's down_proj is
    sharded on its input (intermediate) dim, whose scale blocks a shard must not split. gpt-oss's
    2880 is 90 MXFP4 blocks of 32, which split into whole blocks per rank only for tp in
    {1, 2, 3, 5, 6, 9, 10, ...}, so at tp=8 or 32 every rank takes ceil(90 / tp) whole blocks and
    the last ones are zero-padded (zero gate/up rows and zero down columns add exactly 0)."""
    I = cfg.moe_intermediate_size
    if not cfg.num_experts:
        return 0
    # Only MXFP4 experts are padded: a block-FP8 shard that sits inside one scale block can take
    # that block's scale (the loader's concern), padding it would double its bytes.
    b = cfg.quant_expert_block if quantized and cfg.quant_expert_mxfp4 and cfg.quant_expert_block else 1
    if I % tp == 0 and (I // tp) % b == 0:
        return I // tp
    if b == 1:
        raise ValueError(f"tp={tp} does not divide moe_intermediate_size {I}")
    return -(-I // (tp * b)) * b


def kv_heads_per_rank(num_kv_heads: int, tp: int) -> int:
    """KV heads split across ranks, or, with more ranks than KV heads, replicated so each
    rank holds the ONE head its query heads attend to."""
    return num_kv_heads // tp if tp <= num_kv_heads else 1


def layer_specs(cfg: ModelConfig) -> tuple[AttnSpec, ...]:
    if cfg.attn_layers:
        return cfg.attn_layers
    D = cfg.head_dim
    return (AttnSpec(cfg.num_heads, cfg.num_kv_heads, D, D, D, cfg.rope_theta),) * cfg.num_layers


def heads_fit(spec, atp: int) -> bool:
    """Whether a token mixer's heads split over `atp` attention ranks: linear attention by k and
    v heads, MLA by query heads (its latent is replicated), GQA by query heads with the KV heads
    split or, with more ranks than KV heads, replicated (each rank's query heads under one KV head)."""
    if isinstance(spec, LinearSpec):
        return spec.num_k_heads % atp == 0 and spec.num_v_heads % atp == 0
    if spec.num_heads % atp:
        return False
    if spec.mla is not None:
        return True
    Hkv = spec.num_kv_heads
    return Hkv % atp == 0 if atp <= Hkv else atp % Hkv == 0


def attention_tp(cfg: ModelConfig, tp: int, requested: int | None = None, mtp: bool = False) -> int:
    """The tensor-parallel degree of the token mixers (DecoderForCausalLM: attention TP): `requested`
    when given (it must divide tp and fit every mixer's heads), else the largest divisor of tp that
    every mixer of the model (and the MTP layer's, with mtp) can split. A model whose heads divide
    tp gets tp, which is plain tensor parallelism."""
    specs = list(layer_specs(cfg)) + ([cfg.mtp_spec] if mtp and cfg.mtp_spec is not None else [])
    if requested is not None:
        bad = [s for s in specs if not heads_fit(s, requested)]
        if requested < 1 or tp % requested or bad:
            why = f"does not divide tp={tp}" if requested < 1 or tp % requested else f"does not fit {_heads(bad[0])}"
            raise ValueError(f"attention_tp={requested} {why}")
        return requested
    return max(a for a in range(1, tp + 1) if tp % a == 0 and all(heads_fit(s, a) for s in specs))


def _heads(spec) -> str:
    if isinstance(spec, LinearSpec):
        return f"{spec.num_k_heads} k / {spec.num_v_heads} v linear-attention heads"
    if spec.mla is not None:
        return f"{spec.num_heads} MLA query heads"
    return f"{spec.num_heads} query / {spec.num_kv_heads} KV heads"


def kv_bytes_per_token_rank(cfg: ModelConfig, tp: int, dtype: torch.dtype, attn_tp: int | None = None) -> int:
    """KV bytes one token occupies on ONE tensor-parallel rank, over all layers (KV heads follow
    the attention TP, attention_tp's default unless attn_tp is given)."""
    b = torch.finfo(dtype).bits // 8
    atp = attn_tp or attention_tp(cfg, tp)
    return sum(kv_heads_per_rank(s.num_kv_heads, atp) * (s.head_dim + s.v_head_dim) * b for s in layer_specs(cfg)
               if not isinstance(s, LinearSpec))


def state_bytes_per_token_rank(cfg: ModelConfig, atp: int, dtype: torch.dtype, kv_fp8: bool = False) -> int:
    """Token-slot state bytes one token occupies on ONE rank (DecoderForCausalLM.token_state_shapes):
    Inkling's four short-convolution inputs per layer, k and v sharded like the KV heads (over the
    attention TP, atp) and the attention and MLP outputs replicated (hidden-wide); a pooled DSA
    indexer's separate pool-key cache (models/mla.py pool_key_width), counted for the layers whose KV
    kv_bytes_per_token counts (not the MTP layer's, as for its KV)."""
    b = torch.finfo(dtype).bits // 8
    pool = sum(_mla.pool_key_width(s, kv_fp8) for s in layer_specs(cfg)) * b
    if not cfg.sconv_kernel:
        return pool
    return pool + sum((kv_heads_per_rank(s.num_kv_heads, atp) * (s.head_dim + s.v_head_dim) + 2 * cfg.hidden_size) * b
                      for s in layer_specs(cfg))


# The NKI MoE kernels and the expert blob layout each packs (DecoderLayer.pack_experts): "nki"
# (= "nki-dedupe") is kernels/moe_dedupe.py (tile scales; one load per distinct expert of a call,
# per pair below 16 tokens), "nki-pair" kernels/moe_decode.py (block-32 scales; one load per
# (token, expert) pair). "nki" measured faster or equal at every shape tried on MiMo-V2.6-Flash's
# tp=32 rank shapes (decode B=1..128, prefill C=32 and 128, the 48-layer step at B=1, 4, 16, 32;
# trn1.2xlarge, docs/neuron-notes.md "MoE that reads each selected expert once per call").
NKI_MOE_KERNELS = {"nki": "tiles", "nki-dedupe": "tiles", "nki-pair": "pair"}
# Prefill chunks of a tiles-blob layer (KILN_MOE_KERNEL=nki) from this many tokens on run
# kernels/moe_prefill.py (a grouped GEMM over the whole chunk) instead of moe_dedupe's 128-token
# calls, when KILN_MOE_PREFILL_KERNEL=nki (default xla: off). Measured on GLM-5.3-Flash's tp=32 rank
# shapes, trn1.2xlarge (docs/neuron-notes.md "Prefill MoE as a grouped GEMM").
MOE_PREFILL_KERNEL = os.environ.get("KILN_MOE_PREFILL_KERNEL", "xla")
MOE_PREFILL_MIN_TOKENS = int(os.environ.get("KILN_MOE_PREFILL_MIN_TOKENS", 512))


def moe_ep_enabled(cfg=None) -> bool:
    """KILN_MOE_EP: expert parallelism for the routed experts (kernels/moe_ep.py). Every tensor-parallel rank
    holds E / tp WHOLE experts (rank r: experts r E / tp .. (r + 1) E / tp - 1, every intermediate row) instead of
    a 1 / tp slice of each, and its routed output is the sum over the pairs routed to its own experts only; the
    MoE block's existing all-reduce (or, with sequence-parallel streams, reduce-scatter) adds the ranks' outputs
    (every rank already holds every row of the MoE input: DP attention runs the MLP over all groups' rows,
    sequence-parallel prefill streams gather them), so the layout moves no extra byte between ranks. The shared
    experts and dense MLPs stay tensor-parallel. On the device the layers run kernels/moe_ep.py (KILN_MOE_KERNEL=nki
    packs them for it) at every batch shape, decode included; the host path computes the rank's experts over every
    row with the routing weights of their pairs. Read when a model is built, so a test can set it per engine.

    KILN_MOE_EP=1 / 0 forces it on / off. Unset or "auto": on where it was measured, i.e. for GLM-5.3-Flash's
    family (glm5_next) on trn1 / trn1n / trn2 and on a host without a Neuron device (the CPU paths, where EP equals TP:
    tests/test_moe_ep.py), off for every other model and platform until measured there. trn2 since 2026-10-06, after
    the LNC=2 scatter fix (kernels/moe_ep.py _dummy_lanes): kiln-t2-cb2 (trn2.48xlarge, SDK 2.32, real weights, tp=32,
    DP attention 4, one engine per half of the box, both at once), EP with the loader's tile-scale fit against the trn2
    defaults (TP experts) at conc 32 / 64 / 128: 141.2 / 158.2 / 170.4 against 102.7 / 113.1 / 120.1 out tok/s (+37.5 /
    +39.9 / +41.9%), prefill call 0.588-0.603 against 0.902-0.914 s, decode call 56 / 74 / 99 against 59 / 77 / 96 ms; wikitext
    -0.5495 against -0.5498 (|dlogprob| mean 0.057, greedy agreement 97.4%); docs/neuron-notes.md "Expert parallelism on
    trn2 after the fix". The measurement: kiln-mimo-trn1
    (trn1.32xlarge, SDK 2.32, real weights, tp=32), the G64-4096-KV1.5-S20-P12-K serve_sweep at conc 64 on engine-v0
    b4f400f, 2026-10-04: TP 94.9 -> EP 106.4 out tok/s, TTFT p50 10.1 -> 8.4 s, ITL p50 604.5 -> 528.7 ms (logs s3
    logs/kiln-mimo-trn1/20261004T185347Z-serve_sweep.log, 20261004T190618Z-ep-b4f4-serve_sweep.log); real-weight ppl
    -2.074 (TP -2.073), wikitext -0.552 (-0.551); docs/neuron-notes.md "Expert parallelism".

    The model also leaves the automatic default off below ep_auto_min_decode_rows() decode rows per DP-attention
    group (max_num_seqs / dp_attention; DecoderForCausalLM), where the small-lane decode kernel's per-expert
    passes cost more than TP's moe_dedupe. With kiln_moe_ep_small (KILN_MOE_EP_SMALL_V=1) that is 8: measured on
    kiln-mimo-trn1 with the q/final-c33a391 configs (feat/moe-ep 24ed0fa, 2026-10-04), G16 (16 seqs over 4 groups = 4
    rows) EP 80.5 against TP 82.9 out tok/s, ITL 168.8 against 156 ms (log s3
    logs/kiln-mimo-trn1/20261004T204415Z-ep24-G16-sweep.log), F0 (32 over 4 = 8 rows) EP 98.9 against TP 90.8, ITL
    287.5 against 311 ms (20261004T203230Z-ep24-F0-sweep.log). With kiln_moe_ep_small2 (the default from 2026-10-05,
    no pass for an expert without pairs) it is 4: kiln-ut-32 (trn1.32xlarge, SDK 2.32, real weights, conc 16, 4 decode
    rows per group, 128 requests, same box back to back, every arm at --max-num-seqs 32 --kv-cache-gb 1.5
    --decode-buckets 4), TP 86.9 out tok/s, EP v1 88.4 (+1.7%), EP v2 98.0 (+12.8%; decode call 0.066 -> 0.084 s,
    prefill call 0.895 -> 0.592 s against TP; logs s3 logs/kiln-ut-32/ut-g16t32.log, ut-g16e.log, ut-g16e-v2.log with
    their .log.cmd). The final G16 config (--max-num-seqs 16 --kv-cache-gb 0.65, TP) measured 87.5 on the same box."""
    v = os.environ.get("KILN_MOE_EP", "auto")
    if v in ("0", "1"):
        return v == "1"
    if v != "auto":
        raise ValueError(f"KILN_MOE_EP must be 0, 1 or auto, not {v!r}")
    hy = getattr(cfg, "hybrid", None)
    if hy is None or hy.family not in EP_AUTO_MODEL_FAMILIES:
        return False
    from .. import platform

    t = platform.target()
    return t is None or platform.family_of(t) in EP_AUTO_PLATFORMS


EP_AUTO_MODEL_FAMILIES = ("glm5_next",)
EP_AUTO_PLATFORMS = ("trn1", "trn1n", "trn2")
EP_AUTO_MIN_DECODE_ROWS = 4  # decode rows per DP-attention group with kiln_moe_ep_small2 (moe_ep_enabled)
EP_AUTO_MIN_DECODE_ROWS_V1 = 8  # ... with kiln_moe_ep_small (KILN_MOE_EP_SMALL_V=1)


def ep_auto_min_decode_rows() -> int:
    """The automatic expert-parallel default's least decode rows per DP-attention group (moe_ep_enabled): it follows
    the small-lane decode kernel in use (kernels/moe_ep.py SMALL_V)."""
    from ..kernels import moe_ep

    return EP_AUTO_MIN_DECODE_ROWS_V1 if moe_ep.SMALL_V == 1 else EP_AUTO_MIN_DECODE_ROWS


# KILN_TOPK_CLAMP=1 (opt-in): the sigmoid router's top-k indices clamped into [0, E) before the routing weights are gathered.
# On trn2 (SDK 2.32, LNC=2) torch.topk over a row holding NaN returns 0xFFFFFFFF for every index (tools/probe_topk_nan.py:
# NaN rows -> [4294967295] x 8; -inf rows, all-tied rows and zero logits give in-range indices), and the XLA gather at them
# faults ("scatter/gather (indirect memory copy via vector DGE) out-of-bound access", nrta 1006): the trn2 long-prompt engine's
# bucket warmup, whose all-zero inputs give NaN rows, died on it (docs/neuron-notes.md "trn2 top-k of a NaN row"). Clamping
# leaves every finite row's indices as they were; a NaN row's routing is garbage either way. Off by default: it changes the
# router's graph, so turning it on re-keys every MoE graph.
TOPK_CLAMP = os.environ.get("KILN_TOPK_CLAMP", "0") == "1"


# KILN_DENSE_FP8 (default "1"): a checkpoint's FP8 weights outside the routed experts (attention projections,
# dense and shared-expert MLPs) stay FP8 with fp32 block scales and are dequantized inside every graph right
# before their matmul (models/quant.py dequant); "0" dequantizes them once at load into the model dtype, the
# same values (dequant computes w * scale in fp32 and rounds to the model dtype either way) at twice their
# FP8 bytes. The routed experts stay FP8 (the NKI kernels read the FP8 blob). Measured in a GLM-5.3-Flash
# decode group graph (tp=32, DP attention 4, 16 rows per group, trn1.32xlarge, neuron-explorer replay of
# 0140553b, 2026-10-04): each pooled DSA layer's q_a + kv_a, q_b and o_proj dequantize f32 [2048, 4096],
# [2048, 1536] and [4096, 2048] per step, 0.40 ms of the layer's 4.1 ms (docs/neuron-notes.md "Where a
# GLM-5.3-Flash decode step goes, from a device profile").
DENSE_FP8 = os.environ.get("KILN_DENSE_FP8", "1") == "1"


# Sequence-parallel residual streams for hyper-connection prefill (KILN_PREFILL_SP; default: where proven):
# between the layers of a prefill chunk every tensor-parallel rank holds only its own 1 / tp of the
# rows of the [R, hc * H] streams instead of all of them, so the hyper-connection arithmetic
# (models/hybrid.py _mhc and mix_out, fp32 over [R, hc, H]: replicated on every rank otherwise) runs
# on R / tp rows. Each block's normalised input is gathered to every rank by a zero-padded world
# all-reduce (_sp_gather; all_gather_into_tensor NEFFs do not reload in a later process, see _head),
# the block runs as before over all R rows, and each rank keeps its rows of the reduced output
# (_sp_rows). Measured on GLM-5.3-Flash at the sweep's prefill shape (tp=32, DP attention 4, 1024
# rows, trn1.32xlarge, 2026-10-04, docs/neuron-notes.md "Where a GLM-5.3-Flash prefill step goes
# on trn1"): the collapse and the output mix were 0.95 and 1.8 ms per block, twice per layer. Same
# per-row arithmetic either way; decode, verify and MTP keep the replicated streams. Read when a model
# is built (prefill_sp), so a test can set it per engine.
#
# KILN_PREFILL_SP=1 / 0 turns them on / off; unset, they are on where real-weight ppl says the device
# still rounds the streams with them: the trn1 families (trn1.32xlarge, GLM-5.3-Flash tp=32, 4 sentences
# -2.073 with them against -1.980 without, CPU bf16 reference -2.098; wikitext-2 -0.548 against -0.552),
# and a host without a Neuron device (the CPU paths, tests/test_dp_attention.py); and trn2: on
# trn2.48xlarge (tp=32, LNC=2, 2026-10-04, engine-v0 21e9c8c) tools/probe_mhc_rounding.py finds the streams'
# Veltkamp rounding bit-identical to the host's at 1 to 1024 rows, and the wikitext-2 slice scores -0.549 with
# them against -0.552 without (|dlogprob| mean 0.058, greedy agreement 97.4%, tools/compare_ppl.py). The
# 4-sentence set had moved to -1.807 with them on an older tree (-2.100 without), all of it "Water boils",
# the sentence that flips under small perturbations (docs/neuron-notes.md "knife-edge"). Off on trn3 and
# inf2, where nothing was measured.
PREFILL_SP_FAMILIES = ("trn1", "trn1n", "trn2")


# KILN_SP_GROUP (auto | 1 | 0): with sequence-parallel streams under DP attention a token mixer block gathers only
# its attention group's rows (the group's ranks hold exactly them) over the group and reduce-scatters its output
# over the group (models/hybrid.py _attn_rows), instead of a world gather of every row and a world reduce-scatter of
# the zero-padded DP batch: a quarter of the bytes at DP 4, inside four chips. A subgroup collective's replica groups
# enter the compile key, so each attention group compiles its own NEFF of the prefill pieces (each rank loads only
# its group's). auto (the default): on where measured, the trn1 families (and hosts without a Neuron target, the
# CPU tests); 1 / 0 force it. Read when a model is built (DecoderForCausalLM.sp_group): the platform lookup reads
# files, which a traced graph may not.
# trn2 (2026-10-05, trn2.48xlarge, feat/trn2-fast): with the MoE prefill skip, one engine at conc 32 88.6 -> 94.8 out tok/s, the
# 4096-row prefill call 1.017 -> 0.922 s; wikitext-2 with the decode set -0.554 -> -0.544, greedy agreement 97.8%.
SP_GROUP_FAMILIES = ("trn1", "trn1n", "trn2")


def sp_group_enabled() -> bool:
    v = os.environ.get("KILN_SP_GROUP", "auto")
    if v in ("0", "1"):
        return v == "1"
    from .. import platform

    t = platform.target()
    return t is None or platform.family_of(t) in SP_GROUP_FAMILIES


# KILN_PLP_VP=1 (opt-in): prompt logprobs scored on each rank's vocabulary shard (DecoderForCausalLM._score_rows) when
# the lm_head is vocab-parallel, instead of gathering every row's logits over the whole vocabulary first. The gathered
# form makes the 4096-row prompt-logprob post graph of GLM-5.3-Flash at tp 32 [4096, 154,880] fp32 plus its temporaries:
# a 4.76 GiB scratchpad (tools/hbm_estimate.py, against 0.75 for the prefill pieces), which does not load next to the
# long-context engines' KV on trn1 ("Allocation Failure", 2026-10-06). The same rank, target logprob (its log-sum-exp
# is combined from the shards' own, so it may differ from the gathered form in the last fp32 bits) and top-N.
PLP_VP = os.environ.get("KILN_PLP_VP", "0") == "1"


def prefill_sp_enabled() -> bool:
    v = os.environ.get("KILN_PREFILL_SP")
    if v is not None:
        return v == "1"
    from .. import platform

    t = platform.target()
    return t is None or platform.family_of(t) in PREFILL_SP_FAMILIES


# KILN_DECODE_SP (default "0"): the same sequence-parallel streams for decode steps: each rank keeps N B / tp of a decode
# call's N B rows (N DP-attention groups of B) between layers, so the hyper-connection arithmetic, replicated over
# every row on every rank otherwise, runs on N B / tp rows; each block gathers the rows it needs and reduce-scatters its
# output (two collectives per block instead of one all-reduce), and the post graph gathers the final hidden state for
# the lm_head. Measured in a decode layer-group graph (tp=32, DP attention 4, 16 rows per group, trn1.32xlarge,
# neuron-explorer replay, 2026-10-04): the mix-out plus hyper-connection part of each block ~0.2 ms over 64 replicated
# rows, ~0.4 ms per layer. Only decode calls whose rows divide over tp (ModelRunner turns it off otherwise); not with
# an MTP head. Read when a model is built.
DECODE_SP_FAMILIES = ("trn1", "trn2")  # where the A/Bs with the decode kernels ran (kernels/kda_decode.py)


def decode_sp_enabled() -> bool:
    """KILN_DECODE_SP unset: on for a trn1 / trn2 target, off elsewhere; "1" / "0" force it."""
    v = os.environ.get("KILN_DECODE_SP")
    if v:
        return v == "1"
    from .. import platform

    t = platform.target()
    return t is not None and platform.family_of(t) in DECODE_SP_FAMILIES


def moe_prefill_down(layer):
    """The prefill kernel's per-column down scales of a layer (DecoderLayer.pack_experts), or None."""
    dsc = getattr(layer, "moe_prefill_dsc", None)
    return None if dsc is None else (dsc, layer.moe_prefill_dfr)


def mixed_chunk_rows(x: torch.Tensor, mixed) -> int:
    """Rows of a mixed batch's prefill chunk in x (one group's rows: the chunk's, then the decode
    rows'; DecoderForCausalLM.forward_mixed)."""
    return x.shape[0] - mixed[0].shape[0]


# KILN_MIXED_DECODE_SPLIT=s: inside a mixed graph the token mixers (KDA, MLA) take the decode rows s at a
# time (each slice its own decode-form call), not all D at once; unset, one call over all of them. A
# probe of how the size of the decode part drives the compiler's spills in a mixed group graph: at
# GLM-5.3-Flash's G64 shapes 16 decode rows per group took the 12-layer groups from 3.2-3.3M to 4.8-4.9M
# queue-instance spill runs and the first one no longer loaded on trn1, while 8 rows (F0) left them
# where the prefill groups are (docs/neuron-notes.md "Mixed batches").
MIXED_DECODE_SPLIT = int(os.environ.get("KILN_MIXED_DECODE_SPLIT", 0))
# KILN_MIXED_MIXERS: how a mixed graph's token mixers treat the decode rows. "joint" (default): every
# projection over the chunk's and the decode rows at once (each weight read once per call), only what is
# per sequence (KV and state reads and writes, the conv, the recurrence, the selection, the attention) per
# part, and each cache or state pool written once (linear_attn._mix_joint, mla.attention_joint). "calls":
# the chunk form and the decode form as two calls on the rows (each projection twice). Measured on
# GLM-5.3-Flash at tp=32, DP 4, trn1.32xlarge (2026-10-04, docs/neuron-notes.md "Mixed batches"): at the G64
# sweep shapes (1024 + 16 rows per group) a "joint" mixed call took 1198.5 ms against 1088.1 for the prefill
# call and 160.2 for the decode call it replaces, and the sweep ran 90.4 out tok/s against 88.3 unmixed;
# "calls" ran 84.8 (at 1024 + 8 rows: 1285 ms against 1090 + 110).
MIXED_MIXERS = os.environ.get("KILN_MIXED_MIXERS", "joint")
if MIXED_MIXERS not in ("joint", "calls"):
    raise ValueError(f"KILN_MIXED_MIXERS must be joint or calls, not {MIXED_MIXERS!r}")


def mixed_decode_slices(mixed) -> list[tuple[int, int]]:
    """(start, stop) of each decode-row slice a token mixer takes in a mixed graph (MIXED_DECODE_SPLIT)."""
    D = mixed[0].shape[0]
    s = MIXED_DECODE_SPLIT or D
    return [(i, min(i + s, D)) for i in range(0, D, s)]


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, spec: AttnSpec, dtype: torch.dtype, tp: int, tp_rank: int, moe: bool,
                 index: int = 0, keep_fp8: bool = False, packed_mxfp4: bool = False, prefix: str | None = None,
                 moe_kernel: str = "xla", attn_tp: int | None = None, attn_rank: int | None = None,
                 plain: bool = False, ep: bool = False, ep_extra: list[int] | None = None,
                 qkv_in: int | None = None):
        """qkv_in: the q / k / v projections' input width when it is not the hidden size (an EAGLE-3 draft
        layer's cat(embeds, hidden), models/eagle3.py). tp / tp_rank shard the MLP (dense or experts); ep: the routed experts are expert-parallel instead
        (moe_ep_enabled: this rank holds experts tp_rank E / tp .. whole); ep_extra: with ep, the experts the
        redundant slots of EVERY rank hold, rank-major (models/eplb.py: s = len / tp slots per rank, this rank's
        after its own E / tp; None or empty: none); attn_tp / attn_rank (default: the same)
        shard the token mixer's heads (DecoderForCausalLM: attention TP).
        keep_fp8: weights the checkpoint quantizes stay FP8 (with fp32 block scales, see
        models/quant.py) and are dequantized inside the graph right before their matmul.
        packed_mxfp4: MXFP4 experts stay 4-bit (half the bytes of FP8) instead of converting
        losslessly to FP8; the in-graph decode costs about 11 ms per 32 experts on trn1.
        moe_kernel "nki" / "nki-pair": the experts are repacked into w_blob after loading
        (pack_experts) for kernels/moe_dedupe.py / kernels/moe_decode.py (NKI_MOE_KERNELS)."""
        super().__init__()
        self.spec = spec
        H = cfg.hidden_size

        def p(*shape):
            return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)

        def lin(name: str, module: str, *shape: int, expert: bool = False, transposed: bool = False):
            """A weight [..., out, in], FP8 + scale when the checkpoint quantizes it.
            transposed: stored [..., in, out] (scale [..., in / bk, out]); see w_down below."""
            quant = (keep_fp8 and (expert or DENSE_FP8)
                     and cfg.is_quantized(f"{prefix or f'model.layers.{index}'}.{module}", expert))
            if quant and expert and cfg.quant_expert_mxfp4 and packed_mxfp4:
                # Packed MXFP4, decoded in-graph (models/quant.py); scales are powers of two,
                # exact in bf16.
                setattr(self, name, nn.Parameter(torch.empty(*shape[:-1], shape[-1] // 2, dtype=torch.uint8),
                                                 requires_grad=False))
                setattr(self, name + "_scale", nn.Parameter(
                    torch.empty(*shape[:-1], shape[-1] // 32, dtype=torch.bfloat16), requires_grad=False))
            elif quant:
                bk = cfg.quant_expert_block if expert else cfg.quant_block
                # MXFP4-derived scales are powers of two: exact in bf16 at half the bytes.
                sdt = torch.bfloat16 if expert and cfg.quant_expert_mxfp4 else torch.float32
                *lead, n_out, n_in = shape
                nb = -(-n_in // bk)
                setattr(self, name, nn.Parameter(torch.empty(*lead, *((n_in, n_out) if transposed else (n_out, n_in)),
                                                             dtype=FP8), requires_grad=False))
                setattr(self, name + "_scale", nn.Parameter(
                    torch.empty(*lead, *((nb, n_out) if transposed else (n_out, nb)), dtype=sdt), requires_grad=False))
            else:
                *lead, n_out, n_in = shape
                setattr(self, name, p(*lead, *((n_in, n_out) if transposed else (n_out, n_in))))
                setattr(self, name + "_scale", None)

        atp = tp if attn_tp is None else attn_tp
        arank = tp_rank if attn_rank is None else attn_rank
        if not heads_fit(spec, atp):
            raise ValueError(f"attention tp={atp} does not fit {_heads(spec)}")
        self.in_norm = p(H)
        self.o_bias = self.rel_q = self.rel_q_scale = self.rel_proj = None
        self.k_sconv = self.v_sconv = self.a_sconv = self.m_sconv = None
        if isinstance(spec, LinearSpec):  # recurrent-state token mixer (models/linear_attn.py)
            linear_attn.init_mixer(self, cfg, spec, atp, p, lin)
        elif spec.mla is not None:  # multi-head latent attention (models/mla.py)
            _mla.init_layer(self, cfg, spec, p, lin, atp)
            self.o_gate = self.o_gate_scale = None
        else:
            Hkv = spec.num_kv_heads
            self.nh = spec.num_heads // atp
            self.nkv = kv_heads_per_rank(Hkv, atp)
            # First KV head on this rank. Replicated (atp > Hkv): attention rank r's query heads
            # [r * nh, (r + 1) * nh) all map to KV head r * Hkv // atp under GQA.
            self.kv_offset = arank * self.nkv if atp <= Hkv else arank * Hkv // atp
            Dk, Dv = spec.head_dim, spec.v_head_dim
            q, k, v = self.nh * Dk, self.nkv * Dk, self.nkv * Dv
            self.split = (q, k, v)
            qkv_module = "self_attn.qkv_proj" if cfg.fused_qkv_weights else "self_attn.q_proj"
            lin("qkv", qkv_module, q + k + v, qkv_in or H)  # q_proj, k_proj, v_proj fused on the output dim
            self.qkv_bias = p(q + k + v) if cfg.qkv_bias else None
            self.q_norm = p(Dk) if cfg.qk_norm else None
            self.k_norm = p(Dk) if cfg.qk_norm else None
            self.sink = p(self.nh) if spec.sink else None  # per-query-head attention-sink logit
            lin("o", "self_attn.o_proj", H, self.nh * Dv)
            # Output gate (Qwen3.5 attn_output_gate): attention output * sigmoid(x W_gate^T)
            # before o_proj, W_gate being the gate half of each head's q_proj rows.
            if cfg.attn_output_gate:
                lin("o_gate", "self_attn.q_proj", q, H)
            else:
                self.o_gate = self.o_gate_scale = None
            self.o_bias = p(H) if cfg.o_bias else None  # gpt-oss: rank 0's copy is real, the others zero
            # Inkling: relative position logits (r_proj sharded with the query heads, the profile
            # bank replicated) and causal short convolutions, fp32 weights [channels, kernel] as
            # transformers keeps them (_keep_in_fp32_modules_strict), over token-slot histories.
            if spec.rel_extent:
                lin("rel_q", "self_attn.r_proj", self.nh * cfg.d_rel, H)
                self.rel_proj = p(cfg.d_rel, spec.rel_extent)
            if cfg.sconv_kernel:
                def conv(c):
                    return nn.Parameter(torch.empty(c, cfg.sconv_kernel, dtype=torch.float32), requires_grad=False)

                self.k_sconv, self.v_sconv, self.a_sconv, self.m_sconv = conv(k), conv(v), conv(H), conv(H)
        self.k_hist = self.v_hist = self.a_hist = self.m_hist = None  # [slots, channels], bound by the runner
        self.post_norm = p(H)
        self.moe = moe
        self.moe_blob = False  # experts packed into w_blob (pack_experts)
        self.moe_tiles = False  # ... in kernels/moe_dedupe.py's tile-scale layout
        self.pack_moe = False  # the loader packs them (an NKI moe_kernel)
        self.pack_tiles = False  # ... into the tile-scale layout
        self.moe_ep = bool(ep and moe)  # whole experts per rank (moe_ep_enabled), packed for kernels/moe_ep.py
        if moe:
            E = cfg.num_experts
            if self.moe_ep:
                if E % tp:
                    raise ValueError(f"KILN_MOE_EP=1: tp={tp} does not divide the {E} routed experts")
                if cfg.expert_bias or cfg.moe_act != "silu" or cfg.router_shared_rows:
                    raise NotImplementedError(f"KILN_MOE_EP=1: {cfg.architecture}'s experts (biases, moe_act "
                                              f"{cfg.moe_act}, router-weighted shared experts) are not implemented")
                El, Im = E // tp, cfg.moe_intermediate_size
                # models/eplb.py: s redundant slots per rank after the El primaries (physical ids E + r s + j).
                extra = list(ep_extra or [])
                if len(extra) % tp:
                    raise ValueError(f"{len(extra)} redundant expert slots do not divide over tp={tp}")
                self.ep_s = len(extra) // tp
                self.ep_extra_all = extra
                # Logical expert held by each local slot: the primaries, then this rank's redundant slots.
                self.ep_experts = list(range(tp_rank * El, (tp_rank + 1) * El)) + extra[tp_rank * self.ep_s:
                                                                                         (tp_rank + 1) * self.ep_s]
                self.ep_first, self.ep_count = tp_rank * El, El + self.ep_s  # first primary, local slots
                El = El + self.ep_s
            else:
                El, Im = E, moe_inter_per_rank(cfg, tp, keep_fp8)
            # DeepSeek-V3-style shared experts: a dense MLP beside the routed experts
            # (transformers DeepseekV3MoE.shared_experts, intermediate moe_intermediate_size *
            # n_shared_experts), sharded like the dense MLP; also Hy3's, K2-Horizon's and Inkling's.
            if (cfg.moe_intermediate_size * cfg.n_shared_experts) % tp:
                raise ValueError(f"tp={tp} does not divide the shared experts' intermediate size")
            Is = cfg.moe_intermediate_size * cfg.n_shared_experts // tp
            if Is:
                lin("shared_gate_up", "mlp.shared_experts.gate_proj", 2 * Is, H)
                lin("shared_down", "mlp.shared_experts.down_proj", H, Is)
            else:
                self.shared_gate_up = self.shared_gate_up_scale = self.shared_down = self.shared_down_scale = None
            self.router = p(E + cfg.router_shared_rows, H)  # Inkling scores its shared experts too
            self.router_scale = (nn.Parameter(torch.empty(1, dtype=torch.float32), requires_grad=False)
                                 if cfg.router_global_scale else None)
            self.router_bias = (nn.Parameter(torch.empty(E, dtype=torch.float32), requires_grad=False)
                                if cfg.router_bias else None)
            self.router_logit_bias = p(E) if cfg.router_logit_bias else None
            # Expert biases (gpt-oss): gate_up per rank's rows, down full width on rank 0 only
            # (zero elsewhere) since the ranks' partial sums are all-reduced.
            self.w_gu_bias = p(El, 2 * Im) if cfg.expert_bias else None
            self.w_down_bias = p(El, H) if cfg.expert_bias else None
            # Router-weighted shared experts (Inkling): which shared expert each of this rank's
            # shared intermediate units belongs to, one-hot [Is, n] (data, so ranks share graphs).
            self.shared_units = p(Is, cfg.router_shared_rows) if cfg.router_shared_rows and Is else None
            lin("w_gu", "mlp.experts", El, 2 * Im, H, expert=True)  # per expert: gate rows, then up
            # Stored [E, Im, H] (input dim first) unless packed: dequantizing [E, H, Im] with its
            # 64-wide inner dim took 11.4 ms for 32 experts on trn1 against 0.35 ms transposed
            # (tools/profile_moe.py --parts, MiMo-V2.6-Flash at tp=32, 2026-10-02).
            self.down_t = not (keep_fp8 and cfg.quant_expert_mxfp4 and packed_mxfp4)
            lin("w_down", "mlp.experts", El, H, Im, expert=True, transposed=self.down_t)
            if self.moe_ep:
                self.register_buffer("ep_lmap", torch.zeros(1, E + tp * self.ep_s + 1, dtype=torch.int32, device="cpu"),
                                     persistent=False)  # filled per rank by DecoderForCausalLM.ep_buffers
                # The expert tensors' per-slot shapes and dtypes, before pack_experts replaces them (an EPLB rebalance
                # loads new slots into the same layout: models/loader.py load_ep_slots).
                self.ep_meta = {n: (None if getattr(self, n) is None else (tuple(getattr(self, n).shape[1:]),
                                                                            getattr(self, n).dtype))
                                for n in ("w_gu", "w_gu_scale", "w_down", "w_down_scale")}
            if self.moe_ep and moe_kernel in NKI_MOE_KERNELS:
                from ..kernels import moe_ep

                if not moe_ep.supports(self.w_gu, self.w_gu_scale, self.w_down, self.w_down_scale, self.down_t):
                    raise ValueError(f"layer {index}: KILN_MOE_EP=1 with KILN_MOE_KERNEL={moe_kernel} needs FP8 experts "
                                     f"with fp32 128 x 128 block scales stored [E, I, H]; have w_gu "
                                     f"{tuple(self.w_gu.shape)} {self.w_gu.dtype}, down_t={self.down_t}")
                self.pack_moe = True
            elif moe_kernel in NKI_MOE_KERNELS:
                from ..kernels import moe_decode, moe_dedupe

                if cfg.expert_bias or cfg.moe_act != "silu":
                    # gpt-oss: kernels/moe_dedupe.py has its activation (ACTS "swiglu_oai") but not the
                    # experts' gate_up / down biases, and no layer passes that activation yet.
                    raise ValueError(f"layer {index}: KILN_MOE_KERNEL={moe_kernel} computes experts without biases; "
                                     f"{cfg.architecture} (moe_act {cfg.moe_act}, expert_bias {cfg.expert_bias}) "
                                     "needs the XLA path (KILN_MOE_KERNEL=xla)")
                self.pack_tiles = NKI_MOE_KERNELS[moe_kernel] == "tiles"
                lay = moe_dedupe if self.pack_tiles else moe_decode
                if not lay.supports(self.w_gu, self.w_gu_scale, self.w_down, self.w_down_scale, self.down_t):
                    raise ValueError(f"layer {index}: KILN_MOE_KERNEL={moe_kernel} needs FP8 experts with bf16 block-32 "
                                     f"scales (or, for nki, fp32 128 x 128 block scales), 2 x moe_intermediate / tp "
                                     f"= 128 and w_down stored [E, Im, H]; have "
                                     f"w_gu {tuple(self.w_gu.shape)} {self.w_gu.dtype}, down_t={self.down_t}")
                self.pack_moe = True
                if MOE_PREFILL_KERNEL == "nki" and not self.pack_tiles:
                    raise ValueError("KILN_MOE_PREFILL_KERNEL=nki reads the tiles blob: it needs KILN_MOE_KERNEL=nki, "
                                     f"not {moe_kernel}")
            elif MOE_PREFILL_KERNEL == "nki" and not self.moe_ep:
                raise ValueError("KILN_MOE_PREFILL_KERNEL=nki reads the tiles blob: it needs KILN_MOE_KERNEL=nki")
            elif moe_kernel != "xla":
                raise ValueError(f"moe_kernel must be xla or one of {tuple(NKI_MOE_KERNELS)}, not {moe_kernel!r}")
        else:
            I = cfg.intermediate_size // tp
            lin("gate_up", "mlp.gate_proj", 2 * I, H)  # gate_proj, up_proj fused
            lin("down", "mlp.down_proj", H, I)
            self.mlp_scale = p(1) if cfg.dense_mlp_scale else None  # Inkling's global_scale
        self.rope = ""  # name suffix of this layer's RoPE table on the model
        self.k_cache: torch.Tensor | None = None  # [num_pages * page_size, nkv, Dk]
        self.v_cache: torch.Tensor | None = None  # [num_pages * page_size, nkv, Dv]
        # A plain layer of a hyper-connection model (GLM-5.3-Flash's MTP layer) has none of their
        # streams' parameters and runs as an ordinary residual block (DecoderForCausalLM._layer).
        self.plain = plain
        if cfg.hybrid is not None and not plain:  # hyper-connection models: their streams' parameters (models/hybrid.py)
            _hybrid.init_layer(self, cfg, spec, index, tp, p)

    def pack_dense_mlp(self) -> None:
        """kernels/nkilib_dense.py's layout beside the dense MLP's own (filled, on the host): gate and up [H, I] and
        down [I, H], what nkilib's MLP kernel takes. The fused gate_up [2I, H] and down [H, I] stay for every call
        the kernel does not take (decode-sized ones), so those graphs and their keys are unchanged."""
        I = self.gate_up.shape[0] // 2
        self.mlp_gate_t = nn.Parameter(self.gate_up.data[:I].t().contiguous(), requires_grad=False)
        self.mlp_up_t = nn.Parameter(self.gate_up.data[I:].t().contiguous(), requires_grad=False)
        self.mlp_down_t = nn.Parameter(self.down.data.t().contiguous(), requires_grad=False)

    def pack_experts(self) -> None:
        """Replace w_gu, w_gu_scale, w_down, w_down_scale (filled, on the host) by w_blob, the
        layout of kernels/moe_decode.py or (pack_tiles) kernels/moe_dedupe.py; the XLA paths read it
        back through that module's unpack."""
        from ..kernels import moe_decode, moe_dedupe

        if getattr(self, "moe_ep", False):  # kernels/moe_ep.py's layout (getattr: tools pack bare namespaces)
            from ..kernels import moe_ep

            blob = moe_ep.pack(self.w_gu.data, self.w_gu_scale.data, self.w_down.data, self.w_down_scale.data,
                               getattr(self, "ep_tiles", False))
            self.ep_names = tuple(blob)  # the layout's tensors (the tile-scale form has no per-row split scales)
            for n, t in blob.items():
                setattr(self, "ep_" + n, nn.Parameter(t, requires_grad=False))
            for n in ("w_gu", "w_gu_scale", "w_down", "w_down_scale"):
                setattr(self, n, None)
            self.moe_blob = True
            return
        lay = moe_dedupe if self.pack_tiles else moe_decode
        H = self.w_gu.shape[-1]
        self.w_blob = nn.Parameter(lay.pack(self.w_gu.data, self.w_gu_scale.data, self.w_down.data,
                                            self.w_down_scale.data), requires_grad=False)
        for n in ("w_gu", "w_gu_scale", "w_down", "w_down_scale"):
            setattr(self, n, None)
        self.moe_blob = True
        self.moe_tiles = self.pack_tiles
        if self.moe_tiles and MOE_PREFILL_KERNEL == "nki":
            from ..kernels.moe_prefill import check_blob, down_factors

            # True: its dequantize-first path. down_factors: the down scales per output column as a
            # chunk scale times a power of two (bf16 MXFP4 scales; fp32 ones that fit_e4m3_max doubled
            # per row, as GLM-5.3-Flash's), or None; it raises for a layout the kernel cannot take.
            self.moe_prefill_dq = check_blob(self.w_blob.data, H)
            down = down_factors(self.w_blob.data, H)
            self.moe_prefill_dsc, self.moe_prefill_dfr = (
                (None, None) if down is None else (nn.Parameter(t, requires_grad=False) for t in down))


class DecoderForCausalLM(nn.Module):
    """Tensor parallelism is Megatron-style over `tp` ranks: QKV and gate/up (each expert's,
    for MoE layers) are split on their output dim (heads, intermediate), o_proj and down on
    their input dim followed by an all-reduce. The final norm is replicated. With
    vocab_parallel (vLLM VocabParallelEmbedding / ParallelLMHead) each rank holds 1 / tp of the
    vocabulary rows of the embedding and lm_head: a lookup is a masked local lookup plus an
    all-reduce, and the logits are the local shard all-gathered, which equals the replicated
    matmul column for column. Either way every rank samples the same tokens from the same
    noise and only rank 0's output is read.

    Attention TP: the token mixers (GQA / MLA / QSA attention, Gated DeltaNet, KDA) may run at a
    smaller degree attn_tp, a divisor of tp (attention_tp: by default the largest one their head
    counts allow, so a model whose heads divide tp runs exactly as plain TP). The tp ranks form
    tp / attn_tp groups of consecutive ranks (rank r is attention rank r % attn_tp of group
    r // attn_tp; on trn1 a group of 2 is one chip), every group holds the whole set of heads
    split attn_tp ways and computes the same mixer output for every token, reduced over the GROUP
    (attn_group, a torch.distributed subgroup; no collective at all when attn_tp is 1). The MLP,
    experts, embedding and lm_head keep the world tp and its all-reduces. KV caches and
    recurrent states hold the attention rank's heads.

    DP attention (dp_attention = N > 1; SGLang --enable-dp-attention, v0.5.21
    srt/layers/dp_attention.py): the same N groups of attn_tp = tp / N ranks, but each group serves
    its OWN requests. Every graph's batch is the N groups' batches, each padded to one bucket and
    laid out group-major (rows g * T .. (g + 1) * T - 1 are group g's): the residual stream, the
    embedding, the MLP / experts, the lm_head and sampling run over all N * T rows on every rank,
    exactly as at plain TP, while a token mixer takes only its group's T rows (_attn_in) with that
    group's positions, block tables, KV slots and state rows (graph inputs of shape [T, ...] that
    differ per group; engine/model_runner.py PerGroup), and its head-partial output goes back into
    the N * T rows zero-padded outside the group and summed over the WORLD (_attn_all_reduce): one
    all-reduce that is both the attention-TP reduction and the gather of every group's rows, SGLang's
    _dp_gather_via_all_reduce (DpPaddingMode.SUM_LEN) with the attention reduction folded in. Nothing
    needs scattering back: the next mixer selects its rows again. The group is a buffer (dp_index /
    dp_onehot), not a Python int, so every rank traces the same graphs. KV caches and recurrent
    states hold only the group's requests: the per-rank KV a model needs drops N times."""

    def __init__(self, cfg: ModelConfig, dtype: torch.dtype, max_positions: int,
                 tp_rank: int = 0, tp_size: int = 1, tp_group=None, keep_fp8: bool = False,
                 vocab_parallel: bool = False, packed_mxfp4: bool = False, mtp: bool = False,
                 moe_kernel: str = "xla", attn_tp: int | None = None, attn_group=None, dp_attention: int = 1,
                 max_num_seqs: int | None = None, pd_role: str | None = None, eagle3=None):
        """eagle3: a models/eagle3.py DraftConfig, with mtp: the draft head is that EAGLE-3 checkpoint's instead of
        the model's own MTP layer. attn_tp: the attention TP (None: attention_tp's default); attn_group: this rank's
        attention group when 1 < attn_tp < tp_size (engine/tp.py attention_group). dp_attention: the
        number of DP-attention groups (attn_tp is then tp_size / dp_attention, and no attention
        group is needed: the mixers reduce over the world, see above). max_num_seqs: the engine's
        (None: unknown), for the automatic expert-parallel default (moe_ep_enabled). pd_role: the engine's role in a
        disaggregated deployment (EngineConfig.pd_role), which decides that default by itself (below)."""
        super().__init__()
        self.keep_fp8 = keep_fp8
        if cfg.intermediate_size % tp_size:
            raise ValueError(f"tp={tp_size} does not divide the MLP intermediate size")
        self.cfg = cfg
        self.dtype = dtype
        self.tp_rank, self.tp_size, self.tp_group = tp_rank, tp_size, tp_group
        self.dp = dp_attention
        if self.dp > 1:
            if tp_size % self.dp:
                raise ValueError(f"dp_attention={self.dp} does not divide tp={tp_size}")
            if attn_tp not in (None, tp_size // self.dp):
                raise ValueError(f"dp_attention={self.dp} at tp={tp_size} runs attention TP {tp_size // self.dp}, "
                                 f"not {attn_tp}")
            attn_tp = tp_size // self.dp
            if cfg.sconv_kernel:  # the a / m convolutions run on all groups' rows with one group's slots
                raise NotImplementedError("DP attention for Inkling's short convolutions (their attention- and "
                                          "MLP-output inputs span every group's rows, their histories one group's)")
            if cfg.hybrid is not None and cfg.hybrid.ple is not None:
                raise NotImplementedError("DP attention for a Per-Layer Embedding model (its n-gram rows and PLE "
                                          "state are per request but mixed into every stream)")
        self.attn_tp = attention_tp(cfg, tp_size, attn_tp, mtp)
        self.attn_rank = tp_rank % self.attn_tp
        self.attn_group = tp_group if self.attn_tp == tp_size else attn_group
        self.dp_group = tp_rank // self.attn_tp if self.dp > 1 else 0
        self._sp_rs = False  # set while a sequence-parallel block is traced (_sp_rs_call): see _out_reduce
        self._sp_grp = False  # ... with group collectives for the token mixer (_sp_grp_call)
        # Sequence-parallel prefill streams (prefill_sp_enabled above): GLM-5.3-Flash's mHC only. With an MTP
        # head too: the MTP graph after a prefill chunk takes this rank's rows of the chunk's last hidden
        # state and gathers their final norms itself (_mtp_pass sp_onehot), as post_prefill does.
        hy = cfg.hybrid
        self.prefill_sp = (prefill_sp_enabled() and tp_size > 1 and hy is not None and hy.family == "glm5_next"
                           and hy.ple is None)
        self.sp_group = self.prefill_sp and sp_group_enabled()  # the SP token mixers' group collectives
        # Sequence-parallel decode streams (decode_sp_enabled): not with an MTP head, whose decode-side drafting reads the
        # step's final hidden states of every row; that path is not measured with SP decode.
        self.decode_sp = (decode_sp_enabled() and tp_size > 1 and hy is not None and hy.family == "glm5_next"
                          and hy.ple is None and not mtp)
        # Expert parallelism (moe_ep_enabled): only with more than one rank; the automatic default also only where
        # the ranks divide the experts (KILN_MOE_EP=1 raises in DecoderLayer instead) and from
        # ep_auto_min_decode_rows() decode rows per DP-attention group on (4 with the default decode kernel, 8 with
        # KILN_MOE_EP_SMALL_V=1: moe_ep_enabled).
        # A disaggregated engine runs one phase, so the automatic default follows its role rather than its decode
        # rows: a prefill engine takes EP (its 4096-row calls are ~0.3 s faster than TP's, moe_ep_enabled), a
        # decode engine TP experts (the decode agent's ST measurement: KILN_MOE_EP=0 is the faster decode, 2026-10-05).
        # KILN_MOE_EP=0 / 1 still decides, and an engine without a role keeps the decode-row rule.
        ep = moe_ep_enabled(cfg) and tp_size > 1 and bool(cfg.num_experts)
        if ep and os.environ.get("KILN_MOE_EP", "auto") == "auto":
            if cfg.num_experts % tp_size:
                ep = False
            elif pd_role == "decode":
                ep = False
            elif pd_role != "prefill" and max_num_seqs is not None \
                    and max_num_seqs < ep_auto_min_decode_rows() * dp_attention:
                ep = False
        self.moe_ep = ep
        # Redundant expert slots (models/eplb.py, KILN_EP_REDUNDANT): the experts every rank's redundant slots hold,
        # per MoE layer, from a statistics file (KILN_EPLB_INIT) or the placeholder; identical on every rank.
        self.ep_s = _eplb.redundant_slots() if ep else 0
        self.ep_extra: dict[int, list[int]] = {}
        if self.ep_s:
            E = cfg.num_experts
            init = os.environ.get("KILN_EPLB_INIT")
            loads = _eplb.load_file(init, E) if init else {}
            for i in cfg.moe_layers:
                self.ep_extra[i] = (_eplb.replicas(loads[i], tp_size, self.ep_s) if i in loads
                                    else _eplb.default_extra(E, tp_size, self.ep_s))
        self.dp_buffers()
        self.inter = cfg.intermediate_size // tp_size
        self.moe_inter = moe_inter_per_rank(cfg, tp_size, keep_fp8)
        specs = layer_specs(cfg)
        self.vocab_parallel = vocab_parallel and tp_size > 1
        V = cfg.vocab_size
        self.vocab_rows = -(-V // tp_size) if self.vocab_parallel else V  # rows on this rank
        self.vocab_start = tp_rank * self.vocab_rows if self.vocab_parallel else 0
        self.embed = nn.Parameter(torch.empty(self.vocab_rows, cfg.hidden_size, dtype=dtype), requires_grad=False)
        # A tensor, not a Python int, so that every rank traces the SAME graph: a per-rank
        # constant gave each of 32 ranks its own compile cache key (32 compiles of every graph,
        # MiMo-V2.6-Flash on trn1.32xlarge, 2026-10-02).
        self.register_buffer("vocab_start_t", torch.tensor(self.vocab_start, dtype=torch.int64), persistent=False)
        self.layers = nn.ModuleList(DecoderLayer(cfg, specs[i], dtype, tp_size, tp_rank, moe=i in cfg.moe_layers,
                                                 index=i, keep_fp8=keep_fp8, packed_mxfp4=packed_mxfp4,
                                                 moe_kernel=moe_kernel, attn_tp=self.attn_tp,
                                                 attn_rank=self.attn_rank, ep=self.moe_ep,
                                                 ep_extra=self.ep_extra.get(i))
                                    for i in range(cfg.num_layers))
        self.norm = nn.Parameter(torch.empty(cfg.hidden_size, dtype=dtype), requires_grad=False)
        self.embed_norm = (nn.Parameter(torch.empty(cfg.hidden_size, dtype=dtype), requires_grad=False)
                           if cfg.embed_norm else None)
        # Multi-token prediction (vLLM mimo_v2_mtp.py, v0.30.0: only the first MTP layer,
        # applied recursively): h = eh_proj(cat(enorm(embed(next token)), hnorm(previous
        # hidden))), one decoder layer (MiMo-V2: sliding-window attention, dense MLP; DeepSeek-V3 /
        # GLM-5: MLA, DSA, MoE, models/mtp.py), its final norm, and the shared lm_head. It owns one
        # more KV cache (kv_layers()).
        self.mtp = None
        self.mtp_index_share = False
        self.eagle = eagle3 if mtp else None
        if self.eagle is not None:
            # EAGLE-3 (models/eagle3.py): one Llama layer over cat(embeds, hidden) [2H], fc over the target's
            # auxiliary hidden states, the draft's final norm and lm_head over its own vocabulary (replicated on every
            # rank: 32000 x H is small, and a replicated head needs no collective), d2t to the target's ids, and the
            # draft's own embedding when its checkpoint has one (vocabulary-parallel like the target's).
            d, H = self.eagle, cfg.hidden_size
            dspec = layer_specs(d.layer)[0]
            self.mtp = DecoderLayer(d.layer, dspec, dtype, tp_size, tp_rank, moe=False, keep_fp8=False,
                                    prefix=d.prefix.rstrip("."), attn_tp=self.attn_tp, attn_rank=self.attn_rank,
                                    qkv_in=2 * H)
            self.mtp.hidden_norm = nn.Parameter(torch.empty(H, dtype=dtype), requires_grad=False)
            self.eagle_fc = nn.Parameter(torch.empty(H, len(d.aux_layers) * d.target_hidden_size, dtype=dtype),
                                         requires_grad=False)
            self.mtp_norm = nn.Parameter(torch.empty(H, dtype=dtype), requires_grad=False)
            self.eagle_head = nn.Parameter(torch.empty(d.draft_vocab_size, H, dtype=dtype), requires_grad=False)
            self.register_buffer("eagle_d2t", torch.zeros(d.draft_vocab_size, dtype=torch.int64), persistent=False)
            self.eagle_embed = (nn.Parameter(torch.empty(self.vocab_rows, H, dtype=dtype), requires_grad=False)
                                if d.has_embed else None)
        elif mtp:
            if not cfg.mtp_layers:
                raise ValueError("the checkpoint has no MTP layers (num_nextn_predict_layers)")
            H = cfg.hidden_size
            self.mtp = DecoderLayer(cfg, cfg.mtp_spec, dtype, tp_size, tp_rank, moe=cfg.mtp_moe, keep_fp8=keep_fp8,
                                    prefix=(cfg.mtp_prefix or "model.mtp.layers.0"), moe_kernel=moe_kernel,
                                    packed_mxfp4=packed_mxfp4, attn_tp=self.attn_tp, attn_rank=self.attn_rank,
                                    plain=cfg.hybrid is not None, ep=self.moe_ep)
            self.mtp_index_share = _mtp.index_share(cfg)
            self.mtp_enorm, self.mtp_hnorm, self.mtp_norm = (
                nn.Parameter(torch.empty(H, dtype=dtype), requires_grad=False) for _ in range(3))
            self.mtp_eh = nn.Parameter(torch.empty(H, 2 * H, dtype=dtype), requires_grad=False)
        self.lm_head = (
            None if cfg.tie_word_embeddings
            else nn.Parameter(torch.empty(self.vocab_rows, cfg.hidden_size, dtype=dtype), requires_grad=False)
        )
        if cfg.hybrid is not None:
            _hybrid.init_model(self)
        # One RoPE table per distinct (rotary dims, theta, rotating frequencies, scaling). An EAGLE-3 draft layer keeps
        # its own checkpoint's scaling, as vLLM builds the draft's rotary from the draft config
        # (RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3: rope_scaling null under a llama3-scaled target).
        tables: dict[tuple, str] = {}
        for layer in self.kv_layers():
            rs = self.eagle.layer.rope_scaling if self.eagle is not None and layer is self.mtp else cfg.rope_scaling
            key = (layer.spec.rope_dim, layer.spec.rope_theta, layer.spec.rope_freqs, rs)
            if not key[0]:  # no RoPE (Inkling)
                continue
            if key not in tables:
                name = str(len(tables))
                tables[key] = name
                cos, sin = self._rope_table(key[0], key[1], max_positions, key[2], key[3])
                self.register_buffer(f"rope_cos{name}", cos.to(dtype), persistent=False)
                self.register_buffer(f"rope_sin{name}", sin.to(dtype), persistent=False)
            layer.rope = tables[key]
        self._rope_specs = {v: k for k, v in tables.items()}
        self.ep_buffers()
        windows = {l.spec.window for l in self.kv_layers() if l.spec.window is not None}
        if len(windows) > 1:
            raise NotImplementedError(f"more than one sliding-window size ({sorted(windows)})")
        # Sliding-window layers read only the pages under their window (swa_table inputs).
        self.window = windows.pop() if windows else None
        # Relative position logits and short-convolution histories address keys by their
        # position in the block table, so every layer reads the full table (no swa_table).
        self.full_tables = bool(cfg.sconv_kernel or any(l.spec.rel_extent for l in self.kv_layers()))
        self.page_size = 0
        self.fp8_max = None
        # Long-context pooled DSA (models/dsa_long.py): every attention layer is a pooled DSA indexer layer
        # without RoPE (GLM-5.3-Flash), so a bucket past dsa_long.LONG_KEYS keys runs every layer's long path
        # and the prep graphs build no [C, L] visibility for it (long_ctx).
        self.long_dsa = bool(self.layers) and all(
            isinstance(l.spec, LinearSpec) or _mla.long_capable(l.spec) for l in self.layers) and any(
            not isinstance(l.spec, LinearSpec) for l in self.layers)
        # Context parallelism of the DSA caches over the attention group (models/dsa_long.py, KILN_DSA_CP=1): each
        # rank holds 1 / cp of every sequence's latent, indexer rows and pool keys, and every bucket runs the long
        # path. Needs page_size / kpool to be a multiple of the attention TP (checked in bind_kv_cache).
        self.cp = self.attn_tp if (_dsa_long.cp_enabled() and self.long_dsa and self.attn_tp > 1) else 1
        if self.cp > 1 and getattr(self, "mtp", None) is not None:
            raise NotImplementedError("context-parallel DSA with an MTP layer")
        # KILN_DSA_CP_DEGREE (models/dsa_long.py cp_degree_env): context parallelism over cp < attn_tp ranks, the
        # attention group split into cp_rows row groups of cp consecutive ranks (models/mla.py attention_cp_rows).
        # cp_group: the row group's process group (engine/engine.py build_shard sets it; None: the attention group).
        self.cp_rows = 1
        self.cp_group = None
        if self.cp > 1:
            self.cp = _dsa_long.cp_degree(self.attn_tp)
            self.cp_rows = self.attn_tp // self.cp
            if self.cp_rows > 1 and self.attn_tp != tp_size:
                raise NotImplementedError(f"KILN_DSA_CP_DEGREE={self.cp} below the attention TP {self.attn_tp} needs "
                                          f"DP attention 1 and attention TP = TP (have tp={tp_size})")
        # Back-compat for single-spec models (tools and tests read these).
        first = next(iter(self.kv_layers()), None)
        self.nh, self.nkv, self.kv_offset = (first.nh, first.nkv, first.kv_offset) if first is not None else (0, 0, 0)

    def ep_layers(self) -> list:
        """The expert-parallel MoE layers (moe_ep_enabled), the MTP layer's included."""
        layers = list(getattr(self, "layers", [])) + ([self.mtp] if getattr(self, "mtp", None) is not None else [])
        return [l for l in layers if getattr(l, "moe_ep", False)]

    def ep_buffers(self, device=None) -> None:
        """Each expert-parallel layer's ep_lmap int32 [1, E + 1] (kernels/moe_ep.local_map: an expert's index among
        this rank's, or the rank's count for another rank's and for the padding expert E), a buffer so that every
        rank traces the same graphs (see vocab_start_t)."""
        if not self.ep_layers():
            return
        from ..kernels.moe_ep import local_map

        E = self.cfg.num_experts
        owner = torch.arange(E, device="cpu") // (E // self.tp_size)  # (the model may be built on meta)
        lmap = local_map(owner, self.tp_rank)
        for l in self.ep_layers():
            s = getattr(l, "ep_s", 0)
            if not s:
                l.register_buffer("ep_lmap", lmap.clone().to(device), persistent=False)
                continue
            # models/eplb.py: physical ids (primaries e, redundant slot j of rank r E + r s + j), the routing ids
            # mapped to them by remap with these tables (data, so a rebalance changes no graph).
            l.register_buffer("ep_lmap", _eplb.physical_lmap(E, self.tp_size, s, self.tp_rank).to(device),
                              persistent=False)
            self._ep_tables(l, device)
            if _eplb.record_enabled():
                st = getattr(l, "ep_stats", None)
                l.register_buffer("ep_stats", (st if st is not None and st.device.type != "meta"
                                               else torch.zeros(E, dtype=torch.float32)).to(device), persistent=False)

    def _ep_tables(self, l, device=None) -> None:
        """A layer's remap tables from its ep_extra_all (models/eplb.py tables): prefill calls spread a replicated
        expert's pairs over its copies, decode calls too unless KILN_EPLB_DECODE=0. Registered at build, copied
        into the existing device buffers at a rebalance (device None)."""
        E, s = self.cfg.num_experts, l.ep_s
        ids, mp = _eplb.tables(l.ep_extra_all, E, self.tp_size, s, spread=True)
        _, mpd = _eplb.tables(l.ep_extra_all, E, self.tp_size, s, spread=_eplb.decode_replicas())
        for name, t in (("ep_rep_ids", ids), ("ep_rep_map", mp), ("ep_rep_map_d", mpd)):
            cur = getattr(l, name, None)
            if device is None and cur is not None and cur.shape == t.shape and cur.device.type != "meta":
                cur.copy_(t.to(cur.device))
            else:
                l.register_buffer(name, t.to(device) if device is not None else t, persistent=False)

    def _ep_phys(self, layer, topi: torch.Tensor, decode: bool) -> torch.Tensor:
        """Routing ids -> physical expert ids of a layer with redundant slots (models/eplb.py remap)."""
        return _eplb.remap(topi, layer.ep_rep_ids, layer.ep_rep_map_d if decode else layer.ep_rep_map)

    def dp_buffers(self, device=None) -> None:
        """DP attention: this rank's group as buffers (an index [1] and a one-hot [N] row in the
        model dtype), so that every rank traces the same graph (a per-rank Python int would give
        each rank its own compile, see vocab_start_t). The same for the sequence-parallel prefill
        streams (prefill_sp): this rank's row block as an index [1] and a one-hot [tp_size] row. Also the expert-
        parallel layers' local maps (ep_buffers)."""
        self.ep_buffers(device)
        if (getattr(self, "prefill_sp", False) or getattr(self, "decode_sp", False)) and self.tp_size > 1:
            sp_onehot = torch.zeros(self.tp_size, dtype=self.dtype, device="cpu")
            sp_onehot[self.tp_rank] = 1
            self.register_buffer("sp_index", torch.tensor([self.tp_rank], dtype=torch.int64, device="cpu").to(device),
                                 persistent=False)
            self.register_buffer("sp_onehot", sp_onehot.to(device), persistent=False)
            if self.dp > 1:  # the rank's row block within its attention group (_sp_group_gather)
                g1 = torch.zeros(self.attn_tp, dtype=self.dtype, device="cpu")
                g1[self.attn_rank] = 1
                self.register_buffer("sp_grp_index", torch.tensor([self.attn_rank], dtype=torch.int64,
                                                                  device="cpu").to(device), persistent=False)
                self.register_buffer("sp_grp_onehot", g1.to(device), persistent=False)
        if getattr(self, "long_dsa", False) and self.attn_tp > 1:
            # Long-context DSA prefill (models/mla.py _long_prefill_select): this rank's block of a chunk's queries
            # within its attention group, as an index [1] and an fp32 one-hot [attn_tp] row (fp32: the gathered
            # pool indices go up to 262,144, which bf16 cannot hold). With row groups (cp_rows > 1,
            # KILN_DSA_CP_DEGREE): the rank within its row group of cp ranks, and the row group as cp_row_index [1]
            # / cp_row_onehot [cp_rows].
            rows = getattr(self, "cp_rows", 1)
            A = self.cp if rows > 1 else self.attn_tp
            g2 = torch.zeros(A, dtype=torch.float32, device="cpu")
            g2[self.attn_rank % A] = 1
            self.register_buffer("long_grp_index", torch.tensor([self.attn_rank % A], dtype=torch.int64,
                                                                device="cpu").to(device), persistent=False)
            self.register_buffer("long_grp_onehot", g2.to(device), persistent=False)
            if rows > 1 and _mla.CP_FLAGSTAT:  # the long-context selection's local-list statistics (opt-in)
                for l in self.kv_layers():
                    if _mla.long_capable(l.spec):
                        l.register_buffer("cp_flagstat", torch.zeros(2 + 2 * len(_mla.CP_FLAG_KS), dtype=torch.float32,
                                                                     device="cpu").to(device), persistent=False)
            if rows > 1 and _mla.CP_LOCAL_K > 0:  # KILN_DSA_CP_LOCAL_K's counts: [rows, failed, past CP_LOCAL_F]
                for l in self.kv_layers():
                    if _mla.long_capable(l.spec):
                        l.register_buffer("cp_localk", torch.zeros(3, dtype=torch.float32, device="cpu").to(device),
                                          persistent=False)
            if rows > 1:
                g3 = torch.zeros(rows, dtype=torch.float32, device="cpu")
                g3[self.attn_rank // A] = 1
                self.register_buffer("cp_row_index", torch.tensor([self.attn_rank // A], dtype=torch.int64,
                                                                  device="cpu").to(device), persistent=False)
                self.register_buffer("cp_row_onehot", g3.to(device), persistent=False)
        if self.dp == 1:
            return
        # Built on the host and copied: eager ops on the neuron device failed here ("Expected
        # self.dtype() == dst.dtype()", trn1, SDK 2.32, 2026-10-03).
        onehot = torch.zeros(self.dp, dtype=self.dtype, device="cpu")
        onehot[self.dp_group] = 1
        self.register_buffer("dp_index", torch.tensor([self.dp_group], dtype=torch.int64, device="cpu").to(device),
                             persistent=False)
        self.register_buffer("dp_onehot", onehot.to(device), persistent=False)

    def kv_layers(self) -> list:
        """Every layer with a KV cache: the decoder layers, then the MTP layer if any.
        Linear-attention layers hold a recurrent state instead (state_layers)."""
        layers = [l for l in self.layers if not isinstance(l.spec, LinearSpec)]
        return layers + ([self.mtp] if getattr(self, "mtp", None) is not None else [])

    def state_layers(self) -> list:
        """Layers whose per-sequence state is a fixed-size recurrent state (linear attention)."""
        return [l for l in self.layers if isinstance(l.spec, LinearSpec)]

    def state_shapes(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Per state layer, this rank's per-sequence (conv state, recurrent state) shapes."""
        return [linear_attn.state_shapes(l) for l in self.state_layers()]

    def bind_state(self, conv_states: list[torch.Tensor], rec_states: list[torch.Tensor]) -> None:
        """Device pools [slots, ...] of every state layer (engine/state_pool.py)."""
        for layer, c, r in zip(self.state_layers(), conv_states, rec_states):
            layer.conv_state = c
            layer.rec_state = r

    def aux_kv_shapes(self) -> list[tuple[int, ...]]:
        """Extra per-token caches (shape per token) the runner allocates beside K and V (models/hybrid.py)."""
        return _hybrid.aux_kv_shapes(self) if self.cfg.hybrid is not None else []

    def bind_aux_kv(self, caches: list[torch.Tensor]) -> None:
        if caches:
            _hybrid.bind_aux_kv(self, caches)

    def aux_state_shapes(self) -> list[tuple[tuple[int, ...], torch.dtype]]:
        """Extra per-request state rows the state pool allocates (models/hybrid.py)."""
        return _hybrid.aux_state_shapes(self) if self.cfg.hybrid is not None else []

    def bind_aux_state(self, states: list[torch.Tensor]) -> None:
        if states:
            _hybrid.bind_aux_state(self, states)

    def rope_specs(self) -> dict[str, tuple[int, float]]:
        """RoPE buffer suffix -> (rotary dims, theta)."""
        return {name: key[:2] for name, key in self._rope_specs.items()}

    def rope_tables(self, n: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """RoPE buffer suffix -> (cos, sin) fp32 tables over n positions."""
        return {name: self._rope_table(key[0], key[1], n, key[2], key[3]) for name, key in self._rope_specs.items()}

    def _rope_table(self, dim: int, theta: float, n: int, freqs: int | None = None, scaling="cfg"):
        """scaling: ModelConfig.rope_scaling items; "cfg" for the model's own."""
        rs = dict((self.cfg.rope_scaling if scaling == "cfg" else scaling) or ())
        scale = 1.0
        if freqs is not None:  # only the first `freqs` pairs rotate, at a 2 * freqs-dim RoPE's rates (K2-Horizon)
            inv_freq = 1.0 / (theta ** (torch.arange(0, 2 * freqs, 2, dtype=torch.int64).float() / (2 * freqs)))
            inv_freq = torch.cat([inv_freq, torch.zeros(dim // 2 - freqs)])
        elif rs.get("rope_type") == "yarn":  # DeepSeek-V3, Kimi K2, gpt-oss: cos / sin scaled by attention_factor
            inv_freq, scale = _mla.yarn_inv_freq(dim, theta, rs, self.cfg.max_position_embeddings)
        else:
            inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        if rs.get("rope_type") == "llama3":
            inv_freq = llama3_inv_freq(inv_freq, rs)
        freqs_t = torch.outer(torch.arange(n, dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        return emb.cos() * scale, emb.sin() * scale

    def kv_shapes(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Per layer: ((nkv, Dk), (nkv, Dv)) of the K and V caches on this rank."""
        return [((l.nkv, l.spec.head_dim), (l.nkv, l.spec.v_head_dim)) for l in self.kv_layers()]

    def token_state_shapes(self) -> list[dict[str, tuple[int, ...]]]:
        """Per layer: token-slot state caches beside K and V on this rank, name -> per-token shape
        (Inkling's short-convolution inputs, kept per token so prefix caching and speculative
        rollback need nothing more; a pooled DSA indexer's pool-key pieces, models/mla.py
        pool_key_width); empty for other models."""
        out = []
        for l in self.kv_layers():
            d = ({n: (getattr(l, n[0] + "_sconv").shape[0],) for n in ("k_hist", "v_hist", "a_hist", "m_hist")}
                 if getattr(l, "k_sconv", None) is not None else {})
            w = _mla.pool_key_width(l.spec, self.fp8_max is not None)
            if w:
                d["pool_key"] = (w,)
            out.append(d)
        return out

    def bind_token_states(self, caches: list[dict[str, torch.Tensor]]) -> None:
        for layer, d in zip(self.kv_layers(), caches):
            for n, t in d.items():
                setattr(layer, n, t)

    def bind_kv_cache(self, k_caches: list[torch.Tensor], v_caches: list[torch.Tensor], page_size: int,
                      fp8_max: float | None = None, max_rows: int = 1, max_keys: int | None = None) -> None:
        """fp8_max: the cache is float8 e4m3 and values are clamped to +-fp8_max on write.
        max_rows / max_keys: the most query tokens one call carries and the longest context one
        reads (size DSA's selection scratch; max_keys defaults to the cache's slots)."""
        for layer, k, v in zip(self.kv_layers(), k_caches, v_caches):
            layer.k_cache = k
            layer.v_cache = v
        self.page_size = page_size
        self.fp8_max = fp8_max
        if self.cp > 1:
            kp = next(l.spec.mla.dsa.kpool for l in self.kv_layers())
            if page_size % kp or (page_size // kp) % self.cp:
                raise ValueError(f"context-parallel DSA: {page_size // kp} pools per page do not divide over "
                                 f"attention TP {self.cp} (KILN_DSA_CP=1 needs page_size a multiple of kpool x it)")
            max_keys = 1  # every bucket runs the long path, which stages nothing
        _mla.bind_scratch(self, max_rows, k_caches[0].device if k_caches else None,
                          max_keys or (k_caches[0].shape[0] if k_caches else None))

    def _store(self, cache: torch.Tensor, slots: torch.Tensor, x: torch.Tensor) -> None:
        if self.fp8_max is not None:
            # KV scale 1.0 (vLLM's default without calibration); clamp first, because a
            # finite value past the format's max would become inf / NaN on the cast.
            x = x.float().clamp(-self.fp8_max, self.fp8_max).to(cache.dtype)
        cache.index_put_((slots,), x)

    def _load(self, cache: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        kv = self._gather(cache, block_table)
        return kv.to(self.dtype) if self.fp8_max is not None else kv

    # -- shared pieces --------------------------------------------------------------

    def _all_reduce(self, x: torch.Tensor, group=None) -> torch.Tensor:
        """Sum over the tensor-parallel ranks (or over `group`, a subgroup of them)."""
        if self.tp_size == 1:
            return x
        group = self.tp_group if group is None else group
        if x.device.type == "cpu":
            import torch.distributed as dist

            y = x.clone()
            dist.all_reduce(y, group=group)
            return y
        # Lowered by libtorch_neuronx_lite to an in-graph collective (_c10d_functional.all_reduce,
        # libtorch_neuronx_lite/overrides/xla_collectives.py); compile flow only. Its replica
        # groups are the group's ranks (_get_replica_groups_from_group_name), which also enter the
        # compile cache key (compile/cache.py create_cache_hash), so attention groups each get
        # their own NEFF of a graph holding a group all-reduce.
        import torch.distributed._functional_collectives as funcol

        return funcol.all_reduce(x, "sum", group)

    def _attn_in(self, x: torch.Tensor) -> torch.Tensor:
        """A token mixer's input: under DP attention this rank's group's rows of the group-major
        batch x [N * T, ...] (an index_select by the dp_index buffer), else x itself; x itself also while a
        sequence-parallel block is traced with group collectives (_sp_grp_call: x is already the group's)."""
        if self.dp == 1 or self._sp_grp:
            return x
        return x.reshape(self.dp, -1, *x.shape[1:]).index_select(0, self.dp_index)[0]

    def _sp_group_gather(self, x: torch.Tensor) -> torch.Tensor:
        """The rows [attn_tp * r, ...] of this rank's attention group from each member's own x [r, ...], in
        rank order: x in its block of a zero [attn_tp, r, ...] summed over the group (DP attention with
        sequence-parallel streams: a group's ranks hold exactly its rows, the group-major batch's rows
        dp_group T .. (dp_group + 1) T). Exact, as _sp_gather."""
        from ..kernels import sp_gather

        if sp_gather.enabled(x, group=True):  # KILN_SP_GATHER=nki-all: an NKI kernel's all_gather (kernels/sp_gather.py)
            return sp_gather.gather(x, self.tp_size, self.attn_tp)
        full = x.unsqueeze(0) * self.sp_grp_onehot.view(self.attn_tp, *[1] * x.dim())
        return self._all_reduce(full.reshape(-1, *x.shape[1:]), self.attn_group)

    def _group_reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        """Block attn_rank of attn_tp equal row blocks of x [attn_tp * r, ...] summed over the attention group:
        a group reduce-scatter (on the host: the group all-reduce and those rows)."""
        if x.device.type == "cpu":
            y = self._all_reduce(x, self.attn_group)
            return y.reshape(self.attn_tp, -1, *x.shape[1:]).index_select(0, self.sp_grp_index)[0]
        import torch.distributed._functional_collectives as funcol

        return funcol.reduce_scatter_tensor(x, "sum", 0, self.attn_group)

    def _sp_grp_ok(self) -> bool:
        """Whether a sequence-parallel token mixer block can run on group collectives (_sp_grp_call): DP attention
        with an attention group of more than one rank."""
        return (self.tp_size > 1 and self.dp > 1 and self.attn_tp > 1 and self.attn_group is not None
                and hasattr(self, "sp_grp_onehot"))

    def _sp_grp_call(self, fn, *args):
        """fn(*args) on this attention group's rows with the token mixer's output reduction a group
        reduce-scatter (_attn_all_reduce): this rank's own rows."""
        self._sp_grp = True
        try:
            return fn(*args)
        finally:
            self._sp_grp = False

    def _attn_all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """A token mixer's output projection summed over this rank's attention group: the world
        all-reduce at attn_tp == tp (the very same graph as plain TP), nothing at attn_tp == 1.
        DP attention: x is the group's [T, H] head partial; it is placed in its group's rows of a
        zero [N * T, H] (a one-hot multiply: index_copy gave NaN on the device, see _head) and
        all-reduced over the world, which sums each group's head partials and gathers every
        group's rows at once (SGLang _dp_gather_via_all_reduce: zero buffer, own rows, all_reduce). While a
        sequence-parallel block is traced with group collectives (_sp_grp_call): the group reduce-scatter of
        the group's [T, H] partial, this rank's own rows."""
        if self._sp_grp:
            return self._group_reduce_scatter(x)
        if self.dp > 1:
            full = x.unsqueeze(0) * self.dp_onehot.view(self.dp, *[1] * x.dim())
            return self._out_reduce(full.reshape(-1, *x.shape[1:]))
        if self.attn_tp == 1 or self.tp_size == 1:
            return x
        if self.attn_tp == self.tp_size:
            return self._out_reduce(x)
        if self._sp_rs:
            raise RuntimeError("a reduce-scatter of a block output needs DP attention or attention TP = TP")
        if self.attn_group is None:
            raise RuntimeError(f"attention tp={self.attn_tp} of tp={self.tp_size} needs its attention group")
        return self._all_reduce(x, self.attn_group)

    def _reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        """Block tp_rank of tp_size equal row blocks of x [tp_size * r, ...] summed over the world: the rows
        _sp_rows would keep of _all_reduce(x), from one world reduce-scatter (half the bytes). A graph
        holding one reloads from the compile cache in a later process, unlike an all-gather
        (tools/probe_rs_reload.py: both runs correct, trn1.32xlarge, 2026-10-04). On the host (gloo) it is
        that all-reduce and those rows."""
        if self.tp_size == 1:
            return x
        if x.device.type == "cpu":
            return self._sp_rows(self._all_reduce(x))
        import torch.distributed._functional_collectives as funcol

        return funcol.reduce_scatter_tensor(x, "sum", 0, self.tp_group)

    def _out_reduce(self, x: torch.Tensor) -> torch.Tensor:
        """A block output's reduction over the ranks: the all-reduce, or while a sequence-parallel block is
        traced with KILN_SP_RS (_sp_rs_call) the reduce-scatter that hands each rank its own rows."""
        return self._reduce_scatter(x) if self._sp_rs else self._all_reduce(x)

    def _sp_rs_call(self, fn, *args):
        """fn(*args) with its block output reduced by a reduce-scatter (_out_reduce): this rank's rows."""
        self._sp_rs = True
        try:
            return fn(*args)
        finally:
            self._sp_rs = False

    def _sp_rs_attn(self) -> bool:
        """Whether the token mixers' output reduction can be a world reduce-scatter (_attn_all_reduce):
        under DP attention (the zero-padded world all-reduce) or with attention TP = TP."""
        return self.tp_size > 1 and (self.dp > 1 or self.attn_tp == self.tp_size)

    def _sp_on(self) -> bool:
        """Whether a prefill chunk's streams are sequence-parallel (prefill_sp_enabled)."""
        return self.prefill_sp and self.tp_size > 1

    def _sp_on_decode(self) -> bool:
        """Whether a decode call's streams are sequence-parallel (decode_sp_enabled)."""
        return getattr(self, "decode_sp", False) and self.tp_size > 1

    def _sp_rows(self, x: torch.Tensor) -> torch.Tensor:
        """This rank's rows of a batch x [R, ...] held by every rank: block tp_rank of tp_size equal
        blocks (an index_select by the sp_index buffer, as _attn_in)."""
        return x.reshape(self.tp_size, -1, *x.shape[1:]).index_select(0, self.sp_index)[0]

    def _sp_gather_kernel_ok(self) -> bool:
        """Whether the world row gather may run as kernels/sp_gather.py's NKI all_gather: a model without routed experts,
        or with expert-parallel ones. With tensor-parallel experts (KILN_MOE_EP=0, or the automatic expert parallelism
        left off, as check_ppl's 4 sequences leave it) neuronx-cc 2.27 fails GLM-5.3-Flash's prefill pieces with
        [NCC_ISCH719] "topological order violations" (12- and 6-layer pieces, and check_ppl --chunk 1024's; trn1,
        SDK 2.32, 2026-10-05, docs/neuron-notes.md "Collectives issued from an NKI kernel on trn1"), so those keep the
        zero-padded all-reduce."""
        return bool(getattr(self, "moe_ep", False)) or not self.cfg.num_experts

    def _sp_gather(self, x: torch.Tensor, onehot: torch.Tensor | None = None) -> torch.Tensor:
        """Every rank's rows [tp_size * r, ...] from each rank's own x [r, ...], in rank order: x in
        its block of a zero [tp_size, r, ...] (a one-hot multiply, as _attn_all_reduce) summed over
        the world. Exact: each element is one rank's value plus zeros. onehot: the sp_onehot buffer
        when a graph takes it as an argument (forward_mtp_k)."""
        from ..kernels import sp_gather

        if onehot is None and self._sp_gather_kernel_ok() and sp_gather.enabled(x):  # KILN_SP_GATHER=nki
            return sp_gather.gather(x, self.tp_size, self.tp_size)
        oh = self.sp_onehot if onehot is None else onehot
        full = x.unsqueeze(0) * oh.view(self.tp_size, *[1] * x.dim())
        return self._all_reduce(full.reshape(-1, *x.shape[1:]))

    def _w(self, layer: DecoderLayer, name: str) -> torch.Tensor:
        return dequant(getattr(layer, name), getattr(layer, name + "_scale"), self.dtype)

    def _all_gather0(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate every rank's x along dim 0 (rank order)."""
        if x.device.type == "cpu":
            import torch.distributed as dist

            parts = [torch.empty_like(x) for _ in range(self.tp_size)]
            dist.all_gather(parts, x.contiguous(), group=self.tp_group)
            return torch.cat(parts)
        # _c10d_functional.all_gather_into_tensor, which libtorch_neuronx_lite lowers in the
        # compile flow (overrides/neuron_collectives.py); it concatenates along dim 0.
        import torch.distributed._functional_collectives as funcol

        return funcol.all_gather_tensor(x.contiguous(), 0, self.tp_group)

    def _embed(self, ids: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        """weight: another embedding of the target's vocabulary layout (an EAGLE-3 draft's own, eagle_embed). The
        weight is read where the embedding is taken, AFTER vocab_start_t: dynamo lifts graph inputs in the order the
        trace touches them, and reading self.embed first swapped two inputs of every graph holding the embedding (two
        GLM-5.3-Flash G64 graphs per rank re-keyed, tools/compile_farm.py capture against bd3416a, 2026-10-07)."""
        if not self.vocab_parallel:
            e = F.embedding(ids, self.embed if weight is None else weight)
        else:
            local = ids - self.vocab_start_t
            mine = (local >= 0) & (local < self.vocab_rows)
            e = F.embedding(local.clamp(0, self.vocab_rows - 1), self.embed if weight is None else weight)
            e = self._all_reduce(torch.where(mine.unsqueeze(-1), e, torch.zeros_like(e)))
        return rms_norm(e, self.embed_norm, self.cfg.rms_norm_eps) if self.embed_norm is not None else e

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        return self._head(self._final(h))

    def _final(self, h: torch.Tensor) -> torch.Tensor:
        """The last hidden state, normalised for the lm_head."""
        if self.cfg.hybrid is not None:  # collapse the residual streams (models/hybrid.py)
            return _hybrid.final(self, h)
        return rms_norm(h, self.norm, self.cfg.rms_norm_eps)

    def _hidden_in(self, input_ids: torch.Tensor, ngram_ids: torch.Tensor | None = None,
                   sp: bool = False) -> torch.Tensor:
        """sp: only this rank's rows (sequence-parallel prefill streams, prefill_sp_enabled)."""
        if self.cfg.hybrid is not None:  # residual streams (+ n-gram embedding rows)
            return _hybrid.hidden_in(self, input_ids, ngram_ids, sp)
        return self._embed(input_ids)

    def _head(self, h: torch.Tensor) -> torch.Tensor:
        """lm_head over hidden states that are already normalised."""
        w = self.embed if self.lm_head is None else self.lm_head
        if self.cfg.logits_divisor != 1.0:  # Inkling: lm_head(h / logits_mup_width_multiplier)
            h = h / self.cfg.logits_divisor
        if not self.vocab_parallel:
            return F.linear(h, w).float()
        # Gathered with an all-reduce of zero-padded shards, not all_gather_into_tensor: a
        # cached NEFF holding an all-gather, loaded by a LATER process, failed with "replica
        # group signature mismatch ... mismatched collectives between peers" on all 32 ranks
        # (MiMo-V2.6-Flash, trn1.32xlarge, SDK 2.32, 2026-10-03), while all-reduce graphs
        # reload fine. The shard's column offset is a buffer, so every rank traces one graph.
        # The shard is placed by a one-hot multiply, not index_copy: with index_copy every prompt
        # logprob came back NaN on the device (same run, same graphs otherwise).
        # The shard is found by multiplying, not dividing: with vocab_start_t // Vr, gpt-oss-120b
        # at tp=32 on trn1 returned its top tokens one shard (Vr = 6284 ids) too low (SDK 2.32,
        # 2026-10-03; tools/probe_gqa_moe.py intdiv and docs/neuron-notes.md).
        T, Vr = h.shape[0], self.vocab_rows
        local = F.linear(h, w)  # [T, Vr]
        mine = (torch.arange(self.tp_size, device=h.device) * Vr == self.vocab_start_t).to(local.dtype)
        full = (local.unsqueeze(1) * mine.view(1, -1, 1)).reshape(T, self.tp_size * Vr)
        return self._all_reduce(full)[:, : self.cfg.vocab_size].float()

    def _score_rows(self, hn: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """sampler.score_rows(self._head(hn), targets): [R, 2 + 2N] (rank, logprob, top-N ids, top-N logprobs). With
        KILN_PLP_VP and a vocab-parallel head, from each rank's shard of the logits: the target's logit is its
        owner's (one value plus zeros over the world: exact), its rank the shards' counts of larger logits summed, the
        log-sum-exp combined from the shards' (logsumexp_large of each), the top N the best of the shards' top N."""
        if not (PLP_VP and self.vocab_parallel and self.tp_size > 1):
            return score_rows(self._head(hn), targets)
        w = self.embed if self.lm_head is None else self.lm_head
        h = hn / self.cfg.logits_divisor if self.cfg.logits_divisor != 1.0 else hn
        R, Vr, tp, N = h.shape[0], self.vocab_rows, self.tp_size, NUM_TOP_LOGPROBS
        local = F.linear(h, w).float()  # [R, Vr]: _head's values (bf16 matmul, then fp32)
        # Ids in fp32 (exact below 2^24): a cast of an int64 id sum to fp32 failed to lower ("shift-left with different
        # element types: s64[] and s32[]", the capture, 2026-10-06), so the shard's first id is a float from the one-hot.
        mine = (torch.arange(tp, device=h.device) * Vr == self.vocab_start_t).to(torch.float32)  # [tp] one-hot
        start = (mine * torch.arange(tp, device=h.device, dtype=torch.float32)).sum() * float(Vr)
        col = torch.arange(Vr, device=h.device, dtype=torch.float32).view(1, Vr) + start
        real = col < torch.full_like(col, float(self.cfg.vocab_size))  # the last shard's padding columns are not ids
        local = torch.where(real, local, torch.full_like(local, -1e30))
        rel = targets.view(R, 1).to(torch.float32) - start
        own = (rel >= torch.zeros_like(rel)) & (rel < torch.full_like(rel, float(Vr)))
        at = rel.clamp(0.0, float(Vr - 1)).to(torch.int64)
        tl = torch.where(own, torch.gather(local, 1, at), torch.zeros_like(rel))
        tgt = self._all_reduce(tl)  # [R, 1]
        rank = self._all_reduce((local > tgt).to(torch.float32).sum(1, keepdim=True)) + 1.0
        lse_all = self._all_reduce(logsumexp_large(local) * mine.view(1, tp))  # [R, tp]
        lse = torch.logsumexp(lse_all, dim=-1, keepdim=True)
        vals, idx = topk_large(local, N)
        gid = idx.float() + start
        cv = self._all_reduce((vals.unsqueeze(1) * mine.view(1, tp, 1)).reshape(R, tp * N))
        ci = self._all_reduce((gid.unsqueeze(1) * mine.view(1, tp, 1)).reshape(R, tp * N))
        tv, tj = torch.topk(cv, N, dim=-1)
        ti = torch.gather(ci, 1, tj)
        return torch.cat([rank, tgt - lse, ti, tv - lse], dim=1)

    def _qkv(self, layer: DecoderLayer, x: torch.Tensor, positions: torch.Tensor, ctx=None):
        """ctx: (positions, slot_mapping, table) of a paged forward, for layers with short
        convolutions (their history caches); None for a whole sequence from position 0."""
        cfg, sp = self.cfg, layer.spec
        T = x.shape[0]
        qkv = F.linear(x, self._w(layer, "qkv"), layer.qkv_bias)
        # Explicit slices, not torch.split: libtorch_neuronx_lite (SDK 2.32) miscompiles an
        # unequal torch.split on the last dim (measured, tools/debug_device.py split_variants).
        qs, ks, _ = layer.split
        q, kf, vf = qkv[..., :qs], qkv[..., qs : qs + ks], qkv[..., qs + ks :]
        if layer.k_sconv is not None:  # Inkling: k and v pass their short convolutions before k_norm
            kf, vf = self._sconv(layer, "k", kf, ctx), self._sconv(layer, "v", vf, ctx)
        q = q.reshape(T, layer.nh, sp.head_dim)
        k = kf.reshape(T, layer.nkv, sp.head_dim)
        v = vf.reshape(T, layer.nkv, sp.v_head_dim)
        if layer.q_norm is not None:
            q = rms_norm(q, layer.q_norm, cfg.rms_norm_eps)
            k = rms_norm(k, layer.k_norm, cfg.rms_norm_eps)
        if cfg.v_scale is not None:
            v = v * cfg.v_scale
        if not sp.rope_dim:
            return q, k, v
        cos = getattr(self, f"rope_cos{layer.rope}")[positions].unsqueeze(1)
        sin = getattr(self, f"rope_sin{layer.rope}")[positions].unsqueeze(1)
        r = sp.rope_dim
        if r == sp.head_dim:
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        else:  # partial rotary: RoPE on the first r dims only
            q = torch.cat([q[..., :r] * cos + rotate_half(q[..., :r]) * sin, q[..., r:]], dim=-1)
            k = torch.cat([k[..., :r] * cos + rotate_half(k[..., :r]) * sin, k[..., r:]], dim=-1)
        return q, k, v

    def _softmax(self, layer: DecoderLayer, s: torch.Tensor, kv_axis: int) -> torch.Tensor:
        """s: fp32 scores whose axes kv_axis, kv_axis + 1 are (KV head, query group) and whose
        last axis is keys. An attention-sink logit (one per query head) joins the softmax
        normaliser but carries no value, exactly as MiMo-V2's eager attention appends a
        column and drops it after the softmax."""
        if layer.sink is None:
            return torch.softmax(s, dim=-1).to(self.dtype)
        shape = [1] * s.dim()
        shape[kv_axis], shape[kv_axis + 1] = layer.nkv, layer.nh // layer.nkv
        sink = layer.sink.float().view(shape).expand(*s.shape[:-1], 1)
        return torch.softmax(torch.cat([s, sink], dim=-1), dim=-1)[..., :-1].to(self.dtype)

    def _mlp(self, layer: DecoderLayer, h: torch.Tensor, ctx=None) -> torch.Tensor:
        x = rms_norm(h, layer.post_norm, self.cfg.rms_norm_eps)
        if self.cfg.hybrid is not None:  # a plain layer of a hybrid: its MLP as the hybrid's (clamped SwiGLU)
            return h + _hybrid._mlp(self, layer, x)
        if layer.moe and layer.shared_units is not None:  # Inkling: the router weights the shared experts too
            topv, topi, gammas = self._route_shared(layer, x)
            y = self._moe_routed(layer, x, topv, topi)
            y = y + self._swiglu_mlp(layer, x, "shared_gate_up", "shared_down",
                                     (gammas.to(self.dtype) @ layer.shared_units.t()))
        elif layer.moe:
            y = self._moe(layer, x)
            if layer.shared_gate_up is not None:  # shared experts join the routed sum before the all-reduce
                y = y + self._swiglu_mlp(layer, x, "shared_gate_up", "shared_down")
        else:
            y = self._swiglu_mlp(layer, x, "gate_up", "down")
            if layer.mlp_scale is not None:
                y = y * layer.mlp_scale
        y = self._all_reduce(y)
        if layer.m_sconv is not None:
            y = self._sconv(layer, "m", y, ctx)
        return h + y

    def _swiglu_mlp(self, layer: DecoderLayer, x: torch.Tensor, gate_up: str, down: str,
                    unit_scale: torch.Tensor | None = None) -> torch.Tensor:
        """unit_scale [T, intermediate]: per-token weights of the intermediate units (Inkling's
        router-weighted shared experts)."""
        gt = getattr(layer, "mlp_gate_t", None) if gate_up == "gate_up" and unit_scale is None else None
        if gt is not None and x.dim() == 2 and _nkilib_dense.can_use_mlp(x.shape[0], x.shape[1], gt.shape[1]):
            # A prefill-sized call through nkilib's MLP kernel (KILN_DENSE_MLP_KERNEL=nkilib, DecoderLayer.pack_dense_mlp)
            return _nkilib_dense.mlp(x, gt, layer.mlp_up_t, layer.mlp_down_t)
        gu = F.linear(x, self._w(layer, gate_up))
        half = gu.shape[-1] // 2
        a = F.silu(gu[..., :half]) * gu[..., half:]
        return F.linear(a * unit_scale if unit_scale is not None else a, self._w(layer, down))

    def _route_shared(self, layer: DecoderLayer, x: torch.Tensor):
        """Inkling (transformers 5.15 InklingTopkRouter): the k routed experts are chosen by
        sigmoid(logit) + selection bias; their and the shared experts' weights are sigmoid(logit)
        normalised over all k + n_shared of them (exp of logsigmoid minus its logsumexp), times
        route_scale and the router's global_scale. Returns (topv, topi, shared weights [T, n])."""
        cfg = self.cfg
        E, k = cfg.num_experts, cfg.num_experts_per_tok
        logits = F.linear(x.float(), layer.router.float())  # [T, E + n]
        routed, shared = logits[:, :E], logits[:, E:]
        choice = routed.sigmoid() + layer.router_bias if layer.router_bias is not None else routed.sigmoid()
        _, topi = torch.topk(choice, k, dim=-1)
        lp = F.logsigmoid(torch.cat([torch.gather(routed, 1, topi), shared], dim=-1))
        w = torch.exp(lp - torch.logsumexp(lp, dim=-1, keepdim=True)) * cfg.routed_scaling_factor
        if layer.router_scale is not None:
            w = w * layer.router_scale
        return w[:, :k].to(self.dtype), topi, w[:, k:]

    def _sconv(self, layer: DecoderLayer, which: str, x: torch.Tensor, ctx) -> torch.Tensor:
        """Inkling's causal depthwise short convolution plus its input, in fp32
        (InklingShortConvolution): out[t] = x[t] + sum_j w[:, j] * x[t - (K - 1) + j], zeros before
        the sequence start. x [T, C]. With ctx (positions, slot_mapping, table) the inputs are
        stored per token slot and the K - 1 earlier ones read back through the block table; ctx
        None is a whole sequence from position 0."""
        w = getattr(layer, which + "_sconv")  # [C, K] fp32
        K = w.shape[-1]
        if ctx is None:
            xf = x.float()
            win = torch.stack([F.pad(xf, (0, 0, K - 1 - j, 0))[: xf.shape[0]] for j in range(K)], dim=1)
        else:
            positions, slot_mapping, table = ctx
            cache = getattr(layer, which + "_hist")
            cache.index_put_((slot_mapping,), x)
            prev = positions.unsqueeze(-1) - torch.arange(K - 1, 0, -1, device=x.device)  # [T, K - 1]
            hist = cache[self._slots(prev.clamp(min=0), table)]
            hist = torch.where((prev >= 0).unsqueeze(-1), hist, torch.zeros_like(hist))
            win = torch.cat([hist, x.unsqueeze(1)], dim=1).float()  # [T, K, C]
        return ((win * w.t().unsqueeze(0)).sum(dim=1) + x.float()).to(x.dtype)

    def _slots(self, pos: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        """Cache slots of positions pos [T, n] (non-negative) through a block table: [P] for one
        sequence's chunk, [B, P] for B sequences whose T = B * Q rows are in order."""
        ps = self.page_size
        if table.dim() == 1:
            pages = table[pos // ps]
        else:
            B = table.shape[0]
            pages = torch.gather(table, 1, (pos // ps).reshape(B, -1)).reshape(pos.shape)
        return pages * ps + pos % ps

    def _rel_logits(self, layer: DecoderLayer, x: torch.Tensor, positions: torch.Tensor, L: int):
        """Inkling's relative position logits [T, nh, L] for keys at positions 0 .. L - 1
        (InklingRelativeLogits: r_proj(x) mixes the d_rel profiles of rel_logits_proj into one
        bias per backward distance, zero outside 0 <= distance < rel_extent), and the log scaling
        tau [T] of global layers (1 + alpha * log(max((position + 1) / n_floor, 1))), or None."""
        T, E = x.shape[0], layer.spec.rel_extent
        r = F.linear(x, self._w(layer, "rel_q")).view(T, layer.nh, -1).float()
        rel = torch.einsum("thd,de->the", r, layer.rel_proj.float())  # [T, nh, E]
        dist = positions.unsqueeze(-1) - torch.arange(L, device=x.device)  # [T, L]
        b = torch.gather(rel, 2, dist.clamp(0, E - 1).unsqueeze(1).expand(T, layer.nh, L))
        b = torch.where(((dist >= 0) & (dist < E)).unsqueeze(1), b, torch.zeros_like(b))
        tau = None
        if layer.spec.log_scaled and self.cfg.log_scaling is not None:
            n_floor, alpha = self.cfg.log_scaling
            tau = 1.0 + alpha * torch.log(((positions + 1).float() / n_floor).clamp(min=1.0))
        return b, tau

    # Up to this many (token, expert) pairs, a MoE layer gathers just the selected experts'
    # weights (decode: the step is bound by reading them anyway); above it, every expert
    # runs over every token and the routing weights zero the rest (prefill). Both are exact.
    MOE_GATHER_MAX_PAIRS = int(os.environ.get("KILN_MOE_GATHER_MAX_PAIRS", 512))
    # With the NKI MoE kernel (w_blob layers on the device) every call runs the kernel, on
    # chunks of tokens with at most this many pairs each (moe_decode.moe_selected). The XLA paths
    # cannot read the blob there: LNL rejects the in-graph uint8 -> float8 view moe_decode.unpack
    # needs ("Expected XLA tensor. Got: XLAFloat8_e4m3fnType", SDK 2.32, 2026-10-03), so they
    # serve blob layers on the CPU only.
    MOE_KERNEL_MAX_PAIRS = int(os.environ.get("KILN_MOE_KERNEL_MAX_PAIRS", 512))

    def _route(self, layer: DecoderLayer, x: torch.Tensor):
        cfg = self.cfg
        k = cfg.num_experts_per_tok
        bias = getattr(layer, "router_logit_bias", None)
        bias = bias.float() if bias is not None else None
        logits = F.linear(x.float(), layer.router.float(), bias)  # [T, E] fp32
        if cfg.router_scoring == "topk_softmax":
            # gpt-oss (GptOssTopKRouter): softmax over the k largest logits only.
            topv, topi = torch.topk(logits, k, dim=-1)
            topv = torch.softmax(topv, dim=-1)
        elif cfg.router_scoring == "sigmoid":
            # DeepSeek-V3 style (MiMo-V2 noaux_tc): select on score + correction bias, weight
            # by the uncorrected score, renormalise, scale.
            scores = logits.sigmoid()
            choice = scores + layer.router_bias if layer.router_bias is not None else scores
            if cfg.n_group > 1:  # DeepSeek-V3 node-limited routing
                choice = _mla.group_limited(choice, cfg.n_group, cfg.topk_group)
            _, topi = torch.topk(choice, k, dim=-1)
            if TOPK_CLAMP:  # trn2: a NaN row's indices are 0xFFFFFFFF, and the gather below then faults
                topi = topi.clamp(0, choice.shape[-1] - 1)
            topv = torch.gather(scores, 1, topi)
            if cfg.norm_topk_prob and k > 1:
                topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-20)
            topv = topv * cfg.routed_scaling_factor
        else:
            probs = torch.softmax(logits, dim=-1)
            topv, topi = torch.topk(probs, k, dim=-1)
            if cfg.norm_topk_prob:
                topv = topv / topv.sum(dim=-1, keepdim=True)
        return topv.to(self.dtype), topi

    def _experts(self, layer: DecoderLayer, name: str, idx: torch.Tensor | None = None) -> torch.Tensor:
        """Expert weights [E or len(idx), ...] in the model dtype: w_gu as [out, in], w_down as
        [in, out] (Im, H). Packed MXFP4 (uint8) or FP8 + scale are dequantized."""
        if layer.moe_blob and getattr(layer, "moe_ep", False):  # kernels/moe_ep.py's layout, read back
            from ..kernels import moe_ep

            w_gu, s_gu, w_down, s_down = moe_ep.unpack(self._ep_blob(layer))
            w, sc = (w_gu, s_gu) if name == "w_gu" else (w_down, s_down)
            if idx is not None:
                w, sc = w[idx], sc[idx]
            return dequant(w, sc, self.dtype) if name == "w_gu" else dequant_t(w, sc, self.dtype)
        if layer.moe_blob:
            from ..kernels import moe_decode, moe_dedupe

            lay = moe_dedupe if getattr(layer, "moe_tiles", False) else moe_decode
            if name == "w_gu":
                return dequant(*lay.unpack_gu(layer.w_blob, self.cfg.hidden_size, idx), self.dtype)
            return dequant_t(*lay.unpack_down(layer.w_blob, self.cfg.hidden_size, idx), self.dtype)
        w, sc = getattr(layer, name), getattr(layer, name + "_scale")
        if idx is not None:
            w, sc = w[idx], (sc[idx] if sc is not None else None)
        if w.dtype == torch.uint8:
            out = dequant_mxfp4(w, sc, self.dtype)
            return out.transpose(-1, -2) if name == "w_down" else out
        if name == "w_down" and layer.down_t:
            return dequant_t(w, sc, self.dtype)
        return dequant(w, sc, self.dtype)

    def _expert_act(self, layer: DecoderLayer, gu: torch.Tensor, idx: torch.Tensor | None) -> torch.Tensor:
        """The GLU of gate_up outputs gu (gate half first): [T*k, 2Im] for the experts idx, or
        [E, T, 2Im] for every expert (idx None), with gpt-oss's expert biases when present."""
        bias = getattr(layer, "w_gu_bias", None)  # (tools and tests build bare layer namespaces)
        if bias is not None:
            gu = gu + (bias[idx] if idx is not None else bias.unsqueeze(1))
        Im = gu.shape[-1] // 2
        if getattr(self.cfg, "moe_act", "silu") == "swiglu_oai":
            return swiglu_oai(gu[..., :Im], gu[..., Im:], self.cfg.swiglu_limit)
        return F.silu(gu[..., :Im]) * gu[..., Im:]

    def _moe(self, layer: DecoderLayer, x: torch.Tensor) -> torch.Tensor:
        topv, topi = self._route(layer, x)
        return self._moe_routed(layer, x, topv, topi)

    @staticmethod
    def _ep_blob(layer: DecoderLayer) -> dict:
        """An expert-parallel layer's kernels/moe_ep.py layout (DecoderLayer.pack_experts)."""
        names = getattr(layer, "ep_names", ("gu", "sgu", "dn", "sdn", "dsg", "dsd"))
        return {n: getattr(layer, "ep_" + n) for n in names}

    def _moe_ep(self, layer: DecoderLayer, x: torch.Tensor, topv: torch.Tensor, topi: torch.Tensor, act: int,
                lim: float = 0.0, phys: bool = False) -> torch.Tensor:
        """An expert-parallel layer's routed output (moe_ep_enabled): sum over the pairs whose expert is this rank's
        of topv expert(x), 0 for the rest (the block's all-reduce adds the ranks). act 0 SiLU, 1 SiLU clamped at
        lim (kernels/moe_dedupe.ACTS). On the device kernels/moe_ep.py; on the host every local expert over every
        row, weighted by the routing weight of its pair (0 where the token is not routed to it). phys: topi holds
        physical expert ids already (models/eplb.py); a layer with redundant slots maps routing ids here otherwise."""
        if getattr(layer, "ep_s", 0) and not phys:
            from ..kernels.moe_ep import uses_small

            topi = self._ep_phys(layer, topi, decode=uses_small(x.shape[0]))
        if layer.moe_blob and x.device.type != "cpu":
            from ..kernels.moe_ep import moe_ep

            return moe_ep(x, topv, topi, self._ep_blob(layer), layer.ep_lmap, act, lim)
        T = x.shape[0]
        El = layer.ep_count
        loc = layer.ep_lmap.view(-1)[topi].long()  # [T, k]: local expert, El for another rank's
        w = torch.zeros(T, El + 1, dtype=torch.float32, device=x.device).scatter_add(1, loc, topv.float())[:, :El]
        gu = torch.einsum("th,eih->eti", x, self._experts(layer, "w_gu"))  # [El, T, 2I]
        I = gu.shape[-1] // 2
        g, u = gu[..., :I], gu[..., I:]
        if act == 1:
            g, u = g.clamp(max=lim), u.clamp(min=-lim, max=lim)
        ye = torch.einsum("eti,eih->eth", F.silu(g) * u, self._experts(layer, "w_down"))  # [El, T, H]
        return torch.einsum("eth,te->th", ye, w.to(ye.dtype))

    def _moe_routed(self, layer: DecoderLayer, x: torch.Tensor, topv: torch.Tensor, topi: torch.Tensor):
        """sum_k topv[t, k] * expert_{topi[t, k]}(x[t]) for routing weights / experts [T, k]."""
        if getattr(layer, "moe_ep", False):
            return self._moe_ep(layer, x, topv, topi, 0)
        cfg = self.cfg
        T, H = x.shape
        k, Im = cfg.num_experts_per_tok, self.moe_inter
        if layer.moe_blob and x.device.type != "cpu":
            if getattr(layer, "moe_tiles", False):
                if MOE_PREFILL_KERNEL == "nki" and T >= MOE_PREFILL_MIN_TOKENS:
                    from ..kernels.moe_prefill import moe_prefill

                    return moe_prefill(x, topv, topi, layer.w_blob, dq=layer.moe_prefill_dq,
                                       down=moe_prefill_down(layer))
                from ..kernels.moe_dedupe import moe_dedupe

                return moe_dedupe(x, topv, topi, layer.w_blob)
            from ..kernels.moe_decode import moe_selected

            return moe_selected(x, topv, topi, layer.w_blob, self.MOE_KERNEL_MAX_PAIRS)
        if T * k <= self.MOE_GATHER_MAX_PAIRS:
            flat = topi.reshape(T * k)
            xs = x.unsqueeze(1).expand(T, k, H).reshape(T * k, H, 1)
            # Only the selected experts' weights are gathered, and only they are dequantized.
            gu = torch.bmm(self._experts(layer, "w_gu", flat), xs).squeeze(-1)  # [T*k, 2Im]
            a = self._expert_act(layer, gu, flat)
            y = torch.bmm(a.unsqueeze(1), self._experts(layer, "w_down", flat)).squeeze(1)  # [T*k, H]
            if getattr(layer, "w_down_bias", None) is not None:
                y = y + layer.w_down_bias[flat]
            return (y.view(T, k, H) * topv.unsqueeze(-1)).sum(dim=1)
        gu = torch.einsum("th,eih->eti", x, self._experts(layer, "w_gu"))  # [E, T, 2Im]
        a = self._expert_act(layer, gu, None)
        ye = torch.einsum("eti,eih->eth", a, self._experts(layer, "w_down"))  # [E, T, H]
        if getattr(layer, "w_down_bias", None) is not None:
            ye = ye + layer.w_down_bias.unsqueeze(1)
        w = torch.zeros(T, cfg.num_experts, dtype=self.dtype, device=x.device).scatter(1, topi, topv)
        return torch.einsum("eth,te->th", ye, w)

    def _gather(self, cache: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        """Paged cache [num_pages * page_size, nkv, D] gathered by a block table [..., P] into
        [..., P * page_size, nkv, D], either by whole pages or by token slots (see GATHER)."""
        ps = self.page_size
        tokens = block_table.shape[-1] * ps
        if GATHER == "token" or (GATHER == "auto" and tokens < PAGE_GATHER_MIN_TOKENS):
            offs = torch.arange(ps, device=block_table.device, dtype=block_table.dtype)
            return cache[(block_table.unsqueeze(-1) * ps + offs).flatten(-2)]
        pages = cache.view(-1, ps, cache.shape[1], cache.shape[2])[block_table]  # [..., P, ps, nkv, D]
        return pages.flatten(-4, -3)

    @staticmethod
    def _bias(visible: torch.Tensor) -> torch.Tensor:
        return torch.where(visible, 0.0, NEG_INF)

    # -- one layer over a batch, shared by every forward ------------------------------
    #
    # Three batch forms, told apart by the block table and the bias:
    #   decode  table [B, P], bias [B, 1, 1, L]     h [B, H]       (one token per sequence)
    #   chunk   table [P],    bias [1, 1, C, L]     h [C, H]       (one sequence, C tokens)
    #   extend  table [B, P], bias [B, 1, 1, Q, L]  h [B * Q, H]   (B sequences x Q tokens)
    # A layer's table and bias are its attention kind's: sliding-window layers get the
    # window-sized ones (see _attn_inputs).

    def _layer(self, layer, h, positions, slot_mapping, table, bias, state_slot=None, mixed=None):
        """state_slot: each sequence's row in the recurrent-state pool, read by linear-attention
        layers (models/linear_attn.py); attention layers ignore it. mixed: the decode rows of a mixed
        batch (forward_mixed): (their block tables [D, P], biases [D, 1, 1, L], state rows [D] or
        None); h then holds a prefill chunk's rows followed by D decode rows (per DP-attention group),
        positions and slot_mapping cover both, table / bias / state_slot are the chunk's."""
        if self.cfg.hybrid is not None and not layer.plain:
            return _hybrid.layer(self, layer, h, positions, slot_mapping, table, bias, state_slot, mixed=mixed)
        if isinstance(layer.spec, LinearSpec):
            return self._mlp(layer, linear_attn.mixer(self, layer, h, positions, slot_mapping, state_slot, mixed))
        ctx = (positions, slot_mapping, table)  # Inkling's short convolutions read their histories through it
        return self._mlp(layer, self._attention(layer, h, positions, slot_mapping, table, bias, mixed), ctx)

    def _attention(self, layer, h, positions, slot_mapping, table, bias, mixed=None):
        """h plus the attention block: norm, qkv, KV write and read, attention, o_proj, all-reduce.
        mixed (_layer): the chunk's rows and the decode rows each run their own batch form, from the
        same input, and one all-reduce takes both."""
        cfg, sp = self.cfg, layer.spec
        x = rms_norm(self._attn_in(h), layer.in_norm, cfg.rms_norm_eps)  # DP attention: this group's rows
        if sp.mla is not None:
            return h + self._attn_all_reduce(_mla.attention(self, layer, x, positions, slot_mapping, table, bias,
                                                            mixed=mixed))
        if mixed is not None:
            if layer.a_sconv is not None or layer.k_sconv is not None or layer.rel_proj is not None:
                raise NotImplementedError("mixed batches for Inkling's short convolutions / relative logits")
            C = mixed_chunk_rows(x, mixed)
            out = torch.cat([self._gqa(layer, x[:C], positions[:C], slot_mapping[:C], table, bias),
                             self._gqa(layer, x[C:], positions[C:], slot_mapping[C:], mixed[0], mixed[1])])
            return h + self._attn_all_reduce(out)
        ctx = (positions, slot_mapping, table)
        out = self._attn_all_reduce(self._gqa(layer, x, positions, slot_mapping, table, bias))
        return h + (self._sconv(layer, "a", out, ctx) if layer.a_sconv is not None else out)

    def _gqa(self, layer, x, positions, slot_mapping, table, bias):
        """The attention block of a (GQA / MHA) layer on its normalised input x, one batch form: the
        output projection before its all-reduce."""
        sp = layer.spec
        ctx = (positions, slot_mapping, table)
        q, k, v = self._qkv(layer, x, positions, ctx)
        self._store(layer.k_cache, slot_mapping, k)
        self._store(layer.v_cache, slot_mapping, v)
        T = x.shape[0]
        if (table.dim() == 1 and _segmented_attn.ENABLED and x.device.type != "cpu"
                and _segmented_attn.eligible(layer, T, self.page_size, layer.k_cache.dtype, self.fp8_max is not None)):
            # KILN_ATTN_PREFILL=segmented: the chunk attends the real context in segments (kernels/segmented_attn.py)
            sc = sp.scale if sp.scale is not None else sp.head_dim ** -0.5
            o = _segmented_attn.attend(q.reshape(T, layer.nh, sp.head_dim), layer.k_cache, layer.v_cache, table,
                                       positions[:1], self.page_size, sc)
            o = o.reshape(T, layer.nh * sp.v_head_dim)
            if layer.o_gate is not None:
                o = o * torch.sigmoid(F.linear(x, self._w(layer, "o_gate")))
            return F.linear(o, self._w(layer, "o"), layer.o_bias)
        kc = self._load(layer.k_cache, table)  # [(B,) L, nkv, Dk]
        vc = self._load(layer.v_cache, table)
        rel = self._rel_logits(layer, x, positions, kc.shape[-3]) if layer.rel_proj is not None else (None, None)
        o = self._attend(layer, q, kc, vc, table, bias, *rel)
        T = x.shape[0]
        o = o.reshape(T, layer.nh * sp.v_head_dim)
        if layer.o_gate is not None:
            o = o * torch.sigmoid(F.linear(x, self._w(layer, "o_gate")))
        return F.linear(o, self._w(layer, "o"), layer.o_bias)

    def _scores(self, layer, qk: torch.Tensor, bias, rel, tau, layout: str) -> torch.Tensor:
        """fp32 scores of one batch form: qk * scale (+ relative position logits) (* tau) + bias.
        rel [T, nh, L] and tau [T] are per query row; layout names the axes of qk."""
        D = layer.spec.head_dim
        s = qk.float() * (layer.spec.scale if layer.spec.scale is not None else D ** -0.5)
        if rel is not None:
            nkv, G = layer.nkv, layer.nh // layer.nkv
            if layout == "hgcl":  # chunk: T = C
                rel, tau = rel.view(-1, nkv, G, rel.shape[-1]).permute(1, 2, 0, 3), tau
                tshape = (1, 1, -1, 1)
            elif layout == "bhgql":  # extend: T = B * Q
                B, Q = s.shape[0], s.shape[3]
                rel = rel.view(B, Q, nkv, G, -1).permute(0, 2, 3, 1, 4)
                tshape = (B, 1, 1, Q, 1)
            else:  # decode
                rel = rel.view(-1, nkv, G, rel.shape[-1])
                tshape = (-1, 1, 1, 1)
            s = s + rel
            if tau is not None:
                s = s * tau.view(tshape)
        return s + bias

    def _attend(self, layer, q, kc, vc, table, bias, rel=None, tau=None):
        """Scores, softmax and the weighted sum of values for one batch form (see above)."""
        G, nkv, D = layer.nh // layer.nkv, layer.nkv, layer.spec.head_dim
        T = q.shape[0]
        if table.dim() == 1:  # chunk
            s = self._scores(layer, torch.einsum("chgd,lhd->hgcl", q.view(T, nkv, G, D), kc), bias, rel, tau, "hgcl")
            return torch.einsum("hgcl,lhd->chgd", self._softmax(layer, s, kv_axis=0), vc)
        if bias.dim() == 5:  # extend
            B, Q = table.shape[0], bias.shape[3]
            s = self._scores(layer, torch.einsum("bqhgd,blhd->bhgql", q.view(B, Q, nkv, G, D), kc), bias, rel, tau,
                             "bhgql")
            return torch.einsum("bhgql,blhd->bqhgd", self._softmax(layer, s, kv_axis=1), vc)
        # decode
        s = self._scores(layer, torch.einsum("bhgd,blhd->bhgl", q.view(T, nkv, G, D), kc), bias, rel, tau, "bhgl")
        return torch.einsum("bhgl,blhd->bhgd", self._softmax(layer, s, kv_axis=1), vc)

    def long_ctx(self, L: int) -> bool:
        """Whether a bucket of L keys runs every attention layer's long-context path (models/dsa_long.py; static
        per bucket)."""
        return self.long_dsa and (self.cp > 1 or _dsa_long.enabled(L))

    def _long_attn(self, block_table, shape):
        """_attn_inputs of a long-context bucket: the long path reads the positions and the block table only, so
        the bias is a [*shape, 1] placeholder (its rank tells the batch forms apart, models/mla.py attention). It is
        made from the block table (times 0) so that every page bucket's prep graph is its own: without the table the
        prep graphs of all long buckets were one graph, one cache key loaded once per bucket, and at tp=2 the ranks'
        loaded copies failed the collective barrier ("MPMD execution is not supported. Most likely some ranks
        recompiled/reloaded a graph", trn1.2xlarge, SDK 2.32, 2026-10-05)."""
        zero = block_table.reshape(-1)[:1].to(torch.float32) * 0.0
        return {None: (zero.expand(int(torch.Size(shape).numel())).reshape(*shape, 1).contiguous(), block_table)}

    def _attn_inputs(self, vis_full, pos, block_table, swa_table, swa_first, shape):
        """(bias, table) per attention kind: key None for full attention, self.window for
        sliding-window layers. vis_full: visibility over block_table's keys [..., L]; pos:
        query positions broadcastable against it; shape: the view of a bias row set."""
        out = {None: (self._bias(vis_full).view(*shape, -1), block_table)}
        if self.window is not None:
            if swa_table is None:
                L = vis_full.shape[-1]
                j = torch.arange(L, device=block_table.device)
                out[self.window] = (self._bias(vis_full & (j > pos - self.window)).view(*shape, -1), block_table)
            else:
                Lw = swa_table.shape[-1] * self.page_size
                jw = torch.arange(Lw, device=swa_table.device) + swa_first.view(*swa_first.shape, *[1] * (pos.dim() - 1))
                vis = (jw <= pos) & (jw > pos - self.window)
                out[self.window] = (self._bias(vis).view(*shape, -1), swa_table)
        return out

    def _run_layers(self, h, positions, slot_mapping, attn, state_slot=None, mixed=None, aux=None):
        """aux: a list that receives the residual stream once n layers ran, for each n in the EAGLE-3 draft's
        aux_layers (models/eagle3.py; vLLM EagleModelMixin._maybe_add_hidden_state(aux, idx + 1, ...))."""
        want = set(self.eagle.aux_layers) if aux is not None else ()
        if 0 in want:
            aux.append(h)
        for n, layer in enumerate(self.layers, 1):
            bias, table = attn[getattr(layer.spec, "window", None)]
            h = self._layer(layer, h, positions, slot_mapping, table, bias, state_slot, mixed)
            if n in want:
                aux.append(h)
        return h

    def eagle_combine(self, *aux):
        """vLLM llama_eagle3.py combine_hidden_states: fc(cat(aux)) [T, H], the hidden state an EAGLE-3 draft's first
        step reads (no norm_before_fc / fc_norm: the checkpoints read have neither)."""
        return F.linear(torch.cat(aux, dim=-1), self.eagle_fc)

    def _draft_out(self, out, h, aux):
        """A forward's output: the drafting head's hidden states beside it (MTP: the last hidden state; EAGLE-3: the
        combined auxiliary ones)."""
        if self.mtp is None:
            return out
        return (out, self.eagle_combine(*aux)) if self.eagle is not None else (out, h)

    # -- piecewise execution: one compiled graph per KIND of layer ---------------------
    #
    # vLLM compiles its model piecewise (split at attention) and SGLang captures per batch
    # size; here the unit is a whole layer. layer_fn(i) is a pure function of tensors, so
    # every layer of one kind (same static attributes, same tensor shapes) traces to the
    # same graph and shares one NEFF: MiMo-V2.6-Flash has 48 layers but 3 kinds, and its
    # whole-model graph took 2270 s to compile at tp=32 (trn1.32xlarge, 2026-10-02).

    def _layer_split(self, i: int):
        layer = self.layers[i]
        tensors = {n: t for n, t in [*layer._parameters.items(), *layer._buffers.items(),
                                     *vars(layer).items()] if isinstance(t, torch.Tensor)}
        static = {n: v for n, v in vars(layer).items()
                  if not n.startswith("_") and n not in tensors and not isinstance(v, (nn.Module, torch.Tensor))}
        static.update({n: None for n, t in [*layer._parameters.items(), *layer._buffers.items()] if t is None})
        return static, tensors

    def layer_kind(self, i: int) -> tuple:
        static, tensors = self._layer_split(i)
        return (tuple(sorted((n, repr(v)) for n, v in static.items())),
                tuple((n, tuple(t.shape), t.dtype) for n, t in tensors.items()))

    def layer_tensors(self, i: int) -> tuple:
        return tuple(self._layer_split(i)[1].values())

    def group_fn(self, idxs):
        """f(h, positions, slot_mapping, table, bias, table_w, bias_w, *tensors, state_slot=None, mixed=None)
        -> h running layers idxs in order, where tensors is layer_tensors of each in turn and
        (table_w, bias_w) serve the sliding-window layers (state_slot the linear-attention
        ones; mixed the decode rows of a mixed batch, see _layer). Valid for every run of layers with
        the same sequence of kinds."""
        parts = []
        for i in idxs:
            static, tensors = self._layer_split(i)
            parts.append((static, tuple(tensors), getattr(static["spec"], "window", None) is not None))

        def f(h, positions, slot_mapping, table, bias, table_w, bias_w, *ts, state_slot=None, mixed=None):
            o = 0
            for static, names, windowed in parts:
                view = _LayerView(static, dict(zip(names, ts[o : o + len(names)])))
                o += len(names)
                h = self._layer(view, h, positions, slot_mapping, *((table_w, bias_w) if windowed else (table, bias)),
                                state_slot, mixed)
            return h

        return f

    # -- decode: B sequences x 1 token ------------------------------------------------

    def prep_decode(self, input_ids, positions, block_table, context_lens, board, read_slot,
                    swa_table=None, swa_first=None, ngram_ids=None):
        input_ids = torch.where(read_slot >= 0, board[read_slot.clamp(min=0)].long(), input_ids)
        # One group's sequences: input_ids holds every DP-attention group's. (Shapes come from
        # input_ids, not block_table: the order a graph first touches its inputs orders its
        # placeholders, which is part of the compile cache key.)
        B = input_ids.shape[0] // self.dp
        if self.long_ctx(block_table.shape[1] * self.page_size):  # models/dsa_long.py: no [B, L] visibility
            return self._hidden_in(input_ids, ngram_ids, self._sp_on_decode()), self._long_attn(block_table, (B, 1, 1))
        j = torch.arange(block_table.shape[1] * self.page_size, device=block_table.device).unsqueeze(0)
        attn = self._attn_inputs(j < context_lens.unsqueeze(1), positions.unsqueeze(1), block_table,
                                 swa_table, swa_first, (B, 1, 1))
        return self._hidden_in(input_ids, ngram_ids, self._sp_on_decode()), attn

    def post_decode(self, h, temperature, top_p, top_k, min_p, noise, board, write_slot,
                    bitmask=None, penalties=None):
        hn = self._final(h)
        if self._sp_on_decode():  # sequence-parallel decode streams: every rank's rows of the final hidden state
            hn = self._sp_gather(hn)
        out = sample(self._head(hn), temperature, top_p, top_k, min_p, noise, bitmask, penalties)
        board.index_put_((write_slot,), out[:, 0])
        return out

    def forward_decode(self, input_ids, positions, block_table, context_lens, slot_mapping,
                       temperature, top_p, top_k, min_p, noise, board, read_slot, write_slot,
                       bitmask=None, penalties=None, swa_table=None, swa_first=None, state_slot=None,
                       ngram_ids=None):
        """board [S] fp32: each request's last sampled token, kept on the device. A row with
        read_slot >= 0 takes its input token from the board instead of input_ids (overlap
        scheduling: the previous step's token was never read back); every row writes its
        sampled token to board[write_slot].

        swa_table [B, Pw] / swa_first [B]: the pages covering each sequence's last `window`
        positions and the position of the first slot of swa_table[:, 0]. Sliding-window
        layers gather only these, so their KV read is window-sized at any context length.

        state_slot [B]: each sequence's row in the recurrent-state pool (linear-attention
        models; padded rows use the scratch row). ngram_ids [B, *]: the host-computed n-gram table
        rows of a model with a Per-Layer Embedding (models/qwen4_exp.py)."""
        h, attn = self.prep_decode(input_ids, positions, block_table, context_lens, board, read_slot,
                                   swa_table, swa_first, ngram_ids)
        aux = [] if self.eagle is not None else None
        h = self._run_layers(h, positions, slot_mapping, attn, state_slot, aux=aux)
        out = self.post_decode(h, temperature, top_p, top_k, min_p, noise, board, write_slot, bitmask, penalties)
        return self._draft_out(out, h, aux)  # MTP / EAGLE-3 draft from the hidden states

    # -- prefill: one sequence x C tokens ---------------------------------------------

    def prep_prefill(self, input_ids, positions, block_table, swa_table=None, swa_first=None, ngram_ids=None):
        """block_table [P] covers the sequence's whole context, swa_table [Pw] (with swa_first
        [1]) just the pages the chunk's windows reach back to. Under DP attention input_ids holds
        every group's chunk [N * C] and the rest this group's."""
        C = input_ids.shape[0] // self.dp
        if self.long_ctx(block_table.shape[0] * self.page_size):  # models/dsa_long.py: no [C, L] visibility
            attn = self._long_attn(block_table, (1, 1, C))
        else:
            j = torch.arange(block_table.shape[0] * self.page_size, device=block_table.device).unsqueeze(0)
            pos = positions.unsqueeze(1)
            attn = self._attn_inputs(j <= pos, pos, block_table, swa_table, swa_first, (1, 1, C))
        sp = self._sp_on()
        if sp and input_ids.shape[0] % self.tp_size:
            raise ValueError(f"sequence-parallel prefill streams (KILN_PREFILL_SP) need the chunk's {input_ids.shape[0]} "
                             f"rows to divide over tp={self.tp_size} (ModelRunner turns them off for such buckets)")
        return self._hidden_in(input_ids, ngram_ids, sp), attn

    def post_prefill(self, h, last_index, temperature, top_p, top_k, min_p, noise, board, write_slot,
                     bitmask=None, penalties=None, plp_targets=None):
        hn = self._final(h)
        if self._sp_on():  # sequence-parallel streams: every rank's rows of the final hidden state
            hn = self._sp_gather(hn)
        out = sample(self._head(hn.index_select(0, last_index)), temperature, top_p, top_k, min_p, noise,
                     bitmask, penalties)
        board.index_put_((write_slot,), out[:, 0])
        if plp_targets is not None and PLP_VP:
            out = torch.cat([out, self._score_rows(hn, plp_targets)])
        elif plp_targets is not None:
            out = torch.cat([out, score_rows(self._head(hn), plp_targets)])
        return out

    def forward_prefill(self, input_ids, positions, block_table, slot_mapping, last_index,
                        temperature, top_p, top_k, min_p, noise, board, write_slot,
                        bitmask=None, penalties=None, swa_table=None, swa_first=None, plp_targets=None,
                        state_slot=None, ngram_ids=None):
        """plp_targets [C]: when given (prompt logprobs), row i of the chunk also scores token
        plp_targets[i]; the output gains C rows of [rank, logprob, top-N ids, top-N logprobs]
        after the sampled row (see sampler.score_rows). state_slot [1]: the sequence's row in
        the recurrent-state pool (linear-attention models)."""
        h, attn = self.prep_prefill(input_ids, positions, block_table, swa_table, swa_first, ngram_ids)
        aux = [] if self.eagle is not None else None
        h = self._run_layers(h, positions, slot_mapping, attn, state_slot, aux=aux)
        out = self.post_prefill(h, last_index, temperature, top_p, top_k, min_p, noise, board, write_slot,
                                 bitmask, penalties, plp_targets)
        return self._draft_out(out, h, aux)

    # -- mixed: one sequence's prefill chunk plus D decoding sequences -----------------
    #
    # vLLM's chunked prefill runs the decode tokens of the running requests in the same forward as
    # the prefill chunks (vllm 0.24.0 vllm/v1/core/sched/scheduler.py Scheduler.schedule: one
    # token_budget, running requests first, then waiting ones). Here each
    # DP-attention group's rows are its chunk's C rows followed by D decode rows, so the residual
    # stream, the MLP / experts and the hyper-connections run over both at once (one pass over the
    # expert weights, one set of graph launches), while every token mixer runs its chunk form on the
    # first C rows and its decode form on the last D (_layer: mixed), exactly the calls an unmixed
    # step makes, inside one graph.

    def prep_mixed(self, input_ids, positions, block_table, board, read_slot, dec_table, dec_ctx, ngram_ids=None):
        """input_ids [N * (C + D)] every group's rows (a decode row whose token is still on the
        device reads it from the board: read_slot >= 0, as prep_decode), positions [C + D] this group's
        (the chunk's, then the decode rows'), block_table [P] the chunk's, dec_table [D, P] and
        dec_ctx [D] the decode rows' tables and context lengths. Returns the hidden state and the
        chunk's and the decode rows' attention inputs (_attn_inputs)."""
        input_ids = torch.where(read_slot >= 0, board[read_slot.clamp(min=0)].long(), input_ids)
        D = dec_table.shape[0]
        C = positions.shape[0] - D
        if self.window is not None:
            raise NotImplementedError("mixed batches for sliding-window layers")
        j = torch.arange(block_table.shape[0] * self.page_size, device=block_table.device).unsqueeze(0)
        pos = positions[:C].unsqueeze(1)
        attn = self._attn_inputs(j <= pos, pos, block_table, None, None, (1, 1, C))
        jd = torch.arange(dec_table.shape[1] * self.page_size, device=dec_table.device).unsqueeze(0)
        dattn = self._attn_inputs(jd < dec_ctx.unsqueeze(1), positions[C:].unsqueeze(1), dec_table, None, None,
                                  (D, 1, 1))
        sp = self._sp_on()
        if self._mixed_split():  # this rank's rows of the chunks, then every decode row (hybrid.MIXED_SP)
            if (self.dp * C) % self.tp_size:
                raise ValueError(f"sequence-parallel prefill streams (KILN_PREFILL_SP) need the mixed batch's "
                                 f"{self.dp * C} chunk rows to divide over tp={self.tp_size}")
            e = _hybrid.mixed_split_in(self, self._embed(input_ids), C, D)
            return e.repeat(1, self.cfg.hybrid.hc), attn, dattn
        if sp and input_ids.shape[0] % self.tp_size:
            raise ValueError(f"sequence-parallel prefill streams (KILN_PREFILL_SP) need the mixed batch's "
                             f"{input_ids.shape[0]} rows to divide over tp={self.tp_size}")
        return self._hidden_in(input_ids, ngram_ids, sp), attn, dattn

    def _mixed_split(self) -> bool:
        """Whether a mixed batch's streams use the "split" sequence-parallel layout (models/hybrid.py MIXED_SP)."""
        return self._sp_on() and self.cfg.hybrid is not None and _hybrid.MIXED_SP == "split"

    def post_mixed(self, h, positions, dec_ctx, last_index, temperature, top_p, top_k, min_p, noise, board, write_slot,
                   bitmask=None, penalties=None, plp_targets=None):
        """post_prefill for a mixed batch's "split" layout (hybrid.MIXED_SP): every rank's chunk rows gathered
        and put back beside the decode rows before the sampled rows are taken. positions [C + D] and dec_ctx
        [D] only give C and D."""
        D = dec_ctx.shape[0]
        hn = _hybrid.mixed_split_out(self, self._final(h), positions.shape[0] - D, D)
        out = sample(self._head(hn.index_select(0, last_index)), temperature, top_p, top_k, min_p, noise,
                     bitmask, penalties)
        board.index_put_((write_slot,), out[:, 0])
        if plp_targets is not None and PLP_VP:
            out = torch.cat([out, self._score_rows(hn, plp_targets)])
        elif plp_targets is not None:
            out = torch.cat([out, score_rows(self._head(hn), plp_targets)])
        return out

    def forward_mixed(self, input_ids, positions, block_table, slot_mapping, last_index,
                      temperature, top_p, top_k, min_p, noise, board, write_slot, read_slot, dec_table, dec_ctx,
                      bitmask=None, penalties=None, plp_targets=None, state_slot=None, dec_state=None,
                      ngram_ids=None):
        """A prefill chunk and D decoding sequences per DP-attention group in one forward (see above):
        positions / slot_mapping [C + D] (the chunk's rows, then the decode rows'), block_table [P] and
        state_slot [1] the chunk's, dec_table [D, P], dec_ctx [D] and dec_state [D] the decode rows'.
        last_index [S] the rows that sample (every group's chunk row, then its decode rows; per-row
        sampling arguments [S]) and write_slot [S] their board slots: post_prefill over S rows, with
        plp_targets [N * (C + D)] scoring every row as forward_prefill's do."""
        h, attn, dattn = self.prep_mixed(input_ids, positions, block_table, board, read_slot, dec_table, dec_ctx,
                                         ngram_ids)
        h = self._run_layers(h, positions, slot_mapping, attn, state_slot, (dec_table, dattn[None][0], dec_state))
        if self._mixed_split():
            out = self.post_mixed(h, positions, dec_ctx, last_index, temperature, top_p, top_k, min_p, noise, board,
                                  write_slot, bitmask, penalties, plp_targets)
        else:
            out = self.post_prefill(h, last_index, temperature, top_p, top_k, min_p, noise, board, write_slot,
                                    bitmask, penalties, plp_targets)
        return (out, h) if self.mtp is not None else out

    # -- extend: B sequences x Q tokens (speculative verification) --------------------

    def prep_extend(self, input_ids, positions, block_table, swa_table=None, swa_first=None):
        B, Q = input_ids.shape  # every DP-attention group's sequences; the rest are one group's
        j = torch.arange(block_table.shape[1] * self.page_size, device=block_table.device).view(1, 1, -1)
        pos = positions.unsqueeze(-1)
        attn = self._attn_inputs(j <= pos, pos, block_table, swa_table, swa_first, (B // self.dp, 1, 1, Q))
        return self._embed(input_ids.reshape(B * Q)), attn

    def prep_verify(self, input_ids, positions, block_table, swa_table=None, swa_first=None, ngram_ids=None):
        """prep_extend with the model's own input hidden state (hyper-connection streams and n-gram
        rows for models/hybrid.py; the token embedding otherwise), for forward_extend."""
        h, attn = self.prep_extend(input_ids, positions, block_table, swa_table, swa_first)
        if self.cfg.hybrid is not None:
            h = self._hidden_in(input_ids.reshape(-1), ngram_ids)
        return h, attn

    def post_extend(self, h, temperature, top_p, top_k, min_p, noise, u_accept, draft):
        hn = self._final(h)
        out = verify_sample(self._head(hn), temperature, top_p, top_k, min_p, noise, u_accept, draft)
        return out

    def forward_extend(self, input_ids, positions, block_table, slot_mapping,
                       temperature, top_p, top_k, min_p, noise, u_accept, draft,
                       swa_table=None, swa_first=None, state_slot=None, ngram_ids=None):
        """input_ids/positions/slot_mapping [B, Q], draft [B * Q]; per-row sampling params [B * Q];
        noise [B * Q, K]. Every position is scored: returns [B * Q, VERIFY_COLS].

        state_slot [B, 1 + Q] (linear-attention models): each sequence's state row to read, then the
        rows that receive its state after each of the Q positions, so a partly rejected draft
        continues from the state after its last accepted position (models/linear_attn.verify_scan).
        ngram_ids [B * Q, *]: a Per-Layer Embedding's host-computed rows (models/qwen4_exp.py)."""
        h, attn = self.prep_verify(input_ids, positions, block_table, swa_table, swa_first, ngram_ids)
        aux = [] if self.eagle is not None else None
        h = self._run_layers(h, positions.reshape(-1), slot_mapping.reshape(-1), attn, state_slot, aux=aux)
        out = self.post_extend(h, temperature, top_p, top_k, min_p, noise, u_accept, draft)
        return self._draft_out(out, h, aux)

    # -- multi-token prediction drafts --------------------------------------------------

    def forward_mtp(self, input_ids, positions, block_table, slot_mapping, hsrc, hidx, last_index, src_norm,
                    swa_table=None, swa_first=None):
        """One MTP pass over B rows x Q positions, in the target's positions (vLLM's EAGLE /
        MTP convention): row b position j takes input_ids[b, j], the token AFTER that
        position, and hsrc[hidx[b, j]], the normalised hidden state of the model (or of the
        previous MTP pass) AT that position; its KV goes to slot_mapping[b, j]. Returns
        [B, 1] fp32 greedy drafts from position last_index[b] and the MTP hidden there
        [B, H], which seeds the next pass."""
        draft, xl, _ = self._mtp_pass(input_ids, positions, block_table, slot_mapping, hsrc, hidx, last_index,
                                      src_norm, swa_table, swa_first, target=True)
        return draft, xl

    def _mtp_pass(self, input_ids, positions, block_table, slot_mapping, hsrc, hidx, last_index, src_norm,
                  swa_table=None, swa_first=None, share=None, target=False, sp_onehot=None):
        """forward_mtp, plus the DSA mask a later pass may reuse (models/mtp.py: with
        index_share_for_mtp_iteration, `share` is the first pass's; None otherwise). sp_onehot (the
        sp_onehot buffer): hsrc holds only this rank's rows of a sequence-parallel prefill chunk
        (prefill_sp), and hidx indexes the chunk's rows in full."""
        cfg = self.cfg
        B, Q = input_ids.shape
        if self.eagle is not None:
            return self._eagle_pass(input_ids, positions, block_table, slot_mapping, hsrc, hidx, last_index,
                                    swa_table, swa_first, target, sp_onehot)
        e, attn = self.prep_extend(input_ids, positions, block_table, swa_table, swa_first)
        # hsrc holds UNNORMALISED hidden states (the target's last layer output, or the previous
        # MTP pass's) and src_norm is their final norm (model.norm or mtp_norm): returning normed
        # states from the post graphs made neuronx-cc 2.27 fail with NCC_IBIR243 on the tp=32
        # verify graph (tools/probe_verify_head.py, 2026-10-03).
        if sp_onehot is not None:  # every rank's rows of the chunk's final hidden state, as post_prefill gathers them
            h_prev = self._sp_gather(self._final(hsrc), sp_onehot).index_select(0, hidx.reshape(-1))
        else:
            h_prev = hsrc.index_select(0, hidx.reshape(-1))
            if target and cfg.hybrid is not None:  # the target's streams, collapsed and normed (models/hybrid.py final)
                h_prev = self._final(h_prev)
            else:
                h_prev = rms_norm(h_prev, src_norm, cfg.rms_norm_eps)
        x = F.linear(torch.cat([rms_norm(e, self.mtp_enorm, cfg.rms_norm_eps),
                                rms_norm(h_prev, self.mtp_hnorm, cfg.rms_norm_eps)], dim=-1), self.mtp_eh)
        bias, table = attn[self.mtp.spec.window]
        rows = torch.arange(B, device=input_ids.device) * Q + last_index
        if self.mtp_index_share:
            x, top = _mtp.mla_layer(self, self.mtp, x, positions.reshape(-1), slot_mapping.reshape(-1), table, bias,
                                    share)
            share = _mtp.shared_rows(bias, top, rows) if share is None else share
        else:
            x = self._layer(self.mtp, x, positions.reshape(-1), slot_mapping.reshape(-1), table, bias)
        xl = x.index_select(0, rows)
        hm = rms_norm(xl, self.mtp_norm, cfg.rms_norm_eps)
        _, top = topk_large(self._head(hm), 1)
        return top[:, :1].float(), xl, share

    def _eagle_pass(self, input_ids, positions, block_table, slot_mapping, hsrc, hidx, last_index, swa_table=None,
                    swa_first=None, target=False, sp_onehot=None):
        """_mtp_pass for an EAGLE-3 draft (models/eagle3.py: vLLM v0.24.0 llama_eagle3.py LlamaDecoderLayer.forward
        with layer_idx 0, LlamaModel.forward, Eagle3LlamaForCausalLM.compute_logits). hsrc: the target call's
        combined auxiliary states (eagle_combine) on the first pass (target), the previous pass's prenorm output on a
        later one. Returns ([B, 1] fp32 drafts in the target's vocabulary, the prenorm hidden at last_index, None)."""
        cfg, d, layer = self.cfg, self.eagle, self.mtp
        B, Q = input_ids.shape
        e, attn = self.prep_extend(input_ids, positions, block_table, swa_table, swa_first)
        if self.eagle_embed is not None:
            e = self._embed(input_ids.reshape(B * Q), self.eagle_embed)
        if sp_onehot is not None:  # every rank's rows of a sequence-parallel prefill chunk's combined states
            h = self._sp_gather(hsrc, sp_onehot).index_select(0, hidx.reshape(-1))
        else:
            h = hsrc.index_select(0, hidx.reshape(-1))
        eps = d.layer.rms_norm_eps
        if d.norm_before_residual:
            hn = rms_norm(h, layer.hidden_norm, eps)
            residual = hn
        else:
            residual = h
            hn = rms_norm(h, layer.hidden_norm, eps)
        x = torch.cat([rms_norm(e, layer.in_norm, eps), hn], dim=-1)  # [B * Q, 2H]
        bias, table = attn[layer.spec.window]
        p, sl = positions.reshape(-1), slot_mapping.reshape(-1)
        h1 = residual + self._attn_all_reduce(self._gqa(layer, x, p, sl, table, bias))
        y = self._mlp(layer, h1)  # post_attention_layernorm, MLP, residual: the prenorm output
        rows = torch.arange(B, device=input_ids.device) * Q + last_index
        yl = y.index_select(0, rows)
        logits = F.linear(rms_norm(yl, self.mtp_norm, eps), self.eagle_head).float()  # [B, Vd], replicated
        # argmax, not topk_large: topk_large's index arithmetic (j % chunk) next to the d2t gather failed neuronx-cc
        # 2.27 with NCC_ILSM901 "LegalizeSundaMacro assertion error: Cannot split" on the remainder (trn1.2xlarge,
        # Qwen3-8B TP=2, B=1, Q=1 and 4, 2026-10-07); the offsets are gathered as fp32 (exact: |d2t| < 2^24).
        top = logits.argmax(dim=-1, keepdim=True)  # [B, 1]
        draft = top.float() + self.eagle_d2t.float().gather(0, top.view(-1)).view(-1, 1)
        return draft, yl, None

    def forward_mtp_k(self, input_ids, positions, block_table, slot_mapping, hsrc, hidx, last_index, src_norm,
                      pos_rest=None, slot_rest=None, swa_table=None, swa_first=None, swa_rest=None,
                      swa_first_rest=None, sp_onehot=None):
        """forward_mtp, then k - 1 more single-position passes in the SAME graph, each fed the
        previous pass's draft and MTP hidden (the recursion ModelRunner.mtp_drafts used to run
        as k graphs). pos_rest / slot_rest [B, k - 1] are the later passes' positions and KV
        slots, swa_rest [B, k - 1, Pw] / swa_first_rest [B, k - 1] their window tables, all
        known on the host in advance. One graph instead of k matters on multi-chip TP, where
        every graph execution holding a cross-chip collective costs about 5 ms at tp=32
        (docs/neuron-notes.md). Returns [B, k] fp32 drafts and the last pass's MTP hidden. sp_onehot:
        the hidden states are a sequence-parallel prefill chunk's, this rank's rows only (_mtp_pass)."""
        draft, xl, share = self._mtp_pass(input_ids, positions, block_table, slot_mapping, hsrc, hidx, last_index,
                                          src_norm, swa_table, swa_first, target=True, sp_onehot=sp_onehot)
        if pos_rest is None:
            return draft, xl
        B = draft.shape[0]
        rows = torch.arange(B, device=draft.device).view(B, 1)
        zero = torch.zeros(B, dtype=last_index.dtype, device=draft.device)
        drafts = [draft]
        for s in range(pos_rest.shape[1]):
            sw = (swa_rest[:, s], swa_first_rest[:, s]) if swa_rest is not None else (None, None)
            draft, xl, _ = self._mtp_pass(draft.long(), pos_rest[:, s : s + 1], block_table, slot_rest[:, s : s + 1],
                                          xl, rows, zero, self.mtp_norm, *sw, share=share)
            drafts.append(draft)
        return torch.cat(drafts, dim=1), xl

    # -- reference: logits for a whole sequence, no cache (tests only) -----------------

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.hybrid is not None:
            return _hybrid.forward_logits(self, input_ids)
        T = input_ids.shape[0]
        positions = torch.arange(T, device=input_ids.device)
        j = positions.unsqueeze(0)
        causal = j <= positions.unsqueeze(1)
        h = self._embed(input_ids)
        state: dict = {}  # MLA: the last DSA indexer's top-k, for shared layers
        for layer in self.layers:
            sp = layer.spec
            if isinstance(sp, LinearSpec):  # from zero state, nothing stored
                h = self._mlp(layer, linear_attn.mixer(self, layer, h, positions, None, None))
                continue
            if sp.mla is not None:
                x = rms_norm(h, layer.in_norm, cfg.rms_norm_eps)
                h = self._mlp(layer, h + self._attn_all_reduce(_mla.reference(self, layer, x, positions, state)))
                continue
            vis = causal if sp.window is None else causal & (j > positions.unsqueeze(1) - sp.window)
            bias = self._bias(vis).view(1, 1, T, T)
            x = rms_norm(h, layer.in_norm, cfg.rms_norm_eps)
            q, k, v = self._qkv(layer, x, positions)
            G = layer.nh // layer.nkv
            qg = q.view(T, layer.nkv, G, sp.head_dim)
            rel = self._rel_logits(layer, x, positions, T) if layer.rel_proj is not None else (None, None)
            s = self._scores(layer, torch.einsum("chgd,lhd->hgcl", qg, k), bias, *rel, "hgcl")
            pr = self._softmax(layer, s, kv_axis=0)
            o = torch.einsum("hgcl,lhd->chgd", pr, v).reshape(T, layer.nh * sp.v_head_dim)
            if layer.o_gate is not None:
                o = o * torch.sigmoid(F.linear(x, self._w(layer, "o_gate")))
            out = self._attn_all_reduce(F.linear(o, self._w(layer, "o"), layer.o_bias))
            h = h + (self._sconv(layer, "a", out, None) if layer.a_sconv is not None else out)
            h = self._mlp(layer, h)
        return self._logits(h)


class _LayerView:
    """A DecoderLayer's static attributes plus tensors passed in as graph inputs."""

    def __init__(self, static: dict, tensors: dict):
        for d in (static, tensors):  # setattr, not __dict__.update: dynamo traces this
            for k, v in d.items():
                setattr(self, k, v)


Qwen3ForCausalLM = DecoderForCausalLM  # the first architecture Kiln ran; kept for imports

# Imported last: models/mla.py and models/hybrid.py import this module's helpers.
from . import hybrid as _hybrid  # noqa: E402
from . import mla as _mla  # noqa: E402
from . import eplb as _eplb  # noqa: E402
from . import mtp as _mtp  # noqa: E402
from ..kernels import nkilib_dense as _nkilib_dense  # noqa: E402
from . import dsa_long as _dsa_long  # noqa: E402
from ..kernels import segmented_attn as _segmented_attn  # noqa: E402
