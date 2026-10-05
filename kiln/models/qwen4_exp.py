"""Qwen3.8-Flash-Next (`qwen4_exp`, Qwen4ExpForConditionalGeneration): the text model.

48 layers = 12 x (3 Gated DeltaNet + 1 Qwen Sparse Attention), every layer MoE (512 experts, top
10, softmax routing) plus a sigmoid-gated shared expert; the residual is 4 streams mixed by
gated-residual hyper-connections (models/hybrid.py), which also stand in for the layer norms
(the block input is the streams' gated mean of their per-stream RMSNorms; there is no
input_layernorm, post_attention_layernorm or final norm). One layer (ple_layer_ids [2], one-indexed)
first adds a Per-Layer Embedding: hashed bigram / trigram embeddings of the token ids. The MTP head
and the vision tower are not loaded.

Numerics follow transformers v5.18.0 src/transformers/models/qwen4_exp/modeling_qwen4_exp.py
(https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py),
config keys its configuration_qwen4_exp.py; the checkpoint's config.json:
https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json

Qwen Sparse Attention (Qwen4ExpTextAttention + Qwen4ExpTextQSAIndexer): the attention is Qwen3.5's
gated attention (per-head [q | gate] rows in q_proj, (1 + w) QK-norm, partial RoPE 64 of 256),
masked to a per-query token selection. A one-key-head indexer (index_qk_proj) gives each token a
raw key and each query Hi heads, (1 + w)-normalised and rotated. Keys are averaged in blocks of
indexer_compress_ratio = 4 consecutive visible tokens, normalised (k_layernorm) and rotated at the
block's first position; a query scores each of its complete blocks by sum_h relu(q_h . k_b) /
sqrt(Di), keeps the indexer_budget / 4 best and always its incomplete tail. Below indexer_budget
tokens that is every visible token, so a bucket at most that long runs plain attention and only
stores the raw keys (an auxiliary per-token cache, aux_kv_shapes). The selection itself is
models/glm5_next.block_mask.

The Per-Layer Embedding's n-gram ids are integer hashes (64-bit multiplies and XORs of the last
ngram_size token ids, reset at EOS) that the device cannot compute exactly, so the HOST computes
them (ngram_ids below, for every row of a graph call: Kiln's scheduler holds every token id) and
passes them in; overlap scheduling, whose next input token is still on the device, is off for these
models (engine._overlap_ok). The tables are looked up in the prep graph and carried to the PLE
layer as a trailing block of the hidden state; their columns are sharded over tensor-parallel ranks
(each head's embedding split by columns, key_proj / value_proj by the matching input columns, one
all-reduce), so a rank of the real model holds 102 GB / tp of them.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..config import AttnSpec, LinearSpec, ModelConfig
from .decoder import NEG_INF, rotate_half

ARCHITECTURES = ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")
# How QSA picks its blocks (glm5_next.block_mask): measured on trn1.2xlarge (SDK 2.32, fp32, one
# QSA layer of a random Qwen3.8-Flash-Next config, decode B=8 over 128 keys, tools/probe_hybrid.py
# qsa, 2026-10-03), torch.topk 1.81 ms per layer against 0.44 ms for the bisection ("range"; the
# exact selections "nki" / "bisect" are block_mask's other values).
SELECT = os.environ.get("KILN_QSA_SELECT", "range")


@dataclass(frozen=True)
class QSASpec(AttnSpec):
    """A gated-attention layer with Qwen Sparse Attention's token indexer."""

    index_heads: int = 0  # indexer_n_heads (indexer_kv_heads must be 1)
    index_head_dim: int = 0
    budget: int = 0  # indexer_budget, tokens
    compress: int = 0  # indexer_compress_ratio, tokens per block

    @property
    def aux_kv_width(self) -> int:  # the raw indexer key cached per token (config.kv_bytes_per_token)
        return self.index_head_dim


@dataclass(frozen=True)
class PLESpec:
    """Per-Layer Embedding (Qwen4ExpTextPLELayer / Qwen4ExpTextNGramEmbedding)."""

    layers: tuple[int, ...]  # 0-indexed decoder layers (config ple_layer_ids are one-indexed)
    embed_dim: int  # ple_embed_dim: all n-gram heads' embeddings concatenated
    ngram: int  # ngram_size; also the dilation of the short conv
    heads_per_ngram: int
    conv_kernel: int  # ple_conv_kernel_size
    eos: int  # the text config's eos_token_id (its first entry)
    head_vocab: tuple[tuple[int, ...], ...]  # per PLE layer, per head: a prime table size
    head_offset: tuple[tuple[int, ...], ...]
    multipliers: tuple[tuple[int, ...], ...]  # per PLE layer: one odd 63-bit multiplier per n-gram position
    rows: tuple[int, ...]  # per PLE layer: table rows, padded to make_ngram_vocab_size_divisible_by
    split_parts: int  # split_ngram_parts: checkpoint shards of each table (dim 0)

    @property
    def heads(self) -> int:
        return (self.ngram - 1) * self.heads_per_ngram

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.heads

    @property
    def conv_state_len(self) -> int:  # Qwen4ExpTextPLELayer.short_conv_state_len
        return (self.conv_kernel - 1) * self.ngram


# -- config.json ------------------------------------------------------------------------------


def _act(name: str) -> str:
    name = {"swish": "silu"}.get(name, name)
    if name not in ("silu", "sigmoid"):
        raise NotImplementedError(f"output gate activation {name!r}")
    return name


def config_from_hf(cls, c: dict, eos_ids: tuple[int, ...]) -> ModelConfig:
    from . import linear_attn
    from .hybrid import HybridSpec

    t = c.get("text_config") or c
    n = t["num_hidden_layers"]
    types = t.get("layer_types") or ["linear_attention" if (i + 1) % t.get("full_attention_interval", 4)
                                     else "full_attention" for i in range(n)]
    if t.get("hidden_act", "silu") != "silu":
        raise NotImplementedError(f"hidden_act {t.get('hidden_act')!r}")
    rope = dict(t.get("rope_parameters") or {})
    if rope.get("rope_type", "default") != "default":
        raise NotImplementedError(f"rope type {rope.get('rope_type')!r}")
    # The rope is "mrope" (three interleaved position streams), whose streams are equal for text:
    # plain RoPE over the first head_dim * partial_rotary_factor dims (Qwen4ExpTextRotaryEmbedding).
    theta = float(rope.get("rope_theta", t.get("rope_theta", 10000.0)))
    prf = float(rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0)))
    heads, hd = t["num_attention_heads"], t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    qsa = dict(index_heads=t.get("indexer_n_heads"), index_head_dim=t.get("indexer_head_dim"),
               budget=t.get("indexer_budget"), compress=t.get("indexer_compress_ratio"))
    if any(v is not None for v in qsa.values()):
        if any(v is None for v in qsa.values()) or t.get("indexer_kv_heads", 1) != 1:
            raise ValueError(f"QSA config incomplete or with more than one key head: {qsa}")
        if qsa["budget"] % qsa["compress"]:
            raise ValueError("indexer_budget must be divisible by indexer_compress_ratio")
        full = QSASpec(heads, t["num_key_value_heads"], hd, hd, int(hd * prf), theta, **qsa)
        if full.rope_dim > full.index_head_dim:
            raise ValueError("the RoPE dims must fit the QSA index head")
    else:
        full = AttnSpec(heads, t["num_key_value_heads"], hd, hd, int(hd * prf), theta)
    lin = linear_attn.gdn_spec(t, _act(t.get("output_gate_type") or t.get("hidden_act", "silu")))
    specs = []
    for kind in types[:n]:
        if kind == "linear_attention":
            specs.append(lin)
        elif kind in ("full_attention", "indexed_attention", "qwen_sparse_attention"):
            specs.append(full)
        else:
            raise NotImplementedError(f"layer type {kind!r}")
    text_eos = _ids(t.get("eos_token_id"))
    hy = HybridSpec("qwen4_exp", hc=int(t.get("hc_count", 4)), lowrank=int(t.get("hc_lowrank", 320)),
                    shared_expert=int(t.get("shared_expert_intermediate_size") or 0), block_norms=False,
                    ple=ple_spec(t, text_eos[0] if text_eos else None))
    return cls(
        architecture=c["architectures"][0], vocab_size=t["vocab_size"], hidden_size=t["hidden_size"],
        intermediate_size=hy.shared_expert, num_layers=n, num_heads=heads, num_kv_heads=t["num_key_value_heads"],
        head_dim=hd, rms_norm_eps=t.get("rms_norm_eps", 1e-6), rope_theta=theta,
        max_position_embeddings=t.get("max_position_embeddings", 262144),
        tie_word_embeddings=bool(t.get("tie_word_embeddings", c.get("tie_word_embeddings", False))),
        eos_token_ids=eos_ids or text_eos, qk_norm=True, qkv_bias=bool(t.get("attention_bias", False)),
        num_experts=t["num_experts"], num_experts_per_tok=t["num_experts_per_tok"],
        moe_intermediate_size=t["moe_intermediate_size"], norm_topk_prob=t.get("norm_topk_prob", True),
        moe_layers=tuple(range(n)), attn_layers=tuple(specs), attn_output_gate=True, norm_offset=True, hybrid=hy,
        **cls._quant(c if "quantization_config" in c else t))


def _ids(e) -> tuple[int, ...]:
    return tuple(e) if isinstance(e, list) else ((e,) if e is not None else ())


# -- n-gram hashing (host) ----------------------------------------------------------------------
# Adapted from transformers v5.18.0 (commit a906d3c4b65095f2308b6a6a193e934d03b8eb5d)
# src/transformers/models/qwen4_exp/modeling_qwen4_exp.py, Qwen4ExpTextNGramEmbedding and its
# helpers _splitmix64, _build_layer_multipliers, _is_prime, _find_nth_prime_after,
# _shift_right_ignore_eos. Copyright 2026 The Qwen Team and The HuggingFace Inc. team, licensed
# under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0); see
# THIRD_PARTY_NOTICES.md. The constants and the hash must stay bit-identical to the checkpoint's.

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _multipliers(vocab: int, ngram: int, ple_index: int, seed: int) -> tuple[int, ...]:
    half = max(1, ((1 << 63) - 1) // max(vocab, 1) // 2)
    base = seed + _PRIME_1 * ple_index
    return tuple(2 * (_splitmix64((base + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half) + 1 for i in range(ngram))


def _is_prime(v: int) -> bool:
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def _primes_after(start: int, count: int) -> list[int]:
    """The first `count` primes above `start` (the reference finds the i-th one afresh per head)."""
    out, p = [], start
    while len(out) < count:
        p += 1
        while not _is_prime(p):
            p += 1
        out.append(p)
    return out


def ple_spec(t: dict, eos: int | None) -> PLESpec | None:
    ids = sorted(set(t.get("ple_layer_ids") or ()))
    if not ids:
        return None
    if eos is None:
        raise ValueError("Qwen4-Exp PLE needs eos_token_id")
    ngram, hpn = int(t.get("ngram_size", 3)), int(t.get("heads_per_ngram", 8))
    heads = (ngram - 1) * hpn
    embed = int(t.get("ple_embed_dim") or t["hidden_size"])
    if embed % heads:
        raise ValueError(f"ple_embed_dim {embed} is not divisible by {heads} n-gram heads")
    base, div = int(t.get("ngram_vocab_size_base", 20_000_000)), int(t.get("make_ngram_vocab_size_divisible_by", 128))
    seed = int(t.get("seed", 1234))
    primes = _primes_after(base - 1, len(ids) * heads)  # head h of PLE layer j: the (j * heads + h + 1)-th
    vocab, offset, mult, rows = [], [], [], []
    for j in range(len(ids)):
        sizes = primes[j * heads : (j + 1) * heads]
        vocab.append(tuple(sizes))
        offset.append(tuple(int(x) for x in np.cumsum([0] + sizes[:-1])))
        mult.append(_multipliers(t["vocab_size"], ngram, j, seed))
        rows.append(-(-sum(sizes) // div) * div)
    return PLESpec(tuple(i - 1 for i in ids), embed, ngram, hpn, int(t.get("ple_conv_kernel_size", 4)), int(eos),
                   tuple(vocab), tuple(offset), tuple(mult), tuple(rows), int(t.get("split_ngram_parts", 512)))


def ngram_ids(spec: PLESpec, tokens, start: int, end: int) -> np.ndarray:
    """[end - start, layers * heads] int64 table rows of positions start .. end - 1 of a sequence
    whose token ids are `tokens` (at least up to end - 1): for every PLE layer and head, the hash
    of the 2- or 3-gram ending at that position, with positions before 0 and every token across
    an EOS read as EOS (_shift_right_ignore_eos)."""
    ctx = spec.ngram - 1
    hist = np.array([tokens[p] if p >= 0 else spec.eos for p in range(start - ctx, end)], dtype=np.int64)
    m = np.arange(ctx, ctx + end - start)
    is_eos = hist == spec.eos
    shifted = [hist[m]]
    for k in range(1, spec.ngram):
        # valid iff no EOS among the k tokens before the position
        crossed = np.zeros(len(m), dtype=bool)
        for i in range(1, k + 1):
            crossed |= is_eos[m - i]
        shifted.append(np.where(crossed, spec.eos, hist[m - k]))
    cols = []
    for j in range(len(spec.layers)):
        mul = spec.multipliers[j]
        for g in range(2, spec.ngram + 1):
            mixed = shifted[0] * mul[0]
            for pos in range(1, g):
                mixed = np.bitwise_xor(mixed, shifted[pos] * mul[pos])
            hs = slice((g - 2) * spec.heads_per_ngram, (g - 1) * spec.heads_per_ngram)
            sizes = np.array(spec.head_vocab[j][hs], dtype=np.int64)
            cols.append(np.remainder(mixed[:, None], sizes[None, :]) + np.array(spec.head_offset[j][hs], dtype=np.int64))
    return np.concatenate(cols, axis=1)


def ple_columns(spec: PLESpec, tp: int, rank: int) -> torch.Tensor:
    """This rank's columns of the concatenated n-gram embedding [heads * head_dim]: a 1 / tp
    column slice of every head (the tables are sharded by columns)."""
    hd = spec.head_dim
    if hd % tp:
        raise ValueError(f"tp={tp} does not divide the n-gram head dim {hd}")
    w = hd // tp
    return torch.cat([torch.arange(h * hd + rank * w, h * hd + (rank + 1) * w) for h in range(spec.heads)])


# -- parameters ----------------------------------------------------------------------------------


def init_layer(layer, cfg: ModelConfig, spec, index: int, tp: int, p, f32) -> None:
    """QSA indexer parameters (replicated on every rank, as the MLA indexer) and, on a PLE layer,
    its projections, norms and conv. index_qk_proj is split by rows into each head's rope rows,
    the rest, and the key: rotating the first dims of each head in place broke neuronx-cc on the
    MLA indexer (models/mla.py _indexer), so the two query parts are never concatenated."""
    H = cfg.hidden_size
    if isinstance(spec, QSASpec):
        hi, di, dr = spec.index_heads, spec.index_head_dim, spec.rope_dim
        layer.idx_q_r, layer.idx_q_p, layer.idx_k = p(hi * dr, H), p(hi * (di - dr), H), p(di, H)
        layer.idx_qn, layer.idx_kn = f32(di), f32(di)  # (1 + w), fp32
        layer.idx_cache = None  # [slots, 1, Di] raw keys, bound by the runner (aux_kv_shapes)
    pl = cfg.hybrid.ple
    layer.ple_slot = None
    if pl is not None and index in pl.layers:
        hc = cfg.hybrid.hc
        e = pl.embed_dim // tp
        layer.ple_slot = pl.layers.index(index)
        layer.ple_key, layer.ple_value = p(hc * H, e), p(H, e)  # input columns: this rank's table columns
        layer.ple_nk, layer.ple_nq, layer.ple_nc = f32(hc * H), f32(hc * H), f32(hc * H)
        layer.ple_conv = p(hc * H, pl.conv_kernel)
        layer.ple_state = None  # [rows, (kernel - 1) * ngram, hc * H] per request (aux_state_shapes)


def load_layer(layer, ck, pre: str, cfg: ModelConfig, r: int, n: int, dtype, dense) -> None:
    """Checkpoint names: Qwen/Qwen3.8-Flash-Next model.safetensors.index.json (self_attn.indexer.
    index_qk_proj / q_layernorm / k_layernorm, ple.key_proj / value_proj / norm_key / norm_query /
    norm_conv / conv1d)."""
    sp = layer.spec
    if isinstance(sp, QSASpec):
        hi, di, dr = sp.index_heads, sp.index_head_dim, sp.rope_dim
        w = dense(pre + "self_attn.indexer.index_qk_proj")
        qw = w[: hi * di].view(hi, di, -1)
        layer.idx_q_r.data.copy_(qw[:, :dr].reshape(hi * dr, -1).to(dtype))
        layer.idx_q_p.data.copy_(qw[:, dr:].reshape(hi * (di - dr), -1).to(dtype))
        layer.idx_k.data.copy_(w[hi * di :].to(dtype))
        layer.idx_qn.data.copy_(1.0 + ck.get(pre + "self_attn.indexer.q_layernorm.weight").float())
        layer.idx_kn.data.copy_(1.0 + ck.get(pre + "self_attn.indexer.k_layernorm.weight").float())
    if layer.ple_slot is not None:
        pl = cfg.hybrid.ple
        a = pre + "ple."
        cols = ple_columns(pl, n, r)
        layer.ple_key.data.copy_(dense(a + "key_proj")[:, cols].to(dtype))
        layer.ple_value.data.copy_(dense(a + "value_proj")[:, cols].to(dtype))
        for attr, name in (("ple_nk", "norm_key"), ("ple_nq", "norm_query"), ("ple_nc", "norm_conv")):
            getattr(layer, attr).data.copy_(1.0 + ck.get(a + name + ".weight").float())
        layer.ple_conv.data.copy_(ck.get(a + "conv1d.weight").reshape(layer.ple_conv.shape).to(dtype))
        e = a + "ple_embedding."
        j = layer.ple_slot
        for nm, want in (("layer_multipliers", pl.multipliers[j]), ("ngram_heads_vocab_sizes", pl.head_vocab[j]),
                         ("ngram_heads_offsets", pl.head_offset[j])):
            if e + nm in ck and ck.get(e + nm).tolist() != list(want):  # the checkpoint's own copy, when stored
                raise ValueError(f"{e + nm} in the checkpoint differs from the config's derivation")


def load_table(param, ck, pre: str, spec: PLESpec, j: int, r: int, n: int) -> None:
    """One PLE layer's n-gram table, this rank's columns, from either layout: one tensor
    (ngram_embedding.weight) or split_ngram_parts row shards (ngram_embedding.shard_{i}.weight,
    transformers conversion_mapping.py "qwen4_exp_text": Concatenate(dim=0))."""
    from .quant import dequant

    base = pre + "ple.ple_embedding.ngram_embedding."
    w = spec.head_dim // n
    cols = slice(r * w, (r + 1) * w)

    def read(name):  # the FP8 checkpoint stores the tables with 128 x 128 block scales
        if name + ".weight_scale_inv" in ck:
            t, s = ck.linear(name, block=128)
            return dequant(t, s, torch.float32)[:, cols]
        return ck.get(name + ".weight", None, cols)

    if base + "weight" in ck:
        param.data.copy_(read(base[:-1]).to(param.dtype))
        return
    o = 0
    for i in range(spec.split_parts):
        part = read(f"{base}shard_{i}")
        param.data[o : o + part.shape[0]].copy_(part.to(param.dtype))
        o += part.shape[0]
    if o != param.shape[0]:
        raise ValueError(f"{base}shard_*: {o} rows, expected {param.shape[0]}")


# -- forward pieces ------------------------------------------------------------------------------


def norm(x: torch.Tensor, w: torch.Tensor, eps: float, group: int | None = None) -> torch.Tensor:
    """Qwen4ExpTextRMSNorm: fp32, (1 + w) held in w (fp32), optionally per `group`-wide slice."""
    xf = x.float()
    if group is not None:
        xf = xf.reshape(*x.shape[:-1], -1, group)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if group is not None:
        xf = xf.reshape(x.shape)
    return (xf * w).to(x.dtype)


def _rope(model, layer, x: torch.Tensor, positions: torch.Tensor, head_axis: bool) -> torch.Tensor:
    """Rotate-half RoPE over the whole last axis of x [..., rope_dim]; head_axis: x is
    [T, heads, rope_dim] for positions [T], else positions index x's second-to-last axis."""
    cos = getattr(model, f"rope_cos{layer.rope}")[positions]
    sin = getattr(model, f"rope_sin{layer.rope}")[positions]
    if head_axis:
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    return x * cos + rotate_half(x) * sin


def _index_q(model, layer, x: torch.Tensor, positions: torch.Tensor):
    """The indexer query [T, Hi, *] as (rotated rope part, rest), q_layernorm'd over both parts."""
    sp = layer.spec
    T = x.shape[0]
    qr = F.linear(x, layer.idx_q_r).view(T, sp.index_heads, sp.rope_dim)
    qp = F.linear(x, layer.idx_q_p).view(T, sp.index_heads, sp.index_head_dim - sp.rope_dim)
    ms = (qr.float().pow(2).sum(-1, keepdim=True) + qp.float().pow(2).sum(-1, keepdim=True)) / sp.index_head_dim
    r = torch.rsqrt(ms + model.cfg.rms_norm_eps)
    dr = sp.rope_dim
    qr = (qr.float() * r * layer.idx_qn[:dr]).to(x.dtype)
    qp = (qp.float() * r * layer.idx_qn[dr:]).to(x.dtype)
    return _rope(model, layer, qr, positions, True), qp


def _select(model, layer, q, kI: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """QSA's additive selection mask [B, Q, L]: q = (rope part [B, Q, Hi, dr], rest), kI [B, L, Di]
    raw keys, vis [B, Q, L] the causal visibility."""
    from .glm5_next import block_mask

    sp = layer.spec
    c, di, dr = sp.compress, sp.index_head_dim, sp.rope_dim
    B, Q, L = vis.shape
    pad = -L % c
    if pad:  # the sequence form: the last block may be incomplete
        kI = torch.cat([kI, kI.new_zeros(B, pad, di)], dim=1)
        vis = torch.cat([vis, torch.full((B, Q, pad), NEG_INF, dtype=vis.dtype, device=vis.device)], dim=-1)
    P = (L + pad) // c
    kb = kI.reshape(B, P, c, di).float().mean(dim=2).to(kI.dtype)
    kb = norm(kb, layer.idx_kn, model.cfg.rms_norm_eps)
    starts = torch.arange(P, device=kI.device) * c  # each block rotated at its first position
    kb_r = _rope(model, layer, kb[..., :dr], starts, False)
    s = torch.einsum("bqhd,bpd->bqhp", q[0].float(), kb_r.float())
    s = s + torch.einsum("bqhd,bpd->bqhp", q[1].float(), kb[..., dr:].float())
    index = torch.relu(s).sum(dim=2) / math.sqrt(di)
    return block_mask(index, vis, c, sp.budget // c, tail=True, select=SELECT)[..., :L]


def attention(model, layer, x: torch.Tensor, positions: torch.Tensor, slot_mapping, table, bias,
              seq: dict | None) -> torch.Tensor:
    """Gated attention (with the QSA mask when the layer has an indexer) on the hyper-connection
    input x, every batch form of DecoderForCausalLM._layer (decode, prefill chunk), or the whole
    sequence without a cache when seq is not None. Returns the o_proj output after its all-reduce over
    the attention group (DecoderForCausalLM: attention TP)."""
    sp = layer.spec
    T = x.shape[0]
    q, k, v = model._qkv(layer, x, positions)
    qsa = isinstance(sp, QSASpec)
    if qsa:
        qi = _index_q(model, layer, x, positions)
        ki = F.linear(x, layer.idx_k)  # raw, un-normalised, unrotated
    if seq is not None:
        j = positions.unsqueeze(0)
        vis = torch.where(j <= positions.unsqueeze(1), 0.0, NEG_INF).view(1, T, T)
        if qsa and T > sp.budget:
            vis = vis + _select(model, layer, (qi[0].unsqueeze(0), qi[1].unsqueeze(0)), ki.unsqueeze(0), vis)
        G = layer.nh // layer.nkv
        s = torch.einsum("chgd,lhd->hgcl", q.view(T, layer.nkv, G, sp.head_dim), k).float() * sp.head_dim ** -0.5
        pr = model._softmax(layer, s + vis.view(1, 1, T, T), kv_axis=0)
        o = torch.einsum("hgcl,lhd->chgd", pr, v)
    else:
        model._store(layer.k_cache, slot_mapping, k)
        model._store(layer.v_cache, slot_mapping, v)
        if qsa:
            model._store(layer.idx_cache, slot_mapping, ki.unsqueeze(1))
        L = table.shape[-1] * model.page_size
        sparse = qsa and L > sp.budget  # static per bucket: at most `budget` keys select themselves all
        # With the selection in the graph, token-granular cache gathers lowered to a 58 ms decode
        # layer (against 1.8 ms with whole pages; trn1.2xlarge, SDK 2.32, tools/probe_hybrid.py qsa
        # with KILN_GATHER=page, 2026-10-03), so a sparse bucket gathers pages.
        if not sparse:
            kc, vc = model._load(layer.k_cache, table), model._load(layer.v_cache, table)
        else:
            kc, vc = _load_pages(model, layer.k_cache, table), _load_pages(model, layer.v_cache, table)
            kI = _load_pages(model, layer.idx_cache, table).squeeze(-2)
            if table.dim() == 1:  # chunk: one sequence, C queries
                B, Q, kI = 1, T, kI.unsqueeze(0)
            elif bias.dim() == 5:  # extend (speculative verify): B sequences x Q queries
                B, Q = table.shape[0], bias.shape[3]
            else:  # decode
                B, Q = T, 1
            vis = bias.reshape(B, Q, L)
            qb = (qi[0].reshape(B, Q, sp.index_heads, -1), qi[1].reshape(B, Q, sp.index_heads, -1))
            bias = (vis + _select(model, layer, qb, kI, vis)).reshape(bias.shape)
        o = model._attend(layer, q, kc, vc, table, bias)
    o = o.reshape(T, layer.nh * sp.v_head_dim) * torch.sigmoid(F.linear(x, model._w(layer, "o_gate")))
    return model._attn_all_reduce(F.linear(o, model._w(layer, "o")))


def _load_pages(model, cache: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """DecoderForCausalLM._load gathering whole pages (its KILN_GATHER=page form)."""
    ps = model.page_size
    kv = cache.view(-1, ps, *cache.shape[1:])[table].flatten(-4, -3)
    return kv.to(model.dtype) if model.fp8_max is not None else kv


def ple(model, layer, streams: torch.Tensor, emb: torch.Tensor, positions: torch.Tensor, slot_mapping,
        state_slot) -> torch.Tensor:
    """streams [T, hc * H] plus the Per-Layer Embedding (Qwen4ExpTextPLELayer.forward) of this
    layer's n-gram embedding rows emb [T, embed_dim / tp]. Its dilated short conv keeps the last
    (kernel - 1) * ngram inputs per sequence in the state pool (layer.ple_state), read and written
    like the linear-attention conv state (models/linear_attn.mixer's three forms)."""
    cfg, pl = model.cfg, model.cfg.hybrid.ple
    hc, H = cfg.hybrid.hc, cfg.hidden_size
    T, eps = streams.shape[0], cfg.rms_norm_eps
    key = F.linear(emb, layer.ple_key)
    value = F.linear(emb, layer.ple_value)
    if model.tp_size > 1:  # each rank projected its columns of the embedding
        kv = model._all_reduce(torch.cat([key, value], dim=-1))
        key, value = kv[:, : hc * H], kv[:, hc * H :]
    kn = norm(key, layer.ple_nk, eps, H).view(T, hc, H)
    qn = norm(streams, layer.ple_nq, eps, H).view(T, hc, H)
    gate = (kn * qn).sum(dim=-1, keepdim=True) / math.sqrt(H)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gv = (torch.sigmoid(gate) * value.unsqueeze(1)).reshape(T, hc * H)
    gn = norm(gv, layer.ple_nc, eps, H)
    S, d = pl.conv_state_len, pl.ngram
    K = pl.conv_kernel
    if state_slot is None:  # sequence form, zero history
        xe = torch.cat([gn.new_zeros(S, hc * H), gn])
        y = _dilated(xe, layer.ple_conv, T, d, K)
    elif state_slot.dim() == 2:  # verify: B sequences x Q tokens, the history after every position kept
        from .linear_attn import _write_rows

        B, Q = state_slot.shape[0], state_slot.shape[1] - 1
        keep = (positions.view(B, Q)[:, 0] > 0).view(B, 1, 1)
        prev = layer.ple_state[state_slot[:, 0]]
        prev = torch.where(keep, prev, torch.zeros_like(prev)).to(gn.dtype)
        xe = torch.cat([prev, gn.view(B, Q, -1)], dim=1)  # [B, S + Q, C]
        y = _dilated(xe, layer.ple_conv, Q, d, K).reshape(T, -1)
        win = torch.stack([xe[:, t + 1 : t + 1 + S] for t in range(Q)], dim=1)
        _write_rows(layer.ple_state, state_slot[:, 1:].reshape(B * Q),
                    win.reshape(B * Q, S, -1).to(layer.ple_state.dtype))
    elif state_slot.shape[0] == T:  # decode
        keep = (positions > 0).view(T, 1, 1)
        prev = layer.ple_state[state_slot]
        prev = torch.where(keep, prev, torch.zeros_like(prev)).to(gn.dtype)
        xe = torch.cat([prev, gn.unsqueeze(1)], dim=1)  # [B, S + 1, C]
        y = _dilated(xe, layer.ple_conv, 1, d, K)[:, 0]
        from .linear_attn import _write_rows

        _write_rows(layer.ple_state, state_slot, xe[:, 1:].to(layer.ple_state.dtype))
    else:  # chunk
        from .linear_attn import _write_rows

        keep = (positions[:1] > 0).view(1, 1)
        prev = layer.ple_state[state_slot][0]
        prev = torch.where(keep, prev, torch.zeros_like(prev)).to(gn.dtype)
        xe = torch.cat([prev, gn])
        y = _dilated(xe, layer.ple_conv, T, d, K)
        valid = slot_mapping >= model.page_size  # real tokens; padding writes the null page
        last = valid.sum() + torch.arange(S, device=xe.device)
        _write_rows(layer.ple_state, state_slot, xe.index_select(0, last).unsqueeze(0).to(layer.ple_state.dtype))
    return streams + gv + F.silu(y.to(gv.dtype))


def _dilated(xe: torch.Tensor, w: torch.Tensor, T: int, d: int, K: int) -> torch.Tensor:
    """Depthwise conv with dilation d over xe [..., (K - 1) d + T, C] (the history first), w [C, K]:
    out[t] = sum_j w[:, j] xe[t + j d], accumulated in fp32 (nn.Conv1d(dilation=d) after the
    reference's left padding)."""
    wf = w.float()
    y = xe[..., 0:T, :].float() * wf[:, 0]
    for j in range(1, K):
        y = y + xe[..., j * d : j * d + T, :].float() * wf[:, j]
    return y
