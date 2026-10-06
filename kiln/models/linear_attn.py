"""Linear-attention token mixers for hybrid decoders: Gated DeltaNet and Kimi Delta Attention.

A linear-attention layer replaces softmax attention over a KV cache with a fixed-size state
per sequence: the last K - 1 inputs of a short causal depthwise conv, and one [Dk, Dv]
matrix per head updated by the gated delta rule

    S_t = exp(g_t) * S_{t-1} + k_t (beta_t (v_t - (exp(g_t) * S_{t-1})^T k_t))^T,   o_t = S_t^T q_t

with q, k L2-normalised and q scaled by Dk^-0.5. Gated DeltaNet ("gdn", Qwen3.5 / Qwen3.8 /
Qwen3-Next) decays each head by one scalar g_t; Kimi Delta Attention ("kda", Kimi Linear,
Kimi K3, GLM-5.3-Flash) decays each key channel separately (g_t is a vector). The layer's
output is RMSNorm(o) * act(gate) followed by the output projection.

Numerics follow the Hugging Face transformers ports, which are the CPU references:
- GDN: transformers models/qwen3_5/modeling_qwen3_5.py (v5.15.0, Qwen3_5GatedDeltaNet,
  torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule, Qwen3_5RMSNormGated); the
  Qwen3.8-Flash-Next variant (models/qwen4_exp, v5.18.0, Qwen4ExpTextGatedDeltaNet) differs
  only in the gate activation (config output_gate_type).
- KDA: transformers models/kimi_linear/modeling_kimi_linear.py (v5.18.0, KimiLinearDeltaAttention,
  KimiLinearForgetGate, chunk_kimi_delta_attention) and models/glm5_next/modeling_glm5_next.py
  (v5.18.0, Glm5NextTextLinearAttention, whose forget gate adds the "safe gate" lower bound
  lower_bound * sigmoid(exp(A_log) * g)). Kimi K3 (moonshotai/Kimi-K3 modeling_kimi_linear.py,
  KimiDeltaAttention) adds a full-rank output gate (g_proj) and the same lower bound.

Static-shape formulation (the same code runs on CPU and on a NeuronCore):
- decode (B sequences x 1 token): the recurrent step above as batched einsums.
- prefill chunk (one sequence x C tokens): the chunked form (fla / the transformers chunk
  kernels) over fixed sub-chunks of CHUNK tokens, unrolled at trace time. The per-chunk
  triangular solve (I + A)^-1, which the references do by forward substitution (CHUNK - 1
  sequential row updates), is computed as prod_j (I + N^(2^j)) for the strictly lower N = -A:
  N is nilpotent, so log2(CHUNK) squarings give the exact inverse with plain matmuls. Padded
  tokens get beta = 0 and g = 0, which leaves the state untouched.
- Every per-sequence state lives in a device pool (engine/state_pool.py) indexed by a state
  slot per sequence; a call reads rows with a gather and writes them back with index_put_.
  A sequence whose first position is 0 starts from zero state (torch.where, so a stale row,
  even one holding NaN, is never read), which also resets a reused slot.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..config import LinearSpec

NEG = -1e30
# Tokens per sub-chunk of the chunked delta rule (the references use 64), per kind. KDA's
# per-channel decay materialises [heads, L, L, Dk] per sub-chunk. One layer, prefill C=128 on
# trn1.2xlarge (tools/profile_linear_attn.py, SDK 2.32, 2026-10-03): GDN at Qwen3.5-0.8B shapes
# 0.61 / 0.52 / 0.46 ms for L = 32 / 64 / 128; KDA at Kimi-Linear-48B shapes (32 heads)
# 3.38 / 3.29 / 4.10 ms for L = 16 / 32 / 64. KILN_LA_CHUNK overrides both.
CHUNKS = {"gdn": 64, "kda": 32}
CHUNK = int(os.environ.get("KILN_LA_CHUNK", 0)) or None

# Architectures whose config.json config_from_hf parses (see each parser for its limits).
ARCHITECTURES = ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM", "KimiLinearForCausalLM")


# -- config.json -------------------------------------------------------------------------


def _act(name: str) -> str:
    name = {"swish": "silu"}.get(name, name)
    if name not in ("silu", "sigmoid"):
        raise NotImplementedError(f"output gate activation {name!r}")
    return name


def gdn_spec(t: dict, gate_act: str = "silu") -> LinearSpec:
    """Keys of the Qwen3.5 / Qwen3.8 text config (transformers models/qwen3_5/
    configuration_qwen3_5.py, Qwen3_5TextConfig: linear_num_key_heads, linear_num_value_heads,
    linear_key_head_dim, linear_value_head_dim, linear_conv_kernel_dim)."""
    return LinearSpec("gdn", t["linear_num_key_heads"], t["linear_num_value_heads"], t["linear_key_head_dim"],
                      t["linear_value_head_dim"], t.get("linear_conv_kernel_dim", 4), gate_act)


def kda_spec(t: dict) -> LinearSpec:
    """`linear_attn_config` of Kimi Linear / Kimi K3 / GLM-5.3-Flash (num_heads, head_dim,
    short_conv_kernel_size; K3 and GLM add gate_lower_bound, K3 use_full_rank_gate: see
    moonshotai/Kimi-K3 modeling_kimi_linear.py KimiDeltaAttention.__init__). The low-rank
    gates g_a_proj / f_a_proj project to head_dim."""
    la = t["linear_attn_config"]
    h, d = la["num_heads"], la["head_dim"]
    lb = la.get("gate_lower_bound")
    return LinearSpec("kda", h, h, d, d, la.get("short_conv_kernel_size", 4), "sigmoid",
                      0 if la.get("use_full_rank_gate") else d, float(lb) if lb is not None else None)


def config_from_hf(cls, arch: str, c: dict, eos_ids: tuple[int, ...]):
    if arch.startswith("Qwen3_5"):
        return _qwen3_5(cls, arch, c, eos_ids)
    return _kimi_linear(cls, c, eos_ids)


def _qwen3_5(cls, arch: str, c: dict, eos_ids):
    """Qwen3.5 / Qwen3.8 dense (Qwen/Qwen3.5-0.8B, Qwen/Qwen3.8-27B config.json): the text
    config sits under text_config for the vision-language checkpoints. Layers follow
    layer_types (else every full_attention_interval-th layer is full attention,
    configuration_qwen3_5.py __post_init__). Full-attention layers are Qwen3-style (QK-norm)
    with an output gate (q_proj emits [q | gate] per head, attn_output_gate) and partial RoPE;
    every RMSNorm outside the linear layers is zero-centred, (1 + w) (Qwen3_5RMSNorm). The
    rope is "mrope", whose three position streams are equal for text, so it is plain RoPE.
    The MTP head (mtp_num_hidden_layers, Qwen3-Next style: one full-attention layer with a dense
    MLP behind fc(cat(norm(embedding), norm(hidden)))) loads with --spec-method mtp; the vision
    tower is not loaded."""
    t = c.get("text_config") or c
    if t.get("num_experts"):
        raise NotImplementedError("Qwen3.5 MoE (qwen3_5_moe) is not supported yet")
    if t.get("hidden_act", "silu") != "silu":
        raise NotImplementedError(f"hidden_act {t.get('hidden_act')!r}")
    # transformers' qwen3_5 gated norm is always silu (Qwen3_5RMSNormGated.activation); Qwen3.8-27B
    # writes output_gate_type "swish", the same function.
    gate = _act(t.get("output_gate_type") or "silu")
    if gate != "silu":
        raise NotImplementedError(f"qwen3_5 with output_gate_type {gate!r} (transformers applies silu)")
    rope = dict(t.get("rope_parameters") or {})
    if rope.get("rope_type", "default") != "default":
        raise NotImplementedError(f"rope type {rope.get('rope_type')!r}")
    theta = float(rope.get("rope_theta", t.get("rope_theta", 10000.0)))
    prf = float(rope.get("partial_rotary_factor", t.get("partial_rotary_factor", 0.25)))
    n = t["num_hidden_layers"]
    types = t.get("layer_types") or ["linear_attention" if (i + 1) % t.get("full_attention_interval", 4)
                                     else "full_attention" for i in range(n)]
    heads, hd = t["num_attention_heads"], t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
    from ..config import AttnSpec

    full = AttnSpec(heads, t["num_key_value_heads"], hd, hd, int(hd * prf), theta)
    lin = gdn_spec(t, gate)
    specs = []
    for kind in types[:n]:
        if kind not in ("full_attention", "linear_attention"):
            raise NotImplementedError(f"layer type {kind!r}")
        specs.append(full if kind == "full_attention" else lin)
    eos = eos_ids or _ids(t.get("eos_token_id"))
    return cls(
        architecture=arch, vocab_size=t["vocab_size"], hidden_size=t["hidden_size"],
        intermediate_size=t["intermediate_size"], num_layers=n, num_heads=heads, num_kv_heads=full.num_kv_heads,
        head_dim=hd, rms_norm_eps=t.get("rms_norm_eps", 1e-6), rope_theta=theta,
        max_position_embeddings=t.get("max_position_embeddings", 262144),
        tie_word_embeddings=bool(t.get("tie_word_embeddings", c.get("tie_word_embeddings", False))),
        eos_token_ids=eos, qk_norm=True, qkv_bias=bool(t.get("attention_bias", False)), attn_layers=tuple(specs),
        attn_output_gate=bool(t.get("attn_output_gate", True)), norm_offset=True,
        mtp_layers=int(t.get("mtp_num_hidden_layers") or 0), mtp_spec=full, mtp_prefix="mtp.layers.0",
        mtp_names="qwen3_5", **cls._quant(t))


def _kimi_linear(cls, c: dict, eos_ids):
    """KimiLinearForCausalLM (moonshotai/Kimi-Linear-48B-A3B-Instruct, moonshotai/Kimi-K3
    config.json): KDA layers listed 1-indexed in linear_attn_config.kda_layers (transformers
    configuration_kimi_linear.py: "types are 1-indexed in the checkpoint"). Only configs whose
    every layer is KDA with a dense SwiGLU MLP are runnable here: the full-attention layers
    are MLA (built on the feat/mla branch) and the MoE has shared experts."""
    la = c["linear_attn_config"]
    n = c["num_hidden_layers"]
    if c.get("layer_types"):  # transformers writes the resolved lists too
        full = [i for i, t in enumerate(c["layer_types"]) if t != "linear_attention"]
    else:
        full = sorted(i - 1 for i in la.get("full_attn_layers") or ())
    if full:
        raise NotImplementedError(f"Kimi Linear MLA layers {full[:4]}...: MLA is not implemented on this branch")
    mlp = c.get("mlp_layer_types") or ["dense" if i < c.get("first_k_dense_replace", 1) else "sparse"
                                       for i in range(n)]
    if any(t != "dense" for t in mlp[:n]):
        raise NotImplementedError("Kimi Linear MoE layers (shared experts) are not supported yet")
    if c.get("hidden_act", "silu") != "silu":
        raise NotImplementedError(f"hidden_act {c.get('hidden_act')!r}")
    spec = kda_spec(c)
    return cls(
        architecture="KimiLinearForCausalLM", vocab_size=c["vocab_size"], hidden_size=c["hidden_size"],
        intermediate_size=c["intermediate_size"], num_layers=n, num_heads=spec.num_k_heads,
        num_kv_heads=spec.num_k_heads, head_dim=spec.head_k_dim, rms_norm_eps=c.get("rms_norm_eps", 1e-5),
        rope_theta=float(c.get("rope_theta") or 10000.0), max_position_embeddings=c.get("max_position_embeddings", 4096),
        tie_word_embeddings=bool(c.get("tie_word_embeddings", False)), eos_token_ids=eos_ids,
        attn_layers=(spec,) * n)


def _ids(e) -> tuple[int, ...]:
    return tuple(e) if isinstance(e, list) else ((e,) if e is not None else ())


# -- parameters ----------------------------------------------------------------------------


def init_mixer(layer: nn.Module, cfg, spec: LinearSpec, tp: int, p, lin) -> None:
    """Parameters of one linear-attention layer on ONE attention rank (tp is the attention TP,
    DecoderForCausalLM), set on the DecoderLayer: heads are split across ranks (k heads and their
    v heads together), the output projection on its input dim (an all-reduce follows). p / lin are
    DecoderLayer's parameter makers (lin: a [out, in] weight, FP8 + scale when the checkpoint
    quantizes it)."""
    if spec.num_k_heads % tp or spec.num_v_heads % tp:
        raise ValueError(f"tp={tp} does not divide {spec.num_k_heads} k / {spec.num_v_heads} v heads")
    if spec.kind not in ("gdn", "kda"):
        raise ValueError(f"linear attention kind {spec.kind!r}")
    H, dk, dv = cfg.hidden_size, spec.head_k_dim, spec.head_v_dim
    nk, nv = spec.num_k_heads // tp, spec.num_v_heads // tp
    layer.nk, layer.nv = nk, nv
    layer.conv_dim = 2 * nk * dk + nv * dv  # conv channels: q, then k, then v

    def f32(*shape):
        return nn.Parameter(torch.empty(*shape, dtype=torch.float32), requires_grad=False)

    gdn = spec.kind == "gdn"
    m = "linear_attn." if gdn else "self_attn."
    lin("in_qkv", m + ("in_proj_qkv" if gdn else "q_proj"), layer.conv_dim, H)
    layer.conv_w = p(layer.conv_dim, spec.conv_kernel)
    layer.A_log = f32(nv)
    if gdn:
        lin("in_z", m + "in_proj_z", nv * dv, H)
        lin("in_b", m + "in_proj_b", nv, H)
        lin("in_a", m + "in_proj_a", nv, H)
        layer.dt_bias = f32(nv)
        layer.o_norm = p(dv)  # Qwen3_5RMSNormGated multiplies in the model dtype
    else:
        layer.f_a = p(dk, H)  # forget gate, low rank (head_dim), replicated
        layer.f_b = p(nv * dk, dk)
        layer.dt_bias = f32(nv * dk)
        lin("in_b", m + "b_proj", nv, H)
        if spec.gate_rank:
            layer.g_a, layer.g_b, layer.g_full = p(spec.gate_rank, H), p(nv * dv, spec.gate_rank), None
        else:
            layer.g_a = layer.g_b = None
            layer.g_full = p(nv * dv, H)
        layer.o_norm = f32(dv)  # KimiLinearRMSNormGated keeps the norm and its weight in fp32
    lin("out", m + ("out_proj" if gdn else "o_proj"), H, nv * dv)
    layer.conv_state = None  # [slots, K - 1, conv_dim], bound by the runner (engine/state_pool.py)
    layer.rec_state = None  # [slots, nv, dk, dv] fp32


def state_shapes(layer) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Per-sequence (conv state, recurrent state) shapes of one layer on this rank."""
    sp = layer.spec
    return (sp.conv_kernel - 1, layer.conv_dim), (layer.nv, sp.head_k_dim, sp.head_v_dim)


def load_mixer(layer, ck, pre: str, cfg, r: int, n: int, dtype: torch.dtype, blk, fp8_max: float) -> None:
    """Fill one linear-attention layer from checkpoint names under `pre` (model.layers.{i}.), as
    attention rank r of n (DecoderForCausalLM.attn_rank / attn_tp).

    GDN: linear_attn.{in_proj_qkv, conv1d, in_proj_z, in_proj_b, in_proj_a, dt_bias, A_log,
    norm, out_proj} (Qwen/Qwen3.5-0.8B model.safetensors.index.json). KDA: self_attn.{q,k,v}_proj,
    {q,k,v}_conv1d, f_a_proj, f_b_proj, dt_bias, A_log [H], b_proj, g_a_proj / g_b_proj or g_proj,
    o_norm, o_proj (zai-org/GLM-5.3-Flash model.safetensors.index.json; moonshotai/Kimi-K3
    modeling_kimi_linear.py). transformers' own layout (conv1d stacked, forget_gate.* and
    A_log [1, 1, H, 1], conversion_mapping.py "kimi_linear") is accepted as well."""
    from .loader import _assign, _concat, _part

    sp = layer.spec
    dk, dv = sp.head_k_dim, sp.head_v_dim
    kd_all, vd_all = sp.num_k_heads * dk, sp.num_v_heads * dv
    qs, vs = _part(kd_all, r, n), _part(vd_all, r, n)
    heads = _part(sp.num_v_heads, r, n)

    def name(*cands):
        for c in cands:
            if c in ck:
                return c
        raise KeyError(cands[0])

    def put(param, nm, rows=None):
        param.data.copy_(ck.get(nm, rows).to(param.dtype))

    def dense(param, nm, rows=None):  # a weight kept in the model dtype, dequantized if stored FP8
        from .quant import dequant

        w, sc = ck.linear(nm[: -len(".weight")], rows, block=blk)
        param.data.copy_(dequant(w, sc, torch.float32).to(param.dtype) if sc is not None else w.to(param.dtype))

    def proj(attr, base, rows=None, cols=None):
        w, sc = ck.linear(base, rows, cols, block=blk)
        _assign(getattr(layer, attr), getattr(layer, attr + "_scale"), w, sc, dtype, blk, fp8_max)

    shift = lambda sl, by: slice(sl.start + by, sl.stop + by)  # noqa: E731
    if sp.kind == "gdn":
        a = pre + "linear_attn."
        rows = [qs, shift(qs, kd_all), shift(vs, 2 * kd_all)]  # this rank's q, k, v rows
        w, sc = _concat([ck.linear(a + "in_proj_qkv", sl, block=blk) for sl in rows])
        _assign(layer.in_qkv, layer.in_qkv_scale, w, sc, dtype, blk, fp8_max)
        conv = ck.get(a + "conv1d.weight")
        layer.conv_w.data.copy_(torch.cat([conv[sl] for sl in rows]).reshape(layer.conv_dim, -1).to(dtype))
        proj("in_z", a + "in_proj_z", vs)
        proj("in_b", a + "in_proj_b", heads)
        proj("in_a", a + "in_proj_a", heads)
        put(layer.dt_bias, a + "dt_bias", heads)
        put(layer.A_log, a + "A_log", heads)
        put(layer.o_norm, a + "norm.weight")
        proj("out", a + "out_proj", cols=vs)
        return
    a = pre + "self_attn."
    w, sc = _concat([ck.linear(a + f"{x}_proj", qs, block=blk) for x in "qkv"])
    _assign(layer.in_qkv, layer.in_qkv_scale, w, sc, dtype, blk, fp8_max)
    if a + "conv1d.weight" in ck:  # transformers' stacked [q | k | v] conv
        conv = ck.get(a + "conv1d.weight")
        parts = [conv[shift(qs, i * kd_all)] for i in range(3)]
    else:
        parts = [ck.get(a + f"{x}_conv1d.weight", qs) for x in "qkv"]
    layer.conv_w.data.copy_(torch.cat(parts).reshape(layer.conv_dim, -1).to(dtype))
    fg = lambda x: name(a + x, a + "forget_gate." + x)  # noqa: E731
    dense(layer.f_a, fg("f_a_proj.weight"))
    dense(layer.f_b, fg("f_b_proj.weight"), qs)
    put(layer.dt_bias, fg("dt_bias"), qs)
    layer.A_log.data.copy_(ck.get(fg("A_log")).float().reshape(-1)[heads])
    proj("in_b", a + "b_proj", heads)
    if layer.g_full is not None:
        dense(layer.g_full, a + "g_proj.weight", qs)
    else:
        dense(layer.g_a, a + "g_a_proj.weight")
        dense(layer.g_b, a + "g_b_proj.weight", qs)
    put(layer.o_norm, a + "o_norm.weight")
    proj("out", a + "o_proj", cols=qs)


# -- the mixer ---------------------------------------------------------------------------------


def _softplus(x: torch.Tensor) -> torch.Tensor:
    """log(1 + exp(x)) without a comparison against a float literal (those can lower to f64
    on neuronx-cc, docs/neuron-notes.md); equal to F.softplus within fp32 rounding."""
    return F.relu(x) + torch.log1p(torch.exp(-x.abs()))


def _act_fn(name: str):
    return torch.sigmoid if name == "sigmoid" else F.silu


def _causal_conv(xe: torch.Tensor, w: torch.Tensor, T: int) -> torch.Tensor:
    """Depthwise causal conv: xe [..., K - 1 + T, D] holds the K - 1 inputs before the T new
    ones, w [D, K]; out[t] = sum_j w[:, j] * xe[t + j], as F.conv1d with K - 1 left padding
    (the references' causal_conv1d_fn / causal_conv1d_update). Accumulated in fp32."""
    K = w.shape[1]
    wf = w.float()
    y = xe[..., 0:T, :].float() * wf[:, 0]
    for j in range(1, K):
        y = y + xe[..., j : j + T, :].float() * wf[:, j]
    return y


INVERSE_BASE = 8  # diagonal blocks of _unit_lower_inverse inverted by repeated squaring


def _unit_lower_inverse(N: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
    """(I - N)^-1 for strictly lower-triangular N [..., L, L], with plain matmuls only.

    Blocks of up to INVERSE_BASE rows are inverted as I + N + ... + N^(b-1) = (I + N)(I + N^2)
    (I + N^4)...; larger ones by merging halves, [[A, 0], [C, D]]^-1 = [[A^-1, 0], [-D^-1 C A^-1,
    D^-1]] (fla's solve_tril: 16 x 16 diagonal blocks merged into 32 and 64, vLLM v0.30.0
    third_party/flash_linear_attention/ops/solve_tril.py). Repeated squaring over a whole sub-chunk
    is exact algebra but not exact arithmetic: the powers of N grow like binomial coefficients
    before they cancel, and with Qwen3.5-0.8B's real weights (63 tokens, layer 6, fp32) one sub-chunk
    of 64 came out wrong by 9.5e7 (32: 0.21, 16: 1.8e-6) against the token-by-token recurrence; the
    blocked form is exact to rounding at every length (tests/test_linear_attn.py).

    When L is b times a power of two every level works on full-size matrices: X holds the inverses
    of the diagonal blocks of size s (zero elsewhere), first for s = b by squaring N restricted to
    those blocks, then each level is X + X (N on the lower-left halves of the 2s blocks) X, which
    places D^-1 N21 A^-1 under every pair: two matmuls per level, as many as a squaring step. The
    block masks are built on the host at trace time: as in-graph arange / expand / compare ops the
    same GDN prefill layer (trn1, C = 128, sub-chunks of 64) took 3.23 ms, as constants 0.51 ms, the
    time of the unstable squaring (tools/profile_linear_attn.py, SDK 2.32, 2026-10-03)."""
    L = N.shape[-1]
    b = INVERSE_BASE
    if L <= b:
        t, pw, span = eye + N, N, 2
        while span < L:
            pw = pw @ pw
            t = t + t @ pw
            span *= 2
        return t
    if L % b == 0 and (L // b) & (L // b - 1) == 0:
        r = np.arange(L)
        block = torch.tensor((r[:, None] // b == r[None, :] // b).astype(np.float32), device=N.device)
        X, pw, span = eye + N * block, N * block, 2
        while span < b:
            pw = pw @ pw
            X = X + X @ pw
            span *= 2
        s = b
        while s < L:
            low = (r[:, None] // (2 * s) == r[None, :] // (2 * s)) & (r[:, None] // s % 2 == 1) & (r[None, :] // s % 2 == 0)
            X = X + X @ (N * torch.tensor(low.astype(np.float32), device=N.device)) @ X
            s *= 2
        return X
    h = 1 << ((L - 1).bit_length() - 1)  # any other L: by halves, the largest power of two first
    a = _unit_lower_inverse(N[..., :h, :h], eye[:h, :h])
    d = _unit_lower_inverse(N[..., h:, h:], eye[h:, h:])
    c = d @ N[..., h:, :h] @ a
    top = torch.cat([a, torch.zeros_like(N[..., :h, h:])], dim=-1)
    return torch.cat([top, torch.cat([c, d], dim=-1)], dim=-2)


def chunk_scan(q, k, v, g, beta, S, chunk: int = 64):
    """The gated delta rule over one sequence in sub-chunks of `chunk` tokens (fla's chunked
    form; transformers torch_chunk_gated_delta_rule / chunk_kimi_delta_attention).

    q, k [T, H, Dk] (q already scaled), v [T, H, Dv], g [T, H] (GDN) or [T, H, Dk] (KDA) log
    decays, beta [T, H], S [H, Dk, Dv]; fp32. T is padded up to a multiple of `chunk` with
    beta = g = 0 tokens, which do not change S. Returns o [T, H, Dv] and the final S."""
    T, H, Dk = k.shape
    L = min(chunk, T)
    pad = -T % L
    if pad:
        q, k, v, g, beta = (torch.cat([x, x.new_zeros((pad, *x.shape[1:]))]) for x in (q, k, v, g, beta))
    per_channel = g.dim() == 3
    i = torch.arange(L, device=q.device)
    lower = i.view(-1, 1) >= i.view(1, -1)  # j <= i, the diagonal included
    strict = i.view(-1, 1) > i.view(1, -1)
    tri = lower.to(torch.float32)
    eye = (i.view(-1, 1) == i.view(1, -1)).to(torch.float32)
    outs = []
    for c in range((T + pad) // L):
        sl = slice(c * L, (c + 1) * L)
        qc, kc, vc = (x[sl].transpose(0, 1) for x in (q, k, v))  # [H, L, D]
        bc = beta[sl].transpose(0, 1).unsqueeze(-1)  # [H, L, 1]
        kb, vb = kc * bc, vc * bc
        if per_channel:
            G = torch.einsum("ij,jhd->hid", tri, g[sl])  # cumulative log decay [H, L, Dk]
            D = G.unsqueeze(2) - G.unsqueeze(1)  # [H, L(i), L(j), Dk]: G_i - G_j
            decay = torch.exp(torch.where(lower.unsqueeze(-1), D, NEG))
            A = (kb.unsqueeze(2) * kc.unsqueeze(1) * decay).sum(-1)
            qk = (qc.unsqueeze(2) * kc.unsqueeze(1) * decay).sum(-1)
            eG, gl = torch.exp(G), G[:, -1]  # gl [H, Dk]
            tail = torch.exp(gl.unsqueeze(1) - G)  # [H, L, Dk]
            s_decay = torch.exp(gl).unsqueeze(-1)  # [H, Dk, 1]
        else:
            G = g[sl].transpose(0, 1) @ tri.t()  # [H, L]
            decay = torch.exp(torch.where(lower, G.unsqueeze(-1) - G.unsqueeze(-2), NEG))  # [H, L, L]
            A = (kb @ kc.transpose(-1, -2)) * decay
            qk = (qc @ kc.transpose(-1, -2)) * decay
            eG, gl = torch.exp(G).unsqueeze(-1), G[:, -1]  # gl [H]
            tail = torch.exp(gl.unsqueeze(-1) - G).unsqueeze(-1)  # [H, L, 1]
            s_decay = torch.exp(gl).view(H, 1, 1)
        Tm = _unit_lower_inverse(-torch.where(strict, A, torch.zeros_like(A)), eye)
        u = Tm @ vb  # [H, L, Dv]
        w = Tm @ (kb * eG)  # [H, L, Dk]
        vn = u - w @ S
        outs.append(((qc * eG) @ S + qk @ vn).transpose(0, 1))
        S = S * s_decay + (kc * tail).transpose(-1, -2) @ vn
    return torch.cat(outs)[:T], S


def recurrent_step(q, k, v, g, beta, S):
    """One token per row: q, k [B, H, Dk], v [B, H, Dv], g [B, H] or [B, H, Dk], beta [B, H],
    S [B, H, Dk, Dv]; fp32 (torch_recurrent_gated_delta_rule / recurrent_kimi_delta_attention).

    The references decay S, read k^T S, update S, then read q^T S. With E = diag(exp(g)) the
    same values come from ONE read of the old state, [q E; k E] S, and one update:
      delta = beta (v - (k E)^T S),   o = (q E)^T S + (q . k) delta,   S' = E S + k delta^T,
    which on trn1 avoids the tensor-engine transposes of three passes over S (the einsum form
    measured 1.05 ms per Qwen3.5-0.8B layer at B=8, 2.0 ms of it transposes, 2026-10-03)."""
    e = torch.exp(g)
    eq = e.unsqueeze(-1) if g.dim() == 2 else e  # [B, H, 1] or [B, H, Dk]
    r = torch.stack([q * eq, k * eq], dim=2) @ S  # [B, H, 2, Dv]
    delta = (v - r[:, :, 1]) * beta.unsqueeze(-1)
    o = r[:, :, 0] + (q * k).sum(-1, keepdim=True) * delta
    S = S * (eq.unsqueeze(-1) if g.dim() == 2 else e.unsqueeze(-1)) + k.unsqueeze(-1) * delta.unsqueeze(-2)
    return o, S


def verify_scan(q, k, v, g, beta, S, form: str | None = None):
    """B sequences x Q tokens each, from their own state (speculative verify): q, k [B, Q, H, Dk]
    (q scaled), v [B, Q, H, Dv], g [B, Q, H] or [B, Q, H, Dk], beta [B, Q, H], S [B, H, Dk, Dv];
    fp32. Returns o [B, Q, H, Dv] and the state after EVERY position, [B, Q, H, Dk, Dv]: a rejected
    draft is rolled back by continuing from the state after the last accepted position (vLLM
    v0.30.0 v1/attention/backends/gdn_attn.py spec_state_indices_tensor: num_spec + 1 state slots
    per request, the next step reading the one at num_accepted_tokens - 1; SGLang v0.5.21
    mem_cache/memory_pool.py intermediate_ssm_state_cache, one state per draft token, scattered
    after verify by update_mamba_state_after_mtp_verify).

    form "recurrent" (default): recurrent_step once per position, the decode step's own
    arithmetic. "chunk": the delta values of every position from one triangular solve (chunk_scan's
    algebra over a single chunk of Q tokens), then the states as S_t = E_t S_{t-1} + k_t delta_t^T,
    which needs no read of S beyond the first. KILN_LA_VERIFY overrides the default."""
    form = form or VERIFY_FORM
    B, Q = k.shape[:2]
    if form == "recurrent":
        outs, states = [], []
        for t in range(Q):
            o, S = recurrent_step(q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], S)
            outs.append(o)
            states.append(S)
        return torch.stack(outs, 1), torch.stack(states, 1)
    if form != "chunk":
        raise ValueError(f"KILN_LA_VERIFY must be recurrent or chunk, not {form!r}")
    per_channel = g.dim() == 4
    i = torch.arange(Q, device=q.device)
    lower = i.view(-1, 1) >= i.view(1, -1)
    strict = i.view(-1, 1) > i.view(1, -1)
    tri = lower.to(torch.float32)
    eye = (i.view(-1, 1) == i.view(1, -1)).to(torch.float32)
    qc, kc, vc = (x.transpose(1, 2) for x in (q, k, v))  # [B, H, Q, D]
    bc = beta.transpose(1, 2).unsqueeze(-1)
    kb, vb = kc * bc, vc * bc
    if per_channel:
        G = torch.einsum("ij,bjhd->bhid", tri, g)  # cumulative log decay [B, H, Q, Dk]
        decay = torch.exp(torch.where(lower.unsqueeze(-1), G.unsqueeze(3) - G.unsqueeze(2), NEG))
        A = (kb.unsqueeze(3) * kc.unsqueeze(2) * decay).sum(-1)
        qk = (qc.unsqueeze(3) * kc.unsqueeze(2) * decay).sum(-1)
        eG = torch.exp(G)
    else:
        G = g.transpose(1, 2) @ tri.t()  # [B, H, Q]
        decay = torch.exp(torch.where(lower, G.unsqueeze(-1) - G.unsqueeze(-2), NEG))
        A = (kb @ kc.transpose(-1, -2)) * decay
        qk = (qc @ kc.transpose(-1, -2)) * decay
        eG = torch.exp(G).unsqueeze(-1)
    Tm = _unit_lower_inverse(-torch.where(strict, A, torch.zeros_like(A)), eye)
    delta = Tm @ vb - (Tm @ (kb * eG)) @ S  # [B, H, Q, Dv]
    o = (qc * eG) @ S + qk @ delta
    e = torch.exp(g)
    states = []
    for t in range(Q):
        et = e[:, t].unsqueeze(-1) if per_channel else e[:, t].view(B, -1, 1, 1)
        S = S * et + kc[:, :, t].unsqueeze(-1) * delta[:, :, t].unsqueeze(-2)
        states.append(S)
    return o.transpose(1, 2), torch.stack(states, 1)


VERIFY_FORM = os.environ.get("KILN_LA_VERIFY", "recurrent")
# Prefill (chunk and sequence forms) on the device through the NKI chunked delta rule
# (kernels/delta_rule.py) when "nki" (default); "torch" keeps chunk_scan. Layers the kernel does not
# take keep chunk_scan: head dims other than 128, and KDA without a gate lower bound (Kimi Linear
# 48B), whose decays are unbounded (see delta_rule's reference sub-chunks). The CPU always runs
# chunk_scan. Default since 2026-10-04: GLM-5.3-Flash at tp=32 serves 42.0 vs 38.4 out tok/s on
# trn1 and 101.7 vs 99.7 on trn2 with it (docs/price-performance.md), real-weight ppl -2.100 on trn2
# and wikitext-2 -0.5515 vs -0.552 on trn1.
LINEAR_ATTN_KERNEL = os.environ.get("KILN_LINEAR_ATTN_KERNEL", "nki")
if LINEAR_ATTN_KERNEL not in ("torch", "nki"):
    raise ValueError(f"KILN_LINEAR_ATTN_KERNEL must be torch or nki, not {LINEAR_ATTN_KERNEL!r}")


def kernel_takes(spec: LinearSpec) -> bool:
    """Whether the NKI delta-rule kernel runs this layer's prefill (KILN_LINEAR_ATTN_KERNEL=nki)."""
    return (LINEAR_ATTN_KERNEL == "nki" and spec.head_k_dim == 128 and spec.head_v_dim == 128
            and (spec.kind == "gdn" or spec.lower_bound is not None))
# Probe only (tools/bench_linear_serving.py --what parts): the verify graph keeps just the state
# after its LAST position, which is what it would write if a rejected draft were rolled back by
# recomputing the accepted tokens instead. Outputs are wrong whenever a draft is rejected.
PROBE_LAST_STATE_ONLY = os.environ.get("KILN_PROBE_VERIFY_LAST_STATE_ONLY") == "1"


def _nki_chunk(q, k, v, g, beta, S):
    """chunk_scan's contract through the NKI kernel (q, k with the layer's k heads: the kernel maps
    each v head onto its k head itself)."""
    from ..kernels import delta_rule

    return delta_rule.chunk(q, k, v, g, beta, S)


def _write_rows(pool: torch.Tensor, rows: torch.Tensor, values: torch.Tensor) -> None:
    """pool[rows] = values. A ONE-row index_put_ is written with the scratch row added (it may
    hold anything): on trn1 (neuronx-cc 2.27, SDK 2.32) a one-row index_put_ of a value computed
    in the graph measured 8.7 ms for a 1 MB state row against 0.14 ms for the same write as
    two rows (tools/profile_linear_attn.py --parts, docs/neuron-notes.md, 2026-10-03)."""
    if rows.shape[0] == 1:
        rows = torch.cat([rows, torch.zeros_like(rows)])
        values = torch.cat([values, values])
    pool.index_put_((rows,), values)


def _l2norm_gdn(x):  # Qwen3.5's l2norm, applied in the input dtype before the fp32 cast
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _l2norm_kda(x):  # Kimi's l2norm, in fp32: x / sqrt(sum + eps)
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def mixer(model, layer, h: torch.Tensor, positions: torch.Tensor, slot_mapping: torch.Tensor,
          state_slot: torch.Tensor | None, mixed=None) -> torch.Tensor:
    """h plus one linear-attention block. Batch forms, told apart by state_slot:
      decode   h [B, H], state_slot [B]    (B sequences, one token each)
      chunk    h [C, H], state_slot [1]    (one sequence; padded rows have slot_mapping in
                                            the null page and are left out of the state)
      verify   h [B * Q, H], state_slot [B, 1 + Q]   (B sequences x Q tokens, speculative
                                            verify: column 0 is the row each sequence reads,
                                            columns 1 .. Q the rows that receive the state after
                                            each of its positions; see verify_scan)
      sequence h [T, H], state_slot None   (forward_logits: a whole sequence from zero state)
      mixed    h [C + D, H], state_slot [1], mixed (table, bias, dec_state [D])   (a chunk, then D
                                            decoding sequences: the chunk form on the first C rows,
                                            the decode form on the last D; DecoderForCausalLM._layer)
    """
    from .decoder import rms_norm

    return h + mix(model, layer, rms_norm(h, layer.in_norm, model.cfg.rms_norm_eps), positions, slot_mapping,
                   state_slot, mixed)


def mix(model, layer, x: torch.Tensor, positions: torch.Tensor, slot_mapping: torch.Tensor,
        state_slot: torch.Tensor | None, mixed=None) -> torch.Tensor:
    """The block itself on an already normalised input x (no residual): the output projection
    after its all-reduce over the attention group (DecoderForCausalLM: attention TP). Hyper-connection decoders (models/hybrid.py) mix it
    into their residual streams themselves. Under DP attention x holds every group's rows and the
    block runs on this rank's group's (DecoderForCausalLM._attn_in). mixed (a mixed batch, see
    mixer): the chunk's rows step their sequence's state by the chunked delta rule and the decode
    rows each step their own by the recurrent form, then one all-reduce takes both."""
    x = model._attn_in(x)
    if mixed is not None:
        from .decoder import MIXED_DECODE_SPLIT, MIXED_MIXERS, mixed_chunk_rows, mixed_decode_slices

        C = mixed_chunk_rows(x, mixed)
        if MIXED_MIXERS == "joint":
            return model._attn_all_reduce(_mix_joint(model, layer, x, positions, slot_mapping, state_slot, mixed[2], C))
        if MIXED_DECODE_SPLIT:
            return model._attn_all_reduce(torch.cat(
                [_mix_rows(model, layer, x[:C], positions[:C], slot_mapping[:C], state_slot)]
                + [_mix_rows(model, layer, x[C + a : C + b], positions[C + a : C + b], slot_mapping[C + a : C + b],
                             mixed[2][a:b]) for a, b in mixed_decode_slices(mixed)]))
        out = torch.cat([_mix_rows(model, layer, x[:C], positions[:C], slot_mapping[:C], state_slot),
                         _mix_rows(model, layer, x[C:], positions[C:], slot_mapping[C:], mixed[2])])
        return model._attn_all_reduce(out)
    # Not stored in a local first: dynamo renames a graph node after the first local a value is stored in
    # (torch/_dynamo STORE_FAST -> TensorVariable.set_name_hint), the node names are part of the compile
    # cache key, and this is how the unmixed graphs were always traced (a local here renamed nodes of
    # every KDA decode and prefill graph, which then missed every cached NEFF: compile farm, 2026-10-04).
    return model._attn_all_reduce(_mix_rows(model, layer, x, positions, slot_mapping, state_slot))


def _mix_joint(model, layer, x: torch.Tensor, positions: torch.Tensor, slot_mapping: torch.Tensor,
               state_slot: torch.Tensor, dec_state: torch.Tensor, C: int) -> torch.Tensor:
    """_mix_rows of a mixed batch's rows (decoder.MIXED_MIXERS "joint"): x [C + D] this group's chunk rows
    then decode rows. Every projection runs once over all C + D rows (each weight read once per call),
    the short conv and the delta rule per part (the chunk form on the first C rows, the decode form on
    the rest, the same arithmetic as _mix_rows), and each state pool is written once for both."""
    cfg, sp = model.cfg, layer.spec
    dt = model.dtype
    T = x.shape[0]
    D = T - C
    K = sp.conv_kernel
    qkv = F.linear(x, model._w(layer, "in_qkv"))  # [T, conv_dim]
    # Short conv: the chunk from its row's K - 1 inputs, each decode row from its own.
    keep_c = (positions[:1] > 0).view(1, 1)
    prev_c = layer.conv_state[state_slot][0]
    prev_c = torch.where(keep_c, prev_c, torch.zeros_like(prev_c)).to(qkv.dtype)
    valid = slot_mapping[:C] >= model.page_size  # the chunk's real tokens
    xe_c = torch.cat([prev_c, qkv[:C]])  # [K - 1 + C, D]
    y_c = _causal_conv(xe_c, layer.conv_w, C)
    last = valid.sum() + torch.arange(K - 1, device=xe_c.device)
    keep_d = (positions[C:] > 0).view(D, 1, 1)
    prev_d = layer.conv_state[dec_state]
    prev_d = torch.where(keep_d, prev_d, torch.zeros_like(prev_d)).to(qkv.dtype)
    xe_d = torch.cat([prev_d, qkv[C:].unsqueeze(1)], dim=1)  # [D, K, D]
    y_d = _causal_conv(xe_d, layer.conv_w, 1)[:, 0]
    _write_rows(layer.conv_state, torch.cat([state_slot, dec_state]),
                torch.cat([xe_c.index_select(0, last).unsqueeze(0), xe_d[:, 1:]]).to(layer.conv_state.dtype))
    y = F.silu(torch.cat([y_c, y_d]).to(dt))
    nk, nv, dk, dv = layer.nk, layer.nv, sp.head_k_dim, sp.head_v_dim
    kd = nk * dk
    q = y[:, :kd].reshape(T, nk, dk)
    k = y[:, kd : 2 * kd].reshape(T, nk, dk)
    v = y[:, 2 * kd :].reshape(T, nv, dv)
    if sp.kind == "gdn":
        z = F.linear(x, model._w(layer, "in_z")).reshape(T, nv, dv)
        beta = torch.sigmoid(F.linear(x, model._w(layer, "in_b"))).float()
        a = F.linear(x, model._w(layer, "in_a")).float()
        g = -torch.exp(layer.A_log) * _softplus(a + layer.dt_bias)  # [T, nv]
        q, k = _l2norm_gdn(q).float(), _l2norm_gdn(k).float()
    else:
        f = F.linear(F.linear(x, layer.f_a), layer.f_b).float() + layer.dt_bias
        f = f.reshape(T, nv, dk)
        rate = torch.exp(layer.A_log).view(1, nv, 1)
        g = sp.lower_bound * torch.sigmoid(rate * f) if sp.lower_bound is not None else -rate * _softplus(f)
        beta = torch.sigmoid(F.linear(x, model._w(layer, "in_b"))).float()
        q, k = _l2norm_kda(q.float()), _l2norm_kda(k.float())
        if layer.g_full is not None:
            z = F.linear(x, layer.g_full).reshape(T, nv, dv)
        else:
            z = F.linear(F.linear(x, layer.g_a), layer.g_b).reshape(T, nv, dv)
    q = q * dk ** -0.5
    v = v.float()
    nki = x.device.type != "cpu" and kernel_takes(sp)

    def heads(t, n):  # each k head serves nv / nk consecutive v heads (repeat_interleave)
        return t.unsqueeze(2).expand(n, nk, nv // nk, dk).reshape(n, nv, dk) if nv != nk else t

    # The chunk: the chunked delta rule from its row (padding rows leave it untouched).
    S_c = layer.rec_state[state_slot][0]
    S_c = torch.where(keep_c.view(1, 1, 1), S_c, torch.zeros_like(S_c))
    vf = valid.view(C, 1)
    beta_c = torch.where(vf, beta[:C], torch.zeros_like(beta[:C]))
    g_c = torch.where(vf.view(C, 1, 1) if g.dim() == 3 else vf, g[:C], torch.zeros_like(g[:C]))
    qc, kc = (q[:C], k[:C]) if nki else (heads(q[:C], C), heads(k[:C], C))
    scan = _nki_chunk if nki else (lambda *a: chunk_scan(*a, CHUNK or CHUNKS[sp.kind]))
    o_c, S_c = scan(qc, kc, v[:C], g_c, beta_c, S_c)
    # The decode rows: one recurrent step each from its own row.
    from . import mla
    from ..kernels import kda_decode

    if (mla.MIXED_KERNELS and kda_decode.KERNEL == "nki" and x.device.type != "cpu"
            and kda_decode.takes(dk, dv, sp.kind == "kda" and g.dim() == 3)):
        # The KDA decode kernel, as _mix_rows runs a decode call: it reads and writes the decode rows' pool rows
        # itself, in place. The chunk's row (another request's) is written first, so the kernel's in-place update is
        # the pool's last write in this layer, as in an unmixed decode graph.
        _write_rows(layer.rec_state, state_slot, S_c.unsqueeze(0))
        o_d = kda_decode.decode_step(layer.rec_state, dec_state, keep_d.view(D), heads(q[C:], D), heads(k[C:], D),
                                     v[C:], g[C:], beta[C:])
        o = torch.cat([o_c, o_d])
    else:
        S_d = layer.rec_state[dec_state]
        S_d = torch.where(keep_d.view(D, 1, 1, 1), S_d, torch.zeros_like(S_d))
        o_d, S_d = recurrent_step(heads(q[C:], D), heads(k[C:], D), v[C:], g[C:], beta[C:], S_d)
        _write_rows(layer.rec_state, torch.cat([state_slot, dec_state]), torch.cat([S_c.unsqueeze(0), S_d]))
        o = torch.cat([o_c, o_d])
    of = o.to(dt).float()
    of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    if sp.kind == "gdn":
        of = layer.o_norm * of.to(dt)
    else:
        of = layer.o_norm * of
    o = (of * _act_fn(sp.gate_act)(z.float())).to(dt)
    return F.linear(o.reshape(T, nv * dv), model._w(layer, "out"))


def _mix_rows(model, layer, x: torch.Tensor, positions: torch.Tensor, slot_mapping: torch.Tensor,
              state_slot: torch.Tensor | None) -> torch.Tensor:
    """mix on one batch form's rows of this rank's group: the output projection before its all-reduce."""
    cfg, sp = model.cfg, layer.spec
    dt = model.dtype
    T = x.shape[0]
    K = sp.conv_kernel
    qkv = F.linear(x, model._w(layer, "in_qkv"))  # [T, conv_dim]
    if state_slot is None:
        form = "sequence"
    elif state_slot.dim() == 2:
        form = "verify"
        B, Q = state_slot.shape[0], state_slot.shape[1] - 1
        read, write = state_slot[:, 0], state_slot[:, 1:].reshape(B * Q)
    elif state_slot.shape[0] == T:
        form = "decode"
    elif state_slot.shape[0] == 1:
        form = "chunk"
    else:
        raise ValueError(f"state_slot {tuple(state_slot.shape)} matches no batch form of {T} tokens")

    # Short conv over q, k, v, with the K - 1 previous inputs from the state.
    if form == "verify":
        keep = (positions.view(B, Q)[:, 0] > 0).view(B, 1, 1)
        prev = layer.conv_state[read]
        prev = torch.where(keep, prev, torch.zeros_like(prev)).to(qkv.dtype)
        xe = torch.cat([prev, qkv.view(B, Q, -1)], dim=1)  # [B, K - 1 + Q, D]
        y = _causal_conv(xe, layer.conv_w, Q).reshape(T, -1)
        win = torch.stack([xe[:, t + 1 : t + K] for t in range(Q)], dim=1)  # the last K - 1 inputs after each position
        if PROBE_LAST_STATE_ONLY:
            _write_rows(layer.conv_state, state_slot[:, 1], win[:, -1].to(layer.conv_state.dtype))
        else:
            _write_rows(layer.conv_state, write, win.reshape(B * Q, K - 1, -1).to(layer.conv_state.dtype))
    elif form == "decode":
        keep = (positions > 0).view(T, 1, 1)
        prev = layer.conv_state[state_slot]
        prev = torch.where(keep, prev, torch.zeros_like(prev)).to(qkv.dtype)
        xe = torch.cat([prev, qkv.unsqueeze(1)], dim=1)  # [B, K, D]
        y = _causal_conv(xe, layer.conv_w, 1)[:, 0]
        _write_rows(layer.conv_state, state_slot, xe[:, 1:].to(layer.conv_state.dtype))
    else:
        if form == "chunk":
            keep = (positions[:1] > 0).view(1, 1)
            prev = layer.conv_state[state_slot][0]
            prev = torch.where(keep, prev, torch.zeros_like(prev)).to(qkv.dtype)
            valid = slot_mapping >= model.page_size  # real tokens; padding writes the null page
        else:
            prev = qkv.new_zeros(K - 1, qkv.shape[1])
        xe = torch.cat([prev, qkv])  # [K - 1 + T, D]
        y = _causal_conv(xe, layer.conv_w, T)
        if form == "chunk":
            # The last K - 1 inputs of the sequence: rows n .. n + K - 2 of xe, n real tokens.
            last = valid.sum() + torch.arange(K - 1, device=xe.device)
            _write_rows(layer.conv_state, state_slot, xe.index_select(0, last).unsqueeze(0).to(layer.conv_state.dtype))
    y = F.silu(y.to(dt))
    nk, nv, dk, dv = layer.nk, layer.nv, sp.head_k_dim, sp.head_v_dim
    kd = nk * dk
    q = y[:, :kd].reshape(T, nk, dk)  # explicit slices, not torch.split (neuron-notes.md)
    k = y[:, kd : 2 * kd].reshape(T, nk, dk)
    v = y[:, 2 * kd :].reshape(T, nv, dv)

    if sp.kind == "gdn":
        z = F.linear(x, model._w(layer, "in_z")).reshape(T, nv, dv)
        beta = torch.sigmoid(F.linear(x, model._w(layer, "in_b"))).float()
        a = F.linear(x, model._w(layer, "in_a")).float()
        g = -torch.exp(layer.A_log) * _softplus(a + layer.dt_bias)  # [T, nv]
        q, k = _l2norm_gdn(q).float(), _l2norm_gdn(k).float()
    else:
        f = F.linear(F.linear(x, layer.f_a), layer.f_b).float() + layer.dt_bias
        f = f.reshape(T, nv, dk)
        rate = torch.exp(layer.A_log).view(1, nv, 1)
        g = sp.lower_bound * torch.sigmoid(rate * f) if sp.lower_bound is not None else -rate * _softplus(f)
        beta = torch.sigmoid(F.linear(x, model._w(layer, "in_b"))).float()
        q, k = _l2norm_kda(q.float()), _l2norm_kda(k.float())
        if layer.g_full is not None:
            z = F.linear(x, layer.g_full).reshape(T, nv, dv)
        else:
            z = F.linear(F.linear(x, layer.g_a), layer.g_b).reshape(T, nv, dv)
    q = q * dk ** -0.5
    v = v.float()
    nki_prefill = form in ("chunk", "sequence") and x.device.type != "cpu" and kernel_takes(sp)
    if nv != nk and not nki_prefill:  # each k head serves nv / nk consecutive v heads (repeat_interleave)
        q = q.unsqueeze(2).expand(T, nk, nv // nk, dk).reshape(T, nv, dk)
        k = k.unsqueeze(2).expand(T, nk, nv // nk, dk).reshape(T, nv, dk)
    scan = _nki_chunk if nki_prefill else (lambda *a: chunk_scan(*a, CHUNK or CHUNKS[sp.kind]))

    if form == "verify":
        S = layer.rec_state[read]
        S = torch.where(keep.view(B, 1, 1, 1), S, torch.zeros_like(S))
        o, states = verify_scan(q.view(B, Q, nv, dk), k.view(B, Q, nv, dk), v.view(B, Q, nv, dv),
                                g.view(B, Q, *g.shape[1:]), beta.view(B, Q, nv), S)
        if PROBE_LAST_STATE_ONLY:
            _write_rows(layer.rec_state, state_slot[:, 1], states[:, -1])
        else:
            _write_rows(layer.rec_state, write, states.reshape(B * Q, nv, dk, dv))
        o = o.reshape(T, nv, dv)
    elif form == "decode":
        from ..kernels import kda_decode

        if (kda_decode.KERNEL == "nki" and x.device.type != "cpu"
                and kda_decode.takes(dk, dv, sp.kind == "kda" and g.dim() == 3)):
            o = kda_decode.decode_step(layer.rec_state, state_slot, keep.view(T), q, k, v, g, beta)
        else:
            keep4 = keep.view(T, 1, 1, 1)
            S = layer.rec_state[state_slot]
            S = torch.where(keep4, S, torch.zeros_like(S))
            o, S = recurrent_step(q, k, v, g, beta, S)
            _write_rows(layer.rec_state, state_slot, S)
    elif form == "chunk":
        S = layer.rec_state[state_slot][0]
        S = torch.where(keep.view(1, 1, 1), S, torch.zeros_like(S))
        vf = valid.view(T, 1)
        beta = torch.where(vf, beta, torch.zeros_like(beta))
        g = torch.where(vf.view(T, 1, 1) if g.dim() == 3 else vf, g, torch.zeros_like(g))
        o, S = scan(q, k, v, g, beta, S)
        _write_rows(layer.rec_state, state_slot, S.unsqueeze(0))
    else:
        o, _ = scan(q, k, v, g, beta, q.new_zeros(nv, dk, dv))

    # Gated RMSNorm per head, then the output projection.
    of = o.to(dt).float()
    of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    if sp.kind == "gdn":  # Qwen3_5RMSNormGated: weight times the normalised value in the model dtype
        of = layer.o_norm * of.to(dt)
    else:  # KimiLinearRMSNormGated: fp32 throughout
        of = layer.o_norm * of
    o = (of * _act_fn(sp.gate_act)(z.float())).to(dt)
    return F.linear(o.reshape(T, nv * dv), model._w(layer, "out"))
