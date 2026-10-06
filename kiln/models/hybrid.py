"""Hyper-connection decoders: GLM-5.3-Flash (models/glm5_next.py) and Qwen3.8-Flash-Next
(models/qwen4_exp.py).

Both keep hc = 4 residual STREAMS per token instead of one hidden state. Around every block
(token mixer, then MLP) the streams are collapsed into the block's input and the block's output
is mixed back into them:

- GLM-5.3-Flash, manifold-constrained hyper-connections (Glm5NextTextHyperConnection): from the
  RMS-normalised concatenated streams one linear map gives pre [hc], post [hc] and comb [hc, hc]
  logits; pre = sigmoid(...) + eps collapses the streams (sum_n pre_n S_n), the block runs on
  that after its usual RMSNorm, and the new streams are post_n y + sum_m comb_mn S_m, comb made
  doubly stochastic by hc_sinkhorn_iters Sinkhorn-Knopp normalisations. The final hidden state is
  the plain mean of the streams, then the model norm.
- Qwen3.8-Flash-Next, gated residual (Qwen4ExpTextGatedResidual): each stream RMS-normalised
  ((1 + w), per stream), a rank-hc_lowrank sigmoid gate over all of them, the block input is the
  gated streams' mean (no further norm), and every stream gets the block output times its own
  2 sigmoid(...) injection weight added. The final hidden state is the same mixer without the
  injection, with no model norm after it.

Numerics: transformers v5.18.0 models/glm5_next/modeling_glm5_next.py and
models/qwen4_exp/modeling_qwen4_exp.py (URLs in those modules). The stream arithmetic is
replicated on every tensor-parallel rank; only the blocks inside are sharded.

The hidden state between graphs is [T, hc * H] (the streams of a token side by side), plus, for
a model with a Per-Layer Embedding, its n-gram embedding rows as a trailing block (looked up in
the prep graph: qwen4_exp). DecoderForCausalLM hands every hybrid layer to `layer` below and its
final normalisation to `final`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ..config import LinearSpec
from .decoder import rms_norm

ARCHITECTURES = ("Glm5NextForConditionalGeneration", "Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")


@dataclass(frozen=True)
class HybridSpec:
    family: str  # "glm5_next" or "qwen4_exp"
    hc: int  # residual streams (hc_mult / hc_count)
    sinkhorn_iters: int = 0  # glm5_next: hc_sinkhorn_iters
    hc_eps: float = 0.0  # glm5_next: hc_eps
    lowrank: int = 0  # qwen4_exp: hc_lowrank
    swiglu_limit: float | None = None  # glm5_next: gate clamped from above, up to [-limit, limit]
    shared_expert: int = 0  # qwen4_exp: intermediate size of the sigmoid-gated shared expert
    block_norms: bool = True  # input_layernorm / post_attention_layernorm / model.norm exist (glm5_next)
    ple: object = None  # qwen4_exp.PLESpec


def config_from_hf(cls, c: dict, eos_ids: tuple[int, ...]):
    arch = c["architectures"][0]
    if arch.startswith("Glm5Next"):
        from . import glm5_next

        return glm5_next.config_from_hf(cls, c, eos_ids)
    from . import qwen4_exp

    return qwen4_exp.config_from_hf(cls, c, eos_ids)


# -- parameters --------------------------------------------------------------------------------


def _f32(*shape):
    return nn.Parameter(torch.empty(*shape, dtype=torch.float32), requires_grad=False)


def init_layer(layer, cfg, spec, index: int, tp: int, p) -> None:
    """The hyper-connection parameters of one layer (both blocks) and the per-model extras, on the
    DecoderLayer (p: a parameter in the model dtype)."""
    hy, H = cfg.hybrid, cfg.hidden_size
    hc = hy.hc
    if hy.family == "glm5_next":
        n = (2 + hc) * hc  # pre, post, comb logits
        for b in ("attn", "ffn"):
            setattr(layer, f"hc_{b}_fn", p(n, hc * H))
            setattr(layer, f"hc_{b}_base", _f32(n))
            setattr(layer, f"hc_{b}_scale", _f32(3))
        if layer.pack_moe and hy.swiglu_limit is not None and not layer.pack_tiles and not getattr(layer, "moe_ep", False):
            raise ValueError("KILN_MOE_KERNEL=nki-pair has no clamped SwiGLU (glm5_next swiglu_limit); "
                             "KILN_MOE_KERNEL=nki has (kernels/moe_dedupe.py act 1)")
        return
    from . import qwen4_exp

    for b in ("attn", "ffn"):
        setattr(layer, f"hc_{b}_norm", _f32(hc * H))  # (1 + w), fp32
        setattr(layer, f"hc_{b}_down", p(hy.lowrank, hc * H))
        setattr(layer, f"hc_{b}_up", p(hc * H, hy.lowrank))
        setattr(layer, f"hc_{b}_inject", p(hc, hc * H))
    layer.sh_gate_up = layer.sh_down = layer.sh_gate = None
    layer.sh_gate_up_scale = layer.sh_down_scale = None  # kept in the model dtype (DecoderForCausalLM._w)
    if hy.shared_expert and layer.moe:
        if hy.shared_expert % tp:
            raise ValueError(f"tp={tp} does not divide the shared expert's intermediate size")
        Is = hy.shared_expert // tp
        layer.sh_gate_up, layer.sh_down, layer.sh_gate = p(2 * Is, H), p(H, Is), p(1, H)
    qwen4_exp.init_layer(layer, cfg, spec, index, tp, p, _f32)


def init_model(model) -> None:
    hy, H = model.cfg.hybrid, model.cfg.hidden_size
    if hy.family != "qwen4_exp":
        return
    hc = hy.hc

    def p(*shape):
        return nn.Parameter(torch.empty(*shape, dtype=model.dtype), requires_grad=False)

    model.hc_final_norm = _f32(hc * H)  # model.hyper_connection_mixer
    model.hc_final_down, model.hc_final_up = p(hy.lowrank, hc * H), p(hc * H, hy.lowrank)
    pl = hy.ple
    if pl is not None:
        if pl.head_dim % model.tp_size:
            raise ValueError(f"tp={model.tp_size} does not divide the n-gram head dim {pl.head_dim}")
        for j, rows in enumerate(pl.rows):  # n-gram tables, this rank's columns of every head
            setattr(model, f"ple_table{j}", p(rows, pl.head_dim // model.tp_size))


def _dense(ck, base: str, blk, rows=None, cols=None) -> torch.Tensor:
    """A checkpoint weight in fp32, dequantized when the checkpoint stores it FP8."""
    from .quant import dequant

    w, s = ck.linear(base, rows, cols, block=blk)
    return dequant(w, s, torch.float32) if s is not None else w.float()


def load_layer(layer, ck, pre: str, cfg, r: int, n: int, dtype) -> None:
    """Checkpoint names: zai-org/GLM-5.3-Flash model.safetensors.index.json (hc_attn_fn /
    _base / _scale, hc_ffn_*), Qwen/Qwen3.8-Flash-Next model.safetensors.index.json
    (attn_hyper_connection / mlp_hyper_connection .hc_norm / .input_mix_weight_down /
    .input_mix_weight_up / .block_inject_weight, mlp.shared_expert.*, mlp.shared_expert_gate)."""
    from .loader import _part

    hy, blk = cfg.hybrid, cfg.quant_block
    if hy.family == "glm5_next":
        for b in ("attn", "ffn"):
            getattr(layer, f"hc_{b}_fn").data.copy_(ck.get(f"{pre}hc_{b}_fn").to(dtype))
            getattr(layer, f"hc_{b}_base").data.copy_(ck.get(f"{pre}hc_{b}_base").float())
            getattr(layer, f"hc_{b}_scale").data.copy_(ck.get(f"{pre}hc_{b}_scale").float())
        return
    from . import qwen4_exp

    for b, name in (("attn", "attn_hyper_connection"), ("ffn", "mlp_hyper_connection")):
        m = f"{pre}{name}."
        getattr(layer, f"hc_{b}_norm").data.copy_(1.0 + ck.get(m + "hc_norm.weight").float())
        getattr(layer, f"hc_{b}_down").data.copy_(_dense(ck, m + "input_mix_weight_down", blk).to(dtype))
        getattr(layer, f"hc_{b}_up").data.copy_(_dense(ck, m + "input_mix_weight_up", blk).to(dtype))
        getattr(layer, f"hc_{b}_inject").data.copy_(_dense(ck, m + "block_inject_weight", blk).to(dtype))
    if layer.sh_gate_up is not None:
        sh = _part(hy.shared_expert, r, n)
        m = pre + "mlp.shared_expert."
        layer.sh_gate_up.data.copy_(torch.cat([_dense(ck, m + "gate_proj", blk, sh),
                                               _dense(ck, m + "up_proj", blk, sh)]).to(dtype))
        layer.sh_down.data.copy_(_dense(ck, m + "down_proj", blk, cols=sh).to(dtype))
        layer.sh_gate.data.copy_(_dense(ck, pre + "mlp.shared_expert_gate", blk).to(dtype))
    qwen4_exp.load_layer(layer, ck, pre, cfg, r, n, dtype, lambda base: _dense(ck, base, blk))


def load_model(model, ck, dtype) -> None:
    hy = model.cfg.hybrid
    if hy.family != "qwen4_exp":
        return
    from . import qwen4_exp

    m = "model.hyper_connection_mixer."
    blk = model.cfg.quant_block
    model.hc_final_norm.data.copy_(1.0 + ck.get(m + "hc_norm.weight").float())
    model.hc_final_down.data.copy_(_dense(ck, m + "input_mix_weight_down", blk).to(dtype))
    model.hc_final_up.data.copy_(_dense(ck, m + "input_mix_weight_up", blk).to(dtype))
    pl = hy.ple
    if pl is not None:
        for j, li in enumerate(pl.layers):
            qwen4_exp.load_table(getattr(model, f"ple_table{j}"), ck, f"model.layers.{li}.", pl, j, model.tp_rank,
                                 model.tp_size)


# -- per-sequence state the runner allocates ----------------------------------------------------


def aux_kv_shapes(model) -> list[tuple[int, ...]]:
    """Per-token caches beside K and V, in kv_layers() order: the QSA layers' raw indexer keys."""
    from .qwen4_exp import QSASpec

    return [(1, l.spec.index_head_dim) for l in model.kv_layers() if isinstance(l.spec, QSASpec)]


def bind_aux_kv(model, caches: list[torch.Tensor]) -> None:
    from .qwen4_exp import QSASpec

    layers = [l for l in model.kv_layers() if isinstance(l.spec, QSASpec)]
    for l, c in zip(layers, caches, strict=True):
        l.idx_cache = c


def _ple_layers(model) -> list:
    return [l for l in model.layers if getattr(l, "ple_slot", None) is not None]


def _open_pool_layers(model) -> list:
    """The pooled DSA layers in the minimal cache layout (models/mla.py KV_LAYOUT), in kv_layers() order."""
    from . import mla as _mla

    return [l for l in model.kv_layers() if l.spec.mla is not None and _mla.minimal_layout(l.spec.mla)]


def aux_state_shapes(model) -> list[tuple[tuple[int, ...], torch.dtype]]:
    """Per-request state rows beside the linear-attention ones: the PLE conv history, then each minimal-layout DSA
    layer's open pool (kpool rows of indexer key and gate logits, models/mla.py write_pool_keys_minimal)."""
    hy = model.cfg.hybrid
    out = [((hy.ple.conv_state_len, hy.hc * model.cfg.hidden_size), model.dtype) for _ in _ple_layers(model)]
    for l in _open_pool_layers(model):
        d = l.spec.mla.dsa
        out.append(((d.kpool, 2 * d.head_dim), model.dtype))
    return out


def bind_aux_state(model, states: list[torch.Tensor]) -> None:
    ple = _ple_layers(model)
    for l, s in zip(ple, states[: len(ple)], strict=True):
        l.ple_state = s
    for l, s in zip(_open_pool_layers(model), states[len(ple):], strict=True):
        l.open_pool = s


# -- forward -----------------------------------------------------------------------------------


def hidden_in(model, input_ids: torch.Tensor, ngram_ids: torch.Tensor | None, sp: bool = False) -> torch.Tensor:
    """The initial hidden state [T, hc * H (+ n-gram embedding rows)]: every stream starts as the
    token embedding (Glm5NextTextModel: expand; Qwen4ExpTextModel: repeat). sp: this rank's rows
    only (sequence-parallel prefill streams, models/decoder.py prefill_sp_enabled; no n-gram rows)."""
    hy = model.cfg.hybrid
    e = model._embed(input_ids)
    if sp:
        e = model._sp_rows(e)
    h = e.repeat(1, hy.hc)
    pl = hy.ple
    if pl is None:
        return h
    if ngram_ids is None:
        raise ValueError("this model's graphs need ngram_ids (qwen4_exp.ngram_ids, computed on the host)")
    T = input_ids.shape[0]
    parts = [h]
    for j in range(len(pl.layers)):
        ids = ngram_ids[:, j * pl.heads : (j + 1) * pl.heads]
        parts.append(F.embedding(ids, getattr(model, f"ple_table{j}")).reshape(T, -1))
    return torch.cat(parts, dim=-1)


def final(model, h: torch.Tensor) -> torch.Tensor:
    """The final normalised hidden state [T, H] the lm_head reads."""
    cfg = model.cfg
    hy, H = cfg.hybrid, cfg.hidden_size
    T = h.shape[0]
    streams = h[:, : hy.hc * H]
    if hy.family == "glm5_next":  # Glm5NextTextHyperHead (an unweighted mean), then the model norm
        return rms_norm(streams.view(T, hy.hc, H).mean(dim=1), model.norm, cfg.rms_norm_eps)
    x, _ = _gated(model, streams, model.hc_final_norm, model.hc_final_down, model.hc_final_up, None)
    return x


# How the hyper-connection stream arithmetic is written (KILN_MHC_FORM); the forms compute the same
# algebra.
# - "elementwise" (default): vLLM v0.30.0's numerics (vllm/model_executor/kernels/mhc/torch.py,
#   mhc_pre_torch / mhc_post_torch) with no [T, hc * H] fp32 temporary and no per-token batched matmul:
#   the RMS scale multiplies the [T, 2 hc + hc^2] logits after the fp32 projection (one scalar per
#   token, (x r) W^T = (x W^T) r), the collapse is hc per-token multiply-adds of [T, H] rows in fp32, and
#   each output stream post_n y + sum_m comb[m, n] S_m is accumulated in fp32 from [T, H] rows. The
#   collapse and the streams are rounded to the model dtype through a bitcast (_round): a plain
#   .to(bf16) between fp32 ops is folded away inside a device graph (XLA allows excess precision), so
#   the streams stayed fp32 across a whole layer group and the device scored differently from the CPU.
# - "elementwise_fp32": the same without the bitcast (the device then keeps excess precision; probe).
# - "bmm": transformers' spelling (Glm5NextTextHyperConnection / Glm5NextTextDecoderLayer, v5.18.0: the
#   normalised streams in fp32, a broadcast product summed over the streams, and the output as
#   post.to(dtype) * y + matmul(comb.to(dtype)^T, residual) in the model dtype).
# On trn1 the bmm output mix made neuronx-cc spill GLM-5.3-Flash's prefill layer graphs (layer 0 at
# C=512, 8 heads per rank: 10.7 ms, and about 80 ms once any NKI call sits in the mixer, against
# 3.8 ms; docs/neuron-notes.md "The layer-graph spill was the hyper-connection arithmetic").
MHC_FORMS = ("elementwise", "elementwise_fp32", "bmm")
MHC_FORM = os.environ.get("KILN_MHC_FORM", "elementwise")
if MHC_FORM not in MHC_FORMS:
    raise ValueError(f"KILN_MHC_FORM must be one of {MHC_FORMS}, not {MHC_FORM!r}")


def _round(x: torch.Tensor, dtype: torch.dtype, barrier: bool) -> torch.Tensor:
    """x.to(dtype); with barrier and a bf16 dtype, x is first rounded to bf16's 8 significant bits in
    fp32 arithmetic (Veltkamp's split: c = (2^16 + 1) x, hi = c - (c - x)), so the value is already
    bf16-representable whether or not the compiler keeps the convert (a convert pair f32 -> bf16 -> f32
    between elementwise ops is foldable, and a bitcast view to int16 is rejected inside an LNL graph:
    "Expected all tensors ... to be XLA tensors. Got: XLABFloat16Type")."""
    if barrier and dtype == torch.bfloat16 and x.dtype == torch.float32:
        c = x * 65537.0
        x = c - (c - x)
    return x.to(dtype)


def _mhc(model, layer, b: str, S: torch.Tensor):
    """Glm5NextTextHyperConnection.forward on streams S [T, hc, H]: (block input [T, H], post
    [T, hc] fp32, comb [T, hc, hc] fp32)."""
    hy = model.cfg.hybrid
    hc, eps = hy.hc, hy.hc_eps
    T = S.shape[0]
    if MHC_FORM != "bmm":
        flat = S.reshape(T, -1)
        r = torch.rsqrt(flat.float().pow(2).mean(-1, keepdim=True) + model.cfg.rms_norm_eps)
        mix = F.linear(flat.float(), getattr(layer, f"hc_{b}_fn").float()) * r  # (x r) W^T = (x W^T) r
    else:
        flat = S.reshape(T, -1).float()
        flat = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + model.cfg.rms_norm_eps)  # unweighted RMSNorm
        mix = F.linear(flat, getattr(layer, f"hc_{b}_fn").float())
    base, scale = getattr(layer, f"hc_{b}_base"), getattr(layer, f"hc_{b}_scale")
    pre = torch.sigmoid(mix[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mix[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = torch.softmax(mix[:, 2 * hc :].view(T, hc, hc) * scale[2] + base[2 * hc :].view(hc, hc), dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)  # Sinkhorn-Knopp: columns, then rows and columns
    for _ in range(hy.sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    if MHC_FORM != "bmm":
        x = pre[:, 0:1] * S[:, 0].float()
        for n in range(1, hc):
            x = x + pre[:, n : n + 1] * S[:, n].float()
        return _round(x, S.dtype, MHC_FORM == "elementwise"), post, comb
    x = (pre.unsqueeze(-1) * S).sum(dim=1).to(S.dtype)
    return x, post, comb


def mix_out(post: torch.Tensor, comb: torch.Tensor, y: torch.Tensor, S: torch.Tensor, form: str) -> torch.Tensor:
    """The new streams [T, hc * H] from the block output y [T, H], the old streams S [T, hc, H], post
    [T, hc] and comb [T, hc, hc] (fp32): stream n = post_n y + sum_m comb[m, n] S_m (KILN_MHC_FORM)."""
    T, hc, H = S.shape
    if form in ("elementwise", "elementwise_fp32"):  # stream n: post_n y + sum_m comb[m, n] S_m, in fp32
        yf, outs = y.float(), []
        for n in range(hc):
            o = post[:, n : n + 1] * yf
            for m in range(hc):
                o = o + comb[:, m, n : n + 1] * S[:, m].float()
            outs.append(_round(o, S.dtype, form == "elementwise"))
        return torch.cat(outs, dim=-1)
    out = post.to(S.dtype).unsqueeze(-1) * y.unsqueeze(1) + comb.to(S.dtype).transpose(-1, -2) @ S
    return out.reshape(T, hc * H)


def _gated(model, h: torch.Tensor, w_norm, w_down, w_up, w_inject):
    """Qwen4ExpTextGatedResidual.forward on h [T, hc * H]: (block input [T, H], injection weights
    [T, hc] or None without block_inject_weight)."""
    from .qwen4_exp import norm

    hy, H = model.cfg.hybrid, model.cfg.hidden_size
    T = h.shape[0]
    hn = norm(h, w_norm, model.cfg.rms_norm_eps, H)
    g = F.silu(F.linear(hn, w_down) / hy.hc)
    g = torch.sigmoid(F.linear(g, w_up)).view(T, hy.hc, H)
    x = (g * hn.view(T, hy.hc, H)).mean(dim=-2)
    if w_inject is None:
        return x, None
    return x, 2 * torch.sigmoid(F.linear(hn, w_inject) / hy.hc)


# KILN_SP_ROUTE (default 1): with sequence-parallel prefill streams, an MoE layer's router runs on each
# rank's own R / tp rows and the [R, 2k] routing is gathered like the rows themselves (_sp_route), instead
# of every rank routing all R gathered rows (1.13 ms of an MoE block at R = 4096 on trn1, 2026-10-04). The
# per-row arithmetic is the same; only the router matmul's row count differs. Read when a graph is traced.
def sp_route_enabled() -> bool:
    return os.environ.get("KILN_SP_ROUTE", "1") == "1"


# KILN_SP_RS (default 1): with sequence-parallel prefill streams, a block's output reduction (the MLP's world
# all-reduce, the token mixers' zero-padded DP all-reduce) is a world reduce-scatter that leaves each rank its
# own rows, instead of the all-reduce of every row followed by DecoderForCausalLM._sp_rows. The same sums;
# the collective's own summation order may differ in the last bf16 bit.
def sp_rs_enabled() -> bool:
    return os.environ.get("KILN_SP_RS", "1") == "1"


def _sp_route(model, layer, x: torch.Tensor):
    """(topv, topi) of every rank's rows, in rank order, from this rank's rows x [r, H]: routed here and
    gathered as one fp32 [R, 2k] tensor (DecoderForCausalLM._sp_gather: one rank's value plus zeros per
    element, so the weights and the expert indices, integers below 2^24, come back exactly). A layer with
    redundant expert slots (models/eplb.py) counts this rank's pairs per expert into ep_stats when recording, and
    maps its ids to physical ones before the gather (only r rows here): then (topv, topi, True)."""
    topv, topi = model._route(layer, x)
    k = topi.shape[-1]
    phys = bool(getattr(layer, "ep_s", 0))
    if phys:
        from . import eplb

        st = getattr(layer, "ep_stats", None)
        if st is not None and eplb.record_enabled():
            st.add_(eplb.counts(topi, st.shape[0]))
        topi = model._ep_phys(layer, topi, decode=False)
    both = model._sp_gather(torch.cat([topv.float(), topi.float()], dim=-1))
    out = both[:, :k].to(topv.dtype), both[:, k:].to(topi.dtype)
    return (*out, True) if phys else out


def _sp_out(model, fn, rs: bool):
    """With sequence-parallel streams, a block fn over every rank's rows y [R, ...] -> this rank's rows of its
    output: its output reduction a reduce-scatter (rs, KILN_SP_RS) or the all-reduce and _sp_rows."""
    if rs:
        return lambda y, *a: model._sp_rs_call(fn, y, *a)
    return lambda y, *a: model._sp_rows(fn(y, *a))


def _attn_rows(model, fn, sp: bool):
    """The token mixer's block fn for _block: fn, or with sequence-parallel streams (sp) on this rank's rows:
    its group's rows gathered and the output reduce-scattered inside the attention group (KILN_SP_GROUP), or
    every row gathered and the output reduce-scattered (KILN_SP_RS) or all-reduced and this rank's taken."""
    if not sp:
        return fn
    if getattr(model, "sp_group", False) and model._sp_grp_ok():  # KILN_SP_GROUP (models/decoder.py sp_group_enabled)
        return lambda x: model._sp_grp_call(fn, model._sp_group_gather(x))
    out = _sp_out(model, fn, sp_rs_enabled() and model._sp_rs_attn())
    return lambda x: out(model._sp_gather(x))


def _ffn(model, layer, sp: bool):
    """The FFN block's fn for _block: _mlp, or with sequence-parallel streams (sp) on this rank's rows,
    gathering them, keeping its own rows of the output, and (SP_ROUTE, an MoE layer) routing them here."""
    if not sp:
        return lambda x: _mlp(model, layer, x)
    out = _sp_out(model, lambda y, r=None: _mlp(model, layer, y, r), sp_rs_enabled())
    if sp_route_enabled() and layer.moe and model.cfg.hybrid.swiglu_limit is not None:
        return lambda x: out(model._sp_gather(x), _sp_route(model, layer, x))
    return lambda x: out(model._sp_gather(x))


def _block(model, layer, b: str, streams: torch.Tensor, norm_w, fn) -> torch.Tensor:
    """streams [T, hc * H] after one block fn(x) -> [T, H] wrapped in its hyper-connection."""
    hy, H = model.cfg.hybrid, model.cfg.hidden_size
    T = streams.shape[0]
    if hy.family == "glm5_next":
        S = streams.view(T, hy.hc, H)
        x, post, comb = _mhc(model, layer, b, S)
        y = fn(rms_norm(x, norm_w, model.cfg.rms_norm_eps))
        return mix_out(post, comb, y, S, MHC_FORM)
    x, inj = _gated(model, streams, getattr(layer, f"hc_{b}_norm"), getattr(layer, f"hc_{b}_down"),
                    getattr(layer, f"hc_{b}_up"), getattr(layer, f"hc_{b}_inject"))
    y = fn(x)
    return streams + (y.unsqueeze(1) * inj.unsqueeze(-1)).reshape(T, hy.hc * H)


def _mix(model, layer, x, positions, slot_mapping, table, bias, state_slot, seq, mixed=None):
    """The token mixer on its input x [T, H], after its attention-group all-reduce. mixed: the decode
    rows of a mixed batch (DecoderForCausalLM._layer)."""
    from . import linear_attn, qwen4_exp
    from . import mla as _mla

    sp = layer.spec
    if isinstance(sp, LinearSpec):  # selects its rows itself
        return linear_attn.mix(model, layer, x, positions, slot_mapping, state_slot, mixed)
    x = model._attn_in(x)  # DP attention: this group's rows
    if sp.mla is not None:
        if seq is not None:
            return model._attn_all_reduce(_mla.reference(model, layer, x, positions, seq))
        if _mla.minimal_layout(sp.mla):  # its open pool lives in the request's state row
            return model._attn_all_reduce(_mla.attention(model, layer, x, positions, slot_mapping, table, bias,
                                                         mixed=mixed, state_slot=state_slot))
        return model._attn_all_reduce(_mla.attention(model, layer, x, positions, slot_mapping, table, bias,
                                                     mixed=mixed))
    if mixed is not None:
        raise NotImplementedError("mixed batches for Qwen3.8-Flash-Next's attention layers")
    return qwen4_exp.attention(model, layer, x, positions, slot_mapping, table, bias, seq)


def _swiglu(model, layer, x, gate_up: str, down: str, limit: float | None) -> torch.Tensor:
    gu = F.linear(x, model._w(layer, gate_up))
    n = gu.shape[-1] // 2
    g, u = gu[..., :n], gu[..., n:]
    if limit is not None:  # Glm5NextTextMLP / Glm5NextTextExperts._apply_gate
        g, u = g.clamp(max=limit), u.clamp(min=-limit, max=limit)
    return F.linear(F.silu(g) * u, model._w(layer, down))


def _moe_clamped(model, layer, x: torch.Tensor, limit: float, route=None) -> torch.Tensor:
    """DecoderForCausalLM._moe_routed with Glm5NextTextExperts' clamped SwiGLU, both of its forms.
    route: (topv, topi) of x's rows made elsewhere (_sp_route), instead of routing x here."""
    cfg = model.cfg
    topv, topi = model._route(layer, x) if route is None else route[:2]
    phys = route is not None and len(route) > 2 and route[2]  # _sp_route mapped the ids already (models/eplb.py)
    if getattr(layer, "moe_ep", False):  # expert parallel (models/decoder.py moe_ep_enabled)
        return model._moe_ep(layer, x, topv, topi, 1, limit, phys=phys)
    T, H = x.shape
    k, Im = cfg.num_experts_per_tok, model.moe_inter

    if getattr(layer, "moe_tiles", False) and x.device.type != "cpu":  # the NKI kernels, clamped
        from ..kernels.moe_dedupe import ACTS, moe_dedupe
        from .decoder import MOE_PREFILL_KERNEL, MOE_PREFILL_MIN_TOKENS, moe_prefill_down

        if MOE_PREFILL_KERNEL == "nki" and T >= MOE_PREFILL_MIN_TOKENS:
            from ..kernels.moe_prefill import moe_prefill

            return moe_prefill(x, topv, topi, layer.w_blob, act=ACTS["silu_clamp"], limit=limit,
                               dq=layer.moe_prefill_dq, down=moe_prefill_down(layer))
        return moe_dedupe(x, topv, topi, layer.w_blob, act=ACTS["silu_clamp"], limit=limit)

    def act(gu):
        return F.silu(gu[..., :Im].clamp(max=limit)) * gu[..., Im:].clamp(min=-limit, max=limit)

    if T * k <= model.MOE_GATHER_MAX_PAIRS:
        flat = topi.reshape(T * k)
        xs = x.unsqueeze(1).expand(T, k, H).reshape(T * k, H, 1)
        a = act(torch.bmm(model._experts(layer, "w_gu", flat), xs).squeeze(-1))
        y = torch.bmm(a.unsqueeze(1), model._experts(layer, "w_down", flat)).squeeze(1)
        return (y.view(T, k, H) * topv.unsqueeze(-1)).sum(dim=1)
    a = act(torch.einsum("th,eih->eti", x, model._experts(layer, "w_gu")))
    ye = torch.einsum("eti,eih->eth", a, model._experts(layer, "w_down"))
    w = torch.zeros(T, cfg.num_experts, dtype=model.dtype, device=x.device).scatter(1, topi, topv)
    return torch.einsum("eth,te->th", ye, w)


def _mlp(model, layer, x: torch.Tensor, route=None) -> torch.Tensor:
    """The MLP block on its input x [T, H], after its all-reduce; route: _moe_clamped's."""
    hy = model.cfg.hybrid
    lim = hy.swiglu_limit
    if not layer.moe:
        return model._out_reduce(_swiglu(model, layer, x, "gate_up", "down", lim))
    y = model._moe(layer, x) if lim is None else _moe_clamped(model, layer, x, lim, route)
    if layer.shared_gate_up is not None:  # glm5_next: mlp.shared_experts, clamped like the experts
        y = y + _swiglu(model, layer, x, "shared_gate_up", "shared_down", lim)
    if getattr(layer, "sh_gate_up", None) is not None:  # qwen4_exp: sigmoid(shared_expert_gate(x)) * shared_expert(x)
        y = y + torch.sigmoid(F.linear(x, layer.sh_gate)) * _swiglu(model, layer, x, "sh_gate_up", "sh_down", None)
    return model._out_reduce(y)


def layer(model, layer, h, positions, slot_mapping, table, bias, state_slot=None, seq: dict | None = None,
          mixed=None):
    """One decoder layer of a hyper-connection model over the hidden state h [T, hc * H (+ n-gram
    rows)], every batch form of DecoderForCausalLM._layer; seq (a dict, MLA's top-k carrier) for the
    whole-sequence reference form without a cache. mixed (a mixed batch, DecoderForCausalLM._layer):
    the streams, the hyper-connections and the MLP run over the chunk's and the decode rows together,
    the token mixer splits them; with sequence-parallel streams each rank holds its 1 / tp of all of
    them (positions covers the group's C + D rows, so the test below holds as for a chunk)."""
    hy, H = model.cfg.hybrid, model.cfg.hidden_size
    W = hy.hc * H
    streams = h[:, :W]
    if getattr(layer, "ple_slot", None) is not None:
        from . import qwen4_exp

        e = hy.ple.embed_dim // model.tp_size
        emb = h[:, W + layer.ple_slot * e : W + (layer.ple_slot + 1) * e]
        streams = qwen4_exp.ple(model, layer, streams, emb, positions, slot_mapping, state_slot)
    if mixed is not None and seq is None and model._sp_on() and MIXED_SP == "split":
        return _layer_mixed_split(model, layer, h, positions, slot_mapping, table, bias, state_slot, mixed)
    # Sequence-parallel prefill streams (models/decoder.py prefill_sp_enabled): h holds this rank's R / tp of a
    # chunk's R rows (R = DP-attention groups x the group's chunk). The hyper-connections run on those
    # rows; each block gathers every rank's normalised input, runs over all R rows exactly as with
    # replicated streams (its own reduction included) and keeps this rank's rows of the output.
    # The same for a decode call with KILN_DECODE_SP (models/decoder.py decode_sp_enabled): R = groups x B rows.
    sp = (seq is None and table is not None and h.shape[0] * model.tp_size == positions.shape[0] * model.dp
          and ((table.dim() == 1 and model._sp_on())
               or (table.dim() == 2 and bias is not None and bias.dim() != 5 and model._sp_on_decode())))

    def rows(fn):
        return _attn_rows(model, fn, sp)

    streams = _block(model, layer, "attn", streams, layer.in_norm,
                     rows(lambda x: _mix(model, layer, x, positions, slot_mapping, table, bias, state_slot, seq,
                                         mixed)))
    streams = _block(model, layer, "ffn", streams, layer.post_norm, _ffn(model, layer, sp))
    return torch.cat([streams, h[:, W:]], dim=-1) if h.shape[1] > W else streams


# How a mixed batch (DecoderForCausalLM.forward_mixed) lays out its sequence-parallel streams (KILN_MIXED_SP):
# - "rows" (default): every row of the call, the decode rows included, in the sequence-parallel partition
#   (each rank holds N (C + D) / tp of them), exactly as a prefill chunk's rows.
# - "split": each rank holds its N C / tp of the chunks' rows and ALL N D decode rows (the decode graphs'
#   replicated streams), so the chunk part keeps the prefill graph's rows per rank. Measured worse on
#   GLM-5.3-Flash at the G64 sweep shapes (tp=32, DP 4, 1024 + 16 rows per group, trn1, 2026-10-04,
#   docs/neuron-notes.md "Mixed batches"): the gathers and splits around every block took the 12-layer
#   mixed groups from 4.8-4.9M to 7.0-8.6M queue-instance spill runs, and the configuration no longer
#   loaded (Allocation Failure).
MIXED_SP = os.environ.get("KILN_MIXED_SP", "rows")
if MIXED_SP not in ("rows", "split"):
    raise ValueError(f"KILN_MIXED_SP must be rows or split, not {MIXED_SP!r}")


def mixed_split_rows(model, positions: torch.Tensor, dec_rows: int) -> tuple[int, int, int]:
    """(C, D, this rank's chunk rows) of a mixed batch in the "split" layout (MIXED_SP)."""
    C = positions.shape[0] - dec_rows
    return C, dec_rows, model.dp * C // model.tp_size


def mixed_split_in(model, x: torch.Tensor, C: int, D: int) -> torch.Tensor:
    """Every group's rows of a mixed batch [N (C + D), ...], group-major, as the "split" layout keeps them:
    this rank's sequence-parallel rows of the chunks, then every decode row."""
    N = model.dp
    xg = x.view(N, C + D, *x.shape[1:])
    return torch.cat([model._sp_rows(xg[:, :C].reshape(N * C, *x.shape[1:])), xg[:, C:].reshape(N * D, *x.shape[1:])])


def mixed_split_out(model, x: torch.Tensor, C: int, D: int) -> torch.Tensor:
    """The inverse of mixed_split_in: every rank's chunk rows gathered (one world all-reduce, _sp_gather) and
    put back group-major beside the decode rows, [N (C + D), ...]."""
    N = model.dp
    rc = N * C // model.tp_size
    full = model._sp_gather(x[:rc]).view(N, C, *x.shape[1:])
    return torch.cat([full, x[rc:].view(N, D, *x.shape[1:])], 1).reshape(N * (C + D), *x.shape[1:])


def _layer_mixed_split(model, layer, h, positions, slot_mapping, table, bias, state_slot, mixed):
    """layer() on a mixed batch in the "split" sequence-parallel layout (MIXED_SP): the hyper-connections of the
    chunk rows (this rank's) and of the decode rows (all of them) run apart, row for row the same arithmetic;
    each block takes every group's rows in the call's order, exactly as with replicated streams, and each part
    keeps its rows of the output."""
    hy, H, eps = model.cfg.hybrid, model.cfg.hidden_size, model.cfg.rms_norm_eps
    if hy.family != "glm5_next" or h.shape[1] != hy.hc * H:
        raise NotImplementedError("the split mixed layout is GLM-5.3-Flash's mHC streams only")
    C, D, rc = mixed_split_rows(model, positions, mixed[0].shape[0])

    def block(b, streams, norm_w, fn):
        S = streams.view(streams.shape[0], hy.hc, H)
        Sc, Sd = S[:rc], S[rc:]
        xc, pc, cc = _mhc(model, layer, b, Sc)
        xd, pd, cd = _mhc(model, layer, b, Sd)
        y = mixed_split_in(model, fn(mixed_split_out(model, torch.cat([rms_norm(xc, norm_w, eps),
                                                                        rms_norm(xd, norm_w, eps)]), C, D)), C, D)
        return torch.cat([mix_out(pc, cc, y[:rc], Sc, MHC_FORM), mix_out(pd, cd, y[rc:], Sd, MHC_FORM)])

    streams = block("attn", h, layer.in_norm,
                    lambda x: _mix(model, layer, x, positions, slot_mapping, table, bias, state_slot, None, mixed))
    return block("ffn", streams, layer.post_norm, lambda x: _mlp(model, layer, x))


def forward_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Logits of a whole sequence without a cache (tests): every layer's sequence form."""
    T = input_ids.shape[0]
    positions = torch.arange(T, device=input_ids.device)
    ngram = None
    pl = model.cfg.hybrid.ple
    if pl is not None:
        from .qwen4_exp import ngram_ids

        ngram = torch.from_numpy(ngram_ids(pl, input_ids.tolist(), 0, T)).to(input_ids.device)
    h = hidden_in(model, input_ids, ngram)
    seq: dict = {}
    for l in model.layers:
        h = layer(model, l, h, positions, None, None, None, None, seq)
    return model._head(final(model, h))
