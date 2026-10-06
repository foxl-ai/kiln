"""Load Hugging Face safetensors checkpoints into Kiln models."""

from __future__ import annotations

import glob
import os

import torch
from safetensors import safe_open

from ..config import LinearSpec, ModelConfig
from . import hybrid, linear_attn, mla
from .decoder import DecoderForCausalLM
from .quant import FP8, dequant, fit_e4m3_max, mxfp4_unpack, quantize_fp8_rows

ARCHITECTURES = {a: DecoderForCausalLM for a in (
    "LlamaForCausalLM", "MistralForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM",
    "MiMoV2ForCausalLM", "GptOssForCausalLM", "HYV3ForCausalLM", "K2HorizonForCausalLM",
    "InklingForConditionalGeneration", *linear_attn.ARCHITECTURES, *mla.ARCHITECTURES, *hybrid.ARCHITECTURES)}

# Checkpoint tensor names that differ by architecture (suffixes after model.layers.{i}.).
NAMES = {
    "sink": "self_attn.attention_sink_bias",  # MiMo-V2
    "router": "mlp.gate.weight",
    "router_bias": "mlp.gate.e_score_correction_bias",
    "shared": "mlp.shared_experts.",
}
ARCH_NAMES = {
    # https://huggingface.co/openai/gpt-oss-120b/raw/main/model.safetensors.index.json
    "GptOssForCausalLM": {"sink": "self_attn.sinks", "router": "mlp.router.weight",
                          "router_logit_bias": "mlp.router.bias"},
    # https://huggingface.co/tencent/Hy3/raw/main/model.safetensors.index.json (transformers 5.15
    # conversion_mapping.py renames these to mlp.gate.weight / e_score_correction_bias / shared_experts)
    "HYV3ForCausalLM": {"router": "mlp.router.gate.weight", "router_bias": "mlp.expert_bias",
                        "shared": "mlp.shared_mlp."},
    # https://huggingface.co/IFM/K2-Horizon-375B-A23B/raw/main/model.safetensors.index.json: the
    # gate's bias is the selection bias (modeling_k2_horizon.py K2HorizonSparseMoeBlock)
    "K2HorizonForCausalLM": {"router_bias": "mlp.gate.bias"},
}


def names(cfg: ModelConfig) -> dict:
    return {**NAMES, **ARCH_NAMES.get(cfg.architecture, {})}


def resolve_model_path(model: str) -> str:
    """A local directory, or a Hugging Face repo id downloaded to the local cache."""
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model"])


class _Alias:
    """A safetensors handle that serves one tensor under another name."""

    def __init__(self, handle, name: str):
        self._h, self._name = handle, name

    def get_slice(self, _name: str):
        return self._h.get_slice(self._name)

    def get_tensor(self, _name: str):
        return self._h.get_tensor(self._name)


class _Checkpoint:
    """Lazy view of a safetensors checkpoint. A tensor-parallel rank reads only its own rows
    or columns of each weight (safetensors get_slice reads just that part through mmap), so
    host memory per rank is its shard, not the model: eight ranks each materialising the
    whole of Qwen3-30B-A3B (61 GB in bf16) exhausted a 495 GB trn1.32xlarge and rebooted it."""

    def __init__(self, path: str):
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {path}")
        self._handles = [safe_open(fn, framework="pt") for fn in files]
        self._where = {name: h for h in self._handles for name in h.keys()}
        # Vision-language checkpoints keep the language model under model.language_model.
        # (Qwen/Qwen3.5-0.8B model.safetensors.index.json); read it under the usual model. names.
        vl = "model.language_model."
        for name in [n for n in self._where if n.startswith(vl)]:
            self._where.setdefault("model." + name[len(vl):], _Alias(self._where[name], name))

    def __contains__(self, name: str) -> bool:
        return name in self._where

    def shape(self, name: str) -> list[int]:
        return self._where[name].get_slice(name).get_shape()

    def linear(self, base: str, rows: slice | None = None, cols: slice | None = None, block: int | None = None):
        """Weight `base`.weight in Kiln's format: (w, scale) with scale fp32 [n, k / bk] for a
        quantized checkpoint weight (FP8 128x128 blocks or MXFP4, see models/quant.py), else
        (w, None). rows / cols select a tensor-parallel shard. A column shard of an FP8 weight
        narrower than or misaligned with its blocks is cut exactly when it lies inside one block
        and returned dequantized (scale None) when it straddles block edges; an MXFP4 column
        shard must be 32-aligned."""
        wname = base + ".weight"
        if base + ".weight_scale_inv" in self:  # FP8, square blocks
            bs = block or 128
            n_full = self.shape(wname)[0]
            w = self.get(wname, rows)
            s = _row_scales(self.get(base + ".weight_scale_inv").float(), n_full, bs)
            s = s[rows] if rows is not None else s
            bk = bs
        elif base + ".weight_scale" in self and str(self._where[wname].get_slice(wname).get_dtype()) == "F8_E4M3":
            # Per-tensor (or per-output-channel) FP8: w = w_fp8 * weight_scale, the compressed-tensors
            # convention vLLM v0.30.0 dequantizes as weight_fp8 * weight_scale
            # (model_executor/layers/quantization/fp8.py, Fp8LinearMethod). Constant along K, so any
            # column shard is exact and the scale is laid out in `block`-wide column blocks.
            bk = block or 128
            w = self.get(wname, rows, cols)
            s = self.get(base + ".weight_scale").float().reshape(-1, 1)
            if s.shape[0] > 1:  # per channel
                s = s[rows] if rows is not None else s
            return w, s.expand(w.shape[0], -(-w.shape[1] // bk)).contiguous()
        elif base + ".weight_scale" in self and str(self._where[wname].get_slice(wname).get_dtype()) == "U8":
            packed, exps = self.get(wname, rows), self.get(base + ".weight_scale", rows)
            if cols is not None:
                # Cut the shard BEFORE unpacking: unpacking whole MXFP4 rows and then cutting
                # took 563 s for one rank of MiMo-V2.6-Flash at tp=32.
                if cols.start % 32 or cols.stop % 32:
                    raise ValueError(f"{base}: column shard {cols} is not aligned to 32-wide MXFP4 blocks")
                packed = packed[:, cols.start // 2 : cols.stop // 2]
                exps = exps[:, cols.start // 32 : cols.stop // 32]
                cols = None
            w, s = mxfp4_unpack(packed, exps)
            bk = 32
        else:
            return self.get(wname, rows, cols), None
        if cols is not None:
            if cols.start % bk == 0 and (cols.stop % bk == 0 or cols.stop == w.shape[1]):
                w, s = w[:, cols], s[:, cols.start // bk : -(-cols.stop // bk)]
            elif cols.start // bk == (cols.stop - 1) // bk:
                # A shard inside one block (GLM-5.3-Flash experts at tp=32: 64 of a 128-wide
                # block) keeps that block's scale: exact, one scale column over the shard.
                b = cols.start // bk
                w, s = w[:, cols], s[:, b : b + 1]
            else:
                # A shard straddling block edges (Qwen3.8-Flash-Next experts at tp=16: 40
                # columns from 120) cannot keep the checkpoint's blocks in [n, k / bk] form:
                # hand back the dequantized values and let _assign quantize them again at the
                # rank's own block width (one extra FP8 rounding).
                sc = s.repeat_interleave(bk, dim=1)[:, : w.shape[1]]
                return (w[:, cols].float() * sc[:, cols]), None
        return w, s

    def mxfp4(self, base: str, rows: slice | None = None, cols: slice | None = None):
        """An MXFP4 weight kept packed: (uint8 [n, k/2], bf16 power-of-two scales [n, k/32])."""
        wname = base + ".weight"
        if base + ".weight_scale" not in self or str(self._where[wname].get_slice(wname).get_dtype()) != "U8":
            raise ValueError(f"{base}: expected an MXFP4 weight (U8 codes + E8M0 weight_scale)")
        packed, exps = self.get(wname, rows), self.get(base + ".weight_scale", rows)
        if cols is not None:
            if cols.start % 32 or cols.stop % 32:
                raise ValueError(f"{base}: column shard {cols} is not aligned to 32-wide MXFP4 blocks")
            packed = packed[:, cols.start // 2 : cols.stop // 2]
            exps = exps[:, cols.start // 32 : cols.stop // 32]
        # E8M0 0xFF encodes NaN (OCP Microscaling Formats v1.0, E8M0 scale); 2^(e - 127) is exact in bf16.
        if bool((exps == 255).any()):
            raise ValueError(f"{base}: NaN (E8M0 255) block scale")
        return packed.contiguous(), torch.exp2(exps.to(torch.float32) - 127.0).to(torch.bfloat16)

    def get(self, name: str, rows: slice | None = None, cols: slice | None = None) -> torch.Tensor:
        if name not in self._where:
            raise KeyError(name)
        if rows is None and cols is None:
            return self._where[name].get_tensor(name)  # get_slice()[:] refuses 0-dim tensors (scalar scales)
        sl = self._where[name].get_slice(name)
        if cols is None:
            return sl[rows]  # contiguous rows: read only those
        # A column slice through get_slice is a strided read and measured far slower than
        # reading the rows and cutting in torch (py-spy: 8 ranks stuck >10 min on expert
        # down_proj column slices of Qwen3-30B-A3B), so read rows, then cut columns.
        return sl[rows or slice(None)][:, cols]


def load_model(path: str, cfg: ModelConfig, dtype: torch.dtype, device: torch.device, max_positions: int,
               tp_rank: int = 0, tp_size: int = 1, tp_group=None, keep_fp8: bool = False,
               fp8_max: float = 240.0, vocab_parallel: bool = False, packed_mxfp4: bool = False,
               mtp: bool = False, moe_kernel: str = "xla", attn_tp: int | None = None, attn_group=None,
               dp_attention: int = 1, max_num_seqs: int | None = None, pd_role: str | None = None):
    """keep_fp8: weights the checkpoint quantizes stay FP8 on the device (dequantized in-graph);
    otherwise they are dequantized to `dtype` here. fp8_max: the device's largest finite e4m3.
    attn_tp / attn_group: the attention TP (DecoderForCausalLM; None = its default); dp_attention:
    DP-attention groups (DecoderForCausalLM: DP attention); max_num_seqs: the engine's (the automatic
    expert-parallel default, decoder.moe_ep_enabled)."""
    cls = ARCHITECTURES.get(cfg.architecture)
    if cls is None:
        raise NotImplementedError(f"architecture {cfg.architecture} is not supported (have: {sorted(ARCHITECTURES)})")
    # Build on the meta device and materialise piece by piece: a module is allocated on the
    # host, filled from the checkpoint and moved to `device` before the next one is
    # allocated, so host memory peaks at one layer per rank instead of the whole shard
    # (12.6 GB per rank for MiMo-V2.6-Flash at tp=32, 32 ranks on a 495 GB host).
    with torch.device("meta"):
        model = cls(cfg, dtype, max_positions, tp_rank, tp_size, tp_group, keep_fp8=keep_fp8,
                    vocab_parallel=vocab_parallel, packed_mxfp4=packed_mxfp4, mtp=mtp, moe_kernel=moe_kernel,
                    attn_tp=attn_tp, attn_group=attn_group, dp_attention=dp_attention, max_num_seqs=max_num_seqs,
                    pd_role=pd_role)
    model.materialize = lambda mod: _materialize(mod, device)
    _load_decoder(model, path, dtype, fp8_max)
    for name, (cos, sin) in model.rope_tables(max_positions).items():
        model.register_buffer(f"rope_cos{name}", cos.to(dtype).to(device), persistent=False)
        model.register_buffer(f"rope_sin{name}", sin.to(dtype).to(device), persistent=False)
    model.register_buffer("vocab_start_t", torch.tensor(model.vocab_start, dtype=torch.int64).to(device),
                          persistent=False)
    model.dp_buffers(device)
    del model.materialize
    stray = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.device != device]
    if stray:
        raise RuntimeError(f"{len(stray)} tensors not on {device} after loading, e.g. {stray[:3]}")
    return model.eval()


def _materialize(mod: torch.nn.Module, device: torch.device):
    """Give `mod`'s own (non-recursive) meta parameters host storage; returns a function
    that moves them to `device` once they are filled."""
    params = [(n, p) for n, p in mod.named_parameters(recurse=False) if p.device.type == "meta"]
    for n, p in params:
        setattr(mod, n, torch.nn.Parameter(torch.empty(p.shape, dtype=p.dtype), requires_grad=False))

    def done():
        # Every host parameter, including ones that replaced the materialised ones while the
        # module was filled (DecoderLayer.pack_experts swaps four expert tensors for w_blob).
        for n, p in list(mod.named_parameters(recurse=False)):
            if p.device != device:
                setattr(mod, n, torch.nn.Parameter(p.data.to(device), requires_grad=False))

    return done


def _row_scales(scale_inv: torch.Tensor, n_rows: int, bs: int) -> torch.Tensor:
    """Per-row scales [n_rows, K / bs] from 128 x 128 block scales.

    A weight saved from T tensor-parallel shards may restart its row blocks at every shard:
    MiMo-V2.6-Flash's full-attention qkv_proj has 13568 rows in 4 shards of 3392 (26.5 blocks)
    and 108 scale rows = 4 x 27, not 106 (SGLang v0.5.21 srt/models/mimo_v2.py,
    _resolve_deferred_qkv_scale_inv, chunks the scales per checkpoint shard). The shard count
    is the one that explains the scale row count."""
    nb = -(-n_rows // bs)
    if scale_inv.shape[0] == nb:
        return scale_inv.repeat_interleave(bs, dim=0)[:n_rows]
    for T in range(2, 129):
        if n_rows % T == 0 and T * -(-(n_rows // T) // bs) == scale_inv.shape[0]:
            per, nbs = n_rows // T, -(-(n_rows // T) // bs)
            return torch.cat([scale_inv[t * nbs : (t + 1) * nbs].repeat_interleave(bs, dim=0)[:per] for t in range(T)])
    raise ValueError(f"{scale_inv.shape[0]} block-scale rows do not fit {n_rows} rows in {bs}-row blocks")


def _grouped_qkv_rows(sp, qs: slice, layer, ckpt_tp: int) -> list[slice]:
    """Row slices of a fused qkv_proj for this rank's q heads (qs, in q-only row units), then
    its KV heads' k rows, then their v rows.

    MiMo-V2 checkpoints store the fused projection TP-interleaved: ckpt_tp contiguous chunks,
    chunk c holding [Q | K | V] of query heads c * nh / ckpt_tp ... and KV heads
    c * nkv / ckpt_tp ..., with ckpt_tp = the config's top-level num_key_value_heads for EVERY
    layer (SGLang v0.5.21 srt/configs/model_config.py get_mimo_v2_fused_qkv_expected_tp_size,
    srt/models/mimo_v2.py _get_ckpt_qkv_shard_sizes; vLLM v0.30.0's _shard_fp8_qkv_proj describes
    the full-attention case, where a chunk is one KV head). Measured on MiMo-V2.6-Flash-RL: the
    per-64-row weight magnitude profile repeats every 3392 rows in layer 0 (Q 16 heads + K 1 +
    V 1) and every 3712 in sliding-window layer 1 (Q 16 + K 2 + V 2), i.e. 4 chunks in both."""
    Dk, Dv = sp.head_dim, sp.v_head_dim
    nh, nkv = sp.num_heads, sp.num_kv_heads
    if nh % ckpt_tp or nkv % ckpt_tp:
        raise ValueError(f"fused qkv_proj: {nh} q / {nkv} kv heads do not split into {ckpt_tp} checkpoint shards")
    qc, kc = nh // ckpt_tp, nkv // ckpt_tp  # heads per chunk
    stride = qc * Dk + kc * Dk + kc * Dv

    def merge(slices):
        out = []
        for sl in slices:
            if out and out[-1].stop == sl.start:
                out[-1] = slice(out[-1].start, sl.stop)
            else:
                out.append(sl)
        return out

    q = [slice((h // qc) * stride + (h % qc) * Dk, (h // qc) * stride + (h % qc + 1) * Dk)
         for h in range(qs.start // Dk, qs.stop // Dk)]
    kv = range(layer.kv_offset, layer.kv_offset + layer.nkv)
    k = [slice((j // kc) * stride + qc * Dk + (j % kc) * Dk, (j // kc) * stride + qc * Dk + (j % kc + 1) * Dk)
         for j in kv]
    v = [slice((j // kc) * stride + qc * Dk + kc * Dk + (j % kc) * Dv,
               (j // kc) * stride + qc * Dk + kc * Dk + (j % kc + 1) * Dv) for j in kv]
    return merge(q) + merge(k) + merge(v)


def _part(n_total: int, rank: int, size: int) -> slice:
    k = n_total // size
    return slice(rank * k, (rank + 1) * k)


def _concat(parts: list[tuple[torch.Tensor, torch.Tensor | None]]):
    """Row-concatenate (w, scale) pieces of one fused weight. Pieces that disagree on being
    quantized are all dequantized, so the result is consistent either way."""
    ws, ss = [p[0] for p in parts], [p[1] for p in parts]
    if all(x is not None for x in ss):
        return torch.cat(ws), torch.cat(ss)
    if all(x is None for x in ss):
        return torch.cat(ws), None
    return torch.cat([dequant(w, s, torch.float32) if s is not None else w.float() for w, s in parts]), None


# How routed experts' FP8 blocks are fitted onto trn's e4m3 max 240 (models/quant.fit_e4m3_max),
# KILN_MOE_E4M3_FIT: "row" (default) halves per (row, block), exact for every normal value; "group"
# halves per (row group, block) for a 128 x 128 block-scaled checkpoint's tensor-parallel shard
# (each rank's 64 gate rows, its 64 up rows, 128 down output rows), so the scales stay block-
# constant and kernels/moe_prefill.py takes its dequantize-first gate_up path and per-chunk down
# drains (GLM-5.3-Flash's real experts: 5.95 against 4.96 ms per C=1024 call, docs/neuron-notes.md);
# it also halves (and so may round codes below 2^-5 of) the rows of a group that did not exceed 240.
# Other weights: per row.
MOE_E4M3_FIT = os.environ.get("KILN_MOE_E4M3_FIT", "row")
if MOE_E4M3_FIT not in ("row", "group"):
    raise ValueError(f"KILN_MOE_E4M3_FIT must be row or group, not {MOE_E4M3_FIT!r}")


def _assign(param, scale_param, w: torch.Tensor, s: torch.Tensor | None, dtype, bk: int | None,
            fp8_max: float, row_group: int | None = None) -> None:
    """Store checkpoint data (w, s) into a model weight: FP8 + scale when the model keeps it
    quantized, otherwise dequantized to the model dtype. row_group: fit_e4m3_max's."""
    if scale_param is None:
        param.data.copy_(dequant(w, s, torch.float32).to(dtype) if s is not None else w.to(dtype))
        return
    if s is None:  # checkpoint left this weight unquantized: quantize it ourselves
        w, s = quantize_fp8_rows(w.float(), bk)
    w, s = fit_e4m3_max(w.to(FP8), s, fp8_max, row_group)
    param.data.copy_(w)
    scale_param.data.copy_(s)


def _load_decoder(model: DecoderForCausalLM, path: str, dtype: torch.dtype, fp8_max: float = 240.0) -> None:
    cfg = model.cfg
    ck = _Checkpoint(path)
    if cfg.architecture == "InklingForConditionalGeneration":
        return _load_inkling(model, ck, dtype, fp8_max)
    materialize = getattr(model, "materialize", lambda mod: (lambda: None))

    def put(param, name, rows=None, cols=None):
        param.data.copy_(ck.get(name, rows, cols).to(dtype))

    def put_norm(param, name):
        """An RMSNorm weight; zero-centred checkpoints (cfg.norm_offset, Qwen3.5) apply 1 + w."""
        w = ck.get(name).float()
        param.data.copy_((w + 1.0 if cfg.norm_offset else w).to(dtype))

    def put_vocab(param, name):
        """This rank's vocabulary rows; rows past the vocabulary (the last shard's padding)
        are zero, so their logits are 0 and are cut off after the gather."""
        start, rows, V = model.vocab_start, model.vocab_rows, cfg.vocab_size
        stop = min(start + rows, V)
        param.data.zero_()
        param.data[: stop - start].copy_(ck.get(name, slice(start, stop)).to(dtype))

    top_done = materialize(model)
    put_vocab(model.embed, "model.embed_tokens.weight")
    # Qwen3.8-Flash-Next has no layer or final norms: its hyper-connections normalise (models/hybrid.py).
    norms = cfg.hybrid is None or cfg.hybrid.block_norms
    if norms:
        put_norm(model.norm, "model.norm.weight")
    if cfg.hybrid is not None:
        hybrid.load_model(model, ck, dtype)
    if model.lm_head is not None:
        if "lm_head.weight" not in ck:
            raise KeyError("checkpoint has untied embeddings but no lm_head.weight")
        put_vocab(model.lm_head, "lm_head.weight")
    if getattr(model, "mtp", None) is not None:  # the MTP head's own parameters live on the model
        from .mtp import names as mtp_names  # not `names`: that would shadow this module's names() below

        m, mtp_post, mtp_final = mtp_names(cfg)
        if cfg.mtp_names == "qwen3_5":  # zero-centred norms, as the rest of Qwen3.5
            for p_, n_ in ((model.mtp_enorm, "pre_fc_norm_embedding"), (model.mtp_hnorm, "pre_fc_norm_hidden"),
                           (model.mtp_norm, "norm")):
                put_norm(p_, f"mtp.{n_}.weight")
            put(model.mtp_eh, "mtp.fc.weight")  # [H, 2H] over cat(embedding, hidden), Kiln's order
        else:
            put(model.mtp_enorm, m + "enorm.weight")
            put(model.mtp_hnorm, m + "hnorm.weight")
            put(model.mtp_norm, m + mtp_final)
            w, sc = ck.linear(m + "eh_proj")  # bf16 in every checkpoint read so far; dequantized if not
            model.mtp_eh.data.copy_((dequant(w, sc, torch.float32) if sc is not None else w).to(dtype))
    top_done()
    r, n = model.tp_rank, model.tp_size  # MLP / experts shard
    ar, an = model.attn_rank, model.attn_tp  # token-mixer shard (attention TP)
    blk = cfg.quant_block
    nm = names(cfg)
    mtp = getattr(model, "mtp", None)
    jobs = [(i, layer, f"model.layers.{i}.", "post_attention_layernorm") for i, layer in enumerate(model.layers)]
    if mtp is not None:  # vLLM mimo_v2_mtp.py: MiMo's MTP block names its post-attention norm pre_mlp_layernorm
        jobs.append(("mtp", mtp, m, mtp_post))
    for i, layer, pre, post_name in jobs:
        layer_done = materialize(layer)
        sp = layer.spec
        try:
            if norms:
                put_norm(layer.in_norm, pre + "input_layernorm.weight")
                put_norm(layer.post_norm, pre + post_name + ".weight")
            if isinstance(sp, LinearSpec):
                linear_attn.load_mixer(layer, ck, pre, cfg, ar, an, dtype, blk, fp8_max)
            elif sp.mla is not None:
                mla.load_layer(layer, ck, pre, cfg, ar, dtype,
                               lambda p_, s_, w_, sc_: _assign(p_, s_, w_, sc_, dtype, blk, fp8_max), _concat, put_norm)
            else:
                Dk, Dv = sp.head_dim, sp.v_head_dim
                qs = _part(sp.num_heads * Dk, ar, an)
                ks = slice(layer.kv_offset * Dk, (layer.kv_offset + layer.nkv) * Dk)
                vs = slice(layer.kv_offset * Dv, (layer.kv_offset + layer.nkv) * Dv)
                if layer.q_norm is not None:
                    put_norm(layer.q_norm, pre + "self_attn.q_norm.weight")
                    put_norm(layer.k_norm, pre + "self_attn.k_norm.weight")
                a = pre + "self_attn."
                if cfg.fused_qkv_weights and os.environ.get("KILN_QKV_LAYOUT", "grouped") == "grouped":
                    parts = [ck.linear(a + "qkv_proj", sl, block=blk)
                             for sl in _grouped_qkv_rows(sp, qs, layer, cfg.num_kv_heads)]
                elif cfg.fused_qkv_weights:  # contiguous [Q | K | V] (debugging only)
                    qn, kn = sp.num_heads * Dk, sp.num_kv_heads * Dk
                    shift = lambda sl, by: slice(sl.start + by, sl.stop + by)  # noqa: E731
                    parts = [ck.linear(a + "qkv_proj", sl, block=blk) for sl in (qs, shift(ks, qn), shift(vs, qn + kn))]
                elif cfg.attn_output_gate:
                    # q_proj holds [q | gate] per head (Qwen3_5Attention: view(..., -1, 2 * head_dim), chunk).
                    qrows = [slice(2 * h * Dk, (2 * h + 1) * Dk) for h in range(qs.start // Dk, qs.stop // Dk)]
                    w, sc = _concat([ck.linear(a + "q_proj", slice(sl.stop, sl.stop + Dk), block=blk) for sl in qrows])
                    _assign(layer.o_gate, layer.o_gate_scale, w, sc, dtype, blk, fp8_max)
                    parts = [ck.linear(a + "q_proj", sl, block=blk) for sl in qrows]
                    parts += [ck.linear(a + "k_proj", ks, block=blk), ck.linear(a + "v_proj", vs, block=blk)]
                else:
                    parts = [ck.linear(a + "q_proj", qs, block=blk), ck.linear(a + "k_proj", ks, block=blk),
                             ck.linear(a + "v_proj", vs, block=blk)]
                w, sc = _concat(parts)
                _assign(layer.qkv, layer.qkv_scale, w, sc, dtype, blk, fp8_max)
                if layer.qkv_bias is not None:
                    layer.qkv_bias.data.copy_(torch.cat([ck.get(a + "q_proj.bias", qs), ck.get(a + "k_proj.bias", ks),
                                                         ck.get(a + "v_proj.bias", vs)]).to(dtype))
                if layer.sink is not None:
                    put(layer.sink, pre + nm["sink"], rows=_part(sp.num_heads, ar, an))
                w, sc = ck.linear(a + "o_proj", cols=_part(sp.num_heads * Dv, ar, an), block=blk)
                _assign(layer.o, layer.o_scale, w, sc, dtype, blk, fp8_max)
                if layer.o_bias is not None:  # added once per attention group: its rank 0 holds it (gpt-oss)
                    put(layer.o_bias, a + "o_proj.bias") if ar == 0 else layer.o_bias.data.zero_()
            if layer.moe:
                _load_experts(layer, ck, pre, cfg, r, n, dtype, fp8_max)
                if layer.shared_gate_up is not None:  # mlp.shared_experts.* (DeepSeek-V3 / GLM-5), Hy3's shared_mlp.*
                    sh, base = _part(cfg.moe_intermediate_size * cfg.n_shared_experts, r, n), pre + nm["shared"]
                    w, sc = _concat([ck.linear(base + "gate_proj", sh, block=blk),
                                     ck.linear(base + "up_proj", sh, block=blk)])
                    _assign(layer.shared_gate_up, layer.shared_gate_up_scale, w, sc, dtype, blk, fp8_max)
                    w, sc = ck.linear(base + "down_proj", cols=sh, block=blk)
                    _assign(layer.shared_down, layer.shared_down_scale, w, sc, dtype, blk, fp8_max)
                if layer.pack_moe:
                    layer.pack_experts()
            else:
                im = _part(cfg.intermediate_size, r, n)
                w, sc = _concat([ck.linear(pre + "mlp.gate_proj", im, block=blk),
                                 ck.linear(pre + "mlp.up_proj", im, block=blk)])
                _assign(layer.gate_up, layer.gate_up_scale, w, sc, dtype, blk, fp8_max)
                w, sc = ck.linear(pre + "mlp.down_proj", cols=im, block=blk)
                _assign(layer.down, layer.down_scale, w, sc, dtype, blk, fp8_max)
            if cfg.hybrid is not None and not layer.plain:
                hybrid.load_layer(layer, ck, pre, cfg, r, n, dtype)
        except KeyError as e:
            raise KeyError(f"layer {i}: checkpoint is missing {e}") from None
        layer_done()


def _load_experts(layer, ck: _Checkpoint, pre: str, cfg: ModelConfig, r: int, n: int, dtype,
                  fp8_max: float = 240.0) -> None:
    """Hub checkpoints name each expert (mlp.experts.{e}.gate_proj.weight ...); transformers 5
    saves them fused as 3-D tensors (mlp.experts.gate_up_proj [E, 2I, H], gate rows first,
    and mlp.experts.down_proj [E, H, I]). Accept both; shard I across tensor-parallel ranks.
    Per-expert weights may be FP8 or MXFP4 (see models/quant.py)."""
    E, I = cfg.num_experts, cfg.moe_intermediate_size
    sl = _part(I, r, n)
    nm = names(cfg)
    layer.router.data.copy_(ck.get(pre + nm["router"]).to(dtype))
    if layer.router_bias is not None:
        layer.router_bias.data.copy_(ck.get(pre + nm["router_bias"]).float())
    if getattr(layer, "router_logit_bias", None) is not None:
        layer.router_logit_bias.data.copy_(ck.get(pre + nm["router_logit_bias"]).to(dtype))
    m = pre + "mlp.experts."
    bk = cfg.quant_expert_block
    if getattr(layer, "moe_ep", False):
        return _load_ep_experts(layer, ck, m, cfg, dtype, bk, fp8_max)
    if m + "gate_up_proj_blocks" in ck:  # gpt-oss: fused MXFP4 experts with biases
        return _load_fused_mxfp4_experts(layer, ck, m, cfg, r, dtype, fp8_max)
    if layer.w_gu.shape[-2] * n != 2 * I:
        raise ValueError(f"{pre}: zero-padded expert shards ({layer.w_gu.shape[-2] // 2} of {I} per rank) are only "
                         "implemented for fused MXFP4 experts")
    if m + "gate_up_proj" in ck:
        gu = ck.get(m + "gate_up_proj")
        _assign_experts(layer, torch.cat([gu[:, sl], gu[:, I:][:, sl]], dim=1), None,
                        ck.get(m + "down_proj")[:, :, sl], None, dtype, bk, fp8_max)
        return
    if layer.w_gu.dtype == torch.uint8:  # packed MXFP4, copied as stored
        for e in range(E):
            g, gs = ck.mxfp4(f"{m}{e}.gate_proj", sl)
            u, us = ck.mxfp4(f"{m}{e}.up_proj", sl)
            d, ds = ck.mxfp4(f"{m}{e}.down_proj", cols=sl)
            layer.w_gu.data[e].copy_(torch.cat([g, u]))
            layer.w_gu_scale.data[e].copy_(torch.cat([gs, us]))
            layer.w_down.data[e].copy_(d)
            layer.w_down_scale.data[e].copy_(ds)
        return
    for e in range(E):
        g, gs = ck.linear(f"{m}{e}.gate_proj", sl, block=bk)
        u, us = ck.linear(f"{m}{e}.up_proj", sl, block=bk)
        d, ds = ck.linear(f"{m}{e}.down_proj", cols=sl, block=bk)
        gus = torch.cat([gs, us]) if gs is not None else None
        group = MOE_E4M3_FIT == "group"  # gate rows and up rows each one group; down 128 output rows
        _assign(_Row(layer.w_gu, e), _Row(layer.w_gu_scale, e) if layer.w_gu_scale is not None else None,
                torch.cat([g, u]), gus, dtype, bk, fp8_max, g.shape[0] if group else None)
        _assign(*_down(layer, e), d, ds, dtype, bk, fp8_max, 128 if group else None)


# How EP experts' FP8 blocks are fitted onto e4m3 max 240, KILN_MOE_EP_FIT: "block" (default) halves a whole 128 x
# 128 block of a block-scaled checkpoint when any of its codes exceeds 240 (fit_e4m3_max row_group 128), so its
# scale stays one value and kernels/moe_ep.py dequantizes each tile by one instruction on two engines (pack()'s
# tsg / tsd); "row" per (row, block) as under TP. Block halving is exact except for codes below 2^-5, which become
# subnormal (an error of at most 2^-10 of the doubled scale); measured on GLM-5.3-Flash: docs/neuron-notes.md
# "Expert parallelism".
EP_FIT = os.environ.get("KILN_MOE_EP_FIT", "block")
if EP_FIT not in ("row", "block"):
    raise ValueError(f"KILN_MOE_EP_FIT must be row or block, not {EP_FIT!r}")


def _load_ep_experts(layer, ck: _Checkpoint, m: str, cfg: ModelConfig, dtype, bk, fp8_max: float) -> None:
    """Expert parallelism (models/decoder.py moe_ep_enabled): this rank's experts layer.ep_first ..
    + layer.ep_count, WHOLE (every intermediate row), from per-expert or fused 3-D checkpoint tensors. FP8 blocks
    are fitted onto e4m3 max per 128 x 128 block (EP_FIT "block") or per (row, 128-column block) as under TP
    ("row"); a TP rank decides per (row, its 64-column half of a down block), so the layouts can halve codes
    differently, which changes only codes below 2^-5 (models/quant.fit_e4m3_max)."""
    first, El = layer.ep_first, layer.ep_count
    # The logical expert of each local slot: the rank's primaries, then its redundant slots (models/eplb.py).
    experts = list(getattr(layer, "ep_experts", range(first, first + El)))
    rg = 128 if EP_FIT == "block" else None
    layer.ep_tiles = False  # block-constant scales (kernels/moe_ep.pack tiles), from the format alone
    if m + "gate_up_proj_blocks" in ck or layer.w_gu.dtype == torch.uint8:
        raise NotImplementedError("KILN_MOE_EP=1 with MXFP4 experts (gpt-oss, packed MXFP4) is not implemented")
    if m + "gate_up_proj" in ck:  # transformers 5's fused [E, 2I, H] (gate rows first) and [E, H, I]
        if experts == list(range(first, first + len(experts))):
            rows = slice(first, first + len(experts))
            gu, dn = ck.get(m + "gate_up_proj", rows), ck.get(m + "down_proj", rows)
        else:
            gu = torch.cat([ck.get(m + "gate_up_proj", slice(e, e + 1)) for e in experts])
            dn = torch.cat([ck.get(m + "down_proj", slice(e, e + 1)) for e in experts])
        _assign_experts(layer, gu, None, dn, None, dtype, bk, fp8_max, rg)
        return
    tiles = rg == 128 and bk == 128
    for le, e in enumerate(experts):
        g, gs = ck.linear(f"{m}{e}.gate_proj", block=bk)
        u, us = ck.linear(f"{m}{e}.up_proj", block=bk)
        d, ds = ck.linear(f"{m}{e}.down_proj", block=bk)
        gus = torch.cat([gs, us]) if gs is not None else None
        _assign(_Row(layer.w_gu, le), _Row(layer.w_gu_scale, le) if layer.w_gu_scale is not None else None,
                torch.cat([g, u]), gus, dtype, bk, fp8_max, rg)
        _assign(*_down(layer, le), d, ds, dtype, bk, fp8_max, rg)
        # A 128 x 128 block-scaled FP8 checkpoint (weight_scale_inv) fitted per block: every kernel tile has one
        # scale (per-tensor / per-channel FP8 and self-quantized weights keep per-row scales).
        tiles = tiles and all(f"{m}{e}.{n}.weight_scale_inv" in ck for n in ("gate_proj", "up_proj", "down_proj"))
    layer.ep_tiles = bool(tiles)


def prepare_ep_slots(path: str, model, layer, index, slots: dict[int, int], fp8_max: float = 240.0,
                     ck: "_Checkpoint | None" = None) -> dict[str, torch.Tensor]:
    """Host tensors for local slots {slot: logical expert} of a loaded expert-parallel layer (an EPLB rebalance,
    models/eplb.py): the experts read from the checkpoint and fitted exactly as at load (_load_ep_experts), packed
    into the layer's layout (kernels/moe_ep.pack, when the layer is packed). {parameter name: [len(slots), ...]},
    in the order of slots; write_ep_slots copies them in. Host work only, so it can run beside serving.
    index: the layer's model.layers index (the MTP layer has no redundant slots)."""
    from types import SimpleNamespace

    if not slots:
        return {}
    cfg = model.cfg
    ck = ck or _Checkpoint(path)
    n = len(slots)
    tmp = SimpleNamespace(ep_first=0, ep_count=n, ep_experts=list(slots.values()), down_t=layer.down_t, moe_ep=True)
    for name, meta in layer.ep_meta.items():
        setattr(tmp, name, None if meta is None else torch.nn.Parameter(torch.empty(n, *meta[0], dtype=meta[1]),
                                                                       requires_grad=False))
    _load_ep_experts(tmp, ck, f"model.layers.{index}.mlp.experts.", cfg, model.dtype, cfg.quant_expert_block, fp8_max)
    if layer.moe_blob:
        from ..kernels import moe_ep

        blob = moe_ep.pack(tmp.w_gu.data, tmp.w_gu_scale.data, tmp.w_down.data, tmp.w_down_scale.data, tmp.ep_tiles)
        return {"ep_" + k: v for k, v in blob.items() if hasattr(layer, "ep_" + k)}
    return {k: getattr(tmp, k).data for k in layer.ep_meta if layer.ep_meta[k] is not None}


def write_ep_slots(layer, slots: list[int], src: dict[str, torch.Tensor]) -> None:
    """Copy prepare_ep_slots' tensors into the layer's slots, in place on its device (one eager copy per slot and
    tensor: the other slots stay bit-identical, tools/probe_eplb_device.py)."""
    for name, t in src.items():
        dst = getattr(layer, name)
        for i, sl in enumerate(slots):
            dst.data[sl].copy_(t[i].to(dst.device))


def load_ep_slots(path: str, model, layer, index, slots: dict[int, int], fp8_max: float = 240.0,
                  ck: "_Checkpoint | None" = None) -> None:
    """prepare_ep_slots then write_ep_slots."""
    write_ep_slots(layer, list(slots), prepare_ep_slots(path, model, layer, index, slots, fp8_max, ck))


class _Row:
    """One expert's slice of a stacked [E, ...] parameter, assignable like a parameter;
    transposed: a view in checkpoint ([out, in]) order of a weight stored [in, out]."""

    def __init__(self, param, e: int, transposed: bool = False):
        self.data = param.data[e].T if transposed else param.data[e]


def _down(layer, e: int):
    t = layer.down_t
    return _Row(layer.w_down, e, t), (_Row(layer.w_down_scale, e, t) if layer.w_down_scale is not None else None)


def _assign_experts(layer, gu, gus, down, downs, dtype, bk, fp8_max, row_group: int | None = None) -> None:
    for e in range(gu.shape[0]):
        _assign(_Row(layer.w_gu, e), _Row(layer.w_gu_scale, e) if layer.w_gu_scale is not None else None,
                gu[e], gus[e] if gus is not None else None, dtype, bk, fp8_max, row_group)
        _assign(*_down(layer, e), down[e], downs[e] if downs is not None else None, dtype, bk, fp8_max, row_group)


def _load_fused_mxfp4_experts(layer, ck: _Checkpoint, m: str, cfg: ModelConfig, r: int, dtype,
                              fp8_max: float) -> None:
    """gpt-oss experts, from https://huggingface.co/openai/gpt-oss-120b (shard headers):
    gate_up_proj_blocks U8 [E, 2I, H/32, 16] with gate_up_proj_scales U8 [E, 2I, H/32] and
    gate_up_proj_bias [E, 2I], down_proj_blocks [E, H, I/32, 16], down_proj_scales [E, H, I/32],
    down_proj_bias [E, H]. Rows are outputs, the 16 bytes of a block are 32 E2M1 codes low nibble
    first, and gate_up's rows interleave gate and up (row 2i gate, 2i + 1 up): transformers 5.15
    integrations/mxfp4.py _convert_moe_packed_tensors dequantizes the blocks to [E, rows, cols]
    and GptOssExperts._apply_gate takes gate = [..., ::2], up = [..., 1::2].

    This rank takes intermediate units [r * Im, (r + 1) * Im) and zero-pads past I (see
    decoder.moe_inter_per_rank); gate and up are de-interleaved into Kiln's [gate rows; up rows]."""
    E, I = cfg.num_experts, cfg.moe_intermediate_size
    Im = layer.w_gu.shape[-2] // 2
    u0, u1 = min(r * Im, I), min((r + 1) * Im, I)
    real = u1 - u0
    gu_sl = ck._where[m + "gate_up_proj_blocks"].get_slice(m + "gate_up_proj_blocks")
    gu_blocks = gu_sl[:, 2 * u0 : 2 * u1]  # [E, 2 * real, H/32, 16]: this rank's units, still interleaved
    gu_exps = ck._where[m + "gate_up_proj_scales"].get_slice(m + "gate_up_proj_scales")[:, 2 * u0 : 2 * u1]
    gu_bias = ck._where[m + "gate_up_proj_bias"].get_slice(m + "gate_up_proj_bias")[:, 2 * u0 : 2 * u1]
    down_blocks, down_exps = ck.get(m + "down_proj_blocks"), ck.get(m + "down_proj_scales")
    if bool((gu_exps == 255).any()) or bool((down_exps == 255).any()):
        raise ValueError(f"{m}: NaN (E8M0 255) block scale")  # OCP Microscaling Formats v1.0, E8M0
    if layer.w_gu_bias is not None:
        layer.w_gu_bias.data.zero_()
        layer.w_gu_bias.data[:, :real].copy_(gu_bias[:, 0::2].to(dtype))
        layer.w_gu_bias.data[:, Im : Im + real].copy_(gu_bias[:, 1::2].to(dtype))
        if r == 0:
            layer.w_down_bias.data.copy_(ck.get(m + "down_proj_bias").to(dtype))
        else:
            layer.w_down_bias.data.zero_()
    H = cfg.hidden_size
    pad = Im - real
    aligned = u0 % 32 == 0 and real % 32 == 0  # this rank's down columns are whole MXFP4 blocks
    if not aligned and layer.w_down_scale is not None:
        raise ValueError(f"{m}: quantized experts need whole 32-column blocks per rank (moe_inter_per_rank)")
    for e in range(E):
        gp, ge = gu_blocks[e].reshape(2 * real, H // 2), gu_exps[e].reshape(2 * real, H // 32)
        # Gate rows, then up rows, each zero-padded to Im (code 0 is +0.0, exponent 127 is 2^0).
        packed = torch.cat([_pad(gp[0::2], pad, 0, 0), _pad(gp[1::2], pad, 0, 0)])
        exps = torch.cat([_pad(ge[0::2], pad, 0, 127), _pad(ge[1::2], pad, 0, 127)])
        dp, de = down_blocks[e].reshape(H, I // 2), down_exps[e]
        if aligned:
            dp, de = _pad(dp[:, u0 // 2 : u1 // 2], pad // 2, 1, 0), _pad(de[:, u0 // 32 : u1 // 32], pad // 32, 1, 127)
        if layer.w_gu.dtype == torch.uint8:  # packed MXFP4 on the device, copied as stored
            layer.w_gu.data[e].copy_(packed)
            layer.w_gu_scale.data[e].copy_(torch.exp2(exps.float() - 127.0).to(torch.bfloat16))
            layer.w_down.data[e].copy_(dp)
            layer.w_down_scale.data[e].copy_(torch.exp2(de.float() - 127.0).to(torch.bfloat16))
            continue
        g, gs = mxfp4_unpack(packed, exps)  # fp8 [2Im, H], fp32 [2Im, H/32]
        d, ds = mxfp4_unpack(dp, de)
        if not aligned:  # dequantized at load (bf16): unpack whole rows, then cut any columns
            d, ds = dequant(d, ds, torch.float32)[:, u0:u1], None
        _assign(_Row(layer.w_gu, e), _Row(layer.w_gu_scale, e) if layer.w_gu_scale is not None else None,
                g, gs, dtype, 32, fp8_max)
        _assign(*_down(layer, e), d, ds, dtype, 32, fp8_max)


def _pad(x: torch.Tensor, n: int, dim: int, value) -> torch.Tensor:
    """x with n entries of `value` appended along dim (0 or 1)."""
    if not n:
        return x
    shape = list(x.shape)
    shape[dim] = n
    return torch.cat([x, torch.full(shape, value, dtype=x.dtype)], dim=dim)


def _deinterleave(w: torch.Tensor, units: slice) -> torch.Tensor:
    """Rows 2u (gate) and 2u + 1 (up) of a gate / up row interleave, for the units of `units`,
    as [gate rows; up rows] (transformers 5.15 core_model_loading.Interleave, used for Inkling in
    conversion_mapping.py "inkling_mm_model")."""
    part = w[2 * units.start : 2 * units.stop]
    return torch.cat([part[0::2], part[1::2]])


def _load_inkling(model: DecoderForCausalLM, ck: _Checkpoint, dtype, fp8_max: float) -> None:
    """thinkingmachines/Inkling text weights, by their hub names (model.llm.*, the shard headers
    of https://huggingface.co/thinkingmachines/Inkling-Small; transformers 5.15 renames them in
    conversion_mapping.py "inkling_mm_model"): attn.wq_du / wk_dv / wv_dv / wr_du / wo_ud for
    q / k / v / r / o, attn_norm and mlp_norm, k_sconv / v_sconv / attn_sconv / mlp_sconv [C, 1, K],
    attn.rel_logits_proj.proj [d_rel, extent]; dense layers mlp.w13_dn (gate / up rows
    interleaved), w2_md and global_scale; MoE layers mlp.gate.weight [E + n_shared, H], its bias
    (selection) and global_scale, experts.w13_weight [E, 2I, H] (interleaved) and w2_weight
    [E, H, I], shared_experts.shared_w13_weight [n, 2I, H] (interleaved) and shared_w2_weight [n, H, I].
    Vision, audio and MTP weights are not read."""
    cfg = model.cfg
    materialize = getattr(model, "materialize", lambda mod: (lambda: None))
    r, n = model.tp_rank, model.tp_size  # MLP / experts shard
    ar, an = model.attn_rank, model.attn_tp  # attention shard
    blk, bk = cfg.quant_block, cfg.quant_expert_block

    def put(param, name, rows=None, cols=None):
        param.data.copy_(ck.get(name, rows, cols).to(param.dtype))

    def put_vocab(param, name):
        start, rows, V = model.vocab_start, model.vocab_rows, cfg.vocab_size
        stop = min(start + rows, V)
        param.data.zero_()
        param.data[: stop - start].copy_(ck.get(name, slice(start, stop)).to(dtype))

    top_done = materialize(model)
    put_vocab(model.embed, "model.llm.embed.weight")
    put_vocab(model.lm_head, "model.llm.unembed.weight")
    put(model.norm, "model.llm.norm.weight")
    if model.embed_norm is not None:
        put(model.embed_norm, "model.llm.embed_norm.weight")
    top_done()
    for i, layer in enumerate(model.layers):
        layer_done = materialize(layer)
        pre, sp = f"model.llm.layers.{i}.", layer.spec
        a = pre + "attn."
        D, Dv = sp.head_dim, sp.v_head_dim
        qs = _part(sp.num_heads * D, ar, an)
        ks = slice(layer.kv_offset * D, (layer.kv_offset + layer.nkv) * D)
        vs = slice(layer.kv_offset * Dv, (layer.kv_offset + layer.nkv) * Dv)
        try:
            put(layer.in_norm, pre + "attn_norm.weight")
            put(layer.post_norm, pre + "mlp_norm.weight")
            put(layer.q_norm, a + "q_norm.weight")
            put(layer.k_norm, a + "k_norm.weight")
            w, sc = _concat([ck.linear(a + "wq_du", qs, block=blk), ck.linear(a + "wk_dv", ks, block=blk),
                             ck.linear(a + "wv_dv", vs, block=blk)])
            _assign(layer.qkv, layer.qkv_scale, w, sc, dtype, blk, fp8_max)
            w, sc = ck.linear(a + "wo_ud", cols=_part(sp.num_heads * Dv, ar, an), block=blk)
            _assign(layer.o, layer.o_scale, w, sc, dtype, blk, fp8_max)
            if layer.rel_proj is not None:
                w, sc = ck.linear(a + "wr_du", _part(sp.num_heads * cfg.d_rel, ar, an), block=blk)
                _assign(layer.rel_q, layer.rel_q_scale, w, sc, dtype, blk, fp8_max)
                put(layer.rel_proj, a + "rel_logits_proj.proj")
            if layer.k_sconv is not None:
                layer.k_sconv.data.copy_(ck.get(a + "k_sconv.weight")[ks, 0].float())
                layer.v_sconv.data.copy_(ck.get(a + "v_sconv.weight")[vs, 0].float())
                layer.a_sconv.data.copy_(ck.get(pre + "attn_sconv.weight")[:, 0].float())
                layer.m_sconv.data.copy_(ck.get(pre + "mlp_sconv.weight")[:, 0].float())
            m = pre + "mlp."
            if not layer.moe:
                units = _part(cfg.intermediate_size, r, n)
                w13 = ck.get(m + "w13_dn.weight", slice(2 * units.start, 2 * units.stop))
                _assign(layer.gate_up, layer.gate_up_scale, _deinterleave(w13, slice(0, units.stop - units.start)),
                        None, dtype, blk, fp8_max)
                w, sc = ck.linear(m + "w2_md", cols=units, block=blk)
                _assign(layer.down, layer.down_scale, w, sc, dtype, blk, fp8_max)
                if layer.mlp_scale is not None:
                    put(layer.mlp_scale, m + "global_scale")
                layer_done()
                continue
            put(layer.router, m + "gate.weight")
            if layer.router_bias is not None:
                layer.router_bias.data.copy_(ck.get(m + "gate.bias").float())
            if layer.router_scale is not None:
                layer.router_scale.data.copy_(ck.get(m + "gate.global_scale").float())
            I, E = cfg.moe_intermediate_size, cfg.num_experts
            units = _part(I, r, n)
            gu = ck._where[m + "experts.w13_weight"].get_slice(m + "experts.w13_weight")[:, 2 * units.start:
                                                                                      2 * units.stop]
            w2 = ck.get(m + "experts.w2_weight")
            for e in range(E):
                _assign(_Row(layer.w_gu, e), _Row(layer.w_gu_scale, e) if layer.w_gu_scale is not None else None,
                        _deinterleave(gu[e], slice(0, units.stop - units.start)), None, dtype, bk, fp8_max)
                _assign(*_down(layer, e), w2[e][:, units], None, dtype, bk, fp8_max)
            # Shared experts as one MLP over the concatenation of their intermediate units; this
            # rank's units [r * Is, (r + 1) * Is) may span experts, shared_units says which.
            ns, Is = cfg.router_shared_rows, layer.shared_gate_up.shape[0] // 2
            s13 = ck.get(m + "shared_experts.shared_w13_weight")  # [n, 2I, H]
            s2 = ck.get(m + "shared_experts.shared_w2_weight")  # [n, H, I]
            glob = torch.arange(r * Is, (r + 1) * Is)
            ex, u = glob // I, glob % I
            gate = torch.stack([s13[e, 2 * j] for e, j in zip(ex.tolist(), u.tolist())])
            up = torch.stack([s13[e, 2 * j + 1] for e, j in zip(ex.tolist(), u.tolist())])
            _assign(layer.shared_gate_up, layer.shared_gate_up_scale, torch.cat([gate, up]), None, dtype, blk, fp8_max)
            down = torch.stack([s2[e, :, j] for e, j in zip(ex.tolist(), u.tolist())], dim=1)  # [H, Is]
            _assign(layer.shared_down, layer.shared_down_scale, down, None, dtype, blk, fp8_max)
            layer.shared_units.data.copy_(torch.nn.functional.one_hot(ex, ns).to(layer.shared_units.dtype))
        except KeyError as err:
            raise KeyError(f"layer {i}: checkpoint is missing {err}") from None
        layer_done()
