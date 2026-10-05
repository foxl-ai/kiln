"""Multi-token prediction (spec_method "mtp") for the DeepSeek-V3 family: DeepSeek-V3 / R1,
DeepSeek-V3.2 (DSA) and GLM-5.x (`glm_moe_dsa`, DSA with IndexShare), next to MiMo-V2's.

One MTP layer, applied recursively for k drafts (DecoderForCausalLM.forward_mtp_k: all k passes
in ONE graph). Per position p of the target (vLLM's EAGLE / MTP convention, as for MiMo):

    x  = eh_proj(cat(enorm(embed(token after p)), hnorm(h_p)))
    y  = decoder layer (MLA, DSA indexer, MoE + shared expert)(x)
    draft = argmax(lm_head(shared_head.norm(y)))

h_p is the target's last hidden state AFTER its final norm, and a later pass takes the previous
pass's y after shared_head.norm (vLLM v0.30.0 model_executor/models/deepseek_mtp.py: "Recycle the
post-final-norm hidden into the next draft step ... Matches SGLang's deepseek_nextn";
SGLang v0.5.21 srt/models/deepseek_nextn.py DeepseekModelNextN.forward). The embedding and the
lm_head are the target's: GLM-5.3's MTP layer stores neither (zai-org/GLM-5.3
model.safetensors.index.json), and SGLang hands the target's to the draft model
(set_embed_and_head) for DeepSeek's, whose stored copies (model.layers.61.embed_tokens,
shared_head.head) Kiln does not load. The layer's MLA latent (and DSA indexer key) is one more
"KV" layer in the shared page pool (DecoderForCausalLM.kv_layers), written for every position the
target computes, which is how the later passes and the next step see it.

Two choices where the reference implementations differ, stated so they can be measured:
- Position 0: vLLM zeroes the embedding there ("masking inputs at position 0, as not needed by
  MTP", deepseek_mtp.py and models/deepseek_v32/nvidia/mtp.py); SGLang does not. Kiln does not
  (SGLang's, and Kiln's MiMo MTP). It changes only the MTP KV of position 0.
- index_share_for_mtp_iteration (GLM-5.3's and GLM-5.3-Flash's config.json set it): passes after
  the first reuse the first pass's DSA selection instead of running the indexer (vLLM v0.30.0
  llm_base_proposer.py set_skip_topk / compact_topk_indices; SGLang v0.5.21
  layers/attention/index_topk_share.py). Kiln follows the config (KILN_MTP_INDEX_SHARE=0 / 1
  overrides it). What is reused is the first pass's selection AT the position its draft came from,
  intersected with what that position could see: so a later pass attends to exactly those keys,
  not to the keys written after them (its own included), as vLLM's reused indices do. In a dense
  bucket (context <= index_topk) that is every key up to that position. The mask path only
  (KILN_DSA=gather keeps every pass's own indexer).
"""

from __future__ import annotations

import os

import torch

from .decoder import rms_norm


def names(cfg) -> tuple[str, str, str]:
    """(checkpoint prefix, post-attention norm, final norm) of the MTP layer: MiMo-V2's
    model.mtp.layers.0 with pre_mlp_layernorm / final_layernorm (vLLM v0.30.0 mimo_v2_mtp.py), or
    DeepSeek-V3's model.layers.<n> with post_attention_layernorm / shared_head.norm."""
    if cfg.mtp_prefix is None:
        return "model.mtp.layers.0.", "pre_mlp_layernorm", "final_layernorm.weight"
    return cfg.mtp_prefix + ".", "post_attention_layernorm", "shared_head.norm.weight"


def index_share(cfg) -> bool:
    """Whether the later draft passes reuse the first pass's DSA selection (module docstring)."""
    from . import mla

    spec = cfg.mtp_spec
    if spec is None or spec.mla is None or spec.mla.dsa is None or mla.DSA_MODE == "gather":
        return False
    v = os.environ.get("KILN_MTP_INDEX_SHARE", "auto")
    return cfg.mtp_index_share if v == "auto" else v == "1"


def mla_layer(model, layer, h, positions, slot_mapping, table, bias, top=None):
    """DecoderForCausalLM._layer for an MLA MTP layer, which also returns the DSA selection the
    attention used (mla.attention want_top), and takes one to use instead (top)."""
    from . import mla

    x = rms_norm(h, layer.in_norm, model.cfg.rms_norm_eps)
    o, top = mla.attention(model, layer, x, positions, slot_mapping, table, bias, top=top, want_top=True)
    return model._mlp(layer, h + model._attn_all_reduce(o)), top


def shared_rows(bias: torch.Tensor, top, rows: torch.Tensor) -> torch.Tensor:
    """The additive mask [B, 1, L] a later pass reuses: the first pass's visibility (bias, B x Q
    rows over L keys) plus its selection (top, None for a dense bucket) at the rows the drafts came
    from (rows [B], flat b * Q + last_index[b])."""
    L = bias.shape[-1]
    share = bias.reshape(-1, L).index_select(0, rows)
    if top is not None:
        share = share + top.reshape(-1, L).index_select(0, rows)
    return share.view(-1, 1, L)
