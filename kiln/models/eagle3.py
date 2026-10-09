"""EAGLE-3 draft heads for dense targets (Llama 3.x, Qwen3), on Kiln's MTP drafting path (spec_method "mtp" with
EngineConfig.spec_draft_model), opt-in.

vllm-neuron runs EAGLE-3 for Llama 3 and gpt-oss (vllm_neuron/model/llama3/eagle3_model.py at release-0.24.0.1.1.0;
docs/model-recipes/llama-3.md). The draft math here follows vLLM v0.24.0 vllm/model_executor/models/llama_eagle3.py
(and its target side, models/llama.py + interfaces.py EagleModelMixin):

- the target hands over its residual stream after layers a for a in aux_layers (EagleModelMixin
  _maybe_add_hidden_state(aux, idx + 1, ...): the stream once idx + 1 layers ran), by default (2, L // 2, L - 3)
  (get_eagle3_default_aux_hidden_state_layers), or the draft config's eagle_aux_hidden_state_layer_ids;
- combine_hidden_states: fused = fc(cat(aux)) [T, H], once per target call;
- one Llama decoder layer (layer 0), on the token AFTER each position and the fused (or, for a later draft step,
  the previous step's prenorm) hidden h:
      embeds = input_layernorm(embed(token))
      norm_before_residual: hn = hidden_norm(h); residual = hn      else: residual = h; hn = hidden_norm(h)
      h1 = residual + o_proj(attention(q/k/v over cat(embeds, hn), 2H wide))
      prenorm = h1 + mlp(post_attention_layernorm(h1))
      draft = argmax(lm_head(norm(prenorm))) over the draft vocabulary, mapped to the target's as draft + d2t[draft];
- a later draft step feeds prenorm back as h (aux_output = hidden_prenorm, norm_output False).

Checkpoint formats read:
- the speculators one (RedHatAI/*-speculator.eagle3: architectures Eagle3Speculator, the layer under
  transformer_layer_config, weights layers.0.*, fc, norm, lm_head, d2t, t2d, an embed_tokens copy);
- the EAGLE repository's (yuhuili/EAGLE3-*: a LlamaForCausalLM-shaped config with draft_vocab_size, weights midlayer.*,
  vLLM renames midlayer. -> layers.0.).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DraftConfig:
    path: str
    layer: object  # kiln.config.ModelConfig of the draft's one decoder layer (a Llama layer)
    draft_vocab_size: int
    norm_before_residual: bool
    prefix: str  # the decoder layer's checkpoint prefix: "layers.0." or "midlayer."
    has_embed: bool  # the checkpoint carries its own embed_tokens (else the target's is used, as vLLM shares it)
    aux_layers: tuple[int, ...]
    target_hidden_size: int


def _weight_names(path: str) -> set[str]:
    from ..capture import header_index

    try:
        return set(header_index(path))
    except FileNotFoundError:  # pytorch_model.bin checkpoints (the EAGLE repository's) are not read yet
        raise NotImplementedError(f"{path}: only safetensors EAGLE-3 checkpoints are read") from None


def load_config(path: str, target) -> DraftConfig:
    """The draft at `path` (a local directory) for the target ModelConfig `target`."""
    from ..config import ModelConfig

    with open(os.path.join(path, "config.json")) as f:
        c = json.load(f)
    arch = (c.get("architectures") or [None])[0]
    if arch == "Eagle3Speculator":  # speculators format
        layer = dict(c["transformer_layer_config"])
        draft_vocab = int(c.get("draft_vocab_size") or layer["vocab_size"])
        nbr = bool(c.get("norm_before_residual", False))
        tgt_h = c.get("target_hidden_size") or target.hidden_size
        aux = c.get("eagle_aux_hidden_state_layer_ids")
    else:  # the EAGLE repository's Llama-shaped config
        layer = dict(c)
        draft_vocab = int(c.get("draft_vocab_size") or c["vocab_size"])
        nbr = bool(c.get("norm_before_residual", False))
        tgt_h = c.get("target_hidden_size") or target.hidden_size
        aux = (c.get("eagle_config") or {}).get("eagle_aux_hidden_state_layer_ids")
    layer["architectures"] = ["LlamaForCausalLM"]
    layer.setdefault("max_position_embeddings", target.max_position_embeddings)
    if layer.get("num_hidden_layers", 1) != 1:
        raise NotImplementedError("EAGLE-3 drafts with more than one decoder layer")
    mcfg = ModelConfig.from_config_dict(layer, target.eos_token_ids)
    names = _weight_names(path)
    prefix = "layers.0." if any(n.startswith("layers.0.") for n in names) else "midlayer."
    L = target.num_layers
    aux_layers = tuple(int(a) for a in aux) if aux else (2, L // 2, L - 3)
    if mcfg.hidden_size != target.hidden_size or int(tgt_h) != target.hidden_size:
        raise NotImplementedError(f"draft hidden {mcfg.hidden_size} / target_hidden_size {tgt_h} against the "
                                  f"target's {target.hidden_size}")
    if "fc.weight" not in names:
        raise NotImplementedError("a draft without fc (single hidden state EAGLE) is not EAGLE-3")
    return DraftConfig(path=path, layer=mcfg, draft_vocab_size=draft_vocab, norm_before_residual=nbr, prefix=prefix,
                       has_embed="embed_tokens.weight" in names, aux_layers=aux_layers,
                       target_hidden_size=int(tgt_h))
