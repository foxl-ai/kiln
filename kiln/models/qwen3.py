"""Compatibility shim: the Qwen3 model is the generic decoder."""

from .decoder import DecoderForCausalLM, DecoderLayer, rms_norm, rotate_half  # noqa: F401

Qwen3ForCausalLM = DecoderForCausalLM
Qwen3Layer = DecoderLayer
