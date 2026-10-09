# SPDX-License-Identifier: Apache-2.0
# Portions Copyright vLLM Neuron contributors (vllm-project/vllm-neuron), Apache License 2.0.
# Modified by Kiln: adapted from vllm-neuron release-0.24.0.1.1.0 (commit f8abae6)
# vllm_neuron/functional/mlp.py (the nkilib MLP call and its _can_use_kernel shape gate); see THIRD_PARTY_NOTICES.md.
"""nki-library (nkilib) kernels for the DENSE decoder's linear layers on NeuronCore-v3+ (trn2 / trn3), opt-in.

vllm-neuron runs a dense model's prefill MLP through nkilib's fused MLP kernel (`nkilib.core.mlp.mlp`, from
vllm_neuron/functional/mlp.py). With KILN_DENSE_MLP_KERNEL=nkilib on a platform whose NKI generation is 3 or
newer, Kiln does the same.

- Weights: the kernel takes gate / up [H, I] and down [I, H]. Kiln's bf16 dense layers keep transposed copies
  beside the fused gate_up [2I, H] and down [H, I] (DecoderLayer.pack_dense_mlp, at load). That costs the MLP's
  weight bytes twice in HBM: the price of leaving decode graphs and their keys unchanged.
- Calls: models/decoder.py _swiglu_mlp routes only prefill-sized calls (more than 128 rows, inside the kernel's
  tiling constraints) to the kernel.

Status: opt-in. It becomes a candidate default only after a one-core A/B (tools/probe_nkilib_dense.py) and the
in-graph TTFT plus a quality gate on trn2.
"""

from __future__ import annotations

import math
import os

import torch

MLP_KERNEL = os.environ.get("KILN_DENSE_MLP_KERNEL", "xla")
if MLP_KERNEL not in ("xla", "nkilib"):
    raise ValueError(f"KILN_DENSE_MLP_KERNEL must be xla or nkilib, not {MLP_KERNEL!r}")

# nkilib's tiling constraints, as vllm_neuron/functional/mlp.py states them ("from nkilib source code").
_NUM_HW_PSUM_BANKS = 8
_SRC_PROJ_INT_DIM_TILE_SIZE = 512
TKG_BS_SEQLEN_THRESHOLD = 128  # nkilib.core.mlp.mlp_parameters TKG_BS_SEQLEN_THRESHOLD: at or below, the decode form
_MIN_H_FOR_I_SHARDING = 7168
_MIN_I_FOR_I_SHARDING = 1024
_MAX_T_FOR_I_SHARDING = 256


def enabled(cfg=None) -> bool:
    """KILN_DENSE_MLP_KERNEL=nkilib on a platform with NKI generation >= 3 (kiln/platform.py NKI_GEN; nkilib's
    kernels need NeuronCore-v3, the reason NxDI dropped trn1, docs/research/neuron-stack.md)."""
    if MLP_KERNEL != "nkilib":
        return False
    from .. import platform

    t = platform.target()
    if t is None:
        return False
    return platform.NKI_GEN.get(platform.family_of(t), 0) >= 3


def can_use_mlp(T: int, H: int, I: int, has_bias: bool = False) -> bool:
    """vllm_neuron/functional/mlp.py _can_use_kernel for bf16 weights, CTE (prefill) form only."""
    if H % 128 != 0 or T <= TKG_BS_SEQLEN_THRESHOLD:
        return False
    can_shard_on_i = (H >= _MIN_H_FOR_I_SHARDING and I >= _MIN_I_FOR_I_SHARDING and T <= _MAX_T_FOR_I_SHARDING
                      and not has_bias)
    effective_i = I // 2 if can_shard_on_i else I
    return math.ceil(effective_i / _SRC_PROJ_INT_DIM_TILE_SIZE) <= _NUM_HW_PSUM_BANKS


_WRAPPED = None


def mlp(x: torch.Tensor, gate_t: torch.Tensor, up_t: torch.Tensor, down_t: torch.Tensor) -> torch.Tensor:
    """SiLU(x @ gate_t) * (x @ up_t) @ down_t through nkilib's MLP kernel: x [T, H], gate_t / up_t [H, I],
    down_t [I, H]; returns [T, H] in x's dtype. The arguments are vllm-neuron's (vllm_neuron/functional/mlp.py),
    with no norm, no bias and no quantization."""
    global _WRAPPED
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from nkilib.core.mlp.mlp import mlp as nkilib_mlp
    from nkilib.core.utils.common_types import ActFnType, DtypeMode, NormType, QuantizationType

    from .. import platform

    if _WRAPPED is None:
        _WRAPPED = wrap_nki(nkilib_mlp)
    out = _WRAPPED[platform.nki_grid()](
        hidden_tensor=x.unsqueeze(0),
        gate_proj_weights_tensor=gate_t,
        up_proj_weights_tensor=up_t,
        down_proj_weights_tensor=down_t,
        normalization_weights_tensor=None,
        gate_proj_bias_tensor=None,
        up_proj_bias_tensor=None,
        down_proj_bias_tensor=None,
        normalization_bias_tensor=None,
        fused_add_tensor=None,
        store_fused_add_result=False,
        activation_fn=ActFnType.SiLU,
        normalization_type=NormType.NO_NORM,
        quantization_type=QuantizationType.NONE,
        gate_w_scale=None,
        up_w_scale=None,
        down_w_scale=None,
        gate_up_in_scale=None,
        down_in_scale=None,
        quant_clipping_bound=0.0,
        output_dtype=None,
        store_output_in_sbuf=False,
        eps=1e-6,
        skip_gate_proj=False,
        use_tkg_gate_up_proj_column_tiling=True,
        use_tkg_down_proj_column_tiling=True,
        use_tkg_down_proj_optimized_layout=False,
        gate_clamp_upper_limit=None,
        gate_clamp_lower_limit=None,
        up_clamp_upper_limit=None,
        up_clamp_lower_limit=None,
        force_cte_mode=False,
        dtype_mode=DtypeMode.AUTO,
    )
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.squeeze(0)
