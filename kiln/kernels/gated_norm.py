"""KDA's gated RMSNorm (KimiLinearRMSNormGated) after the prefill delta rule as one NKI kernel for NeuronCore-v2
(trn1): GLM-5.3-Flash's default since 2026-10-07, KILN_KDA_FUSED_NORM=0 turns it off and =1 on for another KDA model
(models/linear_attn.py _mix_rows, the chunk and sequence forms).

What it computes is _mix_rows's tail before the output projection, per row t and v head h:
    b = fp32(bf16(o[t, h]));  y = w * (b * rsqrt(mean(b^2) + eps));  out[t, h] = bf16(y * sigmoid(z[t, h]))
with o the delta rule's fp32 [T, Hv, Dv] (kernels/delta_rule.py), z the output gate bf16 [T, Hv, Dv] and w the
o_norm weight fp32 [Dv]; out is bf16 [T, Hv * Dv], the output projection's input.

Why a kernel: in the 1M R8 prefill call (lc2's p200 replay, 4096 rows per rank, 2 v heads per rank) neuronx-cc
lowers these few elementwise ops into ~1.6 ms per KDA layer (4096 vector STREAM_TRANSPOSEs 851 us, TENSOR_REDUCE
648 us, scalar casts 759 us), against ~1.4 ms for the delta rule itself and 0.3 ms for the output projection;
34 layers x ~1.1 ms of it per call (docs/neuron-notes.md "The prefill call's compute side"). Here rows are tokens
on the partitions and (head, d) on the free axis, the delta rule's own output layout, so every reduction is along
the free axis and nothing is transposed.

Arithmetic per 128-row tile: one bf16 rounding copy of the tile (vector), per head the sum of squares by
activation_reduce (square, add), rsqrt(ss / Dv + eps) on the scalar engine and (b r) w on the vector engine, then
sigmoid(z) on the scalar engine and one multiply into the bf16 output (rounded once). The same operation order as
the torch code; what differs is the fp32 summation order of the mean and the scalar engine's rsqrt / sigmoid, so
the result is inside the floor, not bit-identical (tools/probe_gated_norm.py measures both against the device's XLA
path and an fp64 reference).

The delta-rule kernel's source is untouched (its REV and every default graph key stay); this kernel has its own.
"""

from __future__ import annotations

import os

import torch

# KILN_KDA_FUSED_NORM: "1" on, "0" off, unset: the model's default (takes(default=...): on for glm5_next since
# 2026-10-07, docs/neuron-notes.md "The prefill call's compute side").
MODE = os.environ.get("KILN_KDA_FUSED_NORM")
P = 128


def reference(o: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """_mix_rows's KDA tail in torch (o fp32 [T, Hv, Dv], z bf16 [T, Hv, Dv], w fp32 [Dv]): bf16 [T, Hv * Dv]."""
    of = o.to(torch.bfloat16).float()
    of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
    of = w * of
    out = (of * torch.sigmoid(z.float())).to(torch.bfloat16)
    return out.reshape(o.shape[0], -1)


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    @nki.jit
    def kiln_gated_norm_kernel(o, z, w, eps: float, rev: int):
        """See the module docstring. o fp32 [To, Hv, Dv] with To >= T (the delta rule's padded output may be passed
        as is: only its first T rows are read), z bf16 [T, Hv, Dv], w fp32 [Dv]; eps the norm's epsilon; rev this
        module's source revision (REV). Returns bf16 [T, Hv * Dv]."""
        f32, bf16 = nl.float32, nl.bfloat16
        T, Hv, Dv = z.shape
        F = Hv * Dv
        out = nl.ndarray((T, F), dtype=bf16, buffer=nl.shared_hbm)
        W = nl.ndarray((P, Dv), dtype=f32, buffer=nl.sbuf)  # w on every partition (a stride-0 read)
        nisa.dma_copy(dst=W, src=w.ap(pattern=[[0, P], [1, Dv]], offset=0))
        E = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=E, value=eps)
        for c in range((T + P - 1) // P):
            n = min(P, T - c * P)  # rows of this tile (the last may be partial)
            ot = nl.ndarray((n, F), dtype=f32, buffer=nl.sbuf)
            nisa.dma_copy(dst=ot, src=o.ap(pattern=[[F, n], [1, F]], offset=c * P * F))
            zt = nl.ndarray((n, F), dtype=bf16, buffer=nl.sbuf)
            nisa.dma_copy(dst=zt, src=z.ap(pattern=[[F, n], [1, F]], offset=c * P * F))
            ob = nl.ndarray((n, F), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=ob, src=ot, engine=nisa.vector_engine)  # bf16(o), rounded once
            sg = nl.ndarray((n, F), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=sg, op=nl.sigmoid, data=zt)
            y = nl.ndarray((n, F), dtype=f32, buffer=nl.sbuf)
            for h in range(Hv):
                sq = nl.ndarray((n, Dv), dtype=f32, buffer=nl.sbuf)
                ss = nl.ndarray((n, 1), dtype=f32, buffer=nl.sbuf)
                nisa.activation_reduce(dst=sq, op=nl.square, data=ob[:, h * Dv:(h + 1) * Dv], reduce_op=nl.add,
                                       reduce_res=ss)
                r = nl.ndarray((n, 1), dtype=f32, buffer=nl.sbuf)
                nisa.activation(dst=r, op=nl.rsqrt, data=ss, scale=1.0 / Dv, bias=E[0:n, :])
                nisa.scalar_tensor_tensor(dst=y[:, h * Dv:(h + 1) * Dv], data=ob[:, h * Dv:(h + 1) * Dv],
                                          op0=nl.multiply, operand0=r, op1=nl.multiply, operand1=W[0:n, :])
            res = nl.ndarray((n, F), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=res, data1=y, data2=sg, op=nl.multiply, engine=nisa.vector_engine)
            nisa.dma_copy(dst=out.ap(pattern=[[F, n], [1, F]], offset=c * P * F), src=res)
        return out
else:
    kiln_gated_norm_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source, passed as the static argument `rev` (LNL's compile-cache key does not
    include NKI kernel source: kernels/delta_rule.py _kernel_rev)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_gated_norm_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev()


def takes(kind: str, gate_act: str, dv: int, default: bool = False) -> bool:
    """Whether the kernel runs this layer's prefill tail: on (KILN_KDA_FUSED_NORM, else the model's default), KDA (fp32
    norm weight), a sigmoid output gate, head_v_dim <= 512, on trn1 (grid 1; the LNC=2 split is not written)."""
    on = MODE == "1" if MODE in ("0", "1") else default
    if not on or kind != "kda" or gate_act != "sigmoid" or dv > 512:
        return False
    from .. import platform

    return platform.nki_grid() == 1


def apply(o: torch.Tensor, z: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """The gated norm on the device, in the caller's graph: o fp32 [To >= T, Hv, Dv], z [T, Hv, Dv], w [Dv];
    returns bf16 [T, Hv * Dv]."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    if kiln_gated_norm_kernel is None:
        raise RuntimeError("the NKI gated-norm kernel needs the nki package (the Neuron venv)")
    return wrap_nki(kiln_gated_norm_kernel)[1](o=o.float().contiguous(), z=z.to(torch.bfloat16).contiguous(),
                                               w=w.float().contiguous(), eps=float(eps), rev=REV)
