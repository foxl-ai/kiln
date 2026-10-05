"""Quantized weights: FP8 storage with fp32 block scales along the input dim.

Kiln's one in-memory format is `w` float8_e4m3fn [N, K] plus `scale` fp32 [N, K / bk], and a
weight is `w * scale` with each scale covering bk consecutive input columns of one row.
Both checkpoint formats MiMo-V2.6 ships convert to it EXACTLY:

- FP8 with 128 x 128 block scales (DeepSeek-V3 convention, `weight_scale_inv` [N/128, K/128],
  dequantised as w_fp8 * scale_inv): repeating each block's scale over its 128 rows gives
  per-row scales, which also lets tensor parallelism slice rows anywhere (a replicated KV
  head of 192 rows is not a multiple of 128).
- MXFP4 (`store_dtype: mxfp4`, block 32): 4-bit E2M1 codes packed two per byte, element 2i
  in the LOW nibble, with one E8M0 exponent per 32 elements, value = E2M1 * 2^(e - 127)
  (transformers integrations/mxfp4.py, _convert_moe_packed_tensors, the gpt-oss reference;
  vLLM loads MiMo-V2.6's experts through the same Mxfp4MoEMethod path). Every E2M1 value
  {0, 0.5, 1, 1.5, 2, 3, 4, 6} is an exact e4m3 value, so the codes become FP8 losslessly
  and the exponent becomes the block scale.

MXFP4 experts can also stay packed on the device (uint8 [N, K/2] plus a bf16 power-of-two scale
per 32 columns, 0.5625 bytes per weight against FP8's 1.0625), decoded in-graph by
mxfp4_decode. That halves their HBM, but the decode is vector-engine bound: 11 ms for 32 of
MiMo-V2.6-Flash's tp=32 expert gate/up blocks on trn1, against 0.5 ms to dequantize the same
blocks from FP8 (tools/profile_moe.py --parts, 2026-10-02). So FP8 is the default and packed is
opt-in (EngineConfig.mxfp4_packed).
"""

from __future__ import annotations

import os

import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
FP8 = torch.float8_e4m3fn


def mxfp4_unpack(packed: torch.Tensor, exponents: torch.Tensor):
    """packed uint8 [..., K/2], exponents uint8 [..., K/32] -> (fp8 [..., K], fp32 scale [..., K/32])."""
    lut = torch.tensor(E2M1, dtype=torch.float32)
    p = packed.to(torch.uint8)
    lo, hi = lut[(p & 0x0F).long()], lut[(p >> 4).long()]
    if MXFP4_NIBBLES == "hi":  # debugging only: high nibble first
        lo, hi = hi, lo
    vals = torch.stack([lo, hi], dim=-1)  # lo nibble first
    vals = vals.reshape(*p.shape[:-1], p.shape[-1] * 2)
    scale = torch.exp2(exponents.to(torch.float32) - 127.0)
    return vals.to(FP8), scale


def mxfp4_decode(packed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """packed uint8 [..., K/2] -> E2M1 values in `dtype` [..., K], inside a compiled graph.

    Multiplies, adds and floors only: no integer bit operations, and no comparison against a
    float literal (neuronx-cc 2.27 rejected this function written with `c >= 8.0` and
    torch.where: "NCC_ESPP004 f64 dtype is not supported", trn1, 2026-10-02). A byte and every
    intermediate below are small integers or quarters, exact in bf16, so the result equals
    mxfp4_unpack's table lookup bit for bit. For a code c = 8s + 2e + f (sign, exponent,
    mantissa bit): |v| = 0.5 f when e = 0, else 2^(e - 1) (1 + 0.5 f), which over e = 0..3 is
    e + [e == 3] + f (0.5 + e (e - 1) / 4), with [e == 3] = floor(0.375 e)."""
    b = packed.to(dtype)
    hi = torch.floor(b * 0.0625)  # high nibble: element 2i + 1
    codes = torch.stack([b - hi * 16.0, hi], dim=-1).flatten(-2)  # low nibble first
    s = torch.floor(codes * 0.125)
    m = codes - s * 8.0
    e = torch.floor(m * 0.5)
    f = m - e * 2.0
    mag = e + torch.floor(e * 0.375) + f * (0.5 + e * (e - 1.0) * 0.25)
    return mag * (1.0 - s * 2.0)


def mxfp4_byte_table(dtype: torch.dtype, device=None) -> torch.Tensor:
    """[256, 2]: the two E2M1 values of every byte, low nibble first."""
    lut = torch.tensor(E2M1, dtype=torch.float32)
    b = torch.arange(256)
    return torch.stack([lut[b & 0x0F], lut[b >> 4]], dim=-1).to(dtype).to(device)


def mxfp4_decode_table(packed: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """mxfp4_decode as one embedding lookup per byte into mxfp4_byte_table."""
    return torch.nn.functional.embedding(packed.long(), table).flatten(-2)


MXFP4_NIBBLES = os.environ.get("KILN_MXFP4_NIBBLES", "lo")
# How graphs decode packed MXFP4: "arith" (mxfp4_decode) or "table" (one lookup per byte).
MXFP4_DECODE = os.environ.get("KILN_MXFP4_DECODE", "arith")


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype,
                  table: torch.Tensor | None = None) -> torch.Tensor:
    """packed uint8 [..., N, K/2] with power-of-two scales [..., N, K/32] -> dtype [..., N, K];
    exact, since E2M1 values times powers of two are exact in bf16. `table` (from
    mxfp4_byte_table) selects the lookup decode."""
    w = mxfp4_decode(packed, dtype) if table is None else mxfp4_decode_table(packed, table)
    return (w.unflatten(-1, (scale.shape[-1], -1)) * scale.to(dtype).unsqueeze(-1)).flatten(-2)


def fp8_block_rows(w: torch.Tensor, scale_inv: torch.Tensor, block: tuple[int, int]):
    """FP8 weight [N, K] with block scales [ceil(N/bn), ceil(K/bk)] -> per-row scales [N, ceil(K/bk)]."""
    bn, _ = block
    rows = scale_inv.repeat_interleave(bn, dim=0)[: w.shape[0]]
    return w.to(FP8), rows.to(torch.float32)


def dequant(w: torch.Tensor, scale: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
    """w [..., N, K] (fp8 or already dtype), scale [..., N, K / bk] or None -> dtype [..., N, K]."""
    if scale is None:
        return w
    K = w.shape[-1]
    bk = -(-K // scale.shape[-1])
    s = scale.repeat_interleave(bk, dim=-1)[..., :K]
    return (w.to(torch.float32) * s).to(dtype)


def dequant_t(w: torch.Tensor, scale: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor:
    """dequant for a weight stored transposed: w [..., K, N], scale [..., K / bk, N]."""
    if scale is None:
        return w
    K, nb = w.shape[-2], scale.shape[-2]
    if nb > 1:
        scale = scale.repeat_interleave(-(-K // nb), dim=-2)[..., :K, :]
    return (w.to(torch.float32) * scale.to(torch.float32)).to(dtype)


def quantize_fp8_rows(w: torch.Tensor, bk: int = 128):
    """Kiln's own quantiser (tests and bf16 checkpoints converted to FP8): per (row, bk
    columns) absmax scale onto e4m3 max 240, the largest finite value on trn1 / trn2."""
    N, K = w.shape
    pad = (-K) % bk
    wf = torch.nn.functional.pad(w.float(), (0, pad)).view(N, -1, bk)
    scale = wf.abs().amax(dim=-1).clamp(min=1e-12) / 240.0
    q = (wf / scale.unsqueeze(-1)).clamp(-240, 240).view(N, -1)[:, :K]
    return q.to(FP8), scale


def fit_e4m3_max(w: torch.Tensor, scale: torch.Tensor, max_finite: float = 240.0, row_group: int | None = None):
    """Checkpoint FP8 is e4m3fn (max 448); trn1 / trn2 FP8 is e4m3 with inf (max 240), which
    neuronx-cc only accepts e4m3fn graphs as under --experimental-unsafe-fp8e4m3fn-as-fp8e4m3
    ("unsafe" because the larger fn values do not fit; AWS's vllm-neuron clamps to 240 before
    it for that reason). Halve the values of any (row, block) whose max exceeds max_finite and
    double its scale: exact for every normal e4m3 value (448 / 2 = 224).

    row_group g: decide the halving per (g consecutive rows, block) instead, where the scales are
    constant over each such group (a 128 x 128 block-scaled checkpoint's shard; ignored otherwise),
    so the scales stay constant over the group: the rows of a group that did not exceed max_finite
    are halved too. Halving is exact except for codes below 2^-5, which it makes subnormal: those
    with an odd last bit round to even, an error of 2^-10 times the doubled scale (4.4e-6 of the
    block's range); the per-row fit rounds the same codes of the rows it halves."""
    N, K = w.shape[-2], w.shape[-1]
    nb = scale.shape[-1]
    bk = -(-K // nb)
    pad = nb * bk - K
    wf = torch.nn.functional.pad(w.float(), (0, pad)).view(*w.shape[:-1], nb, bk)
    over = wf.abs().amax(dim=-1) > max_finite  # [..., N, nb]
    if not bool(over.any()):
        return w, scale
    g = row_group
    if g and g > 1 and N % g == 0 and scale.shape[-2] == N:
        sg = scale.reshape(*scale.shape[:-2], N // g, g, nb)
        if bool(torch.equal(sg, sg[..., :1, :].expand_as(sg))):  # scales constant over each row group
            og = over.reshape(*over.shape[:-2], N // g, g, nb).any(dim=-2, keepdim=True)
            over = og.expand(*over.shape[:-2], N // g, g, nb).reshape(over.shape)
    wf = torch.where(over.unsqueeze(-1), wf / 2, wf)
    return wf.reshape(*w.shape[:-1], nb * bk)[..., :K].to(FP8), torch.where(over, scale * 2, scale)
