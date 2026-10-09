"""Segmented prefill attention for dense GQA / MHA layers: vllm-neuron's kernel, over Kiln's paged KV.

Opt-in by KILN_ATTN_PREFILL=segmented (models/decoder.py _gqa, the chunk form only). The default path stays the XLA
einsum of _attend, whose fp32 scores span the whole padded page bucket ([heads / rank, C, L] per chunk: at a 32k
context a 4096-row chunk computes the full 4096 x 32768 square, about 2x the causal work, through HBM-sized
intermediates).

What runs instead is nkilib's attention_segmented_cte, the kernel vllm-neuron 0.24 calls for every chunked prefill on
trn2 (vllm_neuron/functional/attention/attention_segmented_cte.py segmented_attention, release-0.24.0.1.1.0),
vendored unmodified except for a KV layout switch (kernels/nkilib_vendor, Apache-2.0; THIRD_PARTY_NOTICES.md):
  - the chunk's queries (seqlen_q == the segment size, a multiple of 128 and of the page size) attend the context in
    segments of prior_seg_size keys read from the paged cache through the block table: first the active segment
    (the chunk's own keys, causal, together with a partial prior segment if the prior is not a whole number of
    segments), then each full prior segment from the newest back, unmasked;
  - flash-style online softmax across segments (nkilib core/utils/attention_reduce.py reduce_one_batch): no score
    tensor outside SBUF / PSUM, nothing computed past the context, and nothing above the diagonal of the active
    segment beyond its 128-row groups;
  - GQA natively: q is [heads, C, D] with num_q_heads = heads, query head h reads KV head h * nkv // heads;
  - at LNC=2 the two physical cores split the query heads (nkilib's "primary sharding"), so it runs on
    platform.nki_grid() programs.
Kiln's cache is [slots, nkv, D] = [pages, page_size, nkv, D]; nkilib's is [blocks, nkv, block_size, D]. The vendored
copy reads Kiln's layout directly (kv_bsh=True: a head's rows are nkv * D apart), so the decode path, which reads the
same cache, is untouched.

Conditions (eligible): bf16 KV (no fp8 cache), head_dim <= 128, no attention sink, no sliding window, no relative
logits, the chunk form (one sequence), and C a supported segment size that is a multiple of the page size.
"""

from __future__ import annotations

import os
import zlib

import torch

from .. import platform

MODE = os.environ.get("KILN_ATTN_PREFILL", "xla")
if MODE not in ("xla", "segmented"):
    raise ValueError(f"KILN_ATTN_PREFILL must be xla or segmented, not {MODE!r}")
ENABLED = MODE == "segmented"

# nkilib's segment sizes (vllm_neuron/utils/bucket_utils.py SUPPORTED_KV_SEGMENT_SIZES).
SEGMENT_SIZES = (512, 1024, 2048, 4096, 8192)
MAX_HEAD_DIM = 128

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = ("core/attention/attention_segmented_cte.py", "core/attention/attention_cte.py",
           "core/attention/fused_segmented_attention.py", "core/utils/attention_reduce.py",
           "core/utils/modular_allocator.py", "core/utils/kernel_helpers.py", "core/utils/allocator.py",
           "core/utils/interleave_copy.py", "core/utils/stream_shuffle_broadcast.py")


def gen3_view(text: str) -> str:
    """The vendored source as NeuronCore-v3+ traces it: lines only NeuronCore-v2 needs (ending in "# kiln-gen2")
    dropped, and a shared line changed for NeuronCore-v2 ("<code>  # kiln-gen3: <the line before>") put back as it
    was. The NeuronCore-v2 fallbacks then leave trn2's REV, and so its graph keys, exactly as before them."""
    out = []
    for ln in text.split("\n"):
        if ln.rstrip().endswith("# kiln-gen2"):
            continue
        i = ln.find("  # kiln-gen3: ")
        if i >= 0:
            ln = ln[: len(ln) - len(ln.lstrip())] + ln[i + len("  # kiln-gen3: "):]
        out.append(ln)
    return "\n".join(out)


def _rev(gen2: bool) -> int:
    """CRC-32 of the vendored kernel sources (the NeuronCore-v3+ view unless gen2): LNL's compile cache key does not
    see NKI source, so a kernel edit must change an argument of the call."""
    crc = 0
    for f in _VENDOR:
        with open(os.path.join(_HERE, "nkilib_vendor", f), encoding="utf-8") as fh:
            text = fh.read()
        crc = zlib.crc32((text if gen2 else gen3_view(text)).encode(), crc)
    return crc


REV = _rev(False)  # NeuronCore-v3+ (trn2 / trn3)
REV_GEN2 = _rev(True)  # NeuronCore-v2 (trn1 / inf2)


def _is_gen2() -> bool:
    """Whether this host's NeuronCores are v2, read once at import (not while dynamo traces: kiln/platform.py
    nki_grid); False on a host with no Neuron device (tests)."""
    try:
        p = platform.detect()
    except Exception:
        return False
    return p is not None and p.nki_gen <= 2


GEN2 = _is_gen2()

_jit = None


def _kernel():
    global _jit
    if _jit is None:
        import nki

        from .nkilib_vendor.core.attention.attention_segmented_cte import attention_segmented_cte

        _jit = nki.jit()(attention_segmented_cte)
    return _jit


def eligible(layer, C: int, page_size: int, kv_dtype: torch.dtype, fp8: bool) -> bool:
    sp = layer.spec
    return (ENABLED and not fp8 and kv_dtype == torch.bfloat16 and sp.head_dim <= MAX_HEAD_DIM
            and sp.head_dim == sp.v_head_dim and layer.sink is None and sp.window is None and not sp.rel_extent
            and not sp.log_scaled and getattr(layer, "rel_proj", None) is None and C in SEGMENT_SIZES and C % page_size == 0
            and page_size & (page_size - 1) == 0 and layer.nh % layer.nkv == 0)


def attend(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, table: torch.Tensor,
           prior: torch.Tensor, page_size: int, scale: float) -> torch.Tensor:
    """q [C, nh, D] (the chunk's queries, after RoPE); k_cache / v_cache [slots, nkv, D] with the chunk's K and V
    already stored; table [P] the sequence's pages; prior [] or [1] the chunk's first position (its cached tokens,
    a multiple of page_size). Returns [C, nh, D] in q's dtype."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    C, nh, D = q.shape
    nkv = k_cache.shape[1]
    qh = (q * scale).permute(1, 0, 2).contiguous()  # [nh, C, D]: the kernel's scale stays 1.0, as vllm-neuron's
    kc = k_cache.view(-1, page_size, nkv, D)
    vc = v_cache.view(-1, page_size, nkv, D)
    out = wrap_nki(_kernel())[platform.nki_grid()](
        q=qh, k_cache=kc, v_cache=vc, block_tables=table.reshape(1, -1).to(torch.int32),
        prior_tokens=prior.reshape(1, 1).to(torch.int32), block_size=page_size, prior_seg_size=C, scale=1.0,
        tp_q=True, tp_out=False, num_q_heads=nh, kv_bsh=True, rev=REV_GEN2 if GEN2 else REV)
    return out.permute(1, 0, 2)  # [C, nh, D]
