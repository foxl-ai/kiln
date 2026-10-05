"""GLM-5.3-Flash (`glm5_next`, Glm5NextForConditionalGeneration): the text model.

45 layers: 34 Kimi Delta Attention layers (models/linear_attn.py, the "safe gate" lower bound
-5) and 11 DSA layers, every 4th from layer 3 (config layer_types). A DSA layer is MLA without
RoPE (qk_rope_head_dim 0: 64 heads of 256 nope + 256 value dims over a 512 latent, q_lora 1536)
with a lightning indexer whose keys are scored in POOLS of index_kpool = 4 consecutive tokens
(below). The residual is 4 streams mixed by manifold-constrained hyper-connections
(models/hybrid.py), the MLP a SwiGLU clamped at swiglu_limit 10, first 3 layers dense
(intermediate 12288), then 288 routed experts (sigmoid routing with a selection bias, top 8,
routed_scaling_factor 2.5) plus one shared expert. The MTP layer (layers.45) and the vision
tower are not loaded.

Numerics follow transformers v5.18.0 src/transformers/models/glm5_next/modeling_glm5_next.py
(https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/glm5_next/modeling_glm5_next.py),
config keys its configuration_glm5_next.py; the checkpoint's config.json:
https://huggingface.co/zai-org/GLM-5.3-Flash/raw/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json

The pooled indexer (Glm5NextTextIndexer.forward, get_pooled_states, append_visible_tail): each
token keeps its indexer key k (LayerNorm(wk x)) and gate logits g = index_kpool_compress_gate x.
Pool p holds tokens kpool p .. kpool p + kpool - 1 (pooling starts at the first real token,
which is position 0 here), its key is the per-channel softmax-weighted mean
sum_j softmax_j(g_j + ape_j) k_j, and it is a candidate for a query only once its LAST token is
visible. A query scores every candidate pool with the usual index (sum_h w_h relu(q_h . k_p /
sqrt(Di))), keeps the index_topk / kpool best, expands them back to their tokens, and with
index_kpool_always_select_tail also attends to the tokens of its own incomplete pool. When the
context holds at most index_topk / kpool pools every pool is kept, so below index_topk tokens the
layer is exactly dense causal MLA (as for DeepSeek-V3.2 / GLM-5.3, models/mla.py), which is the
static per-bucket test Kiln uses. Above it the selection is an additive mask (KILN_DSA=gather is
not implemented for pools).
"""

from __future__ import annotations

import os

import torch

from ..config import AttnSpec, LinearSpec, ModelConfig
from . import dsa_select
from .decoder import NEG_INF
from .mla import DSASpec, MLASpec, cache_widths

ARCHITECTURES = ("Glm5NextForConditionalGeneration",)

# configuration_utils.py (v5.18.0) normalises these layer types to "indexed_attention".
_DSA_TYPES = ("deepseek_sparse_attention", "indexed_attention", "full_attention")


def config_from_hf(cls, c: dict, eos_ids: tuple[int, ...]) -> ModelConfig:
    """Keys of Glm5NextTextConfig (configuration_glm5_next.py, v5.18.0); a flat text-only file
    is accepted as well as the composite one (text fields under text_config)."""
    from .hybrid import HybridSpec

    t = c.get("text_config") or c
    n = t["num_hidden_layers"]
    types = t.get("layer_types") or ["linear_attention" if i % 4 != 3 else "indexed_attention" for i in range(n)]
    mlp = t.get("mlp_layer_types") or ["dense"] * min(3, n) + ["sparse"] * (n - 3)
    if t.get("hidden_act", "silu") != "silu":
        raise NotImplementedError(f"hidden_act {t.get('hidden_act')!r}")
    if t.get("qk_rope_head_dim", 0):
        raise NotImplementedError("glm5_next DSA layers are NoPE (transformers rejects qk_rope_head_dim > 0)")
    if t.get("q_lora_rank") is None:
        raise NotImplementedError("glm5_next DSA needs q_lora_rank")
    if t.get("scoring_func", "sigmoid") != "sigmoid" or t.get("topk_method", "noaux_tc") != "noaux_tc":
        raise NotImplementedError("only sigmoid / noaux_tc routing (Glm5NextTextTopkRouter)")
    # Linear attention: linear_attn_config, with transformers' defaults (Glm5NextTextConfig:
    # linear_lower_bound -5.0, also when safe_gate is set without a bound).
    la = dict(t.get("linear_attn_config") or {})
    h, d = la.get("num_heads", t.get("linear_num_heads", 64)), la.get("head_dim", t.get("linear_head_dim", 128))
    lb = la.get("gate_lower_bound", t.get("linear_lower_bound", -5.0))
    if lb is None and la.get("safe_gate", True):
        lb = -5.0
    kda = LinearSpec("kda", h, h, d, d, la.get("short_conv_kernel_size", t.get("linear_conv_kernel_dim", 4)),
                     "sigmoid", d, float(lb) if lb is not None else None)
    # DSA layers. The indexer of every one is "full" in the checkpoint (indexer_types); IndexShare
    # over pooled selections is not implemented.
    from .mla import _indexer_types

    itypes = _indexer_types(t, n)
    dn, dv = t["qk_nope_head_dim"], t["v_head_dim"]
    kpool = int(t.get("index_kpool", 16))
    if t.get("index_topk", 2048) % kpool:
        raise ValueError("index_topk must be divisible by index_kpool")
    if kpool > 1 and not t.get("index_kpool_compress", True):
        raise NotImplementedError("index_kpool_compress=false")
    specs = []
    for i in range(n):
        if types[i] == "linear_attention":
            specs.append(kda)
            continue
        if types[i] not in _DSA_TYPES:
            raise NotImplementedError(f"layer type {types[i]!r}")
        if itypes[i] != "full":
            raise NotImplementedError("glm5_next IndexShare (indexer_types 'shared') over pooled selections")
        dsa = DSASpec(t["index_n_heads"], t["index_head_dim"], t["index_topk"], rope_interleave=False, indexer=True,
                      kpool=kpool, kpool_tail=bool(t.get("index_kpool_always_select_tail", True)))
        # q_a_layernorm / kv_a_layernorm are built with rms_norm_eps (Glm5NextTextAttention).
        m = MLASpec(t["q_lora_rank"], t["kv_lora_rank"], dn, 0, dv, dn ** -0.5, False, rope=False,
                    norm_eps=t.get("rms_norm_eps", 1e-5), dsa=dsa)
        kw, vw = cache_widths(m)
        specs.append(AttnSpec(t["num_attention_heads"], 1, kw, vw, 0, 10000.0, mla=m))
    experts = t.get("n_routed_experts") or 0
    # The MTP layer (num_nextn_predict_layers, model.language_model.layers.<n> in zai-org/GLM-5.3-Flash's
    # model.safetensors.index.json) is a pooled-DSA MLA layer with the MoE and NO hyper-connections
    # (vLLM v0.30.0 models/glm5next/nvidia/model.py Glm5NextDecoderLayer is_mtp_layer: "MTP layers use
    # the non-mHC path"; nvidia/mtp.py Glm5NextMultiTokenPredictorLayer: enorm, hnorm, eh_proj,
    # shared_head). Its later passes run their own indexer: reusing the first pass's pooled selection
    # (index_share_for_mtp_iteration) is not implemented, so drafts can differ from vLLM's there.
    from .mla import mtp_fields

    mtp_dsa = DSASpec(t["index_n_heads"], t["index_head_dim"], t["index_topk"], rope_interleave=False, indexer=True,
                      kpool=kpool, kpool_tail=bool(t.get("index_kpool_always_select_tail", True)))
    mtp_m = MLASpec(t["q_lora_rank"], t["kv_lora_rank"], dn, 0, dv, dn ** -0.5, False, rope=False,
                    norm_eps=t.get("rms_norm_eps", 1e-5), dsa=mtp_dsa)
    mtp_kw, mtp_vw = cache_widths(mtp_m)
    mtp = mtp_fields(t, n, AttnSpec(t["num_attention_heads"], 1, mtp_kw, mtp_vw, 0, 10000.0, mla=mtp_m), bool(experts))
    mtp.pop("mtp_index_share", None)
    hy = HybridSpec("glm5_next", hc=int(t.get("hc_mult", 4)), sinkhorn_iters=int(t.get("hc_sinkhorn_iters", 20)),
                    hc_eps=float(t.get("hc_eps", 1e-6)), swiglu_limit=t.get("swiglu_limit"), block_norms=True)
    first = next((s for s in specs if not isinstance(s, LinearSpec)), None)
    eos = eos_ids or _ids(t.get("eos_token_id"))
    return cls(
        architecture=c["architectures"][0], vocab_size=t["vocab_size"], hidden_size=t["hidden_size"],
        intermediate_size=t["intermediate_size"], num_layers=n, num_heads=t["num_attention_heads"], num_kv_heads=1,
        head_dim=first.head_dim if first is not None else d, rms_norm_eps=t.get("rms_norm_eps", 1e-5),
        rope_theta=10000.0, max_position_embeddings=t.get("max_position_embeddings", 1048576),
        tie_word_embeddings=bool(t.get("tie_word_embeddings", False)), eos_token_ids=eos,
        num_experts=experts, num_experts_per_tok=t.get("num_experts_per_tok", 0) or 0,
        moe_intermediate_size=t.get("moe_intermediate_size", 0) or 0, norm_topk_prob=t.get("norm_topk_prob", True),
        moe_layers=tuple(i for i in range(n) if experts and mlp[i] == "sparse"), attn_layers=tuple(specs),
        router_scoring="sigmoid", router_bias=bool(experts), routed_scaling_factor=float(t.get("routed_scaling_factor") or 1.0),
        n_shared_experts=(t.get("n_shared_experts") or 0) if experts else 0, n_group=t.get("n_group") or 1,
        topk_group=t.get("topk_group") or 1, hybrid=hy, **mtp, **cls._quant(c if "quantization_config" in c else t))


def _ids(e) -> tuple[int, ...]:
    return tuple(e) if isinstance(e, list) else ((e,) if e is not None else ())


def pool_keys(layer, d: DSASpec, rows: torch.Tensor) -> torch.Tensor:
    """Each pool's key [..., Di] from its kpool tokens' indexer rows [..., kpool, 2 Di] (key, then
    gate logits): the per-channel softmax over the pool of gate + ape, weighting the keys
    (get_pooled_states). The same arithmetic wherever a pool key is made: per decode step from the
    gathered context (KILN_DSA_POOL_CACHE=0, the sequence form) or once per written token into the
    pool-key cache (models/mla.py write_pool_keys)."""
    Di = d.head_dim
    k = rows[..., :Di]
    logits = rows[..., Di : 2 * Di].float() + layer.idx_pool_ape.float()
    return (torch.softmax(logits, dim=-2).to(k.dtype) * k).sum(dim=-2)


def pooled_selection(layer, d: DSASpec, q, w: torch.Tensor, kI: torch.Tensor | None, vis: torch.Tensor,
                     pk: torch.Tensor | None = None) -> torch.Tensor:
    """Additive mask [B, Q, L] (0 = selected, NEG_INF = not) of the pooled indexer for B x Q
    queries over L keys: q = (rope part or None, rest [B, Q, Hi, Di - dr]), w [B, Q, Hi] fp32,
    kI [B, L, 2 Di] each token's key then gate logits, or pk [B, L / kpool, Di] the pool keys
    themselves (read from the pool-key cache, models/mla.py), vis [B, Q, L] the causal visibility
    (0 / NEG_INF, a prefix of the context for every query). Glm5NextTextIndexer.forward, with
    get_pooled_states and append_visible_tail, as one batched computation: no per-query loop and
    no comparison against a float literal (docs/neuron-notes.md)."""
    q_r, q_p = q
    kp, Di = d.kpool, d.head_dim
    B, Q, L = vis.shape
    pad = -L % kp  # the sequence form (forward_logits): the last pool may be incomplete
    if pad:
        vis = torch.cat([vis, torch.full((B, Q, pad), NEG_INF, dtype=vis.dtype, device=vis.device)], dim=-1)
    if pk is None:
        if pad:
            kI = torch.cat([kI, kI.new_zeros(B, pad, kI.shape[-1])], dim=1)
        P = (L + pad) // kp
        pk = pool_keys(layer, d, kI[..., : 2 * Di].reshape(B, P, kp, 2 * Di))  # [B, P, Di]
    dr = 0 if q_r is None else q_r.shape[-1]
    if SCORE_KERNEL and dsa_select.SELECT == "nki" and B == 1 and q_r is None and vis.device.type != "cpu":  # noqa
        from ..kernels import dsa_topk

        if dsa_topk.score_supported(Q, d.n_heads, Di, P := (L + pad) // kp):
            # One sequence (a prefill chunk): scores and selection in one kernel (kernels/dsa_topk.py).
            cand = vis.reshape(B, Q, P, kp)[..., kp - 1]
            sel = dsa_topk.score_select(q_p[0], w[0], pk[0], cand[0], d.topk // kp, Di ** -0.5, kp, d.kpool_tail)
            return sel.view(B, Q, L + pad)[..., :L]
    index = pool_index(d, q, w, pk)
    return block_mask(index, vis, kp, d.topk // kp, d.kpool_tail, select=dsa_select.SELECT)[..., :L]


def pool_index(d: DSASpec, q, w: torch.Tensor, pk: torch.Tensor) -> torch.Tensor:
    """The pooled indexer's scores [B, Q, P] of every pool: sum_h w_h relu(q_h . k_p / sqrt(Di)) (q = (rope part or
    None, rest [B, Q, Hi, Di - dr]), w [B, Q, Hi] fp32, pk [B, P, Di] the pool keys)."""
    q_r, q_p = q
    dr = 0 if q_r is None else q_r.shape[-1]
    s = torch.einsum("bqhd,bpd->bqhp", q_p.float(), pk[..., dr:].float())
    if q_r is not None:
        s = s + torch.einsum("bqhd,bpd->bqhp", q_r.float(), pk[..., :dr].float())
    return torch.einsum("bqh,bqhp->bqp", w, torch.relu(s * d.head_dim ** -0.5))


def decode_slots(d: DSASpec, q, w: torch.Tensor, pk: torch.Tensor, vis: torch.Tensor, table: torch.Tensor,
                 page_size: int, slots: int):
    """The pools one decode query per row attends, for kernels/dsa_decode.py: block_mask's selection (the
    d.topk / kpool best-scored complete visible pools, kernels/dsa_topk.py on the device) and, with d.kpool_tail,
    the row's own incomplete pool, as cache rows instead of a mask over the bucket. vis [B, 1, L] (0 / NEG_INF, a
    prefix), table [B, L / page_size] the rows' pages. Returns rows [B, slots] (each slot's pool row, the cache viewed
    as [N / kpool, kpool R]: page * page_size / kpool + the pool's index in its page; a slot past the selection holds
    row 0) and bias [B, slots, kpool] fp32
    (0 for a token to attend, NEG_INF for the rest). Slots 0 .. keep - 1 are the selected pools in pool order (the
    order does not matter to a softmax), slot keep the tail pool."""
    kp = d.kpool
    B, _, L = vis.shape
    P = L // kp
    keep = d.topk // kp
    ppp = page_size // kp
    if page_size % kp or ppp & (ppp - 1) or slots < keep + 1:
        raise ValueError(f"decode_slots: page_size {page_size}, kpool {kp}, {slots} slots for {keep} + 1 pools")
    cand = vis.reshape(B, 1, P, kp)[..., kp - 1]  # a pool is a candidate once its last token is visible
    sc = pool_index(d, q, w, pk) + cand
    from ..kernels import dsa_topk

    # 1.0 for a selected pool, 0.0 else: exp of the 0 / NEG_INF mask (no comparison against a float literal, which
    # neuronx-cc lowers through f64: NCC_ESPP004, measured on this graph 2026-10-04)
    s01 = torch.exp(dsa_topk.select(sc, keep, vis_only=True, kp=1, tail=False).view(B, P))
    # The k-th selected pool (0-based) is the number of pools whose inclusive count of selected pools is <= k, found
    # in two levels in fp32 (exact: counts below 2^24): the groups of G pools wholly before it, then within its group.
    # One NeuronCore of trn1.32xlarge, 2112 pools, 512 kept (/opt/kiln/prof/probe_compact.py on kiln-g2-trn1,
    # 2026-10-04): the int64 scatter of each pool to its rank 5.2 ms at 16 rows (GpSimd, element by element), the
    # count over all pools in int64 16.1 ms, in fp32 0.91 ms (2.45 ms at 64 rows), these two levels 0.40 ms (0.96).
    C = s01.cumsum(-1)  # [B, P]
    G = 8 if P % 8 == 0 else 1
    kk = torch.arange(keep, device=vis.device, dtype=C.dtype).view(1, keep, 1)
    Cg = C.view(B, P // G, G)
    grp = (Cg[:, :, G - 1].unsqueeze(1) <= kk).to(C.dtype).sum(-1)  # [B, keep]: whole groups before it
    gi = grp.to(torch.int64).clamp(max=P // G - 1)
    inner = torch.gather(Cg, 1, gi.unsqueeze(-1).expand(B, keep, G))  # [B, keep, G]: its group's counts
    idx = (grp * G + (inner <= kk).to(C.dtype).sum(-1)).to(torch.int64)  # [B, keep]
    valid = kk.view(1, keep) < C[:, -1:]  # the row's selected count
    # the tail: tokens from kp floor(visible / kp) on (block_mask's), visible count exact in fp32
    nvis = torch.exp(vis.reshape(B, L)).sum(-1, keepdim=True)  # [B, 1]
    ptf = torch.floor(nvis * (1.0 / kp))
    pt = ptf.to(torch.int64).clamp(max=P - 1)  # a full context has no tail pool: its slot reads pool P - 1, all masked
    pools = torch.cat([torch.where(valid, idx, torch.zeros_like(idx)), pt,
                       torch.zeros(B, slots - keep - 1, dtype=torch.int64, device=vis.device)], dim=1)  # [B, slots]
    # page and offset of each pool by multiplying, not dividing (a power of two: exact in fp32; docs/neuron-notes.md
    # on in-graph integer division)
    pg = torch.floor(pools.float() * (1.0 / ppp)).to(torch.int64)
    rows = table.gather(1, pg) * ppp + (pools - pg * ppp)  # pool rows: the cache viewed as [N / kp, kp R]
    t = torch.arange(kp, device=vis.device, dtype=nvis.dtype).view(1, kp)
    tail_vis = ptf * kp + t < nvis  # [B, kp]
    ok = torch.cat([valid.unsqueeze(-1).expand(B, keep, kp), tail_vis.unsqueeze(1),
                    torch.zeros(B, slots - keep - 1, kp, dtype=torch.bool, device=vis.device)], dim=1)
    bias = torch.where(ok, 0.0, NEG_INF).to(torch.float32)
    return rows, bias


# A prefill chunk's (one sequence's) pooled scores and selection as one NKI kernel when the selection
# is "nki" (kernels/dsa_topk.py kiln_dsa_score_topk_kernel): its [C, Hi, L / kpool] score tensor never
# leaves the chip. KILN_DSA_SCORE_KERNEL=0 keeps the torch scores before the selection kernel.
SCORE_KERNEL = os.environ.get("KILN_DSA_SCORE_KERNEL", "1") == "1"

# Rounds of the bisection that finds the keep-th largest block score (block_mask select="range"):
# each halves [min, max] of the scores, so 32 resolve scores closer than 2^-32 of their range.
BISECT_ROUNDS = 32


def block_mask(index: torch.Tensor, vis: torch.Tensor, kp: int, keep: int, tail: bool,
               select: str = "topk") -> torch.Tensor:
    """Additive mask [B, Q, L] keeping, per query, the `keep` best-scored blocks of kp tokens
    among those whose last token it can see (index [B, Q, L / kp] the block scores, vis [B, Q, L]
    0 / NEG_INF with L a multiple of kp), expanded to their tokens, plus (tail) its own incomplete
    block. Shared by GLM-5.3-Flash's pooled DSA and Qwen3.8-Flash-Next's QSA (models/qwen4_exp.py),
    whose selections differ only in how a block is scored. Written without a comparison against a
    float literal (docs/neuron-notes.md: those can lower to f64 on neuronx-cc).

    select: GLM-5.3-Flash's pooled_selection passes KILN_DSA_SELECT (dsa_select.SELECT), QSA its own
    KILN_QSA_SELECT; every value means one algorithm wherever it is used:
    "nki" (KILN_DSA_SELECT's default): kernels/dsa_topk.py, the keep best of the visible candidates,
    the lowest indices among those tied with the keep-th best (all visible ones if fewer), exact by
    construction on the host and on the device; selection, expansion to tokens and tail are one NKI
    kernel on the device (emulate() on the host).
    "bisect" / "radix": models/dsa_select.topk_mask's float-order bisection / bitcast radix (CPU
    only), the same tie rule, restricted to the candidates. (On trn1 the bisection's split points
    come from the device's sqrt, and it was not exact on wide-range scores: tools/probe_dsa_select.py,
    docs/neuron-notes.md.)
    "range" (KILN_QSA_SELECT's default; GLM-5.3-Flash's, under the name "bisect", until 2026-10-04):
    the keep-th largest score v found by BISECT_ROUNDS halvings of [min, max] (a count per round, no
    sort), every block above v kept and as many of those scoring exactly v as fill `keep`, lowest
    index first (a cumulative sum). Exact only when the 32 halvings separate the keep-th score from
    the next lower one; on scores spread over many orders of magnitude it keeps far more than
    `keep` (measured: ~2090 of 2112 pools for keep 512).
    "topk": torch.topk and a scatter (an arbitrary choice among exact ties; transformers' own
    choice is torch.topk's on its own layout, which none reproduces: CPU torch.topk is not
    lowest-index-first among ties)."""
    B, Q, L = vis.shape
    P = L // kp
    cand = vis.reshape(B, Q, P, kp)[..., kp - 1]  # a block is a candidate once its last token is visible
    sc = index + cand
    if select == "nki":  # selection, expansion and tail in one kernel (kernels/dsa_topk.py)
        from ..kernels import dsa_topk

        return dsa_topk.select(sc, keep, vis_only=True, kp=kp, tail=tail)
    if select in ("bisect", "radix"):  # dsa_select's, then no invisible block (a row with fewer)
        sel = torch.where(dsa_select.topk_mask(sc, keep, select), 0.0, NEG_INF) + cand
        sel = torch.maximum(sel, torch.full_like(sel, NEG_INF))
    elif select == "range":
        # Invariant: at least `keep` candidates score >= lo (or every candidate, if fewer); the
        # invisible ones sit at NEG_INF, below the smallest score of any block.
        lo, hi = index.amin(-1, keepdim=True), index.amax(-1, keepdim=True)
        for _ in range(BISECT_ROUNDS):
            mid = (lo + hi) * 0.5
            ok = (sc >= mid).float().sum(-1, keepdim=True) >= keep
            lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
        # v: the smallest score at or above lo; keep the blocks above it, then the first ties.
        v = torch.where(sc >= lo, sc, hi).amin(-1, keepdim=True)
        above, tied = sc > v, sc >= v
        room = keep - above.float().sum(-1, keepdim=True)
        first = (tied & ~above).float().cumsum(-1) <= room
        sel = torch.where(above | (tied & first), 0.0, NEG_INF)
    elif select == "topk":
        top = torch.topk(sc, keep, dim=-1).indices
        sel = torch.full_like(cand, NEG_INF).scatter(-1, top, 0.0)  # [B, Q, P]
    else:
        raise ValueError(f"block selection must be nki, bisect, radix, range or topk, not {select!r}")
    sel = sel.unsqueeze(-1).expand(B, Q, P, kp).reshape(B, Q, L)
    if tail:
        # The query's own incomplete block: visible tokens from kp * floor(visible / kp) on. The
        # visible count is exp(vis).sum() (0 / -inf -> 1 / 0), exact in fp32.
        nvis = torch.exp(vis).sum(-1, keepdim=True)
        start = torch.floor(nvis * (1.0 / kp)) * kp
        j = torch.arange(L, device=vis.device, dtype=vis.dtype)
        sel = torch.maximum(sel, torch.where(j >= start, 0.0, NEG_INF))
    return sel
