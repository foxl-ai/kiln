"""Long-context pooled DSA (GLM-5.3-Flash up to its 1,048,576 positions): a DSA layer that never
materialises anything of the context's size per query.

Why: the bucketed path (models/mla.py attention, models/glm5_next.py pooled_selection) gathers the
whole context's latent and indexer rows per call and works on [C, L] visibility, mask and score
tensors. At 1M keys and a 4096-token chunk that is a 16 GB fp32 visibility, a [C, 32, P] fp32 score
tensor of 137 GB and a dense masked attention over every key (docs/neuron-notes.md "Long context
(1M)"). Here a layer's chunk (or decode batch) goes through three steps whose sizes are per query
O(keep) or O(P / sub), never O(L):

1. scores: the pooled indexer's score of every candidate pool, index[q, p] = sum_h w[q, h]
   relu(scale qI[q, h] . pk[p]) (glm5_next.pool_index, Glm5NextTextIndexer.forward), where the
   candidates of a query at position pos are the COMPLETE pools p < npool(pos) = floor((pos + 1) /
   kpool): a pool is selectable once its last token kpool p + kpool - 1 <= pos. The candidates are a
   prefix of the pool axis, so no mask is needed, only a count.
2. selection: the `keep` = index_topk / kpool best candidates, ties to the lowest pool index, all of
   them when fewer (the rule of kernels/dsa_topk.py and dsa_select.reference_mask), as a list of pool
   indices per query in ascending order plus their count. select_two_level() is the algorithm the
   device kernel runs (below), select_reference() the dense definition; tests/test_dsa_long.py holds
   them equal on adversarial ties.
3. attention over slots: the selected pools' tokens plus the query's own incomplete pool (the tail,
   tokens kpool npool .. pos; Glm5NextTextIndexer.append_visible_tail), gathered by pool row from the
   latent cache and attended with absorbed MLA (kernels/dsa_decode.py's arithmetic, which takes any
   number of query rows): slots() builds the rows and the 0 / NEG_INF bias exactly as
   glm5_next.decode_slots does for a decode batch.

Two-level exact selection (select_two_level). The pools are cut into sub-blocks of `sub`. Let t be
the keep-th largest candidate score and M the keep-th largest sub-block maximum (sub-blocks with no
candidate count as NEG_INF).
- t >= M: `keep` sub-blocks have a maximum >= M, each holds an element >= M, so at least keep
  elements are >= M.
- Select the top-keep sub-blocks by maximum with the same rule (ties to the lowest sub-block index).
  Every element of the exact selection lies in a selected sub-block: an element above t lies in a
  sub-block whose maximum exceeds t >= M, which is selected; a selected element equal to t (one of
  the `room` = keep - #(> t) lowest-index ties) lies in a sub-block B with maximum >= t >= M. If B's
  maximum exceeds M it is selected; if it equals M (= t), every tied sub-block before B holds an
  element equal to t at a lower index than the element, so fewer than room of them precede B, while
  the tied sub-blocks have keep - #(blocks above t) >= keep - #(elements above t) = room places.
- So the exact selection is the exact top-keep (same tie rule) of the selected sub-blocks' elements
  taken in pool order (the sub-blocks gathered in ascending index order keep index order inside).
Work per query: one maximum pass over P, then exact top-keep over P / sub maxima and over keep x sub
candidates instead of over P: at P = 262,144 pools (1M tokens) and sub = 32, 8,192 + 16,384 values
instead of 262,144 (the radix rounds of kernels/dsa_topk.py cost per value, docs/neuron-notes.md).
"""

from __future__ import annotations

import os

import torch

NEG_INF = -1e30  # models/decoder.NEG_INF
VISIBLE = -5e29  # kernels/dsa_topk.VISIBLE: scores at or below it are not candidates

# The long path takes a pooled DSA layer's bucket when its context L (page bucket x page size) exceeds
# LONG_KEYS: below it the bucketed path (and its trn1 kernels, which hold up to 4096 pools = 16,384
# keys) is unchanged, so the G1 graphs (8,448 keys) are untouched.
LONG_KEYS = int(os.environ.get("KILN_DSA_LONG_KEYS", 16384))
# Pools per sub-block of the two-level selection (a power of two).
SUB = int(os.environ.get("KILN_DSA_LONG_SUB", 32))
# Selection on the host (the device runs the kernel's own form): "two_level" (the kernel's algorithm,
# the default) or "reference" (the dense definition).
HOST_SELECT = os.environ.get("KILN_DSA_LONG_SELECT", "two_level")
# The decode form's scores on the device (KILN_DSA_LONG_SCORER): "xla" (scores(): the pool keys gathered by page, an
# einsum) or "index" (kernels/dsa_index.py, the decode agent's kernel from feat/decode-scale 2fbe874: each row's pool
# keys read straight from the paged pool-key cache, 0.68 ms per row per DSA layer on a trn1 core and 0.26 on a trn2
# logical core at 262,144 pools against ~3.1 ms in XLA, measured by that agent 2026-10-05). The kernel needs the
# separate bf16 pool-key cache and a page bucket that is a multiple of 128 pages; elsewhere "xla" is used.
SCORER = os.environ.get("KILN_DSA_LONG_SCORER", "xla")
if SCORER not in ("xla", "index"):
    raise ValueError(f"KILN_DSA_LONG_SCORER must be xla or index, not {SCORER!r}")


def enabled(L: int) -> bool:
    """Whether a pooled DSA bucket of L keys runs the long path (static per bucket)."""
    return L > LONG_KEYS


def npools(positions: torch.Tensor, kp: int) -> torch.Tensor:
    """[N] int64: the complete pools a query at each position may select, floor((pos + 1) / kp). kp is a
    power of two, so the multiply is exact in fp32 for positions below 2^24 (no in-graph integer
    division: it is inexact on trn1, docs/neuron-notes.md)."""
    if kp & (kp - 1):
        raise ValueError(f"kpool {kp} is not a power of two")
    return torch.floor((positions.to(torch.float32) + 1.0) * (1.0 / kp)).to(torch.int64)


def scores(qI: torch.Tensor, w: torch.Tensor, pk: torch.Tensor, npool: torch.Tensor, scale: float) -> torch.Tensor:
    """[N, P] fp32: every pool's index score, NEG_INF added for the non-candidates (p >= npool).
    qI [N, Hi, D], w [N, Hi] fp32, pk [P, D] (one sequence: a prefill chunk) or [N, P, D] (one context
    per row: a decode batch), npool [N]. The arithmetic of glm5_next.pool_index (one fp32 einsum over
    D, then the weighted head sum), so on the host the long path scores exactly as the bucketed one."""
    s = (torch.einsum("nhd,pd->nhp", qI.float(), pk.float()) if pk.dim() == 2
         else torch.einsum("nhd,npd->nhp", qI.float(), pk.float()))
    index = torch.einsum("nh,nhp->np", w.float(), torch.relu(s * scale))
    P = index.shape[-1]
    cand = torch.arange(P, device=index.device).view(1, P) < npool.view(-1, 1)
    return index + torch.where(cand, 0.0, NEG_INF)


def _exact_mask(sc: torch.Tensor, keep: int) -> torch.Tensor:
    """bool [N, M]: kernels/dsa_topk.py's selection (vis_only) of the keep best of each row."""
    from ..kernels import dsa_topk

    return dsa_topk.emulate(sc, keep, True) == 0


def _ascending(sel: torch.Tensor, keep: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(indices [N, keep] int64 ascending, count [N]) of a bool mask with at most keep True per row; the
    slots past the count hold 0."""
    N, M = sel.shape
    j = torch.arange(M, device=sel.device).expand(N, M)
    key = torch.where(sel, j, M + j)  # selected first, each group in index order
    order = torch.sort(key, dim=-1, stable=True).values[:, :keep] if M >= keep else torch.sort(key, dim=-1).values
    cnt = sel.sum(-1)
    if order.shape[1] < keep:
        order = torch.cat([order, order.new_full((N, keep - order.shape[1]), M)], dim=1)
    idx = torch.where(torch.arange(keep, device=sel.device).view(1, keep) < cnt.view(N, 1), order, 0)
    return idx.to(torch.int64), cnt.to(torch.int64)


def select_reference(sc: torch.Tensor, keep: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The exact selection of scores [N, P] (non-candidates at or below VISIBLE) as (pools [N, keep]
    ascending, count [N]): the dense definition."""
    return _ascending(_exact_mask(sc, keep), keep)


def select_two_level(sc: torch.Tensor, keep: int, sub: int = SUB) -> tuple[torch.Tensor, torch.Tensor]:
    """The same selection by sub-block maxima (module docstring), equal to select_reference."""
    N, P = sc.shape
    if sub < 1 or sub & (sub - 1):
        raise ValueError(f"sub-block size {sub} is not a power of two")
    NB = -(-P // sub)
    if NB <= keep:  # every sub-block would be selected: level 2 is the whole row
        return select_reference(sc, keep)
    pad = NB * sub - P
    s = torch.cat([sc, sc.new_full((N, pad + sub), NEG_INF)], dim=1)  # + one all-NEG_INF block, index NB
    blocks = s.view(N, NB + 1, sub)
    bm = blocks[:, :NB].amax(-1)  # [N, NB]
    bidx, bcnt = _ascending(_exact_mask(bm, keep), keep)  # the top-keep sub-blocks, ascending
    bidx = torch.where(torch.arange(keep, device=sc.device).view(1, keep) < bcnt.view(N, 1), bidx, NB)
    cand = torch.gather(blocks, 1, bidx.unsqueeze(-1).expand(N, keep, sub)).reshape(N, keep * sub)
    cidx, cnt = _ascending(_exact_mask(cand, keep), keep)
    pools = torch.gather(bidx, 1, cidx // sub) * sub + cidx % sub
    pools = torch.where(torch.arange(keep, device=sc.device).view(1, keep) < cnt.view(N, 1), pools, 0)
    return pools, cnt


def select(sc: torch.Tensor, keep: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The host selection (HOST_SELECT)."""
    if HOST_SELECT == "reference":
        return select_reference(sc, keep)
    if HOST_SELECT != "two_level":
        raise ValueError(f"KILN_DSA_LONG_SELECT must be two_level or reference, not {HOST_SELECT!r}")
    return select_two_level(sc, keep)


def compact(sel: torch.Tensor, keep: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(indices [N, keep] int64 ascending, count [N]) of a 0 / 1 float mask sel [N, M] with at most keep ones per
    row, in graph-friendly ops (the device form of _ascending): the k-th one is the number of positions whose
    inclusive count of ones is <= k, found in two levels of G ~ sqrt(M) positions (the whole groups before it, then
    inside its group; fp32 counts are exact below 2^24), as glm5_next.decode_slots does at G = 8. No scan runs over
    all M positions: neuronx-cc lowers a cumsum to a reduce-window, and one over the 262,144 pools of a 1M context
    generated 4,194,304 instructions (NCC_EXTP003, trn2 target, 2026-10-05), so the group counts are sums, their scan
    is M / G long, and the scan inside the k-th one's group runs over that group's G values only."""
    N, M = sel.shape
    G = 1
    while G * G < M and M % (G * 2) == 0:
        G *= 2
    NG = M // G
    sv = sel.view(N, NG, G)
    Cg = sv.sum(-1).cumsum(-1)  # [N, NG]: inclusive count at each group's end
    kk = torch.arange(keep, device=sel.device, dtype=Cg.dtype).view(1, keep, 1)
    grp = (Cg.unsqueeze(1) <= kk).to(Cg.dtype).sum(-1)  # [N, keep]: whole groups before the k-th one
    gi = grp.to(torch.int64).clamp(max=NG - 1)
    prev = torch.where(gi > 0, torch.gather(Cg, 1, (gi - 1).clamp(min=0)), torch.zeros_like(grp))  # ones before it
    inner = torch.gather(sv, 1, gi.unsqueeze(-1).expand(N, keep, G)).cumsum(-1) + prev.unsqueeze(-1)  # [N, keep, G]
    idx = (grp * G + (inner <= kk).to(Cg.dtype).sum(-1)).to(torch.int64)
    cnt = Cg[:, -1].to(torch.int64)
    valid = torch.arange(keep, device=sel.device).view(1, keep) < cnt.view(N, 1)
    return torch.where(valid, idx, torch.zeros_like(idx)), cnt


def select_device(sc: torch.Tensor, keep: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The exact selection of scores [N, P] as select_reference returns it, through kernels/dsa_topk.py's
    selection (on the device: rows in groups small enough for its MAX_W scores per partition; on the host its
    emulation) and compact(): the decode batch's form, whose rows each have their own context."""
    from ..kernels import dsa_topk

    N, P = sc.shape
    if keep >= P:
        sel = (sc > torch.full_like(sc, VISIBLE)).to(torch.float32)  # no float-literal compare (NCC_ESPP004, f64)
    else:
        step = N
        while step > 1 and not dsa_topk.supported(step, P):
            step //= 2
        if not dsa_topk.supported(step, P):
            raise NotImplementedError(f"long-context DSA: {P} pools exceed the selection kernel for one row")
        parts = [dsa_topk.select(sc[i:i + step], keep, vis_only=True) for i in range(0, N, step)]
        sel = torch.exp(torch.cat(parts) if len(parts) > 1 else parts[0])  # 0 / NEG_INF -> 1 / 0
    return compact(sel, keep)


def slots(pools: torch.Tensor, cnt: torch.Tensor, npool: torch.Tensor, positions: torch.Tensor,
          table: torch.Tensor, page_size: int, kp: int, n_slots: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The attention's slots for N queries: rows [N, n_slots] int64 (each slot's pool row, the latent
    cache viewed as [slots / kp, kp R]: page * page_size / kp + the pool's index in its page) and bias
    [N, n_slots, kp] fp32 (0 for a token to attend, NEG_INF otherwise). Slots 0 .. keep - 1 are the
    selected pools (pools, cnt from select), slot keep the query's tail pool npool (its tokens kp npool
    .. pos; none when pos + 1 is a multiple of kp), the rest padding. table: [Pp] one sequence's pages
    (a prefill chunk) or [N, Pp] one per row (a decode batch). glm5_next.decode_slots' layout, built by
    broadcasting rather than concatenation (a concatenate of these small int64 / bool pieces failed neuronx-cc
    2.27 with NCC_IFML902 "FlattenMacroLoop ... Cannot remove an edge", trn1, 2026-10-05)."""
    N, keep = pools.shape
    ppp = page_size // kp
    if page_size % kp or ppp & (ppp - 1) or n_slots < keep + 1:
        raise ValueError(f"slots: page_size {page_size}, kpool {kp}, {n_slots} slots for {keep} + 1 pools")
    Pp = table.shape[-1]
    sl = torch.arange(n_slots, device=pools.device).view(1, n_slots)
    tail = npool.clamp(max=Pp * ppp - 1).view(N, 1)  # a full context has no tail: its slot reads the last pool, masked
    sel = torch.gather(pools, 1, sl.clamp(max=keep - 1).expand(N, n_slots))
    pl = torch.where(sl < keep, sel, torch.where(sl == keep, tail, torch.zeros_like(sel)))
    pg = torch.floor(pl.to(torch.float32) * (1.0 / ppp)).to(torch.int64)  # power of two: exact
    tb = table.view(1, Pp).expand(N, Pp) if table.dim() == 1 else table
    rows = tb.gather(1, pg) * ppp + (pl - pg * ppp)
    t = torch.arange(kp, device=pools.device).view(1, 1, kp)
    sel_ok = ((sl < keep) & (sl < cnt.view(N, 1))).unsqueeze(-1)  # [N, S, 1]
    tail_ok = (sl == keep).unsqueeze(-1) & (npool.view(N, 1, 1) * kp + t <= positions.view(N, 1, 1))  # [N, S, kp]
    ok = sel_ok | tail_ok
    return rows, torch.where(ok, 0.0, NEG_INF).to(torch.float32)


def attend(q_lat: torch.Tensor, kc: torch.Tensor, rows: torch.Tensor, bias: torch.Tensor, scale: float) -> torch.Tensor:
    """o [N, H, R] fp32: absorbed MLA of N queries over their slots (kernels/dsa_decode.py's arithmetic:
    its emulation on the host)."""
    from ..kernels import dsa_decode

    return dsa_decode.attend(q_lat, kc, rows, bias, scale)


def reference_mask(pools: torch.Tensor, cnt: torch.Tensor, npool: torch.Tensor, positions: torch.Tensor, L: int,
                   kp: int) -> torch.Tensor:
    """bool [N, L]: the tokens a query attends (selected pools' tokens, its tail, causal), for tests: what
    glm5_next.block_mask plus the visibility select in the bucketed path."""
    N, keep = pools.shape
    tok = torch.zeros(N, L, dtype=torch.bool)
    for n in range(N):
        for k in range(int(cnt[n])):
            p = int(pools[n, k])
            tok[n, p * kp:(p + 1) * kp] = True
        tok[n, int(npool[n]) * kp:int(positions[n]) + 1] = True
    return tok & (torch.arange(L).view(1, L) <= positions.view(N, 1))


# -- context parallelism (KILN_DSA_CP=1) --------------------------------------------------------------------------
#
# Each rank of an attention group of A ranks holds 1 / A of every sequence's DSA cache (latent, indexer rows, pool
# keys): context pool c (tokens kpool c .. kpool c + kpool - 1) lives on rank c % A. With page_size / kpool pools per
# page a multiple of A, rank r owns pools r, r + A, ... of every page, so its local cache has page_size / A slots per
# page and the same logical pages as the host's allocator (prefix cache, host tier and admission unchanged), and its
# m-th local pool of a context is context pool c = m A + r, stored at local pool row table[m // ppl] ppl + m % ppl
# (ppl = page_size / (kpool A) local pools per page): the same row for every rank. Per call each rank:
# - writes the tokens it owns (the rest go to its null page),
# - scores its local candidate pools and keeps their exact top-keep (local index order = context order),
# - gathers every rank's list (score, context pool) over the group and merges them exactly (cp_merge: the global
#   top-keep is inside the union of the local ones, each local list being the first keep of its rank's pools in the
#   global order "score descending, context pool ascending"),
# - attends its own selected pools (and the query's tail pool if it owns it) with EVERY head, returning the
#   normalised partial output and its log-sum-exp, and the partials are combined by their log-sum-exps
#   (cp_combine), each rank keeping its heads for W_UV and o_proj.


def cp_enabled() -> bool:
    return os.environ.get("KILN_DSA_CP", "0") == "1"


# KILN_DSA_CP_DEGREE=<a> (opt-in, with KILN_DSA_CP=1 at DP attention 1): the DSA caches are context-parallel over a
# ranks, not over the whole attention group of A = attn_tp ranks. The group then splits into A / a ROW GROUPS of a
# consecutive ranks (engine/tp.py attention_group(world, a)), each holding a full context-parallel copy of every
# sequence's DSA cache (each rank 1 / a of it), and a prefill chunk's T rows are split over the row groups: row group g
# selects and attends rows g T / R .. (g + 1) T / R with every head (attention_cp_rows), so per rank the selection, the
# merge and the slot attention are those of a T / R-row chunk at CP a, while the chunk itself has T rows. A decode
# batch runs attention_cp unchanged inside each row group (CP collectives over the a ranks, each group attending its
# a x (heads per rank) heads). Unset (0) or a >= A: the context-parallel degree is the attention TP, as before.
def cp_degree_env() -> int:
    return int(os.environ.get("KILN_DSA_CP_DEGREE", "0") or 0)


def cp_degree(attn_tp: int) -> int:
    """The context-parallel degree for an attention TP of attn_tp when KILN_DSA_CP=1 (cp_degree_env, else attn_tp)."""
    a = cp_degree_env()
    if a <= 0 or a >= attn_tp:
        return attn_tp
    if a < 2:
        raise ValueError(f"KILN_DSA_CP_DEGREE={a}: a context-parallel degree is at least 2 (0 or unset: the attention TP)")
    if attn_tp % a or a & (a - 1):
        raise ValueError(f"KILN_DSA_CP_DEGREE={a} must be a power of two dividing the attention TP {attn_tp}")
    return a


def cp_local_slots(positions: torch.Tensor, table: torch.Tensor, page_size: int, kp: int, A: int,
                   rank: torch.Tensor) -> torch.Tensor:
    """[T] int64: the local cache slot of each written token on this rank (rank [1] int64, its attention rank), or
    the DUMP slot when another rank owns it: the null page's last local slot (cp_dump_slot), which no query reads
    (a padded row at position 0 reads its pool's token 0; duplicate scatter destinations are nondeterministic, so
    nothing that is read may be one). table: [Pp] (one sequence) or [T, Pp] (one row each). Exact float
    arithmetic: page_size, kpool and A are powers of two and positions below 2^24."""
    lps = page_size // A
    pf = positions.to(torch.float32)
    i = torch.floor(pf * (1.0 / page_size))
    within = pf - i * page_size
    j = torch.floor(within * (1.0 / kp))  # pool in page
    t = within - j * kp
    jA = torch.floor(j * (1.0 / A))
    owner = j - jA * A
    ii = i.to(torch.int64)
    page = table[ii] if table.dim() == 1 else table.gather(1, ii.view(-1, 1)).view(-1)
    local = page * lps + (jA * kp + t).to(torch.int64)
    return torch.where(owner.to(torch.int64) == rank.view(()), local, torch.full_like(local, cp_dump_slot(page_size, A)))


def cp_dump_slot(page_size: int, A: int) -> int:
    """The local slot unowned and padded tokens write under context parallelism: the null page's last one (token
    kpool - 1 of its last local pool, never visible to the padded rows' position 0)."""
    return page_size // A - 1


def cp_pool_rows(table: torch.Tensor, page_size: int, kp: int, A: int) -> torch.Tensor:
    """[(B,) Pp ppl] int64: the local pool row (the cache viewed as [slots / kp, kp, ...]) of every local pool m of
    the context(s) a block table addresses, m = 0 .. Pp ppl - 1 (local pool m = context pool m A + rank)."""
    ppl = page_size // (kp * A)
    off = torch.arange(ppl, device=table.device, dtype=table.dtype)
    return (table.unsqueeze(-1) * ppl + off).flatten(-2)


def cp_local_count(npool: torch.Tensor, A: int, rank: torch.Tensor) -> torch.Tensor:
    """[N]: the local candidates of each query (context pools m A + rank < npool)."""
    n = torch.floor((npool.to(torch.float32) - rank.to(torch.float32).view(()) + (A - 1)) * (1.0 / A))
    return n.clamp(min=0).to(torch.int64)


# Rows per cp_merge piece for a DECODE batch (attention_cp passes it for a block table [B, P] only): a decode batch above
# it is merged in pieces of it (each row is independent). The 96-row decode graph's merge failed neuronx-cc 2.27
# ([NCC_INIC902] NeuronInstComb error ... APIndex.py:205 on its [96, 4096] tie search, q/dc1-t1 cp-R96; 80 rows
# compiled), so larger batches never form it; at or below it the graph is unchanged. A prefill chunk's merge (thousands
# of rows, compiled as one) stays one piece.
CP_MERGE_ROWS = int(os.environ.get("KILN_DSA_CP_MERGE_ROWS", 64))
# KILN_DSA_CP_MERGE_BOUND=1: the tie search's bits from the bucket's context pools (cp_merge's cpools) instead of 21
# (12 at a 64-page decode bucket of page 256, 15 at 4096 pages); exact, the same lim, opt-in (new graphs).
CP_MERGE_BOUND = os.environ.get("KILN_DSA_CP_MERGE_BOUND", "0") == "1"
TIE_BITS = 21  # context pools < 2^21 (1M tokens are 2^18 pools)


def merge_bits(cpools: int | None) -> int:
    """The tie search's bits for context pool indices below cpools (TIE_BITS without a bound)."""
    if not CP_MERGE_BOUND or not cpools:
        return TIE_BITS
    return max(1, min(TIE_BITS, (int(cpools) - 1).bit_length()))


def cp_merge(vals: torch.Tensor, cpool: torch.Tensor, keep: int, rows: int | None = None,
             cpools: int | None = None) -> torch.Tensor:
    """cp_merge_rows in pieces of at most `rows` rows (one piece, the same graph, when rows is None or N <= rows).
    cpools: the context pools the indices lie below (the bucket's, A x the local pools), for merge_bits."""
    N = vals.shape[0]
    bits = merge_bits(cpools)
    kw = {} if bits == TIE_BITS else {"bits": bits}  # (the unbounded call as before the bound existed)
    if rows is None or N <= rows:
        return cp_merge_rows(vals, cpool, keep, **kw)
    parts = []
    for i in range(0, N, rows):
        parts.append(cp_merge_rows(vals[i:i + rows], cpool[i:i + rows], keep, **kw))
    return torch.cat(parts)


def cp_merge_rows(vals: torch.Tensor, cpool: torch.Tensor, keep: int, bits: int = TIE_BITS) -> torch.Tensor:
    """bool [N, A, keep]: which candidates of every rank's local list belong to the exact global top-keep. vals
    [N, A, keep] fp32 (NEG_INF for an empty entry), cpool [N, A, keep] their context pool indices (fp32, exact). The
    threshold t is the keep-th largest value (kernels/dsa_topk.py's selection over the A keep candidates, whatever its
    tie order, then the smallest selected value); every candidate above t is in, and of those equal to t the `room`
    with the smallest context pool index (a binary search over the index, exact in fp32)."""
    from ..kernels import dsa_topk

    N, A, K = vals.shape
    s = vals.reshape(N, A * K).float()
    c = cpool.reshape(N, A * K).float()
    valid = s > torch.full_like(s, VISIBLE)  # a tensor, not a float literal: those lower to f64 (NCC_ESPP004, trn1)
    if A * K <= keep:
        return valid.view(N, A, K)
    if dsa_topk.supported(N, A * K):
        sel0 = dsa_topk.select(s, keep, vis_only=True) == 0
        big = torch.full_like(s, 3.0e38)
        t = torch.where(sel0, s, big).amin(-1, keepdim=True)  # the keep-th largest (3e38 when nothing is valid)
    else:  # more candidates per row than the kernel holds (A = 32: 16,384)
        t = _keep_th_two_level(s, keep)
    above = s > t
    tied = (s == t) & valid
    room = keep - above.to(torch.float32).sum(-1, keepdim=True)
    lim = torch.zeros(N, 1, dtype=torch.float32, device=s.device)
    for b in range(bits - 1, -1, -1):  # every context pool index < 2^bits (TIE_BITS = 21 by default)
        cand = lim + float(2 ** b)
        ok = (tied & (c < cand)).to(torch.float32).sum(-1, keepdim=True) < room
        lim = torch.where(ok, cand, lim)
    return (above | (tied & (c <= lim))).view(N, A, K)


def _keep_th_two_level(s: torch.Tensor, keep: int) -> torch.Tensor:
    """[N, 1] fp32: the keep-th largest of each row of s [N, M] (3e38 when no entry is above VISIBLE), through
    kernels/dsa_topk.py in two levels when a row is wider than the kernel: the row cut into q equal parts, each part's
    top keep (dsa_compact'ed), then the top keep of their q keep values. Exact, ties included: a part holds fewer than
    keep values above the threshold t (the whole row does), so all of them survive its level, and either some part
    keeps keep values >= t or every part keeps all of its values equal to t; either way the union holds keep values >=
    t and its keep-th largest is t."""
    from ..kernels import dsa_topk

    N, M = s.shape
    q = 2
    while (M % q or not dsa_topk.supported(N * q, M // q)) and M // q > keep:
        q *= 2
    if M % q or not dsa_topk.supported(N * q, M // q) or not dsa_topk.supported(N, q * keep):
        raise NotImplementedError(f"long-context DSA: a merge over {M} candidates per row")
    W = M // q
    s2 = s.reshape(N * q, W)
    sel = (dsa_topk.select(s2, keep, vis_only=True) == 0).to(torch.float32)
    idx, cnt = compact(sel, keep)
    k = torch.arange(keep, device=s.device).view(1, keep)
    v = torch.where(k < cnt.view(-1, 1), torch.gather(s2, 1, idx), torch.full_like(s2[:, :keep], NEG_INF))
    v = v.reshape(N, q * keep)
    sel1 = dsa_topk.select(v, keep, vis_only=True) == 0
    return torch.where(sel1, v, torch.full_like(v, 3.0e38)).amin(-1, keepdim=True)


def cp_attend_partial(q_lat: torch.Tensor, kc: torch.Tensor, rows: torch.Tensor, bias: torch.Tensor,
                      scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(o [N, H, R] fp32 normalised over this rank's slots, lse [N, H] fp32): dsa_decode.emulate's arithmetic and
    the log-sum-exp of its scaled, biased scores (kernels/dsa_slots.py with lse=True on the device)."""
    from ..kernels import dsa_decode

    N, H, R = q_lat.shape
    kc2 = kc.reshape(kc.shape[0], -1)
    if q_lat.device.type != "cpu":
        from ..kernels import dsa_slots  # kernel 2 of the long-context work (lse=True)

        return dsa_slots.attend(q_lat, kc, rows, bias, scale, lse=True)
    KP = dsa_decode.KP
    NS = rows.shape[1]
    tok = (rows.long().unsqueeze(-1) * KP + torch.arange(KP, device=rows.device)).reshape(N, NS * KP)
    rnd = (lambda t: t.to(torch.bfloat16).float()) if kc2.dtype != torch.float32 else (lambda t: t.float())
    K = rnd(kc2[tok])
    s = torch.einsum("bhr,btr->bht", rnd(q_lat), K) * scale + bias.reshape(N, 1, NS * KP)
    lse = torch.logsumexp(s, dim=-1)
    p = rnd(torch.softmax(s, dim=-1))
    return torch.einsum("bht,btr->bhr", p, K), lse
