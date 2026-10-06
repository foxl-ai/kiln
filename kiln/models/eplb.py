"""Expert-parallel load balancing (EPLB) with redundant expert slots, for the expert-parallel MoE layers
(models/decoder.py moe_ep_enabled, kernels/moe_ep.py).

Why: under expert parallelism a MoE layer's step waits for its busiest rank (the block's all-reduce or
reduce-scatter needs every rank's partial sum), and GLM-5.3-Flash routes very unevenly: on the random-token
prompts of bench/serve_sweep.py the busiest of 32 ranks holds 3.8x the mean pairs at 4096 rows, because a few
experts of the deeper layers take most tokens (layer 20: four experts with 3711-4093 of 4096 tokens each;
docs/neuron-notes.md "Expert parallelism"). Placing experts differently cannot split one such expert, so like
SGLang and vLLM this adds redundant copies of the hottest experts and splits their pairs between the copies.

The reference engines (both take DeepSeek's EPLB algorithm):
- SGLang v0.5.21 python/sglang/srt/eplb/: eplb_algorithms/deepseek.py replicate_experts (each redundant slot to
  the expert with the largest load per replica) and balanced_packing; expert_distribution.py records per-layer
  per-expert counts on the GPU into a ring buffer; eplb_manager.py rebalances every
  --eplb-rebalance-num-iterations (1000) forward passes; expert_location_dispatch.py:121-161, without an
  all-to-all backend, picks a pair's replica by its row index modulo the replica count so every rank agrees.
- vLLM v0.30.0 vllm/distributed/eplb/: policy/default.py rebalance_experts (replicate_experts, balanced_packing,
  preserve_intragpu_slots keeps experts that stay on a GPU in their slot); eplb_state.py records counts per
  rank's tokens (fused_moe/router/base_router.py _eplb_map_and_record_i32_kernel) over a window of 1000 steps
  and rearranges every step_interval (3000) steps; the replica pick is a hash of the local token index modulo
  the replica count (base_router.py:51-56).

Kiln's form keeps every expert's PRIMARY copy where the contiguous placement puts it (rank r: experts r El ..
r El + El - 1, El = E / tp) and adds `s` redundant slots per rank and layer (KILN_EP_REDUNDANT) that hold
replicas of the hottest experts. A rebalance therefore rewrites s slots per rank and layer and nothing else
(vLLM's preserve_intragpu_slots taken to its limit), and the graphs never change: the slot count is a shape,
the placement is data. Simulated on real GLM-5.3-Flash routing (tools/eplb_sim.py over tools/ep_routing.py's
saved top-k, 2026-10-05): at 4096 rows the busiest rank's modelled EP-kernel time summed over the 42 MoE layers
is 199 ms contiguous, 140 ms with one redundant slot per rank (137 ms for a full re-placement, 123 ms for an
oracle placement built from the evaluated batches themselves).

Physical expert ids: primary e is p = e; redundant slot j of rank r is p = E + r s + j. A rank's local map
(kernels/moe_ep.local_map form) sends its primaries to slots 0 .. El - 1 and its redundant slots to El .. El + s - 1.
Routing ids are mapped to physical ids before the kernel (remap): a pair (row t, expert e) whose expert has
n copies goes to copy (t mod R) mod n, R = RMAX classes of rows (SGLang's row-index dispatch). Only experts
that have replicas are touched, so the map is a small elementwise sum, exact in int32.
"""

from __future__ import annotations

import functools
import os

import torch

RMAX = 16


def _host(fn):
    """Host-side placement code runs on the CPU even inside the meta-device context a model is built in."""
    @functools.wraps(fn)
    def wrapped(*a, **kw):
        with torch.device("cpu"):
            return fn(*a, **kw)
    return wrapped  # row classes: a pair of row t takes copy (t mod RMAX) mod n of its expert's n copies


def redundant_slots() -> int:
    """KILN_EP_REDUNDANT: redundant expert slots per rank in every expert-parallel layer (default 0: the
    contiguous layout, graphs unchanged)."""
    s = int(os.environ.get("KILN_EP_REDUNDANT", "0"))
    if s < 0:
        raise ValueError(f"KILN_EP_REDUNDANT must be >= 0, not {s}")
    return s


def record_enabled() -> bool:
    """KILN_EPLB_RECORD (default 1; only layers with redundant slots record): prefill graphs with sequence-parallel
    routing count each expert's pairs of this rank's rows into a per-layer device buffer (ep_stats), read at a
    rebalance (SGLang turns its expert-distribution recorder on whenever EPLB is: arg_groups/parallel_hook.py).
    On by default so that a static placement and a periodic rebalance run the same graphs; the count is an
    elementwise compare and sum over the rank's r x k pairs. Read when a graph is traced."""
    return os.environ.get("KILN_EPLB_RECORD", "1") == "1"


def rebalance_interval() -> int:
    """KILN_EPLB_INTERVAL: rebalance the redundant slots every this many prefill calls from the recorded counts
    (0, the default: never; the initial placement stays)."""
    return int(os.environ.get("KILN_EPLB_INTERVAL", "0"))


def decode_replicas() -> bool:
    """KILN_EPLB_DECODE: 1 decode and verify calls spread a replicated expert's pairs over its copies too, 0 sends
    them to the primary (a copy slot then has no decode pairs). Default: 0 with the MoE-kernel agent's decode v2 and
    later (KILN_MOE_EP_SMALL_V >= 2: an expert with no pairs costs nothing, a used copy one more pass and its weight
    reads; measured G64 conc 64 on trn1.32xlarge, kiln-tq-32, 2026-10-05: decode call 0.166 s spreading, 0.162 s on the
    primaries, 135.6 -> 136.7 out tok/s), 1 with the static-pass decode kernel (every slot pays its pass anyway)."""
    v = os.environ.get("KILN_EPLB_DECODE")
    if v is not None:
        return v == "1"
    from ..kernels import moe_ep  # the kernel's own SMALL_V, so this follows its default (2 since 54b2e56)

    return moe_ep.SMALL_V < 2


def default_extra(E: int, tp: int, s: int) -> list[int]:
    """The redundant slots before any statistics: slot j of rank r holds the j-th expert of the next rank (a
    placeholder that no copy shares a rank with its primary)."""
    El = E // tp
    return [(((r + 1) % tp) * El + j) % E for r in range(tp) for j in range(s)]


def _copies(load: torch.Tensor, tp: int, s: int) -> torch.Tensor:
    """Copies per expert (primary included) for E + tp s slots: DeepSeek's replicate, one more copy at a time for
    the expert with the largest load per copy."""
    E = load.shape[0]
    cnt = torch.ones(E, dtype=torch.float64)
    for _ in range(tp * s):
        e = int(torch.argmax(load / cnt))
        cnt[e] += 1
    return cnt


def _place(load: torch.Tensor, tp: int, s: int, keep: list[int] | None) -> list[int]:
    E = load.shape[0]
    El = E // tp
    cnt = _copies(load, tp, s)
    per = load / cnt
    held = [set(range(r * El, (r + 1) * El)) for r in range(tp)]
    tot = [float(per[r * El:(r + 1) * El].sum()) for r in range(tp)]
    slots: list[list[int | None]] = [[None] * s for _ in range(tp)]
    left = (cnt - 1).tolist()  # copies still to place per expert
    if keep is not None:  # a copy whose expert still deserves one stays in its slot (no reload)
        for r in range(tp):
            for j in range(s):
                e = keep[r * s + j]
                if left[e] >= 1 and e not in held[r]:
                    slots[r][j] = e
                    held[r].add(e)
                    tot[r] += float(per[e])
                    left[e] -= 1
    copies = sorted(((float(per[e]), e) for e in range(E) for _ in range(int(left[e]))), key=lambda x: (-x[0], x[1]))
    for w, e in copies:
        cand = [r for r in range(tp) if None in slots[r] and e not in held[r]]
        if not cand:
            continue
        r = min(cand, key=lambda q: (tot[q], q))
        slots[r][slots[r].index(None)] = e
        held[r].add(e)
        tot[r] += w
    n = torch.ones(E, dtype=torch.float64)
    for sl in slots:
        for e in sl:
            if e is not None:
                n[e] += 1
    for r in range(tp):
        while None in slots[r]:
            score = load / n
            for e in held[r]:
                score[e] = -1.0
            e = int(torch.argmax(score))
            slots[r][slots[r].index(None)] = e
            held[r].add(e)
            n[e] += 1
    return [e for sl in slots for e in sl]


def busiest_load(load: torch.Tensor, extra: list[int], tp: int, s: int) -> float:
    """The busiest rank's expected pairs under a placement, each expert's pairs shared evenly by its copies."""
    load = load.to(torch.float64).cpu()
    E = load.shape[0]
    El = E // tp
    n = torch.ones(E, dtype=torch.float64)
    for e in extra:
        n[e] += 1
    per = load / n
    return max(float(per[r * El:(r + 1) * El].sum()) + sum(float(per[e]) for e in extra[r * s:(r + 1) * s])
               for r in range(tp))


@_host
def replicas(load: torch.Tensor, tp: int, s: int, prev: list[int] | None = None, keep_tol: float = 0.05) -> list[int]:
    """The expert each redundant slot holds, [tp s] in slot order (rank-major), from per-expert loads [E]:
    DeepSeek's replicate (one more copy for the expert with the largest load per copy, until E + tp s copies),
    then each new copy, heaviest first, onto the least-loaded rank (the primaries' loads shared by their copies)
    with a free redundant slot and no copy of that expert yet; slots left over (only ranks already holding an
    expert had room for its copy) take the heaviest expert per copy they do not hold. Deterministic: the same
    loads give the same placement on every rank.

    prev (the current placement): sticky. Every copy whose expert still deserves one keeps its slot and only the
    rest are placed, which is taken unless its busiest rank carries more than keep_tol over a fresh placement's
    (vLLM v0.30.0 policy/default.py preserve_intragpu_slots keeps experts that stay on a GPU in their slot; its
    ROCm path and SGLang's eplb_min_rebalancing_utilization_threshold skip a rearrangement that gains too little):
    a stationary load then reloads a few slots per rebalance instead of most of them."""
    load = load.to(torch.float64).cpu() + 1e-6
    fresh = _place(load, tp, s, None)
    if prev is None or len(prev) != tp * s:
        return fresh
    sticky = _place(load, tp, s, list(prev))
    return sticky if busiest_load(load, sticky, tp, s) <= (1 + keep_tol) * busiest_load(load, fresh, tp, s) else fresh


@_host
def physical_lmap(E: int, tp: int, s: int, rank: int) -> torch.Tensor:
    """int32 [1, E + tp s + 1]: each physical expert's slot on `rank` (primaries 0 .. El - 1, redundant slots
    El .. El + s - 1), El + s for another rank's and for the padding expert (the last entry)."""
    El = E // tp
    P = E + tp * s
    m = torch.full((P + 1,), El + s, dtype=torch.int32)
    m[rank * El:(rank + 1) * El] = torch.arange(El, dtype=torch.int32)
    m[E + rank * s:E + (rank + 1) * s] = torch.arange(El, El + s, dtype=torch.int32)
    return m.view(1, -1)


@_host
def tables(extra: list[int], E: int, tp: int, s: int, spread: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """(rep_ids int32 [tp s], rep_map int32 [tp s, RMAX]) for remap: the distinct replicated experts (padded with
    -1, which no routing id equals) and, per row class, the physical id a pair of that expert takes. spread False:
    every class takes the primary (remap is then the identity)."""
    Q = tp * s
    ids = torch.full((Q,), -1, dtype=torch.int32)
    mp = torch.zeros(Q, RMAX, dtype=torch.int32)
    copies: dict[int, list[int]] = {}
    for i, e in enumerate(extra):
        copies.setdefault(e, [e]).append(E + i)
    for q, (e, ps) in enumerate(sorted(copies.items())):
        ids[q] = e
        for c in range(RMAX):
            mp[q, c] = ps[c % len(ps)] if spread else e
    return ids, mp


def remap(topi: torch.Tensor, rep_ids: torch.Tensor, rep_map: torch.Tensor) -> torch.Tensor:
    """Routing ids [T, k] -> physical ids: a pair of row t whose expert is rep_ids[q] takes rep_map[q, t mod RMAX],
    every other pair keeps its id (its primary). Elementwise int32 (no matmul, which auto-cast would round)."""
    T, k = topi.shape
    Q, R = rep_map.shape
    t = topi.to(torch.int32)
    # Row classes as a static one-hot [T, R] (an identity tiled over the rows: no integer division in the graph).
    sel = torch.eye(R, dtype=torch.int32, device=topi.device).repeat(-(-T // R), 1)[:T]
    per_row = (sel.unsqueeze(1) * rep_map.unsqueeze(0)).sum(-1)  # [T, Q]: the copy row t's pairs of expert q take
    hit = (t.unsqueeze(-1) == rep_ids.view(1, 1, Q)).to(torch.int32)  # [T, k, Q]
    delta = (hit * (per_row - rep_ids.view(1, Q)).unsqueeze(1)).sum(-1)  # [T, k]
    return (t + delta).to(topi.dtype)


def counts(topi: torch.Tensor, E: int) -> torch.Tensor:
    """fp32 [E]: pairs routed to each (logical) expert in topi [T, k]."""
    ar = torch.arange(E, device=topi.device, dtype=topi.dtype)
    return (topi.reshape(-1, 1) == ar).to(torch.float32).sum(0)


@_host
def load_file(path: str, E: int) -> dict[int, torch.Tensor]:
    """Per-layer expert loads {model layer index: fp64 [E]} from a statistics file: a recorder dump ({"load":
    {layer: [E]}}, Kiln's eplb dump) or tools/ep_routing.py's routing ({"topi": {layer: [[T, k] per sequence]}})."""
    d = torch.load(path, map_location="cpu")
    if "load" in d:
        return {int(l): torch.as_tensor(v, dtype=torch.float64) for l, v in d["load"].items()}
    if "topi" in d:
        return {int(l): torch.bincount(torch.cat([t.long().flatten() for t in seqs]), minlength=E).to(torch.float64)
                for l, seqs in d["topi"].items()}
    raise ValueError(f"{path}: neither a recorder dump ('load') nor tools/ep_routing.py routing ('topi')")
