"""Expert-parallel load balancing (EPLB) with redundant experts, simulated on REAL GLM-5.3-Flash routing
(tools/ep_routing.py's saved top-k ids), before any device work: what the busiest rank of each MoE layer
would hold under the contiguous placement Kiln serves today and under a statistics-driven placement with
redundant expert slots, in pairs and in passes of the expert-parallel prefill kernel (kernels/moe_ep.py).

    python tools/eplb_sim.py --load ep_routing.pt [--kind random] [--slots 0 1 2] [--prefill 4096]
        [--group 1024] [--decode 64] [--trials 40] [--per-layer]

The placement follows SGLang v0.5.21 / vLLM v0.30.0's EPLB, which both take DeepSeek's algorithm
(sglang python/sglang/srt/eplb/eplb_algorithms/deepseek.py, vllm vllm/distributed/eplb/rebalance_algo.py):
(1) replicate: give one more replica to the expert with the largest load per replica until the
E + R physical slots are used (R = ranks x slots extra per layer); (2) pack: place the physical experts on
the ranks, heaviest first, each rank exactly (E + R) / ranks slots, onto the least-loaded rank with a free
slot (here also never two replicas of one expert on one rank). Loads come from a calibration set (the
other half of the sequences of the same kind, as a periodic rebalance would see the traffic it then
serves) or from the evaluated batches themselves (--oracle: the bound a per-batch placement could reach).

Dispatch to replicas: a pair (row t, expert e) with n replicas goes to replica t mod n (static, every rank
computes the same map from data; SGLang's "static" ep_dispatch_algorithm also picks a fixed replica per
(rank, expert), vLLM's default picks by token index). The kernel's pass model (moe_ep.py, 2ca0773's two
overflow loops): C >= 2048 rows run one static pass of LW = 256 lanes per local slot whatever its pairs,
then an expert's remaining pairs in 512-lane passes, the last remainder that fits 256 lanes in one 256-lane
pass; decode (<= 128 rows) runs kiln_moe_ep_small, one static pass per slot (~140 us each, measured flat)
plus 16-lane overflow passes. The step waits for the busiest rank of every layer.
"""

from __future__ import annotations

import argparse
import math

import torch


def replicate(load: torch.Tensor, phys: int) -> torch.Tensor:
    """Replica count per expert for `phys` physical slots: one each, then repeatedly one more for the expert
    with the largest load per replica (DeepSeek EPLB replicate_experts)."""
    E = load.shape[0]
    cnt = torch.ones(E, dtype=torch.long)
    for _ in range(phys - E):
        e = int(torch.argmax(load / cnt))
        cnt[e] += 1
    return cnt


def pack(load: torch.Tensor, cnt: torch.Tensor, ranks: int) -> list[list[int]]:
    """Physical experts (expert e repeated cnt[e] times, each load[e] / cnt[e]) onto `ranks` ranks of equal slot
    count, heaviest first onto the least-loaded rank with room that does not hold that expert yet."""
    phys = int(cnt.sum())
    per = phys // ranks
    items = sorted(((float(load[e]) / int(cnt[e]), e) for e in range(load.shape[0]) for _ in range(int(cnt[e]))),
                   reverse=True)
    tot = [0.0] * ranks
    slots: list[list[int]] = [[] for _ in range(ranks)]
    for w, e in items:
        cand = [r for r in range(ranks) if len(slots[r]) < per and e not in slots[r]]
        if not cand:
            cand = [r for r in range(ranks) if len(slots[r]) < per]
        r = min(cand, key=lambda q: tot[q])
        slots[r].append(e)
        tot[r] += w
    return slots


def contiguous(E: int, ranks: int) -> list[list[int]]:
    per = E // ranks
    return [list(range(r * per, (r + 1) * per)) for r in range(ranks)]


def fixed_primaries(load: torch.Tensor, ranks: int, s: int) -> list[list[int]]:
    """Kiln's incremental form: every expert stays where the contiguous placement puts it (rank r: experts
    r E / ranks ..), and only s extra slots per rank hold replicas, so a rebalance rewrites s slots per rank and
    layer and nothing else. Replicas: DeepSeek's replicate over E + ranks s slots; each new replica onto the rank
    with the least load (primaries' load shared by replicas) that has a free extra slot and no copy of it."""
    E = load.shape[0]
    cnt = replicate(load, E + ranks * s)
    slots = contiguous(E, ranks)
    per_rep = {e: float(load[e]) / int(cnt[e]) for e in range(E)}
    tot = [sum(per_rep[e] for e in sl) for sl in slots]
    extra = [0] * ranks
    reps = sorted(((per_rep[e], e) for e in range(E) for _ in range(int(cnt[e]) - 1)), reverse=True)
    for w, e in reps:
        cand = [r for r in range(ranks) if extra[r] < s and e not in slots[r]]
        if not cand:
            continue  # only ranks that already hold e have room: the slot goes to another expert below
        r = min(cand, key=lambda q: tot[q])
        slots[r].append(e)
        tot[r] += w
        extra[r] += 1
    for r in range(ranks):  # leftover extra slots: the heaviest expert per replica this rank does not hold
        while extra[r] < s:
            n = {e: sum(e in sl for sl in slots) for e in range(E)}
            e = max((q for q in range(E) if q not in slots[r]), key=lambda q: float(load[q]) / n[q])
            slots[r].append(e)
            extra[r] += 1
    return slots


def rank_slot_pairs(topi: torch.Tensor, slots: list[list[int]], E: int) -> list[list[int]]:
    """Pairs per (rank, slot) for one batch topi [T, k]: a pair (t, e) goes to replica t mod n_e of e, the
    replicas ordered by rank."""
    reps: dict[int, list[tuple[int, int]]] = {}
    for r, sl in enumerate(slots):
        for j, e in enumerate(sl):
            reps.setdefault(e, []).append((r, j))
    T, k = topi.shape
    out = [[0] * len(sl) for sl in slots]
    t_idx = torch.arange(T).unsqueeze(1).expand(T, k).flatten()
    e_idx = topi.long().flatten()
    n = torch.zeros(E, dtype=torch.long)
    for e, rl in reps.items():
        n[e] = len(rl)
    choice = t_idx % n[e_idx]
    counts = torch.zeros(E, int(n.max()), dtype=torch.long)
    counts.index_put_((e_idx, choice), torch.ones_like(e_idx), accumulate=True)
    for e, rl in reps.items():
        for i, (r, j) in enumerate(rl):
            out[r][j] = int(counts[e, i])
    return out


def prefill_passes(per_slot: list[int], lw: int = 256, lw2: int = 512) -> tuple[int, int]:
    """(static 256-lane passes, overflow 512-lane-equivalent passes) of one rank (2ca0773's rule)."""
    static = len(per_slot)
    over = 0.0
    for n in per_slot:
        rem = max(n - lw, 0)
        if rem:
            full, last = divmod(rem, lw2)
            over += full + (0.5 if 0 < last <= lw else (1 if last else 0))
    return static, over


def decode_passes(per_slot: list[int], lw: int = 16) -> int:
    return sum(1 + max(math.ceil(n / lw) - 1, 0) for n in per_slot)


def batches(topi_layer: list[torch.Tensor], sel: list[int], rows: int, group: int, shape: str, g: torch.Generator):
    T = topi_layer[sel[0]].shape[0]
    if shape == "prefill":
        G = min(group, rows)
        return [(sel[int(torch.randint(len(sel), (1,), generator=g))], int(torch.randint(T // G, (1,), generator=g)) * G, G)
                for _ in range(rows // G)]
    return [(sel[int(torch.randint(len(sel), (1,), generator=g))], int(torch.randint(T, (1,), generator=g)), 1)
            for _ in range(rows)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--kind", default="random", choices=("random", "text", "all"))
    ap.add_argument("--ranks", type=int, default=32)
    ap.add_argument("--slots", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--prefill", type=int, default=4096)
    ap.add_argument("--group", type=int, default=1024)
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--per-layer", action="store_true")
    ap.add_argument("--fixed", action="store_true", help="primaries stay contiguous, only the extra slots move")
    ap.add_argument("--lane-model", action="store_true", help="prefill time = 0.15 ms + 0.78 us per executed lane")
    ap.add_argument("--static-ms", type=float, default=0.245, help="one 256-lane static pass (tile-scale form)")
    ap.add_argument("--over-ms", type=float, default=0.40, help="one 512-lane overflow pass")
    a = ap.parse_args()
    d = torch.load(a.load)
    topi, names, E = d["topi"], d["names"], d["experts"]
    layers = sorted(topi)
    sel = [i for i, n in enumerate(names) if a.kind == "all" or n.startswith(a.kind)]
    half_a, half_b = sel[: len(sel) // 2], sel[len(sel) // 2:]
    R = a.ranks
    g = torch.Generator().manual_seed(0)
    print(f"{len(layers)} MoE layers, {E} experts, ranks {R}; kind {a.kind}: eval {[names[i] for i in half_a]}, "
          f"calibration {[names[i] for i in (half_a if a.oracle else half_b)]}")
    for shape, rows in (("prefill", a.prefill), ("decode", a.decode)):
        picks_all = [batches(topi[layers[0]], half_a, rows, a.group, shape, g) for _ in range(a.trials)]
        for s in a.slots:
            tot_ms, tot_pairs_ratio, per_layer = 0.0, [], []
            for li, l in enumerate(layers):
                calset = half_a if a.oracle else half_b
                load = torch.bincount(torch.cat([topi[l][i].long().flatten() for i in calset]), minlength=E).float()
                if s == 0:
                    slots = contiguous(E, R)
                elif a.fixed:
                    slots = fixed_primaries(load + 1e-3, R, s)
                else:
                    cnt = replicate(load + 1e-3, E + R * s)
                    slots = pack(load + 1e-3, cnt, R)
                worst_ms, ratio = [], []
                for picks in picks_all:
                    bt = torch.cat([topi[l][i][o:o + G] for i, o, G in picks])
                    rs = rank_slot_pairs(bt, slots, E)
                    pairs = torch.tensor([sum(x) for x in rs], dtype=torch.float)
                    ratio.append(float(pairs.max() / pairs.mean()))
                    if shape == "prefill" and a.lane_model:
                        # The MoE-kernel agent's fit (kiln-mk-k1, 2026-10-05): 0.15 ms + 0.78 us per executed lane,
                        # lanes = 256 per local slot plus the 512-lane overflow passes (a half pass = 256 lanes).
                        ms = max(0.15 + 0.78e-3 * (256 * st + 512 * ov) for st, ov in (prefill_passes(x) for x in rs))
                    elif shape == "prefill":
                        ms = max(st * a.static_ms + ov * a.over_ms for st, ov in (prefill_passes(x) for x in rs))
                    else:
                        ms = max(decode_passes(x) for x in rs) * 0.140
                    worst_ms.append(ms)
                m = sum(worst_ms) / len(worst_ms)
                tot_ms += m
                tot_pairs_ratio.append(sum(ratio) / len(ratio))
                per_layer.append((l, m, sum(ratio) / len(ratio)))
            label = "contiguous" if s == 0 else f"EPLB +{s} slot{'s' if s > 1 else ''}/rank"
            print(f"  {shape:7s} {rows:5d} rows  {label:22s} busiest/mean pairs {sum(tot_pairs_ratio) / len(tot_pairs_ratio):.2f}"
                  f"  modeled busiest-rank kernel, sum over layers {tot_ms:7.1f} ms")
            if a.per_layer:
                print("    " + " ".join(f"L{l}:{m:.2f}/{r:.1f}x" for l, m, r in per_layer))


if __name__ == "__main__" and not (len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "budget"):
    main()


def budget_main() -> None:
    """--budget N: N redundant slots per rank shared over the layers (a layer takes 0-3), placed greedily by the
    modelled busiest-rank gain of one more slot (lane model), copies placed from the calibration half with
    fixed primaries; reports the sum over layers against one slot on every layer."""
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--kind", default="random")
    ap.add_argument("--budget", type=int, default=42)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--ranks", type=int, default=32)
    a = ap.parse_args(sys.argv[2:])
    d = torch.load(a.load)
    topi, names, E = d["topi"], d["names"], d["experts"]
    layers = sorted(topi)
    sel = [i for i, n in enumerate(names) if a.kind == "all" or n.startswith(a.kind)]
    half_a, half_b = sel[: len(sel) // 2], sel[len(sel) // 2:]
    R = a.ranks
    g = torch.Generator().manual_seed(0)
    picks_all = [batches(topi[layers[0]], half_a, 4096, 1024, "prefill", g) for _ in range(a.trials)]

    def cost(l, s):
        load = torch.bincount(torch.cat([topi[l][i].long().flatten() for i in half_b]), minlength=E).float()
        slots = contiguous(E, R) if s == 0 else fixed_primaries(load + 1e-3, R, s)
        tot = 0.0
        for picks in picks_all:
            bt = torch.cat([topi[l][i][o:o + G] for i, o, G in picks])
            rs = rank_slot_pairs(bt, slots, E)
            tot += max(0.15 + 0.78e-3 * (256 * st + 512 * ov) for st, ov in (prefill_passes(x) for x in rs))
        return tot / len(picks_all)

    table = {l: [cost(l, s) for s in range(4)] for l in layers}
    s_l = {l: 0 for l in layers}
    for _ in range(a.budget):
        l = max(layers, key=lambda q: (table[q][s_l[q]] - table[q][s_l[q] + 1]) if s_l[q] < 3 else -1e9)
        s_l[l] += 1
    uniform = sum(table[l][1] for l in layers)
    var = sum(table[l][s_l[l]] for l in layers)
    print(f"{len(layers)} layers: contiguous {sum(table[l][0] for l in layers):.1f} ms, one slot everywhere "
          f"{uniform:.1f} ms, {a.budget} slots placed by gain {var:.1f} ms; slots per layer "
          + " ".join(f"L{l}:{s_l[l]}" for l in layers), flush=True)


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "budget":
    budget_main()
