"""Expert-parallel balance over a long document, from a KILN_EPLB_TRACE file (one all-reduced expert-count record per
prefill call, ModelRunner.eplb_trace): how stable the document's hot experts are from chunk to chunk, and what the
busiest rank's EP-kernel time per call would be under each placement.

    python tools/eplb_trace_sim.py --trace counts.pt --chunks 63 --docs 5 --eval 0 1 2 --other 3 4 \\
        [--ranks 32] [--stages 0,12,24,36,45] [--window 8] [--every 8]

The calls are split into --docs documents of --chunks calls each, in order (tools/lc_ttft.py --text-file runs each
request as the next disjoint window of the text). Placements, each with s redundant slots per rank in Kiln's form
(tools/eplb_sim.py fixed_primaries: every expert stays at its contiguous rank, only the s extra slots per rank move):
- contiguous: s = 0, what serves today;
- static: one placement from the --other documents' summed counts (a long document the engine saw before);
- online: the placement for chunk t from the same document's chunks t - window .. t - 1, rebuilt every --every chunks
  (KILN_EPLB_INTERVAL; the document's first window runs the static placement);
- oracle: from chunk t's own counts (the bound no causal policy reaches).
A replicated expert's pairs split evenly over its copies (the served dispatch, row t to copy t mod n, splits within one
pair). Time per rank and layer is tools/eplb_sim.py's pass model, measured on the tile-scale EP kernel (one 256-lane
static pass per slot, 0.245 ms, and 512-lane overflow passes, 0.40 ms); a call waits for the busiest rank of every
layer, so the per-call figure is the sum over layers of the busiest rank's time, reported per pipeline stage.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eplb_sim import contiguous, fixed_primaries, prefill_passes  # noqa: E402


def rank_times(count: torch.Tensor, slots: list[list[int]], static_ms: float, over_ms: float) -> list[float]:
    n = torch.zeros(count.shape[0])
    for sl in slots:
        for e in sl:
            n[e] += 1
    per = count.double() / n.clamp(min=1).double()
    out = []
    for sl in slots:
        st, ov = prefill_passes([int(round(float(per[e]))) for e in sl])
        out.append(st * static_ms + ov * over_ms)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--chunks", type=int, required=True)
    ap.add_argument("--docs", type=int, required=True)
    ap.add_argument("--eval", type=int, nargs="+", required=True)
    ap.add_argument("--other", type=int, nargs="+", required=True)
    ap.add_argument("--ranks", type=int, default=32)
    ap.add_argument("--stages", default="0,12,24,36,45", help="layer boundaries of the pipeline stages")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--slots", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--static-ms", type=float, default=0.245)
    ap.add_argument("--over-ms", type=float, default=0.40)
    a = ap.parse_args()
    tr = torch.load(a.trace)
    counts, layers = tr["counts"].long(), tr["layers"]  # [calls, L, E]
    C, L, E = counts.shape
    need = a.chunks * a.docs
    if C < need:
        raise SystemExit(f"{C} traced calls, {a.docs} documents of {a.chunks} need {need}")
    doc = lambda d: counts[d * a.chunks:(d + 1) * a.chunks]  # noqa: E731
    bounds = [int(x) for x in a.stages.split(",")]
    stage_of = [next(i for i in range(len(bounds) - 1) if bounds[i] <= l < bounds[i + 1]) for l in layers]
    S = len(bounds) - 1
    print(f"{C} calls x {L} MoE layers x {E} experts; documents of {a.chunks} calls; eval {a.eval}, other {a.other}; "
          f"stages {bounds}")

    # Stability: per layer, the cosine between consecutive chunks' count vectors and between a chunk and its document's
    # mean; the share of a chunk's pairs on the previous window's 32 hottest experts.
    cos_next, cos_doc, hot_cov = [], [], []
    for d in a.eval:
        x = doc(d).double()
        for t in range(1, a.chunks):
            cos_next.append(torch.nn.functional.cosine_similarity(x[t], x[t - 1], dim=-1).mean())
        m = x.mean(0)
        for t in range(a.chunks):
            cos_doc.append(torch.nn.functional.cosine_similarity(x[t], m, dim=-1).mean())
        for t in range(a.window, a.chunks):
            prev = x[t - a.window:t].sum(0)
            top = prev.topk(32, dim=-1).indices
            hot_cov.append((torch.gather(x[t], -1, top).sum(-1) / x[t].sum(-1)).mean())
    stat = lambda v: f"mean {float(torch.stack(v).mean()):.3f} min {float(torch.stack(v).min()):.3f}"  # noqa: E731
    print(f"stability: cosine(chunk t, chunk t-1) {stat(cos_next)}; cosine(chunk, document mean) {stat(cos_doc)}; "
          f"share of a chunk's pairs on the previous {a.window} chunks' 32 hottest experts {stat(hot_cov)} "
          f"(uniform: {32 / E:.3f})")

    other = torch.stack([doc(d).sum(0) for d in a.other]).sum(0)  # [L, E]
    base = [contiguous(E, a.ranks) for _ in range(L)]

    def run(policy: str, s: int) -> torch.Tensor:
        """[S] mean busiest-rank ms per call per stage over the eval documents' chunks."""
        tot = torch.zeros(S)
        n = 0
        static = [fixed_primaries(other[l].double(), a.ranks, s) for l in range(L)] if s else base
        for d in a.eval:
            x = doc(d)
            place = static
            for t in range(a.chunks):
                if policy == "online" and t >= a.window and (t - a.window) % a.every == 0:
                    w = x[t - a.window:t].sum(0)
                    place = [fixed_primaries(w[l].double(), a.ranks, s) for l in range(L)]
                if policy == "oracle":
                    place = [fixed_primaries(x[t, l].double(), a.ranks, s) for l in range(L)]
                for l in range(L):
                    tot[stage_of[l]] += max(rank_times(x[t, l], place[l], a.static_ms, a.over_ms))
                n += 1
        return tot / n

    rows = [("contiguous (today)", run("contiguous", 0))]
    for s in a.slots:
        rows += [(f"static +{s} (from other documents)", run("static", s)),
                 (f"online +{s} (window {a.window}, every {a.every})", run("online", s)),
                 (f"oracle +{s}", run("oracle", s))]
    ref = rows[0][1]
    print("busiest-rank EP-kernel ms per 4096-row call, by stage (pass model), and the saving against contiguous:")
    print("  " + f"{'placement':42s}" + "".join(f"  stage {i}" for i in range(S)) + "    total   saving")
    for name, v in rows:
        print(f"  {name:42s}" + "".join(f" {float(x):8.1f}" for x in v) + f" {float(v.sum()):8.1f} {float(ref.sum() - v.sum()):8.1f}")


if __name__ == "__main__":
    main()
