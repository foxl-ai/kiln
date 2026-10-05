"""Device time and exactness of DSA's top-k selection as an attention mask, one NeuronCore.

    python tools/probe_dsa_select.py [--rows 8 128] [--keys 4096 8192 16384] [--keep 2048]
    python tools/probe_dsa_select.py --keys 1024 2048 4096 --keep 512     # GLM-5.3-Flash pools

Each case is one compiled graph from fp32 index scores [rows, keys] to the additive mask the
attention adds (0 = selected, -1e30 = not), as models/mla.py builds it:

- "topk + scatter": torch.topk indices scattered into a mask (the KILN_DSA_SELECT=topk path),
- "topk alone": just the indices (what the gather path consumes),
- "bisect-block" / "bisect-cumsum" / "bisect-index": models/dsa_select.py's default (the
  float-order bisection) with each tie fill (KILN_DSA_TIES), "kth-value": its threshold search
  alone, "kth-value-flat": the same without cutting rows into pieces for the counts,
- "radix-cumsum" / "radix-index", "order-key", "kth-key": the bitcast form and its pieces (LNL
  rejected the in-graph bitcast on trn1, SDK 2.32; a failed compile ends the process, so run these
  last or alone).

- "nki": kernels/dsa_topk.py (KILN_DSA_SELECT=nki) with vis_only (GLM-5.3-Flash's pooled form),
  "nki-all" without it (dsa_select's form), "nki-flat" with one piece per row (no cross-partition
  counts); "qsa": glm5_next.block_mask's range bisection over the same pools ("range", QSA's
  default and GLM-5.3-Flash's until 2026-10-04), its 0 / NEG_INF pool mask.

Every mask is read back and compared with the host: the radix masks must EQUAL
dsa_select.reference_mask (a stable sort: descending score, lowest index among ties), the topk
mask must hold `keep` positions whose scores are the host's top-k values; "nki" and "qsa" must equal
the reference among the visible candidates, "nki" must also equal dsa_topk.emulate bit for bit.
Score kinds: "randn" (no ties), "ties" (integers in [-3, 3]: thousands of positions tied at the
threshold), "pooled" (a pooled indexer's sum_h w_h relu(q_h . k): both signs, exact zeros),
"short" (pooled, only the first 300 candidates visible), "zeros" / "wide" / "ulps" / "equal"
(tests/test_dsa_select.py's), "real:<file>" (rows of a torch.save'd [rows, keys] fp32 score tensor,
NEG_INF + score for invisible candidates; tools/dump_dsa_scores.py writes them).
Timings are p50 of synchronous calls (tools/profile_layer.timed).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NEG_INF = -1e30


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[8, 128])
    ap.add_argument("--keys", type=int, nargs="+", default=[4096, 8192, 16384])
    ap.add_argument("--keep", type=int, default=2048)
    ap.add_argument("--kinds", nargs="+", default=["randn", "ties"])
    ap.add_argument("--methods", nargs="+", default=["topk+scatter", "topk", "bisect-block", "bisect-index",
                                                     "kth-value", "kth-value-flat"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--scores", action="store_true",
                    help="time the selection after the lightning indexer's scores in one graph (decode: rows = batch, "
                         "one query each; GLM-5.3 indexer: 32 heads x 128)")
    ap.add_argument("--gather", action="store_true",
                    help="with --scores: the indexer keys gathered from a paged cache by a block table, as the layer "
                         "reads them (32-token pages), instead of a graph input")
    ap.add_argument("--write", action="store_true",
                    help="with --gather: write each row's new key into the cache first (index_put_), as the layer does")
    ap.add_argument("--stage", nargs="*", default=["none"],
                    help="with --scores: how the scores reach the selection: none, or scratch (written to and read back "
                         "from a [rows, keys] buffer)")
    ap.add_argument("--fused", action="store_true",
                    help="the score + selection kernel (dsa_topk.score_select) on --rows queries of one sequence over "
                         "--keys pools of 128-dim keys, 32 heads, against emulate_scores + emulate on the host")
    ap.add_argument("--cpu", action="store_true", help="dry run on the host")
    args = ap.parse_args()
    import profile_layer as pl

    from kiln.models import dsa_select as ds

    pl.setup_device(args.cpu)
    k = args.keep

    def add(sel):
        return torch.where(sel, 0.0, NEG_INF)

    from kiln.kernels import dsa_topk as dk
    from kiln.models.glm5_next import block_mask

    def qsa(index, vis):  # glm5_next.block_mask's default selection over pools of one key, no tail
        R, L_ = index.shape
        return block_mask(index.view(R, 1, L_), vis.view(R, 1, L_), 1, k, False, select="range").view(R, L_)

    fns = {
        "floor": lambda s: s * 0.5,  # a graph that reads and writes the same tensor (launch floor)
        "nki": lambda s: dk.select(s, k),
        "nki-all": lambda s: dk.select(s, k, vis_only=False),
        "nki-flat": lambda s: dk.select(s, k, group=False),
        "qsa": qsa,
        "topk+scatter": lambda s: torch.full_like(s, NEG_INF).scatter(-1, torch.topk(s, k, dim=-1).indices, 0.0),
        "topk": lambda s: torch.topk(s, k, dim=-1).indices,
        "bisect-cumsum": lambda s: add(ds.topk_mask(s, k, "bisect", "cumsum")),
        "bisect-index": lambda s: add(ds.topk_mask(s, k, "bisect", "index")),
        "bisect-block": lambda s: add(ds.topk_mask(s, k, "bisect", "block")),
        "kth-value": lambda s: ds.kth_value(s, k)[0],
        "kth-value-flat": lambda s: ds.kth_value(s, k, group=False)[0],
        "radix-cumsum": lambda s: add(ds.topk_mask(s, k, "radix", "cumsum")),
        "radix-index": lambda s: add(ds.topk_mask(s, k, "radix", "index")),
        "order-key": lambda s: ds.order_key(s),
        "kth-key": lambda s: ds.kth_key(ds.order_key(s), k),
    }
    if args.scores:  # the index scores computed in the same graph, as a DSA layer does (decode form)
        score_cases(args, pl, ds)
        return
    if args.fused:  # kernels/dsa_topk.py's score + selection kernel (a prefill chunk's pooled indexer)
        fused_cases(args, pl)
        return
    for rows in args.rows:
        for L in args.keys:
            for kind in args.kinds:
                g = torch.Generator().manual_seed(rows * 7 + L)
                index, vis = make_scores(kind, rows, L, g)
                rows_, L_ = index.shape
                s = index + vis
                want = ds.reference_mask(s, k)
                want_vis = want & (vis == 0)
                host_top = torch.topk(s, k, dim=-1).values.sort(-1).values
                pl.say(f"rows={rows_} keys={L_} keep={k} scores={kind}:", flush=True)
                dev = s.to(pl.DEV)
                dev_iv = (index.to(pl.DEV), vis.to(pl.DEV))
                for m in args.methods:
                    if m not in fns:
                        continue
                    a = dev_iv if m == "qsa" else (dev,)
                    t = pl.timed(m, fns[m], a, args.iters)
                    if t != t:  # failed to compile or run
                        continue
                    got = torch.compile(fns[m], **pl.OPTS)(*a).cpu()
                    if m == "floor":
                        continue
                    if m in ("nki", "nki-flat", "nki-all", "qsa"):
                        w_ = want if m == "nki-all" else want_vis
                        sel = got == 0
                        bad = (sel != w_).any(-1)
                        msg = f"    exact (equals the tie rule): {not bad.any().item()} ({int(bad.sum())} of {rows_} rows differ)"
                        if m != "qsa":
                            em = dk.emulate(s, k, vis_only=m != "nki-all")
                            msg += f"; equals emulate bit for bit: {torch.equal(got, em)}"
                        else:
                            msg += f"; selected per row {sorted(set(sel.sum(-1).tolist()))[:6]}"
                        pl.say(msg, flush=True)
                    elif m in ("radix-cumsum", "radix-index", "bisect-cumsum", "bisect-index", "bisect-block"):
                        ok = torch.equal(got == 0, want)
                        pl.say(f"    exact (equals the tie rule): {ok}; selected per row "
                               f"{sorted(set((got == 0).sum(-1).tolist()))}", flush=True)
                    elif m == "topk+scatter":
                        sel = got == 0
                        vals = torch.where(sel, s, torch.full_like(s, float("inf"))).sort(-1).values[:, :k]
                        pl.say(f"    {sel.sum(-1).min().item()}..{sel.sum(-1).max().item()} selected; top-k values "
                               f"equal the host's: {torch.equal(vals, host_top)}; same set as the tie rule: "
                               f"{torch.equal(sel, want)}", flush=True)
                    elif m == "order-key":
                        pl.say(f"    equals the host: {torch.equal(got, ds.order_key(s))}", flush=True)
                    elif m.startswith("kth-value"):
                        pl.say(f"    equals the host: {torch.equal(got, ds.kth_value(s, k)[0])}", flush=True)
                    elif m == "kth-key":
                        pl.say(f"    equals the host: {torch.equal(got, ds.kth_key(ds.order_key(s), k))}", flush=True)



def make_scores(kind: str, rows: int, L: int, g: torch.Generator):
    """(index [rows, L] finite, vis [rows, L] 0 / NEG_INF) for a score kind (module docstring)."""
    vis = torch.zeros(rows, L)
    if kind == "randn":
        return torch.randn(rows, L, generator=g), vis
    if kind == "ties":
        return torch.randint(-3, 4, (rows, L), generator=g).float(), vis
    if kind in ("pooled", "short"):
        from tests.test_dsa_topk import pooled

        index = pooled(rows, L, g, zeros=True)
        if kind == "short":
            vis[:, 300:] = NEG_INF
        return index, vis
    if kind.startswith("real:"):
        s = torch.load(kind[5:]).float()
        s = s.reshape(-1, s.shape[-1])
        inv = s < -5e29
        return torch.where(inv, s - NEG_INF, s), torch.where(inv, NEG_INF, 0.0)
    from tests.test_dsa_select import _scores

    s = _scores(kind, rows, L, g)
    return torch.where(s.abs() < 1.2e-38, torch.zeros_like(s), s), vis  # no subnormals (FTZ on the device)


def fused_cases(args, pl) -> None:
    """dsa_topk.score_select on the device against emulate_scores + emulate on the host (pool level and
    expanded to 4 tokens with the tail), and the same scores computed by torch in the graph followed
    by the selection kernel, for time."""
    from kiln.kernels import dsa_topk as dk

    k, scale = args.keep, 128 ** -0.5
    for C in args.rows:
        for P in args.keys:
            for vis in (None, P // 3):
                g = torch.Generator().manual_seed(C + P)
                q = torch.randn(C, 32, 128, generator=g).to(torch.bfloat16)
                pk = torch.randn(P, 128, generator=g).to(torch.bfloat16)
                w = torch.randn(C, 32, generator=g) * 32 ** -0.5
                cand = torch.zeros(C, P) if vis is None else torch.where(torch.arange(P) < vis, 0.0, NEG_INF).expand(C, P)
                cand = cand.contiguous()
                dev = tuple(x.to(pl.DEV) for x in (q, w, pk, cand))
                pl.say(f"fused: C={C} pools={P} keep={k} visible={vis or P}:", flush=True)
                for kp, tail in ((1, False), (4, True)):
                    f = lambda q_, w_, pk_, c_, kp=kp, tail=tail: dk.score_select(q_, w_, pk_, c_, k, scale, kp, tail)  # noqa: E731
                    pl.timed(f"score kernel kp={kp} tail={tail}", f, dev, args.iters)
                    got = torch.compile(f, **pl.OPTS)(*dev).cpu()
                    want = dk.emulate(dk.emulate_scores(q, w, pk, cand, scale), k, True, kp, tail)
                    pl.say(f"    equals emulate: {torch.equal(got, want)} ({int((got != want).any(-1).sum())} rows differ)",
                           flush=True)

                def two(q_, w_, pk_, c_):  # torch scores in the graph, then the selection kernel
                    s_ = torch.einsum("chd,pd->chp", q_.float(), pk_.float())
                    idx = torch.einsum("ch,chp->cp", w_, torch.relu(s_ * scale)) + c_
                    return dk.select(idx, k, kp=4, tail=True)

                pl.timed("torch scores + selection kernel kp=4 tail", two, dev, args.iters)


def score_cases(args, pl, ds) -> None:
    """Selection after the scores, in one graph, as mla._scores computes them (optionally from keys
    gathered out of a paged cache that the graph writes first), through torch.topk and through the
    bisection; --stage scratch writes the scores to a buffer and reads them back first."""
    from kiln.models import mla
    from kiln.models.mla import DSASpec

    d = DSASpec(32, 128, args.keep, True)
    k = args.keep
    for rows in args.rows:
        for L in args.keys:
            g = torch.Generator().manual_seed(L)
            q = (torch.randn(rows, 1, 32, 64, generator=g), torch.randn(rows, 1, 32, 64, generator=g))
            w = torch.randn(rows, 1, 32, generator=g)
            ps, P = 32, L // 32
            if args.gather:  # [slots, 1, 192] pages (rope key + indexer key), each row its own P pages
                kI = torch.randn((1 + rows * P) * ps, 1, 192, generator=g).to(torch.bfloat16)
                table = torch.arange(1, 1 + rows * P).view(rows, P)
            else:
                kI, table = torch.randn(rows, L, 128, generator=g), torch.zeros(1)
            vis = torch.zeros(rows, 1, L)
            scratch = torch.zeros(rows, L)
            dev = [x.to(pl.DEV) for x in (q[0], q[1], w, kI, vis, table, scratch)]
            pl.say(f"scores in the graph: rows={rows} keys={L} keep={k} gather={args.gather}:", flush=True)
            for form in ["einsum"]:
                for stage in args.stage:
                    for sel in ("topk", "bisect"):
                        def f(qr, qp, w, kI, vis, table, scratch, sel=sel, stage=stage):
                            if args.gather and args.write:  # each row's newest token: the last slot of its last page
                                slots = table[:, -1] * ps + ps - 1
                                kI.index_put_((slots,), (qp[:, 0, :1, :].repeat(1, 1, 3) * 0.5).to(kI.dtype))
                            if args.gather:
                                kI = kI.view(-1, ps, 1, 192)[table].flatten(-4, -3).squeeze(-2)[..., 64:].float()
                            index = mla._scores(d, (qr, qp), w, kI, vis)
                            if stage == "scratch":
                                rows_ = torch.arange(index.shape[0], device=index.device)
                                scratch.index_put_((rows_,), index.reshape(index.shape[0], -1))
                                index = scratch[rows_].view(index.shape)
                            return torch.where(ds.topk_mask(index, k, sel), 0.0, NEG_INF)

                        pl.timed(f"{form} scores, stage {stage} + {sel}", f, tuple(dev), args.iters)


if __name__ == "__main__":
    main()
