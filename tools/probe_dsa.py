"""Which DSA top-k formulations neuronx-cc compiles, and what each costs, on one NeuronCore.

    python tools/probe_dsa.py [--rows 8] [--keys 128] [--k 16] [--heads 8] [--dim 128]

Each case is compiled alone as a graph over [rows, 1, keys] index scores (decode: one query per
row) and checked against the same function on the host; it prints compile status, the max
deviation and the p50 time of synchronous calls.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--keys", type=int, default=128)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--cases", nargs="*", default=None)
    args = ap.parse_args()
    import profile_layer as pl

    pl.setup_device()
    R, L, k, Hi, Di = args.rows, args.keys, args.k, args.heads, args.dim
    g = torch.Generator().manual_seed(0)
    q = torch.randn(R, 1, Hi, Di, generator=g)
    w = torch.randn(R, 1, Hi, generator=g)
    kI = torch.randn(R, L, Di, generator=g)
    vis = torch.where(torch.arange(L) < L - 3, 0.0, -1e30).expand(R, 1, L).contiguous()
    sc = torch.randn(R, 1, L, generator=g)
    top = torch.topk(sc, k, dim=-1).indices
    kc = torch.randn(R, L, 512, generator=g)

    def score(q, w, kI, vis):
        s = torch.relu(torch.einsum("bqhd,bld->bqhl", q, kI) * Di ** -0.5)
        return torch.einsum("bqh,bqhl->bql", w, s) + vis

    def topk_idx(s):
        return torch.topk(s, k, dim=-1).indices

    def topk_vals(s):
        return torch.topk(s, k, dim=-1).values

    def kth_threshold_mask(s):  # no indices: keep scores >= the k-th largest
        kth = torch.topk(s, k, dim=-1).values[..., -1:]
        return torch.where(s >= kth, 0.0, -1e30)

    def scatter_mask(s, t):
        return torch.full_like(s, -1e30).scatter(-1, t, 0.0)

    def onehot_mask(s, t):  # compare against every key: [R, 1, k, L]
        hit = (t.unsqueeze(-1) == torch.arange(s.shape[-1], device=s.device)).any(dim=-2)
        return torch.where(hit, 0.0, -1e30)

    def gather_rows(kc, t):
        B = kc.shape[0]
        flat = (t + (torch.arange(B, device=t.device) * L).view(B, 1, 1)).reshape(-1)
        return kc.reshape(B * L, -1)[flat]

    def full_select(q, w, kI, vis):
        s = score(q, w, kI, vis)
        t = torch.topk(s, k, dim=-1).indices
        return vis + torch.full_like(vis, -1e30).scatter(-1, t, 0.0)

    def full_threshold(q, w, kI, vis):
        s = score(q, w, kI, vis)
        kth = torch.topk(s, k, dim=-1).values[..., -1:]
        return vis + torch.where(s >= kth, 0.0, -1e30)

    cases = {
        "score": (score, (q, w, kI, vis)),
        "topk indices": (topk_idx, (sc,)),
        "topk values": (topk_vals, (sc,)),
        "kth threshold mask": (kth_threshold_mask, (sc,)),
        "scatter mask": (scatter_mask, (sc, top)),
        "one-hot mask": (onehot_mask, (sc, top)),
        "gather selected latents": (gather_rows, (kc, top)),
        "score + topk + scatter": (full_select, (q, w, kI, vis)),
        "score + threshold": (full_threshold, (q, w, kI, vis)),
    }
    for name, (fn, a) in cases.items():
        if args.cases and name not in args.cases:
            continue
        want = fn(*a)
        c = torch.compile(fn, **pl.OPTS)
        dev = tuple(x.to(pl.DEV) for x in a)
        t = time.perf_counter()
        try:
            got = c(*dev).cpu()
        except Exception as e:
            print(f"{name:<28} FAILED after {time.perf_counter() - t:.1f}s: {type(e).__name__}: {str(e)[:160]}",
                  flush=True)
            continue
        comp = time.perf_counter() - t
        if got.dtype.is_floating_point:
            err = (got.float().clamp(-1e4, 1e4) - want.float().clamp(-1e4, 1e4)).abs().max().item()
        else:  # index sets: compare as sets per row
            err = float(sum(set(a.tolist()) != set(b.tolist())
                            for a, b in zip(got.reshape(-1, got.shape[-1]), want.reshape(-1, want.shape[-1]))))
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            c(*dev).cpu()
            ts.append(time.perf_counter() - t)
        ts.sort()
        print(f"{name:<28} ok   compile {comp:6.1f}s  err {err:.3g}  p50 {ts[10] * 1e3:.3f} ms", flush=True)


if __name__ == "__main__" and "--layer" not in sys.argv and "--routing" not in sys.argv:
    main()


def layer_variants() -> None:
    """--layer: the real MLA attention block of a DSA layer (GLM-5.3-0.6B shapes by default)
    with one piece swapped at a time, to find what neuronx-cc rejects."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", action="store_true")
    ap.add_argument("--model", default="inference-optimization/GLM-5.3-0.6B-A0.4B")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pages", type=int, default=4)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--variant", default="full")
    args = ap.parse_args()
    import dataclasses

    import profile_layer as pl

    import kiln.models.mla as mla
    from kiln.config import ModelConfig

    pl.setup_device()
    orig = ModelConfig.from_pretrained

    def patched(path):
        cfg = orig(path)
        return dataclasses.replace(cfg, attn_layers=tuple(dataclasses.replace(s, mla=dataclasses.replace(
            s.mla, dsa=dataclasses.replace(s.mla.dsa, topk=args.topk))) for s in cfg.attn_layers))

    ModelConfig.from_pretrained = staticmethod(patched)
    B, P, ps = args.batch, args.pages, 32
    cfg, model = pl.build(args.model, 1, 1, 1 + B * P, ps, P * ps + 1)
    v = args.variant

    def as_select(topk_fn):  # a variant's indices as the mask models/mla.py's _select returns
        return lambda d, q, w, kI, vis: torch.full_like(vis, -1e30).scatter(-1, topk_fn(d, q, w, kI, vis), 0.0)

    if v == "no-share":
        mla._share = lambda layer, top, T: None
    elif v == "int64-share":
        mla._share = lambda layer, top, T: layer.dsa_topk.index_put_(
            (torch.arange(T, device=top.device),), top.reshape(T, -1).to(layer.dsa_topk.dtype))
    elif v == "no-topk":  # selection from a fixed pattern instead of the index scores
        mla._select = as_select(lambda d, q, w, kI, vis: torch.zeros(*vis.shape[:2], d.topk, dtype=torch.int64,
                                                          device=vis.device) + torch.arange(d.topk, device=vis.device))
    elif v == "topk-of-vis":  # topk over the visibility alone (no index scores)
        mla._select = as_select(lambda d, q, w, kI, vis: torch.topk(vis, d.topk, dim=-1).indices)
    elif v == "score-only":  # index scores computed, selection fixed
        def f(d, q, w, kI, vis):
            s = torch.relu(torch.einsum("bqhd,bld->bqhl", q.float(), kI.float()) * d.head_dim ** -0.5)
            idx = torch.einsum("bqh,bqhl->bql", w, s) + vis
            return (torch.zeros(*vis.shape[:2], d.topk, device=vis.device) + idx[..., : d.topk] * 0).long() \
                + torch.arange(d.topk, device=vis.device)
        mla._select = as_select(f)
    import torch.nn.functional as F

    def indexer_variant(model, layer, x, q_resid, positions, rope_q=True, rope_k=True, flat=False, fp32=False):
        d, dr = layer.spec.mla.dsa, layer.spec.mla.qk_rope_head_dim
        T = x.shape[0]
        wq = model._w(layer, "idx_wq")
        qf = F.linear(q_resid.float(), wq.float()) if fp32 else F.linear(q_resid, wq)
        q = qf.view(T, d.n_heads, d.head_dim)
        k = F.linear(x, model._w(layer, "idx_wk"))
        k = F.layer_norm(k.float(), (d.head_dim,), layer.idx_knorm_w.float(), layer.idx_knorm_b.float(),
                         1e-6).to(x.dtype)
        if rope_q:
            q = mla._rope(model, layer, q, positions, dr, True)
        if rope_k:
            k = mla._rope(model, layer, k, positions, dr, False)
        w = F.linear(x.float(), layer.idx_wproj) * d.n_heads ** -0.5
        return q, k, w

    if v == "no-idx-rope-q":
        mla._indexer = lambda *a: indexer_variant(*a, rope_q=False)
    elif v == "no-idx-rope-k":
        mla._indexer = lambda *a: indexer_variant(*a, rope_k=False)
    elif v == "no-idx-rope":
        mla._indexer = lambda *a: indexer_variant(*a, rope_q=False, rope_k=False)
    elif v == "idx-fp32":
        mla._indexer = lambda *a: indexer_variant(*a, fp32=True)
    elif v == "flat-score":
        def f(d, q, w, kI, vis):
            B, Q = vis.shape[:2]
            s = torch.relu(torch.einsum("bnd,bld->bnl", q.reshape(B, Q * d.n_heads, d.head_dim).float(),
                                        kI.float()) * d.head_dim ** -0.5)
            idx = torch.einsum("bqh,bqhl->bql", w, s.view(B, Q, d.n_heads, -1)) + vis
            return torch.topk(idx, d.topk, dim=-1).indices
        mla._select = as_select(f)
    inp = pl.decode_inputs(model, B, P, ps)
    h = torch.randn(B, cfg.hidden_size).to(torch.bfloat16).to(pl.DEV)
    f, ts = pl.with_layer(model, 0, lambda V, h, p, s, tb, bs: model._attention(V, h, p, s, tb, bs))
    pl.timed(f"variant {v}", f, (h, inp["positions"], inp["slots"], inp["table"], inp["bias"], *ts))


if __name__ == "__main__" and "--layer" in sys.argv:
    layer_variants()


def routing_variants() -> None:
    """--routing: DeepSeek-V3 group-limited expert choice (models/mla.py group_limited) and
    its pieces, compiled alone and checked against the host."""
    import profile_layer as pl

    import kiln.models.mla as mla

    pl.setup_device()
    T, G, per, kg = 8, 8, 2, 4
    g = torch.Generator().manual_seed(0)
    x = torch.rand(T, G * per, generator=g)

    def best(x):
        return x.view(T, G, per).topk(2, dim=-1).values.sum(dim=-1)

    def best_max(x):  # top-2 sum of a 2-wide group without topk
        v = x.view(T, G, per)
        return v.amax(dim=-1) + (v.sum(dim=-1) - v.amax(dim=-1) if per == 2 else 0)

    def gidx(x):
        return torch.topk(best(x), kg, dim=-1).indices

    def keep_any(x):
        gi = torch.topk(best(x), kg, dim=-1).indices
        return (gi.unsqueeze(-1) == torch.arange(G, device=x.device)).any(dim=1).float()

    def keep_sum(x):
        gi = torch.topk(best(x), kg, dim=-1).indices
        return (gi.unsqueeze(-1) == torch.arange(G, device=x.device)).float().sum(dim=1)

    def keep_threshold(x):
        b = best(x)
        return (b >= torch.topk(b, kg, dim=-1).values[..., -1:]).float()

    cases = {"group best (topk 2)": best, "group best (max)": best_max, "group topk indices": gidx,
             "keep via any": keep_any, "keep via sum": keep_sum, "keep via threshold": keep_threshold,
             "group_limited": lambda x: mla.group_limited(x, G, kg),
             "group topk from max": lambda x: torch.topk(best_max(x), kg, dim=-1).indices}
    for name, fn in cases.items():
        want = fn(x)
        c = torch.compile(fn, **pl.OPTS)
        try:
            got = c(x.to(pl.DEV)).cpu()
        except Exception as e:
            print(f"{name:<24} FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
            continue
        if got.dtype.is_floating_point:
            err = (got.float().clamp(-1e4, 1e4) - want.float().clamp(-1e4, 1e4)).abs().max().item()
        else:
            err = float(sum(set(a.tolist()) != set(b.tolist()) for a, b in zip(got, want)))
        print(f"{name:<24} err {err:.3g}", flush=True)


if __name__ == "__main__" and "--routing" in sys.argv:
    routing_variants()
