"""Hyper-connection hybrids' pieces on ONE NeuronCore: device time of a layer's token mixer in the
decode form (B sequences x 1 token over P pages), whole and with parts removed, against the same
code on the CPU, to localise a slow or wrong lowering.

    python tools/probe_hybrid.py qsa <checkpoint> [--batch 8] [--pages 4] [--page-size 32]
    python tools/probe_hybrid.py dsa <checkpoint> ...

`qsa`: the first Qwen Sparse Attention layer of a qwen4_exp checkpoint (tools/build_random_hybrid.py);
`dsa`: the first pooled-DSA layer of a glm5_next one. Variants: "full" (the layer's attention as
the engine runs it), "dense" (the selection skipped), and for qsa "no-mask" (the selection computed,
its mask discarded) and "mask-only" (block_mask over random scores). Each is the p50 of
synchronous calls; "max diff" compares the device output with the CPU's.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def timed(fn, args, n=20):
    out = fn(*args)
    (out[0] if isinstance(out, tuple) else out).cpu()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        out = fn(*args)
        (out[0] if isinstance(out, tuple) else out).cpu()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[n // 2], out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["qsa", "dsa"])
    ap.add_argument("model")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pages", type=int, default=4)
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--cpu", action="store_true", help="time on the CPU only (no device)")
    args = ap.parse_args()

    from kiln.config import LinearSpec, ModelConfig
    from kiln.models import mla, qwen4_exp
    from kiln.models.loader import load_model

    dev = torch.device("cpu")
    opts = None
    if not args.cpu:
        import libtorch_neuronx_lite  # noqa: F401

        from kiln.engine.model_runner import canonical_neuron_backend, neuronx_cc_args

        dev = torch.device("neuron:0")
        opts = dict(backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                    options={"compiler_args": neuronx_cc_args(torch.float32)})
    cfg = ModelConfig.from_pretrained(args.model)
    B, P, ps = args.batch, args.pages, args.page_size
    L = P * ps
    models = {}
    for d in ("cpu", "dev"):
        device = torch.device("cpu") if d == "cpu" else dev
        m = load_model(args.model, cfg, torch.float32, device, L + ps)
        slots = (B * P + 1) * ps
        k = [torch.zeros((slots, *a), device=device) for a, _ in m.kv_shapes()]
        v = [torch.zeros((slots, *b), device=device) for _, b in m.kv_shapes()]
        m.bind_kv_cache(k, v, ps, max_rows=128)
        m.bind_aux_kv([torch.zeros((slots, *a), device=device) for a in m.aux_kv_shapes()])
        models[d] = m
        if args.cpu:
            break
    i = next(j for j, s in enumerate(cfg.attn_layers) if not isinstance(s, LinearSpec))
    g = torch.Generator().manual_seed(0)
    H = cfg.hidden_size
    x = torch.randn(B, H, generator=g)
    ctx = torch.randint(ps, L, (B,), generator=g)  # each row's context length
    positions = ctx - 1
    table = (torch.arange(B * P) + 1).view(B, P)  # page 0 is the null page
    slot = table[torch.arange(B), positions // ps] * ps + positions % ps
    j = torch.arange(L).unsqueeze(0)
    bias = torch.where(j < ctx.unsqueeze(1), 0.0, mla.NEG_INF).view(B, 1, 1, L)
    # Fill every cache row of the context with random values (as a prefix written earlier would).
    for m in models.values():
        for c in [*(lay.k_cache for lay in m.kv_layers()), *(lay.v_cache for lay in m.kv_layers()),
                  *[getattr(lay, "idx_cache", None) for lay in m.kv_layers() if getattr(lay, "idx_cache", None) is not None]]:
            c.copy_(torch.randn(c.shape, generator=torch.Generator().manual_seed(1)).to(c.device))

    def run(model, fn_name):
        layer = model.layers[i]

        def f(x, positions, slot, table, bias):
            if args.what == "dsa":
                return mla.attention(model, layer, x, positions, slot, table, bias)
            return qwen4_exp.attention(model, layer, x, positions, slot, table, bias, None)

        if fn_name == "dense":
            import dataclasses

            spec = layer.spec
            if args.what == "qsa":
                layer.spec = dataclasses.replace(spec, budget=1 << 20)
            else:
                m_ = spec.mla
                layer.spec = dataclasses.replace(spec, mla=dataclasses.replace(
                    m_, dsa=dataclasses.replace(m_.dsa, topk=1 << 20)))
            return f, lambda: setattr(layer, "spec", spec)
        if fn_name == "no-mask":
            old = qwen4_exp._select
            qwen4_exp._select = lambda *a: torch.zeros_like(old(*a))
            return f, lambda: setattr(qwen4_exp, "_select", old)
        if fn_name == "bisect":  # the real selection with the bisection threshold
            import kiln.models.glm5_next as gn

            old_bm = gn.block_mask
            gn.block_mask = lambda *a, tail=True: _block_mask_variant("bisect", *a, tail)
            return f, lambda: setattr(gn, "block_mask", old_bm)
        if fn_name.endswith("-qonly") and not fn_name.startswith("sel-"):  # block_mask variants
            import kiln.models.glm5_next as gn

            old_bm = gn.block_mask
            gn.block_mask = lambda *a, tail=True: _block_mask_variant(fn_name.split("-")[0], *a, tail)
            old = qwen4_exp._select
            qwen4_exp._select = lambda *a: _select_variant("qonly", *a)

            def undo():
                gn.block_mask = old_bm
                qwen4_exp._select = old
            return f, undo
        if fn_name.startswith("sel-"):  # _select with one part changed (see _select_variant)
            old = qwen4_exp._select
            qwen4_exp._select = lambda *a: _select_variant(fn_name[4:], *a)
            return f, lambda: setattr(qwen4_exp, "_select", old)
        if fn_name == "keys-mask-only":  # the cached keys' block mask, returned (no attention)
            def fk(x, positions, slot, table, bias):
                kI = model._load(layer.idx_cache, table).squeeze(-2)
                return _select_variant("noq", model, layer, None, kI, bias.reshape(B, 1, L))
            return fk, lambda: None
        if fn_name == "mask-only":
            from kiln.models.glm5_next import block_mask

            def fm(x, positions, slot, table, bias):
                vis = bias.reshape(B, 1, L)
                idx = (x[:, : L // 4] * 0.5).view(B, 1, L // 4)
                return block_mask(idx, vis, 4, layer.spec.budget // 4, True)
            return fm, lambda: None
        return f, lambda: None

    variants = ["full", "dense"] + (["flat-qonly", "unsorted-qonly", "bisect-qonly", "bisect"]
                                    if args.what == "qsa" else [])
    for name in variants:
        res = {}
        for d, m in models.items():
            fn, undo = run(m, name)
            inputs = [t.to(m.embed.device) for t in (x, positions, slot, table, bias)]
            if d == "dev":
                fn = torch.compile(fn, **opts)
            with torch.no_grad():
                t, out = timed(fn, inputs, 3 if d == "cpu" else 20)
            undo()
            res[d] = (t, out.cpu() if isinstance(out, torch.Tensor) else out)
        line = f"{args.what} {name:>9}: cpu {res['cpu'][0] * 1e3:8.2f} ms"
        if "dev" in res:
            diff = (res["dev"][1].float() - res["cpu"][1].float()).abs().max().item()
            line += f"  device {res['dev'][0] * 1e3:8.2f} ms  max diff {diff:.2e}"
        print(line, flush=True)


def _block_mask_variant(kind, index, vis, kp, keep, tail):
    """glm5_next.block_mask with the selection made by a threshold (the keep-th largest score,
    no scatter), by a one-hot comparison of the top-k indices (no scatter), or without the tail."""
    from kiln.models.decoder import NEG_INF

    B, Q, L = vis.shape
    P = L // kp
    cand = vis.reshape(B, Q, P, kp)[..., kp - 1]
    sc = index + cand
    if kind == "thr":
        thr = torch.topk(sc, keep, dim=-1).values[..., keep - 1 :]
        sel = torch.where(sc >= thr, 0.0, NEG_INF)
    elif kind == "flat":  # top-k over a 2-D contiguous copy
        top = torch.topk(sc.reshape(B * Q, P).contiguous(), keep, dim=-1).indices.view(B, Q, keep)
        sel = torch.full_like(cand, NEG_INF).scatter(-1, top, 0.0)
    elif kind == "unsorted":
        top = torch.topk(sc, keep, dim=-1, sorted=False).indices
        sel = torch.full_like(cand, NEG_INF).scatter(-1, top, 0.0)
    elif kind == "bisect":  # the keep-th largest score by bisection over [min, max] of the scores
        lo, hi = index.amin(-1, keepdim=True), index.amax(-1, keepdim=True)
        for _ in range(32):
            mid = (lo + hi) * 0.5
            ok = (sc >= mid).float().sum(-1, keepdim=True) >= keep
            lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
        sel = torch.where(sc >= lo, 0.0, NEG_INF)
    elif kind == "mean":  # no top-k at all
        sel = torch.where(sc >= sc.mean(-1, keepdim=True), 0.0, NEG_INF)
    elif kind == "sort":
        thr = torch.sort(sc, dim=-1, descending=True).values[..., keep - 1 : keep]
        sel = torch.where(sc >= thr, 0.0, NEG_INF)
    elif kind == "max":  # the keep-th largest by keep rounds of max-and-remove
        rest, thr = sc, None
        for _ in range(keep):
            thr = rest.amax(-1, keepdim=True)
            rest = torch.where(rest >= thr, NEG_INF, rest)
        sel = torch.where(sc >= thr, 0.0, NEG_INF)
    elif kind == "onehot":
        top = torch.topk(sc, keep, dim=-1).indices
        hit = (top.unsqueeze(-1) == torch.arange(P, device=vis.device)).any(dim=-2)
        sel = torch.where(hit, 0.0, NEG_INF)
    else:
        top = torch.topk(sc, keep, dim=-1).indices
        sel = torch.full_like(cand, NEG_INF).scatter(-1, top, 0.0)
    sel = sel.unsqueeze(-1).expand(B, Q, P, kp).reshape(B, Q, L)
    if tail and kind != "notail":
        nvis = torch.exp(vis).sum(-1, keepdim=True)
        start = torch.floor(nvis * (1.0 / kp)) * kp
        j = torch.arange(L, device=vis.device, dtype=vis.dtype)
        sel = torch.maximum(sel, torch.where(j >= start, 0.0, NEG_INF))
    return sel


def _select_variant(kind, model, layer, q, kI, vis):
    """qwen4_exp._select with its block-key RoPE removed ("norope"), its k_layernorm removed
    ("nonorm"), or the key split into its rope and pass dims before anything else ("split": two
    parts, the norm's mean square from both, RoPE over a whole tensor, never a slice of one)."""
    import math

    from kiln.models.glm5_next import block_mask
    from kiln.models.qwen4_exp import _rope, norm

    sp = layer.spec
    c, di, dr = sp.compress, sp.index_head_dim, sp.rope_dim
    B, Q, L = vis.shape
    P = L // c
    starts = torch.arange(P, device=kI.device) * c
    if kind == "qonly":  # a block mask from the query alone, then the attention
        index = q[0].float().sum(2)[..., :P]
        return block_mask(index, vis, c, sp.budget // c, tail=True)
    if kind == "kI":  # only read the cached keys
        return torch.zeros_like(vis) + kI.sum() * 1e-30
    if kind == "split":
        kr = kI[..., :dr].reshape(B, P, c, dr).float().mean(dim=2)
        kp = kI[..., dr:].reshape(B, P, c, di - dr).float().mean(dim=2)
        ms = (kr.pow(2).sum(-1, keepdim=True) + kp.pow(2).sum(-1, keepdim=True)) / di
        r = torch.rsqrt(ms + model.cfg.rms_norm_eps)
        kb_r = _rope(model, layer, (kr * r * layer.idx_kn[:dr]).to(kI.dtype), starts, False)
        kb_p = (kp * r * layer.idx_kn[dr:]).to(kI.dtype)
    else:
        kb = kI.reshape(B, P, c, di).float().mean(dim=2).to(kI.dtype)
        if kind != "nonorm":
            kb = norm(kb, layer.idx_kn, model.cfg.rms_norm_eps)
        kb_r = kb[..., :dr] if kind == "norope" else _rope(model, layer, kb[..., :dr], starts, False)
        kb_p = kb[..., dr:]
    if kind == "noq":  # the block keys alone choose
        index = kb_r.float().sum(-1).unsqueeze(1) + kb_p.float().sum(-1).unsqueeze(1)
        return block_mask(index, vis, c, sp.budget // c, tail=True)
    s = torch.einsum("bqhd,bpd->bqhp", q[0].float(), kb_r.float())
    s = s + torch.einsum("bqhd,bpd->bqhp", q[1].float(), kb_p.float())
    index = torch.relu(s).sum(dim=2) / math.sqrt(di)
    if kind == "scores":  # the scores, no top-k
        return index.repeat_interleave(c, dim=-1) * 1e-30
    return block_mask(index, vis, c, sp.budget // c, tail=True)


if __name__ == "__main__":
    main()
