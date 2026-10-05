"""Device time of one tensor-parallel rank's MLA / DSA attention block, through the real code
(DecoderForCausalLM._attention -> models/mla.py attention), at a real model's rank shapes
with random weights in the keep_fp8 layout (tools/profile_layer.py build).

    python tools/profile_mla.py --model zai-org/GLM-5.3 --tp 16 --batch 4 --pages 64 128
    python tools/profile_mla.py --model zai-org/GLM-5.3 --tp 16 --topk 512 --modes mask gather
    python tools/profile_mla.py --model deepseek-ai/DeepSeek-V3 --tp 16 --prefill 128 --pages 8
    python tools/profile_mla.py --model zai-org/GLM-5.3 --tp 16 --batch 8 --pages 128 256 --select bisect topk
    python tools/profile_mla.py --model zai-org/GLM-5.3-Flash --tp 32 --layers 4 --batch 8 --pages 256

For every page bucket P (context L = P * page size) it times the attention block of each
attention kind of the first --layers layers (GLM-5.3: a layer with its own indexer, then a
shared one), and for DSA layers whether the top-k is applied as a mask or a gather
(models/mla.py KILN_DSA). DSA runs dense whenever L <= index_topk, so a bucket below the
top-k measures plain MLA plus the indexer's key write. Also times the indexer's top-k alone
on [rows, L] scores. Each timing is the p50 of synchronous calls (profile_layer.timed).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_layer as pl  # noqa: E402


PROBES = ("noindex", "keysonly", "noselect", "nosoftmax")


def pooled_probe(kind: str):
    """glm5_next.pooled_selection with a part left out, to price the parts (the outputs are WRONG):
    noindex: no pool keys, scores or selection (an all-zero mask); keysonly: the pool keys but no
    scores; noselect: pool keys and scores but no selection; nosoftmax: each pool's key is its first
    token's (no softmax pooling), scores and the nki selection as usual. A part is kept alive by a
    tiny term in the mask (min(0, max(x)) * 1e-30) so the compiler cannot drop it. With the pool-key
    cache (KILN_DSA_POOL_CACHE=1) the pool keys are the cache's and nosoftmax is the full selection."""
    import kiln.models.glm5_next as gn
    from kiln.models import dsa_select

    def alive(x, vis):
        return vis * 0.0 + torch.clamp(x.float().amax(-1, keepdim=True) * 1e-30, max=0.0)

    def f(layer, d, q, w, kI, vis, pk=None):
        q_r, q_p = q
        kp, Di = d.kpool, d.head_dim
        B, Q, L = vis.shape
        if kind == "noindex":
            return torch.zeros_like(vis)
        P = L // kp
        if pk is not None:  # the pool-key cache's (models/mla.py POOL_CACHE)
            pass
        elif kind == "nosoftmax":
            pk = kI[..., :Di].reshape(B, P, kp, Di)[:, :, 0]
        else:
            k = kI[..., :Di].reshape(B, P, kp, Di)
            logits = kI[..., Di : 2 * Di].reshape(B, P, kp, Di).float() + layer.idx_pool_ape.float()
            pk = (torch.softmax(logits, dim=2).to(k.dtype) * k).sum(dim=2)
        if kind == "keysonly":
            return alive(pk.reshape(B, 1, -1).expand(B, Q, P * Di), vis)
        s = torch.einsum("bqhd,bpd->bqhp", q_p.float(), pk.float())
        index = torch.einsum("bqh,bqhp->bqp", w, torch.relu(s * Di ** -0.5))
        if kind == "noselect":
            return alive(index, vis)
        return gn.block_mask(index, vis, kp, d.topk // kp, d.kpool_tail, select=dsa_select.SELECT)

    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zai-org/GLM-5.3")
    ap.add_argument("--tp", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4, help="first n layers (GLM-5.3: 3 full indexers, then shared)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--prefill", type=int, default=0, help="time a C-token prefill chunk instead of decode")
    ap.add_argument("--pages", type=int, nargs="+", default=[16, 64, 128])
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--topk", type=int, default=None, help="override index_topk")
    ap.add_argument("--modes", nargs="+", default=["mask", "gather"])
    ap.add_argument("--select", nargs="+", default=None,
                    help="DSA selections to time (models/dsa_select.py KILN_DSA_SELECT: bisect, topk, nki), each optionally "
                         "with module settings, e.g. bisect:TIES=block,GROUP=0,BISECT_ROUNDS=40; default: the module's. "
                         "Cost probes for a pooled layer (GLM-5.3-Flash; WRONG outputs, see pooled_probe): noindex, "
                         "keysonly, noselect, nosoftmax")
    ap.add_argument("--scratch-rows", type=int, default=None,
                    help="rows of the DSA selection scratch (default: max(rows of the call, 128))")
    ap.add_argument("--select-only", action="store_true",
                    help="time the full DSA layers up to their selection only (mla._core returns the mask)")
    ap.add_argument("--cpu", action="store_true", help="dry run on the host")
    args = ap.parse_args()

    pl.setup_device(args.cpu)
    import kiln.models.mla as mla
    from kiln.config import ModelConfig

    P_max, ps = max(args.pages), args.page_size
    if args.topk is not None:  # rebuild the specs with another index_topk
        orig = ModelConfig.from_pretrained

        def patched(path):
            cfg = orig(path)
            specs = tuple(dataclasses.replace(s, mla=dataclasses.replace(
                s.mla, dsa=dataclasses.replace(s.mla.dsa, topk=args.topk))) if getattr(s, "mla", None) is not None and s.mla.dsa
                else s for s in cfg.attn_layers)
            return dataclasses.replace(cfg, attn_layers=specs)

        ModelConfig.from_pretrained = staticmethod(patched)
    if args.select_only:  # the layer's projections, cache gathers and selection, nothing after
        def core(model_, layer, q_nope, q_pe, kc, kpe, vis, B_, Q_, top=None, index=None, absorb=True):
            d_ = layer.spec.mla.dsa
            (q_r, q_p), wi, kI = index
            q4 = (None if q_r is None else q_r.reshape(B_, Q_, d_.n_heads, -1), q_p.reshape(B_, Q_, d_.n_heads, -1))
            sel = mla._select(d_, q4, wi.view(B_, Q_, d_.n_heads), kI, vis)
            return sel.reshape(B_ * Q_, -1)[:, :1].to(q_nope.dtype).expand(B_ * Q_, layer.o.shape[1]), None

        mla._core = core
    B = 1 if args.prefill else args.batch
    rows = args.prefill or B
    # The DSA scratch holds the rows one call carries, as the engine sizes it (ModelRunner: the
    # largest prefill bucket or decode batch x (1 + spec_k)), and the longest context.
    cfg, model = pl.build(args.model, args.tp, args.layers, 1 + B * P_max, ps, P_max * ps + 1,
                          max_rows=args.scratch_rows or max(rows, 128), max_keys=P_max * ps)
    kinds: dict = {}
    for i, l in enumerate(model.layers):
        if getattr(l.spec, "mla", None) is not None:  # hybrids (GLM-5.3-Flash): only the MLA / DSA layers
            kinds.setdefault(repr(l.spec), i)
    import kiln.models.dsa_select as ds

    selects = args.select or [ds.SELECT]
    defaults = (ds.SELECT, ds.GROUP, ds.BISECT_ROUNDS, ds.TIES, mla.STAGE)
    m0 = model.layers[min(kinds.values())].spec.mla
    pl.say(f"{args.model} tp={args.tp}: {len(model.layers)} layers, attention kinds at {sorted(kinds.values())}; "
           f"nh={model.layers[min(kinds.values())].nh} kv_lora={m0.kv_lora_rank} rope={m0.qk_rope_head_dim} "
           f"dsa={m0.dsa}", flush=True)
    g = torch.Generator().manual_seed(1)
    h = torch.randn(rows, cfg.hidden_size, generator=g).to(torch.bfloat16).to(pl.DEV)
    for P in args.pages:
        inp = pl.prefill_inputs(model, args.prefill, P, ps) if args.prefill else pl.decode_inputs(model, B, P, ps)
        L = P * ps
        pl.say(f"{'prefill C=%d' % args.prefill if args.prefill else 'decode B=%d' % B}, P={P} (L={L}):", flush=True)
        for i in sorted(kinds.values()):
            d = model.layers[i].spec.mla.dsa
            sparse = d is not None and L > d.topk
            for mode in (args.modes if sparse else ["dense"]):
                for sel in (selects if sparse and d.indexer else [ds.SELECT]):
                    mla.DSA_MODE = mode if sparse else "mask"
                    ds.SELECT, _, opts = sel.partition(":")
                    import kiln.models.glm5_next as gn

                    probe = ds.SELECT if ds.SELECT in PROBES else None
                    saved = (gn.block_mask, gn.pooled_selection)
                    if probe:
                        gn.pooled_selection = pooled_probe(probe)
                        ds.SELECT = "nki" if probe == "nosoftmax" else "bisect"
                    for kv in filter(None, opts.split(",")):  # module settings for this run, e.g. TIES=block
                        k_, v_ = kv.split("=")
                        mod = ds if hasattr(ds, k_) else mla  # dsa_select's, else models/mla.py's (STAGE)
                        old_ = getattr(mod, k_)
                        setattr(mod, k_, type(old_)(int(v_)) if isinstance(old_, (bool, int)) else v_)
                    f, ts = pl.with_layer(model, i, lambda V, h, p, s, tb, bs: model._attention(V, h, p, s, tb, bs))
                    label = "mla" if d is None else ("dsa-indexer" if d.indexer else "dsa-shared")
                    if d is not None and d.kpool > 1:
                        label = "dsa-pooled"
                    pl.timed(f"[{i}] {label} attention, {mode}" + (f", {sel}" if sparse and d.indexer else ""), f,
                             (h, inp["positions"], inp["slots"], inp["table"], inp["bias"], *ts))
                    ds.SELECT, ds.GROUP, ds.BISECT_ROUNDS, ds.TIES, mla.STAGE = defaults
                    gn.block_mask, gn.pooled_selection = saved
        if any(model.layers[i].spec.mla.dsa is not None for i in kinds.values()):
            k = m0.dsa.topk
            if L > k:
                sc = torch.randn(rows, L, generator=g).to(pl.DEV)
                pl.timed(f"topk({k}) over [{rows}, {L}] fp32", lambda s: torch.topk(s, k, dim=-1).indices, (sc,))
                top = torch.topk(sc.cpu(), k, dim=-1).indices.to(pl.DEV)
                pl.timed(f"mask from [{rows}, {k}] indices (scatter)",
                         lambda s, t: torch.full_like(s, -1e30).scatter(-1, t, 0.0), (sc, top))


if __name__ == "__main__":
    main()
