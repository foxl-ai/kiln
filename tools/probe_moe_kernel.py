"""The NKI MoE kernels (kiln/kernels/moe_decode.py, one expert load per (token, expert) pair, and
kiln/kernels/moe_dedupe.py, one load per distinct expert) on one NeuronCore against the XLA
gather path, at one tensor-parallel rank's shapes, through the real code
(DecoderForCausalLM._moe_routed with a blob layer and with the natural layout).

    python tools/probe_moe_kernel.py [--batch 1 2 4 8 16 32 64] [--experts 256] [--hidden 4096]
        [--top-k 8] [--no-xla] [--iters 20] [--kernels pair dedupe] [--lanes L]

Random FP8 experts (finite e4m3 bytes, bf16 block-32 scales, as tools/profile_layer.py) for
MiMo-V2.6-Flash at tp=32 by default: 256 experts, hidden 4096, 2 x 64 gate/up rows per rank,
top-8. Per batch: routing [B, 8] with distinct experts per token and positive weights; the
kernel graph, the XLA gather graph and an fp32 host reference computed from the same bytes;
max abs error of each against the reference (and kernel against XLA, dedupe against pair),
relative to the reference's max; then the p50 of synchronous calls of each graph (launch + run +
readback) and the null-graph floor. Layer tensors are graph inputs, as in the piecewise runner.
--batch also stands for a prefill chunk (B tokens of one sequence: the same call). Distinct
experts per call are printed, since they are what the dedupe kernel reads.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def experts(E: int, H: int, seed: int = 0):
    """MXFP4-like random experts, as the checkpoint's after conversion (models/quant.py): E2M1 codes
    as FP8 and power-of-two bf16 scales per 32 input columns, exponents 2^-10 .. 2^-4 (the real
    ones of a 128-column tile differ by at most 8: tools/check_expert_scales.py)."""
    from kiln.models.quant import E2M1, FP8

    g = torch.Generator().manual_seed(seed)
    lut = torch.tensor(E2M1)
    fp8 = lambda *s: lut[torch.randint(0, 16, s, generator=g)].to(FP8)  # noqa: E731
    p2 = lambda *s: torch.exp2(torch.randint(-10, -3, s, generator=g).float()).bfloat16()  # noqa: E731
    w_gu, w_down = fp8(E, 128, H), fp8(E, 64, H)
    return w_gu, p2(E, 128, H // 32), w_down, p2(E, 2, H)


def experts128(E: int, H: int, seed: int = 0):
    """FP8 experts with 128 x 128 block scales as the loader keeps them (GLM-5.3-Flash, DeepSeek
    style): finite e4m3 bytes, fp32 scales per row per 128 input columns (gate_up) and per output
    column over the rank's 64 input rows (down)."""
    from kiln.models.quant import FP8

    g = torch.Generator().manual_seed(seed)
    fp8 = lambda *s: (torch.randint(0, 256, s, dtype=torch.uint8, generator=g) & 0xBF).view(FP8)  # noqa: E731
    return (fp8(E, 128, H), torch.rand(E, 128, H // 128, generator=g) * 0.02 + 0.005, fp8(E, 64, H),
            torch.rand(E, 1, H, generator=g) * 0.02 + 0.005)


ACT = dict(act=0, limit=0.0)  # the activation (kiln/kernels/moe_dedupe.ACTS), set by --act / --limit


def reference(x, topv, topi, w_gu, s_gu, w_down, s_down):
    """fp32 on the host: dequantize each pair's expert, matvec, SiLU gate, matvec, weight, sum."""
    from kiln.models.quant import dequant, dequant_t

    T, K = topi.shape
    flat = topi.reshape(-1)
    out = torch.zeros(T * K, x.shape[1])
    for s in range(0, T * K, 64):  # chunks of pairs keep the dequantized experts small
        idx = flat[s : s + 64]
        wg = dequant(w_gu[idx], s_gu[idx], torch.float32)  # [n, 128, H]
        wd = dequant_t(w_down[idx], s_down[idx], torch.float32)  # [n, 64, H]
        xs = x.float().repeat_interleave(K, dim=0)[s : s + 64]
        gu = torch.einsum("noh,nh->no", wg, xs)
        a = torch.nn.functional.silu(gu[:, :64]) * gu[:, 64:]
        if ACT["act"]:
            from kiln.kernels.moe_dedupe import glu

            a = glu(gu[:, :64], gu[:, 64:], ACT["act"], ACT["limit"])
        out[s : s + 64] = torch.einsum("ni,nih->nh", a, wd) * topv.reshape(-1)[s : s + 64].float().unsqueeze(1)
    return out.view(T, K, -1).sum(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--no-xla", action="store_true", help="skip the XLA gather graph")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--kernels", nargs="+", default=["pair", "dedupe"], choices=["pair", "dedupe"])
    ap.add_argument("--lanes", type=int, default=None, help="dedupe lanes per slot (default by batch)")
    ap.add_argument("--parts", action="store_true", help="also time the dedupe routing plan and kernel alone")
    ap.add_argument("--dump", default=None, help="with --parts: save the dedupe kernel's inputs as input<i>.npy "
                    "(raw bytes) there, for tools/prof_timeline.py --inputs")
    ap.add_argument("--scales", default="mxfp4", choices=["mxfp4", "block128"],
                    help="experts as MXFP4 converts (bf16 block-32) or FP8 with 128 x 128 block scales (fp32)")
    ap.add_argument("--act", default="silu", choices=["silu", "silu_clamp", "swiglu_oai"])
    ap.add_argument("--limit", type=float, default=10.0, help="clamp of silu_clamp / swiglu_oai")
    args = ap.parse_args()

    import profile_layer as pl

    pl.setup_device()
    import types

    from kiln.kernels import moe_decode as md
    from kiln.kernels import moe_dedupe as mdd
    from kiln.models.decoder import DecoderForCausalLM, _LayerView

    E, H, K = args.experts, args.hidden, args.top_k
    ACT.update(act=mdd.ACTS[args.act], limit=args.limit)
    if args.scales == "block128" or ACT["act"]:  # the per-pair kernel has neither
        args.kernels = [k for k in args.kernels if k != "pair"]
    ws = (experts128 if args.scales == "block128" else experts)(E, H)
    blob = mdd.pack(*ws) if "pair" not in args.kernels else md.pack(*ws)
    dev_ws = [t.to(pl.DEV) for t in ws] if not args.no_xla else []
    blobs = {"pair": blob.to(pl.DEV) if "pair" in args.kernels else None,
             "dedupe": mdd.pack(*ws).to(pl.DEV) if "dedupe" in args.kernels else None}
    m = DecoderForCausalLM.__new__(DecoderForCausalLM)
    torch.nn.Module.__init__(m)
    m.cfg = types.SimpleNamespace(hidden_size=H, num_experts=E, num_experts_per_tok=K)
    m.dtype, m.moe_inter = torch.bfloat16, 64
    names = ("w_gu", "w_gu_scale", "w_down", "w_down_scale")

    def xla(x, topv, topi, *t):
        V = _LayerView({"moe_blob": False, "down_t": True}, dict(zip(names, t)))
        if not ACT["act"]:
            return m._moe_routed(V, x, topv, topi)
        # the gather path with the activation (as models/hybrid.py _moe_clamped's)
        T = x.shape[0]
        flat = topi.reshape(T * K)
        xs = x.unsqueeze(1).expand(T, K, H).reshape(T * K, H, 1)
        gu = torch.bmm(m._experts(V, "w_gu", flat), xs).squeeze(-1)
        a = mdd.glu(gu[:, :64], gu[:, 64:], ACT["act"], ACT["limit"])
        y = torch.bmm(a.unsqueeze(1), m._experts(V, "w_down", flat)).squeeze(1)
        return (y.view(T, K, H) * topv.unsqueeze(-1)).sum(dim=1)

    def kern(tiles: bool):
        def f(x, topv, topi, b):  # the blob's layout picks the kernel
            if tiles and ACT["act"]:
                return mdd.moe_dedupe(x, topv, topi, b, act=ACT["act"], limit=ACT["limit"])
            return m._moe_routed(_LayerView({"moe_blob": True, "moe_tiles": tiles}, {"w_blob": b}), x, topv, topi)

        return f

    fns = {"pair": kern(False), "dedupe": kern(True)}

    if args.lanes:
        import functools

        from kiln.kernels import moe_dedupe

        moe_dedupe.default_lanes = functools.partial(lambda n, T: n, args.lanes)

    print(f"experts {E} x (128 x {H} gate/up, 64 x {H} down) fp8, {args.scales} scales, {args.act} "
          f"(limit {args.limit}), dedupe blob {tuple(blobs['dedupe'].shape) if blobs['dedupe'] is not None else None}, "
          f"top-{K}", flush=True)
    pl.timed("null graph (launch + readback)", lambda h: h + 1,
             (torch.zeros(4, H, dtype=torch.bfloat16).to(pl.DEV),), args.iters)
    g = torch.Generator().manual_seed(1)
    for B in args.batch:
        x = torch.randn(B, H, generator=g).bfloat16()
        topi = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(B)])
        topv = (torch.rand(B, K, generator=g) + 0.1).bfloat16()
        ref = reference(x, topv, topi, *ws)
        d = [t.to(pl.DEV) for t in (x, topv, topi)]
        outs = {}
        for k in args.kernels:
            t0 = time.perf_counter()
            outs[k] = torch.compile(fns[k], **pl.OPTS)(*d, blobs[k]).cpu().float()
            print(f"B={B} pairs={B * K} distinct={len(set(topi.reshape(-1).tolist()))}: {k} kernel graph first call "
                  f"{time.perf_counter() - t0:.1f} s", flush=True)
        if not args.no_xla:
            outs["xla"] = torch.compile(xla, **pl.OPTS)(*d, *dev_ws).cpu().float()
        scale = ref.abs().max().item()
        for name, o in outs.items():
            print(f"  {name} vs fp32 reference: max abs {(o - ref).abs().max().item():.3e} "
                  f"(rel {(o - ref).abs().max().item() / scale:.4f}, |ref| max {scale:.3e})", flush=True)
        for a, b in (("pair", "xla"), ("dedupe", "pair"), ("dedupe", "xla")):
            if a in outs and b in outs:
                dd = (outs[a] - outs[b]).abs().max().item()
                print(f"  {a} vs {b}: max abs {dd:.3e} (rel {dd / scale:.4f})", flush=True)
        for k in args.kernels:
            pl.timed(f"B={B} moe, NKI {k} kernel", fns[k], (*d, blobs[k]), args.iters)
        if args.parts:
            from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

            L = args.lanes or mdd.default_lanes(B)
            dev_blob = blobs["dedupe"]
            pl.timed(f"B={B} dedupe routing plan alone (lanes {L})", lambda v, i: mdd.plan(v, i, E, L), (d[1], d[2]),
                     args.iters)
            kw = mdd.kernel_inputs(x, topv, topi, blob, L)  # (only its expert count is read)
            st = {n: kw.pop(n) for n in ("lanes", "group", "ring", "slots", "block", "keep", "act", "limit", "alpha", "debug")}
            names = [n for n in kw if n != "blob"]
            dv = [kw[n].to(pl.DEV) for n in names]
            fk = lambda b, *t: wrap_nki(mdd.kernel())[1](blob=b, **dict(zip(names, t)), **st)  # noqa: E731
            if args.dump:
                import numpy as np

                os.makedirs(args.dump, exist_ok=True)
                order = [n for n in kw]  # the kernel's argument order: x, G, R, slot_e, blob, ...
                full = dict(kw, blob=mdd.pack(*ws))
                for i, n in enumerate(order):
                    t = full[n].contiguous()
                    raw = t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int16 if t.element_size() == 2 else torch.int32)
                    np.save(os.path.join(args.dump, f"input{i}.npy"), raw.numpy())
                print(f"  dumped {order} to {args.dump}", flush=True)
            o = torch.compile(fk, **pl.OPTS)(dev_blob, *dv).cpu().float()
            dd = (o - ref).abs().max().item()
            print(f"  dedupe kernel alone vs fp32 reference: rel {dd / scale:.4f}", flush=True)
            pl.timed(f"B={B} dedupe kernel alone ({mdd.n_slots(B, K, E, L)[0]} slots)", fk,
                     (dev_blob, *dv), args.iters)
        if not args.no_xla:
            pl.timed(f"B={B} moe, XLA gather", xla, (*d, *dev_ws), args.iters)


if __name__ == "__main__":
    main()
