"""The NKI prefill-MoE kernel (kiln/kernels/moe_prefill.py) on one NeuronCore at one tensor-parallel
rank's shapes, against the decode kernels on the same chunk: kernels/moe_dedupe.py (the default
KILN_MOE_KERNEL=nki, same blob, 128-token calls) and kernels/moe_decode.py (per pair).

    python tools/probe_moe_prefill.py [--format glm|mimo] [--chunks 256 512 ... 8192] [--experts 288]
        [--no-decode] [--decode-max 2048] [--pair-max 0] [--skew] [--no-dq] [--block B] [--iters 10]
        [--save-inputs <path>]

--format glm (default): random FP8 experts with fp32 128 x 128 block scales as GLM-5.3-Flash's
checkpoint stores them at tp=32 (finite e4m3 bytes; --format loaded: as the loader then holds them,
after fit_e4m3_max), packed by moe_dedupe.pack: 288 experts, hidden 4096, 2 x 64
gate/up rows and 64 down rows per expert, top-8, SwiGLU clamped at swiglu_limit 10. --format mimo:
MiMo-V2.6-Flash's tp=32 rank shapes (256 experts), E2M1 codes as FP8 with power-of-two block-32
scales (re-based by pack onto bf16 tile scales), SiLU. Per chunk C: routing with distinct
experts per token (uniform, or --skew: every token's first expert from 8 hot ones); the kernel
graph (one kernel call, which also computes the routing plan); its output against the host emulation of the
kernel's arithmetic (moe_dedupe.emulate) and an fp32 reference (max abs error relative to the
reference's max); p50 of synchronous calls of the graph, which reads back out.float().sum(0) (a
full readback of [C, 4096] would time PCIe; the same reduction of a [C, 4096] input is timed alone
and subtracted); achieved TFLOPS counting 2 x pairs x (4096 x 128 + 64 x 4096) flops. The per-pair
decode kernel runs on MiMo-format weights at the same shapes (the only layout it takes) through
moe_decode.moe_selected's 512-pair chunks.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def experts(fmt: str, E: int, H: int, seed: int = 0):
    from kiln.models.quant import FP8

    g = torch.Generator().manual_seed(seed)
    fp8 = lambda *s: (torch.randint(0, 256, s, dtype=torch.uint8, generator=g) & 0xBF).view(FP8)  # noqa: E731
    w_gu, w_down = fp8(E, 128, H), fp8(E, 64, H)
    if fmt == "glm":
        s_gu = (torch.rand(E, 2, H // 128, generator=g) * 0.02 + 0.005).repeat_interleave(64, dim=1)
        s_down = (torch.rand(E, 1, H // 128, generator=g) * 0.02 + 0.005).repeat_interleave(128, dim=2)
    elif fmt == "loaded":  # GLM-5.3-Flash's experts as the loader leaves them (tests/test_moe_prefill.py
        # checkpoint_experts): e4m3fn codes of 128 x 128 blocks (block max 448), the rank's shard,
        # per-row scales, then quant.fit_e4m3_max, which doubles the scale of most rows
        from kiln.models.quant import fit_e4m3_max

        def blocks(rows, cols, sr, sc):
            w = torch.randn(rows, cols, generator=g) * 0.02
            bs = w.abs().view(rows // 128, 128, cols // 128, 128).amax((1, 3)) / 448.0
            cw = (w / bs.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(FP8)[sr, sc]
            rs = bs.repeat_interleave(128, 0)[sr]
            if sc.stop - sc.start < 128:
                rs = rs[:, sc.start // 128:sc.start // 128 + 1]
            return fit_e4m3_max(cw, rs, 240.0)

        out = [[], [], [], []]
        for _ in range(E):
            wg, sg = blocks(128, H, slice(0, 64), slice(0, H))
            wu, su = blocks(128, H, slice(64, 128), slice(0, H))
            wd, sd = blocks(H, 128, slice(0, H), slice(0, 64))
            for lst, t in zip(out, (torch.cat([wg, wu]), torch.cat([sg, su]), wd.T.contiguous(), sd.T.contiguous())):
                lst.append(t)
        return tuple(torch.stack(t) for t in out)
    elif fmt == "mxfp4":  # E2M1 codes as FP8 and power-of-two block-32 scales (tools/probe_moe_kernel.py)
        from kiln.models.quant import E2M1

        lut = torch.tensor(E2M1)
        w_gu, w_down = (lut[torch.randint(0, 16, s, generator=g)].to(FP8) for s in ((E, 128, H), (E, 64, H)))
        p2 = lambda *s: torch.exp2(torch.randint(-10, -3, s, generator=g).float()).bfloat16()  # noqa: E731
        s_gu, s_down = p2(E, 128, H // 32), p2(E, 2, H)
    else:
        s_gu = torch.exp2(torch.randint(-9, -4, (E, 128, H // 32), generator=g).float()).bfloat16()
        s_down = torch.exp2(torch.randint(-9, -4, (E, 2, H), generator=g).float()).bfloat16()
    return w_gu, s_gu, w_down, s_down


def reference(x, topv, topi, w_gu, s_gu, w_down, s_down, lim):
    """fp32 on the host, one expert at a time."""
    from kiln.models.quant import dequant, dequant_t

    out = torch.zeros(x.shape, dtype=torch.float32)
    for e in torch.unique(topi).tolist():
        t, k = (topi == e).nonzero(as_tuple=True)
        gu = x[t].float() @ dequant(w_gu[e], s_gu[e], torch.float32).T
        gate, up = gu[:, :64], gu[:, 64:]
        if lim >= 0:
            gate, up = gate.clamp(max=lim), up.clamp(min=-lim, max=lim)
        y = (torch.nn.functional.silu(gate) * up) @ dequant_t(w_down[e], s_down[e], torch.float32)
        out.index_add_(0, t, y * topv[t, k].float().unsqueeze(1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--format", default="glm", choices=["glm", "loaded", "mimo"],
                    help="glm: fp32 128 x 128 block scales, clamped SwiGLU (GLM-5.3-Flash's checkpoint blocks); "
                         "loaded: the same after the loader's fit_e4m3_max, as GLM-5.3-Flash's experts are held "
                         "(per-row gate_up and per-column down scales); mimo: MXFP4-like E2M1 codes with "
                         "power-of-two block-32 scales, SiLU (MiMo-V2.6-Flash)")
    ap.add_argument("--experts", type=int, default=None, help="default 288 (glm) / 256 (mimo)")
    ap.add_argument("--block-constant", action="store_true",
                    help="with --experts-file: down scales made constant per chunk (22dd2c2's refit), not per column")
    ap.add_argument("--group-fit", action="store_true",
                    help="with --experts-file: gate_up and down as KILN_MOE_E4M3_FIT=group would load them (the rows "
                         "of a 64-row gate / up group or a 128-row down chunk holding a doubled scale halved as well)")
    ap.add_argument("--experts-file", default=None,
                    help="a real layer's experts as the loader holds them (tools/check_moe_prefill_layout.py --save), "
                         "instead of random ones; GLM's clamped SwiGLU")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--decode-max", type=int, default=2048, help="largest chunk the dedupe kernel is run on")
    ap.add_argument("--pair-max", type=int, default=0, help="largest chunk the per-pair decode kernel is run on")
    ap.add_argument("--skew", action="store_true")
    ap.add_argument("--compare", action="append", default=[],
                    help="also run the kernel with these kernel_inputs overrides, e.g. order=0,nyb=0 (repeatable), "
                         "timed, and report bit-identity with the default")
    ap.add_argument("--compare-skip", type=int, default=0,
                    help="also run the kernel with KILN_MOE_PREFILL_SKIP's skp = N on the same inputs and report "
                         "whether its output is bit-identical to this run's (the skip changes no used tile)")
    ap.add_argument("--routing", default="uniform", choices=["uniform", "hot", "maxblocks"],
                    help="hot: every token on the same 8 experts (the fewest blocks); maxblocks: expert loads of 65 "
                         "or 1 pairs, the most blocks any routing needs (every lane-tile segment runs)")
    ap.add_argument("--no-dq", action="store_true", help="the per-row tile-scale path even for block scales")
    ap.add_argument("--block", type=int, default=None, help="lanes per block (default moe_prefill.block_size)")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--save-inputs", default=None, help="torch.save the kernel graph's inputs (per chunk: "
                    "<path>.C<chunk>.pt) for tools/prof_engines.py")
    args = ap.parse_args()

    import profile_layer as pl

    pl.setup_device()
    from kiln.kernels import moe_decode as md
    from kiln.kernels import moe_prefill as mp

    from kiln.kernels import moe_dedupe as mdd

    mimo = args.format == "mimo"
    E = args.experts or (256 if mimo else 288)
    H, K = args.hidden, args.top_k
    lim, act = (-1.0, mdd.ACTS["silu"]) if mimo else (10.0, mdd.ACTS["silu_clamp"])
    if args.experts_file:
        d = torch.load(args.experts_file)
        ws = (d["w_gu"], d["w_gu_scale"], d["w_down"], d["w_down_scale"])
        E = ws[0].shape[0]
        print(f"experts from {args.experts_file}: {d['model']} layer {d['layer']}, rank {d['rank']} of tp={d['tp']}",
              flush=True)
        if args.block_constant:  # down as feat/trn2-bench 22dd2c2's refit leaves it: in each 128-column chunk
            # holding a doubled scale, halve the other columns' codes and double their scales as well
            w_gu, s_gu, w_down, s_down = ws
            sd = s_down[:, 0].view(E, -1, 128)
            hi = sd.amax(-1, keepdim=True)
            half = (sd < hi).view(E, 1, -1)  # columns to halve now
            wd = torch.where(half, w_down.float() / 2, w_down.float()).to(w_down.dtype)
            ws = (w_gu, s_gu, wd, hi.expand_as(sd).reshape(E, 1, -1).contiguous())
            print(f"  down made block-constant as 22dd2c2's refit: {int(half.sum())} more columns halved", flush=True)
        if args.group_fit:  # models/quant.fit_e4m3_max row_group: gate rows, up rows (64 each), down per 128 columns
            w_gu, s_gu, w_down, s_down = ws
            sg = s_gu.view(E, 2, 64, -1)
            hi = sg.amax(2, keepdim=True)
            half = (sg < hi).reshape(E, 128, -1).repeat_interleave(H // s_gu.shape[-1], dim=2)
            wg = torch.where(half, w_gu.float() / 2, w_gu.float()).to(w_gu.dtype)
            sd = s_down[:, 0].view(E, -1, 128)
            hd = sd.amax(-1, keepdim=True)
            halfd = (sd < hd).view(E, 1, -1)
            wd = torch.where(halfd, w_down.float() / 2, w_down.float()).to(w_down.dtype)
            ws = (wg, hi.expand_as(sg).reshape(E, 128, -1).contiguous(), wd, hd.expand_as(sd).reshape(E, 1, -1).contiguous())
            print(f"  group fit: {int(half.sum())} more gate_up and {int(halfd.sum()) * 64} more down values halved",
                  flush=True)
    else:
        ws = experts("mxfp4" if mimo else args.format, E, H)
    blob = mdd.pack(*ws)
    dq = mp.check_blob(blob, H) and not args.no_dq
    dev_blob = blob.to(pl.DEV)
    down = mp.down_factors(blob, H)
    dev_down = None if down is None else tuple(t.to(pl.DEV) for t in down)
    flops_pair = 2 * (H * 128 + 64 * H)
    print(f"{args.format} experts: {E} x (128 x {H} gate/up fp8 + scales {tuple(ws[1].shape[1:])} {ws[1].dtype}, "
          f"64 x {H} down + scales {tuple(ws[3].shape[1:])}), tiles blob {tuple(blob.shape)}, top-{K}, "
          f"swiglu_limit {lim if lim >= 0 else 'none'}, dequantize-first path {dq}", flush=True)
    null = pl.timed("null graph (launch + readback)", lambda h: h + 1,
                    (torch.zeros(4, H, dtype=torch.bfloat16).to(pl.DEV),), args.iters)
    pair_blob = None
    if not args.no_decode:
        pair_blob = md.pack(*experts("mimo", E, H, seed=3)).to(pl.DEV)

    def call(x, topv, topi, b):
        if args.block:
            from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

            return wrap_nki(mp.kernel())[1](**mp.kernel_inputs(x, topv, topi, b, act, max(lim, 0.0), B=args.block,
                                                               dq=dq, down=dev_down))
        return mp.moe_prefill(x, topv, topi, b, act, max(lim, 0.0), dq=dq, down=dev_down)

    def kern(x, topv, topi, b):
        return call(x, topv, topi, b).float().sum(0)

    def dedupe(x, topv, topi, b):
        return mdd.moe_dedupe(x, topv, topi, b, act=act, limit=max(lim, 0.0)).float().sum(0)

    def dec(x, topv, topi, b):
        return md.moe_selected(x, topv, topi, b).float().sum(0)

    g = torch.Generator().manual_seed(1)
    for C in args.chunks:
        x = torch.randn(C, H, generator=g).bfloat16()
        topi = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(C)])
        if args.skew:
            hot = torch.randint(0, 8, (C,), generator=g)
            topi[:, 0] = hot
            for t in range(C):  # keep the experts of a token distinct
                rest = [e for e in torch.randperm(E, generator=g).tolist() if e != int(hot[t])][: K - 1]
                topi[t, 1:] = torch.tensor(rest)
        if args.routing == "hot":
            topi = torch.arange(K).repeat(C, 1)
        elif args.routing == "maxblocks":
            # Loads of 65 pairs (2 blocks of 64) on as many experts as fit, 1 pair on the rest, dealt out so
            # that every token gets K distinct experts: expert slots in descending load, token t takes the
            # pairs t, t + C, t + 2C ... of that list.
            big = min(E, (C * K - E) // 64)
            loads = [65] * big + [1] * (E - big)
            flat = [e for e, n in enumerate(loads) for _ in range(n)][: C * K]
            flat += [E - 1 - i % (E - big) for i in range(C * K - len(flat))]
            topi = torch.tensor(flat).view(K, C).T.contiguous()
        topv = (torch.rand(C, K, generator=g) + 0.1).bfloat16()
        pairs = C * K
        B = args.block or mp.block_size(C)
        NB = mp.n_blocks(pairs, E, B)
        counts = torch.bincount(topi.flatten(), minlength=E)
        used = int((-(-counts // B)).sum())
        print(f"C={C} pairs={pairs} B={B} blocks {NB} static, {used} used, max {int(counts.max())} pairs on one expert",
              flush=True)
        d = [t.to(pl.DEV) for t in (x, topv, topi)]
        t0 = time.perf_counter()
        c_kern = torch.compile(kern, **pl.OPTS)
        try:
            out = None
            got = torch.compile(call, **pl.OPTS)(*d, dev_blob).cpu().float()
            out = got
        except Exception as e:
            print(f"  kernel FAILED {type(e).__name__}: {str(e)[:400]}", flush=True)
            continue
        print(f"  kernel graph first call (compile + load) {time.perf_counter() - t0:.1f} s", flush=True)
        ref = reference(x, topv, topi, *ws, lim)
        emu = mp.emulate(x, topv, topi, blob, act, max(lim, 0.0), dq=dq).float()
        scale = ref.abs().max().item()
        print(f"  kernel vs fp32 reference: max abs {(out - ref).abs().max().item():.3e} "
              f"(rel {(out - ref).abs().max().item() / scale:.4f}); emulation vs reference rel "
              f"{(emu - ref).abs().max().item() / scale:.4f}; kernel vs emulation rel "
              f"{(out - emu).abs().max().item() / scale:.4f}; |ref| max {scale:.3e}", flush=True)
        if args.compare_skip:
            def call_skip(x, topv, topi, b):
                from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

                return wrap_nki(mp.kernel())[1](**mp.kernel_inputs(x, topv, topi, b, act, max(lim, 0.0), dq=dq,
                                                                   down=dev_down, skp=args.compare_skip))
            other = torch.compile(call_skip, **pl.OPTS)(*d, dev_blob).cpu().float()
            print(f"  skp={args.compare_skip} against skp={mp.SKIP}: bit-identical {bool(torch.equal(other, out))}, "
                  f"max abs diff {(other - out).abs().max().item():.3e}", flush=True)
        for spec in args.compare:
            kw = {k: int(v) for k, v in (a.split("=") for a in spec.split(","))}

            def call_kw(x, topv, topi, b, kw=kw):
                from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

                return wrap_nki(mp.kernel())[1](**mp.kernel_inputs(x, topv, topi, b, act, max(lim, 0.0), dq=dq,
                                                                   down=dev_down, **kw))
            other = torch.compile(call_kw, **pl.OPTS)(*d, dev_blob).cpu().float()
            print(f"  {spec} against the default: bit-identical {bool(torch.equal(other, out))}, "
                  f"max abs diff {(other - out).abs().max().item():.3e}", flush=True)
            pl.timed(f"C={C} moe, NKI prefill kernel, {spec}",
                     lambda x, tv, ti, b, f=call_kw: f(x, tv, ti, b).float().sum(0), (*d, dev_blob), args.iters)
        t_k = pl.timed(f"C={C} moe, NKI prefill kernel", kern, (*d, dev_blob), args.iters)
        if args.save_inputs:
            import glob

            path = f"{args.save_inputs}.C{C}.pt"
            torch.save(dict(x=x, topv=topv, topi=topi, b=blob), path)
            cache = "/root/.cache/neuron_libtorch/neuron/compile_cache"
            hs = [h for h in sorted(glob.glob(f"{cache}/*/fxgraph.txt"), key=os.path.getmtime)
                  if "L_topi_" in open(h).read() and "float_" in open(h).read()]
            print(f"  inputs saved to {path}; kernel graph cache entry {os.path.basename(os.path.dirname(hs[-1]))}"
                  if hs else f"  inputs saved to {path}", flush=True)
        t_s = pl.timed(f"C={C} the same sum over a [C, {H}] input", lambda y: y.float().sum(0),
                       (x.to(pl.DEV),), args.iters)
        if t_k == t_k:
            net = t_k - (t_s - null)
            print(f"  -> {net * 1e3:.3f} ms without the readback reduction; {net / pairs * 1e9:.1f} ns per pair; "
                  f"{pairs * flops_pair / net / 1e12:.2f} TFLOPS", flush=True)
        base = [("dedupe kernel", f"{-(-C // 128)} calls of <= 128 tokens", dedupe, dev_blob, args.decode_max),
                ("per-pair decode kernel", f"{-(-pairs // 512)} calls of <= 512 pairs", dec, pair_blob, args.pair_max)]
        for name, how, fn, b, cmax in base:
            if args.no_decode or C > cmax:
                continue
            t = pl.timed(f"C={C} moe, NKI {name} ({how})", fn, (*d, b), args.iters)
            if t == t and t_k == t_k:
                print(f"  -> {name}: {(t - (t_s - null)) * 1e3:.3f} ms without the readback reduction "
                      f"({(t - (t_s - null)) / net:.2f} x the prefill kernel)", flush=True)


if __name__ == "__main__":
    main()
