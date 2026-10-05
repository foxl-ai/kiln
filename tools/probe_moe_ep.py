"""The expert-parallel MoE kernel (kiln/kernels/moe_ep.py) on one NeuronCore at GLM-5.3-Flash's EP rank
shapes: 9 whole experts (288 / 32), hidden 4096, intermediate 2048, FP8 with 128 x 128 block scales as the
loader holds them after fit_e4m3_max (per-row gate_up and per-(input block, column) down scales).

    python tools/probe_moe_ep.py core [--experts 9] [--lanes 128 256] [--passes 9] [--bc 3] [--iters 10]
    python tools/probe_moe_ep.py full [--chunks 64 1024 4096] [--ranks 32] [--routing uniform|real <file>]

`core` times kiln_moe_ep_core: --passes passes of --lanes rows each (pass p on local expert p % experts),
against the host emulation of the kernel's arithmetic (moe_ep.expert_out) and an fp32 reference; p50 of
synchronous calls of a graph that reads back y.float().sum(0), minus the same reduction of an input of y's
shape. `full` times the whole kernel (plan, passes, combine) for one rank of --ranks on C-row chunks.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def ep_experts(El: int, H: int = 4096, I: int = 2048, seed: int = 0, block: bool = False):
    """El whole experts as the loader holds them under EP: e4m3fn codes of 128 x 128 blocks (block max 448)
    of normal weights, per-row block scales, then models/quant.fit_e4m3_max (trn1's e4m3 stops at 240):
    (w_gu [El, 2I, H] fp8, s_gu [El, 2I, H / 128], w_down [El, I, H] fp8 stored input-first, s_down
    [El, I / 128, H])."""
    from kiln.models.quant import FP8, fit_e4m3_max

    g = torch.Generator().manual_seed(seed)

    def blocks(rows, cols):
        w = torch.randn(rows, cols, generator=g) * 0.02
        bs = w.abs().view(rows // 128, 128, cols // 128, 128).amax((1, 3)) / 448.0
        cw = (w / bs.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(FP8)
        return fit_e4m3_max(cw, bs.repeat_interleave(128, 0), 240.0, 128 if block else None)

    out = [[], [], [], []]
    for _ in range(El):
        wgu, sgu = blocks(2 * I, H)
        wd, sd = blocks(H, I)  # checkpoint order [out H, in I], scale [H, I / 128]
        for lst, t in zip(out, (wgu, sgu, wd.T.contiguous(), sd.T.contiguous())):
            lst.append(t)
    return tuple(torch.stack(t) for t in out)


def reference_expert(x, w_gu, s_gu, w_down, s_down, lim):
    from kiln.models.quant import dequant, dequant_t

    I = w_gu.shape[0] // 2
    gu = x.float() @ dequant(w_gu, s_gu, torch.float32).T
    g, u = gu[:, :I].clamp(max=lim), gu[:, I:].clamp(min=-lim, max=lim)
    return (torch.nn.functional.silu(g) * u) @ dequant_t(w_down, s_down, torch.float32)


def core(args) -> None:
    import profile_layer as pl

    pl.setup_device()
    from kiln.kernels import moe_ep

    H = 4096
    t0 = time.time()
    ws = ep_experts(args.experts, H)
    blob = moe_ep.pack(*ws, tiles=getattr(args, "fit", "row") == "block")
    back = moe_ep.unpack(blob)
    same = all(torch.equal(a.view(torch.uint8) if a.dtype != torch.float32 else a,
                           b.view(torch.uint8) if b.dtype != torch.float32 else b) for a, b in zip(ws, back))
    print(f"{args.experts} experts built and packed in {time.time() - t0:.0f} s; unpack(pack()) == input: {same}; "
          f"blob bytes per expert {sum(t[0].numel() * t.element_size() for t in blob.values()) / 1e6:.2f} MB",
          flush=True)
    dev = {k: v.to(pl.DEV) for k, v in blob.items()}
    null = pl.timed("null graph", lambda h: h + 1, (torch.zeros(4, H, dtype=torch.bfloat16).to(pl.DEV),), args.iters)
    g = torch.Generator().manual_seed(1)
    for LW in args.lanes:
        NP = args.passes
        xe = torch.randn(NP * LW, H, generator=g).bfloat16()
        ex = (torch.arange(NP) % args.experts).view(1, NP).to(torch.int32)
        outs = {}
        dargs = (xe.to(pl.DEV), ex.to(pl.DEV), dev["gu"], dev["sgu"], dev["dn"], dev["sdn"])
        for bc in args.bc:
            def call(xe_, ex_, gu_, sgu_, dn_, sdn_, bc=bc, LW=LW):
                return moe_ep.core(xe_, ex_, dict(gu=gu_, sgu=sgu_, dn=dn_, sdn=sdn_), LW, act=1, lim=10.0, bc=bc)

            t = time.perf_counter()
            try:
                y = torch.compile(call, **pl.OPTS)(*dargs).cpu().float()
            except Exception as e:
                print(f"  LW={LW} bc={bc}: kernel FAILED {type(e).__name__}: {str(e)[:600]}", flush=True)
                continue
            print(f"  LW={LW} bc={bc}: first call (compile + load) {time.perf_counter() - t:.1f} s", flush=True)
            for b0, y0 in outs.items():
                print(f"    bc={bc} against bc={b0}: bit-identical {bool(torch.equal(y, y0))}", flush=True)
            outs[bc] = y
            errs, refs = [], []
            for p in range(min(NP, args.check)):
                e = p % args.experts
                rows = xe[p * LW:(p + 1) * LW]
                emu = moe_ep.expert_out(rows, ws[0][e], ws[1][e], ws[2][e], ws[3][e], 1, 10.0)
                ref = reference_expert(rows, ws[0][e], ws[1][e], ws[2][e], ws[3][e], 10.0)
                got = y[p * LW:(p + 1) * LW]
                sc = ref.abs().max().item()
                errs.append(((got - emu).abs().max().item() / sc, (emu - ref).abs().max().item() / sc,
                             (got - ref).abs().max().item() / sc, int((got != emu).sum())))
            for p, (ke, er, kr, nd) in enumerate(errs):
                print(f"    pass {p}: kernel vs emulation rel {ke:.4f} ({nd} of {LW * H} values differ), emulation vs "
                      f"fp32 reference {er:.4f}, kernel vs reference {kr:.4f}", flush=True)

            def kern(xe_, ex_, gu_, sgu_, dn_, sdn_, bc=bc, LW=LW):
                return moe_ep.core(xe_, ex_, dict(gu=gu_, sgu=sgu_, dn=dn_, sdn=sdn_), LW, act=1, lim=10.0,
                                   bc=bc).float().sum(0)

            tk = pl.timed(f"LW={LW} bc={bc}: {NP} passes", kern, dargs, args.iters)
            if args.save_inputs:
                path = f"{args.save_inputs}.LW{LW}.bc{bc}.pt"
                torch.save(dict(xe=xe, ex=ex, gu=blob["gu"], sgu=blob["sgu"], dn=blob["dn"], sdn=blob["sdn"]), path)
                import glob
                cache = "/root/.cache/neuron_libtorch/neuron/compile_cache"
                hs = sorted(glob.glob(f"{cache}/*/fxgraph.txt"), key=os.path.getmtime)
                hs = [h for h in hs if "L_xe__" in open(h).read() and "sum" in open(h).read()]
                print(f"    inputs saved to {path}; graph {os.path.basename(os.path.dirname(hs[-1])) if hs else '?'}",
                      flush=True)
            ts = pl.timed(f"  the same sum over [{NP * LW}, {H}]", lambda a: a.float().sum(0), (xe.to(pl.DEV),),
                          args.iters)
            if tk == tk:
                net = tk - (ts - null)
                fl = 2 * NP * LW * (2 * 2048 * H + 2048 * H)
                print(f"  -> LW={LW} bc={bc}: {net * 1e3:.3f} ms for {NP} passes, {net / NP * 1e6:.1f} us per pass, "
                      f"{fl / net / 1e12:.2f} TFLOPS, {NP * 25.17e6 / net / 1e9:.0f} GB/s of expert weights", flush=True)


def full(args) -> None:
    import profile_layer as pl

    pl.setup_device()
    from kiln.kernels import moe_ep

    H, E, K, R, G = 4096, args.experts_total, 8, args.ranks, args.group
    # --group g: ranks in groups of g share E g / R experts, each rank holding a 1 / g intermediate slice of each
    # (g 1: whole experts). Every rank of a group runs the same pairs, so "rank" below is the group.
    El = E * G // R
    ws = ep_experts(El, H, 2048 // G, block=args.fit == "block")
    blob = moe_ep.pack(*ws, tiles=getattr(args, "fit", "row") == "block")
    dev = {k: v.to(pl.DEV) for k, v in blob.items()}
    owner = torch.arange(E) // El
    R = R // G
    real = torch.load(args.routing_file) if args.routing_file else None
    null = pl.timed("null graph", lambda h: h + 1, (torch.zeros(4, H, dtype=torch.bfloat16).to(pl.DEV),), args.iters)
    g = torch.Generator().manual_seed(1)
    for C in args.chunks:
        x = torch.randn(C, H, generator=g).bfloat16()
        if real is not None:  # tools/ep_routing.py run --save: C / 4 consecutive tokens of each named sequence
            names = real["names"]
            per = C // len(args.sequences)
            parts = []
            for spec in args.sequences:  # name[:first token]
                n, _, o = spec.partition(":")
                o = int(o or 0)
                parts.append(real["topi"][args.layer][names.index(n)][o:o + per].long())
            topi = torch.cat(parts)
        else:
            topi = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(C)])
        if args.routing == "hot":  # every token on experts 0 .. K - 1 (local on rank 0): the most passes
            topi = torch.arange(K).repeat(C, 1)
        elif args.routing == "skew":  # a third of the tokens' first expert is local expert 1 of this rank
            hot = int((owner == args.rank).nonzero()[1])
            sel = torch.rand(C, generator=g) < 1 / 3
            for t in sel.nonzero().flatten().tolist():
                if hot not in topi[t].tolist():
                    topi[t, 0] = hot
        topv = (torch.rand(C, K, generator=g) + 0.1).bfloat16()
        rank = args.rank
        if rank < 0:  # the busiest rank of this routing under the contiguous placement
            rank = int(torch.bincount(owner[topi.flatten()], minlength=R).argmax())
        lmap = moe_ep.local_map(owner, rank)
        loc = lmap.view(-1)[topi]
        print(f"{'group' if G > 1 else 'rank'} {rank} of {R}: {El} local experts of {2048 // G} intermediate rows, "
              f"routing {args.routing_file or args.routing}"
              f"{f' layer {args.layer}' if real is not None else ''}", flush=True)
        n = torch.bincount(loc.flatten(), minlength=El + 1)[:El]
        LW = moe_ep.SMALL_LW if moe_ep.uses_small(C) else moe_ep.lanes(C)
        PM = moe_ep.max_passes(C, K, El, LW)
        print(f"C={C}: {int(n.sum())} local pairs (largest expert {int(n.max())}, {int((n == 0).sum())} with none), "
              f"LW={LW}, {El} first passes + {int((-(-n // LW) - 1).clamp(min=0).sum())} overflow passes (bound {PM})",
              flush=True)
        bn = [k for k in ("gu", "sgu", "dn", "sdn", "dsg", "dsd", "tsg", "tsd") if k in dev]
        d = (x.to(pl.DEV), topv.to(pl.DEV), topi.to(torch.int32).to(pl.DEV), lmap.to(pl.DEV), *[dev[k] for k in bn])
        print(f"  scales: {'one per tile (tile-scale form)' if moe_ep.tile_scales(blob) else 'per row'}", flush=True)

        def call(x, topv, topi, lmap, *bt):
            return moe_ep.moe_ep(x, topv, topi, dict(zip(bn, bt)), lmap, 1, 10.0)

        def unused(x_, topv_, topi_, lmap_, gu_, sgu_, dn_, sdn_):
            return moe_ep.moe_ep(x_, topv_, topi_, dict(gu=gu_, sgu=sgu_, dn=dn_, sdn=sdn_), lmap_, 1, 10.0)

        t = time.perf_counter()
        try:
            got = torch.compile(call, **pl.OPTS)(*d).cpu().float()
        except Exception as e:
            print(f"  kernel FAILED {type(e).__name__}: {str(e)[:1500]}", flush=True)
            continue
        print(f"  first call (compile + load) {time.perf_counter() - t:.1f} s", flush=True)
        emu = moe_ep.emulate(x, topv, topi, lmap, *ws, 1, 10.0).float()
        sc = emu.abs().max().item()
        zero_rows = (loc >= El).all(1)
        print(f"  kernel vs emulation: max abs {(got - emu).abs().max().item():.3e} (rel {(got - emu).abs().max().item() / sc:.4f}), "
              f"{int((got != emu).sum())} of {got.numel()} values differ; rows without a local pair all zero: "
              f"{bool((got[zero_rows] == 0).all())} ({int(zero_rows.sum())} rows)", flush=True)

        def kern(*a):
            return call(*a).float().sum(0)

        tk = pl.timed(f"C={C} EP kernel", kern, d, args.iters)
        if args.save_inputs:
            import glob
            path = f"{args.save_inputs}.C{C}.pt"
            torch.save(dict(x=x, topv=topv, topi=topi.to(torch.int32), lmap=lmap, **blob), path)
            cache = "/root/.cache/neuron_libtorch/neuron/compile_cache"
            hs = [h for h in sorted(glob.glob(f"{cache}/*/fxgraph.txt"), key=os.path.getmtime)
                  if "L_lmap_" in open(h).read() and "sum" in open(h).read()]
            print(f"    inputs saved to {path}; graph {os.path.basename(os.path.dirname(hs[-1])) if hs else '?'}",
                  flush=True)
        ts = pl.timed(f"  the same sum over [{C}, {H}]", lambda a: a.float().sum(0), (x.to(pl.DEV),), args.iters)
        if tk == tk:
            print(f"  -> C={C}: {(tk - (ts - null)) * 1e3:.3f} ms without the readback reduction", flush=True)


def hybrid(args) -> None:
    """The hot-expert hybrid on one rank, real routing (tools/ep_routing.py): the batch's NH most-loaded experts
    are split into S slices of I / S rows each spread over the ranks (rank r of group r // S holds slice r % S of
    its group's experts, the groups' loads balanced greedily), the other experts are whole, El = (E - NH) / R per
    rank. Times the busiest rank's cold part (whole experts, the kernel moe_ep() picks for C rows) and the busiest
    group's hot part (its slices: every pair of their experts) separately, each against moe_ep.emulate."""
    import profile_layer as pl

    pl.setup_device()
    from kiln.kernels import moe_ep

    H, E, K, R, NH, S = 4096, 288, 8, args.ranks, args.hot, args.slices
    I = 2048
    El, Es = (E - NH) // R, NH * S // R
    real = torch.load(args.routing_file)
    names = real["names"]
    g = torch.Generator().manual_seed(1)
    C = args.rows
    per = C // len(args.sequences)
    parts = []
    for spec in args.sequences:
        n, _, o = spec.partition(":")
        parts.append(real["topi"][args.layer][names.index(n)][int(o or 0):int(o or 0) + per].long())
    topi = torch.cat(parts)
    topv = (torch.rand(C, K, generator=g) + 0.1).bfloat16()
    x = torch.randn(C, H, generator=g).bfloat16()
    load = torch.bincount(topi.flatten(), minlength=E)
    order = torch.argsort(load, descending=True)
    hot = order[:NH].tolist()
    cold = sorted(order[NH:].tolist())
    # cold: contiguous among the cold ids; hot: NH / (R / S) groups of experts, balanced greedily by load
    ng = R // S
    per_g = NH // ng
    groups, tot = [[] for _ in range(ng)], [0] * ng
    for h in hot:
        q = min((q for q in range(ng) if len(groups[q]) < per_g), key=lambda q: tot[q])
        groups[q].append(h)
        tot[q] += int(load[h])
    owner_c = torch.full((E,), -1, dtype=torch.long)
    for i, e in enumerate(cold):
        owner_c[e] = i // El
    cold_load = torch.zeros(R, dtype=torch.long).index_add_(0, owner_c[topi.flatten()].clamp(min=0),
                                                             (owner_c[topi.flatten()] >= 0).long())
    rc = int(cold_load.argmax())
    gq = int(torch.tensor(tot).argmax())
    print(f"layer {args.layer}, C={C}: hot {NH} experts carry {int(load[hot].sum())} of {C * K} pairs; cold busiest rank "
          f"{rc}: {int(cold_load[rc])} pairs (mean {int(cold_load.float().mean())}); hot group loads {tot} "
          f"(busiest {gq}: {[int(load[h]) for h in groups[gq]]})", flush=True)
    null = pl.timed("null graph", lambda h: h + 1, (torch.zeros(4, H, dtype=torch.bfloat16).to(pl.DEV),), args.iters)
    ts_ = pl.timed(f"  the sum over [{C}, {H}]", lambda a: a.float().sum(0), (x.to(pl.DEV),), args.iters)
    # cold part: rank rc's whole experts
    wc = ep_experts(El, H, I, seed=5)
    mine = [e for e in cold if owner_c[e] == rc]
    lm_c = torch.full((E + 1,), El, dtype=torch.int32)
    for i, e in enumerate(mine):
        lm_c[e] = i
    # hot part: group gq's experts, slice 0 of each (every slice of a group runs the same pairs)
    ws = ep_experts(Es, H, I // S, seed=6)
    lm_s = torch.full((E + 1,), Es, dtype=torch.int32)
    for i, e in enumerate(groups[gq]):
        lm_s[e] = i
    for what, wts_, lm in (("cold", wc, lm_c), ("hot", ws, lm_s)):
        blob = moe_ep.pack(*wts_)
        dev = {k: v.to(pl.DEV) for k, v in blob.items()}
        lmap = lm.view(1, -1)
        d = (x.to(pl.DEV), topv.to(pl.DEV), topi.to(torch.int32).to(pl.DEV), lmap.to(pl.DEV), *[dev[k] for k in
             ("gu", "sgu", "dn", "sdn", "dsg", "dsd")])

        def call(x, topv, topi, lmap, gu, sgu, dn, sdn, dsg, dsd, what=what):
            small = what == "cold" and args.cold_small
            old = (moe_ep.SMALL_ROWS, moe_ep.SMALL_LW, moe_ep.LW_ENV)
            moe_ep.SMALL_ROWS = 1 << 20 if small else 0
            moe_ep.SMALL_LW = args.cold_lw if small else moe_ep.SMALL_LW
            moe_ep.LW_ENV = str(args.hot_lw) if what == "hot" else old[2]
            try:
                return moe_ep.moe_ep(x, topv, topi, dict(gu=gu, sgu=sgu, dn=dn, sdn=sdn, dsg=dsg, dsd=dsd), lmap, 1,
                                     10.0)
            finally:
                moe_ep.SMALL_ROWS, moe_ep.SMALL_LW, moe_ep.LW_ENV = old

        try:
            got = torch.compile(call, **pl.OPTS)(*d).cpu().float()
        except Exception as e:
            print(f"  {what}: kernel FAILED {type(e).__name__}: {str(e)[:1500]}", flush=True)
            continue
        emu = moe_ep.emulate(x, topv, topi, lmap, *wts_, 1, 10.0, small=(what == "cold" and args.cold_small)).float()
        sc = emu.abs().max().item()
        print(f"  {what}: kernel vs emulation rel {(got - emu).abs().max().item() / sc:.4f}", flush=True)
        tk = pl.timed(f"  {what} part", lambda *a, call=call: call(*a).float().sum(0), d, args.iters)
        if tk == tk:
            print(f"  -> {what}: {(tk - (ts_ - null)) * 1e3:.3f} ms", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("core")
    c.add_argument("--experts", type=int, default=9)
    c.add_argument("--lanes", type=int, nargs="+", default=[128, 256])
    c.add_argument("--passes", type=int, default=9)
    c.add_argument("--bc", type=int, nargs="+", default=[3])
    c.add_argument("--check", type=int, default=2, help="passes checked against the emulation")
    c.add_argument("--iters", type=int, default=10)
    c.add_argument("--save-inputs", default=None, help="torch.save each timed graph's inputs for tools/prof_engines.py")
    f = sub.add_parser("full")
    f.add_argument("--chunks", type=int, nargs="+", default=[128, 1024, 4096])
    f.add_argument("--experts-total", type=int, default=288)
    f.add_argument("--ranks", type=int, default=32)
    f.add_argument("--rank", type=int, default=0, help="-1: the busiest rank of the routing")
    f.add_argument("--group", type=int, default=1, help="ranks per expert group (each a 1 / g slice of its experts)")
    f.add_argument("--fit", default="row", choices=["row", "block"],
                   help="e4m3 fit per (row, block) or per 128 x 128 block (block-constant scales: the tile-scale form)")
    f.add_argument("--routing", default="uniform")
    f.add_argument("--routing-file", default=None, help="tools/ep_routing.py run --save output")
    f.add_argument("--layer", type=int, default=3)
    f.add_argument("--sequences", nargs="+", default=["random0:0", "random1:0", "random0:2048", "random1:2048"],
                   help="with --routing-file: name[:first token] of the C / n consecutive tokens of each part of the batch "
                        "(the DP-attention groups' chunks)")
    f.add_argument("--iters", type=int, default=10)
    f.add_argument("--save-inputs", default=None)
    h = sub.add_parser("hybrid")
    h.add_argument("--routing-file", required=True)
    h.add_argument("--layer", type=int, default=20)
    h.add_argument("--rows", type=int, default=4096)
    h.add_argument("--sequences", nargs="+", default=["random0:0", "random1:0", "random0:2048", "random1:2048"])
    h.add_argument("--ranks", type=int, default=32)
    h.add_argument("--hot", type=int, default=32)
    h.add_argument("--slices", type=int, default=8)
    h.add_argument("--cold-small", action="store_true", help="the cold part on kiln_moe_ep_small")
    h.add_argument("--cold-lw", type=int, default=64)
    h.add_argument("--hot-lw", type=int, default=512)
    h.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()
    if a.cmd == "core":
        core(a)
    elif a.cmd == "hybrid":
        hybrid(a)
    else:
        full(a)


if __name__ == "__main__":
    main()
