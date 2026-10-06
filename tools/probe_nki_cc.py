"""Collectives issued from inside an NKI kernel (nki.collectives, nki 0.6.0) on trn1 (NeuronCore-v2), inside an LNL
graph: do they compile, run at N ranks, give the right values, cost what the XLA collectives cost, survive a reload
of the cached NEFF in a later process, and let the kernel's own engines compute while one is in flight?

Why: neuronx-cc 2.27 schedules no independent compute inside an XLA collective's window (docs/neuron-notes.md,
"Accelerator utilization of the serving graphs": tools/probe_overlap.py, 9 graph shapes, 3 scheduler options), and
in the 4096-token prefill call every engine sits idle with a collective in flight for ~29% of the call. A collective
issued from inside a kernel is the one remaining route to overlap: the kernel interleaves it with its own tiles.
nki 0.6.0 ships nki.collectives (all_reduce, all_gather, reduce_scatter, all_to_all, collective_permute(_implicit),
rank_id, ReplicaGroup; nki/collectives/__init__.pyi in the SDK 2.32 venv: "Tensors can reside on either HBM or SBUF",
"priority ... NeuronCore-v4+ only"); nothing in it names a NeuronCore-v2 restriction for these four, and nothing
documents that they run on trn1 either.

    python tools/probe_nki_cc.py [--ranks 32] [--rows 128] [--hidden 4096] [--cases ag,rs,ar,xag,xrs,ovl] [--iters 20]

Cases (each one graph whose output is reduced to a scalar in the graph, so only 4 bytes come back; p50 of --iters
synchronous calls on rank 0; the value is checked against the host's sum):
  xag   the served gather (models/decoder.py _sp_gather): every rank's [r, H] rows by a zero-padded XLA all-reduce
  ag    the same rows by nki.collectives.all_gather inside a kernel (HBM -> HBM)
  xrs   the served block output reduction: an XLA reduce-scatter of [N r, H] -> [r, H]
  rs    nki.collectives.reduce_scatter inside a kernel
  ar    nki.collectives.all_reduce inside a kernel, [N r, H]
  ovl   overlap inside one kernel: modes g (gathers only, --chunks all_gathers of row slices), c (--reps x 32 bf16
        matmuls on SBUF-resident operands, no dependency on the gathers), gc (gathers issued, then the compute), dep
        (each chunk's rows read and reduced right after its gather, while the later chunks' gathers are in flight).
        If gc is near max(g, c) the kernel's engines run while its collectives are in flight.
Run it twice: the second process loads the first one's NEFFs from the compile cache, the case that broke for XLA
all-gather graphs ("A cached all-gather NEFF breaks in the next process").
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:  # the Neuron venv
    import nki
    import nki.collectives as ncc
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

REV = 1  # bump after editing a kernel below: LNL's cache key does not include NKI kernel source

if nki is not None:
    BF16, F32 = nl.bfloat16, nl.float32

    def _group(world: int):
        ranks = []  # the NKI tracer takes a list literal, not list(range(...))
        for i in range(world):
            ranks.append(i)
        return ncc.ReplicaGroup([ranks])

    def _hbm_copy(dst, src, rows: int, H: int, dt):
        """src -> dst through SBUF, 128 rows at a time (a collective cannot read or write the kernel's IO tensors:
        neuronx-cc 2.27 birverifier checkCollective, "Collective instruction cannot read IO tensors")."""
        T = nl.ndarray((128, H), dtype=dt, buffer=nl.sbuf)
        for t in range(rows // 128):
            nisa.dma_copy(dst=T, src=src[t * 128:(t + 1) * 128, :])
            nisa.dma_copy(dst=dst[t * 128:(t + 1) * 128, :], src=T)

    @nki.jit
    def kiln_cc_ag(x, world: int, rev: int):
        """x [r, H] -> every rank's x, rank order, [world * r, H] (HBM -> HBM all_gather on private scratch)."""
        r, H = x.shape
        xs = nl.ndarray((r, H), dtype=x.dtype, buffer=nl.private_hbm)
        gs = nl.ndarray((world * r, H), dtype=x.dtype, buffer=nl.private_hbm)
        out = nl.ndarray((world * r, H), dtype=x.dtype, buffer=nl.shared_hbm)
        _hbm_copy(xs, x, r, H, x.dtype)
        ncc.all_gather(srcs=[xs], dsts=[gs], replica_group=_group(world), collective_dim=0)
        _hbm_copy(out, gs, world * r, H, x.dtype)
        return out

    @nki.jit
    def kiln_cc_agio(x, world: int, rev: int):
        """kiln_cc_ag with the all_gather writing the kernel's output (an IO tensor) directly."""
        r, H = x.shape
        xs = nl.ndarray((r, H), dtype=x.dtype, buffer=nl.private_hbm)
        out = nl.ndarray((world * r, H), dtype=x.dtype, buffer=nl.shared_hbm)
        _hbm_copy(xs, x, r, H, x.dtype)
        ncc.all_gather(srcs=[xs], dsts=[out], replica_group=_group(world), collective_dim=0)
        return out

    @nki.jit
    def kiln_cc_agsh(x, world: int, rev: int):
        """kiln_cc_ag with the source an internal shared_hbm copy and the destination the kernel's output itself (no
        copy-out), if the verifier lets a collective write an IO tensor from a non-IO one."""
        r, H = x.shape
        xs = nl.ndarray((r, H), dtype=x.dtype, buffer=nl.shared_hbm)
        out = nl.ndarray((world * r, H), dtype=x.dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=xs, src=x)
        ncc.all_gather(srcs=[xs], dsts=[out], replica_group=_group(world), collective_dim=0)
        return out

    @nki.jit
    def kiln_cc_agsb(x, world: int, rev: int):
        """x [r <= 128, H] loaded to SBUF and all-gathered SBUF -> HBM scratch (mixing is not allowed, so SBUF ->
        SBUF along the free axis would be [r, world H]: too wide at 32 ranks; this case tries SBUF -> SBUF on a
        [r, H / world]-wide slice of each... kept simple: the SBUF form with collective_dim 1)."""
        r, H = x.shape
        T = nl.ndarray((r, H), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=T, src=x)
        G = nl.ndarray((r, world * H), dtype=x.dtype, buffer=nl.sbuf)
        ncc.all_gather(srcs=[T], dsts=[G], replica_group=_group(world), collective_dim=1)
        out = nl.ndarray((r, world * H), dtype=x.dtype, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=G)
        return out

    @nki.jit
    def kiln_cc_rs(y, world: int, rev: int):
        """y [world * r, H] summed over the ranks, block rank of world row blocks: [r, H]."""
        R, H = y.shape
        ys = nl.ndarray((R, H), dtype=y.dtype, buffer=nl.private_hbm)
        os_ = nl.ndarray((R // world, H), dtype=y.dtype, buffer=nl.private_hbm)
        out = nl.ndarray((R // world, H), dtype=y.dtype, buffer=nl.shared_hbm)
        _hbm_copy(ys, y, R, H, y.dtype)
        ncc.reduce_scatter(srcs=[ys], dsts=[os_], replica_group=_group(world), collective_dim=0, op=nl.add)
        _hbm_copy(out, os_, R // world, H, y.dtype)
        return out

    @nki.jit
    def kiln_cc_ar(y, world: int, rev: int):
        R, H = y.shape
        ys = nl.ndarray((R, H), dtype=y.dtype, buffer=nl.private_hbm)
        os_ = nl.ndarray((R, H), dtype=y.dtype, buffer=nl.private_hbm)
        out = nl.ndarray((R, H), dtype=y.dtype, buffer=nl.shared_hbm)
        _hbm_copy(ys, y, R, H, y.dtype)
        ncc.all_reduce(srcs=[ys], dsts=[os_], replica_group=_group(world), op=nl.add)
        _hbm_copy(out, os_, R, H, y.dtype)
        return out

    @nki.jit
    def kiln_cc_copy(y, rev: int):
        """The copies the collective kernels add around their collective, alone (their cost to subtract)."""
        R, H = y.shape
        ys = nl.ndarray((R, H), dtype=y.dtype, buffer=nl.private_hbm)
        out = nl.ndarray((R, H), dtype=y.dtype, buffer=nl.shared_hbm)
        _hbm_copy(ys, y, R, H, y.dtype)
        _hbm_copy(out, ys, R, H, y.dtype)
        return out

    @nki.jit
    def kiln_cc_ovl(x, a, b, world: int, chunks: int, reps: int, mode: int, rev: int):
        """mode bits: 1 gathers (chunks all_gathers of x's row slices), 2 compute (reps x 32 matmuls of a [128, 128]
        stationary against b [128, 512], summed in PSUM, independent of the gathers), 4 dependent (each gathered
        chunk read back 128 rows at a time and summed along H, right after its gather). Returns acc [128, 512] fp32
        (the compute, or zeros) and red [world * r, 1] fp32 (the dependent sums, or zeros) and the gathered rows."""
        r, H = x.shape
        rc = r // chunks
        xs = nl.ndarray((r, H), dtype=x.dtype, buffer=nl.private_hbm)
        _hbm_copy(xs, x, r, H, x.dtype)
        g = nl.ndarray((world * r, H), dtype=x.dtype, buffer=nl.shared_hbm)
        acc_o = nl.ndarray((128, 512), dtype=F32, buffer=nl.shared_hbm)
        red_o = nl.ndarray((world * r, 1), dtype=F32, buffer=nl.shared_hbm)
        outs = []
        if mode & 1:
            for c in range(chunks):
                oc = nl.ndarray((world * rc, H), dtype=x.dtype, buffer=nl.private_hbm)
                ncc.all_gather(srcs=[xs[c * rc:(c + 1) * rc, :]], dsts=[oc], replica_group=_group(world),
                               collective_dim=0)
                outs.append(oc)
        A = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
        B = nl.ndarray((128, 512), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=A, src=a)
        nisa.dma_copy(dst=B, src=b)
        P = nl.ndarray((128, 512), dtype=F32, buffer=nl.psum)
        S = nl.ndarray((128, 512), dtype=F32, buffer=nl.sbuf)
        nisa.memset(dst=S, value=0.0)
        if mode & 2:
            for i in range(reps):
                for j in range(32):
                    nisa.nc_matmul(dst=P, stationary=A, moving=B, accumulate=(j > 0))
                nisa.tensor_tensor(dst=S, data1=S, data2=P, op=nl.add)
        nisa.dma_copy(dst=acc_o, src=S)
        T = nl.ndarray((128, H), dtype=x.dtype, buffer=nl.sbuf)
        Rd = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
        if mode & 4:
            for c in range(chunks):
                oc = outs[c]
                for t in range(world * rc // 128):
                    nisa.dma_copy(dst=T, src=oc[t * 128:(t + 1) * 128, :])
                    nisa.tensor_reduce(dst=Rd, op=nl.add, data=T, axis=1)
                    nisa.dma_copy(dst=red_o[c * world * rc + t * 128:c * world * rc + (t + 1) * 128, :], src=Rd)
        else:
            nisa.memset(dst=Rd, value=0.0)
            for t in range(world * r // 128):
                nisa.dma_copy(dst=red_o[t * 128:(t + 1) * 128, :], src=Rd)
        if mode & 1:  # the gathered rows, chunk by chunk, into one output (rank-major inside each chunk)
            for c in range(chunks):
                for t in range(world * rc // 128):
                    nisa.dma_copy(dst=T, src=outs[c][t * 128:(t + 1) * 128, :])
                    nisa.dma_copy(dst=g[c * world * rc + t * 128:c * world * rc + (t + 1) * 128, :], src=T)
        return acc_o, red_o, g


def _p50(fn, iters: int) -> float:
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def rank_main(rank: int, port: int, world: int, r: int, H: int, cases: list[str], iters: int, chunks: int,
              reps: int) -> None:
    import profile_layer as pl
    from kiln.engine import tp

    tp.neuron_env(rank, port, int(os.environ.get("KILN_CORE_BASE", "0")))
    tp.init_rank(rank, world, port)
    pl.setup_device()
    import torch.distributed as dist
    import torch.distributed._functional_collectives as funcol
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    grp = dist.group.WORLD
    bf = torch.bfloat16
    say = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    own = [torch.randn(r, H, generator=torch.Generator().manual_seed(100 + m)).to(bf) for m in range(world)]
    rows_all = torch.cat(own).float()
    full = [torch.randn(world * r, H, generator=torch.Generator().manual_seed(m)).to(bf) for m in range(world)]
    fsum = sum(f.float() for f in full)
    x = own[rank].to(pl.DEV)
    y = full[rank].to(pl.DEV)
    onehot = torch.zeros(world, dtype=bf)
    onehot[rank] = 1
    oh = onehot.to(pl.DEV)

    def check(name, got, ref, tol):
        err = (got.float() - ref).abs().max().item()
        bad = torch.zeros(1)
        bad[0] = float(err > tol)
        dist.all_reduce(bad)  # gloo default group: how many ranks are wrong
        say(f"  {name:<34} max |err| {err:.3e} on rank 0, ranks wrong: {int(bad.item())}")

    def run(name, f, inp, ref, tol):
        try:
            c = torch.compile(f, **pl.OPTS)
            t = time.perf_counter()
            got = c(*inp)
            got = got[0] if isinstance(got, tuple) else got
            got = got.cpu()
            first = time.perf_counter() - t
            if ref is not None:
                check(name, got, ref, tol)
            cs = torch.compile(lambda *a: _sum(f(*a)), **pl.OPTS)
            cs(*inp).cpu()
            ms = _p50(lambda: cs(*inp).cpu(), iters)
            say(f"  {name:<34} first call {first:6.1f} s, p50 {ms:8.3f} ms (sum readback)")
        except Exception as e:  # report and go on: the point is which forms exist on trn1
            say(f"  {name:<34} FAILED: {type(e).__name__}: {str(e)[:1500]}")

    def _sum(o):
        if isinstance(o, tuple):
            s = o[0].float().sum()
            for t in o[1:]:
                s = s + t.float().sum()
            return s
        return o.float().sum()

    say(f"ranks {world}, rows per rank {r}, hidden {H}, bf16; rev {REV}")
    if "xag" in cases:
        run("xag: XLA zero-padded all-reduce", lambda v: funcol.all_reduce(
            (v.unsqueeze(0) * oh.view(world, 1, 1)).reshape(-1, H), "sum", grp), (x,), rows_all, 0.0)
    if "ag" in cases:
        run("ag: nki all_gather", lambda v: wrap_nki(kiln_cc_ag)[1](x=v, world=world, rev=REV), (x,), rows_all, 0.0)
    if "agio" in cases:
        run("agio: nki all_gather into the output", lambda v: wrap_nki(kiln_cc_agio)[1](x=v, world=world, rev=REV), (x,),
            rows_all, 0.0)
    if "agsh" in cases:
        run("agsh: nki all_gather, shared src, IO dst", lambda v: wrap_nki(kiln_cc_agsh)[1](x=v, world=world, rev=REV),
            (x,), rows_all, 0.0)
    if "agsb" in cases:
        xs_ = own[rank][:, :H // world].contiguous().to(pl.DEV)
        ref = torch.cat([o[:, :H // world] for o in own], dim=1).float()
        run("agsb: nki all_gather SBUF, dim 1", lambda v: wrap_nki(kiln_cc_agsb)[1](x=v, world=world, rev=REV), (xs_,),
            ref, 0.0)
    if any(c.startswith("sp") for c in cases):  # kiln/kernels/sp_gather.py as the engine calls it
        from kiln.kernels import sp_gather as spg

        for c in cases:
            if not c.startswith("sp"):
                continue
            gs = world if c[2:3] != "g" else 8  # spg<k>: inside 8-rank groups (the attention groups at DP 4)
            k = int(c[3:] if c[2:3] == "g" else c[2:])
            spg.CHUNKS = k
            g0 = (rank // gs) * gs
            ref = torch.cat(own[g0:g0 + gs]).float()
            run(f"{c}: sp_gather {gs} ranks, chunks {k}", lambda v, gs_=gs: spg.gather(v, world, gs_), (x,), ref, 0.0)
        if world >= 8:
            g0 = (rank // 8) * 8
            oh8 = torch.zeros(8, dtype=bf)
            oh8[rank - g0] = 1
            pgs = [dist.new_group(list(range(g * 8, g * 8 + 8))) for g in range(world // 8)]
            mine = pgs[rank // 8]
            o8 = oh8.to(pl.DEV)
            run("xspg: XLA zero-padded group all-reduce", lambda v: funcol.all_reduce(
                (v.unsqueeze(0) * o8.view(8, 1, 1)).reshape(-1, H), "sum", mine), (x,),
                torch.cat(own[g0:g0 + 8]).float(), 0.0)
    for c in cases:  # K gathers in ONE graph (each of x + i, all consumed): the per-collective cost past the graph's
        # fixed per-execution cost, which a standalone graph pays once (docs/neuron-notes.md "Collectives across chips")
        if c.startswith("xagk") or c.startswith("spk"):
            from kiln.kernels import sp_gather as spg

            k = int(c[4:] if c.startswith("xagk") else c[3:])
            spg.CHUNKS = 1

            def one(v, i):
                v = v + i
                if c.startswith("xagk"):
                    return funcol.all_reduce((v.unsqueeze(0) * oh.view(world, 1, 1)).reshape(-1, H), "sum", grp)
                return spg.gather(v, world, world)

            def many(v):
                out = one(v, 0).float()
                for i in range(1, k):
                    out = out + one(v, i).float()
                return out

            run(f"{c}: {k} gathers in one graph", many, (x,), None, 0.0)
    if "xrs" in cases:
        run("xrs: XLA reduce-scatter", lambda v: funcol.reduce_scatter_tensor(v, "sum", 0, grp), (y,),
            fsum[rank * r:(rank + 1) * r], 2.0)
    if "rs" in cases:
        run("rs: nki reduce_scatter", lambda v: wrap_nki(kiln_cc_rs)[1](y=v, world=world, rev=REV), (y,),
            fsum[rank * r:(rank + 1) * r], 2.0)
    if "xar" in cases:
        run("xar: XLA all-reduce", lambda v: funcol.all_reduce(v, "sum", grp), (y,), fsum, 2.0)
    if "copy" in cases:
        run("copy: the kernels' own copies", lambda v: wrap_nki(kiln_cc_copy)[1](y=v, rev=REV), (y,), full[rank].float(),
            0.0)
    if "ar" in cases:
        run("ar: nki all_reduce", lambda v: wrap_nki(kiln_cc_ar)[1](y=v, world=world, rev=REV), (y,), fsum, 2.0)
    if "ovl" in cases:
        a = (torch.randn(128, 128, generator=torch.Generator().manual_seed(7)) * 0.05).to(bf).to(pl.DEV)
        b = (torch.randn(128, 512, generator=torch.Generator().manual_seed(8)) * 0.05).to(bf).to(pl.DEV)
        rc = r // chunks
        gref = torch.cat([torch.cat([o[c * rc:(c + 1) * rc] for o in own]) for c in range(chunks)]).float()
        for mode, label in ((1, "g"), (2, "c"), (3, "gc"), (5, "dep"), (7, "gc+dep")):
            ref = gref if mode & 1 else None
            run(f"ovl {label} (chunks {chunks}, reps {reps})",
                lambda v, a_, b_, m=mode: _pick(wrap_nki(kiln_cc_ovl)[1](x=v, a=a_, b=b_, world=world, chunks=chunks,
                                                                         reps=reps, mode=m, rev=REV), m),
                (x, a, b), ref, 0.0)


def _pick(o, mode):
    """The gathered rows first when the mode gathers (checked), the whole tuple otherwise."""
    acc, red, g = o
    return (g, acc, red) if mode & 1 else (acc, red, g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, default=32)
    ap.add_argument("--rows", type=int, default=128, help="rows each rank contributes")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--cases", default="xag,ag,xrs,rs,ar,ovl")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()
    import multiprocessing as mp

    from kiln.engine import tp

    port = tp.free_port()
    ctx = mp.get_context("spawn")
    cases = args.cases.split(",")
    ps = [ctx.Process(target=rank_main, args=(r, port, args.ranks, args.rows, args.hidden, cases, args.iters,
                                              args.chunks, args.reps)) for r in range(args.ranks)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print("exit codes:", sorted({p.exitcode for p in ps}), flush=True)


if __name__ == "__main__":
    main()
