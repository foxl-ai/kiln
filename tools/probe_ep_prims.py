"""Device checks of the primitives the expert-parallel small-lane (decode) path needs, on one NeuronCore (trn1):

1. tensor_tensor with its second operand an SBUF tile read through a stride-0 access pattern along the free axis
   (a per-(row, c) scale broadcast over N lanes), the first in PSUM;
2. tensor_reduce over the middle axis of a [P, A, N] tile, read through a permuted access pattern;
3. fp32 scales rebuilt on the partitions from three bf16 parts on 32 partitions by accumulating identity matmuls
   (a transpose that also sums hi + mid + lo), bit for bit.

    python tools/probe_ep_prims.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def k_bcast(p, s):
        """p fp32 [128, A, N] (to PSUM via a copy), s fp32 [128, A]: out[q, n] = sum_a p[q, a, n] s[q, a]."""
        _, A, N = p.shape
        f32 = nl.float32
        pt = nl.ndarray((128, A, N), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=pt, src=p)
        pp = nl.ndarray((128, A, N), dtype=f32, buffer=nl.psum)
        nisa.tensor_copy(dst=pp, src=pt, engine=nisa.vector_engine)
        st = nl.ndarray((128, A), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=st, src=s)
        prod = nl.ndarray((128, A, N), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=pp, data2=st.ap(pattern=[[A, 128], [1, A], [0, N]], offset=0),
                           op=nl.multiply, engine=nisa.vector_engine)
        red = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=red, op=nl.add, data=prod.ap(pattern=[[A * N, 128], [1, N], [N, A]], offset=0), axis=2)
        out = nl.ndarray((128, N), dtype=f32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=red)
        return out

    @nki.jit
    def k_scales(parts):
        """parts bf16 [3, 32, 128]: out fp32 [128, 32] = (parts[0] + parts[1]) + parts[2], transposed, by identity
        matmuls accumulated in PSUM."""
        bf16, f32 = nl.bfloat16, nl.float32
        ident = nl.shared_identity_matrix(n=128, dtype=bf16)
        ps = nl.ndarray((128, 32), dtype=f32, buffer=nl.psum)
        for r in range(3):
            t = nl.ndarray((32, 128), dtype=bf16, buffer=nl.sbuf)
            nisa.dma_copy(dst=t, src=parts[r])
            nisa.nc_matmul(dst=ps, stationary=t, moving=ident[0:32, 0:32], accumulate=(r > 0))
        o = nl.ndarray((128, 32), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o, src=ps, engine=nisa.vector_engine)
        out = nl.ndarray((128, 32), dtype=f32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=o)
        return out


    @nki.jit
    def k_dq_engines(w, s, nv: int, ng: int, reps: int):
        """reps rounds of nv + ng dequantizations of a [128, 512] fp8 tile against an fp32 scale tile, each into its
        own slice of one SBUF tile that is stored at the end (so none is dead code): nv on the vector engine (scale
        in PSUM, as the EP kernel), ng on GpSimd (scale in SBUF: GpSimd cannot read PSUM)."""
        f32, bf16, fp8 = nl.float32, nl.bfloat16, nl.float8_e4m3
        wt = nl.ndarray((128, 512), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=w)
        st = nl.ndarray((128, 512), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=st, src=s)
        sp = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
        nisa.tensor_copy(dst=sp, src=st, engine=nisa.vector_engine)
        n = nv + ng
        big = nl.ndarray((128, n, 512), dtype=bf16, buffer=nl.sbuf)
        for r in range(reps):
            for i in range(nv):
                nisa.tensor_tensor(dst=big[:, i, :], data1=wt.view(fp8), data2=sp, op=nl.multiply,
                                   engine=nisa.vector_engine)
            for i in range(ng):
                nisa.tensor_tensor(dst=big[:, nv + i, :], data1=wt.view(fp8), data2=st, op=nl.multiply,
                                   engine=nisa.gpsimd_engine)
        out = nl.ndarray((128, n, 512), dtype=bf16, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=big)
        return out

    @nki.jit
    def k_dma(w, NB: int, CH: int, dyn: int):
        """Stream w uint8 [E, 128, F] (E experts of 128 x F bytes, as gu / dn hold one expert's weights) from HBM into
        SBUF in chunks of [128, CH] through a ring of NB buffers (static offsets, or dyn: a scalar_offset per expert
        as the kernels' device-loop passes), each chunk read by one tiny vector op so no load is dead; returns the sum
        of the read bytes' first column (fp32 [128, 1])."""
        f32, u8 = nl.float32, nl.uint8
        E, _, F = w.shape
        bufs = []
        for k in range(NB):
            bufs.append(nl.ndarray((128, CH), dtype=u8, buffer=nl.sbuf))
        acc = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        if dyn:
            ei = nl.ndarray((1, E), dtype=nl.int32, buffer=nl.sbuf)
            nisa.iota(dst=ei, pattern=[[1, E]], offset=0, channel_multiplier=0)
        n = 0
        for e in range(E):
            for f0 in range(0, F, CH):
                b = bufs[n % NB]
                if dyn:
                    src = w.ap(pattern=[[F, 128], [1, CH]], offset=f0, scalar_offset=ei[:, e:e + 1], indirect_dim=0)
                else:
                    src = w.ap(pattern=[[F, 128], [1, CH]], offset=e * 128 * F + f0)
                nisa.dma_copy(dst=b, src=src)
                t = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=t, src=b[:, 0:1], engine=nisa.vector_engine)
                nisa.tensor_tensor(dst=acc, data1=acc, data2=t, op=nl.add, engine=nisa.vector_engine)
                n += 1
        out = nl.ndarray((128, 1), dtype=f32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=acc)
        return out

    @nki.jit
    def k_pe2(st, mv, n: int, S: int, Ms: int, N: int):
        """n matmuls, stationary tile i % S of st [128, S, Ms] (Ms output partitions), moving mv [128, N], bf16, each
        into PSUM [Ms, N] (accumulated)."""
        f32, bf16 = nl.float32, nl.bfloat16
        stt = nl.ndarray((128, S, Ms), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=stt, src=st)
        mvt = nl.ndarray((128, N), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=mvt, src=mv)
        ps = nl.ndarray((Ms, N), dtype=f32, buffer=nl.psum)
        for i in range(n):
            nisa.nc_matmul(dst=ps, stationary=stt[:, i % S, :], moving=mvt, accumulate=(i > 0))
        o = nl.ndarray((Ms, N), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o, src=ps, engine=nisa.vector_engine)
        out = nl.ndarray((Ms, N), dtype=f32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=o)
        return out

    @nki.jit
    def k_pe(st, mv, n: int, S: int, N: int, fp8: int):
        """n matmuls accumulated into one PSUM tile [128, N]: stationary tile i % S of st [128, S, 128] (S distinct
        weight tiles), moving mv [128, N]; bf16 operands, or both float8_e4m3 when fp8 (st / mv uint8 codes)."""
        f32, bf16, fp8t = nl.float32, nl.bfloat16, nl.float8_e4m3
        dt = nl.uint8 if fp8 else bf16
        stt = nl.ndarray((128, S, 128), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=stt, src=st)
        mvt = nl.ndarray((128, N), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=mvt, src=mv)
        ps = nl.ndarray((128, N), dtype=f32, buffer=nl.psum)
        for i in range(n):
            a = stt[:, i % S, :]
            b = mvt
            if fp8:
                a, b = a.view(fp8t), b.view(fp8t)
            nisa.nc_matmul(dst=ps, stationary=a, moving=b, accumulate=(i > 0))
        o = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o, src=ps, engine=nisa.vector_engine)
        out = nl.ndarray((128, N), dtype=f32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=o)
        return out

    @nki.jit
    def k_dq_tile(w, s, mode: int, reps: int):
        """reps rounds of dequantizing w uint8 [128, n, 512] (fp8 codes) with ONE scale per 128-column block, s fp32
        [128, n, 4] (the block's scale on every partition), into bf16 [128, n, 512]. mode 0: vector tensor_scalar per
        128 columns (scale a per-partition operand); 1: scalar-engine activation per 128 columns (scale=); 2: 0 and 1
        alternating by block; 3: vector tensor_tensor per 512 columns against the scales read through a stride-0
        access pattern; 4: the current kernel's form, tensor_tensor per 512 columns against a PSUM scale row."""
        f32, bf16, fp8 = nl.float32, nl.bfloat16, nl.float8_e4m3
        _, n, _ = w.shape
        wt = nl.ndarray((128, n, 512), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=w)
        st = nl.ndarray((128, n, 4), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=st, src=s)
        big = nl.ndarray((128, n, 512), dtype=bf16, buffer=nl.sbuf)
        if mode == 5:  # tiles i < n / 2 on the vector engine, the rest on the scalar engine, separate tensors
            h = n // 2
            bv = nl.ndarray((128, h, 512), dtype=bf16, buffer=nl.sbuf)
            ba = nl.ndarray((128, n - h, 512), dtype=bf16, buffer=nl.sbuf)
            for r in range(reps):
                for i in range(n):
                    for j in range(4):
                        src = wt[:, i, j * 128:(j + 1) * 128].view(fp8)
                        if i < h:
                            nisa.tensor_scalar(dst=bv[:, i, j * 128:(j + 1) * 128], data=src, op0=nl.multiply,
                                               operand0=st[:, i, j:j + 1], engine=nisa.vector_engine)
                        else:
                            nisa.activation(dst=ba[:, i - h, j * 128:(j + 1) * 128], op=nl.copy, data=src,
                                            scale=st[:, i, j:j + 1])
            out = nl.ndarray((128, n, 512), dtype=bf16, buffer=nl.shared_hbm)
            nisa.dma_copy(dst=out[:, 0:h, :], src=bv)
            nisa.dma_copy(dst=out[:, h:n, :], src=ba)
            return out
        if mode == 4:
            sp = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
            nisa.tensor_copy(dst=sp, src=st.ap(pattern=[[n * 4, 128], [1, 4], [0, 128]], offset=0),
                             engine=nisa.vector_engine)
        for r in range(reps):
            for i in range(n):
                if mode == 3:
                    nisa.tensor_tensor(dst=big[:, i, :], data1=wt[:, i, :].view(fp8),
                                       data2=st.ap(pattern=[[n * 4, 128], [1, 4], [0, 128]], offset=i * 4),
                                       op=nl.multiply, engine=nisa.vector_engine)
                    continue
                if mode == 4:
                    nisa.tensor_tensor(dst=big[:, i, :], data1=wt[:, i, :].view(fp8), data2=sp, op=nl.multiply,
                                       engine=nisa.vector_engine)
                    continue
                for j in range(4):
                    src = wt[:, i, j * 128:(j + 1) * 128].view(fp8)
                    dst = big[:, i, j * 128:(j + 1) * 128]
                    if mode == 0 or (mode == 2 and j % 2 == 0):
                        nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=st[:, i, j:j + 1],
                                           engine=nisa.vector_engine)
                    else:
                        nisa.activation(dst=dst, op=nl.copy, data=src, scale=st[:, i, j:j + 1])
        out = nl.ndarray((128, n, 512), dtype=bf16, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=big)
        return out


def dq_tile() -> None:
    """--dq-tile: block-constant scales (one per 128 x 128 tile): bit-exactness against bf16(fp32(code) s) and the
    time per [128, 512] of each form, (t(reps 4) - t(reps 1)) / (3 n)."""
    import profile_layer as pl
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from kiln.models.quant import FP8

    g = torch.Generator().manual_seed(0)
    n = 32
    w = torch.randint(0, 256, (128, n, 512), dtype=torch.uint8, generator=g)
    w = torch.where((w & 0x7F) >= 0x78, w & 0xF0, w)  # no NaN codes, at most 240 in magnitude
    sc = torch.rand(n, 4, generator=g) * 1e-3 + 1e-5
    s = sc.unsqueeze(0).expand(128, n, 4).contiguous()
    ref = (w.view(FP8).float().view(128, n, 4, 128) * sc.view(1, n, 4, 1)).view(128, n, 512).bfloat16()
    for mode in (5, 2, 1, 0):
        ts = {}
        for reps in (1, 4):
            def f(a, b, mode=mode, reps=reps):
                return wrap_nki(k_dq_tile)[1](w=a, s=b, mode=mode, reps=reps)
            if reps == 1:
                try:
                    got = torch.compile(f, **pl.OPTS)(w.to(pl.DEV), s.to(pl.DEV)).cpu()
                except Exception as e:
                    print(f"mode {mode}: FAILED {type(e).__name__}: {str(e)[:600]}", flush=True)
                    break
                print(f"mode {mode}: bit-exact {bool(torch.equal(got, ref))}, {int((got != ref).sum())} of "
                      f"{ref.numel()} differ", flush=True)

            def f2(a, b, mode=mode, reps=reps):
                return wrap_nki(k_dq_tile)[1](w=a, s=b, mode=mode, reps=reps)[0:1, 0, 0:4]
            ts[reps] = pl.timed(f"dq mode {mode} reps {reps}", f2, (w.to(pl.DEV), s.to(pl.DEV)), 10)
        if len(ts) == 2:
            print(f"-> mode {mode}: {(ts[4] - ts[1]) / (3 * n) * 1e6:.3f} us per [128, 512]", flush=True)


def dma_rate() -> None:
    """--dma: HBM -> SBUF read bandwidth of one NeuronCore for one layer's EP experts (9 x 25 MB) in chunks of
    [128, CH] bytes through NB buffers, static or per-expert dynamic offsets."""
    import profile_layer as pl
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    E, F = 9, 196608  # 128 x 196608 = 25.2 MB per expert (gate_up and down of GLM-5.3-Flash in fp8)
    w = torch.randint(0, 64, (E, 128, F), dtype=torch.uint8)
    for NB, CH, dyn in ((2, 8192, 0), (4, 8192, 0), (8, 8192, 0), (4, 16384, 0), (4, 32768, 0), (8, 16384, 0),
                        (4, 8192, 1), (8, 16384, 1)):
        def f(a, NB=NB, CH=CH, dyn=dyn):
            return wrap_nki(k_dma)[1](w=a, NB=NB, CH=CH, dyn=dyn)
        t = pl.timed(f"dma NB={NB} CH={CH} dyn={dyn}", f, (w.to(pl.DEV),), 5)
        print(f"-> NB={NB} chunk [128, {CH}] ({128 * CH / 2**20:.1f} MiB) {'dynamic' if dyn else 'static'}: "
              f"{E * 128 * F / t / 1e9:.0f} GB/s ({t * 1e3:.2f} ms for {E * 128 * F / 1e6:.0f} MB)", flush=True)


def pe_shape() -> None:
    """--pe-shape: time per matmul for a stationary of Ms output partitions against N moving columns (S = 16 distinct
    stationaries): which of the two the stationary load costs when the moving operand is short (decode lanes)."""
    import profile_layer as pl
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    g = torch.Generator().manual_seed(0)
    for Ms, N in ((128, 16), (128, 32), (128, 64), (16, 128), (16, 512), (32, 512), (64, 512), (128, 512)):
        ts = {}
        for n in (256, 1024):
            st = torch.randn(128, 16, Ms, generator=g).bfloat16()
            mv = torch.randn(128, N, generator=g).bfloat16()

            def f(a, b, n=n, Ms=Ms, N=N):
                return wrap_nki(k_pe2)[1](st=a, mv=b, n=n, S=16, Ms=Ms, N=N)[0:1, 0:4]
            ts[n] = pl.timed(f"pe Ms={Ms} N={N} n={n}", f, (st.to(pl.DEV), mv.to(pl.DEV)), 10)
        per = (ts[1024] - ts[256]) / 768
        print(f"-> stationary [128, {Ms}], moving [128, {N}]: {per * 1e9:.1f} ns per matmul", flush=True)


def pe_rate() -> None:
    """--pe: the tensor engine's time per matmul as a function of the moving width N, distinct stationaries S and
    the operand type: (t(n2) - t(n1)) / (n2 - n1) per matmul, from synchronous calls of whole graphs."""
    import profile_layer as pl
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    g = torch.Generator().manual_seed(0)
    for fp8 in (0, 1):
        for S in (1, 16):
            for N in (128, 256, 512):
                ts = {}
                for n in (256, 1024):
                    if fp8:
                        st = (torch.randint(0, 256, (128, S, 128), dtype=torch.uint8, generator=g) & 0x3F)
                        mv = (torch.randint(0, 256, (128, N), dtype=torch.uint8, generator=g) & 0x3F)
                    else:
                        st = torch.randn(128, S, 128, generator=g).bfloat16()
                        mv = torch.randn(128, N, generator=g).bfloat16()

                    def f(a, b, n=n, S=S, N=N, fp8=fp8):
                        return wrap_nki(k_pe)[1](st=a, mv=b, n=n, S=S, N=N, fp8=fp8)[0:1, 0:4]
                    ts[n] = pl.timed(f"pe fp8={fp8} S={S} N={N} n={n}", f, (st.to(pl.DEV), mv.to(pl.DEV)), 10)
                per = (ts[1024] - ts[256]) / 768
                print(f"-> fp8={fp8} S={S} N={N}: {per * 1e9:.1f} ns per matmul, {per * 1e9 / N:.3f} ns per moving "
                      f"column ({N / per / 1e9:.2f} G columns/s)", flush=True)


def main() -> None:
    import profile_layer as pl
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from kiln.kernels.moe_ep import split3

    pl.setup_device()
    if "--dma" in sys.argv:
        return dma_rate()
    if "--pe-shape" in sys.argv:
        return pe_shape()
    if "--pe" in sys.argv:
        return pe_rate()
    if "--dq-tile" in sys.argv:
        return dq_tile()
    A, N = 32, 16

    g = torch.Generator().manual_seed(0)
    p = torch.randn(128, A, N, generator=g)
    s = torch.rand(128, A, generator=g) * 1e-3
    got = torch.compile(lambda a, b: wrap_nki(k_bcast)[1](p=a, s=b), **pl.OPTS)(p.to(pl.DEV), s.to(pl.DEV)).cpu()
    want = (p * s.unsqueeze(-1)).sum(1)
    print(f"1+2. stride-0 broadcast multiply + permuted reduce: max |err| {(got - want).abs().max().item():.3e} "
          f"(|want| max {want.abs().max().item():.3e})", flush=True)
    import sys as _s
    if "--engines" in _s.argv:
        from kiln.models.quant import FP8
        w = (torch.randint(0, 256, (128, 512), dtype=torch.uint8, generator=g) & 0xBF)
        sc4 = torch.rand(128, 512, generator=g) * 1e-3
        for nv, ng in ((64, 0), (0, 64), (48, 16), (40, 24), (32, 32), (0, 1)):
            def f(a, b, nv=nv, ng=ng):
                return wrap_nki(k_dq_engines)[1](w=a, s=b, nv=nv, ng=ng, reps=4)[:, 0, 0:4]
            pl.timed(f"dequant x4: {nv} on vector + {ng} on gpsimd", f, (w.to(pl.DEV), sc4.to(pl.DEV)), 10)
        return
    sc = torch.rand(32, 128, generator=g) * 1e-3 * (1 + (torch.rand(32, 128, generator=g) < 0.5).float())
    parts = split3(sc).permute(2, 0, 1).contiguous()  # [3, 32, 128]
    got = torch.compile(lambda a: wrap_nki(k_scales)[1](parts=a), **pl.OPTS)(parts.to(pl.DEV)).cpu()
    print(f"3. scales rebuilt by identity matmuls: bit-identical {bool(torch.equal(got, sc.T))}, "
          f"{int((got != sc.T).sum())} of {sc.numel()} differ", flush=True)


if __name__ == "__main__":
    main()
