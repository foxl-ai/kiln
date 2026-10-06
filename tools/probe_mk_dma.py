"""HBM -> SBUF DMA rate of one NeuronCore (trn1) for the MoE kernels' weight streams, by the shape of each DMA: bytes
per partition row (one descriptor element each), static or dynamic (scalar_offset: software DGE on GpSimd, how every
kernel weight load runs today), partitions per DMA, and DMAs in flight. One layer's EP experts (9 x 25.2 MB) are
streamed through a ring of buffers; each buffer is read by one tiny vector op so no load is dead code.

    python tools/probe_mk_dma.py [--variants 8192:128:0:4 32768:128:1:4 ...] [--save-inputs <prefix>]

A variant is c<CH>:<NB> (every [128, CH] chunk one contiguous HBM block) or CH:PARTS:DYN:NB = bytes per partition per DMA, partitions per DMA (a [128, CH] chunk is moved as 128 /
PARTS DMAs of [PARTS, CH]), dynamic or static offsets, ring buffers. Reported: GB/s over the call minus the null graph.
With --save-inputs each variant's graph inputs are saved (as tools/prof_engines.py reads them) and its compile-cache
hash printed, so its packet sizes and per-engine rates can be read (tools/prof_ops.py).
"""

from __future__ import annotations

import argparse
import glob
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
    def k_dma_c(w, NB: int):
        """Stream w uint8 [N, 128, CH] (every chunk one contiguous block of 128 x CH bytes) into SBUF, one DMA per chunk,
        through NB ring buffers; returns the sum of every chunk's first byte per partition."""
        f32, u8 = nl.float32, nl.uint8
        N, _, CH = w.shape
        bufs = []
        for _ in range(NB):  # the NKI tracer takes no comprehension
            bufs.append(nl.ndarray((128, CH), dtype=u8, buffer=nl.sbuf))
        acc = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        for k in range(N):
            b = bufs[k % NB]
            nisa.dma_copy(dst=b, src=w[k])
            t = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=t, src=b[:, 0:1], engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=t, op=nl.add, engine=nisa.vector_engine)
        out = nl.ndarray((128, 1), dtype=f32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=acc)
        return out

    @nki.jit
    def k_dma(w, CH: int, PARTS: int, DYN: int, NB: int):
        """Stream w uint8 [E, 128, F] into SBUF in [128, CH] chunks, each as 128 / PARTS DMAs of [PARTS, CH], through NB
        ring buffers; returns the sum of every chunk's first byte per partition (fp32 [128, 1])."""
        f32, u8 = nl.float32, nl.uint8
        E, _, F = w.shape
        bufs = []
        for _ in range(NB):  # the NKI tracer takes no comprehension
            bufs.append(nl.ndarray((128, CH), dtype=u8, buffer=nl.sbuf))
        acc = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        ei = nl.ndarray((1, E), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=ei, pattern=[[1, E]], offset=0, channel_multiplier=0)
        n = 0
        for e in range(E):
            for f0 in range(0, F, CH):
                b = bufs[n % NB]
                for p0 in range(0, 128, PARTS):
                    if DYN:
                        src = w.ap(pattern=[[F, PARTS], [1, CH]], offset=p0 * F + f0, scalar_offset=ei[:, e:e + 1],
                                   indirect_dim=0)
                    else:
                        src = w.ap(pattern=[[F, PARTS], [1, CH]], offset=e * 128 * F + p0 * F + f0)
                    nisa.dma_copy(dst=b[p0:p0 + PARTS, :], src=src)
                t = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=t, src=b[:, 0:1], engine=nisa.vector_engine)
                nisa.tensor_tensor(dst=acc, data1=acc, data2=t, op=nl.add, engine=nisa.vector_engine)
                n += 1
        out = nl.ndarray((128, 1), dtype=f32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=out, src=acc)
        return out


def main() -> None:
    import profile_layer as pl
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=["8192:128:1:4", "8192:128:0:4", "16384:128:1:4", "32768:128:1:4", "32768:128:0:4",
                             "8192:32:1:4", "65536:128:0:2"])
    ap.add_argument("--experts", type=int, default=9)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--save-inputs", default=None)
    a = ap.parse_args()
    pl.setup_device()
    E, F = a.experts, 196608  # 128 x 196608 B = 25.2 MB per expert (GLM-5.3-Flash gate_up + down, fp8)
    w = torch.randint(0, 64, (E, 128, F), dtype=torch.uint8)
    wd = w.to(pl.DEV)
    null = pl.timed("null graph", lambda h: h + 1, (torch.zeros(4, 128).to(pl.DEV),), a.iters)
    for v in a.variants:
        if v.startswith("c"):  # c<CH>:<NB>: the same bytes laid out so that every [128, CH] chunk is contiguous
            CH, NB = (int(t) for t in v[1:].split(":"))
            wc = w.view(E, 128, F // CH, CH).permute(0, 2, 1, 3).reshape(E * (F // CH), 128, CH).contiguous().to(pl.DEV)

            def fc(x, NB=NB):
                return wrap_nki(k_dma_c)[1](w=x, NB=NB)

            t = pl.timed(f"dma {v}", fc, (wc,), a.iters)
            print(f"-> contiguous [128, {CH}] chunks, {NB} buffers: {E * 128 * F / (t - null) / 1e9:.0f} GB/s", flush=True)
            continue
        CH, PARTS, DYN, NB = (int(t) for t in v.split(":"))

        def f(x, CH=CH, PARTS=PARTS, DYN=DYN, NB=NB):
            return wrap_nki(k_dma)[1](w=x, CH=CH, PARTS=PARTS, DYN=DYN, NB=NB)

        t = pl.timed(f"dma {v}", f, (wd,), a.iters)
        net = t - null
        print(f"-> CH={CH} B/partition, {PARTS} partitions per DMA ({128 // PARTS} per chunk), "
              f"{'dynamic' if DYN else 'static'}, {NB} buffers: {E * 128 * F / net / 1e9:.0f} GB/s "
              f"({net * 1e3:.3f} ms for {E * 128 * F / 1e6:.0f} MB)", flush=True)
        if a.save_inputs:
            path = f"{a.save_inputs}.{v.replace(':', '-')}.pt"
            torch.save({"x": w}, path)
            cache = "/root/.cache/neuron_libtorch/neuron/compile_cache"
            hs = [h for h in sorted(glob.glob(f"{cache}/*/fxgraph.txt"), key=os.path.getmtime) if "L_x_" in open(h).read()]
            print(f"   inputs {path}; graph {os.path.basename(os.path.dirname(hs[-1])) if hs else '?'}", flush=True)


if __name__ == "__main__":
    main()
