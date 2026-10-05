"""DMA throughput of a prefill MoE lane tile's loads (kernels/moe_prefill.py), on one NeuronCore-v2:

    python tools/probe_prefill_dma.py [variant]     # no argument: every variant, one process each

Each variant runs nt tiles (4 and 68; the time per tile is the difference / 64) over a ring of 4
buffers, each buffer touched by one tiny vector op so that nothing is dead code (a store nobody reads
back IS removed: variant 9). Measured 2026-10-03 (trn1.2xlarge, SDK 2.32, nki 0.6.0): one x gather
of 128 rows x 8 KB 4.01 us, one blob load 3.05, both 8.15 (they do not overlap); split into 8 DMAs
each over their columns 2.18 us for both (19), also with the dequantization's vector work beside them
(23: 3.32). Inside the kernel the same splits made C=8192 slower (15.2 -> 17.2 ms with two-way splits,
23.4 with eight), because every dynamic DMA costs the GpSimd sequencer about 1 us (a TENSOR_LOAD of
its offset, address arithmetic, the trigger: neuron-explorer timeline), so the kernel keeps one DMA
per load. Variants (argv):
 0 x gather: indirect DMA of 128 rows x 8 KB (vector_offset), per tile
 1 expert blob: [128, 6400] B by scalar_offset, per tile
 2 0 + 1
 3 0 + 1 + the two scale rows broadcast to every partition (scalar_offset, partition stride 0)
 4 static [128, 8 KB] load, per tile
 5 0 + 1 + a static [128, 8 KB] store, per tile
 6 x gather as two DMAs of [128, 4 KB] (column halves)
 7 x gather as two DMAs of 64 rows
 8 1 as two DMAs of 64 partitions
 9 static [128, 8 KB] store only
10 0 + 1 + 9, the loads two tiles ahead of the touch
11 x gather as four DMAs of [128, 2 KB]
12 blob as two DMAs of [128, 3200 B] (column halves)
13 blob as four DMAs of [128, 1600 B]
14 6 + 12 (x halves and blob halves)
15 6 + 1
16 11 + 13
17 6 + 12 + a static [128, 8 KB] store per tile, Y read back at the end
18 9 with Y read back at the end
19 x gather in 8 DMAs + blob in 8
20 16 + the scale rows + a static store (Y read back)
21 x gather in 8 DMAs only
22 2 + 64 vector tensor_scalar ops of [128, 64] fp8 -> bf16 per tile (the dequantization's SBUF traffic)
23 19 + the same vector work
24 2 + 64 scalar-engine activations of [128, 64] per tile
25 2 + the vector work of 22 on half the ops and the scalar work of 24 on the other half
26 x gather as ONE DMA with rows cut into 8 elements of 1 KB (3-D access pattern)
27 blob as ONE DMA with rows cut into 8 elements of 800 B
28 26 + 27
29 28 with 4 elements per row
30 28 with 16 elements per row
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import re

import nki, nki.isa as nisa, nki.language as nl
from nki.isa.constants import oob_mode

V = int(sys.argv[1]) if len(sys.argv) > 1 else -1
H, F, E = 4096, 6400, 288


@nki.jit
def k(x, blob, ts, be, var: int, nt: int):
    f32, bf16 = nl.float32, nl.bfloat16
    C = x.shape[0]
    NTL = ts.shape[1]
    NB = be.shape[1]
    out = nl.ndarray((128, 8), dtype=f32, buffer=nl.shared_hbm)
    Y = nl.ndarray((128 * 4, H), dtype=bf16, buffer=nl.private_hbm)
    tsb = nl.ndarray((128, NTL), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tsb, src=ts)
    beb = nl.ndarray((1, NB), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=beb, src=be)
    acc = nl.ndarray((128, 8), dtype=f32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    xr, wr, sr, yr = [], [], [], []
    for r in range(4):
        xr.append(nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf))
        wr.append(nl.ndarray((128, F), dtype=nl.uint8, buffer=nl.sbuf))
        sr.append(nl.ndarray((128, 2, 128), dtype=nl.uint8, buffer=nl.sbuf))
        yb = nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=yb, value=0.0)
        yr.append(yb)
    lead = 2 if var == 10 else 0
    for i in range(-lead, nt):
        tl = i + lead
        if tl < nt:
            r = tl % 4
            e = beb.ap(pattern=[[NB, 1], [1, 1]], offset=tl)
            if var in (0, 2, 3, 5, 10, 22, 24, 25):
                nisa.dma_copy(dst=xr[r], src=x.ap(pattern=[[H, 128], [1, H]], offset=0,
                                                  vector_offset=tsb.ap(pattern=[[NTL, 128], [1, 1]], offset=tl),
                                                  indirect_dim=0), oob_mode=oob_mode.skip)
            if var in (6, 11, 14, 15, 16, 17, 19, 20, 21, 23):
                nh = 8 if var in (19, 21, 23) else (4 if var in (11, 16, 20) else 2)
                for hh in range(nh):
                    w = H // nh
                    nisa.dma_copy(dst=xr[r][:, hh * w:(hh + 1) * w],
                                  src=x.ap(pattern=[[H, 128], [1, w]], offset=hh * w,
                                           vector_offset=tsb.ap(pattern=[[NTL, 128], [1, 1]], offset=tl),
                                           indirect_dim=0), oob_mode=oob_mode.skip)
            if var == 7:
                for hh in range(2):
                    nisa.dma_copy(dst=xr[r][hh * 64:(hh + 1) * 64, :],
                                  src=x.ap(pattern=[[H, 64], [1, H]], offset=0,
                                           vector_offset=tsb.ap(pattern=[[NTL, 64], [1, 1]], offset=hh * 64 * NTL + tl),
                                           indirect_dim=0), oob_mode=oob_mode.skip)
            if var in (26, 28, 29, 30):
                ne = 4 if var == 29 else (16 if var == 30 else 8)
                nisa.dma_copy(dst=xr[r].reshape((128, ne, H // ne)),
                              src=x.ap(pattern=[[H, 128], [H // ne, ne], [1, H // ne]], offset=0,
                                       vector_offset=tsb.ap(pattern=[[NTL, 128], [1, 1]], offset=tl), indirect_dim=0),
                              oob_mode=oob_mode.skip)
            if var in (27, 28, 29, 30):
                ne = 4 if var == 29 else (16 if var == 30 else 8)
                nisa.dma_copy(dst=wr[r].reshape((128, ne, F // ne)),
                              src=blob.ap(pattern=[[F, 128], [F // ne, ne], [1, F // ne]], offset=0, scalar_offset=e,
                                          indirect_dim=0), oob_mode=oob_mode.skip)
            if var in (12, 13, 14, 16, 17, 19, 20, 23):
                nh = 8 if var in (19, 23) else (4 if var in (13, 16, 20) else 2)
                for hh in range(nh):
                    w = F // nh
                    nisa.dma_copy(dst=wr[r][:, hh * w:(hh + 1) * w], src=blob.ap(pattern=[[F, 128], [1, w]], offset=hh * w,
                                                                                 scalar_offset=e, indirect_dim=0),
                                  oob_mode=oob_mode.skip)
            if var in (1, 2, 3, 5, 10, 15, 22, 24, 25):
                nisa.dma_copy(dst=wr[r], src=blob.ap(pattern=[[F, 128], [1, F]], offset=0, scalar_offset=e,
                                                     indirect_dim=0), oob_mode=oob_mode.skip)
            if var == 8:
                for hh in range(2):
                    nisa.dma_copy(dst=wr[r][hh * 64:(hh + 1) * 64, :],
                                  src=blob.ap(pattern=[[F, 64], [1, F]], offset=hh * 64 * F, scalar_offset=e,
                                              indirect_dim=0), oob_mode=oob_mode.skip)
            if var in (3, 20):
                nisa.dma_copy(dst=sr[r], src=blob.ap(pattern=[[0, 128], [64 * F, 2], [1, 128]], offset=H + H // 2,
                                                     scalar_offset=e, indirect_dim=0), oob_mode=oob_mode.skip)
            if var == 4:
                nisa.dma_copy(dst=xr[r], src=x[(tl % 16) * 128:(tl % 16 + 1) * 128, :])
        if i < 0:
            continue
        r = i % 4
        if var in (0, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30):
            nisa.tensor_tensor(dst=acc[:, 0:1], data1=acc[:, 0:1], data2=xr[r][:, 4095:4096], op=nl.add,
                               engine=nisa.vector_engine)
        if var in (1, 2, 3, 5, 8, 10, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 24, 25, 27, 28, 29, 30):
            nisa.tensor_tensor(dst=acc[:, 1:2], data1=acc[:, 1:2], data2=wr[r][:, 6399:6400], op=nl.add,
                               engine=nisa.vector_engine)
        if var in (22, 23, 24, 25):  # dequantize the blob's first 4096 bytes, as the kernel does
            wq = nl.ndarray((128, 32, 128), dtype=bf16, buffer=nl.sbuf)
            for c in range(32):
                for hf in range(2):
                    d = wq[:, c, hf * 64:(hf + 1) * 64]
                    s8 = wr[r][:, c * 128 + hf * 64:c * 128 + (hf + 1) * 64].view(nl.float8_e4m3)
                    if var == 24 or (var == 25 and c % 2 == 1):
                        nisa.activation(dst=d, op=nl.copy, data=s8, scale=acc[:, 7:8])
                    else:
                        nisa.tensor_scalar(dst=d, data=s8, op0=nl.multiply, operand0=acc[:, 7:8],
                                           engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=acc[:, 6:7], data1=acc[:, 6:7], data2=wq[:, 31, 127:128], op=nl.add,
                               engine=nisa.vector_engine)
        if var in (3, 20):
            nisa.tensor_tensor(dst=acc[:, 2:3], data1=acc[:, 2:3], data2=sr[r][:, 1, 127:128], op=nl.add,
                               engine=nisa.vector_engine)
        if var in (5, 9, 10, 17, 18, 20):
            nisa.tensor_scalar(dst=yr[r][:, 0:1], data=acc[:, 0:1], op0=nl.add, operand0=1.0, engine=nisa.vector_engine)
            nisa.dma_copy(dst=Y[r * 128:(r + 1) * 128, :], src=yr[r])
    if var in (17, 18, 20):
        yy = nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=yy, src=Y[0:128, :])
        nisa.tensor_tensor(dst=acc[:, 3:4], data1=acc[:, 3:4], data2=yy[:, 4095:4096], op=nl.add, engine=nisa.vector_engine)
    nisa.dma_copy(dst=out, src=acc)
    return out

if V < 0:  # every variant, each in its own process (a kernel that fails to compile ends it)
    n = int(re.findall(r"^ ?(\d+) ", __doc__, re.M)[-1]) + 1
    for v in range(n):
        r = subprocess.run([sys.executable, __file__, str(v)], capture_output=True, text=True)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("var ")]
        print(lines[-1] if lines else f"var {v}: FAILED (exit {r.returncode})", flush=True)
    sys.exit(0)

import profile_layer as pl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
pl.setup_device()
g = torch.Generator().manual_seed(0)
C = 8192
x = torch.randn(C, H, generator=g).bfloat16()
blob = torch.randint(0, 256, (E, 128, F), dtype=torch.uint8, generator=g)
NT = 68
ts = torch.stack([torch.randperm(C, generator=g)[:128] for _ in range(NT)], 1).to(torch.int32)  # [128, NT]
be = torch.randint(0, E, (1, NT), generator=g).to(torch.int32)
call = wrap_nki(k)[1]
args = tuple(t.to(pl.DEV) for t in (x, blob, ts, be))
res = {}
try:
    for nt in (4, 68):
        res[nt] = pl.timed(f"var {V} nt {nt}", lambda a, b, c, d, nt=nt: call(x=a, blob=b, ts=c, be=d, var=V, nt=nt).sum(), args, 10)
    print(f"var {V}: {(res[68] - res[4]) / 64 * 1e6:.2f} us per tile", flush=True)
except Exception as e:
    print(f"var {V}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
