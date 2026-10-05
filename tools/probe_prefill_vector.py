"""Vector / scalar engine cost of the prefill MoE kernel's gate_up dequantization (kernels/moe_prefill.py):

    python tools/probe_prefill_vector.py [variant]  # no argument: every variant, one process each

One block's [128, 4096] fp8 gate_up tiles to bf16 with a scale per (128-column tile, 64-column
half), SBUF-resident, nt blocks (4 and 36; time per block = difference / 32), every result read by
32 matmuls so nothing is dead code. Measured 2026-10-03 (trn1.2xlarge, SDK 2.32, nki 0.6.0), us per
block: the broadcast tensor_tensor the kernel used first (0) 8.50; 64 tensor_scalar ops of [128, 64]
with the scale as a per-partition scalar (6) 5.84; 64 scalar-engine activations (13) 4.43; half on
each engine (14) 2.69; an SBUF x SBUF tensor_tensor runs at half the rate of tensor_scalar, a PSUM x
SBUF one at full rate (4 vs 2, 11).
Variants: one block's gate_up tiles ([128, 4096] fp8 -> bf16 with a
scale per (128-column tile, 64-column half)), SBUF-resident, nt blocks; variants as kernel
arguments. argv: var.
 0 tensor_tensor, fp32 scales as a broadcast AP (the kernel's form), 8 x [128, 512]
 1 the same with bf16 scales
 2 tensor_scalar, per-partition scalar, 8 x [128, 512] (no scale per column: the rate alone)
 3 tensor_copy fp8 -> bf16, 8 x [128, 512]
 4 tensor_tensor with a full bf16 [128, 512] second operand
 5 activation copy with a per-partition scale (scalar engine), 8 x [128, 512]
 6 tensor_scalar per (tile, half): 64 x [128, 64]
 7 tensor_tensor, fp32 broadcast, 2 x [128, 2048]
 8 tensor_tensor, fp32 broadcast, 1 x [128, 4096]
 9 activation copy fp32 PSUM -> bf16 SBUF [128, 512] (a drain), 8 per block
10 tensor_copy fp32 PSUM -> bf16 SBUF on the vector engine [128, 512], 8 per block
11 tensor_tensor PSUM x broadcast scale row -> bf16 [128, 512] (a y drain with 4 chunk scales), 8 per block
12 nothing (the consuming matmuls alone)
13 activation per (tile, half): 64 x [128, 64] on the scalar engine, scale per-partition
14 half of 6 on the vector engine and half of 13 on the scalar engine (tiles 0-15 / 16-31)
Every block's wq is read by 32 matmuls, so no variant's work is dead code.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import re

import nki, nki.isa as nisa, nki.language as nl

V = int(sys.argv[1]) if len(sys.argv) > 1 else -1


@nki.jit
def k(ws, ss, var: int, nt: int):
    f32, bf16, fp8 = nl.float32, nl.bfloat16, nl.float8_e4m3
    out = nl.ndarray((128, 8), dtype=f32, buffer=nl.shared_hbm)
    wr = []
    for r in range(2):
        a = nl.ndarray((128, 4096), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(dst=a, src=ws[r])
        wr.append(a)
    sc = nl.ndarray((128, 2, 32), dtype=f32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sc, src=ss)
    scb = nl.ndarray((128, 2, 32), dtype=bf16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=scb, src=sc, engine=nisa.vector_engine)
    full = nl.ndarray((128, 512), dtype=bf16, buffer=nl.sbuf)
    nisa.memset(dst=full, value=0.5)
    acc = nl.ndarray((128, 8), dtype=f32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    ps = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
    nisa.memset(dst=ps, value=0.25)
    for t in range(nt):
        r = t % 2
        wq = nl.ndarray((128, 32, 128), dtype=bf16, buffer=nl.sbuf)
        w8 = wr[r]
        if var in (0, 1, 2, 3, 4, 5, 9, 10, 11):
            for c4 in range(8):
                d = wq[:, c4 * 4:(c4 + 1) * 4, :]
                s8 = w8[:, c4 * 512:(c4 + 1) * 512].view(fp8)
                if var == 0:
                    tb = sc.ap(pattern=[[64, 128], [1, 4], [32, 2], [0, 64]], offset=c4 * 4)
                    nisa.tensor_tensor(dst=d.reshape((128, 4, 2, 64)), data1=s8.reshape((128, 4, 2, 64)), data2=tb,
                                       op=nl.multiply, engine=nisa.vector_engine)
                elif var == 1:
                    tb = scb.ap(pattern=[[64, 128], [1, 4], [32, 2], [0, 64]], offset=c4 * 4)
                    nisa.tensor_tensor(dst=d.reshape((128, 4, 2, 64)), data1=s8.reshape((128, 4, 2, 64)), data2=tb,
                                       op=nl.multiply, engine=nisa.vector_engine)
                elif var == 2:
                    nisa.tensor_scalar(dst=d, data=s8.reshape((128, 4, 128)), op0=nl.multiply, operand0=sc[:, 0, c4:c4 + 1],
                                       engine=nisa.vector_engine)
                elif var == 3:
                    nisa.tensor_copy(dst=d, src=s8.reshape((128, 4, 128)), engine=nisa.vector_engine)
                elif var == 4:
                    nisa.tensor_tensor(dst=d, data1=s8.reshape((128, 4, 128)), data2=full.reshape((128, 4, 128)),
                                       op=nl.multiply, engine=nisa.vector_engine)
                elif var == 5:
                    nisa.activation(dst=d, op=nl.copy, data=s8.reshape((128, 4, 128)), scale=sc[:, 0, c4:c4 + 1])
                elif var == 9:
                    nisa.activation(dst=d, op=nl.copy, data=ps.reshape((128, 4, 128)), scale=sc[:, 0, c4:c4 + 1])
                elif var == 10:
                    nisa.tensor_copy(dst=d, src=ps.reshape((128, 4, 128)), engine=nisa.vector_engine)
                else:
                    tb = sc.ap(pattern=[[64, 128], [1, 4], [0, 128]], offset=c4 * 4)
                    nisa.tensor_tensor(dst=d, data1=ps.reshape((128, 4, 128)), data2=tb, op=nl.multiply,
                                       engine=nisa.vector_engine)
        elif var in (13, 14):
            for c in range(32):
                for hf in range(2):
                    d = wq[:, c, hf * 64:(hf + 1) * 64]
                    s8 = w8[:, c * 128 + hf * 64:c * 128 + (hf + 1) * 64].view(fp8)
                    if var == 13 or c >= 16:
                        nisa.activation(dst=d, op=nl.copy, data=s8, scale=sc[:, hf, c:c + 1])
                    else:
                        nisa.tensor_scalar(dst=d, data=s8, op0=nl.multiply, operand0=sc[:, hf, c:c + 1],
                                           engine=nisa.vector_engine)
        elif var == 6:
            for c in range(32):
                for hf in range(2):
                    nisa.tensor_scalar(dst=wq[:, c, hf * 64:(hf + 1) * 64],
                                       data=w8[:, c * 128 + hf * 64:c * 128 + (hf + 1) * 64].view(fp8), op0=nl.multiply,
                                       operand0=sc[:, hf, c:c + 1], engine=nisa.vector_engine)
        elif var in (7, 8):
            n = 2 if var == 7 else 1
            q = 32 // n
            for c4 in range(n):
                tb = sc.ap(pattern=[[64, 128], [1, q], [32, 2], [0, 64]], offset=c4 * q)
                nisa.tensor_tensor(dst=wq[:, c4 * q:(c4 + 1) * q, :].reshape((128, q, 2, 64)),
                                   data1=w8[:, c4 * q * 128:(c4 + 1) * q * 128].view(fp8).reshape((128, q, 2, 64)),
                                   data2=tb, op=nl.multiply, engine=nisa.vector_engine)
        # consume all of wq (the tensor engine reads every tile), so nothing is dead code
        pc = nl.ndarray((128, 64), dtype=f32, buffer=nl.psum)
        for c in range(32):
            nisa.nc_matmul(dst=pc, stationary=wq[:, c, :], moving=full[:, 0:64], accumulate=(c > 0))
        nisa.tensor_tensor(dst=acc[:, 0:1], data1=acc[:, 0:1], data2=pc[:, 63:64], op=nl.add,
                           engine=nisa.vector_engine)
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
ws = torch.randint(0, 256, (2, 128, 4096), dtype=torch.uint8, generator=g) & 0xBF
ss = torch.rand(128, 2, 32, generator=g) * 0.02
call = wrap_nki(k)[1]
args = tuple(t.to(pl.DEV) for t in (ws, ss))
res = {}
try:
    for nt in (4, 36):
        res[nt] = pl.timed(f"var {V} nt {nt}", lambda a, b, nt=nt: call(ws=a, ss=b, var=V, nt=nt).sum(), args, 10)
    print(f"var {V}: {(res[36] - res[4]) / 32 * 1e6:.2f} us per block", flush=True)
except Exception as e:
    print(f"var {V}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
