"""Instruction throughput of each NeuronCore-v2 engine on one core, for kernel rooflines: the time one more
instruction of a kind costs when nothing else waits on it (the difference between nt = N1 and N2 repeats of the
same instruction, over N2 - N1), each variant in its own process.

    python tools/probe_engine_rates.py [variant ...]   # no argument: every variant

Each repeat writes a different tile of a ring (the compiler tracks dependencies per tensor, so ring entries
that are tensors of their own pipeline) and every result is read at the end (nothing is dead code). Variants:
 0 PE matmul bf16, stationary [128, 128], moving [128, 512] -> PSUM fp32
 1 PE matmul bf16, stationary [128, 128], moving [128, 128]
 2 PE matmul fp32 operands, stationary [128, 128], moving [128, 512]
 3 PE matmul bf16 stationary, fp8 (e4m3) moving [128, 512]
 4 PE matmul bf16, stationary [128, 8] (8 columns), moving [128, 512]
 5 PE matmul bf16, stationary [128, 8], moving [128, 128]
 6 PE matmul bf16 stationary [128, 128], moving = identity [128, 128] (a transpose as kernels/dsa_decode does it)
 7 PE nc_transpose bf16 [128, 128] -> PSUM bf16
 8 PE matmul bf16, stationary [64, 128] (K = 64 rows), moving [64, 512]
 9 PE matmul bf16, stationary [1, 128] (K = 1, an outer product), moving [1, 512]
10 DVE tensor_reduce max over [128, 512] fp32 read from PSUM
11 DVE tensor_reduce max over [128, 2048] fp32 in SBUF
12 ACT activation exp(x - m) of [128, 512] fp32 PSUM -> bf16 SBUF, with the row sum (reduce_res)
13 ACT activation copy [128, 512] fp32 PSUM -> bf16 SBUF
14 DVE tensor_copy [128, 512] fp32 PSUM -> bf16 SBUF
15 DVE scalar_tensor_tensor (PSUM x scalar) + SBUF fp32 [128, 512] -> SBUF fp32
16 DVE tensor_scalar SBUF fp32 [128, 2048] -> SBUF fp32
17 DVE tensor_scalar_reduce (>= scalar, sum) over [128, 2112] fp32 (one radix round of kernels/dsa_topk.py)
18 GpSimd tensor_scalar SBUF fp32 [128, 512]
19 DMA HBM -> SBUF static [128, 8192] bf16 (2 MiB)
20 DMA indirect gather of 128 rows x 2048 bytes (kernels/dsa_decode.py's pool gather)
21 DVE tensor_reduce add over [128, 128] fp32 SBUF (the per-instruction overhead at small free sizes)
22 ACT activation copy [128, 128] fp32 SBUF -> fp32 SBUF
23 DVE tensor_tensor SBUF x SBUF fp32 [128, 512]
24 DVE tensor_tensor PSUM x SBUF fp32 [128, 512] -> SBUF
25 PE matmul fp32, stationary [128, 1] (one column), moving [128, 128] (kernels/kda_decode.py's reads)
26 PE matmul bf16, stationary [128, 128], moving [128, 256]
27 ACT activation exp of [128, 2048] fp32 SBUF -> bf16 SBUF with the row sum
28 PE matmul fp32 stationary [128, 128], fp32 moving [128, 128] (kernels/delta_rule.py's every matmul)
29 PE matmul bf16 stationary [128, 128], fp32 moving [128, 128]
30 PE matmul fp32 stationary [128, 128], bf16 moving [128, 128]
31 PE nc_transpose fp32 [128, 128] -> PSUM fp32
32 PE matmul fp32 stationary [128, 128], moving = fp32 identity [128, 128] (delta_rule's transposes)
33 DVE tensor_copy fp32 SBUF [128, 128] -> bf16 SBUF (the hi part of a bf16 split)
34 DVE tensor_tensor fp32 SBUF - bf16 SBUF [128, 128] -> bf16 (the lo part of a bf16 split)
35 ACT activation copy fp32 SBUF [128, 128] -> bf16 SBUF

Measured results and their use: docs/neuron-notes.md "Engine rates on trn1".
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402

V = [int(a) for a in sys.argv[1:]]
if len(V) != 1:  # every variant (or the listed ones), each in its own process: a failed compile ends one
    todo = V or list(range(int(re.findall(r"^ ?(\d+) ", __doc__, re.M)[-1]) + 1))
    for v in todo:
        r = subprocess.run([sys.executable, __file__, str(v)], capture_output=True, text=True)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("var ")]
        print(lines[-1] if lines else f"var {v}: FAILED (exit {r.returncode}) {r.stderr[-600:]}", flush=True)
    sys.exit(0)
V = V[0]

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402

RING = 4


@nki.jit
def k(xb, xf, x8, ident, idx, var: int, nt: int):
    """xb bf16 [128, 8192], xf fp32 [128, 2112], x8 uint8 [128, 512] (fp8 bytes), ident bf16 [128, 128],
    idx int32 [128, 1] row indices into xb viewed as [N, 1024] bf16 rows (2048 bytes)."""
    F32, BF16 = nl.float32, nl.bfloat16
    out = nl.ndarray((128, 1), dtype=F32, buffer=nl.shared_hbm)
    B = nl.ndarray((128, 8192), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=B, src=xb)
    Fs = nl.ndarray((128, 2112), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=Fs, src=xf)
    E8 = nl.ndarray((128, 512), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.dma_copy(dst=E8, src=x8)
    IB = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=IB, src=ident)
    ix = nl.ndarray((128, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ix, src=idx)
    acc = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.memset(dst=acc, value=0.0)
    col = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.memset(dst=col, value=0.5)
    pe = var <= 9 or var in (25, 26) or 28 <= var <= 32
    if pe:  # PSUM accumulators in a ring, every one read at the end
        P = []
        for _ in range(RING):
            dt = BF16 if var == 7 else F32
            P.append(nl.ndarray((128, 512), dtype=dt, buffer=nl.psum))
        Fb = None
        if var in (2, 25) or 28 <= var <= 32:
            Fb = nl.ndarray((128, 512), dtype=F32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=Fb, src=B[:, 0:512], engine=nisa.vector_engine)
        if var == 32:
            IF = nl.ndarray((128, 128), dtype=F32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=IF, src=IB, engine=nisa.vector_engine)
        for t in range(nt):
            p = P[t % RING]
            first = t < RING
            if var == 0:
                nisa.nc_matmul(dst=p, stationary=B[:, 0:128], moving=B[:, 512:1024], accumulate=not first)
            elif var == 1:
                nisa.nc_matmul(dst=p[:, 0:128], stationary=B[:, 0:128], moving=B[:, 512:640], accumulate=not first)
            elif var == 2:
                nisa.nc_matmul(dst=p, stationary=Fb[:, 0:128], moving=Fb, accumulate=not first)
            elif var == 3:
                nisa.nc_matmul(dst=p, stationary=B[:, 0:128], moving=E8.view(nl.float8_e4m3), accumulate=not first)
            elif var == 4:
                nisa.nc_matmul(dst=p[0:8, :], stationary=B[:, 0:8], moving=B[:, 512:1024], accumulate=not first)
            elif var == 5:
                nisa.nc_matmul(dst=p[0:8, 0:128], stationary=B[:, 0:8], moving=B[:, 512:640], accumulate=not first)
            elif var == 6:
                nisa.nc_matmul(dst=p[:, 0:128], stationary=B[:, (t % 8) * 128:(t % 8 + 1) * 128], moving=IB,
                               accumulate=not first)
            elif var == 7:
                nisa.nc_transpose(dst=p[:, 0:128], data=B[:, (t % 8) * 128:(t % 8 + 1) * 128],
                                  engine=nisa.tensor_engine)
            elif var == 8:
                nisa.nc_matmul(dst=p, stationary=B[0:64, 0:128], moving=B[0:64, 512:1024], accumulate=not first)
            elif var == 9:
                nisa.nc_matmul(dst=p, stationary=B[0:1, 0:128], moving=B[0:1, 512:1024], accumulate=not first)
            elif var == 25:
                nisa.nc_matmul(dst=p[0:1, 0:128], stationary=Fb[:, 0:1], moving=Fb[:, 0:128], accumulate=not first)
            elif var == 26:
                nisa.nc_matmul(dst=p[:, 0:256], stationary=B[:, 0:128], moving=B[:, 512:768], accumulate=not first)
            elif var == 28:
                nisa.nc_matmul(dst=p[:, 0:128], stationary=Fb[:, 0:128], moving=Fb[:, 128:256], accumulate=not first)
            elif var == 29:
                nisa.nc_matmul(dst=p[:, 0:128], stationary=B[:, 0:128], moving=Fb[:, 128:256], accumulate=not first)
            elif var == 30:
                nisa.nc_matmul(dst=p[:, 0:128], stationary=Fb[:, 0:128], moving=B[:, 512:640], accumulate=not first)
            elif var == 31:
                nisa.nc_transpose(dst=p[:, 0:128], data=Fb[:, (t % 4) * 128:(t % 4 + 1) * 128], engine=nisa.tensor_engine)
            elif var == 32:
                nisa.nc_matmul(dst=p[:, 0:128], stationary=Fb[:, (t % 4) * 128:(t % 4 + 1) * 128], moving=IF,
                               accumulate=not first)
        for i in range(RING):
            r = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
            src = P[i][0:1, 0:128] if var == 25 else (P[i][0:8, :] if var in (4, 5) else P[i])
            n = 1 if var == 25 else (8 if var in (4, 5) else 128)
            nisa.tensor_reduce(dst=r[0:n, :], op=nl.add, data=src, axis=1)
            nisa.tensor_tensor(dst=acc[0:n, :], data1=acc[0:n, :], data2=r[0:n, :], op=nl.add,
                               engine=nisa.vector_engine)
    else:
        ps = nl.ndarray((128, 512), dtype=F32, buffer=nl.psum)
        nisa.nc_matmul(dst=ps, stationary=B[:, 0:128], moving=B[:, 512:1024], accumulate=False)
        R = []
        for _ in range(RING):
            if var in (10, 11, 17, 21):
                R.append(nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf))
            elif var in (33, 34, 35):
                R.append(nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf))
            elif var in (12, 13, 14, 27):
                R.append(nl.ndarray((128, 2048 if var == 27 else 512), dtype=BF16, buffer=nl.sbuf))
            elif var == 19:
                R.append(nl.ndarray((128, 8192), dtype=BF16, buffer=nl.sbuf))
            elif var == 20:
                R.append(nl.ndarray((128, 1024), dtype=BF16, buffer=nl.sbuf))
            else:
                R.append(nl.ndarray((128, 2048), dtype=F32, buffer=nl.sbuf))
        sums = []
        for _ in range(RING):
            sums.append(nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf))
        scr = nl.ndarray((128, 2112), dtype=F32, buffer=nl.sbuf)
        xv = xb.reshape((128 * 8, 1024))
        for t in range(nt):
            d = R[t % RING]
            if var == 10:
                nisa.tensor_reduce(dst=d, op=nl.maximum, data=ps, axis=1)
            elif var == 11:
                nisa.tensor_reduce(dst=d, op=nl.maximum, data=Fs[:, 0:2048], axis=1)
            elif var == 12:
                nisa.activation_reduce(dst=d, op=nl.exp, data=ps, reduce_op=nl.add, reduce_res=sums[t % RING],
                                       bias=col, scale=1.0)
            elif var == 13:
                nisa.activation(dst=d, op=nl.copy, data=ps)
            elif var == 14:
                nisa.tensor_copy(dst=d, src=ps, engine=nisa.vector_engine)
            elif var == 15:
                nisa.scalar_tensor_tensor(dst=d[:, 0:512], data=ps, op0=nl.multiply, operand0=0.5, op1=nl.add,
                                          operand1=Fs[:, 0:512])
            elif var == 16:
                nisa.tensor_scalar(dst=d, data=Fs[:, 0:2048], op0=nl.multiply, operand0=0.5,
                                   engine=nisa.vector_engine)
            elif var == 17:
                nisa.tensor_scalar_reduce(dst=scr, data=Fs, op0=nl.greater_equal, operand0=col, reduce_op=nl.add,
                                          reduce_res=d)
            elif var == 18:
                nisa.tensor_scalar(dst=d[:, 0:512], data=Fs[:, 0:512], op0=nl.multiply, operand0=0.5,
                                   engine=nisa.gpsimd_engine)
            elif var == 19:
                nisa.dma_copy(dst=d, src=xb)
            elif var == 20:
                nisa.dma_copy(dst=d, src=xv.ap(pattern=[[1024, 128], [1, 1024]], offset=0,
                                                vector_offset=ix.ap(pattern=[[1, 128], [1, 1]], offset=0),
                                                indirect_dim=0))
            elif var == 21:
                nisa.tensor_reduce(dst=d, op=nl.add, data=Fs[:, (t % 8) * 128:(t % 8 + 1) * 128], axis=1)
            elif var == 22:
                nisa.activation(dst=d[:, 0:128], op=nl.copy, data=Fs[:, 0:128])
            elif var == 23:
                nisa.tensor_tensor(dst=d[:, 0:512], data1=Fs[:, 0:512], data2=Fs[:, 512:1024], op=nl.multiply,
                                   engine=nisa.vector_engine)
            elif var == 24:
                nisa.tensor_tensor(dst=d[:, 0:512], data1=ps, data2=Fs[:, 512:1024], op=nl.multiply,
                                   engine=nisa.vector_engine)
            elif var == 27:
                nisa.activation_reduce(dst=d, op=nl.exp, data=Fs[:, 0:2048], reduce_op=nl.add,
                                       reduce_res=sums[t % RING], bias=col, scale=1.0)
            elif var == 33:
                nisa.tensor_copy(dst=d, src=Fs[:, (t % 8) * 128:(t % 8 + 1) * 128], engine=nisa.vector_engine)
            elif var == 34:
                nisa.tensor_tensor(dst=d, data1=Fs[:, (t % 8) * 128:(t % 8 + 1) * 128], data2=B[:, (t % 8) * 128:(t % 8 + 1) * 128],
                                   op=nl.subtract, engine=nisa.vector_engine)
            elif var == 35:
                nisa.activation(dst=d, op=nl.copy, data=Fs[:, (t % 8) * 128:(t % 8 + 1) * 128])
        for i in range(RING):
            r = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
            if var in (10, 11, 17, 21):
                nisa.tensor_copy(dst=r, src=R[i], engine=nisa.vector_engine)
            elif var in (12, 27):
                nisa.tensor_copy(dst=r, src=sums[i], engine=nisa.vector_engine)
            else:
                w = 128 if var in (22, 33, 34, 35) else 512
                nisa.tensor_reduce(dst=r, op=nl.add, data=R[i][:, 0:w], axis=1)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=r, op=nl.add, engine=nisa.vector_engine)
    nisa.dma_copy(dst=out, src=acc)
    return out


import profile_layer as pl  # noqa: E402
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki  # noqa: E402

pl.setup_device()
g = torch.Generator().manual_seed(0)
xb = (torch.randn(128, 8192, generator=g) * 0.1).to(torch.bfloat16)
xf = torch.randn(128, 2112, generator=g)
x8 = torch.randint(0, 120, (128, 512), dtype=torch.uint8, generator=g)
ident = torch.eye(128).to(torch.bfloat16)
idx = torch.randperm(1024, generator=g)[:128].to(torch.int32).view(128, 1)
call = wrap_nki(k)[1]
args = tuple(t.to(pl.DEV) for t in (xb, xf, x8, ident, idx))
N1, N2 = (16, 272) if V not in (19,) else (4, 36)
res = {}
try:
    for nt in (N1, N2):
        res[nt] = pl.timed(f"var {V} nt {nt}", lambda a, b, c, d, e, nt=nt: call(xb=a, xf=b, x8=c, ident=d, idx=e,
                                                                               var=V, nt=nt), args, 20)
    print(f"var {V}: {(res[N2] - res[N1]) / (N2 - N1) * 1e9:.1f} ns per instruction "
          f"({res[N1] * 1e3:.3f} / {res[N2] * 1e3:.3f} ms at nt {N1} / {N2})", flush=True)
except Exception as e:
    print(f"var {V}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
