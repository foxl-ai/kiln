"""What paces the pooled indexer's score on trn1: 32 heads x 512 pools per chunk, 128 queries, each head a PE
matmul, a relu and a weighted add into an fp32 accumulator, in variants (one NeuronCore, one variant per process).

    python tools/probe_lc_score.py [variant ...]   # no argument: every variant

 0 ACT relu in place in PSUM, DVE weighted add from that PSUM into an SBUF accumulator (dsa_long_select today)
 1 ACT relu PSUM -> SBUF fp32, DVE weighted add from SBUF into an SBUF accumulator
 2 ACT relu PSUM -> SBUF fp32, DVE weighted add from SBUF into a PSUM accumulator
 3 no relu: DVE weighted add straight from the matmul's PSUM (wrong values; the DVE alone)
 4 ACT relu in place only (no add; the ACT alone)
 5 variant 0 with the heads split: even heads' adds on DVE, odd heads' on GpSimd from an SBUF relu (two accumulators)
 6 PE matmuls only
 7 variant 0 with the ACT relu of every other head done by DVE (max with 0, then the weighted add)
 8 variant 3 with 2 accumulators, heads alternating
 9 variant 3 with 4 accumulators
10 variant 3 with an immediate scalar instead of the per-partition weight
11 variant 0 with 2 accumulators
12 variant 0 with 4 accumulators
13 ACT relu PSUM -> another PSUM bank, DVE weighted add from it, 2 accumulators
14 ACT relu PSUM -> SBUF fp32, DVE weighted add from SBUF, 2 SBUF accumulators
15 variant 11 with the PSUM ring 3 deep (half the banks)
Each prints ns per head-chunk (the call's p50 over 32 chunks x 32 heads, minus nothing).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

V = [int(a) for a in sys.argv[1:]]
if len(V) != 1:
    todo = V or list(range(int(re.findall(r"^ ?(\d+) ", __doc__, re.M)[-1]) + 1))
    for v in todo:
        r = subprocess.run([sys.executable, __file__, str(v)], capture_output=True, text=True)
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("var ")]
        print(lines[-1] if lines else f"var {v}: FAILED (exit {r.returncode}) {r.stderr[-600:]}", flush=True)
    sys.exit(0)

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402

F32, BF16 = nl.float32, nl.bfloat16
VE, GE = nisa.vector_engine, nisa.gpsimd_engine
NCH, HI = 32, 32


@nki.jit
def k(qs_in, kt_in, w_in, var: int):
    out = nl.ndarray((128, 512), dtype=F32, buffer=nl.shared_hbm)
    qs = nl.ndarray((128, HI, 128), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=qs, src=qs_in)
    kt = nl.ndarray((128, 512), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=kt, src=kt_in)
    ws = nl.ndarray((128, HI), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ws, src=w_in)
    scl = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.memset(dst=scl, value=0.088)
    zb = nl.ndarray((128, 1), dtype=F32, buffer=nl.sbuf)
    nisa.memset(dst=zb, value=0.0)
    SP = []
    for _ in range(3 if var == 13 or var == 15 else 6):
        SP.append(nl.ndarray((128, 512), dtype=F32, buffer=nl.psum))
    RP = []
    for _ in range(3 if var == 13 else 0):
        RP.append(nl.ndarray((128, 512), dtype=F32, buffer=nl.psum))
    RS = []
    for _ in range(4):
        RS.append(nl.ndarray((128, 512), dtype=F32, buffer=nl.sbuf))
    acc = nl.ndarray((128, 512), dtype=F32, buffer=nl.sbuf)
    acc2 = nl.ndarray((128, 512), dtype=F32, buffer=nl.sbuf)
    ACC = []
    for _ in range(4):
        ACC.append(nl.ndarray((128, 512), dtype=F32, buffer=nl.sbuf))
        nisa.memset(dst=ACC[-1], value=0.0)
    accp = nl.ndarray((128, 512), dtype=F32, buffer=nl.psum)
    nisa.memset(dst=acc, value=0.0)
    nisa.memset(dst=acc2, value=0.0)
    if var == 2:
        nisa.memset(dst=accp, value=0.0)
    for c in range(NCH):
        for h in range(HI):
            i = c * HI + h
            sp = SP[i % len(SP)]
            nisa.nc_matmul(dst=sp, stationary=qs[:, h, :], moving=kt, accumulate=False)
            if var == 6:
                continue
            if var == 0 or var == 5 and h % 2 == 0 or var == 7 and h % 2 == 0:
                nisa.activation(dst=sp, op=nl.relu, data=sp, scale=scl, bias=zb)
                nisa.scalar_tensor_tensor(dst=acc, data=sp, op0=nl.multiply, operand0=ws[:, h:h + 1], op1=nl.add,
                                          operand1=acc)
            elif var == 7:
                r = RS[i % 4]
                nisa.tensor_scalar(dst=r, data=sp, op0=nl.multiply, operand0=0.088, op1=nl.maximum, operand1=0.0,
                                   engine=VE)
                nisa.scalar_tensor_tensor(dst=acc, data=r, op0=nl.multiply, operand0=ws[:, h:h + 1], op1=nl.add,
                                          operand1=acc)
            elif var == 1 or var == 2:
                r = RS[i % 4]
                nisa.activation(dst=r, op=nl.relu, data=sp, scale=scl, bias=zb)
                a = accp if var == 2 else acc
                nisa.scalar_tensor_tensor(dst=a, data=r, op0=nl.multiply, operand0=ws[:, h:h + 1], op1=nl.add,
                                          operand1=a)
            elif var == 3:
                nisa.scalar_tensor_tensor(dst=acc, data=sp, op0=nl.multiply, operand0=ws[:, h:h + 1], op1=nl.add,
                                          operand1=acc)
            elif var == 4:
                nisa.activation(dst=sp, op=nl.relu, data=sp, scale=scl, bias=zb)
            elif var == 8 or var == 9:
                a = ACC[h % (2 if var == 8 else 4)]
                nisa.scalar_tensor_tensor(dst=a, data=sp, op0=nl.multiply, operand0=ws[:, h:h + 1], op1=nl.add,
                                          operand1=a)
            elif var == 10:
                nisa.scalar_tensor_tensor(dst=acc, data=sp, op0=nl.multiply, operand0=0.37, op1=nl.add,
                                          operand1=acc)
            elif var == 13:
                rp = RP[i % 3]
                nisa.activation(dst=rp, op=nl.relu, data=sp, scale=scl, bias=zb)
                a = ACC[h % 2]
                nisa.scalar_tensor_tensor(dst=a, data=rp, op0=nl.multiply, operand0=ws[:, h:h + 1], op1=nl.add,
                                          operand1=a)
            elif var == 14:
                r = RS[i % 4]
                nisa.activation(dst=r, op=nl.relu, data=sp, scale=scl, bias=zb)
                a = ACC[h % 2]
                nisa.scalar_tensor_tensor(dst=a, data=r, op0=nl.multiply, operand0=ws[:, h:h + 1], op1=nl.add,
                                          operand1=a)
            elif var == 11 or var == 12 or var == 15:
                a = ACC[h % (4 if var == 12 else 2)]
                nisa.activation(dst=sp, op=nl.relu, data=sp, scale=scl, bias=zb)
                nisa.scalar_tensor_tensor(dst=a, data=sp, op0=nl.multiply, operand0=ws[:, h:h + 1], op1=nl.add,
                                          operand1=a)
            elif var == 5:
                r = RS[i % 4]
                nisa.activation(dst=r, op=nl.relu, data=sp, scale=scl, bias=zb)
                nisa.scalar_tensor_tensor(dst=acc2, data=r, op0=nl.multiply, operand0=ws[:, h:h + 1], op1=nl.add,
                                          operand1=acc2, engine=GE)
        if var == 4 or var == 6:
            nisa.tensor_copy(dst=acc, src=SP[(c * HI) % len(SP)], engine=VE)
    if var == 2:
        nisa.tensor_copy(dst=acc, src=accp, engine=VE)
    if var == 5:
        nisa.tensor_tensor(dst=acc, data1=acc, data2=acc2, op=nl.add, engine=VE)
    if var >= 8 and var != 10:
        for j in range(4):
            nisa.tensor_tensor(dst=acc, data1=acc, data2=ACC[j], op=nl.add, engine=VE)
    nisa.dma_copy(dst=out, src=acc)
    return out


def main() -> None:
    import profile_layer as pl
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    v = V[0]
    pl.setup_device(False)
    g = torch.Generator().manual_seed(0)
    qs = torch.randn(128, HI, 128, generator=g).to(torch.bfloat16)
    kt = torch.randn(128, 512, generator=g).to(torch.bfloat16)
    w = torch.randn(128, HI, generator=g)
    f = lambda a, b, c: wrap_nki(k)[1](qs_in=a, kt_in=b, w_in=c, var=v)  # noqa: E731
    t = pl.timed(f"var {v}", f, tuple(x.to(pl.DEV) for x in (qs, kt, w)), 10)
    print(f"var {v}: {t * 1e9 / (NCH * HI):.0f} ns per head-chunk ({t * 1e3:.3f} ms)", flush=True)


if __name__ == "__main__":
    main()
