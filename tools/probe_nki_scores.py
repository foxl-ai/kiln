"""A trn1 PSUM accumulation hazard, minimal: the score step of kernels/dsa_decode.py on device against nki.simulate.

    NEURON_RT_VISIBLE_CORES=0 python tools/probe_nki_scores.py <mode>

20 blocks of K [128 tokens, 512] bf16 are transposed on the tensor engine in four 128-column pieces, and q^T [512 (4 x
128), 8] (as four stationary pieces) against them gives scores [8, 128] per block, accumulated over the four pieces
(accumulate=False, then True three times). Modes: 0 every block into its own PSUM tile (a ring of 2), read right after;
1 the four pieces into separate PSUM slices summed on the vector engine; 2 mode 0 with each block's rows gathered by an
indirect DMA; 3 four consecutive blocks into the four 128-column slices of one [8, 512] PSUM tile (a ring of 2), read
after the fourth; 4 mode 3 with [128, 512] tiles; 5 mode 3 with a ring of 3. Measured on kiln-g2-trn1 (trn1.32xlarge,
SDK 2.32, nki 0.6.0, 2026-10-04): the simulator is exact (7e-7) in every mode; the device is exact in modes 0, 1, 2 and
in modes 3, 4, 5 block 4 (the first of the second tile) keeps only the last of its four terms (max |err| 3.44,
equal to the lc = 3 product alone), the other 19 blocks exact. Prints the error per block and what a wrong block is."""
import itertools
import os
import sys

import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kiln import platform
platform.configure_runtime_env()
import libtorch_neuronx_lite  # noqa
import nki, nki.isa as nisa, nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from kiln.engine.model_runner import canonical_neuron_backend, neuronx_cc_args

F32, BF16 = nl.float32, nl.bfloat16
MODE = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ACC_NONE = len(sys.argv) > 2 and sys.argv[2] == "none"

@nki.jit
def kern(K, q, identb, src, idx, mode: int):
    # K [NB, 128, 512] bf16, q [8, 512] bf16 -> s [NB, 8, 128] fp32; mode 2: block i's rows gathered from src [N, 512]
    # by idx [128, NB] (row of partition p of block i at idx[p, i])
    NB = K.shape[0]
    ix = nl.ndarray((128, NB), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ix, src=idx)
    s = nl.ndarray((NB, 8, 128), dtype=F32, buffer=nl.shared_hbm)
    IB = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=IB, src=identb)
    qr = nl.ndarray((128, 512), dtype=BF16, buffer=nl.sbuf)
    nisa.dma_copy(dst=qr[0:8, :], src=q)
    pq = nl.ndarray((128, 4, 128), dtype=F32, buffer=nl.psum)
    for lc in range(4):
        nisa.nc_matmul(dst=pq[:, lc, 0:8], stationary=qr[0:8, lc * 128:(lc + 1) * 128], moving=IB[0:8, 0:8], accumulate=False)
    QT = nl.ndarray((128, 4, 8), dtype=BF16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=QT, src=pq[:, :, 0:8], engine=nisa.vector_engine)
    PT = (nl.ndarray((128, 4, 128), dtype=F32, buffer=nl.psum), nl.ndarray((128, 4, 128), dtype=F32, buffer=nl.psum))
    if mode == 5:  # mode 3 with a ring of 3 PSUM tiles
        PS = (nl.ndarray((8, 512), dtype=F32, buffer=nl.psum), nl.ndarray((8, 512), dtype=F32, buffer=nl.psum),
              nl.ndarray((8, 512), dtype=F32, buffer=nl.psum))
    elif mode == 4:  # mode 3 with full-partition PSUM tiles (each its own bank), using partitions 0-7
        PS = (nl.ndarray((128, 512), dtype=F32, buffer=nl.psum), nl.ndarray((128, 512), dtype=F32, buffer=nl.psum))
    else:
        PS = (nl.ndarray((8, 512), dtype=F32, buffer=nl.psum), nl.ndarray((8, 512), dtype=F32, buffer=nl.psum))
    KT = (nl.ndarray((128, 4, 128), dtype=BF16, buffer=nl.sbuf), nl.ndarray((128, 4, 128), dtype=BF16, buffer=nl.sbuf),
          nl.ndarray((128, 4, 128), dtype=BF16, buffer=nl.sbuf), nl.ndarray((128, 4, 128), dtype=BF16, buffer=nl.sbuf))
    KS = (nl.ndarray((128, 512), dtype=BF16, buffer=nl.sbuf), nl.ndarray((128, 512), dtype=BF16, buffer=nl.sbuf))
    SO = (nl.ndarray((8, 128), dtype=F32, buffer=nl.sbuf), nl.ndarray((8, 128), dtype=F32, buffer=nl.sbuf))
    SO4 = (nl.ndarray((8, 512), dtype=F32, buffer=nl.sbuf), nl.ndarray((8, 512), dtype=F32, buffer=nl.sbuf))
    for i in range(NB):
        ks = KS[i % 2]
        if mode == 2:
            nisa.dma_copy(dst=ks, src=src.ap(pattern=[[512, 128], [1, 512]], offset=0,
                                             vector_offset=ix.ap(pattern=[[NB, 128], [1, 1]], offset=i), indirect_dim=0))
        else:
            nisa.dma_copy(dst=ks, src=K[i])
        pt = PT[i % 2]
        for lc in range(4):
            nisa.nc_matmul(dst=pt[:, lc, :], stationary=ks[:, lc * 128:(lc + 1) * 128], moving=IB, accumulate=False)
        kt = KT[i % 4]
        nisa.tensor_copy(dst=kt, src=pt, engine=nisa.vector_engine)
        ps = PS[(i // 4) % len(PS)] if mode in (3, 4, 5) else PS[i % 2]
        if mode == 4:
            ps = ps[0:8, :]
        if mode in (3, 4, 5):  # four consecutive blocks into the four 128-column slices of one PSUM tile, read after the fourth
            t = i % 4
            for lc in range(4):
                if ACC_NONE:
                    nisa.nc_matmul(dst=ps[:, t * 128:(t + 1) * 128], stationary=QT[:, lc, :], moving=kt[:, lc, :])
                else:
                    nisa.nc_matmul(dst=ps[:, t * 128:(t + 1) * 128], stationary=QT[:, lc, :], moving=kt[:, lc, :],
                                   accumulate=(lc > 0))
            if t == 3:
                so4 = SO4[(i // 4) % 2]
                nisa.tensor_copy(dst=so4, src=ps, engine=nisa.vector_engine)
                for tt in range(4):
                    nisa.dma_copy(dst=s[i - 3 + tt], src=so4[:, tt * 128:(tt + 1) * 128])
            continue
        if mode in (0, 2):  # as the kernel: stationary q^T block [128, 8], moving K^T block, accumulate over blocks
            for lc in range(4):
                nisa.nc_matmul(dst=ps[:, 0:128], stationary=QT[:, lc, :], moving=kt[:, lc, :], accumulate=(lc > 0))
        else:  # each block into its own PSUM slice, summed on the vector engine
            for lc in range(4):
                nisa.nc_matmul(dst=ps[:, lc * 128:(lc + 1) * 128], stationary=QT[:, lc, :], moving=kt[:, lc, :], accumulate=False)
        so = SO[i % 2]
        if mode in (0, 2):
            nisa.tensor_copy(dst=so, src=ps[:, 0:128], engine=nisa.vector_engine)
        else:
            nisa.tensor_copy(dst=so, src=ps[:, 0:128], engine=nisa.vector_engine)
            for lc in range(1, 4):
                nisa.tensor_tensor(dst=so, data1=so, data2=ps[:, lc * 128:(lc + 1) * 128], op=nl.add)
        nisa.dma_copy(dst=s[i], src=so)
    return s

NB = 20
g = torch.Generator().manual_seed(0)
K = torch.randn(NB, 128, 512, generator=g).to(torch.bfloat16)
q = (torch.randn(8, 512, generator=g) * 0.05).to(torch.bfloat16)
src = torch.randn(4096, 512, generator=g).to(torch.bfloat16)
idx = torch.randperm(4096, generator=g)[:128 * NB].view(NB, 128)
if MODE == 2:
    K = src[idx]
idx_t = idx.t().contiguous().to(torch.int32)
ref = torch.einsum("hr,btr->bht", q.float(), K.float())
sim = torch.as_tensor(nki.simulate(kern)(K=K, q=q, identb=torch.eye(128).to(torch.bfloat16), src=src, idx=idx_t, mode=MODE)).float()
print("mode", MODE, "simulator err", float((sim - ref).abs().max()))
dev = torch.device("neuron:0")
f = torch.compile(lambda K, q, I, src, idx: wrap_nki(kern)[1](K=K, q=q, identb=I, src=src, idx=idx, mode=MODE),
                  backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                  options={"compiler_args": neuronx_cc_args(torch.bfloat16)})
out = f(K.to(dev), q.to(dev), torch.eye(128).to(torch.bfloat16).to(dev), src.to(dev), idx_t.to(dev)).cpu().float()
e = (out - ref).abs().amax((1, 2))
print("mode", MODE, "device err per block", [round(float(x), 4) for x in e])
bad = [i for i in range(NB) if float(e[i]) > 1e-3]
for i in bad:
    errs = {j: float((out[i] - ref[j]).abs().max()) for j in range(NB)}
    j = min(errs, key=errs.get)
    print(f"block {i}: closest ref block {j} err {errs[j]:.3e}; |out| max {float(out[i].abs().max()):.3e}; "
          f"rows wrong {[(r, round(float((out[i, r] - ref[i, r]).abs().max()), 3)) for r in range(8)]}")
    cols = (out[i] - ref[i]).abs().amax(0)
    print("   cols wrong:", (cols > 1e-3).nonzero().flatten().tolist()[:20], "count", int((cols > 1e-3).sum()))
for i in bad:
    parts = [torch.einsum("hr,tr->ht", q.float()[:, lc * 128:(lc + 1) * 128], K[i].float()[:, lc * 128:(lc + 1) * 128]) for lc in range(4)]
    for n in range(1, 5):
        for sub in itertools.combinations(range(4), n):
            v = sum(parts[k] for k in sub)
            er = float((out[i] - v).abs().max())
            if er < 1e-2:
                print(f"block {i} equals the sum of lc blocks {sub} (err {er:.2e})")
    # other blocks' lc parts?
    for j in range(NB):
        pj = [torch.einsum("hr,tr->ht", q.float()[:, lc * 128:(lc + 1) * 128], K[j].float()[:, lc * 128:(lc + 1) * 128]) for lc in range(4)]
        for lc in range(4):
            res = out[i] - (ref[i] - parts[lc]) - pj[lc]
            if float(res.abs().max()) < 1e-2:
                print(f"block {i}: its lc {lc} term came from block {j}")
