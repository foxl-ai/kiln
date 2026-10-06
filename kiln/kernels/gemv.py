"""A decode-sized projection y = x W^T (few rows, a large bf16 weight) as one NKI kernel that streams the weight at the
DMA rate: the building block of the fused per-layer decode kernels.

Why: in a GLM-5.3-Flash decode call at 1 row per DP group (KILN_MOE_EP=0, KILN_DENSE_FP8=0, KILN_DSA_PREFIX=mm,
KILN_DECODE_WHOLE=1), every block of every layer moves its weights at ~130 GB/s (KDA mixer 32 MB in ~0.25 ms, FFN 37 MB
in ~0.29 ms, DSA mixer 76 MB in ~0.57 ms; tools/prof_step.py on the 32-rank replay, trn1.32xlarge, 2026-10-05), while one
core's practical HBM -> SBUF stream is ~272 GB/s (docs/neuron-notes.md "MoE kernels against their floors"). The step is
then weight-streaming at half the rate the hardware gives.

Layout: the weight comes pre-transposed, wT [K, N] (K = the input features on the partitions, 128 at a time), so each
[128, NT] tile of it is one DMA of 128 contiguous rows; x [T, K] is transposed once on the tensor engine into [128, K / 128,
T] column tiles. Each output tile [T, NT] accumulates its K / 128 matmuls in one PSUM bank (x^T stationary, the weight
moving), NT = 512 columns. Weight k-tiles [128, N] are loaded RING deep ahead.
"""

from __future__ import annotations

import os

import torch

RING = int(os.environ.get("KILN_GEMV_RING", 3))  # weight k-tiles in flight ([128, N] each)
NT = 512  # output columns per PSUM tile


def emulate(x: torch.Tensor, wT: torch.Tensor) -> torch.Tensor:
    return (x.float() @ wT.float()).to(x.dtype)


try:  # the Neuron venv
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

if nki is not None:
    F32, BF16 = nl.float32, nl.bfloat16

    @nki.jit
    def kiln_gemv_kernel(x, wT, identb, ring: int, rev: int):
        """x bf16 [T <= 128, K], wT bf16 [K, N] (K and N multiples of 128 and 512), identb bf16 [128, 128]: x wT
        bf16 [T, N] (fp32 accumulation)."""
        T, K = x.shape
        N = wT.shape[1]
        KT = K // 128
        out = nl.ndarray((T, N), dtype=BF16, buffer=nl.shared_hbm)
        IB = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=IB, src=identb)
        xs = nl.ndarray((T, K), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=xs, src=x)
        XT = nl.ndarray((128, KT, T), dtype=BF16, buffer=nl.sbuf)
        for k0 in range(0, KT, 4):  # x^T by PE transposes, 4 per PSUM bank
            kn = min(4, KT - k0)
            pt = nl.ndarray((128, 4, 128), dtype=F32, buffer=nl.psum)
            for j in range(kn):
                kt = k0 + j
                nisa.nc_matmul(dst=pt[:, j, 0:T], stationary=xs[:, kt * 128:(kt + 1) * 128], moving=IB[0:T, 0:T],
                               accumulate=False)
            nisa.tensor_copy(dst=XT[:, k0:k0 + kn, :], src=pt[:, 0:kn, 0:T], engine=nisa.vector_engine)
        # One DMA per k-tile: the weight's 128 rows of that tile across all N columns ([128, N]: N * 2 bytes per
        # partition, 8 KB at N = 4096). One core's HBM -> SBUF stream reaches ~261 GB/s only with >= 8 KB per partition
        # per DMA (docs/neuron-notes.md "MoE kernels against their floors": 16 DMA engines at ~17 GB/s each); [128, 512]
        # tiles (1 KB per partition) streamed at ~95 GB/s here. Every output tile accumulates in its own PSUM bank.
        NN = N // NT
        assert NN <= 8, "N <= 4096: one PSUM bank per 512 output columns"
        W = []
        for _ in range(ring):
            W.append(nl.ndarray((128, N), dtype=BF16, buffer=nl.sbuf))
        P = []
        for _ in range(NN):
            P.append(nl.ndarray((128, NT), dtype=F32, buffer=nl.psum))
        Y = nl.ndarray((128, N), dtype=BF16, buffer=nl.sbuf)
        for kt in range(min(ring - 1, KT)):  # prefetch
            nisa.dma_copy(dst=W[kt % ring], src=wT[kt * 128:(kt + 1) * 128, :])
        for kt in range(KT):
            j = kt + ring - 1
            if j < KT:
                nisa.dma_copy(dst=W[j % ring], src=wT[j * 128:(j + 1) * 128, :])
            w = W[kt % ring]
            for n in range(NN):
                nisa.nc_matmul(dst=P[n][0:T, :], stationary=XT[:, kt, :], moving=w[:, n * NT:(n + 1) * NT],
                               accumulate=(kt > 0))
        for n in range(NN):
            if n % 2 == 0:
                nisa.activation(dst=Y[0:T, n * NT:(n + 1) * NT], op=nl.copy, data=P[n][0:T, :])
            else:
                nisa.tensor_copy(dst=Y[0:T, n * NT:(n + 1) * NT], src=P[n][0:T, :], engine=nisa.vector_engine)
        nisa.dma_copy(dst=out, src=Y[0:T, :])
        return out

    @nki.jit
    def kiln_gemv_chain_kernel(x, wTs, identb, ring: int, rev: int):
        """x bf16 [T <= 128, K], wTs bf16 [C, K, K]: x wTs[0] wTs[1] ... wTs[C - 1] (each product rounded to bf16), in
        ONE kernel: the next product's weight k-tiles stream in while the current one finishes (a probe of what a
        fused multi-projection decode kernel streams, against C kernel calls)."""
        T, K = x.shape
        C = wTs.shape[0]
        KT = K // 128
        NN = K // NT
        out = nl.ndarray((T, K), dtype=BF16, buffer=nl.shared_hbm)
        IB = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=IB, src=identb)
        Y = nl.ndarray((128, K), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=Y[0:T, :], src=x)
        XT = nl.ndarray((128, KT, T), dtype=BF16, buffer=nl.sbuf)
        W = []
        for _ in range(ring):
            W.append(nl.ndarray((128, K), dtype=BF16, buffer=nl.sbuf))
        P = []
        for _ in range(NN):
            P.append(nl.ndarray((128, NT), dtype=F32, buffer=nl.psum))
        tiles = []
        for c in range(C):
            for kt in range(KT):
                tiles.append((c, kt))
        for i in range(min(ring - 1, len(tiles))):
            nisa.dma_copy(dst=W[i % ring], src=wTs[tiles[i][0], tiles[i][1] * 128:(tiles[i][1] + 1) * 128, :])
        for i in range(len(tiles)):
            c = tiles[i][0]
            kt = tiles[i][1]
            if kt == 0:  # this product's input, transposed (PE transposes 4 to a PSUM bank, one bank borrowed)
                for k0 in range(0, KT, 4):
                    pt = P[(k0 // 4) % NN]
                    for j in range(4):
                        nisa.nc_matmul(dst=pt[:, j * 128:j * 128 + T], stationary=Y[0:T, (k0 + j) * 128:(k0 + j + 1) * 128],
                                       moving=IB[0:T, 0:T], accumulate=False)
                    for j in range(4):
                        nisa.tensor_copy(dst=XT[:, k0 + j, :], src=pt[:, j * 128:j * 128 + T], engine=nisa.vector_engine)
            j2 = i + ring - 1
            if j2 < len(tiles):
                nisa.dma_copy(dst=W[j2 % ring], src=wTs[tiles[j2][0], tiles[j2][1] * 128:(tiles[j2][1] + 1) * 128, :])
            w = W[i % ring]
            for n in range(NN):
                nisa.nc_matmul(dst=P[n][0:T, :], stationary=XT[:, kt, :], moving=w[:, n * NT:(n + 1) * NT],
                               accumulate=(kt > 0))
            if kt == KT - 1:
                for n in range(NN):
                    if n % 2 == 0:
                        nisa.activation(dst=Y[0:T, n * NT:(n + 1) * NT], op=nl.copy, data=P[n][0:T, :])
                    else:
                        nisa.tensor_copy(dst=Y[0:T, n * NT:(n + 1) * NT], src=P[n][0:T, :], engine=nisa.vector_engine)
        nisa.dma_copy(dst=out, src=Y[0:T, :])
        return out
else:
    kiln_gemv_kernel = None
    kiln_gemv_chain_kernel = None


def _kernel_rev() -> int:
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_gemv_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev()


def gemv(x: torch.Tensor, wT: torch.Tensor, ring: int | None = None) -> torch.Tensor:
    """x [T, K] @ wT [K, N] (the kernel on a Neuron device, emulate() elsewhere)."""
    if x.device.type == "cpu":
        return emulate(x, wT)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    eye = torch.eye(128, device=x.device).to(torch.bfloat16)
    return wrap_nki(kiln_gemv_kernel)[1](x=x.to(torch.bfloat16).contiguous(), wT=wT.to(torch.bfloat16), identb=eye,
                                        ring=int(ring or RING), rev=REV)


def gemv_chain(x: torch.Tensor, wTs: torch.Tensor, ring: int | None = None) -> torch.Tensor:
    """x [T, K] through wTs [C, K, K] in one kernel (probes)."""
    if x.device.type == "cpu":
        for i in range(wTs.shape[0]):
            x = emulate(x, wTs[i])
        return x
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    eye = torch.eye(128, device=x.device).to(torch.bfloat16)
    return wrap_nki(kiln_gemv_chain_kernel)[1](x=x.to(torch.bfloat16).contiguous(), wTs=wTs.to(torch.bfloat16),
                                              identb=eye, ring=int(ring or RING), rev=REV)
