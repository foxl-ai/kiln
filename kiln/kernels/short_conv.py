"""The short causal (depthwise) conv of a linear-attention layer's prefill chunk (models/linear_attn.py _causal_conv) as one NKI
kernel, the default on trn2 at LNC=2 without context-parallel DSA (KILN_LA_CONV_KERNEL=xla turns it off, =nki on elsewhere): y [T, C] fp32 = sum_j xe[j : j + T] * w[:, j] for xe [K - 1 + T, C], the same products
summed in the same order as _causal_conv, so the same bits.

Why a kernel (trn2.48xlarge at LNC=2, SDK 2.32, neuronx-cc 2.27, 2026-10-07; docs/neuron-notes.md "trn2 G1 prefill"):
- At LNC=2 neuronx-cc splits the XLA ops of a graph between the two physical cores of the logical core along the token rows.
  The conv's K - 1 row shifts then cross that split. Its default lowering is a serialized chain of partition-shifting DMAs and
  spills, 7.0-7.2 ms of an 8.9 ms GLM-5.3-Flash KDA token mixer with every engine idle (the whole layer is 3.07 ms per
  1024-token chunk at LNC=1 and 8.59 ms at LNC=2, tools/profile_linear_attn.py).
- The faster XLA forms (channels first, or the shifts as a matmul against shifted identities) give 2.5-2.7 ms but rows T/2 ..
  T/2 + 2 of every call come out wrong at LNC=2 (the halo of the split; exact at LNC=1, and exact whenever the graph holds no
  state write after the conv).

Here the split is by channels, which needs no halo: each program (physical core, grid 2) convolves its half of the 128-channel
tiles over every row. Per tile: the rows come in as [128 rows, 128 channels] tiles (contiguous row segments), are transposed on
the tensor engine (a matmul against the identity, exact) into one [128 channels, K - 1 + T] SBUF tile, the conv runs on the
vector engine with the row shifts as free-axis offsets and w[:, j] as a per-partition scalar (x w0, then + x_j w_j for j = 1 ..
K - 1, fp32), and the result is transposed back per 128 rows and stored. Grid 1: every tile on the one program.
"""

from __future__ import annotations

import os

import torch

try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    F32 = nl.float32

    @nki.jit
    def kiln_short_conv_kernel(xe, w, ib, i32, rev: int):
        """xe bf16 [K - 1 + T, C] (C a multiple of 128), w bf16 [C, K], ib bf16 / i32 fp32 [128, 128] identities; rev: this
        module's kernel source revision. Returns y fp32 [T, C]."""
        N, C = xe.shape
        K = w.shape[1]
        T = N - (K - 1)
        NT = C // 128
        y = nl.ndarray((T, C), dtype=F32, buffer=nl.shared_hbm)
        IB = nl.ndarray((128, 128), dtype=xe.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=IB, src=ib)
        IF = nl.ndarray((128, 128), dtype=F32, buffer=nl.sbuf)
        nisa.dma_copy(dst=IF, src=i32)
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        half = (NT + 1) // 2
        c_lo, c_hi = (0, NT) if npg == 1 else ((0, half) if pid == 0 else (half, NT))
        for ct in range(c_lo, c_hi):
            c0 = ct * 128
            wb = nl.ndarray((128, K), dtype=w.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wb, src=w[c0:c0 + 128, :])
            wf = nl.ndarray((128, K), dtype=F32, buffer=nl.sbuf)
            nisa.activation(dst=wf, op=nl.copy, data=wb)
            X = nl.ndarray((128, N), dtype=F32, buffer=nl.sbuf)  # channels on the partitions, rows along the free axis
            for r0 in range(0, N, 128):
                n = min(128, N - r0)
                a = nl.ndarray((128, 128), dtype=xe.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=a[0:n, :], src=xe[r0:r0 + n, c0:c0 + 128])
                pt = nl.ndarray((128, 128), dtype=F32, buffer=nl.psum)
                nisa.nc_matmul(dst=pt[:, 0:n], stationary=a[0:n, :], moving=IB[0:n, 0:n], accumulate=False)
                nisa.activation(dst=X[:, r0:r0 + n], op=nl.copy, data=pt[:, 0:n])
            Y = nl.ndarray((128, T), dtype=F32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=Y, data=X[:, 0:T], op0=nl.multiply, operand0=wf[:, 0:1], engine=nisa.vector_engine)
            for j in range(1, K):
                nisa.scalar_tensor_tensor(dst=Y, data=X[:, j:j + T], op0=nl.multiply, operand0=wf[:, j:j + 1],
                                          op1=nl.add, operand1=Y)
            for t0 in range(0, T, 128):
                n = min(128, T - t0)
                po = nl.ndarray((128, 128), dtype=F32, buffer=nl.psum)
                nisa.nc_matmul(dst=po[0:n, :], stationary=Y[:, t0:t0 + n], moving=IF, accumulate=False)
                ob = nl.ndarray((128, 128), dtype=F32, buffer=nl.sbuf)
                nisa.activation(dst=ob[0:n, :], op=nl.copy, data=po[0:n, :])
                nisa.dma_copy(dst=y[t0:t0 + n, c0:c0 + 128], src=ob[0:n, :])
        if npg > 1:  # LNC: both programs' channels written before either ends (whatever runs next reads all of them)
            nisa.core_barrier(data=y, cores=(0, 1))
        return y
else:
    kiln_short_conv_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source, passed as the static argument `rev` (LNL's compile-cache key does not include
    NKI kernel source: kernels/delta_rule.py _kernel_rev)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_short_conv_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev()


def _default_kernel(target: str | None = None) -> str:
    """KILN_LA_CONV_KERNEL unset: nki on trn2 at LNC=2, where the XLA conv's split between the two physical cores is the cost
    (docs/neuron-notes.md "trn2 G1 prefill": wikitext -0.5507 against -0.5495, check_mixed 29 / 32 equal; G1 serving on half
    the box 142.0 / 159.1 / 171.5 -> 186.8 / 217.8 / 241.5 out tok/s at conc 32 / 64 / 128, the prefill call 0.583 -> 0.368 s);
    xla elsewhere, so trn1's graphs (and their keys) are the ones they were.

    Not with context-parallel DSA (KILN_DSA_CP=1, the long-context engines): with the kernel in them, the lever-1 R8 engine's
    4096-page piece graphs fault (vector-DGE out-of-bound in every rank class, kiln-t2-cb2 box72), the compile defect of the
    4096-page bucket in another form ("The 4096-page bucket answers wrong on trn2"). Those engines keep the XLA conv, and so
    the graphs their needles passed with."""
    from .. import platform

    t = target or platform.target()  # target: the one a capture traces for (tools/compile_farm.py), else this host's
    if t is None or os.environ.get("KILN_DSA_CP", "0") == "1":
        return "xla"
    fam = platform.family_of(t)
    return "nki" if fam == "trn2" and platform.lnc(fam) == 2 else "xla"


KERNEL = os.environ.get("KILN_LA_CONV_KERNEL") or _default_kernel()
if KERNEL not in ("xla", "nki"):
    raise ValueError(f"KILN_LA_CONV_KERNEL must be xla or nki, not {KERNEL!r}")


def takes(xe: torch.Tensor, w: torch.Tensor, T: int) -> bool:
    """Whether the kernel runs this conv: KILN_LA_CONV_KERNEL=nki, on a Neuron device, a chunk or sequence form (xe 2-D [K -
    1 + T, C], T > 1), bf16 operands and whole 128-channel tiles."""
    return (KERNEL == "nki" and xe.device.type != "cpu" and xe.dim() == 2 and T > 1 and xe.shape[0] == T + w.shape[1] - 1
            and xe.shape[1] % 128 == 0 and xe.dtype == torch.bfloat16 and w.dtype == torch.bfloat16)


def conv(xe: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """_causal_conv(xe, w, T) on the device inside the caller's graph (takes() holds): y fp32 [T, C]."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    ib = torch.eye(128, dtype=xe.dtype, device=xe.device)
    i32 = torch.eye(128, dtype=torch.float32, device=xe.device)
    return wrap_nki(kiln_short_conv_kernel)[platform.nki_grid()](xe=xe.contiguous(), w=w.contiguous(), ib=ib, i32=i32,
                                                                  rev=REV)
