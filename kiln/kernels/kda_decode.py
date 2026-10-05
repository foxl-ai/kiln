"""One decode step of KDA (Kimi delta attention) for B sequences as one NKI kernel for NeuronCore-v2 (trn1),
behind KILN_KDA_DECODE_KERNEL=nki: each row's recurrent state is read from the state pool, updated and written
back IN PLACE, and the step's output returned.

Why a kernel: in a GLM-5.3-Flash decode layer-group graph (tp=32, DP attention 4, attention TP 8: 8 heads of
[128, 128] fp32 per rank) neuronx-cc lays the updated state out for its arithmetic and then rewrites it into the
pool's layout with 1024 vector-engine STREAM_TRANSPOSEs per layer (32 x 32 fp32 blocks, ~0.24 ms per layer),
scheduled at the end of the graph together with the pool scatters. That tail grew faster than the rows: 0.68 /
2.75 / 7.72 / 13.76 ms for the graph's 9 KDA layers at 8 / 16 / 32 / 64 rows per DP group (neuron-explorer
replays of the layers 12-23 decode graph on 32 cores, rank 0, trn1.32xlarge, SDK 2.32, 2026-10-04;
docs/neuron-notes.md "Where a GLM-5.3-Flash decode step goes, from a device profile").

Arithmetic (linear_attn.recurrent_step, fp32): e = exp(g); r0 = (q e)^T S, r1 = (k e)^T S;
delta = beta (v - r1); o = r0 + (q . k) delta; S' = diag(e) S + k delta^T.

Layout: q, k, v and g come in as row-major [B, H, 128] and are transposed once on the tensor engine into
[128 d, B H] column tiles (q e, k e, k, v, e), with beta and q . k as rows on partition 0. Per sequence row, its
states are one DMA from the pool row slots[0, b] into SBUF [128 dk, H, 128 dv] (the pool's row-major layout puts
dk on the partitions, no transpose); a read row >= R (padding, a sequence at position 0) skips the DMA and leaves
the zeroed tile, i.e. the state starts at zero as torch.where(keep, S, 0) does. Per head, four matmuls with a
one-column stationary land r0, r1 and the rows of v and k on partition 0 (a column against the head's state, or
against the identity); delta and o are row ops there; k delta^T is a matmul contracting over one partition; S' =
S e + k delta^T is one scalar_tensor_tensor; the row's states are DMAed back to the pool row slots[1, b]. The
pool is returned as the kernel's output (NKI detect_must_alias: an input returned as an output).
"""

from __future__ import annotations

import os

import torch

RING = 4  # rows in flight: their state tiles (each ring entry a tensor of its own)
PD = 2  # rows whose states are loaded ahead of the row being computed (PD < RING)
HR = 6  # heads in flight: their PSUM and partition-0 row tiles (HR > SKEW)
SKEW = 2  # heads between a head's stage A and its stage B


def emulate(pool: torch.Tensor, slots: torch.Tensor, q, k, v, g, beta):
    """The kernel's arithmetic in torch (CPU): (o [B, H, D], the pool after the step). slots [2, B]: read rows
    (>= R: start from zero) and write rows, written in row order (a later row wins a shared write row)."""
    from ..models.linear_attn import recurrent_step

    R = pool.shape[0]
    B = q.shape[0]
    S = torch.zeros(B, *pool.shape[1:], dtype=torch.float32)
    rd = slots[0].long()
    ok = rd < R
    S[ok] = pool[rd[ok]].float()
    o, S2 = recurrent_step(q.float(), k.float(), v.float(), g.float(), beta.float(), S)
    out = pool.clone()
    for b, w in enumerate(slots[1].long().tolist()):
        if w < R:
            out[w] = S2[b]
    return o, out


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    F32 = nl.float32
    I32 = nl.int32

    def _sb(shape, dt=None):
        return nl.ndarray(shape, dtype=dt or F32, buffer=nl.sbuf)

    def _ps(shape):
        return nl.ndarray(shape, dtype=F32, buffer=nl.psum)

    def _load(X, b):
        """Row b's states into its ring entry (prefetched PD rows ahead); a read row >= R skips the DMA."""
        B, H, D = X["B"], X["H"], X["D"]
        S = X["Sr"][b % RING]
        nisa.memset(dst=S, value=0.0, engine=nisa.gpsimd_engine)
        ridx = X["sl"].ap(pattern=[[2 * B, 1], [1, 1]], offset=b)
        nisa.dma_copy(dst=S, src=X["pool"].ap(pattern=[[D, 128], [D * D, H], [1, D]], offset=0, scalar_offset=ridx,
                                              indirect_dim=0), oob_mode=nisa.oob_mode.skip)

    def _stage_a(X, c):
        """Head c = b H + j: r0, r1 - v and k's row on partition 0, then delta and o."""
        H = X["H"]
        b, j = c // H, c % H
        S, orow = X["Sr"][b % RING], X["Or"][b % RING]
        pr, dl, kr = X["Pr"][c % HR], X["Dl"][c % HR], X["Kr"][c % HR]
        nisa.nc_matmul(dst=pr[:, 0, :], stationary=X["QE"][:, c:c + 1], moving=S[:, j, :], accumulate=False)
        nisa.nc_matmul(dst=pr[:, 1, :], stationary=X["KE"][:, c:c + 1], moving=S[:, j, :], accumulate=False)
        nisa.nc_matmul(dst=pr[:, 1, :], stationary=X["NVT"][:, c:c + 1], moving=X["I"], accumulate=True)
        nisa.nc_matmul(dst=pr[:, 2, :], stationary=X["KT"][:, c:c + 1], moving=X["I"], accumulate=False)
        # delta = beta (v - r1) = -beta (r1 - v); o = r0 + (q . k) delta
        nisa.activation(dst=dl, op=nl.copy, data=pr[:, 1, :], scale=X["NBR"][:, c:c + 1])
        nisa.activation(dst=kr, op=nl.copy, data=pr[:, 2, :])
        nisa.scalar_tensor_tensor(dst=orow[:, j, :], data=dl, op0=nl.multiply, operand0=X["QK"][:, c:c + 1],
                                  op1=nl.add, operand1=pr[:, 0, :])

    def _stage_b(X, c):
        """Head c: S' = diag(e) S + k delta^T; after a row's last head, its o row and states go out."""
        B, H, D = X["B"], X["H"], X["D"]
        b, j = c // H, c % H
        S, Sx, orow = X["Sr"][b % RING], X["Sn"][b % RING], X["Or"][b % RING]
        po = X["Po"][c % HR]
        nisa.nc_matmul(dst=po, stationary=X["Kr"][c % HR], moving=X["Dl"][c % HR], accumulate=False)
        nisa.scalar_tensor_tensor(dst=Sx[:, j, :], data=S[:, j, :], op0=nl.multiply, operand0=X["E"][:, c:c + 1],
                                  op1=nl.add, operand1=po)
        if j == H - 1:
            nisa.dma_copy(dst=X["of"][:, b * H * D:(b + 1) * H * D], src=orow.flatten_dims(1, 2))
            widx = X["sl"].ap(pattern=[[2 * B, 1], [1, 1]], offset=B + b)
            nisa.dma_copy(dst=X["pool"].ap(pattern=[[D, 128], [D * D, H], [1, D]], offset=0, scalar_offset=widx,
                                           indirect_dim=0), src=Sx, oob_mode=nisa.oob_mode.skip)

    @nki.jit
    def kiln_kda_decode_kernel(pool, slots, q, k, v, g, beta, ident, rev: int):
        """pool fp32 [R, H, D, D] (rows slots[1] rewritten in place and returned), slots int32 [2, B] (read rows,
        R = none; write rows), q (scaled), k, v, g (log decay) fp32 [B, H, D], beta fp32 [B, H], ident fp32
        [128, 128] the identity (PE transposes); rev: this module's kernel source revision. Returns o fp32
        [B, H, D] and pool."""
        R, H, D, _ = pool.shape
        B = q.shape[0]
        N = B * H
        o = nl.ndarray((B, H, D), dtype=F32, buffer=nl.shared_hbm)
        sl = _sb((1, 2 * B), I32)
        nisa.dma_copy(dst=sl, src=slots.reshape((1, 2 * B)))
        I = _sb((128, 128))
        nisa.dma_copy(dst=I, src=ident)
        ones = _sb((128, 1))
        nisa.memset(dst=ones, value=1.0, engine=nisa.gpsimd_engine)
        # Column tiles [128 d, N], column c = b H + h: q e, k e, k, -v, e; on partition 0: beta, -beta and q . k.
        QE, KE, KT, NVT, E = _sb((128, N)), _sb((128, N)), _sb((128, N)), _sb((128, N)), _sb((128, N))
        BR = _sb((1, N))
        nisa.dma_copy(dst=BR, src=beta.reshape((1, N)))
        NBR = _sb((1, N))  # -beta
        nisa.tensor_scalar(dst=NBR, data=BR, op0=nl.multiply, operand0=-1.0, engine=nisa.vector_engine)
        QK = _sb((1, N))
        q2, k2, v2, g2 = q.reshape((N, D)), k.reshape((N, D)), v.reshape((N, D)), g.reshape((N, D))
        for t in range(0, N, 128):
            n = min(128, N - t)
            rows = []
            for src in (q2, k2, v2, g2):
                a = _sb((128, D))
                nisa.dma_copy(dst=a[0:n, :], src=src[t:t + n, :])
                rows.append(a)
            pt = _ps((128, 4, 128))
            for i in range(4):
                nisa.nc_matmul(dst=pt[:, i, 0:n], stationary=rows[i][0:n, :], moving=I[0:n, 0:n], accumulate=False)
            nisa.activation(dst=E[:, t:t + n], op=nl.exp, data=pt[:, 3, 0:n])
            nisa.tensor_tensor(dst=QE[:, t:t + n], data1=pt[:, 0, 0:n], data2=E[:, t:t + n], op=nl.multiply)
            nisa.tensor_tensor(dst=KE[:, t:t + n], data1=pt[:, 1, 0:n], data2=E[:, t:t + n], op=nl.multiply)
            nisa.activation(dst=KT[:, t:t + n], op=nl.copy, data=pt[:, 1, 0:n])
            nisa.activation(dst=NVT[:, t:t + n], op=nl.copy, data=pt[:, 2, 0:n], scale=-1.0)
            qkp = _sb((128, 128))
            nisa.tensor_tensor(dst=qkp[:, 0:n], data1=pt[:, 0, 0:n], data2=KT[:, t:t + n], op=nl.multiply)
            pq = _ps((1, 128))
            nisa.nc_matmul(dst=pq[:, 0:n], stationary=ones, moving=qkp[:, 0:n], accumulate=False)  # sum over d
            nisa.activation(dst=QK[:, t:t + n], op=nl.copy, data=pq[:, 0:n])
        of = o.reshape((1, N * D))
        # Rings: every entry a tensor of its own (the compiler tracks dependencies per tensor).
        Sr, Sn, Or = [], [], []
        for _ in range(RING):
            Sr.append(_sb((128, H, D)))
            Sn.append(_sb((128, H, D)))
            Or.append(_sb((1, H, D)))
        Pr, Po, Dl, Kr = [], [], [], []
        for _ in range(HR):
            Pr.append(_ps((1, 3, D)))
            Po.append(_ps((128, D)))
            Dl.append(_sb((1, D)))
            Kr.append(_sb((1, D)))
        X = {"pool": pool, "sl": sl, "Sr": Sr, "Sn": Sn, "Or": Or, "Pr": Pr, "Po": Po, "Dl": Dl, "Kr": Kr, "QE": QE,
             "KE": KE, "KT": KT, "NVT": NVT, "E": E, "I": I, "NBR": NBR, "QK": QK, "of": of, "B": B, "H": H, "D": D}
        # Software pipeline: the in-order engine queues get head c's stage A before head c - SKEW's stage B, so the
        # tensor engine's outer product of a head never waits on the scalar engine's copies of the same head.
        for b in range(min(PD, B)):
            _load(X, b)
        for c in range(N + SKEW):
            if c < N and c % H == 0 and c // H + PD < B:
                _load(X, c // H + PD)
            if c < N:
                _stage_a(X, c)
            if c >= SKEW:
                _stage_b(X, c - SKEW)
        return o, pool
else:
    kiln_kda_decode_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source, passed as the static argument `rev` (LNL's compile-cache key does
    not include NKI kernel source: CLAUDE.md, kernels/delta_rule.py _kernel_rev)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_kda_decode_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev()


def takes(Dk: int, Dv: int, kda: bool) -> bool:
    """Whether the kernel runs a decode step of these shapes: KDA (per-key decay), 128 x 128 heads."""
    return kda and Dk == 128 and Dv == 128


def decode_step(pool: torch.Tensor, state_slot: torch.Tensor, keep: torch.Tensor, q, k, v, g, beta) -> torch.Tensor:
    """One decode step on the device, in the caller's graph: pool fp32 [R, H, D, D] updated in place at rows
    state_slot [B]; keep [B] bool (False: the row starts from a zero state, as torch.where(keep, S, 0) does);
    q (scaled), k, v, g fp32 [B, H, D]; beta [B, H]. Returns o fp32 [B, H, D]."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if kiln_kda_decode_kernel is None:
        raise RuntimeError("the NKI KDA decode kernel needs the nki package (the Neuron venv)")
    R = pool.shape[0]
    rd = torch.where(keep, state_slot, torch.full_like(state_slot, R))
    slots = torch.stack([rd, state_slot]).to(torch.int32)
    ident = torch.eye(128, dtype=torch.float32, device=q.device)
    o, _ = wrap_nki(kiln_kda_decode_kernel)[platform.nki_grid()](
        pool=pool, slots=slots, q=q.float().contiguous(), k=k.float().contiguous(), v=v.float().contiguous(),
        g=g.float().contiguous(), beta=beta.float().contiguous(), ident=ident, rev=REV)
    return o


KERNEL = os.environ.get("KILN_KDA_DECODE_KERNEL", "xla")
if KERNEL not in ("xla", "nki"):
    raise ValueError(f"KILN_KDA_DECODE_KERNEL must be xla or nki, not {KERNEL!r}")
