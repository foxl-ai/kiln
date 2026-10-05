"""Selected-expert MoE (decode) as one NKI kernel inside the LNL graph, for NeuronCore-v2 (trn1).

Why a kernel: trn1 has no hardware DMA descriptor generation, so every gathered (dynamic) DMA
packet is generated in software and a gather costs its packet count. neuronx-cc 2.27 lowers the
XLA expert gather `w[idx]` (DecoderForCausalLM._moe) to a packet count it picks per graph: 19,312
packets for one MiMo-V2.6-Flash MoE layer at decode B=4, 6.4 million for six of them in one
graph (docs/neuron-notes.md). Here every (token, expert) pair is ONE dynamic DMA of a whole
expert, whose descriptor count is fixed by the layout below (one per partition), whatever the
graph around it holds.

Why not nkilib's moe_tkg (nkilib/core/moe/moe_tkg, SDK 2.32): it does not trace for trn1, in
FP8 or bf16 ("dge_mode.hwdge is only supported for NeuronCore-v3 or newer, but current target is
... gen2", from mlp_tkg_gate_up_projection_lhs_rhs_swap.py:133; tools/probe_nkilib_moe.py), and its
FP8 modes are ROW (one scale per output channel, [E, 2, I] / [E, H]) and STATIC (one per expert);
MX weights need gen4. Kiln's experts carry one bf16 scale per 32 INPUT columns (MXFP4-derived,
models/quant.py), which no per-output-channel scale can express, and bf16 experts do not fit
(MiMo-V2.6-Flash at tp=32: 18.9 GB of experts per rank against 16 GB of HBM per NeuronCore).

Layout. One expert of one rank (H hidden, 2 Im = 128 gate/up rows, w_down stored [Im, H], block
QB = 32) becomes one uint8 row block blob[e] of [128, F] bytes, F = H + H / 2 + H / 16 + H / 32:
  [0, H)            gate_up, fp8: blob[p, c * 128 + o] = w_gu[o, c * 128 + p]
                    (partition p, H tile c, gate/up row o), so a tile is a [128 p, 128 o]
                    matmul stationary with the contraction (input) dim on partitions;
  [H, 3H/2)         down, fp8: blob[q, h'] = w_down[q % Im, (q // Im) * H/2 + h'] (two halves of
                    H stacked on the partitions, so the 64-row down weight fills all 128);
  [3H/2, 3H/2+H/16) gate_up scales, bf16 [128 o, H / 32], as stored;
  [.., F)           down scales, bf16 [128 p, H / 128]: column 2 g + j holds the scale of rows
                    32 j .. 32 j + 31 of output column h(p, g) (see below).
Same bytes as the four tensors (MiMo-V2.6-Flash: 816 KB per expert).

Math per pair (token t, expert e, routing weight w), all accumulation fp32:
- gate_up: the moving operand is x block-diagonal over the four 32-row scale blocks of a
  128-row tile, xbd[p, j] = x[c * 128 + p] [p // 32 == j], so psum[o, 4 c + j] is the partial
  dot product of row o over scale block b = 4 c + j alone; g[o] = sum_b scale[o, b] psum[o, b].
  The fp8 weights are the stationary operand as stored (nc_matmul takes fp8 x bf16 on gen2,
  nki/isa nc_matmul "Data types"), so nothing is dequantized.
- One 0/1 matmul folds the up rows (partitions 64..127) onto the gate rows and replicates both
  over the two partition halves; a = silu(gate) * up * w.
- down: moving a2[q, col] = a[q % Im] [q // 32 == col] (col = half * 2 + down scale block j);
  psum[p, 4 c' + col] over 16 chunks c' of 128 columns; y[h] = sum_j scale * psum, where
  h = half * H/2 + c' * 128 + p and g = 2 c' + half.
The kernel writes out[p, t, g] and the wrapper permutes it to [T, H].
"""

from __future__ import annotations

import torch

P = 128  # partitions
QB = 32  # expert scale block along the input dim (MiMo-V2.6 MXFP4 experts)
NBLK = P // QB  # scale blocks per 128-row tile
FP8 = torch.float8_e4m3fn


def blob_cols(H: int) -> int:
    """Bytes per partition of one packed expert (see the module docstring)."""
    return H + H // 2 + (H // QB) * 2 + (H // P) * 2 * 2


def supports(w_gu: torch.Tensor, s_gu, w_down: torch.Tensor, s_down, down_t: bool) -> bool:
    """The layout this kernel packs: FP8 experts, bf16 block-32 scales, 2 Im = 128 rows per rank,
    w_down stored [E, Im, H]."""
    if w_gu.dtype != FP8 or w_down.dtype != FP8 or s_gu is None or s_down is None or not down_t:
        return False
    E, R, H = w_gu.shape
    return (R == P and H % (2 * P) == 0 and tuple(w_down.shape) == (E, R // 2, H)
            and s_gu.dtype == torch.bfloat16 and tuple(s_gu.shape) == (E, R, H // QB)
            and s_down.dtype == torch.bfloat16 and tuple(s_down.shape) == (E, R // 2 // QB, H))


def pack(w_gu: torch.Tensor, s_gu: torch.Tensor, w_down: torch.Tensor, s_down: torch.Tensor) -> torch.Tensor:
    """w_gu fp8 [E, 128, H], s_gu bf16 [E, 128, H/32], w_down fp8 [E, 64, H] (input dim first),
    s_down bf16 [E, 2, H] -> blob uint8 [E, 128, blob_cols(H)]. Exact (a byte permutation)."""
    E, R, H = w_gu.shape
    Im, C, C2 = R // 2, H // P, H // 2 // P
    u8 = lambda t: t.contiguous().view(torch.uint8)  # noqa: E731
    gu = w_gu.view(E, R, C, P).permute(0, 3, 2, 1).reshape(E, P, C * R)  # [e, p, (c, o)]
    dw = w_down.view(E, Im, 2, H // 2).permute(0, 2, 1, 3).reshape(E, P, H // 2)  # [e, (half, i), h']
    jn = Im // QB
    # s_down[e, j, half * H/2 + c' * 128 + p] -> [e, p, (c', half, j)]
    sd = s_down.view(E, jn, 2, C2, P).permute(0, 4, 3, 2, 1).reshape(E, P, C2 * 2 * jn)
    return torch.cat([u8(gu), u8(dw), u8(s_gu), u8(sd)], dim=-1)


def _offsets(H: int) -> tuple[int, int, int]:
    """Byte offsets of the down weights, the gate_up scales and the down scales in a row."""
    return H, H + H // 2, H + H // 2 + (H // QB) * 2


def _as(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.contiguous().view(dtype)


def unpack_gu(blob: torch.Tensor, H: int, idx: torch.Tensor | None = None):
    """(w_gu fp8 [.., 128, H], s_gu bf16 [.., 128, H/32]) of blob [E, 128, F] (rows idx if given),
    as slices, views and permutes (usable inside a graph; the slices come before the gather)."""
    _, o2, o3 = _offsets(H)
    w, s = blob[..., :H], blob[..., o2:o3]
    if idx is not None:
        w, s = w[idx], s[idx]
    lead = w.shape[:-2]
    gu = _as(w, FP8).view(*lead, P, H // P, P)  # [.., p, c, o]
    return gu.permute(*range(len(lead)), -1, -2, -3).reshape(*lead, P, H), _as(s, torch.bfloat16)


def unpack_down(blob: torch.Tensor, H: int, idx: torch.Tensor | None = None):
    """(w_down fp8 [.., 64, H], s_down bf16 [.., 2, H]) of blob, as unpack_gu."""
    o1, o2, o3 = _offsets(H)
    w, s = blob[..., o1:o2], blob[..., o3:]
    if idx is not None:
        w, s = w[idx], s[idx]
    lead = w.shape[:-2]
    Im = P // 2
    jn = Im // QB
    dw = _as(w, FP8).view(*lead, 2, Im, H // 2)  # [.., half, i, h']
    sd = _as(s, torch.bfloat16).view(*lead, P, H // 2 // P, 2, jn)  # [.., p, c', half, j]
    return (dw.permute(*range(len(lead)), -2, -3, -1).reshape(*lead, Im, H),
            sd.permute(*range(len(lead)), -1, -2, -3, -4).reshape(*lead, jn, H))


def unpack(blob: torch.Tensor, H: int):
    """Inverse of pack: (w_gu, s_gu, w_down, s_down)."""
    return (*unpack_gu(blob, H), *unpack_down(blob, H))


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

# Defined at import, not inside the traced code: dynamo cannot trace a kernel object created
# while it is tracing ("id() with unsupported args ... NestedUserFunctionVariable").
if nki is not None:
    @nki.jit
    def kiln_moe_decode_v1(xT, wts, idx, blob, mask4, mhalf, fold, im: int):
        """xT bf16 [128, T, C] (x[t, c * 128 + p] at [p, t, c]), wts fp32 [128, N] (routing
        weight of pair n = t * K + k on every partition), idx int32 [1, N], blob uint8 [E, 128,
        F], mask4 fp32 [128, 4] = [p // 32 == j], mhalf bf16 [128, 2] = [p < im, p >= im], fold
        bf16 [128, 128] = [o % im == q % im]. Returns bf16 [128, T, C], y[t, h(p, g)] at [p, t, g].
        LNL keys its NKI cache on this function's source only: keep every helper inside it."""
        _, T, C = xT.shape
        N = idx.shape[1]
        K = N // T
        F = blob.shape[2]
        H = C * 128
        DW = H // 2
        C2 = DW // 128
        jn = im // 32
        o_dw, o_sg, o_sd = H, H + DW, H + DW + (H // 32) * 2
        out = nl.ndarray((128, T, C), dtype=nl.bfloat16, buffer=nl.shared_hbm)

        x_sb = nl.ndarray((128, T, C), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_sb, src=xT)
        m4 = nl.ndarray((128, 4), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=m4, src=mask4)
        mh = nl.ndarray((128, 2), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=mh, src=mhalf)
        fd = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=fd, src=fold)
        w_sb = nl.ndarray((128, N), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w_sb, src=wts)
        i_sb = nl.ndarray((1, N), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=i_sb, src=idx)
        # Block-diagonal x: xbd[p, t, c, j] = x[t, c * 128 + p] * [p // 32 == j].
        xbd = nl.ndarray((128, T, C, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
        for j in range(4):
            nisa.tensor_scalar(dst=xbd[:, :, :, j], data=x_sb, op0=nl.multiply, operand0=m4[:, j:j + 1])
        o_sb = nl.ndarray((128, T, C), dtype=nl.bfloat16, buffer=nl.sbuf)

        for t in range(T):
            acc = nl.ndarray((128, C, jn), dtype=nl.float32, buffer=nl.sbuf)
            for k in range(K):
                n = t * K + k
                buf = nl.ndarray((128, F), dtype=nl.uint8, buffer=nl.sbuf)
                e = i_sb.ap(pattern=[[N, 1], [1, 1]], offset=n)
                nisa.dma_copy(dst=buf, src=blob.select(0, e))
                wg = buf[:, 0:H].view(nl.float8_e4m3)
                pg = nl.ndarray((128, C * 4), dtype=nl.float32, buffer=nl.psum)
                for c in range(C):
                    nisa.nc_matmul(dst=pg[:, c * 4:(c + 1) * 4], stationary=wg[:, c * 128:(c + 1) * 128],
                                   moving=xbd[:, t, c, :], accumulate=False)
                q = nl.ndarray((128, C * 4), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=q, data1=pg, data2=buf[:, o_sg:o_sd].view(nl.bfloat16), op=nl.multiply)
                g = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=g, op=nl.add, data=q, axis=1)
                mm = nl.ndarray((128, 2), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=mm, data=mh, op0=nl.multiply, operand0=g)
                pf = nl.ndarray((128, 2), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=pf, stationary=fd, moving=mm, accumulate=False)
                sl = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=sl, op=nl.silu, data=pf[:, 0:1])
                au = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=au, data=sl, op0=nl.multiply, operand0=pf[:, 1:2],
                                   op1=nl.multiply, operand1=w_sb[:, n:n + 1])
                a2 = nl.ndarray((128, 4), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=a2, data=m4, op0=nl.multiply, operand0=au)
                wd = buf[:, o_dw:o_sg].view(nl.float8_e4m3)
                pd = nl.ndarray((128, C2 * 4), dtype=nl.float32, buffer=nl.psum)
                for c in range(C2):
                    nisa.nc_matmul(dst=pd[:, c * 4:(c + 1) * 4], stationary=wd[:, c * 128:(c + 1) * 128],
                                   moving=a2, accumulate=False)
                sd = buf[:, o_sd:F].view(nl.bfloat16)
                if k == 0:
                    nisa.tensor_tensor(dst=acc.reshape((128, C * jn)), data1=pd, data2=sd, op=nl.multiply)
                else:
                    q2 = nl.ndarray((128, C * jn), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=q2, data1=pd, data2=sd, op=nl.multiply)
                    nisa.tensor_tensor(dst=acc.reshape((128, C * jn)), data1=acc.reshape((128, C * jn)),
                                       data2=q2, op=nl.add)
            nisa.tensor_reduce(dst=o_sb[:, t, :], op=nl.add, data=acc, axis=2)
        nisa.dma_copy(dst=out, src=o_sb)
        return out
else:
    kiln_moe_decode_v1 = None


def kernel():
    """The NKI kernel (raises where the NKI package is missing)."""
    if kiln_moe_decode_v1 is None:
        raise RuntimeError("the NKI MoE kernel needs the nki package (the Neuron venv)")
    return kiln_moe_decode_v1


def emulate(x: torch.Tensor, topv: torch.Tensor, topi: torch.Tensor, blob: torch.Tensor) -> torch.Tensor:
    """The kernel's arithmetic in torch, step for step (fp32 accumulation, g and a * w rounded
    to bf16 where the kernel stores them as matmul operands): the CPU reference for its layout
    and math."""
    import torch.nn.functional as F

    T, H = x.shape
    K = topi.shape[1]
    N, im = T * K, P // 2
    jn = im // QB
    w_gu, s_gu, w_down, s_down = unpack(blob[topi.reshape(N)], H)
    xs = x.float().repeat_interleave(K, dim=0)  # [N, H], row n = token n // K
    part = (w_gu.float() * xs.unsqueeze(1)).view(N, P, H // QB, QB).sum(-1)  # per scale block
    g = (part * s_gu.float()).sum(-1).bfloat16().float()  # [N, 128]: gate rows, then up rows
    a = (F.silu(g[:, :im]) * g[:, im:] * topv.reshape(N, 1).float()).bfloat16().float()  # [N, im]
    pdj = (a.view(N, jn, QB, 1) * w_down.float().view(N, jn, QB, H)).sum(2)  # [N, jn, H]
    y = (pdj * s_down.float()).sum(1)  # [N, H]
    return y.view(T, K, H).sum(1).to(x.dtype)


def consts(im: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(mask4, mhalf, fold): the kernel's 0/1 constants (see kiln_moe_decode_v1)."""
    p = torch.arange(P, device=device)
    mask4 = (p.view(P, 1) // QB == torch.arange(NBLK, device=device).view(1, NBLK)).float()
    mhalf = torch.stack([p < im, p >= im], dim=1).to(torch.bfloat16)
    fold = (p.view(P, 1) % im == p.view(1, P) % im).to(torch.bfloat16)
    return mask4, mhalf, fold


def kernel_inputs(x: torch.Tensor, topv: torch.Tensor, topi: torch.Tensor, blob: torch.Tensor):
    """The kernel's arguments for x [T, H], routing weights / experts [T, K] and one layer's blob."""
    T, H = x.shape
    N = topi.numel()
    im = P // 2
    xT = x.view(T, H // P, P).permute(2, 0, 1).contiguous()
    wts = topv.float().reshape(1, N).expand(P, N).contiguous()
    idx = topi.reshape(1, N).to(torch.int32)
    mask4, mhalf, fold = consts(im, x.device)
    return dict(xT=xT, wts=wts, idx=idx, blob=blob, mask4=mask4, mhalf=mhalf, fold=fold, im=im)


def from_kernel(out: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Kernel output [128, T, H / 128] (column g = 2 c' + half) -> [T, H]."""
    _, T, C = out.shape
    return out.view(P, T, C // 2, 2).permute(1, 3, 2, 0).reshape(T, C * P).to(dtype)


def moe_selected(x: torch.Tensor, topv: torch.Tensor, topi: torch.Tensor, blob: torch.Tensor,
                 max_pairs: int = 512) -> torch.Tensor:
    """sum_k topv[t, k] * expert_{topi[t, k]}(x[t]) on the device, inside the caller's graph: one
    kernel call per chunk of tokens holding at most max_pairs pairs (the kernel unrolls its pair
    loop, so its instruction count and compile time grow with the pairs it holds: 2.6 s for 8,
    24 s for 512 on trn1.2xlarge), every full chunk the same compiled kernel."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    call = wrap_nki(kernel())[platform.nki_grid()]
    T, K = topi.shape
    step = max(1, max_pairs // K)
    outs = [from_kernel(call(**kernel_inputs(x[s : s + step], topv[s : s + step], topi[s : s + step], blob)), x.dtype)
            for s in range(0, T, step)]
    return outs[0] if len(outs) == 1 else torch.cat(outs)
