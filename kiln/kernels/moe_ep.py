"""Expert-parallel MoE (KILN_MOE_EP=1): each tensor-parallel rank holds WHOLE experts (E / tp of them,
every one of its intermediate rows) instead of a 1 / tp slice of every expert, and computes only the
(token, expert) pairs routed to its own experts, as one NKI kernel for NeuronCore-v2 (trn1).

Why no all-to-all: at the MoE input every rank already holds every row of the batch (DP attention runs
the MLP over all groups' rows; sequence-parallel prefill streams gather the block's input to every rank by
a world all-reduce, models/decoder.py _sp_gather), and the block's output is summed over the world anyway
(the shared expert is still tensor-parallel). So a rank's routed output is the sum over ITS experts'
pairs, zero for every other pair, and the block's existing all-reduce adds the ranks' outputs: the expert
parallel layout moves no extra byte between ranks (vLLM's all-gather / reduce-scatter EP, with the gather
and the reduction the TP block already pays).

What it saves (docs/neuron-notes.md "Expert parallelism"): under TP each rank gathers the x row of EVERY
routed pair, writes EVERY pair's full-width y and re-reads it in the combine (~1.2 GB of DMA per rank per
MoE layer at 4096 rows), on 64-row slices that keep every vector instruction small. Here a rank reads its
own pairs' rows only (1 / tp of them) and applies each local expert to all of its rows at once.

Layout (pack(), per local expert e, from the loader's EP tensors: w_gu [El, 2 I, H] fp8 gate rows then
up rows with fp32 scales [El, 2 I, H / 128] per (row, 128 columns), w_down stored [El, I, H] (input dim
first) with fp32 scales [El, I / 128, H] per (128 input rows, output column), both after
models/quant.fit_e4m3_max):
  gu  uint8 [El, 128 p, M, 2, CT, 128 i]: gu[e, p, m, g, c, i] = w_gu[e, g I + m 128 + i, c 128 + p], the
      stationary [h, i] tiles of I-chunk m (M = I / 128 chunks), gate (g 0) and up (g 1), h-tile c;
  sgu bf16 [El, M, 2, CT / 4, 3, 512]: the scale of w_gu row (g, m, i) and h-block c at column
      (c % 4) 128 + i of chunk c // 4, as three bf16 parts hi + mid + lo (exactly the fp32 value);
  dn  uint8 [El, 128 p, M, H]: dn[e, p, m, h] = w_down[e, m 128 + p, h] (the moving [i, h] rows of
      I-chunk m);
  sdn bf16 [El, H / 512, M, 3, 512]: the down scale of (m, h = q 512 + j) in three parts;
  tsg fp32 [El, M, 2, CT], tsd fp32 [El, M, CT] (only where the scales are block-constant, as the EP loader fits
      a 128 x 128 block-scaled checkpoint): one scale per kernel tile, gate_up tile (m, g, c) and the down tile of
      I-chunk m and output columns b 128 .. b 128 + 127. The kernel then dequantizes each [128, 128] tile by one
      per-partition-scalar instruction, alternately on the vector and the scalar engine (TILES), instead of
      broadcasting per-row scale rows.
The scales vary along a tile's free axis (per output row of gate_up, per output column of down), so a
tile cannot take them as a per-partition scalar: each 512-column scale row is broadcast to all 128
partitions by three accumulating 0/1 matmuls (hi, mid, lo: exact, every partial sum is an fp32 value)
into PSUM, and the vector engine dequantizes the fp8 tile against it (one PSUM operand: full rate).

A PASS (one local expert, up to LW lanes = rows of its pairs): x rows gathered and transposed to [h,
lane] on the tensor engine; per I-chunk m: gate and up tiles dequantized to bf16 (fp8 x scale in fp32,
rounded once: models/quant.dequant), accumulated over the 32 h-tiles in PSUM with x^T as the moving
operand, g and u rounded to bf16, a = silu(min(g, lim)) clamp(u, -lim, lim) rounded to bf16 (the
transposed activation a^T [i, lane] of chunk m); then per 512-column chunk q of the output: the down tiles
of every m dequantized, y = sum_m a_m W_down,m accumulated in PSUM over the whole intermediate dimension,
rounded once to bf16. emulate() is that arithmetic in torch.
"""

from __future__ import annotations

import os

import torch

P = 128
# The dequantize-first kernel's scale form: KILN_MOE_EP_TILES=1 (default) takes one scale per [128, 128] tile where
# pack() finds the scales block-constant (blob "tsg" / "tsd", and then no per-row split scales), each tile
# dequantized by one instruction on the vector or the scalar engine; 0 (or per-row scales) the per-row form (each
# 512-column scale row broadcast to the partitions by a matmul, the vector engine alone). The same products either
# way (bf16(fp32 code x scale)). Read when a layer is packed.
TILES = os.environ.get("KILN_MOE_EP_TILES", "1") == "1"


def split3(s: torch.Tensor) -> torch.Tensor:
    """fp32 s [...] -> bf16 [..., 3] (hi, mid, lo) with hi + mid + lo == s exactly, summed in that order in
    fp32 (hi + mid has at most 17 significant bits, and the last sum's exact result is s itself)."""
    s = s.float()
    hi = s.bfloat16()
    r1 = s - hi.float()
    mid = r1.bfloat16()
    lo = (r1 - mid.float()).bfloat16()
    if not torch.equal((hi.float() + mid.float()) + lo.float(), s):
        raise ValueError("an fp32 scale does not split into three bf16 parts exactly")
    return torch.stack([hi, mid, lo], dim=-1)


def supports(w_gu, s_gu, w_down, s_down, down_t: bool) -> bool:
    """The shapes pack() takes: FP8 whole experts with fp32 128 x 128 block scales as the loader keeps them (per
    row and 128 input columns for gate_up, per 128 input rows and output column for down, stored [E, I, H]),
    I a multiple of 128 and H of 512."""
    from ..models.quant import FP8

    if w_gu.dtype != FP8 or w_down.dtype != FP8 or s_gu is None or s_down is None or not down_t:
        return False
    El, R, H = w_gu.shape
    I = R // 2
    return (R % 256 == 0 and H % 512 == 0 and tuple(w_down.shape) == (El, I, H) and s_gu.dtype == torch.float32
            and s_down.dtype == torch.float32 and tuple(s_gu.shape) == (El, R, H // P)
            and tuple(s_down.shape) == (El, I // P, H))


def pack(w_gu: torch.Tensor, s_gu: torch.Tensor, w_down: torch.Tensor, s_down: torch.Tensor,
         tiles: bool = False) -> dict:
    """The kernel's layout (module docstring) of El whole experts as the loader holds them under EP. tiles: the
    scales are block-constant (the loader fitted a 128 x 128 block-scaled checkpoint per block,
    models/loader.py EP_FIT; decided from the checkpoint's format, never from values, so a capture on zeros
    packs as the device does): add the tile scales tsg / tsd, checked against the values (a ValueError if a
    tile's scales differ), and with TILES leave out the per-row split scales the kernel then never reads."""
    from ..models.quant import FP8

    El, R, H = w_gu.shape
    I = R // 2
    M, CT = I // P, H // P
    if w_gu.dtype != FP8 or w_down.dtype != FP8 or s_gu.dtype != torch.float32 or s_down.dtype != torch.float32:
        raise ValueError("moe_ep.pack takes FP8 experts with fp32 128-block scales")
    if tuple(w_down.shape) != (El, I, H) or tuple(s_gu.shape) != (El, R, CT) or tuple(s_down.shape) != (El, M, H):
        raise ValueError(f"moe_ep.pack: shapes {tuple(w_gu.shape)} {tuple(s_gu.shape)} {tuple(w_down.shape)} "
                         f"{tuple(s_down.shape)} are not whole experts with 128-block scales")
    gu = w_gu.view(torch.uint8).view(El, 2, M, P, CT, P).permute(0, 5, 2, 1, 4, 3).contiguous()
    sg = s_gu.view(El, 2, M, P, CT).permute(0, 2, 1, 4, 3).reshape(El, M, 2, CT // 4, 4 * P)
    sgu = split3(sg).permute(0, 1, 2, 3, 5, 4).contiguous()  # [El, M, 2, CT / 4, 3, 512]
    dn = w_down.view(torch.uint8).view(El, M, P, H).permute(0, 2, 1, 3).contiguous()
    sd = s_down.view(El, M, H // 512, 512).permute(0, 2, 1, 3)  # [El, Q, M, 512]
    sdn = split3(sd).permute(0, 1, 2, 4, 3).contiguous()  # [El, Q, M, 3, 512]
    # The small-lane kernel's scales, fp32 with the rows / columns they scale on the partitions:
    # dsg[e, g, m, i, c] = s_gu[e, g I + m 128 + i, c] (the loader's layout as it is), dsd[e, h % 128, h // 128, m].
    dsg = s_gu.reshape(El, 2, M, P, CT).contiguous()
    dsd = s_down.view(El, M, CT, P).permute(0, 3, 2, 1).contiguous()
    blob = {"gu": gu, "sgu": sgu, "dn": dn, "sdn": sdn, "dsg": dsg, "dsd": dsd}
    # Block-constant scales (a 128 x 128 block-scaled checkpoint fitted per block, models/loader.py EP_FIT): one
    # scale per kernel tile, tsg[e, m, g, c] for gate_up tile (m, g, c) and tsd[e, m, b] for the down rows of
    # I-chunk m and output columns b 128 .. b 128 + 127 (the dequantize-first kernel's tile-scale form, TILES).
    if tiles:
        sg5 = s_gu.view(El, 2, M, P, CT)
        sd4 = s_down.view(El, M, CT, P)
        if not (torch.equal(sg5, sg5[:, :, :, :1].expand_as(sg5)) and torch.equal(sd4, sd4[..., :1].expand_as(sd4))):
            raise ValueError("moe_ep.pack(tiles=True): the experts' scales are not constant over each 128 x 128 tile "
                             "(the loader fits a block-scaled checkpoint per block only with KILN_MOE_EP_FIT=block)")
        blob["tsg"] = sg5[:, :, :, 0].permute(0, 2, 1, 3).contiguous()  # [El, M, 2, CT]
        blob["tsd"] = sd4[..., 0].contiguous()  # [El, M, CT]
        if TILES:  # the per-row split scales are then never read (10.5 MB per GLM-5.3-Flash layer and rank)
            del blob["sgu"], blob["sdn"]
    return blob


def unpack(blob: dict):
    """(w_gu, s_gu, w_down, s_down) back from pack()'s layout (the CPU check of the layout)."""
    from ..models.quant import FP8

    gu, dn = blob["gu"], blob["dn"]
    El, _, M, _, CT, _ = gu.shape
    H = CT * P
    w_gu = gu.permute(0, 3, 2, 5, 4, 1).reshape(El, 2 * M * P, H).contiguous().view(FP8)
    w_down = dn.permute(0, 2, 1, 3).reshape(El, M * P, H).contiguous().view(FP8)
    if "sgu" not in blob:  # the tile-scale form keeps the scales as the small-lane copies (and per tile)
        s_gu = blob["dsg"].reshape(El, 2 * M * P, CT).contiguous()
        s_down = blob["dsd"].permute(0, 3, 2, 1).reshape(El, M, H).contiguous()
        if not (torch.equal(blob["tsg"].permute(0, 2, 1, 3).unsqueeze(3).expand(El, 2, M, P, CT).reshape(s_gu.shape),
                            s_gu)
                and torch.equal(blob["tsd"].unsqueeze(-1).expand(El, M, CT, P).reshape(El, M, H), s_down)):
            raise ValueError("moe_ep blob: the tile scales differ from the small-lane ones")
        return w_gu, s_gu, w_down, s_down
    sgu, sdn = blob["sgu"], blob["sdn"]
    s3 = (sgu[..., 0, :].float() + sgu[..., 1, :].float()) + sgu[..., 2, :].float()
    s_gu = s3.view(El, M, 2, CT, P).permute(0, 2, 1, 4, 3).reshape(El, 2 * M * P, CT).contiguous()
    d3 = (sdn[..., 0, :].float() + sdn[..., 1, :].float()) + sdn[..., 2, :].float()  # [El, Q, M, 512]
    s_down = d3.permute(0, 2, 1, 3).reshape(El, M, H).contiguous()
    if "dsg" in blob:  # the small-lane copies agree with the split ones
        if not (torch.equal(blob["dsg"].reshape(El, 2 * M * P, CT), s_gu)
                and torch.equal(blob["dsd"].permute(0, 3, 2, 1).reshape(El, M, H), s_down)):
            raise ValueError("moe_ep blob: the small-lane scales differ from the split ones")
    return w_gu, s_gu, w_down, s_down


def glu_ref(g: torch.Tensor, u: torch.Tensor, act: int, lim: float) -> torch.Tensor:
    from .moe_dedupe import glu

    return glu(g, u, act, lim)


def expert_out(x: torch.Tensor, w_gu, s_gu, w_down, s_down, act: int = 1, lim: float = 10.0,
               rounded: bool = True) -> torch.Tensor:
    """One whole expert on rows x [n, H] with the kernel's arithmetic: gate_up and down dequantized to bf16
    (models/quant.dequant), g = bf16(x W_g^T), u likewise (fp32 sums), a = bf16(glu(g, u)), y = a W_down with the sum
    over the whole intermediate dimension in fp32, rounded to bf16 (rounded=False: left in fp32). fp32 [n, H]."""
    from ..models.quant import dequant, dequant_t

    I = w_gu.shape[0] // 2
    wg = dequant(w_gu, s_gu, torch.bfloat16).float()
    wd = dequant_t(w_down, s_down, torch.bfloat16).float()
    gu = x.float() @ wg.T
    g, u = gu[:, :I].bfloat16().float(), gu[:, I:].bfloat16().float()
    a = glu_ref(g, u, act, lim).bfloat16().float()
    y = a @ wd
    return y.bfloat16().float() if rounded else y


def expert_out_small(x: torch.Tensor, w_gu, s_gu, w_down, s_down, act: int = 1, lim: float = 10.0) -> torch.Tensor:
    """One whole expert with the small-lane kernel's arithmetic: per 128-column tile of each weight the fp32 dot
    product with the codes as stored, times that tile's per-row scale, summed in fp32 (moe_dedupe's per-row
    arithmetic); g, u rounded to bf16; a = bf16(glu); y likewise over the down tiles (one per 128 input rows),
    rounded to bf16. fp32 [n, H] (bf16 values)."""
    R, H = w_gu.shape
    I = R // 2
    CT, M = H // P, I // P
    xp = x.float().view(-1, CT, P)
    part = torch.einsum("ncp,ocp->noc", xp, w_gu.float().view(R, CT, P))  # [n, 2I, CT]
    gu = (part * s_gu.float().unsqueeze(0)).sum(-1)
    g, u = gu[:, :I].bfloat16().float(), gu[:, I:].bfloat16().float()
    a = glu_ref(g, u, act, lim).bfloat16().float().view(-1, M, P)
    dp = torch.einsum("nmp,mph->nhm", a, w_down.float().view(M, P, H))  # [n, H, M]
    return (dp * s_down.float().t().unsqueeze(0)).sum(-1).bfloat16().float()


def emulate(x, topv, topi, lmap, w_gu, s_gu, w_down, s_down, act: int = 1, lim: float = 10.0,
            small: bool | int | None = None) -> torch.Tensor:
    """The kernel's output in torch: for every token, its pairs whose expert is local (lmap[e] < El, the local
    index; w_* are the El local experts) in local-expert order, out = bf16(out + bf16(w y)) with y the expert's fp32
    output (kiln_moe_ep_kernel: the drain scales by w and rounds once, a bf16 read-modify-write adds). small (default:
    the kernel moe_ep() runs for this many rows): kiln_moe_ep_small's arithmetic, y = expert_out_small (bf16); small=2
    kiln_moe_ep_small2's: experts with at most SMALL_LW pairs that way, first, then the others dequantized
    (expert_out), each group in local-expert order."""
    El = w_gu.shape[0]
    T, H = x.shape
    if small is None:
        small = (2 if SMALL_V in (2, 3, 5) else True) if uses_small(T) else False
    out = torch.zeros(T, H, dtype=torch.bfloat16)
    loc = lmap.view(-1)[topi.long()]  # [T, k] local index or El
    order = list(range(El))
    if small == 2:
        n = [int((loc == le).sum()) for le in range(El)]
        order = [le for le in order if n[le] <= SMALL_LW] + [le for le in order if n[le] > SMALL_LW]
    for le in order:
        t, k = (loc == le).nonzero(as_tuple=True)
        if t.numel() == 0:
            continue
        if small == 2 and t.numel() > SMALL_LW:
            y = expert_out(x[t], w_gu[le], s_gu[le], w_down[le], s_down[le], act, lim, rounded=False)
        elif small:
            y = expert_out_small(x[t], w_gu[le], s_gu[le], w_down[le], s_down[le], act, lim)
        else:
            y = expert_out(x[t], w_gu[le], s_gu[le], w_down[le], s_down[le], act, lim, rounded=False)
        v = (y * topv[t, k].float().unsqueeze(1)).bfloat16()
        out[t] = (out[t].float() + v.float()).bfloat16()
    return out.to(x.dtype)


# --- kernel ---------------------------------------------------------------------------------------

try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from nki.isa.constants import oob_mode
except ImportError:
    nki = None


if nki is not None:
    DGU = nisa.dge_mode.unknown  # the compiler picks the DGE mode (dma_copy's default)

    def _consts(bc: int):
        """The 0/1 stationaries [3, 128] of the scale broadcast: bc 1 one all-ones (hi + mid + lo summed inside
        one matmul), bc 3 three row selectors (one matmul per part, accumulated in PSUM in that order)."""
        pi = nl.ndarray((3, 128), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=pi, pattern=[[0, 128]], offset=0, channel_multiplier=1)  # the partition index
        pf = nl.ndarray((3, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=pf, src=pi, engine=nisa.vector_engine)
        if bc == 1:
            one = nl.ndarray((3, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(dst=one, value=1.0)
            return [one]
        sels = []
        for r in range(3):
            sr = nl.ndarray((3, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=sr, data=pf, op0=nl.equal, operand0=float(r), engine=nisa.vector_engine)
            sels.append(sr)
        return sels

    def _bcast(sels, s3, j0, f32):
        """PSUM [128, 512] fp32 holding the scale row j0 of s3 (an SBUF tile [3, n, 512] of three bf16 parts on
        partitions 0-2) on every partition: their sum by one ones matmul, or hi, mid and lo each selected by
        its own matmul and accumulated in that order (exact)."""
        ps = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
        for r in range(len(sels)):
            nisa.nc_matmul(dst=ps, stationary=sels[r], moving=s3[:, j0, :], accumulate=(r > 0))
        return ps

    def _x_rows(S, io, sb):
        """The x rows of 128-lane sub-block sb of a pass as an SBUF [128, H] bf16 tile: io["xe"] rows io["r0"] +
        sb 128 .. (xkind "rows": the core probe), or (xkind "gather") gathered from io["x"] by the lanes' token ids
        io["ts"][:, sb] into io["xr"][sb], zeroed before (empty lanes hold the out-of-range id C and are skipped, so
        the row keeps finite values: the transposes sum over lanes, so one NaN lane would reach every lane)."""
        H = S["H"]
        if io["xkind"] == "rows":
            xr = nl.ndarray((128, H), dtype=nl.bfloat16, buffer=nl.sbuf)
            r0 = io["r0"] + sb * 128
            nisa.dma_copy(dst=xr, src=io["xe"][r0:r0 + 128, :])
            return xr
        xr = io["xr"][sb]
        nisa.dma_copy(dst=xr, src=io["x"].ap(pattern=[[H, 128], [1, H]], offset=0,
                                              vector_offset=io["ts"].ap(pattern=[[S["NS"], 128], [1, 1]], offset=sb),
                                              indirect_dim=0), oob_mode=oob_mode.skip, dge_mode=S.get("DG", DGU))
        return xr

    def _wsrc(S, io, name, m_off, pattern):
        """Expert tensor S[name]'s source view for a weight DMA: expert io["e"] (a static int) at element offset m_off,
        or the SBUF scalar io["e_sb"] as a dynamic offset (a device-loop pass)."""
        t = S[name]
        if io["ekind"] == "static":
            H, M, CT = S["H"], S["M"], S["CT"]
            if S["tsc"]:  # tile scales in the sgu / sdn slots: tsg [El, M, 2, CT], tsd [El, M, CT]
                per = {"gu": 128 * M * 2 * CT * 128, "sgu": M * 2 * CT, "dn": 128 * M * H, "sdn": M * CT}[name]
            else:
                per = {"gu": 128 * M * 2 * CT * 128, "sgu": M * 2 * (CT // 4) * 3 * 512, "dn": 128 * M * H,
                       "sdn": (H // 512) * M * 3 * 512}[name]  # elements per expert (pack()'s layout)
            return t.ap(pattern=pattern, offset=io["e"] * per + m_off)
        return t.ap(pattern=pattern, offset=m_off, scalar_offset=io["e_sb"], indirect_dim=0)

    def _dq_tile(dst, src, sc, eng: int):
        """One [128, 128] fp8 tile times its scale sc (fp32 [128, 1], the same on every partition) into bf16, rounded
        once (models/quant.dequant): on the vector engine (eng 0) or the scalar engine (eng 1), which together
        dequantize a tile in about 0.6 of the time one of them takes (tools/probe_ep_prims.py --dq-tile)."""
        if eng == 0:
            nisa.tensor_scalar(dst=dst, data=src, op0=nl.multiply, operand0=sc, engine=nisa.vector_engine)
        else:
            nisa.activation(dst=dst, op=nl.copy, data=src, scale=sc)

    def _lnc_setup(spl: int, M: int, H: int, lws):
        """The LNC layout of a kernel call: dict(npg, pid, sp, wr, rcv). At LNC=2 (trn2) every kernel of this module is
        launched with grid 2 and traced once per program, the two physical cores of the logical core ("kernel is
        traced LNC times with different program_id_value", nki/_backends/mlir_tracer, so npg / pid are Python ints).
        Grid 1 is not an option there: neuronx-cc compiles a grid-1 kernel's device loops on core 0 only and refuses
        the graph ("[NCC_IXGM002] Expected function sg0001 in subgraph 1 to have 9 basic blocks, but on core 1 it has
        1 basic blocks", measured on the EP prefill kernel, 2026-10-05). sp (spl 1 and two programs): the split, each
        program half of the I-chunks and half of the output columns (_pass), with the receive buffers of the a^T swap
        allocated here, before anything else, in the same order in both traces, so they sit at the same SBUF address
        on both cores (the peer's sendrecv writes there). Two programs without the split: both run the whole kernel
        (the same device loops on both cores, as NCC_IXGM002 requires) and only program 0 writes out (wr), since every
        write is a read-modify-write add that a second program would repeat. One program (trn1): exactly the kernel
        as before."""
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        sp = spl == 1 and npg == 2  # split() checks that M and H // 512 are even
        rcv = {}  # str keys: the NKI tracer takes no other dict keys ("'in' expected ... (str, dict)")
        if sp:
            for lw in lws:
                if str(lw) not in rcv:
                    rcv[str(lw)] = nl.ndarray((128, M // 2, lw), dtype=nl.bfloat16, buffer=nl.sbuf)
        return dict(npg=npg, pid=pid, sp=sp, wr=sp or pid == 0, rcv=rcv)

    def _rcv(rcv, lw):
        """The swap buffer of passes of lw lanes, or None (unsplit)."""
        return rcv[str(lw)] if str(lw) in rcv else None

    def _zero_out(out, C, H, L):
        """out = 0 by the programs that write it (L: _lnc_setup), each its own columns when split."""
        h_lo, hw = (L["pid"] * (H // 2), H // 2) if L["sp"] else (0, H)
        zt = nl.ndarray((128, hw), dtype=out.dtype, buffer=nl.sbuf)
        nisa.memset(dst=zt, value=0.0)
        if L["wr"]:
            for t in range(C // 128):
                nisa.dma_copy(dst=out[t * 128:(t + 1) * 128, h_lo:h_lo + hw], src=zt)

    def _lnc_end(out, L):
        """With two programs, both wait until out is whole before either kernel ends: whatever runs next on either
        physical core reads it (nisa.core_barrier on a shared HBM tensor, nki/isa/_lnc.py)."""
        if L["npg"] == 2:
            nisa.core_barrier(data=out, cores=(0, 1))

    def _pass(S, io):
        """One pass of a local expert over its LW lanes: x rows from _x_rows; outputs y bf16 [128, H] per 128-lane
        sub-block either stored to io["y"] rows io["r0"] + sb 128 (ykind "rows", the core probe) or (ykind "rmw")
        scaled by each lane's routing weight io["w"][:, sb] in the PSUM drain and added into io["out"] rows of the
        lanes' tokens by a scatter read-modify-write DMA (dma_compute; empty lanes skipped). S: constants."""
        H, M, CT, LW, NS = S["H"], S["M"], S["CT"], S["LW"], S["NS"]
        gu, sgu, dn, sdn = S["gu"], S["sgu"], S["dn"], S["sdn"]
        one, act, lim = S["sels"], S["act"], S["lim"]
        f32, bf16, fp8, u8 = nl.float32, nl.bfloat16, nl.float8_e4m3, nl.uint8
        Q = H // 512
        # LNC split (S["npg"] 2: trn2 at LNC=2 with KILN_LNC_SPLIT naming moe_ep): program pid computes gate_up for its
        # half of the I-chunks, the two halves of a^T are swapped between the physical cores (nisa.sendrecv into
        # S["rcv"], allocated before anything else so it sits at the same SBUF address in both programs' traces),
        # and each program computes the down projection for its half of the output columns from the whole a^T,
        # accumulated over all I-chunks in one PSUM tile in the unsplit order: every output value is the unsplit
        # kernel's, bit for bit. Each program read-modify-writes only its own columns of out.
        sp, pid, wr = S.get("sp", False), S.get("pid", 0), S.get("wr", True)
        Mh, Qh = M // 2, Q // 2
        m_lo, m_hi = (pid * Mh, (pid + 1) * Mh) if sp else (0, M)
        q_lo, q_hi = (pid * Qh, (pid + 1) * Qh) if sp else (0, Q)
        xT = nl.ndarray((128, CT, LW), dtype=bf16, buffer=nl.sbuf)  # x^T [h, c, lane]
        for sb in range(NS):
            xr = _x_rows(S, io, sb)
            for c4 in range(CT // 4):
                px = nl.ndarray((128, 4, 128), dtype=f32 if nisa.get_nc_version() == nisa.nc_version.gen2
                                else bf16, buffer=nl.psum)
                for j in range(4):
                    c = c4 * 4 + j
                    nisa.nc_transpose(dst=px[:, j, :], data=xr[:, c * 128:(c + 1) * 128], engine=nisa.tensor_engine)
                nisa.activation(dst=xT[:, c4 * 4:(c4 + 1) * 4, sb * 128:(sb + 1) * 128], op=nl.copy, data=px)
        aT = nl.ndarray((128, M, LW), dtype=bf16, buffer=nl.sbuf)  # a^T [i, m, lane]
        for m in range(m_lo, m_hi):
            wq = nl.ndarray((128, 2, CT * 128), dtype=u8, buffer=nl.sbuf)
            nisa.dma_copy(dst=wq, src=_wsrc(S, io, "gu", m * 2 * CT * 128, [[M * 2 * CT * 128, 128], [1, 2 * CT * 128]]))
            if S["tsc"]:  # chunk m's 2 CT tile scales on every partition
                s3 = nl.ndarray((128, 2 * CT), dtype=f32, buffer=nl.sbuf)
                nisa.dma_copy(dst=s3, src=_wsrc(S, io, "sgu", m * 2 * CT, [[0, 128], [1, 2 * CT]]))
            else:
                s3 = nl.ndarray((3, 2 * (CT // 4), 512), dtype=bf16, buffer=nl.sbuf)
                nisa.dma_copy(dst=s3, src=_wsrc(S, io, "sgu", m * 2 * (CT // 4) * 3 * 512,
                                                [[512, 3], [3 * 512, 2 * (CT // 4)], [1, 512]]))
            wd = nl.ndarray((128, 2, CT * 128), dtype=bf16, buffer=nl.sbuf)
            if S["tsc"]:  # one scale per [128, 128] tile: alternately the vector and the scalar engine
                for g in range(2):
                    for c in range(CT):
                        _dq_tile(wd[:, g, c * 128:(c + 1) * 128], wq[:, g, c * 128:(c + 1) * 128].view(fp8),
                                 s3[:, g * CT + c:g * CT + c + 1], c % 2)
            else:
                for g in range(2):
                    for cq in range(CT // 4):
                        ps = _bcast(one, s3, g * (CT // 4) + cq, f32)
                        nisa.tensor_tensor(dst=wd[:, g, cq * 512:(cq + 1) * 512],
                                           data1=wq[:, g, cq * 512:(cq + 1) * 512].view(fp8), data2=ps,
                                           op=nl.multiply, engine=nisa.vector_engine)
            pg = nl.ndarray((128, LW), dtype=f32, buffer=nl.psum)
            pu = nl.ndarray((128, LW), dtype=f32, buffer=nl.psum)
            for c in range(CT):
                nisa.nc_matmul(dst=pg, stationary=wd[:, 0, c * 128:(c + 1) * 128], moving=xT[:, c, :], accumulate=(c > 0))
            for c in range(CT):
                nisa.nc_matmul(dst=pu, stationary=wd[:, 1, c * 128:(c + 1) * 128], moving=xT[:, c, :], accumulate=(c > 0))
            # g, u rounded to bf16 (a clamp commutes with the rounding: bf16(min(v, lim)) = min(bf16(v), lim)
            # for a bf16 limit), a = silu(g) u in fp32, rounded once into a^T.
            gc = nl.ndarray((128, LW), dtype=bf16, buffer=nl.sbuf)
            uc = nl.ndarray((128, LW), dtype=bf16, buffer=nl.sbuf)
            if act == 1:
                nisa.tensor_scalar(dst=gc, data=pg, op0=nl.minimum, operand0=lim, engine=nisa.vector_engine)
                nisa.tensor_scalar(dst=uc, data=pu, op0=nl.minimum, operand0=lim, op1=nl.maximum, operand1=-lim,
                                   engine=nisa.vector_engine)
            else:
                nisa.tensor_copy(dst=gc, src=pg, engine=nisa.vector_engine)
                nisa.tensor_copy(dst=uc, src=pu, engine=nisa.vector_engine)
            sl = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=sl, op=nl.silu, data=gc)
            nisa.tensor_tensor(dst=aT[:, m, :], data1=sl, data2=uc, op=nl.multiply, engine=nisa.vector_engine)
        # (no inner function for "which half holds chunk m": the NKI tracer refuses a direct call of one inside a
        # device-loop body, "inner functions can only be used as fori_loop/while_loop body arguments")
        rcv = S["rcv"] if sp else aT
        if sp:  # the other program's half of a^T, I-chunks (1 - pid) Mh .. into rcv
            nisa.sendrecv(src=aT[:, m_lo:m_hi, :], dst=rcv, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0)
        o_lo = (1 - pid) * Mh if sp else 0  # the first I-chunk rcv holds

        ys = []
        for sb in range(NS):
            ys.append(nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf))
        for q in range(q_lo, q_hi):
            dq8 = nl.ndarray((128, M, 512), dtype=u8, buffer=nl.sbuf)
            nisa.dma_copy(dst=dq8, src=_wsrc(S, io, "dn", q * 512, [[M * H, 128], [H, M], [1, 512]]))
            dd = nl.ndarray((128, M, 512), dtype=bf16, buffer=nl.sbuf)
            if S["tsc"]:  # the scales of (m, output block 4 q + j) on every partition, one per [128, 128] tile
                s3 = nl.ndarray((128, M, 4), dtype=f32, buffer=nl.sbuf)
                nisa.dma_copy(dst=s3, src=_wsrc(S, io, "sdn", q * 4, [[0, 128], [CT, M], [1, 4]]))
                for m in range(M):
                    for j in range(4):
                        _dq_tile(dd[:, m, j * 128:(j + 1) * 128], dq8[:, m, j * 128:(j + 1) * 128].view(fp8),
                                 s3[:, m, j:j + 1], (m * 4 + j) % 2)
            else:
                s3 = nl.ndarray((3, M, 512), dtype=bf16, buffer=nl.sbuf)
                nisa.dma_copy(dst=s3, src=_wsrc(S, io, "sdn", q * M * 3 * 512, [[512, 3], [3 * 512, M], [1, 512]]))
                for m in range(M):
                    ps = _bcast(one, s3, m, f32)
                    nisa.tensor_tensor(dst=dd[:, m, :], data1=dq8[:, m, :].view(fp8), data2=ps, op=nl.multiply,
                                       engine=nisa.vector_engine)
            for sb in range(NS):
                py = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
                for m in range(M):
                    a_src, a_m = (aT, m) if (not sp or m_lo <= m < m_hi) else (rcv, m - o_lo)
                    nisa.nc_matmul(dst=py, stationary=a_src[:, a_m, sb * 128:(sb + 1) * 128], moving=dd[:, m, :],
                                   accumulate=(m > 0))
                if io["ykind"] == "rmw":  # y w, rounded once
                    nisa.activation(dst=ys[sb][:, q * 512:(q + 1) * 512], op=nl.copy, data=py, scale=io["w"][:, sb:sb + 1])
                else:
                    nisa.activation(dst=ys[sb][:, q * 512:(q + 1) * 512], op=nl.copy, data=py)
        h_lo, hw = (pid * (H // 2), H // 2) if sp else (0, H)  # this program's output columns
        for sb in range(NS if wr else 0):  # unsplit with two programs: program 0 alone writes
            if io["ykind"] == "rows":
                r0 = io["r0"] + sb * 128
                nisa.dma_copy(dst=io["y"][r0:r0 + 128, h_lo:h_lo + hw], src=ys[sb][:, h_lo:h_lo + hw])
            else:
                dst = io["out"].ap(pattern=[[H, 128], [1, hw]], offset=h_lo,
                                   vector_offset=io["ts"].ap(pattern=[[NS, 128], [1, 1]], offset=sb), indirect_dim=0)
                nisa.dma_compute(dst=dst, srcs=[dst, ys[sb][:, h_lo:h_lo + hw]], reduce_op=nl.add,
                                 oob_mode=oob_mode.skip)

    @nki.jit
    def kiln_moe_ep_core(xe, ex, gu, sgu, dn, sdn, LW: int, act: int, lim: float, bc: int, rev: int, tsc: int = 0,
                         spl: int = 0):
        """The passes alone (a probe of the per-pass cost): xe bf16 [NP LW, H] the lanes' rows of NP passes in
        order, ex int32 [1, NP] each pass's local expert; returns y bf16 [NP LW, H]."""
        R, H = xe.shape
        NP = ex.shape[1]
        M = gu.shape[2]
        CT = H // 128
        L = _lnc_setup(spl, M, H, [LW])
        S = dict(H=H, M=M, CT=CT, LW=LW, NS=LW // 128, gu=gu, sgu=sgu, dn=dn, sdn=sdn, sels=_consts(bc), act=act,
                 lim=lim, tsc=tsc, sp=L["sp"], pid=L["pid"], wr=L["wr"], rcv=_rcv(L["rcv"], LW))
        exs = nl.ndarray((1, NP), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=exs, src=ex)
        y = nl.ndarray((R, H), dtype=xe.dtype, buffer=nl.shared_hbm)
        for p in range(NP):
            _pass(S, dict(xkind="rows", ykind="rows", ekind="dyn", e_sb=exs.ap(pattern=[[NP, 1], [1, 1]], offset=p),
                          xe=xe, y=y, r0=p * LW))
        _lnc_end(y, L)
        return y

    def _lane_tokens(S, P, ngi, nt0, sb, ts):
        """ts[:, sb] (int32 [128, NS]) = the token of each lane of sub-block sb: lane j of a pass of expert e starting
        at rank j0 holds the (j0 + j + 1)-th token with a pair on e, which is #{t : incl_e(t) <= j0 + j} for the
        inclusive token-order prefix count incl_e (non-decreasing), and C past the expert's last pair (an empty
        lane). Counted on the scalar and tensor engines: per token tile T, sign(j0 + j + 0.5 - incl_e(t)) as an
        activation over P["jrow"][sb] (j = sb 128 + 0 .. 127 on every partition) with the per-token bias column
        ngi[:, nt0 + T] = j0 + 0.5 - incl_e(t) (+-1 in bf16: exact), summed over the tokens by a ones matmul into
        PSUM [128 lanes, 1]; count = (C + sum) / 2, exact in fp32."""
        NT, C = P["NT"], P["C"]
        cn = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
        for T in range(NT):
            sg = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=sg, op=nl.sign, data=P["jrow"][sb], bias=ngi[:, nt0 + T:nt0 + T + 1], scale=1.0)
            nisa.nc_matmul(dst=cn, stationary=sg, moving=P["ones1"], accumulate=(T > 0))
        tf = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=tf, data=cn, op0=nl.add, operand0=float(C), op1=nl.multiply, operand1=0.5,
                           engine=nisa.vector_engine)
        nisa.tensor_copy(dst=ts[:, sb:sb + 1], src=tf, engine=nisa.vector_engine)

    def _lane_weights(S, P, ts, e_cmp, w):
        """w[:, sb] (fp32 [128, NS]) = each lane's routing weight: the token's (gathered) loc row compared with the
        pass's local expert e_cmp (a float, or a per-partition [128, 1] tile), times its wts row, summed over k
        (a token has at most one pair per expert). Empty lanes (token C) are skipped: their weight is garbage,
        and their output is never stored."""
        K, NS = P["K"], S["NS"]
        for sb in range(NS):
            lr = nl.ndarray((128, K), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=lr, src=P["loc_h"].ap(pattern=[[K, 128], [1, K]], offset=0,
                                                    vector_offset=ts.ap(pattern=[[NS, 128], [1, 1]], offset=sb),
                                                    indirect_dim=0), oob_mode=oob_mode.skip, dge_mode=S.get("DG", DGU))
            wr = nl.ndarray((128, K), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=wr, src=P["wts"].ap(pattern=[[K, 128], [1, K]], offset=0,
                                                  vector_offset=ts.ap(pattern=[[NS, 128], [1, 1]], offset=sb),
                                                  indirect_dim=0), oob_mode=oob_mode.skip, dge_mode=S.get("DG", DGU))
            mk = nl.ndarray((128, K), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mk, data=lr, op0=nl.equal, operand0=e_cmp, engine=nisa.vector_engine)
            pr = nl.ndarray((128, K), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pr, data1=mk, data2=wr, op=nl.multiply, engine=nisa.vector_engine)
            nisa.tensor_reduce(dst=w[:, sb:sb + 1], op=nl.add, data=pr, axis=1)

    @nki.jit
    def kiln_moe_ep_kernel(x, topi, wts, lmap, gu, sgu, dn, sdn, LW: int, LW2: int, PMAX: int, act: int, lim: float,
                           bc: int, rev: int, tsc: int = 0, spl: int = 0, dge: int = 0):
        """x bf16 [C, H] (C a multiple of 128); topi int32 [C, K] each token's experts and wts bf16 [C, K] their
        routing weights; lmap int32 [1, E + 1]: each expert's index among this rank's El local experts, El for
        another rank's and for the padding expert E (local_map; data, so every rank traces the same graph); gu,
        sgu, dn, sdn: the local experts (pack()); LW lanes per first pass (128 or 256), LW2 per big overflow pass (128
        .. 512): an expert's pairs past its first LW take passes of LW2 lanes and a last one of LW lanes when what is
        left fits it; PMAX = max_passes(C, K, El, LW2), the big overflow table's length.
        Returns bf16 [C, H]: per token the sum over its pairs whose expert is local of bf16(w expert(x)), added in
        the order of the local experts (bf16 read-modify-writes), 0 for a token with none.

        0. The plan, exact integers in fp32: each pair's local expert (lmap gathered per partition), membership
           mem_e(t) per token (a token has at most one pair per expert), its exclusive token-order prefix rank_e(t)
           (a strictly-lower-triangular matmul within each token tile plus the earlier tiles' totals), counts n_e,
           passes ceil(n_e / LW) and the overflow passes past each expert's first (their table: expert and first
           rank).
        1. out zeroed.
        2. Each local expert's first pass, static (all El of them run, also an expert with no pair here: its lanes
           are all empty), then a device loop over the overflow passes (trip count from the plan, so any routing
           runs exactly). A pass: its lanes' tokens (_lane_tokens), their routing weights (_lane_weights), the x
           rows gathered, _pass, its outputs times the weights added into out at the lanes' tokens."""
        C, H = x.shape
        K = topi.shape[1]
        E = lmap.shape[1]
        El = gu.shape[0]
        M = gu.shape[2]
        CT = H // 128
        NS = LW // 128
        NT = C // 128
        f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
        NS2 = LW2 // 128
        LB2 = 7  # LW2 = 2 ** LB2 (128, 256 or 512)
        while (1 << LB2) < LW2:
            LB2 += 1
        L = _lnc_setup(spl, M, H, [LW, LW2])  # first: the swap buffers at the same address on both cores
        sels = _consts(bc)
        # (keys spelled out: the NKI tracer has no ** expansion)
        # dge 1 (KILN_MOE_EP_DGE=sw, experiment): the kernel's gathers on software DGE (dge_mode swdge)
        DG = nisa.dge_mode.swdge if dge else DGU
        S = dict(H=H, M=M, CT=CT, LW=LW, NS=NS, gu=gu, sgu=sgu, dn=dn, sdn=sdn, sels=sels, act=act, lim=lim, tsc=tsc,
                 rcv=_rcv(L["rcv"], LW), sp=L["sp"], pid=L["pid"], wr=L["wr"], DG=DG)
        S2 = dict(H=H, M=M, CT=CT, LW=LW2, NS=NS2, gu=gu, sgu=sgu, dn=dn, sdn=sdn, sels=sels, act=act, lim=lim,
                  tsc=tsc, rcv=_rcv(L["rcv"], LW2), sp=L["sp"], pid=L["pid"], wr=L["wr"], DG=DG)

        # 0. The plan.
        tk = nl.ndarray((128, NT, K), dtype=i32, buffer=nl.sbuf)  # topi[T 128 + p, k]
        nisa.dma_copy(dst=tk, src=topi.ap(pattern=[[K, 128], [128 * K, NT], [1, K]], offset=0))
        tku = nl.ndarray((128, NT, K), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tku, src=tk, engine=nisa.vector_engine)
        lmi = nl.ndarray((128, E), dtype=i32, buffer=nl.sbuf)  # lmap on every partition
        nisa.dma_copy(dst=lmi, src=lmap.ap(pattern=[[0, 128], [1, E]], offset=0))
        lmf = nl.ndarray((128, E), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lmf, src=lmi, engine=nisa.vector_engine)
        loc = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)  # local expert of each pair, El if not here
        nisa.nc_n_gather(dst=loc, data=lmf, indices=tku)
        loc_h = nl.ndarray((C, K), dtype=f32, buffer=nl.private_hbm)  # for the lanes' weights
        nisa.dma_copy(dst=loc_h.ap(pattern=[[K, 128], [128 * K, NT], [1, K]], offset=0), src=loc)
        EN = El * NT
        mem = nl.ndarray((128, EN), dtype=bf16, buffer=nl.sbuf)  # mem[p, e NT + T] = [token T 128 + p has e]
        for e in range(El):
            eq = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=eq, data=loc, op0=nl.equal, operand0=float(e), engine=nisa.vector_engine)
            nisa.tensor_reduce(dst=mem[:, e * NT:(e + 1) * NT], op=nl.add, data=eq, axis=2)
        uu = nl.ndarray((128, 128), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=uu, pattern=[[1, 128]], offset=0, channel_multiplier=-1)  # q - p
        uuf = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=uuf, src=uu, engine=nisa.vector_engine)
        lt = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)  # [p < q]
        nisa.tensor_scalar(dst=lt, data=uuf, op0=nl.greater, operand0=0.0, engine=nisa.vector_engine)
        on128 = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=on128, value=1.0)
        wit = nl.ndarray((128, EN), dtype=f32, buffer=nl.sbuf)  # earlier tokens of the same tile with e
        tot = nl.ndarray((128, EN), dtype=f32, buffer=nl.sbuf)  # the tile's tokens with e, on every partition
        for c0 in range(0, EN, 512):
            cw = min(512, EN - c0)
            pw = nl.ndarray((128, cw), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pw, stationary=lt, moving=mem[:, c0:c0 + cw], accumulate=False)
            nisa.tensor_copy(dst=wit[:, c0:c0 + cw], src=pw, engine=nisa.vector_engine)
            pt = nl.ndarray((128, cw), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pt, stationary=on128, moving=mem[:, c0:c0 + cw], accumulate=False)
            nisa.tensor_copy(dst=tot[:, c0:c0 + cw], src=pt, engine=nisa.vector_engine)
        inc = nl.ndarray((128, El, NT), dtype=f32, buffer=nl.sbuf)  # inclusive prefix over the tiles, per expert
        nisa.tensor_copy(dst=inc, src=tot.ap(pattern=[[EN, 128], [NT, El], [1, NT]], offset=0), engine=nisa.vector_engine)
        sh = 1
        while sh < NT:
            prev = nl.ndarray((128, El, NT), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=prev, src=inc, engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=inc[:, :, sh:NT], data1=prev[:, :, sh:NT], data2=prev[:, :, 0:NT - sh], op=nl.add,
                               engine=nisa.vector_engine)
            sh *= 2
        # ngi = 0.5 - incl_e(t), incl = rank + mem, rank = wit + (inc - tot): the lane-token bias (_lane_tokens).
        ngi = nl.ndarray((128, EN), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ngi, data1=inc.ap(pattern=[[EN, 128], [1, EN]], offset=0), data2=tot, op=nl.subtract,
                           engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=ngi, data1=ngi, data2=wit, op=nl.add, engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=ngi, data1=ngi, data2=mem, op=nl.add, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=ngi, data=ngi, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=0.5,
                           engine=nisa.vector_engine)
        ngi_h = nl.ndarray((El, 128, NT), dtype=f32, buffer=nl.private_hbm)  # for the loop's dynamic expert
        nisa.dma_copy(dst=ngi_h.ap(pattern=[[NT, 128], [128 * NT, El], [1, NT]], offset=0), src=ngi)
        # Overflow passes per expert (partition 0), n_e = inc[:, e, NT - 1], past its first LW ranks r = max(n_e - LW,
        # 0): nb = floor(r / LW2) + [r mod LW2 > LW] passes of LW2 lanes, then ns = [what is left > 0] one of LW lanes.
        cnt = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cnt, src=inc[0:1, :, NT - 1], engine=nisa.vector_engine)
        rf = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rf, data=cnt, op0=nl.add, operand0=float(-LW), op1=nl.maximum, operand1=0.0,
                           engine=nisa.vector_engine)
        ri = nl.ndarray((1, El), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ri, src=rf, engine=nisa.vector_engine)
        qi = nl.ndarray((1, El), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=qi, data=ri, op0=nl.right_shift, operand0=LB2)
        qf = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=qf, src=qi, engine=nisa.vector_engine)
        rem = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)  # r - q LW2
        nisa.scalar_tensor_tensor(dst=rem, data=qf, op0=nl.multiply, operand0=float(-LW2), op1=nl.add, operand1=rf)
        big = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)  # [rem > LW]
        nisa.tensor_scalar(dst=big, data=rem, op0=nl.greater, operand0=float(LW), engine=nisa.vector_engine)
        nb = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=nb, data1=qf, data2=big, op=nl.add, engine=nisa.vector_engine)
        nsm = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)  # [rem > 0] [rem <= LW]
        nisa.tensor_scalar(dst=nsm, data=rem, op0=nl.greater, operand0=0.0, engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=nsm, data1=nsm, data2=big, op=nl.subtract, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=nsm, data=nsm, op0=nl.maximum, operand0=0.0, engine=nisa.vector_engine)
        b0 = nl.ndarray((1, El + 1), dtype=f32, buffer=nl.sbuf)  # first rank of each expert's big passes: LW
        nisa.memset(dst=b0, value=float(LW))
        b1 = nl.ndarray((1, El + 1), dtype=f32, buffer=nl.sbuf)  # ... of its small pass: LW + nb LW2
        nisa.memset(dst=b1, value=float(LW))
        nisa.scalar_tensor_tensor(dst=b1[:, 0:El], data=nb, op0=nl.multiply, operand0=float(LW2), op1=nl.add,
                                  operand1=b0[:, 0:El])
        tb = _ovtable(nb, b0, LW2, El, PMAX)
        ts_ = _ovtable(nsm, b1, LW, El, El)
        # Constants of the lane-token count.
        jrow = []
        for sb in range(max(NS, NS2)):
            ji = nl.ndarray((128, 128), dtype=i32, buffer=nl.sbuf)
            nisa.iota(dst=ji, pattern=[[1, 128]], offset=sb * 128, channel_multiplier=0)
            jr = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=jr, src=ji, engine=nisa.vector_engine)
            jrow.append(jr)
        ones1 = nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=ones1, value=1.0)
        Pd = dict(NT=NT, C=C, K=K, jrow=jrow, ones1=ones1, loc_h=loc_h, wts=wts)

        # 1. out = 0.
        out = nl.ndarray((C, H), dtype=x.dtype, buffer=nl.shared_hbm)
        _zero_out(out, C, H, L)

        # 2. Each local expert's first pass (static), then the overflow passes in a device loop.
        for e in range(El):
            ts = nl.ndarray((128, NS), dtype=i32, buffer=nl.sbuf)
            for sb in range(NS):
                _lane_tokens(S, Pd, ngi, e * NT, sb, ts)
            w = nl.ndarray((128, NS), dtype=f32, buffer=nl.sbuf)
            _lane_weights(S, Pd, ts, float(e), w)
            xrj = []
            for sb in range(NS):
                xb = nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf)
                nisa.memset(dst=xb, value=0.0, engine=nisa.gpsimd_engine)
                xrj.append(xb)
            _pass(S, dict(xkind="gather", ykind="rmw", ekind="static", e=e, x=x, ts=ts, xr=xrj, w=w, out=out))
        xrs = []  # the loop's x rows, zeroed once: a skipped (empty) lane keeps finite values
        for sb in range(NS2):
            xb = nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf)
            nisa.memset(dst=xb, value=0.0)
            xrs.append(xb)
        xrs1 = []  # the small loop's
        for sb in range(NS):
            xb = nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf)
            nisa.memset(dst=xb, value=0.0)
            xrs1.append(xb)
        rp = nisa.register_alloc()
        nisa.register_load(rp, tb["n"])
        T_ = dict(etab=tb["e"], eftab=tb["ef"], jtab=tb["j0"], ngi_h=ngi_h, x=x, xrs=xrs, out=out)

        def body(it):
            _ep_overflow_pass(S2, Pd, T_, it)

        nl.fori_loop(0, rp, body)
        rq = nisa.register_alloc()
        nisa.register_load(rq, ts_["n"])
        T1 = dict(etab=ts_["e"], eftab=ts_["ef"], jtab=ts_["j0"], ngi_h=ngi_h, x=x, xrs=xrs1, out=out)

        def body1(it):
            _ep_overflow_pass(S, Pd, T1, it)

        nl.fori_loop(0, rq, body1)
        _lnc_end(out, L)
        return out

    def _ovtable(ov, b, Lp, El, PMAX):
        """The table of a run of overflow passes: ov [1, El] passes per local expert (fp32 integers), b [1, El + 1] the
        first rank of each expert's first pass of this run (anything in column El), Lp lanes per pass, PMAX entries:
        pass o's expert e_o = #{e : incl_e <= o} (El past the end) and first rank b[e_o] + (o - excl[e_o]) Lp, in
        HBM (int32 and fp32 expert, fp32 rank), and the pass count (an int32 SBUF scalar)."""
        f32, i32 = nl.float32, nl.int32
        oi = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)  # inclusive prefix
        nisa.tensor_copy(dst=oi, src=ov, engine=nisa.vector_engine)
        sh = 1
        while sh < El:
            prev = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=prev, src=oi, engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=oi[:, sh:El], data1=prev[:, sh:El], data2=prev[:, 0:El - sh], op=nl.add,
                               engine=nisa.vector_engine)
            sh *= 2
        ox = nl.ndarray((1, El + 1), dtype=f32, buffer=nl.sbuf)  # exclusive prefix, 0 past the end
        nisa.memset(dst=ox, value=0.0)
        nisa.tensor_tensor(dst=ox[:, 0:El], data1=oi, data2=ov, op=nl.subtract, engine=nisa.vector_engine)
        jj = nl.ndarray((1, PMAX), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=jj, pattern=[[1, PMAX]], offset=0, channel_multiplier=0)
        jf = nl.ndarray((1, PMAX), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=jf, src=jj, engine=nisa.vector_engine)
        ejf = nl.ndarray((1, PMAX), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ejf, value=0.0)
        for e in range(El):
            ge = nl.ndarray((1, PMAX), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ge, data=jf, op0=nl.greater_equal, operand0=oi[:, e:e + 1], engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=ejf, data1=ejf, data2=ge, op=nl.add, engine=nisa.vector_engine)
        eju = nl.ndarray((1, PMAX), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eju, src=ejf, engine=nisa.vector_engine)
        oxg = nl.ndarray((1, PMAX), dtype=f32, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=oxg, data=ox, indices=eju)
        bg = nl.ndarray((1, PMAX), dtype=f32, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=bg, data=b, indices=eju)
        j0f = nl.ndarray((1, PMAX), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=j0f, data1=jf, data2=oxg, op=nl.subtract, engine=nisa.vector_engine)
        nisa.scalar_tensor_tensor(dst=j0f, data=j0f, op0=nl.multiply, operand0=float(Lp), op1=nl.add, operand1=bg)
        eji = nl.ndarray((1, PMAX), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eji, src=ejf, engine=nisa.vector_engine)
        etab = nl.ndarray((PMAX, 1), dtype=i32, buffer=nl.private_hbm)
        nisa.dma_copy(dst=etab.ap(pattern=[[PMAX, 1], [1, PMAX]], offset=0), src=eji)
        eftab = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.private_hbm)
        nisa.dma_copy(dst=eftab.ap(pattern=[[PMAX, 1], [1, PMAX]], offset=0), src=ejf)
        jtab = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.private_hbm)
        nisa.dma_copy(dst=jtab.ap(pattern=[[PMAX, 1], [1, PMAX]], offset=0), src=j0f)
        nov = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=nov, src=oi[:, El - 1:El], engine=nisa.vector_engine)
        return dict(e=etab, ef=eftab, j0=jtab, n=nov)

    def _ep_overflow_pass(S, Pd, T_, it):
        """Overflow pass `it` (a device-loop register): its expert and first rank from the table, the expert's
        lane-token bias row (0.5 - incl_e) shifted by the first rank, then as a static pass."""
        NT, NS = Pd["NT"], S["NS"]
        e_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=e_sb, src=T_["etab"].ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=it, indirect_dim=0))
        ef = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)  # on every partition
        nisa.dma_copy(dst=ef, src=T_["eftab"].ap(pattern=[[0, 128], [1, 1]], offset=0, scalar_offset=it, indirect_dim=0))
        j0 = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=j0, src=T_["jtab"].ap(pattern=[[0, 128], [1, 1]], offset=0, scalar_offset=it, indirect_dim=0))
        ng = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ng, src=T_["ngi_h"].ap(pattern=[[NT, 128], [1, NT]], offset=0, scalar_offset=e_sb,
                                                 indirect_dim=0))
        nisa.tensor_scalar(dst=ng, data=ng, op0=nl.add, operand0=j0, engine=nisa.vector_engine)
        ts = nl.ndarray((128, NS), dtype=nl.int32, buffer=nl.sbuf)
        for sb in range(NS):
            _lane_tokens(S, Pd, ng, 0, sb, ts)
        w = nl.ndarray((128, NS), dtype=nl.float32, buffer=nl.sbuf)
        _lane_weights(S, Pd, ts, ef, w)
        _pass(S, dict(xkind="gather", ykind="rmw", ekind="dyn", e_sb=e_sb, x=T_["x"], ts=ts, xr=T_["xrs"], w=w,
                      out=T_["out"]))

    # --- end of the dequantize-first kernel's source (REV hashes up to here) ---

    # --- the small-lane kernel (decode): per-row scales on few lanes, no dequantization ---

    def _plan_s(topi, wts, lmap, C, K, E, El, LW, PMAX):
        """kiln_moe_ep_kernel's plan for passes of LW lanes (a copy, so that the dequantize-first kernel's source,
        which its REV hashes, stays as it is), plus each local expert's weight-load index eeff [1, El] int32: e if
        it has a pair here, else El (the static pass of an unused expert skips its loads)."""
        f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
        NT = C // 128
        LB = 4  # LW = 2 ** LB (the tracer takes no dict literal)
        while (1 << LB) < LW:
            LB += 1
        tk = nl.ndarray((128, NT, K), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=tk, src=topi.ap(pattern=[[K, 128], [128 * K, NT], [1, K]], offset=0))
        tku = nl.ndarray((128, NT, K), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tku, src=tk, engine=nisa.vector_engine)
        lmi = nl.ndarray((128, E), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lmi, src=lmap.ap(pattern=[[0, 128], [1, E]], offset=0))
        lmf = nl.ndarray((128, E), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lmf, src=lmi, engine=nisa.vector_engine)
        loc = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=loc, data=lmf, indices=tku)
        loc_h = nl.ndarray((C, K), dtype=f32, buffer=nl.private_hbm)
        nisa.dma_copy(dst=loc_h.ap(pattern=[[K, 128], [128 * K, NT], [1, K]], offset=0), src=loc)
        EN = El * NT
        mem = nl.ndarray((128, EN), dtype=bf16, buffer=nl.sbuf)
        for e in range(El):
            eq = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=eq, data=loc, op0=nl.equal, operand0=float(e), engine=nisa.vector_engine)
            nisa.tensor_reduce(dst=mem[:, e * NT:(e + 1) * NT], op=nl.add, data=eq, axis=2)
        uu = nl.ndarray((128, 128), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=uu, pattern=[[1, 128]], offset=0, channel_multiplier=-1)
        uuf = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=uuf, src=uu, engine=nisa.vector_engine)
        lt = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lt, data=uuf, op0=nl.greater, operand0=0.0, engine=nisa.vector_engine)
        on128 = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=on128, value=1.0)
        wit = nl.ndarray((128, EN), dtype=f32, buffer=nl.sbuf)
        tot = nl.ndarray((128, EN), dtype=f32, buffer=nl.sbuf)
        for c0 in range(0, EN, 512):
            cw = min(512, EN - c0)
            pw = nl.ndarray((128, cw), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pw, stationary=lt, moving=mem[:, c0:c0 + cw], accumulate=False)
            nisa.tensor_copy(dst=wit[:, c0:c0 + cw], src=pw, engine=nisa.vector_engine)
            pt = nl.ndarray((128, cw), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pt, stationary=on128, moving=mem[:, c0:c0 + cw], accumulate=False)
            nisa.tensor_copy(dst=tot[:, c0:c0 + cw], src=pt, engine=nisa.vector_engine)
        inc = nl.ndarray((128, El, NT), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=inc, src=tot.ap(pattern=[[EN, 128], [NT, El], [1, NT]], offset=0), engine=nisa.vector_engine)
        sh = 1
        while sh < NT:
            prev = nl.ndarray((128, El, NT), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=prev, src=inc, engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=inc[:, :, sh:NT], data1=prev[:, :, sh:NT], data2=prev[:, :, 0:NT - sh], op=nl.add,
                               engine=nisa.vector_engine)
            sh *= 2
        ngi = nl.ndarray((128, EN), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ngi, data1=inc.ap(pattern=[[EN, 128], [1, EN]], offset=0), data2=tot, op=nl.subtract,
                           engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=ngi, data1=ngi, data2=wit, op=nl.add, engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=ngi, data1=ngi, data2=mem, op=nl.add, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=ngi, data=ngi, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=0.5,
                           engine=nisa.vector_engine)
        ngi_h = nl.ndarray((El, 128, NT), dtype=f32, buffer=nl.private_hbm)
        nisa.dma_copy(dst=ngi_h.ap(pattern=[[NT, 128], [128 * NT, El], [1, NT]], offset=0), src=ngi)
        cnt = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cnt, src=inc[0:1, :, NT - 1], engine=nisa.vector_engine)
        ci = nl.ndarray((1, El), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ci, data=cnt, op0=nl.add, operand0=float(LW - 1), engine=nisa.vector_engine)
        nbi = nl.ndarray((1, El), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nbi, data=ci, op0=nl.right_shift, operand0=LB)
        ov = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ov, src=nbi, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=ov, data=ov, op0=nl.add, operand0=-1.0, op1=nl.maximum, operand1=0.0,
                           engine=nisa.vector_engine)
        oi = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=oi, src=ov, engine=nisa.vector_engine)
        sh = 1
        while sh < El:
            prev = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=prev, src=oi, engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=oi[:, sh:El], data1=prev[:, sh:El], data2=prev[:, 0:El - sh], op=nl.add,
                               engine=nisa.vector_engine)
            sh *= 2
        ox = nl.ndarray((1, El + 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ox, value=0.0)
        nisa.tensor_tensor(dst=ox[:, 0:El], data1=oi, data2=ov, op=nl.subtract, engine=nisa.vector_engine)
        jj = nl.ndarray((1, PMAX), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=jj, pattern=[[1, PMAX]], offset=0, channel_multiplier=0)
        jf = nl.ndarray((1, PMAX), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=jf, src=jj, engine=nisa.vector_engine)
        ejf = nl.ndarray((1, PMAX), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ejf, value=0.0)
        for e in range(El):
            ge = nl.ndarray((1, PMAX), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ge, data=jf, op0=nl.greater_equal, operand0=oi[:, e:e + 1], engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=ejf, data1=ejf, data2=ge, op=nl.add, engine=nisa.vector_engine)
        eju = nl.ndarray((1, PMAX), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eju, src=ejf, engine=nisa.vector_engine)
        oxg = nl.ndarray((1, PMAX), dtype=f32, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=oxg, data=ox, indices=eju)
        j0f = nl.ndarray((1, PMAX), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=j0f, data1=jf, data2=oxg, op=nl.subtract, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=j0f, data=j0f, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=float(LW),
                           engine=nisa.vector_engine)
        eji = nl.ndarray((1, PMAX), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eji, src=ejf, engine=nisa.vector_engine)
        etab = nl.ndarray((PMAX, 1), dtype=i32, buffer=nl.private_hbm)
        nisa.dma_copy(dst=etab.ap(pattern=[[PMAX, 1], [1, PMAX]], offset=0), src=eji)
        eftab = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.private_hbm)
        nisa.dma_copy(dst=eftab.ap(pattern=[[PMAX, 1], [1, PMAX]], offset=0), src=ejf)
        jtab = nl.ndarray((PMAX, 1), dtype=f32, buffer=nl.private_hbm)
        nisa.dma_copy(dst=jtab.ap(pattern=[[PMAX, 1], [1, PMAX]], offset=0), src=j0f)
        nov = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=nov, src=oi[:, El - 1:El], engine=nisa.vector_engine)
        # eeff[e] = e if n_e > 0 else El
        ei0 = nl.ndarray((1, El), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ei0, pattern=[[1, El]], offset=0, channel_multiplier=0)
        ef0 = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ef0, src=ei0, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=ef0, data=ef0, op0=nl.subtract, operand0=float(El), engine=nisa.vector_engine)
        used = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=used, data=cnt, op0=nl.greater, operand0=0.0, engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=ef0, data1=ef0, data2=used, op=nl.multiply, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=ef0, data=ef0, op0=nl.add, operand0=float(El), engine=nisa.vector_engine)
        eeff = nl.ndarray((1, El), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eeff, src=ef0, engine=nisa.vector_engine)
        ji = nl.ndarray((128, LW), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ji, pattern=[[1, LW]], offset=0, channel_multiplier=0)
        jr = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=jr, src=ji, engine=nisa.vector_engine)
        ones1 = nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=ones1, value=1.0)
        return dict(NT=NT, C=C, K=K, loc_h=loc_h, wts=wts, ngi=ngi, ngi_h=ngi_h, etab=etab, eftab=eftab, jtab=jtab,
                    nov=nov, eeff=eeff, jrow=jr, ones1=ones1)

    def _lanes_s(P, ngi, nt0, LW, ts):
        """ts [LW, 1] int32: the lanes' tokens (as _lane_tokens, on LW partitions)."""
        NT, C = P["NT"], P["C"]
        cn = nl.ndarray((LW, 1), dtype=nl.float32, buffer=nl.psum)
        for T in range(NT):
            sg = nl.ndarray((128, LW), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.activation(dst=sg, op=nl.sign, data=P["jrow"], bias=ngi[:, nt0 + T:nt0 + T + 1], scale=1.0)
            nisa.nc_matmul(dst=cn, stationary=sg, moving=P["ones1"], accumulate=(T > 0))
        tf = nl.ndarray((LW, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=tf, data=cn, op0=nl.add, operand0=float(C), op1=nl.multiply, operand1=0.5,
                           engine=nisa.vector_engine)
        nisa.tensor_copy(dst=ts, src=tf, engine=nisa.vector_engine)

    def _weights_s(P, ts, e_cmp, LW, w):
        """w [LW, 1] fp32: the lanes' routing weights (as _lane_weights, on LW partitions)."""
        K = P["K"]
        lr = nl.ndarray((LW, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lr, src=P["loc_h"].ap(pattern=[[K, LW], [1, K]], offset=0,
                                                vector_offset=ts.ap(pattern=[[1, LW], [1, 1]], offset=0), indirect_dim=0),
                      oob_mode=oob_mode.skip)
        wr = nl.ndarray((LW, K), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wr, src=P["wts"].ap(pattern=[[K, LW], [1, K]], offset=0,
                                              vector_offset=ts.ap(pattern=[[1, LW], [1, 1]], offset=0), indirect_dim=0),
                      oob_mode=oob_mode.skip)
        mk = nl.ndarray((LW, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mk, data=lr, op0=nl.equal, operand0=e_cmp, engine=nisa.vector_engine)
        pr = nl.ndarray((LW, K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=pr, data1=mk, data2=wr, op=nl.multiply, engine=nisa.vector_engine)
        nisa.tensor_reduce(dst=w, op=nl.add, data=pr, axis=1)

    def _pass_s(S, e_dma, ts, w, xr, out, skip: bool):
        """One small-lane pass of a local expert over LW lanes (LW partitions): the lanes' x rows gathered into the
        pre-zeroed xr and transposed; per I-chunk m and gate / up, the 32 h-tile partials of the fp8 weights as
        stored (stationary [h, i]) with x^T moving, stacked in PSUM [i, c, lane], times their per-row tile scales
        (dsg, i on the partitions: a stride-0 read over the lanes), summed over c, rounded to bf16, the clamped
        SiLU product rounded to bf16 (a^T [i, m, lane]); per 128-column tile of the output the 16 I-chunk partials
        of the down weights as stored (stationary [i, h]) with a^T moving, times their scales (dsd, h on the
        partitions), summed over m, rounded to bf16, transposed back to [lane, h], times the lanes' routing
        weights in the drain (rounded once) and added into out at the lanes' tokens (dma_compute). e_dma: the
        SBUF int32 scalar of the expert's weight offset; skip (static passes only: a DMA that may skip inside a
        device loop fails the backend, NCC_IGCA103 "Cannot spill memorylocation defined by DMA skipping inside of
        loop") lets e_dma be El for an unused expert, whose weight loads are then skipped."""
        H, M, CT, LW = S["H"], S["M"], S["CT"], S["LW"]
        gu, dsg, dn, dsd, act, lim = S["gu"], S["dsg"], S["dn"], S["dsd"], S["act"], S["lim"]
        f32, bf16, fp8, u8 = nl.float32, nl.bfloat16, nl.float8_e4m3, nl.uint8
        CG = min(CT, 512 // LW)  # h-tiles per PSUM stack (2 KB per partition)
        # LNC split as in _pass: gate_up over this program's half of the I-chunks, a^T halves swapped, down for its
        # half of the 512-column output chunks from the whole a^T (the unsplit arithmetic, bit for bit).
        sp, pid, wr = S.get("sp", False), S.get("pid", 0), S.get("wr", True)
        Mh, Q = M // 2, H // 512
        m_lo, m_hi = (pid * Mh, (pid + 1) * Mh) if sp else (0, M)
        q_lo, q_hi = (pid * (Q // 2), (pid + 1) * (Q // 2)) if sp else (0, Q)
        wom = oob_mode.skip if skip else oob_mode.error
        nisa.dma_copy(dst=xr, src=S["x"].ap(pattern=[[H, LW], [1, H]], offset=0,
                                             vector_offset=ts.ap(pattern=[[1, LW], [1, 1]], offset=0), indirect_dim=0),
                      oob_mode=oob_mode.skip)
        xT = nl.ndarray((128, CT, LW), dtype=bf16, buffer=nl.sbuf)
        for c4 in range(CT // 4):
            px = nl.ndarray((128, 4, LW), dtype=f32 if nisa.get_nc_version() == nisa.nc_version.gen2 else bf16,
                            buffer=nl.psum)
            for j in range(4):
                c = c4 * 4 + j
                nisa.nc_transpose(dst=px[:, j, :], data=xr[:, c * 128:(c + 1) * 128], engine=nisa.tensor_engine)
            nisa.activation(dst=xT[:, c4 * 4:(c4 + 1) * 4, :], op=nl.copy, data=px)
        aT = nl.ndarray((128, M, LW), dtype=bf16, buffer=nl.sbuf)
        for m in range(m_lo, m_hi):
            wq = nl.ndarray((128, 2, CT * 128), dtype=u8, buffer=nl.sbuf)
            nisa.dma_copy(dst=wq, src=gu.ap(pattern=[[M * 2 * CT * 128, 128], [1, 2 * CT * 128]], offset=m * 2 * CT * 128,
                                            scalar_offset=e_dma, indirect_dim=0), oob_mode=wom)
            sc = nl.ndarray((128, 2, CT), dtype=f32, buffer=nl.sbuf)  # [i, g, c]
            nisa.dma_copy(dst=sc, src=dsg.ap(pattern=[[CT, 128], [M * 128 * CT, 2], [1, CT]], offset=m * 128 * CT,
                                             scalar_offset=e_dma, indirect_dim=0), oob_mode=wom)
            gs = []
            for g in range(2):
                acc = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
                for h0 in range(0, CT, CG):
                    pp = nl.ndarray((128, CG, LW), dtype=f32, buffer=nl.psum)
                    for c in range(h0, h0 + CG):
                        nisa.nc_matmul(dst=pp[:, c - h0, :], stationary=wq[:, g, c * 128:(c + 1) * 128].view(fp8),
                                       moving=xT[:, c, :], accumulate=False)
                    prod = nl.ndarray((128, CG, LW), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=prod, data1=pp, data2=sc.ap(pattern=[[2 * CT, 128], [1, CG], [0, LW]],
                                                                       offset=g * CT + h0),
                                       op=nl.multiply, engine=nisa.vector_engine)
                    if h0 == 0:
                        nisa.tensor_reduce(dst=acc, op=nl.add, data=prod.ap(pattern=[[CG * LW, 128], [1, LW], [LW, CG]],
                                                                             offset=0), axis=2)
                    else:
                        part = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
                        nisa.tensor_reduce(dst=part, op=nl.add, data=prod.ap(pattern=[[CG * LW, 128], [1, LW], [LW, CG]],
                                                                              offset=0), axis=2)
                        nisa.tensor_tensor(dst=acc, data1=acc, data2=part, op=nl.add, engine=nisa.vector_engine)
                gs.append(acc)
            gc = nl.ndarray((128, LW), dtype=bf16, buffer=nl.sbuf)
            uc = nl.ndarray((128, LW), dtype=bf16, buffer=nl.sbuf)
            if act == 1:
                nisa.tensor_scalar(dst=gc, data=gs[0], op0=nl.minimum, operand0=lim, engine=nisa.vector_engine)
                nisa.tensor_scalar(dst=uc, data=gs[1], op0=nl.minimum, operand0=lim, op1=nl.maximum, operand1=-lim,
                                   engine=nisa.vector_engine)
            else:
                nisa.tensor_copy(dst=gc, src=gs[0], engine=nisa.vector_engine)
                nisa.tensor_copy(dst=uc, src=gs[1], engine=nisa.vector_engine)
            sl = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=sl, op=nl.silu, data=gc)
            nisa.tensor_tensor(dst=aT[:, m, :], data1=sl, data2=uc, op=nl.multiply, engine=nisa.vector_engine)
        rcv = S["rcv"] if sp else aT  # (no inner function: see _pass)
        if sp:  # the other program's half of a^T into rcv
            nisa.sendrecv(src=aT[:, m_lo:m_hi, :], dst=rcv, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0)
        o_lo = (1 - pid) * Mh if sp else 0

        ys = nl.ndarray((LW, H), dtype=bf16, buffer=nl.sbuf)
        for q in range(q_lo, q_hi):
            dq8 = nl.ndarray((128, M, 512), dtype=u8, buffer=nl.sbuf)
            nisa.dma_copy(dst=dq8, src=dn.ap(pattern=[[M * H, 128], [H, M], [1, 512]], offset=q * 512,
                                             scalar_offset=e_dma, indirect_dim=0), oob_mode=wom)
            sd = nl.ndarray((128, 4, M), dtype=f32, buffer=nl.sbuf)  # [h, the chunk's 4 h-tiles, m]
            nisa.dma_copy(dst=sd, src=dsd.ap(pattern=[[CT * M, 128], [M, 4], [1, M]], offset=q * 4 * M,
                                             scalar_offset=e_dma, indirect_dim=0), oob_mode=wom)
            # the transposes of yt write it: fp32 PSUM on gen2, the input's bf16 from gen3 on ("nc_matmul (transpose
            # mode) dst dtype must match input dtype on gen3+", nki/isa/_validation.py); the same bf16 values
            py = nl.ndarray((LW, 512), dtype=f32 if nisa.get_nc_version() == nisa.nc_version.gen2 else bf16,
                            buffer=nl.psum)
            MG = min(M, 512 // LW)  # I-chunks per PSUM stack (2 KB per partition)
            for hh in range(4):
                yt = nl.ndarray((128, LW), dtype=bf16, buffer=nl.sbuf)
                ya = None
                for m0 in range(0, M, MG):
                    pd = nl.ndarray((128, MG, LW), dtype=f32, buffer=nl.psum)
                    for m in range(m0, m0 + MG):
                        a_src, a_m = (aT, m) if (not sp or m_lo <= m < m_hi) else (rcv, m - o_lo)
                        nisa.nc_matmul(dst=pd[:, m - m0, :], stationary=dq8[:, m, hh * 128:(hh + 1) * 128].view(fp8),
                                       moving=a_src[:, a_m, :], accumulate=False)
                    prod = nl.ndarray((128, MG, LW), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=prod, data1=pd, data2=sd.ap(pattern=[[4 * M, 128], [1, MG], [0, LW]],
                                                                       offset=hh * M + m0),
                                       op=nl.multiply, engine=nisa.vector_engine)
                    red = prod.ap(pattern=[[MG * LW, 128], [1, LW], [LW, MG]], offset=0)
                    if MG == M:  # one stack: reduced straight to bf16
                        nisa.tensor_reduce(dst=yt, op=nl.add, data=red, axis=2)
                    elif m0 == 0:
                        ya = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
                        nisa.tensor_reduce(dst=ya, op=nl.add, data=red, axis=2)
                    else:
                        part = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
                        nisa.tensor_reduce(dst=part, op=nl.add, data=red, axis=2)
                        if m0 + MG < M:
                            nisa.tensor_tensor(dst=ya, data1=ya, data2=part, op=nl.add, engine=nisa.vector_engine)
                        else:  # the last stack: the fp32 sum rounded once
                            nisa.tensor_tensor(dst=yt, data1=ya, data2=part, op=nl.add, engine=nisa.vector_engine)
                nisa.nc_transpose(dst=py[:, hh * 128:(hh + 1) * 128], data=yt, engine=nisa.tensor_engine)
            nisa.activation(dst=ys[:, q * 512:(q + 1) * 512], op=nl.copy, data=py, scale=w)
        h_lo, hw = (pid * (H // 2), H // 2) if sp else (0, H)  # this program's output columns
        if wr:  # unsplit with two programs: program 0 alone writes
            dst = out.ap(pattern=[[H, LW], [1, hw]], offset=h_lo,
                         vector_offset=ts.ap(pattern=[[1, LW], [1, 1]], offset=0), indirect_dim=0)
            nisa.dma_compute(dst=dst, srcs=[dst, ys[:, h_lo:h_lo + hw]], reduce_op=nl.add, oob_mode=oob_mode.skip)

    @nki.jit
    def kiln_moe_ep_small(x, topi, wts, lmap, gu, dsg, dn, dsd, LW: int, PMAX: int, act: int, lim: float, rev: int):
        """kiln_moe_ep_kernel's contract for small batches (decode, verify): the same plan with passes of LW lanes
        (16 or 32), every local expert's first pass static (an unused expert's loads skipped), the overflow passes
        in a device loop, each pass _pass_s (per-row scales, no dequantization). dsg fp32 [El, 2, M, 128, CT] and dsd
        fp32 [El, 128, CT, M]: the scales with the output rows / columns on the partitions (pack())."""
        C, H = x.shape
        K = topi.shape[1]
        E = lmap.shape[1]
        El = gu.shape[0]
        M = gu.shape[2]
        CT = H // 128
        f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
        P = _plan_s(topi, wts, lmap, C, K, E, El, LW, PMAX)
        S = dict(H=H, M=M, CT=CT, LW=LW, gu=gu, dsg=dsg, dn=dn, dsd=dsd, act=act, lim=lim, x=x)
        out = nl.ndarray((C, H), dtype=x.dtype, buffer=nl.shared_hbm)
        zt = nl.ndarray((128, H), dtype=x.dtype, buffer=nl.sbuf)
        nisa.memset(dst=zt, value=0.0)
        for t in range(C // 128):
            nisa.dma_copy(dst=out[t * 128:(t + 1) * 128, :], src=zt)
        for e in range(El):
            ts = nl.ndarray((LW, 1), dtype=i32, buffer=nl.sbuf)
            _lanes_s(P, P["ngi"], e * P["NT"], LW, ts)
            w = nl.ndarray((LW, 1), dtype=f32, buffer=nl.sbuf)
            _weights_s(P, ts, float(e), LW, w)
            xr = nl.ndarray((LW, H), dtype=bf16, buffer=nl.sbuf)
            nisa.memset(dst=xr, value=0.0, engine=nisa.gpsimd_engine)
            _pass_s(S, P["eeff"][:, e:e + 1], ts, w, xr, out, True)
        xrs = nl.ndarray((LW, H), dtype=bf16, buffer=nl.sbuf)  # the loop's, zeroed once
        nisa.memset(dst=xrs, value=0.0)
        rp = nisa.register_alloc()
        nisa.register_load(rp, P["nov"])

        def body(it):
            _small_overflow_pass(S, P, it, xrs, out)

        nl.fori_loop(0, rp, body)
        return out

    def _small_overflow_pass(S, P, it, xrs, out):
        """Overflow pass `it` of kiln_moe_ep_small (as _ep_overflow_pass)."""
        NT, LW = P["NT"], S["LW"]
        e_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=e_sb, src=P["etab"].ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=it, indirect_dim=0))
        ef = nl.ndarray((LW, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ef, src=P["eftab"].ap(pattern=[[0, LW], [1, 1]], offset=0, scalar_offset=it, indirect_dim=0))
        j0 = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=j0, src=P["jtab"].ap(pattern=[[0, 128], [1, 1]], offset=0, scalar_offset=it, indirect_dim=0))
        ng = nl.ndarray((128, NT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ng, src=P["ngi_h"].ap(pattern=[[NT, 128], [1, NT]], offset=0, scalar_offset=e_sb,
                                                indirect_dim=0))
        nisa.tensor_scalar(dst=ng, data=ng, op0=nl.add, operand0=j0, engine=nisa.vector_engine)
        ts = nl.ndarray((LW, 1), dtype=nl.int32, buffer=nl.sbuf)
        _lanes_s(P, ng, 0, LW, ts)
        w = nl.ndarray((LW, 1), dtype=nl.float32, buffer=nl.sbuf)
        _weights_s(P, ts, ef, LW, w)
        _pass_s(S, e_sb, ts, w, xrs, out, False)

    # --- end of the small-lane kernel's source ---

    # --- the small-lane kernel v2 (KILN_MOE_EP_SMALL_V=2): every local expert with pairs exactly once ---

    def _plan_s2(topi, wts, lmap, C, K, E, El, LW):
        """_plan_s's membership, prefix counts and lane-token bias rows (ngi, ngi_h), and two pass tables in place of
        static first passes and overflow passes: the experts with 1 .. LW pairs (one small-lane pass each) and the
        experts with more (one dequantize-first pass of 128 lanes each: C <= 128 rows, so no expert has more than 128
        pairs). Each table is local-expert order (_ovtable with one pass per selected expert, first rank 0), its
        count an int32 SBUF scalar for a device loop, so an expert with no pair costs nothing and none is loaded
        twice."""
        f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
        NT = C // 128
        tk = nl.ndarray((128, NT, K), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=tk, src=topi.ap(pattern=[[K, 128], [128 * K, NT], [1, K]], offset=0))
        tku = nl.ndarray((128, NT, K), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tku, src=tk, engine=nisa.vector_engine)
        lmi = nl.ndarray((128, E), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=lmi, src=lmap.ap(pattern=[[0, 128], [1, E]], offset=0))
        lmf = nl.ndarray((128, E), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lmf, src=lmi, engine=nisa.vector_engine)
        loc = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=loc, data=lmf, indices=tku)
        loc_h = nl.ndarray((C, K), dtype=f32, buffer=nl.private_hbm)
        nisa.dma_copy(dst=loc_h.ap(pattern=[[K, 128], [128 * K, NT], [1, K]], offset=0), src=loc)
        EN = El * NT
        mem = nl.ndarray((128, EN), dtype=bf16, buffer=nl.sbuf)
        for e in range(El):
            eq = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=eq, data=loc, op0=nl.equal, operand0=float(e), engine=nisa.vector_engine)
            nisa.tensor_reduce(dst=mem[:, e * NT:(e + 1) * NT], op=nl.add, data=eq, axis=2)
        uu = nl.ndarray((128, 128), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=uu, pattern=[[1, 128]], offset=0, channel_multiplier=-1)
        uuf = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=uuf, src=uu, engine=nisa.vector_engine)
        lt = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lt, data=uuf, op0=nl.greater, operand0=0.0, engine=nisa.vector_engine)
        on128 = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=on128, value=1.0)
        wit = nl.ndarray((128, EN), dtype=f32, buffer=nl.sbuf)
        tot = nl.ndarray((128, EN), dtype=f32, buffer=nl.sbuf)
        for c0 in range(0, EN, 512):
            cw = min(512, EN - c0)
            pw = nl.ndarray((128, cw), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pw, stationary=lt, moving=mem[:, c0:c0 + cw], accumulate=False)
            nisa.tensor_copy(dst=wit[:, c0:c0 + cw], src=pw, engine=nisa.vector_engine)
            pt = nl.ndarray((128, cw), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pt, stationary=on128, moving=mem[:, c0:c0 + cw], accumulate=False)
            nisa.tensor_copy(dst=tot[:, c0:c0 + cw], src=pt, engine=nisa.vector_engine)
        inc = nl.ndarray((128, El, NT), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=inc, src=tot.ap(pattern=[[EN, 128], [NT, El], [1, NT]], offset=0), engine=nisa.vector_engine)
        sh = 1
        while sh < NT:
            prev = nl.ndarray((128, El, NT), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=prev, src=inc, engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=inc[:, :, sh:NT], data1=prev[:, :, sh:NT], data2=prev[:, :, 0:NT - sh], op=nl.add,
                               engine=nisa.vector_engine)
            sh *= 2
        ngi = nl.ndarray((128, EN), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ngi, data1=inc.ap(pattern=[[EN, 128], [1, EN]], offset=0), data2=tot, op=nl.subtract,
                           engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=ngi, data1=ngi, data2=wit, op=nl.add, engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=ngi, data1=ngi, data2=mem, op=nl.add, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=ngi, data=ngi, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=0.5,
                           engine=nisa.vector_engine)
        ngi_h = nl.ndarray((El, 128, NT), dtype=f32, buffer=nl.private_hbm)
        nisa.dma_copy(dst=ngi_h.ap(pattern=[[NT, 128], [128 * NT, El], [1, NT]], offset=0), src=ngi)
        cnt = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cnt, src=inc[0:1, :, NT - 1], engine=nisa.vector_engine)
        big = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)  # [n_e > LW]
        nisa.tensor_scalar(dst=big, data=cnt, op0=nl.greater, operand0=float(LW), engine=nisa.vector_engine)
        sml = nl.ndarray((1, El), dtype=f32, buffer=nl.sbuf)  # [n_e > 0] - [n_e > LW]
        nisa.tensor_scalar(dst=sml, data=cnt, op0=nl.greater, operand0=0.0, engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=sml, data1=sml, data2=big, op=nl.subtract, engine=nisa.vector_engine)
        b0 = nl.ndarray((1, El + 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=b0, value=0.0)
        tsm = _ovtable(sml, b0, LW, El, El)
        tbg = _ovtable(big, b0, 128, El, El)
        ji = nl.ndarray((128, LW), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ji, pattern=[[1, LW]], offset=0, channel_multiplier=0)
        jr = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=jr, src=ji, engine=nisa.vector_engine)
        jb = nl.ndarray((128, 128), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=jb, pattern=[[1, 128]], offset=0, channel_multiplier=0)
        jbf = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=jbf, src=jb, engine=nisa.vector_engine)
        ones1 = nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=ones1, value=1.0)
        return dict(NT=NT, C=C, K=K, loc_h=loc_h, wts=wts, ngi_h=ngi_h, tsm=tsm, tbg=tbg, jrow=jr, jrow_b=[jbf],
                    ones1=ones1)

    def _table_pass(P, tab, it, LWp):
        """Pass `it` of a two-table plan: its expert (int32 SBUF scalar, and fp32 on LWp partitions) and the expert's
        lane-token bias rows (its first rank is 0)."""
        e_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=e_sb, src=tab["e"].ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=it, indirect_dim=0))
        ef = nl.ndarray((LWp, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ef, src=tab["ef"].ap(pattern=[[0, LWp], [1, 1]], offset=0, scalar_offset=it, indirect_dim=0))
        ng = nl.ndarray((128, P["NT"]), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ng, src=P["ngi_h"].ap(pattern=[[P["NT"], 128], [1, P["NT"]]], offset=0, scalar_offset=e_sb,
                                                indirect_dim=0))
        return e_sb, ef, ng

    @nki.jit
    def kiln_moe_ep_small2(x, topi, wts, lmap, gu, dsg, dn, dsd, sgu, sdn, LW: int, act: int, lim: float, bc: int,
                           rev: int, tsc: int = 1, spl: int = 0):
        """kiln_moe_ep_small's contract (C <= 128 rows), every local expert with pairs run exactly once and nothing
        run for one without: experts with 1 .. LW pairs by a small-lane pass (_pass_s: per-row scales dsg / dsd, the
        fp8 weights as stored, no dequantization), experts with more by one dequantize-first pass of 128 lanes
        (_pass: the scales sgu / sdn, tile scales with tsc as kiln_moe_ep_kernel takes them), each kind in its own
        device loop over the plan's table (_plan_s2). Per token the pairs are added small passes first, then the
        dequantized ones, each in local-expert order (emulate(..., small=2))."""
        C, H = x.shape
        K = topi.shape[1]
        E = lmap.shape[1]
        El = gu.shape[0]
        M = gu.shape[2]
        CT = H // 128
        f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
        L = _lnc_setup(spl, M, H, [LW, 128])  # first: the swap buffers at the same address on both cores
        P = _plan_s2(topi, wts, lmap, C, K, E, El, LW)
        S = dict(H=H, M=M, CT=CT, LW=LW, gu=gu, dsg=dsg, dn=dn, dsd=dsd, act=act, lim=lim, x=x, rcv=_rcv(L["rcv"], LW),
                 sp=L["sp"], pid=L["pid"], wr=L["wr"])
        SB = dict(H=H, M=M, CT=CT, LW=128, NS=1, gu=gu, sgu=sgu, dn=dn, sdn=sdn, sels=_consts(bc), act=act, lim=lim,
                  tsc=tsc, rcv=_rcv(L["rcv"], 128), sp=L["sp"], pid=L["pid"], wr=L["wr"])
        Pd = dict(NT=P["NT"], C=C, K=K, jrow=P["jrow_b"], ones1=P["ones1"], loc_h=P["loc_h"], wts=wts)
        out = nl.ndarray((C, H), dtype=x.dtype, buffer=nl.shared_hbm)
        _zero_out(out, C, H, L)
        xrs = nl.ndarray((LW, H), dtype=bf16, buffer=nl.sbuf)  # the small loop's x rows, zeroed once
        nisa.memset(dst=xrs, value=0.0)
        xrb = nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf)  # the dequantize-first loop's
        nisa.memset(dst=xrb, value=0.0)
        rs = nisa.register_alloc()
        nisa.register_load(rs, P["tsm"]["n"])

        def body_s(it):
            e_sb, ef, ng = _table_pass(P, P["tsm"], it, LW)
            ts = nl.ndarray((LW, 1), dtype=i32, buffer=nl.sbuf)
            _lanes_s(P, ng, 0, LW, ts)
            w = nl.ndarray((LW, 1), dtype=f32, buffer=nl.sbuf)
            _weights_s(P, ts, ef, LW, w)
            _pass_s(S, e_sb, ts, w, xrs, out, False)

        nl.fori_loop(0, rs, body_s)
        rb = nisa.register_alloc()
        nisa.register_load(rb, P["tbg"]["n"])

        def body_b(it):
            e_sb, ef, ng = _table_pass(P, P["tbg"], it, 128)
            ts = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
            _lane_tokens(SB, Pd, ng, 0, 0, ts)
            w = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
            _lane_weights(SB, Pd, ts, ef, w)
            _pass(SB, dict(xkind="gather", ykind="rmw", ekind="dyn", e_sb=e_sb, x=x, ts=ts, xr=[xrb], w=w, out=out))

        nl.fori_loop(0, rb, body_b)
        _lnc_end(out, L)
        return out

    # --- end of the small-lane kernel v2's source ---

    # --- the small-lane kernel v3 (KILN_MOE_EP_SMALL_V=3): v2 with a small pass's weights in few, large DMAs ---

    def _pass_s3(S, e_reg, ts, w, xr, out):
        """_pass_s's arithmetic, instruction for instruction (the same matmuls, PSUM stacks, products and reductions on
        the same values), with the expert's bytes moved in 12 DMAs instead of 48, all offset by one register instead of a
        TENSOR_LOAD each: the whole down projection [128, M, H] (64 KB per partition, one DMA), gate_up two I-chunks
        per DMA ([128, 2, 2, CT 128], 16 KB per partition) through a ring three deep, and the tile scales tsg [M, 2, CT]
        and tsd [M, CT] broadcast to every partition (on a block-fitted checkpoint each is the per-row scale dsg / dsd of
        every row of its tile, so the products are the same numbers)."""
        H, M, CT, LW = S["H"], S["M"], S["CT"], S["LW"]
        gu, tsg, dn, tsd, act, lim = S["gu"], S["tsg"], S["dn"], S["tsd"], S["act"], S["lim"]
        f32, bf16, fp8, u8 = nl.float32, nl.bfloat16, nl.float8_e4m3, nl.uint8
        CG = min(CT, 512 // LW)
        MP = 2  # I-chunks per gate_up DMA
        NP = M // MP
        GU = M * 2 * CT * 128  # gate_up bytes per partition and expert
        sg = nl.ndarray((128, M, 2, CT), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sg, src=tsg.ap(pattern=[[0, 128], [1, M * 2 * CT]], offset=0, scalar_offset=e_reg,
                                         indirect_dim=0))
        gq = []
        for k in range(min(2, NP)):
            b = nl.ndarray((128, MP, 2, CT * 128), dtype=u8, buffer=nl.sbuf)
            nisa.dma_copy(dst=b, src=gu.ap(pattern=[[GU, 128], [1, MP * 2 * CT * 128]], offset=k * MP * 2 * CT * 128,
                                           scalar_offset=e_reg, indirect_dim=0))
            gq.append(b)
        nisa.dma_copy(dst=xr, src=S["x"].ap(pattern=[[H, LW], [1, H]], offset=0,
                                             vector_offset=ts.ap(pattern=[[1, LW], [1, 1]], offset=0), indirect_dim=0),
                      oob_mode=oob_mode.skip)
        dw = nl.ndarray((128, M, H), dtype=u8, buffer=nl.sbuf)
        nisa.dma_copy(dst=dw, src=dn.ap(pattern=[[M * H, 128], [1, M * H]], offset=0, scalar_offset=e_reg, indirect_dim=0))
        sdt = nl.ndarray((128, M, CT), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sdt, src=tsd.ap(pattern=[[0, 128], [1, M * CT]], offset=0, scalar_offset=e_reg, indirect_dim=0))
        xT = nl.ndarray((128, CT, LW), dtype=bf16, buffer=nl.sbuf)
        for c4 in range(CT // 4):
            px = nl.ndarray((128, 4, LW), dtype=f32 if nisa.get_nc_version() == nisa.nc_version.gen2 else bf16,
                            buffer=nl.psum)
            for j in range(4):
                c = c4 * 4 + j
                nisa.nc_transpose(dst=px[:, j, :], data=xr[:, c * 128:(c + 1) * 128], engine=nisa.tensor_engine)
            nisa.activation(dst=xT[:, c4 * 4:(c4 + 1) * 4, :], op=nl.copy, data=px)
        aT = nl.ndarray((128, M, LW), dtype=bf16, buffer=nl.sbuf)
        for m in range(M):
            k = m // MP
            if m % MP == 0 and k + 2 < NP:  # the pair two ahead
                b = nl.ndarray((128, MP, 2, CT * 128), dtype=u8, buffer=nl.sbuf)
                nisa.dma_copy(dst=b, src=gu.ap(pattern=[[GU, 128], [1, MP * 2 * CT * 128]],
                                               offset=(k + 2) * MP * 2 * CT * 128, scalar_offset=e_reg, indirect_dim=0))
                gq.append(b)
            wq = gq[k]
            gs = []
            for g in range(2):
                acc = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
                for h0 in range(0, CT, CG):
                    pp = nl.ndarray((128, CG, LW), dtype=f32, buffer=nl.psum)
                    for c in range(h0, h0 + CG):
                        nisa.nc_matmul(dst=pp[:, c - h0, :], stationary=wq[:, m % MP, g, c * 128:(c + 1) * 128].view(fp8),
                                       moving=xT[:, c, :], accumulate=False)
                    prod = nl.ndarray((128, CG, LW), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=prod, data1=pp, data2=sg.ap(pattern=[[M * 2 * CT, 128], [1, CG], [0, LW]],
                                                                       offset=m * 2 * CT + g * CT + h0),
                                       op=nl.multiply, engine=nisa.vector_engine)
                    if h0 == 0:
                        nisa.tensor_reduce(dst=acc, op=nl.add, data=prod.ap(pattern=[[CG * LW, 128], [1, LW], [LW, CG]],
                                                                             offset=0), axis=2)
                    else:
                        part = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
                        nisa.tensor_reduce(dst=part, op=nl.add, data=prod.ap(pattern=[[CG * LW, 128], [1, LW], [LW, CG]],
                                                                              offset=0), axis=2)
                        nisa.tensor_tensor(dst=acc, data1=acc, data2=part, op=nl.add, engine=nisa.vector_engine)
                gs.append(acc)
            gc = nl.ndarray((128, LW), dtype=bf16, buffer=nl.sbuf)
            uc = nl.ndarray((128, LW), dtype=bf16, buffer=nl.sbuf)
            if act == 1:
                nisa.tensor_scalar(dst=gc, data=gs[0], op0=nl.minimum, operand0=lim, engine=nisa.vector_engine)
                nisa.tensor_scalar(dst=uc, data=gs[1], op0=nl.minimum, operand0=lim, op1=nl.maximum, operand1=-lim,
                                   engine=nisa.vector_engine)
            else:
                nisa.tensor_copy(dst=gc, src=gs[0], engine=nisa.vector_engine)
                nisa.tensor_copy(dst=uc, src=gs[1], engine=nisa.vector_engine)
            sl = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=sl, op=nl.silu, data=gc)
            nisa.tensor_tensor(dst=aT[:, m, :], data1=sl, data2=uc, op=nl.multiply, engine=nisa.vector_engine)
        ys = nl.ndarray((LW, H), dtype=bf16, buffer=nl.sbuf)
        MG = min(M, 512 // LW)  # I-chunks per PSUM stack (2 KB per partition)
        for q in range(H // 512):
            # the transposes of yt write it: fp32 PSUM on gen2, the input's bf16 from gen3 on ("nc_matmul (transpose
            # mode) dst dtype must match input dtype on gen3+", nki/isa/_validation.py); the same bf16 values
            py = nl.ndarray((LW, 512), dtype=f32 if nisa.get_nc_version() == nisa.nc_version.gen2 else bf16,
                            buffer=nl.psum)
            for hh in range(4):
                b0 = q * 512 + hh * 128
                yt = nl.ndarray((128, LW), dtype=bf16, buffer=nl.sbuf)
                ya = None
                for m0 in range(0, M, MG):
                    pd = nl.ndarray((128, MG, LW), dtype=f32, buffer=nl.psum)
                    for m in range(m0, m0 + MG):
                        nisa.nc_matmul(dst=pd[:, m - m0, :], stationary=dw[:, m, b0:b0 + 128].view(fp8),
                                       moving=aT[:, m, :], accumulate=False)
                    prod = nl.ndarray((128, MG, LW), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=prod, data1=pd, data2=sdt.ap(pattern=[[M * CT, 128], [CT, MG], [0, LW]],
                                                                        offset=m0 * CT + 4 * q + hh),
                                       op=nl.multiply, engine=nisa.vector_engine)
                    red = prod.ap(pattern=[[MG * LW, 128], [1, LW], [LW, MG]], offset=0)
                    if MG == M:
                        nisa.tensor_reduce(dst=yt, op=nl.add, data=red, axis=2)
                    elif m0 == 0:
                        ya = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
                        nisa.tensor_reduce(dst=ya, op=nl.add, data=red, axis=2)
                    else:
                        part = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
                        nisa.tensor_reduce(dst=part, op=nl.add, data=red, axis=2)
                        if m0 + MG < M:
                            nisa.tensor_tensor(dst=ya, data1=ya, data2=part, op=nl.add, engine=nisa.vector_engine)
                        else:
                            nisa.tensor_tensor(dst=yt, data1=ya, data2=part, op=nl.add, engine=nisa.vector_engine)
                nisa.nc_transpose(dst=py[:, hh * 128:(hh + 1) * 128], data=yt, engine=nisa.tensor_engine)
            nisa.activation(dst=ys[:, q * 512:(q + 1) * 512], op=nl.copy, data=py, scale=w)
        dst = out.ap(pattern=[[H, LW], [1, H]], offset=0, vector_offset=ts.ap(pattern=[[1, LW], [1, 1]], offset=0),
                     indirect_dim=0)
        nisa.dma_compute(dst=dst, srcs=[dst, ys], reduce_op=nl.add, oob_mode=oob_mode.skip)

    @nki.jit
    def kiln_moe_ep_small3(x, topi, wts, lmap, gu, dn, tsg, tsd, LW: int, act: int, lim: float, bc: int, rev: int):
        """kiln_moe_ep_small2 (the same plan, tables, pass kinds and arithmetic) on the tile-scale layout only, its small
        passes through _pass_s3 (few large weight DMAs, one register offset); the dequantize-first passes as v2's."""
        C, H = x.shape
        K = topi.shape[1]
        E = lmap.shape[1]
        El = gu.shape[0]
        M = gu.shape[2]
        CT = H // 128
        f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
        P = _plan_s2(topi, wts, lmap, C, K, E, El, LW)
        S = dict(H=H, M=M, CT=CT, LW=LW, gu=gu, tsg=tsg, dn=dn, tsd=tsd, act=act, lim=lim, x=x)
        SB = dict(H=H, M=M, CT=CT, LW=128, NS=1, gu=gu, sgu=tsg, dn=dn, sdn=tsd, sels=_consts(bc), act=act, lim=lim,
                  tsc=1)
        Pd = dict(NT=P["NT"], C=C, K=K, jrow=P["jrow_b"], ones1=P["ones1"], loc_h=P["loc_h"], wts=wts)
        out = nl.ndarray((C, H), dtype=x.dtype, buffer=nl.shared_hbm)
        zt = nl.ndarray((128, H), dtype=x.dtype, buffer=nl.sbuf)
        nisa.memset(dst=zt, value=0.0)
        for t in range(C // 128):
            nisa.dma_copy(dst=out[t * 128:(t + 1) * 128, :], src=zt)
        xrs = nl.ndarray((LW, H), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=xrs, value=0.0)
        xrb = nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=xrb, value=0.0)
        er = nisa.register_alloc()
        rs = nisa.register_alloc()
        nisa.register_load(rs, P["tsm"]["n"])

        def body_s(it):
            e_sb, ef, ng = _table_pass(P, P["tsm"], it, LW)
            nisa.register_load(er, e_sb)
            ts = nl.ndarray((LW, 1), dtype=i32, buffer=nl.sbuf)
            _lanes_s(P, ng, 0, LW, ts)
            w = nl.ndarray((LW, 1), dtype=f32, buffer=nl.sbuf)
            _weights_s(P, ts, ef, LW, w)
            _pass_s3(S, er, ts, w, xrs, out)

        nl.fori_loop(0, rs, body_s)
        rb = nisa.register_alloc()
        nisa.register_load(rb, P["tbg"]["n"])

        def body_b(it):
            e_sb, ef, ng = _table_pass(P, P["tbg"], it, 128)
            ts = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
            _lane_tokens(SB, Pd, ng, 0, 0, ts)
            w = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
            _lane_weights(SB, Pd, ts, ef, w)
            _pass(SB, dict(xkind="gather", ykind="rmw", ekind="dyn", e_sb=e_sb, x=x, ts=ts, xr=[xrb], w=w, out=out))

        nl.fori_loop(0, rb, body_b)
        return out

    # --- end of the small-lane kernel v3's source ---

    # --- the small-lane kernel v5 (KILN_MOE_EP_SMALL_V=5): v3's small passes two per loop iteration ---

    def _pass_s5(S, e_reg, ts, w, xr, out):
        """_pass_s3 (the same instructions on the same values) with the whole-down DMA issued right after the last gate_up
        DMA instead of after the first two: alone a pass is no faster (its down phase waits for those bytes either way),
        but two passes in one loop iteration (kiln_moe_ep_small5) then stream the second expert's gate_up under the
        first one's down phase."""
        H, M, CT, LW = S["H"], S["M"], S["CT"], S["LW"]
        gu, tsg, dn, tsd, act, lim = S["gu"], S["tsg"], S["dn"], S["tsd"], S["act"], S["lim"]
        f32, bf16, fp8, u8 = nl.float32, nl.bfloat16, nl.float8_e4m3, nl.uint8
        CG = min(CT, 512 // LW)
        MP = 2  # I-chunks per gate_up DMA
        NP = M // MP
        GU = M * 2 * CT * 128  # gate_up bytes per partition and expert
        sg = nl.ndarray((128, M, 2, CT), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sg, src=tsg.ap(pattern=[[0, 128], [1, M * 2 * CT]], offset=0, scalar_offset=e_reg,
                                         indirect_dim=0))
        gq = []
        for k in range(min(2, NP)):
            b = nl.ndarray((128, MP, 2, CT * 128), dtype=u8, buffer=nl.sbuf)
            nisa.dma_copy(dst=b, src=gu.ap(pattern=[[GU, 128], [1, MP * 2 * CT * 128]], offset=k * MP * 2 * CT * 128,
                                           scalar_offset=e_reg, indirect_dim=0))
            gq.append(b)
        nisa.dma_copy(dst=xr, src=S["x"].ap(pattern=[[H, LW], [1, H]], offset=0,
                                             vector_offset=ts.ap(pattern=[[1, LW], [1, 1]], offset=0), indirect_dim=0),
                      oob_mode=oob_mode.skip)
        sdt = nl.ndarray((128, M, CT), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sdt, src=tsd.ap(pattern=[[0, 128], [1, M * CT]], offset=0, scalar_offset=e_reg, indirect_dim=0))
        dw = nl.ndarray((128, M, H), dtype=u8, buffer=nl.sbuf)
        dwq = []
        xT = nl.ndarray((128, CT, LW), dtype=bf16, buffer=nl.sbuf)
        for c4 in range(CT // 4):
            px = nl.ndarray((128, 4, LW), dtype=f32 if nisa.get_nc_version() == nisa.nc_version.gen2 else bf16,
                            buffer=nl.psum)
            for j in range(4):
                c = c4 * 4 + j
                nisa.nc_transpose(dst=px[:, j, :], data=xr[:, c * 128:(c + 1) * 128], engine=nisa.tensor_engine)
            nisa.activation(dst=xT[:, c4 * 4:(c4 + 1) * 4, :], op=nl.copy, data=px)
        aT = nl.ndarray((128, M, LW), dtype=bf16, buffer=nl.sbuf)
        for m in range(M):
            k = m // MP
            if m % MP == 0 and k + 2 < NP:  # the pair two ahead
                b = nl.ndarray((128, MP, 2, CT * 128), dtype=u8, buffer=nl.sbuf)
                nisa.dma_copy(dst=b, src=gu.ap(pattern=[[GU, 128], [1, MP * 2 * CT * 128]],
                                               offset=(k + 2) * MP * 2 * CT * 128, scalar_offset=e_reg, indirect_dim=0))
                gq.append(b)
            if m % MP == 0 and len(dwq) == 0 and (k + 2 >= NP - 1):  # the last gate_up DMA is out: the down weights next
                nisa.dma_copy(dst=dw, src=dn.ap(pattern=[[M * H, 128], [1, M * H]], offset=0, scalar_offset=e_reg,
                                                indirect_dim=0))
                dwq.append(dw)
            wq = gq[k]
            gs = []
            for g in range(2):
                acc = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
                for h0 in range(0, CT, CG):
                    pp = nl.ndarray((128, CG, LW), dtype=f32, buffer=nl.psum)
                    for c in range(h0, h0 + CG):
                        nisa.nc_matmul(dst=pp[:, c - h0, :], stationary=wq[:, m % MP, g, c * 128:(c + 1) * 128].view(fp8),
                                       moving=xT[:, c, :], accumulate=False)
                    prod = nl.ndarray((128, CG, LW), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=prod, data1=pp, data2=sg.ap(pattern=[[M * 2 * CT, 128], [1, CG], [0, LW]],
                                                                       offset=m * 2 * CT + g * CT + h0),
                                       op=nl.multiply, engine=nisa.vector_engine)
                    if h0 == 0:
                        nisa.tensor_reduce(dst=acc, op=nl.add, data=prod.ap(pattern=[[CG * LW, 128], [1, LW], [LW, CG]],
                                                                             offset=0), axis=2)
                    else:
                        part = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
                        nisa.tensor_reduce(dst=part, op=nl.add, data=prod.ap(pattern=[[CG * LW, 128], [1, LW], [LW, CG]],
                                                                              offset=0), axis=2)
                        nisa.tensor_tensor(dst=acc, data1=acc, data2=part, op=nl.add, engine=nisa.vector_engine)
                gs.append(acc)
            gc = nl.ndarray((128, LW), dtype=bf16, buffer=nl.sbuf)
            uc = nl.ndarray((128, LW), dtype=bf16, buffer=nl.sbuf)
            if act == 1:
                nisa.tensor_scalar(dst=gc, data=gs[0], op0=nl.minimum, operand0=lim, engine=nisa.vector_engine)
                nisa.tensor_scalar(dst=uc, data=gs[1], op0=nl.minimum, operand0=lim, op1=nl.maximum, operand1=-lim,
                                   engine=nisa.vector_engine)
            else:
                nisa.tensor_copy(dst=gc, src=gs[0], engine=nisa.vector_engine)
                nisa.tensor_copy(dst=uc, src=gs[1], engine=nisa.vector_engine)
            sl = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
            nisa.activation(dst=sl, op=nl.silu, data=gc)
            nisa.tensor_tensor(dst=aT[:, m, :], data1=sl, data2=uc, op=nl.multiply, engine=nisa.vector_engine)
        ys = nl.ndarray((LW, H), dtype=bf16, buffer=nl.sbuf)
        MG = min(M, 512 // LW)  # I-chunks per PSUM stack (2 KB per partition)
        for q in range(H // 512):
            # the transposes of yt write it: fp32 PSUM on gen2, the input's bf16 from gen3 on ("nc_matmul (transpose
            # mode) dst dtype must match input dtype on gen3+", nki/isa/_validation.py); the same bf16 values
            py = nl.ndarray((LW, 512), dtype=f32 if nisa.get_nc_version() == nisa.nc_version.gen2 else bf16,
                            buffer=nl.psum)
            for hh in range(4):
                b0 = q * 512 + hh * 128
                yt = nl.ndarray((128, LW), dtype=bf16, buffer=nl.sbuf)
                ya = None
                for m0 in range(0, M, MG):
                    pd = nl.ndarray((128, MG, LW), dtype=f32, buffer=nl.psum)
                    for m in range(m0, m0 + MG):
                        nisa.nc_matmul(dst=pd[:, m - m0, :], stationary=dw[:, m, b0:b0 + 128].view(fp8),
                                       moving=aT[:, m, :], accumulate=False)
                    prod = nl.ndarray((128, MG, LW), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=prod, data1=pd, data2=sdt.ap(pattern=[[M * CT, 128], [CT, MG], [0, LW]],
                                                                        offset=m0 * CT + 4 * q + hh),
                                       op=nl.multiply, engine=nisa.vector_engine)
                    red = prod.ap(pattern=[[MG * LW, 128], [1, LW], [LW, MG]], offset=0)
                    if MG == M:
                        nisa.tensor_reduce(dst=yt, op=nl.add, data=red, axis=2)
                    elif m0 == 0:
                        ya = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
                        nisa.tensor_reduce(dst=ya, op=nl.add, data=red, axis=2)
                    else:
                        part = nl.ndarray((128, LW), dtype=f32, buffer=nl.sbuf)
                        nisa.tensor_reduce(dst=part, op=nl.add, data=red, axis=2)
                        if m0 + MG < M:
                            nisa.tensor_tensor(dst=ya, data1=ya, data2=part, op=nl.add, engine=nisa.vector_engine)
                        else:
                            nisa.tensor_tensor(dst=yt, data1=ya, data2=part, op=nl.add, engine=nisa.vector_engine)
                nisa.nc_transpose(dst=py[:, hh * 128:(hh + 1) * 128], data=yt, engine=nisa.tensor_engine)
            nisa.activation(dst=ys[:, q * 512:(q + 1) * 512], op=nl.copy, data=py, scale=w)
        return ys

    def _rmw_s(S, ts, ys, out):
        """A small pass's output rows added into out at its lanes' tokens (the scatter read-modify-write that ends
        _pass_s; kiln_moe_ep_small5 issues it after both passes of an iteration, so that the DMA queue does not wait
        on the first pass's last drain before the second pass's weights are requested)."""
        H, LW = S["H"], S["LW"]
        dst = out.ap(pattern=[[H, LW], [1, H]], offset=0, vector_offset=ts.ap(pattern=[[1, LW], [1, 1]], offset=0),
                     indirect_dim=0)
        nisa.dma_compute(dst=dst, srcs=[dst, ys], reduce_op=nl.add, oob_mode=oob_mode.skip)


    @nki.jit
    def kiln_moe_ep_small5(x, topi, wts, lmap, gu, dn, tsg, tsd, LW: int, act: int, lim: float, bc: int, rev: int):
        """kiln_moe_ep_small3's plan, pass kinds and arithmetic, its small passes two per device-loop iteration (table
        entries 2 it and 2 it + 1, _pass_s5 each, in program order, so the second expert's weights stream while the
        first one's down phase computes) and an odd last one in a second loop of trip count n % 2; the
        dequantize-first passes as v2's."""
        C, H = x.shape
        K = topi.shape[1]
        E = lmap.shape[1]
        El = gu.shape[0]
        M = gu.shape[2]
        CT = H // 128
        f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
        P = _plan_s2(topi, wts, lmap, C, K, E, El, LW)
        S = dict(H=H, M=M, CT=CT, LW=LW, gu=gu, tsg=tsg, dn=dn, tsd=tsd, act=act, lim=lim, x=x)
        SB = dict(H=H, M=M, CT=CT, LW=128, NS=1, gu=gu, sgu=tsg, dn=dn, sdn=tsd, sels=_consts(bc), act=act, lim=lim,
                  tsc=1)
        Pd = dict(NT=P["NT"], C=C, K=K, jrow=P["jrow_b"], ones1=P["ones1"], loc_h=P["loc_h"], wts=wts)
        out = nl.ndarray((C, H), dtype=x.dtype, buffer=nl.shared_hbm)
        zt = nl.ndarray((128, H), dtype=x.dtype, buffer=nl.sbuf)
        nisa.memset(dst=zt, value=0.0)
        for t in range(C // 128):
            nisa.dma_copy(dst=out[t * 128:(t + 1) * 128, :], src=zt)
        xra = nl.ndarray((LW, H), dtype=bf16, buffer=nl.sbuf)  # the pair loop's x rows (two sets), zeroed once
        nisa.memset(dst=xra, value=0.0)
        xrc = nl.ndarray((LW, H), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=xrc, value=0.0)
        xrb = nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=xrb, value=0.0)
        # trip counts: n // 2 pairs, then n % 2; the remainder is table entry n - 1
        n_s = P["tsm"]["n"]
        npr = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=npr, data=n_s, op0=nl.right_shift, operand0=1)
        n2 = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=n2, data=npr, op0=nl.left_shift, operand0=1)
        nrm = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=nrm, data1=n_s, data2=n2, op=nl.subtract, engine=nisa.vector_engine)
        ilast = nl.ndarray((1, 1), dtype=i32, buffer=nl.sbuf)  # n - 1 (only read when n is odd)
        nisa.tensor_scalar(dst=ilast, data=n2, op0=nl.add, operand0=0)
        # the pair tables: entries 0, 2, 4, .. and 1, 3, 5, .. of the small table, by strided HBM -> HBM copies (a loop
        # register cannot be stored to SBUF on trn1, NCC_IXCG832, so the loop indexes these directly)
        PA = (El + 1) // 2  # entries 0, 2, .. < El
        PC = El // 2  # entries 1, 3, .. < El
        tae = nl.ndarray((PA, 1), dtype=i32, buffer=nl.private_hbm)
        nisa.dma_copy(dst=tae, src=P["tsm"]["e"].ap(pattern=[[2, PA], [1, 1]], offset=0))
        taf = nl.ndarray((PA, 1), dtype=f32, buffer=nl.private_hbm)
        nisa.dma_copy(dst=taf, src=P["tsm"]["ef"].ap(pattern=[[2, PA], [1, 1]], offset=0))
        tce = nl.ndarray((PC, 1), dtype=i32, buffer=nl.private_hbm)
        nisa.dma_copy(dst=tce, src=P["tsm"]["e"].ap(pattern=[[2, PC], [1, 1]], offset=1))
        tcf = nl.ndarray((PC, 1), dtype=f32, buffer=nl.private_hbm)
        nisa.dma_copy(dst=tcf, src=P["tsm"]["ef"].ap(pattern=[[2, PC], [1, 1]], offset=1))
        tpa = dict(e=tae, ef=taf)
        tpc = dict(e=tce, ef=tcf)
        era = nisa.register_alloc()
        erc = nisa.register_alloc()
        rp = nisa.register_alloc()
        nisa.register_load(rp, npr)

        def body_p(it):
            ea, efa, nga = _table_pass(P, tpa, it, LW)
            nisa.register_load(era, ea)
            tsa = nl.ndarray((LW, 1), dtype=i32, buffer=nl.sbuf)
            _lanes_s(P, nga, 0, LW, tsa)
            wa = nl.ndarray((LW, 1), dtype=f32, buffer=nl.sbuf)
            _weights_s(P, tsa, efa, LW, wa)
            ec, efc, ngc = _table_pass(P, tpc, it, LW)
            nisa.register_load(erc, ec)
            tsc_ = nl.ndarray((LW, 1), dtype=i32, buffer=nl.sbuf)
            _lanes_s(P, ngc, 0, LW, tsc_)
            wc = nl.ndarray((LW, 1), dtype=f32, buffer=nl.sbuf)
            _weights_s(P, tsc_, efc, LW, wc)
            ysa = _pass_s5(S, era, tsa, wa, xra, out)
            ysc = _pass_s5(S, erc, tsc_, wc, xrc, out)
            _rmw_s(S, tsa, ysa, out)
            _rmw_s(S, tsc_, ysc, out)

        nl.fori_loop(0, rp, body_p)
        rr = nisa.register_alloc()
        nisa.register_load(rr, nrm)

        def body_r(it):
            e_sb, ef, ng = _table_pass(P, P["tsm"], ilast, LW)
            nisa.register_load(era, e_sb)
            ts = nl.ndarray((LW, 1), dtype=i32, buffer=nl.sbuf)
            _lanes_s(P, ng, 0, LW, ts)
            w = nl.ndarray((LW, 1), dtype=f32, buffer=nl.sbuf)
            _weights_s(P, ts, ef, LW, w)
            ys = _pass_s5(S, era, ts, w, xra, out)
            _rmw_s(S, ts, ys, out)

        nl.fori_loop(0, rr, body_r)
        rb = nisa.register_alloc()
        nisa.register_load(rb, P["tbg"]["n"])

        def body_b(it):
            e_sb, ef, ng = _table_pass(P, P["tbg"], it, 128)
            ts = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
            _lane_tokens(SB, Pd, ng, 0, 0, ts)
            w = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
            _lane_weights(SB, Pd, ts, ef, w)
            _pass(SB, dict(xkind="gather", ykind="rmw", ekind="dyn", e_sb=e_sb, x=x, ts=ts, xr=[xrb], w=w, out=out))

        nl.fori_loop(0, rb, body_b)
        return out

    # --- end of the small-lane kernel v5's source ---

else:
    kiln_moe_ep_core = kiln_moe_ep_kernel = kiln_moe_ep_small = kiln_moe_ep_small2 = kiln_moe_ep_small3 = kiln_moe_ep_small5 = None


def _kernel_rev(end: str = "    # --- end of the dequantize-first kernel's source") -> int:
    """CRC-32 of the kernel source from the start of the NKI block up to the marker `end` (LNL's cache key does not
    see NKI source: moe_prefill._kernel_rev). Each kernel hashes its own range, so editing one does not recompile
    every graph of the other."""
    import zlib

    src = open(__file__).read()
    a = src.index("if nki is not None:\n    DGU = ")
    b = src.index(end, a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev()
REV_SMALL = _kernel_rev("    # --- end of the small-lane kernel's source")
REV_SMALL2 = _kernel_rev("    # --- end of the small-lane kernel v2's source")
REV_SMALL3 = _kernel_rev("    # --- end of the small-lane kernel v3's source")
REV_SMALL5 = _kernel_rev("    # --- end of the small-lane kernel v5's source")
# The scale broadcast (kernel argument bc): 3 matmuls per 512-column scale row (hi, mid, lo selected and
# accumulated in PSUM, exact by construction) or 1 (one ones matmul over the three parts: exact only if the
# tensor engine's sum over its three partitions is, which tools/probe_moe_ep.py --bc 1 checks).
BCAST = int(os.environ.get("KILN_MOE_EP_BCAST", "1"))
# Lanes per pass (kernel argument LW): 128, or 256 from this many rows on (KILN_MOE_EP_LW forces one).
LW_ENV = os.environ.get("KILN_MOE_EP_LW")


def lanes(C: int) -> int:
    if LW_ENV:
        return int(LW_ENV)
    return 256 if C >= 2048 else 128


# Lanes per overflow pass of the dequantize-first kernel (kernel argument LW2): KILN_MOE_EP_LW2, else 512 for
# experts of at most 2048 intermediate rows (16 chunks: the pass's SBUF fits), the first pass's LW otherwise.
LW2_ENV = os.environ.get("KILN_MOE_EP_LW2")


def overflow_lanes(LW: int, M: int) -> int:
    if LW2_ENV:
        return int(LW2_ENV)
    return 512 if M <= 16 else LW


# Batches of at most this many rows (after padding to a multiple of 128) run kiln_moe_ep_small (decode, verify),
# with passes of SMALL_LW lanes; larger ones the dequantize-first kernel. KILN_MOE_EP_SMALL_ROWS=0 turns it off.
SMALL_ROWS = int(os.environ.get("KILN_MOE_EP_SMALL_ROWS", "128"))
SMALL_LW = int(os.environ.get("KILN_MOE_EP_SMALL_LW", "16"))
# Which small-lane kernel: 1 kiln_moe_ep_small (every local expert's first pass static, an expert with more than
# SMALL_LW pairs reloading its weights per overflow pass); 2 kiln_moe_ep_small2 (one pass per expert WITH pairs: a
# small-lane pass for at most SMALL_LW pairs, else one dequantize-first pass of 128 lanes; none for an expert without);
# 3 kiln_moe_ep_small3 (v2 with each small pass's weights in 12 large DMAs; tile-scale layouts only, else v2); 5
# kiln_moe_ep_small5 (v3's small passes two per loop iteration; tile-scale layouts only). v2, v3 and v5 give the same
# output bit for bit. Default 2 (2026-10-05, docs/neuron-notes.md "Decode v2 in serving": G64 122.8 -> 129.5 and F0
# 110.1 -> 117.1 out tok/s on trn1.32xlarge against v1; decode-path NLL change +0.0005 / -0.0008 nats per token on
# LONG_TEXT / wikitext-2, every flip at a near-tie: "The decode-path numerics gate").
SMALL_V_DEFAULT = 2
# KILN_MOE_EP_DGE=sw (experiment, kiln_moe_ep_kernel's dge): its gathers on software DGE.
DGE = os.environ.get("KILN_MOE_EP_DGE", "")
SMALL_V = int(os.environ.get("KILN_MOE_EP_SMALL_V", str(SMALL_V_DEFAULT)))


def uses_small(T: int) -> bool:
    return -(-T // P) * P <= SMALL_ROWS


def max_passes(C: int, K: int, El: int, LW: int) -> int:
    """The static pass bound: every routing of C tokens x K experts over El local experts fits, an expert with n
    pairs taking ceil(n / LW) passes and at most C min(K, El) pairs being local."""
    N = C * min(K, El)
    return (N + min(El, N) * (LW - 1)) // LW


def tile_scales(blob: dict) -> bool:
    return TILES and "tsg" in blob


def kernel_inputs(x, topv, topi, blob, lmap, act: int = 1, lim: float = 10.0, LW: int | None = None,
                  bc: int | None = None, LW2: int | None = None, tsc: bool | None = None) -> dict:
    """kiln_moe_ep_kernel's arguments for x [C, H] (C a multiple of 128), routing [C, K], the local experts'
    pack() blob and lmap int32 [1, E + 1] (local_map). tsc (default tile_scales(blob)): the tile-scale form."""
    C, K = topi.shape
    El = blob["gu"].shape[0]
    LW = LW or lanes(C)
    LW2 = LW2 or overflow_lanes(LW, blob["gu"].shape[2])
    PM = max_passes(C, K, El, LW2)
    tsc = tile_scales(blob) if tsc is None else tsc
    sg, sd = (blob["tsg"], blob["tsd"]) if tsc else (blob["sgu"], blob["sdn"])
    args = dict(x=x, topi=topi.to(torch.int32), wts=topv.to(torch.bfloat16), lmap=lmap.to(torch.int32),
                gu=blob["gu"], sgu=sg, dn=blob["dn"], sdn=sd, LW=LW, LW2=LW2, PMAX=PM,
                act=act, lim=float(lim), bc=BCAST if bc is None else bc, rev=REV, tsc=int(tsc))
    if split(blob["gu"].shape[2], x.shape[1]):  # only then: an unsplit call's arguments, and so its graph key, are as before
        args["spl"] = 1
    if DGE == "sw":
        args["dge"] = 1
    return args


def local_map(owner: torch.Tensor, rank: int) -> torch.Tensor:
    """lmap int32 [1, E + 1]: expert e's index among `rank`'s experts (in expert order) where owner[e] == rank,
    else the rank's expert count El (owner: [E] the rank of each expert); entry E (El everywhere) is the expert
    padding rows are routed to, which no rank holds."""
    mine = owner.cpu() == rank
    El = int(mine.sum())
    idx = torch.cumsum(mine.to(torch.int64), 0) - 1
    m = torch.where(mine, idx, torch.full_like(idx, El))
    return torch.cat([m, torch.full((1,), El, dtype=m.dtype, device="cpu")]).to(torch.int32).view(1, -1)


def grid() -> int:
    """The grid every kernel of this module is launched at: the runtime's LNC (platform.nki_grid: 1 on trn1, 2 on
    trn2). At LNC=2 the kernels see two programs (_lnc_setup): unsplit, both cores run the whole kernel and only
    program 0 writes out (each pass is a read-modify-write add, so a second writer would add every pair twice: what
    engine-v0's grid-2 launch did on trn2, 2026-10-05); split (split()), each core does half. Grid 1 at LNC=2 does not
    compile: neuronx-cc builds the kernel's device loops for core 0 only ("[NCC_IXGM002] Expected function sg0001 in
    subgraph 1 to have 9 basic blocks, but on core 1 it has 1 basic blocks")."""
    from .. import platform

    return platform.nki_grid()


def split(M: int | None = None, H: int | None = None) -> int:
    """1 when kiln_moe_ep_kernel / kiln_moe_ep_small2 / kiln_moe_ep_core split their work over the two physical cores
    of an LNC=2 logical core (kiln/platform.py KILN_LNC_SPLIT names moe_ep, and the runtime's LNC is 2): each program
    computes gate_up for half of the I-chunks and the down projection for half of the output columns, with the two
    halves of a^T swapped between the cores (_pass), the unsplit arithmetic bit for bit. Launched with grid 2 then.
    Part of the graph key (the kernels' `spl` argument). M (I-chunks) and H: the split needs both halves whole
    (M and H / 512 even); otherwise the call runs unsplit."""
    from .. import platform

    try:
        lnc = platform.nki_grid()
    except RuntimeError:  # no runtime configured (host tests, emulation): nothing is launched
        return 0
    if M is not None and (M % 2 or (H // 512) % 2):
        return 0
    return int(lnc == 2 and platform.lnc_split("moe_ep"))


def moe_ep(x, topv, topi, blob, lmap, act: int = 1, lim: float = 10.0) -> torch.Tensor:
    """This rank's share of the routed experts' output for x [T, H] inside the caller's graph (one kernel call;
    the rows padded to a multiple of 128, routed to expert E, lmap's no-rank entry, with weight 0)."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    T = x.shape[0]
    pad = -T % P
    if pad:
        x = torch.cat([x, x.new_zeros(pad, x.shape[1])])
        topi = torch.cat([topi, topi.new_full((pad, topi.shape[1]), lmap.shape[1] - 1)])
        topv = torch.cat([topv, topv.new_zeros(pad, topv.shape[1])])
    if uses_small(T) and SMALL_V in (3, 5) and tile_scales(blob):
        kern, rev = {3: (kiln_moe_ep_small3, REV_SMALL3), 5: (kiln_moe_ep_small5, REV_SMALL5)}[SMALL_V]
        if grid() != 1:
            raise ValueError(f"KILN_MOE_EP_SMALL_V={SMALL_V} has no LNC=2 form: use the default 2")
        out = wrap_nki(kern)[grid()](
            x=x, topi=topi.to(torch.int32), wts=topv.to(torch.bfloat16), lmap=lmap.to(torch.int32), gu=blob["gu"],
            dn=blob["dn"], tsg=blob["tsg"], tsd=blob["tsd"], LW=SMALL_LW, act=act, lim=float(lim), bc=BCAST, rev=rev)
    elif uses_small(T) and SMALL_V in (2, 3, 5):
        tsc = tile_scales(blob)
        sg, sd = (blob["tsg"], blob["tsd"]) if tsc else (blob["sgu"], blob["sdn"])
        spl = split(blob["gu"].shape[2], x.shape[1])
        out = wrap_nki(kiln_moe_ep_small2)[grid()](
            x=x, topi=topi.to(torch.int32), wts=topv.to(torch.bfloat16), lmap=lmap.to(torch.int32), gu=blob["gu"],
            dsg=blob["dsg"], dn=blob["dn"], dsd=blob["dsd"], sgu=sg, sdn=sd, LW=SMALL_LW, act=act, lim=float(lim),
            bc=BCAST, rev=REV_SMALL2, tsc=int(tsc), **({"spl": 1} if spl else {}))
    elif uses_small(T):
        C, K = topi.shape
        El = blob["gu"].shape[0]
        if grid() != 1:
            raise ValueError("KILN_MOE_EP_SMALL_V=1 has no LNC=2 form: use the default 2")
        out = wrap_nki(kiln_moe_ep_small)[grid()](
            x=x, topi=topi.to(torch.int32), wts=topv.to(torch.bfloat16), lmap=lmap.to(torch.int32), gu=blob["gu"],
            dsg=blob["dsg"], dn=blob["dn"], dsd=blob["dsd"], LW=SMALL_LW, PMAX=max_passes(C, K, El, SMALL_LW), act=act,
            lim=float(lim), rev=REV_SMALL)
    else:
        out = wrap_nki(kiln_moe_ep_kernel)[grid()](**kernel_inputs(x, topv, topi, blob, lmap, act, lim))
    return out[:T] if pad else out


def core(xe, ex, blob, LW: int, act: int = 1, lim: float = 10.0, bc: int | None = None, tsc: bool | None = None):
    """kiln_moe_ep_core inside the caller's graph."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    tsc = tile_scales(blob) if tsc is None else tsc
    sg, sd = (blob["tsg"], blob["tsd"]) if tsc else (blob["sgu"], blob["sdn"])
    spl = split(blob["gu"].shape[2], xe.shape[1])
    return wrap_nki(kiln_moe_ep_core)[grid()](
        xe=xe, ex=ex.to(torch.int32), gu=blob["gu"], sgu=sg, dn=blob["dn"], sdn=sd, LW=LW, act=act,
        lim=float(lim), bc=BCAST if bc is None else bc, rev=REV, tsc=int(tsc), **({"spl": 1} if spl else {}))
