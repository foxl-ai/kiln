"""Prefill MoE (C = 256 ... 8192 tokens of one chunk) as a grouped GEMM in one NKI kernel, for
NeuronCore-v2 (trn1), on the experts as kernels/moe_dedupe.py packs them ("tiles" blob), so the
decode and the prefill kernels share one copy of the experts.

Why not the per-pair or dedupe decode kernels: kernels/moe_decode.py loads one whole expert per
(token, expert) pair (about 4 us per pair on trn1, DMA-bound: 262 ms for the 65,536 pairs of an
8192-token chunk at top-8), and kernels/moe_dedupe.py holds a call's tokens on the partitions, so
it runs a chunk 128 tokens at a time and reads nearly every expert once per 128 tokens (64 times
for C=8192). With C x k pairs over E experts each expert serves C k / E tokens (228 at C=8192
for GLM-5.3-Flash), so here each expert is applied to all of its tokens as matmuls.

Plan (computed inside the kernel; plan() is its CPU reference): the pairs are ordered by expert and
each expert's pairs are padded to whole BLOCKS of B lanes (B = 64 or 128), so a block holds the
pairs of one expert; 128 / B consecutive blocks form a lane TILE of 128 lanes. The block count is
static, NB = (C k + E (B - 1)) // B rounded up to whole tiles, which every routing fits (an expert
with n pairs takes ceil(n / B) blocks): the kernel is exact for any routing and has no capacity
or overflow path, and blocks beyond the routing's own have expert E and skip their loads.
slot[t, k] = block * B + lane is where pair (t, k) lives, bexp[b] the expert of block b. Every
step is a 0/1 or small-integer matmul, a compare, an integer shift or mask, or a per-partition
gather (no sort, and one scatter of 4-byte token ids).

Kernel (kiln_moe_prefill_kernel), all control static:
0. The plan: each pair's rank among the earlier pairs of its expert (prefix counts on the tensor
   engine, exact in fp32), blocks per expert, a log-step prefix sum over the experts, slot, and
   each block's expert by a comparison matmul.
1. tos[lane, tile] = token of the slot (C for an empty lane), written by one indirect DMA per
   (token tile, k) (vector_offset, software DGE: 128 rows of 4 bytes each).
2. Per lane tile: each block's expert blob (one dynamic DMA, skipped for unused blocks); the tile's
   128 token rows of x gathered by one indirect DMA (empty lanes skip and keep finite older rows:
   the tensor-engine transposes sum over lanes, so a NaN in one lane would reach all); x transposed
   to [h, lane] tiles on the tensor engine; per block, the gate_up matmuls with the blob's [h, o]
   tiles as the stationary operand: either dequantized to bf16 in SBUF first (fp8 times the tile
   scale of the gate rows and of the up rows, each 64-column half one vector op with a per-partition
   scalar) and accumulated over the tiles of h in PSUM (`dq`: GLM-5.3-Flash at tp=32), or as stored
   with one PSUM partial per tile scaled per row o on the vector engine and summed in fp32 (any
   per-row tile scales: bf16 ones of MiMo-V2.6-Flash, fp32 ones of other ranks); g rounded to bf16
   (moe_dedupe's arithmetic); the up rows folded onto the gate rows by a 0/1 matmul (both halves of
   the partitions get both); a = glu(gate, up) (silu, or GLM-5.3-Flash's clamped silu) in fp32,
   rounded to bf16; y = a W_down on the fp8 down weights as stored (their two H halves on partitions
   0-63 and 64-127), both blocks of a tile into one 128-partition PSUM tile (each block's activation
   sits in its own 64 columns of a zero-padded stationary), drained with the down scale of each
   128-column chunk as a per-partition scale on the scalar engine (fp32 128 x 128 block scales as
   stored are constant over a chunk; where they are not, the loaded GLM-5.3-Flash experts after
   fit_e4m3_max and bf16 re-based MXFP4 scales, down_factors splits each column's scale into the
   chunk's smallest times a power of two, and the vector engine multiplies the drained bf16 product
   by that factor, broadcast onto each block's lanes by a 0/1 matmul: exact), and rounded to bf16,
   then stored to Y in slot order by a static DMA. The tiles are software-pipelined in four stages
   (A: x transposes and dequantization of tile i; B: gate_up of tile i - 1; F: fold and activation
   of tile i - 2; C: down, drain and store of tile i - 3), loads one or two tiles ahead, so every
   engine has work from several tiles in its in-order instruction stream and every hand-off between
   engines has other work in between.
3. Per token tile: its k slots' rows of Y gathered (indirect DMA) and summed on the tensor engine
   with diag(routing weight) as the stationary operand: out = sum_k w_k Y_k in fp32, rounded once.

Numerics: moe_dedupe.emulate(pair_bf16=True)'s; emulate() is it one expert at a time. gate_up
tile partials in fp32 times the tile scales (dq: the weights dequantized to bf16 first, as
models/quant.dequant), g rounded to bf16, a rounded to bf16, the down product times its column
scale rounded to bf16, the fp32 sum of weight x output over a token's k experts. Against the XLA
paths (dequantize to bf16 first) that is a rounding-order difference.
"""

from __future__ import annotations

import os

import torch

P = 128  # partitions


def block_size(C: int) -> int:
    """Lanes per block for a chunk of C tokens. At least 64: the lanes are the stationary operand of
    the down matmul, and neuronx-cc 2.27 rejects a 32-column stationary on trn1 ("[NCC_IBIR058]
    Matmult instruction does not support PE tile size 32x32 on CoreV2", 2026-10-03)."""
    return 64 if C <= 2048 else 128


def n_blocks(N: int, E: int, B: int) -> int:
    """Static block count every routing of N pairs over E experts fits, in whole tiles of 128 lanes."""
    G = P // B
    nb = (N + min(E, N) * (B - 1)) // B  # sum over experts of ceil(n_e / B), at most
    return -(-nb // G) * G


def blob_scale_bytes(blob: torch.Tensor, H: int) -> int:
    """2 (bf16 re-based MXFP4 tile scales) or 4 (fp32 128 x 128 block scales): moe_dedupe's layout."""
    from .moe_dedupe import scale_bytes

    return scale_bytes(blob.shape[-1], H)


def check_blob(blob: torch.Tensor, H: int) -> bool:
    """Whether the kernel takes its dequantize-first path (`dq`) on this tiles blob: fp32 tile
    scales (FP8 with 128 x 128 block scales) where the gate rows (0-63) share each gate_up tile scale
    and the up rows (64-127) theirs. The kernel runs every layout moe_dedupe.pack writes; the others
    take the per-row path (each tile's partial scaled per row o). Down scales: down_factors.

    The real GLM-5.3-Flash checkpoint at tp=32 has neither structure as the loader stores it,
    although its 128 x 128 blocks do: the rank's 64 gate rows, 64 up rows and 64 down input rows each
    sit inside one block, but models/quant.fit_e4m3_max (trn1's e4m3 stops at 240, the checkpoint's
    e4m3fn at 448) halves the codes of every (row, block) whose largest value exceeds 240 and doubles
    that row's scale: measured with tools/check_moe_prefill_layout.py, layer 3, rank 0, 2026-10-04,
    2.1% of (expert, tile) keep one gate scale, 17 of 9216 down chunks one scale (expert 0, columns
    0-127: 32 columns at 0.000174386, 96 at twice that)."""
    from .moe_dedupe import _offsets

    sb = blob_scale_bytes(blob, H)
    if sb == 2:
        return False
    o3 = _offsets(H, sb)[2]
    o2 = _offsets(H, sb)[1]
    sg = blob[..., o2:o3].contiguous().view(torch.float32)  # [E, 128 o, C]
    return bool(torch.equal(sg[:, :64], sg[:, :1].expand(-1, 64, -1))
                and torch.equal(sg[:, 64:], sg[:, 64:65].expand(-1, 64, -1)))


def down_factors(blob: torch.Tensor, H: int):
    """The down scales of a tiles blob for the kernel's per-column path, or None when it does not
    need it. Down scales are one per output column h (moe_dedupe's layout). Where they are constant
    over each 128-column chunk (fp32 128 x 128 block scales as stored in the checkpoint) the kernel
    reads them per chunk from the blob and this returns None. Otherwise (bf16 re-based MXFP4 scales,
    always; fp32 ones after models/quant.fit_e4m3_max doubled some columns' scales, as the loader
    leaves GLM-5.3-Flash's experts: check_blob's docstring) it returns (dsc fp32 [E, CT], dfr bf16
    [E, H]) with s[h] = dsc[chunk(h)] * dfr[h] exactly, dsc the smallest scale of each chunk in the
    kernel's chunk order (column 2 c' + half for h = half H / 2 + c' 128 + p) and dfr a power of two:
    the kernel drains each chunk times dsc and then multiplies the bf16 product by dfr, which is exact
    for a power of two. Raises if a column's scale is not a power of two times its chunk's smallest.
    Inspects the blob's values: call it at load (the decoder keeps the result), not in a graph."""
    from .moe_dedupe import unpack_down

    sb = blob_scale_bytes(blob, H)
    s = unpack_down(blob, H)[1][:, 0, :].float()  # [E, H], h in natural order
    E, CT, C2 = s.shape[0], H // P, H // 2 // P
    ch = s.view(E, CT, P)
    if sb == 4 and torch.equal(ch, ch[..., :1].expand_as(ch)):
        return None
    lo = ch.amin(-1, keepdim=True)
    f = ch / lo
    if not bool((lo > 0).all()) or not torch.equal(torch.exp2(torch.round(torch.log2(f))), f) or \
            not torch.equal(lo * f, ch):
        raise ValueError("KILN_MOE_PREFILL_KERNEL=nki: a down scale is not a power of two times its 128-column "
                         "chunk's smallest; the kernel's per-column path needs that to be exact")
    # The kernel multiplies the bf16 product by these factors (pc): exact for a power of two 2^j with
    # 0 <= j <= 16, far inside bf16's normal range.
    if not bool(((f >= 1) & (f <= 2.0 ** 16)).all()):
        raise ValueError("KILN_MOE_PREFILL_KERNEL=nki: a down column factor is outside [1, 2^16]")
    k = torch.arange(CT)  # chunk k = h // 128 of h = half H / 2 + c' 128 + p: half = k // C2, c' = k % C2
    dsc = torch.empty(E, CT)
    dsc[:, 2 * (k % C2) + k // C2] = lo[..., 0]
    return dsc.contiguous(), f.reshape(E, H).to(torch.bfloat16).contiguous()


# --- plan (torch, in the caller's graph) ----------------------------------------------------------


def _lower(n: int, device) -> torch.Tensor:
    """Strictly lower-triangular 0/1 [n, n] fp32: L[i, j] = [j < i]."""
    i = torch.arange(n, device=device, dtype=torch.float32)
    return (i.view(1, n) < i.view(n, 1)).to(torch.float32)


def _small_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b for a 0/1 matrix a and non-negative integers b < 2^18, exact even if the compiler casts
    matmul operands to bf16 (integers above 256 are not all bf16 values): b is split into three
    parts below 64, each exact in bf16, and the products are sums of at most 2^18 terms in fp32."""
    hi = torch.floor(b * (1.0 / 4096))
    r = b - hi * 4096
    mid = torch.floor(r * (1.0 / 64))
    lo = r - mid * 64
    return a @ lo + (a @ mid) * 64 + (a @ hi) * 4096


def _tile_order(topi: torch.Tensor) -> torch.Tensor:
    """Pairs [T, k] flattened in the kernel's order: token tile, then k, then the token."""
    T, K = topi.shape
    return topi.reshape(T // P, P, K).permute(0, 2, 1).reshape(T * K)


def _from_tile_order(v: torch.Tensor, T: int, K: int) -> torch.Tensor:
    return v.reshape(T // P, K, P).permute(0, 2, 1).reshape(T, K)


def plan(topi: torch.Tensor, E: int, B: int):
    """(slot [T, k] int32, tpos [T, k] int32, bexp [1, NB] int32) for experts topi [T, k] (T a
    multiple of 128), as the kernel computes them (this is its CPU reference): pair (t, j) lives in
    lane slot % B of block slot // B, i.e. lane slot % 128 of lane tile slot // 128; tpos =
    (slot % 128) * NTL + slot // 128 (NTL = NB B / 128 tiles) is its index in the kernel's tos table;
    bexp[b] is block b's expert, E for unused blocks. Pairs of one expert take consecutive lanes in
    the kernel's pair order: token tile, then j, then the token. Dense ops on [N, E] or smaller,
    exact in fp32."""
    T, K = topi.shape
    N = T * K
    NB = n_blocks(N, E, B)
    dev = topi.device
    f32 = torch.float32
    e = _tile_order(topi).to(f32)
    O = (e.view(N, 1) == torch.arange(E, device=dev, dtype=f32).view(1, E)).to(f32)  # [N, E] one-hot
    O3 = O.view(N // P, P, E)
    local = torch.einsum("pq,tqe->tpe", _lower(P, dev), O3)  # earlier pairs of the same group, per expert
    tot = O3.sum(1)  # [N / 128, E], each <= 128
    pre = _lower(N // P, dev) @ tot  # pairs of the same expert in earlier groups
    rank = ((local + pre.unsqueeze(1)) * O3).sum(-1).reshape(N)  # pairs before this one with its expert
    cnt = tot.sum(0)
    nblk = torch.floor((cnt + (B - 1)) * (1.0 / B))  # blocks per expert (B a power of two: exact)
    off = _small_matmul(_lower(E, dev), nblk.view(E, 1)).view(E)  # first block of each expert
    slot = (O * off.view(1, E)).sum(-1) * B + rank
    tile = torch.floor(slot * (1.0 / P))
    tpos = (slot - tile * P) * (NB * B // P) + tile
    incl = off + nblk  # one past each expert's last block
    bexp = (incl.view(1, E) <= torch.arange(NB, device=dev, dtype=f32).view(NB, 1)).to(f32).sum(-1)
    return (_from_tile_order(slot.to(torch.int32), T, K), _from_tile_order(tpos.to(torch.int32), T, K),
            bexp.to(torch.int32).view(1, NB))


def plan_reference(topi: torch.Tensor, E: int, B: int):
    """plan() written with loops (the CPU check of plan's arithmetic)."""
    T, K = topi.shape
    N = T * K
    NB = n_blocks(N, E, B)
    flat = _tile_order(topi).tolist()
    count = [0] * E
    rank = []
    for e in flat:
        rank.append(count[e])
        count[e] += 1
    off, b = [], 0
    for e in range(E):
        off.append(b)
        b += -(-count[e] // B)
    slot = torch.tensor([off[e] * B + r for e, r in zip(flat, rank)], dtype=torch.int32)
    NTL = NB * B // P
    tpos = (slot % P) * NTL + slot // P
    bexp = torch.full((NB,), E, dtype=torch.int32)
    for e in range(E):
        nb = -(-count[e] // B)
        bexp[off[e] : off[e] + nb] = e
    return _from_tile_order(slot, T, K), _from_tile_order(tpos, T, K), bexp.view(1, NB)


# --- kernel ---------------------------------------------------------------------------------------

try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from nki.isa.constants import oob_mode
except ImportError:
    nki = None

# The dequantize-first path's half tiles (j * 2 + gate / up of each round's four tiles) in the order
# they move to the scalar engine as asp grows (spread over the round's tiles and both halves).
ACT_ORDER = (7, 2, 5, 0, 3, 6, 1, 4)

# Defined at import, not inside the traced code (see kernels/moe_decode.py).
if nki is not None:
    @nki.jit
    def kiln_moe_prefill_kernel(x, topi, wts, blob, dsc, dfr, fold_g, fold_u, B: int, NB: int, act: int,
                                lim: float, dq: int, pc: int, asp: int, skp: int, ord_: int, nyb: int, rev: int,
                                spl: int = 1, inb: int = 0, nsx: int = 0, bng: int = 0):
        """x bf16 [C, H]; topi int32 [C, K] the experts of each token and wts bf16 [C, K] their routing
        weights; blob uint8 [E, 128, F] (moe_dedupe tiles layout, fp32 or bf16 tile scales); dsc fp32
        [E, CT] and dfr bf16 [E, H]: the down scales as s[h] = dsc[chunk] * dfr[h] (down_factors; dfr
        powers of two), read when pc = 1; fold_g / fold_u
        bf16 [128, 128] = [o == q % 64] / [o == 64 + q % 64]; B lanes per block (64 or 128) and NB =
        n_blocks(C K, E, B); act 0 silu, 1 silu with gate <= lim and |up| <= lim (moe_dedupe.ACTS);
        dq 1: the blob's gate rows share each tile scale and so do its up rows (check_blob), so the
        gate_up tiles are dequantized to bf16 in SBUF and accumulated in PSUM; dq 0: each tile's
        partial is scaled per row on the vector engine; pc 1: down scales per output column (dsc, dfr:
        the drain applies dsc per 128-column chunk, then the bf16 product is multiplied by dfr, exact for
        a power of two), 0: one per 128-column chunk, read from the blob; asp: dq only, how many of
        the eight half-tile dequantizations of each block and round run on the scalar engine instead
        of the vector engine (ACT_ORDER; the same fp32 product rounded once to bf16); rev: this
        function's source revision (REV, unused here: see _kernel_rev); skp: 0 runs every static lane
        tile, n > 0 runs the tiles past the first n in chunks of n that are skipped once the routing
        has no block left in them (device loops of trip count 0 or 1); ord_: 1 issues each pipeline round's
        stage B (gate_up matmuls and their scaling) before its stage C down matmuls (_stage_b), 0 after;
        nyb: stage C's down outputs in a ring of nyb buffers (0: one allocation per step). Neither
        changes an operation or its order on any accumulator: the output is the same bit for bit.
        Returns bf16 [C, H]."""
        C, H = x.shape
        K = topi.shape[1]
        E = blob.shape[0]
        G = 128 // B
        NTL = NB // G
        F = blob.shape[2]
        DW = H // 2
        CT = H // 128
        SB = (F - H - DW) // (2 * CT)
        o_sg = H + DW
        o_sd = o_sg + CT * SB
        MX = 1 if SB == 2 else 0  # bf16 tile scales (re-based MXFP4): per-row gate_up scales in bf16
        PC = pc  # down scales per output column: dsc per chunk times dfr per column (always for MX)
        NT = C // 128
        fp8 = nl.float8_e4m3
        f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
        # LNC (trn2 at LNC=2, grid 2): the kernel is traced once per program (the two physical cores of
        # the logical core; nki/_backends/mlir_tracer program_id: "kernel is traced LNC times with
        # different program_id_value"), so npg / pid are Python ints. Each program runs half of the lane
        # tiles and half of the combine's token tiles; the plan is computed by both. With grid 1 (trn1)
        # npg = 1 and every step is the single-core kernel's.
        # spl 0 (KILN_LNC_SPLIT): both programs do all of the work, as before the split.
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if spl and nl.program_ndim() != 0 else (1, 0)
        if npg > 1 and inb:  # KILN_MOE_PREFILL_INBAR (experiment): both programs at the kernel's start before either
            # reads its inputs (an input written by XLA ops on the other physical core)
            nisa.core_barrier(data=topi, cores=(0, 1))
            nisa.core_barrier(data=x, cores=(0, 1))
            nisa.core_barrier(data=wts, cores=(0, 1))

        fg = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=fg, src=fold_g)
        fu = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=fu, src=fold_u)
        ident = nl.shared_identity_matrix(n=128, dtype=bf16)

        # 0. The plan (plan() is its CPU reference), all integers exact in fp32: a pair's rank among
        # the earlier pairs of its expert in the order (token tile, k, token), then its block, lane,
        # slot and tos position, and each block's expert. Constants first.
        io = nl.ndarray((128, E), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=io, pattern=[[1, E]], offset=0, channel_multiplier=0)
        iof = nl.ndarray((128, E), dtype=f32, buffer=nl.sbuf)  # e on every partition
        nisa.tensor_copy(dst=iof, src=io, engine=nisa.vector_engine)
        uu = nl.ndarray((128, 128), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=uu, pattern=[[1, 128]], offset=0, channel_multiplier=-1)  # q - p
        uuf = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=uuf, src=uu, engine=nisa.vector_engine)
        lt = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)  # [p < q]: earlier tokens of a group
        nisa.tensor_scalar(dst=lt, data=uuf, op0=nl.greater, operand0=0.0, engine=nisa.vector_engine)
        r64 = nl.ndarray((1, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=r64, value=64.0)
        r1 = nl.ndarray((1, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=r1, value=1.0)
        on64 = nl.ndarray((128, 64), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=on64, value=1.0)
        tk = nl.ndarray((128, NT, K), dtype=i32, buffer=nl.sbuf)  # token T * 128 + p
        nisa.dma_copy(dst=tk, src=topi.ap(pattern=[[K, 128], [128 * K, NT], [1, K]], offset=0))
        tkf = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tkf, src=tk, engine=nisa.vector_engine)
        # Earlier groups' one-hots, summed per partition (token), in ranges of 256 groups so that every
        # entry is an integer <= 256 (bf16-exact); a ones matmul sums them over the partitions into
        # each expert's count of earlier pairs, on every partition at once.
        NG = NT * K
        on128 = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=on128, value=1.0)
        srs = []
        for r in range(-(-NG // 256)):
            s0 = nl.ndarray((128, E), dtype=bf16, buffer=nl.sbuf)
            nisa.memset(dst=s0, value=0.0)
            srs.append(s0)
        rk = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)
        for T in range(NT):
            for k in range(K):
                gi = T * K + k
                oh = nl.ndarray((128, E), dtype=bf16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=oh, data=iof, op0=nl.equal, operand0=tkf[:, T, k:k + 1],
                                   engine=nisa.vector_engine)
                pp = nl.ndarray((128, E), dtype=f32, buffer=nl.psum)
                nisa.nc_matmul(dst=pp, stationary=lt, moving=oh, accumulate=False)
                for r in range(-(-gi // 256)):  # ranges holding earlier groups
                    nisa.nc_matmul(dst=pp, stationary=on128, moving=srs[r], accumulate=True)
                tmp = nl.ndarray((128, E), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=tmp, data1=oh, data2=pp, op=nl.multiply, engine=nisa.vector_engine)
                nisa.tensor_reduce(dst=rk[:, T, k:k + 1], op=nl.add, data=tmp, axis=1)
                nisa.tensor_tensor(dst=srs[gi // 256], data1=srs[gi // 256], data2=oh, op=nl.add,
                                   engine=nisa.vector_engine)
        pc = nl.ndarray((64, E), dtype=f32, buffer=nl.psum)  # pairs per expert (64 equal rows)
        for r in range(len(srs)):
            nisa.nc_matmul(dst=pc, stationary=on64, moving=srs[r], accumulate=(r > 0))
        cnt = nl.ndarray((1, E), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=cnt, src=pc[0:1, :], engine=nisa.vector_engine)
        # Blocks per expert, ceil(cnt / B), and the first block of each: an exclusive prefix sum over
        # the experts on one partition, in log2(E) shifted adds.
        LB = 7 if B == 128 else 6  # B = 2 ** LB (the tracer takes no int methods)
        ci = nl.ndarray((1, E), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ci, data=cnt, op0=nl.add, operand0=float(B - 1), engine=nisa.vector_engine)
        nbi = nl.ndarray((1, E), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=nbi, data=ci, op0=nl.right_shift, operand0=LB)
        nbk = nl.ndarray((1, E), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=nbk, src=nbi, engine=nisa.vector_engine)
        inc = nl.ndarray((1, E), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=inc, src=nbk, engine=nisa.vector_engine)
        sh = 1
        while sh < E:
            prev = nl.ndarray((1, E), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=prev, src=inc, engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=inc[:, sh:E], data1=prev[:, sh:E], data2=prev[:, 0:E - sh], op=nl.add,
                               engine=nisa.vector_engine)
            sh *= 2
        off = nl.ndarray((1, E), dtype=f32, buffer=nl.sbuf)  # exclusive prefix; < 2^12
        nisa.tensor_tensor(dst=off, data1=inc, data2=nbk, op=nl.subtract, engine=nisa.vector_engine)
        # off on every partition, exactly: 64 hi + lo with both parts bf16-exact
        oi = nl.ndarray((1, E), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=oi, src=off, engine=nisa.vector_engine)
        os_ = nl.ndarray((1, E), dtype=i32, buffer=nl.sbuf)
        olob = nl.ndarray((1, E), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=os_, data=oi, op0=nl.bitwise_and, operand0=63)
        nisa.tensor_copy(dst=olob, src=os_, engine=nisa.vector_engine)
        ohib = nl.ndarray((1, E), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=os_, data=oi, op0=nl.right_shift, operand0=6)
        nisa.tensor_copy(dst=ohib, src=os_, engine=nisa.vector_engine)
        pob = nl.ndarray((128, E), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pob, stationary=r64, moving=ohib, accumulate=False)
        nisa.nc_matmul(dst=pob, stationary=r1, moving=olob, accumulate=True)
        offb = nl.ndarray((128, E), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=offb, src=pob, engine=nisa.vector_engine)
        # Each pair's block offset off[e], gathered inside each partition (GpSimd); then slot, tile and
        # tos index, all at once.
        tku = nl.ndarray((128, NT, K), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tku, src=tk, engine=nisa.vector_engine)
        bo = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=bo, data=offb, indices=tku)
        sf = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)  # slot = off B + rank
        nisa.scalar_tensor_tensor(dst=sf, data=bo, op0=nl.multiply, operand0=float(B), op1=nl.add, operand1=rk)
        st = nl.ndarray((128, NT, K), dtype=i32, buffer=nl.sbuf)  # the combine's gather rows
        nisa.tensor_copy(dst=st, src=sf, engine=nisa.vector_engine)
        li = nl.ndarray((128, NT, K), dtype=i32, buffer=nl.sbuf)  # tos index: (slot & 127) NTL + (slot >> 7)
        nisa.tensor_scalar(dst=li, data=st, op0=nl.bitwise_and, operand0=127)
        lf = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lf, src=li, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=li, data=st, op0=nl.right_shift, operand0=7)
        tf = nl.ndarray((128, NT, K), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tf, src=li, engine=nisa.vector_engine)
        nisa.scalar_tensor_tensor(dst=tf, data=lf, op0=nl.multiply, operand0=float(NTL), op1=nl.add, operand1=tf)
        tp = nl.ndarray((128, NT, K), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tp, src=tf, engine=nisa.vector_engine)
        # Block b's expert: the number of experts whose blocks end at or before b (E past the
        # routing's blocks). incl = off + nblk goes onto the partitions (exactly, 64 hi + lo through
        # two bf16 transposes), each compared with b on the free axis and summed by a matmul.
        # The row is padded to EP = 64-multiple experts (a transpose takes 64 or more columns on trn1)
        # with 16383 = 64 * 255 + 63 (bf16-exact parts, past every block), which counts for no block.
        EP = -(-E // 64) * 64
        incl = nl.ndarray((1, EP), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=incl, value=16383.0)
        nisa.tensor_tensor(dst=incl[:, 0:E], data1=off, data2=nbk, op=nl.add, engine=nisa.vector_engine)
        ii = nl.ndarray((1, EP), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ii, src=incl, engine=nisa.vector_engine)
        is_ = nl.ndarray((1, EP), dtype=i32, buffer=nl.sbuf)
        ilob = nl.ndarray((1, EP), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=is_, data=ii, op0=nl.bitwise_and, operand0=63)
        nisa.tensor_copy(dst=ilob, src=is_, engine=nisa.vector_engine)
        ihib = nl.ndarray((1, EP), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=is_, data=ii, op0=nl.right_shift, operand0=6)
        nisa.tensor_copy(dst=ihib, src=is_, engine=nisa.vector_engine)
        one1 = nl.ndarray((1, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=one1, value=1.0)
        ib = nl.ndarray((128, NB), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ib, pattern=[[1, NB]], offset=0, channel_multiplier=0)  # b on every partition
        ibf = nl.ndarray((128, NB), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ibf, src=ib, engine=nisa.vector_engine)
        cmps = []
        for e0 in range(0, EP, 128):  # chunks of 128 experts on the partitions, the last 64 or more
            ec = min(128, EP - e0)
            pt = nl.ndarray((ec, 2), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pt[:, 0:1], stationary=ihib[:, e0:e0 + ec], moving=one1)  # incl hi, on partitions
            nisa.nc_matmul(dst=pt[:, 1:2], stationary=ilob[:, e0:e0 + ec], moving=one1)
            pts = nl.ndarray((ec, 2), dtype=f32, buffer=nl.sbuf)  # (the vector engine reads one PSUM operand)
            nisa.tensor_copy(dst=pts, src=pt, engine=nisa.vector_engine)
            icl = nl.ndarray((ec, 1), dtype=f32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=icl, data=pts[:, 0:1], op0=nl.multiply, operand0=64.0, op1=nl.add,
                                      operand1=pts[:, 1:2])
            cm = nl.ndarray((ec, NB), dtype=bf16, buffer=nl.sbuf)  # [incl_e <= b]
            nisa.tensor_scalar(dst=cm, data=ibf[0:ec, :], op0=nl.greater_equal, operand0=icl,
                               engine=nisa.vector_engine)
            cmps.append((ec, cm))
        bef = nl.ndarray((1, NB), dtype=f32, buffer=nl.sbuf)
        for b0 in range(0, NB, 512):
            nbc = min(512, NB - b0)
            pbx = nl.ndarray((64, nbc), dtype=f32, buffer=nl.psum)
            for ci in range(len(cmps)):
                ec, cm = cmps[ci]
                nisa.nc_matmul(dst=pbx, stationary=on64[0:ec, :], moving=cm[:, b0:b0 + nbc], accumulate=(ci > 0))
            nisa.tensor_copy(dst=bef[:, b0:b0 + nbc], src=pbx[0:1, :], engine=nisa.vector_engine)
        be = nl.ndarray((1, NB), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=be, src=bef, engine=nisa.vector_engine)

        # 1. tos[lane, tile] = token of the slot, C (out of range: skipped) where empty.
        tos = nl.ndarray((128 * NTL, 1), dtype=i32, buffer=nl.private_hbm)
        fill = nl.ndarray((128, NTL), dtype=i32, buffer=nl.sbuf)
        # nsx (KILN_MOE_PREFILL_NOSKIPX, experiment): empty lanes gather token 0 instead of skipping an
        # out-of-range row C (their Y rows are never read)
        nisa.memset(dst=fill, value=0 if nsx else C)
        nisa.dma_copy(dst=tos.ap(pattern=[[NTL, 128], [1, NTL]], offset=0), src=fill)
        tid = nl.ndarray((128, NT), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=tid, pattern=[[128, NT]], offset=0, channel_multiplier=1)
        for t in range(NT):
            for j in range(K):
                nisa.dma_copy(dst=tos.ap(pattern=[[1, 128], [1, 1]], offset=0,
                                         vector_offset=tp.ap(pattern=[[NT * K, 128], [1, 1]], offset=t * K + j),
                                         indirect_dim=0),
                              src=tid[:, t:t + 1])
        ts = nl.ndarray((128, NTL), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ts, src=tos.ap(pattern=[[NTL, 128], [1, NTL]], offset=0))

        # 2. Lane tiles, software-pipelined in four stages so that every engine has independent work
        # from several tiles in its in-order instruction stream, and every hand-off between engines
        # has work of other tiles between producer and consumer: step i issues the loads of tile
        # i + PF (expert blobs, x rows); then, interleaved in eight rounds, one 512-column chunk of
        # the down matmuls and scaled drains of tile i - 3 (stage C), four x transposes and their
        # gate_up dequantization of tile i (stage A), four gate_up matmuls of tile i - 1 (stage B),
        # and spread over rounds 0, 1, 2, 3 and 5 the fold and activation of tile i - 2 (stage F:
        # vector copy of its gate_up sums, fold matmuls, clamp, SiLU, product); then the store of
        # tile i - 3. Rings hold each stage's inputs until the last stage that reads them: expert
        # buffers PF + 4 tiles (i + PF .. i - 3), x rows PF + 1, x^T, dequantized tiles and activations 2. The x
        # rows ring is zeroed once, so a skipped (empty) lane holds finite values; the activation
        # stationaries are zero outside their block's own columns.
        # Y in slot order. With two programs each writes the rows of its own lane tiles and the combine
        # reads any row, so Y is in HBM both cores share, and a core barrier (nisa.core_barrier: "two
        # NeuronCores both need to write to disjoint portions of a shared HBM tensor ... and they both
        # need to consume the tensor after both cores have finished", nki/isa/_lnc.py) separates the
        # stores of stage 2 from the gathers of stage 3.
        Y = nl.ndarray((NTL * 128, H), dtype=bf16, buffer=nl.private_hbm if npg == 1 else nl.shared_hbm)
        PF = 2 if G == 1 else 1  # loads run PF tiles ahead (SBUF: one block of 128 lanes per tile, or two)
        NW = PF + 4
        NX = PF + 1
        wr, xr, xtr, ar, tbr, wqr = [], [], [], [], [], []
        for r in range(NW):
            row = []
            for g in range(G):
                row.append(nl.ndarray((128, F), dtype=nl.uint8, buffer=nl.sbuf))
            wr.append(row)
        for r in range(NX):
            row2 = []
            for g in range(G):
                row2.append(nl.ndarray((128, 2, CT * 4), dtype=nl.uint8, buffer=nl.sbuf))
            tbr.append(row2)  # dq: the gate and the up rows' tile scales, on every partition
            xb0 = nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf)
            nisa.memset(dst=xb0, value=0.0)
            xr.append(xb0)
        for r in range(2):
            xtr.append(nl.ndarray((128, CT, 128), dtype=bf16, buffer=nl.sbuf))
            row2 = []
            for g in range(G if dq else 0):  # the per-row path never reads them: no SBUF for it
                row2.append(nl.ndarray((128, CT, 128), dtype=bf16, buffer=nl.sbuf))
            wqr.append(row2)  # dq: dequantized gate_up tiles [h, c, o]
            row = []
            for g in range(G):
                a0 = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
                nisa.memset(dst=a0, value=0.0)
                row.append(a0)
            ar.append(row)

        scr, frr = [], []  # PC: each block's chunk scales on its lanes' partitions, its column factors as
        if PC:  # row g; stage C. selg [G, 128]: row g ones over block g's lanes (broadcasts row g there)
            for r in range(2):
                scr.append(nl.ndarray((128, CT), dtype=f32, buffer=nl.sbuf))
                frr.append(nl.ndarray((G, H), dtype=bf16, buffer=nl.sbuf))
            sgi = nl.ndarray((G, 128), dtype=i32, buffer=nl.sbuf)
            nisa.iota(dst=sgi, pattern=[[1, 128]], offset=0, channel_multiplier=-B)  # lane - B g
            sgf = nl.ndarray((G, 128), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=sgf, src=sgi, engine=nisa.vector_engine)
            sga = nl.ndarray((G, 128), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=sga, data=sgf, op0=nl.greater_equal, operand0=0.0, engine=nisa.vector_engine)
            sgb = nl.ndarray((G, 128), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=sgb, data=sgf, op0=nl.less, operand0=float(B), engine=nisa.vector_engine)
            selg = nl.ndarray((G, 128), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=selg, data1=sga, data2=sgb, op=nl.multiply, engine=nisa.vector_engine)
        ybr = []  # nyb > 0: a ring of stage C's bf16 down outputs (measured: 3 is faster than one allocation
        for r in range(nyb):  # per step, 2 slower; see KILN_MOE_PREFILL_NYB)
            ybr.append(nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf))
        accr = []  # each block's gate_up sums of tile t at accr[t % 2] (PSUM for dq), stage B to F
        for r in range(2):
            row = []
            for g in range(G):
                if dq:
                    row.append(nl.ndarray((128, B), dtype=f32, buffer=nl.psum))
                else:
                    row.append(nl.ndarray((128, B), dtype=f32, buffer=nl.sbuf))
            accr.append(row)
        NCH = 2 * (DW // 512)  # 512-column chunks of the down output
        NR = max(CT // 4, NCH)
        r1, r2, r3, r5 = min(1, NR - 1), min(2, NR - 1), min(3, NR - 1), min(5, NR - 1)  # stage F's rounds
        S = dict(NTL=NTL, PF=PF, G=G, NB=NB, F=F, NW=NW, NX=NX, be=be, blob=blob, wr=wr, tbr=tbr, o_sg=o_sg,
                 o_sd=o_sd, CT=CT, x=x, H=H, xr=xr, ts=ts, PC=PC, scr=scr, frr=frr, dsc=dsc, dfr=dfr, B=B, f32=f32,
                 bf16=bf16, fp8=fp8, MX=MX, NR=NR, NCH=NCH, DW=DW, r1=r1, r2=r2, r3=r3, r5=r5, ar=ar,
                 xtr=xtr, wqr=wqr, fg=fg, fu=fu, act=act, lim=lim, selg=selg if PC else None, asp=asp, dq=dq, Y=Y,
                 bfirst=ord_ == 1, ybr=ybr)
        T1 = NTL - skp * ((NTL - (NTL + 1) // 2) // skp) if skp > 0 else NTL  # tiles that always run
        # This program's share of the tiles that always run: all of them (one program), else the first or
        # the second half (every tile costs the same, empty ones included).
        t_lo, t_hi = (0, T1) if npg == 1 else ((0, (T1 + 1) // 2) if pid == 0 else ((T1 + 1) // 2, T1))
        for i in range(t_lo - PF, t_hi + 3):
            _tile_step(S, accr, i, t_lo, t_hi)
        if T1 < NTL:
            # The tiles past T1 in segments of skp, each its own pipeline (prologue to epilogue) in a device
            # loop of trip count f[s] = [first tile of segment s < U / G], U the routing's block count
            # (the last entry of the inclusive prefix sum `inc`; blocks are laid out by expert, the rest
            # have expert E): segments wholly past the routing's blocks do not run. Device loops with
            # static addresses run on trn1 ("MoE that reads each selected expert once per call"); a PSUM
            # tile referenced in two device-loop regions fails to compile ([NCC_IBIR092] Live-in/Live-out
            # MemoryLocation ... not allocated to MemoryType: DRAM, 2026-10-04), so each region gets its
            # own gate_up accumulators.
            KS = (NTL - T1) // skp
            # With two programs every program runs every segment's device loop on its own half of the
            # segment's tiles (s_off .. s_off + s_n of it) with the SAME trip count, read at the segment's
            # first tile: the LNC=2 compile requires both cores' programs to have the same basic blocks
            # ("[NCC_IXGM002] Expected function ... to have X basic blocks, but on core 1 it has Y",
            # measured 2026-10-04 when the segments alternated between the programs), and trip counts that
            # differ between the cores (each read at its own half) hung the device: execution timeout,
            # GpSimd waiting (C=4096, KILN_MOE_PREFILL_SKIP=20, kiln-trn2-b 2026-10-04).
            s_off, s_n = (0, skp) if npg == 1 else ((0, (skp + 1) // 2) if pid == 0 else ((skp + 1) // 2, skp // 2))
            kio = nl.ndarray((1, KS), dtype=i32, buffer=nl.sbuf)
            nisa.iota(dst=kio, pattern=[[skp * G, KS]], offset=T1 * G, channel_multiplier=0)
            kif = nl.ndarray((1, KS), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=kif, src=kio, engine=nisa.vector_engine)
            fk = nl.ndarray((1, KS), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=fk, data=kif, op0=nl.less, operand0=inc[:, E - 1:E], engine=nisa.vector_engine)
            fli = nl.ndarray((1, KS), dtype=i32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=fli, src=fk, engine=nisa.vector_engine)
            for k in range(KS):
                lo = T1 + k * skp + s_off
                rk = nisa.register_alloc()
                nisa.register_load(rk, fli.ap(pattern=[[KS, 1], [1, 1]], offset=k))

                def segment(it, lo=lo):
                    _segment(S, lo, lo + s_n, it)

                nl.fori_loop(0, rk, segment)

        # 3. Combine: out[t] = sum_k w[t, k] Y[slot[t, k]], fp32 on the tensor engine; with two programs
        # each takes half of the token tiles, after both have stored their Y rows.
        out = nl.ndarray((C, H), dtype=x.dtype, buffer=nl.shared_hbm)
        # bng (KILN_MOE_PREFILL_BARENG, experiment): the engine the barriers run on, 0 GpSimd (the default), 1 sync, 2 vector
        beng = (nisa.engine.gpsimd, nisa.engine.sync, nisa.engine.vector)[bng]
        if npg > 1:
            nisa.core_barrier(data=Y, cores=(0, 1), engine=beng)
        c_lo, c_hi = (0, NT) if npg == 1 else ((0, (NT + 1) // 2) if pid == 0 else ((NT + 1) // 2, NT))
        for t in range(c_lo, c_hi):
            wv = nl.ndarray((128, K), dtype=f32, buffer=nl.sbuf)
            nisa.dma_copy(dst=wv, src=wts[t * 128:(t + 1) * 128, :])
            yk, dg = [], []
            for j in range(K):
                yj = nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf)
                nisa.dma_copy(dst=yj, src=Y.ap(pattern=[[H, 128], [1, H]], offset=0,
                                               vector_offset=st.ap(pattern=[[NT * K, 128], [1, 1]], offset=t * K + j),
                                               indirect_dim=0))
                yk.append(yj)
                dj = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=dj, data=ident, op0=nl.multiply, operand0=wv[:, j:j + 1],
                                   engine=nisa.vector_engine)
                dg.append(dj)
            ob = nl.ndarray((128, H), dtype=x.dtype, buffer=nl.sbuf)
            for ch in range(H // 512):
                po = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
                for j in range(K):
                    nisa.nc_matmul(dst=po, stationary=dg[j], moving=yk[j][:, ch * 512:(ch + 1) * 512],
                                   accumulate=(j > 0))
                nisa.activation(dst=ob[:, ch * 512:(ch + 1) * 512], op=nl.copy, data=po)
            nisa.dma_copy(dst=out[t * 128:(t + 1) * 128, :], src=ob)
        if npg > 1:
            # LNC split: every program has written its share of the output before either program's kernel ends,
            # so that whatever runs next on either physical core reads the whole output (nisa.core_barrier on the
            # shared HBM output, nki/isa/_lnc.py).
            nisa.core_barrier(data=out, cores=(0, 1), engine=beng)
        return out

    def _segment(S, lo, hi, it):
        """Lane tiles lo .. hi - 1 as a pipeline of their own (_tile_step), with gate_up accumulators of
        its own: the body of a device loop (it: its induction register, unused)."""
        accs = []
        for r in range(2):
            row = []
            for g in range(S["G"]):
                row.append(nl.ndarray((128, S["B"]), dtype=S["f32"], buffer=nl.psum if S["dq"] else nl.sbuf))
            accs.append(row)
        for i in range(lo - S["PF"], hi + 3):
            _tile_step(S, accs, i, lo, hi)

    def _stage_b(S, acc, sgs, tb, c4):
        """Stage B of tile tb, round c4: four gate_up tiles of each block, accumulated in PSUM (dq) or
        each tile's partial scaled per row and summed in fp32 on the vector engine. With ord_ 1 the
        rounds issue it before stage C's down matmuls, so that the tensor engine's in-order stream does
        not hold the partials the vector engine waits for behind matmuls that wait on the vector engine
        (stage C reads stage F's activations): C=4096 9.725 -> 9.551 ms on GLM-5.3-Flash's real experts
        (trn1, 2026-10-04, docs/neuron-notes.md), the output the same bit for bit."""
        G = S["G"]
        B = S["B"]
        f32 = S["f32"]
        fp8 = S["fp8"]
        NW = S["NW"]
        wr = S["wr"]
        wqr = S["wqr"]
        xtr = S["xtr"]
        if S["dq"]:  # four dequantized tiles
            for g in range(G):
                for j in range(4):
                    c = c4 * 4 + j
                    nisa.nc_matmul(dst=acc[g], stationary=wqr[tb % 2][g][:, c, :],
                                   moving=xtr[tb % 2][:, c, g * B:(g + 1) * B], accumulate=(c > 0))
            return
        for g in range(G):  # four tiles per block, scaled
            wf = wr[tb % NW][g].view(fp8)
            sg = sgs[g]  # [128 o, CT] tile scales
            pg = nl.ndarray((128, 4, B), dtype=f32, buffer=nl.psum)
            for j in range(4):  # accumulate=False: the default (None) accumulates onto a reused tile
                c = c4 * 4 + j
                nisa.nc_matmul(dst=pg[:, j, :], stationary=wf[:, c * 128:(c + 1) * 128],
                               moving=xtr[tb % 2][:, c, g * B:(g + 1) * B], accumulate=False)
            for j in range(4):
                c = c4 * 4 + j
                if c == 0:
                    nisa.tensor_scalar(dst=acc[g], data=pg[:, j, :], op0=nl.multiply,
                                       operand0=sg[:, 0:1], engine=nisa.vector_engine)
                else:
                    nisa.scalar_tensor_tensor(dst=acc[g], data=pg[:, j, :], op0=nl.multiply,
                                              operand0=sg[:, c:c + 1], op1=nl.add, operand1=acc[g])

    def _tile_step(S, accr, i, lo, hi):
        """Step i of the lane-tile pipeline over tiles lo .. hi - 1 (kiln_moe_prefill_kernel, part 2):
        the loads of tile i + PF, then the stages of tiles i (A), i - 1 (B), i - 2 (F) and i - 3 (C),
        and the store of tile i - 3, each only for a tile in the range. S: the kernel's tensors and
        constants by name; accr: the gate_up accumulators' ring. A module-level helper, so that the
        same step can be issued inside and outside a device loop."""
        NTL = S["NTL"]
        PF = S["PF"]
        G = S["G"]
        NB = S["NB"]
        F = S["F"]
        NW = S["NW"]
        NX = S["NX"]
        be = S["be"]
        blob = S["blob"]
        wr = S["wr"]
        tbr = S["tbr"]
        o_sg = S["o_sg"]
        o_sd = S["o_sd"]
        CT = S["CT"]
        x = S["x"]
        H = S["H"]
        xr = S["xr"]
        ts = S["ts"]
        PC = S["PC"]
        scr = S["scr"]
        frr = S["frr"]
        dsc = S["dsc"]
        dfr = S["dfr"]
        B = S["B"]
        f32 = S["f32"]
        bf16 = S["bf16"]
        fp8 = S["fp8"]
        MX = S["MX"]
        NR = S["NR"]
        NCH = S["NCH"]
        DW = S["DW"]
        r1 = S["r1"]
        r2 = S["r2"]
        r3 = S["r3"]
        r5 = S["r5"]
        ar = S["ar"]
        xtr = S["xtr"]
        wqr = S["wqr"]
        fg = S["fg"]
        fu = S["fu"]
        act = S["act"]
        lim = S["lim"]
        selg = S["selg"]
        asp = S["asp"]
        dq = S["dq"]
        Y = S["Y"]
        if lo <= i + PF < hi:  # loads of tile i + PF
            tl = i + PF
            for g in range(G):
                e = be.ap(pattern=[[NB, 1], [1, 1]], offset=tl * G + g)
                nisa.dma_copy(dst=wr[tl % NW][g], src=blob.ap(pattern=[[F, 128], [1, F]], offset=0,
                                                               scalar_offset=e, indirect_dim=0),
                              oob_mode=oob_mode.skip)
                if dq:  # rows 0 (gate) and 64 (up) of the tile scales, broadcast to every partition
                    nisa.dma_copy(dst=tbr[tl % NX][g], src=blob.ap(pattern=[[0, 128], [64 * F, 2], [1, CT * 4]],
                                                                   offset=o_sg, scalar_offset=e, indirect_dim=0),
                                  oob_mode=oob_mode.skip)
            nisa.dma_copy(dst=xr[tl % NX], src=x.ap(pattern=[[H, 128], [1, H]], offset=0,
                                                   vector_offset=ts.ap(pattern=[[NTL, 128], [1, 1]], offset=tl),
                                                   indirect_dim=0), oob_mode=oob_mode.skip)
        if i < lo:
            return
        if PC and lo <= i - 2 < hi:  # tile i - 2's down scales (stage C next step)
            for g in range(G):
                e = be.ap(pattern=[[NB, 1], [1, 1]], offset=(i - 2) * G + g)
                nisa.dma_copy(dst=scr[(i - 2) % 2][g * B:(g + 1) * B, :],
                              src=dsc.ap(pattern=[[0, B], [1, CT]], offset=0, scalar_offset=e, indirect_dim=0),
                              oob_mode=oob_mode.skip)
                nisa.dma_copy(dst=frr[(i - 2) % 2][g:g + 1, :],
                              src=dfr.ap(pattern=[[H, 1], [1, H]], offset=0, scalar_offset=e, indirect_dim=0),
                              oob_mode=oob_mode.skip)
        ta, tb, tf, tc = i, i - 1, i - 2, i - 3
        doA, doB, doF, doC = i < hi, lo <= tb < hi, lo <= tf < hi, lo <= tc < hi
        yb = None
        if doC:  # stage C, tile i - 3: every partition of a blob holds each chunk's down scale
            if PC:
                sdr = scr[tc % 2]
            else:
                sdr = nl.ndarray((128, CT), dtype=f32, buffer=nl.sbuf)
                for g in range(G):
                    nisa.tensor_copy(dst=sdr[g * B:(g + 1) * B, :],
                                     src=wr[tc % NW][g][g * B:(g + 1) * B, o_sd:F].view(f32),
                                     engine=nisa.vector_engine)
            if S["ybr"]:
                yb = S["ybr"][tc % len(S["ybr"])]
            else:
                yb = nl.ndarray((128, H), dtype=bf16, buffer=nl.sbuf)
        gbs, pfs, gcs, sls = [], [], [], []
        if doF:  # stage F, tile i - 2: its gate_up sums rounded to bf16 (vector, first in this step)
            for g in range(G):
                gb = nl.ndarray((128, B), dtype=bf16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=gb, src=accr[tf % 2][g], engine=nisa.vector_engine)
                gbs.append(gb)
        acc = accr[tb % 2]
        sgs = []  # per-row tile scales of tile i - 1's blocks, fp32 [128 o, CT] (stage B, not dq)
        if doB and not dq:
            for g in range(G):
                if MX:
                    sgf = nl.ndarray((128, CT), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=sgf, src=wr[tb % NW][g][:, o_sg:o_sd].view(bf16), engine=nisa.vector_engine)
                    sgs.append(sgf)
                else:
                    sgs.append(wr[tb % NW][g][:, o_sg:o_sd].view(f32))
        for c4 in range(NR):
            py = None
            if doB and c4 < CT // 4 and S["bfirst"]:  # ord_ 1: stage B first in the round (see _stage_b)
                _stage_b(S, acc, sgs, tb, c4)
            if doC and c4 < NCH:  # stage C, tile i - 3: one chunk's down matmuls
                hf, c5 = c4 // (DW // 512), c4 % (DW // 512)
                py = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
                for g in range(G):
                    nisa.nc_matmul(dst=py, stationary=ar[tc % 2][g][hf * 64:(hf + 1) * 64, :],
                                   moving=wr[tc % NW][g][hf * 64:(hf + 1) * 64,
                                                         H + c5 * 512:H + (c5 + 1) * 512].view(fp8),
                                   accumulate=(g > 0))
                if PC:  # each block's column factors broadcast onto its lanes (0/1 selector: exact)
                    pfb = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
                    nisa.nc_matmul(dst=pfb, stationary=selg, moving=frr[tc % 2][:, hf * DW + c5 * 512:
                                                                                 hf * DW + (c5 + 1) * 512])
            if doA and c4 < CT // 4:  # stage A, tile i: four transposes, one drain, four tiles dequantized
                # A tensor-engine transpose writes fp32 PSUM on gen2 (trn1) but must write its input's
                # dtype from gen3 on: "gen3: transpose dst must match input dtype" (nki/isa/
                # _validation.py, matmul dst checks; trn2 refused fp32 here). The values are x's
                # bf16 either way, copied into the bf16 xtr below.
                px = nl.ndarray((128, 4, 128), dtype=f32 if nisa.get_nc_version() == nisa.nc_version.gen2
                                else nl.bfloat16, buffer=nl.psum)
                for j in range(4):
                    c = c4 * 4 + j
                    nisa.nc_transpose(dst=px[:, j, :], data=xr[ta % NX][:, c * 128:(c + 1) * 128],
                                      engine=nisa.tensor_engine)
                nisa.activation(dst=xtr[ta % 2][:, c4 * 4:(c4 + 1) * 4, :], op=nl.copy, data=px)
                if dq:  # [h, c, (gate | up) o]: each 64-column half times its scale, a per-partition
                    # scalar (the scale on every partition; the vector engine runs a tensor-scalar op
                    # at twice the rate of a tensor-tensor op on two SBUF operands, measured)
                    # asp of them on the scalar engine, which otherwise only drains PSUM tiles.
                    for g in range(G):
                        for j in range(4):
                            c = c4 * 4 + j
                            for hh in range(2):
                                wdst = wqr[ta % 2][g][:, c, hh * 64:(hh + 1) * 64]
                                wsrc = wr[ta % NW][g][:, c * 128 + hh * 64:c * 128 + (hh + 1) * 64].view(fp8)
                                wsc = tbr[ta % NX][g].ap(pattern=[[2 * CT, 128], [1, 1]], offset=hh * CT + c, dtype=f32)
                                on_act = False
                                for q in range(asp):
                                    if ACT_ORDER[q] == j * 2 + hh:
                                        on_act = True
                                if on_act:
                                    nisa.activation(dst=wdst, op=nl.copy, data=wsrc, scale=wsc)
                                else:
                                    nisa.tensor_scalar(dst=wdst, data=wsrc, op0=nl.multiply, operand0=wsc,
                                                       engine=nisa.vector_engine)
            if doF and c4 == r1:  # stage F: the up rows folded onto the gate rows (both partition halves)
                for g in range(G):
                    pf = nl.ndarray((128, 2, B), dtype=f32, buffer=nl.psum)
                    nisa.nc_matmul(dst=pf[:, 0, :], stationary=fg, moving=gbs[g])
                    nisa.nc_matmul(dst=pf[:, 1, :], stationary=fu, moving=gbs[g])
                    pfs.append(pf)
            if doB and c4 < CT // 4 and not S["bfirst"]:
                _stage_b(S, acc, sgs, tb, c4)
            if doF and c4 == r2:  # stage F: gate (clamped) to SBUF, SiLU on the scalar engine
                for g in range(G):
                    gc = nl.ndarray((128, B), dtype=f32, buffer=nl.sbuf)
                    if act == 1:
                        nisa.tensor_scalar(dst=gc, data=pfs[g][:, 0, :], op0=nl.minimum, operand0=lim,
                                           engine=nisa.vector_engine)
                    else:
                        nisa.tensor_copy(dst=gc, src=pfs[g][:, 0, :], engine=nisa.vector_engine)
                    uc = nl.ndarray((128, B), dtype=f32, buffer=nl.sbuf)
                    if act == 1:
                        nisa.tensor_scalar(dst=uc, data=pfs[g][:, 1, :], op0=nl.minimum, operand0=lim, op1=nl.maximum,
                                           operand1=-lim, engine=nisa.vector_engine)
                    else:
                        nisa.tensor_copy(dst=uc, src=pfs[g][:, 1, :], engine=nisa.vector_engine)
                    gcs.append((gc, uc))
            if py is not None:  # stage C: that chunk's drains, scaled per 128 columns
                for j in range(4):
                    cc = c5 * 4 + j  # chunk c' of this half: output columns hf * DW + cc * 128
                    col = hf * DW + cc * 128
                    nisa.activation(dst=yb[:, col:col + 128], op=nl.copy, data=py[:, j * 128:(j + 1) * 128],
                                    scale=sdr[:, 2 * cc + hf:2 * cc + hf + 1])
                if PC:  # times the column factors: a power of two, so bf16(y s_c) f = bf16(y s_c f)
                    col = hf * DW + c5 * 512
                    nisa.tensor_tensor(dst=yb[:, col:col + 512], data1=yb[:, col:col + 512], data2=pfb,
                                       op=nl.multiply, engine=nisa.vector_engine)
            if doF and c4 == r3:
                for g in range(G):
                    sl = nl.ndarray((128, B), dtype=f32, buffer=nl.sbuf)
                    nisa.activation(dst=sl, op=nl.silu, data=gcs[g][0])
                    sls.append(sl)
            if doF and c4 == r5:  # stage F: a = silu(gate) up, bf16, into the block's own columns
                for g in range(G):
                    nisa.tensor_tensor(dst=ar[tf % 2][g][:, g * B:(g + 1) * B], data1=sls[g], data2=gcs[g][1],
                                       op=nl.multiply, engine=nisa.vector_engine)
        if yb is not None:  # Y rows in slot order: a static DMA (every lane, empty or not)
            nisa.dma_copy(dst=Y[tc * 128:(tc + 1) * 128, :], src=yb)

else:
    kiln_moe_prefill_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of the kernel's source text, passed to it as the static argument `rev`. LNL's graph
    cache key (libtorch_neuronx_lite/compile/cache.py, create_cache_hash, SDK 2.32) hashes the FX
    graph with each NKI call's backend config minus the kernel binary's path, i.e. its name, operand
    names, grid and MAC count, and not its source: a kernel edit that keeps those (say, a DMA made
    static) reuses the NEFF compiled from the old source (measured 2026-10-03: same graph hash, a
    0.7 s "compile", the old kernel's time). Static arguments are in the hashed graph text."""
    import zlib

    src = open(__file__).read()
    a = src.index("    @nki.jit\n    def kiln_moe_prefill_kernel")
    b = src.index("else:\n    kiln_moe_prefill_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev()  # at import: the traced caller reads a constant


def kernel():
    """The NKI kernel (raises where the NKI package is missing)."""
    if kiln_moe_prefill_kernel is None:
        raise RuntimeError("the NKI prefill MoE kernel needs the nki package (the Neuron venv)")
    return kiln_moe_prefill_kernel


def consts(device):
    """(fold_g, fold_u): the kernel's 0/1 fold matrices (see kiln_moe_prefill_kernel)."""
    p = torch.arange(P, device=device)
    o, q = p.view(P, 1), p.view(1, P)
    return (o == q % 64).to(torch.bfloat16), (o == 64 + q % 64).to(torch.bfloat16)


ACTS = (0, 1)  # moe_dedupe.ACTS "silu", "silu_clamp"


# Lane tiles per skippable chunk (kernel argument skp; 0: every static tile runs).
SKIP = int(os.environ.get("KILN_MOE_PREFILL_SKIP", "0"))
# Half-tile dequantizations per block and round on the scalar engine (kernel argument asp), dq path:
# the vector engine is the busy one there (128 tensor_scalar ops per lane tile, about 140 ns each in the
# kernel), the scalar engine drains PSUM and has room. One call at GLM-5.3-Flash's tp=32 rank shapes,
# random dq-layout experts (tools/probe_moe_prefill.py --format glm --chunks 1024 8192, trn1.2xlarge,
# 2026-10-04): asp 0 / 2 / 3 / 4: C=1024 4.969 / 4.656 / 4.447 / 4.814 ms, C=8192 15.237 / 14.791 /
# 14.690 / - ms, the output the same distance from the reference every time.
ACT_SPLIT = int(os.environ.get("KILN_MOE_PREFILL_ACT_SPLIT", "3"))
# Instruction order of each pipeline round (kernel argument ord_, _stage_b) and the ring of stage C outputs
# (nyb). GLM-5.3-Flash's real experts of layer 3, rank 0, C=4096, KILN_MOE_PREFILL_SKIP=20 (trn1.2xlarge,
# tools/probe_moe_prefill.py --experts-file ... --compare, 2026-10-04): ord_ 0 / 1 9.725 / 9.551 ms; with
# ord_ 1, nyb 0 / 2 / 3 / 4 9.563 / 10.559 / 9.340 / 9.350 ms; outputs bit-identical in every case.
ORDER = int(os.environ.get("KILN_MOE_PREFILL_ORDER", "1"))
# KILN_MOE_PREFILL_INBAR=1 (experiment, LNC split only): core barriers on the inputs at the kernel's start.
INBAR = int(os.environ.get("KILN_MOE_PREFILL_INBAR", "0"))
# KILN_MOE_PREFILL_NOSKIPX=1 (experiment): empty lanes gather row 0 instead of the out-of-range row C (oob_mode=skip).
NOSKIPX = int(os.environ.get("KILN_MOE_PREFILL_NOSKIPX", "0"))
# KILN_MOE_PREFILL_BARENG (experiment): the engine of the split's core barriers, 0 GpSimd (default), 1 sync, 2 vector.
BARENG = int(os.environ.get("KILN_MOE_PREFILL_BARENG", "0"))
NYB = int(os.environ.get("KILN_MOE_PREFILL_NYB", "3"))


def _platform():
    from .. import platform

    return platform


def kernel_inputs(x, topv, topi, blob, act: int = 0, limit: float = 0.0, B: int | None = None, dq: bool = False,
                  down=None, asp: int | None = None, skp: int | None = None, order: int | None = None,
                  nyb: int | None = None):
    """The kernel's arguments for x [C, H] (C a multiple of 128), routing [C, k] and one layer's
    tiles blob [E, 128, F]; dq as check_blob and down as down_factors returned for that blob (both
    computed once at load: they read the blob's values); asp: the kernel's (default ACT_SPLIT)."""
    if act not in ACTS:
        raise NotImplementedError(f"prefill MoE kernel: activation {act} (moe_dedupe.ACTS) is not implemented")
    C, K = topi.shape
    E, H = blob.shape[0], x.shape[1]
    B = B or block_size(C)
    fold_g, fold_u = consts(x.device)
    if blob_scale_bytes(blob, H) == 2:
        if dq:
            raise ValueError("bf16 tile scales have no dequantize-first path (check_blob)")
        if down is None:
            raise ValueError("bf16 tile scales are per output column: pass down_factors(blob, H)")
    if down is None:  # not read (fp32 scales constant over each chunk: per chunk, from the blob)
        dsc, dfr, pc = fold_g, fold_g, 0
    else:
        dsc, dfr = down
        pc = 1
    return dict(x=x, topi=topi.to(torch.int32), wts=topv.to(torch.bfloat16), blob=blob, dsc=dsc, dfr=dfr,
                fold_g=fold_g, fold_u=fold_u, B=B, NB=n_blocks(C * K, E, B), act=act, lim=float(limit), dq=int(dq),
                pc=pc, asp=(ACT_SPLIT if asp is None else asp) if dq else 0, skp=SKIP if skp is None else skp,
                ord_=ORDER if order is None else order, nyb=NYB if nyb is None else nyb, rev=REV,
                spl=int(_platform().lnc_split("moe_prefill")), inb=INBAR, nsx=NOSKIPX, bng=BARENG)


def emulate(x, topv, topi, blob, act: int = 0, limit: float = 0.0, dq: bool = False) -> torch.Tensor:
    """The kernel's arithmetic in torch, one expert at a time (moe_dedupe.emulate with pair_bf16,
    which gathers every pair's expert and does not fit host memory past C = 512): per pair the
    gate_up tile dot products in fp32 times the tile scales, g rounded to bf16, a = glu rounded to
    bf16, the down product times its column scale rounded to bf16, the fp32 sum of weight x output
    over the token's experts. dq: gate_up dequantized to bf16 first (fp8 x scale in fp32, rounded
    once, as models/quant.dequant), then x W^T in fp32 rounded to bf16."""
    from .moe_dedupe import glu, unpack_down, unpack_gu

    T, H = x.shape
    C = H // P
    out = torch.zeros(T, H, dtype=torch.float32)
    for e in torch.unique(topi).tolist():
        t, k = (topi == e).nonzero(as_tuple=True)
        w_gu, s_gu = unpack_gu(blob[e:e + 1], H)
        w_down, s_down = unpack_down(blob[e:e + 1], H)
        sg = s_gu[0].float()  # [128 o, C] (fp32) or [128 o, H / 32] (bf16, repeated per tile)
        sg = sg[:, :: sg.shape[1] // C]
        if dq:
            wd = (w_gu[0].float().view(P, C, P) * sg.unsqueeze(-1)).bfloat16().float().view(P, H)
            g = (x[t].float() @ wd.T).bfloat16().float()
        else:
            part = torch.einsum("nci,oci->noc", x[t].float().view(-1, C, P), w_gu[0].float().view(P, C, P))
            g = (part * sg).sum(-1).bfloat16().float()  # [n, 128]
        a = glu(g[:, :64], g[:, 64:], act, limit).bfloat16().float()
        y = ((a @ w_down[0].float()) * s_down[0, 0].float()).bfloat16().float()
        out.index_add_(0, t, y * topv[t, k].float().unsqueeze(1))
    return out.to(x.dtype)


def moe_prefill(x, topv, topi, blob, act: int = 0, limit: float = 0.0, dq: bool = False,
                down=None) -> torch.Tensor:
    """sum_k topv[t, k] * expert_{topi[t, k]}(x[t]) for a prefill chunk, on the device inside the
    caller's graph, as one kernel call on the layer's tiles blob (the chunk padded to a multiple of
    128 tokens: the padding tokens are routed to expert 0 with weight 0 and dropped); dq and down
    as check_blob and down_factors give them for the blob."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    T = x.shape[0]
    pad = -T % P
    if pad:
        x = torch.cat([x, x.new_zeros(pad, x.shape[1])])
        topi = torch.cat([topi, topi.new_zeros(pad, topi.shape[1])])
        topv = torch.cat([topv, topv.new_zeros(pad, topv.shape[1])])
    out = wrap_nki(kernel())[platform.nki_grid()](**kernel_inputs(x, topv, topi, blob, act, limit, dq=dq, down=down))
    return out[:T] if pad else out
