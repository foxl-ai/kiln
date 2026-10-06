"""Selected-expert MoE that reads each selected expert ONCE per call (decode batches, MTP verify,
prefill chunks), as one NKI kernel for NeuronCore-v2 (trn1).

kernels/moe_decode.py loads one expert blob per (token, expert) pair, so its HBM traffic follows
the pairs: with T tokens x top-8 of 256 experts, T=32 reads 256 blobs for about 163 distinct
experts. Here the pairs are grouped by expert into SLOTS of up to `lanes` pairs each, and a slot
loads its expert once and applies it to all of its lanes (one matmul column per lane).

What trn1 allows (tools/probe_nki_dynamic.py, SDK 2.32, nki 0.6.0, 2026-10-03): only DMAs take
runtime addresses. A compute instruction whose operand sits at an offset held in a register or an
SBUF tensor fails in the BIR backend ("[NCC_IBIR829] Requested Argument index 1 out of bounds",
"[NCC_IBIR040] Matmult stationary input tile size must be <= 128x128") although nki.simulate runs
it. A DMA reading blob[e] at an index from SBUF works, and with oob_mode=skip an index >= E skips
the transfer (256 expert DMAs 0.98 ms, all skipped 0.34 ms). Device loops of 0 or 1 iterations
(nl.dynamic_range over a register) do run, but they may not nest ("operand #3 does not dominate
this use"), each taken one costs about 1.4 us, and around the kernel's DMAs and matmuls they gave
wrong results (NaN) and a slower kernel (2.2 ms against 1.2 ms static at T=32). So the kernel is
static code over a static number of slots, and everything that depends on the routing is DATA the
kernel computes first from topi / topv (plan()'s arithmetic, which is also its CPU reference; the
same plan as torch ops in the graph cost 0.1-0.5 ms per call, tools/probe_moe_plan.py):
- slot_e[s]: the expert of slot s; slots beyond the real ones hold E, so their DMA is skipped.
- G [T, lanes]: the 0/1 token of each lane. The kernel gathers the lanes' inputs with it on the
  tensor engine (x^T G per 128-row tile of x: one 1.0 per column, so exact).
- R [lanes, T] (per block of lanes): the routing weight of each lane's pair at its token, 0
  elsewhere; the lanes' outputs are summed back into their tokens by a matmul with it.
Calls under PAIRS_BELOW tokens run kiln_moe_tiles_pairs_v1 instead: one load per pair in order, the
pairs' tokens static (no plan, gather or route), outputs accumulated per token in fp32.

Static slot count: an expert with n pairs takes ceil(n / lanes) slots, so N = T x k pairs over at
most min(E, N) experts take at most (N + min(E, N) (lanes - 1)) / lanes slots (n_slots). For
T x k <= E that is N whatever `lanes` is, so the compute is static per slot and the saving is in
the expert loads (DMA) only.

Layout ("tiles", pack()): moe_decode's blob carries one bf16 scale per 32 input columns, which the
per-pair kernel applies after the matmul over four 32-row blocks of every 128-row tile (x
block-diagonal, then 128 partial sums x 128 scales per pair on the vector engine). MXFP4 scales
are powers of two, so each block can be re-based on its tile's scale instead: code * 2^(k_b - K)
is the same e4m3 value shifted, exact unless it leaves e4m3's range (240) or falls off its 2^-9
grid; pack() picks the tile exponent K inside that window and refuses a tile with none (measured on
XiaomiMiMo/MiMo-V2.6-Flash-RL, tools/check_expert_scales.py: block exponents of a 128-column tile
differ by at most 8, and no code would round). Per expert per rank, F = H + H/2 + H/64 + H/64 bytes
per partition:
  [0, H)           gate_up, fp8: blob[p, c * 128 + o] = w_gu'[o, c * 128 + p] (as moe_decode);
  [H, 3H/2)        down, fp8: blob[q, h'] = w_down'[q % Im, (q // Im) * H/2 + h'] (as moe_decode);
  [3H/2, +H/64)    gate_up tile scales, bf16 [128 o, H / 128]: 2^K of row o, tile c;
  [.., F)          down scales, bf16 [128 p, H / 128]: column 2 c' + half holds 2^K of output
                   column h = half * H/2 + c' * 128 + p (its tile is the rank's Im input rows).
So one 128-row tile is one matmul whose PSUM column is already the tile's dot product: x needs no
block-diagonal expansion, a pair's gate_up scales are 32 multiplies (not 128) and its down scales
32 (not 64, plus no sum over the two 32-row blocks). w' * 2^K equals w * 2^k_b element for element,
so dequantizing either layout gives the same weights (unpack()).

Per slot (static): one DMA, 32 gate_up matmuls with the slot's lanes as the moving columns, scales,
a 0/1 matmul folding the up rows onto the gate rows, SiLU, 16 down matmuls, scales into the block's
output columns. Every engine runs its instructions in program order and the compiler tracks
dependencies per tensor (tools/prof_timeline.py), so: slots go in groups of SG whose vector and
scalar instructions are shared; the program is skewed (step i issues the loads of group i + RING -
2, stage A (gate_up, scales) of group i, stages B (fold, SiLU) and C (down, output) of group i - 1);
every ring entry is a tensor of its own; and the scales are copied out of the expert buffers in
stage A, so that a buffer's next load waits only on the tensor engine's down matmuls. Per block of
lanes, the lanes' x is gathered first and the block's outputs [128 p, 32, lanes] are transposed on
the tensor engine and routed into an fp32 [T, H] accumulator after its last group.

Numerics: fp32 accumulation; g and a are rounded to bf16 as matmul operands (as moe_decode); the
routing weight multiplies a pair's output instead of a, after the output is rounded to bf16 (the
transposes and the routing matmul take bf16); the sum over a token's k experts is fp32 (the routing
matmul). emulate() is that arithmetic in torch.

Expert buffers are a static ring, so a skipped DMA leaves the expert an earlier slot loaded there
(finite bytes). Slots are compacted (the real ones first), and the padded slots that would be the
first to use a ring buffer load expert 0 instead (plan's `keep`), so no slot computes on a buffer
that was never written (uninitialised SBUF may hold NaN bytes, and 0 x NaN is NaN). A padded slot's
lanes are empty (x = 0, weight 0), so it adds 0.
"""

from __future__ import annotations

import os

import torch

P = 128  # partitions
QB = 32  # MXFP4 scale block along the input dim
FP8 = torch.float8_e4m3fn
E4M3_MAX = 240.0  # trn1 e4m3 (with inf), docs/neuron-notes.md


# --- layout -----------------------------------------------------------------------------------

def blob_cols(H: int, sbytes: int = 2) -> int:
    """Bytes per partition of one packed expert (module docstring); sbytes 2 for bf16 tile scales
    (re-based MXFP4), 4 for fp32 ones (FP8 with 128 x 128 block scales)."""
    return H + H // 2 + (H // P) * sbytes + (H // P) * sbytes


def scale_bytes(F: int, H: int) -> int:
    """Bytes per tile scale of a blob with F bytes per partition (2: bf16, 4: fp32)."""
    sb = (F - H - H // 2) // (2 * (H // P))
    if blob_cols(H, sb) != F or sb not in (2, 4):
        raise ValueError(f"not a moe_dedupe blob: {F} bytes per partition at hidden {H}")
    return sb


def _offsets(H: int, sbytes: int = 2) -> tuple[int, int, int]:
    """Byte offsets of the down weights, the gate_up tile scales and the down tile scales."""
    return H, H + H // 2, H + H // 2 + (H // P) * sbytes


def _block128(w_gu, s_gu, w_down, s_down) -> bool:
    """FP8 experts with 128 x 128 block scales as the loader keeps them (models/quant.py: fp32, one
    per row per 128 input columns for gate_up, and one per output column for down, whose Im = 64
    input rows of a rank sit inside one 128-row block): the tile scales as they are."""
    E, R, H = w_gu.shape
    return (s_gu.dtype == torch.float32 and tuple(s_gu.shape) == (E, R, H // P)
            and s_down.dtype == torch.float32 and tuple(s_down.shape) == (E, 1, H))


def supports(w_gu: torch.Tensor, s_gu, w_down: torch.Tensor, s_down, down_t: bool) -> bool:
    """The shapes pack() takes: FP8 experts, 2 Im = 128 gate/up rows per rank, w_down stored
    [E, Im, H], and either bf16 block-32 scales (MXFP4; whether every tile re-bases exactly is
    checked by pack()) or fp32 128 x 128 block scales (_block128)."""
    if w_gu.dtype != FP8 or w_down.dtype != FP8 or s_gu is None or s_down is None or not down_t:
        return False
    E, R, H = w_gu.shape
    if not (R == P and H % (2 * P) == 0 and tuple(w_down.shape) == (E, R // 2, H)):
        return False
    return _block128(w_gu, s_gu, w_down, s_down) or (
        s_gu.dtype == torch.bfloat16 and tuple(s_gu.shape) == (E, R, H // QB)
        and s_down.dtype == torch.bfloat16 and tuple(s_down.shape) == (E, R // 2 // QB, H))


def _window(w: torch.Tensor, k: torch.Tensor):
    """w fp8 [..., n, G, B] (G blocks of B values along a tile, the last dim), k int [..., n, G]
    block exponents -> (lo, hi, K) int [..., n]: the tile exponents K for which every code v of
    block b stays an e4m3 value as v * 2^(k_b - K) are lo <= K <= hi (|v| 2^(k_b - K) <= 240, and a
    multiple of 2^-9, e4m3's finest step); K is the block-exponent maximum clamped into it."""
    wf = w.float()
    a = wf.abs()
    nz = a > 0
    big = a.amax(-1)  # [.., n, G]
    # lowest set bit of each value: v = i * 2^-9 for an integer i (e4m3's finest step)
    iv = torch.round(a * 512).to(torch.int64)
    low = torch.where(nz, torch.log2((iv & -iv).clamp(min=1).double()).to(torch.int64) - 9, 10 ** 6)
    lsb = low.amin(-1)  # [.., n, G]
    k = k.to(torch.int64)
    has = nz.any(-1)
    neg = -(10 ** 6)
    up = torch.where(has, k + torch.ceil(torch.log2(big.double().clamp(min=1e-30) / E4M3_MAX)).to(torch.int64), neg)
    lo = up.amax(-1)  # K >= every block's k + log2(max / 240)
    hi = torch.where(has, k + lsb + 9, 10 ** 6).amin(-1)  # K <= every block's k + lsb + 9
    K = torch.maximum(torch.minimum(torch.where(has, k, neg).amax(-1), hi), lo)
    K = torch.where(has.any(-1), K, k.amax(-1))  # an all-zero tile: any scale
    return lo, hi, K


def _rebase(w: torch.Tensor, k: torch.Tensor, what: str):
    """(w' fp8 [..., n, G, B], K int [..., n]) with w' * 2^K == w * 2^k exactly (_window)."""
    lo, hi, K = _window(w, k)
    if bool((lo > hi).any()):
        n = int((lo > hi).sum())
        raise ValueError(f"{what}: {n} tiles have block scales too far apart to share one e4m3 scale exactly "
                         "(KILN_MOE_KERNEL=nki-pair keeps the block-32 layout)")
    wf = w.float()
    k = k.to(torch.int64)
    sh = (k - K.unsqueeze(-1)).clamp(-200, 200)  # [.., n, G]
    out = (wf * torch.exp2(sh.double()).unsqueeze(-1).float()).to(FP8)
    if not torch.equal(out.float().double() * torch.exp2(K.double()).unsqueeze(-1).unsqueeze(-1),
                       wf.double() * torch.exp2(k.double()).unsqueeze(-1)):
        raise ValueError(f"{what}: re-basing a tile on one scale is not exact")
    return out, K


def _exponents(s: torch.Tensor, what: str) -> torch.Tensor:
    """Exact integer log2 of power-of-two bf16 scales (MXFP4 E8M0); anything else raises."""
    k = torch.round(torch.log2(s.double()))
    if not torch.equal(torch.exp2(k), s.double()):
        raise ValueError(f"{what}: the tile layout needs power-of-two scales (MXFP4 experts)")
    return k.to(torch.int64)


def pack(w_gu: torch.Tensor, s_gu: torch.Tensor, w_down: torch.Tensor, s_down: torch.Tensor) -> torch.Tensor:
    """w_gu fp8 [E, 128, H], w_down fp8 [E, 64, H] (input dim first) with either s_gu bf16
    [E, 128, H/32] and s_down bf16 [E, 2, H] (MXFP4 blocks: re-based on one scale per tile, bf16)
    or s_gu fp32 [E, 128, H/128] and s_down fp32 [E, 1, H] (128 x 128 blocks: the tile scales as
    they are, fp32) -> blob uint8 [E, 128, blob_cols(H, 2 or 4)] (module docstring). Exact or raises."""
    E, R, H = w_gu.shape
    Im, C, C2 = R // 2, H // P, H // 2 // P
    jn = Im // QB
    u8 = lambda t: t.contiguous().view(torch.uint8)  # noqa: E731
    if _block128(w_gu, s_gu, w_down, s_down):
        wg, wd = w_gu, w_down
        sg = s_gu.contiguous()  # [E, 128 o, C] fp32
        sdh = s_down[:, 0, :]  # [E, H] fp32
    else:
        k_gu = _exponents(s_gu, "gate_up scales")  # [E, 128, H/32]
        wg, Kg = _rebase(w_gu.view(E, R, C, P // QB, QB), k_gu.view(E, R, C, P // QB), "gate_up")
        wg = wg.reshape(E, R, H)  # [E, o, i]
        k_d = _exponents(s_down, "down scales")  # [E, jn, H]
        # tile = the rank's Im input rows of one output column: [E, H, jn blocks, 32]
        wd, Kd = _rebase(w_down.view(E, jn, QB, H).permute(0, 3, 1, 2), k_d.permute(0, 2, 1), "down")
        wd = wd.permute(0, 2, 3, 1).reshape(E, Im, H)  # back to [E, i, h]
        sg = torch.exp2(Kg.double()).to(torch.bfloat16)  # [E, 128 o, C]
        sdh = torch.exp2(Kd.double()).to(torch.bfloat16)  # [E, H]
    gu = wg.view(E, R, C, P).permute(0, 3, 2, 1).reshape(E, P, C * R)  # [e, p, (c, o)]
    dw = wd.view(E, Im, 2, H // 2).permute(0, 2, 1, 3).reshape(E, P, H // 2)  # [e, (half, i), h']
    # down scale of h = half * H/2 + c' * 128 + p at [e, p, 2 c' + half]
    sd = sdh.view(E, 2, C2, P).permute(0, 3, 2, 1).reshape(E, P, 2 * C2)
    return torch.cat([u8(gu), u8(dw), u8(sg), u8(sd)], dim=-1)


def _as(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t.contiguous().view(dtype)


def _sdt(blob: torch.Tensor, H: int):
    sb = scale_bytes(blob.shape[-1], H)
    return sb, (torch.bfloat16 if sb == 2 else torch.float32)


def unpack_gu(blob: torch.Tensor, H: int, idx: torch.Tensor | None = None):
    """(w_gu fp8 [.., 128, H], s_gu [.., 128, H/32] bf16 or [.., 128, H/128] fp32) of blob
    [E, 128, F] (rows idx if given): the codes and the tile scales (bf16 ones repeated over a tile's
    four blocks), which dequantize to pack's input exactly. Slices, views and permutes, so usable
    inside a graph."""
    sb, sdt = _sdt(blob, H)
    _, o2, o3 = _offsets(H, sb)
    w, s = blob[..., :H], blob[..., o2:o3]
    if idx is not None:
        w, s = w[idx], s[idx]
    lead = w.shape[:-2]
    gu = _as(w, FP8).view(*lead, P, H // P, P)  # [.., p, c, o]
    sc = _as(s, sdt)  # [.., o, c]
    if sb == 2:
        sc = sc.unsqueeze(-1).expand(*sc.shape, P // QB).reshape(*lead, P, H // QB)
    return gu.permute(*range(len(lead)), -1, -2, -3).reshape(*lead, P, H), sc


def unpack_down(blob: torch.Tensor, H: int, idx: torch.Tensor | None = None):
    """(w_down fp8 [.., 64, H], s_down [.., 2, H] bf16 or [.., 1, H] fp32) of blob, as unpack_gu."""
    sb, sdt = _sdt(blob, H)
    o1, o2, o3 = _offsets(H, sb)
    w, s = blob[..., o1:o2], blob[..., o3:]
    if idx is not None:
        w, s = w[idx], s[idx]
    lead = w.shape[:-2]
    Im = P // 2
    dw = _as(w, FP8).view(*lead, 2, Im, H // 2)  # [.., half, i, h']
    sd = _as(s, sdt).view(*lead, P, H // 2 // P, 2)  # [.., p, c', half]
    sh = sd.permute(*range(len(lead)), -1, -2, -3).reshape(*lead, 1, H)  # h = half * H/2 + c' * 128 + p
    return (dw.permute(*range(len(lead)), -2, -3, -1).reshape(*lead, Im, H),
            sh.expand(*lead, Im // QB, H) if sb == 2 else sh)


def unpack(blob: torch.Tensor, H: int):
    """(w_gu, s_gu, w_down, s_down) in the natural layout, equal to pack's input after dequant."""
    return (*unpack_gu(blob, H), *unpack_down(blob, H))


# --- routing plan -----------------------------------------------------------------------------

def default_lanes(T: int) -> int:
    """Lanes per slot by batch: about the most pairs an expert gets at this T (top-8 of 256,
    T / 32 pairs per expert on average). Below 8 tokens pairs rarely share an expert."""
    return 1 if T < 8 else 2 if T <= 16 else 4 if T <= 64 else 8


# Slots per group: each of the kernel's vector and scalar instructions serves a group.
SG = int(os.environ.get("KILN_MOE_DEDUPE_GROUP", 4))
# Groups of expert buffers (loads run RING - 2 groups ahead).
RING = int(os.environ.get("KILN_MOE_DEDUPE_RING", 3))
# kiln_moe_dedupe_v9's expert ring (192 rows on one trn1 core: 2.302 -> 2.152 ms at 4 without the segments below; with
# them 3 is as fast or faster and spills less SBUF: 1.835 against 1.966 ms, docs/neuron-notes.md).
RING9 = int(os.environ.get("KILN_MOE_DEDUPE_RING9", 3))
# kiln_moe_dedupe_v9's skippable segments, in blocks of lanes (its kernel argument skp; 0: every static slot runs).
SKIP9 = int(os.environ.get("KILN_MOE_DEDUPE_SKIP9", 6))
# and the share of its static slots (percent) that always runs before them (its kernel argument hdf).
HEAD9 = int(os.environ.get("KILN_MOE_DEDUPE_HEAD9", 75))
# Below this many tokens a call runs the per-pair kernel on the same layout (kiln_moe_tiles_pairs):
# pairs then seldom share an expert, and the plan, gather and route cost more than they save.
PAIRS_BELOW = int(os.environ.get("KILN_MOE_DEDUPE_PAIRS_BELOW", 16))


def group_size(lanes: int) -> int:
    """Slots per group: SG, fewer where a group's PSUM tiles (slots x lanes x 32 fp32 columns) would
    not fit one 2 KB bank ("[NCC_IBIR038] Output size doesn't fit PSUM", lanes 8 with 4 slots)."""
    return max(1, min(SG, 16 // lanes))


def n_slots(T: int, K: int, E: int, lanes: int) -> tuple[int, int]:
    """(static slot count, lanes per block): at most (N + min(E, N) (lanes - 1)) / lanes slots for
    N = T x K pairs, rounded to whole groups (group_size); lanes are processed in blocks of up to
    128, and above that the slot count is rounded to whole blocks."""
    N = T * K
    q = group_size(lanes)
    s = -(-((N + min(E, N) * (lanes - 1)) // lanes) // q) * q
    if s * lanes <= P:
        return s, s * lanes
    per = P // lanes
    return -(-s // per) * per, P


def plan(topv: torch.Tensor, topi: torch.Tensor, E: int, lanes: int, keep: int = 0):
    """Routing [T, K] -> (slot_e int32 [1, S], G bf16 [T, S * lanes], R bf16 [BL, blocks, T]),
    with torch ops that lower inside the graph: int32 compares and sums only (no sort, no integer
    division, no float literals in comparisons; docs/neuron-notes.md).

    Pair n = t * K + j has rank r (its index among the earlier pairs of its expert) and lane
    lanes * base[e] + r, where base is the exclusive prefix sum of the experts' slot counts
    ceil(count / lanes): an expert's slots are consecutive, and so are its lanes. G is the 0/1
    token of each lane; R[lane, t] is the routing weight of that pair (0 for an empty lane).
    Padded slots below `keep` load expert 0 instead of skipping their DMA: the kernel's expert
    buffers are first written there, and a never-written buffer may hold NaN bytes."""
    T, K = topi.shape
    N, L = T * K, lanes
    S, BL = n_slots(T, K, E, L)
    SL = S * L
    dev = topi.device
    i32 = torch.int32
    e = topi.reshape(N).to(i32)
    ar_e = torch.arange(E, device=dev, dtype=i32)
    oh = (e.unsqueeze(1) == ar_e.unsqueeze(0)).to(i32)  # [N, E]
    cnt = oh.sum(0, dtype=i32)  # [E]
    ar_n = torch.arange(N, device=dev, dtype=i32)
    earlier = ar_n.unsqueeze(0) < ar_n.unsqueeze(1)  # [n, n']: n' < n
    r = ((e.unsqueeze(1) == e.unsqueeze(0)) & earlier).to(i32).sum(1, dtype=i32)  # [N]
    nsl = (cnt.unsqueeze(1) > torch.arange(0, -(-T // L), device=dev, dtype=i32).unsqueeze(0) * L).to(i32).sum(
        1, dtype=i32)  # ceil(cnt / L)
    base = (nsl.unsqueeze(0) * (ar_e.unsqueeze(0) < ar_e.unsqueeze(1)).to(i32)).sum(1, dtype=i32)  # [E]
    lane = (oh * base.unsqueeze(0)).sum(1, dtype=i32) * L + r  # [N]
    ar_s = torch.arange(S, device=dev, dtype=i32)
    inr = ((ar_s.unsqueeze(1) >= base.unsqueeze(0)) & (ar_s.unsqueeze(1) < (base + nsl).unsqueeze(0))).to(i32)
    real = inr.sum(1, dtype=i32)  # [S] 1 for a real slot
    slot_e = (inr * ar_e.unsqueeze(0)).sum(1, dtype=i32) + (1 - real) * E  # padded: E (DMA skipped)
    early = (1 - real) * (ar_s < keep).to(i32)
    slot_e = slot_e - early * E
    hit = (lane.unsqueeze(1) == torch.arange(SL, device=dev, dtype=i32).unsqueeze(0))  # [N, SL]
    G = hit.to(i32).view(T, K, SL).sum(1, dtype=i32).to(torch.bfloat16)  # [T, SL]
    Rw = (hit.to(torch.float32) * topv.reshape(N, 1).to(torch.float32)).view(T, K, SL).sum(1)  # [T, SL]
    R = Rw.to(torch.bfloat16).t().reshape(SL // BL, BL, T).permute(1, 0, 2)  # [BL, blocks, T]
    return slot_e.view(1, S), G, R.contiguous()


def consts(im: int, device):
    """(mhalf fp32 [128, 2] = [p < im, p >= im], fold bf16 [128, 128] = [o % im == q % im],
    ident bf16 [128, 128])."""
    p = torch.arange(P, device=device)
    mhalf = torch.stack([p < im, p >= im], dim=1).float()
    fold = (p.view(P, 1) % im == p.view(1, P) % im).to(torch.bfloat16)
    ident = (p.view(P, 1) == p.view(1, P)).to(torch.bfloat16)
    return mhalf, fold, ident


def _lnc_split() -> bool:
    from .. import platform

    return platform.lnc_split("moe_dedupe")


def kernel_inputs(x: torch.Tensor, topv: torch.Tensor, topi: torch.Tensor, blob: torch.Tensor,
                  lanes: int | None = None, act: int = 0, limit: float = 0.0, alpha: float = 1.702):
    """The kernel's arguments for x [T, H] (T <= 128: kiln_moe_dedupe_v8; 128 < T <= 256: kiln_moe_dedupe_v9),
    routing [T, K] and one layer's blob: the routing as it comes (the kernel plans the slots itself, as plan() does),
    and the static shape parameters."""
    T, H = x.shape
    if T > 2 * P or (V10_ALL and T > P):
        return _kernel_inputs10(x, topv, topi, blob, lanes, act, limit, alpha)
    v9 = T > P or V9_ALL
    if not fits(T, topi.shape[1], blob.shape[0], lanes):
        raise ValueError(f"the dedupe kernel plans at most {4 * P} experts, {2 * P} tokens, {16 * P if v9 else 1024} "
                         f"pairs and {SLOTS9} slots (128 < T) per call")
    L = lanes or default_lanes(T)
    K = topi.shape[1]
    S, BL = n_slots(T, K, blob.shape[0], L)
    q = group_size(L)
    d = dict(x=x.contiguous(), topi=topi.to(torch.int32).contiguous(), topv=topv.to(torch.bfloat16).contiguous(),
             blob=blob, lanes=L, group=q, ring=RING, slots=S, block=BL, keep=RING * q, act=act,
             limit=float(limit), alpha=float(alpha), debug=int(os.environ.get("KILN_MOE_DEDUPE_DEBUG", 0)))
    if v9:
        d = dict(d, ring=RING9, keep=RING9 * q, skp=SKIP9, hdf=HEAD9)
    return dict(d, rev=REV9 if v9 else REV, spl=int(_lnc_split()))


# kiln_moe_dedupe_v9's slot bound: its slot table is one PSUM bank row ([1, S] fp32).
SLOTS9 = 512
# kiln_moe_dedupe_v10's: rows of 512 (T = 512 at lanes 8: 768 slots).
SLOTS10 = 1024
# KILN_MOE_DEDUPE_V10=1: kiln_moe_dedupe_v10 for calls of 129-256 tokens too, instead of v9.
V10_ALL = os.environ.get("KILN_MOE_DEDUPE_V10", "0") == "1"


def fits10(T: int, K: int, E: int, lanes: int | None = None) -> bool:
    """Whether one kiln_moe_dedupe_v10 call takes T tokens: up to 512 of them, top-8 (a pair tile of 128 is 16 tokens),
    T a multiple of 16, at most 512 experts and SLOTS10 static slots."""
    if T > 4 * P or K != 8 or T % 16 or E > 4 * P:
        return False
    return n_slots(T, K, E, lanes or default_lanes(T))[0] <= SLOTS10


def _kernel_inputs10(x, topv, topi, blob, lanes, act, limit, alpha):
    T, H = x.shape
    K = topi.shape[1]
    if not fits10(T, K, blob.shape[0], lanes):
        raise ValueError(f"kiln_moe_dedupe_v10 takes up to {4 * P} tokens (a multiple of 16) x top-8 of at most {4 * P} "
                         f"experts and {SLOTS10} slots per call, not T={T} K={K} E={blob.shape[0]}")
    L = lanes or default_lanes(T)
    S, BL = n_slots(T, K, blob.shape[0], L)
    q = group_size(L)
    return dict(x=x.contiguous(), topi=topi.to(torch.int32).contiguous(), topv=topv.to(torch.bfloat16).contiguous(),
                blob=blob, lanes=L, group=q, ring=RING9, slots=S, block=BL, keep=RING9 * q, act=act,
                limit=float(limit), alpha=float(alpha), debug=int(os.environ.get("KILN_MOE_DEDUPE_DEBUG", 0)), rev=REV10,
                spl=int(_lnc_split()), skp=SKIP9, hdf=HEAD9)


def fits(T: int, K: int, E: int, lanes: int | None = None) -> bool:
    """Whether one dedupe kernel call takes T tokens x top-K of E experts: v8 up to 128 tokens and 1024 pairs, v9 up to
    256 tokens with at most SLOTS9 static slots (GLM-5.3-Flash's 288 experts, top-8, lanes 8: 448 slots at T=192, 512 at
    256)."""
    if E > 4 * P or T > 2 * P:
        return False
    if T <= P and not V9_ALL:
        return T * K <= 1024
    return T * K <= 16 * P and n_slots(T, K, E, lanes or default_lanes(T))[0] <= SLOTS9


# The experts' gated activation (the kernels' static `act`): 0 SiLU, silu(gate) * up; 1 SiLU with
# both halves clamped at `limit` (GLM-5.3-Flash: Glm5NextTextExperts._apply_gate, models/hybrid.py
# _moe_clamped), silu(min(gate, limit)) * clamp(up, -limit, limit); 2 gpt-oss's (transformers
# GptOssExperts._apply_gate), g = min(gate, limit), (clamp(up, -limit, limit) + 1) * g *
# sigmoid(alpha g). The kernels compute exactly these in fp32 (expert biases are not supported).
ACTS = {"silu": 0, "silu_clamp": 1, "swiglu_oai": 2}


def glu(gate: torch.Tensor, up: torch.Tensor, act: int = 0, limit: float = 0.0, alpha: float = 1.702):
    """The activation `act` (ACTS) of the gate and up halves, in torch."""
    import torch.nn.functional as F

    if act == 0:
        return F.silu(gate) * up
    g, u = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
    if act == 1:
        return F.silu(g) * u
    return (u + 1) * (g * torch.sigmoid(g * alpha))


def emulate(x: torch.Tensor, topv: torch.Tensor, topi: torch.Tensor, blob: torch.Tensor, act: int = 0,
            limit: float = 0.0, alpha: float = 1.702, pair_bf16: bool = True) -> torch.Tensor:
    """The kernels' arithmetic in torch: per pair, the gate_up tile dot products in fp32 times the
    tile scales, g rounded to bf16, a = glu(gate, up) rounded to bf16, the down product per output
    column times its scale; the pair's output rounded to bf16 (pair_bf16: the dedupe kernel, whose
    transposes and route matmul take bf16; the per-pair kernel keeps it fp32), then the fp32 sum over
    the token's k experts of weight x output."""
    T, H = x.shape
    K = topi.shape[1]
    N, im = T * K, P // 2
    C = H // P
    sb, sdt = _sdt(blob, H)
    o1, o2, o3 = _offsets(H, sb)
    b = blob[topi.reshape(N)]  # [N, 128, F]
    w_gu = _as(b[..., :o1], FP8).float().view(N, P, C, P).permute(0, 3, 2, 1)  # [N, o, c, i]
    sg = _as(b[..., o2:o3], sdt).float()  # [N, o, c]
    xs = x.float().repeat_interleave(K, dim=0).view(N, 1, C, P)
    g = ((w_gu * xs).sum(-1) * sg).sum(-1).bfloat16().float()  # [N, 128]
    a = glu(g[:, :im], g[:, im:], act, limit, alpha).bfloat16().float()  # [N, im]
    _, s_d = unpack_down(b, H)  # tile scale per output column [N, 1 or 2, H]
    wd = _as(b[..., o1:o2], FP8).float().view(N, 2, im, H // 2).permute(0, 2, 1, 3).reshape(N, im, H)
    y = (a.unsqueeze(-1) * wd).sum(1) * s_d[:, 0].float()  # [N, H]
    if pair_bf16:
        y = y.bfloat16().float()
    return (y * topv.reshape(N, 1).float()).view(T, K, H).sum(1).to(x.dtype)


# --- kernel -----------------------------------------------------------------------------------

try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

# Defined at import, not inside the traced code: dynamo cannot trace a kernel object created
# while it is tracing.
if nki is not None:
    @nki.jit
    def kiln_moe_dedupe_v8(x, topi, topv, blob, lanes: int, group: int, ring: int, slots: int, block: int,
                           keep: int, act: int = 0, limit: float = 0.0, alpha: float = 1.702, debug: int = 0,
                           rev: int = 0, spl: int = 1):
        """x bf16 [T, H] (T <= 128), topi int32 [T, K] and topv bf16 [T, K] (the routing), blob
        uint8 [E, 128, F] (tile layout, bf16 or fp32 tile scales by F); `act` / `limit` / `alpha`
        the activation (ACTS); `lanes` pairs per slot, `slots`
        slots in blocks of `block` lanes (n_slots), `group` slots per vector instruction, `ring`
        groups of expert buffers (loads run ring - 2 groups ahead), padded slots below `keep` load
        expert 0 (plan()); rev: this function's source revision (REV, see _kernel_rev). Returns bf16
        [T, H]. Keep every helper inside it."""
        T, H = x.shape
        K = topi.shape[1]
        N = T * K
        E = blob.shape[0]
        NE = (E + 127) // 128  # tiles of experts on the partitions (rows past E match no pair)
        BL = block
        S = slots
        L = lanes
        Q = group
        NQ = S // Q  # groups
        QB_ = BL // (L * Q)  # groups per block
        F = blob.shape[2]
        C = H // 128
        DW = H // 2
        C2 = DW // 128
        G32 = H // 128  # output column groups of 128 (h // 128)
        NH = G32 // 4  # 512-column chunks of the output
        SB = (F - H - DW) // (2 * C)  # bytes per tile scale: 2 (bf16) or 4 (fp32)
        o_dw, o_sg, o_sd = H, H + DW, H + DW + C * SB
        f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
        sdt = nl.bfloat16 if SB == 2 else nl.float32
        LOG2L = 1 if L == 2 else 2 if L == 4 else 3 if L == 8 else 4 if L == 16 else 0
        out = nl.ndarray((T, H), dtype=bf16, buffer=nl.shared_hbm)
        # LNC (trn2 at LNC=2, grid 2): the kernel is traced once per program, the two physical cores of the
        # logical core (nki/_backends/mlir_tracer program_id: "kernel is traced LNC times with different
        # program_id_value"), so npg / pid are Python ints. With two or more blocks of lanes each program
        # runs a contiguous half of the blocks (their expert loads and matmuls), the fp32 partial sums are
        # exchanged by halves of H (nisa.sendrecv between the two cores, nki/isa/_lnc.py) and each program
        # writes its half of the output columns. Grid 1 (trn1), or one block: one program does everything.
        # spl 0 (KILN_LNC_SPLIT): both programs do all of the work, as before the split.
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if spl and nl.program_ndim() != 0 else (1, 0)
        NBK = S * L // BL  # blocks of lanes
        split = npg == 2 and NBK >= 2
        b_lo, b_hi = (0, NBK) if not split else ((0, (NBK + 1) // 2) if pid == 0 else ((NBK + 1) // 2, NBK))
        g_lo, g_hi = b_lo * QB_, b_hi * QB_  # this program's groups
        mine = split or pid == 0
        if split:  # allocated first, the same in both programs' traces: the peer's sendrecv lands here
            rcv = nl.ndarray((T, H // 2), dtype=nl.float32, buffer=nl.sbuf)

        x_sb = nl.ndarray((T, H), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_sb, src=x)

        # --- constants (iota and compares; no inputs) ---
        ipi = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ipi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        ip = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)  # partition index
        nisa.tensor_copy(dst=ip, src=ipi)
        jfi = nl.ndarray((128, 128), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=jfi, pattern=[[1, 128]], offset=0, channel_multiplier=0)
        dd = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)  # dd[p, j] = j - p
        nisa.tensor_copy(dst=dd, src=jfi)
        nisa.tensor_scalar(dst=dd, data=dd, op0=nl.subtract, operand0=ip)
        idn = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)  # identity
        nisa.tensor_scalar(dst=idn, data=dd, op0=nl.equal, operand0=0.0)
        up = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)  # [p < j]: exclusive prefix sums
        nisa.tensor_scalar(dst=up, data=dd, op0=nl.greater, operand0=0.0)
        f1 = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=f1, data=dd, op0=nl.equal, operand0=64.0)
        f2 = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=f2, data=dd, op0=nl.equal, operand0=-64.0)
        fd = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)  # fold[o, q] = [o % 64 == q % 64]
        nisa.tensor_tensor(dst=fd, data1=idn, data2=f1, op=nl.add)
        nisa.tensor_tensor(dst=fd, data1=fd, data2=f2, op=nl.add)
        mh = nl.ndarray((128, 2), dtype=f32, buffer=nl.sbuf)  # [p < 64, p >= 64]
        nisa.tensor_scalar(dst=mh[:, 0:1], data=ip, op0=nl.less, operand0=64.0)
        nisa.tensor_scalar(dst=mh[:, 1:2], data=ip, op0=nl.greater_equal, operand0=64.0)
        e_sb = nl.ndarray((1, S), dtype=i32, buffer=nl.sbuf)
        one_r = nl.ndarray((1, 128), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=one_r, value=1.0)
        ones_f = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ones_f, value=1.0)
        ones_b = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=ones_b, value=1.0)
        zero_n = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=zero_n, value=0.0)
        isi = nl.ndarray((128, S), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=isi, pattern=[[1, S]], offset=0, channel_multiplier=0)
        iota_s = nl.ndarray((128, S), dtype=f32, buffer=nl.sbuf)  # slot index on every partition
        nisa.tensor_copy(dst=iota_s, src=isi)

        # --- the routing plan (plan()'s arithmetic): every pair's lane, every slot's expert ---
        ti = nl.ndarray((1, N), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ti, src=topi.reshape((1, N)))
        tb = nl.ndarray((1, N), dtype=f32, buffer=nl.sbuf)  # expert ids (fp32: bf16 rounds past 256)
        nisa.tensor_copy(dst=tb, src=ti)
        wv = nl.ndarray((1, N), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wv, src=topv.reshape((1, N)))
        eb = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)  # e of pair n on every partition
        wb = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)  # its routing weight
        for n0 in range(0, N, 512):
            n1 = min(N, n0 + 512)
            pb = nl.ndarray((128, n1 - n0), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pb, stationary=one_r, moving=tb[:, n0:n1], accumulate=False)
            nisa.tensor_copy(dst=eb[:, n0:n1], src=pb)
            pw = nl.ndarray((128, n1 - n0), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pw, stationary=one_r, moving=wv[:, n0:n1], accumulate=False)
            nisa.tensor_copy(dst=wb[:, n0:n1], src=pw)
        # Per tile of 128 experts on the partitions (up to 4 tiles): counts first, then base, then a
        # second pass that recomputes each tile's one-hot for the ranks, so that only one [128, N]
        # one-hot is alive at a time.
        ns_t = (nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf))
        nb_t = (nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf),
                nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf))
        pe_t = (nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf))
        bs_t = (nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf))
        lb_t = (nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf))
        for et in range(NE):
            nisa.tensor_scalar(dst=pe_t[et], data=ip, op0=nl.add, operand0=128.0 * et)
            cnt = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
            oh = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar_reduce(dst=oh, data=eb, op0=nl.equal, operand0=pe_t[et], reduce_op=nl.add,
                                      reduce_res=cnt)
            if L == 1:
                nisa.tensor_copy(dst=ns_t[et], src=cnt)
            else:  # ceil(cnt / L) = (cnt + L - 1) >> log2 L, in int32
                ci = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ci, src=cnt)
                nisa.tensor_scalar(dst=ci, data=ci, op0=nl.add, operand0=L - 1)
                nisa.tensor_scalar(dst=ci, data=ci, op0=nl.right_shift, operand0=LOG2L)
                nisa.tensor_copy(dst=ns_t[et], src=ci)
            nisa.tensor_copy(dst=nb_t[et], src=ns_t[et])
        # base[e] = the slots of the experts before e (exclusive prefix sum over the partitions)
        for et in range(NE):
            pbs = nl.ndarray((128, 1), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pbs, stationary=up, moving=nb_t[et], accumulate=False)
            for t2 in range(et):
                nisa.nc_matmul(dst=pbs, stationary=ones_b, moving=nb_t[t2], accumulate=True)
            nisa.tensor_copy(dst=bs_t[et], src=pbs)
            nisa.tensor_scalar(dst=lb_t[et], data=bs_t[et], op0=nl.multiply, operand0=1.0 * L)
        # lane of pair n = L base[e_n] + its rank among the expert's pairs, summed over the experts
        # (one nonzero term) by fp32 matmuls, on every partition
        lane_bc = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
        NC = (N + 511) // 512
        pls = (nl.ndarray((128, min(N, 512)), dtype=f32, buffer=nl.psum), nl.ndarray((128, min(N, 512)), dtype=f32, buffer=nl.psum))
        for et in range(NE):
            oh = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=oh, data=eb, op0=nl.equal, operand0=pe_t[et])
            cs = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(dst=cs, data0=oh, data1=zero_n, initial=0.0, op0=nl.add, op1=nl.add)
            rk = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=rk, data=cs, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=oh)
            vv = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=vv, data=oh, op0=nl.multiply, operand0=lb_t[et], op1=nl.add, operand1=rk)
            for nc_ in range(NC):
                n0 = nc_ * 512
                n1 = min(N, n0 + 512)
                nisa.nc_matmul(dst=pls[nc_][:, 0:n1 - n0], stationary=ones_f, moving=vv[:, n0:n1], accumulate=et > 0)
        for nc_ in range(NC):
            n0 = nc_ * 512
            n1 = min(N, n0 + 512)
            nisa.tensor_copy(dst=lane_bc[:, n0:n1], src=pls[nc_][:, 0:n1 - n0])
        # slot s holds expert e iff base[e] <= s < base[e] + nslots[e]; none: E (DMA skipped), or
        # expert 0 below `keep` (a buffer's first use must be a real load)
        pse = nl.ndarray((1, S), dtype=f32, buffer=nl.psum)
        psr = nl.ndarray((1, S), dtype=f32, buffer=nl.psum)
        for et in range(NE):
            m1 = nl.ndarray((128, S), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m1, data=iota_s, op0=nl.greater_equal, operand0=bs_t[et])
            bn = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=bn, data1=bs_t[et], data2=ns_t[et], op=nl.add)
            mm_ = nl.ndarray((128, S), dtype=f32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=mm_, data=iota_s, op0=nl.less, operand0=bn, op1=nl.multiply, operand1=m1)
            nisa.nc_matmul(dst=pse, stationary=pe_t[et], moving=mm_, accumulate=et > 0)  # fp32: ids past 256
            nisa.nc_matmul(dst=psr, stationary=ones_f[:, 0:1], moving=mm_, accumulate=et > 0)
        late = nl.ndarray((1, S), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=late, data=iota_s[0:1, :], op0=nl.greater_equal, operand0=1.0 * keep)
        if split and g_lo > 0:  # this program's first `keep` slots are its buffers' first writes too
            s0 = 1.0 * g_lo * Q
            e2 = nl.ndarray((1, S), dtype=f32, buffer=nl.sbuf)  # [s0 <= s < s0 + keep]
            nisa.tensor_scalar(dst=e2, data=iota_s[0:1, :], op0=nl.greater_equal, operand0=s0)
            nisa.scalar_tensor_tensor(dst=e2, data=iota_s[0:1, :], op0=nl.less, operand0=s0 + keep,
                                      op1=nl.multiply, operand1=e2)
            nisa.tensor_scalar(dst=e2, data=e2, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
            nisa.tensor_tensor(dst=late, data1=late, data2=e2, op=nl.multiply)
        pad = nl.ndarray((1, S), dtype=f32, buffer=nl.sbuf)  # 1 - real
        nisa.tensor_scalar(dst=pad, data=psr, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
        nisa.scalar_tensor_tensor(dst=pad, data=pad, op0=nl.multiply, operand0=1.0 * E, op1=nl.multiply, operand1=late)
        sef = nl.ndarray((1, S), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sef, data1=pse, data2=pad, op=nl.add)
        nisa.tensor_copy(dst=e_sb, src=sef)

        acc = nl.ndarray((T, H), dtype=nl.float32, buffer=nl.sbuf)
        # Rings, one tensor per entry (the compiler orders accesses per tensor): NR groups of expert
        # buffers, two of each per-block tile (consecutive blocks overlap by the skew), three of
        # each per-group tile.
        NR = ring
        PD = NR - 2  # a group's buffers are loaded PD steps ahead and read until the step after its own
        bufs = (nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf),
                nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf),
                nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf))[:NR]
        gs = (nl.ndarray((T, BL), dtype=nl.bfloat16, buffer=nl.sbuf), nl.ndarray((T, BL), dtype=nl.bfloat16, buffer=nl.sbuf))
        rs = (nl.ndarray((BL, T), dtype=nl.bfloat16, buffer=nl.sbuf), nl.ndarray((BL, T), dtype=nl.bfloat16, buffer=nl.sbuf))
        xg = (nl.ndarray((128, C, BL), dtype=nl.bfloat16, buffer=nl.sbuf),  # x[tok(lane), c * 128 + i]
              nl.ndarray((128, C, BL), dtype=nl.bfloat16, buffer=nl.sbuf))
        yb = (nl.ndarray((128, G32, BL), dtype=nl.bfloat16, buffer=nl.sbuf),  # [p, h // 128, lane]
              nl.ndarray((128, G32, BL), dtype=nl.bfloat16, buffer=nl.sbuf))
        mmb = (nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf),  # [o, slot, lane, half] g masked
               nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf),
               nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf))
        a2b = (nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf),  # [q, slot, lane, half] a masked
               nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf),
               nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf))
        # the groups' scales, copied out of the expert buffers in stage A (by the scalar engine), so
        # that the buffers' last readers are the tensor engine's down matmuls and the next loads
        # into them wait on nothing else
        sg_r = (nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf), nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf),
                nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf))
        sd_r = (nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf), nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf),
                nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf))
        q_r = (nl.ndarray((128, Q, L, C), dtype=nl.float32, buffer=nl.sbuf),
               nl.ndarray((128, Q, L, C), dtype=nl.float32, buffer=nl.sbuf),
               nl.ndarray((128, Q, L, C), dtype=nl.float32, buffer=nl.sbuf))
        g_r = (nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf), nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf),
               nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf))
        sl_r = (nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf), nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf),
                nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf))
        au_r = (nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf), nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf),
                nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf))
        pg_r = (nl.ndarray((128, Q, C, L), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, C, L), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, C, L), dtype=nl.float32, buffer=nl.psum))
        pf_r = (nl.ndarray((128, Q, L, 2), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, L, 2), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, L, 2), dtype=nl.float32, buffer=nl.psum))
        pd_r = (nl.ndarray((128, Q, C2, L, 2), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, C2, L, 2), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, C2, L, 2), dtype=nl.float32, buffer=nl.psum))

        # Step i: the loads of group i + PD, stage A (gate_up, scales) of group i, then stage B
        # (fold, SiLU) and stage C (down, output) of group i - 1. The expert DMAs are the long
        # pole, so they run ahead; a group's buffers are free after its stage C.
        for it in range(g_lo - PD, (g_hi if mine else g_lo) + 1):
            gl = it + PD
            if not mine:
                break
            if g_lo <= gl < g_hi and debug % 2 == 0:  # loads of group gl (debug bit 1: none, to time the rest)
                bq = bufs[gl % NR]
                for s in range(Q):
                    e = e_sb.ap(pattern=[[S, 1], [1, 1]], offset=gl * Q + s)
                    nisa.dma_copy(dst=bq[:, s, :], src=blob.select(0, e), oob_mode=nisa.oob_mode.skip)
            if g_lo <= it < g_hi and it % QB_ == 0:  # gather the block's lanes' x: xg[i, c, l] = x[tok(l), c * 128 + i]
                bk = it // QB_
                pa = bk % 2
                g01 = nl.ndarray((BL, T), dtype=bf16, buffer=nl.sbuf)
                # the block's lanes on the partitions: eq[l, n] = [lane(n) == bk * BL + l]
                lid = nl.ndarray((BL, 1), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=lid, data=ip[0:BL, :], op0=nl.add, operand0=1.0 * bk * BL)
                eq = nl.ndarray((BL, T, K), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=eq.flatten_dims(1, 2), data=lane_bc[0:BL, :], op0=nl.equal, operand0=lid)
                nisa.tensor_reduce(dst=g01, op=nl.add, data=eq, axis=2)
                ew = nl.ndarray((BL, T, K), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=ew.flatten_dims(1, 2), data1=eq.flatten_dims(1, 2), data2=wb[0:BL, :],
                                   op=nl.multiply)
                nisa.tensor_reduce(dst=rs[pa], op=nl.add, data=ew, axis=2)  # R: weight of lane l's pair at token t
                pgt = nl.ndarray((T, BL), dtype=f32, buffer=nl.psum)  # G = R's 0/1 pattern, transposed
                nisa.nc_matmul(dst=pgt, stationary=g01, moving=idn[0:BL, 0:BL], accumulate=False)
                nisa.tensor_copy(dst=gs[pa], src=pgt)
                for c in range(C):
                    px = nl.ndarray((128, BL), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=px, stationary=x_sb[:, c * 128:(c + 1) * 128], moving=gs[pa], accumulate=False)
                    if c % 2 == 0:  # PSUM -> SBUF copies alternate between the scalar and vector engines
                        nisa.activation(dst=xg[pa][:, c, :], op=nl.copy, data=px)
                    else:
                        nisa.tensor_copy(dst=xg[pa][:, c, :], src=px, engine=nisa.engine.vector)
            if g_lo <= it < g_hi:  # stage A of group it
                gq = it
                pa = (gq // QB_) % 2
                l0 = (gq % QB_) * Q * L
                bq = bufs[gq % NR]
                pg = pg_r[gq % 3]
                for s in range(Q):
                    wg = bq[:, s, 0:H].view(nl.float8_e4m3)
                    for c in range(C):
                        nisa.nc_matmul(dst=pg[:, s, c, :], stationary=wg[:, c * 128:(c + 1) * 128],
                                       moving=xg[pa][:, c, l0 + s * L:l0 + (s + 1) * L], accumulate=False)
                nisa.activation(dst=sg_r[gq % 3], op=nl.copy, data=bq[:, :, o_sg:o_sd].view(sdt))
                nisa.activation(dst=sd_r[gq % 3], op=nl.copy, data=bq[:, :, o_sd:F].view(sdt))
                # g[o, s, l] = sum over tiles c of pg[o, s, c, l] * s_gu[o, c] of the slot's expert
                sg = sg_r[gq % 3].expand_dim(3).broadcast(3, L)  # [o, s, c, l]
                q = q_r[gq % 3]
                nisa.tensor_tensor(dst=q.permute((0, 1, 3, 2)), data1=pg, data2=sg, op=nl.multiply)
                g = g_r[gq % 3]
                nisa.tensor_reduce(dst=g, op=nl.add, data=q, axis=3)
                nisa.tensor_tensor(dst=mmb[gq % 3], data1=g.expand_dim(3).broadcast(3, 2),
                                   data2=mh.expand_dim(1).expand_dim(1).broadcast(1, Q).broadcast(2, L), op=nl.multiply)
            if g_lo + 1 <= it <= g_hi:  # stages B and C of group it - 1
                gq = it - 1
                bk = (gq * Q * L) // BL
                pa = bk % 2
                l0 = (gq % QB_) * Q * L
                # B: fold the up rows onto the gate rows, SiLU
                pf = pf_r[gq % 3]
                nisa.nc_matmul(dst=pf.flatten_dims(1, 3), stationary=fd, moving=mmb[gq % 3].flatten_dims(1, 3),
                               accumulate=False)
                sl = sl_r[gq % 3]
                au = au_r[gq % 3]
                if act == 0:  # silu(gate) * up
                    nisa.activation(dst=sl, op=nl.silu, data=pf[:, :, :, 0])
                    nisa.tensor_tensor(dst=au, data1=sl, data2=pf[:, :, :, 1], op=nl.multiply)
                else:  # gate clamped from above, up into [-limit, limit] (ACTS)
                    gc = nl.ndarray((128, Q, L), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=gc, data=pf[:, :, :, 0], op0=nl.minimum, operand0=limit)
                    uc = nl.ndarray((128, Q, L), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=uc, data=pf[:, :, :, 1], op0=nl.maximum, operand0=-limit, op1=nl.minimum,
                                       operand1=limit)
                    if act == 1:  # silu(gc) * uc
                        nisa.activation(dst=sl, op=nl.silu, data=gc)
                        nisa.tensor_tensor(dst=au, data1=sl, data2=uc, op=nl.multiply)
                    else:  # (uc + 1) * gc * sigmoid(alpha gc)
                        nisa.activation(dst=sl, op=nl.sigmoid, data=gc, scale=alpha)
                        nisa.tensor_tensor(dst=sl, data1=sl, data2=gc, op=nl.multiply)
                        nisa.scalar_tensor_tensor(dst=au, data=uc, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=sl)
                # a2[q, s, l, half] = a[q % 64, s, l] * [q // 64 == half]: the two H halves of
                # w_down are stacked on the partitions
                nisa.tensor_tensor(dst=a2b[gq % 3], data1=au.expand_dim(3).broadcast(3, 2),
                                   data2=mh.expand_dim(1).expand_dim(1).broadcast(1, Q).broadcast(2, L), op=nl.multiply)
                # C: down, scales, into the block's output columns
                bq = bufs[gq % NR]
                pd = pd_r[gq % 3]
                for s in range(Q):
                    wd = bq[:, s, o_dw:o_sg].view(nl.float8_e4m3)
                    for c in range(C2):
                        nisa.nc_matmul(dst=pd[:, s, c, :, :].flatten_dims(1, 2), stationary=wd[:, c * 128:(c + 1) * 128],
                                       moving=a2b[gq % 3][:, s, :, :].flatten_dims(1, 2), accumulate=False)
                # y[h], h = half * H/2 + c' * 128 + p: pd[p, s, c', l, half] * s_down[p, 2 c' + half],
                # into yb[p, half * C2 + c', lane]; one instruction per slot (each its own expert)
                for s in range(Q):
                    sd = sd_r[gq % 3][:, s, :].reshape_dim(1, (C2, 1, 2)).broadcast(2, L)
                    ybv = yb[pa][:, :, l0 + s * L:l0 + (s + 1) * L].reshape_dim(1, (2, C2)).permute((0, 2, 3, 1))
                    nisa.tensor_tensor(dst=ybv, data1=pd[:, s], data2=sd, op=nl.multiply)
                if gq % QB_ == QB_ - 1:  # the block's last group: route its lanes
                    yt = nl.ndarray((BL, G32, 128), dtype=nl.bfloat16, buffer=nl.sbuf)  # [lane, h // 128, p]
                    for g4 in range(NH):
                        pt = nl.ndarray((BL, 4, 128), dtype=nl.float32, buffer=nl.psum)
                        for k in range(4):
                            nisa.nc_matmul(dst=pt[:, k, :], stationary=yb[pa][:, g4 * 4 + k, :], moving=idn,
                                           accumulate=False)
                        if g4 % 2 == 0:
                            nisa.activation(dst=yt[:, g4 * 4:(g4 + 1) * 4, :], op=nl.copy, data=pt)
                        else:
                            nisa.tensor_copy(dst=yt[:, g4 * 4:(g4 + 1) * 4, :], src=pt, engine=nisa.engine.vector)
                    for hc in range(NH):
                        po = nl.ndarray((T, 512), dtype=nl.float32, buffer=nl.psum)
                        nisa.nc_matmul(dst=po, stationary=rs[pa], moving=yt[:, hc * 4:(hc + 1) * 4, :].flatten_dims(1, 2),
                                       accumulate=False)
                        if bk == b_lo:
                            nisa.tensor_copy(dst=acc[:, hc * 512:(hc + 1) * 512], src=po)
                        else:
                            nisa.tensor_tensor(dst=acc[:, hc * 512:(hc + 1) * 512],
                                               data1=acc[:, hc * 512:(hc + 1) * 512], data2=po, op=nl.add)
        if split:  # the other program's partial for this program's columns, added in fp32
            hw = H // 2
            oth = 1 - pid
            nisa.sendrecv(src=acc[:, oth * hw:(oth + 1) * hw], dst=rcv, send_to_rank=oth, recv_from_rank=oth,
                          pipe_id=0)
            mine_acc = acc[:, pid * hw:(pid + 1) * hw]
            nisa.tensor_tensor(dst=mine_acc, data1=mine_acc, data2=rcv, op=nl.add)
            o_h = nl.ndarray((T, hw), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o_h, src=mine_acc)
            nisa.dma_copy(dst=out[:, pid * hw:(pid + 1) * hw], src=o_h)
        elif pid == 0:
            o_sb = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=o_sb, src=acc)
            nisa.dma_copy(dst=out, src=o_sb)
        if npg > 1:  # LNC: the whole output written (both column halves, or program 0's) before either ends
            nisa.core_barrier(data=out, cores=(0, 1))
        return out
    @nki.jit
    def kiln_moe_tiles_pairs_v1(xT, topi, topv, blob, group: int, ring: int, act: int = 0, limit: float = 0.0,
                                alpha: float = 1.702):
        """The same experts and arithmetic per pair, one load per (token, expert) pair in order, for
        small calls: xT bf16 [128, T, C] (x[t, c * 128 + i] at [i, t, c]), topi int32 [T, K], topv
        bf16 [T, K], blob uint8 [E, 128, F] (tile layout). No plan, no gather, no route: a pair's
        token is static. The weighted outputs accumulate in fp32 per token, out bf16 [128, T, C]
        with y[t, h(p, g)] at [p, t, g] (as moe_decode's kernel; from_kernel permutes it).
        `group` pairs share each vector instruction; `ring` groups of expert buffers."""
        _, T, C = xT.shape
        K = topi.shape[1]
        N = T * K
        F = blob.shape[2]
        H = C * 128
        DW = H // 2
        C2 = DW // 128
        Q = group
        NQ = -(-N // Q)
        NP = NQ * Q  # pairs rounded to whole groups (the extra ones load expert 0, weight 0)
        SB = (F - H - DW) // (2 * C)  # bytes per tile scale: 2 (bf16) or 4 (fp32)
        o_dw, o_sg, o_sd = H, H + DW, H + DW + C * SB
        sdt = nl.bfloat16 if SB == 2 else nl.float32
        NR = ring
        PD = NR - 2
        f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
        out = nl.ndarray((128, T, C), dtype=bf16, buffer=nl.shared_hbm)

        x_sb = nl.ndarray((128, T, C), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_sb, src=xT)
        e_sb = nl.ndarray((1, NP), dtype=i32, buffer=nl.sbuf)
        nisa.memset(dst=e_sb, value=0)
        nisa.dma_copy(dst=e_sb[:, 0:N], src=topi.reshape((1, N)))
        # the pairs' weights on every partition (a DMA broadcast of the [1, N] row)
        wb = nl.ndarray((128, NP), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=wb, value=0.0)
        wr = nl.ndarray((1, N), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wr, src=topv.reshape((1, N)))
        one_r = nl.ndarray((1, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=one_r, value=1.0)
        pwb = nl.ndarray((128, N), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pwb, stationary=one_r, moving=wr, accumulate=False)
        nisa.tensor_copy(dst=wb[:, 0:N], src=pwb)
        # constants: fold[o, q] = [o % 64 == q % 64], mh = [p < 64, p >= 64]
        ipi = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ipi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        ip = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ip, src=ipi)
        jfi = nl.ndarray((128, 128), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=jfi, pattern=[[1, 128]], offset=0, channel_multiplier=0)
        dd = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dd, src=jfi)
        nisa.tensor_scalar(dst=dd, data=dd, op0=nl.subtract, operand0=ip)
        fd = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        f1 = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        f2 = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=fd, data=dd, op0=nl.equal, operand0=0.0)
        nisa.tensor_scalar(dst=f1, data=dd, op0=nl.equal, operand0=64.0)
        nisa.tensor_scalar(dst=f2, data=dd, op0=nl.equal, operand0=-64.0)
        nisa.tensor_tensor(dst=fd, data1=fd, data2=f1, op=nl.add)
        nisa.tensor_tensor(dst=fd, data1=fd, data2=f2, op=nl.add)
        mh = nl.ndarray((128, 2), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mh[:, 0:1], data=ip, op0=nl.less, operand0=64.0)
        nisa.tensor_scalar(dst=mh[:, 1:2], data=ip, op0=nl.greater_equal, operand0=64.0)
        acc = nl.ndarray((128, T, C), dtype=f32, buffer=nl.sbuf)  # [p, t, g]: g = 2 c' + half
        nisa.memset(dst=acc, value=0.0)

        bufs = (nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf),
                nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf),
                nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf))[:NR]
        sg_r = (nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf))
        sd_r = (nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf))
        q_r = (nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf),
               nl.ndarray((128, Q, C), dtype=f32, buffer=nl.sbuf))
        g_r = (nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf),
               nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf))
        mm_r = (nl.ndarray((128, Q, 2), dtype=bf16, buffer=nl.sbuf), nl.ndarray((128, Q, 2), dtype=bf16, buffer=nl.sbuf),
                nl.ndarray((128, Q, 2), dtype=bf16, buffer=nl.sbuf))
        sl_r = (nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf))
        au_r = (nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf))
        a2_r = (nl.ndarray((128, Q, 2), dtype=bf16, buffer=nl.sbuf), nl.ndarray((128, Q, 2), dtype=bf16, buffer=nl.sbuf),
                nl.ndarray((128, Q, 2), dtype=bf16, buffer=nl.sbuf))
        y_r = (nl.ndarray((128, Q, C2, 2), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, Q, C2, 2), dtype=f32, buffer=nl.sbuf),
               nl.ndarray((128, Q, C2, 2), dtype=f32, buffer=nl.sbuf))
        pg_r = (nl.ndarray((128, Q, C), dtype=f32, buffer=nl.psum), nl.ndarray((128, Q, C), dtype=f32, buffer=nl.psum),
                nl.ndarray((128, Q, C), dtype=f32, buffer=nl.psum))
        pf_r = (nl.ndarray((128, Q, 2), dtype=f32, buffer=nl.psum), nl.ndarray((128, Q, 2), dtype=f32, buffer=nl.psum),
                nl.ndarray((128, Q, 2), dtype=f32, buffer=nl.psum))
        pd_r = (nl.ndarray((128, Q, C2, 2), dtype=f32, buffer=nl.psum), nl.ndarray((128, Q, C2, 2), dtype=f32, buffer=nl.psum),
                nl.ndarray((128, Q, C2, 2), dtype=f32, buffer=nl.psum))

        # Step i: the loads of group i + PD, stage A of group i, stages B and C of group i - 1
        # (as kiln_moe_dedupe_v8, with one lane per slot).
        for it in range(-PD, NQ + 1):
            gl = it + PD
            if 0 <= gl < NQ:
                bq = bufs[gl % NR]
                for s in range(Q):
                    e = e_sb.ap(pattern=[[NP, 1], [1, 1]], offset=gl * Q + s)
                    nisa.dma_copy(dst=bq[:, s, :], src=blob.select(0, e))
            if 0 <= it < NQ:  # stage A: gate_up, scales
                gq = it
                bq = bufs[gq % NR]
                pg = pg_r[gq % 3]
                for s in range(Q):
                    t = min(T - 1, (gq * Q + s) // K)
                    wg = bq[:, s, 0:H].view(nl.float8_e4m3)
                    for c in range(C):
                        nisa.nc_matmul(dst=pg[:, s, c:c + 1], stationary=wg[:, c * 128:(c + 1) * 128],
                                       moving=x_sb[:, t, c:c + 1], accumulate=False)
                nisa.activation(dst=sg_r[gq % 3], op=nl.copy, data=bq[:, :, o_sg:o_sd].view(sdt))
                nisa.activation(dst=sd_r[gq % 3], op=nl.copy, data=bq[:, :, o_sd:F].view(sdt))
                q = q_r[gq % 3]
                nisa.tensor_tensor(dst=q, data1=pg, data2=sg_r[gq % 3], op=nl.multiply)
                g = g_r[gq % 3]
                nisa.tensor_reduce(dst=g, op=nl.add, data=q, axis=2)
                nisa.tensor_tensor(dst=mm_r[gq % 3], data1=g.expand_dim(2).broadcast(2, 2),
                                   data2=mh.expand_dim(1).broadcast(1, Q), op=nl.multiply)
            if 1 <= it <= NQ:  # stages B and C of group it - 1
                gq = it - 1
                pf = pf_r[gq % 3]
                nisa.nc_matmul(dst=pf.flatten_dims(1, 2), stationary=fd, moving=mm_r[gq % 3].flatten_dims(1, 2),
                               accumulate=False)
                sl = sl_r[gq % 3]
                au = au_r[gq % 3]
                if act == 0:  # silu(gate) * up
                    nisa.activation(dst=sl, op=nl.silu, data=pf[:, :, 0])
                    nisa.tensor_tensor(dst=au, data1=sl, data2=pf[:, :, 1], op=nl.multiply)
                else:  # gate clamped from above, up into [-limit, limit] (ACTS)
                    gc = nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=gc, data=pf[:, :, 0], op0=nl.minimum, operand0=limit)
                    uc = nl.ndarray((128, Q), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=uc, data=pf[:, :, 1], op0=nl.maximum, operand0=-limit, op1=nl.minimum,
                                       operand1=limit)
                    if act == 1:  # silu(gc) * uc
                        nisa.activation(dst=sl, op=nl.silu, data=gc)
                        nisa.tensor_tensor(dst=au, data1=sl, data2=uc, op=nl.multiply)
                    else:  # (uc + 1) * gc * sigmoid(alpha gc)
                        nisa.activation(dst=sl, op=nl.sigmoid, data=gc, scale=alpha)
                        nisa.tensor_tensor(dst=sl, data1=sl, data2=gc, op=nl.multiply)
                        nisa.scalar_tensor_tensor(dst=au, data=uc, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=sl)
                nisa.tensor_tensor(dst=a2_r[gq % 3], data1=au.expand_dim(2).broadcast(2, 2),
                                   data2=mh.expand_dim(1).broadcast(1, Q), op=nl.multiply)
                bq = bufs[gq % NR]
                pd = pd_r[gq % 3]
                for s in range(Q):
                    wd = bq[:, s, o_dw:o_sg].view(nl.float8_e4m3)
                    for c in range(C2):
                        nisa.nc_matmul(dst=pd[:, s, c, :], stationary=wd[:, c * 128:(c + 1) * 128],
                                       moving=a2_r[gq % 3][:, s, :], accumulate=False)
                # y[p, s, c', half] = pd * s_down[p, 2 c' + half]; acc[p, t, :] += w * y
                y = y_r[gq % 3]
                nisa.tensor_tensor(dst=y.flatten_dims(2, 3), data1=pd.flatten_dims(2, 3), data2=sd_r[gq % 3],
                                   op=nl.multiply)
                for s in range(Q):
                    u = gq * Q + s
                    if u < N:
                        t = u // K
                        nisa.scalar_tensor_tensor(dst=acc[:, t, :], data=y[:, s].flatten_dims(1, 2), op0=nl.multiply,
                                                  operand0=wb[:, u:u + 1], op1=nl.add, operand1=acc[:, t, :])
        o_sb = nl.ndarray((128, T, C), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=o_sb, src=acc)
        nisa.dma_copy(dst=out, src=o_sb)
        return out
    @nki.jit
    def kiln_moe_dedupe_v9(x, topi, topv, blob, lanes: int, group: int, ring: int, slots: int, block: int,
                           keep: int, act: int = 0, limit: float = 0.0, alpha: float = 1.702, debug: int = 0,
                           rev: int = 0, spl: int = 1, skp: int = 0, hdf: int = 0):
        """kiln_moe_dedupe_v8 for 128 < T <= 256 tokens in ONE call, so each selected expert is read once for all of
        them (v8 holds a call's tokens on the partitions, and moe_dedupe's 128-token chunks each read nearly every expert:
        at 192 rows two calls of 384 + 352 static slots against one of 448). The tokens sit in TT = 2 tiles of 128
        partitions wherever v8 has them on the partitions (x, the lanes' 0/1 gather matrix, the route's output, the
        fp32 accumulator and the LNC exchange); the gather's matmuls accumulate over the tiles and the route runs once per
        tile. Everything else (the plan over N = T K <= 2048 pairs, the slots, the pipeline, the LNC split) is v8's,
        with the vector engine (the busy one: 1.65 of the call's 1.9 ms at T = 192 in v8's form, tools/prof_ops.py)
        unloaded and the in-order tensor engine kept off its waits: a block's gather is issued one block ahead, its
        vector half (the lanes' 0/1 and weight matrices) in the step after the previous block's first group and its
        tensor-engine half (transposes, the lanes' x) QB_ / 2 steps later; a pair of blocks' route matmuls run two steps
        after their transposes; the PSUM -> SBUF copies of the gather and of the route's transposes run on the scalar engine, the SBUF-only
        products (the routing weights per lane, the masked g and a) on GpSimd, and two consecutive blocks' routes are
        summed in PSUM, so the fp32 accumulator is read and written once per two blocks.
        skp > 0 (one program only, no LNC split): the blocks past those every routing fills (the first ceil(N / L)
        slots) run in segments of skp blocks, each its own pipeline (_dd9_region) in a device loop of trip count [the
        segment's first slot is a real one]: slots are compacted, so a segment past the routing's real slots does not
        run (448 static slots at 192 rows, ~314 real at uniform routing). hdf: the static head is at least hdf% of the
        slots (real routing fills 67-79% of them at 128-256 rows, tools/dedupe_slot_stats.py), so the segments past it
        rarely run.
        x bf16 [T, H] (T <= 256), topi int32 [T, K] and topv bf16 [T, K] (the routing), blob
        uint8 [E, 128, F] (tile layout, bf16 or fp32 tile scales by F); `act` / `limit` / `alpha`
        the activation (ACTS); `lanes` pairs per slot, `slots`
        slots in blocks of `block` lanes (n_slots), `group` slots per vector instruction, `ring`
        groups of expert buffers (loads run ring - 2 groups ahead), padded slots below `keep` load
        expert 0 (plan()); rev: this function's source revision (REV, see _kernel_rev). Returns bf16
        [T, H]. Keep every helper inside it."""
        T, H = x.shape
        K = topi.shape[1]
        N = T * K
        TT = (T + 127) // 128  # token tiles on the partitions: token t = 128 tt + p at [p, tt]
        E = blob.shape[0]
        NE = (E + 127) // 128  # tiles of experts on the partitions (rows past E match no pair)
        BL = block
        S = slots
        L = lanes
        Q = group
        NQ = S // Q  # groups
        QB_ = BL // (L * Q)  # groups per block
        F = blob.shape[2]
        C = H // 128
        DW = H // 2
        C2 = DW // 128
        G32 = H // 128  # output column groups of 128 (h // 128)
        NH = G32 // 4  # 512-column chunks of the output
        SB = (F - H - DW) // (2 * C)  # bytes per tile scale: 2 (bf16) or 4 (fp32)
        o_dw, o_sg, o_sd = H, H + DW, H + DW + C * SB
        f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
        sdt = nl.bfloat16 if SB == 2 else nl.float32
        LOG2L = 1 if L == 2 else 2 if L == 4 else 3 if L == 8 else 4 if L == 16 else 0
        out = nl.ndarray((T, H), dtype=bf16, buffer=nl.shared_hbm)
        # LNC (trn2 at LNC=2, grid 2): the kernel is traced once per program, the two physical cores of the
        # logical core (nki/_backends/mlir_tracer program_id: "kernel is traced LNC times with different
        # program_id_value"), so npg / pid are Python ints. With two or more blocks of lanes each program
        # runs a contiguous half of the blocks (their expert loads and matmuls), the fp32 partial sums are
        # exchanged by halves of H (nisa.sendrecv between the two cores, nki/isa/_lnc.py) and each program
        # writes its half of the output columns. Grid 1 (trn1), or one block: one program does everything.
        # spl 0 (KILN_LNC_SPLIT): both programs do all of the work, as before the split.
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if spl and nl.program_ndim() != 0 else (1, 0)
        NBK = S * L // BL  # blocks of lanes
        split = npg == 2 and NBK >= 2
        b_lo, b_hi = (0, NBK) if not split else ((0, (NBK + 1) // 2) if pid == 0 else ((NBK + 1) // 2, NBK))
        g_lo, g_hi = b_lo * QB_, b_hi * QB_  # this program's groups
        mine = split or pid == 0
        if split:  # allocated first, the same in both programs' traces: the peer's sendrecv lands here
            rcv = nl.ndarray((128, TT, H // 2), dtype=nl.float32, buffer=nl.sbuf)

        x_sb = nl.ndarray((128, TT, H), dtype=bf16, buffer=nl.sbuf)
        for tt in range(TT):
            tn = min(128, T - tt * 128)
            nisa.dma_copy(dst=x_sb[0:tn, tt, :], src=x[tt * 128:tt * 128 + tn, :])

        # --- constants (iota and compares; no inputs) ---
        ipi = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ipi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        ip = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)  # partition index
        nisa.tensor_copy(dst=ip, src=ipi)
        jfi = nl.ndarray((128, 128), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=jfi, pattern=[[1, 128]], offset=0, channel_multiplier=0)
        dd = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)  # dd[p, j] = j - p
        nisa.tensor_copy(dst=dd, src=jfi)
        nisa.tensor_scalar(dst=dd, data=dd, op0=nl.subtract, operand0=ip)
        idn = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)  # identity
        nisa.tensor_scalar(dst=idn, data=dd, op0=nl.equal, operand0=0.0)
        up = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)  # [p < j]: exclusive prefix sums
        nisa.tensor_scalar(dst=up, data=dd, op0=nl.greater, operand0=0.0)
        f1 = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=f1, data=dd, op0=nl.equal, operand0=64.0)
        f2 = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=f2, data=dd, op0=nl.equal, operand0=-64.0)
        fd = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)  # fold[o, q] = [o % 64 == q % 64]
        nisa.tensor_tensor(dst=fd, data1=idn, data2=f1, op=nl.add)
        nisa.tensor_tensor(dst=fd, data1=fd, data2=f2, op=nl.add)
        mh = nl.ndarray((128, 2), dtype=f32, buffer=nl.sbuf)  # [p < 64, p >= 64]
        nisa.tensor_scalar(dst=mh[:, 0:1], data=ip, op0=nl.less, operand0=64.0)
        nisa.tensor_scalar(dst=mh[:, 1:2], data=ip, op0=nl.greater_equal, operand0=64.0)
        e_sb = nl.ndarray((1, S), dtype=i32, buffer=nl.sbuf)
        one_r = nl.ndarray((1, 128), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=one_r, value=1.0)
        ones_f = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ones_f, value=1.0)
        ones_b = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=ones_b, value=1.0)
        zero_n = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=zero_n, value=0.0)
        isi = nl.ndarray((128, S), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=isi, pattern=[[1, S]], offset=0, channel_multiplier=0)
        iota_s = nl.ndarray((128, S), dtype=f32, buffer=nl.sbuf)  # slot index on every partition
        nisa.tensor_copy(dst=iota_s, src=isi)

        # --- the routing plan (plan()'s arithmetic): every pair's lane, every slot's expert ---
        ti = nl.ndarray((1, N), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ti, src=topi.reshape((1, N)))
        tb = nl.ndarray((1, N), dtype=f32, buffer=nl.sbuf)  # expert ids (fp32: bf16 rounds past 256)
        nisa.tensor_copy(dst=tb, src=ti)
        wv = nl.ndarray((1, N), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wv, src=topv.reshape((1, N)))
        eb = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)  # e of pair n on every partition
        wb = nl.ndarray((128, N), dtype=bf16, buffer=nl.sbuf)  # its routing weight (bf16 as given: exact)
        for n0 in range(0, N, 512):
            n1 = min(N, n0 + 512)
            pb = nl.ndarray((128, n1 - n0), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pb, stationary=one_r, moving=tb[:, n0:n1], accumulate=False)
            nisa.tensor_copy(dst=eb[:, n0:n1], src=pb)
            pw = nl.ndarray((128, n1 - n0), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pw, stationary=one_r, moving=wv[:, n0:n1], accumulate=False)
            nisa.tensor_copy(dst=wb[:, n0:n1], src=pw)
        # Per tile of 128 experts on the partitions (up to 4 tiles): counts first, then base, then a
        # second pass that recomputes each tile's one-hot for the ranks, so that only one [128, N]
        # one-hot is alive at a time.
        ns_t = (nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf))
        nb_t = (nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf),
                nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf))
        pe_t = (nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf))
        bs_t = (nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf))
        lb_t = (nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf))
        for et in range(NE):
            nisa.tensor_scalar(dst=pe_t[et], data=ip, op0=nl.add, operand0=128.0 * et)
            cnt = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
            oh = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar_reduce(dst=oh, data=eb, op0=nl.equal, operand0=pe_t[et], reduce_op=nl.add,
                                      reduce_res=cnt)
            if L == 1:
                nisa.tensor_copy(dst=ns_t[et], src=cnt)
            else:  # ceil(cnt / L) = (cnt + L - 1) >> log2 L, in int32
                ci = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ci, src=cnt)
                nisa.tensor_scalar(dst=ci, data=ci, op0=nl.add, operand0=L - 1)
                nisa.tensor_scalar(dst=ci, data=ci, op0=nl.right_shift, operand0=LOG2L)
                nisa.tensor_copy(dst=ns_t[et], src=ci)
            nisa.tensor_copy(dst=nb_t[et], src=ns_t[et])
        # base[e] = the slots of the experts before e (exclusive prefix sum over the partitions)
        for et in range(NE):
            pbs = nl.ndarray((128, 1), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pbs, stationary=up, moving=nb_t[et], accumulate=False)
            for t2 in range(et):
                nisa.nc_matmul(dst=pbs, stationary=ones_b, moving=nb_t[t2], accumulate=True)
            nisa.tensor_copy(dst=bs_t[et], src=pbs)
            nisa.tensor_scalar(dst=lb_t[et], data=bs_t[et], op0=nl.multiply, operand0=1.0 * L)
        # lane of pair n = L base[e_n] + its rank among the expert's pairs, summed over the experts
        # (one nonzero term) by fp32 matmuls, on every partition
        lane_bc = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
        NC = (N + 511) // 512
        pls = []
        for _ in range(NC):
            pls.append(nl.ndarray((128, min(N, 512)), dtype=f32, buffer=nl.psum))
        for et in range(NE):
            oh = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=oh, data=eb, op0=nl.equal, operand0=pe_t[et])
            cs = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(dst=cs, data0=oh, data1=zero_n, initial=0.0, op0=nl.add, op1=nl.add)
            rk = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=rk, data=cs, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=oh)
            vv = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=vv, data=oh, op0=nl.multiply, operand0=lb_t[et], op1=nl.add, operand1=rk)
            for nc_ in range(NC):
                n0 = nc_ * 512
                n1 = min(N, n0 + 512)
                nisa.nc_matmul(dst=pls[nc_][:, 0:n1 - n0], stationary=ones_f, moving=vv[:, n0:n1], accumulate=et > 0)
        for nc_ in range(NC):
            n0 = nc_ * 512
            n1 = min(N, n0 + 512)
            nisa.tensor_copy(dst=lane_bc[:, n0:n1], src=pls[nc_][:, 0:n1 - n0])
        # slot s holds expert e iff base[e] <= s < base[e] + nslots[e]; none: E (DMA skipped), or
        # expert 0 below `keep` (a buffer's first use must be a real load)
        pse = nl.ndarray((1, S), dtype=f32, buffer=nl.psum)
        psr = nl.ndarray((1, S), dtype=f32, buffer=nl.psum)
        for et in range(NE):
            m1 = nl.ndarray((128, S), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m1, data=iota_s, op0=nl.greater_equal, operand0=bs_t[et])
            bn = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=bn, data1=bs_t[et], data2=ns_t[et], op=nl.add)
            mm_ = nl.ndarray((128, S), dtype=f32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=mm_, data=iota_s, op0=nl.less, operand0=bn, op1=nl.multiply, operand1=m1)
            nisa.nc_matmul(dst=pse, stationary=pe_t[et], moving=mm_, accumulate=et > 0)  # fp32: ids past 256
            nisa.nc_matmul(dst=psr, stationary=ones_f[:, 0:1], moving=mm_, accumulate=et > 0)
        late = nl.ndarray((1, S), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=late, data=iota_s[0:1, :], op0=nl.greater_equal, operand0=1.0 * keep)
        if split and g_lo > 0:  # this program's first `keep` slots are its buffers' first writes too
            s0 = 1.0 * g_lo * Q
            e2 = nl.ndarray((1, S), dtype=f32, buffer=nl.sbuf)  # [s0 <= s < s0 + keep]
            nisa.tensor_scalar(dst=e2, data=iota_s[0:1, :], op0=nl.greater_equal, operand0=s0)
            nisa.scalar_tensor_tensor(dst=e2, data=iota_s[0:1, :], op0=nl.less, operand0=s0 + keep,
                                      op1=nl.multiply, operand1=e2)
            nisa.tensor_scalar(dst=e2, data=e2, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
            nisa.tensor_tensor(dst=late, data1=late, data2=e2, op=nl.multiply)
        pad = nl.ndarray((1, S), dtype=f32, buffer=nl.sbuf)  # 1 - real
        nisa.tensor_scalar(dst=pad, data=psr, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
        nisa.scalar_tensor_tensor(dst=pad, data=pad, op0=nl.multiply, operand0=1.0 * E, op1=nl.multiply, operand1=late)
        sef = nl.ndarray((1, S), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sef, data1=pse, data2=pad, op=nl.add)
        nisa.tensor_copy(dst=e_sb, src=sef)

        acc = nl.ndarray((128, TT, H), dtype=nl.float32, buffer=nl.sbuf)
        # Rings, one tensor per entry (the compiler orders accesses per tensor): NR groups of expert
        # buffers, two of each per-block tile (consecutive blocks overlap by the skew), three of
        # each per-group tile.
        NR = ring
        PD = NR - 2  # a group's buffers are loaded PD steps ahead and read until the step after its own
        bufs = (nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf),
                nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf),
                nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf))[:NR]
        gs = (nl.ndarray((128, TT, BL), dtype=nl.bfloat16, buffer=nl.sbuf),  # G: [token p of tile tt, lane]
              nl.ndarray((128, TT, BL), dtype=nl.bfloat16, buffer=nl.sbuf))
        g01s = (nl.ndarray((BL, T), dtype=nl.bfloat16, buffer=nl.sbuf), nl.ndarray((BL, T), dtype=nl.bfloat16, buffer=nl.sbuf))
        rs = (nl.ndarray((BL, T), dtype=nl.bfloat16, buffer=nl.sbuf), nl.ndarray((BL, T), dtype=nl.bfloat16, buffer=nl.sbuf),
              nl.ndarray((BL, T), dtype=nl.bfloat16, buffer=nl.sbuf), nl.ndarray((BL, T), dtype=nl.bfloat16, buffer=nl.sbuf))
        xg = (nl.ndarray((128, C, BL), dtype=nl.bfloat16, buffer=nl.sbuf),  # x[tok(lane), c * 128 + i]
              nl.ndarray((128, C, BL), dtype=nl.bfloat16, buffer=nl.sbuf))
        yb = (nl.ndarray((128, G32, BL), dtype=nl.bfloat16, buffer=nl.sbuf),  # [p, h // 128, lane]
              nl.ndarray((128, G32, BL), dtype=nl.bfloat16, buffer=nl.sbuf))
        mmb = (nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf),  # [o, slot, lane, half] g masked
               nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf),
               nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf))
        a2b = (nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf),  # [q, slot, lane, half] a masked
               nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf),
               nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf))
        # the groups' scales, copied out of the expert buffers in stage A (by the scalar engine), so
        # that the buffers' last readers are the tensor engine's down matmuls and the next loads
        # into them wait on nothing else
        sg_r = (nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf), nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf),
                nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf))
        sd_r = (nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf), nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf),
                nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf))
        q_r = (nl.ndarray((128, Q, L, C), dtype=nl.float32, buffer=nl.sbuf),
               nl.ndarray((128, Q, L, C), dtype=nl.float32, buffer=nl.sbuf),
               nl.ndarray((128, Q, L, C), dtype=nl.float32, buffer=nl.sbuf))
        g_r = (nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf), nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf),
               nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf))
        sl_r = (nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf), nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf),
                nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf))
        au_r = (nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf), nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf),
                nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf))
        St = dict(T=T, TT=TT, K=K, BL=BL, L=L, Q=Q, QB_=QB_, C=C, C2=C2, G32=G32, NH=NH, H=H, F=F, S=S, NR=NR, PD=PD,
                  o_dw=o_dw, o_sg=o_sg, o_sd=o_sd, sdt=sdt, act=act, limit=limit, alpha=alpha, debug=debug, blob=blob,
                  e_sb=e_sb, x_sb=x_sb, lane_bc=lane_bc, wb=wb, ip=ip, idn=idn, fd=fd, mh=mh, acc=acc, bufs=bufs,
                  g01s=g01s, rs=rs, gs=gs, xg=xg, yb=yb, mmb=mmb, a2b=a2b, sg_r=sg_r, sd_r=sd_r, q_r=q_r, g_r=g_r,
                  sl_r=sl_r, au_r=au_r, b_first=b_lo)
        SPB = BL // L  # slots per block
        NHD = -(-(-(-N // L)) // SPB)  # blocks that hold one of the first ceil(N / L) slots: every routing fills them
        NHD = max(NHD, -(-(S * hdf // 100) // SPB))  # hdf: a static head of hdf% of the slots (the routing's usual fill)
        NHD = NHD + NHD % 2  # pairs of blocks share a route
        SEG = skp + skp % 2
        hd = min(b_hi, b_lo + NHD)
        if mine and SEG > 0 and not split and npg == 1 and hd < b_hi:
            KS = -(-(b_hi - hd) // SEG)
            fkf = nl.ndarray((1, KS), dtype=f32, buffer=nl.sbuf)  # [segment k's first slot is real]
            for k in range(KS):
                s0 = (hd + k * SEG) * SPB
                nisa.tensor_copy(dst=fkf[0:1, k:k + 1], src=psr[0:1, s0:s0 + 1])
            fli = nl.ndarray((1, KS), dtype=i32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=fli, src=fkf)
            _dd9_region(St, b_lo, hd)
            for k in range(KS):
                b0 = hd + k * SEG
                b1 = min(b_hi, b0 + SEG)
                rk = nisa.register_alloc()
                nisa.register_load(rk, fli.ap(pattern=[[KS, 1], [1, 1]], offset=k))

                def seg(it, b0=b0, b1=b1):
                    _dd9_region(St, b0, b1)

                nl.fori_loop(0, rk, seg)
        elif mine:
            _dd9_region(St, b_lo, b_hi)
        if split:  # the other program's partial for this program's columns, added in fp32, per token tile
            hw = H // 2
            oth = 1 - pid
            for tt in range(TT):
                tn = min(128, T - tt * 128)
                nisa.sendrecv(src=acc[0:tn, tt, oth * hw:(oth + 1) * hw], dst=rcv[0:tn, tt, :], send_to_rank=oth,
                              recv_from_rank=oth, pipe_id=0)
            for tt in range(TT):
                tn = min(128, T - tt * 128)
                mine_acc = acc[0:tn, tt, pid * hw:(pid + 1) * hw]
                nisa.tensor_tensor(dst=mine_acc, data1=mine_acc, data2=rcv[0:tn, tt, :], op=nl.add)
                o_h = nl.ndarray((128, hw), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=o_h[0:tn, :], src=mine_acc)
                nisa.dma_copy(dst=out[tt * 128:tt * 128 + tn, pid * hw:(pid + 1) * hw], src=o_h[0:tn, :])
        elif pid == 0:
            for tt in range(TT):
                tn = min(128, T - tt * 128)
                o_sb = nl.ndarray((128, H), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=o_sb[0:tn, :], src=acc[0:tn, tt, :])
                nisa.dma_copy(dst=out[tt * 128:tt * 128 + tn, :], src=o_sb[0:tn, :])
        if npg > 1:  # LNC: the whole output written (both column halves, or program 0's) before either ends
            nisa.core_barrier(data=out, cores=(0, 1))
        return out

    def _dd9_region(St, b_lo, b_hi):
        """kiln_moe_dedupe_v9's blocks b_lo .. b_hi - 1 as one software pipeline (loads, gathers, stages A / B / C,
        routes), with PSUM rings of its own: the whole call, its static head, or the body of one of its device loops
        (a PSUM tile referenced in two device-loop regions fails to compile: [NCC_IBIR092], docs/neuron-notes.md
        "Prefill MoE as a grouped GEMM"). Blocks pair up for their routes from b_lo, so a region holds its pairs."""
        T = St["T"]
        TT = St["TT"]
        K = St["K"]
        BL = St["BL"]
        L = St["L"]
        Q = St["Q"]
        QB_ = St["QB_"]
        C = St["C"]
        C2 = St["C2"]
        G32 = St["G32"]
        NH = St["NH"]
        H = St["H"]
        F = St["F"]
        S = St["S"]
        NR = St["NR"]
        PD = St["PD"]
        o_dw = St["o_dw"]
        o_sg = St["o_sg"]
        o_sd = St["o_sd"]
        sdt = St["sdt"]
        act = St["act"]
        limit = St["limit"]
        alpha = St["alpha"]
        debug = St["debug"]
        blob = St["blob"]
        e_sb = St["e_sb"]
        x_sb = St["x_sb"]
        lane_bc = St["lane_bc"]
        wb = St["wb"]
        ip = St["ip"]
        idn = St["idn"]
        fd = St["fd"]
        mh = St["mh"]
        acc = St["acc"]
        bufs = St["bufs"]
        g01s = St["g01s"]
        rs = St["rs"]
        gs = St["gs"]
        xg = St["xg"]
        yb = St["yb"]
        mmb = St["mmb"]
        a2b = St["a2b"]
        sg_r = St["sg_r"]
        sd_r = St["sd_r"]
        q_r = St["q_r"]
        g_r = St["g_r"]
        sl_r = St["sl_r"]
        au_r = St["au_r"]
        f32 = nl.float32
        g_lo = b_lo * QB_
        g_hi = b_hi * QB_
        first = b_lo == St["b_first"]  # this program's first region: its first route writes the accumulator
        pg_r = (nl.ndarray((128, Q, C, L), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, C, L), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, C, L), dtype=nl.float32, buffer=nl.psum))
        pf_r = (nl.ndarray((128, Q, L, 2), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, L, 2), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, L, 2), dtype=nl.float32, buffer=nl.psum))
        pd_r = (nl.ndarray((128, Q, C2, L, 2), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, C2, L, 2), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, C2, L, 2), dtype=nl.float32, buffer=nl.psum))

        # Step i: the loads of group i + PD, stage A (gate_up, scales) of group i, then stage B
        # (fold, SiLU) and stage C (down, output) of group i - 1. The expert DMAs are the long
        # pole, so they run ahead; a group's buffers are free after its stage C.
        DG = QB_ // 2 if QB_ > 1 else 0  # steps from a gather's vector half to its tensor-engine half
        DR = 2  # steps from a route's transposes to its matmuls
        it0 = g_lo - PD
        pend = []  # the pending route: [step, blocks, their transposed outputs, rel]
        for it in range(it0, g_hi + 1 + DR + 1):
            gl = it + PD
            if g_lo <= gl < g_hi and debug % 2 == 0:  # loads of group gl (debug bit 1: none, to time the rest)
                bq = bufs[gl % NR]
                for s in range(Q):
                    e = e_sb.ap(pattern=[[S, 1], [1, 1]], offset=gl * Q + s)
                    nisa.dma_copy(dst=bq[:, s, :], src=blob.select(0, e), oob_mode=nisa.oob_mode.skip)
            # the gather of block bk (xg[i, c, l] = x[tok(l), c * 128 + i]), one block ahead: the first block's in
            # the first step; block bk's vector half in the step after block bk - 1's first group, its tensor-engine
            # half DG steps later (still before block bk's first stage A)
            bk = b_lo if it == it0 else ((it - 1) // QB_ + 1 if (it - 1) % QB_ == 0 else -1)
            if b_lo <= bk < b_hi and (bk == b_lo) == (it == it0):
                # the block's lanes on the partitions: eq[l, n] = [lane(n) == bk * BL + l]
                lid = nl.ndarray((BL, 1), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=lid, data=ip[0:BL, :], op0=nl.add, operand0=1.0 * bk * BL)
                eq = nl.ndarray((BL, T, K), dtype=nl.bfloat16, buffer=nl.sbuf)  # 0 / 1 and weight x 0 / 1: exact
                nisa.tensor_scalar(dst=eq.flatten_dims(1, 2), data=lane_bc[0:BL, :], op0=nl.equal, operand0=lid)
                nisa.tensor_reduce(dst=g01s[bk % 2], op=nl.add, data=eq, axis=2)
                ew = nl.ndarray((BL, T, K), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=ew.flatten_dims(1, 2), data1=eq.flatten_dims(1, 2), data2=wb[0:BL, :],
                                   op=nl.multiply)
                nisa.tensor_reduce(dst=rs[bk % 4], op=nl.add, data=ew, axis=2)  # R: weight of lane l's pair at token t
            jp = it - 1 - DG
            bk = b_lo if it == it0 else (jp // QB_ + 1 if jp % QB_ == 0 else -1)
            if b_lo <= bk < b_hi and (bk == b_lo) == (it == it0):
                pa = bk % 2
                for tt in range(TT):  # G = R's 0/1 pattern, transposed, per token tile
                    tn = min(128, T - tt * 128)
                    pgt = nl.ndarray((128, BL), dtype=f32, buffer=nl.psum)
                    nisa.nc_matmul(dst=pgt[0:tn, :], stationary=g01s[pa][:, tt * 128:tt * 128 + tn],
                                   moving=idn[0:BL, 0:BL], accumulate=False)
                    nisa.activation(dst=gs[pa][0:tn, tt, :], op=nl.copy, data=pgt[0:tn, :])
                for c in range(C):
                    px = nl.ndarray((128, BL), dtype=nl.float32, buffer=nl.psum)
                    for tt in range(TT):  # summed over the token tiles (one 1.0 per column in all of them)
                        tn = min(128, T - tt * 128)
                        nisa.nc_matmul(dst=px, stationary=x_sb[0:tn, tt, c * 128:(c + 1) * 128], moving=gs[pa][0:tn, tt, :],
                                       accumulate=tt > 0)
                    nisa.activation(dst=xg[pa][:, c, :], op=nl.copy, data=px)
            if g_lo <= it < g_hi:  # stage A of group it
                gq = it
                pa = (gq // QB_) % 2
                l0 = (gq % QB_) * Q * L
                bq = bufs[gq % NR]
                pg = pg_r[gq % 3]
                for s in range(Q):
                    wg = bq[:, s, 0:H].view(nl.float8_e4m3)
                    for c in range(C):
                        nisa.nc_matmul(dst=pg[:, s, c, :], stationary=wg[:, c * 128:(c + 1) * 128],
                                       moving=xg[pa][:, c, l0 + s * L:l0 + (s + 1) * L], accumulate=False)
                nisa.activation(dst=sg_r[gq % 3], op=nl.copy, data=bq[:, :, o_sg:o_sd].view(sdt))
                nisa.activation(dst=sd_r[gq % 3], op=nl.copy, data=bq[:, :, o_sd:F].view(sdt))
                # g[o, s, l] = sum over tiles c of pg[o, s, c, l] * s_gu[o, c] of the slot's expert
                sg = sg_r[gq % 3].expand_dim(3).broadcast(3, L)  # [o, s, c, l]
                q = q_r[gq % 3]
                nisa.tensor_tensor(dst=q.permute((0, 1, 3, 2)), data1=pg, data2=sg, op=nl.multiply)
                g = g_r[gq % 3]
                nisa.tensor_reduce(dst=g, op=nl.add, data=q, axis=3)
                nisa.tensor_tensor(dst=mmb[gq % 3], data1=g.expand_dim(3).broadcast(3, 2),
                                   data2=mh.expand_dim(1).expand_dim(1).broadcast(1, Q).broadcast(2, L), op=nl.multiply)
            if g_lo + 1 <= it <= g_hi:  # stages B and C of group it - 1
                gq = it - 1
                bk = (gq * Q * L) // BL
                pa = bk % 2
                l0 = (gq % QB_) * Q * L
                # B: fold the up rows onto the gate rows, SiLU
                pf = pf_r[gq % 3]
                nisa.nc_matmul(dst=pf.flatten_dims(1, 3), stationary=fd, moving=mmb[gq % 3].flatten_dims(1, 3),
                               accumulate=False)
                sl = sl_r[gq % 3]
                au = au_r[gq % 3]
                if act == 0:  # silu(gate) * up
                    nisa.activation(dst=sl, op=nl.silu, data=pf[:, :, :, 0])
                    nisa.tensor_tensor(dst=au, data1=sl, data2=pf[:, :, :, 1], op=nl.multiply)
                else:  # gate clamped from above, up into [-limit, limit] (ACTS)
                    gc = nl.ndarray((128, Q, L), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=gc, data=pf[:, :, :, 0], op0=nl.minimum, operand0=limit)
                    uc = nl.ndarray((128, Q, L), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=uc, data=pf[:, :, :, 1], op0=nl.maximum, operand0=-limit, op1=nl.minimum,
                                       operand1=limit)
                    if act == 1:  # silu(gc) * uc
                        nisa.activation(dst=sl, op=nl.silu, data=gc)
                        nisa.tensor_tensor(dst=au, data1=sl, data2=uc, op=nl.multiply)
                    else:  # (uc + 1) * gc * sigmoid(alpha gc)
                        nisa.activation(dst=sl, op=nl.sigmoid, data=gc, scale=alpha)
                        nisa.tensor_tensor(dst=sl, data1=sl, data2=gc, op=nl.multiply)
                        nisa.scalar_tensor_tensor(dst=au, data=uc, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=sl)
                # a2[q, s, l, half] = a[q % 64, s, l] * [q // 64 == half]: the two H halves of
                # w_down are stacked on the partitions
                nisa.tensor_tensor(dst=a2b[gq % 3], data1=au.expand_dim(3).broadcast(3, 2),
                                   data2=mh.expand_dim(1).expand_dim(1).broadcast(1, Q).broadcast(2, L), op=nl.multiply)
                # C: down, scales, into the block's output columns
                bq = bufs[gq % NR]
                pd = pd_r[gq % 3]
                for s in range(Q):
                    wd = bq[:, s, o_dw:o_sg].view(nl.float8_e4m3)
                    for c in range(C2):
                        nisa.nc_matmul(dst=pd[:, s, c, :, :].flatten_dims(1, 2), stationary=wd[:, c * 128:(c + 1) * 128],
                                       moving=a2b[gq % 3][:, s, :, :].flatten_dims(1, 2), accumulate=False)
                # y[h], h = half * H/2 + c' * 128 + p: pd[p, s, c', l, half] * s_down[p, 2 c' + half],
                # into yb[p, half * C2 + c', lane]; one instruction per slot (each its own expert)
                for s in range(Q):
                    sd = sd_r[gq % 3][:, s, :].reshape_dim(1, (C2, 1, 2)).broadcast(2, L)
                    ybv = yb[pa][:, :, l0 + s * L:l0 + (s + 1) * L].reshape_dim(1, (2, C2)).permute((0, 2, 3, 1))
                    nisa.tensor_tensor(dst=ybv, data1=pd[:, s], data2=sd, op=nl.multiply)
                rel = bk - b_lo
                if gq % QB_ == QB_ - 1 and (rel % 2 == 1 or bk == b_hi - 1):  # this block and the one before: transposed
                    blks = []
                    if rel % 2 == 1:
                        blks.append(bk - 1)
                    blks.append(bk)
                    yts = []
                    for b in blks:
                        yt = nl.ndarray((BL, G32, 128), dtype=nl.bfloat16, buffer=nl.sbuf)  # [lane, h // 128, p]
                        for g4 in range(NH):
                            pt = nl.ndarray((BL, 4, 128), dtype=nl.float32, buffer=nl.psum)
                            for k in range(4):
                                nisa.nc_matmul(dst=pt[:, k, :], stationary=yb[b % 2][:, g4 * 4 + k, :], moving=idn,
                                               accumulate=False)
                            nisa.activation(dst=yt[:, g4 * 4:(g4 + 1) * 4, :], op=nl.copy, data=pt)
                        yts.append(yt)
                    pend = [it + DR, blks, yts, rel]
            if len(pend) > 0 and pend[0] == it:  # the pending route: its lanes' outputs into their tokens
                blks = pend[1]
                yts = pend[2]
                for hc in range(NH):
                    for tt in range(TT):  # one token tile at a time
                        tn = min(128, T - tt * 128)
                        po = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                        for j in range(len(blks)):
                            nisa.nc_matmul(dst=po[0:tn, :], stationary=rs[blks[j] % 4][:, tt * 128:tt * 128 + tn],
                                           moving=yts[j][:, hc * 4:(hc + 1) * 4, :].flatten_dims(1, 2), accumulate=j > 0)
                        av = acc[0:tn, tt, hc * 512:(hc + 1) * 512]
                        if first and pend[3] <= 1:  # this program's first route
                            nisa.activation(dst=av, op=nl.copy, data=po[0:tn, :])
                        else:
                            nisa.tensor_tensor(dst=av, data1=av, data2=po[0:tn, :], op=nl.add)
                pend = []

else:
    kiln_moe_dedupe_v8 = None
    kiln_moe_tiles_pairs_v1 = None
    kiln_moe_dedupe_v9 = None


if nki is not None:  # kiln_moe_dedupe_v10 (its own block and revision, REV10)
    @nki.jit
    def kiln_moe_dedupe_v10(x, topi, topv, blob, lanes: int, group: int, ring: int, slots: int, block: int,
                           keep: int, act: int = 0, limit: float = 0.0, alpha: float = 1.702, debug: int = 0,
                           rev: int = 0, spl: int = 1, skp: int = 0, hdf: int = 0):
        """kiln_moe_dedupe_v9 for up to 512 tokens in ONE call (T = 320 / 384 at 80 / 96 rows per DP group): each selected
        expert read once where v9's 256-token calls read every expert again per call. What would not fit SBUF at 4 token
        tiles is made smaller: x stays in HBM and each block gathers its lanes' rows by one indirect DMA (row tok(l) of x
        onto partition l, then the tensor engine's transposes into xg), the lane of every pair is kept as [pair p, tile
        j] (128 pairs per tile) instead of on every partition, and a block's route matrix R [lanes, T] and its lanes'
        tokens come from matmuls of the pairs' one-hot onto the block's lanes (OH [128 pairs, lanes]) with each pair
        tile's 16-token window of weights (pair n = 8 t + k: tile j holds tokens 16 j .. 16 j + 15) and of token indices
        (16 j + p // 8 as two bf16-exact columns). The slot table takes up to SLOTS10 slots in PSUM rows of 512. The rest
        (slots, groups, segments, routes summed in pairs of blocks, the LNC split) is v9's.
        x bf16 [T, H] (T <= 512), topi int32 [T, K] and topv bf16 [T, K] (the routing), blob
        uint8 [E, 128, F] (tile layout, bf16 or fp32 tile scales by F); `act` / `limit` / `alpha`
        the activation (ACTS); `lanes` pairs per slot, `slots`
        slots in blocks of `block` lanes (n_slots), `group` slots per vector instruction, `ring`
        groups of expert buffers (loads run ring - 2 groups ahead), padded slots below `keep` load
        expert 0 (plan()); rev: this function's source revision (REV, see _kernel_rev). Returns bf16
        [T, H]. Keep every helper inside it."""
        T, H = x.shape
        K = topi.shape[1]
        N = T * K
        TT = (T + 127) // 128  # token tiles on the partitions: token t = 128 tt + p at [p, tt]
        E = blob.shape[0]
        NE = (E + 127) // 128  # tiles of experts on the partitions (rows past E match no pair)
        BL = block
        S = slots
        L = lanes
        Q = group
        NQ = S // Q  # groups
        QB_ = BL // (L * Q)  # groups per block
        F = blob.shape[2]
        C = H // 128
        DW = H // 2
        C2 = DW // 128
        G32 = H // 128  # output column groups of 128 (h // 128)
        NH = G32 // 4  # 512-column chunks of the output
        SB = (F - H - DW) // (2 * C)  # bytes per tile scale: 2 (bf16) or 4 (fp32)
        o_dw, o_sg, o_sd = H, H + DW, H + DW + C * SB
        f32, bf16, i32 = nl.float32, nl.bfloat16, nl.int32
        sdt = nl.bfloat16 if SB == 2 else nl.float32
        LOG2L = 1 if L == 2 else 2 if L == 4 else 3 if L == 8 else 4 if L == 16 else 0
        out = nl.ndarray((T, H), dtype=bf16, buffer=nl.shared_hbm)
        # LNC (trn2 at LNC=2, grid 2): the kernel is traced once per program, the two physical cores of the
        # logical core (nki/_backends/mlir_tracer program_id: "kernel is traced LNC times with different
        # program_id_value"), so npg / pid are Python ints. With two or more blocks of lanes each program
        # runs a contiguous half of the blocks (their expert loads and matmuls), the fp32 partial sums are
        # exchanged by halves of H (nisa.sendrecv between the two cores, nki/isa/_lnc.py) and each program
        # writes its half of the output columns. Grid 1 (trn1), or one block: one program does everything.
        # spl 0 (KILN_LNC_SPLIT): both programs do all of the work, as before the split.
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if spl and nl.program_ndim() != 0 else (1, 0)
        NBK = S * L // BL  # blocks of lanes
        split = npg == 2 and NBK >= 2
        b_lo, b_hi = (0, NBK) if not split else ((0, (NBK + 1) // 2) if pid == 0 else ((NBK + 1) // 2, NBK))
        g_lo, g_hi = b_lo * QB_, b_hi * QB_  # this program's groups
        mine = split or pid == 0
        if split:  # allocated first, the same in both programs' traces: the peer's sendrecv lands here
            rcv = nl.ndarray((128, TT, H // 2), dtype=nl.float32, buffer=nl.sbuf)

        NJ = N // 128  # pair tiles (pair n = 128 j + p at [p, j]); T is a multiple of 16 (fits10)

        # --- constants (iota and compares; no inputs) ---
        ipi = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ipi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        ip = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)  # partition index
        nisa.tensor_copy(dst=ip, src=ipi)
        jfi = nl.ndarray((128, 128), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=jfi, pattern=[[1, 128]], offset=0, channel_multiplier=0)
        dd = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)  # dd[p, j] = j - p
        nisa.tensor_copy(dst=dd, src=jfi)
        nisa.tensor_scalar(dst=dd, data=dd, op0=nl.subtract, operand0=ip)
        idn = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)  # identity
        nisa.tensor_scalar(dst=idn, data=dd, op0=nl.equal, operand0=0.0)
        up = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)  # [p < j]: exclusive prefix sums
        nisa.tensor_scalar(dst=up, data=dd, op0=nl.greater, operand0=0.0)
        f1 = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=f1, data=dd, op0=nl.equal, operand0=64.0)
        f2 = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=f2, data=dd, op0=nl.equal, operand0=-64.0)
        fd = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)  # fold[o, q] = [o % 64 == q % 64]
        nisa.tensor_tensor(dst=fd, data1=idn, data2=f1, op=nl.add)
        nisa.tensor_tensor(dst=fd, data1=fd, data2=f2, op=nl.add)
        mh = nl.ndarray((128, 2), dtype=f32, buffer=nl.sbuf)  # [p < 64, p >= 64]
        nisa.tensor_scalar(dst=mh[:, 0:1], data=ip, op0=nl.less, operand0=64.0)
        nisa.tensor_scalar(dst=mh[:, 1:2], data=ip, op0=nl.greater_equal, operand0=64.0)
        e_sb = nl.ndarray((1, S), dtype=i32, buffer=nl.sbuf)
        one_r = nl.ndarray((1, 128), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=one_r, value=1.0)
        ones_f = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ones_f, value=1.0)
        ones_b = nl.ndarray((128, 128), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=ones_b, value=1.0)
        zero_n = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=zero_n, value=0.0)
        isi = nl.ndarray((128, S), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=isi, pattern=[[1, S]], offset=0, channel_multiplier=0)
        iota_s = nl.ndarray((128, S), dtype=f32, buffer=nl.sbuf)  # slot index on every partition
        nisa.tensor_copy(dst=iota_s, src=isi)

        # --- the routing plan (plan()'s arithmetic): every pair's lane, every slot's expert ---
        ti = nl.ndarray((1, N), dtype=i32, buffer=nl.sbuf)
        nisa.dma_copy(dst=ti, src=topi.reshape((1, N)))
        tb = nl.ndarray((1, N), dtype=f32, buffer=nl.sbuf)  # expert ids (fp32: bf16 rounds past 256)
        nisa.tensor_copy(dst=tb, src=ti)
        wv = nl.ndarray((1, N), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=wv, src=topv.reshape((1, N)))
        eb = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)  # e of pair n on every partition
        for n0 in range(0, N, 512):
            n1 = min(N, n0 + 512)
            pb = nl.ndarray((128, n1 - n0), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pb, stationary=one_r, moving=tb[:, n0:n1], accumulate=False)
            nisa.tensor_copy(dst=eb[:, n0:n1], src=pb)
        # pair-major: the weight of pair n = 128 j + p at [p, j] (a strided DMA of the [T K] routing), each pair tile's
        # window of its 16 tokens W[p, j, t'] = w [t' == p // 8], and the tokens' bf16-exact parts [p // 8, j]
        wJ = nl.ndarray((NJ, 128), dtype=bf16, buffer=nl.sbuf)  # [j, p] as stored, then transposed (exact)
        nisa.dma_copy(dst=wJ, src=topv.reshape((NJ, 128)))
        pwT = nl.ndarray((128, NJ), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pwT, stationary=wJ, moving=idn[0:NJ, 0:NJ], accumulate=False)
        wT = nl.ndarray((128, NJ), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=wT, src=pwT)
        p8i = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=p8i, data=ipi, op0=nl.right_shift, operand0=3)
        p8 = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)  # p // 8
        nisa.tensor_copy(dst=p8, src=p8i)
        t16i = nl.ndarray((128, 16), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=t16i, pattern=[[1, 16]], offset=0, channel_multiplier=0)
        t16 = nl.ndarray((128, 16), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=t16, src=t16i)
        m16 = nl.ndarray((128, 16), dtype=bf16, buffer=nl.sbuf)  # [t' == p // 8]
        nisa.tensor_scalar(dst=m16, data=t16, op0=nl.equal, operand0=p8)
        Wp = nl.ndarray((128, NJ, 16), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=Wp, data1=m16.expand_dim(1).broadcast(1, NJ), data2=wT.expand_dim(2).broadcast(2, 16),
                           op=nl.multiply)
        tkp = nl.ndarray((128, NJ, 2), dtype=bf16, buffer=nl.sbuf)  # [p // 8, j]: token 16 j + p // 8, exact in bf16
        jfj = nl.ndarray((128, NJ), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=jfj, pattern=[[1, NJ]], offset=0, channel_multiplier=0)
        nisa.tensor_copy(dst=tkp[:, :, 1], src=jfj)
        zj = nl.ndarray((128, NJ), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=zj, value=0.0)
        nisa.tensor_scalar(dst=tkp[:, :, 0], data=zj, op0=nl.add, operand0=p8)  # p // 8 on every column
        ilf = nl.ndarray((128, BL), dtype=f32, buffer=nl.sbuf)  # lane index within a block, on every partition
        nisa.tensor_copy(dst=ilf, src=jfi[:, 0:BL])
        # Per tile of 128 experts on the partitions (up to 4 tiles): counts first, then base, then a
        # second pass that recomputes each tile's one-hot for the ranks, so that only one [128, N]
        # one-hot is alive at a time.
        ns_t = (nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf))
        nb_t = (nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf),
                nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=bf16, buffer=nl.sbuf))
        pe_t = (nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf))
        bs_t = (nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf))
        lb_t = (nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf), nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf))
        for et in range(NE):
            nisa.tensor_scalar(dst=pe_t[et], data=ip, op0=nl.add, operand0=128.0 * et)
            cnt = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
            oh = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar_reduce(dst=oh, data=eb, op0=nl.equal, operand0=pe_t[et], reduce_op=nl.add,
                                      reduce_res=cnt)
            if L == 1:
                nisa.tensor_copy(dst=ns_t[et], src=cnt)
            else:  # ceil(cnt / L) = (cnt + L - 1) >> log2 L, in int32
                ci = nl.ndarray((128, 1), dtype=i32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ci, src=cnt)
                nisa.tensor_scalar(dst=ci, data=ci, op0=nl.add, operand0=L - 1)
                nisa.tensor_scalar(dst=ci, data=ci, op0=nl.right_shift, operand0=LOG2L)
                nisa.tensor_copy(dst=ns_t[et], src=ci)
            nisa.tensor_copy(dst=nb_t[et], src=ns_t[et])
        # base[e] = the slots of the experts before e (exclusive prefix sum over the partitions)
        for et in range(NE):
            pbs = nl.ndarray((128, 1), dtype=f32, buffer=nl.psum)
            nisa.nc_matmul(dst=pbs, stationary=up, moving=nb_t[et], accumulate=False)
            for t2 in range(et):
                nisa.nc_matmul(dst=pbs, stationary=ones_b, moving=nb_t[t2], accumulate=True)
            nisa.tensor_copy(dst=bs_t[et], src=pbs)
            nisa.tensor_scalar(dst=lb_t[et], data=bs_t[et], op0=nl.multiply, operand0=1.0 * L)
        # lane of pair n = L base[e_n] + its rank among the expert's pairs, summed over the experts (one nonzero
        # term): the experts' partitions contracted against a ones column, pair-major ([p, j]). The lanes are the
        # matmul's stationary operand here, whose values the tensor engine does not keep at fp32 for integers past
        # bf16's 8 bits (measured: v10 read x out of bounds with the lanes as fp32 stationary), so they go as two
        # bf16-exact parts, lane = 64 hi + lo.
        vacc = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)  # every expert tile's vv summed: one nonzero per column
        for et in range(NE):
            oh = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=oh, data=eb, op0=nl.equal, operand0=pe_t[et])
            cs = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(dst=cs, data0=oh, data1=zero_n, initial=0.0, op0=nl.add, op1=nl.add)
            rk = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=rk, data=cs, op0=nl.subtract, operand0=1.0, op1=nl.multiply, operand1=oh)
            vv = nl.ndarray((128, N), dtype=f32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=vv, data=oh, op0=nl.multiply, operand0=lb_t[et], op1=nl.add, operand1=rk)
            if et == 0:
                nisa.tensor_copy(dst=vacc, src=vv)
            else:
                nisa.tensor_tensor(dst=vacc, data1=vacc, data2=vv, op=nl.add)
        # the lanes as two bf16-exact parts (lane = 64 hi + lo), each pair tile's column transposed by one matmul (no PSUM
        # accumulation across the expert tiles: interleaved accumulations into one PSUM tensor came out wrong on trn1)
        vi = nl.ndarray((128, N), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=vi, src=vacc)
        hi_i = nl.ndarray((128, N), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=hi_i, data=vi, op0=nl.right_shift, operand0=6)
        vh = nl.ndarray((128, N), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=vh, src=hi_i)
        lo_i = nl.ndarray((128, N), dtype=i32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=lo_i, data=vi, op0=nl.bitwise_and, operand0=63)
        vl = nl.ndarray((128, N), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=vl, src=lo_i)
        plh = nl.ndarray((128, NJ), dtype=f32, buffer=nl.psum)
        pll = nl.ndarray((128, NJ), dtype=f32, buffer=nl.psum)
        for j in range(NJ):
            nisa.nc_matmul(dst=plh[:, j:j + 1], stationary=vh[:, j * 128:(j + 1) * 128], moving=ones_b[:, 0:1],
                           accumulate=False)
        for j in range(NJ):
            nisa.nc_matmul(dst=pll[:, j:j + 1], stationary=vl[:, j * 128:(j + 1) * 128], moving=ones_b[:, 0:1],
                           accumulate=False)
        lam = nl.ndarray((128, NJ), dtype=f32, buffer=nl.sbuf)  # lane of pair 128 j + p = 64 hi + lo
        lhs = nl.ndarray((128, NJ), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=lhs, src=plh)
        nisa.scalar_tensor_tensor(dst=lam, data=lhs, op0=nl.multiply, operand0=64.0, op1=nl.add, operand1=pll)
        # slot s holds expert e iff base[e] <= s < base[e] + nslots[e]; none: E (DMA skipped), or
        # expert 0 below `keep` (a buffer's first use must be a real load)
        SC = (S + 511) // 512  # PSUM rows of 512 slots
        pse_c = []
        psr_c = []
        for _ in range(SC):
            pse_c.append(nl.ndarray((1, 512), dtype=f32, buffer=nl.psum))
            psr_c.append(nl.ndarray((1, 512), dtype=f32, buffer=nl.psum))
        for et in range(NE):
            m1 = nl.ndarray((128, S), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=m1, data=iota_s, op0=nl.greater_equal, operand0=bs_t[et])
            bn = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=bn, data1=bs_t[et], data2=ns_t[et], op=nl.add)
            mm_ = nl.ndarray((128, S), dtype=f32, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=mm_, data=iota_s, op0=nl.less, operand0=bn, op1=nl.multiply, operand1=m1)
            for sc in range(SC):
                s0 = sc * 512
                s1 = min(S, s0 + 512)
                nisa.nc_matmul(dst=pse_c[sc][:, 0:s1 - s0], stationary=pe_t[et], moving=mm_[:, s0:s1],
                               accumulate=et > 0)  # fp32: ids past 256
                nisa.nc_matmul(dst=psr_c[sc][:, 0:s1 - s0], stationary=ones_f[:, 0:1], moving=mm_[:, s0:s1],
                               accumulate=et > 0)
        pse = nl.ndarray((1, S), dtype=f32, buffer=nl.sbuf)
        psr = nl.ndarray((1, S), dtype=f32, buffer=nl.sbuf)
        for sc in range(SC):
            s0 = sc * 512
            s1 = min(S, s0 + 512)
            nisa.tensor_copy(dst=pse[:, s0:s1], src=pse_c[sc][:, 0:s1 - s0])
            nisa.tensor_copy(dst=psr[:, s0:s1], src=psr_c[sc][:, 0:s1 - s0])
        late = nl.ndarray((1, S), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=late, data=iota_s[0:1, :], op0=nl.greater_equal, operand0=1.0 * keep)
        if split and g_lo > 0:  # this program's first `keep` slots are its buffers' first writes too
            s0 = 1.0 * g_lo * Q
            e2 = nl.ndarray((1, S), dtype=f32, buffer=nl.sbuf)  # [s0 <= s < s0 + keep]
            nisa.tensor_scalar(dst=e2, data=iota_s[0:1, :], op0=nl.greater_equal, operand0=s0)
            nisa.scalar_tensor_tensor(dst=e2, data=iota_s[0:1, :], op0=nl.less, operand0=s0 + keep,
                                      op1=nl.multiply, operand1=e2)
            nisa.tensor_scalar(dst=e2, data=e2, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
            nisa.tensor_tensor(dst=late, data1=late, data2=e2, op=nl.multiply)
        pad = nl.ndarray((1, S), dtype=f32, buffer=nl.sbuf)  # 1 - real
        nisa.tensor_scalar(dst=pad, data=psr, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
        nisa.scalar_tensor_tensor(dst=pad, data=pad, op0=nl.multiply, operand0=1.0 * E, op1=nl.multiply, operand1=late)
        sef = nl.ndarray((1, S), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sef, data1=pse, data2=pad, op=nl.add)
        nisa.tensor_copy(dst=e_sb, src=sef)

        acc = nl.ndarray((128, TT, H), dtype=nl.float32, buffer=nl.sbuf)
        # Rings, one tensor per entry (the compiler orders accesses per tensor): NR groups of expert
        # buffers, two of each per-block tile (consecutive blocks overlap by the skew), three of
        # each per-group tile.
        NR = ring
        PD = NR - 2  # a group's buffers are loaded PD steps ahead and read until the step after its own
        bufs = (nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf),
                nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf),
                nl.ndarray((128, Q, F), dtype=nl.uint8, buffer=nl.sbuf))[:NR]
        XL = nl.ndarray((BL, H), dtype=nl.bfloat16, buffer=nl.sbuf)  # a block's lanes' rows of x (one block at a time)
        tkr = (nl.ndarray((BL, 1), dtype=nl.int32, buffer=nl.sbuf), nl.ndarray((BL, 1), dtype=nl.int32, buffer=nl.sbuf))
        rs = (nl.ndarray((BL, T), dtype=nl.bfloat16, buffer=nl.sbuf), nl.ndarray((BL, T), dtype=nl.bfloat16, buffer=nl.sbuf),
              nl.ndarray((BL, T), dtype=nl.bfloat16, buffer=nl.sbuf), nl.ndarray((BL, T), dtype=nl.bfloat16, buffer=nl.sbuf))
        xg = (nl.ndarray((128, C, BL), dtype=nl.bfloat16, buffer=nl.sbuf),  # x[tok(lane), c * 128 + i]
              nl.ndarray((128, C, BL), dtype=nl.bfloat16, buffer=nl.sbuf))
        yb = (nl.ndarray((128, G32, BL), dtype=nl.bfloat16, buffer=nl.sbuf),  # [p, h // 128, lane]
              nl.ndarray((128, G32, BL), dtype=nl.bfloat16, buffer=nl.sbuf))
        mmb = (nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf),  # [o, slot, lane, half] g masked
               nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf),
               nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf))
        a2b = (nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf),  # [q, slot, lane, half] a masked
               nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf),
               nl.ndarray((128, Q, L, 2), dtype=nl.bfloat16, buffer=nl.sbuf))
        # the groups' scales, copied out of the expert buffers in stage A (by the scalar engine), so
        # that the buffers' last readers are the tensor engine's down matmuls and the next loads
        # into them wait on nothing else
        sg_r = (nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf), nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf),
                nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf))
        sd_r = (nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf), nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf),
                nl.ndarray((128, Q, C), dtype=nl.float32, buffer=nl.sbuf))
        q_r = (nl.ndarray((128, Q, L, C), dtype=nl.float32, buffer=nl.sbuf),
               nl.ndarray((128, Q, L, C), dtype=nl.float32, buffer=nl.sbuf),
               nl.ndarray((128, Q, L, C), dtype=nl.float32, buffer=nl.sbuf))
        g_r = (nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf), nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf),
               nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf))
        sl_r = (nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf), nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf),
                nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf))
        au_r = (nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf), nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf),
                nl.ndarray((128, Q, L), dtype=nl.float32, buffer=nl.sbuf))
        St = dict(T=T, TT=TT, K=K, BL=BL, L=L, Q=Q, QB_=QB_, C=C, C2=C2, G32=G32, NH=NH, H=H, F=F, S=S, NR=NR, PD=PD,
                  o_dw=o_dw, o_sg=o_sg, o_sd=o_sd, sdt=sdt, act=act, limit=limit, alpha=alpha, debug=debug, blob=blob,
                  e_sb=e_sb, x=x, lam=lam, Wp=Wp, tkp=tkp, ilf=ilf, XL=XL, tkr=tkr, NJ=NJ, ip=ip, idn=idn, fd=fd, mh=mh,
                  acc=acc, bufs=bufs, rs=rs, xg=xg, yb=yb, mmb=mmb, a2b=a2b, sg_r=sg_r, sd_r=sd_r, q_r=q_r, g_r=g_r,
                  sl_r=sl_r, au_r=au_r, b_first=b_lo)
        SPB = BL // L  # slots per block
        NHD = -(-(-(-N // L)) // SPB)  # blocks that hold one of the first ceil(N / L) slots: every routing fills them
        NHD = max(NHD, -(-(S * hdf // 100) // SPB))  # hdf: a static head of hdf% of the slots (the routing's usual fill)
        NHD = NHD + NHD % 2  # pairs of blocks share a route
        SEG = skp + skp % 2
        hd = min(b_hi, b_lo + NHD)
        if mine and SEG > 0 and not split and npg == 1 and hd < b_hi:
            KS = -(-(b_hi - hd) // SEG)
            fkf = nl.ndarray((1, KS), dtype=f32, buffer=nl.sbuf)  # [segment k's first slot is real]
            for k in range(KS):
                s0 = (hd + k * SEG) * SPB
                nisa.tensor_copy(dst=fkf[0:1, k:k + 1], src=psr[0:1, s0:s0 + 1])
            fli = nl.ndarray((1, KS), dtype=i32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=fli, src=fkf)
            _dd10_region(St, b_lo, hd)
            for k in range(KS):
                b0 = hd + k * SEG
                b1 = min(b_hi, b0 + SEG)
                rk = nisa.register_alloc()
                nisa.register_load(rk, fli.ap(pattern=[[KS, 1], [1, 1]], offset=k))

                def seg(it, b0=b0, b1=b1):
                    _dd10_region(St, b0, b1)

                nl.fori_loop(0, rk, seg)
        elif mine:
            _dd10_region(St, b_lo, b_hi)
        if split:  # the other program's partial for this program's columns, added in fp32, per token tile
            hw = H // 2
            oth = 1 - pid
            for tt in range(TT):
                tn = min(128, T - tt * 128)
                nisa.sendrecv(src=acc[0:tn, tt, oth * hw:(oth + 1) * hw], dst=rcv[0:tn, tt, :], send_to_rank=oth,
                              recv_from_rank=oth, pipe_id=0)
            for tt in range(TT):
                tn = min(128, T - tt * 128)
                mine_acc = acc[0:tn, tt, pid * hw:(pid + 1) * hw]
                nisa.tensor_tensor(dst=mine_acc, data1=mine_acc, data2=rcv[0:tn, tt, :], op=nl.add)
                o_h = nl.ndarray((128, hw), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=o_h[0:tn, :], src=mine_acc)
                nisa.dma_copy(dst=out[tt * 128:tt * 128 + tn, pid * hw:(pid + 1) * hw], src=o_h[0:tn, :])
        elif pid == 0:
            for tt in range(TT):
                tn = min(128, T - tt * 128)
                o_sb = nl.ndarray((128, H), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=o_sb[0:tn, :], src=acc[0:tn, tt, :])
                nisa.dma_copy(dst=out[tt * 128:tt * 128 + tn, :], src=o_sb[0:tn, :])
        if npg > 1:  # LNC: the whole output written (both column halves, or program 0's) before either ends
            nisa.core_barrier(data=out, cores=(0, 1))
        return out

    def _dd10_region(St, b_lo, b_hi):
        """kiln_moe_dedupe_v10's blocks b_lo .. b_hi - 1 as one software pipeline (loads, gathers, stages A / B / C,
        routes), with PSUM rings of its own: the whole call, its static head, or the body of one of its device loops
        (a PSUM tile referenced in two device-loop regions fails to compile: [NCC_IBIR092], docs/neuron-notes.md
        "Prefill MoE as a grouped GEMM"). Blocks pair up for their routes from b_lo, so a region holds its pairs."""
        T = St["T"]
        TT = St["TT"]
        K = St["K"]
        BL = St["BL"]
        L = St["L"]
        Q = St["Q"]
        QB_ = St["QB_"]
        C = St["C"]
        C2 = St["C2"]
        G32 = St["G32"]
        NH = St["NH"]
        H = St["H"]
        F = St["F"]
        S = St["S"]
        NR = St["NR"]
        PD = St["PD"]
        o_dw = St["o_dw"]
        o_sg = St["o_sg"]
        o_sd = St["o_sd"]
        sdt = St["sdt"]
        act = St["act"]
        limit = St["limit"]
        alpha = St["alpha"]
        debug = St["debug"]
        blob = St["blob"]
        e_sb = St["e_sb"]
        x = St["x"]
        lam = St["lam"]
        Wp = St["Wp"]
        tkp = St["tkp"]
        ilf = St["ilf"]
        XL = St["XL"]
        tkr = St["tkr"]
        NJ = St["NJ"]
        bf16 = nl.bfloat16
        ip = St["ip"]
        idn = St["idn"]
        fd = St["fd"]
        mh = St["mh"]
        acc = St["acc"]
        bufs = St["bufs"]
        rs = St["rs"]
        xg = St["xg"]
        yb = St["yb"]
        mmb = St["mmb"]
        a2b = St["a2b"]
        sg_r = St["sg_r"]
        sd_r = St["sd_r"]
        q_r = St["q_r"]
        g_r = St["g_r"]
        sl_r = St["sl_r"]
        au_r = St["au_r"]
        f32 = nl.float32
        g_lo = b_lo * QB_
        g_hi = b_hi * QB_
        first = b_lo == St["b_first"]  # this program's first region: its first route writes the accumulator
        pg_r = (nl.ndarray((128, Q, C, L), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, C, L), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, C, L), dtype=nl.float32, buffer=nl.psum))
        pf_r = (nl.ndarray((128, Q, L, 2), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, L, 2), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, L, 2), dtype=nl.float32, buffer=nl.psum))
        pd_r = (nl.ndarray((128, Q, C2, L, 2), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, C2, L, 2), dtype=nl.float32, buffer=nl.psum),
                nl.ndarray((128, Q, C2, L, 2), dtype=nl.float32, buffer=nl.psum))

        # Step i: the loads of group i + PD, stage A (gate_up, scales) of group i, then stage B
        # (fold, SiLU) and stage C (down, output) of group i - 1. The expert DMAs are the long
        # pole, so they run ahead; a group's buffers are free after its stage C.
        DG = QB_ // 2 if QB_ > 1 else 0  # steps from a gather's vector half to its tensor-engine half
        DR = 2  # steps from a route's transposes to its matmuls
        it0 = g_lo - PD
        pend = []  # the pending route: [step, blocks, their transposed outputs, rel]
        for it in range(it0, g_hi + 1 + DR + 1):
            gl = it + PD
            if g_lo <= gl < g_hi and debug % 2 == 0:  # loads of group gl (debug bit 1: none, to time the rest)
                bq = bufs[gl % NR]
                for s in range(Q):
                    e = e_sb.ap(pattern=[[S, 1], [1, 1]], offset=gl * Q + s)
                    nisa.dma_copy(dst=bq[:, s, :], src=blob.select(0, e), oob_mode=nisa.oob_mode.skip)
            # the gather of block bk (xg[i, c, l] = x[tok(l), c * 128 + i]), one block ahead: the first block's in
            # the first step; block bk's vector half in the step after block bk - 1's first group, its tensor-engine
            # half DG steps later (still before block bk's first stage A)
            bk = b_lo if it == it0 else ((it - 1) // QB_ + 1 if (it - 1) % QB_ == 0 else -1)
            if b_lo <= bk < b_hi and (bk == b_lo) == (it == it0):
                # the pairs' one-hot onto the block's lanes, per pair tile: OH[p, l] = [lane(128 j + p) == bk BL + l];
                # R[l, t] = sum over pairs of OH x the tile's 16-token window of weights (each window written once),
                # the lanes' tokens 16 j + p // 8 from the bf16-exact parts [p // 8, j] (a padded lane: token 0, weight 0)
                pr = nl.ndarray((128, 512), dtype=f32, buffer=nl.psum)
                ptk = nl.ndarray((128, 2), dtype=f32, buffer=nl.psum)
                lamk = nl.ndarray((128, NJ), dtype=f32, buffer=nl.sbuf)  # the pairs' lanes relative to this block
                nisa.tensor_scalar(dst=lamk, data=lam, op0=nl.subtract, operand0=1.0 * bk * BL)
                ohs = []
                for j in range(NJ):
                    oh = nl.ndarray((128, BL), dtype=bf16, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=oh, data=ilf, op0=nl.equal, operand0=lamk[:, j:j + 1])
                    nisa.nc_matmul(dst=pr[0:BL, j * 16:(j + 1) * 16], stationary=oh, moving=Wp[:, j, :], accumulate=False)
                    ohs.append(oh)
                for j in range(NJ):  # the tokens' accumulation on its own, uninterleaved
                    nisa.nc_matmul(dst=ptk[0:BL, :], stationary=ohs[j], moving=tkp[:, j, :], accumulate=j > 0)
                nisa.activation(dst=rs[bk % 4], op=nl.copy, data=pr[0:BL, 0:T])
                tk2 = nl.ndarray((128, 2), dtype=f32, buffer=nl.sbuf)  # out of PSUM first: the vector engine reads one
                nisa.tensor_copy(dst=tk2[0:BL, :], src=ptk[0:BL, :])  # PSUM operand per instruction
                tkf = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=tkf[0:BL, :], data=tk2[0:BL, 1:2], op0=nl.multiply, operand0=16.0,
                                          op1=nl.add, operand1=tk2[0:BL, 0:1])
                nisa.tensor_scalar(dst=tkf[0:BL, :], data=tkf[0:BL, :], op0=nl.maximum, operand0=0.0, op1=nl.minimum,
                                   operand1=1.0 * (T - 1))  # the gather never leaves x
                nisa.tensor_copy(dst=tkr[bk % 2], src=tkf[0:BL, :])
            jp = it - 1 - DG
            bk = b_lo if it == it0 else (jp // QB_ + 1 if jp % QB_ == 0 else -1)
            if b_lo <= bk < b_hi and (bk == b_lo) == (it == it0):
                pa = bk % 2
                # the lanes' rows of x by one indirect DMA (row tok(l) onto partition l), transposed into xg
                nisa.dma_copy(dst=XL, src=x.ap(pattern=[[H, BL], [1, H]], offset=0,
                                               vector_offset=tkr[pa].ap(pattern=[[1, BL], [1, 1]], offset=0),
                                               indirect_dim=0))
                for c in range(C):
                    px = nl.ndarray((128, BL), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(dst=px, stationary=XL[:, c * 128:(c + 1) * 128], moving=idn[0:BL, 0:BL], accumulate=False)
                    nisa.activation(dst=xg[pa][:, c, :], op=nl.copy, data=px)
            if g_lo <= it < g_hi:  # stage A of group it
                gq = it
                pa = (gq // QB_) % 2
                l0 = (gq % QB_) * Q * L
                bq = bufs[gq % NR]
                pg = pg_r[gq % 3]
                for s in range(Q):
                    wg = bq[:, s, 0:H].view(nl.float8_e4m3)
                    for c in range(C):
                        nisa.nc_matmul(dst=pg[:, s, c, :], stationary=wg[:, c * 128:(c + 1) * 128],
                                       moving=xg[pa][:, c, l0 + s * L:l0 + (s + 1) * L], accumulate=False)
                nisa.activation(dst=sg_r[gq % 3], op=nl.copy, data=bq[:, :, o_sg:o_sd].view(sdt))
                nisa.activation(dst=sd_r[gq % 3], op=nl.copy, data=bq[:, :, o_sd:F].view(sdt))
                # g[o, s, l] = sum over tiles c of pg[o, s, c, l] * s_gu[o, c] of the slot's expert
                sg = sg_r[gq % 3].expand_dim(3).broadcast(3, L)  # [o, s, c, l]
                q = q_r[gq % 3]
                nisa.tensor_tensor(dst=q.permute((0, 1, 3, 2)), data1=pg, data2=sg, op=nl.multiply)
                g = g_r[gq % 3]
                nisa.tensor_reduce(dst=g, op=nl.add, data=q, axis=3)
                nisa.tensor_tensor(dst=mmb[gq % 3], data1=g.expand_dim(3).broadcast(3, 2),
                                   data2=mh.expand_dim(1).expand_dim(1).broadcast(1, Q).broadcast(2, L), op=nl.multiply)
            if g_lo + 1 <= it <= g_hi:  # stages B and C of group it - 1
                gq = it - 1
                bk = (gq * Q * L) // BL
                pa = bk % 2
                l0 = (gq % QB_) * Q * L
                # B: fold the up rows onto the gate rows, SiLU
                pf = pf_r[gq % 3]
                nisa.nc_matmul(dst=pf.flatten_dims(1, 3), stationary=fd, moving=mmb[gq % 3].flatten_dims(1, 3),
                               accumulate=False)
                sl = sl_r[gq % 3]
                au = au_r[gq % 3]
                if act == 0:  # silu(gate) * up
                    nisa.activation(dst=sl, op=nl.silu, data=pf[:, :, :, 0])
                    nisa.tensor_tensor(dst=au, data1=sl, data2=pf[:, :, :, 1], op=nl.multiply)
                else:  # gate clamped from above, up into [-limit, limit] (ACTS)
                    gc = nl.ndarray((128, Q, L), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=gc, data=pf[:, :, :, 0], op0=nl.minimum, operand0=limit)
                    uc = nl.ndarray((128, Q, L), dtype=f32, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=uc, data=pf[:, :, :, 1], op0=nl.maximum, operand0=-limit, op1=nl.minimum,
                                       operand1=limit)
                    if act == 1:  # silu(gc) * uc
                        nisa.activation(dst=sl, op=nl.silu, data=gc)
                        nisa.tensor_tensor(dst=au, data1=sl, data2=uc, op=nl.multiply)
                    else:  # (uc + 1) * gc * sigmoid(alpha gc)
                        nisa.activation(dst=sl, op=nl.sigmoid, data=gc, scale=alpha)
                        nisa.tensor_tensor(dst=sl, data1=sl, data2=gc, op=nl.multiply)
                        nisa.scalar_tensor_tensor(dst=au, data=uc, op0=nl.add, operand0=1.0, op1=nl.multiply, operand1=sl)
                # a2[q, s, l, half] = a[q % 64, s, l] * [q // 64 == half]: the two H halves of
                # w_down are stacked on the partitions
                nisa.tensor_tensor(dst=a2b[gq % 3], data1=au.expand_dim(3).broadcast(3, 2),
                                   data2=mh.expand_dim(1).expand_dim(1).broadcast(1, Q).broadcast(2, L), op=nl.multiply)
                # C: down, scales, into the block's output columns
                bq = bufs[gq % NR]
                pd = pd_r[gq % 3]
                for s in range(Q):
                    wd = bq[:, s, o_dw:o_sg].view(nl.float8_e4m3)
                    for c in range(C2):
                        nisa.nc_matmul(dst=pd[:, s, c, :, :].flatten_dims(1, 2), stationary=wd[:, c * 128:(c + 1) * 128],
                                       moving=a2b[gq % 3][:, s, :, :].flatten_dims(1, 2), accumulate=False)
                # y[h], h = half * H/2 + c' * 128 + p: pd[p, s, c', l, half] * s_down[p, 2 c' + half],
                # into yb[p, half * C2 + c', lane]; one instruction per slot (each its own expert)
                for s in range(Q):
                    sd = sd_r[gq % 3][:, s, :].reshape_dim(1, (C2, 1, 2)).broadcast(2, L)
                    ybv = yb[pa][:, :, l0 + s * L:l0 + (s + 1) * L].reshape_dim(1, (2, C2)).permute((0, 2, 3, 1))
                    nisa.tensor_tensor(dst=ybv, data1=pd[:, s], data2=sd, op=nl.multiply)
                rel = bk - b_lo
                if gq % QB_ == QB_ - 1 and (rel % 2 == 1 or bk == b_hi - 1):  # this block and the one before: transposed
                    blks = []
                    if rel % 2 == 1:
                        blks.append(bk - 1)
                    blks.append(bk)
                    yts = []
                    for b in blks:
                        yt = nl.ndarray((BL, G32, 128), dtype=nl.bfloat16, buffer=nl.sbuf)  # [lane, h // 128, p]
                        for g4 in range(NH):
                            pt = nl.ndarray((BL, 4, 128), dtype=nl.float32, buffer=nl.psum)
                            for k in range(4):
                                nisa.nc_matmul(dst=pt[:, k, :], stationary=yb[b % 2][:, g4 * 4 + k, :], moving=idn,
                                               accumulate=False)
                            nisa.activation(dst=yt[:, g4 * 4:(g4 + 1) * 4, :], op=nl.copy, data=pt)
                        yts.append(yt)
                    pend = [it + DR, blks, yts, rel]
            if len(pend) > 0 and pend[0] == it:  # the pending route: its lanes' outputs into their tokens
                blks = pend[1]
                yts = pend[2]
                for hc in range(NH):
                    for tt in range(TT):  # one token tile at a time
                        tn = min(128, T - tt * 128)
                        po = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                        for j in range(len(blks)):
                            nisa.nc_matmul(dst=po[0:tn, :], stationary=rs[blks[j] % 4][:, tt * 128:tt * 128 + tn],
                                           moving=yts[j][:, hc * 4:(hc + 1) * 4, :].flatten_dims(1, 2), accumulate=j > 0)
                        av = acc[0:tn, tt, hc * 512:(hc + 1) * 512]
                        if first and pend[3] <= 1:  # this program's first route
                            nisa.activation(dst=av, op=nl.copy, data=po[0:tn, :])
                        else:
                            nisa.tensor_tensor(dst=av, data1=av, data2=po[0:tn, :], op=nl.add)
                pend = []
else:
    kiln_moe_dedupe_v10 = None


def _kernel_rev() -> int:
    """CRC-32 of kiln_moe_dedupe_v8's source text, passed to it as the static argument `rev`: LNL's graph
    cache key (libtorch_neuronx_lite/compile/cache.py create_cache_hash, SDK 2.32) hashes each NKI call's
    name, operands, grid, static arguments and MAC count, not the kernel source (CLAUDE.md; the same
    measurement made kernels/moe_prefill.py carry one), so without it an edit of the kernel would run the
    NEFF compiled from the old source on any host with a warm cache."""
    import zlib

    src = open(__file__).read()
    a = src.index("    @nki.jit\n    def kiln_moe_dedupe_v8")
    b = src.index("    @nki.jit\n    def kiln_moe_tiles_pairs_v1", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev()  # at import: the traced caller reads a constant


def _kernel_rev9() -> int:
    """kiln_moe_dedupe_v9's source revision (as _kernel_rev for v8)."""
    import zlib

    src = open(__file__).read()
    a = src.index("    @nki.jit\n    def kiln_moe_dedupe_v9")
    b = src.index("else:\n    kiln_moe_dedupe_v8 = None", a)
    return zlib.crc32(src[a:b].encode())


REV9 = _kernel_rev9()


def _kernel_rev10() -> int:
    """kiln_moe_dedupe_v10's source revision (its block)."""
    import zlib

    src = open(__file__).read()
    a = src.index("if nki is not None:  # kiln_moe_dedupe_v10")
    b = src.index("    kiln_moe_dedupe_v10 = None", a)
    return zlib.crc32(src[a:b].encode())


REV10 = _kernel_rev10()


def kernel():
    """The NKI kernel (raises where the NKI package is missing)."""
    if kiln_moe_dedupe_v8 is None:
        raise RuntimeError("the NKI MoE kernel needs the nki package (the Neuron venv)")
    return kiln_moe_dedupe_v8


def kernel10():
    """The 256 < T <= 512 dedupe kernel (raises where NKI is missing)."""
    if kiln_moe_dedupe_v10 is None:
        raise RuntimeError("the NKI MoE kernel needs the nki package (the Neuron venv)")
    return kiln_moe_dedupe_v10


def kernel9():
    """The 128 < T <= 256 dedupe kernel (raises where NKI is missing)."""
    if kiln_moe_dedupe_v9 is None:
        raise RuntimeError("the NKI MoE kernel needs the nki package (the Neuron venv)")
    return kiln_moe_dedupe_v9


def pairs_kernel():
    """The small-call per-pair NKI kernel on the tile layout (raises where NKI is missing)."""
    if kiln_moe_tiles_pairs_v1 is None:
        raise RuntimeError("the NKI MoE kernel needs the nki package (the Neuron venv)")
    return kiln_moe_tiles_pairs_v1


def pairs_inputs(x: torch.Tensor, topv: torch.Tensor, topi: torch.Tensor, blob: torch.Tensor, act: int = 0,
                 limit: float = 0.0, alpha: float = 1.702):
    """kiln_moe_tiles_pairs_v1's arguments for x [T, H], routing [T, K] and one layer's blob."""
    T, H = x.shape
    q = group_size(1)
    return dict(xT=x.view(T, H // P, P).permute(2, 0, 1).contiguous(), topi=topi.to(torch.int32).contiguous(),
                topv=topv.to(torch.bfloat16).contiguous(), blob=blob, group=q, ring=RING, act=act,
                limit=float(limit), alpha=float(alpha))


def from_pairs(out: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """kiln_moe_tiles_pairs_v1's output [128, T, H / 128] (column g = 2 c' + half) -> [T, H]."""
    _, T, C = out.shape
    return out.view(P, T, C // 2, 2).permute(1, 3, 2, 0).reshape(T, C * P).to(dtype)


def emulate_pairs(x, topv, topi, blob, act: int = 0, limit: float = 0.0, alpha: float = 1.702):
    """kiln_moe_tiles_pairs_v1's arithmetic: emulate() without the per-pair bf16 rounding."""
    return emulate(x, topv, topi, blob, act, limit, alpha, pair_bf16=False)


# Tokens per dedupe kernel call (KILN_MOE_DEDUPE_MAX_TOKENS): 128 is kiln_moe_dedupe_v8 alone; 256 puts a call of 129 to
# 256 tokens on kiln_moe_dedupe_v9, ONE call that reads each selected expert once (two v8 calls read nearly every expert
# twice at 192 rows, docs/neuron-notes.md "Decode at scale"); 512 puts a call of 257 to 512 tokens (a multiple of 16) on
# kiln_moe_dedupe_v10. Opt-in: it changes the graphs of every call above 128 tokens (decode buckets above 128 rows,
# dedupe-sized prefill chunks).
MAX_TOKENS = int(os.environ.get("KILN_MOE_DEDUPE_MAX_TOKENS", P))
# KILN_MOE_DEDUPE_V9=1: kiln_moe_dedupe_v9 for calls of PAIRS_BELOW to 128 tokens too (one token tile), instead of v8.
V9_ALL = os.environ.get("KILN_MOE_DEDUPE_V9", "0") == "1"


def moe_dedupe(x: torch.Tensor, topv: torch.Tensor, topi: torch.Tensor, blob: torch.Tensor,
               max_tokens: int | None = None, lanes: int | None = None, act: int = 0, limit: float = 0.0,
               alpha: float = 1.702) -> torch.Tensor:
    """sum_k topv[t, k] * expert_{topi[t, k]}(x[t]) on the device, inside the caller's graph: one
    kernel call per chunk of at most max_tokens tokens (default MAX_TOKENS; the kernel holds a chunk's
    tokens on the partitions, v8 up to 128 of them and v9 up to 256), each selected expert of a chunk
    read once; calls below PAIRS_BELOW tokens run the per-pair kernel on the same layout instead.
    act / limit / alpha: the activation (ACTS)."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    T = x.shape[0]
    if T < PAIRS_BELOW and lanes is None:
        return from_pairs(wrap_nki(pairs_kernel())[platform.nki_grid()](**pairs_inputs(x, topv, topi, blob, act, limit, alpha)),
                          x.dtype)
    mt = max_tokens or MAX_TOKENS
    if mt > 2 * P and not fits10(min(T, mt), topi.shape[1], blob.shape[0], lanes):
        mt = 2 * P
    if P < mt <= 2 * P and not fits(min(T, mt), topi.shape[1], blob.shape[0], lanes):
        mt = P
    outs = []
    for s in range(0, T, mt):
        n = min(mt, T - s)
        v10 = n > 2 * P or (V10_ALL and n > P)
        v9 = not v10 and (n > P or V9_ALL)
        # v8's calls slice x[s:s + max_tokens] as they always have: the slice's end is part of the traced graph, so
        # x[s:s + n] gave the 192-row decode graphs new keys (measured: the farm's STL graphs missed)
        e = s + n if v9 or v10 else s + mt
        call = wrap_nki(kernel10() if v10 else kernel9() if v9 else kernel())[platform.nki_grid()]
        outs.append(call(**kernel_inputs(x[s:e], topv[s:e], topi[s:e], blob, lanes, act, limit, alpha)))
    out = outs[0] if len(outs) == 1 else torch.cat(outs)
    return out.to(x.dtype)
