"""Rooflines of the MoE kernels at GLM-5.3-Flash's serving shapes on REAL routing (tools/ep_routing.py's
saved routers): per kernel and shape the work each engine and the DMA queues must do at the least, from the
kernels' own pass / tile structure, so a measured time can be read against its floors.

    python tools/moe_roofline.py --routing-file ep_routing.pt [--layers 3 20 44] [--prefill 4096]
        [--decode 16 32 64 128] [--sequences random0:0 random1:0 random0:2048 random1:2048]

Kernels (kiln/kernels/): moe_ep.kiln_moe_ep_kernel (EP prefill, tile-scale form), moe_ep.kiln_moe_ep_small (EP
decode), moe_prefill (TP prefill), moe_dedupe (TP decode). Shapes: H 4096, I 2048 (EP: 9 whole experts per rank,
288 / 32), TP: 64 intermediate rows of all 288 experts per rank. The busiest rank (contiguous placement, the
engine's) paces a layer under EP, every rank is the same under TP.

Rates (trn1, one NeuronCore, measured in docs/neuron-notes.md; each floor states which):
- PE 2.45 G moving columns/s for bf16 and fp8 alike ("Where the busiest rank's 8 ms went"); a matmul costs at
  least ~30 ns however few its columns (16-lane passes).
- DVE / ACT: fp8 -> bf16 dequantization of one [128, 128] tile by a per-partition scalar ~0.12-0.15 us on one
  engine, the two engines alternating ~0.07-0.09 us per tile ("The tile-scale form", --dq-tile).
- HBM -> SBUF 224-230 GB/s per NeuronCore in every DMA variant measured ("The decode floor is bytes"); the core's
  hbm_ddr_bandwidth 410 GB/s (neuron-explorer 2.32).
"""

from __future__ import annotations

import argparse

import torch

H, I_EP, E, K, R = 4096, 2048, 288, 8, 32
P = 128
CT, M = H // P, I_EP // P  # 32 h-tiles, 16 I-chunks
EXPERT_BYTES = 2 * I_EP * H + I_EP * H  # fp8 gate_up + down of one whole expert (25.17 MB)
PE_COLS = 2.45e9
DQ_TILE_2ENG = 0.08e-6  # s per [128, 128] tile, vector and scalar alternating
DQ_TILE_1ENG = 0.135e-6
BW = 228e9
BW_PEAK = 410e9
MM_MIN = 30e-9


def ep_passes(n: int, C: int):
    """kiln_moe_ep_kernel's passes of one local expert with n pairs at C rows: [lanes of each pass]."""
    LW = 256 if C >= 2048 else 128
    LW2 = 512
    r = max(n - LW, 0)
    q, rem = divmod(r, LW2)
    big = rem > LW
    nb = q + big
    ns = 1 if (rem > 0 and not big) else 0
    return [LW] + [LW2] * nb + [LW] * ns


def ep_small_passes(n: int, LW: int = 16):
    return [LW] * max(1, -(-n // LW))


def busiest(topi: torch.Tensor):
    owner = torch.arange(E) // (E // R)
    load = torch.bincount(owner[topi.flatten().long()], minlength=R)
    r = int(load.argmax())
    n = torch.bincount(topi.flatten().long(), minlength=E)[r * 9:(r + 1) * 9]
    return r, n.tolist(), float(load.float().mean())


def ep_prefill(n: list, C: int) -> dict:
    passes = [ep_passes(x, C) for x in n]
    lanes = sum(sum(p) for p in passes)
    npass = sum(len(p) for p in passes)
    pairs = sum(n)
    cols_per_lane = 1024 + 512 + 32  # gate_up (2 x 16 x 32 tiles, one column per lane each), down (8 x 16 x 512 / 128), x^T
    t_pe_pairs = pairs * cols_per_lane / PE_COLS
    t_pe_lanes = lanes * cols_per_lane / PE_COLS
    tiles = npass * (2 * M * CT + M * CT)  # dequantized [128, 128] tiles
    t_dq2 = tiles * DQ_TILE_2ENG
    by = npass * EXPERT_BYTES + lanes * H * 2 * 3 + C * H * 2  # weights, x gather + out RMW (read, write), out = 0
    used = sum(1 for x in n if x)
    by_min = used * EXPERT_BYTES + pairs * H * 2 * 3 + C * H * 2
    # DMAs: per pass 16 gate_up chunks + 16 scale rows + 8 down chunks + 8 scale rows (static or dynamic), NS x-row
    # gathers, NS RMW scatters, 2 NS lane-weight gathers; per descriptor 128 partition rows.
    dmas = sum(16 + 16 + 8 + 8 + 4 * (L // P) for p in passes for L in p)
    return dict(pairs=pairs, used=used, passes=npass, lanes=lanes, pe_pairs=t_pe_pairs, pe_lanes=t_pe_lanes,
                dq_tiles=tiles, dq2=t_dq2, bytes=by, bytes_min=by_min, t_bytes=by / BW, t_bytes_min=by_min / BW,
                t_bytes_peak=by_min / BW_PEAK, dmas=dmas, largest=max(n))


def ep_decode(n: list, C: int, LW: int = 16) -> dict:
    passes = [ep_small_passes(x, LW) for x in n]
    npass = sum(len(p) for p in passes)
    used = sum(1 for x in n if x)
    loads = used + sum(len(p) - 1 for p in passes)  # an unused expert's first pass skips its loads
    by = loads * EXPERT_BYTES
    by_min = used * EXPERT_BYTES
    mm = npass * (2 * M * CT + (H // 512) * 4 * M + CT + CT)  # gate_up, down, x^T and y transposes per pass
    return dict(pairs=sum(n), used=used, passes=npass, loads=loads, bytes=by, t_bytes=by / BW,
                t_bytes_min=by_min / BW, t_bytes_peak=by_min / BW_PEAK, mm=mm, t_mm=mm * MM_MIN, largest=max(n))


def tp_prefill(topi: torch.Tensor, C: int) -> dict:
    """moe_prefill on one TP rank (64 gate + 64 up + 64 down rows of every expert): lane tiles of 128 at B = 128
    (C > 2048), blocks ceil(n_e / B); per tile x rows 1 MB in, the blob 0.8 MB, Y 1 MB out, the combine's Y gather."""
    B = 64 if C <= 2048 else 128
    n = torch.bincount(topi.flatten().long(), minlength=E)
    blocks = int((-(-n // B)).sum())
    tiles = -(-blocks * B // P)
    blob = 6400 * P  # bytes per expert slice (fp8 gate_up 128 x 4096, down 64 x 4096, fp32 tile scales)
    by = tiles * (P * H * 2 + (P // B) * blob + P * H * 2) + C * K * H * 2 + C * H * 2
    cols = tiles * (CT * P + CT * P + 4 * 512 + P)  # gate_up, x^T, down (4 x 512 per tile), fold
    return dict(blocks=blocks, tiles=tiles, bytes=by, t_bytes=by / BW, t_pe=cols / PE_COLS, pairs=int(n.sum()))


def tp_decode(topi: torch.Tensor) -> dict:
    """moe_dedupe on one TP rank: each distinct expert's 64-row slice read once (0.8 MB)."""
    d = int((torch.bincount(topi.flatten().long(), minlength=E) > 0).sum())
    by = d * 6400 * P
    return dict(distinct=d, bytes=by, t_bytes=by / BW)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing-file", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 20, 44])
    ap.add_argument("--prefill", type=int, default=4096)
    ap.add_argument("--decode", type=int, nargs="+", default=[16, 32, 64, 128], help="decode rows (4 DP groups)")
    ap.add_argument("--sets", nargs="+", default=["random", "text"])
    a = ap.parse_args()
    d = torch.load(a.routing_file)
    names = d["names"]
    for L in a.layers:
        for s in a.sets:
            seqs = [x for x in names if x.startswith(s)]
            if not seqs:
                continue
            C = a.prefill
            per = C // 4
            parts = [d["topi"][L][names.index(seqs[i % len(seqs)])][(i // len(seqs)) * (4096 // 2):][:per]
                     for i in range(4)]
            topi = torch.cat(parts).long()
            r, n, mean = busiest(topi)
            ep = ep_prefill(n, C)
            tp = tp_prefill(topi, C)
            print(f"L{L} {s} prefill C={C}: busiest rank {r} {ep['pairs']} pairs (mean {mean:.0f}), largest {ep['largest']}, "
                  f"{ep['used']} used, {ep['passes']} passes, {ep['lanes']} lanes | PE floor pairs {ep['pe_pairs'] * 1e3:.2f} "
                  f"ms, lanes {ep['pe_lanes'] * 1e3:.2f} ms | dequant {ep['dq_tiles']} tiles {ep['dq2'] * 1e3:.2f} ms "
                  f"(2 engines) | HBM {ep['bytes'] / 1e6:.0f} MB {ep['t_bytes'] * 1e3:.2f} ms at 228 GB/s, min "
                  f"{ep['bytes_min'] / 1e6:.0f} MB {ep['t_bytes_min'] * 1e3:.2f} ms ({ep['t_bytes_peak'] * 1e3:.2f} at 410) | "
                  f"{ep['dmas']} DMAs | TP: {tp['tiles']} lane tiles, PE {tp['t_pe'] * 1e3:.2f} ms, HBM {tp['bytes'] / 1e9:.2f} "
                  f"GB {tp['t_bytes'] * 1e3:.2f} ms", flush=True)
            for T in a.decode:
                per = T // 4
                parts = [d["topi"][L][names.index(seqs[i % len(seqs)])][(i // len(seqs)) * 2048 + 1000:][:per]
                         for i in range(4)]
                topi = torch.cat(parts).long()
                r, n, mean = busiest(topi)
                ed = ep_decode(n, T)
                td = tp_decode(topi)
                print(f"  decode {T} rows ({T // 4}/group): busiest rank {r} {ed['pairs']} pairs (mean {mean:.1f}), largest "
                      f"{ed['largest']}, {ed['used']} used, {ed['passes']} passes, {ed['loads']} expert loads | HBM "
                      f"{ed['bytes'] / 1e6:.0f} MB {ed['t_bytes'] * 1e3:.2f} ms (once per used expert {ed['t_bytes_min'] * 1e3:.2f}, "
                      f"at 410 {ed['t_bytes_peak'] * 1e3:.2f}) | {ed['mm']} matmuls >= {ed['t_mm'] * 1e3:.2f} ms | TP: "
                      f"{td['distinct']} distinct, {td['bytes'] / 1e6:.0f} MB {td['t_bytes'] * 1e3:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
