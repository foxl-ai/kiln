"""Check a real checkpoint's routed experts, as the loader stores them for one tensor-parallel rank,
against the NKI prefill MoE kernel's layout (kernels/moe_prefill.check_blob), on the CPU: no
NeuronCore and no Neuron runtime (nothing here imports libtorch_neuronx_lite).

    python tools/check_moe_prefill_layout.py --model <checkpoint dir> [--layers 3 45] [--tp 32] [--rank 0]
        [--save <experts.pt>]

Per layer: the experts go through the loader's own path (loader._load_experts: the rank's shard of
each expert, models/quant.fit_e4m3_max for trn1's e4m3 range), are packed by moe_dedupe.pack, and
the script reports
- the checkpoint's own scales: one weight_scale_inv per 128 x 128 block, so before fit_e4m3_max the
  rank's 64 gate rows share a scale per 128-column tile and so do its up rows, and the down scales
  are constant over each 128 output columns;
- after fit_e4m3_max (which halves the codes of each (row, block) whose largest value exceeds 240
  and doubles that row's scale): how many rows were doubled, and whether gate / up rows and down
  columns still share their block's scale;
- what check_blob and down_factors return (the kernel's gate_up path, its per-column down scales).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[3])
    ap.add_argument("--tp", type=int, default=32)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--save", default=None, help="torch.save the first layer's loaded experts (w_gu, w_gu_scale, "
                    "w_down, w_down_scale) here, for tools/probe_moe_prefill.py --experts-file on a device box")
    args = ap.parse_args()

    from kiln.config import ModelConfig
    from kiln.kernels import moe_dedupe as mdd
    from kiln.kernels import moe_prefill as mp
    from kiln.models import loader
    from kiln.models.quant import FP8

    path = loader.resolve_model_path(args.model)
    cfg = ModelConfig.from_pretrained(path)
    ck = loader._Checkpoint(path)
    E, I, n, r = cfg.num_experts, cfg.moe_intermediate_size, args.tp, args.rank
    Im = I // n
    sl = loader._part(I, r, n)
    print(f"{path}: {cfg.architecture}, {E} experts, moe_intermediate {I} ({Im} per rank at tp={n}, rank {r}: "
          f"columns {sl.start}..{sl.stop}), hidden {cfg.hidden_size}, expert block {cfg.quant_expert_block}", flush=True)
    for L in args.layers:
        pre = f"model.layers.{L}."
        m = pre + "mlp.experts."
        if f"{m}0.down_proj.weight" not in ck:
            print(f"layer {L}: no {m}0.down_proj.weight in the checkpoint", flush=True)
            continue
        H = ck.shape(f"{m}0.down_proj.weight")[0]
        bk = cfg.quant_expert_block or 128
        # the checkpoint as stored, one expert
        raw_s = ck.get(f"{m}0.down_proj.weight_scale_inv").float()
        raw_w = ck.get(f"{m}0.down_proj.weight")
        blk = sl.start // bk
        over = raw_w[:, sl].float().abs().amax(1) > 240
        print(f"layer {L}, expert 0 as stored: down_proj {tuple(raw_w.shape)} {raw_w.dtype}, weight_scale_inv "
              f"{tuple(raw_s.shape)}; the rank's columns lie in block {blk}; rows of that shard whose largest "
              f"|code| exceeds 240: {int(over.sum())} of {H}", flush=True)
        p = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt)  # noqa: E731
        layer = types.SimpleNamespace(
            router=p(E, cfg.hidden_size, dt=torch.bfloat16), down_t=True,
            router_bias=p(E) if cfg.router_bias else None,
            router_logit_bias=p(E, dt=torch.bfloat16) if getattr(cfg, "router_logit_bias", False) else None,
            w_gu=p(E, 2 * Im, H, dt=FP8), w_gu_scale=p(E, 2 * Im, H // bk),
            w_down=p(E, Im, H, dt=FP8), w_down_scale=p(E, 1, H))
        t0 = time.time()
        loader._load_experts(layer, ck, pre, cfg, r, n, torch.bfloat16)
        print(f"layer {L}: {E} experts of rank {r} through loader._load_experts in {time.time() - t0:.0f} s", flush=True)
        sg, sd = layer.w_gu_scale, layer.w_down_scale[:, 0, :]  # [E, 128 o, C], [E, H]
        C = H // 128
        for name, rows in (("gate", sg[:, :Im]), ("up", sg[:, Im:])):
            lo = rows.amin(1, keepdim=True)
            ratio = rows / lo
            print(f"  {name} rows: scale / the smallest of the {Im} rows in its tile: values "
                  f"{sorted(set(ratio.flatten().tolist()))[:6]}; rows sharing their tile's scale exactly in "
                  f"{float((ratio == 1).all(1).float().mean()) * 100:.1f}% of (expert, tile)", flush=True)
        ch = sd.view(E, C, 128)
        lo = ch.amin(-1, keepdim=True)
        ratio = ch / lo
        bad = ~(ratio == 1).all(-1)  # [E, C]
        print(f"  down: per-column scale / the chunk's smallest: values {sorted(set(ratio.flatten().tolist()))[:6]}; "
              f"chunks of 128 columns with one scale: {int((~bad).sum())} of {E * C}", flush=True)
        if bool(bad.any()):
            e, c = [int(v) for v in bad.nonzero()[0]]
            vals, cnt = torch.unique(ch[e, c], return_counts=True)
            print(f"  example: expert {e}, output columns {c * 128}..{c * 128 + 127}: down scales "
                  + ", ".join(f"{v:.6g} x {k}" for v, k in zip(vals.tolist(), cnt.tolist())), flush=True)
        if args.save and L == args.layers[0]:
            torch.save(dict(w_gu=layer.w_gu, w_gu_scale=layer.w_gu_scale, w_down=layer.w_down,
                            w_down_scale=layer.w_down_scale, layer=L, rank=r, tp=n, model=path), args.save)
            print(f"  saved the loaded experts to {args.save}", flush=True)
        blob = mdd.pack(layer.w_gu, layer.w_gu_scale, layer.w_down, layer.w_down_scale)
        try:
            dq = mp.check_blob(blob, H)
            down = mp.down_factors(blob, H)
            print(f"  check_blob: dequantize-first path {dq}; down_factors "
                  f"{'(chunk scales, column factors ' + str(sorted(set(down[1].float().flatten().tolist()))) + ')' if down is not None else 'None (one scale per chunk)'}",
                  flush=True)
            if down is not None:  # the factors give the loaded per-column scales back exactly
                dsc, dfr = down
                C2 = C // 2
                k = torch.arange(C)
                per_chunk = dsc[:, 2 * (k % C2) + k // C2]  # [E, C] in natural chunk order
                assert torch.equal(per_chunk.repeat_interleave(128, dim=1) * dfr.float(), sd), "down_factors inexact"
                print("  down_factors: chunk scale x column factor equals the loaded down scales exactly", flush=True)
        except Exception as ex:  # the layout the kernel did not take
            print(f"  FAILED: {type(ex).__name__}: {ex}", flush=True)


if __name__ == "__main__":
    main()
