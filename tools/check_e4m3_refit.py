"""How exactly a refit of a checkpoint's e4m3fn experts onto trn1 / trn2's e4m3 range (largest finite
value 240 instead of 448) keeps their values, on the CPU (no NeuronCore, no Neuron runtime):

    python tools/check_e4m3_refit.py --model <checkpoint dir> [--layers 3] [--tp 32] [--rank 0] [--experts 288]

For each expert of a layer, the rank's shards as the loader reads them (loader._Checkpoint.linear:
gate_proj and up_proj rows, down_proj columns, scales expanded per row) are refitted the way the
loader's _assign does (gate and up concatenated, down as read), by
- rows:   models/quant.fit_e4m3_max as on engine-v0: halve each (row, block) whose largest code
          exceeds 240, double that row's scale;
- groups: feat/trn2-bench 22dd2c2's variant: where the scales are constant over groups of
          gcd(rows, 128) rows, halve the whole (group, block) when any of its rows exceeds 240;
and dequantized. Reported per refit: how many codes changed value after dequantization, the largest
absolute error against the checkpoint's own value (code x weight_scale_inv) over all blocks and
relative to its block's range (weight_scale_inv x 448), and the effect on each expert's output for random inputs (fp32
y = down(silu(gate x) * up x) with the refitted weights against the checkpoint's).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fit_groups(w: torch.Tensor, scale: torch.Tensor, max_finite: float = 240.0):
    """feat/trn2-bench 22dd2c2's fit_e4m3_max (kiln/models/quant.py), copied as committed."""
    from kiln.models.quant import FP8

    N, K = w.shape[-2], w.shape[-1]
    nb = scale.shape[-1]
    bk = -(-K // nb)
    pad = nb * bk - K
    wf = torch.nn.functional.pad(w.float(), (0, pad)).view(*w.shape[:-1], nb, bk)
    over = wf.abs().amax(dim=-1) > max_finite  # [..., N, nb]
    if not bool(over.any()):
        return w, scale
    g = math.gcd(N, 128)
    if g > 1 and scale.shape[-2] == N:
        sg = scale.reshape(*scale.shape[:-2], N // g, g, nb)
        if bool(torch.equal(sg, sg[..., :1, :].expand_as(sg))):  # scales constant over each row group
            og = over.reshape(*over.shape[:-2], N // g, g, nb).any(dim=-2, keepdim=True)
            over = og.expand(*over.shape[:-2], N // g, g, nb).reshape(over.shape)
    wf = torch.where(over.unsqueeze(-1), wf / 2, wf)
    return wf.reshape(*w.shape[:-1], nb * bk)[..., :K].to(FP8), torch.where(over, scale * 2, scale)


def values(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """code x per-row block scale, fp32 (s [n, k / bk] or [n, 1] for a shard inside one block)."""
    bk = -(-w.shape[1] // s.shape[1])
    return w.float() * s.repeat_interleave(bk, dim=1)[:, : w.shape[1]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[3])
    ap.add_argument("--tp", type=int, default=32)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--experts", type=int, default=None, help="the first N experts (default all)")
    ap.add_argument("--other-weights", action="store_true",
                    help="also every other FP8 weight of the layer (whole tensors, as one rank at tp=1 reads them)")
    args = ap.parse_args()

    from kiln.config import ModelConfig
    from kiln.models import loader
    from kiln.models.quant import fit_e4m3_max

    path = loader.resolve_model_path(args.model)
    cfg = ModelConfig.from_pretrained(path)
    ck = loader._Checkpoint(path)
    E = args.experts or cfg.num_experts
    I, n, r = cfg.moe_intermediate_size, args.tp, args.rank
    sl = loader._part(I, r, n)
    bk = cfg.quant_expert_block or 128
    g = torch.Generator().manual_seed(0)
    x = torch.randn(16, cfg.hidden_size, generator=g)
    refits = {"rows": fit_e4m3_max, "groups": fit_groups}
    for L in args.layers:
        m = f"model.layers.{L}.mlp.experts."
        st = {k: dict(changed=0, total=0, err=0.0, rel=0.0, yrel=0.0, halved_rows=0, rows=0) for k in refits}
        for e in range(E if f"{m}0.down_proj.weight" in ck else 0):
            gw, gs = ck.linear(f"{m}{e}.gate_proj", sl, block=bk)
            uw, us = ck.linear(f"{m}{e}.up_proj", sl, block=bk)
            dw, ds = ck.linear(f"{m}{e}.down_proj", cols=sl, block=bk)
            guw, gus = torch.cat([gw, uw]), torch.cat([gs, us])
            true_gu, true_d = values(guw, gus), values(dw, ds)
            ref_y = (torch.nn.functional.silu(x @ true_gu[: len(gw)].T) * (x @ true_gu[len(gw):].T)) @ true_d.T
            for k, fit in refits.items():
                (w1, s1), (w2, s2) = fit(guw, gus, 240.0), fit(dw, ds, 240.0)
                v_gu, v_d = values(w1, s1), values(w2, s2)
                for v, t, s0, s in ((v_gu, true_gu, gus, s1), (v_d, true_d, ds, s2)):
                    d = (v - t).abs()
                    st[k]["changed"] += int((d > 0).sum())
                    st[k]["total"] += d.numel()
                    st[k]["err"] = max(st[k]["err"], float(d.max()))
                    # relative to the block's range: its weight_scale_inv x 448 (e4m3fn's largest value)
                    st[k]["rel"] = max(st[k]["rel"], float((d / (values(torch.ones_like(v), s0) * 448)).max()))
                    st[k]["halved_rows"] += int((s != s0).any(1).sum())
                    st[k]["rows"] += s.shape[0]
                y = (torch.nn.functional.silu(x @ v_gu[: len(gw)].T) * (x @ v_gu[len(gw):].T)) @ v_d.T
                st[k]["yrel"] = max(st[k]["yrel"], float((y - ref_y).abs().max() / ref_y.abs().max()))
        if args.other_weights:
            for name in sorted(k for k in ck._where if k.startswith(f"model.layers.{L}.") and k.endswith(".weight_scale_inv")
                               and ".mlp.experts." not in k):
                base = name[: -len(".weight_scale_inv")]
                w, s0 = ck.linear(base, block=bk)
                t = values(w, s0)
                out = []
                for k, fit in refits.items():
                    w1, s1 = fit(w, s0, 240.0)
                    d = (values(w1, s1) - t).abs()
                    out.append(f"{k} changed {int((d > 0).sum())}, max abs {float(d.max()):.2e}, rel "
                               f"{float((d / (values(torch.ones_like(t), s0) * 448)).max()):.2e}")
                print(f"  {base[len(f'model.layers.{L}.'):]:<40} {tuple(w.shape)}: " + "; ".join(out), flush=True)
        if f"{m}0.down_proj.weight" not in ck:
            print(f"layer {L}: no routed experts", flush=True)
            continue
        print(f"layer {L}, rank {r} of tp={n}, {E} experts (gate_up rows {2 * (sl.stop - sl.start)} x "
              f"{cfg.hidden_size}, down {cfg.hidden_size} x {sl.stop - sl.start}):", flush=True)
        for k, v in st.items():
            print(f"  {k:<7} rows with a doubled scale {v['halved_rows']} of {v['rows']}; dequantized values that "
                  f"changed {v['changed']} of {v['total']} ({100 * v['changed'] / v['total']:.4f}%); largest abs "
                  f"error {v['err']:.3e}, relative to its block's range (scale x 448) {v['rel']:.3e}; expert outputs "
                  f"(16 random inputs) max rel error {v['yrel']:.3e}", flush=True)


if __name__ == "__main__":
    main()
