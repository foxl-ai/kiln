"""How many lane blocks kernels/moe_prefill.py needs for REAL routing: GLM-5.3-Flash's first MoE layer's
router on the real hidden states of a real text, on the host CPU (no device), against the static block
count every routing fits and the routing a uniform random choice gives.

    python tools/routing_stats.py [--model zai-org/GLM-5.3-Flash] [--text-file <wikitext>] [--tokens 4096]
        [--layer 3]

Loads the checkpoint truncated to --layer + 1 layers (models/loader.py, real weights, bf16 on the host), runs
layers 0 .. --layer - 1 in the sequence form (models/hybrid.py layer, no cache) and the attention block of
--layer, then routes that layer's FFN input (DecoderForCausalLM._route) and stops before its experts (the
layers before the first MoE layer are dense, so the routing is exact). For chunks of C = 512 ... --tokens
consecutive tokens it prints the experts used, the largest load, the blocks used sum_e ceil(n_e / B) against
the static count moe_prefill.n_blocks, and the lane tiles a KILN_MOE_PREFILL_SKIP=20 kernel then runs.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zai-org/GLM-5.3-Flash")
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--skip", type=int, default=20)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from kiln.config import ModelConfig
    from kiln.kernels import moe_prefill as mp
    from kiln.models import hybrid
    from kiln.models.loader import load_model, resolve_model_path

    path = resolve_model_path(args.model)
    cfg = ModelConfig.from_pretrained(path).truncated(args.layer + 1)
    if args.layer not in cfg.moe_layers:
        raise SystemExit(f"layer {args.layer} is not a MoE layer ({cfg.moe_layers[:4]}...)")
    model = load_model(path, cfg, torch.bfloat16, torch.device("cpu"), args.tokens + 64, 0, 1, None, keep_fp8=True)
    tok = AutoTokenizer.from_pretrained(path)
    text = open(args.text_file).read() if args.text_file else " ".join(["The quick brown fox jumps over the lazy dog."] * 2000)
    ids = torch.tensor(tok(text)["input_ids"][: args.tokens])
    T = ids.shape[0]
    seen = {}

    def ffn_input(model_, layer, x):  # stand in for the last layer's MLP: keep its input, skip the experts
        seen["x"] = x
        return torch.zeros_like(x)

    with torch.no_grad():
        positions = torch.arange(T)
        h = hybrid.hidden_in(model, ids, None)
        seq: dict = {}
        real_mlp = hybrid._mlp
        for i, l in enumerate(model.layers):
            if i == args.layer:
                hybrid._mlp = ffn_input
            try:
                h = hybrid.layer(model, l, h, positions, None, None, None, None, seq)
            finally:
                hybrid._mlp = real_mlp
            print(f"layer {i} done", flush=True)
        _, topi = model._route(model.layers[args.layer], seen["x"])
    E, K = cfg.num_experts, cfg.num_experts_per_tok
    g = torch.Generator().manual_seed(0)
    for C in (512, 1024, 2048, 4096):
        if C > T:
            break
        for name, ti in (("real", topi[:C]), ("uniform", torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(C)]))):
            B = mp.block_size(C)
            NB = mp.n_blocks(C * K, E, B)
            n = torch.bincount(ti.flatten(), minlength=E)
            used = int((-(-n // B)).sum())
            G = 128 // B
            NTL = NB // G
            utiles = -(-used // G)
            skp = args.skip
            T1 = NTL - skp * ((NTL - (NTL + 1) // 2) // skp)
            runs = max(T1, T1 + -(-(utiles - T1) // skp) * skp) if utiles > T1 else T1
            print(f"C={C:5d} {name:>7}: experts used {int((n > 0).sum())}, max load {int(n.max())}, blocks used "
                  f"{used} of {NB} static (B={B}), lane tiles used {utiles} of {NTL}, run with skp={skp}: "
                  f"{min(runs, NTL)}", flush=True)


if __name__ == "__main__":
    main()
