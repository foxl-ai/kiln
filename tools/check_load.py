"""Parse a checkpoint's config and load ONE tensor-parallel rank's shard on the host CPU.

    python tools/check_load.py --model XiaomiMiMo/MiMo-V2.6-Flash-RL --tp 32 --rank 0

Catches naming, shape and quantization mistakes in minutes, before any device compile.
"""

import argparse
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()
    from kiln.config import ModelConfig
    from kiln.models.loader import load_model, resolve_model_path
    from kiln.models.decoder import kv_bytes_per_token_rank

    path = resolve_model_path(args.model)
    cfg = ModelConfig.from_pretrained(path)
    kinds = {}
    for s in cfg.attn_layers or ():
        kinds[(s.window, s.num_kv_heads, s.head_dim, s.v_head_dim, s.rope_dim, s.rope_theta, s.sink)] = \
            kinds.get((s.window, s.num_kv_heads, s.head_dim, s.v_head_dim, s.rope_dim, s.rope_theta, s.sink), 0) + 1
    print("arch", cfg.architecture, "layers", cfg.num_layers, "moe layers", len(cfg.moe_layers),
          "experts", cfg.num_experts, "top-k", cfg.num_experts_per_tok)
    print("attention kinds (window, kv, dk, dv, rope, theta, sink): count", kinds)
    print("quant block", cfg.quant_block, "expert block", cfg.quant_expert_block, "ignored", len(cfg.quant_ignored),
          [x for x in cfg.quant_ignored if "o_proj" not in x])
    t = time.time()
    m = load_model(path, cfg, torch.bfloat16, torch.device("cpu"), args.max_model_len, args.rank, args.tp,
                   None, keep_fp8=True)
    by = {}
    for name, p in m.named_parameters():
        by[str(p.dtype)] = by.get(str(p.dtype), 0) + p.numel() * p.element_size()
    for k, v in sorted(by.items()):
        print(f"  {k:<22} {v / 2**30:7.2f} GiB")
    print(f"rank {args.rank}/{args.tp} shard: {sum(by.values()) / 2**30:.2f} GiB loaded in {time.time() - t:.0f}s; "
          f"KV {kv_bytes_per_token_rank(cfg, args.tp, torch.bfloat16) / 1024:.1f} KiB/token/rank (bf16)")
    nan = [n for n, p in m.named_parameters() if p.dtype != torch.float8_e4m3fn and not torch.isfinite(p.float()).all()]
    print("non-finite params:", nan[:5] or "none")
    l1 = m.layers[1]
    w = m._experts(l1, "w_gu", torch.tensor([0]))[0].float()  # packed MXFP4 or FP8, as the graph decodes it
    print(f"layer 1 expert 0 gate/up: std {w.std():.4f}, absmax {w.abs().max():.3f}; "
          f"qkv std {(l1.qkv.float() * l1.qkv_scale.repeat_interleave(cfg.quant_block, dim=-1)[:, :l1.qkv.shape[1]]).std():.4f}")


if __name__ == "__main__":
    main()
