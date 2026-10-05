"""A random-weight MiMo-V2 checkpoint for tools/check_device.py: tests/test_mimo_v2.py's geometry
(4 layers: full attention 4 query / 2 KV heads, sliding-window attention 4 / 4 heads with window 8
and per-head attention sinks, Dk 24 != Dv 16, value scale, partial RoPE; sigmoid MoE with a
correction bias), a real tokenizer and its vocabulary, and the reference greedy continuations of
check_device's prompts in tools/hf_reference.py's format, computed by Xiaomi's own modeling code
(tests/reference/mimo_v2), which AutoModelForCausalLM cannot load (it is the model repository's
remote code).

    python tools/build_random_mimo.py /opt/kiln/work/rand-mimo --tokens 32
    python tools/check_device.py --model /opt/kiln/work/rand-mimo --dtype fp32 \\
        --reference-json /opt/kiln/work/rand-mimo/reference.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from tests.reference.mimo_v2.configuration_mimo_v2 import MiMoV2Config
    from tests.reference.mimo_v2.modeling_mimo_v2 import MiMoV2ForCausalLM
    from tools.check_device import PROMPTS

    tok_dir = snapshot_download(args.tokenizer, allow_patterns=["tokenizer*", "vocab.json", "merges.txt", "config.json"])
    with open(os.path.join(tok_dir, "config.json")) as f:
        vocab = json.load(f)["vocab_size"]
    cfg = MiMoV2Config(  # tests/test_mimo_v2.py build_reference, at the tokenizer's vocabulary
        vocab_size=vocab, hidden_size=args.hidden, intermediate_size=2 * args.hidden, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048, layernorm_epsilon=1e-6,
        rope_theta=1.0e7, attention_value_scale=0.707, head_dim=24, v_head_dim=16,
        swa_num_attention_heads=4, swa_num_key_value_heads=4, swa_head_dim=24, swa_v_head_dim=16,
        swa_rope_theta=1.0e4, sliding_window=8, sliding_window_size=8, add_swa_attention_sink_bias=True,
        add_full_attention_sink_bias=False, hybrid_layer_pattern=[0, 1, 1, 0], partial_rotary_factor=0.334,
        n_routed_experts=8, moe_intermediate_size=args.hidden // 2, num_experts_per_tok=2, scoring_func="sigmoid",
        topk_method="noaux_tc", n_group=1, topk_group=1, norm_topk_prob=True, moe_layer_freq=[0, 1, 1, 1],
        attention_projection_layout="split", tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"
    model = MiMoV2ForCausalLM(cfg)
    g = torch.Generator().manual_seed(args.seed)
    with torch.no_grad():  # unit-variance matmul outputs, so the logits spread and greedy has margins
        for name, p in model.named_parameters():
            r = torch.randn(p.shape, generator=g)
            if "sink" in name or "correction" in name:
                p.copy_(0.5 * r)
            elif "norm" in name:
                p.copy_(1 + 0.1 * r)
            elif "embed" in name:
                p.copy_(r)
            else:
                p.copy_(r * p.shape[-1] ** -0.5)
    model.eval()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    for f in os.listdir(tok_dir):  # the snapshot directory also holds whatever else the cache has (weights)
        if f.startswith("tokenizer") or f in ("vocab.json", "merges.txt"):
            shutil.copy(os.path.join(tok_dir, f), args.out)
    tok = AutoTokenizer.from_pretrained(args.out)
    out = {"model": args.out, "tokens": args.tokens, "prompts": []}
    for text in PROMPTS:  # as tools/hf_reference.py
        ids = tok(text)["input_ids"]
        x = torch.tensor([ids])
        with torch.no_grad():
            g = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=args.tokens, do_sample=False,
                               eos_token_id=None, pad_token_id=0, output_logits=True, return_dict_in_generate=True)
        margins = [float(t.topk(2).values[0, 0] - t.topk(2).values[0, 1]) for t in g.logits]
        out["prompts"].append({"text": text, "ids": ids, "greedy": g.sequences[0, len(ids):].tolist(),
                               "margins": margins})
    with open(os.path.join(args.out, "reference.json"), "w") as f:
        json.dump(out, f)
    m = min(min(p["margins"]) for p in out["prompts"])
    print(f"built MiMo-V2 random ({vocab} vocab, hidden {args.hidden}) -> {args.out}; smallest top-2 margin {m:.2e}")


if __name__ == "__main__":
    main()
