"""How far transformers itself drifts in bf16 from its own fp32, on check_device.py's prompts:
the yardstick for a Kiln bf16 run (a divergence transformers bf16 shares is the model's, not
Kiln's).

    python tools/hf_dtype_drift.py --model tencent/Youtu-LLM-2B --tokens 32
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=32)
    args = ap.parse_args()
    from check_device import PROMPTS
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kiln.models.loader import resolve_model_path

    path = resolve_model_path(args.model)
    tok = AutoTokenizer.from_pretrained(path)
    models = {dt: AutoModelForCausalLM.from_pretrained(path, dtype=dt).eval() for dt in (torch.float32, torch.bfloat16)}
    for p in PROMPTS:
        ids = tok(p)["input_ids"]
        outs = {}
        with torch.no_grad():
            for dt, m in models.items():
                outs[dt] = m.generate(torch.tensor([ids]), max_new_tokens=args.tokens, do_sample=False,
                                      eos_token_id=None, pad_token_id=0)[0, len(ids):].tolist()
        a, b = outs[torch.float32], outs[torch.bfloat16]
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
        print(f"  bf16 matches fp32 {n:>3}/{len(a)}  {tok.decode(b)!r}"[:160], flush=True)


if __name__ == "__main__":
    main()
