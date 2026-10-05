"""transformers reference outputs (fp32, CPU) for tools/check_device.py and tools/check_ppl.py,
written to JSON so the comparison can run where the serving venv's transformers is too old for
the model (the SDK 2.32 vLLM venv predates qwen3_5).

    python tools/hf_reference.py --model Qwen/Qwen3.5-0.8B --tokens 32 --out ref.json
    python tools/check_device.py --model Qwen/Qwen3.5-0.8B --reference-json ref.json

For every check_device prompt: its token ids, the reference's greedy continuation, and the
top-2 logit margin at each generated position (the margin check_device reports at a
divergence). For every check_ppl sentence: the reference's prompt logprobs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.check_device import PROMPTS  # noqa: E402
from tools.check_ppl import TEXTS  # noqa: E402


def load_reference(path: str):
    """transformers' fp32 model for a checkpoint: its causal LM, or for a composite vision-language
    checkpoint that has none (GLM-5.3-Flash, Glm5NextForConditionalGeneration) its image-text-to-text
    class, which generates from text ids alike."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    except ValueError:
        return AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float32).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    from kiln.models.loader import resolve_model_path

    path = resolve_model_path(args.model)
    tok = AutoTokenizer.from_pretrained(path)
    model = load_reference(path)
    out = {"model": args.model, "tokens": args.tokens, "prompts": [], "ppl": []}
    for text in PROMPTS:
        ids = tok(text)["input_ids"]
        x = torch.tensor([ids])
        with torch.no_grad():
            g = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=args.tokens, do_sample=False,
                               eos_token_id=None, pad_token_id=0, output_logits=True, return_dict_in_generate=True)
        margins = [float(t.topk(2).values[0, 0] - t.topk(2).values[0, 1]) for t in g.logits]
        out["prompts"].append({"text": text, "ids": ids, "greedy": g.sequences[0, len(ids):].tolist(),
                               "margins": margins})
    for text in TEXTS:
        ids = tok(text)["input_ids"]
        with torch.no_grad():
            logp = torch.log_softmax(model(torch.tensor([ids])).logits[0].float(), -1)
        lps = [float(logp[i - 1, ids[i]]) for i in range(1, len(ids))]
        out["ppl"].append({"text": text, "mean_logprob": sum(lps) / len(lps), "tokens": len(lps)})
        print(f"  {sum(lps) / len(lps):7.3f}  {text[:50]!r}")
    n = sum(p["tokens"] for p in out["ppl"])
    out["mean_prompt_logprob"] = sum(p["mean_logprob"] * p["tokens"] for p in out["ppl"]) / n
    print(f"RESULT reference mean_prompt_logprob={out['mean_prompt_logprob']:.3f} tokens={n}")
    with open(args.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
