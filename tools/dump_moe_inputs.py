"""The inputs of the first MoE layer's routed experts for a real prompt, from Kiln on the host CPU with the real
checkpoint truncated to its first layers: x [T, H] (the experts' input) and the routing (topv, topi), saved with
torch.save, so a kernel can be replayed on exactly that routing (tools/probe_lnc_split.py --replay).

    python tools/dump_moe_inputs.py --model zai-org/GLM-5.3-Flash --layers 4 --text-file wikitext2_test.txt \\
        --tokens 256 --out moe-inputs.pt

The routing is DecoderForCausalLM._route's on the host in bf16; on the device the same rows route the same way
except where two expert scores are nearly tied, which changes a pair, not the shape of the plan.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--text-file", default="wikitext2_test.txt")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--offset", type=int, default=0, help="first token of the slice")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams
    from kiln.models.decoder import DecoderForCausalLM
    from kiln.models.loader import resolve_model_path

    calls = []
    route = DecoderForCausalLM._route

    def recording(self, layer, x):
        topv, topi = route(self, layer, x)
        calls.append((x.detach().clone(), topv.detach().clone(), topi.detach().clone()))
        return topv, topi

    DecoderForCausalLM._route = recording
    T = args.tokens
    eng = LLMEngine(EngineConfig(model_path=resolve_model_path(args.model), device="cpu", dtype=torch.bfloat16,
                                 tp=1, num_layers=args.layers, weight_dtype="bf16", max_num_seqs=1,
                                 max_model_len=-(-(T + 64) // 256) * 256, max_prefill_tokens=T,
                                 prefill_token_buckets=(T,), page_size=32, kv_cache_gb=0.5))
    with open(args.text_file) as f:
        ids = eng.tokenizer(f.read())["input_ids"][args.offset:args.offset + T]
    eng.generate([ids], SamplingParams(max_new_tokens=1))
    eng.close()
    x, topv, topi = calls[0]
    counts = torch.bincount(topi.flatten(), minlength=int(topi.max()) + 1)
    print(f"{len(calls)} routed calls; first: x {tuple(x.shape)} {x.dtype}, topi {tuple(topi.shape)}, "
          f"max pairs on one expert {int(counts.max())}, experts used {int((counts > 0).sum())}; "
          f"x finite {bool(torch.isfinite(x).all())}, |x| max {x.float().abs().max().item():.3e}", flush=True)
    torch.save({"x": x, "topv": topv, "topi": topi, "calls": len(calls)}, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
