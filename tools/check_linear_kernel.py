"""A linear-attention model served on the device against transformers on the host, with a prompt long
enough to cross several prefill chunks and several of the delta-rule kernel's 128-token chunks:
teacher-forced prompt logprobs at every position and the greedy continuation.

    KILN_LINEAR_ATTN_KERNEL=nki KILN_CC_ARGS=--auto-cast=none \
        python tools/check_linear_kernel.py --model /opt/kiln/work/rand-glm5n --len 700 --prefill-tokens 256

The checkpoint is a random-weight one from tools/build_random_hybrid.py (GLM-5.3-Flash: KDA with the
-5 lower bound, head dim 128; Qwen3.8-Flash-Next: GDN) or any linear-attention model transformers
loads (transformers >= 5.18 for glm5_next / qwen4_exp, put first on PYTHONPATH; imported after the
engine is up, as tools/check_device.py does). The prompt is random token ids (seeded). Prints the max
|dlogprob| over the prompt, the greedy tokens matched, and the reference's top-2 margin where they
first differ.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--len", type=int, default=700, help="prompt tokens")
    ap.add_argument("--tokens", type=int, default=16, help="greedy tokens generated")
    ap.add_argument("--prefill-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--piecewise", action="store_true")
    args = ap.parse_args()

    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams
    from kiln.models import linear_attn as la

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    ps = 32
    max_len = -(-(args.len + args.tokens + 1) // ps) * ps
    cfg = EngineConfig(model_path=args.model, device=args.device, dtype=dtype, page_size=ps, max_num_seqs=1,
                       max_model_len=max_len, max_prefill_tokens=args.prefill_tokens, kv_cache_gb=1.0,
                       piecewise=args.piecewise, decode_batch_buckets=(1,),
                       prefill_token_buckets=(args.prefill_tokens,), page_buckets=(max_len // ps,))
    t0 = time.perf_counter()
    eng = LLMEngine(cfg)
    V = eng.model.cfg.vocab_size
    g = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(100, min(V, 30000), (args.len,), generator=g).tolist()
    print(f"engine up in {time.perf_counter() - t0:.1f}s; KILN_LINEAR_ATTN_KERNEL={la.LINEAR_ATTN_KERNEL}, "
          f"KILN_CC_ARGS={os.environ.get('KILN_CC_ARGS', '')!r}, {args.dtype}, prompt {args.len} tokens, prefill "
          f"chunks of {args.prefill_tokens}", flush=True)
    t0 = time.perf_counter()
    (r,) = eng.generate([ids], SamplingParams(max_new_tokens=args.tokens, ignore_eos=True, prompt_logprobs=0))
    print(f"generate (includes compiles) {time.perf_counter() - t0:.1f}s", flush=True)

    from tools.hf_reference import load_reference  # after the engine (LLMEngine.__init__)

    ref = load_reference(args.model)
    with torch.no_grad():
        logp = torch.log_softmax(ref(torch.tensor([ids])).logits[0].double(), -1)
        want = ref.generate(torch.tensor([ids]), max_new_tokens=args.tokens, do_sample=False, eos_token_id=None,
                            pad_token_id=0)[0, len(ids):].tolist()
    dlp = [abs(r.prompt_logprobs[q][0] - logp[q - 1, ids[q]].item()) for q in range(1, len(ids))]
    worst = max(range(len(dlp)), key=lambda i: dlp[i])
    n = next((i for i, (a, b) in enumerate(zip(r.output_ids, want)) if a != b), len(want))
    margin = None
    if n < len(want):
        with torch.no_grad():
            lg = ref(torch.tensor([ids + want[:n]])).logits[0, -1]
        top = lg.topk(2).values
        margin = float(top[0] - top[1])
    print(f"prompt logprobs vs transformers: max |dlogprob| {max(dlp):.3e} (position {worst + 1}), mean "
          f"{sum(dlp) / len(dlp):.3e}; greedy {n}/{len(want)} tokens match, margin at divergence {margin}", flush=True)


if __name__ == "__main__":
    main()
