"""Decode step time of vllm-neuron (the managed Neuron path) at a fixed batch, for a same-instance
comparison with tools/check_device.py's "decode B=..." line.

    python bench/vllm_neuron_decode.py --model <path> --tp 32 --batch 4

B identical prompts (check_device's first prompt), greedy, ignore_eos, --tokens new tokens each;
reports the time of a second, warm generate divided by its tokens per sequence. vllm-neuron 0.24
targets Trn2 / Trn3 (DESIGN.md section 1: its utils/hardware_config.py knows trn2, trn3pd and
trn3pds); this script is how that is checked on trn1 rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=1024)
    args = ap.parse_args()
    os.environ.pop("NEURON_RT_VISIBLE_CORES", None)  # vllm-neuron 0.24 picks cores by NEURON_VISIBLE_DEVICES
    os.environ.setdefault("NEURON_VISIBLE_DEVICES", ",".join(str(i) for i in range(args.tp)))
    os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "7200")
    # The worker pins each rank to an EFA interface through a table with only trn2 / trn3pd / trn3pds
    # rows (vllm_neuron/utils/hardware_config.py _INSTANCE_CONFIGS) and an LNC of 2; its own error text
    # names this switch as the way to skip that affinity (a CPU optimisation, not correctness).
    os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
    from importlib import metadata

    from vllm import LLM, SamplingParams

    t = time.perf_counter()
    # gpt-oss as the SDK's own recipe launches it (docs/tutorials/tutorial-gpt-oss.md in the vllm-neuron
    # 0.24 DLAMI venv, "Launch the server"): the checkpoint's mxfp4 quantization_config emptied, since
    # vLLM's platform check refuses gpt_oss_mxfp4 on neuron and the backend picks BF16 or MXFP4 itself;
    # the hybrid KV cache manager on (sliding-window layers); one batch and one token bucket.
    kw = {}
    if "gpt-oss" in args.model or "gpt_oss" in args.model:
        kw = dict(hf_overrides={"quantization_config": {}}, disable_hybrid_kv_cache_manager=False)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_num_seqs=args.batch, max_model_len=args.max_model_len,
              # max_num_batched_tokens == max_model_len is single-shot prefill, and vllm-neuron 0.24 then
              # refuses prefix caching (neuron_model_runner.py: APC needs segmented prefill); this
              # benchmark reuses no prefix, so it is off.
              max_num_batched_tokens=args.max_model_len, enable_prefix_caching=False, dtype="bfloat16",
              additional_config={"neuron_config": {"num_seqs_buckets": [args.batch],
                                                   "num_batched_tokens_buckets": [args.max_model_len]}}, **kw)
    up = time.perf_counter() - t
    prompt = "The capital of France is"
    sp = SamplingParams(temperature=0.0, max_tokens=args.tokens, ignore_eos=True)
    llm.generate([prompt] * args.batch, sp)
    t = time.perf_counter()
    outs = llm.generate([prompt] * args.batch, sp)
    el = time.perf_counter() - t
    n = min(len(o.outputs[0].token_ids) for o in outs)
    print(f"vllm-neuron {metadata.version('vllm-neuron')}: engine up {up:.0f}s; B={args.batch} {n} tokens each in "
          f"{el:.2f}s = {el / n * 1e3:.1f} ms per step (prefill included)")
    print("RESULT", json.dumps({"engine": "vllm-neuron", "batch": args.batch, "ms_per_step": el / n * 1e3,
                                "tokens": n, "text": outs[0].outputs[0].text[:200]}))


if __name__ == "__main__":
    main()
