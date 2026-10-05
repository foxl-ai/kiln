"""The same offline workload as bench/offline.py, served by vLLM (vllm-neuron) instead.

    python bench/baseline_vllm.py --tp 4 --max-num-seqs 12              # vllm-neuron 0.24, trn2
    python bench/baseline_vllm.py --tp 1 --dp 4 --max-num-seqs 3        # four TP=1 replicas
    python bench/baseline_vllm.py --tp 2 --max-num-seqs 6 --num-blocks 9   # vllm-neuron 0.5.3 (NxDI)

Two managed stacks, picked by the installed vllm-neuron version:

- **0.5.x** (NxD Inference, SDK 2.31.1 DLAMI `pytorch-inference-vllm-0.16`): the configuration
  Mini-SGLang-Neuron benchmarked against (benchmark/offline/bench_vllm_neuron.py in that repo):
  Qwen3-0.6B bf16, TP=2, max_num_seqs=6, max_model_len=2048, max_num_batched_tokens=8192,
  block_size=128, num_gpu_blocks_override=9, prefix caching off; cores via NEURON_RT_NUM_CORES.
- **0.24.x** (libtorch_neuronx_lite, SDK 2.32 DLAMI, trn2 / trn3 only: "Supported EC2
  instances: Trn2, Trn3"). Cores are chosen with NEURON_VISIBLE_DEVICES, never
  NEURON_RT_VISIBLE_CORES ("NEURON_RT_VISIBLE_CORES cannot be used with multi-processing
  execution on vLLM. Set NEURON_VISIBLE_DEVICES", vllm_neuron/vllm/worker/neuron_worker.py,
  _get_visible_devices); the KV cache sizes itself (VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION,
  default 0.30, docs/guides/reference-configuration.md) unless --num-blocks is given; compile
  timeouts come from VLLM_NEURON_COMPILATION_TIMEOUT (docs/getting-started/
  quickstart-offline-serving.md). Everything else is vllm-neuron's default unless a flag says
  otherwise, and the RESULT line records the arguments used.

--dp N runs N independent engines on disjoint cores, each serving every N-th request of the
workload, timed from a barrier after all are warm: the same arrangement as bench/offline.py --dp.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from offline import workload  # noqa: E402


def stack_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("vllm-neuron")
    except metadata.PackageNotFoundError:
        return "unknown"


def build_llm(args, legacy: bool):
    from vllm import LLM

    kw = dict(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp,
              max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
              max_num_batched_tokens=args.max_num_batched_tokens, enable_prefix_caching=args.prefix_caching)
    if legacy:
        blocks = 9 if args.num_blocks < 0 else args.num_blocks
        kw.update(block_size=args.block_size or 128, num_gpu_blocks_override=blocks if blocks > 0 else None)
        return LLM(**kw)
    if args.block_size:
        kw["block_size"] = args.block_size
    if args.num_blocks > 0:
        kw["num_gpu_blocks_override"] = args.num_blocks
    neuron_config = json.loads(args.neuron_config) if args.neuron_config else {}
    if args.context_buckets:
        neuron_config["decode_context_length_buckets"] = [int(x) for x in args.context_buckets.split(",")]
    if neuron_config:
        kw["additional_config"] = {"neuron_config": neuron_config}
    return LLM(**kw)


def run_replica(args, rank: int, barrier=None, results=None):
    legacy = stack_version().startswith("0.5")
    if legacy:
        os.environ["NEURON_RT_NUM_CORES"] = str(args.tp)
    else:
        os.environ.pop("NEURON_RT_VISIBLE_CORES", None)
        first = args.core_base + rank * args.tp
        os.environ["NEURON_VISIBLE_DEVICES"] = ",".join(str(first + i) for i in range(args.tp))
        # Kiln's instances carry no EFA interface; vllm_neuron/utils/hardware_config.py
        # (get_efa_interface_from_bdf): "EFA affinity is a CPU performance optimization, not a
        # correctness requirement; set NEURON_SKIP_EFA_AFFINITY=1 to skip it on instances
        # without EFA" (measured: FileNotFoundError .../infiniband on trn2.48xlarge without it).
        os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
        os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "7200")
        os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "7200")
    from vllm import SamplingParams, TokensPrompt

    t = time.perf_counter()
    llm = build_llm(args, legacy)
    print(f"replica {rank}: engine up in {time.perf_counter() - t:.1f}s", flush=True)
    prompts, max_tokens = workload(args.num_seqs, args.max_input_len, args.max_output_len)
    prompts, max_tokens = prompts[rank::args.dp], max_tokens[rank::args.dp]
    params = [SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=m) for m in max_tokens]
    llm.generate([TokensPrompt(prompt_token_ids=[1, 2, 3])], SamplingParams(temperature=0.1, max_tokens=1))
    if barrier is not None:
        barrier.wait()
    t = time.perf_counter()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params)
    elapsed = time.perf_counter() - t
    total = sum(max_tokens)
    produced = sum(len(o.outputs[0].token_ids) for o in outs)
    if results is not None:
        results[rank] = (total, elapsed, produced)
        return
    print(f"Total: {total} tok, Time: {elapsed:.2f}s, Throughput: {total / elapsed:.2f} tok/s (produced {produced})")
    print("RESULT", json.dumps({"engine": "vllm-neuron", "vllm_neuron": stack_version(),
                                "throughput_tok_s": total / elapsed, "elapsed_s": elapsed,
                                "total_tokens": total, "produced": produced, "args": vars(args)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--dp", type=int, default=1, help="independent engines on disjoint cores")
    ap.add_argument("--core-base", type=int, default=0, help="0.24: first NeuronCore of replica 0")
    ap.add_argument("--max-num-seqs", type=int, default=6, help="per engine")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--block-size", type=int, default=0, help="0 = the stack's default (128 on 0.5.x)")
    ap.add_argument("--num-blocks", type=int, default=-1,
                    help="num_gpu_blocks_override; 0 = sized by vLLM; default 9 on 0.5.x, sized by vLLM on 0.24")
    ap.add_argument("--prefix-caching", action="store_true")
    ap.add_argument("--context-buckets", default=None,
                    help="0.24: neuron_config decode_context_length_buckets, comma list (< max_model_len, /128)")
    ap.add_argument("--neuron-config", default=None, help="0.24: extra neuron_config as JSON")
    ap.add_argument("--num-seqs", type=int, default=256)
    ap.add_argument("--max-input-len", type=int, default=1024)
    ap.add_argument("--max-output-len", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()
    if args.dp == 1:
        run_replica(args, 0)
        return
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(args.dp)
    results = ctx.Manager().dict()
    procs = [ctx.Process(target=run_replica, args=(args, r, barrier, results)) for r in range(args.dp)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    if len(results) != args.dp:
        raise SystemExit(f"only {len(results)} of {args.dp} replicas reported")
    total = sum(v[0] for v in results.values())
    produced = sum(v[2] for v in results.values())
    elapsed = max(v[1] for v in results.values())
    print(f"Total: {total} tok, Time: {elapsed:.2f}s, Throughput: {total / elapsed:.2f} tok/s "
          f"(dp={args.dp}, produced {produced})")
    print("RESULT", json.dumps({"engine": "vllm-neuron", "vllm_neuron": stack_version(),
                                "throughput_tok_s": total / elapsed, "elapsed_s": elapsed, "total_tokens": total,
                                "produced": produced, "per_replica": {r: v[:2] for r, v in results.items()},
                                "args": vars(args)}))


if __name__ == "__main__":
    main()
