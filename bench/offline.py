"""Offline throughput, reproducing the workload Mini-SGLang-Neuron and nano-vllm publish.

    python bench/offline.py --model Qwen/Qwen3-0.6B --device neuron --max-num-seqs 6

The prompt set is generated exactly like
mini-sglang-neuron/benchmark/offline/bench_vllm_neuron.py (random.seed(0); 256 prompts of
randint(100, 1024) tokens drawn from randint(0, 10000); max_tokens randint(100, 1024);
temperature 0.6; ignore_eos), so the numbers are comparable with the ones that repo
reports: throughput = sum(max_tokens) / wall time, after a one-token warmup.
"""

from __future__ import annotations

import argparse
import json
import time
from random import randint, seed

import torch


def workload(num_seqs: int, max_input_len: int, max_output_len: int):
    seed(0)
    prompts = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    max_tokens = [randint(100, max_output_len) for _ in range(num_seqs)]
    return prompts, max_tokens


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--num-seqs", type=int, default=256)
    ap.add_argument("--max-input-len", type=int, default=1024)
    ap.add_argument("--max-output-len", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=6)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--prefill-tokens", type=int, default=512)
    ap.add_argument("--kv-cache-gb", type=float, default=6.0)
    ap.add_argument("--decode-buckets", default=None, help="comma list, default power-of-two ladder")
    ap.add_argument("--page-buckets", default=None, help="comma list of pages per sequence")
    ap.add_argument("--prefill-buckets", default=None, help="comma list of chunk sizes")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--warmup", action="store_true", help="compile every bucket before timing")
    ap.add_argument("--dp", type=int, default=1, help="engine replicas, one per NeuronCore")
    ap.add_argument("--core-base", type=int, default=0, help="first NeuronCore of replica 0")
    ap.add_argument("--overlap", action="store_true", help="overlap scheduling")
    ap.add_argument("--piecewise", action="store_true", help="one compiled graph per layer kind")
    ap.add_argument("--piecewise-group", type=int, default=None, help="layers per piecewise graph")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel ranks per replica")
    ap.add_argument("--compile-cache", default=None, help="s3:// prefix for shared compiled graphs")
    ap.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8"])
    ap.add_argument("--weight-dtype", default="auto", choices=["auto", "bf16", "fp8", "fp8-experts"])
    return ap


def run_replica(args, rank: int, barrier=None, results=None):
    """One engine on one NeuronCore, serving every dp-th request of the workload."""
    import os

    if args.device == "neuron" and (args.dp > 1 or args.core_base) and args.tp == 1:
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(args.core_base + rank)
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    def ladder(s):
        return tuple(int(x) for x in s.split(",")) if s else None

    cfg = EngineConfig(
        model_path=args.model, device=args.device, dtype=torch.bfloat16, page_size=args.page_size,
        max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
        max_prefill_tokens=args.prefill_tokens, kv_cache_gb=args.kv_cache_gb,
        decode_batch_buckets=ladder(args.decode_buckets), page_buckets=ladder(args.page_buckets),
        prefill_token_buckets=ladder(args.prefill_buckets), overlap=args.overlap, piecewise=args.piecewise, piecewise_group=args.piecewise_group,
        tp=args.tp, tp_core_base=args.core_base + rank * args.tp, compile_cache_uri=args.compile_cache,
        kv_cache_dtype=args.kv_cache_dtype, weight_dtype=args.weight_dtype,
    )
    eng = LLMEngine(cfg)
    if args.device == "neuron" and rank == 0:
        from kiln import platform

        print("PLATFORM", json.dumps(platform.describe(platform.runtime_target())), flush=True)
    prompts, max_tokens = workload(args.num_seqs, args.max_input_len, args.max_output_len)
    prompts, max_tokens = prompts[rank::args.dp], max_tokens[rank::args.dp]
    params = [SamplingParams(max_new_tokens=m, temperature=args.temperature, ignore_eos=True) for m in max_tokens]

    t = time.perf_counter()
    eng.generate([[1, 2, 3]], SamplingParams(max_new_tokens=1, temperature=0.1))
    print(f"warmup generate {time.perf_counter() - t:.1f}s")
    if args.warmup:
        w = eng.warmup()
        print(f"bucket warmup {w['seconds']:.1f}s over {w['graphs']} graphs "
              f"(compile cache: {w.get('cache_pulled', 0)} pulled, {w.get('cache_pushed', 0)} pushed)")

    compiled_before = set(eng.runner.compile_seconds)
    if barrier is not None:
        barrier.wait()  # every replica warm before the clock starts
    t = time.perf_counter()
    reqs = eng.generate(prompts, params)
    elapsed = time.perf_counter() - t
    total = sum(max_tokens)
    if results is not None:
        results[rank] = (total, elapsed, {str(k): round(v, 1) for k, v in eng.runner.compile_seconds.items()
                                          if k not in compiled_before})
        return
    assert sum(len(r.output_ids) for r in reqs) == total
    new_compiles = {str(k): round(v, 1) for k, v in eng.runner.compile_seconds.items() if k not in compiled_before}
    ttft = sorted(r.first_token_time - r.arrival_time for r in reqs)
    print(f"Total: {total} tok, Time: {elapsed:.2f}s, Throughput: {total / elapsed:.2f} tok/s")
    print(f"TTFT p50 {ttft[len(ttft) // 2]:.2f}s; preemptions {eng.scheduler.num_preemptions}; "
          f"compiles during the timed run: {new_compiles or 'none'}")
    print("RESULT", json.dumps({
        "throughput_tok_s": total / elapsed, "elapsed_s": elapsed, "total_tokens": total,
        "max_num_seqs": args.max_num_seqs, "device": args.device, "model": args.model,
        "compiles_in_timed_run": new_compiles,
        "graph_calls": {str(k): v for k, v in eng.runner.calls.items()},
    }))


def main() -> None:
    args = build_parser().parse_args()
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
    elapsed = max(v[1] for v in results.values())
    print(f"Total: {total} tok, Time: {elapsed:.2f}s, Throughput: {total / elapsed:.2f} tok/s (dp={args.dp})")
    print("RESULT", json.dumps({"throughput_tok_s": total / elapsed, "elapsed_s": elapsed, "total_tokens": total,
                                "dp": args.dp, "max_num_seqs_per_replica": args.max_num_seqs, "overlap": args.overlap,
                                "kv_cache_dtype": args.kv_cache_dtype,
                                "per_replica": {r: v[:2] for r, v in results.items()},
                                "compiles_in_timed_run": {r: v[2] for r, v in results.items()}}))


if __name__ == "__main__":
    main()
