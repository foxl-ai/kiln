"""Lone-request time to first token, Kiln against vllm-neuron, on the same prompts and cores.

    python bench/ttft_compare.py --engine vllm --model Qwen/Qwen3-8B --tp 4 --text-file lc-long.txt \
        --lengths 2048 8192 16384 32768 --max-model-len 32768
    python bench/ttft_compare.py --engine kiln --model Qwen/Qwen3-8B --tp 4 --text-file lc-long.txt \
        --lengths 2048 8192 16384 32768 --max-model-len 32768 -- --piecewise --prefill-tokens 4096 ...

Prompts: the text file tokenized once with the model's tokenizer, cut into disjoint windows in a fixed order, so
the n-th request of a length reads the same tokens on both engines and no two requests share a prefix. Per length,
in order: one warm request (not reported), then --runs lone requests with max_tokens 1, greedy: TTFT is the wall
time of that generate() call (prefill plus the first sampled token, the same measurement on both engines). Kiln
also reports its own request clock (first_token_time - arrival_time). Then one request with max_tokens
--output-len on the next window: ITL = (its wall - the length's mean TTFT) / (output - 1).

vllm-neuron 0.24 chunks a prefill only at batch size 1 ("Currently Neuron only supports chunking prefills with batch
size of 1. Mixing prefill and decode in the same batch is not supported", vllm_neuron/vllm/platform.py); these are
lone requests, so --vllm-batched-tokens N < max_model_len runs a long prompt in chunks of N (enable_chunked_prefill),
and N = max_model_len (the default) runs it as one prefill. Its token buckets stay at the stack's default (powers of
two from 128 up to N, utils/bucket_utils.py) unless --vllm-token-buckets is given. Cores: NEURON_VISIBLE_DEVICES
for vllm-neuron (bench/baseline_vllm.py), tp_core_base for Kiln.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def windows(tok_ids: list[int], lengths: list[int], runs: int):
    """[(length, kind, ids)] in the order both engines run them: per length warm, runs x ttft, one itl."""
    out, a = [], 0
    for n in lengths:
        for kind in ["warm"] + ["ttft"] * runs + ["itl"]:
            if a + n > len(tok_ids):
                raise SystemExit(f"text file has {len(tok_ids)} tokens, {a + n} needed")
            out.append((n, kind, list(tok_ids[a:a + n])))
            a += n
    return out


def vllm_engine(args):
    os.environ.pop("NEURON_RT_VISIBLE_CORES", None)
    os.environ["NEURON_VISIBLE_DEVICES"] = ",".join(str(args.core_base + i) for i in range(args.tp))
    os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
    os.environ.setdefault("VLLM_NEURON_COMPILATION_TIMEOUT", "7200")
    os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "7200")
    from vllm import LLM, SamplingParams, TokensPrompt

    kw = dict(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp, max_num_seqs=1,
              max_model_len=args.max_model_len, max_num_batched_tokens=args.vllm_batched_tokens or args.max_model_len,
              enable_prefix_caching=False)
    if kw["max_num_batched_tokens"] < args.max_model_len:
        kw["enable_chunked_prefill"] = True
    nc = json.loads(args.vllm_neuron_config) if args.vllm_neuron_config else {}
    if args.vllm_token_buckets:
        nc["num_batched_tokens_buckets"] = [int(x) for x in args.vllm_token_buckets.split(",")]
    if nc:
        kw["additional_config"] = {"neuron_config": nc}
    t = time.perf_counter()
    llm = LLM(**kw)
    up = time.perf_counter() - t

    def gen(ids, max_tokens):
        t = time.perf_counter()
        out = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                           SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True), use_tqdm=False)
        wall = time.perf_counter() - t
        assert len(out[0].outputs[0].token_ids) == max_tokens
        return wall, None

    from importlib import metadata

    return gen, up, {"vllm": metadata.version("vllm"), "vllm_neuron": metadata.version("vllm-neuron"),
                     "engine_kwargs": {k: v for k, v in kw.items() if k != "model"}}


def kiln_engine(args, rest):
    import torch
    import offline

    ka = offline.build_parser().parse_args(rest + ["--model", args.model, "--tp", str(args.tp),
                                                   "--core-base", str(args.core_base),
                                                   "--max-model-len", str(args.max_model_len)])
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    def ladder(s):
        return tuple(int(x) for x in s.split(",")) if s else None

    cfg = EngineConfig(
        model_path=ka.model, device="neuron", dtype=torch.bfloat16, page_size=ka.page_size, max_num_seqs=ka.max_num_seqs,
        max_model_len=ka.max_model_len, max_prefill_tokens=ka.prefill_tokens, kv_cache_gb=ka.kv_cache_gb,
        decode_batch_buckets=ladder(ka.decode_buckets), page_buckets=ladder(ka.page_buckets),
        prefill_token_buckets=ladder(ka.prefill_buckets), overlap=ka.overlap, piecewise=ka.piecewise,
        piecewise_group=ka.piecewise_group, tp=ka.tp, tp_core_base=ka.core_base, compile_cache_uri=ka.compile_cache,
        kv_cache_dtype=ka.kv_cache_dtype, weight_dtype=ka.weight_dtype,
        # EAGLE-3 (kiln/models/eagle3.py): --kiln-spec-draft <checkpoint> drafts --kiln-spec-k tokens per step
        **({"spec_method": "mtp", "spec_draft_model": args.kiln_spec_draft, "spec_k": args.kiln_spec_k}
           if args.kiln_spec_draft else {}),
    )
    t = time.perf_counter()
    eng = LLMEngine(cfg)
    t_init = time.perf_counter() - t
    eng.generate([[1, 2, 3]], SamplingParams(max_new_tokens=1))
    w = eng.warmup() if ka.warmup else {}
    up = time.perf_counter() - t
    if w:
        print(f"bucket warmup {w['seconds']:.1f}s over {w['graphs']} graphs", flush=True)

    def gen(ids, max_tokens):
        t = time.perf_counter()
        reqs = eng.generate([ids], SamplingParams(max_new_tokens=max_tokens, temperature=0.0, ignore_eos=True))
        wall = time.perf_counter() - t
        assert len(reqs[0].output_ids) == max_tokens
        last_ids[:] = list(reqs[0].output_ids)  # the ITL request's greedy tokens, to compare runs (speculation)
        return wall, reqs[0].first_token_time - reqs[0].arrival_time

    last_ids: list[int] = []

    # Where the start-up went: the weight load (build_shard), the parallel precompile (KILN_PRECOMPILE_WORKERS, None
    # when off) and the warmup.
    start = {"engine_init_s": round(t_init, 1), "load_s": round(eng.load_seconds, 1), "precompile": eng.precompile,
             "warmup_s": round(w["seconds"], 1) if w else None}
    def spec():
        return {"proposed": eng.spec_proposed, "accepted": eng.spec_accepted}

    return gen, up, {"kiln_args": vars(ka), "compile_seconds": eng.runner.compile_seconds, "start": start,
                     "spec": spec, "last_ids": last_ids}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["kiln", "vllm"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--core-base", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 16384, 32768])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--vllm-batched-tokens", type=int, default=0, help="vllm: max_num_batched_tokens (chunk size)")
    ap.add_argument("--vllm-token-buckets", default=None, help="vllm: neuron_config num_batched_tokens_buckets")
    ap.add_argument("--vllm-neuron-config", default=None, help="vllm: extra neuron_config JSON")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--kiln-spec-draft", default=None, help="kiln: an EAGLE-3 draft checkpoint (speculative decoding)")
    ap.add_argument("--kiln-spec-k", type=int, default=3, help="kiln: draft tokens per step with --kiln-spec-draft")
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    # Kiln's engine BEFORE transformers is imported here: a process that imported transformers and then spawned the
    # tensor-parallel ranks traces torch.topk differently on rank 0 (kiln/engine/engine.py, tools/
    # probe_fx_normalisation.py). Measured 2026-10-07 on trn1.2xlarge, Qwen3-1.7B TP=2: with the tokenizer first,
    # rank 0 compiled its own copies of the 3 post graphs (18 cache entries against the 15 every other process
    # computes), and kiln/precompile.py's capture could not see them.
    if args.engine == "kiln":
        gen, up, info = kiln_engine(args, rest)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    with open(args.text_file) as f:
        ids = tok(f.read())["input_ids"]
    plan = windows(ids, args.lengths, args.runs)

    if args.engine == "vllm":
        gen, up, info = vllm_engine(args)
    print(f"{args.engine}: engine up (load + compile + warmup) {up:.1f}s", flush=True)
    rows, per = [], {}
    for n, kind, p in plan:
        if kind == "itl":
            wall, _ = gen(p, args.output_len)
            ttft = statistics.mean(x[0] for x in per[n])
            per_len = {"length": n, "ttft_s": [round(x[0], 4) for x in per[n]],
                       "ttft_mean_s": round(ttft, 4), "ttft_min_s": round(min(x[0] for x in per[n]), 4),
                       "engine_clock_ttft_s": [None if x[1] is None else round(x[1], 4) for x in per[n]],
                       "prefill_tok_s": round(n / ttft, 1),
                       "itl_ms": round(1000 * (wall - ttft) / (args.output_len - 1), 2),
                       "wall_out64_s": round(wall, 3)}
            if args.engine == "kiln":
                per_len["itl_output_ids"] = list(info.get("last_ids", []))
            rows.append(per_len)
            print(json.dumps(per_len), flush=True)
            continue
        wall, clock = gen(p, 1)
        print(f"{kind} {n}: {wall:.3f}s" + ("" if clock is None else f" (engine clock {clock:.3f}s)"), flush=True)
        if kind == "ttft":
            per.setdefault(n, []).append((wall, clock))
    if callable(info.get("spec")):
        info["spec"] = info["spec"]()
    res = {"engine": args.engine, "model": args.model, "tp": args.tp, "core_base": args.core_base,
           "max_model_len": args.max_model_len, "engine_up_s": round(up, 1), "rows": rows,
           "info": {k: (v if k != "compile_seconds" else {str(a): round(b, 1) for a, b in v.items()})
                    for k, v in info.items()}}
    print("RESULT", json.dumps(res, default=str))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(res, f, indent=1, default=str)
    sys.stdout.flush()
    # vllm-neuron's engine shutdown can hang after an error and asserts in PyTorch's allocator on a clean exit
    # ("Allocator for neuron is not a DeviceAllocator"); the results are written, so leave without the teardown. Its
    # engine-core and worker processes outlive os._exit and keep the NeuronCores ("cores busy" for the next run), so
    # they are killed first.
    try:
        import psutil

        for c in psutil.Process().children(recursive=True):
            c.kill()
    except Exception as e:  # noqa: BLE001
        print(f"child cleanup: {e}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
