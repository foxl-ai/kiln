"""Greedy generation on a real device, checked against transformers on the host CPU.

    python tools/check_device.py --model Qwen/Qwen3-0.6B --device neuron --tokens 32

For each prompt it prints how many generated tokens match transformers (fp32, CPU), and
at the first divergence the reference's top-2 logit margin: a bf16 device run is expected
to drift only where the reference itself is nearly tied. Also prints per-bucket compile
time and the steady-state step time of a batched decode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Trainium is a machine learning accelerator built by",
    "1, 2, 3, 5, 8, 13,",
    "In 1969, the first humans to walk on the Moon were",
    "The quick brown fox",
    "Explain why the sky is blue in one sentence:",
    "SELECT name FROM users WHERE",
]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                    help="fp32 with KILN_CC_ARGS=--auto-cast=none checks the device path without bf16 rounding")
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--prefill-tokens", type=int, default=128)
    ap.add_argument("--bench-steps", type=int, default=64)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--core-base", type=int, default=0, help="first NeuronCore (tp_core_base)")
    ap.add_argument("--attention-tp", type=int, default=None,
                    help="tensor parallelism of the attention / linear-attention heads (EngineConfig.attention_tp)")
    ap.add_argument("--dp-attention", type=int, default=1,
                    help="DP-attention groups (EngineConfig.dp_attention); --max-num-seqs stays the total")
    ap.add_argument("--out-json", default=None,
                    help="write every prompt's generated ids and chosen-token logprobs here (run-to-run comparisons)")
    ap.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8"])
    ap.add_argument("--weight-dtype", default="auto", choices=["auto", "bf16", "fp8", "fp8-experts"])
    ap.add_argument("--piecewise", action="store_true", help="one graph per layer kind")
    ap.add_argument("--piecewise-group", type=int, default=None, help="layers per piecewise graph")
    ap.add_argument("--kv-cache-gb", type=float, default=2.0)
    ap.add_argument("--no-reference", action="store_true", help="skip the transformers comparison")
    ap.add_argument("--host-reference", action="store_true",
                    help="compare against Kiln's own CPU path in fp32 at tp=1 (CPU-checked against transformers on "
                         "random weights), for models whose transformers fp32 copy does not fit the host")
    ap.add_argument("--mxfp4-packed", action="store_true")
    ap.add_argument("--bench-only", action="store_true", help="skip the generate pass and its comparison")
    ap.add_argument("--bench-batches", default=None,
                    help="comma list of decode batch sizes to time, each its own bucket (default: --max-num-seqs)")
    ap.add_argument("--reference-file", default=None,
                    help="JSON cache of the host reference: written by --host-reference-only, read by --reference host")
    ap.add_argument("--host-reference-only", action="store_true",
                    help="compute the host reference into --reference-file and exit (no device engine)")
    ap.add_argument("--reference-json", default=None,
                    help="compare against tools/hf_reference.py output instead of running transformers here")
    ap.add_argument("--no-vocab-parallel", action="store_true", help="replicate embedding and lm_head")
    ap.add_argument("--spec-method", default=None, choices=["ngram", "suffix", "mtp"])
    ap.add_argument("--spec-k", type=int, default=3)
    ap.add_argument("--tokenizer", default=None,
                    help="tokenize the prompts with this repo (a checkpoint whose tokenizer needs remote code)")
    ap.add_argument("--reference", default=None,
                    help="checkpoint transformers loads for the reference (e.g. the dequantized twin of an FP8 one)")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=JSON",
                    help="replace a config.json key for both Kiln and the reference, e.g. index_topk=16")
    return ap


def engine_config(args, path: str):
    """The EngineConfig this check runs with (tools/compile_farm.py capture --tool check_device)."""
    from kiln.config import EngineConfig

    batches = [int(b) for b in args.bench_batches.split(",")] if args.bench_batches else [args.max_num_seqs]
    return EngineConfig(
        model_path=path, device=args.device, dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32,
        page_size=args.page_size,
        max_num_seqs=max(args.max_num_seqs, *batches), max_model_len=args.max_model_len,
        max_prefill_tokens=args.prefill_tokens, kv_cache_gb=args.kv_cache_gb, piecewise=args.piecewise, vocab_parallel=not args.no_vocab_parallel,
        spec_method=args.spec_method, spec_k=args.spec_k, piecewise_group=args.piecewise_group,
        decode_batch_buckets=tuple(sorted({-(-n // args.dp_attention) for n in (args.max_num_seqs, *batches)})),
        prefill_token_buckets=(args.prefill_tokens // args.dp_attention,),
        page_buckets=(4, args.max_model_len // args.page_size), tp=args.tp,
        tp_core_base=getattr(args, "core_base", 0), attention_tp=args.attention_tp,
        dp_attention=args.dp_attention,
        kv_cache_dtype=args.kv_cache_dtype, weight_dtype=args.weight_dtype, mxfp4_packed=args.mxfp4_packed,
    )


def main() -> None:
    args = build_parser().parse_args()
    batches = [int(b) for b in args.bench_batches.split(",")] if args.bench_batches else [args.max_num_seqs]

    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams
    from kiln.models.loader import resolve_model_path

    path = resolve_model_path(args.model)  # huggingface_hub only; transformers is not imported yet
    if args.override:
        path = with_overrides(path, args.override)
    if args.host_reference_only:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path)
        refs = host_reference(path, [tok(p)["input_ids"] for p in PROMPTS], args)
        with open(args.reference_file, "w") as f:
            json.dump(refs, f)
        for out, _ in refs:
            print(f"  host {tok.decode(out)!r}")
        return
    cfg = engine_config(args, path)
    t0 = time.perf_counter()
    eng = LLMEngine(cfg)
    print(f"engine up in {time.perf_counter() - t0:.1f}s (weights {eng.load_seconds:.1f}s), pages={eng.pool.num_pages}, "
          f"tp={cfg.tp} attention_tp={eng.model.attn_tp} dp_attention={cfg.dp_attention}")

    tok = eng.tokenizer
    if args.tokenizer:
        from transformers import AutoTokenizer  # after the engine: see LLMEngine.__init__

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
    prompts = [tok(p)["input_ids"] for p in PROMPTS]
    t0 = time.perf_counter()
    reqs = eng.generate(prompts, SamplingParams(max_new_tokens=args.tokens, ignore_eos=True, logprobs=0))
    print(f"first generate (includes compiles) {time.perf_counter() - t0:.1f}s")
    if args.spec_method:
        print(f"speculation {args.spec_method} k={args.spec_k}: accepted {eng.spec_accepted}/{eng.spec_proposed} "
              f"drafted tokens ({eng.spec_accepted / max(eng.spec_proposed, 1):.0%})")
    for k, v in sorted(eng.runner.compile_seconds.items()):
        print(f"  compile {k}: {v:.1f}s")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"tp": cfg.tp, "attention_tp": eng.model.attn_tp, "dp_attention": cfg.dp_attention,
                       "prompts": prompts,
                       "ids": [r.output_ids for r in reqs], "logprobs": [[x[0] for x in r.logprobs] for r in reqs]}, f)

    results = []
    if args.bench_only:
        pass
    elif args.host_reference and not args.no_reference:
        import os

        if args.reference_file and os.path.exists(args.reference_file):
            with open(args.reference_file) as f:
                refs = json.load(f)
        else:
            refs = host_reference(path, prompts, args)
        # Teacher forcing the other way round: the device scores the host's tokens (prompt
        # logprobs from the end of the prompt), against the host's own logprob of each, over all
        # positions rather than up to the first divergence.
        forced = eng.generate([ids + ref[0][: args.tokens] for ids, ref in zip(prompts, refs)],
                              [SamplingParams(max_new_tokens=1, ignore_eos=True, prompt_logprobs=1,
                                              prompt_logprobs_start=len(ids)) for ids in prompts])
        for ids, r, f, (ref, top2) in zip(prompts, reqs, forced, refs):
            ref = ref[: args.tokens]
            n_match = next((i for i, (a, b) in enumerate(zip(r.output_ids, ref)) if a != b), len(ref))
            margin = float(top2[n_match][0] - top2[n_match][1]) if n_match < len(ref) else None
            dlp = max(abs(f.prompt_logprobs[len(ids) + i][0] - top2[i][0]) for i in range(len(ref)))
            worst_rank = max(f.prompt_logprobs[len(ids) + i][1] for i in range(len(ref)))
            results.append((n_match, margin, dlp, worst_rank))
            print(f"  match {n_match:>3}/{len(ref)}  margin_at_divergence={margin}  teacher-forced "
                  f"max|dlogprob|={dlp:.3f} worst rank={worst_rank}  {tok.decode(r.output_ids)!r}")
    elif args.no_reference:
        # Models too large for a host-side transformers run (MiMo-V2.6-Flash is 1.2 TB in
        # fp32): report the text and the chosen-token logprobs instead.
        for ids, r in zip(prompts, reqs):
            lp = [x[0] for x in r.logprobs]
            mean_lp = sum(lp) / max(len(lp), 1)
            results.append((len(r.output_ids), mean_lp))
            print(f"  mean chosen logprob {mean_lp:7.3f}  {tok.decode(r.output_ids)!r}")
    elif args.reference_json:
        with open(args.reference_json) as f:
            refs = json.load(f)["prompts"]
        for ids, r, want in zip(prompts, reqs, refs):
            if want["ids"] != ids:
                raise SystemExit(f"tokenization differs from the reference for {want['text']!r}")
            ref = want["greedy"][: args.tokens]
            n_match = next((i for i, (a, b) in enumerate(zip(r.output_ids, ref)) if a != b), len(ref))
            margin = want["margins"][n_match] if n_match < len(ref) else None
            results.append((n_match, margin))
            print(f"  match {n_match:>3}/{len(ref)}  margin_at_divergence={margin}  {tok.decode(r.output_ids)!r}")
    else:
        from tools.hf_reference import load_reference  # after the engine: see LLMEngine.__init__

        ref_path = path if args.reference is None else resolve_model_path(args.reference)
        if args.reference is not None and args.override:
            ref_path = with_overrides(ref_path, args.override)
        ref_model = load_reference(ref_path)
        for ids, r in zip(prompts, reqs):
            with torch.no_grad():
                ref = ref_model.generate(torch.tensor([ids]), max_new_tokens=args.tokens, do_sample=False,
                                         eos_token_id=None, pad_token_id=0)[0, len(ids):].tolist()
            n_match = next((i for i, (a, b) in enumerate(zip(r.output_ids, ref)) if a != b), len(ref))
            margin = None
            if n_match < len(ref):
                with torch.no_grad():
                    logits = ref_model(torch.tensor([ids + ref[:n_match]])).logits[0, -1]
                top2 = torch.topk(logits, 2).values
                margin = float(top2[0] - top2[1])
            # The reference scores Kiln's own tokens (teacher forcing): a precision measure that
            # does not stop at the first divergence.
            with torch.no_grad():
                lp = torch.log_softmax(ref_model(torch.tensor([ids + r.output_ids])).logits[0].double(), -1)
            ref_lp = lp[len(ids) - 1 : -1].gather(1, torch.tensor(r.output_ids).unsqueeze(1)).squeeze(1)
            dlps = [abs(a[0] - b) for a, b in zip(r.logprobs, ref_lp.tolist())]
            dlp = max(dlps)
            results.append((n_match, margin, dlp))
            print(f"  match {n_match:>3}/{len(ref)}  margin_at_divergence={margin}  max|dlogprob|={dlp:.2e} "
                  f"(token {dlps.index(dlp)})  {tok.decode(r.output_ids)!r}")

    # Steady-state decode: a full batch of long-running requests, per batch size; two untimed
    # steps first (a bucket's first launch compiles or loads its graphs).
    per_step = 1 + (args.spec_k if args.spec_method else 0)  # long enough to stay running through the bench
    timed = {}
    for B in batches:
        bench = [eng.add_request(prompts[0], SamplingParams(max_new_tokens=(args.bench_steps + 12) * per_step,
                                                            ignore_eos=True)) for _ in range(B)]
        while eng.scheduler.waiting or any(not r.is_decoding for r in eng.scheduler.running):
            eng.step()
        for _ in range(2):
            eng.step()
        times = []
        before = sum(len(r.output_ids) for r in bench)
        for _ in range(args.bench_steps):
            eng.step()
            times.append(eng.last_step.seconds)
        produced = sum(len(r.output_ids) for r in bench) - before
        print(f"decode tokens: {produced} in {sum(times):.2f}s = {produced / sum(times):.1f} tok/s "
              f"({produced / max(len(times), 1) / max(B, 1):.2f} tokens per sequence per step)")
        times.sort()
        b = eng.last_step.num_decode
        p50 = times[len(times) // 2]
        print(f"decode B={b}: step p50 {p50 * 1e3:.2f} ms, p90 {times[int(len(times) * 0.9)] * 1e3:.2f} ms, "
              f"{b / p50:.0f} tok/s")
        timed[b] = p50 * 1e3
        for r in bench:  # free the batch before the next size
            eng.abort(r)
        while eng.scheduler.running or eng.scheduler.waiting:
            eng.step()
    print("RESULT", json.dumps({"match": results, "decode_p50_ms": timed,
                                "compile_s": {str(k): v for k, v in eng.runner.compile_seconds.items()}}))


def host_reference(path: str, prompts: list[list[int]], args) -> list:
    """Greedy on the host, in fp32 and at tp=1, from the same weights (FP8 experts stay FP8 and are
    dequantized per matmul): per prompt (tokens, top-2 logprobs at each step); the margin at a
    divergence is the host's top-2 logprob gap there."""
    import os

    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    torch.set_num_threads(os.cpu_count())
    host = LLMEngine(EngineConfig(
        model_path=path, device="cpu", dtype=torch.float32, page_size=args.page_size,
        max_num_seqs=len(prompts), max_model_len=args.max_model_len, max_prefill_tokens=args.prefill_tokens,
        num_pages=len(prompts) * (args.max_model_len // args.page_size) + 8, weight_dtype=args.weight_dtype))
    t = time.perf_counter()
    refs = host.generate(prompts, SamplingParams(max_new_tokens=args.tokens, ignore_eos=True, logprobs=2))
    print(f"host reference: {len(prompts)} x {args.tokens} tokens in {time.perf_counter() - t:.0f}s")
    host.close()
    return [(h.output_ids, [list(lp[2][:2]) for lp in h.logprobs]) for h in refs]


def with_overrides(path: str, overrides: list[str]) -> str:
    """A directory of symlinks to `path` whose config.json has the given keys replaced."""
    import os
    import tempfile

    out = tempfile.mkdtemp(prefix="kiln-override-")
    for f in os.listdir(path):
        if f != "config.json":
            os.symlink(os.path.join(path, f), os.path.join(out, f))
    with open(os.path.join(path, "config.json")) as fh:
        c = json.load(fh)
    for kv in overrides:
        k, v = kv.split("=", 1)
        c[k] = json.loads(v)
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(c, fh)
    print(f"config overrides {overrides} -> {out}")
    return out


def print_piece_times() -> None:
    from kiln.engine.model_runner import PIECE_TIMES

    for (shape, gi), ts in sorted(PIECE_TIMES.items()):
        ts = sorted(ts)
        print(f"  piece {gi:>2} h{list(shape)}: n={len(ts)} p50 {ts[len(ts) // 2] * 1e3:.2f} ms")
    from kiln.engine.model_runner import EXEC_TIMES

    for key, ts in sorted(EXEC_TIMES.items(), key=lambda kv: str(kv[0])):  # KILN_PROFILE_EXEC=1
        ts = sorted(ts)
        print(f"  exec {key}: n={len(ts)} p50 {ts[len(ts) // 2] * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
    print_piece_times()
