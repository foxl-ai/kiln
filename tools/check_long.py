"""Long-context quality on the device as served (GLM-5.3-Flash up to 1M tokens, models/dsa_long.py):

    # NLL of a real long document by position band (prompt logprobs through the long path)
    python tools/check_long.py nll --model zai-org/GLM-5.3-Flash --tp 32 --dp-attention 4 --piecewise \
        --text-file /opt/kiln/data/long.txt --max-tokens 1044480
    # needle retrieval: a sentence with a number at depth d of a haystack of real text, asked at the end
    python tools/check_long.py needle --model ... --text-file /opt/kiln/data/long.txt --lengths 131072 524288 \
        --depths 0.1 0.5 0.9

A model whose long path is broken shows an NLL that stops falling (or jumps) with position, and fails the needle
at depths far from the end. tools/fetch_long_text.py makes the text (public-domain books, Project Gutenberg).

Page buckets: the dense ladder up to dsa_long.LONG_KEYS keys (the bucketed path), then powers of two up to the
longest context: every bucket is a graph set, so the ladder is coarse above LONG_KEYS (the long path's work grows
with the bucket, by at most 2x within one).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NEEDLE = "The special magic number for the Kiln project is {n}."
QUESTION = "What is the special magic number for the Kiln project mentioned in the text above? Answer with the number."
# GLM-5.3-Flash's chat format (chat_template.jinja at eb9eb208: "[gMASK]<sop>", "<|user|>", "<|assistant|><think>"),
# with an empty thinking block and the answer's first words, so the number is the next thing generated.
CHAT = "[gMASK]<sop><|user|>{body}<|assistant|><think></think>The special magic number is"


def page_buckets(max_len: int, page_size: int, long_keys: int) -> tuple[int, ...]:
    """Dense buckets of 64 .. long_keys / page_size pages, then powers of two up to the longest context."""
    top = -(-max_len // page_size)
    dense = [b for b in (64, 128, 264, 512) if b * page_size <= long_keys]
    out = [b for b in dense if b < top]
    b = 1 << max(0, (long_keys // page_size)).bit_length()
    while b < top:
        out.append(b)
        b *= 2
    out.append(top)
    return tuple(sorted(set(out)))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["nll", "needle"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dp-attention", type=int, default=1)
    ap.add_argument("--core-base", type=int, default=0)
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--piecewise", action="store_true")
    ap.add_argument("--kv-cache-gb", type=float, default=1.5)
    ap.add_argument("--kv-cache-dtype", default="fp8")
    ap.add_argument("--prefill-tokens", type=int, default=4096, help="prefill tokens per step (all DP groups)")
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--max-tokens", type=int, default=131072, help="nll: tokens scored")
    ap.add_argument("--skip-tokens", type=int, default=0,
                    help="nll: start this many tokens into the text (the same span scored without what precedes it)")
    ap.add_argument("--lengths", type=int, nargs="+", default=[131072], help="needle: prompt lengths in tokens")
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--answer-tokens", type=int, default=12)
    ap.add_argument("--max-num-seqs", type=int, default=None, help="default: one per DP group")
    ap.add_argument("--page-buckets", type=int, nargs="+", default=None)
    ap.add_argument("--page-size", type=int, default=32, help="context-parallel DSA wants 32 x attention TP")
    # To run on a serving configuration's graphs (bench/serve_sweep.py), match its engine config exactly:
    ap.add_argument("--max-model-len", type=int, default=0, help="default: the longest prompt + answer + 64")
    ap.add_argument("--decode-buckets", type=int, nargs="+", default=None, help="per DP group")
    ap.add_argument("--state-checkpoints", type=int, default=None)
    ap.add_argument("--overlap", action="store_true")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--forced", action="store_true",
                    help="needle: teacher-forced, the answer's tokens appended to the prompt and scored by prompt "
                         "logprobs (a case passes when every answer token is the argmax, rank 1: what greedy decoding "
                         "would produce), one prefill and no decode (a pipeline stage, engine/pp.py, decodes nothing)")
    # One long prefill over several engines (engine/pp.py): the same command on every stage (its own --pp-stage).
    ap.add_argument("--pp-stages", type=int, default=1)
    ap.add_argument("--pp-stage", type=int, default=0)
    ap.add_argument("--pp-split", type=int, nargs="+", default=None)
    ap.add_argument("--pp-listen", default=None)
    ap.add_argument("--pp-next", default=None)
    return ap


def max_len(args) -> int:
    """The engine's max_model_len for a run: the scored tokens (nll) or the longest needle prompt plus its answer."""
    if getattr(args, "max_model_len", 0):
        return args.max_model_len
    if args.mode == "nll":
        return args.max_tokens + 64
    return max(args.lengths) + args.answer_tokens + 64


def engine_config(args, path: str | None = None):
    """The EngineConfig this check runs with (tools/compile_farm.py capture --tool check_long builds the same one;
    nll runs prompt logprobs, so capture it with --plp)."""
    from kiln.config import EngineConfig
    from kiln.models import dsa_long
    from kiln.models.loader import resolve_model_path

    max_len_ = max_len(args)
    dp = args.dp_attention
    seqs = args.max_num_seqs or dp
    return EngineConfig(
        model_path=path or resolve_model_path(args.model), device=args.device, dtype=torch.bfloat16,
        page_size=args.page_size,
        max_num_seqs=seqs, max_model_len=max_len_, max_prefill_tokens=args.prefill_tokens, kv_cache_gb=args.kv_cache_gb,
        kv_cache_dtype=args.kv_cache_dtype,
        decode_batch_buckets=tuple(args.decode_buckets) if getattr(args, "decode_buckets", None) else (-(-seqs // dp),),
        overlap=bool(getattr(args, "overlap", False)),
        **({"state_checkpoints": args.state_checkpoints} if getattr(args, "state_checkpoints", None) is not None else {}),
        prefill_token_buckets=(args.prefill_tokens // dp,),
        page_buckets=tuple(args.page_buckets) if args.page_buckets else page_buckets(max_len_, args.page_size,
                                                                                     dsa_long.LONG_KEYS),
        tp=args.tp, tp_core_base=args.core_base, dp_attention=dp, piecewise=args.piecewise,
        **({"pp_stages": args.pp_stages, "pp_stage": args.pp_stage, "pp_listen": args.pp_listen,
            "pp_next": args.pp_next, "pp_split": tuple(args.pp_split) if args.pp_split else None}
           if getattr(args, "pp_stages", 1) > 1 else {}))


def _text_ids(tok, path: str, n: int, skip: int = 0) -> list[int]:
    with open(path) as f:
        text = f.read()
    ids = tok(text)["input_ids"]
    if len(ids) < skip + n:
        raise SystemExit(f"{path}: {len(ids)} tokens, fewer than the {skip} + {n} asked for")
    return ids[skip:skip + n]


def bands(n: int) -> list[tuple[int, int]]:
    """Position bands [0, 1k), [1k, 4k), [4k, 16k), ... up to n."""
    out, a, b = [], 0, 1024
    while a < n:
        out.append((a, min(b, n)))
        a, b = b, b * 4
    return out


def run_nll(args) -> dict:
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    eng = LLMEngine(engine_config(args))
    try:
        ids = _text_ids(eng.tokenizer, args.text_file, args.max_tokens, args.skip_tokens)
        t0 = time.time()
        (r,) = eng.generate([ids], SamplingParams(max_new_tokens=1, prompt_logprobs=0))
        dt = time.time() - t0
        _local_k_line(eng, None, "nll")
    finally:
        eng.close()
    lps = [v[0] for _, v in sorted(r.prompt_logprobs.items())]  # position q: log p(ids[q] | ids[:q]), q >= 1
    out = {"tokens": len(ids), "seconds": dt, "mean_nll": -sum(lps) / len(lps), "bands": [], "logprobs": lps}
    print(f"RESULT mean_nll={out['mean_nll']:.4f} tokens={len(ids)} prefill_s={dt:.1f} tok_s={len(ids) / dt:.1f}")
    for a, b in bands(len(lps) + 1):
        seg = lps[max(a - 1, 0):b - 1]
        if seg:
            nll = -sum(seg) / len(seg)
            out["bands"].append({"from": a, "to": b, "nll": nll, "n": len(seg)})
            print(f"  positions [{a:>8}, {b:>8}): nll {nll:.4f} over {len(seg)} tokens")
    return out


def _local_k_line(eng, before, what: str) -> None:
    """KILN_DSA_CP_LOCAL_K (models/mla.py): this run's certificate counters (rank 0's row group), when they are on."""
    model = getattr(getattr(eng, "runner", None), "model", None)
    if model is None:
        return
    from kiln.models import mla

    line = mla.local_k_report(before, mla.local_k_counts(model), what)
    if line:
        print(line, flush=True)


def needle_prompt(tok, hay: list[int], length: int, depth: float, n: int) -> list[int]:
    """CHAT around a haystack of real text with the needle sentence inserted at `depth` (a fraction of the
    haystack, moved to the next sentence end), trimmed so the whole prompt is `length` tokens."""
    needle = tok("\n" + NEEDLE.format(n=n) + "\n", add_special_tokens=False)["input_ids"]
    head = tok(CHAT.split("{body}")[0], add_special_tokens=False)["input_ids"]
    tail = tok("\n\n" + QUESTION + CHAT.split("{body}")[1], add_special_tokens=False)["input_ids"]
    room = length - len(head) - len(tail) - len(needle)
    body = hay[:room]
    at = int(room * depth)
    dot = tok(".", add_special_tokens=False)["input_ids"][-1]
    while at < len(body) and body[at - 1] != dot:
        at += 1
    return head + body[:at] + needle + body[at:] + tail


def run_needle(args) -> dict:
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    longest = max(args.lengths)
    eng = LLMEngine(engine_config(args))
    results = []
    try:
        hay = _text_ids(eng.tokenizer, args.text_file, longest)
        cases = []
        for L in args.lengths:
            for d in args.depths:
                n = 1000003 + int(L * 7 + d * 1e6) % 8999991  # a 7-digit number per case
                cases.append((L, d, n, needle_prompt(eng.tokenizer, hay, L, d, n)))
        if getattr(args, "forced", False):
            return _needle_forced(eng, cases, args)
        t0 = time.time()
        reqs = eng.generate([c[3] for c in cases], SamplingParams(max_new_tokens=args.answer_tokens, ignore_eos=True))
        dt = time.time() - t0
        _local_k_line(eng, None, "needle")
        for (L, d, n, p), r in zip(cases, reqs):
            text = eng.tokenizer.decode(r.output_ids)
            ok = str(n) in text
            results.append({"length": len(p), "depth": d, "number": n, "answer": text, "ok": ok})
            print(f"  length {len(p):>8} depth {d:4.2f}: {'PASS' if ok else 'FAIL'} want {n} got {text!r}")
    finally:
        eng.close()
    passed = sum(r["ok"] for r in results)
    print(f"RESULT needle {passed}/{len(results)} seconds={dt:.1f}")
    return {"cases": results, "passed": passed, "seconds": dt}


def _needle_forced(eng, cases, args) -> dict:
    """The needle cases teacher-forced: each prompt plus the tokens of " <number>." (what the chat format's "The
    special magic number is" continues with), scored by prompt logprobs from the first answer token on; a case passes
    when every answer token is the argmax (rank 1). One request at a time, in order (every pipeline stage submits the
    same requests in the same order). The caller closes the engine."""
    from kiln.engine.request import SamplingParams

    results = []
    t0 = time.time()
    for L, d, n, p in cases:
        ans = eng.tokenizer(" " + str(n) + ".", add_special_tokens=False)["input_ids"]
        ids = p + ans
        sp = SamplingParams(max_new_tokens=1, ignore_eos=True, prompt_logprobs=0, prompt_logprobs_start=len(p))
        (r,) = eng.generate([ids], sp)
        got = [r.prompt_logprobs.get(len(p) + i) for i in range(len(ans))]
        ranks = [g[1] if g is not None else None for g in got]
        ok = all(x == 1 for x in ranks)
        results.append({"length": len(ids), "depth": d, "number": n, "answer_tokens": ans, "ranks": ranks,
                        "logprobs": [g[0] if g is not None else None for g in got], "ok": ok})
        print(f"  length {len(ids):>8} depth {d:4.2f}: {'PASS' if ok else 'FAIL'} number {n} answer ranks {ranks}",
              flush=True)
    dt = time.time() - t0
    _local_k_line(eng, None, "needle-forced")
    passed = sum(r["ok"] for r in results)
    print(f"RESULT needle-forced {passed}/{len(results)} seconds={dt:.1f}")
    return {"cases": results, "passed": passed, "seconds": dt, "forced": True}


def main() -> None:
    args = build_parser().parse_args()
    out = run_nll(args) if args.mode == "nll" else run_needle(args)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
