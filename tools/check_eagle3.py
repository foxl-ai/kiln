"""EAGLE-3 on a real target and draft: acceptance, per-token latency and greedy agreement against no speculation.

    python tools/check_eagle3.py --mode plain --model Qwen/Qwen3-8B --tp 2 --out-json plain.json
    python tools/check_eagle3.py --mode eagle3 --model Qwen/Qwen3-8B --draft RedHatAI/Qwen3-8B-speculator.eagle3 \
        --tp 2 --k 3 --out-json eagle3.json
    python tools/check_eagle3.py --compare plain.json eagle3.json

Chat prompts (the distribution EAGLE-3 drafts are trained on: assistant answers, not raw text), rendered with the
model's chat template (thinking off for Qwen3), batch 1, greedy. For each prompt: tokens per second without and
with speculation, the drafts proposed / accepted, and the first position where the two greedy outputs differ (bf16
verify and decode graphs can round a near tie differently; a difference there is not a draft bug, since the verify
step scores with the target).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain how a hash map handles collisions, in three short paragraphs.",
    "A train leaves at 9:40 and arrives at 13:05. How long is the trip? Show the steps.",
    "Summarize the plot of Romeo and Juliet in five sentences.",
    "List five practical tips for writing clear technical documentation.",
    "Translate into French: The weather is nice today, so we will walk to the market.",
]


def _prompt_ids(tok) -> list[list[int]]:
    ids = []
    for p in PROMPTS:
        t = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True,
                                    enable_thinking=False)
        ids.append(list(t["input_ids"] if hasattr(t, "keys") else t))
    return ids


def run_kiln(a) -> list[dict]:
    import torch

    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    ladder = lambda v: tuple(int(x) for x in v.split(","))  # noqa: E731
    kw = dict(model_path=a.model, device=a.device, dtype=torch.bfloat16, page_size=a.page_size, max_num_seqs=1,
              max_model_len=a.max_model_len, kv_cache_gb=a.kv_cache_gb, tp=a.tp, piecewise=not a.no_piecewise,
              decode_batch_buckets=(1,), page_buckets=ladder(a.page_buckets),
              prefill_token_buckets=ladder(a.prefill_buckets), max_prefill_tokens=max(ladder(a.prefill_buckets)),
              overlap=a.overlap, decode_whole=a.decode_whole)  # overlap: speculative steps run synchronously at k > 1
    if a.mode == "eagle3":
        kw.update(spec_method="mtp", spec_draft_model=a.draft, spec_k=a.k)
    elif a.mode == "ngram":  # the same verify path with prompt-lookup drafts: tells verify rounding from a draft defect
        kw.update(spec_method="ngram", spec_k=a.k)
    eng = LLMEngine(EngineConfig(**kw))
    ids = _prompt_ids(eng.tokenizer)
    sp = SamplingParams(max_new_tokens=a.max_new_tokens, temperature=0.0, ignore_eos=True,
                        logprobs=2 if a.logprobs else None)
    eng.generate([ids[0]], SamplingParams(max_new_tokens=8, temperature=0.0))  # the first calls compile or load
    rows = [{"tok_s": [], "proposed": 0, "accepted": 0, "ids": []} for _ in ids]
    for _ in range(a.passes):
        for row, x in zip(rows, ids):
            prop0, acc0 = eng.spec_proposed, eng.spec_accepted
            t0 = time.perf_counter()
            r = eng.generate([x], sp)[0]
            dec = time.perf_counter() - t0 - (r.first_token_time - r.arrival_time)
            row["tok_s"].append(round((a.max_new_tokens - 1) / dec, 2))
            row["proposed"] += eng.spec_proposed - prop0
            row["accepted"] += eng.spec_accepted - acc0
            row["ids"].append(list(r.output_ids))
            if a.logprobs:  # per output position: the top-1 minus top-2 logprob of the greedy choice
                row.setdefault("margins", []).append([round(t[2][0] - t[2][1], 5) for t in r.logprobs])
    return rows


def run_vllm(a) -> list[dict]:
    """vllm-neuron 0.24 on the same prompts and cores: LLM(...) offline, its EAGLE-3 through --speculative-config's
    dict (vllm-neuron docs/tutorials/tutorial-eagle3-speculative-decoding-llama-3-1.md), acceptance from the engine's
    spec-decode counters (llm.get_metrics()). The decode time of a request is its wall minus a max_tokens=1 run's."""
    os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
    from vllm import LLM, SamplingParams, TokensPrompt

    kw = dict(model=a.model, dtype="bfloat16", tensor_parallel_size=a.tp, max_num_seqs=1, max_model_len=a.max_model_len,
              enable_prefix_caching=False, disable_log_stats=False)
    if a.mode == "eagle3":
        kw["speculative_config"] = {"method": "eagle3", "model": a.draft, "num_speculative_tokens": a.k}
    llm = LLM(**kw)
    ids = _prompt_ids(llm.get_tokenizer())

    def counters():
        out = {}
        try:
            for m in llm.get_metrics():
                if "spec_decode" in m.name and hasattr(m, "value"):
                    out[m.name] = m.value
        except Exception as e:  # noqa: BLE001
            out["error"] = str(e)
        return out

    full = SamplingParams(max_tokens=a.max_new_tokens, temperature=0.0, ignore_eos=True)
    one = SamplingParams(max_tokens=1, temperature=0.0)
    llm.generate([TokensPrompt(prompt_token_ids=ids[0])], SamplingParams(max_tokens=8, temperature=0.0))
    rows = [{"tok_s": [], "proposed": 0, "accepted": 0, "ids": []} for _ in ids]
    for _ in range(a.passes):
        for row, x in zip(rows, ids):
            t0 = time.perf_counter()
            llm.generate([TokensPrompt(prompt_token_ids=x)], one, use_tqdm=False)
            ttft = time.perf_counter() - t0
            c0 = counters()
            t0 = time.perf_counter()
            r = llm.generate([TokensPrompt(prompt_token_ids=x)], full, use_tqdm=False)[0]
            wall = time.perf_counter() - t0
            c1 = counters()
            row["tok_s"].append(round((a.max_new_tokens - 1) / (wall - ttft), 2))
            row["proposed"] += int(c1.get("vllm:spec_decode_num_draft_tokens", 0) - c0.get("vllm:spec_decode_num_draft_tokens", 0))
            row["accepted"] += int(c1.get("vllm:spec_decode_num_accepted_tokens", 0)
                                   - c0.get("vllm:spec_decode_num_accepted_tokens", 0))
            row["ids"].append(list(r.outputs[0].token_ids))
            row["counters"] = c1
    return rows


def _first_diff(u, v):
    return next((j for j, (x, y) in enumerate(zip(u, v)) if x != y), None)


def compare(plain: list[dict], eagle: list[dict], k: int) -> dict:
    """Per prompt: tok/s (mean over passes), acceptance, and three greedy comparisons: plain pass 0 vs plain pass 1
    (the engine's own run-to-run jitter), plain vs eagle3 (first difference), eagle3 pass 0 vs pass 1."""
    out = {"prompts": []}
    for p, pl, ea in zip(PROMPTS, plain, eagle):
        steps = max(1, ea["proposed"] // k)
        mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
        row = {"prompt": p[:48], "plain_tok_s": round(mean(pl["tok_s"]), 2), "eagle3_tok_s": round(mean(ea["tok_s"]), 2),
               "speedup": round(mean(ea["tok_s"]) / mean(pl["tok_s"]), 3), "accepted": ea["accepted"],
               "proposed": ea["proposed"], "accept_rate": round(ea["accepted"] / max(1, ea["proposed"]), 3),
               "acceptance_length": round(1 + ea["accepted"] / steps, 2),
               "plain_vs_plain_first_diff": _first_diff(*pl["ids"][:2]) if len(pl["ids"]) > 1 else "n/a",
               "plain_vs_eagle3_first_diff": _first_diff(pl["ids"][0], ea["ids"][0]),
               "eagle3_vs_eagle3_first_diff": _first_diff(*ea["ids"][:2]) if len(ea["ids"]) > 1 else "n/a"}
        d = row["plain_vs_eagle3_first_diff"]
        if d is not None and pl.get("margins"):  # how close the plain run's own choice was where the two part
            row["plain_margin_at_diff"] = pl["margins"][0][d]
            row["plain_margin_rank"] = sum(m <= pl["margins"][0][d] for m in pl["margins"][0])  # of max_new_tokens
        out["prompts"].append(row)
        print(json.dumps(row), flush=True)
    tot_a, tot_p = sum(r["accepted"] for r in out["prompts"]), sum(r["proposed"] for r in out["prompts"])
    out["accept_rate"] = round(tot_a / max(1, tot_p), 3)
    out["acceptance_length"] = round(1 + tot_a / max(1, tot_p // k), 2)
    out["mean_speedup"] = round(sum(r["speedup"] for r in out["prompts"]) / len(PROMPTS), 3)
    print("SUMMARY", json.dumps({key: v for key, v in out.items() if key != "prompts"}), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--mode", choices=["plain", "eagle3", "ngram"], default=None, help="run one engine, write --out-json")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("PLAIN_JSON", "EAGLE3_JSON"))
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--kv-cache-gb", type=float, default=1.0)
    ap.add_argument("--page-buckets", default="64")
    ap.add_argument("--prefill-buckets", default="256")
    ap.add_argument("--passes", type=int, default=2, help="generations of every prompt (run-to-run jitter)")
    ap.add_argument("--engine", choices=["kiln", "vllm"], default="kiln")
    ap.add_argument("--overlap", action="store_true", help="Kiln: overlap scheduling (the plain arm's best decode)")
    ap.add_argument("--decode-whole", action="store_true", help="Kiln: a decode call as one graph (KILN_DECODE_WHOLE)")
    ap.add_argument("--no-piecewise", action="store_true",
                    help="Kiln: whole-model graphs for every call (prefill, decode, verify, MTP), as vllm-neuron compiles")
    ap.add_argument("--logprobs", action="store_true",
                    help="Kiln: keep each greedy choice's top-1 minus top-2 margin (where a plain and a speculative "
                         "output part, a near tie is rounding, a wide margin a verify-path defect)")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    if a.compare:
        pl, ea = (json.load(open(f)) for f in a.compare)
        res = compare(pl["rows"], ea["rows"], ea["k"])
        if a.out_json:
            json.dump(res, open(a.out_json, "w"), indent=1)
        return
    rows = run_kiln(a) if a.engine == "kiln" else run_vllm(a)
    if a.out_json:
        json.dump({"model": a.model, "draft": a.draft, "mode": a.mode, "engine": a.engine, "k": a.k, "rows": rows},
                  open(a.out_json, "w"))
    print(json.dumps([{key: v for key, v in r.items() if key not in ("ids", "counters")} for r in rows]), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
