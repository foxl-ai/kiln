"""The device correctness gate of a disaggregated deployment: tools/check_mixed.py's prompts (natural-text windows of
different lengths, greedy, --new-tokens each, top-2 logprobs) sent over HTTP to any OpenAI-compatible Kiln front (the
PD router), written in check_mixed's output format so the two compare with its --compare:

    python tools/check_mixed.py <serve_sweep args> --out ref.json            # one engine, in process (reference)
    python tools/check_pd.py --url http://router:8000 --tokenizer zai-org/GLM-5.3-Flash --out pd.json
    python tools/check_mixed.py --compare ref.json pd.json

Same prompt construction as check_mixed (LONG_TEXT, windows 997 tokens apart, --lengths cycled) and the same
--concurrency in flight. Outputs come back with return_token_ids (token ids and each token's chosen / top logprobs).
In a disaggregated run the first output token is computed by the prefill engine (a prefill chunk) and every later
one by the decode engine (a decode call).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def prompts_like_check_mixed(tok, n_req: int, lengths: list[int], text_file: str | None) -> list[list[int]]:
    from tools.check_ppl import LONG_TEXT

    text = LONG_TEXT
    if text_file:
        with open(text_file) as f:
            text = f.read()
    ids = tok(text)["input_ids"]
    while len(ids) < max(lengths) + n_req * 997:
        ids = ids + ids
    return [ids[i * 997 : i * 997 + lengths[i % len(lengths)]] for i in range(n_req)]


async def run(url: str, prompts, new_tokens: int, conc: int):
    import httpx

    out: list = [None] * len(prompts)
    nxt = iter(range(len(prompts)))
    async with httpx.AsyncClient(timeout=None) as client:
        async def worker():
            for i in nxt:
                body = {"prompt": prompts[i], "max_tokens": new_tokens, "temperature": 0.0, "ignore_eos": True,
                        "logprobs": 2, "return_token_ids": True}
                r = await client.post(url + "/v1/completions", json=body)
                if r.status_code != 200:
                    raise RuntimeError(f"request {i}: HTTP {r.status_code} {r.text[:300]}")
                out[i] = r.json()["choices"][0]

        await asyncio.gather(*(worker() for _ in range(conc)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--tokenizer", default="zai-org/GLM-5.3-Flash")
    ap.add_argument("--out", required=True)
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--lengths", default="8192,1500,6000,700,4096,3000,8000,2500")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    prompts = prompts_like_check_mixed(tok, a.requests, [int(x) for x in a.lengths.split(",")], a.text_file)
    t0 = time.perf_counter()
    got = asyncio.run(run(a.url.rstrip("/"), prompts, a.new_tokens, a.concurrency))
    wall = time.perf_counter() - t0
    reqs = []
    for p, c in zip(prompts, got):
        ids = c["token_ids"]
        reqs.append({"prompt_len": len(p), "output_ids": ids, "text": tok.decode(ids),
                     "logprobs": c["kiln_logprobs"],
                     "where": ["prefill chunk"] + ["decode call"] * (len(ids) - 1)})
    with open(a.out, "w") as f:
        json.dump({"mixed": 0, "wall_s": round(wall, 1), "calls": {}, "requests": reqs, "url": a.url}, f)
    print(f"RESULT requests={len(reqs)} wall_s={wall:.1f}", flush=True)
    for x in reqs[:4]:
        print(f"  {x['prompt_len']:>5}: {x['text'][:200]!r}", flush=True)


if __name__ == "__main__":
    main()
