"""Where a disaggregated request's time to first token goes, request by request: the client's send and first token,
the router's intervals (its kiln:pd_router_* metric deltas around each request: t0 -> prefill call start, the prefill
call, prefill end -> first chunk) and both engines' step logs (GET /debug/steps). Sequential lone requests; the first
is a warm-up. Run it on the host of the router and both engines, so their monotonic clocks are one
(docs/neuron-notes.md "The 8K TTFT outside the prefill call").

    python tools/pd_timeline.py --router http://127.0.0.1:8000 --prefill http://127.0.0.1:8100 \
        --decode http://127.0.0.1:8101 --n 10 --input-len 8192 --out timeline.json [--text-file f.txt --tokenizer X]

--text-file sends string prompts (disjoint windows of the file, --input-len tokens each by --tokenizer) instead of
random ids, so the servers' (or the router's) tokenization is part of the path.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time

import httpx


def metric(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith(name + " "):
            return float(line.split()[1])
    return 0.0


NAMES = ["kiln:pd_router_e2e_ttft_seconds_sum", "kiln:pd_router_prefill_seconds_sum",
         "kiln:pd_router_prefill_call_seconds_sum", "kiln:pd_router_queue_seconds_sum",
         "kiln:pd_router_prefill_queue_seconds_sum"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", required=True)
    ap.add_argument("--prefill", required=True)
    ap.add_argument("--decode", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--input-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=100000)
    ap.add_argument("--text-file", default=None, help="send text prompts (each a different window), not ids")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(0)
    texts = []
    if a.text_file:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        ids = tok(open(a.text_file).read())["input_ids"]
        L = a.input_len
        for i in range(a.n + 1):
            texts.append(tok.decode(ids[i * L:(i + 1) * L]))
    c = httpx.Client(timeout=600)
    recs = []
    for i in range(a.n + 1):  # the first is a warm-up
        prompt = texts[i] if texts else [rng.randrange(a.vocab) for _ in range(a.input_len)]
        body = {"prompt": prompt, "max_tokens": a.max_tokens, "temperature": 0.0, "ignore_eos": True,
                "stream": True, "logprobs": 0}
        m0 = c.get(a.router + "/metrics").text
        t_enc0 = time.monotonic()
        raw = json.dumps(body)
        t_enc = time.monotonic() - t_enc0
        t_send = time.monotonic()
        first = hdr = None
        n = 0
        with c.stream("POST", a.router + "/v1/completions", content=raw,
                      headers={"content-type": "application/json"}) as r:
            hdr = time.monotonic()
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                k = len((ev["choices"][0].get("logprobs") or {}).get("tokens") or [])
                if k and first is None:
                    first = time.monotonic()
                n += k
        t_end = time.monotonic()
        m1 = c.get(a.router + "/metrics").text
        d = {k.split("pd_router_")[1].replace("_sum", ""): metric(m1, k) - metric(m0, k) for k in NAMES}
        ps = c.get(a.prefill + "/debug/steps", params={"since": t_send}).json()["steps"]
        ds = c.get(a.decode + "/debug/steps", params={"since": t_send}).json()["steps"]
        pre = [s for s in ps if s[3] > 0]  # the request's prefill chunks (a trailing empty step is not one)
        rec = {"i": i, "tokens": n, "json_encode_s": t_enc, "client_ttft": first - t_send, "client_headers": hdr - t_send,
               "router": d,
               "prefill_steps": [[round(s[0] - s[1] - t_send, 4), round(s[1], 4), s[3]] for s in pre],
               "prefill_steps_all": [[round(s[0] - s[1] - t_send, 4), round(s[1], 4), s[3]] for s in ps],
               "decode_steps": [[round(s[0] - s[1] - t_send, 4), round(s[1], 4), s[2], s[4]] for s in ds
                                if s[0] - s[1] - t_send < first - t_send + 0.5],
               "first_t": first - t_send, "end_t": t_end - t_send}
        recs.append(rec)
        print(json.dumps(rec), flush=True)
    keep = recs[1:]

    def mean(f):
        return statistics.mean(f(r) for r in keep)

    def med(f):
        return statistics.median(f(r) for r in keep)

    psum = lambda r: sum(s[1] for s in r["prefill_steps"])  # noqa: E731
    pfirst = lambda r: min(s[0] for s in r["prefill_steps"])  # noqa: E731
    plast = lambda r: max(s[0] + s[1] for s in r["prefill_steps"])  # noqa: E731
    out = {
        "n": len(keep),
        "client_ttft": med(lambda r: r["client_ttft"]),
        "client_minus_router_e2e": med(lambda r: r["client_ttft"] - r["router"]["e2e_ttft_seconds"]),
        "router_t0_to_prefill_call": med(lambda r: r["router"]["prefill_seconds"] - r["router"]["prefill_call_seconds"]),
        "prefill_call": med(lambda r: r["router"]["prefill_call_seconds"]),
        "prefill_steps_sum": med(psum),
        "send_to_first_prefill_step": med(pfirst),
        "last_prefill_step_end_to_first_token": med(lambda r: r["first_t"] - plast(r)),
        "router_prefill_end_to_first_chunk": med(lambda r: r["router"]["e2e_ttft_seconds"] - r["router"]["prefill_seconds"]),
        "prefill_call_minus_steps": med(lambda r: r["router"]["prefill_call_seconds"] - psum(r)),
        "json_encode": med(lambda r: r["json_encode_s"]),
        "means": {"client_ttft": mean(lambda r: r["client_ttft"]),
                  "router_e2e": mean(lambda r: r["router"]["e2e_ttft_seconds"])},
    }
    print(json.dumps(out, indent=1), flush=True)
    with open(a.out, "w") as f:
        json.dump({"summary": out, "requests": recs}, f)


if __name__ == "__main__":
    main()
