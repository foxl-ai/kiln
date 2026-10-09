"""Lone requests through an OpenAI-compatible server (the PD router, or one engine's server) with token-id prompts,
one at a time: per request the TTFT (send to first streamed token), every later token's arrival (ITL p50 / p90 / max)
and the greedy token ids, to compare a deployment with one engine on the same prompts.

    python tools/pd_client.py --url http://router:8000 --ids-npy prompts.npy --rows 1 2 --lengths 131072 1044480 \\
        --max-tokens 64 --warm-rows 0 --warm-lengths 8192 --gap-s 10 --out-json c.json
"""

from __future__ import annotations

import argparse
import json
import time


def one(url: str, ids: list[int], max_tokens: int, timeout: float) -> dict:
    import httpx

    body = {"prompt": ids, "max_tokens": max_tokens, "temperature": 0, "stream": True, "logprobs": 0,
            "return_token_ids": True, "ignore_eos": True}
    t0 = time.perf_counter()
    times, toks, lps = [], [], []
    with httpx.stream("POST", url.rstrip("/") + "/v1/completions", json=body, timeout=timeout) as r:
        if r.status_code != 200:
            raise RuntimeError(f"{url} answered {r.status_code}: {r.read()[:500]!r}")
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if "error" in ev:
                raise RuntimeError(f"stream error: {ev['error']}")
            c = ev["choices"][0]
            new = c.get("token_ids") or []
            now = time.perf_counter()
            times.extend([now] * len(new))
            toks.extend(new)
            if c.get("logprobs"):
                lps.extend(c["logprobs"]["token_logprobs"])
    gaps = sorted(b - a for a, b in zip(times, times[1:]))
    pct = lambda q: round(gaps[min(len(gaps) - 1, int(q * len(gaps)))] * 1e3, 2) if gaps else None  # noqa: E731
    return {"input_len": len(ids), "ttft_s": round(times[0] - t0, 3) if times else None,
            "itl_ms_p50": pct(0.5), "itl_ms_p90": pct(0.9), "itl_ms_max": round(gaps[-1] * 1e3, 2) if gaps else None,
            "output_tokens": len(toks), "tokens": toks, "logprobs": lps, "wall_s": round(time.perf_counter() - t0, 2)}


def main() -> None:
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--ids-npy", required=True)
    ap.add_argument("--rows", type=int, nargs="+", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", required=True)
    ap.add_argument("--warm-rows", type=int, nargs="*", default=[])
    ap.add_argument("--warm-lengths", type=int, nargs="*", default=[])
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--gap-s", type=float, default=10.0)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--wait-s", type=float, default=3600.0, help="wait this long for the server's /health first")
    a = ap.parse_args()
    import httpx

    end = time.monotonic() + a.wait_s
    while True:
        try:
            if httpx.get(a.url.rstrip("/") + "/health", timeout=5).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if time.monotonic() > end:
            raise SystemExit(f"{a.url} did not answer /health within {a.wait_s:g} s")
        time.sleep(5)
    arr = np.load(a.ids_npy, mmap_mode="r")
    for r, n in zip(a.warm_rows, a.warm_lengths):
        out = one(a.url, [int(x) for x in arr[r, :n]], 4, a.timeout)
        print(f"warm row {r} {n} tokens: TTFT {out['ttft_s']} s", flush=True)
    rows = []
    for r, n in zip(a.rows, a.lengths):
        time.sleep(a.gap_s)
        out = one(a.url, [int(x) for x in arr[r, :n]], a.max_tokens, a.timeout)
        out["row"] = r
        rows.append(out)
        print(f"row {r} {n} tokens: TTFT {out['ttft_s']} s, ITL p50 {out['itl_ms_p50']} p90 {out['itl_ms_p90']} ms "
              f"over {out['output_tokens']} tokens", flush=True)
    res = {"url": a.url, "rows": rows}
    print("RESULT", json.dumps({**res, "rows": [{k: v for k, v in x.items() if k not in ("tokens", "logprobs")}
                                                for x in rows]}), flush=True)
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(res, f)


if __name__ == "__main__":
    main()
