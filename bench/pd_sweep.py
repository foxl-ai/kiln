"""Closed-loop serving sweep over HTTP: the G1 workload (bench/serve_sweep.py: random-token prompts of
--input-len, --output-len tokens out with ignore_eos, greedy, streaming) against any OpenAI-compatible
Kiln front, here a disaggregated deployment's router (kiln.server.pd_router).

    python bench/pd_sweep.py --url http://router:8000 --concurrency 64 128 192 --requests 384 \\
        --input-len 8192 --output-len 256 --prefill-boxes 3 --decode-boxes 1 --box-price 2.15

Every level keeps exactly `concurrency` requests in flight (a finished one is replaced at once) until
--requests have finished, after --warmup-requests of the same shape. Per request the client records the
time to its first token (the first stream chunk carrying a token: logprobs=0 makes every token its own
entry) and its inter-token time (first to last token, over output - 1). Reported per level: TTFT and
per-request ITL p50 / p90, output tokens per second over the level's wall, requests per second, and the
deployment's cost: all boxes' $/h over the throughput as $ per 1M output tokens all-in, $ per request, and
the split $ per 1M input (the prefill boxes' share) and $ per 1M output (the decode boxes' share). With
--metrics, each server's /metrics is scraped after every level into the JSON record.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time

import httpx


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


async def one(client, url, prompt, out_len, timeout, level_t0=0.0):
    body = {"prompt": prompt, "max_tokens": out_len, "temperature": 0.0, "ignore_eos": True, "stream": True,
            "logprobs": 0}
    t0 = time.perf_counter()
    first = last = None
    n = 0
    async with client.stream("POST", url + "/v1/completions", json=body, timeout=timeout) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {(await resp.aread())[:300]!r}")
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if "error" in ev:
                raise RuntimeError(f"stream error: {ev['error']}")
            lp = ev["choices"][0].get("logprobs") or {}
            k = len(lp.get("tokens") or [])
            if k:
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
                n += k
    if n != out_len:
        raise RuntimeError(f"got {n} tokens, wanted {out_len}")
    return {"ttft_s": first - t0, "itl_s": (last - first) / max(n - 1, 1), "out": n, "e2e_s": last - t0,
            "first_t": first - level_t0, "last_t": last - level_t0}


def steady(recs, wall, lo=0.25, hi=0.75, input_len=0):
    """Rates over the middle of the level ([lo, hi] of its wall): a closed-loop level starts with every request in
    prefill at once and ends with the prefill side idle while the last ones decode, so its whole-level rate
    understates the steady state. Each request's tokens count as emitted evenly between its first and last token."""
    a, b = lo * wall, hi * wall
    out = 0.0
    for r in recs:
        x, y, n = r["first_t"], r["last_t"], r["out"]
        if y <= x:
            out += n if a <= x <= b else 0
            continue
        out += max(0.0, min(b, y) - max(a, x)) / (y - x) * (n - 1) + (1 if a <= x <= b else 0)
    firsts = sum(1 for r in recs if a <= r["first_t"] <= b)
    return {"window_s": round(b - a, 1), "out_tok_s": round(out / (b - a), 1),
            "req_s": round(firsts / (b - a), 3), "in_tok_s": round(firsts * input_len / (b - a), 1)}


async def open_level(url, rate, n_req, prompts, out_len, timeout, seed=0):
    """Open loop: n_req requests arriving as a Poisson process of `rate` per second (exponential gaps, seeded),
    each sent at its arrival time whatever is in flight. TTFT then includes queueing under that load, without the
    closed loop's start burst."""
    recs, errors = [], []
    rng = random.Random(seed)
    lim = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    t_level = time.perf_counter()
    async with httpx.AsyncClient(timeout=None, limits=lim) as client:
        async def one_at(at, p):
            await asyncio.sleep(max(0.0, at - (time.perf_counter() - t_level)))
            try:
                recs.append(await one(client, url, p, out_len, timeout, t_level))
            except Exception as e:  # noqa: BLE001  counted, reported, fails the level
                errors.append(repr(e))

        at, tasks = 0.0, []
        for p in prompts[:n_req]:
            tasks.append(asyncio.create_task(one_at(at, p)))
            at += rng.expovariate(rate)
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t_level
    return recs, errors, wall


async def level(url, conc, n_req, prompts, out_len, timeout):
    recs, errors = [], []
    it = iter(prompts)
    lim = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    t_level = time.perf_counter()
    async with httpx.AsyncClient(timeout=None, limits=lim) as client:
        async def worker():
            while True:
                try:
                    p = next(it)
                except StopIteration:
                    return
                try:
                    recs.append(await one(client, url, p, out_len, timeout, t_level))
                except Exception as e:  # noqa: BLE001  counted, reported, fails the level
                    errors.append(repr(e))

        t0 = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(min(conc, n_req))))
        wall = time.perf_counter() - t0
    return recs, errors, wall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[64])
    ap.add_argument("--rate", type=float, nargs="+", default=None,
                    help="open loop instead: Poisson arrivals at each of these rates (req/s), one level each")
    ap.add_argument("--requests", type=int, default=0, help="per level; default 2 x concurrency")
    ap.add_argument("--warmup-requests", type=int, default=4)
    ap.add_argument("--input-len", type=int, default=8192)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--prefill-boxes", type=int, default=1)
    ap.add_argument("--decode-boxes", type=int, default=1)
    ap.add_argument("--box-price", type=float, default=2.15, help="$/h of one box (trn1.32xlarge spot 2.15)")
    ap.add_argument("--prefill-box-price", type=float, default=None)
    ap.add_argument("--decode-box-price", type=float, default=None)
    ap.add_argument("--metrics", default="", help="comma list of server URLs whose /metrics to record per level")
    ap.add_argument("--out", default=None, help="append one JSON line per level")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    def prompt():
        return [rng.randrange(1000, a.vocab) for _ in range(a.input_len)]

    pp = a.prefill_box_price if a.prefill_box_price is not None else a.box_price
    dp = a.decode_box_price if a.decode_box_price is not None else a.box_price
    p_cost, d_cost = a.prefill_boxes * pp, a.decode_boxes * dp
    if a.warmup_requests:
        recs, errs, wall = asyncio.run(level(a.url, a.warmup_requests, a.warmup_requests,
                                             [prompt() for _ in range(a.warmup_requests)], a.output_len, a.timeout))
        print(f"warm-up: {len(recs)} requests in {wall:.1f} s, errors {errs[:3]}", flush=True)
        if errs:
            sys.exit(1)
    servers = [u.rstrip("/") for u in a.metrics.split(",") if u] if a.metrics else []

    def clocks():
        out = {}
        for u in servers:
            try:
                out[u] = httpx.get(u + "/debug/steps", params={"since": 1e18}, timeout=30).json()["now"]
            except (httpx.HTTPError, ValueError, KeyError):
                pass
        return out

    def step_summary(start: dict) -> dict:
        """Per server, its engine loop's steps during the level (GET /debug/steps): busy fraction, mean step
        seconds, rows, and the decode step time with and without a handoff copied in."""
        out = {}
        for u, t0 in start.items():
            try:
                d = httpx.get(u + "/debug/steps", params={"since": t0}, timeout=60).json()
            except (httpx.HTTPError, ValueError):
                continue
            st = d["steps"]
            if not st:
                continue
            span = d["now"] - t0
            busy = sum(x[1] for x in st)
            dec = [x for x in st if x[2] > 0 and x[3] == 0]
            top = max((x[2] for x in dec), default=0)
            full = [x[1] for x in dec if x[2] == top]
            out[u] = {"steps": len(st), "busy_frac": round(busy / span, 3) if span else None,
                      "mean_step_s": round(busy / len(st), 4), "prefill_tokens": sum(x[3] for x in st),
                      "decode_rows_mean": round(sum(x[2] for x in dec) / len(dec), 1) if dec else 0,
                      "decode_rows_max": top, "full_step_p50_s": round(pct(full, 0.5), 4) if full else None,
                      "full_steps": len(full),
                      "inject_step_p50_s": round(pct([x[1] for x in dec if x[4]], 0.5), 4) if any(x[4] for x in dec) else None,
                      "plain_step_p50_s": round(pct([x[1] for x in dec if not x[4]], 0.5), 4) if any(not x[4] for x in dec) else None}
        return out

    levels = [("rate", r) for r in a.rate] if a.rate else [("conc", c) for c in a.concurrency]
    for kind, conc in levels:
        n = a.requests or (int(conc * 120) if kind == "rate" else 2 * conc)
        start = clocks()
        if kind == "rate":
            recs, errs, wall = asyncio.run(open_level(a.url, conc, n, [prompt() for _ in range(n)], a.output_len,
                                                      a.timeout, a.seed))
        else:
            recs, errs, wall = asyncio.run(level(a.url, conc, n, [prompt() for _ in range(n)], a.output_len, a.timeout))
        out_tok = sum(r["out"] for r in recs)
        tps = out_tok / wall
        rps = len(recs) / wall
        hourly = p_cost + d_cost
        row = {("rate" if kind == "rate" else "concurrency"): conc, "requests": len(recs), "errors": len(errs),
               "wall_s": round(wall, 1),
               "ttft_p50_ms": round(pct([r["ttft_s"] for r in recs], 0.5) * 1e3),
               "ttft_p90_ms": round(pct([r["ttft_s"] for r in recs], 0.9) * 1e3),
               "ttft_p99_ms": round(pct([r["ttft_s"] for r in recs], 0.99) * 1e3),
               "itl_p50_ms": round(pct([r["itl_s"] for r in recs], 0.5) * 1e3, 1),
               "itl_p90_ms": round(pct([r["itl_s"] for r in recs], 0.9) * 1e3, 1),
               "out_tok_s": round(tps, 1), "in_tok_s": round(rps * a.input_len, 1), "req_s": round(rps, 3),
               "prefill_boxes": a.prefill_boxes, "decode_boxes": a.decode_boxes, "usd_per_h": hourly,
               "usd_per_m_out_all_in": round(hourly / (tps * 3600 / 1e6), 3) if tps else None,
               "usd_per_req": round(hourly / 3600 / rps, 7) if rps else None,
               "usd_per_m_in_prefill_boxes": round(p_cost / (rps * a.input_len * 3600 / 1e6), 4) if rps else None,
               "usd_per_m_out_decode_boxes": round(d_cost / (tps * 3600 / 1e6), 4) if tps else None,
               "input_len": a.input_len, "output_len": a.output_len, "error_samples": errs[:3]}
        st = steady(recs, wall, input_len=a.input_len)
        row["steady"] = dict(st, usd_per_m_out_all_in=round(hourly / (st["out_tok_s"] * 3600 / 1e6), 3) if st["out_tok_s"]
                             else None,
                             usd_per_m_in_prefill_boxes=round(p_cost / (st["in_tok_s"] * 3600 / 1e6), 4) if st["in_tok_s"]
                             else None,
                             usd_per_m_out_decode_boxes=round(d_cost / (st["out_tok_s"] * 3600 / 1e6), 4) if st["out_tok_s"]
                             else None)
        print(f"  steady (middle half, {st['window_s']} s): {st['out_tok_s']} out tok/s, {st['req_s']} req/s, "
              f"${row['steady']['usd_per_m_out_all_in']}/1M out all-in, split ${row['steady']['usd_per_m_in_prefill_boxes']}"
              f"/1M in + ${row['steady']['usd_per_m_out_decode_boxes']}/1M out", flush=True)
        print(f"{'rate' if kind == 'rate' else 'conc'} {conc}: {row['out_tok_s']} out tok/s ({row['in_tok_s']} in tok/s, {row['req_s']} req/s), TTFT "
              f"p50 {row['ttft_p50_ms']} p90 {row['ttft_p90_ms']} ms, ITL p50 {row['itl_p50_ms']} p90 "
              f"{row['itl_p90_ms']} ms; ${row['usd_per_m_out_all_in']}/1M out all-in at ${hourly}/h, "
              f"${row['usd_per_req']}/req, split ${row['usd_per_m_in_prefill_boxes']}/1M in + "
              f"${row['usd_per_m_out_decode_boxes']}/1M out; errors {len(errs)}", flush=True)
        if servers:
            row["steps"] = step_summary(start)
            for u, v in row["steps"].items():
                print(f"  steps {u}: {v}", flush=True)
        if a.metrics:
            row["metrics"] = {}
            for u in a.metrics.split(","):
                try:
                    row["metrics"][u] = httpx.get(u.rstrip("/") + "/metrics", timeout=30).text
                except httpx.HTTPError as e:
                    row["metrics"][u] = repr(e)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
        if errs:
            print("errors:", errs[:5], flush=True)


if __name__ == "__main__":
    main()
