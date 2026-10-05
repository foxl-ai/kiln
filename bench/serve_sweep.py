"""Closed-loop serving sweep: fixed input / output lengths at fixed concurrency levels,
reproducing the SageMaker GLM-5.3-Flash benchmark this engine is compared against
(ml.p5en.48xlarge, vLLM TP=4 / DP=2 / EP=8, 8192 tokens in / 256 out, streaming, concurrency
16 / 32 / 64 / 128; its table is REFERENCE below).

    python bench/serve_sweep.py --model zai-org/GLM-5.3-Flash --tp 32 --piecewise \
        --concurrency 16 32 64 128 --input-len 8192 --output-len 256

The engine runs in this process (no HTTP), every level keeps exactly `concurrency` requests in
flight (a finished request is replaced at once) until --requests have finished, after one
warm-up request of the same shape. Prompts are random token ids (seeded), outputs ignore EOS,
so every request costs exactly input + output tokens. Reported per level: TTFT p50 / p90 (add to
first token), per-request inter-token time (first token to last, over output - 1) p50, output
tokens per second over the level's wall time, requests per second, and dollars per million
output tokens at each --price.

--shared-prefix-len N --num-prefixes K (prompt caching): every prompt is one of K random prefixes of N
tokens, chosen at random per request, followed by input_len - N unique tokens, so real traffic's
shared system prompts / few-shot prefixes reach the radix cache (and, for a linear-attention model,
its state checkpoints). Several N run as several levels of one engine (each concurrency x each N).
The cache is flushed before each level when any N is set (the warm-up request's prompt is unique), so
every level starts cold, unless --keep-cache: then N repeated measures the cache warm (the same K
prefixes, new suffixes), which is what a server that has run for a while sees. Reported per level beside the usual columns: the cache hit
rate (prompt tokens served from the cache / prompt tokens of the finished requests), the cost per
request at each --price, and the effective price per 1M tokens against each --provider: the
provider's bill for the same requests (uncached input, cached input and output at its list prices,
the same tokens cached) and Kiln's cost as a fraction of it, which is also Kiln's price at the
provider's ratios.

--dp N runs N independent engines (replica r on cores core_base + r * tp ...), each keeping
concurrency / N requests in flight; every level starts on all replicas at once (a barrier) and is
reported over the union of their requests, with the level's wall time the slowest replica's, as
bench/offline.py --dp does. The table's concurrency is the total in flight.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import torch

# The comparison target, as published with the SageMaker notebook (ml.p5en.48xlarge,
# $72.795/h in us-east-2, SageMaker hosting price from the Pricing API, instanceName filter).
REFERENCE = {16: (1428, 12316, 20.1, 311, 1.2), 32: (979, 4672, 25.0, 842, 3.3),
             64: (634, 2724, 32.3, 1458, 5.7), 128: (415, 1561, 43.4, 2359, 9.3)}
REFERENCE_PRICE = 72.795


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dp", type=int, default=1,
                    help="independent engines of --tp ranks on consecutive cores, each serving concurrency / dp")
    ap.add_argument("--core-base", type=int, default=0, help="first NeuronCore (tp_core_base)")
    ap.add_argument("--attention-tp", type=int, default=None)
    ap.add_argument("--dp-attention", type=int, default=1, help="DP-attention groups (EngineConfig.dp_attention)")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--input-len", type=int, default=8192)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--requests", type=int, default=0, help="per level; default 2 x concurrency")
    ap.add_argument("--max-seconds", type=float, default=1800.0, help="per level")
    ap.add_argument("--max-model-len", type=int, default=0, help="default input + output")
    ap.add_argument("--prefill-tokens", type=int, default=512)
    ap.add_argument("--kv-cache-gb", type=float, default=4.0)
    ap.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8"],
                    help="the GPU reference runs auto (bf16): vLLM's FP8 MLA cache does not take NoPE MLA")
    ap.add_argument("--piecewise", action="store_true")
    ap.add_argument("--overlap", action="store_true")
    # One GLM-5.3-Flash layer-group graph compiles in 10-16 min at tp=32 (decode (4, 4) 761 s,
    # prefill (32, 4) 983 s on trn1.32xlarge), so a sweep pins the buckets its workload needs
    # instead of the default power-of-two ladders, and compiles them before the clock starts.
    ap.add_argument("--decode-buckets", default=None, help="comma list (per DP-attention group)")
    ap.add_argument("--page-buckets", default=None, help="comma list of pages per sequence")
    ap.add_argument("--prefill-buckets", default=None, help="comma list of chunk sizes (per group)")
    ap.add_argument("--warmup", action="store_true", help="compile every bucket before timing")
    # Every loaded NEFF reserves its own DMA-ring spill memory (docs/neuron-notes.md "HBM per
    # NeuronCore"), so on trn1 a big model runs ONE decode bucket per engine: one sweep run per
    # concurrency level. max_num_seqs is a graph input shape (state-pool rows, KV pool), so it is
    # pinned across those runs to keep hitting the compile cache.
    ap.add_argument("--max-num-seqs", type=int, default=0, help="default max(--concurrency) / dp, per engine")
    # Speculative decoding (EngineConfig.spec_method / spec_k). An MTP engine's verify graphs keep 1 + k
    # recurrent-state rows per request (engine/state_pool.py), so --state-checkpoints below can hold the pool's
    # row count, a graph input shape, at the baseline's.
    ap.add_argument("--spec-method", default=None, choices=["mtp", "ngram", "suffix"])
    ap.add_argument("--spec-k", type=int, default=1, help="drafted tokens per verify")
    # A linear-attention model's state pool holds 1 + max_num_seqs + state_checkpoints rows per DP-attention group
    # (engine/state_pool.py; EngineConfig.state_checkpoints, default 2 x max_num_seqs). GLM-5.3-Flash at tp=32 DP 4:
    # 18.4 MB per row per rank by the state shapes (34 KDA layers, 8 heads of [128, 128] fp32 plus [3, 3072] bf16
    # conv rows), so the 32 default
    # checkpoint rows at concurrency 64 are 0.59 GB of each core's 16 GiB. Without --shared-prefix-len this sweep's
    # random prompts share no prefix and its requests end before a decode checkpoint (state_track_interval 256): no
    # row is ever used; with K shared prefixes each DP group holds about K junction checkpoints (the kv: line).
    # The pool is a graph input: changing this changes every graph holding a linear-attention layer.
    ap.add_argument("--state-checkpoints", type=int, default=None,
                    help="state-checkpoint rows per DP-attention group (default: EngineConfig's)")
    # A decode-only box (tools/time_decode.py's cost curve, a prefill/decode split) keeps no prefix cache: the
    # radix cache's state checkpoints are 2 x max_num_seqs rows of recurrent state per DP group
    # (model_runner.state_checkpoint_rows), 18.4 MB per row per rank for GLM-5.3-Flash at attention TP 8. --state-checkpoints 0
    # sizes the same pool; this also turns the radix cache off.
    ap.add_argument("--no-prefix-caching", action="store_true", help="EngineConfig.prefix_caching=False")
    ap.add_argument("--price", action="append", default=[],
                    help="name=dollars_per_hour, repeatable (default trn1.32xlarge on-demand and spot)")
    ap.add_argument("--shared-prefix-len", type=int, nargs="+", default=[0],
                    help="tokens of every prompt taken from one of --num-prefixes shared prefixes (0: all unique); "
                         "several values run one level each per concurrency (one engine)")
    ap.add_argument("--num-prefixes", type=int, default=1)
    ap.add_argument("--input-len-min", type=int, default=0,
                    help="prompt lengths uniform in [this, --input-len] (0: every prompt --input-len tokens)")
    ap.add_argument("--output-len-min", type=int, default=0,
                    help="output lengths uniform in [this, --output-len] (0: every request --output-len): requests then "
                         "finish out of step, as with EOS or speculative decoding")
    ap.add_argument("--dp-prefill-pack", nargs="+", default=None, choices=["off", "trim", "hold"],
                    help="EngineConfig.dp_prefill_pack per level (no graph changes): several values run every level once "
                         "per value on one engine, each from a flushed cache with the same requests")
    ap.add_argument("--dp-prefill-pack-min", type=int, default=None)
    ap.add_argument("--dp-prefill-hold-steps", type=int, default=None)
    ap.add_argument("--keep-cache", action="store_true",
                    help="do not flush the prefix cache between levels: a level repeated (--shared-prefix-len N N) "
                         "then measures a warm cache, a server's steady state")
    # Scheduling knobs that change no graph (the keys of a farm capture stay those of the plain sweep).
    ap.add_argument("--schedule-policy", default=None, help="EngineConfig.schedule_policy (fcfs, lpm, spf, priority)")
    ap.add_argument("--state-checkpoint-interval", type=int, default=None,
                    help="EngineConfig.state_checkpoint_interval (tokens between periodic prefill checkpoints)")
    ap.add_argument("--hicache-host-gb", type=float, default=0.0,
                    help="EngineConfig.hicache_host_gb: host-memory KV tier per rank (engine/hicache.py; no graph changes)")
    ap.add_argument("--no-ckpt-lookahead", action="store_true",
                    help="EngineConfig.state_checkpoint_lookahead=False (junction checkpoints only one request late)")
    ap.add_argument("--provider", action="append", default=[],
                    help="name=input,cached_input,output dollars per 1M tokens, repeatable (default: the list price "
                         "0.15,0.03,0.50 and DeepInfra's discounted 0.075,0.015,0.25, whose cached price is assumed "
                         "at the list's 1/5 of input)")
    return ap


def engine_config(args, core_base: int = 0):
    """The EngineConfig a sweep runs with; tools/compile_farm.py capture builds the same one, so
    the graphs it captures are the graphs this sweep compiles."""
    from kiln.config import EngineConfig

    def ladder(v):
        return tuple(int(x) for x in v.split(",")) if v else None

    top = max(args.concurrency) // getattr(args, "dp", 1)
    return EngineConfig(
        model_path=args.model, device=args.device, dtype=torch.bfloat16, tp=args.tp, tp_core_base=core_base,
        max_num_seqs=args.max_num_seqs or top, max_model_len=args.max_model_len or args.input_len + args.output_len,
        max_prefill_tokens=args.prefill_tokens, kv_cache_gb=args.kv_cache_gb, kv_cache_dtype=args.kv_cache_dtype,
        piecewise=args.piecewise, overlap=args.overlap, dp_attention=args.dp_attention,
        decode_batch_buckets=ladder(args.decode_buckets), page_buckets=ladder(args.page_buckets),
        prefill_token_buckets=ladder(args.prefill_buckets),
        **({"prefix_caching": False} if getattr(args, "no_prefix_caching", False) else {}),
        **({"attention_tp": args.attention_tp} if args.attention_tp else {}),
        **({"spec_method": args.spec_method, "spec_k": args.spec_k} if getattr(args, "spec_method", None) else {}),
        **({"state_checkpoints": args.state_checkpoints} if getattr(args, "state_checkpoints", None) is not None
           else {}),
        **({"schedule_policy": args.schedule_policy} if getattr(args, "schedule_policy", None) else {}),
        **({"state_checkpoint_interval": args.state_checkpoint_interval}
           if getattr(args, "state_checkpoint_interval", None) is not None else {}),
        **({"state_checkpoint_lookahead": False} if getattr(args, "no_ckpt_lookahead", False) else {}),
        **({"hicache_host_gb": args.hicache_host_gb} if getattr(args, "hicache_host_gb", 0) else {}),
    )


def make_engine(args, core_base: int):
    from kiln.engine.engine import LLMEngine

    t = time.perf_counter()
    eng = LLMEngine(engine_config(args, core_base))
    print(f"engine up {time.perf_counter() - t:.1f}s (cores {core_base}..{core_base + args.tp - 1})", flush=True)
    return eng


def warm(eng, args, prompt, params, reverse: bool = False) -> None:
    if args.warmup:
        w = eng.warmup(reverse=reverse)
        print(f"bucket warmup {w['seconds']:.1f}s over {w['graphs']} graphs", flush=True)
    t = time.perf_counter()
    eng.generate([getattr(prompt, "unique", prompt)()], params() if callable(params) else params)
    print(f"warm-up request {time.perf_counter() - t:.1f}s (includes compiles)", flush=True)


def run_level(eng, args, conc: int, n_req: int, prompt, params):
    """Keep conc requests in flight until n_req finished; per finished request (TTFT ms, ITL ms,
    output tokens, prompt tokens, prompt tokens served from the cache), and the wall time."""
    from kiln.engine.request import Status

    if any(args.shared_prefix_len) and not getattr(args, "keep_cache", False):
        eng.flush_cache()  # every level starts cold
    live, done = [], []
    spec0 = (eng.spec_proposed, eng.spec_accepted, getattr(eng, "spec_verifies", 0), getattr(eng, "mtp_seconds", 0.0))
    t0 = time.perf_counter()
    started = 0
    running = []  # sequences admitted (running) after each step: a KV-bound run admits fewer than conc
    steps = []  # per step() call: (wall seconds, decode graph calls, prefill graph calls) it launched
    while len(done) < n_req and time.perf_counter() - t0 < args.max_seconds:
        while len(live) < conc and started < n_req:
            live.append(eng.add_request(prompt(), params() if callable(params) else params))
            started += 1
        eng.step()
        st = eng.last_step
        steps.append((st.seconds, st.decode_calls, st.prefill_calls, st.prefill_chunks, st.num_prefill_tokens,
                      st.deferred_chunks))
        still = []
        for r in live:
            (done if r.finish_time is not None else still).append(r)
        live = still
        running.append(sum(r.status == Status.RUNNING for r in live))
    wall = time.perf_counter() - t0
    kv_report(eng, args, conc, running, done)
    if eng.cfg.spec_method:
        spec_report(eng, conc, spec0, wall)
    for r in live:  # unfinished at the time limit: not counted
        eng.abort(r)
    while eng.has_work():
        eng.step()
    recs = [((r.first_token_time - r.arrival_time) * 1e3 if r.first_token_time else None,
             (r.finish_time - r.first_token_time) * 1e3 / max(len(r.output_ids) - 1, 1), len(r.output_ids),
             r.num_prompt, max(r.num_cached_tokens, 0))
            for r in done]
    split = device_split(steps, eng.cfg.overlap)
    pack = pack_report(eng, steps)
    if split is not None and pack is not None:
        split.update(pack)
    if split:
        print(f"device time split: prefill call {split['prefill_call_s']:.3f} s, decode call "
              f"{split['decode_call_s']:.3f} s, per step {split['step_s']:.3f} s; prefill {split['prefill_share']:.1%} "
              f"of the attributed time over {split['steps']} steps", flush=True)
    return recs, wall, split


def set_pack(eng, args, pack: str) -> None:
    """Switch a DP-attention engine's prefill packing between levels (engine/dp.py; the scheduler only)."""
    sch = eng.scheduler
    if not hasattr(sch, "pack"):
        return
    sch.pack = pack
    if args.dp_prefill_pack_min is not None:
        sch.pack_min = args.dp_prefill_pack_min
    if args.dp_prefill_hold_steps is not None:
        sch.hold_steps = args.dp_prefill_hold_steps


def pack_report(eng, steps: list) -> dict | None:
    """How full the level's prefill calls were: a DP-attention prefill call carries at most one chunk per
    group and costs about the same however many it carries, so its packed minimum is ceil(chunks / groups);
    chunk use is chunks / (calls x groups), token use the prefill tokens / (calls x groups x bucket)."""
    N = getattr(eng, "dp", 1)
    calls = sum(s[2] for s in steps)
    if not calls:
        return None
    chunks, toks, deferred = sum(s[3] for s in steps), sum(s[4] for s in steps), sum(s[5] for s in steps)
    bucket = max(eng.runner.prefill_buckets)
    out = {"pack_prefill_calls": calls, "pack_chunks": chunks, "pack_min_calls": -(-chunks // N),
           "pack_chunk_use": round(chunks / (calls * N), 4), "pack_token_use": round(toks / (calls * N * bucket), 4),
           "pack_deferred": deferred}
    print(f"prefill packing: {calls} prefill calls for {chunks} chunks over {N} groups (packed minimum "
          f"{out['pack_min_calls']}, {calls / out['pack_min_calls']:.3f}x); chunk slots used {out['pack_chunk_use']:.1%}, "
          f"token slots {out['pack_token_use']:.1%}; {deferred} chunks deferred", flush=True)
    return out


def device_split(steps: list, overlap: bool) -> dict | None:
    """Least-squares split of the level's step wall times into a cost per prefill graph call, per decode
    graph call and per step: t = a * decode_calls + b * prefill_calls + c. Under overlap a step() call
    waits for the step launched by the call before it, so its time is paired with that launch."""
    import numpy as np

    rows = [(steps[i][0], *steps[i - 1][1:]) for i in range(1, len(steps))] if overlap else list(steps)
    rows = [r for r in rows if r[1] or r[2]]
    if len(rows) < 8:
        return None
    t = np.array([r[0] for r in rows])
    X = np.array([[r[1], r[2], 1.0] for r in rows])
    (a, b, c), *_ = np.linalg.lstsq(X, t, rcond=None)
    d_sum, p_sum = X[:, 0].sum(), X[:, 1].sum()
    tot = a * d_sum + b * p_sum
    return {"decode_call_s": float(a), "prefill_call_s": float(b), "step_s": float(c), "steps": len(rows),
            "prefill_share": float(b * p_sum / tot) if tot > 0 else 0.0,
            "decode_calls": int(d_sum), "prefill_calls": int(p_sum)}


def kv_report(eng, args, conc: int, running: list, done: list) -> None:
    """One line saying whether the KV pool bounded the run: pages per DP-attention group against the
    pages its share of conc full-length sequences needs, the most and the mean sequences admitted at
    once, and the preemptions of the finished requests."""
    ps = eng.cfg.page_size
    per_seq = -(-(args.input_len + args.output_len) // ps)
    groups = len(getattr(eng, "pools", [eng.pool]))
    pages = eng.pool.num_usable
    need = -(-conc // groups) * per_seq
    mean = sum(running) / len(running) if running else 0.0
    ck = ""
    st = getattr(eng.runner, "state", None)
    if st is not None and st.num_ckpt_rows:
        ck = (f"; state checkpoints per group {[r.num_ckpts for r in getattr(eng, 'radixes', [eng.radix])]} "
              f"of {st.num_ckpt_rows} rows")
    print(f"kv: {pages} pages per group x {groups} groups, {conc} x {per_seq} pages need {need} per group "
          f"({'KV-bound' if pages < need else 'fits'}); running max {max(running, default=0)} mean {mean:.1f} "
          f"of {conc}; preemptions {sum(r.num_preemptions for r in done)}{ck}", flush=True)


def spec_report(eng, conc: int, before: tuple, wall: float) -> None:
    """One line on speculative decoding over the level: drafts proposed and accepted, the mean tokens a
    verify gives one sequence (its accepted drafts plus the replacement or bonus token), the acceptance of
    each draft position given the earlier ones were accepted (cumulative over the engine's life), and the
    share of the wall spent drafting (MTP graphs and their read-back, engine.mtp_seconds)."""
    now = (eng.spec_proposed, eng.spec_accepted, getattr(eng, "spec_verifies", 0), getattr(eng, "mtp_seconds", 0.0))
    p, a, v, ms = (x - y for x, y in zip(now, before))
    pos = " ".join(f"{x / max(r, 1):.3f}" for x, r in zip(getattr(eng, "spec_pos_accepted", []),
                                                          getattr(eng, "spec_pos_reached", [])))
    print(f"spec: {eng.cfg.spec_method} k={eng.cfg.spec_k} conc {conc}: {v} verifies, {a} of {p} drafts accepted "
          f"({a / max(p, 1):.1%}), {1 + a / max(v, 1):.3f} tokens per verify; by position {pos}; drafting "
          f"{ms:.1f} s of {wall:.1f} s wall", flush=True)


DEFAULT_PROVIDERS = {"list": (0.15, 0.03, 0.50), "deepinfra": (0.075, 0.015, 0.25)}


def provider_prices(specs: list[str]) -> dict:
    """--provider name=input,cached,output ($ per 1M tokens) -> {name: (input, cached, output)}."""
    out = {}
    for spec in specs:
        name, vals = spec.split("=")
        p = tuple(float(x) for x in vals.split(","))
        if len(p) != 3:
            raise SystemExit(f"--provider {spec}: want name=input,cached_input,output")
        out[name] = p
    return out or dict(DEFAULT_PROVIDERS)


def against_providers(usd_per_req: float, uncached_in: float, cached_in: float, out: float,
                      providers: dict) -> dict:
    """Per provider: its bill for one mean request (dollars) at its list prices for the same uncached
    input, cached input and output tokens, Kiln's cost as a fraction of it, and Kiln's price per 1M
    tokens at the provider's own ratios (the fraction times each list price)."""
    res = {}
    for name, (p_in, p_cached, p_out) in providers.items():
        bill = (uncached_in * p_in + cached_in * p_cached + out * p_out) / 1e6
        f = usd_per_req / bill if bill else float("nan")
        res[name] = {"provider_usd_per_req": round(bill, 7), "kiln_fraction": round(f, 3),
                     "kiln_usd_per_m": [round(f * p, 4) for p in (p_in, p_cached, p_out)]}
    return res


def summarize(conc: int, recs, wall: float, prices: dict, providers: dict | None = None,
              split: dict | None = None) -> dict:
    out_tokens = sum(r[2] for r in recs)
    ttft = [r[0] for r in recs if r[0] is not None]
    itl = [r[1] for r in recs]
    row = {"concurrency": conc, "requests": len(recs), "wall_s": round(wall, 1),
           "ttft_p50_ms": round(pct(ttft, 0.5)), "ttft_p90_ms": round(pct(ttft, 0.9)),
           "itl_p50_ms": round(pct(itl, 0.5), 1), "out_tok_s": round(out_tokens / wall, 1),
           "req_s": round(len(recs) / wall, 2)}
    for name, rate in prices.items():
        row[f"usd_per_m_out[{name}]"] = round(float(rate) / (row["out_tok_s"] * 3600 / 1e6), 2) \
            if row["out_tok_s"] else None
    if recs and len(recs[0]) > 3:
        prompt_toks, cached = sum(r[3] for r in recs), sum(r[4] for r in recs)
        row["cache_hit_rate"] = round(cached / prompt_toks, 4) if prompt_toks else 0.0
        for kind, hit in (("hit", True), ("miss", False)):
            ts = [r[0] for r in recs if r[0] is not None and (r[4] > 0) == hit]
            row[f"ttft_p50_{kind}_ms"] = round(pct(ts, 0.5)) if ts else None
        row["hit_requests"] = sum(r[4] > 0 for r in recs)
        n = len(recs)
        req_s = len(recs) / wall if wall else 0.0
        for name, rate in prices.items():
            usd = float(rate) / 3600 / req_s if req_s else float("nan")
            row[f"usd_per_req[{name}]"] = round(usd, 7)
            row[f"vs_providers[{name}]"] = against_providers(
                usd, (prompt_toks - cached) / n, cached / n, out_tokens / n, providers or DEFAULT_PROVIDERS)
            if split and prompt_toks > cached and out_tokens:
                # At cost: the level's dollars split by the device time attributed to prefill and decode
                # calls (device_split); cached prompt tokens cost no prefill call (their restores are
                # inside the attributed calls).
                total = float(rate) * wall / 3600
                row[f"at_cost_usd_per_m[{name}]"] = {
                    "uncached_input": round(total * split["prefill_share"] / (prompt_toks - cached) * 1e6, 4),
                    "output": round(total * (1 - split["prefill_share"]) / out_tokens * 1e6, 4)}
    if split:
        row["device_split"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in split.items()}
    ref = REFERENCE.get(conc)
    ref_s = (f"  | p5en vLLM: TTFT {ref[0]}/{ref[1]} ms, ITL {ref[2]} ms, {ref[3]} out tok/s, "
             f"${REFERENCE_PRICE / (ref[3] * 3600 / 1e6):.2f}/M out") if ref else ""
    print(f"conc {conc:>4}: TTFT p50 {row['ttft_p50_ms']} ms p90 {row['ttft_p90_ms']} ms, ITL p50 "
          f"{row['itl_p50_ms']} ms, {row['out_tok_s']} out tok/s, {row['req_s']} req/s, "
          + ", ".join(f"${v}/M out [{k[len('usd_per_m_out['):-1]}]" for k, v in row.items()
                      if k.startswith("usd_per_m_out")) + ref_s, flush=True)
    if "cache_hit_rate" in row:
        print(f"cache: hit rate {row['cache_hit_rate']:.3f} of prompt tokens, {row['hit_requests']} of "
              f"{row['requests']} requests hit; TTFT p50 hits {row['ttft_p50_hit_ms']} ms, misses "
              f"{row['ttft_p50_miss_ms']} ms", flush=True)
        for k, v in row.items():
            if k.startswith("usd_per_req["):
                name = k[len("usd_per_req["):-1]
                vs = "; ".join(f"{p}: provider ${d['provider_usd_per_req']:.6f}/req, Kiln {d['kiln_fraction']:.3f}x = "
                               f"${d['kiln_usd_per_m'][0]} in / ${d['kiln_usd_per_m'][1]} cached / "
                               f"${d['kiln_usd_per_m'][2]} out per 1M"
                               for p, d in row[f"vs_providers[{name}]"].items())
                print(f"cost [{name}]: ${v:.6f}/req; {vs}", flush=True)
                ac = row.get(f"at_cost_usd_per_m[{name}]")
                if ac:
                    print(f"at cost [{name}]: ${ac['uncached_input']} per 1M uncached input (prefill calls), "
                          f"${ac['output']} per 1M output (decode calls), cached input ~0", flush=True)
    return row


def print_profile() -> None:
    """KILN_PROFILE_PIECES=1 / KILN_PROFILE_EXEC=1 (model_runner): p50 per layer-group graph by its
    hidden shape (decode rows vs prefill rows), and per graph call."""
    from kiln.engine.model_runner import EXEC_TIMES, PIECE_TIMES

    for (shape, gi), ts in sorted(PIECE_TIMES.items()):
        ts = sorted(ts)
        print(f"  piece {gi:>2} h{list(shape)}: n={len(ts)} p50 {ts[len(ts) // 2] * 1e3:.2f} ms", flush=True)
    for key, ts in sorted(EXEC_TIMES.items(), key=lambda kv: str(kv[0])):
        ts = sorted(ts)
        print(f"  exec {key}: n={len(ts)} p50 {ts[len(ts) // 2] * 1e3:.2f} ms", flush=True)
    from kiln.engine.engine import STEP_TIMES

    for (kinds, phase), ts in sorted((STEP_TIMES or {}).items()):  # KILN_PROFILE_STEP=1
        ts = sorted(ts)
        print(f"  step [{kinds}] {phase}: n={len(ts)} p50 {ts[len(ts) // 2] * 1e3:.2f} ms mean "
              f"{sum(ts) / len(ts) * 1e3:.2f} ms total {sum(ts):.1f} s", flush=True)


def sampling(args, seed: int = 0):
    """The requests' SamplingParams, or with --output-len-min a function drawing each request's (seeded)."""
    from kiln.engine.request import SamplingParams

    if not getattr(args, "output_len_min", 0):
        return SamplingParams(max_new_tokens=args.output_len, temperature=0.0, ignore_eos=True)
    rng = random.Random(seed * 104729 + 3)
    return lambda: SamplingParams(max_new_tokens=rng.randint(args.output_len_min, args.output_len), temperature=0.0,
                                  ignore_eos=True)


def prompter(args, vocab: int, seed: int, n: int = 0):
    """Random-token prompts of input_len. With a shared prefix length n, each is one of --num-prefixes
    seeded random prefixes of n tokens (chosen at random) plus input_len - n unique tokens. The
    returned function's .unique() is always a fully unique prompt (the warm-up request's)."""
    rng = random.Random(seed if not n else seed * 1_000_003 + n)  # the same prefixes for every level of n
    lo = 1000 if vocab > 2000 else 0  # (tiny test vocabularies)
    unique = lambda n=args.input_len: [rng.randrange(lo, vocab) for _ in range(n)]  # noqa: E731
    lmin = getattr(args, "input_len_min", 0) or 0
    lrng = random.Random(seed * 7919 + 1 + n)  # lengths from their own stream: fixed-length prompts unchanged

    def length() -> int:
        return lrng.randint(lmin, args.input_len) if lmin else args.input_len

    if not n:
        f = lambda: unique(length())  # noqa: E731
        f.unique = unique
        return f
    if not 0 < n < max(args.input_len, 1) or (lmin and lmin <= n):
        raise SystemExit(f"--shared-prefix-len {n} must be in (0, --input-len {args.input_len}) and below "
                         f"--input-len-min")
    prefixes = [unique(n) for _ in range(max(1, args.num_prefixes))]

    def shared():
        return list(rng.choice(prefixes)) + unique(length() - n)

    shared.unique = unique
    return shared


def replica(args, r: int, barrier, results, loaded=None) -> None:
    # Replicas load one after another: two tp=32 GLM-5.3-Flash replicas loading at once took
    # 904 / 958 s against 75 s for one alone from a warm page cache (trn2.48xlarge, 2026-10-03).
    if loaded is not None and r > 0:
        loaded[r - 1].wait()
    eng = make_engine(args, args.core_base + r * args.tp)
    if loaded is not None:
        loaded[r].set()
    vocab = min(eng.mcfg.vocab_size, 100_000)
    prompt, params = prompter(args, vocab, r), sampling(args)
    warm(eng, args, prompt, params, reverse=r % 2 == 1)  # two replicas compile two graphs at once
    mixes: dict = {}
    for conc in args.concurrency:
        for i, n in enumerate(args.shared_prefix_len):
            barrier.wait()
            if n and n not in mixes:
                mixes[n] = prompter(args, vocab, r, n)
            results[(r, conc, i)] = run_level(eng, args, conc // args.dp, (args.requests or 2 * conc) // args.dp,
                                              mixes[n] if n else prompt, params)[:2]
    eng.close()


def main() -> None:
    args = build_parser().parse_args()
    prices = dict(p.split("=") for p in args.price) or {"trn1.32xlarge on-demand": "21.50",
                                                        "trn1.32xlarge spot": "2.15"}
    providers = provider_prices(args.provider)
    rows = []
    if args.dp == 1:
        eng = make_engine(args, args.core_base)
        vocab = min(eng.mcfg.vocab_size, 100_000)
        prompt, params = prompter(args, vocab, 0), sampling(args)
        warm(eng, args, prompt, params)
        mixes: dict = {}  # one prompter per shared length: a repeated level keeps its prefixes, new suffixes
        packs = args.dp_prefill_pack or [None]
        for conc in args.concurrency:
            for pi, pack in enumerate(packs):
                if pack is not None:
                    set_pack(eng, args, pack)
                    if len(packs) > 1:  # every mode from the same state with the same requests
                        eng.flush_cache()
                        mixes, prompt, params = {}, prompter(args, vocab, 1000 + conc), sampling(args, conc)
                for n in args.shared_prefix_len:
                    if any(args.shared_prefix_len) or pack is not None:
                        print(f"level: concurrency {conc}, shared prefix {n} of {args.input_len} tokens, "
                              f"{args.num_prefixes} prefixes" + (f", dp prefill pack {pack}" if pack else ""), flush=True)
                    if n and n not in mixes:
                        mixes[n] = prompter(args, vocab, 0, n)
                    recs, wall, split = run_level(eng, args, conc, args.requests or 2 * conc,
                                                  mixes[n] if n else prompt, params)
                    rows.append({**summarize(conc, recs, wall, prices, providers, split), "shared_prefix_len": n,
                                 **({"dp_prefill_pack": pack} if pack else {})})
        print_profile()
    else:
        import multiprocessing as mp

        if any(c % args.dp for c in args.concurrency):
            raise SystemExit(f"every --concurrency must divide by --dp {args.dp}")
        ctx = mp.get_context("spawn")
        barrier = ctx.Barrier(args.dp)
        results = ctx.Manager().dict()
        loaded = [ctx.Event() for _ in range(args.dp)]
        procs = [ctx.Process(target=replica, args=(args, r, barrier, results, loaded)) for r in range(args.dp)]
        for p in procs:
            p.start()
        for conc in args.concurrency:
            for i, n in enumerate(args.shared_prefix_len):
                while not all((r, conc, i) in results for r in range(args.dp)):
                    if not all(p.is_alive() for p in procs) and not all((r, conc, i) in results for r in range(args.dp)):
                        raise SystemExit(f"a replica exited before level {conc} finished")
                    time.sleep(5)
                parts = [results[(r, conc, i)] for r in range(args.dp)]
                rows.append({**summarize(conc, [x for recs, _ in parts for x in recs], max(w for _, w in parts),
                                         prices, providers), "shared_prefix_len": n})
        for p in procs:
            p.join()
    print("RESULT", json.dumps({"model": args.model, "tp": args.tp, "dp": args.dp, "input_len": args.input_len,
                                "output_len": args.output_len, "shared_prefix_len": args.shared_prefix_len,
                                "num_prefixes": args.num_prefixes, "prices": prices, "providers": providers,
                                "levels": rows}))


if __name__ == "__main__":
    main()
