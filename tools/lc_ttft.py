"""Time to first token and inter-token time of LONE long requests, several prompt lengths on one engine.

    python tools/lc_ttft.py --lengths 32768 131072 307200 1044480 --warm-lengths 8192 307200 -- <serve_sweep args>

The serve_sweep arguments build the engine exactly as bench/serve_sweep.py does (engine_config: so the graphs are the
ones a compile-farm capture of the same arguments holds; --input-len there is ignored, --output-len is each request's
output). With --warmup every bucket's graphs are loaded first; then one lone request of each --warm-lengths value
(random tokens, not reported: the first runs of the graphs the bucket warmup did not run, e.g. both page buckets'
prefill and decode at real positions), then one lone request of each --lengths value in order, each alone in the
engine (concurrency 1, unique random prompts, so no prefix-cache hit). Per length: TTFT (arrival to first token), ITL
p50 (first to last token over output - 1), the device split of serve_sweep.run_level (prefill call, decode call), and
the wall; a RESULT json line at the end. It replaces one serve_sweep process per length (each with its own full-length
warm-up request: the 1M warm request alone is ~20 min at CP 8, docs/neuron-notes.md "Long context (1M)").
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _barrier(addr: str, stage: int, stages: int):
    """A meeting point of the pipeline's stage processes (one TCP connection per earlier stage to the last stage):
    each call returns on every stage once all of them called it (the last stage answers each earlier one)."""
    import socket

    host, port = addr.rsplit(":", 1)
    if stage == stages - 1:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", int(port)))
        srv.listen(stages)
        peers = [srv.accept()[0] for _ in range(stages - 1)]

        def meet():
            for p in peers:
                if p.recv(1) != b"r":
                    raise RuntimeError("pipeline barrier: a stage went away")
            for p in peers:
                p.sendall(b"g")
        return meet
    deadline = time.time() + 1800
    while True:
        try:
            c = socket.create_connection((host, int(port)), timeout=30)
            break
        except OSError:
            if time.time() > deadline:
                raise
            time.sleep(1.0)
    c.settimeout(None)
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def meet():
        c.sendall(b"r")
        if c.recv(1) != b"g":
            raise RuntimeError("pipeline barrier: the last stage went away")
    return meet


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", required=True)
    ap.add_argument("--warm-lengths", type=int, nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--text-file", default=None,
                    help="real text instead of random token ids: every request (warm ones included) is the next disjoint "
                         "window of this file's tokens, so no two share a prefix (random ids route pathologically: one "
                         "expert took 4090 of 4096 rows, docs/neuron-notes.md, which inflates the expert-parallel wait)")
    ap.add_argument("--ids-npy", default=None,
                    help="token ids instead of random ones: an int array [requests, tokens] (.npy); every request (warm "
                         "ones included) takes the next row of --ids-rows, cut to its length")
    ap.add_argument("--ids-rows", type=int, nargs="*", default=None, help="rows of --ids-npy, in request order")
    ap.add_argument("--pp-barrier", default=None,
                    help="host:port: a pipeline (--pp-stages > 1, engine/pp.py) of one process per stage: before every "
                         "request the stages meet here (the last stage listens, the others connect), so each stage "
                         "adds the same request at the same moment and the last stage's TTFT is the pipeline's")
    ap.add_argument("--gap-s", type=float, default=0.0,
                    help="seconds between requests: with --pp-follow (stage 0 of a following pipeline, which meets no "
                         "barrier) the previous request's last chunks leave the later stages before the next one starts")
    ap.add_argument("--record-tokens", action="store_true",
                    help="record each request's first sampled token and its logprob (SamplingParams.logprobs=0: host only, "
                         "no graph change), to compare a pipeline's last stage with one engine on the same seeded prompts")
    own, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    import serve_sweep as ss

    args = ss.build_parser().parse_args(rest + ["--concurrency", "1", "--requests", "1"])
    args.skip_warm_request = True
    eng = ss.make_engine(args, args.core_base)
    vocab = min(eng.mcfg.vocab_size, 100_000)
    rng = random.Random(own.seed)
    lo = 1000 if vocab > 2000 else 0

    if own.ids_npy:
        import numpy as np

        arr = np.load(own.ids_npy)
        id_rows = list(own.ids_rows if own.ids_rows is not None else range(arr.shape[0]))  # not `rows`: main's results

        def prompt_of(n: int):
            def take():
                if not id_rows:
                    raise SystemExit("--ids-rows ran out: more requests than rows")
                r = id_rows.pop(0)
                if n > arr.shape[1]:
                    raise SystemExit(f"--ids-npy rows hold {arr.shape[1]} tokens, {n} needed")
                return [int(x) for x in arr[r, :n]]
            return take
    elif own.text_file:
        with open(own.text_file) as f:
            text_ids = eng.tokenizer(f.read())["input_ids"]
        cursor = [0]

        def prompt_of(n: int):
            def take():
                a = cursor[0]
                if a + n > len(text_ids):
                    raise SystemExit(f"--text-file has {len(text_ids)} tokens, {a + n} needed")
                cursor[0] = a + n
                return list(text_ids[a:a + n])
            return take
    else:
        def prompt_of(n: int):
            return lambda: [rng.randrange(lo, vocab) for _ in range(n)]

    params = ss.sampling(args)
    made = []
    if own.record_tokens:
        import dataclasses

        params = dataclasses.replace(params, logprobs=0)
        add = eng.add_request
        eng.add_request = lambda *a, **k: made.append(add(*a, **k)) or made[-1]
        tok_times: list[float] = []  # when the newest request's each output token arrived (its ITL distribution)
        step = eng.step

        def timed_step():
            n0 = len(made[-1].output_ids) if made else 0
            out = step()
            if made and len(made[-1].output_ids) > n0:
                tok_times.extend([time.perf_counter()] * (len(made[-1].output_ids) - n0))
            return out

        eng.step = timed_step
    meet = _barrier(own.pp_barrier, args.pp_stage, args.pp_stages) if own.pp_barrier else (lambda: None)
    ss.warm(eng, args, prompt_of(1), params)  # the bucket warmup (--warmup); no warm request here
    if getattr(args, "pp_follow", False):  # a following pipeline (tools/pp_follow.py): meet once, every stage warm;
        meet()  # then stage 0 alone paces the requests (--gap-s lets the previous one leave the later stages)
        meet = lambda: None  # noqa: E731
    for n in own.warm_lengths:
        meet()
        t = time.perf_counter()
        eng.generate([prompt_of(n)()], params)
        print(f"warm request {n} tokens: {time.perf_counter() - t:.1f} s", flush=True)
    rows = []

    def flagstats():
        """KILN_DSA_CP_FLAGSTAT (models/mla.py): rank 0's per-layer counters (its row group's rows), summed."""
        from kiln.models import mla

        if not mla.CP_FLAGSTAT:
            return None
        tot = None
        for l in eng.runner.model.kv_layers():
            b = getattr(l, "cp_flagstat", None)
            if b is not None:
                v = b.cpu().double()
                tot = v if tot is None else tot + v
        return tot

    from kiln.models import mla as _mla

    for n in own.lengths:
        meet()
        if own.gap_s:
            time.sleep(own.gap_s)
        f0, k0 = flagstats(), _mla.local_k_counts(eng.runner.model)
        recs, wall, split = ss.run_level(eng, args, 1, 1, prompt_of(n), params)
        f1, k1 = flagstats(), _mla.local_k_counts(eng.runner.model)
        line = _mla.local_k_report(k0, k1, f"{n} tokens")
        if line:
            print(line, flush=True)
        if f1 is not None:
            from kiln.models import mla

            d = (f1 - f0).tolist()
            pairs, nrows = d[0], d[1]
            stat = {f"K{K}": {"pairs": d[2 + 2 * i], "rows": d[3 + 2 * i],
                              "pair_rate": d[2 + 2 * i] / max(pairs, 1), "row_rate": d[3 + 2 * i] / max(nrows, 1)}
                    for i, K in enumerate(mla.CP_FLAG_KS)}
            print(f"flagstat {n} tokens over {pairs:.0f} (row, rank) pairs, {nrows:.0f} rows: "
                  + ", ".join(f"K={K}: {v['pairs']:.0f} pairs ({v['pair_rate']:.2e}) {v['rows']:.0f} rows "
                              f"({v['row_rate']:.2e})" for K, v in zip(mla.CP_FLAG_KS, stat.values())), flush=True)
        ttft, itl, out_toks = recs[0][0], recs[0][1], recs[0][2]
        row = {"input_len": n, "ttft_s": round(ttft / 1e3, 3), "itl_p50_ms": round(itl, 2), "output_tokens": out_toks,
               "wall_s": round(wall, 1)}
        if own.record_tokens and made:
            r = made[-1]
            row["first_token"] = r.output_ids[0] if r.output_ids else None
            row["first_logprob"] = r.logprobs[0][0] if r.logprobs else None
            row["top_ids"] = r.logprobs[0][1][:5] if r.logprobs else None
            row["tokens"] = list(r.output_ids)
            row["logprobs"] = [x[0] for x in r.logprobs]
            gaps = sorted(b - a for a, b in zip(tok_times[-len(r.output_ids):], tok_times[-len(r.output_ids) + 1:]))
            if gaps:
                row["itl_ms_p50"] = round(gaps[len(gaps) // 2] * 1e3, 2)
                row["itl_ms_p90"] = round(gaps[min(len(gaps) - 1, int(0.9 * len(gaps)))] * 1e3, 2)
                row["itl_ms_max"] = round(gaps[-1] * 1e3, 2)
        if split:
            row["prefill_call_s"] = round(split["prefill_call_s"], 4)
            row["decode_call_s"] = round(split["decode_call_s"], 4)
            row["prefill_calls"] = split["prefill_calls"]
        rows.append(row)
        print(f"lone request {n} tokens: TTFT {row['ttft_s']} s, ITL {row['itl_p50_ms']} ms over {out_toks} tokens"
              + (f", prefill call {row['prefill_call_s']} s x {row['prefill_calls']}" if split else ""), flush=True)
    ss.print_profile()
    res = {"model": args.model, "tp": args.tp, "dp_attention": args.dp_attention, "prefill_tokens": args.prefill_tokens,
           "output_len": args.output_len, "rows": rows}
    print("RESULT", json.dumps(res), flush=True)
    if own.out_json:
        with open(own.out_json, "w") as f:
            json.dump(res, f)
    eng.close()


if __name__ == "__main__":
    main()
