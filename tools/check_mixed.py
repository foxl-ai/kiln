"""Greedy text through a serving configuration, to compare mixed batches (KILN_MIXED_BATCH=1) with the
unmixed engine on the device: the same engine bench/serve_sweep.py builds from the same arguments (so
the same graphs, from the same compile cache or farm queue), fed natural-text prompts of different
lengths instead of random ids, `--concurrency` in flight, greedy, --new-tokens each, and every output
written to --out.

    KILN_MIXED_BATCH=1 python tools/check_mixed.py <serve_sweep args> --out mixed.json
    KILN_MIXED_BATCH=0 python tools/check_mixed.py <serve_sweep args> --out plain.json
    python tools/check_mixed.py --compare plain.json mixed.json

Prompts are consecutive windows of --text-file (default: tools/check_ppl.py's LONG_TEXT, repeated),
their lengths cycling through --lengths, so prefill chunks of some requests meet the decodes of
others in most steps. --compare prints, per request, whether the outputs are equal and where they
first differ, and the share of output tokens that agree position by position.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def compare(a_path: str, b_path: str) -> None:
    """a: the reference (unmixed) run, b: the run under test. Besides equality, for runs written with
    logprobs: every position before a pair's first difference has the same input tokens in both runs, so the
    chosen-token logprobs there are a teacher-forced comparison; it is split by where b computed the token (a
    decode row inside a mixed call, a decode call, or a prefill chunk), and at each first difference a's
    top-1 minus top-2 logprob margin says how close that choice was."""
    with open(a_path) as f:
        a = json.load(f)
    with open(b_path) as f:
        b = json.load(f)
    same = agree = total = 0
    dl: dict[str, list[float]] = {"mixed decode row": [], "decode call": [], "prefill chunk": []}
    ds: dict[str, list[float]] = {k: [] for k in dl}  # the same, signed: under test minus reference
    margins = []
    for x, y in zip(a["requests"], b["requests"]):
        if x["prompt_len"] != y["prompt_len"]:
            raise SystemExit("the two runs did not use the same prompts")
        ox, oy = x["output_ids"], y["output_ids"]
        first = next((i for i, (p, q) in enumerate(zip(ox, oy)) if p != q), None)
        n = min(len(ox), len(oy))
        agree += sum(p == q for p, q in zip(ox, oy))
        total += n
        same += ox == oy
        lx, ly = x.get("logprobs"), y.get("logprobs")
        where = y.get("where")
        if lx and ly:
            for i in range(n if first is None else first + 1):  # inputs equal up to and including position first
                kind = where[i] if where else "decode call"
                d = (ly[i][0] - lx[i][0] if i < (n if first is None else first) else
                     ly[i][2][0] - lx[i][2][0])  # at the difference: the two top-1 logprobs
                dl[kind].append(abs(d))
                ds[kind].append(d)
            if first is not None:
                top = lx[first][2]
                margins.append((x["prompt_len"], first, top[0] - top[1], ly[first][2][0] - ly[first][2][1],
                                where[first] if where else "?"))
        print(f"  prompt {x['prompt_len']:>5}: {'equal' if ox == oy else f'first difference at output {first}'}"
              f"  {x['text'][:60]!r}" + ("" if ox == oy else f" | {y['text'][:60]!r}"))
    for p, i, mx, my, w in margins:
        print(f"  margin at the first difference, prompt {p} output {i} ({w} in the run under test): "
              f"top-1 minus top-2 logprob {mx:.4f} (reference) / {my:.4f} (under test)")
    for kind, v in dl.items():
        if v:
            v = sorted(v)
            sg = ds[kind]
            print(f"  teacher-forced |dlogprob| over {kind}s: n={len(v)} max {v[-1]:.4f} p99 {v[int(0.99 * (len(v) - 1))]:.4f}"
                  f" mean {sum(v) / len(v):.5f}; signed (under test minus reference) mean {sum(sg) / len(sg):+.5f}")
    print(f"RESULT equal={same}/{len(a['requests'])} token_agreement={agree / max(total, 1):.4f} ({agree}/{total})")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
        return
    from bench import serve_sweep

    ap = serve_sweep.build_parser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--lengths", default="8192,1500,6000,700,4096,3000,8000,2500",
                    help="comma list of prompt lengths in tokens, cycled over the requests")
    ap.add_argument("--new-tokens", type=int, default=64,
                    help="output tokens per request (--input-len / --output-len keep sizing the engine as the sweep's)")
    args = ap.parse_args()
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams
    from tools.check_ppl import LONG_TEXT

    conc = args.concurrency[0]
    n_req = args.requests or conc
    eng = LLMEngine(serve_sweep.engine_config(args, args.core_base))  # the sweep's --core-base
    text = LONG_TEXT
    if args.text_file:
        with open(args.text_file) as f:
            text = f.read()
    ids = eng.tokenizer(text)["input_ids"]
    lengths = [int(x) for x in args.lengths.split(",")]
    while len(ids) < max(lengths) + n_req * 997:
        ids = ids + ids
    prompts = [ids[i * 997 : i * 997 + lengths[i % len(lengths)]] for i in range(n_req)]
    if args.warmup:
        w = eng.warmup()
        print(f"bucket warmup {w['seconds']:.1f}s over {w['graphs']} graphs", flush=True)
    sp = SamplingParams(max_new_tokens=args.new_tokens, temperature=0.0, ignore_eos=True, logprobs=2)
    # Where each output token was computed: its first in a prefill chunk; a decode row of a mixed call
    # (ModelRunner.mixed) or a decode call after that.
    mixed_rows: set = set()
    if eng.runner.mixed_rows:
        real = eng.runner.mixed

        def tag(chunks, decs):
            mixed_rows.update((s.req.rid, s.start) for s in decs)
            return real(chunks, decs)

        eng.runner.mixed = tag
    t0 = time.perf_counter()
    reqs, live, started = [], [], 0
    while started < n_req or live:
        while len(live) < conc and started < n_req:
            r = eng.add_request(prompts[started], sp)
            reqs.append(r)
            live.append(r)
            started += 1
        eng.step()
        live = [r for r in live if r.finish_time is None]
    wall = time.perf_counter() - t0
    calls: dict = {}
    for key, n in eng.runner.calls.items():
        calls[key[0]] = calls.get(key[0], 0) + n
    out = {"mixed": eng.runner.mixed_rows, "wall_s": round(wall, 1), "calls": calls,
           "requests": [{"prompt_len": len(p), "output_ids": r.output_ids, "text": eng.tokenizer.decode(r.output_ids),
                         "logprobs": [[lp, list(map(int, ids)), list(map(float, lps))] for lp, ids, lps in r.logprobs],
                         "where": ["prefill chunk"] + ["mixed decode row" if (r.rid, len(p) + i) in mixed_rows
                                                       else "decode call" for i in range(len(r.output_ids) - 1)]}
                        for p, r in zip(prompts, reqs)]}
    eng.close()
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"RESULT mixed_rows={out['mixed']} wall_s={out['wall_s']} calls={calls} requests={n_req}", flush=True)
    for x in out["requests"][:4]:
        print(f"  {x['prompt_len']:>5}: {x['text'][:200]!r}", flush=True)


if __name__ == "__main__":
    main()
