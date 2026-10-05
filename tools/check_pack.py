"""DP prefill packing (engine/dp.py) on the device: the same batch of real-text prompts of uneven lengths, run
with every packing mode on one engine (the cache flushed before each), greedy tokens and the chosen and top-5
logprobs compared with the first `off` run; a second `off` run is the noise floor of re-running the batch.

A packing mode decides only WHEN a chunk runs, never what it computes, but a deferred chunk can start where its
request's next chunk would have and it shares its call with other requests' chunks, so on the device it carries the
rounding of a different chunking and composition (docs/neuron-notes.md "Prompt caching for GLM-5.3-Flash on the
device": ~0.01-0.05 nats); greedy tokens part only at near ties. On CPU in fp32 the modes are exactly equal
(tests/test_linear_serving.py::test_dp_prefill_pack_keeps_every_token).

    python tools/check_pack.py --text-file wikitext2_test.txt --prompts 16 --min-len 1024 --new-tokens 48 \\
        --modes off off trim hold -- <bench/serve_sweep.py arguments>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    import serve_sweep
    from check_prefix_cache import compare

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--min-len", type=int, default=1024)
    ap.add_argument("--new-tokens", type=int, default=48)
    ap.add_argument("--modes", nargs="+", default=["off", "off", "trim", "hold"])
    ap.add_argument("--hold-steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("sweep", nargs=argparse.REMAINDER, help="-- then bench/serve_sweep.py's arguments")
    a = ap.parse_args()
    args = serve_sweep.build_parser().parse_args(a.sweep[1:] if a.sweep[:1] == ["--"] else a.sweep)
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    eng = LLMEngine(serve_sweep.engine_config(args))
    try:
        if args.warmup:
            w = eng.warmup()
            print(f"bucket warmup {w['seconds']:.1f}s over {w['graphs']} graphs", flush=True)
        with open(a.text_file) as f:
            ids = eng.tokenizer(f.read())["input_ids"]
        rng = random.Random(a.seed)
        lens = [rng.randint(a.min_len, args.input_len) for _ in range(a.prompts)]
        pos, prompts = 0, []
        for n in lens:
            prompts.append(ids[pos : pos + n])
            pos += n
        if pos > len(ids):
            raise SystemExit(f"{a.text_file} has {len(ids)} tokens, the prompts need {pos}")
        sp = SamplingParams(max_new_tokens=a.new_tokens, logprobs=5)
        runs = []
        for mode in a.modes:
            eng.scheduler.pack, eng.scheduler.hold_steps = mode, a.hold_steps
            eng.flush_cache()
            t0, deferred, calls = time.perf_counter(), 0, 0
            reqs = [eng.add_request(p, sp) for p in prompts]
            while eng.has_work():
                eng.step()
                deferred += eng.last_step.deferred_chunks
                calls += eng.last_step.prefill_calls
            runs.append(reqs)
            print(f"mode {mode}: {time.perf_counter() - t0:.1f}s, {calls} prefill calls, {deferred} chunks deferred, "
                  f"groups {[r.dp_group for r in reqs]}", flush=True)
        ref = runs[0]
        res = {}
        for i, (mode, got) in enumerate(zip(a.modes[1:], runs[1:]), 1):
            cs = [compare(x, y, False) for x, y in zip(ref, got)]
            key = f"{mode}#{i}"
            res[key] = {"tokens_equal": sum(c["tokens_equal"] for c in cs), "of": len(cs),
                        "out_lp_mean": sum((c["out_lp_mean"] or 0) for c in cs) / len(cs),
                        "out_lp_max": max((c["out_lp_max"] or 0) for c in cs),
                        "first_diffs": [c["first_diff"] for c in cs if c["first_diff"] is not None],
                        "margins_at_diff": [round(c["at_diff"]["ref"][3][0] - c["at_diff"]["ref"][3][1], 3)
                                            for c in cs if c.get("at_diff")]}
            print(f"{key} vs off#0: {json.dumps(res[key])}", flush=True)
        print("RESULT " + json.dumps({"prompt_lens": lens, **res}), flush=True)
    finally:
        eng.close()


if __name__ == "__main__":
    main()
