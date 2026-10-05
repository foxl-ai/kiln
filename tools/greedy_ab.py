"""Greedy tokens of one engine at a serving configuration, saved, so that two trees can be compared token for token
on real text at the shapes a sweep runs (DP attention, chunked prefill, decode buckets, the same graphs).

    python tools/greedy_ab.py --out-json a.json --text-file wikitext2_test.txt --prompts 64 --tokens 64 -- \\
        <bench/serve_sweep.py engine arguments: --model ... --tp 32 --dp-attention 4 --piecewise ...>
    python tools/greedy_ab.py --compare a.json b.json

The engine is built exactly as bench/serve_sweep.py builds it (engine_config), so a farm queue captured for that
sweep holds these graphs. Prompts are consecutive slices of the text file's tokens with lengths spread from
--min-len to --max-len (all prompts submitted at once, so prefill chunks and decode batches mix as in serving);
each generates --tokens tokens greedily (temperature 0, EOS ignored). --compare prints, per prompt, the number of
leading tokens that agree, and the totals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))


def compare(a_path: str, b_path: str) -> None:
    with open(a_path) as f:
        a = json.load(f)
    with open(b_path) as f:
        b = json.load(f)
    same = agree = total = 0
    firsts = []
    for i, (x, y) in enumerate(zip(a["out"], b["out"])):
        n = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        agree += n
        total += max(len(x), len(y))
        same += x == y
        if x != y:
            firsts.append((i, a["lens"][i], n))
    print(f"{same} of {len(a['out'])} prompts identical; leading agreement {agree} of {total} tokens "
          f"({agree / max(total, 1):.4f})")
    for i, L, n in firsts[:20]:
        print(f"  prompt {i} ({L} tokens): first difference at generated token {n}")


def main() -> None:
    argv = sys.argv[1:]
    own, rest = (argv[: argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--text-file", default="wikitext2_test.txt")
    ap.add_argument("--prompts", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--min-len", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=8192)
    args = ap.parse_args(own)
    if args.compare:
        compare(*args.compare)
        return
    import serve_sweep as ss

    sargs = ss.build_parser().parse_args(rest)
    from kiln.engine.request import SamplingParams

    eng = ss.make_engine(sargs, sargs.core_base)
    with open(args.text_file) as f:
        ids = eng.tokenizer(f.read())["input_ids"]
    n = args.prompts
    lens = [args.min_len + (args.max_len - args.min_len) * i // max(1, n - 1) for i in range(n)]
    prompts, pos = [], 0
    for L in lens:
        if pos + L > len(ids):
            pos = 0
        prompts.append(ids[pos:pos + L])
        pos += L
    if sargs.warmup:
        w = eng.warmup()
        print(f"bucket warmup {w['seconds']:.1f}s over {w['graphs']} graphs", flush=True)
    reqs = eng.generate(prompts, SamplingParams(max_new_tokens=args.tokens, temperature=0.0, ignore_eos=True))
    out = [list(r.output_ids) for r in reqs]
    eng.close()
    with open(args.out_json, "w") as f:
        json.dump({"lens": lens, "out": out}, f)
    print(f"saved {args.out_json}: {n} prompts of {lens[0]}..{lens[-1]} tokens, {args.tokens} generated each", flush=True)


if __name__ == "__main__":
    main()
