"""MTP speculative decoding on a sweep's own engine: greedy equality with the engine without MTP, and acceptance.

    # 1. the reference: the sweep's engine WITHOUT MTP (the baseline configuration's graphs), greedy
    python tools/check_mtp.py --sets random,wikitext,chat --n 16 --gen 256 --out-json ref.json -- <serve_sweep args>
    # 2. the same prompts with MTP, compared token for token with the reference
    python tools/check_mtp.py --sets random,wikitext,chat --n 16 --gen 256 --reference-json ref.json -- \
        <serve_sweep args> --spec-method mtp --spec-k 2 --state-checkpoints 0

The engine is bench/serve_sweep.py's (engine_config from the same arguments), so every graph is one the sweep's
compile-farm configuration already holds, and the reference is the engine the baseline sweep runs (its plain
decode graphs; an MTP engine runs a draftless sequence in its verify graph, EngineConfig.spec_verify_plain). Per
prompt set it generates the prompts greedily in one batch and reports, with MTP: drafts proposed / accepted,
tokens per verify (accepted + 1), the acceptance of each draft position given the ones before it were accepted;
against a reference: how many prompts give token-identical output, and at the first divergence the reference's
top-2 logprob margin (a verify graph scores a position with different shapes from a decode graph, and bf16 can
flip a near tie).

Sets: random = serve_sweep's prompts (random token ids, --input-len long, seeded as the sweep's replica 0) with
ignore_eos, what bench/serve_sweep.py measures; wikitext = slices of --text-file (the wikitext-2 test split,
s3://<your-bucket>/data/wikitext2_test.txt) of --text-len tokens, continued; chat = short questions
through the tokenizer's chat template. --gen tokens each, ignore_eos for all (a fixed count to compare).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))

CHAT = [
    "What is the capital of Australia, and why was it chosen over Sydney and Melbourne?",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain in three sentences how a refrigerator keeps food cold.",
    "List five common causes of a car battery dying overnight.",
    "Translate into French: The meeting has been moved to Thursday afternoon.",
    "What is the difference between TCP and UDP? Answer briefly.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "Give me a SQL query that counts orders per customer in a table named orders.",
    "Why is the sky blue? Explain it to a ten year old.",
    "What are the main differences between a virus and a bacterium?",
    "Write a haiku about autumn rain.",
    "How do I reverse a linked list in C? Show the code.",
    "Name three advantages and three disadvantages of remote work.",
    "What happens during a solar eclipse?",
    "Explain what a hash table is and how collisions are handled.",
    "Write a short polite email declining a meeting invitation.",
]


def own_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="random,wikitext,chat")
    ap.add_argument("--n", type=int, default=16, help="prompts per set (at most the engine's max_num_seqs)")
    ap.add_argument("--gen", type=int, default=256, help="tokens generated per prompt")
    ap.add_argument("--text-file", default="/opt/kiln/work/wikitext2_test.txt")
    ap.add_argument("--text-len", type=int, default=2048, help="tokens per wikitext prompt")
    ap.add_argument("--reference-json", default=None, help="an --out-json of the run without MTP, to compare with")
    ap.add_argument("--out-json", default=None, help="prompts, outputs and top-2 logprobs per set, and the stats")
    return ap


def prompt_sets(eng, args, sweep) -> dict:
    import serve_sweep

    tok = eng.tokenizer
    out = {}
    for name in args.sets.split(","):
        if name == "random":
            p = serve_sweep.prompter(sweep, min(eng.mcfg.vocab_size, 100_000), 0)
            out[name] = [p() for _ in range(args.n)]
        elif name == "wikitext":
            with open(args.text_file) as f:
                ids = tok(f.read(), add_special_tokens=False)["input_ids"]
            step = max(args.text_len, (len(ids) - args.text_len) // max(args.n, 1))
            out[name] = [ids[i : i + args.text_len] for i in range(0, step * args.n, step)
                         if i + args.text_len <= len(ids)][: args.n]
        elif name == "chat":
            qs = (CHAT * (-(-args.n // len(CHAT))))[: args.n]
            # the template as text, then token ids: tokenize=True returns a BatchEncoding in transformers 5.15
            out[name] = [tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                                     add_generation_prompt=True), add_special_tokens=False)["input_ids"]
                         for q in qs]
        else:
            raise SystemExit(f"unknown set {name!r}")
    return out


def counters(eng):
    return (eng.spec_proposed, eng.spec_accepted, eng.spec_verifies, list(eng.spec_pos_reached),
            list(eng.spec_pos_accepted))


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: check_mtp.py [options] -- <serve_sweep args>")
    i = argv.index("--")
    args = own_parser().parse_args(argv[:i])
    import serve_sweep

    sweep = serve_sweep.build_parser().parse_args(argv[i + 1 :])
    mtp = sweep.spec_method == "mtp"
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    ref = None
    if args.reference_json:
        with open(args.reference_json) as f:
            ref = json.load(f)["sets"]
    t = time.perf_counter()
    eng = LLMEngine(serve_sweep.engine_config(sweep))
    print(f"engine up {time.perf_counter() - t:.1f}s", flush=True)
    if sweep.warmup:
        w = eng.warmup()
        print(f"bucket warmup {w['seconds']:.1f}s over {w['graphs']} graphs", flush=True)
    params = SamplingParams(max_new_tokens=args.gen, temperature=0.0, ignore_eos=True, logprobs=2)
    sets = prompt_sets(eng, args, sweep)
    report = {"spec_method": sweep.spec_method, "spec_k": sweep.spec_k if mtp else 0, "sets": {}}
    try:
        for name, ps in sets.items():
            c0 = counters(eng) if mtp else None
            t = time.perf_counter()
            got = eng.generate(ps, params)
            dt = time.perf_counter() - t
            rec = {"prompts": ps, "ids": [r.output_ids for r in got],
                   "top2": [[list(map(float, lp[2][:2])) for lp in r.logprobs] for r in got],
                   "prompt_tokens_mean": sum(map(len, ps)) / len(ps), "seconds": round(dt, 1)}
            line = f"{name}: {len(ps)} prompts ({rec['prompt_tokens_mean']:.0f} tokens) x {args.gen} in {dt:.1f}s"
            if mtp:
                c1 = counters(eng)
                p, a, v = (c1[j] - c0[j] for j in range(3))
                reached = [x - (c0[3][j] if j < len(c0[3]) else 0) for j, x in enumerate(c1[3])]
                acc = [x - (c0[4][j] if j < len(c0[4]) else 0) for j, x in enumerate(c1[4])]
                rec.update(proposed=p, accepted=a, verifies=v, acceptance=a / max(p, 1),
                           tokens_per_verify=1 + a / max(v, 1),
                           position_acceptance=[x / max(r, 1) for x, r in zip(acc, reached)])
                line += (f"; k={sweep.spec_k}: {a} of {p} drafts accepted ({rec['acceptance']:.1%}), "
                         f"{rec['tokens_per_verify']:.3f} tokens per verify over {v} verifies; by position "
                         + " ".join(f"{x:.3f}" for x in rec["position_acceptance"]))
            if ref is not None and name in ref:
                want = ref[name]
                if want["prompts"] != ps:
                    raise SystemExit(f"{name}: the reference was run on other prompts")
                same, divs = 0, []
                for j, ids in enumerate(rec["ids"]):
                    w = want["ids"][j]
                    n = next((q for q, (x, y) in enumerate(zip(w, ids)) if x != y), min(len(w), len(ids)))
                    if n == len(w) == len(ids):
                        same += 1
                        continue
                    top = want["top2"][j][n] if n < len(want["top2"][j]) else None
                    divs.append({"prompt": j, "at": n, "ref_margin": top[0] - top[1] if top and len(top) > 1 else None})
                rec.update(identical=same, divergences=divs)
                line += f"; greedy identical to the reference {same}/{len(ps)}" + (
                    "" if not divs else " (diverge at " + ", ".join(
                        f"token {d['at']} ref margin {d['ref_margin']:.4f}" if d["ref_margin"] is not None
                        else f"token {d['at']}" for d in divs) + ")")
            print(line, flush=True)
            report["sets"][name] = rec
            if args.out_json:  # after every set, so a later failure keeps the earlier sets
                with open(args.out_json, "w") as f:
                    json.dump(report, f)
    finally:
        eng.close()
    print("RESULT", json.dumps({n: {k: v for k, v in r.items() if k not in ("prompts", "ids", "top2")}
                                for n, r in report["sets"].items()}))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(report, f)


if __name__ == "__main__":
    main()
