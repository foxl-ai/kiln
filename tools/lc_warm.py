"""Second-turn TTFT on a long document: one engine, a first request over a doc-len-token document (its own question
at the end) that generates answer-len tokens, then a second request whose prompt is the first one's prompt, its answer
and new-len new tokens (a chat's next turn on the same document). The second request's TTFT is a WARM number: its
prefix is served from the radix cache, and for a recurrent model (GLM-5.3-Flash's KDA layers) only up to the deepest
state checkpoint, so the first request runs with EngineConfig.state_checkpoint_prompt=True (its last prompt page
boundary is checkpointed) unless --no-ckpt-prompt. The cold TTFT of the first request is printed beside it.

    python tools/lc_warm.py --doc-len 1040384 --answer-len 16 --new-len 4096 --out-json w.json -- <serve_sweep args>

The serve_sweep arguments must give a prefix cache and at least one state-checkpoint row (--state-checkpoints >= 1;
the state pool's rows, 1 + max_num_seqs + state_checkpoints per DP group, are a graph input shape, so trade a
max_num_seqs row for a checkpoint row to keep a config's keys: --max-num-seqs 3 --state-checkpoints 1 sizes the pool as
--max-num-seqs 4 --state-checkpoints 0 does)."""

from __future__ import annotations

import dataclasses
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-len", type=int, default=1_040_384)
    ap.add_argument("--answer-len", type=int, default=16)
    ap.add_argument("--new-len", type=int, default=4096)
    ap.add_argument("--turns", type=int, default=2, help="turns after the first, each on the previous turn's prompt")
    ap.add_argument("--no-ckpt-prompt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    own, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    import serve_sweep as ss
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    args = ss.build_parser().parse_args(rest + ["--concurrency", "1", "--requests", "1"])
    args.skip_warm_request = True
    cfg = ss.engine_config(args, args.core_base)
    if not own.no_ckpt_prompt:
        cfg = dataclasses.replace(cfg, state_checkpoint_prompt=True)
    t = time.perf_counter()
    eng = LLMEngine(cfg)
    print(f"engine up {time.perf_counter() - t:.1f}s; prefix_caching {cfg.prefix_caching}, state_checkpoints "
          f"{cfg.state_checkpoints}, state_checkpoint_prompt {cfg.state_checkpoint_prompt}", flush=True)
    vocab = min(eng.mcfg.vocab_size, 100_000)
    rng = random.Random(own.seed)

    def toks(n: int) -> list[int]:
        return [rng.randrange(1000, vocab) for _ in range(n)]

    ss.warm(eng, args, lambda: toks(1), ss.sampling(args))
    sp = SamplingParams(max_new_tokens=own.answer_len, temperature=0.0, ignore_eos=True)
    rows = []
    prompt = toks(own.doc_len)
    for turn in range(1 + own.turns):
        r = eng.add_request(prompt, sp)
        while r.finish_time is None:
            eng.step()
        ttft = r.first_token_time - r.arrival_time
        itl = (r.finish_time - r.first_token_time) / max(len(r.output_ids) - 1, 1)
        row = {"turn": turn, "prompt_tokens": len(prompt), "cached_tokens": max(r.num_cached_tokens, 0),
               "new_tokens": len(prompt) - max(r.num_cached_tokens, 0), "ttft_s": round(ttft, 3),
               "itl_ms": round(itl * 1e3, 2), "kind": "cold" if turn == 0 else "warm"}
        rows.append(row)
        print(f"turn {turn}: {row}", flush=True)
        prompt = prompt + list(r.output_ids) + toks(own.new_len)
        if len(prompt) + own.answer_len > cfg.max_model_len:
            print(f"stopping: the next turn ({len(prompt)} tokens) would pass max_model_len {cfg.max_model_len}")
            break
    res = {"model": args.model, "doc_len": own.doc_len, "answer_len": own.answer_len, "new_len": own.new_len,
           "state_checkpoint_prompt": cfg.state_checkpoint_prompt, "rows": rows}
    print("RESULT", json.dumps(res), flush=True)
    if own.out_json:
        with open(own.out_json, "w") as f:
            json.dump(res, f)
    eng.close()


if __name__ == "__main__":
    main()
