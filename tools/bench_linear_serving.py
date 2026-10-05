"""Prefix caching and speculative decoding for a linear-attention model on a real device.

    python tools/bench_linear_serving.py --model Qwen/Qwen3.5-0.8B --what parity --dtype fp32
    python tools/bench_linear_serving.py --model Qwen/Qwen3.5-0.8B --what ttft,spec,parts

parity: greedy tokens of the same prompts with the prefix cache off (the baseline), on, and with
  n-gram and suffix speculation, plus the baseline against transformers (fp32, host) on the
  longest prompts. Workload: one system prompt shared by several users (three rounds, so later
  users resume from the junction checkpoint), a second turn of two of the conversations, and
  repetitive text for the drafters. fp32 runs with KILN_CC_ARGS=--auto-cast=none (set here).
ttft: time to first token of users sharing a long system prompt, one request at a time, cache off
  against on; and a workload without sharing, to show what the checkpoints cost when nothing hits.
spec: decode tokens/s of plain greedy decoding against n-gram and suffix speculation on repetitive
  prompts, batch 1 and batch --max-num-seqs, with the outputs compared.
parts: per-graph device time (KILN_PROFILE_EXEC) of the decode graph, the verify graph with the
  state after every position (the default and the "chunk" form), the verify graph keeping only the
  last state (what a recompute-based rollback would need from it, KILN_PROBE_VERIFY_LAST_STATE_ONLY,
  measured in a child process), and the checkpoint copy graph; and the state bytes behind each.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SYSTEM = (
    "You are Kiln, a careful assistant for engineers who build inference systems. Follow these rules in "
    "every answer. First, restate the question in one short sentence. Second, answer it directly, with "
    "numbers where they exist. Third, if the question is ambiguous, say which reading you chose and why. "
    "Fourth, never invent a citation; when unsure, say so plainly. Fifth, keep the answer under eighty "
    "words unless the user asks for more. You know about accelerators, compilers, memory hierarchies, "
    "batching, caching, quantization, attention variants, state space models and linear attention, "
    "speculative decoding, and the trade-offs between latency and throughput. When the user writes code, "
    "answer with code first and an explanation after it. When the user asks for a list, number the items. "
    "When the user asks about cost, state the instance type and the price you assume.\n"
)
USERS = ["What is a radix tree used for in an inference server?", "Why does decode read the whole KV cache?",
         "Explain chunked prefill in two sentences.", "What does a recurrent state replace in linear attention?",
         "Give three ways to reduce time to first token.", "How do you roll back a rejected speculative token?",
         "What limits batch size on one accelerator?", "Why are static shapes hard for variable-length text?"]
REPEAT = ["Repeat after me exactly, five times: the cat sat on the mat by the door. the cat sat on the mat by the door.",
          "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\ndef mul(a, b):\n    return a * b\n\ndef div(a, b):\n",
          "List: apple, banana, cherry, date, apple, banana, cherry, date, apple, banana, cherry, date, apple,",
          "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,"]


# Compile-pass prompts for the spec section: different text, so suffix decoding's memory of earlier
# outputs (it keeps every finished request's) cannot draft the timed prompts from a rehearsal.
WARM = ["Count with me: one two three four, one two three four, one two three four, one two",
        "for i in range(10):\n    print(i)\nfor j in range(10):\n    print(j)\nfor k in range(10):\n",
        "Colors: red, green, blue, red, green, blue, red, green, blue, red, green,",
        "a b c d e f g h i j k l m n o p q r s t u v w x y z a b c d e f g h i j k"]


def make(args, **kw):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.models.loader import resolve_model_path

    base = dict(model_path=resolve_model_path(args.model), device=args.device,
                dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32, page_size=args.page_size,
                max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len, max_prefill_tokens=args.prefill_tokens,
                kv_cache_gb=args.kv_cache_gb, piecewise=True, piecewise_group=args.piecewise_group,
                decode_batch_buckets=tuple(sorted({1, args.max_num_seqs})), prefill_token_buckets=(args.prefill_tokens,),
                page_buckets=tuple(args.page_buckets), state_track_interval=args.track,
                state_checkpoint_prompt=bool(args.ckpt_prompt))
    base.update(kw)
    t = time.perf_counter()
    eng = LLMEngine(EngineConfig(**base))
    # Every bucket graph before anything is timed (a decode graph first used inside a timed run of a
    # speculative engine cost 0.7-1.1 s there: its warm-up prompt had drafted at every step).
    w = eng.warmup()
    print(f"engine {kw} up in {time.perf_counter() - t:.1f}s ({w['graphs']} graphs warm)", flush=True)
    return eng


def gen(eng, prompts, n, one_at_a_time=False):
    from kiln.engine.request import SamplingParams

    sp = SamplingParams(max_new_tokens=n, ignore_eos=True, logprobs=2)
    if one_at_a_time:
        return [eng.generate([p], sp)[0] for p in prompts]
    return eng.generate(prompts, sp)


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


PARITY = {"baseline": dict(prefix_caching=False), "prefix": {},
          "ngram": dict(prefix_caching=False, spec_method="ngram"),
          "suffix": dict(prefix_caching=False, spec_method="suffix"),
          "mtp": dict(prefix_caching=False, spec_method="mtp"),
          "prefix+ngram": dict(spec_method="ngram")}


def has_mtp(args) -> bool:
    from kiln.config import ModelConfig
    from kiln.models.loader import resolve_model_path

    return ModelConfig.from_pretrained(resolve_model_path(args.model)).mtp_layers > 0


def parity_one(args, name: str, base_json: str):
    """One engine configuration (its own process: device memory of an engine is not returned to the
    runtime when the engine object is dropped). The baseline writes its outputs to base_json; every
    other configuration builds the same second turns from them and compares."""
    kw = dict(PARITY[name])
    if "spec_method" in kw:
        kw["spec_k"] = args.spec_k
    eng = make(args, **kw)
    tok = eng.tokenizer
    shared = [tok(SYSTEM + u)["input_ids"] for u in USERS]
    rep = [tok(r)["input_ids"] for r in REPEAT]
    N = args.tokens
    base = None
    if name != "baseline":
        with open(base_json) as f:
            base = json.load(f)
    if args.oracle and base is not None and kw.get("spec_method") in ("ngram", "suffix", "mtp"):
        # Random-weight models seldom repeat themselves (and a random MTP head predicts nothing), so
        # drafts would rarely be accepted: draft the baseline's own tokens with one position corrupted
        # per call instead (cycling: none, the 1st, 2nd, 3rd), so full, partial and zero acceptance
        # all occur. An MTP engine still runs its drafting graph after every step; its drafts are
        # replaced.
        want = {tuple(pr): ids for pr, ids in zip(base["prompts"], base["ids"])}
        calls = [0]

        def draft(req):
            w = want.get(tuple(req.prompt_ids))
            L = len(req.output_ids)
            if w is None:
                return []
            d = list(w[L : min(L + args.spec_k, len(w) - 1)])
            c = (args.spec_k, 0, 1, 2)[calls[0] % 4]
            calls[0] += 1
            if c < len(d):
                d[c] = (d[c] + 1) % eng.mcfg.vocab_size
            return d

        for sch in getattr(eng.scheduler, "groups", [eng.scheduler]):
            sch.draft_fn = draft
    # One request at a time (decode batch 1) for the shared rounds and second turns, so every engine
    # runs them through graphs of the same batch size; the repetitive prompts batched.
    out = []
    for g in (shared[:1], shared[1:2], shared[2:]):
        out += gen(eng, g, N, one_at_a_time=True)
    first = base["ids"] if base else [r.output_ids for r in out]
    t2 = [shared[0] + first[0] + tok(" Now answer in one word.")["input_ids"],
          shared[3] + first[3] + tok(" Give an example.")["input_ids"]]
    out += gen(eng, t2, N, one_at_a_time=True)
    out += gen(eng, rep, N)
    ids = [r.output_ids for r in out]
    # top-1 minus top-2 logprob at every position: how close a divergence was to a tie
    margins = [[lp[2][0] - lp[2][1] if len(lp[2]) > 1 else None for lp in r.logprobs] for r in out]
    res = dict(cached=[r.num_cached_tokens for r in out], accepted=eng.spec_accepted, proposed=eng.spec_proposed,
               ckpts=eng.radix.num_ckpts, prompt_tokens=[r.num_prompt for r in out])
    if base is None:
        with open(base_json, "w") as f:
            json.dump({"ids": ids, "prompts": [r.prompt_ids for r in out], "margins": margins}, f)
    else:
        same = [first_diff(a, b) for a, b in zip(ids, base["ids"])]
        res.update(exact=sum(x == N for x in same), of=len(same), first_diff=same,
                   margin_at_diff=[round(base["margins"][i][d], 4) for i, d in enumerate(same) if d < N])
    print(name, json.dumps(res), flush=True)
    print("RESULT", json.dumps({name: res}))


def parity(args):
    base_json = os.path.abspath(f"parity-{args.dtype}-base.json")
    for name in PARITY:
        if name == "mtp" and not has_mtp(args):
            continue
        subprocess.run([sys.executable, __file__, *sys.argv[1:], "--child", f"parity_one:{name}:{base_json}"],
                       check=False)
    if args.no_reference:
        return
    from tools.hf_reference import load_reference

    from kiln.models.loader import resolve_model_path

    with open(base_json) as f:
        base = json.load(f)
    ref = load_reference(resolve_model_path(args.model))
    res = []
    for i in (0, 1, len(USERS), len(USERS) + 1):  # two shared-prompt users and the two second turns
        ids = base["prompts"][i]
        with torch.no_grad():
            r = ref.generate(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                             max_new_tokens=args.tokens, do_sample=False, eos_token_id=None,
                             pad_token_id=0)[0, len(ids):].tolist()
        res.append((len(ids), first_diff(base["ids"][i], r)))
    print("baseline vs transformers (prompt tokens, tokens equal of", args.tokens, "):", res, flush=True)
    print("RESULT", json.dumps({"baseline_vs_transformers": res}))


def ttft(args):
    for cache in ("off", "on"):
        subprocess.run([sys.executable, __file__, *sys.argv[1:], "--child", f"ttft_one:{cache}"], check=False)


def ttft_one(args, which: str):
    import random

    from kiln.engine.request import SamplingParams

    out = {}
    for cache in (which == "on",):
        eng = make(args, prefix_caching=cache)
        tok = eng.tokenizer
        sys_ids = tok(SYSTEM * args.system_repeat)["input_ids"]
        users = [tok(u)["input_ids"] for u in USERS]
        rng = random.Random(0)
        vocab = eng.mcfg.vocab_size
        unshared = [[rng.randrange(1000, vocab - 1000) for _ in range(len(sys_ids) + 16)] for _ in range(6)]
        sp = SamplingParams(max_new_tokens=2, ignore_eos=True)
        for warm in range(2):  # the first pass compiles every graph the workload needs
            # A different first token per pass: the timed pass must not hit what the warm-up cached.
            shared = [[1000 + warm] + sys_ids + u for u in users]
            res = []
            for p in shared:
                r = eng.generate([p], sp)[0]
                res.append((r.first_token_time - r.arrival_time, r.num_cached_tokens))
            nos = []
            for p in unshared:
                r = eng.generate([[t + warm for t in p]], sp)[0]
                nos.append(r.first_token_time - r.arrival_time)
        key = "on" if cache else "off"
        out[key] = dict(shared_ttft_ms=[round(t * 1e3, 1) for t, _ in res], cached=[c for _, c in res],
                        unshared_ttft_ms=[round(t * 1e3, 1) for t in nos], prompt_tokens=len(shared[0]),
                        ckpt_rows=eng.runner.state.num_ckpt_rows, row_mb=eng.runner.state.bytes_per_row() / 2**20)
        print(key, json.dumps(out[key]), flush=True)
    print("RESULT", json.dumps(out))


SPEC = {"plain": {}, "ngram": dict(spec_method="ngram"), "suffix": dict(spec_method="suffix"),
        "mtp": dict(spec_method="mtp")}


def spec(args):
    want_json = os.path.abspath(f"spec-{args.dtype}-plain.json")
    for name in SPEC:
        if name == "mtp" and not has_mtp(args):
            continue
        subprocess.run([sys.executable, __file__, *sys.argv[1:], "--child", f"spec_one:{name}:{want_json}"], check=False)


def spec_one(args, name: str, want_json: str):
    from kiln.engine.request import SamplingParams

    out = {}
    want = None
    if name != "plain":
        with open(want_json) as f:
            want = {int(k): v for k, v in json.load(f).items()}
    for name, kw in ((name, SPEC[name]),):
        kw = dict(kw, spec_k=args.spec_k) if kw else kw
        eng = make(args, prefix_caching=False, **kw)
        tok = eng.tokenizer
        ps = [tok(r)["input_ids"] for r in REPEAT]
        warm = [tok(r)["input_ids"] for r in WARM]
        res = {}
        for B in sorted({1, args.max_num_seqs}):
            batch = (ps * B)[:B]
            sp = SamplingParams(max_new_tokens=args.spec_tokens, ignore_eos=True)
            eng.generate((warm * B)[:B], sp)  # compiles
            a0, p0 = eng.spec_accepted, eng.spec_proposed
            t = time.perf_counter()
            reqs = eng.generate(batch, sp)
            dt = time.perf_counter() - t
            ids = [r.output_ids for r in reqs]
            if name == "plain":
                want = want or {}
                want[B] = ids
                with open(want_json, "w") as f:
                    json.dump(want, f)
            same = sum(a == b for a, b in zip(ids, want[B]))
            res[B] = dict(tok_s=round(sum(map(len, ids)) / dt, 1), seconds=round(dt, 3), identical=f"{same}/{B}",
                          accepted=eng.spec_accepted - a0, proposed=eng.spec_proposed - p0)
        out[name] = res
        print(name, json.dumps(res), flush=True)
    print("RESULT", json.dumps(out))


def parts(args):
    """Device time per graph, from KILN_PROFILE_EXEC's synchronous timings (model_runner.EXEC_TIMES)."""
    from kiln.engine import model_runner as mr

    if not mr.PROFILE_EXEC:
        raise SystemExit("run with KILN_PROFILE_EXEC=1 (set by main for --what parts)")
    from kiln.engine.request import SamplingParams

    out = {}
    eng = make(args, prefix_caching=True, spec_method="ngram", spec_k=args.spec_k)
    tok = eng.tokenizer
    ps = [tok(r)["input_ids"] for r in REPEAT]
    B = args.max_num_seqs
    batch = (ps * B)[:B]
    sp = SamplingParams(max_new_tokens=args.spec_tokens, ignore_eos=True)
    eng.generate(batch, sp)
    mr.EXEC_TIMES.clear()
    eng.generate(batch, sp)
    for k, v in sorted(mr.EXEC_TIMES.items(), key=lambda kv: str(kv[0])):
        if k[0] in ("decode", "verify") and len(v) >= 3:
            v = sorted(v)
            out[str(k)] = dict(p50_ms=round(v[len(v) // 2] * 1e3, 3), n=len(v))
    # The copy graph, timed directly (it does not go through _exec).
    st = eng.runner.state
    for n in eng.runner.copy_buckets:
        if n > st.num_free_ckpts:
            continue
        src, dst = list(range(1, n + 1)), st.free_ckpt_rows()[-n:]  # n running rows onto n free checkpoint rows
        ts = []
        for _ in range(20):
            if eng.device.type != "cpu":
                torch.zeros(1, device=eng.device).cpu()
            t = time.perf_counter()
            eng.runner.copy_state(src, dst)
            if eng.device.type != "cpu":
                st.rec[0][0, 0, 0, :1].cpu()
            ts.append(time.perf_counter() - t)
        ts.sort()
        out[f"copy n={n}"] = dict(p50_ms=round(ts[len(ts) // 2] * 1e3, 3))
    out["state_row_mb"] = st.bytes_per_row() / 2**20
    out["rows"] = dict(running=args.max_num_seqs * (1 + args.spec_k), ckpt=st.num_ckpt_rows,
                       total_gb=st.rows * st.bytes_per_row() / 2**30)
    print(json.dumps(out), flush=True)
    print("RESULT", json.dumps(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--what", default="parity,ttft,spec,parts")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--spec-tokens", type=int, default=128)
    ap.add_argument("--spec-k", type=int, default=4)
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--page-buckets", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--prefill-tokens", type=int, default=128)
    ap.add_argument("--kv-cache-gb", type=float, default=2.0)
    ap.add_argument("--piecewise-group", type=int, default=4)
    ap.add_argument("--track", type=int, default=64, help="state_track_interval")
    ap.add_argument("--ckpt-prompt", type=int, default=1, help="state_checkpoint_prompt (1 / 0)")
    ap.add_argument("--system-repeat", type=int, default=6, help="ttft: copies of the system prompt (about 160 tokens each)")
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument("--oracle", action="store_true",
                    help="parity: n-gram / suffix engines draft the baseline's tokens, one position corrupted per call")
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.dtype == "fp32":
        os.environ.setdefault("KILN_CC_ARGS", "--auto-cast=none")
    if args.child and ":" in args.child:  # one engine configuration of a section, in this process
        fn, *rest = args.child.split(":")
        {"parity_one": parity_one, "ttft_one": ttft_one, "spec_one": spec_one}[fn](args, *rest)
        return
    whats = args.child.split(",") if args.child else args.what.split(",")
    for what in whats:
        if what == "parts" and not args.child:
            # Each verify variant in a child process: the knobs are read at import time.
            for env in ({}, {"KILN_LA_VERIFY": "chunk"}, {"KILN_PROBE_VERIFY_LAST_STATE_ONLY": "1"}):
                e = dict(os.environ, KILN_PROFILE_EXEC="1", **env)
                print(f"== parts {env}", flush=True)
                subprocess.run([sys.executable, __file__, *sys.argv[1:], "--child", "parts"], env=e, check=False)
            continue
        print(f"== {what}", flush=True)
        {"parity": parity, "ttft": ttft, "spec": spec, "parts": parts}[what](args)


if __name__ == "__main__":
    main()
