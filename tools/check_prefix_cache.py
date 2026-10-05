"""Prefix-cache correctness on the device as served: a request whose prompt prefix comes from the
radix cache (MLA latent pages, the DSA pool keys stored in them, and for linear-attention layers the
recurrent state resumed from a checkpoint) must give the greedy tokens and logprobs a cold run of the
same prompt gives.

The engine is bench/serve_sweep.py's (its arguments after --, so a farm capture of that sweep has
every graph this needs; --plp adds the prompt-logprob post graph), and each phase submits one prompt
per shared prefix P_k at once, so the DP-attention groups, sequence-parallel streams and the cached and
uncached requests run together in the same calls as in serving. References are cold runs (after a
flush) of X_k = P_k + S_k1, Y_k, Z_k (other suffixes of P_k) and W_k, V_k (P_k[:U] plus suffixes, U not a
page multiple); then, compared with them:

    repeat    X_k cold again: the device's own run-to-run difference (the noise floor)
    save      X_k with Y_k queued beside it: X_k computes P_k and checkpoints the junction ahead (the
              copy must not disturb it), Y_k waits for it and resumes from it (`lookahead-hit`)
    hit       Z_k, a suffix never computed: resumes from the checkpoint after P_k, the uncached chunks
              on the same chunk grid as its cold run, so the arithmetic is the same
    hit-identical  X_k again: its whole prompt's KV is cached too; resumes from P_k and checkpoints the
              KV junction near its end, which moves a chunk boundary
    unaligned W_k with V_k queued: the cached prefix ends on the page below U, off the prefill chunk
              grid, so V_k's uncached chunks start elsewhere than in its cold run (rounding differs)

Measured on trn1.32xlarge (docs/neuron-notes.md "Prompt caching for GLM-5.3-Flash"): a cold run
repeated in the same batch composition is bit-identical, but changing what else shares the calls
(other requests' rows in the MoE kernels' batches) moves chosen-token logprobs by about 0.01 nats, so the
batched phases above measure serving noise. --solo runs every request alone (the other DP groups pad),
which makes the composition of the calls identical between a cold run and a cached one:

    solo-save     Y_k after X_k: the KV of P_k is cached, no checkpoint, so Y_k recomputes P_k on the
                  same chunk grid and copies the state out at the junction; against Y_k cold
    solo-hit      Z_k: resumes from that checkpoint, uncached chunks on the cold run's grid: the
                  arithmetic of the cold run, so any difference is the cache path's
    solo-unaligned  V_k after W_k and X_k: resumes from the page below U, off the chunk grid: against
                  V_k cold (the rounding of a different chunking) and against V_k cold with its chunks
                  split where the hit resumes (`solo-unaligned-same-grid`: the same arithmetic again);
                  `chunking` is the two cold runs against each other (no cache at all)

Text comes from --text-file (wikitext), cut into distinct prefixes and suffixes. Reported per request
of a cached phase against its cold run: cached tokens, the first differing output token, max |d| of
the chosen-token logprobs over the tokens both runs share, and (--plp) max / mean |d| of the prompt
logprobs of every uncached position.

    python tools/check_prefix_cache.py --text-file /opt/kiln/wikitext2_test.txt --prefix-len 6144 \\
        --prefixes 4 --new-tokens 32 -- --model zai-org/GLM-5.3-Flash --device neuron --tp 32 ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


CHUNKS: dict = {}  # request id -> [(start, end)] of every prefill entry it ran (--solo)


def compare(a, b, plp: bool) -> dict:
    """b against the reference a: requests of the same prompt."""
    n = min(len(a.output_ids), len(b.output_ids))
    first = next((i for i in range(n) if a.output_ids[i] != b.output_ids[i]), None)
    same = n if first is None else first
    lp = [abs(x[0] - y[0]) for x, y in zip(a.logprobs[:same], b.logprobs)]  # the same chosen tokens
    # The top-n alternatives at those positions: |d| of every id both runs list.
    top = [abs(dict(zip(x[1], x[2]))[i] - v) for x, y in zip(a.logprobs[:same], b.logprobs)
           for i, v in zip(y[1], y[2]) if i in x[1]]
    out = {"cached": b.num_cached_tokens, "group": b.dp_group, "tokens_equal": first is None, "first_diff": first,
           "out_lp_max": max(lp) if lp else None, "out_lp_mean": sum(lp) / len(lp) if lp else None,
           "top_lp_max": max(top) if top else None, "top_n": len(top)}
    out["lp_diffs"] = [round(x, 5) for x in lp[:16]]
    if CHUNKS:
        out["chunks_ref"], out["chunks_got"] = CHUNKS.get(a.rid), CHUNKS.get(b.rid)
    if first is not None:
        out["at_diff"] = {"ref": [a.output_ids[first], a.logprobs[first][0], a.logprobs[first][1][:3],
                                  [round(v, 3) for v in a.logprobs[first][2][:3]]],
                          "got": [b.output_ids[first], b.logprobs[first][0], b.logprobs[first][1][:3],
                                  [round(v, 3) for v in b.logprobs[first][2][:3]]]}
    if plp:
        qs = sorted(set(a.prompt_logprobs) & set(b.prompt_logprobs))
        d = [abs(a.prompt_logprobs[q][0] - b.prompt_logprobs[q][0]) for q in qs]
        top = sum(a.prompt_logprobs[q][2][0] == b.prompt_logprobs[q][2][0] for q in qs)
        out.update(plp_n=len(qs), plp_max=max(d) if d else None, plp_mean=sum(d) / len(d) if d else None,
                   plp_top1_agree=round(top / len(qs), 4) if qs else None,
                   plp_mean_lp=sum(b.prompt_logprobs[q][0] for q in qs) / len(qs) if qs else None)
    return out


def time_copies(eng, n_rep: int = 20) -> None:
    """p50 wall times on rank 0 alone (no collective in these graphs; the scratch row copied onto itself,
    the null page saved and loaded, so no live state changes): the state-checkpoint copy graph per row
    bucket (one restore or save of a checkpoint is one row), and the host tier's eager copies of one KV page
    and one state row (engine/hicache.py, ModelRunner._kv_save / _kv_load / _state_save / _state_load)."""
    import numpy as np
    import torch

    r = eng.runner
    out = {"state_row_bytes_rank": r.state.bytes_per_row() if r.state is not None else 0,
           "kv_page_bytes_rank": r.kv_page_bytes()}

    def p50(f):
        ts = []
        for _ in range(n_rep):
            t = time.perf_counter()
            f()
            ts.append(time.perf_counter() - t)
        return round(sorted(ts)[len(ts) // 2] * 1e3, 3)

    if r.state is not None:
        for n in r.copy_buckets:
            z = torch.from_numpy(np.zeros(n, np.int64)).to(r.device)
            out[f"copy_graph_{n}_rows_ms"] = p50(lambda: r._copy(z, z, *r.state.pools()).cpu())
        out["state_save_ms"] = p50(lambda: r._state_save(0, 0))
        out["state_load_ms"] = p50(lambda: (r._state_load(0, 0), r.state.rec[0][0, 0, 0].cpu()))
    out["kv_save_ms"] = p50(lambda: r._kv_save(0, 0))
    out["kv_load_ms"] = p50(lambda: (r._kv_load(0, 0), r.k_caches[0][0].cpu()))
    print("COPIES " + json.dumps(out), flush=True)


def main() -> None:
    import serve_sweep

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--prefix-len", type=int, default=6144)
    ap.add_argument("--prefixes", type=int, default=4, help="prompts per phase, one per shared prefix")
    ap.add_argument("--unaligned", type=int, default=0, help="U of the unaligned phase (default prefix-len - 144)")
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--plp", action="store_true",
                    help="also compare prompt logprobs of the uncached positions (adds the prompt-logprob post graph, "
                         "which the F0-4096 KV 1.5 GB configuration on trn1 has no HBM left for)")
    ap.add_argument("--solo", action="store_true", help="every request alone (identical call composition)")
    ap.add_argument("--solo-phases", default="repeat,save,hit,unaligned", help="comma list of the --solo comparisons")
    ap.add_argument("--time-copies", action="store_true",
                    help="also time rank 0's state-checkpoint copy graph and its host-tier copies (after the phases)")
    ap.add_argument("--copies-only", action="store_true", help="--time-copies without the phases")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("sweep", nargs=argparse.REMAINDER, help="-- then bench/serve_sweep.py's arguments")
    a = ap.parse_args()
    argv = a.sweep[1:] if a.sweep[:1] == ["--"] else a.sweep
    args = serve_sweep.build_parser().parse_args(argv)
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    L, K = args.input_len, a.prefixes
    U = a.unaligned or a.prefix_len - 144
    t = time.perf_counter()
    eng = LLMEngine(serve_sweep.engine_config(args))
    print(f"engine up {time.perf_counter() - t:.1f}s", flush=True)
    try:
        if args.warmup:
            w = eng.warmup()
            print(f"bucket warmup {w['seconds']:.1f}s over {w['graphs']} graphs", flush=True)
        if a.copies_only:
            time_copies(eng)
            return
        with open(a.text_file) as f:
            ids = eng.tokenizer(f.read())["input_ids"]
        need = K * (a.prefix_len + 3 * (L - a.prefix_len) + 2 * (L - U))
        if len(ids) < need:
            raise SystemExit(f"{a.text_file} has {len(ids)} tokens, the prompts need {need}")
        pos = 0

        def take(n):
            nonlocal pos
            pos += n
            return ids[pos - n : pos]

        P = [take(a.prefix_len) for _ in range(K)]
        X, Y, Z = ([P[k] + take(L - a.prefix_len) for k in range(K)] for _ in range(3))
        W, V = ([P[k][:U] + take(L - U) for k in range(K)] for _ in range(2))

        def sp(prompt_start):
            if a.plp:
                return SamplingParams(max_new_tokens=a.new_tokens, logprobs=5, prompt_logprobs=0,
                                      prompt_logprobs_start=prompt_start)
            return SamplingParams(max_new_tokens=a.new_tokens, logprobs=5)

        def phase(name, prompts, start, flush=False):
            if flush:
                eng.flush_cache()
            t0 = time.perf_counter()
            if a.solo:
                reqs = [eng.generate([p], sp(start))[0] for p in prompts]
            else:
                reqs = eng.generate(prompts, sp(start))
            print(f"phase {name}: {time.perf_counter() - t0:.1f}s, cached {[r.num_cached_tokens for r in reqs]}, "
                  f"groups {[r.dp_group for r in reqs]}, checkpoints per group "
                  f"{[r.num_ckpts for r in eng.radixes]}", flush=True)
            return reqs

        # With --plp the scored positions start where the cached prefix ends, so the cold and the cached
        # run score the same uncached positions (prompt logprobs limit the cache to the pages before them).
        s0, su = a.prefix_len + 1, U // eng.cfg.page_size * eng.cfg.page_size + 1
        ps_ = eng.cfg.page_size
        if a.solo:
            orig = eng.runner.prefill

            def logged(sq):
                for x in (sq if isinstance(sq, list) else [sq]):
                    CHUNKS.setdefault(x.req.rid, []).append((x.start, x.end))
                return orig(sq)

            eng.runner.prefill = logged
            jp = a.prefix_len // ps_ * ps_

            def state_at(ids):
                """Rank 0's shard of the checkpoint after ids[:jp], read to the host (None: none)."""
                m, _ = eng.radixes[0].match_checkpoint(ids, jp // ps_)
                if m.num_pages * ps_ != jp or m.node.ckpt is None:
                    return None
                return [t[m.node.ckpt].to("cpu", copy=True) for t in eng.runner.state.pools()]

            want = set(a.solo_phases.split(","))
            got, exp, ref = {}, {}, {}
            if want & {"repeat", "save", "hit"}:
                ref.update({n: phase(f"cold {n}", ps, st, flush=True) for n, ps, st in (("X", X, s0), ("Y", Y, s0))})
            if "repeat" in want:
                got["repeat"], exp["repeat"] = (ref["X"], phase("repeat", X, s0, flush=True)), 0
            if want & {"save", "hit"}:
                # Cold Z leaves a checkpoint after P_k on the chunk grid (a periodic target, no split): the
                # state a cold run has there, to compare with the junction checkpoint the hit restores.
                eng.scheduler.cfg.ckpt_interval = jp
                eng.flush_cache()
                ref["Z"], cold_state = [], []
                for p in Z:
                    ref["Z"] += phase("cold Z (interval checkpoint)", [p], s0)
                    cold_state.append(state_at(p))
                eng.scheduler.cfg.ckpt_interval = 0
                phase("solo prep X", X, s0, flush=True)
                got["solo-save"], exp["solo-save"] = (ref["Y"], phase("solo-save", Y, s0)), 0
                for k, p in enumerate(Z):
                    j = state_at(p)
                    d = None if j is None or cold_state[k] is None else max(
                        (x.float() - y.float()).abs().max().item() for x, y in zip(j, cold_state[k]))
                    print(f"checkpoint after P_{k}: junction (recomputed by Y_{k}) vs cold Z_{k}: max |d| {d}",
                          flush=True)
                got["solo-hit"], exp["solo-hit"] = (ref["Z"], phase("solo-hit", Z, s0)), jp
            if "host" in want:
                # The host tier (sweep argument --hicache-host-gb): every group's device cache evicted to host
                # memory, then Z_k must come back from it (pages and the junction checkpoint) and match cold.
                if not eng.host_tiers:
                    raise SystemExit("--solo-phases host needs --hicache-host-gb in the sweep arguments")
                if "Z" not in ref:
                    ref["Z"] = phase("cold Z", Z, s0, flush=True)
                phase("host prep X", X, s0, flush=True)
                phase("host prep Y (junction)", Y, s0)
                t0 = time.perf_counter()
                for rad in eng.radixes:
                    rad.evict(rad.total_pages())
                print(f"evicted every group's cache to the host tiers in {time.perf_counter() - t0:.2f}s: "
                      f"{[t.saved for t in eng.host_tiers]} pages, {[t.ckpts_saved for t in eng.host_tiers]} "
                      f"checkpoints saved; device pages left {[r.total_pages() for r in eng.radixes]}", flush=True)
                hit = phase("host-hit", Z, s0)
                print(f"restored {[t.restored for t in eng.host_tiers]} pages, "
                      f"{[t.ckpts_restored for t in eng.host_tiers]} checkpoints", flush=True)
                got["host-hit"], exp["host-hit"] = (ref["Z"], hit), jp
            if "unaligned" in want:
                ju = U // ps_ * ps_
                ref["V"] = phase("cold V", V, su, flush=True)
                # The same cold run with a checkpoint target at the page below U, the step's prefill of it
                # ending there: its later chunks start where the unaligned hit's do (ju, ju + 1024, ...),
                # so the two run the same arithmetic.
                import types

                scheds = getattr(eng.scheduler, "groups", [eng.scheduler])
                saved = [sch._chunks for sch in scheds]

                def first_only(self, req, start, n, _orig=type(scheds[0])._chunks):
                    out = _orig(self, req, start, n)
                    cut = next((i + 1 for i, e in enumerate(out) if e.save is not None), len(out))
                    return out[:cut]

                eng.scheduler.cfg.ckpt_interval = ju
                for sch in scheds:
                    sch._chunks = types.MethodType(first_only, sch)
                ref["V-grid"] = phase("cold V (chunks split where the hit resumes)", V, su, flush=True)
                for sch, o in zip(scheds, saved):
                    sch._chunks = o
                eng.scheduler.cfg.ckpt_interval = 0
                got["chunking"], exp["chunking"] = (ref["V"], ref["V-grid"]), 0
                phase("solo prep W", W, su, flush=True)
                phase("solo prep X (junction below U)", X, su)
                hit = phase("solo-unaligned", V, su)
                got["solo-unaligned"], exp["solo-unaligned"] = (ref["V"], hit), ju
                got["solo-unaligned-same-grid"], exp["solo-unaligned-same-grid"] = (ref["V-grid"], hit), ju
        else:
            ref = {n: phase(f"cold {n}", ps, st, flush=True)
                   for n, ps, st in (("X", X, s0), ("Y", Y, s0), ("Z", Z, s0), ("W", W, su), ("V", V, su))}
            got = {"repeat": (ref["X"], phase("repeat", X, s0, flush=True))}
            both = phase("save + lookahead-hit", X + Y, s0, flush=True)
            got["save"], got["lookahead-hit"] = (ref["X"], both[:K]), (ref["Y"], both[K:])
            got["hit"] = (ref["Z"], phase("hit", Z, s0))
            got["hit-identical"] = (ref["X"], phase("hit-identical", X, s0))
            both = phase("unaligned", W + V, su, flush=True)
            got["unaligned-save"], got["unaligned-hit"] = (ref["W"], both[:K]), (ref["V"], both[K:])
            exp = {"repeat": 0, "save": 0, "lookahead-hit": a.prefix_len // ps_ * ps_,
                   "hit": a.prefix_len // ps_ * ps_, "hit-identical": a.prefix_len // ps_ * ps_,
                   "unaligned-save": 0, "unaligned-hit": U // ps_ * ps_}
        res = {}
        for name, (r0, r1) in got.items():
            res[name] = [compare(x, y, a.plp) for x, y in zip(r0, r1)]
            for k, c in enumerate(res[name]):
                print(f"{name} prompt {k}: {json.dumps(c)}", flush=True)
        ok = {n: [c["cached"] for c in res[n]] for n in exp if any(c["cached"] != exp[n] for c in res[n])}
        print("RESULT " + json.dumps({"cached_as_expected": not ok, "unexpected_cached": ok, **{
            n: {"tokens_equal": sum(c["tokens_equal"] for c in rs), "of": len(rs),
                "out_lp_max": max((c["out_lp_max"] or 0) for c in rs),
                "out_lp_mean": sum((c["out_lp_mean"] or 0) for c in rs) / len(rs),
                "top_lp_max": max((c["top_lp_max"] or 0) for c in rs),
                **({"plp_max": max(c["plp_max"] for c in rs), "plp_mean": sum(c["plp_mean"] for c in rs) / len(rs),
                    "plp_top1_agree": min(c["plp_top1_agree"] for c in rs)} if a.plp else {})}
            for n, rs in res.items()}}), flush=True)
        if a.out_json:
            with open(a.out_json, "w") as f:
                json.dump(res, f)
        if a.time_copies:
            time_copies(eng)
    finally:
        eng.close()


if __name__ == "__main__":
    main()
