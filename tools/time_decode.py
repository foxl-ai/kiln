"""Decode step time on a real device, piece by piece, to compare compiler flags like for like.

    KILN_PROFILE_PIECES=1 KILN_PROFILE_EXEC=1 KILN_CC_ARGS="-O1" python tools/time_decode.py \\
        --steps 64 -- <bench/serve_sweep.py arguments>

Builds the engine exactly as the sweep does (serve_sweep.engine_config, real weights), then
launches the first decode bucket's step --steps times on rows of random token ids (so the MoE
routing spreads over experts as real tokens do, unlike warmup's all-zero rows). The KV rows are
the null page, as in warmup (a step-only, null-page number: every row reads one page), or with --real-kv every
row's own pages (random contents) at the sweep's context; every graph has static shapes, so the work per step is the
work of a step at that bucket. Prints the p50 of each piece (KILN_PROFILE_PIECES: prep, every layer group,
post, each timed synchronously on rank 0) and of the whole step (KILN_PROFILE_EXEC), the first
`--skip` steps dropped. Only decode graphs are built: with the farm (tools/compile_farm.py capture
--skip-prefill) they come from the compile cache.

With --spec-method mtp in the sweep arguments it then times, the same way, the verify step at that bucket
(Q = 1 + k rows per sequence, random ids and random drafts) and the MTP draft graph that follows a verify
(q = 1 + k, every draft accepted) and a decode step (q = 1): what one speculative step costs on the device
against a decode step.

--all-buckets runs every decode bucket in turn (each followed by its verify and MTP timings under --spec-method mtp)
instead of only the first, and every decode bucket also prints the host wall time per step (launch to logits on the
host), which holds without KILN_PROFILE_PIECES / KILN_PROFILE_EXEC's synchronous piece timing. --buckets picks the
decode buckets to time. --pieces runs each bucket's steps a second time with every piece timed synchronously
(KILN_PROFILE_PIECES' piece lines, from one engine; the wall line stays the plain pass's), and --price name=$/h adds
a decode-only cost line per bucket at the end: rows per step over the step's wall, and $ per 1M output tokens.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--skip", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-decode", action="store_true",
                    help="no plain decode graph (an MTP engine builds none): only the verify and the MTP after it")
    ap.add_argument("--all-buckets", action="store_true", help="every decode bucket in turn, not only the first")
    ap.add_argument("--buckets", default=None, help="comma list of decode buckets to time (instead of --all-buckets)")
    ap.add_argument("--pieces", action="store_true",
                    help="after the plain steps, the same steps again with every piece timed synchronously "
                         "(KILN_PROFILE_PIECES; the wall line is the plain pass's)")
    ap.add_argument("--price", type=lambda v: (v.split("=")[0], float(v.split("=")[1])), action="append",
                    default=[], help="name=dollars_per_hour for the decode-only $ per 1M output tokens")
    ap.add_argument("--real-kv", action="store_true",
                    help="every row its own pages (distinct ids through each group's pool, filled once with random "
                         "values, fp8 clamped) at a context of the sweep's --input-len plus a spread of its "
                         "--output-len, and its own state row; refuses a bucket whose rows do not fit the pool")
    ap.add_argument("sweep", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    sweep = a.sweep[1:] if a.sweep[:1] == ["--"] else a.sweep
    import serve_sweep

    from kiln.engine import model_runner as mr
    from kiln.engine.engine import LLMEngine
    from kiln.engine.kv_pool import NULL_PAGE
    from kiln.engine.model_runner import BOARD

    args = serve_sweep.build_parser().parse_args(sweep)
    cfg = serve_sweep.engine_config(args, args.core_base)  # --core-base: a disjoint core range of a shared box
    t0 = time.perf_counter()
    eng = LLMEngine(cfg)
    print(f"engine up {time.perf_counter() - t0:.1f}s cc_args={os.environ.get('KILN_CC_ARGS', '')!r}", flush=True)
    r = eng.runner
    rng = np.random.default_rng(a.seed)
    vocab = min(eng.mcfg.vocab_size, 100_000)
    if a.buckets:
        buckets = [int(b) for b in a.buckets.split(",")]
    else:
        buckets = list(r.decode_buckets) if a.all_buckets else list(r.decode_buckets[:1])
    if a.real_kv:
        lps = getattr(r, "lps", r.ps)  # cache rows per page on a rank (context-parallel DSA: ps / cp)
        npg = r._paged_caches()[0].shape[0] // lps
        t0 = time.perf_counter()
        for s0 in range(lps, npg * lps, 4096):  # every page but the null page, on every rank of every group
            r.fill_slots(s0, min(npg * lps, s0 + 4096), a.seed + s0)
        print(f"real KV: {npg} pages per group pool filled in {time.perf_counter() - t0:.1f}s", flush=True)
    rows = []
    for B in buckets:
        P, N = r.page_buckets[0], r.dp
        if a.real_kv:
            host, p50 = _decode_bucket_real(a, r, B, P, N, rng, vocab, mr, NULL_PAGE, BOARD, args)
        else:
            host, p50 = _decode_bucket(a, r, B, P, N, rng, vocab, mr, NULL_PAGE, BOARD)
        if p50 is not None:
            rows.append((B, N, p50))
        if cfg.spec_method == "mtp":
            # an engine whose draftless rows verify (EngineConfig.spec_verify_plain) builds no MTP-after-decode graph
            time_spec(r, a, rng, vocab, B, P, N, None if a.skip_decode or cfg.spec_merged else host)
            report(mr)
    for B, N, p50 in rows:
        line = (f"curve B={B} rows/step={N * B} step_ms={p50 * 1e3:.2f} tok_s={N * B / p50:.1f} "
                f"kv={'real' if a.real_kv else 'null-page'}")
        for name, usd in a.price:
            line += f" usd_per_m_out[{name}]={usd / 3600 / (N * B / p50) * 1e6:.3f}"
        print(line, flush=True)
    eng.close()


def _decode_bucket(a, r, B, P, N, rng, vocab, mr, NULL_PAGE, BOARD):
    """--steps decode steps at bucket B (rows per DP-attention group, unless --skip-decode): the pieces' and the
    step's p50 when KILN_PROFILE_PIECES / KILN_PROFILE_EXEC are set (with --pieces: a second pass with the pieces
    timed), and the host wall time per step. Returns the decode arguments (time_spec drafts after a decode step from
    them) and the wall p50 in seconds (None with --skip-decode)."""
    ids = np.zeros(B, np.int64)
    table = np.full((B, P), NULL_PAGE, np.int64)
    host = [rng.integers(min(1000, vocab // 2), vocab, N * B).astype(np.int64), ids, table, np.ones(B, np.int64), ids,
            *r._sampling([None] * (N * B)), BOARD, np.full(N * B, -1, np.int64), np.full(N * B, r.scratch_slot, np.int64)]
    Pw = r._swa_pages(1, P)
    if Pw:
        host += [None, None, np.full((B, Pw), NULL_PAGE, np.int64), np.zeros(B, np.int64)]
    host = r._with_state("decode", host, [0] * B)
    host = r._with_ngram("decode", host, [], N * B)
    if a.skip_decode:
        return host, None
    t0 = time.perf_counter()
    r._exec("decode", ("decode", B, P), host)  # first launch: load (or compile) every decode graph
    print(f"B={B}: first step (graph load or compile) {time.perf_counter() - t0:.1f}s", flush=True)
    wall = []
    for pieces in ([False, True] if a.pieces else [mr.PROFILE]):
        mr.PROFILE = pieces  # read by model_runner._piecewise's layers() at call time
        mr.PIECE_TIMES.clear()
        mr.EXEC_TIMES.clear()
        for i in range(a.steps + a.skip):
            if i == a.skip:
                mr.PIECE_TIMES.clear()
                mr.EXEC_TIMES.clear()
            host[0] = rng.integers(min(1000, vocab // 2), vocab, N * B).astype(np.int64)
            t = time.perf_counter()
            r._exec("decode", ("decode", B, P), host).cpu()
            if i >= a.skip and not (a.pieces and pieces):
                wall.append(time.perf_counter() - t)
    report(mr, f"B={B} ")
    wall.sort()
    print(f"B={B} rows/group, {N * B} rows/step: wall p50 {wall[len(wall) // 2] * 1e3:.3f} ms min {wall[0] * 1e3:.3f}"
          f" -> {N * B / wall[len(wall) // 2]:.1f} out tok/s", flush=True)
    return host, wall[len(wall) // 2]


def _decode_bucket_real(a, r, B, P, N, rng, vocab, mr, NULL_PAGE, BOARD, sweep_args):
    """_decode_bucket with real KV: in every group row b holds pages 1 + b S .. of its own (S the pages its context
    can reach; the pool's first B S pages after the null page, filled by --real-kv's fill_slots), its context L = --input-len + 1 + a spread over
    --output-len (positions L - 1, the new token's slot in its last page) and state row 1 + b, as
    ModelRunner.decode builds a running batch's arguments."""
    ps = r.ps
    npg = r._paged_caches()[0].shape[0] // getattr(r, "lps", ps)
    base, span = sweep_args.input_len, max(1, sweep_args.output_len)
    stride = min(P, -(-(base + span) // ps))  # pages a row can reach (the page bucket may be wider than the context)
    if 1 + B * stride > npg:
        raise SystemExit(f"--real-kv: {B} rows x {stride} pages need {1 + B * stride} pages per group, the pool has {npg}")
    if r.state is not None and r.state.rows < 1 + B:
        raise SystemExit(f"--real-kv: {B} rows need {1 + B} state rows, the pool has {r.state.rows}")
    pos = np.zeros((N, B), np.int64)
    table = np.full((N, B, P), NULL_PAGE, np.int64)
    ctx = np.ones((N, B), np.int64)
    slot = np.zeros((N, B), np.int64)
    srow = np.zeros((N, B), np.int64)
    for g in range(N):
        for b in range(B):
            L = min(P * ps, base + 1 + (37 * b + 11 * g) % span)
            n = -(-L // ps)
            pages = 1 + b * stride + np.arange(n)
            table[g, b, :n] = pages
            pos[g, b], ctx[g, b] = L - 1, L
            slot[g, b] = pages[(L - 1) // ps] * ps + (L - 1) % ps
            srow[g, b] = 1 + b
    G = r._grp
    host = [rng.integers(min(1000, vocab // 2), vocab, N * B).astype(np.int64), G(pos), G(table), G(ctx), G(slot),
            *r._sampling([None] * (N * B)), BOARD, np.full(N * B, -1, np.int64), np.full(N * B, r.scratch_slot, np.int64)]
    Pw = r._swa_pages(1, P)
    if Pw:
        raise SystemExit("--real-kv: sliding-window layers not covered")
    host = r._with_state("decode", host, G(srow))
    host = r._with_ngram("decode", host, [], N * B)
    print(f"B={B}: real KV, contexts {int(ctx.min())}..{int(ctx.max())} tokens, {int((ctx // ps + 1).sum())} pages "
          f"over {N} groups", flush=True)
    t0 = time.perf_counter()
    r._exec("decode", ("decode", B, P), host)
    print(f"B={B}: first step (graph load or compile) {time.perf_counter() - t0:.1f}s", flush=True)
    wall = []
    for pieces in ([False, True] if a.pieces else [mr.PROFILE]):
        mr.PROFILE = pieces
        mr.PIECE_TIMES.clear()
        mr.EXEC_TIMES.clear()
        for i in range(a.steps + a.skip):
            if i == a.skip:
                mr.PIECE_TIMES.clear()
                mr.EXEC_TIMES.clear()
            host[0] = rng.integers(min(1000, vocab // 2), vocab, N * B).astype(np.int64)
            t = time.perf_counter()
            r._exec("decode", ("decode", B, P), host).cpu()
            if i >= a.skip and not (a.pieces and pieces):
                wall.append(time.perf_counter() - t)
    report(mr, f"B={B} ")
    wall.sort()
    print(f"B={B} rows/group, {N * B} rows/step (real KV): wall p50 {wall[len(wall) // 2] * 1e3:.3f} ms min "
          f"{wall[0] * 1e3:.3f} -> {N * B / wall[len(wall) // 2]:.1f} out tok/s", flush=True)
    return host, wall[len(wall) // 2]


def report(mr, prefix: str = "") -> None:
    for (shape, gi), ts in sorted(mr.PIECE_TIMES.items()):
        ts = sorted(ts)
        print(f"{prefix}piece group {gi:>2} h{list(shape)}: n={len(ts)} p50 {ts[len(ts) // 2] * 1e3:.3f} ms", flush=True)
    for key, ts in mr.EXEC_TIMES.items():
        ts = sorted(ts)
        print(f"{prefix}step {key}: n={len(ts)} p50 {ts[len(ts) // 2] * 1e3:.3f} ms min {ts[0] * 1e3:.3f}", flush=True)
    mr.PIECE_TIMES.clear()
    mr.EXEC_TIMES.clear()


def time_spec(r, a, rng, vocab, B, P, N, dhost) -> None:
    """The verify graph at (B, 1 + k, P) on random ids and drafts, then the MTP graphs drafting after it
    and after a decode step, each --steps times after --skip, as the decode loop above."""
    from kiln.engine.kv_pool import NULL_PAGE
    from kiln.engine.model_runner import PARAM

    k = r.spec_k
    Q = 1 + k
    z = np.zeros((B, Q), np.int64)
    ids = rng.integers(min(1000, vocab // 2), vocab, (N * B, Q)).astype(np.int64)
    draft = ids.copy()
    draft[:, :k] = ids[:, 1:]  # the drafts are the inputs after the newest token
    vhost = [ids, z, np.full((B, P), NULL_PAGE, np.int64), z, *r._sampling([None] * (N * B * Q)),
             np.full(N * B * Q, 0.5, np.float32), draft.reshape(-1)]
    vhost = r._with_state("verify", vhost, np.zeros((B, 1 + Q), np.int64))
    vhost = r._with_ngram("verify", vhost, [], N * B * Q)
    rest = np.zeros((B, k - 1), np.int64) if k > 1 else None

    def mhost(q, hkey):
        return [rng.integers(min(1000, vocab // 2), vocab, (N * B, q)).astype(np.int64), np.zeros((B, q), np.int64),
                np.full((B, P), NULL_PAGE, np.int64), np.zeros((B, q), np.int64), hkey,
                np.tile(np.arange(q, dtype=np.int64), (N * B, 1)) + np.arange(N * B, dtype=np.int64)[:, None] * q,
                np.full(N * B, q - 1, np.int64), PARAM + "norm", rest, rest]

    import kiln.engine.model_runner as mr

    names = ("verify", "mtp after verify") + (("mtp after decode",) if dhost is not None else ())
    for name, run in zip(names, (lambda: r._exec("verify", ("verify", B, Q, P), vhost), None, None)):
        if name != "verify":
            q = Q if name == "mtp after verify" else 1  # after a decode step: one hidden row per sequence
            if q == Q:
                r._exec("verify", ("verify", B, Q, P), vhost).cpu()
            else:
                r._exec("decode", ("decode", B, P), dhost).cpu()
            hk, h = r.last_hidden, r._hidden[r.last_hidden]

            def run(hk=hk, h=h, q=q):
                r._hidden[hk] = h  # each MTP call stores its own hidden; keep this one from being evicted
                return r._exec("mtp", r._mtp_key(B, q, P, hk), mhost(q, hk))
        loop(name, run, a, mr)
    # The MTP pass after a prefill step (one chunk per group, the largest prefill bucket), drafting from the hidden
    # states of a real prefill call (warmup's arguments with random ids; this rank's rows only under
    # sequence-parallel streams, ModelRunner._hidden_sp): every rank must hold them, so they come from a graph call.
    from kiln.engine.model_runner import BOARD

    C = r.prefill_buckets[-1]
    z = np.zeros(C, np.int64)
    pargs = [rng.integers(min(1000, vocab // 2), vocab, N * C).astype(np.int64), z, np.full(P, NULL_PAGE, np.int64), z,
             np.zeros(N, np.int64), *r._sampling([None] * N), BOARD, np.full(N, r.scratch_slot, np.int64)]
    pargs = r._with_state("prefill", pargs, [0])
    pargs = r._with_ngram("prefill", pargs, [], N * C)
    r._exec("prefill", ("prefill", C, P), pargs).cpu()
    hk = r.last_hidden
    rows = r._hidden[hk].shape[0]
    z1 = np.zeros((1, C), np.int64)
    rest1 = np.zeros((1, k - 1), np.int64) if k > 1 else None
    margs = [rng.integers(min(1000, vocab // 2), vocab, (N, C)).astype(np.int64), z1, np.full((1, P), NULL_PAGE, np.int64), z1, hk,
             np.arange(N * C, dtype=np.int64).reshape(N, C), np.full(N, C - 1, np.int64), PARAM + "norm", rest1, rest1]
    loop(f"mtp after a prefill step ({N} x {C} rows, hidden {rows} rows)",
         lambda: r._exec("mtp", r._mtp_key(1, C, P, hk), r._with_sp_src(list(margs), hk)), a, mr)


def loop(name, run, a, mr) -> None:
    run().cpu()  # load
    mr.PIECE_TIMES.clear()
    mr.EXEC_TIMES.clear()
    for i in range(a.steps + a.skip):
        if i == a.skip:
            mr.PIECE_TIMES.clear()
            mr.EXEC_TIMES.clear()
        run().cpu()
    print(f"-- {name}", flush=True)
    report(mr)


if __name__ == "__main__":
    main()
