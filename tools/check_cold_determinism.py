"""Cold-run reproducibility of one engine: the same prompts, each run ALONE (so every call's batch
composition is the same), after a cache flush, before and after other requests have used the engine's
KV pages, state rows and checkpoint rows. The same graph on the same inputs should be exact, so a
difference means some input differed (a reused KV page, state row or checkpoint row, the token board,
host-side state) or the device is not deterministic.

--ops is a comma list run in order (every prompt phase flushes the prefix cache first and runs each
prompt alone):

    X Y V W Z    the prompts of tools/check_prefix_cache.py: X_k = P_k + S_k, Y_k / Z_k other suffixes of
                 P_k, W_k / V_k = P_k[:U] + suffixes; X@3;4 runs only X_3 and X_4
    Zc           Z with a periodic checkpoint at the end of P_k (pc-chk5's "cold Z (interval
                 checkpoint)"), each checkpoint read back to the host as pc-chk5 did; Zn the same
                 checkpoints never read back
    zero         every device buffer a call reads besides the weights back to zeros (KV and per-token
                 caches, state pools, token board, DSA scratch: ModelRunner.device_buffers), every rank
    alloc        the host allocators back to their state at engine start: page free lists, state-row
                 and checkpoint-row free lists, token-board slots
    reset        zero + alloc: the engine's memory and allocators as right after start-up
    env:K=V      set an environment variable on rank 0 (read when a call's arguments are built)
    RD@k:n       replay ONE decode step of X_k n times on identical inputs: X_k runs to output token
                 --replay-at, its state row is saved to a checkpoint row, then n times (restore the row,
                 run the decode call at the same position) with every call's whole output compared
                 bitwise with the first (RD@k:n:scratch_row+null_page zeroes those buffers on every
                 rank before each call: ModelRunner.zero_buffers; :fill_cur / fill_next / fill_page fill the
                 decoded slot / the slots after it / both with different seeded values before each call, as
                 an earlier request's leftovers in a reused page; :fill_null0 the null page's slot 0, the key
                 every padded row reads, so the padded rows compute different values in every call); RP@k:n the
                 same for X_k's last prefill chunk; RA@k:n the decode replay launched ahead (up to 4 calls in
                 flight before the first is read, as overlap scheduling runs). A fill is an eager host-to-device
                 copy, which does not wait for calls already in flight: with RA it can land inside an earlier
                 call (docs/neuron-notes.md "Padded rows wrote the slot they read"), so RA fills are diagnostics
                 only; RD@k:n:...:t replays at output token t instead of --replay-at
    OW@k:n       whether an eager host-to-device copy waits for a graph call launched before it (write_order);
                 OWs@k:n the control, the call waited for before the copy; OWh@k:n the copy made by the host
                 tier's restore (ModelRunner.kv_load of a saved page into the decoded page), which must land last

Every X run is compared with the FIRST X run of the same prompt: tokens, chosen-token and top-5
logprobs, and the first output position that differs. Each call's host arguments are hashed too: after
`alloc` the page ids repeat, so the calls of two runs must hash alike, and with `zero` as well the
device sees identical inputs (any output difference is then the device's own).

pc-chk5's order (trn1.32xlarge, GLM-5.3-Flash, docs/neuron-notes.md "Prompt caching"), where X_3 and
X_4 differed past output token 16: --ops X,Y,V,Zc,X.

Two engines: --build <dir> makes a random GLM-5.3-Flash config (tests/test_glm5_next.py) and runs it
with this script's own engine arguments (CPU by default); or `-- <bench/serve_sweep.py arguments>`
with --text-file builds the sweep's engine and wikitext prompts as check_prefix_cache.py does.

    python tools/check_cold_determinism.py --build /tmp/glm --tp 4 --dp-attention 2 --overlap --piecewise
    python tools/check_cold_determinism.py --text-file /opt/kiln/wikitext2_test.txt --prefixes 6 \\
        --ops X,Y,V,Zc,X -- --model zai-org/GLM-5.3-Flash --device neuron --tp 32 ...
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compare(a, b) -> dict:
    n = min(len(a.output_ids), len(b.output_ids))
    first = next((i for i in range(n) if a.output_ids[i] != b.output_ids[i]), None)
    same = n if first is None else first
    lp = [abs(x[0] - y[0]) for x, y in zip(a.logprobs[:same], b.logprobs)]
    top = [abs(dict(zip(x[1], x[2]))[i] - v) for x, y in zip(a.logprobs[:same], b.logprobs)
           for i, v in zip(y[1], y[2]) if i in x[1]]
    tops = [max([abs(dict(zip(x[1], x[2]))[i] - v) for i, v in zip(y[1], y[2]) if i in x[1]] or [0.0])
            for x, y in zip(a.logprobs[:same], b.logprobs)]
    first_lp = next((i for i, (d, t) in enumerate(zip(lp, tops)) if d != 0 or t != 0), None)
    return {"tokens_equal": first is None, "first_token_diff": first, "first_lp_diff": first_lp,
            "lp_max": max(lp) if lp else None, "top_max": max(top) if top else None,
            "len": [len(a.output_ids), len(b.output_ids)],
            "lp_diffs": [float(f"{d:.3g}") for d in lp]}


def _digest(x) -> str:
    return hashlib.sha1(pickle.dumps(x, protocol=4)).hexdigest()[:12]


class Recorder:
    """Rank 0's graph calls (name, key, hash of the host arguments) in order, per phase."""

    def __init__(self, runner):
        self.calls: list = []
        orig = runner._exec

        def rec(name, key, host_args):
            self.calls.append((name, str(key), _digest(host_args)))
            return orig(name, key, host_args)

        runner._exec = rec
        orig_copy = runner.copy_state

        def rec_copy(src, dst, groups=None):
            if src:
                self.calls.append(("state_copy", "", _digest((list(src), list(dst), groups))))
            return orig_copy(src, dst, groups)

        runner.copy_state = rec_copy


def snapshot_allocators(eng) -> dict:
    r = eng.runner
    out = {"pages": [(list(p._free), bytes(p._is_free)) for p in eng.pools],
           "slots": (list(r._free_slots), dict(r._slot))}
    if r.state is not None:
        out["state"] = copy.deepcopy((r.state._frees, r.state._ckpt_frees))
    return out


def restore_allocators(eng, snap: dict) -> None:
    r = eng.runner
    for p, (free, is_free) in zip(eng.pools, snap["pages"]):
        if p.num_free != len(free):  # only after a flush with nothing running: every page is free again
            raise RuntimeError(f"{p.num_usable - p.num_free} pages still allocated")
        p._free, p._is_free = list(free), bytearray(is_free)
    r._free_slots, r._slot = list(snap["slots"][0]), dict(snap["slots"][1])
    if r.state is not None:
        if r.state._held:
            raise RuntimeError(f"state rows still held: {list(r.state._held)}")
        frees, ckpt = copy.deepcopy(snap["state"])
        r.state._frees, r.state._ckpt_frees = frees, ckpt
    r._rngs.clear()


TAPS: list = []  # (piece name, output on the host) of rank 0's piecewise graphs while TAP_ON
TAP_ON = [False]


def install_taps() -> None:
    """--tap: wrap every piecewise graph rank 0 builds (prep, layer groups, post) so that its output is
    copied to the host while TAP_ON (each copy also waits for the graph). Must run before the engine is
    built; the graphs themselves are unchanged."""
    from kiln.engine import model_runner as mr

    orig = mr._piecewise

    def tapped(model, compile_, *a, **k):
        n = [0]

        def c2(f):
            g = compile_(f)
            name = f"{getattr(f, '__name__', 'piece')}#{n[0]}"
            n[0] += 1

            def h(*args, **kw):
                out = g(*args, **kw)
                if TAP_ON[0]:
                    t = out[0] if isinstance(out, tuple) else out
                    TAPS.append((name, t.to("cpu", copy=True)))
                return out
            return h
        return orig(model, c2, *a, **k)

    mr._piecewise = tapped


def tap_diff(a: list, b: list) -> dict:
    """The first piece whose output differs between two calls' taps, and per piece the rows that differ."""
    out = {"pieces": len(a), "first": None, "per_piece": []}
    for i, ((na, ta), (nb, tb)) in enumerate(zip(a, b)):
        if ta.shape != tb.shape or not __import__("torch").equal(ta, tb):
            d = (ta.double() - tb.double()).abs() if ta.shape == tb.shape else None
            rows = (sorted({int(x) for x in (d.reshape(d.shape[0], -1).amax(-1) > 0).nonzero().flatten()})
                    if d is not None else "shape")
            out["per_piece"].append({"i": i, "name": na, "rows": rows,
                                     "max": float(d.max()) if d is not None else None})
            if out["first"] is None:
                out["first"] = i
    return out


def replay(eng, prompt, decode: bool, n: int, at: int, sp, ahead: int = 0, tap: int = 0, zero: str = "") -> dict:
    """RD / RP (module docstring): one call replayed n times on identical inputs."""
    import dataclasses

    import torch

    from kiln.engine.scheduler import ScheduledSeq

    r = eng.runner
    params = dataclasses.replace(sp, max_new_tokens=at + 8) if decode else sp
    req = eng.add_request(prompt, params)
    L = len(prompt)
    C = max(r.prefill_buckets)  # a group's chunk: the last one starts at (L - 1) // C * C
    start = L - 1 + at if decode else (L - 1) // C * C
    while eng.has_work():
        eng.step()
        if req.num_computed >= start and (req.is_decoding or not decode):
            break
    eng._drain()
    sch = eng.scheduler.groups[req.dp_group] if hasattr(eng.scheduler, "groups") else eng.scheduler
    if decode:
        start = req.num_computed
        seq = ScheduledSeq(req, start, 1, True)
        if not sch._reserve(req, start + 1):
            raise RuntimeError("no page for the replayed decode position")
    else:
        if req.num_computed > start:
            raise RuntimeError(f"prefill went past {start} ({req.num_computed})")
        start = req.num_computed
        seq = ScheduledSeq(req, start, L - start, True)
        if not sch._reserve(req, L):
            raise RuntimeError("no pages for the replayed chunk")
    g = req.dp_group
    row = r.state.row(req)
    ck = r.state.alloc_ckpt(g)
    r.copy_state([row], [ck], [g])
    outs, diffs, pending = [], [], []

    row_calls: dict = {}  # output row -> calls in which it differed from the first

    def check(i, out):
        if not outs:
            outs.append(out)
        elif not torch.equal(out, outs[0]):
            bad = (out != outs[0]).nonzero().tolist()
            for w in {b_[0] for b_ in bad}:
                row_calls[w] = row_calls.get(w, 0) + 1
            diffs.append({"i": i, "n": len(bad), "where": bad[:8], "rows": sorted({w[0] for w in bad}),
                          "max": float((out.double() - outs[0].double()).abs().max())})

    taps, snaps = [], []

    def snap():
        """Rank 0's (group 0's) scratch state row, its null page and DSA scratch, on the host."""
        out = {}
        if r.state is not None:
            for j, t_ in enumerate(r.state.pools()):
                out[f"state{j}[0]"] = t_[0].to("cpu", copy=True)
        for j, c in enumerate(r._paged_caches()):
            out[f"cache{j}[null page]"] = c[: r.ps].to("cpu", copy=True)
        rest = r.device_buffers()[len(r._paged_caches()) + (len(r.state.pools()) if r.state else 0) + 1:]
        for j, t_ in enumerate(rest):
            out[f"scratch{j}"] = t_.to("cpu", copy=True)
        out["board"] = r.board.to("cpu", copy=True)
        return out

    t0 = time.perf_counter()
    for i in range(n):
        r.copy_state([ck], [row], [g])
        for w in [x for x in zero.split("+") if x]:
            if w == "fill_null0":  # what padded rows read: slot 0 of the null page (ModelRunner.pad_slots)
                r.fill_slots(0, 1, 1000 + i)
            elif w in ("fill_cur", "fill_next", "fill_page"):
                # The request's current page around the decoded position: fill_cur the slot being written this
                # call, fill_next the slots after it (not yet written, masked), fill_page both; seeded per call.
                pg = req.pages[start // r.ps] * r.ps
                lo = pg + start % r.ps + (1 if w == "fill_next" else 0)
                hi = pg + start % r.ps + 1 if w == "fill_cur" else pg + r.ps
                if hi > lo:
                    r.fill_slots(lo, hi, 1000 + i)
            else:
                r.zero_buffers(w)
        if i < tap:
            snaps.append(snap())
        TAP_ON[0] = i < tap
        TAPS.clear()
        pending.append((i, r.decode([seq]) if decode else r.prefill([seq])))
        if TAP_ON[0]:
            taps.append(list(TAPS))
            TAP_ON[0] = False
        while len(pending) > ahead:
            j, o = pending.pop(0)
            check(j, o.cpu())
    for j, o in pending:
        check(j, o.cpu())
    secs = time.perf_counter() - t0
    r.state.free_ckpt(ck, g)
    eng.abort(req)
    eng.flush_cache()
    out = {"kind": "decode" if decode else "prefill", "position": start, "rows": list(outs[0].shape), "runs": n,
           "differ": len(diffs), "rows_differ": sorted(row_calls),
           "row_calls": {str(k): v for k, v in sorted(row_calls.items())},
           "diffs": diffs[:20], "seconds": round(secs, 1)}
    out["ahead"] = ahead
    out["group"] = g
    out["request_row"] = (r.last_layout or [(0,)])[0][0] if decode else None
    if len(snaps) > 1:
        import torch as _t
        ch = []
        for j in range(1, len(snaps)):
            ch.append(sorted(k for k in snaps[0] if not _t.equal(snaps[j][k], snaps[j - 1][k])))
        out["buffers_changed"] = ch
        print(f"rank-0 buffers that changed before each call: {json.dumps(ch)}", flush=True)
    if len(taps) > 1:
        out["taps"] = [tap_diff(taps[0], t) for t in taps[1:]]
        for j, t in enumerate(out["taps"]):
            print(f"taps call {j + 1} vs 0: {json.dumps(t)}", flush=True)
        out["tap_shapes"] = [[nm, list(x.shape)] for nm, x in taps[0]]
        if os.environ.get("KILN_TAP_DUMP"):
            import torch as _t
            _t.save({"taps": taps, "out0": outs[0], "snaps": snaps}, os.environ["KILN_TAP_DUMP"])
    print(f"replay {out['kind']} (ahead {ahead}) at {start}{' ' + zero if zero else ''}: {n} runs, {len(diffs)} differ "
          f"from the first, calls per differing output row {out['row_calls']} (the request's row: {out['request_row']}, "
          f"group {g}), {secs:.1f}s; "
          f"{json.dumps(diffs[:5])}", flush=True)
    return out


def write_order(eng, prompt, n: int, at: int, sp, wait: bool = False, host: bool = False) -> dict:
    """OW@k:n: is an eager host-to-device copy ordered after graph calls launched before it? X_k decodes to
    output token `at`; then n times: launch its decode call (which writes the decoded slot of every paged
    cache), WITHOUT waiting copy seeded data into that slot from the host (ModelRunner.fill_slots, as the
    host tier's restores do), then wait and read the slot back on rank 0. In program order the copy comes
    last, so the slot must hold the copied data; a slot holding the decode's own write means the copy ran
    before a call queued ahead of it."""
    import dataclasses

    import torch

    from kiln.engine.scheduler import ScheduledSeq

    r = eng.runner
    req = eng.add_request(prompt, dataclasses.replace(sp, max_new_tokens=at + 8))
    start = len(prompt) - 1 + at
    while eng.has_work():
        eng.step()
        if req.num_computed >= start and req.is_decoding:
            break
    eng._drain()
    sch = eng.scheduler.groups[req.dp_group] if hasattr(eng.scheduler, "groups") else eng.scheduler
    start = req.num_computed
    if not sch._reserve(req, start + 1):
        raise RuntimeError("no page for the decoded position")
    seq = ScheduledSeq(req, start, 1, True)
    g, row = req.dp_group, r.state.row(req)
    ck = r.state.alloc_ckpt(g)
    r.copy_state([row], [ck], [g])
    page = req.pages[start // r.ps]
    slot = page * r.ps + start % r.ps
    caches = r._paged_caches()
    overtaken = 0
    if host:
        # OWh: the copy is the host tier's restore (ModelRunner.kv_load) of a page saved with seeded data.
        pool = eng.pools[g] if len(eng.pools) > 1 else eng.pools[0]
        (scratch,) = pool.alloc(1)
        r.fill_slots(scratch * r.ps, (scratch + 1) * r.ps, 4999)
        r.kv_save(-4999, scratch, g)
        saved = [h.clone() for h in r._host_kv[-4999]] if r._mine(g) else None
    for i in range(n):
        r.copy_state([ck], [row], [g])
        out = r.decode([seq])
        if wait:  # OWs, the control: the decode call is done before the copy is issued
            out.cpu()
        if host:
            r.kv_load(-4999, page, g)
        else:
            r.fill_slots(slot, slot + 1, 5000 + i)
        out.cpu()
        if host:
            want = [h[start % r.ps : start % r.ps + 1] for h in saved]
        else:
            gen = torch.Generator().manual_seed(5000 + i)
            want = [torch.randn((1, *c.shape[1:]), generator=gen).to(c.dtype) for c in caches]
        got = [c[slot : slot + 1].to("cpu", copy=True) for c in caches]
        if not all(torch.equal(a_, b_) for a_, b_ in zip(want, got)):
            overtaken += 1
    if host:
        r.kv_drop(-4999, g)
        pool.free([scratch])
    r.state.free_ckpt(ck, g)
    eng.abort(req)
    eng.flush_cache()
    res = {"kind": "write-order", "position": start, "runs": n, "copy_overtaken": overtaken, "wait": wait, "host": host}
    print(f"write order at {start}{' (decode waited for first)' if wait else ''}{' (host-tier kv_load)' if host else ''}: "
          f"the eager copy issued after a decode "
          f"call was overwritten by that call in {overtaken} of {n} calls", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ops", default="X,Y,V,Zc,X")
    ap.add_argument("--prefixes", type=int, default=6, help="prompts per phase (K)")
    ap.add_argument("--new-tokens", type=int, default=None, help="default 64 (sweep) / 24 (--build)")
    ap.add_argument("--replay-at", type=int, default=20, help="RD: output tokens before the replayed step")
    ap.add_argument("--tap", type=int, default=0, help="record every piecewise graph's output in the first N replays")
    ap.add_argument("--assert-stable", action="store_true",
                    help="exit 1 unless every X rerun equals the first and every replay without a fill is bitwise "
                         "stable on every row (the device probe for docs/neuron-notes.md \"Padded rows wrote the "
                         "slot they read\")")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--dump-json", default=None,
                    help="every X run's outputs (token ids, chosen-token and top-5 logprobs per prompt), to compare "
                         "two trees offline")
    # The sweep engine (after --) and its prompts.
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--prefix-len", type=int, default=None, help="default 6144 (sweep) / 64 (--build)")
    # The random engine.
    ap.add_argument("--build", help="build a random GLM-5.3-Flash config here first (tests/test_glm5_next.py)")
    ap.add_argument("--model", help="an existing checkpoint directory for the random-engine arguments")
    ap.add_argument("--index-topk", type=int, default=16)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--dp-attention", type=int, default=1)
    ap.add_argument("--overlap", action="store_true")
    ap.add_argument("--piecewise", action="store_true")
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument("--page-size", type=int, default=8)
    ap.add_argument("--num-pages", type=int, default=400)
    ap.add_argument("--prompt-len", type=int, default=96)
    ap.add_argument("--prefill-tokens", type=int, default=32, help="max_prefill_tokens (engine total)")
    ap.add_argument("--prefill-buckets", default="16")
    ap.add_argument("--decode-buckets", default="4")
    ap.add_argument("--page-buckets", default=None)
    ap.add_argument("--max-num-seqs", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=384)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("sweep", nargs=argparse.REMAINDER, help="-- then bench/serve_sweep.py's arguments")
    a = ap.parse_args()

    import torch

    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    sweep = a.sweep[1:] if a.sweep[:1] == ["--"] else a.sweep
    if sweep:
        import serve_sweep

        args = serve_sweep.build_parser().parse_args(sweep)
        cfg = serve_sweep.engine_config(args)
        L, Pn, new = args.input_len, a.prefix_len or 6144, a.new_tokens or 64
        warm = args.warmup
    else:
        path = a.model
        if a.build:
            from tests.test_glm5_next import build

            build(a.build, seed=1, layers=a.layers, index_topk=a.index_topk)
            path = a.build
        ladder = lambda v: tuple(int(x) for x in v.split(",")) if v else None  # noqa: E731
        L, Pn, new = a.prompt_len, a.prefix_len or 64, a.new_tokens or 24
        max_len = L + new + 2 * a.page_size
        cfg = EngineConfig(
            model_path=path, device=a.device, dtype=getattr(torch, a.dtype), page_size=a.page_size,
            num_pages=a.num_pages, max_num_seqs=a.max_num_seqs, max_model_len=max_len,
            max_prefill_tokens=a.prefill_tokens, tp=a.tp, dp_attention=a.dp_attention, overlap=a.overlap,
            piecewise=a.piecewise, prefix_caching=not a.no_prefix_cache,
            prefill_token_buckets=ladder(a.prefill_buckets), decode_batch_buckets=ladder(a.decode_buckets),
            page_buckets=ladder(a.page_buckets) or (-(-max_len // a.page_size),))
        warm = False
    K = a.prefixes
    ps_ = cfg.page_size
    U = Pn - 144 if sweep else Pn - ps_ // 2
    jp = Pn // ps_ * ps_
    if a.tap:
        install_taps()
    t = time.perf_counter()
    eng = LLMEngine(cfg)
    print(f"engine up {time.perf_counter() - t:.1f}s", flush=True)
    res: dict = {}
    try:
        if warm:
            w = eng.warmup()
            print(f"bucket warmup {w['seconds']:.1f}s over {w['graphs']} graphs", flush=True)
        if sweep:
            with open(a.text_file) as f:
                ids = eng.tokenizer(f.read())["input_ids"]
        else:
            g = torch.Generator().manual_seed(a.seed)
            ids = torch.randint(2, a.vocab, (K * (Pn + 3 * (L - Pn) + 2 * (L - U)),), generator=g).tolist()
        pos = 0

        def take(n):
            nonlocal pos
            pos += n
            return ids[pos - n : pos]

        # The same draws, in the same order, as tools/check_prefix_cache.py.
        P = [take(Pn) for _ in range(K)]
        X, Y, Z = ([P[k] + take(L - Pn) for k in range(K)] for _ in range(3))
        W, V = ([P[k][:U] + take(L - U) for k in range(K)] for _ in range(2))
        prompts = {"X": X, "Y": Y, "Z": Z, "W": W, "V": V}
        sp = SamplingParams(max_new_tokens=new, logprobs=5, **({} if sweep else {"ignore_eos": True}))
        rec = Recorder(eng.runner)
        start = snapshot_allocators(eng)
        scheds = getattr(eng.scheduler, "groups", [eng.scheduler])
        ref: dict = {}
        runs, dumps = [], []

        for op in [o for o in a.ops.split(",") if o]:
            if op.startswith("env:"):  # env:KEY=VAL on rank 0 (where every call's host arguments are built)
                key, _, val = op[4:].partition("=")
                os.environ[key] = val
                print(f"env {key}={val!r}", flush=True)
                continue
            name, _, sub = op.partition("@")
            if name == "zero":
                t0 = time.perf_counter()
                eng.runner.zero_buffers()
                print(f"zero: {len(eng.runner.device_buffers())} buffers in {time.perf_counter() - t0:.1f}s", flush=True)
                continue
            if name in ("alloc", "reset"):
                eng.flush_cache()
                if name == "reset":
                    eng.runner.zero_buffers()
                restore_allocators(eng, start)
                print(f"{name}: allocators as at start", flush=True)
                continue
            if name in ("OW", "OWs", "OWh"):
                k, n, *z = sub.split(":")
                eng.flush_cache()
                res[f"{op} #{len(res)}"] = write_order(eng, X[int(k)], int(n), int(z[0]) if z else a.replay_at, sp,
                                                       wait=name == "OWs", host=name == "OWh")
                continue
            if name in ("RD", "RP", "RA"):
                k, n, *z = sub.split(":")
                k, n = int(k), int(n)
                eng.flush_cache()
                res[f"{op} #{len(res)}"] = replay(eng, X[k], name != "RP", n,
                                                    int(z[1]) if len(z) > 1 else a.replay_at, sp,
                                                    4 if name == "RA" else 0, a.tap, z[0] if z else "")
                continue
            if name not in ("X", "Y", "Z", "W", "V", "Zc", "Zn"):
                raise SystemExit(f"unknown op {op}")
            which = [int(x) for x in sub.split(";")] if sub else list(range(K))
            if sub and "," in sub:
                raise SystemExit("subsets are written X@3;4")
            ps = prompts[name[0]]
            eng.flush_cache()
            if name in ("Zc", "Zn"):
                for s in scheds:
                    s.cfg.ckpt_interval = jp
            t0 = time.perf_counter()
            reqs, calls = {}, {}
            for k in which:
                n0 = len(rec.calls)
                reqs[k] = eng.generate([ps[k]], sp)[0]
                calls[k] = rec.calls[n0:]
                if name == "Zc":  # pc-chk5's state_at: the checkpoint after P_k read to the host
                    m, _ = eng.radixes[0].match_checkpoint(ps[k], jp // ps_)
                    if m.num_pages * ps_ == jp and m.node.ckpt is not None:
                        [t_[m.node.ckpt].to("cpu", copy=True) for t_ in eng.runner.state.pools()]
            if name in ("Zc", "Zn"):
                for s in scheds:
                    s.cfg.ckpt_interval = 0
            print(f"phase {op}: {time.perf_counter() - t0:.1f}s, cached {[r.num_cached_tokens for r in reqs.values()]}, "
                  f"groups {[r.dp_group for r in reqs.values()]}", flush=True)
            if name != "X":
                continue
            run = len(runs)
            runs.append(op)
            dumps.append({k: {"ids": r.output_ids, "logprobs": r.logprobs} for k, r in reqs.items()})
            for k, r in reqs.items():
                if k not in ref:
                    ref[k] = (r, calls[k], run)
                    continue
                r0, c0, run0 = ref[k]
                c = compare(r0, r)
                same_calls = sum(x == y for x, y in zip(c0, calls[k]))
                c.update(calls=[len(c0), len(calls[k])], calls_equal=same_calls,
                         first_call_diff=next((i for i, (x, y) in enumerate(zip(c0, calls[k])) if x != y), None))
                res.setdefault(f"X run {run} ({op}) vs run {run0}", {})[k] = c
                print(f"X run {run} prompt {k} vs run {run0}: {json.dumps(c)}", flush=True)
    finally:
        eng.close()
    if a.dump_json:
        with open(a.dump_json, "w") as f:
            json.dump({"runs": runs, "outputs": dumps}, f)
    summary = {n: ({"exact": sum(c["tokens_equal"] and c["first_lp_diff"] is None for c in rs.values()),
                    "of": len(rs), "lp_max": max((c["lp_max"] or 0) for c in rs.values()),
                    "differ": [k for k, c in rs.items() if not (c["tokens_equal"] and c["first_lp_diff"] is None)]}
                   if "runs" not in rs else {k: v for k, v in rs.items() if k in ("kind", "position", "runs", "differ", "row_calls",
                                                                                 "request_row", "copy_overtaken")})
               for n, rs in res.items()}
    print("RESULT " + json.dumps(summary), flush=True)
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(res, f)
    if a.assert_stable:
        def unstable(n, v):
            if "runs" not in res[n]:
                return bool(v["differ"])
            if res[n]["kind"] == "write-order":  # OW / OWs report the runtime; the host tier's restore must land last
                return res[n]["host"] and res[n]["copy_overtaken"] > 0
            if "fill_null0" in n or (":fill" in n and res[n]["ahead"]):  # diagnostics (module docstring)
                return False
            if ":fill" in n:  # foreign data where the request's row must not look: that row stays exact
                return str(v["request_row"]) in v["row_calls"]
            return bool(v["differ"])

        bad = [n for n, v in summary.items() if unstable(n, v)]
        print(f"STABLE {not bad}" + (f": {bad}" if bad else ""), flush=True)
        if bad:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
