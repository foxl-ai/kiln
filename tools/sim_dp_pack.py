"""The DP-attention scheduler alone, driven like bench/serve_sweep.py's closed loop, with a measured cost per graph
call instead of a model: how many prefill calls a workload takes against its packed minimum, and what a prefill
packing policy (engine/dp.py "Prefill packing") does to that, to out tok/s and to TTFT. Nothing touches a device;
the scheduler, radix cache, page pools and the DP placement are the engine's own classes, and a linear-attention
model's state checkpoints are counted by a stand-in row allocator.

    python tools/sim_dp_pack.py --concurrency 64 --max-num-seqs 64 --requests 256 --shared-prefix-len 6144 \\
        --num-prefixes 4 --input-len-min 2048 --pack off trim hold

Cost model (s per call): --prefill-call (one 4096-row call, ~0.97 s TP / 0.75 s EP on GLM-5.3-Flash tp=32 trn1)
and --decode-call (~0.15 s at 16 rows per group), the device split of docs/price-performance.md "G1b". A step costs
its decode calls plus its prefill calls; outputs are the token 1 (never EOS: the sweep ignores EOS). Under the
sweep's overlap the host side is hidden, so the simulated time is the device time.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))


class Rows:
    """Checkpoint rows per group, as engine/state_pool.StatePool hands them out (the scheduler's only use)."""

    def __init__(self, groups: int, rows: int):
        self.free = [list(range(rows, 0, -1)) for _ in range(groups)]

    def alloc_ckpt(self, group: int = 0):
        return self.free[group].pop() if self.free[group] else None

    def free_ckpt(self, row: int, group: int = 0) -> None:
        self.free[group].append(row)


def build(a, pack: str):
    import functools

    from kiln.engine.dp import DPScheduler
    from kiln.engine.kv_pool import PagePool
    from kiln.engine.radix_cache import RadixCache
    from kiln.engine.scheduler import Scheduler, SchedulerConfig

    N = a.dp_attention
    cfg = SchedulerConfig(page_size=32, max_num_seqs=-(-a.max_num_seqs // N), max_prefill_tokens=a.prefill_tokens // N,
                          max_model_len=a.input_len + a.output_len, recurrent=True, ckpt_track=256,
                          ckpt_lookahead=not a.no_lookahead, policy=a.policy)
    rows = Rows(N, 2 * cfg.max_num_seqs)
    groups = []
    for g in range(N):
        pool = PagePool(a.pages)
        radix = RadixCache(pool, 32)
        sch = Scheduler(cfg, pool, radix)
        sch.states, sch.dp_group = rows, g
        radix.free_ckpt = functools.partial(rows.free_ckpt, group=g)
        groups.append(sch)
    return DPScheduler(groups, pack=pack, pack_min=a.pack_min, hold_steps=a.hold_steps)


def run(a, pack: str, mixes: list[int]) -> list[dict]:
    import serve_sweep

    from kiln.engine.request import Request, SamplingParams, Status

    dps = build(a, pack)
    import random

    orng = random.Random(5)
    sp = SamplingParams(max_new_tokens=a.output_len, ignore_eos=True)
    out, ids, clock = [], 0, 0.0
    prompters = {}
    for n in mixes:
        if a.flush:
            for g in dps.groups:
                g.radix.evict(g.radix.total_pages())
        if n not in prompters:
            prompters[n] = serve_sweep.prompter(a, 100_000, 0, n)
        prompt = prompters[n]
        live, done, started = [], [], 0
        t0 = clock
        calls = chunks = deferred = 0
        while len(done) < a.requests:
            while len(live) < a.concurrency and started < a.requests:
                n_out = orng.randint(a.output_len_min, a.output_len) if a.output_len_min else a.output_len
                r = Request(f"r{ids}", prompt(), SamplingParams(max_new_tokens=n_out, ignore_eos=True) if
                            a.output_len_min else sp, arrival_time=clock)
                ids += 1
                dps.add(r)
                live.append(r)
                started += 1
            plan = dps.schedule()
            per = [0] * a.dp_attention
            for s in plan.prefills:
                per[s.req.dp_group] += 1
            pcalls = max(per)
            dec = [0] * a.dp_attention
            for s in plan.decodes:
                dec[s.req.dp_group] += 1
            dcalls = math.ceil(max(dec) / a.decode_bucket) if plan.decodes else 0
            clock += pcalls * a.prefill_call + dcalls * a.decode_call + (0.0 if plan else 1e-3)
            calls, chunks, deferred = calls + pcalls, chunks + len(plan.prefills), deferred + plan.deferred
            dps.update(plan, [1] * len(plan.seqs()))
            for s in plan.seqs():  # the scheduler stamps monotonic time; the simulation keeps its own clock
                r = s.req
                if s.sample and getattr(r, "_t1", None) is None:
                    r._t1 = clock
            still = []
            for r in live:
                if r.status is Status.FINISHED:
                    r._t2 = clock
                    done.append(r)
                else:
                    still.append(r)
            live = still
        wall = clock - t0
        ttft = sorted(r._t1 - r.arrival_time for r in done)
        toks = sum(len(r.output_ids) for r in done)
        cached = sum(max(r.num_cached_tokens, 0) for r in done) / sum(r.num_prompt for r in done)
        out.append({"pack": pack, "shared": n, "calls": calls, "chunks": chunks, "min": -(-chunks // a.dp_attention),
                    "deferred": deferred, "out_tok_s": round(toks / wall, 1), "wall_s": round(wall, 1),
                    "ttft_p50": round(ttft[len(ttft) // 2], 2), "ttft_p90": round(ttft[int(0.9 * len(ttft))], 2),
                    "hit": round(cached, 3)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dp-attention", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--requests", type=int, default=256)
    ap.add_argument("--input-len", type=int, default=8192)
    ap.add_argument("--input-len-min", type=int, default=0)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--output-len-min", type=int, default=0,
                    help="output lengths uniform in [this, --output-len]: requests finish out of step, as with MTP")
    ap.add_argument("--prefill-tokens", type=int, default=4096)
    ap.add_argument("--decode-bucket", type=int, default=16, help="decode rows per group per call")
    ap.add_argument("--pages", type=int, default=5500, help="KV pages per group")
    ap.add_argument("--shared-prefix-len", type=int, nargs="+", default=[0])
    ap.add_argument("--num-prefixes", type=int, default=4)
    ap.add_argument("--flush", action="store_true", help="flush the cache before each mix (the sweep's cold levels)")
    ap.add_argument("--no-lookahead", action="store_true")
    ap.add_argument("--policy", default="fcfs")
    ap.add_argument("--pack", nargs="+", default=["off", "trim", "hold"])
    ap.add_argument("--pack-min", type=int, default=None, help="default: the engine's (every group)")
    ap.add_argument("--hold-steps", type=int, default=1)
    ap.add_argument("--prefill-call", type=float, default=0.97)
    ap.add_argument("--decode-call", type=float, default=0.15)
    a = ap.parse_args()
    for pack in a.pack:
        for row in run(a, pack, a.shared_prefix_len):
            print(row, flush=True)


if __name__ == "__main__":
    main()
