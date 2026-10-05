"""Steady-state device time of the decode graph, per bucket, with host work excluded.

    python tools/profile_decode.py --batch 6 --pages 4,16,64

Inputs are built once and the compiled graph is called back to back, so the number is
the graph's own cost. The slope over the page axis is the attention (KV gather) cost;
the intercept is weights, sampler and launch.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--pages", default="4,16,64")
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--gather", default=None, help="page | token (sets KILN_GATHER)")
    args = ap.parse_args()
    import os

    if args.gather:
        os.environ["KILN_GATHER"] = args.gather

    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    pages = [int(p) for p in args.pages.split(",")]
    eng = LLMEngine(EngineConfig(model_path=args.model, device="neuron", dtype=torch.bfloat16,
                                 page_size=args.page_size, max_num_seqs=args.batch,
                                 max_model_len=max(pages) * args.page_size, kv_cache_gb=4.0,
                                 decode_batch_buckets=(args.batch,), page_buckets=tuple(pages)))
    r = eng.runner
    B = args.batch
    rng = np.random.default_rng(0)
    for P in pages:
        L = P * args.page_size
        table = np.arange(1, 1 + B * P, dtype=np.int64).reshape(B, P)
        ctx = np.full(B, L, np.int64)
        pos = ctx - 1
        slot = table[:, -1] * args.page_size + (L - 1) % args.page_size
        ids = rng.integers(0, 1000, B).astype(np.int64)
        from kiln.engine.model_runner import BOARD

        samp = r._sampling([None] * B)
        host = [ids, pos, table, ctx, slot, *samp, BOARD, np.full(B, -1, np.int64),
                np.full(B, r.scratch_slot, np.int64)]
        a = [r._dev(x) for x in host]
        t = time.perf_counter()
        r._decode(*a).cpu()
        compile_s = time.perf_counter() - t
        if P == pages[0]:
            # LNL's execution queue is bounded ("Execution Queue Full"): find its depth.
            depth = 0
            for n in (2, 4, 8, 16, 32):
                try:
                    outs = [r._decode(*a) for _ in range(n)]
                    outs[-1].cpu()
                    depth = n
                except RuntimeError as e:
                    print(f"queue depth: {n} back-to-back launches fail ({str(e)[:60]}); {depth} succeed")
                    time.sleep(1)
                    break
            else:
                print(f"queue depth: >= {depth}")
        r._decode(*a).cpu()
        t = time.perf_counter()
        for i in range(args.iters):
            out = r._decode(*a)
            if i % 2 == 1:
                out.cpu()  # stay inside the queue bound
        out.cpu()
        dt = (time.perf_counter() - t) / args.iters
        kv_mb = 2 * eng.mcfg.num_layers * B * L * eng.mcfg.num_kv_heads * eng.mcfg.head_dim * 2 / 2**20
        print(f"[{os.environ.get('KILN_GATHER', 'auto')}] B={B} P={P:>3} ctx={L:>5}: {dt * 1e3:7.2f} ms/step  (compile {compile_s:.0f}s, KV read {kv_mb:.0f} MB)")


if __name__ == "__main__":
    main()
