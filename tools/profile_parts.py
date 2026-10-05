"""Device time of the decode step's parts, each compiled as its own graph.

    python tools/profile_parts.py --batch 6
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

import libtorch_neuronx_lite  # noqa: F401

DEV = torch.device("neuron:0")
OPTS = dict(backend="neuron_libtorch", fullgraph=True, dynamic=False)


def timed(name, fn, *args, iters=40):
    g = torch.compile(fn, **OPTS)
    t = time.perf_counter()
    g(*args)[0].cpu()
    c = time.perf_counter() - t
    t = time.perf_counter()
    for i in range(iters):
        out = g(*args)
        if i % 2 == 1:
            out[0].cpu()
    out[0].cpu()
    print(f"  {name:<28} {(time.perf_counter() - t) / iters * 1e3:7.3f} ms   (compile {c:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=6)
    args = ap.parse_args()
    from kiln.config import ModelConfig
    from kiln.engine.sampler import sample
    from kiln.models.loader import load_model, resolve_model_path

    path = resolve_model_path("Qwen/Qwen3-0.6B")
    cfg = ModelConfig.from_pretrained(path)
    m = load_model(path, cfg, torch.bfloat16, DEV, 2048)
    B, H, V = args.batch, cfg.hidden_size, cfg.vocab_size
    h = torch.randn(B, H, dtype=torch.bfloat16).to(DEV)
    logits = torch.randn(B, V).to(DEV)
    f32 = lambda v: torch.full((B,), v).to(DEV)  # noqa: E731
    i64 = torch.zeros(B, dtype=torch.int64).to(DEV)
    noise = torch.rand(B, 64).to(DEV)
    noise1 = torch.rand(B, 1).to(DEV)
    L = m.layers[0]

    timed("lm_head matmul", lambda x: (F.linear(x, m.embed),), h)
    timed("lm_head + argmax", lambda x: (torch.argmax(F.linear(x, m.embed).float(), -1),), h)
    timed("argmax over vocab", lambda x: (torch.argmax(x, -1),), logits)
    timed("logsumexp over vocab", lambda x: (torch.logsumexp(x, -1),), logits)
    timed("topk 64 over vocab", lambda x: torch.topk(x, 64, -1), logits)
    timed("topk 20 over vocab", lambda x: torch.topk(x, 20, -1), logits)
    from kiln.engine.sampler import logsumexp_large, topk_large

    timed("topk_large 64", lambda x: topk_large(x, 64), logits)
    timed("logsumexp_large", lambda x: (logsumexp_large(x),), logits)
    timed("amax folded [B,C,128]", lambda x: (x.view(B, -1, 128).amax(-1),), logits)
    t6, one, zero = f32(0.6), f32(1.0), f32(0.0)
    timed("sampler K=64", lambda x, a, b, c, d, n: (sample(x, a, b, d, c, n),), logits, t6, one, zero, i64, noise)
    timed("sampler K=1 (greedy)", lambda x, a, b, c, d, n: (sample(x, a, b, d, c, n),), logits, zero, one, zero, i64, noise1)
    timed("MLP (1 layer)", lambda x: (m._mlp(L, x),), h)
    timed("QKV + norms + rope (1 layer)", lambda x, p: m._qkv(L, x, p), h, i64)
    timed("28 x MLP", lambda x: (_loop(m, x),), h)


def _loop(m, x):
    for L in m.layers:
        x = m._mlp(L, x)
    return x


if __name__ == "__main__":
    main()
