"""A small decode call's MoE FFN on one NeuronCore: the served form (the XLA router and top-k, the per-pair expert kernel,
the XLA shared expert) against kernels/moe_ffn.py's one kernel, at GLM-5.3-Flash's tp=32 rank shapes (hidden 4096, 288
FP8 experts with 128 x 128 block scales in moe_dedupe's tile layout, 64 intermediate rows per rank, clamped SwiGLU, the
shared expert at the same shard size).

    python tools/probe_moe_ffn.py [--rows 1 4 15]

Reported per row count: max |error| of each form against the fused kernel's emulation relative to its max, and the time
per call chained (IN_FLIGHT launches queued) and with a read-back each.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

H, E, LIM, SCALE = 4096, 288, 10.0, 2.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 4, 15])
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    from kiln import platform

    platform.configure_runtime_env()
    import libtorch_neuronx_lite  # noqa: F401
    import torch._dynamo as dynamo

    from kiln.engine.model_runner import IN_FLIGHT, canonical_neuron_backend, neuronx_cc_args
    from kiln.kernels import moe_dedupe as mdd
    from kiln.kernels import moe_ffn as mf
    from kiln.models.quant import dequant, dequant_t
    from tests.test_moe_dedupe import experts128

    dynamo.config.cache_size_limit = 256
    dev = torch.device("neuron:0")
    g = torch.Generator().manual_seed(0)
    ws = experts128(e=E + 1, h=H)
    blob = mdd.pack(*ws)
    sh_gu = dequant(ws[0][E:], ws[1][E:], torch.float32)[0].bfloat16()  # [128, H]: the shared expert as XLA reads it
    sh_dn = dequant_t(ws[2][E:], ws[3][E:], torch.float32)[0].t().contiguous().bfloat16()  # [H, 64]
    w_router = (torch.randn(E, H, generator=g) * H ** -0.5).bfloat16()
    bias = torch.randn(E, generator=g) * 0.05
    rT = mf.router_t(w_router)
    d = {k: v.to(dev) for k, v in dict(blob=blob, sh_gu=sh_gu, sh_dn=sh_dn, w_router=w_router, bias=bias, rT=rT,
                                        blob_r=blob[:E].contiguous()).items()}

    def served(x, blob, sh_gu, sh_dn, w_router, bias, rT, blob_r):
        logits = F.linear(x.float(), w_router.float())
        scores = logits.sigmoid()
        _, topi = torch.topk(scores + bias, 8, dim=-1)
        topv = torch.gather(scores, 1, topi)
        topv = (topv / (topv.sum(-1, keepdim=True) + 1e-20) * SCALE).to(torch.bfloat16)
        y = mdd.moe_dedupe(x, topv, topi, blob_r, act=mdd.ACTS["silu_clamp"], limit=LIM)  # the routed experts' own blob
        gu = F.linear(x, sh_gu)
        gg, uu = gu[..., :64].clamp(max=LIM), gu[..., 64:].clamp(min=-LIM, max=LIM)
        return y + F.linear(F.silu(gg) * uu, sh_dn)

    def fused(x, blob, sh_gu, sh_dn, w_router, bias, rT, blob_r):
        return mf.moe_ffn(x, rT, w_router, bias, blob, 8, SCALE, True, act=mdd.ACTS["silu_clamp"], limit=LIM)

    for T in a.rows:
        x = torch.randn(T, H, generator=g).bfloat16()
        ref = mf.emulate(x, w_router, bias, blob, 8, SCALE, True, act=1, limit=LIM).float()
        xd = x.to(dev)
        for name, fn in (("served", served), ("fused", fused)):
            f = torch.compile(fn, backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                              options={"compiler_args": neuronx_cc_args(torch.bfloat16, True)})
            args = (xd, d["blob"], d["sh_gu"], d["sh_dn"], d["w_router"], d["bias"], d["rT"], d["blob_r"])
            try:
                o = f(*args).cpu().float()
            except Exception as e:
                print(f"T={T} {name}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                continue
            err = ((o - ref).abs().max() / ref.abs().max()).item()
            outs = []
            t = time.perf_counter()
            for _ in range(a.iters):
                outs.append(f(*args))
                if len(outs) > IN_FLIGHT:
                    outs[-1 - IN_FLIGHT].cpu()
            outs[-1].cpu()
            chained = (time.perf_counter() - t) / a.iters
            t = time.perf_counter()
            for _ in range(5):
                f(*args).cpu()
            sync = (time.perf_counter() - t) / 5
            print(f"T={T:3d} {name:6s}: err {err:.2e} | {chained * 1e3:.3f} ms chained, {sync * 1e3:.3f} ms with read-back",
                  flush=True)


if __name__ == "__main__":
    main()
