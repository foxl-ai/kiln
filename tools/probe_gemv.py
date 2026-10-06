"""A decode-sized projection on one NeuronCore: XLA's F.linear against kernels/gemv.py, GB/s of weight streamed.

    python tools/probe_gemv.py [--rows 1 4 16] [--shapes 4096x3072 1024x4096 4096x2048] [--rings 4 6 8]

Shapes are K x N (input features x output features), bf16 weights; XLA reads W [N, K] (nn.Linear's layout), the kernel
wT [K, N]. Reported: max |error| against the fp32 host product relative to its max, time per call chained (IN_FLIGHT
launches queued), and the weight bytes over that time.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--shapes", nargs="+", default=["4096x3072", "1024x4096", "4096x2048"])
    ap.add_argument("--rings", type=int, nargs="+", default=[6])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--chain", type=int, default=1, help="chained projections per graph (square K x K weights, "
                    "each its own tensor), so the graph's own launch cost does not hide the stream rate")
    a = ap.parse_args()
    from kiln import platform

    platform.configure_runtime_env()
    import libtorch_neuronx_lite  # noqa: F401
    import torch._dynamo as dynamo

    from kiln.engine.model_runner import IN_FLIGHT, canonical_neuron_backend, neuronx_cc_args
    from kiln.kernels import gemv as gk

    dynamo.config.cache_size_limit = 256
    dynamo.config.accumulated_cache_size_limit = 1024
    dev = torch.device("neuron:0")
    for shp in a.shapes:
        K, N = (int(v) for v in shp.split("x"))
        g = torch.Generator().manual_seed(K + N)
        C = a.chain
        if C > 1 and K != N:
            raise SystemExit("--chain needs square shapes")
        Ws = [(torch.randn(N, K, generator=g) * K ** -0.5).to(torch.bfloat16) for _ in range(C)]
        W = Ws[0]
        Wd = torch.stack(Ws).to(dev)  # [C, N, K]
        wTd = torch.stack([w.t().contiguous() for w in Ws]).to(dev)  # [C, K, N]
        for T in a.rows:
            x = (torch.randn(T, K, generator=g)).to(torch.bfloat16)
            ref = x.float()
            for w in Ws:
                ref = (ref @ w.float().t()).to(torch.bfloat16).float()
            xd = x.to(dev)
            def xla(x, W, wT):
                for i in range(C):
                    x = torch.nn.functional.linear(x, W[i])
                return x

            def nk(x, W, wT, r):
                for i in range(C):
                    x = gk.gemv(x, wT[i], r)
                return x

            forms = [("xla", xla, None)]
            if C > 1:
                forms.append(("nki one", lambda x, W, wT: gk.gemv_chain(x, wT), None))
            for r in a.rings:
                forms.append((f"nki r{r}", lambda x, W, wT, r=r: nk(x, W, wT, r), r))
            for name, fn, _ in forms:
                f = torch.compile(fn, backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                                  options={"compiler_args": neuronx_cc_args(torch.bfloat16)})
                try:
                    o = f(xd, Wd, wTd).cpu().float()
                except Exception as e:
                    print(f"{shp} T={T} {name}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
                    continue
                err = ((o - ref).abs().max() / ref.abs().max()).item()
                outs = []
                t = time.perf_counter()
                for _ in range(a.iters):
                    outs.append(f(xd, Wd, wTd))
                    if len(outs) > IN_FLIGHT:
                        outs[-1 - IN_FLIGHT].cpu()
                outs[-1].cpu()
                dt = (time.perf_counter() - t) / a.iters
                print(f"{shp} T={T:3d} {name:8s}: err {err:.1e} | {dt * 1e3:.3f} ms chained, {C * K * N * 2 / dt / 1e9:6.1f} GB/s "
                      f"of weight", flush=True)


if __name__ == "__main__":
    main()
