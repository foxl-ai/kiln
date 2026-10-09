"""A prefill-sized dense SwiGLU MLP on one logical NeuronCore: Kiln's XLA form against nkilib's MLP kernel.

    python tools/probe_nkilib_dense.py [--rows 2048 4096] [--shapes 4096x3072 5120x3200]

Shapes are H x I per rank (Qwen3-8B at TP=4: 4096 x 3072; Qwen3-32B at TP=8: 5120 x 3200). The XLA form is
models/decoder.py _swiglu_mlp on Kiln's layout: gate_up [2I, H] and down [H, I]. The kernel form is
kernels/nkilib_dense.py mlp on the transposed copies (DecoderLayer.pack_dense_mlp). Reported per form:
- max |error| relative to the max of the fp32 host reference;
- time per call, with IN_FLIGHT launches chained;
- MFU against the logical core's dense BF16 peak.
The peak is 667 TFLOPS per trn2 chip over its 4 logical cores at LNC=2 (docs/research/models.md); the flop count
is 6 T H I (three matmuls).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PEAK_PER_LOGICAL_CORE = 667e12 / 4  # trn2: 667 dense BF16 TFLOPS per chip, 4 logical cores at LNC=2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument("--shapes", nargs="+", default=["4096x3072", "5120x3200"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--reduce", action="store_true",
                    help="return row sums, so a call's time is not its 2 T H-byte output's device-to-host copy")
    a = ap.parse_args()
    os.environ.setdefault("KILN_DENSE_MLP_KERNEL", "nkilib")
    from kiln import platform

    platform.configure_runtime_env()
    import libtorch_neuronx_lite  # noqa: F401
    import torch._dynamo as dynamo
    import torch.nn.functional as F

    from kiln.engine.model_runner import IN_FLIGHT, canonical_neuron_backend, neuronx_cc_args
    from kiln.kernels import nkilib_dense

    dynamo.config.cache_size_limit = 256
    dev = torch.device("neuron:0")
    for shp in a.shapes:
        H, I = (int(v) for v in shp.split("x"))
        g = torch.Generator().manual_seed(H + I)
        gate_up = (torch.randn(2 * I, H, generator=g) * H ** -0.5).to(torch.bfloat16)
        down = (torch.randn(H, I, generator=g) * I ** -0.5).to(torch.bfloat16)
        gu_d, down_d = gate_up.to(dev), down.to(dev)
        gate_t = gate_up[:I].t().contiguous().to(dev)
        up_t = gate_up[I:].t().contiguous().to(dev)
        down_t = down.t().contiguous().to(dev)

        def xla(x, gu, dn, gt, ut, dt):
            y = F.linear(x, gu)
            return F.linear(F.silu(y[..., :I]) * y[..., I:], dn)

        def nk(x, gu, dn, gt, ut, dt):
            return nkilib_dense.mlp(x, gt, ut, dt)

        if a.reduce:  # time the MLP, not its [T, H] output's copy to the host: each call returns its row sums [T]
            xla = (lambda f: lambda *t: f(*t).float().sum(-1))(xla)
            nk = (lambda f: lambda *t: f(*t).float().sum(-1))(nk)

        for T in a.rows:
            x = torch.randn(T, H, generator=g).to(torch.bfloat16)
            gu32 = x.float() @ gate_up.float().t()
            ref = (F.silu(gu32[:, :I]) * gu32[:, I:]) @ down.float().t()
            xd = x.to(dev)
            ok = nkilib_dense.can_use_mlp(T, H, I)
            for name, fn in (("xla", xla), ("nkilib", nk)):
                if name == "nkilib" and not ok:
                    print(f"H={H} I={I} T={T} nkilib: outside the kernel's tiling constraints, skipped", flush=True)
                    continue
                f = torch.compile(fn, backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                                  options={"compiler_args": neuronx_cc_args(torch.bfloat16)})
                args = (xd, gu_d, down_d, gate_t, up_t, down_t)
                try:
                    o = f(*args).cpu().float()
                except Exception as e:  # noqa: BLE001
                    print(f"H={H} I={I} T={T} {name}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                    continue
                r = ref.sum(-1) if a.reduce else ref
                err = ((o - r).abs().max() / r.abs().max()).item()
                outs = []
                t = time.perf_counter()
                for _ in range(a.iters):
                    outs.append(f(*args))
                    if len(outs) > IN_FLIGHT:
                        outs[-1 - IN_FLIGHT].cpu()
                outs[-1].cpu()
                dt = (time.perf_counter() - t) / a.iters
                mfu = 6 * T * H * I / dt / PEAK_PER_LOGICAL_CORE
                print(f"H={H} I={I} T={T} {name}: {dt * 1e3:.3f} ms/call, MFU {100 * mfu:.1f}%, "
                      f"max|err|/max {err:.2e}", flush=True)


if __name__ == "__main__":
    main()
