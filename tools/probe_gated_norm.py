"""KDA's gated RMSNorm (kiln/kernels/gated_norm.py) on one NeuronCore against the XLA path it replaces
(models/linear_attn.py _mix_rows's tail: the same torch code compiled by neuronx-cc), at one rank's prefill shapes.

    python tools/probe_gated_norm.py --shape 4096x2 1024x8 [--pad 0]

--shape TxHv: rows x v heads per rank (4096x2: the 1M R8 call, 2 KDA v heads per rank at tp 32; 1024x8: the 8K G64
call, 8 per rank at attention TP 8); Dv = 128. Inputs: o fp32 ~ N(0, s^2) per head with s varied per (row, head)
over two decades (the delta rule's output scale varies), z bf16 ~ N(0, 2), w = 1 + N(0, 0.1). --pad rows of o past T
(the delta rule's padded output, passed as is). Reported per shape: p50 of synchronous calls of a graph returning the
fp32 sum of the output, for the XLA path and the kernel (and the same sum over an input of the output's shape alone,
the floor); against an fp64 reference on the host, each path's max |error| in bf16 ulps of the reference and the share
of outputs that equal the bf16-rounded reference; the share where the kernel equals the XLA path, and its max
difference in ulps.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def ulps(a: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """|a - ref| in units of the bf16 spacing at |ref| (ref fp64)."""
    r = ref.abs().clamp_min(1e-30)
    sp = torch.pow(2.0, torch.floor(torch.log2(r)) - 7)
    return (a.double() - ref).abs() / sp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs="+", default=["4096x2", "1024x8"])
    ap.add_argument("--pad", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    import profile_layer as pl

    pl.setup_device()
    from kiln.kernels import gated_norm as gn

    Dv = 128
    for shape in args.shape:
        T, Hv = (int(x) for x in shape.split("x"))
        g = torch.Generator().manual_seed(T + Hv)
        scale = torch.pow(10.0, torch.rand(T + args.pad, Hv, 1, generator=g) * 2 - 1)
        o = torch.randn(T + args.pad, Hv, Dv, generator=g) * scale
        z = (torch.randn(T, Hv, Dv, generator=g) * 2).to(torch.bfloat16)
        w = 1 + 0.1 * torch.randn(Dv, generator=g)
        eps = args.eps
        od, zd, wd = o.to(pl.DEV), z.to(pl.DEV), w.to(pl.DEV)

        def xla(o, z, w):
            return gn.reference(o[:T], z, w, eps)

        def kern(o, z, w):
            return gn.apply(o, z, w, eps)

        print(f"T {T}, Hv {Hv}, Dv {Dv}, pad {args.pad}, eps {eps}, rev {gn.REV:#010x}", flush=True)
        pl.timed(f"{shape} XLA path (_mix_rows's tail)", lambda *a: xla(*a).float().sum(), (od, zd, wd), args.iters)
        pl.timed(f"{shape} gated-norm kernel", lambda *a: kern(*a).float().sum(), (od, zd, wd), args.iters)
        flo = torch.randn(T, Hv * Dv).to(torch.bfloat16).to(pl.DEV)
        pl.timed(f"{shape} floor (the sum alone)", lambda a: a.float().sum(), (flo,), args.iters)

        cx = torch.compile(xla, **pl.OPTS)(od, zd, wd).cpu()
        ck = torch.compile(kern, **pl.OPTS)(od, zd, wd).cpu()
        of = o[:T].to(torch.bfloat16).double()
        ref = (w.double() * (of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps))
               * torch.sigmoid(z.double())).reshape(T, -1)
        rb = ref.to(torch.bfloat16)
        for name, a in (("XLA", cx), ("kernel", ck)):
            u = ulps(a, ref)
            print(f"  {name:<7} vs fp64: max {u.max().item():.3f} ulp, mean {u.mean().item():.4f}, "
                  f"equal to bf16(ref) {(a == rb).float().mean().item() * 100:.3f}%", flush=True)
        d = ulps(ck, cx.double())
        print(f"  kernel vs XLA: equal {(ck == cx).float().mean().item() * 100:.3f}%, max {d.max().item():.3f} ulp, "
              f"{int((d > 1.01).sum())} of {d.numel()} differ by more than 1 ulp", flush=True)


if __name__ == "__main__":
    main()
