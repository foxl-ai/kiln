"""Several consecutive hyper-connection blocks (models/hybrid.py _block: the mHC collapse, a stand-in block,
the output mix) in ONE device graph against the same code on the host, per KILN_MHC_FORM, at
GLM-5.3-Flash's widths: what a one-block probe (tools/probe_mhc_output.py) cannot show, the device keeping
or dropping precision between blocks of one graph.

    python tools/probe_mhc_blocks.py [--rows 256] [--blocks 8] [--outlier 300]

The stand-in block is y = rms_norm(x) * 0.5 + 0.1 tanh(x) in bf16 (cheap, nonlinear, no matmul), the mHC
parameters random at the checkpoint's scales (fn ~ N(0, 1 / sqrt(4 x 4096)), base ~ N(0, 0.1), scales
~ 1 + N(0, 0.1)). Reported per form: max abs error of the device's final streams against the host's
(relative to their max).
"""

from __future__ import annotations

import argparse
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def fake_model(hc: int, H: int):
    from kiln.models.hybrid import HybridSpec

    cfg = types.SimpleNamespace(hybrid=HybridSpec("glm5_next", hc=hc, sinkhorn_iters=20, hc_eps=1e-6),
                                rms_norm_eps=1e-5, hidden_size=H)
    return types.SimpleNamespace(cfg=cfg)


def params(n_blocks: int, hc: int, H: int, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_blocks):
        k = 2 * hc + hc * hc
        out.append((torch.randn(k, hc * H, generator=g) / (hc * H) ** 0.5, torch.randn(k, generator=g) * 0.1,
                    1 + 0.1 * torch.randn(3, generator=g), torch.ones(H, dtype=torch.bfloat16)))
    return out


def layers_of(ps):
    return [(types.SimpleNamespace(hc_attn_fn=fw, hc_attn_base=b, hc_attn_scale=sc), nw) for fw, b, sc, nw in ps]


def run(model, streams, layers):
    """The blocks in order on streams [T, hc * H] (bf16), with the current KILN_MHC_FORM."""
    from kiln.models import hybrid
    from kiln.models.decoder import rms_norm

    for layer, nw in layers:
        def block(x, nw=nw):  # noqa: B023
            return (rms_norm(x, nw, 1e-5) * 0.5 + 0.1 * torch.tanh(x.float()).to(x.dtype)).to(x.dtype)
        streams = hybrid._block(model, layer, "attn", streams, nw, block)
    return streams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--outlier", type=float, default=300.0)
    ap.add_argument("--hidden", type=int, default=4096)
    args = ap.parse_args()
    import profile_layer as pl

    pl.setup_device()
    from kiln.models.hybrid import MHC_FORMS

    hc, H, T = 4, args.hidden, args.rows
    model = fake_model(hc, H)
    ps = params(args.blocks, hc, H)
    g = torch.Generator().manual_seed(0)
    S = torch.randn(T, hc, H, generator=g)
    idx = torch.randint(0, H, (hc, 8), generator=g)
    for m in range(hc):
        S[:, m, idx[m]] *= args.outlier
    streams = S.reshape(T, hc * H).bfloat16()
    from kiln.models import hybrid

    host_layers = layers_of(ps)
    dev_layers = layers_of([tuple(t.to(pl.DEV) for t in p) for p in ps])
    for form in MHC_FORMS:
        hybrid.MHC_FORM = form
        host = run(model, streams, host_layers).double()

        def graph(s):
            return run(model, s, dev_layers)
        got = torch.compile(graph, **pl.OPTS)(streams.to(pl.DEV)).cpu().double()
        scale = host.abs().max().item()
        e = (got - host).abs().max().item() / scale
        print(f"{form:<17} {len(ps)} blocks, {T} rows: device vs host {e:.2e} (|streams| max {scale:.1f})",
              flush=True)
        torch._dynamo.reset()


if __name__ == "__main__":
    main()
