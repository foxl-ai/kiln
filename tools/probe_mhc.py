"""Time GLM-5.3-Flash's hyper-connection around one block on a NeuronCore, alone: kiln/models/hybrid.py
_mhc (unweighted fp32 RMSNorm of the 4 streams, fp32 mix projection, sigmoid pre / post, softmax +
Sinkhorn-Knopp comb) and _block's output combination, at T tokens, against the same graph with the
block replaced by its input (so the difference is the hyper-connection).

    python tools/probe_mhc.py --tokens 256 [--iters 20]

Random weights at GLM-5.3-Flash's shapes (hc_mult 4, hidden 4096, hc_sinkhorn_iters 20); the code
under test is hybrid._mhc itself, called on a stand-in model / layer carrying those tensors.
"""

from __future__ import annotations

import argparse
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--hc", type=int, default=4)
    ap.add_argument("--sinkhorn", type=int, default=20)
    args = ap.parse_args()
    import profile_layer as pl

    pl.setup_device()
    from kiln.models import hybrid

    T, H, hc = args.tokens, args.hidden, args.hc
    g = torch.Generator().manual_seed(0)
    d = lambda t: t.to(pl.DEV)  # noqa: E731
    hy = types.SimpleNamespace(hc=hc, hc_eps=1e-6, sinkhorn_iters=args.sinkhorn, family="glm5_next")
    model = types.SimpleNamespace(cfg=types.SimpleNamespace(hybrid=hy, rms_norm_eps=1e-5, hidden_size=H))
    fn = d(torch.randn(hc * (2 + hc), hc * H, generator=g) * 0.01)
    base = d(torch.randn(hc * (2 + hc), generator=g) * 0.1)
    scale = d(torch.ones(3))
    S = d((torch.randn(T, hc, H, generator=g) * 0.5).bfloat16())

    layer = types.SimpleNamespace(hc_attn_fn=fn, hc_attn_base=base, hc_attn_scale=scale)

    def with_hc(S):
        x, post, comb = hybrid._mhc(model, layer, "attn", S)
        y = x * 1.0  # the block itself replaced by its input
        out = post.to(S.dtype).unsqueeze(-1) * y.unsqueeze(1) + comb.to(S.dtype).transpose(-1, -2) @ S
        return out.reshape(T, hc * H)

    def mhc_only(S):  # _mhc alone (norm, projection, Sinkhorn), no output combination
        x, post, comb = hybrid._mhc(model, layer, "attn", S)
        return x, post, comb

    def combine_bmm(S, post, comb, y):  # _block's combination as written
        return post.to(S.dtype).unsqueeze(-1) * y.unsqueeze(1) + comb.to(S.dtype).transpose(-1, -2) @ S

    def combine_fma32(S, post, comb, y):  # the same sum as hc explicit fp32 multiply-adds
        Sf, c = S.float(), comb.float()
        out = post.unsqueeze(-1) * y.float().unsqueeze(1)
        for j in range(hc):
            out = out + c[:, j, :].unsqueeze(-1) * Sf[:, j : j + 1, :]
        return out.to(S.dtype)

    def combine_fma16(S, post, comb, y):  # bf16 multiply-adds
        c = comb.to(S.dtype)
        out = post.to(S.dtype).unsqueeze(-1) * y.unsqueeze(1)
        for j in range(hc):
            out = out + c[:, j, :].unsqueeze(-1) * S[:, j : j + 1, :]
        return out

    def without_hc(S):
        x = S.mean(dim=1)
        return (S + x.unsqueeze(1)).reshape(T, hc * H)

    print(f"hyper-connection, T={T}, hc={hc}, H={H}, Sinkhorn {args.sinkhorn}", flush=True)
    pl.timed(f"T={T} _mhc + output combination", with_hc, (S,))
    pl.timed(f"T={T} stand-in without it", without_hc, (S,))
    x, post, comb = torch.compile(mhc_only, **pl.OPTS)(S)
    y = (torch.randn(T, H, generator=g) * 0.5).bfloat16().to(pl.DEV)
    pl.timed(f"T={T} _mhc alone (norm, mix, Sinkhorn)", mhc_only, (S,))
    ref = combine_bmm(S.cpu().float(), post.cpu(), comb.cpu(), y.cpu().float())
    for name, f in (("bmm (as written)", combine_bmm), ("fp32 multiply-adds", combine_fma32),
                    ("bf16 multiply-adds", combine_fma16)):
        got = torch.compile(f, **pl.OPTS)(S, post, comb, y).cpu().float()
        err = (got - ref).abs().max().item() / ref.abs().max().item()
        print(f"  combination {name}: max rel error vs fp32 host {err:.2e}", flush=True)
        pl.timed(f"T={T} combination, {name}", f, (S, post, comb, y))


if __name__ == "__main__":
    main()
