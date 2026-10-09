"""kernels/gated_norm.py: KDA's gated RMSNorm as an NKI kernel. reference() against _mix_rows's tail on the host,
the kernel in the NKI CPU simulator against reference() (Neuron venv only). Device parity and speed:
tools/probe_gated_norm.py."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from kiln.kernels import gated_norm as gn


def _inputs(T, Hv, Dv=128, pad=0, seed=0):
    g = torch.Generator().manual_seed(seed)
    scale = torch.pow(10.0, torch.rand(T + pad, Hv, 1, generator=g) * 2 - 1)
    o = torch.randn(T + pad, Hv, Dv, generator=g) * scale
    z = (torch.randn(T, Hv, Dv, generator=g) * 2).to(torch.bfloat16)
    w = 1 + 0.1 * torch.randn(Dv, generator=g)
    return o, z, w


def test_reference_is_mix_rows_tail():
    """reference() is the KDA branch of linear_attn._mix_rows's gated norm, element for element."""
    o, z, w = _inputs(64, 3)
    eps = 1e-5
    of = o.to(torch.bfloat16).float()
    of = of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + eps)
    of = w * of
    want = (of * torch.sigmoid(z.float())).to(torch.bfloat16).reshape(64, -1)
    assert torch.equal(gn.reference(o, z, w, eps), want)


def test_takes_follows_the_flag_then_the_model_default(monkeypatch):
    monkeypatch.setattr(gn, "MODE", None)
    assert not gn.takes("kda", "sigmoid", 128)  # a model without the default (default=False)
    monkeypatch.setattr(gn, "MODE", "0")
    assert not gn.takes("kda", "sigmoid", 128, default=True)  # KILN_KDA_FUSED_NORM=0 turns glm5_next's off
    monkeypatch.setattr(gn, "MODE", "1")
    assert not gn.takes("gdn", "sigmoid", 128)
    assert not gn.takes("kda", "silu", 128)


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("T,Hv,pad", [(256, 2, 0), (200, 3, 56), (128, 1, 0)])
def test_nki_simulator_matches_reference(T, Hv, pad, monkeypatch):
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    o, z, w = _inputs(T, Hv, pad=pad, seed=T + Hv)
    got = torch.as_tensor(nki.simulate(gn.kiln_gated_norm_kernel)(o=o, z=z, w=w, eps=1e-5, rev=gn.REV))
    want = gn.reference(o[:T], z, w, 1e-5)
    assert got.shape == want.shape and got.dtype == torch.bfloat16
    diff = (got.float() - want.float()).abs()
    spacing = torch.pow(2.0, torch.floor(torch.log2(want.float().abs().clamp_min(1e-30))) - 7)
    assert (diff <= spacing * 1.01).all()  # at most one bf16 ulp (fp32 summation order, the final rounding)
    assert (got == want).float().mean().item() > 0.99
