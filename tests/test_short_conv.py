"""kernels/short_conv.py (KILN_LA_CONV_KERNEL=nki): the short causal conv of a linear-attention prefill chunk as an NKI kernel,
in the NKI simulator against models/linear_attn.py _causal_conv on the host, bit for bit, at grid 1 and at grid 2 (trn2 at LNC=2:
the two programs split the 128-channel tiles), for a whole number of row tiles and for ragged ones."""

from __future__ import annotations

import importlib.util

import pytest
import torch

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")


def _inputs(T: int, C: int, K: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    xe = (torch.randn(K - 1 + T, C, generator=g) * 0.5).to(torch.bfloat16)
    w = (torch.randn(C, K, generator=g) * 0.3).to(torch.bfloat16)
    return xe, w


def test_takes_only_device_chunk_forms(monkeypatch):
    """takes(): never on the host, never for a decode / verify form (3-D) or one row, only bf16 and whole channel tiles."""
    from kiln.kernels import short_conv as sc

    xe, w = _inputs(16, 256)
    monkeypatch.setattr(sc, "KERNEL", "nki")
    assert not sc.takes(xe, w, 16)  # a CPU tensor: the torch path
    monkeypatch.setattr(sc, "KERNEL", "xla")
    assert not sc.takes(xe, w, 16)


def test_default_is_nki_on_trn2_at_lnc2_only(monkeypatch):
    """KILN_LA_CONV_KERNEL unset: nki on trn2 at LNC=2 without context-parallel DSA; xla with it, on trn2 at LNC=1, on trn1 and
    on a host (trn1 keys unchanged)."""
    from kiln.kernels import short_conv as sc

    monkeypatch.delenv("KILN_LA_CONV_KERNEL", raising=False)
    monkeypatch.delenv("KILN_DSA_CP", raising=False)
    for tgt, lnc, want in (("trn2", "2", "nki"), ("trn2", "1", "xla"), ("trn1", "1", "xla")):
        monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", tgt)
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
        assert sc._default_kernel() == want, (tgt, lnc)
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    monkeypatch.setenv("KILN_DSA_CP", "1")  # the long-context engines keep the XLA conv (box72's 4096-page faults)
    assert sc._default_kernel() == "xla"
    monkeypatch.delenv("KILN_DSA_CP")
    monkeypatch.delenv("NEURON_PLATFORM_TARGET_OVERRIDE")
    assert sc._default_kernel("trn2") == "nki"  # a capture's target, on a host that is none (compile_farm config.json)
    monkeypatch.setattr("kiln.platform.target", lambda: None)
    assert sc._default_kernel() == "xla"


@needs_nki
@pytest.mark.parametrize("T,C", [(1024, 256), (300, 384), (128, 128)])
def test_short_conv_kernel_equals_causal_conv(monkeypatch, T, C):
    """y = sum_j xe[j : j + T] w[:, j] in fp32: the kernel's grid 1 and grid 2 outputs equal _causal_conv's bits."""
    import nki

    from kiln.kernels import short_conv as sc
    from kiln.models.linear_attn import _causal_conv

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    xe, w = _inputs(T, C)
    want = _causal_conv(xe, w, T)
    args = dict(xe=xe, w=w, ib=torch.eye(128).to(torch.bfloat16), i32=torch.eye(128), rev=sc.REV)
    one = torch.as_tensor(nki.simulate(sc.kiln_short_conv_kernel)(**args))
    two = torch.as_tensor(nki.simulate(sc.kiln_short_conv_kernel[2])(**args))
    assert one.shape == (T, C) and one.dtype == torch.float32
    assert torch.equal(one, want)
    assert torch.equal(two, want)
