"""kernels/kda_decode.py: the KDA decode step kernel's emulation against models/linear_attn.py's decode form, and the
kernel in the NKI simulator against its emulation (rows that start from a zero state, a shared padding write row)."""

from __future__ import annotations

import importlib.util

import pytest
import torch


def inputs(B=6, R=9, H=4, D=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    pool = torch.randn(R, H, D, D, generator=g) * 0.05
    slot = torch.randperm(R - 1, generator=g)[:B]
    keep = torch.ones(B, dtype=torch.bool)
    slot[-2:], keep[-2:] = R - 1, False  # two padding rows: the scratch row, from zero
    q = torch.nn.functional.normalize(torch.randn(B, H, D, generator=g), dim=-1) * D ** -0.5
    k = torch.nn.functional.normalize(torch.randn(B, H, D, generator=g), dim=-1)
    v = torch.randn(B, H, D, generator=g)
    gl = -5.0 * torch.sigmoid(torch.randn(B, H, D, generator=g))
    beta = torch.sigmoid(torch.randn(B, H, generator=g))
    return pool, slot, keep, q, k, v, gl, beta


def test_emulation_is_the_decode_form():
    from kiln.kernels.kda_decode import emulate
    from kiln.models.linear_attn import _write_rows, recurrent_step

    pool, slot, keep, q, k, v, g, beta = inputs()
    R = pool.shape[0]
    want_pool = pool.clone()
    S = want_pool[slot]
    S = torch.where(keep.view(-1, 1, 1, 1), S, torch.zeros_like(S))
    o_want, S = recurrent_step(q, k, v, g, beta, S)
    _write_rows(want_pool, slot, S)
    rd = torch.where(keep, slot, torch.full_like(slot, R))
    o, got_pool = emulate(pool, torch.stack([rd, slot]), q, k, v, g, beta)
    real = keep.nonzero().flatten()
    assert torch.equal(o[real], o_want[real])
    assert torch.equal(got_pool[slot[real]], want_pool[slot[real]])
    other = torch.ones(R, dtype=torch.bool)
    other[slot] = False
    assert torch.equal(got_pool[other], pool[other])


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
def test_nki_simulator_matches_emulation(monkeypatch):
    import nki

    from kiln.kernels import kda_decode as kd

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    pool, slot, keep, q, k, v, g, beta = inputs(H=8)
    R = pool.shape[0]
    rd = torch.where(keep, slot, torch.full_like(slot, R))
    slots = torch.stack([rd, slot]).to(torch.int32)
    o_want, pool_want = kd.emulate(pool, slots, q, k, v, g, beta)
    sim_pool = pool.clone()
    o, pool_out = nki.simulate(kd.kiln_kda_decode_kernel)(pool=sim_pool, slots=slots, q=q, k=k, v=v, g=g, beta=beta,
                                                          ident=torch.eye(128), rev=kd.REV)
    o, pool_out = torch.as_tensor(o), torch.as_tensor(pool_out)
    real = keep.nonzero().flatten()
    assert (o[real] - o_want[real]).abs().max().item() < 1e-5
    assert (pool_out[slot[real]] - pool_want[slot[real]]).abs().max().item() < 1e-5
