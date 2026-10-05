"""kernels/delta_rule.py: the NKI chunked delta rule. Its algorithm (emulate) against the torch path
(linear_attn.chunk_scan) and the token-by-token recurrence on the host; the kernel itself in the NKI
CPU simulator against emulate (Neuron venv only). Device parity and speed: tools/probe_delta_rule.py."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from kiln.kernels import delta_rule as dr
from kiln.models.linear_attn import chunk_scan, recurrent_step


def inputs(T, Hk, Hv, kda, seed=0, lower_bound=-5.0, correlated=False, D=128):
    """Torch-path inputs as linear_attn.mixer builds them: l2-normalised q (scaled) and k, gates in
    [lower_bound, 0) for KDA (the safe gate, lower_bound * sigmoid) or -exp(A) softplus for GDN."""
    g = torch.Generator().manual_seed(seed)
    if correlated:  # nearly parallel keys, beta near 1, slow decay: the regime of trained weights
        base = torch.randn(Hk, D, generator=g)
        k = torch.nn.functional.normalize(base + 0.05 * torch.randn(T, Hk, D, generator=g), dim=-1)
        beta = 0.9 + 0.1 * torch.rand(T, Hv, generator=g)
    else:
        k = torch.nn.functional.normalize(torch.randn(T, Hk, D, generator=g), dim=-1)
        beta = torch.rand(T, Hv, generator=g)
    q = torch.nn.functional.normalize(torch.randn(T, Hk, D, generator=g), dim=-1) * D ** -0.5
    v = torch.randn(T, Hv, D, generator=g)
    if kda:
        gate = lower_bound * torch.sigmoid(torch.randn(T, Hv, D, generator=g) * 2 - (4 if correlated else 0))
    else:
        gate = -torch.exp(torch.randn(Hv, generator=g)) * torch.nn.functional.softplus(torch.randn(T, Hv, generator=g))
        if correlated:
            gate = gate * 0.01
    S0 = torch.randn(Hv, D, D, generator=g) * 0.1
    return q, k, v, gate, beta, S0


def expand(q, k, Hv):
    rep = Hv // q.shape[1]
    return q.repeat_interleave(rep, 1), k.repeat_interleave(rep, 1)


def recurrence(q, k, v, g, beta, S0):
    """Token by token in float64 (the reference of the chunked forms)."""
    S, outs = S0.double().unsqueeze(0), []
    for t in range(q.shape[0]):
        o, S = recurrent_step(*(x[t : t + 1].double() for x in (q, k, v, g, beta)), S)
        outs.append(o)
    return torch.cat(outs), S[0]


@pytest.mark.parametrize("kda,Hk,Hv,T", [(True, 2, 2, 300), (False, 1, 3, 200), (True, 1, 1, 128)])
@pytest.mark.parametrize("correlated", [False, True])
def test_emulation_matches_the_recurrence(kda, Hk, Hv, T, correlated):
    """The kernel's factorisation (16-row reference sub-chunks, the transposed blocked inverse, the
    chunk as an affine map of S) equals the float64 recurrence to float32 rounding, padding a ragged
    last chunk, from a non-zero state; chunk_scan (the torch path) too."""
    q, k, v, gate, beta, S0 = inputs(T, Hk, Hv, kda, correlated=correlated)
    qe, ke = expand(q, k, Hv)
    want, S_want = recurrence(qe, ke, v, gate, beta, S0)
    got, S_got = dr.emulate(q, k, v, gate, beta, S0)
    scale = want.abs().max().item()
    assert (got.double() - want).abs().max().item() < 2e-5 * max(1.0, scale)
    assert (S_got.double() - S_want).abs().max().item() < 2e-5 * max(1.0, S_want.abs().max().item())
    ref, S_ref = chunk_scan(qe, ke, v, gate, beta, S0, 64)
    assert (got - ref).abs().max().item() < 2e-5 * max(1.0, scale)


def test_strong_decay_stays_finite():
    """Every KDA gate at the lower bound (-5 per token, e^-640 over a chunk): the reference
    sub-chunks keep every factor finite, the result is the recurrence's."""
    q, k, v, gate, beta, S0 = inputs(256, 1, 1, True, seed=4)
    gate = torch.full_like(gate, -4.999)
    want, S_want = recurrence(q, k, v, gate, beta, S0)
    got, S_got = dr.emulate(q, k, v, gate, beta, S0)
    assert torch.isfinite(got).all() and torch.isfinite(S_got).all()
    assert (got.double() - want).abs().max().item() < 2e-5 * max(1.0, want.abs().max().item())


def test_constants():
    c = dr.consts_np()
    I, U, BL, BU = (torch.tensor(c[:, i]) for i in (dr.C_I, dr.C_U, dr.C_BL, dr.C_BU))
    assert torch.equal(I, torch.eye(128)) and torch.equal(BU, BL.T)
    assert torch.equal(U, torch.triu(torch.ones(128, 128)))
    for i, s in enumerate(dr.LEVELS):
        lo = torch.tensor(c[:, dr.C_LO + i])
        assert lo.sum().item() == 128 // (2 * s) * s * s


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("kda,Hk,Hv,T", [(True, 2, 2, 256), (True, 3, 3, 128), (False, 1, 2, 256)])
def test_nki_simulator_matches_emulation(kda, Hk, Hv, T, monkeypatch):
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    q, k, v, gate, beta, S0 = inputs(T, Hk, Hv, kda, seed=2, correlated=True)
    args = dr.kernel_inputs(q, k, v, gate, beta, S0, device=torch.device("cpu"))
    o, S = nki.simulate(dr.kernel())(**args)
    want, S_want = dr.emulate(q, k, v, gate, beta, S0)
    o, S = torch.as_tensor(o), torch.as_tensor(S)
    assert (o - want).abs().max().item() < 1e-5 * max(1.0, want.abs().max().item())
    assert (S - S_want).abs().max().item() < 1e-5 * max(1.0, S_want.abs().max().item())
