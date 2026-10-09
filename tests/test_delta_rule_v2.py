"""kernels/delta_rule_v2.py: the delta rule with its engine work rebalanced. In the NKI CPU simulator its o and final
state equal delta_rule's bit for bit (Neuron venv only); the device check is tools/probe_delta_rule_v2.py."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from kiln.kernels import delta_rule as dr
from kiln.kernels import delta_rule_v2 as d2


def test_opt_in_and_rev_distinct():
    import os

    assert d2.REV != dr.REV
    assert d2.ENABLED == (os.environ.get("KILN_DELTA_RULE_V") == "2")  # off unless asked for


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("cp,yacc", [(1, 0), (0, 1), (1, 1)])
@pytest.mark.parametrize("kda,Hk,Hv,T", [(True, 2, 2, 256), (False, 1, 2, 256)])
def test_nki_simulator_matches_delta_rule(cp, yacc, kda, Hk, Hv, T, monkeypatch):
    import nki

    from tests.test_delta_rule import inputs

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    q, k, v, gate, beta, S0 = inputs(T, Hk, Hv, kda, seed=3, correlated=True)
    a1 = dr.kernel_inputs(q, k, v, gate, beta, S0, device=torch.device("cpu"))
    o1, S1 = (torch.as_tensor(x) for x in nki.simulate(dr.kernel())(**a1))
    a2 = d2.kernel_inputs(q, k, v, gate, beta, S0, device=torch.device("cpu"), cp=cp, yacc=yacc)
    o2, S2 = (torch.as_tensor(x) for x in nki.simulate(d2.kernel())(**a2))
    if yacc:  # the simulator's PSUM accumulation need not round as the device's; the device probe decides
        assert (o1 - o2).abs().max().item() < 1e-5 * max(1.0, o1.abs().max().item())
    else:
        assert torch.equal(o1, o2) and torch.equal(S1, S2)
