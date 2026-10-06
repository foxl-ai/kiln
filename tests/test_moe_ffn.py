"""The fused small-call MoE FFN kernel (kiln/kernels/moe_ffn.py): its routing against DecoderForCausalLM._route's
sigmoid / noaux_tc arithmetic, its emulation against the separate router + per-pair experts + shared expert, and (where the
NKI package is installed) the kernel under the NKI CPU simulator against its emulation."""

import importlib.util

import pytest
import torch

from kiln.kernels import moe_dedupe as mdd
from kiln.kernels import moe_ffn as mf
from tests.test_moe_dedupe import experts128

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")


def case(T, E=40, H=4096, seed=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, H, generator=g).bfloat16()
    w_router = (torch.randn(E, H, generator=g) * H ** -0.5).bfloat16()
    bias = torch.randn(E, generator=g) * 0.05
    blob = mdd.pack(*experts128(e=E + 1, h=H))  # the last one is the shared expert
    return x, w_router, bias, blob


def test_routing_matches_the_graph_router():
    """Top-8 of sigmoid(logits) + bias, weights the chosen sigmoids normalised and scaled (DecoderForCausalLM._route,
    router_scoring "sigmoid", norm_topk_prob, routed_scaling_factor 2.5): the same experts and bf16 weights."""
    x, w, bias, _ = case(12)
    topv, topi = mf.route(x, w, bias, 8, 2.5)
    logits = x.float() @ w.float().t()
    scores = logits.sigmoid()
    _, ti = torch.topk(scores + bias, 8, dim=-1)
    tv = torch.gather(scores, 1, ti)
    tv = (tv / (tv.sum(-1, keepdim=True) + 1e-20) * 2.5).bfloat16()
    assert torch.equal(topi.sort(-1).values, ti.sort(-1).values)
    assert torch.equal(torch.gather(topv, 1, topi.argsort(-1)), torch.gather(tv, 1, ti.argsort(-1)))


def test_emulation_is_the_separate_router_experts_and_shared_expert():
    x, w, bias, blob = case(5)
    E = w.shape[0]
    got = mf.emulate(x, w, bias, blob, 8, 2.5, True, act=1, limit=10.0).float()
    topv, topi = mf.route(x, w, bias, 8, 2.5)
    routed = mdd.emulate_pairs(x, topv, topi, blob, 1, 10.0).float()
    shared = mdd.emulate_pairs(x, torch.ones(5, 1).bfloat16(), torch.full((5, 1), E), blob, 1, 10.0).float()
    assert (got - (routed + shared)).abs().max() <= 1e-2 * got.abs().max()


@needs_nki
@pytest.mark.parametrize("T,shared", [(1, True), (4, True), (15, True), (4, False)])
def test_nki_simulator_matches_emulation(T, shared, monkeypatch):
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    x, w, bias, blob = case(T, E=40 if shared else 41)
    if not shared:
        blob = blob[:41]
    args = mf.kernel_inputs(x, mf.router_t(w), bias, blob, 8, 2.5, shared, act=1, limit=10.0)
    st = {k: args.pop(k) for k in ("K", "shared", "scale", "group", "ring", "act", "limit", "rev")}
    got = mdd.from_pairs(torch.as_tensor(nki.simulate(mf.kiln_moe_ffn_pairs_v1)(**args, **st)), torch.float32)
    want = mf.emulate(x, w, bias, blob, 8, 2.5, shared, act=1, limit=10.0).float()
    assert (got - want).abs().max() <= 0.01 * want.abs().max()
