"""kernels/segmented_attn.py: nkilib's segmented prefill attention (vendored, kernels/nkilib_vendor) over Kiln's paged
KV layout, in the NKI simulator, against the decoder's own chunk-form arithmetic (fp32 scores over the page bucket,
j <= position, softmax, P V)."""

from __future__ import annotations

import importlib.util

import pytest
import torch

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")


def _reference(q, kc, vc, table, prior, ps, scale):
    C, nh, D = q.shape
    nkv = kc.shape[1]
    G = nh // nkv
    k = kc.view(-1, ps, nkv, D)[table].flatten(0, 1).float()
    v = vc.view(-1, ps, nkv, D)[table].flatten(0, 1).float()
    L = k.shape[0]
    pos = prior + torch.arange(C)
    vis = torch.arange(L).unsqueeze(0) <= pos.unsqueeze(1)
    s = torch.einsum("chgd,lhd->hgcl", q.float().view(C, nkv, G, D), k) * scale
    s = s.masked_fill(~vis, float("-inf"))
    return torch.einsum("hgcl,lhd->chgd", torch.softmax(s, -1), v).reshape(C, nh, D)


def _case(C, prior, nh=4, nkv=2, D=128, ps=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    P = -(-(prior + C) // ps)
    pages = P + 3
    table = torch.randperm(pages, generator=g)[:P]
    kc = torch.randn(pages * ps, nkv, D, generator=g).to(torch.bfloat16)
    vc = torch.randn(pages * ps, nkv, D, generator=g).to(torch.bfloat16)
    q = torch.randn(C, nh, D, generator=g).to(torch.bfloat16)
    return q, kc, vc, table


def test_eligible_and_mode(monkeypatch):
    from kiln.config import AttnSpec
    from kiln.kernels import segmented_attn as sa

    class L:  # the attributes eligible() reads off a DecoderLayer
        def __init__(self, **kw):
            self.spec = AttnSpec(8, 2, 128, 128, 128, 1e6, **kw)
            self.nh, self.nkv, self.sink, self.rel_proj = 8, 2, None, None

    monkeypatch.setattr(sa, "ENABLED", True)
    assert sa.eligible(L(), 4096, 32, torch.bfloat16, False)
    assert not sa.eligible(L(), 4096, 32, torch.bfloat16, True)  # fp8 KV
    assert not sa.eligible(L(), 3072, 32, torch.bfloat16, False)  # not a segment size
    assert not sa.eligible(L(window=4096), 4096, 32, torch.bfloat16, False)
    assert not sa.eligible(L(), 4096, 48, torch.bfloat16, False)  # page size not a power of two
    monkeypatch.setattr(sa, "ENABLED", False)
    assert not sa.eligible(L(), 4096, 32, torch.bfloat16, False)


@needs_nki
@pytest.mark.parametrize("prior", [0, 512, 544, 1536])
@pytest.mark.parametrize("grid", [1, 2])
def test_simulator_kiln_layout(prior, grid, monkeypatch):
    """The vendored kernel with kv_bsh=True reads Kiln's [pages, ps, nkv, D] cache: no prior, whole prior segments
    and a partial one (prior = 544, a page past a segment), at grid 1 (trn1) and grid 2 (trn2 LNC=2)."""
    import nki

    from kiln.kernels import segmented_attn as sa

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2" if grid == 2 else "trn1")
    C, ps = 512, 32
    q, kc, vc, table = _case(C, prior, ps=ps)
    scale = 128 ** -0.5
    k = sa._kernel()
    out = nki.simulate(k[grid] if grid > 1 else k)(
        q=(q.float() * scale).to(torch.bfloat16).permute(1, 0, 2).contiguous(), k_cache=kc.view(-1, ps, 2, 128),
        v_cache=vc.view(-1, ps, 2, 128), block_tables=table.view(1, -1).to(torch.int32),
        prior_tokens=torch.tensor([[prior]], dtype=torch.int32), block_size=ps, prior_seg_size=C, scale=1.0,
        tp_q=True, tp_out=False, num_q_heads=4, kv_bsh=True, rev=sa.REV)
    got = torch.as_tensor(out).float().permute(1, 0, 2)
    want = _reference(q, kc, vc, table, prior, ps, scale)
    err = ((got - want).abs().max() / want.abs().max()).item()
    assert err < 2e-2, err


@needs_nki
def test_simulator_layouts_agree(monkeypatch):
    """kv_bsh=True over Kiln's cache equals the unmodified layout path over the same values stored head-major."""
    import nki

    from kiln.kernels import segmented_attn as sa

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    C, ps, prior = 512, 32, 512
    q, kc, vc, table = _case(C, prior, ps=ps, seed=3)
    qh = q.permute(1, 0, 2).contiguous()
    common = dict(q=qh, block_tables=table.view(1, -1).to(torch.int32),
                  prior_tokens=torch.tensor([[prior]], dtype=torch.int32), block_size=ps, prior_seg_size=C,
                  scale=1.0, tp_q=True, tp_out=False, num_q_heads=4, rev=sa.REV)
    a = nki.simulate(sa._kernel())(k_cache=kc.view(-1, ps, 2, 128), v_cache=vc.view(-1, ps, 2, 128), kv_bsh=True,
                                   **common)
    hm = lambda c: c.view(-1, ps, 2, 128).permute(0, 2, 1, 3).contiguous()  # noqa: E731  [pages, nkv, ps, D]
    b = nki.simulate(sa._kernel())(k_cache=hm(kc), v_cache=hm(vc), kv_bsh=False, **common)
    assert torch.equal(torch.as_tensor(a), torch.as_tensor(b))


def test_gen3_view_keeps_trn2_keys():
    """The NeuronCore-v2 fallbacks are tagged so that trn2's REV hashes the source as before them: every
    "kiln-gen2" line dropped and every "kiln-gen3:" line restored; trn1 hashes the whole text."""
    import os

    from kiln.kernels import segmented_attn as sa

    for f in sa._VENDOR:
        with open(os.path.join(sa._HERE, "nkilib_vendor", f), encoding="utf-8") as fh:
            v = sa.gen3_view(fh.read())
        assert "kiln-gen2" not in v and "kiln-gen3:" not in v and "_kiln_gen2_range_mask" not in v, f
    assert sa.REV != sa.REV_GEN2
    assert sa.gen3_view("a\n  b  # kiln-gen3: c\nd  # kiln-gen2\n") == "a\n  c\n"
