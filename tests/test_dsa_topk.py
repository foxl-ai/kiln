"""kernels/dsa_topk.py: the exact DSA top-k selection kernel. Its arithmetic (emulate) against the tie
rule spelled out by a stable sort (dsa_select.reference_mask) on the host, the kernel itself in the
NKI CPU simulator against emulate (Neuron venv only), and GLM-5.3-Flash's pooled selection
(glm5_next.block_mask) through it against the torch bisection it replaces. Device parity and speed:
tools/probe_dsa_select.py --methods nki ... and tools/profile_mla.py --select nki.

The tie rule under test: per row the `keep` largest scores; every selected score >= every
unselected one; among the scores equal to the keep-th largest, the lowest indices; with vis_only,
nothing at or below dsa_topk.VISIBLE (an invisible candidate, NEG_INF + score) is selected, so a
row with fewer visible scores than `keep` selects exactly its visible ones.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from kiln.kernels import dsa_topk as dk
from kiln.models import dsa_select as ds
from tests.test_dsa_select import KINDS, _scores

NEG_INF = -1e30


def want(s: torch.Tensor, keep: int, vis_only: bool) -> torch.Tensor:
    sel = ds.reference_mask(s, keep)
    if vis_only:
        sel = sel & (s > dk.VISIBLE)
    return sel


def pooled(rows: int, n: int, g: torch.Generator, visible: int | None = None, zeros: bool = False) -> torch.Tensor:
    """Scores as glm5_next.pooled_selection forms them: index = sum_h w_h relu(q_h . k_p) (w of both
    signs, so negative and exactly-zero scores occur) plus 0 / NEG_INF for invisible pools."""
    q = torch.randn(rows, 8, 16, generator=g)
    k = torch.randn(n, 16, generator=g)
    w = torch.randn(rows, 8, generator=g)
    rel = torch.relu(torch.einsum("rhd,nd->rhn", q, k))
    if zeros:  # dead keys: every head's relu zero
        rel[..., ::5] = 0.0
    index = torch.einsum("rh,rhn->rn", w, rel)
    if visible is not None:
        index = index + torch.where(torch.arange(n) < visible, 0.0, NEG_INF)
    return index


@pytest.mark.parametrize("vis_only", [False, True])
@pytest.mark.parametrize("kind", KINDS)
def test_emulation_follows_the_tie_rule(kind, vis_only):
    g = torch.Generator().manual_seed(200 + KINDS.index(kind))
    for _ in range(10):
        n = int(torch.randint(16, 3000, (1,), generator=g))
        keep = int(torch.randint(1, n, (1,), generator=g))
        s = _scores(kind, 4, n, g)
        got = dk.emulate(s, keep, vis_only) == 0
        assert torch.equal(got, want(s, keep, vis_only)), (kind, n, keep)
        if not vis_only:
            assert (got.sum(-1) == keep).all()


@pytest.mark.parametrize("visible", [None, 3000, 1500, 400, 3])
def test_emulation_on_pooled_scores(visible):
    """GLM-5.3-Flash's decode shape (8 rows, 2112 pools, keep 512) and a prefill chunk's (256 rows),
    with all pools visible or only the first `visible` (fewer than keep: all of them)."""
    g = torch.Generator().manual_seed(7)
    for rows in (8, 256):
        for zeros in (False, True):
            s = pooled(rows, 2112, g, visible, zeros)
            got = dk.emulate(s, 512) == 0
            assert torch.equal(got, want(s, 512, True))
            if visible is not None and visible < 512:
                assert torch.equal(got, (torch.arange(2112) < visible).expand(rows, -1))


def test_pieces():
    assert dk.pieces(8, 2112) == 16 and dk.pieces(256, 2112) == 1 and dk.pieces(3, 100) == 4
    assert dk.pieces(1, 2112) == 64 and dk.pieces(8, 2112, group=False) == 1 and dk.pieces(128, 4096) == 1


def test_dsa_select_nki_is_the_tie_rule():
    """dsa_select.topk_mask(select="nki") (the host path runs emulate) is the module's own rule."""
    g = torch.Generator().manual_seed(3)
    s = torch.randint(-3, 4, (5, 1000), generator=g).float()
    assert torch.equal(ds.topk_mask(s, 300, "nki"), ds.reference_mask(s, 300))
    s = torch.randn(2, 3, 900, generator=g)
    assert torch.equal(ds.topk_mask(s, 100, "nki"), ds.topk_mask(s, 100, "bisect"))


@pytest.mark.parametrize("visible", [None, 6000, 1000])
def test_block_mask_nki_matches_bisect(visible):
    """glm5_next.block_mask through the kernel ("nki") equals dsa_select's bisection ("bisect") and
    the range bisection GLM-5.3-Flash ran until 2026-10-04 ("range") on pooled scores where the
    latter is exact, tail and expansion included, decode and prefill forms."""
    from kiln.models.glm5_next import block_mask

    g = torch.Generator().manual_seed(11)
    kp, L, keep = 4, 8448, 512
    for B, Q in ((8, 1), (1, 64)):
        P = L // kp
        index = pooled(B * Q, P, g).view(B, Q, P)
        n = L if visible is None else visible
        # Visible prefix per query: decode rows each see n tokens; a chunk's queries see n - Q + 1 + q.
        nv = torch.full((B, Q), n) if Q == 1 else (n - Q + 1 + torch.arange(Q)).view(1, Q)
        vis = torch.where(torch.arange(L).view(1, 1, L) < nv.unsqueeze(-1), 0.0, NEG_INF)
        b = block_mask(index, vis, kp, keep, True, select="nki")
        for other in ("range", "bisect"):
            assert torch.equal(block_mask(index, vis, kp, keep, True, select=other), b), other
        # and the pool selection is the tie rule's
        cand = vis.reshape(B, Q, P, kp)[..., kp - 1]
        sc = index + cand
        assert torch.equal(dk.select(sc, keep) == 0, want(sc.reshape(-1, P), keep, True).view(B, Q, P))


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("rows,n,keep,kind", [
    (8, 2112, 512, "pooled"), (8, 2112, 512, "integers"), (8, 2112, 512, "zeros"), (8, 2112, 512, "invisible"),
    (256, 2112, 512, "pooled"), (3, 100, 7, "wide"), (5, 96, 40, "equal"), (8, 2112, 512, "ulps"),
])
def test_nki_simulator_matches_emulation(rows, n, keep, kind, monkeypatch):
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    g = torch.Generator().manual_seed(rows * 31 + n)
    s = pooled(rows, n, g, visible=n // 3 if kind == "pooled" and rows == 256 else None) if kind == "pooled" \
        else _scores(kind, rows, n, g)
    if kind == "wide":  # no subnormals (the device may flush them; the simulator need not)
        s = torch.where(s.abs() < 1.2e-38, torch.zeros_like(s), s)
    for vis_only, kp, tail in ((True, 1, False), (False, 1, False), (True, 4, True), (True, 4, False)):
        args = dk.kernel_inputs(s, keep, vis_only, kp=kp, tail=tail)
        out = torch.as_tensor(nki.simulate(dk.kernel())(**args)).reshape(rows, n * kp)
        assert torch.equal(out, dk.emulate(s, keep, vis_only, kp, tail)), (kind, vis_only, kp, tail)


def test_token_output_is_the_expanded_pool_selection():
    """kp > 1 repeats each pool's value on its tokens; tail adds every non-candidate pool."""
    g = torch.Generator().manual_seed(5)
    s = pooled(4, 600, g, visible=450)
    pool = dk.emulate(s, 100) == 0
    tok = dk.emulate(s, 100, kp=4) == 0
    assert torch.equal(tok, pool.repeat_interleave(4, -1))
    tok = dk.emulate(s, 100, kp=4, tail=True) == 0
    assert torch.equal(tok, (pool | (torch.arange(600) >= 450)).repeat_interleave(4, -1))


def test_default_pooled_selection_is_exact_on_wide_range_scores():
    """Regression (2026-10-04): GLM-5.3-Flash's pooled selection ran block_mask's range bisection
    (32 halvings of [min, max], then called "bisect"), which on scores spread over many orders of
    magnitude keeps nearly every visible pool. The default (KILN_DSA_SELECT unset: "nki") keeps
    exactly `keep`, the tie rule's; the old path does not."""
    from kiln.models.glm5_next import block_mask

    g = torch.Generator().manual_seed(9)
    B, kp, P, keep = 8, 4, 2112, 512
    index = torch.sign(torch.randn(B, 1, P, generator=g)) * torch.exp(torch.randn(B, 1, P, generator=g) * 20)
    index = index.clamp(-1e20, 1e20)  # finite, as pooled scores are
    vis = torch.zeros(B, 1, P * kp)
    assert ds.SELECT == "nki"
    got = block_mask(index, vis, kp, keep, False, select=ds.SELECT) == 0
    want_ = ds.reference_mask(index.view(B, P), keep).repeat_interleave(kp, -1).view(B, 1, P * kp)
    assert torch.equal(got, want_)
    old = block_mask(index, vis, kp, keep, False, select="range") == 0
    assert (old.view(B, P, kp)[..., 0].sum(-1) > keep).all()  # the old path over-selects


def _score_inputs(C, Hi, P, g, visible=None):
    q = torch.randn(C, Hi, 128, generator=g).to(torch.bfloat16)
    pk = torch.randn(P, 128, generator=g).to(torch.bfloat16)
    w = torch.randn(C, Hi, generator=g) * Hi ** -0.5
    cand = torch.zeros(C, P) if visible is None else torch.where(torch.arange(P) < visible, 0.0, NEG_INF).expand(C, P)
    return q, w, pk, cand.contiguous()


def test_emulated_scores_are_the_pooled_indexer_scores():
    """emulate_scores (the score kernel's order: head 0 first) equals pooled_selection's einsum
    form to fp32 rounding, and the selections agree on these well-separated scores."""
    g = torch.Generator().manual_seed(21)
    q, w, pk, cand = _score_inputs(64, 32, 2112, g, visible=1500)
    s = torch.einsum("qhd,pd->qhp", q.float(), pk.float())
    want = torch.einsum("qh,qhp->qp", w, torch.relu(s * 128 ** -0.5)) + cand
    got = dk.emulate_scores(q, w, pk, cand, 128 ** -0.5)
    vis = cand == 0
    assert torch.allclose(got[vis], want[vis], rtol=1e-5, atol=1e-5) and torch.equal(got[~vis], want[~vis])
    assert torch.equal(dk.emulate(got, 512), dk.emulate(want, 512))


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("C,P,visible", [(256, 2112, None), (200, 1024, 700), (64, 2112, 300)])
def test_nki_simulator_score_kernel(C, P, visible, monkeypatch):
    """The fused score + selection kernel in the NKI simulator selects what emulate_scores + emulate
    select (pool level and expanded with the tail)."""
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    g = torch.Generator().manual_seed(C + P)
    q, w, pk, cand = _score_inputs(C, 32, P, g, visible)
    for kp, tail in ((1, False), (4, True)):
        args = dict(qT=q.permute(1, 2, 0).contiguous(), w=w, pkT=pk.t().contiguous(), cand=cand, keep=512,
                    scale=128 ** -0.5, nbits=P.bit_length(), kp=kp, tail=int(tail), tiles=dk.TILES, rev=dk.REV)
        out = torch.as_tensor(nki.simulate(dk.score_kernel())(**args)).reshape(C, P * kp)
        want = dk.emulate(dk.emulate_scores(q, w, pk, cand, 128 ** -0.5), 512, True, kp, tail)
        assert torch.equal(out, want), (kp, tail, (out != want).sum().item())
