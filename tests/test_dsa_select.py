"""Exact top-k selection without torch.topk (models/dsa_select.py), against a stable sort.

The tie rule under test: exactly `keep` positions per row; every selected score >= every
unselected one; among the scores equal to the keep-th largest, the lowest indices. torch.topk picks
the same SET whenever the keep-th and (keep + 1)-th scores differ.
"""

import math

import pytest
import torch

from kiln.models import dsa_select as ds


def _scores(kind: str, rows: int, n: int, g: torch.Generator) -> torch.Tensor:
    if kind == "randn":
        return torch.randn(rows, n, generator=g)
    if kind == "integers":  # thousands tied at the threshold
        return torch.randint(-3, 4, (rows, n), generator=g).float()
    if kind == "zeros":  # a relu-sum indexer with dead keys: exact 0s (both signs) beside tiny values
        s = torch.relu(torch.randn(rows, n, generator=g)) * (torch.rand(rows, n, generator=g) - 0.5) * 1e-6
        s[:, ::3] = 0.0
        s[:, 1::5] = -0.0
        return s
    if kind == "wide":  # magnitudes from underflow to overflow, both signs, infinities
        return torch.sign(torch.randn(rows, n, generator=g)) * torch.exp(torch.randn(rows, n, generator=g) * 30)
    if kind == "invisible":  # a DSA row: visible scores, the rest at NEG_INF + score
        s = torch.randn(rows, n, generator=g) * 10
        s[:, n // 3 :] += -1e30
        return s
    if kind == "ulps":  # distinct scores one ulp apart at the threshold
        s = torch.randn(rows, n, generator=g)
        s[:, : n // 2] = 1.0 + torch.randint(0, 4, (rows, n // 2), generator=g).float() * 2**-23
        return s
    if kind == "equal":
        return torch.full((rows, n), 0.37)
    raise ValueError(kind)


KINDS = ["randn", "integers", "zeros", "wide", "invisible", "ulps", "equal"]


@pytest.mark.parametrize("select", ["bisect", "radix"])
@pytest.mark.parametrize("ties", ["auto", "block", "cumsum", "index"])
@pytest.mark.parametrize("kind", KINDS)
def test_selection_follows_the_tie_rule(select, ties, kind):
    g = torch.Generator().manual_seed(KINDS.index(kind))
    for _ in range(12):
        n = int(torch.randint(16, 3000, (1,), generator=g))
        keep = int(torch.randint(1, n, (1,), generator=g))
        s = _scores(kind, 4, n, g)
        got = ds.topk_mask(s, keep, select, ties)
        assert torch.equal(got, ds.reference_mask(s, keep)), (kind, n, keep)
        assert (got.sum(-1) == keep).all()


@pytest.mark.parametrize("kind", KINDS)
def test_bisection_ends_on_adjacent_floats(kind):
    """lo is the keep-th largest score and hi the next float up (or lo == hi == max): exact, not
    exact up to a resolution, in every kind of row and within BISECT_ROUNDS."""
    g = torch.Generator().manual_seed(100 + KINDS.index(kind))
    for _ in range(8):
        n = int(torch.randint(16, 3000, (1,), generator=g))
        keep = int(torch.randint(1, n, (1,), generator=g))
        s = _scores(kind, 4, n, g)
        lo, hi = ds.kth_value(s, keep)
        t = torch.sort(s, dim=-1, descending=True).values[:, keep - 1 : keep]
        assert torch.equal(lo, t)
        assert ((lo == hi) | (torch.nextafter(lo, torch.full_like(lo, math.inf)) == hi)).all()


@pytest.mark.parametrize("shape", [(1, 300), (3, 4096), (2, 3, 2080), (128, 1000)])
def test_grouped_counts_and_odd_shapes(shape, monkeypatch):
    """Rows cut into pieces for the count (any row count, a length no power of two divides
    evenly) and leading batch dims: the same selection with grouping off."""
    g = torch.Generator().manual_seed(11)
    s = torch.randint(-5, 6, shape, generator=g).float() + torch.randn(shape, generator=g) * 0.01
    keep = shape[-1] // 3
    want = ds.reference_mask(s.reshape(-1, shape[-1]), keep).reshape(shape)
    assert torch.equal(ds.topk_mask(s, keep), want)
    monkeypatch.setattr(ds, "GROUP", False)
    assert torch.equal(ds.topk_mask(s, keep), want)


def test_same_set_as_torch_topk_without_ties():
    g = torch.Generator().manual_seed(7)
    s = torch.randn(64, 8192, generator=g)
    want = torch.zeros_like(s, dtype=torch.bool).scatter(-1, torch.topk(s, 2048, dim=-1).indices, True)
    assert torch.equal(ds.topk_mask(s, 2048), want)


def test_fewer_scores_than_keep_selects_all():
    s = torch.randn(3, 10)
    assert ds.topk_mask(s, 10).all() and ds.topk_mask(s, 50).all()


def test_dsa_layers_use_the_mask_selection():
    """mla._select in mask mode returns the additive form of the same selection, and the gather
    path's indices are its positions in order."""
    from kiln.models import mla

    g = torch.Generator().manual_seed(3)
    B, Q, L, k = 2, 3, 200, 16
    index = torch.randn(B, Q, L, generator=g)
    sel = ds.topk_mask(index, k)
    idx = mla.mask_indices(sel, k)
    assert torch.equal(idx, torch.stack([torch.nonzero(r).squeeze(-1) for r in sel.reshape(-1, L)]).view(B, Q, k))
