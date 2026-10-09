"""kernels/dsa_split.py (KILN_DSA_SPLIT_SELECT): the selection split over the attention group gives kernels/dsa_fused.py's
result. On the CPU: the two emulations composed equal dsa_fused.emulate, and select()'s rows of any row block equal
the whole call's. In nki.simulate: the selection kernel's 0 / 1 pools equal dsa_fused's kernel's own (its dbg output)
and the attention kernel on them equals dsa_fused's kernel, bit for bit.
"""

from __future__ import annotations

import pytest
import torch

from kiln.kernels import dsa_fused, dsa_split


def _inputs(H, C, P, off, Hi=4, D=128, R=512, seed=0):
    g = torch.Generator().manual_seed(seed + off)
    qI = torch.randn(C, Hi, D, generator=g).bfloat16().float()
    w = torch.rand(C, Hi, generator=g)
    pk = torch.randn(P, D, generator=g).bfloat16().float()
    pos = torch.arange(C, dtype=torch.int64) + off
    q_lat = (torch.randn(C, H, R, generator=g) * 0.05).bfloat16().float()
    kc = torch.randn(4 * P, R, generator=g).bfloat16().float()
    return qI, w, pk, pos, q_lat, kc


@pytest.mark.parametrize("off,keep", [(0, 128), (900, 128), (2800, 128), (500, 300)])
def test_split_emulation_is_dsa_fused_emulate(off, keep):
    qI, w, pk, pos, q_lat, kc = _inputs(8, 256, 768, off)
    si, sa = 128 ** -0.5, 512 ** -0.5
    want = dsa_fused.emulate(qI, w, pk, pos, q_lat, kc, keep, si, sa)
    halves = [dsa_split.select(qI[i:i + 128], w[i:i + 128], pk, pos[i:i + 128], pos, kc.shape[0], keep, si)
              for i in (0, 128)]
    sel = torch.cat(halves)
    assert sel.dtype == torch.bfloat16 and sel.shape == (256, 768)
    assert torch.equal(sel, dsa_split.select(qI, w, pk, pos, pos, kc.shape[0], keep, si))
    got = dsa_split.attend(sel, pos, q_lat, kc, sa)
    assert torch.equal(got, want)


class _Model:  # the attention-group surface attend_split reads (DecoderForCausalLM), one rank of a group of 2
    def __init__(self, rank, others):
        self.attn_tp, self._sp_grp = 2, True
        self.sp_grp_index = torch.tensor([rank])
        self._others = others

    def _sp_group_gather(self, x):
        return torch.cat([x, self._others] if int(self.sp_grp_index) == 0 else [self._others, x])


def test_attend_split_assembles_the_rows_in_order(monkeypatch):
    monkeypatch.setattr(dsa_split, "SPLIT", True)
    qI, w, pk, pos, q_lat, kc = _inputs(8, 256, 768, 900)
    si, sa, keep = 128 ** -0.5, 512 ** -0.5, 128
    want = dsa_fused.emulate(qI, w, pk, pos, q_lat, kc, keep, si, sa)
    for rank in (0, 1):
        other = 1 - rank
        rows = slice(other * 128, other * 128 + 128)
        others = dsa_split.select(qI[rows], w[rows], pk, pos[rows], pos, kc.shape[0], keep, si)
        m = _Model(rank, others)
        assert dsa_split.split_takes(m, 256) and not dsa_split.split_takes(m, 200)
        got = dsa_split.attend_split(m, qI, w, pk, pos, q_lat, kc, keep, si, sa)
        assert torch.equal(got, want)


@pytest.mark.parametrize("off,keep", [(0, 128), (900, 128), (2800, 128), (500, 300)])
def test_split_kernels_equal_dsa_fused_kernel(off, keep):
    pytest.importorskip("nki")
    qI, w, pk, pos, q_lat, kc = _inputs(8, 256, 768, off)
    si, sa = 128 ** -0.5, 512 ** -0.5
    o_f, _, dsel, _ = dsa_fused.attend(qI, w, pk, pos, q_lat, kc, keep, si, sa, dbg=1, simulate=True)
    sel = torch.cat([dsa_split.select(qI[i:i + 128], w[i:i + 128], pk, pos[i:i + 128], pos, kc.shape[0], keep, si,
                                      simulate=True) for i in (0, 128)])
    # the selection kernel writes zeros past its variant's pools (its ladder: dsa_split.sel_ladder), which every query
    # of the call is before
    nb = -(-(int(pos.max()) + 1) // 1024)
    P_k = min(768, 256 * min(k for k in dsa_split.sel_ladder(3) if k >= nb))
    assert torch.equal(sel[:, :P_k], dsel[:, :P_k].to(torch.bfloat16)), (off, keep)
    assert not sel[:, P_k:].any()
    o = dsa_split.attend(sel, pos, q_lat, kc, sa, simulate=True)
    print(f"off={off} keep={keep}: equal {torch.equal(o, o_f)}, max |d| {float((o - o_f).abs().max()):.3e}")
    assert torch.equal(o, o_f)


@pytest.mark.parametrize("off,keep", [(900, 128), (2800, 128), (500, 300)])
def test_split_kernels_lnc2_equal_grid1(off, keep, monkeypatch):
    """trn2 at LNC=2: both kernels at grid 2 with their tiles split over the two physical cores give grid 1's bits."""
    pytest.importorskip("nki")
    monkeypatch.setenv("KILN_LNC_SPLIT", "all")
    qI, w, pk, pos, q_lat, kc = _inputs(8, 256, 768, off)
    si, sa = 128 ** -0.5, 512 ** -0.5
    s1 = dsa_split.select(qI, w, pk, pos, pos, kc.shape[0], keep, si, simulate=True)
    s2 = dsa_split.select(qI, w, pk, pos, pos, kc.shape[0], keep, si, simulate=True, grid=2)
    assert torch.equal(s1, s2)
    o1 = dsa_split.attend(s1, pos, q_lat, kc, sa, simulate=True)
    o2 = dsa_split.attend(s1, pos, q_lat, kc, sa, simulate=True, grid=2)
    assert torch.equal(o1, o2)
