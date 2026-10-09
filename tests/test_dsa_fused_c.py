"""kernels/dsa_fused_c.py (KILN_DSA_FUSED_CAUSAL): the causal fused DSA prefill kernel against kernels/dsa_fused.py's
kernel, bit for bit, in nki.simulate through both attend()s exactly as the device calls make them, and the variant
one-hot on the CPU.

The exactness claim of the module docstring is what these check: skipping the key blocks past the call's last
position and selecting over only the pools before them changes no output bit, at chunk positions that pick every
variant of the ladder, with P_k above keep (the radix runs) and at or below it (the all-ones selection), with padded
rows, and with fewer than three heads (dsa_fused.MIN_HEADS's padding).
"""

from __future__ import annotations

import pytest
import torch

from kiln.kernels import dsa_fused_c


def test_variant_onehot_picks_the_smallest_covering_count():
    L = 8448  # 8 blocks of 1024 and a 256-key tail: NB 9
    ks = dsa_fused_c.ladder(9)
    assert ks == tuple(range(1, 10))
    for last, want in ((0, 1), (1023, 1), (1024, 2), (5000, 5), (8191, 8), (8192, 9), (8447, 9)):
        pos = torch.tensor([0, 3, last, 0], dtype=torch.int32)
        vt = dsa_fused_c.variant_onehot(pos, L, ks)
        assert vt.dtype == torch.int32 and vt.shape == (1, 9)
        assert vt.sum().item() == 1 and ks[int(vt.argmax())] == want, (last, vt)


def test_ladder_from_the_environment(monkeypatch):
    monkeypatch.setenv("KILN_DSA_FUSED_CAUSAL_LADDER", "2,4,6,8,12")
    ks = dsa_fused_c.ladder(9)
    assert ks == (2, 4, 6, 8, 9)
    assert dsa_fused_c._ladder_of(dsa_fused_c._ladder_bits(ks)) == list(ks)
    for last, want in ((100, 2), (2048, 4), (6143, 6), (6144, 8), (8300, 9)):
        vt = dsa_fused_c.variant_onehot(torch.tensor([last]), 8448, ks)
        assert ks[int(vt.argmax())] == want and vt.sum().item() == 1


def test_cpu_path_is_dsa_fused_emulate():
    from kiln.kernels import dsa_fused

    qI, w, pk, pos, q_lat, kc = _inputs(4, 128, 512, 0, 128)
    a = dsa_fused_c.attend(qI, w, pk, pos, q_lat, kc, 128, 128 ** -0.5, 512 ** -0.5)
    b = dsa_fused.attend(qI, w, pk, pos, q_lat, kc, 128, 128 ** -0.5, 512 ** -0.5)
    assert torch.equal(a, b)


def test_dsa_fused_attend_hands_the_call_over_only_when_asked(monkeypatch):
    """dsa_fused.attend (every model call site) runs dsa_fused_c only with KILN_DSA_FUSED_CAUSAL=1 (grid 1 and 2);
    with it unset nothing changes (the default graphs and their keys)."""
    from kiln.kernels import dsa_fused

    qI, w, pk, pos, q_lat, kc = _inputs(4, 128, 512, 0, 128)
    calls = []
    monkeypatch.setattr(dsa_fused_c, "attend", lambda *a, **k: calls.append(a) or "causal")
    args = (qI, w, pk, pos, q_lat, kc, 128, 128 ** -0.5, 512 ** -0.5)
    monkeypatch.setattr(dsa_fused_c, "CAUSAL", False)
    assert torch.is_tensor(dsa_fused.attend(*args)) and not calls
    monkeypatch.setattr(dsa_fused_c, "CAUSAL", True)
    from kiln import platform

    monkeypatch.setattr(platform, "nki_grid", lambda: 1)
    assert dsa_fused.attend(*args) == "causal" and len(calls) == 1
    monkeypatch.setattr(platform, "nki_grid", lambda: 2)  # trn2 at LNC=2 too (its tiles split over the two cores)
    assert dsa_fused.attend(*args) == "causal" and len(calls) == 2


def _inputs(H, C, P, off, n_real, Hi=4, D=128, R=512, seed=0):
    """A chunk of n_real rows at positions off .. off + n_real - 1, then C - n_real padded rows at position 0."""
    g = torch.Generator().manual_seed(seed + off)
    qI = torch.randn(C, Hi, D, generator=g).bfloat16().float()
    w = torch.rand(C, Hi, generator=g)
    pk = torch.randn(P, D, generator=g).bfloat16().float()
    pos = torch.zeros(C, dtype=torch.int64)
    pos[:n_real] = torch.arange(n_real) + off
    q_lat = (torch.randn(C, H, R, generator=g) * 0.05).bfloat16().float()
    kc = torch.randn(4 * P, R, generator=g).bfloat16().float()
    return qI, w, pk, pos, q_lat, kc


# (heads, rows, real rows, offset, keep): P = 768 pools = 3 blocks of 1024 keys, so the ladder is 1, 2, 3 with P_k 256,
# 512, 768. keep 128: the radix runs in every variant; keep 300: variant 1 is the all-ones selection.
CASES = [
    (8, 256, 256, 0, 128),  # variant 1
    (8, 256, 256, 900, 128),  # last position 1155: variant 2, tiles straddling a block edge
    (8, 256, 256, 2800, 128),  # variant 3 (the whole bucket: dsa_fused's own body)
    (8, 256, 200, 1500, 128),  # padded rows at position 0, variant 2
    (8, 256, 256, 500, 300),  # variant 1 with P_1 = 256 <= keep: the all-ones selection
    (8, 256, 256, 1100, 300),  # variant 2, P_2 = 512 > keep
    (2, 128, 100, 1800, 128),  # fewer than MIN_HEADS heads and a partial tile
]


@pytest.mark.parametrize("H,C,n_real,off,keep", CASES)
def test_causal_kernel_equals_the_static_kernel(H, C, n_real, off, keep):
    pytest.importorskip("nki")
    from kiln.kernels import dsa_fused

    P = 768
    qI, w, pk, pos, q_lat, kc = _inputs(H, C, P, off, n_real)
    D, R = qI.shape[-1], kc.shape[-1]
    args = (qI[:n_real], w[:n_real], pk, pos[:n_real], q_lat[:n_real], kc, keep, D ** -0.5, R ** -0.5)
    if n_real == C:
        args = (qI, w, pk, pos, q_lat, kc, keep, D ** -0.5, R ** -0.5)
    want = dsa_fused.attend(*args, simulate=True)
    got = dsa_fused_c.attend(*args, simulate=True)
    ref = dsa_fused.emulate(*args[:6], keep, D ** -0.5, R ** -0.5)
    assert got.shape == want.shape == ref.shape
    err = (got.float() - ref).abs().max().item()
    print(f"H={H} C={C} real={n_real} off={off} keep={keep}: equal {torch.equal(got, want)}, max |o - emulate| {err:.3e}")
    assert torch.equal(got, want)
    assert err < 1e-2


@pytest.mark.parametrize("off,keep", [(0, 128), (2800, 128), (1100, 300)])
def test_causal_kernel_lnc2_split_equals_grid1(off, keep, monkeypatch):
    """trn2 at LNC=2: the kernel at grid 2 with the query tiles split over the two physical cores (both programs run
    the same 0 / 1 device loops) gives grid 1's bits."""
    pytest.importorskip("nki")
    monkeypatch.setenv("KILN_LNC_SPLIT", "all")
    qI, w, pk, pos, q_lat, kc = _inputs(8, 256, 768, off, 256)
    args = (qI, w, pk, pos, q_lat, kc, keep, 128 ** -0.5, 512 ** -0.5)
    one = dsa_fused_c.attend(*args, simulate=True)
    two = dsa_fused_c.attend(*args, simulate=True, grid=2)
    assert torch.equal(one, two)


def test_default_on_only_for_the_gated_trn2_configuration(monkeypatch):
    """Unset, KILN_DSA_FUSED_CAUSAL turns on only for the gated trn2 G1 8192-row whole-prefill calls: grid 2 on a trn2
    target, 2048 rows, 8448 keys, KILN_PREFILL_WHOLE=1. Everything else keeps dsa_fused. KILN_DSA_SPLIT_SELECT stays
    opt-in, also there (not bit for bit in those graphs)."""
    from kiln import platform
    from kiln.kernels import dsa_split

    monkeypatch.setattr(dsa_fused_c, "CAUSAL", None)
    monkeypatch.setattr(dsa_split, "SPLIT", False)  # its default
    monkeypatch.setenv("KILN_PREFILL_WHOLE", "1")
    monkeypatch.setattr(platform, "target", lambda: "trn2")
    monkeypatch.setattr(platform, "family_of", lambda t: t)
    monkeypatch.setattr(platform, "nki_grid", lambda: 2)
    assert dsa_fused_c.default_on(2048, 8448) and dsa_fused_c.takes(2048, 8448)
    assert dsa_fused_c.ladder(9) == (2, 4, 6, 8, 9)
    for C, L in ((1024, 8448), (2048, 4224), (2048, 3328), (4096, 8448)):
        assert not dsa_fused_c.takes(C, L)
    assert not dsa_fused_c.takes()

    class M:  # the attention-group surface split_takes reads
        _sp_grp, attn_tp = True, 8
        sp_grp_index = torch.tensor([0])

    assert not dsa_split.split_takes(M(), 2048)  # the split selection: off in the gated configuration too
    monkeypatch.setattr(dsa_split, "SPLIT", True)
    assert dsa_split.split_takes(M(), 2048)  # =1 turns it on
    monkeypatch.setattr(dsa_split, "SPLIT", False)
    monkeypatch.setenv("KILN_PREFILL_WHOLE", "0")  # the piecewise prefill graphs: not gated
    assert not dsa_fused_c.takes(2048, 8448)
    monkeypatch.setenv("KILN_PREFILL_WHOLE", "1")
    monkeypatch.setattr(platform, "nki_grid", lambda: 1)  # trn1
    monkeypatch.setattr(platform, "target", lambda: "trn1")
    assert not dsa_fused_c.takes(2048, 8448) and dsa_fused_c.ladder(9) == tuple(range(1, 10))
    monkeypatch.setattr(dsa_fused_c, "CAUSAL", False)  # an explicit 0 wins
    monkeypatch.setattr(platform, "nki_grid", lambda: 2)
    monkeypatch.setattr(platform, "target", lambda: "trn2")
    assert not dsa_fused_c.takes(2048, 8448)
