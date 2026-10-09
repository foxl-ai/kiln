"""The trn2 work split of Kiln's NKI kernels (feat/trn2-max) in the NKI simulator: each kernel launched with
grid 2, as on trn2 at LNC=2 (kiln/platform.py nki_grid), where the two programs are the two physical cores of
the logical core and each runs half of the work (nki.simulate(kernel[2]) runs both programs, their core
barrier and sendrecv: nki/_backends/simulator/lnc_ops.py). Grid 2 must give what grid 1 gives: the same bits
for the kernels whose programs write disjoint outputs (moe_prefill, delta_rule, dsa_topk), and for the decode
MoE kernel (moe_dedupe: each program sums its blocks, then the two fp32 partials are added) the same within
the per-pair emulation's tolerance and, with the same inputs, the same bits on every run."""

from __future__ import annotations

import importlib.util

import pytest
import torch

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")


def test_lnc_split_switch(monkeypatch):
    """KILN_LNC_SPLIT: the proven kernels by default (delta_rule, dsa_topk; at LNC=2 also moe_dedupe, kda_decode,
    dsa_decode, dsa_fused, moe_ep), all with "all", none with 0, a named subset, and an unknown name refused."""
    from kiln import platform

    monkeypatch.delenv("KILN_LNC_SPLIT", raising=False)
    monkeypatch.setattr(platform, "_LNC", 1)  # trn1 / a host: the old default
    assert [platform.lnc_split(k) for k in platform.LNC_SPLIT_KERNELS] == [False, False, True, True, False, False, False, False]
    monkeypatch.setattr(platform, "_LNC", 2)  # trn2 at LNC=2: + the decode splits measured there
    assert [platform.lnc_split(k) for k in platform.LNC_SPLIT_KERNELS] == [False, True, True, True, True, True, True, True]
    monkeypatch.setenv("KILN_LNC_SPLIT", "all")
    assert all(platform.lnc_split(k) for k in platform.LNC_SPLIT_KERNELS)
    monkeypatch.setenv("KILN_LNC_SPLIT", "0")
    assert not any(platform.lnc_split(k) for k in platform.LNC_SPLIT_KERNELS)
    monkeypatch.setenv("KILN_LNC_SPLIT", "moe_prefill,dsa_topk")
    assert [platform.lnc_split(k) for k in platform.LNC_SPLIT_KERNELS] == [True, False, False, True, False, False, False, False]
    monkeypatch.setenv("KILN_LNC_SPLIT", "moe")
    with pytest.raises(ValueError):
        platform.lnc_split("moe_prefill")


def _sim(kernel, grid, **args):
    import nki

    return nki.simulate(kernel[grid] if grid > 1 else kernel)(**args)


@needs_nki
def test_moe_prefill_halves_are_the_whole_kernel(monkeypatch):
    """Lane tiles and combine token tiles split between the programs, Y in shared HBM behind a core
    barrier: the output equals grid 1's bit for bit (GLM-5.3-Flash's loaded expert layout, both block
    sizes)."""
    from kiln.kernels import moe_dedupe as mdd
    from kiln.kernels import moe_prefill as mp
    from tests.test_moe_prefill import checkpoint_experts, routing

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setenv("KILN_LNC_SPLIT", "all")  # the split itself, whatever the default
    w_gu, s_gu, w_down, s_down = checkpoint_experts()[0]
    blob = mdd.pack(w_gu, s_gu, w_down, s_down)
    down = mp.down_factors(blob, 1024)
    x, topv, topi = routing(256, skew=True)
    # skp 4: the tiles past the first half in device-loop segments, each program its half of every segment
    for B, skp in ((64, 0), (128, 0), (64, 4)):
        args = mp.kernel_inputs(x, topv, topi, blob, 1, 10.0, B=B, dq=False, down=down, skp=skp)
        one = torch.as_tensor(_sim(mp.kernel(), 1, **args))
        two = torch.as_tensor(_sim(mp.kernel(), 2, **args))
        assert torch.equal(one, two), (B, skp, (one != two).sum().item())
        want = mp.emulate(x, topv, topi, blob, 1, 10.0).float()
        assert (two.float() - want).abs().max() <= 0.01 * want.abs().max()


@needs_nki
def test_moe_dedupe_halves_sum_to_the_whole_kernel(monkeypatch):
    """Blocks of lanes split between the programs and the fp32 partials exchanged by halves of H: within
    the emulation's tolerance of grid 1, the same bits on a second run, and grid 1 itself unchanged."""
    from kiln.kernels import moe_dedupe as mdd
    from tests.test_moe_dedupe import experts128

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setenv("KILN_LNC_SPLIT", "all")  # the split itself, whatever the default
    Hd = 1024
    ws = experts128(e=128, h=Hd)
    blob = mdd.pack(*ws)
    g = torch.Generator().manual_seed(4)
    for T in (16, 32):
        x = torch.randn(T, Hd, generator=g).bfloat16()
        topi = torch.stack([torch.randperm(128, generator=g)[:8] for _ in range(T)])
        topv = (torch.rand(T, 8, generator=g) + 0.1).bfloat16()
        args = mdd.kernel_inputs(x, topv, topi, blob, lanes=2, act=1, limit=10.0)
        S, L, BL = args["slots"], args["lanes"], args["block"]
        assert S * L // BL >= 2, "the case must have two blocks to split"
        one = torch.as_tensor(_sim(mdd.kernel(), 1, **args)).float()
        two = torch.as_tensor(_sim(mdd.kernel(), 2, **args)).float()
        again = torch.as_tensor(_sim(mdd.kernel(), 2, **args)).float()
        want = mdd.emulate(x, topv, topi, blob, 1, 10.0).float()
        assert torch.equal(two, again)
        assert (one - want).abs().max() <= 0.01 * want.abs().max()
        assert (two - want).abs().max() <= 0.01 * want.abs().max()
        # bf16 outputs of fp32 sums taken in another order: at most one bf16 step apart, plus an fp32-sized
        # absolute slack for outputs that are near zero by cancellation
        tol = one.abs() * 2.0 ** -7 + one.abs().max() * 2.0 ** -20
        assert ((two - one).abs() <= tol).all(), (two - one).abs().max()


@needs_nki
def test_moe_dedupe_v9_halves_sum_to_the_whole_kernel(monkeypatch):
    """The one-call kernel for 128 < T <= 256 (kiln_moe_dedupe_v9): its two token tiles exchanged per tile between the
    programs, within the emulation's tolerance and one bf16 step of grid 1, the same bits on a second run."""
    from kiln.kernels import moe_dedupe as mdd
    from tests.test_moe_dedupe import experts128

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setenv("KILN_LNC_SPLIT", "all")
    Hd = 1024
    ws = experts128(e=128, h=Hd)
    blob = mdd.pack(*ws)
    g = torch.Generator().manual_seed(5)
    T = 150
    x = torch.randn(T, Hd, generator=g).bfloat16()
    topi = torch.stack([torch.randperm(16, generator=g)[:8] for _ in range(T)]) * 8
    topv = (torch.rand(T, 8, generator=g) + 0.1).bfloat16()
    args = mdd.kernel_inputs(x, topv, topi, blob, lanes=8, act=1, limit=10.0)
    assert args["rev"] == mdd.REV9 and args["spl"] == 1
    one = torch.as_tensor(_sim(mdd.kernel9(), 1, **args)).float()
    two = torch.as_tensor(_sim(mdd.kernel9(), 2, **args)).float()
    again = torch.as_tensor(_sim(mdd.kernel9(), 2, **args)).float()
    want = mdd.emulate(x, topv, topi, blob, 1, 10.0).float()
    assert torch.equal(two, again)
    assert (one - want).abs().max() <= 0.01 * want.abs().max()
    assert (two - want).abs().max() <= 0.01 * want.abs().max()
    tol = one.abs() * 2.0 ** -7 + one.abs().max() * 2.0 ** -20
    assert ((two - one).abs() <= tol).all(), (two - one).abs().max()


@needs_nki
def test_delta_rule_heads_split_is_the_whole_kernel(monkeypatch):
    """Each program runs its own v heads: o and the final states equal grid 1's bit for bit (KDA, GDN with
    grouped k heads)."""
    from kiln.kernels import delta_rule as dr
    from tests.test_delta_rule import inputs

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setenv("KILN_LNC_SPLIT", "all")  # the split itself, whatever the default
    for kda, Hk, Hv in ((True, 4, 4), (False, 1, 2)):
        q, k, v, gate, beta, S0 = inputs(256, Hk, Hv, kda, seed=3, correlated=True)
        args = dr.kernel_inputs(q, k, v, gate, beta, S0, device=torch.device("cpu"))
        o1, S1 = (torch.as_tensor(t) for t in _sim(dr.kernel(), 1, **args))
        o2, S2 = (torch.as_tensor(t) for t in _sim(dr.kernel(), 2, **args))
        assert torch.equal(o1, o2) and torch.equal(S1, S2), (kda, Hk, Hv)


@needs_nki
def test_dsa_topk_row_split_is_the_whole_kernel(monkeypatch):
    """Groups of row tiles alternate between the programs: the selection and the fused score + selection
    equal grid 1's and the emulation's exactly."""
    from kiln.kernels import dsa_topk as dk
    from tests.test_dsa_topk import _score_inputs, pooled

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setenv("KILN_LNC_SPLIT", "all")  # the split itself, whatever the default
    g = torch.Generator().manual_seed(9)
    s = pooled(512, 600, g, visible=500)  # 4 row tiles: two groups of TILES = 2
    args = dk.kernel_inputs(s, 100, True)
    one = torch.as_tensor(_sim(dk.kernel(), 1, **args))
    two = torch.as_tensor(_sim(dk.kernel(), 2, **args))
    assert torch.equal(one, two)
    assert torch.equal(two.reshape(512, 600), dk.emulate(s, 100, True))
    C, P = 512, 640
    q, w, pk, cand = _score_inputs(C, 4, P, g, visible=600)
    sargs = dict(qT=q.permute(1, 2, 0).contiguous(), w=w, pkT=pk.t().contiguous(), cand=cand, keep=128,
                 scale=128 ** -0.5, nbits=P.bit_length(), kp=1, tail=0, tiles=dk.TILES, rev=dk.REV)
    one = torch.as_tensor(_sim(dk.score_kernel(), 1, **sargs))
    two = torch.as_tensor(_sim(dk.score_kernel(), 2, **sargs))
    assert torch.equal(one, two)


@needs_nki
def test_moe_ep_split_is_the_whole_kernel(monkeypatch):
    """kernels/moe_ep.py split over the two programs (KILN_LNC_SPLIT=moe_ep): gate_up over each program's half of the
    I-chunks, the a^T halves swapped by sendrecv, the down projection over each program's half of the output columns,
    each program read-modify-writing only its own columns of out. The prefill kernel (first passes and overflow
    passes, both tile-scale forms) and the decode kernel v2 (small passes and a dequantize-first pass) give grid 1's
    output bit for bit, and grid 1 matches the host emulation. Unsplit at grid 2 (both programs run the whole kernel,
    program 0 alone writes: grid 1 does not compile at LNC=2) gives grid 1's output too. At grid 2 both kernels return
    128 dummy rows past C (the empty lanes' scatter destinations, moe_ep._dummy_lanes), which nobody reads."""
    from kiln.kernels import moe_ep
    from tests.test_moe_ep import _experts

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    H, I, E, K, El, rank = 1024, 256, 12, 4, 3, 1  # M = 2 I-chunks, H / 512 = 2 output chunks: one each per program
    owner = torch.arange(E) // El
    lmap = moe_ep.local_map(owner, rank)
    g = torch.Generator().manual_seed(5)
    for block in (False, True):
        ws = _experts(El, H, I, seed=3, block=block)
        blob = moe_ep.pack(*ws, tiles=block)
        for C, skew in ((128, False), (256, True)):
            x = torch.randn(C, H, generator=g).bfloat16()
            if skew:  # most pairs on this rank's first expert: overflow passes past its first LW lanes
                rest = torch.tensor([e for e in range(E) if e != El * rank])
                topi = torch.stack([torch.cat([torch.tensor([El * rank]), rest[torch.randperm(E - 1, generator=g)[:K - 1]]])
                                    for _ in range(C)])
            else:
                topi = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(C)])
            topv = (torch.rand(C, K, generator=g) + 0.1).bfloat16()
            args = moe_ep.kernel_inputs(x, topv, topi, blob, lmap, 1, 10.0)
            one = torch.as_tensor(_sim(moe_ep.kiln_moe_ep_kernel, 1, **args))
            two = torch.as_tensor(_sim(moe_ep.kiln_moe_ep_kernel, 2, **dict(args, spl=1)))
            assert two.shape[0] == C + 128 and torch.equal(one, two[:C]), ("prefill", block, C)
            both = torch.as_tensor(_sim(moe_ep.kiln_moe_ep_kernel, 2, **args))  # unsplit: program 0 alone writes
            assert torch.equal(one, both[:C]), ("prefill unsplit grid 2", block, C)
            want = moe_ep.emulate(x, topv, topi, lmap, *ws, act=1, lim=10.0, small=False).float()
            assert (one.float() - want).abs().max() <= 0.01 * want.abs().max()
        # decode v2: C = 128 rows, experts with few pairs (small passes) and one with many (a dequantize-first pass)
        x = torch.randn(128, H, generator=g).bfloat16()
        rest = torch.tensor([e for e in range(E) if e != El * rank])
        topi = torch.stack([torch.cat([torch.tensor([El * rank]), rest[torch.randperm(E - 1, generator=g)[:K - 1]]])
                            if r < 40 else torch.randperm(E, generator=g)[:K] for r in range(128)])
        # 40+ pairs on the first local expert: more than SMALL_LW, so a dequantize-first pass besides small ones
        topv = (torch.rand(128, K, generator=g) + 0.1).bfloat16()
        sg, sd = (blob["tsg"], blob["tsd"]) if moe_ep.tile_scales(blob) else (blob["sgu"], blob["sdn"])
        a2 = dict(x=x, topi=topi.to(torch.int32), wts=topv, lmap=lmap, gu=blob["gu"], dsg=blob["dsg"], dn=blob["dn"],
                  dsd=blob["dsd"], sgu=sg, sdn=sd, LW=moe_ep.SMALL_LW, act=1, lim=10.0, bc=moe_ep.BCAST,
                  rev=moe_ep.REV_SMALL2, tsc=int(moe_ep.tile_scales(blob)))
        one = torch.as_tensor(_sim(moe_ep.kiln_moe_ep_small2, 1, **a2))
        two = torch.as_tensor(_sim(moe_ep.kiln_moe_ep_small2, 2, **dict(a2, spl=1)))
        assert two.shape[0] == 256 and torch.equal(one, two[:128]), ("small2", block)
        both = torch.as_tensor(_sim(moe_ep.kiln_moe_ep_small2, 2, **a2))
        assert torch.equal(one, both[:128]), ("small2 unsplit grid 2", block)
        want = moe_ep.emulate(x, topv, topi, lmap, *ws, act=1, lim=10.0, small=2).float()
        assert (one.float() - want).abs().max() <= 0.01 * want.abs().max()


@needs_nki
def test_moe_ep_lnc2_scatter_indices_are_in_range_and_unique(monkeypatch):
    """At LNC=2 every scatter read-modify-write of kernels/moe_ep.py (dma_compute into out at the lanes' tokens) gets
    in-range, pairwise distinct row indices: an out-of-range index under oob_mode=skip there hangs trn2 (a later DMA
    aborts, the execution times out; tools/probe_ep_race.py, docs/neuron-notes.md "The EP hang at LNC=2 is the
    scatter's out-of-bound skip"), and NKI requires unique scatter indices (nki/isa/_advanced.py dma_compute). Before
    the dummy rows every empty lane scattered to the out-of-range token C. Checked in the simulator on the indices the
    kernel passes (the simulator itself skips out-of-range rows silently): the dequantize-first kernel split and unsplit,
    with experts that leave lanes empty and one with overflow passes, and decode v2's small and dequantize-first passes.
    At grid 1 (trn1) the kernels are unchanged and still scatter empty lanes to C, which the device skips."""
    import numpy as np
    import nki._backends.simulator as sim

    from kiln.kernels import moe_ep
    from tests.test_moe_ep import _experts

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    seen = []
    orig = sim.dma_copy_indirect

    def hook(**kw):
        if kw.get("dst_index") is not None and kw.get("dst_rmw_op") is not None:  # a scatter RMW
            seen.append((np.asarray(kw["dst_index"].get_data()).astype(np.int64).ravel(), int(kw["dst_indirect_max_index"])))
        return orig(**kw)

    monkeypatch.setattr(sim, "dma_copy_indirect", hook)
    H, I, E, K, El, rank = 1024, 256, 12, 4, 3, 1
    owner = torch.arange(E) // El
    lmap = moe_ep.local_map(owner, rank)
    g = torch.Generator().manual_seed(7)
    ws = _experts(El, H, I, seed=3, block=False)
    blob = moe_ep.pack(*ws, tiles=False)

    def check(what, grid1):
        bad = [(len(i), m, int(i.min()), int(i.max())) for i, m in seen if (i < 0).any() or (i >= m).any()
               or len(np.unique(i)) != len(i)]
        if grid1:  # trn1's kernels as before: empty lanes at the out-of-range token C (the device skips them there)
            assert seen and bad, (what, "grid 1 no longer scatters empty lanes to C")
        else:
            assert seen and not bad, (what, f"{len(bad)} of {len(seen)} scatters with an index out of range or repeated",
                                      bad[:3])
        seen.clear()

    C = 256  # rank 1's first expert takes most pairs (overflow passes past its first LW lanes), the others leave lanes empty
    rest = torch.tensor([e for e in range(E) if e != El * rank])
    topi = torch.stack([torch.cat([torch.tensor([El * rank]), rest[torch.randperm(E - 1, generator=g)[:K - 1]]])
                        for _ in range(C)])
    topv = (torch.rand(C, K, generator=g) + 0.1).bfloat16()
    x = torch.randn(C, H, generator=g).bfloat16()
    args = moe_ep.kernel_inputs(x, topv, topi, blob, lmap, 1, 10.0)
    _sim(moe_ep.kiln_moe_ep_kernel, 1, **args)
    check("prefill grid 1", True)
    _sim(moe_ep.kiln_moe_ep_kernel, 2, **dict(args, spl=1))
    check("prefill split", False)
    _sim(moe_ep.kiln_moe_ep_kernel, 2, **args)
    check("prefill unsplit grid 2", False)
    x = torch.randn(128, H, generator=g).bfloat16()
    topi = torch.stack([torch.cat([torch.tensor([El * rank]), rest[torch.randperm(E - 1, generator=g)[:K - 1]]])
                        if r < 40 else torch.randperm(E, generator=g)[:K] for r in range(128)])
    topv = (torch.rand(128, K, generator=g) + 0.1).bfloat16()
    a2 = dict(x=x, topi=topi.to(torch.int32), wts=topv, lmap=lmap, gu=blob["gu"], dsg=blob["dsg"], dn=blob["dn"],
              dsd=blob["dsd"], sgu=blob["sgu"], sdn=blob["sdn"], LW=moe_ep.SMALL_LW, act=1, lim=10.0, bc=moe_ep.BCAST,
              rev=moe_ep.REV_SMALL2, tsc=0)
    _sim(moe_ep.kiln_moe_ep_small2, 2, **dict(a2, spl=1))
    check("decode v2 split", False)
    _sim(moe_ep.kiln_moe_ep_small2, 2, **a2)
    check("decode v2 unsplit grid 2", False)


def test_moe_ep_lnc2_lines_leave_trn1_rev_unchanged():
    """The LNC=2 fix's lines (marked "# lnc2") are left out of the source moe_ep hashes for grid-1 launches, so trn1's
    REV values, and with them every trn1 EP graph key, are what engine-v0 fdea9af had; grid-2 launches pass the REV
    of the whole text. A deliberate trn1 re-key (an edit to these kernels' shared source) updates the numbers here.
    """
    from kiln.kernels import moe_ep

    assert (moe_ep.REV, moe_ep.REV_SMALL, moe_ep.REV_SMALL2, moe_ep.REV_SMALL3, moe_ep.REV_SMALL5) == (
        2569844845, 2049637701, 3087051072, 2852879506, 1388171348)
    assert moe_ep.REV_LNC2 != moe_ep.REV and moe_ep.REV_SMALL2_LNC2 != moe_ep.REV_SMALL2


@needs_nki
def test_kda_decode_row_split_is_the_whole_kernel(monkeypatch):
    """kernels/kda_decode.py split over the two programs by rows (KILN_LNC_SPLIT=kda_decode): each program runs the
    head pipeline of its half of the rows; o and every real row's state equal grid 1's bit for bit (an odd row count,
    rows starting from zero, two padding rows sharing the scratch write row)."""
    import nki

    from kiln.kernels import kda_decode as kd
    from tests.test_kda_decode import inputs

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    for B in (5, 8):
        pool, slot, keep, q, k, v, g, beta = inputs(B=B, R=12, H=8)
        R = pool.shape[0]
        rd = torch.where(keep, slot, torch.full_like(slot, R))
        slots = torch.stack([rd, slot]).to(torch.int32)
        args = dict(slots=slots, q=q, k=k, v=v, g=g, beta=beta, ident=torch.eye(128), rev=kd.REV)
        o1, p1 = nki.simulate(kd.kiln_kda_decode_kernel)(pool=pool.clone(), **args)
        o2, p2 = nki.simulate(kd.kiln_kda_decode_kernel[2])(pool=pool.clone(), spl=1, **args)
        o1, p1, o2, p2 = (torch.as_tensor(t) for t in (o1, p1, o2, p2))
        real = keep.nonzero().flatten()
        assert torch.equal(o1[real], o2[real]), B
        assert torch.equal(p1[slot[real]], p2[slot[real]]), B
        untouched = torch.ones(R, dtype=torch.bool)
        untouched[slot] = False
        assert torch.equal(p2[untouched], pool[untouched]), B


@needs_nki
def test_dsa_decode_row_split_is_the_whole_kernel(monkeypatch):
    """kernels/dsa_decode.py split over the two programs by rows (KILN_LNC_SPLIT=dsa_decode): o equals grid 1's bit
    for bit (3 rows, bf16 cache, 512 selected pools and a tail pool, padding slots masked)."""
    import nki

    from kiln.kernels import dsa_decode as dd

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    g = torch.Generator().manual_seed(4)
    B, H, R, PS, pages = 3, 8, 512, 32, 70
    kc = torch.randn(B * pages * PS, R, generator=g).to(torch.bfloat16)
    q_lat = (torch.randn(B, H, R, generator=g) * 0.05).to(torch.bfloat16)
    rows = torch.zeros(B, dd.NCH * 128, dtype=torch.long)
    bias = torch.full((B, dd.NCH * 128, dd.KP), dd.NEG_INF)
    for b in range(B):
        pools = torch.randperm(pages * PS // dd.KP - 1, generator=g)[:513]
        rows[b, :513] = b * pages * PS // dd.KP + pools
        bias[b, :512] = 0.0
        bias[b, 512, :3] = 0.0
    args = dict(q_lat=q_lat, kc=kc.reshape(-1, dd.KP * R),
                rows_t=rows.to(torch.int32).reshape(B, dd.NCH, 128).permute(2, 0, 1).contiguous(),
                bias_t=bias.reshape(B, dd.NCH, 128, dd.KP).permute(0, 1, 3, 2).contiguous(),
                identb=torch.eye(128).to(torch.bfloat16), scale=256 ** -0.5, fp8=0, rev=dd.REV)
    one = torch.as_tensor(nki.simulate(dd.kiln_dsa_decode_kernel)(**args))
    two = torch.as_tensor(nki.simulate(dd.kiln_dsa_decode_kernel[2])(spl=1, **args))
    assert torch.equal(one, two)


@needs_nki
def test_dsa_fused_query_split_is_the_whole_kernel(monkeypatch):
    """kernels/dsa_fused.py split over the two programs by query tiles (KILN_LNC_SPLIT=dsa_fused): o equals grid 1's
    bit for bit (4 query tiles at the end of a 1024-key chunk, 256 pools of 4 keys, a top-64 selection), and grid 1
    matches the host emulation (the selection's ties aside, as the serving path's test does)."""
    import nki

    from kiln.kernels import dsa_fused as df

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    g = torch.Generator().manual_seed(5)
    C, Hi, D, H, R, L = 512, 4, 128, 2, 512, 1024
    P = L // 4
    qI = (torch.randn(C, Hi, D, generator=g) * 0.1).to(torch.bfloat16)
    w = torch.rand(C, Hi, generator=g)
    pk = (torch.randn(P, D, generator=g) * 0.1).to(torch.bfloat16)
    pos = torch.arange(C) + (L - C)
    q_lat = (torch.randn(C, H, R, generator=g) * 0.05).to(torch.bfloat16)
    kc = torch.randn(L, R, generator=g).to(torch.bfloat16)
    args = dict(qT=qI.permute(1, 2, 0).contiguous(), w=w.contiguous(), pkT=pk.t().contiguous(), posf=pos.float(),
                q_lat=q_lat, kc=kc, identb=torch.eye(128).to(torch.bfloat16), keep=64, nbits=P.bit_length(),
                scale_i=D ** -0.5, scale_a=R ** -0.5, rev=df.REV)
    one = torch.as_tensor(nki.simulate(df.kiln_dsa_fused_kernel)(**args))
    two = torch.as_tensor(nki.simulate(df.kiln_dsa_fused_kernel[2])(spl=1, **args))
    assert torch.equal(one, two)
    ref = df.emulate(qI, w, pk, pos, q_lat, kc, 64, D ** -0.5, R ** -0.5)
    assert (one.float() - ref).abs().max() < 2e-2 * ref.abs().max()


@needs_nki
def test_dsa_long_select_lnc2_program_1_leaves_the_scratch_alone(monkeypatch):
    """kernels/dsa_long_select.py at grid 2 (trn2, LNC=2): program 0 alone selects and program 1 waits at a core
    barrier, so the HBM score scratch, one tensor shared by the two physical cores, is written and read back (the
    level-2 indirect gathers) by one program only. Before, both programs ran the whole kernel on the same scratch and
    the R8 long-context engine's 4096-page graphs answered wrong on trn2 (docs/neuron-notes.md "Long prompts on trn2").
    In the simulator: grid 2 issues exactly grid 1's indirect gathers and gives grid 1's pools and scores, which equal
    the host emulation's selection."""
    import nki._backends.simulator as sim

    from kiln.kernels import dsa_long_select as dls

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    n = [0]
    orig = sim.dma_copy_indirect

    def hook(**kw):
        n[0] += 1
        return orig(**kw)

    monkeypatch.setattr(sim, "dma_copy_indirect", hook)
    g = torch.Generator().manual_seed(11)
    N, Hi, D, P, keep = 128, 4, 128, 2048, 64
    qI = torch.randn(N, Hi, D, generator=g).bfloat16().float()
    w = torch.randn(N, Hi, generator=g)
    pk = torch.randn(P, D, generator=g).bfloat16().float()
    npool = torch.randint(keep + 1, P, (N,), generator=g)
    sub = dls.pick_sub(P, keep)
    assert sub > 0  # two levels: the scratch is read back by indirect gathers
    args = dict(dls.kernel_inputs(qI, w, pk, npool), keep=keep, scale=D ** -0.5, sub=sub, lsub=sub.bit_length() - 1,
                one=0, loop=0)
    res = {}
    for grid, rev in ((1, dls.REV), (2, dls.REV_LNC2)):
        n[0] = 0
        out, outv = _sim(dls.kiln_dsa_long_select_kernel, grid, rev=rev, **args)
        res[grid] = (torch.as_tensor(out).reshape(N, keep), torch.as_tensor(outv).reshape(N, keep), n[0])
    assert res[1][2] > 0 and res[2][2] == res[1][2], f"indirect gathers: grid 1 {res[1][2]}, grid 2 {res[2][2]}"
    assert torch.equal(res[1][0], res[2][0]) and torch.equal(res[1][1], res[2][1])
    want, cnt, _ = dls.emulate(qI, w, pk, npool, keep, D ** -0.5)
    for r in range(N):
        c = int(cnt[r])
        assert set(res[2][0][r, :c].tolist()) == set(want[r, :c].tolist()), r


@needs_nki
def test_dsa_long_pipe_and_trip_kernels_lnc2_program_1_leaves_the_scratch_alone(monkeypatch):
    """The other long-context selection kernels that run dsa_long_select's tile on HBM score scratch (kernels/
    dsa_long_pipe.py, every tile in one call; kernels/dsa_long_select_t.py, one tile under a run-time trip count) take
    the same grid-2 form: program 0 alone selects. In the simulator grid 2 issues exactly grid 1's indirect gathers and
    gives grid 1's pools and scores."""
    import nki._backends.simulator as sim

    from kiln.kernels import dsa_long_pipe as dp
    from kiln.kernels import dsa_long_select as dls
    from kiln.kernels import dsa_long_select_t as dst

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    n = [0]
    orig = sim.dma_copy_indirect

    def hook(**kw):
        n[0] += 1
        return orig(**kw)

    monkeypatch.setattr(sim, "dma_copy_indirect", hook)
    g = torch.Generator().manual_seed(12)
    Hi, D, P, keep = 4, 128, 2048, 64
    sub = dls.pick_sub(P, keep)
    common = dict(keep=keep, scale=D ** -0.5, sub=sub, lsub=sub.bit_length() - 1, one=0)
    for name, N in (("pipe", 256), ("trip", 128)):
        qI = torch.randn(N, Hi, D, generator=g).bfloat16().float()
        w = torch.randn(N, Hi, generator=g)
        pk = torch.randn(P, D, generator=g).bfloat16().float()
        npool = torch.randint(keep + 1, P, (N,), generator=g)
        args = dls.kernel_inputs(qI, w, pk, npool)
        res = {}
        for grid in (1, 2):
            n[0] = 0
            if name == "pipe":
                out, outv = _sim(dp.kiln_dsa_long_pipe_kernel, grid, rev=dp.REV if grid == 1 else dp.REV_LNC2,
                                 per_chunk=dp.PER_CHUNK, ktall=1, **args, **common)
            else:
                out, outv = _sim(dst.kiln_dsa_long_select_t_kernel, grid, rev=dst.REV if grid == 1 else dst.REV_LNC2,
                                 trip=torch.ones(1, 1, dtype=torch.int32), **args, **common)
            res[grid] = (torch.as_tensor(out).reshape(N, keep), torch.as_tensor(outv).reshape(N, keep), n[0])
        assert res[1][2] > 0 and res[2][2] == res[1][2], (name, res[1][2], res[2][2])
        assert torch.equal(res[1][0], res[2][0]) and torch.equal(res[1][1], res[2][1]), name


def test_dsa_long_select_lnc2_lines_leave_trn1_rev_unchanged():
    """The grid-2 lines of the long-context selection kernels (marked "# lnc2") are left out of their grid-1 revisions:
    trn1's graphs keep the keys they had before them; grid-2 launches pass the revisions of the whole texts."""
    from kiln.kernels import dsa_long_pipe as dp
    from kiln.kernels import dsa_long_pipe_x as dpx
    from kiln.kernels import dsa_long_select as dls
    from kiln.kernels import dsa_long_select_t as dst

    assert (dls.REV, dp.REV, dst.REV, dpx.REV) == (2642970146, 1604308206, 1934064275, 1270195910)
    assert dls.REV_LNC2 != dls.REV and dp.REV_LNC2 != dp.REV and dst.REV_LNC2 != dst.REV
