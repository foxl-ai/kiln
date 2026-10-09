"""Expert parallelism (KILN_MOE_EP=1, models/decoder.py moe_ep_enabled, kernels/moe_ep.py) on the CPU: the kernel's
layout and host arithmetic, and engines whose ranks hold whole experts giving the tokens and logprobs of the
tensor-parallel layout (CPU tensor parallelism over gloo).

The device kernel itself is checked against moe_ep.emulate on a NeuronCore by tools/probe_moe_ep.py (full: plan,
passes, overflow passes, scatter-add output; core: the passes alone) and end to end by real-weight ppl.
"""

import pytest
import torch

LOGPROB_TOL = 1e-4  # fp32; measured differences are printed


def _experts(El, H, I, seed=0, block=False):
    """El whole experts as the loader holds them: e4m3fn codes of 128 x 128 blocks, per-row scales, then
    models/quant.fit_e4m3_max (most rows of a block whose max exceeds 240 get their scale doubled; block: the
    whole 128 x 128 block, as the EP loader fits a block-scaled checkpoint, so every tile keeps one scale)."""
    from kiln.models.quant import FP8, fit_e4m3_max

    g = torch.Generator().manual_seed(seed)

    def blocks(rows, cols):
        w = torch.randn(rows, cols, generator=g) * 0.02
        bs = w.abs().view(rows // 128, 128, cols // 128, 128).amax((1, 3)) / 448.0
        cw = (w / bs.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(FP8)
        return fit_e4m3_max(cw, bs.repeat_interleave(128, 0), 240.0, 128 if block else None)

    out = [[], [], [], []]
    for _ in range(El):
        wgu, sgu = blocks(2 * I, H)
        wd, sd = blocks(H, I)
        for lst, t in zip(out, (wgu, sgu, wd.T.contiguous(), sd.T.contiguous())):
            lst.append(t)
    return tuple(torch.stack(t) for t in out)


def test_pack_round_trip_and_supports():
    from kiln.kernels import moe_ep

    ws = _experts(2, 512, 256)
    assert moe_ep.supports(*ws, True)
    assert not moe_ep.supports(*ws, False)
    blob = moe_ep.pack(*ws)
    assert blob["gu"].shape == (2, 128, 2, 2, 4, 128) and blob["dn"].shape == (2, 128, 2, 512)
    assert blob["dsg"].shape == (2, 2, 2, 128, 4) and blob["dsd"].shape == (2, 128, 4, 2)
    for a, b in zip(ws, moe_ep.unpack(blob)):
        assert torch.equal(a.view(torch.uint8) if a.dtype != torch.float32 else a,
                           b.view(torch.uint8) if b.dtype != torch.float32 else b)
    # the tile the kernel loads for I-chunk m, gate (g 0) / up (g 1), h-tile c: [h, i] = w_gu[g I + m 128 + i, c 128 + h]
    w_gu = ws[0].view(torch.uint8)
    e, m, g, c = 1, 1, 1, 3
    tile = blob["gu"][e, :, m, g, c, :]
    assert torch.equal(tile, w_gu[e, g * 256 + m * 128:g * 256 + (m + 1) * 128, c * 128:(c + 1) * 128].T)


def test_pack_tile_scales():
    """Block-constant scales (KILN_MOE_EP_FIT=block): pack(tiles=True) adds one scale per kernel tile, tsg[e, m, g, c]
    for gate_up rows g I + m 128 .. and columns c 128 .., tsd[e, m, b] for down input rows m 128 .. and output
    columns b 128 .., leaves out the per-row split scales (KILN_MOE_EP_TILES) and still unpacks to the same experts;
    per-row scales refuse it."""
    from kiln.kernels import moe_ep

    ws = _experts(2, 512, 256, seed=5, block=True)
    blob = moe_ep.pack(*ws, tiles=True)
    assert moe_ep.tile_scales(blob) and "sgu" not in blob and "sdn" not in blob
    assert blob["tsg"].shape == (2, 2, 2, 4) and blob["tsd"].shape == (2, 2, 4)
    s_gu, s_down = ws[1], ws[3]
    e, m, g, c, b = 1, 1, 1, 2, 3
    assert bool((s_gu[e, g * 256 + m * 128:g * 256 + (m + 1) * 128, c] == blob["tsg"][e, m, g, c]).all())
    assert bool((s_down[e, m, b * 128:(b + 1) * 128] == blob["tsd"][e, m, b]).all())
    for a, u in zip(ws, moe_ep.unpack(blob)):
        assert torch.equal(a.view(torch.uint8) if a.dtype != torch.float32 else a,
                           u.view(torch.uint8) if u.dtype != torch.float32 else u)
    # the tile-scale form is a layout of the same arithmetic: the emulation reads the same experts
    assert not moe_ep.tile_scales(moe_ep.pack(*ws))
    row = _experts(2, 512, 256, seed=5)
    assert not torch.equal(row[1], ws[1])  # some rows of an over-240 block were not halved per row
    with pytest.raises(ValueError):
        moe_ep.pack(*row, tiles=True)


def test_split3_is_exact():
    from kiln.kernels.moe_ep import split3

    g = torch.Generator().manual_seed(0)
    s = torch.cat([torch.rand(100000, generator=g) * 1e-3, torch.exp2(torch.randint(-30, 10, (1000,), generator=g).float()),
                   torch.rand(1000, generator=g) * 1e6, torch.tensor([1.0, 3.0, 1.5e-30, 0.000174386])])
    p = split3(s)
    assert p.dtype == torch.bfloat16
    assert torch.equal((p[..., 0].float() + p[..., 1].float()) + p[..., 2].float(), s)


def test_local_map():
    from kiln.kernels.moe_ep import local_map

    owner = torch.arange(8) // 2  # 4 ranks, 2 experts each
    assert local_map(owner, 1).tolist() == [[2, 2, 0, 1, 2, 2, 2, 2, 2]]
    assert local_map(owner, 3).tolist() == [[2, 2, 2, 2, 2, 2, 0, 1, 2]]


def test_max_passes_bounds_every_routing():
    from kiln.kernels.moe_ep import max_passes

    g = torch.Generator().manual_seed(0)
    E, K, R = 288, 8, 32
    El = E // R
    for C in (128, 1024, 4096):
        for LW in (128, 256):
            bound = max_passes(C, K, El, LW)
            routings = [torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(C)]),
                        torch.arange(K).repeat(C, 1)]  # every token on experts 0..7: all local on rank 0
            for ti in routings:
                n = torch.bincount(ti.flatten(), minlength=E)[:El]
                assert int((-(-n // LW)).sum()) <= bound
            # the worst case: as many experts as possible one pair past a multiple of LW
            worst = torch.full((El,), C)
            worst[: (C * min(K, El)) // C] = C
            assert int((-(-worst[: min(K, El)] // LW)).sum()) <= bound


def test_small_lane_arithmetic_is_the_per_row_one():
    """moe_ep.expert_out_small (kiln_moe_ep_small: per-tile fp32 dot products times per-row scales, as moe_dedupe)
    agrees with the dequantize-first arithmetic (expert_out) to bf16 rounding."""
    from kiln.kernels import moe_ep

    ws = _experts(1, 512, 256, seed=3)
    x = torch.randn(16, 512, generator=torch.Generator().manual_seed(4)).bfloat16()
    a = moe_ep.expert_out_small(x, ws[0][0], ws[1][0], ws[2][0], ws[3][0])
    b = moe_ep.expert_out(x, ws[0][0], ws[1][0], ws[2][0], ws[3][0])
    err = (a - b).abs().max().item() / b.abs().max().item()
    print(f"small-lane vs dequantize-first arithmetic: rel {err:.4f}")
    assert err < 0.02


def test_emulation_is_the_local_pairs_sum():
    """moe_ep.emulate (the kernel's arithmetic: bf16(w y) added in bf16 in local-expert order) against an fp32
    reference over the same local pairs; rows without a local pair are exactly zero."""
    from kiln.kernels import moe_ep
    from kiln.models.quant import dequant, dequant_t

    H, I, E, K, El, rank = 512, 256, 8, 2, 2, 1
    ws = _experts(El, H, I, seed=1)
    owner = torch.arange(E) // El
    lmap = moe_ep.local_map(owner, rank)
    g = torch.Generator().manual_seed(2)
    T = 64
    x = torch.randn(T, H, generator=g).bfloat16()
    topi = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])
    topv = (torch.rand(T, K, generator=g) + 0.1).bfloat16()
    got = moe_ep.emulate(x, topv, topi, lmap, *ws, act=1, lim=10.0, small=False).float()
    ref = torch.zeros(T, H)
    loc = lmap.view(-1)[topi]
    for le in range(El):
        t, k = (loc == le).nonzero(as_tuple=True)
        gu = x[t].float() @ dequant(ws[0][le], ws[1][le], torch.float32).T
        a = torch.nn.functional.silu(gu[:, :I].clamp(max=10.0)) * gu[:, I:].clamp(min=-10.0, max=10.0)
        ref.index_add_(0, t, (a @ dequant_t(ws[2][le], ws[3][le], torch.float32)) * topv[t, k].float().unsqueeze(1))
    none = (loc >= El).all(1)
    assert none.any() and bool((got[none] == 0).all())
    err = (got - ref).abs().max().item() / ref.abs().max().item()
    print(f"emulation vs fp32 reference: rel {err:.4f}")
    assert err < 0.02


def test_small2_emulation_mixes_the_two_arithmetics():
    """kiln_moe_ep_small2 (KILN_MOE_EP_SMALL_V=2) runs a local expert with at most SMALL_LW pairs by the small-lane
    arithmetic (expert_out_small) and one with more by the dequantize-first one (expert_out), small ones first, each
    group in local-expert order; emulate(small=2) is that, and equals the composition done by hand here."""
    from kiln.kernels import moe_ep

    H, I, E, K, El, rank, T = 512, 256, 8, 2, 4, 1, 48
    ws = _experts(El, H, I, seed=5, block=True)
    owner = torch.arange(E) // El
    lmap = moe_ep.local_map(owner, rank)
    g = torch.Generator().manual_seed(6)
    x = torch.randn(T, H, generator=g).bfloat16()
    topi = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])
    topi[:20, 0] = 4  # local expert 0 (expert 4 on rank 1) gets more than SMALL_LW pairs
    topi[:20, 1] = torch.where(topi[:20, 1] == 4, 5, topi[:20, 1])
    topv = (torch.rand(T, K, generator=g) + 0.1).bfloat16()
    loc = lmap.view(-1)[topi]
    n = [int((loc == le).sum()) for le in range(El)]
    assert n[0] > moe_ep.SMALL_LW and any(0 < c <= moe_ep.SMALL_LW for c in n[1:])
    got = moe_ep.emulate(x, topv, topi, lmap, *ws, act=1, lim=10.0, small=2)
    out = torch.zeros(T, H, dtype=torch.bfloat16)
    order = [le for le in range(El) if n[le] <= moe_ep.SMALL_LW] + [le for le in range(El) if n[le] > moe_ep.SMALL_LW]
    for le in order:
        t, k = (loc == le).nonzero(as_tuple=True)
        if not t.numel():
            continue
        args = (x[t], ws[0][le], ws[1][le], ws[2][le], ws[3][le], 1, 10.0)
        y = moe_ep.expert_out(*args, rounded=False) if n[le] > moe_ep.SMALL_LW else moe_ep.expert_out_small(*args)
        out[t] = (out[t].float() + (y * topv[t, k].float().unsqueeze(1)).bfloat16().float()).bfloat16()
    assert torch.equal(got, out)
    small1 = moe_ep.emulate(x, topv, topi, lmap, *ws, act=1, lim=10.0, small=True).float()
    err = (got.float() - small1).abs().max().item() / small1.abs().max().item()
    print(f"small2 vs small-lane emulation: rel {err:.4f}")
    assert err < 0.02


def test_ep_default_is_glm5_next_on_trn1(monkeypatch):
    """KILN_MOE_EP unset (auto): on for the glm5_next family (GLM-5.3-Flash) on trn1 / trn1n / trn2 and on a host
    without a Neuron device, off for other models and on trn3 / inf2 (not measured there); 1 / 0 force it; the
    model turns the automatic default off where tp does not divide the experts."""
    import types

    from kiln.models import decoder

    glm = types.SimpleNamespace(hybrid=types.SimpleNamespace(family="glm5_next"))
    other = types.SimpleNamespace(hybrid=types.SimpleNamespace(family="qwen4_exp"))
    plain = types.SimpleNamespace(hybrid=None)
    monkeypatch.delenv("KILN_MOE_EP", raising=False)
    for target, want in (("trn1", True), ("trn1n", True), ("trn2", True), ("trn3", False), ("inf2", False)):
        monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", target)
        assert decoder.moe_ep_enabled(glm) is want, target
        assert not decoder.moe_ep_enabled(other) and not decoder.moe_ep_enabled(plain)
    monkeypatch.setenv("KILN_MOE_EP", "1")
    assert decoder.moe_ep_enabled(plain)  # forced on (trn1n target still set)
    monkeypatch.setenv("KILN_MOE_EP", "0")
    assert not decoder.moe_ep_enabled(glm)
    monkeypatch.setenv("KILN_MOE_EP", "maybe")
    with pytest.raises(ValueError):
        decoder.moe_ep_enabled(glm)


def test_ep_default_builds_glm5_next_expert_parallel(tmp_path, monkeypatch):
    """With KILN_MOE_EP unset a GLM-5.3-Flash model built at tp 2 holds whole experts, one with 6 experts at tp 4
    does not (the automatic default steps aside where tp does not divide them), and a MiMo-V2 model stays TP."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from kiln.config import ModelConfig
    from kiln.models.decoder import DecoderForCausalLM
    from tests.test_glm5_next import build
    from tests.test_mimo_v2 import build_reference

    monkeypatch.delenv("KILN_MOE_EP", raising=False)
    monkeypatch.delenv("NEURON_PLATFORM_TARGET_OVERRIDE", raising=False)
    build(str(tmp_path / "glm"))
    build(str(tmp_path / "glm6"), n_routed_experts=6)
    build_reference(str(tmp_path / "mimo"))
    glm, glm6, mimo = (ModelConfig.from_pretrained(str(tmp_path / n)) for n in ("glm", "glm6", "mimo"))
    assert glm6.num_experts == 6
    with torch.device("meta"):
        assert DecoderForCausalLM(glm, torch.float32, 64, 0, 2).moe_ep
        assert DecoderForCausalLM(glm6, torch.float32, 64, 0, 2).moe_ep
        assert not DecoderForCausalLM(glm6, torch.float32, 64, 0, 4).moe_ep
        assert not DecoderForCausalLM(mimo, torch.float32, 64, 0, 2).moe_ep


def test_ep_default_needs_eight_decode_rows_per_group(tmp_path, monkeypatch):
    """Under KILN_MOE_EP_SMALL_V=1 the automatic default stays off below 8 decode rows per DP-attention group
    (max_num_seqs / dp_attention): G16's 16 sequences over 4 groups (4 rows) measured EP 2.9% below TP, F0's 32 over 4
    (8 rows) 8.9% above (models/decoder.py moe_ep_enabled); KILN_MOE_EP=1 still forces it, and an unknown max_num_seqs
    leaves it on."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from kiln.config import ModelConfig
    from kiln.models.decoder import DecoderForCausalLM
    from tests.test_glm5_next import build

    import importlib

    from kiln.kernels import moe_ep

    monkeypatch.delenv("KILN_MOE_EP", raising=False)
    monkeypatch.delenv("NEURON_PLATFORM_TARGET_OVERRIDE", raising=False)
    build(str(tmp_path))
    glm = ModelConfig.from_pretrained(str(tmp_path))

    def ep(dp, seqs):
        with torch.device("meta"):
            return DecoderForCausalLM(glm, torch.float32, 64, 0, 4, dp_attention=dp, max_num_seqs=seqs).moe_ep

    monkeypatch.setenv("KILN_MOE_EP_SMALL_V", "1")
    try:
        importlib.reload(moe_ep)
        assert not ep(4, 16) and ep(4, 32) and ep(4, 64)  # G16 / F0 / G64
        assert not ep(1, 4) and ep(1, 8) and ep(2, 16) and not ep(2, 15)
        assert ep(4, None)
    finally:
        monkeypatch.delenv("KILN_MOE_EP_SMALL_V", raising=False)
        importlib.reload(moe_ep)


def test_ep_default_needs_four_decode_rows_per_group(tmp_path, monkeypatch):
    """With the default decode kernel (kiln_moe_ep_small2) the automatic default turns expert parallelism on from 4
    decode rows per DP-attention group: conc 16 at 4 rows per group measured EP v2 98.0 against TP 86.9 out tok/s
    (models/decoder.py moe_ep_enabled); below that it stays on TP, unmeasured."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from kiln.config import ModelConfig
    from kiln.kernels import moe_ep
    from kiln.models.decoder import DecoderForCausalLM, ep_auto_min_decode_rows
    from tests.test_glm5_next import build

    monkeypatch.delenv("KILN_MOE_EP", raising=False)
    monkeypatch.delenv("NEURON_PLATFORM_TARGET_OVERRIDE", raising=False)
    assert moe_ep.SMALL_V == 2 and ep_auto_min_decode_rows() == 4
    build(str(tmp_path))
    glm = ModelConfig.from_pretrained(str(tmp_path))

    def ep(dp, seqs):
        with torch.device("meta"):
            return DecoderForCausalLM(glm, torch.float32, 64, 0, 4, dp_attention=dp, max_num_seqs=seqs).moe_ep

    assert ep(4, 16) and ep(4, 32) and ep(4, 64) and not ep(4, 15)  # G16 / F0 / G64
    assert ep(1, 4) and not ep(1, 3) and ep(2, 8) and not ep(2, 7)
    assert ep(4, None)
    monkeypatch.setenv("KILN_MOE_EP", "1")
    assert ep(4, 16)


def _engine(path, tp, ep, monkeypatch, **kw):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    monkeypatch.setenv("KILN_MOE_EP", "1" if ep else "0")
    base = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256, max_num_seqs=3,
                max_model_len=256, max_prefill_tokens=16, tp=tp)
    base.update(kw)
    return LLMEngine(EngineConfig(**base))


def _run(path, tp, ep, monkeypatch, prompts, sp, **kw):
    eng = _engine(path, tp, ep, monkeypatch, **kw)
    try:
        assert eng.runner.model.moe_ep == ep
        reqs = eng.generate(prompts, sp)
        return [r.output_ids for r in reqs], [[x[0] for x in r.logprobs] for r in reqs]
    finally:
        eng.close()


def _compare(name, a, b):
    (ids, lp), (ids0, lp0) = a, b
    err = max(abs(x - y) for p, q in zip(lp, lp0) for x, y in zip(p, q))
    print(f"{name}: tokens {'equal' if ids == ids0 else 'DIFFER'}, max |dlogprob| {err:.2e}")
    assert ids == ids0, name
    assert err < LOGPROB_TOL, (name, err)


def test_glm5_next_expert_parallel_equals_tensor_parallel(tmp_path, monkeypatch):
    """GLM-5.3-Flash (clamped SwiGLU experts, a TP shared expert, mHC, DP attention, sequence-parallel prefill
    streams): every rank holding E / tp whole experts gives the tokens and the logprobs (to fp32 noise) of every
    rank holding 1 / tp of every expert, at tp 4 with DP attention 2 (piecewise, prefill chunks long enough that
    every expert is used) and at tp 2."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from kiln.engine.request import SamplingParams
    from tests.test_glm5_next import build, prompts

    build(str(tmp_path), index_topk=16)
    ps = prompts(9, (13, 40, 27))
    sp = SamplingParams(max_new_tokens=8, ignore_eos=True, logprobs=1)
    for kw in (dict(tp=4, dp_attention=2, piecewise=True, piecewise_group=2), dict(tp=2)):
        tp = kw.pop("tp")
        _compare(f"glm5_next tp={tp} {kw}", _run(str(tmp_path), tp, True, monkeypatch, ps, sp, **kw),
                 _run(str(tmp_path), tp, False, monkeypatch, ps, sp, **kw))


def test_mimo_v2_expert_parallel_equals_tensor_parallel(tmp_path, monkeypatch):
    """A plain MoE (MiMo-V2: sigmoid routing with a correction bias, SiLU experts, no shared expert): EP at tp 2
    and 4 against TP at the same tp."""
    from kiln.engine.request import SamplingParams
    from tests.test_mimo_v2 import build_reference

    build_reference(str(tmp_path))
    ps = [[5, 9, 11, 200, 3, 77, 12, 13, 14, 15, 16], list(range(40, 70))]
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True, logprobs=1)
    for tp in (2, 4):
        _compare(f"mimo_v2 tp={tp}", _run(str(tmp_path), tp, True, monkeypatch, ps, sp, num_pages=128, max_num_seqs=2,
                                          max_model_len=128, max_prefill_tokens=8),
                 _run(str(tmp_path), tp, False, monkeypatch, ps, sp, num_pages=128, max_num_seqs=2, max_model_len=128,
                      max_prefill_tokens=8))


def test_expert_parallel_layers_hold_whole_experts(tmp_path, monkeypatch):
    """The model built at tp 4 rank 1 with KILN_MOE_EP=1: each MoE layer holds E / tp experts of the whole
    intermediate size, its local map points experts 2 and 3 at 0 and 1, and the router keeps every expert."""
    from kiln.config import ModelConfig
    from kiln.models.loader import load_model
    from tests.test_mimo_v2 import build_reference

    build_reference(str(tmp_path))
    cfg = ModelConfig.from_pretrained(str(tmp_path))
    monkeypatch.setenv("KILN_MOE_EP", "1")
    model = load_model(str(tmp_path), cfg, torch.float32, torch.device("cpu"), 128, 1, 4, None)
    moe = [l for l in model.layers if l.moe]
    assert moe and all(l.moe_ep for l in moe)
    l = moe[0]
    assert l.w_gu.shape == (2, 2 * cfg.moe_intermediate_size, cfg.hidden_size)
    assert l.router.shape[0] == cfg.num_experts
    assert l.ep_lmap.tolist() == [[2, 2, 0, 1, 2, 2, 2, 2, 2]]


def test_small_kernel_default_is_v2(monkeypatch):
    """KILN_MOE_EP_SMALL_V unset runs kiln_moe_ep_small2 (one pass per local expert with pairs) for decode-sized calls,
    and the host emulation then follows its arithmetic (small=2); v1 stays selectable by the variable."""
    import importlib

    from kiln.kernels import moe_ep

    monkeypatch.delenv("KILN_MOE_EP_SMALL_V", raising=False)
    try:
        m = importlib.reload(moe_ep)
        assert m.SMALL_V_DEFAULT == 2 and m.SMALL_V == 2
        H, I, E, K, El, rank, T = 512, 256, 8, 2, 4, 1, 48
        ws = _experts(El, H, I, seed=7, block=True)
        lmap = m.local_map(torch.arange(E) // El, rank)
        g = torch.Generator().manual_seed(8)
        x = torch.randn(T, H, generator=g).bfloat16()
        topi = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])
        topi[:20, 0] = 4
        topi[:20, 1] = torch.where(topi[:20, 1] == 4, 5, topi[:20, 1])
        topv = (torch.rand(T, K, generator=g) + 0.1).bfloat16()
        assert m.uses_small(T)
        assert torch.equal(m.emulate(x, topv, topi, lmap, *ws), m.emulate(x, topv, topi, lmap, *ws, small=2))
        monkeypatch.setenv("KILN_MOE_EP_SMALL_V", "1")
        m = importlib.reload(moe_ep)
        assert m.SMALL_V == 1
        assert torch.equal(m.emulate(x, topv, topi, lmap, *ws), m.emulate(x, topv, topi, lmap, *ws, small=True))
    finally:
        monkeypatch.delenv("KILN_MOE_EP_SMALL_V", raising=False)
        importlib.reload(moe_ep)
