"""The expert-deduplicating NKI MoE kernel's CPU-checkable parts (kiln/kernels/moe_dedupe.py): the
tile-scale expert layout (exact re-basing, refusals), the routing plan (slots, lanes, gather /
route matrices), the kernel's arithmetic emulated in torch, and (where the NKI package is
installed) the kernel under the NKI CPU simulator. The device run is tools/probe_moe_kernel.py."""

import importlib.util

import pytest
import torch

from kiln.kernels import moe_dedupe as mdd
from kiln.models.quant import E2M1, FP8, dequant, dequant_t

H = 512


def experts(seed=0, e=6, h=H, spread=6):
    """MXFP4-like experts: E2M1 codes as FP8, power-of-two bf16 block scales whose exponents
    differ by up to `spread` (as the real checkpoint's: tools/check_expert_scales.py)."""
    g = torch.Generator().manual_seed(seed)
    lut = torch.tensor(E2M1)

    def codes(*s):
        return lut[torch.randint(0, 16, s, generator=g)].to(FP8)

    def scales(*s):
        return torch.exp2(-10.0 + torch.randint(0, spread + 1, s, generator=g).float()).bfloat16()

    return codes(e, 128, h), scales(e, 128, h // 32), codes(e, 64, h), scales(e, 2, h)


def reference(x, topv, topi, w_gu, s_gu, w_down, s_down):
    """fp32: dequantize, matvec, SiLU gate, matvec, weight, sum over k."""
    wg, wd = dequant(w_gu, s_gu, torch.float32), dequant_t(w_down, s_down, torch.float32)
    out = torch.zeros(x.shape, dtype=torch.float32)
    for t in range(x.shape[0]):
        for k in range(topi.shape[1]):
            e = int(topi[t, k])
            gu = wg[e] @ x[t].float()
            out[t] += topv[t, k].float() * ((torch.nn.functional.silu(gu[:64]) * gu[64:]) @ wd[e])
    return out


def routing(T, K=8, E=256, seed=1, h=H):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, h, generator=g).bfloat16()
    topi = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])
    topv = (torch.rand(T, K, generator=g) + 0.1).bfloat16()
    return x, topv, topi


def test_pack_is_exact():
    ws = experts()
    assert mdd.supports(*ws, down_t=True)
    blob = mdd.pack(*ws)
    assert blob.dtype == torch.uint8 and tuple(blob.shape) == (6, 128, mdd.blob_cols(H))
    w_gu, s_gu, w_down, s_down = mdd.unpack(blob, H)
    assert torch.equal(dequant(w_gu, s_gu, torch.float32), dequant(ws[0], ws[1], torch.float32))
    assert torch.equal(dequant_t(w_down, s_down, torch.float32), dequant_t(ws[2], ws[3], torch.float32))
    idx = torch.tensor([3, 0, 3])
    a = dequant(*mdd.unpack_gu(blob, H, idx), torch.float32)
    assert torch.equal(a, dequant(ws[0][idx], ws[1][idx], torch.float32))
    b = dequant_t(*mdd.unpack_down(blob, H, idx), torch.float32)
    assert torch.equal(b, dequant_t(ws[2][idx], ws[3][idx], torch.float32))


def test_pack_places_the_kernel_operands():
    """Spot-check the byte positions the kernel reads (module docstring)."""
    ws = experts()
    blob = mdd.pack(*ws)
    o1, o2, o3 = mdd._offsets(H)
    e, p, c, o = 2, 37, 3, 101
    sg = blob[e, :, o2:o3].contiguous().view(torch.bfloat16)  # [o, c]
    w = blob[e, p, c * 128 + o:c * 128 + o + 1].view(FP8).float() * sg[o, c].float()
    assert w == ws[0][e, o, c * 128 + p].float() * ws[1][e, o, (c * 128 + p) // 32].float()
    q, h = 64 + 5, 77  # second half of H on partitions 64..127
    hh = H // 2 + h  # its output column
    cp, pp = (hh % (H // 2)) // 128, hh % 128
    sd = blob[e, :, o3:].contiguous().view(torch.bfloat16)  # [p, (c', half)]
    wd = blob[e, q, o1 + h:o1 + h + 1].view(FP8).float() * sd[pp, 2 * cp + 1].float()
    assert wd == ws[2][e, 5, hh].float() * ws[3][e, 5 // 32, hh].float()


def test_rebase_moves_the_tile_scale_inside_e4m3():
    """A tile whose blocks are 12 binades apart is still exact: the largest block is shifted up
    (6 x 2^5 = 192 <= 240) so the smallest is not shifted off e4m3's 2^-9 grid; 14 apart is refused,
    and so are scales that are not powers of two."""
    w = torch.tensor([6.0, 0.5] * 16).view(1, 1, 1, 32).expand(1, 1, 4, 32).to(FP8)
    for spread, ok in ((8, True), (12, True), (14, False)):
        k = torch.tensor([[[0, -spread, -spread, -spread]]])
        if ok:
            out, K = mdd._rebase(w, k, "t")
            assert torch.equal(out.float() * torch.exp2(K.float()).view(1, 1, 1, 1),
                               w.float() * torch.exp2(k.float()).unsqueeze(-1))
        else:
            with pytest.raises(ValueError, match="too far apart"):
                mdd._rebase(w, k, "t")
    ws = list(experts())
    ws[1] = ws[1] * 1.5
    with pytest.raises(ValueError, match="power-of-two"):
        mdd.pack(*ws)


def check_plan(topv, topi, E, L):
    T, K = topi.shape
    slot_e, G, R = mdd.plan(topv, topi, E, L)
    S = slot_e.shape[1]
    SL = S * L
    S0, BL = mdd.n_slots(T, K, E, L)
    assert S == S0 and SL % BL == 0 and BL % L == 0
    assert G.shape == (T, SL) and R.shape == (BL, SL // BL, T)
    Rt = R.permute(1, 0, 2).reshape(SL, T).t().float()  # [T, SL] routing weight per lane
    assert torch.equal(Rt != 0, G != 0)
    Gi = G.to(torch.int64)
    assert torch.equal(Gi.sum(1), torch.full((T,), K))  # every pair has a lane
    assert int(Gi.sum(0).max()) <= 1  # a lane holds at most one pair
    se = slot_e.view(S).to(torch.int64)
    lane_e = se.repeat_interleave(L)  # expert of each lane's slot
    for t in range(T):
        lanes = torch.nonzero(Gi[t]).view(-1)
        assert sorted(lane_e[lanes].tolist()) == sorted(topi[t].tolist())
        for ln in lanes.tolist():
            k = int(torch.nonzero(topi[t] == lane_e[ln]).view(-1)[0])
            assert Rt[t, ln] == topv[t, k].float()
    real = int((se < E).sum())
    assert torch.all(se[:real] < E) and torch.all(se[real:] == E)  # compacted, padded = E
    kept = mdd.plan(topv, topi, E, L, keep=S)[0].view(S).to(torch.int64)
    assert torch.equal(kept[:real], se[:real]) and torch.all(kept[real:] == 0)  # padded below keep: expert 0
    # each expert's slots are consecutive, ceil(count / L) of them
    cnt = torch.bincount(topi.reshape(-1), minlength=E)
    assert torch.equal(torch.bincount(se[:real], minlength=E), (cnt + L - 1) // L)
    return real


@pytest.mark.parametrize("T,L", [(1, 1), (3, 1), (4, 2), (16, 2), (32, 4), (64, 4), (128, 8)])
def test_plan_assigns_every_pair_one_lane(T, L):
    x, topv, topi = routing(T)
    real = check_plan(topv, topi, 256, L)
    assert real >= len(set(topi.reshape(-1).tolist()))


@pytest.mark.parametrize("L", [1, 2, 4, 8])
def test_plan_worst_cases_fit_the_static_slot_count(L):
    E, K = 256, 8
    # every pair a different expert: the bound is tight
    topi = torch.arange(32 * K).view(32, K)
    assert check_plan(torch.ones(32, K).bfloat16(), topi, E, L) == 32 * K
    # every token the same experts: K experts x ceil(T / L) slots
    topi = torch.arange(K).repeat(48, 1)
    assert check_plan(torch.ones(48, K).bfloat16(), topi, E, L) == K * -(-48 // L)
    # more pairs than experts
    g = torch.Generator().manual_seed(3)
    topi = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(100)])
    check_plan(torch.rand(100, K, generator=g).bfloat16() + 0.1, topi, E, L)


def test_n_slots_matches_the_measured_shapes():
    assert mdd.n_slots(1, 8, 256, 1) == (8, 8)
    assert mdd.n_slots(3, 8, 256, 1) == (24, 24)
    assert mdd.n_slots(4, 8, 256, 1) == (32, 32)
    assert mdd.n_slots(16, 8, 256, 2) == (128, 128)
    assert mdd.n_slots(32, 8, 256, 4) == (256, 128)
    assert mdd.n_slots(64, 8, 256, 4) == (320, 128)
    assert mdd.n_slots(128, 8, 256, 8) == (352, 128)


@pytest.mark.parametrize("T", [1, 5, 16])
def test_emulated_kernel_matches_fp32(T):
    ws = experts()
    x, topv, topi = routing(T, K=4, E=6)
    ref = reference(x, topv, topi, *ws)
    got = mdd.emulate(x, topv, topi, mdd.pack(*ws)).float()
    assert (got - ref).abs().max() <= 0.01 * ref.abs().max()


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("T,L", [(2, 1), (7, 2), (5, 4)])
def test_nki_simulator_matches_emulation(T, L, monkeypatch):
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    E = 128  # the kernel plans over tiles of 128 experts
    ws = experts(h=4096, e=E)
    blob = mdd.pack(*ws)
    g = torch.Generator().manual_seed(2)
    x = torch.randn(T, 4096, generator=g).bfloat16()
    # experts drawn from 12 of them, so slots hold more than one lane; one token uses expert 127
    topi = torch.stack([torch.randperm(12, generator=g)[:8] for _ in range(T)]) * 10
    topi[0, 0] = 127
    topv = (torch.rand(T, 8, generator=g) + 0.1).bfloat16()
    args = mdd.kernel_inputs(x, topv, topi, blob, lanes=L)
    st = {k: args.pop(k) for k in ("lanes", "group", "ring", "slots", "block", "keep", "act", "limit", "alpha", "debug")}
    out = nki.simulate(mdd.kernel())(**args, **st)
    got = torch.as_tensor(out).float()
    want = mdd.emulate(x, topv, topi, blob).float()
    assert (got - want).abs().max() <= 0.01 * want.abs().max()


def fake_model(E=6):
    import types

    from kiln.models.decoder import DecoderForCausalLM

    m = DecoderForCausalLM.__new__(DecoderForCausalLM)
    torch.nn.Module.__init__(m)
    m.cfg = types.SimpleNamespace(hidden_size=H, num_experts=E, num_experts_per_tok=4)
    m.dtype, m.moe_inter = torch.bfloat16, 64
    return m


@pytest.mark.parametrize("pairs", [512, 0])  # gather path, dense every-expert path
def test_xla_paths_read_the_tile_blob_exactly(pairs):
    import types

    from kiln.models.decoder import DecoderForCausalLM

    m = fake_model()
    w_gu, s_gu, w_down, s_down = experts()
    nat = types.SimpleNamespace(moe_blob=False, down_t=True, w_gu=w_gu, w_gu_scale=s_gu, w_down=w_down,
                                w_down_scale=s_down)
    tiles = types.SimpleNamespace(moe_blob=True, moe_tiles=True, w_blob=mdd.pack(w_gu, s_gu, w_down, s_down))
    x, topv, topi = routing(6, K=4, E=6)
    old = DecoderForCausalLM.MOE_GATHER_MAX_PAIRS
    DecoderForCausalLM.MOE_GATHER_MAX_PAIRS = pairs
    try:
        a = m._moe_routed(nat, x, topv, topi)
        b = m._moe_routed(tiles, x, topv, topi)  # on CPU a blob layer takes the XLA path
    finally:
        DecoderForCausalLM.MOE_GATHER_MAX_PAIRS = old
    assert torch.equal(a, b)


@pytest.mark.parametrize("kernel", ["nki", "nki-dedupe"])
def test_loader_packs_a_quantized_checkpoint_into_tiles(tmp_path, kernel):
    """moe_kernel="nki" (= "nki-dedupe") through load_model on a MiMo-V2 checkpoint in the real quantized
    format (MXFP4 experts, so power-of-two scales), sized so its experts fit the kernel: every MoE
    layer holds the tile-layout w_blob, and the CPU (XLA) paths reading it match Hugging Face."""
    from tests.test_quant import quantized_mimo

    from kiln.models.loader import load_model

    dst, hf, mc = quantized_mimo(tmp_path, hidden_size=256, moe_intermediate_size=64)
    m = load_model(str(dst), mc, torch.float32, torch.device("cpu"), 512, keep_fp8=True, moe_kernel=kernel)
    for i, layer in enumerate(m.layers):
        assert layer.moe_blob == (i in mc.moe_layers) and layer.moe_tiles == layer.moe_blob
        if layer.moe_blob:
            assert tuple(layer.w_blob.shape) == (mc.num_experts, 128, mdd.blob_cols(256))
    ids = torch.randint(0, 384, (37,))
    with torch.no_grad():
        want = hf(ids.unsqueeze(0)).logits[0]
        assert (m.forward_logits(ids) - want).abs().max().item() < 1e-4


def test_unknown_moe_kernel_is_refused():
    from kiln.config import ModelConfig
    from kiln.models.decoder import DecoderForCausalLM

    cfg = ModelConfig(architecture="Qwen3MoeForCausalLM", vocab_size=64, hidden_size=64, intermediate_size=64,
                      num_layers=1, num_heads=2, num_kv_heads=2, head_dim=32, rms_norm_eps=1e-6,
                      rope_theta=1e4, max_position_embeddings=64, tie_word_embeddings=True, eos_token_ids=(0,),
                      num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32, moe_layers=(0,))
    with torch.device("meta"):
        with pytest.raises(ValueError, match="KILN_MOE_KERNEL=nki-dedupe"):
            DecoderForCausalLM(cfg, torch.bfloat16, 64, moe_kernel="nki-dedupe")  # hidden 64: no layout
        with pytest.raises(ValueError, match="moe_kernel must be"):
            DecoderForCausalLM(cfg, torch.bfloat16, 64, moe_kernel="nki-v9")


def test_emulated_pairs_kernel_matches_fp32():
    ws = experts()
    x, topv, topi = routing(5, K=4, E=6)
    ref = reference(x, topv, topi, *ws)
    got = mdd.emulate_pairs(x, topv, topi, mdd.pack(*ws)).float()
    assert (got - ref).abs().max() <= 0.01 * ref.abs().max()


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("T", [1, 3])
def test_nki_simulator_pairs_kernel_matches_emulation(T, monkeypatch):
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    E = 12
    ws = experts(h=4096, e=E)
    blob = mdd.pack(*ws)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(T, 4096, generator=g).bfloat16()
    topi = torch.stack([torch.randperm(E, generator=g)[:8] for _ in range(T)])
    topv = (torch.rand(T, 8, generator=g) + 0.1).bfloat16()
    args = mdd.pairs_inputs(x, topv, topi, blob)
    st = {k: args.pop(k) for k in ("group", "ring", "act", "limit", "alpha")}
    got = mdd.from_pairs(torch.as_tensor(nki.simulate(mdd.pairs_kernel())(**args, **st)), torch.float32)
    want = mdd.emulate_pairs(x, topv, topi, blob).float()
    assert (got - want).abs().max() <= 0.01 * want.abs().max()


def experts128(seed=0, e=6, h=H):
    """FP8 experts with 128 x 128 block scales as the loader keeps them (GLM-5.3-Flash, DeepSeek
    style): arbitrary finite e4m3 bytes, fp32 scales per row per 128 input columns (gate_up) and per
    output column over the rank's 64 input rows (down)."""
    g = torch.Generator().manual_seed(seed)

    def fp8(*s):  # finite e4m3 bytes (exponent bit 3 clear)
        return (torch.randint(0, 256, s, dtype=torch.uint8, generator=g) & 0xBF).view(FP8)

    return (fp8(e, 128, h), torch.rand(e, 128, h // 128, generator=g) * 0.02 + 0.005, fp8(e, 64, h),
            torch.rand(e, 1, h, generator=g) * 0.02 + 0.005)


def test_pack_takes_128_block_fp8_scales_as_they_are():
    ws = experts128()
    assert mdd.supports(*ws, down_t=True)
    blob = mdd.pack(*ws)
    assert tuple(blob.shape) == (6, 128, mdd.blob_cols(H, 4)) and mdd.scale_bytes(blob.shape[-1], H) == 4
    w_gu, s_gu, w_down, s_down = mdd.unpack(blob, H)
    assert s_gu.dtype == torch.float32 and tuple(s_gu.shape) == (6, 128, H // 128)
    assert torch.equal(dequant(w_gu, s_gu, torch.float32), dequant(ws[0], ws[1], torch.float32))
    assert torch.equal(dequant_t(w_down, s_down, torch.float32), dequant_t(ws[2], ws[3], torch.float32))
    assert not mdd.supports(ws[0], ws[1].bfloat16(), ws[2], ws[3], down_t=True)  # bf16 128-blocks: no layout


def reference_act(x, topv, topi, w_gu, s_gu, w_down, s_down, act, limit):
    wg, wd = dequant(w_gu, s_gu, torch.float32), dequant_t(w_down, s_down, torch.float32)
    out = torch.zeros(x.shape, dtype=torch.float32)
    for t in range(x.shape[0]):
        for k in range(topi.shape[1]):
            e = int(topi[t, k])
            gu = wg[e] @ x[t].float()
            out[t] += topv[t, k].float() * (mdd.glu(gu[:64], gu[64:], act, limit) @ wd[e])
    return out


@pytest.mark.parametrize("act,limit", [(1, 10.0), (1, 0.05), (2, 7.0), (2, 0.05)])
@pytest.mark.parametrize("which", [experts, experts128])
def test_emulated_activations_match_fp32(act, limit, which):
    ws = which()
    x, topv, topi = routing(5, K=4, E=6)
    ref = reference_act(x, topv, topi, *ws, act, limit)
    for got in (mdd.emulate(x, topv, topi, mdd.pack(*ws), act, limit),
                mdd.emulate_pairs(x, topv, topi, mdd.pack(*ws), act, limit)):
        assert (got.float() - ref).abs().max() <= 0.01 * ref.abs().max()


def test_glu_forms():
    import torch.nn.functional as F

    g, u = torch.tensor([-20.0, 0.5, 30.0]), torch.tensor([-30.0, 0.25, 20.0])
    assert torch.equal(mdd.glu(g, u, 0), F.silu(g) * u)
    assert torch.equal(mdd.glu(g, u, 1, 10.0), F.silu(torch.tensor([-20.0, 0.5, 10.0])) * torch.tensor([-10.0, 0.25, 10.0]))
    gc, uc = torch.tensor([-20.0, 0.5, 7.0]), torch.tensor([-7.0, 0.25, 7.0])
    assert torch.equal(mdd.glu(g, u, 2, 7.0), (uc + 1) * (gc * torch.sigmoid(gc * 1.702)))


def test_plan_over_288_experts():
    x, topv, topi = routing(32, E=288, seed=5)
    topi[0, :2] = torch.tensor([287, 257])  # ids past 256
    check_plan(topv, topi, 288, 4)


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("act,limit", [(1, 0.02), (2, 0.02)])
def test_nki_simulator_activations_128_blocks_and_160_experts(act, limit, monkeypatch):
    """The dedupe kernel (16 tokens, lanes 2) and the per-pair kernel (3 tokens) on fp32 128-block
    scales, clamped activations whose limit bites, 160 experts (a partial tile of 128)."""
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    E = 160
    ws = experts128(e=E, h=4096)
    blob = mdd.pack(*ws)
    g = torch.Generator().manual_seed(6)
    for T in (16, 3):
        x = torch.randn(T, 4096, generator=g).bfloat16()
        topi = torch.stack([torch.cat([torch.randperm(10, generator=g)[:6], torch.tensor([150, 159])])
                            for _ in range(T)])
        topv = (torch.rand(T, 8, generator=g) + 0.1).bfloat16()
        if T >= mdd.PAIRS_BELOW:
            args = mdd.kernel_inputs(x, topv, topi, blob, lanes=2, act=act, limit=limit)
            st = {k: args.pop(k) for k in ("lanes", "group", "ring", "slots", "block", "keep", "act", "limit", "alpha",
                                           "debug")}
            got = torch.as_tensor(nki.simulate(mdd.kernel())(**args, **st)).float()
            want = mdd.emulate(x, topv, topi, blob, act, limit).float()
        else:
            args = mdd.pairs_inputs(x, topv, topi, blob, act, limit)
            st = {k: args.pop(k) for k in ("group", "ring", "act", "limit", "alpha")}
            got = mdd.from_pairs(torch.as_tensor(nki.simulate(mdd.pairs_kernel())(**args, **st)), torch.float32)
            want = mdd.emulate_pairs(x, topv, topi, blob, act, limit).float()
        assert (got - want).abs().max() <= 0.01 * want.abs().max()


def test_clamped_moe_reads_a_128_block_tile_blob_exactly():
    """models/hybrid.py _moe_clamped (GLM-5.3-Flash's MoE) on a tile-layout layer of FP8 experts
    with 128 x 128 block scales equals the natural layer on the CPU (XLA) path; on the device the
    same layer runs the NKI kernel with act silu_clamp."""
    import types

    from kiln.models import hybrid

    m = fake_model()
    m.cfg.router_scoring = "softmax"
    m.cfg.norm_topk_prob = True
    w_gu, s_gu, w_down, s_down = experts128()
    router = torch.randn(6, H, generator=torch.Generator().manual_seed(9)).bfloat16()
    nat = types.SimpleNamespace(moe_blob=False, down_t=True, w_gu=w_gu, w_gu_scale=s_gu, w_down=w_down,
                                w_down_scale=s_down, router=router, router_bias=None)
    tiles = types.SimpleNamespace(moe_blob=True, moe_tiles=True, w_blob=mdd.pack(w_gu, s_gu, w_down, s_down),
                                  router=router, router_bias=None)
    x = routing(5, K=4, E=6)[0]
    a = hybrid._moe_clamped(m, nat, x, 0.05)
    b = hybrid._moe_clamped(m, tiles, x, 0.05)
    assert torch.equal(a, b)
