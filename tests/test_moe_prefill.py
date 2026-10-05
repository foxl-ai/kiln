"""The NKI prefill-MoE kernel's CPU-checkable parts (kiln/kernels/moe_prefill.py): the routing plan
against a sort-and-loop reference, the kernel's arithmetic (moe_dedupe.emulate on the tiles blob)
against an fp32 reference and against the XLA paths of the decoder, the blob check, and (where the
NKI package is installed) the kernel itself under the NKI CPU simulator. The device run is
tools/probe_moe_prefill.py."""

import importlib.util
import types

import pytest
import torch

from kiln.kernels import moe_dedupe as mdd
from kiln.kernels import moe_prefill as mp
from kiln.models.quant import FP8


def fp8(g, *s):  # finite e4m3 bytes (exponent bit 3 clear), as tools/profile_layer.py
    return (torch.randint(0, 256, s, dtype=torch.uint8, generator=g) & 0xBF).view(FP8)


def experts(fmt: str = "glm", E=6, H=1024, seed=0):
    """fmt "glm": fp32 128 x 128 block scales as GLM-5.3-Flash loads them at tp=32 (gate and up
    rows each inside one row block, down one block of 64 input rows per 128 output columns);
    "mimo": bf16 power-of-two scales per 32 input columns (MXFP4-derived)."""
    g = torch.Generator().manual_seed(seed)
    w_gu, w_down = fp8(g, E, 128, H), fp8(g, E, 64, H)
    if fmt == "glm":
        s_gu = (torch.rand(E, 2, H // 128, generator=g) * 0.02 + 0.005).repeat_interleave(64, dim=1)
        s_down = (torch.rand(E, 1, H // 128, generator=g) * 0.02 + 0.005).repeat_interleave(128, dim=2)
    else:
        s_gu = torch.exp2(torch.randint(-9, -6, (E, 128, H // 32), generator=g).float()).bfloat16()
        s_down = torch.exp2(torch.randint(-9, -6, (E, 2, H), generator=g).float()).bfloat16()
    return w_gu, s_gu, w_down, s_down


def checkpoint_experts(E=6, H=1024, Im=64, seed=0):
    """The routed experts as the loader stores GLM-5.3-Flash's for one rank at tp=32 (layout
    measured on the real checkpoint by tools/check_moe_prefill_layout.py, 2026-10-04): checkpoint
    weights quantized per 128 x 128 block to e4m3fn (largest value of a block 448, as the
    checkpoint's), the rank's Im = 64 rows of gate_proj and of up_proj and its 64 columns of down_proj
    each inside one block, scales expanded per row (loader._row_scales), and models/quant.fit_e4m3_max
    applied as loader._assign does (trn1's e4m3 stops at 240: each (row, block) whose largest code
    exceeds 240 is halved and its scale doubled). Returns (w_gu, s_gu, w_down, s_down) in the
    model's layout and the checkpoint's dequantized values (gate_up [E, 2 Im, H], down [E, H, Im])."""
    from kiln.models.quant import fit_e4m3_max

    g = torch.Generator().manual_seed(seed)
    C = H // 128

    def blocks(rows, cols, shard_rows, shard_cols):
        """A [rows, cols] checkpoint weight quantized per 128 x 128 block, its shard and row scales."""
        w = torch.randn(rows, cols, generator=g) * 0.02
        bs = w.abs().view(rows // 128, 128, cols // 128, 128).amax((1, 3)) / 448.0  # weight_scale_inv
        codes = (w / bs.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(FP8)
        cw = codes[shard_rows, shard_cols]
        rs = bs.repeat_interleave(128, 0)[shard_rows]  # [n, k / 128] per row
        if shard_cols.stop - shard_cols.start < 128:  # inside one block: that block's scale
            rs = rs[:, shard_cols.start // 128:shard_cols.start // 128 + 1]
        true = cw.float() * rs.repeat_interleave(min(128, cw.shape[1]), 1)[:, :cw.shape[1]]
        return fit_e4m3_max(cw, rs, 240.0), true

    gus, gss, ds, dss, tgu, td = [], [], [], [], [], []
    for e in range(E):
        (wg, sg), tg = blocks(128, H, slice(0, Im), slice(0, H))  # the rank's gate rows of a 2 x Im block
        (wu, su), tu = blocks(128, H, slice(Im, 2 * Im), slice(0, H))
        (wd, sd), tdd = blocks(H, 128, slice(0, H), slice(0, Im))  # down [H, I]: the rank's Im columns
        gus.append(torch.cat([wg, wu]))
        gss.append(torch.cat([sg, su]))
        ds.append(wd.T.contiguous())
        dss.append(sd.T.contiguous())
        tgu.append(torch.cat([tg, tu]))
        td.append(tdd)
    return (torch.stack(gus), torch.stack(gss), torch.stack(ds), torch.stack(dss)), (torch.stack(tgu), torch.stack(td))


def routing(T, E=6, K=4, seed=1, H=1024, skew=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, H, generator=g).bfloat16()
    if skew:  # every token picks expert 0, its other experts from the rest
        rest = torch.stack([torch.randperm(E - 1, generator=g)[: K - 1] + 1 for _ in range(T)])
        topi = torch.cat([torch.zeros(T, 1, dtype=torch.long), rest], dim=1)
    else:
        topi = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])
    topv = (torch.rand(T, K, generator=g) + 0.1).bfloat16()
    return x, topv, topi


def reference(x, topv, topi, w_gu, s_gu, w_down, s_down, lim=None):
    """fp32 everywhere: dequantize, matvec, (clamped) SwiGLU, matvec, weight, sum."""
    from kiln.models.quant import dequant, dequant_t

    out = torch.zeros(x.shape, dtype=torch.float32)
    wg, wd = dequant(w_gu, s_gu, torch.float32), dequant_t(w_down, s_down, torch.float32)
    for t in range(x.shape[0]):
        for k in range(topi.shape[1]):
            e = int(topi[t, k])
            gu = wg[e] @ x[t].float()
            gate, up = gu[:64], gu[64:]
            if lim is not None:
                gate, up = gate.clamp(max=lim), up.clamp(min=-lim, max=lim)
            out[t] += topv[t, k].float() * ((torch.nn.functional.silu(gate) * up) @ wd[e])
    return out


@pytest.mark.parametrize("T,E,K,B", [(128, 6, 4, 64), (256, 6, 4, 128), (256, 288, 8, 64), (512, 16, 8, 128)])
@pytest.mark.parametrize("skew", [False, True])
def test_plan_matches_the_reference(T, E, K, B, skew):
    g = torch.Generator().manual_seed(T + E)
    if skew:
        topi = torch.stack([torch.cat([torch.zeros(1, dtype=torch.long), torch.randperm(E - 1, generator=g)[: K - 1] + 1])
                            for _ in range(T)])
    else:
        topi = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])
    got, want = mp.plan(topi, E, B), mp.plan_reference(topi, E, B)
    for a, b in zip(got, want):
        assert a.dtype == torch.int32 and torch.equal(a, b)
    slot, tpos, bexp = got
    NB = bexp.shape[1]
    # every pair has its own slot, inside a block of its own expert; the static block count covers
    # the routing in whole 128-lane tiles; tpos indexes the kernel's [lane in tile, tile] table
    assert len(set(slot.flatten().tolist())) == T * K
    assert torch.equal(bexp[0][(slot // B).long()], topi.to(torch.int32))
    assert NB == mp.n_blocks(T * K, E, B) and (NB * B) % 128 == 0
    NTL = NB * B // 128
    assert torch.equal(tpos, (slot % 128) * NTL + slot // 128)


def test_plan_is_exact_at_8192_tokens():
    """Counts past 256 (bf16's exact integers) and 2^16: the largest prefill bucket at GLM-5.3-Flash's
    288 experts, skewed so one expert holds every token."""
    g = torch.Generator().manual_seed(7)
    T, E, K, B = 8192, 288, 8, 128
    topi = torch.stack([torch.randperm(E - 1, generator=g)[:K] + 1 for _ in range(T)])
    topi[:, 0] = 0
    for a, b in zip(mp.plan(topi, E, B), mp.plan_reference(topi, E, B)):
        assert torch.equal(a, b)


def test_plan_fits_the_worst_routing():
    """Every token on the same k experts: the static block count still holds every pair."""
    T, E, K, B = 256, 288, 8, 64
    topi = torch.arange(K).repeat(T, 1)
    slot, _, bexp = mp.plan(topi, E, B)
    assert int(slot.max()) < bexp.shape[1] * B
    assert torch.equal(bexp[0][(slot // B).long()], topi.to(torch.int32))


@pytest.mark.parametrize("act,lim", [(0, None), (1, 10.0), (1, 0.05)])
@pytest.mark.parametrize("dq", [False, True])
def test_emulation_matches_fp32(act, lim, dq):
    ws = experts("glm")
    blob = mdd.pack(*ws)
    x, topv, topi = routing(24)
    ref = reference(x, topv, topi, *ws, lim=lim)
    got = mp.emulate(x, topv, topi, blob, act, lim or 0.0, dq=dq).float()
    assert (got - ref).abs().max() <= 0.01 * ref.abs().max()


@pytest.mark.parametrize("act,lim", [(0, 0.0), (1, 0.05)])
def test_emulation_is_the_dedupe_arithmetic(act, lim):
    """emulate (one expert at a time) equals moe_dedupe.emulate(pair_bf16=True) (every pair at once)
    up to fp32 summation order."""
    blob = mdd.pack(*experts("glm"))
    x, topv, topi = routing(24)
    a = mp.emulate(x, topv, topi, blob, act, lim).float()
    b = mdd.emulate(x, topv, topi, blob, act, lim, pair_bf16=True).float()
    assert (a - b).abs().max() <= 1e-3 * b.abs().max()


def _model(E, K, H):
    from kiln.models.decoder import DecoderForCausalLM

    m = DecoderForCausalLM.__new__(DecoderForCausalLM)
    torch.nn.Module.__init__(m)
    m.cfg = types.SimpleNamespace(hidden_size=H, num_experts=E, num_experts_per_tok=K)
    m.dtype, m.moe_inter = torch.bfloat16, 64
    return m


@pytest.mark.parametrize("pairs", [512, 0])  # the XLA gather path, the every-expert path
def test_emulation_matches_the_xla_paths(pairs):
    """Within bf16 rounding noise of DecoderForCausalLM._moe_routed on the natural tensors."""
    from kiln.models.decoder import DecoderForCausalLM

    w_gu, s_gu, w_down, s_down = experts("glm")
    x, topv, topi = routing(24)
    m = _model(6, 4, 1024)
    layer = types.SimpleNamespace(moe_blob=False, down_t=True, w_gu=w_gu, w_gu_scale=s_gu, w_down=w_down,
                                  w_down_scale=s_down)
    old = DecoderForCausalLM.MOE_GATHER_MAX_PAIRS
    DecoderForCausalLM.MOE_GATHER_MAX_PAIRS = pairs
    try:
        xla = m._moe_routed(layer, x, topv, topi).float()
    finally:
        DecoderForCausalLM.MOE_GATHER_MAX_PAIRS = old
    ref = reference(x, topv, topi, w_gu, s_gu, w_down, s_down)
    got = mp.emulate(x, topv, topi, mdd.pack(w_gu, s_gu, w_down, s_down)).float()
    scale = ref.abs().max()
    assert (got - ref).abs().max() <= 0.01 * scale
    assert (got - xla).abs().max() <= 0.01 * scale


def test_check_blob():
    assert mp.check_blob(mdd.pack(*experts("glm")), 1024)  # gate rows and up rows each share a scale
    w_gu, s_gu, w_down, s_down = experts("glm")
    s_gu = s_gu.clone()
    s_gu[1, 3, 0] *= 2  # one gate row with a scale of its own: the per-row path
    assert not mp.check_blob(mdd.pack(w_gu, s_gu, w_down, s_down), 1024)
    assert not mp.check_blob(mdd.pack(*experts("mimo")), 1024)  # bf16 scales: per row and per column
    assert mp.down_factors(mdd.pack(*experts("glm")), 1024) is None  # one down scale per chunk
    w_gu, s_gu, w_down, s_down = experts("glm")
    s_down = s_down.clone()
    s_down[2, 0, 5] *= 2  # one column at twice its chunk's scale: the per-column path, exact
    dsc, dfr = mp.down_factors(mdd.pack(w_gu, s_gu, w_down, s_down), 1024)
    k = torch.arange(8)
    assert torch.equal(dsc[:, 2 * (k % 4) + k // 4].repeat_interleave(128, 1) * dfr.float(), s_down[:, 0])
    assert sorted(set(dfr.float().flatten().tolist())) == [1.0, 2.0]
    s_down[3, 0, 7] *= 1.5  # not a power of two times its chunk's smallest
    with pytest.raises(ValueError, match="power of two"):
        mp.down_factors(mdd.pack(w_gu, s_gu, w_down, s_down), 1024)


def test_the_loaded_glm_layout():
    """GLM-5.3-Flash's experts as the loader leaves them (checkpoint_experts): fit_e4m3_max doubled
    the scales of most rows, so the rank's gate rows no longer share a tile scale (the per-row path)
    and the down scales differ within a chunk by powers of two (down_factors). The kernel's
    arithmetic (emulate) on them is within bf16 noise of the fp32 reference on the loaded weights
    and on the checkpoint's own values (fit_e4m3_max is exact for normal e4m3 values)."""
    from kiln.models.quant import dequant, dequant_t

    ws, (true_gu, true_down) = checkpoint_experts()
    w_gu, s_gu, w_down, s_down = ws
    doubled = (s_down[:, 0].view(6, 8, 128) / s_down[:, 0].view(6, 8, 128).amin(-1, keepdim=True)) == 2
    assert 0.2 < float(doubled.float().mean()) < 0.98  # the real checkpoint: 2687 of 4096 rows (expert 0)
    blob = mdd.pack(*ws)
    assert not mp.check_blob(blob, 1024)
    down = mp.down_factors(blob, 1024)
    assert down is not None and sorted(set(down[1].float().flatten().tolist())) == [1.0, 2.0]
    # the loaded weights dequantize to the checkpoint's values but for halved subnormal codes
    dg, dd = dequant(w_gu, s_gu, torch.float32), dequant_t(w_down, s_down, torch.float32)
    assert (dg - true_gu).abs().max() <= 2.0 ** -10 * s_gu.max()
    assert (dd - true_down.transpose(1, 2)).abs().max() <= 2.0 ** -10 * s_down.max()
    for act, lim in [(0, None), (1, 0.05)]:
        x, topv, topi = routing(24)
        got = mp.emulate(x, topv, topi, blob, act, lim or 0.0).float()
        ref = reference(x, topv, topi, *ws, lim=lim)
        assert (got - ref).abs().max() <= 0.01 * ref.abs().max()


def test_prefill_knob_needs_the_tiles_kernel(monkeypatch):
    """KILN_MOE_PREFILL_KERNEL=nki reads the dedupe kernel's tiles blob: any other moe_kernel is
    refused when the model is built, not at the first prefill."""
    from kiln.config import ModelConfig
    from kiln.models import decoder

    monkeypatch.setattr(decoder, "MOE_PREFILL_KERNEL", "nki")
    cfg = ModelConfig(architecture="Qwen3MoeForCausalLM", vocab_size=64, hidden_size=64, intermediate_size=64,
                      num_layers=1, num_heads=2, num_kv_heads=2, head_dim=32, rms_norm_eps=1e-6,
                      rope_theta=1e4, max_position_embeddings=64, tie_word_embeddings=True, eos_token_ids=(0,),
                      num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32, moe_layers=(0,))
    with torch.device("meta"):
        with pytest.raises(ValueError, match="KILN_MOE_PREFILL_KERNEL=nki reads the tiles blob"):
            decoder.DecoderForCausalLM(cfg, torch.bfloat16, 64, moe_kernel="xla")


@pytest.mark.parametrize("fmt", ["glm", "glm-rows", "mimo", "loaded"])
def test_pack_experts_records_the_prefill_path(fmt, monkeypatch):
    """With the knob on, DecoderLayer.pack_experts checks the blob once and records the kernel path
    the layer's prefill calls take (decoder._moe_routed, hybrid._moe_clamped pass dq and
    moe_prefill_down). loaded: GLM-5.3-Flash's experts as the loader leaves them."""
    from kiln.models import decoder

    monkeypatch.setattr(decoder, "MOE_PREFILL_KERNEL", "nki")
    if fmt == "loaded":
        w_gu, s_gu, w_down, s_down = checkpoint_experts()[0]
    else:
        w_gu, s_gu, w_down, s_down = experts("mimo" if fmt == "mimo" else "glm")
    if fmt == "glm-rows":
        s_gu = s_gu * (1 + torch.rand(s_gu.shape, generator=torch.Generator().manual_seed(3)))
    layer = types.SimpleNamespace(pack_tiles=True, w_gu=w_gu, w_gu_scale=s_gu, w_down=w_down, w_down_scale=s_down)
    decoder.DecoderLayer.pack_experts(layer)
    assert layer.moe_tiles and layer.w_gu is None
    assert layer.moe_prefill_dq == (fmt == "glm")
    if fmt in ("mimo", "loaded"):  # per-column down scales: chunk scales times powers of two
        dsc, dfr = decoder.moe_prefill_down(layer)
        assert dsc.dtype == torch.float32 and dfr.dtype == torch.bfloat16 and tuple(dfr.shape) == (6, 1024)
        k = torch.arange(8)
        assert torch.equal(dsc[:, 2 * (k % 4) + k // 4].repeat_interleave(128, 1) * dfr.float(),
                           mdd.unpack_down(layer.w_blob.data, 1024)[1][:, 0, :].float())
    else:
        assert decoder.moe_prefill_down(layer) is None


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("act,lim,skew,B,fmt", [(0, 0.0, False, 64, "dq"), (1, 0.05, True, 64, "dq"),
                                               (0, 0.0, False, 128, "dq"), (1, 0.05, False, 64, "rows"),
                                               (0, 0.0, True, 128, "rows"), (0, 0.0, False, 64, "mimo"),
                                               (1, 0.05, True, 128, "mimo"), (1, 0.05, False, 64, "loaded"),
                                               (0, 0.0, True, 128, "loaded"), (1, 10.0, True, 64, "loaded")])
def test_nki_simulator_matches_emulation(act, lim, skew, B, fmt, monkeypatch):
    """fmt dq: GLM-format scales (the dequantize-first path); rows: fp32 per-row gate_up scales;
    mimo: bf16 scales (per row and tile for gate_up, per output column for down); loaded:
    GLM-5.3-Flash's experts as the loader leaves them (checkpoint_experts: per-row gate_up scales and
    per-column down scales, chunk scale times 1 or 2)."""
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    if fmt == "loaded":
        w_gu, s_gu, w_down, s_down = checkpoint_experts()[0]
    else:
        w_gu, s_gu, w_down, s_down = experts("mimo" if fmt == "mimo" else "glm")
    if fmt == "rows":  # per-row gate_up scales
        s_gu = s_gu * (1 + torch.rand(s_gu.shape, generator=torch.Generator().manual_seed(5)))
    blob = mdd.pack(w_gu, s_gu, w_down, s_down)
    dq = fmt == "dq"
    assert mp.check_blob(blob, 1024) == dq
    down = mp.down_factors(blob, 1024)
    assert (down is not None) == (fmt in ("mimo", "loaded"))
    x, topv, topi = routing(256, skew=skew)
    args = mp.kernel_inputs(x, topv, topi, blob, act, lim, B=B, dq=dq, down=down)
    out = nki.simulate(mp.kernel())(**args)
    got = torch.as_tensor(out).float()
    want = mp.emulate(x, topv, topi, blob, act, lim, dq=dq).float()
    assert (got - want).abs().max() <= 0.01 * want.abs().max()


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("B,fmt,order,nyb", [(128, "loaded", 1, 3), (64, "loaded", 1, 3), (128, "mimo", 0, 3),
                                             (128, "loaded", 1, 0), (128, "dq", 1, 2), (64, "rows", 0, 2)])
def test_nki_simulator_round_order_and_output_ring(B, fmt, order, nyb, monkeypatch):
    """The order of each pipeline round's instructions (kernel argument ord_) and the ring of stage C
    output buffers (nyb) leave the kernel's output as it was, bit for bit: the same operations reach
    each accumulator in the same order."""
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    if fmt == "loaded":
        w_gu, s_gu, w_down, s_down = checkpoint_experts()[0]
    else:
        w_gu, s_gu, w_down, s_down = experts("mimo" if fmt == "mimo" else "glm")
    if fmt == "rows":
        s_gu = s_gu * (1 + torch.rand(s_gu.shape, generator=torch.Generator().manual_seed(5)))
    blob = mdd.pack(w_gu, s_gu, w_down, s_down)
    dq = fmt == "dq"
    down = mp.down_factors(blob, 1024)
    x, topv, topi = routing(256, skew=True)
    base = mp.kernel_inputs(x, topv, topi, blob, 1, 0.05, B=B, dq=dq, down=down, order=0, nyb=0)
    var = mp.kernel_inputs(x, topv, topi, blob, 1, 0.05, B=B, dq=dq, down=down, order=order, nyb=nyb)
    a = torch.as_tensor(nki.simulate(mp.kernel())(**base))
    b = torch.as_tensor(nki.simulate(mp.kernel())(**var))
    assert torch.equal(a, b)
    want = mp.emulate(x, topv, topi, blob, 1, 0.05, dq=dq).float()
    assert (b.float() - want).abs().max() <= 0.01 * want.abs().max()
