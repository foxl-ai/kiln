"""The NKI decode-MoE kernel's CPU-checkable parts (kiln/kernels/moe_decode.py): the expert blob
layout, the kernel's arithmetic emulated in torch, the XLA paths reading the blob, the gating
knobs, and (where the NKI package is installed) the kernel itself under the NKI CPU simulator.
The device run is tools/probe_moe_kernel.py."""

import importlib.util
import os
import types

import pytest
import torch

from kiln.kernels import moe_decode as md
from kiln.models.quant import FP8, dequant, dequant_t

E, H = 5, 512


def experts(seed=0, e=E, h=H):
    g = torch.Generator().manual_seed(seed)

    def fp8(*s):  # finite e4m3 bytes (exponent bit 3 clear), as tools/profile_layer.py
        return (torch.randint(0, 256, s, dtype=torch.uint8, generator=g) & 0xBF).view(FP8)

    w_gu, w_down = fp8(e, 128, h), fp8(e, 64, h)
    s_gu = (torch.rand(e, 128, h // 32, generator=g) * 0.02 + 0.005).bfloat16()
    s_down = (torch.rand(e, 2, h, generator=g) * 0.02 + 0.005).bfloat16()
    return w_gu, s_gu, w_down, s_down


def bits(t):
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(torch.int16)


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


def routing(T, K=4, seed=1, e=E):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, H, generator=g).bfloat16()
    topi = torch.stack([torch.randperm(e, generator=g)[:K] for _ in range(T)])
    topv = (torch.rand(T, K, generator=g) + 0.1).bfloat16()
    return x, topv, topi


def test_pack_is_an_exact_byte_permutation():
    ws = experts()
    assert md.supports(*ws, down_t=True)
    blob = md.pack(*ws)
    assert blob.dtype == torch.uint8 and tuple(blob.shape) == (E, 128, md.blob_cols(H))
    assert blob.numel() == sum(t.numel() * t.element_size() for t in ws)  # no padding
    for a, b in zip(md.unpack(blob, H), ws):
        assert a.shape == b.shape and a.dtype == b.dtype and torch.equal(bits(a), bits(b))
    idx = torch.tensor([3, 0, 3])
    for a, b in zip((*md.unpack_gu(blob, H, idx), *md.unpack_down(blob, H, idx)), ws):
        assert torch.equal(bits(a), bits(b[idx]))


def test_pack_places_the_kernel_operands():
    """Spot-check the byte positions the kernel reads (module docstring)."""
    w_gu, s_gu, w_down, s_down = experts()
    blob = md.pack(w_gu, s_gu, w_down, s_down)
    e, p, c, o = 2, 37, 3, 101
    assert blob[e, p, c * 128 + o] == w_gu[e, o, c * 128 + p].view(torch.uint8)
    q, h = 64 + 5, 77  # second half of H on partitions 64..127
    assert blob[e, q, H + h] == w_down[e, 5, H // 2 + h].view(torch.uint8)
    o1, o2, o3 = H, H + H // 2, H + H // 2 + (H // 32) * 2
    assert torch.equal(blob[e, :, o2:o3].contiguous().view(torch.bfloat16), s_gu[e])
    sd = blob[e, :, o3:].contiguous().view(torch.bfloat16)  # [p, (c', half, j)]
    cp, half, j = 1, 1, 0
    assert sd[p, cp * 4 + half * 2 + j] == s_down[e, j, half * (H // 2) + cp * 128 + p]


def test_supports_only_the_kernel_layout():
    w_gu, s_gu, w_down, s_down = experts()
    assert not md.supports(w_gu, s_gu.float(), w_down, s_down, down_t=True)  # fp32 scales
    assert not md.supports(w_gu, s_gu, w_down, s_down, down_t=False)
    assert not md.supports(w_gu.float(), s_gu, w_down, s_down, down_t=True)
    assert not md.supports(*experts(h=384), down_t=True)  # H not a multiple of 256


@pytest.mark.parametrize("T", [1, 3, 8])
def test_emulated_kernel_matches_fp32(T):
    ws = experts()
    x, topv, topi = routing(T)
    ref = reference(x, topv, topi, *ws)
    got = md.emulate(x, topv, topi, md.pack(*ws)).float()
    assert (got - ref).abs().max() <= 0.01 * ref.abs().max()


def fake_model():
    from kiln.models.decoder import DecoderForCausalLM

    m = DecoderForCausalLM.__new__(DecoderForCausalLM)
    torch.nn.Module.__init__(m)
    m.cfg = types.SimpleNamespace(hidden_size=H, num_experts=E, num_experts_per_tok=4)
    m.dtype, m.moe_inter = torch.bfloat16, 64
    return m


def layers():
    w_gu, s_gu, w_down, s_down = experts()
    nat = types.SimpleNamespace(moe_blob=False, down_t=True, w_gu=w_gu, w_gu_scale=s_gu, w_down=w_down,
                                w_down_scale=s_down)
    blob = types.SimpleNamespace(moe_blob=True, w_blob=md.pack(w_gu, s_gu, w_down, s_down))
    return nat, blob


@pytest.mark.parametrize("pairs", [512, 0])  # gather path, dense every-expert path
def test_xla_paths_read_the_blob_exactly(pairs):
    from kiln.models.decoder import DecoderForCausalLM

    m, (nat, blob) = fake_model(), layers()
    x, topv, topi = routing(6)
    old = DecoderForCausalLM.MOE_GATHER_MAX_PAIRS
    DecoderForCausalLM.MOE_GATHER_MAX_PAIRS = pairs
    try:
        a = m._moe_routed(nat, x, topv, topi)
        b = m._moe_routed(blob, x, topv, topi)  # on CPU a blob layer takes the XLA path
    finally:
        DecoderForCausalLM.MOE_GATHER_MAX_PAIRS = old
    assert torch.equal(a, b)


def test_knobs(monkeypatch):
    from kiln.config import EngineConfig
    from kiln.engine.model_runner import piecewise_groups

    assert EngineConfig(model_path="x").moe_kernel == os.environ.get("KILN_MOE_KERNEL", "xla")
    monkeypatch.setenv("KILN_MOE_KERNEL", "nki")
    monkeypatch.setenv("KILN_PIECEWISE_MOE_GROUP", "6")
    c = EngineConfig(model_path="x")
    assert c.moe_kernel == "nki" and c.piecewise_moe_group == 6
    assert c.piecewise_prefill_moe_group is None  # prefill follows decode unless set
    monkeypatch.setenv("KILN_PIECEWISE_PREFILL_MOE_GROUP", "1")
    assert EngineConfig(model_path="x").piecewise_prefill_moe_group == 1
    # what _piecewise runs with moe_group 6 on MiMo-V2.6-Flash's 48 layers (dense layer 0)
    assert [len(r) for r in piecewise_groups(48, 6)] == [6] * 8


def test_unsupported_layout_fails_loudly():
    """moe_kernel="nki" on experts the kernel cannot pack raises at construction (the tiny
    MiMo-V2 test config: hidden 64), instead of silently running the XLA path."""
    from kiln.config import ModelConfig
    from kiln.models.decoder import DecoderForCausalLM

    cfg = ModelConfig(architecture="Qwen3MoeForCausalLM", vocab_size=64, hidden_size=64, intermediate_size=64,
                      num_layers=1, num_heads=2, num_kv_heads=2, head_dim=32, rms_norm_eps=1e-6,
                      rope_theta=1e4, max_position_embeddings=64, tie_word_embeddings=True, eos_token_ids=(0,),
                      num_experts=4, num_experts_per_tok=2,
                      moe_intermediate_size=32, moe_layers=(0,))
    with torch.device("meta"):
        DecoderForCausalLM(cfg, torch.bfloat16, 64)  # the XLA path takes any layout
        with pytest.raises(ValueError, match="KILN_MOE_KERNEL=nki"):
            DecoderForCausalLM(cfg, torch.bfloat16, 64, moe_kernel="nki")


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("T", [1, 2])
def test_nki_simulator_matches_emulation(T, monkeypatch):
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    ws = experts(h=4096, e=6)
    blob = md.pack(*ws)
    g = torch.Generator().manual_seed(2)
    x = torch.randn(T, 4096, generator=g).bfloat16()
    topi = torch.stack([torch.randperm(6, generator=g)[:4] for _ in range(T)])
    topv = (torch.rand(T, 4, generator=g) + 0.1).bfloat16()
    args = md.kernel_inputs(x, topv, topi, blob)
    im = args.pop("im")
    out = nki.simulate(md.kernel())(**args, im=im)
    got = md.from_kernel(torch.as_tensor(out), torch.float32)
    want = md.emulate(x, topv, topi, blob).float()
    assert (got - want).abs().max() <= 0.01 * want.abs().max()


def test_loader_packs_a_quantized_checkpoint(tmp_path):
    """moe_kernel="nki-pair" through load_model on a MiMo-V2 checkpoint in the real quantized
    format, sized so its experts fit the kernel (hidden 256, 2 x 64 gate/up rows at tp=1): every MoE
    layer holds only w_blob, and the CPU (XLA) paths reading it match the Hugging Face reference."""
    from tests.test_quant import quantized_mimo

    from kiln.models.loader import load_model

    dst, hf, mc = quantized_mimo(tmp_path, hidden_size=256, moe_intermediate_size=64)
    m = load_model(str(dst), mc, torch.float32, torch.device("cpu"), 512, keep_fp8=True, moe_kernel="nki-pair")
    for i, layer in enumerate(m.layers):
        assert layer.moe_blob == (i in mc.moe_layers) and not layer.moe_tiles
        if layer.moe_blob:
            assert layer.w_gu is None and layer.w_down is None
            assert tuple(layer.w_blob.shape) == (mc.num_experts, 128, md.blob_cols(256))
    ids = torch.randint(0, 384, (37,))
    with torch.no_grad():
        want = hf(ids.unsqueeze(0)).logits[0]
        assert (m.forward_logits(ids) - want).abs().max().item() < 1e-4
        m.MOE_GATHER_MAX_PAIRS = 0
        assert (m.forward_logits(ids) - want).abs().max().item() < 1e-4
