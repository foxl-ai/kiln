"""FP8 / MXFP4 checkpoints in MiMo-V2.6's exact on-disk format, loaded two ways.

A tiny MiMo-V2 is quantized the way the real checkpoint is (quantization_config fp8 e4m3,
square weight blocks with weight_scale_inv; experts stored MXFP4 with E8M0 weight_scale;
o_proj left bf16 via ignored_layers). The Hugging Face reference then runs on the
dequantized values, and Kiln must match it both keeping FP8 on the device and dequantizing
at load.
"""

import json
import os

import torch
from safetensors.torch import load_file, save_file

from kiln.models.quant import E2M1, mxfp4_unpack

BLOCK = 16


def fp8_blocks(w):
    """e4m3fn with one scale per BLOCK x BLOCK (checkpoint style, max 448)."""
    N, K = w.shape
    wb = w.float().view(N // BLOCK, BLOCK, K // BLOCK, BLOCK)
    s = wb.abs().amax(dim=(1, 3)).clamp(min=1e-12) / 448.0
    q = (wb / s[:, None, :, None]).view(N, K).to(torch.float8_e4m3fn)
    return q, s


def mxfp4(w):
    """Nearest E2M1 code per element, one power-of-two E8M0 per 32, packed low nibble first."""
    N, K = w.shape
    wb = w.float().view(N, K // 32, 32)
    e = torch.ceil(torch.log2(wb.abs().amax(-1).clamp(min=1e-12) / 6.0))
    scaled = wb / torch.exp2(e).unsqueeze(-1)
    lut = torch.tensor(E2M1[:8])
    codes = (scaled.abs().unsqueeze(-1) - lut).abs().argmin(-1) + 8 * (scaled < 0)
    codes = codes.view(N, K).to(torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, (e + 127).to(torch.uint8)


def quantized_mimo(tmp_path, **shape):
    """A tiny MiMo-V2 saved in the real checkpoint's quantized format; returns (dir, the HF
    reference loaded with the dequantized values, ModelConfig)."""
    from tests.test_mimo_v2 import build_reference

    from kiln.config import ModelConfig

    src = tmp_path / "bf16"
    hf = build_reference(str(src), layout="fused_qkv", **shape)
    tensors = load_file(str(src / "model.safetensors"))
    out, ref_sd = {}, {}
    for name, t in tensors.items():
        base = name[: -len(".weight")] if name.endswith(".weight") else None
        if base and base.endswith("qkv_proj") or (base and base.endswith(("gate_proj", "up_proj", "down_proj"))
                                                  and ".experts." not in base):
            q, s = fp8_blocks(t)
            out[name], out[base + ".weight_scale_inv"] = q, s
            ref_sd[name] = (q.float().view(t.shape[0] // BLOCK, BLOCK, -1, BLOCK) * s[:, None, :, None]).view(t.shape)
            if base.endswith("qkv_proj"):  # stored grouped by KV head; the reference splits contiguously
                from tests.test_mimo_v2 import from_grouped, qkv_dims

                ref_sd[name] = from_grouped(ref_sd[name], *qkv_dims(hf.config, int(name.split(".")[2])))
        elif base and ".experts." in base:
            packed, exps = mxfp4(t)
            out[name], out[base + ".weight_scale"] = packed, exps
            qv, sc = mxfp4_unpack(packed, exps)
            ref_sd[name] = qv.float() * sc.repeat_interleave(32, dim=-1)
        else:
            out[name] = t
            ref_sd[name] = t.float()
    dst = tmp_path / "quant"
    os.makedirs(dst)
    save_file(out, str(dst / "model.safetensors"))
    cfg = json.load(open(src / "config.json"))
    cfg["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
                                  "weight_block_size": [BLOCK, BLOCK], "store_dtype": "mxfp4",
                                  "mxfp4_block_size": 32,
                                  "ignored_layers": [f"model.layers.{i}.self_attn.o_proj" for i in range(4)]}
    json.dump(cfg, open(dst / "config.json", "w"))
    hf.load_state_dict({k: v.to(torch.float32) for k, v in ref_sd.items()}, strict=False)
    return dst, hf, ModelConfig.from_pretrained(str(dst))


def test_quantized_mimo_checkpoint(tmp_path):
    from kiln.models.loader import load_model

    dst, hf, mc = quantized_mimo(tmp_path)
    assert mc.quant_block == BLOCK and mc.quant_expert_block == 32 and not mc.is_quantized(
        "model.layers.0.self_attn.o_proj")
    ids = torch.randint(0, 384, (37,))
    with torch.no_grad():
        want = hf(ids.unsqueeze(0)).logits[0]
        for keep, packed in ((True, False), (True, True), (False, False)):
            m = load_model(str(dst), mc, torch.float32, torch.device("cpu"), 512, keep_fp8=keep,
                           packed_mxfp4=packed)
            l1 = m.layers[1]
            if keep and packed:  # MXFP4 experts stay 4-bit, half a byte per weight
                assert l1.w_gu.dtype == torch.uint8 and l1.w_gu.shape[-1] * 2 == mc.hidden_size
            elif keep:  # MXFP4 experts converted to FP8 with bf16 power-of-two scales, down stored [E, Im, H]
                assert l1.w_gu.dtype == torch.float8_e4m3fn and l1.w_gu_scale.dtype == torch.bfloat16
                assert l1.w_down.shape[1:] == (mc.moe_intermediate_size, mc.hidden_size) and l1.down_t
            if keep:
                assert l1.qkv.dtype == torch.float8_e4m3fn
                assert l1.o.dtype == torch.float32  # ignored layer stays unquantized
            got = m.forward_logits(ids)
            assert (got - want).abs().max().item() < 1e-4, (keep, packed)
            m.MOE_GATHER_MAX_PAIRS = 0  # the dense path (prefill-sized batches) too
            assert (m.forward_logits(ids) - want).abs().max().item() < 1e-4, (keep, packed)


def test_block_scales_restarting_at_every_checkpoint_shard(tmp_path):
    """Scales saved per tensor-parallel shard (each shard's rows start a new block), as MiMo's
    full-attention qkv_proj is: 2 shards of 40 rows in 32-row blocks need 4 scale rows, not 3."""
    from safetensors.torch import save_file

    from kiln.models.loader import _Checkpoint

    g = torch.Generator().manual_seed(0)
    w = torch.randn(80, 64, generator=g)
    q, scales = [], []
    for t in range(2):
        part = w[t * 40 : (t + 1) * 40]
        pad = torch.nn.functional.pad(part, (0, 0, 0, 24))  # 40 -> 64 rows: 2 blocks of 32
        blk = pad.view(2, 32, 2, 32).abs().amax(dim=(1, 3)) / 200.0
        qq = (pad / blk.repeat_interleave(32, 0).repeat_interleave(32, 1)).to(torch.float8_e4m3fn)[:40]
        q.append(qq)
        scales.append(blk)
    save_file({"x.weight": torch.cat(q), "x.weight_scale_inv": torch.cat(scales)}, str(tmp_path / "m.safetensors"))
    got_w, got_s = _Checkpoint(str(tmp_path)).linear("x", block=32)
    deq = got_w.float() * got_s.repeat_interleave(32, dim=1)
    assert got_s.shape == (80, 2) and (deq - w).abs().max() < 0.05 * w.abs().max()
    wrong = got_w.float() * torch.cat(scales).repeat_interleave(32, 0)[:80].repeat_interleave(32, 1)
    assert (wrong - w).abs().max() > 0.1 * w.abs().max()  # what global blocks would have given


def test_fp8_column_shards_narrower_than_a_block(tmp_path):
    """Tensor-parallel column shards of a 128x128 block-scaled FP8 weight that do not start on
    a block edge: inside one block (64 of 128, GLM-5.3-Flash experts at tp=32) the shard keeps
    that block's scale and dequantizes exactly; straddling a block edge (40 columns from 120,
    Qwen3.8-Flash-Next experts at tp=16) it comes back dequantized for _assign to quantize."""
    from safetensors.torch import save_file

    from kiln.models.loader import _Checkpoint, _assign
    from kiln.models.quant import FP8, dequant

    g = torch.Generator().manual_seed(0)
    w = torch.randn(256, 640, generator=g)
    blk = w.view(2, 128, 5, 128).abs().amax(dim=(1, 3)) / 448.0  # checkpoint-style 128 x 128 blocks
    q = (w / blk.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(torch.float8_e4m3fn)
    save_file({"x.weight": q, "x.weight_scale_inv": blk}, str(tmp_path / "m.safetensors"))
    ck = _Checkpoint(str(tmp_path))
    full = dequant(*ck.linear("x"), torch.float32)
    for cols in (slice(64, 128), slice(0, 64), slice(512, 576), slice(40, 80)):
        got_w, got_s = ck.linear("x", cols=cols)
        assert got_s is not None and got_s.shape == (256, 1)
        assert torch.equal(dequant(got_w, got_s, torch.float32), full[:, cols])
    for cols in (slice(120, 160), slice(600, 640)):
        got_w, got_s = ck.linear("x", cols=cols)
        if cols.start // 128 == (cols.stop - 1) // 128:
            assert torch.equal(dequant(got_w, got_s, torch.float32), full[:, cols])
            continue
        assert got_s is None and torch.equal(got_w, full[:, cols])
        param = torch.nn.Parameter(torch.empty(256, 40, dtype=FP8), requires_grad=False)
        scale = torch.nn.Parameter(torch.empty(256, 1), requires_grad=False)
        _assign(param, scale, got_w, got_s, torch.bfloat16, 128, 240.0)
        again = dequant(param.data, scale.data, torch.float32)
        assert (again - full[:, cols]).abs().max() <= 0.07 * full[:, cols].abs().max()  # one more e4m3 rounding


def test_vision_language_alias_serves_whole_tensors(tmp_path):
    """Checkpoints that keep the language model under model.language_model. (GLM-5.3-Flash,
    Qwen3.5) are read through _Alias; whole-tensor reads (0-dim scales go through get_tensor)
    must work on an alias too: the merge of feat/gqa-moe broke every GLM-5.3-Flash load with
    "'_Alias' object has no attribute 'get_tensor'" (2026-10-03)."""
    from safetensors.torch import save_file

    from kiln.models.loader import _Checkpoint

    save_file({"model.language_model.x.weight": torch.arange(6.0).view(2, 3),
               "model.language_model.s": torch.tensor(0.5)}, str(tmp_path / "m.safetensors"))
    ck = _Checkpoint(str(tmp_path))
    assert torch.equal(ck.get("model.x.weight"), torch.arange(6.0).view(2, 3))
    assert float(ck.get("model.s")) == 0.5


def test_fit_e4m3_max_row_groups():
    """fit_e4m3_max per (row, block) and per (row group, block): both give back the checkpoint's
    values except codes below 2^-5 in the rows they halve (halving makes them subnormal, and an odd
    last bit rounds to even: an error of 2^-10 times the doubled scale); the grouped fit halves whole
    groups, so its scales are constant over each group, and it rounds at least as many codes."""
    from kiln.models.quant import FP8, dequant, fit_e4m3_max

    g = torch.Generator().manual_seed(0)
    N, K, bk = 128, 512, 128
    cf = (torch.randn(N, K, generator=g) * 60).clamp(-448, 448)
    big = torch.rand(N, K // bk, generator=g) < 0.4  # some (row, block) pairs reach above 240
    cf = torch.where(big.repeat_interleave(bk, 1) & (torch.arange(K) % bk == 0), torch.full_like(cf, 400.0), cf)
    codes = cf.to(torch.float8_e4m3fn)
    scale = (torch.rand(2, K // bk, generator=g) * 0.01 + 1e-3).repeat_interleave(N // 2, 0)  # two 64-row groups
    want = codes.float() * scale.repeat_interleave(bk, 1)
    sub = codes.float().abs() < 2.0 ** -5  # codes that are subnormal once halved
    changed = {}
    for grp in (None, 64):
        w, s = fit_e4m3_max(codes.to(FP8), scale, 240.0, grp)
        assert w.float().abs().max() <= 240
        got = dequant(w, s, torch.float32)
        err = (got - want).abs() / s.repeat_interleave(bk, 1)
        assert err.max() <= 2.0 ** -10 * 1.001 and not bool((err > 1e-6)[~sub].any())  # fp32 rounding of err itself
        changed[grp] = int((err > 1e-6).sum())
        if grp is not None:
            sg = s.view(N // grp, grp, -1)
            assert torch.equal(sg, sg[:, :1].expand_as(sg))
    assert changed[64] >= changed[None]
