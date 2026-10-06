"""kiln/profiling.py (host timeline, NEFF input capture) and tools/util_report.py's model arithmetic, on CPU."""

import json
import os
import sys

import numpy as np
import torch

from kiln import profiling

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import util_report  # noqa: E402


def test_parse_at():
    assert profiling.parse_at("prefill:40, decode:300,decode:301") == {"prefill": {40}, "decode": {300, 301}}
    assert profiling.parse_at("") == {}


def test_raw_keeps_bytes():
    x = torch.randn(3, 5).to(torch.bfloat16)
    r = profiling.raw(x)
    assert r.dtype == np.int16 and r.shape == (3, 5)
    assert torch.equal(torch.from_numpy(r).view(torch.bfloat16), x)
    f8 = torch.randn(4).to(torch.float8_e4m3fn)
    assert torch.equal(torch.from_numpy(profiling.raw(f8)).view(torch.float8_e4m3fn).float(), f8.float())
    assert profiling.raw(torch.tensor([True, False])).tolist() == [1, 0]


class _Meta:
    neff_id, artifact_dir, device_id = "abc", "/x", 3


def test_capture_writes_armed_calls_once(tmp_path):
    cap = profiling.InputCapture(str(tmp_path), 3, profiling.parse_at("prefill:2"))
    w = torch.arange(12, dtype=torch.float32).reshape(3, 4)  # a "weight": the same object in every call
    for n in range(1, 4):
        cap.begin("prefill")
        cap.hook((w, torch.full((2,), n, dtype=torch.int32), w[1:]), _Meta())  # w[1:]: a new object
        cap.end()
        cap.begin("decode")
        cap.hook((w,), _Meta())
        cap.end()
    lines = [json.loads(x) for x in open(tmp_path / "r3" / "manifest.jsonl")]
    assert [x["call"] for x in lines] == ["prefill:2"]  # only the second prefill call, no decode
    e = lines[0]
    assert e["neff_id"] == "abc" and e["device_id"] == 3 and len(e["inputs"]) == 3
    assert len(set(e["inputs"])) == 3  # the view w[1:] is its own file
    assert np.array_equal(np.load(e["inputs"][0]).view(np.float32), w.numpy())  # raw bytes, as raw() writes
    assert np.load(e["inputs"][1]).tolist() == [2, 2]
    assert np.array_equal(np.load(e["inputs"][2]).view(np.float32), w[1:].numpy())
    cap.at = {"prefill": {4}}
    cap.begin("prefill")
    cap.hook((w,), _Meta())
    cap.end()
    lines = [json.loads(x) for x in open(tmp_path / "r3" / "manifest.jsonl")]
    assert lines[1]["inputs"][0] == e["inputs"][0]  # a tensor seen before is referenced, not rewritten


def test_timeline_records_and_flushes(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "TIMELINE", [])
    profiling.record("exec", 1.0, 2.0, "decode", "k", 1.1, 1.2)
    p = tmp_path / "tl.jsonl"
    profiling.flush(str(p))
    assert [json.loads(x) for x in open(p)] == [["exec", 1.0, 2.0, "decode", "k", 1.1, 1.2]]
    assert profiling.TIMELINE == []


def _toy():
    tc = {"num_hidden_layers": 2, "n_routed_experts": 4, "num_experts_per_tok": 2, "num_attention_heads": 2,
          "qk_head_dim": 8, "v_head_dim": 8, "index_topk": 4, "index_kpool": 2, "index_n_heads": 1,
          "index_head_dim": 4, "kv_lora_rank": 16, "layer_types": ["linear_attention", "deepseek_sparse_attention"],
          "linear_attn_config": {"num_heads": 2, "head_dim": 4}}
    h = {"model.layers.0.self_attn.q_proj.weight": ["BF16", [8, 16]],
         "model.layers.1.self_attn.q_b_proj.weight": ["F8_E4M3", [16, 4]],
         "model.layers.1.self_attn.indexer.wk.weight": ["BF16", [4, 16]],
         "model.layers.1.mlp.experts.0.up_proj.weight": ["F8_E4M3", [32, 16]],
         "model.layers.1.mlp.experts.1.up_proj.weight": ["F8_E4M3", [32, 16]],
         "model.layers.1.mlp.experts.2.up_proj.weight": ["F8_E4M3", [32, 16]],
         "model.layers.1.mlp.experts.3.up_proj.weight": ["F8_E4M3", [32, 16]],
         "model.layers.1.mlp.gate.weight": ["BF16", [4, 16]],
         "model.layers.2.mlp.experts.0.up_proj.weight": ["F8_E4M3", [32, 16]],  # the MTP layer: left out
         "lm_head.weight": ["BF16", [10, 16]]}
    return tc, {k: tuple(v) for k, v in h.items()}


def test_model_counts_and_flops():
    tc, h = _toy()
    c = util_report.model_counts(tc, h)
    assert c["routed experts"] == (4 * 512 * 2 / 4, 4 * 512)  # top-2 of 4 experts touched, 1 byte each
    assert c["KDA projections"][0] == 128 and c["DSA projections"] == (64, 64)
    assert c["router"] == (64, 128) and c["DSA indexer"] == (64, 128) and c["lm_head"] == (0, 320)
    tot, parts = util_report.prefill_flops_per_token(tc, h, ctx=6)
    a = util_report.attention_flops(tc, 6)
    assert a["mean dsa keys"] == (1 + 2 + 3 + 4 + 4 + 4) / 6
    assert a["dsa core"] == 2 * 2 * a["mean dsa keys"] * 16
    assert tot == sum(parts.values()) == 2 * (1024 + 128 + 64 + 64 + 64) + a["dsa core"] + a["dsa indexer scores"] \
        + a["kda recurrence"]


def test_decode_bytes_per_rank():
    tc, h = _toy()
    d = util_report.decode_bytes_per_rank(tc, h, tp=2, attn_tp=2, rows_step=2, rows_group=1, ctx=8, kv="fp8")
    assert d["expert coverage"] == 1 - (1 - 2 / 4) ** 2
    assert d["routed experts touched"] == 4 * 512 / 2 * d["expert coverage"]
    assert d["kv rows"] == 1 * 8 * (16 + 2 * 4 + 2 * 4 / 2)  # one DSA layer, 8 keys
    assert d["kda state r+w"] == 2 * 1 * 1 * 4 * 4 * 4  # one KDA layer, 2 heads / attn_tp 2, read + write fp32
    assert d["dense weights"] == 256 / 2 + 64 / 2 + 128 + 128 + 320 / 2
