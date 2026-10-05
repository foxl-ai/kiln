"""SGLang /update_weights_from_disk: swap in another checkpoint without rebuilding the engine."""

import os

import pytest
import torch
from safetensors.torch import load_file, save_file


@pytest.mark.parametrize("tp", [1, 2])
def test_update_weights_matches_a_fresh_engine_and_flushes_the_cache(tmp_path, tp):
    from tests.test_architectures import build

    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    a, b = tmp_path / "a", tmp_path / "b"
    build("qwen3", str(a))
    os.makedirs(b)
    for f in os.listdir(a):
        if f.endswith(".safetensors"):
            t = load_file(str(a / f))
            g = torch.Generator().manual_seed(7)
            save_file({k: v + 0.05 * torch.randn(v.shape, generator=g).to(v.dtype) for k, v in t.items()},
                      str(b / f), metadata={"format": "pt"})
        else:
            (b / f).write_bytes((a / f).read_bytes())
    kw = dict(device="cpu", dtype=torch.float32, page_size=4, num_pages=64, max_num_seqs=2, max_model_len=64,
              max_prefill_tokens=16)
    prompt = list(range(30, 50))
    sp = SamplingParams(max_new_tokens=8, ignore_eos=True)
    want_b = LLMEngine(EngineConfig(model_path=str(b), **kw)).generate([prompt], sp)[0].output_ids
    eng = LLMEngine(EngineConfig(model_path=str(a), tp=tp, **kw))
    try:
        out_a = eng.generate([prompt], sp)[0].output_ids
        ok, msg = eng.update_weights_from_disk(str(b), weight_version="v2")
        assert ok, msg
        r = eng.generate([prompt], sp)[0]
        assert r.output_ids == want_b != out_a
        assert r.num_cached_tokens == 0 and eng.weight_version == "v2"  # the old KV was flushed
        eng.add_request(prompt, sp)
        ok, msg = eng.update_weights_from_disk(str(a))
        assert not ok and "abort_all_requests" in msg
        ok, _ = eng.update_weights_from_disk(str(a), abort_all_requests=True)
        assert ok and eng.generate([prompt], sp)[0].output_ids == out_a
    finally:
        eng.close()
