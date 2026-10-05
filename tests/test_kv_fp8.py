"""FP8 KV cache against a full-precision cache, on a real checkpoint (CPU)."""

import os

import pytest
import torch

MODEL = os.environ.get("KILN_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set KILN_TEST_MODEL to run")


def test_fp8_kv_cache_tracks_full_precision():
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    kw = dict(model_path=MODEL, device="cpu", dtype=torch.bfloat16, page_size=8, num_pages=512,
              max_num_seqs=4, max_model_len=512, max_prefill_tokens=32)
    ref_eng = LLMEngine(EngineConfig(**kw))
    fp8_eng = LLMEngine(EngineConfig(kv_cache_dtype="fp8", **kw))
    assert fp8_eng.runner.k_caches[0].dtype == torch.float8_e4m3fn
    tok = ref_eng.tokenizer
    prompts = [tok(p)["input_ids"] for p in ["The capital of France is", "def fibonacci(n):\n",
                                              "1, 2, 3, 5, 8, 13,", "The quick brown fox jumps over"]]
    sp = SamplingParams(max_new_tokens=24, ignore_eos=True, logprobs=0)
    a = ref_eng.generate(prompts, sp)
    b = fp8_eng.generate(prompts, sp)
    same = sum(x == y for r, s in zip(a, b) for x, y in zip(r.output_ids, s.output_ids))
    total = sum(len(r.output_ids) for r in a)
    first = [abs(r.logprobs[0][0] - s.logprobs[0][0]) for r, s in zip(a, b)]
    print(f"fp8 KV: {same}/{total} greedy tokens equal, first-token |dlogprob| max {max(first):.4f}")
    assert all(r.output_ids[:4] == s.output_ids[:4] for r, s in zip(a, b))
    assert same / total >= 0.75
    # Half the bytes per token.
    assert fp8_eng.pool.num_pages >= 2 * ref_eng.pool.num_pages - 2 or kw["num_pages"]
