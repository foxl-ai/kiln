"""Overlap scheduling against synchronous scheduling with a real model (CPU)."""

import os

import pytest
import torch

MODEL = os.environ.get("KILN_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set KILN_TEST_MODEL to run")


def make(overlap, num_pages=512, **kw):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    return LLMEngine(EngineConfig(model_path=MODEL, device="cpu", dtype=torch.float32, page_size=4,
                                  num_pages=num_pages, max_num_seqs=4, max_model_len=256,
                                  max_prefill_tokens=16, overlap=overlap, **kw))


def test_overlap_outputs_equal_sync_outputs():
    from kiln.engine.request import SamplingParams

    sync, over = make(False), make(True)
    tok = sync.tokenizer
    prompts = [tok(p)["input_ids"] for p in [
        "The capital of France is", "def fibonacci(n):\n", "Once upon a time", "1, 2, 3, 5, 8,",
        "The quick brown fox", "SELECT * FROM", "In 1969,", "Water boils at"]]
    params = [SamplingParams(max_new_tokens=n, ignore_eos=True, logprobs=1) for n in (1, 7, 20, 13, 30, 2, 9, 16)]
    params[3] = SamplingParams(max_new_tokens=40, stop=(" 21",))
    a = sync.generate(prompts, params)
    b = over.generate(prompts, params)
    for x, y in zip(a, b):
        assert x.output_ids == y.output_ids, (tok.decode(x.output_ids), tok.decode(y.output_ids))
        assert x.finish_reason == y.finish_reason
        if y.params.logprobs is not None:
            assert len(y.logprobs) == len(y.output_ids)
    assert over.pool.num_free + over.radix.total_pages() == over.pool.num_usable


def test_overlap_under_kv_pressure():
    from kiln.engine.request import SamplingParams

    sync, over = make(False, num_pages=40, admission="eager"), make(True, num_pages=40, admission="eager")
    tok = sync.tokenizer
    prompts = [tok(f"Count from {i}:")["input_ids"] for i in range(6)]
    sp = SamplingParams(max_new_tokens=40, ignore_eos=True)
    a = sync.generate(prompts, sp)
    b = over.generate(prompts, sp)
    assert over.scheduler.num_preemptions > 0
    for x, y in zip(a, b):
        assert x.output_ids == y.output_ids
