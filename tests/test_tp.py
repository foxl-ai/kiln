"""Tensor parallelism on CPU over gloo: tp=2 must generate what tp=1 generates."""

import os

import pytest
import torch

MODEL = os.environ.get("KILN_TEST_MODEL")



@pytest.mark.skipif(not MODEL, reason="set KILN_TEST_MODEL to run")
def test_tp2_matches_tp1():
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    kw = dict(model_path=MODEL, device="cpu", dtype=torch.float32, page_size=8, num_pages=256,
              max_num_seqs=4, max_model_len=256, max_prefill_tokens=32)
    one = LLMEngine(EngineConfig(**kw))
    prompts = [one.tokenizer(p)["input_ids"] for p in
               ["The capital of France is", "def fibonacci(n):\n", "1, 2, 3, 5, 8,", "Once upon a time"]]
    sp = SamplingParams(max_new_tokens=20, ignore_eos=True)
    ref = [r.output_ids for r in one.generate(prompts, sp)]
    two = LLMEngine(EngineConfig(tp=2, **kw))
    try:
        got = [r.output_ids for r in two.generate(prompts, sp)]
        logits_ref = one.model.forward_logits(torch.tensor(prompts[0]))
        # rank 0's own forward needs rank 1 in the collectives, so compare tokens only
    finally:
        two.close()
    assert got == ref
    assert logits_ref.shape[0] == len(prompts[0])


def test_tp2_moe_matches_tp1(tmp_path):
    """Expert intermediate dims sharded across ranks, no download (tiny HF-built MoE)."""
    from tests.test_architectures import build

    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    build("qwen3_moe", str(tmp_path))
    kw = dict(model_path=str(tmp_path), device="cpu", dtype=torch.float32, page_size=4, num_pages=128,
              max_num_seqs=3, max_model_len=128, max_prefill_tokens=8)
    prompts = [[5, 9, 11, 200, 3], list(range(40, 70)), [7] * 9]
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True)
    ref = [r.output_ids for r in LLMEngine(EngineConfig(**kw)).generate(prompts, sp)]
    two = LLMEngine(EngineConfig(tp=2, **kw))
    try:
        got = [r.output_ids for r in two.generate(prompts, sp)]
    finally:
        two.close()
    assert got == ref


def test_tp4_replicates_kv_heads(tmp_path):
    """tp=4 over a model with 2 KV heads: each head is held by two ranks."""
    from tests.test_architectures import build

    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    build("qwen3", str(tmp_path))  # 4 query heads, 2 KV heads
    kw = dict(model_path=str(tmp_path), device="cpu", dtype=torch.float32, page_size=4, num_pages=128,
              max_num_seqs=2, max_model_len=128, max_prefill_tokens=8)
    prompts = [[5, 9, 11, 200, 3], list(range(40, 60))]
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True)
    ref = [r.output_ids for r in LLMEngine(EngineConfig(**kw)).generate(prompts, sp)]
    four = LLMEngine(EngineConfig(tp=4, **kw))
    try:
        got = [r.output_ids for r in four.generate(prompts, sp)]
    finally:
        four.close()
    assert got == ref


@pytest.mark.parametrize("tie", [False, True])
def test_vocab_parallel_with_a_padded_last_shard(tmp_path, tie):
    """vocab_parallel shards embedding and lm_head rows (vLLM VocabParallelEmbedding /
    ParallelLMHead); 383 rows over 4 ranks leaves the last shard one row of padding, which
    must never be sampled. Greedy tokens and logprobs equal the replicated tp=1 run."""
    from tests.test_architectures import build

    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    build("qwen3", str(tmp_path), vocab_size=383, tie_word_embeddings=tie)
    kw = dict(model_path=str(tmp_path), device="cpu", dtype=torch.float32, page_size=4, num_pages=128,
              max_num_seqs=2, max_model_len=128, max_prefill_tokens=8)
    prompts = [[5, 9, 11, 382, 3], list(range(340, 360))]
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True, logprobs=3)
    ref = LLMEngine(EngineConfig(**kw)).generate(prompts, sp)
    four = LLMEngine(EngineConfig(tp=4, **kw))
    try:
        assert four.model.vocab_parallel and four.model.embed.shape[0] == 96
        got = four.generate(prompts, sp)
    finally:
        four.close()
    for a, b in zip(ref, got):
        assert a.output_ids == b.output_ids
        assert all(abs(x[0] - y[0]) < 1e-4 and x[1] == y[1] for x, y in zip(a.logprobs, b.logprobs))
