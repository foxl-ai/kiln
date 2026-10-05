"""vLLM thinking_token_budget: reasoning is cut at the budget by forcing reasoning_end_str."""

import os

import pytest
import torch

MODEL = os.environ.get("KILN_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set KILN_TEST_MODEL to run (a <think> model, e.g. Qwen3)")


@pytest.mark.parametrize("end", ["</think>", "I have to give the solution based on the reasoning directly now.</think>"])
def test_reasoning_stops_at_the_budget(end):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    eng = LLMEngine(EngineConfig(model_path=MODEL, device="cpu", dtype=torch.float32, page_size=8, num_pages=256,
                                 max_num_seqs=2, max_model_len=512, max_prefill_tokens=64,
                                 reasoning_start_str="<think>", reasoning_end_str=end, overlap=True))
    tok = eng.tokenizer
    ids = tok.apply_chat_template([{"role": "user", "content": "9.11 and 9.8, which is greater?"}],
                                  add_generation_prompt=True, tokenize=True)
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    start, end_ids = eng.think_start_ids, eng.think_end_ids
    budget = 10
    free, capped = eng.generate([ids, ids], [SamplingParams(max_new_tokens=60, ignore_eos=True),
                                             SamplingParams(max_new_tokens=60, ignore_eos=True,
                                                            thinking_token_budget=budget)])
    out = capped.output_ids
    s = out.index(start[-1]) + 1
    assert out[s + budget : s + budget + len(end_ids)] == end_ids  # exactly `budget` reasoning tokens
    assert free.output_ids[: s + budget] == out[: s + budget]  # same text up to the cut
    assert end_ids[-1] not in free.output_ids[: s + budget + len(end_ids)]  # the free run was still thinking
    assert capped.think_done and not free.think_done
