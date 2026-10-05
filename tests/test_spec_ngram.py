import os

import pytest
import torch

from kiln.engine.spec_ngram import NgramProposer

MODEL = os.environ.get("KILN_TEST_MODEL")


def test_proposes_the_continuation_of_the_latest_match():
    p = NgramProposer(k=3, min_n=2, max_n=3)
    assert p.propose([1, 2, 3, 4, 5, 9, 9, 1, 2, 3]) == [4, 5, 9]
    assert p.propose([7, 8, 1, 2, 6, 6, 1, 2]) == [6, 6, 1]  # latest occurrence of [1, 2]
    assert p.propose([1, 2, 3, 4]) == []
    assert p.propose([5, 6, 5, 6], k=1) == [5]


@pytest.fixture(scope="module")
def engines():
    if not MODEL:
        pytest.skip("set KILN_TEST_MODEL to run")
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    def make(**kw):
        return LLMEngine(EngineConfig(model_path=MODEL, device="cpu", dtype=torch.float32, page_size=8,
                                      num_pages=512, max_num_seqs=4, max_model_len=512,
                                      max_prefill_tokens=64, **kw))

    return make(), make(spec_method="ngram", spec_k=4), make(spec_method="suffix", spec_k=6)


PROMPTS = [
    "Repeat after me exactly: the cat sat on the mat. the cat sat on the mat. the cat sat on the mat.",
    "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n\ndef mul(a, b):\n",
    "List: apple, banana, cherry, apple, banana, cherry, apple, banana,",
]


@pytest.mark.parametrize("method", ["ngram", "suffix"])
def test_greedy_speculative_output_equals_plain_decoding(engines, method):
    from kiln.engine.request import SamplingParams

    plain, spec = engines[0], engines[1 if method == "ngram" else 2]
    ids = [plain.tokenizer(p)["input_ids"] for p in PROMPTS]
    sp = SamplingParams(max_new_tokens=40, ignore_eos=True, logprobs=2)
    a = plain.generate(ids, sp)
    b = spec.generate(ids, sp)
    for x, y in zip(a, b):
        assert x.output_ids == y.output_ids
        assert len(y.logprobs) == len(y.output_ids)
        assert all(abs(p[0] - q[0]) < 1e-3 for p, q in zip(x.logprobs, y.logprobs))
    assert spec.spec_accepted > 0, "repetitive prompts should accept some drafts"
    print(f"{method}: accepted {spec.spec_accepted}/{spec.spec_proposed} drafted tokens")
