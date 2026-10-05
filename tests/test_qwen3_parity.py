"""Kiln's Qwen3 against Hugging Face transformers, on CPU, with a real checkpoint.

Gated on KILN_TEST_MODEL (a local path or a repo id such as Qwen/Qwen3-0.6B) because it
downloads weights. Runs in fp32 so that differences are bugs, not rounding.
"""

import os

import pytest
import torch

MODEL = os.environ.get("KILN_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set KILN_TEST_MODEL to run")

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Trainium is a machine learning accelerator built by",
    "1, 2, 3, 5, 8, 13,",
]


@pytest.fixture(scope="module")
def hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kiln.models.loader import resolve_model_path

    path = resolve_model_path(MODEL)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    return path, tok, model


def hf_greedy(model, ids, n):
    with torch.no_grad():
        out = model.generate(torch.tensor([ids]), max_new_tokens=n, do_sample=False,
                             eos_token_id=None, pad_token_id=0)
    return out[0, len(ids):].tolist()


def make_engine(path, **kw):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    cfg = EngineConfig(model_path=path, device="cpu", dtype=torch.float32, page_size=4,
                       num_pages=512, max_num_seqs=4, max_model_len=256, max_prefill_tokens=16,
                       **kw)
    return LLMEngine(cfg)


def test_logits_match_transformers(hf):
    path, tok, model = hf
    eng = make_engine(path)
    ids = tok(PROMPTS[0])["input_ids"]
    with torch.no_grad():
        ref = model(torch.tensor([ids])).logits[0]
        got = eng.model.forward_logits(torch.tensor(ids))
    assert (got - ref).abs().max().item() < 2e-3


def test_batched_chunked_paged_greedy_matches_transformers(hf):
    from kiln.engine.request import SamplingParams

    path, tok, model = hf
    eng = make_engine(path)
    prompts = [tok(p)["input_ids"] for p in PROMPTS]
    n = 24
    reqs = eng.generate(prompts, SamplingParams(max_new_tokens=n, ignore_eos=True))
    for ids, r in zip(prompts, reqs):
        assert r.output_ids == hf_greedy(model, ids, n), tok.decode(ids)


def test_prefix_cache_hit_gives_the_same_tokens(hf):
    from kiln.engine.request import SamplingParams

    path, tok, model = hf
    eng = make_engine(path)
    system = tok("You are a careful assistant. Answer in one short sentence. " * 3)["input_ids"]
    q1 = system + tok("What is two plus two?")["input_ids"]
    q2 = system + tok("Name a color.")["input_ids"]
    eng.generate([q1], SamplingParams(max_new_tokens=8, ignore_eos=True))
    (r2,) = eng.generate([q2], SamplingParams(max_new_tokens=8, ignore_eos=True))
    assert r2.num_cached_tokens >= (len(system) // 4) * 4 - 4
    assert r2.output_ids == hf_greedy(model, q2, 8)
