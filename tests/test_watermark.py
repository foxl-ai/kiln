"""Gumbel-max watermarking (vLLM --watermark-config; engine/watermark.py)."""

import math
import random

import torch

from kiln.engine.watermark import Watermark, _gamma_sf


def test_choice_is_an_exact_sample_of_the_distribution():
    wm = Watermark(key=7, context_width=2)
    probs = [0.5, 0.25, 0.15, 0.1]
    lps = [math.log(p) for p in probs]
    rng = random.Random(0)
    n = 20000
    counts = [0] * 4
    for _ in range(n):
        ctx = (rng.randrange(1 << 30), rng.randrange(1 << 30))
        counts[wm.choose(ctx, [0, 1, 2, 3], lps, temperature=1.0)] += 1
    for c, p in zip(counts, probs):
        assert abs(c / n - p) < 4 * math.sqrt(p * (1 - p) / n)


def test_top_p_and_temperature_shape_the_candidates():
    wm = Watermark(key=1, context_width=1)
    lps = [math.log(p) for p in (0.6, 0.3, 0.1)]
    picks = {wm.choose((c,), [10, 11, 12], lps, temperature=1.0, top_p=0.5) for c in range(200)}
    assert picks == {10}  # 0.6 alone already covers top_p
    picks = {wm.choose((c,), [10, 11, 12], lps, temperature=1.0, top_k=2) for c in range(400)}
    assert picks == {10, 11}


def test_gamma_tail_matches_its_definition():
    assert abs(_gamma_sf(1.0, 1) - math.exp(-1.0)) < 1e-12
    assert abs(_gamma_sf(2.0, 2) - 3 * math.exp(-2.0)) < 1e-12


def test_engine_output_is_detectable_and_opt_out_is_not(tmp_path):
    """A random tiny model at temperature 1 (high entropy, the easy case for a watermark)."""
    from tests.test_architectures import build

    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    build("qwen3", str(tmp_path))
    eng = LLMEngine(EngineConfig(model_path=str(tmp_path), device="cpu", dtype=torch.float32, page_size=4,
                                 num_pages=256, max_num_seqs=4, max_model_len=256, max_prefill_tokens=16,
                                 overlap=True, watermark={"algorithm": "gumbel", "key": 1234}))
    prompts = [[5, 9, 11, 200, 3], list(range(40, 60))]
    on = eng.generate(prompts, SamplingParams(max_new_tokens=96, ignore_eos=True, temperature=1.0, seed=1))
    off = eng.generate(prompts, SamplingParams(max_new_tokens=96, ignore_eos=True, temperature=1.0, seed=1,
                                               watermarking=False))
    for r in on:
        d = eng.watermark.detect(r.output_ids)
        assert d["num_scored"] > 60 and d["p_value"] < 1e-6, d
    for r in off:
        assert eng.watermark.detect(r.output_ids)["p_value"] > 1e-3
    other = Watermark(key=999)
    assert all(other.detect(r.output_ids)["p_value"] > 1e-3 for r in on)  # the key matters
