import json
import os
import re

import pytest
import torch

from kiln.engine.sampler import unpack_bitmask

MODEL = os.environ.get("KILN_TEST_MODEL")


def test_unpack_bitmask_matches_xgrammar():
    xgr = pytest.importorskip("xgrammar")
    V = 1000
    mask = torch.randint(-(2**31), 2**31 - 1, (3, (V + 31) // 32), dtype=torch.int32)
    logits = torch.randn(3, V)
    ref = logits.clone()
    xgr.apply_token_bitmask_inplace(ref, mask)
    ours = torch.where(unpack_bitmask(mask, V), logits, float("-inf"))
    assert torch.equal(torch.isinf(ours), torch.isinf(ref))
    assert torch.equal(ours[~torch.isinf(ours)], ref[~torch.isinf(ref)])


@pytest.fixture(scope="module")
def engine():
    if not MODEL:
        pytest.skip("set KILN_TEST_MODEL to run")
    pytest.importorskip("xgrammar")
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    return LLMEngine(EngineConfig(model_path=MODEL, device="cpu", dtype=torch.float32, page_size=8,
                                  num_pages=512, max_num_seqs=4, max_model_len=512, max_prefill_tokens=64))


def chat_ids(eng, text):
    ids = eng.tokenizer.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                            tokenize=True, enable_thinking=False)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def test_json_schema_output_validates(engine):
    jsonschema = pytest.importorskip("jsonschema")
    from kiln.engine.grammar import GrammarSpec
    from kiln.engine.request import SamplingParams

    schema = {"type": "object", "properties": {"city": {"type": "string"}, "population": {"type": "integer"}},
              "required": ["city", "population"], "additionalProperties": False}
    sp = SamplingParams(max_new_tokens=60, grammar=GrammarSpec("json_schema", json.dumps(schema)))
    (r,) = engine.generate([chat_ids(engine, "Give the largest city in Japan as JSON.")], sp)
    text = engine.tokenizer.decode(r.output_ids, skip_special_tokens=True)
    jsonschema.validate(json.loads(text), schema)
    assert r.finish_reason == "stop"


def test_regex_and_choice(engine):
    from kiln.engine.grammar import GrammarSpec
    from kiln.engine.request import SamplingParams

    pat = r"\d{3}-\d{4}"
    reqs = engine.generate(
        [chat_ids(engine, "Make up a phone number."), chat_ids(engine, "Is the sky blue? Answer yes or no.")],
        [SamplingParams(max_new_tokens=20, temperature=0.8, grammar=GrammarSpec("regex", pat)),
         SamplingParams(max_new_tokens=20, grammar=GrammarSpec.choice(["yes", "no"]))])
    t0 = engine.tokenizer.decode(reqs[0].output_ids, skip_special_tokens=True)
    t1 = engine.tokenizer.decode(reqs[1].output_ids, skip_special_tokens=True)
    assert re.fullmatch(pat, t0), t0
    assert t1 in ("yes", "no"), t1


# -- jump-forward decoding ------------------------------------------------------------


def run_steps(eng, prompts, params, jump: bool):
    """Generate with jump-forward on or off; returns (requests, engine steps taken)."""
    eng.cfg.jump_forward = jump
    try:
        reqs = [eng.add_request(p, sp) for p, sp in zip(prompts, params)]
        steps = 0
        while eng.has_work():
            eng.step()
            steps += 1
    finally:
        eng.cfg.jump_forward = False
    return reqs, steps


def test_jump_forward_tokens(engine):
    """GrammarBackend.jump_forward on Qwen3's tokenizer: the start boundary re-encodes the newest
    token with the jump string (SGLang), the end boundary drops tokens a longer allowed token could
    replace (llguidance docs/fast_forward.md), and the matcher is never changed."""
    from kiln.engine.grammar import GrammarBackend, GrammarSpec

    be = GrammarBackend(engine.tokenizer, engine.mcfg.vocab_size)
    enc = engine._encode
    V = be.token_bytes
    show = lambda r: None if r is None else (r[0], [V[t] for t in r[1]])  # noqa: E731

    def after(spec, text):
        m = be.matcher(spec)
        ids = enc(text)
        for t in ids:
            assert m.accept_token(t)
        return m, ids[-1]

    schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"],
              "additionalProperties": False}
    m, last = after(GrammarSpec("json_schema", json.dumps(schema)), '{"')
    assert m.find_jump_forward_string() == 'city"'
    # `"` is dropped: `":` (or `" :`) may follow, and that is the split the model was trained on.
    assert show(be.jump_forward(m, last, enc)) == (False, [b"city"])
    assert m.find_jump_forward_string() == 'city"'  # unchanged

    regex = GrammarSpec("regex", r'"name": [0-9]+, "ok": (true|false)')
    m, last = after(regex, '"')
    # The newest token `"` merges with the jump string into `"name`, which replaces it.
    assert show(be.jump_forward(m, last, enc)) == (True, [b'"name', b'":', b" "])
    assert be.jump_forward(m, last, enc, replace_ok=False) is None  # no non-canonical `"` `name`
    m, last = after(regex, '"name": 12, "ok')
    assert show(be.jump_forward(m, last, enc)) == (False, [b'":'])  # ` true` / ` false` absorb the space

    # Inside a multi-byte character xgrammar cannot return the partial jump string; after it,
    # the newest token's bytes are not text on their own, so the string is encoded alone.
    ch = "\U0002a6d6"  # three tokens in Qwen3's vocabulary
    ids = enc(ch)
    assert len(ids) == 3
    m = be.matcher(GrammarSpec("regex", ch + "ab"))
    assert m.accept_token(ids[0])
    assert be.jump_forward(m, ids[0], enc) is None
    for t in ids[1:]:
        assert m.accept_token(t)
    assert show(be.jump_forward(m, ids[-1], enc)) == (False, [b"ab"])


def schema(*keys):
    types = {"city_name": "string", "country_code": "string", "population_estimate": "integer",
             "is_capital": "boolean"}
    return {"type": "object", "properties": {k: {"type": types[k]} for k in keys}, "required": list(keys),
            "additionalProperties": False}


def test_jump_forward_json_schema(engine):
    """The same schema-constrained greedy output token for token, with real logprobs for the
    forced tokens, in fewer engine steps: each forced token is one decode step saved."""
    jsonschema = pytest.importorskip("jsonschema")
    from kiln.engine.grammar import GrammarSpec
    from kiln.engine.request import SamplingParams

    sch = schema("city_name", "country_code", "population_estimate")
    sp = SamplingParams(max_new_tokens=80, logprobs=2, grammar=GrammarSpec("json_schema", json.dumps(sch)))
    prompt = chat_ids(engine, "Describe Osaka as a JSON object.")
    (ref,), ref_steps = run_steps(engine, [prompt], [sp], jump=False)
    before = engine.jump_forward_tokens
    (jf,), jf_steps = run_steps(engine, [prompt], [sp], jump=True)
    text = engine.tokenizer.decode(jf.output_ids, skip_special_tokens=True)
    jsonschema.validate(json.loads(text), sch)
    assert jf.finish_reason == ref.finish_reason == "stop"
    assert jf.output_ids == ref.output_ids, (text, engine.tokenizer.decode(ref.output_ids))
    assert jf.num_forced > 0 and engine.jump_forward_tokens - before == jf.num_forced
    assert ref_steps == len(ref.output_ids) and jf_steps == ref_steps - jf.num_forced
    # Forced tokens are scored by the prefill rows that compute them: the logprob a decode step
    # reported for the same token, up to prefill vs decode numerics.
    assert len(jf.logprobs) == len(jf.output_ids) == len(ref.logprobs)
    for (a, ta, _), (b, tb, _) in zip(jf.logprobs, ref.logprobs):
        assert abs(a - b) < 1e-3 and ta == tb
    print(f"jump-forward: {ref_steps} -> {jf_steps} steps, {jf.num_forced}/{len(jf.output_ids)} tokens forced")


def test_jump_forward_canonical_split(engine):
    """Where the model's own split of a forced string is not the tokenizer's, jump-forward emits
    the tokenizer's: measured on Qwen3-0.6B (CPU, fp32, greedy), the masked model writes the key
    is_capital as `is` `_c` `ap` `ital` and the forced run is `is` `_cap` `ital`. The text is the
    same; the tokens, and so what later steps are conditioned on, are not (SGLang's jump-forward
    has the same property: it encodes the string with the tokenizer)."""
    jsonschema = pytest.importorskip("jsonschema")
    from kiln.engine.grammar import GrammarSpec
    from kiln.engine.request import SamplingParams

    sch = schema("city_name", "country_code", "population_estimate", "is_capital")
    sp = SamplingParams(max_new_tokens=80, grammar=GrammarSpec("json_schema", json.dumps(sch)))
    prompt = chat_ids(engine, "Describe Osaka as a JSON object.")
    (ref,), ref_steps = run_steps(engine, [prompt], [sp], jump=False)
    (jf,), jf_steps = run_steps(engine, [prompt], [sp], jump=True)
    text = engine.tokenizer.decode(jf.output_ids, skip_special_tokens=True)
    jsonschema.validate(json.loads(text), sch)
    assert text == engine.tokenizer.decode(ref.output_ids, skip_special_tokens=True)
    canon = engine._encode('is_capital"')[:-1]
    assert engine._encode(' "' + 'is_capital') == engine._encode(' "') + canon
    assert any(jf.output_ids[i : i + len(canon)] == canon for i in range(len(jf.output_ids)))
    assert jf_steps == len(jf.output_ids) - jf.num_forced < ref_steps == len(ref.output_ids)
    print(f"jump-forward: {ref_steps} -> {jf_steps} steps, {jf.num_forced}/{len(jf.output_ids)} tokens forced")


def test_jump_forward_length_and_stop(engine):
    """A forced token finishes a request exactly where a sampled one would: max_new_tokens inside
    a forced run, a stop string completed by a forced token. With logprobs, a run that would
    finish is decoded token by token instead (a finished request never runs the scoring prefill).

    The full run is checked against the grammar, not against token-by-token decoding: measured on
    Qwen3-0.6B (CPU, fp32, greedy), after "The answer is yes." the masked model, which wants to
    stop, writes ` ` `Con` `fi` `d` `ence` (` ` at logprob -10.2 beats ` Confidence` at -19.4)
    and then "85"; jump-forward forces the tokenizer's ` Confidence` and the model then writes
    "10". Same grammar, different conditioning."""
    from kiln.engine.grammar import GrammarSpec
    from kiln.engine.request import SamplingParams

    pat = r"The answer is (yes|no)\. Confidence: [0-9]{2} percent\."
    g = GrammarSpec("regex", pat)
    prompt = chat_ids(engine, "Is water wet? Answer in the given format.")
    cases = [SamplingParams(max_new_tokens=2, grammar=g), SamplingParams(max_new_tokens=40, stop=(" is",), grammar=g),
             SamplingParams(max_new_tokens=2, logprobs=0, grammar=g), SamplingParams(max_new_tokens=40, grammar=g)]
    refs, _ = run_steps(engine, [prompt] * len(cases), cases, jump=False)
    jfs, _ = run_steps(engine, [prompt] * len(cases), cases, jump=True)
    for ref, jf in zip(refs[:3], jfs[:3]):
        assert jf.output_ids == ref.output_ids and jf.finish_reason == ref.finish_reason, (
            engine.tokenizer.decode(jf.output_ids), engine.tokenizer.decode(ref.output_ids))
    assert (jfs[0].finish_reason, len(jfs[0].output_ids), jfs[0].num_forced) == ("length", 2, 1)
    assert jfs[1].finish_reason == "stop" and jfs[1].num_forced == 2
    assert engine.tokenizer.decode(jfs[1].output_ids).endswith(" is")
    assert jfs[2].num_forced == 0 and len(jfs[2].logprobs) == 2
    text = engine.tokenizer.decode(jfs[3].output_ids, skip_special_tokens=True)
    assert re.fullmatch(pat, text) and jfs[3].finish_reason == "stop" and jfs[3].num_forced > 0, text


def test_jump_forward_replaces_newest_token(engine):
    """The start boundary in the engine. Measured on Qwen3-0.6B (CPU, fp32, greedy): under
    `"name": [0-9]{2}` the prefill samples a lone `"`; the jump string `name": ` re-encodes with it
    as `"name` `":` ` `, which replaces it, and with logprobs the replacement is re-scored from the
    prompt's last row (its KV recomputed in place). Where that row's page belongs to the radix
    tree the replacement is refused and the step does not jump."""
    from kiln.engine.grammar import GrammarSpec
    from kiln.engine.request import SamplingParams

    regex = r'"name": [0-9]{2}'
    ps = engine.cfg.page_size
    quote = engine._encode('"')[0]

    def prompt_with(shared: bool):
        for n in range(4 * ps):
            ids = chat_ids(engine, 'Reply with "name": and a two-digit number.' + " ok" * n)
            if (len(ids) % ps == 0) == shared:
                return ids
        raise AssertionError("no prompt length found")

    sp = SamplingParams(max_new_tokens=8, logprobs=0, grammar=GrammarSpec("regex", regex))
    prompt = prompt_with(shared=False)
    (ref,), _ = run_steps(engine, [prompt], [sp], jump=False)
    assert ref.output_ids[0] == quote
    (jf,), _ = run_steps(engine, [prompt], [sp], jump=True)
    out = engine.tokenizer.decode(jf.output_ids, skip_special_tokens=True)
    assert re.fullmatch(regex, out), out
    assert jf.output_ids[:3] == engine._encode('"name": ') and jf.num_forced >= 3
    assert len(jf.logprobs) == len(jf.output_ids)
    # The replacement's logprob is the score of `"name` after the prompt (as prompt logprobs).
    name = jf.output_ids[0]
    (score,) = engine.generate([prompt + [name]], SamplingParams(max_new_tokens=1, prompt_logprobs=0,
                                                                 prompt_logprobs_start=len(prompt)))
    assert abs(jf.logprobs[0][0] - score.prompt_logprobs[len(prompt)][0]) < 1e-3

    prompt = prompt_with(shared=True)
    (jf,), _ = run_steps(engine, [prompt], [sp], jump=True)
    assert jf.output_ids[0] == quote  # not replaced: its row's KV is in a shared page
    assert re.fullmatch(regex, engine.tokenizer.decode(jf.output_ids, skip_special_tokens=True))
    assert len(jf.logprobs) == len(jf.output_ids)
