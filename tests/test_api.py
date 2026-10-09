"""The OpenAI-compatible server over a real engine (CPU, real checkpoint).

Gated on KILN_TEST_MODEL like the parity tests.
"""

import json
import math
import os

import pytest
import torch

MODEL = os.environ.get("KILN_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set KILN_TEST_MODEL to run")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.server.api import build_app

    eng = LLMEngine(EngineConfig(model_path=MODEL, device="cpu", dtype=torch.float32, page_size=8,
                                 num_pages=256, max_num_seqs=4, max_model_len=512, max_prefill_tokens=64))
    with TestClient(build_app(eng, "test-model")) as c:
        c.engine = eng
        yield c


def test_completion_greedy_with_logprobs(client):
    r = client.post("/v1/completions", json={"prompt": "The capital of France is", "max_tokens": 4,
                                             "temperature": 0, "logprobs": 3})
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    assert "Paris" in choice["text"]
    lp = choice["logprobs"]
    assert len(lp["token_logprobs"]) == 4 and all(len(t) == 3 for t in lp["top_logprobs"])
    # Greedy: the chosen token is the most likely one.
    for chosen, top in zip(lp["token_logprobs"], lp["top_logprobs"]):
        assert math.isclose(chosen, max(top.values()), abs_tol=1e-5)
    assert r.json()["usage"]["completion_tokens"] == 4


def test_stop_string_is_cut(client):
    r = client.post("/v1/completions", json={"prompt": "1, 2, 3, 4,", "max_tokens": 30,
                                             "temperature": 0, "stop": [" 7"]})
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "7" not in choice["text"] and choice["text"].strip().endswith("6,")


def test_streaming_chat(client):
    body = {"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 12,
            "temperature": 0, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    chunks, done = [], False
    with client.stream("POST", "/v1/chat/completions", json=body) as r:
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                done = True
                break
            chunks.append(json.loads(line[6:]))
    assert done and chunks
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text.strip()
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")


@pytest.fixture(scope="module")
def agent_client():
    from fastapi.testclient import TestClient

    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.server.api import build_app

    eng = LLMEngine(EngineConfig(model_path=MODEL, device="cpu", dtype=torch.float32, page_size=8,
                                 num_pages=512, max_num_seqs=4, max_model_len=1024, max_prefill_tokens=128))
    with TestClient(build_app(eng, "test-model", reasoning_parser="qwen3", tool_call_parser="hermes")) as c:
        yield c


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def test_tool_call_is_parsed(agent_client):
    body = {"messages": [{"role": "user", "content": "What is the weather in Paris right now?"}],
            "tools": TOOLS, "max_tokens": 200, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    r = agent_client.post("/v1/chat/completions", json=body)
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls", choice
    call = choice["message"]["tool_calls"][0]["function"]
    assert call["name"] == "get_weather" and "paris" in json.loads(call["arguments"])["city"].lower()


def test_reasoning_is_separated(agent_client):
    body = {"messages": [{"role": "user", "content": "What is 2+3? Reply with the number."}],
            "max_tokens": 300, "temperature": 0}
    msg = agent_client.post("/v1/chat/completions", json=body).json()["choices"][0]["message"]
    assert msg.get("reasoning_content"), msg
    assert "<think>" not in (msg["content"] or "") and "</think>" not in (msg["content"] or "")


def test_sglang_generate_and_metrics(client):
    r = client.post("/generate", json={"text": "The capital of France is",
                                       "sampling_params": {"max_new_tokens": 4, "temperature": 0}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "Paris" in body["text"] and len(body["output_ids"]) == 4
    assert body["meta_info"]["finish_reason"]["type"] == "length"
    m = client.get("/metrics").text
    assert "kiln:generation_tokens_total" in m and "kiln:time_to_first_token_seconds_count" in m
    gen = [l for l in m.splitlines() if l.startswith("kiln:generation_tokens_total ")][0]
    assert int(gen.split()[-1]) >= 4
    # vllm-neuron's start-up and per-graph metrics (kiln/metrics.py startup_lines)
    assert "kiln:model_load_time_seconds " in m and "kiln:model_load_size_bytes " in m
    execs = [l for l in m.splitlines() if l.startswith("kiln:neff_execution_count{")]
    assert execs and sum(int(l.split()[-1]) for l in execs) >= 4
    assert any(l.startswith('kiln:compilation_time_seconds{bucket_name="') for l in m.splitlines())


def test_jump_forward_streams_aligned_logprobs(client):
    """Jump-forward tokens get their logprobs one step after they are appended; the stream holds
    them back until then, so every streamed token carries its own logprob."""
    schema = {"type": "object", "properties": {"city_name": {"type": "string"}, "country_code": {"type": "string"}},
              "required": ["city_name", "country_code"], "additionalProperties": False}
    body = {"messages": [{"role": "user", "content": "Describe Osaka as a JSON object."}], "max_tokens": 40,
            "temperature": 0, "logprobs": True, "top_logprobs": 2, "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "city", "schema": schema}}}

    def stream():
        text, entries = "", []
        with client.stream("POST", "/v1/chat/completions", json=dict(body, stream=True)) as r:
            for line in r.iter_lines():
                if line.startswith("data: {"):
                    c = json.loads(line[6:])["choices"][0]
                    text += c["delta"].get("content", "")
                    entries += (c.get("logprobs") or {}).get("content", [])
        return text, entries

    eng = client.engine
    ref_text, ref = stream()
    before = eng.jump_forward_tokens
    eng.cfg.jump_forward = True
    try:
        text, got = stream()
        full = client.post("/v1/chat/completions", json=body).json()
    finally:
        eng.cfg.jump_forward = False
    assert eng.jump_forward_tokens > before
    assert json.loads(text) and text == ref_text
    assert [e["token"] for e in got] == [e["token"] for e in ref]
    for a, b in zip(got, ref):
        assert abs(a["logprob"] - b["logprob"]) < 1e-3, (a, b)
    choice = full["choices"][0]
    assert choice["message"]["content"] == text
    assert [e["token"] for e in choice["logprobs"]["content"]] == [e["token"] for e in got]


def test_prompt_logprobs_vllm_and_sglang_shapes(client):
    """vLLM `prompt_logprobs` (completions: on the choice; chat: on the response) and SGLang
    `logprob_start_len` report the same numbers; streaming with prompt_logprobs > 0 is a 400
    as in vLLM."""
    prompt = "The capital of France is Paris, and the capital of Spain is"
    r = client.post("/v1/completions", json={"prompt": prompt, "max_tokens": 2, "temperature": 0,
                                             "prompt_logprobs": 2})
    assert r.status_code == 200, r.text
    plp = r.json()["choices"][0]["prompt_logprobs"]
    assert plp[0] is None and all(isinstance(p, dict) for p in plp[1:])
    for p in plp[1:]:  # the prompt token plus the top 2 (which may include it)
        assert len(p) in (2, 3) and {1, 2} <= {v["rank"] for v in p.values()}
    paris = [p for p in plp[1:] if any(v["decoded_token"] == " Paris" for v in p.values())]
    assert paris and max(v["logprob"] for v in paris[0].values()) > -2  # " Paris" is likely here
    g = client.post("/generate", json={"text": prompt, "return_logprob": True, "logprob_start_len": 3,
                                       "top_logprobs_num": 2, "sampling_params": {"max_new_tokens": 2,
                                                                                  "temperature": 0}})
    meta = g.json()["meta_info"]
    got = meta["input_token_logprobs"]
    assert len(got) == len(plp) - 3 and len(meta["input_top_logprobs"]) == len(got)
    for (lp, tid), p in zip(got, plp[3:]):
        assert abs(lp - p[str(tid)]["logprob"]) < 1e-4
    chat = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}],
                                                     "max_tokens": 1, "prompt_logprobs": 0})
    body = chat.json()
    assert body["prompt_logprobs"][0] is None and all(len(p) == 1 for p in body["prompt_logprobs"][1:])
    bad = client.post("/v1/completions", json={"prompt": prompt, "stream": True, "prompt_logprobs": 1})
    assert bad.status_code == 400


def test_session_id_and_close_session(client):
    body = {"text": "A long agent conversation about capitals: France, Spain, Italy and Japan.",
            "session_id": "agent-7", "sampling_params": {"max_new_tokens": 2, "temperature": 0}}
    assert client.post("/generate", json=body).status_code == 200
    r = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 1, "session_id": "agent-7"})
    assert r.status_code == 200
    assert client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 1, "session_id": 5}).status_code == 400
    assert client.post("/close_session", json={"session_id": "agent-7"}).json() == {"success": True}
    assert client.post("/close_session", json={}).status_code == 400


def test_thinking_budget_needs_reasoning_boundaries(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}],
                                                  "max_tokens": 4, "thinking_token_budget": 3})
    assert r.status_code == 400 and "reasoning" in r.text


def test_watermark_detect_needs_a_config(client):
    assert client.post("/v1/watermark/detect", json={"token_ids": [1, 2, 3]}).status_code == 400


def test_sglang_score_api(client):
    """SGLang /v1/score: label probabilities after query + item over the full vocabulary,
    equal to the prompt logprob of the same sequence."""
    import math

    q = "The capital of France is"
    r = client.post("/v1/completions", json={"prompt": q + " Paris", "max_tokens": 1, "prompt_logprobs": 0})
    plp = r.json()["choices"][0]["prompt_logprobs"]
    tid = int(next(iter(plp[-1])))
    want = plp[-1][str(tid)]["logprob"]
    london = client.post("/v1/completions", json={"prompt": q + " London", "max_tokens": 1, "prompt_logprobs": 0})
    lid = int(next(iter(london.json()["choices"][0]["prompt_logprobs"][-1])))
    s = client.post("/v1/score", json={"query": q, "items": [""], "label_token_ids": [tid, lid],
                                       "return_token_logprobs": True}).json()
    assert s["object"] == "scoring" and abs(s["token_logprobs"][0][0] - want) < 1e-4
    assert abs(s["scores"][0][0] - math.exp(want)) < 1e-5
    soft = client.post("/v1/score", json={"query": q, "items": ["", " Rome is not, but"], "label_token_ids": [tid, lid],
                                          "apply_softmax": True}).json()["scores"]
    assert abs(sum(soft[0]) - 1) < 1e-6 and soft[0][0] > 0.9 and len(soft) == 2
    assert client.post("/v1/score", json={"query": q, "items": [""], "label_token_ids": [tid],
                                          "temperature": 0}).status_code == 400


def test_update_weights_from_disk_endpoint(client):
    r = client.post("/update_weights_from_disk", json={"model_path": MODEL, "weight_version": "same"})
    assert r.status_code == 200 and r.json()["success"] and r.json()["num_paused_requests"] == 0
    assert client.post("/update_weights_from_disk", json={}).status_code == 400
    bad = client.post("/update_weights_from_disk", json={"model_path": "/nonexistent/model"})
    assert bad.status_code == 400 and not bad.json()["success"]

# SGLang /v1/decisions (entrypoints/openai/serving_decisions.py, v0.5.21).

TICKET = ("I've been trying to connect my Stripe account for 3 days and the integration keeps "
          "failing. I'm losing sales.")
DECISION_QUESTIONS = [
    {"id": "team", "type": "choice", "question": "Which team should handle this ticket?",
     "options": [{"name": "billing", "description": "Payment or subscription issues"},
                 {"name": "technical", "description": "Bugs or integration problems"},
                 {"name": "sales"}]},
    {"id": "frustration", "type": "score", "question": "How frustrated is the customer?",
     "levels": ["Calm", "Frustrated but civil", "Very angry"]},
    {"id": "urgent", "type": "yes_no", "question": "The customer needs an answer today.",
     "no": "It can wait"},
]
# The user message of each question, as decision_models.mdx and SGLang's PROMPT_FIXTURES
# (test/registered/unit/entrypoints/openai/test_serving_decisions.py, v0.5.21) word it.
DECISION_MESSAGES = {
    "team": [TICKET, "", "Question: Which team should handle this ticket?",
             "A: billing - Payment or subscription issues", "B: technical - Bugs or integration problems",
             "C: sales", "Answer with the letter of one option only."],
    "frustration": [TICKET, "", "Question: How frustrated is the customer?", "0: Calm",
                    "1: Frustrated but civil", "2: Very angry", "Answer with the number of one level only."],
    "urgent": [TICKET, "", "Is the following true? The customer needs an answer today.", "no: It can wait",
               "Answer with yes or no only."],
}
DECISION_LABELS = {"team": (["billing", "technical", "sales"], ["A", "B", "C"]),
                   "frustration": (["0", "1", "2"], ["0", "1", "2"]),
                   "urgent": (["yes", "no"], ["yes", "no"])}


def _replay(client, answer, temperature=1.0):
    """The documented replay: the returned ids through /v1/score."""
    r = client.post("/v1/score", json={"query": [], "items": [answer["prompt_token_ids"]],
                                       "label_token_ids": [answer["label_token_ids"]], "apply_softmax": True,
                                       "temperature": temperature, "return_token_logprobs": True})
    assert r.status_code == 200, r.text
    return r.json()["scores"][0], r.json()["token_logprobs"][0]


def test_decisions_choice_score_yes_no(client):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    r = client.post("/v1/decisions", json={"input": TICKET, "questions": DECISION_QUESTIONS,
                                           "prompt_format_version": 1, "return_prompt_token_ids": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "decisions" and body["model"] == "default" and body["prompt_format_version"] == 1
    assert list(body["answers"]) == ["team", "frustration", "urgent"]
    answers = body["answers"]
    for qid, (names, tokens) in DECISION_LABELS.items():
        a = answers[qid]
        assert a["type"] == next(q["type"] for q in DECISION_QUESTIONS if q["id"] == qid)
        assert list(a["probabilities"]) == names
        assert abs(math.fsum(a["probabilities"].values()) - 1) < 1e-9
        assert 0 < a["label_mass"] <= 1 + 1e-6
        # Labels are the vocabulary tokens after the non-thinking prompt.
        assert a["label_token_ids"] == tok.convert_tokens_to_ids(tokens)
        prompt = tok.decode(a["prompt_token_ids"])
        want = tok.apply_chat_template([{"role": "user", "content": "\n".join(DECISION_MESSAGES[qid])}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert prompt == want and prompt.endswith("<think>\n\n</think>\n\n")
        # Replaying the ids through /v1/score reproduces the answer (up to prefix-cache numerics).
        scores, lps = _replay(client, a)
        assert all(abs(x - y) < 1e-4 for x, y in zip(scores, a["probabilities"].values()))
        assert abs(math.fsum(math.exp(x) for x in lps) - a["label_mass"]) < 1e-4
    team = answers["team"]
    assert team["choice"] == max(team["probabilities"], key=team["probabilities"].get) and "score" not in team
    fr = answers["frustration"]
    assert abs(fr["score"] - sum(int(k) * p for k, p in fr["probabilities"].items())) < 1e-12 and "choice" not in fr
    assert set(answers["urgent"]) == {"type", "probabilities", "label_mass", "prompt_token_ids", "label_token_ids"}
    n = sum(len(a["prompt_token_ids"]) for a in answers.values())
    assert body["usage"] == {"prompt_tokens": n, "total_tokens": n, "completion_tokens": 0, "reasoning_tokens": 0}
    # Without return_prompt_token_ids the ids are left out; an explicit enable_thinking=False is
    # accepted and renders the same prompt.
    plain = client.post("/v1/decisions", json={"input": TICKET, "questions": DECISION_QUESTIONS[2:],
                                               "chat_template_kwargs": {"enable_thinking": False},
                                               "model": "my-model"}).json()
    assert plain["model"] == "my-model"
    assert set(plain["answers"]["urgent"]) == {"type", "probabilities", "label_mass"}
    assert abs(plain["answers"]["urgent"]["probabilities"]["yes"] - answers["urgent"]["probabilities"]["yes"]) < 1e-4


def test_decisions_temperature_scales_probabilities_not_label_mass(client):
    q = {"input": "The integration keeps failing.", "return_prompt_token_ids": True,
         "questions": [{"id": "u", "type": "yes_no", "question": "The customer needs an answer today."}]}
    cold = client.post("/v1/decisions", json=q).json()["answers"]["u"]
    hot = client.post("/v1/decisions", json=dict(q, temperature=2.0)).json()["answers"]["u"]
    assert hot["prompt_token_ids"] == cold["prompt_token_ids"]
    assert abs(hot["label_mass"] - cold["label_mass"]) < 1e-4
    scores, lps = _replay(client, hot, temperature=2.0)
    # softmax(lp / T) over the labels, which equals a softmax of the label logits / T.
    w = [math.exp(x / 2.0) for x in lps]
    assert all(abs(x / sum(w) - p) < 1e-4 for x, p in zip(w, hot["probabilities"].values()))
    assert all(abs(x - p) < 1e-4 for x, p in zip(scores, hot["probabilities"].values()))
    p_cold, p_hot = cold["probabilities"]["yes"], hot["probabilities"]["yes"]
    assert abs(p_hot - 0.5) < abs(p_cold - 0.5)  # a higher temperature flattens the answer


YES_NO = [{"id": "u", "type": "yes_no", "question": "Urgent?"}]


@pytest.mark.parametrize("body, reason", [
    ({"input": "s", "questions": [{"id": "u", "type": "rank", "question": "Q"}]}, "does not match any of the expected tags"),
    ({"input": "s", "questions": YES_NO, "top_p": 0.5}, "Extra inputs are not permitted"),
    ({"input": "s", "questions": [dict(YES_NO[0], true="x")]}, "Extra inputs are not permitted"),
    ({"input": "s", "questions": [{"id": "c", "type": "choice", "question": "Q", "options": [{"name": "a"}]}]},
     "at least 2 items"),
    ({"input": "s", "questions": [{"id": "c", "type": "choice", "question": "Q",
                                   "options": [{"name": f"o{i}"} for i in range(27)]}]}, "at most 26 items"),
    ({"input": "s", "questions": [{"id": "l", "type": "score", "question": "Q", "levels": [str(i) for i in range(11)]}]},
     "at most 10 items"),
    ({"input": "s", "questions": [{"id": "l", "type": "score", "question": "Q", "levels": ["low", " "]}]}, "must not be blank"),
    ({"input": " ", "questions": YES_NO}, "must not be blank"),
    ({"input": "s", "questions": [dict(YES_NO[0], question={})]}, "must not be blank"),
    ({"input": "s", "questions": [dict(YES_NO[0], id=" ")]}, "must not be blank"),
    ({"input": "s", "questions": []}, "at least 1 item"),
    ({"input": "s", "questions": YES_NO * 2}, "question id 'u' repeats another question"),
    ({"input": "s", "questions": [{"id": "c", "type": "choice", "question": "Q", "options": [{"name": "a"}, {"name": " A"}]}]},
     "option name ' A' repeats another option"),
    ({"input": "s", "questions": [{"id": "c", "type": "choice", "question": "Q",
                                   "options": [{"name": "a\nB: b"}, {"name": "c"}]}]}, "control or line break"),
    ({"input": "s", "questions": YES_NO, "temperature": 0}, "greater than 0"),
    ({"input": "s", "questions": YES_NO, "chat_template_kwargs": {"enable_thinking": True}},
     "chat_template_kwargs sets 'enable_thinking' to True, but decisions need it false or unset"),
    ({"input": "s", "questions": YES_NO, "chat_template_kwargs": {"enable_thinking": None}}, "sets 'enable_thinking' to None"),
    # The version is checked before the reasoning settings.
    ({"input": "s", "questions": YES_NO, "prompt_format_version": 2, "chat_template_kwargs": {"enable_thinking": True}},
     "prompt_format_version 2 is not served, this server uses version 1"),
    ({"input": "s", "questions": YES_NO, "model": "base:adapter"}, "model names the LoRA adapter 'adapter'"),
    ({"input": "word " * 600, "questions": YES_NO}, "question 'u': the prompt has"),
])
def test_decisions_refusals(client, body, reason):
    r = client.post("/v1/decisions", json=body)
    assert r.status_code == 400, r.text
    err = r.json()
    assert err["object"] == "error" and err["code"] == 400 and reason in err["message"], err["message"]


def test_decisions_template_refusals():
    """Refusals that depend on the chat template, on the real tokenizer with SGLang's test
    templates (test_serving_decisions.py, v0.5.21)."""
    from transformers import AutoTokenizer

    from kiln.server.decisions import Decider, DecisionRequest

    tok = AutoTokenizer.from_pretrained(MODEL)
    choice = DecisionRequest.model_validate(
        {"input": "s", "questions": [{"id": "first", "type": "choice", "question": "Q",
                                      "options": [{"name": "a"}, {"name": "b"}]}]})
    no_block = ("{% for m in messages %}{% if m['role'] == 'user' %}{{ m['content'] }}\nassistant:\n\n"
                "{% else %}<think></think>{{ m['content'] }}{% endif %}{% endfor %}")
    cases = [
        # The label merges with the trailing space.
        ("{{ messages[0]['content'] }}\nAnswer: ", None, "question 'first': the answer label 'A' is not one distinct token"),
        # Reasoning the detected toggle does not control.
        ("{{ messages[0]['content'] }}\nassistant\n<think>\n", None, "leaves a reasoning block open"),
        # A parser whose answers start inside reasoning needs a closed block.
        (no_block, "deepseek_r1", "expects answers to start with a reasoning block"),
        # A template that opens reasoning before every answer.
        ("{% for m in messages %}{% if m['role'] == 'user' %}{{ m['content'] }}\nassistant:\n\n{% else %}"
         "<think></think>{{ m['content'] }}\n{% endif %}{% endfor %}", "qwen3", "starts every answer with a reasoning block"),
        ("{% for m in messages %}{{ m['content'] }}{% endfor %}"
         "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n<think>\\n' }}{% endif %}", None,
         "does not support chat templates that always reason before answering"),
    ]
    for template, parser, reason in cases:
        tok.chat_template = template
        d = Decider(tok, 512, parser)
        with pytest.raises(ValueError, match=reason):
            if (msg := d.validate(choice)) is not None:
                raise ValueError(msg)
            list(d.questions(choice))
    # Accepted: a closed block for that parser, and the same template without the reply's block.
    for template, parser in (("{{ messages[0]['content'] }}\nassistant:\n<think>\n\n</think>\n\n", "deepseek_r1"),
                             ("{{ messages[0]['content'] }}\nassistant:\n\n", "qwen3")):
        tok.chat_template = template
        d = Decider(tok, 512, parser)
        assert d.validate(choice) is None
        (_, _, prompt_ids, label_ids), = d.questions(choice)
        assert label_ids == tok.convert_tokens_to_ids(["A", "B"])


def test_decisions_prompt_wording():
    """PROMPT_FORMAT_VERSION 1 wording for objects, descriptions and yes or no details
    (SGLang PROMPT_FIXTURES, test_serving_decisions.py, v0.5.21)."""
    from kiln.server.decisions import PROMPT_FORMAT_VERSION, DecisionRequest, question_view, render_question

    req = DecisionRequest.model_validate({"input": {"ticket": "Refund please", "tags": ["billing"]}, "questions": [
        {"id": "choice", "type": "choice", "question": {"question": "Which team?"},
         "options": [{"name": "billing", "description": "Payments"}, {"name": "sales"},
                     {"name": "other", "description": {"k": 1}}]},
        {"id": "score", "type": "score", "question": "Mood?", "levels": ["Calm", "Angry"]},
        {"id": "yes_no", "type": "yes_no", "question": "Urgent", "no": "Can wait"}]})
    text = '{"ticket":"Refund please","tags":["billing"]}'
    want = {
        "choice": ['Question: {"question":"Which team?"}', "A: billing - Payments", "B: sales", 'C: other - {"k":1}',
                   "Answer with the letter of one option only."],
        "score": ["Question: Mood?", "0: Calm", "1: Angry", "Answer with the number of one level only."],
        "yes_no": ["Is the following true? Urgent", "no: Can wait", "Answer with yes or no only."],
    }
    labels = {"choice": ["A", "B", "C"], "score": ["0", "1"], "yes_no": ["yes", "no"]}
    assert PROMPT_FORMAT_VERSION == 1
    for q in req.questions:
        assert render_question(text, question_view(q), labels[q.id]) == "\n".join([text, "", *want[q.id]])


def test_tokenize_and_detokenize(client):
    """vLLM's /tokenize and /detokenize shapes (v0.24.0 vllm/entrypoints/serve/tokenize/protocol.py)."""
    tok = client.engine.tokenizer
    r = client.post("/tokenize", json={"prompt": "The capital of France is", "return_token_strs": True})
    assert r.status_code == 200, r.text
    body = r.json()
    ids = tok("The capital of France is")["input_ids"]
    assert body["tokens"] == ids and body["count"] == len(ids)
    assert body["max_model_len"] == client.engine.cfg.max_model_len
    assert body["token_strs"] == tok.convert_ids_to_tokens(ids)
    r = client.post("/tokenize", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and r.json()["count"] > 1 and "token_strs" not in r.json()
    r = client.post("/detokenize", json={"tokens": ids})
    assert r.status_code == 200 and r.json()["prompt"] == tok.decode(ids)
    assert client.post("/detokenize", json={"tokens": [-1]}).status_code == 400
    assert client.post("/tokenize", json={}).status_code == 400
