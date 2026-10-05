import json
import random

from kiln.server.parsers import StreamParser, parse_full

OUT = ('<think>\nThe user wants weather.\n</think>\n\nLet me check.\n<tool_call>\n'
       '{"name": "get_weather", "arguments": {"city": "Paris", "unit": "c"}}\n</tool_call>')


def test_full_parse_splits_reasoning_content_and_tools():
    m = parse_full(OUT, reasoning=True, tools=True, thinking_open=False)
    assert m["reasoning_content"] == "The user wants weather."
    assert m["content"] == "Let me check."
    (call,) = m["tool_calls"]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris", "unit": "c"}


def test_thinking_open_in_prompt():
    m = parse_full("reasoning here</think>\n\nanswer", reasoning=True, tools=False, thinking_open=True)
    assert m["reasoning_content"] == "reasoning here" and m["content"] == "answer"


def test_plain_text_is_untouched():
    m = parse_full("just an answer", reasoning=True, tools=True, thinking_open=False)
    assert m["content"] == "just an answer" and "tool_calls" not in m and "reasoning_content" not in m


def stream(text, chunks, **kw):
    p = StreamParser(**kw)
    deltas = []
    i = 0
    for n in chunks:
        deltas += p.push(text[i : i + n], final=False)
        i += n
    deltas += p.push(text[i:], final=True)
    return deltas


def test_streaming_matches_full_parse_for_any_chunking():
    rng = random.Random(0)
    for _ in range(200):
        chunks = []
        left = len(OUT)
        while left > 0:
            n = rng.randint(1, 7)
            chunks.append(min(n, left))
            left -= n
        deltas = stream(OUT, chunks[:-1], reasoning=True, tools=True, thinking_open=False)
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        content = "".join(d.get("content", "") for d in deltas)
        calls = [c for d in deltas for c in d.get("tool_calls", [])]
        assert reasoning.strip("\n") == "The user wants weather."
        assert content.strip() == "Let me check."
        assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"
        assert "<" not in content


def test_hermes_every_chunking_and_parallel_calls():
    from tests.parser_util import check_streaming

    two = OUT + '\n<tool_call>\n{"name": "get_time", "arguments": {"tz": "CET"}}\n</tool_call>'
    m = check_streaming(two, forbidden=("<think>", "</think>", "<tool_call>", "</tool_call>"),
                        reasoning="qwen3", tools="hermes")
    assert [c["function"]["name"] for c in m["tool_calls"]] == ["get_weather", "get_time"]
    assert m["tool_calls"][1]["function"]["arguments"] == '{"tz": "CET"}'
    m = parse_full(two, reasoning="qwen3", tools="hermes", parallel_tool_calls=False)
    assert [c["function"]["name"] for c in m["tool_calls"]] == ["get_weather"]


def test_hermes_malformed_json_is_content():
    """vLLM's hermes parser returns the output as content when a block is not JSON
    (hermes_tool_parser.py extract_tool_calls); Kiln does it per block."""
    bad = 'Sure.<tool_call>{"name": "get_weather", "arguments": {"city": </tool_call>'
    m = parse_full(bad, reasoning="qwen3", tools="hermes", thinking_open=False)
    assert "tool_calls" not in m and m["content"] == bad
    deltas = stream(bad, range(1, len(bad)), reasoning="qwen3", tools="hermes", thinking_open=False)
    assert "".join(d.get("content", "") for d in deltas) == bad
    assert not any("tool_calls" in d for d in deltas)


def test_tool_choice_rules():
    from kiln.server.parsers import chat_finish_reason, tool_choice_error

    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    assert tool_choice_error("auto", tools) is None and tool_choice_error("required", tools) is None
    assert tool_choice_error({"type": "function", "function": {"name": "get_weather"}}, tools) is None
    assert "does not match" in tool_choice_error({"type": "function", "function": {"name": "nope"}}, tools)
    assert "Invalid value for `tool_choice`" in tool_choice_error("sometimes", tools)
    assert "Expected field `name`" in tool_choice_error({"type": "function", "function": {}}, tools)
    # "tool_calls" once a call was parsed; a named choice keeps "stop" (vLLM serving.py).
    assert chat_finish_reason("stop", True, "auto") == "tool_calls"
    assert chat_finish_reason("length", True, "required") == "tool_calls"
    assert chat_finish_reason("stop", True, {"type": "function", "function": {"name": "get_weather"}}) == "stop"
    assert chat_finish_reason("stop", False, "auto") == "stop" and chat_finish_reason(None, True, None) is None
