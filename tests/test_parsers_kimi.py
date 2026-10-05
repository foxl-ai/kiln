"""Kimi K2 reasoning and tool calls (kiln/server/parsers.py).

Outputs are written the way the chat templates write an assistant turn
(https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/chat_template.jinja,
https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/chat_template.jinja,
https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/chat_template.jinja):
`<think>R</think>C<|tool_calls_section_begin|><|tool_call_begin|>ID<|tool_call_argument_begin|>
ARGS<|tool_call_end|>...<|tool_calls_section_end|>`. K2-Thinking's generation prompt ends at
`<|im_middle|>` (the model opens `<think>`); K2.5's ends `<think>`, or `<think></think>` with
thinking=false. Expected values follow vLLM v0.30 tests/tool_parsers/test_kimi_k2_tool_parser.py
and tests/reasoning/test_kimi_k2_reasoning_parser.py where they have a case; `_tool` is
that file's helper (a space after the id).
"""

from kiln.server.parsers import REASONING_PARSERS, TOOL_PARSERS, parse_full
from tests.parser_util import args, check_streaming, names

SECTION_BEGIN, SECTION_END = "<|tool_calls_section_begin|>", "<|tool_calls_section_end|>"
TOOL_BEGIN, TOOL_END, ARG_BEGIN = "<|tool_call_begin|>", "<|tool_call_end|>", "<|tool_call_argument_begin|>"
TAGS = ("<think>", "</think>", SECTION_BEGIN, SECTION_END, TOOL_BEGIN, TOOL_END, ARG_BEGIN)
KW = dict(reasoning="kimi_k2", tools="kimi_k2")


def _tool(tool_id: str, arguments: str) -> str:
    return f"{TOOL_BEGIN}{tool_id} {ARG_BEGIN}{arguments}{TOOL_END}"


def _wrap(*tools: str) -> str:
    return SECTION_BEGIN + "".join(tools) + SECTION_END


def test_registered_under_vllm_and_sglang_names():
    assert "kimi_k2" in TOOL_PARSERS and REASONING_PARSERS["kimi_k2"].tool_start == SECTION_BEGIN


def test_thinking_model_with_a_call():
    out = "<think>Need the weather.</think>I'll check. " + _wrap(_tool("functions.get_weather:0", '{"city": "Beijing"}'))
    m = check_streaming(out, forbidden=TAGS, **KW)
    assert m["reasoning_content"] == "Need the weather."
    assert m["content"] == "I'll check. "  # kimi_k2 strip_content_whitespace_with_tools=False
    assert m["tool_calls"] == [{"id": "functions.get_weather:0", "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'}}]


def test_k25_thinking_on_prompt_opened_the_block():
    m = check_streaming("step by step reasoning</think>final answer", forbidden=TAGS, thinking_open=True, **KW)
    assert m["reasoning_content"] == "step by step reasoning" and m["content"] == "final answer"


def test_k25_thinking_off_and_k2_instruct():
    """thinking=false (the prompt ends `<think></think>`), or K2-Instruct, which never thinks."""
    m = check_streaming("final answer", forbidden=TAGS, **KW)
    assert "reasoning_content" not in m and m["content"] == "final answer"


def test_empty_thinking():
    m = check_streaming("<think></think>final answer", forbidden=TAGS, **KW)
    assert m["reasoning_content"] == "" and m["content"] == "final answer"


def test_tool_section_ends_reasoning():
    out = "<think>some reasoning" + _wrap(_tool("functions.get_weather:0", '{"city": "Tokyo"}'))
    m = check_streaming(out, forbidden=TAGS, **KW)
    assert m["reasoning_content"] == "some reasoning" and args(m) == [{"city": "Tokyo"}]
    assert m["content"] is None


def test_three_calls_keep_native_ids():
    out = "Multiple tasks. " + _wrap(
        _tool("functions.get_weather:0", '{"city": "New York"}'),
        _tool("functions.get_news:1", '{"topic": "technology"}'),
        _tool("functions.send_email:2", '{"to": "user@example.com", "subject": "Daily Update"}'))
    m = check_streaming(out, forbidden=TAGS, **KW)
    assert names(m) == ["get_weather", "get_news", "send_email"]
    assert [c["id"] for c in m["tool_calls"]] == ["functions.get_weather:0", "functions.get_news:1",
                                                  "functions.send_email:2"]
    assert args(m)[2] == {"to": "user@example.com", "subject": "Daily Update"}
    m = check_streaming(out, forbidden=TAGS, parallel_tool_calls=False, **KW)
    assert names(m) == ["get_weather"]


def test_multiline_and_angle_bracket_arguments():
    out = _wrap(_tool("functions.process_html:0", '{"html": "<div>content</div>"}'),
                _tool("functions.process_data:1", '{\n  "name": "test",\n  "value": 123\n}'))
    m = check_streaming(out, forbidden=TAGS, **KW)
    assert args(m) == [{"html": "<div>content</div>"}, {"name": "test", "value": 123}]


def test_id_without_functions_prefix_and_empty_arguments():
    m = check_streaming(_wrap(_tool("get_weather:0", '{"city": "Tokyo"}'), _tool("functions.test:1", "")),
                        forbidden=TAGS, **KW)
    assert names(m) == ["get_weather", "test"] and args(m) == [{"city": "Tokyo"}, {}]
    assert m["tool_calls"][0]["id"] == "get_weather:0"


def test_invalid_json_arguments_are_passed_through():
    """vLLM test_invalid_json_still_extracted: the call is kept, its arguments verbatim."""
    out = "Help. " + _wrap(_tool("functions.bad:0", '{"city": "Beijing"'), _tool("functions.good:1", '{"city": "Shanghai"}'))
    m = check_streaming(out, forbidden=TAGS, **KW)
    assert names(m) == ["bad", "good"]
    assert m["tool_calls"][0]["function"]["arguments"] == '{"city": "Beijing"'


def test_malformed_id_is_skipped():
    """vLLM test_invalid_funcall_id_skipped: a header without `:N` is not a call."""
    out = "Help. " + _wrap(_tool("functions.invalid.0", '{"city": "Beijing"}'), _tool("functions.valid:1", '{"city": "Shanghai"}'))
    m = check_streaming(out, forbidden=TAGS + ("invalid", "Beijing"), **KW)
    assert names(m) == ["valid"] and m["content"] == "Help. "


def test_noise_in_and_after_the_section_is_suppressed():
    out = "Before. " + SECTION_BEGIN + " spurious noise " + _tool("functions.test:0", '{"k": "v"} ') + SECTION_END \
        + " After tools."
    m = check_streaming(out, forbidden=TAGS + ("spurious", "After"), **KW)
    assert args(m) == [{"k": "v"}] and m["content"] == "Before. "


def test_empty_section():
    m = check_streaming("Reasoning. " + SECTION_BEGIN + SECTION_END, forbidden=TAGS, **KW)
    assert "tool_calls" not in m and m["content"] == "Reasoning. "


def test_truncated_call_keeps_partial_arguments():
    """vLLM test_truncated_tool_call_no_end_marker."""
    out = "I'll check. " + SECTION_BEGIN + TOOL_BEGIN + "functions.get_weather:0 " + ARG_BEGIN + '{"city": "Bei'
    m = check_streaming(out, forbidden=TAGS, **KW)
    assert m["tool_calls"][0]["id"] == "functions.get_weather:0"
    assert m["tool_calls"][0]["function"]["arguments"] == '{"city": "Bei'


def test_truncated_header_is_not_a_call():
    m = parse_full("Hi. " + SECTION_BEGIN + TOOL_BEGIN + "functions.get_wea", **KW)
    assert "tool_calls" not in m and m["content"] == "Hi. "


def test_without_reasoning_parser():
    out = "<think>r</think>x" + _wrap(_tool("functions.f:0", "{}"))
    m = check_streaming(out, tools="kimi_k2")
    assert "reasoning_content" not in m and m["content"] == "<think>r</think>x" and names(m) == ["f"]
