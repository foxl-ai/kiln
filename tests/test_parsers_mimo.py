"""MiMo-V2 / Qwen3-Coder XML tool calls and MiMo reasoning (kiln/server/parsers.py).

Outputs are written the way the chat templates write an assistant turn:
- MiMo-V2.6 (render_assistant_message / render_tool_calls,
  https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/5711b268169967567844e1e560e8a3966da959b1/chat_template.jinja):
  `<think>R</think>C<tool_call><function=N><parameter=K>V</parameter></function></tool_call>`,
  strings raw, other values tojson. The generation prompt ends `<|im_start|>assistant\\n`
  (the model opens `<think>` itself) or, with enable_thinking=false, `<think></think>`.
  Rendered with transformers 5.15.0 on 2026-10-02 to confirm both.
- MiMo-V2-Flash (tokenizer_config.json chat_template,
  https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/blob/1afd314a2406c282e0956375c34a676501c78649/tokenizer_config.json):
  `<tool_call>\\n<function=N>\\n<parameter=K>V</parameter>\\n</function>\\n</tool_call>`.
"""

import json

from kiln.server.parsers import REASONING_PARSERS, TOOL_PARSERS, StreamParser, parse_full
from tests.parser_util import args, check_streaming, collect, names, stream

TAGS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=",
        "</parameter>")
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean"},
        "opts": {"type": "object"}, "hours": {"type": "array", "items": {"type": "integer"}},
        "lat": {"type": "number"}, "note": {"type": ["string", "null"]}}}}},
    {"type": "function", "function": {"name": "run", "parameters": {
        "type": "object", "properties": {"cmd": {"type": "string"}}}}}]
KW = dict(reasoning="mimo", tools="mimo", tool_defs=TOOLS)

V26 = ("<think>The user wants the weather in Paris.</think>Let me check."
       "<tool_call><function=get_weather><parameter=city>Paris</parameter><parameter=days>3</parameter>"
       '<parameter=opts>{"unit": "c"}</parameter></function></tool_call>')


def test_registered_under_vllm_and_sglang_names():
    for name in ("mimo", "qwen3_coder", "qwen3_xml"):
        assert TOOL_PARSERS[name] is TOOL_PARSERS["mimo"]
    assert REASONING_PARSERS["mimo"] == REASONING_PARSERS["qwen3"]


def test_v26_thinking_on():
    m = check_streaming(V26, forbidden=TAGS, **KW)
    assert m["reasoning_content"] == "The user wants the weather in Paris."
    assert m["content"] == "Let me check."
    assert names(m) == ["get_weather"]
    # Schema types applied, separators as json.dumps.
    assert m["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris", "days": 3, "opts": {"unit": "c"}}'
    assert m["tool_calls"][0]["id"].startswith("call_")


def test_v26_thinking_off_prompt_closed_the_block():
    """enable_thinking=false: the prompt ends `<think></think>`, the output is the answer."""
    m = check_streaming("Paris is sunny.", forbidden=TAGS, **KW)
    assert "reasoning_content" not in m and m["content"] == "Paris is sunny."
    m = check_streaming("Checking.<tool_call><function=run><parameter=cmd>ls</parameter></function></tool_call>",
                        forbidden=TAGS, **KW)
    assert "reasoning_content" not in m and m["content"] == "Checking." and args(m) == [{"cmd": "ls"}]


def test_empty_reasoning_block():
    m = check_streaming("<think></think>Hello.", forbidden=TAGS, **KW)
    assert m["reasoning_content"] == "" and m["content"] == "Hello."


def test_truncated_reasoning_is_all_reasoning():
    m = check_streaming("<think>Still thinking about", forbidden=TAGS, **KW)
    assert m["reasoning_content"] == "Still thinking about" and m["content"] is None


def test_v2_flash_layout_with_newlines():
    out = ("<tool_call>\n<function=get_weather>\n<parameter=city>New York</parameter>\n"
           "<parameter=days>2</parameter>\n</function>\n</tool_call>")
    m = check_streaming(out, forbidden=TAGS, **KW)
    assert args(m) == [{"city": "New York", "days": 2}] and m["content"] is None


def test_one_wrapping_newline_trimmed_rest_preserved():
    """qwen3.py _trim_wrapping_newlines: one leading and one trailing newline per value."""
    out = "<tool_call><function=run><parameter=cmd>\n\nline 1\n  line 2\n\n</parameter></function></tool_call>"
    m = check_streaming(out, forbidden=TAGS, **KW)
    assert args(m) == [{"cmd": "\nline 1\n  line 2\n"}]


def test_values_with_angle_brackets():
    out = "<tool_call><function=run><parameter=cmd>if a<b and c>d: echo '<x>'</parameter></function></tool_call>"
    m = check_streaming(out, forbidden=("<tool_call>", "</tool_call>"), **KW)
    assert args(m) == [{"cmd": "if a<b and c>d: echo '<x>'"}]


def test_schema_types():
    out = ("<tool_call><function=get_weather><parameter=city>42</parameter><parameter=days>7</parameter>"
           "<parameter=metric>true</parameter><parameter=hours>[1, 2]</parameter><parameter=lat>48.85</parameter>"
           "<parameter=note>null</parameter><parameter=extra>5</parameter></function></tool_call>")
    m = check_streaming(out, forbidden=TAGS, **KW)
    # city is a string by schema; an undeclared key stays a string (vLLM _fix_arg_types).
    assert args(m) == [{"city": "42", "days": 7, "metric": True, "hours": [1, 2], "lat": 48.85, "note": None,
                        "extra": "5"}]


def test_bad_value_for_its_type_stays_a_string():
    out = "<tool_call><function=get_weather><parameter=days>three</parameter></function></tool_call>"
    assert args(parse_full(out, **KW)) == [{"days": "three"}]


def test_no_tool_definitions_keeps_strings():
    out = "<tool_call><function=get_weather><parameter=days>3</parameter></function></tool_call>"
    m = check_streaming(out, forbidden=TAGS, reasoning="mimo", tools="mimo")
    assert args(m) == [{"days": "3"}]


def test_parallel_calls_have_stable_indices_and_ids():
    out = ("Two cities.<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>"
           "<tool_call><function=get_weather><parameter=city>Rome</parameter></function></tool_call>")
    m = check_streaming(out, forbidden=TAGS, **KW)
    assert args(m) == [{"city": "Paris"}, {"city": "Rome"}] and m["content"] == "Two cities."
    assert len({c["id"] for c in m["tool_calls"]}) == 2
    _, _, calls = collect(stream(out, range(1, len(out)), **KW))
    assert [c["arguments"] for c in calls] == ['{"city": "Paris"}', '{"city": "Rome"}']


def test_two_functions_in_one_block():
    """vLLM qwen3.py (TOOL_BETWEEN, FUNC_PREFIX): another call before `</tool_call>`."""
    out = ("<tool_call><function=run><parameter=cmd>pwd</parameter></function>"
           "<function=run><parameter=cmd>ls</parameter></function></tool_call>")
    m = check_streaming(out, forbidden=TAGS, **KW)
    assert args(m) == [{"cmd": "pwd"}, {"cmd": "ls"}]


def test_parallel_tool_calls_false_keeps_the_first():
    out = ("<tool_call><function=run><parameter=cmd>pwd</parameter></function></tool_call>"
           "<tool_call><function=run><parameter=cmd>ls</parameter></function></tool_call>")
    m = check_streaming(out, forbidden=TAGS, parallel_tool_calls=False, **KW)
    assert args(m) == [{"cmd": "pwd"}]


def test_arguments_stream_incrementally():
    """The name is announced before the arguments finish, and a long string value streams in
    pieces rather than at `</tool_call>`."""
    value = "echo " + "x" * 40
    out = f"<tool_call><function=run><parameter=cmd>{value}</parameter></function></tool_call>"
    p = StreamParser(**KW)
    pieces = []
    for i, ch in enumerate(out):
        for d in p.push(ch, final=i == len(out) - 1):
            pieces += d.get("tool_calls", [])
    assert pieces[0]["function"]["name"] == "run"
    assert len(pieces) > 20
    assert "".join(c["function"]["arguments"] for c in pieces) == json.dumps({"cmd": value})


def test_non_string_value_waits_for_its_close():
    """An integer value is not streamed until `</parameter>`: 3 may still become 30."""
    p = StreamParser(**KW)
    sent = "".join(c["function"]["arguments"] for d in
                   p.push("<tool_call><function=get_weather><parameter=days>3", False) for c in d.get("tool_calls", []))
    assert sent == '{"days": '
    sent += "".join(c["function"]["arguments"] for d in p.push("0</parameter></function></tool_call>", True)
                    for c in d.get("tool_calls", []))
    assert json.loads(sent) == {"days": 30}


def test_block_without_a_function_is_content():
    """No `<function=` in the block: not a call, returned verbatim (vLLM and SGLang drop it)."""
    out = 'Sure.<tool_call>{"name": "run", "arguments": {"cmd": "ls"}}</tool_call>'
    m = check_streaming(out, **KW)
    assert "tool_calls" not in m
    assert m["content"] == out


def test_truncated_call_keeps_partial_argument():
    out = "<tool_call><function=get_weather><parameter=city>Par"
    m = check_streaming(out, forbidden=TAGS, **KW)
    assert args(m) == [{"city": "Par"}]


def test_tool_call_inside_reasoning_ends_it():
    """vLLM qwen3.py (REASONING, TOOL_START) and SGLang Qwen3Detector tool_start_token."""
    out = "<think>I should call it<tool_call><function=run><parameter=cmd>ls</parameter></function></tool_call>"
    m = check_streaming(out, forbidden=TAGS, **KW)
    assert m["reasoning_content"] == "I should call it" and args(m) == [{"cmd": "ls"}]


def test_without_reasoning_parser_think_is_content():
    out = "<think>r</think>x<tool_call><function=run><parameter=cmd>ls</parameter></function></tool_call>"
    m = check_streaming(out, tools="mimo", tool_defs=TOOLS)
    assert "reasoning_content" not in m and m["content"] == "<think>r</think>x" and args(m) == [{"cmd": "ls"}]


def test_without_tool_parser_markup_is_content():
    out = "<think>r</think>x<tool_call><function=run></function></tool_call>"
    m = check_streaming(out, reasoning="mimo")
    assert m["reasoning_content"] == "r" and m["content"] == "x<tool_call><function=run></function></tool_call>"


def test_duplicate_end_tag_dropped():
    m = check_streaming("<think>r</think>Answer.</think> Done.", forbidden=TAGS, **KW)
    assert m["content"] == "Answer. Done."


def test_repeated_parameter_streams_consistently():
    out = "<tool_call><function=run><parameter=cmd>pwd</parameter><parameter=cmd>ls</parameter></function></tool_call>"
    m = check_streaming(out, forbidden=TAGS, **KW)
    assert args(m) == [{"cmd": "ls"}]  # the last one wins, as vLLM's dict
