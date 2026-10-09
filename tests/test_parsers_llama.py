"""Llama 3.x JSON tool calls (kiln/server/parsers.py _Llama3JsonFormat; vLLM v0.24.0
vllm/tool_parsers/llama_tool_parser.py Llama3JsonToolParser)."""

from kiln.server.parsers import parse_full
from tests.parser_util import args, check_streaming, names

ONE = '{"name": "get_weather", "parameters": {"city": "Paris", "unit": "c"}}'


def test_one_call_every_chunking():
    m = check_streaming(ONE, tools="llama3_json")
    assert names(m) == ["get_weather"] and args(m) == [{"city": "Paris", "unit": "c"}]
    assert m["content"] is None


def test_semicolon_separated_calls_and_the_arguments_key():
    two = ONE + '; {"name": "get_time", "arguments": {"tz": "CET"}}'
    m = check_streaming(two, tools="llama3_json")
    assert names(m) == ["get_weather", "get_time"] and args(m)[1] == {"tz": "CET"}
    m = parse_full(two, tools="llama3_json", parallel_tool_calls=False)
    assert names(m) == ["get_weather"]


def test_leading_whitespace_still_opens_a_call():
    m = check_streaming("\n  " + ONE, tools="llama3_json")
    assert names(m) == ["get_weather"]


def test_a_brace_inside_text_is_content():
    """Only a message that starts with `{` is a tool call (vLLM's streaming rule)."""
    text = 'The set is {1, 2} and ' + ONE
    m = check_streaming(text, tools="llama3_json")
    assert "tool_calls" not in m and m["content"] == text


def test_json_without_a_name_or_cut_off_is_content():
    for text in ('{"answer": 42}', '{"name": "get_weather", "parameters": {"city": '):
        m = check_streaming(text, tools="llama3_json")
        assert "tool_calls" not in m and m["content"] == text


def test_aliases_name_the_same_format():
    for alias in ("llama3_json", "llama4_json", "llama3"):
        assert names(parse_full(ONE, tools=alias)) == ["get_weather"]
