"""GLM-4.5 / GLM-4.6 / GLM-4.7 / GLM-5 reasoning and tool calls (kiln/server/parsers.py).

Outputs are written the way the chat templates write an assistant turn:
- GLM-4.5 and GLM-4.6 (identical templates,
  https://huggingface.co/zai-org/GLM-4.5/blob/cbb2c7cfb52fa128a9660cb1a7a78e017899e115/chat_template.jinja):
  `\\n<think>R</think>\\nC\\n<tool_call>N\\n<arg_key>K</arg_key>\\n<arg_value>V</arg_value>\\n</tool_call>`;
  the generation prompt ends `<|assistant|>` (the model opens `<think>`), or
  `<|assistant|>\\n<think></think>` with enable_thinking=false.
- GLM-4.7 and GLM-5 (https://huggingface.co/zai-org/GLM-4.7/blob/602d01efcdd332c5238ca4bcede555defbe83eb7/chat_template.jinja,
  https://huggingface.co/zai-org/GLM-5/blob/c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2/chat_template.jinja):
  `<tool_call>N<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`; the generation
  prompt ends `<|assistant|><think>` (output starts inside reasoning) or `<|assistant|></think>`.
Strings are written raw, other values with tojson. Expected values follow vLLM v0.30
tests/tool_parsers/test_glm47_moe_tool_parser.py where it has a case.
"""

from kiln.server.parsers import REASONING_PARSERS, TOOL_PARSERS, parse_full
from tests.parser_util import args, check_streaming, names

TAGS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
        "</arg_value>")
TOOLS = [{"type": "function", "function": {"name": "get_current_date", "parameters": {}}},
         {"type": "function", "function": {"name": "get_weather", "parameters": {
             "type": "object", "properties": {"city": {"type": "string"}, "date": {"type": "string"},
                                              "days": {"type": "integer"}, "filters": {"type": "object"}}}}}]
KW = dict(reasoning="glm47", tools="glm47", tool_defs=TOOLS)


def test_registered_under_vllm_and_sglang_names():
    for name in ("glm", "glm45", "glm47"):
        assert TOOL_PARSERS[name] is TOOL_PARSERS["glm47"]
    assert REASONING_PARSERS["glm45"] is REASONING_PARSERS["glm47"]


def test_glm47_thinking_on_prompt_opened_the_block():
    out = ("The user wants Beijing weather.</think>Checking.<tool_call>get_weather<arg_key>city</arg_key>"
           "<arg_value>Beijing</arg_value><arg_key>days</arg_key><arg_value>2</arg_value></tool_call>")
    m = check_streaming(out, forbidden=TAGS, thinking_open=True, **KW)
    assert m["reasoning_content"] == "The user wants Beijing weather."
    assert m["content"] == "Checking."
    assert m["tool_calls"][0]["function"]["arguments"] == '{"city": "Beijing", "days": 2}'


def test_glm47_thinking_off():
    m = check_streaming("Beijing is sunny.", forbidden=TAGS, **KW)
    assert "reasoning_content" not in m and m["content"] == "Beijing is sunny."


def test_glm45_layout_thinking_on():
    out = ("\n<think>Need the tool.</think>\nChecking.\n<tool_call>get_weather\n<arg_key>city</arg_key>\n"
           "<arg_value>Beijing</arg_value>\n<arg_key>date</arg_key>\n<arg_value>2024-06-27</arg_value>\n</tool_call>")
    m = check_streaming(out, forbidden=TAGS, reasoning="glm45", tools="glm45", tool_defs=TOOLS)
    assert m["reasoning_content"] == "Need the tool." and m["content"] == "Checking."
    assert args(m) == [{"city": "Beijing", "date": "2024-06-27"}]


def test_glm45_thinking_off():
    """enable_thinking=false: the prompt ends `\\n<think></think>`."""
    m = check_streaming("\nBeijing is sunny.", forbidden=TAGS, reasoning="glm45", tools="glm45", tool_defs=TOOLS)
    assert "reasoning_content" not in m and m["content"] == "\nBeijing is sunny."


def test_zero_argument_calls():
    for out in ("<tool_call>get_current_date</tool_call>", "<tool_call>get_current_date\n</tool_call>"):
        m = check_streaming(out, forbidden=TAGS, **KW)
        assert names(m) == ["get_current_date"] and args(m) == [{}] and m["content"] is None


def test_whitespace_preserved_in_values():
    out = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>  Beijing  </arg_value></tool_call>"
    assert args(check_streaming(out, forbidden=TAGS, **KW)) == [{"city": "  Beijing  "}]


def test_whitespace_only_content_is_none():
    m = check_streaming("  \n  <tool_call>get_current_date</tool_call>", forbidden=TAGS, **KW)
    assert m["content"] is None and names(m) == ["get_current_date"]


def test_object_value_follows_the_schema():
    out = ('<tool_call>get_weather<arg_key>filters</arg_key><arg_value>{"rain": true, "hours": [9, 12]}</arg_value>'
           "<arg_key>city</arg_key><arg_value>Seoul</arg_value></tool_call>")
    m = check_streaming(out, forbidden=TAGS, **KW)
    assert args(m) == [{"filters": {"rain": True, "hours": [9, 12]}, "city": "Seoul"}]


def test_parallel_calls():
    out = ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
           "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Shanghai</arg_value></tool_call>")
    m = check_streaming(out, forbidden=TAGS, **KW)
    assert args(m) == [{"city": "Beijing"}, {"city": "Shanghai"}]
    assert len({c["id"] for c in m["tool_calls"]}) == 2
    m = check_streaming(out, forbidden=TAGS, parallel_tool_calls=False, **KW)
    assert args(m) == [{"city": "Beijing"}]


def test_unknown_function_is_content():
    """vLLM glm47_moe validate_tool_names and SGLang parse_base_json reject a name the request
    did not offer; Kiln returns the block as content rather than dropping it."""
    out = "Let me look.<tool_call>search_web<arg_key>q</arg_key><arg_value>x</arg_value></tool_call>"
    m = check_streaming(out, **KW)
    assert "tool_calls" not in m and m["content"] == out


def test_unknown_function_next_to_a_valid_call():
    out = ("<tool_call>search_web<arg_key>q</arg_key><arg_value>x</arg_value></tool_call>"
           "<tool_call>get_current_date</tool_call>")
    m = check_streaming(out, **KW)
    assert names(m) == ["get_current_date"]
    assert m["content"] == "<tool_call>search_web<arg_key>q</arg_key><arg_value>x</arg_value></tool_call>"


def test_block_without_a_name_is_content():
    m = check_streaming("<tool_call></tool_call>", **KW)
    assert "tool_calls" not in m and m["content"] == "<tool_call></tool_call>"


def test_tool_call_inside_reasoning_ends_it():
    out = "still thinking<tool_call>get_current_date</tool_call>"
    m = check_streaming(out, forbidden=TAGS, thinking_open=True, **KW)
    assert m["reasoning_content"] == "still thinking" and names(m) == ["get_current_date"]


def test_second_reasoning_block_reopens():
    """vLLM glm47_moe (CONTENT, THINK_START): a later `<think>` opens reasoning again."""
    out = "<think>first</think>Step one.<think>second</think>Step two."
    m = check_streaming(out, forbidden=TAGS, **KW)
    assert m["reasoning_content"] == "firstsecond" and m["content"] == "Step one.Step two."


def test_truncated_call_keeps_partial_argument():
    out = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Bei"
    assert args(check_streaming(out, forbidden=TAGS, **KW)) == [{"city": "Bei"}]


def test_no_tool_definitions_accepts_any_name():
    out = "<tool_call>anything<arg_key>n</arg_key><arg_value>5</arg_value></tool_call>"
    m = parse_full(out, reasoning="glm47", tools="glm47")
    assert names(m) == ["anything"] and args(m) == [{"n": "5"}]
