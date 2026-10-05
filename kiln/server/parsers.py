"""Reasoning and tool-call parsing for chat completions.

vLLM's `--reasoning-parser` / `--tool-call-parser` and SGLang's equivalents, registered under
both projects' names (REASONING_PARSERS, TOOL_PARSERS below):
- reasoning: `<think> ... </think>` ahead of the answer (Qwen3, MiMo, GLM-4.5 to GLM-5,
  Kimi K2, DeepSeek-R1). With thinking off the chat template closes the block in the PROMPT,
  so the output starts with the answer; templates that open it in the prompt leave the output
  inside it (`thinking_open`).
- tool calls: Hermes / Qwen2.5 JSON in `<tool_call>` blocks; MiMo / Qwen3-Coder XML
  (`<function=NAME><parameter=KEY>VALUE</parameter></function>`); GLM key / value tags
  (`NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>`); Kimi K2's special-token
  sections (`<|tool_call_begin|>functions.NAME:N<|tool_call_argument_begin|>{JSON}`).

The streaming parser is one state machine over the detokenized text for every format: it
holds back any suffix that could still become a tag, so a tag is never streamed as content
or reasoning. A tool call is announced (index, id, name) once its name is complete and its
arguments then stream as JSON deltas that only ever extend what was sent. The non-streaming
parse runs the same machine over the whole text, so both paths return the same calls.

Behaviour follows vLLM v0.30.0 (commit ced6857afa0ea7b2e3f0846a62e1394e90f15607:
vllm/parser/qwen3.py, glm47_moe.py, kimi_k2.py, vllm/parser/engine/parser_engine.py,
vllm/tool_parsers/utils.py) where vLLM and SGLang v0.5.21 (commit
e00930c5489053f26d86b179cee0d087f846acbb: python/sglang/srt/function_call/*_detector.py,
parser/reasoning_parser.py) agree or SGLang has no equivalent. Portions adapted from vLLM,
Copyright contributors to the vLLM project, Apache License 2.0; see THIRD_PARTY_NOTICES.md.
Deliberate differences, each with its reason:
- Whether the output starts inside a reasoning block is read from the rendered prompt and
  the output's first tag, not from a per-parser `enable_thinking` default (vLLM assumes
  thinking on, SGLang has per-parser defaults). The two agree whenever the model follows its
  chat template, and this way MiMo-V2-Flash (template default off) and MiMo-V2.6 (default
  on) both parse without a per-model switch.
- A tool block that yields no call (no function name; for GLM a name the request did not
  offer) is returned verbatim as content, as Kiln's Hermes path always did, instead of being
  dropped (vLLM and SGLang drop it). Kimi K2's own section framing is the exception: text
  between and after its calls is suppressed, as in both references.
- A call cut off by max_tokens keeps its partial last argument in both paths (vLLM's
  streaming and non-streaming disagree there), and Kimi arguments are passed through
  verbatim in both paths (vLLM retypes them only when not streaming).
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"

# MiMo-V2 / Qwen3-Coder XML. MiMo-V2.6 writes `<tool_call><function=NAME><parameter=KEY>VALUE
# </parameter></function></tool_call>` with no newlines (render_tool_calls,
# https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/blob/5711b268169967567844e1e560e8a3966da959b1/chat_template.jinja);
# MiMo-V2-Flash puts a newline after each tag (tokenizer_config.json chat_template,
# https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/blob/1afd314a2406c282e0956375c34a676501c78649/tokenizer_config.json).
# Strings are written raw, other values as JSON. `<think>` and `<tool_call>` are single added
# tokens with special=false, which the server's skip_special_tokens decode keeps (measured:
# tools/check_parser_formats.py --tokenizer). Tag spellings and the value regexes: vLLM
# https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/parser/qwen3.py (`mimo` maps to it in
# vllm/tool_parsers/__init__.py), SGLang
# https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/function_call/mimo_detector.py.
FUNC_OPEN, FUNC_CLOSE = "<function=", "</function>"
PARAM_OPEN, PARAM_CLOSE = "<parameter=", "</parameter>"
_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]*)>(.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
                       re.DOTALL)
_PARTIAL_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]+)>(.*)$", re.DOTALL)

# GLM-4.5 / GLM-4.6 write `<tool_call>NAME\n<arg_key>KEY</arg_key>\n<arg_value>VALUE</arg_value>\n
# </tool_call>`
# (https://huggingface.co/zai-org/GLM-4.5/blob/cbb2c7cfb52fa128a9660cb1a7a78e017899e115/chat_template.jinja,
# identical at https://huggingface.co/zai-org/GLM-4.6/blob/be72194883d968d7923a07e2f61681ea9a2826d1/chat_template.jinja);
# GLM-4.7 and GLM-5 drop the newlines
# (https://huggingface.co/zai-org/GLM-4.7/blob/602d01efcdd332c5238ca4bcede555defbe83eb7/chat_template.jinja,
# https://huggingface.co/zai-org/GLM-5/blob/c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2/chat_template.jinja).
# Strings raw, other values `tojson`. `<think>`, `<tool_call>` and the four arg tags are added
# tokens with special=false (tokenizer_config.json), so the server's skip_special_tokens decode
# keeps them (measured for MiMo's tokens, flagged the same, with tools/check_parser_formats.py
# --tokenizer; not run for GLM). Regexes: vLLM
# https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/parser/glm47_moe.py (glm45 and glm47
# both map to it), SGLang
# https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/function_call/glm4_moe_detector.py.
ARG_KEY_OPEN, ARG_KEY_CLOSE = "<arg_key>", "</arg_key>"
ARG_VALUE_OPEN, ARG_VALUE_CLOSE = "<arg_value>", "</arg_value>"
_ARG_RE = re.compile(r"<arg_key>(?P<key>.*?)</arg_key>\s*<arg_value>(?P<value>.*?)</arg_value>", re.DOTALL)
_PARTIAL_ARG_RE = re.compile(r"<arg_key>(?P<key>.*?)</arg_key>\s*<arg_value>(?P<value>.*)$", re.DOTALL)

# Kimi K2 writes `<|tool_calls_section_begin|><|tool_call_begin|>ID<|tool_call_argument_begin|>
# ARGS<|tool_call_end|>...<|tool_calls_section_end|>`, ID being the call id the client sent
# back, `functions.NAME:N`
# (https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/chat_template.jinja,
# https://huggingface.co/moonshotai/Kimi-K2-Thinking/blob/a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55/chat_template.jinja,
# https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/chat_template.jinja;
# Moonshot's own parser in
# https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/docs/tool_call_guidance.md).
# The five
# markers are added tokens with special=false, outside additional_special_tokens
# (tokenizer_config.json), and tokenization_kimi.py hands a skip_special_tokens decode to
# PreTrainedTokenizer.decode, which drops only special ids: the server's decode keeps them
# (read from the source; Kiln does not load Kimi K2 yet). Id regex and name rule: vLLM
# https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/parser/kimi_k2.py.
KIMI_SECTION_OPEN, KIMI_SECTION_CLOSE = "<|tool_calls_section_begin|>", "<|tool_calls_section_end|>"
KIMI_CALL_OPEN, KIMI_CALL_CLOSE = "<|tool_call_begin|>", "<|tool_call_end|>"
KIMI_ARGS_OPEN = "<|tool_call_argument_begin|>"
_KIMI_ID_RE = re.compile(r"(?P<id>.+:\d+)")


# ---------------------------------------------------------------------------------------
# Reasoning formats
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ReasoningFormat:
    start: str
    end: str
    # A tool-call opener that ends reasoning without the end tag: vLLM's (REASONING,
    # TOOL_START) transitions, SGLang's tool_start_token.
    tool_start: str | None = None
    # The output starts inside reasoning even without a start tag (DeepSeek-R1).
    always: bool = False
    # A start tag in the answer opens another reasoning block (vLLM glm47 (CONTENT,
    # THINK_START)).
    reopen: bool = False


_QWEN3_REASONING = ReasoningFormat(THINK_OPEN, THINK_CLOSE, tool_start=TOOL_OPEN)
_GLM_REASONING = ReasoningFormat(THINK_OPEN, THINK_CLOSE, tool_start=TOOL_OPEN, reopen=True)
REASONING_PARSERS = {
    "qwen3": _QWEN3_REASONING,
    # vLLM deepseek_r1_reasoning_parser.py, SGLang DeepSeekR1Detector (force_reasoning).
    "deepseek_r1": ReasoningFormat(THINK_OPEN, THINK_CLOSE, always=True),
    # vLLM `mimo` -> Qwen3ParserReasoningAdapter (vllm/reasoning/__init__.py); SGLang
    # _MimoDetector (Qwen3Detector tokens).
    "mimo": _QWEN3_REASONING,
    # vLLM glm45 / glm47 -> Glm47MoeParserReasoningAdapter; SGLang Glm45Detector.
    "glm45": _GLM_REASONING,
    "glm47": _GLM_REASONING,
    # vLLM kimi_k2 (vllm/parser/kimi_k2.py), SGLang KimiK2Detector: the tool section ends
    # reasoning.
    "kimi_k2": ReasoningFormat(THINK_OPEN, THINK_CLOSE, tool_start=KIMI_SECTION_OPEN),
}


# ---------------------------------------------------------------------------------------
# Argument typing (vLLM ParserEngine._fix_arg_types over vllm/tool_parsers/utils.py)
# ---------------------------------------------------------------------------------------

# utils.py _TYPE_ALIASES
_TYPE_ALIASES = {
    "str": "string", "text": "string", "varchar": "string", "char": "string", "enum": "string",
    "int": "integer", "int32": "integer", "int64": "integer", "uint": "integer", "uint32": "integer",
    "uint64": "integer", "long": "integer", "short": "integer", "unsigned": "integer",
    "float": "number", "float32": "number", "float64": "number", "double": "number",
    "bool": "boolean", "dict": "object", "arr": "array", "list": "array", "sequence": "array",
}


def _schema_types(schema) -> set[str]:
    """utils.py extract_types_from_schema: `type` (string or list), types implied by `enum`,
    and anyOf / oneOf / allOf, recursively; {"string"} when nothing is declared."""
    if not isinstance(schema, dict):
        return {"string"}
    types: set[str] = set()
    t = schema.get("type")
    if isinstance(t, str):
        types.add(t)
    elif isinstance(t, list):
        types.update(x for x in t if isinstance(x, str))
    enum = schema.get("enum")
    for v in enum if isinstance(enum, list) else ():
        types.add("null" if v is None else "boolean" if isinstance(v, bool) else "integer" if isinstance(v, int)
                  else "number" if isinstance(v, float) else "array" if isinstance(v, list)
                  else "object" if isinstance(v, dict) else "string")
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(schema.get(key), list):
            for choice in schema[key]:
                types |= _schema_types(choice)
    return types or {"string"}


def _finite(obj) -> bool:
    try:
        json.dumps(obj, allow_nan=False)
        return True
    except (ValueError, TypeError):
        return False


def _coerce(value: str, types: set[str]):
    """utils.py coerce_to_schema_type: the first type in null > integer > number > boolean >
    object > array > string order that the raw value converts to; else JSON, else the string."""
    norm = {_TYPE_ALIASES.get(t.strip().lower(), t.strip().lower()) for t in types}
    for t in ("null", "integer", "number", "boolean", "object", "array", "string"):
        if t not in norm:
            continue
        if t == "null":
            if value.lower() == "null":
                return None
        elif t == "string":
            return value
        elif t == "integer":
            try:
                return int(value)
            except ValueError:
                pass
        elif t == "number":
            try:
                f = float(value)
            except ValueError:
                continue
            if math.isfinite(f):
                return f if f != int(f) else int(f)
        elif t == "boolean":
            low = value.lower().strip()
            if low in ("true", "1"):
                return True
            if low in ("false", "0"):
                return False
        else:
            try:
                parsed = json.loads(value)
            except ValueError:
                continue
            if _finite(parsed):
                return parsed
    try:
        parsed = json.loads(value)
    except ValueError:
        return value
    return parsed if _finite(parsed) else value


def _args_json(items: list[tuple[str, str, bool]], complete: bool, props: dict) -> str:
    """The arguments JSON of an XML-style call from its (key, raw value, closed) items, values
    typed by the tool's schema (vLLM ParserEngine._fix_arg_types: only declared keys are
    retyped, the rest stay strings). While the call is open this is the prefix that cannot
    change (vLLM _safe_arg_prefix): closed values are final; the value still being written
    streams only when it will stay a string, otherwise its key waits with it; the closing
    brace waits for the call to end. Same separators as json.dumps, so the complete text is
    json.dumps of the arguments. A repeated key is written again rather than overwritten, so the
    stream stays a prefix of the final text (a JSON reader keeps the last one, as vLLM's dict
    does)."""
    parts = []
    for k, v, closed in items:
        key = json.dumps(k, ensure_ascii=False) + ": "
        schema = props.get(k)
        if closed or complete:
            value = _coerce(v, _schema_types(schema)) if isinstance(schema, dict) else v
            parts.append(key + json.dumps(value, ensure_ascii=False))
            continue
        if not isinstance(schema, dict) or _schema_types(schema) == {"string"}:
            parts.append(key + json.dumps(v, ensure_ascii=False)[:-1])
        else:
            parts.append(key)
        return "{" + ", ".join(parts)
    if complete:
        return "{" + ", ".join(parts) + "}"
    return "{" + ", ".join(parts) if parts else ""


# ---------------------------------------------------------------------------------------
# Tool-call formats
# ---------------------------------------------------------------------------------------

@dataclass
class _View:
    """One call found in a tool block so far."""
    name: str | None  # None while the name is still being written; "" when there is none
    args: str = ""  # raw argument text
    closed: bool = False  # the call's own end was seen
    call_id: str | None = None  # the model's own id (Kimi K2)
    ok: bool = True  # False: a malformed header, the call is skipped (Kimi K2)


class _ToolFormat:
    opener: str
    closer: str | None  # None: the block runs to the end of the output
    tags: tuple[str, ...]  # every tag that can appear in a block, held back when split
    strip_content: bool = True  # content is .strip()ped when calls were made (vLLM default)
    validate_names: bool = False  # vLLM validate_tool_names
    verbatim_on_failure: bool = True  # a block that yields no call is returned as content

    def scan(self, body: str, done: bool) -> list[_View]:
        raise NotImplementedError

    def arguments(self, view: _View, complete: bool, props: dict) -> str:
        raise NotImplementedError


class _HermesFormat(_ToolFormat):
    """`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` (Hermes 2 Pro, Qwen2.5, Qwen3):
    each block is one call, announced whole when it closes."""
    opener, closer, tags = TOOL_OPEN, TOOL_CLOSE, (TOOL_CLOSE,)

    def scan(self, body, done):
        if not done:
            return []
        try:
            obj = json.loads(body.strip())
        except json.JSONDecodeError:
            return []
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            return []
        args = obj.get("arguments", obj.get("parameters", {}))
        return [_View(obj["name"], args if isinstance(args, str) else json.dumps(args, ensure_ascii=False), True)]

    def arguments(self, view, complete, props):
        return view.args


class _XmlFormat(_ToolFormat):
    """MiMo-V2 / Qwen3-Coder: `<function=NAME>` calls with `<parameter=KEY>VALUE</parameter>`
    arguments inside a `<tool_call>` block (vLLM qwen3.py). Each value loses one wrapping
    newline at each end (qwen3.py _trim_wrapping_newlines, the MiMo-V2-Flash layout)."""
    opener, closer = TOOL_OPEN, TOOL_CLOSE
    tags = (TOOL_CLOSE, FUNC_OPEN, FUNC_CLOSE, PARAM_OPEN, PARAM_CLOSE)

    def scan(self, body, done):
        views, pos = [], 0
        while (i := body.find(FUNC_OPEN, pos)) >= 0:
            j = body.find(">", i)
            if j < 0:
                views.append(_View(None if not done else ""))
                break
            name = body[i + len(FUNC_OPEN):j].strip()
            k = body.find(FUNC_CLOSE, j)
            if k < 0:
                views.append(_View(name, body[j + 1:], False))
                break
            views.append(_View(name, body[j + 1:k], True))
            pos = k + len(FUNC_CLOSE)
        return views

    @staticmethod
    def _trim(value: str) -> str:
        if value.startswith("\n"):
            value = value[1:]
        return value[:-1] if value.endswith("\n") else value

    def arguments(self, view, complete, props):
        items = [(m.group(1).strip(), self._trim(m.group(2)), True) for m in _PARAM_RE.finditer(view.args)]
        m = _PARTIAL_PARAM_RE.search(_PARAM_RE.sub("", view.args))
        if m and m.group(1).strip():
            items.append((m.group(1).strip(), self._trim(m.group(2)), False))
        return _args_json(items, complete, props)


class _GlmFormat(_ToolFormat):
    """GLM-4.5 to GLM-5: `<tool_call>NAME` then `<arg_key>` / `<arg_value>` pairs (vLLM
    glm47_moe.py, which reads both the GLM-4.5 newline layout and GLM-4.7's inline one). The
    name ends at the first `<arg_key>`, newline or `</tool_call>`; names the request did not
    offer are not calls (glm47_moe validate_tool_names=True, SGLang parse_base_json)."""
    opener, closer = TOOL_OPEN, TOOL_CLOSE
    tags = (TOOL_CLOSE, ARG_KEY_OPEN, ARG_KEY_CLOSE, ARG_VALUE_OPEN, ARG_VALUE_CLOSE)
    validate_names = True

    def scan(self, body, done):
        i = body.find(ARG_KEY_OPEN)
        head = (body if i < 0 else body[:i]).lstrip()
        if i < 0 and not done and "\n" not in head:
            return [_View(None)]
        return [_View(head.split("\n", 1)[0].strip(), body[i:] if i >= 0 else "", done)]

    def arguments(self, view, complete, props):
        items = [(m.group("key").strip(), m.group("value"), True) for m in _ARG_RE.finditer(view.args)]
        m = _PARTIAL_ARG_RE.search(_ARG_RE.sub("", view.args))
        if m and m.group("key").strip():
            items.append((m.group("key").strip(), m.group("value"), False))
        return _args_json(items, complete, props)


class _KimiFormat(_ToolFormat):
    """Kimi K2: a `<|tool_calls_section_begin|>` section of `<|tool_call_begin|>ID
    <|tool_call_argument_begin|>JSON<|tool_call_end|>` calls (vLLM kimi_k2.py). Once the section
    opens the output never returns to content: text outside calls, and after the section, is
    suppressed (kimi_k2_config keeps TOOL_PREAMBLE). The id is the model's own
    `functions.NAME:N`, the name its first component; a header without `:N` is skipped. The
    arguments are passed through verbatim, stripped, `{}` when empty (kimi_k2.py
    _extract_args_json), invalid JSON included."""
    opener, closer = KIMI_SECTION_OPEN, None
    tags = (KIMI_SECTION_OPEN, KIMI_SECTION_CLOSE, KIMI_CALL_OPEN, KIMI_CALL_CLOSE, KIMI_ARGS_OPEN)
    strip_content = False  # kimi_k2_config strip_content_whitespace_with_tools=False
    verbatim_on_failure = False

    def scan(self, body, done):
        parts = body.split(KIMI_CALL_OPEN)[1:]
        views = []
        for n, part in enumerate(parts):
            ended = done or n < len(parts) - 1  # cut off by the end or by the next call
            ends = [x for x in (part.find(KIMI_CALL_CLOSE), part.find(KIMI_SECTION_CLOSE)) if x >= 0]
            e = min(ends) if ends else -1
            a = part.find(KIMI_ARGS_OPEN)
            if a >= 0 and (e < 0 or a < e):
                header, args, closed = part[:a], part[a + len(KIMI_ARGS_OPEN):e if e >= 0 else len(part)], e >= 0
            elif e >= 0:
                header, args, closed = part[:e], "", True
            elif ended:
                header, args, closed = part, "", True
            else:
                views.append(_View(None))
                break
            m = _KIMI_ID_RE.match(header.strip())
            if m is None:
                views.append(_View("", ok=False))
                continue
            call_id = m.group("id").strip()
            views.append(_View(call_id.split(":")[0].removeprefix("functions."), args, closed or ended, call_id))
        return views

    def arguments(self, view, complete, props):
        s = view.args.lstrip()
        return (s.strip() or "{}") if complete else s.rstrip()


_HERMES, _XML, _GLM, _KIMI = _HermesFormat(), _XmlFormat(), _GlmFormat(), _KimiFormat()
TOOL_PARSERS = {
    "hermes": _HERMES,  # vLLM, SGLang
    "qwen25": _HERMES,  # SGLang Qwen25Detector
    "qwen": _HERMES,  # SGLang
    "qwen3": _HERMES,  # Kiln: Qwen3 writes the Hermes format
    "mimo": _XML,  # vLLM -> Qwen3EngineToolParser, SGLang MiMoDetector
    "qwen3_coder": _XML,  # vLLM, SGLang Qwen3CoderDetector
    "qwen3_xml": _XML,  # vLLM
    "glm": _GLM,  # SGLang alias of glm45
    "glm45": _GLM,  # vLLM -> Glm47MoeModelToolParser, SGLang Glm4MoeDetector
    "glm47": _GLM,  # vLLM, SGLang Glm47MoeDetector
    "kimi_k2": _KIMI,  # vLLM, SGLang KimiK2Detector
}


# ---------------------------------------------------------------------------------------
# Request-level rules
# ---------------------------------------------------------------------------------------

def tool_choice_error(tool_choice, tools) -> str | None:
    """vLLM's checks of tool_choice (chat_completion/protocol.py check_tool_usage, v0.30.0),
    without its refusal of tool_choice auto / required when no tools are sent."""
    if tool_choice is None or tool_choice in ("none", "auto", "required"):
        return None
    if not isinstance(tool_choice, dict):
        return (f"Invalid value for `tool_choice`: {tool_choice}! Only named tools, \"none\", \"auto\" "
                "or \"required\" are supported.")
    usage = 'Correct usage: `{"type": "function", "function": {"name": "my_function"}}`'
    function = tool_choice.get("function")
    if not isinstance(function, dict):
        return f"Invalid value for `function`: `{function}` in `tool_choice`! {usage}"
    if "name" not in function:
        return f"Expected field `name` in `function` in `tool_choice`! {usage}"
    name = function["name"]
    if not isinstance(name, str) or not name:
        return f"Invalid `name` in `function`: `{name}` in `tool_choice`! {usage}"
    if not tools:
        return "When using `tool_choice`, `tools` must be set."
    if not any((t.get("function") or {}).get("name") == name for t in tools if isinstance(t, dict)):
        return "The tool specified in `tool_choice` does not match any of the specified `tools`"
    return None


def chat_finish_reason(finish: str | None, called: bool, tool_choice) -> str | None:
    """vLLM chat_completion/serving.py: "tool_calls" once a call was parsed, except for a named
    tool_choice, which keeps the engine's reason ("stop"), as OpenAI's API does."""
    if finish is not None and called and not isinstance(tool_choice, dict):
        return "tool_calls"
    return finish


# ---------------------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------------------

def _find(text: str, tags) -> tuple[int, str | None]:
    """The earliest tag in text (the longest one on a tie)."""
    best, found = -1, None
    for tag in tags:
        i = text.find(tag)
        if i >= 0 and (best < 0 or i < best or (i == best and len(tag) > len(found))):
            best, found = i, tag
    return best, found


def _safe_len(text: str, tags, final: bool) -> int:
    """How much of text can be emitted without risking a split tag."""
    if final:
        return len(text)
    hold = 0
    for tag in tags:
        for k in range(min(len(tag) - 1, len(text)), hold, -1):
            if tag.startswith(text[-k:]):
                hold = k
                break
    return len(text) - hold


@dataclass
class _Slot:
    index: int | None = None  # set once the call is announced
    skip: bool = False
    muted: bool = False  # parallel_tool_calls=False: calls after the first are dropped
    id: str = ""
    name: str = ""
    sent: str = ""  # arguments streamed so far


def _format(value, table: dict, default: str):
    if not value:
        return None
    if value is True:
        value = default
    return table[value]


class StreamParser:
    """Feed detokenized text with push(); returns OpenAI chat `delta` dicts to emit.

    reasoning / tools: a REASONING_PARSERS / TOOL_PARSERS name (True for qwen3 / hermes),
    or None. thinking_open: the prompt ended inside a reasoning block. tool_defs: the
    request's OpenAI `tools`, used to type XML arguments and check GLM names."""

    def __init__(self, reasoning=None, tools=None, thinking_open: bool = False, tool_defs: list | None = None,
                 parallel_tool_calls: bool = True):
        self.rfmt: ReasoningFormat | None = _format(reasoning, REASONING_PARSERS, "qwen3")
        self.tfmt: _ToolFormat | None = _format(tools, TOOL_PARSERS, "hermes")
        self.tool_defs = tool_defs
        self.parallel = parallel_tool_calls
        self.saw_tool = False
        self.reasoning_opened = False
        self._buf = ""
        self._state = "content"
        self._closed = False  # a reasoning block ended, so a further end tag is a duplicate
        self._skip_nl = False  # drop the newlines that follow the end tag
        self._slots: list[_Slot] = []
        self._calls = 0
        if self.rfmt is not None:
            self._state = "start"
            if thinking_open or self.rfmt.always:
                self._enter_think()
        self._ends = ()  # tags that end reasoning without the end tag
        if self.rfmt is not None and self.rfmt.tool_start:
            self._ends = (self.rfmt.tool_start,)
            if self.tfmt is not None and self.tfmt.opener != self.rfmt.tool_start:
                self._ends += (self.tfmt.opener,)

    def _enter_think(self):
        self._state = "think"
        self.reasoning_opened = True

    def push(self, delta: str, final: bool) -> list[dict]:
        self._buf += delta
        out: list[dict] = []
        while True:
            st, r = self._state, self.rfmt
            if st == "start":
                s = self._buf.lstrip()
                if s.startswith(r.start):
                    self._buf = s[len(r.start):]
                    self._enter_think()
                    continue
                if not final and r.start.startswith(s):
                    return _coalesce(out)  # might still become the start tag
                self._state = "content"
                continue
            if st == "think":
                i, tag = _find(self._buf, (r.end, r.start) + self._ends)
                if tag is None:
                    n = _safe_len(self._buf, (r.end, r.start) + self._ends, final)
                    if n:
                        out.append({"reasoning_content": self._buf[:n]})
                        self._buf = self._buf[n:]
                    return _coalesce(out)
                if i:
                    out.append({"reasoning_content": self._buf[:i]})
                if tag == r.start:  # a repeated start tag is absorbed (vLLM (REASONING, THINK_START))
                    self._buf = self._buf[i + len(tag):]
                    continue
                self._closed = True
                self._state = "content"
                if tag == r.end:
                    self._buf = self._buf[i + len(tag):]
                    self._skip_nl = True
                else:  # the tool opener stays for the content state
                    self._buf = self._buf[i:]
                continue
            if st == "content":
                if self._skip_nl:
                    self._buf = self._buf.lstrip("\n")
                    if not self._buf and not final:
                        return _coalesce(out)
                    self._skip_nl = False
                tags = []
                if self.tfmt is not None:
                    tags.append(self.tfmt.opener)
                if r is not None and self._closed:
                    tags.append(r.end)  # vLLM drops a duplicate end tag ((CONTENT, THINK_END))
                if r is not None and r.reopen:
                    tags.append(r.start)
                i, tag = _find(self._buf, tags)
                if tag is None:
                    n = _safe_len(self._buf, tags, final)
                    if n:
                        out.append({"content": self._buf[:n]})
                        self._buf = self._buf[n:]
                    return _coalesce(out)
                before, self._buf = self._buf[:i], self._buf[i + len(tag):]
                if self.tfmt is not None and tag == self.tfmt.opener:
                    if before.strip():
                        out.append({"content": before})
                    self._state, self._slots = "block", []
                    continue
                if before:
                    out.append({"content": before})
                if r is not None and tag == r.start:
                    self._enter_think()
                continue
            # st == "block"
            f = self.tfmt
            j = self._buf.find(f.closer) if f.closer else -1
            if j >= 0 or final:
                body = self._buf[:j] if j >= 0 else self._buf
                self._buf = self._buf[j + len(f.closer):] if j >= 0 else ""
                out += self._block(body, True, f.closer if j >= 0 else "")
                self._state = "content"
                if not self._buf:
                    return _coalesce(out)
                continue
            out += self._block(self._buf[:_safe_len(self._buf, f.tags, False)], False, "")
            return _coalesce(out)

    def _props(self, name: str) -> dict:
        for t in self.tool_defs or ():
            fn = t.get("function") if isinstance(t, dict) else None
            if isinstance(fn, dict) and fn.get("name") == name:
                params = fn.get("parameters")
                return (params.get("properties") or {}) if isinstance(params, dict) else {}
        return {}

    def _offered(self, name: str) -> bool:
        if self.tool_defs is None:
            return True
        return any(isinstance(t, dict) and (t.get("function") or {}).get("name") == name for t in self.tool_defs)

    def _block(self, body: str, done: bool, closer: str) -> list[dict]:
        f, out = self.tfmt, []
        for k, view in enumerate(f.scan(body, done)):
            if k == len(self._slots):
                self._slots.append(_Slot())
            slot = self._slots[k]
            if slot.skip:
                continue
            announce = slot.index is None
            if announce:
                if view.name is None:
                    break  # the name is still being written
                if not view.ok or not view.name or (f.validate_names and not self._offered(view.name)):
                    slot.skip = True
                    continue
                slot.index, self._calls = self._calls, self._calls + 1
                slot.id = view.call_id or "call_" + uuid.uuid4().hex[:24]
                slot.name = view.name
                slot.muted = not self.parallel and slot.index > 0
            args = f.arguments(view, view.closed or done, self._props(slot.name))
            diff = ""
            if args.startswith(slot.sent):  # never retract what was streamed
                diff, slot.sent = args[len(slot.sent):], args
            if slot.muted or not (announce or diff):
                continue
            call = {"index": slot.index, "function": {"arguments": diff}}
            if announce:
                call = {"index": slot.index, "id": slot.id, "type": "function",
                        "function": {"name": slot.name, "arguments": diff}}
            out.append({"tool_calls": [call]})
            self.saw_tool = True
        if done:
            if f.verbatim_on_failure and not any(s.index is not None for s in self._slots):
                out.append({"content": f.opener + body + closer})
            self._slots = []
        return out


def _coalesce(deltas: list[dict]) -> list[dict]:
    """Merge neighbouring deltas of one kind; argument pieces of one call become one."""
    out: list[dict] = []
    for d in deltas:
        (key,) = d
        if out and key in out[-1] and len(out[-1]) == 1:
            if key != "tool_calls":
                out[-1][key] += d[key]
                continue
            calls = out[-1]["tool_calls"]
            for c in d["tool_calls"]:
                if calls and calls[-1]["index"] == c["index"] and "id" not in c:
                    calls[-1]["function"]["arguments"] += c["function"]["arguments"]
                else:
                    calls.append(c)
            continue
        out.append(d)
    return out


def parse_full(text: str, reasoning=None, tools=None, thinking_open: bool = False, tool_defs: list | None = None,
               parallel_tool_calls: bool = True) -> dict:
    """Non-streaming: split a whole completion into content, reasoning_content, tool_calls."""
    p = StreamParser(reasoning, tools, thinking_open, tool_defs, parallel_tool_calls)
    deltas = p.push(text, final=True)
    msg: dict = {"role": "assistant", "content": None}
    if p.reasoning_opened:
        msg["reasoning_content"] = "".join(d.get("reasoning_content", "") for d in deltas).strip("\n")
    content = "".join(d.get("content", "") for d in deltas)
    by_index: dict[int, dict] = {}
    for d in deltas:
        for c in d.get("tool_calls", ()):
            if "id" in c:
                by_index[c["index"]] = {"id": c["id"], "type": "function",
                                        "function": {"name": c["function"]["name"], "arguments": ""}}
            by_index[c["index"]]["function"]["arguments"] += c["function"]["arguments"]
    calls = [by_index[i] for i in sorted(by_index)]
    if calls:
        content = content.strip() if p.tfmt.strip_content else (content if content.strip() else "")
    elif "reasoning_content" in msg:
        content = content.strip("\n")
    msg["content"] = content if content else None
    if calls:
        msg["tool_calls"] = calls
    return msg
