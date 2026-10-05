"""SGLang's `/v1/decisions`: typed questions about an input, answered from a chat model.

Each question (choice, score or yes_no) becomes one user message rendered with the model's
chat template and thinking turned off. Its answers get one-token labels (A to Z, 0 to 9, yes
and no), each label is checked to be one distinct token at the answer position, and the
labels' full-vocabulary next-token logprobs come from the scoring core of `/v1/score`. No text
is generated.

Adapted from sgl-project/sglang v0.5.21 (commit e00930c5489053f26d86b179cee0d087f846acbb).
Portions Copyright 2026 SGLang Team, licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0); see THIRD_PARTY_NOTICES.md. Sources:
- python/sglang/srt/entrypoints/openai/protocol.py: DecisionRequest, the question types,
  DecisionAnswer, DecisionResponse, check_option_names.
- python/sglang/srt/entrypoints/openai/serving_decisions.py: the prompt wording, the label
  checks, the reasoning-block refusals and the answers.
- python/sglang/srt/parser/template_detection.py and parser/reasoning_parser.py: the subset
  of the reasoning-toggle and reasoning-parser detection for the templates Kiln serves.
- docs/docs/supported-models/decision_models.mdx: the request and response reference and the
  400 cases.

Not ported, because Kiln has no such mode: `--enable-mis`, `--dllm-algorithm`, built-in
conversation templates and built-in chat encoders (Kiln's chat route always renders the
tokenizer's Jinja template), `--default-chat-template-kwargs`, non-generation models.
"""

from __future__ import annotations

import json
import math
import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

import jinja2
import jinja2.ext
import jinja2.nodes
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

# serving_decisions.py PROMPT_FORMAT_VERSION: the version of the server-owned prompt wording
# and answer labels. Any change to a rendering this route can produce needs a new version.
PROMPT_FORMAT_VERSION = 1
# protocol.py DEFAULT_MODEL_NAME: echoed as the response `model` when the request has none.
DEFAULT_MODEL_NAME = "default"
# serving_decisions.py _REPLY_SENTINEL: answer text of a finished reply, rendered only to see
# what the chat template puts before an answer.
REPLY_SENTINEL = "DECISION_ANSWER"
# serving_decisions.py _PARSER_TOGGLE_MODES: parser defaults that name the chat template
# kwarg toggling reasoning.
_PARSER_TOGGLE_MODES = ("thinking", "enable_thinking", "explicit_thinking", "explicit_enable_thinking")


# ---------------------------------------------------------------------------------------
# Request and response (protocol.py)
# ---------------------------------------------------------------------------------------

def is_blank(value: Any) -> bool:
    """protocol.py is_blank_decision_text: an empty or whitespace string, or an empty object
    or array."""
    return not (value.strip() if isinstance(value, str) else value)


def _nonblank(value: Any) -> Any:
    if is_blank(value):
        raise ValueError("must not be blank")
    return value


def check_option_names(names) -> None:
    """protocol.py check_option_names: refuse names that would make the option lines ambiguous."""
    seen = set()
    for name in names:
        key = name.strip().casefold()
        if not key:
            raise ValueError("option names must be nonempty")
        # Each option is rendered as one prompt line.
        if any(unicodedata.category(c) in ("Cc", "Zl", "Zp") for c in name):
            raise ValueError(f"option name {name!r} must not contain control or line break characters")
        if key in seen:
            raise ValueError(f"option name {name!r} repeats another option")
        seen.add(key)


# Objects and arrays are rendered into the prompt as compact JSON.
DecisionText = Union[str, Dict[str, Any], List[Any]]
RequiredDecisionText = Annotated[DecisionText, AfterValidator(_nonblank)]
QuestionId = Annotated[str, AfterValidator(_nonblank)]


class DecisionOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: Optional[DecisionText] = None


class DecisionChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: QuestionId
    type: Literal["choice"]
    question: RequiredDecisionText
    # Options are labeled A to Z in order, so at most 26.
    options: List[DecisionOption] = Field(min_length=2, max_length=26)

    @model_validator(mode="after")
    def _option_names_distinct(self):
        try:
            check_option_names(option.name for option in self.options)
        except ValueError as e:
            raise ValueError(f"question {self.id!r}: {e}") from None
        return self


class DecisionScoreQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: QuestionId
    type: Literal["score"]
    question: RequiredDecisionText
    # Levels are labeled 0 to 9 in order, so at most 10.
    levels: List[RequiredDecisionText] = Field(min_length=2, max_length=10)


class DecisionYesNoQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: QuestionId
    type: Literal["yes_no"]
    question: RequiredDecisionText
    yes: Optional[DecisionText] = None
    no: Optional[DecisionText] = None


DecisionQuestion = Annotated[
    Union[DecisionChoiceQuestion, DecisionScoreQuestion, DecisionYesNoQuestion],
    Field(discriminator="type"),
]


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: RequiredDecisionText
    questions: List[DecisionQuestion] = Field(min_length=1)
    # Scales option probabilities only, not label_mass.
    temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    # Applied with the template reasoning toggle off.
    chat_template_kwargs: Dict[str, Any] = Field(default_factory=dict)
    # Pins the server-owned prompt wording. A different served version is refused.
    prompt_format_version: Optional[int] = None
    return_prompt_token_ids: bool = False
    model: str = DEFAULT_MODEL_NAME

    @field_validator("questions")
    @classmethod
    def _question_ids_distinct(cls, questions):
        seen = set()
        for question in questions:
            if question.id in seen:
                raise ValueError(f"question id {question.id!r} repeats another question")
            seen.add(question.id)
        return questions


def error_body(message: str, err_type: str, code: int = 400) -> dict:
    """protocol.py ErrorResponse, as serving_base.py create_error_response sends it."""
    return {"object": "error", "message": message, "type": err_type, "param": None, "code": code}


# ---------------------------------------------------------------------------------------
# Prompt wording (serving_decisions.py, PROMPT_FORMAT_VERSION 1)
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class QuestionView:
    kind: str  # choice, score or yes_no
    question: Any
    names: list[str]  # option names, level indices, or yes and no, in candidate order
    details: list[Any]  # option descriptions, levels, or the yes and no descriptions


def render_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def question_view(question) -> QuestionView:
    if isinstance(question, DecisionChoiceQuestion):
        return QuestionView("choice", question.question, [o.name for o in question.options],
                            [o.description for o in question.options])
    if isinstance(question, DecisionScoreQuestion):
        return QuestionView("score", question.question, [str(i) for i in range(len(question.levels))],
                            list(question.levels))
    return QuestionView("yes_no", question.question, ["yes", "no"], [question.yes, question.no])


def default_labels(view: QuestionView) -> list[str]:
    """Single-token labels in candidate order: A to Z, level indices, or yes and no."""
    if view.kind == "choice":
        return list(string.ascii_uppercase[: len(view.names)])
    return list(view.names)


def render_question(text: str, view: QuestionView, labels: list[str]) -> str:
    """The user message of PROMPT_FORMAT_VERSION 1: the input, a blank line, the question line,
    one line per option, level or described yes or no answer, and the closing instruction."""
    question_text = "" if is_blank(view.question) else render_text(view.question)
    if view.kind == "choice":
        lines = [f"Question: {question_text}"] if question_text else []
        for label, name, description in zip(labels, view.names, view.details):
            detail = render_text(description)
            lines.append(f"{label}: {name} - {detail}" if detail else f"{label}: {name}")
        lines.append("Answer with the letter of one option only.")
    elif view.kind == "score":
        lines = [f"Question: {question_text}"] if question_text else []
        lines += [f"{label}: {render_text(level)}" for label, level in zip(labels, view.details)]
        lines.append("Answer with the number of one level only.")
    else:
        lines = [f"Is the following true? {question_text}" if question_text else "Is the following true?"]
        for label, description in zip(labels, view.details):
            detail = render_text(description)
            if detail:
                lines.append(f"{label}: {detail}")
        lines.append("Answer with yes or no only.")
    return "\n".join([text, "", *lines])


# ---------------------------------------------------------------------------------------
# Label checks (serving_decisions.py label_context / label_token_id / _encode_labels)
# ---------------------------------------------------------------------------------------

def label_context(tok, prompt: str, prompt_ids: list[int], added_tokens: dict[int, str]):
    """Text and ids after which labels are checked. Added tokens are split off before
    tokenization, so the text after the last one tokenizes on its own and the check does not
    grow with the input; otherwise the whole prompt is checked."""
    last = next((i for i in reversed(range(len(prompt_ids))) if prompt_ids[i] in added_tokens), None)
    if last is not None:
        token = added_tokens[prompt_ids[last]]
        start = prompt.rfind(token)
        suffix = prompt[start + len(token):]
        if start >= 0 and tok.encode(suffix, add_special_tokens=False) == prompt_ids[last + 1:]:
            return suffix, prompt_ids[last + 1:]
    return prompt, prompt_ids


def label_token_id(tok, text: str, text_ids: list[int], label: str) -> int | None:
    """The token a label adds after the text, or None when it is not exactly one."""
    ids = tok.encode(text + label, add_special_tokens=False)
    if len(ids) != len(text_ids) + 1 or ids[:-1] != text_ids:
        return None
    return ids[-1]


def encode_labels(tok, prompt: str, prompt_ids: list[int], labels: list[str],
                  added_tokens: dict[int, str]) -> list[int]:
    """Check that each label adds exactly one distinct token after the prompt."""
    text, text_ids = label_context(tok, prompt, prompt_ids, added_tokens)
    label_ids: list[int] = []
    for label in labels:
        token_id = label_token_id(tok, text, text_ids, label)
        if token_id is None or token_id in label_ids:
            raise ValueError(f"the answer label {label!r} is not one distinct token after the "
                             "chat prompt for this tokenizer, so this model is not supported")
        label_ids.append(token_id)
    return label_ids


# ---------------------------------------------------------------------------------------
# Reasoning detection (template_detection.py, reasoning_parser.py): the Kiln subset
# ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ReasoningToggle:
    """template_detection.py ReasoningToggleConfig."""
    toggle_param: str | None = None
    default_enabled: bool | None = None
    special_case: str | None = None

    @property
    def always_on(self) -> bool:
        return self.special_case == "always"


class _GenerationTag(jinja2.ext.Extension):
    """Parse-only `{% generation %}` blocks (transformers assistant masking), so templates that
    use them still parse (template_detection.py _GenerationTagExtension)."""
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return jinja2.nodes.Scope(body).set_lineno(lineno)


def _has_toggle_default(template: str, param: str, default: bool) -> bool:
    """template_detection.py _has_toggle_default_assignment: the template applies Jinja's
    `default` (or `d`) filter to `param` with working-toggle semantics and this default, e.g.
    `{%- set enable_thinking = enable_thinking | default(false) -%}`."""
    try:
        tree = jinja2.Environment(extensions=[jinja2.ext.loopcontrols, _GenerationTag]).parse(template)
    except Exception:
        return False
    for node in tree.find_all(jinja2.nodes.Filter):
        if node.name not in ("default", "d") or node.dyn_args or node.dyn_kwargs:
            continue
        if not isinstance(node.node, jinja2.nodes.Name) or node.node.name != param:
            continue
        args = list(node.args)
        kwargs = {kw.key: kw.value for kw in node.kwargs}
        if not args or not isinstance(args[0], jinja2.nodes.Const) or not isinstance(args[0].value, bool):
            continue
        boolean_arg = args[1] if len(args) > 1 else kwargs.get("boolean")
        if boolean_arg is not None and not isinstance(boolean_arg, jinja2.nodes.Const):
            continue
        if boolean_arg is not None and bool(boolean_arg.value) and args[0].value:
            continue  # default(true, true) replaces an explicit false: not a working toggle
        if args[0].value is default:
            return True
    return False


def _toggle_rules(param: str):
    """The `enable_thinking` / `thinking` rules of REASONING_MODE_RULES, default-off first."""
    p = re.escape(param)
    off = lambda t: (re.search(r"{%\s*if\s+not\s+" + p + r"\s+is\s+defined\s*%}.*?{%\s*set\s+" + p
                               + r"\s*=\s*(?:false|False)\s*%}", t, re.DOTALL) is not None
                     or _has_toggle_default(t, param, False))
    on_patterns = (
        r"{%\s*if\s+not\s+" + p + r"\s+is\s+defined\s*%}.*?{%\s*set\s+" + p + r"\s*=\s*(?:true|True)\s*%}",
        r"set\s+" + p + r"\s*=\s*" + p + r"\s+if\s+" + p + r"\s+is\s+defined\s+else\s+(?:true|True)",
        p + r"\s+is\s+defined\s+and\s+(?:" + p + r"\s+is\s+false|not\s+" + p + r")",
        p + r"\s+is\s+not\s+defined\s+or\s+" + p,
        r"namespace\([^)]*" + p + r"\s*=\s*true",
    )
    on = lambda t: (any(re.search(x, t, re.DOTALL if i == 0 else 0) for i, x in enumerate(on_patterns))
                    or _has_toggle_default(t, param, True))
    return off, on


_ET_OFF, _ET_ON = _toggle_rules("enable_thinking")
_T_OFF, _T_ON = _toggle_rules("thinking")

# REASONING_MODE_RULES in order, without the mistral and hunyuan reasoning_effort special
# cases (they name no toggle and Kiln serves neither family).
_REASONING_MODE_RULES = (
    ("k2_v3_reasoning_effort", ReasoningToggle(special_case="always"),
     lambda t: "<ifm|think>" in t and "<ifm|think_fast>" in t and "reasoning_effort" in t),
    ("gpt_oss_channel_markers", ReasoningToggle(special_case="always"), lambda t: "<|channel|>" in t),
    ("force_reasoning_pattern", ReasoningToggle(special_case="always"),
     lambda t: (re.search(r"<\|im_start\|>assistant\\n<think>\\n", t) is not None
                and "enable_thinking" not in t and "thinking" not in t)),
    ("glm53_always_think", ReasoningToggle(special_case="always"),
     lambda t: ("[gMASK]<sop>" in t and "Reasoning Effort:" in t and "enable_thinking" not in t
                and "<tool_call>" in t and "<arg_key>" in t and "<arg_value>" in t)),
    ("explicit_enable_thinking_default_false", ReasoningToggle("enable_thinking", False), _ET_OFF),
    ("nemotron_3_super_low_effort", ReasoningToggle("enable_thinking", True),
     lambda t: "low_effort" in t and "truncate_history_thinking" in t),
    ("enable_thinking_default_true", ReasoningToggle("enable_thinking", True), _ET_ON),
    ("explicit_thinking_default_false", ReasoningToggle("thinking", False), _T_OFF),
    ("thinking_default_true", ReasoningToggle("thinking", True), _T_ON),
)


def detect_reasoning_toggle(template: str | None) -> ReasoningToggle | None:
    """template_detection.py detect_reasoning_pattern (Kiln subset)."""
    if not template:
        return None
    for _, value, predicate in _REASONING_MODE_RULES:
        if predicate(template):
            return value
    return None


# Reasoning parsers as SGLang's detectors define them (reasoning_parser.py): think tags and
# reasoning_default. Only "always" means answers start inside a reasoning block.
REASONING_PARSERS = {
    "qwen3": ("<think>", "</think>", "enable_thinking"),  # Qwen3Detector
    "deepseek_r1": ("<think>", "</think>", "always"),  # DeepSeekR1Detector (base default)
    "mimo": ("<think>", "</think>", "explicit_enable_thinking"),  # _MimoDetector
    "glm45": ("<think>", "</think>", "enable_thinking"),  # Glm45Detector
    "glm47": ("<think>", "</think>", "enable_thinking"),  # vLLM's name; SGLang serves GLM-4.7 with glm45
    "kimi_k2": ("<think>", "</think>", "thinking"),  # KimiK2Detector
}


def suggested_reasoning_parser(template: str | None, config: ReasoningToggle | None) -> str | None:
    """template_detection.py REASONING_PARSER_RULES, the rules for parsers Kiln mirrors, in
    order: mimo, qwen3, deepseek-r1 (forced reasoning, then any think tag)."""
    if not template:
        return None
    if config == ReasoningToggle("enable_thinking", False):
        return "mimo"
    if config == ReasoningToggle("enable_thinking", True):
        return "qwen3"
    if (config is not None and config.always_on) or "<think>" in template or "</think>" in template:
        return "deepseek_r1"
    return None


# ---------------------------------------------------------------------------------------
# The route's logic, independent of the engine
# ---------------------------------------------------------------------------------------

def _round_trip_is_lossy(tok) -> bool:
    """serving_chat.py _probe_prompt_text_round_trip: does rendering the chat template to text
    and encoding it lose anything against the ids the chat route encodes?"""
    probe = [{"role": "user", "content": "x"}]
    try:
        rendered = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
        kw = {"add_special_tokens": False} if len(tok.encode("")) > 0 else {}
        via_text = tok.encode(rendered, **kw)
        via_ids = tok.apply_chat_template(probe, tokenize=True, add_generation_prompt=True, return_dict=False)
        if hasattr(via_ids, "keys"):
            via_ids = via_ids["input_ids"]
        return list(via_text) != list(via_ids)
    except Exception:
        return False  # a template that needs kwargs this probe does not supply tells nothing


class Decider:
    """serving_decisions.py OpenAIServingDecisions over a Kiln tokenizer: request checks, the
    per-question prompt and label ids, and the answers. Scoring is the caller's."""

    route = "/v1/decisions"

    def __init__(self, tok, max_model_len: int, reasoning_parser: str | None = None):
        self.tok = tok
        self.max_model_len = max_model_len
        self.lossy = _round_trip_is_lossy(tok) if tok is not None else False
        try:
            self.added_tokens = {i: t for t, i in tok.get_added_vocab().items()}
        except Exception:
            self.added_tokens = {}  # other tokenizers check the full prompt
        template = getattr(tok, "chat_template", None)
        template = template if isinstance(template, str) else None
        self.reasoning_config = detect_reasoning_toggle(template)
        # The configured reasoning parser, or the one the chat template suggests, tells where
        # reasoning blocks start and end and whether answers open one.
        parser = reasoning_parser or suggested_reasoning_parser(template, self.reasoning_config)
        config = self.reasoning_config
        self.reasoning_toggle = config.toggle_param if config is not None else None
        self.reasoning_markers = None
        self.answers_open_reasoning = False
        if parser in REASONING_PARSERS:
            start, end, mode = REASONING_PARSERS[parser]
            if config is None and mode in _PARSER_TOGGLE_MODES:
                self.reasoning_toggle = mode.removeprefix("explicit_")
            self.reasoning_markers = (start, end)
            self.answers_open_reasoning = mode == "always"

    def validate(self, request: DecisionRequest) -> str | None:
        """Refusals before any question is rendered, in SGLang's order."""
        route = self.route
        if self.tok is None:
            return f"{route} requires the server tokenizer"
        if self.lossy:
            return (f"{route} places answer labels on the rendered chat text, which this "
                    "tokenizer does not encode back to the same ids")
        if ":" in request.model:  # serving_base.py _parse_model_parameter
            adapter = request.model.split(":", 1)[1].strip() or None
            if adapter is not None:
                return f"model names the LoRA adapter {adapter!r}, which {route} does not support"
        version = request.prompt_format_version
        if version is not None and version != PROMPT_FORMAT_VERSION:
            return (f"prompt_format_version {version} is not served, this server uses version "
                    f"{PROMPT_FORMAT_VERSION}")
        config = self.reasoning_config
        if config is not None and config.always_on:
            return f"{route} does not support chat templates that always reason before answering"
        toggle, kwargs = self.reasoning_toggle, request.chat_template_kwargs
        if toggle in kwargs and kwargs[toggle] is not False:
            return (f"chat_template_kwargs sets {toggle!r} to {kwargs[toggle]!r}, but decisions need "
                    "it false or unset")
        return None

    def chat_template_kwargs(self, request_kwargs: dict) -> dict:
        """Reasoning off, then the request kwargs (Kiln has no server defaults)."""
        kwargs = {self.reasoning_toggle: False} if self.reasoning_toggle is not None else {}
        kwargs.update(request_kwargs)
        return kwargs

    def questions(self, request: DecisionRequest):
        """(question, view, prompt ids, label ids) for each question in request order, lazily
        so the handler can let other requests run between questions. A refusal raises
        ValueError naming the question."""
        text = render_text(request.input)
        kwargs = self.chat_template_kwargs(request.chat_template_kwargs)
        for question in request.questions:
            view = question_view(question)
            try:
                prompt_ids, label_ids = self.encode_question(text, view, default_labels(view), kwargs)
            except ValueError as e:
                raise ValueError(f"question {question.id!r}: {e}") from e
            yield question, view, prompt_ids, label_ids

    def _apply(self, messages: list[dict], kwargs: dict, generation: bool) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation, **kwargs)

    def encode_question(self, text: str, view: QuestionView, labels: list[str], kwargs: dict):
        content = render_question(text, view, labels)
        try:
            prompt = self._apply([{"role": "user", "content": content}], kwargs, True)
        except (jinja2.TemplateError, TypeError) as e:  # serving_chat.py _CHAT_TEMPLATE_CLIENT_ERRORS
            raise ValueError(f"the chat template failed: {e}") from e
        if self.reasoning_markers is not None:
            # Look only after the message, whose last line is fixed text.
            closing = content.rsplit("\n", 1)[-1]
            cut = prompt.rfind(closing)
            generation_prompt = prompt if cut < 0 else prompt[cut + len(closing):]
            start, end = self.reasoning_markers
            opened, closed = generation_prompt.rfind(start), generation_prompt.rfind(end)
            if opened > closed:
                raise ValueError("the chat template leaves a reasoning block open at the answer "
                                 "position, so this model is not supported with these chat_template_kwargs")
            if self.answers_open_reasoning and closed < 0:
                raise ValueError("the reasoning parser for this model expects answers to start with a "
                                 "reasoning block, and the chat template does not close one. Send "
                                 "chat_template_kwargs that turn thinking off, if the template supports it")
            # The template's own finished reply shows whether answers start with a reasoning
            # block that the generation prompt leaves out.
            try:
                reply = self._apply([{"role": "user", "content": closing},
                                     {"role": "assistant", "content": REPLY_SENTINEL}], kwargs, False)
            except (jinja2.TemplateError, TypeError):
                reply = None  # the generation prompt checks above still apply
            begin = reply.rfind(closing) if reply is not None else -1
            answer = reply.find(REPLY_SENTINEL, begin) if begin >= 0 else -1
            if answer >= 0 and reply[begin:answer].count(start) > generation_prompt.count(start):
                raise ValueError("the chat template starts every answer with a reasoning block, so "
                                 "this model is not supported")
        prompt_ids = self.tok.encode(prompt, add_special_tokens=False)
        # Refused here rather than truncated. Each label is scored as the last token of
        # prompt + label, and Kiln's scheduler admits a sequence only below max_model_len
        # (engine/scheduler.py Scheduler.add): SGLang's num_reserved_tokens is 1 here.
        if len(prompt_ids) + 1 >= self.max_model_len:
            raise ValueError(f"the prompt has {len(prompt_ids)} tokens, which does not fit the "
                             f"context length of {self.max_model_len} tokens")
        label_ids = encode_labels(self.tok, prompt, prompt_ids, labels, self.added_tokens)
        return prompt_ids, label_ids


def build_answer(question, view: QuestionView, probabilities: list[float], token_logprobs: list[float],
                 prompt_ids: list[int] | None = None, label_ids: list[int] | None = None) -> dict:
    """serving_decisions.py _build_answer, dumped like DecisionAnswer with exclude_none."""
    mass = math.fsum(math.exp(lp) for lp in token_logprobs)
    if not all(math.isfinite(v) for v in [*probabilities, mass]):
        # A server fault, reported as 500 (as SGLang's /v1/systemone does), never as an answer.
        raise RuntimeError(f"question {question.id!r} scored non-finite values")
    answer: dict = {"type": question.type, "probabilities": dict(zip(view.names, probabilities)),
                    "label_mass": mass}
    if view.kind == "choice":
        answer["choice"] = view.names[probabilities.index(max(probabilities))]
    elif view.kind == "score":
        answer["score"] = math.fsum(i * p for i, p in enumerate(probabilities))
    if prompt_ids is not None:
        answer["prompt_token_ids"] = prompt_ids
        answer["label_token_ids"] = label_ids
    return answer
