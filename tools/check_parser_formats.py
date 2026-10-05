"""Check the reasoning and tool-call parsers against the models' own chat templates, fetched
from Hugging Face at the revisions kiln/server/parsers.py cites (network, CPU, no weights).

For each template: the assistant turn it renders for a reasoning block, some text and two
tool calls parses back into the same reasoning, text and calls, whole and one character at a
time; and its generation prompt with thinking unset, on and off leaves the output inside a
reasoning block exactly when the chat route's `thinking_open` says so. With --tokenizer the
MiMo-V2.6 turn is also streamed token by token through the server's incremental detokenizer
(skip_special_tokens=True, as served), which must keep every tag.

    python tools/check_parser_formats.py [--tokenizer]
"""

from __future__ import annotations

import argparse
import json
import tempfile
import urllib.request

HF = "https://huggingface.co/{repo}/resolve/{sha}/{file}"
# (repo, revision, file holding the template, reasoning parser, tool parser, the generation
# prompt's assistant header, the turn's end marker)
TEMPLATES = [
    ("XiaomiMiMo/MiMo-V2.6-Flash-RL", "5711b268169967567844e1e560e8a3966da959b1", "chat_template.jinja",
     "mimo", "mimo", "<|im_start|>assistant\n", "<|im_end|>"),
    ("XiaomiMiMo/MiMo-V2-Flash", "1afd314a2406c282e0956375c34a676501c78649", "tokenizer_config.json",
     "mimo", "mimo", "<|im_start|>assistant\n", "<|im_end|>"),
    ("zai-org/GLM-4.5", "cbb2c7cfb52fa128a9660cb1a7a78e017899e115", "chat_template.jinja",
     "glm45", "glm45", "<|assistant|>", "<|observation|>"),
    ("zai-org/GLM-4.6", "be72194883d968d7923a07e2f61681ea9a2826d1", "chat_template.jinja",
     "glm45", "glm45", "<|assistant|>", "<|observation|>"),
    ("zai-org/GLM-4.7", "602d01efcdd332c5238ca4bcede555defbe83eb7", "chat_template.jinja",
     "glm47", "glm47", "<|assistant|>", "<|observation|>"),
    ("zai-org/GLM-5", "c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2", "chat_template.jinja",
     "glm47", "glm47", "<|assistant|>", "<|observation|>"),
    ("moonshotai/Kimi-K2-Instruct", "fd1984e2b7a3350dbf7305fe73a4ede25c14de50", "chat_template.jinja",
     None, "kimi_k2", "<|im_assistant|>assistant<|im_middle|>", "<|im_end|>"),
    ("moonshotai/Kimi-K2-Thinking", "a51ccc050d73dab088bf7b0e2dd9b30ae85a4e55", "chat_template.jinja",
     "kimi_k2", "kimi_k2", "<|im_assistant|>assistant<|im_middle|>", "<|im_end|>"),
    ("moonshotai/Kimi-K2.5", "4d01dfe0332d63057c186e0b262165819efb6611", "chat_template.jinja",
     "kimi_k2", "kimi_k2", "<|im_assistant|>assistant<|im_middle|>", "<|im_end|>"),
]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
    "city": {"type": "string"}, "days": {"type": "integer"}, "opts": {"type": "object"}}}}}]
CALLS = [("get_weather", {"city": "Beijing", "days": 3, "opts": {"unit": "c", "hours": [9, 12]}}),
         ("get_weather", {"city": "a < b, \"quoted\"\nsecond line"})]
MESSAGES = [{"role": "user", "content": "Weather?"},
            {"role": "assistant", "reasoning_content": "Two cities.", "content": "Checking both.",
             "tool_calls": [{"id": f"functions.{n}:{i}", "type": "function", "function": {"name": n, "arguments": a}}
                            for i, (n, a) in enumerate(CALLS)]}]


def fetch(repo: str, sha: str, file: str) -> str:
    with urllib.request.urlopen(HF.format(repo=repo, sha=sha, file=file), timeout=60) as r:
        text = r.read().decode()
    return json.loads(text)["chat_template"] if file.endswith(".json") else text


def check_turn(turn: str, reasoning, tools, thinking_open: bool, kimi: bool) -> None:
    from kiln.server.parsers import StreamParser, parse_full

    kw = dict(reasoning=reasoning, tools=tools, thinking_open=thinking_open, tool_defs=TOOLS)
    for msg in (parse_full(turn, **kw), _streamed(StreamParser(**kw), turn)):
        got = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in msg.get("tool_calls", [])]
        assert got == CALLS, (got, turn)
        assert msg["content"] == "Checking both.", msg
        assert msg.get("reasoning_content") == ("Two cities." if reasoning and "Two cities." in turn else None), msg
        if kimi:
            assert [c["id"] for c in msg["tool_calls"]] == ["functions.get_weather:0", "functions.get_weather:1"]


def _streamed(p, text: str) -> dict:
    deltas = []
    for i, ch in enumerate(text):
        deltas += p.push(ch, final=i == len(text) - 1)
    return _message(deltas, p)


def _message(deltas: list[dict], p) -> dict:
    calls: dict[int, dict] = {}
    for d in deltas:
        for c in d.get("tool_calls", ()):
            calls.setdefault(c["index"], {"id": c.get("id"), "function": {"name": c["function"].get("name"),
                                                                          "arguments": ""}})
            calls[c["index"]]["function"]["arguments"] += c["function"]["arguments"]
    msg = {"content": "".join(d.get("content", "") for d in deltas).strip() or None,
           "tool_calls": [calls[i] for i in sorted(calls)]}
    if p.reasoning_opened:
        msg["reasoning_content"] = "".join(d.get("reasoning_content", "") for d in deltas).strip("\n")
    return msg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", action="store_true", help="also stream MiMo-V2.6 tokens through _Detok")
    args = ap.parse_args()
    from transformers.utils.chat_template_utils import render_jinja_template

    from kiln.server.parsers import REASONING_PARSERS

    for repo, sha, file, rp, tp, header, end in TEMPLATES:
        template = fetch(repo, sha, file)

        def render(messages, generation, **kw):
            out = render_jinja_template([messages], tools=TOOLS, chat_template=template,
                                        add_generation_prompt=generation, **kw)
            return (out[0] if isinstance(out, tuple) else out)[0]

        text = render(MESSAGES, False)
        turn = text[text.rfind(header) + len(header):]
        turn = turn[:turn.find(end)] if end in turn else turn
        turn = turn.rstrip()
        check_turn(turn, rp, tp, False, tp == "kimi_k2")
        prompts = []
        for thinking in (None, True, False):
            kw = {} if thinking is None else {"enable_thinking": thinking, "thinking": thinking}
            prompt = render(MESSAGES[:1], True, **kw)
            fmt = REASONING_PARSERS.get(rp)
            opened = fmt is not None and prompt.rstrip().endswith(fmt.start)  # the chat route's thinking_open
            if opened:  # the model continues after the prompt's own start tag
                start = turn.find(fmt.start)
                check_turn(turn[start + len(fmt.start):], rp, tp, True, tp == "kimi_k2")
            prompts.append(f"{thinking}: ...{prompt[-28:]!r} thinking_open={opened}")
        print(f"ok {repo}@{sha[:7]}\n   turn {turn[:96]!r}...\n   " + "\n   ".join(prompts))

    if args.tokenizer:
        from transformers import AutoTokenizer

        from kiln.server.api import _Detok
        from kiln.server.parsers import StreamParser

        repo, sha = TEMPLATES[0][:2]
        with tempfile.TemporaryDirectory() as d:
            for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
                urllib.request.urlretrieve(HF.format(repo=repo, sha=sha, file=f), f"{d}/{f}")
            tok = AutoTokenizer.from_pretrained(d)
        text = render_jinja_template([MESSAGES], tools=TOOLS, chat_template=fetch(repo, sha, "chat_template.jinja"))
        text = (text[0] if isinstance(text, tuple) else text)[0]
        turn = text[text.rfind(TEMPLATES[0][5]) + len(TEMPLATES[0][5]):].removesuffix("<|im_end|>")
        ids = tok.encode(turn, add_special_tokens=False)
        for tag in ("<think>", "</think>", "<tool_call>", "</tool_call>"):
            assert len(tok.encode(tag, add_special_tokens=False)) == 1, tag
        detok, p, deltas = _Detok(tok), StreamParser("mimo", "mimo", False, TOOLS), []
        for i, t in enumerate(ids):
            last = i == len(ids) - 1
            deltas += p.push(detok.push([t], last), last)
        msg = _message(deltas, p)
        assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in msg["tool_calls"]] == CALLS
        assert msg["reasoning_content"] == "Two cities." and msg["content"] == "Checking both.", msg
        print(f"ok {repo}@{sha[:7]} tokenizer: {len(ids)} tokens streamed one at a time through _Detok")


if __name__ == "__main__":
    main()
