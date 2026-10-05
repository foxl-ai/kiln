"""Shared checks for the reasoning / tool-call parser tests (kiln/server/parsers.py)."""

import json
import random

from kiln.server.parsers import StreamParser, parse_full


def stream(text: str, cuts, **kw) -> list[dict]:
    """Feed text split at the given positions, then finish."""
    p = StreamParser(**kw)
    deltas, last = [], 0
    for c in cuts:
        deltas += p.push(text[last:c], final=False)
        last = c
    deltas += p.push(text[last:], final=True)
    return deltas


def collect(deltas: list[dict]):
    """(reasoning, content, calls) from streamed deltas, checking the OpenAI streaming shape:
    a call is announced once with its index, id, type and name, and every later piece carries
    only its index and an argument fragment."""
    reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
    content = "".join(d.get("content", "") for d in deltas)
    calls: dict[int, dict] = {}
    for d in deltas:
        assert len(d) == 1, d
        for c in d.get("tool_calls", ()):
            i = c["index"]
            if "id" in c:
                assert i not in calls, f"call {i} announced twice"
                assert c["type"] == "function" and c["function"]["name"]
                calls[i] = {"id": c["id"], "name": c["function"]["name"], "arguments": ""}
            else:
                assert i in calls and set(c) == {"index", "function"} and set(c["function"]) == {"arguments"}, c
            calls[i]["arguments"] += c["function"]["arguments"]
    assert sorted(calls) == list(range(len(calls))), "indices are 0..n-1"
    return reasoning, content, [calls[i] for i in sorted(calls)]


def chunkings(text: str, seed: int = 0, n_random: int = 100):
    """Every two-chunk split (so each tag is cut at every offset), one character at a time,
    and random chunkings of 1 to 7 characters."""
    for i in range(1, len(text)):
        yield [i]
    yield list(range(1, len(text)))
    rng = random.Random(seed)
    for _ in range(n_random):
        cuts, pos = [], 0
        while True:
            pos += rng.randint(1, 7)
            if pos >= len(text):
                break
            cuts.append(pos)
        yield cuts


def check_streaming(text: str, forbidden=(), **kw) -> dict:
    """Streaming at every chunking gives the non-streaming result: the same reasoning and
    content (up to the whitespace the full parse trims), the same calls with byte-identical
    arguments, and no `forbidden` tag text in reasoning or content. Returns the full parse."""
    full = parse_full(text, **kw)
    full_calls = full.get("tool_calls", [])
    for cuts in chunkings(text):
        reasoning, content, calls = collect(stream(text, cuts, **kw))
        assert reasoning.strip("\n") == full.get("reasoning_content", ""), (cuts, reasoning)
        assert content.strip() == (full["content"] or "").strip(), (cuts, content)
        assert [(c["name"], c["arguments"]) for c in calls] == \
            [(c["function"]["name"], c["function"]["arguments"]) for c in full_calls], (cuts, calls)
        for tag in forbidden:
            assert tag not in reasoning and tag not in content, (cuts, tag)
    return full


def args(msg: dict) -> list:
    return [json.loads(c["function"]["arguments"]) for c in msg.get("tool_calls", [])]


def names(msg: dict) -> list:
    return [c["function"]["name"] for c in msg.get("tool_calls", [])]
