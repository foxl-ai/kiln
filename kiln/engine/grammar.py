"""Structured outputs through xgrammar, the grammar engine SGLang and vLLM both default to.

The grammar runs on the host: a matcher per request fills a packed token bitmask
(int32 [B, ceil(V / 32)], bit set = allowed) before each step, and the graph unpacks it and
masks the logits it samples from. Accepting the sampled token advances the matcher.

Jump-forward decoding (`GrammarBackend.jump_forward`): when the grammar allows exactly one
continuation string, its tokens are appended at once and computed as a prefill chunk.
"""

from __future__ import annotations

import bisect
import json
import re
from dataclasses import dataclass

import numpy as np
import torch

# llguidance docs/fast_forward.md ("Safely converting FF-strings to FF-tokens",
# https://github.com/guidance-ai/llguidance/blob/main/docs/fast_forward.md): look this many
# tokens back from the end of the forced string for a token that spans its end.
BACKOFF_TOKENS = 4


@dataclass(frozen=True)
class GrammarSpec:
    kind: str  # "json" (any JSON object) | "json_schema" | "regex" | "ebnf" | "choice"
    value: str = ""

    @staticmethod
    def choice(options: list[str]) -> "GrammarSpec":
        return GrammarSpec("regex", "(" + "|".join(re.escape(o) for o in options) + ")")


class GrammarBackend:
    def __init__(self, tokenizer, vocab_size: int):
        import xgrammar as xgr

        self.xgr = xgr
        self.vocab_size = vocab_size
        info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
        self.compiler = xgr.GrammarCompiler(info)
        self._cache: dict[GrammarSpec, object] = {}
        # Each token's bytes as the matcher sees them (xgrammar TokenizerInfo.decoded_vocab)
        # and the ids xgrammar treats as special (never accepted, so never forced).
        self.token_bytes: list[bytes] = info.decoded_vocab
        self._special = frozenset(info.special_token_ids) | frozenset(info.stop_token_ids)
        self._sorted: tuple[list[bytes], np.ndarray] | None = None  # built on the first jump

    def compile(self, spec: GrammarSpec):
        hit = self._cache.get(spec)
        if hit is not None:
            return hit
        c = self.compiler
        if spec.kind == "json":
            g = c.compile_builtin_json_grammar()
        elif spec.kind == "json_schema":
            g = c.compile_json_schema(spec.value)
        elif spec.kind == "regex":
            g = c.compile_regex(spec.value)
        elif spec.kind == "ebnf":
            g = c.compile_grammar(spec.value)
        else:
            raise ValueError(f"unknown grammar kind {spec.kind!r}")
        self._cache[spec] = g
        return g

    def matcher(self, spec: GrammarSpec):
        return self.xgr.GrammarMatcher(self.compile(spec))

    def bitmask(self, matchers: list) -> torch.Tensor:
        """Rows for `matchers` (None = unconstrained, every bit set)."""
        mask = self.xgr.allocate_token_bitmask(len(matchers), self.vocab_size)
        mask.fill_(-1)
        for i, m in enumerate(matchers):
            if m is not None:
                m.fill_next_token_bitmask(mask, i)
        return mask

    # -- jump-forward decoding ----------------------------------------------------

    def jump_forward(self, matcher, last: int, encode, replace_ok: bool = True) -> tuple[bool, list[int]] | None:
        """Tokens to force after `matcher` accepted `last` (the newest sampled token), as
        (replace_last, tokens), or None. With replace_last the tokens take `last`'s place
        (they start with its text, re-tokenized). The matcher is not changed.

        xgrammar 0.2.8 (python/xgrammar/matcher.py): GrammarMatcher.find_jump_forward_string
        is "the longest string that certainly conforms with the current grammar" and leaves
        the matcher as it was; rollback(n) has no depth limit (max_rollback_tokens is
        deprecated, "always unlimited").

        SGLang v0.4.3.post2 (the last release whose scheduler calls it; PR #4032 removed it
        in v0.4.4, and v0.5.21's constrained/xgrammar_backend.py keeps try_jump_forward /
        jump_and_retokenize with no caller) takes xgrammar's find_jump_forward_string, then
        Req.jump_forward_and_retokenize (managers/schedule_batch.py) re-encodes prompt text +
        output text + jump string, keeps the prompt tokens, and XGrammarGrammar.
        jump_and_retokenize rolls the matcher back to the first changed output token and
        accepts the new ones. Here the start boundary re-encodes only the newest token with
        the jump string: every older token has already been streamed (the API sends token
        ids) and may sit in a shared radix page. If that merge changes the newest token and
        the caller cannot take a replacement, there is no jump: forcing the string's own
        tokens after it would be the non-canonical split the re-encoding exists to avoid.

        The end boundary follows llguidance (docs/fast_forward.md): the forced string's last
        tokens are dropped from the first byte where some token allowed by the grammar
        starts and runs past the string's end (`"` is dropped when `":` may follow), so the
        model picks the split it was trained on. SGLang's xgrammar helper forces the whole
        string; its llguidance helper gets this back-off from llguidance's compute_ff_tokens.
        """
        try:
            s = matcher.find_jump_forward_string()
        except UnicodeDecodeError:
            # Measured, xgrammar 0.2.8: inside a multi-byte character the binding cannot
            # convert the partial string. SGLang skips a jump whose pending text ends in an
            # incomplete character (Req.get_next_inc_detokenization); so does this.
            return None
        if not s:
            return None
        vocab = self.token_bytes
        sb = s.encode()
        head = vocab[last]
        try:
            head_text = head.decode("utf-8")
        except UnicodeDecodeError:  # the newest token ends a multi-byte character: no merge
            head, head_text = b"", ""
        ids = list(encode(head_text + s))
        if b"".join(vocab[i] for i in ids) != head + sb:
            return None  # the tokenizer does not round-trip the fragment (e.g. a prepended space)
        replace = bool(head)
        if replace and ids and ids[0] == last:
            ids, head, replace = ids[1:], b"", False
        if replace and not replace_ok:
            return None
        base = matcher.fork()
        if replace:
            base.rollback(1)
        kept = self._back_off(base, ids, len(head))
        if sum(len(vocab[i]) for i in kept) <= len(head):
            return None
        probe = base.fork()
        if not all(t not in self._special and probe.accept_token(t) for t in kept):
            return None
        return replace, kept

    def _back_off(self, base, ids: list[int], start: int) -> list[int]:
        """The longest prefix of `ids` that ends at or before the first byte (>= start) from
        which a grammar-allowed token runs past the end of their bytes. `base` is the matcher
        state before ids."""
        vocab = self.token_bytes
        data = b"".join(vocab[i] for i in ids)
        lo = max(start, len(data) - sum(len(vocab[i]) for i in ids[-BACKOFF_TOKENS:]))
        cut = len(data)
        for idx in range(lo, len(data)):
            cands = self._extensions(data[idx:])
            if not len(cands):
                continue
            m = base.fork()
            if idx and not m.accept_string(data[:idx]):
                return []
            mask = self.xgr.allocate_token_bitmask(1, self.vocab_size)
            m.fill_next_token_bitmask(mask, 0)
            words = mask[0].numpy().view(np.uint32)
            if ((words[cands >> 5] >> (cands & 31).astype(np.uint32)) & 1).any():
                cut = idx
                break
        kept, n = [], 0
        for t in ids:
            n += len(vocab[t])
            if n > cut:
                break
            kept.append(t)
        return kept

    def _extensions(self, prefix: bytes) -> np.ndarray:
        """Ids of non-special tokens whose bytes start with `prefix` and are longer: one
        contiguous run of the byte-sorted vocabulary."""
        if self._sorted is None:
            pairs = sorted((b, i) for i, b in enumerate(self.token_bytes) if b and i not in self._special)
            self._sorted = ([b for b, _ in pairs], np.array([i for _, i in pairs], np.int64))
        keys, ids = self._sorted
        lo = bisect.bisect_right(keys, prefix)  # past the tokens equal to prefix
        stem = prefix.rstrip(b"\xff")  # the smallest byte string above every extension
        hi = bisect.bisect_left(keys, stem[:-1] + bytes([stem[-1] + 1])) if stem else len(keys)
        return ids[lo:hi]


def spec_from_openai(body: dict) -> GrammarSpec | None:
    """OpenAI `response_format`, vLLM `structured_outputs` and the older `guided_*` fields."""
    rf = body.get("response_format") or {}
    if rf.get("type") == "json_object":
        return GrammarSpec("json")
    if rf.get("type") == "json_schema":
        js = rf.get("json_schema") or {}
        schema = js.get("schema", js)
        return GrammarSpec("json_schema", schema if isinstance(schema, str) else json.dumps(schema))
    so = body.get("structured_outputs") or {}
    guided = {
        "json": so.get("json", body.get("guided_json")),
        "regex": so.get("regex", body.get("guided_regex")),
        "choice": so.get("choice", body.get("guided_choice")),
        "grammar": so.get("grammar", body.get("guided_grammar")),
    }
    if guided["json"] is not None:
        j = guided["json"]
        return GrammarSpec("json_schema", j if isinstance(j, str) else json.dumps(j))
    if guided["regex"] is not None:
        return GrammarSpec("regex", guided["regex"])
    if guided["choice"] is not None:
        return GrammarSpec.choice(list(guided["choice"]))
    if guided["grammar"] is not None:
        return GrammarSpec("ebnf", guided["grammar"])
    return None
