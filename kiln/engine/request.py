from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from .grammar import GrammarSpec  # noqa: F401  (annotation)
from .radix_cache import RadixNode


@dataclass(frozen=True)
class SamplingParams:
    max_new_tokens: int = 128
    temperature: float = 0.0  # 0 means greedy
    top_p: float = 1.0
    top_k: int = 0  # 0 means disabled
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logprobs: int | None = None  # None: off; n >= 0: chosen token plus top-n (n <= 20)
    # Prompt logprobs (vLLM `prompt_logprobs`): for every prompt position from
    # prompt_logprobs_start on (SGLang `logprob_start_len`), the logprob and rank of the prompt
    # token plus the top-n alternatives. Position 0 has none.
    prompt_logprobs: int | None = None
    prompt_logprobs_start: int = 0
    stop: tuple[str, ...] = ()  # stop strings, matched on detokenized output
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False
    seed: int | None = None
    grammar: "GrammarSpec | None" = None  # structured output constraint (engine/grammar.py)
    # vLLM's thinking_token_budget: once this many tokens follow the reasoning start, the
    # reasoning end string is forced (engine.LLMEngine._advance_thinking).
    thinking_token_budget: int | None = None
    # vLLM's per-request watermark opt-out (engine/watermark.py); only meaningful when the
    # engine has a watermark config.
    watermarking: bool = True


class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(eq=False)
class Request:
    rid: str
    prompt_ids: list[int]
    params: SamplingParams
    arrival_time: float = field(default_factory=time.monotonic)
    priority: int = 0  # vLLM convention: lower values are served first
    # SGLang's top-level session_id: a finished request's prefix stays referenced by its
    # session (soft eviction protection) until the session is closed.
    session_id: str | None = None
    session_gen: int = -1

    # Prompt followed by generated tokens. After a preemption the whole list is the
    # "prompt" of the recompute, so it is kept as one list.
    token_ids: list[int] = field(init=False)
    status: Status = Status.WAITING
    # Pages holding KV for positions [0, num_computed), in position order.
    pages: list[int] = field(default_factory=list)
    num_computed: int = 0
    # Deepest radix node this request has locked, if any.
    node: RadixNode | None = None
    # Tokens of the prompt served from the prefix cache at first admission.
    num_cached_tokens: int = -1
    num_preemptions: int = 0
    finish_reason: str | None = None
    # Per output token, when params.logprobs is set: (logprob, top ids, top logprobs).
    logprobs: list[tuple[float, list[int], list[float]]] = field(default_factory=list)
    # Per prompt position >= max(1, params.prompt_logprobs_start), when prompt_logprobs is set:
    # position -> (logprob, rank, top ids, top logprobs).
    prompt_logprobs: dict[int, tuple[float, int, list[int], list[float]]] = field(default_factory=dict)
    first_token_time: float | None = None
    finish_time: float | None = None
    matcher: object | None = None  # xgrammar GrammarMatcher when params.grammar is set
    spec_proposed: int = 0
    spec_accepted: int = 0
    # DP attention (engine/dp.py): the attention group whose KV pages and state rows this request
    # uses; fixed when it is queued (a preempted request stays in its group).
    dp_group: int = 0
    # The tokens its group was charged for it when it was placed (prompt minus what the group held).
    dp_charge: int | None = None
    # Steps this request's prefill sat out to share a later call (engine/dp.py hold), over its life.
    prefill_holds: int = 0
    # Thinking budget state: inside the reasoning span, its token count, output tokens
    # already scanned, tokens still to force, and whether the span has ended.
    think_in: bool = False
    think_count: int = 0
    think_seen: int = 0
    think_force: list[int] = field(default_factory=list)
    think_done: bool = False
    watermarked: bool = False
    mtp_draft: list[int] = field(default_factory=list)  # MTP drafts for the next step
    spec_inflight: int = 0  # blind speculative steps launched and not yet committed (engine/spec_async.py)
    spec_ready: bool = False  # its speculative board row is the device's (else the host initializes it)
    # Jump-forward decoding (engine.LLMEngine._jump_forward): output tokens the grammar forced
    # rather than the model sampled, and the end of the newest forced run. Their logprobs come
    # from the prefill rows that compute them (forced_logprob_rows), never from a sample.
    num_forced: int = 0
    forced_end: int = 0
    wm_seen: set = field(default_factory=set)  # contexts already watermarked (deduplication)
    # Recurrent-state checkpoints (engine/scheduler.py): prefill positions still to save at, and
    # saved rows waiting for their pages to enter the radix tree, as (position, row, from decode).
    ckpt_targets: list[int] = field(default_factory=list)
    ckpt_pending: list[tuple[int, int, bool]] = field(default_factory=list)
    ckpt_restore: int | None = None  # the checkpoint row a prefix hit restores, until the copy is launched
    # Prefill / decode disaggregation (engine/disagg.py). Prefill side: (transfer id, decode receiver
    # "host:port") of a request to hand off once its first token is sampled, the request's own params
    # (it runs with max_new_tokens 1 here), and its pages at release (Scheduler._release), which the
    # handoff reads before anything is scheduled again. Decode side: the complete handoff a request was
    # admitted from, and whether its parts are on the device yet (engine._pd_inject_plan).
    handoff: tuple[str, str] | None = None
    handoff_params: SamplingParams | None = None
    handoff_pages: list[int] | None = None
    handoff_meta: dict | None = None  # the meta sent to the decode side (engine._pd_handoff)
    pd_meta: dict | None = None
    pd_injected: bool = False

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise ValueError("empty prompt")
        self.token_ids = list(self.prompt_ids)

    @property
    def host_bound(self) -> bool:
        """Needs every previous token on the host before its next step (no overlap, no
        speculative drafts): grammar, penalties, or a thinking budget still running."""
        return (self.matcher is not None or self.has_penalties or self.watermarked
                or (self.params.thinking_token_budget is not None and not self.think_done))

    @property
    def has_penalties(self) -> bool:
        p = self.params
        return p.repetition_penalty != 1.0 or p.frequency_penalty != 0.0 or p.presence_penalty != 0.0

    @property
    def prompt_logprob_rows(self) -> tuple[int, int] | None:
        """[first, last) positions whose output rows yield the prompt logprobs still missing
        (row q scores prompt token q + 1), or None when nothing is pending."""
        p = self.params
        if p.prompt_logprobs is None:
            return None
        first = max(1, p.prompt_logprobs_start) - 1
        if first >= self.num_prompt - 1 or len(self.prompt_logprobs) >= self.num_prompt - 1 - first:
            return None
        return first, self.num_prompt - 1

    @property
    def forced_logprob_rows(self) -> tuple[int, int] | None:
        """[first, last) positions whose output rows yield the logprobs of forced tokens still
        missing (row q scores token q + 1, as for prompt logprobs), or None. Every output token
        before them has its logprob, so the first is the row before the first one without."""
        if self.params.logprobs is None:
            return None
        nxt = self.num_prompt + len(self.logprobs)
        if nxt >= self.forced_end:
            return None
        return nxt - 1, self.forced_end - 1

    @property
    def num_prompt(self) -> int:
        return len(self.prompt_ids)

    @property
    def output_ids(self) -> list[int]:
        """Generated tokens known on the host. Under overlap scheduling the newest entries
        of token_ids can be placeholders (-1) for steps still in flight; they are left out."""
        out = self.token_ids[self.num_prompt :]
        while out and out[-1] == -1:
            out = out[:-1]
        return out

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def is_decoding(self) -> bool:
        """Every known token except the newest has KV in cache, so the next step is one
        query token. After a preemption the recompute is a prefill over every known token."""
        return self.num_computed == self.num_tokens - 1

    @property
    def remaining_prefill(self) -> int:
        return self.num_tokens - self.num_computed
