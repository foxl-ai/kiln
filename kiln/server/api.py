"""OpenAI-compatible HTTP server.

The engine is single-threaded and owned by one thread that runs the step loop. HTTP
handlers talk to it only through a queue, and each request streams its tokens back
through an asyncio queue. A client that disconnects aborts its request, so the engine
stops spending device time on output nobody will read.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import math
import os
import queue
import signal
import sys
import threading
import time
import traceback
import uuid
import weakref
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Request as HTTPRequest
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import ValidationError

from ..metrics import Metrics

from ..engine.engine import LLMEngine
from ..engine.grammar import spec_from_openai
from ..engine.request import Request, SamplingParams, Status
from ..engine.scheduler import QueueFull
from .decisions import PROMPT_FORMAT_VERSION, Decider, DecisionRequest, build_answer, error_body
from .parsers import REASONING_PARSERS, TOOL_PARSERS, StreamParser, chat_finish_reason, parse_full, tool_choice_error


@dataclass
class _Stream:
    req: Request
    loop: asyncio.AbstractEventLoop
    out: asyncio.Queue
    sent: int = 0


@dataclass
class _Submit:
    rid: str
    prompt_ids: list[int]
    params: SamplingParams
    priority: int
    loop: asyncio.AbstractEventLoop
    session_id: str | None = None
    out: asyncio.Queue = field(default_factory=asyncio.Queue)
    # Prefill / decode disaggregation (engine/disagg.py): on a prefill engine, (transfer id, decode receiver)
    # to hand the request to; on a decode engine, the transfer id whose handoff this request waits for.
    handoff: tuple[str, str] | None = None
    await_xfer: str | None = None
    created: float = field(default_factory=time.monotonic)


# How long a decode engine keeps a handoff nobody waits for, and a request waits for its handoff.
PD_AWAIT_TIMEOUT_S = float(os.environ.get("KILN_PD_AWAIT_TIMEOUT_S", "600"))

# Engine loops of this process, held weakly: at exit each loop's thread stops after its step and then its engine
# closes (LLMEngine.close is idempotent; engine.py's own exit hook, which runs after this one, holds only
# tensor-parallel engines, so a tp 1 pipeline stage 0 never sent its following stages the close frame).
_LOOPS: weakref.WeakSet = weakref.WeakSet()


@atexit.register
def _stop_loops() -> None:
    for lp in list(_LOOPS):
        lp.stop()
        lp.engine.close()


def serve(app: FastAPI, **kwargs) -> None:
    """uvicorn.run(app, **kwargs), with a SIGTERM that ends the process through SystemExit (status 143) once the server
    has shut down. uvicorn 0.52 restores the previous handler and raises the signal again (Server.capture_signals);
    under SIG_DFL that killed the process there, before atexit: no LLMEngine.close (the tensor-parallel workers died in
    a broadcast, "Connection closed by peer"), no close frame from a pipeline's stage 0, no KILN_DSA_CP_LOCAL_K report,
    and every server's KILN_TIMELINE 0 bytes (kiln-pd4-dec, 2026-10-07)."""
    import uvicorn

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    uvicorn.run(app, **kwargs)


class EngineLoop:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._stopping = threading.Event()
        self.metrics = Metrics()
        self.inbox: queue.Queue = queue.Queue()
        self.streams: dict[str, _Stream] = {}
        # Decode engine of a disaggregated deployment: requests waiting for their handoff, and handoffs that
        # arrived before their request (transfer id -> (_Submit or meta, time)).
        self.awaits: dict[str, _Submit] = {}
        self.arrived: dict[str, tuple[dict, float]] = {}
        self.pd_expired = 0
        # A decode server streams every running request each step (hundreds): batch the hand-over to the event loop and
        # detokenize incrementally (_Detok). Other servers keep the per-stream path.
        self.fast_stream = engine.pd_role == "decode" and os.environ.get("KILN_FAST_STREAM", "1") == "1"
        # Per step() call: (monotonic time, seconds, decode rows, prefill tokens, handoffs copied in): GET /debug/steps.
        import collections

        self.steps: collections.deque = collections.deque(maxlen=int(os.environ.get("KILN_STEP_LOG", "20000")))
        engine.pd_wake = lambda: self.inbox.put(("wake", None))
        self.thread = threading.Thread(target=self._run, name="kiln-engine", daemon=True)
        self.thread.start()
        _LOOPS.add(self)

    def stop(self, wait: float = 120.0) -> None:
        """End the engine thread after its current step (at exit, before the engine closes). Idempotent."""
        self._stopping.set()
        self.inbox.put(("wake", None))
        if self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(wait)
            if self.thread.is_alive():
                print(f"kiln: the engine thread did not stop within {wait:.0f} s; closing the engine anyway",
                      flush=True)

    def submit_await(self, xfer: str, prompt_ids: list[int], params: SamplingParams) -> _Submit:
        """A decode engine's half of a disaggregated request: stream the request handoff `xfer` brings."""
        sub = _Submit(xfer, prompt_ids, params, 0, asyncio.get_running_loop(), await_xfer=xfer)
        self.inbox.put(("await", sub))
        return sub

    def _handoff(self, sub: _Submit, meta: dict) -> None:
        """A complete handoff (or the prefill side's end of the request) for a waiting request."""
        post = lambda *m: sub.loop.call_soon_threadsafe(sub.out.put_nowait, m)  # noqa: E731
        if meta.get("error"):
            self.engine.pd_release(meta["xfer"])
            post("error", meta["error"])
            return
        if meta.get("prompt_ids") is not None and list(meta["prompt_ids"]) != list(sub.prompt_ids):
            self.engine.pd_release(meta["xfer"])
            post("error", "the prefill engine tokenized the prompt differently from this decode engine")
            return
        lps = [tuple(x) for x in meta["logprobs"]] if meta.get("logprobs") is not None else None
        if meta.get("done") is not None:  # it ended on the prefill engine: nothing to decode
            self.engine.pd_release(meta["xfer"])
            first = [meta["token"]] if meta.get("token") is not None else []
            post("tokens", first, meta["done"] if meta["done"] != "abort" else "abort", lps if first else None)
            return
        try:
            req = self.engine.add_prefilled(meta)
        except ValueError as e:
            self.engine.pd_release(meta["xfer"])
            post("error", str(e))
            return
        req.arrival_time = sub.created
        self.streams[req.rid] = _Stream(req, sub.loop, sub.out, sent=1)
        post("tokens", req.output_ids[:1], None, req.logprobs[:1] if req.params.logprobs is not None else None)

    def _pd_poll(self) -> None:
        eng = self.engine
        while True:
            try:
                meta = eng.pd_ready.get_nowait()
            except queue.Empty:
                break
            sub = self.awaits.pop(meta["xfer"], None)
            if sub is None:
                self.arrived[meta["xfer"]] = (meta, time.monotonic())
            else:
                self._handoff(sub, meta)
        if self.awaits or self.arrived:
            now = time.monotonic()
            for x, sub in list(self.awaits.items()):
                if now - sub.created > PD_AWAIT_TIMEOUT_S:
                    del self.awaits[x]
                    self.pd_expired += 1
                    sub.loop.call_soon_threadsafe(sub.out.put_nowait, (
                        "error", f"no handoff for {x} within {PD_AWAIT_TIMEOUT_S:g} s (KILN_PD_AWAIT_TIMEOUT_S)"))
            for x, (meta, t) in list(self.arrived.items()):
                if now - t > PD_AWAIT_TIMEOUT_S:
                    del self.arrived[x]
                    self.pd_expired += 1
                    eng.pd_release(x)
                    print(f"kiln pd: dropped handoff {x}: no request waited for it within {PD_AWAIT_TIMEOUT_S:g} s",
                          flush=True)

    def submit(self, prompt_ids: list[int], params: SamplingParams, priority: int = 0,
               session_id: str | None = None, handoff: tuple[str, str] | None = None) -> _Submit:
        if not prompt_ids or not all(isinstance(t, int) for t in prompt_ids):
            raise HTTPException(status_code=400, detail="prompt must be a non-empty list of token ids")
        if session_id is not None and not isinstance(session_id, str):
            raise HTTPException(status_code=400, detail="session_id must be a string")
        sub = _Submit(uuid.uuid4().hex if handoff is None else handoff[0], prompt_ids, params, priority,
                      asyncio.get_running_loop(), session_id, handoff=handoff)
        self.inbox.put(("add", sub))
        return sub

    def abort(self, rid: str) -> None:
        self.inbox.put(("abort", rid))

    def close_session(self, session_id: str) -> None:
        self.inbox.put(("close_session", session_id))

    async def call(self, fn, *args):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.inbox.put(("call", (fn, args, loop, fut)))
        return await fut

    def _drain(self, block: bool) -> None:
        try:
            item = self.inbox.get(timeout=0.05) if block else self.inbox.get_nowait()
        except queue.Empty:
            return
        while True:
            kind, payload = item
            if kind == "await":
                payload.loop.call_soon_threadsafe(payload.out.put_nowait, ("accepted", None))
                got = self.arrived.pop(payload.await_xfer, None)
                if got is not None:
                    self._handoff(payload, got[0])
                else:
                    self.awaits[payload.await_xfer] = payload
            elif kind == "wake":
                pass
            elif kind == "add":
                try:
                    req = self.engine.add_request(payload.prompt_ids, payload.params, rid=payload.rid,
                                                  priority=payload.priority, session_id=payload.session_id,
                                                  handoff=payload.handoff)
                except QueueFull as e:
                    payload.loop.call_soon_threadsafe(payload.out.put_nowait, ("busy", str(e)))
                except ValueError as e:
                    payload.loop.call_soon_threadsafe(payload.out.put_nowait, ("error", str(e)))
                else:
                    self.streams[payload.rid] = _Stream(req, payload.loop, payload.out)
                    payload.loop.call_soon_threadsafe(payload.out.put_nowait, ("accepted", req))
            elif kind == "close_session":
                self.engine.close_session(payload)
            elif kind == "call":  # run fn on the engine thread, between steps
                fn, args, loop, fut = payload
                try:
                    res = fn(*args)
                except Exception as e:  # noqa: BLE001  reported to the caller
                    loop.call_soon_threadsafe(fut.set_exception, e)
                else:
                    loop.call_soon_threadsafe(fut.set_result, res)
            elif kind == "abort":
                st = self.streams.pop(payload, None)
                if st is not None and st.req.status is not Status.FINISHED:
                    self.engine.abort(st.req)
                if self.awaits.pop(payload, None) is not None:  # waiting for a handoff: drop it when it comes
                    self.arrived[payload] = ({"xfer": payload, "done": "abort"}, 0.0)
            try:
                item = self.inbox.get_nowait()
            except queue.Empty:
                return

    def _run(self) -> None:
        while not self._stopping.is_set():
            self._drain(block=not self.engine.has_work())
            if self.engine.pd_role == "decode":
                self._pd_poll()
            if not self.engine.has_work():
                continue
            try:
                inj = self.engine.pd_counts["injected"]
                self.engine.step()
                st = self.engine.last_step
                self.steps.append((time.monotonic(), st.seconds, st.num_decode, st.num_prefill_tokens,
                                   self.engine.pd_counts["injected"] - inj))
            except Exception as e:  # never let the engine thread die silently
                traceback.print_exc()
                for rid, st in list(self.streams.items()):
                    st.loop.call_soon_threadsafe(st.out.put_nowait, ("error", f"engine step failed: {e}"))
                    if st.req.status is not Status.FINISHED:
                        self.engine.abort(st.req)
                self.streams.clear()
                continue
            batch: dict = {}  # fast_stream: one cross-thread call per event loop and step, not one per stream
            for rid, st in list(self.streams.items()):
                out = st.req.output_ids
                finished = st.req.status is Status.FINISHED
                with_lp = st.req.params.logprobs is not None
                # Tokens forced by jump-forward decoding get their logprobs one step later, from
                # the prefill that computes them: hold them back until then, so tokens and
                # logprobs stay aligned.
                ready = len(out) if finished or not with_lp else min(len(out), len(st.req.logprobs))
                if ready > st.sent or finished:
                    new = out[st.sent:ready]
                    lps = st.req.logprobs[st.sent:ready] if with_lp else None
                    st.sent = ready
                    item = ("tokens", new, st.req.finish_reason if finished else None, lps)
                    if self.fast_stream:
                        batch.setdefault(st.loop, []).append((st.out, item))
                    else:
                        st.loop.call_soon_threadsafe(st.out.put_nowait, item)
                if finished:
                    del self.streams[rid]
                    self.metrics.on_finish(st.req)
            for loop, items in batch.items():
                loop.call_soon_threadsafe(_deliver, items)


def _deliver(items) -> None:
    for q, item in items:
        q.put_nowait(item)


class _Detok:
    """Incremental detokenizer for streaming.

    Never emits half of a multi-byte character, and with stop strings it holds back the
    last `max(len(stop)) - 1` characters until they cannot start a stop string, so a stop
    sequence is never partially streamed. Output is cut before the first stop string, as
    the OpenAI API specifies.
    """

    def __init__(self, tokenizer, stops: tuple[str, ...] = (), incremental: bool = False):
        self.tok = tokenizer
        self.stops = stops
        self.hold = max((len(x) for x in stops), default=1) - 1
        self.ids: list[int] = []
        self.emitted = 0
        self.stopped = False
        # incremental (a decode server of a disaggregated deployment, EngineLoop.fast_stream): the text so far is kept
        # and extended by decoding a window of the newest ids (from the last point where the text was complete, TGI's
        # prefix / read offsets), instead of decoding every id on every push, which is O(n^2) per stream and with
        # 384 streams per step held the GIL for tens of ms per step on the G1 decode box. The final push decodes the
        # whole output once, so the complete text is the full decode.
        self.incremental = incremental
        self._text = ""
        self._po = self._ro = 0

    def _decoded(self, final: bool) -> str:
        if final or not self.incremental:
            return self.tok.decode(self.ids, skip_special_tokens=True)
        prefix = self.tok.decode(self.ids[self._po : self._ro], skip_special_tokens=True)
        new = self.tok.decode(self.ids[self._po :], skip_special_tokens=True)
        if len(new) > len(prefix) and not new.endswith("\ufffd"):
            self._text += new[len(prefix):]
            self._po, self._ro = self._ro, len(self.ids)
            return self._text
        return self._text + "\ufffd"  # an incomplete character: held back below, as the full decode's would be

    def push(self, ids: list[int], final: bool) -> str:
        if self.stopped:
            return ""
        self.ids.extend(ids)
        text = self._decoded(final)
        cut = min((i for i in (text.find(x) for x in self.stops) if i >= 0), default=-1)
        if cut >= 0:
            self.stopped = True
            end = cut
        elif final:
            end = len(text)
        else:
            if text.endswith("\ufffd"):
                return ""
            end = max(self.emitted, len(text) - self.hold)
        delta = text[self.emitted:end]
        self.emitted = end
        return delta


def _params(body: dict, default_max: int, chat: bool) -> SamplingParams:
    stop_ids = tuple(body.get("stop_token_ids") or ())
    stop = body.get("stop") or ()
    stop = (stop,) if isinstance(stop, str) else tuple(stop)
    if chat:
        logprobs = int(body.get("top_logprobs") or 0) if body.get("logprobs") else None
    else:
        logprobs = int(body["logprobs"]) if body.get("logprobs") is not None else None
    if logprobs is not None and not 0 <= logprobs <= 20:
        raise HTTPException(status_code=400, detail="logprobs / top_logprobs must be between 0 and 20")
    # vLLM's prompt_logprobs (entrypoints/openai/{completion,chat_completion}/protocol.py,
    # v0.30.0): an int, rejected with stream=True when > 0; -1 (full vocabulary) is not
    # supported here because the device returns at most 20 alternatives.
    plp = body.get("prompt_logprobs")
    if plp is not None:
        if not isinstance(plp, int) or not 0 <= plp <= 20:
            raise HTTPException(status_code=400, detail="prompt_logprobs must be an integer between 0 and 20")
        if body.get("stream") and plp > 0:
            raise HTTPException(status_code=400, detail="`prompt_logprobs` are not available when `stream=True`.")
    return SamplingParams(
        max_new_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or default_max),
        temperature=float(body.get("temperature", 1.0)),
        top_p=float(body.get("top_p", 1.0)),
        top_k=int(body.get("top_k", 0) or 0),
        min_p=float(body.get("min_p", 0.0) or 0.0),
        repetition_penalty=float(body.get("repetition_penalty", 1.0) or 1.0),
        frequency_penalty=float(body.get("frequency_penalty", 0.0) or 0.0),
        presence_penalty=float(body.get("presence_penalty", 0.0) or 0.0),
        logprobs=logprobs,
        prompt_logprobs=plp,
        thinking_token_budget=(int(body["thinking_token_budget"])
                               if body.get("thinking_token_budget") is not None else None),
        watermarking=bool(body.get("watermarking", True)),
        prompt_logprobs_start=int(body.get("prompt_logprobs_start", 0) or 0),
        stop=stop,
        stop_token_ids=stop_ids,
        grammar=spec_from_openai(body),
        ignore_eos=bool(body.get("ignore_eos", False)),
        seed=body.get("seed"),
    )


def _label_scores(lps: list[float], apply_softmax: bool, temperature: float) -> list[float]:
    """SGLang's score readout (tokenizer_manager_score_mixin.py, v0.5.21): a softmax over the
    labels' logprobs divided by `temperature` (the vocabulary normaliser cancels), or each
    label's full-vocabulary probability exp(logprob)."""
    if not lps:
        return []
    if apply_softmax:
        top = max(lps)
        w = [math.exp((x - top) / temperature) for x in lps]
        return [x / sum(w) for x in w]
    return [math.exp(x) for x in lps]


def build_app(engine: LLMEngine, model_name: str, reasoning_parser: str | None = None,
              tool_call_parser: str | None = None) -> FastAPI:
    for name, val, ok in (("reasoning parser", reasoning_parser, tuple(REASONING_PARSERS)),
                          ("tool-call parser", tool_call_parser, tuple(TOOL_PARSERS))):
        if val is not None and val not in ok:
            raise ValueError(f"unknown {name} {val!r}; have {ok}")
    app = FastAPI(title="kiln")
    loop_ = EngineLoop(engine)
    tok = engine.tokenizer

    @app.get("/health")
    async def health():
        out = {"status": "ok", "role": engine.pd_role or "both", "max_num_seqs": engine.cfg.max_num_seqs}
        if engine.pd_role is not None:
            out["layout"] = engine.runner.pd_signature()
        if getattr(engine.cfg, "pp_stages", 1) > 1:  # a pipeline stage (engine/pp.py): the PD router's prefill units
            lo, hi = getattr(engine.runner.model, "load_range", None) or (None, None)
            out["pp"] = {"stage": engine.cfg.pp_stage, "stages": engine.cfg.pp_stages, "layers": [lo, hi],
                         "follow": bool(getattr(engine.cfg, "pp_follow", False))}
        if engine.pd_role == "decode":
            out["pd_address"] = getattr(engine, "pd_address", None)
            out["pd_buffer_bytes"] = int(engine.cfg.pd_buffer_gb * 2**30)
            out["bypass_prefill"] = engine.cfg.pd_bypass_prefill
        return out

    @app.get("/debug/steps")
    async def steps(since: float = 0.0):
        """The engine loop's step log (EngineLoop.steps): [monotonic time, seconds, decode rows, prefill tokens,
        handoffs copied in] per step() call after `since` (this host's monotonic clock), and the clock now."""
        return {"now": time.monotonic(), "steps": [list(x) for x in list(loop_.steps) if x[0] > since]}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(loop_.metrics.render(engine, loop_), media_type="text/plain; version=0.0.4")

    @app.post("/generate")
    async def generate(http: HTTPRequest):
        """SGLang's native API: text or input_ids in, text + output_ids + meta_info out."""
        body = await http.json()
        sp = dict(body.get("sampling_params") or {})
        if "max_new_tokens" in sp:
            sp["max_tokens"] = sp.pop("max_new_tokens")
        for k_sg, k_vllm in (("json_schema", "guided_json"), ("regex", "guided_regex"), ("ebnf", "guided_grammar")):
            if k_sg in sp:
                sp[k_vllm] = sp.pop(k_sg)
        start = body.get("logprob_start_len")
        start = -1 if start is None else int(start)
        if body.get("return_logprob"):
            sp["logprobs"] = int(body.get("top_logprobs_num") or 0)
            if start >= 0:  # SGLang: input logprobs from this prompt position; -1 = output only
                sp["prompt_logprobs"], sp["prompt_logprobs_start"] = sp["logprobs"], start
        params = _params(sp, 128, chat=False)
        if "input_ids" in body:
            ids = list(body["input_ids"])
        elif isinstance(body.get("text"), str):
            ids = tok(body["text"])["input_ids"]
        else:
            raise HTTPException(status_code=400, detail="text (string) or input_ids is required")
        if body.get("stream"):
            raise HTTPException(status_code=400, detail="streaming /generate is not supported; use /v1/completions")
        sub = loop_.submit(ids, params, int(body.get("priority", 0)), body.get("session_id"))
        first = await sub.out.get()
        if first[0] in ("busy", "error"):
            raise HTTPException(status_code=429 if first[0] == "busy" else 400, detail=first[1])
        out_ids, finish, lps = [], None, []
        try:
            while finish is None:
                kind, *rest = await sub.out.get()
                if kind == "error":
                    raise HTTPException(status_code=500, detail=rest[0])
                new, finish, lp = rest
                out_ids += new
                if lp:
                    lps += lp
        except asyncio.CancelledError:
            loop_.abort(sub.rid)
            raise
        text = tok.decode(out_ids, skip_special_tokens=True)
        for stop in params.stop:
            if stop in text:
                text = text[: text.index(stop)]
        meta = {"prompt_tokens": len(ids), "completion_tokens": len(out_ids),
                "finish_reason": {"type": finish}, "id": sub.rid}
        if body.get("return_logprob"):
            meta["output_token_logprobs"] = [[lp, t] for t, (lp, _, _) in zip(out_ids, lps)]
            if params.prompt_logprobs is not None:
                plp = first[1].prompt_logprobs
                span = range(min(start, len(ids)), len(ids))
                meta["input_token_logprobs"] = [[plp[q][0] if q in plp else None, ids[q]] for q in span]
                if params.prompt_logprobs:
                    meta["input_top_logprobs"] = [[[v, i] for i, v in zip(plp[q][2], plp[q][3])] if q in plp else None
                                                  for q in span]
        return {"text": text, "output_ids": out_ids, "meta_info": meta}

    async def _run(ids: list[int], params: SamplingParams) -> Request:
        """Submit one request and wait for it to finish; returns the engine's Request."""
        sub = loop_.submit(ids, params)
        first = await sub.out.get()
        if first[0] in ("busy", "error"):
            raise HTTPException(status_code=429 if first[0] == "busy" else 400, detail=first[1])
        while True:
            kind, *rest = await sub.out.get()
            if kind == "error":
                raise HTTPException(status_code=500, detail=rest[0])
            if rest[1] is not None:
                return first[1]

    async def _label_logprobs(seqs: list[list[int]], labels: list[list[int]]) -> list[list[float]]:
        """The scoring core of /v1/score and /v1/decisions: the full-vocabulary next-token
        logprob of every label after its sequence. Each label's logprob is the last prompt
        logprob of seq + [label], so all of them run as one batch and the shared seq is
        computed once and served from the radix cache."""
        jobs = []
        for seq, ls in zip(seqs, labels):
            for lab in ls:
                sp = SamplingParams(max_new_tokens=1, temperature=0.0, prompt_logprobs=0,
                                    prompt_logprobs_start=len(seq))
                jobs.append(_run(seq + [int(lab)], sp))
        done = iter(await asyncio.gather(*jobs))
        return [[next(done).prompt_logprobs[len(seq)][0] for _ in ls] for seq, ls in zip(seqs, labels)]

    @app.post("/v1/score")
    async def score(http: HTTPRequest):
        """SGLang's Score API (entrypoints/openai/protocol.py ScoringRequest / ScoringResponse
        and managers/tokenizer_manager_score_mixin.py, v0.5.21), pointwise: for each item, the
        next-token probabilities of label_token_ids after query + item (item + query with
        item_first), over the full vocabulary. Text queries get special tokens, text items do
        not. apply_softmax normalises over the labels at `temperature`; otherwise each score is
        exp(logprob). See _label_logprobs."""
        body = await http.json()
        query, items, labels = body.get("query"), body.get("items"), body.get("label_token_ids")
        if items is None or labels is None:
            raise HTTPException(status_code=400, detail="items and label_token_ids are required")
        temp = float(body.get("temperature", 1.0))
        if not temp > 0 or temp == float("inf"):
            raise HTTPException(status_code=400, detail="temperature must be a finite number above 0")
        q = tok(query)["input_ids"] if isinstance(query, str) else list(query or [])
        items = [items] if isinstance(items, str) else items
        seqs = []
        for it in items:
            iv = tok(it, add_special_tokens=False)["input_ids"] if isinstance(it, str) else list(it)
            seqs.append(iv + q if body.get("item_first") else q + iv)
        if labels and isinstance(labels[0], list):
            if len(labels) != len(seqs):
                raise HTTPException(status_code=400, detail="label_token_ids must have one list per item")
            per_item = labels
        else:
            per_item = [labels] * len(seqs)
        if any(not seq for seq in seqs):
            raise HTTPException(status_code=400, detail="query + item is empty")
        raw = await _label_logprobs(seqs, per_item)
        scores = [_label_scores(lps, bool(body.get("apply_softmax")), temp) for lps in raw]
        out = {"scores": scores, "model": body.get("model", model_name), "object": "scoring",
               "usage": {"prompt_tokens": sum(len(x) for x in seqs), "completion_tokens": 0,
                         "total_tokens": sum(len(x) for x in seqs)}}
        if body.get("return_token_logprobs"):
            out["token_logprobs"] = raw
        return out

    decider = Decider(tok, engine.cfg.max_model_len, reasoning_parser)

    @app.post("/v1/decisions")
    async def decisions(http: HTTPRequest):
        """SGLang's Decisions API (entrypoints/openai/serving_decisions.py, v0.5.21; see
        kiln/server/decisions.py): typed choice, score and yes or no questions about an input,
        each rendered as one user message with thinking off, answered with the probability of
        every one-token label at the answer position through the scoring core of /v1/score.
        Errors use SGLang's ErrorResponse shape."""
        def fail(message: str, err_type: str, code: int = 400) -> JSONResponse:
            return JSONResponse(error_body(message, err_type, code), status_code=code)

        try:
            body = await http.json()
        except ValueError as e:
            return fail(f"the request body is not JSON: {e}", "Bad Request")
        try:
            request = DecisionRequest.model_validate(body)
        except ValidationError as e:
            return fail(str(e), "Bad Request")
        message = decider.validate(request)
        if message is not None:
            return fail(message, "BadRequestError")
        encoded = []
        try:
            for item in decider.questions(request):
                encoded.append(item)
                # Each question renders and tokenizes the whole input on the event loop.
                await asyncio.sleep(0)
        except ValueError as e:
            return fail(str(e), "BadRequest")
        prompts = [p for _, _, p, _ in encoded]
        labels = [ls for _, _, _, ls in encoded]
        try:
            raw = await _label_logprobs(prompts, labels)
        except HTTPException as e:
            return fail(str(e.detail), str(e.status_code), e.status_code)
        answers = {}
        for (question, view, prompt_ids, label_ids), lps in zip(encoded, raw):
            try:
                answers[question.id] = build_answer(
                    question, view, _label_scores(lps, True, request.temperature), lps,
                    *((prompt_ids, label_ids) if request.return_prompt_token_ids else ()))
            except RuntimeError as e:
                return fail(f"Internal server error: {e}", "InternalServerError", 500)
        n = sum(len(p) for p in prompts)
        return {"object": "decisions", "model": request.model, "prompt_format_version": PROMPT_FORMAT_VERSION,
                "answers": answers,
                "usage": {"prompt_tokens": n, "total_tokens": n, "completion_tokens": 0, "reasoning_tokens": 0}}

    @app.post("/v1/watermark/detect")
    async def watermark_detect(http: HTTPRequest):
        """Score token ids (or text, tokenized without special tokens) against this server's
        watermark key: p_value is the chance of a score this high in unwatermarked text."""
        if engine.watermark is None:
            raise HTTPException(status_code=400, detail="this server has no watermark config")
        body = await http.json()
        ids = body.get("token_ids")
        if ids is None and isinstance(body.get("text"), str):
            ids = tok(body["text"], add_special_tokens=False)["input_ids"]
        if not isinstance(ids, list) or not all(isinstance(t, int) for t in ids):
            raise HTTPException(status_code=400, detail="token_ids (list of ints) or text is required")
        return engine.watermark.detect(ids)

    @app.post("/update_weights_from_disk")
    async def update_weights_from_disk(http: HTTPRequest):
        """SGLang's endpoint (srt/entrypoints/http_server.py, UpdateWeightFromDiskReqInput in
        srt/managers/io_struct.py, v0.5.21): model_path, abort_all_requests, flush_cache,
        weight_version; answers {success, message, num_paused_requests}, 400 on failure."""
        body = await http.json()
        if not isinstance(body.get("model_path"), str):
            raise HTTPException(status_code=400, detail="model_path (string) is required")
        ok, msg = await loop_.call(engine.update_weights_from_disk, body["model_path"],
                                   bool(body.get("abort_all_requests", False)), bool(body.get("flush_cache", True)),
                                   body.get("weight_version"))
        return JSONResponse({"success": ok, "message": msg, "num_paused_requests": 0}, status_code=200 if ok else 400)

    @app.post("/close_session")
    async def close_session(http: HTTPRequest):
        """SGLang: release a session's prefix references; its KV stays cached until evicted."""
        body = await http.json()
        sid = body.get("session_id")
        if not isinstance(sid, str):
            raise HTTPException(status_code=400, detail="session_id (string) is required")
        loop_.close_session(sid)
        return {"success": True}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "kiln"}]}

    async def _prefill_only(prompt_ids: list[int], params: SamplingParams, pd: dict, priority: int):
        """A prefill engine's whole answer: run the request to its first token, hand it to pd["dest"], and say
        how it went (the client's stream comes from the decode engine)."""
        if not isinstance(pd.get("xfer"), str) or not isinstance(pd.get("dest"), str):
            raise HTTPException(status_code=400, detail="kiln_pd needs xfer and dest on a prefill engine")
        sub = loop_.submit(prompt_ids, params, priority, handoff=(pd["xfer"], pd["dest"]))
        first = await sub.out.get()
        if first[0] in ("busy", "error"):
            raise HTTPException(status_code=429 if first[0] == "busy" else 400, detail=first[1])
        finish = None
        try:
            while finish is None:
                kind, *rest = await sub.out.get()
                if kind == "error":
                    raise HTTPException(status_code=500, detail=rest[0])
                finish = rest[1]
        except asyncio.CancelledError:
            loop_.abort(sub.rid)
            raise
        req = first[1]
        return JSONResponse({"kiln_pd": {"xfer": pd["xfer"], "handed_off": finish == "handoff",
                                         "finish_reason": finish, "prompt_tokens": len(prompt_ids),
                                         "cached_tokens": max(req.num_cached_tokens, 0),
                                         "prefill_seconds": (req.finish_time or time.monotonic()) - req.arrival_time}})

    async def _generate(http: HTTPRequest, prompt_ids: list[int], params: SamplingParams, chat: bool,
                        stream: bool, priority: int = 0, parse: dict | None = None, session_id: str | None = None,
                        tool_choice=None, pd: dict | None = None, return_ids: bool = False):
        if engine.pd_role == "prefill":
            if pd is None:
                raise HTTPException(status_code=400, detail="this is a prefill engine: requests come from the "
                                                            "router with kiln_pd {xfer, dest}")
            return await _prefill_only(prompt_ids, params, pd, priority)
        if pd is not None:
            if engine.pd_role != "decode" or not isinstance(pd.get("xfer"), str):
                raise HTTPException(status_code=400, detail="kiln_pd {xfer} goes to a decode engine")
            sub = loop_.submit_await(pd["xfer"], prompt_ids, params)
        else:
            sub = loop_.submit(prompt_ids, params, priority, session_id)
        # Admission errors (bad prompt, full queue) arrive before any token: surface them as
        # HTTP status codes rather than inside a 200 stream.
        first = await sub.out.get()
        if first[0] == "busy":
            raise HTTPException(status_code=429, detail=first[1])
        if first[0] == "error":
            raise HTTPException(status_code=400, detail=first[1])
        created = int(time.time())
        oid = ("chatcmpl-" if chat else "cmpl-") + sub.rid
        obj = "chat.completion" if chat else "text_completion"
        detok = _Detok(tok, params.stop, incremental=loop_.fast_stream)

        def lp_block(lps):
            if lps is None:
                return None
            if chat:
                def entry(t, lp):
                    s = tok.decode([t])
                    return {"token": s, "logprob": lp, "bytes": list(s.encode())}
                return {"content": [dict(entry(t, lp), top_logprobs=[entry(i, v) for i, v in zip(ti, tl)])
                                    for t, (lp, ti, tl) in lps]}
            return {"tokens": [tok.decode([t]) for t, _ in lps],
                    "token_logprobs": [lp for _, (lp, _, _) in lps],
                    "top_logprobs": [{tok.decode([i]): v for i, v in zip(ti, tl)} for _, (_, ti, tl) in lps]}

        def chunk(text: str, finish: str | None, lps=None) -> dict:
            if chat:
                choice = {"index": 0, "delta": {"content": text} if text else {}, "finish_reason": finish}
            else:
                choice = {"index": 0, "text": text, "finish_reason": finish}
            if lps is not None:
                choice["logprobs"] = lp_block(lps)
            return {"id": oid, "object": obj + (".chunk" if chat else ""), "created": created,
                    "model": model_name, "choices": [choice]}

        sparser = StreamParser(**parse) if chat and parse else None

        def chat_chunks(text: str, finish: str | None, pairs):
            if sparser is None:
                return [chunk(text, finish, pairs)]
            deltas = sparser.push(text, finish is not None)
            finish = chat_finish_reason(finish, sparser.saw_tool, tool_choice)
            if not deltas:
                return [chunk("", finish, pairs)] if (finish or pairs) else []
            out = []
            for i, d in enumerate(deltas):
                last = i == len(deltas) - 1
                c = chunk("", finish if last else None, pairs if last else None)
                c["choices"][0]["delta"] = d
                out.append(c)
            return out

        async def events():
            completed = False
            try:
                while True:
                    kind, *rest = await sub.out.get()
                    if kind == "error":
                        completed = True
                        yield f"data: {json.dumps({'error': {'message': rest[0]}})}\n\n"
                        return
                    ids, finish, lps = rest
                    text = detok.push(ids, finish is not None)
                    pairs = list(zip(ids, lps)) if lps is not None else None
                    if text or finish or pairs:
                        for c in (chat_chunks(text, finish, pairs) if chat else [chunk(text, finish, pairs)]):
                            if return_ids and ids:  # vLLM's return_token_ids, per streamed chunk too
                                c["choices"][0]["token_ids"] = [int(t) for t in ids]
                                ids = []
                            yield f"data: {json.dumps(c)}\n\n"
                    if finish is not None:
                        completed = True
                        yield "data: [DONE]\n\n"
                        return
            finally:
                # The generator is closed early when the client goes away.
                if not completed:
                    loop_.abort(sub.rid)

        if stream:
            return StreamingResponse(events(), media_type="text/event-stream")

        text, finish, n_out, all_lps, out_ids = "", None, 0, [], []
        try:
            while finish is None:
                kind, *rest = await sub.out.get()
                if kind == "error":
                    raise HTTPException(status_code=400, detail=rest[0])
                ids, finish, lps = rest
                n_out += len(ids)
                out_ids += ids
                if lps is not None:
                    all_lps.extend(zip(ids, lps))
                text += detok.push(ids, finish is not None)
        except asyncio.CancelledError:
            loop_.abort(sub.rid)
            raise
        usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": n_out,
                 "total_tokens": len(prompt_ids) + n_out}
        if chat:
            if parse:
                msg = parse_full(text, **parse)
                finish = chat_finish_reason(finish, bool(msg.get("tool_calls")), tool_choice)
            else:
                msg = {"role": "assistant", "content": text}
            choice = {"index": 0, "message": msg, "finish_reason": finish}
        else:
            choice = {"index": 0, "text": text, "finish_reason": finish}
        if params.logprobs is not None:
            choice["logprobs"] = lp_block(all_lps)
        if return_ids:  # vLLM's return_token_ids; kiln_logprobs: (chosen logprob, top ids, top logprobs) per token
            choice["token_ids"] = [int(t) for t in out_ids]
            if params.logprobs is not None:
                choice["kiln_logprobs"] = [[lp, [int(i) for i in ti], list(tl)] for _, (lp, ti, tl) in all_lps]
        resp = {"id": oid, "object": obj, "created": created, "model": model_name,
                "choices": [choice], "usage": usage}
        if params.prompt_logprobs is not None:
            # vLLM: per prompt position None or {token_id: {logprob, rank, decoded_token}} with
            # the prompt token and its top-n alternatives; on the choice for completions, on
            # the response for chat (CompletionResponseChoice / ChatCompletionResponse).
            plp = first[1].prompt_logprobs

            def position(q):
                if q not in plp:
                    return None
                lp, rank, ti, tl = plp[q]
                d = {str(i): {"logprob": v, "rank": k + 1, "decoded_token": tok.decode([i])}
                     for k, (i, v) in enumerate(zip(ti, tl))}
                d[str(prompt_ids[q])] = {"logprob": lp, "rank": rank, "decoded_token": tok.decode([prompt_ids[q]])}
                return d

            (resp if chat else choice)["prompt_logprobs"] = [position(q) for q in range(len(prompt_ids))]
        return JSONResponse(resp)

    @app.post("/v1/completions")
    async def completions(http: HTTPRequest):
        body = await http.json()
        prompt = body.get("prompt", "")
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            ids = prompt
        elif isinstance(prompt, str):
            ids = tok(prompt)["input_ids"]
        else:
            raise HTTPException(status_code=400, detail="prompt must be a string or a list of token ids")
        return await _generate(http, ids, _params(body, 16, chat=False), chat=False,
                               stream=bool(body.get("stream")), priority=int(body.get("priority", 0)),
                               session_id=body.get("session_id"), pd=body.get("kiln_pd"),
                               return_ids=bool(body.get("return_token_ids")))

    @app.post("/v1/chat/completions")
    async def chat(http: HTTPRequest):
        body = await http.json()
        messages = body.get("messages")
        if not messages:
            raise HTTPException(status_code=400, detail="messages is required")
        kwargs = dict(body.get("chat_template_kwargs") or {})
        tool_choice = body.get("tool_choice")
        error = tool_choice_error(tool_choice, body.get("tools"))
        if error:
            raise HTTPException(status_code=400, detail=error)
        tools = body.get("tools") if tool_choice != "none" else None
        if tools:
            kwargs["tools"] = tools
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kwargs)
        if hasattr(ids, "keys"):  # transformers 5 returns a BatchEncoding, not a dict
            ids = ids["input_ids"]
        ids = list(ids)
        parse = None
        if reasoning_parser or (tool_call_parser and tools):
            # A template that opens the reasoning block in the prompt leaves the output inside
            # it; one that closes it (thinking off) or stops before it lets the output decide.
            fmt = REASONING_PARSERS.get(reasoning_parser)
            tail = tok.decode(ids[-4:]).rstrip()
            # parallel_tool_calls=False keeps only the first call (vLLM
            # maybe_filter_parallel_tool_calls, entrypoints/serve/utils/tool_calls_utils.py).
            parse = dict(reasoning=reasoning_parser, tools=tool_call_parser if tools else None,
                         thinking_open=fmt is not None and tail.endswith(fmt.start), tool_defs=tools,
                         parallel_tool_calls=body.get("parallel_tool_calls") is not False)
        return await _generate(http, ids, _params(body, 512, chat=True), chat=True,
                               stream=bool(body.get("stream")), priority=int(body.get("priority", 0)),
                               parse=parse, session_id=body.get("session_id"), tool_choice=tool_choice,
                               pd=body.get("kiln_pd"))

    # vLLM's /tokenize and /detokenize (v0.24.0 vllm/entrypoints/serve/tokenize/protocol.py: TokenizeCompletionRequest
    # {prompt, add_special_tokens=True, return_token_strs=False}, TokenizeChatRequest {messages,
    # add_generation_prompt=True, continue_final_message=False, add_special_tokens=False, chat_template_kwargs, tools},
    # TokenizeResponse {count, max_model_len, tokens, token_strs}, DetokenizeRequest {tokens}, DetokenizeResponse
    # {prompt}), which vllm-neuron serves through vLLM's API server.
    @app.post("/tokenize")
    async def tokenize(http: HTTPRequest):
        body = await http.json()
        if "messages" in body:
            if body.get("continue_final_message") and body.get("add_generation_prompt", True):
                raise HTTPException(status_code=400, detail="Cannot set both `continue_final_message` and "
                                                            "`add_generation_prompt` to True.")
            kwargs = dict(body.get("chat_template_kwargs") or {})
            if body.get("tools"):
                kwargs["tools"] = body["tools"]
            text = tok.apply_chat_template(body["messages"], add_generation_prompt=body.get("add_generation_prompt", True),
                                           continue_final_message=bool(body.get("continue_final_message")),
                                           tokenize=False, **kwargs)
            ids = tok(text, add_special_tokens=bool(body.get("add_special_tokens", False)))["input_ids"]
        elif isinstance(body.get("prompt"), str):
            ids = tok(body["prompt"], add_special_tokens=bool(body.get("add_special_tokens", True)))["input_ids"]
        else:
            raise HTTPException(status_code=400, detail="prompt (a string) or messages is required")
        out = {"count": len(ids), "max_model_len": engine.cfg.max_model_len, "tokens": list(ids)}
        if body.get("return_token_strs"):
            out["token_strs"] = tok.convert_ids_to_tokens(list(ids))
        return out

    @app.post("/detokenize")
    async def detokenize(http: HTTPRequest):
        body = await http.json()
        ids = body.get("tokens")
        if not isinstance(ids, list) or not all(isinstance(t, int) and t >= 0 for t in ids):
            raise HTTPException(status_code=400, detail="tokens must be a list of non-negative integers")
        return {"prompt": tok.decode(ids)}

    return app
