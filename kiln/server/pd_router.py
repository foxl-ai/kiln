"""Router of a disaggregated deployment: prefill engines, decode engines, one OpenAI-compatible front.

    python -m kiln.server.pd_router --prefill-urls http://p1:8100,http://p2:8100 --decode-urls http://d1:8100 \\
        --tokenizer zai-org/GLM-5.3-Flash --threshold 4096 --port 8000

Each worker is a Kiln server started with a role (bench/pd_serve.py --pd-role prefill | decode). A request:

1. Threshold routing (SageMaker HyperPod's routingThreshold, default 4096 input tokens there): a prompt of
   fewer than --threshold tokens goes straight to a decode engine, which prefills it itself (it must have
   been started with --pd-bypass-prefill); a longer one is disaggregated. --threshold 0 disaggregates every
   request.
2. Backpressure: a disaggregated request takes one of the chosen decode engine's credits (--decode-credits,
   default its max_num_seqs plus 25%: the running requests plus the handoffs waiting in its receive
   buffer) before anything is sent anywhere, and gives it back when its stream ends. Without a free credit it
   WAITS here, in arrival order, and is never dropped; the wait is measured (kiln:pd_router_queue_seconds).
3. The decode engine gets the request with kiln_pd {xfer}: it registers the transfer and returns the client's
   stream once the handoff arrives. At the same time (not after the decode engine answers) the least-loaded
   prefill engine gets it with kiln_pd {xfer, dest = the decode engine's receiver}: it prefills, samples the first
   token and sends the request's state to dest. The decode engine accepts a request only between two of its steps,
   so waiting for its answer first put up to one decode step in front of every prefill (slo4: 0.267 s steps on the
   decode box, ~0.26 s of a 1.60 s 8K TTFT in a closed loop); a handoff that arrives before its request is kept
   until the request comes (api.EngineLoop.arrived). The client reads the decode engine's stream through the
   router, unchanged (the first token included). A prefill failure ends the stream with an error event; a decode
   engine that refuses the request cancels its prefill call (a handoff that was already sent then waits out
   KILN_PD_AWAIT_TIMEOUT_S on the decode engine, which no request makes the router refuse in practice: both
   engines parse the same body).
   Prompt token ids: when the router tokenizes a /v1/completions string prompt itself (threshold routing with
   --tokenizer), it forwards those ids as the prompt, so neither engine tokenizes it again (~11 ms per 8K-token
   prompt each). Only when its tokenizer is the served model (the decode engine's /v1/models id equals --tokenizer;
   otherwise it says so at start and forwards the string), and never for chat (the engines apply the template).
4. Prefill queue-depth cap (opt-in, --prefill-depth N): at most N requests in flight on each prefill engine; the rest
   wait HERE, in arrival order, and each goes to the first engine with a free slot when it frees (late binding), so a
   burst does not pile onto the engines that happened to be least loaded when it arrived. Latency prefill engines
   (--latency-prefill-urls: DP attention 1, one request per call over every rank) are capped at --latency-depth
   (default 1) and preferred whenever one has a free slot; a request then waits for a latency engine only when
   every throughput engine is capped and full too. Without a cap (the default) nothing waits here for prefill.

5. Prefill units (--prefill-units): a layer pipeline of S stage engines (engine/pp.py, every stage started with
   --pp-follow: stage 0 by bench/pd_serve.py --pd-role prefill, the others by tools/pp_follow.py --pd-role prefill
   --health-port) is ONE prefill engine to the router: a request is posted to its stage 0 only, which carries it to
   the others in its plan frames, and every stage hands its own layers' share to the decode engine the request names
   (engine/disagg.py combines them there). The router checks that the stages answer in order, follow stage 0, tile the
   model's layers and share one handoff layout. A unit is preferred like a latency engine and capped at --unit-depth
   requests in flight (default 1: one long prompt through the pipeline at a time).

Start-up role check: every worker's /health names its role; the router refuses to start without at least one
prefill and one decode engine, or when their handoff layouts differ.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from ..metrics import Histogram, LATENCY_BUCKETS


class Worker:
    def __init__(self, url: str, health: dict, kind: str = "throughput", depth: int = 0):
        self.url = url.rstrip("/")
        self.role = health["role"]
        self.health = health
        self.inflight = 0  # requests of this worker not finished
        self.credits = 0  # decode engines: credits left
        self.kind = kind  # prefill engines: "latency" (preferred, one request per call) or "throughput"
        self.depth = depth  # prefill engines: requests in flight at most (0: no cap)


def check_unit(urls: list[str], healths: list[dict]) -> None:
    """Refuse a prefill unit whose stages cannot form one pipeline (main, --prefill-units)."""
    pps = [h.get("pp") or {} for h in healths]
    if any(h.get("role") != "prefill" for h in healths):
        raise SystemExit(f"prefill unit {urls}: every stage must be a prefill engine, roles {[h.get('role') for h in healths]}")
    if [p.get("stage") for p in pps] != list(range(len(urls))) or any(p.get("stages") != len(urls) for p in pps):
        raise SystemExit(f"prefill unit {urls}: stages answer as {[(p.get('stage'), p.get('stages')) for p in pps]}, "
                         f"need 0 .. {len(urls) - 1} of {len(urls)} in this order")
    if not all(p.get("follow") for p in pps):
        raise SystemExit(f"prefill unit {urls}: every stage must run with --pp-follow (stage 0 alone takes requests)")
    ranges = [p.get("layers") for p in pps]
    if ranges[0][0] not in (0, None) or any(a[1] != b[0] for a, b in zip(ranges, ranges[1:])):
        raise SystemExit(f"prefill unit {urls}: the stages' layers {ranges} do not tile the model")
    if len({h.get("layout") for h in healths}) != 1:
        raise SystemExit(f"prefill unit {urls}: the stages' handoff layouts differ")


def check_workers(prefill: list[dict], decode: list[dict]) -> None:
    """Refuse a deployment that cannot work (the router's start-up role check)."""
    roles = [h.get("role") for h in prefill + decode]
    if any(h.get("role") != "prefill" for h in prefill) or any(h.get("role") != "decode" for h in decode):
        raise SystemExit(f"role check failed: prefill urls report {[h.get('role') for h in prefill]}, decode urls "
                         f"report {[h.get('role') for h in decode]} (each engine logs 'kiln role: ...' at start)")
    if not prefill or not decode:
        raise SystemExit(f"role check failed: need at least one prefill and one decode engine, have roles {roles}")
    layouts = {h.get("layout") for h in prefill + decode}
    if len(layouts) != 1:
        raise SystemExit(f"handoff layouts differ across engines: {sorted(map(str, layouts))}")
    for h in decode:
        if not h.get("pd_address"):
            raise SystemExit("a decode engine has no handoff receiver (start it with --pd-listen)")


class PDRouter:
    def __init__(self, prefill: list[Worker], decode: list[Worker], threshold: int, tokenizer=None,
                 credits: int | None = None):
        self.prefill, self.decode = prefill, decode
        self.threshold, self.tok = threshold, tokenizer
        for d in decode:
            d.credits = credits or int(d.health["max_num_seqs"] * 1.25)
        self.cv = asyncio.Condition()
        self.waiting = 0
        self.pwait: collections.deque = collections.deque()  # futures of requests waiting for a prefill slot, FIFO
        self.prefill_queue_s = Histogram(LATENCY_BUCKETS + (81.92, 163.84))
        self.counts = {"disaggregated": 0, "bypass": 0, "prefill_errors": 0, "handed_off": 0, "done_at_prefill": 0,
                       "forwarded_ids": 0}
        self.forward_ids = False  # forward the ids count_tokens made (main: only when the tokenizer is the served model)
        self.queue_s = Histogram()
        self.e2e_ttft = Histogram(LATENCY_BUCKETS + (81.92, 163.84))
        self.prefill_s = Histogram(LATENCY_BUCKETS + (81.92, 163.84))
        self.prefill_call_s = Histogram(LATENCY_BUCKETS + (81.92, 163.84))

    def count_tokens(self, path: str, body: dict) -> int | None:
        if path.endswith("chat/completions"):
            if self.tok is None:
                return None
            ids = self.tok.apply_chat_template(body.get("messages") or [], add_generation_prompt=True, tokenize=True)
            return len(ids["input_ids"] if hasattr(ids, "keys") else ids)
        p = body.get("prompt", body.get("text", ""))
        if isinstance(p, list) and (not p or isinstance(p[0], int)):
            return len(p)
        if isinstance(body.get("input_ids"), list):
            return len(body["input_ids"])
        if self.tok is None or not isinstance(p, str):
            return None
        ids = self.tok(p)["input_ids"]
        if self.forward_ids and isinstance(body.get("prompt"), str):
            body["prompt"] = list(ids)  # what both engines would make of the string (api.completions: tok(prompt))
            self.counts["forwarded_ids"] += 1
        return len(ids)

    def bypass(self, path: str, body: dict) -> bool:
        if self.threshold <= 0:
            return False
        if any(k in body for k in ("response_format", "guided_json", "guided_regex", "guided_grammar",
                                   "guided_choice", "thinking_token_budget")):
            return True  # host state only the engine that runs the whole request has (engine/disagg.py)
        n = self.count_tokens(path, body)
        if n is None:
            raise HTTPException(status_code=400, detail="threshold routing needs the prompt's token count: start the "
                                                        "router with --tokenizer, or --threshold 0")
        return n < self.threshold

    async def take_decode(self) -> Worker:
        """The decode engine with the most free credits, waiting (FIFO) until one has a credit."""
        t = time.perf_counter()
        async with self.cv:
            self.waiting += 1
            try:
                while True:
                    d = max(self.decode, key=lambda w: (w.credits, -w.inflight))
                    if d.credits > 0:
                        d.credits -= 1
                        d.inflight += 1
                        break
                    await self.cv.wait()
            finally:
                self.waiting -= 1
        self.queue_s.observe(time.perf_counter() - t)
        return d

    async def give_back(self, d: Worker) -> None:
        async with self.cv:
            d.credits += 1
            d.inflight -= 1
            self.cv.notify(1)

    def pick_prefill(self) -> Worker | None:
        """The prefill engine for the next request: a latency engine with a free slot, else the throughput engine with
        the fewest in flight among those under their cap (without caps and latency engines: the least loaded, as
        before the cap existed). None when every engine is at its cap."""
        free = [w for w in self.prefill if not w.depth or w.inflight < w.depth]
        if not free:
            return None
        return min(free, key=lambda w: (w.kind != "latency", w.inflight))

    async def take_prefill(self) -> Worker:
        """A prefill slot: at once when an engine has one and nobody waits, else in arrival order when a slot frees
        (give_prefill hands it straight to the first waiter, so a later arrival cannot overtake)."""
        t = time.perf_counter()
        w = self.pick_prefill() if not self.pwait else None
        if w is None:
            fut = asyncio.get_running_loop().create_future()
            self.pwait.append(fut)
            try:
                w = await fut
            except asyncio.CancelledError:
                if fut.done() and not fut.cancelled():
                    self.give_prefill(fut.result())  # granted, then abandoned: the slot goes to the next waiter
                else:
                    try:
                        self.pwait.remove(fut)
                    except ValueError:
                        pass
                raise
        else:
            w.inflight += 1
        self.prefill_queue_s.observe(time.perf_counter() - t)
        return w

    def give_prefill(self, w: Worker) -> None:
        w.inflight -= 1
        while self.pwait:
            nxt = self.pick_prefill()
            if nxt is None:
                return
            fut = self.pwait.popleft()
            if fut.done():  # cancelled while waiting
                continue
            nxt.inflight += 1
            fut.set_result(nxt)

    def render(self) -> str:
        lines = []
        for k, v in self.counts.items():
            lines += [f"# TYPE kiln:pd_router_{k}_total counter", f"kiln:pd_router_{k}_total {v}"]
        lines += ["# TYPE kiln:pd_router_waiting gauge", f"kiln:pd_router_waiting {self.waiting}"]
        lines += ["# TYPE kiln:pd_router_prefill_waiting gauge", f"kiln:pd_router_prefill_waiting {len(self.pwait)}"]
        for w in self.prefill + self.decode:
            lines.append(f'kiln:pd_router_inflight{{role="{w.role}",url="{w.url}"}} {w.inflight}')
        for w in self.prefill:
            lines.append(f'kiln:pd_router_prefill_depth{{kind="{w.kind}",url="{w.url}"}} {w.depth}')
        for d in self.decode:
            lines.append(f'kiln:pd_router_decode_credits{{url="{d.url}"}} {d.credits}')
        for name, h in (("kiln:pd_router_queue_seconds", self.queue_s),
                        ("kiln:pd_router_prefill_queue_seconds", self.prefill_queue_s),
                        ("kiln:pd_router_e2e_ttft_seconds", self.e2e_ttft),
                        ("kiln:pd_router_prefill_seconds", self.prefill_s),
                        ("kiln:pd_router_prefill_call_seconds", self.prefill_call_s)):
            lines.append(f"# TYPE {name} histogram")
            lines += h.lines(name)
        return "\n".join(lines) + "\n"


def build_pd_app(r: PDRouter, client: httpx.AsyncClient | None = None) -> FastAPI:
    """client: the router's connection to the engines (tests pass one over another transport)."""
    app = FastAPI(title="kiln-pd-router")
    # Idle connections are dropped here (30 s) before the engines would close them (bench/pd_serve.py keeps them 600 s):
    # a request written onto a connection the server is closing fails with ReadError.
    client = client or httpx.AsyncClient(timeout=None, limits=httpx.Limits(
        max_connections=None, max_keepalive_connections=512, keepalive_expiry=30.0))

    @app.get("/health")
    async def health():
        return {"status": "ok", "role": "router", "prefill": [w.url for w in r.prefill],
                "decode": [w.url for w in r.decode], "counts": r.counts, "waiting": r.waiting,
                "prefill_waiting": len(r.pwait), "credits": {d.url: d.credits for d in r.decode},
                "prefill_kinds": {w.url: {"kind": w.kind, "depth": w.depth, "inflight": w.inflight} for w in r.prefill}}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(r.render(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models():
        x = await client.get(r.decode[0].url + "/v1/models")
        return JSONResponse(x.json(), status_code=x.status_code)

    async def bypass(path: str, body: dict):
        d = min(r.decode, key=lambda w: w.inflight)
        r.counts["bypass"] += 1
        d.inflight += 1
        if body.get("stream"):
            resp = await client.send(client.build_request("POST", d.url + path, json=body), stream=True)

            async def relay():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp.aclose()
                    d.inflight -= 1

            return StreamingResponse(relay(), status_code=resp.status_code,
                                     media_type=resp.headers.get("content-type", "text/event-stream"))
        try:
            x = await client.post(d.url + path, json=body)
        finally:
            d.inflight -= 1
        return Response(x.content, status_code=x.status_code, media_type=x.headers.get("content-type"))

    async def run_prefill(path: str, body: dict, xfer: str, d: Worker, t0: float, p: Worker):
        """The prefill call on p (a slot taken with take_prefill; the caller gives it back)."""
        t = time.perf_counter()
        try:
            x = await client.post(p.url + path, json={**body, "stream": False,
                                                       "kiln_pd": {"xfer": xfer, "dest": d.health["pd_address"]}})
            if x.status_code != 200:
                r.counts["prefill_errors"] += 1
                return f"prefill engine {p.url} answered {x.status_code}: {x.text[:500]}"
            info = x.json().get("kiln_pd", {})
            r.counts["handed_off" if info.get("handed_off") else "done_at_prefill"] += 1
            r.prefill_call_s.observe(time.perf_counter() - t)
            r.prefill_s.observe(time.perf_counter() - t0)
            return None
        except httpx.HTTPError as e:
            r.counts["prefill_errors"] += 1
            return f"prefill engine {p.url} failed: {e!r}"

    async def disaggregate(path: str, body: dict):
        t0 = time.perf_counter()
        d = await r.take_decode()
        r.counts["disaggregated"] += 1
        xfer = uuid.uuid4().hex
        stream = bool(body.get("stream"))
        dbody = {**body, "kiln_pd": {"xfer": xfer}}
        released = False

        async def release():
            nonlocal released
            if not released:
                released = True
                await r.give_back(d)

        try:
            p = await r.take_prefill()
        except BaseException:
            await release()
            raise
        try:
            if stream:
                # The prefill call starts now, not once the decode engine has answered (module docstring, 3.).
                pre = asyncio.create_task(run_prefill(path, body, xfer, d, t0, p))
                # A done callback, not a finally in run_prefill: it runs even for a task cancelled before it started.
                pre.add_done_callback(lambda _t, p=p: r.give_prefill(p))
                try:
                    resp = await client.send(client.build_request("POST", d.url + path, json=dbody), stream=True)
                except BaseException:
                    pre.cancel()
                    await release()
                    raise
                if resp.status_code != 200:  # refused before it registered: its prefill call is not needed
                    content = await resp.aread()
                    await resp.aclose()
                    pre.cancel()
                    await release()
                    return Response(content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))

                async def relay():
                    first = True
                    try:
                        it = resp.aiter_raw().__aiter__()
                        while True:
                            nxt = asyncio.ensure_future(it.__anext__())
                            while True:
                                done, _ = await asyncio.wait({nxt, pre} if not pre.done() else {nxt},
                                                             return_when=asyncio.FIRST_COMPLETED)
                                if nxt in done:
                                    break
                                err = pre.result()
                                if err is not None:  # the handoff will never come: end the stream loudly
                                    nxt.cancel()
                                    yield f"data: {json.dumps({'error': {'message': err}})}\n\n".encode()
                                    return
                            try:
                                chunk = nxt.result()
                            except StopAsyncIteration:
                                return
                            if first:
                                first = False
                                r.e2e_ttft.observe(time.perf_counter() - t0)
                            yield chunk
                    finally:
                        await resp.aclose()  # closing the decode stream aborts the request there
                        if not pre.done():
                            pre.cancel()
                        await release()

                return StreamingResponse(relay(), status_code=200,
                                         media_type=resp.headers.get("content-type", "text/event-stream"))
            dec = asyncio.create_task(client.post(d.url + path, json=dbody))
            try:
                err = await run_prefill(path, body, xfer, d, t0, p)
            finally:
                r.give_prefill(p)
            if err is not None:
                dec.cancel()
                raise HTTPException(status_code=502, detail=err)
            x = await dec
            r.e2e_ttft.observe(time.perf_counter() - t0)
            return Response(x.content, status_code=x.status_code, media_type=x.headers.get("content-type"))
        finally:
            if not stream:
                await release()

    def route(path: str):
        async def handler(http: Request):
            body = await http.json()
            if r.bypass(path, body):
                return await bypass(path, body)
            return await disaggregate(path, body)

        return handler

    for path in ("/v1/completions", "/v1/chat/completions"):
        app.add_api_route(path, route(path), methods=["POST"])
    return app


def main() -> None:
    ap = argparse.ArgumentParser(prog="kiln.server.pd_router")
    ap.add_argument("--prefill-urls", default="", help="comma list of (throughput) prefill engines")
    ap.add_argument("--decode-urls", required=True, help="comma list of decode engines")
    ap.add_argument("--threshold", type=int, default=4096,
                    help="prompts of fewer tokens go straight to a decode engine (0: disaggregate all)")
    ap.add_argument("--tokenizer", default=None, help="HF id or path, to count prompt tokens for --threshold")
    ap.add_argument("--no-forward-ids", action="store_true",
                    help="send a completions string prompt on as text even when the router tokenized it")
    ap.add_argument("--decode-credits", type=int, default=None,
                    help="disaggregated requests in flight per decode engine (default its max_num_seqs x 1.25)")
    ap.add_argument("--latency-prefill-urls", default="",
                    help="comma list of latency prefill engines (DP attention 1), preferred while one has a free slot")
    ap.add_argument("--prefill-depth", type=int, default=0,
                    help="requests in flight at most per throughput prefill engine, the rest wait at the router in "
                         "arrival order (0: no cap)")
    ap.add_argument("--latency-depth", type=int, default=1, help="the same per latency prefill engine (0: no cap)")
    ap.add_argument("--prefill-units", default="",
                    help="layer pipelines as prefill engines: stage urls in order, comma-separated, units separated by ';'")
    ap.add_argument("--unit-depth", type=int, default=1, help="requests in flight at most per prefill unit (0: no cap)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--wait", type=float, default=7200.0, help="seconds to wait for every worker's /health")
    a = ap.parse_args()

    def healths(urls):
        out, end = [], time.monotonic() + a.wait
        for u in urls:
            while True:
                try:
                    x = httpx.get(u.rstrip("/") + "/health", timeout=5)
                    if x.status_code == 200:
                        out.append(x.json())
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > end:
                    raise SystemExit(f"{u} did not answer /health within {a.wait:g} s")
                time.sleep(5)
        return out

    pu, du = [u for u in a.prefill_urls.split(",") if u], a.decode_urls.split(",")
    lu = [u for u in a.latency_prefill_urls.split(",") if u]
    units = [[u for u in g.split(",") if u] for g in a.prefill_units.split(";") if g.strip()]
    ph, lh, dh = healths(pu), healths(lu), healths(du)
    uh = [healths(g) for g in units]
    for g, hs in zip(units, uh):
        check_unit(g, hs)
        print(f"kiln pd router: prefill unit {g}: layers {[h['pp']['layers'] for h in hs]}", flush=True)
    check_workers(ph + lh + [hs[0] for hs in uh], dh)
    for u, h in list(zip(pu, ph)) + list(zip(lu, lh)) + list(zip(du, dh)):
        print(f"kiln pd router: {u} role {h['role']} layout {h.get('layout')}{' (latency)' if u in lu else ''}",
              flush=True)
    tok = None
    if a.tokenizer and a.threshold > 0:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    prefill = [Worker(u, h, "throughput", a.prefill_depth) for u, h in zip(pu, ph)] + \
        [Worker(u, h, "latency", a.latency_depth) for u, h in zip(lu, lh)] + \
        [Worker(g[0], hs[0], "latency", a.unit_depth) for g, hs in zip(units, uh)]  # a unit: posts go to stage 0
    r = PDRouter(prefill, [Worker(u, h) for u, h in zip(du, dh)], a.threshold, tok, a.decode_credits)
    if tok is not None:
        try:
            served = httpx.get(du[0].rstrip("/") + "/v1/models", timeout=30).json()["data"][0]["id"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            served = f"unknown ({e!r})"
        r.forward_ids = not a.no_forward_ids and served == a.tokenizer
        print(f"kiln pd router: prompt token ids forwarded to the engines: {'on' if r.forward_ids else 'off'} "
              f"(tokenizer {a.tokenizer}, served model {served}{', --no-forward-ids' if a.no_forward_ids else ''})",
              flush=True)
    import uvicorn

    uvicorn.run(build_pd_app(r), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
