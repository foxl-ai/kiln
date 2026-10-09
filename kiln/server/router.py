"""Cache-aware router over data-parallel Kiln workers (SGLang model gateway's policy).

Each worker is a full Kiln server on its own NeuronCore(s). The router keeps, per worker,
an approximate prefix tree of the request text it sent there, and routes a request:

1. If the load is imbalanced (max - min in-flight > balance_abs and max > balance_rel *
   min), to the least-loaded worker: load wins over locality.
2. Else to the worker whose tree matches the longest prefix, if that match covers at least
   `cache_threshold` of the request; its radix cache very likely still holds the KV.
3. Else to the least-loaded worker, the smallest tree (the most room for new prefixes) breaking ties.

The tree is text-level, like SGLang's router, so it needs no tokenizer; it is pruned to
`max_tree_chars` per worker by evicting least-recently-used leaves.

    python -m kiln.server.router --workers 2 --port 8000 -- --model Qwen/Qwen3-0.6B --device neuron
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import signal
import subprocess
import sys
import time

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


class _Node:
    __slots__ = ("children", "text", "last", "parent")

    def __init__(self, text: str = "", parent: "_Node | None" = None):
        self.children: dict[str, _Node] = {}
        self.text = text
        self.last = 0.0
        self.parent = parent


class PrefixTree:
    """Character radix tree with LRU leaf pruning."""

    def __init__(self, max_chars: int):
        self.root = _Node()
        self.max_chars = max_chars
        self.size = 0

    def match(self, s: str) -> int:
        node, i = self.root, 0
        now = time.monotonic()
        while i < len(s):
            child = node.children.get(s[i])
            if child is None:
                break
            n = 0
            while n < len(child.text) and i + n < len(s) and child.text[n] == s[i + n]:
                n += 1
            child.last = now
            i += n
            if n < len(child.text):
                break
            node = child
        return i

    def insert(self, s: str) -> None:
        node, i = self.root, 0
        now = time.monotonic()
        while i < len(s):
            child = node.children.get(s[i])
            if child is None:
                leaf = _Node(s[i:], node)
                leaf.last = now
                node.children[s[i]] = leaf
                self.size += len(leaf.text)
                break
            n = 0
            while n < len(child.text) and i + n < len(s) and child.text[n] == s[i + n]:
                n += 1
            if n < len(child.text):  # split
                upper = _Node(child.text[:n], node)
                upper.last = now
                node.children[s[i]] = upper
                child.text = child.text[n:]
                child.parent = upper
                upper.children[child.text[0]] = child
                child = upper
            child.last = now
            i += n
            node = child
        while self.size > self.max_chars:
            self._evict_one()

    def _evict_one(self) -> None:
        leaves, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if not n.children and n is not self.root:
                leaves.append(n)
            stack.extend(n.children.values())
        if not leaves:
            self.size = 0
            return
        victim = min(leaves, key=lambda n: n.last)
        self.size -= len(victim.text)
        del victim.parent.children[victim.text[0]]


class Router:
    def __init__(self, workers: list[str], cache_threshold: float = 0.5, balance_abs: int = 32,
                 balance_rel: float = 1.5, max_tree_chars: int = 4_000_000):
        self.workers = workers
        self.trees = [PrefixTree(max_tree_chars) for _ in workers]
        self.load = [0] * len(workers)
        self.cache_threshold, self.balance_abs, self.balance_rel = cache_threshold, balance_abs, balance_rel
        self.routed = {"cache": 0, "balance": 0, "smallest": 0}

    def pick(self, text: str) -> int:
        lo, hi = min(self.load), max(self.load)
        if hi - lo > self.balance_abs and hi > self.balance_rel * max(lo, 1):
            self.routed["balance"] += 1
            w = self.load.index(lo)
        else:
            matches = [t.match(text) for t in self.trees]
            best = max(range(len(matches)), key=lambda i: (matches[i], -self.load[i]))
            if text and matches[best] / len(text) >= self.cache_threshold:
                self.routed["cache"] += 1
                w = best
            else:
                self.routed["smallest"] += 1
                # Load first, then the smaller tree. Tree size alone degenerates once the trees reach max_tree_chars:
                # the worker that receives a request is pruned back below the other one's size, so it wins the next
                # request too, and only the balance rule (an imbalance over balance_abs) ever sends one elsewhere.
                # Measured on kiln-t2-cb2 (trn2.48xlarge, two GLM-5.3-Flash engines, unrelated 8192-token real-text
                # prompts, 2026-10-08): at conc 32 one engine served 63 of 64 requests and the other was 19% busy.
                w = min(range(len(self.trees)), key=lambda i: (self.load[i], self.trees[i].size))
        self.trees[w].insert(text)
        return w


def _request_text(path: str, body: dict) -> str:
    if path.endswith("chat/completions"):
        return json.dumps(body.get("messages", []), ensure_ascii=False)
    p = body.get("prompt", body.get("text", ""))
    return p if isinstance(p, str) else json.dumps(p)


def build_router_app(router: Router) -> FastAPI:
    app = FastAPI(title="kiln-router")
    # A streaming request holds its connection to the worker until the last token, so the pool's connection cap is a cap
    # on requests in flight at every worker together. httpx's default (Limits(max_connections=100), shared by all hosts)
    # held a whole-box run at 100 in flight at conc 128..256: the rest waited inside client.send() here, which the client
    # measured as TTFT (kiln-t2-cb2, trn2.48xlarge, 2026-10-07: two engines behind this router served 401.9 out tok/s at
    # conc 128 and 316.9 at conc 256; one engine alone served 252.9 at conc 64 in-process). No cap, as the PD router
    # (pd_router.py build_pd_app), with its keep-alive expiry: idle connections close here before an engine closes them.
    client = httpx.AsyncClient(timeout=None, limits=httpx.Limits(
        max_connections=None, max_keepalive_connections=512, keepalive_expiry=30.0))

    @app.get("/health")
    async def health():
        return {"status": "ok", "workers": router.workers, "load": router.load, "routed": router.routed}

    @app.get("/v1/models")
    async def models():
        r = await client.get(router.workers[0] + "/v1/models")
        return JSONResponse(r.json(), status_code=r.status_code)

    async def proxy(path: str, http: Request):
        body = await http.json()
        w = router.pick(_request_text(path, body))
        url = router.workers[w] + path
        router.load[w] += 1
        if body.get("stream"):
            req = client.build_request("POST", url, json=body)
            resp = await client.send(req, stream=True)

            async def relay():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp.aclose()  # closing upstream aborts the request on the worker
                    router.load[w] -= 1

            return StreamingResponse(relay(), status_code=resp.status_code,
                                     media_type=resp.headers.get("content-type", "text/event-stream"))
        try:
            r = await client.post(url, json=body)
        finally:
            router.load[w] -= 1
        return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))

    def route(path: str):
        async def handler(http: Request):  # annotated so FastAPI passes the request
            return await proxy(path, http)

        return handler

    for path in ("/v1/completions", "/v1/chat/completions", "/generate"):
        app.add_api_route(path, route(path), methods=["POST"])
    return app


def _exit_on_signal(signum, frame) -> None:
    raise SystemExit(128 + signum)


def _dies_with(parent: int):
    """preexec_fn for a worker: on Linux the kernel sends it SIGTERM when the router dies by any
    signal, SIGKILL included, which no handler in the router can see (prctl PR_SET_PDEATHSIG). A
    router that died before the prctl took effect has already reparented the worker: it exits."""

    def set_pdeathsig() -> None:
        if sys.platform == "linux":
            import ctypes

            ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)  # 1 = PR_SET_PDEATHSIG
            if os.getppid() != parent:
                os._exit(1)

    return set_pdeathsig


def stop_workers(procs: list[subprocess.Popen], timeout: float = 30.0) -> None:
    """SIGTERM every worker, wait for them all, SIGKILL whatever is still running at the deadline."""
    for p in procs:
        p.terminate()
    deadline = time.monotonic() + timeout
    for p in procs:
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def main() -> None:
    argv = sys.argv[1:]
    engine_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(prog="kiln.server.router")
    ap.add_argument("--workers", type=int, default=2, help="worker processes to launch")
    ap.add_argument("--worker-urls", default=None, help="comma list of existing workers instead")
    ap.add_argument("--cores-per-worker", type=int, default=1)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cache-threshold", type=float, default=0.5)
    args = ap.parse_args(argv)
    # uvicorn (0.52.3) re-raises a SIGTERM it caught, after its graceful shutdown, under the handler
    # it found (server.py capture_signals). Python's default handler ends the process there, so the
    # `finally` below never ran and stopping the router orphaned every worker it launched, each
    # holding its NeuronCores: tests/test_router.py left two `kiln --port 1865x` workers running
    # under init on kiln-g2-trn1 (2026-10-04). As an exception the signal unwinds through the
    # finally, also during the workers' start-up, which was outside it.
    signal.signal(signal.SIGTERM, _exit_on_signal)
    procs: list[subprocess.Popen] = []
    try:
        if args.worker_urls:
            urls = args.worker_urls.split(",")
        else:
            urls = []
            for i in range(args.workers):
                port = args.port + 1 + i
                env = dict(os.environ)
                first = i * args.cores_per_worker
                env["NEURON_RT_VISIBLE_CORES"] = (f"{first}" if args.cores_per_worker == 1
                                                  else f"{first}-{first + args.cores_per_worker - 1}")
                procs.append(subprocess.Popen([sys.executable, "-m", "kiln", "--port", str(port), *engine_args],
                                              env=env, preexec_fn=_dies_with(os.getpid())))
                urls.append(f"http://127.0.0.1:{port}")
            for u in urls:  # wait for every worker to come up
                for _ in itertools.count():
                    try:
                        if httpx.get(u + "/health", timeout=2).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if any(p.poll() is not None for p in procs):
                        raise SystemExit("a worker exited during start-up")
                    time.sleep(2)
        import uvicorn

        uvicorn.run(build_router_app(Router(urls, args.cache_threshold)), host=args.host, port=args.port)
    finally:
        stop_workers(procs)

if __name__ == "__main__":
    main()
