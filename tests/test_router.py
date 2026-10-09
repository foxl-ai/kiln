import random

from kiln.server.router import PrefixTree, Router


def test_prefix_tree_match_insert_and_prune():
    t = PrefixTree(max_chars=1000)
    t.insert("You are a helpful assistant. Q: what is 2+2?")
    t.insert("You are a helpful assistant. Q: name a color")
    assert t.match("You are a helpful assistant. Q: what is 3+3?") == len("You are a helpful assistant. Q: what is ")
    assert t.match("Something else") == 0
    small = PrefixTree(max_chars=50)
    for i in range(20):
        small.insert(f"prefix-{i}-" + "x" * 10)
    assert small.size <= 50


def test_router_keeps_a_shared_prefix_on_one_worker():
    r = Router(["a", "b", "c"], cache_threshold=0.5)
    system = "SYSTEM: you are an expert SQL assistant. " * 10
    first = r.pick(system + "query 0")
    for i in range(1, 30):
        assert r.pick(system + f"query {i}") == first
    assert r.routed["cache"] == 29


def test_router_spreads_unrelated_requests_and_respects_load():
    rng = random.Random(0)
    r = Router(["a", "b"], cache_threshold=0.5, balance_abs=4)
    seen = {r.pick("".join(rng.choice("abcdefghij") for _ in range(40))) for _ in range(20)}
    assert seen == {0, 1}
    r.load = [10, 0]
    assert r.pick("anything") == 1 and r.routed["balance"] == 1


def _start_router(model, port):
    import subprocess
    import sys
    import time

    import httpx

    proc = subprocess.Popen([sys.executable, "-m", "kiln.server.router", "--workers", "2", "--port", str(port),
                             "--", "--model", model, "--device", "cpu", "--dtype", "float32",
                             "--max-model-len", "512", "--kv-cache-gb", "0.5", "--max-num-seqs", "4"])
    for _ in range(120):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(1)
    return proc


def _workers_answering(port, wait=0.0):
    """Worker ports (the router's port + 1, + 2) still answering /health, after up to `wait` seconds."""
    import time

    import httpx

    deadline = time.monotonic() + wait
    while True:
        alive = []
        for u in (f"http://127.0.0.1:{port + 1}", f"http://127.0.0.1:{port + 2}"):
            try:
                httpx.get(u + "/health", timeout=2)
                alive.append(u)
            except httpx.HTTPError:
                pass
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(1)


def test_router_end_to_end_with_two_cpu_workers():
    """Routing over two CPU workers; then a SIGTERM to the router stops both workers before it exits
    (they outlived it, under init, until uvicorn's signal re-raise was handled: server/router.py)."""
    import os

    import httpx
    import pytest

    model = os.environ.get("KILN_TEST_MODEL")
    if not model:
        pytest.skip("set KILN_TEST_MODEL to run")
    port = 18600 + os.getpid() % 500
    proc = _start_router(model, port)
    base = f"http://127.0.0.1:{port}"
    try:
        system = "You are a terse assistant. Answer with one word. " * 4
        outs = []
        for q in ("Capital of France?", "Capital of Italy?", "Capital of Spain?"):
            r = httpx.post(base + "/v1/completions", json={"prompt": system + q, "max_tokens": 3, "temperature": 0},
                           timeout=120)
            assert r.status_code == 200, r.text
            outs.append(r.json()["choices"][0]["text"])
        with httpx.stream("POST", base + "/v1/completions", timeout=120,
                          json={"prompt": "1, 2, 3,", "max_tokens": 5, "temperature": 0, "stream": True}) as r:
            lines = [l for l in r.iter_lines() if l.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        h = httpx.get(base + "/health").json()
        assert h["routed"]["cache"] >= 2, h  # the shared system prompt stayed on one worker
        assert len(_workers_answering(port)) == 2
    finally:
        proc.terminate()
        proc.wait(timeout=60)
    assert _workers_answering(port) == []  # the router waited for them


def test_router_sigkill_takes_its_workers():
    """A router killed outright (no handler runs) still takes its workers with it: the kernel sends
    each one SIGTERM (PR_SET_PDEATHSIG, Linux only)."""
    import os
    import signal
    import sys

    import pytest

    model = os.environ.get("KILN_TEST_MODEL")
    if not model:
        pytest.skip("set KILN_TEST_MODEL to run")
    if sys.platform != "linux":
        pytest.skip("PR_SET_PDEATHSIG is Linux only")
    port = 18600 + (os.getpid() + 250) % 500
    proc = _start_router(model, port)
    try:
        assert len(_workers_answering(port)) == 2
    finally:
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    assert _workers_answering(port, wait=60) == []


def test_router_does_not_cap_streams_in_flight():
    """Every concurrent stream reaches a worker: the router's connection pool is not a cap on requests in flight
    (httpx's default, 100 connections shared by every worker, held whole-box runs at 100 in flight and queued the
    rest inside the router, server/router.py). Two real HTTP workers on loopback hold each stream open until all of
    them have arrived; 130 streams go through the router at once."""
    import asyncio
    import socket

    import httpx
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    from kiln.server.router import build_router_app

    n = 130

    def free_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    async def main() -> int:
        seen = {"open": 0, "max": 0}
        release = asyncio.Event()
        worker = FastAPI()

        @worker.post("/v1/completions")
        async def completions():
            seen["open"] += 1
            seen["max"] = max(seen["max"], seen["open"])
            if seen["open"] == n:
                release.set()

            async def body():
                try:
                    yield b"data: {}\n\n"
                    await release.wait()
                    yield b"data: [DONE]\n\n"
                finally:
                    seen["open"] -= 1

            return StreamingResponse(body(), media_type="text/event-stream")

        ports = [free_port() for _ in range(3)]
        apps = [worker, worker, build_router_app(Router([f"http://127.0.0.1:{p}" for p in ports[:2]]))]
        servers = [uvicorn.Server(uvicorn.Config(a, host="127.0.0.1", port=p, log_level="warning"))
                   for a, p in zip(apps, ports)]
        tasks = [asyncio.create_task(s.serve()) for s in servers]
        try:
            while not all(s.started for s in servers):
                await asyncio.sleep(0.05)
            lim = httpx.Limits(max_connections=None, max_keepalive_connections=None)
            async with httpx.AsyncClient(timeout=60, limits=lim) as c:

                async def one(i: int) -> str:
                    body = {"prompt": f"request {i} " + "x" * (i % 7), "max_tokens": 1, "stream": True}
                    async with c.stream("POST", f"http://127.0.0.1:{ports[2]}/v1/completions", json=body) as r:
                        return [line async for line in r.aiter_lines() if line.startswith("data: ")][-1]

                calls = asyncio.gather(*(one(i) for i in range(n)))
                try:
                    await asyncio.wait_for(release.wait(), timeout=20)
                except asyncio.TimeoutError:
                    release.set()  # let the streams that did arrive finish, then report how many there were
                lasts = await calls
            assert lasts == ["data: [DONE]"] * n
            return seen["max"]
        finally:
            for s in servers:
                s.should_exit = True
            await asyncio.gather(*tasks)

    assert asyncio.run(main()) == n


def test_router_spreads_unrelated_requests_once_the_trees_are_full():
    """Requests that share no prefix go to the least-loaded worker even after both prefix trees reached
    max_tree_chars. By tree size alone the worker that just received a request was pruned back below the other
    one, won every following request, and only an imbalance over balance_abs moved one (server/router.py)."""
    import json

    rng = random.Random(1)
    r = Router(["a", "b"], cache_threshold=0.5, max_tree_chars=200_000)
    for _ in range(64):  # 64 in flight, none finished: the router's load counts every one of them
        w = r.pick(json.dumps([rng.randrange(1000, 150_000) for _ in range(8192)]))
        r.load[w] += 1
    assert all(t.size <= 200_000 for t in r.trees) and max(t.size for t in r.trees) > 150_000  # the trees are full
    assert abs(r.load[0] - r.load[1]) <= 1, r.load
