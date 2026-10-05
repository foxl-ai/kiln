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
