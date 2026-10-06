"""The disaggregated deployment over HTTP on CPU (kiln.server.pd_router + bench/pd_serve.py): a prefill server, a
decode server and the router, as separate processes on loopback, against one engine with the same configuration.

Covers: OpenAI-compatible streaming and non-streaming completions and chat through the router, threshold routing
(a short prompt goes straight to the decode server, which prefills it), backpressure (more requests than the decode
server's credits wait at the router and all finish), the role check at the router's start, and the metrics.
Needs KILN_TEST_MODEL (a real checkpoint with a tokenizer, as tests/test_router.py).
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import time

import pytest

from kiln.server.pd_router import check_workers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_role_check_refuses_incomplete_or_mismatched_deployments():
    p = {"role": "prefill", "layout": "a", "max_num_seqs": 4}
    d = {"role": "decode", "layout": "a", "max_num_seqs": 4, "pd_address": "127.0.0.1:1"}
    check_workers([p], [d])
    with pytest.raises(SystemExit, match="role check failed"):
        check_workers([p], [])
    with pytest.raises(SystemExit, match="role check failed"):
        check_workers([dict(p, role="both")], [d])
    with pytest.raises(SystemExit, match="role check failed"):
        check_workers([p], [dict(d, role="prefill")])
    with pytest.raises(SystemExit, match="layouts differ"):
        check_workers([p], [dict(d, layout="b")])
    with pytest.raises(SystemExit, match="no handoff receiver"):
        check_workers([p], [dict(d, pd_address=None)])


def test_prefill_depth_cap_late_binds_in_arrival_order():
    """--prefill-depth / --latency-prefill-urls: latency engines first while one has a free slot, then the least loaded
    throughput engine under its cap; once every engine is at its cap requests wait at the router in arrival order, a
    freed slot goes straight to the first waiter (a newcomer cannot overtake it), and a waiter that gives up (or is
    granted a slot and then gives up) leaves no slot behind. Without caps nothing waits."""
    import asyncio

    from kiln.server.pd_router import PDRouter, Worker

    ph = {"role": "prefill", "max_num_seqs": 4}
    dh = {"role": "decode", "max_num_seqs": 4, "pd_address": "127.0.0.1:1"}

    async def capped():
        t1, t2 = Worker("t1", ph, "throughput", 2), Worker("t2", ph, "throughput", 2)
        lat = Worker("l", ph, "latency", 1)
        r = PDRouter([t1, t2, lat], [Worker("d", dh)], 0)
        held = [await r.take_prefill() for _ in range(5)]
        assert [w.url for w in held] == ["l", "t1", "t2", "t1", "t2"]
        order = []

        async def waiter(i):
            w = await r.take_prefill()
            order.append((i, w.url))
            return w

        tasks = [asyncio.create_task(waiter(i)) for i in range(4)]
        await asyncio.sleep(0)
        assert len(r.pwait) == 4 and not order
        tasks[1].cancel()  # gives up while waiting
        await asyncio.sleep(0)
        r.give_prefill(held[3])  # t1 frees a slot: the first waiter's
        await asyncio.sleep(0)
        r.give_prefill(held[0])  # the latency engine frees: waiter 2 (waiter 1 left)
        await asyncio.sleep(0)
        assert order == [(0, "t1"), (2, "l")]
        newcomer = asyncio.create_task(r.take_prefill())
        await asyncio.sleep(0)
        assert not newcomer.done() and len(r.pwait) == 2  # behind waiter 3
        r.give_prefill(held[4])  # t2 frees: waiter 3, then cancelled before it runs
        tasks[3].cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert newcomer.done() and newcomer.result().url == "t2"  # the abandoned grant passed on
        assert (t1.inflight, t2.inflight, lat.inflight) == (2, 2, 1) and not r.pwait
        assert r.prefill_queue_s.n == 5 + 2 + 1
        assert 'kiln:pd_router_prefill_depth{kind="latency",url="l"} 1' in r.render()

    async def uncapped():
        t1, t2 = Worker("t1", ph), Worker("t2", ph)
        r = PDRouter([t1, t2], [Worker("d", dh)], 0)
        got = [await r.take_prefill() for _ in range(6)]
        assert [w.url for w in got] == ["t1", "t2"] * 3 and not r.pwait

    asyncio.run(capped())
    asyncio.run(uncapped())


def _sweep_args(model):
    return ["--model", model, "--device", "cpu", "--tp", "1", "--max-num-seqs", "4", "--max-model-len", "512",
            "--prefill-tokens", "64", "--kv-cache-gb", "0.25", "--concurrency", "4"]


def _wait(url, proc, secs=600):
    import httpx

    end = time.monotonic() + secs
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"{url}: process exited {proc.returncode}")
        try:
            if httpx.get(url + "/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError(url)


def _stream(base, body):
    import httpx

    toks, lps, text = [], [], ""
    with httpx.stream("POST", base + "/v1/completions", json={**body, "stream": True, "logprobs": 0},
                      timeout=600) as r:
        assert r.status_code == 200, r.read()
        lines = [l for l in r.iter_lines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]", lines[-3:]
    for l in lines[:-1]:
        ev = json.loads(l[6:])
        assert "error" not in ev, ev
        c = ev["choices"][0]
        text += c["text"]
        if c.get("logprobs"):
            toks += c["logprobs"]["tokens"]
            lps += c["logprobs"]["token_logprobs"]
    return text, toks, lps


def test_pd_deployment_over_http():
    import httpx

    model = os.environ.get("KILN_TEST_MODEL")
    if not model:
        pytest.skip("set KILN_TEST_MODEL to run")
    sys.path.insert(0, os.path.join(ROOT, "bench"))
    import serve_sweep

    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    base_port = 19100 + os.getpid() % 400
    pp, dp, rp, hp = base_port, base_port + 1, base_port + 2, base_port + 3
    env = dict(os.environ, PYTHONPATH=ROOT)
    serve = [sys.executable, os.path.join(ROOT, "bench", "pd_serve.py")]
    procs = []
    try:
        procs.append(subprocess.Popen(serve + ["--pd-role", "prefill", "--port", str(pp), "--"] + _sweep_args(model),
                                      env=env))
        procs.append(subprocess.Popen(serve + ["--pd-role", "decode", "--pd-listen", f"127.0.0.1:{hp}",
                                               "--pd-bypass-prefill", "--port", str(dp), "--"] + _sweep_args(model),
                                      env=env))
        _wait(f"http://127.0.0.1:{pp}", procs[0])
        _wait(f"http://127.0.0.1:{dp}", procs[1])
        assert httpx.get(f"http://127.0.0.1:{pp}/health").json()["role"] == "prefill"
        procs.append(subprocess.Popen([sys.executable, "-m", "kiln.server.pd_router", "--prefill-urls",
                                       f"http://127.0.0.1:{pp}", "--decode-urls", f"http://127.0.0.1:{dp}",
                                       "--threshold", "16", "--tokenizer", model, "--decode-credits", "2",
                                       "--port", str(rp)], env=env))
        base = f"http://127.0.0.1:{rp}"
        _wait(base, procs[2])
        long_prompt = "The history of the city goes back many centuries, and its people " * 3
        short_prompt = "Hello there"

        # The reference: one engine, configured as the servers are.
        args = serve_sweep.build_parser().parse_args(_sweep_args(model))
        ref = LLMEngine(serve_sweep.engine_config(args))
        try:
            tok = ref.tokenizer
            sp = SamplingParams(max_new_tokens=12, logprobs=0)
            want = {p: ref.generate([tok(p)["input_ids"]], sp)[0] for p in (long_prompt, short_prompt)}
        finally:
            ref.close()

        for p in (long_prompt, short_prompt):  # disaggregated, then bypassed
            text, toks, lps = _stream(base, {"prompt": p, "max_tokens": 12, "temperature": 0})
            w = want[p]
            assert text == tok.decode(w.output_ids, skip_special_tokens=True), (p, text)
            assert len(lps) == len(w.output_ids)
            err = max(abs(a - b[0]) for a, b in zip(lps, w.logprobs))
            print(f"pd http {'long' if p is long_prompt else 'short'}: text equal, max |dlogprob| {err:.2e}")
            assert err < 0.05, err  # bf16 on CPU: the last prompt token runs in a prefill chunk, not a decode
        h = httpx.get(base + "/health").json()
        assert h["counts"]["disaggregated"] == 1 and h["counts"]["bypass"] == 1, h

        r = httpx.post(base + "/v1/completions", json={"prompt": long_prompt, "max_tokens": 12, "temperature": 0},
                       timeout=600)
        assert r.status_code == 200 and r.json()["choices"][0]["text"] == text_of(want[long_prompt], tok)
        r = httpx.post(base + "/v1/chat/completions", timeout=600,
                       json={"messages": [{"role": "user", "content": long_prompt}], "max_tokens": 6, "temperature": 0})
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] is not None, r.text

        # Backpressure: 6 at once against 2 decode credits: all finish, some waited at the router.
        with concurrent.futures.ThreadPoolExecutor(6) as ex:
            outs = list(ex.map(lambda i: _stream(base, {"prompt": long_prompt + f" {i}", "max_tokens": 8,
                                                         "temperature": 0}), range(6)))
        assert all(len(o[2]) == 8 for o in outs)
        m = httpx.get(base + "/metrics").text
        assert "kiln:pd_router_queue_seconds_count" in m
        dm = httpx.get(f"http://127.0.0.1:{dp}/metrics").text
        assert 'kiln:pd_role{role="decode"} 1' in dm and "kiln:pd_buffer_full_events_total 0" in dm, dm
        assert "kiln:pd_injected_total 9" in dm, [l for l in dm.splitlines() if "pd_" in l]
        pm = httpx.get(f"http://127.0.0.1:{pp}/metrics").text
        assert "kiln:pd_handed_off_total 9" in pm, [l for l in pm.splitlines() if "pd_" in l]
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


def text_of(req, tok):
    return tok.decode(req.output_ids, skip_special_tokens=True)


def test_incremental_detokenizer_streams_the_same_text():
    """A decode server's streams detokenize incrementally (api._Detok incremental): the text streamed token by token,
    with and without stop strings, equals the full-decode detokenizer's, on natural text with multi-byte characters
    and on random ids. Needs KILN_TEST_MODEL (its tokenizer)."""
    import random

    model = os.environ.get("KILN_TEST_MODEL")
    if not model:
        pytest.skip("set KILN_TEST_MODEL to run")
    from transformers import AutoTokenizer

    from kiln.server.api import _Detok

    tok = AutoTokenizer.from_pretrained(model)
    rng = random.Random(0)
    texts = ["Grüße aus Köln: 東京タワーは高さ333メートル。 Ünïcödé — and emoji \U0001F600 too.\n\tdone.",
             "def f(x):\n    return x ** 2  # squares\n" * 3]
    seqs = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    seqs += [[rng.randrange(0, min(tok.vocab_size, 150000)) for _ in range(200)] for _ in range(4)]
    for ids in seqs:
        for stops in ((), ("tower",), ("\n\n", "return")):
            outs = []
            for inc in (False, True):
                d = _Detok(tok, stops, incremental=inc)
                outs.append("".join(d.push([t], i == len(ids) - 1) for i, t in enumerate(ids)))
            assert outs[0] == outs[1], (stops, outs)
