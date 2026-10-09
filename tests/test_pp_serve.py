"""A layer pipeline served as one prefill engine of a disaggregated deployment (engine/pp.py "Following stages",
kiln.server.pd_router --prefill-units): stage 0 a pd_serve prefill server, stage 1 a following stage (tools/pp_follow.py
--pd-role prefill), a decode server, and the router in front, each its own process over HTTP, on KILN_TEST_MODEL (the
CPU model of tests/test_pd_router.py). A request posted to the router reaches stage 0 only, stage 1 learns it from
stage 0's plan frame, both stages hand their own layers' share to the decode engine, and the client's stream equals
one engine's text, its logprobs within the CPU bf16 tolerance of tests/test_pd_router.py; plus the router's unit
checks."""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

from kiln.server.pd_router import check_unit
from tests.test_pd_router import ROOT, _stream, _sweep_args, _wait, text_of


def _h(stage, stages, layers, layout="L", role="prefill", follow=True):
    return {"role": role, "layout": layout, "pp": {"stage": stage, "stages": stages, "layers": layers, "follow": follow}}


def test_unit_check_refuses_stages_that_are_not_one_pipeline():
    urls = ["a", "b", "c"]
    check_unit(urls, [_h(0, 3, [0, 4]), _h(1, 3, [4, 9]), _h(2, 3, [9, 12])])
    for bad in ([_h(0, 3, [0, 4]), _h(2, 3, [4, 9]), _h(1, 3, [9, 12])],  # out of order
                [_h(0, 3, [0, 4]), _h(1, 3, [5, 9]), _h(2, 3, [9, 12])],  # a gap
                [_h(0, 3, [0, 4]), _h(1, 3, [4, 9], follow=False), _h(2, 3, [9, 12])],  # a lockstep stage
                [_h(0, 3, [0, 4]), _h(1, 3, [4, 9], layout="M"), _h(2, 3, [9, 12])],  # another layout
                [_h(0, 3, [0, 4]), _h(1, 3, [4, 9], role="decode"), _h(2, 3, [9, 12])]):
        with pytest.raises(SystemExit):
            check_unit(urls, bad)


def test_pipeline_unit_over_http():
    import httpx

    model = os.environ.get("KILN_TEST_MODEL")
    if not model:
        pytest.skip("set KILN_TEST_MODEL to run")
    sys.path.insert(0, os.path.join(ROOT, "bench"))
    import serve_sweep

    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    base = 19600 + os.getpid() % 300
    p0, h1, link, dp, hp, rp = base, base + 1, base + 2, base + 4, base + 5, base + 6
    env = dict(os.environ, PYTHONPATH=ROOT, KILN_PP_ASYNC="1")
    sweep = _sweep_args(model) + ["--piecewise"]
    pp = ["--pp-stages", "2", "--pp-split", "14", "--pp-follow"]
    procs = []
    try:
        procs.append(subprocess.Popen([sys.executable, os.path.join(ROOT, "tools", "pp_follow.py"), "--pd-role",
                                       "prefill", "--health-port", str(h1), "--"] + sweep + pp +
                                      ["--pp-stage", "1", "--pp-listen", f"127.0.0.1:{link}"], env=env))
        procs.append(subprocess.Popen([sys.executable, os.path.join(ROOT, "bench", "pd_serve.py"), "--pd-role",
                                       "prefill", "--port", str(p0), "--"] + sweep + pp +
                                      ["--pp-stage", "0", "--pp-next", f"127.0.0.1:{link}"], env=env))
        procs.append(subprocess.Popen([sys.executable, os.path.join(ROOT, "bench", "pd_serve.py"), "--pd-role",
                                       "decode", "--pd-listen", f"127.0.0.1:{hp}", "--port", str(dp), "--"]
                                      + _sweep_args(model), env=env))
        _wait(f"http://127.0.0.1:{h1}", procs[0])
        _wait(f"http://127.0.0.1:{p0}", procs[1])
        _wait(f"http://127.0.0.1:{dp}", procs[2])
        hs = [httpx.get(f"http://127.0.0.1:{u}/health").json() for u in (p0, h1)]
        assert [h["pp"]["stage"] for h in hs] == [0, 1] and hs[0]["pp"]["layers"][1] == hs[1]["pp"]["layers"][0], hs
        procs.append(subprocess.Popen([sys.executable, "-m", "kiln.server.pd_router", "--prefill-units",
                                       f"http://127.0.0.1:{p0},http://127.0.0.1:{h1}", "--decode-urls",
                                       f"http://127.0.0.1:{dp}", "--threshold", "0", "--tokenizer", model,
                                       "--port", str(rp)], env=env))
        router = f"http://127.0.0.1:{rp}"
        _wait(router, procs[3])
        prompts = ["The history of the city goes back many centuries, and its people " * 3,
                   "A short note on rivers, lakes and the sea, written for children " * 2]

        args = serve_sweep.build_parser().parse_args(_sweep_args(model))
        ref = LLMEngine(serve_sweep.engine_config(args))
        try:
            tok = ref.tokenizer
            sp = SamplingParams(max_new_tokens=12, logprobs=0)
            want = {p: ref.generate([tok(p)["input_ids"]], sp)[0] for p in prompts}
        finally:
            ref.close()
        for p in prompts:
            text, toks, lps = _stream(router, {"prompt": p, "max_tokens": 12, "temperature": 0})
            w = want[p]
            assert text == text_of(w, tok), (p, text)
            err = max(abs(a - b[0]) for a, b in zip(lps, w.logprobs))
            print(f"pp unit over http: text equal, max |dlogprob| {err:.2e}")
            assert err < 0.05, err
        h = httpx.get(router + "/health").json()
        assert h["counts"]["disaggregated"] == 2 and h["counts"]["prefill_errors"] == 0, h
        # A SIGTERM closes stage 0's engine (kiln.server.api.serve): it exits through SystemExit, not by the signal
        # uvicorn raises again, and its close frame ends the following stage's run, which exits 0.
        procs[1].terminate()
        assert procs[1].wait(timeout=120) == 128 + signal.SIGTERM
        assert procs[0].wait(timeout=120) == 0
        procs[2].terminate()
        assert procs[2].wait(timeout=120) == 128 + signal.SIGTERM
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
