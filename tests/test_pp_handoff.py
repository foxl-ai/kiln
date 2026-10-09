"""A pipeline's stages hand a prefilled request off to one decode engine (engine/pp.py with engine/disagg.py: every stage
the caches and state rows of its own layers, disagg.combine_stages on the decode side): 2 and 3 stage engines, each its
own process with pd_role prefill and the same requests in the same order under the same transfer ids, into a decode
engine in this process, on the tiny GLM-5.3 config (KDA + pooled DSA + MoE, tests/test_glm5_next.py's builder,
transformers >= 5.18), over the host path and over NIXL (UCX on the CPU): every request's tokens equal one engine's and
its logprobs are within LOGPROB_TOL, the first token and sampler state coming from the last stage. A handoff whose
stages do not tile the model's layers is refused and every stage's share is released."""

from __future__ import annotations

import multiprocessing as mp
import os
import time

import pytest

from tests.test_disagg import compare, drive, params
from tests.test_pp import _free_port_block, glm


def test_combine_stages_tiles_the_layers():
    from kiln.engine.disagg import combine_stages

    def meta(t, lo, hi, ranges, **kw):
        return {"xfer": "x", "token": 7 if t == len(ranges) - 1 else None, "parts": [f"p{t}.s0", f"p{t}.r0"],
                "pp": {"stage": t, "stages": len(ranges), "layers": [lo, hi], "ranges": ranges}, **kw}

    r = [[0, 3], [3, 8]]
    got = combine_stages({0: meta(0, 0, 3, r), 1: meta(1, 3, 8, r)})
    assert "error" not in got and got["token"] == 7 and got["parts"] == ["p0.s0", "p0.r0", "p1.s0", "p1.r0"]
    assert got["pp"]["ranges"] == r
    gap = combine_stages({0: meta(0, 0, 3, [[0, 3], [5, 8]]), 1: meta(1, 5, 8, [[0, 3], [5, 8]])})
    assert "do not tile" in gap["error"]
    two = combine_stages({0: meta(0, 0, 3, [[0, 3], [3, 8]]), 1: meta(1, 3, 8, [[0, 4], [4, 8]])})
    assert "different splits" in two["error"]
    nx = {"engine": "e", "bytes": 10, "deadline": 100.0, "reply": "h:1"}
    mixed = combine_stages({0: meta(0, 0, 3, r, transport="nixl", nixl=nx), 1: meta(1, 3, 8, r)})
    assert "different transports" in mixed["error"]
    both = combine_stages({0: meta(0, 0, 3, r, transport="nixl", nixl=nx),
                           1: meta(1, 3, 8, r, transport="nixl", nixl=dict(nx, bytes=5, deadline=50.0))})
    assert "error" not in both and both["nixl"]["bytes"] == 15 and both["nixl"]["deadline"] == 50.0
    assert [x["bytes"] for x in both["nixl"]["pp"]] == [10, 5]


KW = dict(page_size=8, max_prefill_tokens=12, max_num_seqs=4, piecewise=True, piecewise_group=1)
ENV = {"KILN_DSA_LONG_KEYS": "16"}  # the long path past 16 keys (tests/test_pp.py): every prompt here takes it


def _stage(path, ps, sps, stage, stages, split, ports, dest, env, q, kw=None):
    os.environ.update(env)
    from kiln.models import dsa_long, mla

    dsa_long.LONG_KEYS = int(env["KILN_DSA_LONG_KEYS"])
    if "KILN_DSA_POOL_CACHE" in env:
        mla.POOL_CACHE = env["KILN_DSA_POOL_CACHE"]
    try:
        if not isinstance(dest, str):  # a queue: the decode engine's address, once it is up
            dest = dest.get(timeout=600)
        pkw = dict(pp_stage=stage, pp_stages=stages, pp_split=split[stage] if isinstance(split, dict) else split,
                   pp_listen=f"127.0.0.1:{ports[stage]}" if stage > 0 else None,
                   pp_next=f"127.0.0.1:{ports[stage + 1]}" if stage < stages - 1 else None)
        eng = glm.engine(path, pd_role="prefill", **(kw or KW), **pkw)
        try:
            for i, (p, sp) in enumerate(zip(ps, sps)):
                eng.add_request(p, sp, rid=f"r{i}", handoff=(f"x{i}", dest))
            end = time.monotonic() + 300
            while (eng.has_work() or eng._pd_pins) and time.monotonic() < end:
                if eng.has_work():
                    eng.step()
                    eng.pd_flush()
                else:
                    time.sleep(0.01)
            eng.pd_flush()
            q.put(("ok", stage, dict(eng.pd_counts), len(eng._pd_pins)))
        finally:
            eng.close()
    except BaseException as e:  # noqa: BLE001  reported to the test
        q.put(("error", stage, repr(e), -1))
        raise


def _pipeline_into(path, D, ps, sps, stages, split, env):
    """Start the stage processes towards decode engine D, drive D until every request ends, and return
    (D's outputs, the stages' final statuses)."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    base = _free_port_block(stages)
    ports = [base + s for s in range(stages)]
    procs = [ctx.Process(target=_stage, args=(path, ps, sps, s, stages, split, ports, D.pd_address, env, q))
             for s in range(stages)]
    for p in procs:
        p.start()
    try:
        got = drive(None, D, ps, sps, timeout=600)
        status = {}
        for _ in range(stages):
            st = q.get(timeout=300)
            status[st[1]] = st
        for p in procs:
            p.join(timeout=120)
        return got, status
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=30)


def _one_engine(path, ps, sps):
    eng = glm.engine(path, **KW)
    try:
        reqs = eng.generate(ps, sps)
        return [(r.output_ids, [x[0] for x in r.logprobs], r.finish_reason, dict(r.prompt_logprobs)) for r in reqs]
    finally:
        eng.close()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    path = str(tmp_path_factory.mktemp("pp_handoff"))
    glm.build(path, seed=3, index_topk=16, max_position_embeddings=4096)
    return path


def _env(monkeypatch, transport):
    # dsa_long reads KILN_DSA_LONG_KEYS once, at import: import it BEFORE the variable is set, or a first import here
    # leaves LONG_KEYS at 16 for every later test in the process (monkeypatch then restores 16 as the "original"), and
    # their prompts take the long path.
    from kiln.models import dsa_long

    monkeypatch.setattr(dsa_long, "LONG_KEYS", 16)
    env = dict(ENV, KILN_PD_TRANSPORT=transport)
    if transport == "nixl":
        pytest.importorskip("nixl._api")
        env["KILN_PD_NIXL_REPLY_LISTEN"] = "127.0.0.1:0"
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return env


@pytest.mark.parametrize("transport", ["host", "nixl"])
@pytest.mark.parametrize("stages,split", [(2, None), (3, (2, 5))])
def test_pipeline_stages_hand_off_to_one_decode_engine(tiny, monkeypatch, transport, stages, split):
    env = _env(monkeypatch, transport)
    ps = glm.prompts(7, (70, 41, 97, 33))
    sps = params(len(ps), new_tokens=6)
    want = _one_engine(tiny, ps, sps)
    D = glm.engine(tiny, pd_role="decode", pd_listen="127.0.0.1:0", **KW)
    try:
        got, status = _pipeline_into(tiny, D, ps, sps, stages, split, env)
        for s, st in status.items():
            assert st[0] == "ok" and st[3] == 0, st  # every stage released (nixl: its pins) everything it held
            assert st[2].get("handed_off", 0) == len(ps), st
        compare(got, want, f"{stages} stages into one decode engine, {transport}")
        assert D.pd_counts["admitted"] == len(ps) and D.pd_counts["injected"] == len(ps)
        D.pd_receiver.drain()
        assert D.pd_receiver.stats.held_bytes == 0 and not D.pd_receiver.nixl_reply
    finally:
        D.close()


def test_stages_that_do_not_tile_the_layers_are_refused_and_released(tiny, monkeypatch):
    """Stage 0 believes the split is (3,), stage 1 that it is (5,): their ranges [0, 3) and [5, 8) leave layers 3 and 4
    out, so the decode engine refuses every request (its receiver says why) and both stages' shares are released."""
    env = _env(monkeypatch, "nixl" if os.environ.get("KILN_TEST_NIXL_REFUSE") else "host")
    ps = glm.prompts(9, (40, 23))
    sps = params(len(ps), new_tokens=4)
    D = glm.engine(tiny, pd_role="decode", pd_listen="127.0.0.1:0", **KW)
    try:
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        base = _free_port_block(2)
        ports = [base, base + 1]
        procs = [ctx.Process(target=_stage, args=(tiny, ps, sps, s, 2, {0: (3,), 1: (5,)}, ports, D.pd_address, env, q))
                 for s in range(2)]
        for p in procs:
            p.start()
        refused, end = [], time.monotonic() + 600
        while len(refused) < len(ps) and time.monotonic() < end:
            assert not D.pd_poll()  # nothing is admitted
            refused = [m for m in D.pd_done if m.get("error")]
            time.sleep(0.05)
        assert len(refused) == len(ps) and all("do not tile" in m["error"] for m in refused), refused
        for _ in range(2):
            st = q.get(timeout=300)
            assert st[0] == "ok" and st[3] == 0, st
        for p in procs:
            p.join(timeout=120)
        D.pd_receiver.drain()
        assert D.pd_receiver.stats.held_bytes == 0
    finally:
        D.close()


@pytest.mark.parametrize("transport", ["host", "nixl"])
def test_stages_at_attention_tp_4_into_a_context_parallel_decode_engine(tmp_path, monkeypatch, transport):
    """The latency deployment's shape through a pipeline: 2 stages, each at tp 4 / DP attention 1 (attention TP 4, page
    size 4), into a decode engine at tp 4 / DP attention 2 (attention TP 2, page size 8) under KILN_DSA_CP=1. Every
    stage's KDA state rows are regrouped from 4 sender ranks into 2 and its DSA rows read whole and CP-selected, one
    stage at a time (ModelRunner._pd_assemble / _pd_pull): tokens equal one engine's at tp 4, every stage released."""
    dsa_long = pytest.importorskip("kiln.models.dsa_long")
    if not hasattr(dsa_long, "cp_local_slots"):
        pytest.skip("no context parallelism in this tree")
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from kiln.models import mla

    env = _env(monkeypatch, transport)
    env["KILN_DSA_POOL_CACHE"] = "separate"
    monkeypatch.setenv("KILN_DSA_POOL_CACHE", "separate")
    monkeypatch.setattr(mla, "POOL_CACHE", "separate")
    path = str(tmp_path)
    glm.build(path, seed=1, index_topk=16, max_position_embeddings=4096)
    ps = glm.prompts(13, (37, 70, 9, 21))
    sps = params(len(ps), new_tokens=6)
    skw = dict(KW, page_size=4, tp=4)
    eng = glm.engine(path, **skw)
    try:
        reqs = eng.generate(ps, sps)
        want = [(r.output_ids, [x[0] for x in r.logprobs], r.finish_reason, dict(r.prompt_logprobs)) for r in reqs]
    finally:
        eng.close()
    ctx = mp.get_context("spawn")
    q, addr = ctx.Queue(), ctx.Queue()
    base = _free_port_block(2 * 4)  # every rank of a stage listens on its own port (pp_listen's port + rank)
    ports = [base, base + 4]
    procs = [ctx.Process(target=_stage, args=(path, ps, sps, s, 2, None, ports, addr, env, q, skw)) for s in range(2)]
    for p in procs:
        p.start()
    D = None
    try:
        monkeypatch.setenv("KILN_DSA_CP", "1")  # the decode engine only (the stages were spawned without it)
        D = glm.engine(path, pd_role="decode", pd_listen="127.0.0.1:0", **dict(KW, page_size=8, tp=4, dp_attention=2))
        for _ in range(2):
            addr.put(D.pd_address)
        got = drive(None, D, ps, sps, timeout=900)
        for _ in range(2):
            st = q.get(timeout=300)
            assert st[0] == "ok" and st[3] == 0 and st[2].get("handed_off", 0) == len(ps), st
        for p in procs:
            p.join(timeout=120)
        compare(got, want, f"2 stages at attention TP 4 into a CP decode engine at attention TP 2, {transport}")
        D.pd_receiver.drain()
        assert D.pd_receiver.stats.held_bytes == 0
    finally:
        if D is not None:
            D.close()
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=30)
