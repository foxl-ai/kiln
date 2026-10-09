"""The NIXL handoff transport (KILN_PD_TRANSPORT=nixl, engine/nixl_kv.py) on CPU: the same disaggregated serving as
tests/test_disagg.py, but every decode rank reads the request's rows out of the prefill ranks' memory with a real NIXL
transfer (the UCX backend over loopback, DRAM regions: the code path the device runs with LIBFABRIC and VRAM). The gate
is test_disagg's: tokens equal to one engine's, chosen-token logprobs within LOGPROB_TOL. Besides it: the prefill engine
keeps every handed-off request's pages and state row until the decode engine's release and then frees all of them, the
receive buffer counts a nixl handoff's bytes from arrival to release (and makes the next one wait when full), and a
nixl handoff into a decode engine without the transport is refused and still released.

Skipped where nixl (the DLAMI venv's nixl 1.3.2) is not installed.
"""

from __future__ import annotations

import json
import os
import socket
import time

import numpy as np
import pytest
import torch

pytest.importorskip("nixl._api")

from tests.test_disagg import compare, engine, params, prompts, single  # noqa: E402
from tests.test_linear_attn import build_qwen3_5  # noqa: E402
from tests.test_mla import build as build_mla  # noqa: E402


@pytest.fixture(autouse=True)
def nixl_env(monkeypatch):
    monkeypatch.setenv("KILN_PD_TRANSPORT", "nixl")
    monkeypatch.setenv("KILN_PD_NIXL_REPLY_LISTEN", "127.0.0.1:0")


@pytest.fixture(scope="module")
def gdn(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("nixl_qwen3_5"))
    build_qwen3_5(path)
    return path


@pytest.fixture(scope="module")
def dsa(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("nixl_glm_dsa"))
    build_mla("glm_moe_dsa", path)
    return path


def _wait_unpinned(P, timeout=60.0):
    end = time.monotonic() + timeout
    while P._pd_pins and time.monotonic() < end:
        P.has_work()  # frees what released handoffs held (engine._pd_unpin_ready)
        time.sleep(0.01)
    assert not P._pd_pins, f"{len(P._pd_pins)} handoffs were never released"


def drive(P, D, ps, sps, prefill_proc=None, timeout=300.0, tag=""):
    """test_disagg's drive, interleaved: a prefill engine whose pins fill its seats (max_num_seqs) admits more only
    after the decode engine has read and released them."""
    from tests.test_disagg import drive as drive_tcp

    got = drive_tcp(P, D, ps, sps, interleave=True, prefill_proc=prefill_proc, timeout=timeout, tag=tag)
    if P is not None:
        _wait_unpinned(P)
    return got


def pd(path, ps, sps, prefill_kw=None, decode_kw=None, **kw):
    D = engine(path, pd_role="decode", pd_listen="127.0.0.1:0", **{**kw, **(decode_kw or {})})
    try:
        assert D.pd_nixl
        P = engine(path, pd_role="prefill", **{**kw, **(prefill_kw or {})})
        try:
            free0 = [p.num_free for p in P.pools]
            got = drive(P, D, ps, sps)
            assert P.pd_counts["handed_off"] + P.pd_counts["done_at_prefill"] == len(ps)
            assert P.pd_counts["unpinned_released"] == P.pd_counts["handed_off"]
            for sch in getattr(P.scheduler, "groups", [P.scheduler]):  # every page back: free or in the prefix cache
                sch.radix.evict(sch.radix.total_pages())
            assert [p.num_free for p in P.pools] == free0
        finally:
            P.close()
        D.pd_receiver.drain()
        assert D.pd_receiver.stats.held_bytes == 0 and D.pd_receiver.stats.nixl == D.pd_counts["admitted"]
        assert not D.pd_receiver.nixl_reply
        return got, D.pd_counts
    finally:
        D.close()


# -- pieces -------------------------------------------------------------------------------------------


def test_segments_pair_runs_consecutive_on_both_sides():
    from kiln.engine import nixl_kv

    pos = np.arange(10)
    rs = nixl_kv.pos_slots([3, 4, 9], 4, pos)  # 12..19 then 36, 37
    ls = nixl_kv.pos_slots([1, 7], 8, pos)  # 8..15 then 56, 57
    assert rs.tolist() == [12, 13, 14, 15, 16, 17, 18, 19, 36, 37]
    assert nixl_kv.segments(rs, ls) == [(12, 8, 8), (36, 56, 2)]
    assert nixl_kv.segments(np.array([5, 6, 7]), np.array([1, 2, 9])) == [(5, 1, 2), (7, 9, 1)]
    assert nixl_kv.segments(np.array([], np.int64), np.array([], np.int64)) == []
    assert nixl_kv.local_page_slots([2, 5, 6], 9, 8, 4).tolist() == [8, 9, 10, 11, 20, 21, 22, 23]


def test_receiver_holds_a_nixl_handoff_s_bytes_until_release(tmp_path):
    """The hello frame opens the connection and lands in the receiver's directory; a nixl meta counts its declared bytes
    against the buffer until release, so a second one that does not fit waits (backpressure as for parts); a meta from
    an engine whose hello never came breaks its connection and gives back the bytes it had reserved."""
    from kiln.engine import disagg

    got = []
    rcv = disagg.Receiver("127.0.0.1:0", got.append, 1000, store_dir=str(tmp_path / "s"), signature="mine")
    hello = json.dumps({"engine": "e1", "ranks": [{"tp_rank": 0}]}).encode()
    snd = disagg.Sender(hello=lambda: ({"kind": "hello", "xfer": ""}, [hello]))
    try:
        dest = rcv.address()
        for x in ("a", "b"):
            meta = {"xfer": x, "parts": [], "signature": "mine", "transport": "nixl",
                    "nixl": {"engine": "e1", "bytes": 600, "reply": "127.0.0.1:9"}}
            snd.enqueue(dest, {"kind": "meta", "xfer": x}, [json.dumps(meta).encode()])
        snd.flush()
        for _ in range(300):
            if got:
                break
            time.sleep(0.01)
        time.sleep(0.3)
        assert [m["xfer"] for m in got] == ["a"], "b waits for room"
        with open(got[0]["nixl"]["exports"]) as f:
            assert json.load(f)["engine"] == "e1"
        assert rcv.stats.held_bytes == 600 and rcv.stats.buffer_full_events == 1 and rcv.stats.hellos == 1
        assert rcv.pop_reply("a") == "127.0.0.1:9" and rcv.pop_reply("a") is None
        rcv.release("a")
        for _ in range(300):
            if len(got) == 2:
                break
            time.sleep(0.01)
        assert [m["xfer"] for m in got] == ["a", "b"]
        rcv.release("b")
        rcv.drain()
        assert rcv.stats.held_bytes == 0 and rcv.stats.nixl == 2 and rcv.stats.nixl_bytes == 1200
        s = socket.create_connection(rcv.address().rsplit(":", 1))  # no hello on this one
        disagg.send_frame(s, {"kind": "meta", "xfer": "c"}, [json.dumps(
            {"xfer": "c", "parts": [], "transport": "nixl", "nixl": {"engine": "e2", "bytes": 1, "reply": "x:1"}}).encode()])
        time.sleep(0.3)
        # refused after its byte was reserved: the reader gives it back (it used to stay held for good)
        assert [m["xfer"] for m in got] == ["a", "b"] and rcv.stats.held_bytes == 0
        s.close()
    finally:
        snd.close()
        rcv.close()


# -- end to end ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("overlap", [False, True])
def test_gdn_over_nixl_matches_one_engine(gdn, overlap):
    """Qwen3.5: GQA KV pages (split over attention TP) and Gated DeltaNet state rows, read by the decode engine
    straight into its own pages; with overlap the decode engine reads without waiting for the device (pd_hold)."""
    ps = prompts(3, (5, 12, 33, 8, 17))
    sps = params(len(ps))
    kw = dict(overlap=True, max_num_seqs=2) if overlap else {}
    got, counts = pd(gdn, ps, sps, decode_kw=kw, prefill_kw=dict(overlap=overlap))
    compare(got, single(gdn, ps, sps), f"qwen3_5 nixl overlap={overlap}")
    assert counts["injected"] == len(ps)


def test_dsa_over_nixl_matches_one_engine(dsa):
    """GLM-5.3: MLA latent and DSA indexer caches, with prompt logprobs."""
    from kiln.engine.request import SamplingParams

    ps = prompts(4, (6, 16, 29)) + [prompts(9, (11,))[0]]
    sps = params(3) + [SamplingParams(max_new_tokens=6, ignore_eos=True, prompt_logprobs=2, logprobs=1)]
    got, _ = pd(dsa, ps, sps)
    compare(got, single(dsa, ps, sps), "glm_moe_dsa nixl")


def test_requests_that_end_on_the_prefill_side_keep_nothing(gdn):
    from kiln.engine.request import SamplingParams

    ps = prompts(6, (7, 9, 10))
    first = single(gdn, ps[1:2], [SamplingParams(max_new_tokens=1)])[0][0][0]
    sps = [SamplingParams(max_new_tokens=1, logprobs=0), SamplingParams(max_new_tokens=5, stop_token_ids=(first,)),
           SamplingParams(max_new_tokens=5, ignore_eos=True)]
    got, counts = pd(gdn, ps, sps)
    compare(got, single(gdn, ps, sps), "nixl, ends on the prefill side")
    assert counts["injected"] == 1


def test_nixl_handoff_into_a_host_path_decode_engine_is_refused_and_released(gdn, monkeypatch):
    """A decode engine without the transport cannot read a nixl handoff: the request errors loudly, and the release
    still reaches the prefill engine, which frees what it held."""
    from kiln.engine.request import SamplingParams

    monkeypatch.setenv("KILN_PD_TRANSPORT", "host")
    D = engine(gdn, pd_role="decode", pd_listen="127.0.0.1:0")
    monkeypatch.setenv("KILN_PD_TRANSPORT", "nixl")
    try:
        P = engine(gdn, pd_role="prefill")
        try:
            P.add_request(prompts(17, (9,))[0], SamplingParams(max_new_tokens=4), rid="r0", handoff=("x0", D.pd_address))
            while P.has_work():
                P.step()
            P.pd_flush()
            assert len(P._pd_pins) == 1
            meta = None
            for _ in range(500):
                got = list(D.pd_ready.queue)
                if got:
                    meta = D.pd_ready.get()
                    break
                time.sleep(0.01)
            with pytest.raises(ValueError, match="without it"):
                D.add_prefilled(meta)
            D.pd_release(meta["xfer"])
            _wait_unpinned(P)
        finally:
            P.close()
    finally:
        D.close()


def _prefill_process(path, kw, ps, sps, dest, out):
    """The prefill engine in its own process: it hands everything off, then stays until every handoff is released
    (the decode ranks read its memory)."""
    try:
        P = engine(path, pd_role="prefill", **kw)
        try:
            for i, (p, sp) in enumerate(zip(ps, sps)):
                P.add_request(p, sp, rid=f"r{i}", handoff=(f"x{i}", dest))
            end = time.monotonic() + 300
            while (P.has_work() or P._pd_pins) and time.monotonic() < end:
                if P.has_work():
                    P.step()
                    P.pd_flush()
                else:
                    time.sleep(0.01)
            P.pd_flush()
            out.put(("ok", dict(P.pd_counts), len(P._pd_pins)))
        finally:
            P.close()
    except BaseException as e:  # noqa: BLE001  reported to the test
        out.put(("error", repr(e), -1))
        raise


def _prefill_process_later(path, kw, ps, sps, addr_q, out):
    _prefill_process(path, kw, ps, sps, addr_q.get(timeout=600), out)


def pd_two_processes(path, ps, sps, prefill_kw, decode_kw, decode_env=None, monkeypatch=None):
    """The prefill engine in its own (spawned) process, the decode engine in this one (decode_env set only for it)."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q, addr = ctx.Queue(), ctx.Queue()
    proc = ctx.Process(target=_prefill_process_later, args=(path, prefill_kw, ps, sps, addr, q))
    proc.start()
    D = None
    try:
        for k, v in (decode_env or {}).items():
            monkeypatch.setenv(k, v)
        D = engine(path, pd_role="decode", pd_listen="127.0.0.1:0", **decode_kw)
        addr.put(D.pd_address)
        got = drive(None, D, ps, sps, prefill_proc=proc)
        status = q.get(timeout=300)
        assert status[0] == "ok" and status[2] == 0, status
        assert status[1].get("unpinned_released", 0) == status[1].get("handed_off", 0), status
        proc.join(timeout=120)
        assert proc.exitcode == 0
        D.pd_receiver.drain()
        assert D.pd_receiver.stats.held_bytes == 0 and D.pd_counts["refused"] == 0
        return got
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=30)
        if D is not None:
            D.close()


@pytest.mark.parametrize("layout", ["attention_tp2", "dp_attention2"])
def test_tensor_parallel_engines_over_nixl(gdn, dsa, layout, monkeypatch):
    """Both engines at tp=2: every decode rank reads from its own sender rank (split GQA heads and state rows) or the
    replicated MLA rows whole; under DP attention from the ranks of the request's group."""
    kw = {"attention_tp2": dict(tp=2), "dp_attention2": dict(tp=2, dp_attention=2)}[layout]
    for path, n in ((gdn, (6, 19, 11, 30)), (dsa, (9, 23, 14))):
        ps = prompts(10, n)
        sps = params(len(ps))
        compare(pd_two_processes(path, ps, sps, kw, kw, monkeypatch=monkeypatch), single(path, ps, sps),
                f"{os.path.basename(path)} {layout} nixl")


# -- GLM-5.3-Flash (transformers >= 5.18) ------------------------------------------------------------


def _glm5_next(tmp_path, monkeypatch, cp=False):
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from tests.test_glm5_next import build

    path = str(tmp_path)
    if cp:
        dsa_long = pytest.importorskip("kiln.models.dsa_long")
        if not hasattr(dsa_long, "cp_local_slots"):
            pytest.skip("no context parallelism in this tree")
        from kiln.models import mla

        build(path, seed=1, index_topk=16, max_position_embeddings=4096)
        monkeypatch.setenv("KILN_DSA_POOL_CACHE", "separate")
        monkeypatch.setattr(mla, "POOL_CACHE", "separate")
    else:
        build(path, seed=1, index_topk=16)
    return path


def test_glm5_next_over_nixl(tmp_path, monkeypatch):
    """KDA state, the NoPE DSA latent, the pooled indexer's keys and its pool-key cache, in process and at tp=2."""
    path = _glm5_next(tmp_path, monkeypatch)
    ps = prompts(12, (10, 37, 22), vocab=384)
    sps = params(len(ps))
    compare(pd(path, ps, sps)[0], single(path, ps, sps), "glm5_next nixl")
    compare(pd_two_processes(path, ps, sps, dict(tp=2), dict(tp=2), monkeypatch=monkeypatch),
            single(path, ps, sps, tp=2), "glm5_next tp=2 nixl")


@pytest.mark.parametrize("case", ["more_sender_ranks", "fewer_sender_ranks"])
def test_glm5_next_across_attention_tp_over_nixl(tmp_path, monkeypatch, case):
    """Regrouping: the MLA rows read whole from one sender rank, the KDA state rows of the sender ranks read into a
    registered host buffer and regrouped there."""
    path = _glm5_next(tmp_path, monkeypatch)
    ps = prompts(15, (10, 37, 22))
    sps = params(len(ps))
    a2, a1 = dict(tp=2), dict(tp=2, dp_attention=2)
    pk, dk = (a2, a1) if case == "more_sender_ranks" else (a1, a2)
    compare(pd_two_processes(path, ps, sps, pk, dk, monkeypatch=monkeypatch), single(path, ps, sps, tp=2),
            f"glm5_next {case} nixl")


def test_glm5_next_into_a_context_parallel_decode_engine_over_nixl(tmp_path, monkeypatch):
    """A non-CP prefill engine (page size 4) into a KILN_DSA_CP=1 decode engine (page size 8): each decode rank reads
    only the positions its pools hold, from position-ordered remote slots into its local ones."""
    path = _glm5_next(tmp_path, monkeypatch, cp=True)
    ps = prompts(13, (37, 70, 9, 21))
    sps = params(len(ps))
    want = single(path, ps, sps, tp=2, page_size=4)
    got = pd_two_processes(path, ps, sps, dict(tp=2, page_size=4), dict(tp=2, page_size=8),
                           decode_env={"KILN_DSA_CP": "1"}, monkeypatch=monkeypatch)
    compare(got, want, "glm5_next non-CP prefill -> CP decode, nixl")


def test_glm5_next_context_parallel_to_context_parallel_over_nixl(tmp_path, monkeypatch):
    """CP to CP of the same degree and page size: whole local pages, rank to rank."""
    path = _glm5_next(tmp_path, monkeypatch, cp=True)
    ps = prompts(14, (37, 70, 9, 21))
    sps = params(len(ps))
    want = single(path, ps, sps, tp=2, page_size=8)
    monkeypatch.setenv("KILN_DSA_CP", "1")
    got = pd_two_processes(path, ps, sps, dict(tp=2, page_size=8), dict(tp=2, page_size=8), monkeypatch=monkeypatch)
    compare(got, want, "glm5_next CP prefill -> CP decode, nixl")


def test_glm5_next_latency_prefill_into_a_context_parallel_decode_engine_over_nixl(tmp_path, monkeypatch):
    """The latency deployment's shape: attention TP 4 prefill into attention TP 2 + CP 2 decode, regrouped state rows
    and CP-selected MLA rows."""
    path = _glm5_next(tmp_path, monkeypatch, cp=True)
    ps = prompts(16, (37, 70, 9, 21))
    sps = params(len(ps))
    want = single(path, ps, sps, tp=4, page_size=4)
    got = pd_two_processes(path, ps, sps, dict(tp=4, page_size=4), dict(tp=4, dp_attention=2, page_size=8),
                           decode_env={"KILN_DSA_CP": "1"}, monkeypatch=monkeypatch)
    compare(got, want, "glm5_next attention TP 4 prefill -> attention TP 2 CP decode, nixl")


def test_glm5_next_into_context_parallel_row_groups_over_nixl(tmp_path, monkeypatch):
    """The R8LK decode's shape at CPU scale: a non-CP prefill (tp=4) into a decode whose CP degree is below its
    attention TP (KILN_DSA_CP_DEGREE=2 at attention TP 4: two row groups of 2), so ranks 2 and 3 read the positions of
    CP index 0 and 1 again (ModelRunner._pd_cp_select, a % cp)."""
    path = _glm5_next(tmp_path, monkeypatch, cp=True)
    ps = prompts(17, (37, 70, 9, 21))
    sps = params(len(ps))
    want = single(path, ps, sps, tp=4, page_size=4)
    got = pd_two_processes(path, ps, sps, dict(tp=4, page_size=4), dict(tp=4, page_size=8),
                           decode_env={"KILN_DSA_CP": "1", "KILN_DSA_CP_DEGREE": "2"}, monkeypatch=monkeypatch)
    compare(got, want, "glm5_next non-CP prefill -> CP row-group decode, nixl")
