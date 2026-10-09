"""Prefill / decode disaggregation (engine/disagg.py) on CPU: a prefill engine hands each request off to a decode
engine over a real TCP connection after its first token, and the decode engine finishes it.

The gate: for the same prompts and seeds, the tokens of disaggregated serving equal one engine's exactly and
their chosen-token logprobs agree within fp32 reduction-order noise (LOGPROB_TOL, as tests/test_dp_attention.py
and tests/test_mixed_batch.py). Models: Qwen3.5 (Gated DeltaNet recurrent state + GQA KV, whose heads split
over attention TP), GLM-5.3 (MLA + DSA indexer: replicated latent caches, sliced over the attention group),
KDA only (state, no KV), and, with transformers 5.18, GLM-5.3-Flash (KDA + DSA with the pooled indexer and its
pool-key cache). Cases: greedy and seeded sampling, logprobs and prompt logprobs, chunked prefill, prompts
ending inside a page and on its boundary, requests finishing on the prefill side (EOS, max_tokens 1, a stop
string), the prefill engine in its own process, both engines at tp=2 (attention TP 2, and DP attention 2), and
a decode engine admitting handoffs while it decodes others.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest
import torch

from tests.test_linear_attn import build_kda, build_qwen3_5
from tests.test_mla import build as build_mla

LOGPROB_TOL = 1e-4


def engine(path, **kw):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    base = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256, max_num_seqs=4,
                max_model_len=256, max_prefill_tokens=8)
    base.update(kw)
    return LLMEngine(EngineConfig(**base))


def prompts(seed, lengths, vocab=384):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(3, vocab, (n,), generator=g).tolist() for n in lengths]


def params(n, new_tokens=8, **kw):
    """Greedy with logprobs for even requests, seeded sampling for odd ones."""
    from kiln.engine.request import SamplingParams

    out = []
    for i in range(n):
        if i % 2:
            out.append(SamplingParams(max_new_tokens=new_tokens, ignore_eos=True, temperature=0.8, top_k=20,
                                      seed=100 + i, logprobs=2, **kw))
        else:
            out.append(SamplingParams(max_new_tokens=new_tokens, ignore_eos=True, logprobs=2, **kw))
    return out


def single(path, ps, sps, **kw):
    eng = engine(path, **kw)
    try:
        reqs = eng.generate(ps, sps)
        return [(r.output_ids, [x[0] for x in r.logprobs], r.finish_reason, dict(r.prompt_logprobs)) for r in reqs]
    finally:
        eng.close()


def drive(P, D, ps, sps, interleave=False, prefill_proc=None, timeout=300.0, tag=""):
    """Hand every prompt from P (a prefill engine in this process, or None when prefill_proc runs it elsewhere)
    to D and run both until every request ends. interleave: step the two engines in turns, so handoffs arrive
    while D decodes others."""
    n = len(ps)
    if P is not None:
        for i, (p, sp) in enumerate(zip(ps, sps)):
            P.add_request(p, sp, rid=f"{tag}r{i}", handoff=(f"{tag}x{i}", D.pd_address))
    reqs: dict[str, object] = {}
    deadline = time.monotonic() + timeout
    while True:
        if P is not None and P.has_work():
            P.step()
            if not interleave:
                while P.has_work() and time.monotonic() < deadline:
                    P.step()
            P.pd_flush()
        for r in D.pd_poll():
            reqs[r.rid] = r
        done = {m["rid"]: m for m in D.pd_done if m["rid"].startswith(f"{tag}r")}
        if D.has_work():
            D.step()
        elif len(reqs) + len(done) == n and all(r.finish_reason is not None for r in reqs.values()):
            break
        else:
            if prefill_proc is not None and prefill_proc.exitcode not in (None, 0):
                raise RuntimeError(f"the prefill process failed (exit {prefill_proc.exitcode})")
            time.sleep(0.01)
        if time.monotonic() > deadline:
            raise TimeoutError(f"{len(reqs)} handed off, {len(done)} done on the prefill side, of {n}")
    out = []
    for i in range(n):
        r = reqs.get(f"{tag}r{i}")
        if r is None:
            m = done[f"{tag}r{i}"]
            out.append(([m["token"]] if m["token"] is not None else [], [x[0] for x in m["logprobs"] or []],
                        m["done"], {int(k): tuple(v) for k, v in (m.get("prompt_logprobs") or {}).items()}))
        else:
            out.append((r.output_ids, [x[0] for x in r.logprobs], r.finish_reason, dict(r.prompt_logprobs)))
    return out


def compare(got, want, what=""):
    err = 0.0
    for (ids, lps, fin, plp), (wids, wlps, wfin, wplp) in zip(got, want):
        assert ids == wids, (what, ids, wids)
        assert fin == wfin, (what, fin, wfin)
        err = max([err] + [abs(a - b) for a, b in zip(lps, wlps)])
        assert sorted(plp) == sorted(wplp), what
        err = max([err] + [abs(plp[q][0] - wplp[q][0]) for q in plp])
    print(f"disaggregated vs one engine {what}: tokens equal, max |dlogprob| {err:.2e}")
    assert err < LOGPROB_TOL, (what, err)


def pd(path, ps, sps, interleave=False, prefill_kw=None, decode_kw=None, **kw):
    D = engine(path, pd_role="decode", pd_listen="127.0.0.1:0", **{**kw, **(decode_kw or {})})
    try:
        P = engine(path, pd_role="prefill", **{**kw, **(prefill_kw or {})})
        try:
            got = drive(P, D, ps, sps, interleave)
            assert P.pd_counts["handed_off"] + P.pd_counts["done_at_prefill"] == len(ps)
        finally:
            P.close()
        D.pd_receiver.drain()
        assert D.pd_receiver.stats.held_bytes == 0, "every received part is deleted after its copy"
        assert not os.listdir(D.pd_receiver.dir)
        return got, D.pd_counts
    finally:
        D.close()


@pytest.fixture(scope="module")
def gdn(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("pd_qwen3_5"))
    build_qwen3_5(path)
    return path


@pytest.fixture(scope="module")
def dsa(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("pd_glm_dsa"))
    build_mla("glm_moe_dsa", path)
    return path


# -- transport --------------------------------------------------------------------------------------


def test_frames_roundtrip_through_the_receiver(tmp_path):
    from kiln.engine import disagg

    got = []
    rcv = disagg.Receiver("127.0.0.1:0", got.append, 1 << 20, store_dir=str(tmp_path / "s"), signature="sig")
    snd = disagg.Sender()
    try:
        a = torch.randn(5, 3).to(torch.bfloat16)
        b = torch.randn(4, 2).clamp(-200, 200).to(torch.float8_e4m3fn)
        c = torch.randn(2, 2, 2)
        dest = rcv.address()
        for part, named in (("s0", [("x", a), ("y", c)]), ("r0", [("z", b), ("e", torch.empty(0, 7))])):
            table, bufs = disagg.pack_arrays(named)
            snd.enqueue(dest, {"kind": "part", "xfer": "t1", "part": part, "arrays": table}, bufs)
        snd.enqueue(dest, {"kind": "meta", "xfer": "t1"},
                    [json.dumps({"xfer": "t1", "parts": ["s0", "r0"], "signature": "sig"}).encode()])
        snd.flush()
        for _ in range(500):
            if got:
                break
            time.sleep(0.01)
        assert len(got) == 1 and "error" not in got[0]
        parts = disagg.read_parts(got[0], ["s0", "r0"])
        assert torch.equal(parts["s0"]["x"].view(torch.int16), a.view(torch.int16))
        assert torch.equal(parts["s0"]["y"], c)
        assert torch.equal(parts["r0"]["z"].view(torch.uint8), b.view(torch.uint8))
        assert parts["r0"]["e"].shape == (0, 7)
        assert rcv.stats.held_bytes > 0
        rcv.release("t1")
        rcv.drain()
        assert rcv.stats.held_bytes == 0 and not os.listdir(rcv.dir)
    finally:
        snd.close()
        rcv.close()


@pytest.mark.parametrize("how", ["close", "reset"])
def test_a_part_closed_mid_frame_leaves_no_file(tmp_path, how):
    """A part frame is received straight into its file (disagg._recv_file: sized, mapped, filled by recv_into); a peer
    that closes (recv returns 0) or resets (recv_into raises) inside the frame leaves neither a handoff nor a
    half-written file behind."""
    import socket
    import struct

    from kiln.engine import disagg

    got = []
    rcv = disagg.Receiver("127.0.0.1:0", got.append, 1 << 20, store_dir=str(tmp_path / "s"))
    try:
        table, bufs = disagg.pack_arrays([("x", torch.arange(1000, dtype=torch.uint8))])
        host, port = rcv.address().rsplit(":", 1)
        sock = socket.create_connection((host, int(port)))
        h = json.dumps({"kind": "part", "xfer": "t9", "part": "s0", "arrays": table}).encode()
        sock.sendall(disagg._HDR.pack(disagg.MAGIC, len(h), 1000) + h + bytes(bufs[0])[:100])
        if how == "reset":
            time.sleep(0.2)  # the receiver is inside recv_into when the RST lands
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        sock.close()
        # Wait for the receiver to finish the connection, not for an empty directory: its reader closes its side
        # (Receiver._read's finally) only after _recv_file has returned or raised, i.e. after the unlink, while the
        # directory is ALSO empty before the reader has opened the part file. Waiting on the directory let the close
        # case pass that window and then see the file appear (3 of 40 in-process repeats in the Python 3.13 image).
        def finished():
            return bool(rcv._conns) and all(c.fileno() == -1 for c in rcv._conns)
        for _ in range(250):
            if finished():
                break
            time.sleep(0.02)
        assert finished(), "the receiver did not finish the connection within 5 s"
        assert not got and not os.listdir(rcv.dir), os.listdir(rcv.dir)
    finally:
        rcv.close()


@pytest.mark.parametrize("how", ["close", "reset", "close-in-memory", "nixl-before-hello"])
def test_a_frame_cut_mid_way_gives_its_bytes_back(tmp_path, how):
    """The bytes Receiver._room reserves for a frame come back, exactly once, when the frame never completes: a part
    whose peer closes (EOF) or resets inside it, on disk or in memory, and a NIXL meta refused after its bytes were
    reserved (no hello frame from its engine). So repeated broken handoffs do not shrink the receive buffer: after five
    of them a handoff of the whole buffer still goes through without waiting, and its release() brings the held bytes
    back to zero, not below. (Before the fix each broken frame kept its bytes held for good.)"""
    import socket
    import struct

    from kiln.engine import disagg

    budget = 4000
    got = []
    rcv = disagg.Receiver("127.0.0.1:0", got.append, budget, store_dir=str(tmp_path / "s"),
                          in_memory=how == "close-in-memory")
    snd = disagg.Sender()
    try:
        host, port = rcv.address().rsplit(":", 1)
        table, bufs = disagg.pack_arrays([("x", torch.arange(1000, dtype=torch.uint8))])
        for i in range(5):
            sock = socket.create_connection((host, int(port)))
            if how == "nixl-before-hello":
                meta = json.dumps({"xfer": f"cut{i}", "parts": [], "transport": "nixl",
                                   "nixl": {"bytes": 1000, "engine": "e0", "reply": "127.0.0.1:1"}}).encode()
                h = json.dumps({"kind": "meta", "xfer": f"cut{i}"}).encode()
                sock.sendall(disagg._HDR.pack(disagg.MAGIC, len(h), len(meta)) + h + meta)
            else:
                h = json.dumps({"kind": "part", "xfer": f"cut{i}", "part": "s0", "arrays": table}).encode()
                sock.sendall(disagg._HDR.pack(disagg.MAGIC, len(h), 1000) + h + bytes(bufs[0])[:100])
            time.sleep(0.2)  # the reader has reserved the frame's bytes and waits inside it
            if how == "reset":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            sock.close()
            for _ in range(250):  # the reader closes its side last, after giving the bytes back
                if len(rcv._conns) == i + 1 and all(c.fileno() == -1 for c in rcv._conns):
                    break
                time.sleep(0.02)
            assert len(rcv._conns) == i + 1 and all(c.fileno() == -1 for c in rcv._conns)
            assert rcv.stats.held_bytes == 0, (i, rcv.stats.held_bytes)
        assert not got
        table, bufs = disagg.pack_arrays([("v", torch.zeros(budget // 4))])  # exactly the whole buffer
        snd.enqueue(rcv.address(), {"kind": "part", "xfer": "whole", "part": "s0", "arrays": table}, bufs)
        snd.enqueue(rcv.address(), {"kind": "meta", "xfer": "whole"},
                    [json.dumps({"xfer": "whole", "parts": ["s0"]}).encode()])
        snd.flush()
        for _ in range(300):
            if got:
                break
            time.sleep(0.01)
        assert [m["xfer"] for m in got] == ["whole"] and rcv.stats.buffer_full_events == 0
        assert rcv.stats.held_bytes == budget
        rcv.release("whole")
        rcv.drain()
        assert rcv.stats.held_bytes == 0
    finally:
        snd.close()
        rcv.close()


def test_in_memory_receiver(tmp_path):
    """Receiver(in_memory=True): parts stay in memory (no file), read_parts reads them, release frees their bytes."""
    from kiln.engine import disagg

    got = []
    rcv = disagg.Receiver("127.0.0.1:0", got.append, 1 << 20, store_dir=str(tmp_path / "m"), in_memory=True)
    snd = disagg.Sender()
    try:
        h = torch.randn(4, 8).to(torch.bfloat16)
        table, bufs = disagg.pack_arrays([("h", h)])
        snd.enqueue(rcv.address(), {"kind": "part", "xfer": "r-c0", "part": "h", "arrays": table}, bufs)
        snd.enqueue(rcv.address(), {"kind": "meta", "xfer": "r-c0"}, [json.dumps({"xfer": "r-c0", "parts": ["h"]}).encode()])
        snd.flush()
        for _ in range(500):
            if got:
                break
            time.sleep(0.01)
        assert not os.listdir(rcv.dir) and "buf" in got[0]["parts"]["h"]
        assert torch.equal(disagg.read_parts(got[0], ["h"])["h"]["h"].view(torch.int16), h.view(torch.int16))
        rcv.release("r-c0")
        rcv.drain()
        assert rcv.stats.held_bytes == 0
    finally:
        snd.close()
        rcv.close()


def test_receiver_buffer_backpressure_and_refusals(tmp_path):
    """A part that does not fit the receive buffer waits for room (counted, never dropped); a handoff from an
    engine with another layout is refused; a part larger than the whole buffer is an error."""
    from kiln.engine import disagg

    got = []
    rcv = disagg.Receiver("127.0.0.1:0", got.append, 1000, store_dir=str(tmp_path / "s"), signature="mine")
    snd = disagg.Sender()
    try:
        dest = rcv.address()
        for x in ("a", "b"):
            table, bufs = disagg.pack_arrays([("v", torch.zeros(150))])  # 600 bytes
            snd.enqueue(dest, {"kind": "part", "xfer": x, "part": "s0", "arrays": table}, bufs)
            snd.enqueue(dest, {"kind": "meta", "xfer": x},
                        [json.dumps({"xfer": x, "parts": ["s0"], "signature": "mine"}).encode()])
        snd.flush()
        for _ in range(300):
            if got:
                break
            time.sleep(0.01)
        time.sleep(0.3)
        assert [m["xfer"] for m in got] == ["a"], "b waits for room"
        assert rcv.stats.buffer_full_events == 1
        rcv.release("a")
        for _ in range(300):
            if len(got) == 2:
                break
            time.sleep(0.01)
        assert [m["xfer"] for m in got] == ["a", "b"] and rcv.stats.buffer_full_seconds > 0.2
        rcv.release("b")
        snd.enqueue(dest, {"kind": "meta", "xfer": "c"},
                    [json.dumps({"xfer": "c", "parts": [], "signature": "other"}).encode()])
        snd.flush()
        for _ in range(300):
            if len(got) == 3:
                break
            time.sleep(0.01)
        assert "differs from this engine's" in got[2]["error"] and rcv.stats.refused == 1
        with pytest.raises(RuntimeError, match="exceeds the whole receive buffer"):
            rcv._room(2000, "big")
    finally:
        snd.close()
        rcv.close()


_KEYS = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
from kiln.engine.hicache import HostTier
from kiln.engine import disagg
ids = list(range(1000, 1257))
print(json.dumps([HostTier(None, 16, 4).hashes(ids, 16), disagg.signature({"caches": [["k0", True, [1, 576], "fp8"]]})]))
"""


def test_cache_keys_do_not_depend_on_the_process():
    """The host tier's page keys and the handoff layout signature are equal in two processes with different
    hash seeds (LMCache keys prompts with Python's hash() and misses every lookup across processes without
    PYTHONHASHSEED=0): page keys hash tuples of ints, which CPython does not salt, and the signature is a
    sha256."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outs = []
    for seed in ("1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outs.append(subprocess.run([sys.executable, "-c", _KEYS, root], env=env, capture_output=True, text=True,
                                   check=True, timeout=120).stdout)
    assert outs[0] == outs[1] and json.loads(outs[0])[0]


def test_regroup_rebuilds_a_rank_s_heads_across_attention_tp():
    """disagg.regroup: a receiver rank's state row from the sender ranks holding its heads, component by component,
    for a sender with more ranks (concatenate) and with fewer (take a share), equal to slicing the whole-model row."""
    from kiln.engine import disagg

    g = torch.Generator().manual_seed(0)
    comps = [16, 16, 32]  # a conv state's q, k, v channels over the whole model
    full = torch.randn(3, sum(comps), generator=g)

    def shard(A, a):  # attention rank a of A: its contiguous share of every component (models/loader.py _part)
        out, off = [], 0
        for c in comps:
            out.append(full[:, off + c * a // A : off + c * (a + 1) // A])
            off += c
        return torch.cat(out, dim=1)

    for As, A in ((8, 2), (4, 4), (2, 8), (1, 4), (4, 1)):
        for a in range(A):
            src = disagg.sender_ranks(As, A, a)
            got = disagg.regroup({b: shard(As, b) for b in src}, 1, [c // A for c in comps], As, A, a)
            assert torch.equal(got, shard(A, a)), (As, A, a)
    rec = torch.randn(8, 2, 3, generator=g)  # a recurrent state: heads on axis 0
    for As, A in ((8, 2), (2, 8)):
        for a in range(A):
            src = disagg.sender_ranks(As, A, a)
            got = disagg.regroup({b: rec[8 * b // As : 8 * (b + 1) // As] for b in src}, 0, [8 // A], As, A, a)
            assert torch.equal(got, rec[8 * a // A : 8 * (a + 1) // A])
    assert not disagg.regroup_ok(3, 2)
    with pytest.raises(ValueError, match="cannot be split"):
        disagg.sender_ranks(3, 2, 0)


def test_slot_runs():
    from kiln.engine import disagg

    assert disagg.slot_runs([5, 6, 2], 0, 10, 4) == [(20, 8), (8, 2)]
    assert disagg.slot_runs([5], 0, 0, 4) == []
    assert disagg.slot_runs([3, 4], 1, 7, 4) == [(13, 6)]
    assert [disagg.rep_slice(10, a, 4) for a in range(4)] == [(0, 2), (2, 5), (5, 7), (7, 10)]


# -- engines ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("interleave,overlap", [(False, False), (True, False), (True, True)])
def test_gdn_matches_one_engine(gdn, interleave, overlap):
    """Qwen3.5: recurrent state rows plus GQA KV pages; prompts across page boundaries, chunked prefill. With
    overlap the decode engine holds freed pages and state rows for a step and copies handoffs without waiting for
    the device (PagePool.hold); max_num_seqs 2 makes admissions wait for those slots."""
    ps = prompts(3, (5, 12, 33, 8, 17))
    sps = params(len(ps))
    want = single(gdn, ps, sps)
    kw = dict(overlap=True, max_num_seqs=2) if overlap else {}
    got, counts = pd(gdn, ps, sps, interleave=interleave, decode_kw=kw, prefill_kw=dict(overlap=overlap))
    compare(got, want, f"qwen3_5 interleave={interleave} overlap={overlap}")
    assert counts["injected"] == len(ps)


def test_dsa_matches_one_engine(dsa):
    """GLM-5.3 (MLA latent and DSA indexer keys), with prompt logprobs."""
    from kiln.engine.request import SamplingParams

    ps = prompts(4, (6, 16, 29))
    sps = params(len(ps)) + [SamplingParams(max_new_tokens=6, ignore_eos=True, prompt_logprobs=2, logprobs=1)]
    ps.append(prompts(9, (11,))[0])
    want = single(dsa, ps, sps)
    got, _ = pd(dsa, ps, sps)
    compare(got, want, "glm_moe_dsa")
    assert got[-1][3], "prompt logprobs came back"


def test_kda_state_only(tmp_path):
    build_kda(str(tmp_path))
    ps = prompts(5, (9, 21))
    sps = params(len(ps))
    compare(pd(str(tmp_path), ps, sps)[0], single(str(tmp_path), ps, sps), "kda")


def test_idle_decode_engine_releases_what_it_holds(gdn):
    """With overlap a decode engine holds freed pages, state rows and slots until the step in flight is read back
    (pd_hold), and pages the prefix cache evicts go there too. An engine that went idle has nothing in flight and
    never reads a step back again, so it must release them itself. Here the first batch leaves 28 of 35 pages in the
    prefix cache and 7 free; the next handoff needs 8, eviction moved the cached pages into the hold, and the idle
    engine used to wait forever with its thread spinning in _admit_prefilled (seen on trn2: 30 handoffs queued, KV
    99% used, nothing running)."""
    D = engine(gdn, pd_role="decode", pd_listen="127.0.0.1:0", overlap=True, num_pages=36)
    try:
        P = engine(gdn, pd_role="prefill")
        try:
            a, b = prompts(19, (23,) * 4), prompts(20, (31,))
            compare(drive(P, D, a, params(4), timeout=120, tag="a"), single(gdn, a, params(4)), "first batch")
            assert D.pd_hold and D.pools[0].num_free < 8 <= D.pools[0].num_free + D.scheduler.radix.evictable_pages
            compare(drive(P, D, b, params(1), timeout=60, tag="b"), single(gdn, b, params(1)), "into the idle engine")
        finally:
            P.close()
    finally:
        D.close()


def test_requests_that_end_on_the_prefill_side(gdn):
    """max_tokens 1, or the first token a stop token: the prefill engine finishes the request and tells the
    decode engine, which receives no state."""
    from kiln.engine.request import SamplingParams

    ps = prompts(6, (7, 9, 10))
    first = single(gdn, ps[1:2], [SamplingParams(max_new_tokens=1)])[0][0][0]
    sps = [SamplingParams(max_new_tokens=1, logprobs=0),
           SamplingParams(max_new_tokens=5, stop_token_ids=(first,)),
           SamplingParams(max_new_tokens=5, ignore_eos=True)]
    want = single(gdn, ps, sps)
    got, counts = pd(gdn, ps, sps)
    compare(got, want, "end on the prefill side")
    assert [g[2] for g in got] == ["length", "stop", "length"]
    assert counts["injected"] == 1


def test_refusals(gdn):
    from kiln.engine.request import SamplingParams

    P = engine(gdn, pd_role="prefill")
    try:
        with pytest.raises(ValueError, match="serves handoffs only"):
            P.add_request([5, 6, 7], SamplingParams())
        with pytest.raises(ValueError, match="thinking token budget cannot be disaggregated"):
            P.add_request([5, 6, 7], SamplingParams(thinking_token_budget=4), handoff=("x", "127.0.0.1:1"))
    finally:
        P.close()
    D = engine(gdn, pd_role="decode")
    try:
        with pytest.raises(ValueError, match="takes handed-off requests only"):
            D.add_request([5, 6, 7], SamplingParams())
    finally:
        D.close()
    with pytest.raises(NotImplementedError, match="speculative"):
        engine(gdn, pd_role="decode", spec_method="ngram")


def test_decode_engine_with_bypass_serves_whole_requests(gdn):
    """pd_bypass_prefill: the router sends short prompts straight to the decode engine, which prefills them."""
    from kiln.engine.request import SamplingParams

    ps = prompts(7, (6, 10))
    sp = SamplingParams(max_new_tokens=5, ignore_eos=True)
    want = [w[0] for w in single(gdn, ps, [sp, sp])]
    D = engine(gdn, pd_role="decode", pd_bypass_prefill=True)
    try:
        assert [r.output_ids for r in D.generate(ps, sp)] == want
    finally:
        D.close()


def _prefill_process(path, kw, ps, sps, dest, out):
    try:
        P = engine(path, pd_role="prefill", **kw)
        try:
            for i, (p, sp) in enumerate(zip(ps, sps)):
                P.add_request(p, sp, rid=f"r{i}", handoff=(f"x{i}", dest))
            while P.has_work():
                P.step()
            P.pd_flush()
            out.put(("ok", dict(P.pd_counts)))
        finally:
            P.close()
    except BaseException as e:  # noqa: BLE001  reported to the test
        out.put(("error", repr(e)))
        raise


def pd_two_processes(path, ps, sps, prefill_kw, decode_kw, **kw):
    """The prefill engine in its own (spawned) process, the decode engine in this one."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    D = engine(path, pd_role="decode", pd_listen="127.0.0.1:0", **{**kw, **decode_kw})
    q = ctx.Queue()
    proc = ctx.Process(target=_prefill_process, args=(path, {**kw, **prefill_kw}, ps, sps, D.pd_address, q))
    try:
        proc.start()
        got = drive(None, D, ps, sps, prefill_proc=proc)
        status = q.get(timeout=120)
        assert status[0] == "ok", status
        proc.join(timeout=120)
        assert proc.exitcode == 0
        D.pd_receiver.drain()
        assert D.pd_receiver.stats.held_bytes == 0
        return got
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=30)
        D.close()


def test_prefill_engine_in_another_process(gdn):
    ps = prompts(8, (7, 26, 13))
    sps = params(len(ps))
    compare(pd_two_processes(gdn, ps, sps, {}, {}), single(gdn, ps, sps), "two processes")


@pytest.mark.parametrize("layout", ["attention_tp2", "dp_attention2"])
def test_tensor_parallel_engines(gdn, dsa, layout):
    """Both engines at tp=2, each rank sending and receiving its own shard: attention TP 2 (GQA KV heads and
    recurrent state split, MLA caches replicated and sent as halves) and DP attention 2 (each group holds its own
    requests; the decode engine places a handoff in a group of its own choosing)."""
    kw = {"attention_tp2": dict(tp=2), "dp_attention2": dict(tp=2, dp_attention=2)}[layout]
    for path, n in ((gdn, (6, 19, 11, 30)), (dsa, (9, 23, 14))):
        ps = prompts(10, n)
        sps = params(len(ps))
        want = single(path, ps, sps)
        compare(pd_two_processes(path, ps, sps, kw, kw), want, f"{os.path.basename(path)} {layout}")


def test_layouts_must_match(gdn):
    """A decode engine refuses a handoff from a prefill engine with another layout (here attention TP), loudly."""
    from kiln.engine.request import SamplingParams

    ps = prompts(11, (9,))
    D = engine(gdn, pd_role="decode", pd_listen="127.0.0.1:0")
    q = None
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_prefill_process, args=(gdn, dict(tp=2), ps, [SamplingParams(max_new_tokens=4)],
                                                      D.pd_address, q))
    try:
        proc.start()
        assert q.get(timeout=300)[0] == "ok"
        proc.join(timeout=120)
        for _ in range(500):
            D.pd_poll()
            if D.pd_done:
                break
            time.sleep(0.01)
        assert "differs from this engine's" in D.pd_done[0]["error"]
        D.pd_receiver.drain()
        assert D.pd_counts["refused"] == 1 and D.pd_receiver.stats.held_bytes == 0
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=30)
        D.close()


# -- GLM-5.3-Flash (transformers >= 5.18) ------------------------------------------------------------


def test_glm5_next_matches_one_engine(tmp_path):
    """KDA state, the NoPE DSA latent, the pooled indexer's keys and its pool-key cache (separate under FP8-free
    fp32 KV: KILN_DSA_POOL_CACHE=separate), mHC and the shared-expert MoE."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from tests.test_glm5_next import build

    build(str(tmp_path), seed=1, index_topk=16)
    ps = prompts(12, (10, 37, 22), vocab=384)
    sps = params(len(ps))
    want = single(str(tmp_path), ps, sps)
    compare(pd(str(tmp_path), ps, sps)[0], want, "glm5_next")
    compare(pd_two_processes(str(tmp_path), ps, sps, dict(tp=2), dict(tp=2)), single(str(tmp_path), ps, sps, tp=2),
            "glm5_next tp=2")


def _prefill_process_later(path, kw, ps, sps, addr_q, out):
    """_prefill_process, its decode engine's address arriving on addr_q (started before that engine exists, so it
    does not inherit an environment set for the decode side)."""
    _prefill_process(path, kw, ps, sps, addr_q.get(timeout=600), out)


def test_glm5_next_into_a_context_parallel_decode_engine(tmp_path, monkeypatch):
    """A non-CP prefill engine (tp=2, page size 4, every rank holding all of a request's DSA rows) hands off to a decode
    engine under KILN_DSA_CP=1 (tp=2, page size 8: each rank holds the pools m * 2 + its rank, 4 local slots per
    page): each decode rank picks its pools' positions out of the position-ordered rows. Same tokens as one non-CP
    engine at tp=2, logprobs within LOGPROB_TOL. Needs models/dsa_long.py (feat/long-context) and transformers 5.18."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    dsa_long = pytest.importorskip("kiln.models.dsa_long")
    if not hasattr(dsa_long, "cp_local_slots"):
        pytest.skip("no context parallelism in this tree")
    import multiprocessing as mp

    from kiln.models import mla
    from tests.test_glm5_next import build

    path = str(tmp_path)
    build(path, seed=1, index_topk=16, max_position_embeddings=4096)
    monkeypatch.setenv("KILN_DSA_POOL_CACHE", "separate")
    monkeypatch.setattr(mla, "POOL_CACHE", "separate")
    ps = prompts(13, (37, 70, 9, 21))
    sps = params(len(ps))
    want = single(path, ps, sps, tp=2, page_size=4)
    ctx = mp.get_context("spawn")
    q, addr = ctx.Queue(), ctx.Queue()
    proc = ctx.Process(target=_prefill_process_later, args=(path, dict(tp=2, page_size=4), ps, sps, addr, q))
    proc.start()
    D = None
    try:
        monkeypatch.setenv("KILN_DSA_CP", "1")
        D = engine(path, pd_role="decode", pd_listen="127.0.0.1:0", tp=2, page_size=8)
        assert D.model.cp == 2 and D.runner.lps == 4
        addr.put(D.pd_address)
        got = drive(None, D, ps, sps, prefill_proc=proc)
        assert q.get(timeout=120)[0] == "ok"
        proc.join(timeout=120)
        compare(got, want, "glm5_next non-CP prefill -> CP decode")
        D.pd_receiver.drain()
        assert D.pd_receiver.stats.held_bytes == 0
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=30)
        if D is not None:
            D.close()


def test_glm5_next_into_context_parallel_row_groups(tmp_path, monkeypatch):
    """A non-CP prefill engine (tp=4, page size 4) into a decode engine whose CP degree is below its attention TP
    (KILN_DSA_CP=1 KILN_DSA_CP_DEGREE=2, tp=4 at DP attention 1: two row groups of 2 consecutive ranks, page size 8, 4
    local slots per page): each decode rank picks the positions of pools m * 2 + (its attention rank mod 2). Selecting
    by the attention rank itself left ranks 2 and 3 with no positions (IndexError in _pd_slots; the 8K pipeline into the
    R8LK decode, CP 8 at attention TP 32). Same tokens as one non-CP engine, logprobs within LOGPROB_TOL."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    dsa_long = pytest.importorskip("kiln.models.dsa_long")
    if not hasattr(dsa_long, "cp_degree_env"):
        pytest.skip("no context-parallel row groups in this tree")
    import multiprocessing as mp

    from kiln.models import mla
    from tests.test_glm5_next import build

    path = str(tmp_path)
    build(path, seed=1, index_topk=16, max_position_embeddings=4096)
    monkeypatch.setenv("KILN_DSA_POOL_CACHE", "separate")
    monkeypatch.setattr(mla, "POOL_CACHE", "separate")
    ps = prompts(13, (37, 70, 9, 21))
    sps = params(len(ps))
    want = single(path, ps, sps, tp=4, page_size=4)
    ctx = mp.get_context("spawn")
    q, addr = ctx.Queue(), ctx.Queue()
    proc = ctx.Process(target=_prefill_process_later, args=(path, dict(tp=4, page_size=4), ps, sps, addr, q))
    proc.start()
    D = None
    try:
        monkeypatch.setenv("KILN_DSA_CP", "1")
        monkeypatch.setenv("KILN_DSA_CP_DEGREE", "2")
        D = engine(path, pd_role="decode", pd_listen="127.0.0.1:0", tp=4, page_size=8)
        assert D.model.cp == 2 and D.model.cp_rows == 2 and D.runner.lps == 4
        addr.put(D.pd_address)
        got = drive(None, D, ps, sps, prefill_proc=proc)
        assert q.get(timeout=120)[0] == "ok"
        proc.join(timeout=120)
        compare(got, want, "glm5_next non-CP prefill -> CP row-group decode")
        D.pd_receiver.drain()
        assert D.pd_receiver.stats.held_bytes == 0
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=30)
        if D is not None:
            D.close()


def test_expert_layout_follows_the_role(tmp_path, monkeypatch):
    """The automatic expert-parallel default (models/decoder.py, KILN_MOE_EP unset) of a disaggregated engine follows its
    role: a prefill engine takes EP whatever its max_num_seqs, a decode engine TP experts; an engine without a role keeps
    the decode-row rule (EP off below ep_auto_min_decode_rows() rows per group), and KILN_MOE_EP still decides. CPU
    counts as an EP platform (moe_ep_enabled), GLM-5.3-Flash is the EP family."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from tests.test_glm5_next import build

    path = str(tmp_path)
    build(path, seed=1, index_topk=16)
    monkeypatch.delenv("KILN_MOE_EP", raising=False)

    def ep(**kw):
        eng = engine(path, tp=2, **kw)
        try:
            return eng.model.moe_ep
        finally:
            eng.close()

    assert ep(max_num_seqs=1) is False  # no role: 1 decode row per group is below the rule
    assert ep(max_num_seqs=8) is True
    assert ep(pd_role="prefill", max_num_seqs=1) is True
    assert ep(pd_role="decode", max_num_seqs=8) is False
    monkeypatch.setenv("KILN_MOE_EP", "1")
    assert ep(pd_role="decode", max_num_seqs=8) is True
    monkeypatch.setenv("KILN_MOE_EP", "0")
    assert ep(pd_role="prefill", max_num_seqs=8) is False


def test_glm5_next_context_parallel_to_context_parallel(tmp_path, monkeypatch):
    """A CP prefill engine into a CP decode engine of the same degree and page size (KILN_DSA_CP=1 on both, tp=2, page
    size 8: W1M's trn1 CP prefill -> trn2 CP decode layout): each rank ships its whole local pages to the same rank, no
    position map. Same tokens as one non-CP engine at tp=2. A CP decode engine with another page size refuses the
    handoff loudly."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    dsa_long = pytest.importorskip("kiln.models.dsa_long")
    if not hasattr(dsa_long, "cp_local_slots"):
        pytest.skip("no context parallelism in this tree")
    from kiln.models import mla
    from tests.test_glm5_next import build

    path = str(tmp_path)
    build(path, seed=1, index_topk=16, max_position_embeddings=4096)
    monkeypatch.setenv("KILN_DSA_POOL_CACHE", "separate")
    monkeypatch.setattr(mla, "POOL_CACHE", "separate")
    ps = prompts(14, (37, 70, 9, 21))
    sps = params(len(ps))
    want = single(path, ps, sps, tp=2, page_size=8)
    monkeypatch.setenv("KILN_DSA_CP", "1")
    got = pd_two_processes(path, ps, sps, dict(tp=2, page_size=8), dict(tp=2, page_size=8))
    compare(got, want, "glm5_next CP prefill -> CP decode")


@pytest.mark.parametrize("case", ["more_sender_ranks", "fewer_sender_ranks"])
def test_glm5_next_across_attention_tp(tmp_path, case):
    """A handoff between engines of different attention TP degrees (ModelRunner.pd_regroupable: GLM-5.3-Flash's caches
    are MLA, the same rows at any degree; its KDA state is head-split): tp=2 at attention TP 2 into tp=2 DP attention 2
    (attention TP 1: each decode rank concatenates both sender ranks' heads), and the reverse (each decode rank takes
    its half of the one sender rank's heads). Same tokens as one engine at tp=2, logprobs within LOGPROB_TOL."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from tests.test_glm5_next import build

    path = str(tmp_path)
    build(path, seed=1, index_topk=16)
    ps = prompts(15, (10, 37, 22))
    sps = params(len(ps))
    a2, a1 = dict(tp=2), dict(tp=2, dp_attention=2)
    pk, dk = (a2, a1) if case == "more_sender_ranks" else (a1, a2)
    compare(pd_two_processes(path, ps, sps, pk, dk), single(path, ps, sps, tp=2), f"glm5_next {case}")


def test_glm5_next_latency_prefill_into_a_context_parallel_decode_engine(tmp_path, monkeypatch):
    """The G1 latency deployment's shape on CPU: a prefill engine at DP attention 1 (tp=4, attention TP 4, page size 4)
    into a decode engine at DP attention 2 under KILN_DSA_CP=1 (tp=4, attention TP 2 and CP 2, page size 8): each
    decode rank regroups two sender ranks' KDA heads and picks its CP pools' positions out of the position-ordered
    MLA rows. Same tokens as one engine at tp=4, logprobs within LOGPROB_TOL. The run is also the layout check: the
    receiver refuses a handoff whose signature differs, and the two engines' attention TP differs."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    dsa_long = pytest.importorskip("kiln.models.dsa_long")
    if not hasattr(dsa_long, "cp_local_slots"):
        pytest.skip("no context parallelism in this tree")
    import multiprocessing as mp

    from kiln.models import mla
    from tests.test_glm5_next import build

    path = str(tmp_path)
    build(path, seed=1, index_topk=16, max_position_embeddings=4096)
    monkeypatch.setenv("KILN_DSA_POOL_CACHE", "separate")
    monkeypatch.setattr(mla, "POOL_CACHE", "separate")
    ps = prompts(16, (37, 70, 9, 21))
    sps = params(len(ps))
    want = single(path, ps, sps, tp=4, page_size=4)
    ctx = mp.get_context("spawn")
    q, addr = ctx.Queue(), ctx.Queue()
    proc = ctx.Process(target=_prefill_process_later, args=(path, dict(tp=4, page_size=4), ps, sps, addr, q))
    proc.start()
    D = None
    try:
        monkeypatch.setenv("KILN_DSA_CP", "1")
        D = engine(path, pd_role="decode", pd_listen="127.0.0.1:0", tp=4, dp_attention=2, page_size=8)
        assert D.attn_tp == 2 and D.model.cp == 2
        addr.put(D.pd_address)
        got = drive(None, D, ps, sps, prefill_proc=proc)
        assert q.get(timeout=300)[0] == "ok"
        proc.join(timeout=120)
        compare(got, want, "glm5_next attention TP 4 prefill -> attention TP 2 CP decode")
        D.pd_receiver.drain()
        assert D.pd_receiver.stats.held_bytes == 0 and D.pd_counts["refused"] == 0
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join(timeout=30)
        if D is not None:
            D.close()
