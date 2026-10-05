"""A tensor-parallel engine that fails to start must leave its process able to start the next one.

The failure measured in the 63f683b single-process suite (kiln-cf-4, 2026-10-04; tests/conftest.py
has the account): engine/tp.py free_port probed 127.0.0.1 only and returned a port a gloo listener
of an earlier engine held on the host's address, rank 0's TCPStore bind (on every interface)
failed, the spawned worker kept waiting for that store, and torch's group counter stayed at 1, so
the next tensor-parallel engine of the session hung in gloo's full mesh until its 4-hour timeout.
"""

import multiprocessing
import os
import socket

import pytest
import torch

_KEYS = ("MASTER_ADDR", "MASTER_PORT", "NEURON_RT_VISIBLE_CORES", "NEURON_RT_ROOT_COMM_ID")


def _state():
    """What rank 0's process may not keep changed: init_rank's environment and thread count."""
    return {k: os.environ.get(k) for k in _KEYS}, torch.get_num_threads()


def _restore(saved):
    env, threads = saved
    for k, v in env.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    torch.set_num_threads(threads)


def _listener(host: str = "0.0.0.0") -> socket.socket:
    s = socket.socket()
    s.bind((host, 0))
    s.listen()
    return s


def test_port_free_sees_a_port_held_on_another_local_address():
    """127.0.0.2 stands in for the host's address (gloo binds the address the hostname resolves
    to): a port held there is free on 127.0.0.1, which is all free_port used to ask."""
    from kiln.engine import tp

    try:
        held = _listener("127.0.0.2")
    except OSError:
        pytest.skip("127.0.0.2 is not a local address here (Linux routes all of 127/8 to loopback)")
    with held:
        p = held.getsockname()[1]
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", p))  # the old check: "free"
        assert not tp.port_free(p)
    p = tp.free_port()
    assert tp.port_free(p) and tp.port_free(p + 1)


def test_a_failed_rendezvous_leaves_no_process_group_state():
    """init_rank on a port another socket holds raises, and the next init in this process names its
    default group "0", as every freshly spawned rank does."""
    import torch.distributed as dist
    from torch.distributed import distributed_c10d as c10d

    from kiln.engine import tp

    saved = _state()
    try:
        with _listener() as held:
            with pytest.raises(Exception, match="(?i)address already in use|EADDRINUSE|errno: 98"):
                tp.init_rank(0, 1, held.getsockname()[1])
        assert not dist.is_initialized()
        assert c10d._world.group_count == 0
        tp.init_rank(0, 1, tp.free_port())
        try:
            assert c10d._get_default_group().group_name == "0"
        finally:
            dist.destroy_process_group()
    finally:
        _restore(saved)


def test_an_engine_whose_rank_0_cannot_start_stops_its_workers(tmp_path, monkeypatch):
    """LLMEngine(tp=2) whose rank-0 rendezvous fails raises, terminates the worker it spawned and
    leaves no process group, environment or thread-count change; the next tp=2 engine in the same
    process then generates what tp=1 generates."""
    import torch.distributed as dist
    from tests.test_architectures import build
    from torch.distributed import distributed_c10d as c10d

    from kiln.config import EngineConfig
    from kiln.engine import tp
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    build("qwen3", str(tmp_path))
    kw = dict(model_path=str(tmp_path), device="cpu", dtype=torch.float32, page_size=4, num_pages=64,
              max_num_seqs=2, max_model_len=64, max_prefill_tokens=8)
    before, kids = _state(), set(multiprocessing.active_children())
    with _listener() as held:
        monkeypatch.setattr(tp, "free_port", lambda: held.getsockname()[1])
        with pytest.raises(Exception, match="(?i)address already in use|EADDRINUSE|errno: 98"):
            LLMEngine(EngineConfig(tp=2, **kw))
    monkeypatch.undo()
    assert set(multiprocessing.active_children()) == kids
    assert not dist.is_initialized() and c10d._world.group_count == 0
    assert _state() == before
    sp = SamplingParams(max_new_tokens=6, ignore_eos=True)
    prompts = [[5, 9, 11, 200, 3], list(range(40, 52))]
    want = [r.output_ids for r in LLMEngine(EngineConfig(**kw)).generate(prompts, sp)]
    two = LLMEngine(EngineConfig(tp=2, **kw))
    try:
        assert [r.output_ids for r in two.generate(prompts, sp)] == want
    finally:
        two.close()
    assert set(multiprocessing.active_children()) == kids and not dist.is_initialized()
    assert _state() == before
