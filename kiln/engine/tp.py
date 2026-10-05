"""Tensor-parallel process group.

One process per rank; rank 0 runs the scheduler and the API, ranks > 0 replay every graph
call rank 0 makes (ModelRunner.serve), so all ranks execute identical graphs on identical
inputs and meet at the same in-graph collectives. Host-side coordination is gloo.

On Neuron each rank sees exactly one NeuronCore (NEURON_RT_VISIBLE_CORES) and all ranks
share one CCOM bootstrap endpoint (NEURON_RT_ROOT_COMM_ID), the arrangement vllm-neuron's
worker uses (vllm_neuron/vllm/worker/neuron_worker.py, _init_neuron_distributed_environment
_and_runtime and rendezvous_ccom_bootstrap). The CCOM port is the gloo port + 1 so the two
never collide (libtorch_neuronx_lite defaults the CCOM root to MASTER_PORT otherwise).
"""

from __future__ import annotations

import datetime
import errno
import os
import socket

import torch.distributed as dist

# Ranks > 0 wait in a broadcast while rank 0 compiles a new bucket, which can take tens of
# minutes for a large model; the default 30-minute gloo timeout would kill a healthy job.
TIMEOUT = datetime.timedelta(hours=4)


def port_free(p: int) -> bool:
    """Whether p can be bound on EVERY local address, as rank 0's TCPStore binds it (it listens on
    all interfaces). Free on 127.0.0.1 is not enough: gloo's own listeners sit on the host's
    address, and a long-lived process collects them (measured, 63f683b single-process suite on
    kiln-cf-4, 2026-10-04: pytest held 24 gloo listeners on 172.31.38.185, free_port probing
    127.0.0.1 returned 44417, one of them, and rank 0's TCPStore bind failed)."""
    families = [(socket.AF_INET, "0.0.0.0")] + ([(socket.AF_INET6, "::")] if socket.has_ipv6 else [])
    for fam, host in families:
        try:
            with socket.socket(fam) as s:
                s.bind((host, p))
        except OSError as e:
            if fam == socket.AF_INET6 and e.errno in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):  # no IPv6 here
                continue
            return False
    return True


def free_port() -> int:
    """A port p such that p and p + 1 were both free on every local address a moment ago."""
    for _ in range(50):
        with socket.socket() as a:
            a.bind(("0.0.0.0", 0))
            p = a.getsockname()[1]
        if port_free(p) and port_free(p + 1):
            return p
    raise RuntimeError("no free port pair")


# Process state init_rank / neuron_env change in the calling process, which rank 0 (the engine's
# own process: a server, a test session) restores on LLMEngine.close.
_ENV_KEYS = ("MASTER_ADDR", "MASTER_PORT", "NEURON_RT_VISIBLE_CORES", "NEURON_RT_ROOT_COMM_ID")


def process_state() -> dict:
    import torch

    return {"env": {k: os.environ.get(k) for k in _ENV_KEYS}, "threads": torch.get_num_threads()}


def restore_process_state(saved: dict) -> None:
    import torch

    for k, v in saved["env"].items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    torch.set_num_threads(saved["threads"])


def neuron_env(rank: int, port: int, core_base: int) -> None:
    """Must run before libtorch_neuronx_lite is imported in this process."""
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_base + rank)
    os.environ["NEURON_RT_ROOT_COMM_ID"] = f"127.0.0.1:{port + 1}"


def init_rank(rank: int, world: int, port: int) -> None:
    # Each rank's torch would otherwise start one intra-op thread per CPU: measured load
    # average 364 on a 128-vCPU trn1.32xlarge with 8 ranks loading weights.
    import torch
    from torch.distributed import distributed_c10d as c10d

    torch.set_num_threads(max(1, (os.cpu_count() or 1) // world))
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # torch 2.11 init_process_group names the default group from _world.group_count BEFORE its
    # rendezvous, so a rendezvous that raises leaves the count at 1 with no group to destroy. The
    # next init in this process then names its group "1" while fresh ranks name theirs "0", and
    # gloo's full mesh waits for keys under the other prefix until TIMEOUT: measured, rank 0 and
    # three workers of tests/test_attention_tp.py test_qwen3_5_gdn all in _new_process_group_helper
    # for 10+ minutes after test_mla's rank 0 failed to bind its store (py-spy, kiln-cf-4,
    # 2026-10-04). Put the count back so a failed init leaves the process as it found it.
    count = c10d._world.group_count
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}",
                                timeout=TIMEOUT)
    except BaseException:
        if not dist.is_initialized():
            c10d._world.group_count = count
        raise


def attention_group(world: int, attn_tp: int):
    """This rank's attention group (models/decoder.py DecoderForCausalLM: attention TP): ranks
    g * attn_tp .. (g + 1) * attn_tp - 1 for g = rank // attn_tp, consecutive so that a group spans
    as few chips as possible (two consecutive ranks are one trn1 chip, and an all-reduce over ranks
    0-1 inside a 32-rank world measured 0.15 ms against 2.55 ms over ranks 0-7, docs/neuron-notes.md
    "Collectives across chips"). None when no subgroup is needed (attn_tp 1 runs no attention
    collective, attn_tp == world uses the world group). dist.new_group is collective over the
    world, so EVERY rank must call this, with the same arguments, at the same point."""
    if attn_tp in (1, world):
        return None
    if world % attn_tp:
        raise ValueError(f"attention tp={attn_tp} does not divide tp={world}")
    rank, mine = dist.get_rank(), None
    for g in range(world // attn_tp):
        pg = dist.new_group(list(range(g * attn_tp, (g + 1) * attn_tp)))
        if g == rank // attn_tp:
            mine = pg
    return mine


def send(msg) -> None:
    dist.broadcast_object_list([msg], src=0)


def recv():
    box = [None]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def worker_main(rank: int, cfg, path: str, num_pages: int, port: int) -> None:
    if cfg.device == "neuron":
        neuron_env(rank, port, cfg.tp_core_base)
    init_rank(rank, cfg.tp, port)
    from .engine import build_shard

    _, runner = build_shard(cfg, path, num_pages, rank)
    runner.serve(recv)
    dist.destroy_process_group()
