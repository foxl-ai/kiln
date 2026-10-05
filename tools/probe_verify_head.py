"""Does the speculative-verify post graph compile at MiMo-V2.6-Flash's tp=32 shapes?

    python tools/probe_verify_head.py [--rows 16]

Two ranks on the two NeuronCores of a trn1.2xlarge run, in one graph, what post_extend runs on
a tp=32 rank: final rms_norm, the rank's lm_head shard [4768, 4096], the one-hot placement into
[rows, 32 x 4768] and a real all-reduce (over 2 ranks; the shapes are the tp=32 ones), then
verify_sample. At rows = B x Q = 16 this graph failed in neuronx-cc 2.27 with NCC_IBIR243
"Access pattern out of bounds" on trn1.32xlarge (2026-10-03), while verify_sample alone, and
the 4-row decode post graph, compile. Variant "split" runs the head and verify_sample as two
graphs.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def rank_main(rank: int, port: int, rows: int) -> None:
    from kiln.engine import tp

    tp.neuron_env(rank, port, 0)
    tp.init_rank(rank, 2, port)
    import libtorch_neuronx_lite  # noqa: F401
    import torch.distributed as dist
    import torch.distributed._functional_collectives as funcol

    from kiln.engine.model_runner import neuronx_cc_args
    from kiln.engine.sampler import verify_sample

    dev = torch.device("neuron:0")
    opts = dict(backend="neuron_libtorch", fullgraph=True, dynamic=False,
                options={"compiler_args": neuronx_cc_args(torch.bfloat16)})
    grp = dist.group.WORLD
    TP, Vr, V, H = 32, 4768, 152576, 4096
    g = torch.Generator().manual_seed(0)
    h = torch.randn(rows, H, generator=g).to(torch.bfloat16).to(dev)
    norm = torch.ones(H, dtype=torch.bfloat16).to(dev)
    w = (torch.randn(Vr, H, generator=g) * 0.02).to(torch.bfloat16).to(dev)
    start = torch.tensor(rank * Vr).to(dev)
    rest = [x.to(dev) for x in (torch.zeros(rows), torch.ones(rows), torch.zeros(rows, dtype=torch.int64),
                                torch.zeros(rows), torch.full((rows, 64), 0.5), torch.full((rows,), 0.5),
                                torch.zeros(rows, dtype=torch.int64))]

    def head(h, norm, w, start, with_hn=False):
        x = h.float()
        hn = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16) * norm
        local = torch.nn.functional.linear(hn, w)
        mine = (torch.arange(TP, device=h.device) == start // Vr).to(local.dtype)
        full = (local.unsqueeze(1) * mine.view(1, -1, 1)).reshape(rows, TP * Vr)
        logits = funcol.all_reduce(full, "sum", grp)[:, :V].float()
        return (logits, hn) if with_hn else logits

    say = print if rank == 0 else (lambda *a, **k: None)
    rest_draft2d = rest[:-1] + [torch.zeros(rows // 4, 4, dtype=torch.int64).to(dev)]

    def mtp_post(h, norm, w, start, *r):  # what post_extend returns when an MTP head is loaded
        logits, hn = head(h, norm, w, start, True)
        return verify_sample(logits, *r[:-1], r[-1].reshape(-1)), hn

    for name in ("fused", "split", "fused_hn"):
        try:
            if name == "fused_hn":
                out = torch.compile(mtp_post, **opts)(h, norm, w, start, *rest_draft2d)[0]
            elif name == "fused":
                out = torch.compile(lambda *a: verify_sample(head(*a[:4]), *a[4:]), **opts)(h, norm, w, start, *rest)
            else:
                logits = torch.compile(head, **opts)(h, norm, w, start)
                out = torch.compile(verify_sample, **opts)(logits, *rest)
            out.cpu()
            say(f"  {name} rows={rows}: compiled and ran", flush=True)
        except Exception as e:  # report and continue
            say(f"  {name} rows={rows}: FAILED {type(e).__name__} {str(e)[:160]}", flush=True)
    dist.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=16)
    args = ap.parse_args()
    import multiprocessing as mp

    from kiln.engine.tp import free_port

    port = free_port()
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=rank_main, args=(r, port, args.rows)) for r in (0, 1)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()


if __name__ == "__main__":
    main()
