"""KDA decode step on one NeuronCore: the XLA form (models/linear_attn.py decode: gather the state rows, zero the
rows that start fresh, recurrent_step, write the rows back) against kernels/kda_decode.py, both against the CPU
emulation, at GLM-5.3-Flash's attention-TP-8 rank shape (8 heads of 128 x 128).

    python tools/probe_kda_decode.py --batch 4 16 32 64 --rows 65 [--iters 20] [--forms xla nki]

Each form runs in its own compiled graph on the same pool contents; reported per batch: max |error| of o and of
the written state rows against emulate() (and that every other pool row is untouched), and the time per call
chained (IN_FLIGHT launches queued, as the runner issues them) and with a read-back each.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def inputs(B: int, R: int, H: int, D: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    pool = torch.randn(R, H, D, D, generator=g) * 0.05
    perm = torch.randperm(R - 1, generator=g)[:B]
    slot = perm.clone()
    keep = torch.ones(B, dtype=torch.bool)
    if B >= 4:  # two padding rows: the scratch row (R - 1), starting from zero
        slot[-2:] = R - 1
        keep[-2:] = False
    q = torch.nn.functional.normalize(torch.randn(B, H, D, generator=g), dim=-1) * D ** -0.5
    k = torch.nn.functional.normalize(torch.randn(B, H, D, generator=g), dim=-1)
    v = torch.randn(B, H, D, generator=g)
    gl = -5.0 * torch.sigmoid(torch.randn(B, H, D, generator=g))
    beta = torch.sigmoid(torch.randn(B, H, generator=g))
    return pool, slot, keep, q, k, v, gl, beta


def xla_form(pool, slot, keep, q, k, v, g, beta):
    from kiln.models.linear_attn import _write_rows, recurrent_step

    S = pool[slot]
    S = torch.where(keep.view(-1, 1, 1, 1), S, torch.zeros_like(S))
    o, S = recurrent_step(q, k, v, g, beta, S)
    _write_rows(pool, slot, S)
    return o


def nki_form(pool, slot, keep, q, k, v, g, beta):
    from kiln.kernels.kda_decode import decode_step

    return decode_step(pool, slot, keep, q, k, v, g, beta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[4, 16, 32, 64])
    ap.add_argument("--rows", type=int, default=65)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--forms", nargs="+", default=["xla", "nki"])
    a = ap.parse_args()
    from kiln import platform

    platform.configure_runtime_env()  # before libtorch_neuronx_lite: the LNC an NKI kernel is traced for
    import libtorch_neuronx_lite  # noqa: F401

    from kiln.engine.model_runner import IN_FLIGHT, canonical_neuron_backend, neuronx_cc_args
    from kiln.kernels.kda_decode import emulate

    dev = torch.device("neuron:0")
    D = 128
    forms = {"xla": xla_form, "nki": nki_form}
    for B in a.batch:
        pool, slot, keep, q, k, v, g, beta = inputs(B, a.rows, a.heads, D, B)
        R = a.rows
        rd = torch.where(keep, slot, torch.full_like(slot, R))
        o_ref, pool_ref = emulate(pool, torch.stack([rd, slot]), q, k, v, g, beta)
        real = keep.nonzero().flatten()
        for name in a.forms:
            f = torch.compile(forms[name], backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                              options={"compiler_args": neuronx_cc_args(torch.bfloat16)})
            dpool = pool.to(dev)
            args = [x.to(dev) for x in (slot, keep, q, k, v, g, beta)]
            t = time.perf_counter()
            o = f(dpool, *args).cpu()
            first = time.perf_counter() - t
            got = dpool.cpu()
            eo = (o[real] - o_ref[real]).abs().max().item()
            es = (got[slot[real]] - pool_ref[slot[real]]).abs().max().item()
            others = torch.ones(R, dtype=torch.bool)
            others[slot] = False
            untouched = bool(torch.equal(got[others], pool[others]))
            outs = []
            t = time.perf_counter()
            for _ in range(a.iters):
                outs.append(f(dpool, *args))
                if len(outs) > IN_FLIGHT:
                    outs[-1 - IN_FLIGHT].cpu()
            outs[-1].cpu()
            chained = (time.perf_counter() - t) / a.iters
            t = time.perf_counter()
            for _ in range(5):
                f(dpool, *args).cpu()
            sync = (time.perf_counter() - t) / 5
            print(f"B={B:3d} {name}: o max|err| {eo:.2e} state max|err| {es:.2e} other rows untouched {untouched} | "
                  f"{chained * 1e3:.3f} ms chained, {sync * 1e3:.3f} ms with read-back (first call {first:.1f} s)",
                  flush=True)


if __name__ == "__main__":
    main()
