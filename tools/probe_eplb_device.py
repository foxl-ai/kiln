"""Device checks the EPLB rebalance relies on (models/eplb.py, ModelRunner._eplb), on one NeuronCore:

    python tools/probe_eplb_device.py

(1) an eager host-to-device copy into ONE slot of a device tensor laid out like an expert-parallel blob
([El + s, ...] uint8 / fp32) rewrites that slot and leaves the others bit-identical; (2) the routing-id remap
and the per-expert count, compiled as a graph with the device backend, equal their CPU values, including the
in-place accumulation into a buffer (ep_stats) across calls.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch


def main() -> None:
    import libtorch_neuronx_lite  # noqa: F401

    from kiln.engine.model_runner import canonical_neuron_backend
    from kiln.models import eplb

    dev = torch.device("neuron:0")
    g = torch.Generator().manual_seed(0)
    # (1) one-slot copies into blob-shaped device tensors
    for shape, dt in (((10, 128, 16, 2, 32, 128), torch.uint8), ((10, 16, 2, 32), torch.float32),
                      ((10, 128, 16, 4096), torch.uint8)):
        host = (torch.randint(0, 255, shape, generator=g, dtype=torch.int32).to(dt) if dt == torch.uint8
                else torch.randn(shape, generator=g))
        d = host.to(dev)
        new = (torch.randint(0, 255, shape[1:], generator=g, dtype=torch.int32).to(dt) if dt == torch.uint8
               else torch.randn(shape[1:], generator=g))
        d.data[9].copy_(new.to(dev))
        back = d.cpu()
        ok_slot = torch.equal(back[9], new)
        ok_rest = torch.equal(back[:9], host[:9])
        print(f"slot copy {tuple(shape)} {dt}: slot {ok_slot}, others {ok_rest}", flush=True)
        assert ok_slot and ok_rest
    # (2) remap + counts + in-place accumulation in a compiled graph
    E, tp, s, T, k = 288, 32, 1, 128, 8
    load = torch.rand(E, generator=g)
    load[[20, 21, 100]] += 50
    extra = eplb.replicas(load, tp, s)
    ids, mp = eplb.tables(extra, E, tp, s)
    stats = torch.zeros(E, dtype=torch.float32)

    def f(topi, ids_, mp_, st):
        st.add_(eplb.counts(topi, E))
        return eplb.remap(topi, ids_, mp_)

    fc = torch.compile(f, backend=canonical_neuron_backend(), fullgraph=True, dynamic=False)
    std = stats.to(dev)
    want_stats = torch.zeros(E)
    for it in range(3):
        topi = torch.randint(0, E, (T, k), generator=g)
        topi[:, 0] = 20
        got = fc(topi.to(dev), ids.to(dev), mp.to(dev), std).cpu()
        want = eplb.remap(topi, ids, mp)
        want_stats += eplb.counts(topi, E)
        print(f"remap call {it}: equal {torch.equal(got, want)}, ids >= E {int((got >= E).sum())}", flush=True)
        assert torch.equal(got, want)
    ok = torch.equal(std.cpu(), want_stats)
    print(f"stats accumulated in place over 3 calls: equal {ok}", flush=True)
    assert ok
    print("PROBE_EPLB_OK", flush=True)


if __name__ == "__main__":
    main()
