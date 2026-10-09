"""Which SBUF / HBM state an NKI device loop (nl.fori_loop) may carry in and out, one NeuronCore (neuronx-cc 2.27,
nki 0.6.0): each case is a tiny kernel compiled and run alone, printed as compiled / failed and its output checked.

    python tools/probe_nki_loop_liveout.py [--cases sbuf_out sbuf_in hbm_out sbuf_out_rw]

  sbuf_in      an SBUF tile written before the loop, read inside it (kernels/dsa_slots_n.py's form)
  sbuf_out     an SBUF tile written inside a 0 / 1-trip loop, read after it
  sbuf_out_rw  an SBUF tile written before the loop, overwritten inside it, read after it
  hbm_out      the loop writes an HBM scratch, read back into SBUF after it
  psum_outside a PSUM tile allocated before the loop, written and read inside it only (kernels/dsa_slots.py allocates
               its PSUM inside the loop body)
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


try:  # the Neuron venv
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

if nki is not None:
    F32, I32 = nl.float32, nl.int32

    @nki.jit
    def k_loop(x, flag, case: int):
        """x fp32 [128, 64]; flag int32 [1, 1]; out = x + 1 (case 0: sbuf_in, 1: sbuf_out, 2: sbuf_out_rw, 3:
        hbm_out, 4: psum_outside), or x itself where the loop did not run."""
        out = nl.ndarray((128, 64), dtype=F32, buffer=nl.shared_hbm)
        scr = nl.ndarray((128, 64), dtype=F32, buffer=nl.shared_hbm)
        xs = nl.ndarray((128, 64), dtype=F32, buffer=nl.sbuf)
        nisa.dma_copy(dst=xs, src=x)
        ys = nl.ndarray((128, 64), dtype=F32, buffer=nl.sbuf)
        fl = nl.ndarray((1, 1), dtype=I32, buffer=nl.sbuf)
        nisa.dma_copy(dst=fl, src=flag)
        reg = nisa.register_alloc()
        nisa.register_load(reg, fl)
        ps = nl.ndarray((128, 64), dtype=F32, buffer=nl.psum)
        if case == 2 or case == 3:
            nisa.tensor_copy(dst=ys, src=xs, engine=nisa.vector_engine)
            if case == 3:
                nisa.dma_copy(dst=scr, src=ys)

        def body(i):
            if case == 0:
                zs = nl.ndarray((128, 64), dtype=F32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=zs, data=xs, op0=nl.add, operand0=1.0, engine=nisa.vector_engine)
                nisa.dma_copy(dst=out, src=zs)
            elif case == 3:
                zs = nl.ndarray((128, 64), dtype=F32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=zs, data=xs, op0=nl.add, operand0=1.0, engine=nisa.vector_engine)
                nisa.dma_copy(dst=scr, src=zs)
            elif case == 4:
                nisa.activation(dst=ps, op=nl.copy, data=xs, bias=None, scale=1.0)
                zs = nl.ndarray((128, 64), dtype=F32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=zs, data=ps, op0=nl.add, operand0=1.0, engine=nisa.vector_engine)
                nisa.dma_copy(dst=out, src=zs)
            else:
                nisa.tensor_scalar(dst=ys, data=xs, op0=nl.add, operand0=1.0, engine=nisa.vector_engine)

        nl.fori_loop(0, reg, body)
        if case == 1 or case == 2:
            nisa.dma_copy(dst=out, src=ys)
        elif case == 3:
            rs = nl.ndarray((128, 64), dtype=F32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rs, src=scr)
            nisa.dma_copy(dst=out, src=rs)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["sbuf_in", "sbuf_out", "sbuf_out_rw", "hbm_out", "psum_outside"])
    a = ap.parse_args()
    import profile_layer as pl

    pl.setup_device(False)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from kiln import platform

    names = {"sbuf_in": 0, "sbuf_out": 1, "sbuf_out_rw": 2, "hbm_out": 3, "psum_outside": 4}
    x = torch.randn(128, 64)
    for name in a.cases:
        for f in (1, 0):
            if name in ("sbuf_in", "psum_outside") and f == 0:
                continue  # out unwritten

            def fn(x_, fl_, case=names[name]):
                return wrap_nki(k_loop)[platform.nki_grid()](x=x_, flag=fl_, case=case)

            try:
                got = torch.compile(fn, **pl.OPTS)(x.to(pl.DEV), torch.tensor([[f]], dtype=torch.int32).to(pl.DEV)).cpu()
                want = x + 1.0 if f else x
                pl.say(f"  {name} flag {f}: compiled, output {'right' if torch.equal(got, want) else 'WRONG'}",
                       flush=True)
            except Exception as e:
                pl.say(f"  {name} flag {f}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    main()
