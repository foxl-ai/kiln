"""The NKI facts kernels/delta_rule.py rests on, measured on trn1 (NeuronCore-v2) inside an LNL graph
(or in nki.simulate with --sim): how exact a float32 matmul is on the tensor engine, a transpose
done as a matmul with the identity, the scalar engine's exp with a per-partition bias, a matmul
writing a strided PSUM region, scalar_tensor_tensor with a PSUM operand, and whether a kernel may
call a module-level helper that issues instructions.

    python tools/probe_delta_prims.py [--sim]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402


def _copy_helper(dst, src):
    nisa.tensor_copy(dst=dst, src=src, engine=nisa.vector_engine)


@nki.jit
def k_prims(a, b, x, bias, ident):
    """a, b, x fp32 [128, 128], bias fp32 [128, 1], ident fp32 [128, 128]. Returns
    mm = a^T b (fp32 on the tensor engine), tr = a^T (matmul with the identity), ex = exp(x + bias)
    and exn = exp(-x + bias) (scalar engine), st[:, :, 16:32] = (a^T b)[:, (0..15, 16..31)] through a
    strided PSUM destination (zero elsewhere), stt = (x * 2) - (a^T b) with the PSUM tile as
    scalar_tensor_tensor's operand1, hc = a copied by a module-level helper."""
    f32 = nl.float32
    outs = []  # (the kernel compiler rejects list comprehensions: "unsupported expression")
    for _ in range(5):
        outs.append(nl.ndarray((128, 128), dtype=f32, buffer=nl.shared_hbm))
    st = nl.ndarray((128, 2, 128), dtype=f32, buffer=nl.shared_hbm)
    hc = nl.ndarray((128, 128), dtype=f32, buffer=nl.shared_hbm)
    a_s = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_s, src=a)
    b_s = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    nisa.dma_copy(dst=b_s, src=b)
    x_s = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x_s, src=x)
    bi = nl.ndarray((128, 1), dtype=f32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bi, src=bias)
    I = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    nisa.dma_copy(dst=I, src=ident)

    pm = nl.ndarray((128, 128), dtype=f32, buffer=nl.psum)
    nisa.nc_matmul(dst=pm, stationary=a_s, moving=b_s)
    r0 = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r0, src=pm, engine=nisa.vector_engine)
    nisa.dma_copy(dst=outs[0], src=r0)

    pt = nl.ndarray((128, 128), dtype=f32, buffer=nl.psum)
    nisa.nc_matmul(dst=pt, stationary=a_s, moving=I)
    r1 = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    nisa.activation(dst=r1, op=nl.copy, data=pt)  # (tensor_copy has no scalar engine on gen2)
    nisa.dma_copy(dst=outs[1], src=r1)

    r2 = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    nisa.activation(dst=r2, op=nl.exp, data=x_s, bias=bi, scale=1.0)
    nisa.dma_copy(dst=outs[2], src=r2)
    r3 = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    nisa.activation(dst=r3, op=nl.exp, data=x_s, bias=bi, scale=-1.0)
    nisa.dma_copy(dst=outs[3], src=r3)

    mv = nl.ndarray((128, 2, 16), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=mv[:, 0, :], src=b_s[:, 0:16], engine=nisa.vector_engine)
    nisa.tensor_copy(dst=mv[:, 1, :], src=b_s[:, 16:32], engine=nisa.vector_engine)
    ps = nl.ndarray((128, 2, 128), dtype=f32, buffer=nl.psum)
    zm = nl.ndarray((128, 2, 128), dtype=f32, buffer=nl.sbuf)
    nisa.memset(dst=zm, value=0.0)
    zs = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    nisa.memset(dst=zs, value=0.0)
    nisa.nc_matmul(dst=ps, stationary=zs, moving=zm)  # zero the tile by a matmul
    nisa.nc_matmul(dst=ps[:, :, 16:32], stationary=a_s, moving=mv)
    r4 = nl.ndarray((128, 2, 128), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=r4, src=ps, engine=nisa.vector_engine)
    nisa.dma_copy(dst=st, src=r4)

    r5 = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=r5, data=x_s, op0=nl.multiply, operand0=2.0, op1=nl.subtract, operand1=pm)
    nisa.dma_copy(dst=outs[4], src=r5)

    r6 = nl.ndarray((128, 128), dtype=f32, buffer=nl.sbuf)
    _copy_helper(r6, a_s)
    nisa.dma_copy(dst=hc, src=r6)
    return outs[0], outs[1], outs[2], outs[3], st, outs[4], hc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="nki.simulate on the host, no device")
    args = ap.parse_args()
    if args.sim:
        os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")

        def run(k, *a, **kw):
            return tuple(torch.as_tensor(o) for o in nki.simulate(k)(*a, **kw))
    else:
        import profile_layer as pl

        pl.setup_device()
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        def run(k, *a, **kw):
            import inspect

            names = [n for n in inspect.signature(k.func).parameters][: len(a)]
            f = torch.compile(lambda *t: wrap_nki(k)[1](**dict(zip(names, t)), **kw), **pl.OPTS)
            o = f(*[t.to(pl.DEV) for t in a])
            return tuple(x.cpu() for x in o)

    g = torch.Generator().manual_seed(0)
    a = torch.randn(128, 128, generator=g)
    b = torch.randn(128, 128, generator=g)
    x = torch.rand(128, 128, generator=g) * 170 - 85  # exp arguments over most of fp32's range
    bias = torch.rand(128, 1, generator=g) * 4 - 2
    mm, tr, ex, exn, st, stt, hc = run(k_prims, a, b, x, bias, torch.eye(128))
    ref = a.double().T @ b.double()
    rel = lambda got, want: ((got.double() - want).abs().max() / want.abs().max()).item()  # noqa: E731
    print(f"fp32 matmul a^T b: max rel err {rel(mm, ref):.3e} (bf16 operands would be ~4e-3)")
    print(f"transpose by identity matmul: exact {torch.equal(tr, a.T)}, max abs err {(tr - a.T).abs().max().item():.3e}")
    for name, got, arg in (("exp(x + b)", ex, x.double() + bias.double()), ("exp(-x + b)", exn, -x.double() + bias.double())):
        want = torch.exp(arg)
        ok = (arg > -87) & (arg < 88)
        r = ((got.double() - want).abs() / want)[ok]
        print(f"{name}: max rel err {r.max().item():.3e}, median {r.median().item():.3e} over {int(ok.sum())} in range; "
              f"below -87: max {got[arg <= -87].abs().max().item() if (arg <= -87).any() else 0:.3e}")
    want_st = torch.zeros(128, 2, 128, dtype=torch.float64)
    want_st[:, 0, 16:32] = ref[:, 0:16]
    want_st[:, 1, 16:32] = ref[:, 16:32]
    print(f"strided PSUM dst: max abs err {(st.double() - want_st).abs().max().item():.3e}")
    print(f"scalar_tensor_tensor with PSUM operand1: max rel err {rel(stt, 2 * x.double() - ref):.3e}")
    print(f"module-level helper call: exact {torch.equal(hc, a)}")


if __name__ == "__main__":
    main()
