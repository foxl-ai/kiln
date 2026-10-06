"""The pooled-DSA indexer's decode scores at long context on one NeuronCore: the XLA form (the page gather of the pool
keys, then glm5_next.pool_index's einsums) against kernels/dsa_index.py, both against the CPU emulation, at
GLM-5.3-Flash's shapes (32 indexer heads of 128, pools of 4 tokens, 8 pools per 32-token page).

    python tools/probe_dsa_index.py --batch 1 4 16 --pages 264 4096 32768 [--forms xla nki] [--select] [--simulate]

Each row gets its own block table (random pages of a pool of pages) and every pool of its bucket complete except the
last one (cand NEG_INF there). --select adds kernels/dsa_topk.py's selection of the 512 best pools (decode_slots'
call) to every form. Reported per (batch, pages): the max |error| of the scores against emulate() relative to the
max |score|, the time per call chained (IN_FLIGHT launches queued) and with a read-back each, and the pool-key bytes
per call over that time. --simulate runs the kernel in nki.simulate on the host instead (correctness only; small
shapes; NEURON_PLATFORM_TARGET_OVERRIDE picks the target).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

HI, D, PS, KP, KEEP = 32, 128, 32, 4, 512
PPP = PS // KP
NEG_INF = -1e30


FP8 = False  # --fp8


def case(B: int, pages: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    NPg = B * pages + 1
    pkc = (torch.randn(NPg, PPP * D, generator=g) * 0.5).to(torch.bfloat16)  # page rows: a page's PPP pools in order
    if FP8:
        pkc = pkc.to(torch.float8_e4m3fn)
    table = (torch.randperm(NPg - 1, generator=g)[: B * pages].view(B, pages) + 1).to(torch.int32)
    q = (torch.randn(B, HI, D, generator=g) * 0.1).to(torch.bfloat16)
    w = torch.randn(B, HI, generator=g) * D ** -0.5
    P = pages * PPP
    cand = torch.zeros(B, P)
    cand[:, -1] = NEG_INF
    return pkc, table, q, w, cand


def xla_form(q, w, pkc, table, cand):
    """The decode path's form: the pages' pool keys gathered in order, scored per head, relu, weighted head sum."""
    B, NPg = table.shape
    pk = pkc[table.long()].view(B, NPg * PPP, D)
    s = torch.einsum("bhd,bpd->bhp", q.float(), pk.float())
    return torch.einsum("bh,bhp->bp", w, torch.relu(s)) + cand


def nki_form(q, w, pkc, ppg, cand):
    from kiln.kernels import dsa_index

    return dsa_index.scores(q, w, pkc, ppg, cand)


def with_select(fn):
    from kiln.kernels import dsa_topk

    def f(*a):
        sc = fn(*a)
        return sc, dsa_topk.select(sc, KEEP, vis_only=True, kp=1, tail=False)

    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--pages", type=int, nargs="+", default=[264, 4096, 32768])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--forms", nargs="+", default=["xla", "nki"])
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--fp8", action="store_true", help="an fp8 e4m3 pool-key cache (the minimal KV layout's)")
    a = ap.parse_args()
    global FP8
    FP8 = a.fp8
    from kiln.kernels import dsa_index

    if a.simulate:
        import nki

        for B in a.batch:
            for pages in a.pages:
                pkc, table, q, w, cand = case(B, pages, B + pages)
                ppg = dsa_index.page_groups(table)
                ref = dsa_index.scores(q, w, pkc, ppg, cand)  # emulate() on the host
                xr = xla_form(q, w, pkc, table, cand)
                qT = q.reshape(B * HI, D).t().contiguous()
                wb = w.float().repeat(1, 4).unsqueeze(1).expand(B, 128, 4 * HI).contiguous()
                eye = torch.eye(128).to(torch.bfloat16)
                grid = int(os.environ.get("KILN_SIM_GRID", "1"))
                k = dsa_index.kiln_dsa_index_fp8_kernel if FP8 else dsa_index.kiln_dsa_index_kernel
                o = torch.as_tensor(nki.simulate(k[grid] if grid > 1 else k)(
                    qT=qT, wb=wb, pkc=pkc, ppg=ppg, cand=cand, identb=eye, rev=dsa_index.REV8 if FP8 else dsa_index.REV,
                    dge=int(os.environ.get("KILN_SIM_DGE", "0")), spl=1)).float()
                ok = ref > -1e29
                err = ((o - ref).abs()[ok].max() / ref[ok].abs().max()).item()
                xerr = ((xr - ref).abs()[ok].max() / ref[ok].abs().max()).item()
                print(f"simulate B={B} pages={pages} grid {grid}: kernel err {err:.2e} (xla form vs emulate {xerr:.2e}); "
                      f"masked agree {bool(((o < -1e29) == ~ok).all())}", flush=True)
        return
    from kiln import platform

    platform.configure_runtime_env()
    import libtorch_neuronx_lite  # noqa: F401

    from kiln.engine.model_runner import IN_FLIGHT, canonical_neuron_backend, neuronx_cc_args

    import torch._dynamo as dynamo

    dynamo.config.cache_size_limit = 256  # one compile per (form, shape): past dynamo's default 8 it refuses
    dynamo.config.accumulated_cache_size_limit = 1024
    dev = torch.device("neuron:0")
    print("kiln platform", platform.describe() if hasattr(platform, "describe") else platform.target(), flush=True)
    for pages in a.pages:
        for B in a.batch:
            pkc, table, q, w, cand = case(B, pages, B + pages)
            ppg = dsa_index.page_groups(table)
            ref = dsa_index.scores(q, w, pkc, ppg, cand)  # emulate() on the host
            ok = ref > -1e29
            kbytes = B * pages * PPP * D * pkc.element_size()
            for name in a.forms:
                fn, args = ((xla_form, (q, w, pkc, table, cand)) if name == "xla"
                            else (nki_form, (q, w, pkc, ppg, cand)))
                if a.select:
                    fn = with_select(fn)
                f = torch.compile(fn, backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                                  options={"compiler_args": neuronx_cc_args(torch.bfloat16, FP8)})
                dargs = [x.to(dev) if isinstance(x, torch.Tensor) else x for x in args]
                try:
                    t = time.perf_counter()
                    o = f(*dargs)
                    o = o[0].cpu() if a.select else o.cpu()
                    first = time.perf_counter() - t
                except Exception as e:  # a shape the form cannot compile or load is a result too
                    print(f"B={B:3d} pages={pages} {name}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                    continue
                err = ((o - ref).abs()[ok].max() / ref[ok].abs().max()).item()
                outs = []
                t = time.perf_counter()
                for _ in range(a.iters):
                    outs.append(f(*dargs))
                    if len(outs) > IN_FLIGHT:
                        x = outs[-1 - IN_FLIGHT]
                        (x[0] if a.select else x).cpu()
                x = outs[-1]
                (x[0] if a.select else x).cpu()
                chained = (time.perf_counter() - t) / a.iters
                t = time.perf_counter()
                for _ in range(5):
                    x = f(*dargs)
                    (x[0] if a.select else x).cpu()
                sync = (time.perf_counter() - t) / 5
                print(f"B={B:3d} pages={pages:6d} pools={pages * PPP:7d} {name}{'+select' if a.select else ''}: "
                      f"err {err:.2e} | {chained * 1e3:.3f} ms chained ({kbytes / chained / 1e9:.1f} GB/s of pool keys), "
                      f"{sync * 1e3:.3f} ms with read-back (first call {first:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
