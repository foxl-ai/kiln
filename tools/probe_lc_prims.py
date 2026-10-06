"""Which selection primitives run on trn1 (NeuronCore-v2) inside an LNL graph, exactly, and at what cost:
nisa.max8, nisa.nc_match_replace8 (with dst_idx), rounds of the two as a top-k extraction, nisa.nc_n_gather,
nisa.nc_find_index8. For kernels/dsa_long_select.py.

    python tools/probe_lc_prims.py [case ...]   # max8 mr8 extract ngather find8 (default: all)

Each case compares the device's result with the host's definition and prints its time (p50 of synchronous
calls) at a few widths.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402

F32, I32, U32 = nl.float32, nl.int32, nl.uint32
REPL = -3.0e38  # what an extracted element is replaced with (below NEG_INF = -1e30)


@nki.jit
def k_max8(src):
    N, W = src.shape
    out = nl.ndarray((N, 8), dtype=F32, buffer=nl.shared_hbm)
    s = nl.ndarray((N, W), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=s, src=src)
    v = nl.ndarray((N, 8), dtype=F32, buffer=nl.sbuf)
    nisa.max8(dst=v, src=s)
    nisa.dma_copy(dst=out, src=v)
    return out


@nki.jit
def k_mr8(src):
    N, W = src.shape
    out = nl.ndarray((N, W), dtype=F32, buffer=nl.shared_hbm)
    oi = nl.ndarray((N, 8), dtype=I32, buffer=nl.shared_hbm)
    ov = nl.ndarray((N, 8), dtype=F32, buffer=nl.shared_hbm)
    s = nl.ndarray((N, W), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=s, src=src)
    v = nl.ndarray((N, 8), dtype=F32, buffer=nl.sbuf)
    nisa.max8(dst=v, src=s)
    d = nl.ndarray((N, W), dtype=F32, buffer=nl.sbuf)
    ix = nl.ndarray((N, 8), dtype=U32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=ix, data=s, vals=v)
    nisa.nc_match_replace8(dst=d, data=s, vals=v, imm=REPL)
    nisa.dma_copy(dst=out, src=d)
    nisa.dma_copy(dst=oi, src=ix.view(I32))
    nisa.dma_copy(dst=ov, src=v)
    return out, oi, ov


@nki.jit
def k_extract(src, rounds: int, inplace: int):
    """rounds of max8 + match_replace8: the top 8 rounds values [N, 8 rounds] and their positions."""
    N, W = src.shape
    ov = nl.ndarray((N, 8 * rounds), dtype=F32, buffer=nl.shared_hbm)
    oi = nl.ndarray((N, 8 * rounds), dtype=I32, buffer=nl.shared_hbm)
    a = nl.ndarray((N, W), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=src)
    b = a if inplace else nl.ndarray((N, W), dtype=F32, buffer=nl.sbuf)
    V = nl.ndarray((N, 8 * rounds), dtype=F32, buffer=nl.sbuf)
    I = nl.ndarray((N, 8 * rounds), dtype=U32, buffer=nl.sbuf)
    cur, nxt = a, b
    for r in range(rounds):
        nisa.max8(dst=V[:, 8 * r:8 * r + 8], src=cur)
        nisa.nc_find_index8(dst=I[:, 8 * r:8 * r + 8], data=cur, vals=V[:, 8 * r:8 * r + 8])
        nisa.nc_match_replace8(dst=nxt, data=cur, vals=V[:, 8 * r:8 * r + 8], imm=REPL)
        if not inplace:
            cur, nxt = nxt, cur
    nisa.dma_copy(dst=ov, src=V)
    nisa.dma_copy(dst=oi, src=I.view(I32))
    return ov, oi


@nki.jit
def k_ngather(data, idx):
    N, W = data.shape
    out = nl.ndarray(idx.shape, dtype=F32, buffer=nl.shared_hbm)
    d = nl.ndarray((N, W), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=d, src=data)
    ii = nl.ndarray(idx.shape, dtype=I32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ii, src=idx)
    o = nl.ndarray(idx.shape, dtype=F32, buffer=nl.sbuf)
    nisa.nc_n_gather(dst=o, data=d, indices=ii.view(U32))
    nisa.dma_copy(dst=out, src=o)
    return out


@nki.jit
def k_find8(data, vals):
    N, W = data.shape
    out = nl.ndarray((N, 8), dtype=I32, buffer=nl.shared_hbm)
    d = nl.ndarray((N, W), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=d, src=data)
    v = nl.ndarray((N, 8), dtype=F32, buffer=nl.sbuf)
    nisa.dma_copy(dst=v, src=vals)
    o = nl.ndarray((N, 8), dtype=U32, buffer=nl.sbuf)
    nisa.nc_find_index8(dst=o, data=d, vals=v)
    nisa.dma_copy(dst=out, src=o.view(I32))
    return out


@nki.jit
def k_gather(scr, ro, sub: int, form: int):
    """Gather keep rows of `sub` fp32 per partition from scr [R, sub] by ro [128, keep] int32: form 0 one dma_copy per
    index column, form 1 one dma_copy for all (a 2-D vector_offset)."""
    R, S = scr.shape
    keep = ro.shape[1]
    out = nl.ndarray((128, keep * sub), dtype=F32, buffer=nl.shared_hbm)
    r = nl.ndarray((128, keep), dtype=I32, buffer=nl.sbuf)
    nisa.dma_copy(dst=r, src=ro)
    c = nl.ndarray((128, keep * sub), dtype=F32, buffer=nl.sbuf)
    if form == 0:
        for j in range(keep):
            nisa.dma_copy(dst=c[:, j * sub:(j + 1) * sub],
                          src=scr.ap(pattern=[[sub, 128], [1, sub]], offset=0,
                                     vector_offset=r.ap(pattern=[[keep, 128], [1, 1]], offset=j), indirect_dim=0))
    else:
        nisa.dma_copy(dst=c.reshape((128, keep, sub)),
                      src=scr.ap(pattern=[[sub, 128], [sub, keep], [1, sub]], offset=0,
                                 vector_offset=r.ap(pattern=[[keep, 128], [1, keep]], offset=0), indirect_dim=0))
    nisa.dma_copy(dst=out, src=c)
    return out


def gather_case(pl, run, g):
    for sub in (2, 8, 32):
        NB = 4096
        scr = torch.randn(128 * NB, sub, generator=g)
        ro = (torch.randint(0, NB, (128, 512), generator=g) + torch.arange(128).view(128, 1) * NB).to(torch.int32)
        want = scr[ro.long()].reshape(128, -1)
        for form in (0, 1):
            got = run(f"gather 512 rows of {sub} fp32 per partition, form {form}", k_gather,
                      {"scr": scr.to(pl.DEV), "ro": ro.to(pl.DEV)}, dict(sub=sub, form=form))
            if got is not None:
                pl.say(f"    equals: {torch.equal(got.cpu(), want)}", flush=True)


def host_order(s: torch.Tensor, k: int):
    """The first k of each row in (value descending, index ascending) order: values and positions."""
    N, W = s.shape
    j = torch.arange(W)
    out_v, out_i = [], []
    for n in range(N):
        order = sorted(range(W), key=lambda p: (-float(s[n, p]), p))[:k]
        out_i.append(order)
        out_v.append([float(s[n, p]) for p in order])
    return torch.tensor(out_v), torch.tensor(out_i)


def main() -> None:
    import profile_layer as pl
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    cases = sys.argv[1:] or ["max8", "mr8", "extract", "ngather", "find8"]
    pl.setup_device(False)
    g = torch.Generator().manual_seed(0)

    def run(name, k, named, kw, iters=10):
        names = list(named)
        args = tuple(named[n] for n in names)
        f = lambda *a: wrap_nki(k)[1](**dict(zip(names, a)), **kw)  # noqa: E731
        t = pl.timed(name, f, args, iters)
        if t != t:
            return None
        return torch.compile(f, **pl.OPTS)(*args)

    for case in cases:
        if case == "max8":
            for W in (512, 4096, 16384):
                s = torch.randint(-50, 50, (128, W), generator=g).float()  # many duplicates
                got = run(f"max8 [128, {W}]", k_max8, {"src": s.to(pl.DEV)}, {})
                if got is not None:
                    want = torch.sort(s, dim=-1, descending=True).values[:, :8]
                    pl.say(f"    max8 equals the 8 largest with duplicates: {torch.equal(got.cpu(), want)}", flush=True)
        elif case == "mr8":
            for W in (512, 16384):
                s = torch.randint(-3, 4, (128, W), generator=g).float()
                got = run(f"max8 + match_replace8 [128, {W}]", k_mr8, {"src": s.to(pl.DEV)}, {})
                if got is None:
                    continue
                d, ix, v = (x.cpu() for x in got)
                wv, wi = host_order(s[:16], 8)
                ix = ix.to(torch.int64)
                okv = torch.equal(v[:16], wv)
                okset = all(sorted(ix[n].tolist()) == sorted(wi[n].tolist()) for n in range(16))
                okpair = all(float(s[n, int(ix[n, c])]) == float(v[n, c]) for n in range(16) for c in range(8))
                repl = s.clone()
                for n in range(128):
                    repl[n, ix[n]] = REPL
                pl.say(f"    values {okv}; positions = the 8 lowest-index (value desc) set {okset}; each idx holds its "
                       f"value {okpair}; replaced tile equals {torch.equal(d, repl)}; idx row 0 {ix[0].tolist()} "
                       f"values {v[0].tolist()}", flush=True)
        elif case == "extract":
            for W, R in ((512, 64), (8192, 64), (16384, 64), (16384, 8)):
                for inplace in (1, 0):
                    s = torch.randint(-20, 20, (128, W), generator=g).float()
                    s[:, ::3] = torch.randn(128, (W + 2) // 3, generator=g)
                    got = run(f"extract {R} rounds [128, {W}] inplace={inplace}", k_extract, {"src": s.to(pl.DEV)},
                              dict(rounds=R, inplace=inplace))
                    if got is None:
                        continue
                    v, ix = (x.cpu() for x in got)
                    ix = ix.to(torch.int64)
                    wv, wi = host_order(s[:8], 8 * R)
                    okset = all(set(ix[n].tolist()) == set(wi[n].tolist()) for n in range(8))
                    okv = torch.equal(v[:8], wv)
                    pl.say(f"    top {8 * R} set exact (value desc, index asc): {okset}; values in order {okv}",
                           flush=True)
        elif case == "ngather":
            for W, K in ((512, 512), (512, 2048)):
                data = torch.randn(128, W, generator=g)
                idx = torch.randint(0, W, (128, K), generator=g).to(torch.int32)
                got = run(f"nc_n_gather [128, {W}] -> [128, {K}]", k_ngather,
                          {"data": data.to(pl.DEV), "idx": idx.to(pl.DEV)}, {})
                if got is not None:
                    want = torch.gather(data, 1, idx.long())
                    pl.say(f"    equals torch.gather: {torch.equal(got.cpu(), want)}", flush=True)
        elif case == "gather":
            gather_case(pl, run, g)
        elif case == "find8":
            data = torch.randint(-3, 4, (128, 512), generator=g).float()
            vals = torch.sort(data, -1, descending=True).values[:, :8].contiguous()
            got = run("nc_find_index8 [128, 512]", k_find8, {"data": data.to(pl.DEV), "vals": vals.to(pl.DEV)}, {})
            if got is not None:
                pl.say(f"    row 0 {got.cpu()[0].tolist()} for vals {vals[0].tolist()}", flush=True)


if __name__ == "__main__":
    main()
