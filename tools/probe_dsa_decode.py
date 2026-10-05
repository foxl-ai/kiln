"""The pooled-DSA decode attention on one NeuronCore: the XLA mask form (models/mla.py _core over every key of the
bucket, the selection as an additive mask) against kernels/dsa_decode.py (the selected pools gathered), both
against the CPU emulation, at GLM-5.3-Flash's attention-TP-8 rank shape (8 heads, latent 512, pools of 4 tokens,
512 selected of the bucket's pools plus the tail pool).

    python tools/probe_dsa_decode.py --batch 4 16 32 64 --pages 264 [--kv fp8|bf16] [--forms xla nki]

Each row gets its own block table (random pages of a pool of pages), a context of `--context` tokens (default: the
whole bucket), 512 random complete pools selected (all of them when fewer are visible) and the query's own
incomplete pool. Reported per batch: max |error| of o against emulate() relative to its max, and the time per call
chained (IN_FLIGHT launches queued) and with a read-back each.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

H, R, PS, KP, KEEP = 8, 512, 32, 4, 512
NEG_INF = -1e30


def case(B: int, pages: int, context: int, dtype, seed: int):
    from kiln.kernels.dsa_decode import NCH

    g = torch.Generator().manual_seed(seed)
    NP = B * pages + 1
    kc = (torch.randn(NP * PS, 1, R, generator=g)).clamp(-8, 8).to(dtype)
    table = torch.randperm(NP - 1, generator=g)[: B * pages].view(B, pages) + 1
    L = pages * PS
    P = L // KP
    nvis = torch.full((B,), context)
    q_lat = (torch.randn(B, H, R, generator=g) * 0.05).to(torch.bfloat16)
    ncand = nvis // KP  # complete pools
    rows = torch.zeros(B, NCH * 128, dtype=torch.long)
    bias = torch.full((B, NCH * 128, KP), NEG_INF)
    mask = torch.full((B, L), NEG_INF)
    for b in range(B):
        nc = int(ncand[b])
        sel = torch.randperm(nc, generator=g)[: min(KEEP, nc)].sort().values
        pools = list(sel.tolist())
        n = len(pools)
        pt = int(nvis[b]) // KP  # the tail pool (its visible tokens), if any
        for j, p in enumerate(pools + [min(pt, P - 1)]):
            page = int(table[b, p * KP // PS])
            rows[b, j if j < n else KEEP] = (page * PS + (p * KP) % PS) // KP  # pool rows
        bias[b, :n] = 0.0
        for t in range(KP):
            if pt < P and pt * KP + t < int(nvis[b]):
                bias[b, KEEP, t] = 0.0
        for p in pools:
            mask[b, p * KP:(p + 1) * KP] = 0.0
        mask[b, pt * KP:int(nvis[b])] = 0.0
    return kc, table, q_lat, rows, bias, mask


def xla_form(kc, table, q_lat, mask, scale):
    """The mask form: every key of the bucket gathered by pages, converted, attended with the selection mask."""
    B, P = table.shape
    tok = (table.unsqueeze(-1) * PS + torch.arange(PS, device=table.device)).reshape(B, P * PS)
    K = kc.reshape(-1, R)[tok].to(torch.bfloat16)  # [B, L, R]
    s = torch.einsum("bhr,blr->bhl", q_lat, K)
    p = torch.softmax(s.float() * scale + mask.unsqueeze(1), dim=-1).to(torch.bfloat16)
    return torch.einsum("bhl,blr->bhr", p, K).float()


def nki_form(kc, rows, q_lat, bias, scale):
    from kiln.kernels.dsa_decode import attend

    return attend(q_lat, kc, rows, bias, scale)


def dbg_form(kc, rows, q_lat, bias, scale):
    from kiln.kernels.dsa_decode import attend

    return attend(q_lat, kc, rows, bias, scale, dbg=1)


def dbg2_form(kc, rows, q_lat, bias, scale):
    from kiln.kernels.dsa_decode import attend

    return attend(q_lat, kc, rows, bias, scale, dbg=2)


def dbg4_form(kc, rows, q_lat, bias, scale):
    from kiln.kernels.dsa_decode import attend

    return attend(q_lat, kc, rows, bias, scale, dbg=4)


def dbg5_form(kc, rows, q_lat, bias, scale):
    from kiln.kernels.dsa_decode import attend

    return attend(q_lat, kc, rows, bias, scale, dbg=5)


def dbg3_form(kc, rows, q_lat, bias, scale):
    from kiln.kernels.dsa_decode import attend

    return attend(q_lat, kc, rows, bias, scale, dbg=3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[4, 16, 32, 64])
    ap.add_argument("--pages", type=int, default=264)
    ap.add_argument("--context", type=int, default=0, help="visible tokens per row (default: the whole bucket)")
    ap.add_argument("--kv", default="fp8", choices=["fp8", "bf16"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--forms", nargs="+", default=["xla", "nki"])
    a = ap.parse_args()
    from kiln import platform

    platform.configure_runtime_env()
    import libtorch_neuronx_lite  # noqa: F401

    from kiln.engine.model_runner import IN_FLIGHT, canonical_neuron_backend, neuronx_cc_args
    from kiln.kernels.dsa_decode import emulate

    dev = torch.device("neuron:0")
    dtype = torch.float8_e4m3fn if a.kv == "fp8" else torch.bfloat16
    scale = 256 ** -0.5  # GLM-5.3-Flash: qk_head_dim 256
    for B in a.batch:
        kc, table, q_lat, rows, bias, mask = case(B, a.pages, a.context or a.pages * PS - 1, dtype, B)
        ref = emulate(q_lat, kc.reshape(-1, R), rows, bias, scale)
        for name in a.forms:
            if name == "xla":
                fn, args = xla_form, (kc, table, q_lat, mask, scale)
            elif name in ("dbg4", "dbg5"):  # raw scores / the bias as the kernel holds them
                from kiln.kernels.dsa_decode import KP as KP_, NCH

                f = torch.compile(dbg4_form if name == "dbg4" else dbg5_form, backend=canonical_neuron_backend(),
                                  fullgraph=True, dynamic=False, options={"compiler_args": neuronx_cc_args(dtype)})
                o, sd = f(*[x.to(dev) if isinstance(x, torch.Tensor) else x for x in (kc, rows, q_lat, bias, scale)])
                sd = sd.cpu()
                tok = (rows.unsqueeze(-1) * KP_ + torch.arange(KP_)).reshape(B, -1)
                K = kc.reshape(-1, R)[tok].to(torch.bfloat16).float()
                raw = torch.einsum("bhr,btr->bht", q_lat.float(), K)
                want = raw if name == "dbg4" else bias.reshape(B, 1, -1).expand(B, H, -1)
                want = want.reshape(B, H, NCH, 128, KP_).permute(0, 1, 2, 4, 3).reshape(B, H, -1)
                e = (sd - want).abs()
                if name == "dbg4":  # recover the q the kernel used for (b 0, head 0) per chunk by least squares
                    Kt = K.view(B, NCH, 128, KP_, R).permute(0, 1, 3, 2, 4).reshape(B, NCH, KP_ * 128, R)
                    for ch in range(NCH - 1):
                        sol = torch.linalg.lstsq(Kt[0, ch].double(), sd[0, 0, ch * KP_ * 128:(ch + 1) * KP_ * 128].double()).solution.float()
                        qq = q_lat.float().view(B * H, R // 128, 128)
                        sb = sol.view(R // 128, 128)
                        match = [[round(float(torch.nn.functional.cosine_similarity(sb[lc], qq[c, lc2], dim=0)), 3)
                                  for lc2 in range(R // 128)] for lc in range(R // 128) for c in [0]]
                        best = [(lc, max(((float(torch.nn.functional.cosine_similarity(sb[lc], qq[c, lc2], dim=0)), c, lc2)
                                          for c in range(B * H) for lc2 in range(R // 128)))) for lc in range(R // 128)]
                        print(f"  ch {ch}: recovered q block vs (row/head c, block lc2) best cos: "
                              + ", ".join(f"lc{lc}->c{bc} lc{bl} {bv:.3f}" for lc, (bv, bc, bl) in best), flush=True)
                for ch in range(NCH):
                    for t in range(KP_):
                        got = sd[0, :, (ch * KP_ + t) * 128:(ch * KP_ + t + 1) * 128]
                        errs = [float((got - want[0, :, (c2 * KP_ + t2) * 128:(c2 * KP_ + t2 + 1) * 128]).abs().max())
                                for c2 in range(NCH) for t2 in range(KP_)]
                        k = min(range(len(errs)), key=lambda z: errs[z])
                        print(f"   block ch {ch} t {t}: own err {errs[ch * KP_ + t]:.2e}, best match ch {k // KP_} t {k % KP_} "
                              f"err {errs[k]:.2e}", flush=True)
                    blk = e[0, :, ch * KP_ * 128:(ch + 1) * KP_ * 128]
                    print(f"B={B} {name} ch {ch}: max|err| {blk.max():.3e} (|want| max {want[0].abs().max():.3e})",
                          flush=True)
                continue
            elif name == "dbg3":  # the kernel's transposed latent blocks against the host's
                from kiln.kernels.dsa_decode import KP as KP_, NCH

                f = torch.compile(dbg3_form, backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                                  options={"compiler_args": neuronx_cc_args(dtype)})
                o, ktd = f(*[x.to(dev) if isinstance(x, torch.Tensor) else x for x in (kc, rows, q_lat, bias, scale)])
                ktd = ktd.cpu().float()  # [B, NCH, KP, 128 r, LC, 128 p]
                kc2 = kc.reshape(-1, KP_ * R).float()
                g_ = kc2[rows].view(B, NCH, 128, KP_, R // 128, 128)  # [b, ch, p, t, lc, r]
                want = g_.permute(0, 1, 3, 5, 4, 2)  # [b, ch, t, r, lc, p]
                for ch in range(NCH):
                    for t in range(KP_):
                        for lc in range(R // 128):
                            ok = (ktd[0, ch, t, :, lc, :] == want[0, ch, t, :, lc, :])
                            if not ok.all():
                                print(f"B={B} dbg3 ch {ch} t {t} lc {lc}: {int((~ok).sum())} of {ok.numel()} wrong", flush=True)
                print(f"B={B} dbg3 done", flush=True)
                continue
            elif name == "dbg2":  # the kernel's gathered latent against the host's
                from kiln.kernels.dsa_decode import KP as KP_, NCH

                f = torch.compile(dbg2_form, backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                                  options={"compiler_args": neuronx_cc_args(dtype)})
                o, kd = f(*[x.to(dev) if isinstance(x, torch.Tensor) else x for x in (kc, rows, q_lat, bias, scale)])
                kd = kd.cpu().float()  # [B, NCH, 128, KP R]
                kc2 = kc.reshape(-1, KP_ * R).float()
                want = kc2[rows].view(B, NCH, 128, KP_ * R)
                for ch in range(NCH):
                    ok = (kd[0, ch] == want[0, ch]).all(-1)
                    print(f"B={B} dbg2 ch {ch}: {int(ok.sum())} of 128 slots gathered right; first wrong "
                          f"{(~ok).nonzero().flatten()[:6].tolist()}", flush=True)
                    if not ok.all():
                        p0 = int((~ok).nonzero()[0])
                        hit = (kc2 == kd[0, ch, p0]).all(-1).nonzero().flatten().tolist()[:3]
                        print(f"   slot {p0}: wanted pool row {int(rows[0, ch * 128 + p0])}, got pool row(s) {hit}",
                              flush=True)
                continue
            elif name == "dbg":  # the kernel's scores against the host's, in the kernel's token order
                from kiln.kernels.dsa_decode import KP as KP_, NCH

                f = torch.compile(dbg_form, backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                                  options={"compiler_args": neuronx_cc_args(dtype)})
                o, sd = f(*[x.to(dev) if isinstance(x, torch.Tensor) else x for x in (kc, rows, q_lat, bias, scale)])
                o, sd = o.cpu(), sd.cpu()
                tok = (rows.unsqueeze(-1) * KP_ + torch.arange(KP_)).reshape(B, -1)  # (slot, t)
                K = kc.reshape(-1, R)[tok].to(torch.bfloat16).float()
                sh = torch.einsum("bhr,btr->bht", q_lat.float(), K) * scale + bias.reshape(B, 1, -1)
                # kernel order: (ch, t, p) for slot ch * 128 + p
                sh = sh.view(B, H, NCH, 128, KP_).permute(0, 1, 2, 4, 3).reshape(B, H, -1)
                ok = sh > -1e29
                e = (sd - sh).abs()[ok.expand_as(sd)]
                print(f"B={B} dbg: score max|err| {e.max():.3e} (|s| max {sh[ok].abs().max():.3e}); "
                      f"masked agree {bool(((sd < -1e29) == ~ok).all())}; o err {((o - ref).abs().max() / ref.abs().max()):.3e}",
                      flush=True)
                for h in range(2):
                    bad = ((sd[0, h] - sh[0, h]).abs() > 1e-2) & ok[0, h]
                    idx = bad.nonzero().flatten()[:8].tolist()
                    print(f"  head {h}: {int(bad.sum())} bad of {int(ok[0, h].sum())}; first {idx}", flush=True)
                    bv = bad.view(NCH, KP_, 128)
                    print("   bad by t:", bv.sum((0, 2)).tolist(), "by ch:", bv.sum((1, 2)).tolist(),
                          "by p%8:", bv.view(NCH, KP_, 16, 8).sum((0, 1, 2)).tolist(), flush=True)
                    for j in idx[:3]:  # where does the kernel's value sit among the host's scores?
                        hit = ((sh[0, h] - sd[0, h, j]).abs() < 1e-4).nonzero().flatten().tolist()[:4]
                        hit_any = [(hh, (sh[0, hh] - sd[0, h, j]).abs().argmin().item(),
                                    (sh[0, hh] - sd[0, h, j]).abs().min().item()) for hh in range(H)]
                        print(f"   j={j} kernel {sd[0, h, j]:.5f} host {sh[0, h, j]:.5f} same-head hits {hit} "
                              f"best per head {[(a, b, round(c, 6)) for a, b, c in hit_any][:4]}", flush=True)
                continue
            else:
                fn, args = nki_form, (kc, rows, q_lat, bias, scale)
            f = torch.compile(fn, backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                              options={"compiler_args": neuronx_cc_args(dtype)})
            dargs = [x.to(dev) if isinstance(x, torch.Tensor) else x for x in args]
            t = time.perf_counter()
            o = f(*dargs).cpu()
            first = time.perf_counter() - t
            err = ((o - ref).abs().max() / ref.abs().max()).item()
            o2, o3 = f(*dargs).cpu(), f(*dargs).cpu()
            print(f"  {name}: calls agree {torch.equal(o, o2)} {torch.equal(o2, o3)}; per-row err "
                  f"{[round(float((o[b] - ref[b]).abs().max() / ref.abs().max()), 4) for b in range(B)]}", flush=True)
            outs = []
            t = time.perf_counter()
            for _ in range(a.iters):
                outs.append(f(*dargs))
                if len(outs) > IN_FLIGHT:
                    outs[-1 - IN_FLIGHT].cpu()
            outs[-1].cpu()
            chained = (time.perf_counter() - t) / a.iters
            t = time.perf_counter()
            for _ in range(5):
                f(*dargs).cpu()
            sync = (time.perf_counter() - t) / 5
            print(f"B={B:3d} {name} kv {a.kv}: o max|err| / max|o| {err:.2e} | {chained * 1e3:.3f} ms chained, "
                  f"{sync * 1e3:.3f} ms with read-back (first call {first:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
