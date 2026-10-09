"""kernels/segmented_attn.py (nkilib's segmented prefill attention over Kiln's paged KV) against the XLA chunk form
of models/decoder.py _attend, on one device's cores, for a dense GQA layer's shapes.

    python tools/probe_segmented_attn.py --heads 8 --kv-heads 2 --chunk 4096 --priors 0 4096 28672
    python tools/probe_segmented_attn.py --cpu                      # the reference logic only, no device

Per (chunk C, prior p): random bf16 q [C, nh, D] and a paged cache [pages x ps, nkv, D] whose pages the block table
lists in a shuffled order, the chunk's K / V already in it (as _gqa stores them before attending). The reference is
the decoder's own arithmetic (fp32 scores over the page bucket, j <= position, softmax, P V) on the CPU in fp32. It
prints max |kernel - ref| / max |ref|, the same for the device's XLA path, and both device times (p50).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def xla_chunk(q, kc_all, vc_all, table, pos0, ps, scale):
    """decoder._gqa + _attend's chunk form: gather the bucket's pages, fp32 scores, causal bias, softmax, P V."""
    C, nh, D = q.shape
    nkv = kc_all.shape[1]
    G = nh // nkv
    kc = kc_all.view(-1, ps, nkv, D)[table].flatten(0, 1)  # [L, nkv, D]
    vc = vc_all.view(-1, ps, nkv, D)[table].flatten(0, 1)
    L = kc.shape[0]
    pos = pos0 + torch.arange(C, device=q.device)
    j = torch.arange(L, device=q.device)
    bias = torch.where(j.unsqueeze(0) <= pos.unsqueeze(1), 0.0, -1e30).view(1, 1, C, L)
    s = torch.einsum("chgd,lhd->hgcl", q.view(C, nkv, G, D), kc).float() * scale + bias
    p = torch.softmax(s, dim=-1).to(q.dtype)
    return torch.einsum("hgcl,lhd->chgd", p, vc).reshape(C, nh, D)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--chunk", type=int, nargs="+", default=[4096])
    ap.add_argument("--priors", type=int, nargs="+", default=[0, 4096, 28672])
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--no-xla", action="store_true", help="skip the device XLA path (its fp32 square is big)")
    ap.add_argument("--head-major", action="store_true",
                    help="also time the kernel's own layout ([pages, nkv, ps, D], kv_bsh=False) on the same values")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    import profile_layer as pl

    pl.setup_device(args.cpu)
    from kiln.kernels import segmented_attn as sa

    g = torch.Generator().manual_seed(0)
    nh, nkv, D, ps = args.heads, args.kv_heads, args.head_dim, args.page_size
    scale = D ** -0.5
    for C in args.chunk:
        for prior in args.priors:
            ctx = prior + C
            need = -(-ctx // ps)
            P = 1 << (need - 1).bit_length()  # the page bucket: a power of two pages, as Kiln's ladders
            pages = P + 8
            perm = torch.randperm(pages, generator=g)[:P]  # the sequence's pages, shuffled
            kc = (torch.randn(pages * ps, nkv, D, generator=g)).to(torch.bfloat16)
            vc = (torch.randn(pages * ps, nkv, D, generator=g)).to(torch.bfloat16)
            q = (torch.randn(C, nh, D, generator=g)).to(torch.bfloat16)
            pos0 = torch.tensor([prior], dtype=torch.int64)
            ref = xla_chunk(q.float(), kc.float(), vc.float(), perm, prior, ps, scale)
            den = float(ref.abs().max())
            tag = f"C {C:5d} prior {prior:6d} bucket {P * ps:6d} keys"
            if args.cpu:
                pl.say(f"{tag}: reference only (--cpu), |ref| max {den:.3f}")
                continue
            dev = [x.to(pl.DEV) for x in (q, kc, vc, perm, pos0)]

            def kern(q_, kc_, vc_, t_, p_):
                return sa.attend(q_, kc_, vc_, t_, p_, ps, scale)

            got = torch.compile(kern, **pl.OPTS)(*dev).cpu().float()
            ek = float((got - ref).abs().max()) / den
            tk = pl.timed(f"{tag} segmented", kern, dev, args.iters)
            line = f"{tag}: segmented err {ek:.2e} {tk * 1e3:8.3f} ms"
            if args.head_major:
                from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

                from kiln import platform

                def kern_hm(q_, kc_, vc_, t_, p_):
                    qh = (q_ * scale).permute(1, 0, 2).contiguous()
                    return wrap_nki(sa._kernel())[platform.nki_grid()](
                        q=qh, k_cache=kc_, v_cache=vc_, block_tables=t_.reshape(1, -1).to(torch.int32),
                        prior_tokens=p_.reshape(1, 1).to(torch.int32), block_size=ps, prior_seg_size=C, scale=1.0,
                        tp_q=True, tp_out=False, num_q_heads=nh, kv_bsh=False, rev=sa.REV).permute(1, 0, 2)

                hm = lambda c: c.view(-1, ps, nkv, D).permute(0, 2, 1, 3).contiguous()  # noqa: E731
                devh = [dev[0], hm(kc).to(pl.DEV), hm(vc).to(pl.DEV), dev[3], dev[4]]
                gh = torch.compile(kern_hm, **pl.OPTS)(*devh).cpu().float()
                eh = float((gh - ref).abs().max()) / den
                th = pl.timed(f"{tag} segmented head-major", kern_hm, devh, args.iters)
                line += f" | head-major err {eh:.2e} {th * 1e3:8.3f} ms"
            if not args.no_xla:
                def xla(q_, kc_, vc_, t_, p_):
                    return xla_chunk(q_, kc_, vc_, t_, p_[0], ps, scale)

                gx = torch.compile(xla, **pl.OPTS)(*dev).cpu().float()
                ex = float((gx - ref).abs().max()) / den
                tx = pl.timed(f"{tag} xla", xla, dev, args.iters)
                line += f" | xla err {ex:.2e} {tx * 1e3:8.3f} ms | speedup {tx / tk:5.2f}x"
            flop = 4 * nh * D * C * (prior + C / 2)  # causal: QK and PV over the visible keys
            line += f" | {flop / tk / 1e12:6.1f} TFLOP/s"
            pl.say(line, flush=True)


if __name__ == "__main__":
    main()
