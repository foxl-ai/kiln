"""models/mla.py _cp_attend_classes_x (kernels/dsa_slots_x.py, KILN_DSA_CP_SLOTS_X) against _cp_attend_classes on one
NeuronCore: the same o and lse bit for bit, and each form's time per call.

    python tools/probe_cp_slots_x.py [--rows 1024] [--heads 64] [--fits 0.7 1.0 0.0] [--iters 10]

The inputs are a lever-1 row group's (models/mla.py attention_cp_rows): T rows of every head's latent query (R 512), a
rank's value-ordered local list of keep 512 pools with this rank's selected entries a prefix-like run (each kept with
probability 0.9) whose end falls before slot 127 for a --fits fraction of the rows (the small class) and anywhere up
to keep for the rest (the full class), the tail pool owned by one rank in 8 and partly visible, an fp8 latent cache of
4096 pages of 32 tokens; buffers of 128 and 640 slots (CP_SLOTS_SMALL, dsa_decode.NCH x 128). The timed form returns
o and lse reduced on the device (the same reduction in both, so the difference is the attention's).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_inputs(T: int, H: int, fits: float, g: torch.Generator, keep: int = 512, small: int = 128):
    R, kp, pages, ps = 512, 4, 4096, 32
    kc = (torch.randn(pages * ps, 1, R, generator=g) * 2).clamp(-200, 200).to(torch.float8_e4m3fn)
    q_all = (torch.randn(T, H, R, generator=g) * 0.05).to(torch.bfloat16)
    nf = int(round(T * fits))
    end = torch.cat([torch.randint(0, small, (nf,), generator=g),  # one past the last selected entry, <= small - 1
                     torch.randint(small, keep + 1, (T - nf,), generator=g)])[torch.randperm(T, generator=g)]
    j = torch.arange(keep).view(1, keep)
    mine = (j < end.view(T, 1)) & (torch.rand(T, keep, generator=g) < 0.9)
    mine[torch.arange(T), (end - 1).clamp(min=0)] |= end > 0  # the run ends where its class says
    rows_sel = torch.randint(0, pages * ps // kp, (T, keep), generator=g)
    rows_tail = torch.randint(0, pages * ps // kp, (T, 1), generator=g)
    tail_own = torch.rand(T, generator=g) < 0.125
    npool = torch.randint(1000, 30000, (T,), generator=g)
    positions = npool * kp + torch.randint(0, kp, (T,), generator=g)
    return q_all, kc, rows_sel, rows_tail, mine, tail_own, npool, positions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--fits", type=float, nargs="+", default=[0.7, 1.0, 0.0])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    import profile_layer as pl

    pl.setup_device(a.cpu)
    if not a.cpu and (pl.DEV is None or pl.DEV.type == "cpu"):
        raise SystemExit("probe_cp_slots_x needs a NeuronCore (both forms would be the host emulation)")
    from kiln.kernels import dsa_decode
    from kiln.models import mla

    small, full, kp, scale = 128, (dsa_decode.NCH * 128 if not a.cpu else 640), 4, 192 ** -0.5
    for fi, fits in enumerate(a.fits):
        g = torch.Generator().manual_seed(11 + fi)
        args = make_inputs(a.rows, a.heads, fits, g)
        dev = tuple(x.to(pl.DEV) for x in args)

        def ref(*x):
            return mla._cp_attend_classes(*x, scale, kp, small, full)

        def new(*x):
            return mla._cp_attend_classes_x(*x, scale, kp, small, full)

        name = f"T={a.rows} H={a.heads} fits={fits}"
        ro, rl = (t.cpu() for t in torch.compile(ref, **pl.OPTS)(*dev))
        go, gl = (t.cpu() for t in torch.compile(new, **pl.OPTS)(*dev))
        same_o = torch.equal(ro.view(torch.int32), go.view(torch.int32))
        same_l = torch.equal(rl.view(torch.int32), gl.view(torch.int32))
        nfit = int((torch.where(args[4], torch.arange(512).view(1, 512) + 1, 0).amax(-1) <= small - 1).sum())
        msg = (f"  {name}: {nfit} small-class rows, {a.rows - nfit} full; o bit-identical {same_o} "
               f"({int((ro != go).flatten(1).any(-1).sum())} rows differ, max |d| {(ro - go).abs().max().item():.3e}), "
               f"lse bit-identical {same_l} ({int((rl != gl).any(-1).sum())} rows differ); finite "
               f"{bool(torch.isfinite(go).all() and torch.isfinite(gl).all())}")
        pl.say(msg, flush=True)
        if not a.cpu:
            t1 = pl.timed(f"classes   {name}", lambda *x: sum(t.clamp(min=-1.0).sum() for t in ref(*x)), dev, a.iters)
            t2 = pl.timed(f"classes_x {name}", lambda *x: sum(t.clamp(min=-1.0).sum() for t in new(*x)), dev, a.iters)
            if t1 == t1 and t2 == t2:
                pl.say(f"    {name}: classes {t1 * 1e3:.3f} ms, classes_x {t2 * 1e3:.3f} ms, "
                       f"{(t2 - t1) * 1e3:+.3f} ms ({t2 / t1:.3f}x)", flush=True)


if __name__ == "__main__":
    main()
