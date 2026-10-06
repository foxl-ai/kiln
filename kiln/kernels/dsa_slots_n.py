"""kernels/dsa_slots.py with the number of rows given at run time: the same arithmetic and layout, over the first n rows
of its inputs only, n a device tensor (the loop register is loaded from it instead of from the static row count).

Why: under context parallelism (models/dsa_long.py, models/mla.py attention_cp) a rank's real share of a row's
selected pools is about keep / A, but a row may hold more, up to keep. Sized for the worst case, every row pays for
640 slots (39.5 us per row at 64 heads, fp8, trn1); 128 slots cost 14.9 us (tools/probe_dsa_long.py slots --slots,
kiln-lc-k3, 2026-10-06: 14.9 / 20.5 / 27.5 / 39.5 us per row at 128 / 256 / 384 / 640 slots). attention_cp's slot
classes (KILN_DSA_CP_SLOT_CLASSES=1) put the rows that fit in 128 slots into one buffer and the rest into another,
each a full-size static buffer with its live rows first, and call this kernel once per buffer with that buffer's row
count: every row is attended over all of its slots (exact for any split), and the time follows the real split.

Rows at or past n are not computed and their outputs are not written (the caller never reads them). The rows
processed are ceil(n / RPI) RPI: the last iteration's extra rows are the buffer's own (finite) rows.

The kernel is a separate module so that kernels/dsa_slots.py's source, and so its REV and the keys of every graph that
calls it, are unchanged; its REV is this module's kernel source and dsa_slots.REV.
"""

from __future__ import annotations

import torch

from . import dsa_slots

NEG_INF = dsa_slots.NEG_INF
KP = dsa_slots.KP


def emulate(q_lat: torch.Tensor, kc: torch.Tensor, rows: torch.Tensor, bias: torch.Tensor, scale: float,
            n: torch.Tensor, lse: bool = False):
    """dsa_slots.emulate over the first n rows; the rows past n are zero (the kernel leaves them unwritten, so a caller
    that read one would see garbage on the device and zeros here)."""
    out = dsa_slots.emulate(q_lat, kc, rows, bias, scale, lse=lse)
    N = q_lat.shape[0]
    live = torch.arange(N, device=q_lat.device) < n.reshape(()).to(torch.int64)
    if lse:
        o, ls = out
        return o * live.view(N, 1, 1), ls * live.view(N, 1)
    return out * live.view(N, 1, 1)


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None and dsa_slots.nki is not None:
    from .dsa_slots import KTR, _pair, _psum, _row, _sb  # the kernel tracer resolves names, not module attributes

    F32, BF16, I32 = nl.float32, nl.bfloat16, nl.int32

    @nki.jit
    def kiln_dsa_slots_n_kernel(q_lat, kc, rows_t, bias, identb, n_iter, scale: float, fp8: int, want_lse: int,
                                rev: int, qkw: int = 1):
        """kiln_dsa_slots_kernel's arguments plus n_iter int32 [1, 1]: the device-loop iterations to run (RPI rows
        each), at most NI."""
        NI, rpi, H, R = q_lat.shape
        NCH = rows_t.shape[3]
        LC = R // 128
        T = NCH * KP * 128
        o = nl.ndarray((NI, rpi, H, R), dtype=F32, buffer=nl.shared_hbm)
        lse = None
        if want_lse:
            lse = nl.ndarray((NI, rpi, H, 1), dtype=F32, buffer=nl.shared_hbm)
        KD = kc.dtype
        if fp8:
            KD = nl.float8_e4m3
        X = dict(H=H, R=R, LC=LC, NCH=NCH, T=T, rpi=rpi, q_lat=q_lat, kc=kc, rows_t=rows_t, bias=bias, o=o, lse=lse,
                 want_lse=want_lse, qkw=qkw)
        IB = _sb((128, 128), BF16)
        nisa.dma_copy(dst=IB, src=identb)
        X["IB"] = IB
        X["scale"] = scale
        X["K"] = []
        for _ in range(2):
            kv = []
            for _ in range(NCH):
                kv.append(_sb((128, KP * R), KD))
            X["K"].append(kv)
        X["KT"] = []
        if qkw:
            X["KTA"] = _pair((128, LC, T), BF16)
            X["BR"] = _pair((1, T), F32)
            X["RSC"] = _sb((1, 128))
            nisa.memset(dst=X["RSC"], value=1.0 / scale)
        else:
            for _ in range(KTR):
                X["KT"].append(_sb((128, LC, 128), BF16))
        X["RS"] = _pair((128, NCH), I32)
        X["QR"] = _pair((H, R), BF16)
        X["QT"] = _pair((128, LC, H), BF16)
        X["S"] = _pair((H, T), F32)
        X["P"] = _pair((H, T), BF16)
        X["OB"] = _pair((H, R), F32)
        X["MX"] = _pair((H, 1), F32)
        X["NMX"] = _pair((H, 1), F32)
        X["DEN"] = _pair((H, 1), F32)
        X["RDEN"] = _pair((H, 1), F32)
        X["LS"] = _pair((H, 1), F32)
        X["PTb"] = _pair((128, H), BF16)
        cnt = _sb((1, 1), I32)
        nisa.dma_copy(dst=cnt, src=n_iter)  # the run-time iteration count (dsa_slots: iota of the static NI)
        rt = nisa.register_alloc()
        nisa.register_load(rt, cnt)

        def body(it):
            X["psum"] = _psum(H, LC)
            for k in range(rpi):
                _row(X, it, k)

        nl.fori_loop(0, rt, body)
        if want_lse:
            return o, lse
        return o
else:
    kiln_dsa_slots_n_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source and dsa_slots.REV (the row code it runs is dsa_slots')."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_dsa_slots_n_kernel = None", a)
    return zlib.crc32(src[a:b].encode() + str(dsa_slots.REV).encode())


REV = _kernel_rev()


def attend(q_lat: torch.Tensor, kc: torch.Tensor, rows: torch.Tensor, bias: torch.Tensor, scale: float,
           n: torch.Tensor, lse: bool = False):
    """dsa_slots.attend over the first n rows (n an int tensor of one element, 0 .. N): o [N, H, R] fp32 (and lse
    [N, H]); the rows past n are unwritten on the device (zeros on the CPU)."""
    kc2 = kc.reshape(kc.shape[0], -1)
    if q_lat.device.type == "cpu":
        return emulate(q_lat, kc2, rows, bias, scale, n, lse)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if kiln_dsa_slots_n_kernel is None:
        raise RuntimeError("the NKI DSA slots kernel needs the nki package (the Neuron venv)")
    N, H, R = q_lat.shape
    NS = rows.shape[1]
    if NS % 128:
        raise ValueError(f"dsa_slots_n: {NS} slots per row is not a multiple of 128")
    NCH = NS // 128
    rpi = dsa_slots.RPI
    Np = -(-N // rpi) * rpi
    q = q_lat.to(torch.bfloat16)
    rw, bs = rows, bias.float()
    if Np != N:  # padded rows read pool row 0, every slot masked
        q = torch.cat([q, q.new_zeros(Np - N, H, R)])
        rw = torch.cat([rw, rw.new_zeros(Np - N, NS)])
        bs = torch.cat([bs, bs.new_full((Np - N, NS, KP), NEG_INF)])
    fp8 = kc2.dtype == torch.float8_e4m3fn
    kcp = kc2.reshape(kc2.shape[0] // KP, KP * R)
    NI = Np // rpi
    rows_t = rw.to(torch.int32).reshape(NI, rpi, NCH, 128).permute(0, 1, 3, 2).contiguous()
    bias_t = bs.reshape(NI, rpi, NCH, 128, KP).permute(0, 1, 2, 4, 3).reshape(NI, rpi, NCH * KP * 128).contiguous()
    eye = torch.eye(128, device=q_lat.device).to(torch.bfloat16)
    # ceil(n / rpi) in exact fp32 arithmetic (n <= 2^24)
    it = torch.floor((n.reshape(1, 1).to(torch.float32) + (rpi - 1)) * (1.0 / rpi)).to(torch.int32)
    out = wrap_nki(kiln_dsa_slots_n_kernel)[platform.nki_grid()](
        q_lat=q.reshape(NI, rpi, H, R).contiguous(), kc=kcp, rows_t=rows_t, bias=bias_t, identb=eye, n_iter=it,
        scale=float(scale), fp8=int(fp8), want_lse=int(lse), rev=REV, qkw=dsa_slots.QKW)
    if lse:
        o, ls = out
        return o.reshape(Np, H, R)[:N], ls.reshape(Np, H)[:N]
    return out.reshape(Np, H, R)[:N]
