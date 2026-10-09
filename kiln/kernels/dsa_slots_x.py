"""kernels/dsa_slots_n.py over the rows of a full-size buffer named by an index list, for several slot classes in one
call: each class's rows are read (q_lat, slot rows, bias) at row idx[i] of [T, ...] inputs and written (o, lse) at row
idx[i] of ONE [T, ...] output, a device loop per class over its run-time row count. The same arithmetic, instruction
for instruction, as dsa_slots_n (so the same o and lse, bit for bit); only the five DMAs that address a row differ.

Why: models/mla.py _cp_attend_classes (KILN_DSA_CP_SLOT_CLASSES) gathers each class's rows into a compact buffer
(q_all[idx], rows[idx], bias[idx]), runs dsa_slots_n over its first n rows, and puts the outputs back in row order
(o[pos], ls[pos], then a torch.where over the two classes): at 1024 rows x 64 heads of 512 that is ~128 MB of
[T, H, R] fp32 gathered per class plus the select, ~7 ms of DMA_INDIRECT and ~2 ms of tail gathers per DSA layer in
the 1M replay (call 200, kiln-lc2-32, 2026-10-06). Here no row moves outside the kernel: the iteration's RPI row
indices are one more DMA in the loop, and every row of the output is written by exactly one class (the classes
partition the rows), so there is nothing to merge.

Measured alone (tools/probe_cp_slots_x.py, 1024 rows x 64 heads, 128 / 640 slots, fp8 latent, kiln-lc2-k1, 2026-10-07):
o and lse bit-identical to _cp_attend_classes' in every case, but no faster: 31.52 -> 31.52 ms per call with 717 rows
in the small class and 307 in the full one, 23.6 -> 22.6 ms all small, 49.7 -> 52.0 ms all full. Both calls are the
slots kernel's time (device profile, tools/prof_attn_kernels.py cp_classes / cp_classes_x: each engine 25-30% busy over
33.3 ms, the XLA gathers and select 1.7 ms of vector and 0.34 ms of GpSimd time, overlapped with the kernel), so the
gathers this removes (284 MB less HBM read and 150 MB less written per call) cost nothing there; whether they cost the
1M layer graph its ~7 ms is that graph's measurement (KILN_DSA_CP_SLOTS_X=1 in its replay).

The index list of a class holds its n rows first (models/dsa_long.compact); the loop runs ceil(n / RPI) iterations and
the entries from n to the next multiple of RPI must name a row of the class too (the caller repeats the class's last
row): those iterations compute that row again and write the same bytes.

The kernel is a separate module so that the sources of kernels/dsa_slots.py and kernels/dsa_slots_n.py, so their REVs
and the keys of every graph that calls them, are unchanged; its REV is this module's kernel source and dsa_slots.REV.
"""

from __future__ import annotations

import torch

from . import dsa_slots

NEG_INF = dsa_slots.NEG_INF
KP = dsa_slots.KP


def emulate(q_lat: torch.Tensor, kc: torch.Tensor, classes, scale: float):
    """(o [T, H, R], lse [T, H]) fp32: per class (rows [T, NS] int, bias [T, NS, KP] fp32, idx [T] int, n [1] int),
    dsa_slots.emulate of the rows idx[:n] at those rows; rows no class names are zero (unwritten on the device)."""
    T, H, R = q_lat.shape
    o = torch.zeros(T, H, R, dtype=torch.float32, device=q_lat.device)
    ls = torch.zeros(T, H, dtype=torch.float32, device=q_lat.device)
    for rows, bias, idx, n in classes:
        sel = idx.reshape(-1)[:int(n.reshape(()))].long()
        if sel.numel():
            oc, lc = dsa_slots.emulate(q_lat[sel], kc, rows[sel], bias[sel], scale, lse=True)
            o[sel], ls[sel] = oc, lc
    return o, ls


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None and dsa_slots.nki is not None:
    from .dsa_slots import KTR, _pair, _psum, _sb  # the kernel tracer resolves names, not module attributes

    F32, BF16, I32 = nl.float32, nl.bfloat16, nl.int32
    VE = nisa.vector_engine

    def _row_x(X, ix, k):
        """dsa_slots._row for row (it, k) of a class's loop, the row being ix [1, 1] int32 (its index list's entry):
        the same instructions, with the row-addressed DMAs at ix of the [T, ...] tensors instead of at (it, k)."""
        H, R, LC, NCH, T, rpi = X["H"], X["R"], X["LC"], X["NCH"], X["T"], X["rpi"]
        x = k % 2
        IB = X["IB"]
        PT, PS, PP, PO, PQ = X["psum"]
        # the row's slot indices, q_lat, bias (broadcast over its H head partitions)
        rs = X["RS"][x]
        nisa.dma_copy(dst=rs, src=X["rows_t"].ap(pattern=[[NCH, 128], [1, NCH]], offset=0, scalar_offset=ix,
                                                 indirect_dim=0))
        Kv = X["K"][x]
        for ch in range(NCH):
            nisa.dma_copy(dst=Kv[ch], src=X["kc"].ap(pattern=[[KP * R, 128], [1, KP * R]], offset=0,
                                                       vector_offset=rs.ap(pattern=[[NCH, 128], [1, 1]], offset=ch),
                                                       indirect_dim=0))
        qr = X["QR"][x]
        nisa.dma_copy(dst=qr, src=X["q_lat"].ap(pattern=[[R, H], [1, R]], offset=0, scalar_offset=ix, indirect_dim=0))
        S = X["S"][x]
        if X["qkw"]:
            br = X["BR"][x]
            nisa.dma_copy(dst=br, src=X["bias"].ap(pattern=[[T, 1], [1, T]], offset=0, scalar_offset=ix,
                                                   indirect_dim=0))
        else:
            nisa.dma_copy(dst=S, src=X["bias"].ap(pattern=[[0, H], [1, T]], offset=0, scalar_offset=ix,
                                                  indirect_dim=0))
        for lc in range(LC):
            nisa.nc_matmul(dst=PQ[:, lc, :], stationary=qr[:, lc * 128:(lc + 1) * 128], moving=IB[0:H, 0:H],
                           accumulate=False)
        qt = X["QT"][x]
        nisa.tensor_copy(dst=qt, src=PQ, engine=VE)
        if X["qkw"]:
            kta = X["KTA"][x]
            NG = T // 512
            for g in range(NG + 1):
                if g < NG:
                    for b in range(4):
                        i = g * 4 + b
                        ch = i // KP
                        t = i % KP
                        pt = PT[i % 2]
                        for lc in range(LC):
                            c0 = t * R + lc * 128
                            nisa.nc_matmul(dst=pt[:, lc, :], stationary=Kv[ch][:, c0:c0 + 128], moving=IB,
                                           accumulate=False)
                        if i % 2 == 0:
                            nisa.tensor_copy(dst=kta[:, :, i * 128:(i + 1) * 128], src=pt, engine=VE)
                        else:
                            nisa.activation(dst=kta[:, :, i * 128:(i + 1) * 128], op=nl.copy, data=pt)
                if g >= 1:
                    j = (g - 1) * 512
                    ps = PS[(g - 1) % 2]
                    for lc in range(LC):
                        nisa.nc_matmul(dst=ps, stationary=qt[:, lc, :], moving=kta[:, lc, j:j + 512],
                                       accumulate=True if lc > 0 else False)
                    nisa.nc_matmul(dst=ps, stationary=X["RSC"][:, 0:H], moving=br[:, j:j + 512], accumulate=True)
                    nisa.activation(dst=S[:, j:j + 512], op=nl.copy, data=ps, scale=X["scale"])
        else:
            for ch in range(NCH):
                for t in range(KP):
                    i = ch * KP + t
                    pt = PT[i % 2]
                    for lc in range(LC):
                        c0 = t * R + lc * 128
                        nisa.nc_matmul(dst=pt[:, lc, :], stationary=Kv[ch][:, c0:c0 + 128], moving=IB,
                                       accumulate=False)
                    kt = X["KT"][i % KTR]
                    if i % 2 == 0:
                        nisa.tensor_copy(dst=kt, src=pt, engine=VE)
                    else:
                        nisa.activation(dst=kt, op=nl.copy, data=pt)
                    ps = PS[i % 2]
                    for lc in range(LC):
                        nisa.nc_matmul(dst=ps[:, 0:128], stationary=qt[:, lc, :], moving=kt[:, lc, :],
                                       accumulate=(lc > 0))
                    j = i * 128
                    nisa.scalar_tensor_tensor(dst=S[:, j:j + 128], data=ps[:, 0:128], op0=nl.multiply,
                                              operand0=X["scale"], op1=nl.add, operand1=S[:, j:j + 128])
        mx, nmx, den, rden = X["MX"][x], X["NMX"][x], X["DEN"][x], X["RDEN"][x]
        nisa.tensor_reduce(dst=mx, op=nl.maximum, data=S, axis=1)
        nisa.tensor_scalar(dst=nmx, data=mx, op0=nl.multiply, operand0=-1.0, engine=VE)
        P = X["P"][x]
        nisa.activation(dst=P, op=nl.exp, data=S, bias=nmx, scale=1.0, reduce_op=nl.add, reduce_res=den,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        for ch in range(NCH):
            for t in range(KP):
                i = ch * KP + t
                j = i * 128
                g = 512 // H
                pp = PP[:, (i % g) * H:(i % g + 1) * H]
                nisa.nc_matmul(dst=pp, stationary=P[:, j:j + 128], moving=IB[0:H, 0:H], accumulate=False)
                pT = X["PTb"][i % 2]
                nisa.activation(dst=pT, op=nl.copy, data=pp)
                nisa.nc_matmul(dst=PO, stationary=pT, moving=Kv[ch][:, t * R:(t + 1) * R], accumulate=(i > 0))
        nisa.reciprocal(dst=rden, data=den)
        ob = X["OB"][x]
        nisa.tensor_scalar(dst=ob, data=PO, op0=nl.multiply, operand0=rden, engine=VE)
        nisa.dma_copy(dst=X["o"].ap(pattern=[[R, H], [1, R]], offset=0, scalar_offset=ix, indirect_dim=0), src=ob)
        if X["want_lse"]:
            ls = X["LS"][x]
            nisa.activation(dst=ls, op=nl.log, data=den)
            nisa.tensor_tensor(dst=ls, data1=ls, data2=mx, op=nl.add, engine=VE)
            nisa.dma_copy(dst=X["lse"].ap(pattern=[[1, H], [1, 1]], offset=0, scalar_offset=ix, indirect_dim=0),
                          src=ls)

    def _head(pair, T: int, Tm: int):
        """The first T columns of each of a pair of [p, Tm] tiles (the tiles themselves at T = Tm)."""
        if T == Tm:
            return pair
        return [pair[0][:, 0:T], pair[1][:, 0:T]]

    def _head3(pair, T: int, Tm: int):
        """_head of a pair of [p, LC, Tm] tiles."""
        if T == Tm:
            return pair
        return [pair[0][:, :, 0:T], pair[1][:, :, 0:T]]

    def _class_loop(X, rows_t, bias, idx, n_iter):
        """One class's device loop: n_iter iterations of RPI rows, row (it, k) being idx[it, k]."""
        NCH = rows_t.shape[2]
        T = NCH * KP * 128
        Tm = X["Tm"]
        X["NCH"] = NCH
        X["T"] = T
        X["rows_t"] = rows_t
        X["bias"] = bias
        X["RS"] = _pair((128, NCH), I32)
        X["S"] = _head(X["Sm"], T, Tm)  # the class's first T tokens of the largest class's tiles
        X["P"] = _head(X["Pm"], T, Tm)
        if X["qkw"]:
            X["KTA"] = _head3(X["KTAm"], T, Tm)
            X["BR"] = _head(X["BRm"], T, Tm)
        rpi = X["rpi"]
        cnt = _sb((1, 1), I32)
        nisa.dma_copy(dst=cnt, src=n_iter)
        rt = nisa.register_alloc()
        nisa.register_load(rt, cnt)
        IX = X["IX"]

        def body(it):
            X["psum"] = _psum(X["H"], X["LC"])
            # the iteration's RPI row indices, one DMA (a DMA per row put its latency before each row's loads, and an
            # on-chip copy per row from the whole list was no faster: tools/probe_cp_slots_x.py, kiln-lc2-k1)
            nisa.dma_copy(dst=IX, src=idx.ap(pattern=[[rpi, 1], [1, rpi]], offset=0, scalar_offset=it,
                                             indirect_dim=0))
            for k in range(rpi):
                _row_x(X, IX[0:1, k:k + 1], k)

        nl.fori_loop(0, rt, body)

    @nki.jit
    def kiln_dsa_slots_x_kernel(q_lat, kc, rows_a, bias_a, idx_a, n_a, rows_b, bias_b, idx_b, n_b, identb,
                                scale: float, fp8: int, want_lse: int, rev: int, qkw: int = 1):
        """q_lat bf16 [T, H, R]; kc as kiln_dsa_slots_kernel's; per class (a, b): rows_t int32 [T, 128, NCH] (slot
        ch 128 + p of row r at [r, p, ch]), bias fp32 [T, NCH KP 128] (dsa_slots' per-row token order), idx int32
        [NI, RPI] the class's rows (first its n, then repeats of a row of it), n int32 [1, 1] the device-loop
        iterations. Returns o fp32 [T, H, R] (and lse fp32 [T, H, 1]), every row written by the class naming it."""
        Tq, H, R = q_lat.shape
        rpi = idx_a.shape[1]
        LC = R // 128
        Tm = max(rows_a.shape[2], rows_b.shape[2]) * KP * 128
        NCHm = Tm // (KP * 128)
        o = nl.ndarray((Tq, H, R), dtype=F32, buffer=nl.shared_hbm)
        lse = None
        if want_lse:
            lse = nl.ndarray((Tq, H, 1), dtype=F32, buffer=nl.shared_hbm)
        KD = kc.dtype
        if fp8:
            KD = nl.float8_e4m3
        X = dict(H=H, R=R, LC=LC, Tm=Tm, rpi=rpi, q_lat=q_lat, kc=kc, o=o, lse=lse, want_lse=want_lse, qkw=qkw)
        IB = _sb((128, 128), BF16)
        nisa.dma_copy(dst=IB, src=identb)
        X["IB"] = IB
        X["scale"] = scale
        X["K"] = []
        for _ in range(2):
            kv = []
            for _ in range(NCHm):
                kv.append(_sb((128, KP * R), KD))
            X["K"].append(kv)
        X["KT"] = []
        if qkw:
            X["KTAm"] = _pair((128, LC, Tm), BF16)
            X["BRm"] = _pair((1, Tm), F32)
            X["RSC"] = _sb((1, 128))
            nisa.memset(dst=X["RSC"], value=1.0 / scale)
        else:
            for _ in range(KTR):
                X["KT"].append(_sb((128, LC, 128), BF16))
        X["QR"] = _pair((H, R), BF16)
        X["QT"] = _pair((128, LC, H), BF16)
        X["Sm"] = _pair((H, Tm), F32)
        X["Pm"] = _pair((H, Tm), BF16)
        X["OB"] = _pair((H, R), F32)
        X["MX"] = _pair((H, 1), F32)
        X["NMX"] = _pair((H, 1), F32)
        X["DEN"] = _pair((H, 1), F32)
        X["RDEN"] = _pair((H, 1), F32)
        X["LS"] = _pair((H, 1), F32)
        X["PTb"] = _pair((128, H), BF16)
        X["IX"] = _sb((1, rpi), I32)
        _class_loop(X, rows_a, bias_a, idx_a, n_a)
        _class_loop(X, rows_b, bias_b, idx_b, n_b)
        if want_lse:
            return o, lse
        return o
else:
    kiln_dsa_slots_x_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source and dsa_slots.REV (its row code is a copy of dsa_slots')."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_dsa_slots_x_kernel = None", a)
    return zlib.crc32(src[a:b].encode() + str(dsa_slots.REV).encode())


REV = _kernel_rev()


def _class_inputs(rows: torch.Tensor, bias: torch.Tensor, idx: torch.Tensor, n: torch.Tensor, rpi: int):
    """One class's kernel inputs: rows_t, bias in the kernel's layouts, the index list padded to whole iterations
    (the entries from n on repeat entry n - 1, so the extra rows are the class's own), the iteration count."""
    T, NS = rows.shape
    NCH = NS // 128
    rows_t = rows.to(torch.int32).reshape(T, NCH, 128).permute(0, 2, 1).contiguous()
    bias_t = bias.float().reshape(T, NCH, 128, KP).permute(0, 1, 3, 2).reshape(T, NCH * KP * 128).contiguous()
    L = idx.numel()
    Lp = -(-L // rpi) * rpi
    ix = idx.reshape(L).to(torch.int64)
    nn = n.reshape(1).to(torch.int64)
    if Lp != L:
        ix = torch.cat([ix, ix.new_zeros(Lp - L)])
    last = torch.gather(ix, 0, (nn - 1).clamp(min=0))
    ix = torch.where(torch.arange(Lp, device=ix.device) < nn, ix, last).to(torch.int32).reshape(Lp // rpi, rpi)
    it = torch.floor((n.reshape(1, 1).to(torch.float32) + (rpi - 1)) * (1.0 / rpi)).to(torch.int32)
    return rows_t, bias_t, ix, it


def attend(q_lat: torch.Tensor, kc: torch.Tensor, classes, scale: float):
    """(o [T, H, R], lse [T, H]) fp32 of two slot classes (rows [T, NS] int, NS a multiple of 128; bias [T, NS, KP]
    fp32; idx [T] int, the class's rows first; n [1] int) in one call: dsa_slots_n.attend of each class's rows idx[:n]
    put at those rows (emulate() on the host). Rows no class names are unwritten on the device."""
    kc2 = kc.reshape(kc.shape[0], -1)
    if q_lat.device.type == "cpu":
        return emulate(q_lat, kc2, classes, scale)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if kiln_dsa_slots_x_kernel is None:
        raise RuntimeError("the NKI DSA slots kernel needs the nki package (the Neuron venv)")
    if len(classes) != 2:
        raise ValueError(f"dsa_slots_x: two slot classes per call, not {len(classes)}")
    T, H, R = q_lat.shape
    for rows, _, _, _ in classes:
        if rows.shape[1] % 128:
            raise ValueError(f"dsa_slots_x: {rows.shape[1]} slots per row is not a multiple of 128")
    (ra, ba, ia, na), (rb, bb, ib, nb) = (_class_inputs(*c, dsa_slots.RPI) for c in classes)
    fp8 = kc2.dtype == torch.float8_e4m3fn
    kcp = kc2.reshape(kc2.shape[0] // KP, KP * R)
    eye = torch.eye(128, device=q_lat.device).to(torch.bfloat16)
    o, ls = wrap_nki(kiln_dsa_slots_x_kernel)[platform.nki_grid()](
        q_lat=q_lat.to(torch.bfloat16).contiguous(), kc=kcp, rows_a=ra, bias_a=ba, idx_a=ia, n_a=na, rows_b=rb,
        bias_b=bb, idx_b=ib, n_b=nb, identb=eye, scale=float(scale), fp8=int(fp8), want_lse=1, rev=REV,
        qkw=dsa_slots.QKW)
    return o, ls.reshape(T, H)
