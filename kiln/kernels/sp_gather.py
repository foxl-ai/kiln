"""The sequence-parallel row gather as an all_gather issued from an NKI kernel (nki.collectives), behind
KILN_SP_GATHER=nki: every rank's rows [r, W] -> [n r, W] in rank order, over the world (models/decoder.py _sp_gather)
or inside each attention group (_sp_group_gather).

Why: the served form is a zero-padded all-reduce (x in its block of a zero [n, r, W], summed over the ranks), because
a graph holding an XLA all-gather (`_c10d_functional.all_gather_into_tensor`) breaks when a later process loads its
NEFF from the compile cache ("replica group signature mismatch", docs/neuron-notes.md "A cached all-gather NEFF breaks
in the next process"). A ring all-reduce of the 32 MiB buffer moves ~2 (n - 1) / n of it per rank, an all-gather
(n - 1) / n. Measured alone at 32 ranks on trn1.32xlarge (SDK 2.32, nki 0.6.0, `NEURON_RT_DISABLE_EXECUTION_BARRIER=1`,
tools/probe_nki_cc.py, 2026-10-05): the zero-padded all-reduce of [4096, 4096] bf16 2.27 ms, the NKI all_gather 1.58 ms
with its copies in and out (~0.55 ms of them), exact, and its cached NEFF reloads in a later process.

nki.collectives constraint (neuronx-cc 2.27 birverifier checkCollective): "Collective instruction cannot read IO
tensors", and writing the kernel's output directly fails to compile too, so the rows are copied into private HBM
scratch, gathered there and copied to the output (HBM -> HBM DMAs). With chunks > 1 the rows are gathered in that many
row slices, each slice's copy-out (one strided DMA: rank j's slice to rows j r + c rc ..) overlapping the next slice's
collective.

Exactness: a gather moves bytes; the output equals the zero-padded all-reduce's bit for bit (one rank's value plus
zeros per element).
"""

from __future__ import annotations

import os

import torch

# KILN_SP_GATHER: xla (default: the zero-padded all-reduce), nki (this kernel for the world gather only) or nki-all (also
# inside the attention groups). Measured with the runtime's execution barrier on (the default), standalone block graphs
# at the served shape (tools/profile_layer.py --what hcblocks --part-layers 3 4 --sp, 32 ranks, trn1.32xlarge, SDK 2.32,
# 2026-10-05): the SP FFN block (world gather) 14.24 -> 13.22 ms (layer 3) and 14.34 -> 13.37 ms (layer 4), the SP
# attention block with group collectives (8-rank group gather) 8.86 -> 10.09 and 6.14 -> 7.10 ms. So the group gather
# stays the zero-padded group all-reduce unless nki-all. KILN_SP_GATHER_CHUNKS: row slices.
# Unset: nki on trn1, where the serving A/Bs and the gates ran (G64 156.1 -> 164.7, F0 133.6 -> 140.1, G64 + EPLB 167.1 ->
# 177.3 out tok/s; wikitext-2 through check_ppl --chunk 1024 -0.54759 -> -0.54730, signed +0.00029 +/- 0.00054;
# docs/neuron-notes.md "Collectives issued from an NKI kernel"), xla elsewhere (trn2, inf2 and a host without a Neuron
# device: the CPU tests take the zero-padded all-reduce, which is the same rows bit for bit).
SP_GATHER_FAMILIES = ("trn1",)


def _default_mode() -> str:
    from .. import platform

    t = platform.target()
    return "nki" if t is not None and platform.family_of(t) in SP_GATHER_FAMILIES else "xla"


MODE = os.environ.get("KILN_SP_GATHER") or _default_mode()
if MODE not in ("xla", "nki", "nki-all"):
    raise ValueError(f"KILN_SP_GATHER must be xla, nki or nki-all, not {MODE!r}")
CHUNKS = int(os.environ.get("KILN_SP_GATHER_CHUNKS", "1"))
# Rows per rank from which the kernel replaces the zero-padded all-reduce: a prefill chunk's 128 (4096 rows at tp 32),
# not a decode call's 2 (KILN_DECODE_SP: 64 rows), so the decode graphs keep their keys until measured.
MIN_ROWS = int(os.environ.get("KILN_SP_GATHER_MIN_ROWS", "16"))
# Row width from which it does: the hidden rows (4096), not the routing's [r, 2k] fp32 (16 wide). In the replay of a
# 4096-row G64 prefill call (tools/util_report.py, 32 ranks' captured inputs, trn1.32xlarge, 2026-10-05) the kernel's
# routing gathers waited 0.218 ms each to start (42 of them, 9.9 ms per call) against 0.024 ms for the zero-padded
# all-reduce of the same 256 KB, while the hidden-row gathers took 0.34 ms against 4 x 0.41 ms. In serving the choice is
# neutral (G64 164.7 with the kernel, 164.2 without; prefill call 0.4765 / 0.4782 s): that wait is rank skew, which the
# next collective waits out otherwise. The zero-padded all-reduce stays, as the longer-proven form.
MIN_WIDTH = int(os.environ.get("KILN_SP_GATHER_MIN_WIDTH", "256"))

try:  # the Neuron venv
    import nki
    import nki.collectives as ncc
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:

    def _groups(world: int, gsize: int):
        """[[0 .. gsize - 1], [gsize .. 2 gsize - 1], ..] as list literals (the NKI tracer takes no list(range()))."""
        out = []
        for g in range(world // gsize):
            ranks = []
            for i in range(gsize):
                ranks.append(g * gsize + i)
            out.append(ranks)
        return ncc.ReplicaGroup(out)

    @nki.jit
    def kiln_sp_gather_kernel(x, world: int, gsize: int, chunks: int, rev: int):
        """x [r, W] this rank's rows -> [gsize r, W]: its replica group's rows in group-rank order. world: every rank of
        the graph (the replica groups are the gsize-rank blocks of it); chunks: row slices gathered one by one; rev: this
        module's kernel source revision."""
        r, W = x.shape
        rc = r // chunks
        out = nl.ndarray((gsize * r, W), dtype=x.dtype, buffer=nl.shared_hbm)
        xs = nl.ndarray((r, W), dtype=x.dtype, buffer=nl.private_hbm)
        nisa.dma_copy(dst=xs, src=x)
        rg = _groups(world, gsize)
        if chunks == 1:
            gs = nl.ndarray((gsize * r, W), dtype=x.dtype, buffer=nl.private_hbm)
            ncc.all_gather(srcs=[xs], dsts=[gs], replica_group=rg, collective_dim=0)
            nisa.dma_copy(dst=out, src=gs)
            return out
        for c in range(chunks):
            gc = nl.ndarray((gsize * rc, W), dtype=x.dtype, buffer=nl.private_hbm)
            ncc.all_gather(srcs=[xs[c * rc:(c + 1) * rc, :]], dsts=[gc], replica_group=rg, collective_dim=0)
            # rank j's slice c (rows j rc .. of gc) -> rows j r + c rc .. of the output
            nisa.dma_copy(dst=out.ap(pattern=[[r * W, gsize], [W, rc], [1, W]], offset=c * rc * W), src=gc)
        return out

else:
    kiln_sp_gather_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source, passed as the static argument `rev` (LNL's compile-cache key does not
    include NKI kernel source: CLAUDE.md)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_sp_gather_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev()

# --- the LNC = 2 form (trn2): a separate kernel, so the grid-1 kernel's source and REV (and every trn1 key) stay put ---

if nki is not None:

    @nki.jit
    def kiln_sp_gather_lnc_kernel(x, world: int, gsize: int, rev: int):
        """kiln_sp_gather_kernel (chunks 1) for a logical NeuronCore of two physical cores (trn2 at LNC = 2, grid 2): only
        program 0 copies the rows in, issues the all_gather and copies the gathered rows out; program 1 issues nothing
        (two programs issuing one collective is not a form nki documents), and a core barrier on the output orders both
        programs' exit after the write. Not measured yet (2026-10-05): the trn2 agent's A/B is its first run."""
        r, W = x.shape
        out = nl.ndarray((gsize * r, W), dtype=x.dtype, buffer=nl.shared_hbm)
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        if pid == 0:
            xs = nl.ndarray((r, W), dtype=x.dtype, buffer=nl.private_hbm)
            nisa.dma_copy(dst=xs, src=x)
            gs = nl.ndarray((gsize * r, W), dtype=x.dtype, buffer=nl.private_hbm)
            ncc.all_gather(srcs=[xs], dsts=[gs], replica_group=_groups(world, gsize), collective_dim=0)
            nisa.dma_copy(dst=out, src=gs)
        if npg > 1:
            nisa.core_barrier(data=out, cores=(0, 1))
        return out

else:
    kiln_sp_gather_lnc_kernel = None


def _lnc_rev() -> int:
    import zlib

    src = open(__file__).read()
    a = src.index("# --- the LNC = 2 form (trn2)")
    b = src.index("    kiln_sp_gather_lnc_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV_LNC = _lnc_rev()


def enabled(x: torch.Tensor, group: bool = False) -> bool:
    """Whether the kernel gathers x (group: inside the attention groups, _sp_group_gather)."""
    return (MODE in (("nki-all",) if group else ("nki", "nki-all")) and x.device.type != "cpu" and x.dim() == 2
            and x.shape[0] >= MIN_ROWS and x.shape[1] >= MIN_WIDTH and kiln_sp_gather_kernel is not None)


def gather(x: torch.Tensor, world: int, gsize: int) -> torch.Tensor:
    """Every rank's x [r, W] of this rank's replica group (the gsize-rank blocks of the world), [gsize r, W], by the
    kernel (enabled() must hold)."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    r = x.shape[0]
    grid = platform.nki_grid()
    if grid > 1:  # trn2 at LNC = 2: the program-0 form
        return wrap_nki(kiln_sp_gather_lnc_kernel)[grid](x=x.contiguous(), world=world, gsize=gsize, rev=REV_LNC)
    chunks = CHUNKS if CHUNKS > 1 and r % CHUNKS == 0 else 1
    return wrap_nki(kiln_sp_gather_kernel)[grid](x=x.contiguous(), world=world, gsize=gsize, chunks=chunks, rev=REV)
