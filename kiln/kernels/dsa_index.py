"""The pooled DSA indexer's decode scores at long context (GLM-5.3-Flash, models/glm5_next.py) as one NKI kernel:
every complete pool's score for B decode rows, reading each row's pool keys straight from the paged pool-key cache.

Why: the decode step's selection scores every complete pool of the context (glm5_next.pool_index). At the G1 bucket
(8448 keys, 2112 pools) that is small, but at 1M context a row has 262,144 pools per DSA layer: 64 MB of bf16 pool
keys per row per layer (11 layers: 0.74 GB per row per step). The XLA form gathers the keys by page into a [B, P, 128]
tensor, then materialises [B, Hi, P] fp32 scores (32 MB per row per layer) before the weighted head sum, so it moves
several times the key bytes. Here 128 pages of a row (8 pools of 128 keys each, 2 KB) are gathered into SBUF by one
indirect DMA, each pool's keys put on the partitions' other axis by the tensor engine (against a bf16 identity, exact
in the fp32 PSUM), scored against the row's Hi queries, and reduced to one score per pool on the vector engine; only
[B, P] fp32 scores leave the chip (4 bytes per pool against the key's 256).

What it computes, per decode row b and pool j (the arithmetic of glm5_next.pool_index, with Di ** -0.5 folded into w):
score[b, j] = sum_h w[b, h] relu(q[b, h] . k_j) + cand[b, j], with k_j the pool's key (bf16) and cand 0 for a complete
visible pool and NEG_INF for the rest. Products are bf16 x bf16 accumulated in fp32 (exact per product), the head sum
in fp32: the same values as the XLA form up to the order of the fp32 sums.

Layout and order: the pool-key cache (models/mla.py pool_key_width: pool i's key in the 4 token slots of its tokens)
viewed as page rows [Npages, PPP D] holds a page's PPP = 8 pools in order. Group g of row b gathers page p NG + g of the
row onto partition p (NG = pages / 128), so pool j of that page is pool p NT + g PPP + j of the row (NT = NG PPP), and
the row's [128, NT] score tile in SBUF is its P scores in pool order: out [B, P] is one contiguous DMA per row, and it
is the piece layout of kernels/dsa_topk.py (g = 128 pieces of W = NT), so the selection can follow on the same tile.

Why pages and not pools: an indirect gather costs about the same per descriptor up to 2 KB on trn1 (software DGE).
Measured with this kernel's first form (one 256-byte pool row per partition, tools/probe_dsa_index.py, trn1.2xlarge,
SDK 2.32, 2026-10-05): 20 GB/s at every shape (3.3 ms per row at 262,144 pools), where kernels/dsa_decode.py's 128 x
2 KB pool gathers reach 185 GB/s (docs/neuron-notes.md "The attention kernels against their floors").
"""

from __future__ import annotations

import os

import torch

NEG_INF = -1e30  # models/decoder.NEG_INF
NR = int(os.environ.get("KILN_DSA_INDEX_NR", 4))  # gathered page tiles in flight (256 KB each)
QR = int(os.environ.get("KILN_DSA_INDEX_QR", 4))  # quarters (512 pools) in flight between the engines
D = 128  # index_head_dim (GLM-5.3-Flash)
DGE = int(os.environ.get("KILN_DSA_INDEX_DGE", "0"))  # the page gathers' DGE: 0 compiler's choice, 1 software, 2 hardware


def emulate(q: torch.Tensor, w: torch.Tensor, pkc: torch.Tensor, prow: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
    """The kernel's arithmetic in torch (CPU): q [B, Hi, D] (bf16 values), w [B, Hi] fp32 (the head weights times
    Di ** -0.5), pkc [N, D] the pool-key cache as pool rows, prow [B, P] int each pool's row, cand [B, P] fp32 (0 /
    NEG_INF) -> scores [B, P] fp32."""
    k = pkc[prow.long()].float()  # [B, P, D]
    s = torch.einsum("bhd,bpd->bhp", q.float(), k)
    return torch.einsum("bh,bhp->bp", w.float(), torch.relu(s)) + cand.float()


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None


if nki is not None:
    F32, BF16, I32 = nl.float32, nl.bfloat16, nl.int32

    def _programs(spl: int = 1):
        """(programs, this program) of the launch grid: at LNC=2 (grid 2) the kernel is traced once per physical core
        (kernels/dsa_topk.py _programs); grid 1: (1, 0)."""
        return (nl.num_programs(axes=0), nl.program_id(axis=0)) if spl and nl.program_ndim() != 0 else (1, 0)

    # Pipeline stages of kiln_dsa_index_kernel, at module level (the device tracer takes inner functions only as loop
    # bodies); X holds the kernel's tensors, rings and work lists.
    def _row_setup(X, i, b):
        x = i % 3
        B, NG, Hi, NT, P = X["B"], X["NG"], X["Hi"], X["NT"], X["P"]
        nisa.dma_copy(dst=X["OF"][x], src=X["ppg"].ap(pattern=[[B * NG, 128], [1, NG]], offset=b * NG))
        nisa.dma_copy(dst=X["WB"][x], src=X["wb"].ap(pattern=[[4 * Hi, 128], [1, 4 * Hi]], offset=b * 128 * 4 * Hi))
        nisa.dma_copy(dst=X["C"][x], src=X["cand"].ap(pattern=[[NT, 128], [1, NT]], offset=b * P))

    def _gather(X, k):
        i = X["G"][k][0]
        g = X["G"][k][2]
        NG, PPP, nr, pkc = X["NG"], X["PPP"], X["nr"], X["pkc"]
        kk = X["K"][k % nr]
        if X["dbg"] == 1:
            nisa.dma_copy(dst=kk, src=pkc.ap(pattern=[[PPP * D, 128], [1, PPP * D]], offset=g * 128 * PPP * D))
        elif X["dbg"] != 2 or g < nr:
            nisa.dma_copy(dst=kk, src=pkc.ap(pattern=[[PPP * D, 128], [1, PPP * D]], offset=0,
                                             vector_offset=X["OF"][i % 3].ap(pattern=[[NG, 128], [1, 1]], offset=g),
                                             indirect_dim=0), dge_mode=X["dmode"])

    def _stage_a(X, n):  # 4 of a page's pools: [128 pages, D] -> [D, 128 pages], copied to bf16
        k = X["QS"][n][0]
        h4 = X["QS"][n][4]
        qr = X["qr"]
        kk = X["K"][k % X["nr"]]
        pt = X["PT"][n % qr]
        for j in range(4):
            c0 = (4 * h4 + j) * D
            nisa.nc_matmul(dst=pt[:, j * 128:(j + 1) * 128], stationary=kk[:, c0:c0 + D], moving=X["IB"],
                           accumulate=False)
        nisa.activation(dst=X["KT"][n % qr], op=nl.copy, data=pt)

    def _stage_b(X, n):  # scores [128 pages, Hi] = K Q^T over D per pool, relu times w, the head sum
        i = X["QS"][n][1]
        b = X["QS"][n][2]
        g = X["QS"][n][3]
        h4 = X["QS"][n][4]
        qr, Hi, PPP, NG, NT, P = X["qr"], X["Hi"], X["PPP"], X["NG"], X["NT"], X["P"]
        kt = X["KT"][n % qr]
        ps = X["PS"][n % qr]
        r = X["R"][n % qr]
        for j in range(4):
            nisa.nc_matmul(dst=ps[:, j * Hi:(j + 1) * Hi], stationary=kt[:, j * 128:(j + 1) * 128],
                           moving=X["QT"][:, b * Hi:(b + 1) * Hi], accumulate=False)
        nisa.scalar_tensor_tensor(dst=r, data=ps, op0=nl.maximum, operand0=0.0, op1=nl.multiply, operand1=X["WB"][i % 3])
        c = g * PPP + 4 * h4
        S = X["S"][i % 3]
        nisa.tensor_reduce(dst=S[:, c:c + 4], op=nl.add, data=r.reshape((128, 4, Hi)), axis=2)
        if g == NG - 1 and h4 == PPP // 4 - 1:  # the row's last quarter: its candidate bias, then out
            nisa.tensor_tensor(dst=S, data1=S, data2=X["C"][i % 3], op=nl.add, engine=nisa.vector_engine)
            nisa.dma_copy(dst=X["out"].ap(pattern=[[NT, 128], [1, NT]], offset=b * P), src=S)

    @nki.jit
    def kiln_dsa_index_kernel(qT, wb, pkc, ppg, cand, identb, rev: int, dge: int = 0, spl: int = 1, nr: int = 4,
                              qr: int = 4, dbg: int = 0):
        """qT bf16 [D, B Hi] (column b Hi + h: row b's query of head h); wb fp32 [B, 128, 4 Hi] (row b's head weights
        repeated 4 times along the free axis, the same on every partition); pkc bf16 [Npages, PPP D] the pool-key
        cache as page rows (PPP pools of D per page, pool order); ppg int32 [128, B, NG] the page of row b holding its
        pools (p NG + g) PPP .. + PPP - 1 at [p, b, g]; cand fp32 [B, P] (P = 128 NG PPP); identb bf16 [128, 128];
        rev: this module's kernel source revision; dge: the page gathers' descriptor generation (0 the compiler's
        choice, 1 software, 2 hardware: NeuronCore-v3+); spl: at LNC=2 (grid 2) the two programs take alternate rows;
        nr: page tiles in flight; qr: 512-pool quarters in flight between the engines (static arguments, so in the
        compile-cache key); dbg (probes only): 1 the page gathers as plain contiguous DMAs of the same size, 2 no
        gathers after each row's first (the engines' work alone).
        Returns fp32 [B, P] scores.

        Group g of a row gathers 128 pages, page p NG + g on partition p (one indirect DMA of 128 descriptors of PPP D
        bf16 = 2 KB: an indirect gather costs about the same per descriptor up to 2 KB, so whole pages, not pools); its
        pool j sits on partition p at column (g PPP + j) of the row's [128, NT] score tile, which is pool
        p NT + g PPP + j: the row's scores in pool order (NT = NG PPP)."""
        Dk, BH = qT.shape
        B = cand.shape[0]
        Hi = BH // B
        NG = ppg.shape[2]
        PPP = pkc.shape[1] // Dk
        NT = NG * PPP
        P = NT * 128
        assert Dk == D and PPP % 4 == 0 and wb.shape[2] == 4 * Hi and qr * 4 * Hi <= 512
        out = nl.ndarray((B, P), dtype=F32, buffer=nl.shared_hbm)
        IB = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=IB, src=identb)
        QT = nl.ndarray((D, BH), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=QT, src=qT)
        # rings, every tile allocated once (dsa_decode.py: tiles allocated inside unrolled loops overran SBUF); plain
        # loops, not comprehensions: the device tracer refuses those ("unsupported expression"; the simulator takes them)
        Kr, KTr, PTr, PSr, Rr, OFr, WBr, Sr, Cr = [], [], [], [], [], [], [], [], []
        for _ in range(qr):  # PSUM first: one 2 KB bank per partition each (transposes of a quarter: 512 pools)
            PTr.append(nl.ndarray((D, 512), dtype=F32, buffer=nl.psum))
        PSb = nl.ndarray((128, 512), dtype=F32, buffer=nl.psum)  # one bank: qr quarters' [128, 4 Hi] scores
        for i in range(qr):
            PSr.append(PSb[:, i * 4 * Hi:(i + 1) * 4 * Hi])
        for _ in range(nr):
            Kr.append(nl.ndarray((128, PPP * D), dtype=BF16, buffer=nl.sbuf))
        for _ in range(qr):
            KTr.append(nl.ndarray((D, 512), dtype=BF16, buffer=nl.sbuf))
            Rr.append(nl.ndarray((128, 4 * Hi), dtype=F32, buffer=nl.sbuf))
        for _ in range(3):  # per-row tiles: row i in slot i % 3 (the next rows' setup runs ahead of this one's tail)
            OFr.append(nl.ndarray((128, NG), dtype=I32, buffer=nl.sbuf))
            WBr.append(nl.ndarray((128, 4 * Hi), dtype=F32, buffer=nl.sbuf))
            Sr.append(nl.ndarray((128, NT), dtype=F32, buffer=nl.sbuf))
            Cr.append(nl.ndarray((128, NT), dtype=F32, buffer=nl.sbuf))
        dmode = nisa.dge_mode.hwdge if dge == 2 else (nisa.dge_mode.swdge if dge == 1 else nisa.dge_mode.unknown)
        npg, pid = _programs(spl)
        rows = []
        for b in range(B):
            if b % npg == pid:  # LNC: rows alternate between the two programs (physical cores)
                rows.append(b)
        # The work as a flat list of 512-pool quarters (row, page group, quarter of the page's pools), issued as a
        # software pipeline in program order (each engine runs its queue in order, so an unskewed loop made the tensor
        # engine wait for each quarter's copy before transposing the next): gathers PRE groups ahead, transposes and
        # their copy LAG quarters ahead of the scores and the head sums.
        # (no tuple targets in loops or assignments: the device tracer wants simple variables)
        G = []
        for i in range(len(rows)):
            for g in range(NG):
                G.append((i, rows[i], g))
        QS = []
        for k in range(len(G)):
            for h4 in range(PPP // 4):
                QS.append((k, G[k][0], G[k][1], G[k][2], h4))
        # Ahead distances that keep every row's tiles live in their 3 slots: the setup of the row PRE groups ahead never
        # overwrites a row whose last head sums (LAG quarters behind) are still to come (PRE <= NG, LAG <= PPP / 4).
        PRE = min(nr - 1, 2, NG)
        LAG = min(qr - 1, 2, PPP // 4)

        X = dict(B=B, NG=NG, Hi=Hi, NT=NT, P=P, PPP=PPP, nr=nr, qr=qr, dbg=dbg, dmode=dmode, ppg=ppg, wb=wb, cand=cand,
                 pkc=pkc, out=out, IB=IB, QT=QT, OF=OFr, WB=WBr, S=Sr, C=Cr, K=Kr, KT=KTr, PT=PTr, PS=PSr, R=Rr, G=G, QS=QS)
        if len(rows) > 0:
            _row_setup(X, 0, rows[0])
            for k in range(min(PRE, len(G))):
                if G[k][2] == 0 and G[k][0] > 0:
                    _row_setup(X, G[k][0], G[k][1])
                _gather(X, k)
        for n in range(len(QS) + LAG):
            if n < len(QS):
                k = QS[n][0]
                if QS[n][4] == 0 and k + PRE < len(G):  # the group PRE ahead
                    if G[k + PRE][2] == 0:
                        _row_setup(X, G[k + PRE][0], G[k + PRE][1])
                    _gather(X, k + PRE)
                _stage_a(X, n)
            if n >= LAG:
                _stage_b(X, n - LAG)
        if npg > 1:  # LNC: every program's rows written before either ends (kernels/dsa_topk.py)
            nisa.core_barrier(data=out, cores=(0, 1))
        return out
else:
    kiln_dsa_index_kernel = None


if nki is not None:  # the fp8 pool-key form (its own block and revision, REV8)
    def kiln_dsa_index_fp8_kernel(qT, wb, pkc, ppg, cand, identb, rev: int, dge: int = 0, spl: int = 1, nr: int = 4,
                              qr: int = 4, dbg: int = 0):
        """kiln_dsa_index_kernel on an fp8 (e4m3) pool-key cache: the minimal KV layout's pool keys (KILN_DSA_KV=minimal,
        models/mla.py: V holds them in the KV dtype). Each page row is gathered as it is stored (PPP D bytes: 1 KB at 8
        pools of 128) and goes through the transposes as the fp8 stationary operand against the bf16 identity (exact in
        the fp32 PSUM; the KT copy to bf16 is exact for every e4m3 value), so the cache is read once at half the bytes
        and never converted as a whole. A separate kernel so that the bf16 kernel's source, and so its graphs' keys
        (REV), are unchanged.
        qT bf16 [D, B Hi] (column b Hi + h: row b's query of head h); wb fp32 [B, 128, 4 Hi] (row b's head weights
        repeated 4 times along the free axis, the same on every partition); pkc bf16 [Npages, PPP D] the pool-key
        cache as page rows (PPP pools of D per page, pool order); ppg int32 [128, B, NG] the page of row b holding its
        pools (p NG + g) PPP .. + PPP - 1 at [p, b, g]; cand fp32 [B, P] (P = 128 NG PPP); identb bf16 [128, 128];
        rev: this module's kernel source revision; dge: the page gathers' descriptor generation (0 the compiler's
        choice, 1 software, 2 hardware: NeuronCore-v3+); spl: at LNC=2 (grid 2) the two programs take alternate rows;
        nr: page tiles in flight; qr: 512-pool quarters in flight between the engines (static arguments, so in the
        compile-cache key); dbg (probes only): 1 the page gathers as plain contiguous DMAs of the same size, 2 no
        gathers after each row's first (the engines' work alone).
        Returns fp32 [B, P] scores.

        Group g of a row gathers 128 pages, page p NG + g on partition p (one indirect DMA of 128 descriptors of PPP D
        bf16 = 2 KB: an indirect gather costs about the same per descriptor up to 2 KB, so whole pages, not pools); its
        pool j sits on partition p at column (g PPP + j) of the row's [128, NT] score tile, which is pool
        p NT + g PPP + j: the row's scores in pool order (NT = NG PPP)."""
        Dk, BH = qT.shape
        B = cand.shape[0]
        Hi = BH // B
        NG = ppg.shape[2]
        PPP = pkc.shape[1] // Dk
        NT = NG * PPP
        P = NT * 128
        assert Dk == D and PPP % 4 == 0 and wb.shape[2] == 4 * Hi and qr * 4 * Hi <= 512
        out = nl.ndarray((B, P), dtype=F32, buffer=nl.shared_hbm)
        IB = nl.ndarray((128, 128), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=IB, src=identb)
        QT = nl.ndarray((D, BH), dtype=BF16, buffer=nl.sbuf)
        nisa.dma_copy(dst=QT, src=qT)
        # rings, every tile allocated once (dsa_decode.py: tiles allocated inside unrolled loops overran SBUF); plain
        # loops, not comprehensions: the device tracer refuses those ("unsupported expression"; the simulator takes them)
        Kr, KTr, PTr, PSr, Rr, OFr, WBr, Sr, Cr = [], [], [], [], [], [], [], [], []
        for _ in range(qr):  # PSUM first: one 2 KB bank per partition each (transposes of a quarter: 512 pools)
            PTr.append(nl.ndarray((D, 512), dtype=F32, buffer=nl.psum))
        PSb = nl.ndarray((128, 512), dtype=F32, buffer=nl.psum)  # one bank: qr quarters' [128, 4 Hi] scores
        for i in range(qr):
            PSr.append(PSb[:, i * 4 * Hi:(i + 1) * 4 * Hi])
        for _ in range(nr):
            Kr.append(nl.ndarray((128, PPP * D), dtype=nl.float8_e4m3, buffer=nl.sbuf))
        for _ in range(qr):
            KTr.append(nl.ndarray((D, 512), dtype=BF16, buffer=nl.sbuf))
            Rr.append(nl.ndarray((128, 4 * Hi), dtype=F32, buffer=nl.sbuf))
        for _ in range(3):  # per-row tiles: row i in slot i % 3 (the next rows' setup runs ahead of this one's tail)
            OFr.append(nl.ndarray((128, NG), dtype=I32, buffer=nl.sbuf))
            WBr.append(nl.ndarray((128, 4 * Hi), dtype=F32, buffer=nl.sbuf))
            Sr.append(nl.ndarray((128, NT), dtype=F32, buffer=nl.sbuf))
            Cr.append(nl.ndarray((128, NT), dtype=F32, buffer=nl.sbuf))
        dmode = nisa.dge_mode.hwdge if dge == 2 else (nisa.dge_mode.swdge if dge == 1 else nisa.dge_mode.unknown)
        npg, pid = _programs(spl)
        rows = []
        for b in range(B):
            if b % npg == pid:  # LNC: rows alternate between the two programs (physical cores)
                rows.append(b)
        # The work as a flat list of 512-pool quarters (row, page group, quarter of the page's pools), issued as a
        # software pipeline in program order (each engine runs its queue in order, so an unskewed loop made the tensor
        # engine wait for each quarter's copy before transposing the next): gathers PRE groups ahead, transposes and
        # their copy LAG quarters ahead of the scores and the head sums.
        # (no tuple targets in loops or assignments: the device tracer wants simple variables)
        G = []
        for i in range(len(rows)):
            for g in range(NG):
                G.append((i, rows[i], g))
        QS = []
        for k in range(len(G)):
            for h4 in range(PPP // 4):
                QS.append((k, G[k][0], G[k][1], G[k][2], h4))
        # Ahead distances that keep every row's tiles live in their 3 slots: the setup of the row PRE groups ahead never
        # overwrites a row whose last head sums (LAG quarters behind) are still to come (PRE <= NG, LAG <= PPP / 4).
        PRE = min(nr - 1, 2, NG)
        LAG = min(qr - 1, 2, PPP // 4)

        X = dict(B=B, NG=NG, Hi=Hi, NT=NT, P=P, PPP=PPP, nr=nr, qr=qr, dbg=dbg, dmode=dmode, ppg=ppg, wb=wb, cand=cand,
                 pkc=pkc, out=out, IB=IB, QT=QT, OF=OFr, WB=WBr, S=Sr, C=Cr, K=Kr, KT=KTr, PT=PTr, PS=PSr, R=Rr, G=G, QS=QS)
        if len(rows) > 0:
            _row_setup(X, 0, rows[0])
            for k in range(min(PRE, len(G))):
                if G[k][2] == 0 and G[k][0] > 0:
                    _row_setup(X, G[k][0], G[k][1])
                _gather(X, k)
        for n in range(len(QS) + LAG):
            if n < len(QS):
                k = QS[n][0]
                if QS[n][4] == 0 and k + PRE < len(G):  # the group PRE ahead
                    if G[k + PRE][2] == 0:
                        _row_setup(X, G[k + PRE][0], G[k + PRE][1])
                    _gather(X, k + PRE)
                _stage_a(X, n)
            if n >= LAG:
                _stage_b(X, n - LAG)
        if npg > 1:  # LNC: every program's rows written before either ends (kernels/dsa_topk.py)
            nisa.core_barrier(data=out, cores=(0, 1))
        return out
else:
    kiln_dsa_index_fp8_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of this module's kernel source, passed as the static argument `rev` (LNL's compile-cache key does not
    include NKI kernel source: CLAUDE.md)."""
    import zlib

    src = open(__file__).read()
    a = src.index("try:  # the Neuron venv")
    b = src.index("    kiln_dsa_index_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV = _kernel_rev()


def _kernel_rev8() -> int:
    """kiln_dsa_index_fp8_kernel's source revision (its block; the helpers it shares are in REV's)."""
    import zlib

    src = open(__file__).read()
    a = src.index("if nki is not None:  # the fp8 pool-key form")
    b = src.index("    kiln_dsa_index_fp8_kernel = None", a)
    return zlib.crc32(src[a:b].encode())


REV8 = _kernel_rev8()


def page_groups(table: torch.Tensor) -> torch.Tensor:
    """ppg [128, B, NG] int32 for the kernel from the rows' page table [B, NPg] (NPg = 128 NG pages per row): the page
    of group g on partition p is page p NG + g of the row. Shared by every DSA layer of a step."""
    B, NPg = table.shape
    if NPg % 128:
        raise ValueError(f"dsa_index: {NPg} pages per row, the kernel needs a multiple of 128")
    return table.reshape(B, 128, NPg // 128).permute(1, 0, 2).contiguous().to(torch.int32)


def scores(q: torch.Tensor, w: torch.Tensor, pkc: torch.Tensor, ppg: torch.Tensor, cand: torch.Tensor,
           dge: int | None = None) -> torch.Tensor:
    """[B, P] fp32 scores (the kernel on a Neuron device, emulate() elsewhere): q [B, Hi, D], w [B, Hi] fp32 (head weights
    times Di ** -0.5), pkc [Npages, PPP D] the pool-key cache as page rows (bf16, or fp8 e4m3: the minimal layout's,
    read as stored by kiln_dsa_index_fp8_kernel), ppg [128, B, NG] int32 (page_groups), cand [B, P] fp32."""
    B, Hi, _ = q.shape
    if q.device.type == "cpu":
        table = ppg.permute(1, 0, 2).reshape(B, -1)
        PPP = pkc.shape[1] // D
        rows = (table.long().unsqueeze(-1) * PPP + torch.arange(PPP)).reshape(B, -1)
        return emulate(q, w, pkc.reshape(-1, D), rows, cand)
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if kiln_dsa_index_kernel is None:
        raise RuntimeError("the NKI DSA index kernel needs the nki package (the Neuron venv)")
    qT = q.to(torch.bfloat16).reshape(B * Hi, -1).t().contiguous()
    wb = w.float().repeat(1, 4).unsqueeze(1).expand(B, 128, 4 * Hi).contiguous()
    eye = torch.eye(128, device=q.device).to(torch.bfloat16)
    if pkc.dtype == torch.float8_e4m3fn:  # read as stored (kiln_dsa_index_fp8_kernel), never converted as a whole
        kern, rev = kiln_dsa_index_fp8_kernel, REV8
    else:
        kern, rev = kiln_dsa_index_kernel, REV
        pkc = pkc.to(torch.bfloat16)
    return wrap_nki(kern)[platform.nki_grid()](
        qT=qT, wb=wb, pkc=pkc, ppg=ppg, cand=cand.float(), identb=eye, rev=rev,
        dge=int(DGE if dge is None else dge), spl=1, nr=NR, qr=QR, dbg=int(os.environ.get("KILN_DSA_INDEX_DBG", 0)))  # rows are independent: at LNC=2 each physical core takes every other row
