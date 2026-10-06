"""Long-context pooled DSA (models/dsa_long.py): the two-level exact selection against the dense definition
on adversarial ties, the slots against the bucketed path's token selection, and GLM-5.3-Flash end to end on
the long path against transformers and against the bucketed path. The engine tests need transformers >= 5.18
(tests/test_glm5_next.py's builder); the selection and slot tests need nothing."""

import pytest
import torch

from kiln.models import dsa_long

NEG = dsa_long.NEG_INF


def _ref_mask(sc: torch.Tensor, keep: int) -> torch.Tensor:
    """The tie rule written out (dsa_select.reference_mask restricted to candidates): per row, the candidates
    (> VISIBLE) in descending score, ties by ascending index, the first keep."""
    N, P = sc.shape
    out = torch.zeros(N, P, dtype=torch.bool)
    for n in range(N):
        cand = [p for p in range(P) if sc[n, p] > dsa_long.VISIBLE]
        cand.sort(key=lambda p: (-float(sc[n, p]), p))
        out[n, cand[:keep]] = True
    return out


def _as_mask(pools, cnt, P):
    N = pools.shape[0]
    m = torch.zeros(N, P, dtype=torch.bool)
    for n in range(N):
        c = int(cnt[n])
        idx = pools[n, :c]
        assert torch.all(idx[1:] > idx[:-1]), "pools must be ascending and distinct"
        m[n, idx] = True
    return m


def _cases():
    """(name, scores [N, P], keep): random, integer ties, exact zeros, ties straddling sub-block boundaries,
    the keep-th score tied in several sub-blocks whose maximum it is, candidate prefixes shorter than keep,
    exactly keep, and fewer visible sub-blocks than keep."""
    g = torch.Generator().manual_seed(0)
    out = []
    for P in (37, 64, 257, 1000, 4096):
        for keep in (1, 4, 16, 64):
            if keep >= P:
                continue
            N = 6
            npool = torch.tensor([0, 1, keep, keep + 1, P // 2, P])
            cand = torch.arange(P).view(1, P) < npool.view(N, 1)
            def put(s, cand=cand):
                return torch.where(cand, s, NEG)
            out.append((f"randn P{P} k{keep}", put(torch.randn(N, P, generator=g)), keep))
            out.append((f"int P{P} k{keep}", put(torch.randint(0, 5, (N, P), generator=g).float()), keep))
            out.append((f"zeros P{P} k{keep}", put(torch.zeros(N, P)), keep))
            out.append((f"relu P{P} k{keep}", put(torch.relu(torch.randn(N, P, generator=g) - 1.5)), keep))
            # a plateau at the threshold value, spanning sub-block boundaries, with a few larger values after it
            s = torch.full((N, P), 1.0)
            s[:, P // 3:] = 2.0
            s[:, -3:] = 5.0
            out.append((f"plateau P{P} k{keep}", put(s), keep))
            # every sub-block of 4 has the same maximum; the tied elements sit at varying offsets
            s = torch.zeros(N, P)
            for p in range(P):
                if (p * 7) % 4 == p % 4:
                    s[:, p] = 3.0
            out.append((f"blockmax-ties P{P} k{keep}", put(s), keep))
            # signed zeros and one-ulp neighbours
            s = torch.where(torch.rand(N, P, generator=g) < 0.5, torch.tensor(0.0), torch.tensor(-0.0))
            s = s + torch.where(torch.rand(N, P, generator=g) < 0.1, torch.tensor(1e-45), torch.tensor(0.0))
            out.append((f"signed-zero P{P} k{keep}", put(s), keep))
    return out


@pytest.mark.parametrize("sub", [1, 2, 4, 32])
def test_two_level_equals_reference(sub):
    """select_two_level == select_reference == the tie rule written out, for sub-blocks of 1 to 32 pools."""
    n = 0
    for name, sc, keep in _cases():
        want = _ref_mask(sc, keep)
        p_ref, c_ref = dsa_long.select_reference(sc, keep)
        p_two, c_two = dsa_long.select_two_level(sc, keep, sub)
        P = sc.shape[1]
        assert torch.equal(_as_mask(p_ref, c_ref, P), want), name
        assert torch.equal(_as_mask(p_two, c_two, P), want), f"{name} sub {sub}"
        assert torch.equal(p_two, p_ref) and torch.equal(c_two, c_ref), f"{name} sub {sub}"
        n += 1
    assert n > 100


def test_two_level_corner_tie_room():
    """The proof's corner, as data: t equals the keep-th sub-block maximum M, the elements equal to t sit in
    sub-blocks with a larger maximum AND in tied sub-blocks, and the lowest-index ties are in the LATER tied
    sub-blocks' predecessors. keep 3, sub 4: sub-block 0 = [9, 5, 5, 5] (max 9 > t), sub-blocks 1, 2 = [5, 0, 0, 0]
    (max 5 = t); the exact selection is 9 and the two lowest-index 5s, both in sub-block 0."""
    sc = torch.tensor([[9.0, 5, 5, 5, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0]])
    want = _ref_mask(sc, 3)
    assert want[0].nonzero().flatten().tolist() == [0, 1, 2]
    p, c = dsa_long.select_two_level(sc, 3, 4)
    assert c.tolist() == [3] and p[0].tolist() == [0, 1, 2]


def test_slots_match_bucketed_selection():
    """The tokens slots() attends (selected pools, tail, causal) are the bucketed path's selection
    (glm5_next.block_mask, kernels/dsa_topk.py's rule with the tail) plus the causal visibility, for a chunk and
    for a decode batch, pages of 4 and 8 tokens."""
    from kiln.models import glm5_next

    g = torch.Generator().manual_seed(1)
    kp, keep = 4, 3
    for ps in (4, 8):
        for L, positions in ((64, torch.tensor([0, 2, 3, 4, 11, 12, 13, 30, 47, 63])),
                             (96, torch.tensor([95, 50, 17, 7]))):
            P = L // kp
            N = positions.shape[0]
            npool = dsa_long.npools(positions, kp)
            index = torch.randn(N, P, generator=g)
            index[:, ::5] = 0.0
            sc = index + torch.where(torch.arange(P).view(1, P) < npool.view(N, 1), 0.0, NEG)
            pools, cnt = dsa_long.select_two_level(sc, keep, 2)
            vis = torch.where(torch.arange(L).view(1, L) <= positions.view(N, 1), 0.0, NEG).view(N, 1, L)
            want = (glm5_next.block_mask(index.view(N, 1, P), vis, kp, keep, True, select="nki")[:, 0] == 0)
            want &= vis[:, 0] == 0
            assert torch.equal(dsa_long.reference_mask(pools, cnt, npool, positions, L, kp), want)
            # through the block table: rows / bias address exactly those tokens
            pages = L // ps
            table = torch.randperm(200, generator=g)[:pages] + 1
            rows, bias = dsa_long.slots(pools, cnt, npool, positions, table, ps, kp, keep + 3)
            tok = (rows.unsqueeze(-1) * kp + torch.arange(kp)).reshape(N, -1)  # cache slots
            live = (bias.reshape(N, -1) == 0)
            slot_of = (table.view(-1, 1) * ps + torch.arange(ps)).reshape(-1)  # context position -> cache slot
            for n in range(N):
                got = sorted(tok[n][live[n]].tolist())
                assert got == sorted(slot_of[want[n]].tolist()), (ps, L, n)
            # one table per row (decode form) gives the same
            rows2, bias2 = dsa_long.slots(pools, cnt, npool, positions, table.expand(N, pages), ps, kp, keep + 3)
            assert torch.equal(rows2, rows) and torch.equal(bias2, bias)


# -- the engine on the long path ----------------------------------------------------------------------------

def _glm():
    """tests/test_glm5_next.py's builder (transformers >= 5.18), or a skip."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from tests import test_glm5_next

    return test_glm5_next


class _Lazy:
    def __getattr__(self, name):
        return getattr(_glm(), name)


glm = _Lazy()


@pytest.fixture(scope="module")
def sparse(tmp_path_factory):
    """index_topk 16 (4 pools of 4 kept), positions up to 4096."""
    path = tmp_path_factory.mktemp("glm5_next_long")
    return str(path), glm.build(str(path), seed=1, index_topk=16, max_position_embeddings=4096)


def _generate(path, ps, n, long_keys, monkeypatch, **kw):
    """Generate on an engine whose buckets past long_keys keys run the long path; asserts that it ran (and
    that it did not when long_keys is out of reach)."""
    from kiln.engine.request import SamplingParams
    from kiln.models import mla

    calls = {"long": 0}
    real = mla.attention_long

    def counted(*a, **k):
        calls["long"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mla, "attention_long", counted)
    monkeypatch.setattr(dsa_long, "LONG_KEYS", long_keys)
    eng = glm.engine(path, **kw)
    reqs = eng.generate(ps, SamplingParams(max_new_tokens=n, ignore_eos=True, logprobs=0))
    monkeypatch.setattr(mla, "attention_long", real)
    assert (calls["long"] > 0) == (long_keys < 1 << 20), calls
    return eng, reqs


@pytest.mark.parametrize("page_size,chunk,cache", [(4, 16, "auto"), (8, 6, "separate"), (4, 8, "off")])
def test_long_path_matches_transformers(sparse, page_size, chunk, cache, monkeypatch):
    """Every bucket past 16 keys on the long path (scores, two-level selection, slots, absorbed attention), through
    chunked prefill with chunks that cut pools, a radix-cache hit and batched decode, against transformers' greedy
    tokens, and against the bucketed path's logprobs (fp32: the same selection, the same keys)."""
    from kiln.models import mla

    monkeypatch.setattr(mla, "POOL_CACHE", cache)
    path, hf = sparse
    ps = glm.prompts(2, (37, 70, 9))
    ps.append(ps[1][:48] + ps[0][:20])  # shares 48 tokens (12 pages of 4) with prompt 1: a prefix hit
    kw = dict(page_size=page_size, max_prefill_tokens=chunk)
    eng_l, got = _generate(path, ps, 12, 16, monkeypatch, **kw)
    assert eng_l.model.long_ctx(32) and not eng_l.model.long_ctx(16)
    _, want = _generate(path, ps, 12, 1 << 30, monkeypatch, **kw)
    assert [r.output_ids for r in got] == [r.output_ids for r in want] == [glm.hf_greedy(hf, x, 12) for x in ps]
    for a, b in zip(got, want):
        torch.testing.assert_close(torch.tensor([x[0] for x in a.logprobs]), torch.tensor([x[0] for x in b.logprobs]),
                                   rtol=0, atol=2e-5)


def test_long_path_fp8_kv(sparse, monkeypatch):
    """Under an FP8 KV cache (the separate bf16 pool-key cache) the long path serves the bucketed path's tokens."""
    path, _ = sparse
    ps = glm.prompts(7, (41, 66))
    _, got = _generate(path, ps, 8, 16, monkeypatch, kv_cache_dtype="fp8")
    _, want = _generate(path, ps, 8, 1 << 30, monkeypatch, kv_cache_dtype="fp8")
    assert [r.output_ids for r in got] == [r.output_ids for r in want]


@pytest.mark.parametrize("sub", [2, 32])
def test_long_context_against_transformers(sparse, sub, monkeypatch):
    """A 1,500-token prompt (375 pools against 4 kept: the 1M regime's ratio of candidates to selection is 512)
    prefilled in chunks of 64 on the long path, then decoded: transformers' greedy tokens, and the bucketed path's
    logprobs. sub 2 and 32 make the two-level selection run with 188 and 12 sub-blocks."""
    monkeypatch.setattr(dsa_long, "SUB", sub)
    path, hf = sparse
    ps = glm.prompts(9, (1500,), 384)
    kw = dict(max_model_len=2048, num_pages=600, max_prefill_tokens=64, max_num_seqs=1)
    _, got = _generate(path, ps, 6, 64, monkeypatch, **kw)
    _, want = _generate(path, ps, 6, 1 << 30, monkeypatch, **kw)
    assert [r.output_ids for r in got] == [r.output_ids for r in want] == [glm.hf_greedy(hf, ps[0], 6)]
    torch.testing.assert_close(torch.tensor([x[0] for x in got[0].logprobs]),
                               torch.tensor([x[0] for x in want[0].logprobs]), rtol=0, atol=2e-5)


def test_long_scratch_is_capped(sparse, monkeypatch):
    """The selection scratch (models/mla.py bind_scratch) of a model whose every DSA layer can run long is at
    most LONG_KEYS wide: a 1M max_model_len no longer allocates [rows, 1M]."""
    monkeypatch.setattr(dsa_long, "LONG_KEYS", 64)
    path, _ = sparse
    eng = glm.engine(path, max_model_len=4096, num_pages=64)
    for l in eng.model.kv_layers():
        if l.dsa_mask is not None:
            assert l.dsa_mask.shape[1] == 64


def test_device_selection_forms_equal_reference():
    """select_device (kernels/dsa_topk.py's selection in row groups, then compact()) and compact() alone give
    select_reference's pools and counts on the adversarial cases: the decode batch's device form."""
    for name, sc, keep in _cases():
        want = dsa_long.select_reference(sc, keep)
        got = dsa_long.select_device(sc, keep)
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]), name
        mask = _ref_mask(sc, keep).to(torch.float32)
        assert torch.equal(dsa_long.compact(mask, keep)[0], want[0]), name


def test_long_path_tp2_shards_the_selection(sparse, monkeypatch):
    """At tp=2 (attention TP 2) each rank selects half of a chunk's queries and the halves are gathered over the
    attention group (models/mla.py _long_prefill_select): the same tokens as tp=1 on the long path."""
    from kiln.engine.request import SamplingParams
    from kiln.models import mla

    path, _ = sparse
    monkeypatch.setenv("KILN_DSA_LONG_KEYS", "16")
    monkeypatch.setattr(dsa_long, "LONG_KEYS", 16)
    calls = {"shard": 0}
    real = mla._long_prefill_select

    def counted(model, *a, **k):
        if getattr(model, "long_grp_onehot", None) is not None:
            calls["shard"] += 1
        return real(model, *a, **k)

    monkeypatch.setattr(mla, "_long_prefill_select", counted)
    ps = glm.prompts(3, (11, 41, 70))
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True)
    want = [r.output_ids for r in glm.engine(path, max_num_seqs=3, max_prefill_tokens=16).generate(ps, sp)]
    two = glm.engine(path, max_num_seqs=3, max_prefill_tokens=16, tp=2)
    try:
        assert two.model.long_grp_onehot.tolist() == [1.0, 0.0]
        got = [r.output_ids for r in two.generate(ps, sp)]
    finally:
        two.close()
    assert got == want
    assert calls["shard"] > 0


@pytest.mark.parametrize("A", [2, 4, 8])
def test_cp_merge_equals_global_selection(A):
    """Context parallelism's merge (dsa_long.cp_merge): every rank's exact local top-keep over its pools (context
    pools c = m A + rank), gathered and merged, is the global exact selection, ties included (the adversarial cases,
    whose ties straddle ranks)."""
    for name, sc, keep in _cases():
        N, P = sc.shape
        want = _ref_mask(sc, keep)
        vals, cps = [], []
        for r in range(A):
            loc = sc[:, r::A]
            if loc.shape[1] == 0:
                loc = torch.full((N, 1), NEG)
            lp, lc = dsa_long.select_reference(loc, keep)
            ok = torch.arange(keep).view(1, keep) < lc.view(N, 1)
            vals.append(torch.where(ok, loc.gather(1, lp), torch.tensor(NEG)))
            cps.append((lp * A + r).float())
        sel = dsa_long.cp_merge(torch.stack(vals, 1), torch.stack(cps, 1), keep)
        got = torch.zeros(N, P, dtype=torch.bool)
        for r in range(A):
            for n in range(N):
                for k in range(keep):
                    if sel[n, r, k]:
                        got[n, int(cps[r][n, k])] = True
        assert torch.equal(got, want), f"{name} A={A}"


@pytest.mark.parametrize("A", [16, 32])
def test_cp_merge_wide(A):
    """cp_merge at A = 16 / 32 with keep 512: A keep = 8,192 / 16,384 candidates per row, wider than the selection
    kernel's 4,096 per partition, take the two-level threshold (dsa_long._keep_th_two_level); still the global exact
    selection with ties (integer scores, a plateau straddling ranks, candidate prefixes around keep)."""
    g = torch.Generator().manual_seed(4)
    keep, P, N = 512, 32768, 4
    npool = torch.tensor([keep - 3, keep + 1, P // 3, P])
    cand = torch.arange(P).view(1, P) < npool.view(N, 1)
    plateau = torch.full((N, P), 1.0)
    plateau[:, P // 2:] = 2.0
    for name, sc in (("int", torch.randint(0, 7, (N, P), generator=g).float()), ("randn", torch.randn(N, P, generator=g)),
                     ("plateau", plateau)):
        sc = torch.where(cand, sc, torch.tensor(NEG))
        want = _ref_mask(sc, keep)
        vals, cps = [], []
        for r in range(A):
            loc = sc[:, r::A]
            lp, lc = dsa_long.select_reference(loc, keep)
            ok = torch.arange(keep).view(1, keep) < lc.view(N, 1)
            vals.append(torch.where(ok, loc.gather(1, lp), torch.tensor(NEG)))
            cps.append((lp * A + r).float())
        sel = dsa_long.cp_merge(torch.stack(vals, 1), torch.stack(cps, 1), keep)
        got = torch.zeros(N, P, dtype=torch.bool)
        for r in range(A):
            m = sel[:, r]
            rows, ks = m.nonzero(as_tuple=True)
            got[rows, cps[r][rows, ks].long()] = True
        assert torch.equal(got, want), f"{name} A={A}"


def test_cp_slots_and_pool_rows():
    """A token's local slot (dsa_long.cp_local_slots) and a context's local pool rows (cp_pool_rows) agree: the
    rows of local pool m hold exactly the slots its owner wrote for context pool m A + rank, and every token is
    written by exactly one rank."""
    ps, kp = 32, 4
    for A in (2, 4, 8):
        table = torch.tensor([5, 2, 9, 7])
        pos = torch.arange(table.numel() * ps)
        owners = []
        for r in range(A):
            ls = dsa_long.cp_local_slots(pos, table, ps, kp, A, torch.tensor([r]))
            mine = (pos // kp) % A == r
            owners.append(mine)
            rows = dsa_long.cp_pool_rows(table, ps, kp, A)  # [Pl]
            for m in range(rows.numel()):
                c = m * A + r
                toks = torch.arange(c * kp, c * kp + kp)
                assert torch.equal(ls[toks], rows[m] * kp + torch.arange(kp)), (A, r, m)
            assert torch.all(ls[~mine] == dsa_long.cp_dump_slot(ps, A))
        assert torch.equal(torch.stack(owners).sum(0), torch.ones(pos.numel(), dtype=torch.long))


@pytest.mark.parametrize("tp,page_size", [(2, 8), (4, 16)])
def test_cp_engine_matches_replicated(sparse, tp, page_size, monkeypatch):
    """KILN_DSA_CP=1 at tp 2 and 4 (attention TP = tp, each rank holding 1 / tp of the DSA caches, a 4x / 8x
    smaller local page): the same greedy tokens as the replicated long path at tp=1 and as transformers, through
    chunked prefill (chunks that cut pools), a prefix hit and batched decode."""
    from kiln.engine.request import SamplingParams
    from kiln.models import mla

    path, hf = sparse
    monkeypatch.setenv("KILN_DSA_POOL_CACHE", "separate")
    monkeypatch.setattr(mla, "POOL_CACHE", "separate")
    ps = glm.prompts(4, (37, 70, 9))
    ps.append(ps[1][:48] + ps[0][:20])
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True)
    _, want = _generate(path, ps, 10, 16, monkeypatch, page_size=page_size, max_prefill_tokens=12, max_num_seqs=4)
    want = [r.output_ids for r in want]
    assert want == [glm.hf_greedy(hf, x, 10) for x in ps]
    monkeypatch.setenv("KILN_DSA_CP", "1")
    eng = glm.engine(path, page_size=page_size, max_prefill_tokens=12, max_num_seqs=4, tp=tp)
    try:
        assert eng.model.cp == tp
        assert eng.runner.lps == page_size // tp
        got = [r.output_ids for r in eng.generate(ps, sp)]
    finally:
        eng.close()
    assert got == want


@pytest.mark.parametrize("small", [1, 3, 8])
def test_cp_slot_classes_equal_fixed_slots(small):
    """models/mla.py _cp_attend_classes (KILN_DSA_CP_SLOT_CLASSES): every row attended over a buffer sized by where its
    selected entries end (the list's first small - 1 slots and the tail, or the fixed layout) gives attention_cp's
    fixed-slot partial attention and log-sum-exp, rows of both classes in one call (selections in any order, counts
    0 .. keep, the tail owned or not, partly visible)."""
    from kiln.models import dsa_long, mla

    g = torch.Generator().manual_seed(7)
    T, keep, kp, R, H, NP = 9, 8, 4, 16, 3, 40
    mine = torch.rand(T, keep, generator=g) < torch.linspace(0.0, 1.0, T).view(T, 1)
    mine[0] = False
    mine[-1] = True
    tail_own = torch.rand(T, generator=g) < 0.6
    rows_sel = torch.randint(0, NP, (T, keep), generator=g)
    rows_tail = torch.randint(0, NP, (T, 1), generator=g)
    npool = torch.randint(5, 20, (T,), generator=g)
    positions = npool * kp + torch.randint(0, kp, (T,), generator=g)
    kc = torch.randn(NP * kp, 1, R, generator=g)
    q_all = torch.randn(T, H, R, generator=g) * 0.3
    scale = 0.25
    # attention_cp's fixed slots (its default branch, n_slots = keep + 1 on the CPU)
    n_slots = keep + 1
    sl = torch.arange(n_slots).view(1, n_slots)
    cl = sl.clamp(max=keep - 1).expand(T, n_slots)
    srows = torch.where(sl < keep, torch.gather(rows_sel, 1, cl),
                        torch.where(sl == keep, rows_tail, torch.zeros_like(rows_tail)))
    t4 = torch.arange(kp).view(1, 1, kp)
    sel_ok = ((sl < keep) & torch.gather(mine, 1, cl)).unsqueeze(-1)
    tail_ok = ((sl == keep) & tail_own.view(T, 1)).unsqueeze(-1) & (npool.view(T, 1, 1) * kp + t4
                                                                     <= positions.view(T, 1, 1))
    sbias = torch.where(sel_ok | tail_ok, 0.0, mla.NEG_INF).to(torch.float32)
    want_o, want_l = dsa_long.cp_attend_partial(q_all, kc, srows, sbias, scale)
    got_o, got_l = mla._cp_attend_classes(q_all, kc, rows_sel, rows_tail, mine, tail_own, npool, positions, scale, kp,
                                          small, keep + 1)
    bound = torch.where(mine, torch.arange(keep).view(1, keep) + 1, 0).amax(-1)  # the class rule (any order)
    assert (bound <= min(small - 1, keep)).any() and (bound > min(small - 1, keep)).any()
    real = want_l > -1e29
    torch.testing.assert_close(got_o[real.all(-1)], want_o[real.all(-1)], rtol=0, atol=1e-5)
    torch.testing.assert_close(got_l[real], want_l[real], rtol=0, atol=1e-5)
    assert (got_l[~real] < -1e29).all()


@pytest.mark.parametrize("small", [1, 2])
def test_cp_engine_slot_classes(sparse, small, monkeypatch):
    """KILN_DSA_CP_SLOT_CLASSES=1 at tp 2 with a small buffer of 1 / 2 slots (so both classes have rows; prefill chunks
    take the classes, decode batches the fixed slots): the CP engine's greedy tokens equal the replicated long path's
    and transformers'."""
    from kiln.engine.request import SamplingParams
    from kiln.models import mla

    path, hf = sparse
    monkeypatch.setenv("KILN_DSA_POOL_CACHE", "separate")
    monkeypatch.setattr(mla, "POOL_CACHE", "separate")
    ps = glm.prompts(4, (37, 70, 9))
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True)
    kw = dict(page_size=8, max_prefill_tokens=12, max_num_seqs=4)
    _, want = _generate(path, ps, 10, 16, monkeypatch, **kw)
    want = [r.output_ids for r in want]
    assert want == [glm.hf_greedy(hf, x, 10) for x in ps]
    monkeypatch.setenv("KILN_DSA_CP", "1")
    monkeypatch.setenv("KILN_DSA_CP_SLOT_CLASSES", "1")
    monkeypatch.setenv("KILN_DSA_CP_SLOTS_SMALL", str(small))
    monkeypatch.setattr(mla, "CP_SLOT_CLASSES", True)
    monkeypatch.setattr(mla, "CP_SLOTS_SMALL", small)
    eng = glm.engine(path, tp=2, **kw)
    try:
        got = [r.output_ids for r in eng.generate(ps, sp)]
    finally:
        eng.close()
    assert got == want


def test_cp_engine_fewer_local_pools_than_keep(tmp_path, monkeypatch):
    """KILN_DSA_CP=1 where a rank holds fewer local pools than keep (index_topk 64: keep 16 pools; page 8 at A = 2:
    one local pool per page, so a short context's bucket gives each rank a few): every local pool is a candidate, and
    the tokens equal the replicated long path's and transformers' (the decode agent's 33-page bucket at A = 8, 264
    local pools against keep 512, failed to trace)."""
    from kiln.engine.request import SamplingParams
    from kiln.models import mla

    path = str(tmp_path)
    hf = glm.build(path, seed=2, index_topk=64, max_position_embeddings=4096)
    monkeypatch.setenv("KILN_DSA_POOL_CACHE", "separate")
    monkeypatch.setattr(mla, "POOL_CACHE", "separate")
    ps = glm.prompts(3, (41, 9, 66))
    kw = dict(page_size=8, max_prefill_tokens=12, max_num_seqs=4)
    _, want = _generate(path, ps, 8, 16, monkeypatch, **kw)
    want = [r.output_ids for r in want]
    assert want == [glm.hf_greedy(hf, x, 8) for x in ps]
    monkeypatch.setenv("KILN_DSA_CP", "1")
    eng = glm.engine(path, tp=2, **kw)
    seen = []
    real = mla.attention_cp

    def spy(model, layer, x, positions, slot_mapping, table):
        seen.append(table.shape[-1] * (model.page_size // (4 * model.cp)))
        return real(model, layer, x, positions, slot_mapping, table)

    monkeypatch.setattr(mla, "attention_cp", spy)
    try:
        got = [r.output_ids for r in eng.generate(ps, SamplingParams(max_new_tokens=8, ignore_eos=True))]
    finally:
        eng.close()
    assert got == want
    if seen:  # the spy sees rank 0's calls when it runs in this process
        assert min(seen) < 16


@pytest.mark.parametrize("long_keys", [16, 1 << 30])
def test_minimal_kv_matches_full(tmp_path, long_keys, monkeypatch):
    """KILN_DSA_KV=minimal (models/mla.py KV_LAYOUT): V holds only the pool-key pieces and each request's open pool
    lives in its state row. In fp32 it keeps every pool key bit for bit, so the tokens and logprobs equal the full
    layout's, on the bucketed path and on the long path, through chunks of 6 that cut pools of 4 (the open pool
    crossing a chunk boundary), a prefix hit and batched decode; and its cache is 512 + 32 values per token."""
    from kiln.config import ModelConfig
    from kiln.models import mla

    path = str(tmp_path)
    hf = glm.build(path, seed=1, index_topk=16, max_position_embeddings=4096)
    ps = glm.prompts(5, (37, 70, 9, 23))
    ps.append(ps[1][:48] + ps[0][:21])
    kw = dict(page_size=8, max_prefill_tokens=6, max_num_seqs=4)
    _, want = _generate(path, ps, 12, long_keys, monkeypatch, **kw)
    monkeypatch.setattr(mla, "KV_LAYOUT", "minimal")
    cfg = ModelConfig.from_pretrained(path)
    dsa = [s for s in cfg.attn_layers if getattr(s, "mla", None) is not None]
    assert dsa and all((s.head_dim, s.v_head_dim) == (512, 32) for s in dsa)
    eng, got = _generate(path, ps, 12, long_keys, monkeypatch, **kw)
    layers = [l for l in eng.model.kv_layers() if l.spec.mla is not None]
    assert all(l.open_pool.shape[1:] == (4, 256) and l.pool_key is None and not l.pool_inplace for l in layers)
    assert [r.output_ids for r in got] == [r.output_ids for r in want] == [glm.hf_greedy(hf, x, 12) for x in ps]
    for a, b in zip(got, want):
        torch.testing.assert_close(torch.tensor([x[0] for x in a.logprobs]), torch.tensor([x[0] for x in b.logprobs]),
                                   rtol=0, atol=2e-5)


@pytest.mark.parametrize("tp,page_size", [(2, 8), (4, 16)])
def test_cp_minimal_matches_replicated(tmp_path, tp, page_size, monkeypatch):
    """KILN_DSA_CP=1 over the minimal layout (KILN_DSA_KV=minimal): V holds the owner's pool-key pieces, every rank keeps
    the open-pool row. The same greedy tokens and logprobs as the replicated minimal long path at tp=1 and as
    transformers, through chunks of 6 that cut pools of 4, a prefix hit and batched decode; each rank's page is 1 / tp."""
    from kiln.engine.request import SamplingParams
    from kiln.models import mla

    path = str(tmp_path)
    hf = glm.build(path, seed=1, index_topk=16, max_position_embeddings=4096)
    monkeypatch.setenv("KILN_DSA_KV", "minimal")
    monkeypatch.setattr(mla, "KV_LAYOUT", "minimal")
    ps = glm.prompts(5, (37, 70, 9, 23))
    ps.append(ps[1][:48] + ps[0][:21])
    kw = dict(page_size=page_size, max_prefill_tokens=6, max_num_seqs=4)
    _, want = _generate(path, ps, 10, 16, monkeypatch, **kw)
    assert [r.output_ids for r in want] == [glm.hf_greedy(hf, x, 10) for x in ps]
    monkeypatch.setenv("KILN_DSA_CP", "1")
    eng = glm.engine(path, tp=tp, **kw)
    try:
        assert eng.model.cp == tp and eng.runner.lps == page_size // tp
        layers = [l for l in eng.model.kv_layers() if l.spec.mla is not None]
        assert all(l.pool_key is None and l.open_pool is not None for l in layers)
        got = eng.generate(ps, SamplingParams(max_new_tokens=10, ignore_eos=True, logprobs=0))
    finally:
        eng.close()
    assert [r.output_ids for r in got] == [r.output_ids for r in want]
    for a, b in zip(got, want):
        torch.testing.assert_close(torch.tensor([x[0] for x in a.logprobs]), torch.tensor([x[0] for x in b.logprobs]),
                                   rtol=0, atol=2e-5)


def test_minimal_kv_bytes_per_token(monkeypatch):
    """GLM-5.3-Flash's real config (tests/reference/hybrid_configs/GLM-5.3-Flash.json): 9,152 bytes per token under
    FP8 KV in the full layout (latent 512 + indexer key and gates 256, plus the 64-byte bf16 pool-key state, x 11 DSA
    layers), 5,984 in the minimal one (512 + 32)."""
    import json
    import os

    from kiln.config import ModelConfig
    from kiln.models import mla
    from kiln.models.decoder import state_bytes_per_token_rank

    src = os.path.join(os.path.dirname(__file__), "reference", "hybrid_configs", "GLM-5.3-Flash.json")
    with open(src) as f:
        c = json.load(f)

    def per_token():
        from kiln.models import glm5_next

        cfg = glm5_next.config_from_hf(ModelConfig, c, ())
        return cfg.kv_bytes_per_token(torch.float8_e4m3fn) + state_bytes_per_token_rank(cfg, 8, torch.bfloat16, True)

    assert per_token() == 9152
    monkeypatch.setattr(mla, "KV_LAYOUT", "minimal")
    assert per_token() == 5984


def test_qshard_tp2_matches_tp1(sparse, monkeypatch):
    """KILN_DSA_QSHARD=1 at tp=2: each rank selects and attends its half of a long-context chunk's rows with every
    head (whole-head q_b / kv_b / o_proj copies) and the halves are summed by the attention reduction; the tokens
    equal tp=1's on the long path (and transformers')."""
    from kiln.engine.request import SamplingParams
    from kiln.models import mla

    path, hf = sparse
    monkeypatch.setenv("KILN_DSA_LONG_KEYS", "16")
    monkeypatch.setattr(dsa_long, "LONG_KEYS", 16)
    ps = glm.prompts(8, (37, 70, 9))
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True)
    want = [r.output_ids for r in glm.engine(path, max_num_seqs=3, max_prefill_tokens=16).generate(ps, sp)]
    assert want == [glm.hf_greedy(hf, x, 10) for x in ps]
    monkeypatch.setenv("KILN_DSA_QSHARD", "1")
    monkeypatch.setattr(mla, "QSHARD", True)
    calls = {"q": 0}
    real = mla.attention_long_qshard

    def counted(*a, **k):
        calls["q"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mla, "attention_long_qshard", counted)
    two = glm.engine(path, max_num_seqs=3, max_prefill_tokens=16, tp=2)
    try:
        assert all(getattr(l, "qshard", False) for l in two.model.kv_layers())
        got = [r.output_ids for r in two.generate(ps, sp)]
    finally:
        two.close()
    assert got == want and calls["q"] > 0


@pytest.mark.parametrize("minimal", [False, True])
def test_index_scorer_layout_matches_scores(minimal):
    """kernels/dsa_index.py (the decode agent's scorer, KILN_DSA_LONG_SCORER=index) as attention_long feeds it: the
    separate pool-key cache [slots, Di / kp] viewed as page rows, the rows' tables as page groups, the scale folded
    into w; its scores (its emulation on the host) equal dsa_long.scores over the same pools, candidates included.
    minimal: the minimal layout's V [slots, 1, Di / kp] in fp8 (the kernel's fp8 form), viewed the same way."""
    from kiln.kernels import dsa_index

    g = torch.Generator().manual_seed(3)
    ps, kp, Di, Hi, B = 32, 4, 128, 32, 3
    pages = 256  # 2 x 128-page groups per row
    cache = torch.randn(1024 * ps, Di // kp, generator=g).to(torch.bfloat16)  # 1024 pages of slots
    if minimal:
        cache = cache.unsqueeze(1).to(torch.float8_e4m3fn)  # V [slots, 1, Di / kp]
    table = torch.stack([torch.randperm(1023, generator=g)[:pages] + 1 for _ in range(B)])
    q = torch.randn(B, Hi, Di, generator=g).to(torch.bfloat16)
    w = torch.randn(B, Hi, generator=g)
    positions = torch.tensor([pages * ps - 1, 3000, 17])
    npool = dsa_long.npools(positions, kp)
    scale = Di ** -0.5
    pk = cache.view(-1, ps, Di // kp)[table].reshape(B, pages * ps // kp, Di)  # what context_pool_keys gathers
    want = dsa_long.scores(q, w, pk.to(torch.bfloat16), npool, scale)
    P = pages * ps // kp
    cand = torch.where(torch.arange(P).view(1, P) < npool.view(B, 1), 0.0, NEG)
    got = dsa_index.scores(q, w * scale, cache.view(-1, (ps // kp) * Di), dsa_index.page_groups(table), cand)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-4)


def test_compact_at_1m_pools():
    """dsa_long.compact at the 262,144 pools of a 1M context (G = 512 groups of 512, no full-length scan) equals the
    reference ordering, for sparse, clustered and empty masks."""
    g = torch.Generator().manual_seed(5)
    M, keep = 262144, 512
    sel = torch.zeros(4, M)
    sel[0, torch.randperm(M, generator=g)[:keep]] = 1  # spread
    sel[1, 1000:1000 + keep] = 1  # one run across group boundaries
    sel[2, torch.randperm(M, generator=g)[:37]] = 1  # fewer than keep
    want = dsa_long._ascending(sel > 0, keep)
    got = dsa_long.compact(sel, keep)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_cp_merge_in_row_pieces_equals_one_piece(monkeypatch):
    """dsa_long.cp_merge with rows (attention_cp passes CP_MERGE_ROWS for a decode batch: the 96-row decode graph's
    one-piece merge failed neuronx-cc) merges in pieces of it: the same mask as one piece, since every row is merged on
    its own; without rows (a prefill chunk) it is one piece whatever N."""
    g = torch.Generator().manual_seed(11)
    N, A, keep = 10, 4, 6
    vals = torch.randint(0, 5, (N, A, keep), generator=g).float()
    vals[0, 1, 3] = NEG
    cps = torch.stack([torch.randperm(A * keep, generator=g)[: A * keep].view(A, keep) for _ in range(N)]).float()
    whole = dsa_long.cp_merge_rows(vals, cps, keep)
    for rows in (None, 3, 4, 10, 64):
        assert torch.equal(dsa_long.cp_merge(vals, cps, keep, rows), whole), rows
    pieces = []
    real = dsa_long.cp_merge_rows

    def counted(v, c, k):
        pieces.append(v.shape[0])
        return real(v, c, k)

    monkeypatch.setattr(dsa_long, "cp_merge_rows", counted)
    dsa_long.cp_merge(vals, cps, keep)
    dsa_long.cp_merge(vals, cps, keep, 3)
    assert pieces == [10, 3, 3, 3, 1]


def _cp_selection_path(sc, nloc, table, ps, kp, A, rank, keep):
    """attention_cp's local list on the device (the default decode branch): select_device + compact, the list's values
    and pool_row of the block table."""
    T, Pl = sc.shape
    lp, lc = dsa_long.select_device(sc, keep)
    k = torch.arange(keep).view(1, keep)
    fill = sc.new_full((T, keep), NEG) if Pl < keep else torch.full_like(sc[:, :keep], NEG)
    lval = torch.where(k < lc.view(T, 1), sc.gather(1, lp), fill)
    ppl = ps // (kp * A)
    pg = torch.floor(lp.to(torch.float32) * (1.0 / ppl)).to(torch.int64)
    rows = table.gather(1, pg.clamp(max=table.shape[1] - 1)) * ppl + (lp - pg * ppl)
    return lp, lc, lval, rows, lp * A + rank


@pytest.mark.parametrize("A,ps", [(2, 8), (8, 32), (8, 256)])
def test_cp_local_all_equals_selection(A, ps):
    """models/mla.py _cp_local_all (KILN_DSA_CP_ALL_LOCAL): a decode batch whose Pl local pools fit keep gets the same
    local list as select_device + compact, element for element (pools, count, values, slot rows, context pools), on
    contexts of keep A context pools, one below, A below, a few, one and none (each rank's nloc from cp_local_count),
    with integer ties, exact zeros (relu) and NEG_INF past the visible prefix, at Pl = keep, keep - ppl and keep / 2."""
    from kiln.models import mla

    g = torch.Generator().manual_seed(5)
    kp, keep, Hi, D = 4, 16, 3, 8
    ppl = ps // (kp * A)
    for Pp in sorted({keep // ppl, keep // ppl - 1, max(keep // (2 * ppl), 1)}):
        Pl = Pp * ppl
        cap = Pl * A  # context pools the bucket holds
        npools = [cap, cap - 1, cap - A, keep * A, keep * A - 1, 7, 1, 0]
        npool = torch.tensor([min(max(n, 0), cap) for n in npools])
        T = npool.numel()
        table = torch.stack([torch.randperm(4 * Pp, generator=g)[:Pp] for _ in range(T)])
        qI = torch.randint(-2, 3, (T, Hi, D), generator=g).float()
        w = torch.randint(0, 3, (T, Hi), generator=g).float()
        pk = torch.randint(-1, 2, (T, Pl, D), generator=g).float()
        pk[:, ::3] = pk[:, :1]  # repeated keys: tied scores
        pk[1, : Pl // 2] = 0.0  # exact zeros
        for r in range(A):
            rank = torch.tensor([r])
            nloc = dsa_long.cp_local_count(npool, A, rank)
            assert int(nloc.max()) <= Pl <= keep
            sc = dsa_long.scores(qI, w, pk, nloc, 0.5)
            prow = dsa_long.cp_pool_rows(table, ps, kp, A)
            want = _cp_selection_path(sc, nloc, table, ps, kp, A, rank, keep)
            lp, lc, lval, rows = mla._cp_local_all(sc, nloc, prow, keep)
            got = (lp, lc, lval, rows, lp * A + rank)
            for name, a, b in zip(("lp", "lc", "lval", "rows", "cpool"), got, want):
                assert a.dtype == b.dtype and torch.equal(a, b), (A, ps, Pl, r, name)
            # the host selection (the CPU engine's form) agrees on the listed entries
            hp, hc = dsa_long.select(sc, keep)
            assert torch.equal(hc, lc)
            on = torch.arange(keep).view(1, keep) < lc.view(T, 1)
            assert torch.equal(torch.where(on, hp, 0), lp)
    with pytest.raises(ValueError):  # past keep local pools the selection is real: never the identity
        mla._cp_local_all(torch.zeros(2, keep + 1), torch.tensor([3, keep + 1]), torch.zeros(2, keep + 1), keep)


def test_cp_page_keys_equal_pool_rows():
    """KILN_DSA_CP_PAGE_KEYS: the pool-key cache viewed as [pages, ppl Di] and indexed by the block table holds the rows
    cp_pool_rows names, in order."""
    for A, ps, kp, Di in ((2, 8, 4, 16), (8, 256, 4, 128)):
        ppl = ps // (kp * A)
        pages = 12
        cache = torch.randn(pages * (ps // A), Di // kp)  # local slots x the pool-key piece (Di / kpool per slot)
        table = torch.tensor([[3, 7, 1], [0, 11, 5]])
        prow = dsa_long.cp_pool_rows(table, ps, kp, A)
        want = cache.view(-1, Di)[prow]
        got = cache.view(-1, ppl * Di)[table].view(table.shape[0], prow.shape[-1], Di)
        assert torch.equal(got, want), (A, ps)


@pytest.mark.parametrize("all_local,page_keys,layout", [(1, 0, "full"), (0, 1, "full"), (1, 1, "full"),
                                                        (1, 1, "minimal")])
def test_cp_engine_all_local(tmp_path, all_local, page_keys, layout, monkeypatch):
    """KILN_DSA_CP_ALL_LOCAL / KILN_DSA_CP_PAGE_KEYS at tp 2 (index_topk 64: keep 16 pools; page 8 at A = 2: one local
    pool per page): decode contexts below, at and past keep A kpool = 128 tokens, so the direct list is taken where the
    bucket's local pools fit keep and declined where they do not. The greedy tokens and logprobs equal the CP engine's
    without the flags exactly (and transformers')."""
    from kiln.engine.request import SamplingParams
    from kiln.models import mla

    path = str(tmp_path)
    hf = glm.build(path, seed=2, index_topk=64, max_position_embeddings=4096)
    if layout == "minimal":
        monkeypatch.setenv("KILN_DSA_KV", "minimal")
        monkeypatch.setattr(mla, "KV_LAYOUT", "minimal")
    else:
        monkeypatch.setenv("KILN_DSA_POOL_CACHE", "separate")
        monkeypatch.setattr(mla, "POOL_CACHE", "separate")
    ps = glm.prompts(6, (41, 9, 66, 119, 125))
    sp = SamplingParams(max_new_tokens=8, ignore_eos=True, logprobs=0)
    kw = dict(page_size=8, max_prefill_tokens=12, max_num_seqs=4)
    monkeypatch.setenv("KILN_DSA_CP", "1")
    eng = glm.engine(path, tp=2, **kw)
    try:
        want = eng.generate(ps, sp)
    finally:
        eng.close()
    assert [r.output_ids for r in want] == [glm.hf_greedy(hf, x, 8) for x in ps]
    monkeypatch.setenv("KILN_DSA_CP_ALL_LOCAL", str(all_local))
    monkeypatch.setenv("KILN_DSA_CP_PAGE_KEYS", str(page_keys))
    monkeypatch.setattr(mla, "CP_ALL_LOCAL", bool(all_local))
    monkeypatch.setattr(mla, "CP_PAGE_KEYS", bool(page_keys))
    seen = []
    real = mla._cp_local_all

    def spy(sc, nloc, prow, keep):
        seen.append(prow.shape[-1])
        return real(sc, nloc, prow, keep)

    monkeypatch.setattr(mla, "_cp_local_all", spy)
    eng = glm.engine(path, tp=2, **kw)
    try:
        got = eng.generate(ps, sp)
    finally:
        eng.close()
    assert [r.output_ids for r in got] == [r.output_ids for r in want]
    for a, b in zip(got, want):
        assert [x[0] for x in a.logprobs] == [x[0] for x in b.logprobs]
    if seen and all_local:  # the spy sees rank 0's calls when it runs in this process
        assert max(seen) <= 16


@pytest.mark.parametrize("A", [2, 8])
def test_cp_merge_bounded_bits(A, monkeypatch):
    """KILN_DSA_CP_MERGE_BOUND (dsa_long.merge_bits): the tie search over ceil(log2(cpools)) bits instead of 21 gives the
    same selection, ties straddling ranks included (every case of _cases, cpools = the context's pools rounded up to A)."""
    monkeypatch.setattr(dsa_long, "CP_MERGE_BOUND", True)
    assert dsa_long.merge_bits(4096) == 12 and dsa_long.merge_bits(None) == 21 and dsa_long.merge_bits(1) == 1
    for name, sc, keep in _cases():
        N, P = sc.shape
        vals, cps = [], []
        for r in range(A):
            loc = sc[:, r::A]
            if loc.shape[1] == 0:
                loc = torch.full((N, 1), NEG)
            lp, lc = dsa_long.select_reference(loc, keep)
            ok = torch.arange(keep).view(1, keep) < lc.view(N, 1)
            vals.append(torch.where(ok, loc.gather(1, lp), torch.tensor(NEG)))
            cps.append((lp * A + r).float())
        v, c = torch.stack(vals, 1), torch.stack(cps, 1)
        cpools = -(-P // A) * A
        want = dsa_long.cp_merge_rows(v, c, keep)
        got = dsa_long.cp_merge(v, c, keep, cpools=cpools)
        assert torch.equal(got, want), f"{name} A={A}"
