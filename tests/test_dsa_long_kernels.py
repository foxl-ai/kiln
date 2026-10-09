"""The long-context DSA kernels' host forms (kernels/dsa_long_select.py, kernels/dsa_slots.py): the selection kernel's
algorithm step for step (sub-block maxima, extraction rounds, ascending sub-block order, candidates, the final sort)
against models/dsa_long.select_reference on adversarial ties, the emulation's scores and outputs, the kernel's input
layout, and the slot attention's emulation (dsa_decode's arithmetic plus the log-sum-exp). The device kernels are
checked against these on a NeuronCore by tools/probe_dsa_long.py."""

import pytest
import torch

from kiln.kernels import dsa_long_select as dl
from kiln.kernels import dsa_slots
from kiln.models import dsa_long

NEG = dsa_long.NEG_INF


def _subs(P: int, keep: int):
    """Every sub-block size the kernel takes at P pools (0: one level)."""
    out = []
    for sub in (0, 2, 4, 8, 16, 32):
        if dl.supported(32, 128, P, keep, sub):
            out.append(sub)
    return out


def _wide_cases():
    """(name, scores, keep) at the kernel's keep sizes: tests/test_dsa_long.py's adversarial kinds at P where keep
    divides by 8, plus keep 512 over thousands of pools with plateaus at the threshold across sub-block boundaries."""
    from tests.test_dsa_long import _cases

    out = [(n, s, k) for n, s, k in _cases() if k % 8 == 0]
    g = torch.Generator().manual_seed(2)
    for P in (2112, 8448):
        N = 6
        npool = torch.tensor([0, 511, 512, 513, P // 2, P])
        cand = torch.arange(P).view(1, P) < npool.view(N, 1)
        for kind in ("randn", "int", "plateau", "zeros"):
            if kind == "randn":
                s = torch.randn(N, P, generator=g)
            elif kind == "int":
                s = torch.randint(0, 3, (N, P), generator=g).float()
            elif kind == "plateau":
                s = torch.full((N, P), 1.0)
                s[:, ::7] = 2.0
                s[:, -5:] = 9.0
            else:
                s = torch.zeros(N, P)
            out.append((f"{kind} P{P} k512", torch.where(cand, s, NEG), 512))
    return out


def test_selection_algorithm_equals_reference():
    """emulate_algorithm (the kernel's steps) == select_reference, pools and their scores, for every sub-block size
    the kernel takes, on adversarial ties."""
    n = 0
    for name, sc, keep in _wide_cases():
        want_p, want_c = dsa_long.select_reference(sc, keep)
        k = torch.arange(keep).view(1, keep)
        want_v = torch.where(k < want_c.view(-1, 1), sc.gather(1, want_p), NEG)
        for sub in _subs(sc.shape[1], keep):
            got_p, got_v = dl.emulate_algorithm(sc, keep, sub)
            assert torch.equal(got_p, want_p), f"{name} sub {sub}"
            assert torch.equal(got_v, want_v), f"{name} sub {sub}"
            n += 1
    assert n > 50


def test_pick_sub():
    """The two levels are used only where they extract over fewer values than the whole row."""
    assert dl.pick_sub(2112, 512) in (0, 2)
    for P in (32768, 131072, 262144):
        sub = dl.pick_sub(P, 512)
        Pp = -(-P // dl.CH) * dl.CH
        assert sub and Pp // sub + 512 * sub < Pp and dl.supported(32, 128, P, 512, sub)
    with pytest.raises(NotImplementedError):
        dl.pick_sub(16384 * 64, 512)


def test_emulate_is_reference_on_its_scores():
    """emulate(): the scores of emulate_scores (two accumulator chains), then the exact selection, with the selected
    pools' scores and NEG_INF past the count; those scores are dsa_topk.emulate_scores' (one head-order chain) up to
    fp32 rounding."""
    from kiln.kernels import dsa_topk

    g = torch.Generator().manual_seed(4)
    N, Hi, D, P, keep = 12, 32, 128, 3000, 64
    q = torch.randn(N, Hi, D, generator=g).to(torch.bfloat16)
    w = torch.randn(N, Hi, generator=g) * Hi ** -0.5
    pk = torch.randn(P, D, generator=g).to(torch.bfloat16)
    npool = torch.tensor([0, 1, 63, 64, 65, 1500, 2999, 3000, 3000, 700, 9, 2048])
    pools, cnt, vals = dl.emulate(q, w, pk, npool, keep, 128 ** -0.5)
    cand = torch.where(torch.arange(P).view(1, P) < npool.view(N, 1), 0.0, NEG)
    sc = dl.emulate_scores(q, w, pk, cand, 128 ** -0.5)
    rp, rc = dsa_long.select_reference(sc, keep)
    assert torch.equal(pools, rp) and torch.equal(cnt, rc) and torch.equal(cnt, npool.clamp(max=keep))
    k = torch.arange(keep).view(1, keep)
    assert torch.equal(vals, torch.where(k < cnt.view(N, 1), sc.gather(1, pools), NEG))
    one = dsa_topk.emulate_scores(q, w, pk, cand, 128 ** -0.5)
    assert ((one - sc).abs() <= 1e-5 * one.abs().clamp(min=1)).all()  # the same score up to fp32 rounding


def test_kernel_input_layout():
    """kernel_inputs: tile t's queries transposed head-major ([t, d, h, q] = qI[128 t + q, h, d]), weights and
    candidate counts per tile, padded query rows with no candidate, pools padded to whole chunks of 512."""
    g = torch.Generator().manual_seed(5)
    N, Hi, D, P = 200, 32, 128, 1000
    q = torch.randn(N, Hi, D, generator=g)
    w = torch.randn(N, Hi, generator=g)
    pk = torch.randn(P, D, generator=g)
    npool = torch.randint(0, P + 1, (N,), generator=g)
    kin = dl.kernel_inputs(q, w, pk, npool)
    assert kin["qT"].shape == (2, D, Hi, 128) and kin["pk"].shape == (1024, D)
    t, qq, h, d = 1, 50, 7, 33
    assert kin["qT"][t, d, h, qq] == q[128 * t + qq, h, d].to(torch.bfloat16)
    assert kin["w"][t, qq, h] == w[128 * t + qq, h]
    assert kin["npool"][t, qq, 0] == float(npool[128 * t + qq])
    assert (kin["npool"].reshape(-1)[N:] == 0).all() and (kin["pk"][P:] == 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_slots_emulation(dtype):
    """dsa_slots.emulate: dsa_decode.emulate's output, and the log-sum-exp of the scaled, biased scores per (row,
    head); a row whose every slot is NEG_INF stays finite (uniform weights, lse ~NEG_INF)."""
    from kiln.kernels import dsa_decode

    g = torch.Generator().manual_seed(6)
    N, H, R, NS = 5, 4, 64, 6
    kc = torch.randn(64 * 4, 1, R, generator=g).to(dtype)
    q = torch.randn(N, H, R, generator=g) * 0.1
    rows = torch.randint(0, 64, (N, NS), generator=g)
    bias = torch.zeros(N, NS, 4)
    bias[:, 4:] = NEG
    bias[1, 3, 2:] = NEG
    bias[3] = NEG
    o, lse = dsa_slots.emulate(q, kc, rows, bias, 0.125, lse=True)
    assert torch.equal(o, dsa_decode.emulate(q, kc.reshape(-1, R), rows, bias, 0.125))
    assert torch.isfinite(o).all() and torch.isfinite(lse).all()
    tok = (rows.unsqueeze(-1) * 4 + torch.arange(4)).reshape(N, -1)
    K = kc.reshape(-1, R)[tok].float()
    if dtype != torch.float32:
        q = q.to(torch.bfloat16).float()
    s = torch.einsum("bhr,btr->bht", q, K) * 0.125 + bias.reshape(N, 1, -1)
    torch.testing.assert_close(lse, torch.logsumexp(s, -1), rtol=1e-6, atol=1e-5)
    assert (lse[3] < -1e29).all()
    tol = 1e-5 if dtype == torch.float32 else 5e-3  # (a bf16 cache rounds p to bf16, as the kernel does)
    torch.testing.assert_close(o[3], K[3].mean(0).expand(H, R), rtol=tol, atol=tol)  # uniform over the row's tokens


def test_long_select_value_order():
    """kernels/dsa_long_select.py vorder (the CP slot classes' local lists): the same pools, scores and count as the
    ascending form, ordered by score descending and equal scores by pool ascending, 0 / NEG_INF past the count (integer
    scores for ties; a context shorter than keep)."""
    from kiln.kernels import dsa_long_select

    g = torch.Generator().manual_seed(9)
    N, Hi, D, P, keep = 6, 4, 128, 700, 64
    qI = torch.randint(-2, 3, (N, Hi, D), generator=g).to(torch.bfloat16)
    w = torch.randint(0, 3, (N, Hi), generator=g).float()
    pk = torch.randint(-1, 2, (P, D), generator=g).to(torch.bfloat16)
    npool = torch.tensor([0, 5, keep, keep + 3, 400, P])
    a_p, a_c, a_v = dsa_long_select.emulate(qI, w, pk, npool, keep, 0.25)
    b_p, b_c, b_v = dsa_long_select.emulate(qI, w, pk, npool, keep, 0.25, vorder=True)
    assert torch.equal(a_c, b_c)
    for n in range(N):
        c = int(a_c[n])
        assert sorted(a_p[n, :c].tolist()) == sorted(b_p[n, :c].tolist())
        pairs = list(zip(b_v[n, :c].tolist(), b_p[n, :c].tolist()))
        assert pairs == sorted(pairs, key=lambda x: (-x[0], x[1]))
        assert (b_p[n, c:] == 0).all() and (b_v[n, c:] < -1e29).all()
        got = dict(zip(b_p[n, :c].tolist(), b_v[n, :c].tolist()))
        assert got == dict(zip(a_p[n, :c].tolist(), a_v[n, :c].tolist()))


def _slots_c_case(N=6, H=8, R=128, NS=256, pools=512, seed=1, dtype=torch.float32):
    """(q, kc, rows, bias): the context-parallel decode shape in small, each row a random set of live slots (a different
    count per row), the tail pool's visible prefix on every other row, one row fully masked."""
    g = torch.Generator().manual_seed(seed)
    kc = torch.randn(pools * 4, R, generator=g).to(dtype)
    q = (torch.randn(N, H, R, generator=g) * 0.1).to(dtype)
    rows = torch.randint(0, pools, (N, NS), generator=g)
    bias = torch.full((N, NS, 4), NEG)
    for b in range(N):
        sel = torch.randperm(NS - 56, generator=g)[: 10 + 9 * b]
        bias[b, sel] = 0.0
        if b % 2 == 0:
            bias[b, NS - 56, : 1 + b % 4] = 0.0
    bias[N // 2] = NEG
    return q, kc, rows, bias


def test_slots_c_plan():
    """dsa_slots_c.plan: every live slot of a row exactly once in its compacted list (pool row and live tokens), the
    rest row 0 with nv 0, partition-major order; `fits` false past C live slots."""
    from kiln.kernels import dsa_slots_c

    q, kc, rows, bias = _slots_c_case(N=7, NS=640)
    bias[6, :300] = 0.0  # over C
    nv = dsa_slots_c.live_tokens(bias)
    crow, cnv, fits = dsa_slots_c.plan(rows, nv)
    for b in range(6):
        live = (nv[b] > 0).nonzero().view(-1)
        n = live.numel()
        assert bool(fits[b]) and n <= dsa_slots_c.C
        p, ch = live % 128, live // 128
        order = live[torch.argsort(p * 8 + ch)]  # partition-major
        assert torch.equal(crow[b, :n], rows[b, order]) and torch.equal(cnv[b, :n], nv[b, order])
        assert (crow[b, n:] == 0).all() and (cnv[b, n:] == 0).all()
    assert not bool(fits[6])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_slots_c_emulation(dtype):
    """dsa_slots_c.emulate: each row over its compacted live slots equals dsa_slots.emulate over all slots (to fp32
    summation order) on every row with a visible token; a call with a row over C live slots is dsa_slots.emulate
    exactly (the kernel's full loop)."""
    from kiln.kernels import dsa_slots_c

    q, kc, rows, bias = _slots_c_case(NS=640, dtype=dtype)
    o, lse = dsa_slots_c.emulate(q, kc, rows, bias, 0.125, lse=True)
    wo, wl = dsa_slots.emulate(q, kc, rows, bias, 0.125, lse=True)
    real = (bias > NEG / 2).any(-1).any(-1)
    assert not real.all() and real.sum() == q.shape[0] - 1
    tol = 1e-5 if dtype == torch.float32 else 5e-3
    torch.testing.assert_close(o[real], wo[real], rtol=tol, atol=tol)
    torch.testing.assert_close(lse[real], wl[real], rtol=1e-6, atol=1e-5)
    assert (lse[~real] < -1e29).all() and torch.isfinite(o).all()
    over = bias.clone()
    over[0, :200] = 0.0
    o2, l2 = dsa_slots_c.emulate(q, kc, rows, over, 0.125, lse=True)
    w2, wl2 = dsa_slots.emulate(q, kc, rows, over, 0.125, lse=True)
    assert torch.equal(o2, w2) and torch.equal(l2, wl2)


@pytest.mark.parametrize("over", [False, True])
def test_slots_c_simulator(over, monkeypatch):
    """kiln_dsa_slots_c_kernel under nki.simulate (trn1): every row equals dsa_slots_c.emulate (rows padded to RPI);
    a call with a row over C live slots gives the kernel no iteration (attend's dsa_slots_n call takes it)."""
    nki = pytest.importorskip("nki")
    from kiln.kernels import dsa_slots_c

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    q, kc, rows, bias = _slots_c_case(N=6, R=512, NS=256, dtype=torch.bfloat16)  # (the kernel's PSUM holds R = 512)
    if over:
        bias[0, :150] = 0.0
    kw, N, Np, ov = dsa_slots_c.kernel_args(q, kc, rows, bias, 0.125, lse=True)
    assert float(ov) == float(over) and int(kw["n_c"]) == (0 if over else Np // dsa_slots_c.RPI)
    if over:
        return
    o, ls = nki.simulate(dsa_slots_c.kiln_dsa_slots_c_kernel)(**kw)
    H, R = q.shape[1], q.shape[2]
    o = torch.as_tensor(o).float().reshape(Np, H, R)[:N]
    ls = torch.as_tensor(ls).float().reshape(Np, H)[:N]
    wo, wl = dsa_slots_c.emulate(q.float(), kc, rows, bias, 0.125, lse=True)
    real = (bias > NEG / 2).any(-1).any(-1)
    err = ((o - wo)[real].abs().max() / wo[real].abs().max()).item()
    assert err < 1e-2, err
    assert ((ls - wl)[real].abs().max()).item() < 1e-2
    assert (ls[~real] < -1e29).all()
