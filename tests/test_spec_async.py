"""Speculative decoding under overlap scheduling (engine/spec_async.py, EngineConfig.spec_async): the board graphs on
hand-built boards, and engines whose MTP steps are scheduled blind (the accepted count, newest token, positions,
state rows and drafts kept on the device) giving the tokens, logprobs and acceptance of the synchronous MTP engine
(CPU, fp32): GLM-5.3-Flash's architecture (KDA state rows, pooled DSA, MoE, hyper-connections) at tp 1 and under
DP attention, and DeepSeek-V3 MTP."""

import pytest
import torch

from kiln.engine import spec_async as sa

LOGPROB_TOL = 1e-4


def test_board_graphs_follow_the_accepted_count():
    Q, k, ps = 2, 1, 4
    S = 6
    W = sa.width(Q, k)
    board = torch.zeros(S, W)
    # slot 1: newest token 50 at position 9 (T[0], acc 0), state row 7, draft 60; slot 3: last verify accepted
    # (T = 61, 70 with acc 1): newest token 70 at position 13, state row 9, draft 80
    board[1, :Q] = torch.tensor([50.0, 0.0])
    board[1, Q:Q + 4] = torch.tensor([0.0, 9.0, 7.0, 1.0])
    board[1, Q + 4] = 60.0
    board[3, :Q] = torch.tensor([61.0, 70.0])
    board[3, Q:Q + 4] = torch.tensor([1.0, 13.0, 9.0, 1.0])
    board[3, Q + 4] = 80.0
    slot_idx = torch.tensor([1, 3, 5])  # 5: padding (scratch)
    valid = torch.tensor([1.0, 1.0, 0.0])
    table = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [0, 0, 0, 0, 0]])
    rows = torch.tensor([[3, 4], [5, 6], [0, 0]])
    pad = torch.full((3, Q), -7, dtype=torch.int64)
    ids, draft, pos, slot, st = sa.spec_prep(board, slot_idx, table, rows, pad, valid, Q, k, ps, 1)
    assert ids[:2].tolist() == [[50, 60], [70, 80]]
    assert draft.view(3, Q)[:2].tolist() == [[60, 0], [80, 0]]
    assert pos[:2].tolist() == [[9, 10], [13, 14]]
    assert slot[:2].tolist() == [[12 * 4 + 1, 12 * 4 + 2], [23 * 4 + 1, 23 * 4 + 2]] and slot[2].tolist() == [-7, -7]
    assert st[:2].tolist() == [[7, 3, 4], [9, 5, 6]]
    # verify output: row 0 of slot 1 rejects (replacement 55), row 0 of slot 3 accepts, bonus 90
    cols = 6
    out = torch.zeros(3 * Q, cols)
    out[0] = torch.tensor([51.0, 0.0, 55.0, -1, -2, -3])
    out[1] = torch.tensor([52.0, 0.0, 0.0, -1, -2, -3])
    out[2] = torch.tensor([80.0, 1.0, 81.0, -1, -2, -3])
    out[3] = torch.tensor([90.0, 0.0, 91.0, -1, -2, -3])
    res = sa.spec_post(board, slot_idx, out, ids, rows.float(), valid, Q, k)
    assert res[:2].tolist() == [[55.0, 0.0, 0.0, 1.0], [80.0, 90.0, 1.0, 1.0]]
    assert board[1, :Q + 4].tolist() == [55.0, 0.0, 0.0, 10.0, 3.0, 1.0]  # newest 55 at 10, state row rows[0] = 3
    assert board[3, :Q + 4].tolist() == [80.0, 90.0, 1.0, 15.0, 6.0, 1.0]  # newest 90 at 15, state row rows[1] = 6
    T, last, mpos, mslot = sa.mtp_prep(board, slot_idx, table, pad, valid, Q, ps, 1)
    assert T[:2].tolist() == [[55, 0], [80, 90]] and last[:2].tolist() == [0, 1]
    assert mpos[:2].tolist() == [[9, 10], [13, 14]]
    sa.mtp_post(board, slot_idx, torch.tensor([[66.0], [99.0], [5.0]]), valid, Q, k)
    assert board[1, Q + 4] == 66.0 and board[3, Q + 4] == 99.0 and board[5].abs().sum() == 0
    # the next prep: slot 1's newest is T[0] = 55, slot 3's T[1] = 90
    ids2, _, pos2, _, st2 = sa.spec_prep(board, slot_idx, table, rows, pad, valid, Q, k, ps, 1)
    assert ids2[:2].tolist() == [[55, 66], [90, 99]] and pos2[:2].tolist() == [[10, 11], [15, 16]]
    assert st2[:2, 0].tolist() == [3, 6]
    # after a prefill: the token board's sampled token, acc 0
    tb = torch.zeros(S)
    tb[4] = 77.0
    sa.spec_init(board, tb, torch.tensor([4, 5]), torch.tensor([20.0, 0.0]), torch.tensor([2.0, 0.0]),
                 torch.tensor([1.0, 0.0]), Q, k)
    assert board[4, :Q + 4].tolist() == [77.0, 0.0, 0.0, 20.0, 2.0, 0.0]  # no drafts yet
    # a draftless row (nd 0) takes the sample y_0 whatever the accept flags
    out2 = torch.zeros(2 * Q, cols)
    out2[0] = torch.tensor([33.0, 1.0, 34.0, -1, -2, -3])
    res2 = sa.spec_post(board, torch.tensor([4, 5]), out2, torch.tensor([[77, 0], [0, 0]]), torch.tensor([[1.0, 2.0],
                        [0.0, 0.0]]), torch.tensor([1.0, 0.0]), Q, k)
    assert res2[0].tolist() == [33.0, 0.0, 0.0, 0.0] and board[4, Q + 1] == 21.0
    sa.spec_host_init(board, torch.tensor([2]), torch.tensor([44.0]), torch.tensor([30.0]), torch.tensor([5.0]),
                      torch.tensor([1.0]), Q, k)
    assert board[2, :Q + 4].tolist() == [44.0, 0.0, 0.0, 30.0, 5.0, 0.0]
    got = sa.mtp_prefill_ids(tb, torch.tensor([4, 5]), torch.tensor([[1, 2, 3], [4, 5, 6]]), torch.tensor([2, -1]))
    assert got.tolist() == [[1, 2, 77], [4, 5, 6]]


def _gen(eng, prompts, n, logprobs=1):
    from kiln.engine.request import SamplingParams

    reqs = eng.generate(prompts, SamplingParams(max_new_tokens=n, ignore_eos=True, logprobs=logprobs))
    return ([r.output_ids for r in reqs], [[x[0] for x in r.logprobs] for r in reqs],
            (eng.spec_proposed, eng.spec_accepted))


def _compare(name, got, ref, spec=True):
    (ids, lp, acc), (ids0, lp0, acc0) = got, ref
    err = max(abs(x - y) for p, q in zip(lp, lp0) for x, y in zip(p, q))
    print(f"{name}: tokens {'equal' if ids == ids0 else 'DIFFER'}, max |dlogprob| {err:.2e}, accepted "
          f"{acc[1]}/{acc[0]} (sync {acc0[1]}/{acc0[0]})")
    assert ids == ids0, name
    assert err < LOGPROB_TOL, (name, err)
    if spec:  # a blind step always carries k drafts (the synchronous one none at a request's last token, and its
        assert acc[1] >= acc0[1], (name, acc, acc0)  # tokens past the limit are dropped at commit): at least as many


def test_glm5_next_async_mtp_equals_sync(tmp_path):
    """GLM-5.3-Flash with its MTP layer, k=1: blind speculative steps under overlap scheduling give the
    synchronous MTP engine's tokens, logprobs and accepted counts (and plain greedy's tokens), at tp 1 with
    chunked prefill and under DP attention (tp 2, two groups)."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from tests.test_glm5_next import engine as eng_, prompts as mk
    from tests.test_linear_serving import build_glm5_next_mtp

    build_glm5_next_mtp(str(tmp_path))
    ps = mk(61, (9, 21, 14))
    plain = _gen(eng_(str(tmp_path), max_num_seqs=3, max_prefill_tokens=16), ps, 12)
    for kw in (dict(), dict(tp=2, dp_attention=2, max_num_seqs=4)):
        base = dict(max_num_seqs=3, max_prefill_tokens=16, spec_method="mtp", spec_k=1)
        base.update(kw)
        sync = eng_(str(tmp_path), **base)
        try:
            ref = _gen(sync, ps, 12)
        finally:
            sync.close()
        asy = eng_(str(tmp_path), overlap=True, spec_async=True, **base)
        try:
            assert asy.runner.spec_async
            got = _gen(asy, ps, 12)
        finally:
            asy.close()
        _compare(f"glm5_next async MTP {kw}", got, ref)
        assert got[0] == plain[0]


@pytest.mark.parametrize("name", ["deepseek_v3", "glm_moe_dsa"])
def test_mla_async_mtp_with_accepted_drafts_equals_sync(tmp_path, name):
    """DeepSeek-V3 and GLM-5.3 (MLA, DSA) whose MTP layer is the target's own copy (tests/test_mtp_mla.py
    build_with_mtp copy_main: most drafts accepted, so both branches of every board update run), k=1: the blind
    engine's tokens, logprobs and accepted count equal the synchronous MTP engine's, and its tokens plain greedy's."""
    from tests.test_mtp_mla import MODELS, build_with_mtp, engine

    build_with_mtp(name, str(tmp_path), seed=2, copy_main=True, num_hidden_layers=1, **MODELS[name])
    g = torch.Generator().manual_seed(4)
    ps = [torch.randint(0, 384, (n,), generator=g).tolist() for n in (20, 45, 9)]
    plain = engine(str(tmp_path), spec_method=None, max_num_seqs=3)
    want = _gen(plain, ps, 24)
    plain.close()
    sync = engine(str(tmp_path), max_num_seqs=3, spec_k=1)
    ref = _gen(sync, ps, 24)
    sync.close()
    asy = engine(str(tmp_path), max_num_seqs=3, spec_k=1, overlap=True, spec_async=True)
    assert asy.runner.spec_async
    w = asy.warmup()  # the small graphs too, on scratch slots only: the output below is unchanged
    assert any(k[0] == "spec_prep" for k in asy.runner.compile_seconds), w
    got = _gen(asy, ps, 24)
    asy.close()
    _compare(f"{name} async MTP (copied layer)", got, ref)
    assert got[0] == want[0]
    assert ref[2][1] / max(ref[2][0], 1) > 0.5  # the accepted branch ran


def test_glm5_next_async_with_oracle_drafts_accepts_and_keeps_greedy(tmp_path, monkeypatch):
    """GLM-5.3-Flash with oracle drafts (each MTP pass's draft replaced by plain greedy decoding's next token, read
    from the board's own position): nearly every draft is accepted, so the KDA layers continue from the request's
    second state row (cur = rows[acc]) step after step, and the output is still plain greedy's, at tp 1 and under
    tp 1 (the hook sees rank 0's calls only)."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from tests.test_glm5_next import engine as eng_, prompts as mk
    from tests.test_linear_serving import build_glm5_next_mtp

    build_glm5_next_mtp(str(tmp_path))
    ps = mk(62, (9, 21, 14))
    plain = _gen(eng_(str(tmp_path), max_num_seqs=3, max_prefill_tokens=16), ps, 12)
    full = [p + o for p, o in zip(ps, plain[0])]
    real_post = sa.mtp_post
    for kw in (dict(),):
        base = dict(max_num_seqs=3, max_prefill_tokens=16, spec_method="mtp", spec_k=1, overlap=True, spec_async=True)
        base.update(kw)
        asy = eng_(str(tmp_path), **base)
        try:
            slot_of = {}

            def oracle(board, slot_idx, drafts, valid, Q, k):
                d = drafts.clone()
                for i, (s_, v) in enumerate(zip(slot_idx.tolist(), valid.tolist())):
                    if v > 0.5 and s_ in slot_of:
                        nxt = int(board[s_, Q + 1]) + 1  # the draft is the token after the newest one
                        toks = full[slot_of[s_]]
                        if nxt < len(toks):
                            d[i, 0] = toks[nxt]
                return real_post(board, slot_idx, d, valid, Q, k)

            monkeypatch.setattr(sa, "mtp_post", oracle)
            from kiln.engine.request import SamplingParams

            reqs = [asy.add_request(p, SamplingParams(max_new_tokens=12, ignore_eos=True)) for p in ps]
            while asy.has_work():
                for j, r in enumerate(reqs):
                    if r.rid in asy.runner._slot:
                        slot_of[asy.runner._slot[r.rid]] = j
                asy.step()
            out = [r.output_ids for r in reqs]
            acc = (asy.spec_proposed, asy.spec_accepted)
        finally:
            monkeypatch.setattr(sa, "mtp_post", real_post)
            asy.close()
        print(f"glm5_next oracle drafts {kw}: accepted {acc[1]}/{acc[0]}, output {'equal' if out == plain[0] else 'DIFFERS'}")
        assert out == plain[0]
        assert acc[1] / acc[0] > 0.5



@pytest.mark.parametrize("name", ["glm5_next", "deepseek_v3"])
def test_async_mtp_sampled_equals_sync(tmp_path, monkeypatch, name):
    """Sampled requests (temperature 0.8, seeded), the prompts twice in one engine: the blind engine draws each
    request's noise in the synchronous engine's order, so its tokens and logprobs are the synchronous MTP engine's
    up to each request's last token (there the synchronous engine has no room for a draft and takes the sample y_0,
    the blind step verifies one and emits the accepted draft, the replacement or the bonus token: another exact
    sample of the same distribution). DeepSeek-V3's second round hits the prefix cache for whole prompts, so each
    such request's newest token is a prompt token the host holds (spec_host_init, a row without drafts): it
    reports the sample y_0's logprob and counts as a decode, as the synchronous engine's draftless verify rows do."""
    from kiln.engine.request import SamplingParams

    if name == "glm5_next":
        pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
        from tests.test_glm5_next import engine as eng_, prompts as mk
        from tests.test_linear_serving import build_glm5_next_mtp

        build_glm5_next_mtp(str(tmp_path))
        ps = mk(63, (9, 21, 14, 30))
        base = dict(max_num_seqs=4, max_prefill_tokens=16, spec_method="mtp", spec_k=1)
    else:
        from tests.test_mtp_mla import MODELS, build_with_mtp, engine as eng_

        build_with_mtp(name, str(tmp_path), seed=2, copy_main=True, num_hidden_layers=1, **MODELS[name])
        g = torch.Generator().manual_seed(5)
        ps = [torch.randint(0, 384, (n,), generator=g).tolist() for n in (20, 45, 9)]
        base = dict(max_num_seqs=3, spec_k=1)
    sp = SamplingParams(max_new_tokens=12, ignore_eos=True, logprobs=1, temperature=0.8, seed=5)

    def run(eng):
        try:
            reqs = eng.generate(ps, sp) + eng.generate(ps, sp)
            return ([r.output_ids for r in reqs], [[x[0] for x in r.logprobs] for r in reqs],
                    (eng.spec_proposed, eng.spec_accepted))
        finally:
            eng.close()

    ref = run(eng_(str(tmp_path), **base))
    inits = []
    real_init = sa.spec_host_init

    def counted(board, slot_idx, tok, b, cur, valid, Q, k):
        inits.append(int((valid > 0).sum()))
        return real_init(board, slot_idx, tok, b, cur, valid, Q, k)

    monkeypatch.setattr(sa, "spec_host_init", counted)
    got = run(eng_(str(tmp_path), overlap=True, spec_async=True, **base))
    print(f"{name}: host-initialized rows {sum(inits)}")
    cut = lambda r: ([x[:-1] for x in r[0]], [x[:-1] for x in r[1]], r[2])  # noqa: E731
    _compare(f"{name} async MTP sampled", cut(got), cut(ref), spec=False)
    if name == "deepseek_v3":
        assert sum(inits) > 0  # the draftless path ran
