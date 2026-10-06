"""kernels/dsa_index.py, the decode-side indexer scores: the host path against the dense definition, and (where the NKI
package is installed) both kernels, bf16 and fp8 e4m3 pool keys, under the NKI simulator."""

import importlib.util

import pytest
import torch

from kiln.kernels import dsa_index

HI, D, PPP = 4, 128, 8


def case(B=2, pages=128, seed=0, fp8=False):
    g = torch.Generator().manual_seed(seed)
    npg = B * pages + 1
    pkc = (torch.randn(npg, PPP * D, generator=g) * 0.5).to(torch.bfloat16)
    if fp8:
        pkc = pkc.to(torch.float8_e4m3fn)
    table = (torch.randperm(npg - 1, generator=g)[: B * pages].view(B, pages) + 1).to(torch.int32)
    q = (torch.randn(B, HI, D, generator=g) * 0.1).to(torch.bfloat16)
    w = torch.randn(B, HI, generator=g) * D ** -0.5
    cand = torch.zeros(B, pages * PPP)
    cand[:, -3:] = dsa_index.NEG_INF
    return q, w, pkc, table, cand


def dense(q, w, pkc, table, cand):
    B, npg = table.shape
    k = pkc[table.long()].reshape(B, npg * PPP, D).float()
    s = torch.einsum("bhd,bpd->bhp", q.float(), k)
    return torch.einsum("bh,bhp->bp", w, torch.relu(s)) + cand


@pytest.mark.parametrize("fp8", [False, True])
def test_host_scores_are_the_dense_definition(fp8):
    q, w, pkc, table, cand = case(fp8=fp8)
    got = dsa_index.scores(q, w, pkc, dsa_index.page_groups(table), cand)
    want = dense(q, w, pkc, table, cand)
    ok = want > -1e29
    assert torch.allclose(got[ok], want[ok], rtol=1e-5, atol=1e-6)
    assert ((got < -1e29) == ~ok).all()


def test_fp8_keys_score_as_their_bf16_values():
    q, w, pkc8, table, cand = case(fp8=True)
    a = dsa_index.scores(q, w, pkc8, dsa_index.page_groups(table), cand)
    b = dsa_index.scores(q, w, pkc8.to(torch.bfloat16), dsa_index.page_groups(table), cand)
    assert torch.equal(a, b)


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
@pytest.mark.parametrize("fp8", [False, True])
def test_nki_simulator_matches_the_host(fp8, monkeypatch):
    import nki

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    q, w, pkc, table, cand = case(B=2, pages=128, seed=3, fp8=fp8)
    ppg = dsa_index.page_groups(table)
    want = dsa_index.scores(q, w, pkc, ppg, cand)
    B = q.shape[0]
    qT = q.reshape(B * HI, D).t().contiguous()
    wb = w.float().repeat(1, 4).unsqueeze(1).expand(B, 128, 4 * HI).contiguous()
    eye = torch.eye(128).to(torch.bfloat16)
    k = dsa_index.kiln_dsa_index_fp8_kernel if fp8 else dsa_index.kiln_dsa_index_kernel
    got = torch.as_tensor(nki.simulate(k)(qT=qT, wb=wb, pkc=pkc, ppg=ppg, cand=cand, identb=eye,
                                          rev=dsa_index.REV8 if fp8 else dsa_index.REV, dge=0, spl=1)).float()
    ok = want > -1e29
    assert ((got - want).abs()[ok].max() / want[ok].abs().max()).item() < 1e-5
    assert ((got < -1e29) == ~ok).all()
