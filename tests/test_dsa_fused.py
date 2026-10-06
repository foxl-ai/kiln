"""kernels/dsa_fused.py in nki.simulate, through attend() exactly as the device call makes it, against emulate().

The head counts are the point: attention TP 8 gives GLM-5.3-Flash 8 MLA heads per rank, attention TP 32 (DP attention 1)
gives 2. With fewer than 3 heads per call the kernel's two-block latent ring was overwritten before its last head read
it (head 1 off by 2.1 at H = 2), which made every DP-attention-1 prefill on the device wrong; attend() now pads such
calls to MIN_HEADS. Also a query count that is not a whole tile (attend's row padding) and a context of several key
blocks (the hazard needs a third block). Needs the nki package (the Neuron venv); no device.
"""

from __future__ import annotations

import pytest
import torch

nki = pytest.importorskip("nki")

from kiln.kernels import dsa_fused  # noqa: E402


def inputs(H, C, P, Hi=4, D=128, R=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    L = 4 * P
    qI = torch.randn(C, Hi, D, generator=g).bfloat16().float()
    w = torch.rand(C, Hi, generator=g)
    pk = torch.randn(P, D, generator=g).bfloat16().float()
    pos = torch.sort(torch.randint(0, L, (C,), generator=g)).values
    pos[-1] = L - 1
    q_lat = (torch.randn(C, H, R, generator=g) * 0.05).bfloat16().float()
    kc = torch.randn(L, R, generator=g).bfloat16().float()
    return qI, w, pk, pos, q_lat, kc


@pytest.mark.parametrize("H,C", [(8, 128), (3, 128), (2, 128), (1, 128), (2, 100)])
def test_fused_kernel_matches_emulate_at_every_head_count(H, C):
    P, keep = 768, 128  # 3 key blocks of 1024 keys
    qI, w, pk, pos, q_lat, kc = inputs(H, C, P)
    D, R = qI.shape[-1], kc.shape[-1]
    want = dsa_fused.emulate(qI, w, pk, pos, q_lat, kc, keep, D ** -0.5, R ** -0.5)
    got = dsa_fused.attend(qI, w, pk, pos, q_lat, kc, keep, D ** -0.5, R ** -0.5, simulate=True).float()
    assert got.shape == want.shape == (C, H, R)
    err = [(got[:, h] - want[:, h]).abs().max().item() for h in range(H)]
    print(f"H={H} C={C}: max |kernel - emulate| per head {[round(e, 4) for e in err]}")
    assert max(err) < 1e-2, err
