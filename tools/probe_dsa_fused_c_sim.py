"""Sanity check that nki.simulate honours the causal kernel's 0 / 1 trips: force a wrong variant and see the output move.

    python tools/probe_dsa_fused_c_sim.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import nki
from kiln.kernels import dsa_fused, dsa_fused_c as c
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
from test_dsa_fused_c import _inputs

H, C, P, keep = 8, 128, 768, 128
qI, w, pk, pos, q_lat, kc = _inputs(H, C, P, 2800, C)
D, R = 128, 512
want = dsa_fused.attend(qI, w, pk, pos, q_lat, kc, keep, D ** -0.5, R ** -0.5, simulate=True)
ks = c.ladder(3)
eye = torch.eye(128).to(torch.bfloat16)
for name, vt in (("right (k=3)", c.variant_onehot(pos, 4 * P, ks)), ("forced k=1", torch.tensor([[1, 0, 0]], dtype=torch.int32)),
                 ("none", torch.zeros(1, 3, dtype=torch.int32))):
    o = torch.as_tensor(nki.simulate(c.kiln_dsa_fused_c_kernel)(
        qT=qI.to(torch.bfloat16).permute(1, 2, 0).contiguous(), w=w.float().contiguous(),
        pkT=pk.to(torch.bfloat16).t().contiguous(), posf=pos.float().contiguous(),
        q_lat=q_lat.to(torch.bfloat16).contiguous(), kc=kc.to(torch.bfloat16).contiguous(), identb=eye, vt=vt,
        keep=keep, nbits=P.bit_length(), scale_i=D ** -0.5, scale_a=R ** -0.5, lad=c._ladder_bits(ks), rev=c.REV))
    print(name, vt.tolist(), "equal", torch.equal(o, want), "max|d|", float((o - want).abs().max()), flush=True)
