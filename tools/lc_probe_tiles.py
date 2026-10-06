"""Check models/mla._select_tiles (one dsa_long_select call per 128 queries) against the kernel's emulation at N > 128
inside one compiled graph (trn1)."""
import sys

import torch

sys.path.insert(0, ".")


def main():
    from kiln import platform

    platform.configure_runtime_env()
    import libtorch_neuronx_lite  # noqa: F401

    from kiln.kernels import dsa_long_select as dl
    from kiln.models import mla

    g = torch.Generator().manual_seed(0)
    for N, P in ((1024, 512), (1024, 2048), (384, 8448)):
        q = torch.randn(N, 32, 128, generator=g).to(torch.bfloat16)
        w = torch.randn(N, 32, generator=g)
        pk = torch.randn(P, 128, generator=g).to(torch.bfloat16)
        npool = torch.randint(0, P + 1, (N,), generator=g)
        want = dl.emulate(q, w, pk, npool, 512, 128 ** -0.5)
        f = torch.compile(lambda a, b, c, d: mla._select_tiles(a, b, c, d, 512, 128 ** -0.5), backend="neuron_libtorch",
                          fullgraph=True)
        dev = torch.device("neuron:0")
        got = f(q.to(dev), w.to(dev), pk.to(dev), npool.to(dev))
        p_got, c_got = got[0].cpu(), got[1].cpu()
        bad = 0
        for n in range(N):
            a = set(p_got[n, : int(c_got[n])].tolist())
            b = set(want[0][n, : int(want[1][n])].tolist())
            bad += a != b
        print(f"N={N} P={P}: {bad} of {N} rows differ from emulate", flush=True)


if __name__ == "__main__":
    main()
