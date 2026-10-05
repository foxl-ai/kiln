"""Which process state makes libtorch_neuronx_lite record torch.topk with normalised
kwargs? Each variant runs in a fresh subprocess and reports what LNL wrote to fxgraph.txt.

    python tools/probe_fx_normalisation.py
"""

import os
import subprocess
import sys
import textwrap

BODY = """
import glob, os, sys, time
variant = sys.argv[1]
import torch
if "tf" in variant:
    from transformers import AutoTokenizer  # noqa: F401
if "dist" in variant:
    import torch.distributed as dist
    dist.init_process_group("gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:%d" % (29000 + os.getpid() % 900))
if "spawn" in variant:
    import multiprocessing as mp
    p = mp.get_context("spawn").Process(target=print, args=("child",)); p.start(); p.join()
import libtorch_neuronx_lite  # noqa: F401
dev = torch.device("neuron:0")
k = float(len(variant)) + 0.5  # distinct constant per variant -> distinct cache entry
def f(x):
    v, i = torch.topk(x * k, 8, dim=-1)
    return v, i
t0 = time.time()
torch.compile(f, backend="neuron_libtorch", fullgraph=True)(torch.randn(4, 64).to(dev))[0].cpu()
root = os.path.expanduser("~/.cache/neuron_libtorch/neuron/compile_cache")
new = [d for d in glob.glob(root + "/*/fxgraph.txt") if os.path.getmtime(d) >= t0 - 1]
line = [l.strip() for l in open(new[0]) if "torch.topk" in l][0] if new else "no new entry"
print("RESULT", variant, "normalised" if "largest" in line else "plain", "|", line[:140])
"""

if __name__ == "__main__":
    path = "/tmp/kiln_fx_probe.py"
    open(path, "w").write(textwrap.dedent(BODY))
    for v in ["plain", "tf", "dist", "spawn", "tf+dist+spawn", "dist+spawn"]:
        out = subprocess.run([sys.executable, path, v], capture_output=True, text=True, timeout=900, cwd="/tmp")
        res = [l for l in (out.stdout + out.stderr).splitlines() if l.startswith("RESULT")]
        print(res[0] if res else f"{v}: no result, exit {out.returncode}: {(out.stdout + out.stderr)[-300:]}")
