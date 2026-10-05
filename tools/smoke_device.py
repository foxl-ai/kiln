"""Probe which device execution paths work on this Neuron instance.

Each probe is written to its own file and run in its own subprocess: NKI parses kernel
SOURCE, so a kernel defined under `python -c` fails with "entry function not found",
and a crash in one path must not hide the others.

Run on the box: `infra/fleet.sh py <name> tools/smoke_device.py [probe ...]`.
"""

import os
import subprocess
import sys
import tempfile
import textwrap

VLLM_VENV = "/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0"
JAX_VENV = "/opt/aws_neuronx_venv_jax_0_10"

PROBES = {
    # What the runtime reports, next to what kiln/platform.py chose before starting it.
    "platform": (VLLM_VENV, """
        import os, subprocess, torch
        from kiln import platform
        import libtorch_neuronx_lite
        print("kiln", platform.check_runtime())
        info = torch.classes.neuron.Runtime().get_instance_info()
        print("nrt instance info", list(info))
        print("NEURON_LOGICAL_NC_CONFIG", os.environ.get("NEURON_LOGICAL_NC_CONFIG"))
        out = subprocess.run(["neuron-ls", "--json-output"], capture_output=True, text=True).stdout
        print("neuron-ls", " ".join(out.split())[:1500])
    """),
    # NKI 0.6: an np.ndarray argument "compiles and executes standalone kernel, without a
    # framework" (nki/__init__.pyi, jit docstring). trn1/inf2 are target "gen2".
    "nki_standalone": (VLLM_VENV, """
        import time, numpy as np
        import nki, nki.language as nl

        @nki.jit
        def add(a, b):
            out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
            nl.store(out, nl.add(nl.load(a), nl.load(b)))
            return out

        a = np.random.rand(128, 512).astype(np.float32)
        b = np.random.rand(128, 512).astype(np.float32)
        t = time.time(); out = add(a, b); t1 = time.time() - t
        t = time.time(); out = add(a, b); t2 = time.time() - t
        print("max_err", float(np.abs(out - (a + b)).max()), f"first={t1:.2f}s second={t2:.3f}s")
    """),
    "lnl_native_device": (VLLM_VENV, """
        import time, torch
        import libtorch_neuronx_lite  # registers PrivateUse1 as "neuron" (_config.py: default "lite")
        print("device count", torch.neuron.device_count() if hasattr(torch.neuron, "device_count") else "n/a")
        x = torch.randn(256, 256, dtype=torch.bfloat16)
        y = torch.randn(256, 256, dtype=torch.bfloat16)
        t = time.time()
        z = (x.to("neuron:0") @ y.to("neuron:0")).cpu()
        print("eager matmul max_err", float((z.float() - (x.float() @ y.float())).abs().max()), f"{time.time()-t:.2f}s")
        import torch._dynamo
        print("dynamo backends", [b for b in torch._dynamo.list_backends() if "neuron" in b.lower()])
    """),
    # Steady-state cost of ONE call into a compiled graph: the floor under every decode
    # step. A tiny graph so device time is negligible and what remains is host overhead.
    "lnl_compiled_call_overhead": (VLLM_VENV, """
        import time, torch
        import libtorch_neuronx_lite
        dev = torch.device("neuron:0")
        w1 = torch.randn(1024, 1024, dtype=torch.bfloat16, device=dev)
        w2 = torch.randn(1024, 1024, dtype=torch.bfloat16, device=dev)
        def mlp(x):
            return torch.nn.functional.silu(x @ w1) @ w2
        for backend in ("neuron_libtorch", "neuron_libtorch_graph_capture"):
            try:
                f = torch.compile(mlp, backend=backend, fullgraph=True)
                x = torch.randn(8, 1024, dtype=torch.bfloat16, device=dev)
                t = time.time(); y = f(x); y.cpu(); first = time.time() - t
                n = 200
                t = time.perf_counter()
                for _ in range(n):
                    y = f(x)
                y.cpu()
                per_async = (time.perf_counter() - t) / n
                t = time.perf_counter()
                for _ in range(50):
                    f(x).cpu()
                per_sync = (time.perf_counter() - t) / 50
                ref = mlp(x.cpu().float() if False else x).cpu()
                print(backend, f"first={first:.1f}s per_call_async={per_async*1e6:.0f}us per_call_with_d2h={per_sync*1e6:.0f}us")
            except Exception as e:
                print(backend, "FAILED", type(e).__name__, str(e)[:300])
    """),
    # vllm-neuron calls NKI kernels inside compiled graphs through LNL's HOP wrapper:
    # `wrap_nki(kernel)[lnc](**kwargs)` (vllm_neuron/functional/attention/attention_cte.py,
    # libtorch_neuronx_lite/nki/nki_hop.py:641). Plain `nki.jit` on torch tensors needs
    # torch_neuronx, which this venv does not ship.
    "nki_from_torch": (VLLM_VENV, """
        import time, torch
        import libtorch_neuronx_lite
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
        import nki, nki.language as nl

        @nki.jit
        def add_kernel(a, b):
            out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
            nl.store(out, nl.add(nl.load(a), nl.load(b)))
            return out

        add = wrap_nki(add_kernel)
        dev = torch.device("neuron:0")
        # The grid must equal the runtime's LNC: "The LNC value must match the
        # NEURON_LOGICAL_NC_CONFIG environment variable" (nki/__init__.py).
        lnc = int(__import__("os").environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))
        print("lnc", lnc)
        def f(a, b):
            return add[lnc](a=a, b=b) * 2.0
        g = torch.compile(f, backend="neuron_libtorch", fullgraph=True)
        a = torch.rand(128, 512, device=dev); b = torch.rand(128, 512, device=dev)
        t = time.time(); out = g(a, b); out.cpu(); first = time.time() - t
        t = time.perf_counter()
        for _ in range(100):
            out = g(a, b)
        out.cpu(); per = (time.perf_counter() - t) / 100
        print("max_err", float((out.cpu() - 2 * (a.cpu() + b.cpu())).abs().max()), f"first={first:.1f}s per_call={per*1e6:.0f}us")
    """),
    "jax": (JAX_VENV, """
        import time, jax, jax.numpy as jnp
        print("devices", jax.devices())
        f = jax.jit(lambda a, b: jnp.tanh(a @ b))
        a = jnp.ones((512, 512), jnp.bfloat16); b = jnp.ones((512, 512), jnp.bfloat16)
        t = time.time(); r = f(a, b).block_until_ready(); t1 = time.time() - t
        t = time.time(); r = f(a, b).block_until_ready(); t2 = time.time() - t
        print("jit ok", r.dtype, r.shape, float(r[0, 0]), f"first={t1:.2f}s second={t2*1e3:.2f}ms", r.devices())
    """),
}


def main() -> int:
    names = sys.argv[1:] or list(PROBES)
    workdir = tempfile.mkdtemp(prefix="kiln-smoke-")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from kiln import platform

    platform.configure_runtime_env()  # every probe inherits the LNC Kiln runs at
    print("platform (before the runtime)", platform.describe())
    for name in names:
        venv, code = PROBES[name]
        path = os.path.join(workdir, f"probe_{name}.py")
        with open(path, "w") as f:
            f.write(textwrap.dedent(code))
        env = dict(os.environ, PATH=f"{venv}/bin:/opt/aws/neuron/bin:" + os.environ.get("PATH", ""))
        proc = subprocess.run([f"{venv}/bin/python", path], capture_output=True, text=True,
                              timeout=1800, env=env, cwd=workdir)
        lines = [l for l in (proc.stdout + proc.stderr).strip().splitlines() if "INFO" not in l]
        print(f"== {name}: exit {proc.returncode}")
        for line in lines[-14:]:
            print("   ", line[:300])
    return 0


if __name__ == "__main__":
    sys.exit(main())
