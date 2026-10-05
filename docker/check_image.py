"""Check that this environment holds every Neuron piece Kiln imports, and that they work.

    python docker/check_image.py            # imports, the compiler CLI, the torch.compile backend
    python docker/check_image.py --device   # also compiles one graph and runs it on a NeuronCore

The container build runs the first form (docker/Dockerfile), so an image that lacks a piece is never
built; the second form needs a NeuronCore (`docker run --device=/dev/neuron0 ...`). Every failure is
printed and the exit code is 1: nothing here falls back.

What is checked is what kiln/ imports (grep for libtorch_neuronx_lite and nki under kiln/):
  - libtorch_neuronx_lite (LNL): registers the "neuron" torch device and the torch.compile backend
    "neuron_libtorch" that kiln/engine/model_runner.py canonical_neuron_backend wraps; plus the
    compile-cache, capture and NKI-wrapping modules kiln/compile_cache.py, kiln/capture.py and
    kiln/kernels/*.py use;
  - nki (nki.isa, nki.language, nki.isa.constants.oob_mode): every kernel in kiln/kernels/;
  - neuronx-cc: the compiler LNL and the compile farm (kiln/compile_farm.py) run as a CLI;
  - the `aws` CLI: kiln/compile_cache.py reaches S3 compile caches and farm queues through it;
  - every kiln module imports.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import pkgutil
import shutil
import subprocess
import sys

FAILED: list[str] = []


def check(name: str, fn):
    try:
        out = fn()
    except BaseException as e:  # noqa: BLE001 - report every failure, then exit 1
        FAILED.append(name)
        print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
        return None
    print(f"ok   {name}" + (f": {out}" if out not in (None, "") else ""), flush=True)
    return out


def attr(module: str, *names: str):
    """`from module import names`: an attribute, or a submodule imported on demand."""
    m = importlib.import_module(module)
    for n in names:
        if not hasattr(m, n):
            importlib.import_module(f"{module}.{n}")
    return None


def cli(*argv: str) -> str:
    exe = shutil.which(argv[0])
    if exe is None:
        raise FileNotFoundError(f"{argv[0]} is not on PATH")
    p = subprocess.run([exe, *argv[1:]], capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} exited {p.returncode}: {(p.stderr or p.stdout).strip()[-300:]}")
    return " ".join((p.stdout or p.stderr).split())[:160]


def kiln_versions():
    import kiln

    want = os.environ.get("KILN_VERSION")
    meta = importlib.metadata.version("kiln")
    if meta != kiln.__version__:
        raise RuntimeError(f"installed distribution {meta} != kiln.__version__ {kiln.__version__}")
    if want and want != kiln.__version__:
        raise RuntimeError(f"KILN_VERSION={want} but kiln.__version__ is {kiln.__version__}")
    return f"kiln {kiln.__version__} from {os.path.dirname(kiln.__file__)}"


def kiln_modules():
    import kiln

    names = [m.name for m in pkgutil.walk_packages(kiln.__path__, "kiln.")]
    for n in names:
        importlib.import_module(n)
    return f"{len(names)} modules"


def backend():
    """LNL registers "neuron_libtorch" at import only when /dev/neuron* exists (libtorch_neuronx_lite/__init__.py
    _register_compile_backends, SDK 2.32), so without a device only the function it registers is checked."""
    import glob

    import torch._dynamo as dynamo

    import libtorch_neuronx_lite  # noqa: F401  registers the "neuron" device and, with a device, the backend
    from libtorch_neuronx_lite.compile.backend import compile as lnl_compile

    if not glob.glob("/dev/neuron*"):
        dynamo.lookup_backend("neuron_libtorch_graph_capture")
        return f"no /dev/neuron*: {lnl_compile.__module__}.compile present, registered when a device is"
    dynamo.lookup_backend("neuron_libtorch")
    from kiln.engine.model_runner import canonical_neuron_backend

    return f"neuron_libtorch registered, wrapped as {canonical_neuron_backend().__name__}"


def device_graph():
    """One small graph on a NeuronCore the way Kiln compiles its own (kiln/engine/engine.py and
    model_runner.py: the runtime env first, then LNL, device neuron:0, Kiln's backend wrapper and
    compiler arguments), against the CPU."""
    import torch

    from kiln import platform

    platform.configure_runtime_env()
    import libtorch_neuronx_lite  # noqa: F401
    from kiln.engine.model_runner import canonical_neuron_backend, neuronx_cc_args

    def f(x, w):
        return torch.nn.functional.silu(x @ w) * 2.0 + 1.0

    g = torch.Generator().manual_seed(0)
    x, w = torch.randn(64, 128, generator=g), torch.randn(128, 256, generator=g)
    want = f(x, w)
    dev = torch.device("neuron:0")
    fn = torch.compile(f, backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                       options={"compiler_args": neuronx_cc_args(torch.bfloat16)})
    got = fn(x.to(dev), w.to(dev)).cpu()
    err = (got - want).abs().max().item()
    if not err < 0.05 * want.abs().max().item():  # neuronx-cc auto-casts fp32 matmuls to bf16 on trn1
        raise RuntimeError(f"max |device - cpu| {err}")
    return f"max |device - cpu| {err:.3g} (output max {want.abs().max().item():.3g}), {platform.describe().get('target')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", action="store_true", help="also run a graph on a NeuronCore")
    args = ap.parse_args()

    check("python", lambda: sys.version.split()[0])
    check("kiln version", kiln_versions)
    for dist in ("torch", "libtorch-neuronx-lite", "nki", "neuronx-cc", "transformers", "tokenizers",
                 "safetensors", "huggingface_hub", "fastapi", "uvicorn", "pydantic", "numpy", "xgrammar"):
        check(f"dist {dist}", lambda d=dist: importlib.metadata.version(d))
    check("nki", lambda: attr("nki", "jit", "simulate"))
    check("nki.isa", lambda: attr("nki.isa", "nc_matmul", "dma_copy", "activation", "memset"))
    check("nki.language", lambda: attr("nki.language", "ndarray", "sbuf", "psum", "shared_hbm"))
    check("nki.isa.constants.oob_mode", lambda: attr("nki.isa.constants", "oob_mode"))
    check("libtorch_neuronx_lite.nki.nki_hop.wrap_nki", lambda: attr("libtorch_neuronx_lite.nki.nki_hop", "wrap_nki"))
    check("libtorch_neuronx_lite.compile.cache", lambda: attr("libtorch_neuronx_lite.compile", "cache"))
    check("libtorch_neuronx_lite.compile.backend",
          lambda: attr("libtorch_neuronx_lite.compile.backend", "_parse_compiler_args",
                       "_apply_platform_compiler_args", "preprocess_and_validate_inputs"))
    check("libtorch_neuronx_lite.compile.capture_backend",
          lambda: attr("libtorch_neuronx_lite.compile.capture_backend", "run_fx_to_hlo_pipeline",
                       "setup_workdir_common"))
    check("libtorch_neuronx_lite.compile.hlo", lambda: attr("libtorch_neuronx_lite.compile.hlo", "load_hlo_module"))
    check("libtorch_neuronx_lite.compile.platform",
          lambda: attr("libtorch_neuronx_lite.compile.platform", "resolve_target", "get_platform_target"))
    check("libtorch_neuronx_lite.compile.schema", lambda: attr("libtorch_neuronx_lite.compile.schema", "create_metadata"))
    check("libtorch_neuronx_lite.pyhlo.xla_data_pb2", lambda: attr("libtorch_neuronx_lite.pyhlo", "xla_data_pb2"))
    check("libtorch_neuronx_lite.overrides",
          lambda: attr("libtorch_neuronx_lite.overrides", "neuron_collectives", "xla_collectives"))
    check("torch.compile backend neuron_libtorch", backend)
    check("neuronx-cc --version", lambda: cli("neuronx-cc", "--version"))
    check("aws --version", lambda: cli("aws", "--version"))
    check("kiln modules", kiln_modules)
    if args.device:
        check("neuron-ls", lambda: cli("neuron-ls"))
        check("graph on a NeuronCore", device_graph)
    if FAILED:
        print(f"{len(FAILED)} check(s) failed: {', '.join(FAILED)}", flush=True)
        return 1
    print("all checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
