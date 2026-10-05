"""Device tensor bytes of one rank's engine, without a NeuronCore: the shard is built on the meta device
exactly as the compile-farm capture builds it (tools/compile_farm.py _capture_rank up to build_shard,
no warmup), and every unique tensor the model and runner hold is summed (parameters, buffers, the KV
and state bound to each layer, scratch, runner pools). Run it with the same KILN_* environment as the
device run (the MoE kernels change how the experts are packed).

    KILN_MOE_KERNEL=nki ... python tools/tensor_bytes.py <shape dir> <target> <rank> -- <serve_sweep args>

docs/neuron-notes.md "Prefill group size P on the new layout" adds the farm's per-graph numbers
(rings, instructions, spill space) to this for an HBM estimate per configuration.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def tensor_bytes(model, runner) -> dict[str, int]:
    import torch

    seen: set[int] = set()
    by: dict[str, int] = {}

    def add(t, what):
        if isinstance(t, torch.Tensor) and id(t) not in seen:
            seen.add(id(t))
            by[what] = by.get(what, 0) + t.numel() * t.element_size()

    def walk(obj, what, depth=0):
        if isinstance(obj, torch.Tensor):
            add(obj, what)
        elif isinstance(obj, (list, tuple)) and depth < 3:
            for x in obj:
                walk(x, what, depth + 1)
        elif isinstance(obj, dict) and depth < 3:
            for x in obj.values():
                walk(x, what, depth + 1)

    for n, p in model.named_parameters():
        expert = any(s in n for s in (".w_blob", "w_gu", "w_down", "moe_prefill"))
        add(p, "experts" if expert else "weights")
    for _, b in model.named_buffers():
        add(b, "buffers")
    for layer in [*model.layers, *([model.mtp] if getattr(model, "mtp", None) is not None else [])]:
        for n, v in vars(layer).items():
            walk(v, "kv_cache" if n in ("k_cache", "v_cache") else "layer_state")
    for n, v in vars(runner).items():
        if n != "model":
            walk(v, "runner")
    return by


def main() -> None:
    shape_dir, target, rank = sys.argv[1], sys.argv[2], int(sys.argv[3])
    argv = sys.argv[sys.argv.index("--") + 1:]
    from kiln import capture

    capture.configure_env(target)
    import compile_farm as cf
    import torch

    mod = cf._tool("serve_sweep")
    cfg = mod.engine_config(mod.build_parser().parse_args(argv))
    torch.set_num_threads(2)
    if cfg.tp > 1:  # before LNL, as a device rank and the capture do
        capture.init_fake_world(rank, cfg.tp)
    from kiln import platform as kplatform

    kplatform.configure_runtime_env()
    import libtorch_neuronx_lite  # noqa: F401

    from kiln.config import ModelConfig
    from kiln.engine.engine import build_shard, pool_pages

    capture.install_header_checkpoint()
    mcfg = ModelConfig.from_pretrained(shape_dir)
    num_pages = pool_pages(cfg, mcfg)
    model, runner = build_shard(dataclasses.replace(cfg, device="meta"), shape_dir, num_pages, rank)
    by = tensor_bytes(model, runner)
    print(json.dumps({"num_pages": num_pages, **{k: round(v / 1e9, 3) for k, v in sorted(by.items())},
                      "total_gb": round(sum(by.values()) / 1e9, 3)}), flush=True)


if __name__ == "__main__":
    main()
