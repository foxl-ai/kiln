"""gpt-oss / Hy3 pieces on ONE NeuronCore against the CPU, to localize a device divergence.

    python tools/probe_gqa_moe.py act                    # swiglu_oai alone, bf16
    python tools/probe_gqa_moe.py intdiv                 # int64 // constant in a graph (vocab shard index)
    python tools/probe_gqa_moe.py small                  # a random small gpt-oss, forward_logits
    python tools/probe_gqa_moe.py real <path> <layers>   # the first layers of a real checkpoint, tp=1

Each prints the max difference and the argmax agreement of the device against the CPU in the same
dtype (bf16) and against fp32.
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

import libtorch_neuronx_lite  # noqa: F401

DEV = torch.device("neuron:0")
OPTS = dict(backend="neuron_libtorch", fullgraph=True, dynamic=False)


def act():
    from kiln.models.decoder import swiglu_oai

    g = torch.Generator().manual_seed(0)
    gate, up = (torch.randn(64, 96, generator=g) * 10).bfloat16(), (torch.randn(64, 96, generator=g) * 10).bfloat16()
    ref = swiglu_oai(gate, up, 7.0)
    out = torch.compile(lambda a, b: swiglu_oai(a, b, 7.0), **OPTS)(gate.to(DEV), up.to(DEV)).cpu()
    print(f"act: max diff {float((out.float() - ref.float()).abs().max()):.4f} (|ref| max {float(ref.abs().max()):.1f})")
    for name, f in [("clamp max", lambda a: a.clamp(max=7.0)), ("clamp both", lambda a: a.clamp(min=-7.0, max=7.0)),
                    ("sigmoid", lambda a: torch.sigmoid(a * 1.702))]:
        o = torch.compile(f, **OPTS)(gate.to(DEV)).cpu()
        print(f"  {name}: max diff {float((o.float() - f(gate).float()).abs().max()):.4f}")


def intdiv():
    """int64 floor division by a constant in a graph, as DecoderForCausalLM._head once found its
    vocabulary shard (vocab_start // rows): exact on the CPU, and in fp32 a reciprocal multiply
    would floor r * 6284 / 6284 to r - 1 for 21 of 32 ranks (0 for 4768)."""
    pos = torch.arange(1 << 20, dtype=torch.int64)  # positions up to 1M, by a page size of 32
    out = torch.compile(lambda p: p // 32, **OPTS)(pos.to(DEV)).cpu()
    print(f"intdiv: positions // 32 exact up to 2^20: {torch.equal(out, pos // 32)}")
    for rows in (6284, 4768):
        starts = torch.arange(32, dtype=torch.int64) * rows
        out = torch.compile(lambda s: s // rows, **OPTS)(starts.to(DEV)).cpu()
        wrong = (out != torch.arange(32)).nonzero().flatten().tolist()
        print(f"intdiv: start // {rows} wrong for ranks {wrong}")
        out = torch.compile(lambda s: torch.arange(32, device=s.device) * rows == s.unsqueeze(-1),
                            **OPTS)(starts.to(DEV)).cpu()
        print(f"  arange * {rows} == start one-hot exact: {torch.equal(out, torch.eye(32, dtype=torch.bool))}")


def compare(path: str, cfg, ids: torch.Tensor, **kw):
    from kiln.models.loader import load_model

    with torch.no_grad():
        ref32 = load_model(path, cfg, torch.float32, torch.device("cpu"), 256, **kw).forward_logits(ids)
        ref16 = load_model(path, cfg, torch.bfloat16, torch.device("cpu"), 256, **kw).forward_logits(ids)
        dev = load_model(path, cfg, torch.bfloat16, DEV, 256, **kw)
        from kiln.engine.model_runner import neuronx_cc_args

        opts = dict(OPTS, options={"compiler_args": neuronx_cc_args(torch.bfloat16, fp8_weights=True)})
        out = torch.compile(dev.forward_logits, **opts)(ids.to(DEV)).cpu()
    for name, ref in (("cpu bf16", ref16), ("cpu fp32", ref32)):
        agree = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
        print(f"  device vs {name}: max diff {float((out - ref).abs().max()):.3f} of {float(ref.abs().max()):.2f}, "
              f"argmax agreement {agree:.2f}")
    agree = (ref16.argmax(-1) == ref32.argmax(-1)).float().mean().item()
    print(f"  cpu bf16 vs fp32: max diff {float((ref16 - ref32).abs().max()):.3f}, argmax agreement {agree:.2f}")


def small():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tests.test_gpt_oss import SMALL, build

    from kiln.config import ModelConfig

    d = tempfile.mkdtemp()
    build(d, **SMALL)
    ids = torch.randint(0, 384, (32,), generator=torch.Generator().manual_seed(3))
    print("small gpt-oss, FP8 experts:")
    compare(d, ModelConfig.from_pretrained(d), ids, keep_fp8=True, fp8_max=240.0)


def real(path: str, layers: int):
    from kiln.config import ModelConfig

    cfg = ModelConfig.from_pretrained(path).truncated(layers)
    ids = torch.tensor([200006, 17360, 200008, 976, 9029, 328, 10128, 382, 12650, 13, 623, 9029, 328, 23154, 382])
    print(f"{path}: first {layers} layers, FP8 experts, tp=1:")
    compare(path, cfg, ids, keep_fp8=cfg.quant_block is not None or cfg.quant_expert_block is not None,
            fp8_max=240.0)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "real":
        real(sys.argv[2], int(sys.argv[3]))
    else:
        globals()[what]()
