"""Isolate device-vs-CPU divergence one mechanism at a time.

    python tools/debug_device.py [probe ...]

Probes:
  write_then_read  in-place index_put_ into a cache, then a gather from it, in ONE graph
  gather2d         cache[2-D index] in a graph, no write
  logits           the whole model without a KV cache (forward_logits), device vs CPU
  verify           the speculative-verify sampler alone at a vocabulary and row count
  queue            how many back-to-back launches fit the execution queue, by graph input count
  mxfp4            packed MXFP4 experts decoded in-graph (models/quant.py), gathered and
                   multiplied as the MoE gather path does, device vs CPU (bit-exact decode)
"""

from __future__ import annotations

import os
import sys

import torch

from kiln import platform

platform.configure_runtime_env()  # NEURON_LOGICAL_NC_CONFIG on trn2 / trn3, before the runtime
import libtorch_neuronx_lite  # noqa: E402,F401

DEV = torch.device("neuron:0")
OPTS = dict(backend="neuron_libtorch", fullgraph=True, dynamic=False)


def write_then_read():
    cache = torch.zeros(64, 2, 4, dtype=torch.bfloat16)
    dcache = cache.to(DEV)
    slots = torch.tensor([3, 17, 40], dtype=torch.long)
    new = torch.randn(3, 2, 4, dtype=torch.bfloat16)
    idx = torch.tensor([[3, 4, 17], [40, 41, 3]], dtype=torch.long)

    def f(c, s, x, i):
        c.index_put_((s,), x)
        return c[i] * 1.0

    ref = f(cache.clone(), slots, new, idx)
    g = torch.compile(f, **OPTS)
    out = g(dcache, slots.to(DEV), new.to(DEV), idx.to(DEV)).cpu()
    after = dcache.cpu()
    print("write_then_read: read sees write:", torch.equal(out, ref),
          "| cache updated in place:", torch.equal(after[slots], new),
          "| max diff", float((out.float() - ref.float()).abs().max()))


def gather2d():
    cache = torch.randn(64, 2, 4, dtype=torch.bfloat16)
    idx = torch.randint(0, 64, (3, 8), dtype=torch.long)

    def f(c, i):
        return c[i] * 1.0

    out = torch.compile(f, **OPTS)(cache.to(DEV), idx.to(DEV)).cpu()
    print("gather2d equal:", torch.equal(out, f(cache, idx)))


def logits():
    from kiln.config import ModelConfig
    from kiln.models.loader import load_model, resolve_model_path
    from transformers import AutoTokenizer

    path = resolve_model_path("Qwen/Qwen3-0.6B")
    cfg = ModelConfig.from_pretrained(path)
    ids = AutoTokenizer.from_pretrained(path)("The capital of France is")["input_ids"]
    cpu = load_model(path, cfg, torch.bfloat16, torch.device("cpu"), 256)
    with torch.no_grad():
        ref = cpu.forward_logits(torch.tensor(ids))
    dev = load_model(path, cfg, torch.bfloat16, DEV, 256)
    g = torch.compile(dev.forward_logits, **OPTS)
    with torch.no_grad():
        out = g(torch.tensor(ids).to(DEV)).cpu()
    agree = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
    print(f"logits: max diff {float((out - ref).abs().max()):.3f}, argmax agreement {agree:.2f}, "
          f"next-token cpu={int(ref[-1].argmax())} dev={int(out[-1].argmax())}")


def mxfp4():
    from kiln.models.quant import dequant_mxfp4

    g = torch.Generator().manual_seed(0)
    packed = torch.randint(0, 256, (16, 128, 256), dtype=torch.uint8, generator=g)  # E, N, K/2
    scale = torch.exp2(torch.randint(-8, 3, (16, 128, 16), generator=g).float()).to(torch.bfloat16)
    idx = torch.tensor([3, 3, 7, 15, 0, 9], dtype=torch.long)
    x = torch.randn(6, 512, 1, dtype=torch.bfloat16, generator=g)

    def decode(p, s):
        return dequant_mxfp4(p, s, torch.bfloat16) * 1.0

    def moe(p, s, i, v):
        return torch.bmm(dequant_mxfp4(p[i], s[i], torch.bfloat16), v)

    ref = decode(packed, scale)
    out = torch.compile(decode, **OPTS)(packed.to(DEV), scale.to(DEV)).cpu()
    print("mxfp4 decode bit-exact:", torch.equal(out, ref), "| max diff", float((out.float() - ref.float()).abs().max()))
    ref = moe(packed, scale, idx, x).float()
    out = torch.compile(moe, **OPTS)(packed.to(DEV), scale.to(DEV), idx.to(DEV), x.to(DEV)).cpu().float()
    rel = float((out - ref).abs().max() / ref.abs().max())
    print(f"mxfp4 gathered bmm: rel max diff {rel:.5f} (bf16 accumulation order)")


def verify():
    """verify_sample over [R, V] fp32 logits for several R (MiMo-V2.6: V=152576, R=B*Q=16 failed to
    compile with NCC_IBIR243 inside the tp=32 post graph)."""
    from kiln.engine.sampler import sample, verify_sample

    V = int(os.environ.get("KILN_V", 152576))
    for R in [int(x) for x in os.environ.get("KILN_R", "4,16").split(",")]:
        for name, fn in (("sample", lambda l, t, p, k, m, n, u, d: sample(l, t, p, k, m, n)),
                         ("verify", verify_sample)):
            g = torch.Generator().manual_seed(0)
            args = [torch.randn(R, V, generator=g), torch.zeros(R), torch.ones(R), torch.zeros(R, dtype=torch.int64),
                    torch.zeros(R), torch.full((R, 64), 0.5), torch.full((R,), 0.5), torch.zeros(R, dtype=torch.int64)]
            try:
                out = torch.compile(fn, **OPTS)(*[a.to(DEV) for a in args]).cpu()
                ref = fn(*args)
                print(f"  {name} R={R} V={V}: ok, max diff {float((out - ref).abs().max()):.4f}")
            except Exception as e:  # report and continue
                print(f"  {name} R={R} V={V}: FAILED {type(e).__name__} {str(e)[:120]}")


def verify_head():
    """lm_head shard + one-hot vocab placement (tp=32 shapes, all-reduce omitted) + verify_sample,
    in variants, to find which form neuronx-cc compiles."""
    from kiln.engine.sampler import verify_sample

    R, H, tp, Vr, V = 16, 4096, 32, 4768, 152576
    g = torch.Generator().manual_seed(0)
    h = torch.randn(R, H, generator=g).to(torch.bfloat16)
    w = (torch.randn(Vr, H, generator=g) * 0.02).to(torch.bfloat16)
    start = torch.tensor(5 * Vr)
    rest = [torch.zeros(R), torch.ones(R), torch.zeros(R, dtype=torch.int64), torch.zeros(R),
            torch.full((R, 64), 0.5), torch.full((R,), 0.5), torch.zeros(R, dtype=torch.int64)]

    def place_onehot(local, start):
        mine = (torch.arange(tp, device=local.device) == start // Vr).to(local.dtype)
        return (local.unsqueeze(1) * mine.view(1, -1, 1)).reshape(R, tp * Vr)[:, :V].float()

    def place_pad(local, start):  # zero-pad on both sides with a dynamic offset via where over a column index
        col = torch.arange(tp * Vr, device=local.device) - start
        idx = col.clamp(0, Vr - 1)
        g = local.index_select(1, idx)
        return torch.where(((col >= 0) & (col < Vr)).unsqueeze(0), g, torch.zeros_like(g))[:, :V].float()

    def place_onehot_f32(local, start):
        mine = (torch.arange(tp, device=local.device) == start // Vr).float()
        return (local.float().unsqueeze(1) * mine.view(1, -1, 1)).reshape(R, tp * Vr)[:, :V]

    for name, place in (("onehot", place_onehot), ("onehot_f32", place_onehot_f32), ("index_pad", place_pad)):
        def f(h, w, start, *r, _place=place):
            return verify_sample(_place(torch.nn.functional.linear(h, w), start), *r)
        try:
            out = torch.compile(f, **OPTS)(h.to(DEV), w.to(DEV), start.to(DEV), *[a.to(DEV) for a in rest]).cpu()
            print(f"  {name}: ok, max diff vs cpu {float((out - f(h, w, start, *rest)).abs().max()):.4f}")
        except Exception as e:  # report and continue
            print(f"  {name}: FAILED {type(e).__name__} {str(e)[:150]}")


def queue():
    import os

    print("NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS =", os.environ.get("NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS"))
    for n_in, dim in ((1, 256), (4, 256), (16, 256), (16, 4096)):  # the last one is slow on device
        ws = [torch.randn(dim, dim, dtype=torch.bfloat16).to(DEV) * dim ** -0.5 for _ in range(n_in)]

        def f(x, *w):
            for t in w:
                x = x @ t
            return x

        g = torch.compile(f, **OPTS)
        x = torch.randn(dim, dim, dtype=torch.bfloat16).to(DEV)
        g(x, *ws).cpu()  # compile
        launched = 0
        try:
            y = x
            for _ in range(200):
                y = g(y, *ws)
                launched += 1
            y.cpu()
            print(f"  inputs {n_in + 1:>2} dim {dim}: 200 chained launches ok")
        except RuntimeError as e:
            print(f"  inputs {n_in + 1:>2} dim {dim}: failed after {launched} launches: {str(e)[:80]}")


def split_variants():
    """Which ways of cutting a tensor along its last dim survive LNL compilation."""
    x = torch.randn(5, 4096, dtype=torch.bfloat16)
    variants = {
        "split_unequal": lambda t: torch.split(t, (2048, 1024, 1024), dim=-1),
        "split_equal": lambda t: torch.split(t, 2048, dim=-1),
        "chunk2": lambda t: t.chunk(2, dim=-1),
        "tensor_split_idx": lambda t: torch.tensor_split(t, (2048, 3072), dim=-1),
        "slice": lambda t: (t[..., :2048], t[..., 2048:3072], t[..., 3072:]),
        "narrow": lambda t: (t.narrow(-1, 0, 2048), t.narrow(-1, 2048, 1024), t.narrow(-1, 3072, 1024)),
    }
    for name, fn in variants.items():
        ref = [a * 1.0 for a in fn(x)]
        got = torch.compile(lambda t: tuple(a * 1.0 for a in fn(t)), **OPTS)(x.to(DEV))
        ok = all(torch.equal(a, b.cpu()) for a, b in zip(ref, got))
        print(f"  {name:<17} {'ok' if ok else 'WRONG'}")


def layer0():
    """Every intermediate of layer 0 from one compiled graph, each compared with CPU."""
    import torch.nn.functional as F

    from kiln.config import ModelConfig
    from kiln.models.loader import load_model, resolve_model_path
    from kiln.models.qwen3 import rms_norm, rotate_half
    from transformers import AutoTokenizer

    path = resolve_model_path("Qwen/Qwen3-0.6B")
    cfg = ModelConfig.from_pretrained(path)
    ids = torch.tensor(AutoTokenizer.from_pretrained(path)("The capital of France is")["input_ids"])

    def stages(m, ids):
        L = m.layers[0]
        T = ids.shape[0]
        pos = torch.arange(T, device=ids.device)
        out = {}
        h = F.embedding(ids, m.embed); out["embed"] = h
        x = rms_norm(h, L.in_norm, cfg.rms_norm_eps); out["in_norm"] = x
        qkv = F.linear(x, L.qkv); out["qkv"] = qkv
        q, k, v = torch.split(qkv, L.split, dim=-1)
        out["q_split"] = q * 1.0; out["v_split"] = v * 1.0
        q = rms_norm(q.view(T, cfg.num_heads, cfg.head_dim), L.q_norm, cfg.rms_norm_eps); out["q_norm"] = q
        cos = m.rope_cos[pos].unsqueeze(1); out["cos"] = cos * 1.0
        out["rot"] = rotate_half(q)
        q = q * cos + rotate_half(q) * m.rope_sin[pos].unsqueeze(1); out["q_rope"] = q
        k = rms_norm(k.view(T, cfg.num_kv_heads, cfg.head_dim), L.k_norm, cfg.rms_norm_eps)
        k = k * cos + rotate_half(k) * m.rope_sin[pos].unsqueeze(1)
        v = v.view(T, cfg.num_kv_heads, cfg.head_dim)
        G = cfg.num_heads // cfg.num_kv_heads
        qg = q.view(T, cfg.num_kv_heads, G, cfg.head_dim)
        s = torch.einsum("chgd,lhd->hgcl", qg, k).float() * m.scale; out["scores"] = s
        causal = pos.unsqueeze(0) <= pos.unsqueeze(1)
        s = s + torch.where(causal, 0.0, -1e30).view(1, 1, T, T)
        pr = torch.softmax(s, dim=-1).to(m.dtype); out["probs"] = pr
        o = torch.einsum("hgcl,lhd->chgd", pr, v).reshape(T, -1); out["attn"] = o
        h = h + F.linear(o, L.o); out["resid"] = h
        out["mlp"] = m._mlp(L, h)
        return tuple(out.values()), list(out)

    cpu = load_model(path, cfg, torch.bfloat16, torch.device("cpu"), 256)
    with torch.no_grad():
        ref, names = stages(cpu, ids)
    dev = load_model(path, cfg, torch.bfloat16, DEV, 256)
    g = torch.compile(lambda i: stages(dev, i)[0], **OPTS)
    with torch.no_grad():
        got = [t.cpu() for t in g(ids.to(DEV))]
    for n, a, b in zip(names, ref, got):
        d = float((a.float() - b.float()).abs().max())
        rel = d / max(float(a.float().abs().max()), 1e-6)
        print(f"  {n:<9} max_abs {d:10.4f}  rel {rel:.4f}  shape {tuple(a.shape)}")


if __name__ == "__main__":
    print("platform", platform.check_runtime())
    names = sys.argv[1:] or ["write_then_read", "gather2d", "logits"]
    for n in names:
        try:
            globals()[n]()
        except Exception as e:  # report and continue: one broken probe must not hide the rest
            print(n, "FAILED", type(e).__name__, str(e)[:400])
