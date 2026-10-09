"""One linear-attention layer (models/linear_attn.py) on a NeuronCore against the same layer in
fp32 on the host: decode and chunked-prefill graphs, numerics and steady-state time per call.

    python tools/profile_linear_attn.py --config Qwen/Qwen3.5-0.8B --batch 8 --chunk 128
    python tools/profile_linear_attn.py --kind kda --heads 32 --dim 128 --hidden 2304

Random weights at the given shapes (a GDN layer from the model's config.json, or a KDA layer
from --heads / --dim / --hidden). The state pool lives on the device; the prefill graph runs two
chunks (the second continues the first's state) and the decode graph a few steps after them,
and every output is compared with the host's.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def build(args, device, dtype):
    from safetensors.torch import save_file

    from kiln.config import LinearSpec, ModelConfig
    from kiln.models.linear_attn import gdn_spec
    from kiln.models.loader import load_model

    if args.kind == "gdn":
        from kiln.models.loader import resolve_model_path

        with open(os.path.join(resolve_model_path(args.config), "config.json")) as f:
            c = json.load(f)
        t = c.get("text_config") or c
        spec, H, eps = gdn_spec(t), t["hidden_size"], t.get("rms_norm_eps", 1e-6)
    else:
        spec = LinearSpec("kda", args.heads, args.heads, args.dim, args.dim, 4, "sigmoid", args.dim,
                          args.lower_bound)
        H, eps = args.hidden, 1e-5
    g = torch.Generator().manual_seed(0)
    r = lambda *s, sc=0.02: (torch.randn(*s, generator=g) * sc)  # noqa: E731
    nk, nv, dk, dv = spec.num_k_heads, spec.num_v_heads, spec.head_k_dim, spec.head_v_dim
    p = "model.layers.0."
    if spec.kind == "gdn":
        cd = 2 * nk * dk + nv * dv
        t = {"linear_attn.in_proj_qkv.weight": r(cd, H), "linear_attn.conv1d.weight": r(cd, 1, 4, sc=0.3),
             "linear_attn.in_proj_z.weight": r(nv * dv, H), "linear_attn.in_proj_b.weight": r(nv, H),
             "linear_attn.in_proj_a.weight": r(nv, H), "linear_attn.dt_bias": r(nv, sc=1.0),
             "linear_attn.A_log": torch.empty(nv).uniform_(-4, 1, generator=g), "linear_attn.norm.weight": 1 + r(dv, sc=0.1),
             "linear_attn.out_proj.weight": r(H, nv * dv)}
    else:
        D = nk * dk
        t = {f"self_attn.{x}_proj.weight": r(D, H) for x in "qkv"}
        t.update({f"self_attn.{x}_conv1d.weight": r(D, 1, 4, sc=0.3) for x in "qkv"})
        t.update({"self_attn.f_a_proj.weight": r(dk, H), "self_attn.f_b_proj.weight": r(D, dk, sc=0.1),
                  "self_attn.dt_bias": r(D, sc=1.0), "self_attn.A_log": torch.empty(nk).uniform_(-4, 1, generator=g),
                  "self_attn.b_proj.weight": r(nk, H), "self_attn.g_a_proj.weight": r(dk, H),
                  "self_attn.g_b_proj.weight": r(D, dk, sc=0.1), "self_attn.o_norm.weight": 1 + r(dk, sc=0.1),
                  "self_attn.o_proj.weight": r(H, D)})
    t = {p + k: v for k, v in t.items()}
    for n in ("input_layernorm", "post_attention_layernorm"):
        t[p + n + ".weight"] = torch.ones(H)
    for n, shp in (("gate_proj", (8, H)), ("up_proj", (8, H)), ("down_proj", (H, 8))):
        t[f"{p}mlp.{n}.weight"] = torch.zeros(shp)
    t["model.norm.weight"] = torch.ones(H)
    t["model.embed_tokens.weight"] = torch.zeros(8, H)
    d = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"kiln-la-{spec.kind}")
    os.makedirs(d, exist_ok=True)
    save_file(t, os.path.join(d, "model.safetensors"))
    cfg = ModelConfig(architecture="KimiLinearForCausalLM", vocab_size=8, hidden_size=H, intermediate_size=8,
                      num_layers=1, num_heads=1, num_kv_heads=1, head_dim=8, rms_norm_eps=eps, rope_theta=1e4,
                      max_position_embeddings=64, tie_word_embeddings=True, eos_token_ids=(), attn_layers=(spec,))
    return load_model(d, cfg, dtype, device, 64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="gdn", choices=["gdn", "kda"])
    ap.add_argument("--config", default="Qwen/Qwen3.5-0.8B", help="GDN shapes from this model's config.json")
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=2304)
    ap.add_argument("--lower-bound", type=float, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--parts", action="store_true", help="also time the decode state gather / scatter alone")
    args = ap.parse_args()
    from kiln.engine.state_pool import StatePool
    from kiln.models import linear_attn

    if args.device == "neuron":  # the engine's runtime env, LNC and compiler arguments (trn1 and trn2), as profile_layer
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import profile_layer as pl

        pl.setup_device()
        dev = pl.DEV
        compile_ = lambda f: torch.compile(f, **pl.OPTS)  # noqa: E731
    else:
        dev, compile_ = torch.device("cpu"), (lambda f: f)
    host = build(args, torch.device("cpu"), torch.float32)
    model = build(args, dev, torch.bfloat16)
    B, C = args.batch, args.chunk
    for m, d in ((host, torch.device("cpu")), (model, dev)):
        StatePool(m, B, d, m.dtype)
        m.page_size = 4
    H = host.cfg.hidden_size

    def step(m):
        layer = m.layers[0]

        def f(h, pos, slots, state_slot):
            return linear_attn.mixer(m, layer, h, pos, slots, state_slot)
        return f

    f_host, f_dev = step(host), compile_(step(model))
    g = torch.Generator().manual_seed(1)
    n1, n2 = C, (3 * C) // 4 - 1  # a full chunk, then a padded one that continues its state
    h = torch.randn(n1 + n2 + 4, H, generator=g)
    outs = {"host": [], "dev": []}
    for name, f, m in (("host", f_host, host), ("dev", f_dev, model)):
        d = m.layers[0].in_norm.device
        start = 0
        for n in (n1, n2):
            x = torch.zeros(C, H)
            x[:n] = h[start : start + n]
            pos = torch.zeros(C, dtype=torch.long)
            pos[:n] = torch.arange(start, start + n)
            slots = torch.zeros(C, dtype=torch.long)
            slots[:n] = pos[:n] + 4
            t = time.perf_counter()
            o = f(x.to(m.dtype).to(d), pos.to(d), slots.to(d), torch.tensor([1]).to(d)).cpu().float()
            if name == "dev" and start == 0:
                print(f"prefill C={C}: first call (compile) {time.perf_counter() - t:.1f}s", flush=True)
            outs[name].append(o[:n] - h[start : start + n])
            start += n
        for t_ in range(start, start + 4):  # decode: batch row 0 continues state row 1, the rest is padding
            x = torch.zeros(B, H)
            x[0] = h[t_]
            pos = torch.zeros(B, dtype=torch.long)
            pos[0] = t_
            rows = torch.zeros(B, dtype=torch.long)
            rows[0] = 1
            t = time.perf_counter()
            o = f(x.to(m.dtype).to(d), pos.to(d), (pos + 4).to(d), rows.to(d)).cpu().float()
            if name == "dev" and t_ == start:
                print(f"decode B={B}: first call (compile) {time.perf_counter() - t:.1f}s", flush=True)
            outs[name].append(o[0:1] - h[t_ : t_ + 1])
    a, b = torch.cat(outs["host"]), torch.cat(outs["dev"])
    err = ((a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-6))
    print(f"relative error per token vs fp32 host: max {err.max():.4f} mean {err.mean():.4f} "
          f"(prefill rows {n1 + n2}, decode rows 4)")
    top = torch.topk(err, min(6, err.numel()))
    print("  worst rows (index: error; prefill rows first, then decode):",
          ", ".join(f"{int(i)}: {float(e):.3f}" for e, i in zip(top.values, top.indices)))
    # Steady state on the device.
    d = dev
    xs = torch.randn(B, H).to(torch.bfloat16).to(d)
    pos, rows = torch.full((B,), 5, dtype=torch.long).to(d), torch.arange(B).to(d)
    for label, args_ in (("decode", (xs, pos, pos + 4, rows)),
                         ("prefill", (torch.randn(C, H).to(torch.bfloat16).to(d), torch.arange(C).to(d),
                                      (torch.arange(C) + 4).to(d), torch.tensor([1]).to(d)))):
        from kiln.engine.model_runner import IN_FLIGHT

        f_dev(*args_).cpu()
        t = time.perf_counter()
        outs = []
        for _ in range(args.iters):  # chained, at most IN_FLIGHT queued (the runtime's queue fills)
            outs.append(f_dev(*args_))
            if len(outs) > IN_FLIGHT:
                outs[-1 - IN_FLIGHT].cpu()
        outs[-1].cpu()
        chained = (time.perf_counter() - t) / args.iters
        t = time.perf_counter()
        for _ in range(5):
            f_dev(*args_).cpu()
        print(f"{label}: {chained * 1e3:.3f} ms per call chained ({args.iters}), "
              f"{(time.perf_counter() - t) / 5 * 1e3:.3f} ms with a read-back each")
    if args.parts:
        parts(model, compile_, dev, B, args.iters)
    print("RESULT", json.dumps({"max_rel_err": float(err.max()), "mean_rel_err": float(err.mean())}))


def timed(label, f, args_, iters):
    from kiln.engine.model_runner import IN_FLIGHT

    f(*args_).cpu()
    t = time.perf_counter()
    outs = []
    for _ in range(iters):
        outs.append(f(*args_))
        if len(outs) > IN_FLIGHT:
            outs[-1 - IN_FLIGHT].cpu()
    outs[-1].cpu()
    print(f"  {label:<44} {(time.perf_counter() - t) / iters * 1e3:8.3f} ms per call chained", flush=True)


def parts(model, compile_, dev, B, iters):
    """Where a decode step's time goes: the recurrent-state rows read and written by slot
    (dynamic DMA) against the same math on a dense [B, ...] state."""
    from kiln.models import linear_attn

    layer = model.layers[0]
    pool = layer.rec_state
    rows = torch.arange(1, B + 1).to(dev)
    S = torch.zeros(B, *pool.shape[1:]).to(dev)

    def gather(p, r):
        return p[r].sum(dim=(1, 2, 3))

    def scatter(p, r, x):
        p.index_put_((r,), x)
        return x[:, 0, 0, 0]

    def gather_scatter(p, r):
        x = p[r]
        p.index_put_((r,), x * 0.5)
        return x[:, 0, 0, 0]

    def rec_math(q, k, v, g, beta, S):
        o, S2 = linear_attn.recurrent_step(q, k, v, g, beta, S)
        return o + S2.sum(-2)

    H, Dk, Dv = pool.shape[1:]
    q = torch.randn(B, H, Dk).to(dev)
    k = torch.nn.functional.normalize(torch.randn(B, H, Dk), dim=-1).to(dev)
    v = torch.randn(B, H, Dv).to(dev)
    g = -torch.rand(B, H).to(dev)
    beta = torch.rand(B, H).to(dev)
    timed(f"state gather pool[rows] ({B} x {H}x{Dk}x{Dv} fp32)", compile_(gather), (pool, rows), iters)
    timed("state scatter index_put_", compile_(scatter), (pool, rows, S), iters)
    timed("state gather + scatter", compile_(gather_scatter), (pool, rows), iters)
    timed("recurrent step math on a dense state", compile_(rec_math), (q, k, v, g, beta, S), iters)
    x = torch.randn(B, model.cfg.hidden_size).to(torch.bfloat16).to(dev)

    def projections(x):
        y = torch.nn.functional.linear(x, model._w(layer, "in_qkv"))
        z = torch.nn.functional.linear(x, model._w(layer, "in_z"))
        return torch.nn.functional.linear(z, model._w(layer, "out")) + y[:, : x.shape[1]]

    if layer.spec.kind == "gdn":
        timed("in_qkv + in_z + out projections", compile_(projections), (x,), iters)

    # Prefill: one row of the pools, and the chunked scan on dense inputs.
    row = torch.tensor([1]).to(dev)
    timed("prefill: state gather pool[row] (1 row)", compile_(gather), (pool, row), iters)
    timed("prefill: state scatter index_put_ (1 row)", compile_(scatter), (pool, row, S[:1]), iters)
    conv = layer.conv_state

    def conv_rw(c, r, x, n):
        prev = c[r][0]
        xe = torch.cat([prev, x])
        last = n + torch.arange(c.shape[1], device=x.device)
        c.index_put_((r,), xe.index_select(0, last).unsqueeze(0))
        return xe[:, 0]

    Cn = 128
    xc = torch.randn(Cn, conv.shape[2]).to(conv.dtype).to(dev)
    timed("prefill: conv state read + tail write", compile_(conv_rw), (conv, row, xc, torch.tensor(Cn - 5).to(dev)), iters)
    qc = (torch.nn.functional.normalize(torch.randn(Cn, H, Dk), dim=-1) * Dk ** -0.5).to(dev)
    kc = torch.nn.functional.normalize(torch.randn(Cn, H, Dk), dim=-1).to(dev)
    vc = torch.randn(Cn, H, Dv).to(dev)
    gc = (-torch.rand(Cn, H) if layer.spec.kind == "gdn" else -torch.rand(Cn, H, Dk)).to(dev)
    bc = torch.rand(Cn, H).to(dev)
    S0 = torch.zeros(H, Dk, Dv).to(dev)

    def scan(q, k, v, g, b, S):
        o, S2 = linear_attn.chunk_scan(q, k, v, g, b, S)
        return o.sum(-1) + S2.sum()

    timed(f"prefill: chunk_scan alone (C={Cn})", compile_(scan),
          (qc, kc, vc, gc, bc, S0), iters)

    # Writing back ONE row whose value the graph computed (the chunk form's final state):
    # index_put_ of one row against two rows (the row plus the scratch row, as _write_rows does)
    # and against a masked rewrite of the whole pool.
    a, b2 = torch.randn(H, 64, Dk).to(dev), torch.randn(H, 64, Dv).to(dev)

    def new_row(p, r, a, b2):
        return p[r][0] + a.transpose(-1, -2) @ b2

    def one_row(p, r, a, b2):
        p.index_put_((r,), new_row(p, r, a, b2).unsqueeze(0))
        return a[0, 0]

    def two_rows(p, r, a, b2):
        n = new_row(p, r, a, b2)
        p.index_put_((torch.cat([r, torch.zeros_like(r)]),), torch.stack([n, n]))
        return a[0, 0]

    def whole_pool(p, r, a, b2):
        mask = (torch.arange(p.shape[0], device=r.device) == r).view(-1, 1, 1, 1)
        p.copy_(torch.where(mask, new_row(p, r, a, b2).unsqueeze(0), p))
        return a[0, 0]

    def gathered_only(p, r, a, b2):
        p.index_put_((r,), p[r] * 0.5)
        return a[0, 0]

    for label, f in (("write 1 row: index_put_ of computed row", one_row), ("write 1 row as 2 rows (+ scratch)", two_rows),
                     ("write 1 row by a masked whole-pool copy_", whole_pool),
                     ("write 1 row: index_put_ of the gathered row * 0.5", gathered_only)):
        timed(label, compile_(f), (pool, row, a, b2), iters)


if __name__ == "__main__":
    main()
