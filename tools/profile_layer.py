"""Device time of ONE tensor-parallel rank's decoder layers, whole and piece by piece, on one
NeuronCore, through the real Kiln code (DecoderForCausalLM.group_fn, _layer, _attention,
_attend, _qkv, _store, _load, _softmax, _mlp, _moe, _route, _experts, model_runner._piecewise).

    python tools/profile_layer.py [--model XiaomiMiMo/MiMo-V2.6-Flash-RL] [--tp 32]
        [--batch 4] [--pages 16] [--page-size 32] [--what parts layers groups step]
    python tools/profile_layer.py --prefill 32 --what layers    # one sequence x 32 tokens
    python tools/profile_layer.py --ranks 2 --what layers step --layers 48 --legacy-groups 6
    python tools/profile_layer.py --allreduce [--batch 4] [--ranks 32]   # in-graph all-reduce
    KILN_MOE_KERNEL=nki python tools/profile_layer.py --ranks 2 --what layers step --layers 48 \
        --step-groups 1 --moe-groups 2 6 12        # the NKI MoE kernel, MoE layers grouped

The model is built from config.json alone at rank 0's shapes for --tp, with random weights in
exactly the layout keep_fp8 loads (FP8 + per-row block scales, bf16 where the checkpoint
ignores quantization, FP8 experts from MXFP4 with bf16 block-32 scales), and then told it is
alone (tp_size = 1), so every all-reduce is the identity and what is timed is one rank's
compute. With --ranks 2 (the two NeuronCores of a trn1.2xlarge) two processes hold the same
rank-0 shapes and every all-reduce is real, over a group of 2: graph contents then match a
tp=32 rank's (MiMo-V2.6-Flash: 56 ms per 6-layer graph here against 59 ms on trn1.32xlarge).
Layer tensors are graph INPUTS, as in the piecewise runner, not constants. Each timing is the
p50 of synchronous calls (launch + run + reading a small output back); the "null graph" row is
that fixed overhead. Group rows also report the chained cost per call (launches back to back,
waiting IN_FLIGHT launches behind, as the runner does). `step` times the runner's whole
piecewise decode (prep graph, layer graphs, post graph; needs --ranks 2 for the vocab-parallel
head), with --legacy-groups also under the grouping before MoE layers got a graph each.
KILN_MOE_KERNEL=nki builds the MoE layers with their experts packed for kiln/kernels/moe_dedupe.py
(KILN_MOE_KERNEL=nki-pair: for kiln/kernels/moe_decode.py; the same random experts, E2M1 codes with
power-of-two block scales as the checkpoint's MXFP4 ones), and --moe-groups times the step with MoE
layers grouped (model_runner._piecewise moe_group).

--allreduce spawns --ranks ranks (2 by default; one NeuronCore each, tp.neuron_env / init_rank
as the engine does) and times graphs of 1 and 12 all-reduces of the hidden state against the
same graph without them, synchronously and as 48 chained launches (the runner's decode step).
KILN_CORE_BASE=<n> puts rank 0 on NeuronCore n (several probes side by side on a 64-core trn2).
KILN_PROBE_COLLECTIVES=1 adds other collectives, subgroups holding rank 0, and attention TP's
shape: every group of 2 (and 8) consecutive ranks all-reducing over itself at once, each
group's sum checked (needs --ranks 4 or more). KILN_PROBE_DP=N times DP attention's exchange
(models/decoder.py DecoderForCausalLM: DP attention) for N groups of ranks / N, each holding a
[T, H] head partial of its own T = --batch tokens, against its alternatives, each result checked
against the sum it must equal: Kiln's zero-padded [N * T, H] world all-reduce (SGLang
_dp_gather_via_all_reduce, DpPaddingMode.SUM_LEN, attention reduction folded in), SGLang's MAX_LEN
form (reduce-scatter over the attention group, then a world all-gather of [T / attn_tp, H]; v0.5.21
srt/layers/dp_attention.py _dp_gather_via_all_gather), and a group all-reduce followed by a world
all-gather of the whole [T, H] (attn_tp copies of every group's rows).

Do not run --prefill with --what groups on a trn1.2xlarge: a 6-layer MoE prefill graph
compiled for over 10 minutes and starved the host (docs/neuron-notes.md).
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn.functional as F

OPTS: dict = {}
DEV = None
RANK = 0  # with --ranks > 1 only rank 0 prints


def say(*a, **k) -> None:
    if RANK == 0:
        print(*a, **k)


def setup_device(cpu: bool = False) -> None:
    global DEV
    import torch._dynamo as dynamo

    # As ModelRunner: many graphs share one code object (group_fn's f, with_layer's f).
    for knob in ("cache_size_limit", "recompile_limit", "accumulated_cache_size_limit",
                 "accumulated_recompile_limit"):
        if hasattr(dynamo.config, knob):
            setattr(dynamo.config, knob, 1 << 20)
    if cpu:  # dry run of this script's logic, no device
        DEV = torch.device("cpu")
        OPTS.update(backend="eager", fullgraph=True, dynamic=False)
        return
    # The engine's runtime knobs (engine.build_shard), set before LNL is imported.
    os.environ.setdefault("NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS", "63")
    os.environ.setdefault("NEURON_RT_IO_RING_CACHE_SIZE", "32")
    from kiln import platform

    platform.configure_runtime_env()  # NEURON_LOGICAL_NC_CONFIG on trn2 / trn3, as the engine
    import libtorch_neuronx_lite  # noqa: F401

    from kiln.engine.model_runner import canonical_neuron_backend, neuronx_cc_args

    DEV = torch.device("neuron:0")
    OPTS.update(backend=canonical_neuron_backend(), fullgraph=True, dynamic=False,
                options={"compiler_args": neuronx_cc_args(torch.bfloat16, True)})
    if os.environ.get("KILN_PROFILE_COMPILE_TIMEOUT"):  # the other ranks' wait for rank 0's compile (LNL: 600 s)
        OPTS["options"]["compilation_timeout"] = int(os.environ["KILN_PROFILE_COMPILE_TIMEOUT"])


def _sync(out) -> None:
    while isinstance(out, (tuple, list)):
        out = out[0]
    out.cpu()


NEFF = False  # --neff: print each newly compiled graph's instruction counts (tools/neff_instructions.py)


def timed(name: str, fn, args, iters: int = int(os.environ.get("KILN_PROFILE_ITERS", "20"))) -> float:
    c = torch.compile(fn, **OPTS)
    t = time.perf_counter()
    t_wall = time.time()
    try:
        _sync(c(*args))
    except Exception as e:  # one piece failing must not hide the rest
        say(f"  {name:<46} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
        if os.environ.get("KILN_PROFILE_TRACEBACK") and RANK == 0:
            import traceback

            traceback.print_exc()
        return float("nan")
    compile_s = time.perf_counter() - t
    if NEFF and RANK == 0:
        import neff_instructions as ni

        for e in ni.latest(1, since=t_wall):
            say(f"  {'':<46} {ni.line(e)}", flush=True)
    _sync(c(*args))
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        _sync(c(*args))
        ts.append(time.perf_counter() - t)
    ts.sort()
    say(f"  {name:<46} p50 {ts[iters // 2] * 1e3:8.3f} ms   min {ts[0] * 1e3:8.3f}   (compile {compile_s:6.1f} s)",
          flush=True)
    return ts[iters // 2]


def random_fill(t: torch.Tensor, name: str, g: torch.Generator) -> torch.Tensor:
    from kiln.models.quant import E2M1, FP8

    if name in ("w_gu", "w_down") and t.dtype == FP8:
        # Experts as the checkpoint's MXFP4 ones convert (models/quant.py): E2M1 codes as FP8 ...
        return torch.tensor(E2M1)[torch.randint(0, 16, t.shape, generator=g)].to(FP8)
    if name in ("w_gu_scale", "w_down_scale") and t.dtype == torch.float32 and t.dim() == 3:
        # FP8 128 x 128 block scales as the loader keeps them (GLM-5.3-Flash at tp=32): one per row
        # block of gate and of up rows per 128 input columns; one per 128 output columns of down
        # (the rank's 64 input rows sit in one block). kernels/moe_prefill.py relies on the latter.
        E, R, Cn = t.shape
        if name == "w_gu_scale":
            sc = (torch.rand(E, 2, Cn, generator=g) * 0.02 + 0.005).repeat_interleave(R // 2, dim=1)
        else:
            sc = (torch.rand(E, R, Cn // 128, generator=g) * 0.02 + 0.005).repeat_interleave(128, dim=2)
        if os.environ.get("KILN_PROFILE_EXPERT_LAYOUT") == "loaded":
            # As the loader leaves GLM-5.3-Flash's real experts (models/quant.fit_e4m3_max per (row, block):
            # most rows' scales doubled, so neither the gate rows' tile scales nor the down chunks' are
            # constant, and kernels/moe_prefill.py takes its per-row and per-column paths)
            sc = sc * (1.0 + (torch.rand(sc.shape, generator=g) < 0.6).float())
        return sc
    if name in ("w_gu_scale", "w_down_scale") and t.dtype == torch.bfloat16:
        # ... with power-of-two block scales (what kernels/moe_dedupe.py's tile layout needs)
        return torch.exp2(torch.randint(-10, -3, t.shape, generator=g).float()).bfloat16()
    if t.dtype == FP8:
        # Random bytes with exponent bit 3 cleared: finite e4m3 values up to 1.875, never the
        # all-ones exponent that trn1's e4m3 reads as inf / NaN.
        return (torch.randint(0, 256, t.shape, dtype=torch.uint8, generator=g) & 0xBF).view(FP8)
    if name.endswith("_scale"):
        return (torch.rand(t.shape, generator=g) * 0.02 + 0.005).to(t.dtype)
    if "norm" in name:
        return torch.ones(t.shape, dtype=t.dtype)
    if name in ("sink", "router_bias"):
        return torch.randn(t.shape, generator=g).to(t.dtype)
    return (torch.randn(t.shape, generator=g) * 0.02).to(t.dtype)


def build(model_id: str, tp: int, n_layers: int, num_pages: int, ps: int, max_pos: int, ranks: int = 1,
          max_rows: int = 4096, max_keys: int | None = None, attn_tp: int | None = None, dp_attention: int = 1):
    from huggingface_hub import hf_hub_download

    from kiln.config import ModelConfig
    from kiln.models.decoder import DecoderForCausalLM

    cfg_dir = os.path.dirname(hf_hub_download(model_id, "config.json"))
    cfg = ModelConfig.from_pretrained(cfg_dir).truncated(n_layers)
    with torch.device("meta"):
        model = DecoderForCausalLM(cfg, torch.bfloat16, max_pos, 0, tp, None, keep_fp8=cfg.quant_block is not None,
                                   vocab_parallel=True, moe_kernel=os.environ.get("KILN_MOE_KERNEL", "xla"),
                                   attn_tp=attn_tp, dp_attention=dp_attention)
    g = torch.Generator().manual_seed(0)
    for n, p in list(model.named_parameters(recurse=False)):  # vocab-parallel embed / lm_head, final norm
        setattr(model, n, torch.nn.Parameter(random_fill(p, n, g).to(DEV), requires_grad=False))
    model.register_buffer("vocab_start_t", torch.tensor(model.vocab_start, dtype=torch.int64).to(DEV),
                          persistent=False)
    for layer in model.layers:
        for n, p in list(layer.named_parameters(recurse=False)):
            setattr(layer, n, torch.nn.Parameter(random_fill(p, n, g), requires_grad=False))
        if layer.pack_moe:  # same bytes, the kernel's layout
            layer.pack_experts()
        for n, p in list(layer.named_parameters(recurse=False)):
            setattr(layer, n, torch.nn.Parameter(p.data.to(DEV), requires_grad=False))
    for name, (dim, theta) in model.rope_specs().items():
        cos, sin = model._rope_table(dim, theta, max_pos)
        model.register_buffer(f"rope_cos{name}", cos.to(torch.bfloat16).to(DEV), persistent=False)
        model.register_buffer(f"rope_sin{name}", sin.to(torch.bfloat16).to(DEV), persistent=False)
    slots = num_pages * ps
    ks, vs = [], []
    for (k, v) in model.kv_shapes():
        ks.append((torch.randn(slots, *k, generator=g) * 0.5).to(torch.bfloat16).to(DEV))
        vs.append((torch.randn(slots, *v, generator=g) * 0.5).to(torch.bfloat16).to(DEV))
    model.bind_kv_cache(ks, vs, ps, max_rows=max_rows, max_keys=max_keys or max_pos)  # DSA's scratch (models/mla.py)
    # Token-slot states beside K and V (Inkling's conv inputs, a pooled DSA indexer's pool keys).
    states = [{n: (torch.randn(slots, *shp, generator=g) * 0.5).to(torch.bfloat16).to(DEV) for n, shp in d.items()}
              for d in model.token_state_shapes()]
    if any(states):
        model.bind_token_states(states)
    if ranks == 1:
        model.tp_size = 1  # all-reduces become the identity: one rank's compute only
    else:  # rank-0 shapes for --tp, but the all-reduces run over the `ranks` live processes
        import torch.distributed as dist

        model.tp_size, model.tp_group = ranks, dist.group.WORLD
    model.attn_tp, model.attn_group = model.tp_size, model.tp_group  # attention all-reduces alike
    model.dp_buffers(DEV)  # DP attention: rank 0's group (group 0)
    return cfg, model


def decode_inputs(model, B: int, P: int, ps: int):
    """Host-built decode inputs as ModelRunner.decode builds them: B sequences on distinct
    pages, each at the end of a full P-page context, plus the sliding-window table."""
    import numpy as np

    from kiln.engine.model_runner import ModelRunner

    L = P * ps
    table = np.arange(1, 1 + B * P, dtype=np.int64).reshape(B, P)
    ctx = np.full(B, L, np.int64)
    pos = ctx - 1
    slot = table[:, -1] * ps + (L - 1) % ps
    win = model.window
    shim = type("S", (), {"window": win, "ps": ps})()
    Pw = -(-(win + 1 - 1) // ps) + 1 if win is not None else 0
    Pw = Pw if 0 < Pw < P else 0
    swa = np.zeros((B, Pw), np.int64)
    first = np.zeros(B, np.int64)
    for b in range(Pw and B):
        first[b] = ModelRunner._swa_row(shim, list(table[b]), int(pos[b]), Pw, swa[b])
    t = lambda a: torch.from_numpy(a)  # noqa: E731
    j = torch.arange(L).unsqueeze(0)
    attn = model._attn_inputs(j < t(ctx).unsqueeze(1), t(pos).unsqueeze(1), t(table),
                              t(swa) if Pw else None, t(first) if Pw else None, (B, 1, 1))
    d = lambda a: a.to(DEV)  # noqa: E731
    bias, tab = attn[None]
    bias_w, tab_w = attn[win] if win is not None else (bias, tab)
    return dict(positions=d(t(pos)), slots=d(t(slot)), table=d(tab), bias=d(bias),
                table_w=d(tab_w), bias_w=d(bias_w))


def verify_inputs(model, B: int, Q: int, P: int, ps: int):
    """Host-built verify inputs as ModelRunner.verify builds them: B sequences on distinct pages, each
    scoring its last Q positions of a full P-page context (the newest token and Q - 1 drafted ones)."""
    import numpy as np

    if model.window is not None:
        raise SystemExit("--verify: sliding-window tables are not built here")
    L = P * ps
    table = np.arange(1, 1 + B * P, dtype=np.int64).reshape(B, P)
    pos = (L - Q + np.arange(Q, dtype=np.int64))[None].repeat(B, 0)  # [B, Q]
    slot = np.take_along_axis(table, pos // ps, 1) * ps + pos % ps
    t = lambda a: torch.from_numpy(a)  # noqa: E731
    j = torch.arange(L).view(1, 1, -1)
    p = t(pos).unsqueeze(-1)
    attn = model._attn_inputs(j <= p, p, t(table), None, None, (B, 1, 1, Q))
    d = lambda a: a.to(DEV)  # noqa: E731
    bias, tab = attn[None]
    return dict(positions=d(t(pos).reshape(-1)), slots=d(t(slot).reshape(-1)), table=d(tab), bias=d(bias),
                table_w=d(tab), bias_w=d(bias), Q=Q)


def prefill_inputs(model, C: int, P: int, ps: int, off: int = 0):
    """One sequence's C-token chunk at positions off..off+C-1 on pages 1..P (ModelRunner.prefill)."""
    import numpy as np

    from kiln.engine.model_runner import ModelRunner

    table = np.arange(1, 1 + P, dtype=np.int64)
    pos = np.arange(off, off + C, dtype=np.int64)
    slot = table[pos // ps] * ps + pos % ps
    win = model.window
    shim = type("S", (), {"window": win, "ps": ps})()
    Pw = -(-(win + C - 1) // ps) + 1 if win is not None else 0
    Pw = Pw if 0 < Pw < P else 0
    swa = np.zeros(Pw, np.int64)
    first = np.array([ModelRunner._swa_row(shim, list(table), 0, Pw, swa)] if Pw else [0], np.int64)
    t = lambda a: torch.from_numpy(a)  # noqa: E731
    j = torch.arange(P * ps).unsqueeze(0)
    p = t(pos).unsqueeze(1)
    attn = model._attn_inputs(j <= p, p, t(table), t(swa) if Pw else None, t(first) if Pw else None, (1, 1, C))
    d = lambda a: a.to(DEV)  # noqa: E731
    bias, tab = attn[None]
    bias_w, tab_w = attn[win] if win is not None else (bias, tab)
    return dict(positions=d(t(pos)), slots=d(t(slot)), table=d(tab), bias=d(bias),
                table_w=d(tab_w), bias_w=d(bias_w))


def with_layer(model, i: int, body):
    """f(*inputs, *layer i's tensors) = body(view of layer i, *inputs), the layer's tensors
    passed as graph inputs exactly as group_fn passes them."""
    from kiln.models.decoder import _LayerView

    static, tensors = model._layer_split(i)
    names = tuple(tensors)

    def f(*a):
        n = len(a) - len(names)
        return body(_LayerView(static, dict(zip(names, a[n:]))), *a[:n])

    return f, tuple(tensors.values())


def linear_parts(model, cfg, i: int, inp: dict, T: int, ss, R: int | None = None) -> None:
    """A hyper-connection model's linear-attention layer piece by piece (models/hybrid.py layer,
    models/linear_attn.mix): both blocks with their hyper-connections, the mixer and the MLP alone,
    the mHC collapse alone, and the mixer's own pieces; ss: the state slot (chunk form). R: the hidden
    state's rows (T x DP-attention groups; the attention block and the mixer take them all and run their group's
    T, linear_attn.mix's _attn_in), T the rows of every per-group piece."""
    import torch.nn.functional as F_

    from kiln.models import hybrid as hy
    from kiln.models import linear_attn as la
    from kiln.models.decoder import rms_norm

    L = model.layers[i]
    sp = L.spec
    g = torch.Generator().manual_seed(1)
    H = cfg.hidden_size
    W = cfg.hybrid.hc * H
    pos, slots = inp["positions"], inp["slots"]
    eps = cfg.rms_norm_eps
    R = R or T
    streams = (torch.randn(R, W, generator=g)).to(torch.bfloat16).to(DEV)
    xR = rms_norm(torch.randn(R, H, generator=g).to(torch.bfloat16), torch.ones(H, dtype=torch.bfloat16), eps).to(DEV)
    x = xR[:T].contiguous()
    say(f"layer {i} ({sp.kind}): {L.nk} k / {L.nv} v heads x {sp.head_k_dim} on this rank, conv {L.conv_dim}, "
        f"KILN_LINEAR_ATTN_KERNEL={la.LINEAR_ATTN_KERNEL} (takes this layer: {la.kernel_takes(sp)})", flush=True)

    def run(name, body, *args):
        f, ts = with_layer(model, i, body)
        return timed(f"[{i}] {name}", lambda *a: _small(f(*a)), (*args, *ts))

    def mixer(V, x, p, s):
        return la.mix(model, V, x, p, s, ss)

    run("attn block (mHC + mixer)", lambda V, st, p, s: hy._block(model, V, "attn", st, V.in_norm,
                                                                    lambda x: mixer(V, x, p, s)), streams, pos, slots)
    run("ffn block (mHC + MLP)", lambda V, st: hy._block(model, V, "ffn", st, V.post_norm,
                                                         lambda x: hy._mlp(model, V, x)), streams)
    run("mHC collapse alone", lambda V, st: hy._mhc(model, V, "attn", st.view(R, cfg.hybrid.hc, H))[0], streams)
    run("mixer alone (linear_attn.mix)", mixer, xR, pos, slots)
    from kiln.kernels import gated_norm as gn

    if la.kernel_takes(sp) and sp.kind == "kda":  # the same mixer with the gated norm as a kernel toggled
        was = gn.MODE
        gn.MODE = "0" if gn.takes("kda", sp.gate_act, sp.head_v_dim, default=True) and was != "0" else "1"
        try:
            run(f"mixer alone, KILN_KDA_FUSED_NORM={gn.MODE}", mixer, xR, pos, slots)
        finally:
            gn.MODE = was
    run("MLP alone", lambda V, x: hy._mlp(model, V, x), x)
    run("in_qkv projection", lambda V, x: F_.linear(x, model._w(V, "in_qkv")), x)
    run("KDA gates (f_a, f_b, b, g_a, g_b)", lambda V, x: (F_.linear(F_.linear(x, V.f_a), V.f_b).float().sum()
                                                          + F_.linear(x, model._w(V, "in_b")).float().sum()
                                                          + F_.linear(F_.linear(x, V.g_a), V.g_b).float().sum()), x)
    if sp.kind == "kda" and L.g_a is not None:  # the projections of x as one matmul (fix 4's sizing), against both
        cd, dk_, gr = L.conv_dim, sp.head_k_dim, L.g_a.shape[0]

        def unfused(V, x):
            return (F_.linear(x, model._w(V, "in_qkv")).float().sum()
                    + F_.linear(F_.linear(x, V.f_a), V.f_b).float().sum() + F_.linear(x, model._w(V, "in_b")).float().sum()
                    + F_.linear(F_.linear(x, V.g_a), V.g_b).float().sum())

        def fused(V, x):
            y = F_.linear(x, torch.cat([model._w(V, "in_qkv"), V.f_a, V.g_a, model._w(V, "in_b")]))
            return (y[:, :cd].float().sum() + F_.linear(y[:, cd:cd + dk_], V.f_b).float().sum()
                    + y[:, cd + dk_ + gr:].float().sum() + F_.linear(y[:, cd + dk_:cd + dk_ + gr], V.g_b).float().sum())

        run("in_qkv + gates, one graph", unfused, x)
        run("in_qkv + gates, one concatenated matmul", fused, x)
    D = L.conv_dim
    xe = torch.randn(T + sp.conv_kernel - 1, D, generator=g).to(torch.bfloat16).to(DEV)
    run("short conv", lambda V, xe: la._causal_conv(xe, V.conv_w, T), xe)
    run("output projection", lambda V, o: F_.linear(o, model._w(V, "out")), torch.randn(T, L.nv * sp.head_v_dim).to(torch.bfloat16).to(DEV))
    if sp.kind == "kda":  # the gated norm between the delta rule and the output projection, both ways
        o32 = torch.randn(T, L.nv, sp.head_v_dim, generator=g).to(DEV)
        zg = torch.randn(T, L.nv, sp.head_v_dim, generator=g).to(torch.bfloat16).to(DEV)
        run("gated norm (XLA, _mix_rows's tail)", lambda V, o, z: gn.reference(o, z, V.o_norm, eps), o32, zg)
        if la.kernel_takes(sp):
            run("gated norm (kernels/gated_norm.py)", lambda V, o, z: gn.apply(o, z, V.o_norm, eps), o32, zg)
    from tests.test_delta_rule import inputs

    q, k, v, gate, beta, S0 = (t.to(DEV) for t in inputs(T, L.nk, L.nv, sp.kind == "kda", seed=5,
                                                          lower_bound=sp.lower_bound or -5.0))
    from kiln.kernels import delta_rule as dr

    if la.kernel_takes(sp):
        timed(f"[{i}] delta rule, NKI kernel", lambda *a: _small(dr.chunk(*a)), (q, k, v, gate, beta, S0))
    timed(f"[{i}] delta rule, chunk_scan (sub-chunks {la.CHUNKS[sp.kind]})",
          lambda *a: _small(la.chunk_scan(*a, la.CHUNKS[sp.kind])), (q, k, v, gate, beta, S0))


def hc_blocks(model, cfg, i: int, inp: dict, T: int, R: int, ss) -> None:
    """A hyper-connection layer (models/hybrid.py layer) of a prefill chunk block by block, at the
    hidden state's R rows (R = T x DP-attention groups; the token mixer takes its group's T): both
    blocks with their hyper-connections, the mHC collapse and the output mix alone, the token mixer
    and the MLP alone (each with its all-reduce), the MoE experts without it, the all-reduces
    themselves, and a DSA layer's projections, selection and attention core (models/mla.py). Every
    piece reads back the fp32 sum of its output (_small)."""
    import torch.nn.functional as F_

    from kiln.config import LinearSpec
    from kiln.models import hybrid as hy
    from kiln.models import mla
    from kiln.models.decoder import rms_norm

    L = model.layers[i]
    sp = L.spec
    g = torch.Generator().manual_seed(1)
    H = cfg.hidden_size
    hc = cfg.hybrid.hc
    W = hc * H
    pos, slots, table, bias = inp["positions"], inp["slots"], inp["table"], inp["bias"]
    eps = cfg.rms_norm_eps
    streams = torch.randn(R, W, generator=g).to(torch.bfloat16).to(DEV)
    xr = rms_norm(torch.randn(R, H, generator=g).to(torch.bfloat16), torch.ones(H, dtype=torch.bfloat16), eps)
    x = xr.to(DEV)
    kind = ("linear" if isinstance(sp, LinearSpec) else "mla") + (" moe" if L.moe else " dense")
    say(f"layer {i} ({kind}) blocks at {R} rows ({T} per token-mixer group)", flush=True)

    def run(name, body, *args):
        f, ts = with_layer(model, i, body)
        return timed(f"[{i}] {name}", lambda *a: _small(f(*a)), (*args, *ts))

    def mixer(V, x, p, s, tb, bs):
        return hy._mix(model, V, x, p, s, tb, bs, ss, None)

    run("attn block (mHC + mixer + all-reduce + mix)", lambda V, st, p, s, tb, bs: hy._block(
        model, V, "attn", st, V.in_norm, lambda x: mixer(V, x, p, s, tb, bs)), streams, pos, slots, table, bias)
    run("ffn block (mHC + MLP + all-reduce + mix)", lambda V, st: hy._block(
        model, V, "ffn", st, V.post_norm, lambda x: hy._mlp(model, V, x)), streams)
    run("mHC collapse alone (_mhc)", lambda V, st: hy._mhc(model, V, "attn", st.view(R, hc, H))[0], streams)
    post = torch.rand(R, hc, generator=g).to(DEV)
    comb = torch.rand(R, hc, hc, generator=g).to(DEV)
    timed(f"[{i}] mHC output mix alone (mix_out)", lambda po, co, y, st: _small(hy.mix_out(
        po, co, y, st.view(R, hc, H), hy.MHC_FORM)), (post, comb, x, streams))
    run("token mixer alone, with its all-reduce", mixer, x, pos, slots, table, bias)
    run("MLP alone (experts + shared + all-reduce)", lambda V, x: hy._mlp(model, V, x), x)
    if L.moe:
        lim = cfg.hybrid.swiglu_limit
        run("MoE routed experts alone (route + kernel)", lambda V, x: hy._moe_clamped(model, V, x, lim), x)
        run("MoE router alone", lambda V, x: model._route(V, x), x)
        run("shared expert alone", lambda V, x: hy._swiglu(model, V, x, "shared_gate_up", "shared_down", lim), x)
    # sequence-parallel streams (prefill chunks only): R / tp rows per rank
    if model._sp_on() and R % model.tp_size == 0 and inp["table"].dim() == 1:
        r = R // model.tp_size
        st_sp = streams[:r].contiguous()
        x_sp = x[:r].contiguous()
        rows = lambda fn: (lambda y: model._sp_rows(fn(model._sp_gather(y))))  # noqa: E731 (models/hybrid.py layer)
        run(f"SP attn block ({r} rows: mHC, gather, mixer, all-reduce, own rows, mix)",
            lambda V, st, p, s, tb, bs: hy._block(model, V, "attn", st, V.in_norm,
                                                  rows(lambda y: mixer(V, y, p, s, tb, bs))), st_sp, pos, slots, table, bias)
        run(f"SP ffn block ({r} rows)", lambda V, st: hy._block(model, V, "ffn", st, V.post_norm,
                                                               rows(lambda y: hy._mlp(model, V, y))), st_sp)
        if L.moe:
            run(f"SP ffn block ({r} rows), own rows routed (KILN_SP_ROUTE)", lambda V, st: hy._block(
                model, V, "ffn", st, V.post_norm, lambda y: model._sp_rows(hy._mlp(
                    model, V, model._sp_gather(y), hy._sp_route(model, V, y)))), st_sp)
            run(f"SP routing alone ({r} rows routed, [{R}, 2k] gathered)", lambda V, y: hy._sp_route(model, V, y), x_sp)
        # as models/hybrid.py layer builds them, with each output reduction a reduce-scatter (KILN_SP_RS)
        old_rs = os.environ.get("KILN_SP_RS")
        os.environ["KILN_SP_RS"] = "1"
        try:
            run(f"SP ffn block ({r} rows), as layer builds it, reduce-scatter output",
                lambda V, st: hy._block(model, V, "ffn", st, V.post_norm, hy._ffn(model, V, True)), st_sp)
            if model._sp_rs_attn():
                rsx = lambda fn: (lambda y, f=hy._sp_out(model, fn, True): f(model._sp_gather(y)))  # noqa: E731
                run(f"SP attn block ({r} rows), reduce-scatter output", lambda V, st, p, s, tb, bs: hy._block(
                    model, V, "attn", st, V.in_norm, rsx(lambda y: mixer(V, y, p, s, tb, bs))),
                    st_sp, pos, slots, table, bias)
            if model._sp_grp_ok():
                run(f"SP attn block ({r} rows), group gather and reduce-scatter (KILN_SP_GROUP)",
                    lambda V, st, p, s, tb, bs: hy._block(model, V, "attn", st, V.in_norm, hy._attn_rows(
                        model, lambda y: mixer(V, y, p, s, tb, bs), True)), st_sp, pos, slots, table, bias)
                timed(f"[{i}] SP group gather [{r} -> {model.attn_tp * r}, {H}]",
                      lambda y: _small(model._sp_group_gather(y)), (x_sp,))
        finally:
            if old_rs is None:
                os.environ.pop("KILN_SP_RS")
            else:
                os.environ["KILN_SP_RS"] = old_rs
        if L.moe:  # the same routing both ways on the device: own rows routed and gathered, against all rows routed
            f1, ts1 = with_layer(model, i, lambda V, y: hy._sp_route(model, V, y))
            f2, ts2 = with_layer(model, i, lambda V, y: model._route(V, model._sp_gather(y)))
            (v1, i1), (v2, i2) = (tuple(t.cpu() for t in torch.compile(f, **OPTS)(x_sp, *ts)) for f, ts in ((f1, ts1), (f2, ts2)))
            i1, i2 = i1.to(torch.int64), i2.to(torch.int64)  # the device's topk indices are uint32
            say(f"  [{i}] SP routing: expert indices equal {(i1 == i2).float().mean().item():.6f}, weights max |diff| "
                f"{(v1.float() - v2.float()).abs().max().item():.3e} (own rows routed vs all {R} rows routed)", flush=True)
        run(f"SP mHC collapse alone ({r} rows)", lambda V, st: hy._mhc(model, V, "attn", st.view(r, hc, H))[0], st_sp)
        timed(f"[{i}] SP mHC output mix alone ({r} rows)", lambda po, co, y, st: _small(hy.mix_out(
            po, co, y, st.view(r, hc, H), hy.MHC_FORM)), (post[:r].contiguous(), comb[:r].contiguous(), x_sp, st_sp))
        timed(f"[{i}] SP gather [{r} -> {R}, {H}] (zero-padded world all-reduce)",
              lambda y: _small(model._sp_gather(y)), (x_sp,))
    timed(f"[{i}] world all-reduce [{R}, {H}] bf16", lambda y: _small(model._all_reduce(y)), (x,))
    xt = x[:T].contiguous()
    timed(f"[{i}] attention all-reduce of a [{T}, {H}] partial", lambda y: _small(model._attn_all_reduce(y)), (xt,))
    if getattr(sp, "mla", None) is None:
        return
    m = sp.mla
    d = m.dsa

    def proj(V, x, p):
        q_nope, q_pe, c, k_pe, q_resid = mla._project(model, V, x, p)
        out = [q_nope, c]
        if d is not None and d.indexer:
            qi, ki, wi = mla._indexer(model, V, x, q_resid, p)
            out += [qi[1], ki, wi]
        return tuple(out)

    run("MLA projections + indexer (group rows)", proj, xt, pos)
    pooled = L.pool_key is not None
    if pooled:
        # The rows' own slots as slot_mapping (every row real here).
        run("pool-key write (write_pool_keys)",
            lambda V, p, tb: (mla.write_pool_keys(model, V, d, p, tb, model._slots(p.unsqueeze(-1), tb).reshape(-1)),
                              p + 1)[1], pos, table)
    run("cache loads (latents, rope/index keys" + (", pool keys)" if pooled else ")"),
        lambda V, tb: _small((model._load(V.k_cache, tb), model._load(V.v_cache, tb))
                             + ((model._gather(V.pool_key.unsqueeze(1), tb),) if pooled else ())), table)
    kc = model._load(L.k_cache, table).squeeze(-2)  # [(B,) L, r] (mla.attention)
    kv = model._load(L.v_cache, table).squeeze(-2)
    pk = model._gather(L.pool_key.unsqueeze(1), table).squeeze(-2) if pooled else None
    if table.dim() == 1:  # chunk: one sequence, T queries
        B_, Q_ = 1, T
        kc, kv = kc.unsqueeze(0), kv.unsqueeze(0)
        pk = pk.unsqueeze(0) if pooled else None
    elif inp.get("Q"):  # verify: T / Q sequences, Q queries each
        B_, Q_ = T // inp["Q"], inp["Q"]
    else:  # decode: T sequences, one query each
        B_, Q_ = T, 1
    Lk = kc.shape[1]
    vis = bias.reshape(B_, Q_, Lk)
    nh, dn, dv = L.nh, m.qk_nope_head_dim, m.v_head_dim
    qn = (torch.randn(T, nh, dn, generator=g) * 0.1).to(torch.bfloat16).to(DEV)
    qp = qn[..., :0]
    if d is not None and d.indexer:
        qi = (torch.randn(B_, Q_, d.n_heads, d.head_dim, generator=g) * 0.1).to(DEV)
        wi = (torch.randn(B_, Q_, d.n_heads, generator=g)).to(DEV)
        kI = kv[..., : d.head_dim * (2 if d.kpool > 1 else 1)]
        if pooled:
            pkr = pk.reshape(B_, Lk // d.kpool, d.head_dim)
            run(f"DSA selection alone (pooled_selection, cached pool keys, {Lk} keys)",
                lambda V, qi, wi, kI, vs, pk_: mla._layer_selection(V, d, ((None, qi), wi, kI, pk_), vs, B_, Q_),
                qi, wi, kI, vis, pkr)
        run(f"DSA selection alone (pooled_selection, {Lk} keys)", lambda V, qi, wi, kI, vs: mla._layer_selection(
            V, d, ((None, qi), wi, kI), vs, B_, Q_), qi, wi, kI, vis)
    sel = torch.zeros(B_, Q_, Lk).to(DEV)
    for form in ("expand", "absorb"):
        run(f"MLA core with a given mask, {form} ({Lk} keys)", lambda V, qn, kc, kp, vs, top, ab=(form == "absorb"):
            mla._core(model, V, qn, qp, kc, kp, vs, B_, Q_, top, None, ab)[0], qn, kc, kv[..., :0], vis, sel)
    run("MLA attention block (mla.attention)", lambda V, x, p, s, tb, bs: mla.attention(model, V, x, p, s, tb, bs),
        xt, pos, slots, table, bias)
    run("o_proj", lambda V, o: F_.linear(o, model._w(V, "o")), torch.randn(T, nh * dv, generator=g).to(torch.bfloat16).to(DEV))


def _with_state_slot(f, ss):
    """group_fn's f with state_slot bound (the chunk form of a linear-attention layer)."""
    def g(*a):
        return f(*a, state_slot=ss)
    return g


def _small(out):
    """A small readback of a piece's output(s): the sum of every element in fp32."""
    if isinstance(out, (tuple, list)):
        return sum(o.float().sum() for o in out)
    return out.float().sum()


def parts(model, cfg, i: int, inp: dict, T: int, label: str) -> None:
    from kiln.models import decoder as dec
    from kiln.models.decoder import rms_norm

    L = model.layers[i]
    windowed = L.spec.window is not None
    table, bias = (inp["table_w"], inp["bias_w"]) if windowed else (inp["table"], inp["bias"])
    g = torch.Generator().manual_seed(1)
    H = cfg.hidden_size
    h = (torch.randn(T, H, generator=g)).to(torch.bfloat16).to(DEV)
    x = rms_norm(h.cpu(), torch.ones(H, dtype=torch.bfloat16), cfg.rms_norm_eps).to(DEV)
    pos, slots = inp["positions"], inp["slots"]
    eps = cfg.rms_norm_eps
    say(f"layer {i} ({label}): nh={L.nh} nkv={L.nkv} Dk={L.spec.head_dim} Dv={L.spec.v_head_dim} "
          f"rope={L.spec.rope_dim} window={L.spec.window} sink={L.sink is not None} keys={bias.shape[-1]} "
          f"qkv {tuple(L.qkv.shape)} {L.qkv.dtype} scale {None if L.qkv_scale is None else tuple(L.qkv_scale.shape)}",
          flush=True)

    def run(name, body, *args):
        f, ts = with_layer(model, i, body)
        return timed(f"[{i}] {name}", f, (*args, *ts))

    run("attention block", lambda V, h, p, s, tb, bs: model._attention(V, h, p, s, tb, bs), h, pos, slots, table, bias)
    run("mlp block", lambda V, h: model._mlp(V, h), h)
    run("rms_norm", lambda V, h: rms_norm(h, V.in_norm, eps), h)
    run("qkv weight dequant", lambda V: model._w(V, "qkv").sum(dim=0))
    run("qkv dequant + matmul", lambda V, x: F.linear(x, model._w(V, "qkv"), V.qkv_bias), x)
    wq = (torch.randn(*L.qkv.shape, generator=g) * 0.02).to(torch.bfloat16).to(DEV)
    timed(f"[{i}] qkv matmul, bf16 weight", lambda w, x: F.linear(x, w), (wq, x))
    run("_qkv (matmul, split, rope, v_scale)", lambda V, x, p: model._qkv(V, x, p), x, pos)
    sp = L.spec
    q, k, v = (torch.randn(T, n, d, generator=g).to(torch.bfloat16).to(DEV)
               for n, d in ((L.nh, sp.head_dim), (L.nkv, sp.head_dim), (L.nkv, sp.v_head_dim)))

    def store(V, k, v, s):
        model._store(V.k_cache, s, k)
        model._store(V.v_cache, s, v)
        return s + 1

    run("store k, v (index_put_)", store, k, v, slots)
    for mode in ("token", "page"):
        dec.GATHER = mode
        run(f"gather k, v ({mode})", lambda V, tb: (model._load(V.k_cache, tb).float().sum(dim=-3),
                                                     model._load(V.v_cache, tb).float().sum(dim=-3)), table)
    dec.GATHER = "auto"
    kc, vc = model._load(L.k_cache.cpu(), table.cpu()), model._load(L.v_cache.cpu(), table.cpu())
    kc, vc = kc.to(DEV), vc.to(DEV)
    run("attend (einsums + softmax)", lambda V, q, kc, vc, tb, bs: model._attend(V, q, kc, vc, tb, bs),
        q, kc, vc, table, bias)
    G = L.nh // L.nkv
    if table.dim() == 2:
        s = torch.randn(T, L.nkv, G, bias.shape[-1], generator=g).to(DEV)
        run("softmax (+ sink), fp32", lambda V, s: model._softmax(V, s, kv_axis=1), s)
    o = torch.randn(T, L.nh * L.spec.v_head_dim, generator=g).to(torch.bfloat16).to(DEV)
    run(f"o_proj ({L.o.dtype})", lambda V, o: F.linear(o, model._w(V, "o")), o)
    if L.moe and L.moe_blob:
        run("moe (route + NKI kernel)", lambda V, x: model._moe(V, x), x)
        run("moe route", lambda V, x: model._route(V, x), x)
    elif L.moe:
        run("moe (route + gather + dequant + bmm)", lambda V, x: model._moe(V, x), x)
        run("moe route", lambda V, x: model._route(V, x), x)
        k_ = cfg.num_experts_per_tok
        flat = torch.randint(0, cfg.num_experts, (T * k_,), generator=g).to(DEV)
        run("moe w_gu gather + dequant", lambda V, f: model._experts(V, "w_gu", f).sum(dim=(1, 2)), flat)
        run("moe w_down gather + dequant", lambda V, f: model._experts(V, "w_down", f).sum(dim=(1, 2)), flat)
        say(f"    experts: w_gu {tuple(L.w_gu.shape)} scale {tuple(L.w_gu_scale.shape)} {L.w_gu_scale.dtype}; "
              f"w_down {tuple(L.w_down.shape)} scale {tuple(L.w_down_scale.shape)}", flush=True)
    else:
        I = model.inter
        run("dense gate_up weight dequant", lambda V: model._w(V, "gate_up").sum(dim=0))
        run("dense gate_up dequant + matmul", lambda V, x: F.linear(x, model._w(V, "gate_up")), x)
        a = torch.randn(T, I, generator=g).to(torch.bfloat16).to(DEV)
        run("dense down weight dequant", lambda V: model._w(V, "down").sum(dim=0))
        run("dense down dequant + matmul", lambda V, a: F.linear(a, model._w(V, "down")), a)
        wd = (torch.randn(*L.down.shape, generator=g) * 0.02).to(torch.bfloat16).to(DEV)
        timed(f"[{i}] dense down matmul, bf16 weight", lambda w, a: F.linear(a, w), (wd, a))
        say(f"    dense: gate_up {tuple(L.gate_up.shape)} scale {tuple(L.gate_up_scale.shape)}; "
              f"down {tuple(L.down.shape)} scale {tuple(L.down_scale.shape)}", flush=True)


def groups(model, cfg, idx_runs, inp: dict, T: int, chain: int = 24) -> None:
    from kiln.engine.model_runner import IN_FLIGHT

    g = torch.Generator().manual_seed(2)
    h = torch.randn(T, cfg.hidden_size, generator=g).to(torch.bfloat16).to(DEV)
    for idxs in idx_runs:
        fn = model.group_fn(idxs)
        ts = tuple(t for i in idxs for t in model.layer_tensors(i))
        args = (h, inp["positions"], inp["slots"], inp["table"], inp["bias"], inp["table_w"], inp["bias_w"], *ts)
        kinds = "".join("D" if not model.layers[i].moe else ("w" if getattr(model.layers[i].spec, "window", None) else "F")
                        for i in idxs)
        name = f"group layers {idxs.start}-{idxs.stop - 1} [{kinds}]"
        timed(name, fn, args)
        c = torch.compile(fn, **OPTS)  # same graph: a compile-cache hit
        x, outs = h, []
        t = time.perf_counter()
        for _ in range(chain):
            x = c(x, *args[1:])
            outs.append(x)
            if len(outs) > IN_FLIGHT:
                outs[-1 - IN_FLIGHT].cpu()
        x.cpu()
        say(f"  {name:<46} chained {chain} calls: {(time.perf_counter() - t) / chain * 1e3:8.3f} ms per call",
              flush=True)


def step(model, cfg, B: int, P: int, ps: int, group_sizes, legacy=(), iters: int = 20, moe_groups=()) -> None:
    """The runner's piecewise decode step (model_runner._piecewise: prep graph, layer-group
    graphs with IN_FLIGHT waits, post graph with the sampler), host inputs already on the
    device, per group size. `legacy` group sizes are run with the grouping before MoE layers
    got a graph each (consecutive runs of the group size, whatever the layers are);
    `moe_groups` through _piecewise's moe_group (the NKI MoE kernel's grouping)."""
    import numpy as np

    from kiln.engine import model_runner as mr
    from kiln.engine.model_runner import _piecewise

    inp = decode_inputs(model, B, P, ps)
    d = lambda a: torch.from_numpy(a).to(DEV)  # noqa: E731
    z = np.zeros(B, np.int64)
    board = torch.zeros(2 * B + 8, dtype=torch.float32).to(DEV)
    ctx = d(np.full(B, P * ps, np.int64))
    args = [d(z), inp["positions"], inp["table"], ctx, inp["slots"], d(np.zeros(B, np.float32)),
            d(np.ones(B, np.float32)), d(z), d(np.zeros(B, np.float32)), d(np.full((B, 64), 0.5, np.float32)),
            board, d(np.full(B, -1, np.int64)), d(np.arange(B, dtype=np.int64)), None, None,
            inp["table_w"], swa_first(model, inp, B, P, ps)]
    current = mr.piecewise_groups
    runs = [(g, "now") for g in group_sizes] + [(g, "old") for g in legacy] + [(g, "moe") for g in moe_groups]
    for gs, how in runs:
        old = how == "old"
        mr.piecewise_groups = (lambda n, g, alone=(): current(n, g)) if old else current
        if how == "moe":
            decode, *_ = _piecewise(model, lambda f: torch.compile(f, **OPTS), None, moe_group=gs)
            n_graphs = len(current(len(model.layers), gs))
        else:
            decode, *_ = _piecewise(model, lambda f: torch.compile(f, **OPTS), gs)
            n_graphs = len(current(len(model.layers), gs, () if old else
                                   [i for i, l in enumerate(model.layers) if l.moe]))
        mr.piecewise_groups = current
        t = time.perf_counter()
        decode(*args).cpu()
        compile_s = time.perf_counter() - t
        decode(*args).cpu()
        ts = []
        for _ in range(iters):
            t = time.perf_counter()
            decode(*args).cpu()
            ts.append(time.perf_counter() - t)
        ts.sort()
        rule = {"old": "pre-fix grouping", "now": "MoE layers alone", "moe": "MoE layers grouped"}[how]
        if any(l.moe_blob for l in model.layers):
            rule += ", NKI MoE kernel"
        say(f"  decode step, {len(model.layers)} layers, group {gs} ({rule}, {n_graphs} layer graphs)   p50 {ts[iters // 2] * 1e3:8.3f} ms   "
            f"min {ts[0] * 1e3:8.3f}   (first call {compile_s:6.1f} s)", flush=True)


def swa_first(model, inp, B, P, ps):
    import numpy as np

    from kiln.engine.model_runner import ModelRunner

    shim = type("S", (), {"window": model.window, "ps": ps})()
    Pw = inp["table_w"].shape[-1]
    first = np.zeros(B, np.int64)
    pos = inp["positions"].cpu().numpy()
    table = inp["table"].cpu().numpy()
    for b in range(B):
        first[b] = ModelRunner._swa_row(shim, list(table[b]), int(pos[b]), Pw, np.zeros(Pw, np.int64))
    return torch.from_numpy(first).to(DEV)


def allreduce_rank(rank: int, port: int, T: int, H: int, n: int, world: int = 2) -> None:
    from kiln.engine import tp

    tp.neuron_env(rank, port, int(os.environ.get("KILN_CORE_BASE", "0")))
    tp.init_rank(rank, world, port)
    setup_device()
    import torch.distributed as dist
    import torch.distributed._functional_collectives as funcol

    grp = dist.group.WORLD
    x = torch.randn(T, H, generator=torch.Generator().manual_seed(rank)).to(torch.bfloat16).to(DEV)
    out = print if rank == 0 else (lambda *a, **k: None)

    def ar(x, reps):
        for _ in range(reps):
            x = funcol.all_reduce(x, "sum", grp) * 0.5
        return x

    def plain(x, reps):
        for _ in range(reps):
            x = (x + 1.0) * 0.5
        return x

    world_size = dist.get_world_size()
    pair = dist.new_group([0, 1])  # the two NeuronCores of one chip, inside the full world
    eight = dist.new_group(list(range(8))) if world_size >= 8 else None  # new_group refuses more ranks than the world

    def ag(x):  # all-gather the partial sums, reduce locally (an all-reduce built from a gather)
        return funcol.all_gather_tensor(x, 0, grp).view(world_size, *x.shape).sum(0) * 0.5

    def rs(x):
        return funcol.reduce_scatter_tensor(x.repeat(world_size, 1), "sum", 0, grp) * 0.5

    def a2a(x):
        return funcol.all_to_all_single(x.repeat(world_size, 1), None, None, grp)[: x.shape[0]] * 0.5

    cases = [("null graph", lambda x: x + 1.0), ("1 all-reduce", lambda x: ar(x, 1)),
             (f"{n} all-reduces", lambda x: ar(x, n)), (f"{n} adds (no all-reduce)", lambda x: plain(x, n))]
    if os.environ.get("KILN_PROBE_COLLECTIVES"):
        cases += [("all-gather + local sum", ag), ("reduce-scatter", rs), ("all-to-all", a2a)]
        if rank < 2:
            cases.append(("all-reduce, ranks 0-1 only", lambda x: funcol.all_reduce(x, "sum", pair) * 0.5))
        if rank < 8 and eight is not None:
            cases.append(("all-reduce, ranks 0-7 only", lambda x: funcol.all_reduce(x, "sum", eight) * 0.5))
        # Attention TP (models/decoder.py): EVERY group of G consecutive ranks all-reduces over itself
        # at once, through the engine's own groups (engine/tp.py attention_group). Each group's
        # sum is checked against the members' inputs, which every rank can regenerate.
        for G in (2, 8):
            if G < world_size and world_size % G == 0:
                ag_ = tp.attention_group(world_size, G)
                first = rank // G * G
                want = sum(torch.randn(T, H, generator=torch.Generator().manual_seed(m)).to(torch.bfloat16).float()
                           for m in range(first, first + G))
                got = torch.compile(lambda y, g=ag_: funcol.all_reduce(y, "sum", g), **OPTS)(x)  # compile flow only
                err = (got.cpu().float() - want).abs().max().item()
                if rank % G == 0:
                    print(f"  tp={world} attention groups of {G}: group {rank // G} sum max |err| {err:.3e}", flush=True)
                cases.append((f"all-reduce, all groups of {G}", lambda x, g=ag_: funcol.all_reduce(x, "sum", g) * 0.5))
    if os.environ.get("KILN_PROBE_A2A"):
        # Expert-parallel dispatch / combine shapes: every rank sends T = --batch rows of H to every
        # rank ([world T, H] in, [world T, H] out; T is the per-destination capacity), 1 and 12 per
        # graph, against the world all-reduce of the same [world T, H] (the gather / combine the
        # MoE block already pays under DP attention and sequence-parallel streams). The all-to-all's
        # result is checked: block s of rank r's output is rank s's block r.
        def a2a_n(y, n):
            z = y.repeat(world_size, 1)
            for _ in range(n):
                z = funcol.all_to_all_single(z, None, None, grp) * 0.5
            return z[: y.shape[0]]

        def ar_full(y, n):
            z = y.repeat(world_size, 1)
            for _ in range(n):
                z = funcol.all_reduce(z, "sum", grp) * (1.0 / world_size)
            return z[: y.shape[0]]

        # Column 0: the sender's rank; column 1: the destination block (both small integers, bf16-exact).
        dst = torch.arange(world_size * T) // T
        tag = torch.stack([torch.full_like(dst, rank), dst], 1).float()
        got = torch.compile(lambda y: funcol.all_to_all_single(y, None, None, grp), **OPTS)(
            tag.to(torch.bfloat16).to(DEV)).cpu().float()
        want = torch.stack([torch.arange(world_size * T) // T, torch.full_like(dst, rank)], 1).float()
        out(f"  tp={world} all-to-all [{world_size * T}, 2] block exchange max |err| {(got - want).abs().max().item():.3e}",
            flush=True)
        for n_ in (1, 12):
            cases += [(f"all-to-all [{world_size * T}] x{n_}", lambda y, n_=n_: a2a_n(y, n_)),
                      (f"all-reduce [{world_size * T}] x{n_}", lambda y, n_=n_: ar_full(y, n_))]
    nsp = int(os.environ.get("KILN_PROBE_SP", "0"))
    if nsp > 1 and world_size % nsp == 0:
        # A sequence-parallel residual stream under DP attention with nsp groups: every rank holds T
        # rows of its own; a layer all-gathers its group's rows for the token mixer, reduce-scatters
        # the mixer's head partials back over the group, all-gathers every rank's rows for the MoE
        # and reduce-scatters its output over the world. Against the replicated stream's two world
        # all-reduces of the whole [world T, H] batch per layer (the token mixer's zero-padded one,
        # DecoderForCausalLM._attn_all_reduce, and the MLP's). 1 and 12 layers' exchanges per graph.
        atp = world_size // nsp
        sub = tp.attention_group(world_size, atp) if atp > 1 else None

        def ag_g(y):
            return funcol.all_gather_tensor(y.contiguous(), 0, sub) if sub is not None else y

        def rs_g(y):
            return funcol.reduce_scatter_tensor(y, "sum", 0, sub) if sub is not None else y

        def ag_w(y):
            return funcol.all_gather_tensor(y.contiguous(), 0, grp)

        def rs_w(y):
            return funcol.reduce_scatter_tensor(y, "sum", 0, grp)

        def sp_layers(y, n):
            for _ in range(n):
                y = rs_g(ag_g(y) * 0.5)
                y = rs_w(ag_w(y) * 0.5)
            return y

        def rep_layers(y, n):
            z = y.repeat(world_size, 1)
            for _ in range(n):
                z = funcol.all_reduce(z * 0.5, "sum", grp)
                z = funcol.all_reduce(z * 0.5, "sum", grp)
            return z[: y.shape[0]]

        got = torch.compile(lambda y: rs_w(ag_w(y)), **OPTS)(x).cpu().float()
        want = x.cpu().float() * world_size
        out(f"  tp={world} sp: world all-gather + reduce-scatter of [{T}, {H}] rows, max |err| "
            f"{(got - want).abs().max().item():.3e} against world x the rows", flush=True)
        cases += [(f"sp group all-gather [{T}->{atp * T}]", lambda y: ag_g(y)[: y.shape[0]] * 0.5),
                  (f"sp group reduce-scatter [{atp * T}->{T}]", lambda y: rs_g(y.repeat(atp, 1)) * 0.5),
                  (f"sp world all-gather [{T}->{world_size * T}]", lambda y: ag_w(y)[: y.shape[0]] * 0.5),
                  (f"sp world reduce-scatter [{world_size * T}->{T}]", lambda y: rs_w(y.repeat(world_size, 1)) * 0.5),
                  (f"world all-reduce [{world_size * T}]",
                   lambda y: funcol.all_reduce(y.repeat(world_size, 1), "sum", grp)[: y.shape[0]] * 0.5)]
        for n in (1, 12):
            cases += [(f"sp exchange, {n} layers", lambda y, n=n: sp_layers(y, n) * 0.5),
                      (f"replicated exchange, {n} layers", lambda y, n=n: rep_layers(y, n) * 0.5)]
    ndp = int(os.environ.get("KILN_PROBE_DP", "0"))
    if ndp > 1 and world_size % ndp == 0:
        atp = world_size // ndp
        g, a = rank // atp, rank % atp
        sub = tp.attention_group(world_size, atp) if atp > 1 else None
        onehot = (torch.arange(ndp) == g).to(torch.bfloat16).to(DEV)
        # Every rank's partial is its seed's normals; group g's rows of the result are its members' sum.
        parts = [torch.randn(T, H, generator=torch.Generator().manual_seed(m)).to(torch.bfloat16).float()
                 for m in range(world_size)]
        want = torch.cat([sum(parts[q * atp : (q + 1) * atp]) for q in range(ndp)])

        def dp_ar(y):  # Kiln: one world all-reduce of the zero-padded rows
            return funcol.all_reduce((y.unsqueeze(0) * onehot.view(ndp, 1, 1)).reshape(-1, H), "sum", grp)

        def dp_rs_ag(y):  # SGLang MAX_LEN: group reduce-scatter, then a world all-gather
            if sub is not None:
                y = funcol.reduce_scatter_tensor(y, "sum", 0, sub)
            return funcol.all_gather_tensor(y.contiguous(), 0, grp)

        def dp_ar_ag(y):  # group all-reduce, then a world all-gather of attn_tp copies
            if sub is not None:
                y = funcol.all_reduce(y, "sum", sub)
            return funcol.all_gather_tensor(y.contiguous(), 0, grp).view(ndp, atp, T, H)[:, 0].reshape(-1, H)

        for nm, f in (("dp all-reduce (Kiln)", dp_ar), ("dp reduce-scatter + all-gather", dp_rs_ag),
                      ("dp all-reduce + all-gather", dp_ar_ag)):
            if nm.startswith("dp reduce-scatter") and T % atp:
                continue
            got = torch.compile(f, **OPTS)(x).cpu().float()
            err = (got - want).abs().max().item()
            out(f"  tp={world} dp={ndp} {nm}: max |err| {err:.3e} against the group sums", flush=True)
            # Timed as [T, H] -> [T, H] so that launches chain: the first T rows of the result, halved.
            cases.append((f"{nm} dp={ndp}", lambda y, f=f: f(y)[:T] * 0.5))
    for name, f in cases:
        c = torch.compile(f, **OPTS)
        c(x).cpu()
        c(x).cpu()
        ts = []
        for _ in range(30):
            t = time.perf_counter()
            c(x).cpu()
            ts.append(time.perf_counter() - t)
        ts.sort()
        out(f"  tp={world} {name:<28} [{T}, {H}] bf16  p50 {ts[15] * 1e3:8.3f} ms  min {ts[0] * 1e3:8.3f}", flush=True)
        # As the runner issues a decode step: 48 launches back to back, waiting on the output
        # IN_FLIGHT (6) launches behind, one readback at the end. Per-launch cost.
        ts = []
        for _ in range(10):
            t = time.perf_counter()
            y, outs = x, []
            for _ in range(48):
                y = c(y)
                outs.append(y)
                if len(outs) > 6:
                    _sync(outs.pop(0))
            y.cpu()
            ts.append(time.perf_counter() - t)
        ts.sort()
        out(f"  tp={world} {name:<28} 48 chained: {ts[5] * 1e3 / 48:8.3f} ms per launch", flush=True)
    dist.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="XiaomiMiMo/MiMo-V2.6-Flash-RL")
    ap.add_argument("--tp", type=int, default=32, help="tensor-parallel degree whose rank-0 shapes are built")
    ap.add_argument("--batch", type=int, default=4, help="decode sequences")
    ap.add_argument("--prefill", type=int, default=0, help="time a C-token prefill chunk instead of decode")
    ap.add_argument("--verify", type=int, default=0,
                    help="time the speculative verify form instead of decode: --batch sequences x Q positions")
    ap.add_argument("--pages", type=int, default=16, help="page bucket of the full-attention table")
    ap.add_argument("--prefill-offset", type=int, default=0,
                    help="prefill: the chunk's first position (the KV of the positions before it is whatever the cache "
                         "holds; positions + chunk must fit the page bucket)")
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--layers", type=int, default=12, help="layers built (0..n-1)")
    ap.add_argument("--group", type=int, default=6, help="layers per piecewise graph")
    ap.add_argument("--what", nargs="+", default=["parts", "layers", "groups"],
                    help="parts layers groups step mlp laparts hcblocks (mlp: the first MoE layer's MLP block alone; "
                         "laparts: a linear-attention layer's pieces; hcblocks: a hyper-connection layer's blocks at "
                         "the DP-attention row count)")
    ap.add_argument("--step-groups", type=int, nargs="*", default=[6, 1],
                    help="layers per graph for --what step (the runner's piecewise decode)")
    ap.add_argument("--legacy-groups", type=int, nargs="*", default=[],
                    help="also time --what step with the pre-fix grouping at these group sizes")
    ap.add_argument("--moe-groups", type=int, nargs="*", default=[],
                    help="also time --what step with MoE layers grouped (needs KILN_MOE_KERNEL=nki)")
    ap.add_argument("--part-layers", type=int, nargs="+", default=None,
                    help="layers whose parts (and, with --what layers, the layer) are timed")
    ap.add_argument("--allreduce", action="store_true")
    ap.add_argument("--ranks", type=int, default=1,
                    help="processes (one NeuronCore each) whose in-graph all-reduces are real (2 on trn1.2xlarge)")
    ap.add_argument("--cpu", action="store_true", help="dry run on the host (eager), to check the script")
    ap.add_argument("--attn-tp", type=int, default=None,
                    help="attention TP whose rank-0 shapes the token mixers get (default: the model's for --tp; "
                         "GLM-5.3-Flash at --tp 32 --dp-attention 4 runs 8)")
    ap.add_argument("--neff", action="store_true", help="print each compiled graph's instruction counts")
    ap.add_argument("--layer-groups", nargs="*", default=[],
                    help="prefill: also time these layer runs as one graph each, e.g. 0-1 2-3 4-5 (L/A: linear "
                         "attention or attention, m/d: MoE or dense)")
    ap.add_argument("--dp-attention", type=int, default=1,
                    help="prefill, --what layers: DP-attention groups (DecoderForCausalLM dp_attention); the "
                         "hidden state holds --prefill rows per group, the token mixers run rank 0's group's")
    ap.add_argument("--la-compare", action="store_true",
                    help="layers: linear-attention layers also through KILN_LINEAR_ATTN_KERNEL's other setting, "
                         "timed, and the two outputs compared")
    ap.add_argument("--scratch-rows", type=int, default=None,
                    help="rows of the DSA selection scratch (default: the call's rows, as ModelRunner sizes it)")
    ap.add_argument("--sp-only", action="store_true",
                    help="--layer-groups: only the sequence-parallel form (KILN_PROFILE_INSPECT=<dir>: rank 0's runtime "
                         "device profile; KILN_PROFILE_ITERS: timed iterations)")
    ap.add_argument("--sp", action="store_true",
                    help="--layer-groups: also time each group with sequence-parallel prefill streams (KILN_PREFILL_SP; "
                         "needs --ranks > 1: every rank holds R / ranks rows)")
    ap.add_argument("--sum-readback", action="store_true",
                    help="layers / mlp: read back out.float().sum() instead of the whole output (a hybrid "
                         "model's [C, hc * H] prefill output takes longer over PCIe than the layer runs); the "
                         "null graph is then timed the same way")
    args = ap.parse_args()

    if args.allreduce:
        import multiprocessing as mp

        from kiln.engine import tp

        port = tp.free_port()
        ctx = mp.get_context("spawn")
        world = max(args.ranks, 2)
        ps = [ctx.Process(target=allreduce_rank, args=(r, port, args.batch, 4096, 12, world)) for r in range(world)]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        return

    if args.ranks > 1:
        import multiprocessing as mp

        from kiln.engine import tp

        port = tp.free_port()
        ctx = mp.get_context("spawn")
        ps = [ctx.Process(target=run, args=(args, r, port)) for r in range(args.ranks)]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        return
    run(args)


def run(args, rank: int = 0, port: int = 0) -> None:
    global RANK
    RANK = rank
    insp = os.environ.get("KILN_PROFILE_INSPECT")
    if insp and rank == 0:  # a runtime device profile of every graph rank 0 runs (neuron-explorer reads it)
        os.makedirs(insp, exist_ok=True)
        os.environ.update(NEURON_RT_INSPECT_ENABLE="1", NEURON_RT_INSPECT_DEVICE_PROFILE="1",
                          NEURON_RT_INSPECT_OUTPUT_DIR=insp)
    if args.ranks > 1:
        from kiln.engine import tp

        tp.neuron_env(rank, port, int(os.environ.get("KILN_CORE_BASE", "0")))
        tp.init_rank(rank, args.ranks, port)
    setup_device(args.cpu)
    B, P, ps = args.batch, args.pages, args.page_size
    Qv = args.verify
    T = args.prefill or B * (Qv or 1)
    num_pages = 1 + (P if args.prefill else B * P)
    global NEFF
    NEFF = args.neff
    # The DSA selection scratch as ModelRunner sizes it (the most query rows one call carries).
    cfg, model = build(args.model, args.tp, args.layers, num_pages, ps, P * ps, args.ranks,
                       max_rows=args.scratch_rows or T, attn_tp=args.attn_tp, dp_attention=args.dp_attention)
    R = T * args.dp_attention  # rows of the hidden state: every DP-attention group's (the mixers take T)
    if args.sp or args.sp_only:  # sequence-parallel streams: each live rank its own row block (decoder.dp_buffers' sp buffers)
        if args.ranks < 2 or not model.prefill_sp or R % args.ranks:  # (--sp-only implies --sp)
            raise SystemExit(f"--sp needs --ranks > 1 dividing {R} rows and a model with sequence-parallel prefill streams")
        model.tp_rank = rank
        if args.dp_attention > 1 and os.environ.get("KILN_PROFILE_REAL_GROUPS", "1") == "1":
            # the engine's attention groups (ranks g * atp .. (g + 1) * atp - 1), so that group collectives
            # (models/hybrid.py KILN_SP_GROUP) are the real ones; every rank still runs group 0's inputs
            from kiln.engine import tp as _tp

            atp = args.ranks // args.dp_attention
            model.attn_tp, model.attn_group, model.attn_rank = atp, _tp.attention_group(args.ranks, atp), rank % atp
        model.dp_buffers(DEV)
    if args.prefill:
        inp = prefill_inputs(model, T, P, ps, args.prefill_offset)
    else:
        inp = verify_inputs(model, B, Qv, P, ps) if Qv else decode_inputs(model, B, P, ps)
    form = f"prefill C={T}" if args.prefill else (f"verify B={B} x Q={Qv}" if Qv else f"decode B={B}")
    who = "rank 0 alone" if args.ranks == 1 else f"rank 0 of {args.ranks} live ranks (real all-reduces)"
    say(f"{args.model} tp={args.tp} {who}, {form}, P={P} x {ps} (table {tuple(inp['table'].shape)}, "
          f"window table {tuple(inp['table_w'].shape)}), attention TP {args.attn_tp or args.tp // args.dp_attention}"
        f"{f', DP attention {args.dp_attention} ({R} rows)' if args.dp_attention > 1 else ''}", flush=True)
    red = (lambda f: (lambda *a: f(*a).float().sum())) if args.sum_readback else (lambda f: f)
    width = cfg.hidden_size * (cfg.hybrid.hc if cfg.hybrid is not None else 1)
    timed("null graph (launch + readback)", red(lambda h: h + 1),
          (torch.zeros(T, width if args.sum_readback else cfg.hidden_size, dtype=torch.bfloat16).to(DEV),))

    from kiln.config import LinearSpec

    kinds = {}
    for i, layer in enumerate(model.layers):

        mixer = "linear" if isinstance(layer.spec, LinearSpec) else ("swa" if getattr(layer.spec, "window", None) else "full")
        kinds.setdefault(("dense" if not layer.moe else "moe") + " " + mixer, i)
    if "mlp" in args.what:  # the MoE block alone: experts, shared expert, all-reduce (hybrid-aware)
        from kiln.models import hybrid as _hy

        i = next(j for j, layer in enumerate(model.layers) if layer.moe)
        body = (lambda V, x: _hy._mlp(model, V, x)) if cfg.hybrid is not None else (lambda V, x: model._mlp(V, x))
        f, ts = with_layer(model, i, body)
        x = (torch.randn(R, cfg.hidden_size, generator=torch.Generator().manual_seed(3)) * 0.5).to(torch.bfloat16)
        timed(f"layer {i} MoE block (experts + shared expert + all-reduce)", red(f), (x.to(DEV), *ts))
    ss = None
    if any(isinstance(layer.spec, LinearSpec) for layer in model.layers):
        from kiln.engine.state_pool import StatePool

        if args.prefill:  # the chunk form: state row 1 (zero state at position 0)
            StatePool(model, 2, DEV, torch.bfloat16)
            ss = torch.tensor([1]).to(DEV)
        elif Qv:  # the verify form: Q rows per sequence, read the first, the state after each position into all Q
            StatePool(model, B, DEV, torch.bfloat16, rows_per_req=Qv)
            rows = torch.arange(1, 1 + B * Qv).view(B, Qv)
            ss = torch.cat([rows[:, :1], rows], dim=1).to(DEV)
        else:  # the decode form: one state row per sequence (rows 1 .. B)
            StatePool(model, B + 1, DEV, torch.bfloat16)
            ss = torch.arange(1, B + 1).to(DEV)
    if "hcblocks" in args.what:
        for label, i in kinds.items():
            if args.part_layers is None or i in args.part_layers:
                hc_blocks(model, cfg, i, inp, T, R, ss)
    if "laparts" in args.what:
        for label, i in kinds.items():
            if "linear" in label and (args.part_layers is None or i in args.part_layers):
                linear_parts(model, cfg, i, inp, T, ss, R)
    if "parts" in args.what:
        for label, i in kinds.items():
            if args.part_layers is None or i in args.part_layers:
                parts(model, cfg, i, inp, T, label)
    if "layers" in args.what:
        # Hyper-connection models (models/hybrid.py) carry hc residual streams per token.
        from kiln.models import linear_attn as la

        for label, i in kinds.items():
            if args.part_layers is not None and i not in args.part_layers:
                continue
            ts = model.layer_tensors(i)
            x = torch.randn(R, width, generator=torch.Generator().manual_seed(i)).to(torch.bfloat16).to(DEV)
            a = (x, inp["positions"], inp["slots"], inp["table"], inp["bias"], inp["table_w"], inp["bias_w"], *ts)
            settings = [la.LINEAR_ATTN_KERNEL]
            if args.la_compare and "linear" in label:
                settings.append("torch" if la.LINEAR_ATTN_KERNEL == "nki" else "nki")
            outs = []
            for st in settings:
                la.LINEAR_ATTN_KERNEL = st
                torch._dynamo.reset()
                f = model.group_fn([i])
                if "linear" in label:
                    f = _with_state_slot(f, ss)
                tag = f" KILN_LINEAR_ATTN_KERNEL={st}" if len(settings) > 1 else ""
                timed(f"layer {i} alone ({label}){tag}", red(f), a)
                if len(settings) > 1:
                    outs.append(torch.compile(f, **OPTS)(*a).cpu().float())
            la.LINEAR_ATTN_KERNEL = settings[0]
            if len(outs) == 2:
                ref = outs[1] if settings[1] == "torch" else outs[0]
                d = (outs[0] - outs[1]).abs()
                say(f"  {'':<46} layer {i} output, kernel vs torch path: max abs diff {d.max().item():.3e}, "
                    f"relative to the output's max {d.max().item() / ref.abs().max().item():.3e}, "
                    f"mean abs diff {d.mean().item():.3e}", flush=True)
    for spec in args.layer_groups:  # prefill pieces as the runner groups them (KILN_PIECEWISE_PREFILL_MOE_GROUP)
        a_, b_ = (int(v) for v in spec.split("-"))
        idxs = range(a_, b_ + 1)
        f = model.group_fn(idxs)
        if any(isinstance(model.layers[j].spec, LinearSpec) for j in idxs):
            f = _with_state_slot(f, ss)
        ts = tuple(t for j in idxs for t in model.layer_tensors(j))
        kinds = "".join(("L" if isinstance(model.layers[j].spec, LinearSpec) else "A") + ("m" if model.layers[j].moe else "d")
                        for j in idxs)
        forms = ([] if args.sp_only else [False]) + ([True] if args.sp or args.sp_only else [])
        for sp in forms:  # --sp: also with sequence-parallel streams (models/decoder.py prefill_sp_enabled), R / ranks rows
            rows = R // model.tp_size if sp else R
            x = torch.randn(rows, width, generator=torch.Generator().manual_seed(a_)).to(torch.bfloat16).to(DEV)
            tag = f" sequence-parallel streams ({rows} rows)" if sp else ""
            timed(f"group layers {a_}-{b_} [{kinds}]{tag}", red(f), (x, inp["positions"], inp["slots"], inp["table"],
                                                                    inp["bias"], inp["table_w"], inp["bias_w"], *ts))
    if "groups" in args.what:
        from kiln.engine.model_runner import piecewise_groups

        groups(model, cfg, piecewise_groups(args.layers, args.group), inp, T)
    if "step" in args.what:
        step(model, cfg, B, P, ps, args.step_groups, args.legacy_groups, moe_groups=args.moe_groups)


if __name__ == "__main__":
    main()
