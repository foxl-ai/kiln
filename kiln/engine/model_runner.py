"""Pads scheduled work into static-shape buckets and runs the model graphs.

Every call is padded to a bucket: decode to (B sequences, P pages), prefill to
(C tokens, P pages). The page axis is sized to the longest context in the call, not to
max_model_len, so a step reads only the KV it can use. Padded rows write their KV into
the null page and their outputs are discarded.
"""

from __future__ import annotations

import inspect
import os
import time
import zlib
from collections import defaultdict

import numpy as np
import torch

from ..config import EngineConfig, ModelConfig, _pow2_ladder, pick_bucket
from .kv_pool import NULL_PAGE
from .request import Request
from .scheduler import PENDING, ScheduledSeq

# Stands for this rank's device-resident token board in host_args. A string, because the
# args are pickled to the other tensor-parallel ranks and an object() would not survive.
BOARD = "__kiln_board__"
# Prefix of a host-arg string naming a device tensor an earlier graph returned on THIS rank
# (normalised hidden states for MTP drafting). Every rank runs the same calls in the same
# order, so the same counter names the same tensor everywhere.
HIDDEN = "__kiln_hidden__:"
# Hidden states kept: a step drafts only from its own calls' (at most a decode, a verify and a prefill per
# bucket), and each is HBM (GLM-5.3-Flash: a verify's 64 x 3 rows of 4 x 4096 bf16 streams are 6.3 MB per rank).
HIDDEN_KEEP = 8
# Prefix of a host-arg string naming one of the model's parameters (e.g. a norm weight).
PARAM = "__kiln_param__:"


class PerGroup(list):
    """A host argument that differs per DP-attention group (engine/dp.py): element g is group g's
    array, all of one shape; each rank executes with its own group's (ModelRunner._dev). A list
    subclass so that it pickles to the other tensor-parallel ranks."""


def kv_cache_torch_dtype(ecfg: EngineConfig) -> torch.dtype:
    if ecfg.kv_cache_dtype == "auto":
        return ecfg.dtype
    if ecfg.kv_cache_dtype in ("fp8", "fp8_e4m3"):
        return torch.float8_e4m3fn
    raise ValueError(f"kv_cache_dtype must be auto or fp8, not {ecfg.kv_cache_dtype!r}")


def fp8_e4m3_max(device: torch.device) -> float:
    """Largest finite FP8 E4M3 value on the device (kiln/platform.py, fp8_max: 240 on
    trn1/trn2, 448 on trn3). On CPU torch.float8_e4m3fn is the fn variant (448)."""
    # meta: the capture of a device run (kiln/capture.py), which must trace the device's value.
    if device.type not in ("neuron", "meta"):
        return float(torch.finfo(torch.float8_e4m3fn).max)
    from .. import platform

    return platform.fp8_max(platform.runtime_target())


def neuronx_cc_args(kv_dtype: torch.dtype, fp8_weights: bool = False) -> list[str]:
    """Extra neuronx-cc arguments for every graph.

    trn1/trn2 FP8 is e4m3 with inf (max 240) and torch has no such dtype, so an fp8 KV
    cache or FP8 weights are float8_e4m3fn in the graph, which neuronx-cc rejects there: "[NCC_EVRF051]
    Data type F8E4M3FN is not supported on TRN1/TRN2 ... use the
    --experimental-unsafe-fp8e4m3fn-as-fp8e4m3 flag" (measured on trn1, SDK 2.32). The values
    are clamped to 240 on write, so the cast is exact. libtorch_neuronx_lite injects the flag
    only for trn2 (compile/backend.py, _apply_platform_compiler_args), not trn1. On trn2 /
    trn3 kiln/platform.py adds --logical-nc-config and vllm-neuron's argument set.
    KILN_CC_ARGS appends arguments for experiments (e.g. "-O1").
    """
    from .. import platform

    args = platform.neuronx_cc_args(platform.runtime_target(), fp8=kv_dtype == torch.float8_e4m3fn or fp8_weights)
    extra = os.environ.get("KILN_CC_ARGS")
    if extra:
        args += extra.split()
    return args


def canonicalize(gm) -> None:
    """Drop keyword arguments equal to their defaults from every torch call in `gm`, so
    equivalent traces print identically (canonical_neuron_backend)."""
    from torch.fx import Node
    from torch.fx.operator_schemas import get_signature_for_torch_op

    def strip_defaults(node) -> None:
        if not node.kwargs:
            return
        sigs = get_signature_for_torch_op(node.target) or []
        if not sigs:
            return
        kept = {}
        for k, v in node.kwargs.items():
            # Defaults of the overloads that declare one for k (an overload may take k
            # without a default, e.g. a required out=); drop v only if all of them agree.
            declared = [sg.parameters[k] for sg in sigs if k in sg.parameters]
            ds = [p.default for p in declared if p.default is not inspect.Parameter.empty]
            is_default = not isinstance(v, Node) and (
                (bool(ds) and all(type(d) is type(v) and d == v for d in ds))
                # e.g. topk(..., out=None): no schema declares out (its out variant takes
                # values= / indices=), and None means "no output buffer"
                or (not declared and v is None))
            if not is_default:
                kept[k] = v
        node.kwargs = kept

    for node in gm.graph.nodes:
        if node.op == "call_function" and not isinstance(node.target, str):
            try:
                strip_defaults(node)
            except Exception:  # an op without an inspectable schema: leave as traced
                pass
    gm.recompile()


def canonical_neuron_backend():
    """libtorch_neuronx_lite's backend behind a pass that drops keyword arguments equal to
    their defaults from every torch call, so equivalent traces print identically.

    The compile cache key hashes str(graph) (libtorch_neuronx_lite compile/cache.py,
    create_cache_hash). Tensor-parallel rank 0 and the other ranks traced the same model to
    graphs that differed only in spelling: rank 0 recorded torch.topk(x, 64, dim=-1), rank 1
    torch.topk(x, 64, dim=-1, largest=True, sorted=True, out=None). tools/
    probe_fx_normalisation.py pins the trigger on trn1: a process that imported transformers
    and then spawned a child records the short form, every other process the long one. So
    every rank compiled its own copy, and a 32-rank model would run 32 neuronx-cc at once.
    Stripping defaults maps both forms to the short one: one key, one compile, the other
    ranks wait on the cache lock and load it. kiln/capture.py applies the same pass, so a
    graph captured on a CPU host gets the key the device run computes.
    """
    import torch._dynamo as dynamo

    inner = dynamo.lookup_backend("neuron_libtorch")
    # KILN_COMPILE_FARM=<queue uri>: a graph the compile farm holds is waited for and fetched instead
    # of compiled here (kiln/compile_cache.py FarmWait, tools/compile_farm.py).
    farm = None
    if os.environ.get("KILN_COMPILE_FARM"):
        from .. import compile_cache

        farm = compile_cache.FarmWait(os.environ["KILN_COMPILE_FARM"])

    def backend(gm, example_inputs, **kwargs):
        canonicalize(gm)
        if farm is not None:
            key = compile_cache.cache_key(gm, example_inputs, kwargs.get("options", {}))
            how = farm.before_compile(key)
            if how not in ("local", "not-farmed"):
                print(f"kiln compile farm: {key} {how}"
                      + (f" after {farm.waited[key]:.0f}s" if key in farm.waited else ""), flush=True)
        return inner(gm, example_inputs, **kwargs)

    backend.__name__ = "kiln_canonical_neuron_libtorch"
    return backend


def piecewise_groups(n_layers: int, group: int | None, alone=()) -> list[range]:
    """Consecutive runs of up to `group` layers; every layer in `alone` (the MoE layers) is a
    run of its own. The default keeps a dense model's step at about 14 layer launches: the
    runtime's execution queue holds at most 63 requests ("Setting
    NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS > 63 is not supported", libnrt, SDK 2.32) and
    one launch per layer of Qwen3-0.6B (30 per step) already filled it on trn1 before the
    IN_FLIGHT waits below bounded the launches queued at once.

    MoE layers get a graph each because neuronx-cc (2.27, SDK 2.32) lowers the in-graph
    expert-weight gathers (DecoderForCausalLM._moe) catastrophically once two attention + MoE
    layers share a graph: trn1 generates every dynamic DMA packet in software, and the compiler
    splits the gathers into millions of 34-byte packets (one bf16 element per DMA for the expert
    scales) and spills the pf-transposed dequantized gate_up block through HBM. Measured on
    MiMo-V2.6-Flash's tp=32 rank shapes (trn1.2xlarge, decode B=4, tools/probe_lowering.py,
    2026-10-03): 1 layer per graph 1.5 ms and 125 dynamic-access DMAs, 2 layers 39.6 ms /
    16,624, 3 layers 100 ms / 49,499, 6 layers 225 ms / 104,072. The full decode step, 48
    layers with real all-reduces over two NeuronCores (tools/profile_layer.py --ranks 2 --what
    step --legacy-groups 6, B=4, P=4): 550.4 ms at 6 layers per graph (571 ms measured on
    trn1.32xlarge at tp=32) against 55.3 ms. B=1 is the exception, 10% slower (45.5 against
    50.7 ms for 12 layers), and slow either way; see docs/neuron-notes.md. With the NKI MoE
    kernel (kiln/kernels/moe_decode.py) MoE layers group again: _piecewise's moe_group."""
    g = group or -(-n_layers // 14)
    alone = set(alone)
    runs: list[range] = []
    i = 0
    while i < n_layers:
        j = i + 1
        if i not in alone:
            while j < min(i + g, n_layers) and j not in alone:
                j += 1
        runs.append(range(i, j))
        i = j
    return runs


# Layer-graph launches allowed in flight before the host waits. Back-to-back launches of
# whole-model decode graphs failed at 16 with "Execution Queue Full" (tools/profile_decode.py)
# and Qwen3-0.6B's prefill layer graphs (14 per step) failed the same way, while 200 launches
# of a light graph did not (tools/debug_device.py queue): the queue fills when the device is
# the bottleneck, so the bound is in launches, measured at 8 being safe.
IN_FLIGHT = 6
PROFILE = os.environ.get("KILN_PROFILE_PIECES") == "1"
PIECE_TIMES: dict = defaultdict(list)  # (hidden shape, group index) -> seconds, when PROFILE
# KILN_DUMP_PIECES=<dir>: torch.save every piece's input and output hidden state of the first KILN_DUMP_CALLS (1)
# multi-piece calls, per rank, as <dir>/r<rank>-c<call>-p<piece>-{in,out}.pt (debugging: compare two trees or settings
# piece by piece on the device's own values). Host-side only: the graphs and their keys are unchanged.
DUMP_PIECES = os.environ.get("KILN_DUMP_PIECES")
DUMP_CALLS = int(os.environ.get("KILN_DUMP_CALLS", "1"))
_DUMPED = [0]
# KILN_PROFILE_EXEC=1: every ModelRunner._exec call timed synchronously, from the rank-0
# broadcast to the output read back, by compile key.
PROFILE_EXEC = os.environ.get("KILN_PROFILE_EXEC") == "1"
EXEC_TIMES: dict = defaultdict(list)


def _piecewise(model, compile_, group: int | None = None, moe_group: int | None = None,
               prefill_moe_group: int | None = None):
    """forward_decode / forward_prefill / forward_extend as prep graph -> one graph per run of
    layers (shared by every run with the same kinds) -> post graph, with the same signatures.
    Tables are chosen on the host exactly as DecoderForCausalLM._attn_inputs does; the
    compiled prep returns only the biases.

    moe_group: when the MoE layers run the NKI kernel (DecoderLayer.moe_blob), runs of up to
    moe_group layers of any kind; the kernel's expert loads are one DMA per pair whatever the
    graph holds, which is what made MoE layers need graphs of their own.

    prefill_moe_group: the same for prefill chunks (default moe_group). A prefill step is
    compute-bound, so the ~5 ms fixed cost of a cross-chip collective per graph execution that
    makes decode want few graphs hardly matters there, while a big chunk times 12 MoE layers in
    one graph does not compile: GLM-5.3-Flash prefill groups of 12 all-MoE layers at 2048 rows
    failed neuronx-cc with NCC_EBVF030 "Instructions generated by compiler 39723713 exceeds the
    typical limit of 5000000" (captured on a CPU box and compiled there, 2026-10-03)."""
    import inspect

    kinds: dict[tuple, object] = {}
    blob = any(getattr(l, "moe_blob", False) for l in model.layers)

    def make_plan(mg):
        if mg and blob:
            runs = piecewise_groups(len(model.layers), mg)
        else:
            runs = piecewise_groups(len(model.layers), group, [i for i, l in enumerate(model.layers) if l.moe])
        out = []
        for idxs in runs:
            key = tuple(model.layer_kind(i) for i in idxs)
            if key not in kinds:
                kinds[key] = compile_(model.group_fn(idxs))
            out.append((kinds[key], tuple(t for i in idxs for t in model.layer_tensors(i))))
        return out

    plan = make_plan(moe_group)
    plan_p = plan if (prefill_moe_group or moe_group) == moe_group else make_plan(prefill_moe_group)
    windows = [None] + ([model.window] if model.window is not None else [])
    seen: set = set()
    last: list = [None]

    def run(fn, *args, **kw):
        """Launch one piece. The FIRST launch of a piece at a given shape waits for everything
        before it and is waited for itself: when every graph came from the compile cache, the
        ranks ran ahead of each other and the first post graph failed on all 32 ranks with
        "replica group signature mismatch for group 0: likely caused by mismatched collectives
        between peers" (MiMo-V2.6-Flash, trn1.32xlarge, SDK 2.32, 2026-10-03); a compile in the
        same process lines the ranks up behind its lock, which is why it never showed there."""
        key = (id(fn), tuple(tuple(x.shape) if isinstance(x, torch.Tensor) else x is None for x in args[:7]))
        first = key not in seen
        if first and last[0] is not None and last[0].device.type == "neuron":
            last[0].cpu()
        out = fn(*args, **kw)
        t = out[0] if isinstance(out, tuple) else out
        if first:
            if t.device.type == "neuron":
                t.cpu()
            seen.add(key)
        last[0] = t
        return out

    def with_h(out, h):  # an MTP head drafts from the final hidden states (forward_mtp)
        return (out, h) if model.mtp is not None else out

    def flat(prep):
        def f(*a):
            h, attn = prep(*a)
            return (h, *(attn[w][0] for w in windows))
        return f

    def layers(h, positions, slot_mapping, biases, block_table, swa_table, state_slot=None, plan=plan, mixed=None):
        table_w = swa_table if swa_table is not None else block_table
        bias_w = biases.get(model.window) if model.window is not None else None
        kw = {"state_slot": state_slot} if state_slot is not None else {}  # linear-attention layers
        if mixed is not None:  # a mixed batch's decode rows (DecoderForCausalLM.forward_mixed)
            kw["mixed"] = mixed
        outs = []
        dump = DUMP_PIECES and len(plan) > 1 and _DUMPED[0] < DUMP_CALLS
        if dump:
            os.makedirs(DUMP_PIECES, exist_ok=True)
            tag = f"{DUMP_PIECES}/r{model.tp_rank}-c{_DUMPED[0]}"
            _DUMPED[0] += 1
        for gi, (fn, tensors) in enumerate(plan):
            t0 = time.perf_counter() if PROFILE else 0.0
            if dump:
                torch.save(h.cpu(), f"{tag}-p{gi}-in.pt")
            h = run(fn, h, positions, slot_mapping, block_table, biases[None], table_w, bias_w, *tensors, **kw)
            if dump:
                torch.save(h.cpu(), f"{tag}-p{gi}-out.pt")
            if PROFILE and h.device.type == "neuron":  # KILN_PROFILE_PIECES=1: time each group, synchronously
                h.cpu()
                PIECE_TIMES[(tuple(h.shape), gi)].append(time.perf_counter() - t0)
            outs.append(h)
            if len(outs) > IN_FLIGHT and h.device.type == "neuron":
                # libtorch_neuronx_lite has no synchronize; reading an earlier output back
                # waits for it, which bounds the launches queued behind it.
                outs[-1 - IN_FLIGHT].cpu()
        return h

    prep_d, post_d = compile_(flat(model.prep_decode)), compile_(model.post_decode)
    prep_p, post_p = compile_(flat(model.prep_prefill)), compile_(model.post_prefill)
    prep_e, post_e = compile_(flat(model.prep_verify)), compile_(model.post_extend)

    def flat_m(*a):
        h, attn, dattn = model.prep_mixed(*a)
        return h, attn[None][0], dattn[None][0]

    prep_m, post_m = compile_(flat_m), compile_(model.post_mixed)
    sig = {n: inspect.signature(getattr(model, n))
           for n in ("forward_decode", "forward_prefill", "forward_extend", "forward_mixed")}

    def decode(*args):
        a = sig["forward_decode"].bind(*args)
        a.apply_defaults()
        a = a.arguments
        h, *b = run(prep_d, a["input_ids"], a["positions"], a["block_table"], a["context_lens"], a["board"],
                       a["read_slot"], a["swa_table"], a["swa_first"], a["ngram_ids"])
        h = layers(h, a["positions"], a["slot_mapping"], dict(zip(windows, b)), a["block_table"], a["swa_table"],
                   a["state_slot"])
        return with_h(run(post_d, h, a["temperature"], a["top_p"], a["top_k"], a["min_p"], a["noise"], a["board"],
                      a["write_slot"], a["bitmask"], a["penalties"]), h)

    def prefill(*args):
        a = sig["forward_prefill"].bind(*args)
        a.apply_defaults()
        a = a.arguments
        h, *b = run(prep_p, a["input_ids"], a["positions"], a["block_table"], a["swa_table"], a["swa_first"],
                       a["ngram_ids"])
        h = layers(h, a["positions"], a["slot_mapping"], dict(zip(windows, b)), a["block_table"], a["swa_table"],
                   a["state_slot"], plan=plan_p)
        return with_h(run(post_p, h, a["last_index"], a["temperature"], a["top_p"], a["top_k"], a["min_p"], a["noise"],
                      a["board"], a["write_slot"], a["bitmask"], a["penalties"], a["plp_targets"]), h)

    def extend(*args):
        a = sig["forward_extend"].bind(*args)
        a.apply_defaults()
        a = a.arguments
        h, *b = run(prep_e, a["input_ids"], a["positions"], a["block_table"], a["swa_table"], a["swa_first"],
                    a["ngram_ids"])
        h = layers(h, a["positions"].reshape(-1), a["slot_mapping"].reshape(-1), dict(zip(windows, b)),
                   a["block_table"], a["swa_table"], a["state_slot"])
        return with_h(run(post_e, h, a["temperature"], a["top_p"], a["top_k"], a["min_p"], a["noise"], a["u_accept"],
                          a["draft"]), h)

    def mixed(*args):
        """forward_mixed: the prefill groups' graphs (plan_p) over each group's chunk and decode rows; the
        post graph is the prefill's, sampling every chunk's last row and every decode row."""
        a = sig["forward_mixed"].bind(*args)
        a.apply_defaults()
        a = a.arguments
        h, b, db = run(prep_m, a["input_ids"], a["positions"], a["block_table"], a["board"], a["read_slot"],
                       a["dec_table"], a["dec_ctx"], a["ngram_ids"])
        h = layers(h, a["positions"], a["slot_mapping"], {None: b}, a["block_table"], None, a["state_slot"],
                   plan=plan_p, mixed=(a["dec_table"], db, a["dec_state"]))
        if model._mixed_split():  # the "split" sequence-parallel layout (models/hybrid.py MIXED_SP)
            return with_h(run(post_m, h, a["positions"], a["dec_ctx"], a["last_index"], a["temperature"], a["top_p"],
                              a["top_k"], a["min_p"], a["noise"], a["board"], a["write_slot"], a["bitmask"],
                              a["penalties"], a["plp_targets"]), h)
        return with_h(run(post_p, h, a["last_index"], a["temperature"], a["top_p"], a["top_k"], a["min_p"], a["noise"],
                          a["board"], a["write_slot"], a["bitmask"], a["penalties"], a["plp_targets"]), h)

    return decode, prefill, extend, mixed


def _copy_rows(src, dst, *pools):
    """pool[dst] = pool[src] for every recurrent-state pool: a state checkpoint saved or restored
    (engine/state_pool.py). Padding pairs copy the scratch row onto itself (copy_buckets)."""
    for p in pools:
        p.index_put_((dst,), p[src])
    return dst.sum()


def state_checkpoint_rows(ecfg: EngineConfig) -> int:
    """Rows of a recurrent model's state pool kept for prefix-cache checkpoints: the configured
    count, else twice max_num_seqs (SGLang v0.5.21 int8_mamba_ckpt_size's default: 2x the active
    mamba pool), none with the prefix cache off. Per DP-attention group."""
    if not ecfg.prefix_caching:
        return 0
    return ecfg.state_checkpoints if ecfg.state_checkpoints is not None else 2 * ecfg.group_max_num_seqs


def mixed_unsupported(model, mcfg: ModelConfig, ecfg: EngineConfig) -> str | None:
    """Why a model / engine cannot run mixed batches (DecoderForCausalLM.forward_mixed), or None."""
    from ..models.qwen4_exp import QSASpec

    hy = getattr(mcfg, "hybrid", None)
    if ecfg.spec_method:
        return f"speculative decoding ({ecfg.spec_method}): its decodes are verify calls"
    if getattr(model, "mtp", None) is not None:
        return "an MTP head drafts from every call's hidden states"
    if getattr(model, "window", None) is not None:
        return "sliding-window layers"
    if mcfg.sconv_kernel or any(getattr(l.spec, "rel_extent", 0) for l in model.kv_layers()):
        return "Inkling's short convolutions / relative position logits"
    if hy is not None and (hy.ple is not None or any(isinstance(l.spec, QSASpec) for l in model.layers)):
        return "Qwen3.8-Flash-Next's QSA layers / Per-Layer Embedding"
    return None


class ModelRunner:
    def __init__(self, model, mcfg: ModelConfig, ecfg: EngineConfig, num_pages: int, device: torch.device,
                 kv_heads: int | None = None):
        self.model = model
        self.tp_send = None  # rank 0 under tensor parallelism: broadcasts every graph call
        self.mcfg = mcfg
        self.ecfg = ecfg
        self.device = device
        self.ps = ecfg.page_size
        slots = num_pages * self.ps
        kv_dtype = kv_cache_torch_dtype(ecfg)
        shapes = model.kv_shapes()  # per layer, this rank's (nkv, Dk), (nkv, Dv)
        self.k_caches = [torch.zeros((slots, *k), dtype=kv_dtype, device=device) for k, _ in shapes]
        self.v_caches = [torch.zeros((slots, *v), dtype=kv_dtype, device=device) for _, v in shapes]
        # max_rows: the most query rows one graph call carries (a prefill chunk, or a verify batch).
        model.bind_kv_cache(self.k_caches, self.v_caches, self.ps,
                            fp8_max=fp8_e4m3_max(device) if kv_dtype == torch.float8_e4m3fn else None,
                            max_rows=max(max(ecfg.resolved_prefill_token_buckets()),
                                         max(ecfg.resolved_decode_batch_buckets()) * (1 + ecfg.spec_k)),
                            max_keys=max(ecfg.resolved_page_buckets()) * self.ps)
        # Token-slot state beside K and V (Inkling's short-convolution inputs), in the model dtype,
        # addressed by the same slots, so prefix caching and the host tier treat it as KV.
        states = [{n: torch.zeros((slots, *shp), dtype=ecfg.dtype, device=device) for n, shp in d.items()}
                  for d in getattr(model, "token_state_shapes", lambda: [])()]
        if any(states):
            model.bind_token_states(states)
        self.s_caches = [t for d in states for t in d.values()]
        # Per-token caches beside K and V (a QSA layer's raw indexer keys, models/qwen4_exp.py), paged
        # like them (the host KV tier copies them with the page).
        self.aux_caches = [torch.zeros((slots, *a), dtype=kv_dtype, device=device) for a in model.aux_kv_shapes()]
        model.bind_aux_kv(self.aux_caches)
        # DP attention (engine/dp.py): every call holds the N groups' batches, group-major, each
        # padded to one bucket. Arguments a token mixer reads are per group (PerGroup), the rest
        # cover all N * B rows (models/decoder.py DecoderForCausalLM: DP attention).
        self.dp = ecfg.dp_attention
        self.dp_group = getattr(model, "dp_group", 0)
        self.last_layout = None  # DP attention: where the last call put each sequence (see _layout)
        self.decode_buckets = ecfg.resolved_decode_batch_buckets()
        self.prefill_buckets = ecfg.resolved_prefill_token_buckets()
        # Mixed batches (EngineConfig.mixed_batch, KILN_MIXED_BATCH=1): every prefill call is a mixed call
        # (DecoderForCausalLM.forward_mixed) carrying up to mixed_rows decoding sequences per DP-attention
        # group besides its chunk, so a step with prefill work makes no separate decode call for them.
        self.mixed_rows = 0
        if ecfg.mixed_batch:
            why = mixed_unsupported(model, mcfg, ecfg)
            if why:
                raise ValueError(f"mixed batches (KILN_MIXED_BATCH=1): {why}")
            self.mixed_rows = ecfg.mixed_decode_rows or self.decode_buckets[-1]
        # Sequence-parallel prefill streams (models/decoder.py prefill_sp_enabled) split every chunk's rows (the
        # groups' buckets together, plus a mixed call's decode rows) evenly over the tp ranks: off when a
        # bucket does not divide.
        if getattr(model, "prefill_sp", False):
            from ..models import hybrid

            extra = self.mixed_rows if hybrid.MIXED_SP == "rows" else 0  # "split": the decode rows stay out of it
            odd = [b for b in self.prefill_buckets if ((b + extra) * self.dp) % model.tp_size]
            if odd:
                model.prefill_sp = False
                what = f"(prefill buckets {odd} + {extra} mixed decode rows)" if extra else \
                    f"prefill buckets {odd}"
                print(f"kiln: sequence-parallel prefill streams off: {what} x dp_attention {self.dp} "
                      f"do not divide over tp={model.tp_size}", flush=True)
        if getattr(model, "decode_sp", False):  # the same for decode calls (models/decoder.py decode_sp_enabled)
            odd = [b for b in self.decode_buckets if (b * self.dp) % model.tp_size]
            if odd:
                model.decode_sp = False
                print(f"kiln: sequence-parallel decode streams off: decode buckets {odd} x dp_attention {self.dp} "
                      f"do not divide over tp={model.tp_size}", flush=True)
        self.page_buckets = ecfg.resolved_page_buckets()
        self.mtp_q_buckets = tuple(sorted({1, 1 + ecfg.spec_k, *self.prefill_buckets}))
        self.window = getattr(model, "window", None)  # sliding-window layers' span, if any
        self.K = ecfg.sampling_candidates
        self._rngs: dict[str, np.random.Generator] = {}
        self.grammar = None  # GrammarBackend, set by the engine on first use
        # Token board for overlap scheduling: one slot per live request plus a scratch slot
        # that padded rows write to.
        n_slots = 2 * ecfg.max_num_seqs + 8
        self.board = torch.zeros(n_slots, dtype=torch.float32, device=device)
        self.scratch_slot = n_slots - 1
        self._free_slots = list(range(n_slots - 2, -1, -1))
        self._slot: dict[str, int] = {}
        self.compile_seconds: dict[tuple, float] = {}
        self.calls: dict[tuple, int] = defaultdict(int)
        # meta: kiln/capture.py traces the device run's graphs on a host without NeuronCores, through
        # a backend that writes each graph's HLO into the compile cache instead of compiling it.
        if device.type in ("neuron", "meta"):
            import torch._dynamo as dynamo

            # One static graph per bucket; dynamo must not give up on recompiles.
            for knob in ("cache_size_limit", "recompile_limit", "accumulated_cache_size_limit",
                         "accumulated_recompile_limit"):
                if hasattr(dynamo.config, knob):
                    setattr(dynamo.config, knob, 1 << 20)
            os.environ.setdefault("NEURON_LIBTORCH_COMPILATION_TIMEOUT", "7200")
            fp8_weights = any(t.dtype == torch.float8_e4m3fn for t in model.parameters())
            if os.environ.get("KILN_HASH_DUMP"):
                from .. import capture

                capture.dump_hash_inputs(os.environ["KILN_HASH_DUMP"])
            if device.type == "neuron":
                be = canonical_neuron_backend()
            else:
                from .. import capture

                be = capture.backend()
            opts = dict(backend=be, fullgraph=True, dynamic=False,
                        options={"compiler_args": neuronx_cc_args(kv_dtype, fp8_weights)})
            compile_ = lambda f: torch.compile(f, **opts)  # noqa: E731
        else:
            compile_ = lambda f: f  # noqa: E731
        # Linear-attention layers: their recurrent state per running request (engine/state_pool.py),
        # passed to decode / prefill as a trailing state_slot argument.
        self.state = None
        if model.state_layers():
            from .state_pool import StatePool

            # Speculative verify writes the state after each of its 1 + k positions into the
            # request's own rows (state_pool.py), so each request holds 1 + k rows.
            per_req = 1 + ecfg.spec_k if ecfg.spec_method else 1
            self.state = StatePool(model, ecfg.group_max_num_seqs, device, ecfg.dtype, self.dp, per_req,
                                   state_checkpoint_rows(ecfg))
            self._state_arg = {n: list(inspect.signature(getattr(model, f)).parameters).index("state_slot")
                               for n, f in (("decode", "forward_decode"), ("prefill", "forward_prefill"),
                                            ("verify", "forward_extend"))}
            self._copy = compile_(_copy_rows)
            # At least 4 pairs: a 2-row copy of Qwen3.5-0.8B's 18.6 MB rows measured 4.5 ms on trn1
            # against 1.2 ms for 4 rows and 2.1 ms for 8 (tools/bench_linear_serving.py --what parts, SDK
            # 2.32, bf16, 2026-10-03), the same trap as the one-row write in linear_attn._write_rows.
            self.copy_buckets = _pow2_ladder(4, max(4, ecfg.max_num_seqs))
        # A Per-Layer Embedding's n-gram ids are hashed on the host for every row (models/qwen4_exp.py).
        hy = getattr(mcfg, "hybrid", None)
        self.ple = getattr(hy, "ple", None)
        if self.ple is not None:
            self._ngram_arg = {n: list(inspect.signature(getattr(model, f)).parameters).index("ngram_ids")
                               for n, f in (("decode", "forward_decode"), ("prefill", "forward_prefill"),
                                            ("verify", "forward_extend"))}
        if ecfg.piecewise:
            self._decode, self._prefill, self._extend, self._mixed = _piecewise(
                model, compile_, ecfg.piecewise_group, ecfg.piecewise_moe_group, ecfg.piecewise_prefill_moe_group)
        else:
            self._decode = compile_(model.forward_decode)
            self._prefill = compile_(model.forward_prefill)
            self._extend = compile_(model.forward_extend)
            self._mixed = compile_(model.forward_mixed)
        self._mtp = compile_(model.forward_mtp_k) if getattr(model, "mtp", None) is not None else None
        self.spec_k = ecfg.spec_k if ecfg.spec_method == "mtp" else 1  # MTP passes per draft graph
        self._hidden: dict[str, torch.Tensor] = {}
        self._hidden_sp: set[str] = set()  # hidden states holding only this rank's rows of a prefill chunk
        self._host_kv: dict[int, list] = {}  # host KV tier slot -> per-layer (K, V) page copies
        self._host_ckpt: dict[int, list] = {}  # host checkpoint slot -> every state pool's row
        self._hcount = 0
        self._last_out = None  # the output of the last graph call launched on this rank (_settle)
        self.last_hidden: str | None = None

    # -- sampling inputs ----------------------------------------------------------

    def _sampling(self, reqs: list[Request | None]):
        n = len(reqs)
        temperature = np.zeros(n, np.float32)
        top_p = np.ones(n, np.float32)
        top_k = np.zeros(n, np.int64)
        min_p = np.zeros(n, np.float32)
        noise = np.full((n, max(self.K, 1)), 0.5, np.float32)
        for i, r in enumerate(reqs):
            if r is None or r.params.temperature <= 0:
                continue
            p = r.params
            temperature[i], top_p[i], top_k[i], min_p[i] = p.temperature, p.top_p, p.top_k, p.min_p
            rng = self._rngs.get(r.rid)
            if rng is None:
                rng = self._rngs[r.rid] = np.random.default_rng(
                    p.seed if p.seed is not None else (self.ecfg.seed, zlib.crc32(r.rid.encode())))
            noise[i] = rng.random(noise.shape[1], dtype=np.float32)
        return temperature, top_p, top_k, min_p, noise

    def _bitmask(self, reqs: list[Request | None]):
        """Grammar bitmask rows for a call, or None when no request in it is constrained
        (which selects the graph variant without the mask)."""
        forced = [r.think_force[0] if r is not None and r.think_force else None for r in reqs]
        grammar = any(r is not None and r.matcher is not None for r in reqs)
        if not grammar and all(f is None for f in forced):
            return None
        if grammar:
            mask = self.grammar.bitmask([r.matcher if r is not None else None for r in reqs]).numpy()
        else:  # xgrammar's layout: int32 words, bit t % 32 of word t // 32 = token t allowed
            mask = np.full((len(reqs), -(-self.mcfg.vocab_size // 32)), -1, np.int32)
        for i, t in enumerate(forced):
            if t is not None:
                mask[i] = 0
                mask[i, t // 32] = np.array(1 << (t % 32), np.uint32).view(np.int32)
        return mask

    PENALTY_BUCKETS = (256, 1024, 4096, 16384)

    def _penalties(self, reqs: list[Request | None]):
        """(hist_ids, out_counts, rep, freq, pres) for a call, or None when no request in it
        uses a penalty (which selects the graph variant without them)."""
        if not any(r is not None and r.has_penalties for r in reqs):
            return None, None
        rows = []
        for r in reqs:
            if r is None or not r.has_penalties:
                rows.append(({}, ()))
                continue
            counts: dict[int, int] = {}
            for t in r.output_ids:
                counts[t] = counts.get(t, 0) + 1
            seen = set(r.prompt_ids) if r.params.repetition_penalty != 1.0 else set()
            rows.append((counts, seen))
        M = pick_bucket(max(1, max(len(c.keys() | s) for c, s in rows)), self.PENALTY_BUCKETS)
        n = len(reqs)
        ids = np.zeros((n, M), np.int64)
        cnt = np.zeros((n, M), np.float32)
        rep = np.ones(n, np.float32)
        freq = np.zeros(n, np.float32)
        pres = np.zeros(n, np.float32)
        for i, (r, (counts, seen)) in enumerate(zip(reqs, rows)):
            toks = sorted(counts.keys() | seen)
            if not toks:
                continue
            ids[i, :len(toks)] = toks
            cnt[i, :len(toks)] = [counts.get(t, 0) for t in toks]
            ids[i, len(toks):] = toks[0]  # idempotent padding: same id, same count
            cnt[i, len(toks):] = cnt[i, 0]
            p = r.params
            rep[i], freq[i], pres[i] = p.repetition_penalty, p.frequency_penalty, p.presence_penalty
        return (ids, cnt, rep, freq, pres), M

    def _extras(self, rows):
        mask = self._bitmask(rows)
        pen, M = self._penalties(rows)
        key = (("grammar",) if mask is not None else ()) + ((f"pen{M}",) if pen is not None else ())
        return [mask, pen], key

    def _swa_pages(self, queries: int, P: int) -> int:
        """Pages a sliding-window table needs for `queries` consecutive positions per row, or
        0 when the full table (P pages) is no larger. The span of window + queries - 1
        positions starts anywhere inside a page, hence the extra page."""
        if (self.window is None or os.environ.get("KILN_SWA_TABLE", "1") == "0"
                or getattr(self.model, "full_tables", False)):
            return 0
        Pw = -(-(self.window + queries - 1) // self.ps) + 1
        return Pw if Pw < P else 0

    def _swa_row(self, pages, first_pos: int, Pw: int, table: np.ndarray) -> int:
        """Fill `table` [Pw] with the pages from the one holding first_pos - window + 1 on
        (null-padded) and return the position of its first slot."""
        p0 = max(0, first_pos - self.window + 1) // self.ps
        got = pages[p0 : p0 + Pw]
        table[: len(got)] = got
        return p0 * self.ps

    def pad_slots(self, shape) -> np.ndarray:
        """Cache slots for padded rows: slots 1 .. page_size - 1 of the null page in turn, never slot 0.

        A padded row sits at position 0 behind an all-null block table, so the one key it can see is the
        null page's slot 0, and nothing may write that slot while it is read: under DP attention the
        padded rows of a group used to write their own KV there too (one slot, several writers, read back
        by the same rows in the same graph), and in GLM-5.3-Flash's 12-layer decode graphs on trn1 the
        value they read back changed from execution to execution with identical inputs (measured
        2026-10-04, docs/neuron-notes.md "Padded rows wrote the slot they read"). Writes to slots 1 ..
        page_size - 1 are never read unmasked (every padded row's other keys and its pools are invisible),
        and those slots still count as padding where a graph tells real rows from padded ones by
        slot_mapping >= page_size (models/linear_attn.py). Slot 0 is then written by nothing after start-up."""
        n = int(np.prod(shape))
        if self.ps < 2:
            return np.full(shape, NULL_PAGE * self.ps, np.int64)
        return (NULL_PAGE * self.ps + 1 + np.arange(n, dtype=np.int64) % (self.ps - 1)).reshape(shape)

    def slot(self, req: Request) -> int:
        s = self._slot.get(req.rid)
        if s is None:
            s = self._slot[req.rid] = self._free_slots.pop()
        return s

    def forget(self, req: Request) -> None:
        self._rngs.pop(req.rid, None)
        if self.state is not None:
            self.state.release(req)
        s = self._slot.pop(req.rid, None)
        if s is not None:
            self._free_slots.append(s)

    def _with_state(self, name: str, args: list, rows) -> list:
        """args plus the state_slot argument (padding the optional ones before it with None)."""
        if self.state is None:
            return args
        i = self._state_arg[name]
        return args + [None] * (i - len(args)) + [rows if isinstance(rows, PerGroup) else np.asarray(rows, np.int64)]

    def _grp(self, a: np.ndarray):
        """A per-group argument built as [N, ...]: group 0's array itself without DP attention
        (so the call is exactly the plain one), else a PerGroup."""
        return a[0] if self.dp == 1 else PerGroup(a)

    def _place(self, reqs) -> tuple[list[tuple[int, int]], int]:
        """(group, index within the group) of each request, in order, and the largest group's
        count. Without DP attention every entry is group 0's, at its own index."""
        count = [0] * self.dp
        place = []
        for r in reqs:
            g = r.dp_group if self.dp > 1 else 0
            place.append((g, count[g]))
            count[g] += 1
        return place, max(count)

    def _with_ngram(self, name: str, args: list, spans, n: int) -> list:
        """args plus ngram_ids [n, *] for a Per-Layer Embedding model: rows of (request, start, end)
        spans of positions in order, zeros for the padding after them."""
        if self.ple is None:
            return args
        from ..models.qwen4_exp import ngram_ids

        out = np.zeros((n, len(self.ple.layers) * self.ple.heads), np.int64)
        o = 0
        for span in spans:
            req, start, end = span[:3]
            toks = span[3] if len(span) > 3 else (req.token_ids if req is not None else None)
            if toks is not None and PENDING in toks[max(0, start - self.ple.ngram + 1) : end]:
                raise RuntimeError("n-gram ids need every input token on the host (overlap scheduling is off "
                                   "for Per-Layer Embedding models)")
            if toks is not None:
                out[o : o + end - start] = ngram_ids(self.ple, toks, start, end)
            o += end - start
        i = self._ngram_arg[name]
        return args + [None] * (i - len(args)) + [out]

    def _dev(self, a):
        if isinstance(a, PerGroup):
            return self._dev(a[self.dp_group])
        if isinstance(a, str) and a == BOARD:
            return self.board
        if isinstance(a, str) and a.startswith(HIDDEN):
            return self._hidden[a]
        if isinstance(a, str) and a.startswith(PARAM):
            return getattr(self.model, a[len(PARAM):])
        if a is None:
            return None
        if isinstance(a, tuple):
            return tuple(self._dev(x) for x in a)
        return torch.from_numpy(a).to(self.device)

    def _exec(self, name: str, key: tuple, host_args: list):
        """Run graph `name` on host-side arguments (numpy arrays, BOARD, None, or tuples of
        arrays). Under tensor parallelism rank 0 first sends the same call to every other
        rank, so all ranks execute identical graphs on identical inputs."""
        t_exec = time.perf_counter()
        if self.tp_send is not None:
            self.tp_send((name, key, host_args))
        fn = {"decode": self._decode, "prefill": self._prefill, "verify": self._extend, "mtp": self._mtp,
              "mixed": self._mixed}[name]
        args = [self._dev(a) for a in host_args]
        first = key not in self.compile_seconds
        t = time.perf_counter()
        out = fn(*args)
        if isinstance(out, tuple) and name == "mtp":  # (drafts, MTP hidden): nothing reads the MTP hidden back
            out = out[0]
        elif isinstance(out, tuple):  # (output, hidden states kept on the device)
            out, h = out
            self.last_hidden = f"{HIDDEN}{self._hcount}"
            self._hcount += 1
            self._hidden[self.last_hidden] = h
            if name == "prefill" and self.model._sp_on():  # this rank's rows only (sequence-parallel streams)
                self._hidden_sp.add(self.last_hidden)
            if len(self._hidden) > HIDDEN_KEEP:
                old = next(iter(self._hidden))
                del self._hidden[old]
                self._hidden_sp.discard(old)
        if first:
            if self.device.type == "neuron":
                out = out.cpu()  # count the compile, not just the launch
            self.compile_seconds[key] = time.perf_counter() - t
        elif PROFILE_EXEC:
            if self.device.type == "neuron":
                out = out.cpu()
            EXEC_TIMES[key].append(time.perf_counter() - t_exec)
        self.calls[key] += 1
        self._last_out = out
        return out

    # -- warmup -------------------------------------------------------------------

    def warmup(self, spec_q: int | None = None, reverse: bool = False, plp: bool = False) -> dict:
        """Compile (or load from the compile cache) every bucket graph before serving, so no
        request ever waits on neuronx-cc. Dummy calls write only to the null page. Under DP
        attention a bucket is one group's (B, C) and the rows cover all N groups.

        reverse: prefill buckets first and every ladder backwards. Data-parallel replicas that
        warm the same graphs in opposite orders compile two of them at once (LNL compiles one
        graph at a time per process; a second process needing the same graph waits on its lock).
        plp: also every prefill bucket's prompt-logprobs variant (prefill's plp key:
        tools/check_ppl.py)."""
        t0 = time.perf_counter()
        N = self.dp

        def order(xs):
            return tuple(reversed(xs)) if reverse else tuple(xs)

        plain = not (spec_q and self.ecfg.spec_merged)  # no decode launches when draftless rows verify

        def warm_decode():
            for B in order(self.decode_buckets):
                for P in order(self.page_buckets):
                    ids = np.zeros(B, np.int64)
                    table = np.full((B, P), NULL_PAGE, np.int64)
                    args = [np.zeros(N * B, np.int64), ids, table, np.ones(B, np.int64), self.pad_slots(B),
                            *self._sampling([None] * (N * B)), BOARD, np.full(N * B, -1, np.int64),
                            np.full(N * B, self.scratch_slot, np.int64)]
                    Pw = self._swa_pages(1, P)
                    if Pw:
                        args += [None, None, np.full((B, Pw), NULL_PAGE, np.int64), np.zeros(B, np.int64)]
                    args = self._with_state("decode", args, [0] * B)  # the scratch row
                    args = self._with_ngram("decode", args, [], N * B)
                    if plain:
                        self._exec("decode", ("decode", B, P), args)
                        self._warm_mtp(B, 1, P)  # drafting after a decode step, and every recursion
                    if spec_q:
                        z = np.zeros((B, spec_q), np.int64)
                        zr = np.zeros((N * B, spec_q), np.int64)
                        samp = self._sampling([None] * (N * B * spec_q))
                        args = [zr, z, table, self.pad_slots((B, spec_q)), *samp,
                                np.full(N * B * spec_q, 0.5, np.float32), zr.reshape(-1)]
                        Pw = self._swa_pages(spec_q, P)
                        if Pw:
                            args += [np.full((B, Pw), NULL_PAGE, np.int64), np.zeros(B, np.int64)]
                        args = self._with_state("verify", args, np.zeros((B, 1 + spec_q), np.int64))
                        args = self._with_ngram("verify", args, [], N * B * spec_q)
                        self._exec("verify", ("verify", B, spec_q, P), args)
                        verified = self.last_hidden
                        self._warm_mtp(B, 1, P)  # after a verify: 1 or Q new positions
                        self.last_hidden = verified
                        self._warm_mtp(B, spec_q, P)

        def warm_mixed():
            """Every mixed (C, D, P) graph: the prefill buckets' graphs of a mixed-batch engine."""
            D = self.mixed_rows
            for C in order(self.prefill_buckets):
                for P in order(self.page_buckets):
                    R, S = C + D, N + N * D
                    z = np.zeros(R, np.int64)
                    st = (np.zeros(1, np.int64), np.zeros(D, np.int64)) if self.state is not None else (None, None)
                    for scored in ((False, True) if plp else (False,)):
                        args = self._mixed_args(np.zeros(N * R, np.int64), z, np.full(P, NULL_PAGE, np.int64),
                                                self.pad_slots(R),
                                                np.zeros(S, np.int64), [None] * S, np.full(S, self.scratch_slot, np.int64),
                                                np.full(N * R, -1, np.int64), np.full((D, P), NULL_PAGE, np.int64),
                                                np.ones(D, np.int64), [None, None],
                                                np.zeros(N * R, np.int64) if scored else None, *st)
                        self._exec("mixed", ("mixed", C, D, P) + (("plp",) if scored else ()), args)

        def warm_prefill():
            if self.mixed_rows:  # every prefill call is a mixed one
                warm_mixed()
            for C in order(self.prefill_buckets) if not self.mixed_rows else ():
                for P in order(self.page_buckets):
                    ids = np.zeros(C, np.int64)
                    args = [np.zeros(N * C, np.int64), ids, np.full(P, NULL_PAGE, np.int64), self.pad_slots(C),
                            np.zeros(N, np.int64), *self._sampling([None] * N), BOARD,
                            np.full(N, self.scratch_slot, np.int64)]
                    Pw = self._swa_pages(C, P)
                    base = list(args)
                    if Pw:
                        args += [None, None, np.full(Pw, NULL_PAGE, np.int64), np.zeros(1, np.int64)]
                    args = self._with_state("prefill", args, [0])
                    args = self._with_ngram("prefill", args, [], N * C)
                    if plp:  # the argument list prefill() builds for scored rows: extras, swa or two Nones, targets
                        a2 = base + [None, None] + ([np.full(Pw, NULL_PAGE, np.int64), np.zeros(1, np.int64)] if Pw
                                                    else [None, None]) + [np.zeros(N * C, np.int64)]
                        a2 = self._with_state("prefill", a2, [0])
                        a2 = self._with_ngram("prefill", a2, [], N * C)
                        self._exec("prefill", ("prefill", C, P, "plp"), a2)
                    self._exec("prefill", ("prefill", C, P), args)
                    prefilled = self.last_hidden
                    for q in self.mtp_q_buckets if self._mtp is not None else ():
                        if q <= C:  # a chunk of any length up to C fills MTP KV from these rows
                            self.last_hidden = prefilled
                            self._warm_mtp(1, q, P)
            for n in self.copy_buckets if self.state is not None and self.state.num_ckpt_rows else ():
                z = self._grp(np.zeros((self.dp, n), np.int64))
                self._copy_state(z, z)  # the scratch row onto itself

        for warm in ((warm_prefill, warm_decode) if reverse else (warm_decode, warm_prefill)):
            warm()
        if self.device.type == "neuron":
            torch.zeros(1, device=self.device).cpu()
        return {"graphs": len(self.compile_seconds), "seconds": time.perf_counter() - t0}

    def _warm_mtp(self, B: int, q: int, P: int) -> None:
        """One dummy MTP pass over the hidden states the previous warmup call left, so the
        graph for that (rows, B, q, P) is compiled; writes only to the null page."""
        if self._mtp is None:
            return
        N = self.dp
        z = np.zeros((B, q), np.int64)
        zr = np.zeros((N * B, q), np.int64)
        r = self.spec_k - 1
        rest = np.zeros((B, r), np.int64) if r else None
        args = [zr, z, np.full((B, P), NULL_PAGE, np.int64), self.pad_slots((B, q)), self.last_hidden, zr,
                np.zeros(N * B, np.int64), PARAM + "norm", rest, self.pad_slots((B, r)) if r else None]
        Pw, Pw1 = self._swa_pages(q, P), self._swa_pages(1, P)
        if Pw:
            args += [np.full((B, Pw), NULL_PAGE, np.int64), np.zeros(B, np.int64)]
            if r:
                args += [np.full((B, r, Pw1), NULL_PAGE, np.int64), np.zeros((B, r), np.int64)]
        self._exec("mtp", self._mtp_key(B, q, P, self.last_hidden), self._with_sp_src(args, self.last_hidden))

    def _mtp_key(self, B: int, Q: int, P: int, hkey: str) -> tuple:
        sp = ("sp",) if hkey in self._hidden_sp else ()
        return ("mtp", B, Q, P, self._hidden[hkey].shape[0], self.spec_k) + sp

    def _with_sp_src(self, args: list, hkey: str) -> list:
        """forward_mtp_k's arguments, plus the sp_onehot buffer when the hidden states are a
        sequence-parallel prefill chunk's (this rank's rows only; models/decoder.py _mtp_pass)."""
        if hkey not in self._hidden_sp:
            return args
        return args + [None] * (14 - len(args)) + [PARAM + "sp_onehot"]

    def _layout(self, kind: str, place, n: int, Q: int = 1):
        """DP attention: for each sequence of a call, in order, (its output row, its first hidden
        state row, its first scored row) in the all-groups output; None without DP attention.
        n is the call's group bucket (B, or a prefill's C)."""
        if self.dp == 1:
            return None
        if kind == "prefill":  # N sampled rows, then every group's C scored rows
            return [(g, g * n, self.dp + g * n) for g, _ in place]
        return [((g * n + b) * Q, (g * n + b) * Q, None) for g, b in place]

    # -- decode -------------------------------------------------------------------

    def decode(self, seqs: list[ScheduledSeq]) -> torch.Tensor:
        """B sequences x one token. Under DP attention B is the largest group's count and the
        output covers every group's B rows (self.last_layout places each sequence)."""
        n = len(seqs)
        N = self.dp
        place, nb = self._place([s.req for s in seqs])
        B = pick_bucket(nb, self.decode_buckets)
        P = pick_bucket(-(-max(s.end for s in seqs) // self.ps), self.page_buckets)
        ids = np.zeros(N * B, np.int64)
        pos = np.zeros((N, B), np.int64)
        table = np.full((N, B, P), NULL_PAGE, np.int64)
        ctx = np.ones((N, B), np.int64)
        slot = self.pad_slots((N, B))
        read = np.full(N * B, -1, np.int64)
        write = np.full(N * B, self.scratch_slot, np.int64)
        Pw = self._swa_pages(1, P)
        swa = np.full((N, B, Pw), NULL_PAGE, np.int64)
        swa_first = np.zeros((N, B), np.int64)
        rows: list[Request | None] = [None] * (N * B)
        for s, (g, b) in zip(seqs, place):
            r = s.req
            i = g * B + b
            t = r.token_ids[s.start]
            write[i] = self.slot(r)
            if t == PENDING:
                read[i] = write[i]
            else:
                ids[i] = t
            pos[g, b] = s.start
            np_pages = min(len(r.pages), P)
            table[g, b, :np_pages] = r.pages[:np_pages]
            ctx[g, b] = s.end
            slot[g, b] = r.pages[s.start // self.ps] * self.ps + s.start % self.ps
            if Pw:
                swa_first[g, b] = self._swa_row(r.pages, s.start, Pw, swa[g, b])
            rows[i] = r
        G = self._grp
        samp = self._sampling(rows)
        args = [ids, G(pos), G(table), G(ctx), G(slot), *samp, BOARD, read, write]
        extra, key = self._extras(rows)
        if key or Pw:
            args += extra
        if Pw:
            args += [G(swa), G(swa_first)]
        if self.state is not None:
            srow = np.zeros((N, B), np.int64)  # padded rows: the scratch row
            for s, (g, b) in zip(seqs, place):
                srow[g, b] = self.state.row(s.req)
            args = self._with_state("decode", args, G(srow))
        args = self._with_ngram("decode", args, [(s.req, s.start, s.start + 1) for s in seqs], N * B)
        self.last_layout = self._layout("decode", place, B)
        out = self._exec("decode", ("decode", B, P) + key, args)
        return out[:n] if N == 1 else out

    # -- speculative verify -------------------------------------------------------

    def verify(self, seqs: list[ScheduledSeq], Q: int) -> torch.Tensor:
        """Each seq scores its newest token plus its draft in one (B, Q, P) graph; drafts
        shorter than Q - 1 are padded with rows that write to the null page and are never
        read. Returns [B * Q, VERIFY_COLS] (rows of seq b are b*Q .. b*Q + Q - 1); under DP
        attention [N * B * Q, ...], group-major (self.last_layout)."""
        N = self.dp
        place, nb = self._place([s.req for s in seqs])
        B = pick_bucket(nb, self.decode_buckets)
        # The pages under each sequence's real positions (its newest token and its draft): a short or empty
        # draft's padded rows may reach past max_model_len, which a page bucket need not cover.
        P = pick_bucket(-(-max(s.start + 1 + len(s.draft) for s in seqs) // self.ps), self.page_buckets)
        ids = np.zeros((N * B, Q), np.int64)
        pos = np.zeros((N, B, Q), np.int64)
        slot = self.pad_slots((N, B, Q))
        draft = np.zeros((N * B, Q), np.int64)
        table = np.full((N, B, P), NULL_PAGE, np.int64)
        Pw = self._swa_pages(Q, P)
        swa = np.full((N, B, Pw), NULL_PAGE, np.int64)
        swa_first = np.zeros((N, B), np.int64)
        rows: list[Request | None] = [None] * (N * B * Q)
        for s, (g, b) in zip(seqs, place):
            r = s.req
            i = g * B + b
            k = len(s.draft)
            ids[i, 0] = r.token_ids[s.start]
            ids[i, 1 : 1 + k] = s.draft
            draft[i, :k] = s.draft
            pos[g, b] = s.start + np.minimum(np.arange(Q), k)  # padded rows repeat the last real position
            pages = np.asarray(r.pages, np.int64)
            p_real = pos[g, b, : 1 + k]
            slot[g, b, : 1 + k] = pages[p_real // self.ps] * self.ps + p_real % self.ps
            np_pages = min(len(r.pages), P)
            table[g, b, :np_pages] = r.pages[:np_pages]
            if Pw:
                swa_first[g, b] = self._swa_row(r.pages, s.start, Pw, swa[g, b])
            rows[i * Q : (i + 1) * Q] = [r] * Q
        samp = list(self._sampling(rows))
        u = np.array([self._rngs[r.rid].random() if r is not None and r.rid in self._rngs else 0.5
                      for r in rows], np.float32)
        G = self._grp
        # draft goes in flat [B * Q]: reshaping a [B, Q] draft inside the verify post graph made
        # neuronx-cc 2.27 fail with NCC_IBIR243 at MiMo-V2.6-Flash's tp=32 shapes (2026-10-03).
        args = [ids, G(pos), G(table), G(slot), *samp, u, draft.reshape(-1)] + ([G(swa), G(swa_first)] if Pw else [])
        if self.state is not None:  # read the current row, keep the state after every position
            st = np.zeros((N, B, 1 + Q), np.int64)  # padded rows: the scratch row
            for s, (g, b) in zip(seqs, place):
                rs = self.state.rows_of(s.req)
                if len(rs) < Q:
                    raise RuntimeError(f"verify of {Q} positions needs {Q} state rows per request, not {len(rs)}")
                st[g, b, 0] = self.state.row(s.req)
                st[g, b, 1:] = rs[:Q]
            args = self._with_state("verify", args, G(st))
        if self.ple is not None:  # n-gram ids over the drafts (padding positions hash token 0)
            spans = [(None, 0, Q)] * (N * B)
            for s, (g, b) in zip(seqs, place):
                toks = list(s.req.token_ids[: s.start + 1]) + list(s.draft)
                toks += [0] * (s.start + Q - len(toks))
                spans[g * B + b] = (s.req, s.start, s.start + Q, toks)
            args = self._with_ngram("verify", args, spans, N * B * Q)
        self.last_layout = self._layout("verify", place, B, Q)
        return self._exec("verify", ("verify", B, Q, P), args)

    # -- recurrent-state checkpoints (engine/state_pool.py) ---------------------------

    def copy_state(self, src: list[int], dst: list[int], groups: list[int] | None = None) -> None:
        """Copy whole state rows on the device (every linear layer's conv and recurrent row, and
        the model's other per-request rows), in order with the graph calls around it. groups: each
        pair's DP-attention group (its rows exist only on that group's ranks)."""
        if not src:
            return
        N = self.dp
        per = [[] for _ in range(N)]
        for i, (a, b) in enumerate(zip(src, dst)):
            per[groups[i] if groups is not None and N > 1 else 0].append((a, b))
        top = self.copy_buckets[-1]
        for i in range(0, max(len(p) for p in per), top):
            part = [p[i : i + top] for p in per]
            n = pick_bucket(max(len(p) for p in part), self.copy_buckets)
            a, b = np.zeros((N, n), np.int64), np.zeros((N, n), np.int64)  # padding: the scratch row onto itself
            for g, p in enumerate(part):
                for j, (x, y) in enumerate(p):
                    a[g, j], b[g, j] = x, y
            ga, gb = self._grp(a), self._grp(b)
            if self.tp_send is not None:
                self.tp_send(("state_copy", None, [ga, gb]))
            self._copy_state(ga, gb)

    def _copy_state(self, a, b) -> None:
        """One copy call on this rank's group's pairs (a, b: [n] arrays or their PerGroup)."""
        a, b = self._dev(a), self._dev(b)
        key = ("copy", a.shape[0])
        first = key not in self.compile_seconds
        t = time.perf_counter()
        out = self._copy(a, b, *self.state.pools())
        if first:
            if self.device.type == "neuron":
                out.cpu()
            self.compile_seconds[key] = time.perf_counter() - t
        self.calls[key] += 1
        self._last_out = out

    # -- prefill ------------------------------------------------------------------

    def prefill(self, s) -> torch.Tensor:
        """One sequence's chunk, padded to its bucket C. Under DP attention `s` may be a list of
        chunks of DIFFERENT groups (a group without one runs padding), all padded to the largest
        one's bucket: the output is the N sampled rows, then (prompt logprobs) every group's C
        scored rows (self.last_layout)."""
        seqs = s if isinstance(s, list) else [s]
        N = self.dp
        place, nb = self._place([x.req for x in seqs])
        if nb > 1:
            raise ValueError("one prefill chunk per DP-attention group and call")
        C = pick_bucket(max(x.num_tokens for x in seqs), self.prefill_buckets)
        P = pick_bucket(-(-max(x.end for x in seqs) // self.ps), self.page_buckets)
        ids = np.zeros(N * C, np.int64)
        pos = np.zeros((N, C), np.int64)
        table = np.full((N, P), NULL_PAGE, np.int64)
        slot = self.pad_slots((N, C))
        last = np.zeros(N, np.int64)  # a group without a chunk samples its row 0, never read
        rows: list[Request | None] = [None] * N
        write = np.full(N, self.scratch_slot, np.int64)
        Pw = self._swa_pages(C, P)
        swa = np.full((N, Pw), NULL_PAGE, np.int64)
        swa_first = np.zeros((N, 1), np.int64)
        # Scored rows: prompt logprobs, or the logprobs of forced (jump-forward) tokens.
        plp = any(x.req.prompt_logprob_rows is not None or x.req.forced_logprob_rows is not None for x in seqs)
        targets = np.zeros(N * C, np.int64)  # row i scores the token after it (padding: token 0, ignored)
        for x, (g, _) in zip(seqs, place):
            r, n, o = x.req, x.num_tokens, g * C
            ids[o : o + n] = r.token_ids[x.start : x.end]
            pos[g, :n] = np.arange(x.start, x.end)
            np_pages = min(len(r.pages), P)
            table[g, :np_pages] = r.pages[:np_pages]
            pages = np.asarray(r.pages, np.int64)
            slot[g, :n] = pages[pos[g, :n] // self.ps] * self.ps + pos[g, :n] % self.ps
            last[g] = o + n - 1
            if x.sample:
                rows[g] = r
                write[g] = self.slot(r)
            if Pw:
                swa_first[g, 0] = self._swa_row(r.pages, x.start, Pw, swa[g])
            if plp:
                targets[o : o + n - 1] = r.token_ids[x.start + 1 : x.end]
                if x.end < len(r.token_ids):
                    targets[o + n - 1] = r.token_ids[x.end]
        G = self._grp
        samp = self._sampling(rows)
        args = [ids, G(pos), G(table), G(slot), last, *samp, BOARD, write]
        extra, key = self._extras(rows)
        if key or Pw or plp:
            args += extra
        if Pw:
            args += [G(swa), G(swa_first)]
        elif plp:
            args += [None, None]
        if plp:
            args.append(targets)
            key += ("plp",)
        if self.state is not None:
            srow = np.zeros((N, 1), np.int64)  # a group without a chunk: the scratch row
            for x, (g, _) in zip(seqs, place):
                srow[g, 0] = self.state.row(x.req)
            args = self._with_state("prefill", args, G(srow))
        args = self._with_ngram("prefill", args, [(x.req, x.start, x.end) for x in seqs], N * C)
        self.last_layout = self._layout("prefill", place, C)
        return self._exec("prefill", ("prefill", C, P) + key, args)

    # -- mixed: prefill chunks with decode rows riding along -----------------------------

    def mixed(self, chunks: list[ScheduledSeq], decs: list[ScheduledSeq]) -> torch.Tensor:
        """One mixed call (DecoderForCausalLM.forward_mixed): each DP-attention group's prefill chunk (at
        most one; a group without one runs padding), padded to its bucket C, followed by up to D =
        mixed_rows of the group's decoding sequences (padded to D). The page bucket covers every
        sequence of the call. Output rows: the N chunks' sampled rows, then each group's D decode rows
        (N + g * D + b), then, with prompt logprobs, every group's C + D scored rows
        (self.last_layout places each sequence: (its output row, is a decode row))."""
        N, D, ps = self.dp, self.mixed_rows, self.ps
        place_c, nc = self._place([x.req for x in chunks])
        place_d, nd = self._place([x.req for x in decs])
        if nc > 1 or nd > D:
            raise ValueError(f"a mixed call takes one chunk and up to {D} decodes per group, not {nc} / {nd}")
        C = pick_bucket(max(x.num_tokens for x in chunks), self.prefill_buckets)
        P = pick_bucket(-(-max(x.end for x in chunks + decs) // ps), self.page_buckets)
        R, S = C + D, N + N * D
        ids = np.zeros(N * R, np.int64)
        read = np.full(N * R, -1, np.int64)
        pos = np.zeros((N, R), np.int64)
        slot = self.pad_slots((N, R))  # padding rows: the null page (pad_slots)
        table = np.full((N, P), NULL_PAGE, np.int64)
        dtable = np.full((N, D, P), NULL_PAGE, np.int64)
        dctx = np.ones((N, D), np.int64)
        # Sampled rows: a group without a chunk samples its row 0, never read; padded decode rows their own.
        last = np.concatenate([np.arange(N, dtype=np.int64) * R,
                               (np.arange(N, dtype=np.int64)[:, None] * R + C + np.arange(D)).reshape(-1)])
        write = np.full(S, self.scratch_slot, np.int64)
        rows: list[Request | None] = [None] * S
        plp = any(x.req.prompt_logprob_rows is not None or x.req.forced_logprob_rows is not None for x in chunks)
        targets = np.zeros(N * R, np.int64)
        for x, (g, _) in zip(chunks, place_c):
            r, n, o = x.req, x.num_tokens, g * R
            ids[o : o + n] = r.token_ids[x.start : x.end]
            pos[g, :n] = np.arange(x.start, x.end)
            np_pages = min(len(r.pages), P)
            table[g, :np_pages] = r.pages[:np_pages]
            pages = np.asarray(r.pages, np.int64)
            slot[g, :n] = pages[pos[g, :n] // ps] * ps + pos[g, :n] % ps
            last[g] = o + n - 1
            if x.sample:
                rows[g] = r
                write[g] = self.slot(r)
            if plp:
                targets[o : o + n - 1] = r.token_ids[x.start + 1 : x.end]
                if x.end < len(r.token_ids):
                    targets[o + n - 1] = r.token_ids[x.end]
        for s, (g, b) in zip(decs, place_d):
            r, i, k = s.req, g * R + C + b, N + g * D + b
            t = r.token_ids[s.start]
            write[k] = self.slot(r)
            if t == PENDING:
                read[i] = write[k]
            else:
                ids[i] = t
            pos[g, C + b] = s.start
            np_pages = min(len(r.pages), P)
            dtable[g, b, :np_pages] = r.pages[:np_pages]
            dctx[g, b] = s.end
            slot[g, C + b] = r.pages[s.start // ps] * ps + s.start % ps
            rows[k] = r
        G = self._grp
        extra, key = self._extras(rows)
        srow = drow = None
        if self.state is not None:
            sr, dr = np.zeros((N, 1), np.int64), np.zeros((N, D), np.int64)  # padding: the scratch row
            for x, (g, _) in zip(chunks, place_c):
                sr[g, 0] = self.state.row(x.req)
            for s, (g, b) in zip(decs, place_d):
                dr[g, b] = self.state.row(s.req)
            srow, drow = G(sr), G(dr)
        args = self._mixed_args(ids, G(pos), G(table), G(slot), last, rows, write, read, G(dtable), G(dctx),
                                extra, targets if plp else None, srow, drow)
        self.last_layout = [(g, False) for g, _ in place_c] + [(N + g * D + b, True) for g, b in place_d]
        return self._exec("mixed", ("mixed", C, D, P) + key + (("plp",) if plp else ()), args)

    def _mixed_args(self, ids, pos, table, slot, last, rows, write, read, dtable, dctx, extra, targets, srow, drow):
        """forward_mixed's positional arguments (mixed and its warmup build the same list)."""
        if self.ple is not None:
            raise NotImplementedError("mixed batches for a Per-Layer Embedding model")
        return [ids, pos, table, slot, last, *self._sampling(rows), BOARD, write, read, dtable, dctx, *extra, targets,
                srow, drow]

    # -- multi-token prediction drafts ------------------------------------------------

    def mtp_drafts(self, rows: list, hkey: str, k: int, batch: int = 1, prefill: bool = False) -> list[list[int]]:
        """mtp_launch, then its read-back (mtp_read)."""
        return self.mtp_read(self.mtp_launch(rows, hkey, k, batch, prefill))

    def mtp_read(self, launched) -> list[list[int]]:
        """The drafts of an mtp_launch, per row (waits for its graph)."""
        out, place, B, k = launched
        out = out.cpu()
        return [[int(out[g * B + b, s]) for s in range(k)] for g, b in place]

    def mtp_launch(self, rows: list, hkey: str, k: int, batch: int = 1, prefill: bool = False):
        """Launch the graph of greedy MTP drafts, k per row, ONE graph (DecoderForCausalLM.forward_mtp_k).
        rows: (req, first_position, ids, hidden_rows): MTP positions first_position ..
        first_position + len(ids) - 1 take ids (each the token after its position) and rows
        hidden_rows of the hidden states stashed under hkey; the first draft comes from the last
        of them, then k - 1 single-position passes follow inside the graph, pass s at position
        last + s, each seeded by the previous pass's draft and MTP hidden. Every position's page
        must be allocated. Under DP attention rows of every group share the graph, each group's
        padded to the largest group's count.

        batch: the per-group count of the call that produced the hidden states. The graph's batch
        bucket is at least that call's, so its key is one warmup compiled (warmup pairs every bucket
        with the hidden rows of a call of the same bucket): with fewer drafting rows than the verify
        before (e.g. one of eight requests still running) the smaller bucket compiled at run time,
        1.15 s for Qwen3.5-0.8B on trn1, 10-16 min for a GLM-5.3-Flash graph at tp=32.

        prefill: the hidden states are a prefill call's, one chunk per group, so the graph takes one
        row per group (warmup's (1, q, P) graphs), not a decode bucket of them: with decode buckets
        pinned to (16,) a 1024-token chunk's MTP call was padded to 16 x 1024 rows per group, and its
        graph was not one warmup built."""
        if k != self.spec_k:
            raise ValueError(f"MTP graphs were built for k={self.spec_k}, asked for {k}")
        N = self.dp
        place, nb = self._place([r[0] for r in rows])
        B = 1 if prefill else pick_bucket(max(nb, batch), self.decode_buckets)
        if prefill and nb > 1:
            raise ValueError("one prefill chunk per DP-attention group and call")
        Q = pick_bucket(max(len(r[2]) for r in rows), self.mtp_q_buckets)
        last = [r[1] + len(r[2]) - 1 for r in rows]
        P = pick_bucket(-(-(max(last) + k) // self.ps), self.page_buckets)  # covers every pass
        ids = np.zeros((N * B, Q), np.int64)
        pos = np.zeros((N, B, Q), np.int64)
        slot = self.pad_slots((N, B, Q))
        hidx = np.zeros((N * B, Q), np.int64)
        lasti = np.zeros(N * B, np.int64)
        table = np.full((N, B, P), NULL_PAGE, np.int64)
        r_ = k - 1
        pos_r = np.zeros((N, B, r_), np.int64)
        slot_r = self.pad_slots((N, B, r_))
        Pw, Pw1 = self._swa_pages(Q, P), self._swa_pages(1, P)
        swa = np.full((N, B, Pw), NULL_PAGE, np.int64)
        swa_first = np.zeros((N, B), np.int64)
        swa_r = np.full((N, B, r_, Pw1), NULL_PAGE, np.int64)
        swa_first_r = np.zeros((N, B, r_), np.int64)
        for j, ((req, first, ids_i, hrows), (g, b)) in enumerate(zip(rows, place)):
            i = g * B + b
            m = len(ids_i)
            ids[i, :m] = ids_i
            pp = np.arange(first, first + m)
            pos[g, b, :m] = pp
            pages = np.asarray(req.pages, np.int64)
            slot[g, b, :m] = pages[pp // self.ps] * self.ps + pp % self.ps
            hidx[i, :m] = hrows
            lasti[i] = m - 1
            npg = min(len(req.pages), P)
            table[g, b, :npg] = req.pages[:npg]
            if Pw:
                swa_first[g, b] = self._swa_row(req.pages, first, Pw, swa[g, b])
            for s in range(r_):
                p = last[j] + 1 + s
                pos_r[g, b, s] = p
                slot_r[g, b, s] = pages[p // self.ps] * self.ps + p % self.ps
                if Pw1:
                    swa_first_r[g, b, s] = self._swa_row(req.pages, p, Pw1, swa_r[g, b, s])
        G = self._grp
        rest = (G(pos_r), G(slot_r)) if r_ else (None, None)
        args = [ids, G(pos), G(table), G(slot), hkey, hidx, lasti, PARAM + "norm", *rest]
        if Pw:
            args += [G(swa), G(swa_first)] + ([G(swa_r), G(swa_first_r)] if r_ else [])
        out = self._exec("mtp", self._mtp_key(B, Q, P, hkey), self._with_sp_src(args, hkey))
        return out, place, B, k

    # -- weight updates -------------------------------------------------------------

    def reload_weights(self, path: str) -> None:
        if self.tp_send is not None:
            self.tp_send(("reload", None, [path]))
        self._reload(path)

    def _reload(self, path: str) -> None:
        new = dict(self.load_host(path).named_parameters())
        for name, p in self.model.named_parameters():
            src = new.get(name)
            if src is None or src.shape != p.shape or src.dtype != p.dtype:
                raise ValueError(f"{name}: the new checkpoint does not match ({None if src is None else tuple(src.shape)})")
            p.data.copy_(src.to(p.device))

    # -- host KV tier (engine/hicache.py) ---------------------------------------------
    # group: under DP attention a page or state row exists only on its group's ranks (each group has its
    # own pool and its own host tier, engine.py), so the call is broadcast and the other ranks skip it.

    def _mine(self, group: int | None) -> bool:
        return group is None or self.dp == 1 or group == self.dp_group

    def kv_save(self, slot: int, page: int, group: int | None = None) -> None:
        if self.tp_send is not None:
            self.tp_send(("kv_save", None, [slot, page, group]))
        self._kv_save(slot, page, group)

    def kv_load(self, slot: int, page: int, group: int | None = None) -> None:
        if self.tp_send is not None:
            self.tp_send(("kv_load", None, [slot, page, group]))
        self._kv_load(slot, page, group)

    def kv_drop(self, slot: int, group: int | None = None) -> None:
        if self.tp_send is not None:
            self.tp_send(("kv_drop", None, [slot, group]))
        self._kv_drop(slot, group)

    def _paged_caches(self) -> list[torch.Tensor]:
        return self.k_caches + self.v_caches + self.aux_caches + self.s_caches

    def _kv_save(self, slot: int, page: int, group: int | None = None) -> None:
        if not self._mine(group):
            return
        sl = slice(page * self.ps, (page + 1) * self.ps)  # a page is ps contiguous rows of every cache
        # copy=True: on a CPU engine .cpu() would return a view of the live cache.
        self._host_kv[slot] = [c[sl].to("cpu", copy=True) for c in self._paged_caches()]

    def _settle(self) -> None:
        """Wait for every graph call launched on this rank. An eager host-to-device copy into a device tensor
        does NOT wait for calls already queued on the NeuronCore: measured on trn1 (SDK 2.32, 2026-10-05,
        tools/check_cold_determinism.py OW, docs/neuron-notes.md "Padded rows wrote the slot they read"), a
        copy into a KV slot issued right after launching a decode call that writes that slot was overwritten
        by the call's write in 200 of 200 tries. The host tier's restores are such copies, issued while an
        overlapped step may still be in flight (its last write of a finished request's freed tail page,
        which a restore can be given). Calls run in submission order, so reading the last one's output back
        waits for all of them; the flag keeps a batch of restores to one wait."""
        out, self._last_out = self._last_out, None
        if out is not None and self.device.type == "neuron":
            out.cpu()

    def _kv_load(self, slot: int, page: int, group: int | None = None) -> None:
        if not self._mine(group):
            return
        self._settle()
        sl = slice(page * self.ps, (page + 1) * self.ps)
        for h, c in zip(self._host_kv[slot], self._paged_caches()):
            c[sl].copy_(h)

    def _kv_drop(self, slot: int, group: int | None = None) -> None:
        if self._mine(group):
            self._host_kv.pop(slot, None)

    def state_save(self, slot: int, row: int, group: int | None = None) -> None:
        """Host copy of one state row (a checkpoint the radix cache evicted, engine/hicache.py)."""
        if self.tp_send is not None:
            self.tp_send(("state_save", None, [slot, row, group]))
        self._state_save(slot, row, group)

    def state_load(self, slot: int, row: int, group: int | None = None) -> None:
        if self.tp_send is not None:
            self.tp_send(("state_load", None, [slot, row, group]))
        self._state_load(slot, row, group)

    def state_drop(self, slot: int, group: int | None = None) -> None:
        if self.tp_send is not None:
            self.tp_send(("state_drop", None, [slot, group]))
        self._state_drop(slot, group)

    def _state_save(self, slot: int, row: int, group: int | None = None) -> None:
        if self._mine(group):
            self._host_ckpt[slot] = [p[row].to("cpu", copy=True) for p in self.state.pools()]

    def _state_load(self, slot: int, row: int, group: int | None = None) -> None:
        if self._mine(group):
            self._settle()  # as _kv_load
            for h, p in zip(self._host_ckpt[slot], self.state.pools()):
                p[row].copy_(h)

    def _state_drop(self, slot: int, group: int | None = None) -> None:
        if self._mine(group):
            self._host_ckpt.pop(slot, None)

    # -- determinism checks (tools/check_cold_determinism.py) -----------------------------------

    def device_buffers(self) -> list[torch.Tensor]:
        """Every device tensor a call reads besides the weights: the paged caches, the state pools, the
        token board and the DSA selection scratch the layers share."""
        out = self._paged_caches() + (self.state.pools() if self.state is not None else []) + [self.board]
        seen = {id(t) for t in out}
        for layer in getattr(self.model, "layers", []):
            for name in ("dsa_topk", "dsa_mask", "pool_key"):
                t = getattr(layer, name, None)
                if isinstance(t, torch.Tensor) and id(t) not in seen:
                    seen.add(id(t))
                    out.append(t)
        return out

    def zero_buffers(self, which: str = "all") -> None:
        """Back to the zeros they were created with, on every rank: every device_buffers() tensor ("all"),
        or only the scratch state row ("scratch_row") or the null page of every paged cache ("null_page")."""
        if self.tp_send is not None:
            self.tp_send(("zero_buffers", None, [which]))
        self._zero_buffers(which)

    def _zero_buffers(self, which: str = "all") -> None:
        if which == "all":
            parts = self.device_buffers()
        elif which == "scratch_row":
            parts = [t[:1] for t in self.state.pools()] if self.state is not None else []
        elif which == "null_page":
            parts = [c[: self.ps] for c in self._paged_caches()]
        else:
            raise ValueError(f"zero_buffers: {which!r}")
        step = 1 << 26  # bytes per host-to-device copy
        for t in parts:
            rows = max(1, step // max(1, t[0].numel() * t.element_size())) if t.dim() else 1
            for i in range(0, t.shape[0] if t.dim() else 1, rows):
                part = t[i : i + rows] if t.dim() else t
                part.copy_(torch.zeros(part.shape, dtype=part.dtype))
        if self.device.type == "neuron":
            self.board.cpu()

    def fill_slots(self, start: int, end: int, seed: int) -> None:
        """Cache slots [start, end) of every paged cache filled with seeded N(0, 1) values on every rank (a
        stand-in for what an earlier request left in a reused page; tools/check_cold_determinism.py)."""
        if self.tp_send is not None:
            self.tp_send(("fill_slots", None, [start, end, seed]))
        self._fill_slots(start, end, seed)

    def _fill_slots(self, start: int, end: int, seed: int) -> None:
        g = torch.Generator().manual_seed(seed)
        for c in self._paged_caches():
            v = torch.randn((end - start, *c.shape[1:]), generator=g)
            if c.dtype == torch.float8_e4m3fn:
                v = v.clamp(-240, 240)
            c[start:end].copy_(v.to(c.dtype))
        if self.device.type == "neuron":
            self.board.cpu()

    def kv_page_bytes(self) -> int:
        """Host bytes one page takes on this rank (every layer's K and V, and the per-token caches
        and token-slot states beside them)."""
        return sum(self.ps * c[0].numel() * c.element_size() for c in self._paged_caches())

    # -- tensor-parallel worker -----------------------------------------------------

    def serve(self, recv) -> None:
        """Ranks > 0: execute every call rank 0 broadcasts, until it sends None."""
        while True:
            msg = recv()
            if msg is None:
                return
            name, key, host_args = msg
            if name == "reload":
                self._reload(*host_args)
                continue
            if name in ("kv_save", "kv_load", "kv_drop"):
                getattr(self, "_" + name)(*host_args)
                continue
            local = {"state_copy": self._copy_state, "state_save": self._state_save, "state_load": self._state_load,
                     "state_drop": self._state_drop, "zero_buffers": self._zero_buffers,
                     "fill_slots": self._fill_slots}.get(name)
            if local is not None:
                local(*host_args)
                continue
            out = self._exec(name, key, host_args)
            if key not in self.calls or self.calls[key] <= 1:
                out.cpu()  # surface a compile or runtime error on this rank, once
