from __future__ import annotations

import atexit
import collections
import functools
import itertools
import os
import time
import weakref
from dataclasses import dataclass

import torch

from .. import profiling
from ..config import EngineConfig, ModelConfig
from ..models.loader import load_model, resolve_model_path
from . import watchdog
from .kv_pool import PagePool
from .model_runner import ModelRunner
from .radix_cache import RadixCache
from .request import Request, SamplingParams, Status
from .sampler import NUM_TOP_LOGPROBS, unpack
from .scheduler import BLIND, NeedSync, Scheduler, SchedulerConfig

# Tensor-parallel engines not closed yet, held weakly: close() at exit (stopping the workers) without
# keeping every engine a process ever made alive. atexit.register(self.close) did, so a test session
# kept each tp engine's model, runner and process groups (and their gloo listeners) until it ended.
_OPEN: weakref.WeakSet = weakref.WeakSet()


@atexit.register
def _close_open() -> None:
    for eng in list(_OPEN):
        eng.close()


def _page_multiple(tokens: int, page_size: int) -> int:
    return -(-tokens // page_size) * page_size if tokens > 0 else 0


# KILN_PROFILE_STEP=1: the synchronous step's wall time per phase (schedule, launch, collect = waiting for the
# device and reading the outputs back, finish, draft = the MTP graphs and their read-back), keyed by the kinds of
# call the step made (one / verify / prefill); bench/serve_sweep.py prints them.
STEP_TIMES = collections.defaultdict(list) if os.environ.get("KILN_PROFILE_STEP") == "1" else None


@dataclass
class StepStats:
    """The work the last step() launched (decodes, prefill tokens, and its graph calls by kind: one call of
    a kind carries every DP-attention group's batch) and that step() call's wall time. Under overlap the
    call that launches a step reads the previous one, so its seconds belong to the previous launch."""
    num_decode: int = 0
    num_prefill_tokens: int = 0
    seconds: float = 0.0
    decode_calls: int = 0
    prefill_calls: int = 0
    num_restored: int = 0  # state checkpoints copied into requests (prefix hits of a recurrent model)
    prefill_chunks: int = 0  # prefill entries (one sequence's chunk each)
    deferred_chunks: int = 0  # prefill chunks the DP packing deferred to a later step (engine/dp.py)


def weight_config(cfg: EngineConfig, mcfg: ModelConfig) -> ModelConfig:
    """weight_dtype "fp8" / "fp8-experts": a BF16 checkpoint quantized to FP8 at load (all linears
    but embeddings, lm_head and routers / the routed experts only), see ModelConfig.quantized_at_load."""
    if cfg.weight_dtype in ("fp8", "fp8-experts"):
        return mcfg.quantized_at_load(dense=cfg.weight_dtype == "fp8")
    if cfg.weight_dtype not in ("auto", "bf16"):
        raise ValueError(f"weight_dtype must be auto, bf16, fp8 or fp8-experts, not {cfg.weight_dtype!r}")
    return mcfg


def resolve_attention_tp(mcfg: ModelConfig, cfg: EngineConfig) -> int:
    """The token mixers' TP degree: tp / dp_attention under DP attention (engine/dp.py), else
    EngineConfig.attention_tp or its default (models/decoder.py attention_tp)."""
    from ..models.decoder import attention_tp

    mtp = cfg.spec_method == "mtp"
    dp = cfg.dp_attention
    if dp < 1 or cfg.tp % dp:
        raise ValueError(f"dp_attention={dp} does not divide tp={cfg.tp}")
    if dp > 1:
        if cfg.attention_tp not in (None, cfg.tp // dp):
            raise ValueError(f"dp_attention={dp} at tp={cfg.tp} runs attention TP {cfg.tp // dp}, "
                             f"not attention_tp={cfg.attention_tp}")
        return attention_tp(mcfg, cfg.tp, cfg.tp // dp, mtp)
    return attention_tp(mcfg, cfg.tp, cfg.attention_tp, mtp)


def _cp_degree(mcfg: ModelConfig, attn_tp: int) -> int:
    """The context-parallel degree of the DSA caches (DecoderForCausalLM.cp): the attention TP when KILN_DSA_CP=1 and
    every attention layer can run the long path, else 1."""
    from ..config import LinearSpec
    from ..models import dsa_long, mla

    specs = [s for s in (mcfg.attn_layers or ()) if not isinstance(s, LinearSpec)]
    ok = dsa_long.cp_enabled() and attn_tp > 1 and specs and all(mla.long_capable(s) for s in specs)
    return attn_tp if ok else 1


def pool_pages(cfg: EngineConfig, mcfg: ModelConfig) -> int:
    """Pages in each KV pool. The pool is a graph input shape (every layer graph's KV cache), so a
    capture (kiln/capture.py) must size it exactly as the engine does."""
    from ..models.decoder import kv_heads_per_rank, state_bytes_per_token_rank

    from .model_runner import kv_cache_torch_dtype

    # KV for one page on ONE rank: each rank holds its attention rank's share of the KV heads
    # (or one replicated head when the attention TP exceeds the head count).
    attn_tp = resolve_attention_tp(mcfg, cfg)
    page_bytes = (mcfg.kv_bytes_per_token(kv_cache_torch_dtype(cfg)) * cfg.page_size
                  * kv_heads_per_rank(mcfg.num_kv_heads, attn_tp) // mcfg.num_kv_heads)
    page_bytes += state_bytes_per_token_rank(mcfg, attn_tp, cfg.dtype,
                                             kv_cache_torch_dtype(cfg) == torch.float8_e4m3fn) * cfg.page_size
    # Context-parallel DSA (models/dsa_long.py, KILN_DSA_CP=1): a rank holds 1 / attn_tp of every page.
    page_bytes //= _cp_degree(mcfg, attn_tp)
    # A model of linear-attention layers only has no KV: pages then only count positions.
    return cfg.num_pages or (int(cfg.kv_cache_gb * 2**30 // page_bytes) + 1 if page_bytes
                             else cfg.max_num_seqs * cfg.max_pages_per_seq + 1)


def build_shard(cfg: EngineConfig, path: str, num_pages: int, rank: int = 0):
    """Device, model shard and runner for one tensor-parallel rank (the whole model at tp=1)."""
    mcfg = ModelConfig.from_pretrained(path)
    if cfg.num_layers is not None:
        mcfg = mcfg.truncated(cfg.num_layers)
    if cfg.device == "neuron":
        # Runtime knobs vllm-neuron sets for serving (neuron_worker.py,
        # _init_neuron_distributed_environment_and_runtime). The execution queue default
        # is shallow: measured 8 back-to-back launches succeed and 16 fail with
        # "Execution Queue Full" (tools/profile_decode.py), which caps how far overlap
        # scheduling can run ahead.
        # Piecewise execution launches several graphs per step: take the runtime's maximum
        # (libnrt clamps larger values to 63, with a warning).
        os.environ.setdefault("NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS", "63" if cfg.piecewise else "32")
        os.environ.setdefault("NEURON_RT_IO_RING_CACHE_SIZE", "32")
        # NOT NEURON_RT_DISABLE_EXECUTION_BARRIER=1, the third knob vllm-neuron sets there (neuron_worker.py:715-718):
        # without the runtime's per-execution barrier a collective mismatch returns silently wrong numbers, and ranks
        # alternating world and attention-group collectives under host jitter deadlock (docs/neuron-notes.md
        # "Upstream harvest (2026-10)"). Its serving gain (+2-3% on trn1) comes from the hardware barrier instead,
        # which keeps the ordering (platform.configure_runtime_env, HW_BARRIER_FAMILIES).
        from .. import platform

        platform.configure_runtime_env()  # NEURON_LOGICAL_NC_CONFIG on trn2 / trn3, before the runtime
        import libtorch_neuronx_lite  # noqa: F401  registers the "neuron" device

        if rank == 0:
            print("kiln platform", platform.check_runtime(), flush=True)

        device = torch.device("neuron:0")  # each rank sees exactly one NeuronCore
    else:
        device = torch.device(cfg.device)
    mtp = cfg.spec_method == "mtp"
    atp = resolve_attention_tp(mcfg, cfg)
    dp = cfg.dp_attention
    group = attn_group = None
    if cfg.tp > 1:
        import torch.distributed as dist

        from . import tp

        group = dist.group.WORLD
        # Every rank, in the same order. Without DP attention the mixers reduce over it; under DP attention they
        # reduce over the world (models/decoder.py _attn_all_reduce), and only the sequence-parallel prefill
        # blocks' gather and reduce-scatter run inside it (models/hybrid.py KILN_SP_GROUP).
        attn_group = tp.attention_group(cfg.tp, atp)
    from .model_runner import fp8_e4m3_max

    mcfg = weight_config(cfg, mcfg)
    keep_fp8 = cfg.weight_dtype != "bf16" and (mcfg.quant_block is not None or mcfg.quant_expert_block is not None)
    model = load_model(path, mcfg, cfg.dtype, device, cfg.max_model_len, rank, cfg.tp, group,
                       keep_fp8=keep_fp8, fp8_max=fp8_e4m3_max(device), vocab_parallel=cfg.vocab_parallel,
                       packed_mxfp4=cfg.mxfp4_packed, mtp=mtp, moe_kernel=cfg.moe_kernel, attn_tp=atp,
                       attn_group=attn_group, dp_attention=dp, max_num_seqs=cfg.max_num_seqs, pd_role=cfg.pd_role)
    runner = ModelRunner(model, mcfg, cfg, num_pages, device, kv_heads=model.nkv)
    # For an EPLB rebalance (models/eplb.py): this rank's redundant slots reloaded from the same checkpoint.
    from ..models.loader import _Checkpoint, prepare_ep_slots

    runner.ep_prepare = lambda layer, index, slots, ck=None: prepare_ep_slots(path, model, layer, index, slots,
                                                                              fp8_e4m3_max(device), ck)
    runner.ep_checkpoint = lambda: _Checkpoint(path)
    # For update_weights_from_disk: the same shard of another checkpoint, on the host.
    runner.load_host = lambda p: load_model(p, mcfg, cfg.dtype, torch.device("cpu"), cfg.max_model_len, rank, cfg.tp,
                                            None, keep_fp8=keep_fp8, fp8_max=fp8_e4m3_max(device),
                                            vocab_parallel=cfg.vocab_parallel, packed_mxfp4=cfg.mxfp4_packed,
                                            mtp=mtp, moe_kernel=cfg.moe_kernel, attn_tp=atp, dp_attention=dp,
                                            max_num_seqs=cfg.max_num_seqs, pd_role=cfg.pd_role)
    _trim_after_load(rank, device)
    return model, runner


def _rss_gib() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**20
    return 0.0


def _trim_after_load(rank: int, device) -> None:
    """KILN_MALLOC_TRIM=1: hand the heap the loader freed back to the system (glibc keeps freed chunks below its mmap
    threshold in the process heap), and print this rank's host RSS before and after. A GLM-5.3-Flash rank of the 1M
    configuration held ~34.6 GB of host RSS on trn2, ~30 GB of it in two heap regions, and two such engines ran a
    trn2.48xlarge (2 TB) out of memory (docs/neuron-notes.md "Long context (1M)"). Off by default."""
    if os.environ.get("KILN_MALLOC_TRIM", "0") != "1" or device.type == "cpu":
        return
    import ctypes

    before = _rss_gib()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    print(f"kiln rank {rank}: host RSS {before:.1f} GiB after load, {_rss_gib():.1f} GiB after malloc_trim", flush=True)


class LLMEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        path = resolve_model_path(cfg.model_path)
        self.model_path = path
        self.mcfg = ModelConfig.from_pretrained(path)
        if cfg.num_layers is not None:
            self.mcfg = self.mcfg.truncated(cfg.num_layers)
        self.attn_tp = resolve_attention_tp(self.mcfg, cfg)
        # DP attention: dp groups, each with its own pool of num_pages pages (on every rank of the
        # group: a page holds the group's requests only), radix cache and scheduler (engine/dp.py).
        self.dp = cfg.dp_attention
        num_pages = pool_pages(cfg, self.mcfg)
        # Linear-attention layers carry per-request recurrent state (engine/state_pool.py), which
        # only moves forward: a prefix is resumed from a state checkpoint, a verify keeps the state
        # after every drafted position, and nothing ever rewinds it.
        from ..config import LinearSpec

        self.recurrent = any(isinstance(sp, LinearSpec) for sp in self.mcfg.attn_layers or ())
        if self.recurrent:
            if cfg.hicache_host_gb > 0 and not cfg.prefix_caching:
                raise ValueError("hicache needs the prefix cache (prefix_caching=True)")
        self.cache_pulled = 0
        if cfg.compile_cache_uri and cfg.device == "neuron":
            from .. import compile_cache

            self.cache_pulled = compile_cache.pull(cfg.compile_cache_uri)
        self._workers = []
        self._tp_saved = None  # rank 0's process state before init_rank (tp.process_state), restored by close
        if cfg.tp > 1:
            import multiprocessing as mp

            from . import tp

            port = tp.free_port()
            ctx = mp.get_context("spawn")
            self._workers = [ctx.Process(target=tp.worker_main, args=(r, cfg, path, num_pages, port), daemon=True)
                             for r in range(1, cfg.tp)]
            self._tp_saved = tp.process_state()
            try:
                for w in self._workers:
                    w.start()
                if cfg.device == "neuron":
                    tp.neuron_env(0, port, cfg.tp_core_base)
                tp.init_rank(0, cfg.tp, port)
            except BaseException:
                self._stop_tp()
                raise
        # The tokenizer (and so transformers) loads only AFTER tensor-parallel workers are
        # spawned: a process that imported transformers and then spawned a child traces
        # torch.topk differently from every other process (tools/probe_fx_normalisation.py),
        # which gave rank 0 its own compile cache keys and doubled every compile.
        from transformers import AutoTokenizer

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
        except (OSError, ValueError, TypeError):
            self.tokenizer = None  # token-id-only use; stop strings and grammars need one
        # Thinking budget boundaries (vLLM ReasoningConfig): the start string, the string forced
        # at the budget, and the tag that closes reasoning when the model ends it itself.
        enc = (lambda t: self.tokenizer(t, add_special_tokens=False)["input_ids"]) if self.tokenizer else None
        self.think_start_ids = enc(cfg.reasoning_start_str) if enc and cfg.reasoning_start_str else []
        self.think_end_ids = enc(cfg.reasoning_end_str) if enc and cfg.reasoning_end_str else []
        close = cfg.reasoning_end_str or ""
        close = close[close.rfind("</") :] if "</" in close else close
        self.think_close_ids = enc(close) if enc and close else []
        t = time.perf_counter()
        try:
            self.model, self.runner = build_shard(cfg, path, num_pages, 0)
        except BaseException:
            self._stop_tp()
            raise
        self.load_seconds = time.perf_counter() - t
        self.device = self.runner.device
        if cfg.tp > 1:
            from . import tp

            self.runner.tp_send = tp.send
            _OPEN.add(self)
        self.pools = [PagePool(num_pages) for _ in range(self.dp)]
        self.radixes = [RadixCache(p, cfg.page_size, cfg.eviction_policy, cfg.eviction_policy_config)
                        for p in self.pools]
        self.pool, self.radix = self.pools[0], self.radixes[0]
        scfg = SchedulerConfig(
            page_size=cfg.page_size,
            max_num_seqs=cfg.group_max_num_seqs,
            max_prefill_tokens=cfg.group_max_prefill_tokens,
            max_model_len=cfg.max_model_len,
            eos_token_ids=self.mcfg.eos_token_ids,
            policy=cfg.schedule_policy,
            admission=cfg.admission,
            admission_decode_tokens=cfg.admission_decode_tokens,
            max_num_queued_reqs=cfg.max_num_queued_reqs if self.dp == 1 else None,
            max_num_queued_tokens=cfg.max_num_queued_tokens if self.dp == 1 else None,
            prefix_cache=cfg.prefix_caching,
            # A recurrent model's hit stops at its deepest state checkpoint (radix_cache.py).
            recurrent=self.recurrent,
            ckpt_interval=_page_multiple(cfg.state_checkpoint_interval, cfg.page_size),
            ckpt_track=_page_multiple(cfg.state_track_interval, cfg.page_size),
            ckpt_prompt=cfg.state_checkpoint_prompt,
            ckpt_lookahead=cfg.state_checkpoint_lookahead,
        )
        draft_fn = self._propose if cfg.spec_method else None
        groups = [Scheduler(scfg, p, r, draft_fn=draft_fn) for p, r in zip(self.pools, self.radixes)]
        if self.runner.state is not None:  # checkpoint rows of the state pool back the radix nodes, per group
            for g, (sch, rad) in enumerate(zip(groups, self.radixes)):
                sch.states, sch.dp_group = self.runner.state, g
                rad.free_ckpt = functools.partial(self.runner.state.free_ckpt, group=g)
        if self.dp == 1:
            self.scheduler = groups[0]
        else:
            from .dp import DPScheduler

            self.scheduler = DPScheduler(groups, cfg.max_num_queued_reqs, cfg.max_num_queued_tokens,
                                         pack=cfg.dp_prefill_pack, pack_min=cfg.dp_prefill_pack_min,
                                         hold_steps=cfg.dp_prefill_hold_steps)
        self.proposer = None
        if cfg.spec_method == "ngram":
            from .spec_ngram import NgramProposer

            self.proposer = NgramProposer(cfg.spec_k, cfg.spec_ngram_min, cfg.spec_ngram_max)
        elif cfg.spec_method == "suffix":
            from .spec_suffix import SuffixProposer

            self.proposer = SuffixProposer(cfg.spec_k, cfg.suffix_max_tree_depth, cfg.suffix_max_cached_requests,
                                           cfg.suffix_max_spec_factor, cfg.suffix_min_token_prob)
        elif cfg.spec_method not in (None, "mtp"):  # mtp drafts come from the model (runner.mtp_drafts)
            raise ValueError(f"unknown spec_method {cfg.spec_method!r}")
        self.watermark = None
        if cfg.watermark is not None:
            from .watermark import Watermark

            wc = dict(cfg.watermark)
            if wc.pop("algorithm", "gumbel") != "gumbel":
                raise ValueError("only the gumbel watermark algorithm is supported")
            if "key" not in wc:
                raise ValueError("watermark config needs a key")
            self.watermark = Watermark(**wc)
        self.host_tier = None
        self.host_tiers = []
        self.weight_version: str | None = None
        if cfg.hicache_host_gb > 0:
            from .hicache import GroupMoves, HostTier

            budget = cfg.hicache_host_gb * 2**30  # per rank: a rank holds only its own group's copies
            ckpts = 0
            if self.runner.state is not None:  # half the host tier for state checkpoints
                ckpts = max(1, int(budget / 2 // self.runner.state.bytes_per_row()))
                budget -= ckpts * self.runner.state.bytes_per_row()
            page_bytes = self.runner.kv_page_bytes()
            pages = int(budget // page_bytes) if page_bytes else cfg.max_num_seqs * cfg.max_pages_per_seq
            for g, (sch, rad) in enumerate(zip(groups, self.radixes)):
                tier = HostTier(self.runner if self.dp == 1 else GroupMoves(self.runner, g), cfg.page_size,
                                max(1, pages), ckpts)
                rad.offload, rad.offload_ckpt, sch.host = tier.offload, tier.offload_ckpt, tier
                self.host_tiers.append(tier)
            self.host_tier = self.host_tiers[0]
        self.spec_proposed = 0
        self.spec_accepted = 0
        self.spec_verifies = 0  # drafts verified (one per sequence and step); tokens per verify = 1 + accepted / this
        # Per draft position i: verifies that reached it (every earlier draft accepted), and that accepted it.
        self.spec_pos_reached: list[int] = []
        self.spec_pos_accepted: list[int] = []
        self.mtp_seconds = 0.0  # wall time in _mtp_draft (MTP graphs and their read-back), the step's last part
        self._mtp_early: set[int] = set()  # prefill launches (id of the chunk list) whose MTP pass already ran
        self.jump_forward_tokens = 0  # output tokens forced by jump-forward decoding
        self._ids = itertools.count()
        self.last_step = StepStats()
        self._inflight = None  # (plan, launches) of a step whose results are not read yet
        # EPLB rebalances (models/eplb.py, _eplb_tick): only with redundant slots and the recorder on.
        from ..models import eplb as _eplb

        rec = any(getattr(l, "ep_stats", None) is not None for l in getattr(self.runner.model, "layers", []))
        self._eplb_every = _eplb.rebalance_interval() if rec else 0
        self._eplb_max = int(os.environ.get("KILN_EPLB_MAX_REBALANCES", "0"))
        self._eplb_calls = self._eplb_total = 0
        self._eplb_pending = None  # (start time, prefill calls) of a prepared rebalance not committed yet
        self.eplb_log: list[tuple[int, int, float, float, int]] = []
        self._pd_setup()

    # -- prefill / decode disaggregation (engine/disagg.py) ---------------------------------------------

    def _pd_setup(self) -> None:
        import queue

        cfg = self.cfg
        self.pd_role = cfg.pd_role
        self.pd_receiver = None
        self.pd_ready: queue.Queue = queue.Queue()  # complete handoffs (meta dicts), from the receiver thread
        self.pd_wake = None  # called after a handoff completes (server/api.py wakes its engine loop)
        self.pd_counts = collections.Counter()  # handed_off, done_at_prefill, admitted, injected, refused
        self.pd_done = collections.deque(maxlen=4096)  # metas of handoffs that ended on the prefill side (pd_poll)
        self.pd_inject_seconds = 0.0
        self._pd_prefetch_on = os.environ.get("KILN_PD_PREFETCH", "1") == "1"
        self._pd_pf: list = []  # handoffs placed since the last launch, to prefetch on every rank of their group
        if self.pd_role is None:
            return
        if self.pd_role not in ("prefill", "decode"):
            raise ValueError(f"pd_role must be prefill or decode, not {self.pd_role!r}")
        if cfg.spec_method:
            raise NotImplementedError("speculative decoding with disaggregation (the MTP KV and drafts of the prompt "
                                      "would have to be handed off too)")
        if self.mixed_unsupported_by_role():
            raise ValueError(f"mixed batches on a {self.pd_role} engine: it runs one kind of call only")
        print(f"kiln role: {self.pd_role} (layout {self.runner.pd_signature()}, tp {cfg.tp}, attention tp "
              f"{self.attn_tp}, dp_attention {self.dp})", flush=True)
        for p in self.pools:
            p.ascending = True
        if self.pd_role == "prefill":
            self.scheduler.prefill_only = True
            return
        # The handoff's eager copies then never land on a page or state row an overlapped step may still write, so
        # they need not wait for the device (_pd_cool, ModelRunner._pd_inject settle=False).
        self.pd_hold = bool(cfg.overlap) and os.environ.get("KILN_PD_SETTLE", "0") != "1"
        if self.pd_hold:
            for p in self.pools:
                p.hold = True
            if self.runner.state is not None:
                self.runner.state.hold = True
        if cfg.pd_listen:
            from . import disagg

            def done(meta, ready=self.pd_ready):
                ready.put(meta)
                if self.pd_wake is not None:
                    self.pd_wake()

            self.pd_receiver = disagg.Receiver(cfg.pd_listen, done, int(cfg.pd_buffer_gb * 2**30),
                                               signature=self.runner.pd_signature())
            self.pd_address = self.pd_receiver.address(cfg.pd_advertise)
            print(f"kiln pd: decode engine receives handoffs at {self.pd_address} (buffer {cfg.pd_buffer_gb} GB in "
                  f"{self.pd_receiver.dir})", flush=True)

    def mixed_unsupported_by_role(self) -> bool:
        return self.pd_role is not None and bool(self.runner.mixed_rows)

    def add_prefilled(self, meta: dict) -> Request:
        """Decode engine: a complete handoff (disagg.Receiver) as a running request's next step. Its first
        token was sampled by the prefill engine and is the last of token_ids; the step that admits it copies its
        parts onto the device before decoding it (_pd_inject_plan)."""
        from . import disagg

        if self.pd_role != "decode":
            raise ValueError("handoffs go to a decode engine (pd_role='decode')")
        if meta.get("error"):
            raise ValueError(meta["error"])
        if meta.get("done") is not None:
            raise ValueError("a handoff that finished on the prefill engine has nothing to decode")
        if int(meta.get("cp", 1)) > 1 and (int(meta["cp"]) != self.runner._pd_cp()[0]
                                            or int(meta.get("page_size", 0)) != self.cfg.page_size):
            raise ValueError(f"a handoff from a context-parallel engine (CP {meta['cp']}, page size {meta.get('page_size')}) "
                             f"needs a decode engine with the same CP degree and page size, not CP "
                             f"{self.runner._pd_cp()[0]} / page size {self.cfg.page_size}")
        sa = self._pd_sender_a(meta)
        if sa and not disagg.regroup_ok(sa, self.attn_tp):
            raise ValueError(f"a handoff from attention TP {sa} cannot be split for this decode engine's attention TP "
                             f"{self.attn_tp}: one must divide the other")
        params = disagg.params_from_json(meta["params"])
        req = Request(meta["rid"], list(meta["prompt_ids"]), params, priority=int(meta.get("priority", 0)))
        req.token_ids.append(int(meta["token"]))
        if meta.get("logprobs") is not None:
            req.logprobs = [(lp, list(ti), list(tl)) for lp, ti, tl in meta["logprobs"]]
        if meta.get("prompt_logprobs"):
            req.prompt_logprobs = {int(k): (v[0], int(v[1]), list(v[2]), list(v[3]))
                                   for k, v in meta["prompt_logprobs"].items()}
        req.num_cached_tokens = int(meta.get("num_cached_tokens", 0))
        req.first_token_time = time.monotonic()
        req.pd_meta = meta
        self.runner.pd_set_rng(req.rid, meta.get("rng"))
        self.scheduler.add_prefilled(req)
        if self._pd_prefetch_on:  # broadcast with the next step's launch (_pd_inject_plan), one message for all
            self._pd_pf.append((meta["xfer"], meta["parts"], req.dp_group, req.num_prompt, int(meta.get("cp", 1)),
                                self._pd_sender_a(meta)))
        self.pd_counts["admitted"] += 1
        return req

    def _pd_sender_a(self, meta: dict) -> int:
        """The sender's attention TP when it differs from this engine's (ModelRunner.pd_regroupable), else 0."""
        sa = int(meta.get("attention_tp") or 0)
        return sa if sa and sa != self.attn_tp else 0

    def pd_poll(self) -> list[Request]:
        """Admit every complete handoff the receiver holds (an engine driven without server/api.py)."""
        import queue

        out = []
        while True:
            try:
                meta = self.pd_ready.get_nowait()
            except queue.Empty:
                return out
            if meta.get("done") is not None or meta.get("error"):
                self.pd_release(meta["xfer"])
                self.pd_done.append(meta)
                self.pd_counts["refused" if meta.get("error") else "done_at_prefill"] += 1
                continue
            out.append(self.add_prefilled(meta))

    def pd_release(self, xfer: str) -> None:
        if self.pd_receiver is not None:
            self.pd_receiver.release(xfer)

    def _pd_inject_plan(self, plan) -> None:
        """Copy the parts of every handed-off request this step decodes for the first time onto the device,
        before any of the step's calls (they are ordered with the calls on every rank: ModelRunner.serve)."""
        t = time.perf_counter()
        if self._pd_pf:
            self.runner.pd_prefetch_many(self._pd_pf)
            self._pd_pf = []
        items = []
        for s in plan.decodes:
            r = s.req
            if r.pd_meta is None or r.pd_injected:
                continue
            srow = self.runner.state.row(r) if self.runner.state is not None else None
            items.append((None if self._pd_prefetch_on else r.pd_meta["parts"], list(r.pages), r.num_prompt, srow,
                          r.dp_group, not self.pd_hold, int(r.pd_meta.get("cp", 1)), r.pd_meta["xfer"],
                          self._pd_sender_a(r.pd_meta)))
            r.pd_injected = True
            plan.pd_injected.append(r.pd_meta["xfer"])
            self.pd_counts["injected"] += 1
        if items:
            self.runner.pd_inject_many(items)
        self.pd_inject_seconds += time.perf_counter() - t

    def _pd_handoff(self, r: Request) -> None:
        """Prefill engine, a request that just sampled its first token (or finished): decide with the request's
        own params whether it continues, and either hand its state off or tell the decode side it is done. Runs
        inside _finish, before the request's state row is released and before the next schedule."""
        from . import disagg

        orig, (xfer, dest) = r.handoff_params, r.handoff
        out = r.output_ids
        reason = r.finish_reason
        if reason == "length":  # the prefill engine's own limit of one token; the request's limits decide
            reason = None
            if len(out) >= orig.max_new_tokens or r.num_tokens >= self.cfg.max_model_len:
                reason = "length"
            elif orig.stop and self.tokenizer is not None:
                tail = self.tokenizer.decode(out[-(max(len(x) for x in orig.stop) + 8):])
                reason = "stop" if any(x in tail for x in orig.stop) else None
        n = orig.logprobs
        lps = [[lp, list(ti)[:n] if n is not None else [], list(tl)[:n] if n is not None else []]
               for lp, ti, tl in r.logprobs] if n is not None else None
        meta = {"xfer": xfer, "rid": r.rid, "prompt_ids": list(r.prompt_ids), "token": int(out[0]) if out else None,
                "logprobs": lps, "params": disagg.params_to_json(orig), "priority": r.priority,
                "num_cached_tokens": max(r.num_cached_tokens, 0), "signature": self.runner.pd_signature(),
                "prefill_seconds": time.monotonic() - r.arrival_time,
                "prompt_logprobs": {str(k): [v[0], v[1], list(v[2]), list(v[3])] for k, v in r.prompt_logprobs.items()}
                if r.prompt_logprobs else None}
        r.handoff_meta = meta
        if reason is not None or not out:
            meta.update(done=reason or "abort", parts=[])
            r.finish_reason = meta["done"]
            self.runner.pd_send_meta(dest, meta)
            self.pd_counts["done_at_prefill"] += 1
            return
        A, _ = self.runner.pd_attention()
        cp, _ = self.runner._pd_cp()
        meta.update(parts=disagg.part_names(A), rng=self.runner.pd_rng_state(r.rid), cp=cp, page_size=self.cfg.page_size,
                    attention_tp=A)
        srow = self.runner.state.row(r) if self.runner.state is not None else None
        self.runner.pd_extract(xfer, dest, r.handoff_pages, r.num_prompt, srow, r.dp_group, meta)
        r.finish_reason = "handoff"
        self.pd_counts["handed_off"] += 1

    def _pd_gate(self) -> None:
        """Admit handed-off requests in batches (KILN_PD_ADMIT_MIN handoffs waiting, or KILN_PD_ADMIT_WAIT steps since
        the last admission): every step that copies a handoff in is slower, whatever it copies, so fewer such steps
        with more in each trade a little TTFT for decode throughput. Defaults 1 / 0: admit at once."""
        groups = getattr(self.scheduler, "groups", [self.scheduler])
        waiting = sum(len(g.prefilled) for g in groups)
        if not hasattr(self, "_pd_since"):
            self._pd_since = 0
            self._pd_min = int(os.environ.get("KILN_PD_ADMIT_MIN", "1"))
            self._pd_wait = int(os.environ.get("KILN_PD_ADMIT_WAIT", "0"))
        ok = waiting > 0 and (waiting >= self._pd_min or self._pd_since >= self._pd_wait)
        for g in groups:
            g.admit_ok = ok
        self._pd_since = 0 if ok else self._pd_since + 1

    def _pd_cool(self) -> None:
        """The step in flight when the held pages and rows were freed has been read back: nothing writes them now."""
        for p in self.pools:
            p.release_held()
        for sch in getattr(self.scheduler, "groups", [self.scheduler]):
            sch.cooling = 0
        if self.runner.state is not None:
            self.runner.state.release_cooling()

    def pd_flush(self, timeout: float = 600.0) -> None:
        """Wait until every queued handoff frame of this process is written (rank 0's; the other ranks flush
        when the engine closes)."""
        snd = getattr(self.runner, "_pd_snd", None)
        if snd is not None:
            snd.flush(timeout)

    def add_request(self, prompt_ids: list[int], params: SamplingParams, rid: str | None = None,
                    priority: int = 0, session_id: str | None = None, handoff: tuple[str, str] | None = None) -> Request:
        """handoff: (transfer id, decode receiver "host:port"), on a prefill engine (engine/disagg.py): the
        request runs to its first token here and continues on that decode engine."""
        own = params
        if handoff is not None or self.pd_role == "prefill":
            from . import disagg

            if self.pd_role != "prefill" or handoff is None:
                raise ValueError("a prefill engine serves handoffs only, and only a prefill engine hands off")
            why = disagg.unsupported(params)
            if why:
                raise ValueError(f"{why} cannot be disaggregated: send the request to a decode engine directly")
            import dataclasses

            params = dataclasses.replace(params, max_new_tokens=1)
        elif self.pd_role == "decode" and not self.cfg.pd_bypass_prefill:
            raise ValueError("this decode engine takes handed-off requests only (start it with pd_bypass_prefill "
                             "to also serve whole requests)")
        req = Request(rid or f"req-{next(self._ids)}", list(prompt_ids), params, priority=priority,
                      session_id=session_id)
        if handoff is not None:
            req.handoff, req.handoff_params = handoff, own
        req.watermarked = self.watermark is not None and params.watermarking and params.temperature > 0
        if params.thinking_token_budget is not None:
            if not self.think_start_ids:
                raise ValueError("thinking_token_budget needs reasoning_start_str / reasoning_end_str "
                                 "(set by --reasoning-parser or --reasoning-config)")
            # A template that opens the reasoning block in the prompt starts the count now.
            p, st, cl = req.prompt_ids, self.think_start_ids, self.think_close_ids
            last = lambda seq: max((i for i in range(len(p) - len(seq) + 1) if p[i : i + len(seq)] == seq),  # noqa: E731
                                   default=-1)
            req.think_in = last(st) > last(cl)
        if params.grammar is not None:
            if self.runner.grammar is None:
                from .grammar import GrammarBackend

                self.runner.grammar = GrammarBackend(self.tokenizer, self.mcfg.vocab_size)
            req.matcher = self.runner.grammar.matcher(params.grammar)  # raises on a bad grammar
        self.scheduler.add(req)
        return req

    def update_weights_from_disk(self, model_path: str, abort_all_requests: bool = False,
                                 flush_cache: bool = True, weight_version: str | None = None) -> tuple[bool, str]:
        """SGLang's /update_weights_from_disk (UpdateWeightFromDiskReqInput, srt/managers/io_struct.py,
        v0.5.21): load another checkpoint of the same architecture into the live weights.
        Graphs take weights as inputs, so nothing recompiles; the prefix cache (and the host
        KV tier) holds KV of the old weights, so it is flushed unless flush_cache is false."""
        live = list(self.scheduler.running) + list(self.scheduler.waiting)
        if live and not abort_all_requests:
            return False, f"{len(live)} requests are running or queued; retry with abort_all_requests"
        for r in live:
            self.abort(r)
        try:
            self.runner.reload_weights(resolve_model_path(model_path))
        except (OSError, KeyError, ValueError, RuntimeError) as e:
            return False, f"failed to load {model_path}: {e}"
        if flush_cache:
            self.flush_cache()
        self.weight_version = weight_version
        return True, f"weights updated from {model_path}"

    def flush_cache(self) -> None:
        """Drop every cached prefix (device pages and the host tier)."""
        for tier in self.host_tiers:
            tier.clear()
        for radix in self.radixes:
            offload, radix.offload = radix.offload, None
            radix.evict(radix.total_pages())
            radix.offload = offload

    def close_session(self, session_id: str) -> bool:
        """SGLang /close_session: drop the session's prefix references (nothing is freed)."""
        return self.scheduler.close_session(session_id)

    def abort(self, req: Request) -> None:
        self.scheduler.abort(req)
        self.runner.forget(req)
        if req.pd_meta is not None and not req.pd_injected:
            self.pd_release(req.pd_meta["xfer"])
        if req.handoff is not None and req.handoff_meta is None:  # tell the decode side it will not come
            req.handoff_meta = {"xfer": req.handoff[0], "rid": req.rid, "done": "abort", "parts": [], "token": None}
            self.runner.pd_send_meta(req.handoff[1], req.handoff_meta)
        if hasattr(self.proposer, "finish"):
            self.proposer.finish(req.rid, [])

    def close(self) -> None:
        """Stop tensor-parallel workers (rank 0 tells them to exit), destroy the process group and
        give this process back the state init_rank changed. Idempotent."""
        _OPEN.discard(self)
        runner = getattr(self, "runner", None)
        if runner is not None and getattr(runner, "watchdog", None) is not None:
            runner.watchdog.close()
        if runner is not None and getattr(runner, "_pd_snd", None) is not None:
            try:
                runner._pd_snd.flush(60)
            finally:
                runner._pd_snd.close()
                runner._pd_snd = None
        if getattr(self, "pd_receiver", None) is not None:
            self.pd_receiver.close()
            self.pd_receiver = None
        if self._workers and runner is not None and runner.tp_send is not None:
            runner.tp_send(None)
            runner.tp_send = None
        self._stop_tp()

    def _stop_tp(self, wait: float = 30.0) -> None:
        """Workers that did not exit within `wait` seconds (and every worker of an engine whose
        construction failed, which no rank 0 will ever release) are terminated: a daemon worker
        otherwise lives as long as the process, blocked in a rendezvous or a broadcast (measured: a
        worker of a failed tests/test_attention_tp.py engine still waiting for its store an hour
        later, kiln-cf-4, 2026-10-04)."""
        import torch.distributed as dist

        released = getattr(self, "runner", None) is not None and self.runner.tp_send is None
        for w in self._workers:
            if w.pid is not None:
                w.join(timeout=wait if released else 0)
                if w.is_alive():
                    w.terminate()
                    w.join(timeout=10)
        self._workers = []
        if self._tp_saved is not None:
            if dist.is_initialized():
                dist.destroy_process_group()
            from . import tp

            tp.restore_process_state(self._tp_saved)
            self._tp_saved = None

    def warmup(self, reverse: bool = False) -> dict:
        role = self.cfg.pd_role  # a prefill engine never decodes; a decode engine prefills only on bypass
        out = self.runner.warmup(1 + self.cfg.spec_k if self.cfg.spec_method else None, reverse=reverse,
                                 decode=role != "prefill", prefill=role != "decode" or self.cfg.pd_bypass_prefill)
        self.runner.eplb_reset()  # warmup's dummy prefill calls were recorded too (models/eplb.py)
        out["cache_pulled"] = self.cache_pulled
        if self.cfg.compile_cache_uri and self.cfg.device == "neuron":
            from .. import compile_cache

            out["cache_pushed"] = compile_cache.push(self.cfg.compile_cache_uri)
        return out

    def _propose(self, req: Request) -> list[int]:
        # Grammar and penalties depend on every earlier token, so their requests decode
        # one token at a time for now.
        if req.host_bound:
            return []
        room = min(req.params.max_new_tokens - len(req.output_ids), self.cfg.max_model_len - req.num_tokens) - 1
        k = min(self.cfg.spec_k, room)
        if self.cfg.spec_k_per_batch_size:  # the draft length for this many running requests
            n = len(self.scheduler.running)
            k = min(k, next((kk for lo, hi, kk in self.cfg.spec_k_per_batch_size if lo <= n <= hi), self.cfg.spec_k))
        if self.cfg.spec_method == "mtp" and self.runner.spec_async:
            # engine/spec_async.py: the drafts (and the newest token, positions, state row) are on the device; the
            # verify always carries k of them, and tokens past the request's limits are dropped at commit.
            return [BLIND] * self.cfg.spec_k
        if self.cfg.spec_method == "mtp":
            draft, req.mtp_draft = req.mtp_draft, []
            return draft[: max(k, 0)]
        if hasattr(self.proposer, "propose_for"):  # stateful per request (suffix decoding)
            return self.proposer.propose_for(req.rid, req.token_ids, k)
        return self.proposer.propose(req.token_ids, k)

    def has_work(self) -> bool:
        return self.scheduler.has_work() or self._inflight is not None

    def step(self) -> list[Request]:
        """Run one scheduler step. Returns requests that finished in it.

        With `overlap`, step N+1 is scheduled and launched before step N's tokens are read,
        so host scheduling hides behind device time; its inputs come from the on-device
        token board. A step whose requests need every previous token on the host (grammar,
        penalties, speculative drafts) or that must preempt runs synchronously instead.
        """
        t = time.perf_counter()
        self.last_step = StepStats()
        if self.pd_role == "decode":
            if self.pd_hold and self._inflight is None:
                # Nothing is in flight, so nothing can still write what the hold keeps. An engine that went idle never
                # reads a step back again, and without this its held pages and slots (every slot, when its last
                # requests ended together) were never released: handoffs then waited forever (seen on trn2).
                self._pd_cool()
            self._pd_gate()
        if self.cfg.overlap and self._overlap_ok():
            finished = self._step_overlap()
        else:
            finished = self._drain() + self._step_sync()
        self.last_step.seconds = time.perf_counter() - t
        if profiling.TIMELINE is not None:  # KILN_TIMELINE: what this step() call launched
            st = self.last_step
            profiling.record("step", t, t + st.seconds, st.num_decode, st.num_prefill_tokens, st.decode_calls,
                             st.prefill_calls)
        self._eplb_tick()  # after the step's own time (outside last_step.seconds)
        return finished

    def _eplb_tick(self) -> None:
        """KILN_EPLB_INTERVAL (models/eplb.py): every that many prefill graph calls, rebalance the redundant expert
        slots from the counts recorded since the last rebalance, at most KILN_EPLB_MAX_REBALANCES times (0: no
        limit). Non-blocking: ModelRunner.eplb_prepare starts the slots' loading on every rank's host thread and
        each later step polls eplb_commit, which installs them once every rank is done (the only pause is that
        commit: the calls in flight finish and the staged slots are copied in). Each one is logged in eplb_log:
        (prefill calls when it started, slots moved over all ranks, seconds from start to commit, the commit's
        pause in seconds, slots this rank reloaded)."""
        every = self._eplb_every
        if not every:
            return
        self._eplb_calls += self.last_step.prefill_calls
        self._eplb_total += self.last_step.prefill_calls
        if self._eplb_pending is not None:
            moved = self.runner.eplb_commit()
            if moved is not None:
                t0, at = self._eplb_pending
                self._eplb_pending = None
                r = self.runner
                self.eplb_log.append((at, moved, time.perf_counter() - t0, getattr(r, "eplb_commit_seconds", 0.0),
                                      getattr(r, "eplb_loaded", 0)))
                print(f"eplb: rebalance started after {at} prefill calls committed, {moved} slots moved, "
                      f"{self.eplb_log[-1][2]:.2f} s to commit, paused {self.eplb_log[-1][3]:.3f} s", flush=True)
            return
        if self._eplb_calls < every or (self._eplb_max and len(self.eplb_log) >= self._eplb_max):
            return
        self._eplb_calls = 0
        self._eplb_pending = (time.perf_counter(), self._eplb_total)
        self.runner.eplb_prepare()

    def _overlap_ok(self) -> bool:
        if self.cfg.spec_method and not self.runner.spec_async:
            return False
        if getattr(self.mcfg.hybrid, "ple", None) is not None:  # n-gram ids are hashed from host tokens
            return False
        live = list(self.scheduler.running) + list(self.scheduler.waiting)
        return not any(r.host_bound for r in live)

    def _step_sync(self) -> list[Request]:
        t0 = time.perf_counter()
        plan = self.scheduler.schedule()
        if not plan:
            return []
        t1 = time.perf_counter()
        launches = self._launch(plan)
        t2 = time.perf_counter()
        tokens = self._collect(plan, launches)
        t3 = time.perf_counter()
        self._verified_states(launches)
        finished = self._finish(plan, self.scheduler.update(plan, tokens), tokens)
        t4 = time.perf_counter()
        if self.cfg.spec_method == "mtp" and not self.runner.spec_async:
            self._mtp_draft(launches)
            self.mtp_seconds += time.perf_counter() - t4
        if STEP_TIMES is not None:  # KILN_PROFILE_STEP=1: wall per phase, by the kinds the step launched
            kinds = "+".join(sorted({k for _, _, k, _, _ in launches}))
            for phase, dt in (("schedule", t1 - t0), ("launch", t2 - t1), ("collect", t3 - t2), ("finish", t4 - t3),
                              ("draft", time.perf_counter() - t4)):
                STEP_TIMES[(kinds, phase)].append(dt)
        return finished

    def _verified_states(self, launches) -> None:
        """Recurrent models after a verify: the state after the last computed position (the newest
        token and the accepted drafts) is the one the request continues from, and a decode
        checkpoint crossed on the way is copied out of the row that holds it."""
        state = self.runner.state
        if state is None:
            return
        saves = ([], [], [])
        for chunk, _, kind, _, _ in launches:
            if kind != "verify":
                continue
            for s in chunk:
                n = self._new_count.get(id(s), 1)  # positions s.start .. s.start + n - 1 were computed
                rows = state.rows_of(s.req)
                state.set_current(s.req, n - 1)
                sch = self.scheduler.groups[s.req.dp_group] if self.dp > 1 else self.scheduler
                if sch.cfg.ckpt_track and sch.cfg.prefix_cache:
                    row = sch.track(s.req, s.start, n)
                    if row is not None:
                        b = s.req.ckpt_pending[-1][0]
                        saves[0].append(rows[b - s.start - 1])
                        saves[1].append(row)
                        saves[2].append(s.req.dp_group)
        self.runner.copy_state(*saves)

    def _mtp_draft(self, launches) -> None:
        """Draft the next step's tokens with the MTP layer from the hidden states this step's
        graphs left on the device. Every computed position also gets its MTP KV (vLLM's MTP
        positions: position p takes the token at p + 1 and the target hidden at p). Every MTP graph
        of the step is launched before the first is read back. A prefill launch whose chunks are
        all non-final already had its MTP pass launched behind it (_mtp_fill_early)."""
        pending = []
        for chunk, _, kind, hkey, lay in launches:
            if kind == "prefill" and id(chunk) in self._mtp_early:
                continue
            got = self._mtp_rows(chunk, kind, lay)
            if got is not None:
                rows, keep, batch = got
                pending.append((rows, keep, self.runner.mtp_launch(rows, hkey, self.cfg.spec_k, batch,
                                                                   prefill=kind == "prefill")))
        self._mtp_early = set()
        for rows, keep, launched in pending:
            for (r, *_), d, use in zip(rows, self.runner.mtp_read(launched), keep):
                if use:
                    r.mtp_draft = d

    def _mtp_rows(self, chunk, kind, lay):
        """mtp_drafts rows (req, first position, ids, hidden rows) of one launch, which rows' drafts to
        keep, and the producing call's per-group count; None when no sequence drafts."""
        k, Q = self.cfg.spec_k, 1 + self.cfg.spec_k
        rows, keep = [], []
        for i, s in enumerate(chunk):
            r = s.req
            if r.status is not Status.RUNNING or r.host_bound:
                continue
            # The sequence's first row in the call's hidden states (DP attention: lay places it).
            base = lay[i][1] if lay is not None else (i * Q if kind == "verify" else i)
            if kind == "verify":
                n = self._new_count.get(id(s), 1)
                ids, hrows = r.token_ids[s.start + 1 : s.start + 1 + n], [base + j for j in range(n)]
            else:  # decode (one row per sequence) or a prefill chunk (one sequence per group)
                ids = r.token_ids[s.start + 1 : s.end + 1]
                hrows = [base + j for j in range(len(ids))]
            if not ids or not self.scheduler._reserve(r, r.num_tokens + k):
                continue
            rows.append((r, s.start, ids, hrows))
            keep.append(s.sample)  # a non-final prefill chunk only fills MTP KV
        if not rows:
            return None
        batch = 1 if kind == "prefill" else self.runner._place([s.req for s in chunk])[1]
        return rows, keep, batch

    def _mtp_fill_early(self, chunk, lay) -> None:
        """A prefill launch whose chunks are all non-final: its MTP pass (which only fills MTP KV, the
        tokens after each position being prompt tokens) needs nothing from the host, so it is launched
        right behind the prefill and runs while the host reads and finishes the step. Nothing reads its
        drafts back."""
        if self.cfg.spec_method != "mtp" or any(s.sample for s in chunk):
            return
        got = self._mtp_rows(chunk, "prefill", lay)
        if got is not None:
            self.runner.mtp_launch(got[0], self.runner.last_hidden, self.cfg.spec_k, 1, prefill=True)
        self._mtp_early.add(id(chunk))

    def _step_overlap(self) -> list[Request]:
        self.scheduler.in_flight = self._inflight is not None
        t_sched = time.perf_counter()
        try:
            plan = self.scheduler.schedule()
        except NeedSync:
            return self._drain()
        if profiling.TIMELINE is not None:
            profiling.record("sched", t_sched, time.perf_counter())
        if not plan:
            return self._drain()
        launches = self._launch(plan)
        self.scheduler.advance(plan)
        finished = self._drain()  # read step N while the device runs step N+1
        self._inflight = (plan, launches)
        self.scheduler.in_flight = True
        return finished

    def _drain(self) -> list[Request]:
        if self._inflight is None:
            return []
        plan, launches = self._inflight
        self._inflight = None
        self.scheduler.in_flight = False
        tokens = self._collect(plan, launches)
        return self._finish(plan, self.scheduler.commit(plan, tokens), tokens)

    def _launch(self, plan):
        if self.pd_role == "decode":
            self._pd_inject_plan(plan)
        if self.runner.mixed_rows and plan.prefills:
            return self._launch_mixed(plan)
        if self.dp > 1:
            return self._launch_dp(plan)
        max_b = self.runner.decode_buckets[-1]
        merged = self.cfg.spec_merged  # draftless sequences run in the verify graph (EngineConfig.spec_verify_plain)
        plain = [s for s in plan.decodes if not s.draft and not merged]
        spec = [s for s in plan.decodes if s.draft or merged]
        # (seqs, device output, kind, hidden states, layout); all launched before the first read-back
        launches = []
        self._restore_states(plan)
        for i in range(0, len(plain), max_b):
            chunk = plain[i : i + max_b]
            launches.append((chunk, self.runner.decode(chunk), "one", self.runner.last_hidden, None))
            self._save_states(chunk)
        Q = 1 + self.cfg.spec_k
        for i in range(0, len(spec), max_b):
            chunk = spec[i : i + max_b]
            if self.runner.spec_async:
                launches.append((chunk, self.runner.verify_async(chunk, Q), "verify_async", self.runner.last_hidden,
                                 self.runner.last_layout))
                self.runner.mtp_async()
                continue
            launches.append((chunk, self.runner.verify(chunk, Q), "verify", self.runner.last_hidden, None))
        for s in plan.prefills:
            launches.append(([s], self.runner.prefill(s), "prefill", self.runner.last_hidden, None))
            if self.runner.spec_async:
                self.runner.spec_after_prefill([s], self.runner.last_hidden, None)
            else:
                self._mtp_fill_early(launches[-1][0], None)
            self._save_states([s])
        self._step_stats(plan, launches)
        return launches

    def _step_stats(self, plan, launches) -> None:
        """A mixed call (prefill chunks with decode rows riding along) counts as a prefill call."""
        self.last_step = StepStats(len(plan.decodes), sum(s.num_tokens for s in plan.prefills),
                                   decode_calls=sum(k in ("one", "verify") for _, _, k, _, _ in launches),
                                   prefill_calls=sum(k in ("prefill", "mixed") for _, _, k, _, _ in launches),
                                   num_restored=sum(s.restore is not None for s in plan.prefills),
                                   prefill_chunks=len(plan.prefills), deferred_chunks=plan.deferred)

    def _restore_states(self, plan) -> None:
        """Prefix hits of recurrent models: copy each hit's checkpoint into the request's row before
        any of the step's calls."""
        if self.runner.state is None:
            return
        hits = [s for s in plan.prefills if s.restore is not None]
        self.runner.copy_state([s.restore for s in hits], [self.runner.state.row(s.req) for s in hits],
                               [s.req.dp_group for s in hits])
        for s in hits:
            s.req.ckpt_restore = None

    def _save_states(self, seqs) -> None:
        """Checkpoints of recurrent models: copy each entry's state row after it ran (scheduler
        ScheduledSeq.save), before any later call writes the row again."""
        saves = [s for s in seqs if s.save is not None]
        if saves:
            self.runner.copy_state([self.runner.state.row(s.req) for s in saves], [s.save for s in saves],
                                   [s.req.dp_group for s in saves])

    def _launch_dp(self, plan):
        """DP attention: every call carries every group's sequences, up to one bucket per group
        (one chunk per group for a prefill), so a step makes as many calls of each kind as its
        busiest group needs and a group with nothing of that kind runs padding."""
        N, max_b, Q = self.dp, self.runner.decode_buckets[-1], 1 + self.cfg.spec_k
        launches = []
        self._restore_states(plan)
        merged = self.cfg.spec_merged  # draftless sequences run in the verify graph (EngineConfig.spec_verify_plain)
        for kind, seqs, per in (("one", [s for s in plan.decodes if not s.draft and not merged], max_b),
                                ("verify", [s for s in plan.decodes if s.draft or merged], max_b),
                                ("prefill", plan.prefills, 1)):
            groups = [[] for _ in range(N)]
            for s in seqs:
                groups[s.req.dp_group].append(s)
            for i in range(0, max(len(g) for g in groups), per):
                chunk = [s for g in groups for s in g[i : i + per]]
                k_ = kind
                if kind == "one":
                    out = self.runner.decode(chunk)
                elif kind == "verify" and self.runner.spec_async:
                    out, k_ = self.runner.verify_async(chunk, Q), "verify_async"
                elif kind == "verify":
                    out = self.runner.verify(chunk, Q)
                else:
                    out = self.runner.prefill(chunk)
                launches.append((chunk, out, k_, self.runner.last_hidden, self.runner.last_layout))
                if k_ == "verify_async":
                    self.runner.mtp_async()
                if kind == "prefill" and self.runner.spec_async:
                    self.runner.spec_after_prefill(chunk, self.runner.last_hidden, self.runner.last_layout)
                elif kind == "prefill":
                    self._mtp_fill_early(chunk, self.runner.last_layout)
                if kind != "verify":
                    self._save_states(chunk)
        self._step_stats(plan, launches)
        return launches

    def _launch_mixed(self, plan):
        """Mixed batches (EngineConfig.mixed_batch): every prefill call carries decode rows. Each call
        takes one prefill chunk per DP-attention group, and the step's first call also takes up to
        mixed_rows decoding sequences per group (ModelRunner.mixed), so a step with prefill work makes no
        decode call at all unless a group has more decodes than that (the rest go to decode calls
        first, as an unmixed step runs them). The calls touch disjoint requests, so their order changes
        nothing; checkpoint copies follow the call that computed their state, as unmixed."""
        N, D, max_b = self.dp, self.runner.mixed_rows, self.runner.decode_buckets[-1]
        if any(s.draft for s in plan.decodes):
            raise RuntimeError("mixed batches carry plain decodes only (speculative decoding is refused at start)")
        launches = []
        self._restore_states(plan)
        pre, dec = [[] for _ in range(N)], [[] for _ in range(N)]
        for s in plan.prefills:
            pre[s.req.dp_group if N > 1 else 0].append(s)
        for s in plan.decodes:
            dec[s.req.dp_group if N > 1 else 0].append(s)
        rest = [g[D:] for g in dec]
        for i in range(0, max(len(g) for g in rest), max_b):
            chunk = [s for g in rest for s in g[i : i + max_b]]
            launches.append((chunk, self.runner.decode(chunk), "one", self.runner.last_hidden, self.runner.last_layout))
            self._save_states(chunk)
        ride = [s for g in dec for s in g[:D]]
        for i in range(max(len(g) for g in pre)):
            chunks = [g[i] for g in pre if len(g) > i]
            decs = ride if i == 0 else []
            out = self.runner.mixed(chunks, decs)
            launches.append((chunks + decs, out, "mixed", self.runner.last_hidden, self.runner.last_layout))
            self._save_states(chunks + decs)
        self._step_stats(plan, launches)
        return launches

    def _dp_parts(self, chunk, out, kind, lay):
        """A DP-attention (or mixed) call's read-back output as the (seqs, output, kind) pieces the
        single-group calls return: the decode / verify rows in chunk order, and one piece per prefill
        chunk (its sampled row, then its group's scored rows when the call scored any). A mixed call
        (ModelRunner.mixed) gives one prefill piece per chunk and one decode piece."""
        if kind == "one":
            return [(chunk, out[[l[0] for l in lay]], kind)]
        if kind == "verify":
            Q = 1 + self.cfg.spec_k
            return [(chunk, out[[l[0] + q for l in lay for q in range(Q)]], kind)]
        N = self.dp
        if kind == "mixed":
            D = self.runner.mixed_rows
            S = N + N * D
            R = (out.shape[0] - S) // N  # each group's C + D scored rows; 0 without
            C = R - D if R else 0
            parts = [([s], torch.cat([out[l[0] : l[0] + 1], out[S + l[0] * R : S + l[0] * R + C]]), "prefill")
                     for s, l in zip(chunk, lay) if not l[1]]
            decs = [(s, l) for s, l in zip(chunk, lay) if l[1]]
            if decs:
                parts.append(([s for s, _ in decs], out[[l[0] for _, l in decs]], "one"))
            return parts
        C = (out.shape[0] - N) // N  # 0 without scored rows
        return [([s], torch.cat([out[l[0] : l[0] + 1], out[l[2] : l[2] + C]]), kind) for s, l in zip(chunk, lay)]

    def _collect(self, plan, launches) -> list:
        """Read a step's outputs; returns tokens aligned with plan.seqs()."""
        Q = 1 + self.cfg.spec_k
        results: dict[int, object] = {}
        self._new_count = {}
        for chunk_, out_, kind_, _, lay in launches:
            if kind_ == "verify_async":
                self._collect_blind(chunk_, out_, lay, results)
                continue
            t_read = time.perf_counter()
            with watchdog.watched(self.runner.watchdog, f"reading back a {kind_} call"):
                out_ = out_.cpu()
            if profiling.TIMELINE is not None:  # KILN_TIMELINE: the host waiting for this call's output
                profiling.record("read", t_read, time.perf_counter(), kind_)
            for chunk, out, kind in (self._dp_parts(chunk_, out_, kind_, lay) if lay is not None
                                     else [(chunk_, out_, kind_)]):
                if kind != "verify":  # a decode batch, or one prefill chunk
                    tokens, lps, top_ids, top_lps = unpack(out, len(chunk))
                    if kind == "prefill" and out.shape[0] > 1:  # scored rows follow the sampled one
                        self._prompt_logprobs(chunk[0], out[1:])
                        self._forced_logprobs(chunk[0], out[1:])  # before the sampled token's own
                    for i, s in enumerate(chunk):
                        if s.req.watermarked and s.sample:
                            tokens[i], lps[i] = self._watermark(s.req, tokens[i], lps[i], top_ids[i], top_lps[i])
                        results[id(s)] = tokens[i]
                        n = s.req.params.logprobs
                        if s.sample and n is not None and s.req.status is Status.RUNNING:
                            s.req.logprobs.append((lps[i], top_ids[i][:n], top_lps[i][:n]))
                else:
                    for b, s in enumerate(chunk):
                        results[id(s)] = self._accept(s, out[b * Q : (b + 1) * Q])
                        self._new_count[id(s)] = len(results[id(s)])
        for xfer in plan.pd_injected:  # every rank has executed the copies: they precede the calls just read
            self.pd_release(xfer)
        plan.pd_injected = []
        if self.pd_role == "decode" and self.pd_hold:
            self._pd_cool()
        return [results[id(s)] for s in plan.seqs()]

    def _collect_blind(self, chunk, outs, lay, results) -> None:
        """A blind verify's read-back (engine/spec_async.py): spec_post's (tokens, acc) row per sequence gives the
        emitted tokens; the verify's own rows give their logprobs, chosen as _accept chooses them (the draft's for
        an accepted position, then the replacement's or the bonus token's)."""
        out, res = outs
        with watchdog.watched(self.runner.watchdog, "reading back a blind verify"):
            out, res = out.cpu(), res.cpu()
        Q = 1 + self.cfg.spec_k
        N = NUM_TOP_LOGPROBS
        for b, s in enumerate(chunk):
            i = lay[b][0] // Q if lay is not None else b
            row = res[i].tolist()
            acc, drafted = int(row[Q]), row[Q + 1] > 0.5
            new = [int(t) for t in row[: acc + 1]]
            results[id(s)] = new
            req = s.req
            if drafted:  # a draftless row (its newest token from the host, spec_host_init) is a decode, as in _accept
                self.spec_proposed += self.cfg.spec_k
                self.spec_accepted += acc
                self.spec_verifies += 1
                req.spec_proposed += self.cfg.spec_k
                req.spec_accepted += acc
                for j in range(min(acc + 1, self.cfg.spec_k)):
                    if j == len(self.spec_pos_reached):
                        self.spec_pos_reached.append(0)
                        self.spec_pos_accepted.append(0)
                    self.spec_pos_reached[j] += 1
                    self.spec_pos_accepted[j] += j < acc
            n = req.params.logprobs
            if n is not None and req.status is Status.RUNNING:
                rows = out[i * Q : (i + 1) * Q].tolist()
                for j in range(acc + 1):
                    r_ = rows[j]
                    # the draft's (accepted), the replacement's, the bonus token's; a draftless row's is y_0's
                    lp = r_[5] if j < acc else (r_[4] if drafted and acc < self.cfg.spec_k else r_[3])
                    req.logprobs.append((lp, [int(x) for x in r_[6 : 6 + n]], r_[6 + N : 6 + N + n]))

    def _watermark(self, req, tok, lp, top_ids, top_lps):
        """Replace the device's sample with a watermarked one (engine/watermark.py)."""
        ctx = self.watermark.context(req.output_ids, req.wm_seen)
        if ctx is None:
            return tok, lp
        req.wm_seen.add(ctx)
        p = req.params
        new = self.watermark.choose(ctx, top_ids, top_lps, p.temperature, p.top_p, p.top_k, p.min_p)
        return new, top_lps[top_ids.index(new)]

    @staticmethod
    def _prompt_logprobs(s, rows) -> None:
        req = s.req
        span = req.prompt_logprob_rows
        if span is None:
            return
        ranks, lps, top_ids, top_lps = unpack(rows, s.num_tokens)
        n = req.params.prompt_logprobs
        for i, q in enumerate(range(s.start, s.end)):
            if span[0] <= q < span[1]:
                req.prompt_logprobs[q + 1] = (lps[i], int(ranks[i]), top_ids[i][:n], top_lps[i][:n])

    @staticmethod
    def _forced_logprobs(s, rows) -> None:
        """Logprobs of forced (jump-forward) tokens from the rows that score them: raw, like a
        sampled token's (sampler.py), appended in output order. SGLang does the same: a jumped
        request's next extend reports its forced tokens' input logprobs as output logprobs
        (v0.4.3.post2 Scheduler.add_logprob_return_values, last_update_decode_tokens)."""
        req = s.req
        span = req.forced_logprob_rows
        if span is None or req.status is not Status.RUNNING:
            return
        _, lps, top_ids, top_lps = unpack(rows, s.num_tokens)
        n = req.params.logprobs
        for i, q in enumerate(range(s.start, s.end)):
            if q == req.num_prompt + len(req.logprobs) - 1 and q < span[1]:
                req.logprobs.append((lps[i], top_ids[i][:n], top_lps[i][:n]))

    def _finish(self, plan, finished: list[Request], tokens) -> list[Request]:
        seqs = plan.seqs()
        for s in seqs:
            # commit() stops at the first finishing token; drop logprobs past it.
            if s.draft and s.req.params.logprobs is not None:
                del s.req.logprobs[len(s.req.output_ids):]
        self._advance_thinking(seqs)
        finished += self._check_stop_strings(seqs)
        finished += self._advance_grammars(seqs, tokens)
        if self.cfg.jump_forward and self.runner.grammar is not None:
            finished += self._jump_forward(seqs)
        if self.pd_role == "prefill":
            for r in finished:
                if r.handoff is not None and r.handoff_meta is None:
                    self._pd_handoff(r)
        for r in finished:
            self.runner.forget(r)
            if hasattr(self.proposer, "finish"):
                self.proposer.finish(r.rid, r.output_ids if r.finish_reason != "abort" else [])
        return finished

    def _accept(self, s, rows) -> list[int]:
        """Walk a verified draft: keep accepted tokens, then the replacement for the first
        rejected one, or the bonus token after a fully accepted draft."""
        req, draft = s.req, s.draft
        N = NUM_TOP_LOGPROBS
        rows = rows.tolist()
        new, lps = [], []
        for i, d in enumerate(draft):
            row = rows[i]
            if i == len(self.spec_pos_reached):
                self.spec_pos_reached.append(0)
                self.spec_pos_accepted.append(0)
            self.spec_pos_reached[i] += 1
            self.spec_pos_accepted[i] += row[1] > 0.5
            if row[1] > 0.5:
                new.append(d)
                lps.append((row[5], row))
            else:
                new.append(int(row[2]))
                lps.append((row[4], row))
                break
        else:
            row = rows[len(draft)]
            new.append(int(row[0]))
            lps.append((row[3], row))
        req.spec_proposed += len(draft)
        req.spec_accepted += len(new) - 1
        self.spec_proposed += len(draft)
        self.spec_accepted += len(new) - 1
        self.spec_verifies += bool(draft)  # a draftless row in the verify graph (spec_verify_plain) is a decode
        n = req.params.logprobs
        if n is not None:
            for lp, row in lps:
                req.logprobs.append((lp, [int(x) for x in row[6 : 6 + n]], row[6 + N : 6 + N + n]))
        return new

    def _advance_thinking(self, seqs) -> None:
        """Count reasoning tokens of requests with a thinking budget and, at the budget, queue
        the reasoning end string to be forced one token per step (runner._bitmask allows only
        the next forced token). Counting starts after the reasoning start string, as vLLM."""
        start, end = self.think_start_ids, self.think_end_ids
        for s in seqs:
            r = s.req
            if r.params.thinking_token_budget is None or r.think_done:
                continue
            out = r.output_ids
            for i in range(r.think_seen, len(out)):
                if r.think_force:
                    if out[i] != r.think_force[0]:
                        raise RuntimeError(f"{r.rid}: forced token {r.think_force[0]} not sampled ({out[i]})")
                    r.think_force.pop(0)
                    if not r.think_force:
                        r.think_done = True
                        break
                    continue
                if not r.think_in:
                    r.think_in = out[max(0, i + 1 - len(start)) : i + 1] == start
                    continue
                close = self.think_close_ids
                if out[max(0, i + 1 - len(close)) : i + 1] == close:  # the model ended it itself
                    r.think_done = True
                    break
                r.think_count += 1
                if r.think_count >= r.params.thinking_token_budget:
                    r.think_force = list(end)
            r.think_seen = len(out)

    def _advance_grammars(self, seqs, tokens) -> list[Request]:
        done = []
        for s, tok in zip(seqs, tokens):
            m = s.req.matcher
            if m is None or not s.sample:
                continue
            # A finished request (EOS / length) may still advance; the token was sampled
            # under this matcher's own mask, so accept cannot fail unless that mask was not
            # applied, which is a bug worth failing loudly on.
            if not m.is_terminated() and not m.accept_token(tok):
                raise RuntimeError(f"{s.req.rid}: token {tok} rejected by its own grammar")
            if m.is_terminated() and s.req.status is Status.RUNNING:
                self.scheduler.finish_now(s.req, "stop")
                done.append(s.req)
        return done

    def _jump_forward(self, seqs) -> list[Request]:
        """Jump-forward decoding (SGLang v0.4.3.post2, ScheduleBatch.check_for_jump_forward):
        after a grammar request's newest token, the tokens of the one string its grammar allows
        next (GrammarBackend.jump_forward) are appended to its output. The scheduler computes
        them as a prefill chunk in the next step, whose last row samples the token after them,
        so a run of n forced tokens costs one step instead of n. Returns requests a forced
        token finished.

        SGLang moves a jumped request back to the waiting queue (its old sequence cached in the
        radix tree, then re-matched); here it stays running and its next chunk starts where its
        KV ends. SGLang checks max_new_tokens only on the token sampled after the run
        (Req.check_finished), so a run can overshoot it; here a forced token finishes the
        request exactly as a sampled one would (Scheduler.commit: EOS / stop ids,
        max_new_tokens, max_model_len; then stop strings), so the output is cut where
        token-by-token decoding would cut it."""
        done = []
        grammar = self.runner.grammar
        for s in seqs:
            r = s.req
            m = r.matcher
            if (m is None or not s.sample or r.status is not Status.RUNNING or m.is_terminated()
                    or not r.is_decoding or r.think_force
                    or (r.params.thinking_token_budget is not None and not r.think_done)):
                continue
            L = r.num_tokens
            lp = r.params.logprobs is not None
            # Replacing the newest token re-scores it from row L - 2, whose KV is then
            # recomputed in place: only where the radix tree does not own the page, and never for
            # a recurrent model, whose state cannot step back past position L - 2.
            replace_ok = not lp or (not self.recurrent and L - 2 >= self.scheduler.shared_tokens(r))
            res = grammar.jump_forward(m, r.token_ids[-1], self._encode, replace_ok)
            if res is None:
                continue
            replace, toks = res
            base = L - 1 if replace else L  # tokens kept before the forced ones
            # Per token, in the order a sampled one meets them: commit's checks, then stop strings.
            reason, stops = None, r.params.stop
            width = max((len(x) for x in stops), default=0) + 8
            for j, t in enumerate(toks):
                reason = self.scheduler._finish_reason(r, t, length=base + j + 1)
                if reason is None and stops:
                    tail = self.tokenizer.decode((r.token_ids[r.num_prompt : base] + toks[: j + 1])[-width:])
                    reason = "stop" if any(x in tail for x in stops) else None
                if reason is not None:
                    toks = toks[: j + 1]
                    break
            if reason is not None and lp:
                continue  # their logprobs need the prefill a finished request never runs
            # The matcher follows the tokens (XGrammarGrammar.jump_and_retokenize): roll the
            # replaced token back, then accept the forced ones. GrammarBackend.jump_forward has
            # accepted them on a fork already, so a refusal here is a bug.
            if replace:
                m.rollback(1)
            for t in toks:
                if not m.accept_token(t):
                    raise RuntimeError(f"{r.rid}: forced token {t} rejected by its own grammar")
            del r.token_ids[base:]
            r.token_ids.extend(toks)
            r.num_forced += len(toks)
            self.jump_forward_tokens += len(toks)
            if reason is not None:
                self.scheduler.finish_now(r, reason)
                done.append(r)
                continue
            r.forced_end = r.num_tokens
            if lp:
                if replace:
                    del r.logprobs[base - r.num_prompt :]
                # The first row to score is the one before the first token without a logprob.
                r.num_computed = min(r.num_computed, r.num_prompt + len(r.logprobs) - 1)
        return done

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _check_stop_strings(self, seqs) -> list[Request]:
        done = []
        for s in seqs:
            req = s.req
            stops = req.params.stop
            if not stops or not s.sample or req.status is not Status.RUNNING:
                continue
            # Only the tail can newly contain a stop string; decode enough of it to cover
            # the longest one plus multi-token characters.
            tail = self.tokenizer.decode(req.output_ids[-(max(len(x) for x in stops) + 8):])
            if any(x in tail for x in stops):
                self.scheduler.finish_now(req, "stop")
                done.append(req)
        return done

    def generate(self, prompts: list[list[int]], params: SamplingParams | list[SamplingParams]) -> list[Request]:
        plist = params if isinstance(params, list) else [params] * len(prompts)
        reqs = [self.add_request(p, sp) for p, sp in zip(prompts, plist)]
        while self.has_work():
            self.step()
        return reqs
