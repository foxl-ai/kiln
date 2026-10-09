"""Prometheus text exposition, no client library needed.

Names follow vLLM's metric set (docs/design/metrics.md) under a `kiln:` prefix, so an
existing vLLM dashboard maps by replacing the prefix.
"""

from __future__ import annotations

import bisect
import threading

LATENCY_BUCKETS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96)


class Histogram:
    def __init__(self, buckets=LATENCY_BUCKETS):
        self.buckets = buckets
        self.counts = [0] * (len(buckets) + 1)
        self.sum = 0.0
        self.n = 0

    def observe(self, v: float) -> None:
        self.counts[bisect.bisect_left(self.buckets, v)] += 1
        self.sum += v
        self.n += 1

    def lines(self, name: str) -> list[str]:
        out, acc = [], 0
        for b, c in zip(self.buckets, self.counts):
            acc += c
            out.append(f'{name}_bucket{{le="{b}"}} {acc}')
        out.append(f'{name}_bucket{{le="+Inf"}} {self.n}')
        out.append(f"{name}_sum {self.sum}")
        out.append(f"{name}_count {self.n}")
        return out


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.prompt_tokens = 0
        self.generation_tokens = 0
        self.prefix_cache_queries = 0  # prompt tokens looked up
        self.prefix_cache_hits = 0  # of which served from the cache
        self.finished: dict[str, int] = {}
        self.ttft = Histogram()
        self.tpot = Histogram()
        self.e2e = Histogram(LATENCY_BUCKETS + (81.92, 163.84))

    def on_finish(self, req) -> None:
        with self.lock:
            n_out = len(req.output_ids)
            self.prompt_tokens += req.num_prompt
            self.generation_tokens += n_out
            self.prefix_cache_queries += req.num_prompt
            self.prefix_cache_hits += max(req.num_cached_tokens, 0)
            reason = req.finish_reason or "unknown"
            self.finished[reason] = self.finished.get(reason, 0) + 1
            if req.first_token_time is not None:
                self.ttft.observe(req.first_token_time - req.arrival_time)
                if req.finish_time is not None and n_out > 1:
                    self.tpot.observe((req.finish_time - req.first_token_time) / (n_out - 1))
            if req.finish_time is not None:
                self.e2e.observe(req.finish_time - req.arrival_time)

    def render(self, engine, loop=None) -> str:
        s = engine.scheduler
        # Every DP-attention group's pool (one without DP attention).
        pools, radixes = getattr(engine, "pools", [engine.pool]), getattr(engine, "radixes", [engine.radix])
        used = sum(p.num_usable - p.num_free for p in pools) - sum(r.evictable_pages for r in radixes)
        usable = sum(p.num_usable for p in pools)
        lines = [
            "# TYPE kiln:num_requests_running gauge", f"kiln:num_requests_running {len(s.running)}",
            "# TYPE kiln:num_requests_waiting gauge", f"kiln:num_requests_waiting {len(s.waiting)}",
            "# TYPE kiln:kv_cache_usage_perc gauge", f"kiln:kv_cache_usage_perc {used / usable:.6f}",
            "# TYPE kiln:num_preemptions_total counter", f"kiln:num_preemptions_total {s.num_preemptions}",
            "# TYPE kiln:spec_decode_num_draft_tokens_total counter",
            f"kiln:spec_decode_num_draft_tokens_total {engine.spec_proposed}",
            "# TYPE kiln:spec_decode_num_accepted_tokens_total counter",
            f"kiln:spec_decode_num_accepted_tokens_total {engine.spec_accepted}",
            "# TYPE kiln:jump_forward_tokens_total counter",
            f"kiln:jump_forward_tokens_total {engine.jump_forward_tokens}",
        ]
        lines += pd_lines(engine, loop)
        lines += startup_lines(engine)
        with self.lock:
            lines += [
                "# TYPE kiln:prompt_tokens_total counter", f"kiln:prompt_tokens_total {self.prompt_tokens}",
                "# TYPE kiln:generation_tokens_total counter", f"kiln:generation_tokens_total {self.generation_tokens}",
                "# TYPE kiln:prefix_cache_queries_total counter",
                f"kiln:prefix_cache_queries_total {self.prefix_cache_queries}",
                "# TYPE kiln:prefix_cache_hits_total counter", f"kiln:prefix_cache_hits_total {self.prefix_cache_hits}",
                "# TYPE kiln:request_success_total counter",
            ]
            lines += [f'kiln:request_success_total{{finished_reason="{k}"}} {v}' for k, v in sorted(self.finished.items())]
            for name, h in (("kiln:time_to_first_token_seconds", self.ttft),
                            ("kiln:time_per_output_token_seconds", self.tpot),
                            ("kiln:e2e_request_latency_seconds", self.e2e)):
                lines.append(f"# TYPE {name} histogram")
                lines += h.lines(name)
        return "\n".join(lines) + "\n"


def pd_lines(engine, loop=None) -> list[str]:
    """Prefill / decode disaggregation (engine/disagg.py): the engine's role, its handoff counts, the decode
    side's queue of handed-off requests and receive buffer (a part that waited for room counts in
    buffer_full_events / _seconds: backpressure is visible, never a silent retry), the per-handoff transfer
    time (first frame to complete), the copies' time on each side and the prefill side's send queue."""
    role = getattr(engine, "pd_role", None)
    if role is None:
        return []
    out = [f'kiln:pd_role{{role="{role}"}} 1']
    for k, v in sorted(getattr(engine, "pd_counts", {}).items()):
        out += [f"# TYPE kiln:pd_{k}_total counter", f"kiln:pd_{k}_total {v}"]
    r = engine.runner
    out.append(f"kiln:pd_extract_seconds_total {getattr(r, 'pd_extract_seconds', 0.0):.6f}")
    out.append(f"kiln:pd_inject_seconds_total {getattr(r, 'pd_inject_seconds', 0.0):.6f}")
    out.append(f"kiln:pd_inject_read_seconds_total {getattr(r, 'pd_inject_read_seconds', 0.0):.6f}")
    out.append(f"kiln:pd_inject_copies_total {getattr(r, 'pd_inject_copies', 0)}")
    out.append(f"kiln:pd_extract_runs_total {getattr(r, 'pd_extract_runs', 0)}")
    nx = getattr(r, "_nixl", None)
    if nx is not None:  # KILN_PD_TRANSPORT=nixl: rank 0's reads (decode) and the prefill engine's pins
        h = Histogram()
        for t in list(nx.read_times):
            h.observe(t)
        out += [f"kiln:pd_nixl_reads_total {nx.reads}", f"kiln:pd_nixl_read_bytes_total {nx.read_bytes}",
                f"kiln:pd_nixl_read_seconds_total {nx.read_seconds:.6f}", "# TYPE kiln:pd_nixl_read_seconds histogram"]
        out += h.lines("kiln:pd_nixl_read_seconds")
        pins = getattr(engine, "_pd_pins", {})
        seats = sum(g.pd_pinned for g in getattr(engine.scheduler, "groups", [engine.scheduler]))
        out += ["# TYPE kiln:pd_nixl_pins gauge", f"kiln:pd_nixl_pins {len(pins)}", f"kiln:pd_nixl_pinned_seats {seats}"]
    snd = getattr(r, "_pd_snd", None)
    if snd is not None:
        st = snd.stats
        out += [f"kiln:pd_send_bytes_total {st.bytes}", f"kiln:pd_send_frames_total {st.frames}",
                f"kiln:pd_send_queued_bytes {st.queued_bytes}", f"kiln:pd_send_blocked_seconds_total {st.blocked_seconds:.6f}",
                f"kiln:pd_send_seconds_total {st.send_seconds:.6f}", f"kiln:pd_send_errors_total {st.errors}"]
    if role == "decode":
        sch = engine.scheduler
        out += ["# TYPE kiln:pd_decode_queue gauge", f"kiln:pd_decode_queue {len(sch.prefilled)}",
                f"kiln:pd_prefilled_preemptions_total {getattr(sch, 'num_prefilled_preemptions', 0)}"]
        rcv = getattr(engine, "pd_receiver", None)
        if rcv is not None:
            st = rcv.stats
            out += ["# TYPE kiln:pd_buffer_bytes gauge", f"kiln:pd_buffer_bytes {st.held_bytes}",
                    f"kiln:pd_buffer_max_bytes {st.max_held_bytes}", f"kiln:pd_buffer_budget_bytes {rcv.budget}",
                    f"kiln:pd_buffer_full_events_total {st.buffer_full_events}",
                    f"kiln:pd_buffer_full_seconds_total {st.buffer_full_seconds:.6f}",
                    f"kiln:pd_recv_bytes_total {st.bytes}", f"kiln:pd_recv_complete_total {st.complete}",
                    f"kiln:pd_recv_refused_total {st.refused}"]
            h = Histogram()
            for t in list(st.transfer_seconds):
                h.observe(t)
            out.append("# TYPE kiln:pd_transfer_seconds histogram")
            out += h.lines("kiln:pd_transfer_seconds")
            out += [f"kiln:pd_recv_nixl_total {st.nixl}", f"kiln:pd_recv_nixl_bytes_total {st.nixl_bytes}",
                    f"kiln:pd_recv_hellos_total {st.hellos}"]
        # The prefill engine's handoff (its wall clock, in the meta) to rank 0's rows being in this engine's caches, per
        # handoff, either transport: includes the wait for admission here.
        h = Histogram()
        for t in list(getattr(engine, "pd_handoff_seconds", [])):
            h.observe(t)
        out.append("# TYPE kiln:pd_handoff_seconds histogram")
        out += h.lines("kiln:pd_handoff_seconds")
        if loop is not None:
            out += [f"kiln:pd_awaiting {len(loop.awaits)}", f"kiln:pd_arrived_unclaimed {len(loop.arrived)}",
                    f"kiln:pd_expired_total {loop.pd_expired}"]
    return out


def _bucket(key) -> str:
    """A graph key as vllm-neuron's bucket_name label (e.g. prefill_s1024): the key's parts joined by "_"."""
    return "_".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))


def startup_lines(engine) -> list[str]:
    """vllm-neuron 0.24's start-up and per-graph metrics (docs/guides/features-guide.md "Neuron-specific metrics" at
    release-0.24.0.1.1.0), under kiln: with Kiln's graph keys as the bucket label:
    - startup_time_seconds: engine construction to the end of warmup;
    - compilation_time_seconds{bucket}: a graph's first call (compile, or the load of a cached NEFF);
    - model_load_time_seconds: the shard load;
    - model_load_size_bytes: rank 0's parameters on the device;
    - neff_execution_count{bucket}: graph executions;
    - precompile_seconds: kiln/precompile.py's capture + parallel compile, when it ran."""
    runner = getattr(engine, "runner", None)
    out = []
    if getattr(engine, "startup_seconds", None) is not None:
        out += ["# TYPE kiln:startup_time_seconds gauge", f"kiln:startup_time_seconds {engine.startup_seconds:.3f}"]
    if getattr(engine, "load_seconds", None) is not None:
        out += ["# TYPE kiln:model_load_time_seconds gauge", f"kiln:model_load_time_seconds {engine.load_seconds:.3f}"]
    model = getattr(engine, "model", None)
    if model is not None and hasattr(model, "parameters"):
        n = sum(p.numel() * p.element_size() for p in model.parameters())
        out += ["# TYPE kiln:model_load_size_bytes gauge", f"kiln:model_load_size_bytes {n}"]
    pre = getattr(engine, "precompile", None)
    if pre:
        out += ["# TYPE kiln:precompile_seconds gauge", f"kiln:precompile_seconds {pre['wall_seconds']}"]
    if runner is not None:
        cs = dict(getattr(runner, "compile_seconds", {}))
        if cs:
            out.append("# TYPE kiln:compilation_time_seconds gauge")
            out += [f'kiln:compilation_time_seconds{{bucket_name="{_bucket(k)}"}} {v:.3f}' for k, v in cs.items()]
        ec = dict(getattr(runner, "exec_counts", {}))
        if ec:
            out.append("# TYPE kiln:neff_execution_count counter")
            out += [f'kiln:neff_execution_count{{bucket_name="{_bucket(k)}"}} {v}' for k, v in ec.items()]
    return out
