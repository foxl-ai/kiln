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

    def render(self, engine) -> str:
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
