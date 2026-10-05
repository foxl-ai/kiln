"""python -m kiln --model Qwen/Qwen3-0.6B --device neuron --port 8000"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    from .server.parsers import REASONING_PARSERS, TOOL_PARSERS

    ap = argparse.ArgumentParser(prog="kiln")
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--kv-cache-gb", type=float, default=4.0)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-prefill-tokens", type=int, default=512)
    ap.add_argument("--schedule-policy", default="fcfs", choices=["fcfs", "lpm", "spf", "priority"])
    ap.add_argument("--eviction-policy", default="lru", choices=["lru", "lfu", "slru", "priority", "tlru"])
    ap.add_argument("--radix-eviction-policy-config", default=None,
                    help='JSON of the policy\'s own keys, e.g. {"protected_threshold": 4} (SGLang)')
    ap.add_argument("--max-num-queued-reqs", type=int, default=None)
    ap.add_argument("--max-num-queued-tokens", type=int, default=None)
    ap.add_argument("--reasoning-parser", default=None, choices=sorted(REASONING_PARSERS))
    ap.add_argument("--tool-call-parser", default=None, choices=sorted(TOOL_PARSERS))
    ap.add_argument("--spec-method", default=None, choices=["ngram", "suffix", "mtp"])
    ap.add_argument("--spec-k", type=int, default=4)
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel ranks (one NeuronCore each)")
    ap.add_argument("--attention-tp", type=int, default=None,
                    help="tensor parallelism of the attention / linear-attention heads, a divisor of --tp "
                         "(default: the largest one the head counts allow); the MLP and experts keep --tp")
    ap.add_argument("--dp-attention", type=int, default=1,
                    help="DP attention: attention groups of --tp / N ranks, each serving its own requests with its "
                         "own KV pages (SGLang --enable-dp-attention --dp-size N); the MLP and experts keep --tp")
    ap.add_argument("--hicache-host-gb", type=float, default=0.0, help="host KV tier per rank, GB (0 = off)")
    ap.add_argument("--overlap", action="store_true", help="overlap scheduling")
    ap.add_argument("--jump-forward", action="store_true",
                    help="jump-forward decoding for structured outputs (append grammar-forced tokens as a prefill)")
    ap.add_argument("--piecewise", action="store_true", help="one compiled graph per layer kind")
    ap.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8"])
    ap.add_argument("--weight-dtype", default="auto", choices=["auto", "bf16", "fp8", "fp8-experts"],
                    help="auto: keep a quantized checkpoint's FP8 / MXFP4; bf16: dequantize at load; "
                         "fp8 / fp8-experts: quantize a BF16 checkpoint (all linears / experts only) at load")
    ap.add_argument("--mxfp4-packed", action="store_true", help="keep MXFP4 experts 4-bit on the device")
    ap.add_argument("--compile-cache", default=None, help="s3:// prefix for the shared compile cache")
    ap.add_argument("--watermark-config", default=None,
                    help='JSON, e.g. {"algorithm": "gumbel", "key": 42} (vLLM); see kiln/engine/watermark.py')
    ap.add_argument("--reasoning-config", default=None,
                    help='JSON {"reasoning_start_str": ..., "reasoning_end_str": ...} (vLLM); '
                         "defaults to the reasoning parser's tags")
    args = ap.parse_args()
    import json

    rc = json.loads(args.reasoning_config) if args.reasoning_config else {}
    if args.reasoning_parser and not rc:
        fmt = REASONING_PARSERS[args.reasoning_parser]
        rc = {"reasoning_start_str": fmt.start, "reasoning_end_str": fmt.end}

    import uvicorn

    from .config import EngineConfig
    from .engine.engine import LLMEngine
    from .server.api import build_app

    cfg = EngineConfig(
        model_path=args.model, device=args.device, dtype=getattr(torch, args.dtype),
        page_size=args.page_size, kv_cache_gb=args.kv_cache_gb, max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len, max_prefill_tokens=args.max_prefill_tokens,
        schedule_policy=args.schedule_policy, eviction_policy=args.eviction_policy,
        eviction_policy_config=json.loads(args.radix_eviction_policy_config) if args.radix_eviction_policy_config else None,
        max_num_queued_reqs=args.max_num_queued_reqs, max_num_queued_tokens=args.max_num_queued_tokens,
        spec_method=args.spec_method, spec_k=args.spec_k, tp=args.tp, attention_tp=args.attention_tp,
        dp_attention=args.dp_attention, overlap=args.overlap,
        piecewise=args.piecewise, kv_cache_dtype=args.kv_cache_dtype, compile_cache_uri=args.compile_cache,
        weight_dtype=args.weight_dtype, mxfp4_packed=args.mxfp4_packed,
        hicache_host_gb=args.hicache_host_gb, jump_forward=args.jump_forward,
        reasoning_start_str=rc.get("reasoning_start_str"), reasoning_end_str=rc.get("reasoning_end_str"),
        watermark=json.loads(args.watermark_config) if args.watermark_config else None,
    )
    engine = LLMEngine(cfg)
    app = build_app(engine, args.served_model_name or args.model, reasoning_parser=args.reasoning_parser,
                    tool_call_parser=args.tool_call_parser)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
