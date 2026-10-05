# Changelog

All notable changes to Kiln are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and Kiln follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) in its 0.y.z form: a minor release
adds a model family, changes a default or breaks a configuration; a patch release fixes.

## [0.1.0] - 2026-10-05

The first public release.

Kiln is an LLM inference engine for AWS Trainium. It serves large open models without the
managed Neuron inference layer (NxD Inference, vllm-neuron): it keeps the Neuron driver,
runtime, compiler and collectives, and brings its own scheduler, paged KV cache, radix prefix
cache, NKI kernels and an OpenAI-compatible server. Every graph can be compiled ahead of time on
CPU hosts, so a Trainium box serves from a compile cache instead of compiling for hours.

### Added

- OpenAI-compatible HTTP server with streaming, tool-call and reasoning parsers, and a
  cache-aware router over data-parallel replicas.
- Continuous batching with chunked prefill, overlap scheduling, priority and cache-aware
  admission, and opt-in mixed prefill + decode batches.
- Paged KV cache with FP8, radix prefix caching (also for linear-attention models, from
  recurrent-state checkpoints) and a host-memory KV tier.
- Tensor parallelism, DP attention, expert parallelism for MoE and sequence-parallel prefill
  streams.
- NKI kernels for MoE prefill and decode (128x128-block FP8 scales, expert parallel), KDA
  linear attention, DSA top-k selection and sparse decode attention.
- Speculative decoding: MTP heads (DeepSeek-V3 / V3.2, GLM-5.3), n-gram and suffix drafting.
- Piecewise layer-group graphs, a CPU compile farm (`tools/compile_farm.py`) and an HBM
  estimator for which configurations load.
- A container image on the AWS Neuron vLLM inference image (Neuron SDK 2.32); see the README's
  Container section.

### Measured

zai-org/GLM-5.3-Flash (45 layers, 288 experts, FP8, real weights) on one trn1.32xlarge, 8,192
tokens in and 256 out per request, a closed loop at the stated concurrency, 128 requests per
level, tensor parallel 32, DP attention 4, Neuron SDK 2.32, every graph from the compile farm,
measured 2026-10-05 on development commit engine-v0 ebe237e (the `kiln` package of this release
is that code with only its version string changed). Kiln is priced at trn1.32xlarge spot,
$2.15/h. The reference is vLLM (the `vllm/vllm-openai:glm53-flash` image) on a SageMaker
ml.p5en.48xlarge endpoint (8 x H200, TP 4 / DP 2 / EP 8, 311 / 842 / 1,458 output tok/s at
concurrency 16 / 32 / 64), priced at the p5en.48xlarge spot band of $28.77-30.29/h: spot
against spot.

| concurrency | Kiln output tok/s | Kiln $ / 1M output tokens | vLLM on 8 x H200, $ / 1M output tokens | Kiln |
|---:|---:|---:|---:|---|
| 16 | 87.3 | $6.84 | $25.7-27.1 | 73-75% lower |
| 32 | 112.3 | $5.32 | $9.49-9.99 | 44-47% lower |
| 64 | 122.9 | $4.86 | $5.48-5.77 | 11-16% lower |
| 64, opt-in decode kernels | 137.1 | $4.36 | $5.48-5.77 | 20-24% lower |
| 64, opt-in mixed batches | 131.8 | $4.53 | $5.48-5.77 | 17-21% lower |

The default rows use Kiln's defaults. Opt-in decode kernels:
`KILN_KDA_DECODE_KERNEL=nki KILN_DSA_DECODE_KERNEL=nki KILN_DECODE_SP=1`. Mixed batches:
`KILN_MIXED_BATCH=1` with `--state-checkpoints 4`. The two opt-ins were not measured together.

Against model-provider list prices for the same request ($0.15 per 1M input tokens and $0.50
per 1M output tokens, $0.00136 per request), Kiln costs $0.00124 per request at concurrency 64
with its defaults (8% below) and $0.00111 with the decode kernels (18% below). Discounted
providers are still 1.6-1.8x cheaper than Kiln.

Commands, logs and the iteration history: `docs/price-performance.md`.

### Known limitations (measured, not won)

- Latency. At concurrency 64 the median first token takes 6.9 s and each next token about
  452 ms (410 ms with the decode kernels), against 634 ms and 32 ms for vLLM on H200; across
  concurrency 16-64 Kiln's median inter-token latency is 149-452 ms against 20-32 ms. The win
  is cost per token for throughput workloads.
- Concurrency 128 does not fit in trn1's 16 GiB per NeuronCore.
- trn2.48xlarge reaches 189.9 output tok/s at concurrency 128, which at its spot price
  ($15.09/h) is $22.07 per 1M output tokens, against $3.4-3.6 for the H200 at the same
  concurrency.
- On-demand pricing. trn1.32xlarge on-demand is ten times its spot price, so no level is won
  on-demand against SageMaker on-demand.
- GLM-5.3-Flash's graphs take hours to compile on the device; compile them on CPU hosts first
  (`tools/compile_farm.py`, "How to resume" in `docs/price-performance.md`).

[0.1.0]: https://github.com/foxl-ai/kiln/releases/tag/v0.1.0
