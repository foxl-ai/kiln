<p align="center">
  <a href="https://foxl.ai"><img src="assets/readme/foxl.svg" width="64" height="64" alt="Foxl" /></a>
</p>

<h1 align="center">Kiln</h1>

<p align="center">
  <strong>An LLM inference engine for AWS Trainium.</strong><br />
  The serving techniques of vLLM and SGLang, rebuilt for a static-shape accelerator<br />
  whose graphs are compiled ahead of time.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" /></a>
  <img alt="Neuron SDK 2.32" src="https://img.shields.io/badge/Neuron%20SDK-2.32-ff9900.svg" />
  <img alt="trn1 and trn2" src="https://img.shields.io/badge/runs%20on-trn1%20%C2%B7%20trn2-232f3e.svg" />
</p>

<p align="center">
  <a href="#results">Results</a> &nbsp;·&nbsp;
  <a href="#quickstart">Quickstart</a> &nbsp;·&nbsp;
  <a href="DESIGN.md">Design</a> &nbsp;·&nbsp;
  <a href="docs/price-performance.md">Every measurement</a>
</p>

---

Kiln serves large open models on AWS Trainium without the managed Neuron inference layer
(NxD Inference, vllm-neuron). It keeps only what cannot be replaced, the driver, runtime,
compiler and collectives, and brings its own scheduler, paged KV cache, radix prefix cache,
NKI kernels and an OpenAI-compatible server.

Every graph is compiled ahead of time on a CPU compile farm, so a Trainium box starts serving
from cache instead of compiling for hours.

## Results

**GLM-5.3-Flash on one trn1.32xlarge costs 11-75% less per output token than vLLM on
8 x H200**, at concurrency 16, 32 and 64, with spot prices on both sides and Kiln's defaults.

| concurrency | Kiln, trn1.32xlarge | Kiln, $ / 1M output tokens | vLLM, ml.p5en.48xlarge (8 x H200) | vLLM, $ / 1M output tokens | Kiln |
|---:|---:|---:|---:|---:|---|
| 16 | 87.3 tok/s | **$6.84** | 311 tok/s | $25.7-27.1 | 73-75% lower |
| 32 | 112.3 tok/s | **$5.32** | 842 tok/s | $9.49-9.99 | 44-47% lower |
| 64 | 122.9 tok/s | **$4.86** | 1,458 tok/s | $5.48-5.77 | 11-16% lower |
| 64, opt-in decode kernels | 137.1 tok/s | **$4.36** | | | 20-24% lower |
| 64, opt-in mixed batches | 131.8 tok/s | **$4.53** | | | 17-21% lower |

<sub>zai-org/GLM-5.3-Flash (45 layers, 288 experts, FP8), 8,192 tokens in and 256 out per
request, closed loop at the stated concurrency, 128 requests per level, real weights, all
graphs from the compile farm. Kiln: trn1.32xlarge spot at $2.15/h, tensor parallel 32,
DP attention 4, Neuron SDK 2.32, measured 2026-10-05. vLLM: the purpose-built
`vllm/vllm-openai:glm53-flash` image on a SageMaker endpoint (TP 4 / DP 2 / EP 8), priced at the
p5en.48xlarge spot band of $28.77-30.29/h. Opt-in decode kernels:
`KILN_KDA_DECODE_KERNEL=nki KILN_DSA_DECODE_KERNEL=nki KILN_DECODE_SP=1`; mixed batches:
`KILN_MIXED_BATCH=1`. Commands, logs and the full iteration history are in
[docs/price-performance.md](docs/price-performance.md).</sub>

Against model-provider list prices for the same request ($0.15 per 1M input tokens and
$0.50 per 1M output tokens: $0.00136 per request), Kiln's cost is $0.00124 per request with
its defaults, 8% below the list price, and $0.00111 with the decode kernels, 18% below.
Discounted providers are still 1.6-1.8x cheaper.

**What it does not win, measured:**

- **Latency.** At concurrency 64 the first token takes 6.9 s at the median and each next token
  about 450 ms (410 ms with the decode kernels), against 634 ms and 32 ms for vLLM on H200. The
  win is cost per token, for throughput workloads.
- **Concurrency 128** does not fit in trn1's 16 GiB per core.
- **trn2.48xlarge** reaches 190 out tok/s at concurrency 128, but at its spot price
  ($15.09/h) that is $22 per 1M output tokens, 6.5x the H200 at the same concurrency.
- **On-demand pricing.** trn1.32xlarge on-demand is ten times its spot price, so no level is won
  on-demand against SageMaker on-demand.

## What is inside

| Layer | What Kiln does |
|---|---|
| Serving | OpenAI-compatible HTTP API with streaming, tool-call and reasoning parsers, a cache-aware router over data-parallel replicas |
| Scheduler | continuous batching, chunked prefill, decodes that never pause for prefill, overlap scheduling, priority and cache-aware admission, mixed prefill + decode batches |
| KV and state | paged KV with FP8, radix prefix caching, prefix caching for linear-attention models from recurrent-state checkpoints, a host-memory tier |
| Parallelism | tensor parallel, DP attention, expert parallelism for MoE, sequence-parallel prefill streams with group collectives |
| Kernels (NKI) | MoE prefill and decode with 128x128-block FP8 scales, expert-parallel MoE, KDA linear attention (chunked and decode), DSA top-k selection and sparse decode attention, pooled-key caches |
| Speculation | MTP heads (DeepSeek-V3 / V3.2, GLM-5.3), n-gram and suffix drafting |
| Compilation | piecewise layer-group graphs, a CPU compile farm that captures every graph a device will trace, an HBM estimator that predicts which configurations load |

The feature-by-feature comparison with the latest vLLM and SGLang is in
[FEATURES.md](FEATURES.md); the architecture and its reasoning are in [DESIGN.md](DESIGN.md);
which models run with real weights is in [docs/model-coverage.md](docs/model-coverage.md).

## What moved the number

At concurrency 64, from the first end-to-end run to the result above, each step measured on
the same box against the one before it:

| change | out tok/s |
|---|---:|
| first 8192 / 256 sweep at concurrency 64 | 19.4 |
| prefill MoE kernel on real FP8 scales, elementwise hyper-connections | 29.8 |
| 12 MoE layers per graph, prefill 2048 per step | 54.4 |
| sequence-parallel prefill streams | 67.7 |
| DSA top-k and scoring kernels, KDA kernel, prefill 4096, KV sized for every sequence | 88.3 |
| reduce-scatter prefill, separate FP8 pool-key cache, MoE tile skip | 94.9 |
| expert parallelism | 106.9 |
| per-tile FP8 dequantization in the expert-parallel kernel | 114.9 |
| attention collectives over the 8-rank DP group (the final default: 122.9) | 123.1 |
| opt-in KDA and sparse-DSA decode kernels, sequence-parallel decode streams | 137.1 |

Ideas that were measured and did not pay are recorded beside the ones that did, in
[docs/neuron-notes.md](docs/neuron-notes.md): all-to-all gathers (6x slower than a padded
all-reduce on trn1), DeepSeek-style fine-grained FP8 GEMM (4-8x the vector work of
dequantization on one vector engine), static hot-expert placement (hot sets from benchmark
prompts overlap real text's by 13%), and MTP speculative decoding on this prefill-heavy
workload (1.98 tokens per verify, still 4.6% slower end to end).

## Quickstart

Kiln runs in the PyTorch inference venv of the AWS Deep Learning AMI for Neuron, SDK 2.32
(Ubuntu 24.04), on trn1 or trn2.

```bash
git clone https://github.com/foxl-ai/kiln && cd kiln
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/activate
export PYTHONPATH=$PWD

# A small model on one NeuronCore
python -m kiln --model Qwen/Qwen3-0.6B --device neuron --port 8000

# The first request compiles the model's graphs into the local cache; later ones reuse them
curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "Trainium is", "max_tokens": 32}'
```

GLM-5.3-Flash on a trn1.32xlarge uses all 32 NeuronCores. Its graphs take hours to compile on
the device, so compile them on a CPU host first with `tools/compile_farm.py` and serve from
the cache. The exact configuration behind every result above, compile farm included, is under
"How to resume" in [docs/price-performance.md](docs/price-performance.md). The benchmark is
`bench/serve_sweep.py`.

Tests run on CPU:

```bash
KILN_TEST_MODEL=Qwen/Qwen3-0.6B NEURON_RT_VISIBLE_CORES= PYTHONPATH=. python -m pytest -q
```

## License

Apache-2.0; see [LICENSE](LICENSE). Files adapted from vLLM, SGLang and transformers keep their
own Apache-2.0 notices, listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
