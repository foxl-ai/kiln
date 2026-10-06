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
  <a href="#container">Container</a> &nbsp;·&nbsp;
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

**GLM-5.3-Flash on one trn1.32xlarge costs 35-80% less per output token than vLLM on
8 x H200**, at concurrency 16, 32 and 64, with spot prices on both sides and Kiln's defaults.

| concurrency | Kiln, trn1.32xlarge | Kiln, $ / 1M output tokens | vLLM, ml.p5en.48xlarge (8 x H200) | vLLM, $ / 1M output tokens | Kiln |
|---:|---:|---:|---:|---:|---|
| 16 | 110.5 tok/s | **$5.40** | 311 tok/s | $25.7-27.1 | 79-80% lower |
| 32 | 144.8 tok/s | **$4.12** | 842 tok/s | $9.49-9.99 | 57-59% lower |
| 64 | 167.8 tok/s | **$3.56** | 1,458 tok/s | $5.48-5.77 | 35-38% lower |
| 64, opt-in EPLB + one-piece 8192 prefill | 191.3 tok/s | **$3.12** | | | 43-46% lower |

<sub>zai-org/GLM-5.3-Flash (45 layers, 288 experts, FP8), 8,192 tokens in and 256 out per
request, closed loop at the stated concurrency, 128 requests per level, real weights, all
graphs from the compile farm. Kiln: trn1.32xlarge spot at $2.15/h, tensor parallel 32,
DP attention 4, Neuron SDK 2.32, measured 2026-10-05 on development commit engine-v0 8229c3d
(the opt-in row on feat/prefill-mfu-merge). The opt-in row needs
`KILN_EP_REDUNDANT=1` with `KILN_EPLB_INIT=<placement>` and `--eplb-rebalance`, plus
`--prefill-tokens 8192 --prefill-buckets 2048 KILN_PIECEWISE_PREFILL_MOE_GROUP=45
--kv-cache-gb 1.2 --state-checkpoints 4`. vLLM: the purpose-built
`vllm/vllm-openai:glm53-flash` image on a SageMaker endpoint (TP 4 / DP 2 / EP 8), priced at the
p5en.48xlarge spot band of $28.77-30.29/h. Commands, logs and the full iteration history are in
[docs/price-performance.md](docs/price-performance.md).</sub>

Against model-provider list prices for the same request ($0.15 per 1M input tokens and
$0.50 per 1M output tokens: $0.00136 per request), Kiln's cost is $0.000911 per request with
its defaults, 33% below the list price, and $0.000799 with EPLB and the one-piece prefill, 41%
below. Discounted providers are still 1.18-1.34x cheaper.

Two subsystems are opt-in and new in 0.2.0. **Prefill / decode disaggregation**: 3 prefill boxes
to 1 decode box sustain 930.9 out tok/s at $2.57 per 1M output tokens all-in over the middle half
of a concurrency-320 level, with ITL p50 277 ms; whole level against whole level that is $3.02
against $3.12, so on this workload it is close to a wash on cost and buys per-box throughput and
latency instead. A decode-only box reaches $0.470 per 1M output tokens at 8K context with real KV.
How it works and what it measures: [Disaggregated serving](#disaggregated-serving).
**1M context**: GLM-5.3-Flash's full 1,044,480 tokens, needle-in-a-haystack 9 / 9 at 128k / 512k /
1M on three engines, and $0.168 per 1M input tokens on trn1 at 3,549 input tokens/s per box.

**What it does not win, measured:**

- **Latency.** An idle 8K request takes 4.01 s to its first token colocated, and 1.60 s on the
  disaggregated latency-prefill configuration, against 634 ms for vLLM on H200. No configuration
  reaches a p90 of 1 s on trn1, and a long prompt takes minutes (133.0 s at 128k, 1177.1 s at 1M).
  Each next token takes 131-363 ms across concurrency 16-64 against 20-43 ms on the GPU. The win is
  cost per token, for throughput workloads.
- **Concurrency 128** does not fit in trn1's 16 GiB per core.
- **trn2.48xlarge** reaches 229.5 out tok/s at concurrency 128 and 243.9 at 256, but at its spot
  price ($15.343/h) that is $18.57 and $17.47 per 1M output tokens, 4.9x trn1's default. trn2 also
  costs more per token than trn1 at 8K on both sides of a disaggregated deployment, and expert
  parallelism hangs there at LNC=2, so trn2 keeps tensor-parallel experts.
- **Prefill is far from the hardware**: MFU 11.4% on trn1 for the best configuration, 2.8% of BF16
  peak on trn2.
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
| Long context | a long DSA path that forms nothing of the context's size per query, context parallelism over an attention group, a minimal KV layout, an fp8 index scorer: GLM-5.3-Flash's full 1,044,480 tokens |
| Disaggregation | opt-in prefill and decode engine roles, a KV handoff that can cross attention TP degrees, and a router with threshold routing, decode-credit backpressure, latency prefill engines and SLO metrics |
| Compilation | piecewise layer-group graphs, a CPU compile farm that captures every graph a device will trace, an HBM estimator that predicts which configurations load |

The feature-by-feature comparison with the latest vLLM and SGLang is in
[FEATURES.md](FEATURES.md); the architecture and its reasoning are in [DESIGN.md](DESIGN.md);
which models run with real weights is in [docs/model-coverage.md](docs/model-coverage.md).

## Disaggregated serving

Prefill reads a whole prompt in one pass and is bound by compute; decode emits one token per step
for every running request and is bound by the per-step cost of reading weights and KV. A
colocated engine runs both in one process with one set of compiled graphs, so it has to pick one
configuration for both. Kiln can instead run them as separate engines on separate boxes, connected
by a router, so each phase gets the layout it is fastest in. It is opt-in: nothing changes unless
an engine is started with a role.

**Why it pays on Trainium.** On GLM-5.3-Flash the two phases want different expert layouts, and one
engine cannot have both. With tensor-parallel experts plus three decode-only flags, the fixed part
of a decode step drops from 70.3 to 47.5 ms (-32%). The same tensor-parallel layout makes the
4096-row prefill call 475 -> 821 ms (+73%), so a colocated box that adopts it falls from 167.7 to
120.3 out tok/s at concurrency 64. Expert parallelism is the better prefill layout and tensor
parallelism the better decode layout; disaggregation lets each engine keep its own (trn1.32xlarge,
real weights; the decode steps at 1 row per DP group against engine-v0 70ddc1b, the serving
comparison against engine-v0 8229c3d, both in docs/neuron-notes.md "Decode at scale").

**What runs where.**

| Piece | What it does |
|---|---|
| Prefill engine (`--pd-role prefill`) | Loads only prefill graphs. Runs the prompt, samples the first token, and sends the request's state to the decode engine the router names. On trn1 it keeps expert parallelism, EPLB and the one-piece 8192-token prefill. |
| Decode engine (`--pd-role decode`) | Loads only decode graphs. A handed-off request goes straight to running (no prefill graph, no recompile) and its first step is an ordinary decode row. Context-parallel DSA caches and tensor-parallel experts let one box hold 96 rows per DP group. |
| Handoff | Moves every paged cache row of the prompt (MLA latent, DSA indexer and pool keys), the request's recurrent KDA state, the first token and its logprobs, the sampling parameters and the RNG state. Each sending rank ships its own shard over TCP into a receive buffer with an explicit size; a full buffer makes the sender wait and is counted, never dropped. It can cross attention-TP degrees (a DP-attention-1 prefill engine into a DP-attention-4 context-parallel decode engine) and re-shards into context-parallel pages. Mismatched layouts are refused at start-up. |
| Router (`kiln.server.pd_router`) | One OpenAI-compatible front. Prompts shorter than `--threshold` (default 4096 tokens) go straight to a decode engine. A longer one takes a decode credit before anything is sent, so a full decode side queues the request at the router in arrival order instead of overloading it. Optional latency prefill engines and a per-engine queue-depth cap. It refuses to start without at least one engine of each role. |
| Metrics | End-to-end and prefill TTFT, router and decode queues, KV transfer time, receive-buffer use and full events, disaggregated and bypassed request counts. |

The automatic expert-parallel default follows the role: a prefill engine uses expert parallelism,
a decode engine tensor-parallel experts, and an engine without a role keeps today's rule. An
explicit `KILN_MOE_EP` still wins.

**Measured** (GLM-5.3-Flash, 8192 in / 256 out, trn1.32xlarge spot $2.15/h per box, SDK 2.32,
every graph from the compile farm; the trees are named in
[docs/neuron-notes.md](docs/neuron-notes.md) "Prefill / decode disaggregation on the device").
Cost is attributed by role: the prefill boxes over input tokens, the decode box over output tokens,
and all-in is every box over output tokens.

| Deployment | Steady out tok/s | Steady all-in $/1M out | Split: $/1M in + $/1M out | Whole-level all-in $/1M out | ITL p50 |
|---|---:|---:|---|---:|---:|
| 3 prefill : 1 decode (4 boxes), concurrency 320 | 930.9 | $2.57 | $0.060 + $0.642 | $3.02 | 277.0 ms |
| 4 : 1 (5 boxes), concurrency 440, decode box with `KILN_DSA_CP_ALL_LOCAL=1 KILN_DSA_CP_PAGE_KEYS=1` | 1,139.5 | $2.62 | $0.067 + $0.524 | $3.40 | 353.5 ms |
| Colocated, one box, concurrency 64 (EPLB + one-piece prefill) | | | | $3.12 | |

"Steady" is the middle half of a closed-loop level; "whole level" includes the ramp and the tail.
The colocated ITL at concurrency 64 is 363 ms on the defaults (engine-v0 f70c14b).
No colocated steady figure was measured, so compare whole level with whole level: $3.02 against
$3.12, 3.1% lower. On this workload disaggregation is close to a wash on cost. What it buys is per-box
throughput (930.9 / 4 = 232.7 out tok/s per box against 191.3 colocated), a steadier and shorter
inter-token latency, and a decode side whose own cost reaches $0.524 per 1M output tokens in a
running deployment ($0.470 decode-only on the same box design).

**Time to first token.** With the ordinary prefill engines (DP attention 4) an idle 8K request
waits 4.33 s for its first token. A latency prefill engine runs the same model at DP attention 1,
one request per call over all 32 ranks in 4096-row calls, and brings it to 1.60 s:

| Arm (2 latency prefill + 1 decode) | TTFT p50 / p90 | ITL p50 |
|---|---|---:|
| Idle, one request at a time | 1595 / 1607 ms | 267.3 ms |
| Open loop, 0.6 req/s | 1530 / 1775 ms | 269.6 ms |
| Open loop, 1.0 req/s | 1596 / 2268 ms | 270.5 ms |
| Ordinary prefill engines, idle | 4334 / 4338 ms | 267.3 ms |
| Colocated one box, idle | 4013 / 4175 ms | 87.6 ms |

Two latency boxes hold a p90 under 2 s up to about 0.6 req/s. Feeding a whole decode box that way
would take far more prefill boxes than a throughput deployment does, so the latency arm trades
cost for TTFT. No arm reaches a p90 of 1 s for 8K prompts on trn1: the router's own histograms
put the prefill call at 81% of an idle request's TTFT (1.22 s of it), and the router's queue at
microseconds, so the floor is the prefill compute itself. The latency arm's
output matches the colocated reference on 15 of 16 greedy prompts (token agreement 0.964, mean
teacher-forced |dlogprob| 0.0023 on decode); the differences are near-ties.

**Run it.** Start each engine with the same arguments a `bench/serve_sweep.py` run of that
configuration takes (they decide the compiled graphs), plus a role, then the router:

```bash
# decode box
python bench/pd_serve.py --pd-role decode --pd-listen 0.0.0.0:7400 --port 8100 -- <decode serve_sweep arguments>
# each prefill box
python bench/pd_serve.py --pd-role prefill --port 8100 -- <prefill serve_sweep arguments>
# router, on any host that reaches all of them
python -m kiln.server.pd_router --prefill-urls http://p1:8100,http://p2:8100,http://p3:8100 \
  --decode-urls http://d1:8100 --tokenizer zai-org/GLM-5.3-Flash --threshold 4096 --port 8000
```

The exact prefill and decode configurations behind the table above are in
[docs/price-performance.md](docs/price-performance.md) "Prefill / decode disaggregation (G1,
trn1)". `bench/pd_sweep.py` drives the router closed loop or with Poisson arrivals and reports the
steady and whole-level figures; `tools/check_pd.py` compares a disaggregated deployment's output
with a colocated reference.

Not done yet: the handoff runs over host memory and TCP; a device-to-device path over EFA is not
built. trn2 is not a better disaggregation target at 8K: its best real-KV decode box costs $1.21 per
1M output tokens against trn1's $0.89 for the non-context-parallel trn1 box, and trn2 prefill costs
more per token than trn1's.

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
| attention collectives over the 8-rank DP group (0.1.0's default: 122.9) | 123.1 |
| the fused DSA selection-and-attention prefill kernel | 132.0 |
| the KDA and sparse-DSA decode kernels and sequence-parallel decode streams | 148.6 |
| MoE decode v2, and expert parallelism from 4 decode rows per group | 156.2 |
| the prefill world row gather as an NKI kernel | 164.7 |
| the runtime's hardware execution barrier (0.2.0's default: 167.8) | 167.8 |
| opt-in expert-parallel load balancing with one redundant slot per rank | 180.2 |
| opt-in 8192-token prefill as one 45-layer piece | 191.3 |

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

## Container

The image is AWS's Neuron vLLM inference image (Neuron SDK 2.32, Python 3.13, Ubuntu 24.04) with
Kiln installed at `/opt/kiln`; its entrypoint is `python -m kiln`. Run it on a trn1 host with the
Neuron driver (the Deep Learning AMI for Neuron has it) and pass the Neuron devices in.

```bash
docker pull public.ecr.aws/sanghwa/kiln:0.2.0     # also :0.2.0-neuronx-sdk2.32 and :latest
```

The quickstart model on one NeuronCore (`/dev/neuron0` holds two on trn1). The first mount is the
Hugging Face cache; the other two hold the compiled graphs and the NKI kernel binaries, so a
restarted container serves from them instead of compiling again:

```bash
docker run --rm --device=/dev/neuron0 -e NEURON_RT_VISIBLE_CORES=0 -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/neuron_libtorch:/root/.cache/neuron_libtorch \
  -v /var/tmp/nki-intermediate-cache:/var/tmp/nki-intermediate-cache \
  public.ecr.aws/sanghwa/kiln:0.2.0 --model Qwen/Qwen3-0.6B --device neuron --port 8000

curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "Trainium is", "max_tokens": 32}'
```

All 16 devices (32 NeuronCores) of a trn1.32xlarge, one tensor-parallel rank per core:

```bash
docker run --rm $(for i in $(seq 0 15); do echo --device=/dev/neuron$i; done) -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/neuron_libtorch:/root/.cache/neuron_libtorch \
  -v /var/tmp/nki-intermediate-cache:/var/tmp/nki-intermediate-cache \
  public.ecr.aws/sanghwa/kiln:0.2.0 --model <model> --device neuron --tp 32 --port 8000
```

**GLM-5.3-Flash needs farm-compiled graphs.** Its graphs take hours to compile on the device, so
compile them on CPU hosts first with `tools/compile_farm.py` (in the image at
`/opt/kiln/tools/compile_farm.py`) and serve from that cache. The exact configuration behind every
result above, compile farm included, is under "How to resume" in
[docs/price-performance.md](docs/price-performance.md); in the container, run those commands with
`--entrypoint python` (for example `bench/serve_sweep.py ...`). An `s3://` compile cache or farm queue
is read with the `aws` CLI inside the container, so the container needs AWS credentials: the
instance role answers from inside a container when the instance's metadata hop limit is 2 (it was on
a Neuron DLAMI instance) or with `--network host`; otherwise pass credentials in the environment.

The CPU test suite runs in the image too:

```bash
docker run --rm -e NEURON_RT_VISIBLE_CORES= -e KILN_TEST_MODEL=Qwen/Qwen3-0.6B \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint python public.ecr.aws/sanghwa/kiln:0.2.0 -m pytest -q
```

## License

Apache-2.0; see [LICENSE](LICENSE). Files adapted from vLLM, SGLang and transformers keep their
own Apache-2.0 notices, listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
