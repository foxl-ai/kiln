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

**GLM-5.3-Flash on one trn1.32xlarge costs 35-80% less per output token than vLLM on 8 x H200 at
concurrency 16, 32 and 64 on spot prices.** On EC2 Capacity Block prices, the basis where both sides
can actually be bought, it is 58% and 12% cheaper at concurrency 16 and 32, 1.31x the GPU's cost at
concurrency 64, and 2.1x the GPU's best rate (concurrency 128, which trn1 cannot hold). Which basis
applies is a question of capacity: trn1 spot was obtainable for these runs, p5en spot is a price quote.

| concurrency | Kiln, trn1.32xlarge | vLLM, ml.p5en.48xlarge (8 x H200) | Kiln spot $ / 1M out | vLLM spot quote $ / 1M out | Kiln Capacity Block $ / 1M out | vLLM Capacity Block $ / 1M out | Kiln on Capacity Blocks |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 110.5 tok/s | 311 tok/s | **$5.40** | $25.7-27.1 | **$23.96** | $56.41 | 58% lower |
| 32 | 144.8 tok/s | 842 tok/s | **$4.12** | $9.49-9.99 | **$18.29** | $20.84 | 12% lower |
| 64 | 167.8 tok/s | 1,458 tok/s | **$3.56** | $5.48-5.77 | **$15.78** | $12.03 | 1.31x higher |
| 128 | does not fit | 2,359 tok/s | | $3.39-3.57 | | $7.44 | |

<sub>zai-org/GLM-5.3-Flash (45 layers, 288 experts, FP8), 8,192 tokens in and 256 out per request, closed
loop at the stated concurrency, 128 requests per level, real weights, random-id prompts, all graphs from the
compile farm. Kiln: tensor parallel 32, DP attention 4, Neuron SDK 2.32, measured 2026-10-05 on development
commit engine-v0 8229c3d (v0.2.0's defaults). On real text, this release's measurement at concurrency 64 is
164.5 out tok/s (64 requests started together, dd428fd = engine-v0 a6c5dd9 with an opt-in flag off): $3.63
at spot, $16.10 on a Capacity Block. vLLM: the purpose-built `vllm/vllm-openai:glm53-flash` image on a SageMaker
endpoint (TP 4 / DP 2 / EP 8). Prices: trn1.32xlarge spot $2.15/h (obtained), the p5en.48xlarge spot band of
$28.77-30.29/h (`describe-spot-price-history`, read 2026-10-03: a quote, not capacity that was obtained), and
the EC2 Capacity Block prices of https://aws.amazon.com/ec2/capacityblocks/pricing/ (read 2026-10-09):
trn1.32xlarge $9.532/h, p5en.48xlarge $63.158/h, trn2.48xlarge $35.7608/h. Commands, logs and the full
iteration history are in [docs/price-performance.md](docs/price-performance.md).</sub>

**trn2.48xlarge, the whole box** (two tensor-parallel-32 engines behind Kiln's router, measured from the client
over HTTP on real text like the GPU reference, development commit engine-v0 df64441 and a labelled merge of the
causal DSA kernel, 2026-10-08): **632.4 out tok/s at concurrency 256**, $15.71 per 1M output tokens on a
Capacity Block, 2.1x the GPU's best ($7.44 at concurrency 128). At a latency bar of TTFT p90 <= 5 s it holds
concurrency 12-14 at 160-190 out tok/s ($52.25-61.51 per 1M), while vLLM holds that bar at least to concurrency
128, its highest measured level (2,359 out tok/s at p90 1.56 s, $7.44): at least 12.4-14.6x the throughput gap and
7.0-8.3x the cost. The target of 1,000 out tok/s at TTFT p90 <= 5 s on one box was not reached.

New in 0.3.0, all opt-in: **a layer pipeline** that runs one long prefill over several boxes and hands every
stage's KV to one decode engine, **a device-to-device KV handoff over EFA** with NIXL, and **ports from
vllm-neuron 0.24**. A 1,044,480-token prompt now takes 233.6 s on one trn1.32xlarge (1177.1 s in 0.2.0), and
68.62 s through four pipeline stages and a decode box over NIXL. On one trn2 chip against vllm-neuron 0.24 on the
same cores, Kiln's dense prefill is faster from 8k to 32k tokens (Qwen3-32B at 32k: 13.36 against 17.13 s),
its parallel engine start 1.76x faster cold and 2.2x warm, and its EAGLE-3 at 0.981x vllm-neuron's speed with
the same acceptance. How the pipeline and the handoff work: [Disaggregated serving](#disaggregated-serving); every
measurement with its commit: [CHANGELOG.md](CHANGELOG.md).

**What it does not win, measured:**

- **Latency.** On trn1 an idle 8K request takes 4.01 s to its first token colocated and 1.60 s on the
  disaggregated latency-prefill configuration, against 634 ms for vLLM on H200; under load at concurrency 64 the
  median first token takes 38.2 s on real text. Each next token takes 131-363 ms across concurrency 16-64 on trn1
  and 52-368 ms across concurrency 8-256 on the trn2 box, against 20-43 ms on the GPU. A 1M prompt takes minutes,
  not seconds. The win is cost per token on spot capacity, for throughput workloads.
- **Capacity Block pricing** at the same concurrency from concurrency 64 on trn1 and from 32 on trn2 (the trn2
  box is 16% below the GPU at concurrency 16, where the GPU's own TTFT p90 is 12.3 s), and against the GPU's best
  rate everywhere.
- **Concurrency 128** does not fit in trn1's 16 GiB per core.
- **Prefill is far from the hardware**: MFU 9.9% for trn1's 8K call and 17.1% of 32 x 95 TFLOPS for trn2's
  8,192-row call (about 8% of trn2's own BF16 peak). Kiln's dense-model prefill reaches 35-41% on trn2.
- **FP8 MoE** is faster on trn2 but fails the wikitext check, so it is not in this release.
- **Expert-parallel load balancing** (EPLB) does not pay on real text (-1%); its 0.2.0 gains were measured on
  random-id prompts.
- **On-demand pricing.** trn1.32xlarge on-demand is ten times its spot price, so no level is won
  on-demand against SageMaker on-demand.

## What is inside

| Layer | What Kiln does |
|---|---|
| Serving | OpenAI-compatible HTTP API with streaming, tool-call and reasoning parsers (Hermes, Qwen3, GLM, Kimi K2, Llama 3 JSON), `/tokenize` and `/detokenize`, a cache-aware router over data-parallel replicas that balances unmatched requests by load |
| Scheduler | continuous batching, chunked prefill, decodes that never pause for prefill, overlap scheduling, priority and cache-aware admission, mixed prefill + decode batches |
| KV and state | paged KV with FP8, radix prefix caching, prefix caching for linear-attention models from recurrent-state checkpoints, a host-memory tier |
| Parallelism | tensor parallel, DP attention, expert parallelism for MoE (trn1 and trn2), sequence-parallel prefill streams with group collectives, a layer pipeline that runs one long prefill over several boxes |
| Kernels (NKI) | MoE prefill and decode with 128x128-block FP8 scales, expert-parallel MoE, KDA linear attention (chunked, gated norm, short convolution, decode), DSA top-k selection, fused and causal sparse prefill attention, sparse decode attention, pooled-key caches, segmented dense prefill attention (from nkilib) |
| Speculation | MTP heads (DeepSeek-V3 / V3.2, GLM-5.3), EAGLE-3 drafts for dense models, n-gram and suffix drafting, speculation under overlap scheduling |
| Long context | a long DSA path that forms nothing of the context's size per query, context parallelism over an attention group or over row groups of all ranks, an exact local top K per rank, a minimal KV layout, an fp8 index scorer: GLM-5.3-Flash's full 1,044,480 tokens |
| Disaggregation | opt-in prefill and decode engine roles, a KV handoff over TCP or device to device over EFA (NIXL) that can cross attention TP degrees, a pipeline's stages as one prefill unit, and a router with threshold routing, decode-credit backpressure, latency prefill engines and SLO metrics |
| Compilation | piecewise or one-graph prefill, a CPU compile farm that captures every graph a device will trace, parallel compilation at engine start, an HBM estimator that predicts which configurations load |

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

**Device to device over EFA (opt-in).** With `KILN_PD_TRANSPORT=nixl` on both engines and
`NEURON_RT_MAP_HBM=1`, the decode ranks read the prefill engine's caches and state rows straight from its HBM
with NIXL over EFA; only a small meta frame crosses TCP. The host path stays the default. In a 3 : 1 deployment
(trn1.32xlarge with 8 EFA interfaces each, one cluster placement group; the prefill boxes on the colocated G64
graphs, so these two rows compare with each other only; feat/d2d-kv 8a60442): steady 786.0 -> 805.4 out tok/s,
ITL p50 305.3 -> 290.4 ms, the handoff's mean 0.965 -> 0.235 s.

**A long prompt over several boxes (opt-in).** A layer pipeline splits the model's layers into stages, one box
each (`--pp-stages`, `--pp-split`, `--pp-follow` and the stage links, as `bench/serve_sweep.py` / `bench/pd_serve.py`
arguments). A stage loads only its own layers, stage 0 takes the requests,
and every stage hands its own layers' KV and state rows to one decode engine, which admits the request once all
stages are in; the PD router takes the whole pipeline as one prefill unit (`--prefill-units`). One lone
1,044,480-token request, GLM-5.3-Flash on trn1.32xlarge, every graph from the compile farm:

| Layout | TTFT | Measured on |
|---|---:|---|
| One engine, one box | 233.6 s | 1c1f5f6 |
| 4 stages, the pipeline alone | 67.95 s | scratch/pp-final (feat/pp-serve + feat/prefill-compute) |
| 8 stages, the pipeline alone | 38.30 s | the same |
| 4 stages + a decode box, end to end over the host path | 72.36 s | engine-v0 a6c5dd9 + feat/pd-early-inject 2bd146f + 618401f |
| 4 stages + a decode box, end to end over NIXL | **68.62 s** (handoff 0.74 s, token 2 at 155 ms) | engine-v0 a6c5dd9 + feat/pd-early-inject 2bd146f |

The end-to-end rows' 64 greedy tokens and logprobs equal one engine's. feat/pd-early-inject (a stage's share
admitted as soon as it arrives) is not in this release. A warm second turn on the same 1M document, served from
the prefix cache, takes 1.34 s to its first token on one box. The configurations are in
[docs/neuron-notes.md](docs/neuron-notes.md) "The layer pipeline across boxes" and "A pipeline served as one
prefill engine, end to end".

trn2 is not a cheaper disaggregation target at 8K: its best real-KV decode box (5,007 out tok/s per box) costs
$1.98 per 1M output tokens at the trn2 Capacity Block rate, about what trn1's best decode box costs at its own
Capacity Block rate ($1.90), and trn1's runs on spot capacity that exists ($0.428).

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

Every row above was measured on random-id prompts. On real text the last two do not hold as written: EPLB gives
-1% (182.4 / 182.5 out tok/s without it, 180.0 / 181.1 with it, on the one-piece prefill), because random ids skew
the expert routing far more than text does.

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
docker pull public.ecr.aws/sanghwa/kiln:0.3.0     # also :0.3.0-neuronx-sdk2.32 and :latest
```

The quickstart model on one NeuronCore (`/dev/neuron0` holds two on trn1). The first mount is the
Hugging Face cache; the other two hold the compiled graphs and the NKI kernel binaries, so a
restarted container serves from them instead of compiling again:

```bash
docker run --rm --device=/dev/neuron0 -e NEURON_RT_VISIBLE_CORES=0 -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/neuron_libtorch:/root/.cache/neuron_libtorch \
  -v /var/tmp/nki-intermediate-cache:/var/tmp/nki-intermediate-cache \
  public.ecr.aws/sanghwa/kiln:0.3.0 --model Qwen/Qwen3-0.6B --device neuron --port 8000

curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B", "prompt": "Trainium is", "max_tokens": 32}'
```

All 16 devices (32 NeuronCores) of a trn1.32xlarge, one tensor-parallel rank per core:

```bash
docker run --rm $(for i in $(seq 0 15); do echo --device=/dev/neuron$i; done) -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/neuron_libtorch:/root/.cache/neuron_libtorch \
  -v /var/tmp/nki-intermediate-cache:/var/tmp/nki-intermediate-cache \
  public.ecr.aws/sanghwa/kiln:0.3.0 --model <model> --device neuron --tp 32 --port 8000
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
  --entrypoint python public.ecr.aws/sanghwa/kiln:0.3.0 -m pytest -q
```

## License

Apache-2.0; see [LICENSE](LICENSE). Files adapted from vLLM, SGLang and transformers keep their
own Apache-2.0 notices, listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
