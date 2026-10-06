# Changelog

All notable changes to Kiln are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and Kiln follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) in its 0.y.z form: a minor release
adds a model family, changes a default or breaks a configuration; a patch release fixes.

## [0.2.0] - 2026-10-06

A minor release: Kiln's trn1 and trn2 defaults changed, and two opt-in subsystems are new
(prefill / decode disaggregation, and GLM-5.3-Flash's full 1M-token context).

**Every number below is labelled with the development commit it was measured on.** Unlike v0.1.0,
whose numbers were measured on a tree that differed from the release by one version string, this
release's numbers come from several development trees and none of them is byte-identical to the
released `kiln` package. The conditions, commands, logs and iteration history are in
`docs/price-performance.md` and `docs/neuron-notes.md`.

The headline workload throughout is "G1": zai-org/GLM-5.3-Flash@eb9eb208 (45 layers, 288 experts,
FP8, real weights), 8,192 tokens in and 256 out per request, a closed loop at the stated
concurrency, 128 requests per level, tensor parallel 32, DP attention 4, Neuron SDK 2.32, every
graph from the CPU compile farm. Kiln is priced at trn1.32xlarge spot, $2.15/h per box. The GPU
reference is unchanged from v0.1.0: vLLM (`vllm/vllm-openai:glm53-flash`) on a SageMaker
ml.p5en.48xlarge endpoint (8 x H200, TP 4 / DP 2 / EP 8), priced at the p5en.48xlarge spot band of
$28.77-30.29/h, spot against spot.

### Changed

**trn1 defaults.** Each of these was opt-in or absent in v0.1.0 and is on by default now:

- the fused DSA selection-and-attention prefill kernel (`KILN_DSA_FUSED`, with
  `KILN_DSA_PREFILL_KERNEL=nki`): G64 122.9 -> 132.0 out tok/s, the prefill call 0.592 -> 0.519 s
  (measured on feat/attn-kernel-tune 54ec5df).
- the KDA and DSA decode kernels and sequence-parallel decode streams
  (`KILN_KDA_DECODE_KERNEL=nki KILN_DSA_DECODE_KERNEL=nki KILN_DECODE_SP=1`): at conc 64 114.9 ->
  122.2 -> 128.0 out tok/s as each is added (measured on feat/decode-step a64d7f8); the decode-only
  step at 64 rows per DP group 456.5 -> 270.2 -> 235.2 ms (engine-v0 2968ed3 plus that branch).
- MoE decode v2 (`KILN_MOE_EP_SMALL_V=2`): G64 122.8 -> 129.5 out tok/s and the decode call
  0.178 -> 0.158 s, against about 1% run-to-run noise (measured on engine-v0 e449b98 plus the
  kernel additions, tree 4c3d078).
- expert parallelism from 4 decode rows per DP-attention group instead of 8, which it can be
  because v2 is bit-identical to v1 there: at conc 16 and identical shapes TP 86.9 -> EP v2 98.0
  out tok/s (+12.8%), TTFT p50 8.65 -> 6.10 s (measured on engine-v0 e449b98, EP gate f84c751).
- the sequence-parallel prefill world row gather as an NKI kernel (`KILN_SP_GATHER=nki`,
  `kiln/kernels/sp_gather.py`), Kiln's first use of `nki.collectives`: G64 156.1 -> 164.7 out tok/s
  and the prefill call 0.5195 -> 0.4765 s (measured on engine-v0 70ddc1b against 3a39ec8; made the
  default by ca256b7). It applies from 16 rows per rank and 256 columns, so decode graphs keep
  their keys, and a kernel's engines compute while its own collective is in flight, which XLA's
  graphs do not: 4 gathers alone 1.90 ms, 8192 independent matmuls alone 1.89 ms, both in one
  kernel 1.86 ms.
- the runtime's hardware execution barrier (`NEURON_RT_ENABLE_HW_EXECUTION_BARRIER=1`), which makes
  the per-execution cost 5.065 -> 2.800 ms on a 32-rank all-reduce probe and is bit-identical in
  serving: G64 163.7 / 164.4 -> 167.8 / 167.8 out tok/s (measured on engine-v0 b8814ab, recorded in
  8229c3d). `NEURON_RT_DISABLE_EXECUTION_BARRIER=1` gives the same throughput and is NOT taken: it
  deadlocked a race stress test within 250 launches and returns silently wrong numbers on a
  collective mismatch.
- `KILN_MIXED_KERNELS=1`: an opt-in mixed prefill + decode batch now takes the fused prefill kernel
  and the decode kernels instead of bypassing them, which is why mixed batches had stopped paying
  (G64 + mixed 152.0 -> 157.5 out tok/s, measured on 47e3806). **Mixed batches themselves are still
  opt-in** (`KILN_MIXED_BATCH=1`, default off): at G64 they now reach 157.5 against the plain
  default's 156.1 on that tree, where before this change they were below it.

**trn2 defaults** (trn2.48xlarge, LNC=2, two tp 32 engines, DP attention 4, measured on an EC2
Capacity Block; priced at trn2 spot $15.343/h for the like-for-like basis):

- the decode kernel stack with its LNC=2 two-core splits (the KDA and DSA decode kernels, SP decode
  streams, the `moe_dedupe` split, and the decode row splits): the decode-only step at 16 / 32 / 64
  rows per group 143.6 / 219.7 / 1438.7 -> 87.7 / 110.7 / 180.7 ms (-39 / -50 / -87%), measured on
  feat/trn2-fast 118c6ea.
- the fused DSA prefill kernel with its query-tile split over the two physical cores: per engine at
  conc 32 / 64 / 128 101.2 / 110.8 / 117.6 -> 104.4 / 114.7 / 121.9 out tok/s, the prefill call
  0.922-0.937 -> 0.885-0.899 s (measured on baf73ab, default in 1eeed36). **The unsplit kernel is
  neutral on trn2**, so the split is the whole gain there.
- whole box, with these defaults: 133.9 / 180.9 / 208.9 / 229.5 / 243.9 out tok/s at concurrency
  16 / 32 / 64 / 128 / 256, against engine-v0 70ddc1b's 118.5 / 158.9 / 177.3 / 192.0 / 198.4 on the
  same box (+13 to +23%); measured on feat/trn2-fast c08c5ba.

**The G1 result on one trn1.32xlarge**, concurrency 64, against the GPU reference's $5.48-5.77 per
1M output tokens:

| configuration | out tok/s | $ / 1M output tokens | vs the GPU reference | measured on |
|---|---:|---:|---|---|
| v0.1.0's defaults | 122.9 | $4.86 | 11-16% lower | engine-v0 ebe237e |
| v0.2.0's defaults | **167.8** | **$3.56** | 35-38% lower | engine-v0 8229c3d |
| + opt-in EPLB + the one-piece 8192 prefill configuration | **191.3** | **$3.12** | 43-46% lower | feat/prefill-mfu-merge, farm queue q/pf-p8-7ce6a12 |

The third row is NOT a default. It needs expert-parallel load balancing with one redundant expert
slot per rank (`KILN_EP_REDUNDANT=1` with `KILN_EPLB_INIT=<placement>` and `--eplb-rebalance`) and
the 8192-token prefill as one 45-layer piece (`--prefill-tokens 8192 --prefill-buckets 2048
KILN_PIECEWISE_PREFILL_MOE_GROUP=45`), which together need `--kv-cache-gb 1.2 --state-checkpoints 4`
to load. Its two levels measured 191.3 and 190.8 out tok/s. v0.2.0's defaults at the other levels:
110.5 out tok/s / $5.40 at concurrency 16 (79-80% below the reference) and 144.8 / $4.12 at
concurrency 32 (57-59% below).

Prefill MFU on trn1 went 7.7% (v0.1.0's G64 call) to 11.4% for that third row's call. Against
model-provider list prices ($0.15 per 1M input and $0.50 per 1M output tokens, $0.00136 per 8192 /
256 request), v0.2.0's default concurrency-64 level is $0.000911 per request, **33.0% below list**,
and the EPLB plus one-piece-prefill row is $0.000799, 41.2% below. A discounted provider at
$0.00068 for the same request is still 1.18-1.34x cheaper than Kiln. These four figures are
arithmetic on the measured rates (a request is 256 output tokens, so requests/s = out tok/s / 256
and $ per request = $2.15 / 3600 / that); the table and its derivation are in
`docs/price-performance.md`.

### Added

- **Prefill / decode disaggregation (opt-in).** `kiln/engine/disagg.py` gives an engine a role: a
  prefill engine hands a finished prompt's pages, pool keys and KDA state row to a decode engine
  over its own frames, and the decode engine admits the request straight to RUNNING, so it loads no
  prefill graph. `kiln/server/pd_router.py` is the front door, with threshold routing
  (`--threshold`, 4096 input tokens by default), decode-credit backpressure, an opt-in per-engine
  prefill queue-depth cap, opt-in latency prefill engines, and SLO metrics as `kiln:pd_router_*`
  Prometheus series. A handoff crosses attention TP degrees for a regroupable model, so a
  DP-attention-1 prefill engine can hand off into a DP-attention-4 decode engine.

  Measured on trn1 (prefill boxes on feat/disagg over engine-v0 f997d35 and then
  feat/disagg-stream; the decode box on scratch/pd-dec-f997 686d791; `bench/pd_sweep.py`): with 3
  prefill boxes to 1 decode box, 4 boxes at $8.60/h, concurrency 320, the sustained rate over the
  middle half of the level is **930.9 out tok/s at $2.57 per 1M output tokens all-in** ($0.060 per
  1M input on the prefill boxes plus $0.642 per 1M output on the decode box), with ITL p50 **277 ms**
  against the colocated default's 363 ms. At 4 prefill boxes to 1 decode box with the
  context-parallel decode flags, 5 boxes at $10.75/h, concurrency 440: **1,139.5 out tok/s** and
  **$0.524 per 1M output tokens** on the decode side.

  **Read the caveat with the number.** $2.57 is a steady middle-half rate and the colocated $3.12
  above is a whole-level rate; no colocated steady figure was measured, so they are not like for
  like. Whole level against whole level the 3:1 arm is $3.02 against $3.12, **3.1% below**: on this
  workload disaggregation is close to a wash on cost per token. What it does buy is per-box
  throughput (232.7 out tok/s per box against 191.3 colocated) and latency, below.

- **A decode-only box at $0.470 per 1M output tokens** (decode-only, 8K context, real KV, 96 rows
  per DP group under context-parallel DSA with `KILN_DSA_CP_ALL_LOCAL=1 KILN_DSA_CP_PAGE_KEYS=1`),
  1,269 out tok/s on one trn1.32xlarge, 302.4 / 302.8 ms per step over two passes; measured on
  feat/decode-scale-f997 e34272d with `tools/time_decode.py --real-kv`. Input is priced separately.

- **GLM-5.3-Flash's full 1,044,480-token context (opt-in).** A long DSA path that never forms
  anything of the context's size per query (`kiln/models/dsa_long.py`, past
  `KILN_DSA_LONG_KEYS`=16,384 keys), context parallelism over an attention group (`KILN_DSA_CP=1`,
  256-token pages), a minimal KV layout (`KILN_DSA_KV=minimal`, 5,984 against 9,152 bytes per token),
  an fp8 index scorer (`KILN_DSA_LONG_SCORER=index`), and two new NKI kernels
  (`kiln/kernels/dsa_long_select.py`, `kiln/kernels/dsa_slots.py`) whose exactness is checked
  against their host definitions on a NeuronCore.

  Needle-in-a-haystack at 131,072 / 524,288 / 1,044,480 tokens x depths 0.1 / 0.5 / 0.9 is **9 / 9
  on each of three engines**: trn1 context-parallel with the full KV layout (e5ff256 plus 51afd8a),
  trn1 context-parallel with the slot classes (feat/long-context 6b7f0e9), and trn2 with the minimal
  KV layout plus the fp8 index scorer (feat/long-context d1378cf). On trn1 a 1M prompt costs
  **$0.168 per 1M input tokens** at 3,549 input tokens/s per box (6b7f0e9); four concurrent 1M
  requests take the wall time of one, because each DP-attention group runs its own.

  Lone-request TTFT on that 1M engine, one trn1.32xlarge: 8,192 tokens 8.32 s, 32,768 33.2 s,
  131,072 133.0 s, 307,200 320.1 s, 524,288 581.1 s, 1,044,480 1177.1 s; ITL 56.6-60.3 ms short of
  1M and 64.1 ms at 1M. Every length pays the 256k bucket's selection, so a short prompt belongs on
  the 8K serving engine, which answers a lone 8,192-token request in 3.98 s. On trn2 the minimal
  layout's 1M decode ITL p50 is 92.2 ms through the fp8 scorer against 143.1 ms on the XLA scores,
  at a TTFT of 1651.7 s.

- Expert-parallel load balancing with redundant expert slots (`kiln/models/eplb.py`, opt-in
  `KILN_EP_REDUNDANT`), with an online rebalance.
- NKI kernels added since v0.1.0: the fused DSA prefill kernel, the sequence-parallel world gather,
  the long-context selection and slot-attention kernels, the pooled indexer's decode scores
  (`dsa_index`), a small decode call's whole MoE FFN (`moe_ffn`), and a decode-sized projection
  (`gemv`); plus `moe_dedupe` v9 and v10, which read each selected expert once for calls of up to
  512 tokens.
- One long prefill over several engines, as a layer pipeline (`kiln/engine/pp.py`, opt-in
  `pp_stages > 1`, prefill and piecewise only). CPU-tested only; not yet measured on the device.
- Opt-in asynchronous MTP drafting, `KILN_DECODE_WHOLE` (a decode call as one graph under
  piecewise), `KILN_MALLOC_TRIM`, and `KILN_DSA_CP_MERGE_BOUND` (a tie search over the bucket's bits).

### Fixed

All three are in code that is NEW in this release, so none of them changed behaviour a v0.1.0 user
could have seen.

- The fused DSA prefill kernel read a latent ring block that had already been overwritten when a
  call had fewer than 3 MLA heads per rank, so every DP-attention-1 prefill on a 32-rank box
  (attention TP 32, 2 heads per rank) was wrong on the device. That is reachable with no opt-in
  flag, because `--dp-attention` defaults to 1. `attend()` pads such calls to 3 heads outside the
  kernel's hashed source, so no existing configuration's compile-cache key changes. Measured in the
  NKI simulator (head 1 off by 2.15 at 2 heads, 2.26 at 1) and on trn1 (a greedy check against the
  colocated reference went from 10 / 16 equal with a first-token mean |dlogprob| of 0.58, to 15 / 16
  at 0.009).
- Expert-parallel load balancing read the wrong MoE decode version by default: it assumed decode v1
  when `KILN_MOE_EP_SMALL_V` was unset, so with v2 in use it spread a replicated expert's decode
  pairs over its redundant copies, which its own measurement had rejected. It follows the kernel in
  use now: 135.6 -> 136.7 out tok/s and the decode call 0.166 -> 0.162 s at G64 concurrency 64.
- An idle disaggregated decode engine never released the pages its prefix cache had evicted for an
  incoming handoff, because the hold it uses under overlap scheduling is drained only after a step
  in flight is read back, and an idle engine has none. Seen on trn2 as 30 queued handoffs with the
  KV cache 99% used and nothing running. An idle decode engine now drains its own hold before it
  gates admission.

### Known limitations (measured, not won)

- **On-demand pricing.** trn1.32xlarge on-demand is ten times its spot price, so no level is won
  on-demand against SageMaker on-demand. Every win above is spot against spot.
- **Time to first token is far behind GPU serving.** On an idle 8K request: 4.01 s colocated on one
  box, and 1.60 s on the disaggregated latency-prefill configuration, against the reference's TTFT
  p50 of 634 ms at concurrency 64. **No configuration reaches a p90 TTFT of 1 s on trn1**: the best
  p90 measured anywhere is 1607 ms idle, and 1775-2268 ms under open-loop arrivals of 0.4-1.0
  requests/s. Under load it is much worse (TTFT p50 5.37 s and p90 59.9 s at the default
  concurrency 64), and a long prompt takes minutes: 133.0 s at 128k and 1177.1 s at 1M.
- **Inter-token latency.** 131-363 ms across concurrency 16-64 on engine-v0 f70c14b's colocated
  defaults (343 ms at concurrency 64 once the NKI world gather is in), and 267-270 ms on the
  disaggregated decode box, against the reference's 20.1-43.4 ms at concurrency 16-128. The win is
  cost per token for throughput workloads.
- **Prefill is still far from the hardware.** MFU 11.4% on trn1 for the best configuration, and
  2.8% of BF16 peak on trn2. trn2 prefills at about trn1's rate per box (8.8-9.9k against 8.7-10.2k
  tokens/s) despite 3.5x the compute.
- **Expert parallelism hangs on trn2 at LNC=2**, on the first execution of an expert-parallel decode
  graph and not deterministically ("TOPSP ... missing collectives status"), so trn2 keeps
  tensor-parallel experts. Two more trn2 kernels stay unsplit for the same reason: the MoE prefill
  kernel's LNC split fails its wikitext check, and `sp_gather` has no correct LNC=2 form, so trn2
  keeps the XLA zero-padded gather.
- **trn2 is more expensive per token than trn1 at 8K context, on both sides of a disaggregated
  deployment.** Decode-only with real KV: $1.21 per 1M output tokens on a trn2 box against $0.89
  on a trn1 box, and $0.470 for trn1's best configuration. Whole requests: trn2's best level is
  $17.47 per 1M output tokens at concurrency 256, 4.9x trn1's $3.56 default at concurrency 64.
  trn2's case is context lengths trn1 holds less of.
- **Concurrency 128 does not fit** trn1's 16 GiB per NeuronCore at this workload.
- **The disaggregation cost win is not settled** (see the caveat in Added): whole level against
  whole level it is 3.1%, and the steady-rate comparison that reads better has no like-for-like
  colocated measurement behind it.
- GLM-5.3-Flash's graphs still take hours to compile on the device; compile them on CPU hosts first
  with `tools/compile_farm.py`.
- The layer pipeline across engines is CPU-tested only, and a 1M long-document NLL could not be
  measured (its prompt-logprob graphs fail to load beside the context-parallel KV).

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

[0.2.0]: https://github.com/foxl-ai/kiln/releases/tag/v0.2.0
[0.1.0]: https://github.com/foxl-ai/kiln/releases/tag/v0.1.0
