# Price-performance race: GLM-5.3-Flash against vLLM on 8 x H200

Goal (owner, 2026-10-03): iterate until Kiln on Trainium serves zai-org/GLM-5.3-Flash at a
lower cost per token than the SageMaker AI deployment below, on the same workload.

## The reference

vLLM (purpose-built `vllm/vllm-openai:glm53-flash` image) on a SageMaker real-time endpoint,
ml.p5en.48xlarge (8 x H200), TP=4 / DP=2 / EP=8, max_model_len 32768, max_num_seqs 128 per DP
replica, gpu_memory_utilization 0.85, max_num_batched_tokens 8192; SageMaker optimized
generative AI benchmarking, 8192 tokens in / 256 out, streaming. Hosting price $72.795/h in
us-east-2 (Pricing API, AmazonSageMaker, instanceName ml.p5en.48xlarge, usagetype
USE2-Host:ml.p5en.48xlarge, read 2026-10-03).

| concurrency | TTFT p50 | TTFT p90 | ITL p50 | output tok/s | req/s | $ / 1M output tokens |
|---|---|---|---|---|---|---|
| 16 | 1428 ms | 12,316 ms | 20.1 ms | 311 | 1.2 | 65.02 |
| 32 | 979 ms | 4,672 ms | 25.0 ms | 842 | 3.3 | 24.02 |
| 64 | 634 ms | 2,724 ms | 32.3 ms | 1,458 | 5.7 | 13.87 |
| 128 | 415 ms | 1,561 ms | 43.4 ms | 2,359 | 9.3 | 8.57 |

## Pricing basis (corrected 2026-10-08)

- **trn2 spot was obtained once**, on 2026-10-04 (kiln-trn2-b, us-east-2c, $15.0887/h: the rows that name that box).
  Every trn2 run from 2026-10-05 on, v0.2.0's trn2 defaults and this round included, ran on EC2 Capacity Blocks
  because trn2 spot could not be obtained (`ec2 get-spot-placement-scores` for one trn2.48xlarge in us-east-2: 1 of 10,
  read again 2026-10-09 UTC): cr-00ff977628a81fb28 ($37.20/h) and
  cr-046a70209d206fcba ($1,443.55 for 40.37 h, 2026-10-06 19:08 to 2026-10-08 11:30 UTC by
  `describe-capacity-reservations`: $35.76/h); the first trn2 runs (2026-10-03/04) were on cr-01dd0041d61815bee
  ($689.59 for 19 h, $36.29/h). So every figure below priced "at trn2 spot $15.343/h" (and the $14.80/h ones) uses a
  spot QUOTE that was never paid for that run. Those rows stay as written, for the history; read them as spot-quote
  figures, not as costs.
- **The like-for-like basis for trn2 against p5en is the Capacity Block price on both sides**, from
  https://aws.amazon.com/ec2/capacityblocks/pricing/ (read 2026-10-08, again 2026-10-09 UTC; "The current prices are
  scheduled to be updated next in January, 2027"): trn2.48xlarge $35.7608/h (US East (Ohio) and Asia Pacific
  (Hyderabad)), p5en.48xlarge $63.158/h, trn1.32xlarge $9.532/h, OS fee $0.000 on Linux. The trn2 rate equals what
  cr-046a70209d206fcba cost.
- trn1.32xlarge spot at $2.15/h WAS obtained, and the trn1 boxes in this file ran on it unless their section says
  on-demand (some did when the account's spot request count was full, kiln-dsc-32 among them).
  p5en.48xlarge spot ($28.77-30.29/h, `describe-spot-price-history`, read 2026-10-03) is a quote that was never
  obtained: the reference itself ran on a SageMaker ml.p5en.48xlarge endpoint at $72.795/h on-demand hosting.
- Spot figures stay in this file as a labelled second basis only.

The headline rows at Capacity Block rates ($/1M out = $/h / (out tok/s x 3600) x 1e6; the p5en rows are the
reference's measured rates priced at the p5en Capacity Block rate):

| row | out tok/s | Capacity Block $/1M out | second basis, as recorded |
|---|---:|---:|---|
| trn2 whole box, highest conc with TTFT p90 <= 5 s, run 1: conc 14 (engine-v0 df64441) | 190.1 | **52.25** | $22.42 at the never-obtained trn2 spot quote |
| the same bar, run 2: conc 12 (PWC, the DSA causal default) | 161.5 | **61.51** | - |
| trn2 whole box, peak: conc 256 (PWC) | 632.4 | **15.71** | $6.74 at the spot quote |
| trn2 labelled peak-only row, 16384-row calls, conc 256 (engine-v0 df64441) | 639.8 | **15.53** | $6.66 at the spot quote |
| p5en vLLM, conc 64 (TTFT p90 2.72 s; a same-load row) | 1,458 | **12.03** | $5.48-5.77 at the p5en spot quote |
| p5en vLLM, conc 128 (TTFT p90 1.56 s: its highest measured level, its peak and still under the 5 s bar) | 2,359 | **7.44** | $3.39-3.57 at the p5en spot quote |
| trn1.32xlarge, G1 conc 64, real text (dd428fd, the default arm, 64 requests; an on-demand box) | 164.5 | **16.10** | $3.63 at the trn1 spot price |
| trn1.32xlarge, G1 conc 64, v0.2.0's defaults (engine-v0 8229c3d, random ids, 128 requests) | 167.8 | **15.78** | $3.56 at trn1 spot, obtained |

At the TTFT p90 <= 5 s bar trn2 costs **at least 7.0-8.3x** p5en per output token and serves at least 12.4-14.6x less:
p5en holds the bar at least to conc 128 (p90 1.56 s), its highest measured level, so these are lower bounds (trn2 at
conc 12-14 against p5en at conc 128; against p5en's conc 64 it would be 4.3-5.1x). At each side's peak trn2 costs
**2.1x** (trn2 at conc 256 against p5en at conc 128). At the same concurrency (engine-v0 df64441's balanced-router
run against the reference's table) trn2 is 16% below p5en at conc 16 (210.8 out tok/s, $47.12 against $56.41; p5en's
TTFT p90 there is 12.3 s, trn2's 5.16 s) and 1.45x / 1.92x / 2.51x p5en at conc 32 / 64 / 128 ($30.31 / $23.09 / $18.71
against $20.84 / $12.03 / $7.44). trn1 at the Capacity Block rate is
1.31-1.34x p5en at the same conc 64 and 2.12-2.16x p5en's best rate (conc 128, which trn1 cannot hold). On the spot
basis (trn1 spot against a p5en spot quote) trn1's 164.5 out tok/s ($3.63) is 34-37% below p5en at conc 64 and 2-7%
above it at conc 128. TTFT at trn1's conc-64 level is far from the GPU's: p50 38.2 s with ITL p50 230.9 ms on dd428fd (64 requests started
together), against p5en's 634 ms / 32.3 ms (TTFT p90 2.72 s).

## Pricing bases (us-east-2, read 2026-10-03)

| instance | on-demand | spot (describe-spot-price-history) |
|---|---|---|
| ml.p5en.48xlarge (SageMaker hosting) | $72.795/h | - |
| p5en.48xlarge (EC2) | (Pricing API returned no usable rate) | $28.77-30.29/h |
| trn1.32xlarge (EC2) | $21.50/h | $2.15/h |
| trn2.48xlarge (EC2) | (Pricing API returned no usable rate) | $14.80/h |
| trn2.48xlarge (EC2 Capacity Block, ap-south-2b, bought 2026-10-03) | $689.59 for 19 h = $36.29/h, prepaid | - |

A win is claimed only on a like-for-like basis and the basis is always stated: on-demand
against on-demand, spot against spot (p5en spot puts the GPU at $3.39 / 1M output tokens at
concurrency 128). Output tokens per second Kiln needs at concurrency 128 to match: trn1.32xlarge
on-demand ~697; trn1.32xlarge spot against p5en spot ~176.

## What has to change (the work list)

The workload is prefill-heavy (the GPU processed ~76K prompt tokens/s at concurrency 128).

| bottleneck | measured today on trn1.32xlarge tp=32 | work |
|---|---|---|
| MoE prefill | 42.8 ms per layer at C=32 (XLA gather), 1.6 ms with the decode kernel at C=32, DMA-bound per pair | feat/moe-prefill: grouped-GEMM NKI kernel, C up to 8192 |
| MoE decode for GLM | 267.5 ms per step at B=4 (XLA path). The NKI kernel now takes GLM's MoE (clamped SwiGLU, 128-block FP8 scales, 288 experts; feat/moe-kernel-v2): its MoE block on 2 ranks of trn1.2xlarge B=4 0.41 vs 0.90 ms on XLA, B=32 1.10 vs 25.0, B=128 2.00 (docs/neuron-notes.md) | re-measure the step at tp=32 with `KILN_MOE_KERNEL=nki KILN_PIECEWISE_MOE_GROUP=12` |
| KV capacity at 128 x 8448 tokens | MLA latent replicated on every TP rank: 17.0 GiB per rank at tp=32 (16,896 B per token per rank, arithmetic from config.json) | feat/dp-attn: DP attention, per-group KV pools; at `--dp-attention 8` 2.1 GiB KV + 0.6 GiB KDA state + 2.8 GiB mixer weights per rank (instead of 0.6); token-exact on trn1.2xlarge, 32-rank run pending (docs/neuron-notes.md "DP attention") |
| DSA selection at 8K keys | 6.0 ms of a 9.8 ms indexer layer is torch.topk | feat/glm-perf, measured on trn1.2xlarge (tools/profile_mla.py, 2026-10-03): exact bisection selection, GLM-5.3 full-indexer layer 19.8 -> 6.7 ms per 128-token chunk at 8K keys (shared layers 6.3 -> 1.7), decode B=8 9.8 -> 6.2 ms; GLM-5.3-Flash's pooled DSA layer 3.3 -> 2.1 ms per chunk, decode B=8 3.0 -> 2.7 ms (it was never dominated by its top-k). GLM-5.3-Flash's MTP layer runs on feat/linear-serving (next row). **feat/dsa-topk (2026-10-04)**: GLM-5.3-Flash had been running a range bisection, not the exact one; now an exact NKI selection kernel, pool keys cached per token, a prefill chunk's scores inside the kernel: pooled DSA block at tp=8 shapes decode 3.46 -> 2.46 ms (dense 1.96), prefill C=512 9.84 -> 6.28 ms; decode step at tp=32 141.2 -> 128.3 ms (120.7 with 24-layer decode graphs); conc 32 57.3 -> 64.4 out tok/s (docs/neuron-notes.md) |
| prefix cache / speculation on KDA models | feat/linear-serving: prefix cache from state checkpoints (one per radix node; 4.6 MB per rank per checkpoint for GLM-5.3-Flash at attention TP 32, from the config), n-gram / suffix / MTP verify with a state row per drafted position, also per DP-attention group; GLM-5.3-Flash's MTP layer wired; on trn1.2xlarge, Qwen3.5-0.8B: a 1056-token cached prefix 167 -> 19.4 ms to first token, MTP 1.52x at B=1. **feat/prompt-cache (2026-10-04)**: GLM-5.3-Flash at tp=32 / DP 4 on trn1.32xlarge: hits bit-identical to cold runs, 18.45 MB per rank per checkpoint at DP 4, lookahead junctions, shared-prefix sweep: conc 64 with 75% of input cached 90.6 -> 165.7 out tok/s, $0.000923 per request (section "G1b" below) | **feat/glm-mtp (2026-10-04)**: GLM-5.3-Flash's MTP on real weights at tp=32, every level and the G1b warm prefix workload measured: 96-98% of drafts accepted on the sweep's prompts at k=1 (75-88% on wikitext / chat), yet slower at every level (EP conc 64 106.9 -> 102.0, G1b warm 229.6 -> 223.1): the sweep is prefill-bound, and where decode dominates (G1b warm) the ~7% of device time MTP saves is lost to its synchronous step (~31 s of host time and prefill calls over 612 steps); next: drafting on the device so MTP steps can overlap (docs/neuron-notes.md "MTP speculative decoding on GLM-5.3-Flash at serving scale") |
| hardware | trn1: ~4x fewer FLOPS per chip than trn2 | trn2.48xlarge runs (feat/trn2 agent) |

## Where it stands (engine-v0 f70c14b, EPLB rows 25a45c9, 2026-10-05 14:00 UTC)

GLM-5.3-Flash (real weights), 8192 in / 256 out, 128 requests per level, trn1.32xlarge spot $2.15/h, tp=32,
DP attention 4, prefill 4096 per step, 12 MoE layers per graph, every default of engine-v0 f70c14b: on top of
ebe237e's (expert parallelism and group attention collectives by their trn1 auto rules; FP8 KV with the separate
pool-key cache at conc 64), the fused DSA selection-and-attention prefill kernel, the KDA / DSA decode kernels and SP
decode streams, MoE decode v2, and EP from 4 decode rows per group (the conc-16 config is EP now), all trn1 defaults.
One box (kiln-ak-32), farm graphs q/final-f70c14b and q/final-25a45c9 (EPLB), 0 device compiles. Logs
s3://<your-bucket>/logs/kiln-ak-32/ab-fin3-<config>.log, each with a `.log.cmd` holding the exact command; what
each config sets is in docs/neuron-notes.md "The final combined measurement". The trees after f70c14b (25a45c9's EPLB
fix, 80c07a6's opt-in asynchronous MTP, a3ca12b's fleet change) change no default or EPLB serving graph; asynchronous
MTP was measured on an older base and is not in this table.

| concurrency | config | out tok/s | TTFT p50 / p90 | ITL p50 | Kiln spot $/1M out | p5en vLLM spot $/1M out | |
|---|---|---|---|---|---|---|---|
| 16 | G16-4096-KV0.65-S20-P12-K, default | **105.4** | 5.4 / 5.5 s | 131 ms | **$5.67** | $25.7-27.1 | **won, 78-79% below** |
| 32 | F0-4096-KV1.5-S20-P12-K, default | **133.9** | 5.6 / 24.5 s | 213 ms | **$4.46** | $9.49-9.99 | **won, 53-55% below** |
| 32 | F0 + EPLB (opt-in, one redundant slot per rank, after the online rebalance; **random-id prompts only: on real text EPLB does not pay ("EPLB on real text" below)**) | **143.3** | 5.0 / 21.4 s | 201 ms | **$4.17** | $9.49-9.99 | **won, 56-58% below** |
| 32 | F0 + mixed batches (opt-in, `--decode-buckets 8,16`) | 128.8 | 5.9 / 26.5 s | 221 ms | $4.64 | $9.49-9.99 | won, 51-54% below; below the default |
| 64 | G64-4096-KV1.5-S20-P12-K, default | **156.2** | 5.8 / 65.1 s | 363 ms | **$3.82** | $5.48-5.77 | **won, 30-34% below** |
| 64 | G64 + EPLB (opt-in, after the online rebalance; 166.4 / $3.59 on the initial placement; **random-id prompts only: on real text EPLB does not pay ("EPLB on real text" below)**) | **167.2** | 5.3 / 58.9 s | 339 ms | **$3.57** | $5.48-5.77 | **won, 35-38% below** |
| 64 | G64 + mixed batches (opt-in, `--decode-buckets 8,16`) | 152.1 | 6.0 / 68.9 s | 375 ms | $3.93 | $5.48-5.77 | won, 28-32% below; below the default |
| 64 | G64 + mixed batches + EPLB (opt-in) | 163.3 | 5.4 / 62.2 s | 346 ms | $3.66 | $5.48-5.77 | won, 33-37% below; below EPLB alone |
| 128 | does not fit trn1 (16 GiB per core); trn2 whole box 221.6 on feat/trn2-fast 118c6ea (decode kernels, SP decode, the decode LNC splits, SP_GROUP, MoE prefill skip; engine-v0 70ddc1b on the same box: 192.0) | | | | $19.23 at trn2 spot $15.343/h | $3.4-3.6 | far |

**EPLB on real text (2026-10-07, kiln-pc-32): every EPLB win in this file was measured on bench/serve_sweep.py's
random-id prompts, whose routing is more skewed than text's.** Same box, tree, one-piece 8192-row prefill and KV 1.2 + CK4
in all four arms, conc 64, 128 requests per level, levels 1 / 2 (level 2 after the online rebalance); text = non-overlapping
8192-token windows of wikitext-103 (`--prompt-ids`); logs s3 logs/kiln-pc-32/pc-{N,E}-{rand,text}.log:

| prompts | prefill call per 8192 rows, no EPLB -> EPLB | decode call, no EPLB -> EPLB | out tok/s, no EPLB -> EPLB | HBM, fullest core |
|---|---|---|---|---|
| random ids | 0.944 / 0.944 -> 0.806 / 0.801 s | 0.114 -> 0.119 / 0.117 s | 174.2 / 174.3 -> **191.2 / 190.8 (+9.7%)** | 13.80 -> 14.82 GiB |
| wikitext-103 | **0.819 / 0.820** -> 0.811 / 0.798 s | 0.128 / 0.119 -> 0.131 / 0.128 s | **182.4 / 182.5 -> 180.0 / 181.1 (-1%)** | 13.80 -> 14.82 GiB |

So EPLB is NOT a prefill-role default: on text the prefill call without it is already where random ids get with it, the
slot's static decode pass costs more than the 8-22 ms of prefill it saves, and the slot takes 1.02 GiB per core. The
random-id gap (~125 ms per 8192-row call) is a benchmark artifact. At 1M the skew is real on text: docs/neuron-notes.md
"EPLB on real text".

Opt-in switches: EPLB `KILN_EP_REDUNDANT=1` with `KILN_EPLB_INIT=<placement>` and `--eplb-rebalance` (KV 1.5 still fits:
<= 15.5 GiB per rank by the farm's calibrated estimate); mixed batches `KILN_MIXED_BATCH=1 --state-checkpoints 4`, which
no longer pay on this tree (their joint mixers bypass the fused prefill kernel and the decode kernels, the likely reason,
not measured on its own; docs/neuron-notes.md).
Against provider list prices ($0.15 / $0.50 per 1M in / out, $0.00136 per 8192 / 256
request): the default conc 64 level costs $0.00098 per request (28% below), with EPLB $0.00091 (33% below); DeepInfra's
discounted $0.00068 is 1.35-1.44x below Kiln. On-demand is not won (trn1 on-demand is 10x its spot price), and latency
is far behind the GPU (ITL 131-363 ms against 20-32 ms).

Quality of the defaults against ebe237e: wikitext-2 -0.548 (-0.54753 against -0.54723); greedy check_mixed LONG_TEXT
28 / 32 equal with the decode-path mean dlogprob -0.0004 +/- 0.0003 nats per token, wikitext-2 prompts 7 / 32 with
-0.0011 +/- 0.0024; the per-stage attribution of that jitter (the decode kernels, the fused kernel and v2 each add about
the same, none moves the mean) is in docs/neuron-notes.md. EPLB: wikitext is the defaults' by construction (no redundant slot enters
check_ppl's graphs); greedy LONG_TEXT 28 / 32 with -0.0001 +/- 0.0004 against ebe237e and +0.0001 +/- 0.0002 against the
defaults.

## The v0.2.0 standing (engine-v0 8229c3d defaults, 2026-10-05)

The rows "Where it stands" records are engine-v0 f70c14b's. Two trn1 defaults landed after it, the
NKI world row gather (ca256b7) and the runtime hardware execution barrier, so the default numbers a
v0.2.0 user gets are the base column of "The 8192-token prefill as ONE piece" in
docs/neuron-notes.md, measured on engine-v0 8229c3d on kiln-pf-32c (G64) and kiln-pf-32 (F0, G16),
128 requests per level, same box and tree back to back. $ per 1M out is $2.15 / 3600 / out tok/s;
the "below" column is against the p5en vLLM spot band at that concurrency.

| concurrency | config | out tok/s | Kiln spot $/1M out | p5en vLLM spot $/1M out | below |
|---|---|---:|---:|---|---|
| 16 | G16, default | 110.5 | **$5.40** | $25.7-27.1 | **79-80%** |
| 32 | F0, default | 144.8 | **$4.12** | $9.49-9.99 | **57-59%** |
| 64 | G64, default | 167.8 | **$3.56** | $5.48-5.77 | **35-38%** |
| 64 | G64 + EPLB at KV 1.2 + CK4 + the one-piece 8192 prefill (opt-in; **random-id prompts only: on real text EPLB does not pay ("EPLB on real text" below)**) | 191.3 | **$3.12** | $5.48-5.77 | **43-46%** |

G64's TTFT p50 / p90 is 5.37 / 59.9 s at the default and 4.64 / 47.2 s on the opt-in row; its decode
call is 0.118 s and 0.119 s. trn2's best level ($17.47 per 1M out at conc 256, "trn2 standing") is
4.9x the trn1 default's $3.56 per token, and 5.6x the opt-in row's $3.12.

## Per request against provider list prices, derived from the rates above

Each row is arithmetic on a measured rate, shown so it can be checked: a G1 request is 256 output
tokens, so requests/s = out tok/s / 256 and $ per request = $2.15 / 3600 / that. List price for an
8192 / 256 request is $0.00136 ($0.15 per 1M input and $0.50 per 1M output tokens); DeepInfra's
discounted price for the same request is $0.00068.

| configuration | out tok/s | req/s | $ / request | vs list | vs the discounted price |
|---|---:|---:|---:|---|---|
| engine-v0 f70c14b default, conc 64 | 156.2 | 0.6102 | $0.000979 | 28.0% below | 1.44x cheaper than Kiln |
| engine-v0 f70c14b + EPLB, conc 64 | 167.2 | 0.6531 | $0.000914 | 32.8% below | 1.34x |
| engine-v0 8229c3d default, conc 64 | 167.8 | 0.6555 | $0.000911 | 33.0% below | 1.34x |
| + EPLB + the one-piece 8192 prefill | 191.3 | 0.7473 | $0.000799 | 41.2% below | 1.18x |

## Prefill / decode disaggregation (G1, trn1, 2026-10-05/06)

GLM-5.3-Flash@eb9eb208 real weights, 8192 in / 256 out, trn1.32xlarge spot $2.15/h per box, SDK 2.32, every
graph from the compile farm under `NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`. `bench/pd_sweep.py` against
`kiln/server/pd_router.py` with `bench/pd_serve.py` engines. Boxes kiln-pd-d1 (decode, also the router) and
kiln-pd-p1..p4 / kiln-pd-l1..l2 (prefill). Logs and the `.cmd` of every engine:
s3://<your-bucket>/logs/kiln-pd-{d1,p1,p2,p3,p4,l1,l2}/pd-logs*.tgz. The trees are NOT e3ff411 and
each row names its own; see docs/neuron-notes.md "Prefill / decode disaggregation on the device" for the
per-box commands, the trees and what could not be reproduced.

- **Prefill engine** (`--pd-role prefill`): tp 32, DP attention 4, the one-piece 8192 prefill
  (`--prefill-tokens 8192 --prefill-buckets 2048 KILN_PIECEWISE_PREFILL_MOE_GROUP=45`) with EPLB
  (`KILN_EP_REDUNDANT=1 KILN_EPLB_INIT=<placement> --eplb-rebalance`), KV 1.2 fp8, `--state-checkpoints 4`,
  farm queue q/pf-p8-7ce6a12. That is the same configuration as the colocated $3.12 row below.
- **Decode engine** (`--pd-role decode`): tp 32, DP attention 4, context-parallel DSA (`KILN_DSA_CP=1`),
  256-token pages, page bucket 64, `--max-num-seqs 384 --decode-buckets 96` (96 rows per DP group, CP-96),
  TP experts (`KILN_MOE_EP=0`), moe_dedupe v10, farm queue q/dc1-t1. From pd-logs2 on it also carries
  `KILN_DSA_CP_ALL_LOCAL=1 KILN_DSA_CP_PAGE_KEYS=1`.
- **Latency prefill engine** (`tools/pd_box.sh` TP1-P4K): the same env at DP attention 1 (attention TP 32,
  one request per call over all 32 ranks) in 4096-row calls, farm queue q/pf-tp1-pdpre.

**$ is attributed by role**: the prefill boxes' $/h over the input tokens, the decode box's $/h over the
output tokens, and "all-in" is every box's $/h over the output tokens. Each figure below reproduces from the
log's own rate (for example 3:1 steady: 3 x $2.15 / 3600 / (3.645 req/s x 8192) x 1e6 = $0.060 per 1M in;
$2.15 / 3600 / 930.9 x 1e6 = $0.642 per 1M out; 4 x $2.15 / 3600 / 930.9 x 1e6 = $2.566 all-in).

| prefill : decode | boxes, $/h | level | steady out tok/s (middle half) | steady all-in $/1M out | steady split $/1M in + $/1M out | whole-level out tok/s | whole-level all-in $/1M out | whole-level TTFT p50 / p90 | whole-level ITL p50 | log |
|---|---|---|---:|---:|---|---:|---:|---|---:|---|
| 3 : 1 | 4, $8.60 | conc 320 | **930.9** | **$2.566** | $0.060 + $0.6416 | 789.9 | $3.024 | 17.6 / 58.2 s | **277.0 ms** | g1-pd-f |
| 4 : 1 (+ ALL_LOCAL + PAGE_KEYS) | 5, $10.75 | conc 440 | **1,139.5** / 1,094.5 | **$2.621** / $2.728 | $0.067 + **$0.5241** | 879.4 / 849.3 | $3.396 / $3.516 | 8.6 / 70.1 s | 353.5 ms | g1-pd-l |
| 4 : 1, no ALL_LOCAL / PAGE_KEYS | 5, $10.75 | conc 440 | 1,002.2 / 967.5 / 967.4 | $2.980 / $3.086 / $3.087 | $0.087 + $0.5959 | 716.5-829.1 | $3.602-$4.167 | 8.4-11.2 / 60.9-79.1 s | 406.7-417.3 ms | g1-pd-k, -j, -i |

**Device-to-device handoff (`KILN_PD_TRANSPORT=nixl`, opt-in, 2026-10-06).** On kiln-d2d-a..d with EFA, the same
4 boxes and tree for both arms, and `NEURON_RT_MAP_HBM=1` on both. Prefill boxes run G64P (the colocated G64 graphs,
not the P8K-EPLB prefill above, whose keys this tree misses), so these rows compare with each other only:

| 3 : 1, conc 320, G64P prefill, CPA-R96 decode | steady out tok/s | steady all-in $/1M out | whole level | ITL p50 | handoff mean (prefill -> rows in decode HBM) | log |
|---|---:|---:|---|---:|---:|---|
| host path (default) | 786.0 | $3.039 | 634.7, $3.764 | 305.3 ms | 0.965 s | kiln-d2d-b d2d-pd-host-level |
| nixl | **805.4** | **$2.966** | 650.1, $3.675 | **290.4 ms** | **0.235 s** | kiln-d2d-b d2d-pd-nixl-level |

Idle 8K TTFT p50 is 4386 ms on the host path and 4266 ms with nixl. Tokens through check_pd are equal 8 / 8 at
conc 1, and inside the deployment's run-to-run floor at conc 32. Details, histograms and the wire measurements are
in docs/neuron-notes.md "Device-to-device KV handoff over EFA".

Two readings, and they do not say the same thing:

- **Steady (the middle half of a closed-loop level) against the colocated best is NOT like for like.** The
  colocated best in this file is 191.3 out tok/s / **$3.12** per 1M out ("Where it stands" plus the one-piece
  prefill, docs/neuron-notes.md "The 8192-token prefill as ONE piece"), and that is a WHOLE-LEVEL figure over
  128 requests. No colocated steady middle-half figure was measured, so $2.57 against $3.12 compares a steady
  rate with a whole-level one and overstates the gain. Whole level against whole level the 3:1 row is $3.024
  against $3.12, **3.1% below**: on this workload disaggregation is close to a wash on cost.
- **Per box it is a real gain**: 930.9 / 4 = 232.7 out tok/s per box steady against the colocated 191.3,
  and the decode box alone sustains 96 rows per DP group at $0.524 per 1M output tokens. The win is that one
  decode box carries the concurrency of four prefill boxes; the cost of the prefill boxes is what eats it.
- **ITL improves, TTFT does not** (both whole level, conc 320 against the colocated conc 64's 363 ms at
  f70c14b): 277.0 ms against 363 ms. TTFT p50 17.6 s and p90 58.2 s are far worse than the colocated 5.8 /
  65.1 s at conc 64, because the level holds 320 requests.

**Latency prefill boxes (2 latency prefill + 1 decode, 3 boxes, $6.45/h), the TTFT-SLO arm.** Post-fix tree
only (the `dsa_fused` MIN_HEADS bug made a DP-attention-1 prefill wrong on the device before engine-v0
4e5f226, so the pre-fix latency runs are timing only and their text is not cited). Logs
logs/kiln-pd-l1/pd-logs4-l1.tgz, logs/kiln-pd-l2/pd-logs4-l2.tgz, driver slo4.log on kiln-pd-d1:

| arm | TTFT p50 / p90 | ITL p50 | served req/s | log |
|---|---|---:|---|---|
| idle, one request at a time | **1595 / 1607 ms** | 267.3 ms | 0.014 | slo4-idle-lat |
| open loop, requested 0.4 / 0.6 / 0.8 / 1.0 req/s | 1554 / 1530 / 1559 / **1596** ms; p90 2126 / 1775 / 2078 / **2268** ms | 269.2-270.5 ms | 0.234 / 0.357 / 0.449 / 0.579 (steady 0.292 / 0.476 / 0.589 / 0.743) | slo4-open-lat |
| threshold routing, no latency box, idle | 4334 / 4338 ms | 267.3 ms | 0.014 | slo2-idle-thr |
| colocated one box, idle | 4013 / 4175 ms | 87.6 ms | 0.038 | colo-idle (kiln-pd-p1) |

So the latency arm takes an idle 8K TTFT from **4.01 s colocated to 1.60 s**, and **no arm reaches a p90 TTFT
of 1 s**: the best p90 measured anywhere on trn1 is 1607 ms idle and 1775-2268 ms under open-loop arrivals.
Its ITL is 267-270 ms against the colocated 87.6 ms, because the decode box runs 96 rows per group.
Device gate for that arm (`tools/check_mixed.py` against the colocated reference on the fixed tree,
chk-colo-fix on kiln-pd-l1 / slo4.log): **equal 15 / 16**, token agreement 0.9639, teacher-forced
|dlogprob| over decode calls n = 972 mean 0.00228 p99 0.0582 max 0.2980 with signed mean -0.00020, over
prefill chunks n = 16 mean 0.01168 signed -0.00430.

## The 8K lone-request TTFT through a pipeline (trn1.32xlarge, 2026-10-07, feat/lc-scaleout + q/pc-8k-1bae0bd)

GLM-5.3-Flash at DP attention 1, TP 32 per box or stage, the latency prefill env without redundant expert slots
(KILN_MOE_EP=1), 12-layer prefill pieces (15 for 3 stages: the stage bounds fall on piece bounds, so a stage runs only
the single engine's graphs), page buckets 264, FP8 KV; post-MIN_HEADS tree. Real text: lone 8192-token requests, each a
different wikitext-103 window (wt8k.npy row), 10 per arm after 2 warm ones; pipelines with KILN_PP_ASYNC=1
KILN_PP_OVERLAP=1, TTFT on the last stage from stage 0's arrival (lc_ttft). Engine time only: the router, the handoff and
the decode engine's first token are outside it (below). $ at the trn1.32xlarge spot reference $2.15/h; "lone" = boxes x
TTFT (a latency tier idle between requests), "full tier" = the boxes' busy time per request when requests follow each
other through the stages (chunks x each stage's call time).

| layout | boxes | TTFT p50 (min - max) | lone $/request | full-tier $/request | log |
|---|---|---|---|---|---|
| 1 box, 4096-row calls | 1 | 1.223 s (1.198 - 1.254) | $0.00073 | $0.00073 | kiln-pd4-s0 pc8k-L84K-1box-S0 |
| 1 box, 2048-row calls | 1 | 1.456 s (1.436 - 1.490) | $0.00087 | $0.00087 | kiln-pd4-s3 pc8k-L8-1box-S0 |
| 3 stages (15, 30), 2048-row calls | 3 | **0.802 s** (0.791 - 0.834) | $0.00144 | ~$0.00088 | kiln-pd4-s{0,1,2} pc8k-G15-3st-S* |
| 4 stages (12, 24, 36), 2048-row calls | 4 | **0.737 s** (0.721 - 0.751) | $0.00176 | ~$0.00096 | kiln-pd4-s{0..3} pc8k-L8-4st-S* |

- Per-stage prefill calls (exec records, KILN_TIMELINE): 106 / 131 / 131 ms for 3 stages, 85 / 102 / 107 / 107 ms for 4;
  one 2048-row call through all 45 layers on one box is ~0.36 s and a 4096-row one ~0.60 s.
- Outside the engine, measured separately (docs/neuron-notes.md "The 8K TTFT outside the prefill call"): client to
  router 1.4 ms (+ ~28 ms of the router's own tokenization on a text prompt), router to prefill 4 ms, no wait for the
  decode engine since the router fix, and over KILN_PD_TRANSPORT=nixl one meta frame for the handoff (the host path adds
  ~105 ms). So end to end ~0.75 / 0.78 s (ids / text) on 4 stages and ~0.82 / 0.85 s on 3: both under 1.0 s at p50,
  against ~1.24 s for the best one-box layout. Not yet measured as one run: that needs the stages' serving front and
  the per-stage handoff (feat/pp-handoff) on one tree with a decode box.

## Long context (W1M): lone-request TTFT on 1, 4 and 8 boxes (trn1.32xlarge, 2026-10-07, feat/lc-scaleout)

GLM-5.3-Flash, the lever-1 R8 engine per box or per pipeline stage (docs/neuron-notes.md "Lever 1" and "The layer
pipeline across boxes"), SDK 2.32, every graph from the compile farm under NEURON_LIBTORCH_ASSERT_CACHE_HIT. One request at
a time, seeded random prompts; the pipelines with KILN_PP_ASYNC=1 KILN_PP_OVERLAP=1. Cost per request = boxes x TTFT x
the trn1.32xlarge spot price ($2.15/h, us-east-2c), i.e. assuming the boxes are kept busy by back-to-back requests (a
pipeline overlaps successive requests' chunks the same way it overlaps one request's).

| boxes | 32k | 128k | 300k | 1M (1,044,480) | spot $ per 1M-token request | $ / 1M input tokens | log |
|---|---|---|---|---|---|---|---|
| 1 | 8.90 s | 35.43 s | 84.94 s | 319.1 s | $0.191 | $0.183 | kiln-pp-s1 ref1-r8-S0 |
| 4 (split 12, 24, 36) | 3.35 s | 10.72 s | 24.57 s | **89.29 s** | $0.213 | $0.204 | kiln-pp-s* pp4-r8-ao2-S* |
| 8 (split 7, 12, 16, 23, 28, 35, 40) | 2.55 s | 6.84 s | 14.72 s | **52.43 s** | $0.251 | $0.240 | kiln-pp-s* pp8-ao-S* |
| 8, the fixed tree (fb7c6b0 + 840aa7b) | 2.69 s | 6.83 s | 14.81 s | 52.20 s | $0.249 | $0.239 | kiln-pp-s* pp8-d2h2-S* |

- 4 boxes cost 12% more per request than 1 for 3.6x the speed at 1M; 8 boxes 31% more for 6.1x.
- The 8-stage run used 4 spot and 4 on-demand boxes (the spot quota was full). At that mix the request cost $1.38;
  the table prices all eight at spot.
- Gates: 8 stages pass the teacher-forced needle at 128k and 1M, 6 / 6. Prompt logprobs over 128k sit within the spread
  of two single engines (mean abs 0.0235 against 0.0219). The 2-stage pipeline on the single engine's graphs is bit-exact.
- R8 itself at 1M on real text: over two consecutive 1M windows (2,088,958 positions) R8 - CP8CE-m is -0.00006
  nats/token signed (docs/neuron-notes.md "Lever 1 on real text"); the earlier single-window deviation past 256k did not
  replicate.
- A WARM second turn on a 1M document, the prefix served from the cache and one state checkpoint, takes **1.34 s** to
  first token for 4,080 new tokens on one box (kiln-pp-s0 warm1m-1c-S0), about $0.0008 at spot. It is the realistic
  follow-up-question case, not a cold number.
- Decode after a pipelined prefill: every stage hands its own layers' KV and state rows to one decode engine, measured end
  to end below.

**The final 1M config (R8LK: R8 + KILN_DSA_CP_LOCAL_K=120 + LONG_PIPE + MERGE_BOUND, with feat/prefill-compute 1c1f5f6's
defaults; tree scratch/pp-final, feat/pp-serve)**, the same seeded prompts, local-K exact on every engine (0 rows past
CP_LOCAL_F). TTFT of the pipeline alone (lockstep gates, the last stage's first token), costed as above:

| boxes | 128k | 1M (1,044,480) | spot $ per 1M-token request | $ / 1M input tokens | log |
|---|---|---|---|---|---|
| 1 | 26.575 s | 236.92 s | $0.141 | $0.135 | kiln-pd4-dec gate1-lk-S0 |
| 4 (split 12, 24, 36) | 8.274 s | **67.95 s** | $0.162 | $0.155 | kiln-pd4-s* gate4-lk-S* |
| 8 (split 7, 12, 16, 23, 28, 35, 40) | 5.67 s | **38.30 s** | $0.183 | $0.175 | kiln-pd4-s*, kiln-pd8-s* gate8-lk-S* |

End to end (router -> the stages -> host/TCP handoff -> one R8LK decode engine, 64 greedy tokens; docs/neuron-notes.md
"A pipeline served as one prefill engine, end to end"), TTFT and ITL p50 at 1M: one engine 237.13 s / 46.4 ms;
4 stages + decode **85.35 s** / 46.1 ms; 8 stages + decode **55.85 s** / 46.3 ms, tokens and logprobs bit-identical
to one engine on the same graphs. About 17 s of each pipelined TTFT at 1M was the host-path handoff (one box receiving
~38 GB over TCP). It was the decode box's receiver, not the network: with the part frame received straight into its
mapped file (feat/pd-recv-mmap), 4 stages + decode on the EFA session's boxes went from 86.40 s to **72.36 s** at 1M
(handoff 18.41 -> 4.45 s, tokens and logprobs equal; docs/neuron-notes.md "That 2.2 GB/s was the decode box's
receiver"). Over NIXL on EFA (every stage and the decode engine with KILN_PD_TRANSPORT=nixl, the same five boxes) it is
**68.62 s**, the pipeline's own 67.88 s plus a 0.74 s handoff, token 2 at 155 ms, tokens and logprobs still equal
(s3 logs/kiln-nx-dec/nx4n-*; docs/neuron-notes.md "Over NIXL the handoff leaves the TTFT"). Holding S + 1 boxes for the
TTFT, a 1M request's prefill costs $0.205 over NIXL on 4 + 1 boxes ($0.255 on the host path before the receiver fix) and
$0.300 on 8 + 1 over the host path. 8 + 1 over NIXL is not measured; composed from its measured pipeline (38.61 s) and
the measured NIXL handoff it would be about 39.4 s, $0.21.

## trn2 final round (feat/trn2-final, 2026-10-08 00:00-09:00 UTC: real text, two router fixes, and what bounds TTFT)

trn2.48xlarge (Capacity Block cr-046a70209d206fcba, ap-south-2b, kiln-t2-cb2), Neuron SDK 2.32, LNC=2, GLM-5.3-Flash real weights,
fp8 KV, farm graphs, 0 device compiles. The best gated config is E1B-p6PW:
- the trn2 E1B env (expert parallelism with block scales, the LNC=2 split list) + `KILN_PIECEWISE_PREFILL_MOE_GROUP=6` +
  `KILN_PREFILL_WHOLE=1` (feat/prefill-fewer-graphs, opt-in);
- DP attention 4, 8192-row prefill calls (`--prefill-tokens 8192 --prefill-buckets 2048`);
- gates: wikitext -0.55228 against -0.55073, check_mixed 29 / 32 (docs/neuron-notes.md "trn2 final round").
- engine-v0 dc317d7 adds the DSA causal block skip by default for these shapes ("PWC" below).

**Every earlier trn2 G1 number used random token ids.** On real text (wikitext-103 windows, `--prompt-ids`), one engine
(half the box) in-process:

| conc | random ids out tok/s | real text out tok/s | prefill call random / text |
|---|---|---|---|
| 32 | 211.7 | 219.4 | 0.613 / 0.521 s |
| 64 | 252.9 | 272.7 | 0.612 / 0.522 s |
| 128 | 287.6 | 318.1 | 0.613 / 0.521 s |

Random ids skew expert parallelism. On text, the MoE reduce-scatter's wait for the busiest rank halves, from 178 to 88.5 ms
per call on rank 0.

**Over HTTP, measured like the p5en reference** (client side, streaming, 8192 in / 256 out, closed loop with the start
burst counted), one engine behind `kiln.server.router`, real text:

| conc | out tok/s | TTFT p50 / p90 | $/1M out at half the box's trn2 spot ($7.672/h) |
|---|---|---|---|
| 4 | 56.5 | 2.83 / 2.89 s | 37.69 |
| 8 | 91.2 | 3.49 / **4.48 s** | 23.36 |
| 16 | 153.8 | 3.52 / 9.30 s | 13.86 |
| 64 | 251.8 | 3.87 / 31.9 s | 8.46 |
| 128 | 312.9 | 3.95 / 65.2 s | 6.81 |

- **The whole box over HTTP on real text, balanced router** (two engines, engine-v0 df64441, 2026-10-08 05:36 UTC):
  - The p5en-style headline: **190.1 out tok/s at conc 14 with TTFT p90 4.55 s** (p50 2.78 s, $22.42/1M out). conc 16 gives 210.8 at p90 5.16 s. p5en vLLM holds 2.7 s at conc 64 with 1458 out tok/s, and 1.56 s at conc 128 with 2,359 (its highest measured level, so the bar's comparison is against conc 128).
  - Peak: **620.7 out tok/s at conc 256** (654.4 steady, **$6.87/1M out**), TTFT p50 4.2 s / p90 67 s.
  - The same with the DSA causal block skip on (PWC, the DSA agent's default candidate, gated 32 / 32 and wikitext-equal): **632.4 out tok/s at conc 256 ($6.74/1M out)**, and 161.5 at conc 12 with p90 4.51 s.
  - A labelled peak-only row with 16384-row calls (4096 rows per group per call) reaches **639.8 out tok/s at conc 256, $6.66/1M out**. Its TTFT p50 is 5.4-6.8 s, and p90 is not claimed.
  - In the p90 ≤ 5 s region the two configs are equal within run-to-run noise: p90 is the 3rd or 4th slowest of 24-32 TTFTs per level. Quote the pair as conc 12-14 at 160-190 out tok/s.
  - Every level split the requests evenly over the two engines.
  - The 04:00 run before the balance fix gave 140.1 at conc 12 and 557.4 peak, because the router's tree-size rule split requests unevenly from conc 32 on (docs/neuron-notes.md "trn2 final round").
- **At the Capacity Block rate actually paid** ($35.7608/h for the box, "Pricing basis (corrected 2026-10-08)" above;
  the $ figures in the bullets above are at the never-obtained trn2 spot quote of $15.343/h): conc 14 190.1 out tok/s
  = **$52.25** and conc 12 161.5 = **$61.51** per 1M out at TTFT p90 <= 5 s, against p5en's $7.44 at conc 128 (p90 1.56 s,
  its highest measured level: at least 7.0-8.3x); peak 632.4 = **$15.71** and the 16384-row row 639.8 = **$15.53**,
  against the same p5en $7.44 (2.1x).
- **A labelled single-box PD configuration** (DP-1 prefill engine on one half, CPD64E decode on the other, host handoff): ITL 70-88 ms at every load. It holds TTFT p90 ≤ 5 s only at conc 4 (50.3 out tok/s) and saturates at about 327 out tok/s, because the prefill half serves about 1.27 requests/s. It is a latency tier, not a competitor to the colocated headline.
- **The router had capped every earlier whole-box HTTP run at 100 requests in flight** (httpx's default pool; fixed in
  90da152, engine-v0 1a9d6d7). The 2026-10-07 random-id run's 401.9 out tok/s at conc 128 and 316.9 at conc 256 understate
  two engines.
- **Why p90 is hard:**
  - At DP attention 4 a prefill call carries one 2048-row chunk per group. So a lone 8192 prompt takes four calls (TTFT ≥ about 2.1 s), and burst requests finish four at a time every about 2.4 s.
  - A DP-attention-1 prefill engine measured 1.14 s lone TTFT but 0.76 s per request against 0.52 s, so its burst p90 is worse (5.74 s at 8).
  - Only a faster prefill call moves the p90 headline.

## trn2 standing (feat/trn2-kda-conv 3928014, 2026-10-07 15:00 UTC: the NKI short conv in KDA prefill by default)

trn2.48xlarge (Capacity Block cr-046a70209d206fcba, ap-south-2b, kiln-t2-cb2), SDK 2.32, LNC=2, GLM-5.3-Flash real weights,
two tp=32 engines (`--dp 2`), DP attention 4, farm graphs (q/t2f-E1BC@cv), 0 device compiles; $ at trn2 spot $15.343/h.

The trn2 default now runs the KDA layers' short causal conv as an NKI kernel split by channels (kernels/short_conv.py; not
with context-parallel DSA). At LNC=2 the XLA conv's row split between the two physical cores cost 6.6 ms per KDA layer,
215 ms of a 4096-token prefill call (docs/neuron-notes.md "trn2 G1 prefill"). The gates were run on one tree against the
previous default:
- wikitext -0.5507 against -0.5495;
- check_mixed 29 / 32 equal;
- one engine, conc 32 / 64 / 128: 142.0 / 159.1 / 171.5 -> 186.8 / 217.8 / 241.5 out tok/s.

G1 (8192 in / 256 out), whole box, log s3 logs/kiln-t2-cb2/*-t2-U-E1Bdef.log:

| conc | out tok/s | before (E1B on 68504b9, below) | TTFT p50 | ITL p50 | prefill call | $/1M out (trn2 spot) | p5en vLLM spot $/1M out |
|---|---|---|---|---|---|---|---|
| 16 | 201.6 | 171.3 | 3.7 s | 65 ms | 0.359 s | **21.14** | 25.7-27.1 |
| 32 | 301.5 | 239.5 | 3.9 s | 90 ms | 0.367 s | 14.14 | 9.49-9.99 |
| 64 | 374.4 | 282.7 | 4.0 s | 150 ms | 0.368 s | 11.38 | 5.48-5.77 |
| 128 | 435.8 | 316.5 | 4.2 s | 260 ms | 0.370 s | 9.78 | 3.4-3.6 |
| 256 | 483.1 | 340.5 | 4.4 s | 463 ms | 0.382 s | **8.82** | - |

- The gains are +18 / +26 / +32 / +38 / +42%. The "before" column is on an older tree; the same-tree A/B is the one-engine
  run above.
- Prefill per box: 2 engines x 4096 rows / 0.368 s = **22.3k tok/s** (13.9k before; trn1.32xlarge 8.7k, 10.2k with EPLB).
  Prefill is now 40-72% of the device time.
- trn1's G64 default ($3.82) is now 2.3x cheaper per token than trn2's best level, down from 3.3x.

## trn2 standing (feat/trn2-next 68504b9, 2026-10-06 22:40 UTC: expert parallelism by default)

trn2.48xlarge (EC2 Capacity Block cr-046a70209d206fcba, ap-south-2b, kiln-t2-cb2), SDK 2.32, LNC=2, GLM-5.3-Flash real weights, two
tp=32 engines (`--dp 2`), DP attention 4, farm graphs (q/t2f-E1B), 0 device compiles; $ at trn2 spot $15.343/h as below. With the
LNC=2 scatter fix (docs/neuron-notes.md "The EP hang at LNC=2 is the scatter's out-of-bound skip") expert parallelism is the trn2
default for glm5_next (tile scales, the moe_ep split): wikitext -0.5495 against -0.5498 and check_mixed 28 of 32 equal against the
TP-expert defaults ("Expert parallelism on trn2 after the fix"). G1 (8192 in / 256 out), whole box, log s3
logs/kiln-t2-cb2/*-t2-U-E1B.log:

| conc | out tok/s | TP-expert defaults (c08c5ba, below) | TTFT p50 | ITL p50 | prefill call | $/1M out (trn2 spot) | p5en vLLM spot $/1M out |
|---|---|---|---|---|---|---|---|
| 16 | 171.3 | 133.9 | 5.6 s | 71 ms | 0.574 s | **24.88** | 25.7-27.1 |
| 32 | 239.5 | 180.9 | 5.9 s | 110 ms | 0.586 s | 17.80 | 9.49-9.99 |
| 64 | 282.7 | 208.9 | 6.0 s | 198 ms | 0.588 s | 15.08 | 5.48-5.77 |
| 128 | 316.5 | 229.5 | 6.2 s | 363 ms | 0.590 s | 13.47 | 3.4-3.6 |
| 256 | 340.5 | 243.9 | 6.4 s | 676 ms | 0.602 s | 12.52 | - |

(+28 / +32 / +35 / +38 / +40%; prefill now 56-80% of the device time, from 66-87%.) conc 16 is below p5en spot for the first time
on trn2; from conc 32 on trn2 still loses, and trn1's G64 default ($3.82) stays 3.3x cheaper per token than trn2's best level.
Prefill per box: 2 engines x 4096 rows / 0.588 s = **13.9k tok/s** (8.8k before; trn1.32xlarge 8.7k, 10.2k with EPLB).

With 8192-row prefill calls (opt-in arguments `--prefill-tokens 8192 --prefill-buckets 2048 KILN_PIECEWISE_PREFILL_MOE_GROUP=6`,
q/t2f-E1B-p6@pf8k, log *-t2-U-E1Bp6): conc 32 / 64 / 128 / 256 = 255.2 / 305.9 / 346.9 / 379.0 out tok/s, **$16.70 / 13.93 /
12.29 / 11.25** per 1M out (+6.6 to +11.3% over the 4096-row default), prefill call ~1.06 s per 8192 rows = 15.4k tok/s per box.
One engine each, same box: MoE groups of 4 and of 6 per prefill piece give the same rate (152.6 / 172.9 / 188.6 and 152.9 /
173.5 / 189.5 at conc 32 / 64 / 128).

**trn2 decode box with context-parallel DSA** (the trn1 decode box's settings on trn2: `KILN_DSA_CP=1 KILN_DSA_CP_ALL_LOCAL=1
KILN_DSA_CP_PAGE_KEYS=1`, ST, `KILN_MOE_DEDUPE_V9=1 KILN_MOE_DEDUPE_MAX_TOKENS=512`, 256-token pages, page bucket 64, DP attention 4,
the trn2 defaults on top; `tools/time_decode.py --real-kv --all-buckets --steps 64 --skip 8`, 8K contexts, one engine = half the
box, $ at half-box spot $7.6715/h; farm queues q/t2f-CPT<rows>[E]; logs s3 logs/kiln-t2-cb2/*-t2-tdr-CPT*):

| rows per DP group (per step) | TP experts: ms per step | $/1M out | EP (`KILN_MOE_EP=1`, tile scales): ms per step | out tok/s per engine | $/1M out | tensors per rank |
|---|---|---|---|---|---|---|
| 128 (512) | 343.1 | 1.428 | 254.8 | 2,009 | 1.061 | 15.9 GB |
| 256 (1024) | 537.0 (537.5 repeated) | 1.118 | 413.2 | 2,478 | 0.860 | 19.6 GB |
| 320 (1280) | | | **511.2** | **2,504** | **0.851** | 21.3 GB |

(192 rows per group does not compile: `[NCC_IINAR001] ISA validation failed: Matmul ... s3d3_mm_valid_dst_partition`.) The
whole box at 320 rows per group is 5,007 out tok/s, **$0.851 per 1M out, against $1.21 for the trn2 decode box before** (no CP,
96 rows per group) and **$0.470 for trn1's** (CP-96, the same flags). Expert parallelism is the larger lever here, at 1,024 to
1,280 rows per MoE call: the EP kernel takes the whole batch as one dequantize-first call (about 2 ms per layer at C=1024 on a
logical core), where TP's dedupe reads every expert per 512-token chunk.

**Why trn2 decode does not reach trn1's $0.47.** Per dollar, trn1 has twice trn2's HBM bandwidth. trn2.48xlarge is 16 chips x
2.9 TB/s = 46.4 TB/s for $15.343/h (3.0 TB/s per $/h); trn1.32xlarge is 16 x 0.82 TB/s = 13.1 TB/s for $2.15/h (6.1 TB/s per $/h).
So at equal memory-bandwidth utilization a trn2 decode token costs 2.0x trn1's, and the measured best is 1.81x ($0.851 /
$0.470). The byte floor per step at 320 rows per group, from tools/tensor_bytes.py per rank, assuming every expert and every
state row is read and every state row written: weights and experts 12.0 GB + KDA state 2 x 6.2 GB + KV at most 3.0 GB = 27.5 GB.
At 725 GB/s per logical core that is 37.9 ms against the measured 511 ms, an MBU of at most 7.4%. trn1's CP-96 by the same count
is 16.9 GB at 440 GB/s, 38 ms against 302.6 ms, at most 12.7%. The trn2 step is a fixed ~96 ms plus 0.31-0.38 ms per row (EP:
254.8 ms at 512 rows, 413.2 ms at 1,024, 511.2 ms at 1,280). The per-row cost is about 6x the bytes a row adds, so the step is
bound by per-row work and per-step fixed costs, not by bandwidth. Closing the 1.8x on trn2 would take an MBU above trn1's, about
14% here, by cutting those two costs; more rows per group do not get there (320 already sits at the 24 GiB limit).

**trn2 long prompts (lever-1 R8: DP attention 1, CP row groups of 8, EP; one lone request; docs/neuron-notes.md "Long prompts
on trn2"):**
- TTFT for one engine on half the box: 32k 9.93 s, 128k 39.67 s, 260k 79.34 s. That is $0.65 per 1M input tokens at
  half-box spot.
- The whole box as a 2-stage pipeline (async + overlap): 5.91 / 21.82 / 43.06 s. That is 1.84x lower TTFT at $0.71 per 1M.
- trn1.32xlarge, the same R8 configuration (the lc agent's): 8.85 / 35.16 s at 32k / 128k, and 307,200 in 83.98 s. That is
  $0.163 per 1M input at trn1 spot, so a quarter of trn2's price per token for a lone long prompt.
- Before feat/trn2-next 8bb542c + c0391ae, the 4096-page bucket (prompts past 262,144 tokens) answered the needle wrong
  (0 / 3 at 128k, 300k and 1M) or faulted, depending on the piece size, while trn1 passed at 1M.
- The fix is LNC=2 only, so trn1's keys are unchanged. It addresses two causes:
  - a broadcast-table gather that the compiler gets wrong in that graph, now a 1-D gather;
  - one score scratch shared by both programs of the selection kernels, which now select on program 0.
- At the default P=6 it passes the needle 3 / 3 at 128k (4096 bucket only), at 300k and at 1M.
- TTFT on half the box with the fix: 300k in 93.69 s and 1M in 344.32 s, that is $0.65 and $0.70 per 1M input at half-box
  spot. The P=1 mitigation it replaces took 123.06 s and 443.20 s ($0.90 per 1M at 1M).
- The whole box as a 2-stage pipeline (async + overlap) with the fix does 300k in 50.96 s and 1M in 185.84 s. At full-box
  spot that is $0.76 per 1M input at 1M: a 1.85x lower TTFT for 8% more per token.
- lc2's final configuration (R8 + LOCAL_K 120 + LONG_PIPE + MERGE_BOUND) on trn2 with the fix passes the 1M needle 3 / 3. It does
  1M in 315.69 s on half the box ($0.64 per 1M input) and in 172.04 s as the whole box in 2 stages ($0.70 per 1M); the pipeline's
  first tokens equal one engine's bit for bit.
- The trn1.32xlarge R8 does 1M in 317.8 s for $0.18 per 1M.
- So at 1M, trn2 on half a box costs $0.64-0.70 per 1M input against trn1 R8's $0.18. That is 3.6-3.9 times as much, at
  about the same TTFT for one box (315.69 s against 317.8 s). trn2 is correct at 1M now, but it is still not the 1M prefill
  vehicle.

## trn2 standing before EP (feat/trn2-fast c08c5ba, 2026-10-06 03:20 UTC)

trn2.48xlarge (EC2 Capacity Block cr-00ff977628a81fb28, ap-south-2b, kiln-t2-cb), SDK 2.32, LNC=2, GLM-5.3-Flash real weights,
two tp=32 engines (`--dp 2`), DP attention 4, farm graphs (q/t2f-*), 0 device compiles. $ at trn2 spot $15.343/h (us-east-2c, the
like-for-like basis with p5en spot; the block itself cost $37.20/h). Commands, logs and gates: docs/neuron-notes.md "trn2 on
engine-v0 70ddc1b".

**G1 (8192 in / 256 out), whole box, the trn2 defaults of c08c5ba** (`KILN_MOE_PREFILL_SKIP=20` on top; q/t2f-best1F, log
s3 logs/kiln-t2-cb/20261006T*-t2-U-best1F):

| conc | out tok/s | engine-v0 70ddc1b (same box) | TTFT p50 | ITL p50 | prefill call | $/1M out (trn2 spot) | p5en vLLM spot $/1M out |
|---|---|---|---|---|---|---|---|
| 16 | 133.9 | 118.5 | 8.3 s | 88 ms | 0.870 s | 31.83 | 25.7-27.1 |
| 32 | 180.9 | 158.9 | 8.6 s | 142 ms | 0.883 s | 23.56 | 9.49-9.99 |
| 64 | 208.9 | 177.3 | 8.7 s | 267 ms | 0.885 s | 20.40 | 5.48-5.77 |
| 128 | 229.5 | 192.0 | 8.9 s | 506 ms | 0.888 s | 18.57 | 3.4-3.6 |
| 256 | 243.9 | 198.4 | 9.1 s | 966 ms | 0.899 s | 17.47 | - |

(+13 to +23% over engine-v0 on the same box; best1 before the fused DSA default: 131.3 / 176.0 / 202.4 / 221.6 / 235.1.) Not won
anywhere: trn2's best level costs 4.6x trn1's G64 default per token ($17.47 at conc 256 against $3.82 at conc 64, "Where it
stands"), because prefill is 66-87% of the device time and trn2 prefills at about trn1's rate per box.

**Prefill parity** (prefill tok/s per box = 2 engines x rows per call / prefill call): **8.8k** at 4096-row calls, **9.9k** at
8192-row calls (`--prefill-tokens 8192 --prefill-buckets 2048 KILN_PIECEWISE_PREFILL_MOE_GROUP=4`, +9-12% per engine); trn1.32xlarge
8.7k one-piece, 10.2k with EPLB. Parity of BF16 efficiency with trn1 would be ~35k (3.5x the compute). Where the 4096-row call goes
(replayed with captured inputs): MoE blocks 52%, token mixers 38%, 101 GB of spill per rank, tensor engine 13.5% of peak.

**Decode box with real KV** (the PD decode side; tools/time_decode.py --real-kv, 8K contexts, one engine, ST + v9 on D2K2,
feat/trn2-fast-ds2 ef289f8):

| rows per DP group (per engine) | ms per step | out tok/s per engine | $/1M out at trn2 spot | MBU |
|---|---|---|---|---|
| 16 (64) | 78.52 | 815 | 2.61 | 20% |
| 32 (128) | 104.95 | 1,220 | 1.75 | 19% |
| 64 (256) | 147.23 | 1,739 | 1.23 | 17% |
| 96 (384) | 217.50 | 1,766 | **1.21** | 14% |

96 rows per group is the 8K-context ceiling (22.4 GB of tensors per rank; 25.7 GB at 128 does not fit). The whole box is 3,531
out tok/s at $1.21 per 1M out; trn1's decode box is $0.89 (the decode agent, ST + v9, 28 rows per group at real 8K KV), so at 8K
trn1 is the cheaper box on both sides of PD. trn2's case is context lengths trn1 holds less of.

**trn1 decode box with real KV and context-parallel DSA** (the decode agent, feat/decode-scale-f997 e34272d = engine-v0
3df0b63 + this branch; ST + v10 + `KILN_DSA_CP=1` over each group's 8 ranks, 256-token pages, page bucket 64;
tools/time_decode.py --real-kv, 8K contexts, decode-only, trn1.32xlarge spot $2.15/h; `tools/dc_cpA.sh`, logs s3
logs/kiln-dc-32/tdA-*.log; every number two passes on one tree):

| rows per DP group (per step) | config | ms per step | out tok/s | $/1M out |
|---|---|---|---|---|
| 64 (256) | CP base | 237.4 | 1,078 | 0.554 |
| 64 (256) | + `KILN_DSA_CP_ALL_LOCAL=1 KILN_DSA_CP_PAGE_KEYS=1` | 215.6 | 1,187 | 0.503 |
| 96 (384) | CP base | 336.6 | 1,141 | 0.523 |
| 96 (384) | + `KILN_DSA_CP_ALL_LOCAL=1 KILN_DSA_CP_PAGE_KEYS=1` | 302.6 | 1,269 | 0.470 |
| 96 (384) | + `KILN_DSA_CP_MERGE_BOUND=1` (feat/decode-next d4f0700) | 297.1 | 1,293 | 0.462 |
| 96 (384) | + `KILN_DSA_CP_MERGE_BOUND=1 KILN_DSA_CP_DECODE_COMPACT=1` (feat/decode-next d4f0700) | **275.1** | **1,396** | **0.428** |

The decode side of PD under $0.5 per 1M out at 8K (decode-only: input is priced separately). The two flags skip the
local selection when a rank's pools fit keep and read the pool keys as page rows (exact; docs/neuron-notes.md "One DSA
layer under CP, op by op"). The last two rows (the decode agent's next round, two passes each on one tree, kiln-dc2-32,
`tools/dc_next.sh`, logs s3 logs/kiln-dc2-32/tdn-*-m-*.log): the merge's tie search over the bucket's bits (bit-exact on the
device) and each row's live slots compacted in the attention kernel (kernels/dsa_slots_c.py; fp32 summation order only;
the real-text PD gate inside its run-to-run floor). docs/neuron-notes.md "The decode box, next round" has both, and why
CP-112 does not fit.

**Open trn2 bugs** (repro configs in docs/neuron-notes.md; the EP hang at LNC=2 is fixed on feat/trn2-next, where expert
parallelism is the trn2 default): the MoE prefill kernel's LNC split fails the wikitext check (out-of-bound indirect copy, software
DGE too); sp_gather has no LNC=2 form (program 0 alone: NCC_ILLC059; both programs: wrong rows on 32 of 32 ranks, max |err|
7.48), so trn2 keeps the XLA zero-padded gather; the KDA / DSA decode row splits are exact in the simulator but not bit-identical
on the device (inside the decode-path gate).

## Earlier standing (engine-v0 ebe237e, final, 2026-10-05 03:30 UTC)

GLM-5.3-Flash (real weights), 8192 in / 256 out, 128 requests per level, trn1.32xlarge spot $2.15/h, tp=32,
DP attention 4, prefill 4096 per step, 12 MoE layers per graph, every default of engine-v0 ebe237e (expert
parallelism and group attention collectives on by their trn1 auto rules; FP8 KV with the separate pool-key
cache at conc 64). Farm graphs q/final-ebe237e, 0 device compiles. Logs
s3://<your-bucket>/logs/<box>/fin2-<config>.log, each with a `.log.cmd` holding the exact command;
configs in s3://<your-bucket>/compile-farm/q/final-ebe237e/configs/. p5en spot $/1M out is the
reference's vLLM throughput (311 / 842 / 1458 out tok/s at conc 16 / 32 / 64) at the p5en.48xlarge spot band
$28.77-30.29/h.

| concurrency | config (box) | out tok/s | TTFT p50 / p90 | ITL p50 | Kiln spot $/1M out | p5en vLLM spot $/1M out | |
|---|---|---|---|---|---|---|---|
| 16 | G16-4096-KV0.65-S20-P12-K, default (kiln-mimo-trn1) | **87.3** | 8.7 / 8.7 s | 149 ms | **$6.84** | $25.7-27.1 | **won, 73-75% below** |
| 32 | F0-4096-KV1.5-S20-P12-K, default (kiln-mimo-trn1) | **112.3** | 6.5 / 28.5 s | 254 ms | **$5.32** | $9.49-9.99 | **won, 44-47% below** |
| 32 | F0 + KDA / DSA decode kernels (opt-in, kiln-g2-trn1) | 114.7 | 6.4 / 28.2 s | 248 ms | $5.21 | $9.49-9.99 | won, 45-48% below |
| 64 | G64-4096-KV1.5-S20-P12-K, default (kiln-dk-32) | **122.9** | 6.9 / 79.7 s | 452 ms | **$4.86** | $5.48-5.77 | **won, 11-16% below** |
| 64 | G64 + mixed batches (opt-in `KILN_MIXED_BATCH=1`, `--state-checkpoints 4`; kiln-dk-32) | 131.8 | 6.3 / 72.4 s | 419 ms | $4.53 | $5.48-5.77 | won, 17-21% below |
| 64 | G64 + KDA / DSA decode kernels + SP decode streams (opt-in; kiln-g2-trn1) | **137.1** | 6.5 / 75.4 s | 410 ms | **$4.36** | $5.48-5.77 | **won, 20-24% below** |
| 128 | does not fit trn1 (16 GiB per core); trn2 whole box 189.9 (KDA + DSA LNC split) | | | | $22.07 at trn2 spot $15.09/h | $3.4-3.6 | far |

Opt-in switches for the decode rows: `KILN_KDA_DECODE_KERNEL=nki KILN_DSA_DECODE_KERNEL=nki KILN_DECODE_SP=1`.
The combination of mixed batches with the decode kernels was not measured. Against provider list prices
($0.15 / $0.50 per 1M in / out, $0.00136 per 8192 / 256 request): the default conc 64 level costs $0.00124
per request (8% below), the decode-kernel level $0.00111 (18% below); DeepInfra's discounted $0.00068 is
1.6-1.8x below Kiln. On-demand is not won (trn1 on-demand is 10x its spot price), and latency is far behind
the GPU (ITL 149-452 ms against 20-32 ms).

## Utilization baseline (engine-v0 ebe237e, 2026-10-05): MFU 7.7% in prefill, MBU 14.7% in decode

Every later lever reports against these. GLM-5.3-Flash real weights, the G64 config above (q/final-ebe237e graphs),
trn1.32xlarge kiln-ut-32, SDK 2.32, neuron-explorer 2.32; rank 0 of one real prefill call (4 chunks x 1024 rows, real
routing) and one real decode call (16 live rows per group), each NEFF replayed on 32 workers with every rank's own
captured inputs (`KILN_CAPTURE_INPUTS`, `tools/util_report.py replay`, second execution profiled). Peaks per
NeuronCore-v2 from AWS's Trainium architecture page (190 TFLOPS bf16 / cFP8 and 820 GiB/s per 2-core device): 95 TFLOPS,
440 GB/s. Method, per-layer tables and traps: docs/neuron-notes.md "Accelerator utilization of the serving graphs".

| call | device ms | tensor / vector / scalar / gpsimd busy | HBM moved, rate | tensor engine | collectives | model metric |
|---|---|---|---|---|---|---|
| prefill, 4096 tokens | 592 (serving fit; replay 615) | 29.7 / 34.2 / 27.3 / 2.2% | 33.6 GB (18.5 spill), 55 GB/s = 12.4% | 18.1 TFLOP/s = 19% | 183.5 ms; every engine idle with one in flight ~28% | **MFU 7.7%** (33.83 GFLOP per token x 4096 / (0.592 s x 32 x 95 TFLOPS)) |
| decode, 64 rows | 178 (runtime trace; replay 189) | 27.1 / 47.9 / 16.3 / 10.6% | 18.8 GB (6.9 spill), 100 GB/s = 22.6% | 5.05 TFLOP/s = 5.3% | 19 ms; all idle 16.5 ms | **MBU 14.7%** (11.52 GB per rank must move / 0.178 s / 440 GB/s) |

Where the conc 16 / 32 / 64 wall goes (fin2 runs' `device_split`; 32,768 output tokens per level): prefill 228.7 / 151.6 /
151.6 s (61 / 52 / 57%), decode 139.7 / 136.1 / 113.7 s, per output token 11.45 / 8.91 / 8.14 ms. Prefill is the same 256
full calls at every level (token slots 100%) and cannot batch further; decode costs ~50 ms + ~1.6 ms per row per call;
padded decode rows cost 0.8 / 1.4 / 4.7% of the wall; host gaps ~0 (rank 0 blocked on the device in 648 of 648 steps,
device 99.6% busy in the traced window). Even free decode would cap conc 64 at 216 out tok/s.

Measured against it on the same box (kiln-ut-32, conc 64 unless stated; iteration log below): `--decode-buckets 8,16`
126.7 (+3.3%, config only; 4,8,12,16 no better); mixed batches + KDA / DSA decode kernels + SP decode streams together 141.4
(+15.2%), with buckets 8,16 as well 142.9 (+16.5%, $4.18 / 1M spot); conc 16 EP with the MoE agent's decode v2 +12.8% over TP
at identical shapes. Two-batch overlap is closed on neuronx-cc 2.27 (it schedules no independent compute inside a collective);
the prefill call's largest single loss is EP load imbalance, ~117 ms of 592 (EPLB, the techniques agent).

## Earlier standing (engine-v0 9853406, 2026-10-04 23:00 UTC)

GLM-5.3-Flash, 8192 in / 256 out, trn1.32xlarge spot $2.15/h, tp=32, DP attention 4, prefill 4096 per
step (1024 rows per group), 12 MoE layers per graph, `KILN_MOE_PREFILL_SKIP=20`, reduce-scatter SP
(`KILN_SP_RS`, default), FP8 KV with the separate pool-key cache (`KILN_DSA_POOL_CACHE=auto`), KV sized for
every sequence, farm graphs q/final-c33a391 (0 device compiles), 128 requests per level, kiln-g2-trn1;
logs s3://<your-bucket>/logs/kiln-g2-trn1/fin-<config>.log, each with a `.log.cmd` holding the
exact command (also under "How to resume"). p5en spot $/1M out is the notebook's vLLM throughput (842 out
tok/s at conc 32, 1458 at conc 64) at the p5en spot band $28.77-30.29/h.

| concurrency | config | out tok/s | TTFT p50 / p90 | ITL p50 | Kiln spot $/1M out | p5en vLLM spot $/1M out | |
|---|---|---|---|---|---|---|---|
| 16 | G16-4096-KV0.65-S20-P12-K | 82.9 | 9.4 / 9.4 s | 156 ms | $7.20 | $25.7-27.1 | **won, 72-73% below**  EP stays off here (4 decode rows per group: EP measured 80.5, -2.9%) |
| 16, group collectives (2b3bdbe; TP, the EP gate leaves 4 decode rows per group off) | G16, q/pfgrp6-trn1 G16-...-GRP, kiln-dk-32 log 20261005T001414Z-fin-g16-grp | **86.5** | 8.7 / 30.4 s | 149 ms | **$6.90** | $25.7-27.1 | **won, 73-75% below** |
| 32 | F0-4096-KV1.5-S20-P12-K | 90.8 | 9.7 / 42.3 s | 311 ms | $6.58 | $9.49-9.99 | **won, 31-34% below** |
| 32, expert parallelism (engine-v0 cc2a000: on from 8 decode rows per group) | F0 + EP, q/stack-24ed, kiln-mimo-trn1 log 20261004T203230Z-ep24-F0 | **98.9** | 7.9 / 34.3 s | 288 ms | **$6.04** | $9.49-9.99 | **won, 36-40% below** |
| 32, tile-scale EP (engine-v0 f9dc4c4) | F0, q/pfgrp6-trn1 F0-...-EPT-BASE, kiln-pf-32b log 20261005T001744Z-fin-f0-base | 104.0 | 7.1 / 43.5 s | 269 ms | $5.74 | $9.49-9.99 | won, 40-43% below |
| 32, tile-scale EP + group collectives (2b3bdbe) | F0, q/pfgrp6-trn1 F0-...-EPT-GRP, kiln-pf-32b log 20261005T000827Z-fin-f0-grp (same box as the row above) | **110.1** | 6.5 / 39.5 s | 254 ms | **$5.42** | $9.49-9.99 | **won, 43-46% below** |
| 64 | G64-4096-KV1.5-S20-P12-K | 94.9 | 10.1 / 115.8 s | 604 ms | $6.29 | $5.48-5.77 | 9-15% above: 103.5-108.9 out tok/s wins |
| 64, expert parallelism (engine-v0 f974a11: `KILN_MOE_EP=auto`, on for glm5_next on trn1) | the same config + EP, q/ep-trn1 G64-...-EP-2ca0, kiln-mimo-trn1 log 20261004T192313Z | **106.9** | 8.3 / 95.8 s | 526 ms | **$5.59** | $5.48-5.77 | **inside the band**: below its top, 2% above its bottom (108.9 beats all of it) |
| 64, EP + group collectives (feat/prefill-grp2: `KILN_SP_GROUP=auto`, on for trn1) | the same command, farm graphs q/pfgrp3-trn1, kiln-pf-32b log 20261004T212831Z-ep2-grp (same-box EP baseline 107.0, log 20261004T210420Z-ep2-base) | **114.6** | 7.6 / 87.5 s | 489 ms | **$5.21** | $5.48-5.77 | **won, 5-10% below**; ppl at dp-attention 4 unchanged (-2.091, wikitext -0.551) |
| 64, tile-scale EP (engine-v0 9853406, the default) | the same config, q/ept-edc0 G64, kiln-mimo-trn1 log 20261004T222230Z-ept-G64 (kiln-dk-32 measured the same 114.9) | **114.9** | 7.6 / 87.1 s | 487 ms | **$5.20** | $5.48-5.77 | **won, 5-10% below** |
| 64, tile-scale EP + mixed batches (opt-in `KILN_MIXED_BATCH=1`, `--state-checkpoints 4`) | q/ept-edc0 G64-MX-CK4, kiln-dk-32 log 20261004T222144Z-ept-G64-mx-ck4 | **121.3** | 7.0 / 81.2 s | 459 ms | **$4.92** | $5.48-5.77 | **won, 10-15% below** |
| 64, tile-scale EP + group collectives (feat/prefill-grp2 2b3bdbe = engine-v0 f9dc4c4 + `KILN_SP_GROUP=auto`, on for trn1) | the same config, q/pfgrp6-trn1 G64-...-EPT-GRP, kiln-pf-32b log 20261004T235630Z-fin-g64-grp | **123.1** | 6.9 / 79.4 s | 453 ms | **$4.85** | $5.48-5.77 | **won, 11-16% below**; 4096-token prefill call 0.591 s |
| 64, tile-scale EP + mixed batches CK4 + group collectives (2b3bdbe) | q/pfgrp6-trn1 G64-...-EPT-MX-CK4-GRP, kiln-dk-32 log 20261005T002109Z-fin-g64mx-grp | **132.0** | 6.3 / 72.4 s | 419 ms | **$4.52** | $5.48-5.77 | **won, 18-22% below** |
| 128 | does not fit trn1; trn2 whole box 189.9 on feat/trn2-max 7138a25 (the KDA and DSA kernels split over both physical cores, per-token identical to engine-v0; engine-v0: 172.0). The MoE prefill split's 220.6-240.9 is NOT valid: that tree fails the wikitext check with an out-of-bound indirect copy | | | | $22.07 at trn2 spot $15.09/h | $3.4 | far |
| 32, tile-scale EP + KDA and DSA decode kernels (feat/decode-step a64d7f8, opt-in `KILN_KDA_DECODE_KERNEL=nki KILN_DSA_DECODE_KERNEL=nki`) | q/decode-2968ed3 F0-EPT-DK2, kiln-g1-trn1 log ab-F0-EPT-DK2 (same box, base F0-EPT: 106.0, $5.63) | **108.0** | 7.0 / 30.7 s | 264 ms | **$5.53** | $9.49-9.99 | **won, 42-45% below** |
| 64, tile-scale EP + KDA and DSA decode kernels + SP decode streams (feat/decode-step a64d7f8, opt-in, plus `KILN_DECODE_SP=1`) | q/decode-2968ed3 G64-EPT-DKS2 on engine-v0 2968ed3 (before `KILN_SP_GROUP`; the two together are not measured), kiln-dk-32 log ab-G64-EPT-DKS2 (same box: base 114.9, kernels alone 122.2) | **128.0** | 7.2 / 82.6 s | 442 ms | **$4.67** | $5.48-5.77 | **won, 15-19% below** |
| 128 | does not fit trn1; trn2 whole box 172.0 on engine-v0 (the LNC split's 220.6 is NOT valid: that tree fails the wikitext check with an out-of-bound indirect copy, so its sweeps may have read stale rows) | | | | $24.37 at trn2 spot $15.09/h | $3.4 | far |

Expert parallelism (merged, default for glm5_next on trn1): conc 64 94.9 -> 106.9 out tok/s (+12.6%) on the same
base and box, real-weight ppl -2.074 / wikitext -0.551; EP at conc 16 / 32 is being measured. Mixed prefill+decode
batches (merged, opt-in `KILN_MIXED_BATCH=1`): +1.8% on TP at conc 64; on top of EP being measured; trn2 LNC split +19-28% in sweeps that are NOT valid yet
(the split tree fails the trn2 wikitext check with an out-of-bound indirect copy; engine-v0 alone passes). Prompt
caching does not apply to this sweep (no shared prefix); on a shared-prefix workload see GOAL.md G1b. Decode step (feat/decode-step, opt-in): KDA and DSA decode kernels and SP decode streams take conc 64 from 114.9 to
128.0 out tok/s on the same box ($4.67 / M) with the decode-path NLL unchanged; the decode-only step at 64 rows per group
is $0.55 / M (docs/neuron-notes.md "Where a GLM-5.3-Flash decode step goes"). MTP speculative decoding (feat/glm-mtp, opt-in `--spec-method mtp`)
was measured on real weights at every level and on G1b's warm prefix workload and loses everywhere today (EP conc 64
106.9 -> 102.0 out tok/s, conc 16 k=2 80.5 -> 52.9, G1b warm 229.6 -> 223.1) despite 96-97% acceptance on these
prompts: the iteration log and docs/neuron-notes.md "MTP speculative decoding on GLM-5.3-Flash at serving scale".

On-demand vs SageMaker on-demand is not won at any level (trn1 on-demand is 10x its spot price). Latency
is far behind the GPU at every level (TTFT 9-10 s vs ~1 s, ITL 156-604 ms vs 20-43 ms): the win is on
throughput per dollar only.

## G1b: cost per request against model providers, with prompt caching (2026-10-04)

Owner goal G1b (GOAL.md): cheaper per request than any model provider on the G1 workload. **How the provider bill is
computed here:** per request, (uncached input tokens x $0.15 + cached input tokens x $0.03 + output tokens x $0.50) /
1M at the list prices, where the cached tokens are the ones Kiln itself served from its prefix cache in that run (the
measured hit rate: the provider is assumed to cache the same share); DeepInfra's discounted $0.075 / $0.25 with its
cached input assumed at $0.015 (the list's 1/5; no cached price was given). Kiln's cost per request is the instance
price over the measured requests per second ($2.15/h trn1.32xlarge spot / 3600 / req/s). "Kiln / bill" is the ratio
(below 1 wins); Kiln's price at the provider's ratios is that ratio times each list price. "At cost" splits the
level's dollars by the device time a least-squares fit of every step's wall time attributes to prefill and decode
graph calls (bench/serve_sweep.py device_split), per uncached input and per output token; cached input costs one
checkpoint copy (2-6 ms) and page reuse, ~0.

Workload: `bench/serve_sweep.py --shared-prefix-len N --num-prefixes 4`: every 8192-token prompt is one of 4 random
prefixes of N tokens (0 / 4096 / 6144 = 0% / 50% / 75% of the input), chosen at random, plus unique tokens; 256 output
tokens; closed loop at the stated concurrency. "Cold" levels start from a flushed cache (the first requests of each
prefix in each DP group compute it); "warm" levels follow a cold one with `--keep-cache` (the same prefixes, new
suffixes): a server that has been running.

Cold cache, engine-v0 1d63405 graph code + feat/prompt-cache, q/v1-trn1 F0 / G64-4096-KV1.5-S20-P12-K, kiln-dk-32,
128 (conc 32) / 256 (conc 64) requests per level, logs s3://<your-bucket>/logs/kiln-dk-32/pc-sw32.log,
pc-sw64.log:

| conc | cached share of input | hit rate (requests) | out tok/s | TTFT p50 (hits / misses) | ITL p50 | Kiln $/request | list bill | Kiln / bill (DeepInfra) | Kiln at list ratios $/1M in / cached / out | at cost $/1M uncached in / out |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0% | 0 (0/128) | 87.2 | 10.2 s | 324 ms | 0.001753 | 0.001357 | 1.29 (2.58) | 0.194 / 0.039 / 0.646 | 0.153 / 1.95 |
| 32 | 50% | 0.406 (104/128) | 109.5 | 5.7 s (5.7 / 10.2) | 259 ms | 0.001396 | 0.000957 | 1.46 (2.92) | 0.219 / 0.044 / 0.729 | 0.180 / 2.04 |
| 32 | 75% | 0.621 (106/128) | 128.5 | 3.4 s (3.4 / 10.2) | 219 ms | 0.001190 | 0.000746 | 1.59 (3.19) | 0.239 / 0.048 / 0.797 | 0.214 / 2.06 |
| 64 | 0% | 0 (0/256) | 90.6 | 10.8 s | 648 ms | 0.001687 | 0.001357 | 1.24 (2.49) | 0.187 / 0.037 / 0.622 | 0.150 / 1.80 |
| 64 | 50% | 0.461 (236/256) | 126.7 | 6.0 s (6.0 / 15.5) | 472 ms | 0.001206 | 0.000904 | 1.33 (2.67) | 0.200 / 0.040 / 0.667 | 0.173 / 1.73 |
| 64 | 75% | 0.697 (238/256) | 165.7 | 3.6 s (3.6 / 17.8) | 359 ms | 0.000923 | 0.000671 | 1.37 (2.75) | 0.206 / 0.041 / 0.687 | 0.197 / 1.70 |
| 64 | 75%, no lookahead junctions | 0.656 (224/256) | 150.4 | 3.6 s (3.6 / 25.1) | 391 ms | 0.001016 | 0.000712 | 1.43 (2.86) | 0.214 / 0.043 / 0.714 | 0.203 / 1.74 |

The merged tree (feat/prompt-cache on engine-v0 c33a391: prefill-perf3, SP reduce-scatter, separate pool keys under
FP8), q/final-c33a391 F0 / G64-4096-KV1.5-S20-P12-K graphs, kiln-dk-32, one engine per concurrency running the levels
in order with `--keep-cache` (`--shared-prefix-len 0 4096 4096 6144 6144`): each mix cold, then warm; logs pc-swF.log
(2026-10-04 19:33-19:59 UTC), pc-swG.log (20:00-20:41 UTC):

| conc | cached share of input | cache | hit rate (requests) | out tok/s | TTFT p50 (hits / misses) | ITL p50 | Kiln $/request | list bill | Kiln / bill (DeepInfra) | Kiln at list ratios $/1M in / cached / out | at cost $/1M uncached in / out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0% | - | 0 (0/128) | 90.8 | 9.7 s | 311 ms | 0.001684 | 0.001357 | 1.24 (2.48) | 0.186 / 0.037 / 0.621 | 0.144 / 1.97 |
| 32 | 50% | cold | 0.406 (104/128) | 113.5 | 5.4 s (5.4 / 9.7) | 251 ms | 0.001347 | 0.000957 | 1.41 (2.81) | 0.211 / 0.042 / 0.704 | 0.170 / 2.04 |
| 32 | 50% | warm | 0.500 (128/128) | 141.3 | 5.4 s | 204 ms | 0.001082 | 0.000865 | 1.25 (2.50) | 0.188 / 0.038 / 0.625 | 0.144 / 1.93 |
| 32 | 75% | cold | 0.621 (106/128) | 132.7 | 3.2 s (3.2 / 9.7) | 213 ms | 0.001152 | 0.000746 | 1.54 (3.09) | 0.232 / 0.046 / 0.772 | 0.204 / 2.03 |
| 32 | 75% | warm | 0.750 (128/128) | **195.3** | 3.2 s | 151 ms | **0.000783** | 0.000620 | 1.26 (2.53) | 0.190 / 0.038 / 0.632 | 0.145 / 1.89 |
| 64 | 0% | - | 0 (0/256) | 97.3 | 10.1 s | 604 ms | 0.001572 | 0.001357 | 1.16 (2.32) | 0.174 / 0.035 / 0.579 | 0.143 / 1.58 |
| 64 | 50% | cold | 0.461 (236/256) | 136.9 | 5.6 s (5.6 / 14.6) | 437 ms | 0.001117 | 0.000904 | 1.24 (2.47) | 0.185 / 0.037 / 0.618 | 0.164 / 1.53 |
| 64 | 50% | warm | 0.500 (256/256) | 158.6 | 5.6 s | 375 ms | 0.000964 | 0.000865 | 1.11 (2.23) | 0.167 / 0.033 / 0.557 | 0.143 / 1.47 |
| 64 | 75% | cold | 0.697 (238/256) | 180.4 | 3.4 s (3.4 / 16.9) | 329 ms | 0.000848 | 0.000671 | 1.26 (2.53) | 0.189 / 0.038 / 0.631 | 0.186 / 1.51 |
| 64 | 75% | warm | 0.750 (256/256) | **231.6** | 3.4 s | 260 ms | **0.000660** | 0.000620 | **1.065** (2.13) | 0.160 / 0.032 / 0.533 | 0.145 / 1.42 |

With longest-prefix-first admission (`--schedule-policy lpm`, pc-swGl.log, 21:20-21:35 UTC) the conc-64 75% levels
gave 183.7 cold (hit rate 0.691) and 231.5 warm out tok/s: the same within a run's noise, so FCFS stays the default.
The 0% levels match the lead's cold baselines on the same tree (90.8 at conc 32; 97.3 at conc 64 against 94.9 on
kiln-g2-trn1 with 128 requests, likely because a 256-request level spends less of its time ramping down). A warm cache is every
request a hit (4 prefixes, each held by every DP group: 4 junction checkpoints per group of the 16 / 32 rows); the cold
levels' misses are the first requests of each prefix in each group.

On engine-v0 e2c6fad (expert parallelism on by default for glm5_next on trn1), feat/prompt-cache rebased onto it,
q/e2c6fad-trn1 G64-4096-KV1.5-S20-P12-K graphs (the q/final-c33a391 command, EP from the default), kiln-dk-32,
`--shared-prefix-len 0 6144 6144 --keep-cache`, 256 requests per level, log pc-swE.log (2026-10-04 20:52-21:18 UTC):

| conc | cached share of input | cache | hit rate (requests) | out tok/s | TTFT p50 (hits / misses) | ITL p50 | Kiln $/request | list bill | Kiln / bill (DeepInfra) | Kiln at list ratios $/1M in / cached / out | at cost $/1M uncached in / out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 0% | - | 0 (0/256) | 110.9 | 8.3 s | 526 ms | 0.001379 | 0.001357 | **1.016** (2.03) | 0.153 / 0.031 / 0.508 | 0.110 / 1.88 |
| 64 | 75% | cold | 0.697 (238/256) | 187.2 | 2.8 s (2.8 / 13.7) | 318 ms | 0.000817 | 0.000671 | 1.22 (2.43) | 0.183 / 0.037 / 0.608 | 0.146 / 1.77 |
| 64 | 75% | warm | 0.750 (256/256) | 229.9 | 2.8 s | 263 ms | 0.000665 | 0.000620 | 1.073 (2.15) | 0.161 / 0.032 / 0.537 | 0.111 / 1.71 |

Expert parallelism makes a prefill call 23% cheaper (0.746 against 0.973 s per 4096-row call, the device split of the
two runs) and a decode call 18% dearer (0.175 against 0.148 s at 16 rows per group): +14% at 0% cached, where prefill
is 65-74% of the device time, and nothing at 75% cached and warm (229.9 against 231.6 out tok/s), where decode is two
thirds of it. Arithmetic on these two runs: the warm 75% level with EP's prefill call and TP's decode call would run
about 10% faster, ~$0.000602 per request, 0.97x the list bill; both expert layouts at once do not fit (10 GB of experts
per rank each), so it would take an EP decode call as cheap as TP's.

**Mixed traffic and prefill packing** (feat/dp-prefill-pack, engine-v0 f9dc4c4, EP, kiln-g1-trn1, 2026-10-05; prompts
of 6400-8192 tokens of which 6144 shared by 4 prefixes, 192 requests per level, docs/neuron-notes.md "Packing prefill
chunks across DP-attention groups"): conc 64 warm 207.2 out tok/s with packing off and 241.1 with trim (opt-in,
KILN_DP_PREFILL_PACK=trim) ($0.000738 -> $0.000634 per request at spot); cold 164.7 -> 196.9. With prompt lengths spread over the groups a
prefill call had carried one or two of its four chunks, which uniform 8192-token prompts never showed.

**What it says.** With 75% of the input cached and the cache warm, conc 64 serves 231.6 out tok/s (2.38x the
uncached 97.3) at $0.000660 per request: 58% below the same requests uncached ($0.001572), half the uncached list
price of a request ($0.00136), and 1.065x the list bill with cached input billed at $0.03 (2.13x DeepInfra's discounted
prices). The at-cost split is why it is not below: a cached token costs Kiln about nothing, an uncached input token
$0.14-0.15 per 1M warm (the list's $0.15; cold levels run $0.16-0.20 because their prefill calls run partly empty when
only some DP groups have a chunk), and an OUTPUT token $1.42-1.97 per 1M against the list's $0.50. With most input
cached the request is decode-bound, so beating the list bill needs a cheaper output token (more rows per decode call,
MTP, trn2), not more caching.

### trn2.48xlarge against trn1 and the providers (G1b, 2026-10-04 23:45 UTC)

Spot prices read 2026-10-04 (describe-spot-price-history, us-east-2): trn2.48xlarge $15.0887/h (2c), trn1.32xlarge
$2.15/h (2c), p5en.48xlarge $28.79-31.81/h. Cost per request = price per second / (out tok/s / 256), the G1 workload
(8192 in / 256 out). Provider list prices per request: $0.15 / $0.50 per 1M = **$0.00136**; DeepInfra $0.075 / $0.25 =
**$0.00068**; a fully cached prompt at $0.03 / 1M input = $0.00037.

trn2 whole box = two tp=32 engines, DP attention 4, prefill 4096 / 1024, 6 MoE layers per prefill graph, decode buckets
4/8/16/32, KV 2.75 GB fp8, max-num-seqs 128 per engine, SP streams, farm graphs (commands in docs/neuron-notes.md "trn2:
sequence-parallel streams pass wikitext ..."). Valid rows only: the LNC split of the MoE prefill kernel (+19-28% in its
sweeps) fails the wikitext check and is not counted.

| conc | trn2 engine-v0 21e9c8c out tok/s | trn2 feat/trn2-max 7138a25 (KDA + DSA split) out tok/s | trn2 $ / M out | trn2 $ / request | trn1 engine-v0 out tok/s | trn1 $ / M out | trn1 $ / request | p5en spot $ / M out |
|---|---|---|---|---|---|---|---|---|
| 16 | 109.6 | | 38.25 | 0.00979 | 82.9 | 7.20 | 0.00184 | 25.7-27.1 |
| 32 | 145.1 | | 28.89 | 0.00740 | 90.8 | 6.58 | 0.00168 | 9.49-9.99 |
| 64 | 160.2 | **175.5** | 23.88 | 0.00611 | 106.9 (EP) | 5.59 | 0.00143 | 5.48-5.77 |
| 128 | 172.0 | **189.9** | **22.07** | **0.00565** | does not fit | | | 3.4 |
| 256 | 176.9 | | 23.70 | 0.00607 | does not fit | | | |

($ columns: the best valid trn2 number of the row.) Per token trn2 is 3.9x trn1 at their best levels ($22.07 against
$5.59 per 1M out), and a trn2 request costs 4.2x the $0.15 / $0.50 list price and 8.3x DeepInfra's; trn1 at conc 64 with
EP is 1.05x the list price. Matching the list price on trn2 spot needs 3.09 req/s = 791 out tok/s per box (4.2x today);
trn1 needs 112 out tok/s (1.05x today). trn2 has about 3.5x trn1's HBM bandwidth and compute per box at 7x the spot
price, so trn2 reaches trn1's cost per token only with a lever worth more than 2x on trn2 alone: the cost vehicle for G1b
is trn1; trn2's case today is capacity per box (conc 128 and 256 fit) and G4.

## Iteration log

Measured with `bench/serve_sweep.py` (8192 in / 256 out unless stated).

The $ column is as recorded at the time, each row on the basis it names. trn1 spot rows are at the $2.15/h that was
obtained; "trn2 spot" figures are at a spot quote that was never obtained, and "block price" figures at the Capacity Block
actually bought ("Pricing basis (corrected 2026-10-08)" above). The rows are not rewritten.

| date | engine-v0 | instance | config | conc 16 / 32 / 64 / 128 output tok/s | $ / 1M out at conc 128 |
|---|---|---|---|---|---|
| 2026-10-03 | ac19b7b | trn1.32xlarge | tp=32, XLA MoE, decode only, B=4, short prompts | 15 tok/s at B=4 (not this workload) | - |
| 2026-10-03 | c4fdd02 + feat/trn2-bench | trn2.48xlarge (Capacity Block, ap-south-2b) | tp=32 on 32 of 64 cores, NKI MoE kernel, MoE group 4, prefill 256 | conc 8: 16.4 tok/s (TTFT p50 15.1 s, ITL p50 407 ms) | $614.67 at conc 8, whole-box block price $36.29/h (not like for like with SageMaker on-demand) |
| 2026-10-03 | same | same | same, conc 16 | 15.6 tok/s (TTFT p50 127 s, ITL p50 489 ms): prefill-bound, about 500 prompt tok/s per tp=32 engine | $646.19 at conc 16 (same basis) |
| 2026-10-03 | same | same | whole box: tp=32 x dp=2 (`bench/serve_sweep.py --dp 2`) | conc 16 / 32 / 64: 32.7 / 34.0 / 34.5 tok/s (ITL p50 407 / 456 / 456 ms) | $308.27 / $296.49 / $292.19 at the block price $36.29/h; p5en $65.02 / $24.02 / $13.87 on SageMaker on-demand (bases differ) |
| 2026-10-03 | 35bf2c7 | trn1.32xlarge | tp=32, NKI dedupe MoE kernel, 12 layers per graph, decode only, B=4, short prompts | 80.4 tok/s at B=4 (not this workload); ppl -2.086 | - |
| 2026-10-03 | 35bf2c7 | trn1.32xlarge | same, B=32 | 274.5 tok/s at B=32 (not this workload) | - |
| 2026-10-04 | 1d7243e | trn1.32xlarge | **first 8192 / 256 sweep**: tp=32, DP 1, concurrency 8, prefill chunk 512, P=4, NKI decode MoE, `--model-type=transformer`, graphs from the compile farm (0 device compiles) | conc 8: 14.4 out tok/s; TTFT p50 27.2 s / p90 94.7 s; ITL p50 447 ms | spot $41.47 / M out (p5en spot at its lowest level, conc 16: $25.9); on-demand $414.74 (SageMaker p5en: $65.02) |
| 2026-10-04 | trn2 agent tree | trn2.48xlarge (capacity block, $36.29/h equivalent) | tp=32 on 32 of 64 logical cores (LNC=2), DP 1, concurrency 16, default compiler flags | conc 16: 17.9 out tok/s; TTFT p50 109.4 s / p90 174.8 s; ITL p50 426 ms | $563.16 / M out for the whole block (half of the box was idle in this run: $281 for the half used) |
| 2026-10-04 | 1d7243e | trn1.32xlarge | tp=32, DP 4, concurrency 32, prefill 1024 (256 per group), P=2, NKI decode MoE, `--model-type=transformer`, farm graphs | conc 32: 18.4 out tok/s; TTFT p50 96.8 s / p90 344.7 s; ITL p50 1409 ms; ~590 prompt tok/s | spot $32.46 / M out (p5en spot at conc 32: $9.57); on-demand $324.58 (SageMaker p5en: $24.02) |
| 2026-10-04 | engine-v0 before d3cc7fd | trn1.32xlarge | **MiMo-V2.6-Flash** (not the G1 model), tp=32, DP 1, concurrency 32, decode bucket 32, prefill 2048 (one bucket), P=4, NKI decode and prefill MoE kernels, KV 2.0 GB, `--model-type=transformer`, farm graphs | conc 32: 42.7 out tok/s; TTFT p50 152.3 s / p90 158.8 s; ITL p50 139.6 ms | spot $13.99 / M out (p5en spot at conc 32 on GLM-5.3-Flash: $9.57); on-demand $139.86 |
| 2026-10-04 | d3cc7fd | trn2.48xlarge (capacity block) | **whole box**: tp=32 x 2 engines (`--dp 2`, LNC=2), DP attention 4 per engine, concurrency 64 (32 per engine), decode 8 per group, prefill 2048 (512 per group), P=2, NKI decode MoE, `--model-type=transformer`, farm graphs | conc 64: 41.2 out tok/s; TTFT p50 80.8 s / p90 281.3 s; ITL p50 1276 ms | $244.67 / M out at the block price $36.29/h (SageMaker p5en on-demand $13.87; bases differ) |
| 2026-10-04 | 63f683b | trn1.32xlarge | tp=32, DP 4, concurrency 64, decode 16 per group, prefill 1024 (256 per group), P=2, NKI decode MoE, KV 1.0 GB fp8, `--model-type=transformer`, farm graphs | conc 64: 19.4 out tok/s; TTFT p50 414.4 s / p90 666.2 s; ITL p50 1613 ms | spot $30.78 / M out (p5en spot at conc 64: $5.5); on-demand $307.85 (SageMaker p5en: $13.87) |
| 2026-10-04 | 63f683b | trn2.48xlarge (capacity block) | **whole box**: tp=32 x 2 engines, DP attention 4, concurrency 128 (64 per engine), decode 16 per group, prefill 2048 (512 per group), P=2, NKI decode MoE, KV 2.0 GB fp8, `--model-type=transformer`, farm graphs | conc 128: 48.8 out tok/s; TTFT p50 43.4 s / p90 530.9 s; ITL p50 2399 ms | $206.57 / M out at the block price (SageMaker p5en on-demand $8.57; p5en spot $3.4) |
| 2026-10-04 | 63f683b | trn2.48xlarge (capacity block) | **whole box**: tp=32 x 2 engines, DP attention 4, concurrency 16 (8 per engine), decode 4 per group, prefill 2048 (512 per group), P=2, NKI decode MoE, KV 0.5 GB, `--model-type=transformer`, farm graphs | conc 16: 42.6 out tok/s; TTFT p50 41.8 s / p90 79.1 s; ITL p50 212 ms | $236.63 / M out at the block price (SageMaker p5en on-demand $65.02). The whole box gives 42-49 out tok/s at every concurrency: prefill-bound |
| 2026-10-04 | 63f683b | trn1.32xlarge | tp=32, DP 4, concurrency 16, decode 4 per group, prefill 1024 (256 per group), P=2, NKI decode MoE, KV 0.5 GB, `--model-type=transformer`, farm graphs | conc 16: 14.1 out tok/s; TTFT p50 95.6 s / p90 281.3 s; ITL p50 871 ms | spot $42.36 / M out (p5en spot at conc 16: $25.9); on-demand $423.56 (SageMaker p5en: $65.02) |
| 2026-10-04 | scratch/kda-d25e855-equiv c88747a (63f683b + feat/kda-kernel 4fc7ab3) | trn1.32xlarge | **throughput A/B, numerics not yet accepted**: the F0 config (concurrency 32, DP 4, prefill 1024) plus elementwise mHC (`KILN_MHC_FORM=elementwise` at 4fc7ab3 = fp32 mix), the prefill MoE kernel (`KILN_MOE_PREFILL_KERNEL=nki`), torch delta rule | conc 32: 31.4 out tok/s (+71% over 18.4); TTFT p50 54.8 s / p90 197.2 s; ITL p50 813 ms | spot $19.02 / M out (p5en spot $9.57); on-demand $190.20. Real-weight ppl of this form on the device is -1.785 against -2.098 on CPU: the device computes the elementwise output mix differently from the CPU (feat/kda-kernel investigation), so it is not merged |
| 2026-10-04 | scratch/kda-d25e855-equiv c88747a | trn2.48xlarge (capacity block) | **throughput A/B, numerics not yet accepted**: whole box, tp=32 x 2 engines, the T2 config (prefill 2048, 512 per group) plus elementwise mHC (fp32 mix) and the prefill MoE kernel, concurrency 64 | conc 64: 74.3 out tok/s (+80% over 41.2); TTFT p50 41.2 s / p90 145.4 s; ITL p50 693 ms | $135.67 / M out at the block price (SageMaker p5en on-demand $13.87) |
| 2026-10-04 | scratch/kda-d25e855-equiv c88747a (load decisions) | trn2.48xlarge (capacity block) | **throughput A/B**: whole box, 2 engines, T128 config (concurrency 128, 64 per engine, decode 16 per group, prefill 2048, KV 2.0 GB fp8) plus elementwise mHC (fp32 mix) and the prefill MoE kernel | conc 128: 90.4 out tok/s (+85% over 48.8); TTFT p50 22.8 s / p90 277.7 s; ITL p50 1263 ms | $111.51 / M out at the block price (SageMaker p5en on-demand $8.57; at trn2 spot $14.80/h it would be $45.5) |
| 2026-10-04 | 6a7a2e3 (= efa1da4 graph code) | trn1.32xlarge | **merged tree**, F0 config (concurrency 32, DP 4, decode 8 per group, prefill 1024 = 256 per group, KV 1.0 GB) with **12 MoE layers per prefill graph** (`KILN_PIECEWISE_PREFILL_MOE_GROUP=12`: 3 prefill graphs per step, no prefill spill), elementwise mHC (Veltkamp), prefill MoE kernel, farm graphs | conc 32: 38.4 out tok/s; TTFT p50 42.5 s / p90 157.4 s; ITL p50 659 ms | spot $15.55 / M out (p5en spot $9.57); on-demand $155.53 (SageMaker p5en $24.02) |
| 2026-10-04 | 6a7a2e3 | trn1.32xlarge | merged tree, F0 config with P=12 **and the KDA NKI kernel** (`KILN_LINEAR_ATTN_KERNEL=nki`), farm graphs | conc 32: 42.0 out tok/s (+9% over 38.4 without it); TTFT p50 38.4 s / p90 142.7 s; ITL p50 600 ms | spot $14.22 / M out (p5en spot $9.57); on-demand $142.20 |
| 2026-10-04 | 6a7a2e3 | trn1.32xlarge | F0 config with P=12 and **prefill 2048 / bucket 512** (16.57 GB by the farm estimate, above anything loaded before; it loads), torch delta rule, farm graphs (kiln-g1-trn1 log 20261004T080045Z) | conc 32: 41.7 out tok/s (+9% over prefill 1024 at 38.4); TTFT p50 34.6 s / p90 125.2 s; ITL p50 612 ms | spot $14.32 / M out (p5en spot $9.57); on-demand $143.22 |
| 2026-10-04 | 6a7a2e3 | trn1.32xlarge | F0 config, P=12, prefill 2048 / bucket 512, **KDA NKI kernel**, farm graphs (kiln-g1-trn1 log 20261004T081651Z) | conc 32: **46.3 out tok/s**; TTFT p50 30.5 s / p90 110.6 s; ITL p50 548 ms | spot $12.90 / M out (p5en spot $9.57: 35% above it; parity needs ~62 out tok/s); on-demand $128.99 (SageMaker $24.02) |
| 2026-10-04 | feat/prefill-perf (engine-v0 5006371 + sequence-parallel prefill streams) | trn1.32xlarge | the same conc-32 config with `KILN_PREFILL_SP=1` (each rank keeps 1/32 of the mHC stream rows between prefill layers), 64 requests (kiln-g1-trn1 log 20261004T084143Z-sweep-sp1, prefill agent) | conc 32: **57.2 out tok/s** (+23.5% over 46.3); TTFT p50 23.2 s / p90 85.1 s; ITL p50 437 ms; ppl -2.073 | spot $10.44 / M out (p5en spot $9.57: 9% above it); on-demand $104.41 |
| 2026-10-04 | same tree, `KILN_PREFILL_SP=0` | trn1.32xlarge | same-box A/B partner of the row above (kiln-g1-trn1 log 20261004T092112Z-sweep-sp0) | conc 32: 46.3 out tok/s; TTFT p50 30.5 s; ITL p50 548 ms | spot $12.90 / M out |
| 2026-10-04 | 6a7a2e3 (efa1da4 tree) | trn1.32xlarge | **DP attention 2** (attention TP 16), conc 32, decode 16 per group, prefill 2048 / bucket 1024, P=12, KDA kernel, no SP, farm graphs (kiln-mimo-trn1 log 20261004T093404Z) | conc 32: 37.1 out tok/s (DP 4: 46.3); TTFT p50 116.5 s; ITL p50 345 ms | spot $16.10 / M out: DP 2 loses |
| 2026-10-04 | origin/scratch/sp-merge 57ed80a (engine-v0 + feat/prefill-perf 46ebcd0) | trn1.32xlarge | the same conc-32 config (prefill 2048 / bucket 512, P=12, KDA kernel, DP 4, decode 8 per group, KV 1.0 GB), 64 requests, every graph from the cache (`NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`), kiln-dk-32 log sw-Ap.log, the A of the next two rows | conc 32: 57.3 out tok/s; TTFT p50 23.2 s / p90 85.1 s; ITL p50 436.5 ms | spot $10.42 / M out |
| 2026-10-04 | feat/dsa-topk b495338 (sp-merge + exact NKI DSA selection + pool-key cache + prefill score kernel, all on by default) | trn1.32xlarge | the same config and box, right after the row above (kiln-dk-32 sw-Ep.log) | conc 32: **64.4 out tok/s** (+12.4%); TTFT p50 22.2 s / p90 81.0 s; ITL p50 366.0 ms; decode step (tools/time_decode.py) 141.2 -> 128.3 ms | **spot $9.27 / M out, below p5en spot at conc 32 ($9.57)**; on-demand $92.74 (SageMaker p5en $24.02) |
| 2026-10-04 | feat/dsa-topk b495338 | trn1.32xlarge | the row above with two 24-layer decode graphs (`KILN_PIECEWISE_MOE_GROUP=24 KILN_PIECEWISE_PREFILL_MOE_GROUP=12`; decode step 128.3 -> 120.7 ms), same box (sw-Fp.log) | conc 32: 64.7 out tok/s (+0.5%, one run each: the decode step is 6% shorter but the ITL waits behind prefill); TTFT p50 22.1 s / p90 80.7 s; ITL p50 364.4 ms | spot $9.23 / M out; on-demand $92.31 |
| 2026-10-04 | 6a7a2e3 | trn1.32xlarge | merged tree, F0 config with P=2 (23 prefill graphs per step), the A/B partner of the P=12 row above | conc 32: 26.8 out tok/s (P=12: 38.4, **+43% from 12-layer prefill graphs**); TTFT p50 66.4 s / p90 232.4 s; ITL p50 945 ms | spot $22.28 / M out |
| 2026-10-04 | 6a7a2e3 | trn2.48xlarge (capacity block) | merged tree, whole box T2 config (2 engines, concurrency 64, prefill 2048 = 512 per group) with **P=12** (3 prefill graphs per step), farm graphs | conc 64: 99.7 out tok/s (P=2 unrounded: 74.3); TTFT p50 27.9 s / p90 101.8 s; ITL p50 507 ms | $101.11 / M out at the block price (SageMaker p5en on-demand $13.87; at trn2 spot $14.80/h ~$41.2) |
| 2026-10-04 | 6a7a2e3 | trn2.48xlarge (capacity block) | same T2 whole-box config with P=12 **and the KDA NKI kernel** | conc 64: 101.7 out tok/s (+2% over 99.7); TTFT p50 27.3 s / p90 99.5 s; ITL p50 497 ms | $99.12 / M out at the block price |
| 2026-10-04 | 6a7a2e3 | trn2.48xlarge (capacity block) | whole box, concurrency 64, **prefill 4096 / bucket 1024** (1024 rows per group), P=6 (2.2M instructions per prefill group at most), KDA kernel, farm graphs | conc 64: 109.7 out tok/s (+8% over prefill 2048); TTFT p50 26.5 s / p90 93.1 s; ITL p50 449 ms | $91.89 / M out at the block price |
| 2026-10-04 | 6a7a2e3 | trn2.48xlarge (capacity block) | whole box, concurrency 128 (64 per engine, decode 16 per group, KV 2.0 GB fp8), prefill 4096 / bucket 1024, P=6, KDA kernel, farm graphs | conc 128: 132.2 out tok/s (prefill 2048: 125.3); TTFT p50 15.4 s / p90 177.5 s; ITL p50 879 ms | $76.25 / M out at the block price (SageMaker p5en on-demand $8.57; at trn2 spot $14.80/h ~$31.1, p5en spot $3.4) |
| 2026-10-04 | scratch/sp-merge 57ed80a (engine-v0 + SP) | trn2.48xlarge (capacity block) | whole box, concurrency 64, prefill 4096 / bucket 1024, P=6, KDA kernel, **sequence-parallel prefill streams**, farm graphs | conc 64: **131.9 out tok/s** (+20% over 109.7); TTFT p50 20.5 s / p90 72.5 s; ITL p50 369 ms | $76.43 / M out at the block price (at trn2 spot $14.80/h ~$31.2; SageMaker p5en on-demand $13.87) | **Numerics not accepted: SP on trn2 moves real-weight ppl to -1.807 (SP=0: -2.100); throughput only.**
| 2026-10-04 | scratch/sp-merge 57ed80a | trn2.48xlarge (capacity block) | whole box, concurrency 128 (64 per engine, decode 16 per group, KV 2.0 GB fp8), prefill 4096 / 1024, P=6, KDA kernel, SP streams | conc 128: **163.0 out tok/s** (+23% over 132.2); TTFT p50 12.1 s / p90 139.2 s; ITL p50 708 ms | $61.84 / M out at the block price (at trn2 spot ~$25.2; SageMaker p5en on-demand $8.57, p5en spot $3.4) | **Numerics not accepted: SP on trn2 moves real-weight ppl to -1.807 (SP=0: -2.100); throughput only.**
| 2026-10-04 | scratch/sp-merge 57ed80a | trn2.48xlarge (capacity block) | whole box, concurrency 16 (8 per engine, decode 4 per group, KV 0.5 GB), prefill 4096 / 1024, P=6, KDA kernel, SP streams, 48 requests | conc 16: **103.6 out tok/s**; TTFT p50 11.1 s / p90 19.8 s; ITL p50 113 ms | $97.30 / M out at the block price (at trn2 spot ~$39.7; p5en spot $25.7-27.1, trn1 spot $12.04) | **Numerics not accepted: SP on trn2 moves real-weight ppl to -1.807 (SP=0: -2.100); throughput only.**
| 2026-10-04 | 6a7a2e3 | trn2.48xlarge (capacity block) | merged tree, whole box T128 config (2 engines, concurrency 128, decode 16 per group, prefill 2048, KV 2.0 GB fp8) with P=12, farm graphs | conc 128: 122.7 out tok/s (P=2: 88.2); TTFT p50 16.4 s / p90 199.5 s; ITL p50 915 ms | $82.16 / M out at the block price (SageMaker p5en on-demand $8.57; at trn2 spot $14.80/h ~$33.5, p5en spot $3.4) |
| 2026-10-04 | 6a7a2e3 | trn2.48xlarge (capacity block) | same T128 whole-box config with P=12 and the KDA NKI kernel | conc 128: 125.3 out tok/s (+2% over 122.7); TTFT p50 16.1 s / p90 194.9 s; ITL p50 896 ms | $80.45 / M out at the block price |
| 2026-10-04 | 6a7a2e3 | trn2.48xlarge (capacity block) | merged tree, whole box T16 config (2 engines, concurrency 16, decode 4 per group, prefill 2048, KV 0.5 GB) with P=12, 64 requests | conc 16: 86.9 out tok/s; TTFT p50 15.2 s / p90 27.4 s; ITL p50 128 ms | $116.00 / M out at the block price (SageMaker p5en on-demand $65.02); at trn2 spot $14.80/h ~$47.3 against p5en spot $25.7-27.1 and trn1 spot $21.48 |
| 2026-10-04 | 6a7a2e3 | trn2.48xlarge (capacity block) | same T16 whole-box config with P=12 and the KDA NKI kernel, 64 requests | conc 16: 88.1 out tok/s (+1% over 86.9); TTFT p50 14.8 s / p90 26.8 s; ITL p50 127 ms | $114.42 / M out at the block price |
| 2026-10-04 | 6a7a2e3 | trn1.32xlarge | **merged tree, G16 config with P=12** (concurrency 16, DP 4, decode 4 per group, prefill 1024 = 256 per group, KV 0.5 GB, 3 prefill graphs per step), farm graphs, 64 requests, log kiln-mimo-trn1 20261004T064626Z | conc 16: **27.8 out tok/s**; TTFT p50 27.3 s / p90 72.0 s; ITL p50 411 ms (p5en: 1.4 s / 12.3 s / 20.1 ms) | **spot $21.48 / M out, 16-21% below p5en spot at conc 16 ($25.7 at $28.77/h, $27.1 at $30.29/h): the first G1 level won on a like-for-like (spot vs spot) basis**, latency still far behind; on-demand $214.83 vs SageMaker $65.02. **Repeated on a second box** (kiln-g1-trn1, log 20261004T071931Z): 27.8 out tok/s, TTFT p50 27.3 s, ITL 411 ms, $21.48 again |
| 2026-10-04 | 6a7a2e3 | trn1.32xlarge | G16 config with P=12 **and the KDA NKI kernel**, farm graphs, 64 requests (kiln-mimo-trn1 log 20261004T073102Z) | conc 16: **30.4 out tok/s**; TTFT p50 24.7 s / p90 65.9 s; ITL p50 374 ms | **spot $19.65 / M out, 24-27% below p5en spot ($25.7-27.1)**; on-demand $196.45 |
| 2026-10-04 | 6a7a2e3 | trn1.32xlarge | G16 config with P=12 and **prefill 2048 / bucket 512** (512 rows per group; fits HBM at 15.8 GB by the farm estimate, which P=2 at 2048 did not), torch delta rule, farm graphs, 64 requests (kiln-mimo-trn1 log 20261004T075619Z) | conc 16: **36.8 out tok/s** (+32% over prefill 1024); TTFT p50 23.1 s / p90 52.3 s; ITL p50 300 ms | **spot $16.23 / M out, 37-40% below p5en spot ($25.7-27.1)**; on-demand $162.29 (SageMaker $65.02) |
| 2026-10-04 | 6a7a2e3 | trn1.32xlarge | G16 config, P=12, prefill 2048 / bucket 512, **KDA NKI kernel**, farm graphs, 64 requests (kiln-mimo-trn1 log 20261004T081301Z) | conc 16: **40.6 out tok/s**; TTFT p50 20.3 s / p90 47.8 s; ITL p50 271 ms | **spot $14.71 / M out, 43-46% below p5en spot ($25.7-27.1)**; on-demand $147.10 (SageMaker $65.02) |
| 2026-10-04 | scratch/sp-merge 57ed80a | trn1.32xlarge | G16 config, prefill 2048 / bucket 512, P=12, KDA kernel, **SP streams**, farm graphs, 64 requests (kiln-mimo-trn1 log 20261004T100227Z) | conc 16: **49.6 out tok/s**; TTFT p50 15.5 s / p90 40.1 s; ITL p50 221 ms | **spot $12.04 / M out, 53-56% below p5en spot ($25.7-27.1)**; on-demand $120.41 (SageMaker $65.02) |
| 2026-10-04 | scratch/sp-merge 57ed80a | trn1.32xlarge | G16 config with **prefill 4096 / bucket 1024** (1024 rows per group), P=12, KDA kernel, SP streams (17.79 GB by the farm rule, above the 17.18 GB physical, and it loads: the rule overestimates), farm graphs, 64 requests (kiln-mimo-trn1 log 20261004T102205Z) | conc 16: **62.8 out tok/s** (+27% over prefill 2048); TTFT p50 13.3 s / p90 30.8 s; ITL p50 168 ms | **spot $9.51 / M out, 63-65% below p5en spot ($25.7-27.1)**; on-demand $95.10 (SageMaker $65.02) |
| 2026-10-04 | scratch/sp-merge 57ed80a | trn1.32xlarge | the same G16 prefill-4096 config with **KV 0.65 GB** (1291 pages per group vs 4 x 264 = 1056 needed; KV 0.5 had 993), farm graphs, 64 requests (kiln-g1-trn1 log 20261004T123130Z) | conc 16: **72.6 out tok/s** (+15% over KV 0.5); TTFT p50 10.9 s / p90 28.5 s; ITL p50 177 ms | **spot $8.23 / M out, 68-70% below p5en spot ($25.7-27.1)**; on-demand $82.26 (SageMaker $65.02) |
| 2026-10-04 | engine-v0 3a55b9c graph code (DSA tree) | trn1.32xlarge | the same G16 prefill-4096 KV 0.65 config on the DSA tree (1235 pages per group), same box as the row above (kiln-g1-trn1 log 20261004T124134Z) | conc 16: **77.7 out tok/s** (+7% over sp-merge 72.6 on the same box: with KV sized the DSA tree wins here too; its -3.7% at KV 0.5 was the KV shortage); TTFT p50 10.3 s / p90 27.1 s; ITL p50 165 ms | **spot $7.69 / M out, 70-72% below p5en spot**; on-demand $76.86 |
| 2026-10-04 | engine-v0 3a55b9c graph code (DSA tree) | trn1.32xlarge | the same G16 prefill-4096 config on the DSA tree, kiln-dk-32 (log 20261004T104508Z) | conc 16: 60.6 out tok/s; ITL p50 169 ms. **Same-box A/B on kiln-dk-32: sp-merge 62.9 (log 20261004T105824Z), so the DSA tree is -3.7% here** (it is +7..12% at conc 32) | spot $9.86 / M out (sp-merge $9.49) |
| 2026-10-04 | scratch/sp-merge 57ed80a | trn1.32xlarge | F0 config with SP, farm graphs (kiln-g1-trn1 log 20261004T101132Z): confirms the prefill agent's 57.2 on its own tree | conc 32: 57.2 out tok/s; TTFT p50 23.2 s; ITL p50 437 ms | spot $10.44 / M out (p5en spot $9.57) |
| 2026-10-04 | scratch/sp-merge 57ed80a | trn1.32xlarge | F0 config (conc 32) with **prefill 4096 / bucket 1024**, P=12, KDA kernel, SP streams (18.59 GB by the farm rule; it loads), farm graphs, 64 requests (kiln-mimo-trn1 log 20261004T103550Z) | conc 32: **67.3 out tok/s** (prefill 2048: 57.2); TTFT p50 19.2 s / p90 68.9 s; ITL p50 360 ms | **spot $8.87 / M out, 7% below p5en spot ($9.57)**; on-demand $88.74 |
| 2026-10-04 | engine-v0 3a55b9c graph code (DSA tree, tree 787bf1fa) | trn1.32xlarge | F0 config (conc 32), prefill 4096 / bucket 1024, P=12, KDA kernel, SP streams, exact NKI DSA top-k + pool-key cache + fused prefill scores, farm graphs, 64 requests (kiln-mimo-trn1 log 20261004T105200Z) | conc 32: **72.1 out tok/s**; TTFT p50 18.8 s / p90 65.2 s; ITL p50 324 ms | **spot $8.28 / M out, 13.5% below p5en spot ($9.57)**; on-demand $82.83 (SageMaker $24.02). **Repeated on the same box with 96 requests (log 20261004T112723Z): 74.8 out tok/s, TTFT p50 18.8 s / p90 55.9 s, ITL 323 ms, spot $7.98 / M (17% below)** |
| 2026-10-04 | engine-v0 3a55b9c graph code (DSA tree) | trn1.32xlarge | the same conc-32 prefill-4096 config with **KV 1.25 GB** (2375 pages per group vs 8 x 264 = 2112 needed; KV 1.0 had 1900), farm graphs, 96 requests (kiln-mimo-trn1 log 20261004T121340Z) | conc 32: **84.6 out tok/s** (+13% over 74.8 on the same box); TTFT p50 10.6 s / p90 55.8 s; ITL p50 332 ms | **spot $7.06 / M out, 26% below p5en spot ($9.57)**; on-demand $70.59 (SageMaker $24.02) |
| 2026-10-04 | engine-v0 3a55b9c graph code (DSA tree) | trn1.32xlarge | the same with KV 1.5 GB (2850 pages per group) on another box (kiln-g1-trn1 log 20261004T125134Z) | conc 32: 84.6 out tok/s, identical to KV 1.25 on kiln-mimo-trn1: not KV-bound, and reproducible across boxes | spot $7.06 / M out |
| 2026-10-04 | feat/dsa-topk b495338 (sp-merge + exact NKI DSA top-k + pool-key cache + fused prefill score kernel) | trn1.32xlarge | F0 config (conc 32, prefill 2048, P=12, KDA kernel, SP), 64 requests, same-box A/B against sp-merge at 57.3 (DSA agent, kiln-dk-32) | conc 32: **64.4 out tok/s** (+12.4%); TTFT p50 22.2 s / p90 81.0 s; ITL p50 366 ms; 4 sentences -2.073 | **spot $9.27 / M out, below p5en spot at conc 32 ($9.57)**; on-demand $92.70 (SageMaker $24.02). Wikitext-2 -0.5517 vs -0.5481 (greedy 98.0%). **Repeated on kiln-g1-trn1 with the same tree (log 20261004T103213Z): 64.3 out tok/s, $9.29 / M** |
| 2026-10-04 | 6a7a2e3 | trn1.32xlarge | merged tree, G64 config with P=12 (concurrency 64, decode 16 per group, prefill 1024, KV 1.0 GB fp8), farm graphs | conc 64: 39.3 out tok/s; TTFT p50 200.0 s / p90 318.2 s; ITL p50 778 ms. trn1 saturates near 39 out tok/s (conc 32: 38.4) | spot $15.20 / M out (p5en spot $5.5); on-demand $151.96 |
| 2026-10-04 | 6a7a2e3 | trn1.32xlarge | G64 config with P=12 and the KDA NKI kernel, farm graphs | conc 64: 42.7 out tok/s (+9% over 39.3); TTFT p50 182.9 s / p90 290.6 s; ITL p50 712 ms | spot $13.99 / M out (p5en spot $5.5); on-demand $139.86 |
| 2026-10-04 | 6a7a2e3 | trn1.32xlarge | G64 config, P=12, **prefill 2048 / bucket 512**, KDA kernel (17.10 GB by the farm estimate: it loads), farm graphs (kiln-mimo-trn1 log 20261004T083512Z) | conc 64: **54.4 out tok/s** (prefill 1024: 42.7); TTFT p50 31.9 s / p90 218.5 s; ITL p50 976 ms | spot $10.98 / M out (p5en spot $5.5); on-demand $109.78 (SageMaker $13.87) |
| 2026-10-04 | scratch/sp-merge 57ed80a | trn1.32xlarge | G64 config, prefill 2048 / bucket 512, P=12, KDA kernel, **SP streams** (16.54 GB by the farm estimate), farm graphs (kiln-g1-trn1 log 20261004T095509Z) | conc 64: **67.7 out tok/s** (+24% over 54.4); TTFT p50 25.1 s / p90 171.0 s; ITL p50 770 ms | spot $8.82 / M out (p5en spot at conc 64 $5.5); on-demand $88.22 (SageMaker $13.87) |
| 2026-10-04 | engine-v0 3a55b9c graph code (feat/dsa-topk b495338) | trn1.32xlarge | the same G64 config on the DSA tree (exact NKI top-k, pool-key cache, fused prefill scores), same box (kiln-g1-trn1 log 20261004T104257Z) | conc 64: 66.6 out tok/s (sp-merge 67.7: no gain here, unlike conc 32); TTFT p50 42.7 s; ITL p50 713 ms | spot $8.97 / M out. Open: the pool-key cache takes its bytes from --kv-cache-gb (G64 pages 3972 -> 3641), which may cap requests in flight at conc 64 |
| 2026-10-04 | engine-v0 3a55b9c graph code (DSA tree) | trn1.32xlarge | G64 config with **prefill 4096 / bucket 1024** (18.72 GB by the rule; it loads), DSA tree, SP, KDA kernel, farm graphs (kiln-g1-trn1 log 20261004T105929Z) | conc 64: **72.5 out tok/s**; TTFT p50 38.6 s / p90 126.3 s; ITL p50 656 ms | spot $8.24 / M out (p5en spot $5.5: parity needs ~108 out tok/s); on-demand $82.38 |
| 2026-10-04 | engine-v0 3a55b9c graph code (DSA tree) | trn1.32xlarge | G64 prefill-4096 config with **KV 1.25 GB fp8** (the KV 1.0 runs were KV-bound: 3972 pages per group vs 16 x 264 = 4224 needed at 8448 context), farm graphs, 96 requests (kiln-mimo-trn1 log 20261004T115113Z) | conc 64: **84.0 out tok/s** (+16% over KV 1.0); TTFT p50 47.9 s / p90 136.0 s; ITL p50 555 ms | spot $7.11 / M out (p5en spot $5.5: 29% above it); on-demand $71.10 (SageMaker $13.87) |
| 2026-10-04 | engine-v0 3a55b9c graph code (DSA tree) | trn1.32xlarge | the same G64 prefill-4096 config with **KV 1.5 GB fp8** (5462 pages per group vs 4224 needed), farm graphs, 128 requests (kiln-mimo-trn1 log 20261004T122641Z) | conc 64: **88.3 out tok/s**; TTFT p50 11.0 s / p90 126.3 s; ITL p50 652 ms. Repeated on kiln-g2-trn1 (log 20261004T130030Z): 88.3, identical | spot $6.76 / M out (p5en spot $5.5: 23% above it); on-demand $67.64 |
| 2026-10-04 | engine-v0 3a55b9c graph code (DSA tree) | trn1.32xlarge | the same with KV 1.75 GB fp8 (6372 pages), 128 requests (log 20261004T123942Z) | conc 64: 88.3 out tok/s, identical to KV 1.5: no longer KV-bound | spot $6.76 / M out |
| 2026-10-04 | **engine-v0 c7836c0** | trn1.32xlarge | G16-4096 KV 0.65 (conc 16), farm graphs q/v0-trn1, 64 requests (kiln-mimo-trn1 log 20261004T133451Z) | conc 16: **77.8 out tok/s**; TTFT p50 10.3 s / p90 27.1 s; ITL p50 165 ms | **spot $7.68 / M out, 70-72% below p5en spot ($25.7-27.1)**; on-demand $76.76 (SageMaker $65.02) |
| 2026-10-04 | engine-v0 c7836c0, `KILN_MOE_PREFILL_SKIP=20` | trn1.32xlarge | the same conc-16 config with the skip, compiled on the box (kiln-g1-trn1 log 20261004T134756Z) | conc 16: 78.3 out tok/s (+0.6%); TTFT p50 10.2 s; ITL p50 164 ms | spot $7.63 / M out |
| 2026-10-04 | **engine-v0 c7836c0** | trn1.32xlarge | F0-4096 KV 1.5 (conc 32), farm graphs q/v0-trn1, 96 requests (kiln-g2-trn1 log 20261004T133437Z) | conc 32: **84.6 out tok/s**; TTFT p50 10.6 s / p90 55.8 s; ITL p50 332 ms | **spot $7.06 / M out, 26% below p5en spot ($9.57)**; on-demand $70.59 |
| 2026-10-04 | engine-v0 c7836c0, `KILN_MOE_PREFILL_SKIP=20` | trn1.32xlarge | the same conc-32 config with the MoE prefill skip, graphs compiled on the box (2214 s of warm-up: the skip adds 30-150 s per MoE call site) (kiln-g2-trn1 log 20261004T134452Z) | conc 32: 85.5 out tok/s (+1.1%); TTFT p50 10.5 s; ITL p50 329 ms | spot $6.99 / M out |
| 2026-10-04 | engine-v0 1d63405 (v1), SKIP=20 | trn1.32xlarge | G16-4096 KV 0.65 bf16, farm graphs q/v1-trn1, 64 requests (kiln-g2-trn1 log 20261004T171126Z) | conc 16: **79.7 out tok/s** (v0 + skip 78.3); TTFT p50 9.9 s / p90 26.1 s; ITL p50 161 ms | **spot $7.49 / M out, 71-72% below p5en spot** |
| 2026-10-04 | engine-v0 1d63405 (v1), SKIP=20 | trn1.32xlarge | F0-4096 KV 1.5 bf16 (in-place pool keys), farm graphs q/v1-trn1, 96 requests (kiln-g2-trn1 log 20261004T170120Z) | conc 32: **86.8 out tok/s** (v0 + skip 85.5); TTFT p50 10.2 s / p90 53.6 s; ITL p50 324 ms | **spot $6.88 / M out, 28% below p5en spot ($9.57)** |
| 2026-10-04 | engine-v0 1d63405 (v1), SKIP=0 | trn1.32xlarge | G64-4096 KV 1.5 fp8, same box as the v0 baseline row (kiln-g1-trn1 log 20261004T171703Z) | conc 64: 84.2 out tok/s against v0 c7836c0 88.3 on the same box (-4.6%): with fp8 KV the v1 default turns the DSA pool-key cache off | spot $7.09 / M out |
| 2026-10-04 | engine-v0 8f41439 (1d63405 + feat/prefill-perf3: KILN_SP_RS and the MoE prefill kernel's round order / output ring), SKIP=20 | trn1.32xlarge | G64-4096 KV 1.5 fp8, the v1 command, farm graphs q/pfab-trn1, 128 requests, kiln-pf-32b back to back (logs 20261004T170941Z-v1-b1 88.4, 20261004T172416Z-v1-k1 87.8, 20261004T165624Z-v1-rs1 91.6) | conc 64: **91.6 out tok/s** against 88.4 for 1d63405 on the same box (+3.6%). All of it is KILN_SP_RS: profiled prefill steps 1064.8 / 1071.5 / 1011.7 ms (base / + kernel / + reduce-scatter, logs 20261004T174506Z, 181024Z, 175754Z-v1p-*); the kernel change, -3.8% per C=4096 call alone, is neutral in the prefill pieces (docs/neuron-notes.md) | spot $6.52 / M out |
| 2026-10-05 | **feat/prefill-grp2 2b3bdbe** (engine-v0 f9dc4c4 + `KILN_SP_GROUP`; CPU suite 712 passed / 10 skipped) | trn1.32xlarge | G64 / G64 mixed CK4 / F0 / G16 at prefill 4096 / 1024, farm graphs q/pfgrp6-trn1 (all-rank captures), 0 device compiles; kiln-pf-32b and kiln-dk-32 logs *-fin-*.log | conc 64: **123.1** out tok/s (114.9 base, +7.1%), mixed CK4 **132.0** (121.3, +8.8%); conc 32: **110.1** (same-box f9dc4c4 104.0, +5.9%); conc 16: **86.5** (82.9). Mean 4096-token prefill call 0.591 s (G64). ppl at dp-attention 4 unchanged by the group path on the same tree (-2.108 both; wikitext -0.547, -0.550 with it off) | spot $4.85 / $4.52 / $5.42 / $6.90 per M out |
| 2026-10-04 | engine-v0 1d63405 (v1: + in-place / auto DSA pool cache, reserve admission, SP-local routing), SKIP=20 | trn1.32xlarge | G64-4096 KV 1.5 fp8, farm graphs q/v1-trn1, 128 requests, kiln-g2-trn1 (logs 20261004T163054Z reserve, 20261004T164736Z eager) | conc 64: 88.3 (reserve) / 88.4 (eager) out tok/s, kv: fits, running mean 52 of 64, 0 preemptions; below v0 + SKIP=20 on the same box (89.9): under fp8 KV the auto pool cache is off, so pooled keys are recomputed (separate-cache build queued) | spot $6.76 / M out |
| 2026-10-04 | **engine-v0 c7836c0** (tree a8865bc4, every change merged) | trn1.32xlarge | G64-4096 KV 1.5 fp8 baseline (skip off), farm graphs q/v0-trn1, 128 requests (kiln-g1-trn1 log 20261004T130446Z) | conc 64: 88.3 out tok/s; TTFT p50 11.0 s / p90 126.2 s; ITL p50 651 ms | spot $6.76 / M out |
| 2026-10-04 | **engine-v0 c7836c0** (tree a8865bc4, every change merged), `KILN_MOE_PREFILL_SKIP=20` | trn1.32xlarge | G64-4096 KV 1.5 fp8, farm graphs q/v0-trn1, 128 requests (kiln-g2-trn1 log 20261004T131345Z) | conc 64: **89.9 out tok/s** (+1.8% over 88.3 without the skip); TTFT p50 10.8 s / p90 123.6 s; ITL p50 640 ms | spot $6.64 / M out (p5en spot $5.5: 21% above it) |
| 2026-10-04 | engine-v0 c7836c0, `KILN_PIECEWISE_MOE_GROUP=24` (prefill 12) | trn1.32xlarge | the same G64 config with 24-layer decode graphs (kiln-mimo-trn1 log 20261004T131246Z) | conc 64: 88.3 out tok/s, ITL 651.5 ms: identical to 12-layer decode to 0.1 ms, so either decode (25% of device time here) moves too little to show or the decode group knob does not reach the served decode graphs; open | spot $6.76 / M out |
| 2026-10-04 | feat/prefill-moe2 64d33fb (MoE prefill kernel tile skip, `KILN_MOE_PREFILL_SKIP=20` vs 0) | trn1.32xlarge | same-box A/Bs on kiln-g1-trn1 at KV 1.0 (prefill agent): G64-4096 73.8 vs 72.5 (+1.8%), F0-4096 72.8 vs 72.1 (+1.0%); F0-2048 on kiln-pf-32b 59.0 vs 58.0 | +1.0..1.8% end to end against -21..-26% per kernel call: real routing uses more blocks (C=4096: 388 vs 312 for uniform), so the skip leaves 26% of the tiles instead of 41% | merged default-off (compile +30-150 s per MoE call site) |
| 2026-10-04 | engine-v0 3a55b9c graph code (DSA tree) | trn1.32xlarge | G64 prefill-2048 config with 24-layer decode graphs (`KILN_PIECEWISE_MOE_GROUP=24 KILN_PIECEWISE_PREFILL_MOE_GROUP=12`), farm graphs (kiln-mimo-trn1 log 20261004T110810Z) | conc 64: 66.6 out tok/s, the same as 12-layer decode graphs (66.6) | no gain: decode is not what limits conc 64 on trn1 |
| 2026-10-04 | 6a7a2e3 | trn2.48xlarge (capacity block) | merged tree, whole box T128 (concurrency 128, P=2), farm graphs | conc 128: 88.2 out tok/s (unrounded tree: 90.4; the Veltkamp rounding costs ~2.5%); TTFT p50 23.4 s / p90 285.4 s; ITL p50 1294 ms | $114.29 / M out at the block price |
| 2026-10-04 | 6a7a2e3 | trn1.32xlarge | merged tree, G16 config (concurrency 16, P=2), 64 requests | conc 16: 21.2 out tok/s (unrounded tree: 22.7); TTFT p50 36.8 s / p90 93.8 s; ITL p50 541 ms | spot $28.17 / M out (p5en spot $25.7-27.1); on-demand $281.71 |
| 2026-10-04 | scratch/kda-d25e855-equiv c88747a | trn1.32xlarge | **throughput A/B, numerics not yet accepted**: the G64 config (concurrency 64, DP 4, prefill 1024, KV 1.0 GB fp8) plus elementwise mHC (fp32 mix) and the prefill MoE kernel | conc 64: 29.8 out tok/s (+54% over 19.4); TTFT p50 269.0 s / p90 417.9 s; ITL p50 1023 ms | spot $20.04 / M out (p5en spot at conc 64 $5.5); on-demand $200.41. trn1 levels off near 30 out tok/s (conc 32: 31.4) |
| 2026-10-04 | scratch/kda-d25e855-equiv c88747a | trn1.32xlarge | **throughput A/B, numerics not yet accepted**: the G16 config (concurrency 16, DP 4, decode 4 per group, prefill 1024, KV 0.5 GB) plus elementwise mHC (fp32 mix) and the prefill MoE kernel, farm graphs with load decisions | conc 16: 23.9 out tok/s (+70% over 14.1); TTFT p50 53.3 s / p90 161.5 s; ITL p50 505 ms | spot $24.99 / M out (p5en spot at conc 16: $25.7-27.1 at $28.77-30.29/h); on-demand $249.88 (SageMaker p5en $65.02). 32 requests. **Confirmation with 64 requests (log 20261004T060701Z): 22.7 out tok/s, TTFT p50 34.2 s, ITL p50 505 ms, spot $26.31: parity with the GPU on spot, not a win** |
| 2026-10-04 | feat/dsa-shape (engine-v0 b27600e + in-place pool keys + reserve admission) | trn1.32xlarge | **F0-4096** (conc 32, prefill 4096 / 1024, KV 1.0 GB bf16, KV-bound: 1985 pages per group vs 2112), q/dsa-trn1 F0-4096-P12-K, 64 requests, kiln-dk-32 sw-adm-F04 | conc 32: **75.1 out tok/s**, 0 preemptions (running max 28); TTFT p50 20.1 s / p90 65.9 s; ITL p50 299 ms. Same box: sp-merge 67.4 (8 preemptions), merged DSA tree 72.2 (16), this tree with eager admission 71.7 (8) | **spot $7.95 / M out, 17% below p5en spot ($9.57)**; on-demand $79.52 (SageMaker $24.02) |
| 2026-10-04 | feat/dsa-shape | trn1.32xlarge | **F0-2048** (conc 32, prefill 2048 / 512, KV 1.0 GB bf16, KV-bound), kiln-dk-32 sw-adm-F0 | conc 32: 66.7 out tok/s, 0 preemptions; TTFT p50 24.4 s; ITL p50 338 ms. Same box: sp-merge 57.2, merged DSA tree 64.3 (16 preemptions), this tree eager 60.5 (16; also 60.5 on kiln-g2-trn1) | spot $8.95 / M out |
| 2026-10-04 | feat/dsa-shape | trn1.32xlarge | **G16-4096** (conc 16, prefill 4096 / 1024, KV 0.5 GB bf16, KV-bound: 992 vs 1056), kiln-dk-32 sw-new-G16 (eager) and sw-adm-G16 (reserve) | conc 16: eager 66.9 out tok/s (TTFT p50 12.6 s, ITL 157 ms), reserve 66.5 (TTFT 19.5 s, ITL 132 ms, 3 per group). Same box: sp-merge 62.9, merged DSA tree 60.6 (lead's runs) | spot $8.93 / M out (eager) |
| 2026-10-04 | feat/dsa-shape | trn1.32xlarge | G16-4096 with **KV 0.65 GB** (fits), kiln-dk-32 sw-*-G16k | conc 16: this tree 77.2 out tok/s (reserve on kiln-g2-trn1: 77.4); same box: sp-merge 72.7, merged DSA tree 77.8 | spot $7.74 / M out (sp-merge $8.21) |
| 2026-10-04 | feat/dsa-shape | trn1.32xlarge | **G64-2048** (conc 64, KV 1.0 GB FP8, KV-bound: 3971 vs 4224; pool keys off under FP8), kiln-dk-32 | conc 64: reserve 65.1 out tok/s (0 preemptions), eager 64.6 (4). Same box: sp-merge 62.4, merged DSA tree 64.0 (3640 pages). This box's sp-merge is 62.4 against 67.7 on kiln-g1-trn1: compare G64 within a box | spot $9.17 / M out |
| 2026-10-04 | c33a391 | trn1.32xlarge | **final TP stack**: P=12, prefill 4096 / 1024, `KILN_MOE_PREFILL_SKIP=20`, SP + SP-local routing + reduce-scatter, FP8 KV with the separate pool-key cache (G64; G16/F0 bf16 KV), reserve admission, farm queue q/final-c33a391, 128 requests, kiln-g2-trn1 fin-*.log | conc 16 / 32 / 64: 82.9 / 90.8 / 94.9 out tok/s (TTFT p50 9.4 / 9.7 / 10.1 s, ITL p50 156 / 311 / 604 ms) | spot $7.20 / $6.58 / $6.29 per 1M out (p5en spot $25.7-27.1 / $9.49-9.99 / $5.48-5.77) |
| 2026-10-04 | feat/mixed-batch bf1e346 (engine-v0 a1351b7 + mixed batches) | trn1.32xlarge | **mixed batches** (`KILN_MIXED_BATCH=1`, rows SP layout, joint mixers: 16 decode rows per group ride in each prefill call) on the G64-4096-KV1.5 fp8 config with `--state-checkpoints 4` (the default 32 checkpoint rows per group do not leave room for the mixed graphs), farm graphs q/mx-trn1, 128 requests, same box as the unmixed partner (kiln-g1-trn1 logs 20261004T172937Z and 181007Z (repeat) -mx-sweep64-mx-ck4-rowsjoint / -mx-rj-repeat, 154746Z-mx-sweep64-plain-ck4) | conc 64: **90.4 out tok/s** twice (unmixed 88.3, also 88.3 with `--state-checkpoints 4`); TTFT p50 10.7 s / p90 123.5 s; ITL p50 633 ms. The first mixer form ("calls": each projection twice) was 84.8 | spot $6.61 / M out |
| 2026-10-04 | feat/mixed-batch bf1e346 | trn1.32xlarge | mixed batches (rows, joint, 8 decode rows per group) on the F0-4096-KV1.5 config (conc 32, bf16 KV), 96 requests, same box (logs 20261004T182212Z-mx-sweep32-mx-rj, 170233Z-mx-sweep32-plain) | conc 32: **86.0 out tok/s** (unmixed 84.6); TTFT p50 10.3 s; ITL p50 326 ms | spot $6.94 / M out |
| 2026-10-04 | **feat/mixed-batch d6f3a1b** (= engine-v0 b4f400f, i.e. c33a391 + router fix, + mixed batches) | trn1.32xlarge | the final-head G64 config exactly (q/final-c33a391 G64-4096-KV1.5-S20-P12-K: S20, SP route, SP reduce-scatter, pool-key cache separate under fp8, reserve admission) with `KILN_MIXED_BATCH=1 KILN_MIXED_SP=rows KILN_MIXED_MIXERS=joint --state-checkpoints 4`, 128 requests; same box, unmixed partners: the q/final-c33a391 config itself and with `--state-checkpoints 4` (kiln-g1-trn1 logs 20261004T185838Z-mx-final-mx-rj-ck4, 191106Z-mx-final-plain, 184413Z-mx-final-plain-ck4) | conc 64: **96.6 out tok/s** (unmixed 94.9, CK4 94.8: +1.8%); TTFT p50 9.9 s / p90 113.8 s; ITL p50 591 ms (unmixed 10.1 s / 605 ms) | **spot $6.18 / M out** (unmixed $6.29; p5en spot $5.5: 12% above it); on-demand $61.82 |
| 2026-10-04 | **feat/glm-mtp 82cc981** (engine-v0 b4f400f + the MTP fixes), TP | trn1.32xlarge | G64-4096 with KV 1.2 fp8 (MTP at KV 1.5 does not fit HBM) and **MTP k=1** (`--spec-method mtp --spec-k 1 --state-checkpoints 0`), farm q/mtp-82cc981, 128 requests, kiln-g1-trn1 log 20261004T213042Z-sweep-mtp1-g64kv12; same box without MTP (BASE-G64-KV1.2) 94.9 (214241Z) | conc 64: 88.2 out tok/s (-7.1%); 95.8% of drafts accepted, 1.958 tokens per verify; TTFT p50 20.1 s / p90 128.3 s; ITL p50 596 ms | spot $6.77 / M out (without MTP $6.29) |
| 2026-10-04 | feat/glm-mtp 4d38ee0 (engine-v0 e2c6fad merged), **EP** (`KILN_MOE_EP=1`) | trn1.32xlarge | the same with EP, q/mtp-ep-4d38ee0, kiln-mtp-trn1 log 20261004T213725Z-ep-sweep-mtp1-kv12; same box without MTP 106.9 (KV 1.2, 220105Z) / 107.0 (KV 1.5, 214919Z) | conc 64: 102.0 out tok/s (-4.6%); 96.7%, 1.967 tokens per verify; TTFT p50 16.8 s; ITL p50 501 ms | spot $5.86 / M out (without MTP $5.59) |
| 2026-10-04 | feat/glm-mtp 4d38ee0, EP | trn1.32xlarge | F0-4096 KV 1.5 bf16 (conc 32) with MTP k=1 (`--state-checkpoints 8`) and k=2 (`--state-checkpoints 0`), q/mtp-ep16-4d38ee0, kiln-mtp-trn1 logs 20261004T234248Z-ep-sweep-f0-mtp1, 224451Z-ep-sweep-f0-mtp2; same box without MTP 99.1 (20261005T002601Z-ep-sweep-f0-base2) | conc 32: k=1 88.6 out tok/s (-10.6%; 96.2%, 1.962), k=2 76.2 (-23.1%; 93.2%, 2.860) | spot $6.74 / $7.84 per M out (without MTP $6.03) |
| 2026-10-04 | feat/glm-mtp 82cc981 TP / 4d38ee0 EP | trn1.32xlarge | G16-4096 KV 0.65 bf16 (conc 16) with MTP k=2 (k=3 on EP), kiln-g1-trn1 220702Z (TP) and kiln-mtp-trn1 221519Z / 232746Z (EP); same boxes without MTP: TP 82.8 (215526Z), EP 80.5 (223207Z) | conc 16: TP k=2 51.8 (2.838 tokens per verify), EP k=2 52.9 (2.819), EP k=3 61.8 (3.455): prefill steps grow 31% (served-step profile, docs/neuron-notes.md) | spot $11.53 (TP k=2) against $7.21 |
| 2026-10-04 | feat/glm-mtp a8ede27 (engine-v0 34b0d5d merged), EP | trn1.32xlarge | **G1b**: G64-4096 KV 1.2 fp8, `--shared-prefix-len 6144 6144 --num-prefixes 4 --keep-cache`, 256 requests per level, MTP k=1 with 16 checkpoint rows, q/mtp-pc-a8ede27, kiln-g1-trn1 logs 20261004T232915Z-pc-ep-mtp1-c16, 234711Z-pc-ep-base | 75% cached, warm: 223.1 out tok/s against 229.6 without MTP (-2.8%; cold 181.3 against 186.7); 96.7%, 1.967 tokens per verify | spot $2.68 against $2.60 per M out |
| 2026-10-04 | engine-v0 21e9c8c | trn2.48xlarge spot ($15.09/h, us-east-2c, kiln-trn2-b) | **whole box, SP numerics accepted on trn2** (wikitext -0.549 with SP vs -0.552 without): 2 engines tp=32, DP attention 4, prefill 4096 / 1024, P=6, decode 4/8/16/32 per group, KV 2.75 GB fp8, max-num-seqs 128 per engine, one run over all five levels, farm graphs q/t2max-U1 (log 20261004T172300Z-sweep-U1-base) | conc 16 / 32 / 64 / 128 / 256: 109.6 / 145.1 / 160.2 / 172.0 / 176.9 out tok/s; TTFT p50 10.7-12.5 s; ITL p50 106 / 176 / 349 / 672 / 1304 ms | spot $38.25 / 28.89 / 26.17 / 24.37 / 23.70 per M out (p5en spot $25.7-27.1 / 9.57 / 5.5 / 3.4) |
| 2026-10-04 | feat/trn2-max 1be9d85 (21e9c8c + every NKI kernel split over the two physical cores of an LNC=2 core). **NOT VALID: this tree fails the trn2 wikitext check (vector-DGE out-of-bound in the MoE prefill split, docs/neuron-notes.md); throughput only** | trn2.48xlarge spot (kiln-trn2-b) | the same config and box, farm graphs q/t2max-U1S (log 20261004T175821Z-sweep-U1S) | conc 16 / 32 / 64 / 128 / 256: **130.4 / 179.7 / 204.5 / 220.6 / 228.3** out tok/s (+19 / +24 / +28 / +28 / +29%); TTFT p50 8.1-9.7 s; ITL p50 92 / 144 / 273 / 522 / 1001 ms | **spot $32.14 / 23.33 / 20.50 / 19.00 / 18.36 per M out**; per request (8192 / 256) $0.00823 / 0.00597 / 0.00525 / 0.00486 / 0.00470 (trn1 spot conc 64: $0.00167; $0.15 / $0.50 list: $0.00136) |
| 2026-10-04 | engine-v0 e2c6fad | trn2.48xlarge spot (kiln-trn2-b) | the trn2 whole-box config (2 engines, DP attention 4, prefill 4096 / 1024, P=6, decode 4/8/16/32, KV 2.75 fp8, SP), farm graphs q/t2max-EU, `NEURON_LIBTORCH_ASSERT_CACHE_HIT=1` (log 20261004T233801Z-sweep-EU) | conc 64 / 128: 171.8 / 185.5 out tok/s; TTFT p50 10.5 / 10.9 s; ITL p50 325 / 624 ms | spot $24.40 / 22.60 per M out |
| 2026-10-04 | **feat/trn2-max 7138a25** (e2c6fad + the KDA and DSA kernels split over both physical cores of an LNC=2 core, `KILN_LNC_SPLIT` default) | trn2.48xlarge spot (kiln-trn2-b) | the same config and box, q/t2max-FUd, ASSERT_CACHE_HIT (log 20261004T230236Z-sweep-FUd); wikitext and 4 sentences per-token identical to engine-v0, greedy 64 / 64 prompts identical | conc 64 / 128: **175.5 / 189.9** out tok/s (+2.2 / +2.4%); TTFT p50 10.2 / 10.6 s; ITL p50 319 / 609 ms | **spot $23.88 / 22.07 per M out**; per request $0.00611 / 0.00565 |
| 2026-10-05 | **feat/decode-step 7db28da** (engine-v0 2968ed3 + KDA and DSA decode kernels `KILN_KDA_DECODE_KERNEL=nki KILN_DSA_DECODE_KERNEL=nki`, SP decode streams `KILN_DECODE_SP=1`) | trn1.32xlarge | **decode-only step** (tools/time_decode.py, EP on, null-page KV, q/decode-2968ed3 DCT-BASE / DCT-DK / DCT-DKS2, kiln-g2-trn1 logs td-DCT-*.log) | 32 rows per group: 279.3 -> 225.4 (kernels) -> 183.3 ms (+ SP decode); 64 rows per group: 456.5 -> 270.2 -> 233.7 ms | decode-only spot $1.30 -> $0.86 / M out at 128 rows per step, $1.07 -> **$0.55** at 256 (list price $0.50) |
| 2026-10-05 | feat/decode-step a64d7f8 (7db28da + engine-v0 06ce6d5) | trn1.32xlarge | **serving A/B**, EP on, 128 requests, q/decode-2968ed3: G64-EPT base / + decode kernels / + SP decode streams back to back on kiln-dk-32; F0-EPT base / + decode kernels on kiln-g1-trn1 (logs ab-<config>.log) | conc 64: 114.9 -> 122.2 -> **128.0** out tok/s (ITL p50 487 -> 460 -> 442 ms); conc 32: 106.0 -> 108.0 | spot $5.20 -> $4.89 -> **$4.67** / M out at conc 64 (p5en spot $5.48-5.77); $5.63 -> $5.53 at conc 32. Decode-path NLL (check_mixed teacher-forced) +0.0003 / +0.0002 nats per token on LONG_TEXT / wikitext, inside the run-to-run floor; prefill graphs unchanged |
| 2026-10-05 | **feat/moe-kernel-tune 4c3d078** (engine-v0 e449b98 + kernels/moe_ep.py: decode v2 `KILN_MOE_EP_SMALL_V=2`, one pass per local expert with pairs) | trn1.32xlarge spot (kiln-mk-32) | **serving A/B**, EP on, the q/final-ebe237e G64 / F0 commands verbatim, V1 graphs q/final-ebe237e, V2 q/moek-trn1, ASSERT_CACHE_HIT, back to back (logs s3 logs/kiln-mk-32/*-sv-*) | conc 32: 110.1 -> **117.1** (+6.4%), conc 64: 122.8 -> **129.5** (+5.5%); decode call 0.125 -> 0.108 / 0.178 -> 0.158 s; twin v3 (bit-identical) 116.0 / 129.8; conc 16 at 4 rows per group (utilization agent, kiln-ut-32): TP 87.5, EP v2 98.0 | spot $/1M out at conc 32 / 64: $5.42 -> $5.10 / $4.86 -> $4.61 |
| 2026-10-05 | feat/utilization (kiln/ = engine-v0 ebe237e; host-only instrumentation) | trn1.32xlarge spot (kiln-ut-32) | **utilization baseline**: G64 with `KILN_TIMELINE` + `KILN_RT_INSPECT` (log ut-g64-tl), device-profile replays of one real prefill and one real decode call with every rank's captured inputs (section "Utilization baseline" above) | conc 64: 122.7 out tok/s (fin2 122.9); prefill MFU 7.7%, decode MBU 14.7%; host blocked on the device in 648 / 648 steps | spot $4.87 / M out |
| 2026-10-05 | engine-v0 e449b98, `--decode-buckets 8,16` (config only) | trn1.32xlarge spot (kiln-ut-32) | G64 with a second decode bucket, farm q/ut-trn1 G64-DB8, ASSERT_CACHE_HIT, same box as the row above (log ut-g64-db8) | conc 64: 122.7 -> **126.7** (+3.3%); 128 of 640 decode calls in the 8-row bucket; ITL p50 452 -> 444 ms; `4,8,12,16` (G64-DB4, loads): 125.8, no better; greedy check 31 / 32 equal, decode-path NLL +0.00016 | spot $4.87 -> **$4.71** / M out |
| 2026-10-05 | engine-v0 e449b98 TP / `KILN_MOE_EP=1` / + feat/moe-kernel-tune 2f28ff8 `KILN_MOE_EP_SMALL_V=2` | trn1.32xlarge spot (kiln-ut-32) | conc 16, 4 decode rows per group: final G16 (TP, --max-num-seqs 16, KV 0.65) vs EP v1 and EP v2 at --max-num-seqs 32 / KV 1.5 (q/ut-g16ep G16E, q/moek-trn1 G16E-V2; logs ut-g16, ut-g16e, ut-g16e-v2) | conc 16: TP 87.5, EP v1 88.4 (+1.0%: prefill call 0.893 -> 0.592 s, decode call 0.065 -> 0.101 s), **EP v2 98.0 (+12.0%**, decode call 0.084 s, ITL p50 149 -> 140 ms, TTFT p50 8.65 -> 6.10 s); TP at the EP arms' exact shapes (q/ut-trn1 G16T32, log ut-g16t32) 86.9, so at identical shapes EP v1 +1.7%, **EP v2 +12.8%**; greedy check: v2 bit-identical to v1, EP vs TP decode-path NLL -0.00023 | spot $6.83 / $6.76 / **$6.09** / M out |
| 2026-10-05 | engine-v0 e449b98, mixed batches + KDA / DSA decode kernels + SP decode streams (all opt-in, together for the first time) | trn1.32xlarge spot (kiln-ut-32) | G64 + `KILN_MIXED_BATCH=1 KILN_KDA_DECODE_KERNEL=nki KILN_DSA_DECODE_KERNEL=nki KILN_DECODE_SP=1 --state-checkpoints 4`, farm q/ut-trn1 G64-MX-CK4-DKS, ASSERT_CACHE_HIT (log ut-g64-mxdks) | conc 64: 122.7 -> **141.4** (+15.2%; each alone on other boxes: mixed 131.8, decode kernels + SP decode 137.1); ITL p50 452 -> 396 ms, TTFT p50 6.9 -> 6.3 s; with `--decode-buckets 8,16` too (q/ut-trn1 G64-MX-CK4-DKS-DB8, log ut-g64-mxdks-db8): **142.9** (+16.5%) | spot $4.87 -> $4.22 -> **$4.18** / M out |
| 2026-10-05 | feat/attn-kernel-tune 54ec5df (engine-v0 e449b98 + the static DSA prefill attention kernel, then the fused DSA selection-and-attention prefill kernel `KILN_DSA_FUSED=1`) | trn1.32xlarge spot (kiln-ak-32) | **serving A/B**, the q/final-ebe237e G64 / F0 commands, farm q/attnk-448b9f6 (static) and q/attnk-54ec5df (fused), ASSERT_CACHE_HIT, back to back (logs ab-G64-*, ab-F0-*); wikitext DP 4: base -0.547, static -0.548, fused -0.548 | conc 64: 122.9 -> 127.6 (static) -> **132.0** (fused, +7.4%); conc 32: 112.5 -> 116.5 -> **120.0** (+6.7%); prefill call 0.592 -> 0.553 -> 0.519 s | spot $4.86 -> $4.52 / M out at conc 64, $5.31 -> $4.98 at conc 32 |
| 2026-10-05 | feat/attn-kernel-cand 12c05c1 (the fused kernel and the decode kernels + SP decode streams as trn1 defaults) | trn1.32xlarge spot (kiln-ak-32 / kiln-ak-33) | decode-kernel promotion: G16 / F0 with the decode kernels (q/final-ebe237e), and with the fused kernel too (q/attnk-54ec5df); the candidate tree's plain commands with no variable set (logs ab-G16-*, ab-F0-*, ab-G64-DKS-FU, ab-G64-CAND, ab-G16-CAND) | conc 16: 87.6 -> 88.1 (DK) -> 92.2 (candidate); conc 32: 112.4 -> 114.9 (DK) -> 119.0 (DKS) -> 127.2 (with the fused kernel); conc 64: 132.0 (fused) -> **148.5** (+ DKS), the candidate tree's plain command 148.6 | spot $4.02 / M out at conc 64 (candidate), $4.70 at conc 32 (fused + DKS), $6.48 at conc 16 |
| 2026-10-05 | **engine-v0 f70c14b** (fused DSA prefill kernel, decode kernels + SP decode, MoE decode v2 and EP from 4 decode rows per group, all trn1 defaults) | trn1.32xlarge spot (kiln-ak-32) | **final defaults**: the q/final-ebe237e G16 / F0 / G64 commands with no variable set, farm q/final-f70c14b, ASSERT_CACHE_HIT (logs ab-fin3-G16, -F0, -G64) | conc 16 / 32 / 64: **105.4 / 133.9 / 156.2** out tok/s (ebe237e: 87.3 / 112.3 / 122.9); TTFT p50 5.4 / 5.6 / 5.8 s; ITL p50 131 / 213 / 363 ms; prefill call 0.519-0.521 s, decode call 76 / 88 / 118 ms; wikitext -0.548, greedy LONG_TEXT 28 / 32 equal (decode-path -0.0004 +/- 0.0003 nats per token) | **spot $5.67 / $4.46 / $3.82 per M out** (p5en spot $25.7-27.1 / $9.49-9.99 / $5.48-5.77) |
| 2026-10-05 | engine-v0 25a45c9 (f70c14b + EPLB decode pairs on the primaries under v2) | trn1.32xlarge spot (kiln-ak-32) | **+ EPLB, opt-in**: one redundant slot per rank (`KILN_EP_REDUNDANT=1`, `KILN_EPLB_INIT` = the techniques agent's eplb-init-random.pt, `--eplb-rebalance`, two levels), KV 1.5, farm q/final-25a45c9 (logs ab-fin3-G64-EPLB, -F0-EPLB) | conc 32: 143.2 -> 143.3 after the online rebalance; conc 64: 166.4 -> **167.2** (+7.0% over the default); prefill call 0.46 s | spot $4.17 / **$3.57** per M out |
| 2026-10-05 | f70c14b / 25a45c9 | trn1.32xlarge spot (kiln-ak-32) | mixed batches (`KILN_MIXED_BATCH=1 --state-checkpoints 4 --decode-buckets 8,16`) on G64, alone and with EPLB, and on F0 (logs ab-fin3-G64-MX, -G64-MX-EPLB, -F0-MX-r2; the first F0 attempt, ab-fin3-F0-MX, failed at load in the collectives' bootstrap) | conc 64: 152.1 (-2.6% against the default), with EPLB 163.4 / 163.3 (EPLB alone 167.2); conc 32: 128.8 (-3.8%): mixed batches no longer pay | spot $3.93 / $3.66 / $4.64 per M out |
| 2026-10-05 | **engine-v0 70ddc1b** (trn2 defaults: SP prefill, KDA + DSA LNC split; EP, SP_GROUP, fused DSA, decode kernels off) | trn2.48xlarge Capacity Block (kiln-t2-cb, ap-south-2b, $37.20/h; spot $15.343/h used for $) | **whole box baseline**: 2 x tp=32, DP attention 4, prefill 4096 / 1024, P=6, decode 4/8/16/32, KV 2.75 fp8, max-num-seqs 128 per engine, farm q/t2f-base, ASSERT_CACHE_HIT (log s3 logs/kiln-t2-cb/20261005T155613Z-t2-base-U) | conc 16 / 32 / 64 / 128 / 256: **118.5 / 158.9 / 177.3 / 192.0 / 198.4**; TTFT p50 9.5-11.1 s; ITL p50 99 / 161 / 316 / 602 / 1163 ms; prefill call 1.00-1.05 s, decode call 32-171 ms | spot $35.97 / 26.82 / 24.04 / **22.20** / 21.48 per M out (block price: $87.2 ... 52.1) |
| 2026-10-05 | 70ddc1b + feat/trn2-fast (single-engine A/Bs, cores 0-31 / 32-63 at once) | trn2.48xlarge (kiln-t2-cb) | one engine, conc 32 / 64 / 128 (= whole box 64 / 128 / 256): base 88.6 / 96.0 / 99.2; fused DSA prefill kernel (q/t2f-pC) 88.8 / 96.3 / 99.5 (neutral); KDA + DSA decode kernels + SP decode (q/t2f-pB) 89.6 / 97.5 / 103.3 (decode call 73 / 107 / 171 -> 71 / 95 / 133 ms); SP_GROUP + MoE prefill skip 20 (q/t2f-pD) conc 32 94.8 (+7.0%, prefill call 1.017 -> 0.922 s); all trn1 defaults with engine-v0's grid-2 EP hung (logs t2-e1-*, t2-e2-pC, t2-e3-pB, t2-e4-pD) | x 2 for the box; numerics gates pending |
| 2026-10-05 | 70ddc1b + feat/trn2-fast bf9a88c | trn2.48xlarge (kiln-t2-cb) | **decode-only step** (tools/time_decode.py, one engine, DP attention 4, null-page KV, the decode agent's shapes, logs t2-td-*): ms per step at 16 / 32 / 64 rows per group: engine-v0 trn2 default 143.6 / 219.7 / 1438.7; + decode kernels + SP decode 110.7 / 154.8 / 264.2; + moe_dedupe LNC split 93.7 / 121.7 / 204.0; + KDA / DSA decode row splits **87.7 / 110.7 / 180.7** | decode-only $2.92 / 1.84 / **1.50** per M out at trn2 spot (2 engines: ~2,830 out tok/s per box at 256 rows each) |
| 2026-10-05 | **feat/trn2-fast 118c6ea** (engine-v0 8229c3d merged; a922fe9's trn2 defaults: KDA / DSA decode kernels, SP decode, moe_dedupe + KDA / DSA decode LNC splits, SP_GROUP; plus `KILN_MOE_PREFILL_SKIP=20`, "best1") | trn2.48xlarge Capacity Block (kiln-t2-cb) | **whole box**, the baseline's command (2 x tp=32, DP attention 4, prefill 4096 / 1024, P=6, decode 4/8/16/32, KV 2.75 fp8, max-num-seqs 128 per engine), farm q/t2f-best1, ASSERT_CACHE_HIT (log s3 logs/kiln-t2-cb/20261005T222609Z-t2-U-best1); numerics: decode path check_mixed 28/32 equal, signed dlogprob +0.00020; wikitext -0.544 vs engine-v0's -0.554 | conc 16 / 32 / 64 / 128 / 256: **131.3 / 176.0 / 202.4 / 221.6 / 235.1** (+10.8 / +10.8 / +14.2 / +15.4 / +18.5% over 70ddc1b on the same box); TTFT p50 8.6-9.4 s; ITL p50 89 / 146 / 276 / 525 / 1004 ms; prefill call 0.908-0.937 s, decode call 29-97 ms; prefill 66-87% of device time | spot $32.46 / 24.22 / 21.06 / **19.23** / 18.13 per M out (block price $78.7 ... 43.95) |
| 2026-10-05 | feat/trn2-fast 118c6ea, best1 + 8192-row prefill calls | trn2.48xlarge Capacity Block (kiln-t2-cb) | one engine (cores 0-31), best1 with `--prefill-tokens 8192 --prefill-buckets 2048 KILN_PIECEWISE_PREFILL_MOE_GROUP=4`, farm q/t2f-best1-p4@pf8k, ASSERT_CACHE_HIT (log s3 logs/kiln-t2-cb/20261005T225412Z-t2-e7-best1pf8k) | conc 32 / 64 / 128 per engine: **110.7 / 122.9 / 131.6** (best1 at the same load per engine 101.2 / 110.8 / 117.6: +9.4 / +10.9 / +11.9%); TTFT p50 8.7-9.0 s; ITL p50 252 / 479 / 916 ms; prefill call 1.637-1.660 s per 8192 rows = 9.9k prefill tok/s per box (best1 8.8k; trn1 one-piece 8.7k, with EPLB 10.2k) | x 2 for the box: ~$16.2 per M out at conc 256 at trn2 spot |
| 2026-10-06 | feat/trn2-fast-ds2 ef289f8 (feat/trn2-fast 8ba166d + feat/decode-scale 83036ba; scratch, for the measurement) | trn2.48xlarge Capacity Block (kiln-t2-cb) | **decode box with real KV** (PD decode side): tools/time_decode.py --real-kv, one engine, DP attention 4, 8K contexts, KV sized per bucket set (256 seqs / 5.2 GB, 384 seqs / 7.8 GB), farm queues q/t2f-TD*@r / @rb (logs s3 logs/kiln-t2-cb/20261006T*-t2-tdr-*) | ms per step at 16 / 32 / 64 / 96 rows per group: D2K2 96.82 / 111.55 / 179.47 / 245.74; + ST 81.85 / 107.52 / 172.64 / 245.96; + ST + v9 (`KILN_MOE_DEDUPE_MAX_TOKENS=256`) **78.52 / 104.95 / 147.23 / 217.50** = 815 / 1,220 / 1,739 / 1,766 out tok/s per engine; 96 rows per group is the 8K ceiling (tensors 22.4 GB per rank; 25.7 at 128) | ST + v9: $2.61 / 1.75 / 1.23 / **1.21** per M out at half-box spot; whole box 3,531 out tok/s at 96 rows; trn1's decode box $0.89 (decode agent) |
| 2026-10-06 | feat/trn2-fast 1eeed36 (the fused DSA prefill kernel with its LNC split as trn2 defaults) | trn2.48xlarge Capacity Block (kiln-t2-cb) | **whole box**, the best1 command and environment with `KILN_DSA_FUSED=1 KILN_DSA_PREFILL_KERNEL=nki KILN_LNC_SPLIT=...,dsa_fused`, farm q/t2f-best1F (log s3 logs/kiln-t2-cb/20261006T*-t2-U-best1F); gates: wikitext -0.5498 vs -0.5441, check_mixed 28/32 equal | conc 16 / 32 / 64 / 128 / 256: **133.9 / 180.9 / 208.9 / 229.5 / 243.9** (best1 131.3 / 176.0 / 202.4 / 221.6 / 235.1: +2.0 to +3.7%); prefill call 0.870-0.899 s | spot $31.83 / 23.56 / 20.40 / **18.57** / 17.47 per M out |

## What the first sweep says (2026-10-04)

Concurrency 8 on trn1.32xlarge processed 16 x 8192 prompt tokens in 285 s, about **460 prompt
tokens/s**, against ~76K on the GPU at concurrency 128: prefill throughput is the bottleneck, and
the 447 ms ITL is decode waiting behind prefill steps. On the spot-vs-spot basis (trn1.32xlarge
$2.15/h, p5en ~$29/h) Kiln needs at least 23 / 62 / 108 / 175 output tok/s at concurrency 16 /
32 / 64 / 128 to match the GPU's $25.9 / 9.6 / 5.5 / 3.4 per 1M output tokens. Levers, in order:
the KDA prefill kernel (34 of 45 layers; feat/kda-kernel), the MoE prefill kernel on real GLM
scales (feat/moe-prefill2), bigger prefill chunks (`--model-type=transformer` makes them fit), DP
attention at higher concurrency, and trn2.

## How to resume (state 2026-10-04 19:30 UTC, engine-v0 b4f400f)

- Code: engine-v0 (pushed) carries the final TP stack of "Where it stands": elementwise mHC, the prefill MoE
  kernel (real scales, tile skip), the KDA NKI kernel (default), exact NKI DSA top-k with the pool-key cache
  (separate under FP8 KV, c33a391), SP prefill streams with SP-local routing and reduce-scatter, reserve
  admission, the compile farm with load decisions, the router that stops its workers (b4f400f). Branches
  under measurement: feat/moe-ep, feat/mixed (mixed batches), the prompt-cache branch, feat/trn2-max,
  feat/glm-mtp, feat/decode-step.
- Best configuration, trn1 (the G1 levels): the three configs of q/final-c33a391; each run's exact command
  is its `.log.cmd`, and the farm's `configs/<name>.config.json` (s3://<your-bucket>/compile-farm/q/final-c33a391/)
  holds the same line. Shape args (prefill tokens and bucket, decode bucket, KV size and dtype, max-num-seqs)
  all change the graph keys, so copy them verbatim. The conc-64 run:
  ```sh
  env KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 \
    KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/final-c33a391/ \
    python bench/serve_sweep.py --model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 \
    --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup \
    --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64 --concurrency 64 --decode-buckets 16 \
    --kv-cache-gb 1.5 --kv-cache-dtype fp8 --requests 128 --max-seconds 3000
  ```
  conc 32: `--max-num-seqs 32 --concurrency 32 --decode-buckets 8 --kv-cache-gb 1.5` (bf16 KV); conc 16:
  `--max-num-seqs 16 --concurrency 16 --decode-buckets 4 --kv-cache-gb 0.65` (bf16 KV). Size KV for every
  sequence (max-num-seqs x 264 pages per group + ~10%): the `kv:` line of serve_sweep says whether it fits,
  and a KV-bound run reads like a kernel regression.
- Decode kernels (feat/decode-step, opt-in): add `KILN_KDA_DECODE_KERNEL=nki KILN_DSA_DECODE_KERNEL=nki KILN_DECODE_SP=1` to
  the conc-64 command (EP on by default); the farm holds the graphs as q/decode-2968ed3 `configs/G64-EPT-DKS2.config.json`
  (exact command; keys set-equal on the merged tree a64d7f8). At conc 32 the kernels without SP decode: `F0-EPT-DK2`.
- Utilization (feat/utilization): any serving command plus `KILN_TIMELINE=<file>` (host timeline, then `python
  tools/util_report.py timeline <file>`) and `KILN_RT_INSPECT=<dir>` (rank 0's runtime trace; raise
  `NEURON_RT_INSPECT_SYS_TRACE_MAX_EVENTS_PER_NC` for a whole level) costs nothing measurable. Device profiles of the real
  graphs on real inputs: the same command with `--requests 64 KILN_CAPTURE_INPUTS=<nvme dir> KILN_CAPTURE_AT=prefill:20,decode:400`
  (~430 GB for 32 ranks; the counts include the warmup's calls), then `python tools/util_report.py replay <dir> --call
  prefill:20 --out <prof> --keep-ntff [--profile-all]`, `bins <prof>`, `report <prof> --kind prefill --shape-dir <shape dir>
  --call-s <the serving fit>`. docs/neuron-notes.md "Accelerator utilization of the serving graphs" has the traps.
- Graphs: never compile on a device box. On the device run with `KILN_COMPILE_FARM=<queue uri>` (waits for
  farm graphs) or `NEURON_LIBTORCH_ASSERT_CACHE_HIT=1` (fails on any miss). Accept a change only with the
  real-weight 4-sentence ppl (~-2.073) AND the wikitext-2 slice (3071 tokens, within 0.01 of ~-0.551): the
  4-sentence mean has a knife-edge on "Water boils".
- MTP speculative decoding (feat/glm-mtp): measured and off for the sweep; to repeat a measurement pull a config's
  graphs with `python tools/cache_sync.py pull-keys s3://<your-bucket>/compile-cache/trn1-sdk2.32/lnl/
  s3://<your-bucket>/compile-farm/q/<queue>/configs/<name>.keys.json` and run its `configs/<name>.config.json`
  command with `NEURON_LIBTORCH_ASSERT_CACHE_HIT=1` (queues q/mtp-82cc981 TP, q/mtp-ep-4d38ee0 and q/mtp-ep16-4d38ee0 EP,
  q/mtp-pc-a8ede27 G1b). MTP needs `--kv-cache-gb 1.2` at conc 64 (HBM) and `--state-checkpoints` to keep the KDA state
  pool at the baseline's rows (1 + 16 x (1 + k) + checkpoints per group); acceptance / greedy equality with
  tools/check_mtp.py, step costs with tools/time_decode.py, served step phases with `KILN_PROFILE_STEP=1`.
- Sequence-parallel prefill streams (`KILN_PREFILL_SP`) are on by default on trn1. On trn2 the 4-sentence
  ppl move (-1.807 vs -2.100) was that mean's knife-edge, not SP: the wikitext slice accepts SP on trn2
  (docs/neuron-notes.md), and feat/trn2-max makes it the trn2 default together with the LNC split (held: the split
  fails the trn2 wikitext check, see GOAL.md). `tools/probe_mhc_rounding.py` stays the check that the device keeps
  the streams' bf16 rounding at a row count.
- Weights: GLM-5.3-Flash on kiln-mimo-trn1's root volume, kiln-g1-trn1's and kiln-g2-trn1's instance-store RAID0
  (wiped if the instance stops); otherwise `hf download zai-org/GLM-5.3-Flash --max-workers 32`.
- Where the time goes and what is next (conc 64, final TP stack): 256 prefill steps of ~1.0 s (4096 tokens)
  and ~900 decode steps of ~160-200 ms for 128 requests, so prefill is ~60% and decode ~40% of device time
  (profiled with KILN_PROFILE_PIECES, which inflates; shares approximate). Prefill: expert parallelism
  (+8.4%), the unexplained ~8 of ~23 ms per MoE layer in a 12-layer piece (device profile in progress).
  Decode: MTP speculative decoding, and the step's own cost (estimated at several times its HBM-read floor; feat/decode-step measures it). For G1b the
  output token is the cost (prompt-cache split: output ~$1.70-2.06 per 1M at cost against a $0.50 list).
