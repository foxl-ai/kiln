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

## Where it stands (engine-v0 ebe237e, final, 2026-10-05 03:30 UTC)

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
