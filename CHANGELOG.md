# Changelog

All notable changes to Kiln are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and Kiln follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) in its 0.y.z form: a minor release
adds a model family, changes a default or breaks a configuration; a patch release fixes.

## [0.3.0] - 2026-10-09

A minor release. Defaults changed for GLM-5.3-Flash on every target (two KDA prefill kernels), on trn2
(expert parallelism, an NKI short convolution, and a causal DSA kernel inside one opt-in configuration), and in
both routers. New and opt-in: a layer pipeline that runs one long prefill over several boxes, a device-to-device
KV handoff over EFA with NIXL, and ports from vllm-neuron 0.24 (parallel compilation at engine start, EAGLE-3,
segmented dense prefill attention).

**Every number below is labelled with the development commit or branch it was measured on**, and none of those
trees is byte-identical to the released `kiln` package. Conditions, commands, logs and the iteration history are in
`docs/price-performance.md`, `docs/neuron-notes.md`, `docs/research/vllm-neuron-parity.md`,
`docs/research/round-2026-10-07-summary.md` and `bench/results/`.

**Workloads.** "G1" is v0.2.0's: zai-org/GLM-5.3-Flash@eb9eb208 (real weights, FP8), 8,192 tokens in and 256 out
per request, a closed loop at the stated concurrency, Neuron SDK 2.32, every graph from the CPU compile farm.
**G1 is now measured on real text** (consecutive non-overlapping 8,192-token windows of wikitext-103, `--prompt-ids`
from `tools/text_prompt_ids.py`). Every G1 number of v0.1.0 and v0.2.0 used random token ids, which skew expert
routing (busiest against mean expert-parallel rank 3.73x, against about 2.9x on text). "1M" is one 1,044,480-token
prompt.

**Pricing bases, corrected in this release.**
- Between trn2 and the GPU reference the like-for-like basis is the EC2 Capacity Block for ML price on both sides,
  from https://aws.amazon.com/ec2/capacityblocks/pricing/ (read 2026-10-09; the page says its prices are next updated
  in January 2027): trn2.48xlarge $35.7608/h (US East (Ohio) and Asia Pacific (Hyderabad)), p5en.48xlarge $63.158/h,
  trn1.32xlarge $9.532/h, Linux OS fee $0. It is also what the trn2 measurements cost: one trn2.48xlarge Capacity
  Block, $1,443.55 for 40.37 h.
- trn2 spot capacity could not be obtained for this release's runs (placement score 1), so no trn2 figure here is
  priced at spot. **v0.2.0's trn2 dollar figures used a trn2 spot quote of $15.343/h that was never paid** (those runs
  were on Capacity Blocks too); read them as quotes, not as costs.
- trn1.32xlarge spot at $2.15/h was obtained, and most trn1 boxes ran on it (some ran on-demand when the account's
  spot request count was full, among them the box behind the 164.5 out tok/s row below). The p5en.48xlarge spot band of
  $28.77-30.29/h is a `describe-spot-price-history` quote (read 2026-10-03), not capacity anyone obtained; the
  reference itself ran on a SageMaker ml.p5en.48xlarge endpoint ($72.795/h hosting). Spot figures appear below only
  as a labelled second basis.
- $ per 1M output tokens = $/h / (out tok/s x 3600) x 1e6, on the measured rates.

### Measured against the GPU reference, and what this release does NOT win

The reference is unchanged: vLLM (`vllm/vllm-openai:glm53-flash`) on SageMaker ml.p5en.48xlarge (8 x H200, TP 4 / DP
2 / EP 8), 1,458 out tok/s at concurrency 64 with TTFT p90 2.72 s, and 2,359 at concurrency 128 with TTFT p90 1.56
s, its highest measured level. Its measured rates are priced here at the p5en.48xlarge Capacity Block rate. Kiln's
trn2 rows are one trn2.48xlarge, two tp 32 engines (DP attention 4, 8,192-row prefill calls,
`KILN_PIECEWISE_PREFILL_MOE_GROUP=6 KILN_PREFILL_WHOLE=1`) behind `kiln.server.router`, measured from the client
over HTTP like the reference, the start burst counted, 2 x conc requests per level.

| | out tok/s | TTFT p50 / p90 | Capacity Block $ / 1M out | against the reference | measured on |
|---|---:|---|---:|---|---|
| trn2 whole box, highest concurrency with TTFT p90 <= 5 s, run 1: conc 14 | 190.1 | 2.78 / 4.55 s | $52.25 | **at least 7.0x its cost** | engine-v0 df64441 |
| the same bar, run 2: conc 12 | 161.5 | 2.77 / 4.51 s | $61.51 | **at least 8.3x** | with the DSA causal default (below) |
| trn2 whole box, peak: conc 256 | **632.4** | 3.99 / 65.5 s | $15.71 | **2.1x** | with the DSA causal default |
| labelled peak-only row, 16,384-row prefill calls, conc 256 | 639.8 | 6.80 s / not claimed | $15.53 | 2.1x | engine-v0 df64441 |
| GPU reference, conc 128: its highest measured level, both its peak and still under the bar | 2,359 | 0.42 / 1.56 s | $7.44 | | SageMaker endpoint |
| GPU reference, conc 64, a same-load row only | 1,458 | 0.63 / 2.72 s | $12.03 | trn2's bar rows are 4.3x / 5.1x this | SageMaker endpoint |

The engines of the "DSA causal default" rows ran a merge of feat/dsa-causal into feat/trn2-final's engine code
(scratch/t2-dsc 8e34b32) behind engine-v0 df64441's router; this release's graphs for that configuration equal the
gated ones (30 / 30 keys). p90 over 24-32 requests per level is the 3rd or 4th slowest and moves by tenths of a
second between runs, so the bar is quoted as the pair: **conc 12-14 at 160-190 out tok/s**. The reference holds the
TTFT p90 <= 5 s bar at least up to conc 128 (p90 1.56 s), its highest measured level, so at the bar the whole-box gap
is **at least 12.4-14.6x in throughput** (2,359 against 190.1 / 161.5 out tok/s) **and 7.0-8.3x in Capacity Block cost
per token**: lower bounds, since the reference was not measured above conc 128. **The target of 1,000 out tok/s at
TTFT p90 <= 5 s on one box was not reached.**

At the same concurrency on Capacity Block prices (the balanced-router run, engine-v0 df64441), the trn2 box is 16%
below the reference only at conc 16 (210.8 out tok/s, $47.12 against $56.41 per 1M out, where the reference's own
TTFT p90 is 12.3 s and the box's 5.16 s), and 1.45x / 1.92x / 2.51x its cost at conc 32 / 64 / 128 ($30.31 / $23.09 /
$18.71 against $20.84 / $12.03 / $7.44).

- **What bounds that p90.** At DP attention 4 a prefill call carries one 2,048-row chunk per group, so a lone 8,192
  prompt is four sequential calls (TTFT never below about 2.1 s) and a burst finishes four requests at a time every
  about 2.4 s: p90 <= 5 s holds to about 8 requests per engine. Past that, p90 follows prefill throughput, 0.52 s
  per request-equivalent per engine. A DP-attention-1 prefill engine halves the lone TTFT (1.14 s) but costs 0.76 s
  per request, so its burst p90 is worse (5.74 s at conc 8 against 4.48 s), measured on feat/trn2-final.
- **trn1, priced against the GPU at the same concurrency and at the GPU's best.** One trn1.32xlarge at conc 64 on
  real text serves 164.5 out tok/s (64 requests started together; measured on dd428fd = engine-v0 a6c5dd9 with an
  opt-in flag left off). At the Capacity Block rate that is $16.10 per 1M output tokens: **1.34x the reference at
  the same conc 64** ($12.03) and **2.16x the reference at its best rate, conc 128** ($7.44); trn1 cannot hold conc
  128 at this workload. On spot it is $3.63: 34-37% below the p5en spot quote at conc 64 ($5.48-5.77), and 2-7%
  ABOVE it at conc 128 ($3.39-3.57); the GPU side of every spot comparison is a quote, not a purchase. (v0.2.0's
  trn1 defaults at conc 16 / 32, 110.5 / 144.8 out tok/s on random ids, are $23.96 / $18.29 at the Capacity Block
  rate against the reference's $56.41 / $20.84 at the same concurrency: 58% / 12% lower.) **Latency at that level is
  far from the GPU's:** TTFT p50 38.2 s and ITL p50 230.9 ms against the reference's 634 ms and 32.3 ms at conc 64
  (TTFT p90 2.72 s); v0.2.0's defaults measured TTFT p50 / p90 5.37 / 59.9 s at conc 64 with 128 random-id requests.
  At Capacity Block rates trn1 ($16.10 at conc 64) and trn2's peak ($15.71 at conc 256) cost about the same per
  token; trn1 is the cost vehicle only because its spot capacity exists.
- **1M TTFT is minutes, not seconds** (trees in Added, below). One trn1.32xlarge: 233.6 s. Four pipeline stages plus
  a decode box over NIXL: 68.62 s end to end. Eight stages: 38.6 s for the pipeline alone and 55.85 s end to end
  over the host path; about 39.4 s over NIXL is composed from measured parts, not measured.
- **FP8 MoE is not in this release.** In the trn2 G1 graph, at conc 32-128, the FP8 prefill pass alone gave +3.3 to
  +4.9% out tok/s and with the FP8 decode kernel -1.2 to +12.8%, and it failed the wikitext gate (-0.5605 against
  -0.5507); it stays on an unmerged branch (feat/nkilib-trn2). nkilib's per-channel FP8 MoE is not faster in any
  deployable form and has 10x the error.
- **Prefill MFU is far below 50%.** The trn2 8,192-row G1 call takes 531.6 ms on real text, 17.1% of 32 x 95 TFLOPS
  (about 8% of trn2's own BF16 peak; feat/trn2-final); the trn1 8K call is at 9.9% and the trn1 1M call at 4.6%
  (1c1f5f6's defaults). With the current FP8 expert format the per-expert dequantization bounds the MoE call, so 50%
  is not reachable at 4,096 rows.
- **Expert-parallel load balancing does not pay on real text.** v0.2.0's opt-in EPLB row (191.3 out tok/s) was
  measured on random ids. On real text, same box and tree, one-piece 8,192-row prefill, conc 64: 182.4 / 182.5 out
  tok/s without EPLB and 180.0 / 181.1 with it (-1%, +1.02 GiB per core), farm queues of engine-v0 2f946a5 and
  1bd168f. EPLB is not a default anywhere.
- **Latency.** On the trn2 whole box (engine-v0 df64441) ITL p50 is 52 ms at conc 8 and 368 ms at conc 256, against
  the reference's 20.1-43.4 ms, and TTFT p50 is 2.85 s even at conc 8.
- **Prefill and decode on separate halves of one trn2 box** (a labelled latency configuration: a DP-attention-1
  prefill engine and a context-parallel decode engine, host handoff; scratch/t2-best b70ed21 and feat/trn2-kda-conv
  78db25d): ITL 70-88 ms at every load, but TTFT p90 <= 5 s only at conc 4 (50.3 out tok/s), and 293.9 out tok/s at
  conc 128 (326.6 steady), because the prefill half serves about 1.27 requests/s. Colocated serving is better for
  the p90 bar on one box.
- **On-demand pricing** is not won anywhere: trn1.32xlarge on-demand is ten times its spot price.
- Open on trn2: the MoE prefill kernel's LNC=2 split still fails the wikitext check, and `sp_gather` still has no
  correct LNC=2 form. A compiler defect in one broadcast-table gather of the 4096-page long-context graphs is worked
  around (Fixed, below), not fixed.

### Changed

**GLM-5.3-Flash, every target.**
- The KDA layer's gated RMSNorm runs as one NKI kernel after the prefill delta rule, and the delta rule interleaves
  6 units instead of 2 (`KILN_KDA_FUSED_NORM=1 KILN_DELTA_RULE_UNITS=6` by default for glm5_next;
  `KILN_KDA_FUSED_NORM=0 KILN_DELTA_RULE_UNITS=2` restore v0.2.0's path; 1c1f5f6). Measured on feat/prefill-compute
  (engine-v0 bd3416a + feat/long-context-next 3d1a16b) with the flags set: the 8K G64 prefill call, replayed,
  498.53 -> 467.49 ms (-6.2%); a lone real-text 1M request on one trn1.32xlarge 316.19 -> 299.10 s (-5.4%). Gates:
  wikitext -0.5507 -> -0.5483, a deterministic shift toward the CPU bf16 reference's -0.5480; 1M NLL over 2,088,958
  positions -0.00049 nats per token signed, 16-band sd 0.0065, against R8 - CP8 by the same tool -0.00006 / 0.0065;
  needle 6 / 6 at 128k and 1M.
- **Compile GLM-5.3-Flash's graphs again for this release.** The default above gives its prefill pieces new keys,
  so a v0.2.0 compile cache misses them (the opt-in one-piece 8,192 prefill configuration had already moved 16 of its
  25 keys from v0.2.0's farm queue by engine-v0 2f946a5). The decode box's graphs are unchanged by it.

**trn2 (LNC=2).**
- Expert parallelism is the default for glm5_next (68504b9, c857a31), now that its hang is fixed (Fixed, below).
  G1 whole box on random ids, 133.9 / 180.9 / 208.9 / 229.5 / 243.9 -> 171.3 / 239.5 / 282.7 / 316.5 / 340.5 out
  tok/s at conc 16 / 32 / 64 / 128 / 256 (+28 to +40%), measured on feat/trn2-next 68504b9 against v0.2.0's trn2
  defaults (feat/trn2-fast c08c5ba); gates wikitext -0.5495 against -0.5498, check_mixed 28 / 32 equal.
- KDA prefill's short causal convolution is an NKI kernel split by channels (`kernels/short_conv.py`,
  `KILN_LA_CONV_KERNEL` unset = nki on trn2 at LNC=2 and xla elsewhere, so trn1's keys are unchanged; bd5786b). The XLA
  convolution's row split between the two physical cores had cost 6.6 ms per KDA layer. One engine on half the box,
  same tree: 142.0 / 159.1 / 171.5 -> 186.8 / 217.8 / 241.5 out tok/s at conc 32 / 64 / 128, the prefill call
  0.583 -> 0.368 s; the whole box 201.6 / 301.5 / 374.4 / 435.8 / 483.1 at conc 16-256 on feat/trn2-kda-conv
  3928014, random ids. Gates wikitext -0.5507 against -0.5495, check_mixed 29 / 32. Not with context-parallel DSA
  (`KILN_DSA_CP=1`), whose 4096-page graphs fault with it (3928014).
- The causal fused DSA prefill kernel (`kernels/dsa_fused_c.py`, which skips the key blocks past the call's last
  position) is the default for exactly one configuration: LNC=2, 2,048 rows over 8,448 keys, with the opt-in one-graph
  prefill `KILN_PREFILL_WHOLE=1` (the G1 8,192-row calls; 19148ea). It is bit for bit in that graph (check_mixed
  32 / 32, |dlogprob| 0; wikitext -0.5518 both) and measured +2.8% out tok/s at conc 128 in-process on real text
  (303.9 -> 312.4, one run pair), the prefill call 0.517 -> 0.495 s, on scratch/t2-dsc 8e34b32 against scratch/t2-best
  b70ed21. Everything else keeps `dsa_fused`. `KILN_DSA_FUSED_CAUSAL=0` forces it off, `=1` on anywhere. Before this
  release that flag changed no trn2 graph at all: the kernel required `nki_grid() == 1`.

**Routers.**
- `kiln.server.router` sends a request that matches no worker's prefix to the least-loaded worker, the smaller tree
  only breaking ties (9ba2ffd), and no longer caps streams in flight (90da152). Both were bugs; see Fixed.
- `kiln.server.pd_router` starts the prefill call together with the decode stream, and sends a `/v1/completions` string
  prompt it tokenized for threshold routing on as token ids when `--tokenizer` is the served model (on by default;
  `--no-forward-ids` turns it off) (60e1e08).
- An engine loads its tokenizer after building its shard (78b55bc): with the old order rank 0 traced its sampling
  graphs differently and compiled its own copies (Qwen3-1.7B TP 2: 18 cache entries against 15).

### Added

- **A layer pipeline on the device (opt-in).** One long prefill runs over several engines, each holding a contiguous
  range of layers (`--pp-stages`, `--pp-split`, `--pp-listen`, `--pp-next` of `bench/serve_sweep.py` and
  `bench/pd_serve.py`, EngineConfig `pp_*`; `KILN_PP_ASYNC=1 KILN_PP_OVERLAP=1`). A stage loads only its own layers'
  weights (stage 0 of a 4-stage 1M split 5.41 against 14.16 GiB per core, up in 63 against 176 s); following stages
  learn every request from stage 0's frames (`--pp-follow`, `tools/pp_follow.py`); the PD router takes a pipeline as
  one prefill unit (`--prefill-units`); and every stage hands its own layers' KV and state rows to one decode
  engine, which admits the request once all stages are in. Measured on trn1.32xlarge, GLM-5.3-Flash, every graph
  from the farm:
  - 1M, the pipeline alone with the final long-context configuration below: 236.92 s on one engine, **67.95 s** on 4
    stages and **38.30 s** on 8 (scratch/pp-final = feat/pp-serve + feat/prefill-compute). 4 stages, split on the
    one engine's 12-layer prefill runs, equal it bit for bit; 8 stages run differently cut graphs, which one engine
    cut the same way reproduces bit for bit.
  - End to end through the router, the stages and one decode engine, 64 greedy tokens: 4 stages + decode 85.35 s at
    1M over the host path (tokens and logprobs bit-identical to one engine), 72.36 s once the decode receiver writes
    straight into its mapped part file (Fixed), and **68.62 s over NIXL** (a 0.74 s handoff, token 2 at 155 ms,
    tokens and logprobs equal), measured on engine-v0 a6c5dd9 plus feat/pd-early-inject 2bd146f, which is **not** in
    this release. Holding the 5 boxes for the TTFT that is $0.205 of prefill per 1M request at trn1 spot.
  - A warm second turn on a 1M document (the prefix from the cache, 4,080 new tokens) takes 1.34 s to its first token
    on one box (feat/lc-scaleout).
  - An 8K lone request on real text, engine time only: 0.737 s on 4 stages and 0.802 s on 3, against 1.223 s on one
    box (feat/lc-scaleout).
- **Device-to-device KV handoff over EFA (opt-in, `KILN_PD_TRANSPORT=nixl` on both engines).** Decode ranks read the
  prefill engine's paged caches and state rows from its HBM with NIXL (LIBFABRIC), as vLLM's NixlConnector pulls. It
  needs `NEURON_RT_MAP_HBM=1`, which gives a Neuron tensor a real device address and moves no graph key (G64 167.8
  out tok/s with it and without it, check_mixed bit-identical, on engine-v0 fdea9af). Probe: 11.4 GB/s per core
  pair, 23 GB/s per DP-attention group, 91-101 GB/s per box. In a 3 prefill : 1 decode deployment on trn1.32xlarge,
  the only switch the transport (feat/d2d-kv 8a60442): steady 786.0 -> 805.4 out tok/s, ITL p50 305.3 -> 290.4 ms,
  handoff mean 0.965 -> 0.235 s. One of four NIXL engine starts failed once in collectives setup and loaded on an
  immediate restart; that is not attributed.
- **1M on one box, 317.8 -> 233.6 s.** Opt-in long-context levers (`docs/neuron-notes.md` "Lever 1"): one request
  over all 32 ranks in context-parallel row groups (`KILN_DSA_CP_DEGREE`, also over the minimal KV layout), each
  rank's local top K with a certificate that redoes what it cannot prove (`KILN_DSA_CP_LOCAL_K`, exact), the
  long-context selection as a three-stage pipelined kernel (`KILN_DSA_LONG_PIPE`), prompt logprobs on vocabulary shards
  (`KILN_PLP_VP`), and slot classes through one call (`KILN_DSA_CP_SLOTS_X`). One trn1.32xlarge, a lone real-text
  1,044,480-token request: R8 (row groups of CP 8) 317.8 s on feat/long-context-next d6738a1, and R8 +
  `KILN_DSA_CP_LOCAL_K=120 KILN_DSA_LONG_PIPE=1 KILN_DSA_CP_MERGE_BOUND=1` with the KDA default above **233.59 s** on
  1c1f5f6, needle 6 / 6 (v0.2.0's 1M engine: 1177.1 s). A 1M NLL is now measured: R8 against v0.2.0's
  context-parallel engine -0.00006 nats per token over 2,088,958 positions.
- **trn2 at 1M.** Two causes of wrong answers past 262,144 tokens are fixed (Fixed, below). The same final
  configuration passes the 1M needle 3 / 3 and takes 172.04 s with the whole trn2.48xlarge as a 2-stage pipeline
  (315.69 s on half the box), its first tokens bit-equal to one engine's, on feat/trn2-next b8ab49d. At Capacity
  Block rates that box costs 3.75x a trn1.32xlarge per hour, so trn2 is not the 1M prefill vehicle.
- **The trn1 decode box at $0.428 per 1M output tokens** (decode-only, 8K context, real KV, 96 rows per DP group,
  trn1 spot), from $0.470: opt-in `KILN_DSA_CP_MERGE_BOUND=1` (the merge's tie search over the bucket's bits, bit-exact)
  and `KILN_DSA_CP_DECODE_COMPACT=1` (`kernels/dsa_slots_c.py`, each row's live slots compacted in the attention
  kernel), 302.6 -> 275.1 ms per step, on feat/decode-next d4f0700.
- **From vllm-neuron 0.24**, measured on one trn2 chip (trn2.3xlarge, TP 4, SDK 2.32) against vllm-neuron 0.24 on the
  same chip and cores (`docs/research/vllm-neuron-parity.md`):
  - Capture every warmup graph, compile them in parallel, then start (`kiln/precompile.py`, opt-in
    `KILN_PRECOMPILE_WORKERS=N`, with a manifest for warm starts). Qwen3-8B: cold start **278.9 s** against
    vllm-neuron's 491.4 s and Kiln's serial 503.8 s; warm **49.4 s** against 106.9 s (feat/vn-parity c5fbf34 /
    78b55bc).
  - EAGLE-3 drafts for dense targets (opt-in `--spec-method eagle3 --spec-draft-model <draft>`), and speculative steps
    under overlap scheduling at k > 1. On vllm-neuron's tutorial pair (Llama-3.1-8B-Instruct with
    RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3, chat prompts, batch 1, k 3) the acceptance length is 2.49
    against vllm-neuron's 2.48, and Kiln reaches **0.981x its tok/s** (153.2 against 156.1); plain decode 100.2
    against 99.2 tok/s (feat/vn-parity 7c91df0, Kiln with whole-model graphs and overlap scheduling).
  - Segmented prefill attention for dense models: vllm-neuron's nkilib `attention_segmented_cte`, vendored with its
    license and a reader for Kiln's KV layout, plus NeuronCore-v2 fallbacks so it also runs on trn1 (opt-in
    `KILN_ATTN_PREFILL=segmented`). Lone-request TTFT on real text, Kiln against vllm-neuron's best: Qwen3-8B at 8k
    / 16k / 32k **0.563 / 1.298 / 3.321 s against 0.606 / 1.387 / 3.496 s** (34.7-35.6% MFU, chunk 8192); Qwen3-32B
    **2.143 / 5.065 / 13.364 s against 2.349 / 5.981 / 17.126 s** (chunk 4096). Still behind at 2k on 32B (0.545
    against 0.476 s). Kiln's ITL is 1.3-1.6x lower (12.0-14.2 against 19.4-20.0 ms on 8B, 34.7-38.6 against
    49.5-50.2 ms on 32B). Gate: mean prompt logprob -2.1882 against -2.1884. Measured on feat/dense-prefill
    (19657b3); on one trn1 core pair the Qwen3-8B 8k TTFT goes 7.149 -> 3.730 s (vllm-neuron does not run on trn1).
  - The llama3_json / llama4_json / llama3 tool parsers, vLLM's `/tokenize` and `/detokenize`, and start-up metrics
    (`kiln:startup_time_seconds`, `compilation_time_seconds`, `model_load_time_seconds`, `neff_execution_count`).
  - nkilib's dense MLP kernel, opt-in `KILN_DENSE_MLP_KERNEL=nkilib`. It loses (Qwen3-8B 8k TTFT 1.606 -> 1.721 s;
    Kiln's XLA MLP is at 73-82% of the core's peak), so XLA stays the default.
- **One graph per prefill call, and bigger calls (opt-in `KILN_PREFILL_WHOLE=1`).** On trn2 with 8,192-row calls:
  +1.3 to +1.7% over piecewise prefill (random ids), gated (wikitext -0.55228, check_mixed 29 / 32); one engine on
  half the box serves 318.1 out tok/s at conc 128 on real text (scratch/t2-best b70ed21). The 1M DSA selection runs
  in blocks of `MAX_TILES` tiles for larger chunks.
- The sparse-only page-bucket ladder `--page-buckets 132,264` for GLM-5.3-Flash at 8K, bit-exact against 264: lone
  TTFT -2.1%, +1.3% at conc 64, +0.51 GiB on the fullest core (engine-v0 1942731).
- Opt-in and held, each with its reason in `docs/neuron-notes.md`: the DSA selection split over the attention group
  (`KILN_DSA_SPLIT_SELECT=1`: +4.0% on trn2 G1, but check_mixed 28 / 32 in the graph), a tensor-engine head sum for the
  1M indexer (`KILN_DSA_LONG_PE=1`: -7% TTFT, fails the NLL gate), and `KILN_DSA_FUSED_CAUSAL=1` on trn1 (lone 8K TTFT
  -4.7% and conc 64 164.5 -> 168.7 out tok/s on dd428fd, gates not yet run).
- Measurement tools: `--prompt-ids` real-text prompts in `bench/serve_sweep.py` and `bench/pd_sweep.py`,
  `bench/ttft_compare.py` (Kiln against vllm-neuron on the same text and cores), `tools/check_eagle3.py`,
  `tools/nll_bands.py` (the 1M NLL gate), `tools/pd_timeline.py`, `tools/pd_client.py`, `tools/prof_segments.py`,
  `tools/pc_map.py`, `tools/prefill_split.py`, and opt-in traces (`KILN_EPLB_TRACE`, `KILN_DSA_CP_FLAGSTAT`,
  `KILN_PIECEWISE_PREFILL_CUTS`, `kiln:pd_handoff_seconds`).

### Fixed

- **`kiln.server.router` held at most 100 streams in flight** (90da152): httpx's default connection pool, shared by
  every worker, made the rest wait inside the router, and the client counted that wait as TTFT. Every whole-box
  HTTP run before it understated two engines (401.9 out tok/s at conc 128 on trn2, random ids). It now uses the PD
  router's limits; one engine behind it then serves 312.9 out tok/s at conc 128 against 318.1 in-process.
- **`kiln.server.router` sent nearly every request to one worker once both prefix trees were full** (9ba2ffd): a
  request with no prefix match went to the smaller tree, and the worker that received it was pruned below the other
  and won the next one too. On the trn2 whole box, real text: at conc 32 one engine served 63 of 64 requests (215.2
  out tok/s); with least-loaded first every level splits evenly and conc 32 serves 327.7, the peak 557.4 -> 620.7
  out tok/s at conc 256 (feat/trn2-final 8fa86b1 against engine-v0 df64441).
- `kiln.server.pd_router` created the prefill call only after the decode engine had accepted the stream, which it
  does between steps: up to one decode step (about 0.26 s of the trn1 latency arm's 1.60 s 8K TTFT) before every
  prefill started; on a CPU timeline with a busy decode engine the router's start to the prefill call went 36.9 ->
  0.7 ms (engine-v0 1808d75 against the fix). A refused decode request now also cancels its prefill call (60e1e08).
- A server's SIGTERM never reached the engine's close (uvicorn re-raised it under the default handler), so
  tensor-parallel workers died mid-broadcast and no timeline was written (1a456c6).
- **trn2: expert parallelism hung at LNC=2** on the first execution of an expert-parallel decode graph (v0.2.0's
  known limitation): the expert kernel's scatter read-modify-write skipped empty lanes by an out-of-range index, and
  each empty lane now adds into its own in-range dummy row (9fd01b4).
- **trn2: the 4096-page long-context bucket answered wrong past 262,144 tokens** (needle 0 / 3 at 128k, 300k and 1M,
  while trn1 passed). Two causes, both fixed for LNC=2 only, so trn1's keys are unchanged: a broadcast-table gather
  the compiler computes wrong in that graph is now a 1-D gather, and the long-context selection kernels' two
  programs no longer share one score scratch (8bb542c, c0391ae). Needle 3 / 3 at 128k, 300k and 1M after it
  (feat/trn2-next).
- With context-parallel row groups, a CP decode rank selected handed-off positions by its attention rank instead of
  its index in its CP group: a non-CP prefill into such a decode engine crashed 24 of 32 ranks on the host path and
  decoded wrong tokens over NIXL (af4a909). The row groups are new in this release.
- A CP decode rank joined the whole handed-off latent instead of its own ~1/cp of the rows: token 2 after an
  8-stage 1M prefill 1,784.6 -> 708.7 ms (fc6378a).
- The host-path receiver zero-filled every part frame under the GIL before receiving it, about 2.2 GB/s into one
  decode box; it now receives straight into the mapped part file. The 1M 4-stage handoff 18.41 -> 4.45 s (618401f).
- Overlapped speculation: one-row board graphs came back zero at k = 1 on the device (overlapped EAGLE-3 accepted 0
  of 128 drafts), and now run the row twice (0b854db). v0.2.0's opt-in asynchronous MTP drafting runs through the
  same graphs.
- A closing engine left an online EPLB rebalance's loading thread alive, which aborted the process at exit (7e365b9).
- A pipeline stage steps synchronously: an overlapped step could launch a decode for a request that had just ended
  (e8ae946).
- `bench/serve_sweep.py`'s device split was rank-deficient for a lone request and printed a negative decode call
  (5dc1b52).
- **A disaggregated decode engine's receive buffer shrank for good with every handoff frame that never completed**
  (present since v0.2.0). `Receiver._room` counts a frame's bytes as held before reading it, and only a completed
  handoff's `release()` gave bytes back, so a prefill peer that closed or reset inside a part, or a NIXL meta refused
  after its reservation (no hello frame from its engine), kept its bytes held; enough of them and every later
  handoff waits for room that never comes. The reader now gives such bytes back exactly once when it ends, and a
  completed frame's bytes pass to its handoff before anything can raise, so `release()` still frees them and nothing
  is freed twice. `tests/test_disagg.py::test_a_frame_cut_mid_way_gives_its_bytes_back` covers EOF and reset on disk,
  EOF in memory and the NIXL refusal, five broken frames in a row and then a handoff of the whole buffer, which
  goes through without waiting and releases back to zero (red before the fix: 1,000 bytes still held after the first
  broken frame); `tests/test_pd_nixl.py`, which had asserted that a refused NIXL meta's byte stays held, now asserts
  it is given back.
- Tests: `test_a_part_closed_mid_frame_leaves_no_file[close]` raced the receiver (it took a store directory that was
  still empty because the part file was not yet open for one that had been cleaned up) and failed 8 of 100 in-process
  repeats on Python 3.13 and 2 of 100 on 3.12; it now waits for the receiver to finish the connection (200 of 200 on
  both).

[0.3.0]: https://github.com/foxl-ai/kiln/releases/tag/v0.3.0

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
