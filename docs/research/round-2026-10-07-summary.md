# Research round 2026-10-06 19:00 UTC - 2026-10-08 10:30 UTC: summary

One page per question: what was asked, what was measured, what landed, what failed, what is open. Every number
here comes from a measurement whose log path is in docs/neuron-notes.md or docs/price-performance.md under the
named section; estimates are labelled as such. Model: zai-org/GLM-5.3-Flash (glm5_next: 34 KDA + 11 DSA layers,
MoE 288 experts top-8, fp8 128x128 block-scaled weights, mHC). Workloads: G1 = 8192 in / 256 out (the p5en
benchmark shape), W1M = 1,044,480 in.

## 1. Headline numbers (measured)

| question | before the round | after | where |
|---|---|---|---|
| 1M TTFT, one trn1.32xlarge, real text | 317.8 s (R8) | **233.6 s** (R8 + LOCAL_K 120 + KDA defaults), needle 6/6, 1M NLL inside the band | neuron-notes "Lever 1", "The prefill call's compute side" |
| 1M TTFT, PD end to end, 4 prefill stages + 1 decode, NIXL | not built | **68.62 s** (handoff 0.74 s, token 2 155 ms), bit-identical to one engine | neuron-notes "Over NIXL the handoff leaves the TTFT" |
| 1M TTFT, 8 prefill stages | - | 38.6 s pipeline; 55.85 s end to end on the host path; ~39.4 s over NIXL is COMPOSED, not measured | same |
| 1M second turn on the same document (prefix cache) | - | 1.34 s | neuron-notes "The layer pipeline across boxes" |
| 1M on trn2.48xlarge (whole box, 2-stage, LK120) | wrong answers past 262k | 172.0 s, needle 3/3, after two root-caused fixes | neuron-notes "Long prompts on trn2" |
| trn1 decode box (decode-only, 8K, real KV) | $0.470 / 1M out | $0.428 (MERGE_BOUND + DECODE_COMPACT) | neuron-notes "The decode box, next round" |
| trn2 G1, one engine per half box, in-process, real text, conc 128 | 241.6 tok/s (random ids, 4096-row calls) | **318.1 tok/s** (P4 conv + 8192-row calls + PREFILL_WHOLE) | neuron-notes trn2 G1 sections |
| trn2 G1, one engine over HTTP, real text, fixed router: max conc with TTFT p90 <= 5 s | - | conc 8: 91.2 tok/s, p90 4.48 s | same |
| trn2 G1, WHOLE BOX over HTTP, real text, two engines + router (both router fixes): max conc with TTFT p90 <= 5 s | ~400 tok/s at conc 64 with p90 18.9 s (random ids, capped router) | **conc 12-14, 160-190 out tok/s** (two runs: conc 14 at 190.1, p90 4.55 s; conc 12 at 161.5, p90 4.51 s; p90 crosses 5 s at conc 14-16), $22.42 / 1M out | price-performance "trn2 final round", section 9 below |
| trn2 G1, whole box, peak (same setup, DSA causal default) | - | 632.4 out tok/s at conc 256 (667.0 steady), $6.74 / 1M, TTFT p90 65.5 s; 97.6% of two in-process engines | same |
| trn2 G1, in-box PD (labelled: DP1 prefill half + DP4 decode half) | - | p90 <= 5 s only at conc 4 (50.3 tok/s); ITL 70-88 ms at every load; saturates ~294 tok/s | same |
| p5en.48xlarge vLLM reference (8 x H200) | - | 2359 tok/s at conc 128, TTFT p50 0.42 s / p90 1.56 s (its highest measured level, under the 5 s bar); 1458 at conc 64, p90 2.72 s | price-performance "The reference" |
| Qwen3-8B / 32B prefill on one trn2 chip vs vllm-neuron 0.24 | Kiln 2.7-5.0x slower at 8k-32k | Kiln faster at 8k-32k (8B 32k: 3.32 vs 3.50 s; 32B 32k: 13.36 vs 17.13 s), 35-41% MFU | bench/results/2026-10-07-trn2.3xlarge-qwen3-ttft.md, neuron-notes "Dense prefill attention" |
| Engine cold / warm start vs vllm-neuron (Qwen3-8B, trn2) | 503.8 / 48.5 s | 278.9 s cold (parallel precompile, 1.76x vllm-neuron), 49.4 s warm (2.2x) | docs/research/vllm-neuron-parity.md |
| EAGLE-3 (Llama-3.1-8B, trn2) | - | 0.981x vllm-neuron's tok/s at equal acceptance (2.49 vs 2.48) | bench/results/2026-10-07-eagle3-trn2-llama31.md |

Price-performance at spot, G1 (out tok/s per box, $/1M out): trn1 colocated ~165 / ~$3.6; trn2 ~575 in-process /
~$7.2-7.5; p5en vLLM 2359 at conc 128 / $3.39. trn1 remains the cost vehicle; trn2 is faster per box only.
At the TTFT p90 <= 5 s bar the measured whole-box gap is at least 12.4-14.6x: p5en holds the bar at least to conc 128
(2359 tok/s, p90 1.56 s, its highest measured level, so this is a lower bound) against trn2's 160-190 at conc 12-14
(corrected 2026-10-09: an earlier version compared with p5en's conc 64, ~7.7x). The 1000 tok/s target at p90 <= 5 s
was NOT reached.

**Pricing correction (2026-10-08).** The trn2 $ figures in this file ($22.42, $6.74 and the "~$7.2-7.5") are at a trn2
spot quote of $15.343/h that was never paid: trn2 spot could not be obtained, and the box was a Capacity Block at
$35.76/h. On the Capacity Block basis on both sides (https://aws.amazon.com/ec2/capacityblocks/pricing/: trn2.48xlarge
$35.7608/h, p5en.48xlarge $63.158/h, trn1.32xlarge $9.532/h): trn2 at TTFT p90 <= 5 s $52.25-61.51 per 1M out against
p5en's $7.44 at conc 128, its highest measured level and still under the bar (at least 7.0-8.3x; 4.3-5.1x against
p5en's conc 64), trn2's peak $15.71 against the same $7.44 (2.1x), trn1 at conc 64 $16.10. docs/price-performance.md "Pricing basis (corrected 2026-10-08)" has the table.

## 2. What landed in engine-v0 (dc317d7)

In merge order: NIXL device-to-device KV (2f946a5); trn2 EP race fix + EP default (c857a31); P8K-EPLB graphs
(1bd168f); decode-next MERGE_BOUND / DECODE_COMPACT (bd3416a); vllm-neuron TTFT bench (b9c22b8); EPLB trace +
real-text EPLB finding (1808d75); long-context PR #2 + #3 (308b639, 6ca3d0c: R8, R4M, LOCAL_K, PLP_VP,
LONG_PIPE); segmented dense prefill from nkilib (1942731); prefill page-bucket ladder G64PL (3d9be89); PD router
overlap (f3ad6b0); per-stage KV handoff (529302e, a493382); decode inject select (a5e4376); trn2 4096-page fix +
device_split fix (bcfdac4); KDA fused norm + delta-rule units 6 as the glm5_next default (8266a14); layer
pipeline + stage-only loading (09338fd); pipeline serving front + uvicorn SIGTERM fix (a6c5dd9); host-path
recv_into receiver (181505e); vllm-neuron parity PR #4 (505b7c9: precompile, EAGLE-3, tokenizer-order fix,
llama3_json parser, start-up metrics); NIXL docs (5a09cb3); CP-row decode select fix (af4a909, silent wrong
NIXL tokens before it); router httpx pool fix (90da152); trn2 NKI short conv P4 (1a9d6d7). Final block (10-08): router no-match
requests least-loaded first (9ba2ffd; tree size alone sent 63 of 64 requests to one engine once both trees
were full); feat/trn2-final (df64441: KILN_PREFILL_WHOLE opt-in one-graph prefill, the 1M DSA select in
MAX_TILES blocks, pd_sweep --prompt-ids real text); the causal dsa_fused_c kernel as the default for the gated
trn2 G1 8192-row whole-prefill calls only (dc317d7; bit for bit in the graph, +0.9..2.8% out tok/s).

## 3. Negative results (measured; do not retry without a new idea)

- EPLB on real text at 8K: no gain (-1% net, +1.02 GiB per core); its earlier wins were random-id artifacts.
- Online EP rebalance on pipeline stages: the commit takes 36.6 s, longer than a request; loses 6-12%.
- FP8 MoE on trn2 (weights re-blocked for PSUM accumulation): -24..-34% per MoE layer and +12.8% G1 tok/s at
  conc 128, but FAILS wikitext (-0.5605 vs -0.5507). Power-of-two factorisation of the block scales is not exact
  (0.0% of blocks). nkilib's per-channel FP8 MoE is not faster in any deployable form and has 10x the error.
  Exact per-K-tile FP8 costs the same vector work as dequantisation.
- Dense FP8 on trn2: the compiler maps fp8 dots to the fp8 PE (2.4x on one matmul), but the dense FP8 weights are
  only ~1-5% of the G1 call, at 15x the error. Not built.
- PE head sum for the 1M indexer (LONG_PE): -7% TTFT but a 131k band 2x outside the spread. Opt-in only.
- trn2 mixed prefill+decode batches: -13..-22%. KILN_SP_GATHER_ROUTE on trn2: -13..-18%.
- DP-attention-1 prefill: halves lone/p50 TTFT (1.14 s vs ~2.8 s) but worsens burst p90 (prefill call
  0.76 s per request vs 0.52 s per request-equivalent).
- Two micro-batches to overlap collectives: XLA already overlaps a matmul with its consuming RS; no gain.
- MoE lane padding (2.6x executed tensor-engine FLOPs at 4096 rows): costs FLOPs, not time; the per-expert fp8
  dequant floor bounds the call. 50% MFU is not reachable at 4096 rows with the current weight format.
- mHC boundary kernel in the trn2 G1 graph: 0.521 s per prefill call with and without it (the ~30 ms of glue it
  replaces runs under the collectives). Opt-in; it pays only where the boundary is on the critical path.
- DSA split selection (PWS) in the trn2 G1 graph: -32 ms per call, +4.0%, wikitext unbiased, but check_mixed
  28/32 against the bit-exact causal arm's 32/32. Every form short of the whole graph is exact; the cause is
  narrowed to own()'s row select (+ gather) and unproven. Opt-in.
- In-box PD on trn2 (DP1 prefill on half the box): its prefill half serves ~1.27 requests/s, so p90 <= 5 s
  holds only at conc 4. Colocated serving is better for the p90 bar on one box.

## 4. Where the time goes now (measured)

- trn2 8192-row G1 prefill call, real text: 531.6 ms, MFU ~17% against trn1's per-core peak (~8% against
  trn2's). The FFN-output reduce-scatter waits ~72 ms on the busiest EP rank (transfer 16 ms); attention-ending
  segments 181.5 ms, MoE 149 ms, KDA 103 ms, DSA 79 ms. Device ~94% busy at conc 128: continuous batching already
  fills the device; per-call cost is what is left.
- trn2 G1 TTFT under a burst is set by prefill THROUGHPUT: at DP attention 4 requests finish in batches of 4
  every ~2.4 s, so p90 <= 5 s caps at ~8 in flight per engine.
- trn1 1M call (R8 + LK120, call 200): DSA long 48%, collectives 19%, FFN 12%, XLA glue 9%, KDA 9%.

## 5. Code written but NOT tested (owner's code-first phase, 10-07 afternoon)

feat/prefill-fewer-graphs (whole-graph prefill was then gated on trn2: +1.3-1.7%, PASS; bigger chunks),
feat/cc-fused (EP v2 plan, SP_GATHER_ROUTE, bf16 CP combine), feat/pc-rewrite (mHC fused kernel, EP v2 core,
router hookup), feat/dsa-causal (dsa_split, causal fused at grid 2; trn1 in-graph -22 ms per 8K call, gates
open), feat/dsa-long-v2 (certified indexer pre-filter, row-pair slots), feat/seg-parallel (chunk ring for
cross-box segment-parallel prefill, tests written, never run), feat/vn-mfu (MLA dense regime through the latent
flash kernel, NKI router), feat/pd-early-inject (128k token 2 181 -> 140 ms measured; 1M re-run pending),
feat/vn-harmony, feat/vn-parity2 (embeddings), feat/vn-qwen3vl (Qwen3-VL + EPD: image TTFT 1.2 -> 0.28 s on trn1),
feat/vn-fp8kv, feat/trn2-moe-fp8, feat/nkilib-trn2. None merges into engine-v0 before its gates.

## 6. Measurement traps found this round

- serve_sweep's device_split is rank-deficient for a lone request: use TTFT / calls or the timeline.
- Random token ids inflate EP skew (busiest/mean 3.73x vs ~2.9x on text): measure G1 on real text.
- NLL: per-position standard errors are invalid (positions move together); judge on 16 x 131k bands over two
  1M windows (R8-vs-C8 sd 0.0063, net ~0).
- A "bit-identical" check that reuses the first compiled graph for every static-argument value is vacuous.
- Engine-busy from summed instruction durations double-counts; take the union of intervals.
- httpx's default pool caps a router at 100 streams in flight; the wait shows up as TTFT.
- Kernels gated on `nki_grid() == 1` silently do nothing at trn2 LNC=2.
- NEURON_RT_MAP_HBM=1 is required for a device tensor's data_ptr (NIXL registration).
- zsh: `"$H:refs/..."` applies the `:r` modifier; write `"${H}:refs/..."`.
- A prefix-tree router that picks the smaller tree for unmatched requests degenerates once trees hit their cap:
  check the per-engine request split at every level, not just the totals.
- TTFT p90 over 24-32 requests is the 3rd-4th slowest one and moves by tenths of a second between runs: quote
  the max-conc-under-the-bar as a range from two runs.
- Bit-identical on one core (and in nki.simulate) does not imply bit-identical in the compiled whole graph
  (PWS); and nki.simulate passing does not imply the device agrees (mHC's tensor-engine RMS form was 2-10% off).
- Back-to-back 32-rank starts on one trn2 box can fail CCOM bootstrap: wait until no rank process remains, then
  ~75 s, before starting the next engine.

## 7. Open next steps (ranked by expected value)

1. Faster prefill per call is the only lever that moves burst p90 (the whole-box run is measured, section 9).
   Landed: DSA causal. Open: PWS's in-graph defect (sel_own from the one-call selection first, then a selection
   kernel reading full qT / w at a runtime row offset; feat/dsa-split-fix notes); EP balance by structure (TP-k)
   if real-text skew is still worth ~72 ms; the EP prefill kernel's non-matmul ~1.3 of 1.87 ms (nkilib notes).
2. Make PW (one-graph prefill) + 8192-row calls the trn2 G1 default: gated (wikitext, check_mixed), still opt-in
   flags. The attention-ending RS segments (181.5 ms per call), MoE (149 ms) and KDA (103 ms) are what is left.
2a. A future p90 headline: more requests per level or three repeats (24-32 samples move p90 by tenths of a second).
2b. mHC only after the collectives stop hiding the boundary (fewer rows per rank, or fused collectives).
3. A single engine with DP-attention-1 prefill and DP-attention-4 decode (KDA-state / latent regroup inside the
   engine) for lone-request TTFT ~0.6 s; design note only.
4. Segment-parallel (chunk ring) device test on <= 3 trn1 boxes; arithmetic projects 80-82 s at 1M on 3 boxes.
5. The 1M DSA long path (48% of the call): the certified pre-filter and row-pair slots on feat/dsa-long-v2.

## 8. Final trn2 round, last measurements before the pause (10-08 00:00-04:00 UTC)

- FP8 MoE in-graph on trn2 G1, real text, same-time control: FP8 prefill alone +3.3 / +4.1 / +4.9% out tok/s at
  conc 32 / 64 / 128; with the FP8 decode kernel -1.2 / +5.3 / +12.8%. Not a default: wikitext FAIL (above).
- mHC boundary kernel on one trn2 logical core at the G1 rank shape (256 rows, LNC=2 row split): 0.444 vs 0.593 ms
  (mix + pre + route), 0.304 vs 0.362 ms (pre + route); S' bit-identical. In-graph: no gain (section 9).
- Causal dsa_fused / dsa_split on one trn2 logical core (2048 rows x 8448 keys): 5.24 ms -> 3.31 ms average
  (-37%) causal, bit-identical; the earlier "no graph change" was a trn1-only `nki_grid() == 1` gate, fixed.
  In-graph: causal exact and the default, split opt-in (section 9).
- DP-attention-1 prefill (one engine, in-process, real text): lone TTFT 1.14 s, but p90 5.74 s at conc 8
  (DP4: 4.48 s); the call is 0.76 s per request.

## 9. Final block on the paid trn2 Capacity Block (10-08 04:00-10:30 UTC, no new trn)

All rows: kiln-t2-cb2 (trn2.48xlarge), GLM-5.3-Flash G1, wikitext-103 real-text prompts, client-side HTTP, closed
loop with the start burst counted. Logs under s3 logs/kiln-t2-cb2/.

| conc | colocated E1B-p6PW, balanced router (tok/s, TTFT p50 / p90) | + DSA causal default (PWC) | in-box PD (labelled) |
|---|---|---|---|
| 4 | - | - | 50.3, 1.92 / 3.46 s |
| 8 | 126.2, 2.85 / 3.10 s | 126.9, 2.77 / 2.98 s | 87.8, 2.74 / 6.64 s |
| 12 | 166.5, 2.35 / 4.68 s | 161.5, 2.77 / 4.51 s | - |
| 14 | 190.1, 2.78 / 4.55 s | 188.0, 2.98 / 5.14 s | - |
| 16 | 210.8, 3.43 / 5.16 s | 212.6, 3.39 / 5.03 s | 151.5, 1.76 / 11.3 s |
| 32 | 327.7, 3.53 / 9.34 s | 330.9 | 231.7, 4.86 / 21.8 s |
| 64 | 430.3, 3.65 / 17.0 s | 435.7 | - |
| 128 | 531.0, 3.78 / 32.6 s | 535.6 | 293.9, 78 / 83 s |
| 256 | 620.7, 4.21 / 67.3 s | 632.4, 3.99 / 65.5 s | - |

Before the balance fix the same colocated setup gave 215.2 tok/s at conc 32 (63 of 64 requests on one engine);
after it the split is exactly even at every level and both engines are 98.7-99.4% busy.

Gates this block (trn2 G1, the gated whole-prefill graphs):

| arm | check_mixed vs PW0 | wikitext | speed | outcome |
|---|---|---|---|---|
| DSA causal (PWC) | 32/32, abs d 0 | -0.5518 = PPL0 | -22 ms per call, +2.8% at conc 128 | default (dc317d7), gated config only |
| DSA split + causal (PWS) | 28/32 (halves floor 32/32) | -0.5517, unbiased | -32 ms, +4.0% | opt-in; in-graph defect, cause narrowed |
| DSA split kernels, no row split (PWL) | 32/32, abs d 0 | - | - | diagnostic: the kernels are exact |
| mHC boundary kernel | 28/32 | -0.550 (control -0.5523, CPU -0.5480) | none (0.521 s both) | opt-in |

Labelled PEAK-ONLY row, 16384-row prefill calls (E1B-p6PW@b16k, 4096 rows per DP group per call): 428.2 / 533.0
/ 639.8 out tok/s at conc 64 / 128 / 256 ($6.66 / 1M at 256, the round's highest), TTFT p50 5.4-6.8 s, p90 not
claimed. +3.1% over the 8192-row row at conc 256, equal at 64-128 (in-process it was +4.0 / +5.7%).

Hosts: kiln-t2b-cf4 terminated 08:39 UTC, kiln-t2-cb2 (Capacity Block cr-046a70209d206fcba) terminated 09:02 UTC;
no Project=kiln instance, EIP or volume left in us-east-2 or ap-south-2 (describe-*, 10-08).

## 10. Where each workstream's full notes live (branch @ sha, docs/neuron-notes.md section)

| workstream | branch @ sha | section |
|---|---|---|
| trn2 final round (G1 real text, router cap + balance, RS skew split, DP1, whole box, in-box PD, gates) | feat/trn2-final @ ce7f8db (merged) | "trn2 final round" (+ price-performance "trn2 final round") |
| nkilib MoE on trn2, FP8 MoE | feat/nkilib-trn2 @ 6388d10 | "nkilib's MoE kernels on trn2, and FP8 MoE with GLM-5.3-Flash's 128 x 128 blocks" |
| DSA short path (causal default landed; split's in-graph defect) | causal: engine-v0 dc317d7; split investigation: feat/dsa-split-fix @ 7a9bd52 | "Phase 2" sections + "The final trn2 round" |
| mHC kernel, router hookup, EP v2 core | feat/pc-rewrite @ 7e260b4 (trn2 tree feat/pc-t2 @ 312c2df) | "The mHC boundary as one NKI kernel", "The EP prefill kernel v2's expert-slice core and the boundary router" |
| one-graph prefill, bigger chunks, test plan | feat/prefill-fewer-graphs @ 81251fe | "Test plan for feat/prefill-fewer-graphs" |
| collectives, EP v2 plan | feat/cc-fused @ bebaf5d | its notes sections |
| 1M DSA long v2 | feat/dsa-long-v2 @ a178fac | its notes sections |
| segment-parallel chunk ring | feat/seg-parallel @ fdec64f | "Segment-parallel prefill across boxes: the chunk ring" |
| MLA dense kernel, router kernel, MoE padding analysis | feat/vn-mfu @ 24bcd16 | its notes sections |
| early per-stage inject | feat/pd-early-inject @ 2bd146f | its notes sections |
| vllm-neuron feature ports | feat/vn-harmony @ 97d2955, feat/vn-parity2 @ e9bc2c2, feat/vn-qwen3vl @ 6284b96, feat/vn-fp8kv @ 3cbc26d | docs/research/vllm-neuron-parity.md |
| trn2 FP8 MoE kernel (code) | feat/trn2-moe-fp8 @ 18abe9d | "FP8 MoE on trn2" |

These branches carry untested or ungated code next to their notes, so they are NOT merged into engine-v0; this
table is the index. Logs: s3://<your-bucket>/logs/ under each host's name.
