# vllm-neuron 0.24 parity: what it has, what Kiln has, what to port

Owner request (2026-10-07): bring every strength of vllm-neuron into Kiln, port its implementations where they are
the better ones, and beat its efficiency.

**Sources.** vllm-neuron at tag `release-0.24.0.1.1.0`, commit f8abae6
(https://github.com/vllm-project/vllm-neuron, Apache-2.0); paths below are relative to that tree. Kiln is at
engine-v0 b9c22b8; `FEATURES.md` holds Kiln's own status lines with their measurements. The one same-chip
measurement is bench/results/2026-10-07-trn2.3xlarge-qwen3-ttft.md (trn2.3xlarge, SDK 2.32, Qwen3-8B / 32B bf16,
TP=4).

**Scope.** The prefill attention kernel (vllm-neuron's segmented / flash prefill,
`vllm_neuron/functional/attention/attention_segmented_cte.py`, `attention_cte.py`) belongs to the dense-prefill
work (feat/dense-prefill) and is listed here only for completeness. GLM-5.3-Flash's KDA / DSA / MoE kernels and
`kiln/engine/pp.py` have their own owners.

**Licence.** vllm-neuron, vLLM and nki-library are Apache-2.0, like public Kiln. A ported file keeps its copyright
header with a "Modified by Kiln" line, gets a NOTICE entry (repo@tag:path), and its commit body names the origin.
Ports stay opt-in until gated.

## What vllm-neuron runs on, and what that rules out

- **Trn2 and Trn3 only.** Its NKI kernels come from nki-library (nkilib) and need NeuronCore-v3. The 2.29 NxDI note
  "NKI 0.3.0 kernels are not supported on Trn1/Inf2" is the reason (docs/research/neuron-stack.md). Kiln runs trn1,
  trn1n, inf2 and trn2.
- **Five architectures** (`vllm_neuron/model/registry.py:21-25`): LlamaForCausalLM, GptOssForCausalLM,
  Eagle3LlamaForCausalLM, Qwen3ForCausalLM, Qwen3VLForConditionalGeneration. Qwen3-Embedding-8B is a pooling
  model (`docs/model-recipes/qwen3-embedding-8b.md`). None of Kiln's target models (GLM-5.3-Flash, MiMo-V2,
  DeepSeek-V3.x, Kimi, Qwen3.5 / 3.8) is among them.

## The measured head-to-head (one trn2 chip, TP=4, 2026-10-07)

| | vllm-neuron 0.24 | Kiln b9c22b8 | who wins |
|---|---|---|---|
| Qwen3-8B TTFT at 8k / 32k | 0.606 / 3.496 s (33% MFU) | 1.653 / 17.569 s (12% / 6.6%) | vllm-neuron, 2.7x / 5.0x |
| Qwen3-32B TTFT at 2k / 32k | 0.476 / 17.126 s | 0.705 / 61.172 s | vllm-neuron, 1.5x / 3.6x |
| Qwen3-8B / 32B ITL | 19.4-20.0 / 49.5-50.2 ms | 12.1-14.2 / 35.3-39.2 ms | **Kiln, 1.3-1.6x** |
| Cold engine start, 8B | 491.4 s (2 graphs at chunk 8192); 135.0 s at chunk 2048 | 519.2 s (12 graphs, 497 s of serial compiles) | vllm-neuron |
| Warm start (compile cache hit), 8B | 106.9 s | not measured | open |
| Cold engine start, 32B | 214.3 s (chunk 2048) | 795.8 s (8 graphs) | vllm-neuron, 3.7x |

Two prefill gaps. Attention: Kiln's generic decoder materialises fp32 scores over the whole padded page bucket,
and that is the dense-prefill work. The linear layers: at 32B 2k, where attention is a small share of the FLOPs,
Kiln is at 28.1% MFU against 41.6%. vllm-neuron runs qkv, o_proj and the MLP through nkilib kernels
(`functional/attention/qkv.py:9`, `functional/attention/o_proj.py:5`, `functional/mlp.py:7`), with a
sequence-parallel all-gather / reduce-scatter around the MLP (`model/qwen3/model.py:614-632`).

## Capability table

Status: **K+** Kiln has it and is better or broader; **=** both have it; **gap** vllm-neuron has it and Kiln does
not; **n/a** irrelevant to Kiln's targets. Value is for Kiln's workloads: GLM-5.3-Flash PD serving at 8K and 1M,
plus dense Qwen3 / Llama. Effort: S (under a day), M (days), L (a week or more). "Port" says whether to port
vllm-neuron's implementation rather than re-derive it, and from where.

### Compilation and start-up

| capability | vllm-neuron (source) | Kiln | status | value | effort | port |
|---|---|---|---|---|---|---|
| Capture every bucket's graph first, then compile them all in parallel, then warm up | `vllm/worker/neuron_worker.py:1207-1240` (`_extract_graphs`, tp_barrier, `parallel_compile`); `vllm/worker/neuron_model_runner.py:5094-5120` calls `libtorch_neuronx_lite.compile.parallel_compile.parallel_compile` | capture on a CPU host (`kiln/capture.py`) and a parallel compiler (`tools/compile_farm.py compile --workers N`) exist, but an engine's own start compiles serially through LNL's shared compile lock: 12 graphs, 497 s of compiles, one at a time, on the 8B run | **gap** (in-engine) | high: cold start ~519 s -> estimated 200-250 s for 8B. For GLM every config change. | S | idea plus LNL's API; Kiln's capture is already the better extractor (meta device, every TP rank, no NeuronCore) |
| Per-key compile locks: SPMD ranks compile a graph once, MPMD graphs in parallel | `docs/design/compilation/compilation_cache.md` "Parallel Coordination" | LNL's cache lock; rank 0 compiles, the rest wait | = | - | - | no |
| Remote compile cache | `NEURON_LIBTORCH_REMOTE_CACHE` on NFS / FSx (same doc) | S3 cache `compile_cache_uri` + the compile farm queue (`kiln/compile_cache.py`, `tools/compile_farm.py`) | **K+** (S3, multi-host farm, NKI kernel binaries shipped too) | - | - | no |
| Pre-compiled artifact directory | `NEURON_COMPILED_ARTIFACTS` (`docs/guides/features-guide.md:44-56`) | `compile_cache_uri` pull before start | = | - | - | no |
| CPU-only compilation | `VLLM_NEURON_CPU_COMPILE` (`docs/design/compilation/cpu_compilation.md`) | `kiln/capture.py`, compile farm | **K+** (a fan-out queue over many hosts) | - | - | no |
| Warm start time | 106.9 s for 8B (measured) | not measured | open | high for PD scale-out | S to measure | - |
| FX passes: in-place to out-of-place, output aliasing rewrite, device rewriting | `docs/design/compilation/*_pass.md` | LNL defaults + Kiln's canonicalisation (`model_runner.canonical_neuron_backend`) | not compared | unknown | M | read first |

### Bucketing and shapes

| capability | vllm-neuron | Kiln | status | value | effort | port |
|---|---|---|---|---|---|---|
| Prefill token buckets, decode batch buckets | `num_batched_tokens_buckets`, `num_seqs_buckets` | `--prefill-buckets`, `--decode-buckets` | = | - | - | no |
| Decode context-length buckets | opt-in `decode_context_length_buckets` (`docs/design/vllm/decode-context-length-bucketing.md`: the default NEFF reads `ceil(max_model_len / block_size)` blocks every step, a "~32x DMA overhead amplifier") | on by default: the page axis per call (`--page-buckets`) | **K+** | - | - | no |
| Segmented prefill (one prefill shape for any length) | `max_num_batched_tokens` < `max_model_len` turns on `kv_segment_size_buckets` | page-bucketed prefill graphs, one per (prefill bucket, page bucket): 8 of the 12 graphs on the 8B run | gap | high, and it also cuts graph count (start time) | M | dense-prefill agent's |
| Mixed prefill + decode in one step | never mixed: "prefill and decode always run in separate batches" (`features-guide.md:148-152`) | decodes never pause for prefill; chunked prefill, mixed batches opt-in | **K+** | - | - | no |

### KV cache

| capability | vllm-neuron | Kiln | status | value | effort | port |
|---|---|---|---|---|---|---|
| Prefix caching | vLLM's hash APC, on by default; block count via `num_gpu_blocks_override` (`features-guide.md:164-223`) | radix cache, page-granular, linear-attention state checkpoints, session-aware eviction, host tier (`FEATURES.md` "KV cache") | **K+** | - | - | no |
| FP8 KV cache with calibrated per-layer scales from the checkpoint | `kv_cache_dtype=fp8`, `q_scale` / `k_scale` / `v_scale` from llm-compressor checkpoints; trn2 clamp 240, trn3 448 (`features-guide.md:371-413`) | fp8 KV with scale 1.0, clamp 240 / 448 | gap (calibration) | medium: accuracy of fp8 KV | S-M | yes: the scale plumbing (`vllm/platform.py:488-524` validation) |
| KV budget auto-sizing | `gpu_memory_utilization` x HBM minus used, capped at 0.30 of the budget (`features-guide.md:970-1009`, `docs/design/vllm/determine_available_memory_design.md`) | explicit `--kv-cache-gb`; `tools/hbm_estimate.py` advisory | gap (usability) | low-medium | S | design only: Kiln's KV size is a graph shape, so the number must be settled before capture |

### Decoding and sampling

| capability | vllm-neuron | Kiln | status | value | effort | port |
|---|---|---|---|---|---|---|
| On-device sampling: greedy, temperature, top-k (capped 256), top-p | `functional/sampling.py`, distributed `argmax.py` / `topk.py` over the TP-sharded vocab | in-graph sampling + min-p, logprobs and top-20 in one read-back | = (Kiln adds min-p, logprobs) | - | - | compare the vocab-sharded top-k against Kiln's on a dense model; port only if measured faster |
| Async scheduling (device-to-device token feed, batch queue depth 2) | `features-guide.md:443-513` | `--overlap`, on-device token board; spec decode under overlap (`KILN_SPEC_ASYNC`) | = / **K+** (async with speculation; vllm-neuron turns async off under speculation, `features-guide.md:522-525`) | - | - | no |
| EAGLE3 speculative decoding | `model/llama3/eagle3_model.py`, `vllm/spec_decode/eagle.py`, on-device `vllm/sample/rejection_sampler.py`; Llama 3.1 8B TPOT 10.52 -> 6.41 ms (docs/research/trn2-trn3.md, from the tutorial) | n-gram, suffix and MTP done; EAGLE3 missing | **gap** | medium-high for dense Llama / Qwen3 (heads exist); none for GLM-5.3-Flash, which has MTP | M-L | yes: the Eagle3 draft model and proposer, adapted to Kiln's verify graph |
| DFlash | named supported (`features-guide.md:526`) | planned | gap | low: no public head for Kiln's models | L | later |
| Structured outputs (JSON schema, regex, choice, grammar) | vLLM's on-device bitmask, ~0.4 ms/token (`features-guide.md:612-668`) | xgrammar on the host, packed bitmask unpacked in-graph, plus jump-forward | = / **K+** | - | - | no |
| Tool-call parsers | vLLM's, e.g. `Llama3JsonToolParser` for tool_choice auto (`features-guide.md:670-708`); gpt-oss harmony | hermes, qwen3 xml, glm, kimi_k2 + 6 reasoning parsers | gap: llama3_json and gpt-oss (harmony) | medium for Llama / gpt-oss users | S | yes, from vLLM v0.24.0 `vllm/tool_parsers/llama_tool_parser.py` and `gptoss_tool_parser.py` |
| Repetition / frequency / presence penalties | upstream vLLM; vllm-neuron's on-device sampler lists only temperature / top-k / top-p (`features-guide.md:428-441`) | next | neither on device | low | S | no |

### Models and quantization

| capability | vllm-neuron | Kiln | status | value | effort | port |
|---|---|---|---|---|---|---|
| Llama 3.x, Qwen3 dense | `model/llama3/model.py`, `model/qwen3/model.py` | generic decoder, transformers parity | = | - | - | kernels only, below |
| gpt-oss (bf16 on trn2, MXFP4 on trn3) | `model/gpt_oss/model_bf16.py`, `model_mxfp4.py` | done on trn1 tp=32 with MXFP4 experts decoded in-graph | **K+** (MXFP4 on trn1/trn2 too) | - | - | no |
| Qwen3-VL (image + video), vision encoder DP | `model/qwen3_vl/*`, `docs/design/multimodal/*` | none | **gap** | low for the G1 workload; opens multimodal users | L | yes, when wanted: model + mrope + block-packing attention |
| EAGLE3 Llama drafts | `Eagle3LlamaForCausalLM` | none | gap | see EAGLE3 | M | yes |
| Pooling / embeddings (Qwen3-Embedding-8B), `/v1/embeddings` | upstream vLLM pooling on the Neuron runner (`docs/design/vllm/pooling-models.md`) | planned | **gap** | medium (RAG users) | S-M | yes: the last-token pooling path; the API is vLLM's |
| Prompt embeddings | `enable_prompt_embeds` (`features-guide.md:867-910`) | none | gap | low | S | yes |
| FP8 static per-tensor ModelOpt checkpoints | `model/llama3/model_static_fp8.py` | per-row block FP8, per-tensor FP8 (Hy3), FP8 at load | ~= (ModelOpt loader not checked) | low | S | check the format first |
| MXFP8 (trn3) | `model/llama3/model_mx_fp8.py`, `model/qwen3_vl/model_mxfp8.py` | n/a (trn3 ignored, owner 2026-10-06) | n/a | - | - | no |

### Parallelism and serving

| capability | vllm-neuron | Kiln | status | value | effort | port |
|---|---|---|---|---|---|---|
| TP, DP, EP | `docs/design/parallelism/*` | TP, DP router, EP + EPLB, DP attention, CP for DSA, layer pipeline | **K+** | - | - | no |
| MoE all-to-all dispatch / combine | `parallel/all2all.py`, `functional/moe/hierarchical_all2all_*` | measured slower than the padded all-reduce on trn1 (34.7 vs 5.2-6.8 ms) | n/a on trn1; open on trn2 | medium on trn2 GLM | M | MoE owners' |
| Decode context parallel (DCP) for dense attention | `docs/design/parallelism/dcp.md` | CP for DSA layers only | gap for dense models | low today | M | later |
| Disaggregated prefill / decode over NIXL | `vllm/kv_connector/neuron_nixl_connector.py` | PD with TCP or NIXL device-to-device (`kiln/engine/disagg.py`, `nixl_kv.py`) | = / **K+** (trn1, CP-ordered handoff) | - | - | no |
| Encoder disaggregation (EPD) | `vllm/ec_connector/`, `vllm/disaggregated_encoder/` | none | gap | only with Qwen3-VL | L | with Qwen3-VL |
| OpenAI API surface | all of vLLM's server: chat, completions, responses, embeddings, tokenize / detokenize, pooling, rerank | chat, completions, streaming, `/generate`, `/v1/score`, `/v1/decisions`, watermark detect | gap: responses, embeddings, tokenize / detokenize, rerank | medium | S each | port the request / response shapes from vLLM 0.24 |
| Metrics | `vllm_neuron:` startup time, compile time per bucket, model load time / bytes, padding histograms, NEFF execution count (`features-guide.md:912-962`) | `/metrics` with vLLM names under `kiln:` | gap: start-up and padding metrics | low-medium (operability) | S | yes, the metric set |

### Kernels (trn2 only in vllm-neuron)

| kernel | vllm-neuron call site | Kiln equivalent | status | value | effort | port |
|---|---|---|---|---|---|---|
| Prefill qkv projection | `functional/attention/qkv.py:9` (nkilib `core.qkv`) | XLA matmul | gap: part of the linear-prefill gap (32B 2k 28.1% vs 41.6% MFU) | high for dense prefill on trn2 | M | yes (nkilib via vllm-neuron's wrapper) |
| Prefill MLP | `functional/mlp.py:7` (nkilib `core.mlp`), SP gather / scatter around it | XLA matmuls | gap, as above | high | M | yes |
| Prefill o_proj | `functional/attention/o_proj.py:5` (nkilib `output_projection_cte`) | XLA matmul | gap, as above | medium | M | yes |
| Prefill attention (flash / segmented) | `attention_cte.py:8`, `attention_segmented_cte.py:15-18` | dense fp32 scores over the page bucket | gap | highest at long context | M | dense-prefill agent's |
| Decode attention block (fused rmsnorm + qkv + rope + attention + o_proj) | `functional/attention/attention_decode.py:2-5` (nkilib `experimental.transformer.attention_block_tkg`) | XLA decode graph + Kiln's decode kernels for MLA / DSA / KDA | Kiln is already 1.3-1.6x faster on the measured models | uncertain: may push dense ITL further | M | measure nkilib's block inside Kiln first |
| MoE prefill / decode | `functional/moe/moe_cte.py`, `moe_tkg_wrapper.py`, `moe_block_tkg_wrapper.py`, `router.py` (nkilib) | Kiln's own NKI MoE kernels (prefill grouped GEMM, dedupe decode, EP) | not compared on trn2 | for the MoE owners | - | no (owned elsewhere) |
| RMSNorm + quantize | `functional/rmsnorm_quant.py:20` | XLA | small | low | S | maybe, with the fp8 activation path |
| Sampling (cascaded max, cumsum, rotational top-k) | `functional/argmax.py:17`, `cumsum.py:9`, `topk.py:18` | XLA in-graph sampler; trn1 radix top-k for DSA | not compared | low (sampling is a small share of a step) | S | measure first |

### Accuracy and debugging tools

| capability | vllm-neuron | Kiln | status | value | effort | port |
|---|---|---|---|---|---|---|
| Logit validation against goldens, tensor capture / compare / replacement, KV cache analysis, lm-eval task plugins | `vllm_neuron/accuracy/*`, `docs/design/accuracy/*` | `tools/check_ppl.py`, `check_device.py`, `ref_stream.py`, `check_long.py` needle / NLL, `KILN_CAPTURE_INPUTS` + `util_report.py replay` | = (different tools) | medium: tensor replacement bisects a wrong layer faster | M | maybe the tensor-replacement bisection later |
| Input snapshot capture | `vllm_neuron/snapshot/*` | `KILN_CAPTURE_INPUTS` | = | - | - | no |

## Where Kiln is already ahead

- **Decode**: 1.3-1.6x lower ITL on both measured dense models, same chip.
- **Scheduling**: decodes never pause for prefill, and prefill can be mixed in (vllm-neuron never mixes).
  Speculative decoding runs under overlap; vllm-neuron disables async with speculation.
- **Hardware**: trn1, trn1n and inf2 support. Every Kiln trn1 result is a configuration vllm-neuron cannot run.
- **Models**: GLM-5.3-Flash (KDA + DSA + mHC), MiMo-V2, DeepSeek-V3.x / DSA, Kimi, Qwen3.5 / 3.8, Hy3, K2-Horizon,
  Inkling, with MLA latent KV, the DSA long path to 1M, EP + EPLB, CP and the layer pipeline.
- **Caching**: radix prefix cache with linear-attention state checkpoints, a host KV tier, session-aware eviction.
- **Speculation**: MTP, n-gram, suffix decoding, jump-forward.
- **Context-length bucketing** is the default, not an opt-in.

## Ported and measured so far (feat/vn-parity)

| item | what landed | measured |
|---|---|---|
| 1. Capture -> parallel compile -> warmup | kiln/precompile.py, `KILN_PRECOMPILE_WORKERS` (opt-in), a fingerprint manifest for warm starts | trn2.3xlarge, Qwen3-8B TP=4 (bench/results/2026-10-07-precompile-start.md, s3 logs/kiln-vnp/start-q8/): cold **278.9 s** against vllm-neuron 491.4 s and Kiln serial 503.8 s; warm **49.4 s** against vllm-neuron 106.9 s. trn1.2xlarge Qwen3-1.7B TP=2: cold 270.5 -> 137.2 s, warm 33.3 s |
| (found by 1) rank 0's tokenizer order | engine.py loads the tokenizer after build_shard: the ONE default change on the branch (78b55bc) | Qwen3-1.7B TP=2: 18 -> 15 cache entries (rank 0 no longer compiles its own sampling post graphs). GLM-5.3-Flash: a meta-device capture of G64 (ranks 0 and 8, s3 logs/kiln-vnp/capcmp/) gives the branch the same 15 keys per rank as bd3416a, after 7c91df0 restored `_embed`'s graph input order. On kiln-pcf-32 (trn1.32xlarge, s3 logs/kiln-vnp/glm-tokorder2/): G64 colocated on 7c91df0 loads and serves under ASSERT_CACHE_HIT from an empty cache root (0 asserts, 316 entries fetched); bd3416a after it compiled and fetched nothing (the same 34 entries); the CPA-R96 decode server comes up healthy under ASSERT_CACHE_HIT; check_mixed's 32 greedy requests are equal 32/32, 2048/2048 tokens, teacher-forced logprob difference max 0.0000 over decode and prefill rows |
| 3. EAGLE-3 | kiln/models/eagle3.py + decoder / runner / loader (opt-in, `--spec-method eagle3 --spec-draft-model`); the draft keeps its own RoPE (80dfc09); overlapped speculation at k > 1 (3f92035); one-row board fix (0b854db) | **trn2.3xlarge, vllm-neuron's tutorial pair (Llama-3.1-8B-Instruct + RedHatAI EAGLE-3), TP 4, chat prompts, batch 1, k 3** (bench/results/2026-10-07-eagle3-trn2-llama31.md): acceptance length 2.49 against vllm-neuron's 2.48 (within 0.26 on every prompt); 153.2 against 156.1 tok/s (0.981x) with whole-model graphs and overlapped speculation; plain decode 100.2 against 99.2 tok/s. Greedy differences against plain decode are bf16 near ties (margin 0 or one bf16 step), with zero run-to-run jitter in both engines. CPU: tests/test_eagle3.py against a torch rendering of vLLM's math |
| 4. API / parsers | llama3_json / llama4_json / llama3 tool parser; `/tokenize`, `/detokenize` | CPU tests |
| 5. Metrics | kiln:startup_time_seconds, compilation_time_seconds{bucket}, model_load_time_seconds, model_load_size_bytes, neff_execution_count{bucket}, precompile_seconds | CPU test |
| 2. Dense prefill MLP kernel | kernels/nkilib_dense.py + DecoderLayer.pack_dense_mlp, `KILN_DENSE_MLP_KERNEL=nkilib` (opt-in, NKI gen >= 3) | **Loses; XLA stays the default.** Qwen3-8B TP 4 on trn2, in-graph TTFT at chunk 4096: 2k 0.1795 -> 0.2031 s, 8k 1.606 -> 1.721 s, 16k 4.311 -> 4.537 s, 32k 17.10 -> 17.61 s (s3 logs/kiln-vnp/winB/ base8, mlp8). One core, 4096 x 3072 per rank (`tools/probe_nkilib_dense.py --reduce`, winC/probe-reduce.log): XLA 1.270 / 2.347 / 4.521 ms (73-82% MFU) against nkilib 7.219 / 4.205 / 8.374 ms at 2048 / 4096 / 8192 rows. Kiln's XLA MLP is already at 73-82% of the core's dense peak, so a qkv / o_proj port is not worth trying either |

Where Kiln now leads vllm-neuron on one trn2 chip (TP 4): cold start 1.76x, warm start 2.2x, plain decode 1.01x at
max_model_len 2048 (1.4-1.6x at 32768, where vllm-neuron's decode graph spans the whole length), and prefill TTFT at
8k-32k (the dense-prefill agent's segmented attention). It trails slightly on EAGLE-3 (0.981x). The overlapped
speculative step costs 1.63 plain steps against vllm-neuron's 1.58. The four small board graphs around the verify and
draft graphs are the obvious place to recover it.

Not adopted, by measurement: nkilib's MLP kernel (above). Dropped: the nkilib decode attention block A/B (item 6).
Kiln's decode already beats vllm-neuron's on the same chip: 1.4-1.6x at max_model_len 32768 (Qwen3-8B,
bench/results/2026-10-07-trn2.3xlarge-qwen3-ttft.md) and 1.01x at 2048 (Llama-3.1-8B, above). The A/B could only run
on trn2 (nkilib needs NeuronCore-v3), so it would need a trn2 purchase that this margin does not justify. Still open from
the ranked list: gpt-oss harmony (item 4; `/v1/embeddings` is on feat/vn-parity2), Qwen3-VL + EPD (7), FP8 KV
calibrated scales (5).

## Ranked port list (value / effort, Kiln's workloads)

1. **Capture-then-parallel-compile at engine start** (and measure warm start). Cold start for every model and every
   config change; small effort because the capture and compile pieces already exist.
2. **Dense prefill linear kernels on trn2**: nkilib qkv / MLP / o_proj through vllm-neuron's wrappers, plus its SP
   all-gather / reduce-scatter around the MLP. This targets the 28% vs 42% linear-part MFU gap. It is coordinated
   with the dense-prefill agent, which owns attention.
3. **EAGLE3 for Llama / Qwen3**: port Eagle3LlamaForCausalLM and the proposer onto Kiln's verify graph and
   rejection sampler. Measure acceptance and ITL against vllm-neuron's EAGLE3 on the same chip.
4. **API and parser gaps**: llama3_json and gpt-oss harmony tool parsers, `/v1/embeddings` + Qwen3-Embedding
   pooling, tokenize / detokenize, prompt embeddings. Each is small.
5. **Start-up and padding metrics**, then FP8 KV calibrated scales.
6. **The fused decode attention block (nkilib attention_block_tkg) inside Kiln**: only if a one-layer A/B beats
   Kiln's decode graph on a dense model.
7. **Qwen3-VL + EPD**: large; only on an owner request for multimodal.
