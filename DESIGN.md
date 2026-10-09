# Kiln design

Kiln is an LLM serving engine for AWS Trainium and Inferentia. It takes the ideas that
make SGLang, vLLM and TileLang fast and rebuilds them for a static-shape accelerator whose
graphs are compiled ahead of time. It replaces the managed Neuron inference layer
(NxD Inference, vllm-neuron) and keeps only what cannot be replaced.

Research behind this document, with a source URL beside every claim:
`docs/research/neuron-stack.md`, `docs/research/models.md`,
`docs/research/mini-sglang-neuron.md` (all read 2026-10-01).

## 1. Why now

AWS is retiring its own managed inference layer, and the cheapest Neuron capacity is the
part it has abandoned:

- NxD Inference is in maintenance mode since SDK 2.30, Trn2-and-newer only since 2.29, and
  was removed from the DLAMIs in 2.32 (`neuron-stack.md` section 4).
- vllm-neuron 0.24.0.1.1.0, its successor, runs on Trn2/Trn3 only. Its
  `utils/hardware_config.py` defines exactly `trn2`, `trn3pd`, `trn3pds` (read from the
  installed wheel on 2026-10-02). It supports five model families and no mixed
  prefill/decode batching.
- PyTorch/XLA inference on trn1/inf2 is in maintenance mode since 2.31.
- None of the top 13 open-weight models on the Artificial Analysis Intelligence Index
  (v4.3, 2026-10-01) has official Neuron support (`models.md` section 4).
- Spot prices in us-east-2 on 2026-10-02 UTC: trn1.32xlarge $2.15/h (512 GB HBM),
  inf2.48xlarge $1.30/h (384 GB), trn2.48xlarge $14.29/h (1.5 TB), against
  g7e.48xlarge at $15.48/h (768 GB) in the same region.

Mini-SGLang-Neuron (Yotta Labs, 2026-03) showed the appetite, but it is Mini-SGLang's
scheduler over an unmodified NxD Inference model: zero NKI kernels, every shape padded to
the maximum, overlap scheduling disabled, idle since 2026-04-09
(`mini-sglang-neuron.md`).

## 2. Measured ground truth

Measured on trn1.2xlarge, Neuron SDK 2.32.0 DLAMI (multi-framework, Ubuntu 24.04,
`ami-0222021b369f03219`), 2026-10-02, with `tools/smoke_device.py`:

| Path | Result |
|---|---|
| NKI standalone (numpy inputs) | works on trn1; 32.5 s first compile, 1.37 s per call (not a serving path) |
| `libtorch_neuronx_lite` (LNL) `neuron` device, eager | works on trn1 |
| `torch.compile(backend="neuron_libtorch")` | 36.6 s first compile; **94 us per steady-state call**, 157 us with a device-to-host read |
| NKI kernel inside a compiled graph (`wrap_nki(k)[lnc](...)`) | works on trn1, 94 us per call |
| JAX `neuron` platform | works on trn1 (2 NeuronCores visible) |
| torch-xla in the SDK 2.32 vLLM venv | no Neuron PJRT plugin; silently falls back to CPU |

The 94 us is the per-graph-call floor under every decode step on the v0 executor.

## 3. Layers

```
L5  serving      OpenAI-compatible HTTP API, streaming, cache-aware router (multi-replica)
L4  runtime      scheduler, radix prefix cache, paged KV pool, sampler, spec decode, grammar
L3  models       functional PyTorch model code, HF safetensors loading, TP/EP sharding, FP8
L2  kernels      NKI kernels (attention, MoE, norms, sampling) + a torch reference for each
L1  executor     static-shape graph compile cache + async execution on NeuronCores
L0  Neuron       driver, libnrt, libnccom, neuronx-cc, nki   <- kept, not replaced
```

L1 to L5 are Kiln's. The boundary is chosen by what AWS licenses and ships: the driver is
GPL; the runtime, collectives and compiler are binary-only (and not redistributable, so
Kiln ships source and runs on the DLAMI rather than shipping images with those binaries).

### L1 executor

The executor interface is `compile(fn, bucket) -> Executable` and
`Executable.run(inputs) -> outputs` (asynchronous). Two implementations:

- **v0, LNL**: model code runs on the `neuron` PyTorch device and each bucket is one
  `torch.compile(backend="neuron_libtorch", fullgraph=True)` graph. KV cache tensors are
  device buffers mutated in place inside the graph with `index_put_` (the idiom
  vllm-neuron uses, `model/qwen3/model.py`). Kiln has measured this on trn1 only; AWS
  ships the same LNL for trn2 and trn3 (vllm-neuron 0.24), and section 9 is the plan to
  measure it there. LNL is proprietary glue from the vllm-neuron stack, so it is a
  dependency to remove.
- **v1, NRT direct**: Kiln emits HLO itself, compiles with `neuronx-cc`, and drives
  `libnrt` through its documented C API (`nrt_load`, `nrt_execute`, and the explicit
  async queues `nrta_execute_schedule` / `nrta_tensor_*`). This removes every Python
  layer from the step loop. Precedents: ZML's Neuron platform and AWS's nkipy/Spike.

### L2 kernels

NKI is already a tile-level DSL: SBUF/PSUM tiles with a 128-wide partition axis, a
`stationary.T @ moving` matmul into PSUM, explicit DMA. Kiln takes TileLang's discipline
rather than its compiler: every kernel is written against a small set of tile helpers
(load a page set, matmul-accumulate, online softmax, store) with a torch reference beside
it, an autotuned tile config, and a numeric test on a real NeuronCore. A TileLang-to-NKI
lowering is possible later (shared to SBUF, fragments to PSUM, `T.Pipelined` to engine
double buffering) but is not on the critical path.

The NKI Library (`nkilib`, Apache-2.0) ships attention CTE/TKG, decode attention over a
block table, MoE CTE/TKG, MLA, RMSNorm, RoPE and MXFP8 kernels. Kiln reuses a kernel where
it supports the target generation and writes its own where it does not; trn1/inf2 (NKI
target `gen2`) are not what AWS tunes those kernels for.

### L4 runtime

The step loop, borrowed from SGLang and vLLM V1 and fitted to static shapes:

- **One unified step.** A step is a flat batch of `T` tokens from `B` sequences: decode
  tokens and prefill chunks together (chunked prefill and mixed batching). This is the
  single largest structural gap in vllm-neuron and NxD Inference, which run one prefill
  per step and pause decodes during it.
- **Three-axis buckets.** Every step is padded to a bucket `(B, T, P)`: sequences, tokens,
  and KV pages per sequence. The page axis is what stops a decode step from reading
  `max_model_len` worth of KV, the "~32x DMA overhead" AWS documents for its own decode
  path. Bucket ladders are powers of two with a geometric tail; every bucket is compiled at
  warmup and the compile cache is persisted to S3 keyed on (model, TP, SDK, bucket).
- **Radix prefix cache** (SGLang RadixAttention) over the paged KV pool: page-aligned
  token keys, reference counts, LRU eviction of unreferenced leaves, and insertion at
  prefill time so concurrent requests share a prefix while it is still being computed.
- **Preemption by recompute.** Admission does not reserve `max_model_len` per request (the
  vllm-neuron scheduler does, with a 30% KV budget by default). When the pool runs out, the
  youngest request is preempted and its prefix stays in the radix cache.
- **Zero-overhead overlap scheduling.** Step `N+1` is scheduled on the CPU while step `N`
  runs on the device. Sampled token ids never leave the device between steps; the next
  step's input ids are gathered on device from the previous step's output.
- **On-device sampling**: temperature, top-k, top-p and greedy inside the graph, so a step
  reads back `B` int32 token ids, not `B x vocab` logits.
- **Recurrent state for linear-attention layers** (Gated DeltaNet, KDA; `models/linear_attn.py`,
  `engine/state_pool.py`). A layer kind is the type of its spec (`AttnSpec` reads the paged KV
  cache, `LinearSpec` a fixed-size state). Each linear layer owns two device pools, conv
  `[rows, K - 1, channels]` and fp32 recurrent `[rows, heads, Dk, Dv]`, one row per running
  request plus a scratch row for padding; decode and prefill graphs take a `state_slot` per
  sequence, read the rows and write them back in place, so chunked prefill, decode and overlap
  scheduling compose like they do for KV pages. A sequence whose first position is 0 starts
  from zero state inside the graph, which is what resets a reused row or a preempted request.
  Pages are still allocated per token (they hold the attention layers' KV). The state only moves
  forward, so a cached prefix is resumed from a saved state and a rejected draft from a state
  kept for it; nothing ever rewinds a row.
- **State checkpoint pool** (`engine/state_pool.py`, `engine/radix_cache.py`,
  `engine/scheduler.py`): the prefix cache of these models. Rows of the state pool past the
  running ones (`state_checkpoints`, 2 x max_num_seqs by default, as SGLang's int8 checkpoint pool)
  hold checkpoints, the state of every linear layer after a page-aligned prefix, and a radix node
  owns at most one, the state after its last token (SGLang v0.5.21 `mem_cache/unified_cache/
  components/mamba.py`: one mamba state per node). A hit is the deepest node on the matched path
  that has one (SGLang's best_match_node, vLLM v0.30.0 `MambaManager.find_longest_cache_hit`'s
  rightmost cached block); the KV pages up to it come from the tree as for any model, and one small
  copy graph moves the checkpoint into the request's row before its first chunk. Checkpoints are
  taken where a prefill chunk ENDS (the copy runs right after that chunk's graph), so the scheduler
  splits chunks at the positions it wants one: the radix junction past the deepest checkpoint (the
  KV match runs further: another request shares those tokens; SGLang's mamba_branching_seqlen,
  vLLM's shared-prefix junction in `Scheduler._mamba_block_aligned_split`), optionally the
  prompt's last page boundary (vLLM's replay boundary; off by default: on trn1 it is one more
  prefill call on every uncached prompt) and every `state_checkpoint_interval` tokens (vLLM's
  prefix_cache_retention_interval, 0 by default). Decode copies one every `state_track_interval`
  tokens (SGLang mamba_track_interval, 256), keeping only the newest, so the next turn of a
  conversation resumes inside the previous answer. A saved row stays pending with its request until
  the pages under it are inserted, then joins its node (a duplicate is freed); it is never reused
  while pending, so a copy still in flight under overlap scheduling cannot overwrite a checkpoint
  the tree holds. A split leaves the checkpoint on the lower node, evicting a node frees its
  checkpoint, and under pool pressure checkpoints of unlocked nodes go alone in the eviction
  policy's order (SGLang's separate mamba LRU). The host tier copies a checkpoint to host memory
  with the pages under it and restores pages up to the deepest host checkpoint a prompt reaches.
  Per checkpoint: Qwen3.5-0.8B 18.6 MB (bf16 conv, fp32 recurrent state, measured); GLM-5.3-Flash at
  attention TP 32, 2 KDA heads per rank, 4.6 MB per rank, about 270 tokens of its per-rank MLA cache
  (from the config).
- **Speculative verify on these models**: a request holds 1 + k state rows; the verify graph runs
  its Q = 1 + k positions as recurrent steps from the current row (the decode step's own
  arithmetic) and writes the state after every position into the request's rows, and the row after
  the last accepted position becomes current: no copy and no recompute (vLLM v0.30.0
  `v1/attention/backends/gdn_attn.py` spec_state_indices; SGLang's per-draft
  intermediate_ssm_state_cache). Measured on trn1 (Qwen3.5-0.8B bf16, Q = 5): keeping every state
  costs 0.4 ms over keeping only the last at B=1 and 9 ms at B=8, where recomputing the accepted
  tokens would be a second verify-sized pass (41 ms at B=8), so snapshots are the cheaper rollback.
- Later: speculative decoding (EAGLE-3) compatible with the prefix cache,
  structured output (xgrammar masks applied on device), multi-LoRA, PD disaggregation over
  `libnrt` send/receive or NIXL, KV offload to host memory (HiCache).

### L3 models

Model code is functional PyTorch with one module per architecture. The same code runs on
CPU (as the correctness reference) and on the `neuron` device. Attention is a per-layer kind
(`AttnSpec`): grouped-query attention with optional sliding windows and sinks, or multi-head
latent attention (`kiln/models/mla.py`: DeepSeek-V3 / Kimi K2 / GLM-5, with DeepSeek Sparse
Attention for DeepSeek-V3.2 and GLM-5.x). The page pool keeps one K and one V tensor per layer
whatever the kind: for an MLA layer K is the kv_lora latent and V the rope key (plus the DSA
indexer key), one "KV head" of 576 (+128) values per token, replicated on every TP rank while
the heads are split, so the radix cache, chunked prefill, speculative verify, the host tier and
FP8 pages need nothing MLA-specific. Weights load from Hugging
Face safetensors and are sharded for tensor parallelism at load time. FP8 checkpoints are
rescaled: Neuron's FP8 E4M3 saturates at 240 on trn1/trn2, not the 448 Hugging Face
checkpoints assume (`models.md`, community-reported, to be measured).

**Tensor parallelism** is Megatron-style, one process per NeuronCore (`kiln/engine/tp.py`):
QKV / gate-up split on their output dim, o_proj / down on their input dim and all-reduced
in-graph, embedding and lm_head split by vocabulary rows, KV heads split or (more ranks than KV
heads) replicated. The degree is per layer KIND: the token mixers (GQA, MLA and Qwen sparse
attention, Gated DeltaNet, KDA, the MTP layer) run at the **attention TP** `attn_tp`, a divisor
of `tp` (`--attention-tp`; by default the largest one every mixer's head counts allow, which is
`tp` itself whenever they divide it, so those models run, and trace, exactly as before), while
the MLP, the experts, the embedding and the lm_head keep `tp`. The `tp` ranks form
`tp / attn_tp` groups of consecutive ranks (rank r is attention rank `r % attn_tp` of group
`r // attn_tp`, SGLang's rank layout; consecutive ranks share a trn1 chip, and an in-graph
all-reduce inside one chip measured 0.15 ms against 2.55 ms over four chips). Each group holds
every head split `attn_tp` ways and computes the same mixer output for every token, reduced over
the group (a `torch.distributed` subgroup; nothing at `attn_tp = 1`); the KV cache and the
recurrent state of a rank hold its attention rank's heads, and the loader slices mixer weights by
attention rank. This is the replicated form of SGLang's DP attention (`--attn-dp-size`): the same
per-rank layout without giving each group its own requests, so no per-group scheduler and no
token scatter / gather around the FFN, but also no KV saving. What it buys is head counts that
do not divide the TP the experts need: Qwen3.8-Flash-Next's 24 attention and 16 GDN key heads at
tp=16 or 32 run attention TP 8. The cost is that the mixer weights, compute, KV and state per
rank are `tp / attn_tp` times plain TP's; for Qwen3.8-Flash-Next the mixers are about 2.7 B of
its parameters (from config.json), so 0.34 GB per rank in FP8 at attention TP 8 instead of 0.17.

**DP attention** (`--dp-attention N`, `EngineConfig.dp_attention`; SGLang `--enable-dp-attention`,
v0.5.21 `srt/layers/dp_attention.py`; vLLM `data_parallel_size` with expert parallelism) is the
data-parallel form of the same layout: the N groups of `tp / N` ranks each serve DIFFERENT
requests, so the attention TP is `tp / N` and each rank's KV pages and recurrent-state rows hold
only its group's requests. That is what an MLA model needs: its compressed latent is not
head-partitioned, so under TP every rank holds it for every sequence (GLM-5.3-Flash at 128 x 8448
tokens: about 15 GB per rank), and N groups divide it by N.
- **Scheduling** (`engine/dp.py`): one unchanged `Scheduler` per group, each with its own page
  pool and radix cache, as SGLang runs one scheduler per DP rank. A request is placed when it is
  queued, by SGLang's TOTAL_TOKENS balancing (`data_parallel_controller.py`, `DPBudget.dispatch`:
  fewest tokens, then fewest requests) counting only the tokens the group's radix cache does not
  already hold, and stays in its group (a preempted request recomputes from its group's cache).
  `max_num_seqs` and `max_prefill_tokens` stay engine totals (`ceil(max_num_seqs / N)` requests and
  `max_prefill_tokens // N` prefill tokens per group and step, as SGLang divides
  `chunked_prefill_size` by `dp_size`), so the experts see the prefill batch they would see without
  DP attention.
- **Graph calls**: every rank still executes the same graph at the same time, because the MLP /
  experts run over all groups' tokens. A call holds every group's batch padded to one bucket
  (decode: the largest group's B; prefill: one chunk per group), group-major. Arguments a token
  mixer reads (positions, block tables, KV slots, window tables, state rows) are per group
  (`model_runner.PerGroup`, each rank takes its own); the rest (input ids, sampling, the token
  board) cover all `N * B` rows. Idle groups run padding into the null page.
- **Collectives** (`models/decoder.py`): the residual stream, embedding, MLP / experts, lm_head and
  sampling run over all `N * B` rows on every rank, as at plain TP. A mixer takes its group's rows
  (`_attn_in`, an `index_select` by a per-rank buffer, so one graph serves every rank), and its
  head-partial output goes back into the `N * B` rows zero-padded outside the group and is
  all-reduced over the world (`_attn_all_reduce`): one all-reduce that is both the attention-TP
  reduction and the gather of every group's rows (SGLang's `_dp_gather_via_all_reduce`, its SUM_LEN
  mode, with the attention reduction folded in). There is no scatter: the next mixer selects its
  rows again. Against SGLang's MAX_LEN form (attention-group reduce-scatter + world all-gather) it
  moves about twice the bytes (measured equal at decode sizes, 10% slower at 512 rows a group on
  one chip) but needs no subgroup, so every rank compiles and loads ONE NEFF of each graph (a
  subgroup collective gives each group its own, see attention TP above); at 32 ranks a graph's
  cost is a fixed ~5 ms per execution whatever its collectives (docs/neuron-notes.md "DP
  attention"). `dp_attention = 1` builds the same arguments and traces the same graphs as before
  (identical LNL cache keys, measured).

## 4. Technique map

| Technique | Origin | Kiln | Phase |
|---|---|---|---|
| Continuous batching | vLLM / SGLang | unified token-budget step | P0 |
| Paged KV cache | vLLM PagedAttention | page pool + block tables | P0 |
| Radix prefix cache | SGLang RadixAttention | page-aligned radix tree, insert at prefill | P0 |
| Chunked prefill + mixed batches | SGLang / vLLM V1 | ragged step, NKI ragged paged attention | P1 |
| Context-length bucketing | (new for Neuron) | page axis in every bucket | P1 |
| On-device sampling | NxDI | in-graph sampler | P1 |
| Overlap scheduling | SGLang | async executor, device-resident token ids | P1 |
| Preemption by recompute | vLLM | radix-preserving preemption | P1 |
| Tensor parallelism | Megatron | per-core shards, in-graph collectives; token mixers at their own degree (attention TP, replicated over chip-local groups) | P2 |
| DP attention | SGLang / vLLM | attention groups with their own requests, KV pools and schedulers; zero-padded world all-reduce of the mixer output (gather folded into the attention reduction) | P3 |
| FP8 weights + KV | all | 240-saturating rescale, FP8 KV pages | P2 |
| Linear-attention hybrids (Gated DeltaNet, KDA) | Qwen3.8 / Kimi | torch-level chunked scan + device state pool, prefix cache from state checkpoints, verify with per-position states (done); NKI chunked scan kernels next | P2 |
| MoE + expert parallelism | SGLang / DeepEP | NKI grouped GEMM, EP over libnccom | P3 |
| Speculative decoding | SGLang EAGLE-3 / MTP | draft graph + tree verify, radix-compatible | P3 |
| Structured output | SGLang xgrammar | on-device token masks | P3 |
| Cache-aware routing | SGLang router | prefix-affinity router across replicas | P4 |
| PD disaggregation | SGLang / vLLM | KV handoff over the host and TCP (engine/disagg.py); device to device over EFA with NIXL, opt-in (engine/nixl_kv.py) | P4 |
| NRT-direct executor | (ZML, nkipy) | HLO emit + nrta async queues | P4 |
| Tile-level kernel authoring | TileLang | tile helpers + autotune + reference tests | P1 onward |

## 5. Model ladder

From `models.md` section 5, sized against 90% of HBM:

| Step | Model | Target | Why |
|---|---|---|---|
| a | Qwen3-0.6B, Qwen3-1.7B (BF16) | trn1.2xlarge | official reference on every Neuron target; Mini-SGLang-Neuron's published numbers |
| b | Qwen3.8-27B (BF16 55.6 GB) | trn1.32xlarge, 1 trn2 chip | best dense open model (AA 33.7); Gated DeltaNet layers carry to larger Qwen3.8 |
| c | MiMo-V2.6-Flash (FP8 ~315 GB) | trn1.32xlarge | AA 37.9, 309B/15B MoE, MIT; same code path as d |
| d | MiMo-V2.6-Pro (FP8 ~1,040 GB) | trn2.48xlarge | AA #1 open model (46.3), 1.02T/42B MoE, MIT; community port does 220 ms/token |

## 6. Benchmark protocol

- Same instance type, same SDK, same model, same prompts and output lengths for Kiln and the
  baseline. Offline throughput, and online TTFT / TPOT / ITL percentiles under a request
  rate sweep. Raw logs are committed under `bench/results/<date>-<instance>-<model>/`.
- Baselines: vllm-neuron 0.24 on trn2 (Qwen3, Llama 3, gpt-oss). On trn1, where nothing
  maintained exists, the baseline is the last SDK that served it (NxD Inference on SDK 2.28)
  and Mini-SGLang-Neuron's published Qwen3-0.6B numbers (367 tok/s offline, TP=2, 6
  concurrent).
- Correctness gates every benchmark: greedy decoding matches Hugging Face transformers
  token for token on a fixed prompt set, or logits agree within a stated tolerance.

## 7. Phases

- **P0 (bring-up)**: repository, spot fleet, device probes; hardware-independent runtime
  (scheduler, radix cache, page pool, sampler, API server) tested on CPU; Qwen3 dense on
  the LNL executor on trn1.2xlarge with greedy parity against transformers.
- **P1 (beat the baseline on one chip)**: NKI paged attention with page-axis buckets,
  ragged mixed batches, on-device sampling, overlap scheduling; benchmarks on trn1.2xlarge
  and against vllm-neuron 0.24 on trn2.
- **P2 (scale out)**: tensor parallelism across cores and chips, FP8, Qwen3.8-27B with
  linear-attention kernels.
- **P3 (MoE and speculation)**: expert parallelism, MiMo-V2.6-Flash on trn1.32xlarge,
  EAGLE-3 / MTP, structured output.
- **P4 (flagship and fleet)**: MiMo-V2.6-Pro on trn2.48xlarge, NRT-direct executor,
  cache-aware router, PD disaggregation.

## 8. Licensing

- NKI Library, nki, vllm-neuron: Apache-2.0. mini-sglang: MIT. Code adapted from any of
  them keeps its notice and is listed in `THIRD_PARTY_NOTICES.md`.
- Mini-SGLang-Neuron's own additions have no clear license; Kiln does not copy from it.
- `libnrt`, `libnccom`, `neuronx-cc` and LNL are proprietary binaries under the AWS Neuron
  License Agreement (no redistribution). Kiln runs on the DLAMI and ships none of them.

## 9. Roadmap: trn2 and trn3 (support and speed comparison)

Owner priority (2026-10-02): the latest models on trn2 and trn3, supported and measured
against AWS's managed path. Research, sources and costs: `docs/research/trn2-trn3.md`
(read 2026-10-02/03; AWS readings are read-only API calls, no instance launched).

What the research established:

- **Capacity.** trn2.48xlarge is offered in us-east-2 and ap-south-2, trn2.3xlarge (one
  Trainium2) in sa-east-1, ap-southeast-4 and ap-south-2, trn2u.48xlarge in ap-south-2 only.
  No trn3 type exists in EC2 for this account, and UltraServer Capacity Blocks are
  "not authorized". Spot on 2026-10-03: trn2.48xlarge $14.56/h (us-east-2c, 30-day mean
  $12.97), trn2.3xlarge $2.27-2.57/h (30-day mean $1.53 in sa-east-1c); placement score 1 of
  10 for both. Trn spot quota is 256 vCPU per region, so one trn2.48xlarge at a time and
  never beside a trn1.32xlarge in the same region.
- **What changes per chip** (SDK 2.32 sources): LNC=2 by default on trn2/trn3 (4 logical
  cores per chip, 64 per 16-chip instance, compiler and runtime LNC must match); HBM per
  logical core 16 / 24 / 36 GiB (trn1 / trn2 / trn3, LNL `HBM_MEMORY_GB`); FP8 e4m3 max 240
  on trn1/trn2 and OCP e4m3fn 448 on trn3, with the unsafe-cast compiler flag forbidden on
  trn3; MXFP8/MXFP4 matmul (`nc_matmul_mx`) and the NKI Library's MX MoE, MLA and MXFP8
  attention kernels on trn3 only; tensor indirection, Vector-engine exp and BF16 PSUM on trn3
  only. The NKI Library's attention, QKV, norm, router and MoE kernels are tuned for trn2 and
  trn3, which is where Kiln's trn1 kernel gap (16-21% behind at batch 12-32) can close.
- **Baselines.** vllm-neuron 0.24 serves Llama 3, GPT-OSS, Qwen3 dense and Qwen3-VL only;
  none of the top open models (MiMo-V2.6, GLM-5.3, Kimi K3, Qwen3.8, DeepSeek-V4.1) has a
  managed path on either chip. Same-instance, same-SDK comparisons are therefore made on
  control models, with unmerged community ports (vllm-neuron PR #40 MiMo-V2.5, NxDI PR #150
  MiMo-V2.5-Pro) as labeled references.

Engine work (gap list in the research doc, section 6): one platform module (target, NKI gen,
LNC, HBM per logical core, FP8 max, MX support) feeding device setup, compiler arguments, KV
sizing and every bench record; explicit `NEURON_LOGICAL_NC_CONFIG` and `--logical-nc-config`;
vllm-neuron's compiler argument set on trn2/trn3; MX experts through `nc_matmul_mx` on trn3;
chip-aligned TP groups and a measured TP=64 host-broadcast cost; device-free trn2/trn3
compiles into the S3 NEFF cache; a vllm-neuron 0.24 mode for `bench/baseline_vllm.py`;
`infra/fleet.sh` outside us-east-2 and on local NVMe; every trn1 probe in
`docs/neuron-notes.md` re-run at LNC=2.

| Phase | Instance | Measurement | Est. cost |
|---|---|---|---|
| T0 | none | platform module, LNC, compiler args, FP8/KV sizing, harness, fleet region; CPU tests | $0 |
| T1 | trn2.3xlarge spot, sa-east-1c, SDK 2.32 | probes at LNC=2; Qwen3-0.6B and Qwen3-8B parity; Kiln (DP=4 / TP=2x2 / TP=4) vs vllm-neuron 0.24 TP=4 on `bench/offline.py`'s workload, plus 8K-context decode | 6 h, $9-15 (cap $21) |
| T2 | trn2.48xlarge spot, us-east-2c | reproduce AWS's Llama-3.1-8B TP=8 number, then Kiln vs vllm-neuron 0.24 on Llama-3.1-8B and Qwen3-32B; Qwen3-30B-A3B vs NxDI (SDK 2.31.1); MiMo-V2.6-Flash vs the vllm-neuron PR #40 branch | 11 h, $143-160 |
| T3 | trn2.48xlarge | MiMo-V2.6-Pro FP8 and packed MXFP4 at TP=64, concurrency 1/16/48, against PR #150's rows (SDK 2.29, reference only) | 12 h, $156-175 |
| T4 | trn3 | blocked on access (owner request to AWS); meanwhile compile-only checks for `--target trn3`; then GPT-OSS-120B MXFP4 vs vllm-neuron, MiMo-V2.6-Pro native MXFP4, Kimi K3 | no public price |

Placement in the existing phases: T0-T1 belong to P1 (beat the baseline on one chip, now on a
trn2 chip as well as trn1), T2 to P2/P3, T3 to P4. Every result carries instance, AZ, AMI,
SDK, target string, LNC, model, precision, command, commit and date, and $/Mtok at the spot
price paid, since at today's prices trn1.32xlarge buys about 1.4-2x the HBM bandwidth per dollar of
trn2.48xlarge.
