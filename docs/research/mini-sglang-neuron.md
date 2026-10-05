# Mini-SGLang-Neuron and Mini-SGLang: code-level research report

Date: 2026-10-01. Read-only research; nothing was pushed or installed.

## 0. Sources and method

| Item | Value |
|---|---|
| Upstream clone | `mini-sglang/` (https://github.com/sgl-project/mini-sglang), `--depth 1`, HEAD `9a91cfafe` (2026-05-17) |
| Neuron port clone | `mini-sglang-neuron/` (https://github.com/yottalabsai/mini-sglang-neuron), `--depth 1`, default branch `dev`, HEAD `2f9ba6ebd` (2026-04-09) |
| NxD Inference wheel (pinned dependency, for verification) | `nxdi-0.7.15063/` = `neuronx_distributed_inference==0.7.15063+bafa28d5`, downloaded from `pip.repos.neuron.amazonaws.com` and unzipped (not installed) |
| GitHub metadata | `gh api` (repo, commits, PRs, issues, forks, compare) |
| Blog | https://www.yottalabs.ai/post/mini-sglang-neuron-bringing-lightweight-llm-inference-to-aws-trainium-and-inferentia (Mar 17 2026) |
| Neuron release content | https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/prev/content.html and `.../releasecontent.html` |

All paths below are relative to `<local-dir>/research/`. "MS" = `mini-sglang/python/minisgl`, "MSN" = `mini-sglang-neuron/python/minisgl`, "NXDI" = `nxdi-0.7.15063/neuronx_distributed_inference`.

Nothing in this report was executed on Neuron hardware. Statements about runtime behaviour on Trainium are derived from code reading and are marked UNCONFIRMED where they go beyond what the code states.

---

## 1. Upstream Mini-SGLang architecture

### 1.1 Process layout (ZMQ + torch.distributed)

`MS/server/launch.py:40-113`:
- Main process: FastAPI/uvicorn API server (`MS/server/api_server.py`), which calls `start_subprocess`.
- One **scheduler process per TP rank** (`launch.py:59-69`, name `minisgl-TP{i}-scheduler`). Each owns one `Engine` (one GPU).
- One **detokenizer** process (`launch.py:72-87`) and `num_tokenizer` tokenizer processes (`launch.py:88-103`). Default `num_tokenizer=0` (`MS/server/args.py:18`) means tokenizer and detokenizer share one process (`share_tokenizer`, `args.py:22-23`).
- Startup handshake through an `mp.Queue` ack (`launch.py:105-111`).

Transport (`MS/scheduler/config.py:24-33`, `MS/server/args.py:26-34`): ZMQ IPC sockets `ipc:///tmp/minisgl_{0..4}<pid>`.
- tokenizer -> rank-0 scheduler: PUSH/PULL (`MS/scheduler/io.py:35-45`).
- rank 0 -> other ranks: ZMQ PUB/SUB of the raw message bytes, plus a gloo `broadcast` of the message count so every rank consumes the same number (`io.py:49-62`, `88-122`).
- rank 0 -> detokenizer: PUSH (`io.py:124-130`); non-primary ranks reply nothing (`io.py:132-133`).
- Tensor collectives: NCCL / custom PyNCCL (below). A gloo CPU group is used for barriers and memory agreement (`MS/engine/engine.py:112-137`, `170-189`).

### 1.2 Scheduler loop and overlap scheduling

`MS/scheduler/scheduler.py`:
- Policy: prefill first, then decode; never mixed in one batch (`_schedule_next_batch`, `219-225`, with a `TODO: support other policies`).
- `_prepare_batch` (`204-217`): pad batch to a CUDA-graph size, allocate pages, build positions / input mapping / write mapping on pinned host memory, gather `out_loc` from the page table, call `attn_backend.prepare_metadata`.
- `_forward` (`227-233`): read input ids from the on-GPU `token_pool`, run the engine, write sampled tokens back into `token_pool` **on the GPU** so the next step can read them without a host sync.
- **Overlap scheduling** (`overlap_loop`, `83-106`; `run_forever`, `120-131`): two CUDA streams. The scheduler stream does metadata prep; the engine stream runs batch N+1 while the CPU processes batch N's results (`_process_last_data`, `138-167`, waits on a `copy_done` CUDA event). Disabled with `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` (`MS/env.py:69`). Technique credited to NanoFlow (`mini-sglang/docs/features.md`).
- Requests can be aborted end to end (`AbortBackendMsg`, `scheduler.py:190-195`; `MS/tokenizer/server.py:69,104`).
- After a prefill, non-chunked requests are inserted into the prefix cache immediately (`scheduler.py:163-164`), so concurrent requests can share a prefix.

### 1.3 Radix cache

`MS/kvcache/radix_cache.py` (237 lines):
- Tree of `RadixTreeNode` (`17-84`). The child key is the first token (page_size 1) or a tuple of the first page of tokens (`_get_key_fn`, `234-237`), so matching is page-aligned (`align_down` in `_tree_walk`, `205-231`, and in `insert_prefix`, `136-146`).
- Reference counting with `lock_handle` (`113-130`) moves sizes between `evictable_size` and `protected_size`.
- LRU eviction: collect unreferenced leaves, heapify on `timestamp`, pop until enough is freed, and push parents that become leaves (`148-175`, `190-203`).
- Key comparison is C++ `std::mismatch` exposed through tvm-ffi (`MS/kernel/csrc/src/radix.cpp:19-40`, `MS/kernel/radix.py`).
- `naive` alternative: `MS/kvcache/naive_cache.py`.

### 1.4 Paged KV allocator

- KV pool `MHAKVCache`: one tensor `(2, num_layers, num_pages, page_size, local_kv_heads, head_dim)` (`MS/kvcache/mha_pool.py:28-32`). Stores go through a JIT CUDA kernel (`MS/kernel/csrc/jit/store.cu`, `MS/kernel/store.py`).
- The page table is `int32 [max_running_req + 1, align32(max_seq_len)]` and stores **token-level locations** (`MS/engine/engine.py:65-73`). The extra row is a dummy request for padding (`engine.py:89-98`).
- `CacheManager` (`MS/scheduler/cache.py:15-146`): `free_slots` holds page-start token indices. `allocate_paged` allocates `div_ceil` pages per request and evicts from the radix tree on shortage (`_allocate`, `106-113`). `cache_req` reconciles freshly computed tokens with the tree and frees duplicates (`55-79`). `lazy_free_region` batches frees (`93-104`).
- KV size is set from free memory: `memory_ratio * free - model_memory` (`engine.py:148-168`, default ratio 0.9 at `MS/engine/config.py:26`).

### 1.5 Chunked prefill

`MS/scheduler/prefill.py`: `PrefillAdder` (`32-113`) spends a `token_budget = max_extend_tokens` (default 8192, `MS/scheduler/config.py:16`, CLI `--max-prefill-length`). Requests that do not fit are emitted as `ChunkedReq` (`23-29`), which cannot decode. `reserved_size` accounts for in-flight decode tokens (`MS/scheduler/decode.py:28-30`). Admission happens only when `estimated_len + reserved <= available` (`prefill.py:50-54`).

### 1.6 Attention backends

Registry: `MS/attention/__init__.py:22-68`. Auto selection (`MS/engine/engine.py:218-229`): sm100 -> `trtllm`; sm90 -> `fa,fi` (hybrid: prefill FA, decode FlashInfer); otherwise `fi`.
- `fa` (`MS/attention/fa.py`): `sgl_kernel.flash_attn.flash_attn_with_kvcache` with `ver=3`, or 4 on sm100 (`fa.py:46`, `139-182`).
- `fi` (`MS/attention/fi.py`): FlashInfer `BatchPrefill/BatchDecodeWithPagedKVCacheWrapper`, forced to the `fa2` backend ("flashinfer fa3 is slow", `fi.py:93-103`), **page_size 1 only** (`fi.py:57,66,220`), with CUDA-graph decode wrappers (`244-271`).
- `trtllm` (`MS/attention/trtllm.py`): FlashInfer TRT-LLM kernels. Overrides page size to 64 if it is not 16/32/64 (`engine.py:227-229`).
- `HybridBackend` (`MS/attention/base.py`).

### 1.7 CUDA graphs

`MS/engine/graph.py`: decode-only capture (`can_use_cuda_graph`, `149-150`). Batch sizes are `[1,2,4] + range(8, max+1, 8)` with max 256 on GPUs over 80 GB, otherwise 160 (`49-67`). A shared memory pool is used across captures (`128-144`). Batches are padded to the next captured size (`160-166`).

### 1.8 Tensor parallelism

- Megatron-style layers in `MS/layers/linear.py` (`LinearQKVMerged`, `LinearColParallelMerged`, `LinearOProj`/`LinearRowParallel` with all-reduce, `91-127`). KV heads are replicated when `tp > num_kv_heads` (`allow_replicate=True`).
- Communication plugins in `MS/distributed/impl.py`: `TorchDistributedImpl` (NCCL) or a custom **PyNCCL** wrapper (`44-90`; CUDA source in `MS/kernel/csrc/src/pynccl.cu`, bundled `nccl227.h`), enabled by default (`use_pynccl=True`, `MS/engine/config.py:29`).
- MoE is TP-sharded only: the expert intermediate dimension is split and then all-reduced (`MS/layers/moe.py:28-58`). Kernels are Triton fused MoE plus `sgl_kernel` topk/align (`MS/moe/fused.py`, `MS/kernel/triton/fused_moe.py`). No expert parallelism.

### 1.9 Models, sampling, API

- Models (`MS/models/register.py:5-12`): Llama, Qwen2, Qwen3, Qwen3-MoE, Mistral (+ Mistral3 text).
- Sampler: `flashinfer.sampling` on GPU (`MS/engine/sample.py:24-30`).
- API (`MS/server/api_server.py:228-313`): `POST /generate`, `/v1` root, `POST /v1/chat/completions` (streaming, plus `stream=false` since #125 on 2026-05-17), `GET /v1/models`, and an interactive shell (`319-410`). There is no `/v1/completions`, no logprobs and no `n>1`.

### 1.10 Lines of code per component (upstream HEAD)

Raw line counts (including blanks and comments) over `*.py, *.cu, *.cpp, *.h, *.cuh` under `MS/`:

| Component | Lines | Notes |
|---|---:|---|
| kernel | 2664 | includes 571-line `nccl227.h`, 496-line `tensor.h`, CUDA store/index/pynccl, Triton MoE |
| scheduler | 846 | |
| server | 840 | api_server 452, args 268, launch 117 |
| models | 803 | |
| attention | 778 | fi 271, fa 182, trtllm 162 |
| layers | 691 | |
| benchmark (pkg) | 575 | |
| kvcache | 559 | radix 237 |
| engine | 539 | engine 233, graph 171 |
| utils | 522 | |
| moe | 308 | |
| tokenizer | 255 | |
| top-level (`core.py` etc.) | 232 | |
| message | 201 | |
| distributed | 147 | |
| llm | 101 | |
| **Total** | **10061** | Python only: **8090** (README advertises "~5,000 lines of Python") |

---

## 2. What Mini-SGLang-Neuron changed

### 2.1 The fork in one sentence

It keeps Mini-SGLang's front end, scheduler, radix tree and allocator, and it **deletes the whole model/attention/kernel/TP stack**, replacing it with a **NeuronX Distributed Inference (NxDI) model** that is compiled ahead of time (AOT). Nothing model-side is written by Yotta.

Evidence:
- Fork base is upstream `6ffce63b8` (2026-01-19, "[Fix] Fix offline inference problem (#62)"). The first Neuron commit is `b54b7ec8c` on 2026-02-02.
- `gh api compare 6ffce63b8...2f9ba6ebd`: 37 commits, 80 files, **python/ +1157 / -3756** lines. Removed: `attention/*`, `layers/*`, `models/*`, `distributed/impl.py`, `kvcache/mha_pool.py`, CUDA `store.cu`/`index.cu`/`pynccl.cu`/`nccl227.h`, and the kernel tests. Added: `neuron/model_loader.py` (334), `neuron/inputs.py` (150), `server/tool_parser.py` (35).
- Commit messages: "Plug-in Neuron model backend from NxD-inference" (`aa6b19593`), "remove models and layers folders" (`a6f1e38d1`), "remote cuda attention" (`d0861abba`).

### 2.2 Which Neuron stack: NxDI on torch-neuronx, AOT-compiled

- `MSN/neuron/model_loader.py:17-19` imports `neuronx_distributed_inference` (`NeuronConfig`, `MODEL_TYPES`, `load_pretrained_config`).
- `_get_neuron_model_cls` (`54-73`) maps an HF `architectures[0]` to `NXDI MODEL_TYPES[model]["causal-lm"]`, with special cases for `qwen3moe -> qwen3_moe` and `LlavaForConditionalGeneration -> pixtral`.
- Compile and load go through NxDI `model.compile(path)` / `model.load(path)` (`model_loader.py:91-99`, `199-223`). NxDI builds the graphs with `neuronx_distributed.trace.ModelBuilder` / `parallel_model_trace` (`NXDI/models/application_base.py:20,156`, `NXDI/models/model_wrapper.py:21,184`). That is torch-neuronx tracing to NEFF through neuronx-cc. It is **not** lazy torch-xla execution of the model and **not** the native PyTorch Neuron backend.
- `torch_xla` is used only for housekeeping: `xm.get_memory_info` (`MSN/engine/graph.py:21-23`), `torch_xla.sync()` / `xm.wait_device_ops()` (`MSN/engine/engine.py:132-150`, `MSN/scheduler/scheduler.py:81,329-330`), and a no-op `xm.mark_step()` after sampling (`engine.py:178`). The engine "device" is the CPU (`engine.py:45`).
- NxDI parallel layers (`neuronx_distributed`) provide TP (2.6).

### 2.3 NxDI configuration it hard-codes

`MSN/neuron/model_loader.py:290-311` `_default_neuron_config`:

```python
"tp_degree": tp, "ctx_batch_size": 1, "enable_bucketing": False,
"batch_size": max_running_req, "max_context_length": max_model_len,
"max_new_tokens": max_extend_tokens, "pa_block_size": 1,  #load_cfg.block_size,
"pa_num_blocks": num_blocks, "is_block_kv_layout": True, "is_prefix_caching": True,
#"chunked_prefill_config": None,
"is_continuous_batching": (max_batch_size>1), "attn_kernel_enabled": False,
"seq_len": max_model_len,
```

`neuron_config_overrides` can add keys (`MSN/engine/config.py:34`), but **no CLI flag exposes it** (`MSN/server/args.py`). It is reachable only through the Python `LLM(...)` kwargs.

### 2.4 Static shapes: no bucketing, padding to maximum everywhere

- With `enable_bucketing=False` and `is_prefix_caching=True`, NxDI generates exactly these buckets:
  - Context encoding (CTE): `[[max_ctx, 0], [max_ctx, max_ctx]]`, meaning (active tokens, prefix length) (`NXDI/modules/autobucketing.py:149-176`, `22-42`).
  - Token generation (TKG): `[[1, max_length]]` (`autobucketing.py:203-221`).
  - So there are three graphs in total, all at the maximum size.
- Prefill inputs are built as `(batch, max_seq_len)` tensors regardless of chunk length (`MSN/neuron/inputs.py:37-88`, shapes at `43-56`). `ctx_batch_size=1` means NxDI runs a multi-request prefill batch as **sequential batch-1 calls** (`NXDI/models/model_wrapper.py:1499-1560`, split by compiled batch size; `1338-1342` for ctx vs tkg batch). So every prefill chunk costs a full `max_seq_len` context-encoding pass per request (UNCONFIRMED by measurement; follows from the shapes).
- Decode is padded to `neuron_config.batch_size == max_running_req` with dummy requests (`MSN/engine/graph.py:103-111`). Block tables are `(batch, max_seq_len)` (`inputs.py:111-113`).
- Page ID 0 is reserved as the dummy page and real pages are 1..N (`MSN/engine/engine.py:52-59`, `MSN/scheduler/cache.py:14-16`). NxDI requires sorted `seq_ids`, so inputs are sorted and outputs un-sorted (`model_loader.py:101-128`).
- Graph cache: artifacts go to `<model>/neuron-compiled-artifacts/<md5(config json)>`, or `local-models/<model>/...`, or `$NEURON_COMPILED_ARTIFACTS` (`model_loader.py:130-154`). A `fcntl` lock guards against multi-process compile races (`167-182`, `210-223`). Non-local HF ids are first re-saved to `local-models/` (`161-165`). `compile_only` / `skip_compile` / `hlo_debug` / `compile_dry_run` exist on `EngineConfig` (`MSN/engine/config.py:35-40`, used in `MSN/engine/graph.py:46-98`) but are not on the CLI.

### 2.5 Paged attention and KV layout on Neuron

- KV memory is owned **inside the NxDI compiled model** by `BlockKVCacheManager`: per-layer parameters of shape `(num_blocks [+reserved], block_size, num_kv_heads_per_rank, head_dim)` (`NXDI/modules/kvcache/block_kv_cache_manager.py:15,53-86`). Mini-SGLang only manages indices: `slot_mapping` and `block_tables` are built from its token-level page table (`inputs.py:67-68`, `127-128`).
- `pa_block_size = 1`, so every token is its own "block" and the KV gather is token-granular (`model_loader.py:300`; the `#load_cfg.block_size` comment shows larger blocks were tried). The `--page-size` CLI flag was removed and `CacheManager` says `TODO: support page_size > 1` (`MSN/scheduler/cache.py:14`).
- The **attention kernel is native PyTorch ops compiled by neuronx-cc, not NKI.** `attn_kernel_enabled=False` makes `get_flash_attention_strategy` return `NONE` (`NXDI/modules/attention/attention_base.py:1330-1333`). The block-KV token-gen NKI kernel defaults off (`NXDI/models/config.py:415`), as do qkv/mlp kernels (`config.py:391,398`) and `attn_tkg_nki_kernel_enabled` (`config.py:26,413`). RMSNorm uses the `AwsNeuronRmsNorm` compiler custom call, not NKI (`NXDI/modules/custom_calls.py:15-42`).
- KV page count is not memory-derived. The default is `max_seq_len * max_running_req` because "In NxDI, the page number should be pre-allocated before the model weight is loaded" (`MSN/engine/engine.py:115-125`). With the defaults (6 x 2048) the radix cache therefore only has room for the running set unless `--num-pages` is raised (README uses 16384).

### 2.6 Tensor parallelism

- **One scheduler process for all TP ranks.** `launch.py` spawns rank 0 and then `break`s (`MSN/server/launch.py:62-74`, with a commented-out `xmp.spawn` at `76-86`).
- The engine's process group is gloo with `world_size=1` (`MSN/engine/engine.py:93-103`). `minisgl.distributed` is reduced to a `DistributedInfo` dataclass (`MSN/distributed/tp.py`, 36 lines).
- NxDI compiles a single SPMD graph over `tp_degree` NeuronCores. The collectives (all-reduce in row-parallel layers) are compiled into the NEFF. `NEURON_RT_NUM_CORES=TP_SIZE` selects the cores (README, `server.sh:19`).

### 2.7 Sampling

On the host CPU. NxDI returns full-vocab logits to the CPU, and `Sampler._sample_cpu` does temperature, top-k and top-p with torch CPU ops (`MSN/engine/sample.py:47-113`; `engine.py:166-168`). The vectorised version replaced a per-row Python loop in `53a47d667` (2026-03-20, "remove serverl cpu memory-bound performance bottleneck"). NxDI on-device sampling (`on_device_sampling_config`, `NXDI/models/config.py:170-173`) is not used. A placeholder `sampling_params` tensor of ones is passed (`inputs.py:41,96`).

### 2.8 Scheduler changes vs upstream

- **Overlap scheduling disabled**: `run_forever` always calls `normal_loop` with the note "NxDI does not support async execution now" (`MSN/scheduler/scheduler.py:321-326`). This is debatable: the pinned NxDI has `async_mode` (`NXDI/models/config.py:177`), including a prefix-caching branch (`NXDI/models/model_base.py:3316-3336`). Whether it is usable with this configuration is UNCONFIRMED.
- **Radix insert only when a request finishes** (`MSN/scheduler/scheduler.py:102-110` -> `MSN/scheduler/cache.py:63-71`). Upstream inserts right after prefill (`MS/scheduler/scheduler.py:163-164`). Concurrent requests sharing a system prompt therefore cannot reuse each other's KV until one completes.
- **No backend abort**: the fork predates upstream's `AbortBackendMsg`. `api_server.abort_user` only deletes frontend state (`MSN/server/api_server.py:281-287`), so a disconnected client keeps consuming a slot until `max_tokens`.
- Decode batch order is `list(set)` (`MSN/scheduler/decode.py:23-26`). Upstream sorts by uid (#113). This is harmless with a single process.
- The radix tree is a page-size-1 copy of upstream (`MSN/kvcache/radix_manager.py`, 223 lines).

### 2.9 Prefill input layout with a prefix hit (UNCONFIRMED correctness question)

For prefill, the builder passes the **whole sequence** `0..device_len` as `input_ids` with positions `0..device_len`:
- `MSN/scheduler/scheduler.py:239-240` uses `full_load_indices`.
- `MSN/neuron/inputs.py:64-68` builds the tensors.
- `slot_mapping` holds only the new `device_len - cached_len` slots and is left-aligned (`inputs.py:68`).
- `computed_context_lens = cached_len` (`inputs.py:75-77`).

I could not confirm from NxDI's code that this is the layout NxDI's prefix-caching context-encoding path expects (active tokens only, versus full sequence). If it is not, prefix-hit and chunked-prefill outputs would be wrong or would recompute the prefix. The repo contains no correctness test for radix or chunked prefill. The tool-calling doc runs with the default 8192 prefill budget, so it never chunks. **Recommendation: test logits with radix vs naive on shared-prefix prompts before reusing this path.**

### 2.10 NKI kernels shipped

**None.** The repo has no `nki` import anywhere (grep over `*.py, *.sh, *.md, *.cpp, *.h`). The README's "Kernel Acceleration ... (e.g., radix cache key comparison)" is the host-side C++ `fast_compare_key` (`MSN/kernel/csrc/src/radix.cpp:19-40`), inherited unchanged from upstream. `MSN/kernel/csrc/include/minisgl/{utils.cuh,warp.cuh,tensor.h}` are leftover CUDA headers. The blog's "kernel-level optimizations for critical execution paths" has no counterpart in the code.

### 2.11 Model support

- What is tested and documented: **Qwen3 dense** only (`mini-sglang-neuron/docs/features.md` "Supported Models": Qwen-3). The benchmarks use Qwen3-0.6B, and `mini-sglang-neuron/docs/tool_calling.md:3` uses Qwen3-4B on trn1.2xlarge with TP=2.
- What can be loaded in principle: anything in NxDI `MODEL_TYPES`, which is gpt_oss, llama, llama4, mllama, mistral, mixtral, pixtral, dbrx, qwen2, qwen3, qwen3_moe (`NXDI/utils/constants.py`, `MODEL_TYPES`). Whether MoE or multimodal models work with the forced `is_block_kv_layout + is_prefix_caching + pa_block_size=1` config is UNCONFIRMED. Upstream's Llama and Qwen-MoE native implementations were deleted.

### 2.12 Additions beyond the port

- Qwen3 `<tool_call>` parser and OpenAI `tools` / `tool_calls`, streaming and non-streaming (`MSN/server/tool_parser.py`, `MSN/server/api_server.py:92-151,220-240,330-377`; PR #3, 2026-04-09). In streaming mode with tools, the full text is buffered and parsed at the end (`api_server.py:224-229`).
- Per-IP rate limit through slowapi, `--max-req-per-min` (`MSN/server/args.py:20,178-183`, `api_server.py:38-39,304-332,471-472`; PR #1, 2026-04-07).

### 2.13 LOC (Neuron fork)

| Component | Lines |
|---|---:|
| kernel (mostly leftover headers) | 1139 |
| server | 890 |
| scheduler | 832 |
| utils | 449 |
| kvcache | 444 |
| benchmark (pkg) | 513 |
| engine | 504 |
| **neuron** (new) | **488** |
| tokenizer | 248 |
| message | 195 |
| top-level | 192 |
| llm | 101 |
| distributed | 45 |
| **Total** | **6040** (Python 5123) |

---

## 3. `init_setup.sh` and the assumed environment

- `mini-sglang-neuron/init_setup.sh:1-4`: `pip install -e .`, then `conda install -c conda-forge ninja`, and a commented `hf download Qwen/Qwen3-0.6B`. No AMI or SDK installation. It assumes conda exists in the container (UNCONFIRMED that the DLC ships conda).
- Assumed base: the Docker DLC `public.ecr.aws/neuron/pytorch-inference-neuronx:2.9.0-neuronx-py312-sdk2.27.1-ubuntu24.04` (`mini-sglang-neuron/README.md:45`). That is **Neuron SDK 2.27.1, PyTorch 2.9, Python 3.12, Ubuntu 24.04**. No DLAMI is named.
- Pinned packages (`mini-sglang-neuron/pyproject.toml:24-41`): `torch-neuronx==2.9.0.2.11.19912`, `neuronx-cc==2.22.12471.0`, `neuronx-distributed==0.16.25997`, `neuronx_distributed_inference==0.7.15063`, `libneuronxla==2.2.14584.0`, `transformers>=4.56,<=4.57.3`, plus `apache-tvm-ffi`, `pyzmq`, `fastapi`, `uvicorn`, `slowapi`.
  - Neuron's previous-release artifact page lists exactly these under "**Neuron 2.27.1 (01/14/2026)**".
  - The current SDK is **2.32.0 (08/17/2026)** per `releasecontent.html`, so the pins are five minor releases behind.
- Shell scripts `run_minisgl.sh` / `server.sh` check for `ninja` and `g++`/`clang++`, which tvm-ffi needs to JIT-build `radix.cpp`.
- **The package is not installable at HEAD.** `pyproject.toml` line 1 starts with a stray backtick and line 40 has `"slowapi>=0.1.9",,`. Both came in with commit `a6eb9b86b` (PR #1, 2026-04-02). Measured with a dry run in a scratch venv: `pip install --dry-run --no-deps -e .` fails with `tomllib.TOMLDecodeError: Invalid statement (at line 1, column 1)`. The last parseable version is `e984a5b7a` (2026-03-20).

---

## 4. Benchmarks and reported numbers

The repo has no raw result logs. The numbers come from the charts in `mini-sglang-neuron/docs/*.png`. The blog states the baseline is **vLLM-Neuron 0.4.1** and gives no numbers in text.

### 4.1 Offline throughput

Setup:
- Scripts: `mini-sglang-neuron/benchmark/offline/bench.py` and `bench_vllm_neuron.py`.
- Hardware/model: trn1.xlarge, 2 NeuronCores, TP=2, Qwen3-0.6B, bf16.
- Workload: 256 requests, random input of 100-1024 tokens and random output of 100-1024 tokens, temperature 0.6, `ignore_eos`.
- Prompts are random token ids, so there is **no prefix sharing** and radix cannot help.

| Engine | tok/s (`docs/offline_bench.png`) |
|---|---:|
| mini-sglang-neuron, naive | 367.53 |
| mini-sglang-neuron, radix | 367.71 |
| vllm-neuron, no prefix caching | 359.22 |
| vllm-neuron, prefix caching | 231.48 |

Configuration asymmetries:
- Both engines use `max_num_seqs=6` and `max_model_len=2048`.
- mini-sglang-neuron: `num_page_override=16384` at page size 1 (`bench.py:22-30`).
- vLLM, no prefix caching: `block_size=128` with `num_gpu_blocks_override=9` (`bench_vllm_neuron.py:19-30`).
- vLLM, prefix caching: `128` blocks.

### 4.2 Online (Qwen trace)

Setup:
- Script: `mini-sglang-neuron/benchmark/online/bench_qwen.py:37-52`.
- Workload: first 500 requests of the Alibaba Qwen trace A, filtered to input <= 1024 tokens, replayed at time scales `[0.5, 1, 1.5, 2, 2.5]`.
- `dummy=True` builds every prompt as a **prefix of a single shared prompt** (`MSN/benchmark/client.py:440-443`). Prefix sharing is therefore maximal and synthetic; the trace's `hash_ids` are unused (`client.py:430`).
- Servers: `--max-running-requests 6 --max-prefill-length 256 --max-seq-len-override 2048 --num-pages 16384`. vLLM: `--max-num-seqs 6 --max-num-batched-tokens 256 --block-size 128` (README "Online inference").

Approximate reading of `docs/online_bench.png`:

| Engine | Max throughput | TTFT P90 at that point | TPOT P90 |
|---|---|---|---|
| mini-sglang-neuron, radix | ~367 tok/s | ~235 s | ~15.5 ms |
| mini-sglang-neuron, naive | ~337 tok/s | ~265 s | ~15.5 ms |
| vllm-neuron, no prefix caching | ~367 tok/s | ~235 s | ~16.6 ms |
| vllm-neuron, prefix caching | ~208 tok/s | up to ~500 s | ~46.5 ms |

The radix curve is roughly the same as vLLM without prefix caching, and better than mini-sglang naive at mid load.

Interpretation:
- Everything is capped by 6 concurrent sequences. 6 / 15.5 ms is about 387 tok/s, matching the plateau.
- TTFT of hundreds of seconds is queueing.
- Both engines run on NxDI, so this is a scheduler/config comparison on a 0.6B model, not a kernel comparison.
- Nothing above trn1, TP=2 or 0.6B is benchmarked.

---

## 5. Licenses (not legal advice)

- **mini-sglang: MIT**, "Copyright (c) 2026 sgl-project" (`mini-sglang/LICENSE:1-3`; GitHub reports `MIT`). Forking and borrowing are permitted. Keep the copyright and permission notice in copies and substantial portions.
- **mini-sglang-neuron: ambiguous.**
  - The default branch `dev` has **no LICENSE file**; GitHub `license: null` and `/license` returns 404.
  - `pyproject.toml:10,14` declares `license = {text = "MIT"}` and an MIT classifier, inherited from upstream.
  - The orphan `main` branch (a single "Initial commit", 2026-02-23, no common ancestor with `dev`) holds an unfilled **Apache-2.0** `LICENSE`.
  - The upstream-derived portion stays MIT (sgl-project), and its notice should travel with it. `dev` dropped the LICENSE file, which is a compliance gap on Yotta's side.
  - Yotta's own additions (~1.2k Python lines: `neuron/`, engine rewrite, tool parser, rate limit) have no clear grant.
  - **Safest path**: fork upstream mini-sglang (MIT), re-implement the small NxDI glue, and either ask Yotta to clarify the license or treat their additions as unlicensed. If Apache-2.0 is confirmed, keep its notice and mark changes.
- **NxDI dependency**: the wheel contains an Apache-2.0 `LICENSE` but declares `Classifier: License :: Other/Proprietary License` (`nxdi-0.7.15063/*.dist-info/METADATA`). Review this before redistributing NxDI-derived code.

---

## 6. Gaps: roadmap for a better engine

Neither repo has any of the following; a grep found zero hits for speculative decoding, LoRA, quantization, grammar, PD disaggregation, expert parallelism or DP attention:

| Gap | mini-sglang | mini-sglang-neuron | NxDI 0.7 has a building block? |
|---|---|---|---|
| Speculative decoding | no | no | yes: EAGLE/EAGLE3/fused speculation (`NXDI/models/config.py:227-231`), but "We do not yet support integration of EAGLE3 and prefix caching" (`NXDI/models/model_base.py:1715`) |
| MoE + expert parallelism | TP-only MoE (`MS/layers/moe.py:28-58`) | untested via NxDI | `ep_degree`, `moe_ep_degree`, `moe_tp_degree` (`config.py:338,700-701`) |
| FP8 / quantization (weights, KV) | no | only a `quantized` passthrough (`model_loader.py:220-221`) | `quantized`, `quantization_dtype`, `kv_cache_quant` (`config.py:198-224`) |
| Multi-LoRA | no | no | `lora_config` (`config.py:328`) |
| Constrained decoding (JSON/grammar) | no | no | no (host-side work) |
| PD disaggregation | no | no | no |
| DP attention / data parallel | no | no | `attention_dp_degree` (`config.py:336`) |
| Prefix-cache-aware routing / multi-replica | no | no (one instance per host) | no |
| Overlap / async scheduling | yes | removed (`scheduler.py:321-326`) | `async_mode` (`config.py:177`) |
| On-device sampling | GPU (FlashInfer) | CPU | `on_device_sampling_config` |
| Bucketing / small-shape graphs | n/a (CUDA graphs) | off; max shapes only | 2D prefix buckets (`autobucketing.py`) |
| Native chunked prefill / mixed P+D batch | chunked yes, mixed no | emulated via prefix path; no mixing | `chunked_prefill_config` + `flash_pa_with_schedule` NKI (`NXDI/modules/chunked_prefill/`) |
| NKI attention, QKV and MLP kernels | n/a | all off | CTE flash, block-TKG, qkv, mlp kernels (`config.py:389-415`) |
| Page/block size > 1 | yes | no (`pa_block_size=1`) | yes |
| Backend abort on disconnect | yes | no | n/a |
| Larger models (8B-70B+), TP 8/16/32 | yes on GPU | only 0.6B/4B on trn1 | yes |
| **trn2 / trn3** | n/a | **no mention anywhere** (only trn1 in README and docs) | NxDI handles trn2 (LNC=2, `config.py:452,981`, `attention_base.py:1314-1369`) |
| Current SDK | n/a | pinned 2.27.1 | 2.32.0 is current |
| Context parallel / long context | no | no | `cp_degree` (`config.py:334`) |

Implications for a better engine:
1. Own the model and kernels instead of wrapping NxDI. Use NKI paged/radix attention with block size 32-128 and real 2D bucketing, which removes the "max shapes only, sequential batch-1 prefill" penalty.
2. Run mixed prefill+decode batches and async (overlap) execution.
3. Sample on device.
4. Insert into the radix cache at prefill time, with HBM-aware KV sizing.
5. Add speculative decoding that works with the prefix cache.
6. Use FP8 on trn2 and KV quantization.
7. Support MoE with EP across NeuronLink.
8. Target trn2 first, with LNC-aware TP.
9. Add multi-replica, cache-aware routing and PD disaggregation.
10. Support structured output and multi-LoRA.
11. Build a reproducible benchmark at real concurrency (not 6) on 8B-70B models.

---

## 7. Activity (GitHub API, 2026-10-01)

| | mini-sglang-neuron | mini-sglang |
|---|---|---|
| Created | 2026-02-23 (Neuron work from 2026-02-02 in history) | 2025-09-01 |
| Last commit / push | 2026-04-09 (`2f9ba6ebd`, merge of PR #3) | 2026-05-17 (`9a91cfafe`) |
| Stars / forks | 13 / 2 | 5204 / 885 |
| Open issues / PRs | 0 / 0 (1 closed issue #2 "Tool-support", 2 merged PRs #1 #3) | 12 issues / 40 PRs open |
| Default branch | `dev` (`main` = orphan "Initial commit" with LICENSE) | `main` |
| Contributors on Neuron work | shevateng0 (port), zilong-ai-infra (bench/perf), Yosemite979 (rate limit), matheus-1618 (tool calls) | many |
| trn2 mentions | none (grep: only trn1 / Trainium1 in README and docs) | n/a |

- The blog (Mar 17 2026) credits "Yotta Labs, RadixArk, SGLang". Its only roadmap item is to "deepen the integration between SGLang and the AWS Neuron stack", with no trn2 reference.
- Forks: `matheus-1618/mini-sglang-neuron` was pushed 2026-08-28, but its extra branch `aidlc/teste` is 0 commits ahead of `dev`.
- There is no Neuron work in upstream SGLang beyond a closed "[Feature] Introduce hardware plugin system" issue (#20372, 2026-03-11) found by searching for "trainium".
- **The project has had no commits for about 6 months** (last 2026-04-09) and is not installable at HEAD.
