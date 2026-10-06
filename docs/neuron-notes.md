# Neuron platform notes (measured)

Facts about the Neuron stack that Kiln's code depends on. Each one was measured on real
hardware; the probe that measures it is named so it can be re-run on a new SDK.

## libtorch_neuronx_lite (LNL), Neuron SDK 2.32.0, trn1.2xlarge, 2026-10-02

- **An unequal `torch.split` on the last dim is miscompiled.** `torch.split(x, (2048,
  1024, 1024), dim=-1)` returns wrong values; equal `torch.split`, `chunk`,
  `torch.tensor_split` with indices, plain slicing and `narrow` are all exact. Probe:
  `tools/debug_device.py split_variants`. Found because Qwen3's fused QKV split produced
  garbage (argmax agreement 0.20) while the matmul feeding it was exact (rel 0.003);
  `tools/debug_device.py layer0` bisects a layer stage by stage. Kiln slices explicitly.
- In-place `index_put_` into a graph input followed by a gather from it in the SAME
  graph reads the new values, and the input buffer is updated in place (LNL rewrites the
  mutation as an aliased output). Probe: `write_then_read`.
- A gather with a 2-D index tensor (`cache[idx]`) is exact. Probe: `gather2d`.
- `torch.compile(backend="neuron_libtorch")`: ~94 us per steady-state call of a small
  graph. Qwen3-0.6B (28 layers) compiles in 96-157 s per bucket on trn1.2xlarge.
- Plain `nki.jit` on torch tensors needs `torch_neuronx`, which the SDK 2.32 vLLM venv does
  not ship; inside LNL graphs use `libtorch_neuronx_lite.nki.nki_hop.wrap_nki(k)[lnc](...)`.
- torch-xla in that venv has no Neuron PJRT plugin and silently runs on CPU.
- **The compile cache key hashes the printed FX graph, so equivalent traces that are spelled
  differently compile twice.** Tensor-parallel rank 0 traced `torch.topk(x, 64, dim=-1)`
  while rank 1 traced `torch.topk(..., largest=True, sorted=True, out=None)`, so each rank
  compiled its own copy of every graph (4 keys for 2 graphs at tp=2; at tp=32 that would be
  32 concurrent neuronx-cc runs in host memory). `tools/probe_fx_normalisation.py` isolates
  the trigger: a process that imported transformers and THEN spawned a child records the
  short form; every other process the long form. Kiln spawns TP workers before importing
  transformers and strips default-valued kwargs before handing graphs to LNL; measured
  after the fix: 2 keys, each computed by both ranks, one compile per graph.
- neuronx-cc rejects float8_e4m3fn on trn1/trn2 ("[NCC_EVRF051] ... use the
  --experimental-unsafe-fp8e4m3fn-as-fp8e4m3 flag"); LNL injects that flag for trn2 only.
  Kiln passes it itself for an fp8 KV cache and clamps to 240 on write (AWS's vllm-neuron
  dtype_utils.py: trn2 e4m3 max finite 240, trn3 e4m3fn 448).
- LNL runs neuronx-cc with its defaults (`--framework --target --output --logfile` only);
  vllm-neuron passes `--auto-cast=none -O<level>` and backend options explicitly.
- A sync that replaced the source tree under a running job deleted neuronx-cc's scratch
  directory in the cwd mid-compile (exit 70); jobs now run in /opt/kiln/work.
- **The graph cache key does not include an NKI kernel's source** (SDK 2.32, 2026-10-03).
  `libtorch_neuronx_lite/compile/cache.py` `create_cache_hash` hashes the FX graph text with each
  NKI call's backend config minus the kernel binary's path: what is left is the function name,
  operand names, grid, the static arguments (they print in the graph) and the kernel's MAC count.
  An edit that keeps those (here: one DMA made static in kernels/moe_prefill.py) hit the old graph's
  NEFF: the same cache directory, a 0.7 s "compile" and the old kernel's time (18.16 ms, where the
  edit measured 16.51 ms once forced to recompile). The NKI layer itself did compile the new source
  (a new `.colz` under `compile_cache/nki`), so nothing reports the stale NEFF. moe_prefill passes
  a CRC-32 of its kernel's source text as a static argument (`REV`, `_kernel_rev`), which puts the
  source in the key; the other Kiln kernels do not yet, so after editing one on a host with a warm
  cache, either change something in its signature or delete its cache entries.

## AWS account behaviour (not Neuron, but it looked like Neuron failures)

- **A new instance reboots a few minutes after launch** because the account's SSM patch
  association installs updates at agent registration and reboots (RebootIfNeeded).
  Measured from `aws ssm describe-instance-patch-states` and the box's journal on
  kiln-mimo-trn1: patching 10:56-11:03 UTC, "The system will reboot at 11:04:22" scheduled
  through the SSM agent's unit. **Correction:** the reboot of the first Qwen3-30B-A3B run on
  2026-10-02 (launch 09:35, reboot 09:43) was recorded in commit 44e9f65 and
  bench/results as an out-of-memory caused by eight ranks each holding the full 61 GB
  checkpoint. That was an inference, never an observation (no oom-kill line was seen), and
  the timing matches this patch reboot. Per-rank shard loading stays, since it cuts host
  memory by the tensor-parallel factor either way.
- **The patch association is regional.** `aws ssm list-associations` on 2026-10-03 06:00 UTC:
  us-east-2 and sa-east-1 carry `<aws-account-id>-PatchWeekly` (AWS-RunPatchBaseline, targets
  InstanceIds `*`), ap-southeast-4 carries no association at all. `infra/fleet.sh ready` waits
  for the patch reboot where the association exists and only for SSM where it does not.
- **trn2.3xlarge spot was refused 16 times in 21 minutes** (2026-10-03 05:57-06:19 UTC,
  `infra/fleet.sh --region <r> up kiln-trn2 trn2.3xlarge <az>`, one try per AZ every ~2.5 min):
  sa-east-1c answered every time `InsufficientInstanceCapacity ... We currently do not have
  sufficient trn2.3xlarge capacity in the Availability Zone you requested (sa-east-1c) ... You
  can currently get trn2.3xlarge capacity by not specifying an Availability Zone in your request
  or choosing sa-east-1a, sa-east-1b`, and ap-southeast-4c `InsufficientInstanceCapacity ...
  There is no Spot capacity available that matches your request`. Spot prices in effect at the
  time: sa-east-1c $2.569/h, ap-southeast-4c $2.2687/h, sa-east-1b $7.721/h; sa-east-1a has no
  spot price record since 2026-10-01; placement score 1 in every trn2.3xlarge AZ
  (`get-spot-placement-scores`, read 06:10 UTC). The same request with `KILN_DRY_RUN=1` answers
  `DryRunOperation` in both regions, so the request itself (AMI, profile, security group,
  subnet, `MaxPrice`) is valid; only capacity was missing. No instance or spot request was
  created.
- **A second window, 3 h, also refused every time.** 2026-10-03 06:40-09:35 UTC, 17 rounds ten
  minutes apart over sa-east-1a, sa-east-1c and ap-southeast-4c with `KILN_SPOT_MAX_PRICE=2.60`:
  51 of 51 requests answered `InsufficientInstanceCapacity` (none was a price refusal; sa-east-1a,
  which EC2's earlier message named, refused too). Spot prices at 09:45 UTC: sa-east-1c $2.5902,
  ap-southeast-4c $2.3235, sa-east-1b $7.7189. Total: 67 refusals over 3 h 38 min, nothing
  created.
- **With cost lifted, on-demand was refused too, everywhere, for an hour** (2026-10-03
  14:47-15:42 UTC, about 130 requests): trn2.48xlarge on-demand in us-east-2a/2b/2c and
  ap-south-2a/2b/2c, spot there (us-east-2 spot answers `MaxSpotInstanceCountExceeded`: 136 of 256
  vCPU are the trn1 boxes and the increase to 1024 is `CASE_OPENED`), trn2u.48xlarge on-demand
  (`ReservationCapacityExceeded`) and spot in ap-south-2, trn2.3xlarge on-demand and spot in every
  sa-east-1, ap-southeast-4 and ap-south-2 AZ: every answer `InsufficientInstanceCapacity` (or the
  quota / reservation errors named). ap-south-2 carries no SSM association at all.
- **EC2 Capacity Block for ML bought** (2026-10-03 15:42 UTC, owner's cost decision in GOAL.md):
  `cr-01dd0041d61815bee`, 1 x trn2.48xlarge, ap-south-2b, 2026-10-03T16:13Z to 2026-10-04T11:30Z
  (19 h), **$689.59 upfront** ($36.29/h). It was the earliest start and the shortest block offered
  (`describe-capacity-block-offerings` accepts 24 h multiples and offers the remainder to the next
  11:30 UTC boundary; the next full 24 h started 2026-10-04T11:30Z at $858.26). us-east-2 offered
  no trn2.48xlarge block. Launch with `KILN_MARKET=capacity-block
  KILN_CAPACITY_RESERVATION=cr-01dd0041d61815bee`.
- **The Neuron DLAMI cannot be copied across regions** (`copy-image` of ami-0222021b369f03219:
  "You do not have permission to access the storage of this ami"), and ap-south-2 has no SDK 2.32
  DLAMI. Kiln's own image works: launch the DLAMI once on a small instance, stop it before the
  patch association reaches it, `create-image` (ami-08e6feffd9aa50e81 in us-east-2, 100 GB, 2 min),
  `copy-image` (ami-0cff1ca7a18e21334 in ap-south-2, 21 min); all Project=kiln.

- **trn1.32xlarge has 4 x 1.7 TB of instance-store NVMe, unformatted and unmounted on the
  DLAMI** (2026-10-03, lsblk: "Amazon EC2 NVMe Instance Storage", no filesystem). RAID0 over the
  four with mdadm and ext4 gives 6.6 TB at /opt/kiln/nvme, room for BF16 checkpoints that do not
  fit a 700 GB root volume next to the others (K2-Horizon 758 GB, Inkling-Small 532 GB). It is
  wiped when the instance stops or is replaced and is not in fstab, so keep anything that must
  survive on EBS (the root volume can also grow online: modify-volume, growpart, resize2fs).

## Float-literal comparisons can lower to f64 (2026-10-02, SDK 2.32, trn1.2xlarge)

A bf16 elementwise function written with `c >= 8.0`, `m < 4.0` and `torch.where(..., full_like(m, 6.0))`
failed in neuronx-cc 2.27 with `NCC_ESPP004 f64 dtype is not supported` (the FX graph itself had no
f64 tensor; the python float literals in the comparisons were the only candidates). The same
computation written with multiplies, adds and `torch.floor` only compiled and was bit-exact
(`tools/debug_device.py mxfp4`). Multiplying by a float literal is fine everywhere else in the model.

Seen again 2026-10-05 (compile farm q/amtp-9692e97, neuronx-cc 2.27.5334): all seven shapes of the asynchronous-MTP
board graphs (engine/spec_async.py) failed with NCC_ESPP004. One of them, mtp_post, has nothing else a literal could
enter: an index_select, `valid > 0.5`, slices, ones_like, a cat, a torch.where of two tensors and an index_put_.
Comparing the 0 / 1 flags against the integer 0 and writing `torch.where(c, x, torch.zeros_like(x))` instead of a
float scalar compiled all seven. The sampler's `temperature <= 0` and `g > 0` have always compiled. So a comparison
against an integer literal is fine, and the float literal is what lowers to f64.

## Integer division by a constant is inexact in a graph (2026-10-03, SDK 2.32, trn1.32xlarge)

`x // c` on int64 graph values is computed through an fp32 reciprocal: `tools/probe_gqa_moe.py
intdiv` compiles `starts // 6284` for `starts = arange(32) * 6284` and gets `r - 1` instead of `r`
for ranks 1-4, 6-8, 12-16 and 23-31, exactly the r where `floor(fp32(r * 6284) * fp32(1 / 6284)) <
r`; `// 4768` is exact for all 32, and so is `arange(32) * c == start` for both. Found because
`DecoderForCausalLM._head` located its vocabulary shard as `vocab_start // rows`: gpt-oss-120b
(201088 / 32 = 6284 rows per rank) at tp=32 returned its top tokens one shard low (12213 -> 5929,
7698 -> 1414, 13010 -> 6726) in `tools/check_truncated.py --layers 2`, while MiMo (4768 rows) never
showed it, the same engine on host processes (`--device cpu --tp 32`) matched the host 19/20 and
tp=1 on the device matched 20/20. After `arange(tp) * rows == vocab_start`: 19/20 overlap, top-1
equal, max |dlogprob| 0.076 against both the host and transformers' truncated model. Do not divide
in a graph; multiply and compare, or pass the quotient in. (Powers of two divide exactly, so the page
arithmetic `pos // page_size` is safe.)

## A big expert-gather prefill graph compiles, then cannot load (2026-10-03, SDK 2.32, trn1.32xlarge)

gpt-oss-120b, FP8 experts, one prefill chunk of C=32 tokens x top-4 = 128 (token, expert) pairs
through the XLA gather path, one MoE layer per graph (`tools/check_truncated.py --tp 8 --piecewise`
and `tools/check_device.py --tp 16 --piecewise`): at tp=8 (384 expert intermediate units per rank)
the graph compiled for 29 minutes and at tp=16 (192) for 27, and both then failed to load on every
rank with `TDRV:vring_allocate_next_desc Descriptor limit reached. Descriptors in vring 16777200, Max
allowed 16777200` on ring `qSPSpillReload0_0` (`Failed to stage graph to NeuronCore`). At tp=32 (96
units) the same graph compiled in 46 s and runs. At tp=1 the C=32 graph failed in the compiler
instead (`NCC_EBVF030 Instructions generated by compiler 18505833 exceeds the typical limit of
5000000`). With `KILN_MOE_GATHER_MAX_PAIRS=64` the C=32 chunk takes the every-expert path and tp=16
loads (compile 66 s). The same spill-reload ring is what the MoE gather notes above measure growing;
here it overflows the runtime's descriptor limit outright.

## Killing rank 0 leaves the other ranks running; a full page cache starves the runtime's DMA rings (2026-10-03, trn1.32xlarge)

`pkill -f tools/check_ppl.py` stopped a tp=32 job's rank 0 only: its 31 ranks run as `python -c from
multiprocessing.spawn import spawn_main ...` and kept loading (300% CPU and about 209 GB of mapped
checkpoint pages each, K2-Horizon from the instance-store RAID). The next 32-rank job then failed
at runtime start with `TDRV:dmem_alloc_internal Failed to allocate HOST memory (4194304 bytes)`
(`dma rings runtime`). Killing the spawned processes whose environment carries the job's
PYTHONPATH (`/proc/<pid>/environ`) was not enough on its own: the retry failed the same way, with
460 GB of the host in clean page cache (the checkpoint just read from the instance-store RAID) and
almost no free 4 MB (order-10) blocks in `/proc/buddyinfo`. `/proc/sys/vm/drop_caches` is not
writable from the SSM shell; `posix_fadvise(POSIX_FADV_DONTNEED)` over every file in the HF caches
freed 234 GB and thousands of order-10 blocks, and the same job then started. Kill the whole job,
not just the command line that launched it, and evict a big checkpoint's page cache before the
next 32-rank start.

## vllm-neuron 0.24's gpt-oss does not compile on NeuronCore-v2 (2026-10-03, SDK 2.32, trn1.32xlarge)

`bench/vllm_neuron_decode.py --model <gpt-oss-20b> --tp 8 --batch 4` launches the model the way the
SDK's gpt-oss recipe does: `hf_overrides={"quantization_config": {}}`, the hybrid KV cache manager
on, `NEURON_SKIP_EFA_AFFINITY=1` and prefix caching off. On trn1 it reaches vllm-neuron's BF16
gpt-oss, which the factory selects on every platform but trn3. It then fails while tracing the first
decode graph, in the NKI attention-mask kernel
(`vllm_neuron/functional/attention/attention_decode_mask.py`):
`tensor_copy does not support engine.scalar on NeuronCore-v2 (Trn1/Inf2). Use nisa.activation with
op=nl.copy instead.` The recipe (`docs/model-recipes/gpt-oss.md` in the venv) lists Trn2 and Trn3
only. So on trn1 there is no vllm-neuron gpt-oss baseline, and a kernel written for NeuronCore-v3 can
fail on v2 at trace time, not only at run time.

## HBM per NeuronCore and what a NEFF reserves (2026-10-02, SDK 2.32, trn1.32xlarge)

trn1 has 16 GB of HBM per NeuronCore. Loading the first whole-model graph of MiMo-V2.6-Flash
at tp=32 (FP8 experts at the time) failed with `Failed to allocate 54.676MB (usage: dma rings
spill)`; the runtime's table at the failure: tensors 13.684 GB (weights + KV), shared
scratchpad 448 MB, DMA rings spill 1.789 GB, total 15.915 GB. The spill-ring reservation is
per loaded NEFF and grows with the graph, so a big model needs headroom for every graph it
loads, not only for weights and KV.

## Execution queue (2026-10-02, SDK 2.32, trn1.2xlarge)

`NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS` above 63 is clamped with a warning ("Setting ... > 63
is not supported"). Light graphs never fill the queue (200 chained launches of a 256x256
matmul chain succeed); heavy ones do (whole-model decode graphs at 16, Qwen3-0.6B prefill layer
graphs at about 30). `torch.neuron` exposes only `current_device` and `is_available`, so the
way to wait is to read an output back.

## Compile size of the MoE gather path (2026-10-02, SDK 2.32, trn1.32xlarge)

A 6-layer MiMo-V2.6-Flash prefill graph (32 tokens x top-8 = 256 expert pairs, each gathering
and decoding its own packed MXFP4 weights) lowered to 1,805,906 backend instructions and was
still in walrus after 18 minutes (24 GB RSS); the whole-model FP8 graph had taken 2270 s.
`tools/profile_moe.py` measures each MoE formulation in isolation.

## A cached all-gather NEFF breaks in the next process (2026-10-03, SDK 2.32, trn1.32xlarge, tp=32)

Graphs containing `_c10d_functional.all_gather_into_tensor` work in the process that compiled them
(the other 31 ranks wait for the compile and load the same NEFF), but a LATER process that loads
the same NEFF from the compile cache fails on every rank: `ENC:get_nccl_comm replica group
signature mismatch for group 0: likely caused by mismatched collectives between peers`, then
`CCOM WARN Failed to RX` and `NRT_NETWORK_PROXY_FAILURE`, and the run hangs. Moving the cached
graphs that contain an all-gather aside made the identical run work again. All-reduce graphs
reload across processes without trouble. Kiln therefore gathers vocab-parallel logits with an
all-reduce of zero-padded shards (`DecoderForCausalLM._head`).

A run that hits this leaves the runtime's collectives wedged for later runs on the same instance;
a reboot of the instance cleared it (disk and compile cache survive).

## MoE expert gathers: cost is the software-DGE packet count, and graph context sets it (2026-10-03, SDK 2.32, trn1.2xlarge)

**MiMo-V2.6-Flash decode was 10x slow because each piecewise graph held 6 MoE layers.** One
rank's layers were rebuilt at tp=32 shapes with random weights in the keep_fp8 layout
(`tools/profile_layer.py`; FP8 qkv [704, 4096] with per-row block-128 fp32 scales, bf16 o_proj,
FP8 experts with bf16 block-32 scales, w_down stored [E, Im, H]) and run through the real code
(group_fn, _layer, the runner's `_piecewise`). With two processes on the two NeuronCores, each
holding rank-0 shapes and all-reducing over a group of 2 (`--ranks 2`), the 6-layer decode
graphs took 165.7 ms (layers 0-5) and 56.0 ms (6-11) against 175 and 59 ms measured per graph
on trn1.32xlarge at tp=32, so the reproduction holds and the collectives are not the cost: 12
in-graph all-reduces of the [4, 4096] bf16 hidden state add 0.31 ms (about 26 us each,
`--allreduce`).

One graph per layer, same layers and inputs (decode B=4, P=16, `--what layers`, 2 ranks):
dense layer 0 0.66 ms, sliding-window MoE 1.30 ms, full-attention MoE 1.45 ms. Every attention
piece alone (single rank) is within 0.2 ms of the 0.14 ms launch-and-readback floor (qkv dequant +
matmul 0.22, page or token gather 0.19-0.33, attention with sinks 0.18, o_proj 0.19, KV
index_put_ 0.19 ms), so none of the suspects in the attention path (qkv per-row scale dequant,
the token-slot gather, index_put_, the 1-KV-head einsum, fp32 softmax) matters at this shape.

Why: `neuron-explorer capture` / `view --output-format summary-text` on the single-rank NEFFs. trn1 has no hardware descriptor generation
(`hardware_dynamic_dma_size_percent` 0), so every dynamic (gathered) DMA packet is generated in
software, and the time follows the packet count, which neuronx-cc 2.27 picks per graph:

| graph (MiMo tp=32 shapes) | dynamic DMA packets | avg bytes | spill reload | time |
|---|---|---|---|---|
| 1 MoE layer, decode B=4 | 19,312 | 7,083 | 2.2 MB | 1.6 ms |
| 1 MoE layer, decode B=2 | 304,832 | 181 | 2.2 MB | 15.6 ms |
| 6 MoE layers, decode B=4 | 6,415,488 | 34 | 1.27 GB | 225 ms |

`tools/probe_lowering.py` chains blocks in one graph and reads the compiler's "Unrolled DGE
count with Dynamic AP" from the log: attention-only chains and MoE-only chains stay linear and
cheap, but attention + MoE layers explode (decode B=4, single rank: 1 layer 125 dynamic-AP DMAs
/ 1.5 ms, 2 layers 16,624 / 39.6 ms, 3 layers 49,499 / 100 ms, 6 layers 104,072 / 225 ms). In
the bad graphs the compiler log shows the expert SCALE gather lowered one element per pair per
DMA (`bfloat16<32 x 1>`, est. 12 ms) and the dequantized gate_up block (32 pairs x 128 x 4096)
pf-transposed, spilled and reloaded as 4,096 sliding [128 x 3969] windows (4.16 GB, est. 36 ms).
None of these changed it (2 or 6 layers still bad): no KV write, KV read by a static slice,
experts from a fixed index input, bf16 instead of f32 dequant, gate_up stored input-dim first
(2 layers 7.0 ms, 6 layers 82.7 ms), gathering flattened expert rows, `-O1`,
`--model-type=transformer`. Selecting the scales by an exact one-hot matmul (no scale gather)
changed nothing with real all-reduces in the graph at B=1, 2, 4 or 8 (P=16) and made a
single-rank B=4 layer 4x slower.

The fix: `model_runner.piecewise_groups` gives every MoE layer a graph of its own (dense layers
still group). The runner's full decode step, 48 layers, 2 live ranks, `--what step
--legacy-groups 6`: at the production bucket B=4, P=4, 550.4 ms with 6 layers per graph (the
trn1.32xlarge tp=32 run measured 571 ms) against **55.3 ms** with MoE layers alone (9.95x); B=4,
P=16 554.0 against 57.3 ms; the default group of 4 was 680.1 ms. 12 layers: B=8, P=64 290.6
against 22.1 ms; **B=1, P=4 45.5 against 50.7 ms (10% slower)**. Compiles also shrink (a
single-layer graph 5-9 s, a 6-layer one 52-93 s).

Still slow, same mechanism, single-layer graphs with real all-reduces: a MoE layer is 4.6 ms at
B=1 and 8.9 ms at B=2 (against 1.3 ms at B=4 and 2.0 at B=8), and 42.8 ms at prefill C=32
(256 pairs; the dense every-expert path is 233 ms). The packetization of gathered weights cannot
be steered from the XLA level; an NKI kernel that DMAs whole expert blocks (512 KB of w_gu,
256 KB of w_down per pair) is the robust fix.

Practical: a 6-layer prefill group graph at C=32 compiled for over 10 minutes and starved the
SSM agent of kiln-dev-trn1 (trn1.2xlarge, 32 GB) for an hour; LNL's other rank gave up with
"Shared compilation timeout after 657.2s". Do not compile multi-layer MoE prefill graphs on a
trn1.2xlarge.

## Collectives on trn2: the same shape, about half the cost (2026-10-03, SDK 2.32, trn2.48xlarge, LNC=2)

`KILN_PROBE_COLLECTIVES=1 python tools/profile_layer.py --allreduce --ranks N --batch 4` (graphs of
the [4, 4096] bf16 hidden state; one logical NeuronCore per rank, so 4 ranks = one Trainium2 chip;
"chained" = 48 launches back to back waiting 6 behind, per launch):

| ranks (chips) | null graph | 1 all-reduce | 12 all-reduces | all groups of 2 at once | all groups of 8 at once | trn1 1 all-reduce |
|---|---|---|---|---|---|---|
| 2 (half a chip) | 0.086 ms | 0.150 ms | 0.325 ms | - | - | - |
| 4 (1) | 0.154 | 1.040 | 1.023 | 0.213 | - | - |
| 8 (2) | 0.086 | 1.486 | 1.484 | 0.165 | - | 2.45 |
| 16 (4) | 0.145 | 1.982 | 2.030 | 0.145 | 1.559 | 3.76 |
| 32 (8) | 0.132 | 2.541 | 2.515 | 0.157 | 1.334 | 5.10 |
| 64 (16) | 0.108 | 2.973 | 3.214 | 0.161 | 1.419 | - |

Synchronous p50 (launch + run + readback) for 1 all-reduce: 0.31 / 1.35 / 1.82 / 2.19 / 2.84 / 3.21
ms at 2 / 4 / 8 / 16 / 32 / 64 ranks. All-gather, reduce-scatter and all-to-all cost the same as
an all-reduce at each size (within 0.2 ms). Every group sum checked (max |err| 1.6e-2 for groups
of 2, 3.1e-2 for groups of 8, bf16).

So the trn1 rule holds on trn2 with smaller constants: a graph that holds a world collective costs
a fixed 1-3 ms per execution whatever it holds, further collectives in the same graph are almost
free, and it grows slowly with the chips crossed (2.5 ms at 32 ranks against 5.1 on trn1, 3.0 at
64). New on trn2: even ONE chip's 4 logical cores pay 1.0 ms, while a pair of ranks (
two logical cores of one chip) pays 0.15 ms, and 32 groups of 2 reducing at once cost the same
0.16 ms. A decode step is still bounded below by (graphs holding a cross-core collective) x 1-3 ms.

## MoE expert gathers on trn2: the XLA path is worse than on trn1, the NKI kernel is required (2026-10-03, SDK 2.32, trn2.48xlarge, LNC=2)

Hardware DGE on NeuronCore-v3 does not rescue the in-graph XLA expert gather. MiMo-V2.6-Flash
rank-0 shapes at tp=32 (`tools/profile_layer.py --model XiaomiMiMo/MiMo-V2.6-Flash-RL --tp 32
--batch 4 --pages 4`, decode B=4, P=4, random weights in the checkpoint's layout):

| what | XLA MoE (default) | `KILN_MOE_KERNEL=nki` (kiln/kernels/moe_dedupe.py, grid 2) | trn1, XLA |
|---|---|---|---|
| MoE block alone, one rank (`--ranks 1 --layers 2 --what mlp`) | 20.637 ms | **0.284 ms** | 1.3-1.6 ms per MoE layer |
| 12-layer decode step, 2 live ranks, one graph per MoE layer (`--ranks 2 --layers 12 --what step`) | 130.4 ms | 8.5 ms | - |
| same, 6 layers per graph (`--legacy-groups 6`) | 422.3 ms | - | packetization blow-up (see below) |
| same, all 12 MoE layers in one graph (`--moe-groups 12`) | - | **6.674 ms** | - |

(The null graph is 0.13-0.14 ms in every row.) nkilib's `moe_tkg` does run on trn2 (on trn1 it
does not trace): `tools/probe_nkilib_moe.py --batch 4 --dtype fp8` (selective experts, FP8 ROW scales,
E=256, H=4096, I=64, top-8, grid 2) is correct (max abs error 1.2e-3, rel 0.0074, against fp32)
and takes **0.359 ms** per synchronous call, against Kiln's 0.284 ms at the same shape. It refuses
bf16 affinities in selective mode ("'nisa.tensor_scalar_arith' op 'operand0' must be float32, got
'bf16'", selective_expert_impl.py:348); fp32 works. Its FP8 scales are per output channel, not
Kiln's per 32 input columns, so it is a reference, not a drop-in.

Consequence for trn2: MoE models run with `KILN_MOE_KERNEL=nki` and MoE layers grouped
(`KILN_PIECEWISE_MOE_GROUP`), never on the XLA expert path.

## Where a GLM-5.3-Flash step goes at 8K context (2026-10-03, SDK 2.32, trn2.48xlarge, tp=32)

`KILN_PROFILE_PIECES=1 KILN_PROFILE_EXEC=1 KILN_MOE_KERNEL=nki KILN_PIECEWISE_MOE_GROUP=4
bench/serve_sweep.py --model zai-org/GLM-5.3-Flash --tp 32 --piecewise --overlap --concurrency 8
--requests 8 --input-len 8192 --output-len 256 --decode-buckets 8 --page-buckets 264 --prefill-tokens
256 --prefill-buckets 256 --kv-cache-gb 8` (every graph timed synchronously): 12 layer-group graphs
per call (45 layers, MoE groups of 4: three KDA + one DSA layer each).

| call | per layer-group graph p50 | whole call p50 |
|---|---|---|
| decode B=8, 264 pages | 8.66-9.02 ms (last 3.90) | 112.2 ms |
| prefill C=256, 264 pages | 41.8-43.6 ms (first 32.4, last 16.0) | 509.9 ms |

The pieces of one 4-layer group, each alone on one rank (no real collective):
- MoE block (experts + shared expert), `tools/profile_layer.py --what mlp --tp 32 --ranks 1 --pages 264`:
  3.565 ms at T=256, 0.445 ms at T=8 (kiln/kernels/moe_dedupe.py).
- KDA layer (2 heads of 128 per rank), `tools/profile_linear_attn.py --kind kda --heads 2 --dim 128
  --hidden 4096 --lower-bound -5 --chunk 256`: 1.346 ms per 256-token chunk, 0.159 ms per decode call
  (chained); relative error per token vs fp32 host max 0.31, mean 0.044.
- DSA indexer attention, `tools/profile_mla.py --model zai-org/GLM-5.3-Flash --tp 32 --layers 4
  --prefill 256 --pages 264`: 6.11 ms at C=256 over 8,448 keys (mask and gather modes alike); a plain
  `torch.topk(2048)` over [256, 8448] alone would be 14.2 ms.

Sum for one prefill group: 3 x (1.35 + 3.57) + 6.11 + 3.57 = 24.4 ms against 42 ms measured, so about
18 ms per group is outside the attention and the experts: the group's collectives (one fixed cost per
graph, 2.5 ms at 32 ranks) and, the likeliest remainder, GLM's hyper-connections (4 streams of 4,096,
Sinkhorn normalisation with 20 iterations per layer) and norms. Not measured separately yet.
Prefill runs at about 500 tokens/s per tp=32 engine; the p5en reference processed about 76K prompt
tokens/s at concurrency 128.

The remainder is the hyper-connections. `tools/probe_mhc.py --tokens 256` (hybrid._mhc and _block's
output combination on one core, GLM's shapes): 2.16 ms against 0.87 ms for a stand-in graph of the
same input and output, i.e. about 1.3 ms per block, two blocks per layer, about 10 ms per 4-layer
group; with it the group adds up to 38 ms of the 42 measured (MoE 37%, hyper-connections 27%, DSA 16%,
KDA 11%, collectives 9%). The probe also put all of that 1.3 ms on `comb^T @ S` (a batched matmul
with K = hc = 4: 2.18 ms alone, against 0.87 ms as four fp32 multiply-adds with the same error), but
**that rewrite made the real step slower**: prefill 509.9 -> 610.1 ms per call, groups 42 -> 51 ms,
decode unchanged (same profiled sweep). In the whole graph the fp32 [256, 4, 4096] intermediates cost
more than the small matmul did; the isolated timing was dominated by its 8 MB read-back floor. Reverted.
Judge a rewrite inside the step graph, never by an isolated probe alone.

## neuronx-cc internal error on GLM-5.3-Flash's 512-token prefill group (2026-10-03, SDK 2.32, neuronx-cc 2.27.5334.0, trn2)

GLM-5.3-Flash at tp=32, `--piecewise`, `KILN_MOE_KERNEL=nki KILN_PIECEWISE_MOE_GROUP=12`: the
prefill graph of a 12-MoE-layer group at C=512 tokens and 264 pages (8,448 tokens of context; 351
inputs, 45 NKI kernel calls, 24 all-reduces) fails in neuronx-cc after about 4 minutes with
`[INTERNAL_ERROR] [NCC_INAS001] Error namespace neuronxcc does not exist or error code IGAA901 does
not exist`, exit 70; the cause in its log is `Assertion failed: False` raised from
`neuronxcc/starfish/penguin/transforms/LoopFusion.py` inside `parallelCompileSubGraphs` (subgraph
sg0002, after NeuronLoopFusion). The same graph's HLO (`graph.hlo` in the LNL compile-cache entry)
recompiled by hand fails identically with `-O1`, without `--modular-flow-mac-threshold=10`, with
LNL's plain arguments (`--logical-nc-config=2` and the FP8 option only) and with those plus `-O1`
(about 230 s each, run in parallel on the host): the flags are not the trigger. The decode graphs
of the same groups (B=8, 264 pages) compile (202-294 s each), and prefill groups at C=32 and 4
pages compiled on trn2 and trn1.

## Collectives across chips: a fixed cost per graph EXECUTION, not per collective (2026-10-03, SDK 2.32, trn1.32xlarge)

MiMo-V2.6-Flash decode at tp=32 after one graph per MoE layer: `KILN_PROFILE_PIECES=1` shows
48 layer graphs at 5.8-6.5 ms each (B=4) and 46 ms each at prefill C=32, i.e. the whole
306 ms step, against 1.3 ms per MoE layer with two live ranks on one trn1 chip.
`tools/profile_layer.py --allreduce --ranks N` (graphs of the [4, 4096] bf16 hidden state;
"chained" = 48 launches back to back waiting 6 behind, as the runner issues a step):

| ranks | null graph | 1 all-reduce, chained | 12 all-reduces, chained | 12 adds, chained |
|---|---|---|---|---|
| 2 (one chip, trn1.2xlarge) | 0.14 ms | | +0.31 ms for 12 (26 us each) | |
| 8 | 0.11 ms | 2.45 ms | 1.75 ms | 0.11 ms |
| 16 | 0.10 ms | 3.76 ms | 3.87 ms | 0.11 ms |
| 32 | 0.13 ms | 5.10 ms | 5.02 ms | 0.12 ms |
| 32, `NEURON_RT_DBG_CC_NOP=1` | 0.11 ms | 0.14 ms | 0.17 ms | 0.13 ms |

So any graph that holds a cross-chip collective costs a fixed few ms per execution (rising with
the number of chips), and further collectives in the same graph are almost free. Chaining
launches does not hide it (synchronous p50 6.9 ms against 5.1 ms chained at 32 ranks). None of
`NEURON_RT_DBG_MESH_CC=1`, `NEURON_RT_DBG_RDH_CC=1`, `NEURON_RT_DBG_KANGARING_CC=1`,
`NEURON_RT_DBG_HYBRID_RING_CC_EN=1`, `NEURON_RT_DBG_HIERARCHICAL_CC=0`,
`NEURON_RT_DBG_CC_CHECK_SIGS=0`, `NEURON_RT_DBG_ENFORCE_MESH_GLOBAL_HANDSHAKE=0`,
`NEURON_RT_RANKS_PER_NETWORK_PROXY=1` or `NEURON_RT_DBG_CC_STREAM_MODE=1` changed it (4.4-6.0
ms chained); `NEURON_RT_RANKS_PER_NETWORK_PROXY=32` fails runtime init and
`NEURON_RT_ONE_THREAD_PER_CORE=1` fails to schedule. 4 ranks failed with "Failed to schedule
neff execution". The collective TYPE does not matter either (`KILN_PROBE_COLLECTIVES=1`, 32
ranks, chained): all-gather + local sum 4.85 ms, reduce-scatter 4.97, all-to-all 5.00. What
matters is the chips crossed: inside the same 32-rank world an all-reduce over ranks 0-1 (one
chip) is 0.15 ms and over ranks 0-7 (4 chips) 2.55 ms. Consequence: on multi-chip
TP the decode step must be FEW graphs (NxDI compiles one per step); one graph per layer costs
48 x 5 ms here, which is the gap to the 55 ms that two ranks measure.
(2026-10-05: the fixed cost is the runtime's per-execution barrier, which none of the knobs above touches;
`NEURON_RT_ENABLE_HW_EXECUTION_BARRIER=1` halves it and is now the trn1 default, and turning the barrier off is unsafe.
"Upstream harvest (2026-10)" at the end of this file.)

## Decode MoE as one NKI kernel: MoE layers can share graphs again (2026-10-03, SDK 2.32, trn1.2xlarge)

`kiln/kernels/moe_decode.py`, behind `KILN_MOE_KERNEL=nki` (EngineConfig.moe_kernel, default
`xla`). The selected-expert path of `DecoderForCausalLM._moe` (T x k <= 512 pairs) runs as one
NKI kernel inside the LNL graph (`wrap_nki(kernel)[1](...)`), and the loader repacks each MoE
layer's experts into one `w_blob` [E, 128, 6528] uint8: per expert, one 6528-byte row per
partition holding the fp8 gate_up tile (input dim on partitions), the fp8 down weight (its two
H halves stacked on partitions 0-63 / 64-127) and both bf16 scale sets, the same 816 KB as the
four tensors it replaces. Each (token, expert) pair is ONE dynamic DMA of that row block, so its
descriptor count is 128 whatever graph holds the kernel. On the device a blob layer always runs the
kernel, in chunks of at most 512 pairs (`KILN_MOE_KERNEL_MAX_PAIRS`; the kernel unrolls its pair
loop, so compile time follows the pairs per call). The fp8 weights are the matmul
stationary operand as stored (gen2 takes fp8 x bf16), the moving operand is x block-diagonal
over the four 32-row scale blocks of a tile, so PSUM holds one partial dot product per scale
block and the block-32 scales are applied after the matmul: nothing is dequantized. The XLA
gather and dense paths read a blob layer back through `unpack_gu` / `unpack_down` on the CPU only:
on the device LNL rejects the in-graph uint8 -> float8 view they need ("RuntimeError: Expected XLA
tensor. Got: XLAFloat8_e4m3fnType", measured on the dense path at prefill C=128), which is why the
device always takes the kernel.

nkilib's `moe_tkg` (nkilib/core/moe/moe_tkg, SDK 2.32) does not run on trn1.
`tools/probe_nkilib_moe.py` calls it through wrap_nki in selective-expert mode at the same rank
shapes, with FP8 ROW weights and with bf16 weights; both fail while the kernel is traced, in
`mlp_tkg_gate_up_projection_lhs_rhs_swap.py:133` (a `dma_copy` with `dge_mode=hwdge`):
"assertion failed: dge_mode.hwdge is only supported for NeuronCore-v3 or newer, but current target
is ... gen2" (nki/isa/_validation.py:2473). Its layouts would not fit either: ROW quantization is one
scale per output channel ([E, 2, I] gate/up, [E, H] down), STATIC one per expert, and MX weights
assert gen4 ("MX weights are only supported on gen4+ (Trn3+)", `_validate_moe_tkg_inputs`), while
Kiln's experts carry a bf16 scale per 32 INPUT columns (MXFP4-derived), which no per-output-channel
scale represents; and bf16 experts do not fit (18.9 GB of experts per rank at tp=32 against 16 GB
of HBM per NeuronCore).

Checked without the device (`tests/test_moe_kernel.py`): `pack` is an exact byte permutation;
`emulate` (the kernel's arithmetic in torch) is within 1% of max of an fp32 reference; the XLA
paths on a blob layer equal the natural layout bit for bit; load_model with `moe_kernel="nki"` on
a quantized MiMo-V2 checkpoint (hidden 256, 2 x 64 gate/up rows) matches the Hugging Face reference
within 1e-4; and the NKI CPU simulator (`nki.simulate`, nki 0.6.0, target trn1, run on
kiln-dev-trn1's host) matches `emulate` exactly at T=1 and within one bf16 ulp at T=3.

**On the device, the MoE alone** (`python tools/probe_moe_kernel.py --batch ...`, one NeuronCore,
random FP8 experts at MiMo-V2.6-Flash tp=32 rank shapes: 256 experts, hidden 4096, 128 gate/up and
64 down rows, top-8; p50 of synchronous calls, null graph 0.137 ms; max abs error relative to the
fp32 host reference's max):

| B | pairs | kernel ms | XLA gather ms | kernel vs fp32 | XLA vs fp32 | kernel vs XLA | kernel graph first call |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 0.207 | 4.309 | 0.0040 | 0.0057 | 0.0072 | 2.6 s |
| 2 | 16 | 0.241 | 8.431 | 0.0042 | 0.0061 | 0.0080 | 2.9 s |
| 4 | 32 | 0.298 | 2.403 | 0.0044 | 0.0047 | 0.0043 | 3.4 s |
| 8 | 64 | 0.422 | 5.114 | 0.0052 | 0.0052 | 0.0053 | 4.6 s |
| 16 | 128 | 0.672 | 22.778 | 0.0052 | 0.0052 | 0.0053 | 7.2 s |
| 32 | 256 | 1.191 | 54.210 | 0.0046 | 0.0054 | 0.0056 | 13.0 s |
| 64 | 512 | 2.204 | 76.314 | 0.0051 | 0.0071 | 0.0061 | 24.0 s |
| 128 | 1024 (2 calls) | 4.141 | not run | 0.0043 | | | 15.6 s |

The kernel costs about 4 us per pair, linear in B; the XLA gather is erratic across buckets, as
the packet counts above predicted. A hardware profile of the kernel graph (`tools/dma_counts.py`)
at B=32: 37,088 software dynamic DMA packets carrying 214 MB (256 pairs x 816 KB + 4,320 small
ones), dynamic DMA active 0.91 ms of the 1.18 ms call, tensor engine active 0.48 ms, vector 0.41
ms: at B=32 the kernel reads about the whole expert set and is bound by that DMA (235 GB/s).

**One layer, two live ranks** (`KILN_MOE_KERNEL=nki python tools/profile_layer.py --ranks 2
--layers 6 --batch B --what layers`, P=16; the XLA rows at B=1..8 are the ones measured above,
B=16 / 32 measured now with the same command without the knob):

| B | MoE swa, kernel | MoE full, kernel | MoE swa / full, XLA |
|---|---|---|---|
| 1 | 0.429 ms | 0.445 ms | 4.6 ms |
| 2 | 0.499 | 0.565 | 8.9 |
| 4 | 0.600 | 0.745 | 1.30 / 1.45 |
| 8 | 0.812 | 1.066 | 2.0 |
| 16 | 1.225 | 1.752 | 34.6 / 35.2 |
| 32 | 2.021 | 1.965 | 43.3 / 43.2 |

Prefill, one MoE layer, 2 ranks (`--prefill C --what layers`): C=32 (256 pairs) 1.60 ms (swa) and
1.70 ms (full) against 42.8 ms on the XLA gather; C=128 (1024 pairs, two kernel calls) 4.57 / 4.59
ms, where the XLA dense path (above 512 pairs) did not compile on this host at all: walrus_driver
was killed by the kernel's OOM killer at 27 GB RSS. Multi-layer prefill graphs were not compiled
(see the warning above about the trn1.2xlarge host).

**Grouped graphs stay linear.** Single rank, B=4, P=16 (`--ranks 1 --layers 12 --what layers
groups --group G`): MoE layer alone 0.52 (swa) / 0.65 ms (full); 2 layers [ww] 0.81 ms, [wF] 0.97;
6 layers 2.17-2.25 ms; 12 layers 4.19 ms (chained 3.79 ms per call), against 1.5 / 39.6 / 225 ms for
1 / 2 / 6 layers on the XLA path. Hardware profile of those single-rank NEFFs
(`python tools/dma_counts.py --since 12`; times are the profiler's, which runs slower than p50):

| graph (B=4, single rank, kernel) | software dynamic DMA packets | dynamic bytes | spill reload |
|---|---|---|---|
| dense layer 0 alone | 8,288 | 13.8 MB | 0.29 MB |
| 1 MoE layer (swa / full) | 8,784 / 11,632 | 34.4 / 35.3 MB | 0.35 / 0.34 MB |
| 2 layers [Dw] / [ww] / [wF] | 16,640 / 17,152 / 20,064 | 47-70 MB | 0.56-0.58 MB |
| 6 layers [DwwwwF] / [wwwwwF] | 52,768 / 53,408 | 186 / 207 MB | 1.7 MB |
| 12 layers | 105,568 | 393 MB | 3.4 MB |

against 19,312 (1 layer) and 6,415,488 packets with 1.27 GB of spill reload (6 layers) on the XLA
path: about 4,100 of each MoE layer's packets are the kernel's (32 pairs x 128 rows), the rest the
attention's KV gathers, and nothing grows with the graph.

**The decode step**, 48 layers, two live ranks, P=4 (`KILN_MOE_KERNEL=nki python
tools/profile_layer.py --ranks 2 --layers 48 --batch B --pages 4 --what step --step-groups 1
--moe-groups 2 6 12`; `--moe-groups` runs `model_runner._piecewise` with `moe_group`, i.e.
EngineConfig.piecewise_moe_group / KILN_PIECEWISE_MOE_GROUP):

| B | 1 layer per graph (48 graphs) | 2 (24) | 6 (8) | 12 (4) | XLA, MoE layers alone | XLA, 6 per graph |
|---|---|---|---|---|---|---|
| 1 | 16.6 ms | | 10.0 | 9.7 | | |
| 4 | 23.1 | 21.1 | 17.1 | 16.9 | 55.3 | 550.4 |
| 16 | 49.3 | | 41.5 | 41.3 | | |
| 32 | 82.0 | | 68.4 | 66.7 | | |

(First call, i.e. compile and load, of the 12-per-graph step: 57 s at B=1, 74 s at B=4, 104 s at
B=16, 147 s at B=32.) So the grouping that the XLA gather made 10x slower is now the fastest
choice, and at tp=32, where every graph holding a collective costs about 5 ms per execution (section
above), 12 layers per graph is 4 layer graphs per step instead of 48. That last step is a
projection from the two measurements, not a trn1.32xlarge measurement: the kernel has not run at
32 ranks yet.

Not done here: an all-expert NKI mode for large prefill chunks (the selected-expert kernel reads
816 KB per pair, so a 512-token chunk reads 3.4 GB per MoE layer where every expert once is 204
MB), and the per-pair cost at small B (B=1 is 70 us for 8 pairs over the null graph; at B=32 the
kernel is DMA-bound, and 256 pairs of 256 experts hold about 160 distinct experts that are each
read once per pair). The next section does the latter (one load per distinct expert), which also
covers prefill chunks.

## MoE that reads each selected expert once per call (2026-10-03, SDK 2.32, nki 0.6.0, trn1.2xlarge)

`kiln/kernels/moe_dedupe.py`, now what `KILN_MOE_KERNEL=nki` runs (also `nki-dedupe`;
`KILN_MOE_KERNEL=nki-pair` keeps the per-pair kernel of the section above; decoder.NKI_MOE_KERNELS). The per-pair kernel loads one expert blob per
(token, expert) pair, so at T tokens x top-8 of 256 its HBM traffic follows the pairs: T=32 reads 256
blobs for about 163 distinct experts, prefill C=128 1024 for about 255. This kernel groups a call's
pairs by expert into SLOTS of up to `lanes` pairs, loads each slot's expert once and runs it on all
of its lanes (one matmul column each). Calls under 16 tokens run `kiln_moe_tiles_pairs_v1`, the same
arithmetic one load per pair in order (pairs seldom share an expert there, and the plan, gather and
route below cost more than they save).

**What trn1 lets a kernel make dynamic** (`tools/probe_nki_dynamic.py`, each case on the device and
in `nki.simulate`): only DMA addresses. A compute instruction whose operand sits at an offset held in
a register or an SBUF tensor compiles in the simulator and fails in the BIR backend (`[NCC_IBIR829]
Requested Argument index 1 out of bounds (count=1, type=bir::AccessPattern)` for tensor_scalar and a
register load, `[NCC_IBIR040] Matmult stationary input tile size must be <= 128x128 but it was =
128x1024` for nc_matmul). A DMA from `blob.select(0, e)` with e read from SBUF works, and with
`oob_mode=nisa.oob_mode.skip` an index >= E skips the transfer: 256 expert DMAs (6528 bytes x 128
partitions each) 0.98 ms, a quarter skipped 0.81, half 0.66, all 0.34 ms. Device loops
(`nl.dynamic_range` over a register) run when their body has static addresses (64 iterations of one
op 0.257 ms against 0.161 unrolled: about 1.4 us per iteration; 0 iterations about nothing), but they
may not nest (`operand #3 does not dominate this use`), and wrapping the kernel's slot groups in 0/1
loops gave NaN and 2.2 ms against 1.2 ms static at T=32. So the kernel is static code over a static
slot count, and the routing is data: padded slots skip their DMA, the lanes' x is gathered with a 0/1
matmul (x^T G per 128-row tile: exact) and their outputs are summed into their tokens by a matmul
with the routing weights (R).

**Static slot count.** An expert with n pairs takes ceil(n / lanes) slots, so N = T x 8 pairs take at
most (N + min(E, N) (lanes - 1)) / lanes slots: N itself whenever N <= E (all pairs distinct is
possible), whatever `lanes` is. Up to T=32 the slots' compute is therefore as many as the per-pair
kernel's pairs and the saving is the skipped loads only; above, both shrink (T=64: 320 slots of 512
pairs, C=128: 352 of 1024).

**The expert layout: one power-of-two scale per 128-column tile.** The per-pair kernel's block-32
scales cost a block-diagonal x and 128 partial sums times 128 scales per pair on the vector engine.
MXFP4 scales are powers of two, so each block can be re-based on its tile's exponent K, code *
2^(k_b - K), the same e4m3 value shifted; `moe_dedupe.pack` picks K inside the window where every
code stays an e4m3 value (<= 240 and on the 2^-9 grid) and refuses a tile with none.
`python tools/check_expert_scales.py --shards 1` on `XiaomiMiMo/MiMo-V2.6-Flash-RL`
(`model_pp0_ep0_shard0.safetensors`, 1.39-1.43 billion nonzero codes per projection): block
exponents below their tile's maximum by 0 / 1 / 2 / .. binades in 39.2 M / 9.2 M / 0.72 M / .. gate
tiles, at most 8 (gate), 5 (up), 6 (down, the rank's 64 input rows); no code would round. Dequantizing
the packed blob gives the input weights bit for bit (`tests/test_moe_dedupe.py`), and a 128-row
tile is then one matmul whose PSUM column is the tile's dot product: 32 scale multiplies per pair
for gate_up and 32 for down instead of 128 and 64. The blob is 6272 bytes per partition per expert
(6528 before). `KILN_MOE_KERNEL=nki-dedupe` packs this layout at load.

**What bound it, measured with `tools/prof_timeline.py`** (neuron-explorer's per-instruction JSON,
with the kernel's real inputs from `tools/probe_moe_kernel.py --parts --dump`): every engine runs its
instruction stream in order and waits on count semaphores, so a slot written as one block (gate_up
matmuls, scales, fold, SiLU, down, scales) leaves the tensor engine waiting on the vector engine and
back; the dependency tracking is per TENSOR, so ring buffers that are slices of one tensor serialize
consecutive slots (the tensor and vector engines took turns: 487 us of tensor-engine waits on the
vector engine in a 1 ms kernel); and every vector instruction costs about 200 ns before its
elements. Kernel-alone time at T=32 (256 slots, 162 real) through the fixes, with the routing plan
still computed in the graph: 1.26 ms as written, 1.13 skewed (step i issues stage A of slot i, B of
i - 1, C of i - 2), 1.21 on the tile layout, 1.01 with one tensor per ring entry and four slots per
vector instruction, 1.03 with the expert loads two groups
ahead and the scales copied out of the expert buffers by the scalar engine (so the loads wait only
on the tensor engine's down matmuls). The routing plan in the graph (int32 compares and sums) cost
0.10 ms at T=1 to 0.48 ms at T=64 per call (`tools/probe_moe_plan.py`); in the kernel (iota,
compare + reduce, a prefix-sum scan, 0/1 and fp32 matmuls for the sums over partitions, all exact
on small integers) it is part of the 1.04 ms.

**One MoE call** (`python tools/probe_moe_kernel.py --batch 1 4 16 32 64`; one NeuronCore,
MiMo-V2.6-Flash tp=32 rank shapes: 256 experts, hidden 4096, 128 gate/up and 64 down rows, top-8;
random experts as the checkpoint's MXFP4 ones convert, E2M1 codes as FP8 with power-of-two block
scales; p50 of synchronous calls; error = max abs error against the fp32 host reference over its
max; B=128 from `--batch 32 1 4 16 64 128 --no-xla`):

| B | pairs | distinct | per-pair kernel | dedupe | XLA gather | per-pair err | dedupe err | XLA err | dedupe vs per-pair |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 8 | 0.203 ms | 0.199 ms | 4.31 ms | 0.0035 | 0.0038 | 0.0047 | 0.0048 |
| 4 | 32 | 31 | 0.297 | 0.289 | 2.39 | 0.0032 | 0.0036 | 0.0037 | 0.0041 |
| 16 | 128 | 105 | 0.668 | 0.618 | 22.8 | 0.0049 | 0.0049 | 0.0037 | 0.0063 |
| 32 | 256 | 165 | 1.183 | 1.049 | 54.1 | 0.0046 | 0.0036 | 0.0033 | 0.0049 |
| 64 | 512 | 217 | 2.194 | 1.319 | 76.3 | 0.0055 | 0.0055 | 0.0055 | 0.0060 |
| 128 | 1024 | 255 | 4.129 | 1.662 | | 0.0040 | 0.0055 | | 0.0052 |

(B=1 and 4 run the tile-layout per-pair kernel; 2 and 8 measured 0.229 / 0.408 against 0.238 /
0.417.) The differences between the kernels are a bf16 ulp of the output's largest values (1.0 at
|y| ~ 150-200). The dedupe kernel rounds each pair's output to bf16 before the sum over the token's
experts (its transposes and route matmul take bf16; the per-pair kernels keep it in fp32), within
the same error. At T=32 the kernel is still 1.05 ms where its 162 expert loads alone are about 0.55:
the 256 static slots' tensor-engine work (49 matmuls each, about 1.6 us) plus the gather and route
per block of 128 lanes (about 17 us of tensor engine each) keep it compute- and latency-bound.

**One layer, two live ranks** (`KILN_MOE_KERNEL=<k> python tools/profile_layer.py --ranks 2 --layers
6 --batch B --what layers`, P=16; `--prefill C` for a C-token chunk; layer 1 is a sliding-window MoE
layer, layer 5 a full-attention one; at B=1 the two kernels are within the run-to-run spread):

| shape | nki-pair swa / full | nki (dedupe) swa / full |
|---|---|---|
| decode B=1 | 0.424 / 0.444 ms | 0.447 / 0.446 ms |
| decode B=1, again (two more runs each) | 0.468 / 0.463, 0.440 / 0.446 | 0.399 / 0.424, 0.421 / 0.444 |
| decode B=4 | 0.610 / 0.742 | 0.594 / 0.732 |
| decode B=16 | 1.253 / 1.760 | 1.136 / 1.643 |
| decode B=32 | 2.035 / 1.961 | 1.795 / 1.722 |
| decode B=64 | 3.106 / 4.197 | 2.115 / 3.242 |
| prefill C=32 | 1.576 / 1.600 | 1.356 / 1.376 |
| prefill C=128 | 4.583 / 4.597 | 1.951 / 2.003 |

**The decode step**, 48 layers, two live ranks, 12 layers per graph (`KILN_MOE_KERNEL=<k> python
tools/profile_layer.py --ranks 2 --layers 48 --batch B --pages 4 --what step --step-groups
--moe-groups 12`, i.e. KILN_PIECEWISE_MOE_GROUP=12):

| B | nki-pair | nki (dedupe) | first call (compile, load) |
|---|---|---|---|
| 1 | 9.71 ms | 9.58 ms | 56.5 / 62.0 s |
| 4 | 16.94 | 16.36 | 74.7 / 70.0 s |
| 16 | 41.34 | 36.19 | 112.8 / 100.2 s |
| 32 | 66.70 | 56.11 | 145.9 / 149.0 s |

So it is the default for the NKI path: faster or equal at every shape measured, 16% off the step at
B=32 (the MTP verify step's 16 rows per call are the B=16 row), 2.3x on a 128-token prefill chunk's
MoE layer.

Not measured here: the kernel at 32 ranks on trn1.32xlarge, and a model's perplexity through it
(the layout and the arithmetic are checked against fp32 per call above, and `load_model` with
`moe_kernel="nki-dedupe"` on a quantized MiMo-V2 checkpoint matches Hugging Face within 1e-4 on the
CPU path).

**The whole checkpoint re-bases exactly** (`python tools/check_expert_scales.py --shards 0 --delete
--worker I 4` for I = 0..3 side by side on trn1.2xlarge, 2026-10-03; all 64 expert shards of
`XiaomiMiMo/MiMo-V2.6-Flash-RL`): 788.5 M gate tiles, 788.5 M up tiles and 1577.1 M down tiles
(the rank's 64 input rows at tp=32), **0 refused**. Block exponents sit at most 9 binades below
their tile's largest (gate and down; up 7), and 3 down tiles would round with the tile exponent at
the block maximum, which `_window` avoids by shifting the tile. So `pack` loads the real checkpoint
(it refuses an inexact tile at load either way).

### Variants: GLM-5.3-Flash's MoE (FP8 128 x 128 block scales, clamped SwiGLU) and gpt-oss's activation

What the kernel (both `kiln_moe_dedupe_v8` and `kiln_moe_tiles_pairs_v1`) takes:

- **Expert scales**: MXFP4 (bf16 power-of-two block-32 scales, re-based per tile as above, 6272
  bytes per partition), or FP8 with **128 x 128 block scales** as the loader keeps GLM-5.3-Flash /
  DeepSeek-style checkpoints (`quantization_config.weight_block_size [128, 128]`): fp32 scales per
  (row, 128 input columns) for gate_up, and for down one per output column over the rank's 64 input
  rows (half of one 128-row block at tp=32). Those already are one scale per tile, so `pack` stores
  them unchanged as fp32 (6400 bytes per partition) and the kernels read the scale width from the
  blob's size. Dequantizing the blob gives the input bit for bit (`tests/test_moe_dedupe.py`).
- **Activation** (a static kernel argument, `moe_dedupe.ACTS`): 0 `silu(g) * u`; 1 `silu_clamp`,
  `silu(min(g, limit)) * clamp(u, -limit, limit)` (GLM-5.3-Flash, `swiglu_limit 10.0`, as
  `models/hybrid.py _moe_clamped`); 2 `swiglu_oai`, `(clamp(u, -limit, limit) + 1) * g *
  sigmoid(1.702 g)` with `g = min(gate, limit)` (gpt-oss, limit 7). Interleaved gate/up is a
  load-time row permutation; expert BIASES (gpt-oss has them) are not supported.
- **Experts**: up to 512 (four tiles of 128 for the in-kernel plan; expert ids are kept in fp32,
  since bf16 rounds ids past 256). GLM-5.3-Flash has 288.
- **Shapes**: 2 Im = 128 gate/up rows per rank exactly (MiMo-V2.6-Flash and GLM-5.3-Flash at
  tp=32: `moe_intermediate_size` 2048 / 32 = 64), hidden a multiple of 256, at most 1024 pairs and
  128 tokens per call (`moe_dedupe` chunks longer inputs). So gpt-oss at tp=32 (Im 2880 / 32 = 90),
  and GLM-5.3-Flash at any other tp (tp=16: 128 rows of Im per rank), take the XLA path.

`hybrid._moe_clamped` runs the kernel on the device when the layer holds the tile blob
(`KILN_MOE_KERNEL=nki`); before this it raised for clamped models under the NKI kernel.

**One MoE call, GLM-5.3-Flash rank shapes** (`python tools/probe_moe_kernel.py --experts 288
--scales block128 --act silu_clamp --limit 10 --batch 1 4 16 32 64`, and `--batch 64 128 --no-xla`;
trn1.2xlarge, one NeuronCore, SDK 2.32; 288 experts x (128 x 4096 gate/up, 64 x 4096 down), random
finite e4m3 bytes with fp32 128-block scales, top-8; error = max abs error against the fp32 host
reference over its max):

| B | pairs | distinct | kernel | XLA gather | kernel err | XLA err | kernel vs XLA |
|---|---|---|---|---|---|---|---|
| 1 | 8 | 8 | 0.199 ms | 0.310 ms | 0.0031 | 0.0059 | 0.0069 |
| 4 | 32 | 30 | 0.293 | 0.717 | 0.0052 | 0.0060 | 0.0051 |
| 16 | 128 | 99 | 0.651 | 36.8 | 0.0051 | 0.0067 | 0.0090 |
| 32 | 256 | 176 | 1.077 | 23.3 | 0.0052 | 0.0063 | 0.0052 |
| 64 | 512 | 241 | 1.442 | 46.1 | 0.0045 | 0.0067 | 0.0070 |
| 64, other routing | 512 | 237 | 1.436 | | 0.0052 | | |
| 128 | 1024 | 280 | 1.859 | | 0.0057 | | |

(The kernel against XLA is a bf16 ulp at the output's largest values.) Lanes per slot at B=128,
one routing (283 distinct; `--batch 128 --no-xla --lanes L`, run beside four CPU-bound host
processes): 4 lanes 2.070 ms, 8 (the default) 1.972, 16 2.403. `swiglu_oai` at limit 7
(`--act swiglu_oai --limit 7 --batch 4 32`, with `--scales block128 --experts 288` and with
`--scales mxfp4 --experts 256`): kernel 0.294 / 1.082 ms against XLA 0.720 / 23.3 at B=4 / 32 on the
128-block experts; kernel error 0.0031-0.0045, XLA 0.0025-0.0078. The limit bites in the simulator
tests (limit 0.02 with 128-block scales, 160 experts: a partial tile of 128).

**The MoE block of a GLM-5.3-Flash layer, two live ranks** (`KILN_MOE_KERNEL=<k> python
tools/profile_layer.py --model zai-org/GLM-5.3-Flash --tp 32 --ranks 2 --layers 4 --batch B --what
mlp`; trn1.2xlarge, SDK 2.32, random weights in the checkpoint's formats; layer 3's routed experts,
shared expert and all-reduce as one graph, `hybrid._mlp`):

| B | nki | xla | null graph |
|---|---|---|---|
| 1 | 0.381 ms | 0.422 ms | 0.125 ms |
| 4 | 0.406 | 0.899 | 0.135 |
| 16 | 0.693 | 37.38 | 0.156 |
| 32 | 1.102 | 25.02 | 0.198 |
| 64 | 1.559 | 50.34 | 0.236 |
| 128 | 1.998 | (not run) | 0.306 |

The XLA path in a graph of its own is close at B=1 (the gather is few packets there); at B >= 16 it
is 25-50 ms per layer. Not measured here: GLM-5.3-Flash's decode step at tp=32 on trn1.32xlarge
through the kernel (267.5 ms/step at B=4 on the XLA path, `docs/model-coverage.md`), and its ppl
through the kernel.

## Prefill MoE as a grouped GEMM (2026-10-03, SDK 2.32, nki 0.6.0, trn1.2xlarge)

`kiln/kernels/moe_prefill.py`, on when `KILN_MOE_PREFILL_KERNEL=nki` (with `KILN_MOE_KERNEL=nki`,
whose tiles blob it reads; the layer is refused at load otherwise) for prefill chunks of at least
`KILN_MOE_PREFILL_MIN_TOKENS` (512) tokens; shorter chunks and decode keep moe_dedupe. The decode
kernels cannot serve an 8192-token chunk: the per-pair kernel loads an expert per (token, expert)
pair (3.7 us per pair, 65,536 pairs), and moe_dedupe holds its tokens on the partitions, so it runs
a chunk 128 tokens at a time and reads nearly every expert once per call (64 times at C=8192).
Here every expert is applied once to all of its tokens.

**Layout and plan.** The C x k pairs are ordered by expert and each expert's pairs padded to whole
BLOCKS of B lanes (B = 64 up to C = 2048, 128 above; `block_size`), so each block is one expert;
128 / B blocks make a lane tile. The block count is static, `n_blocks` = (C k + E (B - 1)) / B in
whole tiles, which every routing fits: the kernel is exact for any routing, with no capacity
factor and no overflow path, and blocks past the routing's own are expert E, whose DMAs are
skipped (`oob_mode.skip`). The plan runs inside the kernel (its CPU reference is `plan()`, checked
against a sort-and-loop `plan_reference` up to 8192 tokens x 288 experts): per (token tile, k)
group a one-hot of the experts, the earlier pairs of each expert by a strictly-lower-triangular
matmul plus the running per-token counts of earlier groups (summed over the partitions by a ones
matmul, kept below 256 so bf16 operands stay exact), blocks per expert by an integer shift, a
log-step prefix sum over the experts, each pair's slot through `nc_n_gather`, the token of every
slot by one 4-byte indirect scatter per group, and each block's expert by a comparison matmul.

**A lane tile** (static code, software-pipelined in four stages so every engine's in-order stream
holds work of four tiles and every hand-off between engines has other work in between): the
block's expert blob (one dynamic DMA) and the lanes' x rows (one indirect DMA) two tiles ahead; x
transposed on the tensor engine (32 [128, 128] transposes, drained by the scalar engine); gate_up
dequantized to bf16 in SBUF (fp8 x the gate rows' and the up rows' tile scale, 64 `tensor_scalar`
ops of [128, 64]) and accumulated over the 32 tiles of h in PSUM; the up rows folded onto the gate
rows by a 0/1 matmul; a = glu(gate, up) (SiLU, or GLM-5.3-Flash's clamp at 10) in fp32, rounded to
bf16; y = a W_down on the fp8 down weights as stored, drained with each 128-column chunk's scale;
y stored to Y in slot order by a static DMA. A last pass gathers each token's k rows of Y and sums
them on the tensor engine with diag(routing weight) stationaries, in fp32, rounded once.

**Expert formats** (`check_blob`, `down_factors`, at load): gate_up takes the dequantize-first path
when the rank's gate rows share each tile scale and so do its up rows (`layer.moe_prefill_dq`),
otherwise the per-row path (one PSUM partial per tile, scaled per row and summed in fp32 on the
vector engine), which takes any per-row tile scales. Down scales are one per output column in the
blob: constant over each 128-column chunk, the scalar engine's drain applies one per chunk;
otherwise `down_factors` splits each into its chunk's smallest scale times a power of two (exact,
checked at load: it raises if a ratio is not a power of two), the drain applies the chunk scale and
the vector engine multiplies the bf16 product by the factor, which a 0/1 matmul broadcasts onto each
block's lanes (bf16(y s_c) f = bf16(y s_c f) for a power of two f). The layer keeps the factors as
`moe_prefill_dsc` fp32 [E, H / 128] and `moe_prefill_dfr` bf16 [E, H] (2.4 MB per layer at
GLM-5.3-Flash's rank shapes). The bf16 scales of re-based MXFP4 (MiMo-V2.6-Flash) always take both
per-row paths.

**Numerics**: the kernel's arithmetic is `moe_dedupe.emulate(pair_bf16=True)`'s (fp32 partials
times the tile scales, or bf16(fp8 x scale) first on the dequantize-first path, as
`models/quant.dequant` does; g, a and y rounded to bf16; the k-sum in fp32); `emulate()` computes it
one expert at a time. On the device the kernel matches it within 0.0018-0.0042 of the output's
max, and both sit 0.005-0.010 from the fp32 reference (the same distance for both, i.e. bf16
rounding), at every C below. Against the XLA paths of the decoder (dequantize, then bf16
matmuls) it is a rounding-order difference (`tests/test_moe_prefill.py`, CPU).

**One MoE call, GLM-5.3-Flash rank shapes** (`python tools/probe_moe_prefill.py --chunks 256 512
1024 2048 4096 8192 --decode-max 8192 --pair-max 4096 --iters 5`; trn1.2xlarge, one NeuronCore;
288 experts x (128 x 4096 gate/up, 64 x 4096 down) random finite e4m3 bytes with fp32 128-block
scales, top-8 with distinct uniform experts per token, clamp 10; p50 of a graph that reads back
out.float().sum(0), minus the same reduction of a [C, 4096] input; TFLOPS counting 2 x pairs x
(4096 x 128 + 64 x 4096); trn1's tensor engine peaks at 92 TFLOPS bf16 / fp8; the per-pair decode
kernel runs MiMo-layout weights of the same shapes, the only layout it takes; instructions: the
kernel graph's NEFF, the sum of its engines' instruction streams at 64 bytes per instruction, which
is what neuron-explorer counts; the prefill column from the final kernel, the two decode-kernel
columns from the same command run an hour earlier, whose prefill times were the same within 0.02 ms):

| C | pairs | blocks static / used (B) | prefill kernel | TFLOPS | instructions | moe_dedupe (128-token calls) | per-pair kernel (512-pair calls) |
|---|---|---|---|---|---|---|---|
| 256 | 2048 | 316 / 288 (64) | 3.570 ms | 0.90 | 71,716 | 3.268 ms | 7.635 ms |
| 512 | 4096 | 348 / 288 (64) | 4.012 | 1.61 | 79,302 | 6.315 | 15.172 |
| 1024 | 8192 | 412 / 288 (64) | 4.986 | 2.58 | 94,618 | 12.390 | 30.253 |
| 2048 | 16384 | 540 / 331 (64) | 6.755 | 3.81 | 125,033 | 24.657 | 60.760 |
| 4096 | 32768 | 541 / 308 (128) | 9.518 | 5.42 | 158,061 | 52.170 (1,521,742 instructions) | 121.427 (3,712,252) |
| 8192 | 65536 | 797 / 584 (128) | 15.237 | 6.77 (7.4% of peak) | 236,669 | 104.306 (3,043,539) | not run |

So 6.8x moe_dedupe at C=8192 and 2.5x at 1024; below 512 tokens moe_dedupe is faster (all 288
experts are loaded either way, and the prefill kernel's plan and static blocks cost more), hence the
512-token threshold. The instruction count matters for multi-layer prefill graphs (neuronx-cc stops
at 5 M, `NCC_EBVF030`): moe_dedupe unrolls one kernel per 128 tokens, 47,600 instructions each
(1,521,742 / 32 and 3,043,539 / 64 above), so a 2048-row MoE layer is about 16 x 47,600 = 0.76 M
through it and 125 k through this kernel. A whole GLM-5.3-Flash layer at C=2048 is 737,052
instructions with DSA attention and 882,360 with KDA (the single-layer graphs of the layer table
below, each including this kernel's MoE block of 130,747), so a prefill group of 12 layers at 2048
rows stays above the limit even so: the attention is 600-750 k instructions per layer; about 5
layers per group fit at C=2048. The per-pair kernel is linear at 3.7 us per pair over the five sizes
it ran (about 240 ms at 8192, extrapolated, not measured: its 128-call graph was not compiled). The
XLA paths are no comparison at these sizes: the every-expert path used above 512 pairs did not
compile at C=128 on this host at MiMo-V2.6-Flash's rank shapes (walrus_driver killed at 27 GB RSS,
"Decode MoE as one NKI kernel" above), and the gather path took 46.1 ms for 512 pairs at these
shapes (decode B=64, "Variants" above). Block size, measured at C=1024 / 2048 / 4096 / 8192 with
`--block`: B=64 4.97 / 6.75 / 10.25 / 17.31 ms, B=128 5.57 / 7.18 / 9.52 / 15.24. MiMo-V2.6-Flash
rank shapes (`--format mimo --chunks 1024 8192 --decode-max 1024 --pair-max 0`: 256 experts, E2M1
codes with power-of-two block-32 scales, SiLU): C=1024 5.33 ms (moe_dedupe 11.28), C=8192 19.16 ms
(5.38 TFLOPS), kernel vs emulation 0.0037 / 0.0042.

**Where the time goes at C=8192** (neuron-explorer profile of the kernel graph, `python
tools/prof_engines.py <cache hash> <inputs>` with the inputs `--save-inputs` wrote; the four-stage
kernel before its loads moved two tiles ahead, 15.11 ms): the plan about 0.9 ms, vector-bound, then
about 1 ms for the 512 scatter DMAs of the token table (128 x 4 bytes each, GpSimd descriptor
generation, nothing to overlap with); the 580 used tiles about 14.3 us each, bound by the
software-DGE DMAs (x rows and expert blob, about 125 GB/s together); the 217 static tiles past the
routing's own about 9.6 us each (no loads; the tensor engine 77% busy, vector and scalar about 58%);
the combine about 2.3 ms (512 MB of Y rows gathered at about 236 GB/s). In those static tiles the
tensor engine works about 7.4 us (32 transposes, 32 gate_up and 8 down matmuls, each with its
weight load), the vector (dequantization) and the scalar engine (drains) about 5.6 each.

**What moved it**, C=8192 and C=1024 through the same probe: natural pair order with the plan in
torch, 25.02 / 8.96 ms; the plan inside the kernel 20.08 / 6.72; the plan's first pass without
single-partition chains 18.17 / 6.48; Y stored in slot order by a static DMA instead of an indirect
one that skipped empty lanes (the indirect store held the GpSimd queue, which also issues every
load) 16.51 / 4.97; four pipeline stages instead of three and the dequantization as per-half
`tensor_scalar` ops 15.11 / 4.98. Measured and not kept: the loads two tiles ahead (15.20, kept for
B=128 as neutral); splitting each load into 2 or 8 DMAs (17.23 / 23.43 ms at C=8192), although
alone eight DMAs of [128, 1 KB] move a tile's 1.8 MB in 2.2 us against 8.2 us for one x gather and
one blob load (`tools/probe_prefill_dma.py` variants 19 and 2): each dynamic DMA costs the GpSimd sequencer
about 1 us (a TENSOR_LOAD of its offset, address arithmetic, the trigger), so more of them starve
the queue that issues all of them; four DMAs per Y gather in the combine (16.13); the per-row path
for GLM's block scales (20.39 / 5.35 before the per-half dequantization, 7.66 / 5.59 at C=2048 /
1024 after it, against 6.75 / 4.97).

trn1 facts behind those choices (`tools/probe_prefill_vector.py`, one [128, 4096] block, us): a
`tensor_tensor` with two SBUF operands runs at half the element rate of a `tensor_scalar` (fp8 x
broadcast fp32 scales 8.50, 64 `tensor_scalar` ops of [128, 64] with the scale as a per-partition
scalar 5.84), and at full rate when one operand is in PSUM (3.74 for 8 x [128, 512]); the scalar
engine dequantizes the same block in 4.43 (64 activations), both engines together in 2.69, but it
also drains every PSUM tile, so the kernel leaves the dequantization on the vector engine.

**A GLM-5.3-Flash layer at prefill, two live ranks** (`KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki
python tools/profile_layer.py --model zai-org/GLM-5.3-Flash --tp 32 --ranks 2 --prefill C --what
layers mlp --layers 5 --pages P --page-size 32 --sum-readback`; random weights in the checkpoint's
formats; layer 0 dense with KDA, 3 MoE with DSA attention, 4 MoE with KDA; the MoE block is layer
3's routed experts, shared expert and all-reduce in one graph):

| C | MoE block | layer 0: dense MLP, KDA | layer 3: MoE, DSA | layer 4: MoE, KDA | null graph |
|---|---|---|---|---|---|
| 512 | 4.112 ms (80,552 instructions) | 37.580 ms | 15.270 ms (253,940) | 16.885 ms (264,770) | 0.215 ms |
| 2048 | 7.702 (130,747) | 133.640 (1,002,198) | 45.993 (737,052) | 84.086 (882,360) | 0.485 |

(`--pages 16` at C=512, `--pages 64` at 2048; compile 80-580 s per single-layer graph at 2048.) The
MoE block is 9-27% of a MoE layer here; the rest is mostly the token mixer (about 76 of layer 4's 84
ms at C=2048 is outside the MoE block, KDA's chunked recurrence the largest part), which is where
prefill time goes next. C=8192 layers were not compiled (the KDA graph took 577 s at 2048 and its
instruction count grows with C).

**The real GLM-5.3-Flash checkpoint as the loader holds it (2026-10-04).** The first kernel assumed
the layout of the checkpoint's 128 x 128 blocks: at tp=32 the rank's 64 gate rows, 64 up rows and 64
down input rows each sit inside one block, so gate rows share a tile scale and the down scales are
constant over 128 output columns. Every rank of a real-weights run then refused it at load
(`ValueError ... needs fp32 down scales constant over each 128-column chunk`): models/quant.py
`fit_e4m3_max` (trn1's e4m3 has no values above 240, the checkpoint's e4m3fn reaches 448) halves the
codes of each (row, block) whose largest value exceeds 240 and doubles that row's scale, and in this
checkpoint most rows exceed it. `python tools/check_moe_prefill_layout.py --model <GLM-5.3-Flash>
--layers L --rank R` runs the loader's own expert path (`loader._load_experts`) on the CPU: layer
3, rank 0, expert 0 has 2687 of 4096 down rows above 240; after the fit 2.1% of (expert, tile) keep
one gate scale and 17 of 9216 down chunks one scale (expert 0, columns 0-127: 32 columns at
0.000174386 and 96 at twice that); the ratios are 1 or 2 everywhere, and `down_factors` gives the
loaded scales back exactly in every MoE layer (3-44 and the MTP layer 45) of ranks 0 and 7, and
layers 3 of rank 31. trn1 itself needs the fit: all 256 e4m3fn codes converted on the device (an
LNL graph, as `dequant` does) match the host except 0x78-0x7E and 0xF8-0xFE (256 to 448), which come
back inf or NaN; subnormals are exact.

The kernel now takes that layout as it is (per-row gate_up path, per-column down factors) instead of
the loader changing it. One MoE call on layer 3's real experts of rank 0 (`python
tools/check_moe_prefill_layout.py ... --save`, then `tools/probe_moe_prefill.py --experts-file
<file> --chunks 512 2048 8192 --decode-max 2048`, trn1.2xlarge): kernel vs emulation 0.0017 /
0.0034 / 0.0031 of the max, the same distance from the fp32 reference as the emulation (0.0053 /
0.0071 / 0.0060); C=512 5.064 ms (moe_dedupe 6.312), C=2048 8.049 (24.658), C=8192 21.466;
74,416 / 117,014 / 237,203 instructions. The same experts with down made block-constant as
feat/trn2-bench 22dd2c2's refit would leave it (`--block-constant`: per-chunk drains, per-row
gate_up): 4.555 / 7.657 / 19.603 ms at C=512 / 2048 / 8192, kernel vs emulation 0.0017 / 0.0034 /
0.0023. Random GLM-format
experts (gate rows sharing tile scales, the dequantize-first path) 4.031 / 15.211 ms at 512 / 8192,
unchanged by the new path; the per-row gate_up path alone on them (`--no-dq`) 4.498 / 19.615.
MiMo-V2.6-Flash shapes through the factors instead of a broadcast DMA of each block's scale row:
5.619 / 19.244 ms at C=1024 / 8192 (5.33 / 19.16 before).

The two refits keep the checkpoint's values equally well (`python tools/check_e4m3_refit.py --model
<GLM-5.3-Flash> --layers 3 45 --other-weights`, CPU; error against code x weight_scale_inv):

| tensors | per (row, block) halving (engine-v0) | per (row group, block) halving (22dd2c2) |
|---|---|---|
| layer 3 experts, rank 0 (288 x gate_up 128 x 4096, down 4096 x 64) | 20,339 of 226.5 M values changed (0.0090%), max abs 5.11e-7, 4.36e-6 of the block's range (scale x 448); expert outputs 5.5e-6 | 23,662 (0.0104%), max abs 5.11e-7, 4.36e-6; 5.5e-6 |
| layer 45 (MTP) experts, rank 0 | 7,265 (0.0032%), max abs 3.82e-6, 4.36e-6; 8.5e-6 | 17,886 (0.0079%), max abs 7.63e-6, 4.36e-6; 4.2e-5 |
| layer 3 shared experts, attention (q_a, q_b, kv_a, o), whole tensors | 109 to 6577 values per tensor, at most 4.36e-6 of the range | 262 to 8119, at most 4.36e-6 |

Halving is exact for normal e4m3 values; the only values that change are subnormal codes with an odd
last bit, by half the smallest subnormal step (2^-10 x scale = 4.36e-6 of 448 x scale). So both refits
represent the checkpoint within that step on every FP8 tensor checked (layers 0, 3, 4 and 45:
experts, shared experts, dense MLP, attention projections), and the ppl difference measured between
them on trn2 (-2.073 against -1.843; the CPU bf16 reference without any fit, -2.098, sides with the
per-(row, block) one, bench/results/2026-10-03-trn2.48xlarge-moe.md) does not come from the weight
values the two produce; engine-v0 keeps the per-(row, block) fit and the kernel runs it as loaded.
CPU suite on feat/moe-prefill2 at 714a1dd (engine-v0 4ca86ce merged), one pytest process per file on
kiln-mp2-trn1: 566 passed, 25 skipped, 0 failed.

The CPU suite on the merged tree (engine-v0 617ad73 into feat/moe-prefill), one pytest process per
test file on kiln-mp-trn1 (one process for all of it is OOM-killed on a trn1.2xlarge host): 553
passed, 25 skipped, 0 failed (2026-10-03).

Not done here: the kernel at 32 ranks and a GLM-5.3-Flash prefill through it end to end (the
model's perplexity through the kernel); skipping the static blocks past the routing's own (device
loops over a register run only with static addresses, see the dedupe section); B = 32 (the down
matmul's stationary would be 32 columns: "[NCC_IBIR058] PE tile size 32x32 on CoreV2"); expert
biases and gpt-oss's activation in this kernel (moe_dedupe has the latter).

## Linear attention (Gated DeltaNet, KDA) on trn1 (2026-10-03, SDK 2.32, trn1.2xlarge)

Measured with `tools/profile_linear_attn.py` (one layer, random bf16 weights, Qwen3.5-0.8B GDN
shapes unless noted; `--parts` times the pieces) and neuron-explorer on the layer NEFFs.

- **A ONE-row `index_put_` of a value the graph computed is the slow path.** Writing a
  sequence's 1 MB recurrent state ([1, 16, 128, 128] fp32) back into the state pool took 8.7 ms
  when the value was `row + a^T b`, also as `index_copy_`, computed transposed, or with a
  broadcast multiply-sum instead of a matmul; the gather-only write (`p[r] * 0.5`) and a matmul on
  the gathered row without the write-back were 0.13 and 0.14 ms. The same write as TWO rows (the
  row plus the pool's scratch row) is 0.14 ms, and a masked rewrite of the whole 9-row pool
  (`copy_(where(...))`) 0.17 ms. The profile of the slow graph: 72,960 software dynamic-DMA
  packets, GPSIMD (software descriptor generation) 1.48 ms active. Kiln writes single rows as two
  (`linear_attn._write_rows`): the GDN prefill layer at C=128 went 9.24 -> 0.52 ms. Decode
  batches of B >= 2 rows were never affected (B=8 index_put_ 0.13 ms); B=1 decode takes the same
  two-row write. Probe: `--parts`, the "write 1 row" lines.
- **The einsum form of the delta-rule step lowered to tensor-engine transposes.** Decay S, read
  k^T S, rank-1 update, read q^T S (three passes over S [B, H, Dk, Dv]) measured 1.05 ms per
  layer at B=8 with 3.95 GFLOP of transposes in a graph of 0.18 GFLOP model flops (12,901
  tensor-engine instructions). Reading the old state once as [q E; k E] @ S and updating
  elementwise (`linear_attn.recurrent_step`) is the same algebra and 0.59 ms.
- **Eager dtype casts of device tensors fail**: `x.float()` and `x.to(torch.float32)` on a bf16
  `neuron` tensor raise "Expected self.dtype() == dst.dtype() to be true"; `x.cpu().float()` and
  in-graph casts work. Read results back first, then convert.
- Sub-chunk length of the chunked prefill (C=128): GDN 0.61 / 0.52 / 0.46 ms at 32 / 64 / 128;
  KDA (Kimi-Linear-48B shapes, its [heads, L, L, Dk] decay) 3.38 / 3.29 / 4.10 ms at 16 / 32 / 64.
- fp32 weights with `KILN_CC_ARGS=--auto-cast=none` (neuronx-cc otherwise computes fp32 matmuls
  in bf16) make a rounding-free device check: Qwen3.5-0.8B 8/8 prompts 32/32 token-exact
  against transformers fp32, where bf16 diverges on 2 at reference margins of 0.06 and 0.04
  (`bench/results/2026-10-03-trn1.2xlarge-qwen3.5-0.8b.md`). Those prompts are all shorter than 16
  tokens, which is why they did not see the next item.

## Linear-attention serving: prefix cache, verify, state copies (2026-10-03, SDK 2.32, neuronx-cc 2.27.5334, trn1.2xlarge)

Qwen/Qwen3.5-0.8B (revision 2fc06364), piecewise 4 layers per graph, page 32, prefill bucket 128,
`tools/bench_linear_serving.py` (full numbers: `bench/results/2026-10-03-trn1.2xlarge-qwen3.5-0.8b-serving.md`).

- **The chunked delta rule's sub-chunk inverse was numerically wrong with trained weights, on CPU and
  device alike.** `(I + A)^-1` over a sub-chunk was computed as `(I + N)(I + N^2)(I + N^4)...` (exact
  algebra, plain matmuls): with nearly parallel keys, beta near 1 and slow decay (what trained weights
  give) the powers of N grow like binomial coefficients before they cancel, and fp32 loses everything.
  Measured on CPU against the token-by-token recurrence, Qwen3.5-0.8B layer 6 on a 63-token prompt:
  9.5e7 off at a 64-token sub-chunk, 0.21 at 32, 1.8e-6 at 16; a synthetic correlated-keys case 1.2e3 /
  7.6e18 at 32 / 64. Every prefill sub-chunk past 16 tokens was affected (GDN used 64, KDA 32), so every
  prompt longer than 16 tokens, on every linear-attention model; earlier device checks used prompts of
  at most 16 tokens. Fixed by inverting 8-row diagonal blocks by squaring and merging halves
  (`[[A, 0], [C, D]]^-1 = [[A^-1, 0], [-D^-1 C A^-1, D^-1]]`, fla's solve_tril scheme): 1e-7 at every
  length. Device after the fix (fp32, `--auto-cast=none`): 187-228-token prompts 32/32 tokens equal to
  transformers fp32 on 4 of 4. Test: `tests/test_linear_attn.py::test_chunk_scan_is_stable_on_correlated_keys`.
- **How the blocked inverse is written decides its cost, and in-graph index masks are the trap.** One
  layer, prefill C=128 (`KILN_LA_CHUNK` / inverse variants through `tools/profile_linear_attn.py`, bf16,
  random weights): the unstable squaring at sub-chunks of 64 was GDN 0.505 ms. Stable variants: block
  by block (recursion with slices and cats) 2.77 ms; each level one batched op over its blocks
  (slices + stack) 2.34 ms; full-size matrices with block masks built in the graph from arange /
  expand / reshape / compare 3.23 ms; the same with the masks as trace-time constants (numpy on the
  host, `torch.tensor` in the traced function) **0.511 ms**. KDA (sub-chunks of 32): 3.36 ms with
  constant masks against 3.29 before. Pure squaring at 16-token sub-chunks (stable enough on these
  weights, 1.8e-6) was GDN 0.887 ms: more sub-chunks cost more than the merges.
- **Verify keeps the state after every drafted position; that beats recomputing.** Per graph, synchronous
  (`KILN_PROFILE_EXEC=1`), bf16, Q = 5 (k = 4): decode B=1 9.4-9.7 ms, B=8 17.4-18.8 ms; verify B=1
  12.1 ms with every state kept, 11.7 ms keeping only the last (`KILN_PROBE_VERIFY_LAST_STATE_ONLY=1`,
  which is all a recompute-based rollback would need from it), B=8 50.6 against 41.3 ms. Rolling back
  by recompute would add a second pass over the accepted tokens of about the verify's own size, so
  snapshots cost 0.4 ms (B=1) / 9 ms (B=8) where recompute would cost about 12 / 41 ms. The verify's
  "chunk" form (`KILN_LA_VERIFY=chunk`: one triangular solve, then the states by accumulation) measured
  13.4 ms at B=1 and 46.6 ms at B=8; the recurrent form stays the default (the decode step's own
  arithmetic).
- **At B=8 a verify step is 2.8x a decode step** for this model: a 4-layer group (3 GDN + 1 gated
  attention) takes 7.9 ms for 40 verify rows against 9.5 ms for a 128-token prefill chunk; the 5
  sequential delta-rule steps over [8, 16, 128, 128] states dominate (`KILN_PROFILE_PIECES=1`). n-gram
  speculation (k = 4, repetitive prompts) is 1.50x faster than plain decoding at B=1 (153 vs 102
  tok/s) and 0.47x at B=8 (157 vs 336 tok/s); `spec_k_per_batch_size` (vLLM's
  num_speculative_tokens_per_batch_size) turns drafting off by batch size.
- **A 2-row state copy is the slow path again.** The checkpoint copy graph (`pool[dst] = pool[src]` over
  every state pool, 18.6 MB per row): 4.5 ms for 2 rows, 1.1 ms for 4, 2.0 ms for 8 (p50 of 30,
  interleaved). Copies are padded to at least 4 rows (scratch onto itself).
- **Checkpoint cost.** One row is 18.6 MB for Qwen3.5-0.8B (18 GDN layers, bf16 conv, fp32 recurrent).
  A 1081-token prompt that resumes 1056 cached tokens: 19.4 ms to first token against 167 ms; the
  prompt-boundary checkpoint (`state_checkpoint_prompt`, now off by default) costs one more prefill call
  on every uncached prompt, 167 -> 186 ms there.

## MLA and DSA on trn1 (2026-10-03, SDK 2.32, neuronx-cc 2.27, trn1.2xlarge)

Three things that compiled and ran on CPU broke on the device; each has a probe in
`tools/probe_dsa.py` and the formulation that works is in `kiln/models/mla.py`.

- **An int64 top-k index written into an int64 scratch tensor failed in LNL's tracer**:
  `Check failed: self.scalar_type() == values.scalar_type()` at the `index_put_` (the DSA
  top-k scratch shared layers read). The scratch is int32 with explicit casts on write and
  read.
- **Partial in-place RoPE on the indexer query broke neuronx-cc**: slicing the first 64 dims of
  each of the indexer's heads, rotating them and concatenating the rest back (`cat([rope(q[...,
  :64]), q[..., 64:]])` on [8, 8, 128]) made the decode layer of GLM-5.3-0.6B-A0.4B fail with
  `[NCC_INLA001] BIR verification failed: Pattern accesses 64 (> 32) partitions starting at
  partition 32` (`tools/probe_dsa.py --layer --variant ...`: removing the query rotation alone
  compiles; the same on the key, which has no head axis, is fine). Only the sparse regime is
  affected: below index_topk the indexer query is dead code and the compiler drops it. Kiln
  splits wq_b by rows at load time into every head's rope rows and the rest, so the two parts
  come from two matmuls and are never concatenated; the score is the sum of two einsums.
- **`x.view(T, 8, 2).topk(2, dim=-1).values` came back WRONG** (max error 1.54 against the
  host, `tools/probe_dsa.py --routing`): a top-k over an axis of size 2. DeepSeek-V3's
  group-limited routing takes each group's two best scores; with 16 experts in 8 groups the
  whole decode was wrong (max |dlogprob| 0.36-1.7 on a random DeepSeek-V3 config) and exact
  with `n_group=1`. Kiln computes the top-2 sum as `amax + amax(rest)` with the first argmax
  masked out; everything then matches (below).

Measured after those fixes, `tools/check_device.py` (8 prompts, greedy, against transformers
5.15 fp32 on the host; `max|dlogprob|` is the reference scoring Kiln's own tokens), decode
bucket B=8, prefill bucket 128, pages of 32 tokens:

| model (all with Kiln's MLA cache) | dtype | DSA | result | decode step p50 |
|---|---|---|---|---|
| inference-optimization/GLM-5.3-0.6B-A0.4B (glm_moe_dsa, real weights) | fp32 | dense (topk 2048) | 8/8 prompts 32/32 tokens | 20.4 ms |
| same | bf16 | dense | 3/8 to 32 tokens, the rest diverge at margins 0.005-0.06 | 9.20 ms |
| same, `--kv-cache-dtype fp8` | bf16 | dense | 1/8 to 32 tokens (margins 0.004-0.10) | 9.25 ms |
| same, `KILN_MLA_PREFILL=absorb` | fp32 | dense | 8/8 32/32, max\|dlogprob\| 4.4e-5 | 20.4 ms |
| same, `--spec-method ngram --spec-k 3` (verify graph) | fp32 | dense | 8/8 64/64, 55% of drafts accepted | 21.2 ms |
| BAAI/OpenSeek-Small-v1-SFT (deepseek_v3 without q_lora, 64 experts + 2 shared, real weights) | bf16 | - | 5/8 to 32 tokens, the rest at margins 0.014-0.076 | 12.6 ms |
| tencent/Youtu-LLM-2B (dense MLA, real weights) | fp32 | - | 8/8 32/32, max\|dlogprob\| 1.2e-4 | 59.3 ms |
| same | bf16 | - | 6/8 to 32 tokens; transformers' own bf16 also matches its fp32 on 6/8 and makes the same first-token switch on prompt 2 (`tools/hf_dtype_drift.py`) | 23.6 ms |
| random DeepSeek-V3 config (yarn x40, 8 expert groups) | fp32 | - | 8/8 8/8, max\|dlogprob\| 2.8e-5 | |
| random DeepSeek-V3.2 config, `--override index_topk=16` | fp32 | sparse, mask | 8/8 160/160, max\|dlogprob\| 3.2e-5 | 6.99 ms |
| same, `KILN_DSA=gather` | fp32 | sparse, gather | 8/8 160/160, max\|dlogprob\| 3.3e-5 | 6.86 ms |
| random GLM-5.3 config (IndexShare), index_topk 16, piecewise | fp32 | sparse, mask | 8/8 160/160, max\|dlogprob\| 3.7e-5 | 7.57 ms |
| same, one whole-model graph | fp32 | sparse, mask | 8/8 160/160 | 7.32 ms |
| same, `--tp 2` (the two NeuronCores: heads split, latent and indexer replicated) | fp32 | sparse, mask | 8/8 64/64, max\|dlogprob\| 2.0e-5 | 7.68 ms |
| same in FP8 (e4m3fn, 128x128 block scales) vs its dequantized twin | fp32 | sparse, mask | 8/8 64/64, max\|dlogprob\| 2.5e-5 | 7.96 ms |

Commands: `tools/check_device.py --model <m> [--dtype fp32] --piecewise --tokens <n>` (decode
bucket 8, prefill bucket 128, pages 4 and 32 of 32 tokens), plus `--override index_topk=16`,
`--kv-cache-dtype fp8`, `--spec-method ngram --spec-k 3`, `--reference <dequantized twin>` or the
environment variable named in the row; OpenSeek with `--tokenizer Qwen/Qwen2.5-0.5B` (its own
tokenizer is remote code) and, like the n-gram run, `KILN_MOE_GATHER_MAX_PAIRS=64` (below).
Random configs are tools/build_random_mla.py (tests/test_mla.py's builder: the real
config.json, 128 hidden, 4 heads, real MLA / indexer head geometry). The real GLM-5.3-0.6B at a
small index_topk is not a usable sparse reference: its 8 indexer heads, lightly trained, often
score a key exactly 0 (every head's ReLU zero), and measured on 71 queries per full layer 1-3
had the k-th and (k+1)-th score tied at exactly 0. Any tie-break is a valid top-k, transformers'
and Kiln's differ, and the outputs part there (CPU fp32: 6/8 prompts exact to 160 tokens at
index_topk 16; trn1: 4/8).

**A MoE verify graph in fp32 starved the host.** `--spec-method ngram` on GLM-5.3-0.6B in fp32:
the verify bucket (8 x 4 rows, 256 expert pairs, under KILN_MOE_GATHER_MAX_PAIRS 512) took the
expert-gather path, whose gathered fp32 weights are about 1.6 GB per layer graph; neuronx-cc ran
for over 20 minutes at 50-70% CPU, SSM lost the instance and it needed a reboot. With
`KILN_MOE_GATHER_MAX_PAIRS=64` (verify on the every-expert path) the run above compiled the verify
graph in 25.7 s.

**What DSA costs on trn1, and where it goes** (`tools/profile_mla.py --model zai-org/GLM-5.3 --tp 16`:
one rank of GLM-5.3 at tp=16, 4 heads, kv_lora 512, a 32 x 128 indexer replicated, index_topk
2048, FP8 weights and bf16 activations, random weights; the attention block alone, p50 of
synchronous calls):

| batch form | L (keys) | full-indexer layer | shared layer (IndexShare) | topk(2048) alone | scatter mask alone |
|---|---|---|---|---|---|
| decode B=8 | 512 | 1.67 ms (dense) | 1.15 ms (dense) | | |
| decode B=8 | 2048 | 1.88 ms (dense) | 1.05 ms (dense) | | |
| decode B=8 | 4096 | 6.00 mask / 7.21 gather | 1.33 mask / 1.81 gather | 3.19 ms | 0.48 ms |
| decode B=8 | 8192 | 9.82 mask / 10.60 gather | 2.06 mask / 2.06 gather | 6.00 ms | 0.51 ms |
| prefill C=128 | 2048 | 1.39 ms (dense) | 1.13 ms (dense) | | |
| prefill C=128 | 4096 | 12.49 ms (mask) | 6.55 ms (mask) | 3.33 ms | 6.66 ms |

So below index_topk the indexer adds about 0.6-0.8 ms per full layer (its projections and the key
write) to plain MLA, the mask beats the gather, and above it two ops dominate: `torch.topk` of
2048 over the context (3.2 ms at 4K keys, 6.0 ms at 8K, for 8 rows; it scales with keys, not
rows: 3.3 ms for 128 rows at 4K), and in prefill the scatter that turns 128 x 2048 indices into a
mask (6.7 ms, element-wise DMAs, paid again by every shared layer). GLM-5.3 has 21 full and 57
shared layers, so at 8K context the top-k alone would be about 126 ms of a decode step. A
selection kernel (NKI top-k, or a threshold found without sorting) is the next step for long
contexts; nothing else in the layer is near it. (Done without a kernel: the next section.)

## DSA top-k without torch.topk, and MTP for MLA models (2026-10-03, SDK 2.32, neuronx-cc 2.27, trn1.2xlarge)

`kiln/models/dsa_select.py` replaces `torch.topk(scores, index_topk)` + scatter with a count
search for the keep-th largest score, returned as a mask (KILN_DSA_SELECT=bisect, default). The
first form tried, a 32-round search over each score's bit pattern (an order-preserving int32 key),
does not compile: LNL rejects an in-graph dtype view (`scores.view(torch.int32)` -> "common_device
INTERNAL ASSERT FAILED ... XLANativeFunctions.cpp:1387" at the first use of the view,
`tools/probe_dsa_select.py --methods radix-cumsum`, and the failed compile ends the process). The
form that runs bisects the ORDER of the fp32 values: split at 0 while (lo, hi) holds both signs,
at the geometric mean while one end is more than twice the other, else at the arithmetic mean, 48
rounds of `count(s >= mid)`. That ends on adjacent floats (at most 33 rounds over random, integer-
tied, signed-zero, underflow-to-infinity, NEG_INF-masked and one-ulp-apart rows, CPU), so the
threshold is the exact keep-th score; the scores above it plus the lowest-index ones equal to it
are selected. A bisection of the values themselves (QSA's) cannot be exact here: the invisible keys
sit at -1e30, and the threshold is often an exact 0.

**The selection alone** (`python tools/probe_dsa_select.py --rows 8 128 --keys 4096 8192 16384`:
fp32 scores [rows, keys] in, additive mask out, keep 2048, p50 of synchronous calls; the mask
equals the stable-sort reference on random and on integer-tied scores, every case):

| rows x keys | torch.topk + scatter | bisection, whole mask | its threshold search alone |
|---|---|---|---|
| 8 x 4096 | 3.46 ms | 1.02 ms | 0.97 ms |
| 8 x 8192 | 6.30 | 1.54 | 1.48 |
| 8 x 16384 | 11.95 | 2.70 | 2.57 |
| 128 x 4096 | 8.93 | 1.34 | 0.82 |
| 128 x 8192 | 11.90 | 2.07 | 1.26 |
| 128 x 16384 | 17.82 | 3.67 | 2.15 |

(The tie fill there is the blockwise running count; cutting each row into pieces so rows x pieces
fills the 128 partitions took 8 x 8192 from 1.72 to 1.48 ms; counting against 3 or 7 candidates per
pass instead of one was slower, 1.43 / 2.10 ms against 0.97 at 8 x 4096; computing the index
scores in the same graph, as an einsum or as a matmul head sum, and gathering their keys from a
paged cache written first (`--scores --gather --write`) left 8 x 4096 at 1.08-1.22 ms.)

**Inside the layer it is a different program**, and two things decided the cost there, both found
with `tools/profile_mla.py` (GLM-5.3 at tp=16 rank shapes: 4 heads, a 32 x 128 indexer, FP8 weights,
bf16 activations, random weights; the attention block alone, p50; layer 0 runs its indexer, layer 3
is IndexShare "shared"):
- the tie fill: a cumsum (blockwise or whole-row) inside a DECODE layer made it 9.6 ms at B=8 over
  4K keys and 18.2 ms at B=32 over 8K, against 5.7 and 13.9 ms with the count search over positions;
  in a prefill chunk the blockwise cumsum is the faster one (7.1 against 10.8 ms over 8K keys).
  `KILN_DSA_TIES=auto` takes "block" for a [1, C, L] chunk and "index" otherwise.
- feeding the selection both to the attention and to the IndexShare scratch write: 10.0 ms at B=8
  over 4K keys, 16.8 ms over 8K. Writing it to the scratch and attending with what is read back
  (`KILN_DSA_STAGE=1`, default; every layer that runs its indexer, pooled ones included, not the
  MTP layer, whose graphs carry more rows than the scratch): 4.4 / 5.9 ms. The profile of the slow graph (`tools/dma_counts.py`): 50,101 tensor-engine matmul
  instructions and 51 MB of spill reload for a layer whose scores are 128 KB. The scratch must stay
  small: with a [4096, 8193] scratch (profile_layer's old default) the staged layer was 93 ms at
  B=32 over 8K keys, with [128 or 512, 8192] (what ModelRunner allocates: the largest prefill bucket
  or decode batch x (1 + spec_k) rows) 13.8 / 13.7 ms.

GLM-5.3, one rank at tp=16, full-indexer (and shared) layer, `torch.topk` path = engine-v0
(35bf2c7, same instance, same session) against this branch (`python tools/profile_mla.py --model
zai-org/GLM-5.3 --tp 16 --layers 4 --modes mask [--prefill 128 | --batch B] --pages 64 128 256 512`):

| batch form | keys | full layer, torch.topk | full layer, bisection | shared layer before | shared layer now |
|---|---|---|---|---|---|
| decode B=8 | 2048 | 1.87 ms (dense, no selection) | 1.87 | 1.05 | 1.05 |
| decode B=8 | 4096 | 6.01 | 4.48 | 1.34 | 1.19 |
| decode B=8 | 8192 | 9.83 | 6.20 | 2.15 | 2.10 |
| decode B=8 | 16384 | 17.22 | 9.39 | 3.12 | 3.36 |
| decode B=32 | 8192 | 21.0 (this branch's topk mode) | 13.8 | | 5.68 |
| prefill C=128 | 2048 | 1.37 (dense) | 1.37 | 1.12 | 1.12 |
| prefill C=128 | 4096 | 12.47 | 4.37 | 6.69 | 1.40 |
| prefill C=128 | 8192 | 19.81 | 6.66 | 6.31 | 1.73 |
| prefill C=128 | 16384 | 27.06 | 12.23 | 7.59 | 2.98 |

GLM-5.3 has 21 full and 57 shared layers: at 8K context a 128-token prefill chunk's DSA layers go
from about 21 x 19.8 + 57 x 6.3 = 775 ms to 21 x 6.7 + 57 x 1.7 = 238 ms per rank. KILN_DSA=gather
did not gain (decode B=8: 6.8 / 11.9 / 27.6 ms at 4K / 8K / 16K; its indices come from the mask by a
scatter of every position).

GLM-5.3-Flash's pooled indexer (glm5_next.block_mask, now the same bisection over 2048 pools of 4
at 8K keys, keep 512; `--model zai-org/GLM-5.3-Flash --tp 32 --layers 4 --select bisect topk`, the
pooled DSA layer, topk -> bisection): decode B=8 1.63 -> 1.45 ms (4K), 3.00 -> 2.66 (8K), 5.07 ->
4.69 (16K), B=32 at 8K 9.13 -> 8.73; prefill C=128 (before the staging above) 2.72 -> 1.40, 3.28 ->
2.12, 4.84 -> 3.87. Choosing among a quarter as many candidates, the pooled layer was never
dominated by its top-k. GLM-5.3's decode layer at B=128 over 8K keys did not compile on the
trn1.2xlarge (neuronx-cc killed for host memory, `[F137]`), so decode batches above 32 are unmeasured
here and need a bigger host.

**Device checks of the MTP path** (`tools/build_random_mla.py <family> <dir> --mtp [--index-topk 16]
[--copy-main --layers 1]`: tests/test_mtp_mla.py's checkpoints; `KILN_CC_ARGS=--auto-cast=none
python tools/check_device.py --model <dir> --dtype fp32 --tokens 160 --piecewise [--spec-method mtp
--spec-k 3]`, 8 prompts against transformers 5.15 fp32 on the host, decode bucket 8, prefill 128):

| checkpoint | MTP | result | drafts accepted | decode B=8 step |
|---|---|---|---|---|
| random GLM-5.3 config, 5 layers + MTP layer, index_topk 16 | off | 7/8 160/160; prompt 5 from token 146 (a near tie, below) | | 9.14 ms |
| same | on (k=3, index_share_for_mtp_iteration from the config) | 8/8 160/160, max \|dlogprob\| 3.6e-5 | 0 of 3768 (a random MTP layer) | 15.41 ms |
| random DeepSeek-V3.2 config, 3 layers + MTP layer, index_topk 16 | off | 7/8 160/160; prompt 1 from token 91 (a near tie, below) | | 8.34 ms |
| same | on (k=3) | 7/8, the same prompt and token | 0 of 3768 | 16.66 ms |
| random DeepSeek-V3 config, 4 layers + MTP layer (dense MLA, MoE MTP layer) | on (k=3) | 8/8 160/160, max \|dlogprob\| 3.1e-5 | 0 of 3768 | 9.29 ms |
| GLM-5.3 config, 1 layer, MTP layer = its copy (`--copy-main`) | off | 8/8 160/160 | | 4.42 ms |
| same | on (k=3, index sharing) | 8/8 160/160, max \|dlogprob\| 3.3e-5 | 633 of 1885 (34%) | 11.31 ms at 1.97 tokens per sequence per step |

The toy's speculation is a net loss (5.7 ms per token against 4.4 without it): a one-layer target
costs less than the verify and MTP graphs that come with drafting; with the target's 78 layers the
step is the target. Real acceptance needs real MTP weights (GLM-5.3 on trn2, or GLM-5.3-Flash once
its KDA state can roll back). The two divergences are the same thing and are not MTP's: both happen
with MTP off, both are identical with `KILN_DSA_SELECT=topk` (GLM checked), and Kiln on the host CPU
matches the reference 8/8 on both checkpoints (V3.2 with MTP on). At the diverging step the 16th and
17th index scores of that query are 5.96e-7 apart (6.96e-7 relative; GLM, query position 159) and
1.37e-6 apart (1.93e-6 relative; V3.2, position 95), within fp32 rounding of a 32- or 64-head sum,
so the device's summation order selects the other key. A near tie of the indexer, not the selection.

## Hyper-connection hybrids on trn1: GLM-5.3-Flash and Qwen3.8-Flash-Next (2026-10-03, SDK 2.32, neuronx-cc 2.27, trn1.2xlarge)

Random-weight configs from the real config.json files (`tools/build_random_hybrid.py <glm5_next |
qwen4_exp> <dir> --sparse`: the CPU tests' builders, 8 layers, 128 hidden, the real block geometry,
the models' own tokenizers and vocabularies, index_topk / indexer_budget 16 so every bucket of the
run takes the pooled / block selection; the dense rows are built without --sparse), `tools/check_device.py --model <dir> --dtype fp32 --tokens 64
--piecewise [--tp 2]` with `KILN_CC_ARGS=--auto-cast=none` against transformers 5.18.0 fp32 on the
host (installed beside the SDK venv with `pip install --no-deps --target`, on PYTHONPATH), decode
bucket 8, prefill bucket 128, pages 4 and 32 of 32 tokens:

| model | tp | result | decode B=8 step p50 |
|---|---|---|---|
| GLM-5.3-Flash, 6 KDA + 2 pooled-DSA layers, mHC, clamped SwiGLU, MoE + shared | 1 | 8/8 prompts 64/64 tokens, max \|dlogprob\| 3.9e-5 | 7.94 ms |
| same | 2 | 8/8 64/64, max \|dlogprob\| 3.8e-5 | 8.37 ms |
| Qwen3.8-Flash-Next, 6 GDN + 2 QSA layers (indexer 16 heads), PLE, gated residual, MoE + gated shared | 1 | 8/8 64/64, max \|dlogprob\| 9.6e-6 | 7.35 ms |
| same | 2 | 8/8 64/64, max \|dlogprob\| 9.6e-6 | 7.87 ms |
| GLM-5.3-Flash, dense regime (index_topk 2048: plain NoPE MLA) | 1 | 8/8 64/64, max \|dlogprob\| 3.6e-5 | 6.83 ms |
| Qwen3.8-Flash-Next, dense regime (indexer_budget 2048, the real 4 indexer heads) | 1 | 8/8 64/64, max \|dlogprob\| 6.4e-6 | 6.75 ms |
| Qwen, first version (sparse, indexer 4 heads, torch.topk, token gathers) | 1 | 8/8 64/64; three prompts at max \|dlogprob\| 3.5e-3, 9.7e-3, 2.3e-2 | 122 ms |

- **A token-granular KV gather in a graph that also selects blocks made the QSA decode layer 58 ms**
  (`tools/probe_hybrid.py qsa <checkpoint>`, one QSA layer, B=8 over 128 keys, the layer's
  attention block alone): 57.8 ms with the selection, 0.32 ms without it. The probe's variants, all
  with token gathers: the selection's pieces alone are fast (block scores 0.39 ms; the mask returned
  without the attention 0.25 ms); a mask built from the query alone (no cached keys) and fed to the
  attention was slow through torch.topk or torch.sort (57.6 / 58.1 ms) and fast through a mean, rounds
  of amax or the bisection below (0.61 / 0.63 / 0.65 ms); the real selection (cached keys) through
  the bisection was slow again (57.8 ms). With whole-page gathers (`KILN_GATHER=page`) every variant
  was fast: the layer 1.81 ms with torch.topk, 0.44 ms with the bisection. GLM's pooled DSA layer,
  with the same top-k but MLA's latent gathers, was 0.91 ms (0.33 ms dense) with token gathers. Kiln's
  QSA gathers pages whenever its bucket is sparse (`qwen4_exp._load_pages`) and picks blocks by
  bisection (`KILN_QSA_SELECT`, default `bisect`; `glm5_next.block_mask`); the cause inside
  neuronx-cc is not known.
- **A bisection replaces the top-k.** 32 halvings of [min, max] of the block scores, one count per
  round, find the keep-th largest; the blocks above it and the first of those tied with it are kept.
  Exactly torch.topk's set unless two boundary scores are within 2^-32 of the range; at an exact tie
  both keep `keep` blocks, chosen differently (CPU torch.topk is not lowest-index-first among ties:
  1465 of 2000 random tie cases differed, measured on the host). QSA's score is a sum of 4 ReLUs, exactly 0 for about one
  block in 16, so the real model's 4-head indexer does tie; the random check configs use 16 heads.
- `torch.topk(..., sorted=False)` gave a different selection on the device than on the CPU (the
  layer output 0.94 apart in the probe); Kiln never calls it.
- The three larger \|dlogprob\| of the first Qwen run, with every token still equal, are exact
  ties: on that checkpoint (4 indexer heads) the reference's greedy texts put a QSA query's 4th and
  5th best complete blocks at exactly the same score (0) in prompts 1, 3 and 5 and in no other
  (one query each), the three prompts that deviated; the device's top-k kept a different one of the
  tied blocks than the host reference. The same weights in the dense regime (indexer_budget 2048)
  ran 8/8 64/64 at max \|dlogprob\| 6.4e-6 (below), and the CPU engine with torch.topk matched the
  reference on all eight (at most 2.1e-6), its tie-break happening to coincide.
- Nothing else of these models needed a device-specific form: the 20 Sinkhorn iterations on [T, 4, 4],
  the clamped SwiGLU (`clamp` with float literal bounds), the PLE gate's `abs().clamp_min(1e-6)
  .sqrt() * sign()`, its dilated conv over a state-pool history, the n-gram table lookup (int64 ids
  hashed on the host) and the NoPE MLA (no zero-width tensor is concatenated) compiled and matched as
  written. Compile on a cold cache: GLM prefill 135 s and decode 54 s per bucket, piecewise.

## Attention TP smaller than the world TP (2026-10-03, SDK 2.32, neuronx-cc 2.27.5334, trn1.2xlarge)

`DecoderForCausalLM` runs the token mixers at attention TP `attn_tp` (a divisor of tp, replicated
over tp / attn_tp groups of consecutive ranks) and the MLP / experts at tp (DESIGN.md L3). How LNL
treats the attention group's collective, read in libtorch_neuronx_lite 2.11.0.1.0.1284
(`overrides/xla_collectives.py`, `compile/cache.py`):

- `funcol.all_reduce(x, "sum", subgroup)` lowers to `xm.all_reduce(..., groups=[ranks])` with ONLY
  this rank's group as the replica groups (`_get_replica_groups_from_group_name` returns
  `[dist.get_process_group_ranks(group)]`), not the full partition of every group.
- The NEFF cache key hashes the resolved replica groups beside the FX graph
  (`create_cache_hash`: "Include resolved replica group ranks so that different process group
  configurations produce distinct cache keys"). So with 1 < attn_tp < tp every group compiles its
  own NEFF of each graph that holds an attention all-reduce (tp / attn_tp parallel compiles, by the
  first rank of each group; the other ranks of a group wait on its lock), and no group can load
  another group's NEFF by accident. attn_tp = tp keeps the world group, attn_tp = 1 has no attention
  collective at all; in both cases every rank traces one graph.
- attn_tp = tp is plain TP to the byte: engine-v0 74a74f6 (before attention TP) re-ran Qwen/Qwen3-0.6B
  fp32 piecewise at tp=2 against the compile cache this branch had just filled at attention TP 2 and
  hit all 12 graphs (0 misses, `atp-base-q06-f32.log`).

Measured with the two NeuronCores of a trn1.2xlarge at tp=2, so attention TP 2 (the world group)
against 1 (attention replicated on both cores, its all-reduce gone, each core holding every head and
twice the KV): `tools/check_device.py --tp 2 --attention-tp {2,1}`, 8 prompts, decode bucket 8,
prefill bucket 128, pages 4 and 32 of 32 tokens; fp32 rows with `--dtype fp32` and
`KILN_CC_ARGS=--auto-cast=none`; "a2 vs a1" from `--out-json` of the two runs (generated ids and
chosen-token logprobs):

| model | form | attn_tp 2 | attn_tp 1 | a2 vs a1 | decode B=8 step p50, attn_tp 2 / 1 |
|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B (real), fp32 | piecewise | 8/8 32/32 vs transformers, max \|dlogprob\| 2.7e-5 | 8/8 32/32, 2.1e-5 | 256/256 tokens equal, max \|dlogprob\| 7.6e-6 | 14.60 / 15.43 / 15.83 against 16.95 / 17.00 / 19.83 ms (3 runs each; the 19.83 run had p90 34 ms) |
| Qwen/Qwen3-0.6B (real), bf16 | one graph | 6/8 to 32 tokens vs transformers fp32 (5 and 10 tokens at margins 0.024, 0.012) | the same 6/8, same divergences | 256/256 equal, max \|dlogprob\| 0.10 | 8.57 / 8.66 / 8.94 against 9.24 / 9.24 / 9.25 ms (3 runs each) |
| random MiMo-V2 (`tools/build_random_mimo.py`: SWA window 8 + sinks, 4 / 2 and 4 / 4 heads, MoE), fp32 | one graph | 8/8 32/32 vs Xiaomi's modeling code | 8/8 32/32 | 256/256 equal, 1.9e-6 | 3.10 / 3.30 ms |
| random Qwen3.8-Flash-Next (`tools/build_random_hybrid.py qwen4_exp --sparse`: 6 GDN + 2 QSA layers, PLE, MoE), fp32 | piecewise | 8/8 64/64 vs transformers 5.18 | 8/8 64/64 | 512/512 equal, 1.9e-6 | 7.85 / 8.17 ms |

On one chip attention TP 1 is slower (4-10%, middle run of each row): the attention all-reduce it
saves is cheap there (about 26 us each, "Collectives across chips"), while each core computes every
head and reads twice the KV. Attention TP below tp pays for itself only where it is the price of loading the model at
all (heads that do not divide tp), or across chips, where a collective costs milliseconds; which
of the two wins at tp=16 / 32 is not measured. Neither is the shape it takes there: several groups
of 8 ranks (4 chips each) all-reducing at once in the same graph as a world all-reduce. A
trn1.2xlarge cannot run it (its only proper subgroups have one rank);
`KILN_PROBE_COLLECTIVES=1 tools/profile_layer.py --allreduce --ranks 32` on a trn1.32xlarge times
every group of 2 and of 8 reducing at once and checks each group's sum (written with this change, not run yet).

Commands (logs and `--out-json` files in `s3://<your-bucket>/logs/kiln-atp-trn1/`):
`tools/check_device.py --model Qwen/Qwen3-0.6B --tp 2 --attention-tp A [--dtype fp32 --piecewise]
--tokens 32`; `--model <rand-mimo> --tokens 32 --reference-json <rand-mimo>/reference.json`;
`--model <rand-q4> --tokens 64 --piecewise --reference-json <rand-q4>/reference.json` with
transformers 5.18.0, huggingface_hub 1.33.0 and tokenizers 0.23.2 installed with `pip install
--no-deps --target` and put first on PYTHONPATH (the SDK venv's transformers 5.15 cannot parse a
`qwen4_exp_text` config, and transformers 5.18 needs the newer hub: "cannot import name 'httpx'
from 'huggingface_hub.utils'"); the random references were written on the host by the builders /
`tools/hf_reference.py` (transformers 5.15 for MiMo, 5.18 for Qwen). Repeat step timings with
`--no-reference`. Compile per bucket, cold cache: Qwen3-0.6B bf16 one graph 92-97 s, fp32 piecewise
7-19 s; the hybrid 27-39 s.

## Concurrent attention-subgroup all-reduces at 32 ranks (2026-10-03, SDK 2.32, trn1.32xlarge)

`KILN_PROBE_COLLECTIVES=1 tools/profile_layer.py --allreduce --batch 4 --ranks 32` (engine-v0
b49d6cd), the case feat/attn-tp could not run on a trn1.2xlarge: every attention group of the
32-rank world all-reduces at the same time, through `tp.attention_group`. All 16 groups of 2 and
all 4 groups of 8 sum correctly (max |err| 1.6e-2 and 4.9e-2 on bf16 sums of 2 and 8 standard
normals, within one bf16 ulp). Chained per launch: all groups of 2 at once **0.19 ms**, all groups
of 8 at once **0.94 ms**, against 4.67 ms for one world all-reduce in the same run (a group of 8
alone, the other ranks idle: 2.49 ms). So a graph whose collectives stay inside chips, or inside
4-chip groups that all run together, does not pay the ~5 ms that a world collective costs.


## DP attention on trn1 (2026-10-03, SDK 2.32, neuronx-cc 2.27.5334, trn1.2xlarge)

`--dp-attention N` (DESIGN.md L3, `engine/dp.py`, `models/decoder.py` DecoderForCausalLM: DP attention):
N groups of tp / N ranks serve their own requests from their own KV pages and state rows; a mixer
takes its group's rows of the group-major batch and its head partial goes back zero-padded into
an all-reduce over the world, the MLP / experts run over every group's rows. Measured with the two
NeuronCores of a trn1.2xlarge, tp=2, so dp 2 is attention TP 1 (each core holds every head for its
own requests) against dp 1 (attention TP 2, heads split, every core holding every request).

**dp_attention=1 is the old engine to the byte.** engine-v0 9758932 and then this branch at dp 1
ran on the same box against one LNL cache (`/root/.cache/neuron_libtorch`): Qwen/Qwen3-0.6B fp32
piecewise, 6 of 6 cache keys identical, 0 misses, 0 compiles (`grep 'Compilation cache key'`,
`'Local cache miss'`, `'Compiling...'` in the two logs); a random MiMo-V2 in one graph, 2 of 2. The
first attempt missed one key, and the reason is worth knowing before touching any graph: prep
prefill read its chunk length from `positions.shape` instead of `input_ids.shape`, which changed
the ORDER in which the traced function first touched its inputs, hence the order of the FX
placeholders, hence the FX text LNL hashes for the key (`fxgraph.txt` of the two keys differed in
exactly those two placeholder lines). The prep functions now take every shape from `input_ids`.

| model | form | dp 1 vs reference | dp 2 vs reference | dp 1 vs dp 2 | decode B=8 p50, dp 1 / dp 2 |
|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B (real), fp32 | piecewise | 8/8 32/32 vs transformers, max \|dlogprob\| 2.7e-5 | 8/8 32/32, 3.5e-5 | 256/256 tokens equal, max \|dlogprob\| 7.6e-6 | 14.63 / 16.07 ms |
| random MiMo-V2 (`tools/build_random_mimo.py`: SWA window 8 + sinks, MoE), fp32 | one graph | 8/8 32/32 vs Xiaomi's code | 8/8 32/32 | 256/256 equal, 1.9e-6 | 3.73 / 3.40 ms |
| random GLM-5.3 MLA / DSA (`tools/build_random_mla.py glm_moe_dsa --index-topk 16`: IndexShare, sparse past 16 keys), fp32 | piecewise | 8/8 64/64 vs transformers, 2.0e-5 | 8/8 64/64, 1.9e-5 | 512/512 equal, 2.9e-6 | 7.78 / 6.58 ms |
| Qwen/Qwen3-0.6B, fp32, dp 2 with `--spec-method ngram --spec-k 3` (verify graphs of both groups' drafts) | piecewise | | 70 of 138 drafts accepted | 256/256 equal to dp 1 without speculation, 7.6e-6 | |

Qwen/Qwen3-0.6B bf16, one graph, decode step p50 at total batch B (dp 2: B / 2 per group), two runs
each:

| B | dp 1 | dp 2 |
|---|---|---|
| 8 | 8.61 / 8.65 ms | 8.27 / 8.31 ms |
| 32 | 14.93 / 14.84 ms | 13.47 / 13.47 ms |
| 64 | 24.08 / 23.62 ms | 23.57 / 23.96 ms |

On one chip this is not where DP attention earns its keep: a GQA model whose KV heads divide tp
holds the same KV bytes per rank either way (half the heads of every request, or every head of
half the requests), and the per-layer collective count is the same (attention TP 2's o_proj
all-reduce, or DP's zero-padded one). What changes is that each core reads all the attention
weights for half the tokens; bf16 came out 0-10% faster, fp32 (twice the weight bytes) 10% slower.
The reason for DP attention is the MLA latent, which TP replicates on every rank.

**The exchange: one zero-padded all-reduce, against SGLang's all-gather forms.**
`KILN_PROBE_DP=2 tools/profile_layer.py --allreduce --ranks 2 --batch T` (every result checked
against its group sums, max |err| 0 for all three): per chained launch, [T, 4096] bf16 per group,

| T per group | Kiln: zero-padded [2T, H] all-reduce | group reduce-scatter + all-gather (SGLang MAX_LEN) | group all-reduce + all-gather | one plain [T, H] all-reduce | null graph |
|---|---|---|---|---|---|
| 4 | 0.200 ms | 0.193 ms | 0.197 ms | 0.196 ms | 0.107 ms |
| 64 | 0.259 ms | 0.259 ms | 0.263 ms | 0.257 ms | |
| 512 | 0.638 ms | 0.583 ms | 0.582 ms | 0.583 ms | |

(at attention TP 1 the two all-gather forms have no group collective, so both are one all-gather).
Equal at decode sizes; the all-reduce moves about twice the bytes and shows it at 512 rows a
group, 10% on one chip. Kiln keeps it because it needs no subgroup: a subgroup collective's replica
groups enter the LNL cache key, so every attention group compiles its own NEFF of every graph
(see "Attention TP smaller than the world TP"), where the zero-padded all-reduce is one NEFF for
all 32 ranks. At 32 ranks a graph holding a cross-chip collective costs ~5 ms per execution
whatever it holds ("Collectives across chips"), so the latency of the two forms is expected to be
the same there; the byte difference is measured by the same probe with `KILN_PROBE_DP=8 --ranks
32` (not run yet).

**What it buys GLM-5.3-Flash** (arithmetic from config.json and the model's own shapes,
`DecoderForCausalLM(..., tp_size=32, dp_attention=N, keep_fp8=True)` on the meta device; not a
device measurement): KV is 16,896 bytes per token per rank (11 DSA layers x (512 latent + 256
pooled indexer key) x 2 bytes, replicated over the group), so 128 sequences of 8448 tokens need

| dp_attention | attention TP | KV per rank | KDA state per rank | mixer weights per rank |
|---|---|---|---|---|
| 1 | 32 | 17.02 GiB | 0.55 GiB | 0.62 GiB |
| 4 | 8 | 4.25 GiB | 0.57 GiB | 1.56 GiB |
| 8 | 4 | 2.13 GiB | 0.58 GiB | 2.80 GiB |
| 16 | 2 | 1.06 GiB | 0.62 GiB | 5.30 GiB |

dp 8 is the smallest total (5.5 GiB beside the experts, against 18.2 at dp 1). The 32-rank run on
kiln-mimo-trn1 (not run by this change; correctness first, then the exchange probe, then the
sweep):

```sh
KILN_BOX_SRC=/opt/kiln/src-dpa infra/fleet.sh bg kiln-mimo-trn1 tools/check_ppl.py --model zai-org/GLM-5.3-Flash \
    --tp 32 --dp-attention 8 --piecewise --kv-cache-gb 1.0          # dp 1 gave mean prompt logprob -2.141
KILN_BOX_SRC=/opt/kiln/src-dpa KILN_PY_ENV="KILN_PROBE_DP=8" infra/fleet.sh bg kiln-mimo-trn1 \
    tools/profile_layer.py --allreduce --ranks 32 --batch 16
KILN_BOX_SRC=/opt/kiln/src-dpa KILN_PY_ENV="KILN_MOE_KERNEL=nki KILN_PIECEWISE_MOE_GROUP=12" infra/fleet.sh bg \
    kiln-mimo-trn1 bench/serve_sweep.py --model zai-org/GLM-5.3-Flash --tp 32 --dp-attention 8 --piecewise \
    --overlap --concurrency 16 32 64 128 --prefill-tokens 512 --kv-cache-gb 2.5
```

`--prefill-tokens 512` is the engine total (64 per group and step, so the experts see 512 rows,
as at dp 1); `--kv-cache-gb` is per rank and per group pool, at least 2.13 for 16 sequences of 8448
tokens per group.

Commands (logs and `--out-json` files in `s3://<your-bucket>/logs/kiln-dpa-trn1/` and
`/opt/kiln/work` on the box): `tools/check_device.py --model <m> --tp 2 --dp-attention {1,2}
[--dtype fp32 --piecewise] --tokens <n> --out-json <f>` with `KILN_CC_ARGS=--auto-cast=none` for
fp32 (Qwen3-0.6B: reference = transformers on the box; MiMo: `--reference-json
<rand-mimo>/reference.json`); the bf16 step times with `--no-reference --max-num-seqs B --bench-steps
64 --max-model-len 1024 --tokens 16`.

## A compile cache copied to another host needs the NKI kernel binaries too (2026-10-03, SDK 2.32)

A fresh trn1.32xlarge that pulled only the graph entries of LNL's compile cache
(`~/.cache/neuron_libtorch/neuron/compile_cache/<hash>/`) from another box failed its first
graph that calls an NKI kernel: neuronx-cc "[NCC_EVRF059] Kernel file
'/var/tmp/nki-intermediate-cache/nki_0.6.0+.../kiln.kernels.moe_dedupe.kiln_moe_dedupe_v8_....colz'
referenced by AwsNeuronCustomNativeKernel instruction does not exist on the host", exit 70; the
dying compile-lock holder ("Lock holder process died - cache invalidated") then took every rank
of the 32-rank engine down, and rank 0 surfaced only as a gloo "Connection closed by peer". The
NKI compiler writes kernel binaries OUTSIDE the LNL cache, under /var/tmp/nki-intermediate-cache,
and LNL's own NKI results (`<cache>/nki/<key>.json`) have no completion marker, so neither moved.
`kiln/compile_cache.py` now syncs both (kernels before any graph marker); verified with a push from
kiln-dev-trn1 and a pull into an empty root: 177 of 177 kernels and 105 of 105 NKI results.

## GLM-5.3-Flash on trn1.32xlarge: HBM per core decides the DP-attention layout (2026-10-03, SDK 2.32)

Four `bench/serve_sweep.py` runs at tp=32, `--dp-attention 8` (attention TP 4), 8192 in / 256 out,
failed while loading graphs after warmup compiles: `Could not load the model status=4
message=Allocation Failure`, NMGR "Failed to stage graph to NeuronCore", with 3 decode buckets
(concurrency 16 / 32 / 64), then with one decode bucket per engine (concurrency 64 and 32 alone),
KV pool 1.3 to 2.5 GB. `/opt/aws/neuron/bin/neuron-monitor` (not on the SSM shell's PATH) sampled
once per second, core 0, at concurrency 64 with KV 1.3 GB: **tensors 13.93 GiB right after the
weights loaded** (weights, KV pool, KDA state rows), then model_code 0.17, shared scratchpad 0.19,
constants 0.02 while graphs loaded, peak total 14.31 GiB of the 16; DMA-ring reservations are not
in that breakdown. Under DP attention each group holds its own copy of the attention-side weights
(KDA and DSA mixers), M / attention-TP per rank, about 12.8 GB in total for this model (the DP
agent's estimate of +2.8 GiB at DP 8 over plain tp=32 matches the measured total), while the KV a
rank holds is that of its own group's sequences. Per core, from the measured 13.93: experts and the
rest about 9.4 GiB; DP 8 mixers 3.2 + KV for 8 sequences x 8448 tokens 1.0 (11 DSA layers x (512
latent + 128 indexer key) x 2 B = 14 KB per token); DP 4 mixers 1.6 + KV for 16 sequences 1.9; DP 2
0.8 + 3.8. DP 4 leaves the most room for graphs, so that is what the next runs use.


## Compile farm: neuronx-cc on CPU hosts, and capturing graphs without a NeuronCore (2026-10-03, SDK 2.32, neuronx-cc 2.27.5334)

Tools: `tools/compile_farm.py` (`capture`, `enqueue`, `work`, `check`, `status`, `bench`, `compile`,
`fetch`, `shapes`), `kiln/compile_farm.py`, `kiln/capture.py`, `tools/ncc_options.py` (neuronx-cc's
real option table), `tools/hlo_diff.py` (two graph.hlo files modulo op metadata). Every number
below is from a Deep Learning AMI Neuron (Ubuntu 24.04, SDK 2.32) host in us-east-2, BF16/FP8
GLM-5.3-Flash graphs unless named otherwise; spot prices were read at 22:41 UTC.

**A compile is a CLI call on an HLO file, and a CPU host builds the same NEFF.** An LNL entry
`<cache>/<key>/` holds graph.hlo, `.artifact_metadata_v0.json` and command.txt; replaying
command.txt (`tools/compile_farm.py compile|bench`) runs on any DLAMI host. A NEFF is a 1024-byte
header plus a tar.gz (`tail -c +1025 graph_<key>.neff | tar xz`). The farm's NEFF and the device
box's NEFF of one key differ ONLY in info.json (its "name" is the output path) and the debug_info
`*.dbg` files: Qwen3-0.6B tp=2, 6 of 6 graphs (40-44 members each, farm c8i.48xlarge vs
trn1.2xlarge); GLM-5.3-Flash tp=32 prefill 12-layer group f12431f1 (NKI MoE kernel, 265.9 MB NEFF),
farm m8i.48xlarge vs kiln-g1-trn1: 66 of 66 members, 0 differences outside those.

**Cost of one graph** (default flags, the GLM-5.3-Flash tp=32 layer groups of kiln-g1-trn1's DP-8
sweep; trn1.32xlarge minutes are command.txt to NEFF mtime during the sweep; peak = largest sum of
the neuronx-cc process tree's RSS, sampled every second; CPU = user + system of the whole tree):

| graph (graph.hlo) | kind | trn1.32xlarge | c8i.48xlarge alone | m8i.48xlarge, 7 at once | peak RSS | CPU s | NEFF |
|---|---|---|---|---|---|---|---|
| 66437b34 (3.10 MB) | prefill, 12 layers (3 dense + 9 MoE), 2048 rows | 45 min | - | 2179 s | 174.9 GB | 13418 | 265.9 MB |
| 7dfa061a (2.93 MB) | prefill, 12 layers | 43 min | - | 2146 s | 174.8 GB | 13502 | 265.6 MB |
| 2f163214 (1.85 MB) | decode, 12 layers | 8 min | 375.7 s | 418 s | 11.4-11.8 GB | 1290-1374 | 21.7 MB |
| b561260a (1.84 MB) | decode, 12 layers | 7 min | - | 400 s | 11.0 GB | 1222 | 19.2 MB |
| a9c00d50 (1.68 MB) | decode, 12 layers | 7 min | - | 394 s | 11.3 GB | 1349 | 21.4 MB |
| 5d3b5bfc (1.36 MB) | decode, last group | 5 min | - | 287 s | 8.8 GB | 954 | 15.9 MB |
| 1b18e005 (1.24 MB) | decode | 5 min | - | 272 s | 8.4 GB | 956 | 15.7 MB |

A big graph keeps ONE walrus_driver process busy on about 6 cores for most of its life (CPU s /
wall = 6.2), and its RSS climbs late: 40-48 GB at 10 minutes, 174.9 GB at the end. The 45-53 GB
reported from the device box was a mid-compile reading. So host memory, not cores, limits how
many big graphs a host compiles at once. Commands: `python tools/compile_farm.py bench --keys <k,...>
[--repeat N] [--extra=<flags>]`.

**Compiler flags.** Sources: the CLI reference
(awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/index.html:
`--optlevel` 1/2/3, default 2, "-O1 ... minimizing compile time", "-O3 ... longer compile times";
`--model-type generic|transformer|unet-inference`; `--auto-cast` default none). The installed
compiler's own table (`python tools/ncc_options.py`, which reads
the driver's argparse because `--help` crashes on a `%` in a help string) adds hidden options,
among them `--jobs/-j`, `--num-parallel-jobs` (default 1), `--internal-max-instruction-limit`
(default 0, meaning walrus's own limit, `walrus_driver --help`: "--max-instruction-limit ...
Maximum allowed number of unrolled instructions") and `--internal-backend-options` (vllm-neuron
passes `--enable-verifier=false` through it). Decode group 2f163214 on c8i.48xlarge (7 variants at
once, each ~4 cores):

| flags | compile | peak RSS | NEFF |
|---|---|---|---|
| default (-O2) | 401 s (375.7 s alone) | 11.4 GB | 21.7 MB |
| -O1 | 114 s | 4.4 GB | 12.3 MB |
| -O1 --model-type=transformer | 97 s | 3.9 GB | 11.8 MB |
| --model-type=transformer | 294 s | 8.8 GB | 20.2 MB |
| -O3 | 396 s | 11.7 GB | 21.7 MB |
| --internal-backend-options=--enable-verifier=false | 387 s | 11.3 GB | 21.7 MB |
| --enable-fast-loading-neuron-binaries | 391 s | 11.5 GB | 160.7 MB (uncompressed) |
| --jobs 8 (alone) | 395.5 s, CPU 524 s instead of 1374 | 10.3 GB | 21.6 MB |
| --num-parallel-jobs 8 (alone) | 375.6 s | 11.4 GB | 21.7 MB |

Prefill group 66437b34 (m8i.48xlarge): default 2179 s / 174.9 GB; `--model-type=transformer`
2083 s / 151.8 GB / NEFF 215.5 MB; verifier off 2197 s / 177.6 GB; **-O1 FAILS** after 203 s,
`[NCC_EBVF030] Instructions generated by compiler 38363223 exceeds the typical limit of 5000000`.
So -O1 cuts a decode group's compile 3.5x but cannot build a big prefill group, and nothing except
a smaller graph shortens the long pole. Runtime: no difference measured where it could be measured.
Qwen3-0.6B tp=2 decode on trn1.2xlarge (`KILN_CC_ARGS=<flags> python bench/serve_sweep.py --model
Qwen/Qwen3-0.6B --tp 2 --piecewise --concurrency 8 --requests 32 --input-len 512 --output-len 128
--decode-buckets 8 --page-buckets 48 --prefill-buckets 512 --kv-cache-gb 1 --warmup`, two runs
each) gave ITL p50 30.1 / 30.1 ms (default), 30.1 / 30.3 (-O1), 30.2 / 30.1 (transformer) and 30.3
(both). One GLM-5.3-Flash MoE block, NKI kernel, tp=32 rank shapes over 2 live ranks
(`KILN_MOE_KERNEL=nki python tools/profile_layer.py --model zai-org/GLM-5.3-Flash --tp 32 --ranks 2
--layers 4 --batch 8 --what mlp`) gave p50 0.598 / 0.521 / 0.516 ms, with the null graph at 0.165 /
0.144 / 0.145, so the spread is launch noise. profile_layer's `groups` mode does not take hyper-connection states, so the 12-layer groups were
timed through the engine instead.

**The decode step at tp=32 (kiln-g1-trn1, trn1.32xlarge, 2026-10-04 00:00-00:20 UTC).**
GLM-5.3-Flash, real weights, `--tp 32 --dp-attention 4 --max-num-seqs 32 --decode-buckets 8
--page-buckets 264 --kv-cache-gb 1.0`, `KILN_MOE_KERNEL=nki KILN_PIECEWISE_MOE_GROUP=12`, and
`KILN_PROFILE_PIECES=1 KILN_PROFILE_EXEC=1 KILN_CC_ARGS=<flags> python tools/time_decode.py --steps 64 --skip 8
-- <those serve_sweep args>`: p50 of 64 decode steps of random token ids (32 rows into the MoE),
each piece timed synchronously on rank 0. Every graph came from the farm (`KILN_COMPILE_FARM`,
`NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`, 0 compiles on the device):

| flags | group 0 (3 dense + 9 MoE) | groups 1, 2 (12 MoE) | group 3 (11 MoE) | step p50 (min) |
|---|---|---|---|---|
| default, seed 0 | 43.2 ms | 39.6, 39.7 | 30.2 | 162.0 ms (156.4) |
| default, seed 1 | 43.7 | 39.9, 40.0 | 30.4 | 163.7 (160.4) |
| --model-type=transformer, seed 0 | 39.0 | 35.4, 35.3 | 26.7 | 146.8 (141.1) |

`--model-type=transformer` takes 9-10% off every layer group and off the step, beyond the 1%
spread between the two default runs (one transformer run; output correctness not checked), and it
also compiles faster. `-O1` is out for these graphs: decode group 0 failed to compile, "[NCC_INAS001]
Error namespace neuronxcc does not exist or error code IGAA901 does not exist" (58 s, internal
compiler error), and the prefill groups fail NCC_EBVF030.

**How many at once.** Copies of decode group 2f163214 on c8i.48xlarge (192 vCPU, 371 GB):

| concurrent | wall | graphs / h | host peak used | CPU mean / max |
|---|---|---|---|---|
| 1 | 375.7 s | 9.6 | 18.2 GB | 1.9 / 23 % |
| 12 | 427.9 s | 101 | 129.4 GB | 17 / 85 % |
| 24 | 458.2 s | 188.6 | 248.4 GB | 30 / 92 % |
| 48 at -O1 | 187.8 s | 920 | 166.7 GB | 64 / 100 % |

At default flags memory runs out (about 30 at 10.5 GB each) before the cores do. -O1 at 48 at once
saturates the CPU. Per dollar, spot: 24 decode groups at once on c8i.48xlarge ($2.78/h) is 68
graphs/h/$. A trn1.32xlarge compiling them one at a time (LNL's lock) is 7.4 graphs/h, 3.4/$ at
$2.15 spot or 0.34/$ on demand, and all 32 NeuronCores sit idle meanwhile. Big prefill groups
(175 GB each, about 36 min): m8i.48xlarge (743 GB, $3.33/h) holds 4, 6.6 graphs/h = 2.0/$.
c8i.48xlarge holds 2 (1.2/$). A memory-heavy type (r8i.48xlarge 1.5 TB, $4.68-5.42/h) would hold 8;
not measured.

**Prefill layer groups and the instruction limit (config A of the DP-4 runs:** `--tp 32 --dp-attention 4
--prefill-buckets 512`, 2048 rows into the MoE, `KILN_PIECEWISE_PREFILL_MOE_GROUP=P`; counts read by
compiling with `--internal-max-instruction-limit=1`, where the verifier reports every module's count
and stops). P=1: 0.68-0.71M instructions; dense 349 s / 14.2 GB, MoE 504 s / 20.6 GB and 557 s / 23.6 GB.
P=2: 1.50-1.54M (31% of the limit); dense pair 729 s / 25.9 GB, MoE pairs 1002 s / 42.9 GB and
1070 s / 43.3 GB. P=3: FAILS. macCnt 4.4-4.9e11 (the compiler's modular threshold is 2e11) sends the
graph through the modular split, and module 1 comes out at 38,364,773 instructions (module 2
9,222,547), the same ~38.4M module the 12-layer groups produced. The count jumps at the split
instead of growing with P. The 12-layer prefill groups of kiln-g1-trn1's DP-8 sweep failed the same
way on the farm (groups 2 and 3, all MoE: modules of 39.7M / 38.4M / 10.1M) while group 1 (3 dense
layers) compiled. The device box would have reached those failures about an hour into its
warmup; the farm found them 4 minutes after capture. With the grouped-GEMM prefill MoE kernel
(`KILN_MOE_PREFILL_KERNEL=nki`, same shapes, `--max-num-seqs 32 --kv-cache-gb 1.0`) every group is
one module and fits: P=4, 4 MoE layers 2,361,817 instructions, 3 dense + 1 MoE 3,075,600 (the
largest: the dense layers, not the MoE), the last group 469,564; P=3: 1.70M / 1.74M MoE, 2.42M
dense x 3.

**Capture without a NeuronCore** (`kiln/capture.py`; `python tools/compile_farm.py capture
--shape-dir <dir> -- <serve_sweep args>`, one process per TP rank on the meta device, a fake process
group of the real world size, shapes from safetensors headers, `NEURON_PLATFORM_TARGET_OVERRIDE=trn1
NEURON_LIBTORCH_CPU_COMPILE=1`). GLM-5.3-Flash tp=32: 32 ranks in 3:15-3:19 on m8i.24xlarge, 1.6 GB
per rank process; rank 0 alone 2:36 (model build 39.5 s, warmup trace 87.5 s). DP-8 sweep config:
10 keys, the same on all 32 ranks; 7 were already in kiln-g1-trn1's cache under the same key, with
graph.hlo identical modulo op metadata (`tools/hlo_diff.py`); the other 3 were the graphs g1 had not
reached. DP-4 configs A and B: 17 keys each, every key on all 32 ranks. MiMo-V2.6-Flash-RL tp=32: 10 keys on
all 32 ranks, but 12 min per rank to build the shard, because the loader unpacks MXFP4 even when the
values are zeros; the farm compiled rank 0's graphs while the other ranks were still building. Two traps, both measured on
Qwen3-0.6B tp=2 against a trn1.2xlarge:

- **The process group must be created before libtorch_neuronx_lite is imported**, as a device rank
  does (engine/tp.py init_rank, then build_shard imports LNL). With LNL first and then a process
  group (fake or gloo), dynamo recorded `torch.topk(x, k, dim=-1)` as a call of a Python function
  (`_VariableFunctionsClass.topk`) that has no schema, so the default-kwarg strip could not touch
  it, and `torch.argmax(x, dim=-1, keepdim=True)` kept its keywords. With the group first, both are
  LNL's override spelling (topk with largest / sorted / out spelled out, argmax positional), which is
  what the device hashes. The post graph got another key until the order was fixed.
- **LNL's collective lowerings must be registered before lowering** (compile() imports
  overrides.neuron_collectives / xla_collectives; capture_backend does not). Without them torch_xla
  lowered `_c10d_functional.all_reduce` itself: no `replica_groups`, `constrain_layout: true`. The
  HLO was written under the very key the device computed, because the key hashes the FX graph and
  not the HLO, so the farm would have built a different NEFF for a correct-looking key.
  `tools/hlo_diff.py` caught it; `KILN_HASH_DUMP=<dir>` writes the exact string LNL hashes for each
  key, to diff a capture against a device run.
- **A graph chosen from weight VALUES at load is outside what shapes can give.** The capture loads
  zeros; kernels/moe_prefill.py check_blob picks its dequantize-first path (`dq`) from whether the
  scales are equal within a block, which zeros satisfy, the same answer its docstring gives for
  GLM-5.3-Flash at tp=32. A device run with `NEURON_LIBTORCH_ASSERT_CACHE_HIT=1` is the check.

**End to end on a device** (trn1.2xlarge, Qwen3-0.6B tp=2, `--piecewise --decode-buckets 4
--page-buckets 8 --prefill-buckets 64 --warmup`): capture on c8i.48xlarge (8.6 s), farm compile (6
graphs, 5.8 s at 6 at once), push, pull into an empty cache, then the sweep with
`NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`: every key was a local hit, 0 "Compiling...", bucket warmup
2.0 s (35.5 s when the same box compiled them). `KILN_COMPILE_FARM=<queue uri>` makes a device run
wait for, and fetch, a graph the farm holds instead of compiling it, so the device box can load
weights while the farm compiles.

**What a graph costs in HBM besides tensors, read from the NEFF** (`python tools/neff_info.py <cache
dir | s3 entry prefix>`). Each engine's `sg00/<engine>.json` lists its DMA queue entries; the runtime
builds a descriptor ring per queue at load and reports the spill queues as "dma rings spill". The
tool counts the contiguous runs each queue moves. Calibration, one point: c38f97c9 (config A, a
P=2 dense prefill pair) has 67,709,320 runs on qSPSpillReload0, and at 32 B per run over its 16
queues that is 135.4 MB per queue. The runtime's own failure on kiln-g1-trn1 was "Failed to
allocate 136.532MB (usage: dma rings spill)" for qSPSpillReload0_10 while loading that graph. So
ring bytes ~= 32 B x spill runs (estimate, not a documented formula). Measured on the DP-4 graphs
(512 rows x 4 groups into the MoE):

| graph | spill runs | est. spill rings | compiler "DRAM spill space" |
|---|---|---|---|
| P=2 prefill pairs, moe_dedupe kernel (A', B': 8 graphs) | 68.3-70.7M | 2.08-2.16 GB each | 1.41-2.49 GB |
| P=2 last group (1 layer) | 34.1M | 1.04 GB | 0.63 GB |
| P=4 prefill, 4 MoE, grouped-GEMM kernel (e4dbb70d) | 4.55M | 139 MB | 5.28 GB |
| P=4 last group, grouped-GEMM kernel (43fa2240) | 0.71M | 22 MB | 0.99 GB |
| decode 12-layer groups | 0.08-0.11M | 2-3 MB | 0.15-0.18 GB |

A P=2 prefill set under moe_dedupe is ~9.5 GB of spill rings per NeuronCore, next to ~13 GB of
tensors on a 16 GB core: that is why configs A' and B' loaded every graph from the cache and then
failed on HBM. Whether the runtime adds the compiler's spill space per graph or shares it across
graphs was not determined; its table at that failure showed "shared scratchpad 0.32 GB".

**A NEFF records its output path.** info.json "name" is neuronx-cc's `--output`, and the runtime prints
that name in load errors: a farm NEFF built under /opt/kiln/farmroot-p was reported as
"/opt/kiln/farmroot-p/.../graph_<key>.neff" on a host where that directory never existed, while the
file loaded was ~/.cache/.../graph_<key>.neff. `tools/compile_farm.py work` now compiles under LNL's
default root, so the name matches a device-built NEFF.

**Fitting the prefill graphs into HBM, without the prefill kernel** (DP 4, decode bucket 8,
`--max-num-seqs 32 --kv-cache-gb 1.0`, bf16 KV, default flags; `python tools/neff_info.py --keys-file
<capture keys.json> --cache <s3 cache> --queue <farm queue>` over every graph the configuration loads):

| prefill rows per group (tokens / bucket) | P | graphs | est. spill rings, sum | compiler spill space, max / sum | slowest graph |
|---|---|---|---|---|---|
| 512 (2048 / 512), config B' | 2 | 16 | 10.0 GB | 2.49 / 9.10 GB | 1158 s |
| 256 (1024 / 256) | 2 | 16 | 5.01 GB | 1.20 / 4.65 GB | 643 s |
| 128 (512 / 128) | 2 | 16 | 2.51 GB | 0.51 / 2.26 GB | 479 s |
| 256 (1024 / 256) | 1 | 14 | 1.65 GB | 0.40 / 1.55 GB | 346 s |
| 128 (512 / 128) | 1 | 14 | 0.83 GB | 0.22 / 0.98 GB | 354 s |
| with `KILN_CC_ARGS=--model-type=transformer`: | | | | | |
| 256 (1024 / 256) | 2 | 16 | 0.147 GB | 1.02 / 4.86 GB | 295 s |
| 256 (1024 / 256) | 1 | 14 | 0.048 GB | 0.38 / 1.67 GB | 268 s |
| 128 (512 / 128) | 2 | 16 | 0.083 GB | 0.49 / 2.52 GB | 268 s |
| 128 (512 / 128) | 1 | 14 | 0.033 GB | 0.19 / 1.05 GB | 268 s |
| DP 1, 512 rows (512 / 512), P=4, transformer (config F1, 32 ranks) | 4 | 12 | 0.103 GB | 1.26 / 3.11 GB | 399 s |
| MiMo-V2.6-Flash-RL, tp=32, prefill 2048, P=4, prefill kernel, transformer (config F2) | 4 | 10 | 0.101 GB | 1.93 / 6.34 GB | 157 s |

At default flags spill rings scale with rows x layers per graph: 0.27 GB per MoE layer at 128
rows per group, 0.55 GB at 256, and 2.1 GB for a P=2 graph at 512 rows, the one that failed to load.
`--model-type=transformer` removes most of them, 15-35x less: in a transformer prefill NEFF the SP
engine's spill-reload queue holds about 4k entries instead of about 530k. Together with its 9-10%
faster decode step and its passing ppl check (the lead's check_ppl, -2.102 against -2.091 at default
flags), it became the default flag for every GLM-5.3-Flash configuration on 2026-10-04.

### The hidden HBM item: DMA-ring spill reservations (2026-10-04, GLM-5.3-Flash tp=32 DP 4)

neuron-monitor's per-core breakdown (tensors, model code, scratchpad, constants) omits the DMA
rings. The runtime's own table, written at an allocation failure to
/tmp/neuron_mem_table_device_<nd>_nc_<nc>.log (and "Failed to allocate 136.532MB (alignment: none,
usage: dma rings spill)" in the log), for the A' sweep config (DP 4, decode bucket 16, prefill
512 per group, P=2 prefill pairs, KV 1.0 GB FP8): TOTAL 15.814 GB of 16 = tensors 12.997 GB + DMA
rings spill 2.121 GB + model code 0.301 + shared scratchpad 0.320 + collectives 0.080 + IO rings
0.002. Spill rings are reserved per loaded NEFF for the compiler's spill/reload queues
(qSPSpillReload*), so many large prefill graphs (2048 rows into a 2-layer MoE group) add up; the
decode 12-layer groups reserve about 20 MB each. On trn1 a big model therefore has to budget for
spill rings per loaded graph, not only weights, KV and scratchpad.

### --model-type=transformer is adopted (2026-10-04)

Compile farm, on kiln-g1-trn1 (GLM-5.3-Flash tp=32, DP 4, decode bucket 8 per group, real
weights, graphs from the farm): decode step 162.0 / 163.7 ms at default flags (two runs) against
146.8 ms with `KILN_CC_ARGS=--model-type=transformer` (one run), 9-10% off every layer group; it
also compiles faster (decode group -27%). Correctness: `tools/check_ppl.py --model
zai-org/GLM-5.3-Flash --device neuron --tp 32 --piecewise --kv-cache-gb 2.0` with
`KILN_MOE_KERNEL=nki KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=2`: mean prompt
logprob -2.102 with the flag against -2.091 without (same config). `-O1` does not compile these
graphs (decode group: NCC_INAS001 internal error; prefill groups: NCC_EBVF030). Changing the flag
changes every cache key.


**Configurations the farm built for the GLM-5.3-Flash sweeps (2026-10-04, all with
`KILN_CC_ARGS=--model-type=transformer KILN_MOE_KERNEL=nki KILN_PIECEWISE_MOE_GROUP=12
KILN_PIECEWISE_PREFILL_MOE_GROUP=2`, tp=32, page bucket 264; keys checked on all 32 ranks where
marked; totals over every graph the configuration loads, rings at 32 B per spill run):**

| config | graphs | est. spill rings | compiler spill space max / sum | slowest compile |
|---|---|---|---|---|
| G16: DP 4, decode 4, KV 0.5, prefill 2048 / 512 (32 ranks) | 15 | 0.29 GB | 2.12 / 8.96 GB | 647 s |
| G64: DP 4, decode 16, KV 1.0 fp8 (32 ranks) | 17 | 0.35 GB | 2.12 / 11.43 GB | 637 s |
| G128: DP 4, decode 32, KV 2.0 fp8 (32 ranks) | 18 | 0.43 GB | 2.12 / 16.42 GB | 633 s |
| G128 at DP 8, decode 16, KV 1.0 fp8 | 18 | 0.70 GB | 4.28 / 27.47 GB | 1195 s |
| F4-G16: as G16, prefill 4096 / 1024 | 15 | 0.80 GB | 5.65 / 20.75 GB | 1268 s |
| F4-G64 | 17 | 0.86 GB | 5.65 / 23.20 GB | 1277 s |
| F4-G128 (DP 4) | 18 | 0.94 GB | 5.65 / 28.20 GB | 1269 s |

On the device (the lead's runs, 2026-10-04 02:43-02:56 UTC): G64 and G128 at prefill 2048 / 512
loaded from this cache and then failed on HBM ("nrt_tensor_allocate status=4", "Could not load the
model ... Allocation Failure") even with the transformer flag, so trn1 runs prefill 1024 / 256; the
same three configurations at 1024 / 256 (G16p1024 15, G64p1024 17, G128p1024 18 graphs, all keys on
all 32 ranks) are in the cache, with their exact command lines in
s3://<your-bucket>/compile-farm/keys/<name>.config.json. Since summed spill rings are well
under 1 GB here, the compiler's per-graph spill space adding up across loaded graphs is the likely
reason 2048 / 512 does not fit; not measured directly. DP 8 doubles the rows a prefill MoE call takes
(8 x 512) and with them the slowest compile and the per-graph spill space; at 1024 rows per group (F4), P=2 still fits the 5M-instruction limit with the
transformer flag. trn2 (`capture --target trn2`, NEURON_LOGICAL_NC_CONFIG=2; the platform module adds
`--logical-nc-config=2` and vllm-neuron's argument set): the T configuration (DP 4, decode 8, prefill
2048 / 512) captured and compiled on CPU hosts with the NKI MoE kernel, 16 graphs on all 32 ranks,
decode groups 290-313 s, prefill pairs 549-568 s. Its keys changed when kernel grids began to follow
the runtime LNC (platform.nki_grid), which the capture must set up before LNL as build_shard does
(platform.configure_runtime_env).

### Load-time decisions that read weight values (2026-10-04)

A capture serves the weights as zeros, and that is wrong wherever the loader looks at values to
choose a layout. With `KILN_MOE_PREFILL_KERNEL=nki`, DecoderLayer.pack_experts asks
kernels/moe_prefill.check_blob (the static flag `moe_prefill_dq`) and down_factors (two more
parameters per MoE layer, `moe_prefill_dsc` / `moe_prefill_dfr`, or None). On zeros the answers are
True / None. On GLM-5.3-Flash at tp=32 they are False / two tensors. Measured on the d25e855 G64p1024
run on kiln-g1-trn1 (log 20261004T033657Z-serve_sweep.log, ASSERT_CACHE_HIT), the miss was key
ddfec8b1fbe8c499da25196100cd8c40. Its fxgraph.txt against the capture's ef4b4a25 (graph.hlo both
2,264,167 bytes, the same 354 input shapes in order) differs only in placeholder names. The
device's 12-layer decode group has two more tensors in each of its 9 MoE layers (unused in decode,
so dynamo prunes them, but the list indices move: L_ts_108_ became L_ts_110_ and so on). The prefill
groups also compute different paths.

`python tools/compile_farm.py decisions --model zai-org/GLM-5.3-Flash --shape-dir <dir> --tp 32`
now writes kiln-load-decisions.json into the shape directory, and the capture replays it
(kiln/capture.py "Load-time decisions"). A capture that needs an answer the file does not hold
raises. The tool runs the rank's shard of one expert at a time through loader._load_experts and
moe_dedupe.pack, by HTTP range reads from the Hub with no download, until both answers are known.
GLM-5.3-Flash@eb9eb208 (the revision on kiln-g1-trn1) took 46.5 s on kiln-cf-1 (44 processes): 1376
(layer, rank) answers over the 43 MoE layers, the MTP layer included, all dq False / down present,
each decided by expert 0. The Hub limits an IP without a token to 3000 "resolver" requests per 300
s, so the offset table is built once and handed to the workers (the first try, with each worker
reading all 62 files' metadata, got 429). The G64p1024 recapture on tree e902f99 produces ddfec8b. 7
of its 17 keys changed, and the 32-rank capture gives every key on all 32 ranks.

### Prefill group size P on the new layout (2026-10-04, feat/kda-kernel efa1da4)

The configurations were captured and compiled by the farm (tree d0b04230, `KILN_CC_ARGS=--model-type=transformer
KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_PIECEWISE_MOE_GROUP=12`, elementwise mHC by
default, load-time decisions replayed). Columns:
- instructions: read from each NEFF by tools/neff_instructions.py, 64 B per instruction, the count
  NCC_EBVF030's 5M limit is about.
- spill runs: the runs in Spill DMA queues.
- spill space: the compiler's `[DRAM_Allocator] spill space` line, max over subgraphs.
- compile seconds and host peaks: from the queue's done records.

No graph failed and none hit NCC_EBVF030.

| prefill groups | rows per group | instructions per group | spill runs | spill space per group | compile |
|---|---|---|---|---|---|
| trn2 T2, P=12 (3 groups) | 512 | 2,527,829 / 2,267,321 / 1,767,370 | 0 | 1.61 GB | 401-435 s, 17-30 GB |
| trn2 T2, P=6 (4 groups) | 512 | 1,547,760 / 1,177,475 / 1,102,947, 3-layer tail 1,123,456 | 0, tail 2.10M | 0.54-0.87 GB, tail 1.33 | 134-214 s |
| trn1 F0, P=12 (3 groups) | 256 | 1,180,903 / 927,638 / 729,963 | 0 | 1.48 GB | 225-269 s, 8-12 GB |
| trn1 F0, P=6 (4 groups) | 256 | 926,057 / 879,055 / 656,856 / 467,673 | 1.05-2.27M | 1.19-2.33 GB | 374-377 s |
| trn1 F0, P=2 (pairs) | 256 | 146,468-332,143 | 0.38-0.78M | 0.32-0.83 GB | |

Whole configurations (every graph loaded; rings at 32 B per spill run; spill space max / sum):

| config | graphs | est. rings | spill space |
|---|---|---|---|
| T2-P12 | 14 | 0.041 GB | 1.61 / 5.53 GB |
| T2-P6 | 15 | 0.108 GB | 1.33 / 3.98 GB |
| T128-P2 | 17 | 0.230 GB | 0.97 / 4.82 GB |
| T16-P2 | 14 | 0.233 GB | 0.98 / 3.56 GB |
| F0-P12 | 14 | 0.015 GB | 1.48 / 5.06 GB |
| F0-P6 | 15 | 0.237 GB | 2.33 / 8.05 GB |
| F0-P2 | 16 | 0.099 GB | 0.83 / 3.51 GB |
| G64p1024-P2 | 17 | 0.150 GB | 0.84 / 5.63 GB |
| G16p1024-P2 | 15 | 0.096 GB | 0.83 / 3.23 GB |

With the prefill kernel, P=12 uses about half of the 5M limit at 512 rows (trn2) and a quarter at
256 rows (trn1), consistent with ~126K instructions per MoE layer at 256 rows with the kernel. It
also removes every prefill spill ring and leaves 3 prefill graphs per step instead of 23. Its price
is the spill space per graph (about 1.5 GB per prefill group against 0.3-0.8 GB for a pair). P=6 at
256 rows spills more than P=12, which was measured and not explained. Compile host peaks on this
layout are 5-30 GB (80 compiles), so kiln/compile_farm.peak_gb_estimate no longer assumes the bmm
form's 175-200 GB for HLOs of 2 MB and over.

Several farm workers on one host (one per queue) must share one memory budget. Each used to budget
90% of the host for itself: on kiln-cf-1 (c8i.48xlarge, 371 GiB) on 2026-10-04 the v1-trn1, mx-trn1
and pfab-trn1 workers held 330 + 294 + 133 GB of reservations at once, and between 16:16 and 16:22
UTC the OOM killer took 11 compiles (dmesg: walrus_driver at 21-25 GB anon-rss each). The check of
free memory at launch could not see it: a compile reaches its peak minutes after it starts. Workers
now record their reservations and the keys they compile in /dev/shm/kiln-farm-ledger (one file per
pid, check-and-reserve under one flock; compile_farm.HostLedger, compile_farm.admit). A key that
another local worker is compiling waits for that NEFF, because two queues can hold one graph
(t2max-U1 and -U0 share 14 decode keys) and two compiles of one key on one host write the same LNL
entry directory. Size is a weak proxy for a single graph's peak: an F0 mixed group of the size the
G64 groups compile in at 23-30 GB was estimated at 26.9 GB and killed at 107.3 GB. An OOM-killed
graph is retried with max(2x its estimate, 1.5x the tree RSS at the kill), remembered per key in
`<queue>/oom/<key>.json` (compile_farm.oom_reservation), so the next worker does not repeat the kill.

With `KILN_LINEAR_ATTN_KERNEL=nki` (the chunked KDA kernel), the P=12 prefill groups take 5-12%
fewer instructions: trn2 2,398,362 / 2,006,798 / 1,662,702, trn1 1,123,812 / 856,863 / 691,218.
Spill space is the same and spill runs stay at 0. The HLO grows to 8.9 MB because the kernel
binaries travel in backend_config. At prefill 2048 / 512 on trn1 (512 rows per group), P=12 groups
take 1,805,857 / 1,451,289 / 1,115,603 instructions with 0 spill runs, but 2.64 GB of spill space
each.

HBM per rank, estimated without a device, as the Neuron runtime accounts it (`tools/hbm_estimate.py`;
the model is in kiln/compile_farm.py). The runtime prints its table only on an allocation failure. The
one on kiln-g1-trn1 at 02:47 UTC (/tmp/neuron_mem_table_device_12_nc_1.log, the old-layout G64 at
prefill 2048, P=2) lists 10 NEFFs, all in the trn1 cache, so each column could be matched. The table
prints MB / GB, which are MiB / GiB:

- Model Code: the NEFFs' instruction streams (sg<N>/<Engine><N>.bin). NEFF 1002 has 62.66 MB of
  streams and printed 59.87 MiB; 1010 has 190.998 MB and printed 182.81 MiB.
- Scratchpad: each NEFF reserves the compiler's `[BackendDriver] ... Peak scratchpad usage: local`
  value, exactly: 0.276024 GiB = 282.648 MiB (1002), 0.251030 = 257.055 (1004), 0.361675 = 370.354
  (1008). The shared scratchpad is the largest over the loaded NEFFs, rounded up to 64 MiB:
  370.479 -> 384.
- DMA Rings Spill: 32 B per run of the Spill DMA queues, within 4-11% per NEFF; 310.7 MiB summed
  against 333.3 printed.
- IO and collectives rings, runtime, profiler and constants: 0.13 GiB together.

Tensors 14.247 + code 0.835 + scratchpad 0.375 + rings 0.325 + 0.13 = 15.91 GiB, the table's total
for the failed load (16 GiB per NeuronCore). The listed NEFFs' code sums to 757.6 of the 855 MiB
printed: the NEFF being loaded when the allocation failed had probably taken the rest.

The DRAM_Allocator "spill space" line, used before this as the scratchpad term, is a different
quantity: 0.8-2.1 GB on the same NEFFs. It equals the reservation only for one-subgraph NEFFs (1001:
1.0 MiB, 1006: 32 MiB). That is why the earlier rule (tensors + rings + instructions + largest spill
space) overestimated by 2.6-4.5 GB: G16-4096-P12-K loaded at 17.79 GB and F0-4096-P12-K at 18.59 GB
by it, above the physical 17.18.

Per rank, GiB (tensors from `tools/tensor_bytes.py`):

| config (trn1, tp=32, DP 4) | tree | tensors | code | scratchpad | spill rings | total | on the device |
|---|---|---|---|---|---|---|---|
| G16p1024-P12 | efa1da4 | 11.887 | 0.232 | 0.312 | 0.010 | 12.57 | loads |
| G16-2048-P12 | efa1da4 | 11.894 | 0.324 | 0.500 | 0.010 | 12.86 | loads |
| F0-P12, prefill 1024 | efa1da4 | 12.594 | 0.268 | 0.312 | 0.014 | 13.32 | loads |
| sp G16-4096-P12-K | e5c43963 | 11.906 | 0.343 | 1.000 | 0.009 | 13.39 | loads |
| F0-2048-P12 | efa1da4 | 12.600 | 0.360 | 0.500 | 0.015 | 13.61 | loads |
| G64p1024-P12 | efa1da4 | 13.007 | 0.306 | 0.312 | 0.062 | 13.82 | loads |
| dsa G64-2048-P12-K | 787bf1fa | 13.006 | 0.306 | 0.375 | 0.063 | 13.88 | |
| G64-2048-P12-K | efa1da4 | 13.011 | 0.379 | 0.438 | 0.065 | 14.02 | loads |
| sp F0-4096-P12-K | e5c43963 | 12.612 | 0.379 | 1.000 | 0.013 | 14.13 | loads |
| dsa G64-4096-P12-K | 787bf1fa | 13.017 | 0.370 | 0.750 | 0.064 | 14.33 | |
| G128p1024-KV1.5-P12-K | efa1da4 | 14.331 | 0.352 | 0.562 | 0.137 | 15.51 | |
| G128p1024-P12-K (KV 2.0) | efa1da4 | 14.831 | 0.352 | 0.562 | 0.137 | 16.01 | |
| the failed load (old layout, G64-2048 P=2) | | 14.247 | 0.835 | 0.375 | 0.325 | 15.91 | failed |

So the per-graph cost of a bigger prefill chunk is its scratchpad (0.31 GiB at 256 rows per group,
0.44-0.50 at 512, 0.75-1.00 at 1024), not the multi-GB spill space. trn2 has 24 GiB per logical core.

**Open term R: spill rings of queue-instance DMA entries.** Newer NEFFs name a DMA entry's queue as an
instance (`"instance_name": "qSPSpillReload0_defId_1"`, listed under the queue's `queue_instances` in
`sg<N>/def.json`) instead of `"queue"`. Every SP prefill group at 1024 rows per group and above has
only those. For queue-named entries the runtime's spill rings are 32 B per run, within 1-4% on the
decode NEFFs of a second failure table (kiln-mimo-trn1, 2026-10-04 ~11:49 UTC,
/tmp/neuron_mem_table_device_13_nc_1.log): NEFF a3453db7 has 3.233 MiB of runs for 3.111 printed,
8e65b98c 2.344 for 2.335. Instance entries are sized differently, and not linearly:

- G64-8192-P12-K's prefill group d730c656 (2048 rows per group, DSA tree 787bf1fa) has 6.15M
  instance runs, 187 MiB at 32 B. Its load failed with the table's DMA Rings Spill at 2.682 GiB, of
  which the six loaded NEFFs hold 11.5 MiB, so this one NEFF brought ~2.67 GiB (x14.6).
- The 1024-row groups have 4.2M instance runs each. sp F0-4096-P12-K loads with three of them, so R
  for that configuration is at most 16 - (12.61 + 0.38 + 1.00 + 0.13) = 1.88 GiB (<= ~4.6x).

tools/hbm_estimate.py counts the instance runs at 32 B x the queue's num_queues (16). That is an upper
bound that fits the 2048-row group (2.92 against 2.67 GiB) and overestimates the 1024-row ones, so a
configuration whose estimate is over 16 GiB only because of instance rings needs a device run. Prefill
8192 at 2048 rows per group, with three such groups (~8 GiB of rings), does not fit trn1.

A tighter bound for the EP + SP_GROUP serving layout (engine-v0 e449b98, 2026-10-05). G64-DB4 (the
final G64 config with `--decode-buckets 4,8,12,16`) loaded and served on kiln-ut-32 (125.8 out tok/s,
the utilization agent's run). Per rank, tools/hbm_estimate.py over its own attention group's graphs
gives tensors 13.341 + code 0.629 + scratchpad 0.312 + fixed 0.13 = 14.41 GiB, plus 11,537,328
instance runs (EP decode groups and 1024-row prefill groups). Fitting in 16 GiB means those runs cost
at most 1.59 GiB, i.e. **<= ~148 B per instance run** on this layout, against the estimator's 512 B
and the ~469 B the 2048-row group needed. So at 1024 rows and in the decode groups, an instance run
costs under a third of the bound. Rank the deltas of two such configs with the estimator; do not
compare its absolute totals with 16 GiB.

## Where a GLM-5.3-Flash prefill step goes on trn1 (2026-10-04, sweep shapes)

`KILN_PROFILE_PIECES=1 KILN_PROFILE_EXEC=1` on the F0 sweep config (tp=32, DP 4, prefill 1024
tokens = 256 rows per group, `KILN_PIECEWISE_PREFILL_MOE_GROUP=2`, NKI decode MoE kernel, no prefill
MoE kernel, `--model-type=transformer`, graphs from the compile farm; kiln-mimo-trn1 log
/opt/kiln/logs/20261004T015942Z-serve_sweep.log): a prefill step is **2115 ms** (exec p50, n=96)
for 1024 prompt tokens, about 484 tok/s; its 23 two-layer pieces (hidden [1024, 16384], the four
hyper-connection streams) take 57 and 72 ms, then **82-87 ms each**, and 49 ms for the last single
layer. A decode step at 8 sequences per group is 143.8 ms over four 12-layer pieces (38, 35, 35, 27
ms). So prefill costs ~42 ms per layer at 256 rows per group, uniformly over KDA and DSA layers,
against 10.7 ms (dense + KDA) and 14.8 ms (MoE + KDA) for one isolated layer at 512 rows per group
measured by the KDA kernel work; the isolated MoE + DSA layer showed the same inflated cost (83 ms)
with a DMA spill signature (6,814 spill entries / 882K runs). The gap between a layer alone and the
same layer inside the sweep's prefill graphs is the main prefill lever on trn1.

One layer per prefill graph does not help (2026-10-04, same config with
`KILN_PIECEWISE_PREFILL_MOE_GROUP=1`, kiln-mimo-trn1 log 20261004T021735Z): 45 one-layer pieces at
33.5-48.4 ms (p50 ~47 ms for the MoE layers, n=97), a prefill step of **2588 ms** against 2115 ms
for the 2-layer graphs. So the ~42 ms per layer is inside the layer, not the price of grouping two
layers into one graph, and every extra graph adds its fixed collective cost on top. The KDA kernel
work then located it: the hyper-connection (mHC) stream arithmetic, written as an fp32 broadcast-sum
and per-token [4x4]@[4x4096] bmm over [T, 16384] streams, is what spills; the same algebra as
per-token multiply-adds of [T, 4096] rows (feat/kda-kernel, `KILN_MHC_FORM=elementwise`) takes a
dense + KDA layer at 512 rows per group from 10.7 to 8.2 ms on the torch path and to 3.79 ms with
the KDA kernel (32K instructions instead of 177K). Any NKI call inside the bmm-form layer graph
inflated it to ~81 ms, a null kernel included.

Prefill 2048 still does not fit trn1 with `--model-type=transformer` (2026-10-04): GLM-5.3-Flash
tp=32, DP 4, prefill 2048 / bucket 512, graphs from the farm (0 compiles), failed at load on both
G64 (decode 16, KV 1.0 GB fp8: `nrt_tensor_allocate status=4`, kiln-g1-trn1 log 20261004T024347Z)
and G128 (decode 32, KV 2.0 GB fp8: `Could not load the model status=4 message=Allocation Failure`,
kiln-mimo-trn1 log 20261004T024401Z). The flag shrinks the DMA rings, not the tensors; trn1 sweeps stay
at prefill 1024 / bucket 256 (512 rows per group would need the attention-side weights off the core,
i.e. a smaller DP replica footprint), and prefill 4096 configs are trn2-only.
Concurrency 128 does not fit trn1 at DP 4 either, even at prefill 1024: G128p1024 (decode 32 per group,
KV 2.0 GB fp8 per rank, 18 farm graphs) failed at load with `Allocation Failure` (kiln-mimo-trn1 log
20261004T025611Z). Concurrency 128 runs on trn2 (24 GiB per logical core at LNC=2).

The prefill MoE kernel on real weights end to end (2026-10-04, engine-v0 63f683b, kiln-mimo-trn1 log
20261004T030348Z-check_ppl): `KILN_MOE_KERNEL=nki KILN_PIECEWISE_MOE_GROUP=12 KILN_MOE_PREFILL_KERNEL=nki
KILN_MOE_PREFILL_MIN_TOKENS=1 KILN_CC_ARGS=--model-type=transformer tools/check_ppl.py --model
zai-org/GLM-5.3-Flash --tp 32 --piecewise`: France -2.214, Water boils -3.528, def add -0.933, quick fox
-1.692, mean **-2.086** (44 tokens), the same as the dedupe-kernel path (-2.086) and within 0.04 per
sentence of the CPU bf16 reference (-2.098).

The new prefill layout at 32 ranks (2026-10-04, tree e902f99 = engine-v0 63f683b + feat/kda-kernel
4fc7ab3, F0 config, `KILN_MHC_FORM=elementwise` (fp32 mix) `KILN_MOE_PREFILL_KERNEL=nki`, torch delta
rule, `KILN_PROFILE_PIECES=1 KILN_PROFILE_EXEC=1`, kiln-mimo-trn1 log 20261004T052214Z): 2-layer
prefill pieces 56-70 ms (alternating ~67 / ~70 over the KDA and DSA pairs), prefill exec p50 1963 ms
for 1024 tokens (bmm form: 82-87 ms pieces, 2115 ms), decode at 8 per group 136.8 ms (143.3). The
same groups measured 21-34 ms with `tools/profile_layer.py --ranks 2`, so most of a piece's cost
appears only with 32 live ranks; the unprofiled sweep still went 18.4 -> 31.4 out tok/s at
concurrency 32 (per-piece profiling synchronises every piece and breaks the host/device overlap,
so its absolute numbers are an upper bound).
The same profile on trn2 (kiln-trn2-48 log 20261004T052638Z, T2 config: prefill 2048 = 512 rows per
group, LNC=2): 2-layer prefill pieces h[2048, 16384] 69-84 ms for the first six, then **99-101 ms**
each, prefill exec p50 2524 ms per 2048 tokens (~810 prompt tok/s per engine), decode at 8 per group
132.7 ms (four pieces 25.8-33.9 ms). Decode is no faster than on trn1 (136.8 ms): at 8 rows per
group the step is the fixed per-graph collective cost, not compute.

Why elementwise mHC moved real-weight ppl (found 2026-10-04 by the KDA kernel work,
tools/probe_mhc_output.py): the fp32 output mix itself is computed bit-identically to the host. But
inside a graph neuronx-cc folds an f32 -> bf16 -> f32 convert pair between the mix and the next
block's fp32 ops (excess-precision folding), so the residual streams stay fp32 across a whole
12-layer group on the device while the CPU rounds them every block. Explicit converts therefore
change nothing, and the bmm form is immune because its rounding comes from a real bf16 matmul. A
bitcast barrier (`.view(int16)`) is rejected in LNL graphs; rounding with a Veltkamp split in fp32
(`c = 65537 * x; hi = c - (c - x)`, exactly bf16 round-to-nearest-even, checked on 100K values)
leaves nothing to fold. **Any explicit dtype round trip inside a Neuron graph may be optimised
away; a rounding that must happen has to be arithmetic.**

**`KILN_PROFILE_PIECES` roughly doubles prefill piece times; read the real step off throughput.**
The same tree at 32 ranks in `tools/profile_layer.py --ranks 32 --prefill 256 --dp-attention 4`
(random weights, kiln-mimo-trn1 log 20261004T054002Z): groups 0-1 [KDA dense x2] 22.6 ms, 2-3
[KDA dense + DSA MoE] 26.2, 4-5 [KDA MoE x2] 31.9, 6-7 [KDA MoE + DSA MoE] 30.7; 217-287K
instructions each. The profiled sweep showed 56-70 ms for the same pieces and 1963 ms per prefill
step, but the unprofiled sweep's own numbers rule that out: 31.4 out tok/s at concurrency 32 is
0.123 req/s = ~1005 prompt tok/s = 0.98 prefill steps/s plus ~1 decode step/s at 137 ms, which
leaves **~0.88 s per 1024-token prefill step** (old layout, same arithmetic: ~1.6 s). So a real piece
is ~35 ms, the isolated group plus ~5-7 ms of per-graph cost, and prefill is still ~86% of device
time at concurrency 32. Per-piece profiling synchronises every graph and breaks the overlap, so use
it for relative shares only.
On trn2 at 512 rows per group (LNC=2, log 20261004T054021Z), the same groups: 2-3 30.9 ms, 4-5 40.0,
6-7 36.0. The KDA kernel end to end (trn1, concurrency 32, same tree and config,
`KILN_LINEAR_ATTN_KERNEL=nki`, kiln-g1-trn1 log 20261004T053222Z): 28.6 out tok/s against 31.4 with
the torch delta rule, so it stays opt-in; its isolated wins (4-5x on the recurrence) do not survive
the 2-layer prefill graphs at 32 ranks.


## The chunked delta rule as one NKI kernel (2026-10-04, SDK 2.32, nki 0.6.0, trn1.2xlarge)

`kiln/kernels/delta_rule.py`, on when `KILN_LINEAR_ATTN_KERNEL=nki` (default `torch`): the prefill
(chunk and sequence forms) of a Gated DeltaNet layer or a KDA layer with a gate lower bound, head dims
128, on the device; decode and verify keep `linear_attn.recurrent_step`, Kimi-Linear-48B (KDA without
a lower bound) keeps `chunk_scan`. The kernel takes the torch path's tensors after the short conv, the
l2 norms and the gates (q, k [T, Hk, 128], v [T, Hv, 128], g [T, Hv, 128] or [T, Hv], beta, the state
row) and returns o and the final state, so the state pool, the prefix-cache checkpoints (whole-row
copies after a chunk) and the verify snapshots are untouched. The algorithm is fla's chunked form
(module docstring: sources in fla 3e52d5a, vLLM v0.30.0, SGLang v0.5.21) laid onto the engines:
128-token chunks; every matmul in fp32; KDA's decayed products through 16-row reference sub-chunks
(fla's chunk_kda_fwd_intra scheme), finite in fp32 only because the safe gate bounds every log decay
(e^75 at most); the UT transform computed transposed by the blocked inverse (8 x 8 blocks by squaring,
four merges: 18 matmuls per chunk); the chunk as an affine map S' = P S + Q, so per chunk only two
matmuls wait for the previous state. `emulate()` is its arithmetic in torch.

The NKI facts it rests on (`tools/probe_delta_prims.py`, on the device): an fp32 x fp32 matmul on the
tensor engine is exact to 1.6e-7 relative (bf16 operands would be ~4e-3); a transpose as a matmul with
the identity is bit exact; the scalar engine's exp with a per-partition bias is within 1.5e-5 relative
over arguments -85..85 (median 1.5e-6) and flushes below -87 to 0; a matmul may write a strided PSUM
region (`z[:, :, 16 I:16 I + 16]`); kernels may call module-level helper functions; the kernel
compiler rejects list comprehensions ("unsupported expression"); GpSimd cannot read PSUM, and a GpSimd
fp32 tensor_tensor failed NCC_IXCG965 on trn1.

**One call** (`python tools/probe_delta_rule.py --kind kda --heads H --chunks 512 2048 8192`, random
inputs as the mixer builds them, lower bound -5, p50 of a graph reading back o.sum(0) and S.sum(0);
error = max abs error relative to the max, against the token-by-token recurrence in float64):

| shapes | C | kernel | torch chunk_scan (sub-chunks 32) | kernel error o / S | torch error o / S |
|---|---|---|---|---|---|
| GLM-5.3-Flash tp=32: 2 KDA heads | 512 | 0.311 ms | 0.925 ms | 7.2e-6 / 1.6e-6 | 1.5e-6 / 2.0e-6 |
| same | 2048 | 0.678 | 3.190 | 7.4e-6 / 2.6e-6 | |
| same | 8192 | 2.175 | not compiled | 9.2e-6 / 2.5e-6 | |
| attention TP 8 (the DP-4 sweeps): 8 heads | 512 | 0.698 | 3.009 | 7.6e-6 / 2.6e-6 | 1.4e-6 / 3.5e-6 |
| same | 2048 | 2.241 | 11.343 | 6.9e-6 / 2.8e-6 | |
| same | 8192 | 8.372 | not compiled | 7.8e-6 / 2.0e-6 | |

| Qwen3.8-Flash-Next tp=8 GDN: 2 k / 6 v heads | 512 | 0.471 | 0.487 (sub-chunks 64) | 4.6e-6 / 5.7e-6 | 7.8e-6 / 1.7e-6 |
| same | 2048 | 1.303 | 1.405 | 5.8e-6 / 2.7e-6 | 5.0e-6 / 3.4e-6 |
| same | 8192 | 4.656 | not compiled | 7.6e-6 / 1.6e-6 | |

(GDN gains little: its decay is a scalar per token and head, so chunk_scan is already a few large
matmuls; KDA's per-channel decay is what the torch path pays for. With correlated keys, beta near 1 and
slow decay (`--correlated`, the regime that broke the unstable inverse) GDN at C=2048: 2.9e-6 / 2.6e-6.
The 2-head rows ran the first version, which interleaved two heads; the 8-head rows the current one,
`KILN_DELTA_RULE_UNITS` = 2 (chunk, head) units interleaved: at 1 / 2 / 4 / 6 / 8 units C=8192 took
10.84 / 8.37 / 8.73 / 8.68 / 8.66 ms and C=2048 2.85 / 2.24 / 2.32 / 2.28 / 2.29, so it is
throughput-bound, about 16 us per (chunk, head), not latency-bound.) Compile: 2.3-13.9 s per kernel
graph at C <= 8192 (2 heads), 4.7 / 13 / 50 s at 8 heads, against 27 s (C=512) and 131-154 s (C=2048)
for chunk_scan. Instructions, 8 heads, C=512: 8,952 for the kernel graph (PE 5,031, DVE 2,237, ACT
1,244) against 25,665 for chunk_scan.

**Inside the model.** A served random-weight GLM-5.3-Flash (`tools/build_random_hybrid.py glm5_next
--sparse`: 6 KDA + 2 pooled-DSA layers, head dim 128) on kiln-kda2-trn1, fp32 with
`KILN_CC_ARGS=--auto-cast=none`, piecewise, a 700-token prompt through 256-token prefill chunks
(`tools/check_linear_kernel.py --len 700 --prefill-tokens 256 --piecewise`) against transformers
5.18.0 fp32: teacher-forced max |dlogprob| 3.2e-5 (mean 6.0e-6) through the kernel, 3.0e-5 (4.4e-6)
through chunk_scan; greedy 16/16 both. A random Qwen3.8-Flash-Next (`build_random_hybrid.py qwen4_exp
--sparse`, GDN with Dk = Dv = 128) the same way through the kernel: max |dlogprob| 9.7e-6 (mean 1.9e-6),
greedy 16/16. The whole layer at C=512 with 8 heads per rank (`tools/
profile_layer.py --la-compare`) differs from the torch path's by 6.2e-3 (dense + KDA) and 8.5e-3 (MoE +
KDA) of the output's max, bf16 rounding of the surrounding layer.

At the sweep's shapes (DP 4, 256 rows per group, `--dp-attention 4 --prefill 256`, Veltkamp
elementwise mHC, --model-type=transformer), the kernel against the torch delta rule: 2 ranks, dedupe
MoE: layer 0 5.74 / 13.20 ms, layer 4 17.63 / 25.73, groups 0-1 / 2-3 / 4-5 11.56 / 23.00 / 35.37
against 23.62 / 34.85 / 48.81; 32 live ranks on kiln-kda-32b (trn1.32xlarge), prefill MoE kernel:
layer 0 7.51 / 13.92, layer 4 12.39 / 19.16, group 0-1 14.00 / 24.67, group 4-5 22.26 ms with the
kernel. An end-to-end sweep A/B on an earlier tree (unrounded elementwise mHC) measured the opposite,
28.6 against 31.4 output tok/s at concurrency 32, so the kernel stays opt-in until an A/B on the
merged tree.

What the kernel buys depends on the graph around it: in GLM-5.3-Flash's layer graphs the NKI call first
made the layer 8x SLOWER (next section) until the hyper-connection arithmetic was rewritten; with that,
the KDA mixer at C=512, 8 heads, is 1.78 ms of which the kernel is 0.78 ms (chunk_scan: 3.01).

CPU suite on the merged tree (feat/kda-kernel 033c5a9 = engine-v0 45572db + this work), one pytest process
per test file on a trn1.2xlarge host, transformers 5.18 on PYTHONPATH (test_inkling.py under the venv's
5.15, which it needs): 624 passed, 9 skipped, 0 failed (2026-10-04).

Not done here: decode and verify through a kernel (they keep recurrent_step); Kimi-Linear-48B's
unbounded KDA gates (the diagonal sub-chunk would need fla's per-token path); trn2 (the kernel has no
generation-specific instruction; it was not compiled or run on trn2).

## The layer-graph spill was the hyper-connection arithmetic (2026-10-04, SDK 2.32, trn1.2xlarge)

GLM-5.3-Flash's layer graphs at prefill were 3-8x the sum of their parts, and putting the KDA kernel
in made them worse. `tools/profile_layer.py --model zai-org/GLM-5.3-Flash --tp 32 --attn-tp 8 --ranks 2
--prefill 512 --what layers --part-layers 0 --sum-readback --neff` (new flags: `--attn-tp` builds the
mixers at attention TP 8's shapes with the MoE at tp=32's, `--neff` prints each graph's instructions
and spill DMA queues through tools/neff_instructions.py, `--la-compare` runs a linear layer both ways),
`KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki`, layer 0 (dense MLP + KDA, 8 KDA heads per rank):

| mHC form | delta rule | KILN_CC_ARGS | layer 0 | instructions | spill DMA entries / runs |
|---|---|---|---|---|---|
| bmm (transformers' spelling) | torch chunk_scan | --model-type=transformer | 10.715 ms | 177,318 | 1,070 / 143,860 |
| bmm | NKI kernel (1, 2 or 4 units) | --model-type=transformer | 81.3 / 81.1 / 81.1 | 154,788 | 7,012 / 896,449 |
| bmm | NULL NKI kernel (no work) | --model-type=transformer | 81.618 | 144,919 | 6,832 / 863,425 |
| bmm | NKI kernel | (none) | 36.349 | 250,370 | 66,517 / 8,509,127 |
| bmm | torch | (none) | 38.852 | 273,513 | 66,449 / 8,499,010 |
| elementwise | torch | --model-type=transformer | 8.196 | 69,074 | 881 / 108,207 |
| elementwise | NKI kernel | --model-type=transformer | **3.790** | 32,020 | 781 / 98,873 |

(The same layer's blocks alone, bmm form and kernel: attention block 7.17 ms, FFN block 5.52, the mHC
collapse 3.68, the KDA mixer 1.90, the MLP 0.47; elementwise: attention block 2.96. Layer 4, MoE +
KDA: 14.79 ms torch / 79.85 ms kernel with the bmm form. Layer 3, MoE + DSA, no linear attention:
83.1 ms with the bmm form, 232K instructions, 6,814 / 882K spill.) A null kernel inflated the layer
exactly as the real one did, so it was not the kernel's body but any NKI call in that graph. The
neuronx-cc log of the inflated graph shows the SB allocator spilling 43% of 4,371 tensors in one round
(52.7 MB/partition requested, against 0.76 MB for the torch-path graph) and 26.7 GB of spill/reload DMA
traffic (4.4 GB of it optimised away), against 1.07 GB. The streams are [T, 4 x 4096]; the bmm form
normalises them to an fp32 [T, 16384] copy before the 24-logit projection, multiplies the fp32 pre
weights into a [T, 4, 4096] fp32 product to collapse them, and mixes them back as a per-token [4, 4]
by [4, 4096] batched matmul: large fp32 temporaries and thousands of tiny matmuls around the mixer.

`models/hybrid.py` `KILN_MHC_FORM`: `elementwise` (default) is vLLM v0.30.0's numerics
(vllm/model_executor/kernels/mhc/torch.py, mhc_pre_torch / mhc_post_torch) written without the big
temporaries: the RMS scale multiplies the 24 logits after the fp32 projection ((x r) W^T = (x W^T) r),
the collapse is four [T, 4096] multiply-adds in fp32, and each output stream post_n y + sum_m comb[m, n]
S_m is accumulated in fp32 from [T, 4096] rows; the collapse and each stream are then rounded to bf16
through a Veltkamp split in fp32 (c = 65537 x, hi = c - (c - x): exactly round-to-nearest-even bf16,
checked on 100K values) before the cast. `elementwise_fp32` is the same without the split (a probe:
see the numerics below), `bmm` transformers' spelling. All three are the same algebra; in fp32 they agree
with transformers to rounding (CPU: tests/test_glm5_next.py, test_linear_serving.py, test_qwen4_exp.py
pass under each).

**At the sweep's shapes** (`--tp 32 --dp-attention 4 --prefill 256`: 256 rows per DP group, the hidden
state [1024, 16384]; `--layers 8 --layer-groups 0-1 2-3 4-5 6-7` times the runner's 2-layer prefill
pieces; KILN_CC_ARGS=--model-type=transformer; ms, p50, trn1.2xlarge with two live ranks):

| MoE path | mHC / delta rule | layer 0 | layer 3 | layer 4 | group 0-1 | group 2-3 | group 4-5 | group 6-7 |
|---|---|---|---|---|---|---|---|---|
| prefill kernel (KILN_MOE_PREFILL_KERNEL=nki) | bmm / torch | 16.93 | 24.19 | 21.85 | 37.24 | 41.05 | 47.20 | 45.85 |
| same | elementwise_fp32 / NKI | 4.99 | 9.53 | 9.77 | 10.03 | 14.42 | 18.97 | 18.93 |
| dedupe kernel (the sweep's) | bmm / torch | 16.93 | 22.33 | 29.37 | 37.11 | 48.56 | | |
| same | elementwise_fp32 / NKI | 4.84 | 16.83 | 16.86 | 9.84 | 21.40 | 33.96 | 33.93 |

| prefill kernel | elementwise (Veltkamp rounding) / NKI | 5.87 | 10.26 | 10.43 | 11.40 | 15.74 | | |
| dedupe kernel | elementwise (Veltkamp rounding) / NKI | 5.74 | 17.65 | 17.63 | 11.56 | 23.00 | 35.37 | 35.36 |

(The sweep's own pieces at this shape were 57 / 72 / 82-87 ms, section above. Instructions, the last row:
layer 0 37,430, layer 3 414,136, layer 4 417,511, groups 75,030 / 451,279 / 834,733 / 831,089, i.e. about
0.42 M per MoE layer through the dedupe kernel (343 K of them on the PE), so a prefill group of 12 MoE
layers at 256 rows per group sits at neuronx-cc's 5 M limit and 10 fit; through the prefill kernel a MoE
layer was 126-130 K. Rounding the streams with the Veltkamp split costs about 1 ms per layer over the
unrounded form.)

**Numerics on the real weights** (zai-org/GLM-5.3-Flash at tp=32 on kiln-kda-32 / kiln-kda-32b,
trn1.32xlarge, `KILN_MOE_KERNEL=nki KILN_PIECEWISE_MOE_GROUP=12 KILN_CC_ARGS=--model-type=transformer
tools/check_ppl.py --tp 32 --piecewise --kv-cache-gb 1.0`; CPU: kiln-kda-cpu, x2iedn.32xlarge, `--device cpu
--weight-dtype bf16`; the wikitext-2 slice is `--text-file` of the joined test split, md5 3ce70e93,
first 3072 tokens through 256-token chunks; mean prompt logprob, per sentence France / Water boils /
def add / quick fox):

| form | device, 4 sentences | CPU, 4 sentences | device wikitext | CPU wikitext |
|---|---|---|---|---|
| bmm | -2.110 (-2.262 / -3.516 / -0.977 / -1.697) | -2.086 (-2.241 / -3.485 / -0.917 / -1.721) | -0.548 | -0.548 |
| elementwise_fp32 (no rounding barrier) | -1.785 (-2.072 / -2.537 / -0.953 / -1.517) | -2.105 (-2.190 / -3.600 / -0.969 / -1.699) | | |
| converts written at transformers' rounding points | -1.776 (-2.036 / -2.589 / -0.924 / -1.507) | -2.170 (-2.193 / -3.914 / -0.924 / -1.722) | -0.550 | |
| elementwise logits and collapse, bmm output mix | -2.090 (-2.186 / -3.524 / -0.967 / -1.718) | | | |
| elementwise (Veltkamp rounding), torch delta rule | -1.973 (-2.003 / -3.502 / -0.958 / -1.470) | | -0.552 | |
| elementwise (Veltkamp rounding), NKI delta rule | -1.980 (-2.013 / -3.514 / -0.979 / -1.446) | | -0.5515 | |

The four sentences are oversensitive ("Water boils" moved a nat from a few subnormal weight changes,
the loader A/B in bench/results/2026-10-03-trn2.48xlarge-moe.md); the 3071-token slice is the eval, and
on it bmm scores the same on the device and the CPU and so does the unrounded elementwise form (device
bmm against it: |dlogprob| mean 0.058, greedy agreement 97.4%, `tools/compare_ppl.py`). Per position
on the slice: device bmm against CPU bmm |dlogprob| mean 0.054 (max 1.68), greedy agreement 97.4%;
the Veltkamp form with the NKI delta rule against device bmm 0.056 (2.58), 97.8%, against CPU bmm 0.060
(2.37), 97.3%: as close to the CPU reference as bmm on the device is. By chunk (256 tokens) the
Veltkamp form with the kernel: -0.777 -0.693 -1.101 -0.929 -1.216 -0.180 -0.350 -0.141 -0.297 -0.273
-0.336 -0.325; device bmm -0.763 -0.693 -1.097 -0.927 -1.201 -0.181 -0.351 -0.140 -0.305 -0.272 -0.325
-0.323; CPU bmm -0.750 -0.708 -1.095 -0.920 -1.206 -0.184 -0.351 -0.143 -0.294 -0.271 -0.318 -0.335. On the CPU the
forms agree except on that sentence; on the device the unrounded form moved France and fox as well.
Two mechanisms, each measured: (1) the output mix itself is right on the device (`tools/
probe_mhc_output.py`: the fp32 mix bit-identical to the host, and a following fp32 op sees the rounded
value in a one-block graph), but inside the 12-layer group graphs a plain f32 -> bf16 -> f32 round trip
between elementwise ops was not kept (the "Water boils" jump disappears once the rounding is done by
the Veltkamp split, which leaves nothing to fold); (2) France and fox stay ~0.2 above the CPU with the
split in place, which neither the one-block probe nor eight consecutive blocks in one graph reproduce
(`tools/probe_mhc_blocks.py`, 256 rows, outlier streams, a cheap stand-in block: device against host
9.3e-3 of the streams' max for both elementwise forms, 1.4e-2 for bmm): not yet explained. Next step
when it matters: dump the streams after every layer for the France prompt on the device (tp=32, the 12-
layer groups) and on the CPU (`--device cpu --weight-dtype bf16`), and find the first layer where they
part by more than the bf16 noise of the bmm pair; the wikitext slice says the residue is not a quality
loss (the elementwise form is as close to the CPU reference there as bmm on the device is). Only the bmm output mix
(a bf16 batched matmul) matched the CPU on all four, and it is the form that spills (79 ms for layer 0
at the sweep's shapes with the KDA kernel in the graph).

## Where a GLM-5.3-Flash decode step goes (2026-10-04, SDK 2.32, trn1, the sweep's decode shape)

The sweep's decode step at 8 rows per DP group (DP 4, tp=32, 264 pages, `KILN_MOE_KERNEL=nki
KILN_PIECEWISE_MOE_GROUP=12`) is ~137 ms over four 12-layer pieces (38 / 35 / 35 / 27 ms, section "Where a
GLM-5.3-Flash prefill step goes"). `tools/profile_layer.py --model zai-org/GLM-5.3-Flash --tp 32
--dp-attention 4 --batch 8 --pages 264 --page-size 32 --what layers mlp --part-layers 0 3 4 --layers 24
--layer-groups 0-11 12-23 --sum-readback`, elementwise (Veltkamp) mHC, --model-type=transformer, random
weights; 2 live ranks on kiln-kda-32 cores 0-1, 32 live ranks on kiln-kda-32b (both trn1.32xlarge):

| graph | 2 ranks | 32 ranks |
|---|---|---|
| layer 0 (dense MLP + KDA) | 1.32 ms | 6.48 |
| layer 3 (MoE + pooled DSA) | 5.37 | 5.81 |
| layer 4 (MoE + KDA) | 2.18 | 5.95 |
| layer 3's MoE block (routed experts + shared expert + all-reduce, 32 rows) | 1.29 | 6.39 |
| layers 0-11 as one graph (3 dense KDA, 6 MoE KDA, 3 MoE DSA) | 27.26 (532,464 instructions) | 29.70 |
| layers 12-23 as one graph (9 MoE KDA, 3 MoE DSA) | 29.99 | 32.67 |

So the step is compute-bound: a 12-layer graph pays only 2.4-2.7 ms more at 32 ranks than at 2 (a one-layer
graph pays ~5 ms, the per-execution collective cost amortised over the group), and four such graphs plus
the prep / post graphs make the ~137 ms. Per layer at 2 ranks, summed over the 45 layers (the groups run
~0.8x the sum of their layers): the 42 MoE blocks ~54 ms (the dedupe kernel at 256 pairs reads ~160
distinct experts per layer), the 11 DSA attention blocks ~40 ms, the 34 KDA mixers with their mHC ~31 ms.

The DSA attention block alone (`tools/profile_mla.py --model zai-org/GLM-5.3-Flash --tp 8 --layers 4
--batch 8 --pages ...`, attention TP 8's shapes, one NeuronCore, gather mode, bisection selection): L =
512 / 2048 (dense regime) 1.11 / 1.01 ms, 4096 1.96 ms, 8448 3.66 ms (mask mode 3.66 too); and at 8448
with the top-k raised to cover every key (`--topk 8448`, attending all keys, no selection) 2.14 ms. The
selection over [8, 2112] pooled scores costs more than attending all 8448 keys: about 1.5 ms of the 3.66.
(`torch.topk` over the same rows: 6.34 ms; the mask scatter from 2048 indices 0.54 ms.)

Levers, by expected saving per step at this shape: (1) the DSA top-k selection as an NKI kernel (rows on
partitions, max8 / match_replace rounds or an in-SBUF bisection, the mask or the gather indices written
in place): the 1.5 ms per DSA layer over dense, ~15 ms per step, and the dense-2048 floor (1.0 ms) says
~2.6 ms per layer, ~29 ms per step, is the most the block can give; (2) two 24-layer decode graphs
instead of four 12-layer ones: ~2.4 ms per graph saved, ~5 ms per step (instructions ~1.1 M per graph,
within the 5 M limit; compile time doubles); (3) the MoE blocks are the largest item but already a
DMA-bound kernel at 32 rows; they amortise better at 16-32 rows per group.

## Exact DSA top-k as one NKI kernel, and what the selection really cost (2026-10-04, SDK 2.32, nki 0.6.0, trn1)

`kiln/kernels/dsa_topk.py` selects the `keep` largest of each row's scores in one NKI kernel:
the sign of the keep-th score from one count, then a radix search over its bit pattern (31 rounds of
`count(s >= float(prefix | 2^b))`, the candidate built by integer OR on the bit pattern, so no
float arithmetic ever rounds the threshold), then the lowest-index `room` of the tied scores by a
binary search over the position (12 rounds at 2112 pools). Rows are cut into pieces so rows x pieces
fill the 128 partitions (8 decode rows of 2112 pools: 16 pieces of 132), a row's count is one fp32
matmul with a block-diagonal 0/1 matrix. For GLM-5.3-Flash it also expands the pool selection to
tokens and adds the tail (the non-candidate pools), so `glm5_next.block_mask` returns its output
directly. `KILN_DSA_SELECT=nki` is now the default for every DSA selection (pooled and token-level,
`dsa_select.topk_mask`); on the host it runs `emulate()`, the same arithmetic in torch. Its source
CRC is a static `rev` argument (LNL's cache key does not hash NKI source).

**GLM-5.3-Flash was not running the selection the notes said.** `pooled_selection` passed
`KILN_DSA_SELECT` ("bisect", meaning dsa_select's exact float-order bisection) into
`glm5_next.block_mask`, whose own "bisect" was QSA's RANGE bisection (32 halvings of [min, max]).
That one is exact only when the halvings separate the keep-th score from the next lower one; on
scores spread over many orders of magnitude it keeps nearly every pool (device: 2066-2100 of 2112
for keep 512). It is renamed `range` (still `KILN_QSA_SELECT`'s default); `bisect`, `radix`, `nki`
and `topk` now mean the same algorithm in both modules. Regression test:
`tests/test_dsa_topk.py::test_default_pooled_selection_is_exact_on_wide_range_scores` (fails on
`range`).

**Exactness on the device** (`python tools/probe_dsa_select.py --rows 8 256 --keys 2112 --keep 512
--kinds pooled short ties zeros ulps wide equal randn --methods floor nki nki-flat nki-all qsa
bisect-index`, kiln-dk-2 trn1.2xlarge, every mask read back; "pooled" = sum_h w_h relu(q_h . k) with
w of both signs and exact zeros, "short" = only 300 pools visible): the kernel equals the stable-sort
tie rule AND its host emulation bit for bit on all 8 kinds at both shapes, with and without
vis_only. The range bisection ("qsa") fails on "wide" (8 / 8 and 256 / 256 rows wrong), dsa_select's
bisection fails on "wide" at 8 rows (414-430 selected: its split points come from the device's
sqrt), every other kind exact.

**Time of the selection alone** (same probe, p50 of synchronous calls, fp32 scores in, additive mask
out; "floor" is a graph that reads and writes the same tensor):

| rows x pools, keep 512 | floor | nki (16 pieces / 1) | nki, one piece per row | range bisection | dsa_select bisect |
|---|---|---|---|---|---|
| 8 x 2112 | 0.145 ms | 0.225 | 0.325 | 0.443 | 0.863 |
| 256 x 2112 | 0.40 | 0.68 | 0.68 | 1.00-1.02 | 1.50 |

**Inside the layer the selection was never the 1.5 ms** (`tools/profile_mla.py --model
zai-org/GLM-5.3-Flash --tp 8 --layers 4 --batch 8 --pages 264 --modes mask --select ...`,
`KILN_CC_ARGS=--model-type=transformer`, the pooled DSA attention block of layer 3 at attention TP 8,
8 decode rows over 8448 keys, random weights, kiln-dk-2):

| selection | decode block p50 |
|---|---|
| range (the old default) | 3.457 / 3.467 ms |
| nki | 3.299 / 3.321 |
| nki, also expanding to tokens and the tail in the kernel | 3.331 |
| dense (`--topk 8448`: no indexer scores, no selection) | 1.958 |
| cost probes (`--select` noindex / keysonly / noselect / nosoftmax, tools/profile_mla.py pooled_probe; WRONG outputs): no pool keys, scores or selection | 2.096 |
| pool keys only | 3.237 |
| pool keys and scores, no selection | 3.052 |
| pool keys without the softmax pooling (first token's key), scores, nki | 2.748 |

So of the 1.5 ms over dense, the selection is ~0.3 ms (0.15 after the kernel); ~1.0-1.1 ms is
rebuilding every pool's softmax-weighted key from all 8448 cached tokens' keys and gates in every
step (8 x 2112 x 4 x 128 on the vector engine, plus reading the 34.6 MB key-and-gate cache). That
is the next section's pool-key cache. A prefill chunk (`--prefill 256 --pages 264`, its first 256
positions) gives range 5.404 ms, dsa_select bisect 5.162, nki 4.872, dense 2.850. GLM-5.3 (`--model
zai-org/GLM-5.3 --tp 16 --batch 8`, its full-indexer layer, token-level DSA over 4096 / 8192 keys)
gives bisect 3.235 / 4.704 ms, nki 2.951 / 4.165, so the kernel is the default there too.

**Real weights** (zai-org/GLM-5.3-Flash at tp=32 on kiln-dk-32, trn1.32xlarge, `KILN_MOE_KERNEL=nki
KILN_PIECEWISE_MOE_GROUP=12 KILN_CC_ARGS=--model-type=transformer python tools/check_ppl.py --model
zai-org/GLM-5.3-Flash --tp 32 --piecewise --kv-cache-gb 1.0 [--text-file wikitext2_test.txt]`, feat/dsa-topk
d879f7f = engine-v0 e6ebe16 + this): 4 sentences -1.980 (-2.013 / -3.514 / -0.979 / -1.446), the same
as before because those contexts are dense (below 2048 tokens nothing is selected); the 3071-token
wikitext-2 slice, which runs every 256-token chunk in the sparse regime (104-page bucket, 3328 keys),
-0.5544, by chunk -0.770 -0.713 -1.103 -0.921 -1.217 -0.186 -0.358 -0.141 -0.301 -0.269 -0.323 -0.350.
The same tree and graphs layout with the old selection (`KILN_DSA_SELECT=range`, same box, same day):
-0.5515, by chunk -0.777 -0.693 -1.101 -0.929 -1.216 -0.180 -0.350 -0.141 -0.297 -0.273 -0.336 -0.325,
exactly engine-v0's numbers. `tools/compare_ppl.py` range vs nki: +0.0028 mean, |dlogprob| mean 0.049
(max 2.19), greedy agreement 97.6%, the spread of two compiles of one model (device bmm against the
Veltkamp form: 0.056, 97.8%, above); chunk 0, where every query sees fewer than 512 pools and both
select all of them, moves by 0.007 too, so part of the spread is the compiler, not the selection.
Logs: s3://<your-bucket>/logs/kiln-dk-32/20261004T082417Z-check_ppl.log, 20261004T085817Z-check_ppl.log;
/opt/kiln/logs/wt-range.log on kiln-dk-32 (capture, local compile and run: /opt/kiln/dk-wtr.sh).

CPU suite on that tree (kiln-dk-cf, r7i.48xlarge, one pytest process, `KILN_TEST_MODEL=Qwen/Qwen3-0.6B`,
the venv's transformers 5.15): 624 passed, 25 skipped, 0 failed in 20 min; that suite skips the GLM-5.3-Flash
and Qwen3.8-Flash-Next tests (they need transformers >= 5.18), which with transformers 5.18.0 on
PYTHONPATH (`pip install --no-deps --target`, with huggingface_hub 1.33.0, tokenizers 0.23.2) give
tests/test_glm5_next.py, test_qwen4_exp.py, test_linear_serving.py, test_dsa_topk.py, test_dsa_select.py:
158 passed, 1 skipped.

## Where a GLM-5.3-Flash prefill layer goes, block by block, and sequence-parallel streams (2026-10-04, SDK 2.32, trn1.32xlarge)

**The breakdown.** `KILN_CC_ARGS=--model-type=transformer KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki
KILN_LINEAR_ATTN_KERNEL=nki python tools/profile_layer.py --model zai-org/GLM-5.3-Flash --tp 32
--dp-attention 4 --ranks 32 --prefill 256 --pages 264 --layers 12 --what layers hcblocks --part-layers 0 3 4
--layer-groups 0-11 --sum-readback --neff` (the sweep's prefill shape: 256 rows per DP group, 1024 rows
into the MoE, 8448 keys; random weights in the checkpoint's formats, so the MoE takes the dequantize-
first path; kiln-pf-32, logs /opt/kiln/logs/20261004T075601Z-profile_layer.log; `--what hcblocks` is new:
each block of a hyper-connection layer as its own graph, at the hidden state's row count). p50 of
synchronous calls with a sum readback; a graph holding a cross-chip collective also pays the ~3-5 ms
fixed cost of "Collectives across chips" above, so the blocks with an all-reduce read high alone:

| graph (32 live ranks) | layer 0 (dense, KDA) | layer 3 (MoE, pooled DSA) | layer 4 (MoE, KDA) |
|---|---|---|---|
| whole layer | 7.50 ms | 15.85 | 12.28 |
| attention block: mHC + mixer + all-reduce + output mix | 6.15 | 7.88 | 6.62 |
| FFN block: mHC + MLP + all-reduce + output mix | 6.25 | 8.19 | 8.19 |
| mHC collapse alone (`_mhc`, [1024, 4, 4096]) | 1.01 | 0.96 | 0.80 |
| mHC output mix alone (`mix_out`) | 1.75 | 1.81 | 1.63 |
| token mixer alone, with its all-reduce | 5.97 | 6.09 | 6.99 |
| MLP alone, with its all-reduce | 5.76 | 6.33 | 6.87 |
| MoE routed experts alone (router + prefill kernel) | | 5.46 | 5.48 |
| router / shared expert alone | | 0.44 / 0.49 | 0.53 / 0.30 |
| world all-reduce [1024, 4096] bf16 alone | 6.84 | 5.79 | 5.27 |

DSA layer 3's attention block without its all-reduce (`mla.attention`, the group's 256 rows) 5.13 ms:
projections and indexer 0.86, the pooled selection over 2112 pools of 8448 keys 2.21, the attention core
with a given mask 2.30 (decompressed keys, KILN_MLA_PREFILL=expand) or 2.14 (absorbed), o_proj 0.50.
The 12-layer group 0-11 (the first prefill graph of a P=12 step: 3 dense KDA, 6 MoE KDA, 3 MoE DSA layers)
is 121.3 ms, 887,809 instructions, no spill. So the hyper-connection arithmetic was the largest item
after the MoE: the collapse and the output mix together ~2.7 ms per block alone, twice per layer, on all
1024 rows, replicated on every rank in fp32.

**Collectives inside a graph are cheap at this size** (`KILN_PROBE_SP=4 python tools/profile_layer.py
--allreduce --batch 32 --ranks 32`, new probe, log 20261004T074753Z, chained launches): a graph with one
world all-reduce of [32, 4096] 3.15 ms, with 12 of them 3.10; one world all-reduce of [1024, 4096] 5.80;
the replicated stream's per-layer exchange (two world all-reduces of [1024, 4096]) 5.73 ms for 1 layer and
8.83 ms for 12 in one graph, i.e. ~0.14 ms per further 8 MB all-reduce; a sequence-parallel exchange
(group all-gather [32 -> 256] + group reduce-scatter + world all-gather [32 -> 1024] + world
reduce-scatter per layer) 5.55 / 6.91 ms for 1 / 12 layers.

**Sequence-parallel streams (KILN_PREFILL_SP, default on for GLM-5.3-Flash).** Between the layers of a
prefill chunk each rank keeps only its own 1 / tp of the rows of the [R, 4 x 4096] streams; the mHC runs
on those R / tp rows, each block's normalised input is gathered to every rank by a zero-padded world
all-reduce (not all_gather_into_tensor: "A cached all-gather NEFF breaks in the next process" above), the
block runs exactly as before over all R rows (its own all-reduce included) and the rank keeps its rows
of the output; the prep graph takes the rank's rows of the embedding, the post graph gathers the final
hidden state. Two more world all-reduces per layer, the same per-row arithmetic (models/decoder.py
prefill_sp_enabled, models/hybrid.py layer; decode, verify and MTP keep replicated streams; ModelRunner
turns it off when a prefill bucket times dp_attention does not divide over tp). Same profile with
`--layers 24 --layer-groups 0-11 12-23 --sp` (log 20261004T081241Z):

| 12-layer prefill group, 32 live ranks | replicated streams | sequence-parallel (32 rows per rank) |
|---|---|---|
| layers 0-11 (3 dense KDA, 6 MoE KDA, 3 MoE DSA) | 121.3 ms, 887,809 instructions | 89.1 ms (-27%), 749,990 |
| layers 12-23 (9 MoE KDA, 3 MoE DSA) | 133.9 ms | 102.2 ms (-24%) |

A P=12 prefill step runs groups 0-11, 12-23, 24-35 (the 12-23 graph) and 36-44: about 490 -> 370 ms of
groups per 1024-token step at this shape. CPU (tests/test_dp_attention.py
test_glm5_next_sequence_parallel_streams, gloo, fp32 random GLM-5.3-Flash truncated): greedy tokens equal
to KILN_PREFILL_SP=0 at tp=4 / dp_attention 2 (piecewise) and tp=2, chosen-token logprobs within
4.3e-6 / 2.2e-6 (a CPU fp32 matmul over a few rows rounds differently from the same rows inside a bigger
one), and test_glm5_next against tp=1 within 3.8e-6.

End to end (kiln-g1-trn1, trn1.32xlarge, this tree, real weights, graphs compiled on the box: 1855 s of
bucket warm-up over 6 graphs): `KILN_PREFILL_SP=1 KILN_CC_ARGS=--model-type=transformer KILN_MOE_KERNEL=nki
KILN_MOE_PREFILL_KERNEL=nki KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12
KILN_LINEAR_ATTN_KERNEL=nki python bench/serve_sweep.py --model zai-org/GLM-5.3-Flash --device neuron --tp 32
--dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup
--max-num-seqs 32 --concurrency 32 --decode-buckets 8 --kv-cache-gb 1.0 --prefill-tokens 2048
--prefill-buckets 512 --price trn1.32xlarge-on-demand=21.50 --price trn1.32xlarge-spot=2.15` (log
/opt/kiln/logs/20261004T084143Z-sweep-sp1.log, s3 logs/kiln-g1-trn1/): conc 32 **57.2 out tok/s**, TTFT p50
23.2 s / p90 85.1 s, ITL p50 437 ms, spot $10.44 / M out, against 46.3 out tok/s (TTFT p50 30.5 s, ITL
548 ms) for the same configuration with replicated streams on the same box earlier the same day.

Real-weight ppl with the streams sequence-parallel (kiln-pf-32, `KILN_PREFILL_SP=1 KILN_CC_ARGS=--model-type=transformer
KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12
KILN_LINEAR_ATTN_KERNEL=nki python tools/check_ppl.py --model zai-org/GLM-5.3-Flash --tp 32 --piecewise
--kv-cache-gb 1.0`, log 20261004T085555Z-ppl4-sp1.log): France -2.140, Water boils -3.580, def add -0.932, quick
fox -1.690, mean -2.073 (44 tokens).

Baselines for the A/B above, the same tree with `KILN_PREFILL_SP=0`: the 4 sentences France -2.013, Water boils
-3.514, def add -0.979, quick fox -1.446, mean -1.980 (log 20261004T093541Z-ppl4-sp0.log); wikitext -0.552, by
chunk -0.777 -0.693 -1.101 -0.929 -1.216 -0.180 -0.350 -0.141 -0.297 -0.273 -0.336 -0.325 (20261004T095056Z;
with the streams sequence-parallel -0.548, by chunk -0.753 -0.693 -1.097 -0.912 -1.204 -0.172 -0.359 -0.141
-0.296 -0.265 -0.345 -0.339, 20261004T091054Z). The sweep with `KILN_PREFILL_SP=0` on kiln-g1-trn1, the same
command: 46.3 out tok/s, TTFT p50 30.5 s / p90 110.7 s, ITL p50 548 ms (log 20261004T092112Z-sweep-sp0.log). So
the sequence-parallel streams moved France and fox, the two sentences that sat ~0.2 above the CPU bf16
reference (-2.215 / -1.725) in "The layer-graph spill was the hyper-connection arithmetic" above, to 0.075 and
0.035 from it: with dp_attention 1 at tp=32 each rank's mHC runs on 1 row of a 32-token chunk, and the device
result is then closer to the host's than the same arithmetic over all 32 rows replicated.

## Pool keys cached, a prefill chunk's scores in the selection kernel, 24-layer decode graphs (2026-10-04, SDK 2.32, trn1)

**The pool-key cache** (`models/mla.py` pool_key_width / write_pool_keys, `KILN_DSA_POOL_CACHE=1` by
default). The previous section's probes put ~1.0-1.1 ms of the decode DSA block in rebuilding every
pool's key (the per-channel softmax of gate + ape over its 4 tokens, weighting their keys) from all
8448 cached tokens in every step. Now each pool's key (128 values) is stored as 4 pieces of 32, piece
i in the slot of the pool's i-th token, in a bf16 token-slot state beside K and V
(`DecoderForCausalLM.token_state_shapes`, so prefix sharing, the host tier and the page gathers carry
it with no new code; kept in the model dtype under an FP8 KV cache, computed from the cached rows
exactly as the per-call form does). Every write of a token recomputes its pool's key from the 4
cached indexer rows and rewrites the 4 pieces, so a pool holds the key of its current rows; once its
last token is written it is the key the per-call form computes, and only complete pools are ever
scored (the query's own incomplete pool is the tail). A page gather of the context reads [L, 32] =
[L / 4, 128]: the pool keys in order, 4.3 MB instead of the 34.6 MB key-and-gate gather at 8 rows x
8448 keys. CPU (`tests/test_glm5_next.py::test_pool_key_cache`, pages of 4 and 8, chunks of 6 that cut
pools in two): every complete pool's cached key equals `glm5_next.pool_keys` of its rows bit for bit
at every step, and tokens and logprobs equal the per-call form's; prefix caching, n-gram and MTP
speculation, DP attention, attention TP and the host tier (`test_linear_serving.py::test_hybrid_host_tier`,
new, both hybrids) pass unchanged.

**A prefill chunk's pooled scores inside the kernel** (`kiln_dsa_score_topk_kernel`,
`dsa_topk.score_select`, `KILN_DSA_SCORE_KERNEL=1` by default, one sequence only: B = 1). In a chunk
the pool scores sum_h w_h relu(q_h . k_p / sqrt(128)) are a [C, 32, P] fp32 tensor (69 MB at C = 256)
that XLA writes and reads back; the probes put ~1.5 ms of the C = 256 block there. The kernel keeps a
row tile's scores in SBUF: per head a bf16 matmul (the queries as the stationary operand, the pool keys
as moving, fp32 accumulation in PSUM; bf16 x bf16 products are exact in fp32), its scale and relu on the
scalar engine, the weighted head sum in fp32 on the vector engine (head 0 first), then the selection of
the same kernel source. On trn1.2xlarge it equals its host emulation (`emulate_scores` + `emulate`) on
256 and 512 rows, all or a third of the pools visible, at pool and token level (`python
tools/probe_dsa_select.py --fused --rows 256 512 --keys 2112 --keep 512`); the torch einsum form and the
kernel's order differ only in the last bits of the fp32 sums (`test_emulated_scores_are_the_pooled_indexer_scores`).

**The DSA block** (`tools/profile_mla.py --model zai-org/GLM-5.3-Flash --tp 8 --layers 4 --pages 264
--modes mask --select ...`, attention TP 8 shapes, 8448 keys, random weights, one NeuronCore of
kiln-dk-2; prefill: the chunk at positions 0..C-1):

| form | decode, 8 rows | prefill C = 256 | prefill C = 512 |
|---|---|---|---|
| range (engine-v0) | 3.46 ms | 5.40 / 5.52 | 9.84 |
| nki selection | 3.32 | 4.87 | 8.25 |
| + pool-key cache | 2.46 | 4.89 | |
| + scores in the kernel (prefill only) | 2.46 | 3.96 | 6.28 |
| dense (`--topk 8448`) | 1.96 | 2.85 | |

**The decode step at tp=32** (kiln-dk-32, trn1.32xlarge, random token ids, `KILN_PROFILE_PIECES=1
KILN_PROFILE_EXEC=1 KILN_CC_ARGS=--model-type=transformer KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki
KILN_MOE_PREFILL_KERNEL=nki KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PIECEWISE_MOE_GROUP=<G>
NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 python tools/time_decode.py --steps 64 --skip 8 -- --model
zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192
--output-len 256 --page-buckets 264 --warmup --max-num-seqs 32 --concurrency 32 --decode-buckets 8
--kv-cache-gb 1.0 --prefill-tokens 2048 --prefill-buckets 512 --max-seconds 1800`, every graph from the
compile cache; group pieces are timed synchronously on rank 0, the step is p50 (min) of 64 steps):

| tree | groups (ms) | step p50 (min) |
|---|---|---|
| A: feat/dsa-topk d879f7f with `KILN_DSA_SELECT=range` (engine-v0's selection), G = 12 | 37.4 / 34.1 / 34.1 / 25.8 | 141.2 ms (135.5) |
| B: d879f7f, nki selection, G = 12 | 36.9 / 33.4 / 33.5 / 25.6 | 139.2 (136.1) |
| C: + pool-key cache (wip 804a620), G = 12 | 33.9 / 30.6 / 30.7 / 23.4 | 128.3 (124.1) |
| D: C with two 24-layer decode graphs, G = 24 (prefill groups stay 12) | 60.5 / 50.2 | **120.7 (116.7)** |

So the DSA work takes 12.9 ms off the step (each 12-layer group with three DSA layers 3.4-3.5 ms, ~1.15
ms per DSA layer), and halving the graph count another 7.6 ms (the estimate was ~5): 141.2 -> 120.7 ms,
-14.5%. On the compile host (kiln-dk-cf, r7i.48xlarge, 8 graphs at once, `tools/compile_farm.py compile`)
a 24-layer decode graph took 732 / 662 s, peak 11.0 / 9.5 GB, NEFF 21.3 / 20.1 MB (a 12-layer one 365 s,
5.6 GB, 9.4 MB): twice the compile, the same NEFF bytes per step.

**End to end** (kiln-dk-32, trn1.32xlarge, one box, back to back on 2026-10-04 10:05-10:35 UTC, every graph
from the compile cache with `NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`; `KILN_CC_ARGS=--model-type=transformer
KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_PIECEWISE_PREFILL_MOE_GROUP=12
KILN_PIECEWISE_MOE_GROUP=<G> python bench/serve_sweep.py --model zai-org/GLM-5.3-Flash --device neuron --tp 32
--dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup
--max-num-seqs 32 --concurrency 32 --decode-buckets 8 --kv-cache-gb 1.0 --prefill-tokens 2048 --prefill-buckets
512 --requests 64 --max-seconds 3000 --price trn1.32xlarge-on-demand=21.50 --price trn1.32xlarge-spot=2.15`;
logs s3://<your-bucket>/logs/kiln-dk-32/sw-Ap.log, sw-Ep.log, sw-Fp.log):

| tree | G | out tok/s | TTFT p50 / p90 | ITL p50 | spot $ / M out |
|---|---|---|---|---|---|
| origin/scratch/sp-merge 57ed80a (engine-v0 + sequence-parallel prefill streams) | 12 | 57.3 | 23.2 / 85.1 s | 436.5 ms | 10.42 |
| feat/dsa-topk b495338 (that + the three changes above) | 12 | **64.4** | 22.2 / 81.0 s | 366.0 ms | **9.27** |
| the same with 24-layer decode graphs | 24 | 64.7 | 22.1 / 80.7 s | 364.4 ms | 9.23 |

+12.4% at concurrency 32, and below the p5en's $9.57 per 1M output tokens on spot. The 24-layer decode
graphs are within one run's noise here: the decode step is 6% shorter, but at concurrency 32 the ITL is
mostly decode steps waiting behind prefill steps. They stay a flag (`KILN_PIECEWISE_MOE_GROUP=24` with
`KILN_PIECEWISE_PREFILL_MOE_GROUP=12`, no code); at concurrency 64, where decode is a larger share of the
device time, they may matter: not measured.

**Real weights on the final tree** (same box and flags as the step-1 checks: `KILN_MOE_KERNEL=nki
KILN_PIECEWISE_MOE_GROUP=12 KILN_CC_ARGS=--model-type=transformer python tools/check_ppl.py --model
zai-org/GLM-5.3-Flash --tp 32 --piecewise --kv-cache-gb 1.0 [--text-file ...]`): 4 sentences -2.073 (-2.140
/ -3.580 / -0.932 / -1.690), exactly sp-merge's own (prefill agent, kiln-pf-32, above); wikitext slice
-0.5517 against sp-merge -0.5481 measured on the same box (by chunk -0.766 -0.700 -1.099 -0.933 -1.213
-0.175 -0.347 -0.143 -0.295 -0.281 -0.330 -0.338 against -0.753 -0.693 -1.097 -0.912 -1.204 -0.172
-0.359 -0.141 -0.296 -0.265 -0.345 -0.339); `tools/compare_ppl.py`: +0.0035, |dlogprob| mean 0.045 (max
1.88), greedy agreement 98.0%; engine-v0 (range) against sp-merge for scale: 0.057, 97.3%. CPU suite on
b495338 (one process, transformers 5.15): 630 passed, 28 skipped; the GLM-family and DSA files with
transformers 5.18: 269 passed, 8 skipped.

Measured but not adopted: without the selection scratch (`KILN_DSA_STAGE=0`; GLM-5.3-Flash has no IndexShare
layer that reads it) the decode block is 2.46 -> 2.38 ms and the C = 512 prefill block 6.25 -> 6.24 ms, ~0.9 ms
per decode step: not worth another recompile of every graph on its own.

**Open (2026-10-04): sequence-parallel streams on trn2 move the 4-sentence ppl.** The sp-merge tree
(57ed80a) on kiln-trn2-48 (tp=32 on 32 logical cores, LNC=2, `KILN_PREFILL_SP=1 KILN_LINEAR_ATTN_KERNEL=nki
KILN_MOE_KERNEL=nki KILN_PIECEWISE_MOE_GROUP=12 KILN_CC_ARGS=--model-type=transformer`, log
20261004T103253Z-check_ppl): France -2.122, Water boils -2.591, def add -0.961, quick fox -1.482, mean
**-1.807**, against -2.073 for the same tree on trn1 (Water boils -3.580) and -2.100 for the efa1da4 tree
(no SP) on trn2 (Water boils -3.521). "Water boils" near -2.5 is the signature the device showed when the
stream rounding was folded away (elementwise_fp32 on the device: -2.537), so the suspicion is that on
trn2 the Veltkamp rounding or the SP gather path is simplified differently. The trn2 throughput rows
with SP are therefore not backed by a ppl check on trn2. **Confirmed the same day:** `KILN_PREFILL_SP=0` on
the same tree and box (log 20261004T104937Z-check_ppl) gives France -2.229, Water boils -3.521, def add
-0.967, fox -1.705, mean -2.100, identical to the no-SP tree and to the CPU within 0.02. So SP folds the
stream rounding on trn2 only (check_ppl's short prompts put about one row per rank); SP stays proven on
trn1 only until a trn2 probe (tools/probe_mhc_rounding.py) shows a safe row count.

**So the default follows the platform** (models/decoder.py prefill_sp_enabled, PREFILL_SP_FAMILIES): with
KILN_PREFILL_SP unset the streams are sequence-parallel on trn1 and trn1n (kiln/platform.py target(), which
honours LNL's NEURON_PLATFORM_TARGET_OVERRIDE, so a compile-farm capture for a target decides as the device
does) and on a host without a Neuron device, and replicated on trn2, trn3 and inf2; KILN_PREFILL_SP=1 / 0
forces them either way (tests/test_dp_attention.py test_sequence_parallel_streams_default_per_platform). Two
guards for the trn2 question: test_sequence_parallel_streams_round_where_replicated_do (CPU: the same
hyper-connection blocks on rows split as SP splits them against the same rows in one batch agree to a CPU
matmul's rounding, 1 value in 1000 one bf16 step apart at most, while the blocks with the rounding left out
differ in most values), and `tools/probe_mhc_rounding.py` (device: 8 consecutive blocks in one graph at 1, 2,
8, 32, 64, 256 and 1024 rows against the host with the streams kept bf16 and kept fp32 between blocks,
"rounded" or "FOLDED" per row count). The command for the next trn2 session is in docs/price-performance.md
"How to resume". The trn1 reference (kiln-pf-k1, trn1.2xlarge, `KILN_CC_ARGS=--model-type=transformer python
tools/probe_mhc_rounding.py`, log 20261004T105823Z-mhcround2.log): at 1, 2, 8, 32, 64, 256 and 1024 rows the
device's rounding residual round(x) - x of the collapse is bit-identical to the host's (max |diff| 0, zero
residuals 0-0.02% on both): "rounded" at every row count.

**An EC2 Capacity Block shuts its instances down ~30 minutes before the block's end time.** Measured
2026-10-04 on cr-01dd0041d61815bee (ap-south-2b, end 11:30 UTC): kiln-trn2-48 was `shutting-down` with
`Client.UserInitiatedShutdown` at ~11:00 UTC, while nothing on our side had terminated it. Plan the last
run to end, and the cache/log backup to finish, before end time minus 30 minutes.

**The 4-sentence ppl has a knife-edge on "Water boils".** That sentence moves between about -3.5 and
-2.5 under three unrelated small perturbations (an experts-only subnormal-level weight change via
`KILN_MOE_E4M3_FIT=group` on trn1: mean -1.850 vs -2.072; 22dd2c2's fit on trn2; SP on trn2), while the
other three sentences move by at most ~0.04. It looks like one discrete decision (a MoE routing or DSA
near-tie on some token) flipping, not a broad numerics change. So the 4-sentence mean alone cannot
accept or reject a change; the wikitext slice (3071 tokens) is the eval that decides. The trn2 SP
question stays open until `tools/probe_mhc_rounding.py` and the SP=1/0 wikitext pair run on a trn2.

**Prefill 8192 at conc 64 does not fit trn1 (2026-10-04).** G64-8192-P12-K on the DSA tree (one 8K chunk,
2048 rows per DP group; the runtime-exact farm model said 14.87 GiB) failed with `Could not load the
model status=4 message=Allocation Failure` on kiln-mimo-trn1 (log 20261004T114039Z). The runtime's table
(/tmp/neuron_mem_table_device_13_nc_1.log): TOTAL 15.893 GB = code 228.7 MB + tensors 12.585 GB + shared
scratchpad 320 MB + **DMA rings spill 2.682 GB** + small rest, with the loaded NEFFs (decode groups) at
~2-3 MB of spill rings each, so the 2048-row prefill group being loaded carried ~2.67 GB of spill DMA
rings that the farm's spill-run count did not see. Until the estimator reads those rings, configs with
2048-row prefill groups need a device try.

**The sweep configs were KV-bound (found 2026-10-04 by the DSA work).** At the 8448-token context each
in-flight sequence needs 264 pages, and the KV budgets used all day were short of max-num-seqs x 264 per
DP group: G16 at 0.5 GB had 993 pages per group vs 4 x 264 = 1056, F0 at 1.0 GB ~1986 vs 2112, G64 at
1.0 GB fp8 3972 vs 4224 (3641 with the DSA tree's separate pool-key cache). So requests waited for
admission. Sizing KV for all sequences plus ~10% is a throughput lever: G64-4096 at KV 1.25 fp8 serves
84.0 out tok/s against 72.5 at KV 1.0. serve_sweep on feat/dsa-shape prints a `kv:` line per level
(pages per group vs need, KV-bound or fits, admitted sequences, preemptions).

## The prefill MoE kernel on the real expert layout: skipping lane tiles past the routing, the dequantize-first path (2026-10-04, SDK 2.32, nki 0.6.0, trn1.2xlarge)

One call of kernels/moe_prefill.py on layer 3's real experts of rank 0 as the loader holds them (per-row gate_up
tile scales, per-column down factors: the kernel's per-row and per-column paths; `python tools/probe_moe_prefill.py
--experts-file <glm53f_l3_r0_experts.pt> --chunks 1024 2048 --decode-max 0 --iters 20`, kiln-pf-k1; uniform
routing, distinct experts per token; ms without the readback reduction):

| form | C=1024 | C=2048 |
|---|---|---|
| engine-v0 (per-row gate_up, per-column down) | 5.948 | 8.035 |
| down made chunk-constant only (`--block-constant`) | 5.548 | |
| + every unused lane-tile segment skipped (`KILN_MOE_PREFILL_SKIP=20`) | **5.156** | **6.339** (skewed routing `--skew`: 6.322 against 8.039) |
| opt-in `KILN_MOE_E4M3_FIT=group` (dequantize-first gate_up, per-chunk down), scalar-engine share 0 / 3 | - / 4.547 | 6.738 / 6.363 |
| group fit, share 3, segments of 20 | 4.213 | 6.326 |

Kernel against its emulation 0.0034-0.0036 of the output's max on the per-row forms (as before) and 0.0018
on the dequantize-first ones, every row.

**Where the time went** (`tools/prof_engines.py` on the C=1024 graphs, instructions binned per 250 us): the
lane-tile loop is all but ~0.5 ms of the call, and in it the vector engine is the busy one on both paths:
the dequantize-first path issues 128 tensor_scalar ops per lane tile (64 per block, ~140 ns each in the
kernel: ~20.8 us per tile), the per-row path 64 scalar_tensor_tensor ops reading each tile's PSUM partial
(~180-270 ns each) and 8 tensor_tensor multiplies by the down factors (~320 ns), ~25.8 us per tile, while the
scalar engine is ~35% busy and the tensor engine waits on the vector engine. Of the 206 lane tiles at C=1024
(412 blocks of 64 lanes, the static bound every routing fits) the routing uses ~144; at C=2048 ~166 of 270.
The unused ones run the same instructions on stale data (their DMAs are skipped, their outputs never read).

**Skipping them (`KILN_MOE_PREFILL_SKIP=n`, kernel argument skp, default off for now).** The tiles past the
first half form segments of n tiles, each its own four-stage pipeline (prologue to epilogue) inside a device
loop (`nl.fori_loop`) of trip count [first tile of the segment < the routing's block count / 2], the count
being the last entry of the plan's inclusive prefix sum. The arithmetic of every used tile is unchanged. What
it took on trn1 / nki 0.6.0, each measured:
- `for _ in nl.dynamic_range(r)` around this kernel's steps segfaulted the NKI tracer (no message);
  `nl.fori_loop(0, r, body)` with a nested `def body(it)` traces (and warns that dynamic_range is deprecated);
  `functools.partial` as the body is rejected ("NKI classes must inherit from either Enum or NKIObject").
- A PSUM tensor referenced in two device-loop regions fails the backend: `[NCC_IBIR092] Live-in/Live-out
  MemoryLocation: ... not allocated to MemoryType: DRAM`; SBUF tensors may be. So each segment allocates its own
  gate_up accumulators, and the pipeline does not continue across segments.
- Compile time grows with the number of loop regions: 10-tile chunks with a drain loop after each (41 regions
  at C=1024) took 455-494 s per kernel graph, nearly all of it in the backend's ModuleForkPass build_fdeps /
  dep_opt (~140 s each, three times), against ~20 s without loops; segments of 20 (5 regions at C=1024, 6 at
  C=2048) take 30-90 s. Segments of 12 / 20 / 30 at C=2048: 6.572 / 6.339 / 6.591 ms (more pipeline restarts
  against coarser skipping).

The skip changes no used tile: `tools/probe_moe_prefill.py --experts-file <...> --chunks 1024 2048 4096
--compare-skip 20` (kiln-pf-k1, log 20261004T112748Z-cmp.log) found the skp=20 output bit-identical to the
skp=0 one at C=1024, 2048 and 4096 (block size 128) and with skewed routing, and `--routing hot` (every token
on 8 experts: every segment skipped) and `--routing maxblocks` (65 or 1 pairs per expert, 539 of 540 blocks
used: every segment runs) keep the kernel's distance from its emulation (0.0016-0.0033). The cost of the worst
case: maxblocks at C=2048 9.52 ms against 8.12 without the skip (the segments' pipeline restarts). C=4096
(B=128, 542 lane tiles, 308 used): skp 0 / 40 / 20 12.96 / 9.85 / 9.55 ms, compile 104 / - / 154 s per kernel
graph. Real-weight ppl with the skip (kiln-pf-32, `KILN_PREFILL_SP=1 KILN_MOE_PREFILL_MIN_TOKENS=1
KILN_MOE_PREFILL_SKIP=20`, logs 20261004T111433Z-ppl4-skip20, 20261004T110219Z-pplwiki-skip20): the 4 sentences
-2.127 / -3.543 / -0.980 / -1.686, the same values as without it; wikitext -0.550 (-0.551 without; the two
runs come from trees whose kernel source differs, |dlogprob| mean 0.044 per position, greedy 98.1%).

**End to end the skip buys ~2%** (same box, same tree, `KILN_MOE_PREFILL_SKIP=20` against `=0`): kiln-pf-32b,
feat/prefill-moe (a89807b + the MoE commits), conc 32, prefill 2048 / 512, P=12, KDA kernel, SP: 59.0 against
58.0 out tok/s (TTFT p50 22.4 / 22.9 s, ITL p50 423 / 431 ms; logs 20261004T105635Z-sweep-skip20,
20261004T113811Z-sweep-skip0); kiln-g1-trn1, feat/prefill-moe2 64d33fb (engine-v0 b3a4c7a + the MoE commits),
farm graphs (q/pm2-trn1), the DSA tree's G64-4096-P12-K command (conc 64, prefill 4096 / 1024, decode 16, KV 1.0
GB fp8, `KILN_DSA_SELECT=nki`): 73.8 against 72.5 (TTFT p50 37.8 / 38.6 s, ITL p50 644 / 656 ms; logs
20261004T112521Z-g64-s20, 20261004T114448Z-g64-s0), and F0-4096-P12-K (conc 32, decode 8, KV 1.0 GB): 72.8
against 72.1 (TTFT p50 18.6 / 18.8 s, ITL p50 320 / 324 ms; logs 20261004T115910Z-f0-s20,
20261004T122159Z-f0-s0). These configurations were also KV-bound (see "The sweep configs were KV-bound"),
which caps what a faster prefill step returns. Real routing needs more blocks than uniform routing
(`python tools/routing_stats.py --text-file <wikitext> --tokens 4096` on kiln-pf-32b's host CPU: layer 3's
router on the real hidden states of one wikitext sequence; log 20261004T120452Z-routing.log):

| C (B) | blocks used, real / uniform | static blocks | lane tiles a skp=20 kernel runs, real / uniform (of all) |
|---|---|---|---|
| 512 (64) | 283 / 288 | 348 | 154 / 154 (of 174) |
| 1024 (64) | 296 / 288 | 412 | 166 / 146 (of 206) |
| 2048 (64) | 383 / 333 (largest load 167 against 78) | 540 | 210 / 170 (of 270) |
| 4096 (128) | 388 / 312 (largest load 359 against 146) | 541 | 401 / 321 (of 541) |

so on one sequence's chunk the skip leaves 22-26% of the tiles instead of 37-41% (a serving chunk mixes
several sequences, between the two).

**`KILN_MOE_E4M3_FIT=group`** (models/loader.py, opt-in): the routed experts' FP8 blocks halved per (row group,
block) instead of per (row, block), the rank's 64 gate rows, 64 up rows and 128 down output rows each one
group, so the scales are block-constant and the kernel takes its dequantize-first gate_up and per-chunk down
paths. Halving is exact except for codes below 2^-5 (made subnormal), which round to even; the per-row fit
rounds the same codes of the rows it halves (tests/test_quant.py test_fit_e4m3_max_row_groups). It is the
experts-only form of feat/trn2-bench 22dd2c2's fit, which moved one of the four sentences by a nat on trn2
(bench/results/2026-10-03-trn2.48xlarge-moe.md, "Loader A/B"); it stays off unless real-weight ppl at tp=32
agrees with the per-row fit. **The scalar-engine share** (kernel argument asp, `KILN_MOE_PREFILL_ACT_SPLIT`,
default 3, dequantize-first path only): of each block's eight half-tile dequantizations per round, asp run on
the scalar engine (`nisa.activation` copy with the scale, the same fp32 product rounded once to bf16): random
dq-layout experts at C=1024 asp 0 / 2 / 3 / 4: 4.969 / 4.656 / 4.447 / 4.814 ms, C=8192 15.237 / 14.791 /
14.690; group-fitted real experts at C=2048 asp 0 / 2 / 3 / 4 / 5: 6.738 / 6.557 / 6.363 / 6.696 / 6.426.

**The group fit fails real-weight ppl** (kiln-pf-32, trn1.32xlarge, tp=32, `KILN_PREFILL_SP=1
KILN_MOE_PREFILL_MIN_TOKENS=1` so the 32- and 256-token chunks take the prefill kernel, the sweep's env,
`tools/check_ppl.py --model zai-org/GLM-5.3-Flash --tp 32 --piecewise --kv-cache-gb 1.0 [--text-file ...
--out-json ...]`, logs 20261004T103939Z-ppl4-row, 20261004T105018Z-ppl4-grp, 20261004T101558Z-pplwiki-row,
20261004T102743Z-pplwiki-grp):

| fit | France | Water boils | def add | quick fox | mean | wikitext-2 |
|---|---|---|---|---|---|---|
| per (row, block) (engine-v0) | -2.127 | -3.543 | -0.980 | -1.686 | -2.072 | -0.551 |
| per (row group, block) | -2.166 | **-2.518** | -0.966 | -1.695 | **-1.850** | -0.554 |

Per position on the wikitext slice (`tools/compare_ppl.py`): |dlogprob| mean 0.053, max 2.23, greedy
agreement 97.6% (that includes the two paths' different arithmetic, dequantize-first against per-row). So
the 4-sentence mean moves by 0.22, nearly all of it Water boils, as it did on trn2 with feat/trn2-bench
22dd2c2's fit of every FP8 tensor (-2.486) and with sequence-parallel streams on trn2 (-2.591): three
unrelated perturbations, the same sentence, the same value, the other three sentences within 0.04. That
sentence looks bimodal (a discrete choice for one of its tokens, a routing or selection near-tie, flipping
either way), so "Water boils near -2.5" is not by itself evidence of the streams' rounding being folded; the
wikitext slice and tools/probe_mhc_rounding.py are the tests for that. KILN_MOE_E4M3_FIT stays "row".

## Where a GLM-5.3-Flash prefill layer goes with sequence-parallel streams (2026-10-04, trn1.32xlarge, prefill 2048 / 512 per group)

`KILN_CC_ARGS=--model-type=transformer KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_LINEAR_ATTN_KERNEL=nki
KILN_PROFILE_EXPERT_LAYOUT=loaded python tools/profile_layer.py --model zai-org/GLM-5.3-Flash --tp 32
--dp-attention 4 --ranks 32 --prefill 512 --pages 264 --sum-readback --neff --layers 12 --what layers hcblocks
--part-layers 0 3 4 --layer-groups 0-11 --sp` (kiln-pf-32b, logs 20261004T100759Z-prof512.log and, with
`KILN_MOE_PREFILL_SKIP=20`, 20261004T103617Z-prof512-skip20.log; `KILN_PROFILE_EXPERT_LAYOUT=loaded` gives the
random experts the per-row and per-column scales the loader leaves GLM-5.3-Flash's real ones with, so the
MoE kernel takes its real paths; 32 live ranks; a block graph holding a collective also pays the ~3-5 ms
fixed cost; the null graph is 0.25 ms):

| piece (graph) | layer 0, dense KDA | layer 3, MoE DSA | layer 4, MoE KDA |
|---|---|---|---|
| whole layer, replicated streams | 11.70 ms | 26.32 | 20.24 |
| mHC collapse / output mix, replicated (2048 rows) | 1.64 / 3.07 | 1.62 / 3.12 | 1.66 / 3.11 |
| the same, sequence-parallel (64 rows) | 0.47 / 0.54 | 0.38 / 0.43 | 0.50 / 0.34 |
| SP attention block / SP FFN block | 6.68 / 6.66 | 11.66 / 13.89 | 6.47 / 14.44 |
| SP FFN block with `KILN_MOE_PREFILL_SKIP=20` | | 11.09 | |
| MoE routed (router + kernel) / with the skip | | 8.94 / 6.62 | 8.83 / - |
| router / shared expert | | 0.56 / 0.34 | 0.73 / 0.34 |
| DSA attention without its all-reduce (mla.attention) | | 7.84: projections 0.95, pooled selection 3.64, core 4.76 (expand) / 5.01 (absorb), o_proj 0.71 | |

12-layer prefill group 0-11: replicated streams 221.8 ms, sequence-parallel 159.5 ms; with the MoE skip
243.0 / **136.2 ms** (the replicated group slower with it: not investigated, SP is the default on trn1).
The KDA mixer at attention TP 8's shapes, 512 rows, one rank (`profile_layer.py --tp 32 --attn-tp 8 --prefill
512 --what laparts --part-layers 4`, kiln-pf-k1): 1.61 ms, of which the delta-rule kernel 0.77 (chunk_scan
3.01), in_qkv 0.41, the gates 0.23, the short conv 0.22, the output projection 0.23.

So per 2048-token prefill step with SP and the skip, by layer kind (42 MoE layers, 11 DSA, 34 KDA, 45 mHC
pairs): the routed MoE about 42 x (6.6 + 0.9) = 315 ms, DSA attention 11 x 7.8 = 86 ms (selection 40, core
52), KDA 34 x 1.6 = 54 ms, four world all-reduces of [2048, 4096] per layer about 45 x 1.1 = 50 ms, the
sequence-parallel mHC about 45 x 0.7 = 32 ms. The MoE stays more than half; at C=2048 its used lane tiles are
bound by their software-DGE loads (x rows and expert blobs, ~2.6 MB per tile; the combine re-reads 128 MB of
Y), which neither the skip nor the dequantize-first path changes.

engine-v0 c7836c0 (every change of the day merged: DSA tree, SP trn1-default, KDA kernel default, MoE
prefill skip off) on real weights at tp=32 (kiln-mimo-trn1 log 20261004T125630Z-check_ppl): France -2.140,
Water boils -3.580, def add -0.932, fox -1.690, mean -2.073, identical per sentence to the sp-merge and DSA
trees.

## The pool-key cache's bytes, and keeping the keys in the KV rows instead (2026-10-04, SDK 2.32, trn1)

The DSA tree (engine-v0 3a55b9c) lost to sp-merge at two sweep shapes (same-box A/Bs, price-performance
iteration log): concurrency 16 with prefill 4096 / bucket 1024 and a 0.5 GB bf16 KV cache, 62.9 -> 60.6 out
tok/s, ITL 168 -> 169 ms but TTFT p50 13.3 -> 14.9 s; concurrency 64 with prefill 2048 / 512 and a 1.0 GB FP8
cache, 67.7 -> 66.6, ITL 770 -> 713 ms but TTFT p50 25.1 -> 42.7 s. Faster steps and later first tokens is
admission, not compute: at 8448-token contexts every one of these configurations is KV-bound (G16: 993 pages
per DP group against the 4 x 264 its four sequences need; G64 FP8: 3972 against 16 x 264), and the separate
pool-key cache (bf16, 32 values per token per DSA layer, also counted for the unbuilt MTP layer) took 4% of
the pages from a bf16 cache and 8.3% from an FP8 one (G64: 3972 -> 3641), so fewer sequences fit and more
were preempted.

**The kernels are not it.** `tools/profile_mla.py --model zai-org/GLM-5.3-Flash --tp 8 --layers 4 --pages
264 --modes mask` (kiln-dk-2b, trn1.2xlarge, attention TP 8 shapes, 8448 keys, random weights), at the two
shapes the earlier tables did not cover:

| form | prefill chunk, 1024 rows | decode, 4 rows | decode, 16 rows |
|---|---|---|---|
| range selection (sp-merge), keys rebuilt per call | 18.15 ms | 2.21 | 5.82 |
| nki selection, torch scores, keys rebuilt per call | 15.21 | 2.06 | 5.67 |
| nki selection and the fused score kernel, keys rebuilt per call (`KILN_DSA_POOL_CACHE=off`) | 11.16 | 2.06 | 5.67 |
| ... + separate pool-key cache (the merged default) | 11.19 | 1.63 | 4.03 |
| ... + in-place pool keys, read from the context's rows | 11.24 | 1.81 | 4.69 |
| ... + in-place pool keys, read through the strided pool view (new default) | | 1.70 | 4.34 |
| dense (`--topk 8448`) | 7.39 | | |

**In-place pool keys** (`KILN_DSA_POOL_CACHE=inplace`, models/mla.py write_pool_keys_inplace and
_inplace_keys). Once a pool's key exists its tokens' indexer keys and gate logits are dead, so the key
overwrites the indexer-key half of the pool's LAST cached row. Only that row is ever overwritten, so a
pool can always be recomputed from its first three rows and a fresh last row: a rejected speculative
token at a pool's end, a later write of an incomplete pool, a padded row (position 0, never a pool's last)
all find what they need. No bytes: the page count is sp-merge's. The read takes the key part of every
pool's last row of the whole cache as a strided view, [slots / 4, 128], and gathers that by the block
table, which is ~0.3 ms per layer cheaper at 16 rows than gathering the context's rows and slicing.
`KILN_DSA_POOL_CACHE=auto` (default) is inplace for a bf16 / fp32 KV cache and off for FP8 (an FP8 row
would round the key to e4m3, and a separate bf16 cache costs 8% of the pages); `separate` keeps the merged
form, and 1 / 0 are its old spellings. CPU (`tests/test_glm5_next.py::test_pool_key_cache`, two engines in
lockstep on the same requests): every complete pool's last row in the inplace engine holds the separate
engine's key and every other row is unchanged; inplace, separate, off and transformers' greedy give the same
tokens; `test_pool_cache_off_under_fp8_kv`; the prefix-cache, speculation, MTP, DP-attention and host-tier
tests run inplace by default. (Those per-ulp key checks compare with a 1e-6 tolerance now: on a 192-vCPU host
CPU torch rounds one exp of the same softmax differently by the shape of the batch it sits in.)

`bench/serve_sweep.py` now prints a `kv:` line per level: pages per DP group against the pages its share of
the requests in flight needs at full length, KV-bound or fits, the most and mean sequences admitted at once,
and the preemptions of the finished requests.

**Same-box sweeps** (kiln-dk-32, trn1.32xlarge, 2026-10-04 12:00-14:00 UTC, the q/dsa-trn1 and q/sp-trn1
farm configs, 64 requests, every graph from the cache, `bench/serve_sweep.py` with the kv: line; "old" is
engine-v0 3a55b9c / b27600e graph code with the separate pool cache, "new" this branch with eager admission;
out tok/s, then pages per group, preemptions, ITL p50):

| config | KV | sp-merge 57ed80a | old DSA tree | new (in-place / off under FP8) |
|---|---|---|---|---|
| G16-4096 (conc 16, prefill 4096 / 1024) | 0.5 GB bf16, KV-bound | 62.9 (lead's run, same box) | 60.6 (lead's run, 950 pages) | **66.9** (992, 0, 157 ms) |
| G16-4096 | 0.65 GB bf16, fits | 72.7 (1290, 0, 177 ms) | 77.8 (1234, 0, 165 ms) | 77.2 (1290, 0, 166 ms) |
| G64-2048 (conc 64, prefill 2048 / 512) | 1.0 GB FP8, KV-bound | 62.4 (3971, 4, 515 ms) | 64.0 (3640, 4, 515 ms) | **64.6** (3971, 4, 498 ms) |
| F0-4096 (conc 32, prefill 4096 / 1024) | 1.0 GB bf16, KV-bound | 67.4 (1985, 8, 360 ms) | 72.2 (1899, 16, 323 ms) | 71.7 (1985, 8, 338 ms) |
| F0-2048 (conc 32, prefill 2048 / 512) | 1.0 GB bf16, KV-bound | 57.3 (my earlier run) | 64.4 (my earlier run) | 60.5 (1985, 16, 413 ms) |

(G64's sp-merge is 62.4 here against 67.7 on kiln-g1-trn1: at concurrency 64 with 64 requests one wave of
preemptions decides the run, so G64 compares within a box only.) Once the pool fits, the in-place cache equals
the separate one (77.2 against 77.8) at no bytes; where the pool binds, it gives back the pages.

## Admission: reserve the pages a request will need (2026-10-04, SDK 2.32, trn1)

At 8448-token contexts every sweep configuration above is KV-bound, and the scheduler admitted a waiting
request whenever its first prefill chunk fitted, then preempted the youngest running request when decode
ran the pool dry. For a linear-attention model a preemption recomputes from the victim's last state
checkpoint, so admitting a prompt that cannot be finished wastes its prefill and stalls the decodes
behind it. The kv: line shows it: at F0-2048 with a 1.0 GB bf16 pool (1985 pages per group against the
2112 its eight full-length sequences need) the in-place tree preempted 16 of 64 requests and gave 60.5
out tok/s, the same on three runs on two boxes, while the merged DSA tree, whose separate pool-key cache
leaves only 1899 pages, preempted 16 too and gave 64.3: with more pages more prompts get in and are
preempted later, after more work, though its decode step is the same (tools/time_decode.py at that shape:
130.9 against 129.8 ms).

`SchedulerConfig.admission = "reserve"` (EngineConfig.admission, `KILN_ADMISSION`; default): a waiting
request is admitted only when the free plus evictable pages cover its whole prompt and its own decode
reserve (up to `admission_decode_tokens` = 256 more tokens) on top of what the running requests still
owe (the rest of their prompts plus up to 256 decode tokens each); with nothing running it is admitted
anyway, and preemption stays as the fallback when a generation outruns its reserve. "eager" is the old
behaviour. CPU: tests/test_scheduler.py (a 40-page pool for eight 30-page sequences: reserve runs one at a
time with 0 preemptions and exact outputs, eager preempts; 120 pages: four at a time, 0 preemptions; a
2-page decode reserve overcommits and falls back to preemption, exact; overlapped scheduling, 0
preemptions); the tests of the preemption path pin `admission="eager"`.

Measured (kiln-dk-32 unless noted, same configs and box as the table above, out tok/s, then preemptions,
the most sequences admitted at once, TTFT p50 / ITL p50):

| config | eager | reserve (decode reserve 256 tokens) |
|---|---|---|
| F0-2048, KV 1.0 bf16, in-place | 60.5 (16, 32, 22.7 s / 413 ms) | **66.7** (0, 28, 24.4 s / 338 ms); 32-token reserve 66.6 |
| F0-2048, separate pool cache (1906 pages), kiln-g2-trn1 | 63.8 (16, 32, 22.1 s / 374 ms) | 67.3 (0, 28, 24.3 s / 336 ms) |
| F0-4096, KV 1.0 bf16, in-place | 71.7 (8, 32, 18.7 s / 338 ms) | **75.1** (0, 28, 20.1 s / 299 ms) |
| G16-4096, KV 0.5 bf16, in-place | 66.9 (0, 16, 12.6 s / 157 ms) | 66.5 (0, 12, 19.5 s / 132 ms) |
| G16-4096, KV 0.65 bf16 (fits), in-place | 77.2 (0, 16) | 77.4 (0, 16), kiln-g2-trn1 |
| G64-2048, KV 1.0 FP8, pool keys off | 64.6 (4, 64, 113 s / 498 ms) | 65.1 (0, 60, 113 s / 488 ms) |

Where eager admission preempted, reserving takes 4-10% more throughput and lowers the ITL; where it did
not (G16-4096 at 0.5 GB), the throughput is the same and the latency moves from ITL to TTFT (it runs 3 per
group instead of letting a fourth in that would have fit only because others finish first); where the
pool fits it changes nothing. Against the separate pool cache the in-place one gives back the pages but,
with reserve admission, those pages admitted no more sequences at F0-2048 (7 per group either way) and the
separate read is ~0.1 ms per layer cheaper at 8 rows: 66.7 against 67.3 (two boxes). In-place stays the
default for its zero bytes; at shapes where 4% of the pool decides a whole sequence per group it is the one
that admits it.

**Checks of feat/dsa-shape** (merged with engine-v0 8f45cd4): real weights on kiln-dk-32 (tp=32,
`KILN_MOE_KERNEL=nki KILN_PIECEWISE_MOE_GROUP=12 KILN_CC_ARGS=--model-type=transformer tools/check_ppl.py
--model zai-org/GLM-5.3-Flash --tp 32 --piecewise --kv-cache-gb 1.0 --text-file wikitext2_test.txt`, in-place
pool keys, graphs built on kiln-dk-cf3): wikitext-2 slice -0.5510, by chunk -0.772 -0.702 -1.091 -0.932 -1.207
-0.170 -0.360 -0.138 -0.288 -0.265 -0.344 -0.341, the same before and after the merge; sp-merge -0.5481 on the
same box (+0.0028, |dlogprob| mean 0.046, greedy agreement 97.7%); the merged DSA tree -0.5517 (0.045, 98.1%).
CPU suite, one process, transformers 5.15: 637 passed, 28 skipped; the GLM-family and DSA files with
transformers 5.18: 233 passed, 8 skipped (kiln-dk-cf3, r7i.48xlarge).

## The sweep's 4096-token prefill step at conc 64, block by block, and each rank routing its own rows (2026-10-04, trn1.32xlarge)

**Where conc 64 spends the device.** The engine-v0 conc-64 run (G64 prefill 4096 / bucket 1024, decode 16 rows per
group, KV 1.5 GB fp8, 128 requests; kiln-mimo-trn1 log 20261004T122641Z) took 371.1 s of wall. With the step times of a
profiled run of the same configuration (`KILN_PROFILE_PIECES=1 KILN_PROFILE_EXEC=1`, kiln-pf-32b log
20261004T122442Z-g64p-s20: prefill step 1.089 s, decode step 0.180 s), 128 prompts of 8192 tokens are 256 prefill
steps (278.8 s) and 128 x 256 output tokens at up to 64 rows per step are >= 512 decode steps (92.0 s): 370.8 s, so
both run packed and the prefill steps are ~75% of the device time. Per output token that is 8.5 ms of prefill (32
prompt tokens at 3760 tok/s) and 2.8 ms of decode. The profiled run's own counts (154 prefill, 787 decode steps over
255.6 s) were a 64-request run at KV 1.0: every request arrived at once, KV-bound, the warm-up request's ~255 decode
steps included, ~48% decode occupancy. Matching p5en's conc-64 spot price ($5.5 / M) needs 108.6 tok/s, i.e. a
4096-token prefill step of ~0.82 s with decode unchanged.

**Per block at the sweep's shape** (`KILN_CC_ARGS=--model-type=transformer KILN_MOE_KERNEL=nki
KILN_MOE_PREFILL_KERNEL=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_DSA_SELECT=nki KILN_PROFILE_EXPERT_LAYOUT=loaded
KILN_MOE_PREFILL_SKIP=20 python tools/profile_layer.py --model zai-org/GLM-5.3-Flash --tp 32 --dp-attention 4 --ranks
32 --prefill 1024 --pages 264 --sum-readback --neff --layers 12 --what hcblocks --part-layers 0 3 4 --sp`, kiln-pf-32b,
log 20261004T124817Z-prof1024.log; 4096 rows, 1024 per token-mixer group, sequence-parallel streams at 128 rows per
rank; p50 ms of one graph each, a graph holding a collective also pays the ~3-5 ms per execution of "Collectives
across chips"):

| graph | layer 0 (dense, KDA) | layer 3 (MoE, pooled DSA) | layer 4 (MoE, KDA) |
|---|---|---|---|
| SP attention block (mHC, gather, mixer, all-reduce, own rows, mix) | 10.1 | 16.15 | 9.69 |
| SP FFN block | 6.48 | 16.70 | 16.67 |
| token mixer alone, with its all-reduce | 5.86 | 12.26 | 6.05 |
| MLP alone (experts + shared + all-reduce) | 6.91 | 11.69 | 11.74 |
| MoE routed experts alone (router + prefill kernel) | | 9.97 | 9.94 |
| router alone / shared expert alone | | 1.14 / 0.55 | 1.13 / 0.64 |
| SP gather [128 -> 4096] (zero-padded world all-reduce) / world all-reduce [4096, 4096] | 6.3 / 6.39 | 6.10 / 6.23 | 5.17 / 4.89 |
| SP mHC collapse / output mix (128 rows) | 0.50 / 0.38 | 0.79 / 0.56 | 0.69 / 0.30 |

DSA layer 3 at 1024 queries over 8448 keys: projections and indexer 1.45 ms, the pooled selection 3.80, the
attention core with a given mask 8.77 (expand) / 8.76 (absorb), mla.attention 10.76, o_proj 0.80. The served step's
four 12-layer pieces (KILN_PROFILE_PIECES above) are 260 / 294 / 300 / 224 ms. Scaled onto them, the routed-expert
kernel is the largest item, ~40% of the step (~9.9 ms per MoE layer at the profiler's random routing, and real
routing needs ~25% more lane tiles: "The prefill MoE kernel on the real expert layout" above), then the KDA mixers
(~20%), the DSA attention (~12%), the collectives (~8%) and the router (4.4%).

**Each rank routes its own rows (`KILN_SP_ROUTE`, models/hybrid.py _sp_route, default on).** With
sequence-parallel streams every rank gathered all R rows and then routed all of them: the router (an fp32 [R, H] x
[H, 288] matmul, sigmoid, bias, group top-k) ran R rows on every rank. Now each rank routes its own R / tp rows and the
[R, 2k] routing (weights and indices, fp32) is gathered beside the rows with the same zero-padded all-reduce
(exact: one rank's value plus zeros per element; indices below 2^24). The per-row arithmetic is unchanged, only the
router matmul's row count differs. On the device (same command, `--part-layers 4`, logs 20261004T133332Z /
20261004T133827Z-profroute.log): the SP FFN block 16.684 -> 16.100 ms p50 (min 16.343 -> 15.437); the expert
indices of own-rows routing against all-rows routing on the same input equal 1.000000 and the weights differ by
0.000e+00. CPU (`tests/test_dp_attention.py::test_glm5_next_sequence_parallel_streams`, now also with
`KILN_SP_ROUTE=0`): greedy tokens equal to replicated streams at tp=4 / dp_attention 2 and tp=2 either way.

End to end, kiln-pf-32b, back to back (q/v0-trn1's G64-4096-KV1.5-S20-P12-K command: `KILN_CC_ARGS=--model-type=transformer
KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20
KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 python bench/serve_sweep.py --model
zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256
--page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64
--concurrency 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8`, 128 requests):

| tree | conc 64 out tok/s | TTFT p50 / p90 | ITL p50 | spot $ / M out | log |
|---|---|---|---|---|---|
| engine-v0 6b96c5f, farm graphs (q/v0-trn1) | 89.9 | 10.8 / 123.7 s | 640 ms | 6.64 | 20261004T134805Z-g64-base |
| + KILN_SP_ROUTE (ba56c78), prefill graphs compiled on the box | **91.3** (+1.6%) | 10.6 / 121.4 s | 630 ms | 6.54 | 20261004T135953Z-g64-sr |

Real-weight ppl with it (kiln-pf-32b, the sweep's env plus `KILN_MOE_PREFILL_MIN_TOKENS=1`, `tools/check_ppl.py --model
zai-org/GLM-5.3-Flash --tp 32 --piecewise --kv-cache-gb 1.0 [--text-file wikitext2_test.txt]`, logs
20261004T144902Z-ppl4-sr / 20261004T150008Z-pplwiki-sr): the 4 sentences -2.127 / -3.543 / -0.980 / -1.686, mean
-2.072, the values engine-v0 gives; wikitext -0.551, by chunk -0.769 -0.700 -1.103 -0.923 -1.209 -0.171 -0.360
-0.143 -0.291 -0.268 -0.331 -0.338.

## The prefill MoE kernel's pipeline at C=4096: what paces it, a round order and an output ring (2026-10-04, SDK 2.32, nki 0.6.0, trn1.2xlarge)

One call of kernels/moe_prefill.py on layer 3's real experts of rank 0, uniform routing, `KILN_MOE_PREFILL_SKIP=20`
(`python tools/probe_moe_prefill.py --experts-file glm53f_l3_r0_experts.pt --chunks 4096 1024 2048 --decode-max 0
--iters 10 --compare order=0,nyb=0`, kiln-pf-k1, log 20261004T155426Z-kver.log; p50 with the sum readback; the alternatives below from the same probe's
`--compare`, logs 20261004T134538Z-ord, 135821Z-ord2, 140334Z-pcw, 144102Z-dbg, 145420Z-nyb, 151417Z-ysd, 153157Z-rng,
161243Z-pfd; the engine profile 20261004T142238Z-prof2):

| C (B) | before (order 0, one stage C buffer per step) | now (KILN_MOE_PREFILL_ORDER=1, KILN_MOE_PREFILL_NYB=3) | output |
|---|---|---|---|
| 1024 (64) | 5.210 ms | **4.620** (-11.3%) | bit-identical |
| 2048 (64) | 6.456 | **5.874** (-9.0%) | bit-identical |
| 4096 (128) | 9.725 | **9.352** (-3.8%) | bit-identical |

**What paces it.** An engine profile of the C=4096 call (`tools/prof_engines.py`; the instruction records' gaps
attributed to the engine named in each instruction's wait semaphore, over 40% of the lane-tile loop) has no engine
saturated: the vector engine busy ~62% of the time (31 scalar_tensor_tensor per-row scalings per lane tile, ~240-330
ns each, are most of it), idle ~38% waiting on the scalar engine (~20%), the tensor engine (~15%) and DMA arrivals;
the tensor engine idle 50% (waiting on the vector engine), the scalar engine 66%, GpSimd 94%; DMA at a flat ~175
GB/s (43% of the core's 410). A lane tile takes ~20 us against ~12 us of vector work: the loop is bound by the
hand-offs between engines, not by any engine's throughput. Removing pieces to time them (debug flags, wrong
output): without the routing combine (the Y gather and its matmuls) 8.42 ms against 9.55, so the combine is ~1.1 ms;
without the Y store the compiler also drops stage C (its outputs then have no reader), 6.71 ms; with a ring of 3
stage C buffers and no Y store 9.075 against 9.340, so the 336 MB Y store itself is ~0.27 ms.

**Measured and kept** (each bit-identical to the kernel before it):
- Stage B first in each round (`ord_` 1, kernels/moe_prefill.py _stage_b): the tensor engine's in-order stream
  issues the round's gate_up matmuls before stage C's down matmuls, which wait on stage F's activations from the
  vector engine; the vector engine's scalings of those partials no longer wait behind them. 9.725 -> 9.551 ms.
- Stage C's down outputs in a ring of buffers (`nyb`): with order 1, one allocation per step 9.563, a ring of 2
  10.559, of 3 **9.340**, of 4 9.350. A ring of 2 is slower than none (the compiler's own reuse); 3 is the default.
- The per-row path no longer allocates the dequantize-first path's [128, 32, 128] bf16 tiles (16 KB per partition
  it never read).

**Measured and dropped** (all bit-identical, none faster): the per-row scaling issued one round after its matmuls
9.730; the down column factors applied to the down weights in bf16 before the matmul instead of to the drained
product (exact: a power of two before an fp32 sum) 9.690 alone, 9.634 with order 1; rings of 3 or 4 for the x^T, the
activations and the gate_up sums 9.324 / 9.311 (against 9.340); loads 3 or 4 tiles ahead instead of 2 9.350;
deferring each Y store into the next step's rounds 10.56 (it needs a ring of 2). A vector-engine microbenchmark
could not move work to GpSimd: `nisa.tensor_tensor(..., engine=nisa.gpsimd_engine)` fails in the backend
("[NCC_IXCG965] Instruction engine check failed (Pool)") with fp8, bf16 and fp32 operands alike.

**The routing combine stays.** Writing each pair's output straight into the token rows with the DMA engines'
scatter-add (`nisa.dma_compute`, read-modify-write) would need an fp32 output (twice the HBM bytes of the store and
the gather it replaces) or bf16 accumulation (a rounding per expert), and two tiles adding to the same token rows
on different DMA engines are not ordered. The combine is ~1.1 ms of the call and its Y gather (268 MB) runs at ~240
GB/s.

## Sequence-parallel block outputs as reduce-scatters, and the stack end to end (2026-10-04, SDK 2.32, trn1.32xlarge)

**`KILN_SP_RS` (default on; models/hybrid.py _sp_out, DecoderForCausalLM._reduce_scatter / _out_reduce).** With
sequence-parallel streams each block ran over all R gathered rows, all-reduced its output over the world (the MLP's
reduction; under DP attention the token mixers' zero-padded one) and kept this rank's R / tp rows. Now the
output reduction is one world reduce-scatter that leaves each rank exactly those rows: the same sums, half the
bytes. The rows a reduce-scatter hands rank r are block r of tp_size row blocks, which under DP attention is the
group-major batch's rows of rank r: the rows _sp_rows took. A cached reduce-scatter NEFF reloads in a later process
(`python tools/probe_rs_reload.py --ranks 32 --rows 128`, run twice on kiln-pf-32b after a reboot, logs
20261004T154740Z / 154847Z-rsreload1/2: both runs correct; [4096, 4096] -> [128, 4096] bf16 6.06 / 6.36 ms against
6.69 / 7.14 for the all-reduce and own rows), unlike an all-gather ("A cached all-gather NEFF breaks in the next
process"; the gathers stay zero-padded all-reduces). A crashed run (a `dist.barrier()` the neuron backend does not
implement) left the runtime's collectives wedged ("Failed to schedule neff execution" for every later graph) until
the instance was rebooted. The collective's own summation order may differ from the all-reduce's in the last bf16
bit, so the outputs are not bit-identical. Blocks of layer 3 (`tools/profile_layer.py ... --prefill 1024
--part-layers 3 4 --sp`, the new kernel defaults, log 20261004T161121Z-profrs.log): the SP FFN block with own-rows
routing 14.813 -> 14.128 ms, the SP attention block 16.097 -> 15.789 ms.

**End to end on the merged tree** (kiln-pf-32b, back to back 16:56-17:35 UTC, q/v1-trn1's G64-4096-KV1.5-S20-P12-K
command: `KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki
KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20
KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 python bench/serve_sweep.py ...
--prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64 --concurrency 64 --decode-buckets 16 --kv-cache-gb 1.5
--kv-cache-dtype fp8`, 128 requests; every graph from the compile farm, q/pfab-trn1):

| tree | conc 64 out tok/s | TTFT p50 / p90 | ITL p50 | spot $ / M out | log |
|---|---|---|---|---|---|
| engine-v0 1d63405 | 88.4 | 10.8 / 123.0 s | 648 ms | 6.76 | 20261004T170941Z-v1-b1 |
| + the kernel's round order and output ring (9f96cea) | 87.8 | 10.8 / 123.8 s | 652 ms | 6.80 | 20261004T172416Z-v1-k1 |
| + KILN_SP_RS (a22bfe9) | **91.6** (+3.6%) | 10.3 / 117.5 s | 624 ms | **6.52** | 20261004T165624Z-v1-rs1 |

Real-weight ppl of a22bfe9 (kiln-pf-32b, the sweep's env plus `KILN_MOE_PREFILL_MIN_TOKENS=1`, farm graphs, logs
20261004T173608Z-ppl4-rs1 / 20261004T174035Z-pplwiki-rs1): the 4 sentences -2.146 / -3.523 / -0.976 / -1.687, mean
-2.073; wikitext -0.547, by chunk -0.772 -0.690 -1.097 -0.924 -1.212 -0.176 -0.337 -0.134 -0.292 -0.268 -0.323
-0.333.

**Where the +3.6% comes from: all of it is `KILN_SP_RS`.** The same three trees with `KILN_PROFILE_PIECES=1
KILN_PROFILE_EXEC=1` (kiln-pf-32b, logs 20261004T174506Z-v1p-b1, 20261004T181024Z-v1p-k1, 20261004T175754Z-v1p-rs1;
264 prefill steps each; synchronous timing, so the profiled throughput is lower):

| tree | prefill pieces (ms) | sum | prefill step exec p50 | decode step exec p50 |
|---|---|---|---|---|
| engine-v0 1d63405 | 254.5 / 287.3 / 293.9 / 219.0 | 1054.7 | 1064.8 ms | 200.4 ms |
| + the kernel's round order and output ring | 256.3 / 289.3 / 295.6 / 220.8 | 1061.9 (+0.7%) | 1071.5 | 199.9 |
| + KILN_SP_RS | 239.0 / 274.0 / 280.4 / 208.7 | **1002.1** (-5.0%) | 1011.7 | 199.9 |

The kernel's -3.8% per C=4096 call (the section above) does not transfer into the prefill pieces: +7 ms per
4096-token step against the -16 ms its standalone numbers predict (42 MoE layers x 0.37 ms), i.e. neutral within the
runs' noise. The outputs are bit-identical either way and the defaults stay (moving them would move every trn1
prefill graph key). What that says about the in-graph MoE: the standalone harness times the kernel alone in its
graph, where the lane-tile loop is paced by hand-offs between its own engines; inside a 12-layer piece the same
call does not get faster when those hand-offs do, so whatever bounds it there (contention with the surrounding
graph's DMA queues and SBUF, or the real routing's ~25% more lane tiles changing which stage paces it) is not in the
standalone measurement, and kernel work should be judged in a piece, not alone. "Where a prefill piece goes" below
measures that directly.

**An all-to-all instead of the zero-padded all-reduce gathers does not pay** (`tools/probe_rs_reload.py`, 32 ranks,
two processes, logs 20261004T183218Z / 183316Z-a2areload1/2): every rank's [128, 4096] bf16 rows to every rank by
`all_to_all_single` of this rank's rows repeated 32 times is exact and its cached NEFF reloads in a later process,
but takes 34.7 / 39.2 ms against 5.2-6.8 ms for the zero-padded world all-reduce gather (`profile_layer` hcblocks'
SP gather piece) and 6.1 ms for the reduce-scatter in the same runs. The two gathers per layer stay all-reduces.

## Where a prefill piece goes: a device profile of one 12-layer sequence-parallel piece at 32 ranks (2026-10-04, SDK 2.32, trn1.32xlarge)

**Method** (reusable for any graph that holds world collectives). `KILN_PROFILE_INSPECT=<dir>` in tools/profile_layer.py
sets `NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_DEVICE_PROFILE=1 NEURON_RT_INSPECT_OUTPUT_DIR=<dir>` in rank 0's
process only, before its runtime starts. In the real 32-rank run that wrote only a system trace (ntrace.pb) and a
copy of every NEFF rank 0 executed (`<dir>/<instance>_pid_<pid>/<id>/neff_*_vnc_0.neff`), no device profile. The
NEFF is then replayed with its collectives on all 32 cores and worker 0 profiled: `neuron-explorer capture -n
group.neff -s profile.ntff -r 32 -i 0 --ignore-exec-errors` (27 s; it writes `profile_rank_0.ntff`, 541 MB for this
graph) and `neuron-explorer view -n group.neff -s profile_rank_0.ntff --output-format json --output-file full.json
--ignore-nc-buf-usage` (7.2 GB). Caveat: without input files every input is zero, so data-dependent work is not
the real one (here the MoE router sends every token to the same 8 experts: 256 full lane tiles against ~311 for
uniform and ~388 for real routing); inputs can be given as `input<i> <file.npy>` pairs in the NEFF's placeholder
order (tools/prof_engines.py). Attribution: each engine's instruction occupancy in 0.1 ms bins (the instruction
records carry no op names for kernel code, and `hlo_name` is empty for 99% of them), classified as the MoE kernel's
lane-tile loop (tensor > 90%, vector > 50%, GpSimd active: its steady pattern), other compute, or idle, and each
idle bin by whether collective DMA (`CCdma` queues) or other DMA moved in it. Traps: LNL waits 600 s for another
rank's compile of the same graph and then fails (`KILN_PROFILE_COMPILE_TIMEOUT` raises the torch.compile option
`compilation_timeout`); a crashed collective run left every later graph failing with "Failed to schedule neff
execution" until a reboot.

**The piece** (layers 12-23: 9 KDA + 3 DSA layers, all MoE; `KILN_CC_ARGS=--model-type=transformer KILN_MOE_KERNEL=nki
KILN_MOE_PREFILL_KERNEL=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_DSA_SELECT=nki KILN_PROFILE_EXPERT_LAYOUT=loaded
KILN_MOE_PREFILL_SKIP=20 KILN_DSA_POOL_CACHE=separate python tools/profile_layer.py --model zai-org/GLM-5.3-Flash --tp 32
--dp-attention 4 --ranks 32 --prefill 1024 --pages 264 --sum-readback --layers 24 --what none --layer-groups 12-23
--sp-only`, engine-v0 b4f400f, kiln-pf-32b, log 20261004T185551Z-profinsp.log: p50 231.7 ms; the replay 233.5 ms):

| phase (0.1 ms bins) | ms | share |
|---|---|---|
| the MoE kernel's lane-tile loop | 82.6 | 35.4% |
| other compute (KDA / DSA mixers, the MoE combine and plan, mHC, norms, projections, router, shared expert) | 103.5 | 44.3% |
| every engine idle, collective DMA moving | 15.3 | 6.6% |
| every engine idle, nothing moving | 24.8 | 10.6% |
| every engine idle, other DMA moving | 2.3 | 1.0% |
| compute with collective DMA | 5.0 | 2.1% |

Engines active over the execution: tensor 30%, vector 38%, scalar 24%, GpSimd 3%. DMA 28.5 GB per execution: 10.5 GB
on the kernels' dynamic queues (x row gathers, expert blobs, the MoE combine's Y), 16.8 GB on the compiler's static
spill / reload queues (the zero-padded [4096, 4096] gather and DP buffers, 31/32 and 3/4 zeros, and every
HBM-resident intermediate), 1.2 GB of collectives at ~35-56 GB/s. In the 1 ms timeline each layer shows 2-4 ms in
which every engine is near 0% around its five collectives (the attention block's gather and reduce-scatter, the
FFN block's gather, its routing gather, its reduce-scatter): so of the ~8 ms per MoE layer that the blocks' own
costs do not explain, ~3.5 ms is engines waiting on collectives (about half of it with collective data moving,
half with nothing moving: synchronisation and latency), and the rest is the HBM traffic of the XLA ops around the
kernels. It is not contention between the kernels' DMA queues.

## Dynamo names graph nodes after Python locals: a refactor can change every cache key (2026-10-04, torch 2.11.0)

The compile cache key hashes the FX graph text (LNL compile/cache.py create_cache_hash), and that text
names each node. Dynamo renames a node after the FIRST Python local its value is stored in, in whatever
inlined frame that happens: `STORE_FAST` calls `loaded_vt.set_name_hint(name)` (torch/_dynamo/
symbolic_convert.py, line 1924 in the SDK 2.32 venv's torch 2.11.0) and `TensorVariable.set_name_hint`
renames the proxy node once (torch/_dynamo/variables/tensor.py:1859). A value passed straight into a
call keeps its default name (`linear_7`). So a refactor that only moves code into a helper, or stores an
intermediate in a new local, changes the key of every graph tracing that code while the HLO stays the
same, and every cached NEFF of those graphs misses.

Measured on feat/mixed-batch dd0affe (compile farm, q/mx-trn1, capture of the G64 / F0 sweep configs at
trn1 shapes, 2026-10-04): `linear_attn.mix` had become `out = _mix_rows(...); return
model._attn_all_reduce(out)`, which renamed the KDA output projection `%out` (`%out = linear(%reshape_7,
%l_ts_11_)`) and pushed the next `out` (the MoE kernel's output) to `%out_1`. All three 12-layer decode
groups of the conc-32 config got new keys (680d78c7 / ae4489f1 / c9b817da against v0-trn1's b087812f /
82d47896 / d630b556) with identical HLO (tools/hlo_diff.py SAME for ae4489f1 vs 82d47896, 23,832
instructions, and c9b817da vs d630b556, 31,649). Restoring the expression (the projection passed
straight to the all-reduce, bf3448c) is the fix.

Rule for any edit of code the graphs trace (models/*, the piecewise wrappers): keep the existing paths'
statements, locals and their order exactly, add new behaviour in branches, and before a device run
capture the edited tree and `tools/compile_farm.py check` its keys against the previous queue's cache.

## Mixed batches: decode rows inside the prefill graphs (2026-10-04, SDK 2.32, trn1.32xlarge, feat/mixed-batch)

**What it is.** `KILN_MIXED_BATCH=1` (EngineConfig.mixed_batch, off by default): every prefill call becomes one
graph over each DP-attention group's chunk rows followed by up to `KILN_MIXED_DECODE_ROWS` (default: the
largest decode bucket) of the group's decoding sequences (DecoderForCausalLM.forward_mixed, ModelRunner.mixed,
LLMEngine._launch_mixed), the way vLLM's chunked prefill runs decode tokens and prefill chunks in one forward
(vllm 0.24.0 vllm/v1/core/sched/scheduler.py Scheduler.schedule: one token_budget, running requests first).
The scheduler is unchanged, so a mixed engine runs the steps an unmixed one runs; a step with prefill work makes
no decode call for the rows that ride along, decode-only steps keep the decode graphs, and the prefill-only
graphs are not warmed. The residual stream, the mHC streams (sequence-parallel: the decode rows are rows of the
partition like the chunk's), the MLP and the experts run over all rows; every token mixer runs its chunk form
on the chunk's rows and its decode form on the decode rows inside the same graph, and the post graph samples
every chunk row and every decode row (prompt logprobs scored from the same call). Per layer kind:

- KDA (linear_attn._mix_joint): one in_qkv / gate / beta / output projection over all C + D rows; the short conv
  from the chunk's state row for the chunk and from each decode row's own row; the chunked delta rule (the NKI
  kernel on the device) on the chunk, one recurrent step per decode row; conv and recurrent state pools each
  written once for the chunk row and the decode rows.
- MLA / pooled DSA (mla.attention_joint): q_a / kv_a / q_b and the indexer's query, key and weights once over
  all rows; the latent and indexer rows (and the pool keys, separate or in place) written once; then the chunk
  reads its context through its block table and attends in the prefill form (decompressed keys, the fused score
  and selection kernel), the decode rows through theirs in the decode form (absorbed, the selection kernel);
  o_proj once. The selection is attended with directly (no scratch: it would be written twice in one graph).
- mHC: row-wise, over all rows; with sequence-parallel streams each rank holds N (C + D) / tp rows.
- MTP and speculative decoding: refused at start (their decodes are verify calls / draft from every call).

Knobs kept for the record of what was measured: `KILN_MIXED_MIXERS=calls` (the chunk form and the decode form
as two calls on the rows: every projection twice, every cache written twice) and `KILN_MIXED_SP=split` (each
rank's share of the chunk rows plus every decode row replicated). Both measured worse, below.

**CPU equivalence** (tests/test_mixed_batch.py, fp32, m7i.8xlarge, transformers 5.18 for the GLM cases, 14
tests): greedy tokens identical to the same engine with mixed batches off and chosen-token logprobs within
6.2e-6, for a truncated random GLM-5.3-Flash (8 layers: KDA, pooled DSA with index_topk 16 so prefill and decode
both select pools, mHC, MoE; 6 prompts of 5-60 tokens over 3-4 running slots in chunks of 16), graph and
piecewise execution, overlap scheduling, tp=2, tp=4 with DP attention 2 and sequence-parallel streams (with
SP-local routing and the reduce-scatter of engine-v0 c33a391), both mixer forms, both layouts, the three
pool-key forms (in place, separate, per call), decode rows in slices (`KILN_MIXED_DECODE_SPLIT`), more decodes
than mixed rows (the rest in a decode call), prompt logprobs; and Qwen3 (GQA) and a KDA-only model.

**Device, same box, first base** (kiln-g1-trn1, GLM-5.3-Flash real weights, tp=32, DP 4, feat/mixed-batch
bf1e346 = engine-v0 a1351b7 + this work, env `KILN_CC_ARGS=--model-type=transformer KILN_DSA_SELECT=nki
KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=0
KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1` plus `KILN_MIXED_BATCH=1
KILN_MIXED_SP=... KILN_MIXED_MIXERS=...`, the G64-4096-KV1.5 (conc 64, 128 requests, fp8 KV) and F0-4096-KV1.5
(conc 32, 96 requests, bf16 KV) serve_sweep argv of q/v0-trn1, graphs from the compile farm q/mx-trn1 with
`NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`; logs s3://<your-bucket>/logs/kiln-g1-trn1/<stamp>-mx-<name>.log):

| run | conc | out tok/s | TTFT p50 / ITL p50 | log |
|---|---|---|---|---|
| unmixed | 64 | 88.3 | 11.0 s / 652 ms | 20261004T144651Z-mx-sweep64-plain |
| unmixed, `--state-checkpoints 4` | 64 | 88.3 | 11.0 s / 651 ms | 20261004T154746Z-mx-sweep64-plain-ck4 |
| mixed calls, default state pool | 64 | does not load (Allocation Failure, 2nd mixed group) | | 20261004T144109Z-mx-sweep64-mx |
| mixed calls, `--state-checkpoints 4` | 64 | 84.8 (-4.0%) | 11.5 s / 676 ms | 20261004T153321Z-mx-sweep64-mx-ck4 |
| mixed calls, split layout, CK4 | 64 | does not load (Allocation Failure) | | 20261004T164554Z-mx-sweep64-mx-ck4-split |
| **mixed joint, CK4** | 64 | **90.4 (+2.4%)** | 10.7 s / 633 ms | 20261004T172937Z-mx-sweep64-mx-ck4-rowsjoint |
| mixed joint, CK4, repeat | 64 | 90.4 | 10.7 s / 633 ms | 20261004T181007Z-mx-sweep64-mx-rj-repeat |
| unmixed | 32 | 84.6 | 10.6 s / 332 ms | 20261004T170233Z-mx-sweep32-plain |
| mixed calls | 32 | 79.2 (-6.4%) | 11.5 s / 352 ms | 20261004T150321Z-mx-sweep32-mx |
| **mixed joint** | 32 | **86.0 (+1.7%)** | 10.3 s / 326 ms | 20261004T182212Z-mx-sweep32-mx-rj |

`--state-checkpoints 4` (bench/serve_sweep.py, new): the state pool's 32 default checkpoint rows per group are
0.59 GB of each core at G64 by the state shapes (18.4 MB a row: 34 KDA layers x 8 heads x [128, 128] fp32 plus
the conv rows); this sweep never uses one (no shared prefix, and a request ends before its first decode
checkpoint). neuron-monitor on the loaded unmixed G64 config: tensors 14.64 GB, model code 0.97 GB, shared
scratchpad 0.87 GB, constants 0.09 GB per core of 17.18 GB, before the DMA rings; the mixed groups need more.

**Where a mixed call's time goes** (`KILN_PROFILE_EXEC=1`: every call synchronous, 32 requests; the two
diagnostic shapes were compiled for this):

| graph call, rows per DP group | p50 | log |
|---|---|---|
| F0: prefill 1024 | 1090.3 ms | 20261004T152602Z-mx-prof32-plain |
| F0: prefill **1032** (unmixed, 8 more chunk rows) | **1165.0 ms** | 20261004T160800Z-mx-prof1032-plain |
| F0: mixed calls 1024 + 8 | 1285.3 ms | 20261004T151830Z-mx-prof32-mx |
| F0: mixed calls 1016 + 8 (1024 in all) | 1266.2 ms | 20261004T160000Z-mx-prof1016-mx |
| F0: decode, 8 rows | 110.0 ms | 20261004T151830Z-mx-prof32-mx |
| G64: prefill 1024 | 1088.1 ms | 20261004T180209Z-mx-prof64-plain-ck4 |
| G64: **mixed joint 1024 + 16** | **1198.5 ms** | 20261004T175404Z-mx-prof64-mx-rowsjoint |
| G64: decode, 16 rows | 160.2 ms | 20261004T180209Z-mx-prof64-plain-ck4 |

- **Row counts off a multiple of 128 cost far more than their rows**: an unmixed prefill at 1032 rows per group
  (129 per rank under sequence-parallel streams) is +6.9% time for +0.8% rows. Shapes of the big graphs should
  keep rows per group, and per rank, multiples of 128.
- **The joint mixers are what make mixing pay**: the 16 decode rows of a G64 step cost 110 ms inside the prefill
  graph against 160 ms as their own decode call; with the "calls" mixers they cost more than the call they
  replaced (F0: ~176 ms for 8 rows even with the call's rows aligned, against 110 ms; per 12-layer group
  (`KILN_PROFILE_PIECES=1`, logs 161603Z-mx-pieces32-mx / 162303Z-mx-pieces32-plain) +39-54 ms over the prefill
  group while a whole decode group is 24-35 ms).
- **Most of what is left is probably the row misalignment** (1040 rows per group, 130 per rank): the 1032-row
  unmixed prefill alone costs +75 ms. Taking the decode rows from the chunk's 1024 instead (1008 + 16) does not
  recover it here: 8192-token prompts would need 9 chunks, and a step that ends one request's prompt and starts
  the next one's then holds two chunks of one group, i.e. a second mixed call (about +32 calls of ~1.1 s against
  at most 256 x 75 ms won back at conc 64). The way to recover it is packed variable-length chunks (two
  sequences' chunks in one group's 1024 rows: the KDA and DSA chunk forms would have to take a sequence boundary
  inside the chunk), not attempted.
- **Spill** (queue-instance spill runs per 12-layer group, tools/hbm_estimate.py on the farm's NEFFs): unmixed G64
  3.24M / 3.32M / 2.13M; mixed calls (16 decode rows) 4.84M / 4.94M / 3.01M (scratchpad 0.82 against 0.69 GiB);
  mixed joint 4.92M / 4.93M / 3.06M; split layout 7.05M / 8.63M / 4.76M with calls and 7.10M / 8.65M / 4.82M
  with joint (code 145 MB for the largest group), which is why the split configurations do not load. F0 (8 decode
  rows, bf16 KV) with calls: 3.09M / 3.16M / 2.13M, as unmixed. The split layout with bf16 KV failed to compile
  (NCC_IXCG967, a 37,152 / 38,313 step into a 16-bit isa_static_pattern.step_elem field), the rows layout did not.

**Numerics on the device** (same box). check_ppl runs prompt logprobs with max_new_tokens 1, so through mixed
graphs it scores only the CHUNK part (with `KILN_MIXED_DECODE_ROWS=32` padding decode rows), not decode rows:

| check | unmixed | mixed joint | mixed calls | logs |
|---|---|---|---|---|
| 4 sentences (France / Water boils / def add / fox) | -2.073 (-2.140 / -3.580 / -0.932 / -1.690) | -2.077 (-2.187 / -3.555 / -0.912 / -1.687) | -1.790 (-2.103 / -2.584 / -0.941 / -1.459) | 165225Z-mx-ppl4-plain, 183246Z-mx-ppl4-mx-rj, 143717Z-mx-ppl4-mx |
| wikitext-2 slice (3071 tokens, md5 3ce70e93) | -0.552 | -0.550 | -0.554 | 165805Z-mx-wt-plain, 183639Z-mx-wt-mx-rj, 145901Z-mx-wt-mx |

Greedy text (`tools/check_mixed.py`, new: the sweep's engine from its argv with `--state-checkpoints 4`, 64
natural-text prompts of 700-8192 tokens, windows of check_ppl's LONG_TEXT repeated, 64 new tokens each, 64 in
flight): joint 57 of 64 outputs identical to unmixed, calls 56 of 64 (logs 174236Z-mx-text64-mx-ck4-rowsjoint,
162955Z-mx-text64-mx-ck4, 163802Z-mx-text64-plain-ck4; outputs s3://<your-bucket>/logs/kiln-g1-trn1/mx/).
Every output that differs is one of the 700-token prompts, the only windows that do not contain a repeat of the
~1000-token passage (longer windows continue a passage they have already seen, a confident copy); all are
coherent English.

**On the final engine-v0 head** (feat/mixed-batch d6f3a1b = engine-v0 b4f400f, i.e. c33a391 with SP-local routing,
SP reduce-scatter, the MoE kernel's round order, the separate pool-key cache under FP8 and reserve admission,
plus this work; env and argv exactly q/final-c33a391's G64-4096-KV1.5-S20-P12-K; the compile farm checked that
this tree's unmixed capture is set-equal to q/final-c33a391's keys; kiln-g1-trn1, 128 requests):

| run | out tok/s | TTFT p50 / p90 | ITL p50 | log |
|---|---|---|---|---|
| unmixed (the q/final-c33a391 config itself) | 94.9 | 10.1 s / 115.8 s | 605 ms | 20261004T191106Z-mx-final-plain |
| unmixed, `--state-checkpoints 4` | 94.8 | 10.1 s / 115.8 s | 605 ms | 20261004T184413Z-mx-final-plain-ck4 |
| **mixed (rows, joint), `--state-checkpoints 4`** | **96.6 (+1.8%)** | 9.9 s / 113.8 s | 591 ms | 20261004T185838Z-mx-final-mx-rj-ck4 |

The reduce-scatter compiles and runs on the mixed graphs' 130 rows per rank. Spot $6.18 against $6.29 per 1M
output tokens (p5en spot at conc 64: $5.5).

**Is the divergence noise?** `tools/check_mixed.py --compare` on the final head (64 prompts, 64 new tokens, top-2
logprobs; logs 20261004T192241Z-mx-text-final-mx-rj-ck4, 193014Z-mx-text-final-plain-ck4): every position before a
pair's first difference has identical inputs in both runs, so its chosen-token logprob is teacher-forced; the
tool splits it by where the mixed run computed the token. As a noise floor, the same comparison between two
UNMIXED engines that differ only by changes engine-v0 accepted as numerically neutral (bf1e346's engine against
the final head's: SP-local routing, SP reduce-scatter, the MoE kernel's round order, the pool-key cache form;
20261004T194129Z-mx-text-old-plain-ck4 against the final unmixed run):

| | mixed vs unmixed (final head) | unmixed old base vs unmixed final head |
|---|---|---|
| outputs identical | 56 / 64 (the 8 others: the 8 prompts of 700 tokens) | 56 / 64 (the same 8 prompts) |
| first difference at output | 0, 2, 6, 11, 16, 19, 29, 31 | 1, 6, 21, 27, 29, 29, 31, 33 |
| reference top-1 minus top-2 at the first difference | 0.875, 0.375, 0.000, 0.125, 0.125, 0.250, 0.125, 0.125 | 0.125, 0.125, 0.250, 0.250, 0.750, 0.125, 0.125, 0.3125 |
| teacher-forced \|dlogprob\|, decode rows inside mixed calls | n=2608, max 0.378, p99 0.053, mean 0.0020 | |
| ... decode calls | n=1034, max 0.016, p99 0.0006, mean 0.00006 | n=3705, max 0.414, p99 0.068, mean 0.0023 |
| ... first tokens (prefill chunks) | n=64, max 0.575, p99 0.091, mean 0.0123 | n=64, max 0.253, p99 0.124, mean 0.0085 |

The sampler reports logprobs in steps of 0.125 at these magnitudes (bf16 logits). The decode rows inside the
mixed graph differ from the unmixed decode calls about as much as the two accepted unmixed engines differ from
each other (p99 0.053 against 0.068, mean 0.0020 against 0.0023), the divergences fall on the same 8 low-confidence
prompts with the same spread of margins, and one first token from a mixed prefill chunk (margin 0.875 in the
reference, a tie in the mixed run) is the largest single difference; the CPU equivalence is exact. So: the
graph-level noise of the same arithmetic in a different graph (the decode rows' MoE through the prefill kernel,
130 rows per rank, a reduce-scatter over 4160 rows), not a mixing defect.

## Expert parallelism for GLM-5.3-Flash's routed experts (2026-10-04, SDK 2.32, nki 0.6.0, trn1)

`KILN_MOE_EP=1` (models/decoder.py moe_ep_enabled, kernels/moe_ep.py, feat/moe-ep): each of the 32 ranks holds 9
WHOLE experts (288 / 32; rank r experts 9r .. 9r + 8, every one of the 2048 intermediate rows) instead of 64
intermediate rows of all 288, and its routed output is the sum over the (token, expert) pairs routed to its own
experts. The shared expert and the dense MLPs stay tensor-parallel.

**No all-to-all.** At the MoE input every rank already holds every row of the batch (DP attention runs the MLP over
all groups' rows; sequence-parallel prefill streams gather the FFN block's input with a world all-reduce), and the
block's output is all-reduced anyway (the TP shared expert needs it). So the EP layout moves no extra byte between
ranks: a rank computes its own experts' pairs and the existing all-reduce adds the ranks (vLLM's all-gather /
reduce-scatter EP with the gather and the reduction the TP block already pays). For the record, an all-to-all costs
what an all-reduce of the same buffer costs at 32 ranks (`KILN_PROBE_A2A=1 python tools/profile_layer.py --allreduce
--ranks 32 --batch T`, kiln-mimo-trn1, logs s3 logs/kiln-mimo-trn1/ep-a2a-b{32,64,128}.log; T rows per destination,
[32 T, 4096] bf16 per rank, chained launches, per launch; the block exchange checked exactly):

| T (MB per rank) | all-to-all x1 | all-reduce x1 | all-to-all x12 | all-reduce x12 |
|---|---|---|---|---|
| 32 (8 MB) | 5.25 ms | 5.41 | 4.70 | 4.62 |
| 64 (16 MB) | 5.29 | 5.30 | 8.81 | 8.62 |
| 128 (32 MB) | 5.95 | 5.24 | 17.09 | 16.80 |

so a dispatch / combine EP would add two such collectives per MoE layer on top of the gather and reduction the
shared expert still needs.

**What it saves.** Under TP each rank gathers the x row of EVERY routed pair, writes every pair's full-width y and
re-reads it in the combine (~1.2 GB of DMA per rank per MoE layer at 4096 rows), on 64-row expert slices that keep
every vector instruction small ([128, 64]). Under EP a rank reads only its own ~1/32 of the pairs' rows and applies
each local expert to all of its rows in one weight pass.

**The kernel** (`kiln_moe_ep_kernel`, batches above 128 rows). Layout (`moe_ep.pack`, from the loader's EP tensors
after fit_e4m3_max): gate_up as the stationary [h, i] tiles of each 128-row I-chunk, down as the moving [i, h] rows,
the fp32 per-row scales as three bf16 parts hi + mid + lo (exactly the fp32 value; `split3`). A pass = one local
expert over up to LW lanes (128, or 256 from 2048 rows): x rows gathered and transposed; per I-chunk the gate and
up tiles DEQUANTIZED to bf16 (fp8 x scale in fp32, rounded once, as models/quant.dequant: the scales vary along a
tile's free axis, so each 512-column scale row is broadcast to the 128 partitions by one matmul of a [3, 128] ones
stationary over the three parts, into PSUM, and the vector engine multiplies the fp8 tile by it with one PSUM
operand), accumulated over the 32 h-tiles in PSUM with x^T moving; g, u rounded to bf16, the clamped SiLU product
rounded to bf16 (a^T [i, lane]); per 512-column chunk of the output the down tiles of all 16 I-chunks dequantized and
accumulated in PSUM over the whole intermediate dimension; the drain multiplies by each lane's routing weight and
rounds once; the rows are added into the output at the lanes' tokens by a scatter read-modify-write DMA
(`nisa.dma_compute`, bf16). The plan runs in the kernel: each pair's local expert (lmap, a per-rank buffer so every
rank traces the same graph), per-token membership, its token-order prefix count per expert (a triangular matmul per
token tile plus the earlier tiles' totals), passes per expert. Each lane's token is COUNTED rather than scattered: lane
j of a pass starting at rank j0 holds the (j0 + j + 1)-th token with a pair on the expert, which is #{t : incl_e(t) <=
j0 + j} for the inclusive prefix incl_e; the scalar engine computes sign(j0 + j + 0.5 - incl_e(t)) per token tile
(+-1, exact in bf16) and a ones matmul sums it over the tokens (count = (C + sum) / 2), C past the expert's last pair
(an empty lane, skipped by every DMA). Every local expert's first pass is static code; passes past an expert's first
(an expert with more than LW pairs here) run in a device loop whose trip count the plan computes, so any routing runs
exactly.

What it took, measured on kiln-ep-k1 (trn1.2xlarge, one NeuronCore, rank 0 of 32, 9 random experts in the loaded
layout: e4m3fn codes of 128 x 128 blocks then fit_e4m3_max, top-8 of 288 uniform unless stated; `python
tools/probe_moe_ep.py core|full`, p50 of synchronous calls minus the readback reduction):
- One pass (`core`, static passes over pre-gathered rows): 250 us per 128 lanes, 318 us per 256 lanes (9 passes 2.25 /
  2.86 ms, 25.8 / 40.5 TFLOPS). Vector-bound on the dequantization (tools/prof_engines.py: TENSOR_TENSOR 2142 us of
  2109 us of 9 passes, 0.62 us per [128, 512] op). The one-matmul scale broadcast is bit-identical to three
  accumulating selector matmuls (`--bc 3 1`) and 25% faster; kernel vs its host emulation 0.0030-0.0035 of the output's
  max (a bf16 ulp, as the TP kernel), both 0.005 from the fp32 reference.
- First full kernel (scatter of a token table, every pass in a device loop, a PE combine): C=4096 4.71 ms, C=1024
  3.26, C=128 3.07. Its profile (`tools/prof_bins.py`, new: engine busy per 200 us bin) at C=4096: plan 0.2 ms, the
  token-table scatter 0.55 ms (256 GpSimd indirect DMAs with every other engine idle), passes 2.95 ms, combine 0.9 ms
  (tensor-engine bound); device-loop passes cost ~90-170 us more each than static ones.
- As described above (lane tokens counted, scatter-add output, first passes static), against the TP prefill kernel on
  the same box and experts (`tools/probe_moe_prefill.py --format loaded`, engine-v0, KILN_MOE_PREFILL_SKIP=0 as the
  sweep runs it):

| rows C | local pairs (largest expert) | passes | EP kernel | TP prefill kernel |
|---|---|---|---|---|
| 128, uniform | 21 (5) | 9 | 2.74 ms (dequantize-first) | 4.17 |
| 1024, uniform | 251 (33) | 9 | 3.26 | 5.97 |
| 4096, uniform | 1050 (135) | 9 | **3.33** | **12.97** |
| 1024, a third of the tokens on local expert 1 | 588 (382) | 9 + 2 overflow | 3.41 | |
| 4096, the same skew | 2356 (1450) | 9 + 5 overflow | 5.13 | |
| 256, every token on experts 0..7 (all local) | 2048 (256) | 9 + 8 | 5.29 | |
| 1024, the same | 8192 (1024) | 9 + 56 | 20.86 | |

  All within 0.0017-0.0065 of their emulation (moe_ep.emulate; the hot cases add 8 bf16 read-modify-writes per
  token), rows with no local pair exactly zero. Overflow passes cost ~320-360 us each.

**Small batches** (`kiln_moe_ep_small`, at most 128 rows: decode at 16 per DP group is 64 rows): the dequantization is
a full 25 MB weight pass whatever the lane count, so a decode MoE call on the kernel above took 2.74 ms (9 passes)
against moe_dedupe's 1.44 ms. The small kernel instead runs passes of 16 lanes on the fp8 weights as stored: per
I-chunk and gate / up the 32 h-tile partials stacked in one PSUM tile [i, c, lane], multiplied by the per-row scales
(fp32, i on the partitions, read with a stride-0 access pattern over the lanes) and summed over c by a reduce over a
permuted access pattern, one instruction each; the down partials the same way per 128-column output tile ([h, m,
lane]), rounded to bf16 and transposed back to [lane, h] (moe_dedupe's per-row arithmetic; `tools/probe_ep_prims.py`
checks the two access patterns, max error 1.9e-9 of fp32 summation order, and an identity-matmul rebuild of fp32
scales from three bf16 parts, bit-identical). An unused expert's weight loads are skipped (a skipping DMA is allowed in
static code only: inside a device loop neuronx-cc refuses it, NCC_IGCA103 "Cannot spill memorylocation defined by
DMA skipping inside of loop"). C=64 (12 local pairs, 3 experts unused): **1.41 ms**; C=128: 1.46 ms; kernel vs
emulation 0.0011-0.0041. Tensor-engine bound (29K instructions, one LDWEIGHTS per 128 x 128 weight tile: the same
tile count TP's moe_dedupe loads for the same pairs). It needs the scales a second time with their rows on the
partitions (`dsg`, `dsd`, 768 KB per expert, ~7 MB per layer per rank).

**The MoE block** (router + experts + TP shared expert + all-reduce, `KILN_MOE_EP=<0|1> KILN_CC_ARGS=--model-type=transformer
KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_PROFILE_EXPERT_LAYOUT=loaded python tools/profile_layer.py --model
zai-org/GLM-5.3-Flash --tp 32 --dp-attention 4 --ranks 2 --prefill 1024|--batch 16 --pages 264 --layers 4 --what mlp
--sum-readback`, kiln-ep-k1, two live ranks, rank-0 shapes, random weights): 4096 rows TP 16.03 ms, EP **5.49 ms**;
64 rows (decode) TP 1.43 ms, EP 3.17 ms on the dequantize-first kernel (before the small kernel).

CPU (tests/test_moe_ep.py, gloo): EP equals TP on GLM-5.3-Flash (transformers 5.18; tp 4 with DP attention 2 and
piecewise graphs, and tp 2: tokens equal, max |dlogprob| 1.7e-6 / 1.4e-6) and on MiMo-V2 (tp 2 and 4: equal, 4.8e-7);
the layout round trip, split3's exactness, local_map, max_passes' bound and the emulation against an fp32 reference.

**Real routing, and why the random-token benchmark is EP's pessimistic case** (2026-10-04). `python tools/ep_routing.py run
--text-file wikitext2_test.txt --texts 2 --random 2 --tokens 4096 --save ep_routing.pt` (kiln-mimo-trn1's host CPU, the
real checkpoint, every MoE layer's router on the real hidden states of two 4096-token wikitext-2 pieces and two
random-token sequences drawn as bench/serve_sweep.py draws its prompts; ~17 min per sequence; data s3
logs/kiln-mimo-trn1/ep_routing.pt), then `stats` (batches as the sweep forms them: 4 chunks of 1024 tokens; log
ep-routing-stats.log). Busiest rank / mean pairs per rank at 4096 rows, mean over the 42 layers (mean 1024 pairs):

| placement | wikitext | random tokens |
|---|---|---|
| contiguous (rank r: experts 9r .. 9r + 8) | 2.8-3.0x | 3.7-3.8x |
| balanced greedily on the other half's loads | 2.3x | 3.0x |

The cause is near-always-on routed experts in the deeper layers: layer 20 on random tokens has four experts with 3711-4093
of 4096 tokens each (wikitext: 1909-2091, one or two experts); layers 3-6 have none. The EP kernel on the BUSIEST rank of
real routing (`python tools/probe_moe_ep.py full --chunks 4096 --routing-file ep_routing.pt --layer L --rank -1
[--sequences text0:0 text1:0 text0:2048 text1:2048]`, kiln-ep-k1, the kernel of 0af6710; TP's prefill kernel 12.97 ms
whatever the routing, its skip off):

| layer | wikitext, busiest rank | random tokens, busiest rank | random, overflow passes of 512 lanes (a104a13) |
|---|---|---|---|
| 3 | 1254 pairs, 3.68 ms | 1545 pairs, 4.40 ms | |
| 20 | 3337 pairs (largest expert 1909), 6.90 ms | 5084 pairs (largest 4090), 9.41 ms | 8.21 ms |
| 44 | 2970 pairs (largest 1275), 6.57 ms | 4876 pairs (largest 3132), 9.39 ms | |

So EP is faster on real text than on the benchmark's random tokens; the p5en vLLM reference ran the same random-token
dataset, so the G1 comparison stays like for like and real traffic (G1b) should do better.

**Hot experts are a property of the input distribution, so no static hot set.** Per layer, the 32 most-loaded experts on
wikitext and on random tokens overlap in 13% of their members on average (the top 8: 1.8%); the random-token top 32 holds
only ~11% of wikitext's pairs (wikitext's own top 32: 21-60%), and the wikitext top 32 ~5-35% of the random tokens' (their
own: 30-91%). A placement tuned on either is an artifact for the other, and choosing the hot set per batch at run time
would need every expert's slices resident on every rank (the TP layout's bytes again). What does carry over: a hot
expert's pairs past its first 256 run in overflow passes of 512 lanes (`kiln_moe_ep_kernel` LW2), one dequantization per
512 lanes instead of per 256 (layer 20 random 9.41 -> 8.21 ms, the output identical). A static hot-slice hybrid measured
before dropping it (`tools/probe_moe_ep.py hybrid`: the batch's own top 32 split into 8 slices of 256 rows over 8-rank
groups, the rest whole): layer 20 random cold part 3.53 ms + hot part 3.21 ms, layer 3 6.81 + 1.59 ms (the cold part on
the small-lane kernel at 64 lanes, which costs what a dequantization costs there): no better than plain EP even on the
distribution it was tuned for.

**End to end** (kiln-mimo-trn1, trn1.32xlarge, real weights, G64-4096-KV1.5 at conc 64, 128 requests, farm graphs;
`bench/serve_sweep.py --model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap
--input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets
1024 --max-num-seqs 64 --concurrency 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8 --requests 128` with
the env of the queue configs named):

| tree, config | out tok/s | TTFT p50 / p90 | ITL p50 | spot $ / M out | log (s3 logs/kiln-mimo-trn1/) |
|---|---|---|---|---|---|
| engine-v0 c7836c0, TP (q/v0-trn1 G64-4096-KV1.5-P12-K), earlier today | 88.3 | 11.0 / 126.3 s | 652 ms | 6.76 | 20261004T122641Z |
| feat/moe-ep 0af6710 (v0 base), EP (q/ep-trn1 G64-4096-KV1.5-P12-K-EP) | **97.3** | 9.4 / 108.0 s | 580.5 ms | 6.14 | 20261004T173518Z-serve_sweep.log |
| feat/moe-ep 1aebc7f (engine-v0 a2be5e2 merged), TP, KILN_MOE_EP=0 (q/v1-trn1 G64-4096-KV1.5-S20-SEP-P12-K) | 91.3 | 10.6 / 121.4 s | 629.5 ms | 6.54 | 20261004T175631Z-serve_sweep.log |
| feat/moe-ep 1aebc7f, EP (q/ep-trn1 G64-4096-KV1.5-S20-SEP-P12-K-EP), right after the row above | **99.0** | 9.2 / 105.7 s | 570.4 ms | **6.03** | 20261004T181307Z-serve_sweep.log |
| feat/moe-ep 2d140cf (engine-v0 b4f400f merged), TP, KILN_MOE_EP=0 (farm graphs q/final-c33a391) | 94.9 | 10.1 / 115.8 s | 604.5 ms | 6.29 | 20261004T185347Z-serve_sweep.log |
| feat/moe-ep 7c5cd7d (same base; overflow passes all 512 lanes, a104a13), EP (q/ep-trn1) | **106.4** | 8.4 / 96.5 s | 528.7 ms | **5.61** | 20261004T190618Z-ep-b4f4-serve_sweep.log |
| feat/moe-ep 2d140cf (same base; final kernel, two overflow loops, 2ca0773), EP (q/ep-trn1 -2ca0 graphs) | **106.9** | 8.3 / 95.8 s | 526.2 ms | **5.59** | 20261004T192313Z-serve_sweep.log |

So on one base and one box EP is +8.4% throughput, -13% TTFT p50 and p90, -9% ITL at conc 64 (the prefill steps are
shorter; decode is at parity, see the small-lane kernel). On the engine-v0 b4f400f base (env of the last three rows:
`KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki
KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20
KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1`, the fp8 KV cache
making the DSA pool cache separate) EP is **+12.1%** (94.9 -> 106.4 out tok/s), TTFT p50 10.1 -> 8.4 s and p90 115.8 ->
96.5 s (-17% both), ITL p50 604.5 -> 528.7 ms (-13%), spot $6.29 -> $5.61 / M out; the final kernel adds +0.5% (106.9,
$5.59), inside run-to-run noise. The p5en vLLM reference is $5.48-5.77 / M out at spot, so conc 64 sits inside that
band; strict parity with $5.48 needs ~109 out tok/s.

**Default** (feat/moe-ep 23df914): `KILN_MOE_EP` unset or `auto` turns EP on for the glm5_next family on trn1 / trn1n
(and on a host without a Neuron device, so the CPU tests exercise it) when tp > 1 and tp divides the routed experts, and
leaves every other model and platform on TP until measured there; `KILN_MOE_EP=1` / `0` force it either way
(models/decoder.py moe_ep_enabled; tests/test_moe_ep.py test_ep_default_*).

**Real-weight checks with EP** (kiln-mimo-trn1, tp=32, `KILN_MOE_EP=1 KILN_MOE_KERNEL=nki KILN_PIECEWISE_MOE_GROUP=12
KILN_CC_ARGS=--model-type=transformer KILN_LINEAR_ATTN_KERNEL=nki python tools/check_ppl.py --model zai-org/GLM-5.3-Flash
--tp 32 --piecewise --kv-cache-gb 1.0 [--text-file wikitext2_test.txt]`, farm graphs q/ep-trn1 ppl4-EP / pplwiki-EP, tree
0af6710; logs s3 logs/kiln-mimo-trn1/ep-ppl4.log, ep-pplwiki.log): France -2.147, Water boils -3.527, def add -0.956,
quick fox -1.714, mean **-2.074** (engine-v0 -2.073); wikitext slice **-0.552** over 3071 tokens (engine-v0 -0.551), by
chunk -0.768 -0.685 -1.102 -0.926 -1.211 -0.180 -0.361 -0.140 -0.295 -0.281 -0.337 -0.335. On the final tree
(2d140cf: engine-v0 b4f400f, two overflow loops; graphs q/ep-trn1 ppl4 / pplwiki -2ca0; logs ep-ppl4-2ca0.log,
ep-pplwiki-2ca0.log, ep-pplwiki-2ca0.json): the same four sentences -2.147 / -3.527 / -0.956 / -1.714, mean **-2.074**;
wikitext **-0.551** over 3071 tokens, by chunk -0.770 -0.693 -1.101 -0.929 -1.206 -0.183 -0.355 -0.137 -0.298 -0.275
-0.334 -0.333.

**Suite** (kiln-ep-ci, CPU, 23df914 with the auto default; one pytest process, `KILN_TEST_MODEL=Qwen/Qwen3-0.6B`):
654 passed, 30 skipped; the transformers-5.18 files (glm5_next, qwen4_exp, linear_serving, dsa_topk, dsa_select,
moe_ep, dp_attention, mtp_mla, mla) 258 passed, 8 skipped, plus the two test_ep_default_* tests on 3a2a065 (2 passed).
After merging engine-v0 6723dba (mixed batches; 24ed0fa): 658 passed, 40 skipped; the transformers-5.18 files plus
test_mixed_batch.py 273 passed, 8 skipped (logs s3 logs/kiln-ep-ci/ep-suite4.log, ep-suite4-tf518.log).

**Overflow passes, after the routing measurements** (kiln-ep-k1, busiest rank of random-token routing at 4096 rows):

| overflow passes | layer 3 | layer 20 | layer 44 |
|---|---|---|---|
| 256 lanes (the sweep's kernel) | 4.40 ms | 9.41 | 9.39 |
| 512 lanes (a104a13) | 5.19 | 8.21 | 8.20 |
| 512-lane passes for an expert's bulk, then one 256-lane pass for a remainder that fits it (2ca0773) | **4.22** | **8.05** | **8.03** |

(the plan builds two pass tables and the kernel runs two device loops; every expert on one rank at 1024 rows 20.9 ->
11.4 ms; outputs unchanged). Not available: GpSimd as a second dequantization engine (a tensor_tensor with an fp8 operand
on GpSimd fails neuronx-cc, NCC_IXCG965 "Instruction engine check failed (Pool)", `tools/probe_ep_prims.py --engines`), so
the vector engine bounds a pass (of the per-row form; the tile-scale form below is tensor-engine-bound, see the 2026-10-05
correction after "Where the busiest rank's 8 ms went").

## The token mixers' collectives inside their attention group, and the routing gather that stayed separate (2026-10-04, SDK 2.32, trn1.32xlarge)

**Group collectives for the token-mixer blocks (`KILN_SP_GROUP`, auto: on for the trn1 families; models/hybrid.py
_attn_rows, DecoderForCausalLM._sp_group_gather / _group_reduce_scatter / _sp_grp_call).** With sequence-parallel
streams under DP attention, the ranks of attention group g (ranks 8g .. 8g + 7 at DP 4, tp=32) hold exactly the
group-major batch's rows of that group (the SP row blocks 8g .. 8g + 7), and a token mixer reads only its group's
rows. So the block now gathers them over the group (a zero-padded [1024, 4096] all-reduce over 8 ranks, 4 chips)
instead of every row over the world (a zero-padded [4096, 4096] all-reduce over 32 ranks), runs the mixer on them
as before (`_attn_in` is the identity while the block is traced this way), and reduce-scatters the mixer's head
partials over the group: each rank keeps exactly its own SP rows, the ones `_sp_rows` took, so the FFN side (the
world gather the MoE and the expert-parallel path read) is unchanged. A subgroup collective's replica groups
enter the compile key, so each of the 4 attention groups compiles its own NEFF of every prefill piece (a rank
loads only its group's). Blocks of layer 4 (KDA, MoE) at 4096 rows, 32 ranks with the engine's real attention
groups (`tools/profile_layer.py ... --prefill 1024 --part-layers 4 --sp`, feat/prefill-grp, kiln-pf-32b, log
20261004T195414Z-profgrp.log; standalone graphs, each paying its collectives' fixed cost):

| piece | ms (p50) |
|---|---|
| SP attention block, world gather + world reduce-scatter (KILN_SP_RS) | 8.714 |
| SP attention block, group gather + group reduce-scatter (KILN_SP_GROUP) | **5.690** |
| world gather [128 -> 4096, 4096] alone | 6.848 |
| group gather [128 -> 1024, 4096] alone | 4.609 |

CPU (gloo; tests/test_dp_attention.py test_glm5_next_sequence_parallel_streams at tp=4 / dp_attention 2, where
the group is 2 ranks): greedy tokens equal to replicated streams with it on and with it off.

**The routing did not ride in the FFN gather.** Packing each rank's routing beside its rows as bf16 columns (the
weights as bf16, which both the prefill and the expert-parallel kernels read; each expert index as two exact
small integers) turns the FFN block's two gathers into one, but the block got slower (layer 4, same harness, log
20261004T192942Z-proffuse.log): 15.196 ms with the routing in the rows' gather against 14.104 ms with a gather of
its own. The packed payload is [r, 4096 + 24] per rank; the likely cost is that width, which is not a multiple of
128: the all-reduce of a [4096, 4120] buffer and the copies that slice the [4096, 4096] rows and the routing back
out of it, against a separate [4096, 16] fp32 gather that costs ~0.5 ms. Dropped; the routing keeps its own gather.
**Method: one NEFF's time varies between processes.** The same cached graph (`first call 0.4 s`) of the layer-20
busiest-rank kernel ran 9.55, then 8.04 and 8.04 ms in three consecutive processes on kiln-ep-k1 (and 9.67 / 8.03 in
other runs); null graph and readback reduction unchanged. From here kernel numbers are the minimum over 3 processes.

**Expert groups (a negative result).** `tools/probe_moe_ep.py full --group g`: g ranks share 9 g experts, each rank a
1 / g intermediate slice of each (distribution-agnostic: a hot expert's pairs spread over g ranks). Busiest group's
rank at 4096 rows, min of 3 (log s3 logs/kiln-ep-k1/*-epgrp2-*):

| layer, routing | g 1 | g 2 | g 4 |
|---|---|---|---|
| 20 random | 8.03 ms | 6.81 | 7.86 |
| 20 wikitext | 6.21 | 5.77 | 6.15 |
| 44 random | 8.02 | 7.55 | 7.26 |
| 44 wikitext | 5.85 | 6.28 | 5.85 |
| 3 random | 4.21 | 4.59 | 5.69 |

Halving each pair's matmuls does not pay for the lanes every rank of the group gathers, transposes and writes back at
full width, plus twice the passes; dropped.

**Where the busiest rank's 8 ms went** (layer 20 random, `tools/prof_engines.py` / `prof_bins.py` on the probe's
graph): the vector engine 5.9 ms busy (TENSOR_TENSOR, the dequantization, 4.4 ms) and waiting 5.4 ms, the tensor engine
waiting 5.2 ms; the per-row form's 384 scale-broadcast matmuls per pass also tie the two engines together through PSUM.
The tensor engine itself (`tools/probe_ep_prims.py --pe`, 768 chained matmuls): 0.39-0.48 ns per moving column at
every width, i.e. ~2.45 G columns/s, the same for bf16 and fp8 operands; `--pe-shape`: against 16 moving lanes a matmul
costs 30 ns (the stationary load hidden), a [128, 16] stationary against 512 columns 215 ns.

**Correction (2026-10-05): the tile-scale kernel is tensor-engine-bound, not vector-bound, and "busy" above was a sum of
overlapping durations.** tools/prof_engines.py adds up each instruction's `duration`, and an engine's instructions overlap
in the profile (a matmul's record spans its pipeline latency; consecutive vector ops overlap too), so its "busy" can exceed
the wall time (the tensor engine read 11.3 ms busy over a 5.4 ms kernel). The right reading is the UNION of an engine's
instruction intervals, which tools/prof_ops.py (new) prints. Re-measured on the default tile-scale form (kiln-mk-k1,
trn1.2xlarge, SDK 2.32, nki 0.6.0, engine-v0 e449b98 kernel, layer 20's busiest rank of the real routers on random tokens,
`tools/probe_moe_ep.py full --fit block --chunks 4096 --routing-file ep_routing.pt --layer 20 --rank -1 --save-inputs ...`,
then `tools/prof_engines.py <hash> <inputs>` and `tools/prof_ops.py <hash>`): the tensor engine is busy 83% of the 5.40
ms window and 85-89% inside every pass, the vector engine 62% (the fp8 dequantization, 13824 TENSOR_SCALAR at 279 ns mean),
the scalar engine 54%, DMA active 66%. The figures "vector 5.09 ms busy (dequant 4.15), scalar 4.03 ms" quoted for this
kernel were the overlapping sums. Details and what follows from it: "MoE kernels against their floors" below.

**The tile-scale form** (feat/moe-ep edc04c1, KILN_MOE_EP_FIT=block, KILN_MOE_EP_TILES=1, both default). The EP loader
fits GLM-5.3-Flash's 128 x 128-block FP8 per block (fit_e4m3_max row_group 128: a block with a code past 240 is
halved whole, which changes only codes below 2^-5), so every kernel tile has one scale; `moe_ep.pack(tiles=True)`
keeps them as tsg [El, M, 2, CT] / tsd [El, M, CT] (decided from the checkpoint's format, weight_scale_inv with block
128, never from values, so a capture on zeros packs as the device does; the values are checked) and leaves out the
per-row split scales (10.5 MB per layer and rank, ~440 MB per rank). Each [128, 128] tile is dequantized by one
per-partition-scalar instruction, alternately on the vector and the scalar engine (`--dq-tile`, per [128, 512]:
tensor_tensor against a PSUM scale row 0.47 us, tensor_scalar per 128 columns 0.48-0.60, activation(scale=) 0.45-0.47,
the two engines alternating 0.27-0.35, all bit-exact against bf16(fp32 code x scale); against SBUF scales read through
a stride-0 pattern 0.99), with no broadcast matmuls. Busiest rank at 4096 rows, min of 3 (kiln-ep-k1, logs *-epts-*):

| routing | per-row form | tile-scale form |
|---|---|---|
| layer 3 random | 4.21 ms | **2.77** |
| layer 20 random | 8.03 | **5.61** |
| layer 44 random | 8.02 | **5.54** |
| uniform, rank 0 | 3.34 | 2.18-3.29 (noisy) |

Kernel vs emulation as before (rel 0.0045 at layer 20). Real weights (kiln-dk-32, edc04c1, farm queue q/ept-edc0 with
KILN_MOE_EP unset; logs s3 logs/kiln-dk-32/20261004T214106Z-ept-ppl4.log, 20261004T214609Z-ept-pplwiki.log): 4
sentences -2.177 / -3.534 / -0.950 / -1.726, mean **-2.087** (TP -2.073; row-form EP -2.074); wikitext **-0.547** over
3071 tokens (TP -0.551), by chunk -0.774 -0.690 -1.084 -0.924 -1.213 -0.179 -0.340 -0.141 -0.298 -0.257 -0.329 -0.332
(row-form EP 2d140cf minus these: -0.004 +0.003 +0.017 +0.005 -0.007 +0.004 +0.015 -0.004 0.000 +0.018 +0.005
+0.001). G64-4096-KV1.5-S20-P12-K at conc 64 (kiln-dk-32, 128 requests, log 20261004T215127Z-ept-G64-sweep.log):
**114.9 out tok/s**, TTFT p50 7.6 s / p90 87.1 s, ITL p50 486.7 ms, spot $5.20 / M out (row-form EP on kiln-mimo-trn1:
106.9).

**EP with mixed batches** (kiln-mimo-trn1, feat/moe-ep 24ed0fa = engine-v0 6723dba + EP, farm queue q/stack-24ed,
G64 at `--state-checkpoints 4`, 128 requests): EP 106.9 out tok/s (TTFT 8.3 / 95.8 s, ITL 526 ms; log
20261004T205716Z-ep24-G64-ck4-sweep.log), EP + KILN_MIXED_BATCH=1 **112.7** (TTFT 7.7 / 89.6 s, ITL 496 ms, spot
$5.30; 20261004T210933Z-ep24-G64-mx-ck4-sweep.log). Greedy text (`tools/check_mixed.py`, 64 prompts, 64 new tokens):
on check_ppl's LONG_TEXT 56 / 64 outputs identical (the 8 others the 700-token prompts), teacher-forced |dlogprob| over
mixed decode rows max 0.378, p99 0.104, mean 0.0029, signed (mixed minus unmixed) mean -0.00055 (TP's mixed: p99 0.053,
the unmixed-vs-unmixed floor 0.068); on wikitext-2 6 / 64 identical, mixed decode rows mean |d| 0.039, p99 0.29, signed
-0.0035, decode calls mean 0.045 / signed -0.0049, prefill chunks 0.027 / +0.0073 (logs ep24-text-compare.log,
ep24-wtext-compare.log). Natural text is far less confident than the repeated passage, so its floor was measured on
the same 64 windows with two numerically neutral unmixed engines, EP row form (kiln-mimo-trn1) against EPT (kiln-dk-32,
edc04c1): 9 / 64 identical, token agreement 0.454, teacher-forced decode calls mean |d| 0.037, p99 0.30, signed -0.0028,
prefill chunks 0.038 / +0.021. Against it: EP mixed vs unmixed 6 / 64, 0.473, mixed decode rows 0.039 / p99 0.29 /
signed -0.0035; EPT mixed (CK4) vs EPT unmixed 8 / 64, 0.447, mixed decode rows 0.042 / p99 0.42 / signed +0.0048,
decode calls 0.035 / +0.0022, prefill chunks 0.034 / -0.0072 (logs s3 logs/kiln-dk-32/ept-wtext-compare.log). So on
natural text mixed batches sit at the floor of two accepted engines, with no systematic shift. EPT + mixed at G64 CK4
(kiln-dk-32, edc04c1, q/ept-edc0, 128 requests): **121.3 out tok/s**, TTFT p50 7.0 s / p90 81.2 s, ITL p50 458.7 ms,
spot **$4.92 / M out** (log 20261004T222144Z-ept-G64-mx-ck4-sweep.log); EPT alone on kiln-mimo-trn1, the box of the
row-form 106.9: 114.9 (20261004T222230Z-ept-G64-sweep.log).

**The automatic default needs 8 decode rows per DP-attention group** (38bac15): G16 (16 seqs over 4 groups = 4 rows)
EP 80.5 against TP 82.9 out tok/s, ITL 168.8 against 156 ms (20261004T204415Z-ep24-G16-sweep.log); F0 (8 rows) EP 98.9
against TP 90.8, ITL 287.5 against 311 ms, TTFT 7.9 / 34.3 s against 9.7 / 42.3 (20261004T203230Z-ep24-F0-sweep.log).
So KILN_MOE_EP=auto is off below max_num_seqs / dp_attention = 8 (EP_AUTO_MIN_DECODE_ROWS); the compile farm captured
G16 on 38bac15 to exactly q/final-c33a391's 13 TP keys and F0 to exactly q/stack-24ed's 14 EP keys.

**Decode** (the small-lane kernel; MoE block from `tools/profile_layer.py --what mlp --batch B --ranks 2 --dp-attention
4`, rank 0, random weights, uniform routing, kiln-ep-k1):

| rows per group | TP block | EP block |
|---|---|---|
| 4 | 0.70 ms | 1.57 |
| 8 | 1.08 | 1.55 |
| 16 | 1.42 | 1.63 |
| 32 | 1.88 | 1.64 |

EP is flat: each of the 9 static passes costs ~140 us of compute whether its expert has pairs or not (C=16 with 7 of 9
unused 1.28 ms, all 9 used 1.48; static weight DMAs instead of the skipping dynamic ones 1.50). The profile at C=64
(1.20 ms): the tensor engine issues a 16-lane matmul every ~21.5 ns when it runs (0.31 ms of the 1.2), and idles 352 us
waiting on the GpSimd dynamic-DMA queue (each pass's x-row gather, ~19 us at its start, and the weight loads), 235 us on
the vector engine and 63 us on the scalar engine. Skipping unused experts cannot close the gap at 16 rows per group,
because the step waits for the busiest rank: with uniform routing (simulated, 400 draws) the busiest of 32 ranks uses
6.2 / 8.2 / 9.0 / 9.0 of its 9 experts at 4 / 8 / 16 / 32 rows per group (the mean rank 3.3 / 5.4 / 7.5 / 8.8). A first
tile-scale small kernel (each tile's scale folded into the bf16 moving operand, no PSUM stacks: branch ep-small-tiles,
not merged) is correct (rel 0.0017 against its emulation) but not faster: 1.39 / 1.58 / 1.96 / 2.68 ms at C = 16 / 32
/ 64 / 128 on layer 20's busiest rank against 1.31 / 1.49 / 1.85 / 2.59 for the per-row kernel. On the way: four
accumulation groups in 128-column slices of one PSUM tile, read once after the fourth, lost terms on the device (rel
0.25 against the emulation; one PSUM tile per group, read right after it, fixed it), as the decode agent found on its
score kernel.

**The decode floor is bytes.** HBM -> SBUF on one NeuronCore (`tools/probe_ep_prims.py --dma`, one layer's 9 x 25 MB
of EP experts in chunks of 1-4 MiB through 2-8 buffers, static or per-expert dynamic offsets): 224-230 GB/s in every
variant, so a rank's 9 whole experts take 1.0 ms. With the busiest rank's used experts above, EP's floor is 0.68 / 0.91
/ 0.98 / 0.98 ms at 4 / 8 / 16 / 32 rows per group, while TP reads the touched experts' slices on every rank (~81 /
133 / 188 / 219 MB: 0.35 / 0.58 / 0.82 / 0.95 ms). EP decode therefore cannot match TP at 8 rows per group whatever
the kernel, and at 16 only within ~15% of its floor (the kernel is ~70% over it). Interleaving the first passes across
the experts step by step (KILN_MOE_EP_SMALL_IL=1, adff2eb) cuts the layer-20 busiest rank's kernel 1.31 / 1.49 / 1.85 /
2.59 -> 1.20 / 1.35 / 1.71 / 2.45 ms at C = 16 / 32 / 64 / 128.

**The decode step, end to end** (`tools/time_decode.py --all-buckets`, d17a26a: wall p50 of 64 steps launch to logits,
no piece profiling; kiln-mimo-trn1, tp 32 DP 4, 12-layer decode graphs, page bucket 264, farm queue q/epil-adff on
adff2eb, two repetitions; logs s3 logs/kiln-mimo-trn1/*-tdw-*):

| rows per group | TP | EPT (KILN_MOE_EP_SMALL_IL=0) | EPIL (interleaved) |
|---|---|---|---|
| 8 | 112.6 / 113.0 ms | 139.4 / 139.2 | 139.1 |
| 16 | 161.6 / 161.1 | 188.0 / 187.1 | 186.6 |
| 32 | 252.1 / 252.3 | 280.8 / 280.3 | 281.5 |

So EP's decode step is +24 / +16 / +11% over TP's, and the interleave that cut the kernel alone by 9% changes nothing in
the step (EPIL at G64 conc 64: 115.4 out tok/s against EPT's 114.9; with mixed batches 121.6 against 121.3). The 64-lane
overflow passes from 64 rows on (6449a2c; the busiest rank's kernel at C = 64 / 128 1.71 / 2.46 -> 1.57 / 2.20 ms in
the probe) made the step slower: 139.9 / 138.7, 194.5 / 195.9, 286.1 / 285.5 ms at 8 / 16 / 32 rows per group
(q/eplw2-6449, logs *-tdw-eplw2-*), the wider passes mostly empty for the experts most layers have just past 16
pairs. Both are reverted on feat/moe-ep (the kernel back to engine-v0 9853406's).

**Prefill 8192 (2048 rows per group) still does not fit trn1 with EP** (compile farm, tools/hbm_estimate.py over
q/dsa-trn1 G64-8192-P12-K's compiled NEFFs plus tools/tensor_bytes.py on 9853406 with the final G64 env, KV 1.5 fp8, 64
sequences): the queue-instance spill rings sit entirely in the three 2048-row prefill groups (5.70 / 6.11 / 4.02 M runs,
6.9 GiB at the ~469 B per run one of them showed on the device), in their XLA parts: at 4096 the EP kernel removed only
8-17% of a prefill group's instance runs. EPT tensors 13.36 GiB + code ~0.34 + scratchpad ~0.31 + rings ~6.1 + decode
spill 0.06-1.0 + 0.13 fixed = ~20.3-21.3 GiB, 4.3-5.3 GiB over the core (TP ~21.8); the decode agent's decode kernels
free another 0.8-0.96 GiB (feat/decode-step), which does not close it. A 1536-row group was never captured; its rings
are not linear in rows (the 1024-row groups have 4.2 M runs each and three of them load).

## Prompt caching for GLM-5.3-Flash on the device: exactness, noise, a shared-prefix sweep (2026-10-04, SDK 2.32, trn1.32xlarge)

kiln-dk-32 (trn1.32xlarge, us-east-2c), zai-org/GLM-5.3-Flash bf16 (eb9eb208), feat/prompt-cache on the engine-v0
1d63405 graph code (tree 1a67cd25; the branch changes only the scheduler, bench and tools, so every graph came from
the q/v1-trn1 farm captures, 0 compiles on the box). Configuration: the q/v1-trn1 configs' env and argv verbatim
(`KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki
KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20
KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1`, `--tp 32
--dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup
--prefill-tokens 4096 --prefill-buckets 1024`, and F0-4096-KV1.5-S20-P12-K `--max-num-seqs 32 --concurrency 32
--decode-buckets 8 --kv-cache-gb 1.5` (bf16 KV: in-place DSA pool keys) or G64-4096-KV1.5-S20-P12-K `--max-num-seqs
64 --concurrency 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8` (pool keys off at that tree),
run through `/opt/kiln/pcrun.sh <name> <config uri> <tool> ...` (pulls the cache, applies the config, logs to
s3://<your-bucket>/logs/kiln-dk-32/pc-<name>.log).

**Hits work end to end** with DP attention, sequence-parallel streams, SP routing, the in-place DSA pool keys,
reserve admission and overlap. `tools/check_prefix_cache.py --text-file wikitext2_test.txt --prefix-len 6144
--prefixes 8 --new-tokens 64 -- <F0 argv>` (pc-chk3): 8 prompts per phase sharing one 6144-token wikitext prefix
each with a cold run, one per DP group, all in the same calls; every hit resumed from 6144 (5984 for a prefix of
6000 tokens: the page below), a phase of 8 hits took 10.8 s against 23.7 s cold.

**The cached path is exact; the device is not run-to-run exact in every composition.** Measured:

| comparison (64 greedy tokens, chosen-token and top-5 logprobs) | tokens equal | mean / max \|dlogprob\| |
|---|---|---|
| cold vs cold, same prompts, same composition (pc-chk3 8 prompts, pc-chk4 4) | 12/12 | 0 / 0 |
| cold vs cold after a flush (pc-chk5, 6 prompts each alone) | 6/6 | 0.0025 / 0.109: 4 identical, 2 differ only after token 16 |
| the same cold run with another request queued beside it (pc-chk3 `save`: no cache used at all) | 3/8 | 0.0094 / 0.18 |
| hit vs cold, batched (pc-chk3 `hit`: a suffix never computed, resumes from the junction checkpoint) | 6/8 | 0.0080 / 0.18 |
| hit vs cold, each request alone (`--solo`, pc-chk5 `solo-hit`) | **6/6** | **0 / 0** |
| junction checkpoint (recomputed by a second request) vs a cold run's state at 6144 (rank 0's shard, 68 pools) | | **0 / 0** on 6/6 prefixes |
| hit resuming from 5984 (off the 1024-token chunk grid) vs cold (`solo-unaligned`) | 0/6 | 0.054 / 0.75 |

So: what else shares a call moves logprobs by ~0.01 nats (the MoE kernels see a different batch), a different chunking
by ~0.05 (bf16 rounding of the chunked delta rule and the selection that follows), and with the composition and the
chunk grid held fixed a hit is bit-identical to a cold run: the KDA conv and recurrent state restored from a
checkpoint, the MLA latent pages, the DSA pool keys stored in them and the sequence-parallel streams all reproduce the
cold arithmetic. (Pools never straddle the cached / uncached split: a cached prefix is a whole number of 32-token
pages and a pool is 4 tokens.) Divergences in the batched runs happen at near ties (the reference's top-2 margin
0.06-0.13 nats at the first differing token). The unaligned case against a cold run whose chunks start where the
hit's do (`solo-unaligned-same-grid`, the merged tree below, pc-chk7): **4/4 bit-identical**, and the two cold runs
against each other (`chunking`, no cache at all) differ exactly as much as the hit does from the default-grid cold run
(1/4 equal, mean 0.0171, max 0.22): an off-grid hit costs only the rounding of a different chunking.

**The same on the merged tree** (feat/prompt-cache with engine-v0 c33a391: the prefill-perf3 MoE pipeline, SP
reduce-scatter, separate pool keys under FP8; q/final-c33a391 farm graphs F0-4096-KV1.5-S20-P12-K, pc-chk7,
2026-10-04 19:04-19:18 UTC): solo-save 4/4 and solo-hit 4/4 bit-identical over 64 tokens, the junction checkpoint
equal to the cold state on 4/4 prefixes, unaligned on the same grid 4/4 bit-identical. FP8 KV with the separate bf16
pool-key cache (G64-4096-KV1.5-S20-P12-K, pc-chk8, 19:20-19:31 UTC): repeat 4/4, solo-save 4/4 and solo-hit 4/4
bit-identical over 64 tokens, the junction checkpoint equal to the cold state on 4/4 prefixes.

`--plp` (prompt logprobs of the 2048 uncached positions) does not fit at F0 KV 1.5 GB: the prompt-logprob post graph
compiled (83 s) and failed to load with `Allocation Failure` (pc-chk1), so the comparison above uses output tokens.

**The second request sharing a prefix used to recompute it** (a junction is checkpointed by the request that finds
the KV of an earlier one without a state after it: SGLang's mamba_branching_seqlen). The scheduler now takes junctions
ahead (`SchedulerConfig.ckpt_lookahead`, EngineConfig.state_checkpoint_lookahead, default on): the longest page-aligned
prefix a request shares with another queued or running request of its group becomes a checkpoint target of the one
that computes it first, and the other waits until it can resume from it. DP placement counts a prefix a group will
compute for a queued request as held there, charges a group the tokens its requests were charged, and puts a group
whose requests fill max_num_seqs after any group with a free slot. Same-box A/B at conc 64 (G64 config, 256
requests, 4 prefixes of 6144 tokens, a cold cache, the new placement in both; pc-sw64 / pc-sw64n with
`--no-ckpt-lookahead`): lookahead 238 of 256 requests hit (hit rate 0.697 of the 0.75 possible), 165.7 out tok/s,
TTFT p50 3.6 s, $0.000923 per request at spot; without it 224 hit (0.656), 150.4 out tok/s, $0.001016: **+10.2%
throughput, misses 32 -> 18**. Longest-prefix-first admission inside each group (`--schedule-policy lpm`, SGLang's
lpm) on the merged tree (pc-swGl against pc-swG, same box, G64 c33a391 graphs, 75% shared): cold 183.7 against 180.4
out tok/s with FCFS (+1.8%, one run each; hit rate 0.691 against 0.697), warm 231.5 against 231.6 (with every request
hitting the same length the order does not change). FCFS stays the default.

**What a checkpoint costs** (pc-chk7 `--time-copies`, rank 0 alone, p50 of 20, F0 shapes, trn1): one state row is
18,452,480 B per rank at attention TP 8 (DP 4): 34 KDA layers x 8 heads x 128 x 128 fp32 plus the conv history, 4x the
4.6 MB at attention TP 32. The copy graph (a restore or a save is one row; buckets of at least 4 pairs) takes 1.91 /
2.45 / 3.99 / 6.25 ms for 4 / 8 / 16 / 32 rows: against a ~1.03 s prefill call, nothing. The checkpoint rows
(state_checkpoints default 2 x max_num_seqs per group: 16 at F0, 32 at G64) hold 0.295 / 0.590 GB per rank of the
14.08 / 14.52 GB of tensors (tools/tensor_bytes.py: layer_state 0.487 / 1.054 GB); tools/hbm_estimate.py puts the
whole configuration at 18.08 / 18.59 GiB by the farm rule (q/final-c33a391 keys; the rule overestimates, both load).
The sweeps (docs/price-performance.md "G1b") held 4-5 checkpoints per group, so 16 rows are plenty at K = 4 prefixes; the
rows are a graph shape, so shrinking them to give the bytes to KV means new graphs, and neither config is KV-bound.

**Checkpoint policy for this model.** Junction checkpoints (exactly where requests branch, ahead with lookahead) are
the default and what the sweep uses; `state_checkpoint_interval` stays 0. A periodic checkpoint is only free when it
falls on the per-group chunk grid (1024 tokens here): anywhere else it splits a chunk, and with the single 1024-row
prefill bucket a split chunk costs a whole extra ~1.03 s call for every group. The same holds for a junction that is
off the grid (one extra call once per prefix and group, repaid by the first hit) and for an identical prompt repeated,
whose KV junction sits one page before its end (`hit-identical` in pc-chk3: chunks 6144-7168, 7168-8160, 8160-8192).
Decode checkpoints every 256 tokens (state_track_interval) cost one row copy each and serve the next turn of a
conversation.

**The host tier under DP attention** (engine/hicache.py GroupMoves, new here: each group's evicted pages and
checkpoints go to its own ranks' host memory, and placement counts them): rank 0's eager copies (pc-chk7, p50) are
0.313 ms to save a 540,672-byte page (per rank: 11 DSA layers' latent, rope and indexer rows of 32 tokens) and 0.290 ms
to load one, 3.01 / 2.53 ms to save / load an 18.45 MB state row. In the engine at tp=32 every page is one broadcast
call that the group's 8 ranks execute and the rest skip, and that round trip dominates: **device check** (pc-chk9,
merged tree, F0 graphs, `--solo-phases host -- --hicache-host-gb 2`, 2026-10-04 20:43-20:50 UTC): X_k and Y_k of 4
prefixes computed (junction checkpoints), every group's whole cache evicted to its host tier (1288 pages and 4
checkpoints in 4.44 s, 3.4 ms per page), then Z_k: each restored its 192 pages and the checkpoint (768 pages, 4
checkpoints) and resumed from 6144, **4/4 bit-identical to the cold runs over 64 tokens**; the phase took 36.2 s
against 33.6 s for the same hits from device memory (pc-chk7), ~0.65 s per 6144-token restore, while recomputing it is
~1.5 s of device time (six 1024-row chunk slots of ~1.0 s calls shared by four groups) and ~6 s of that group's
prefill. One message per eviction or restore instead of one per page would bring it toward the 58 ms of the copies
themselves. At 4 GB per rank (128 GB of the host's 495 GB) a group's tier holds 3,984 pages (127K tokens, 20 prefixes
of 6144) plus 116 checkpoints, beside the device's 2,979 pages per group at F0; the sweeps here (4 prefixes) fit on
the device and do not use it.

**Suite** (kiln-pc-ci2, m7i.8xlarge, logs s3://<your-bucket>/logs/kiln-pc-ci2/pc-suite4.log and
pc-suite4-518.log; one pytest process, `KILN_TEST_MODEL=Qwen/Qwen3-0.6B`, the venv's transformers
5.15) on feat/prompt-cache 20ad5ec (rebased onto engine-v0 e2c6fad; later commits are docs): 662 passed, 40 skipped;
the transformers-5.18 files (glm5_next, qwen4_exp, linear_serving, dsa_topk, dsa_select, moe_ep, dp_attention,
mtp_mla, mla, mixed_batch; `pip install --no-deps --target` transformers 5.18.0, huggingface_hub 1.33.0, tokenizers
0.23.2 first on PYTHONPATH) 277 passed, 8 skipped. New tests: test_linear_serving.py
test_burst_sharing_a_prefix_hits_from_the_second_request, test_request_arriving_mid_prefill_resumes_from_the_shared_prefix
(sync and overlap), test_dp_attention_host_tier; test_dp_attention.py test_placement_follows_queued_prefixes_and_free_slots.

**End to end with expert parallelism** (kiln-pf-32b, back to back 21:04-21:35 UTC, the q/ep-trn1 EP-2ca0 command:
`KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki
KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_EP=1 KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20
KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 python
bench/serve_sweep.py ... --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64 --concurrency 64
--decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8`, 128 requests, every graph from the compile farm,
q/pfgrp3-trn1, whose capture of the group path came from all 32 ranks: 24 distinct graphs against 15, the three
prefill pieces once per attention group):

| tree | conc 64 out tok/s | TTFT p50 / p90 | ITL p50 | spot $ / M out | log |
|---|---|---|---|---|---|
| engine-v0 e2c6fad (EP merged) | 107.0 | 8.3 / 95.8 s | 526 ms | 5.58 | 20261004T210420Z-ep2-base |
| + KILN_SP_GROUP (feat/prefill-grp2 0f247d6) | **114.6** (+7.1%) | 7.6 / 87.5 s | 489 ms | **5.21** | 20261004T212831Z-ep2-grp |

The first attempt (log 20261004T211644Z-ep2-grp) never reached the group path for two reasons, both fixed: the
engine created attention groups only without DP attention (so `_sp_grp_ok` was false in serving and in the farm's
capture, which then equalled the baseline's keys), and the auto rule's platform lookup opens files, which dynamo
cannot trace (`Unsupported: Failed to trace builtin operator open`): the rule is now read when the model is built.

Real-weight ppl at `--dp-attention 4`, so that the group path runs (`tools/check_ppl.py --model zai-org/GLM-5.3-Flash
--tp 32 --dp-attention 4 --piecewise --kv-cache-gb 1.0 [--text-file ...]` with the serving env plus
`KILN_MOE_PREFILL_MIN_TOKENS=1`, farm graphs q/pfgrp3-trn1; kiln-pf-32b):

| tree | France | Water boils | def add | quick fox | mean | wikitext-2 | logs |
|---|---|---|---|---|---|---|---|
| engine-v0 e2c6fad | -2.215 | -3.518 | -0.949 | -1.709 | -2.091 | -0.556 | 20261004T215357Z, 215900Z-*-dp4-base |
| + KILN_SP_GROUP | -2.215 | -3.518 | -0.949 | -1.709 | -2.091 | -0.551 | 20261004T214252Z, 214751Z-*-dp4-grp |

wikitext by chunk with it: -0.770 -0.687 -1.100 -0.932 -1.214 -0.182 -0.358 -0.146 -0.308 -0.264 -0.330 -0.326 (base:
-0.801 -0.706 -1.099 -0.935 -1.206 -0.188 -0.356 -0.143 -0.300 -0.270 -0.331 -0.332).

## The final measurement: group collectives on the merged tree (2026-10-05 UTC, SDK 2.32, trn1.32xlarge)

feat/prefill-grp2 **2b3bdbe** (tree e5c1ea4c) is engine-v0 f9dc4c4 (tile-scale expert parallelism the default, mixed
batches opt-in) merged with `KILN_SP_GROUP`. The CPU suite on it (kiln-pf-ci4, one process, log
/opt/kiln/logs/20261004T232522Z-pytest-g6-main.log): 712 passed, 10 skipped; test_inkling.py 7 passed. Every graph
from the compile farm, q/pfgrp6-trn1 (the group path's configs captured on all 32 ranks, since each attention group
has its own prefill keys; the f9dc4c4 baseline on rank 0), 0 device compiles in every run. The command is the farm
config's, `KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto
KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki
KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1
KILN_SP_ROUTE=1 KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/pfgrp6-trn1/ python
bench/serve_sweep.py --model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap
--input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096
--prefill-buckets 1024` plus, per config: G64 `--max-num-seqs 64 --concurrency 64 --decode-buckets 16
--kv-cache-gb 1.5 --kv-cache-dtype fp8` (mixed: `KILN_MIXED_BATCH=1` and `--state-checkpoints 4`), F0
`--max-num-seqs 32 --concurrency 32 --decode-buckets 8 --kv-cache-gb 1.5`, G16 `--max-num-seqs 16 --concurrency 16
--decode-buckets 4 --kv-cache-gb 0.65`. `KILN_SP_GROUP` is not set: auto turns it on for trn1. 128 / 64 / 32
requests at conc 64 / 32 / 16. Logs under s3://<your-bucket>/logs/kiln-pf-32b/ and logs/kiln-dk-32/.

| config | baseline | + KILN_SP_GROUP (2b3bdbe) | TTFT p50 / p90 | ITL p50 | spot $ / M out | prefill call | log |
|---|---|---|---|---|---|---|---|
| G64, tile-scale EP | 114.9 (engine-v0 9853406, kiln-mimo-trn1 20261004T222230Z-ept-G64; kiln-dk-32 the same) | **123.1** (+7.1%) | 6.9 / 79.4 s | 453 ms | **4.85** | 0.591 s | kiln-pf-32b 20261004T235630Z-fin-g64-grp |
| G64, tile-scale EP + mixed batches CK4 | 121.3 (kiln-dk-32 20261004T222144Z-ept-G64-mx-ck4) | **132.0** (+8.8%) | 6.3 / 72.4 s | 419 ms | **4.52** | n/a | kiln-dk-32 20261005T002109Z-fin-g64mx-grp |
| F0, tile-scale EP | 104.0 (f9dc4c4, same box, 20261005T001744Z-fin-f0-base; row-form EP was 98.9) | **110.1** (+5.9%) | 6.5 / 39.5 s | 254 ms | **5.42** | 0.593 s (base 0.665) | kiln-pf-32b 20261005T000827Z-fin-f0-grp |
| G16, TP (the EP gate leaves 4 decode rows per group on TP) | 82.9 (c33a391, kiln-g2-trn1 fin-G16) | **86.5** (+4.3%) | 8.7 / 30.4 s | 149 ms | **6.90** | 0.893 s | kiln-dk-32 20261005T001414Z-fin-g16-grp |

"prefill call" is serve_sweep's device time split (the mean prefill call over the run's steps, fitted from the
steps' wall times); with mixed batches a step carries both kinds of rows, so its fit (0.403 s prefill, a negative
decode) does not separate them. G64's 4096-token prefill call at 0.591 s is under the 0.82 s target. The G16
baseline is another tree and box (the farm found G16's keys identical to cb5d1b0's GRP graphs); the F0 row is a
same-box A/B on the exact parent tree.

Real-weight ppl at `--dp-attention 4`, so that the group path runs (`tools/check_ppl.py --model zai-org/GLM-5.3-Flash
--tp 32 --dp-attention 4 --piecewise --kv-cache-gb 1.0 [--text-file wikitext2_test.txt]`, the serving env plus
`KILN_MOE_PREFILL_MIN_TOKENS=1`, farm graphs q/pfgrp6-trn1 configs ppl4/wt-EPT-GRP-DP4 and -NOGRP-DP4; kiln-dk-32):

| tree | France | Water boils | def add | quick fox | mean | wikitext-2 | logs |
|---|---|---|---|---|---|---|---|
| 2b3bdbe, `KILN_SP_GROUP=0` | -2.210 | -3.595 | -0.947 | -1.715 | -2.108 | -0.550 | 20261005T005356Z-fin-ppl-n4, 005741Z-fin-ppl-nw |
| 2b3bdbe (group collectives on) | -2.210 | -3.595 | -0.947 | -1.715 | -2.108 | -0.547 | 20261005T003224Z-fin-ppl-p4, 003613Z-fin-ppl-w4 |

The group collectives leave the 4 sentences unchanged to three decimals on the same tree. The mean moved from
-2.091 on the earlier EP tree (e2c6fad, the table above) to -2.108 here with the group path on and off alike, so it
came with the engine-v0 merge (tile-scale EP and the rest of f9dc4c4); nearly all of it is Water boils (-3.518 -> -3.595), the sentence
with the known knife-edge ("The 4-sentence ppl has a knife-edge on Water boils" above), while the other three moved
by at most 0.006. wikitext with the group path, by chunk: -0.730 -0.692 -1.092 -0.927 -1.219 -0.172 -0.358 -0.138
-0.291 -0.270 -0.342 -0.334 (with it off: -0.783 -0.692 -1.096 -0.923 -1.210 -0.179 -0.358 -0.140 -0.279 -0.274 -0.342
-0.326). Both wikitext readings are within 0.01 of the -0.5515 reference.

The merged head graph for graph: the farm captured pfgrp6's G64-4096-KV1.5-S20-P12-K-EPT-GRP command verbatim on
engine-v0 b5e078d (the merge of feat/prefill-grp3 15b584c; ranks 0, 8, 16, 24) and got exactly q/pfgrp6-trn1's 24
keys, so the MTP merge left the serving graphs unchanged and every number above holds for engine-v0 b5e078d.

## Where the 4096-token prefill call goes after the group collectives (2026-10-05 UTC, SDK 2.32, trn1.32xlarge)

Both measurements on the measured tree 2b3bdbe (kiln/ identical to engine-v0 b5e078d), kiln-pf-32c, logs under
s3://<your-bucket>/logs/kiln-pf-32c/.

**The call, piece by piece** (the G64 EPT + group-collectives command of the section above, farm graphs
q/pfgrp6-trn1, with `KILN_PROFILE_PIECES=1 KILN_PROFILE_EXEC=1`, which times every graph synchronously: 108.2 out
tok/s instead of 123.1; log 20261005T013732Z-pieces.log, 264 prefill and 896 decode calls):

| graph | p50 ms | layers |
|---|---|---|
| piece 0 | 149.7 | 0-11: 3 dense KDA, 6 MoE KDA, 3 MoE DSA |
| piece 1 | 175.4 | 12-23: 9 MoE KDA, 3 MoE DSA |
| piece 2 | 172.1 | 24-35: the same kinds |
| piece 3 | 134.0 | 36-44: 7 MoE KDA, 2 MoE DSA |
| sum of pieces | 631.3 | |
| whole prefill call (exec, prep and post graphs included) | 640.8 | |
| decode call, 16 rows per group (pieces 55.5 / 53.8 / 53.3 / 41.8) | 213.2 | |

Unprofiled, the serving run's fit gives 0.591 s per prefill call (section above), so the synchronous profile
inflates the call by ~8%. The same method on engine-v0 8f41439 (TP, world collectives) gave 1002.1 ms of pieces
("Sequence-parallel block outputs as reduce-scatters" above): the call is 37% shorter since, with expert
parallelism, tile-scale EP and the group collectives among the changes.

**Block by block at the served shape** (4096 rows, 1024 per token-mixer group, 128 per rank; expert parallelism and
group collectives on: `KILN_CC_ARGS=--model-type=transformer KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki
KILN_LINEAR_ATTN_KERNEL=nki KILN_DSA_SELECT=nki KILN_DSA_POOL_CACHE=auto KILN_PROFILE_EXPERT_LAYOUT=loaded
KILN_MOE_PREFILL_SKIP=20 KILN_MOE_EP=1 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 python tools/profile_layer.py --model
zai-org/GLM-5.3-Flash --tp 32 --dp-attention 4 --ranks 32 --prefill 1024 --pages 264 --sum-readback --neff --layers 12
--what hcblocks --part-layers 3 4 --sp`, log 20261005T015559Z-blocks-l34.log; layer 0 from 20261005T015036Z-blocks.log,
which then stopped at the routing comparison: 2b3bdbe's profile_layer runs it for a dense layer too, fixed on
engine-v0 by 4d38ee0; p50 ms of one graph each, random input rows, real expert weights):

| graph | layer 0 (dense, KDA) | layer 3 (MoE, pooled DSA) | layer 4 (MoE, KDA) |
|---|---|---|---|
| SP attention block as served (group gather, mixer, group reduce-scatter, mHC) | 6.06 | 12.19 | 5.42 |
| the same block with world collectives (gather + world reduce-scatter) | 8.83 | 15.50 | 9.05 |
| SP FFN block as served (world gather of rows and routing, MLP, world reduce-scatter, mHC) | 6.60 | 14.42 | 14.27 |
| token mixer alone, with its all-reduce | 6.30 | 12.33 | 5.60 |
| MoE routed experts alone (router + EP kernel, rank 0's experts, no collective) | | 4.70 | 8.78 |
| router alone (4096 rows) / shared expert alone | | 1.22 / 0.49 | 1.14 / 0.48 |
| SP routing alone (128 rows routed, [4096, 16] fp32 gathered) | | 6.24 | 5.77 |
| SP world gather [128 -> 4096, 4096] / SP group gather [128 -> 1024, 4096] | | 6.60 / 3.35 | 4.85 / 3.87 |
| SP mHC collapse / output mix (128 rows) | | 0.39 / 0.29 | 0.52 / 0.56 |

DSA layer 3 at 1024 queries over 8448 keys: projections and indexer 1.09 ms, the pooled selection 3.72, the
attention core with a given mask 8.69 (expand) / 8.77 (absorb), mla.attention 10.73, o_proj 0.75.

**Composing them.** Every standalone graph that holds a collective pays a fixed cost per execution ("Collectives
across chips"); a piece pays it once. With one constant f per graph, piece = f + sum over its layers of (attention
block - f) + (FFN block - f): piece 1 = 9 x (5.42 + 14.27) + 3 x (12.19 + 14.42) - 23 f = 175.4 gives f = 3.55 ms,
and that f predicts piece 0 at 154.3 (measured 149.7) and piece 3 at 130.7 (134.0), and all four pieces at 636 ms
against 631.3 measured. So in the served call a layer costs about 5.6 ms (dense KDA), 12.6 ms (MoE KDA) and 19.5 ms
(MoE DSA), and the 4096-token call divides as:

| part | ms per call | share |
|---|---|---|
| MoE FFN blocks (42 layers x ~10.8: routing and its gather, the rows' world gather, router, EP kernel, shared expert, world reduce-scatter, mHC) | ~452 | **~71%** |
| pooled-DSA attention blocks (11 x ~8.6) | ~95 | ~15% |
| KDA attention blocks (34 x ~1.9-2.5) | ~66 | ~10% |
| dense FFN blocks (3 x ~3.1) | ~9 | ~1.5% |
| fixed cost per piece (4 x ~3.55) | ~14 | ~2% |

**What that makes visible as the next lever** (none of it measured as a change yet):
- The MoE FFN block is now ~71% of the call, and inside it the EP kernel is not the only cost. Standalone, rank 0's
  routed experts take 4.7 ms on layer 3 and 8.8 on layer 4 (how many pairs land on rank 0's 9 experts), and the
  block's output reduce-scatter waits for the busiest rank, so prefill EP load balance is one lever ("Expert
  groups" above is the negative result on splitting experts across ranks). The other is the block's three world
  collectives: the rows' gather is a zero-padded [4096, 4096] bf16 all-reduce (31/32 zeros; a ring all-reduce of the
  32 MB buffer sends ~62 MB per rank, an all-gather of the same rows ~31 MB; kept because a cached all-gather NEFF
  breaks in a later process), the routing's [4096, 16]
  fp32 gather beside it, and the reduce-scatter. The token mixers' version of this, an 8-rank group in place of
  the world, was worth 72 ms per call (F0: 0.665 -> 0.593 s); the FFN side cannot use the group, because the
  experts need every row.
- The pooled-DSA attention, ~15%: its attention core at 8448 keys (8.7 ms standalone per layer, 11 layers) is
  most of it.
- The KDA mixers are ~10% after the group collectives; collectives and fixed per-piece costs are no longer
  large single items.

## MTP speculative decoding on GLM-5.3-Flash at serving scale (2026-10-04, SDK 2.32, trn1.32xlarge, tp=32, DP attention 4)

`--spec-method mtp` with GLM-5.3-Flash's own MTP layer (layers.45: pooled DSA + MoE, no mHC; models/mtp.py), real
weights (zai-org/GLM-5.3-Flash@eb9eb208, FP8 experts, bf16 activations), feat/glm-mtp. Every graph from the compile
farm (q/mtp-ace16b1, q/mtp-82cc981, q/mtp-ep-4d38ee0, q/mtp-ep16-4d38ee0, q/mtp-pc-a8ede27; `NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`,
0 device compiles). Boxes kiln-mtp-trn1 and kiln-g1-trn1; logs under s3://<your-bucket>/logs/<box>/.

**What had to change before a number could be taken** (each with a CPU test that is red on engine-v0):
- Sequence-parallel prefill streams were forced off whenever an MTP head was built (`prefill_sp ... and not mtp`,
  because the MTP pass drafts from the chunk's final hidden states): every MTP run would have paid the ~20% the SP
  streams are worth at conc 64. Now the MTP graph after a prefill chunk takes this rank's rows of the chunk's last
  hidden state and gathers their final norms itself (models/decoder.py `_mtp_pass sp_onehot`, the gather
  post_prefill already does); the runner marks SP-local hidden states (`ModelRunner._hidden_sp`, the graph key gains
  "sp"). tests/test_linear_serving.py test_glm5_next_mtp_with_sequence_parallel_streams: tp=4 / DP 2 piecewise and
  tp=2, SP on, every step's drafts equal those with the streams replicated, output equal to plain greedy.
- A prefill chunk's MTP call took a decode bucket of rows (`pick_bucket(1, (16,))` = 16): 16 x 1024 rows per group
  for one chunk, in a graph warmup never built (warmup builds the (1, q, P) ones; a runtime compile is 10-16 min at
  tp=32). Now one row per group (`mtp_drafts prefill=True`). tests/test_mtp_mla.py
  test_warmup_covers_prefill_drafting_with_pinned_decode_buckets.
- A sequence without a draft (its last token, room 0) got a decode launch of its own beside the step's verify: with a
  pinned bucket that is a whole decode step (161 ms at B=16) for one row, almost every step of a closed loop. Now
  draftless rows ride in the verify graph (EngineConfig.spec_verify_plain, default for mtp) and the plain decode
  graphs are not built at all. test_draftless_rows_run_in_the_verify_graph.
- The verify's page bucket counted a short draft's padded rows, which pass max_model_len (8448 = the 264-page bucket)
  at a request's last tokens: "ValueError: 265 exceeds the largest bucket". Now the real positions; padded rows repeat
  the last real position and write the null page.
- Non-final prefill chunks' MTP passes (they only fill MTP KV) launch right behind their prefill, and every MTP graph
  of a step launches before the first read-back; the MTP graph's own hidden output is not kept (HIDDEN_KEEP 16 -> 8).

**The target's graphs do not move.** `pool_pages` does not count the MTP layer's KV (it is allocated beside the pool),
and the KDA state pool, a graph input of every group holding a linear layer, has 1 + max_num_seqs/group x (1 + k) +
checkpoint rows: `--state-checkpoints` (bench/serve_sweep.py, tools/check_ppl.py) holds it at the baseline's 49 rows
per group (k=1: 16 checkpoint rows, k=2: 0; the random-prompt sweep takes no checkpoint). Rank-0 captures on ace16b1:
MTP k=1 contains all 15 q/final-c33a391 G64 keys plus 11 new ones (verify prep / 3 groups / post, MTP graphs). Under
EP (4d38ee0) the farm compared every family: the MTP configs share all of the baseline's prefill groups and lack only
the plain decode groups and their prep / post graphs; EP-ppl4-MTP1 carries all 11 non-decode target keys of the same
tree's ppl4 capture.

**HBM.** neuron-monitor on the G64-4096-KV1.5 baseline (TP, 128 requests): the fullest core at 15.48 GiB of 16
(tensors 14.66 GB, model code 0.925 GB, shared scratchpad 0.94 GB, constants 0.09 GB). The MTP head adds 0.46 GiB of
tensors (tools/tensor_bytes.py rank 0: experts 10.01 -> 10.25 GB, KV 1.487 -> 1.622, weights 1.972 -> 2.082), and
both MTP k=1 and k=2 at KV 1.5 failed to load ("Could not load the model status=4 message=Allocation Failure" at the
first prefill group, also with the decode graphs dropped). KV 1.2 GB fp8 (4400 pages per group against the 4224 that
64 x 8448 tokens need) frees 0.28 GiB and costs nothing: the baseline at KV 1.2 runs 94.9 (TP) / 106.9 (EP) out tok/s,
the same as at 1.5; MTP k=1 then loads with 0 or 16 checkpoint rows. Under EP the k=2 and k=3 verify graphs at 16 rows
per group do not load even with no prefill graph beside them (tools/time_decode.py, Allocation Failure).

**Acceptance on real weights** (tools/check_mtp.py, G16 shapes: decode bucket 4 per group, KV 0.65 bf16, TP; 16
prompts x 256 greedy tokens, ignore_eos; logs kiln-g1-trn1 20261004T211049Z-mtp1-g16, 211733Z-mtp2-g16,
210136Z-check_mtp (k=3, --max-num-seqs 12); "tokens per verify" = 1 + accepted drafts per drafted verify, position
acceptance = given the earlier drafts were accepted):

| prompts | k=1: accepted, tokens per verify | k=2: accepted, tokens per verify, by position | k=3: accepted, tokens per verify, by position |
|---|---|---|---|
| benchmark (serve_sweep's random-token 8192-token prompts) | 98.2%, 1.982 | 96.5%, 2.926, 0.976 / 0.977 | 80.6%, 3.413, 0.973 / 0.970 / 0.530 |
| wikitext-2 test slices of 2048 tokens (md5 3ce70e93), continued | 74.7%, 1.747 | 60.1%, 2.201, 0.745 / 0.613 | 48.0%, 2.435, 0.744 / 0.605 / 0.545 |
| chat (16 questions through the chat template) | 88.1%, 1.881 | 76.1%, 2.519, 0.863 / 0.761 | 63.9%, 2.907, 0.859 / 0.744 / 0.653 |

In the serving sweeps (benchmark prompts, 128 requests): k=1 at conc 64 95.8% (TP) / 96.7% (EP) = 1.96-1.97 tokens per
verify; k=2 at conc 16 92.2% (TP) / 91.2% (EP) = 2.82-2.84; k=2 at conc 32 (EP) 93.2% = 2.86. The random prompts make
the model repeat itself, so the benchmark's acceptance is higher than real text's.

**Greedy equality.** Against the same G16 engine without MTP (the plain decode graphs), over 256 tokens: k=1 identical
on 10/16 random prompts, 0/16 wikitext, 0/16 chat; k=2 9 / 0 / 1; k=3 9 / 0 / 0. The divergences are the same tokens
at every k (wikitext: tokens 24, 62, 19, 38, 18, 9, ...), each at a reference top-2 logprob margin of 0.0 to 0.375
(1-3 bf16 ulps of the logits), and on the matched prefixes the chosen-token logprobs differ by mean 0.006 / 0.019 /
0.011 (p50 0.0005 / 0.0017 / 0.0000, p99 0.13 / 0.18 / 0.14; random / wikitext / chat, k=1). Controls: the reference
engine run twice gives 16/16 identical on every set (the device is deterministic), and the same engine without MTP at
the G64 shapes (decode bucket 16, fp8 KV) against G16 diverges just as much: 9/16 / 0/16 / 0/16 identical, first
divergence median 21 (wikitext) / 36 (chat) tokens, matched-prefix |dlogprob| mean 0.010 / 0.031 / 0.014. So MTP's
outputs differ from plain decoding only as a verify graph's shapes (2-4 rows per sequence) round differently from a
decode graph's, less than the engine's own cross-shape noise; on CPU in fp32 they are identical (tests/test_mtp*.py,
test_linear_serving.py, test_mtp_sweep.py).

**Step costs** (tools/time_decode.py, `KILN_PROFILE_EXEC=1`, the sweep's engine, random ids, null-page KV, P=264; p50
of 32-48 steps; logs kiln-mtp-trn1 20261004T201449Z / 201954Z-time_decode, 211937Z-ep-td-k1; kiln-g1-trn1
223719Z-td-g16-k1, 230905Z-td-g16-k2b, 231450Z-td-g64kv12-k1b):

| shape | decode | verify k=1 | verify k=2 | MTP draft after a verify (k=1 / k=2) | MTP pass after a prefill step (4 x 1024 rows) |
|---|---|---|---|---|---|
| G64, B=16 per group, TP | 161.3 ms | 220.7 (1.37x) | 645.4 (4.0x) | 15.3 / 23.4 | 51.0 (k=1) |
| G64, B=16 per group, EP | 186.7 | 247.1 (1.32x) | does not load | 16.0 / - | |
| G16, B=4 per group, TP | 80.4 | 114.8 (1.43x) | 138.9 (1.74x) | 11.6 / 15.2 | 57.0 (k=2) |

The TP k=2 verify at 16 rows per group (192 rows into the MoE) is a compiler pathology, not arithmetic: its 12-layer
groups carry 2.40M instructions (DVE 1.25M) against 0.87M (DVE 0.13M) at k=1 (tools/neff_instructions.py, graphs
b11fa6f2 vs cf95551a), and one KDA + MoE layer as a graph is 14.3 ms against 8.3 ms for its two blocks timed apart
(tools/profile_layer.py --verify 3); at 4 rows per group the k=2 groups are 0.52-0.58M instructions, normal.

**What the verify adds, per layer** (`KILN_MOE_EP=0 python tools/profile_layer.py --model zai-org/GLM-5.3-Flash --tp 32
--dp-attention 4 --ranks 2 --batch 16 --pages 264 --sum-readback --layers 5 --what hcblocks layers --part-layers 0 3 4
--scratch-rows 1024 [--verify 2]`, kiln-mtp-trn1 cores 0-1, the sweep's env; p50): a KDA layer's attention block
1.87-1.97 -> 3.01 ms at k=1 (the KDA mixer alone 1.19-1.29 -> 1.46: the rest is the block's fusion at twice the rows),
an MoE block 1.58 -> 2.02 ms (TP), a pooled-DSA attention block 6.50 -> 4.67 ms (cheaper: the cache loads, 5.06 ms,
are per sequence, not per query). Over 34 KDA, 42 MoE and 11 DSA layers that is ~+37, ~+18 and ~-20 ms: the KDA
layers' blocks are what the 59 ms of a k=1 verify over a decode step is.

**Serving A/B** (bench/serve_sweep.py, 8192 in / 256 out, random prompts, 128 requests per level, `--overlap` given
(an MTP engine steps synchronously), same box per pair, spot $2.15/h):

| conc | engine | config | without MTP | with MTP | tokens per verify (accepted) | |
|---|---|---|---|---|---|---|
| 64 | TP (82cc981) | G64-4096, KV 1.2 fp8 (MTP: 0 checkpoint rows) | 94.9 (kiln-g1-trn1 20261004T214241Z; 95.0 at KV 1.5 on kiln-mtp-trn1 190050Z) | k=1 **88.2** (g1 213042Z) | 1.958 (95.8%) | -7.1% |
| 64 | EP (4d38ee0) | the same | 106.9 (kiln-mtp-trn1 220105Z; 107.0 at KV 1.5, 214919Z) | k=1 **102.0** (213725Z) | 1.967 (96.7%) | -4.6% |
| 32 | EP | F0-4096, KV 1.5 bf16 | 99.1 (kiln-mtp-trn1 20261005T002601Z-ep-sweep-f0-base2) | k=1 **88.6** (8 checkpoint rows, 234248Z); k=2 **76.2** (224451Z) | 1.962 (96.2%); 2.860 (93.2%) | -10.6% / -23.1% |
| 16 | TP | G16-4096, KV 0.65 bf16 | 82.8 (g1 215526Z) | k=2 **51.8** (g1 220702Z) | 2.838 (92.2%) | -37% |
| 16 | EP (`KILN_MOE_EP=1`; the default keeps TP at 4 rows per group) | the same | 80.5 (kiln-mtp-trn1 223207Z) | k=2 **52.9** (221519Z); k=3 **61.8** (232746Z) | 2.819 (91.2%); 3.455 (82.2%) | -34% / -23% |

Spot $/1M out at $2.15/h for those: TP conc 64 $6.29 -> $6.77, EP conc 64 $5.59 -> $5.86, EP conc 32 $6.03 -> $6.74 (k=1), TP
conc 16 $7.21 -> $11.53. TTFT p50 rises with MTP everywhere (conc 64 TP 10.1 -> 20.1 s, EP 8.3 -> 16.8 s) and ITL p50
falls only where decode dominates (EP conc 64 526 -> 501 ms; TP conc 16 157 -> 270 ms, the prefill steps below).

**Why it loses here.** The workload is 32 prompt tokens per output token, so every level is prefill-bound: a prefill
step is ~1.0 s of a 4096-row chunk, and the decode tokens of the running sequences ride on it. MTP makes the
decode-only steps 1.9x (G16 k=2) to 1.4x (G64 k=1) cheaper per token, but every step that carries a prefill also pays
the verify's extra rows (+34 to +60 ms), the 4096-row MTP pass (+51-57 ms), and the overlap the sync step loses (the
TP baseline without `--overlap`: 92.4 against 94.9 out tok/s at conc 64, -2.7%). Served step profile of G16 k=2
(`KILN_PROFILE_STEP=1`, 48 requests, log kiln-g1-trn1 20261004T224630Z-sweep-g16-mtp2-prof): 309 verify-only steps of
157 ms (device 128, draft 17, launch 12), 120 prefill + verify steps of 1177 ms, 16 prefill-only steps of 1026 ms.
The same G16 engine without MTP, synchronous too (no `--overlap`; 48 requests, log 20261004T232019Z-sweep-g16-base-sync-
prof): 972 decode-only steps of ~80 ms, 88 prefill + decode steps of ~1057 ms, 16 prefill-only: 77.3 against 66.6 out
tok/s with MTP. So MTP spent 48.5 s on decode-only steps where the baseline spent ~78 s, and lost more than that on
prefill: 136 prefill steps against 104 for the same 384 chunks, each ~11% longer. Under DP attention a prefill step
costs the same whether one group or all four have a chunk, and MTP's faster turnover scatters the admissions across
groups, so fewer chunks share a step (TTFT p90 27.0 against 24.9 s). Neither is a recompile or a reload: every
graph came from the cache and the device step times match tools/time_decode.py.

**Where it should pay: decode-heavy traffic.** The G1b workload, conc 64 with 75% of every prompt one of 4 shared 6144-token
prefixes (`--shared-prefix-len 6144 6144 --num-prefixes 4 --keep-cache`, 256 requests per level, the cold level then
the warm one), EP on a8ede27 (engine-v0 34b0d5d merged), KV 1.2 fp8, MTP k=1 with 16 checkpoint rows (4 used per group),
farm q/mtp-pc-a8ede27, kiln-g1-trn1 logs 20261004T232915Z-pc-ep-mtp1-c16 and 234711Z-pc-ep-base:

| level | without MTP | MTP k=1 | tokens per verify | spot $/1M out | serve_sweep's device split (prefill call / decode call / per step) |
|---|---|---|---|---|---|
| cold (hit rate as the prompt-cache notes) | 186.7 out tok/s, TTFT p50 2.8 s, ITL 318 ms | 181.3, 5.3 s, 307 ms | 1.964 (96.4%) | $3.20 -> $3.29 | 0.759 / 0.182 / 0.00 s against 0.731 / 0.197 / 0.10 s |
| warm (every request a hit) | **229.6**, 2.8 s, 264 ms | **223.1** (-2.8%), 3.8 s, 246 ms | 1.967 (96.7%) | $2.60 -> $2.68 | 0.747 / 0.176 / 0.00 s over 1061 steps against 0.758 / 0.089 / 0.20 s over 612 |

Here decode is two thirds of the baseline's device time: its wall, 285.4 s, is 1061 decode calls x 0.176 s = 187 s
plus ~132 prefill calls x 0.747 s (512 chunks of 1024 make at least 128). With MTP the same tokens take 612 verify
steps x (0.247 + 0.016 s draft, time_decode's EP costs) = 161 s, and 128 prefill calls with their 51 ms MTP pass are
102 s: ~263 s of device time against ~283 s (-7%). The measured wall, 293.7 s, is ~31 s above that. The gap is the
synchronous MTP step's host time (scheduling, argument building, the launches to 31 ranks and the read-backs that
`--overlap` hides for the baseline; at G16 the step profile shows launch 11.7 ms and draft read-back ~5 ms per
verify-only step beside the device wait) plus whatever prefill calls scattered admissions add (the G16 effect above);
this run carried no step profile to split the two.

**So MTP is a win waiting on asynchronous drafting**, and only where decode is most of the device time. What would
turn it: (1) the accept walk and the next draft on the device (the verify's accepted count gathers the hidden rows
and positions inside the MTP graph, the drafts land on the token board), so a step's verify and draft launch back to
back and the next step can be scheduled while they run, as the baseline's overlap does; at G1b warm that recovers at
most the ~31 s: 293.7 -> ~263 s, ~249 out tok/s, ~9% above the baseline's 229.6 (an upper bound: part of the gap may be
prefill calls, and the overlap hides host time only behind device time). (2) A cheaper verify: at k=1 the KDA layers' attention blocks are most of
its extra 59-60 ms over a decode step. (3) The 4096-row MTP pass after each prefill chunk (51-57 ms, ~5% of a prefill
call) as part of the last prefill layer group instead of a graph of its own. On the G1 sweep itself MTP stays off:
every level is prefill-bound there.

**Correctness of the target with the head loaded** (tools/check_ppl.py, EP, the q/ep-trn1 ppl configs plus
`--spec-method mtp --spec-k 1 --state-checkpoints 4`; kiln-mtp-trn1): the 4 sentences -2.147 / -3.527 / -0.956 / -1.714, mean
-2.074, with the head loaded and without it, every sentence equal (logs 20261005T000300Z-ep-ppl4-mtp1, 000827Z-ep-ppl4-base);
the wikitext-2 slice (3071 tokens) -0.551 both, by chunk -0.770 -0.693 -1.101 -0.929 -1.206 -0.183 -0.355 -0.137 -0.298
-0.275 -0.334 -0.333 both (001331Z-ep-pplwiki-mtp1, 001851Z-ep-pplwiki-base), the values EP gave before. On the merged
tree (260ee41, engine-v0 f9dc4c4 in it) the G16 k=1 check (kiln-g1-trn1 log 20261005T001023Z-fin-mtp1-g16) reproduces the
earlier run token for token on all 48 prompts. Suite (kiln-lead-ci3, CPU, one pytest process,
`KILN_TEST_MODEL=Qwen/Qwen3-0.6B`): 666 passed, 48 skipped on 3cc4e7c; the transformers-5.18 files (glm5_next, qwen4_exp,
linear_serving, dsa_topk, dsa_select, moe_ep, dp_attention, mtp_mla, mla, mixed_batch, mtp_sweep) 283 passed, 12 skipped
on 260ee41.

**Commands.** Graphs: q/mtp-82cc981 (TP: BASE-G64-KV1.2, MTP1-G64-KV1.2, MTP2-G64-KV1.2, MTP1/2-F0, MTP1/2-G16,
MTP3-G16-12), q/mtp-ep-4d38ee0 and q/mtp-ep16-4d38ee0 (EP: the EP-TD-* decode sets, EP-BASE / EP-MTP1 / EP-MTP2 at G64
KV 1.2, G16 and F0, EP-ppl4/pplwiki-MTP1), q/mtp-pc-a8ede27 (EP-BASE-G64-KV1.2, EP-MTP1-G64-KV1.2-C16); each
configs/<name>.config.json holds the exact command. Pull a config's graphs with `python tools/cache_sync.py pull-keys
s3://<your-bucket>/compile-cache/trn1-sdk2.32/lnl/ <queue>/configs/<name>.keys.json` (a queue lists only the
keys it compiled), then run that command with `NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`. Acceptance and equality:
`python tools/check_mtp.py --sets random,wikitext,chat --n 16 --gen 256 --out-json ref.json -- <sweep args>`, then the
same with `--reference-json ref.json` and `<sweep args> --spec-method mtp --spec-k K --state-checkpoints C`. Step
costs: `KILN_PROFILE_EXEC=1 python tools/time_decode.py --steps 32 --skip 8 [--skip-decode] -- <sweep args>`. Served
step phases: `KILN_PROFILE_STEP=1` on bench/serve_sweep.py (synchronous steps only).
## trn2: sequence-parallel streams pass wikitext, and every NKI kernel split over the two cores of an LNC=2 logical core (2026-10-04, SDK 2.32, trn2.48xlarge spot)

kiln-trn2-b (trn2.48xlarge spot, us-east-2c, $15.0887/h at 14:00 UTC), GLM-5.3-Flash@eb9eb208 on the instance-store
RAID0, LNC=2 (64 logical cores of 24 GiB).

**Sequence-parallel streams are numerically clean on trn2; the 4-sentence move was the knife-edge.**
`KILN_CC_ARGS=--model-type=transformer python tools/probe_mhc_rounding.py --rows 1 2 8 32 64 256 1024 --chain 4` on
one logical core (the engine's compiler arguments, log /opt/kiln/logs/20261004T162229Z-probe_mhc_rounding.log): the
collapse's Veltkamp rounding residual is bit-identical to the host's at every row count (max |diff| 0, zero residuals
0.00-0.02% on both), "rounded" everywhere, as on trn1. Then the wikitext-2 slice on engine-v0 21e9c8c, both halves of the
box at once (`KILN_CC_ARGS=--model-type=transformer KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki
KILN_MOE_PREFILL_MIN_TOKENS=1 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=<1|0> python
tools/check_ppl.py --model zai-org/GLM-5.3-Flash --tp 32 [--core-base 32] --piecewise --kv-cache-gb 1.0 --text-file
wikitext2_test.txt --out-json ...`, logs 20261004T162940Z-pplwiki-sp1 / 20261004T162949Z-pplwiki-sp0; 21 min of weight
load with two 32-rank loads sharing the 192 vCPUs, then ~20 min of compiles on the box):

| KILN_PREFILL_SP | wikitext mean | by 256-token chunk |
|---|---|---|
| 1 | **-0.549** | -0.779 -0.686 -1.105 -0.925 -1.213 -0.179 -0.352 -0.136 -0.297 -0.268 -0.328 -0.319 |
| 0 | -0.552 | -0.783 -0.675 -1.104 -0.933 -1.219 -0.176 -0.355 -0.138 -0.302 -0.273 -0.340 -0.326 |

`tools/compare_ppl.py`: difference +0.0028, |dlogprob| mean 0.058 (max 3.89), greedy agreement 97.4%, the spread of
every accepted device-vs-device pair (trn1: 97.3-98.0%). Both are within 0.01 of -0.551. So KILN_PREFILL_SP is on by
default on trn2 too (models/decoder.py PREFILL_SP_FAMILIES); the -1.807 of the sp-merge tree was "Water boils" alone.

**No Kiln NKI kernel used the second physical core.** At LNC=2 a logical core is two physical cores and a kernel launched
with grid 2 runs on both (platform.nki_grid); NKI traces it once per program ("kernel is traced LNC times with
different program_id_value", nki/_backends/mlir_tracer/__init__.py program_id), and a kernel that ignores
nl.program_id does all of its work twice. Every Kiln kernel did (delta_rule even ran at grid 1). The engine-v0 MoE
prefill kernel took 14.70 ms at C=4096 on a trn2 logical core against 13.13 ms on one trn1 core. Each kernel now splits
its work by program (npg / pid are Python ints; grid 1 traces exactly the old instruction stream):

- kernels/moe_prefill.py: the lane tiles that always run in two halves, skip segments alternating, Y (slot order) in
  shared HBM, `nisa.core_barrier(data=Y, cores=(0, 1))` ("two NeuronCores both need to write to disjoint portions of a
  shared HBM tensor ... and they both need to consume the tensor after both cores have finished", nki/isa/_lnc.py),
  then the combine's token tiles in two halves. Both programs compute the plan.
- kernels/moe_dedupe.py (decode): the blocks of lanes in two contiguous halves (each program's first ring slots load
  expert 0 where padded, as `keep` does for the first ones), the fp32 partial sums exchanged by halves of H with
  `nisa.sendrecv` (SBUF to SBUF between the two cores), each program adding the other's half and writing its output
  columns. It also gets the source `rev` static argument moe_prefill carries.
- kernels/delta_rule.py (KDA / GDN prefill): each program its own v heads, at platform.nki_grid(). (The NKI tracer takes
  only str keys in a dict: the state tiles are a list; nki.simulate had accepted the dict.)
- kernels/dsa_topk.py (selection, fused scores + selection): groups of row tiles alternate (rows are independent).

Per kernel, one logical core, GLM-5.3-Flash's tp=32 rank shapes (`python tools/probe_lnc_split.py --out <pt>` on each
tree, then `--compare`; random experts in the loaded layout, uniform top-8 routing; logs on kiln-trn2-b
20261004T171039Z-lnc-base-trn2, 20261004T172045Z-lnc-split3-trn2, 20261004T171720Z-lnc-split2-trn2):

| kernel, shape | engine-v0 (both cores the same work) | split | speed-up | outputs |
|---|---|---|---|---|
| MoE prefill C=1024 | 6.702 ms | 3.701 ms | 1.81x | bit-identical |
| MoE prefill C=4096 | 14.695 | 8.118 | 1.81x | bit-identical |
| MoE decode (dedupe) T=16 / 32 / 64 / 128 | 0.574 / 1.209 / 1.526 / 2.049 | 0.385-0.394 / 0.611-0.879 / 1.074-1.077 / 1.038-1.307 | 1.4-2.0x | bit-identical |
| KDA delta rule C=1024 / 4096, 8 heads | 1.592 / 5.235 | 0.877 / 2.699 | 1.81x / 1.94x | bit-identical |
| DSA select R=8 / 32 / 1024 (pooled, keep 512 of 2112) | 0.244 / 0.252 / 1.477 | 0.255 / 0.250 / 0.883 | 1.0 / 1.0 / 1.67x | bit-identical |
| DSA scores + select C=1024 | 2.305 | 1.396 | 1.65x | bit-identical |

(The dedupe split's fp32 sum of two partials is a different order from the sequential one; on these inputs no bf16
output changed. Decode-sized DSA calls are one row tile and do not split.) In the NKI simulator
(`nki.simulate(kernel[2])`, target trn2; tests/test_lnc_split.py, 4 passed on a trn1.2xlarge's CPU) grid 2 equals grid 1
bit for bit for moe_prefill, delta_rule and both DSA kernels, and the dedupe kernel stays within one bf16 step of grid 1,
deterministically.

trn1 is unchanged (kiln-t2-t1, trn1.2xlarge, the same probe on both trees): all 12 cases bit-identical at the same
times (MoE prefill C=4096 13.13 vs 13.38 ms, KDA C=4096 4.288 vs 4.290, ...), and `tools/neff_stream_cmp.py` (the two
trees' cache entries paired with the kernels' `rev` normalised) finds 24 of 24 kernel graphs with byte-identical
instruction streams on every engine. Only the cache keys change (each kernel's `rev`), so trn1 graphs recompile once.

**The MoE prefill split fails the real-weight check; the KDA and DSA splits are exact (found the same evening).** The
wikitext check (`tools/check_ppl.py --tp 32 --piecewise --kv-cache-gb 1.0 --text-file wikitext2_test.txt`, dp_attention
1, 256-token chunks: 8 stream rows per rank, MoE calls of 256 rows) on the split tree died at its first MoE layer group,
every rank: `scatter/gather (indirect memory copy via vector DGE) out-of-bound access`, `nrta status=1006`. Bisect, all on
kiln-trn2-b with real weights, farm graphs unless named:

| tree / setting | wikitext |
|---|---|
| engine-v0 b4f400f (no split) | -0.552 (by chunk -0.775 -0.696 -1.107 -0.926 -1.217 -0.175 -0.353 -0.138 -0.293 -0.265 -0.338 -0.343) |
| split 6870309, KILN_PREFILL_SP=1 / 1 with KILN_SP_RS=0 / SP=0 | out-of-bound, all three |
| split e21e904 (+ a core barrier on every split output before the kernel ends) | out-of-bound |
| 193b9ac `KILN_LNC_SPLIT=moe_prefill` / the same compiled on the box with a fresh cache, no farm | out-of-bound / out-of-bound |
| 193b9ac `KILN_LNC_SPLIT=moe_prefill`, one MoE layer per prefill graph (box compiled) | out-of-bound in a later graph (one MoE + one KDA layer), the first MoE layer's graph ran |
| 193b9ac `KILN_LNC_SPLIT=delta_rule` / `=dsa_topk` | **-0.552 / -0.552, every chunk identical to b4f400f** |
| a0a2060 (merged with engine-v0 e2c6fad, default `KILN_LNC_SPLIT=delta_rule,dsa_topk`) | **-0.552, every chunk identical** |

What the MoE split does NOT reproduce in isolation (tools/probe_lnc_split.py, one logical core, base engine-v0 against the
split, every case BIT-IDENTICAL and no error): uniform, skewed and hot routing (all tokens on 8 or 16 experts) at C = 32 ...
4096 rows (B = 64 and 128), 12 chained calls in one graph, kernel output read across rows by XLA ops between calls
(`--race`), and the wikitext prompt's own layer-3 input and routing (`tools/dump_moe_inputs.py`, Kiln on the host CPU,
real checkpoint truncated to 4 layers) through the real layer-3 experts of rank 0 (`tools/check_moe_prefill_layout.py
--save`). A 32-rank layer graph at the check's shape (`tools/profile_layer.py --tp 32 --ranks 32 --dp-attention 1 --prefill
256 --pages 104 --layer-groups 3-5 / 3-14 --sp`, random weights, box compiled) runs too. So the barrier is not the fix
(the stale-row hypothesis is refuted twice), the farm is not the cause, and what triggers it is a property of the real
served graph that none of these has; it stays open. Until it is found the split ships only where it is proven:
`KILN_LNC_SPLIT` (kiln/platform.py) defaults to `delta_rule,dsa_topk`; `all` or a list turns the others on.

**The hunt, step by step (2026-10-04 20:00 - 2026-10-05, kiln-trn2-b; logs under /opt/kiln/logs and
s3://<your-bucket>/logs/kiln-trn2-b/).**
1. Real data through the kernel alone: `tools/dump_moe_inputs.py --model zai-org/GLM-5.3-Flash --layers 4 --tokens 256`
   (log 20261004T205737Z-dump-moe: 272 experts used, max 30 pairs on one expert, x finite, |x| max 2.2) and
   `tools/check_moe_prefill_layout.py --layers 3 --rank 0 --save experts-l3-r0.pt`, then `KILN_LNC_SPLIT=all
   tools/probe_lnc_split.py --kernels prefill --prefill-chunks 256 --experts-file experts-l3-r0.pt --replay
   moe-inputs-wiki0.pt` on engine-v0 b4f400f and on the split: BIT-IDENTICAL (2.65 vs 4.74 ms; logs
   20261004T213718Z / 214207Z-lncreal-*).
2. 32-rank layer graphs at the failing shape, random weights in the loaded layout, box compiled: `KILN_LNC_SPLIT=moe_prefill
   tools/profile_layer.py --model zai-org/GLM-5.3-Flash --tp 32 --dp-attention 1 --ranks 32 --prefill 256 --pages 104
   --sum-readback --layers 15 --what layers --part-layers 3 --layer-groups 3-14 --sp` (and 3-5): every graph runs (logs
   20261004T2106*Z-repro-*, 211228Z-repro12-*).
3. The failing check with the box compiling every graph itself (fresh `NEURON_LIBTORCH_CACHE_ROOT`, no farm): fails at the
   same key (20261004T212102Z-pplwiki-PVm-box). Not the farm.
4. One MoE layer per prefill graph (`KILN_PIECEWISE_PREFILL_MOE_GROUP=1`, box compiled): the first MoE layer's graph runs,
   a later graph holding one MoE and one KDA layer fails (20261004T212804Z-pplwiki-PVm-g1).
5. Every piece synchronised (`KILN_PROFILE_PIECES=1`, which reads each piece's output back before the next launches):
   still fails at the first MoE group (20261005T000425Z-pplwiki-PVm-sync). Not an asynchrony between graphs.
6. Input side: both programs barrier on x / topi / wts at the kernel's start (`KILN_MOE_PREFILL_INBAR=1`, q/t2max-PIB):
   still fails (20261005T003048Z-pplwiki-PIB). Not a race on the inputs either.
7. The device's own values, piece by piece: `KILN_DUMP_PIECES` (model_runner, host side, keys unchanged) on the
   one-MoE-layer-per-graph check (q/t2max-PG1m split on, q/t2max-PG10 split off; logs 20261005T003053Z-pplwiki-PG1m,
   003921Z-pplwiki-PG10). The split run raises the out-of-bound notification and then crashes, but before it does, all
   45 pieces of the first 256-token chunk ran, and **every piece's output on every one of the 32 ranks is bit-identical
   to the unsplit run** (8 x 16384 stream rows per rank, no non-finite value). So the split computes the right values in
   the real graph; what is out of bounds is a transfer that does not feed the result (the kernel's indirect copies are
   the x-row gathers with `oob_mode=skip` for empty lanes, the token-slot scatter and the Y gathers of the combine, Y
   being shared HBM only in the split), and the runtime treats the notification as fatal.
8. Placement: the 12 chained calls of the probe with 8 / 16 GiB of device memory held below them (`--ballast-gb`, logs
   20261005T00471*Z-lncballast*): clean.
9. The x-row gathers' `oob_mode=skip` for empty lanes: with empty lanes gathering row 0 instead
   (`KILN_MOE_PREFILL_NOSKIPX=1`, q/t2max-PNX) the check still fails (20261005T010832Z-pplwiki-PNX). Not those gathers.
10. Where: with `NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_OUTPUT_DIR=...` (log 20261005T011933Z-pplwiki-PG1m-inspect)
    the notification names the instruction: `model ... graph_bff7b5c5... .neff ... neff instruction index = 161` (the
    one-MoE-layer graph of a KDA + MoE layer; without inspect the index is "unknown"). The message does not name the
    engine. That NEFF has two subgraphs, sg00 and sg01 (the two programs), with 2999 / 2947 GpSimd (Pool) instructions,
    161 / 102 on SP and thousands on the vector, scalar and tensor engines, so 161 is an early instruction of whichever
    engine issued the copy ("vector DGE" may name the vector engine's descriptor generation): early in the graph, likely
    before its MoE kernel, where XLA's own gathers and scatters (state and KV pages, the sequence-parallel row exchange)
    issue indirect copies. Not decoded further yet (neuron-explorer view needs a captured profile of the NEFF).
11. The split's core barriers run on GpSimd by default; `KILN_MOE_PREFILL_BARENG=1` / `2` moves them to the sync /
    vector engine (both compile and match the emulation; graphs in q/t2max-PBS / -PBV, trn2 cache). Neither fixes it: both wikitext runs
    (`tools/check_ppl.py`, P1M's env plus `KILN_LNC_SPLIT=moe_prefill KILN_MOE_PREFILL_BARENG=1|2`, kiln-trn2-b, logs
    20261005T013400Z-pplwiki-PBS / 013403Z-pplwiki-PBV) died at 01:52 UTC on the same "scatter/gather (indirect memory
    copy via vector DGE) out-of-bound access", nrta 1006, this time with `neff instruction index = unknown`.

**State at the stop (2026-10-05 02:00 UTC) and the next step.** Proven: the MoE prefill split is bit-identical to the
unsplit kernel in every isolated form (shapes, routings, real experts and the prompt's own layer-3 inputs, chained calls,
XLA consumers, device memory held below it), and inside the real served graph it computes bit-identical outputs for all
45 pieces on all 32 ranks; what fails is one indirect copy that the runtime flags out of bound (`neff instruction index =
161` of the one-layer KDA + MoE graph under inspect) and treats as fatal. Ruled out: the farm, asynchrony between graphs,
inputs written on the other core, the empty-lane `oob_mode=skip` gathers, the end-of-kernel barriers, Y's placement,
the engine the barriers run on. Next, in this order: (a) decode instruction 161: `neuron-explorer capture -n <graph_bff7b5c5...neff>`
on one rank with the dumped piece inputs (`KILN_DUMP_PIECES`), then `neuron-explorer view` to name the engine and the DMA
queue; (b) if it is an XLA gather after a split kernel, compare that queue's descriptors in the split and unsplit NEFFs
(the split programs issue different DMA counts per core on the kernel's queues, which a shared descriptor ring would not
expect). The split stays off by default until the wikitext check passes; turned on (`KILN_LNC_SPLIT=moe_prefill`) it is
worth +19-28% end to end on trn2 (sweeps U1S / U1M, throughput only).

The MoE prefill split also had two LNC=2 rules to learn on the way, both fixed and kept: the two programs must have the
same basic blocks (with `KILN_MOE_PREFILL_SKIP=20` and the skip segments alternating between programs, 4 graphs failed
`[NCC_IXGM002] Expected function sg000N ... to have X basic blocks, but on core 1 it has Y`), and the same device-loop trip
counts (per-half trip counts hung the device: execution timeout with GpSimd waiting). Every program now runs every segment
on its half with the segment's own trip count (C = 1024 / 4096 with SKIP=20: 4.89 / 9.31 -> 2.84 / 5.50 ms, bit-identical).

**What ships: the KDA and DSA splits, verified against engine-v0 on trn2 and trn1** (feat/trn2-max 7138a25 = engine-v0
e2c6fad merged + the split with `KILN_LNC_SPLIT` defaulting to `delta_rule,dsa_topk`; kiln-trn2-b, farm graphs):

| check | result |
|---|---|
| wikitext-2 slice, tp=32 dp1 chunk 256 SP (q/t2max-FP) vs engine-v0 b4f400f | -0.552 vs -0.552; `tools/compare_ppl.py`: mean abs dlogprob 0.0000, max 0.000, greedy agreement 1.0000 |
| 4 sentences (q/t2max-FP4) vs engine-v0 e2c6fad (q/t2max-EP4) | -2.129 / -2.516 / -0.960 / -1.705 = -1.838 on both, per token identical (the trn2 Water-boils value is engine-v0's own) |
| greedy at the serving config (`tools/greedy_ab.py`: 64 wikitext prompts of 256..8192 tokens at once, 64 greedy tokens each; one engine: DP attention 4, prefill 4096 / 1024, decode buckets 4/8/16/32, KV 2.75 GB fp8; `NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`) vs e2c6fad | **64 of 64 prompts identical, 4096 of 4096 tokens** (e2c6fad against a second run of itself: 64 of 64) |
| the same with the decode split too (`KILN_LNC_SPLIT=delta_rule,dsa_topk,moe_dedupe`) | 57 of 64; 7 long prompts (3909..7940 tokens) part at generated tokens 18..57: the dedupe split's two-partial fp32 sum flips late near-ties, so it stays out of the default |
| trn1 (trn1.2xlarge): probe_lnc_split, final tree vs e2c6fad | 12 of 12 cases bit-identical; `tools/neff_stream_cmp.py` 24 of 24 kernel graphs with identical instruction streams |
| single-process CPU suite on 7138a25 (m7i.4xlarge) | 663 passed, 40 skipped, 0 failed |

Whole box, same box and session, conc 64 / 128 (the configuration above, farm graphs, `NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`,
2 x concurrency requests): engine-v0 e2c6fad 171.8 / 185.5 out tok/s (log 20261004T233801Z-sweep-EU); 7138a25 **175.5 /
189.9** (+2.2 / +2.4%; log 20261004T230236Z-sweep-FUd), TTFT p50 10.2 / 10.6 s, ITL p50 319 / 609 ms, $23.88 / $22.07 per 1M
out at trn2 spot. Against the invalid trees with the MoE prefill split (204.5 / 220.6 on 21e9c8c, 221.5 / 240.9 merged with
b4f400f): the MoE prefill split is most of the +19-28%, so its defect is the trn2 lever left to unlock.

A side finding about the farm: a device run with `KILN_COMPILE_FARM=q/t2max-EU/` compiled 5 decode groups on the box
although all 5 keys were in that queue's capture (configs/EU.keys.json) and complete in the trn2 S3 cache: FarmWait
answers "not-farmed" for a key the queue does not list, and a queue whose keys were all "already cached" at enqueue lists
none of them. Reported to the compile farm (fetch from the queue's cache before "not-farmed"; list every captured key).

**End to end, whole box** (kiln-trn2-b, two tp=32 engines on cores 0-31 / 32-63, every graph from the compile farm,
one serve_sweep per tree over all five levels, 2 x concurrency requests per level, trn2.48xlarge spot $15.09/h:
`KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki
KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=0
KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=6 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_COMPILE_FARM=<queue>
python bench/serve_sweep.py --model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp 2 --dp-attention 4 --piecewise
--overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096
--prefill-buckets 1024 --max-num-seqs 128 --concurrency 16 32 64 128 256 --decode-buckets 4,8,16,32 --kv-cache-gb 2.75
--kv-cache-dtype fp8 --price trn2.48xlarge-spot=15.09`; one KV size and max-num-seqs for every level, so the levels share
their graphs; queues q/t2max-U1 (engine-v0 21e9c8c) and q/t2max-U1S (feat/trn2-max 1be9d85); logs
20261004T172300Z-sweep-U1-base, 20261004T175821Z-sweep-U1S; KV fits at every level, 0 preemptions):

The split rows below include the MoE prefill split, which fails the wikitext check: they are throughput only, NOT valid.

| conc | engine-v0 out tok/s | split out tok/s (not valid) | gain | split TTFT p50 / p90 | split ITL p50 | split $ / M out (trn2 spot) |
|---|---|---|---|---|---|---|
| 16 | 109.6 | **130.4** | +19% | 8.1 / 15.0 s | 91.5 ms | 32.14 |
| 32 | 145.1 | **179.7** | +24% | 8.4 / 29.4 s | 143.8 ms | 23.33 |
| 64 | 160.2 | **204.5** | +28% | 8.6 / 52.4 s | 273.3 ms | 20.50 |
| 128 | 172.0 | **220.6** | +28% | 9.0 / 99.6 s | 521.5 ms | 19.00 |
| 256 | 176.9 | **228.3** | +29% | 9.7 / 208.6 s | 1001.2 ms | 18.36 |

HBM per logical core with this configuration (neuron-monitor memory_used during the engine-v0 run): 18.86 GiB of 24 =
tensors 15.73 + model code 1.44 (32 graphs) + shared scratchpad 1.56 + constants 0.10 (DMA rings not reported there).
Conc 256 (32 rows per DP group in decode) adds 3.5% over conc 128: the box is prefill-bound.

**Where a split step goes** (the same configuration, one engine on cores 32-63, `KILN_PROFILE_PIECES=1 KILN_PROFILE_EXEC=1`,
`--concurrency 32 64 --requests 64`, log 20261004T182545Z-prof-U1S; pieces timed synchronously, so upper bounds):

| graph | p50 |
|---|---|
| prefill step, 4096 tokens (1024 rows per group, 128 stream rows per rank) | 928 ms: pieces 103 (layers 0-8: 3 dense + 6 MoE) / 123 / 126 / 122 / 125 / 123 / 126 (6 MoE layers each) / 65 (3) |
| decode step, 4 / 8 / 16 rows per group | 93.0 / 116.9 / 157.8 ms (pieces 17.8-23.4 / 22.5-29.3 / 27.1-41.4) |

So a prefill MoE layer costs ~20.7 ms, of which the NKI kernels are ~8 (MoE, 7.6 ms at C=4096 on the merged tree) plus
0.9 (KDA) or 1.4 (DSA scores + selection); ~11 ms per layer is XLA (projections, mHC, norms, router, shared expert, the
DSA attention core, the SP gathers / reduce-scatters and the mixer all-reduce). At conc 128 per engine (64 in flight, 16
per group) a request is 2 prefill steps (~1.86 s) and 256 / 64 x 158 ms = 0.63 s of decode: prefill ~75% of the device.

## trn2 on engine-v0 70ddc1b: baseline, the trn1 ports, decode kernels, expert parallelism at LNC=2 (2026-10-05, SDK 2.32, trn2.48xlarge Capacity Block)

kiln-t2-cb: trn2.48xlarge on EC2 Capacity Block cr-00ff977628a81fb28 (ap-south-2b, 2026-10-05 15:44 to 2026-10-06 11:30
UTC, $706.87 prepaid = $37.20/h; spot was refused in us-east-2a/b/c and on-demand in all three at 15:10 UTC), Kiln's DLAMI
copy ami-0cff1ca7a18e21334 (SDK 2.32), LNC=2 (64 logical cores of 24 GiB). GLM-5.3-Flash@eb9eb208 on the instance-store
RAID0 (`hf download --max-workers 48`: 306 GB in 83 s). Every graph from the compile farm (kiln-t2-cf / kiln-t2-cf2,
c8i.48xlarge spot in us-east-2, queues s3://<your-bucket>/compile-farm/q/t2f-*, cache compile-cache/trn2-sdk2.32/lnl/),
`NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`, 0 device compiles. Logs and their `.log.cmd` (exact env and command):
s3://<your-bucket>/logs/kiln-t2-cb/. Prices: trn2 spot $15.343/h (us-east-2c, read 15:10 UTC), half a box $7.67.

**Summary (2026-10-06 03:00 UTC, feat/trn2-fast c08c5ba = engine-v0 f997d35 merged; every number below is measured further down):**
- trn2 defaults now (all gated on real weights, trn1 keys either unchanged or recompiled in q/t1-trn2fast-2e9ea5b-G64 with
  identical NEFF instruction streams): the KDA / DSA decode kernels, SP decode streams, SP_GROUP, the fused DSA prefill kernel, and
  at LNC=2 the work split over both physical cores for delta_rule, dsa_topk, moe_dedupe, kda_decode, dsa_decode and dsa_fused.
- G1 whole box (8192 in / 256 out): engine-v0 70ddc1b 192.0 out tok/s at conc 128 ($22.20 per 1M out at spot) -> 221.6 with best1
  ($19.23) -> the new defaults (best1F, below). 8192-row prefill calls add ~10% on one engine.
- Prefill is the trn2 problem: 8.8k prefill tok/s per box at 4096-row calls, 9.9k at 8192 (trn1: 8.7k, 10.2k with EPLB; parity of
  BF16 efficiency would be ~35k). Per call: MoE blocks 52% (TP slices of 64 columns per expert keep the tensor engine busy at a low
  rate), token mixers 38% (latency-bound, every engine ~20% busy), 101 GB of spill traffic per rank.
- Decode is trn2's strength per box but not per dollar at 8K: with real KV, ST + v9 on D2K2 decodes 384 rows per engine in 217.5 ms
  (3,531 out tok/s per box, $1.21 per 1M out at spot; trn1's decode box $0.89); 96 rows per DP group is the 8K-context ceiling.
- Open trn2 bugs: expert parallelism hangs on the first execution of an EP decode graph at LNC=2, not deterministically (repro
  configs below); the MoE prefill kernel's LNC split fails the wikitext check with an out-of-bound indirect copy (not the DGE mode);
  sp_gather has no LNC=2 form (program 0 alone: NCC_ILLC059; both programs: wrong rows on every rank); the KDA / DSA decode row
  splits are exact in the NKI simulator but not bit-identical on the device (inside the decode-path gate).
- Measured and not levers today: FP8 operands in XLA dots (0.47-1.69x), the unsplit fused DSA kernel, EP decode.

**Baseline** (engine-v0 70ddc1b on trn2, the q/t2max-EU environment: `KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer
KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki
KILN_MOE_PREFILL_SKIP=0 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=6 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1`,
`bench/serve_sweep.py --tp 32 --dp 2 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264
--warmup --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 128 --concurrency 16 32 64 128 256 --decode-buckets 4,8,16,32
--kv-cache-gb 2.75 --kv-cache-dtype fp8`, queue q/t2f-base, log 20261005T155613Z-t2-base-U):

| conc | out tok/s | TTFT p50 | ITL p50 | prefill call | decode call | $/1M out (spot) |
|---|---|---|---|---|---|---|
| 16 | 118.5 | 9.5 s | 99 ms | 1.001 s | 32 ms | 35.97 |
| 32 | 158.9 | 9.9 s | 161 ms | 1.016 s | 48 ms | 26.82 |
| 64 | 177.3 | 10.1 s | 316 ms | 1.017 s | 73 ms | 24.04 |
| 128 | 192.0 | 10.4 s | 602 ms | 1.020 s | 107 ms | 22.20 |
| 256 | 198.4 | 11.1 s | 1163 ms | 1.048 s | 171 ms | 21.48 |

(The previous best valid trn2 number, feat/trn2-max 7138a25, was 189.9 at conc 128.) One engine on cores 0-31 with
`--concurrency 32 64 128` gives exactly half (88.6 / 96.0 / 99.2, log 20261005T163035Z-t2-e1-base), so the A/Bs below run
two engines at once, one config per half of the box, or one engine against that half-box baseline.

Roofline: a 4096-row prefill call is 138.6 TFLOP of model arithmetic (33.83 GFLOP per token, "Accelerator utilization of
the serving graphs") in 1.017 s on an engine of 8 Trainium2 chips: 2.6% of their dense BF16 peak (8 x 667 TFLOPS), 1.3% of
FP8. The TP MoE prefill kernel alone is ~14.7 ms per layer (the trn2 probe above) x 43 MoE layers = ~630 ms of the call.
Parity with trn1 per dollar (trn1 G64 156.2 out tok/s at $2.15/h) needs 156.2 x 15.343 / 2.15 = ~1,115 out tok/s per trn2
box: a 4096-row prefill call of ~0.19 s, ~14% of BF16 peak, where trn1's call runs at ~8.8% of its own.

**The fused DSA prefill kernel is neutral on trn2** (`KILN_DSA_FUSED=1 KILN_DSA_PREFILL_KERNEL=nki`, q/t2f-pC, cores 32-63
beside the baseline: 88.8 / 96.3 / 99.5 out tok/s, prefill call 1.013 / 1.016 / 1.044 s; log 20261005T164347Z-t2-e2-pC). On
trn1 it took the call 0.592 -> 0.519 s. It does not split its work by program, so at LNC=2 both physical cores run all of it,
while the XLA path it replaces is spread over both cores by the compiler.

**The KDA and DSA decode kernels with SP decode streams are a large decode win on trn2** (`KILN_KDA_DECODE_KERNEL=nki
KILN_DSA_DECODE_KERNEL=nki KILN_DECODE_SP=1`, trn1 defaults since e240cf0). Decode-only step, `tools/time_decode.py
--all-buckets --steps 48 --skip 8 --pieces` on the decode agent's shapes (`--max-num-seqs 512 --kv-cache-gb 0.5
--kv-cache-dtype fp8 --no-prefix-caching --decode-buckets 16,32,64`, null-page KV, one engine, q/dc1-t2 configs t2-TD-X /
t2-TD-K; X by the decode agent on cores 32-63, K log 20261005T172648Z-t2-td-K on cores 0-31):

| rows per group (per step) | engine-v0 trn2 default (XLA decode) | K: + decode kernels + SP decode | D2: K + moe_dedupe LNC split | D2K2: D2 + KDA / DSA decode row splits | out tok/s per engine (D2K2) | $/1M out, half box spot (D2K2) |
|---|---|---|---|---|---|---|
| 16 (64) | 143.6 ms | 110.7 ms | 93.7 ms | **87.7 ms** | 730 | 2.92 |
| 32 (128) | 219.7 ms | 154.8 ms | 121.7 ms | **110.7 ms** | 1157 | 1.84 |
| 64 (256) | 1438.7 ms (the XLA DSA mask form collapses) | 264.2 ms | 204.0 ms | **180.7 ms** | 1417 | 1.50 |

D2 is `KILN_LNC_SPLIT=delta_rule,dsa_topk,moe_dedupe` (the dedupe kernel's existing split: its two programs' fp32
partials summed, not bit-identical to grid 1; q/t2f-TDD2, log 20261005T181945Z-t2-td-D2), D2K2 adds this branch's
`kda_decode,dsa_decode` row splits (exact in the simulator, not on the device: see the gate below; q/t2f-TDD2K2, log
20261005T185627Z-t2-td-D2K2). D2K2 against engine-v0's trn2
default: -39 / -50 / -87%; whole box decode-only at 256 rows per engine ~2,830 out tok/s at 8K context, $1.50 per 1M out at
trn2 spot. Its MBU by the same byte count: 18.1 / 18.2 / 14.1%. The decode agent's fixed-cost stack ST
(`KILN_DECODE_WHOLE=1 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm`, feat/decode-scale c502221) on D2 instead of the row splits (its
q/dc1-t2 t2-TD-KST, run here on cores 0-31, log td2-KST): 1 / 4 / 16 / 32 / 64 rows per group 58.9 / 49.2 / 86.9 / 114.3 / 204.2 ms,
i.e. the same as D2K2 at 16 / 32 rows and slower at 64. Stacked, on a local merge of this branch with feat/decode-scale
(feat/trn2-fast-ds 0ea5c69 on engine-v0 b8814ab; buckets 1,4,16,32,64; q/t2f-TDX2 / q/t2f-TDST2, logs 20261005T212245Z-t2-td-X2,
211438Z-t2-td-ST2): D2K2 61.7 / 72.9 / 105.8 / 126.8 / 202.7 ms, D2K2 + ST 59.3 / 48.5 / 84.5 / 105.2 / 182.9 ms (1,400 out tok/s
per engine at 256 rows, $1.52 per 1M out). D2K2 alone is 12-20% slower on that tree than on this branch (87.7 / 110.7 / 180.7 at
16 / 32 / 64), while engine-v0 8229c3d leaves this branch's trn2 keys unchanged (rank-0 captures of the base2 / pBX / best1
serving configs identical on the merge), so the difference sits in feat/decode-scale or in the two extra decode buckets; ST
recovers it there. Against this branch's D2K2, ST is worth -4 / -5 / +1%. The same-bucket check settles it: the TDX2 env on the
merged tree with buckets 16,32,64 only runs 87.8 / 110.5 / 212.3 ms (64-row min 198.8; log 20261005T221314Z-t2-td-X2b), this
branch's D2K2 at 16 / 32, so the slowdown came with the two extra loaded decode buckets, not with feat/decode-scale's code; the
3-bucket ST run could not start (the whole-decode graphs' keys depend on the loaded bucket set: cache miss
4c0311508677364834b668c51f98a98a, log 20261005T220940Z-t2-td-ST2b).

**The trn2 decode box with real KV** (the PD decode side; 2026-10-06 01:25-02:30 UTC). Null-page rows all read one page and
flatter the bandwidth, so these time every row on its own pages and state row: `tools/time_decode.py --real-kv` (decode-scale
83036ba: each group's pool filled once with seeded values, 55-58 s for 28,598 pages), one engine, DP attention 4, contexts
8192 + a spread over 256, on the scratch merge feat/trn2-fast-ds2 ef289f8 (feat/trn2-fast 8ba166d + feat/decode-scale 83036ba).
KV sized to the bucket set (`--max-num-seqs 256 --kv-cache-gb 5.2 --decode-buckets 16,32,64` and `--max-num-seqs 384 --kv-cache-gb
7.8 --decode-buckets 96`); per-rank tensors from tools/tensor_bytes.py (trn2, rank 0, meta device): 18.79 GB (D2K2) / 19.05 GB (with
ST's bf16 dense weights) at 256 rows, 22.17 / 22.43 GB at 384, 25.45 / 25.71 GB at 512, which does not fit beside the graphs (the
long-context engine ran at 24.26 GB per logical core): **96 rows per group is the 8K-context ceiling** of a tp=32 engine. Queues
q/t2f-TDD2K2@r / TDST2@r / TDSTV@r / TDSTVn@r and the @rb set; logs 20261006T*-t2-tdr-{X,ST,STV,STVn,bX,bST,bSTV}.

| rows per group (per step) | X: D2K2 | ST: + `KILN_DECODE_WHOLE=1 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm` | STV: ST + `KILN_MOE_DEDUPE_MAX_TOKENS=256` (v9) | out tok/s per engine (STV) | $/1M out, half box spot (STV) | MBU (STV) |
|---|---|---|---|---|---|---|
| 16 (64) | 96.82 ms | 81.85 | **78.52** | 815 | 2.61 | 20.2% |
| 32 (128) | 111.55 | 107.52 | **104.95** | 1,220 | 1.75 | 19.2% |
| 64 (256) | 179.47 | 172.64 | **147.23** | 1,739 | 1.23 | 17.3% |
| 96 (384) | 245.74 | 245.96 | **217.50** | 1,766 | 1.21 | 14.0% |

(Null-page D2K2 on this branch was 87.7 / 110.7 / 180.7 ms: real pages cost +10% at 16 rows per group and nothing at 32 / 64.) ST is
worth -15 / -4 / -4 / 0% on D2K2 and v9 a further -4 / -2 / -15 / -12%: v9 is the decode lever on trn2 from 64 rows per group, where
every rank's MoE call has 256 tokens that v8 split into two 128-token calls each reading nearly every expert. v9 needs the
moe_dedupe LNC split there: with it off (`KILN_LNC_SPLIT=delta_rule,dsa_topk,kda_decode,dsa_decode`, v9 unsplit with its
slot-skip segments, STVn) the steps are 96.90 / 140.45 / 182.01 / 280.08 ms, worse than D2K2. MBU by the per-rank byte
model above (1.77 GB dense + 9.51 GB x the touched share + 113 MB per row of the group) at 725 GB/s per logical core. The whole box
(two engines) at 96 rows per group: **3,531 out tok/s, $1.21 per 1M out at trn2 spot** ($2.81 at a $35.76/h Capacity Block). trn1's
decode box (the decode agent: ST + v9, 28 rows per group at real 8K KV, trn1.32xlarge spot) is $0.89, so at 8K context trn1 is the
cheaper decode box per token; trn2's case is the context lengths trn1 cannot hold (its 24 GiB per logical core against trn1's 16).

**Numerics gate of D2K2 (the decode path)**: `tools/check_mixed.py` at the serving configuration (one engine, conc 32, 32
LONG_TEXT prompts of 700-8192 tokens x 64 greedy tokens, logprobs; q/t2f-base2 against q/t2f-pBX = K + `KILN_LNC_SPLIT=delta_rule,
dsa_topk,moe_dedupe,kda_decode,dsa_decode`; logs 20261005T205926Z-t2-cm-base2b, 210647Z-t2-cm-pBXb): **28 of 32 requests equal**,
token agreement 0.9175; teacher-forced |dlogprob| over the 1,843 decode-call positions mean 0.00114, p99 0.0351, max 0.187,
**signed mean +0.00020** (under test minus engine-v0's trn2 decode); prefill chunks identical. trn1 accepted its decode kernels at
28 / 32 equal and a signed -0.0004. On 64 wikitext prompts of 256-8192 tokens x 64 greedy tokens (`tools/greedy_ab.py`, which
has no logprobs) the same pair agrees on 12 of 64 prompts (leading agreement 0.476): wikitext prompts sit on near-ties (trn1's
accepted decode kernels gave 7 of 32 there), so greedy agreement there is not the gate; both runs' continuations read as fluent,
correct text. Each config repeats itself exactly (a second greedy_ab run of the baseline: 64 of 64, of pBX: 64 of 64; logs
20261005T220734Z-t2-ga-base2r, 221640Z-t2-ga-pBXr). But the KDA / DSA decode row splits are NOT bit-identical on the device,
although they are in the NKI simulator: K + dedupe split with them (pBX) against without them (pBD, log 20261005T204042Z-t2-ga-pBD)
agrees on 21 of 64 wikitext prompts. Their decode-path numerics are inside the gate above (the check_mixed pair includes them);
why the device differs from the simulator there is not known.

MBU of the K steps from the utilization analysis' per-rank bytes (dense 1.77 GB + 9.51 GB of experts x the share the rows
touch + 113 MB of KV and KDA state per row of the rank's group) at 725 GB/s per logical core (2.9 TB/s per chip / 4): 14.3 /
13.0 / 9.7%. In serving (q/t2f-pB, cores 0-31, log 20261005T165902Z-t2-e3-pB) the same switches give 89.6 / 97.5 / 103.3 out tok/s
at conc 32 / 64 / 128 per engine (+1.1 / +1.6 / +4.1% over the half-box baseline) with the decode call 73 / 107 / 171 -> 71 / 95 /
133 ms: the sweep is prefill-bound, so most of the decode gain does not show there.

**Expert parallelism was broken on trn2, two ways, and grid 1 is no way out.**
- kernels/moe_ep.py hit a gen3 rule at trace: "nc_matmul (transpose mode) dst dtype must match input dtype on gen3+, got
  dst=float32 but input=bfloat16" (the down outputs' transposes into an fp32 PSUM tile). Fixed: a bf16 PSUM tile from gen3 on,
  the same values (589e7c8).
- Every moe_ep kernel was launched at platform.nki_grid() = 2, so both physical cores ran the whole kernel, and its output is
  zeroed and then built by read-modify-write adds (dma_compute): two programs add every pair twice and race the zeroing. An
  engine with every trn1 default forced on (EP, `KILN_SP_GROUP=1`, the fused DSA kernel, the decode kernels, SP decode; q/t2f-port,
  log 20261005T163151Z-t2-e1-port) hung in its first 12-layer decode group graph (9e399d1ab412688431556204798c47eb, `TOPSP ...
  missing collectives status`, every rank dead 31 s later). The decode kernels + SP decode alone (pB above) and SP_GROUP + the MoE
  prefill skip alone (q/t2f-pD, log 20261005T180000Z-t2-e4-pD) both run, which leaves the grid-2 EP kernels. (pD is also a
  prefill win: `KILN_SP_GROUP=1 KILN_MOE_PREFILL_SKIP=20`, conc 32 per engine 94.8 out tok/s against 88.6, prefill call 0.922
  against 1.017 s; numerics not yet checked on trn2.)
- Launching them at grid 1 does not compile at LNC=2: `[NCC_IXGM002] Expected function sg0001 in subgraph 1 to have 9 basic
  blocks, but on core 1 it has 1 basic blocks` (q/t2f-pA1, every EP graph): the compiler gives a grid-1 kernel's device loops to
  core 0 only, and both cores' functions must match. So at LNC=2 every EP kernel runs at grid 2 and program 0 alone writes `out`
  (7fb0d7b, `_lnc_setup` / `wr`): both cores run the same device loops, the second core's adds are dropped.
- The LNC split (`KILN_LNC_SPLIT=...,moe_ep`, off by default; 8368d46, 269513c, 1c89e60): the prefill kernel and decode v2 give
  each physical core half of the I-chunks of gate_up and half of the 512-column output chunks of down; the a^T halves are swapped
  between the cores by `nisa.sendrecv` into buffers allocated before anything else (the same SBUF address in both programs'
  traces), so each core computes its output columns from the whole a^T accumulated over every I-chunk in one PSUM tile, the
  unsplit order, and adds only into its own columns of `out`. In the NKI simulator (tests/test_lnc_split.py
  test_moe_ep_split_is_the_whole_kernel, on a CPU host) the split and the unsplit grid-2 form both equal grid 1 bit for bit for the
  prefill kernel (first and overflow passes, per-row and tile scales) and decode v2 (small and dequantize-first passes); a control
  with the swap removed differs in 86,016 of its outputs. Three tracer rules the simulator does not enforce (each found by a
  capture failing): dict keys must be str ("'in' expected ... (str, dict)"), no direct call of an inner function inside a
  device-loop body ("inner functions can only be used as fori_loop/while_loop body arguments"), no `**` expansion ("keyword expansion
  is not supported"). A capture with `--ranks 0` is the cheapest tracer check (~3 min).
- **moe_ep's main kernel (kiln_moe_ep_kernel, the dequantize-first kernel that decode v2 hands rows above 128 and every
  prefill call goes to) hangs on trn2 hardware, split or not.** EP decode at the decode-only shapes above (K + `KILN_MOE_EP=1
  KILN_LNC_SPLIT=delta_rule,dsa_topk,moe_ep`, q/t2f-TDE2, log 20261005T181111Z-t2-td-E2): 16 / 32 rows per group 97.8 / 123.7
  ms (decode v2, its small passes), then 64 rows per group (256 rows: the main kernel) "TOPSP ... missing collectives status"
  and every rank dead. On one logical core (`tools/probe_moe_ep.py full --chunks 256 --fit row|block`, logs
  20261005T182958Z-t2-probe-ep-unsplit, 182959Z-split, 184031Z-full-row-unsplit): the first execution is correct (rel 0.0019
  against the emulation, rows without a local pair exactly 0) and a later one times out after 30 s with `SW_SEMAPHORE_ERROR`,
  `SW_PSUM_COLLISION_ERROR` and DMA aborts on the dynamic queues (`qActDynamicHW_8` / `qPoolDynamic_8`:
  TX_DATA_AXI_TIMEOUT_ERROR, RDR_NO_DESC_TIMEOUT_HINT). The passes alone (`probe_moe_ep.py core --lanes 128 --passes 9`,
  kiln_moe_ep_core: static rows, dynamic expert) run every execution, unsplit 612 us and split **318 us per 128-lane pass
  (1.92x)** with the same outputs (logs 20261005T183700Z-t2-probe-core-unsplit / 183701Z-core-split). So what hangs is in the
  main kernel's plan, lane-token, gather or scatter-add part, not in the passes.
- **What runs: the main kernel SPLIT with per-row scales** (`KILN_LNC_SPLIT=...,moe_ep KILN_MOE_EP_FIT=row`; probes on one
  logical core, 5 timed executions each after the checked one, every one completing):

  | form | C=256 | C=1024 | C=4096 | hot routing (8 / 56 overflow passes) |
  |---|---|---|---|---|
  | unsplit, row scales | hangs after the first execution | | | |
  | unsplit, tile scales (`KILN_MOE_EP_FIT=block`, the loader's default) | hangs | | | |
  | split, tile scales (with software DGE too) | hangs after the first execution | | | |
  | **split, row scales** | **1.98 ms** (rel 0.0019) | **1.87 ms** (0.0032) | **2.60 ms** (0.0029) | C=256 3.83 ms, C=1024 7.69 ms (0.0058) |
  | split, row scales, software DGE (`KILN_MOE_EP_DGE=sw`) | 2.00 ms | | | |

  (uniform routing over 288 experts, rank 0 of 32 with 9 local experts; logs 20261005T190746Z-t2-probe-full-row-sw-split,
  191017Z-full-row-hw-split, 191244Z-full-row-split-big, 191245Z-full-row-split-hot, 191016Z-full-block-sw-split.) Against the
  TP prefill kernel's 14.70 ms at C=4096 on a trn2 logical core, the EP split is 5.7x faster per layer on uniform routing (real
  routing loads the busiest rank more). Why the tile-scale path and the unsplit forms hang is not known yet; the DGE mode is not
  it.
- **OPEN trn2 bug: EP still hangs in a serving engine, also split with per-row scales.** Repro: env q/t2f-pE (the baseline set
  plus `KILN_MOE_EP=1 KILN_MOE_EP_FIT=row KILN_LNC_SPLIT=delta_rule,dsa_topk,moe_ep`), `bench/serve_sweep.py --tp 32
  --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --prefill-tokens 4096
  --prefill-buckets 1024 --max-num-seqs 128 --concurrency 32 64 128 --decode-buckets 4,8,16,32 --kv-cache-gb 2.75
  --kv-cache-dtype fp8 --core-base 0` (log 20261005T200308Z-t2-e6-pE): bucket warmup passes (every prefill and decode graph runs
  once on zeros), then the warm-up request hangs in a 12-layer decode group graph at 32 rows per group (1db35833a19b34f7075f5ffc0080bf75,
  27 collectives; cores on Neuron devices 5 and 7 time out, DMA queues qSPIO / qSPSpillReload / qSPDynamicHW / qPoolDynamic /
  qActDynamicHW report TDR_PREF_DESC_FRST_ERROR). 128 rows is decode v2 (kiln_moe_ep_small2); real tokens give an expert more
  than SMALL_LW = 16 pairs, which sends it to the dequantize-first pass, and the random-token decode timings never did. Not chased
  further: TP experts are the faster decode layout anyway (the decode agent's trn1 measurement: under EP the slowest rank loads ~4
  whole experts while the median loads none, and KILN_MOE_EP=0 took the fixed step 70.3 -> 55.1 ms), and trn2 decodes with TP
  experts by default.
- **The EP hang is not one pass and not deterministic** (2026-10-06 00:20-00:46 UTC, one engine each, farm graphs, cores 0-31 /
  32-63): every run died in bucket warmup, i.e. on the FIRST execution of an EP decode group graph, with the same "TOPSP ...
  missing collectives status" on all 32 ranks: bestE0 (the decode set + SP_GROUP + EP split with row scales +
  `KILN_MOE_EP_SMALL_ROWS=0`, so decode takes the main kernel: graph 79666893d0e94ea440c0cf19608fe456, one of its 12 new decode
  graphs; log 20261006T002323Z-t2-e8-bestE0), bestEL (the same with `KILN_MOE_EP_SMALL_LW=128`, so small2's dequantize-first loop
  has no trip at C <= 128; log t2-e8-bestEL), and pE itself with `--output-len 1 --decode-buckets 4` (graph
  f8818a2250d753d11ce51b149e28b36a, the same NEFF on all 32 ranks as a decode graph is; log t2-pf-E), although pE passed the same
  bucket warmup at 20:03. So it is a race somewhere in the EP kernels at LNC=2 that a run hits or not, not small2's overflow pass.
- **Prefill-only EP with TP decode does not fit**: it needs both expert layouts resident. Per rank the experts are 9.51 GB under
  either layout (TP: 1/32 of every expert; EP: 9 whole experts), and the best1 engine holds 20.15-20.22 GB per logical core
  (neuron-monitor: tensors 17.06, shared scratchpad 2.22, code 0.70-0.76) of the ~24 GB the runtime gives a logical core. A
  disaggregated prefill engine, which never runs a decode graph, could hold the EP layout alone.

**The MoE prefill kernel's LNC split still fails the trn2 wikitext check with software DGE.** The feat/trn2-max defect (an
out-of-bound vector-DGE notification from an SP-engine instruction, with every output still bit-identical) looked like hardware
DGE handling `oob_mode=skip`, so `KILN_MOE_PREFILL_DGE=sw` (kernel argument dge: every dynamic DMA of the kernel with dge_mode
swdge) was tried on the failing check itself (`tools/check_ppl.py --tp 32 --piecewise --kv-cache-gb 1.0 --text-file
wikitext2_test.txt`, `KILN_LNC_SPLIT=delta_rule,dsa_topk,moe_prefill KILN_MOE_PREFILL_MIN_TOKENS=1`, q/t2f-P1Msw, log
20261005T214318Z-t2-wt-P1Msw): the same "scatter/gather (indirect memory copy via vector DGE) out-of-bound access" on several
ranks. Not the DGE mode; the split stays off.

**Real-weight gates of the prefill-side switches** (`KILN_SP_GROUP=1 KILN_MOE_PREFILL_SKIP=20` on top of the decode set, "best1";
wikitext-2 slice through tools/check_ppl.py at DP attention 4, `--kv-cache-gb 1.0 KILN_MOE_PREFILL_MIN_TOKENS=1`, q/t2f-base2-ppl and
q/t2f-best1-ppl, logs 20261005T*-t2-wt-base2 / -t2-wt-best1): engine-v0's trn2 default **-0.554**, best1 **-0.544** (difference
+0.0103, |dlogprob| mean 0.0554, greedy agreement 97.8%, the device-vs-device spread of every accepted pair).

**Whole box with best1** (feat/trn2-fast 118c6ea = engine-v0 8229c3d merged; the baseline's command and environment plus
`KILN_KDA_DECODE_KERNEL=nki KILN_DSA_DECODE_KERNEL=nki KILN_DECODE_SP=1 KILN_LNC_SPLIT=delta_rule,dsa_topk,moe_dedupe,kda_decode,
dsa_decode KILN_SP_GROUP=1 KILN_MOE_PREFILL_SKIP=20`, i.e. a922fe9's trn2 defaults plus the skip; q/t2f-best1, both engines at once,
log 20261005T222609Z-t2-U-best1):

| conc | out tok/s | vs baseline | TTFT p50 | ITL p50 | prefill call | decode call | prefill share | $/1M out (spot) |
|---|---|---|---|---|---|---|---|---|
| 16 | 131.3 | +10.8% | 8.6 s | 89 ms | 0.908 s | 29 ms | 66% | 32.46 |
| 32 | 176.0 | +10.8% | 8.9 s | 146 ms | 0.922 s | 42 ms | 72% | 24.22 |
| 64 | 202.4 | +14.2% | 9.1 s | 276 ms | 0.924 s | 58 ms | 78% | 21.06 |
| 128 | 221.6 | +15.4% | 9.2 s | 525 ms | 0.927 s | 76 ms | 83% | 19.23 |
| 256 | 235.1 | +18.5% | 9.4 s | 1004 ms | 0.937 s | 97 ms | 87% | 18.13 |

The decode call at 32 rows per group falls 171 -> 97 ms (-43%) and the prefill call 1.017-1.048 -> 0.908-0.937 s (-9 to -11%,
SP_GROUP and the skip). Prefill is 66-87% of the device time, so the sweep follows the prefill call: G1 on trn2 is a prefill
problem (2.8% of BF16 peak at 0.924 s), and at $18.13-19.23 per 1M out it is 3.4x trn1's G64 EP number per dollar ($5.59).

**Prefill tokens per second per box, the metric for trn2 prefill** (2 engines x rows per call / prefill call): best1 4096 rows
in 0.922-0.937 s = **8.8k**; trn1.32xlarge G64 one-piece 8192 (prefill agent, engine-v0 8229c3d) 8.7k, with EPLB 10.2k. trn2 has
3.5x trn1's BF16 compute per box, so parity of BF16 efficiency would be ~35k. **8192-row prefill calls** (best1 + `--prefill-tokens
8192 --prefill-buckets 2048 KILN_PIECEWISE_PREFILL_MOE_GROUP=4`, q/t2f-best1-p4@pf8k, one engine on cores 0-31, log
20261005T225412Z-t2-e7-best1pf8k): conc 32 / 64 / 128 per engine 110.7 / 122.9 / 131.6 out tok/s against best1's 101.2 / 110.8 /
117.6 at the same load per engine (+9.4 / +10.9 / +11.9%; whole box ~263 out tok/s = ~$16.2 per 1M out at conc 256), prefill
call 1.637-1.660 s per 8192 rows = **9.9k** per box (+12%), decode call 23 / 55 / 82 ms.

**Where a best1 4096-row prefill call goes on trn2** (the "Accelerator utilization" method: `KILN_CAPTURE_INPUTS`
of prefill call 20 in a best1 serving run on cores 0-31, 64 requests at conc 32, ~16 GB per rank of inputs on the NVMe; each of
the call's 10 NEFFs replayed on 32 workers with every rank's inputs, rank 0 profiled, `tools/util_report.py replay / bins /
report`; logs 20261005T232558Z-t2-ut-best1, 20261006T*-t2-utrep2-best1): the graphs take 958.6 ms (serving fit 0.92 s), prep
1.6 + 8 pieces of 6 layers (95-135 ms, the last 67) + post 2.1.

| per call, rank 0 | ms | share | engines busy (tensor / vector / scalar / gpsimd) |
|---|---|---|---|
| MoE blocks: 45 segments ending in the world reduce-scatter of 32 MiB, 10.98 ms each | 494.3 | 52% | 85 / 79 / 79 / 34% |
| token mixers (KDA or DSA): 45 segments ending in the attention group's reduce-scatter of 8 MiB, 8.11 ms each | 365.1 | 38% | 23 / 21 / 23 / 3% |
| between: 233 segments ending in an 8 MiB all-reduce (0.07 ms each), 42 of 0.25 MiB, graph ends | 22.6 | 2% | |
| collective transfers (233 x 0.25 + 45 x 0.21 + 45 x 0.07 ms; they overlap the segments) | ~89 | | |

Every engine idle without a collective in flight: 223.7 ms (23%), almost all of it inside the token mixers. HBM: 113.8 GB read
and 53.5 GB written per rank, of which **101 GB is spill save / reload** (47.8 / 53.2 GB), 175 GB/s average (24% of a logical
core's 725 GB/s). Tensor engine 21.6 TFLOP (+ 5.5 of transposes) = 22.5 TFLOP/s, 13.5% of a logical core's ~167 dense BF16. So
the MoE kernel keeps the tensor engine busy (85%) at a low rate: under TP every rank holds a 64-column slice of each of 288
experts (I = 2048 / 32), so each expert's matmuls are ~114 rows x 4096 x 128, a poor shape for the 128 x 128 array (EP's whole
experts run the same layer in 2.6 ms, see above). The token mixers are the latency-bound half: 8 ms per layer with every engine
~20% busy.

**The fused DSA prefill kernel split over the two cores** (baf73ab `KILN_LNC_SPLIT=...,dsa_fused`, with `KILN_DSA_FUSED=1
KILN_DSA_PREFILL_KERNEL=nki` on best1; q/t2f-best1F, one engine on cores 32-63, log 20261006T*-t2-e9-best1F): conc 32 / 64 / 128
per engine 104.4 / 114.7 / 121.9 out tok/s against best1's 101.2 / 110.8 / 117.6 (+3.2 / +3.5 / +3.7%), prefill call 0.885-0.899
against 0.922-0.937 s (-4%). Exact against grid 1 in the NKI simulator (tests/test_lnc_split.py). Gates on the device, against
best1: wikitext (check_ppl at DP attention 4, q/t2f-best1F-ppl, log 20261006T021852Z-t2-wt-best1F) -0.5498 against -0.5441
(engine-v0's trn2 default -0.5543), |dlogprob| mean 0.045, greedy agreement 98.2%, inside the device-vs-device spread of the
accepted pairs (best1 against engine-v0: 0.0554 / 97.8%); check_mixed (conc 32, 32 LONG_TEXT prompts x 64 tokens; logs
*-t2-cm-best1, *-t2-cm-best1F) 28 of 32 equal, token agreement 0.9136, teacher-forced signed dlogprob -0.0020 over the 32 prefill
chunks (|d| mean 0.0129) and +0.0009 over 1,839 decode positions (|d| mean 0.0021). So the fused kernel with its split is the trn2
default now (FUSED_FAMILIES, PREFILL_KERNEL_FAMILIES and the LNC=2 split list; trn1 keys unchanged: both kernels' REV regions are
untouched). The unsplit fused kernel was neutral (pC above).

**sp_gather (the NKI all_gather for the SP row gather) has no working LNC=2 form yet**: probe_nki_cc.py --ranks 32 --cases
xag,sp1 on cores 32-63 (logs *-t2-cc-sp1/2, *-t2-cc2-both / -split). Program 0 alone issuing the collective (c7c43e9) fails to
compile, `[NCC_ILLC059] Could not find MemoryLocation named inst__I-3-0:_mem_0 on core 1`. Both programs issuing it (af9308e
`KILN_SP_GATHER_LNC=both`, and `split`, each copying out half) compile and run, but the gathered rows are wrong on 32 of 32 ranks
(max |err| 7.48 in both forms) at p50 3.78 / 3.22 ms. The XLA zero-padded all-reduce it would replace is exact, at 3.5-5.8 ms for
[32 x 128, 4096] bf16 (trn1: 2.27). trn2 keeps `KILN_SP_GATHER=xla`.

**trn1 is unaffected by this branch's kernel edits.** The REV regions of moe_ep, moe_prefill, kda_decode and dsa_decode moved
(each kernel's source CRC is a static argument), so a rank-0 capture of the trn1 G64 serving config (`--target trn1 --ranks 0`,
`--tp 32 --dp-attention 4 --max-num-seqs 64 --decode-buckets 16 --kv-cache-gb 1.5`, `KILN_MOE_PREFILL_SKIP=20
KILN_PIECEWISE_PREFILL_MOE_GROUP=12`, EP auto) gives 15 keys on engine-v0 8229c3d and 15 on feat/trn2-fast a922fe9, 9 shared and 6
different. All six pairs, compiled on kiln-t2-cf into one local cache, have identical NEFF instruction streams
(`tools/neff_stream_cmp.py`'s streams(), every engine queue; /opt/kiln/trn1cmp.sh). trn1 hosts pay one compile of those six
graphs and run the same code.

**FP8 in XLA dots on trn2: shape-dependent, mostly not faster.** `tools/probe_fp8_matmul.py` (one logical core,
NEURON_RT_VISIBLE_CORES=2, `KILN_CC_ARGS=--model-type=transformer`, 8 matmuls per graph against 8 different weights, p50 of 20
synchronous calls; log 20261005T193220Z-t2-probe-fp8mm2): bf16 x bf16 reaches 106.3 / 106.7 / 123.2 TFLOP/s at M x K x N =
1024 x 4096 x 4096 / 4096 x 4096 x 4096 / 1024 x 4096 x 16384 (64-74% of the logical core's 167 TFLOPS dense: XLA already uses
both physical cores well for a dense matmul); the same products with float8_e4m3fn operands (per-tensor scales, as the graph is
compiled with LNL's unsafe e4m3fn-as-e4m3 flag) run 86.9 / 50.2 / **208.5** TFLOP/s = 0.82 / 0.47 / **1.69x** (the last one above
the BF16 peak, so neuronx-cc used the FP8 double mode there), and casting the fp8 operands to bf16 first gives 0.80 / 0.68 / 1.37x.
Product error against fp32 0.036-0.040 (per-tensor scales). So the 2x FP8 rate is reachable from XLA only for some shapes, and the
model's dense matmuls are a small part of a step anyway: 138.6 TFLOP per 4096-row prefill call is 4.33 TFLOP per logical core,
~39 ms at XLA's measured 110 TFLOP/s against a 1.017 s call. The time is in the MoE kernel, the elementwise and normalisation
work, and the collectives, not in matmul throughput; FP8 compute moves the call by at most a few percent until those shrink.

## Packing prefill chunks across DP-attention groups (2026-10-05, SDK 2.32, trn1.32xlarge)

Under DP attention a prefill call carries at most one chunk per group (each group's token mixers take one sequence
per call) and costs about the same however many groups fill it: the device split of the G1b sweeps puts a 4096-row
call at 0.97 s on TP and 0.75 s with expert parallelism, whether its four 1024-row slots hold four chunks or one. A
step makes as many prefill calls as its busiest group has chunks. Two things leave slots empty:

- **a group's budget split over two entries** (a request's last few hundred tokens plus the next request's first
  chunk, or a chunk split at a checkpoint target): that group alone needs a second call that step;
- **groups without prefill work** while another has a chunk (prompts of different lengths, cache hits beside misses,
  requests finishing out of step, which EOS and speculative decoding cause).

**Where it happens** (prefill calls against the packed minimum ceil(chunks / groups), counted from the G1b sweeps'
device split, docs/price-performance.md "G1b"): the uniform G1 workload is packed exactly (conc 32: 256 calls for
256; conc 64: 512 for 512), and so are the warm shared-prefix levels (128 for 128); the cold shared-prefix levels,
whose misses compute 8 chunks beside hits that compute 2, took 1.16-1.46x (conc 64, 75% shared: 206 calls for 155).

**A scheduler-only model of it** (tools/sim_dp_pack.py: the engine's DPScheduler, Schedulers, radix caches and page
pools in serve_sweep's closed loop, a stand-in checkpoint-row allocator, a step costing its decode calls x
--decode-call plus its prefill calls x --prefill-call, no device) reproduces the device: G64 uniform 512 calls,
97.6 out tok/s (device 97.3, c33a391 TP); 75% shared cold 206 calls, 181.1 (device 206, 180.4); warm 128, 231.9
(device 128, 231.6); 50% cold 320, 137.5 (device 320, 136.9). On workloads the sweeps had not run:

| workload (G64, 256 requests, TP costs 0.97 / 0.15 s) | off: calls / min, out tok/s, TTFT p50 / p90 | trim | hold (4) |
|---|---|---|---|
| prompts 1024-8192 tokens, 256 out | 513 / 353, 98.3, 15.2 / 77.9 s | 391 / 331, **119.2**, 11.2 / 55.6 s | |
| prompts 1024-8192, 128-256 out (EP costs 0.75 / 0.175 s) | 704 / 348, 71.8, 8.9 / 62.4 s | 594 / 331, 81.3, 7.4 / 45.5 s | 449 / 331, 97.9, 10.4 / 45.5 s |
| 8192-token prompts, 128-256 out | 727 / 512, 58.2, 15.7 / 88.4 s | the same (nothing to trim) | 617 / 512, 65.6, 23.0 / 88.4 s |
| 75% shared, prompts 6400-8192, warm | 180 / 123, 197.7, 5.3 / 29.8 s | 112 / 102, **246.2**, 3.2 / 20.4 s | |

So **trim** (a group's 2nd chunk waits a step rather than add a call fewer than every group fills) gains 9-21% where
lengths differ (+21% and +25% on rows 1 and 4) and improves TTFT p50 and p90 with it (the step's decodes are no longer held behind a nearly empty
prefill call, and less device time is spent per prompt); it does nothing on uniform work, where no group ever has two
chunks. **hold** (a sparse lone call sits out while decodes run, at most hold_steps steps per request) is what helps
when requests finish out of step, at a TTFT cost: +13% and TTFT p50 +47% at hold 4 on 8192-token prompts with 128-256
outputs (+27% and +70% at hold 16). Trimming at half the groups (pack_min 2 of 4: 113.4 out tok/s on the first row) gained
less than at every group (119.2), so every group is the default threshold.

**On the device** (engine-v0 f9dc4c4 + feat/dp-prefill-pack edd0d92, q/f9dc4c4-trn1 G64-4096-KV1.5-S20-P12-K: conc 64,
DP attention 4, prefill 4096 / 1024 per group, decode 16 per group, KV 1.5 GB FP8, expert parallelism by default;
`bench/serve_sweep.py ... --dp-prefill-pack off trim hold --dp-prefill-hold-steps 4`: the three modes on one engine,
each from a flushed cache with the same requests). R1: kiln-pack-32 (trn1.32xlarge on demand, 2026-10-05
00:04-00:39 UTC, log s3://<your-bucket>/logs/kiln-pack-32/pc-pkR1.log), 256 requests with prompts of
1024-8192 tokens and 128-256 output tokens (`--input-len-min 1024 --output-len-min 128`). R2: kiln-g1-trn1 (trn1.32xlarge
spot, 00:25-00:55 UTC, log logs/kiln-g1-trn1/pc-pkR2.log), 192 requests per level, 4 prefixes of 6144 tokens, prompts of 6400-8192 tokens,
256 output tokens, each mode cold then warm (`--shared-prefix-len 6144 6144 --input-len-min 6400 --keep-cache`).

| run | mode | prefill calls / packed min | chunk slots used | out tok/s | TTFT p50 / p90 | ITL p50 | spot $ per request |
|---|---|---|---|---|---|---|---|
| R1 | off | 703 / 337 (2.09x) | 47.9% | 74.7 | 8.3 / 58.5 s | 803 ms | 0.001531 |
| R1 | trim | 588 / 320 (1.84x) | 54.4% | **85.4 (+14.3%)** | **7.2 / 39.8 s** | 727 ms | 0.001338 |
| R1 | hold (4) | 445 / 320 (1.39x) | 71.9% | **103.0 (+37.9%)** | 11.1 / 39.8 s | 549 ms | 0.001110 |
| R2 cold | off | 221 / 125 (1.77x) | 56.2% | 164.7 | 10.4 / 36.4 s | 339 ms | 0.000928 |
| R2 cold | trim | 151 / 107 (1.41x) | 70.4% | 196.9 (+19.5%) | 7.6 / 25.9 s | 294 ms | 0.000776 |
| R2 cold | hold (4) | 130 / 105 (1.24x) | 80.6% | 202.2 (+22.8%) | 9.7 / 25.9 s | 253 ms | 0.000756 |
| R2 warm | off | 138 / 97 (1.42x) | 70.3% | 207.2 | 6.9 / 23.5 s | 274 ms | 0.000738 |
| R2 warm | trim | 90 / 80 (1.13x) | 88.1% | **241.1 (+16.4%)** | **4.1 / 15.7 s** | 242 ms | 0.000634 |
| R2 warm | hold (4) | 85 / 79 (1.08x) | 92.3% | 239.6 (+15.6%) | 4.8 / 15.7 s | 237 ms | 0.000638 |

The model predicted them (R1 off / trim / hold 71.8 / 81.3 / 97.9 out tok/s and 704 / 594 / 449 calls; R2 cold 160.2 /
192.3 / 198.2, warm 203.5 / 237.9 / 237.0). Trim needs fewer chunks than off (1280 against 1348 in R1): a split chunk's
second half no longer runs as a separate entry.

**Uniform work is unchanged** (G1 conc 16, q/f9dc4c4-trn1 G16-4096-KV0.65-S20-P12-K, kiln-g1-trn1, 64 requests per level,
log pc-pkG16.log): off 128 prefill calls for 512 chunks (exactly packed), 82.5 out tok/s, TTFT p50 9.39 s, ITL 156.6
ms; trim the same 128 calls, 0 chunks deferred, 82.6 out tok/s, 9.39 s, 156.6 ms. Every 8192-token prompt is eight
whole chunks, so no group ever has two in a step; the conc-32 and conc-64 sweeps above were packed exactly too.

**Tokens** (`tools/check_pack.py --prompts 16 --min-len 1024 --new-tokens 48 --modes off off trim hold` on the G64
command, kiln-g1-trn1, log pc-pkC1.log: 16 wikitext prompts of 1355-7941 tokens submitted together, the cache flushed
before each mode; 28 prefill calls off, 24 with trim or hold): a scheduling mode changes when a chunk runs, never what
it computes, and on CPU in fp32 every mode gives exactly transformers' tokens
(tests/test_linear_serving.py::test_dp_prefill_pack_keeps_every_token, sync and overlap, with checkpoint saves and
hits among the deferred chunks). On the device a deferred chunk starts where its request's next chunk would have (a
request admitted beside another's last few hundred tokens computes 0-1024, 1024-2048, ... instead of 0-832,
832-1856, ...) and shares its call with other chunks, so it carries the rounding of a different chunking: the same
`off` batch re-run after a flush kept 8 of 16 prompts identical over 48 tokens (mean |dlogprob| 0.0094, max 0.24),
trim and hold against off 6 of 16 (mean 0.030, max 0.57; trim and hold gave each other's tokens), the first differing
tokens at reference margins of 0.0-0.25 nats but one at 0.625, the size of the chunking differences measured for
off-grid prefix hits (docs: "Prompt caching for GLM-5.3-Flash on the device", up to 0.75).

**Opt-in, not the default** (`dp_prefill_pack` defaults to "off"; KILN_DP_PREFILL_PACK=trim or hold, or
EngineConfig.dp_prefill_pack). Trim gains where prompt lengths or cache hits differ across groups, lowers TTFT p50 and
p90 there, and leaves uniform work untouched, but it changes device tokens: on mean |dlogprob| trim vs off (0.030) sits
at 3x off's own rerun floor (0.0094), the cause (a deferred chunk on its request's natural chunk grid) is explained but
its direction is not measured, and a default-on scheduler change needs the bias check the decode kernels passed. **The
check that would promote it:** after the determinism fix lowers the rerun floor, teacher-forced signed mean dlogprob of
trim against off on the same prompts (each mode scoring the other's tokens), compared with the same quantity for off
against a rerun of off; a signed mean inside the floor, not just a small |d|, makes trim a default candidate. "hold"
stays opt-in after that too: it is what helps
when requests finish out of step (EOS, MTP / speculative decoding: +37.9% out tok/s at conc 64 with 128-256 outputs)
at the cost of TTFT p50 (+34% there against off, +54% against trim); dp_prefill_hold_steps bounds what one request can
lose (default 1 step; 4 measured). What packing cannot reach: groups that have no prefill work while another does,
which a closed loop at full concurrency creates whenever requests finish at different times (R1 trim still 1.84x the
minimum); an idle group cannot take a chunk of a request placed elsewhere (its KV pages and state rows live on that
group's ranks), and waiting requests to rebalance exist only when the groups are full.

**Suite** (kiln-lead-ci3, m7i.8xlarge, one pytest process, `KILN_TEST_MODEL=Qwen/Qwen3-0.6B`, transformers 5.15) on
feat/dp-prefill-pack 09ce40a, rebased onto engine-v0 aab34ff (with trim then the default; later commits docs): 669 passed, 48
skipped; the transformers-5.18 files (glm5_next, qwen4_exp, linear_serving, dsa_topk, dsa_select, moe_ep, dp_attention,
mtp_mla, mla, mixed_batch) 285 passed, 12 skipped (logs s3://<your-bucket>/logs/kiln-lead-ci3/pk-suite3.log,
pk-suite3-518.log; before the rebase, on f9dc4c4: 667 / 42 and 282 / 8). New: test_linear_serving.py
test_dp_prefill_pack_keeps_every_token[trim / hold], test_dp_prefill_pack_trims_and_holds. After the default went back
to "off" (7935785, plus test_dp_prefill_pack_is_opt_in): the scheduler and packing files (test_scheduler, test_dp_attention,
test_linear_serving, test_radix_cache, test_hicache, transformers 5.18 first on PYTHONPATH) 90 passed, 1 skipped (log
logs/kiln-lead-ci3/pk-suite4.log); the flip changes no other path.
## Padded rows wrote the slot they read: why a cold rerun differed (2026-10-04/05, SDK 2.32, trn1.32xlarge)

pc-chk5 (prompt caching above) saw two of six cold reruns differ from the first cold run past output token 16 with the
batch composition held fixed. Cause: **every padded row of a DP-attention group wrote its KV into slot 0 of the null
page and read that same slot back as its one visible key, in the same 12-layer decode graph**, so the value it read
changed from execution to execution with identical inputs; and a request shares calls with those rows, so their
values reached the request's own arithmetic. Fix: `ModelRunner.pad_slots` (padded rows write slots 1 .. page_size - 1
of the null page, nothing writes slot 0 after start-up). Host-side only: every graph and NEFF is unchanged.

Setup for every device number here: kiln-det-32 (trn1.32xlarge, us-east-2c, on-demand), zai-org/GLM-5.3-Flash
eb9eb208 bf16, the F0-4096-KV1.5-S20-P12-K config of q/final-c33a391 (env and argv verbatim: tp 32, DP attention 4,
piecewise 12-layer groups, overlap, prefill 4096 / buckets 1024, decode bucket 8, page bucket 264, KV 1.5 GB bf16,
in-place pool keys) plus `KILN_MOE_EP=0`, so the farm's TP graphs load with 0 compiles on engine-v0 c94d3d1 (its 11
decode NEFF keys are the ones pc-chk5 ran from q/v1-trn1; only the 3 prefill-group keys differ). Tool:
`tools/check_cold_determinism.py --text-file wikitext2_test.txt --prefixes 6 --ops ... -- <F0 argv>` (the same
wikitext prompts as check_prefix_cache.py: X_k = P_k + S_k with 6144-token prefixes, 8192 tokens, each request alone),
logs s3://<your-bucket>/logs/kiln-det-32/det1.log .. det10.log (+ .json), 2026-10-04 22:26 to
2026-10-05 01:00 UTC.

| run | what | result |
|---|---|---|
| CPU fp32 | random glm5_next config, tp 4 with DP attention 2 and 4, overlap, piecewise, SP prefill, page 8 and 32, pc-chk5's order (`--build`) | 6/6 and 6/6 bit-identical reruns: the host logic (pages, state rows, checkpoint rows, board) is exact |
| det1 | pc-chk5's order `reset,X,X,Y,V,Zc,X,reset,X,X` (Zc: the interval checkpoints, read back to the host), 64 tokens | 30/30 X reruns identical; after `reset` (every KV, state, board and DSA scratch buffer zeroed on all ranks and the allocators restored) 72/72 calls with identical host arguments |
| det2 | `RD@k:n`: X_k to position 8211, its state row checkpointed, ONE decode call replayed n times (row restored first) | the request's row bitwise stable 4000/4000 (sync and 4 calls in flight, X_3 and X_4); the last prefill chunk 60/60; **the 7 padded rows of one DP group differ in 999 of 1000 calls** (the other groups' padding stable) |
| det3, det4 | per-graph outputs of the replay (`--tap`), rank 0's buffers before each call | rows 1-7 differ from the first 12-layer graph on (max 0.004-0.008), rows 1-4 agree with each other and 5-7 do not; null-page slot 0 of the DSA caches and the scratch state rows change between calls |
| det6 | the same replay with one layer per decode graph (`KILN_PIECEWISE_MOE_GROUP=1`, compiled on the box) | 20/20 stable on every row |
| det5 | scratch state row and null page zeroed before every call | still 49/50: not leftover state |
| det8 | padded rows given distinct null-page slots (a temporary switch) / distinct scratch state rows | **30/30 stable** / still 29/30 |
| det9 | default vs distinct-slot padding, 6 prompts x 128 tokens; then default again after `reset` | **all 6 requests differ** between the two (first logprob difference at output tokens 15 / 15 / 50 / 96 / 21 / 116, max 0.13 nats, tokens later); the default rerun with identical memory and 136/136 identical call arguments: **1 of 6 differs from token 91 (max 0.12)**, pc-chk5's symptom |
| det9 | foreign data written into the decoded slot before each of 2000 sync replays (`:fill_cur`) | the request's row never moved: not a stale-page leak |
| det10 | **the fix**, pc-chk5's order at 128 tokens and after `reset` | 18/18 reruns identical |
| det10 | **the fix**, `RD` / `RA` 1000 calls each, `RD:fill_cur` 1000 | 0 calls differ on any of the 32 rows |
| det10 | `:fill_null0` (the padded rows' key changed before every call), positions 8211 and 8251 | the 31 padded rows differ in 99/99 calls, the request row in none: padded values reach a request only at some steps, which is why the reruns differed late and not on every prompt |

**What the race is.** Under DP attention a decode call is 8 rows per group; the request is row 0 of its group and rows
1-7 are padding at position 0 behind an all-null table, so the only key they see is null-page slot 0. Each padded row
stored its own latent, indexer row and (in-place) pool key into that slot (`slot_mapping` 0), then gathered its
context from the cache in the same layer: seven writers and seven readers of one address inside one graph. Which value
is read back (whose write, before or after it lands) is not fixed by the program, and in the 12-layer decode graphs it
varies per execution (det2), while the one-layer graphs happen to be stable (det6). A padded row's output is discarded,
but its hidden state flows on through the call: in det9 padded rows that compute different values change the
request's logprobs by up to 0.13 nats, intermittently (det10 `fill_null0`: not at every step). The likeliest carrier
is the dedupe MoE kernel (kernels/moe_dedupe.py), whose slot / lane / block plan is computed from every row's routing,
so other rows' routing moves where a token's expert outputs are summed; that op was not isolated. The request's own
slot is not involved: it is written and read once per call, and foreign data placed there beforehand never reached its
output in 3000 synchronous replays. Interval checkpoints were not involved either: pc-chk5's repeat just happened to
be the run that lost the race.

**The fix (engine/model_runner.py `pad_slots`).** Padded rows write slots 1 .. page_size - 1 of the null page in turn,
in decode, verify, prefill chunks, mixed calls, MTP drafts and warmup; slot 0, the key they read, keeps the zeros it was
created with. Those slots still read as padding where a graph tells padding by `slot_mapping >= page_size`
(linear_attn's chunk form), and nothing reads them unmasked (a padded row's other positions and pools are invisible).
No graph changed: the same farm NEFFs load (det10, 0 compiles). Request outputs differ from engine-v0 at the level of
det9's first row (padded rows now compute different, fixed values), which is engine-v0's own run-to-run spread.
`tests/test_padded_rows.py` checks the invariant on the host (CPU index_put_ is sequential, so the race itself cannot
show there): slot 0 of the null page of every paged cache is still zero after warmup and serving decode with padded
rows, DP attention (2 groups, one idle while the other prefills), overlap, chunks shorter than their bucket, mixed
batches and n-gram verify with padded draft rows, for Qwen3 and the truncated GLM-5.3-Flash: 6 failed on engine-v0
c94d3d1, 6 passed with the fix (kiln-det-32 host, 2026-10-05). The device probe that keeps it fixed is
`tools/check_cold_determinism.py --assert-stable --ops reset,X,X,RD@3:1000,RA@3:1000 -- <F0 argv>` (exit 1 when any
rerun or replay row moves).

**Gates on the fix** (0a889fb). Real weights (kiln-mimo-trn1, trn1.32xlarge spot, q/ep-trn1's ppl4-EP-2ca0 /
pplwiki-EP-2ca0 env, `KILN_MOE_EP=1 KILN_MOE_KERNEL=nki KILN_PIECEWISE_MOE_GROUP=12 KILN_CC_ARGS=--model-type=transformer
KILN_LINEAR_ATTN_KERNEL=nki python tools/check_ppl.py --model zai-org/GLM-5.3-Flash --tp 32 --piecewise --kv-cache-gb 1.0
[--text-file wikitext2_test.txt]`, logs s3 logs/kiln-mimo-trn1/det-ppl4-EP-2ca0.log, det-pplwiki-EP-2ca0.log,
2026-10-05): France -2.147, Water boils -3.527, def add -0.956, quick fox -1.714, mean **-2.074**; wikitext slice
**-0.551** over 3071 tokens, by chunk -0.770 -0.693 -1.101 -0.929 -1.206 -0.183 -0.355 -0.137 -0.298 -0.275 -0.334
-0.333: the EP tree's values above, chunk for chunk (check_ppl runs without DP attention, where a padded row's one
key is its own sequence's first one, so these could only show a breakage). Greedy against engine-v0 c94d3d1 (fresh
engines on the same box, F0 graphs, the 6 wikitext prompts x 64 tokens, `--ops X --dump-json`, logs det-ab-base /
det-ab-fix): tokens equal on 4 of 6; prompt 1 from token 34 (engine-v0's top-2 margin there 0.375), prompt 2 from token
60 (a tie, margin 0.000); logprobs move from tokens 9-63 by at most 0.10 nats, engine-v0's own rerun spread (det9).
Replay probe on mimo (det11): reruns exact, `RD` / `RA` 1000 calls each stable on all rows, `RD:fill_cur` 500 stable.
CPU suite on 0a889fb (kiln-det-ci2, m7i.8xlarge, one process, `KILN_TEST_MODEL=Qwen/Qwen3-0.6B`): 667 passed, 41
skipped; the transformers-5.18 files (glm5_next, qwen4_exp, linear_serving, dsa_topk, dsa_select, moe_ep,
dp_attention, mtp_mla, mla, mixed_batch) plus test_padded_rows.py: 283 passed, 8 skipped (logs s3
logs/kiln-det-ci2/suite-c1.log, suite-c1-518.log).

**Eager copies into device memory do not wait for queued calls: the host tier's restores (ModelRunner._settle).**
Replays launched 4 ahead WITH a host-to-device fill of the decoded slot before each call (`RA:fill_cur`,
`RA:fill_page`) moved the request row in 69 / 58 of 1000 calls on the fixed tree, the same replays synchronous in 0 of
3000, and served decoding under overlap over reused pages (det10 X reruns after Y, V, Zc: 768 overlapped decode steps
over other requests' data) is exact. The fills, not the decode, were out of order. `OW@3:n` launches X_3's decode
call (which writes the decoded slot of every paged cache), then WITHOUT waiting copies seeded data into that slot from
the host (`ModelRunner.fill_slots`, `c[s:e].copy_(host_tensor)`), then waits and reads the slot back: in program order
it must hold the copied data, and it held the decode's write in **200 of 200** calls (det11) and 100 of 100 (det12-red);
the control with the call waited for first (`OWs`) 0 of 100. So an eager copy from the host is not queued behind the
graph calls already launched on the core; it lands right away. The engine's own such copies are the host tier's
restores (`_kv_load`, `_state_load`), made while scheduling the next step with the previous one in flight, and a
restore can be handed a page that step still writes: the tail page of a request that finished in the step before,
freed at its commit while its overlapped extra decode (launched before the finish was known) writes the request's last
position into it. That write then lands inside another request's restored prefix. `OWh@3:n` is exactly that copy (a
saved page restored with `ModelRunner.kv_load` into the page the just-launched decode writes): overwritten by the
earlier call in **100 of 100** on 0a889fb (det12-red), **0 of 200** with `_settle` (det12): `_kv_load` / `_state_load`
first read back the output of the last call launched on the rank (calls execute in submission order), once per batch
of restores. Only the host tier (`--hicache-host-gb`) makes these copies; the probe tool's own fills and zeroing do
the same and say so. Logs s3 logs/kiln-mimo-trn1/det11.log, det12-red.log, det12.log (2026-10-05).

**The same shape under an FP8 KV cache: padded rows wrote a real sequence's pool keys (models/mla.py
`_pool_key_slots`).** With the separate pool-key cache (`KILN_DSA_POOL_CACHE=auto` under `--kv-cache-dtype fp8`, the
G64 serving configuration), `write_pool_keys` recomputes the key of every written token's pool through the block table,
padded rows included. A padded row sits at position 0, so in a chunk's padded tail, whose table is the real
sequence's, it rewrote that sequence's pool-0 key slots beside the real rows that own them, from whatever those rows
held when it read them; in decode every padded row wrote pool 0 of the null page, i.e. slot 0 again. Now a padded row
(`slot_mapping < page_size`) writes only its own null-page slot. This changes the separate-cache DSA graphs (every
FP8-KV configuration; the bf16 in-place graphs are untouched), so it rides the farm's next recapture and has been
checked on the host only: `tests/test_padded_pool_keys.py` gives the padded tail rows different source data from the
key pool 0 holds and checks that key is untouched, and checks the invariant above (null-page slot 0 never written,
the pool-key cache included) under DP attention: both failed on the host-only fix (0a889fb's code) and pass with it
(kiln-det-ci2, m7i.8xlarge, transformers 5.18, 2026-10-05). CPU suite with it (b73766b, the same code before a
rebase): 667 passed, 42 skipped; the transformers-5.18 files plus both padded-row files 285 passed, 8 skipped (logs s3
logs/kiln-det-ci2/suite-c2.log, suite-c2-518.log). Device check on the final graphs (2026-10-05): engine-v0 ebe237e, farm queue q/final-ebe237e, the G64
configuration's env and argv verbatim (`--kv-cache-dtype fp8`, separate pool-key cache), kiln-mimo-trn1, real weights,
`tools/check_cold_determinism.py --text-file wikitext2_test.txt --prefixes 6 --assert-stable --ops
"reset,X,X,Y,V,Zc,X,reset,X,RD@3:1000,RA@3:1000,RD@3:1000:fill_cur,OWh@3:200"`: every X rerun 6/6 exact (tokens, chosen
and top-5 logprobs, 0.0 difference), including after the interval-checkpoint phase and after reset; the decode replay at
position 8211 0 of 1000 calls differing synchronously and 0 of 1000 with 4 calls in flight, on all 64 output rows;
0 of 1000 with foreign data in the decoded slot; the host-tier copy overtaken in 0 of 200; `STABLE True`, 0 device
compiles (logs s3 logs/kiln-mimo-trn1/det-fp8-ebe237e.log, .json).

## Where a GLM-5.3-Flash decode step goes, from a device profile, and three decode kernels (2026-10-04, SDK 2.32, trn1.32xlarge)

All on kiln-g2-trn1 (trn1.32xlarge, SDK 2.32, neuronx-cc 2.27.5334, nki 0.6.0), GLM-5.3-Flash real weights, tp=32, DP
attention 4 (attention TP 8), `KILN_CC_ARGS=--model-type=transformer`, NKI MoE / DSA-selection / KDA-prefill kernels,
12-layer decode graphs, page bucket 264 (8448-token context), every graph from the compile farm (queues
q/decode-b4f400f, q/decode-e2c6fad, q/decode-2968ed3; `NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`).

**The decode cost curve** (`python tools/time_decode.py --all-buckets --steps 64 --skip 8 --pieces --price trn1.32xlarge-spot=2.15 --
<serve_sweep args> --max-num-seqs 256 --kv-cache-gb 0.5 --kv-cache-dtype fp8 --no-prefix-caching --decode-buckets
1,4,8,16,32,64`; with --all-buckets time_decode times every bucket in one engine, and --price prints a decode-only cost line per bucket; the KV
rows are the null page and the tokens random, so the work is a step's at that bucket; wall p50 of 64 synchronous steps;
$ per 1M output tokens = $2.15/h over rows / step; logs s3 logs/kiln-g2-trn1/20261004T195733Z-time_decode.log (TP),
td-DCE-G12.log (EP)):

| rows per group (per step) | TP, engine-v0 b4f400f (dedupe MoE kernel) | $ / 1M out | EP, engine-v0 e2c6fad (`KILN_MOE_EP=1`) | $ / 1M out |
|---|---|---|---|---|
| 1 (4) | 54.4 ms | 8.12 | 96.9 | 14.48 |
| 4 (16) | 80.3 | 3.00 | 112.9 | 4.21 |
| 8 (32) | 112.2 | 2.09 | 143.6 | 2.68 |
| 16 (64) | 161.5 (the G64 config's own graphs: 160.9) | 1.51 | 192.7 | 1.80 |
| 32 (128) | 251.3 | 1.17 | 277.1 | 1.29 |
| 64 (256) | 466.3 | 1.09 | (Allocation Failure with six buckets' EP graphs loaded) | |

So the step is ~50 ms plus 1.4-1.7 ms per row of the step, and bigger batches alone level off near $1.1 / 1M (the list
price is $0.50). Expert parallelism, a prefill win in serving, costs the decode step +19% at 16 rows per group (~0.75 ms
per MoE layer) and +78% at 1.

**The HBM floor.** Per rank per step the step must read the dense weights once (2.00 GB: the KDA projections at
attention TP 8 1.17 GB, the router 0.25 GB and the DSA indexer 0.16 GB replicated on every rank, the DSA projections,
shared experts, lm_head, mHC; from the checkpoint's safetensors headers), the routed experts it touches (9.91 GB at
full coverage, 1 - (1 - 8/288)^rows of them: 84% at 64 rows) and per row of its group 77 MB of KV (the fp8 latent and
indexer rows of 8448 tokens in 11 DSA layers, read whole by the mask form) plus 36 MB of KDA state read and written. At
410 GB/s per NeuronCore (neuron-explorer 2.32's own `hbm_ddr_bandwidth`; the Neuron docs give 820 GiB/s per
2-core Trainium device, docs.aws "Trainium architecture"; the EC2 trn1 page gives 9.8 TB/s per trn1.32xlarge, 306 GB/s
per core) that is 7.7 / 14.8 / 21.5 / 29.5 / 37.2 / 46.6 ms for the six rows of the table: the measured step is 5-10x
the floor. In-step DMA reached ~240 GB/s (the MoE kernel's expert loads, below).

**Where it goes.** A 32-rank graph can be profiled whole: `neuron-explorer capture -n graph.neff -s g.ntff -r 32 -i 0
--ignore-exec-errors` replays the NEFF on all 32 cores with every input zero and profiles rank 0 (the prefill agent's
recipe), `neuron-explorer view ... --output-format json` (HOME must be set) gives every instruction, DMA and
collective, and `tools/prof_step.py ntff.json --layers KKKDKKKDKKKD` cuts the graph at its collectives into per-layer
segments (each layer holds two all-reduces: the token mixer's and the MLP's), with each engine's busy time, the DMA
bytes by queue and the HLO ops with most time (HLO shapes from graph.hlo). Zero inputs make MoE routing pick the same
experts for every row, so routed-expert time is not the real one (under EP it is the worst case: one rank holds them
all). The layers 12-23 decode graph (9 KDA + 3 DSA layers, 12 MoE), TP, per segment mean:

| rows per group | KDA attention segment | FFN segment (mix-out, mHC, router, MoE kernel, shared expert) | DSA attention segment | graph tail (9 KDA state writes) | graph in the run (p50) |
|---|---|---|---|---|---|
| 1 | 0.48 ms | 0.28 | 1.25 | 0.21 | 14.0 |
| 4 | 0.57 | 0.65 | 1.66 | 0.37 | 20.2 |
| 8 | 0.63 | 1.05 | 2.60 | 0.68 | 28.9 |
| 16 | 0.78 | 1.47 | 4.12 | 2.75 | 42.8 |
| 32 | 1.43 | 2.08 | 6.07 | 7.72 | 66.7 |
| 64 | 2.26 | 4.10 | 11.95 | 13.76 | 124.6 |

(The KDA column includes ~0.2 ms per segment of an idle ~1.9 ms at the replay's start.) At 16 rows: the 24 all-reduces
of [64, 4096] bf16 cost 35-70 us each, 1.5 ms of the 43.9 ms graph; collectives are not the cost. The FFN segment is
0.35 ms of hyper-connection arithmetic (over all 64 rows, replicated on every rank) and ~1.05 ms of MoE kernel (tensor
engine busy 0.75 ms; expert DMA ~240 GB/s); the DSA segment is ~0.9 ms of projections (0.40 ms of it dequantizing the
FP8 q_a/kv_a, q_b and o_proj weights to f32 every step) and ~2.7 ms of attention over all 8448 keys, which gathers
the fp8 latent, converts it to bf16 and spills 0.6-0.9 GB per layer (`GpSimd add f32[16, 1, 8448]`, the mask add, 2.2
ms of instruction time); the graph's tail is the 9 KDA layers' state updates, each written into the state pool through
1024 vector-engine STREAM_TRANSPOSEs (32 x 32 fp32, ~0.24 ms per layer) and a GpSimd scatter.

**KDA decode as one NKI kernel** (`kernels/kda_decode.py`, `KILN_KDA_DECODE_KERNEL=nki`, default xla): per sequence row
its states are one DMA from the pool row (dk on the partitions as the pool's row-major layout has it), q e, k e, k and
-v are column tiles made once by tensor-engine transposes, r0 / r1 - v and the row of k land on partition 0 by
one-column matmuls, delta and o are row ops there, k delta^T a matmul over one partition, S' = S e + k delta^T one
scalar_tensor_tensor, and the row goes back to the pool IN PLACE: the pool is returned as the kernel's output, an NKI
must-alias (an input returned as an output), and libtorch_neuronx_lite's functionalization replaces the input by it.
Rows that start fresh (padding, position 0) skip the read DMA (oob_mode skip on read row R) and use the zeroed tile.
`python tools/probe_kda_decode.py --batch 4 16 32 64` (one NeuronCore of kiln-g2-trn1, 8 heads of 128 x 128, a pool
of 65 rows): exact against the CPU emulation and the XLA form (o 1e-8, state 1e-6, other pool rows untouched); alone
0.21 / 0.35 / 0.49 / 0.96 ms per call at 4 / 16 / 32 / 64 rows against the XLA form's 0.17 / 0.33 / 0.43 / 0.91 (a
software pipeline over heads, stage A of head c before stage B of head c - 2, took it from 1.30 to 0.96 ms at 64; the
tensor engine's fp32 matmuls are 4 passes each and bound it). In the decode graphs, where the XLA form adds the
transposes: EP step 192.7 -> 180.9 ms at 16 rows per group, 277.1 -> 248.8 at 32 (td-DCEK-G12.log).

**Pooled DSA decode attention over the selected pools only** (`kernels/dsa_decode.py`, `KILN_DSA_DECODE_KERNEL=nki`,
default xla; `glm5_next.decode_slots`, `mla.decode_kernel_takes`): the decode step's selection is taken at pool level
(`dsa_topk.select(..., kp=1, tail=False)`, the same 512 of 2112 pools), compacted to pool rows of the cache viewed as
[N / 4, 4 x 512] (pools never straddle a page) plus the query's own incomplete pool, 640 slots per row (5 x 128) with a
0 / NEG_INF bias per token; the kernel gathers each 128-slot chunk with one indirect DMA (vector_offset, one pool row
per partition), transposes the latent on the tensor engine (against a bf16 identity), scores q_lat against it, softmaxes
per head (fp32 scores; the mask form rounds them to bf16 first) and accumulates p K with the gathered fp8 latent as the
moving operand. Equivalent to the mask form: with NEG_INF on every unselected key its softmax is this one.
`python tools/probe_dsa_decode.py --batch 4 16 32 64 --kv fp8` (one NeuronCore, 8 heads, latent 512, 264 pages per row):

| rows | XLA mask form (all 8448 keys) | the kernel (2560 gathered tokens) |
|---|---|---|
| 4 | 1.343 ms | 0.227 ms |
| 16 | 5.726 | 0.733 |
| 32 | 11.424 | 1.411 |
| 64 | 23.537 | 2.765 |

with the output within bf16 rounding of the emulation (2.2-2.5e-3 of max |o|, the mask form 2.3-2.7e-3). CPU:
`tests/test_glm5_next.py::test_dsa_decode_kernel_path_matches` (the path through decode_slots and the emulation gives
transformers' greedy tokens and the mask path's logprobs within 1e-5) and `::test_dsa_decode_simulator_matches_emulation`.

Three things the path needed, each measured. (1) `nisa.nc_matmul` takes trn1's legacy float8_e4m3, not e4m3fn: the
SBUF tile of an fp8 cache is declared float8_e4m3, the graph compiled with --experimental-unsafe-fp8e4m3fn-as-fp8e4m3
(an e4m3fn tile beside it fails NCC_EOCP001, "Mixed use of the two mutually-exclusive types"); an in-graph
.view(torch.uint8) of the cache is refused by LNL ("Expected all tensors ... XLAByteType"). (2) An indirect DMA reads
within one row of its tensor: a vector_offset gather of 4 consecutive token rows of a [N, 512] cache read the wrong
data; viewing the cache as [N / 4, 2048] pool rows made it exact. (3) A comparison against a float literal in the
compaction (`mask > NEG_INF / 2`) failed the whole decode group graph with NCC_ESPP004 "f64 dtype is not supported":
the 0/1 selection is exp of the 0 / NEG_INF mask instead. The compaction itself: the int64 scatter of each selected
pool to its rank cost 5.2 ms per layer at 16 rows on the GpSimd engine (the profile's `scatter s64[16, 513]`), the
same count in int64 16.1 ms, and the k-th selected pool as an fp32 count of the inclusive selection count <= k 0.90 ms
(2.43 ms at 64 rows; `/opt/kiln/prof/probe_compact.py` on kiln-g2-trn1, one NeuronCore). Two levels (which group of 8
pools holds the k-th, from the groups' last counts; then the count inside that group, gathered) took it to 0.40 ms at 16
rows (`glm5_next.decode_slots`). In the decode step it is worth less than that: 185.0 -> 183.3 ms at 32 rows per group and
235.2 -> 233.7 at 64 (DCT-DKS against DCT-DKS2, below); the count overlaps other engines' work.

**A PSUM accumulation hazard on trn1** (`tools/probe_nki_scores.py <mode>`, kiln-g2-trn1, one NeuronCore, nki 0.6.0):
20 blocks of scores, each the sum of four matmuls (accumulate=False, then True three times), four consecutive blocks
written into the four 128-column slices of one [8, 512] fp32 PSUM tile and the tile read after the fourth (mode 3):
nki.simulate is exact (7e-7); on the device block 4, the first block of the second tile, holds only its last term
(equal to the lc = 3 product alone, max |err| 3.44), the other 19 exact. Full-partition [128, 512] tiles (mode 4) and a
3-deep ring (mode 5) fail the same way; one block per PSUM tile, read right after its group (mode 0), and each term in
its own slice summed on the vector engine (mode 1) are exact, also with the rows gathered by indirect DMA (mode 2). The
DSA decode kernel uses mode 0. Before that its scores were wrong for blocks that changed with every code edit, while
the simulator was exact: compare a kernel's device output with nki.simulate block by block.

**Sequence-parallel decode streams** (`KILN_DECODE_SP=1`, default 0): the prefill's sequence-parallel hyper-connection
streams for decode calls whose N B rows divide over tp (each rank keeps N B / tp rows between layers, each block gathers
and reduce-scatters, the post graph gathers the final hidden state for the lm_head). CPU
(`tests/test_dp_attention.py::test_glm5_next_sequence_parallel_decode_streams`): greedy tokens equal to the replicated
streams at tp 4 / DP attention 2 and at tp 2, logprobs within 2e-6.

**The decode step with the kernels, in the graphs** (engine-v0 2968ed3 + this branch, EP on as the engine defaults
from 8 rows per group, `python tools/time_decode.py --all-buckets --steps 64 --skip 8 --pieces --price trn1.32xlarge-spot=2.15 --
<sweep args> --max-num-seqs 256 --kv-cache-gb 0.5 --kv-cache-dtype fp8 --no-prefix-caching --decode-buckets 32,64`,
farm queue q/decode-2968ed3 configs DCT-BASE / DCT-DK / DCT-DKS / DCT-DKS2, kiln-g2-trn1 2026-10-04 23:55 - 10-05 00:39
UTC, logs s3 logs/kiln-g2-trn1/td-DCT-*.log; wall p50 of 64 synchronous steps, $ per 1M output tokens decode-only at
trn1.32xlarge spot $2.15/h):

| rows per group (per step) | base | + KDA and DSA decode kernels | + SP decode streams | + two-level compaction |
|---|---|---|---|---|
| 32 (128) | 279.3 ms, $1.30 | 225.4 ms, $1.05 | 185.0 ms, $0.86 | **183.3 ms, $0.86** |
| 64 (256) | 456.5 ms, $1.07 | 270.2 ms, $0.63 | 235.2 ms, $0.55 | **233.7 ms, $0.55** |

Per 12-layer group graph (the `--pieces` pass, rank 0, synchronous): at 64 rows the base's 125 / 123 / 122 / 89 ms
(groups 0-3; group 3 holds 9 layers) become 75 / 72 / 71 / 52 with the kernels and 64 / 62 / 61 / 45 with SP decode
streams; at 32 rows 79 / 76 / 73 / 54 -> 65 / 62 / 59 / 44 -> 52 / 50 / 46 / 35. The kernels take 41% off the step at 64
rows and 19% at 32, SP decode streams another 13-18% (the hyper-connection arithmetic over all rows on every rank is
gone: each rank keeps rows / 32). What is left grows by 1.6 ms per row of a group (50 ms from 32 to 64 rows): the step
at 64 rows per group, 256 sequences, is now $0.55 per 1M output tokens decode-only against the $0.50 list price.

**Serving A/B** (bench/serve_sweep.py, 128 requests, EP on as the engine defaults, every graph from q/decode-2968ed3 with
`NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`, the merged tree feat/decode-step a64d7f8 = 7db28da + engine-v0 06ce6d5, whose G64 keys
are set-equal to the queue's for all three configs; each run's command in its `.log.cmd`):

| config | box, log (s3 logs/<box>/) | out tok/s | TTFT p50 / p90 | ITL p50 | spot $ / 1M out | at cost, output (decode calls) |
|---|---|---|---|---|---|---|
| conc 64, G64-EPT (base) | kiln-dk-32 ab-G64-EPT.log | 114.9 | 7.6 / 87.1 s | 487 ms | $5.20 | $2.08 |
| conc 64, + KDA and DSA decode kernels | kiln-dk-32 ab-G64-EPT-DK2.log | 122.2 (+6.4%) | 7.3 / 84.8 s | 460 ms | $4.89 | $1.75 |
| conc 64, + SP decode streams | kiln-dk-32 ab-G64-EPT-DKS2.log | **128.0 (+11.4%)** | 7.2 / 82.6 s | 442 ms | **$4.67** | $1.55 |
| conc 32, F0-EPT (base) | kiln-g1-trn1 ab-F0-EPT.log | 106.0 | 7.1 / 30.8 s | 269 ms | $5.63 | $2.49 |
| conc 32, + KDA and DSA decode kernels | kiln-g1-trn1 ab-F0-EPT-DK2.log | 108.0 (+1.9%) | 7.0 / 30.7 s | 264 ms | $5.53 | $2.37 |

At conc 32 a decode call holds 8 rows per group, where the kernels' share of the step is small (the KDA tail and the DSA
mask form grow with rows). At conc 64 prefill is 65-74% of the device time (the device split in docs/price-performance.md
"G1b"), so the decode calls' -16% with the kernels and -25% with SP decode streams too (at cost $2.08 -> $1.75 -> $1.55
per 1M output) come out as +6.4% and +11.4% end to end.

**Correctness on real weights.** Prefill is untouched: under the kernels 12 of the G64 config's 15 graph keys are the
base's, the 3 others its decode group graphs (SP decode: 5 others, the pre and post decode graphs too), so check_ppl,
which scores prefill, gives the base's 4-sentence -2.074 and wikitext -0.551 by construction. The decode path:
`tools/check_mixed.py` on the G64 serving graphs (32 prompts of check_ppl's LONG_TEXT or of wikitext-2, 8192 to 700
tokens, 64 greedy tokens, top-2 logprobs; `--compare` teacher-forces every position before a pair's first difference,
so the signed mean of the chosen tokens' dlogprob over decode calls IS the change in decode-path NLL; kiln-g2-trn1,
logs s3 logs/kiln-g2-trn1/cmg-*.log and cmg-cmp-*.log; the LONG_TEXT floor on kiln-dk-32 cm-cmp-base-base2):

| text | comparison | outputs equal | decode-path NLL change (signed mean, nats / token) | mean / p99 / max \|dlogprob\| | reference margin at each first difference |
|---|---|---|---|---|---|
| LONG_TEXT | base vs base (run to run) | 30 / 32 | +0.00001 | 0.0015 / 0.049 / 0.24 | |
| LONG_TEXT | base vs decode kernels | 29 / 32 | **+0.00028** | 0.0028 / 0.072 / 0.39 | 0, 0.25, 0.125 (the three 700-token prompts) |
| wikitext-2 | base vs base (run to run) | 18 / 32 | -0.00079 | 0.0145 / 0.13 / 0.34 | 14 flips, at most 0.5 |
| wikitext-2 | base vs decode kernels | 10 / 32 | **+0.00015** | 0.026 / 0.23 / 0.45 | 22 flips, at most 0.5 (0.5 twice, 0.375 twice, the rest 0.31 or less) |
| LONG_TEXT | base vs decode kernels + SP decode streams | 30 / 32 | **+0.00010** | 0.0021 / 0.062 / 0.27 | |
| wikitext-2 | base vs decode kernels + SP decode streams | 6 / 32 | **-0.00066** | 0.028 / 0.22 / 0.54 | 26 flips: 0 (9), 0.125 (7), 0.25 (4), 0.3125, 0.375 (2), 0.5 (2), 0.625 (one, prompt 8000 output 56) |

No bias: the decode-path NLL does not move (+0.0003 / +0.0002 nats per token with the kernels, +0.0001 / -0.0007 with SP
decode streams too, against the run-to-run floor's own +0.0000 / -0.0008), and the flips sit at the floor's margins (one
at 0.625 with SP decode streams, where the floor's largest is 0.5; the sampler's logprobs move in steps of 0.125). The per-token jitter is about twice the run-to-run
floor's: the kernels sum the same selected keys in a different order, keep the DSA scores in fp32 (the mask form rounds
them to bf16 before the softmax) and run the KDA state update as fp32 matmuls on the tensor engine.

**Defaults (2026-10-05): on for trn1.** `KILN_KDA_DECODE_KERNEL` and `KILN_DSA_DECODE_KERNEL` default to `nki` and
`KILN_DECODE_SP` to on for a trn1 target (kernels/kda_decode.py, kernels/dsa_decode.py `DECODE_KERNEL_FAMILIES`,
models/decoder.py `DECODE_SP_FAMILIES`), and stay `xla` / off on trn2, inf2 and a host without a Neuron device; each
variable overrides it (`tests/test_glm5_next.py::test_decode_kernel_defaults_are_trn1_only`). SP decode streams still
turn themselves off where the decode buckets x DP attention do not divide over tp (G16: 4 x 4 rows over 32), with an MTP
head, and only glm5_next uses them. The configs that were not measured above, measured for this (feat/attn-kernel-cand,
every graph from the farm with `NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`; logs s3://<your-bucket>/logs/kiln-ak-33/ and
.../kiln-ak-32/, each with its `.cmd`; base = q/final-ebe237e, the same EP-on serving env as the tables above):

| config | base | + decode kernels (DK) | + SP decode streams (DKS) | box |
|---|---|---|---|---|
| G16 (conc 16, 4 rows per group) | 87.6 out tok/s, decode call 66.6 ms, ITL 149 ms | **88.1**, 62.9 ms, 148 ms | (off: does not divide) | kiln-ak-33 |
| F0 (conc 32, 8 rows per group) | 112.4, ITL 254 ms | 114.9 (+2.2%), 248 ms | **119.0 (+5.9%)**, 240 ms | kiln-ak-33 |

With the fused DSA prefill kernel as well (q/attnk-54ec5df graphs; no new key: the decode graphs are the ones above and
the prefill graphs the fused kernel's), kiln-ak-32:

| config | fused prefill kernel | + DKS | the candidate tree with no variable set |
|---|---|---|---|
| G64 (conc 64) | 132.0 / 131.9 out tok/s, decode call 178 / 181 ms, ITL 419 ms | **148.5**, 135 ms, 376 ms | **148.6**, 136 ms, 376 ms |
| F0 (conc 32) | 120.0, 118 ms, 239 ms | **127.2**, 108 ms, 225 ms | |
| G16 (conc 16; SP off, DK only) | | | **92.2**, prefill call 825 ms (base 893), ITL 142 ms |

So against the final tree's base, G64 122.9 -> 148.6 (+20.9%), F0 112.5 -> 127.2 (+13.1%) and G16 87.6 -> 92.2 (+5.3%)
out tok/s; the candidate
tree's plain G64 command, every graph a cache hit, reproduces the variables' run (and on ce333bd, before the decode
defaults, the farm's capture of the plain G64 / F0 / wt-DP4 configs was key-identical to the fused-kernel configs). The decode-path NLL of the decode
kernels and SP decode streams is the check_mixed table above (no change against the run-to-run floor); prefill graphs
are untouched by them, so check_ppl's wikitext is the fused kernel's -0.548. Not measured: trn2, mixed batches
(`KILN_MIXED_BATCH=1`), MTP.

**Prefill / decode disaggregation, an estimate (design only, nothing built).** What a request hands from a prefill box
to a decode box, from GLM-5.3-Flash's config.json (zai-org/GLM-5.3-Flash@eb9eb208): the KDA state of 34 layers, 64 heads
of 128 x 128 fp32 (4.19 MB per layer, 142.6 MB) plus the short-conv tails (kernel 4: 3 positions x 3 x 64 x 128 bf16,
0.15 MB per layer, 5.0 MB), and the DSA cache of 11 layers at ~830 B per token per layer (the fp8 latent of 512, the
indexer key, the pool key; the 77 MB per row per rank of "The HBM floor" above over 8448 tokens): 74.8 MB at 8192 tokens.
About 222 MB per request, 2.2 ms at trn1.32xlarge's 800 Gbit/s EFA (aws.amazon.com/ec2/instance-types/trn1), and at the
sweep's rate (conc 64: 0.45 requests/s) 100 MB/s: the transfer is not the problem. A decode-only box holds per row of a
DP group, per rank, 77 MB of KV (the latent is shared by all heads, so each of a group's 8 attention ranks holds it) and
18 MB of KDA state (8 of the 64 heads); with no prefill graphs (their rings and scratch are most of the graph HBM above)
about 4.8 GB per rank is left after 11.78 GB of weights and the decode graphs, ~50 rows per group with no slack, ~45 with
the 10% KV margin. At 48 rows per group the final step interpolates to ~208 ms (183.3 at 32, 233.7 at 64): ~920 out
tok/s per box, $0.65 per 1M output tokens decode-only at spot. Prefill boxes at the measured EP prefill call (0.746 s per
4096 rows, the prompt-cache device split in docs/price-performance.md) do 0.67 requests/s each, so one decode box needs
~5.4 prefill boxes: $0.00089 of prefill plus $0.00017 of decode per request, $4.13 per 1M output tokens all-in against the
colocated 114.9 out tok/s ($5.20). That gain needs ~190 sequences in flight per decode box: G1's closed loop at conc 64
keeps 16 rows per group however the boxes are split, so for G1 the step's own cost is the lever, and disaggregation is
for a server whose load fills a decode box.

**`KILN_DENSE_FP8=0`** dequantizes the checkpoint's FP8 weights outside the routed experts once at load (+0.26 GB per
rank in bf16: DSA q_a + kv_a 0.092, q_b 0.035, o_proj 0.092, shared experts 0.033, dense MLPs 0.009; from the headers),
removing the 0.40 ms per DSA layer of in-graph dequantization; CPU test `tests/test_mla.py::test_dense_fp8_dequantized_at_load`.
`tools/tensor_bytes.py` on the G64 serving config with the decode kernels: 14.325 GB of tensors per rank, 14.582 with
`KILN_DENSE_FP8=0` (+0.257 GB = 0.24 GiB; weights 11.784 -> 12.042, KV 1.487 and KDA state 1.054 unchanged). Not measured
on the device: the decode kernels came first, and they are what makes room for it (next paragraph).

**The decode kernels give HBM back.** `tools/hbm_estimate.py` over each configuration's compiled graphs (q/decode-2968ed3,
tensors 14.325 GB): the three decode group graphs of the base G64 config (EP on) carry 374 / 382 / 322 MiB of spill rings
and a 0.28 GiB scratchpad; with the KDA and DSA decode kernels and SP decode streams they carry 46 / 49 / 35 MiB and 0.006
GiB. Totals: base 18.40 GiB, decode kernels 17.55, decode kernels + SP decode 17.40, decode kernels + `KILN_DENSE_FP8=0`
17.79. The base loads and serves (114.9 out tok/s), so the estimator over-counts these EP configurations in absolute terms
(the EP prefill graphs alone hold 1359 / 1207 / 820 MiB of rings); the differences are what it measures: 0.85-1.0 GiB per
rank back, of which dense bf16 weights would take 0.24.

## MoE kernels against their floors (2026-10-05, SDK 2.32, nki 0.6.0, neuronx-cc 2.27.5334, trn1)

kiln-mk-k1 (trn1.2xlarge, one NeuronCore), feat/moe-kernel-tune on engine-v0 e449b98. GLM-5.3-Flash EP rank shapes: 9
whole experts (fp8 gate_up 4096 x 4096, down 2048 x 4096, 128 x 128 block scales fitted per block: the tile-scale form),
random values; routing = the real routers on the sweep's random-token prompts (`tools/ep_routing.py` data,
s3 logs/kiln-mimo-trn1/ep_routing.pt, the probe's default four 1024-token pieces), the busiest rank of the contiguous
placement (`tools/probe_moe_ep.py full --fit block --routing-file ep_routing.pt --layer L --rank -1`). Floors from
`tools/moe_roofline.py` (the kernels' own pass structure on that routing); profiles from neuron-explorer on each probe
graph's real inputs (`tools/prof_engines.py`, then the new `tools/prof_ops.py`: busy as the UNION of an engine's
instruction intervals, idle attributed to the setter of the semaphore the next instruction waits on, DMA by queue and
engine). Kernel times are p50 of synchronous calls minus the readback reduction, the same within 0.01-0.05 ms over two
processes unless stated.

**Expert-parallel prefill (kiln_moe_ep_kernel, C = 4096 rows)**

| layer | busiest rank pairs (largest expert) | passes, executed lanes | PE floor on pairs / on executed lanes (2.45 G cols/s) | HBM bytes, at 228 GB/s | measured |
|---|---|---|---|---|---|
| 3 | 1545 (665) | 11, 3072 | 0.99 / 1.97 ms | 386 MB, 1.69 ms | 2.79 ms |
| 20 | 5084 (4090) | 18, 6656 | 3.25 / 4.26 | 650 MB, 2.85 | 5.63 |
| 44 | 4876 (3132) | 18, 6656 | 3.12 / 4.26 | 650 MB, 2.85 | 6.56 (one process) |

(a pair or lane costs 1568 moving columns: gate_up 1024, down 512, x transposes 32.) The layer-20 profile: the tensor
engine is busy 83% of the 5.40 ms window and 85-89% inside every pass, at 2.33 G moving columns per second while busy,
83% of the 2.8 GHz PE clock; the profile's own throttle counters put the core's activity throttle at an 87.5% utilization
limit for 42% of the run (`throttle_activity_0_avg_util_limit_nc0_percent` 0.875), which is where the measured 2.45 G
cols/s ceiling of chained matmuls comes from. The vector engine is busy 62% (13824 fp8 dequantization TENSOR_SCALAR,
median 239 ns, bimodal 220 / 320 ns; standalone the same op is 120-150 ns), the scalar engine 54% (13824 dequantization
ACTIVATE, median 181 ns), DMA active 66% at 166 GB/s, every one of its 143,424 packets on the GpSimd software-DGE queue
and spread evenly over the 16 DMA engines. Static 256-lane first passes take 195 us each, the 512-lane device-loop
overflow passes 406 us: ~0.77 us per executed lane against 0.64 ideal. So the kernel is PE-bound on EXECUTED lanes; the
gap to its pairs floor is padding (6656 lanes for 5084 pairs, nearly all in the 256-lane first passes of six experts
with 0-42 pairs) plus ~11-15% PE idle per pass; and 3.25 ms of layer 20's 5.6 is ONE expert with 4090 of the rank's
5084 pairs, which only a placement change (redundant copies of hot experts) removes.

**Expert-parallel decode (kiln_moe_ep_small, 64 rows = 16 per DP group)**: layer 3 29 pairs on 8 used experts, 1.38 ms
against 0.88 ms of bytes (each used expert's 25.2 MB once at 228 GB/s); layer 20 82 pairs (largest 60) on 6 used
experts, 9 static + 3 overflow passes, 1.85 ms against 0.66; layer 44 83 pairs (largest 40), 1.89 ms. Profile layer 20:
12 passes of ~133 us, DMA active 74% at 202 GB/s; a static pass of an expert with no pair skips its loads but still
costs ~108 us of compute; each overflow pass reloads the whole expert. PE busy 46%, vector 59% (the PSUM-stack multiply
and reduce, ~80 us per pass).

**Tensor-parallel kernels (G16 only)**: moe_prefill C=4096 uniform routing, skip 20 (`KILN_MOE_PREFILL_SKIP=20 python
tools/probe_moe_prefill.py --format loaded --chunks 4096 --decode-max 0`): 9.17 ms against a PE floor of 1.7 ms and
1.5 GB of HBM traffic (x rows in, Y out, the combine's gather), 6.6 ms at 228 GB/s; moe_dedupe (`tools/probe_moe_kernel.py
--experts 288 --scales block128 --act silu_clamp --kernels dedupe`): 16 rows 0.64 ms (105 distinct experts, 86 MB, 0.38
ms of bytes), 64 rows 1.44 ms (244, 200 MB, 0.88 ms).

**The HBM -> SBUF ceiling of one core** (`tools/probe_mk_dma.py`, 9 x 25.2 MB streamed through 2-4 ring buffers; GB/s
over the call minus the null graph, and while DMA was active in the profile): [128, CH] DMAs of CH = 8, 16, 32 or 64 KB
per partition, static or dynamic offsets: 261-264 GB/s (272-275 while active, 16 DMA engines at ~17 GB/s each; packets
of 8 KB and of 32 KB alike); four [32, 8 KB] DMAs per chunk 196 GB/s. So ~272 GB/s, 66% of the 410 GB/s the profiler
names, is the practical weight-stream ceiling; the kernels reach 160-204 GB/s while active.

**NKI device loops put a full all-engine barrier at every iteration.** In the profile of a `nl.fori_loop` the iteration
ends with every engine (Sync, GpSimd, Vector, Scalar, Tensor) meeting on one semaphore (`$S[2]==N` across all queues),
then a COMPARE_BRANCH on each, then fresh instruction fetches (qScalarTable / qDveTable DMAs of 16 KB, queue_type
"instruction"); nothing of iteration i + 1 starts before iteration i has fully drained. A loop pass therefore pays its
pipeline fill and drain every time: from the barrier to the first gate_up matmul of a small-lane pass ~20 us (table
read, lane tokens, the x-row gather, transposes), and no DMA of the next pass overlaps the current one's compute. A
small-lane pass costs ~164 us in a loop against ~135 us static. Inside a larger graph the barrier also stops the
compiler overlapping neighbouring ops with the kernel's loop, which may be part of why kernel speedups measured alone
did not carry into the serving graphs before; the in-graph runs below are the test.

**Decode v2 (kiln_moe_ep_small2, opt-in `KILN_MOE_EP_SMALL_V=2`, 5a6249c).** One pass per local expert WITH pairs and
none for an expert without: an expert with at most 16 pairs takes one small-lane pass (kiln_moe_ep_small's arithmetic),
one with more one 128-lane dequantize-first pass (kiln_moe_ep_kernel's), each kind in its own device loop over a table
the plan builds (`_plan_s2`), so no expert's weights are read twice. Per token the small passes add first, then the
dequantized ones (`emulate(..., small=2)`, `tests/test_moe_ep.py::test_small2_emulation_mixes_the_two_arithmetics`).
v1's source is untouched (its graphs keep their keys). Busiest rank, ms, v1 -> v2 (`KILN_MOE_EP_SMALL_V=1|2 python
tools/probe_moe_ep.py full --fit block --chunks 64 128 32 --routing-file ep_routing.pt --layer L --rank -1`, two
processes each, identical to 0.04 ms):

| rows per group (C) | layer 3 | layer 20 | layer 44 |
|---|---|---|---|
| 8 (32) | 1.38 -> **1.08** | 1.47 -> **0.92** | 1.49 -> **0.91** |
| 16 (64) | 1.38 -> **1.08** | 1.85 -> **1.25** | 1.89 -> **1.42** |
| 32 (128) | 1.55 -> **1.39** | 2.60 -> **1.22** | 2.81 -> **1.58** |

Kernel against its emulation 0.0019-0.0038 of the output's max (v1 0.0019-0.0065). The v2 profile (layer 3, 64 rows: 5
small passes) shows the loop cost: 164 us per pass at 160 GB/s of DMA, against 93 us for its 25.2 MB at the 272 GB/s
ceiling and ~90 us of vector work per pass.

**Decode v2 in the serving graphs** (kiln-mk-32, trn1.32xlarge, SDK 2.32, GLM-5.3-Flash real weights, tp=32, DP
attention 4, feat/moe-kernel-tune 5a6249c, every graph from the compile farm q/moek-trn1 with
`NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`; `python tools/time_decode.py --all-buckets --steps 64 --skip 8 --price
trn1.32xlarge-spot=2.15 -- <the DCT argv: --max-num-seqs 256 --concurrency 256 --kv-cache-gb 0.5 --kv-cache-dtype fp8
--no-prefix-caching --decode-buckets 8,16,32>` with the serving env plus the variant's switch; wall p50 of 64 steps,
launch to logits; logs s3 logs/kiln-mk-32/*-td-{V1,V2,TP}.log with their .cmd):

| rows per group | TP (`KILN_MOE_EP=0`) | EP v1 (engine-v0) | EP v2 (`KILN_MOE_EP_SMALL_V=2`) | v2 against v1 | EP over TP, v1 -> v2 |
|---|---|---|---|---|---|
| 8 | 112.2 ms | 138.7 | **124.7** | -14.0 ms (-10.1%) | +26.5 -> +12.5 ms |
| 16 | 161.0 | 188.0 | **177.3** | -10.7 (-5.7%) | +27.0 -> +16.3 |
| 32 | 252.2 | 280.8 | **258.5** | -22.3 (-7.9%) | +28.6 -> +6.3 |

So the kernel's standalone gain carries into the decode step (unlike the interleave and the 64-lane passes before it),
and halves expert parallelism's decode penalty; the remaining gap is the busiest rank's bytes and the loop passes'
fill and drain.

**v3 (`KILN_MOE_EP_SMALL_V=3`, tile-scale layouts):** v2 with a small pass's weights in 12 DMAs instead of 48 (the
whole down projection in one 64 KB-per-partition DMA, gate_up two I-chunks per DMA through a ring of three, the tile
scales broadcast to every partition, one register for the expert offset). Bit-identical to v2 (the same values differ
from the emulation, element for element). Standalone 3% faster: busiest rank 1.21 / 1.05 / 1.38 ms at 64 rows (layers
20 / 3 / 44) against v2's 1.25 / 1.09 / 1.43; 0.89 / 1.04 / 0.88 at 32 rows against 0.92 / 1.08 / 0.91. Its profile
(layer 3, 64 rows): 155 us per small pass against v2's 163; DMA 223 GB/s while active in 16 / 64 KB packets, but active
only ~123 us of each pass and the tensor engine waiting ~61 us per pass on it: within one loop pass the down weights
cannot stream under any compute (they are needed last), so the pass is the sum of its gate_up phase and its down phase.
v3 in the decode step (TD-SMALLV3 graphs, the same box and method): 127.9 / 175.7 / 260.1 ms at 8 / 16 / 32 rows per
group against v2's 124.7 / 177.3 / 258.5: the same within run-to-run noise.

**Measured and dropped: the down weights per chunk (v4, not kept).** Eight [128, M, 512] down DMAs queued after the last
gate_up one, so that the down matmuls of chunk q wait only for their bytes: no faster (busiest rank within +-3% of v3),
because a [128, 16, 512] source is 16 runs of 512 B per partition and moved at ~160 GB/s against 275 for the whole
contiguous 64 KB-per-partition block, so the down phase was paced by its DMA instead of by its compute.

**v5 (`KILN_MOE_EP_SMALL_V=5`, tile-scale layouts): two small passes per loop iteration.** The small table is split by
strided HBM -> HBM copies into entries 0, 2, .. and 1, 3, .. (a loop register cannot be stored to SBUF on trn1:
`register_store` fails NCC_IXCG832 "TensorSave destination must be DRAM on trn1, due to a HW bug"), the pair loop runs
n // 2 iterations of two _pass_s5 each (the whole-down DMA issued after the last gate_up DMA), a second loop n % 2 times
for the odd one; both passes' scatter-adds at the end of the iteration. Bit-identical to v2 (same elements differ from
the emulation in all nine cases). Busiest rank, ms, v3 -> v5: 64 rows 1.22 / 1.05 / 1.39 -> **1.13 / 0.98 / 1.32**
(layers 20 / 3 / 44); 128 rows 1.21 / 1.34 / 1.59 -> 1.16-1.18 / 1.25-1.26 / 1.50; 32 rows 0.89 / 1.03 / 0.89 -> 0.86 /
0.97-0.98 / 0.86. Against v1 at 64 rows: 1.85 / 1.38 / 1.89 -> 1.13 / 0.98 / 1.32. Its profile (layer 3, 64 rows): ~275
us per pair of experts against ~310 for two v3 passes; the second expert's gate_up DMAs still do not stream under the
first one's down phase, because the compiler gives both passes the same SBUF buffers (the second pass's loads wait on
the tensor engine releasing the first pass's: `S[5] (Tensor)>=80` before its first DMA), and deferring both scatter-adds
to the end of the iteration changed nothing (0.986 against 0.983 ms).

**Measured and dropped: the pair software-pipelined by hand (v6, not kept).** The first expert's down chunks interleaved
one for two with the second expert's gate_up chunks in program order, the second's down weights loaded into the first
one's space after the interleave: the same within 0-4% of v5 at every layer and size (64 rows 1.15 / 1.02 / 1.33 ms at
layers 20 / 3 / 44). Its profile (layer 3, 64 rows) shows why: a pair is still ~280 us, of which ~75 us the first
gate_up (DMA-bound), ~25 us waiting for the first down weights, ~90 us the interleaved phase (vector-bound: 35 us of down
plus 55 of gate_up), ~25 us waiting for the second down weights, ~40 us the second down phase and ~25 us of loop tail and
barrier. A small pass streams its 25.2 MB at the ~270 GB/s ceiling and its down phase can only start once all of its down
bytes are in (16 runs of 512 B per partition per chunk move at ~160 GB/s, so splitting them costs more than it saves), so
with one in-order DMA queue a loop iteration of k experts costs about k x 93 us of DMA plus ~65 us of down phase and
tail that nothing can hide; the loop barrier forbids hiding them under the next iteration. That puts v5's ~137 us per
small expert within ~10% of what this structure allows (~125 us at k = 2).

**Prefill: the padding fix is not worth a recompile.** Over all 42 MoE layers' busiest ranks (4096 rows, the cost model
of the measured passes: static 256-lane first pass 195 us, a device-loop first pass ~215, a small-lane pass ~140, 512-lane
overflow 406, 256-lane tail 211), skipping empty experts and giving tiny ones (<= 16 pairs) small-lane passes in device
loops saves 5.7 ms per 4096-token prefill call on random-token routing (180.7 -> 175.0 ms of busiest-rank kernel time,
32 empty and 85 tiny experts over the 42 layers) and loses 2.2 ms on wikitext routing (140.7 -> 142.9: 4 empty, 60 tiny),
because a device-loop pass pays its fill and drain where a static one overlaps its neighbours. A rank's static first
passes are saturated across engines (tensor 89%, vector 80%, scalar 67% busy in the profile): 256 lanes are 164 us of
tensor-engine streaming and the expert's 1536 dequantized tiles ~190 us of vector + scalar, so a smaller first pass is not
cheaper either. On the serving graphs (utilization agent, prefill piece 1, real routing, every rank's own inputs) a
light rank's MoE segment is 2.55-2.6 ms per layer and the busiest 4.8-5.2, and the world reduce-scatter mirrors it
(MoE + RS = 5.56-5.59 ms on all four group leaders): the kernel-side lever left in prefill is the busiest rank's load, i.e.
expert placement, not the kernel.

**Decode v2 in serving** (kiln-mk-32, trn1.32xlarge spot, SDK 2.32, GLM-5.3-Flash real weights, tree 4c3d078 = engine-v0
e449b98 + kernels/moe_ep.py additions; the q/final-ebe237e G64 and F0 commands verbatim (`--requests 128` / `64
--max-seconds 3000`), V1 graphs from q/final-ebe237e and V2 (`KILN_MOE_EP_SMALL_V=2`) from q/moek-trn1, every run
`NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`, back to back on one box; logs s3 logs/kiln-mk-32/*-sv-{G64,F0}-{V1,V2}.log with
their .cmd):

| config | out tok/s V1 -> V2 | decode call | prefill call | TTFT p50 / p90 | ITL p50 | spot $ / M out |
|---|---|---|---|---|---|---|
| G64 (conc 64, 16 rows per group) | 122.8 -> **129.5** (+5.5%) | 0.178 -> 0.158 s | 0.592 = 0.592 s | 6.9 / 79.7 -> 6.8 / 76.3 s | 453 -> 436 ms | $4.86 -> **$4.61** |
| F0 (conc 32, 8 rows per group) | 110.1 -> **117.1** (+6.4%) | 0.125 -> 0.108 | 0.594 -> 0.592 | 6.5 / 39.6 -> 6.4 / 38.4 | 254 -> 239 | $5.42 -> **$5.10** |

V1 reproduces the final standings (122.9 / 112.3) within 0.1 / 2%. The prefill call is untouched, as it should be (only
the decode group graphs differ). The A/B's noise, from a bit-identical twin: v3 (`KILN_MOE_EP_SMALL_V=3`, the same
outputs as v2 element for element) on the same box right after gave G64 129.8 and F0 116.0 out tok/s (decode call 0.157 /
0.108 s), i.e. ~1% run to run, against the +5.5% / +6.4% measured.

**Future item, not done: the prefill kernel's device-loop pass fill and drain.** In kiln_moe_ep_kernel's 512-lane
overflow passes (layer 20's profile) ~45 us of each 406 us pass has the tensor engine nearly idle: ~15 us of the four
sub-blocks' scatter-adds at the end (all compute done), the all-engine loop barrier, then ~25-30 us before the first
gate_up matmul (the table reads, 128 sign activations of the lane-token count on the scalar engine, 4 MB of x rows
gathered, their transposes, the first weight chunk and its dequantization). Over the 42 layers' busiest ranks
(`tools/ep_routing.py` routing, 4096 rows) there are 238 such passes per prefill call on random-token routing (97 ms of
the ~181 ms busiest-rank kernel time) and 138 on wikitext. Computing the next pass's table reads and lane tokens inside
the current iteration and prefetching its x rows into a second buffer could recover ~30-35 us per pass, ~7-8 ms per call
(~1.3% of the prefill call, ~0.75% end to end at conc 64), less once redundant hot-expert copies (EPLB) shorten those
experts' pass chains. It changes the prefill kernel's source, so every prefill graph recompiles; deferred.

**accumulate=None on nc_matmul (the attention agent's trap, next to "A PSUM accumulation hazard on trn1" above).** With
`accumulate` left at None the compiler infers the flag, and on the attention agent's kernel consecutive matmuls into one
reused PSUM tile were chained into one accumulation group (garbage scores until `accumulate=False` was passed). Audit of
the MoE kernels (2026-10-05): kernels/moe_ep.py (18 matmuls) and moe_dedupe.py (18) pass `accumulate` explicitly
everywhere; moe_prefill.py leaves it implicit on 5, each the only write to its PSUM slice (the plan's `pt[:, 0:1]` /
`pt[:, 1:2]`, the column-factor broadcast `pfb`, the fold's `pf[:, 0, :]` / `pf[:, 1, :]`), and a wrong group there would
scramble the plan's slots or the up rows, which the device checks against the emulation (0.0017-0.0034 of the output's
max) and real-weight ppl rule out. Left as is (an edit recompiles every tensor-parallel prefill graph for the same
output); make them explicit at that kernel's next change.

**v5 in the decode step, against v2 back to back** (kiln-mk-32, 2f28ff8's TD-SMALLV5 graphs, the method above; logs s3
logs/kiln-mk-32/*-td-V5.log, *-td-V2.log of 08:09 and 08:16 UTC): v5 127.4 / 173.3 / 255.2 ms at 8 / 16 / 32 rows per
group, v2 125.4 / 177.1 / 258.7 (its first run 124.7 / 177.3 / 258.5: v2 reproduces within 0.7 ms). So v5 is 2.1% /
1.4% faster at 16 / 32 rows and 1.6% slower at 8 (its two extra loop regions and the four strided table copies are a
fixed ~50 us per call that 8 rows per group do not pay back). v2 stays the default; v5 is `KILN_MOE_EP_SMALL_V=5`.

**The decode-path numerics gate** (`tools/check_mixed.py` on the G64 serving graphs, 32 prompts, 64 greedy tokens,
top-2 logprobs; V1 = `KILN_MOE_EP_SMALL_V=1` from q/final-ebe237e, V2 from q/moek-trn1; logs and json s3
logs/kiln-mk-32/*-cm-{L,W}-{V1,V1b,V2}.log, cm-cmp-*.log):

| text | comparison | outputs equal | decode-path NLL change (signed mean, nats / token) | mean / p99 / max \|dlogprob\| | margins at the first differences |
|---|---|---|---|---|---|
| LONG_TEXT | V1 vs V1 again | 32 / 32 | 0 | 0 / 0 / 0 | - |
| LONG_TEXT | V1 vs V2 | 30 / 32 | **+0.00050** | 0.0017 / 0.049 / 0.30 | 0.125, 0.125 (the two 700-token prompts) |
| wikitext-2 | V1 vs V1 again | 32 / 32 | 0 | 0 / 0 / 0 | - |
| wikitext-2 | V1 vs V2 | 5 / 32 | **-0.00077** | 0.021 / 0.18 / 0.35 | all <= 0.125 |

The engine is now deterministic run to run (engine-v0 c99a373's padded-row fix: the earlier floor of 30 / 32 and 18 /
32 came from a tree before it), so every difference above is v2's arithmetic: a token whose local expert has more than 16
pairs in its call now takes the dequantize-first arithmetic every prefill token already takes, instead of the per-row
one. The prefill chunks are bit-identical (n = 32, max 0), the two texts move the decode-path NLL in opposite directions
by under 0.001 nats per token, and every flip is at a near-tie (top-1 minus top-2 <= 0.125). For scale, the KDA / DSA
decode kernels were accepted at +0.0003 / +0.0002 with mean |dlogprob| 0.0028 / 0.026 (same table above, "Where a
GLM-5.3-Flash decode step goes").

CPU suite on efe8795 (the default flip; kiln-mk-ci2, m7i.4xlarge, one pytest process, `KILN_TEST_MODEL=Qwen/Qwen3-0.6B`,
the venv's transformers 5.15): 687 passed, 51 skipped, 0 failed in 15:46; the GLM-family / DSA / EP files with
transformers 5.18.0 on PYTHONPATH: 294 passed, 12 skipped (logs s3 logs/kiln-mk-ci2/ci-full.log, ci-tf518.log).

**The automatic EP gate follows the decode kernel** (f84c751, models/decoder.py ep_auto_min_decode_rows): from 4 decode
rows per DP-attention group with v2 (8 kept for `KILN_MOE_EP_SMALL_V=1`). The utilization agent's conc-16 A/B on
kiln-ut-32 (4 rows per group, 128 requests, back to back, every arm at `--max-num-seqs 32 --kv-cache-gb 1.5
--decode-buckets 4`): TP 86.9 out tok/s, EP v1 88.4 (+1.7%), EP v2 **98.0** (+12.8%), decode call 0.066 (TP) / 0.101 /
0.084 s, prefill call 0.895 (TP) -> 0.592 s (logs s3 logs/kiln-ut-32/ut-g16t32.log, ut-g16e.log, ut-g16e-v2.log with
their .log.cmd); the final G16 config (16 / 0.65, TP) gave 87.5 on the same box, TTFT p50 8.65 against EP v2's 6.10 s.

**A 128-lane first pass, sized on EPLB's slot counts (closed: ~1% of the prefill call).** Measured (kiln-mk-k2,
trn1.2xlarge, `KILN_MOE_EP_LW=128` against the default 256 in `tools/probe_moe_ep.py full --fit block --chunks 4096`): ten
local experts with uniform routing (1050 pairs, largest 118, so every expert fits one pass) 2.375 -> 2.031 ms, i.e. a
static first pass ~195 -> ~161 us, not half: at 128 lanes the pass is bound by its 1536 dequantized tiles on the vector
and scalar engines instead of by the tensor engine; layer 3's real busiest rank (largest expert 665) 2.796 -> 3.018 ms,
the 128-lane tails costing more than they save. On the techniques agent's per-slot pair counts (s3
logs/kiln-tq-cpu2/eplb-counts-s1.pt: the random-token routing, 4096-row batches as the sweep forms them, 20 batches x 42
layers, s=1 copies; busiest ranks 1678 pairs on average, 54% of their slots <= 128 pairs, 20% 129-256, 26% above), the
lane model (static first pass 195 / 161 us at 256 / 128 lanes, 512-lane loop pass 406, loop tails 211 / ~178, plan 150,
the busiest rank per layer and batch) gives per 4096-token call: all-256 (today) 119.8 ms of busiest-rank kernel time,
all-128 138.1 (+18.3), slots sorted by pairs with the K largest at 256 and the rest at 128 best at K = 5: 114.6 (-5.2),
a per-slot oracle 111.1 (-8.7); on the contiguous placement 188.3 -> 182.3 at best. -5.2 ms is ~1% of EPLB's 0.529 s
prefill call, under the 3% bar for a prefill-kernel change: not done.

CPU suite on f84c751 (the EP gate change; kiln-mk-ci3, m7i.4xlarge on-demand, one pytest process, transformers 5.15):
687 passed, 52 skipped, 0 failed in 15:44; the GLM-family / DSA / EP files with transformers 5.18.0: 295 passed, 12
skipped (logs s3 logs/kiln-mk-ci3/ci-full.log, ci-tf518.log). At 4 decode rows per group v2 is bit-identical to v1 (no
local expert has more than 16 pairs there): the utilization agent's check_mixed on the conc-16 serving graphs, EP v2 vs
EP v1, 16 / 16 outputs equal and dlogprob exactly 0 over 1008 decode positions; EP vs TP there 15 / 16, decode-path
signed mean -0.00023 nats per token (EP's summation order, as when EP was merged; s3 logs/kiln-ut-32/cm-g16*.json).
## Expert-parallel load balancing with redundant expert slots (EPLB) (2026-10-05, SDK 2.32, trn1.32xlarge)

feat/techniques (`kiln/models/eplb.py`, opt-in `KILN_EP_REDUNDANT=s`; default 0 traces exactly engine-v0's graphs).
The reference engines: SGLang v0.5.21 `python/sglang/srt/eplb/` (eplb_algorithms/deepseek.py replicate_experts and
balanced_packing; expert_distribution.py:614-626 per-layer counts into a GPU buffer; eplb_manager.py:89-185 a rebalance
every `--eplb-rebalance-num-iterations` 1000 passes; expert_location_dispatch.py:121-161, without an all-to-all backend,
a pair's copy by its row index modulo the copy count so every rank agrees) and vLLM v0.30.0 `vllm/distributed/eplb/`
(policy/default.py:76 replicate_experts, :192 preserve_intragpu_slots; eplb_state.py:553-716 a 1000-step window and a
3000-step interval; fused_moe/router/base_router.py:51-56 a hash of the local token index modulo the copy count;
vLLM main #52641 batches the weight migration). Neither publishes a speedup.

**Why.** Under expert parallelism a MoE layer waits for its busiest rank (the block's reduce-scatter needs every
rank's partial sum), and on the sweep's random-token prompts the busiest of 32 ranks holds 3.7-3.8x the mean pairs at
4096 rows ("Expert parallelism" above): a few deep-layer experts take most tokens (one expert with 4090 of 4096 tokens
on layer 20). A placement alone cannot split one such expert, and a static hot set did not transfer from random tokens
to text (13% overlap), so the copies are placed from the traffic's own counts and re-placed periodically.

**Kiln's form.** Every expert keeps its PRIMARY copy where the contiguous placement puts it (rank r: experts 9r ..
9r + 8 at tp 32), and each rank adds s redundant slots per MoE layer that hold copies of the hottest experts
(DeepSeek's replicate: each extra slot to the expert with the largest load per copy; each copy, heaviest first, onto
the least-loaded rank without one). Physical ids: primary e is e, slot j of rank r is E + r s + j; the EP kernel's
local map grows to [1, E + tp s + 1] and its blob to El + s experts, the kernel itself unchanged (the MoE-kernel agent
confirmed nothing assumes El = 9). Routing ids are mapped to physical ids before the kernel by an elementwise int32
sum over only the experts that have copies (no matmul, which auto-cast could round; no integer division: the row
classes are a tiled identity), on each rank's own 128 sequence-parallel rows before the routing gather; a pair of row
t takes copy (t mod 16) mod n. The prefill graphs also count each rank's routing ids per expert into a per-layer
buffer (`ep_stats`, an in-place add). A rebalance all-reduces the counts over gloo, places the copies (sticky: a copy
whose expert still deserves one keeps its slot unless a fresh placement's busiest rank is 5% lighter, vLLM's
preserve_intragpu_slots taken further), loads only the changed slots from the checkpoint on a host thread beside
serving, and installs them at a later step once a gloo flag says every rank is done (vLLM's async EPLB; the pause is
the commit: the calls in flight finish and the staged slots are copied in). The slot count is a shape, the placement
data, so a rebalance compiles nothing.

**Simulated first** (`tools/eplb_sim.py` over tools/ep_routing.py's saved top-k of real GLM-5.3-Flash routing, the
copies placed from random1, evaluated on random0's 4096-row batches; 2026-10-05, kiln-tq-cpu): the busiest rank's
EP-kernel time summed over the 42 MoE layers per 4096-row prefill call, with the measured pass costs (0.245 ms per
256-lane static pass, 0.40 per 512-lane overflow pass): contiguous 199.3 ms; +1 slot per rank, primaries fixed, 140.3;
a full re-placement 137.0; an oracle placement from the evaluated batches 122.9; +2 slots no better (140.9: each slot
adds a static pass). Busiest / mean pairs 3.76 -> 1.83. On wikitext routing 166.5 -> 138.7. With the MoE-kernel agent's
lane model (0.15 ms + 0.78 us per executed lane): 188.3 -> 127.5 ms.

**The busiest rank per layer on the device** (`tools/probe_eplb_kernel.py --routing-file ep_routing.pt --init
eplb-init-random1.pt`, kiln-tq-32 one NeuronCore, the served tile-scale EP kernel at GLM-5.3-Flash rank shapes, real
routing of random0's 4 x 1024 tokens, copies placed from random1 only, each layer's busiest rank found by the lane
model and timed, p50 of 5; log s3 logs/kiln-tq-32/probe-eplb-kernel.log): **207.2 -> 145.6 ms summed over the 42
layers (-61.6 ms per 4096-row prefill call)**. Per layer (contiguous ms -> +1 slot ms): L3 2.98 -> 2.56, L4 3.38 ->
2.54, L7 5.25 -> 2.98, L13 5.41 -> 3.19, L17 5.41 -> 3.39, L19 6.06 -> 3.15, L20 6.05 -> 3.80, L21 6.49 -> 3.78, L26
5.40 -> 4.64, L29 5.42 -> 4.24, L35 3.78 -> 3.99 (worse: an expert with 1609 pairs got no copy from random1's
counts), L41 6.30 -> 3.78, L42 6.63 -> 4.00, L44 6.05 -> 4.00; the busiest rank's pairs go from 1572-5884 to 889-3089.
`tools/probe_eplb_device.py` on the same core: an eager host-to-device copy into ONE slot of blob-shaped device
tensors rewrites it and leaves the others bit-identical; remap and the in-place count compiled with the neuron
backend equal their CPU values over 3 calls.

**Serving A/B** (kiln-tq-32, trn1.32xlarge spot, real weights, tp=32, DP attention 4, the G64 command with
`--kv-cache-gb 1.2 --state-checkpoints 4` for both (HBM room for the slot: 64 x 8448 tokens need 4224 of the 4399 pages
per group, and the random-prompt sweep takes no checkpoint), 128 requests per level, farm graphs q/eplb-a2b1414
(configs G64-4096-KV1.2-CK4-S20-P12-K-EPT and -R1), 0 device compiles; logs s3 logs/kiln-tq-32/ab-A.log, ab-B.log with
`.log.cmd`):

| run | out tok/s | TTFT p50 / p90 | ITL p50 | prefill call | decode call | spot $ / M out |
|---|---|---|---|---|---|---|
| A, plain EP (feat/techniques b841426, KILN_EP_REDUNDANT unset) | 122.9 | 6.9 / 79.7 s | 453 ms | 0.592 s | 0.179 s | 4.86 |
| B level 1: +1 slot, copies from host routing of random prompts (`KILN_EPLB_INIT`, ep_routing.pt random0 + random1) | **128.7 (+4.7%)** | 6.4 / 73.7 s | 430 ms | **0.529 s** | 0.185 s | **4.64** |
| B level 2: copies re-placed from level 1's own recorded counts (`--concurrency 64 64 --eplb-rebalance`) | 127.6 (+3.8%) | 7.2 / 76.2 s | 429 ms | 0.527 s | 0.185 s | 4.68 |

A equals engine-v0 ebe237e's G64 default (122.9), so KV 1.2 and 4 checkpoint rows are neutral. The prefill call is
63-65 ms shorter, as the probe predicted; the decode call 6 ms longer: the small decode kernel runs one static pass
per local slot whatever its pairs (~108-133 us, the MoE-kernel agent's measurement, x 42 layers), which its decode v2
(KILN_MOE_EP_SMALL_V=2: passes only for experts with pairs) removes. From each level's recorded counts
(`tools/eplb_level_stats.py`): busiest / mean pairs over the level 3.65 -> 1.61 (level 1) and 3.71 -> 1.58 (level 2);
the lane-model busiest-rank kernel of the level's average batch 185.0 -> 119.9 and 186.6 -> 117.7 ms over 42 layers.
The rebalance between the levels (synchronous in that run, before the non-blocking form): 1069 / 919 of the 1344
slots moved, rank 0 reloaded 32 of its 42, 20.3 / 18.7 s; the policy for a sweep is to rebalance between levels,
outside both timed windows.

**Greedy text, teacher-forced** (`tools/check_mixed.py` on the serving graphs, 32 prompts of 700-8192 tokens, 64
greedy tokens, concurrency 32, top-2 logprobs; tools/eplb_check.sh; logs s3 logs/kiln-tq-32/cm-*.json, cm-cmp-*.log):

| text | comparison | outputs equal | decode calls: mean / p99 / max \|d\|, signed mean | prefill chunks: mean \|d\|, signed |
|---|---|---|---|---|
| LONG_TEXT | A vs A (run to run) | 32 / 32 | 0 / 0 / 0, +0.00000 | 0, +0.00000 |
| LONG_TEXT | A vs B | 30 / 32 | 0.0033 / 0.10 / 0.26, **-0.00077** | 0.015, +0.0151 (n = 32) |
| wikitext-2 | A vs B | 3 / 32 (token agreement 0.42) | 0.034 / 0.29 / 0.94, **-0.0022** | 0.027, -0.0019 |

The engine is deterministic since the cold-determinism fix (A vs A is bit-identical), so any change of the ranks'
summation order moves tokens. The copies compute the same pairs with the same weights; what moves is which rank's
bf16 partial sum a pair joins and so the reduce-scatter's order. Against the floor of two numerically neutral engines
on the same natural text (EP row form vs EPT above: 9 / 64 equal, agreement 0.454, decode calls mean 0.037 / p99 0.30 /
signed -0.0028, prefill chunks 0.038 / +0.021) EPLB sits at that floor; its decode-path NLL change, -0.0008 and -0.0022
nats per token, is of the size of the accepted decode kernels' (+0.0003 / +0.0002) and SP decode streams' (+0.0001 /
-0.0007), with an SE of ~0.002 on wikitext's 760 tokens.

**A per-layer slot budget does not pay** (`python tools/eplb_sim.py budget --load ep_routing.pt --budget N`, kiln-tq-cpu,
2026-10-05; the same out-of-sample split and lane model as above): the same 42 slots per rank given by marginal gain
(0-3 per layer; the cool layers 3, 5 and 35 take none, layers 20, 21 and 29 two) model at 126.7 ms against 127.5 for one
slot on every layer; 63 slots (1.5x the HBM) 124.2. What is left above the floor is the static first pass every local
slot runs whatever its pairs (lane model: 0.15 ms + 10 x 256 lanes x 0.78 us = 2.15 ms per layer, 90 ms over 42), so
more balance has to come from the kernel's passes, not from placement.

**What is left of the imbalance, profiled on every rank** (the utilization agent's method, feat/utilization 805b909:
`KILN_CAPTURE_INPUTS` records the inputs of one real serving prefill call on all 32 ranks, prefill call #20 of a G64
level with all 4 groups busy. `tools/util_report.py replay --seq 2 --profile-all` replays the second 12-layer piece, layers
12-23, under neuron-explorer, and the per-layer table comes from each rank's profile: the MoE segment up to its world
reduce-scatter. tools/eplb_imbalance.sh on scratch/eplb-util; kiln-tq-32; logs s3 logs/kiln-tq-32/imb-table-{A,B}.log):

| piece 1 (layers 12-23) | piece time (rank 0) | MoE segment, median rank | MoE segment, busiest rank per layer | busiest - median | rank 0 waiting in its RS |
|---|---|---|---|---|---|
| A, plain EP | 170.3 ms | 33.4 ms | 68.6 ms | **+35.2 ms (2.05x)** | 36.3 ms |
| B, +1 slot (copies from KILN_EPLB_INIT) | 148.8 ms (-21.6) | 38.4 ms | 46.2 ms | **+7.8 ms (1.20x)** | 11.0 ms |

Per layer, busiest minus median goes from 1.1-4.7 ms to 0.3-1.1 ms. The copies remove 78% of the excess the slowest rank
adds. The median rank pays +5.0 ms over the 12 layers (0.41 ms per layer): its tenth slot's static pass and that
expert's weight reads. Over the call's 42 MoE layers the residual is about 27 ms of a 530 ms prefill call (5%), the
bound on what any further placement could still buy. It agrees with the slot budget above: more slots add static passes
faster than they remove imbalance.

**With the MoE-kernel agent's decode v2** (scratch/eplb-v2 = feat/techniques + feat/moe-kernel-tune 4c3d078,
`KILN_MOE_EP_SMALL_V=2`: decode passes only for local experts with pairs, so an empty slot costs nothing; farm graphs
q/eplb-v2-0697abc; same box, the same G64 KV 1.2 CK4 command, 128 requests per level; logs s3
logs/kiln-tq-32/stack-D.log, stack-C.log with `.log.cmd`):

| run | out tok/s | TTFT p50 / p90 | ITL p50 | prefill call | decode call | spot $ / M out |
|---|---|---|---|---|---|---|
| A, plain EP (the table above) | 122.9 | 6.9 / 79.7 s | 453 ms | 0.592 s | 0.179 s | 4.86 |
| D, decode v2 | 129.6 (+5.5%) | 6.8 / 76.2 s | 436 ms | 0.592 s | 0.156 s | 4.61 |
| C level 1, decode v2 + one redundant slot (copies from KILN_EPLB_INIT) | **135.6 (+10.3%)** | 6.3 / 70.4 s | 414 ms | 0.530 s | 0.166 s | **4.40** |
| C level 2, copies from level 1's recorded window | **135.9 (+10.6%)** | 6.3 / 70.3 s | 413 ms | 0.530 s | 0.166 s | 4.39 |

The two compose: EPLB takes the prefill call to 0.530 s and v2 the decode call to 0.156 s; the copies' 10 ms on the
decode call under v2 is decode spreading a replicated expert's pairs over its copies (each used copy is one more pass
and 25 MB of weight reads on its rank), which `KILN_EPLB_DECODE=0` (decode pairs on the primaries, same graphs) is
measured against below. The rebalances between the levels in this run were the non-blocking, sticky form: 315 and 180
of the 1344 slots moved, rank 0 reloaded 9 and 5 slots on its host thread in 2.9 s, and the engine paused 0.21 / 0.19 s
for the commit (the earlier synchronous, non-sticky form: 1069 / 919 moved, 20.3 / 18.7 s paused).

| run (same box and command) | out tok/s | TTFT p50 / p90 | ITL p50 | prefill call | decode call | spot $ / M out |
|---|---|---|---|---|---|---|
| Cd0: C with decode pairs on the primaries (`KILN_EPLB_DECODE=0`: the v2 default, 0b88bd2 with `KILN_MOE_EP_SMALL_V=2` set and 25a45c9 with it unset; stack-Cd0.log) | **136.7 (+11.2%)** | 6.3 / 70.1 s | 410 ms | 0.530 s | 0.162 s | **4.37** |

So under v2 a copy is worth more to prefill than to decode: spreading decode pairs costs 4 ms per decode call (0.166 ->
0.162 s on the primaries) and gains nothing back.

**G1b with the stack** (tools/eplb_g1b_stack.sh on scratch/eplb-v2 fa4ef92, which carries 0b88bd2's decode default:
conc 64, 75% of every 8192-token prompt one of 4 shared 6144-token prefixes, a cold level then a warm one, 256
requests per level, KV 1.2 fp8, CK4, q/eplb-v2-0697abc; logs s3 logs/kiln-tq-32/g1bs-D.log, g1bs-C.log):

| run | level | out tok/s | wall | TTFT p50 / p90 | ITL p50 | hit rate | prefill call | decode call + per step | spot $ / M out |
|---|---|---|---|---|---|---|---|---|---|
| D, decode v2 | cold | 208.2 | 314.8 s | 2.27 / 25.5 s | 278 ms | 0.674 | 0.590 s | 0.159 s | 2.87 |
| D | warm | 222.6 | 294.4 s | 2.25 / 20.5 s | 280 ms | 0.721 | 0.589 s | 0.158 s | 2.68 |
| C, v2 + one redundant slot | cold | 212.0 (+1.8%) | 309.1 s | 2.26 / 23.3 s | 273 ms | 0.674 | 0.557 s | 0.161 s | 2.82 |
| C | warm | **225.7 (+1.4%)** | 290.3 s | 2.14 / 18.8 s | 277 ms | 0.721 | 0.558 s | 0.160 s | **2.65** |

Read the decode column as the least-squares decode coefficient plus the per-step constant: on G1b every step carries
one decode call, so the two are collinear and the fit alone put +32 ms on C's decode call and -24 / -29 ms on its
constant. Their sum moves by 2 ms. The gain is smaller than on G64 because G1b is decode-bound (prefill 36-45% of
device time, against 57-60% on G64), and a hit request's chunks carry only its 2048 uncached tokens. EPLB's copies
only shorten a prefill call, by 32 ms here against 62 ms on G64's full 4096-row calls.

**F0, conc 32** (tools/eplb_f0.sh: 32 sequences, 8 decode rows per group, bf16 KV 1.3 GB, 4 checkpoint rows, 128
requests, q/eplb-a2b1414, no decode v2; logs s3 logs/kiln-tq-32/f0-A.log, f0-B.log):

| run | out tok/s | TTFT p50 / p90 | ITL p50 | prefill call | decode call | spot $ / M out |
|---|---|---|---|---|---|---|
| A, plain EP | 112.4 | 6.5 / 28.2 s | 254 ms | 0.593 s | 0.124 s | 5.31 |
| B, +1 slot | **116.1 (+3.3%)** | 6.0 / 25.7 s | 247 ms | 0.530 s | 0.131 s | **5.14** |

The same shape as G64 without v2: the prefill call is 63 ms shorter, and the decode call is 6 ms longer from the
static pass of the extra slot. A equals ebe237e's 112.3.

**HBM** (tools/tensor_bytes.py, rank 0, meta build on kiln-tq-cpu2; tools/hbm_estimate.py differences over each
configuration's compiled keys):

| configuration | tensors per rank |
|---|---|
| G64 default (KV 1.5, 2 x 16 checkpoint rows per group; loads) | 14.325 GB |
| + one redundant slot (R1) | 15.415 GB |
| + the decode kernels and SP decode streams (DK; no tensor changes) | 14.325 GB |
| DK + R1 + KV 1.2 | 15.093 GB |
| DK + R1 + CK4 | 14.898 GB |
| R1 + KV 1.2 + CK4 (configuration B above; loads and serves) | 14.576 GB |

R1 adds 0.027 GiB of code and 0.026 GiB of spill rings, summed over all 24 keys: negligible. The 1.09 GB is
the slot's weights: 42 layers, one 26 MB expert each. Without the decode kernels B needs both `--kv-cache-gb 1.2`
and `--state-checkpoints 4` (B is +0.23 GiB over the default, measured to load). With them either one alone is about
as safe as the default:
- DK + R1 + CK4 is -0.0 to -0.4 GiB, counting the decode kernels' 0.85-1.0 GiB of rings given back.
- DK + R1 + KV 1.2 is +0.17 to -0.2 GiB.
- DK + R1 with neither is about +0.5 GiB and has not been tried on a device.

CK4's 4 rows per group still hold G1b's 4 prefix junctions. KV 1.2 keeps 4400 pages per group against the 4224 that
64 x 8448 tokens need. A at KV 1.2 / CK4 serves 122.9 out tok/s, the default's.

**Real-weight ppl with EPLB** (`tools/check_ppl.py --model zai-org/GLM-5.3-Flash --tp 32 --dp-attention 4 --piecewise
--kv-cache-gb 1.0 [--text-file wikitext2_test.txt]` with the serving env, `KILN_MOE_PREFILL_MIN_TOKENS=1` and
`KILN_MOE_EP=1`: check_ppl's 4 sequences per call leave the automatic expert-parallel default off, so a ppl config without
it scores TP, which is what the first captures of this section did by mistake; farm configs ppl4-EP1-GRP-DP4 /
wt-EP1-GRP-DP4 and -R1 in q/eplb-a2b1414; tools/eplb_ppl.sh; kiln-tq-32; logs s3 logs/kiln-tq-32/ppl-*.log):

| tree | France | Water boils | def add | quick fox | mean | wikitext-2 (3071 tokens) |
|---|---|---|---|---|---|---|
| A, plain EP | -2.216 | -3.631 | -0.922 | -1.702 | -2.109 | -0.550 |
| B, +1 slot (copies from KILN_EPLB_INIT) | -2.216 | -3.631 | -0.922 | -1.702 | -2.109 | -0.551, -0.551 (two runs) |

The 4-sentence mean is -2.109 where the brief's reference is ~-2.07: it is Water boils' knife-edge, which moved with the
f9dc4c4 merge at DP attention 4 (-2.108 on 2b3bdbe above), the same on both trees here. wikitext is within 0.001 of
the -0.5515 reference on both.

**Seen once, not reproduced: an NKI trace error on one rank.** The first wikitext run of B failed on rank 6 of 32 with
`<unknown>:0: error: entry function 'kiln.kernels.moe_ep.kiln_moe_ep_kernel' not found` inside a dynamo fake-tensor call
of kiln_moe_ep_kernel (operands (10, 16, 2, 32) / (10, 128, 16, 4096) / (10, 16, 32), static args 128, 512, 13, ...),
so that rank never reached the graph the other 31 executed and the call failed with `Failed to schedule neff
execution. status=2 message=Invalid` and a replica-group signature dump. A second 32-rank job (a G1b serve_sweep started
by a waiter script that read a stale STACK_DONE) was running on the same box at that time, 08:01-08:03 UTC. After the
box ran one job at a time again, the same command passed twice (-0.551, -0.551). If the message recurs, capture that
rank's full traceback: the candidates are the NKI frontend's module lookup or its intermediate cache under two jobs'
32 processes.

## The attention kernels against their floors, and the pooled-DSA prefill attention as one kernel (2026-10-05, SDK 2.32, nki 0.6.0, trn1)

The owner relayed a claim from the SGLang side that Kiln's NKI kernels perform poorly. Each attention kernel was
measured alone at GLM-5.3-Flash's serving rank shapes (tp=32, DP attention 4, attention TP 8: 8 heads, latent 512, a
32 x 128 indexer, 8448 keys = page bucket 264, 8 KDA heads of 128 x 128) on kiln-ak-k1 (trn1.2xlarge spot, one
NeuronCore, random inputs of the right kind) against floors built from measured engine rates.

**Engine rates** (`python tools/probe_engine_rates.py`: one instruction kind repeated nt = 16 and 272 times in one
kernel, each repeat writing its own ring tile, every result read; ns per instruction = the difference / 256; the
graph's launch floor is ~0.145 ms):

| instruction | ns |
|---|---|
| PE matmul bf16, stationary [128, 128], moving [128, 512] | 219 (0.43 ns per moving column: 82 TFLOPS) |
| same, moving [128, 128] / [128, 256] | 48 / 175 |
| same with fp32 operands, moving [128, 512] | 1150 (5.2x bf16) |
| bf16 stationary, fp8 (e4m3) moving [128, 512] | 263 (slower than bf16) |
| stationary [128, 8] (8 columns), moving [128, 512] / [128, 128] | 421 / 68 (no cheaper than a full stationary) |
| stationary [64, 128] (K = 64), moving [64, 512] | 271 |
| outer product, K = 1, moving [1, 512] | 373 |
| transpose as a matmul against the identity, bf16 [128, 128] (stationary changing every time) | 40 |
| fp32 matmul, stationary [128, 1], moving [128, 128] (kernels/kda_decode.py's reads) | 233 |
| DVE tensor_reduce max [128, 512] from PSUM / [128, 2048] from SBUF | 494 / 1908 (0.95 ns per element) |
| DVE tensor_copy PSUM -> bf16 SBUF [128, 512] | 447 |
| DVE scalar_tensor_tensor (PSUM x scalar) + SBUF [128, 512] | 505 |
| DVE tensor_tensor SBUF x SBUF / PSUM x SBUF [128, 512] fp32 | 1039 / 494 (two SBUF operands: half rate) |
| DVE tensor_scalar SBUF [128, 2048] | 1894 |
| DVE tensor_scalar_reduce (>=, sum) [128, 2112] (a radix round of kernels/dsa_topk.py) | 2154 |
| DVE tensor_reduce add [128, 128] (per-instruction overhead: ~25-50 ns back to back) | 147 |
| ACT exp(x - m) [128, 512] PSUM -> bf16 SBUF with the row sum | 466 |
| ACT copy [128, 512] PSUM -> bf16 SBUF / [128, 128] SBUF -> SBUF | 383 / 106 |
| ACT exp [128, 2048] SBUF -> bf16 with the row sum | 1574 |
| indirect DMA gather, 128 rows x 2048 bytes (kernels/dsa_decode.py's pool gather) | 1419 (185 GB/s) |
| GpSimd tensor_scalar fp32 | does not compile on trn1 |

(The static HBM -> SBUF DMA variant measured nothing: repeated loads into the same ring tiles were folded.)

**Each kernel against its floor** (`python tools/prof_attn_kernels.py <case>`: p50 of synchronous calls with the
output reduced to a scalar, then `neuron-explorer capture` of the graph's NEFF on the same inputs in a fresh process
and each engine's busy time from the profile JSON; the floors use the rates above and 410 GB/s of HBM per core):

| kernel at its serving shape | PE floor | vector floor | HBM floor | measured | what the profile says binds |
|---|---|---|---|---|---|
| DSA prefill attention core, XLA (models/mla.py _core expand), 1024 queries x 8448 keys | 106 GFLOP: 1.3 ms (absorbed 146: 1.8) | 69 M scores: max 0.51 ms, exp 0.49, P^T drain 0.41 | ~70 MB: 0.17 ms | 9.49 ms (in the layer: 8.7) | spill DMA: active 8.8 ms, 911 MB saved and 1.19 GB reloaded; PE busy 2.15 ms, DVE 1.72, ACT 2.36 |
| DSA prefill scores + selection (dsa_topk score_select), 1024 queries x 2112 pools, keep 512, kp 4 + tail | 17.7 GFLOP: 0.22 ms | relu 0.41 ms ACT; head sum 0.54 DVE with a PSUM operand (1.08 with two SBUF ones, as written); 47 radix and tie rounds 0.81 DVE | 35 MB out: 0.08 | 2.96 (in the layer: 3.72 with the pool-key read) | DVE: busy 2.54 ms, its scalar_tensor_tensor head sum ~2.7 ms of instruction time; ACT 0.59, PE 0.43 |
| KDA prefill chunk (delta_rule), C = 1024, 8 heads | PE fp32 busy 0.70 ms | DVE 0.68, ACT 0.32 | 23 MB: 0.06 | 1.20 | PE and DVE about equal, overlapping ~60% |
| KDA decode (kda_decode), 4 / 16 / 64 rows | ~1.2 us of one-column fp32 matmuls per (row, head): 0.04 / 0.15 / 0.6 ms | small | 1 MB per row read and written: 0.01 / 0.04 / 0.16 | 0.22 / 0.36 / 0.94 | PE (busy 0.146 ms at 16 rows); two of its five matmuls per head only move a k / v row to partition 0 |
| DSA decode gather (dsa_decode), 4 / 16 / 64 rows of 2560 tokens | ~18 us per row (80 latent transposes, scores and P K with an 8-column stationary) | 20 latent drains of [128, 512] per row (8.4 us), the softmax on 8 of 128 lanes | 1.3 MB per row: 0.05 ms at 16 rows (0.11 at the gather rate) | 0.33 / 0.85 / 2.88 | latency: at 64 rows PE busy 1.34 ms, DVE 1.48, the GpSimd DMA queue 2.24 waiting for ring slots (2 rows in flight) |
| DSA decode selection (dsa_topk select), 16 / 64 rows x 2112 | | | | 0.23 / 0.28 | ~0.1 ms above the launch floor |

So the claim holds for two of them: the XLA prefill attention core runs at ~13% of the tensor engine because its
[8, 1024, 8448] fp32 score tensor spills (2.1 GB of spill traffic per call), and the DSA decode kernel is ~4x its
floor. The KDA kernels and the selection are within 1.5-2.2x of theirs.

**The pooled-DSA prefill attention as one kernel** (`kernels/dsa_prefill.py`, `KILN_DSA_PREFILL_KERNEL=nki`, default
xla; models/mla.py _core: a pooled DSA layer's chunk of one sequence with whole 128-query tiles takes it, and W_UK /
W_UV stay in XLA around it): absorbed flash attention over the latent, the selection plus the causal visibility (the
graph's vis + top) as a bf16 additive mask. Phase 0 transposes the chunk's latent once into K^T [512, 8448] in SBUF
(67.6 KB per partition); per tile of 128 queries, q_lat^T per head by transposes, then for every (512-key block, head)
unit a software pipeline A (4 QK matmuls over R plus the mask as one more accumulating matmul against the identity),
B (block max, running max, alpha and P = exp(scale S - scale m) with its row sum), C (P^T by 4 transposes), D (P K
with P^T stationary and the block's latent rows streamed from HBM; acc = alpha acc + P K); unit u issues A(u),
B(u - 1), C(u - 2), D's matmuls of u - 3 and D's update of u - 4. No list comprehension (the NKI tracer rejects
them: "unsupported expression") and the scalar engine's copy takes no tensor bias (NCC_IBVF043), so l's update stays on
the vector engine.

`python tools/probe_dsa_prefill.py --forms nki xla xla-absorb --offset -1 0` (kiln-ak-k1, 1024 queries over 8448 keys,
512 random selected pools per query plus its tail pool, causal; error = max |o - emulate()| / max |o|):

| form | time | error |
|---|---|---|
| XLA expand (what the engine runs) | 9.42 ms | 3.9e-3 |
| XLA absorbed | 11.50 | 3.0e-4 |
| kernel, first version | 3.34 (chunk at the bucket's end and at position 0 alike: it attends every block) | 1.8e-3 / 1.9e-3 |
| kernel, the -scale m and alpha moved to the scalar engine | 3.34 | |

Its profile (`prof_attn_kernels.py dsa_prefill`): PE busy 2.27 ms, DVE 1.72 (from 2.25), ACT 1.27, spill 51 MB. It is
tensor-engine bound now: the absorbed QK and P K at R = 512 are 1.76 of the ~2.1 us per unit, so what is left is
not attending blocks no query of the tile can see (on average 54.5% of the 17 blocks are visible to a 1024-token
chunk of an 8192-token prompt) or selected.

CPU (`tests/test_glm5_next.py`, kiln-ak-k1 host, transformers 5.18 on PYTHONPATH): `test_dsa_prefill_kernel_path_matches`
(prompts of 300 / 140 / 129 tokens in 128-token chunks over 128-key page buckets: the kernel path ran and gives the mask
path's tokens and logprobs within 1e-5), `test_dsa_prefill_kernel_path_after_a_prefix_hit` (a shared 170-token system
prompt: the third request resumes at token 160, off the 128 grid; same tokens and logprobs),
`test_dsa_prefill_emulation_is_the_masked_softmax` (the emulation against the expand form, a row attending four keys of
the last block only).

**In the layer graph the saving is a third of the standalone one.** `tools/profile_layer.py --model zai-org/GLM-5.3-Flash
--tp 32 --dp-attention 4 --ranks 2 --prefill 1024 --pages 264 --layers 4 --what hcblocks --part-layers 3 --sum-readback`
(the serving env of q/final-ebe237e: EP, SP, nki selection, pool cache auto; 2 live ranks on kiln-ak-k1; `--prefill-offset`
is new: the chunk's first position), layer 3's blocks, p50:

| form | token mixer with its all-reduce | attention block (mHC + mixer + all-reduce + mix) | SP attention block |
|---|---|---|---|
| XLA (engine-v0) | 11.23 ms | 19.00 | 15.70 |
| kernel, static (every key block), chunk at 0 / at 7424 | 9.40 / 9.06 | 17.98 / 17.79 | 13.93 / 13.75 |
| kernel, causal loop (below), chunk at 0 / 3072 / 7424 | 7.27 / 8.56 / 10.10 | 16.31 / 17.52 / 19.24 | |

So the XLA core cost ~5 ms inside the layer graph, not the 8.7-9.5 it costs alone (the graph overlaps its spills with
other work). A neuron-explorer replay of the two mixer graphs (`neuron-explorer capture -r 2 -i 0 --ignore-exec-errors`,
zero inputs; `tools/prof_hlo.py`, new: busy per engine and instruction time per HLO op for a graph whose collectives
prof_step.py cannot cut): XLA form PE busy 6.14 ms, DVE 5.50, ACT 3.86, 1.0 GB spill; kernel form PE 3.84, DVE 5.18,
ACT 2.61, 0.47 GB spill. What is left on the critical path is the selection kernel (DVE-bound) and the attention kernel
(PE-bound) back to back.

**The causal form** (`KILN_DSA_PREFILL_LOOP=1`, with `KILN_DSA_PREFILL_KERNEL=nki`; models/mla.py passes the chunk's
positions): per pass of `KILN_DSA_PREFILL_QPASS` (2) query tiles, a device loop (`nl.fori_loop`, trip count = the pairs of
512-key blocks at or before the pass's last position, counted by comparisons in the graph: `loop_args`) over pairs; each
iteration DMAs its pair's K^T from a pair-major HBM scratch the kernel writes in phase 0 (the loop register as the
offset), its latent rows and masks at the pair's first key read from a table through the register, then runs the static
kernel's pipeline over the pass's (block, query tile, head) units; keys past the last full pair are a static tail block.
What it took: PSUM allocated per region (a PSUM tensor referenced by two device-loop regions fails, NCC_IBIR092); with 4
query tiles per pass the layer graph failed `[NCC_INLA001] ... overlapping with must-pinned memloc DynamicDMAScratchLoc`
(the dynamic DMAs' scratch is pinned in SBUF and the kernel's 64 KB of accumulators plus rings overran it; standalone it
compiled), so 2 tiles per pass and phase 0 in the loop's pair buffers. Alone (`tools/probe_dsa_prefill.py --forms
nki-loop`, 1024 queries over 8448 keys): 1.19 / 2.41 / 4.02 ms with the chunk at 0 / 3072 / 7424 (4 tiles per pass, which
fails in the layer: 1.14 / 2.29 / 3.79), against 3.35 for the static kernel, same error against the emulation
(1.8-2.2e-3). In the layer graph (table above) it is linear in the pairs at ~0.39 ms per pair, so over the 8 chunks of an
8192-token prompt (pairs 1..8) the mixer averages ~8.6 ms against 9.1-9.4 static and 11.2 XLA. trn1's transpose mode
writes only fp32 PSUM ("nc_matmul (transpose mode) dst dtype must be float32 on gen2"), so P^T of a 1024-key unit would
take two banks.

Measured and set aside: the score kernel's head sum reading the relu from PSUM instead of SBUF (2.96 -> 2.73 ms alone,
DVE busy 2.54 -> 1.93), held because any edit of kernels/dsa_topk.py changes its source revision and so every DSA graph
key, decode ones included; it goes into the fused selection-and-attention kernel instead.

**Serving A/B, conc 64** (kiln-ak-32, trn1.32xlarge spot, GLM-5.3-Flash real weights; the final G64 command of
docs/price-performance.md, 128 requests, back to back on one box, every graph from the farm with
`NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`; logs s3://<your-bucket>/logs/kiln-ak-32/ab-G64-base.log, ab-G64-PK.log, each
with its `.cmd`): base (q/final-ebe237e) 122.9 out tok/s, TTFT p50 / p90 6.90 / 79.7 s, ITL p50 452 ms, $4.86 per 1M spot;
`KILN_DSA_PREFILL_KERNEL=nki`, static form (feat/attn-kernel-tune 448b9f6, q/attnk-448b9f6; this tree's kernel is b1166b1's, the same arithmetic with queries padded to whole tiles) **127.6 (+3.8%)**, 6.57 / 75.7 s,
434 ms, $4.68. The farm's `tools/hbm_estimate.py` over the graphs (ranks 0/8/16/24): the kernel frees 0.69 GiB per rank
of the prefill groups' spill rings (G64 16.98 vs 17.67 GiB, DKS 15.98 vs 16.67), the three prefill groups' instance spill
runs 22-30% fewer.

**Serving A/B, conc 32** (same box and method, final F0 command): base 112.5 out tok/s (TTFT p50 6.46 s, ITL 254 ms,
$5.31), static kernel (448b9f6) **116.5 (+3.6%)**, 6.11 s, 245 ms, $5.13 (logs .../kiln-ak-32/ab-F0-*.log).

**Real weights, wikitext-2 at DP attention 4** (`tools/check_ppl.py --model zai-org/GLM-5.3-Flash --tp 32 --dp-attention 4
--piecewise --kv-cache-gb 1.0 --text-file wikitext2_test.txt`, the final serving env plus `KILN_MOE_PREFILL_MIN_TOKENS=1`,
farm queue q/attnk-c9b86e3 configs wt-DP4-*, tree b1166b1 (the static kernel's arithmetic of 448b9f6, queries padded:
check_ppl's chunks are 64 rows per group, which 448b9f6 left to XLA), kiln-ak-32, logs .../kiln-ak-32/ppl-wt-*.log and
.json): base -0.547 (by chunk -0.730 -0.692 -1.092 -0.927 -1.219 -0.172 -0.358 -0.138 -0.291 -0.270 -0.342 -0.334, the
group-collectives reading above to three decimals); static kernel -0.548 (-0.739 -0.701 -1.105 -0.929 -1.209 -0.176 -0.350
-0.137 -0.297 -0.270 -0.328 -0.331). The causal form (`KILN_DSA_PREFILL_LOOP=1`) ended its run with rc 1 and no
message after its graphs loaded (64-row chunks padded to 256, 3328 keys; exact alone at that shape in
`probe_dsa_prefill.py --rows 64 --keys 3328`), not debugged: it stays an experiment.

**Selection and attention as one kernel** (`kernels/dsa_fused.py`, `KILN_DSA_FUSED=1`; models/mla.py attention(): a
pooled DSA layer's chunk with pool keys of 4 and the tail, NoPE, the nki selection, nothing staged or shared). The
selection of query tile t + 1 is issued as micro-steps (each (head, 512-pool chunk) of the scores, then the sign, the 31
radix rounds, the ties' 12, the output) spread evenly between query tile t's attention steps, and it stays on chip: per
pool 0 / 1 in SBUF, the candidates and the visibility from the positions (a pool is a candidate once kp p + 3 <= pos;
token kp p + e is attended iff its pool is selected, the tail included, and kp p <= pos - e), the token mask built per
1024-key block. Attention units of 1024 keys, l's update on the scalar engine as relu(alpha l + rowsum) (all
non-negative; the copy function takes no tensor bias), the selection's relu written back into its own PSUM bank so its
weighted head sum reads one PSUM operand: the 8 banks are attention scores 2 x 2, P^T 2, P K 1, selection 1. One trap:
`nc_matmul`'s default `accumulate=None` chained the selection's 160 matmuls into one reused PSUM tile as an
accumulation group (scores off by up to 62 against 4.7) until `accumulate=False` was passed.

`python tools/probe_dsa_fused.py --forms dbg fused two --offset 7424 0` (kiln-ak-k1, 1024 queries, 8448 keys, 32 x 128
indexer, keep 512): tile 0's scores within 7e-7 of `emulate_scores`, the pool selection and the token mask identical to
the emulation on all 1024 rows at both positions, o within 2.3e-3 / 1.6e-3 of max |o| (the two-kernel path the same).
Time: the two kernels back to back (score_select, visibility, static attention) 6.10 ms; fused with 512-key units 5.22;
with 1024-key units 4.92. Its profile: DVE busy 3.3 ms, PE 2.6, ACT 1.75; the vector engine holds the selection (head
sum ~0.77 ms, radix ~0.75) and the online softmax's max and rescale, and its in-order queue lets a radix round delay the
attention's block max (PE waits of up to 14 us on the scalar engine's exp). In the layer graph (same profile_layer run as
the tables above, layer 3, 2 live ranks):

| form | token mixer with its all-reduce | attention block | SP attention block |
|---|---|---|---|
| XLA | 11.23 ms | 19.00 | 15.70 |
| static attention kernel | 9.06-9.40 | 17.8-18.0 | 13.75-13.93 |
| fused (1024-key units), chunk at 7424 / at 0 | **7.24 / 7.26** | 16.17 / 16.18 | 12.20 / 12.19 |

So in the layer the fused kernel takes ~4 ms off the XLA form's DSA mixer (~2 more than the attention kernel alone),
whatever the chunk's position. CPU: `test_dsa_fused_kernel_path_matches` (128-token chunks and the prefix hit at 160:
the mask path's tokens and logprobs within 1e-5).

**Serving A/Bs with the fused kernel** (kiln-ak-32, same box and method; feat/attn-kernel-tune 54ec5df, farm queue
q/attnk-54ec5df, `KILN_DSA_FUSED=1`; logs s3://<your-bucket>/logs/kiln-ak-32/ab-G64-FU.log, ab-F0-FU.log):

| config | base (final-ebe237e) | static attention kernel | fused selection + attention |
|---|---|---|---|
| G64, out tok/s | 122.9 | 127.6 (+3.8%) | **132.0 (+7.4%)** |
| G64 prefill call / decode call | 0.592 / 0.181 s | 0.553 / 0.179 | 0.519 / 0.178 |
| G64 TTFT p50 / p90, ITL p50, $ per 1M out (spot) | 6.90 / 79.7 s, 452 ms, 4.86 | 6.57 / 75.7 s, 434 ms, 4.68 | 6.26 / 72.2 s, 419 ms, 4.52 |
| F0, out tok/s | 112.5 | 116.5 (+3.6%) | **120.0 (+6.7%)** |
| F0 prefill call / decode call | 0.592 / 0.120 s | 0.553 / 0.124 | 0.521 / 0.118 |
| F0 TTFT p50 / p90, ITL p50, $ per 1M out (spot) | 6.46 s, 254 ms, 5.31 | 6.11 s, 245 ms, 5.13 | 5.82 / 25.2 s, 239 ms, 4.98 |

The prefill call drops 72 ms (12%) of which the static kernel had 39; the decode call does not move (no decode graph
changes). Wikitext-2 at DP attention 4 with the fused kernel (q/attnk-54ec5df wt-DP4-FU; its 64-row chunks are padded to
a query tile and take the kernel): -0.548, base -0.547.

**Defaults.** `KILN_DSA_FUSED` defaults to on for trn1 (kernels/dsa_fused.py `FUSED_FAMILIES`) and
`KILN_DSA_PREFILL_KERNEL` to `nki` there (kernels/dsa_prefill.py `PREFILL_KERNEL_FAMILIES`), where the A/Bs and the
wikitext gates above ran; both are off (`0` / `xla`) on trn2, inf2 and a host without a Neuron device, and either variable
overrides it (`tests/test_glm5_next.py::test_dsa_fused_default_is_trn1_only`, `::test_dsa_prefill_kernel_default_is_trn1_only`).
The fused kernel takes a pooled DSA layer's prefill chunks first; the static attention kernel is what runs where it does
not (a mixed batch's chunk, an IndexShare layer, a layer whose selection is wanted), so G64 / F0 with both defaults are
the FU graphs above. The causal-loop form stays off (`KILN_DSA_PREFILL_LOOP=0`).

## Accelerator utilization of the serving graphs, and why concurrency buys little (2026-10-05, SDK 2.32, trn1.32xlarge)

GLM-5.3-Flash real weights, engine-v0 ebe237e defaults (the final G16 / F0 / G64 configs of docs/price-performance.md,
farm graphs q/final-ebe237e, 0 device compiles), trn1.32xlarge kiln-ut-32, neuronx-cc 2.27.5334, neuron-explorer 2.32,
2026-10-05 04:50-08:00 UTC. Logs and profiles: s3://<your-bucket>/logs/kiln-ut-32/.

**Method** (feat/utilization; nothing changes a graph or its key):
- `KILN_TIMELINE=<file>` (kiln/profiling.py): rank 0 records every graph call (broadcast, argument upload, launch), every
  read-back of a call's output (how long the host blocked on the device) and every step() call. `KILN_RT_INSPECT=<dir>`
  (engine/tp.py) turns on the Neuron runtime's system trace for the ranks in `KILN_RT_INSPECT_RANKS` (default 0);
  `neuron-explorer view -d <dir> --output-format json --ignore-device-profile` gives every execution's device start and
  stop (`nc_exec_running`, its `nc_start_timestamp_ns` / `nc_stop_timestamp_ns`). The trace ring holds ~0.5M events per
  NeuronCore and dropped 5M in a 267 s level (it kept the last 36 s); `NEURON_RT_INSPECT_SYS_TRACE_MAX_EVENTS_PER_NC`
  and `NEURON_RT_INSPECT_EVENT_FILTER_TYPE` (names from `neuron-explorer capture --systrace-get-event-types`; strings in
  libnrt.so.1) are the knobs. With both on, G64 ran 122.7 out tok/s (fin2: 122.9): they cost nothing measurable.
- `KILN_CAPTURE_INPUTS=<dir> KILN_CAPTURE_AT=prefill:20,decode:400` (kiln/profiling.py) writes the exact inputs of every
  NEFF executed during the chosen calls, on every rank, through libtorch_neuronx_lite's `pre_execute_hook`
  (compile/execute_context.py: "the exact input tuple execute receives (post dead-input filtering and RNG-seed append)",
  which is the NEFF's input0..inputN order, neff.json `arg_nodes`). A tensor object seen before (weights, KV and state
  pools) is written once: ~14 GB per rank, ~430 GB for 32 ranks on the instance-store RAID. The counts include the bucket
  warmup and the warm-up request (decode:200 was the warm-up request's 1-live-row decode; decode:400 is a 64-row call after
  all 128 prefill calls of a 64-request level, checked against the timeline). LNL device tensors have no host-visible
  storage pointer (`untyped_storage().data_ptr()` raises "Attempted to access the data pointer on an invalid python
  storage"), so identity is the object, held by a weak reference.
- `tools/util_report.py replay <cap> --call <name:n>` replays each NEFF on 32 workers with every rank's own inputs
  (`neuron-explorer capture --multi-input`, one line per worker) and profiles worker 0. Two traps: (1) the DP-attention
  groups' prefill pieces are different NEFFs (their FX graphs differ only in the group collectives' process group, 1 vs 2:
  tools/hlo_diff + diff of fxgraph.txt), and a group-0 NEFF does not load on rank 8+: "ENC:enc_parse_replica_groups [nec_dev
  31] replica groups (0/2) does not have myself 31", after which the capture hangs. So each NEFF runs as its own
  neuron-explorer process over its own cores (`NEURON_RT_VISIBLE_CORES=8g-8g+7`), joined into one 32-worker collectives
  world by `--collectives-worker-start-id 8g --collectives-worker-count 32` (its multi-node form) and one
  `NEURON_RT_ROOT_COMM_ID`; each process profiles its first worker, so ranks 0 / 8 / 16 / 24 come for free (`--profile-all`
  profiles all 32). (2) A single execution starts cold: the decode prep graph replayed in 6.55 ms (run: 0.36 ms), group 0 in
  47.6 ms (run: 44.45); `--num-exec 2 --profile-nth-exec 2` gives 3.06 and 45.04 ms. The profile is
  `<session>_rank_<r>_exec_2.ntff`. Without captured inputs every input is zero and the MoE routes every row to experts 0-7:
  under EP the whole MoE lands on rank 0 (a decode group replayed in 145.7 ms with zeros, 45.0 with real inputs).
- `view --output-format summary-json` (seconds, no full JSON) gives neuron-explorer's own per-execution counters (engine
  active times, `hbm_read_bytes` / `hbm_write_bytes`, `spill_*_bytes`, `hardware_flops`, `cc_op_time`); its mfu / mbu /
  hfu fields divide by 91.75 TFLOPS (128 x 128 x 2 x 2.8 GHz) and 410e9 B/s. `tools/util_report.py bins` cuts the
  instruction trace (`--output-format json --ignore-dma-trace`) into 0.1 ms bins (an engine's queued instructions overlap
  in the trace, so each engine's busy time is the union of its instruction intervals) and into segments between
  collectives, each labelled by the collective that ends it; `report` prints the table and the model-level MFU / MBU from
  config.json plus the safetensors headers (`util_report.py model`).
- Peaks per NeuronCore-v2 from AWS's Trainium architecture page ("190 FP16/BF16/cFP8/TF32 TFLOPS" and "820 GiB/sec" per
  2-core device): 95 TFLOPS (cFP8 is no faster than bf16 on trn1) and 410 GiB/s = 440 GB/s.

**Model arithmetic** (`python tools/util_report.py model --shape-dir <glm53 shape dir>`): a prefill token of an 8192-token
prompt is 33.83 GFLOP: routed experts 16.91 (8 of 288), KDA projections 9.36, DSA projections 2.58, shared expert 2.11, DSA
attention core 1.29 (64 heads x 512 x the mean 1792 keys of a top-2048 sparse attention), dense MLPs 0.91, KDA recurrence
0.25, indexer 0.25, router 0.10, mHC 0.07; the dense masked core over the 8448-key bucket that the XLA path computes would be
5.91 GFLOP instead of 1.29. A decode step must move per rank: dense weights 1.77 GB (KDA projections at attention TP 8 1.17,
DSA 0.19, indexer and router replicated 0.15 + 0.10, mHC 0.07) + the routed experts the step's rows touch (9.51 GB x (1 -
(1 - 8/288)^rows): 84% at 64 rows) + per row 113 MB of fp8 KV over 8448 keys and KDA state read and written: 11.52 GB at 16
rows per group x 4.

**The table** (rank 0, replayed, real inputs):

| call | graphs (ms) | tensor / vector / scalar / gpsimd | HBM moved, rate | tensor engine | collectives |
|---|---|---|---|---|---|
| prefill, 4096 rows (prefill:20) | prep 4.4, pieces 136.2 / 168.2 / 169.1 / 128.7, post 8.3 = 614.8 (serving fit 592) | 29.7 / 34.2 / 27.3 / 2.2% | 33.6 GB (spill save 7.9 + reload 10.6), 55 GB/s = 12.4% | 11.1 TFLOP, 18.1 TFLOP/s = 19% | cc_op 183.5 ms; every engine idle with one in flight 180.9 ms (29%), idle otherwise 2.8 ms |
| decode, 64 rows (decode:400) | prep 3.1, groups 45.0 / 49.7 / 48.8 / 37.2, post 5.0 = 188.8 (runtime trace in the run: 178.0) | 27.1 / 47.9 / 16.3 / 10.6% | 18.8 GB (6.9 spill), 100 GB/s = 22.6% | 5.05 TFLOP/s = 5.3% | 19 ms; all idle 16.5 ms in the group graphs |

MFU (prefill) = 33.83 GFLOP x 4096 / (0.592 s x 32 x 95 TFLOPS) = **7.7%**; MBU (decode) = 11.52 GB / 0.178 s / 440 GB/s =
**14.7%**. Every engine's active time is below 50% in both: neither call is compute-bound or bandwidth-bound; prefill
waits on collectives, decode is a chain of small dependent ops.

Per layer, from the segments (the segment ending in the attention group's reduce-scatter is the token mixer, the one
ending in the world reduce-scatter the MLP / MoE):

| | prefill (4096 rows) | busy (tensor / vector / scalar / gpsimd) | decode (64 rows) | busy |
|---|---|---|---|---|
| DSA mixer, 11 layers | 12.85 ms (141 ms per call, 23%) | 28.6 / 42.8 / 31.4 / 0.5% | 4.31 ms (47 ms, 27%) | 38.7 / 29.3 / 28.0 / 16.7% |
| KDA mixer, 34 layers | 3.07 ms (104 ms, 17%) | 43.7 / 45.9 / 29.6 / 3.5% | 0.585 ms (20 ms, 11%) | 30.2 / 57.9 / 32.4 / 2.4% |
| MoE FFN, 42 layers (EP kernel, shared expert; decode: + mHC, router) | 3.10 ms (131 ms, 21%) | 69.4 / 61.6 / 53.4 / 6.4% | 1.77 ms (74 ms, 42%) | 34.4 / 68.4 / 9.7 / 14.3% |
| dense MLP, 3 layers | 0.32 ms | 90.6 / 14.4 / 56.7 / 0.3% | 0.40 ms | |
| mHC, norms, router between collectives | ~26 ms per call | | (in the FFN segment) | |
| collectives | world RS 32 MiB 45 x 2.36 ms = 106; 8 MiB all-reduces (the world gather as 4 chunks per layer, the group gather) 233 x (0.13 wait + 0.30) = 100; group RS 7; routing 1 | | 90 all-reduces of 0.5 MiB x 0.29 = 26 ms | |
| decode tail (KDA state writes) | | | 2.9 ms per group graph (12 ms) | vector 74.5% |

**Most of the prefill reduce-scatter time is the busiest rank's MoE, not the transfer.** All 32 ranks of piece 1 (layers
12-23; `replay --profile-all`, every rank's own inputs; `s3://<your-bucket>/scripts/ut/imball.py`): per MoE layer
the MoE segment on the median rank is 2.59-2.89 ms, on the busiest rank 3.90-7.34 ms (a different rank each layer: r0, r6,
r13, r1, r31, r21, r3, r14, r25, r9, r12, r5), and the world reduce-scatter takes 0.34-0.37 ms on the busiest rank (the
transfer) and 1.4-4.9 ms on the median rank (waiting for it). Over the piece the median rank's MoE is 33.8 ms and the
busiest ranks' 67.3 ms: **+33.5 ms per 12 MoE layers, ~117 ms of the 592 ms prefill call (19%)**, more than every collective
transfer of the call together (~64 ms). Expert load balance (EPLB, redundant hot experts) is the lever for it.

**Spill, by region** (the DMA trace of rank 0's piece 1 and decode group 1, spill queues only, each packet assigned to the
segment it moves in): prefill piece 1 moves 10.05 GB of spill-queue traffic: DSA mixers 41% (1.38 GB per DSA layer), KDA
mixers 27% (0.30 GB per layer), MoE segments 23% (0.19 GB per layer), the rest 10%; the spilled tensors are the NKI kernels'
outputs (custom_call, get_tuple_element), residual adds, d2d transposes of kernel outputs and graph inputs, reshape and
all-reduce buffers. Decode group 1: 2.00 GB, 87% in the DSA attention segments (0.58 GB per DSA layer: the gathered fp8
latent, `input140_d2dtranspose_*`, and coalesced spill saves), which the opt-in DSA decode kernel removes ("The decode
kernels give HBM back" above).

**Does neuronx-cc overlap a collective with independent compute in one graph? No** (`tools/probe_overlap.py`, 32 ranks,
rows 128 per rank, H 4096; every case's NEFF replayed with neuron-explorer and the tensor engine's instruction time inside
the collectives' trigger-to-end intervals counted; reps 2 = two SwiGLU MLPs, 5.3 ms of tensor work):

| graph | device ms | collectives ms | tensor time inside collective intervals |
|---|---|---|---|
| gather (the SP world gather: zero-padded all-reduce, as 4 x 8 MiB) | 4.34 | 1.58 | 0 |
| mlp | 5.78 | 0 | 0 |
| gather -> mlp (dependent) | 9.60 | 1.63 | 1.58 ms |
| gather and an unrelated mlp, gather first in program order | 9.75 | 1.50 | **0** |
| the same, mlp first | 8.99 | 1.94 | **0** |
| two half-batches, gather -> mlp each | 12.54 | 3.56 | 1.06 |
| two half-batches, both gathers first | 11.37 | 3.18 | 1.06 |
| gather -> mlp -> reduce-scatter (one FFN block) | 13.77 | 5.40 | 1.58 |
| the same on two half-batches, both gathers issued first | 10.36 | 2.08 | 1.05 |

The only compute that runs under a collective is the dependent chain's: the mlp on the first 8 MiB chunk of the gathered
rows while the next chunks' all-reduces are in flight. An independent computation is scheduled entirely outside the
collective windows whatever its program order, and so is a second half-batch: in the last row the compiler issued half B's
gather after half A's mlp and reduce-scatter, not before. `--internal-backend-options=--enable-SPMD-opt` ("Enable
reordering of collectives", walrus_driver --help), `--policy=3` (time-aware post-scheduler) and `--cc-dma-alignment-mode=0`
(prefetch as much as possible) give the same zero. So two-batch overlap in the SGLang / vLLM sense (one micro-batch's
dispatch and combine under the other's compute) is not available from this compiler by restructuring the graph; it would
need collectives issued from inside an NKI kernel. The last row is faster than the one-batch block (10.36 vs 13.77 ms)
only because that block's 32 MiB reduce-scatter ran 4.65 ms right after the mlp in this probe (two 16 MiB ones 0.35 + 0.30;
a 32 MiB reduce-scatter of an input tensor alone 0.55 ms); in the serving pieces the rank that arrives last finishes its
32 MiB reduce-scatter in ~0.35 ms, so the probe's pathology is not the serving graphs' cost.

**Why 4x concurrency buys 1.41x** (the fin2 runs' `device_split`, engine-v0 ebe237e, 128 requests = 32,768 output and
1,048,576 prompt tokens per level; host and device facts from kiln-ut-32's timeline, runtime trace and replays above):

| | conc 16 | conc 32 | conc 64 |
|---|---|---|---|
| out tok/s (wall) | 87.3 (375.2 s) | 112.3 (291.8 s) | 122.9 (266.6 s) |
| prefill calls x s/call | 256 x 0.894 = 228.7 s (61%; EP off at 4 decode rows per group) | 256 x 0.592 = 151.6 s (52%) | 256 x 0.592 = 151.6 s (57%) |
| decode calls x s/call | 2127 x 0.066 = 139.7 s | 1103 x 0.123 = 136.1 s | 639 x 0.178 = 113.7 s |
| per-step remainder | 6.0 s | 3.0 s | 0.9 s |
| ms per output token: prefill + decode | 6.98 + 4.26 | 4.63 + 4.15 | 4.63 + 3.47 |
| live decode rows per call | 15.3 of 16 | 29.6 of 32 | 51.2 of 64 |
| padded decode rows, at ~1.5 ms per row per call | 3.0 s (0.8%) | 4.0 s (1.4%) | 12.5 s (4.7%) |

1. Prefill is the same 256 full calls at every level (token slots 100%) and is 52-61% of the wall: concurrency has nothing left
   to batch there. conc 16 -> 32 gains +29% almost only because EP turns on at 8 decode rows per group (prefill call 0.894 ->
   0.592 s, -77 s).
2. A decode call costs ~50 ms + ~1.6 ms per row, so 4x the rows per call buy 1.23x per output token; the per-row parts are the
   DSA attention over all 8448 keys of the bucket, the KDA state updates, the MoE FFN's vector work and the DSA spills (table
   above).
3. Padded rows (the 128-request closed loop's burst start and drain: 224 of 640 conc-64 decode calls carry 4-59 live rows) cost
   4.7% at conc 64.
4. Host: ~0. Rank 0 blocked on the device in 648 of 648 step() calls (250.8 of 267.1 s wall), its device was 99.6% busy in the
   traced window with gaps of at most 0.17 ms; the 11.7-12.2 ms of graph launches and 2.3-3.7 ms of broadcast per call are hidden
   behind the call before. Overlap misses: none seen.
5. With free decode, conc 64 would still cap at 32768 / 151.6 s = 216 out tok/s.

**Per output token at conc 64** (8.14 ms of wall each), the replays' shares applied to the serving costs (prefill call
0.592 s for 128 output tokens' worth of prompt = 4.63 ms per output token; decode call 0.178 s over 51.2 live rows = 3.47 ms):
prefill: DSA mixers 1.06, KDA mixers 0.79, MoE compute 0.99, waiting for the busiest EP rank 0.88, collective transfers and
their sync 0.73, mHC / norms / router 0.20 ms; decode: MoE FFN 1.45, DSA attention 0.92, all-reduces 0.51, KDA attention 0.39,
KDA state writes 0.23 ms (of which ~0.7 ms is padded rows). conc 16 / 32 have no replay of their own: their prefill call is
0.894 s (TP) / 0.592 s, their decode call 0.066 / 0.123 s for 15.3 / 29.6 live rows.

**Finding, 2026-10-05: in-graph two-batch overlap is closed on neuronx-cc 2.27 (SDK 2.32).** Do not re-try it by
restructuring Kiln's graphs on this compiler: the probe above shows 0 ms of independent compute inside any collective
window in 9 graph shapes and 3 backend scheduler options, and the prefill call's "idle with a collective in flight" time is
~117 ms of EP imbalance (which no overlap can remove) plus ~64 ms of transfer. The one remaining route to overlap is
collectives issued from inside an NKI kernel, so that the kernel interleaves them with its own tiles. nki 0.6.0 ships
`nki.collectives` (all_reduce, all_gather, all_gather_v, reduce_scatter, all_to_all, all_to_all_v, collective_permute, rank_id,
ReplicaGroup; `nki/collectives/__init__.pyi`; HBM or SBUF tensors, coalesced lists on HBM, "priority ... NeuronCore-v4+
only"), and `nisa.sendrecv` ("Available only on NeuronCore-v3 or newer", within one LNC). Not built and not measured on trn1;
what it would need: a probe that an `nki.collectives` reduce_scatter / all_gather inside an LNL-compiled kernel runs on
NeuronCore-v2 at 32 ranks with the world's replica groups, that it coexists with the graph's XLA collectives (the replica
group signature check that breaks cached all-gather NEFFs, above), and then an EP kernel that starts its first expert tiles
while the rows' gather is still arriving and reduce-scatters finished row blocks while later tiles compute.

**Levers, measured on kiln-ut-32 against the same-box G64 base (122.7 out tok/s; 128 requests, farm graphs,
`NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`, logs s3 logs/kiln-ut-32/ut-<tag>.log with `.log.cmd`):**

| lever | what it attacks (from the tables above) | bound | measured |
|---|---|---|---|
| a second decode bucket, `--decode-buckets 8,16` (config only) | padded decode rows in the burst start and drain (4.7% of the conc-64 wall) | ~3% | 126.7 (+3.3%); 128 of 640 decode calls in the 8-row bucket; loads (farm estimate +0.64 GiB) |
| four decode buckets, `--decode-buckets 4,8,12,16` | the same | ~3% | 125.8 (+2.5%): loads (estimate +2.26 GiB) but no better than 8,16, so 8,16 is the set |
| EP at 4 decode rows per group (conc 16) with the MoE agent's decode v2 | conc 16's TP prefill (0.893 s per call) | +20% if EP decode cost TP's | at identical shapes (--max-num-seqs 32, KV 1.5, decode bucket 4): TP 86.9, EP v1 88.4 (+1.7%), **EP v2 98.0 (+12.8%)**; the final G16 shapes' TP: 87.5 |
| mixed batches + KDA / DSA decode kernels + SP decode streams, together | decode rows outside prefill calls; the DSA mask over 8448 keys, its spills, the KDA state writes | | 141.4 (+15.2%); with decode buckets 8,16 as well 142.9 (+16.5%: mixed batches already take the ramp's decode rows) |
| two-batch overlap in the prefill pieces | collectives with every engine idle (29% of a prefill call) | closed: no overlap from neuronx-cc 2.27 | probe above |
| expert load balance (EPLB, the techniques agent) | the busiest rank's MoE, ~117 ms of the 592 ms prefill call | ~+12% at conc 64 | its own A/B: one redundant slot per rank, prefill call 0.592 -> 0.529 s, 122.9 -> 128.7 |
| host scheduling / launch | ~0.4% of the wall | ~0 | not worth a change |

**Correctness of the levers** (`tools/check_mixed.py` through each serving configuration's own graphs, ASSERT_CACHE_HIT,
kiln-ut-32; LONG_TEXT prompts of 8192 to 700 tokens, 64 greedy tokens, top-2 logprobs; `--compare` teacher-forces every
position before a pair's first difference; logs s3 logs/kiln-ut-32/cm-<tag>.log and .json):

| comparison | outputs equal | decode-path signed mean dlogprob (nats / token) | mean / p99 / max \|d\| | first-difference margins |
|---|---|---|---|---|
| G64 base vs base (32 prompts at conc 32, run to run on this box) | 32 / 32 | 0 | 0 / 0 / 0 (deterministic) | |
| G64 base vs `--decode-buckets 8,16` (8 rows per group: every decode call in the 8-row bucket) | 31 / 32 | +0.00016 | 0.00029 / 0.0030 / 0.14 | 0.125 |
| the same on wikitext-2 prompts | 31 / 32 | -0.00023 | 0.00085 / 0.031 / 0.19 | 0.0 (a tie in the reference) |
| G16 TP vs EP v1 (16 prompts at conc 16) | 15 / 16 | -0.00023 (prefill chunks +0.0117 over 16) | 0.0038 / 0.10 / 0.35 | 0.125 (a prefill token) |
| G16 EP v1 vs EP v2 (feat/moe-kernel-tune 2f28ff8, `KILN_MOE_EP_SMALL_V=2`) | 16 / 16 | 0 | 0 / 0 / 0 (bit-identical) | |
| G64 base vs mixed batches + KDA / DSA decode kernels + SP decode streams (`--state-checkpoints 4`) | 29 / 32 | mixed decode rows -0.00112 (n=572), decode calls -0.00049 (n=1320); prefill chunks +0.00375 (n=32: the mixed prefill graphs) | 0.0039 / 0.085 / 0.34 (mixed rows) | 0.375, 0.125, 0.125 |

The 8-row decode graphs are not bit-identical to the 16-row ones (one flip at a 0.125 margin, the sampler's logprob step),
and the decode-path NLL does not move (+0.00016, against the decode kernels' accepted +0.00028 / +0.00015 and the earlier
floors' +0.00001 / -0.00079 on other boxes). Prefill graphs are unchanged by a decode bucket, so check_ppl, which scores
prefill, is the base's by construction. EP vs TP moves the MoE summation order (the merged EP's ppl check: -2.074 / wikitext
-0.552 vs TP -2.073 / -0.551).
The combination's decode-path NLL moves by -0.0005 to -0.0011 nats per token, the size of the individually accepted levers'
own moves (decode kernels +0.0003 / +0.0002, SP decode streams +0.0001 / -0.0007, the earlier wikitext floor -0.0008) and with
its flips at ordinary margins; it is the sum of three opt-ins each already checked alone, measured together for the first time.

**Measured and set aside** (code on feat/attn-kernel-tune, opt-in there, not in this tree):
- *The KDA short conv as shift matmuls* (`KILN_LA_CONV=block2`, 56193cd): the chunk's conv over `cat([prev, qkv])` moves
  rows across partitions (the token axis), which the compiler round-trips through HBM; per 128-row tile, two 128 x 128
  0 / 1 shift matrices on the tensor engine and no concatenation instead, bit-identical on CPU and device. In a 2-rank
  replay of layer 4 the KDA token mixer went 4.483 -> 3.997 ms with its spill halved (476 -> 275 MB); six forms were
  measured (dense shift constants 4.15, iota-built 4.52, transposed slices 7.99, conv1d 8.02). In serving it bought
  nothing: G64 with the fused kernel 132.0 -> 131.5, F0 120.0 -> 119.7 out tok/s, prefill call unchanged, and the farm's
  static count of the 32-rank graphs' spill runs went up slightly (+104K per rank). The 2-rank replay is not the 32-rank
  serving graph's layout.
- *Decode kernels, rows on the partitions / k and v rows by DMA* (`KILN_DSA_DECODE_ROWS=1`, `KILN_KDA_DECODE_ROWSRC=1`,
  1416cf5 / 9cf8131): alone 0.713 -> 0.453 ms (DSA, 16 rows) and 0.258 -> 0.228 ms (KDA, 16 rows), same error; at 11 /
  34 layers that is ~2.9 + ~1 ms of a ~135 ms G64 decode call, not taken further without a decode-path NLL check.
- *The score kernel's head sum reading PSUM* (2.96 -> 2.73 ms alone): any edit of kernels/dsa_topk.py rekeys every DSA
  graph; the fused kernel carries the same change instead.
- *A 32-core replay of a prefill-group NEFF*: `neuron-explorer capture -r 32` of rank 0's prefill group graph (group
  collectives inside the attention groups) fails at load with `NRT:nrt_load_collectives Failed to load collectives for
  model` (and without `NEURON_RT_ROOT_COMM_ID` the 32 ranks hang in bootstrap): each attention group's ranks compile their
  own NEFF, so one rank's NEFF does not replay on all 32 cores. The decode-graph recipe above needs per-rank NEFFs here.

## Suffix decoding on GLM-5.3-Flash at G64 (2026-10-05, SDK 2.32, trn1.32xlarge, tp=32, DP attention 4)

`--spec-method suffix --spec-k 1` (engine/spec_suffix.py, vLLM v0.30.0 v1/spec_decode/suffix_decoding.py) on the G64
command of the EPLB A/B (plain EP, KV 1.2 fp8, CK4, 128 requests, conc 64, q/eplb-a2b1414 graphs; tools/suffix_g64.sh;
kiln-tq-32, log s3 logs/kiln-tq-32/suf-A.log), against that section's A on the same box:

| run | out tok/s | TTFT p50 / p90 | ITL p50 | spot $ / M out | drafts accepted |
|---|---|---|---|---|---|
| A, plain EP | 122.9 | 6.9 / 79.7 s | 453 ms | 4.86 | - |
| suffix k=1 | **89.5 (-27%)** | 20.5 / 121.1 s | 565 ms | 6.67 | 15903 / 16228 (98.0%), 1.980 tokens per verify, 0.0 s drafting |

The 98% is the random-token prompts making the model repeat itself (the MTP section's acceptance table: real text
accepts much less), so this is the most favourable acceptance the workload can give, and suffix still loses 27%:
- A speculative step runs synchronously, so the overlap the baseline has is lost.
- Its verify graph costs ~1.32x a decode call under EP (the MTP section's 247 against 187 ms).
- Draftless sequences (no suffix match yet) launch their own decode graph beside the verify, because spec_verify_plain
  is off for suffix: 674 decode calls beside 16228 verified rows.
- The level is prefill-bound. The running batch averaged 43.6 of 64 (the KV line), because slower turnover held fewer
  sequences in decode.

Merging the draftless rows into the verify would at best reach sync MTP's -4.6% on the same shape, which still loses.
Overlap scheduling cannot take suffix drafts. They come from the host's token history, which a blind step does not have
yet, unlike MTP's drafts, which come from the device (engine/spec_async.py on feat/async-mtp). Suffix decoding
therefore stays off for G1 / G1b on this model. Its place is repetitive text at low concurrency (FEATURES.md: Qwen3.5-0.8B
B=1 102 -> 148 tok/s).

## Asynchronous MTP drafting under overlap scheduling (2026-10-05, SDK 2.32, trn1.32xlarge, tp=32, DP attention 4)

`KILN_SPEC_ASYNC=1` (EngineConfig.spec_async, MTP k=1, with `--overlap`; engine/spec_async.py) is the first item that
"MTP speculative decoding ... at serving scale" above named for turning MTP into a win: keep the accepted count and the
next drafts on the device, so the next step is scheduled while this one runs. The reference engines' forms:
- vLLM v0.30.0: v1/core/sched/async_scheduler.py:19-49 schedules each decode row with 1 + k tokens as if every draft is
  accepted. Model Runner V2 (worker/gpu/model_runner.py:1997-2181, worker/gpu/states.py:58-73) corrects
  num_computed_tokens on the GPU after the rejection sampler and builds the next input ids and positions from device
  buffers.
- SGLang v0.5.21: speculative/eagle_worker_v2.py:1313-1450 with managers/overlap_utils.py:513-594 publishes the new
  lengths and bonus tokens into a device FutureMap that the next batch gathers from, and reserves slots for both steps
  in flight (mem_cache/allocation_sizing.py:16-59).

**Kiln's form.** One fp32 board row per request slot holds T (the last verify's emitted tokens), acc, the newest
token's position, the KDA state row the next verify reads, a has-drafts flag and the k drafts. Seven small graphs read
and write it around the verify and MTP graphs, which are unchanged, so their compile keys are the synchronous engine's:
- spec_prep: ids, positions, KV slots through the host's page table, state rows and the sampler's draft column.
- spec_post: the accepted prefix, the emitted tokens, base + 1 + acc, and cur = the request's row acc.
- mtp_prep and mtp_post: the MTP pass from the last accepted position, then its drafts into the board.
- spec_init and mtp_prefill_ids: after a final prefill chunk, from the token board the prefill sampled into.
- spec_host_init: a row whose newest token is a prompt token the host holds (a whole-prompt cache hit, a recompute).

The scheduler gives each MTP row a blind step:
- It reserves pages for the upper bound num_computed + (1 + k) x steps in flight.
- It commits the tokens one step later, from spec_post's [rows, Q + 2] read-back.
- A preemption clears the request's board row.

Requests that need every token on the host keep the synchronous step (grammar, penalties, watermarking, a running
thinking budget: Request.host_bound). Integer arithmetic stays in fp32 below 2^24, with floor(pos / page_size) for a
power-of-two page size. No comparison is against a float literal: the first compile of every board graph failed
NCC_ESPP004 f64 ("Float-literal comparisons can lower to f64" above).

**CPU** (tests/test_spec_async.py, fp32, gloo):
- Board graphs on hand-built boards.
- GLM-5.3-Flash with its MTP layer at tp 1 and under DP attention (tp 2, 2 groups): the synchronous MTP engine's tokens,
  logprobs (< 1e-4) and accepted counts, and plain greedy's tokens.
- DeepSeek-V3 and GLM-5.3 whose MTP layer is the target's own copy (most drafts accepted).
- Oracle drafts through the KDA state rows.
- Sampled requests (temperature 0.8, seeded), the prompts twice, so DeepSeek-V3's second round goes through
  spec_host_init: tokens and logprobs equal the synchronous engine's except each request's last token. There the
  synchronous engine has no room for a draft and samples y_0, while the blind step verifies one: another exact sample of
  the same distribution.
- Whole suite on the merged tree (engine-v0 25a45c9 + feat/techniques 2209f68 + this): 719 passed, 72 skipped,
  7 failed. The 7 are tests/test_inkling.py under the tf518 transformers (KeyError 'model.llm.embed...'); they fail the
  same way on engine-v0 and pass with the venv's own transformers (7 passed).

**Serving A/B on G1b** (tools/amtp_g1b.sh: conc 64, 75% of every 8192-token prompt one of 4 shared 6144-token
prefixes, a cold level then a warm one, 256 requests per level, EP, KV 1.2 fp8; MB the default 32 checkpoint rows,
MS / MA 16 so the KDA pool keeps MB's 49 rows per group). Tree feat/async-mtp 9b849da (engine-v0 f54fc18 + this), farm
q/amtp-9692e97 (configs G64-4096-KV1.2-S20-P12-K-EPT-MB / -MA, MS runs MA's graphs), 0 device compiles, one box
(kiln-tq-32), logs s3 logs/kiln-tq-32/amtp-{MB,MS,MA}.log:

| run | level | out tok/s | wall | TTFT p50 / p90 | ITL p50 | tokens per verify (accepted) | host drafting | spot $ / M out |
|---|---|---|---|---|---|---|---|---|
| MB, no MTP | cold | 207.8 | 315.4 s | 2.3 / 26.8 s | 287 ms | - | - | 2.87 |
| MB | warm | 246.9 | 265.4 s | 2.3 / 15.4 s | 246 ms | - | - | 2.42 |
| MS, MTP k=1, synchronous | cold | 213.4 (+2.7%) | 307.1 s | 3.5 / 30.7 s | 268 ms | 1.973 (97.3%) | 19.1 s | 2.80 |
| MS | warm | 249.6 (+1.1%) | 262.6 s | 3.5 / 17.8 s | 225 ms | 1.976 (97.6%) | 16.9 s | 2.39 |
| MA, MTP k=1, async | cold | **222.6 (+7.1%)** | 294.4 s | 4.4 / 30.2 s | 252 ms | 1.965 (96.5%) | 0.0 s | 2.68 |
| MA | warm | **260.4 (+5.5%)** | 251.7 s | 3.5 / 17.4 s | 215 ms | 1.967 (96.7%) | 0.0 s | **2.29** |

Against the synchronous MTP engine the blind steps are worth +4.3% on both levels: warm 262.6 -> 251.7 s, -10.9 s
of the 16.9 s MS spent drafting on the host (with its read-back waits). Part of that goes to the blind step past each
request's last token, which is computed and dropped. Against no MTP, +5.5% warm and $2.42 -> $2.29 per million output tokens. The earlier bound,
~+9% over the baseline, was taken on a tree whose baseline was 229.6. MTP still raises TTFT p50, 2.3 -> 3.5 s, because
every prefill step also carries the verify's extra rows and the 4096-row MTP pass. The acceptance counts differ from
MS because a blind step always carries a draft, also at a request's last token where MS has none. That step's proposal
is counted and its tokens past the limit are dropped. serve_sweep's least-squares device split is not meaningful for
MA: its verify_async calls are neither decode nor prefill calls, so only the steps carrying a prefill enter the fit.

**Greedy equality on the device** (tools/amtp_check.sh: tools/check_mtp.py, 16 prompts x 256 tokens per set, the G64
graphs above, MS writes the reference, MA compares token for token; logs s3 logs/kiln-tq-32/amtp-check-{MS,MA}.log):
| set | acceptance MS / MA | greedy identical MA vs MS | matched prefix: chosen logprob bit-equal, mean / p99 / max \|d\|, signed mean |
|---|---|---|---|
| random (8192-token prompts) | 95.9% / 96.0% | **16 / 16** | 95.8%, 0.00000 / 0.0000 / 0.0004, +0.00000 |
| wikitext (2048-token slices) | 77.3% / 75.8% | 8 / 16 | 64.8%, 0.0035 / 0.080 / 0.245, -0.00030 |
| chat (24-token questions) | 86.9% / 87.2% | 6 / 16 | 71.0%, 0.0032 / 0.076 / 0.263, -0.00008 |

Every first token (the prefill's) is bit-equal. Each divergence is at a reference top-2 margin of 0.0 to 0.375 (0 to 3
bf16 ulps of the logits), and the first differing logprob falls 2 to 63 tokens into decode on natural text, 220 to 252
on random prompts, where requests start to finish. So the blind steps compute the same function, and what moves is
which other rows share a verify call. A control without async MTP shows the synchronous engine is just as sensitive
to that: MS run again with 12 prompts instead of 16 (amtp-check-MS12.json; the first 12 prompts are the same) against
MS on those 12. Chat is 7 / 12 identical, matched-prefix logprobs 89.1% bit-equal, mean |d| 0.0013, signed -0.00015.
Random is 12 / 12, but only 71.8% bit-equal against async's 96.2%. Async against sync on the same 12 chat prompts is
5 / 12, signed +0.00007. The signed means, -0.0003 to +0.0002 nats per token, are an order below the
two-neutral-engines floor of the EPLB checks (-0.0028 on wikitext).

The first attempt of MS's check failed at load on one rank with `NMGR:dlr_build_and_load_cc_resources Failed to build
and load collectives resources` for verify graph 7ed1597e (log amtp-check-MS.fail1.log), and the other 31 ranks waited
in CCOM bootstrap. The box was idle, with no other job. Killing the job and starting it again passed, with every graph
loading.

**Not measured / not done.**
- k > 1: the runner refuses it, because the later MTP passes' positions would come from the device too.
- The merged tree's defaults: engine-v0 25a45c9 turns on the decode kernels, SP decode streams and the fused DSA kernel
  on trn1, which change the verify and MTP graph keys. This A/B is on the f54fc18 base. The same three-way comparison
  on the merged head is `bash tools/amtp_g1b.sh MB MS MA` after a farm capture of its two configs.
- Prompt-heavy G1 (8192 in / 256 out with no shared prefix) stays prefill-bound; MTP is not expected to pay there
  (the section above).

## The final combined measurement (engine-v0 f70c14b / 25a45c9, 2026-10-05, trn1.32xlarge spot)

One box (kiln-ak-32), every row back to back on it, every graph from the farm with `NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`
(0 device compiles), 128 requests per level, GLM-5.3-Flash real weights, 8192 in / 256 out. Logs
s3://<your-bucket>/logs/kiln-ak-32/ab-fin3-<config>.log, each with a `.log.cmd` holding the exact command.

**What each config used.**
- Tree: engine-v0 f70c14b for the plain and mixed-batch rows (= e240cf0: the fused DSA prefill kernel, the trn1 decode
  kernel and SP decode stream defaults, MoE decode v2 `KILN_MOE_EP_SMALL_V=2` as the default and EP at 4 decode rows per
  group (54b2e56), EPLB opt-in, host-only profiling off; plus notes); engine-v0 25a45c9 for the EPLB rows (f70c14b +
  `eplb.decode_replicas()` reading `moe_ep.SMALL_V`, so under v2 decode pairs stay on the primaries; its graphs outside
  EPLB are f70c14b's). Farm queues q/final-f70c14b and q/final-25a45c9, each from a clean export of its tree.
- Env, every row: the q/final-ebe237e serving env, `KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer
  KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki
  KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1
  KILN_SP_ROUTE=1`; nothing else for the defaults (the fused kernel, the decode kernels, SP decode streams, v2 and the EP
  gate are trn1 defaults).
- Argv, every row: `bench/serve_sweep.py --model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4
  --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --prefill-tokens 4096
  --prefill-buckets 1024 --requests 128 --max-seconds 3000 --price trn1.32xlarge-spot=2.15`, plus G16 `--max-num-seqs 16
  --decode-buckets 4 --kv-cache-gb 0.65 --concurrency 16`, F0 `--max-num-seqs 32 --decode-buckets 8 --kv-cache-gb 1.5
  --concurrency 32`, G64 `--max-num-seqs 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8 --concurrency 64`.
- MX (mixed batches): `KILN_MIXED_BATCH=1`, `--state-checkpoints 4 --decode-buckets 8,16`.
- EPLB: `KILN_EP_REDUNDANT=1 KILN_EPLB_INIT=/opt/kiln/eplb-init-random.pt` (s3 logs/kiln-tq-cpu/eplb-init-random.pt, the
  techniques agent's initial placement), `--eplb-rebalance` and two levels (`--concurrency 64 64` / `32 32`): level 1 on
  the initial placement, level 2 after the online rebalance from level 1's counts. KV stays 1.5: the farm's
  `tools/hbm_estimate.py` puts G64-EPLB / F0-EPLB / G64-MX-EPLB at <= 15.48 / 15.05 / 15.92 GiB per rank at the
  calibrated <= ~148 B per spill run (the redundant slot is +1.015 GiB of tensors), and all three loaded.

| config | out tok/s | TTFT p50 / p90 | ITL p50 | spot $ / 1M out | prefill call / decode call |
|---|---|---|---|---|---|
| G16 (f70c14b) | **105.4** | 5.43 / 5.46 s | 131 ms | **$5.67** | 0.519 / 0.076 s |
| F0 | **133.9** | 5.61 / 24.5 s | 213 ms | **$4.46** | 0.521 / 0.088 s |
| F0 + EPLB (25a45c9), level 1 / level 2 | 143.2 / **143.3** | 5.08 / 21.4, 5.03 / 21.4 s | 201 ms | $4.17 / **$4.17** | 0.461 / 0.094, 0.456 / 0.086 s |
| F0 + MX (second attempt; the first failed at load, below) | 128.8 | 5.92 / 26.5 s | 221 ms | $4.64 | (the split does not separate mixed calls) |
| G64 | **156.2** | 5.81 / 65.1 s | 363 ms | **$3.82** | 0.520 / 0.118 s |
| G64 + MX | 152.1 | 5.96 / 68.9 s | 375 ms | $3.93 | (the split does not separate mixed calls) |
| G64 + EPLB (25a45c9), level 1 / level 2 | 166.4 / **167.2** | 5.31 / 58.9, 5.29 / 58.9 s | 338 / 339 ms | $3.59 / **$3.57** | 0.458 / 0.125 s |
| G64 + MX + EPLB, level 1 / level 2 | 163.4 / 163.3 | 5.41 / 62.4, 5.37 / 62.2 s | 348 / 346 ms | $3.65 / $3.66 | |

Against ebe237e's final standing (87.3 / 112.3 / 122.9 at conc 16 / 32 / 64): +21% / +19% / +27% by the defaults, and
+28% / +36% at conc 32 / 64 with EPLB. The prefill call drops 0.592 -> 0.520 s at every concurrency (the fused DSA
kernel; at G16 0.893 -> 0.519 s with the EP gate as well), EPLB takes it to 0.46 s. Per change, from the A/Bs that
promoted them: the fused kernel +7.4% at G64, the decode kernels + SP decode +12.6% on top, v2 +5.1% (148.6 -> 156.2
here, with the rest of the merge), EPLB +7.0%. The online rebalance moves the number by -0.1 to +0.5% against the initial
placement (each rebalance moved 186-348 redundant slots over all ranks and layers, 4.0-5.8 s).

**Mixed batches stopped paying on this tree** (G64 156.2 -> 152.1, -2.6%; F0 133.9 -> 128.8, -3.8%; on ebe237e with the
decode kernels they were +3%). The likely reason is in the code, not measured: a mixed call runs the joint mixers
(`KILN_MIXED_MIXERS=joint`): models/mla.py `attention_joint` attends the chunk through `_core` (the static attention
kernel after a separate selection) and the decode rows in the mask form, and linear_attn.mix runs the decode rows'
recurrent form in XLA, so the chunk loses the fused selection-and-attention kernel and the mixed decode rows lose the DSA
and KDA decode kernels. The device split cannot attribute it (it reports a negative decode-call time when decode rows
ride in prefill calls). F0 + MX's first attempt (log ab-fin3-F0-MX.log) failed at load in the collectives' bootstrap
(`CCOM WARN Unexpected message type ... rank 22 awaiting root parameters`, then `ENC:ncclInitGlobalComm failed` and
`Failed to build and load collectives resources`), one minute after the previous run on the box ended; the second
(ab-fin3-F0-MX-r2.log, started after other runs) loaded and ran, so it reads as a transient of back-to-back starts.

**Quality of the defaults** (f70c14b, against ebe237e):
- Wikitext-2 at DP attention 4 (`tools/check_ppl.py`, wt-DP4, the earlier gate's command): **-0.548** (-0.54753 over
  3071 tokens) against ebe237e's -0.547 (-0.54723); by chunk -0.739 -0.701 -1.086 -0.925 -1.216 -0.174 -0.345 -0.141
  -0.290 -0.264 -0.348 -0.338. Per token it is identical to the fused kernel's run on 54ec5df (max |d| 0): v2 and the
  EP gate do not reach check_ppl's graphs.
- Greedy text (`tools/check_mixed.py`, the utilization agent's shape: the G64 argv with `--concurrency 32 --requests 32`,
  64 greedy tokens, top-2 logprobs, compared with its ebe237e base dumps from kiln-ut-32; ebe237e's base rerun on this
  box is bit-identical to them, 32 / 32 with |d| = 0, so the cross-box floor is zero):

| prompts | outputs equal | decode-path signed mean dlogprob (test - reference, nats / token, +/- SE) | mean / p99 / max \|d\| | first-difference margins |
|---|---|---|---|---|
| LONG_TEXT | 28 / 32 | -0.00039 +/- 0.00030 (n = 1839) | 0.0016 / 0.048 / 0.31 | the four 700-token prompts: 0.375, 0.75, 0.125, 0 |
| wikitext-2 | 7 / 32 | -0.00113 +/- 0.00237 (n = 1010) | 0.035 / 0.29 / 0.87 | 25 flips: 23 at <= 0.25, one at 0.5, one at 1.5 (prompt 700, output 8) |

**Where the wikitext-2 jitter comes from** (the same check, each stage's graphs on this box, wikitext-2 prompts; logs
cm-attr-*-wiki and cm-cmp-*; teacher-forced over decode calls):

| stage | against | equal | signed mean +/- SE | mean / p99 / max \|d\| | prefill-chunk tokens |
|---|---|---|---|---|---|
| ebe237e base, rerun | ebe237e base (kiln-ut-32) | 32 / 32 | 0 | 0 / 0 / 0 | identical |
| + decode kernels + SP decode (ebe237e, DKS) | base | 5 / 32 | -0.00192 +/- 0.00172 | 0.024 / 0.21 / 0.69 | identical |
| + fused prefill kernel (54ec5df FU + DKS) | DKS | 4 / 32 | -0.00256 +/- 0.00236 | 0.034 / 0.29 / 0.52 | mean \|d\| 0.036, -0.0037 |
| + v2 + EP gate (f70c14b defaults) | FU + DKS | 9 / 32 | +0.00105 +/- 0.00165 | 0.024 / 0.21 / 0.54 | identical |

No single stage carries the tail: each numerical change adds per-token jitter of about the same size (mean |d|
0.024-0.034), none moves the mean by more than ~1.3 standard errors, and the LONG_TEXT check, whose SE is 8x smaller,
puts the whole default stack at -0.0004 +/- 0.0003. Greedy flips follow from that jitter at the positions where wikitext
continuations are nearly tied (most first differences sit at margins <= 0.25).

**The fused kernel's prefill numerics, decode-free** (check_ppl wikitext, DP 4, 3071 tokens, per token against XLA):
XLA -0.54723; the static attention kernel -0.54782 (-0.00059 +/- 0.00231, mean |d| 0.042, max 1.59); the fused kernel
-0.54753 (-0.00030 +/- 0.00249, mean |d| 0.043, max 1.53); fused against static +0.00029 +/- 0.00205 (mean |d| 0.033).
Both kernels move per-token logprobs by the same amount against XLA and neither moves the mean. With the selection exact
in both, that points to the attention core's summation order and rounding (P in bf16 into the P K matmul, the online
softmax's rescaling) rather than to anything the fused kernel adds over the static one, and the KV written for the
decode steps inherits it. By the rule for this pass (keep the default if the signed mean stays at the floor and wikitext NLL is
unchanged) the fused kernel stays on. **Follow-up item** (not done here): measure whether an fp32 P K accumulation in the
attention kernels, or XLA's reduction order for the softmax statistics, closes the per-token jitter (mean |d| 0.042 in
prefill); the selection itself is exact against its emulation and is not a source.

**Follow-up closed: the jitter is XLA's bf16 scores, not the kernels** (2026-10-05, kiln-pf-k1 trn1.2xlarge, SDK 2.32,
nki 0.6.0, feat/prefill-mfu 9ba9e42; `python tools/probe_dsa_numerics.py --qscale 1 4 16`, log s3
logs/kiln-pf-k1/20261005T165030Z-probe_dsa_numerics.log). Each form of the pooled-DSA prefill attention core at the rank shape
(1024 queries over 8448 keys, 8 heads, latent 512, 512 random selected pools of 4 plus the tail, causal) against an fp64
softmax attention on the same bf16 inputs, ||o - ref|| / ||ref|| of the latent-space output:

| q scale (softmax) | XLA expand (what _core ran) | XLA absorbed | XLA absorbed with fp32 operands | static kernel (dsa_prefill) | CPU emulation (P rounded after normalising) |
|---|---|---|---|---|---|
| 1 (flat: max p 0.002, 28 keys > 1e-3) | 1.48e-3 | 1.39e-3 | 1.39e-3 | **1.38e-3** | 1.39e-3 |
| 4 (max p 0.027, 230 keys > 1e-3) | 2.07e-3 | 1.15e-3 | 1.15e-3 | **0.91e-3** | 1.15e-3 |
| 16 (peaked: max p 0.56, 24 keys > 1e-3) | 4.79e-3 | 1.47e-3 | 1.47e-3 | **0.41e-3** | 1.47e-3 |

The kernel is the most exact form at every sharpness, up to 12x closer to exact arithmetic than the expand form the
wikitext comparison used as its reference. The XLA forms round the scores to bf16: the einsum of two bf16 tensors returns
bf16, and the expand form also rounds the decompressed keys. They round the normalised p to bf16 as well. The kernel keeps
S in fp32 PSUM and rounds only the unnormalised P, then divides by an fp32 row sum. "fp32 operands" changes nothing on the
device: the graphs compile with the bf16 auto-cast (neuronx_cc_args), so XLA's matmuls round their operands to bf16 whatever
their dtype. The kernel's remaining error (0.4-1.4e-3) is at the level of rounding o to bf16 after it (models/mla.py
`.to(model.dtype)`; a bf16 rounding is ~1.1e-3 rms relative). So an fp32 or hi/lo-split P K accumulation would buy nothing
visible and would cost the kernel's tensor-engine time (P K and the P^T transposes doubled). The per-token |d| of 0.042
against XLA is XLA's own rounding, and no kernel change follows.

**Quality of the best opt-in (EPLB):** wikitext is the defaults' by construction (the farm's wt-DP4-EPLB capture is
key-identical to wt-DP4, so no redundant slot enters check_ppl's graphs).
Greedy text through 25a45c9's G64-EPLB graphs (the initial placement: check_mixed does not rebalance; logs cm-eplb-*,
cm-cmp-eplb-*, cm-cmp-def-VS-eplb-*):

| prompts | against | outputs equal | decode-path signed mean +/- SE | mean / p99 / max \|d\| | first-difference margins |
|---|---|---|---|---|---|
| LONG_TEXT | ebe237e base | 28 / 32 | -0.00007 +/- 0.00040 (n = 1825) | 0.0016 / 0.036 / 0.39 | 0, 0, 0.375, 0.5 |
| LONG_TEXT | f70c14b defaults | 28 / 32 | +0.00010 +/- 0.00021 (n = 1813) | 0.0009 / 0.020 / 0.22 | |
| wikitext-2 | ebe237e base | 2 / 32 | -0.00273 +/- 0.00255 (n = 825) | 0.035 / 0.31 / 0.74 | 30 flips: 25 at <= 0.25, 0.375 twice, 0.5, 0.5625, 0.75 |
| wikitext-2 | f70c14b defaults | 7 / 32 | -0.00004 +/- 0.00199 (n = 973) | 0.029 / 0.25 / 0.54 | |

EPLB against the defaults moves no mean on either text (+0.0001 +/- 0.0002 on LONG_TEXT) and adds per-token jitter the
size of one more numerical change (a replicated expert's rows summed on its copies).

## Long context (1M)

### Kernels (feat/lc-kernels, 2026-10-05, SDK 2.32, nki 0.6.0, neuronx-cc 2.27, trn1.2xlarge kiln-lc-k1)

Two NKI kernels for the long-context DSA path (models/dsa_long.py), each with a torch emulation that is the host
path, checked against it on one NeuronCore by `tools/probe_dsa_long.py` (logs under /opt/kiln/logs on kiln-lc-k1) and
on the host by `tests/test_dsa_long_kernels.py`.

**Selection primitives on trn1** (`python tools/probe_lc_prims.py`, each against the host's definition):
`nisa.max8` (the 8 largest of each partition, duplicates included), `nisa.nc_find_index8` (the first position of each
of 8 values, duplicates paired with ascending positions), `nisa.nc_match_replace8` WITHOUT `dst_idx` and
`nisa.nc_n_gather` (GpSimd, within a partition) all run and are exact. `nc_match_replace8(dst_idx=...)` does not
compile: `[NCC_INLA001] Codegen: Unimplemented instruction ... with OpCode MaxIndexAndMatchReplace`. Rounds of max8 +
find_index8 + match_replace8 extract a row's top-k in (value descending, position ascending) order, which is the DSA
tie rule: 64 rounds (top 512) over [128, 512 / 8192 / 16384] fp32 took 0.35 / 1.68 / 3.10 ms (one call each, the launch
floor ~0.15 ms included). An indirect DMA takes ONE index per partition per instruction (a 2-D `vector_offset` is
rejected: "'src_index' free dimensions total elements must be 1"), and on trn1 it is software DGE on the GpSimd queue:
512 such gathers of one row per partition cost 0.98 / 1.21 / 1.72 ms for rows of 2 / 8 / 32 fp32.

**What paces the pooled indexer's score** (`python tools/probe_lc_score.py`: 32 heads x 512 pools x 128 queries per
chunk, per head a bf16 matmul, relu x scale and the weighted add into an fp32 accumulator; ns per head and chunk): PE
matmuls alone 225, the ACT relu alone 250, the DVE weighted add alone 1004 with one accumulator and 687 with two
(each add no longer waits for the previous one), ACT + DVE together 1121-1324 whether the relu writes in place, to
another PSUM bank or to SBUF: the two engines interfere. A DVE `scalar_tensor_tensor` on GpSimd does not compile. An
ACT activation whose scale or bias is an immediate costs two DVE memsets per instruction (1.7 ms of a 23.5 ms
selection call at 262,144 pools, from the profile); a copy activation only takes an immediate bias (NCC_IBVF043).

**kernels/dsa_long_select.py**: the exact top-keep pools of N queries sharing one context's pool keys (a prefill
chunk), candidates a prefix p < npool(q). Queries on the partitions, 128 per tile in a device loop; per 512-pool chunk
(its prologue issued one chunk ahead): pool keys by DMA, PE transposes, 32 head matmuls, relu in PSUM, the head sum in
two fp32 chains (even / odd heads; `emulate_scores` follows that order), scores to an HBM scratch and sub-block maxima
into SBUF; level 1 the top-keep sub-blocks by extraction rounds, their indices sorted ascending, their scores gathered
back (one indirect DMA per sub-block column), level 2 the top-keep candidates by extraction, mapped to pools by
`nc_n_gather`, then pools ascending with their scores. The sub-block size (`pick_sub`) minimises the extracted values
(16 at 262,144 pools; one level up to 8,704). Output (pools [N, keep] ascending, 0 past the count; count; the exact
fp32 score each selected pool compared, NEG_INF past the count).

Exactness gate (`python tools/probe_dsa_long.py select --rows 8 128 --pools 2112 8448 32768 131072 262144 --diag`, log
20261005T182059Z-probe_dsa_long.log; kinds pooled / randn / ties / zeros / equal / wide / ulps / short, npool per query
cycling over 0, keep - 1, keep, keep + 1, P / 2, P and random): the device's selected pools equal emulate's on every
kind at every P and N except the "ulps" kind (scores a few ulps apart) at P <= 32,768, 1-7 rows of 128, where the PE's
dot-product summation order moves a score's last bits and flips a near tie (device scores differ from emulate_scores'
in most values, max relative difference 1.4e-6 to 2.8e-4 on that kind, 1.8e-5 elsewhere); on every such row the device
selection equals the exact top-keep of the device's OWN scores (dumped through the kernel's dbg output). So the
selection is exact and the score arithmetic matches the emulation to the last bits of each dot product, as for
kernels/dsa_topk.py.

| N = 128 queries, keep 512, 32 heads | 2,112 pools | 8,448 | 32,768 | 131,072 | 262,144 |
|---|---|---|---|---|---|
| p50 per call (ms) | 1.97 | 2.65 | 4.73 | 11.91 | 21.02 |
| ns per (query, pool) | 7.28 | 2.45 | 1.13 | 0.71 | 0.63 |

(N = 8 takes the same time: the kernel is per 128-row tile. 1.37 ns per pair for dsa_topk.score_select at 1024 x 2112
pools, docs above, is the bucketed path's number.) Profile at 262,144 pools (`tools/prof_attn_kernels.py dsa_long`,
KILN_PROF_POOLS=262144, before the two accumulators): kernel 25.3 ms, DVE busy 15.2 ms (the head-sum adds 10.7, max8 /
match_replace8 / find_index8 4.4), ACT 12.5, PE 11.0, GpSimd 0.1: the selection is ~1/4 and the score ~3/4, bound by
the ACT + DVE pair above. At 1M the indexer's 1.51e12 (query, pool) pairs per sequence and 11 layers cost ~950
NeuronCore-seconds at 0.63 ns, ~30 s of a trn1.32xlarge when context parallelism spreads them over its 32 cores.

**kernels/dsa_slots.py**: kernels/dsa_decode.py's absorbed-MLA attention over each row's own slots (pool rows of 4
tokens, an additive bias), for N rows (a prefill chunk's queries), with an optional per-(row, head) log-sum-exp to
combine context-parallel partials (o = sum_r exp(lse_r - LSE) o_r). The rows run in a device loop, 4 per iteration;
every per-row input and output is a DMA at the loop register's offset; the bias sits on one partition and enters the
scores' PSUM through a K = 1 fp32 matmul; the scores are 512 tokens per matmul over the row's whole K^T tile; P and its
fp32 sum come from one ACT instruction. Gate (`python tools/probe_dsa_long.py slots --rows 8 128 1024 4096 --heads 8 64
--kv fp8 bf16`, log 20261005T183508Z-probe_dsa_long.log; 640 slots per row over a 4096-page cache, 512 random selected
pools, a partial tail, padding, one row with every slot masked): max |o - emulate| / max |o| 2.2e-3 to 3.0e-3 in every
case (dsa_decode's own kernel 2.2e-3 / 2.3e-3 on the same rows: the bf16 P), lse within 3.4e-5, the all-masked row
finite with lse ~NEG_INF.

| us per row (outputs reduced on the device) | N = 128 | 1,024 | 4,096 |
|---|---|---|---|
| H 8, fp8 latent | 40.9 (dsa_decode 43.7) | 38.4 | 38.9 |
| H 64, fp8 | 41.8 | 39.6 | 40.1 |
| H 8, bf16 latent | 34.8 (dsa_decode 36.3) | 32.2 | 32.8 |
| H 64, bf16 | 36.6 | 32.9 | 33.8 |

Rows per iteration (1024 rows, fp8, H 8 / 64): 1 row 45.9 / 48.3, 2 rows 41.0 / 43.6, 4 rows 39.0 / 39.5, 8 rows 40.3
/ 39.2 us. A row costs the same at 8 and 64 heads: it is the per-row K^T work (80 PE transposes of [128, 128] latent
blocks, each loading its block as the stationary, and their PSUM -> SBUF copies), not the heads, so the attention
costs ~8x less per (query, head) with all 64 heads of a query on one rank than with attention TP 8. A 4096-query chunk
is ~160 ms per layer per rank head-split, against ~20 ms per layer when each of 8 ranks takes 512 queries with all heads.
Trap: an earlier probe timed the call with its [N, H, R] output read back to the host, which made H = 64 look 1.5-2x
slower at N >= 1024 (82 us per row); reduce the output on the device before timing.

### What stopped 1M, and the arithmetic (2026-10-05, engine-v0 70ddc1b, from code and config.json eb9eb208)

GLM-5.3-Flash's max_position_embeddings is 1,048,576 (W1M: 1,044,480 in + 4,096 out). On engine-v0 nothing past the
dense-masked buckets ran: decoder.prep_prefill built the visibility `j <= pos` as [C, L] and `_bias` made it fp32 (16 GB
at C = 4096, L = 1M); mla.attention `_load`ed the whole context's latent, indexer rows and pool keys per DSA layer and
call (1.5 GB per layer at 1M in bf16); mla.bind_scratch allocated the selection scratch [rows, max_keys] bf16 at bind
time (8.6 GB at 4096 x 1M: the engine died at start-up); pooled_selection / block_mask / pool_index made [B, Q, L] masks
and a [C, 32, P] fp32 score tensor (137 GB at C = 4096, P = 262,144); kernels/dsa_topk.py holds MAX_W = 4096 scores per
partition (decode P <= 65,536 at 8 rows, score_select and dsa_fused P <= 4096); dsa_prefill is a dense masked attention
over all L keys with a [C, L] mask input, and the prefill MLA mode "expand" decompresses every context key in every
chunk (O(L^2 / C)); no context-parallel layout existed (the latent is replicated by MLA design, DP attention divides it
by groups only); page buckets are a geometric ladder to 32,768 pages and every DSA graph's shapes scale with the bucket.
KDA, the NoPE DSA layers' (absent) RoPE tables, positions in fp32 (< 2^24) and the host tier had no length limit.

Bytes and FLOPs (`/tmp/lc/arith.py`-style arithmetic from the config, checked in tests/test_dsa_long.py's byte test):
- KV: today 9,152 B/token under FP8 KV (latent 512 + indexer key and gates 256 in fp8, + the separate bf16 pool-key
  pieces 64; x 11 DSA layers) = 9.60 GB per 1M sequence on every rank of an attention group; the minimal layout (latent +
  fp8 pool key pieces) 5,984 B/token = 6.27 GB. KDA state 0.148 GB per sequence (34 x 64 x 128 x 128 fp32 + conv).
- Prefill: 16.07 B active parameters per token (KDA 4.68, DSA 1.37, MoE 9.56, dense 0.45) = 32.1 GFLOP, + sparse
  attention over 2,052 keys 3.3, + the indexer 0.09 at 8k (0.19 as the bucketed path computes it, every pool of the
  bucket) and 11.81 GFLOP/token averaged over a 1M prompt (23.6 at its end): 35.6 / 47.3 GFLOP per token.
- The indexer over a 1M prompt: sum over queries of their candidate pools = 1.37e11 per layer, 1.51e12 (query, pool)
  pairs per sequence. At score_select's 1.37 ns per pair (1024 x 2112, the score's DVE head sum is most of it; the 47
  radix rounds alone are 0.37 ns) that is 2,070 NeuronCore-seconds, against ~4,600 for the GEMMs at the 8k serving
  efficiency (G64: 6,917 prefill tok/s on a trn1.32xlarge), and x 8 when every rank of an attention group computes the
  same selection (engine-v0's replicated indexer).
- Decode at 1M: 0.74 GB of bf16 pool keys per row per step (0.37 in fp8) + 11.6 MB of selected latent; 23.6 GFLOP of
  indexer per row per step.

### The long path (models/dsa_long.py, models/mla.py attention_long / attention_cp, 2026-10-05)

A pooled DSA bucket past `KILN_DSA_LONG_KEYS` (16,384) keys never forms anything of the context's size per query:
- candidates: a query at pos may select the complete pools p < npool = floor((pos + 1) / 4), a prefix (no mask);
- selection: the exact top 512 (ties to the lowest pool index, all when fewer) as a list of pools and a count; on the
  host select_two_level (proof in the module docstring: the exact selection lies in the top-512 sub-blocks by maximum
  under the same rule, gathered in index order), select_reference the dense definition; a chunk on the device through
  kernels/dsa_long_select.py (above), a decode batch through dsa_topk in row groups that fit MAX_W plus a sqrt(P)-sized
  fp32 count compaction (dsa_long.compact), or with `KILN_DSA_LONG_SCORER=index` the decode agent's kernels/dsa_index.py
  for the scores (copied from feat/decode-scale 2fbe874);
- attention: the selected pools' tokens and the query's tail pool as 640 slots of pool rows with a 0 / NEG_INF bias,
  absorbed MLA over the gathered latent (kernels/dsa_slots.py for a chunk's rows, kernels/dsa_decode.py for a decode
  batch); the prep graphs pass a [*, 1] placeholder bias (made from the block table, see the MPMD trap below), and the
  selection scratch is capped at LONG_KEYS wide.
- `KILN_DSA_QSHARD=1` (opt-in): a chunk's rows are split over the attention group and each rank attends its C / A rows
  with all 64 heads (whole-head q_b, W_UK, W_UV, o_proj copies, ~1.2 GB per rank in FP8): dsa_slots costs the same per
  row at 8 and 64 heads, so per (query, head) this is 8x cheaper than attention TP 8. Without it the chunk's selection is
  still split over the group (each rank C / A queries, gathered by a zero-padded fp32 group all-reduce).
- `KILN_DSA_CP=1` (opt-in): each rank of an attention group of A holds context pools c = m A + rank (page_size / A local
  slots per page; the host's pages, prefix cache and admission unchanged), writes only the tokens it owns (unowned and
  padded tokens to a dump slot nothing reads: duplicate scatter destinations are nondeterministic), selects its exact
  local top 512, merges the ranks' (score, context pool) lists exactly (dsa_long.cp_merge: the threshold from dsa_topk's
  selection, then a binary search for the smallest context pools among its ties) and attends its own selected pools with
  every head; the partials are combined by log-sum-exp (dsa_slots / dsa_decode lse) with a group reduce-scatter onto each
  rank's heads. KV per rank is 1 / A. Use page_size = 32 x A (256 at A = 8): a rank's pools of a page are then 8
  contiguous pools, one 2 KB descriptor (the decode agent's scorer measured one 256 B pool per descriptor at 20 GB/s).
- `KILN_DSA_KV=minimal` (opt-in): V is the pool-key pieces in the KV dtype (5,984 B/token under FP8 KV), each request's
  open pool keeps its indexer rows in a per-request state row (hybrid.aux_state_shapes, write_pool_keys_minimal); not
  with MTP, verify, mixed batches or CP yet. Quality gate pending.

CPU (tests/test_dsa_long.py, transformers 5.18, fp32): two-level == the dense rule == the rule written out on adversarial
ties (plateaus across sub-block boundaries, block-maximum ties, exact zeros, signed zeros, subnormals, npool 0 / keep /
keep + 1) at sub 1-32, the proof's corner as its own test; slots == glm5_next.block_mask + causal visibility (pages of 4
and 8, chunk and decode forms); the engine on the long path == transformers' greedy tokens and the bucketed path's
logprobs (2e-5) through chunks that cut pools, a prefix hit, separate / inplace / off pool caches and FP8 KV, also at
1,500 tokens (375 pools vs 4 kept); tp=2 sharded selection, QSHARD at tp=2 and CP at tp=2 / 4 == tp=1; minimal KV ==
full in fp32 on both paths; cp_merge == the global rule at A = 2 / 4 / 8; the index scorer's layout == dsa_long.scores.

Device traps found on the way (trn1, SDK 2.32, neuronx-cc 2.27):
- **A concatenate of small int64 / bool pieces failed neuronx-cc with `[NCC_IFML902] FlattenMacroLoop error: Pelican
  exception: Cannot remove an edge that is not found`**: the slots' `torch.cat([pools, tail, zeros])` and its bool mask
  at keep 8 (index_topk 32), and engine-v0's own glm5_next.decode_slots at the same config (random GLM-5.3-Flash,
  KILN_DSA_DECODE_KERNEL=nki). dsa_long.slots now builds rows and bias by broadcasting (gather + where), no concatenate.
- **An `importlib.util.find_spec` inside a traced function breaks dynamo** ("skip reason: <missing reason>"): module
  constants only.
- **Float-literal comparisons (`s > VISIBLE`) lowered to f64 again (NCC_ESPP004)** in the CP merge: compare against
  `torch.full_like`.
- **`MPMD execution is not supported. Most likely some ranks recompiled/reloaded a graph`** at tp=2 in
  tools/check_device.py's teacher-forced pass, on engine-v0's bucketed path too (KILN_DSA_DECODE_KERNEL=xla): a prep
  graph identical for the plp and non-plp prefill callables is one cache key loaded twice, and the two loaded copies'
  collective barrier fails. The long path's prep graphs are now made from their block table (a graph per bucket), and
  multi-rank device checks compare `--no-reference --out-json` against a `--host-reference-only` file.

Random GLM-5.3-Flash (tools/build_random_hybrid.py --sparse, index_topk 32), bf16 on trn1.2xlarge against Kiln's fp32 host
path, KILN_DSA_LONG_KEYS=64: tp=1 teacher-forced max |dlogprob| per prompt 0.491 / 0.749 / 0.498 / 1.184 / 0.468 / 1.730 /
0.159 / 0.501 on the long path and 0.491 / 0.749 / 0.498 / 1.185 / 0.468 / 1.730 / 0.159 / 0.501 on the bucketed path
(XLA decode); tp=2 (page 64) greedy tokens matched 2 / 17 / 32 / 25 / 0 / 6 / 12 / 13 of 32 with CP and 2 / 17 / 32 / 26 / 0
/ 6 / 12 / 13 replicated (a random model in bf16 diverges early at margins 0.006-0.24 either way).

### Real weights on trn1 at 16k and 128k (2026-10-05, trn1.32xlarge kiln-lc-32 spot, SDK 2.32, feat/long-context bb0deaf)

GLM-5.3-Flash (eb9eb208, FP8 weights and KV), tp=32, DP attention 4, the trn1 serving env (KILN_CC_ARGS=--model-type=
transformer KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki
KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1
KILN_SP_ROUTE=1), prefill 512 tokens per step (128 per group), every graph from farm queue q/lc-trn1-a with
NEURON_LIBTORCH_ASSERT_CACHE_HIT=1. Text: tools/fetch_long_text.py (Project Gutenberg: War and Peace first, 2.94M GLM
tokens in all). Logs and exact commands: s3://<your-bucket>/logs/kiln-lc-32/lc-<run>.log, .log.cmd, .json.

- **Long path vs the bucketed path** (`python tools/check_long.py nll ... --max-tokens 16320`, A `--page-buckets 512`,
  B `KILN_DSA_LONG_KEYS=4096 --page-buckets 128 512`; per-position logprobs from the .json): positions < 4096 identical
  (both bucketed); 4096-8191 mean(B - A) +0.00009 nats, mean |d| 0.0141, |d| > 0.01 at 9.8%; 8192-16319 -0.00056,
  0.0144, 10.2%; max |d| 1.93 / 2.06. For scale, two compiles of one model differ by mean |d| ~0.05 (the DSA top-k
  section above). Mean NLL 0.0878 (A) / 0.0881 (B). The bucketed path cannot run past 16,384 keys (its kernels' limits),
  so this is the longest equality check against it.
- **NLL at 128k on the long path** (lc-128kN, 131,008 tokens, `--page-buckets 512 1024 2048 4096`): mean 0.1220; by
  position band [0, 1k) 0.0670, [1k, 4k) 0.0632, [4k, 16k) 0.0958, [16k, 64k) 0.1341, [64k, 131k) 0.1214; 571 s (one
  sequence in one group, 128 tokens per call: a correctness configuration). The book is likely memorised (NLL ~0.1).
- **Needle** (lc-128kH, `tools/check_long.py needle --lengths 32000 130900 --depths 0.1 0.5 0.9`, the GLM chat format
  with an empty think block, a 7-digit number in the book): 6 / 6 at 32,000 and 130,900 tokens, 712 s for the six.

**Does the long path use the long context?** (lc-64kS: the same book tokens 65,536-131,007 scored with
`--skip-tokens 65536`, i.e. without the 64k before them, against lc-128kN's same positions with them; mean NLL
0.1678 alone vs 0.1214 with the prefix.) In 2k-token windows of the span, the two agree within the text's own
variation for the first ~20k (8k-20k: 0.100 / 0.104, 0.111 / 0.111, 0.136 / 0.140, 0.097 / 0.107, 0.104 / 0.105,
0.123 / 0.131, 0.099 / 0.089), and from ~22k of the span on the run with the 64k prefix is 0.02-0.045 lower (0.156 /
0.135, 0.157 / 0.112, 0.139 / 0.117, 0.150 / 0.130): the earlier context pays off where the book refers back to it.
Neither run shows a step at 16,384 keys, where the long path takes over from the bucketed one.

**The minimal KV layout against the full one** (lc-16kM: `KILN_DSA_KV=minimal KILN_DSA_LONG_KEYS=4096`, the same
16,320 tokens, farm queue q/lc-trn1-m, kiln-lc-32b): positions < 4096 (the bucketed path reading fp8 pool keys) mean
(minimal - full) -0.00002 nats, mean |d| 0.0048, |d| > 0.01 at 3.1%; 4096-16319 (the long path) +0.00068, |d| 0.0152,
10.0% (full long path vs the bucketed one: 0.0141, 9.8%). Mean NLL 0.0954 against 0.0961 (full, long) and 0.0958
(bucketed). The fp8 pool keys move no mean and add no spread beyond the path's own.

**The selection kernel's device loop over query tiles is wrong past the first tile** (kiln-lc-k3, nki 0.6.0,
`tools/probe_dsa_long.py select --rows 128 1024 --pools 512 2048 --keep 512`): N = 128 exact, N = 1024 112 / 496 of
1024 rows differ (rows of later tiles miss most of their pools). The kernel gate had used N <= 128 only, and the
non-CP long path splits a chunk's queries over the attention group (128 rows per rank at C = 1024, A = 8), so every
real-weight run above was one tile. The context-parallel path selects all of a chunk's queries on every rank and its
first W1M run died with `Out of bounds access` in the prefill graphs. models/mla.py `_select_tiles` now calls the
kernel once per 128 queries (`tools/lc_probe_tiles.py`: 0 of 1024 / 1024 / 384 rows differ at 512 / 2048 / 8448
pools); a call of <= 128 rows traces as before.
At 128k with the minimal layout (lc-128kHM / lc-128kNM, `KILN_DSA_KV=minimal`, q/lc-trn1-m, kiln-lc-32b): needle 6 / 6
at 32,000 and 130,900 tokens (721 s); NLL by band [0, 1k) 0.0670, [1k, 4k) 0.0632, [4k, 16k) 0.0947, [16k, 64k) 0.1337,
[64k, 131k) 0.1213 against the full layout's 0.0670 / 0.0632 / 0.0958 / 0.1341 / 0.1214.

**The minimal layout's 8k G1 gates** (feat/long-context 9656235 = engine-v0 c7c43e9 merged; the G64 serving graphs and
check_ppl's, full and `KILN_DSA_KV=minimal`, farm queue q/lc-g1m, one box kiln-lc-32c, `tools/hv_quality.sh <ppl|long|
wiki> <def|min>`; logs s3 logs/kiln-lc-32c/hv-q-*, comparisons lc-g1-cmp-{ppl,long,wiki}.txt):
- Wikitext-2 slice (check_ppl, 3071 tokens): full -0.5475, minimal -0.5474 (difference -0.0001), |dlogprob| mean
  0.0134, max 1.58, greedy agreement 0.9958; both within 0.01 of the -0.551 reference.
- check_mixed LONG_TEXT (32 prompts of 700-8192 tokens, 64 greedy tokens): 29 / 32 equal; decode-path signed mean
  (minimal - full) -0.00032 nats (n = 1898), mean |d| 0.0023; prefill chunks -0.0051 (n = 32), mean |d| 0.015.
- check_mixed wikitext-2: 3 / 32 equal; decode-path signed -0.00033 (n = 848), mean |d| 0.033; prefill chunks -0.0042
  (n = 32); every first difference at a margin of 0.25 or less (eight at 0.125 or less).
- Against the floors of "The final combined measurement" and the NKI-gather section (two numerically neutral engines:
  LONG_TEXT 28-30 / 32, wikitext 4-18 / 32 at mean |d| 0.015-0.034): no bias, the flips at the floor's margins.

**The G64 default graph keys on the merged tree** (compile_farm capture, ranks 0, 8, 16, 24, the G64 argv of
tools/hv_ab.sh): engine-v0 c7c43e9 and feat/long-context 9656235 give the same 15 keys on each rank (33 distinct over
the 32 ranks, every one already compiled in the trn1 cache).

### 1M on the device: W1M measured (2026-10-05/06, SDK 2.32, every graph from the farm, NEURON_LIBTORCH_ASSERT_CACHE_HIT)

W1M = 1,044,480 input + 4,096 output tokens per request (max_position_embeddings 1,048,576), bench/serve_sweep.py, one
request per DP-attention group, so a level of 4 per engine. Logs and exact commands: s3 logs/<box>/<log>(.cmd).

| | trn2.48xlarge half (cores 32-63, kiln-t2-cb) | trn1.32xlarge (kiln-lc-32) |
|---|---|---|
| layout | minimal KV 6.0 GiB per rank, replicated over the attention group | full KV 1.5 GiB, CP over 8 (page 256) |
| tree / queue | e5ff256 / q/lc-trn2-m | e5ff256 + 51afd8a / q/lc-trn1-cp3 |
| log | lc-t2-W1M-M | lc-W1M-CP3 |
| warm-up request (one 1M request alone) | 2266.7 s | 2173.5 s |
| level: 4 requests, wall | 2237.2 s | 2166.3 s |
| TTFT p50 / p90 | 1650.6 / 1650.9 s | 1938.4 / 1938.8 s |
| ITL p50 at ~1M | 143.1 ms | 55.4 ms |
| out tok/s (all-in) | 7.3 | 7.6 |
| input tok/s (4 x 1,044,480 / TTFT) | 2,531 per engine | 2,155 per box |
| $ / 1M input at spot | 0.81 ($7.40/h per half box) | **0.277** ($2.15/h) |
| decode tok/s at 1M (1 per group fits) | 4 / 143.1 ms = 28 per engine | 4 / 55.4 ms = 72 per box |
| all-in $ / request at spot | 1.15 ($7.40/h half) | **0.323** |

Four concurrent 1M requests take the wall time of one: a lone request already runs its group's 1024 rows per call, and
the four groups run in parallel. trn2 HBM at 1M (neuron-monitor during the run, cores 32-63): 23.91-24.26 GB per
logical core (tensors 19.06, model code 2.74-3.10, shared scratchpad 1.88, runtime 0.04); the full layout replicated
had failed to load (23.857 GB used at the allocation failure, above).

**Where a trn1 CP call goes, by position** (lc-1M-TL on kiln-lc-32b: the CP3 configuration with one request and
KILN_TIMELINE, host-side only; tools/lc_timeline.py; the measured request, after the warm-up):
- Prefill call (1024 rows): 1.766 s at every position in the 1024-page bucket (local pools 8,192 per rank, contexts up
  to 256k) and 1.955 s in the 4096-page bucket (32,768 local pools): flat inside a bucket, because the scores and the
  selection run over the bucket's pools, masked past the context. The bucket step, 0.19 s for 24,576 more local pools
  per rank, is 11 x 1024 x 24,576 x 0.68 ns: the selection kernel's own rate (0.63 ns per pair at N = 128).
- The rest is fixed, 1.70 s against the G1 call's 0.47 s. At the kernels' measured rates, slot attention is 0.445 s of
  it (kernels/dsa_slots.py, 1024 rows x 64 heads x 640 slots: 39.5 us per row, x 11 layers); ~0.79 s is not yet
  attributed (the CP collectives: q_all gather 67 MB and fp32 partial reduce-scatter 134 MB per layer, the merge's
  dense tie search, the pool-key writes). A util_report replay of call prefill:600 is the measurement in progress.
- Decode step at ~1M: 54.4 ms median (55.3 at the end), 4096 steps.
- Longest host-blocked section: 1.96 s (a prefill read-back). The 143.5 s, 87.6 s and 45.0 s steps are graph first
  loads in the warm-up, which the exec watchdog does not watch, so the 300 s default holds for 1M.

**Needle at 1M** (lc-needle-1M-CP, kiln-lc-32: `tools/check_long.py needle` with the W1M-CP3 engine configuration,
`--max-model-len 1048576 --decode-buckets 1 --state-checkpoints 0 --overlap`, its graphs from q/lc-trn1-cp3 under
ASSERT_CACHE_HIT, the key check by `compile_farm.py capture --tool check_long` + `check`: 54 graphs, 0 missing): **9 / 9**
at 131,072 / 524,288 / 1,044,480 tokens x depths 0.1 / 0.5 / 0.9, 3448.8 s for the nine (four groups, one 1M request
per group). The answers are the 7-digit numbers, then the model's own continuation.

**Where the trn1 CP call goes inside, from a replay** (KILN_CAPTURE_INPUTS of call prefill:600, ~600k tokens into a lone
1M request, the 4096-page bucket; `tools/util_report.py replay` on every rank with its own inputs, rank 0 profiled, then
`bins`; kiln-lc-32b, s3 logs/kiln-lc-32b/lc-bins1m.tgz). Pieces: prep 3.9 + 510 + 540 + 542 + 376 + post 5.3 ms = 1.98
s (serving: 1.955). One CP DSA layer is ~108 ms (107.9 / 107.7 / 108.9 in piece 1), by segment between collectives:
- 38.5 ms (vector-heavy): projections, indexer, the K / V / pool-key writes, and the local selection kernel, whose rate
  puts ~21 ms of it there (8 tiles x 128 queries x 32,768 local pools x 0.63 ns).
- 3.1 ms: the CP gathers (vals, cpool, q_all as zero-padded all-reduces of 8-16 MiB).
- 60.7 ms (tensor 18.5, vector 17.1, scalar 11.9): cp_merge, the local list's compaction, slot rows, the q_lat einsum
  and dsa_slots (~40.5 ms at its measured rate, 1024 rows x 64 heads x 640 slots).
- 1.2 ms the lse gather, 4.3 ms the combine (the fp32 partial's reduce-scatter, 128 MiB in 2.06 ms) and o_proj's.

Per call (11 DSA layers, 1.19 s): slot attention 0.445 s, the local selection 0.23 s (0.06 in the 256k bucket),
merge + compaction + slot rows + q_lat 0.22 s, projections + indexer + writes 0.19 s, CP collectives 0.07 s. A KDA
mixer is 3.0 ms and a MoE FFN segment ~13.9 ms. The non-DSA ~0.76 s per call is a configuration difference: the
automatic expert-parallel default (models/decoder.py) is off below 16 sequences at DP attention 4, and W1M runs
--max-num-seqs 4, so its MoE is tensor-parallel; the default's own G16 measurement had the prefill call 0.895 s (TP)
against 0.592 s (EP v2).

**The CP configuration at 300k** (lc-300k-CP8, kiln-lc-32c, feat/long-context f997d35, q/lc-trn1-cpc, one request of
307,200 tokens, `--max-model-len 1048576` for the W1M engine): TTFT 538.0 s, ITL p50 50.0 ms.

**Expert parallelism for W1M** (the same box and request, `KILN_MOE_EP=1`, lc-300k-CP8E): TTFT **436.7 s** against
538.0 s (-18.8%; the prefill call 1.79 -> 1.46 s, the -0.3 s of the EP rule's own G16 measurement), ITL 60.5 ms
against 50.0 (+10.5 ms: the EP decode kernel at one row per group). For a W1M request (1,020 prefill calls, 4,096
decode steps) that is -347 s of TTFT against +43 s of decode, so a colocated W1M engine wants EP; a PD decode-only
engine at one row per group does not.

**TTFT by prompt length, one request alone** (the measured lone requests on trn1; on trn2 the cumulative prefill wall of
W1M-M3's timeline, whose four groups each prefill their own request in the same calls):

| prompt | trn1 CP over 8 (f997d35, TP experts) | trn2 one engine, minimal KV |
|---|---|---|
| 131,072 | 225.8 s (lc-ttft-131072-CP8; ITL 46.0 ms) | 185 s |
| 307,200 | 538.0 s (lc-300k-CP8; with EP 436.7 s) | |
| 524,288 | 952.4 s (lc-ttft-524288-CP8; ITL 50.0 ms) | 806 s |
| 1,044,480 | 1938.4 s (lc-W1M-CP3) | 1650.6 s (t2-W1M-M), 1651.7 s (t2-M3-A) |

**trn2: the minimal layout's 1M decode through the fp8 index scorer** (t2-M3-A, kiln-t2-cb cores 0-31, feat/long-context
d1378cf, q/lc-trn2-m3: 18 new graphs, the decode graphs; `KILN_DSA_LONG_SCORER=index` now takes V's fp8 pool-key pieces,
the decode agent's REV8 kernel; 4 x 1M requests, `--skip-warm-request`): ITL p50 **92.2 ms** against 143.1 ms with the
XLA scores (-36%), TTFT unchanged (1651.7 s). Decode at 1M: 4 rows / 92.2 ms = 43 out tok/s per engine. The prefill
call (1024 rows per group, every group busy) by position: 0.97 s at 4k, 1.22 at 16k, 1.50 at 64k, 1.53 at 128k, 1.58
at 256k, 1.66 from 512k on (the dense part is the G1 call's 1.00-1.05 s).

**Needle at 1M on trn2** (lc-t2-needle-M3 on kiln-t2-cb, the same minimal-layout engine with the fp8 index
scorer: feat/long-context d1378cf, graphs q/lc-trn2-m3 under `NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`, LNC=2,
`KILN_DSA_KV=minimal KILN_DSA_LONG_SCORER=index KILN_SP_GATHER=xla`, tp 32, DP attention 4, KV 6.0 GiB fp8,
page buckets 512 / 2048 / 8192 / 32768, `--max-model-len 1048576`, 2026-10-05 23:10 UTC; log and .json in
s3 logs/kiln-t2-cb/): **9 / 9** at 131,072 / 524,288 / 1,044,480 tokens x depths 0.1 / 0.5 / 0.9, 2391.9 s
for the nine. The .json's `passed` is 9 and every case's `ok` is true, each answer the 7-digit number
followed by the model's own continuation. So the 1M needle passes on trn2's minimal KV plus fp8 scorer as
well as on trn1's context-parallel layout.

**Two 1M engines do not fit one trn2.48xlarge's host memory.** Both halves of kiln-t2-cb at once (two serve_sweep
processes) ended in OOM kills (dmesg 03:28-03:29 and 04:35-04:36, rank processes of 33-37 GB anon RSS); the surviving
ranks reported "Failed to schedule neff execution status=2 message=Invalid" or gloo "Connection closed by peer". A rank
of this configuration holds ~34.6 GB host RSS in serving (smaps: 34.6 GB private dirty anon, two heap regions of 21.3 and
8.6 GB), so 32 ranks are ~1.1 TB of the box's 2 TB; a G1 rank holds ~4 GB (the disaggregation agent's PD boxes). It is
not the load: with `KILN_MALLOC_TRIM=1` (new, opt-in) a rank prints 0.8 GiB after loading its weights (0.7 after the
trim), and the RSS then grows while its graphs load and first run (15.6 GB at 704 graph loads). neuron-monitor puts it all
under the runtime's host application memory (34.0-35.5 GB per runtime); the rank's NEFFs are 1.0 GB. Not attributed
further; the graphs of the large page buckets (up to 32,768 pages) are the suspects.

**CP slot classes: the first form loses, the vorder form** (tools/probe_cp_parts.py, trn1, one core, 1024 rows, A = 8, keep
512, 32,768 local pools): cp_merge 6.46 ms (its dsa_topk 2.39), slot rows + bias 7.81, q_lat 0.21, dsa_slots 40.7. The
first slot-class form's prep (compact of each row's selected local pools, then gathers) took 81.0 ms per layer: compact
of [1024, 512] to 512 outputs 18.1 ms (to 128: 4.7), and the per-element `torch.gather` of [1024, 512] the rest (~60 ms:
one DMA descriptor per element); against ~26 ms the classes save. The vorder form (fbadaa7) needs neither: the selection
kernel returns its extraction order (score descending, pool ascending) without the final sort by pool (device sets equal
to the emulation's at P = 2112 / 32768, every kind; 1-2 rows of 128 order a device-side fp32 tie differently), the merge's
selected entries then lead each rank's list, and the small buffer is the list's first 127 slots and the tail, a slice.
The local selection is ~0.42 s per call, not 0.23: a 128-query tile at 32,768 local pools takes 4.76 ms (1.13 ns per
pair; 0.63 ns holds at 262,144), and CP runs 8 tiles per rank per layer where the replicated path runs one over all pools.

**The vorder slot classes on the device** (feat/long-context 6b7f0e9 = fbadaa7 + engine-v0 2143e0b, q/lc-trn1-m2, 108
graphs compiled clean; kiln-lc-32d, the same box for both arms, EP on in both):

| | CP8E-m (EP) | CP8CE-m (EP + `KILN_DSA_CP_SLOT_CLASSES=1`) |
|---|---|---|
| 300k lone request: TTFT / prefill call / ITL | 436.7 s / 1.46 s / 60.5 ms | **320.1 s** / 1.07 s / 60.3 ms |
| W1M, 4 x 1M one per group: TTFT p50 | 1573.6 s (lc-W1M-CP8E, f997d35 tree) | **1177.1 s** (lc-W1M-CP8CE-m) |
| input tok/s per trn1.32xlarge, $ / 1M input at spot | 2,655, $0.225 | **3,549, $0.168** |
| needle 128k / 512k / 1M x depth 0.1 / 0.5 / 0.9 | | **9 / 9** (lc-needle-1M-CP8CE, 2202.4 s) |

CP8E-m at 300k reproduced the f997d35 tree's CP8E to the millisecond (436.68 / 436.65 s). The classes cut 0.39 s per
call, more than the slot attention alone (~0.28 s): the vorder kernel also skips its final sort by pool, inside the 8
selection tiles per layer.

**Lone-request TTFT on one trn1.32xlarge** (the CP8CE-m engine, its 1024 / 4096-page buckets of page 256; logs
lc-ttft-<n>-CP8CE-m on kiln-lc-32d): 8,192 tokens 8.32 s, 32,768 33.2 s, 131,072 133.0 s, 307,200 320.1 s, 524,288
581.1 s, 1,044,480 1177.1 s (the W1M level's p50: one request per group, which a lone request's group times exactly);
ITL 56.6-60.3 ms short of 1M, 64.1 at 1M. Every length pays the 256k bucket's selection at least: a short prompt belongs
on the G1 engine, whose G64 graphs give a lone 8,192-token request 3.98 s (lc-ttft-8192-G64; ITL 86.4 ms at decode
bucket 16), the 8 calls of one group's 1024 rows.

**Not measured, and why:**
- Lever 1 for a lone request (DP attention 1, CP over 32 ranks, page 1024, EP, slot classes; q/lc-trn1-m3): its prefill
  pieces at 4096 rows per group need more than a c8i.48xlarge gives. 12-MoE-layer pieces were killed at ~220 GB
  (returncode 70); 6-layer pieces ran 2+ hours, one killed by F137 after 73 min, the rest not done by the end of the
  block. A first measurement wants 2048 rows per group or 3-layer pieces.
- The 1M long-document NLL: the prompt-logprob graphs at 4096 rows per call fail to load ("Could not load the model
  status=4 message=Allocation Failure") next to the CP KV at 1.5 GiB and at 1.3 GiB. It needs smaller prompt-logprob
  calls. The needle at 1M (9 / 9 on each of three engines: trn1 CP3 lc-needle-1M-CP, trn1 CP8CE-m
  lc-needle-1M-CP8CE, and trn2 minimal + fp8 scorer lc-t2-needle-M3) and the NLL bands at 128k stand.
- The layer pipeline on the device: the stages' own prep graphs, loading only a stage's layers, and a multi-box EFA run.

**trn2: the 1M decode call with the fp8 scorer, from a replay** (KILN_CAPTURE_INPUTS of decode call 100 of a lone 1M
request on the W1M-M3 engine, all 32 ranks' inputs, rank 0 profiled; s3 logs/kiln-t2-cb/lc-dec1m-bins.tgz): pieces 0.33 +
20.77 + 26.71 + 26.80 + 19.40 + 1.20 = 95.2 ms (ITL measured 92.2). The 11 DSA layers are ~3.5 ms each (38.7 ms, 41%),
the other 34 layers 0.55-0.85 ms segments: past the fp8 scorer the long path is the smaller half of the 1M decode step.

**One long prefill over several engines, the layer pipeline** (engine/pp.py, `pp_stages > 1`, prefill and piecewise only):
stage engines run contiguous layer ranges of the piecewise plan (the split by measured time, a DSA layer weighing
`KILN_PP_DSA_WEIGHT` = 6), a later stage takes the previous stage's hidden stream ([rows, hc 4 x 4096] bf16, 32 KB per
token) instead of the embedding over disagg.py's frames, and every layer's KV, DSA history and KDA state stay on its
stage. CPU (tests/test_pp.py): 2 and 3 stage processes give one engine's first tokens, logprobs and prompt logprobs (atol
2e-5) on the long path. Not yet on the device: the stages' own prep graphs, loading only a stage's layers, the
multi-box run.

**The tie search over the bucket's bits** (`KILN_DSA_CP_MERGE_BOUND=1`, dsa_long.merge_bits): ceil(log2(context pools))
steps instead of 21 (12 at a 64-page decode bucket of page 256), exact (tests/test_dsa_long.py
test_cp_merge_bounded_bits); with the decode agent's row pieces for decode batches only (2e6e2b2: a prefill chunk's merge
stays one piece) and its identity local selection when a rank's pools fit keep (KILN_DSA_CP_ALL_LOCAL, 0079633).

## Collectives issued from an NKI kernel on trn1, and the prefill world gather as one (2026-10-05, SDK 2.32, nki 0.6.0, trn1.32xlarge)

feat/prefill-mfu. GLM-5.3-Flash real weights, tp=32, DP attention 4, every graph from the compile farm with
`NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`. Boxes kiln-pf-32 / kiln-pf-32b (trn1.32xlarge spot), kiln-pf-k1 (trn1.2xlarge).
Logs: s3://<your-bucket>/logs/kiln-pf-32/, logs/kiln-pf-32b/, logs/kiln-pf-k1/, each serving log with a `.cmd`.

**Where the 4096-row prefill call went first** (engine-v0 70ddc1b defaults; `KILN_CAPTURE_INPUTS` at prefill:20 of a G64
level, `tools/util_report.py replay --profile-all`, then `report`; logs rp-base.log, rp-base-report.txt). Rank 0 took 540 ms
replayed (the serving fit is 520), MFU 8.8%, and every engine sat idle with a collective in flight for 180 ms.

| part, per call | ms | what it is |
|---|---|---|
| world reduce-scatter, 45 x 32 MiB | 112 | ~16 ms of transfer; the rest is waiting for the busiest EP rank (EPLB's target) |
| world row gather, the zero-padded all-reduce as 4 x 8 MiB per FFN block (180 of the 233 all-reduces) | ~73 | wait 0.107 + transfer 0.298 ms each, plus 0.11 ms of compute between chunks |
| token-mixer segments, 45 (KDA 34 x ~3.1 ms, DSA 11 x ~6.5) | 175 | KDA layer 4 alone at 1024 rows (`profile_layer.py --what laparts`): delta-rule kernel 1.55, in_qkv 0.69, short conv 0.29, out 0.29, gates 0.25 ms |
| FFN segments (EP kernel, shared expert, router), 45 | 126 | the EP kernel's 9 static passes each dequantize their expert's 1536 fp8 tiles (~1 ms per layer); shared expert 0.47 ms at ~10% of the tensor engine |
| attention-group gather + reduce-scatter | ~25 | |
| prep, post, end of graphs | ~12 | |

**nki.collectives works on NeuronCore-v2 inside an LNL graph** (`tools/probe_nki_cc.py`). nki 0.6.0's
nki/collectives/__init__.pyi lists all_reduce, all_gather, reduce_scatter, all_to_all, collective_permute(_implicit),
rank_id and ReplicaGroup, and names no NeuronCore-v2 restriction for the first four.
- At 2 ranks (one chip, trn1.2xlarge) and at 32 ranks (trn1.32xlarge), all_gather, reduce_scatter and all_reduce give
  the host's values exactly (gather) or within bf16 summation (sums), on every rank.
- A cached NKI all_gather NEFF reloads in a later process, at 32 ranks too ("Local cache hit", exact values on every
  rank). The XLA all-gather does not ("A cached all-gather NEFF breaks in the next process").
- Constraints. The neuronx-cc 2.27 verifier (birverifier checkCollective) refuses both directions: "Collective
  instruction cannot read IO tensors" and, from an internal shared_hbm copy into the kernel's output, "Collective
  instruction cannot write IO tensors". nki's frontend asserts "All src & dst tensors must have the same buffer type".
  SBUF collectives fail with NCC_IBIR428 "SB CC is only supported on trn2+". So the rows are copied into
  `nl.private_hbm` scratch, gathered there and copied to the output by HBM -> HBM DMAs.
- The ReplicaGroup takes list literals (`list(range(n))` fails the tracer: "'list' expected ... got (range)").

At 32 ranks, 32 MiB per gather, a standalone graph each (p50 of 20 calls, a 4-byte sum read back). The barrier-off
numbers isolate the transfer (`NEURON_RT_DISABLE_EXECUTION_BARRIER=1`; not safe to serve with, it deadlocked a stress test).

| | XLA zero-padded all-reduce (served) | NKI all_gather + copies (kernels/sp_gather.py) | XLA reduce-scatter | NKI reduce_scatter + copies |
|---|---|---|---|---|
| barrier on (default) | 5.46 ms | 6.14 (copies through SBUF) | 5.91 | 6.16 |
| barrier off | 2.27-2.78 | **1.16** (copies HBM -> HBM) | 1.38 | 1.45 |

**A kernel's engines compute while its own collective is in flight**, unlike XLA's graphs (0 ms in 9 shapes, "Accelerator
utilization"). `probe_nki_cc.py --cases ovl --reps 256` runs, at 32 ranks with the barrier off:
- 4 all_gathers of 8 MiB row slices alone: 1.90 ms.
- 8192 bf16 matmuls on SBUF-resident operands alone, with no dependency on the gathers: 1.89 ms.
- Both in one kernel: **1.86 ms**, i.e. max, not sum.
So two-batch overlap is possible on this compiler, but only for compute inside the same kernel as the collective.

**KILN_SP_GATHER=nki: the sequence-parallel world row gather as an NKI all_gather** (kernels/sp_gather.py,
models/decoder.py _sp_gather; ca256b7 makes it the trn1 default).
- It applies from 16 rows per rank (prefill chunks; decode SP's 2 rows keep their graphs) and from 256 columns. So the
  routing's [r, 16] fp32 gathers stay the zero-padded all-reduce (c0c8074). In the replay below, the kernel form waited
  0.218 ms to start each of them (9.9 ms per call) against 0.024 ms. In serving the choice is neutral: G64 164.7 with the
  kernel, 164.2 without, prefill call 0.4765 / 0.4782 s. The wait was rank skew, which the next collective waits out
  otherwise, so a replay's per-collective wait is not a saving by itself.
- The 8-rank attention-group gather stays XLA too. As a kernel it was slower: the SP attention block went 8.86 -> 10.09
  ms (layer 3) and 6.14 -> 7.10 ms (layer 4) in `profile_layer.py --what hcblocks --part-layers 3 4 --sp`, barrier on.
  The SP FFN block went 14.24 -> 13.22 and 14.34 -> 13.37 ms. `nki-all` keeps the group form for experiments.
- The gathered rows are identical to the zero-padded all-reduce's (max |err| 0 at 32 ranks).

Serving A/Bs (128 requests; base = engine-v0 70ddc1b with q/final-f70c14b, or final-25a45c9 for EPLB; test = 3a39ec8 with
q/pf-spw-3a39ec8 / pf-eplbspw-3a39ec8; each pair on one box, back to back):

| config | base out tok/s | NKI gather | prefill call | TTFT p50, ITL p50 | spot $ / 1M out |
|---|---|---|---|---|---|
| G64 (kiln-pf-32b) | 156.1 | **164.7 (+5.5%)** | 0.5195 -> 0.4765 s | 5.82 -> 5.44 s, 363 -> 343 ms | 3.83 -> 3.63 |
| F0 (kiln-pf-32b) | 133.6 | **140.1 (+4.9%)** | 0.5215 -> 0.4781 s | 5.61 -> 5.23 s, 214 -> 205 ms | 4.47 -> 4.26 |
| G64 + EPLB, level 1 / 2 (kiln-pf-32b) | 166.5 / 167.1 | **177.3 / 177.3 (+6.5 / +6.1%)** | 0.458 -> 0.416 / 0.413 s | 5.29 -> 4.91 s, 339 -> 317 ms | 3.57 -> **3.37** |

The prefill call is 42-45 ms shorter in every row, so the gather saving stacks with EPLB. The decode call does not move,
because no decode graph changes. MFU of the prefill call at 0.413 s (G64 + EPLB + NKI gather): 11.0%.

**The call with the NKI gather** (the same replay on 3a39ec8; rp-spw.log, rp-spw-report.txt): rank 0 took 496 ms (540
before). The world gathers are now 45 all_gathers of 1 MiB per rank (wait 0.065 + transfer 0.274 ms), and every engine
sat idle with a collective in flight for 146 ms (180 before).

**Quality** (the gather moves bytes, but the graphs around it compile differently, so the outputs are not bit-identical):
- Wikitext-2 through the gather. check_ppl's default 64-row chunks give 8 rows per rank, below the kernel's gate, so
  `tools/check_ppl.py --chunk 1024` (new) runs 256 rows per group = 32 per rank (q/pf-wtc-cb91773, configs wtc-XLA /
  wtc-SPW, kiln-pf-32, logs ppl-wtc-xla / ppl-wtc-spw with .json). XLA gather -0.54759, NKI gather -0.54730 over 3071
  tokens: signed +0.00029 +/- 0.00054, mean |d| 0.0089, max 0.473, greedy agreement 0.9954. Both are within 0.01 of the
  -0.5515 reference.
- Greedy text (`tools/check_mixed.py` through the G64 graphs, 32 prompts, 64 tokens, against the f70c14b defaults' dumps
  from kiln-ak-32; logs cm-spw-*, cm-cmp-spw-*). LONG_TEXT: 28 / 32 equal, decode-path signed +0.00001 (n = 1870), mean
  |d| 0.0025. Wikitext-2: 4 / 32, decode-path signed -0.00102 (n = 743), mean |d| 0.034, prefill chunks +0.0073 (n = 32).
  Both are the two-numerically-neutral-engines floor of "The final combined measurement" (EPLB against the defaults:
  wikitext mean |d| 0.029, signed -0.00004).

**Mixed batches use the fused and decode kernels again** (models/mla.py attention_joint, models/linear_attn.py
_mix_joint; `KILN_MIXED_KERNELS`, default 1; 47e3806).
- A mixed call's chunk rows take the fused DSA kernel wherever an unmixed chunk would (fused_kernel_takes).
- Its decode rows take the DSA decode kernel and the KDA decode kernel. The KDA kernel updates the state pool in place,
  so the chunk's row is written first.
- CPU (tests/test_mixed_batch.py test_glm5_next_mixed_dsa_kernels, both pool-key forms, at the fused kernel's shapes):
  mixed equals unmixed with both kernels' emulations, tokens equal, max |dlogprob| 3.1e-6, and each branch ran.
- Device A/B (kiln-pf-32, q/pf-mx-47e3806 against q/final-f70c14b): G64 + MX 152.0 -> **157.5** (+3.6%, above plain G64
  156.1). F0 + MX 128.7 -> **132.0** (+2.6%, still below plain F0 133.6).
- The cause named in "The final combined measurement" was the right one. With the NKI gather (q/pf-mxspw-3a39ec8,
  compiled) the mixed path is not yet measured.

**Prefill piece and chunk size, with the 15M / 30M instruction limit** (`--internal-max-instruction-limit`, lifting
neuronx-cc's 5M NCC_EBVF030 cap; NxDI's qwen3_moe sets it). Estimates from tools/hbm_estimate.py summed over the 4
attention groups' graphs (read the deltas: its absolute totals overestimate):
- P = 24 (2 pieces per call; q/pf-p24-3a39ec8, kiln-pf-32): G64 156.0 against 156.1, prefill call 0.5176 s. The fixed
  cost per graph execution is not a prefill lever. Spill rings 7.68 -> 4.72 GiB and code 0.71 -> 0.53 GiB, so it frees HBM.
- Prefill 6144 (1536 rows per group, P = 12; q/pf-p6-3a39ec8): the prefill pieces' instance spill runs grow ~65% (rings
  7.68 -> 11.80 GiB). Not tried on a device.
- **Prefill 8192 as ONE 45-layer piece** (2048 rows per group, 30M limit, with the NKI gather; q/pf-p8k45-3a39ec8).
  It compiles, and its single piece has 2.97M spill runs against ~4.5M for a group's four 1024-row pieces. Estimate:
  rings 6.24 GiB, total 20.65 against the default's 22.11. It loads at KV 1.5 fp8. At G64 (kiln-pf-32b,
  pf-g64-p8k45.log): **171.4 out tok/s**, against 164.7 for the NKI gather alone and 156.1 for the base; prefill call
  0.938 s per 8192 rows (0.469 s per 4096, -1.6%), decode call 0.1115 s (0.118), 128 prefill calls instead of 256, TTFT
  p50 / p90 5.34 / 55.5 s. P = 24 at 8192 tokens is the worst layout: 14.6 GiB of rings.

**The 8192-token prefill as ONE piece, on engine-v0 8229c3d** (the hardware execution barrier on), as a serving
configuration: `--prefill-tokens 8192 --prefill-buckets 2048 KILN_PIECEWISE_PREFILL_MOE_GROUP=45`.
- Prefill pieces of more than 12 MoE layers take `--internal-max-instruction-limit=30000000` themselves
  (model_runner.prefill_cc_args, `KILN_PREFILL_CC_ARGS`; f04ad96), so every decode graph keeps its key: the farm
  captured G64 / F0 / G16 with it and only the 4 prefill pieces and the post graph were new (q/pf-p8-7ce6a12).
- Same box, same tree, back to back. Base = engine-v0 8229c3d (q/pf-spw2-c0c8074, the default's keys; for EPLB at
  KV 1.2 + CK4 q/pf-p8-7ce6a12's G64-EPLB-KV12CK4). Test = feat/prefill-mfu-merge (q/pf-p8-7ce6a12). 128 requests;
  logs s3 logs/kiln-pf-32c/pf-g64-{b2,m2,eplb-b2,eplb-m2}.log, logs/kiln-pf-32/pf-{f0,g16}-{b2,m2}.log, each with .cmd.

| config | base out tok/s | one-piece 8192 prefill | prefill call per 4096 rows | decode call | TTFT p50 / p90 | spot $ / 1M out |
|---|---|---|---|---|---|---|
| G64 | 167.8 | **174.3 (+3.9%)** | 0.476 -> 0.472 s | 0.118 -> 0.114 s | 5.37 / 59.9 -> 5.34 / 54.4 s | 3.56 -> 3.43 |
| F0 | 144.8 | **147.8 (+2.1%)** | | | 5.17 / 22.1 -> 5.21 / 20.6 s | 4.12 -> 4.04 |
| G16 | 110.5 | **112.2 (+1.5%)** | | | 5.02 -> 5.13 s | 5.40 -> 5.32 |
| G64 + EPLB at KV 1.2 + CK4, level 1 / 2 | 180.2 / 180.8 | **191.3 / 190.8 (+6.2 / +5.5%)** | 0.415 -> 0.403 / 0.413 -> 0.400 s | 0.122 -> 0.119 s | 4.83 / 53.6 -> 4.64 / 47.2 s | 3.30 -> **3.12** |

- KV 1.2 + CK4 is neutral for EPLB: on b8814ab (before the barrier) EPLB served 176.1 / 176.8 at KV 1.5 and
  176.1 / 176.8 at KV 1.2 + CK4 (kiln-pf-32b, pf-g64-eplb-v / -c). With the one-piece prefill EPLB needs it: at KV 1.5
  the load fails with "Allocation Failure" (c0c8074, kiln-pf-32, pf-g64-eplb-p8k45-spw2.log). KV 1.2 is 4399 pages per
  group against the 4224 that 64 x 8448 tokens need.
- Prefill MFU of the EPLB + one-piece call: 2 x 138.6 TFLOP / (0.800 s x 32 x 95 TFLOPS) = 11.4% (8.8% at the start of
  this work).
- Most of the gain is outside the prefill call's per-token time (-1% to -3%). The step count falls (647 -> 579 at G64,
  half the prefill calls) and the decode call is 2-4% shorter in the fit. The EP kernel's static passes amortise
  better only where the routing is balanced, which EPLB does: +3.9% alone, +5.5 to +6.2% with it.

Quality of the one-piece prefill:
- Greedy text (`tools/check_mixed.py` through G64's graphs, engine-v0 b8814ab against 7ce6a12 with the one piece,
  kiln-pf-32, logs cm-v-* / cm-p8k-*, cm-cmp-p8k-*). LONG_TEXT: 28 / 32 equal, decode-path signed -0.00005 (n = 1813),
  mean |d| 0.0010. Wikitext-2: 5 / 32, decode-path signed +0.00015 (n = 871), mean |d| 0.038, prefill chunks +0.0020.
  That is the floor.
- Wikitext-2 NLL through check_ppl --chunk 1024 at P = 12 and at one 45-layer piece (q/pf-p8w-7ce6a12, with
  `KILN_SP_GATHER_MIN_WIDTH=16`, which compiles check_ppl's tensor-parallel-expert pieces with the NKI gather; logs
  ppl-wtc-p12w / -p45w): -0.5473 and -0.5477, signed -0.00044 +/- 0.00335, greedy agreement 0.978. The per-token
  jitter is larger than any gather change's: mean |d| 0.057, one token at 3.68. The piece boundaries every 12 layers
  were where the streams rounded to bf16 as graph outputs; inside one graph that rounding is the compiler's.

**The NKI world gather with tensor-parallel experts: NCC_ISCH719, so those keep the XLA gather** (e40befb).
- The decode agent's G64-ST (`KILN_MOE_EP=0`, q/dc2-st) and this work's check_ppl --chunk 1024 configs (EP's automatic
  default is off at check_ppl's 4 sequences) failed neuronx-cc 2.27 on 4 of 12 prefill pieces with
  `[INTERNAL_ERROR] [NCC_ISCH719] topological order violations`.
- Reproduced: G64 with `KILN_MOE_EP=0` on engine-v0 8229c3d (q/pf-tp-8229c3d), the same 4 keys of 4 enqueued fail
  (5ac8a791, 4b4fca2d, ba120ca2, 66775c49).
- The same configs compile when the routing's small gather is also the kernel (cb91773's form). They also compile with
  `KILN_SP_GATHER=xla` (the decode agent's G64-STb, 23 of 23 graphs).
- The rule, DecoderForCausalLM._sp_gather_kernel_ok (tests/test_glm5_next.py
  test_sp_gather_kernel_needs_expert_parallel_experts): the world gather runs as the kernel only when the model's
  routed experts are expert-parallel or there are none.
- Checked on the farm, from e40befb's tree:
  - The same G64 + EP=0 capture compiles 12 of 12 new pieces (q/pf-tp-e40befb).
  - Its default G64 (EP) keys are set-equal to q/pf-spw2-c0c8074's (33 of 33).
  - The check_ppl --chunk 1024 keys are set-equal to cb91773's `KILN_SP_GATHER=xla` capture (38 of 38).

**Measured and set aside (2026-10-05, code on feat/prefill-mfu, not in this tree):**
- *The world gather inside the expert-parallel kernel* (`KILN_MOE_EP_AG=1`, kiln_moe_ep_ag_kernel). It is a copy of
  the dequantize-first kernel that takes this rank's rows, all_gathers them into private HBM and copies them out for
  the shared expert, so the gather can run under the plan, the output's zeroing and the first weight loads.
  - Bit-identical on all 32 ranks (`profile_layer.py --what hcblocks --part-layers 3 4 --sp` with
    `KILN_PROFILE_AG_CHECK=1`: max |d| 0).
  - Slower. The SP FFN block went 12.04 -> 13.17 ms (layer 3) and 12.65 -> 13.48 ms (layer 4) with the copy-out after
    the zeroing, and 12.04 -> 13.47 / 12.60 -> 13.32 ms with it after the passes (kiln-pf-32b, standalone block
    graphs).
- *A bf16 split of the delta-rule kernel's fp32 matmuls.* The primitives at its [128, 128] shape, one NeuronCore
  (`tools/probe_engine_rates.py 28-35`, kiln-pf-k3 trn1.2xlarge):

  | instruction | ns |
  |---|---|
  | fp32 x fp32 matmul | 192 (bf16 x bf16: 38) |
  | bf16 x fp32 or fp32 x bf16 matmul | does not compile |
  | nc_transpose of fp32 / an fp32 identity matmul | 208 / 232 |
  | hi = bf16(x): DVE / ACT | 165 / 117 |
  | lo = bf16(x - hi), DVE | 387 |

  A 3-pass bf16 product saves 192 - 3 x 38 = 78 ns of tensor engine per matmul. Splitting its two fp32 operands costs
  ~0.5 us each of ACT and DVE, and the kernel's vector engine is already as busy as its tensor engine (0.68 / 0.70 ms
  of a 1.2 ms chunk at 1024 rows). Transpose mode is no cheaper than the identity matmul in fp32. So no split form pays
  on trn1. Single-pass bf16 operands would round the recurrence's inputs; not tried.
- *A replay profile of the one 45-layer, 8192-row prefill piece.* `util_report.py replay --profile-all` holds 8
  workers' captured inputs per neuron-explorer process (~125 GB RSS each). One process was OOM-killed on the 495 GB
  host, and the other three then hung in the collectives.

**The fused DSA kernel priced by deletion: harvest item C's attention tricks (2026-10-05, kiln-pf-k4 trn1.2xlarge).**
`tools/probe_dsa_fused.py --forms fused --rows 1024 2048 --iters 20` at the attention-TP-8 rank shape (8 heads, latent
512, 8448 keys, keep 512), one NeuronCore, from exp/dsa-cmax 7d0e040. That branch's `KILN_DSA_VAR` deletes one part of
the kernel's attention at a time. It is scratch code for timing, not for merge: every variant except 0, 3 and 5 gives
wrong outputs. Logs: s3://<your-bucket>/logs/kiln-pf-k4/cm-<var>.log. p50 ms:

| KILN_DSA_VAR | what is deleted or changed | C=1024 | C=2048 |
|---|---|---|---|
| 0 | nothing (the kernel) | 4.881 | 9.483 |
| 1 | the running max: no block-max reduces and no max update; bias 0 and alpha 1 (a constant max) | 4.485 | 8.708 |
| 4 | the acc update `acc = alpha acc + P K` (DVE, [128, 512] fp32, reads PSUM), exact max | 3.980 | 7.687 |
| 5 | the l update (ACT, [128, 1]) | 4.885 | 9.468 |
| 3 | nothing deleted: ACT copies P K from PSUM to SBUF before the acc update | 5.095 | 9.895 |
| 6 | the 8 P transposes and the P^T copy per unit (P K reads P as stationary: the K-stationary ceiling) | 4.517 | 8.770 |
| 7 | 6 and 1 | 4.176 | 8.066 |
| 8 | 7, and the acc update on every second key block only (P K summed in PSUM over a block pair) | 3.811 | 7.279 |
| 2 | 1, the acc update and the l update (an accumulator held in PSUM for the whole key range) | 3.450 | 6.579 |

- The acc update is the largest item, 0.90 / 1.80 ms (18-19% of the kernel). Moving its PSUM read onto ACT costs more
  (variant 3, +0.21 / +0.41), so ACT has no room for it. The running max is 0.40 / 0.78 ms (8%). The P transposes and
  the copy are 0.36 / 0.71 ms (7.5%). The l update costs nothing measurable.
- Each trick depends on the one before it:
  - K-stationary QK yields S^T with the keys on partitions. A row max over S^T needs a reduction across partitions, so
    K-stationary needs a constant max.
  - Summing P K in PSUM across key blocks needs one fixed scale per sum (a constant max, or a block pair's max taken
    before its exp). It also needs a PSUM bank per head: 8 heads per rank, and the kernel already uses all 8 banks
    (scores 2 units x 2, P^T 2, P K 1, selection scores 1). So variant 2 cannot be reached, and variant 8 is the most
    that can.
- Per 8192-row prefill call (11 DSA layers at C=2048, the 0.938 s G64 call):
  - Constant max alone: -8.5 ms (0.9%).
  - Plus K-stationary: -15.6 ms (1.7%).
  - Plus pair accumulation: -24.2 ms (2.6%).
  - With the exact max, pair accumulation alone would save half the acc update, an estimated -9.9 ms (1.1%). That
    needs 4 key blocks in SBUF (KR = 4) and a pipeline with the QK of the next pair issued after the current pair's
    exps.
- The 2.6% depends on a safe per-row constant m0, and none has been measured. GLM-5.3-Flash's MLA scores are not
  normalized, so the only cheap bound is Cauchy-Schwarz, |q| max |k|. exp(s - m0) underflows for a key once
  m0 - s > ~87 in scaled units. If the bound overshoots the true row max by a gap g, every key whose weight relative to
  the max is below e^-(87 - g) is dropped. Once g > 87, the whole row is 0 / 0. The first step is to measure g on real
  activations, before any kernel work.

**`KILN_DENSE_FP8=0` at prefill (same box, logs df-1.log / df-0.log).** `profile_layer.py --model zai-org/GLM-5.3-Flash
--tp 32 --dp-attention 4 --ranks 2 --prefill 1024 --pages 264 --sum-readback --layers 4 --what hcblocks --part-layers 0
3`, the serving kernels, engine-v0 c7c43e9. Results with FP8 dense weights dequantized in-graph vs bf16 at load:
- DSA token mixer with its all-reduce: 7.376 -> 7.158 ms.
- SP attention block: 11.572 -> 11.367 ms.
- Shared expert: 0.491 -> 0.448 ms.
- Dense MLP: 1.648 -> 1.613 ms.
- KDA token mixer: 4.570 -> 4.550 ms. The KDA projections are bf16 in the checkpoint, so nothing changes there.

The prefill DSA layer saves 0.21 ms, about half of the 0.40 ms measured at decode, because the dequantization overlaps
the prefill layer's other work. That comes to about 4 ms per call (0.4%), for +0.24 GiB per rank, so it was not taken
to a serving A/B.

The routed-expert rows (4.37 -> 9.16 ms) are not comparable between the two runs. `random_fill` draws every
parameter from one generator, and dense FP8 weights draw bytes where bf16 ones draw normals, so the router weights
after them differ, and with them the routing and the kernel's distinct-expert work. In the real model the routed
experts are the same FP8 blob either way.

**So items 1 and 2 have no measured lever above ~1% of the prefill call with exact numerics.** The KDA mixer's XLA
parts (in_qkv / short conv / out / gates, 0.69 / 0.29 / 0.29 / 0.25 ms standalone at 1024 rows, each including the
~0.15 ms launch) are each under 2% of the call. Not profiled further.

## Upstream harvest (2026-10)

What the newest public AWS Neuron code had that Kiln did not, read 2026-10-05 against SDK 2.32 (still the newest SDK;
Kiln already ran the newest pip builds, so every delta is code). Sources, fresh clones of 2026-10-05:
aws-neuron/nki-library main 92d11f6 plus branches zifan/deepseek_v4_csa (2026-09-14) and
viekash/ncc-9479-qkv-oob-skip (2026-09-23) (the nkilib bundled in SDK 2.32 is byte for byte its origin/2.32_release,
"NKI Lib 2026-08-17"); aws-neuron/nki-moe beb4b63 and the three winners' repos it links; aws-neuron/nki-samples 44d126f;
neuronx-distributed-inference 4bcdc54 (no commit since 2026-07-28); vllm-project/vllm-neuron release-0.24.0.1.1.0 f8abae6
(the build in Kiln's venv); aws-neuron/nkipy, every branch; aws-neuron/torchtitan-neuron 45c6f395; aws-neuron-sdk a6be966.
Device runs: kiln-hv-32 and kiln-hv-32b (trn1.32xlarge spot, us-east-2c, DLAMI SDK 2.32, aws-neuronx-runtime-lib
2.34.10), engine-v0 70ddc1b, GLM-5.3-Flash real weights, every serving graph from the farm queue q/final-f70c14b under
`NEURON_LIBTORCH_ASSERT_CACHE_HIT=1` (0 device compiles). Logs s3://<your-bucket>/logs/kiln-hv-32/ and
kiln-hv-32b/, each with its `.cmd`; drivers tools/hv_*.sh.

### The runtime's per-execution barrier is the ~5 ms "fixed cost per execution"

The fixed ~5 ms per execution of a graph holding a cross-chip collective at 32 ranks ("Collectives across chips" above)
is the Neuron runtime's per-execution barrier: a cross-rank rendezvous (`enc_barrier`, a barrier NEFF the runtime runs
before each execution: "Loading barrier model for LNC", "Barrier EXECUTE" in libnrt.so.1) that the probe above never
turned off. Three knobs, all strings in libnrt.so.1 of runtime 2.34.10: `NEURON_RT_DISABLE_EXECUTION_BARRIER` (what
vllm-neuron 0.24 sets for serving, vllm_neuron/vllm/worker/neuron_worker.py:715-718, "the cross-rank rendezvous wait
(enc_barrier) is the other major component of async-decode model-submit time", and what nkipy sets by default,
nkipy/src/nkipy/runtime/__init__.py, commit 1089b54), `NEURON_RT_ENABLE_HW_EXECUTION_BARRIER` (runtime 2.27 release
notes: NEFF start overhead "up to 50%" lower "with an on-device hardware barrier between ranks"), and
`NEURON_RT_ENABLE_INTERNODE_EXECUTION_BARRIER` (multi-node, off by default).

`tools/profile_layer.py --allreduce --ranks 32 --batch 4` with `KILN_PROBE_COLLECTIVES=1` (tools/hv_barrier_probe.sh;
[4, 4096] bf16, ms per launch over 48 chained launches; logs hv-barrier-{base,off,hw}.log):

| graph | barrier (default) | `NEURON_RT_DISABLE_EXECUTION_BARRIER=1` | `NEURON_RT_ENABLE_HW_EXECUTION_BARRIER=1` |
|---|---|---|---|
| null graph | 0.190 | 0.117 | 0.127 |
| 1 all-reduce | 5.065 | **0.250** | **2.800** |
| 12 all-reduces | 4.816 | 0.403 | 2.863 |
| 12 adds (no collective) | 0.132 | 0.128 | 0.160 |
| all-gather + local sum / reduce-scatter / all-to-all | 4.831 / 5.214 / 4.979 | 0.246 / 0.247 / 0.245 | 2.748 / 2.830 / 2.747 |
| all-reduce over ranks 0-1 (one chip) / 0-7 | 0.198 / 1.868 | 0.229 / 0.235 | 0.245 / 1.815 |
| every group of 2 / of 8 at once | 0.171 / 2.287 | 0.157 / 0.172 | 0.230 / 1.459 |

The group sums the probe checks are identical in all three. So the transfer of a small collective is ~0.1 ms and the
rest was the barrier; across chips the hardware barrier costs about half of the software one. Inside one chip it is
slightly dearer (ranks 0-1 0.198 -> 0.245 ms, groups of 2 0.171 -> 0.230): on a tp=2 trn1.2xlarge it would cost
~0.05 ms per execution, which is left as is (that box is for tests; the default follows the serving case).

**In serving the barrier is mostly hidden** behind the previous execution (G64, F0 and G16 commands of "The final
combined measurement", 128 requests, kiln-hv-32, back to back; tools/hv_ab.sh; logs hv-{G64,F0,G16}-*.log):

| config | barrier (default) | barrier off | hardware barrier |
|---|---|---|---|
| G64 out tok/s (decode call / prefill call) | 156.1, again 156.2 (0.119 / 0.520 s) | 159.2 (+2.0%; 0.114 / 0.518) | **159.2, again 159.2 (+2.0%)** (0.117 / 0.518) |
| F0 out tok/s | 133.8 | 137.8 (+3.0%) | **137.7 (+2.9%)** |
| G16 out tok/s (ITL p50) | 105.3 (131.1 ms) | | **106.8 (+1.4%)** (128.8 ms) |
| G64 + EPLB (25a45c9's opt-in, q/final-25a45c9 graphs), level 1 / level 2 | 166.5 / 167.1 ($3.57) | | **170.0 / 170.7 (+2.1%; $3.50 per 1M out spot)** |
| G64 + hardware barrier + `NEURON_RT_DBG_DMA_PACKETIZATION_SIZE=65536` / F0 the same | | | 158.7 / 137.3 (no gain over the barrier alone) |

About 0.8 ms per execution comes back, not 5: a decode call is 6 executions of 15-45 ms each.

On the merged tree (engine-v0 b8814ab, whose trn1 default adds the NKI world gather `KILN_SP_GATHER=nki`; farm queue
q/pf-spw2-c0c8074 G64-SPW2, the same keys because sp_gather.py's REV covers only its kernel section, unchanged since
c0c8074; kiln-hv-32c, back to back, logs s3 logs/kiln-hv-32c/hv-G64-spw-*.log): G64 163.7 / 164.4 out tok/s with the
default barrier, **167.8 / 167.8 with the hardware barrier (+2.3%, $3.56 per 1M out spot)**, prefill call 0.479 s.

**Removing the barrier is unsafe; the hardware barrier is not.** The barrier is "the runtime's only mid-run detector of
mismatched graphs across ranks" (nkipy 1089b54), and it also orders the ranks' NEFFs. `tools/probe_mismatch.py` builds
both on purpose at 32 ranks (every case first warms every graph on every rank; tools/hv_mismatch.sh; logs hv-mm-*.log):

| case | barrier (default) | hardware barrier | barrier off (`NEURON_RT_EXEC_TIMEOUT=30`) |
|---|---|---|---|
| none (control) | | | 32 ok, one value on every rank (-189.4558) |
| shape: rank 0 all-reduces [4, H], the others [8, H] | 32 raise `nrta status=1206` at 31.6 s, "replica group signature mismatch ... likely caused by mismatched collectives between peers" | 32 raise at once (`Failed to schedule neff execution. status=2`), the same signature message | **32 return "ok" with 31 different values: silently wrong** (also without the timeout) |
| kind: all-reduce against reduce-scatter | | 32 raise at once | 6 ok, 26 raise `nrta status=1200` |
| group: world all-reduce against a group-of-8 one | | (see the log) | 24 ok (3 distinct values), 8 raise |
| order: A then B against B then A | | (see the log) | **32 ok, 27 distinct values: silently wrong** |
| missing: rank 0 skips one collective graph | **31 ranks blocked, no error, still at 120 s** (the probe's limit) | 31 ranks blocked in the next launch | 31 raise `nrta status=1200` after 62 s |
| race: 3000 launches cycling world all-reduce, world reduce-scatter, group-of-8 all-reduce, world all-gather over 4 inputs each, queued 48 deep, every rank sleeping 0-3 ms before a quarter of its launches; every output against its synchronous reference | 0 mismatches on 32 ranks (14.97 s) | **0 mismatches** (14.40 s) | **deadlock** within the first 250 launches: "Failed to receive MODEL_STOP notification from all TOPSPs", ranks 8-15 in the group graph (1 TOPSP) and all others in a world graph (7), the runtime's execution timeout (30 s by default on this runtime) fires with `NRT_EXEC_HW_ERR_COLLECTIVES`, every rank SIGSEGV |
| race, barrier off: world graphs only (ar, rs, ag) / no sleeps / one graph (ar) / world + group (ar, ar8) | | | 0 / 0 / 0 mismatches in 3000 (1.9 / 1.1 / 1.9 s) / **deadlock** |

So without the barrier, a world collective graph and an attention-group collective graph in flight on different ranks
at once can deadlock, given only host jitter, and Kiln's serving graphs mix exactly those (`KILN_SP_GROUP`, attention TP
groups). The two serving runs without it finished cleanly (~900 calls each), which only says the window is narrow.
nkipy's own Qwen-Image example forces it back on, "for correct multi-rank collectives (SPMD-with-collectives)"
(examples/models/qwen_image/qwen_image.py:687-698, 0bc6ed1).

**Taken:**
- `NEURON_RT_ENABLE_HW_EXECUTION_BARRIER=1` by default on trn1 (kiln/platform.py `HW_BARRIER_FAMILIES`, set in
  `configure_runtime_env` unless the environment chose; `=0` restores the software barrier). Quality through the G64
  serving graphs against kiln-ak-32's f70c14b defaults (`tools/check_mixed.py`, the command of "The final combined
  measurement"; tools/hv_quality.sh): LONG_TEXT and wikitext-2 prompts both 32 / 32 equal, every teacher-forced
  logprob identical (max |d| 0 over 2016 decode positions and 32 prefill chunks each; logs hv-cmp-{long,wiki}-hw.log);
  wikitext-2 at DP attention 4 (`tools/check_ppl.py`, ppl-wt-def's command) -0.5475 with every one of the 3071 token
  logprobs equal to kiln-ak-32's (`tools/compare_ppl.py`: max |d| 0, hv-cmp-ppl-hw.log). Bit-identical, as a
  synchronization change must be.
- An exec watchdog (kiln/engine/watchdog.py): the missing case shows a hang with the barrier on that nothing reports.
  Every rank marks the sections where it blocks on the device (a non-first call's launch in ModelRunner._exec, rank 0's
  read-backs in LLMEngine._collect); one lasting longer than `KILN_EXEC_TIMEOUT_S` (default 300, 0 = off) prints the
  rank's last 16 calls (name and key) and exits with code 86. On the device (probe_mismatch.py --case missing with
  `KILN_PROBE_WATCHDOG=20`): 31 / 31 blocked ranks reported and exited 86 at 20 s, with the default barrier (blocked in the
  read-back) and with the hardware one (blocked in the launch). A first call (compile or NEFF load) and the idle time
  between calls are not watched.

**trn2 keeps the runtime default** (kiln-t2-cb, trn2.48xlarge Capacity Block, ap-south-2b, LNC=2, logical cores 32-63
while the trn2 agent ran cores 0-31; an engine-v0 70ddc1b tree, its farm queue q/t2f-base, 2026-10-05 20:05-21:12 UTC;
tools/hv_t2_all.sh, logs s3 logs/kiln-hv-t2/):
- Probe, ms per chained launch, default / barrier off / hardware barrier: one world all-reduce 3.939 / 0.221 / 2.460;
  all-gather 2.826 / 0.212 / 2.365; reduce-scatter 2.914 / 0.209 / 2.421; but every group of 8 at once 0.399 / 0.173 /
  **1.756**, and ranks 0-7 0.569 / 0.195 / 1.630: on trn2 the hardware barrier makes a group collective's graph dearer.
- Race (3000 launches, the mix that deadlocks on trn1 without a barrier): 0 mismatches with both barriers; a shape
  mismatch with the hardware barrier raises on all 32 ranks at once, as on trn1.
- Serving, the trn2 agent's single-engine baseline at concurrency 64 (tools/hv_ab_t2.sh): default 94.1 / 95.5 out tok/s,
  hardware barrier 95.3 / 96.0. +0.9%, inside the default's own run-to-run spread of 1.4, so `HW_BARRIER_FAMILIES` stays
  `("trn1",)`.

**Not taken:** `NEURON_RT_DISABLE_EXECUTION_BARRIER=1` (above). kiln/engine/engine.py build_shard says why next to the
two knobs it does copy from vllm-neuron.

### NxD Inference's collective/compute overlap compiler options: no win at probe scale

NxDI compiles every context-encoding graph with `--tensorizer-options='--enable-ccop-compute-overlap
--cc-pipeline-tiling-factor=2 --vectorize-strided-dma'` and every token-generation graph with tiling factor 1
(neuronx-distributed-inference src/neuronx_distributed_inference/models/model_wrapper.py:85-107, 4bcdc54); Kiln passes
none of them (only `--model-type=transformer` through KILN_CC_ARGS on trn1), and the utilization probe above tried
other scheduler options only. `KILN_CC_ARGS` is now split the way a shell splits it (engine/model_runner.py
neuronx_cc_args), so one quoted argument can carry the spaces these need; an unquoted value splits as before, so no
existing key changes. `tools/probe_overlap.py --ranks 32` with each set (tools/hv_overlap.sh, default barrier, kiln-hv-32b;
ms, 20 queued calls each; logs hv-ov-*.log):

| case | `--model-type=transformer` | + overlap, tiling 2 | + overlap, tiling 2, strided DMA | + overlap, tiling 1, strided DMA |
|---|---|---|---|---|
| gather (zero-padded all-reduce, 4 x 8 MiB) | 5.386 | 5.763 | 5.066 | 5.570 |
| mlp (no collective) | 11.990 | 11.988 | 11.988 | 11.992 |
| gather -> mlp | 13.250 | 13.413 | 13.403 | 13.308 |
| gather and an independent mlp | 13.585 | 13.139 | 13.139 | 13.578 |
| two micro-batches | 12.704 | 12.279 | 12.241 | 12.749 |
| layer (gather -> mlp -> reduce-scatter) | 13.539 | 13.769 | 13.728 | 13.567 |
| layer on two half-batches | 13.268 | 12.660 | 12.678 | 13.309 |
| rs 32 MiB alone | 4.450 | 5.567 | 4.606 | 5.434 |

The option moves independent and two-half-batch graphs by -0.4 to -0.6 ms (-3 to -5%) and the one dependent layer, the
shape Kiln's pieces have, by +0.2 ms; nothing here would show in serving, so no farm build was spent on it. If two-batch
overlap is ever rebuilt in Kiln's graphs, re-measure with `--enable-ccop-compute-overlap`: it is the one setting under
which this compiler ran half B's gather under half A's work at all (the utilization probe above found none without it).

### Other upstream code, and where it goes

Handed to the agents that own the area (through the lead), each with its source:
- **Prefill groups past the 5M-instruction cap:** NxDI passes `--internal-max-instruction-limit=15000000` for graphs
  "over 5 million instructions" (models/qwen3_moe/modeling_qwen3_moe.py:513-543; also qwen3_vl). Kiln's prefill group
  size (P = 12) and chunk are bounded by NCC_EBVF030 at 5M ("Prefill layer groups and the instruction limit" above).
- **DVE-bound DSA prefill attention** (the fused kernel: DVE 3.3 ms, PE 2.6 of 4.92 ms): nkilib
  experimental/attention/attention_const_max.py:403-444 (P V against [V | 1] gives the row sum on the tensor engine;
  K stationary in QK gives S^T, no P transposes; a constant max removes the online max and rescale, a numerics risk) and
  core/attention/attention_cte.py:3113-3120, 3635-3767 (row max and exp-sum chained in the vector accumulator with
  `reduce_cmd`, one read-out).
- **1M context / DeepSeek-V4-style sparse attention:** nki-library branch zifan/deepseek_v4_csa (trn3-only as written:
  DMA `priority=`): experimental/deepseek_v4_csa/csa_prefill_attention.py `nki_compressor_core_kernel` :446 (softmax-gated
  pooling over 2 x ratio overlapped slots, close to GLM-5.3-Flash's pooled DSA keys), `nki_prefill_sparse_attn_kernel`
  :1255 (an O(k) per-query gather instead of a masked dense pass), the bisection threshold mask :764-786;
  csa_decode_attention.py :1552 (indexer score and top-k split over two cores, K-split gathered attention).
- **Long-context settings NxDI turns on from 32k tokens:** compiler `--internal-disable-fma-on-ios` ("reduce dma rings
  io") and `--disable-mixed-precision-accumulation` (model_wrapper.py:100-103; the compiler reference says disabling it
  "may improve performance at the cost of reduced accuracy"), `--internal-enable-dge-levels spill_reload` ("reduction
  in DMA rings memory for long context", config.py:592-593, model_wrapper.py:135-136), and the runtime's
  `NEURON_RT_DBG_SCRATCHPAD_ON_SINGLE_CORE=1` plus `NEURON_SCRATCHPAD_PAGE_SIZE` (utils/runtime_env.py:7-18,
  config.py:613-620); on trn1 the contiguous scratchpad is the default and makes the page size a no-op
  (neuron-runtime/explore/device-memory.rst:167-190). None measured here.
- **Decode MoE as one kernel with routing:** the MLSys nki-moe winner (github.com/thustorage/NKI-MoE 217759f,
  Apache-2.0; Qwen3-30B-A3B, bf16, batch 1, trn2 / trn3): kernels/moe/moe_selective.py:42 fuses RMSNorm, router, softmax,
  top-8 (`nisa.max8` + `nc_find_index8`, router_topk.py:430-456) and the experts, with Python-unrolled expert loops (no
  `fori_loop`, so no per-iteration all-engine barrier), one fused gate+up DMA per expert and the down projection over 8
  PSUM banks (selective_expert_impl.py:126-374). Kiln's decode v2 loops on the device and leaves routing to XLA.

Measured or read and set aside:
- Static DMA priority, `NEURON_RT_DBG_DMA_PACKETIZATION_SIZE=65536` (neuron-runtime/explore/compute-comm-overlap.rst:83-95:
  compute static DMA packets 4 KiB -> 64 KiB raise their priority against collective DMA): with the hardware barrier,
  G64 159.2 -> 158.7 and F0 137.7 -> 137.3 out tok/s (kiln-hv-32, logs hv-{G64,F0}-hwpk.log): nothing.
- nkilib experimental/gdn (Gated DeltaNet prefill and decode, new since 2.32): kernels/delta_rule.py already does the
  same algebra (nilpotent diagonal blocks inverted by squaring, then merge levels) at 1.7x its tensor-engine floor; gdn
  also uses `tensor_copy(engine=scalar)`, which trn1 rejects.
- `DISABLE_NUMERIC_CC_TOKEN=1` (NxDI compile_env.py:20-24): libtorch_neuronx_lite already sets it
  (libtorch_neuronx_lite/__init__.py:44-45).
- vllm-neuron 0.24's other two runtime knobs: `NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS` and
  `NEURON_RT_IO_RING_CACHE_SIZE=32` (neuron_worker.py:700-713) are already engine defaults; `NEURON_RT_MAP_HBM=1` is for
  RDMA (disaggregation).
- The explicit async API (`nrta_*`, nkipy SpikeAsync: gpt-oss-20b decode 41 -> 68 tok/s): Kiln's host is ~0 of the
  wall and the device 99.6% busy ("Accelerator utilization" above); implicit async is gone in 2.32 anyway.
- `-O1` for prefill (NxDI CTE): fails Kiln's prefill groups (NCC_EBVF030) and decode group 0 (NCC_INAS001), above.
- `NEURON_RT_ONE_THREAD_PER_CORE` (runtime 2.29: 2x collective latency for EFA proxy threads): fails to schedule here
  (above).
- nki-samples: contributed/decode_attention.py (no split-K) and attn_fwd_v10 (manual SBUF / PSUM allocation, a skewed
  software pipeline, which Kiln's attention kernels already do); every MX kernel (`nc_matmul_mx`: NeuronCore-v4 only).
- torchtitan-neuron and neuron-agentic-development: no inference kernel, flag or runtime setting not listed here.
- Bug classes worth knowing from nkilib's fixes, both met by Kiln before: `oob_mode.skip` with a sub-tensor-view source
  still writes (core/moe/moe_tkg/all_expert_mx_impl.py:2127), and duplicate indirect-scatter destinations are
  nondeterministic (branch viekash/ncc-9479-qkv-oob-skip; Kiln's padded-row fix c99a373 is the same class).

## Decode at scale: the step's fixed cost, trn2 at large batch, and the 1M-context indexer (2026-10-05, SDK 2.32, feat/decode-scale)

GLM-5.3-Flash real weights (zai-org/GLM-5.3-Flash@eb9eb208), tp=32, DP attention 4, page bucket 264 (8448 keys), engine-v0
70ddc1b defaults plus this branch (no default changed), every device graph from the compile farm with
`NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`. `tools/time_decode.py` times synchronous decode steps on random token ids with
every KV row on the null page (the work of a step at that bucket), wall p50. **Every step time and $ figure in this
section is step-only, null-page** (every row reads one page, which flatters the bandwidth, and none of them checks that
the batch's real KV fits HBM) unless it says "real KV": see "Real KV on the trn1 decode box" at its end, which also
shows that 48 rows per group at 8K do not fit a trn1 rank. Exact commands in each log's `.cmd`
(tools/dc_td1.sh, dc_td3.sh, dc_td2.sh, dc_cap.sh); logs s3://<your-bucket>/logs/kiln-dc-32/ (trn1.32xlarge spot,
us-east-2c) and .../kiln-t2-cb/ (trn2.48xlarge, the trn2 agent's capacity block in ap-south-2b, cores 32-63).

**The trn1 decode step is ~70 ms of fixed latency plus ~3.3 ms per row per DP group** (td-<cfg>-*.log):

| rows per group (per step) | 1 (4) | 4 (16) | 8 (32) | 16 (64) |
|---|---|---|---|---|
| 12-layer groups (the default: prep + 4 + post = 6 graphs) | 70.3 ms | 97.4 / 95.9 | 114.5 | 136.8 |
| 24-layer groups (KILN_PIECEWISE_MOE_GROUP=24, 4 graphs) | 71.3 | 96.5 | | 131.2 |
| one 45-layer layer graph (=48, 3 graphs) | 72.6 | 97.6 | | 137.5 |
| one graph per call (KILN_DECODE_WHOLE=1, this branch) | 71.8 | **93.3** | | **127.9** |
| NEURON_RT_DISABLE_EXECUTION_BARRIER=1 (provisional: deadlocks under skew, harvest agent) | | 91.8 | 105.8 | 128.8 |

(G16 / F0 / G64 serving shapes: 4 rows per group with EP, 0.65 GB bf16 KV; 8 rows, 1.5 GB bf16; 16 rows, 1.5 GB fp8. The 1-row
bucket is G16's.) **The graph count is not the fixed cost**: at 1 row per group the step is 70-73 ms whether the call is 1 or
6 graphs. Merging the call into one graph buys 4-9 ms at 4-16 rows (-4% / -6.5%); 24-layer groups 1-6 ms; one 45-layer layer
graph nothing. The ~70 ms is in-graph latency: ~1.56 ms per layer at 1 row per group.

**Where a call goes** (`KILN_CAPTURE_INPUTS` of one warm call on every rank, then `tools/util_report.py replay / bins / report`,
rank 0; prof-G16, prof-G64; `tools/prof_step.py` on the layers 12-23 graph, logs/kiln-dc-32/g16-002-step.txt):

| | G16 (16 rows) | G64 (64 rows) |
|---|---|---|
| call (replay) | 104.0 ms: prep 2.9, groups 22.7 / 24.7 / 24.9 / 24.3, post 4.5 | 144.2 ms: prep 3.2, groups 34.4 / 35.3 / 37.8 / 27.4, post 6.1 |
| engine busy (tensor / vector / scalar / gpsimd) | 21 / 30 / 15 / 7% | 27 / 30 / 15 / 6% |
| HBM | 52 GB/s (12% of 440) | 61 GB/s (14%) |
| collectives | 92 all-reduces of 128 KB: 0.19 ms waiting for the slowest rank + 0.25 ms transfer each = 39 ms (38%) | 227 = 52 ms (36%): 45 AR 128 KB at 0.38 ms skew wait + 0.07 (20 ms), 45 RS 512 KB 18.7 ms, 47 AR 512 KB 10.9 ms |

Per layer at 4 rows per group (layers 12-23, KKKD x 3): KDA mixer 0.386 ms (36 MB of projection weights read: 95 GB/s; tensor
engine `dot` 348 us busy, the KDA decode kernel), FFN 0.47-0.94 ms, mean 0.65 (EP v2 MoE kernel with its in-kernel CC-core
activity, shared expert, router, mHC; 35-114 MB), **DSA mixer 1.54 ms of which a `reduce-window` on the tensor engine is 415 us**
(the `cumsum` of glm5_next.decode_slots' prefix counts over 2112 pools) and fp32 dequantization multiplies ~0.39 ms (vector
388 us, tensor 196 us: the FP8 q_a / kv_a / q_b / o_proj weights converted every step), collective transfer 0.25 ms mean (the
FFN all-reduces 0.2-0.85 ms: the busiest EP rank). So the fixed cost is 45 layers x (~1.45 ms of chained small ops with every
engine under 50% busy + 2 collectives at ~0.43 ms), plus ~7 ms of prep / post.

Two changes this attribution names, opt-in on this branch: `KILN_DSA_PREFIX=mm` (glm5_next.prefix_counts: the counts as two
small triangular matmuls, exact; tests/test_glm5_next.py::test_prefix_counts_matmul_form_is_exact) and the existing
`KILN_DENSE_FP8=0`; their A/Bs follow below.

**trn2 at large batch** (kiln-t2-cb, one tp=32 engine = half the box, LNC=2, the same command with --core-base, q/dc1-t2,
td2-X.log; K by the trn2 agent on cores 0-31 with this command, 20261005T172648Z-t2-td-K.log):

| rows per group (per step) | 16 (64) | 32 (128) | 64 (256) |
|---|---|---|---|
| engine-v0's trn2 defaults (XLA KDA / DSA decode paths, TP MoE) | 143.6 ms, 446 out tok/s | 219.7, 583 | 1438.7, 178 (the XLA DSA mask form spills) |
| + KDA / DSA decode kernels + SP decode streams (the trn1 defaults; off on trn2) | 110.7, 578 | 154.9, 827 | **264.2, 969** |

Per box (two engines) that is ~1,165 out tok/s on the trn2 defaults ($3.6 / 1M out at $15/h spot, decode only, MBU ~9-10%)
and **~1,940 out tok/s with the decode kernels** ($2.2 / 1M, MBU ~10-14% by the utilization section's per-rank bytes at
725 GB/s per logical core), at 8K context. A trn2 step at 64 rows (110.7 ms) costs what a trn1 step does (136.8 ms with EP):
the fixed latency does not shrink with the faster chip.

**The W1M arithmetic (the lead's, checked against the config)**: 11 DSA layers x 262,144 pools x 128 B = 0.369 GB of fp8
pool keys per row per step; 34 KDA layers x 64 heads x 128 x 128 fp32 = 142.6 MB of state read and the same written; ~455 GB
per step at 187 rows (328 GB of weights, every expert touched, + 187 x 0.67 GB), 9.8 ms at 46.4 TB/s; 1,000 out tok/s = 5%
MBU, $0.5 / 1M at $15/h = 8.3k out tok/s = 44%; a 1M prefill ~45.5 GFLOP per token, 47.5 PFLOP per request. Two corrections
for engine-v0: (a) its DSA cache under FP8 KV is 832 B per token per layer (latent 512, indexer key and gate logits 256, bf16
pool-key pieces 64: models/mla.py cache_widths / pool_key_width), 9.6 GB per 1M sequence, and the pool keys read per step
are bf16 (0.74 GB per row); the 6.27 GB / 0.37 GB need the long-context agent's `KILN_DSA_KV=minimal` (512 + 32 B per token).
(b) Under DP attention the latent is replicated on every rank of an attention group (attention TP 8): 77 GB of HBM per 1M
sequence, about one sequence per attention group, ~8 per trn2 box with two tp=32 engines. ~187 sequences needs context
parallelism of the DSA cache (`KILN_DSA_CP=1`, feat/long-context), the minimal layout and ONE weight copy per box (tp=64; two
tp=32 engines hold 2 x 328 GB): ~7.2 GB of weights + ~3 GB of graphs per 24 GiB rank leaves ~15.5 GB, ~154 sequences.

**The decode-side indexer at 1M context** (`kernels/dsa_index.py`, `tools/probe_dsa_index.py`, one NeuronCore; scores of
every complete pool for B decode rows, each row its own context, read from the paged pool-key cache):

| | trn1.2xlarge, B=1 / 4 | trn2 logical core (LNC=2), B=1 / 4 |
|---|---|---|
| XLA (page gather, einsums: engine-v0's decode scores), 262,144 pools | 3.08 / 4.18 ms (16 rows: fails to compile) | 0.93 / 3.11 |
| the kernel, one 256-byte pool row per gather descriptor (first form) | 3.32 / 13.16 (20 GB/s) | |
| the kernel, one 2 KB page row (8 pools) per descriptor | **0.705 / 2.70 (95-99 GB/s)** | **0.497 / 1.048 (256 GB/s at B=4)** |
| the same with no gathers after the first (engines alone) | 0.412 / 1.58 | |

Exact against emulate() (3e-7 relative) on both chips and in nki.simulate (trn1 grid 1, trn2 grid 2). The profile
(`tools/prof_attn_kernels.py dsa_index`, trn1, B=1): GpSimd busy 486 us generating the indirect DMAs' descriptors (~15 ns per
2 KB page descriptor: trn1's DGE is software), tensor / vector / scalar ~240 us each. So per row per layer: trn1 ~0.68 ms on
one core (0.49 ms of it the page descriptors at 32-token pages), trn2 ~0.26 ms per logical core (the two physical cores take
alternate rows). Consequences: a gather granule under 2 KB is descriptor-bound (20 GB/s at 256 B), so under `KILN_DSA_CP=1`
with A = 8 ranks a rank's share of a 32-token page is one pool (256 B): CP=8 wants pages of at least 256 tokens; and
`dge_mode=hwdge` on these indirect gathers fails at trace on trn2 (not a lever as written). engine-v0 itself cannot trace a
1M decode step (kernels/dsa_topk.py MAX_W: 8192 scores per partition at 4 rows per group); the long-context device forms
are on feat/long-context.

**Verdict (fail-fast, sent 17:35 UTC): ~44% MBU on trn2 at W1M is not reachable on this decode architecture.** A 44% step
is ~22 ms at ~187 rows; the measured fixed latency is ~70 ms on trn1 (1 row per group) and a trn2 engine's step is 110.7 ms
at 64 rows with every kernel on. At zero per-row cost 187 rows / 0.07-0.09 s is ~2-2.7k out tok/s (10-13% MBU), and the
1M per-row work adds ~11 ms per step even spread over 128 trn2 cores. Reaching 44% needs the per-layer latency ~5x lower:
fused per-layer decode kernels (the MoE FFN first, then the KDA mixer), collectives issued from inside kernels or far fewer
of them, per-row work at bandwidth. 5% (1,000 out tok/s per box) looks reachable once ~150 1M sequences fit (tp=64 +
CP + minimal) and the step stays near the trn2 curve's ~150-260 ms at 128-256 rows.

### The fixed cost, stage by stage, and what it was (2026-10-05 evening, trn1.32xlarge, feat/decode-scale)

**At 1 row per group the collectives are rank imbalance, not latency.** `tools/prof_skew.py` over a replay of the 1-row
G16 decode call's layers 12-23 graph with every rank profiled (`util_report.py replay --profile-all --seq 2`, captured
inputs; logs/kiln-dc-32b/skew-G1-002.txt): per FFN segment the median rank computes 240-410 us and the slowest 565-877 us
(a different rank each layer), and rank 0's FFN all-reduce "transfer" (300-550 us) is that difference. The slowest rank's
FFN moves 107 MB (its EP experts: ~4 whole experts of 25 MB), the median's 9.5 MB (shared expert and router only): under
expert parallelism at 4 rows x 8 experts the routed experts land on a few ranks. By contrast a chain of 16 all-reduces of
32-128 KB inside one graph costs ~0.01 ms each over the 32 ranks (`tools/probe_decode_cc.py`, XLA world and 8-rank group,
and the same in-kernel through nki.collectives); with uniform compute between them they hide entirely.

**Four opt-ins, each measured** (time_decode wall p50, ms per step at 1 / 4 / 16 rows per DP group; logs td-<cfg>-<v>.log):

| | 1 | 4 | 16 |
|---|---|---|---|
| engine-v0 70ddc1b (EP experts) | 70.3 | 97.4 | 136.8 |
| KILN_DECODE_WHOLE=1 | 71.8 | 93.3 | 127.9 |
| KILN_DENSE_FP8=0 | 66.3 | 93.5 | 131.6 |
| KILN_DSA_PREFIX=mm | 69.7 | 94.0 | 136.8 |
| KILN_MOE_EP=0 (tensor-parallel experts) | 55.1 | 78.3 | 126.5 |
| **ST: all four** | **47.5** | **68.4** | **110.6** |

The step's fixed part (1 row per group) is down 32%; the per-row slope stays ~3.5 ms per row per group. EP stays the
better prefill layout (its gates), so ST is a decode-box configuration: with prefill / decode disaggregation the decode
engine loads TP experts and never runs a prefill piece. Its serving gate (with prefill graphs, check_mixed's decode-path
NLL against engine-v0 b8814ab) is in flight (q/dc2-st).

**Async MTP on the current tree** (`tools/dc_mtp.sh`: tools/amtp_g1b.sh's G1b workload, KV 1.5 fp8, 256 requests per
level, cold / warm, kiln-dc-32, logs amtp-MB / -MS / -MA): no MTP 275.0 / 338.8 out tok/s, synchronous MTP k=1 243.0 /
276.7, async 248.7 / 298.3, 96.4-96.9% of drafts accepted. MTP now loses (on f54fc18 async was +7.1% / +5.5%): the
engine turns SP decode streams off with an MTP head, and the verify graph (Q = 2) takes the XLA KDA verify scan and the
DSA mask form rather than the decode kernels, so the speculative step lost what the plain decode step gained. MTP pays
again only with verify-capable decode kernels; at 1M context the verify needs the decode-side indexer too.

**The KV handoff through the host** (`tools/probe_kv_handoff.py`, one trn1.2xlarge NeuronCore, bf16, 16 KB pages):
device -> host 2.9-3.5 GB/s from 64 MB up (11.8 GB/s at 16 MB), host -> device 9.2-12.4 GB/s; gathering a sequence's
scattered pages on the device before the read, or scattering after the write, costs ~5%. A 1M sequence under CP = 8 +
minimal is ~0.78 GB per rank of its attention group: ~0.35 s through the host per handoff against ~30 s of 1M prefill; at
8K a request is ~28 MB per rank. EFA (trn2 16 x 200 Gbit) moves 6.3 GB in ~16 ms. The transfer is not a constraint.

**W1M on trn2, a projection (not a measurement):** two tp=32 engines, DP attention 4, `KILN_DSA_CP=1` with A = 8 and
256-token pages (8 local pools = 2 KB per descriptor), `KILN_DSA_KV=minimal`: ~10.5 GB free per 24 GiB rank after
11.8 GB of weights and ~3 GB of graphs = ~13 1M sequences per attention group, ~104 per box. Step at 13 rows per group:
~111 ms (the trn2 decode-kernel curve at 8K) + ~5 ms of indexer (0.26 ms per row-layer / 8 ranks) + selection and merge
= ~120-130 ms for 52 rows per engine, **~830 out tok/s per box** before the fixed-cost changes above are applied on trn2.

**Where the ST step's fixed cost is now, and the price of a kernel call** (2026-10-05, trn1.32xlarge kiln-dc-32b; the ST
1-row call replayed as its one graph with captured inputs, logs/kiln-dc-32b/st-g1-step.txt, rep-G1.report.txt): 38.6 ms
on rank 0 (time_decode wall 48 ms): KDA mixers 34 x 0.301 = 10.2 ms, FFN blocks 45 x 0.272 = 12.2 ms, DSA mixers 11 x
0.594 = 6.5 ms, 91 all-reduces at 0.024 ms wait + 0.067 ms transfer = 8.4 ms (0.41 ms each under EP: the imbalance is
gone), post 1.0 ms. Every block moves its weights at ~130 GB/s (KDA 32 MB in ~0.25 ms, FFN 37 MB in ~0.29, DSA 76 MB in
~0.57; HBM read 3.56 GB per call against 2.89 GB the model needs).

Weight streaming is not the limit by itself (`tools/probe_gemv.py`, one trn1.2xlarge core, bf16, [T, 4096] x [4096,
4096], 8 chained products per graph): XLA's chain of F.linear streams 227 GB/s at T = 1 / 4 / 16; the same chain as ONE
NKI kernel (kernels/gemv.py kiln_gemv_chain_kernel: each weight k-tile one [128, 4096] DMA, the next product's tiles
loaded while the current one finishes) **262.6 GB/s**; as 8 kernel calls in one graph (kiln_gemv_kernel each) 99 GB/s.
So an NKI kernel call inside a graph costs ~0.21 ms over its work ((2.70 - 1.02) / 8): the compiler overlaps nothing
across the call, and the call's prologue (its inputs loaded and transposed, its first weight tiles) starts cold. A decode
layer holds 1-3 kernel calls and a collective between its blocks, so the blocks cannot prefetch each other's weights: the
fused per-layer kernels must each be one call holding all of a block's projections, so their weight tiles stream back to
back.

**The FFN block fused into one call does not pay by itself** (`kernels/moe_ffn.py` kiln_moe_ffn_pairs_v1: the router's
logits from bf16 x bf16 products in one PSUM bank, sigmoid, + bias, max8 / nc_find_index8 for the top-8, the weights
normalised and scaled, then kiln_moe_tiles_pairs_v1's per-pair loop with the shared expert as a ninth pair per token at
weight 1; exact against its emulation on the device (0 at T = 1 / 4, 1.9e-4 at 15) and in nki.simulate,
tests/test_moe_ffn.py). `tools/probe_moe_ffn.py`, one trn1.2xlarge core, GLM-5.3-Flash's rank shapes (288 FP8 experts, 128
x 128 block scales, 64 intermediate rows, clamped SwiGLU): the served form (XLA router and top-k, the per-pair kernel, the
XLA shared expert) 0.140 / 0.200 / 0.508 ms per call at T = 1 / 4 / 15, the one kernel 0.137 / 0.183 / 0.535 ms. The
expert loads (one 0.79 MB tile per pair) dominate and the XLA parts around the kernel were not the cost, so this block is
not integrated; what the step's fixed cost needs is fewer kernel boundaries per LAYER (the mixer, the FFN and their
collectives in one call), not a sub-block fused on its own.

**trn2 with the stack** (the trn2 agent ran this branch's `tools/dc_td2.sh 0 KST` on kiln-t2-cb cores 0-31, q/dc1-t2,
log logs/kiln-t2-cb/td2-KST.log; KST = the decode kernels and SP decode streams + the moe_dedupe LNC split +
KILN_DENSE_FP8=0 + KILN_DSA_PREFIX=mm + KILN_DECODE_WHOLE=1, TP experts being trn2's default): 1 / 4 / 16 / 32 / 64 rows
per group = 58.9 / 49.2 / 86.9 / 114.3 / 204.2 ms per step, 1,254 out tok/s per tp=32 engine at 256 rows, ~2,500 per
box, $1.67 / 1M out at trn2 spot (step-only, null-page) (decode only, 8K context; the decode kernels' greedy agreement on trn2 is still under the
trn2 agent's gate, so throughput only). Not a clean A/B against the decode kernels alone (110.7 / 154.9 / 264.2 at 16 / 32 /
64 rows per group): that run loaded 3 decode buckets and KST 5, and the trn2 agent measured its own decode-kernel stack
(D2K2) 12-20% slower with the 5 buckets loaded (105.8 / 126.8 / 202.7 against 87.7 / 110.7 / 180.7), so on trn2 the
stack's own gain is closer to its same-tree numbers: D2K2 + ST 84.5 / 105.2 / 182.9 ms against D2K2's 87.7 / 110.7 /
180.7 (-4% / -5% / +1%; 1,400 out tok/s per engine at 256 rows, $1.52 / 1M at half-box spot; step-only, null-page). A same-bucket A/B is the
trn2 agent's next run.

### The decode-only box at large batch: the MoE read every expert twice (2026-10-05 night, trn1.32xlarge, feat/decode-scale)

**Single changes under ST, 1 / 4 rows per group** (time_decode wall p50, `tools/dc_td7.sh`, q/dc1-t1, kiln-dc-32, logs
td7-*.log): ST 48.1 / 69.7 ms; the KDA decode kernel off (`KILN_KDA_DECODE_KERNEL=xla`) 47.6 / 70.5; the DSA decode kernel
off (`KILN_DSA_DECODE_KERNEL=xla`) 53.8 / 73.8; DP attention 1 (attention TP 32: a quarter of DP 4's mixer weights per
rank, every rank holding every sequence's KV) 44.1 ms at 4 rows per step and 72.9 at 16, against ST's 48.1 / 69.7 at the
same rows per step (-8% / +5%). The KDA kernel is neutral at small batch, the DSA kernel pays, and DP attention 1 only pays
for the smallest batches.

**ST as a decode-only box** (`tools/dc_queue12.sh` / `dc_td7.sh STL`: decode buckets 32 and 48 rows per group = 128 / 192
rows per step, fp8 KV, no prefix cache, 8K context; ~48 rows per group is what a rank's HBM holds at 8K with no prefill
graphs loaded): 152.1 ms (841.5 out tok/s) and 226.8 ms (846.5 out tok/s, $0.71 / 1M out at $2.15/h trn1 spot;
step-only, null-page: 48 rows per group do not fit with real 8K KV, below). The
plateau was the MoE: with TP experts every rank runs every row of the step, and moe_dedupe called kiln_moe_dedupe_v8 on
chunks of at most 128 tokens (it holds a call's tokens on the partitions), so 192 rows were a 128-token call and a
64-token call that each read nearly every one of the 288 experts (283 and ~240 distinct).

**kiln_moe_dedupe_v9: 129-256 tokens in one call** (`kernels/moe_dedupe.py`, opt-in `KILN_MOE_DEDUPE_MAX_TOKENS=256`; v8's
source and REV are unchanged, so no existing graph moves). The tokens sit in two tiles of 128 partitions wherever v8 holds
them there (x, the lanes' 0/1 gather matrix, the route's output, the fp32 accumulator, the LNC exchange); the plan takes up
to 2048 pairs and 512 static slots (its slot table is one PSUM row: `fits()`, 448 slots at 192 rows, 512 at 256). Exact
against `emulate()` under nki.simulate at 130 / 136 / 150 / 200 / 256 tokens, 128 / 160 / 288 experts, lanes 8 / 16, and at
grid 2 (`tests/test_moe_dedupe.py`, `tests/test_lnc_split.py`); on the device 0.0021 of the output's max at 192 rows. One
trn1.2xlarge core, GLM-5.3-Flash's rank shapes (`tools/probe_moe_kernel.py --experts 288 --scales block128 --act
silu_clamp --kernels dedupe`, ms per graph call including ~0.13 ms of launch and read-back):

| form | 128 rows | 192 rows | 256 rows |
|---|---|---|---|
| v8, 128-token calls (engine-v0) | 1.871 | 3.118 | 3.542 |
| v9, one call (as first written, lanes 8) | | 2.361 | 2.695 |
| v9, lanes 16 (groups of one slot) | | 3.034 | 3.264 |
| v9 tuned: routes of two blocks summed in PSUM, PSUM copies on the scalar engine | | 2.302 | 2.726 |
| + expert ring 4 (ring 5: 2.130 at 192) | | 2.152 | 2.595 |
| + the gather's tensor-engine half and the route matmuls deferred (`KILN_MOE_DEDUPE_RING9=4`, the default) | | **2.136** | **2.524** |
| v9 for 128-token calls too (`KILN_MOE_DEDUPE_V9=1`; 16 / 32 / 64 rows 0.640 / 1.063 / 1.440 against v8's 0.634 / 1.073 / 1.430) | **1.711** | | |

What bounds it (`tools/probe_dedupe_prof.py` + `tools/prof_engines.py` + `tools/prof_ops.py`, the 192-row graph profiled):
not the expert bytes (288 x 0.82 MB = 236 MB, 0.9 ms at 262 GB/s; DMA active 1.07-1.27 ms of the call) and not the tensor
engine (48 matmul pairs per slot at ~33 ns), but the vector engine: 1.65 ms busy of the first form's 1.9 ms, then 1.25 of
the tuned form's 1.67. Per slot it scales the down products (`pd * s_down`, 0.58 us, from PSUM), per group of two slots it
multiplies and sums the gate_up tile partials by their scales (0.96 + 0.66 us) and masks g and a by half (2 x 0.27 us),
per block it builds the lanes' 0/1 and weight matrices (~7.6 us) and adds the route into the accumulator. Every one of the
448 static slots costs that, and at uniform routing only ~314 are real (`n_slots` covers the worst routing), so a device
loop that skips the slot tail past the routing's real count (as moe_prefill's `KILN_MOE_PREFILL_SKIP` segments do) is the
next ~25% of this kernel. GpSimd does not take the SBUF-only products: `nisa.tensor_tensor(engine=gpsimd)` fails
neuronx-cc 2.27 on trn1 with `[NCC_IXCG965] Instruction engine check failed (Pool)` although nki.simulate runs it.

**End to end, one tree** (`tools/dc_td8.sh`, engine-v0 8229c3d + this branch at the first v9 form, q/dc1-t1 configs t1-STL
/ t1-STL9, kiln-dc-32, logs td8-*.log): STL 151.4 / 225.4 ms at 128 / 192 rows per step; STL9 (`KILN_MOE_DEDUPE_MAX_TOKENS=256`)
151.3 / 195.2 ms: 983 out tok/s, $0.61 / 1M out at 192 rows (step-only, null-page; decode only; a first run on the same graphs
195.7), -30 ms = 42 MoE layers x the kernel's -0.76 ms. Two keys to know: the slice of a chunk is in the graph (moe_dedupe slicing `x[s:s + n]` instead of
`x[s:s + max_tokens]` gave the 192-row v8 graphs new keys and the farm's STL graphs missed), so v8's chunks keep the old
slice; and a 1M-context row is a different capacity question, so none of these per-box numbers apply to W1M.

**ST in a mixed serving box does not pay** (`tools/dc_gate.sh`: bench/serve_sweep.py at the G64 shape, conc 64, 128
requests, q/dc2-st, logs gate-serve-*.log). engine-v0 8229c3d 167.7 out tok/s (decode call 117 ms, prefill call 475 ms);
STb (ST + `KILN_SP_GATHER=xla`, below) 120.3 out tok/s (decode call 99.7 ms, -15%; prefill call 821 ms, +73%: TP experts
are the worse prefill layout, as EP's gates said). So ST belongs to the decode engine of a disaggregated pair, with EP on
the prefill engine. The first STb attempt died in the NCCL bootstrap (`Failed to bind(127.0.0.1<40788>) ... Address already
in use`, while the CPU suite ran on the same box) and passed on a rerun.

**ST's numerics** (`tools/check_mixed.py`, 32 prompts of 700-8192 tokens, 64 greedy tokens, conc 32, logs
gate-cmp-*.log): LONG_TEXT engine-v0 against itself 32 / 32 identical; against STb 29 / 32 equal, token agreement 0.937,
decode calls |dlogprob| mean 0.0027 / p99 0.090 / max 0.30, signed **-0.00054**; wikitext-2 4 / 32 equal, agreement 0.558,
0.033 / 0.22 / 0.69, signed **-0.0021**. That is the floor of the numerically neutral changes already accepted (EPLB 30 / 32
and 3 / 32, -0.0008 / -0.0022; EP row form against EPT 9 / 64, -0.0028): the summation order moved, the decode-path NLL did
not.

**TP experts + the NKI SP gather do not compile** (engine-v0 b8814ab / 8229c3d, neuronx-cc 2.27): with `KILN_MOE_EP=0`
and the default `KILN_SP_GATHER=nki`, 4 of the 12 prefill pieces of q/dc2-st G64-ST failed `[NCC_ISCH719] topological
order violations` (keys 28585433..., 84c93cef..., 73a1331f...), as did 4 of 16 with 6-layer pieces (G64-ST6) and the same
without `KILN_DENSE_FP8=0` (G64-STa); `KILN_SP_GATHER=xla` (G64-STb) compiled all 23 graphs. The prefill agent reproduced
it (4 of 4 keys on 8229c3d) and its fallback (e40befb: the NKI world gather only for EP or expert-free models) compiles 12
of 12; until it merges, EP=0 configurations set `KILN_SP_GATHER=xla`.

**trn2, same buckets** (the trn2 agent, cores 32-63, buckets 16 / 32 / 64, logs 20261005T221314Z-t2-td-X2b): D2K2 on this
branch 87.77 / 110.52 / 212.29 ms (64 rows min 198.8; cores 0-31 were busy beside it), the same as D2K2 on its own tree
(87.7 / 110.7), so the branch is clean and the earlier 12-20% was the extra decode graphs loaded: on trn2 every loaded
decode graph costs step time, which matters for serving bucket ladders. ST at three buckets could not run (cache miss
4c031150..., the 5-bucket NEFFs do not serve it); with five buckets each, ST against D2K2 is -20 / -17 / -10% at 16 / 32 /
64 rows per group.

### Real KV on the trn1 decode box (2026-10-06, trn1.32xlarge, feat/decode-scale)

**What fits a rank** (`tools/tensor_bytes.py` on the meta device + `tools/hbm_estimate.py` over the compiled graphs,
`tools/dc_hbm.sh`, q/dc1-t1 t1-STL9's keys; one rank's own keys only, the summary line sums all four captured ranks):
the ST decode box (TP experts, decode buckets 32 / 48) with 0.5 GB of KV is tensors 12.76 GiB (experts 10.01 GB, other
weights 2.23 GB, KV 0.50 GB, layer state 0.97 GB) + code 0.22 + spill rings 0.62 + scratch 0.31 + fixed 0.13 = 14.04 GiB;
with 3.5 GB of KV (12,833 pages) 17.04 GiB. The runtime refused a configuration at 15.91 GiB of the 16 before. A 32-token
page of every DSA layer's fp8 latent, indexer rows and pool keys is ~270 KB per rank plus ~22.5 KB of token-slot state,
REPLICATED over an attention group's 8 ranks; a request's KDA state ~19.4 MB per rank. An 8448-token sequence is then
~96 MB per rank and **28 rows per group (112 per step) is what fits at 8K** (t1-STL9r2: `--max-num-seqs 112
--kv-cache-gb 2.1 --decode-buckets 28`, ~7,770 pages per group for 28 x 264 + 1); 48 per group (the STL / STL9 batches
above) do not. KV 1.95 GB (t1-STL9r) holds only ~7,222 pages: too few for 28 full sequences.

**Timing with real pages** (`tools/time_decode.py --real-kv`): every row of every group gets pages of its own (distinct ids
from each group's pool, all filled once with random values, fp8 clamped, by ModelRunner.fill_slots on every rank), a
context of --input-len + 1 + a spread over --output-len (positions, the new token's slot in its last page) and its own
state row, as ModelRunner.decode builds a running batch; it refuses a bucket whose rows do not fit the pool, and the
curve lines say `kv=real` or `kv=null-page`. Under context-parallel DSA it counts pages in cache rows per page (ps / cp).

**The tuned v9 end to end, step-only, null-page** (`tools/dc_td8.sh`, DC_TDN=10, one tree 83036ba, q/dc1-t1 t1-STL /
t1-STL9d / t1-STL9e, logs td10-*.log), ms per step at 32 / 48 rows per group: STL (v8) 152.7 / 226.9; STL9d (v9 at its
defaults for the 192-row calls) 152.5 / **183.3**; STL9e (and v9 for the 128-row calls, `KILN_MOE_DEDUPE_V9=1`) **145.2** /
184.5. v9 with its slot-tail segment costs farm compile time: a whole-decode graph took 491-510 s at 48 rows, 583-618 s at
32 (v9 everywhere) and 651-703 s at 28, against 328-345 s for the same graphs with v8's 128-token calls (c7i.48xlarge,
one compile per graph).

**The slot tail under real routing** (`tools/dedupe_slot_stats.py` on logs/kiln-mimo-trn1/ep_routing.pt, every MoE layer,
rows drawn one token each from random positions of the saved wikitext / random-token sequences): at 128 / 192 / 256 rows
the routing fills 257 / 320 / 384 slots on average (p90 279 / 332 / 393, max 292 / 343 / 403) of the 384 / 448 / 512
static ones (67-79%); random tokens 232 / 300 / 367. Uniform routing (the probes') sits between them (~314 at 192). So
v9's default head is 75% of the static slots and the rest is one device-loop segment of 6 blocks, which real routing
rarely reaches; one trn1 core, uniform routing, 192 / 256 rows: no segments 2.074 / 2.424 ms, segments of 2 blocks after
the always-real head 1.864 / 2.256 (ring 4) and 1.846 / 2.215 (ring 3), the defaults (head 75%, segment 6, ring 3) 1.835 /
2.229 (head 75% at ring 4: 1.966 / 2.188). With segments the kernel spilled SBUF (67 spill reloads in the profile against 3
without; ring 3 and bf16 lane matrices, exact as 0 / 1 and weight x 0 / 1, are what bring it back), and each run segment
pays a pipeline restart of ~25 us plus the loop barrier.

**The PD decode role's admission** (engine/scheduler.py on 8229c3d): admission always leaves at least one token to compute
(`limit = (num_tokens - 1) // page_size`) and a recurrent match stops at a page-aligned KDA checkpoint, so a request whose
KV arrives through the radix / host-tier restore path would still run a prefill chunk (the 1024-token prefill graph per
admission) on the decode box. feat/disagg admits a handed-off request straight to RUNNING instead (num_computed = prompt
length, the first token from the prefill box, its pages, pool keys and KDA state row written by eager runner copies as
HostTier.kv_load / state_load), so the decode role needs the decode graphs only (capture --skip-prefill).


**The first real-KV decode box** (t1-STL9r2 on the trn1 tree 83036ba, `tools/dc_td8.sh STL9r2` DC_TDN=11, log td11-STL9r2-1.log):
28 rows per group, real 8K KV through 7,700 pages per group (29,172 pages over 4 groups, filled in 20 s; the fill loads no
graph: ASSERT_CACHE_HIT held), contexts 8193-8448: **166.9 ms per step, 671 out tok/s, $0.89 / 1M out** (decode-only, 8K,
real KV, trn1 spot).

**Context-parallel DSA on the 8K decode box** (`KILN_DSA_CP=1`, feat/long-context's attention_cp: each rank of an attention
group of 8 holds 1 / 8 of every sequence's latent, indexer rows and pool keys; 256-token pages so a rank's pools of a page
are one 2 KB run; a scratch merge of this branch and feat/long-context 2c99613, `tools/dc_queue17.sh` / `dc_tdcp.sh`, one
tree, kiln-dc-32, q/dc1-t1 configs cp-R*, logs tdcp-*.log). Tensors per rank (`tools/dc_cp_size.sh`): 12.72 / 13.14 / 13.57
/ 13.99 GiB at 48 / 64 / 80 / 96 rows per group with KV for rows x 33 pages + 1, so ~13.6-14.9 GiB with graphs. Decode-only,
8K, real KV, trn1 spot:

| config | rows / group | rows / step | step ms | out tok/s | $ / 1M out |
|---|---|---|---|---|---|
| no CP, KV replicated over the group (28 is what fits) | 28 | 112 | 167.3 | 669 | 0.89 |
| CP | 48 | 192 | 195.5 | 982 | 0.61 |
| CP | 64 | 256 | **237.3** | **1,079** | **0.55** |
| CP | 80 | 320 | 317.2 | 1,009 | 0.59 |
| CP | 96 | 384 | the graphs fail neuronx-cc 2.27: `[NCC_INIC902] NeuronInstComb error ... APIndex.py:205` | | |

80 rows per group lose to 64 because a 320-row MoE call is v9's 256 plus a 64-row call, each reading every expert again;
v9 holds at most 256 tokens. On the null page the same graphs time 194.7 ms (CP, 48) and 165.4 ms (no CP, 28): distinct
pages cost ~1%. CP's own cost, both on the null page: 194.7 ms against 183.3 ms for the non-CP graph at 48 rows (which does
not fit with real KV), **+11.4 ms (+6%) per step**: the long path's selection, the group merge of the ranks' lists, the
log-sum-exp combine and the head reduce-scatter, net of each rank reading 1 / 8 of the attention bytes. Not split by op: a
replay of a real-KV decode call loads all 32 ranks' inputs (446 GB here, 88% of the host's memory) and hung in its
collective bootstrap (`Timeout waiting for RX`). Two traps on the way: the long path needs at least `keep` (512) local
pools per rank, so an 8K context (33 pages x 8 = 264) failed to trace (`Attempting to broadcast a dimension of length
264 ... [48, 512]`) until the page bucket was 64 (feat/long-context 77d716f takes the short case since); and the CP graphs
differ per DP group but not per rank within one (ranks 1 and 9 have ranks 0 and 8's keys), so captures of ranks 0, 8, 16,
24 cover them. Farm compile per graph: 476-504 s at 48 rows, 346-367 at 64.

**fp8 pool keys in the decode indexer** (`kernels/dsa_index.py` kiln_dsa_index_fp8_kernel, REV8; for the minimal KV layout,
whose pool keys sit in V in the KV dtype): each page row is gathered as stored (1 KB at 8 pools of 128 fp8) and goes through
the transposes as the fp8 stationary operand against the bf16 identity, exact; scores() picks it for an fp8 cache and the
bf16 kernel's source and REV are unchanged. One trn1.2xlarge core, 262,144 pools (`tools/probe_dsa_index.py --fp8`): B=1
0.623 ms (bf16 0.705), B=4 2.375 ms (bf16 2.703), relative error 2e-7. trn1's software DGE is per descriptor, so halving
the bytes buys 12% there; tests/test_dsa_index.py checks both kernels in nki.simulate.

**Step time does not follow the KV pool's size.** The 28-row decode box on the null page took 165.4 ms with a 2.1 GB pool
and 165.0 ms with 0.5 GB (cp-R28s, the same graph shape otherwise; the compiler aliases the 2.6 GiB of caches in place,
`alias size (KiB): 2724780` in its log). What makes 28 rows per group slower than 32 (145.2 ms, STL9e) is still open:
the configurations differ in bucket set (28 alone against 32 + 48) and max-num-seqs (state rows), not in pool size.

**kiln_moe_dedupe_v10: up to 512 tokens in one call** (`KILN_MOE_DEDUPE_MAX_TOKENS=512`; v8's and v9's sources and REVs
unchanged). At 4 token tiles v9's layout does not fit SBUF, so v10 keeps x in HBM and gathers each block's lanes' rows by
one indirect DMA (row tok(l) onto partition l, then 32 transposes into xg), keeps every pair's lane pair-major ([p, j],
pair 128 j + p) instead of on every partition, and builds a block's route matrix R [lanes, T] and its lanes' tokens by
matmuls of the pairs' one-hot onto the block's lanes against each pair tile's 16-token window (pair n = 8 t + k); the slot
table is PSUM rows of 512 (up to 1024 slots). Exact in nki.simulate and on the device (relative error against fp32
0.004-0.009, v9's class). One trn1.2xlarge core, 288 experts, uniform routing, ms per MoE call:

| T | v9, 256-token calls | v10, one call |
|---|---|---|
| 192 | 1.840 | 1.924 |
| 256 | 2.210 | 2.161 |
| 320 | 3.388 (256 + 64) | **2.909** |
| 384 | 3.514 (256 + 128) | **3.098** |
| 512 | 4.148 (256 + 256) | 4.043 |

Less than the expert bytes alone predict: the route's token tiles (one route matmul and one accumulator add per tile per
512 columns) and the plan over T K pairs grow with T.

**A trn1 hazard nki.simulate does not show: several accumulation chains into regions of ONE PSUM tensor lose all but the
last chain's first partial.** Minimal case (`tools/probe_psum_interleave.py`, trn1.2xlarge, neuronx-cc 2.27): column j of
a [128, 4] fp32 PSUM tensor accumulates 3 bf16 matmuls (`accumulate=False`, then `True`, `True`); with the four chains one
after another (or interleaved, or chains of 1 to 16 columns each) columns 0-2 come out as the sum of their LAST TWO
partials (partition 0: 373 = 185 + 188 where 545 = 172 + 185 + 188 is right) and only the last chain is right; with each
chain in a PSUM tensor of its own every column is exact; under nki.simulate every form is exact. A tensor written region by
region with `accumulate=False` only (v9's gate_up tiles, v10's R windows) is fine, as is one chain per tensor
(kernels/dsa_slots.py's PO across a row's blocks, the long-context agent's check). v10's first form hit it twice (each pair
tile's lane accumulated over the expert tiles into column j of one [128, NJ] tensor), which gave lanes of 0, tokens
past T and an `Out of bounds access` (nrta status 1006) on the x gather; summing the expert tiles on the vector engine
before ONE transpose per pair tile made the device match plan() exactly (`tools/probe_v10_plan.py`: lanes, R and tokens
|d| 0). Two other device-only refusals on the way: the vector engine reads at most one PSUM operand per instruction
(`'src0' and 'src1' cannot both read from PSUM`), and an fp32 operand on a matmul's stationary side is not kept at fp32
(v10 sends the lanes as two bf16-exact parts, 64 hi + lo).

**CP's merge at 96 rows per group** failed neuronx-cc 2.27 with `[NCC_INIC902] NeuronInstComb error ... APIndex.py:205` on
an `and` of its tie search ([96, 4096]; 289 `and`s in the graph), while 80 rows compiled; `dsa_long.cp_merge` now merges a
batch above `KILN_DSA_CP_MERGE_ROWS` (64) in pieces of it (each row is independent; at 64 rows and below the graph is the
same).

**Rebased onto engine-v0 f997d35** (feat/long-context merged; branch feat/decode-scale-f997): the CP decode graphs' keys
moved (cp-R64 851bf972 / b9673eb6 / 8a8019e7 / 95fde958 on the scratch merge, 40302a99 / 7ed3b76c / 99f25ec2 / db81995d
rebased, the same env and argv): engine-v0's long-context commits after 2c99613 change the CP decode graph.

**trn2, real KV** (the trn2 agent, feat/trn2-fast-ds2 ef289f8, one tp=32 engine, DP attention 4, 8K + spread, KV per set,
time_decode --real-kv; logs s3 logs/kiln-t2-cb/20261006T*-t2-tdr-*): ms per step at 16 / 32 / 64 / 96 rows per group, D2K2
96.82 / 111.55 / 179.47 / 245.74; + ST 81.85 / 107.52 / 172.64 / 245.96; + ST + v9 with the moe_dedupe LNC split on
**78.52 / 104.95 / 147.23 / 217.50** (815 / 1,220 / 1,739 / 1,766 out tok/s per engine, $2.61 / 1.75 / 1.23 / 1.21 per 1M at
half-box spot); v9 with the split off and its segments on 96.90 / 140.45 / 182.01 / 280.08 (on trn2 keep the split). 96 rows
per group is the 8K ceiling there (tensors 22.4 GB at 384 sequences; 512 do not fit 24 GiB).

**The CP decode box with v10, on feat/decode-scale-f997** (engine-v0 f997d35 + this branch, one tree, kiln-dc-32,
q/dc1-t1 configs cp-R64-rb / cp-R80-v10 / cp-R96-v10 from `tools/dc_queue18.sh`, `tools/dc_tdcp.sh`, logs tdcp-*.log;
decode-only, 8K, real KV, trn1 spot): CP-64 (v9) 238.2 ms, 1,075 out tok/s, $0.556 / 1M (237.3 on the scratch merge);
CP-80 with v10 295.4 ms, 1,083 out tok/s, $0.551 (317.2 with v9's 256 + 64 calls: -21.8 ms, the kernel's -0.48 ms x 42
layers); **CP-96 with v10 and the merge in 64-row pieces 335.7 ms, 1,144 out tok/s, $0.522 / 1M**, at ~14.9 GiB per rank.
Farm compile per decode graph: 312-331 s (CP-64), 382-396 s (CP-80), 437-465 s (CP-96).

**Where the CP-64 decode step goes** (cp-R64p: CP-64 as 6 piecewise graphs, 239.5 ms live against 238.2 for the whole-decode
graph; `tools/dc_cap3.sh`: one warm real-KV decode call captured, each piece replayed on 32 cores, rank 0 profiled; log
s3 logs/kiln-dc-32/rep2-R64p.report.txt). Replayed 250.1 ms; HBM read 20.6 GB per call against an 18.5 GB floor (MBU 16.8%),
and 0.98 / 1.58 GB of spill save / reload. By the collective that ends each segment: the MoE FFN blocks (ending in the 2 MiB
world reduce-scatter) 83.2 ms, 1.85 ms per layer; DSA under CP 68.5 ms, **6.2 ms per DSA layer** (the selection and the
two 1 MiB list gathers 29.4, the merge, slots and partial attention up to the 0.12 MiB lse gather 38.3, the 8 MiB head
reduce-scatter 0.8); the token mixers (ending in the 0.5 MiB group reduce-scatter) 35.5 ms, 0.79 per layer; collectives'
transfers and waits ~52 ms, of which ~20 ms is a per-layer 0.5 MiB all-reduce whose trigger waits 0.37 ms for the slowest
rank (with TP experts every rank routes alike, so not the MoE's slot plan). KDA state read and write is at least 0.25 ms
of a 0.79 ms mixer segment, ~8.5 ms per step, so bf16 state would buy under 2% and is not built. Page bucket 33 instead
of 64 (each rank's 264 local pools all candidates rather than a top-512 over 512 padded ones, cp-R64b33) is 239.0 ms
against 238.2: the padding is not the DSA cost.

**One DSA layer under CP, op by op** (the same cp-R64p replay, rank 0, piece 004, which holds two DSA layers that agree
within 5%; neuron-explorer's JSON view of the NTFF, instructions bucketed by time window and the HLO ops named by their
graph.hlo shapes, `tools/prof_layer_ops.py` cc / window / timeline / named / hlo; `active` = any compute engine busy). A DSA layer under CP is ~7.05 ms, a KDA layer ~3.4 ms (mixer
1.02, MoE 1.90, the rest reduce-scatters and norms):

| window | span ms | active ms | what |
|---|---|---|---|
| projections, cache stores | 0.63 | 0.33 | q / indexer projections, cp_local_slots scatters |
| local selection | 2.15 | 0.57 | scores (pool-key rows [64, 512, 128] at 256 B each 0.40, dot 0.44), then select_device + compact over Pl = keep = 512: compact()'s [64, 512, 32] gather (slice [1, 1, 1], one descriptor per element) 0.73 and its cumsum, sc.gather 0.41, pool_row's table gather 0.42 |
| 3 group all-reduces | 0.31 | 0.00 | the list values, the context pools, q_all |
| cp_merge | 0.92 | 0.80 | vector-bound: the 21-step tie search over [64, 4096] |
| partial attention | 2.58 | 1.91 | kernels/dsa_slots.py, 16 calls of 4 rows x 640 slots |
| combine | 0.46 | 0.12 | lse all-reduce, 8 MiB head reduce-scatter, W_UV, o_proj |

So ~1.6 ms of the selection window is every compute engine idle on XLA element gathers, and at 8K (any context up to keep A
kpool = 16,384 tokens at A = 8) a rank's local pools all fit keep, so that selection is the identity: the visible local
pools are the prefix m < nloc. Each rank also attends 640 slots per row of which only its share of the global top-512
(~64 on average) plus the tail are live: the long-context agent's decode slot classes. **Rank skew is not a lever**: the
every-rank replay (cp-R64pa, all 32 ranks profiled, each viewed with its own group's NEFF: `tools/prof_skew.py
--group-keys`, logs s3 logs/kiln-dc-32/skew2-R64pa-*.txt) gives the slowest rank's compute within 1.1-1.2% of the median
in every piece (median / max ms: 001 51.32 / 51.91, 002 54.14 / 54.73, 003 54.57 / 55.16, 004 39.86 / 40.32; 199.9 / 202.1
over the four layer pieces) and the per-layer 0.5 MiB
all-reduce a trigger wait of ~0.5 us; rank 0's waits are the piece's first collective (the replay's start, 4.4-4.6 ms) and
the 16 KiB all-reduce behind each 2 MiB one (56-301 us). The "~20 ms waiting for the slowest rank" above came from a
rank-0-only replay and does not hold.

**KILN_DSA_CP_ALL_LOCAL / KILN_DSA_CP_PAGE_KEYS** (models/mla.py attention_cp's decode branch, opt-in, 0079633): when Pl <=
keep, `_cp_local_all` writes select_device + compact's list directly (entry k = k below min(nloc, Pl), 0 after; the values,
NEG_INF past it; the slot rows prow[:, k], which is pool_row(k)); PAGE_KEYS reads the local pool keys as page rows (the
cache viewed as [pages, ppl Di] indexed by the block table: 2 KiB rows instead of 256 B). Both exact (tests/test_dsa_long.py:
element for element against the selection path at keep A, one and A context pools below and fewer, Pl = keep, keep - ppl,
keep / 2, ties and zeros; the tp-2 CP engine's tokens and logprobs equal without them, the direct list taken at bucket
Pl 6 / 9 / 14 and declined at 20). Neither changes a cache, a state row or a prefill graph. **CP-64, one tree** (engine-v0
2143e0b merged, `tools/dc_cpA.sh`, kiln-dc-32, the variants in turn and twice, logs tdA-*.log; decode-only, 8K, real KV,
trn1 spot):

| CP-64 | step ms (pass 1 / 2) | out tok/s | $ / 1M out |
|---|---|---|---|
| base | 236.9 / 237.8 | 1,078 | 0.554 |
| + ALL_LOCAL | 220.0 / 220.2 | 1,163 | 0.513 |
| + ALL_LOCAL + PAGE_KEYS | 215.9 / 215.3 | 1,187 | 0.503 |

-17.3 ms for the direct list (11 DSA layers, ~1.6 ms each, the profile's idle gather time), -4.4 ms more for the page rows.
The 12 graphs compiled on the box in 527 s. With engine-v0 3df0b63 (feat/disagg) merged as well (e34272d) the base CP-64
capture's 4 keys are the same, already compiled: the disaggregation work leaves these decode graphs alone.

**cp_merge's row pieces are for decode batches only** (2e6e2b2): 78469a9 split any merge above 64 rows, which would also
have cut a prefill chunk's merge (thousands of rows, one piece on engine-v0) into 64-row pieces; attention_cp now passes
`CP_MERGE_ROWS` for a decode batch's [B, P] table only. Checked by capture against engine-v0 3df0b63
(`tools/dc_g64keys.sh`, ranks 0 / 8 / 16 / 24, prefill graphs included, nothing compiled): `KILN_DSA_CP=1` alone at the
CP-64 shapes 10 / 10 keys per rank equal (28 / 28 distinct), and the CP-64 / CP-96 decode configs with ALL_LOCAL +
PAGE_KEYS re-captured on 2e6e2b2 already compiled (4 / 4 each). **The colocated G64 defaults** (no flag set, hv_ab.sh's
G64 env and argv, prefill and decode graphs): 15 / 15 keys per rank equal to engine-v0 3df0b63's (33 / 33 distinct), on
e4af8a2 and on 2e6e2b2. After engine-v0 401b3a3 (the long-context agent's #2, which carries this branch's three CP
commits plus its bounded tie search, `KILN_DSA_CP_MERGE_BOUND`, opt-in) was merged in (ad37e35): the G64 defaults 15 / 15
per rank and `KILN_DSA_CP=1` alone 10 / 10 per rank equal 401b3a3's, and the CP-96 / CP-64 decode configs with
ALL_LOCAL + PAGE_KEYS still already compiled (4 / 4 each): the measured graphs are the merged tree's.

**CP-96 (v10), one tree** (e34272d: engine-v0 3df0b63 merged; `tools/dc_cpA.sh R96 R96A R96Ak R64`, the same box and
method; decode-only, 8K, real KV, trn1 spot; 12 graphs compiled on the box in 714 s; the CP-64 base on this tree
237.6 ms, as on the 2143e0b one):

| CP-96 | step ms (pass 1 / 2) | out tok/s | $ / 1M out |
|---|---|---|---|
| base | 336.6 / 336.6 | 1,141 | 0.523 |
| + ALL_LOCAL | 306.9 / 306.1 | 1,253 | 0.477 |
| **+ ALL_LOCAL + PAGE_KEYS** | **302.4 / 302.8** | **1,269** | **0.470** |

**$0.470 per 1M output tokens, decode-only, 8K context, real KV, trn1.32xlarge spot at $2.15/h**: the decode box under the
$0.5 target (1,269 out tok/s against the 1,194 it needs). Keys (q/dc1-t1 configs cpA-R96Ak): 13e612ef (rank 0) / 4e19b71c
(8) / 6c7f0d61 (16) / b216626e (24), decode graphs only, the disaggregation agent's decode-role set.

**CPU suite on the final tree** (72f890b; the later commits are docs only; kiln-dc-32, one pytest process,
`KILN_TEST_MODEL=Qwen/Qwen3-0.6B`, logs s3 logs/kiln-dc-32/ci-final.log, ci-final-tf518.log): the venv's transformers 770
passed, 67 skipped, 0 failed in 25:04; the whole suite with transformers 5.18 first on PYTHONPATH 902 passed, 15 skipped,
7 failed in 1:06:13, the 7 all tests/test_inkling.py (`KeyError: 'model.llm.embed_...'`, its checkpoint names under 5.18),
which fail the same 7 on engine-v0 401b3a3 under 5.18 (inkling-tf518-ev401.log) and pass under the venv's transformers.

## Prefill / decode disaggregation on the device (2026-10-05/06, SDK 2.32, trn1.32xlarge spot)

Written up at release time from the disaggregation agent's logs, which were taken while the work was in
progress and were never written into these notes. GLM-5.3-Flash@eb9eb208 real weights, 8192 in / 256 out,
trn1.32xlarge spot $2.15/h per box, every graph from the compile farm under
`NEURON_LIBTORCH_ASSERT_CACHE_HIT=1`. The price-performance table is docs/price-performance.md
"Prefill / decode disaggregation (G1, trn1)". Sources, each archive holding each engine's log and the `.cmd`
that started it: s3://<your-bucket>/logs/kiln-pd-d1/pd-logs{,2,4}-d1.tgz,
logs/kiln-pd-p{1,2,3,4}/pd-logs{,2,3}-p<n>.tgz, logs/kiln-pd-l{1,2}/pd-logs4-l<n>.tgz,
logs/kiln-pd-ci/ci-logs-2026-10-06.tgz.

**NONE of it ran on engine-v0 e3ff411**, so every number is labelled with the tree it ran on (release gate 4):
the prefill boxes on feat/disagg over engine-v0 f997d35 and then feat/disagg-stream, the decode box on
scratch/pd-dec-f997 686d791 (cp-R96-v10), the latency prefill boxes on feat/disagg-stream 8ab4763 plus the
`dsa_fused` MIN_HEADS patch that became engine-v0 4e5f226.

**What the parts are.** `kiln/engine/disagg.py` gives an engine a ROLE: a prefill engine runs the prompt and
hands the request's pages, pool keys and KDA state row to a decode engine over its own frames (a TCP
connection per rank), and the decode engine admits the handed-off request straight to RUNNING
(num_computed = prompt length, the first token from the prefill box) so it needs no prefill graph at all
(`capture --skip-prefill`). `kiln/server/pd_router.py` is the front door: threshold routing (`--threshold`,
4096 input tokens by default, the same knob SageMaker HyperPod calls routingThreshold), decode-credit
backpressure, an opt-in per-engine prefill queue-depth cap (`--prefill-depth`) and latency prefill engines
(`--latency-prefill-urls --latency-depth 1`, preferred while one has a free slot). Its SLO metrics are
Prometheus series named `kiln:pd_router_waiting`, `_prefill_waiting`, `_inflight`, `_prefill_depth`,
`_decode_credits`, `_queue_seconds`, `_prefill_queue_seconds`, `_e2e_ttft_seconds`, `_prefill_seconds`,
`_prefill_call_seconds`, plus `kiln:pd_transfer_seconds`.

**The engines, verbatim from the `.cmd` files.** Prefill (kiln-pd-p1..p4): `pd_serve.py --pd-role prefill
--port 8100 -- --model zai-org/GLM-5.3-Flash --tp 32 --dp-attention 4 --piecewise --overlap --prefill-tokens
8192 --prefill-buckets 2048 --max-num-seqs 64 --decode-buckets 16 --kv-cache-gb 1.2 --kv-cache-dtype fp8
--state-checkpoints 4 --eplb-rebalance --page-buckets 264` with `KILN_PIECEWISE_PREFILL_MOE_GROUP=45
KILN_EP_REDUNDANT=1 KILN_EPLB_INIT=<placement> KILN_EPLB_INTERVAL=200 KILN_EPLB_MAX_REBALANCES=1` and farm
queue q/pf-p8-7ce6a12: the one-piece 8192 prefill with EPLB, i.e. the same configuration as the colocated
191.3 out tok/s / $3.12 row. Decode (kiln-pd-d1): `--pd-role decode --pd-listen 0.0.0.0:7400 --pd-buffer-gb
48 -- ... --tp 32 --dp-attention 4 --prefill-tokens 4096 --prefill-buckets 1024 --kv-cache-dtype fp8
--no-prefix-caching --page-size 256 --page-buckets 64 --max-num-seqs 384 --kv-cache-gb 0.92
--decode-buckets 96` with `KILN_DSA_CP=1 KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DECODE_WHOLE=1
KILN_MOE_DEDUPE_V9=1 KILN_MOE_DEDUPE_MAX_TOKENS=512` and farm queue q/dc1-t1; from pd-logs2 on also
`KILN_DSA_CP_ALL_LOCAL=1 KILN_DSA_CP_PAGE_KEYS=1`. Latency prefill (kiln-pd-l1, l2): the prefill env at
`--dp-attention 1 --prefill-tokens 4096 --prefill-buckets 4096 --max-num-seqs 16`, farm queue
q/pf-tp1-pdpre.

**The closed-loop levels** (`bench/pd_sweep.py`, which prints each level twice: the whole level, and the
steady rate over its middle half, because a closed-loop level's start and tail understate it). Per-engine
`busy_frac` from the router's step metrics in the same log:

| log | prefill : decode | $/h | conc | steady out tok/s | steady all-in | whole level out tok/s | TTFT p50 / p90 | ITL p50 / p90 | decode busy | prefill busy | errors |
|---|---|---|---|---:|---:|---:|---|---|---|---|---|
| g1-pd-f | 3 : 1 | 8.60 | 320 | **930.9** (207.4 s) | **$2.566** | 789.9 | 17553 / 58158 ms | **277.0 / 284.3 ms** | 0.898 | 0.853-0.854 | 0 |
| g1-pd-l pass 1 | 4 : 1, ALL_LOCAL + PAGE_KEYS | 10.75 | 440 | **1,139.5** (192.1 s) | **$2.621** | 879.4 | 8621 / 70059 ms | 353.5 / 370.7 ms | 0.962 | 0.741-0.751 | 0 |
| g1-pd-l pass 2 | the same | 10.75 | 440 | 1,094.5 (198.8 s) | $2.728 | 849.3 | 10943 / 69831 ms | 363.4 / 386.9 ms | 0.964 | 0.712-0.727 | 1 |
| g1-pd-k | 4 : 1, neither flag | 10.75 | 440 | 1,002.2 (157.2 s) | $2.980 | 716.5 | 11229 / 79096 ms | 406.7 / 426.7 ms | 0.890 | 0.614-0.639 | 0 |
| g1-pd-j | 4 : 1, neither flag | 10.75 | 440 | 967.5 (270.9 s) | $3.086 | 829.1 | 8432 / 60910 ms | 414.2 / 464.8 ms | | | 5 |
| g1-pd-i | 4 : 1, neither flag | 10.75 | 440 | 967.4 / 938.3 | $3.087 / $3.182 | 780.9 / 768.1 | 9071 / 69884 ms | 417.3 / 455.1 ms | | | 0 |
| g1-pd-e | earlier arm | 8.60 | 256 | | | 665.7 / 707.9 | 11158 / 54187 ms | 248.7 / 322.4 ms | | | 0 |
| g1-pd-c | earlier arm | 8.60 | 112-168 | | | 466.9-510.7 | 8089 / 26107 ms up | 178.2-183.4 ms | | | 0 |

The two CP flags on the decode box are worth 1,002.2 -> 1,139.5 out tok/s steady (+13.7%) and
$0.5959 -> $0.5241 per 1M output tokens on the decode side, which is the same direction and about the same
size as the decode-only measurement in "The CP decode box with v10" above (1,141 -> 1,269 out tok/s).

**What could NOT be reproduced from the logs, and is therefore not cited anywhere.**
- **g1-pd-g** reports `$2.536` all-in at `$6.45/h`, but its own step metrics list FOUR busy engines (three
  prefill at busy_frac 0.765-0.784 and the decode box), so the run was 4 boxes priced as 3. At 4 x $2.15/h its
  706.6 out tok/s steady is $3.381 all-in, worse than the 3:1 row. The row is excluded.
- There is no colocated STEADY middle-half figure at concurrency 64, so the $2.57 steady figure has no
  like-for-like colocated partner; the honest comparison is whole level against whole level, $3.024 against
  $3.12 (3.1% below). Said plainly in docs/price-performance.md.
- No ITL of 311 ms exists in any disaggregation log (`ITL p50 31x` matches nothing; the values measured are
  243.0, 248.7, 277.0, 287.7, 299.2, 327.1, 353.5, 363.4, 406.7, 414.2, 417.3 and 267-271 ms on the latency
  arm). An earlier draft of the release notes quoted 311 ms as "the colocated best"; it is not measured.
- The design estimate in "Where a GLM-5.3-Flash decode step goes" predicted 5.4 prefill boxes per decode box
  and $4.13 per 1M output tokens all-in. Measured: 3 or 4 prefill boxes per decode box and $2.57-$3.02
  (steady / whole level) at 3:1. The estimate was pessimistic on both counts, and its own caveat (that the
  gain needs ~190 sequences in flight per decode box) is what the conc 320 and 440 levels supply.

**The TTFT-SLO arm (latency prefill boxes).** Driver slo4.log on kiln-pd-d1, 2 latency prefill + 1 decode
= 3 boxes at $6.45/h, the router started with `--latency-prefill-urls` and `--latency-depth 1` (its /debug
reports `"kind":"latency","depth":1` for both). Post-fix tree only: before engine-v0 4e5f226 the fused DSA
kernel read an overwritten latent ring block at fewer than 3 MLA heads, which is exactly what DP attention 1
at tp 32 gives (2 heads per rank), so the earlier latency-box runs (slo2) are timing only.

| arm | TTFT p50 / p90 | ITL p50 | req/s served (steady) | log |
|---|---|---|---|---|
| idle, conc 1 | **1595 / 1607 ms** | 267.3 ms | 0.014 | slo4-idle-lat |
| open loop, rate 0.4 | 1554 / 2126 ms | 269.2 ms | 0.234 (0.292) | slo4-open-lat |
| open loop, rate 0.6 | 1530 / 1775 ms | 269.6 ms | 0.357 (0.476) | slo4-open-lat |
| open loop, rate 0.8 | 1559 / 2078 ms | 270.1 ms | 0.449 (0.589) | slo4-open-lat |
| open loop, rate 1.0 | 1596 / 2268 ms | 270.5 ms | 0.579 (0.743) | slo4-open-lat |
| threshold routing only, idle | 4334 / 4338 ms | 267.3 ms | 0.014 | slo2-idle-thr |
| colocated one box, idle | 4013 / 4175 ms | 87.6 ms | 0.038 | colo-idle, kiln-pd-p1 |

So the latency arm takes the idle 8K TTFT from 4.01 s to 1.60 s and holds it to 1.60 s at an arrival rate of
1 req/s, while its ITL stays at the decode box's 267-270 ms (96 rows per DP group) against the colocated
87.6 ms. **No arm reaches a p90 TTFT of 1 s.** The router's own histograms on the idle arm: 5 requests,
`_e2e_ttft_seconds` 7.589 s total against `_prefill_seconds` 7.057 and `_prefill_call_seconds` 6.121, so the
prefill call is 81% of the TTFT and the router's own queue is 11 microseconds of it.

Device gate for the arm (`tools/check_mixed.py` through the latency prefill engine against the colocated
reference on the fixed tree; chk-colo-fix on kiln-pd-l1, result lines in slo4.log): **equal 15 / 16**, token
agreement 0.9639 (987 / 1024), teacher-forced |dlogprob| over decode calls n = 972 mean 0.00228, p99 0.0582,
max 0.2980, signed mean -0.00020; over prefill chunks n = 16 mean 0.01168, max 0.1278, signed -0.00430. On
the pre-fix tree the same check was 10 / 16 with a first-token mean |dlogprob| of 0.58 (the MIN_HEADS commit
message), which is what the padding fixed.

**An idle decode engine must release what it holds** (engine-v0 9e693cf). A disaggregated decode engine under
overlap holds freed pages and state rows for one step (`pd_hold`) so a handoff copy need not wait for the
device. Pages the radix prefix cache evicted to make room for an incoming handoff went into that hold, and an
engine that had gone idle has no step in flight to read back, so the hold was never drained: on trn2
(T2-BEST1, prefix caching on) a closed-loop level was followed by 30 queued handoffs with KV 99% used and
nothing running, the engine thread spinning in `_admit_prefilled`. An idle decode engine now drains its own
hold before it gates admission; `tests/test_disagg.py::test_idle_decode_engine_releases_what_it_holds` is a
CPU repro (36 pages, the first batch leaves 7 free and 28 cached, the next handoff needs 8).
