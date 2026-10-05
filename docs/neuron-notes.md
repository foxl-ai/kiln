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
the vector engine bounds a pass.

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

**Defaults.** All three are opt-in, default off on every platform: `KILN_KDA_DECODE_KERNEL` and `KILN_DSA_DECODE_KERNEL`
default `xla`, `KILN_DECODE_SP` `0`. Measured only on trn1 (trn1.32xlarge, GLM-5.3-Flash, tp=32, DP attention 4, EP on):
the G64 config with all three (conc 64, 16 rows per group), F0 with the two kernels (conc 32, 8 rows per group), and the
decode-only step at 32 and 64 rows per group. Not measured: G16 (4 rows per group, TP), F0 with SP decode streams, trn2,
mixed batches (`KILN_MIXED_BATCH=1`) and MTP (SP decode streams stay off with an MTP head). Turning them on changes
every decode group graph's key, so a default flip needs those configs' graphs compiled first.

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
