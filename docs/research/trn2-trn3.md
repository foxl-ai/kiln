# trn2 and trn3: availability, SDK differences, Kiln gaps, measurement plan

Reading date: **2026-10-02 PDT / 2026-10-03 UTC**. No instance was launched for this document.
Every AWS fact in section 1 comes from a read-only API call made from account <aws-account-id>
(IAM user `<iam-user>`) between 2026-10-03T02:13Z and 02:45Z; the commands are in section 9 and the
raw readings in appendix A. SDK facts cite the unpacked Neuron SDK 2.32 wheels:

- `SRC` = `<local-dir>/neuron-src`: `libtorch_neuronx_lite`
  2.11.0.1.0.1284+f49d8626 (LNL, the build in the DLAMI's vLLM 0.24 venv Kiln runs in),
  `vllm_neuron` 0.24.0.1.1.0, `nki` 0.6.0+31049202112.g85070674, `nkilib` (build Jul 15 2026).
- `RES` = `<local-dir>/research`: `aws-neuron-sdk` (docs repo,
  HEAD a6be966, 2026-09-30), `vllm-neuron` (branch release-0.24.0.1.1.0),
  `neuronx-distributed-inference` (HEAD 4bcdc54, 2026-07-28), `aws-neuron-driver`.
- GitHub PR bodies read with `gh api` on 2026-10-03.

Conventions as in `models.md` and `neuron-stack.md`: **UNCONFIRMED** = no primary source;
**ANALYSIS** = my inference. No number in this file is a Kiln measurement on trn2 or trn3; there
is none yet.

---

## 0. Headline

1. **trn2 is rentable today; trn3 is not, for this account.** `trn2.48xlarge` is offered in
   us-east-2 (3 AZs) and ap-south-2 (3 AZs); `trn2.3xlarge` (one Trainium2) in ap-south-2,
   ap-southeast-4 and sa-east-1; `trn2u.48xlarge` (UltraServer node) only in ap-south-2. **No
   `trn3*` instance type exists in the EC2 API of any reachable region**, and the account is
   "not authorized to call DescribeCapacityBlockOfferings API when requesting an EC2
   UltraServer". AWS's Trn3 page sells Trn3 only as UltraServers with no sizes or prices.
2. **Spot prices now:** trn2.48xlarge **$14.5593/h** (us-east-2c only; 30-day range
   $8.99-15.37, mean $12.97); trn2.3xlarge **$2.2687/h** (ap-southeast-4c), **$2.5690/h**
   (sa-east-1c, 30-day mean $1.53, min $0.90). Spot placement score is **1 of 10** for both
   types everywhere they exist (trn1.32xlarge scores 9 in us-east-2), so spot capacity is thin.
   No on-demand list price is published for any trn2 size in the Pricing API; a 24 h Capacity
   Block for trn2.48xlarge is offered only in ap-south-2b at **$858.26 ($35.76/h)**.
3. **Quota:** 256 vCPU of Trn spot and 256 vCPU of Trn on-demand per region. One
   trn2.48xlarge (192 vCPU) fits, and it does **not** fit next to a running trn1.32xlarge (128)
   in the same region; Kiln has one running in us-east-2 right now.
4. **What the SDK changes per generation** (section 3): logical NeuronCores (LNC=2 default on
   trn2 and trn3: 4 logical cores per chip, 64 per 16-chip instance), HBM per logical core
   16 / 24 / 36 GiB (trn1 / trn2 / trn3), FP8 E4M3 max 240 on trn1/trn2 and OCP e4m3fn 448 on
   trn3, MX (MXFP8 / MXFP4) matmul only on trn3, and an NKI Library whose MX, fast-exp and
   indirection paths are gen4 (trn3) only. Kiln's FP8 handling already branches correctly on
   `trn3`; everything else in section 6 is missing.
5. **No managed baseline exists on trn2/trn3 for any of the top target models** (MiMo-V2.6,
   GLM-5.3, Kimi K3, Qwen3.8, DeepSeek-V4.1). vllm-neuron 0.24 registers Llama, GPT-OSS, EAGLE3
   Llama, Qwen3 dense and Qwen3-VL only. The same-instance, same-SDK comparisons available are
   therefore on control models (Qwen3 dense, Llama 3.1/3.3, GPT-OSS), with unmerged community
   ports (vllm-neuron PR #40 MiMo-V2.5, NxDI PR #150 MiMo-V2.5-Pro) as the nearest references.
6. **First measurement proposed:** Kiln vs vllm-neuron 0.24 on **one trn2.3xlarge spot in
   sa-east-1c**, SDK 2.32, Qwen3-0.6B then Qwen3-8B, `bench/offline.py`'s workload. Budget 6 h:
   **$9 at the 30-day mean, $15 at today's price**, capped at 8 h ($21).

---

## 1. Availability, price and quota (measured with the AWS API)

### 1.1 Where each type is offered

`aws ec2 describe-instance-type-offerings --location-type region|availability-zone`, filter
`trn*,inf*`, all 34 enabled regions (me-south-1 timed out: "Connect timeout on endpoint URL").
`describe-instance-types --filters Name=instance-type,Values=trn2*,trn3*` in every region agrees.

| Type | Regions (AZs) | Usage classes (describe-instance-types) |
|---|---|---|
| trn2.48xlarge | us-east-2 (2a/2b/2c = use2-az1/2/3), ap-south-2 (2a/2b/2c = aps2-az1/2/3) | capacity-block, on-demand, spot |
| trn2u.48xlarge | ap-south-2 (aps2-az1/2/3) | capacity-block, on-demand, spot |
| trn2.3xlarge | ap-south-2 (3 AZs), ap-southeast-4 (4c), sa-east-1 (1a/1b/1c) | capacity-block, on-demand, spot |
| trn3 (any) | **none** | - |
| trn1.32xlarge | us-east-1, us-east-2, us-west-2, ap-south-1, ap-southeast-2, ap-southeast-4 | |
| trn1.2xlarge | us-east-1, us-east-2, us-west-2 | |
| inf2.48xlarge | 13 regions | |

us-east-1 and us-west-2 offer no trn2 size at all. The CLAUDE.md claim that us-east-2 is "the
only region with trn1, trn1n, trn2 and inf2 together" still holds.

### 1.2 Instance shapes

| Type | Chips | vCPU | Host RAM | Accelerator memory | Network / EFA | Local NVMe | Source |
|---|---|---|---|---|---|---|---|
| trn2.3xlarge | 1 Trainium2 (8 cores, CoreInfo Version 3) | 12 | 128 GiB | 96 GB | 200 Gbit, 1 EFA | 1 x 470 GB | describe-instance-types (ap-south-2); https://aws.amazon.com/ec2/instance-types/trn2/ |
| trn2.48xlarge | 16 Trainium2 | 192 | 2,048 GiB | 1.5 TB | 16 x 200 Gbit, 16 EFA | 4 x 1.92 TB (7,600 GB) | same |
| trn2u.48xlarge | 16 Trainium2 | 192 | 2,048 GiB | 1.5 TB | 16 x 200 Gbit, 16 EFA | 7,600 GB | same; product page: "Available in EC2 UltraServers: Yes" |
| Trn3 | Trn3 UltraServers only: "up to 144 Trainium3 chips", "144 GB of HBM3e and 4.9 TB/s" per chip, "FP32, BF16, MXFP8, and MXFP4" | - | - | - | - | - | https://aws.amazon.com/ec2/instance-types/trn3/ (fetched 2026-10-03; no sizes, no prices) |

**Do not size memory from the EC2 API.** `describe-instance-types` reports
`MemoryInfo.SizeInMiB = 524288` (512 GiB) per Trainium2 and
`TotalNeuronDeviceMemoryInMiB = 8388608` for trn2.48xlarge, and an empty `NeuronInfo` for
trn2u.48xlarge. The product page says 96 GB per chip and 1.5 TB per instance, the Neuron docs
say 96 GiB per chip (`neuron-stack.md` 1.2), and LNL's own table says 24 GiB per logical core x
4 = 96 GiB (section 3.1). The API value is wrong by 5.3x.

Instance-type strings for Trn3 exist only in AWS software, not in EC2:
`nrt.h` enumerates `NRT_INSTANCE_TRN3 = 14`, `NRT_INSTANCE_TRN3PDS98 = 15`
(`RES/aws-neuron-sdk/src/libnrt/include/nrt/nrt.h:66-67`); the driver maps `trn3.48xlarge`,
`trn3sn.48xlarge`, `trn3s-es.48xlarge` to `NEURON_PLATFORM_TYPE_PDS`, `trn3p.48xlarge` to
`ULTRASERVER`, `trn3e.24xlarge` to `MAX`, plus a list of "3xl" Trn3 names
(`RES/aws-neuron-driver/v4/neuron_dhal_v4.c:152-178`); vllm-neuron knows families `trn3pd` and
`trn3pds`, 16 devices each (`SRC/vllm_neuron/utils/hardware_config.py:53-96`), maps any
`trn3*` product name to `trn3pds` (L133-134), and mentions "instances without EFA (e.g. trn3
3xlarge)" (L181). Which of these become customer-launchable is **UNCONFIRMED**.

### 1.3 Spot prices

`describe-spot-price-history --product-descriptions Linux/UNIX --start-time 2026-10-03T02:19:00Z`
(the price in effect per AZ), and the same call over 2026-09-02..2026-10-03 for the trn2 types.

| Type | AZ | Now ($/h) | 30-day min / mean / max ($/h) | Records |
|---|---|---|---|---|
| trn2.48xlarge | us-east-2c | **14.5593** (set 2026-10-03T01:00Z) | 8.9914 / 12.9692 / 15.3679 | 119 |
| trn2.48xlarge | us-east-2a, 2b, all of ap-south-2 | no spot record | - | 0 |
| trn2u.48xlarge | anywhere | no spot record | - | 0 |
| trn2.3xlarge | ap-southeast-4c | **2.2687** | 2.2579 / 3.9655 / 6.9846 | 126 |
| trn2.3xlarge | sa-east-1c | **2.5690** | 0.9026 / 1.5286 / 2.5690 | 90 |
| trn2.3xlarge | sa-east-1b | 7.7264 | 7.5653 / 7.6626 / 7.7483 | 127 |
| trn1.32xlarge | us-east-2c | 2.1500 | (reference) | |
| trn1.2xlarge | us-east-2c | 0.1344 | | |
| inf2.48xlarge | us-east-2a/b | 1.2981 | | |

Spot placement score (`get-spot-placement-scores`, target capacity 1, single AZ): trn2.48xlarge
**use2-az3 = 1** (the only entry); trn2.3xlarge apse4-az3 = 1, sae1-az2 = 1, sae1-az3 = 1;
trn1.32xlarge use2-az3 = 9, usw2-az4 = 9. A score of 1 means a spot request "is not likely to
succeed" by AWS's scale; plan for retries, interruptions and a fallback.

### 1.4 On-demand and Capacity Blocks

- Pricing API (`pricing get-products --service-code AmazonEC2`, instanceType filter): trn2.48xlarge
  returns 6 products, every one `marketoption = CapacityBlock` (`USE2-BoxUsage:trn2.48xlarge`
  at $0.00); **no on-demand list price** for trn2.48xlarge, trn2u.48xlarge or trn2.3xlarge.
  References from the same API: trn1.32xlarge $21.50/h, trn1.2xlarge $1.34375/h,
  inf2.48xlarge $12.98127/h (us-east-2), all effective 2026-09-01.
- `describe-capacity-block-offerings` (read-only), 1 instance, all AZs:
  - trn2.48xlarge in **ap-south-2b**: 8 h $308.14-309.33, **24 h $858.26**
    (2026-10-03T11:30Z..10-04T11:30Z, $35.76/h), 32 h $1,166.40.
  - trn2.48xlarge in us-east-2: no offering for 24, 48, 72 or 168 h.
  - trn2.3xlarge and trn2u.48xlarge: "not supported for Capacity Blocks" in us-east-2 and ap-south-2.
  - UltraServers (`--ultraserver-type u-trn2x64` and guessed trn3 names): "This AWS Account is
    not authorized to call DescribeCapacityBlockOfferings API when requesting an EC2 UltraServer".
- ap-south-2 has **no SDK 2.32 DLAMI**: the public SSM path `/aws/service/neuron/...` is
  "not a valid namespace" there, and the newest Amazon-owned "Deep Learning AMI Neuron" image is
  from 2026-01-26. Using ap-south-2 means copying the AMI (a write action, Project=kiln tagged)
  or installing SDK 2.32 on a base image.

### 1.5 DLAMIs (SSM public parameters, us-east-2)

| Parameter | AMI | Name / SDK | Instances |
|---|---|---|---|
| `multi-framework/ubuntu-24.04` | ami-0222021b369f03219 | DLAMI Neuron 20260818, **SDK 2.32.0** | "Trn1, Trn1n, Inf2, Trn2, Trn3" |
| `pytorch-inference-vllm-0.24.0.1.1.0/ubuntu-24.04` | ami-0cb0faefcc95d3e64 | vLLM 0.24.0.1.1.0, SDK 2.32.0 | "Trn2, Trn3" |
| `pytorch-inference-vllm-0.21.0.1.0.0/ubuntu-24.04` | ami-02748a14810746103 | SDK 2.31.1 | "Trn2, Trn3" |
| `pytorch-inference-vllm-0.16/ubuntu-24.04` | ami-0f1f3007c12486bae | SDK 2.31.1 (vllm-neuron 0.5.3 on NxDI) | Trn1..Trn3 |
| `pytorch-2.9/ubuntu-24.04` | ami-0ec474571d019ce67 | SDK 2.31.1 | Trn1..Trn3 |

sa-east-1 (ami-06f5c32a48089ad7a, SDK 2.32.0) and ap-southeast-4 (ami-01f66e576e60931ed) carry
the multi-framework 2.32 DLAMI too, so a trn2.3xlarge there runs exactly Kiln's current stack.
NxD Inference is not on any SDK 2.32 DLAMI (`neuron-stack.md` 4.1); the newest DLAMI that
carries it is the SDK 2.31.1 vLLM 0.16 image.

### 1.6 Quotas (Service Quotas, `list-service-quotas --service-code ec2`)

| Quota | Code | us-east-2 | ap-south-2 | sa-east-1 | ap-southeast-4 | us-east-1 | us-west-2 |
|---|---|---|---|---|---|---|---|
| All Trn Spot Instance Requests (vCPU) | L-6B0D517C | 256 | 256 | 256 | 256 | 256 | 256 |
| Running On-Demand Trn instances (vCPU) | L-2C3B7624 | 256 | 256 | 256 | 256 | 256 | 256 |
| All Inf Spot Instance Requests | L-B5D1601B | 64 | 64 | 64 | 64 | 64 | 64 |
| Running On-Demand Inf instances | L-1945791B | 64 | 64 | 64 | 64 | 192 | 64 |

All adjustable; no Trn3 or UltraServer quota exists. At 2026-10-03T02:40Z Kiln had
`kiln-dev-trn1` (trn1.2xlarge, 8 vCPU) and `kiln-mimo-trn1` (trn1.32xlarge, 128 vCPU) running on
spot in us-east-2c (`infra/fleet.sh ls`): 136 of 256, so a trn2.48xlarge cannot start in
us-east-2 until the trn1.32xlarge is gone or the quota is raised to at least 320.

---

## 2. Hardware per generation (vendor facts)

Already cited in `neuron-stack.md` 1.2-1.3 and `models.md` 2; the numbers Kiln's plan depends on:

| | trn1 (NeuronCore-v2, NKI gen2) | trn2 (NeuronCore-v3, gen3) | trn3 (NeuronCore-v4, gen4) |
|---|---|---|---|
| Physical cores / chip | 2 | 8 | 8 |
| Logical cores / chip at default LNC | 2 (LNC=1 only) | 4 (LNC=2 default) | 4 (LNC=2, vllm-neuron default) |
| HBM / chip, bandwidth | 32 GiB, 820 GiB/s | 96 GiB, 2.9 TB/s | 144 GiB, 4.9 TB/s |
| HBM / logical core (LNL `HBM_MEMORY_GB`) | 16 GiB | 24 GiB | 36 GiB |
| Dense BF16 / FP8 per chip | 190 / 190 TFLOPS | 667 / 1,299 TFLOPS | 671 BF16, 2,517 MXFP8/MXFP4 |
| FP8 format | e4m3 (max 240), cFP8 | e4m3 (max 240), double-row FP8 = 2x BF16 | OCP e4m3fn (max 448) + MX |
| SBUF / core | 24 MiB | 28 MiB | 32 MiB |
| Scale-up | NeuronLink-v2 2D torus | NeuronLink-v3 4x4 torus per instance; UltraServer 64 chips | NeuronSwitch-v1 all-to-all; UltraServer up to 144 chips |

ANALYSIS (bandwidth per dollar, the number decode throughput follows): at today's spot prices
trn1.32xlarge buys about 4.6-6.6 TB/s of HBM bandwidth per $/h (9.8 TB/s per its product page,
or 16 x 820 GiB/s = 14.1 TB/s per the chip spec, `models.md` 2; / $2.15), trn2.48xlarge about
3.2 (46.4 TB/s / $14.56) and trn2.3xlarge about 1.3 (2.9 TB/s / $2.27). A trn2.48xlarge must
beat a trn1.32xlarge by about 1.4-2x per instance on a bandwidth-bound decode just to break
even in $/Mtok; every comparison below therefore reports both tok/s and $/Mtok.

---

## 3. What the Neuron SDK 2.32 encodes per platform

### 3.1 Platform detection and memory (LNL)

- Target string: `NEURON_PLATFORM_TARGET_OVERRIDE` if set, else NRT
  (`torch.classes.neuron.Runtime().get_instance_info()`, joined `info[0] + info[1]`)
  (`SRC/libtorch_neuronx_lite/compile/platform.py:72-88`). Families: `trn1n, trn1, trn2, trn3,
  inf2`; a target may carry a revision suffix ("e.g. "trn3-rev2"") which "flows verbatim into
  --target and the compile-cache hash" (L14-19). vllm-neuron and nkilib also see `trn3pre`
  (Trn3 A0) versus `trn3` (GA/B1) (`SRC/vllm_neuron/vllm/worker/neuron_model_runner.py:1412`;
  `SRC/nkilib/core/utils/kernel_helpers.py:633-640`, `is_trn3_b1`).
- Per-core HBM table, used when compiling without a device:
  `HBM_MEMORY_GB = {"trn1": 16, "trn1n": 16, "trn2": 24, "trn3": 36, "inf2": 16}`
  (`platform.py:5-12`). Device-free compile requires `NEURON_PLATFORM_TARGET_OVERRIDE`
  (`platform.py:50-69`; NKI accepts `trn1|inf2|gen2`, `trn2|gen3`, `trn3|gen4`,
  `SRC/nki/__init__.py:39-44`).
- neuronx-cc invocation: LNL builds `neuronx-cc compile <hlo> --framework XLA --target <target>
  --output --logfile` plus the caller's remaining args (`SRC/libtorch_neuronx_lite/compile/backend.py:384-410`).
  It never passes an LNC flag itself (grep for `lnc` in LNL finds only the NKI simulator and
  an LNC sharding custom call).

### 3.2 Logical NeuronCores

- Compiler: `--logical-nc-config <shard_degree>` / `-lnc`, values {1, 2}, "(Only available on
  trn2; Default: 2)" (`RES/aws-neuron-sdk/compiler/neuronx-cc/api-reference-guide/index.rst:153-158`);
  `--lnc 2` on trn1 is error NCC_EARG001, "On trn1, only lnc=1 is supported"
  (`compiler/error-codes/EARG001.rst`). The same page lists `--target` values inf2, trn1, trn1n,
  trn2 only (L85-93), although LNL and vllm-neuron pass `trn3` (doc lag).
- Runtime: `NEURON_LOGICAL_NC_CONFIG`; "AWS Neuron currently doesn't support setting the compiler
  flag to a different LNC configuration than the Neuron Runtime environment variable". LNC=2 is
  the trn2 default and "a Trn2.48xlarge instance presents 64 available NeuronCores"; at LNC=1 it
  presents 128, and "both physical NeuronCores have access to the entire 24GB HBM bank"
  (`RES/aws-neuron-sdk/about-neuron/arch/neuron-features/logical-neuroncore-config.rst:33-100`).
- vllm-neuron: `_DEFAULT_LNC_CONFIG = 2`, "Hardcoded to 2 for current Trainium chips"; device
  index of a logical core = `lnc * lnc_config // 8` (8 physical cores per device)
  (`SRC/vllm_neuron/utils/hardware_config.py:197-232`); each worker gets one visible logical core
  through `NEURON_RT_VISIBLE_CORES` (`SRC/vllm_neuron/vllm/worker/neuron_worker.py:688-694`), the
  same one-core-per-rank arrangement as Kiln's `tp.neuron_env`. It rejects
  `NEURON_RT_VISIBLE_CORES` set by the user in multiprocessing mode and expects
  `NEURON_VISIBLE_DEVICES` instead (L527-531). The core allocator's example host is "64 on trn2"
  (`SRC/vllm_neuron/utils/core_allocator.py:40`).
- NKI: `kernel[lnc](...)`, default 1, "The LNC value must match the NEURON_LOGICAL_NC_CONFIG
  environment variable ... Mismatching the two will cause a runtime error" (`SRC/nki/__init__.py:46-51`).
  vllm-neuron's kernels assume LNC=2 in places: MLP tiling for grid=2 (`functional/mlp.py:39`),
  head-parallel SWA attention (`functional/attention/swa_fused.py:304`), MoE blockwise "E must
  be divisible by logical_nc_config (2)" (`functional/moe/moe_blockwise.py:511`).

### 3.3 FP8

- vllm-neuron: "TRN2 uses e4m3 (with inf), max finite = 240. TRN3 uses e4m3fn (no inf), max finite
  = 448"; the clamp is 448 when the target `startswith("trn3")`, else 240
  (`SRC/vllm_neuron/utils/dtype_utils.py:16-37`). nkilib agrees: `qkv_cte` returns
  "448.0 for trn3 (gen4+), 240.0 for earlier hardware (trn1/trn2)" (`SRC/nkilib/core/qkv/qkv_cte.py:82-86`);
  `rmsnorm_quant_torch.py:26-27`.
- LNL injects `--internal-hlo2tensorizer-options=--experimental-unsafe-fp8e4m3fn-as-fp8e4m3`
  **only when the resolved target == "trn2"**: "Trn3+ supports OCP natively and adding this flag
  triggers NCC_EOCP001 against NKI kernels that emit OCP"
  (`SRC/libtorch_neuronx_lite/compile/backend.py:725-748`). The injection is idempotent (L751-764).
  vllm-neuron adds the same option itself unless the target is `trn3` or `trn3pre`
  (`neuron_model_runner.py:1403-1413`).
- NKI ISA: `float8_e4m3fn` is a valid matmul input only from gen4: "float8_e4m3fn (OCP) requires
  Trn3+. Pre-Trn3 hardware only has the legacy float8_e4m3 FP8 mode"
  (`SRC/nki/isa/_validation.py:1257-1260`); legacy and OCP FP8 cannot be mixed in one matmul
  (L1275-1278); FP8 "double performance mode" exists on NeuronCore-v3 and v4
  (`SRC/nki/isa/__init__.pyi:496`).

### 3.4 MX and FP4

- `nc_matmul_mx` (MXFP8 / MXFP4 operands) is "Available only on NeuronCore-v4 and newer"
  (`SRC/nki/isa/__init__.pyi:708-791`).
- vllm-neuron: GPT-OSS `quantization='mxfp4'` "is not supported on TRN2. Please use
  quantization='bf16'"; on trn3 the MXFP4 model is the default (`SRC/vllm_neuron/model/gpt_oss/factory.py:43-79`).
  Llama FP8 static per-tensor routes to the MX FP8 module on `trn3`/`trn3pre` ("Trn3 has STATIC_MX
  kernels (4x prefill speedup)") and to the legacy static FP8 module on trn2
  (`SRC/vllm_neuron/model/llama3/quantization.py:289-304`). The trn2 GPT-OSS tutorial: "Trn2
  does not support the MXFP4 runtime path, so vLLM Neuron runs the model in BF16"
  (`RES/vllm-neuron/docs/tutorials/tutorial-eagle3-speculative-decoding-gpt-oss.md:59-64`).
- nkilib MoE TKG: "MX weights are only supported on gen4+ (Trn3+)" (`SRC/nkilib/core/moe/moe_tkg/moe_tkg.py:458-459`);
  `rmsnorm_mx_quantize_tkg only supports gen4+` (`core/subkernels/norm_tkg_utils.py:611-612`).

### 3.5 Other gen4-only ISA features that change kernel design

- Tensor indirection (runtime-indexed SBUF/PSUM access) on VectorE, ScalarE, GpSimd and TensorE
  instructions: every `validate_*_indirection` asserts `get_nc_version() >= nc_version.gen4`
  (`SRC/nki/isa/_validation.py:542-930`, including `validate_matmul_indirection` at L907-930).
  On trn2 a paged-KV gather must stay on software DGE DMA (`neuron-stack.md` 6.2).
- `nisa.exponential` on VectorE and `activate2` are gen4 (`__init__.pyi:2236-2310`, 1537-1575);
  nkilib uses the Vector-engine exp only on Trn3 B1 (`core/attention/attention_tkg.py:3250-3318`).
- BF16 PSUM output is gen4 only (`_validation.py:1513-1515`; `core/attention/attention_cte.py:731-732`).
- SBUF capacity per partition: gen2 192 KiB, gen3 224 KiB, gen4 256 KiB (`_validation.py:235-237`);
  usable 212,984 B on trn2 and 245,752 B on trn3 (`core/moe_block/moe_block_tkg_utils.py:293-294`).
- `attention_tkg` batches 64 K-DMAs on trn3 and 32 on trn2 "because TDG is unavailable on trn2"
  (`core/attention/attention_tkg.py:2295-2296`).

### 3.6 Compiler arguments vllm-neuron uses (Kiln does not)

`--auto-cast=none --verbose=35 -O<level> --internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10
[--experimental-unsafe-fp8e4m3fn-as-fp8e4m3] --internal-backend-options=--enable-verifier=false
--enable-nested-dynamic-loop`, the last because inlined NKI kernels may contain nested dynamic
loops (`SRC/vllm_neuron/vllm/worker/neuron_model_runner.py:1401-1425`). The same function is
platform-independent apart from the FP8 option.

### 3.7 Collectives and expert parallelism

- Runtime variable-size collectives `AllGatherV`, `ReduceScatterV`, `AllToAllV` "on Trn2 and Trn3"
  (SDK 2.32 what's new, `RES/whatsnew.txt:1289-1295`); max 16 communicators per NEFF
  (`neuron-stack.md` 3.3, [S28]).
- vllm-neuron's `all_gather_v` requires "Group size must be exactly 4 (TP4, intra-chip, LNC=2)"
  (`SRC/vllm_neuron/functional/collectives/all_gather_v.py:105-120`).
- vllm-neuron's EP all-to-all manager needs either NeuronSwitch (`VLLM_NEURON_SWITCH_CC=1`, which
  asserts "requires Trn3+") or a multi-server 2D torus; a single trn2 instance is rejected
  (`SRC/vllm_neuron/parallel/all2all.py:88-111`). On one trn2.48xlarge, MoE therefore runs with
  experts sharded inside a TP group, which is what Kiln does today.

### 3.8 NKI Library per generation

| Kernel family | Path (`SRC/nkilib/...`) | trn1 (gen2) | trn2 (gen3) | trn3 (gen4) | Evidence |
|---|---|---|---|---|---|
| Attention CTE / TKG (block KV via `active_blocks_table`), segmented CTE | `core/attention/` | code paths exist; "NKI 0.3.0 kernels are not supported on Trn1/Inf2" (NxDI 2.29 notes, `neuron-stack.md` 2.1) | yes | yes, extra fast paths | `attention_tkg.py:1562, 2295, 3250-3318, 3354, 3579` |
| QKV CTE/TKG, output projection, RMSNorm, RoPE, router top-k, MLP CTE/TKG | `core/qkv`, `core/output_projection`, `core/rmsnorm`, `core/router_topk`, `core/mlp` | gen2 branches present, not tuned | yes | yes | `qkv_tkg.py:649`, `qkv_cte_utils.py:1117-1169`, `output_projection_tkg.py:770, 1399` |
| MoE CTE/TKG, BF16/FP8 experts | `core/moe/moe_cte`, `core/moe/moe_tkg` | fp32 fallbacks | yes | yes | `moe_cte_utils.py:286`, `bwmm_shard_on_I.py:879` |
| MoE CTE/TKG with **MX** weights (`*_mx*.py`) | `core/moe/...` | no | no | **only** | `moe_tkg.py:458-459`; `all_expert_mx_impl.py:2155-2156` |
| MXFP8 attention TKG, MXFP8 matmul / MLP / MoE / quantize | `experimental/attention_mxfp8`, `matmul_mxfp8`, `mlp_mxfp8`, `moe_mxfp8`, `quantize_mxfp8` | no | no | only (use `nc_matmul_mx` / `quantize_mx`) | grep `nc_matmul_mx|quantize_mx` in those dirs |
| DeepSeek-V3.2 MLA CTE (qkv, sparse attention, v-up + o_proj) | `experimental/mla/deepseek` | no | no (inputs are packed MX: "MX qkv, attention, MX V-up + MX o_proj") | yes | `mla_validate_params.py:15, 97-116` |
| Sparse attention indexer | `experimental/sparse_attention_indexer` | no | bf16 helpers exist; main entry is MX | yes | file list; MX ISA in 5 of 8 files |
| Attention block TKG megakernel | `experimental/transformer/attention_block_tkg.py` | no | yes | yes | "Kernel requires nc-version >= gen3" (L1054-1055) |
| Ring attention fwd | `experimental/attention/ring_attention_fwd.py` | no ("not supported on trn1", L966) | yes | yes | |
| Selective scan, linear scan, SSD (Mamba-style; nearest to GDN/KDA) | `experimental/scan` | | yes ("requires trn2 shared memory", `selective_scan.py:79`) | yes | |
| Fine-grained all-gather, FGCC (all-gather + matmul overlap) | `experimental/collectives` | | yes | yes | |

ANALYSIS: no NKI Library kernel exists for Gated DeltaNet, Kimi Delta Attention, mHC or
DeepSeek-V4 compressed sparse attention on any generation; Kiln writes those regardless of chip.

---

## 4. Managed baselines and published numbers on trn2 / trn3

### 4.1 Support for the target models

| Model | vllm-neuron 0.24 (SDK 2.32) | NxD Inference 0.10 (frozen; SDK 2.31.1 DLAMI) | Nearest community port | Source |
|---|---|---|---|---|
| MiMo-V2.6-Pro | no | no | NxDI PR #150 MiMo-V2.5-Pro (open, SDK 2.29) | registry below; https://github.com/aws-neuron/neuronx-distributed-inference/pull/150 |
| MiMo-V2.6-Flash | no | no | vllm-neuron PR #40 MiMo-V2.5 (open, BF16, on 0.24); NxDI PR #137 MiMo-V2-Flash, #148 MiMo-V2.5 (open) | https://github.com/vllm-project/vllm-neuron/pull/40 |
| GLM-5.3 | no | no | NxDI contrib GLM-5.2 (merged as PR #177, 2026-07-28; NXDI 0.9.0) | `RES/neuronx-distributed-inference/contrib/models/GLM-5.2/README.md:11-22` |
| GLM-5.3-Flash | no | no | none | |
| Kimi K3 | no | no | NxDI PR #145 Kimi-K2.5, #131 Kimi-K2-0905 (open) | PR bodies |
| Qwen3.8-27B / Flash-Next / 2.4T | no (`Qwen3ForCausalLM` is the 2025 Qwen3) | no | vllm-neuron PR #54 (Qwen3.5 dense), #55 (Qwen3.5 MoE); NxDI contrib Qwen3.5-2B and -35B-A3B (merged), PR #173 Qwen3.6-27B (open) | PR bodies; `contrib/models/Qwen3.5-35B-A3B/README.md` |
| DeepSeek-V4.1-Flash | no | no (V3 only) | NxDI contrib DeepSeek-V3 (merged) | `contrib/models/DeepSeek-V3/README.md` |
| Controls: Qwen3 dense, Llama 3.x, GPT-OSS 20B/120B | **yes** | yes | | `SRC/vllm_neuron/model/registry.py:21-25` (`LlamaForCausalLM`, `GptOssForCausalLM`, `Eagle3LlamaForCausalLM`, `Qwen3ForCausalLM`, `Qwen3VLForConditionalGeneration`) |
| Control: Qwen3-30B-A3B / 235B-A22B | no (PR #39 open) | yes (`models/qwen3_moe`) | | `RES/neuronx-distributed-inference/src/neuronx_distributed_inference/models/` |

### 4.2 Published numbers on trn2 (none on trn3)

AWS's benchmark section has pages for inf1, inf2 and trn1 only (`RES/aws-neuron-sdk/about-neuron/benchmarks/`);
there is **no trn2 or trn3 performance page**. Every trn2 number found:

| Model | Stack, SDK | Instance, parallelism, precision | Workload | Result | Source |
|---|---|---|---|---|---|
| Llama 3.3 70B | NxDI | trn2.48xlarge, TP=64, BF16, BS=1 | 10,000 in / 1,501 out, 1 concurrent | TTFT 814.2 ms, TPOT 19.6 ms, 36 tok/s; fused spec (Llama 3.2 1B draft): TPOT 5.3 ms, 144 tok/s | `RES/aws-neuron-sdk/libraries/nxd-inference/tutorials/llama70b_perf_comparison.csv`; `trn2-llama3.3-70b-tutorial.rst:124, 520-557` |
| Llama 3.1 405B | NxDI | trn2.48xlarge, BF16 | tutorial | TTFT 2,442 ms, TPOT 37.9 ms, 25.46 tok/s; fused spec 8.27 ms, 102.4 tok/s | `.../tutorials/llama405b_perf_comparison.csv` |
| GPT-OSS-120B | vllm-neuron, SDK >= 2.31 | trn2.48xlarge, TP=16, BF16 (MXFP4 dequantized at load), max-num-seqs 1 | `vllm bench serve` sonnet 512 in / 128 out, 30 prompts, concurrency 1 | 112.13 output tok/s, median TPOT 7.00 ms; EAGLE3 134.53 / 5.82 | `RES/vllm-neuron/docs/tutorials/tutorial-eagle3-speculative-decoding-gpt-oss.md:50-77, 168-178, 301-306` ("Representative numbers") |
| Llama 3.1 8B Instruct | vllm-neuron, SDK >= 2.32 | trn2.48xlarge, TP=8 | sonnet, concurrency 2 | 184.44 tok/s, TPOT 10.52 ms; EAGLE3 288.63 / 6.41 | `tutorial-eagle3-speculative-decoding-llama-3-1.md:50-67, 273-281` ("Numbers are illustrative", L320) |
| MiMo-V2.5-Pro (V2.6-Pro architecture) | NxDI contrib PR #150, SDK 2.29 | trn2.48xlarge, FP8, TP=64 / moe_ep=64, BS=48 | vLLM serving | c=1: 4.3 out tok/s, TTFT 1,392 ms, TPOT 220 ms; c=16: 35.6, TPOT 422; c=48: 55, TPOT 752 | PR #150 body |
| MiMo-V2.5 (V2.6-Flash architecture) | vllm-neuron PR #40 on 0.24 | trn2.48xlarge, BF16, TP=64/EP=64 | `vllm bench serve` 900 in / 90 out | c=1: 7.67 tok/s, TTFT 308 ms, TPOT 128.5 ms; c=32: 126.19 tok/s, TPOT 232.4 ms; NxDI FP8 path at c=1: TTFT 485, TPOT 58 ms | PR #40 body |
| GLM-5.2 (GLM-5.3 architecture) | NxDI contrib, NXDI 0.9.0, neuronx-cc 2.24 | trn2.48xlarge, TP=64, LNC=2, FP8 experts, BS=1 | seq 2048 | TTFT 6,377 ms, ITL 241.5 ms, 2.96 tok/s; "DSA indexer disabled" | `contrib/models/GLM-5.2/README.md:11-22` |
| DeepSeek-V3-0324 | NxDI contrib, SDK 2.28 | trn2.48xlarge, TP=64, LNC=2, BF16 (FP8 dequantized) | bs=1, seq 512 | TPOT 20.5 ms (48.7 tok/s), TTFT 1,667 ms (256 in) | `contrib/models/DeepSeek-V3/README.md:57-67, 150-158` |
| Kimi-K2.5 | NxDI PR #145, SDK 2.29 | trn2.48xlarge, TP=64, LNC=2, FP8 per-channel experts | seq 512 | TPOT 21.4 ms (46.6 tok/s) | PR #145 body |
| Qwen3.5-35B-A3B | NxDI contrib, neuronx-cc 2.26.6360 | trn2.48xlarge, TP=8, BF16 | seq 512 | TTFT 561.4 ms, TPOT 7.4 ms (136 tok/s) | `contrib/models/Qwen3.5-35B-A3B/README.md:134-157` |
| Qwen3.5-397B-A17B | vllm-neuron PR #55 | trn2.48xlarge, world 64, ep 8, BF16 | batch 1, 128 tokens | TPOT 53.6 ms (54.18 disaggregated) | PR #55 body |
| Qwen3.5-2B | vllm-neuron PR #54 on 0.24 | **trn2.3xlarge**, TP=4 (4 logical cores), BF16 | 1,024 in / 128 out | B=1 TPOT 3.88 ms; B=8 464.8 agg tok/s | PR #54 body |
| Qwen3-30B-A3B | vllm-neuron PR #39 on 0.21 | trn2.48xlarge, FP8 experts | | ITL 25.0 ms, 40.0 tok/s | PR #39 body |

None of these is a Kiln-comparable number: different SDKs (2.28-2.32), mostly batch 1, and
community code. They set the bar Kiln has to clear and say which configuration to reproduce
first. One lesson from PR #40 applies to Kiln directly: `index_put_` on a cache tensor shared by
several layers lowered to full-pool copies under neuronx-cc ("~124 GB of pool traffic per decode
step") because the aliasing pass emits one alias per placeholder, not per write; Kiln keeps one
cache tensor per layer (`kiln/engine/model_runner.py:238-239`), which is the safe shape, and must
keep it.

---

## 5. Memory fit per model (ANALYSIS from cited checkpoint sizes)

Checkpoint sizes are from `models.md` 1.1 (decimal GB). Per-rank budget: 24 GiB = 25.8 GB per
logical core on trn2 and 36 GiB = 38.7 GB on trn3 (LNL `HBM_MEMORY_GB`), minus KV, the shared
scratchpad and per-NEFF DMA-ring reservations (1.8 GB of spill rings and 0.45 GB scratchpad were
measured on a trn1 core for MiMo-V2.6-Flash, `docs/neuron-notes.md`). Trn3 unit: a 16-chip
server, 64 logical cores, 2,304 GiB (`trn3pd`/`trn3pds` have 16 devices; size UNCONFIRMED as a
rentable unit).

| Model (Kiln status) | Weights used | trn2.3xlarge (4 x 25.8 GB) | trn2.48xlarge (64 x 25.8 GB) | Trn3 16-chip (64 x 38.7 GB) |
|---|---|---|---|---|
| Qwen3-0.6B / 8B / 32B (done) | BF16 1.5 / ~16.4 (8.2B params x 2 B, not in `models.md`) / 65.5 | TP=1..4 / TP=1..4 / TP=4: 16.4 per rank | any TP | any TP |
| Qwen3-30B-A3B (done, trn1 tp=8) | BF16 61 | TP=4: 15.3/rank | TP=8: 7.6/rank, DP=8 | yes |
| MiMo-V2.6-Flash (CPU done, trn1 in progress) | FP8 re-quant ~315; packed MXFP4 177.7 | no | TP=32: 9.8 (FP8); TP=64: 4.9 | native MXFP4 TP=16: 11.1 |
| MiMo-V2.6-Pro (CPU done) | FP8 ~1,040; packed MXFP4 573.5 | no | TP=64: 16.3 (FP8) or 9.0 (packed) | native TP=64: 9.0 |
| GLM-5.3 (MLA+DSA planned) | FP8 755.6 | no | TP=64: 11.8 | TP=64: 11.8 |
| GLM-5.3-Flash (KDA+DSA planned) | FP8 328.3 | no | TP=32: 10.3 | TP=16: 20.5 |
| Kimi K3 (KDA planned) | native MXFP4 1,560.9 | no | TP=64: 24.4 of 25.8, **no room for KV: does not fit** | TP=64: 24.4 of 38.7: fits |
| Qwen3.8-27B (GDN planned) | BF16 55.6 / FP8 30.9 | **TP=4: 13.9 / 7.7** | yes | yes |
| Qwen3.8-Flash-Next (GDN+QSA planned) | FP8 185.5 | no | TP=16: 11.6 | yes |
| Qwen3.8-2.4T-A95B (planned) | FP8 2,496 / MXFP4 1,372 | no | FP8 no; MXFP4 TP=64: 21.4, too tight | MXFP4 TP=64: 21.4: fits |
| DeepSeek-V4.1-Flash (CSA2, mHC, Engram planned) | native 510.3; FP8 re-quant ~770 | no | TP=64: 8.0 packed / 12.0 FP8 | native |
| GPT-OSS-120B (arch not in Kiln) | BF16 ~234 / MXFP4 65.2 | no (BF16) | TP=16: 14.6 (BF16, as AWS runs it) | MXFP4 |

---

## 6. Kiln gap list

Ordered by what blocks the first trn2 measurement. File references are to this branch.

1. **One platform module.** Today the target is read in two places (`kiln/engine/model_runner.py:43-75`)
   and nothing else knows the chip. Add one resolver that returns the raw target (`trn1`,
   `trn2`, `trn3`, `trn3pre`, revision suffixes), the family, the NKI gen, the runtime LNC,
   logical cores visible, HBM per logical core (LNL's table: 16 / 24 / 36 GiB), the FP8 max
   (240 / 448) and whether MX matmul exists. Every bench result and log line records target and
   LNC next to instance type and SDK.
2. **LNC.** Set `NEURON_LOGICAL_NC_CONFIG` explicitly before LNL is imported (`kiln/engine/tp.py:42-45`,
   `kiln/engine/engine.py:41-45`), default 2 on trn2/trn3 and 1 on trn1/inf2. Pass
   `--logical-nc-config=<n>` to neuronx-cc only on trn2/trn3 (trn1 rejects it, NCC_EARG001), and
   assert compiler and runtime agree. `NEURON_RT_VISIBLE_CORES` indices become logical cores
   (0..63 on trn2.48xlarge at LNC=2, 0..3 on trn2.3xlarge). Any NKI kernel call uses
   `wrap_nki(k)[lnc]` with the runtime value.
3. **Compiler arguments.** On trn2/trn3 pass the set vllm-neuron passes (section 3.6):
   `--auto-cast=none`, an explicit `-O`, `--modular-flow-mac-threshold=10`, and the
   `--enable-nested-dynamic-loop` backend option once NKI kernels are inlined. A comparison with
   vllm-neuron is only fair with the same compiler options, or with both settings measured.
   Keep the FP8 flag off on any `trn3*` target (LNL: NCC_EOCP001); Kiln's `startswith("trn3")`
   check already covers `trn3pre` and revision suffixes.
4. **FP8.** `fp8_e4m3_max` and the KV clamp already follow the target. `quantize_fp8_rows`
   hard-codes 240 (`kiln/models/quant.py:125-130`); take the device max so trn3 uses its full
   448 range. `fit_e4m3_max` becomes a no-op on trn3 (correct as is). Later: FP8 x FP8 prefill
   matmuls on trn2 (double-row mode, 2x BF16 FLOPs) instead of weight-only dequant.
5. **MX on trn3.** Route MXFP4 / MXFP8 experts to `nc_matmul_mx` (nkilib MoE TKG/CTE MX paths)
   instead of Kiln's in-graph MXFP4 decode, which is VectorE-bound (11 ms for 32 expert blocks
   on trn1, `kiln/models/quant.py:18-23`). On trn2 keep FP8 re-quantization: AWS itself rejects
   MXFP4 on trn2.
6. **Memory sizing.** `kv_cache_gb` defaults to 4.0 per rank (`kiln/config.py:246`). Size KV from
   HBM per logical core minus weights minus measured NEFF reservations (`nrt_get_vnc_memory_stats`,
   `neuron-stack.md` 3.3), never from `describe-instance-types` (section 1.2).
7. **TP at 64 ranks.** Rank 0 pickles and broadcasts every graph call over gloo
   (`kiln/engine/tp.py:60-67`); at 64 ranks that host cost is unmeasured and may dominate small
   decode steps. `init_rank` gives each rank `cpu_count // world` threads: 3 per rank on
   trn2.48xlarge, and trn2.3xlarge has only 12 vCPU for 4 ranks plus the scheduler. Measure
   before choosing TP; consider DP replicas of TP=4 (one chip each) as the default layout.
8. **Collectives.** Keep TP groups chip-aligned (4 logical cores per chip; vllm-neuron's
   `all_gather_v` only supports intra-chip groups of 4). MoE stays TP-sharded on one trn2
   instance; EP all-to-all is a trn3 (NeuronSwitch) or multi-instance feature. Watch the
   16-communicators-per-NEFF limit when TP and DP groups coexist in one graph.
9. **Kernels.** The trn1 benchmark already showed Kiln losing 16-21% to NxDI-era kernels at
   batch 12-32 (`bench/results/2026-10-02-trn1.2xlarge-qwen3-0.6b.md`). On trn2 the NKI Library
   is tuned for the target, so adopt `attention_tkg` (block KV), `attention_cte`, `qkv`,
   `rmsnorm`, `router_topk` and `moe_tkg` behind Kiln's torch references, each checked for
   numeric parity on device. Separate the trn3-only paths (MX, Vector exp, BF16 PSUM, tensor
   indirection) and the A0/B1 split (`is_trn3_b1`).
10. **Compile on cheap hosts.** LNL compiles without a device when
    `NEURON_PLATFORM_TARGET_OVERRIDE` is set (section 3.1), and its cache hash includes the
    target, so trn2/trn3 NEFFs could be built on a trn1 or CPU host and pulled from S3, keeping a
    $14.56/h instance out of neuronx-cc. Whether LNL can trace Kiln's graphs to a NEFF without a
    live `neuron` device is **UNCONFIRMED** (vllm-neuron's `VLLM_NEURON_CPU_COMPILE` path suggests
    yes). Also put target and LNC in the S3 prefix (`kiln/compile_cache.py` pulls every missing
    entry under one prefix per SDK).
11. **Baseline harness.** `bench/baseline_vllm.py` was written for the NxDI-based vllm-neuron
    0.5.3: it sets `NEURON_RT_NUM_CORES` (L34) and `num_gpu_blocks_override` (L41). vllm-neuron
    0.24 expects `NEURON_VISIBLE_DEVICES`, sizes KV through a budget cap (default 0.30 of the
    GMU budget, `neuron-stack.md` 5 #5) and needs `VLLM_NEURON_COMPILATION_TIMEOUT`
    (`RES/vllm-neuron/docs/getting-started/quickstart-offline-serving.md:22-29`). Add a 0.24 mode,
    and a `vllm bench serve` wrapper so AWS's published configurations (section 4.2) can be
    reproduced on the same box first.
12. **Infrastructure.** `infra/fleet.sh` is us-east-2 only (`REGION`, L22; bucket L31). trn2.3xlarge
    needs `KILN_REGION=sa-east-1` (or ap-southeast-4) plus `bootstrap` there for the security group;
    the IAM role is global and S3 in us-east-2 works cross-region. The 300 GB root volume (L28) is
    too small for MiMo-V2.6-Pro (573.5 GB): mount the 4 x 1.92 TB local NVMe on trn2.48xlarge.
    Before a trn2.48xlarge in us-east-2, the trn1.32xlarge must be terminated (quota, section 1.6).
13. **Re-run every trn1 probe.** Each fact in `docs/neuron-notes.md` (unequal `torch.split`
    miscompile, `index_put_` aliasing, 2-D gather, 94 us per-call floor, queue clamp at 63, FX
    normalisation, f64 float literals, per-NEFF HBM reservations) was measured on trn1 with LNC=1
    and must be re-measured on trn2 at LNC=2 with `tools/debug_device.py` and
    `tools/smoke_device.py` before any model runs.
14. **Architectures.** The top models need MLA + DSA (GLM-5.3, DeepSeek-V4), Gated DeltaNet
    (Qwen3.8), KDA (Kimi K3, GLM-5.3-Flash), mHC and Engram, already planned in `FEATURES.md`;
    trn2/trn3 add no new architecture work, but GPT-OSS (the strongest official baseline on both
    chips) is not in Kiln and is the cheapest head-to-head against AWS's best-tuned path.

---

## 7. Measurement plan

Rules (CLAUDE.md): correctness first (greedy token parity or logit tolerance vs transformers on
the same fixed prompts); the baseline runs on the same instance type and SDK with identical
prompts and output lengths; every result records instance, AZ, AMI id, SDK, target string, LNC,
model id, precision, command, commit and date; raw logs under `bench/results/`. Each row also
reports $/Mtok at the spot price paid. Workloads:

- **W1** = `bench/offline.py`'s `workload()` (256 prompts, 100-1,024 in, 100-1,024 out,
  temperature 0.6, ignore_eos), the trn1 series' workload, at equal total concurrency.
- **W2** = AWS's own published configuration for that model (section 4.2), reproduced on the same
  box first so the harness is known to match AWS's number before Kiln is compared.
- **W3** = long context: 8,192 in / 256 out at concurrency 1, 8, 32 (decode page-axis bucketing
  is Kiln's structural advantage; vllm-neuron needs its opt-in second bucket dimension).

### Phase T0: code, no instance ($0)

Gaps 1-4, 6, 10 (S3 prefix), 11 and 12 in section 6, CPU-tested where possible, plus a trn3
compile-only smoke if gap 10 holds (any later trn1/trn2 session, minutes of instance time).

### Phase T1: first trn2 bring-up and comparison (trn2.3xlarge)

| | |
|---|---|
| Instance | trn2.3xlarge spot, sa-east-1c (fallback ap-southeast-4c), multi-framework DLAMI SDK 2.32.0 (ami-06f5c32a48089ad7a in sa-east-1) |
| Steps | `fleet.sh ready`; trn1 probes on trn2 (gap 13); Qwen3-0.6B greedy parity vs transformers; Qwen3-8B parity |
| Kiln | Qwen3-0.6B bf16 at DP=4 x TP=1 and TP=4; Qwen3-8B bf16 at TP=1 x DP=4, TP=2 x DP=2, TP=4; `bench/offline.py --warmup --overlap --piecewise`, page buckets 16/32/64 |
| Baseline | vllm-neuron 0.24.0.1.1.0 from the same DLAMI venv (`/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0`), `Qwen3ForCausalLM`, TP=4 (and DP if it serves several replicas on one chip), same W1 at concurrency 6, 12, 32 |
| Also | W3 at concurrency 1 and 8 for Qwen3-8B |
| Time | ~6 h (patch reboot, probes, ~10 bucket compiles per config, two engines) |
| Cost | 6 h x $1.5286 (30-day mean) = **$9.17**; at today's $2.5690 = **$15.41**; hard cap 8 h = $20.55 |

Why this first: it is the cheapest trn2 box by 6x, it runs the exact SDK and venv Kiln already
runs on trn1, it exercises LNC=2 and logical-core numbering at TP=4 on one chip, and Qwen3 dense
is the one model both engines serve officially, so the result is a valid same-instance,
same-SDK comparison. It also gives a trn1.2xlarge-to-trn2.3xlarge scaling point on the identical
workload (trn1 numbers: 380-690 tok/s at concurrency 6-32).

#### T1 attempt 1 (2026-10-03 UTC): no spot capacity

Code for T1 is on branch `feat/trn2` (CPU-tested, never run on a device): `kiln/platform.py`
(gaps 1-3: target detection before the runtime from `/sys/class/dmi/id/product_name`, the file
the Neuron driver reads in `neuron_arch.c`; `NEURON_LOGICAL_NC_CONFIG` set to 2 on trn2/trn3
with a matching `--logical-nc-config`; vllm-neuron's compiler argument set with one merged
`--internal-hlo2tensorizer-options`; trn1 arguments unchanged), `infra/fleet.sh --region`
(per-region AMI parameter and `kiln-ssm` security group, bucket kept in us-east-2,
`KILN_SPOT_MAX_PRICE`, `KILN_DRY_RUN`, detached `bg` / `log` jobs), `bench/baseline_vllm.py`
for vllm-neuron 0.24 (gap 11: `NEURON_VISIBLE_DEVICES`, vLLM-sized KV, `--dp`,
`--context-buckets`). Security groups `kiln-ssm` (Project=kiln, no inbound) now exist in
sa-east-1 (<security-group>) and ap-southeast-4 (<security-group>).

The launch was refused 16 times over 21 minutes in sa-east-1c and ap-southeast-4c (readings in
`docs/neuron-notes.md`, AWS account behaviour); nothing was launched and nothing was spent.
EC2's own answer names sa-east-1a and sa-east-1b as having capacity: 1b costs $7.72/h (8 h =
$62, three times this phase's cap) and 1a has no spot price record, so neither was tried.
`KILN_SPOT_MAX_PRICE=2.60` makes any AZ safe to try within the $21 cap (the request is refused
above that price).

Attempt 2 (2026-10-03 06:40-09:35 UTC): sa-east-1a, sa-east-1c and ap-southeast-4c every ten
minutes for 3 h at `KILN_SPOT_MAX_PRICE=2.60`, 51 of 51 refused for capacity (readings in
`docs/neuron-notes.md`). No other AZ was at or under $2.60/h. Remaining ways onto a trn2 are
owner decisions: sa-east-1b spot at about $7.72/h, an on-demand trn2.3xlarge (no list price
published, section 1.4), a trn2.48xlarge in us-east-2 (quota blocked while the trn1.32xlarge
runs), or the ap-south-2 Capacity Block (prepaid, AMI copy needed).

Runbook for the next attempt (one job on the box at a time; `bg` returns a log path for `log`):

```sh
export KILN_REGION=sa-east-1        # or ap-southeast-4 (no patch association there)
KILN_SPOT_MAX_PRICE=2.60 infra/fleet.sh up kiln-trn2 trn2.3xlarge sa-east-1c
infra/fleet.sh ready kiln-trn2
infra/fleet.sh py kiln-trn2 tools/smoke_device.py platform lnl_native_device lnl_compiled_call_overhead nki_from_torch nki_standalone
infra/fleet.sh py kiln-trn2 tools/debug_device.py write_then_read gather2d logits split_variants queue
infra/fleet.sh bg kiln-trn2 tools/check_device.py --model Qwen/Qwen3-0.6B --piecewise
infra/fleet.sh py kiln-trn2 tools/check_ppl.py --model Qwen/Qwen3-0.6B
# W1, Qwen3-0.6B bf16, 12 and 32 sequences in flight in total
infra/fleet.sh bg kiln-trn2 bench/baseline_vllm.py --model Qwen/Qwen3-0.6B --tp 4 --max-num-seqs 12 --max-num-batched-tokens 2048
infra/fleet.sh bg kiln-trn2 bench/baseline_vllm.py --model Qwen/Qwen3-0.6B --tp 1 --dp 4 --max-num-seqs 3 --max-num-batched-tokens 2048
infra/fleet.sh bg kiln-trn2 bench/offline.py --model Qwen/Qwen3-0.6B --dp 4 --max-num-seqs 3 --decode-buckets 3 \
    --page-buckets 16,32,64 --prefill-buckets 512 --kv-cache-gb 6 --warmup --overlap --piecewise
infra/fleet.sh bg kiln-trn2 bench/offline.py --model Qwen/Qwen3-0.6B --tp 4 --max-num-seqs 12 --decode-buckets 12 \
    --page-buckets 16,32,64 --prefill-buckets 512 --kv-cache-gb 6 --warmup --overlap --piecewise
# (repeat both engines at 32: --max-num-seqs 32 / --dp 4 --max-num-seqs 8, Kiln --decode-buckets 32 / 8)
# Qwen3-8B: check_device --tp 4 --piecewise, then the same W1 pairs at TP=4, TP=2 x DP=2, TP=1 x DP=4
infra/fleet.sh down kiln-trn2 && infra/fleet.sh ls --all
```

### Phase T2: trn2.48xlarge controls and MiMo-V2.6-Flash

| # | Model | Kiln | Baseline (same instance) | Workload | Est. h |
|---|---|---|---|---|---|
| 1 | Llama-3.1-8B-Instruct bf16 | TP=8 and DP=8 x TP=8 | vllm-neuron 0.24 TP=8, SDK 2.32; first reproduce AWS's 184.44 tok/s at concurrency 2 | W2, W1 | 2 |
| 2 | Qwen3-32B bf16 | TP=8, TP=16 | vllm-neuron 0.24 | W1, W3 | 2 |
| 3 | Qwen3-30B-A3B bf16 | TP=8 x DP=8 | NxDI 0.10 `qwen3_moe` through vllm-neuron 0.5.3, which ships only on the SDK 2.31.1 DLAMI: run Kiln on that DLAMI too if its vLLM 0.21 venv carries LNL (UNCONFIRMED), otherwise report as a different-SDK reference, not a win | W1 | 3 |
| 4 | MiMo-V2.6-Flash FP8 | TP=32 x DP=2, TP=64 | no official baseline; nearest same-SDK: vllm-neuron PR #40 branch (MiMo-V2.5, BF16, same architecture class) built on 0.24; reference rows: PR #137/#148 (SDK 2.29) | PR #40's 900/90 at c=1..32, W1 | 4 |

Time ~11 h. Cost at us-east-2c spot: 11 x $12.9692 = **$142.66** (30-day mean) to
11 x $14.5593 = **$160.15** (today). Preconditions: the us-east-2 trn1.32xlarge terminated
(quota), weights staged on local NVMe, NEFFs pre-built if gap 10 holds. Spot placement score
is 1: if the request fails twice, the fallback is a 24 h Capacity Block in ap-south-2 at
$858.26 (needs the SDK 2.32 AMI copied there), which is an owner decision because it is
prepaid.

### Phase T3: MiMo-V2.6-Pro on trn2.48xlarge (flagship)

Kiln FP8 re-quantized (~1,040 GB, TP=64) and packed MXFP4 (573.5 GB) at concurrency 1, 16, 48,
against PR #150's published MiMo-V2.5-Pro rows (TPOT 220 ms at c=1, 55 out tok/s at c=48; SDK
2.29, so a reference, not a same-SDK baseline). ~12 h including the 573.5 GB download and
compiles (PR #150 reports ~60 min TKG + ~15 min CTE for its first compile). Cost
12 x $12.97-14.56 = **$155.63-174.71**.

### Phase T4: trn3

Blocked on access: no trn3 type in EC2 and no UltraServer authorization for this account
(section 1). Owner action: ask the AWS account team for Trn3 (and Trn2 UltraServer) access and
pricing. Until then: compile-only checks with `NEURON_PLATFORM_TARGET_OVERRIDE=trn3` (gap 10).
Once a box exists, the order is GPT-OSS-120B MXFP4 (vllm-neuron's official trn3 path, the
strongest same-instance baseline), MiMo-V2.6-Pro native MXFP4, then Kimi K3 (the first target
model that fits only on trn3). Cost unknown: AWS publishes no Trn3 price.

### Later (after the architecture work in gap 14)

Qwen3.8-27B BF16 on trn2.3xlarge at TP=4 (fits one chip; baseline: vllm-neuron PR #54's Qwen3.5
dense branch on 0.24), GLM-5.3 FP8 on trn2.48xlarge at TP=64 (reference: GLM-5.2 contrib, 2.96
tok/s at 2K with DSA off), DeepSeek-V4.1-Flash on trn2.48xlarge.

### Budget summary

| Phase | Instance | Hours | Cost (30-day mean - today) |
|---|---|---|---|
| T0 | none | 0 | $0 |
| T1 | trn2.3xlarge, sa-east-1c | 6 (cap 8) | $9.17 - $15.41 (cap $20.55) |
| T2 | trn2.48xlarge, us-east-2c | 11 | $142.66 - $160.15 |
| T3 | trn2.48xlarge, us-east-2c | 12 | $155.63 - $174.71 |
| T4 | trn3 | - | blocked, no price |
| Total trn2 track | | 29 | **$307 - $350** (EBS and S3 under $5) |

---

## 8. UNCONFIRMED and risks

- Customer-launchable Trn3 instance names, sizes, regions and prices (none in EC2 for this account).
- Whether LNL compiles Kiln graphs for trn2/trn3 without a live device (gap 10).
- Whether the SDK 2.31.1 vLLM 0.21 DLAMI venv contains LNL (Phase T2 row 3).
- Whether vllm-neuron 0.24 runs several DP replicas on one trn2.3xlarge chip.
- Effective per-rank HBM at LNC=1 on trn2 (the docs say both physical cores "have access to the
  entire 24GB HBM bank").
- Spot availability: placement score 1 everywhere trn2 is offered.
- The trn1 numerics (bf16 divergence at small top-2 margins) and the LNL miscompiles in
  `docs/neuron-notes.md` may differ on trn2.

---

## 9. How to re-run (read-only)

```sh
export AWS_CONFIG_FILE=$HOME/.aws/config AWS_SHARED_CREDENTIALS_FILE=$HOME/.aws/credentials AWS_PROFILE=default
# bash, not zsh: the region list must word-split
for r in $(aws ec2 describe-regions --region us-east-1 --query 'Regions[].RegionName' --output text); do
  aws ec2 describe-instance-type-offerings --region $r --location-type availability-zone \
    --filters "Name=instance-type,Values=trn*,inf*" --query 'InstanceTypeOfferings[].[InstanceType,Location]' --output text
done
aws ec2 describe-instance-types --region ap-south-2 --filters "Name=instance-type,Values=trn2*" --output json
aws ec2 describe-spot-price-history --region us-east-2 --product-descriptions Linux/UNIX \
  --instance-types trn2.48xlarge --start-time "$(date -u +%FT%TZ)"
aws ec2 describe-spot-price-history --region sa-east-1 --product-descriptions Linux/UNIX \
  --instance-types trn2.3xlarge --start-time 2026-09-02T00:00:00Z
aws ec2 get-spot-placement-scores --region us-east-1 --instance-types trn2.48xlarge \
  --target-capacity 1 --target-capacity-unit-type units --single-availability-zone
aws ec2 describe-capacity-block-offerings --region ap-south-2 --instance-type trn2.48xlarge \
  --instance-count 1 --capacity-duration-hours 24 --all-availability-zones
aws pricing get-products --region us-east-1 --service-code AmazonEC2 \
  --filters Type=TERM_MATCH,Field=instanceType,Value=trn2.48xlarge
aws service-quotas list-service-quotas --region us-east-2 --service-code ec2 \
  --query "Quotas[?contains(QuotaName,'Trn') || contains(QuotaName,'Inf')].[QuotaCode,QuotaName,Value]"
aws ssm get-parameter --region sa-east-1 --name /aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id
```

## Appendix A: raw readings (2026-10-03 UTC)

```
offerings (region): trn2.3xlarge ap-south-2 ap-southeast-4 sa-east-1 | trn2.48xlarge ap-south-2 us-east-2
                    trn2u.48xlarge ap-south-2 | trn3*: none | me-south-1: connect timeout
spot now:   trn2.48xlarge us-east-2c 14.559300 @2026-10-03T01:00Z
            trn2.3xlarge ap-southeast-4c 2.268700 @10-02T23:00Z, sa-east-1c 2.569000 @10-03T00:00Z,
            sa-east-1b 7.726400 @10-02T23:00Z
spot 30d:   trn2.48xlarge us-east-2c n=119 min=8.9914 max=15.3679 mean=12.9692
            trn2.3xlarge ap-southeast-4c n=126 min=2.2579 max=6.9846 mean=3.9655
            trn2.3xlarge sa-east-1b n=127 min=7.5653 max=7.7483 mean=7.6626
            trn2.3xlarge sa-east-1c n=90 min=0.9026 max=2.5690 mean=1.5286
placement:  trn2.48xlarge use2-az3 1 | trn2.3xlarge apse4-az3 1, sae1-az2 1, sae1-az3 1
            trn1.32xlarge use2-az3 9, usw2-az4 9, use1-az4/5/6 1, usw2-az1 1
capacity block: trn2.48xlarge ap-south-2b 8h 308.14 | 24h 858.26 (10-03T11:30Z..10-04T11:30Z) | 32h 1166.40
                trn2.48xlarge us-east-2: none (24/48/72/168h) | trn1.32xlarge us-east-2 24h: none
                UltraServer: UnauthorizedOperation
describe-instance-types: trn2.48xlarge Trainium2 x16, CoreInfo Count 8 Version 3, MemoryInfo 524288 MiB,
                TotalNeuronDeviceMemoryInMiB 8388608 | trn2.3xlarge x1, 524288 | trn2u.48xlarge NeuronInfo {}
```
