# AWS Neuron software stack map, as of 2026-10-01

Scope: primary-source map of the AWS Neuron stack (Trn1/Trn1n, Trn2, Trn2 UltraServer, Trn3, Inf2) for a team building
its own LLM inference engine to replace NxD Inference / transformers-neuronx / the vLLM Neuron plugin.

Method: shallow clones of the public repos (docs repo `aws-neuron/aws-neuron-sdk` at commit `a6be966`, 2026-09-30;
`vllm-project/vllm-neuron` branch `release-0.24.0.1.1.0`; `aws-neuron/nki-library` `92d11f6` 2026-08-28;
`aws-neuron/neuronx-distributed-inference` `4bcdc54` 2026-07-28; `aws-neuron/nkipy` `0bc6ed1` 2026-09-14;
`aws-neuron/aws-neuron-driver` `9cc489e` 2026-08-17; sparse `zml/zml` `platforms/neuron`), the rendered docs (every
docs URL below was fetched on 2026-10-01 and grep-confirmed to contain the cited text), the GitHub API (repo licenses,
releases, issues), and direct inspection of the published wheels / debs on `pip.repos.neuron.amazonaws.com` and
`apt.repos.neuron.amazonaws.com` (METADATA, LICENSE files, file lists). No AWS API calls were made.

Notation: `[Sn]` is a source; the full URL is in section 9 and is also inlined at first use. "UNCONFIRMED" means no
primary source was found. "ANALYSIS" marks my own inference, not a vendor statement.

---

## 0. Headline findings

1. **Latest SDK is Neuron 2.32.0, released 2026-08-17.** No 2.33 exists in the docs repo as of its 2026-09-30 HEAD
   ([S1] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html ,
   [S3] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/whats-new.html ). The only later item is
   "vLLM Omni Neuron Beta" (2026-09-30), a diffusion/video plugin, not an SDK release [S3].
2. **AWS's own managed inference layer is being replaced, by AWS.** NxD Inference (NxDI) is in **maintenance mode**
   (announced 2.30, "no new releases planned" from 2.31/2.32) and is no longer shipped in DLAMIs/DLCs as of 2.32
   ([S36] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/nxd-inference.html ,
   [S37] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/announcements/neuron2.x/announce-nxdi-maintenance-support-2-31.html ,
   [S3]). transformers-neuronx reached end of support in 2.26
   ([S67] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/announcements/neuron2.x/announce-intent-eos-tnx.html ).
   The successor is a **new vLLM Neuron plugin (Beta) that implements models inside the plugin with no NxDI dependency**,
   Trn2/Trn3 only, currently `vllm-neuron 0.24.0.1.1.0` tracking **vLLM 0.24.0** [S3],
   ([S43] https://github.com/vllm-project/vllm-neuron/tree/release-0.24.0.1.1.0 ).
3. **Inf2 and Trn1 have no actively developed LLM serving path.** NxDI dropped Trn1/Inf2 in 2.29 [S36]; Torch/XLA
   inference on Inf2/Trn1 is in maintenance mode from 2.31
   ([S38] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/announcements/neuron2.x/announce-maintenance-pytorch-xla-inference.html );
   a vLLM Neuron maintainer states "The new plugin doesn't support Trn1/Inf2"
   ([S64] https://github.com/vllm-project/vllm-neuron/issues/57 ). That is a gap a custom engine could fill.
4. **A custom engine can bypass all framework layers.** `libnrt` exposes a documented C API (load NEFF, allocate
   tensors, execute, async queues, collectives) with headers published in the public docs repo (the headers
   themselves carry only "Copyright Amazon ... All Rights Reserved"; the repo's summary license covers docs as
   CC-BY-SA-4.0 and sample code as MIT-0, so the header license is UNCONFIRMED beyond that)
   ([S29] https://github.com/aws-neuron/aws-neuron-sdk/tree/master/src/libnrt ,
   [S30] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/guides/nrt-developer-guide.html ). AWS
   says the guide is for "developers building their own ML frameworks" [S30]. Two third-party/experimental precedents
   do exactly this: ZML (via the Neuron PJRT plugin + libnrt) and AWS's own experimental `nkipy`/Spike (direct libnrt
   bindings) ([S60] https://github.com/zml/zml/tree/master/platforms/neuron , [S59] https://github.com/aws-neuron/nkipy ).
5. **What cannot be replaced: driver (GPL-2.0, source public), libnrt + libnccom (proprietary binaries), and the
   compiler backend `neuronx-cc` (proprietary).** Even NKI kernels go NKI -> MLIR -> BIR -> `neuronx-cc` backend ->
   NEFF (from the `nki` wheel's own `ncc_driver.py`, see section 3). The NEFF instruction format is not documented
   beyond its header (UNCONFIRMED that any ISA-level encoding spec is public).
6. **The managed path's structural limits are documented by AWS itself** (section 5): static-shape bucketing, no
   mixed prefill/decode batches, exactly one prefill per step, no preemption (so worst-case KV admission control),
   decode NEFFs that read `max_model_len` worth of KV blocks per request regardless of real length (~32x DMA
   amplification in the doc's own example), no chunked prefill (mixed batching), multi-minute cold compiles, and a
   prefix cache that is vLLM block-hash APC (no radix tree is documented anywhere in the Neuron paths).

---

## 1. SDK release, instance types, chips, dtypes

### 1.1 Latest release and component versions (Neuron 2.32.0, 2026-08-17)

From the release-notes index "Neuron Component Release Notes" table [S1] and the 2.32.0 page
([S2] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/2.32.0.html , "Date of release: August 17, 2026"):

| Component | Latest component version (index table) |
|---|---|
| Neuron Compiler (`neuronx-cc`) | 2.27.5334.0 (the compiler component page header says "Latest version (in 2.32.0): 2.27.4747.0" but its 2.32 section heading says 2.27.5334.0; the pip index carries `neuronx_cc-2.27.5334.0+f702b353`) [S1] [S33] [S53] |
| Neuron Runtime Library (`libnrt`) | 2.34.10.0 [S1]; deb `aws-neuronx-runtime-lib_2.34.10.0-ac18d186d` [S55] |
| Neuron Driver (`aws-neuronx-dkms`) | 2.30.2.0 [S1] |
| Neuron Collectives (`libnccom`) | 2.34.10.0 [S1]; deb `aws-neuronx-collectives_2.34.10.0-74eaafac6` [S56] |
| NKI (Neuron Kernel Interface) | 0.6.0 [S1]; wheel `nki-0.6.0+31049202112.g85070674` [S19] |
| NKI Library | 2.32.0 [S1] |
| NxD Inference | 0.10.18399 (frozen; maintenance) [S1] [S36] |
| vLLM Neuron (Beta) | 0.24.0.1.1.0 [S1] |
| nrtpy (Beta) | 2.34.10 [S1] |
| JAX NeuronX | "0.10.0.* (supports up to JAX 0.9.0)" in the index table, but the JAX component page says "Maximum supported version of JAX (in 2.32.0): 0.10.0" and "JAX versions 0.7.0 to 0.10.0" (internal inconsistency) [S1] [S35] |
| PyTorch NeuronX (`torch-neuronx`) | 2.9.0.2.15.32035 (in 2.31.0); PyTorch 2.9 is the last PyTorch/XLA-based version [S27] |
| Developer tools | 2.32.28.0 [S1] |

Release dates for the previous 12 months [S1]: 2.31.1 08/12/26, 2.31.0 07/07/26, 2.30.0 05/21/26, 2.29.1 and 2.29.0
04/09/26, 2.28.1 03/13/26, 2.28.0 02/26/26, 2.27.1 01/14/26, 2.27.0 12/19/25, 2.26.1 10/29/25, 2.26.0 09/18/25.
Cadence is roughly every 6 to 8 weeks.

Notable 2.32.0 content [S3]: NKI 0.6.0 (`nisa.topk`, `all_gather_v`, runtime loops `fori_loop`/`while_loop` replacing
`nl.dynamic_range`); 13 new NKI Library kernels (DeepSeek-V3.2 sparse-MLA context encoding, MXFP8 flash-decode
attention, MXFP8 blockwise MoE, fused GPT-OSS sliding-window attention block, GpSIMD top-K); runtime variable-size
collectives `AllGatherV`/`ReduceScatterV`/`AllToAllV` on Trn2/Trn3; vLLM Neuron upgraded to vLLM 0.24.0.

### 1.2 Hardware: per-chip and per-instance

| | Trn1 / Trn1n (Trainium) | Inf2 (Inferentia2) | Trn2 (Trainium2) | Trn3 (Trainium3) |
|---|---|---|---|---|
| NeuronCore version | NeuronCore-v2 | NeuronCore-v2 | NeuronCore-v3 | NeuronCore-v4 |
| Physical NeuronCores / chip | 2 [S5] | 2 [S8] | 8 [S6] | 8 [S7] |
| HBM / chip | 32 GiB, 820 GiB/s [S5] | 32 GiB, 820 GiB/s [S8] | 96 GiB, 2.9 TB/s [S6] (NKI guide says 3 TB/s [S16]); four 24 GB banks, each shared by 2 cores [S14] | 144 GiB, 4.9 TB/s [S7] (NKI guide says 4.7 TB/s [S17]) |
| Dense compute / chip | 190 FP16/BF16/cFP8/TF32 TFLOPS, 47.5 FP32, 380 INT8 TOPS [S5] [S8] | same as Trn1 [S8] | 1,299 FP8, 667 BF16/FP16/TF32, 181 FP32 TFLOPS; 2,563 sparse [S6] | 2,517 MXFP8/MXFP4, 671 BF16/FP16/TF32, 183 FP32 TFLOPS; 2,517 FP16/BF16/TF32 sparse [S7] |
| On-chip SRAM | SBUF 24 MiB + PSUM 2 MiB per core [S15] | same [S15] | SBUF 28 MiB per core (224 MiB/chip) + PSUM 2 MiB [S16] [S6] | SBUF 32 MiB per core (256 MiB/chip) + PSUM 2 MiB [S17] [S7] |
| DMA engines | 32 per device, 16 per core [S15] | same | 128 per device [S16] | 128 per device [S17] |
| Chip interconnect | NeuronLink-v2, 384 GB/s/chip, 2D torus [S6] [S9] | NeuronLink-v2, 192 GiB/s/chip [S12] | NeuronLink-v3, 1.28 TB/s/chip, 4x4 2D torus in-instance [S6] [S10] | NeuronLink-v4 2.56 TB/s/device + NeuronSwitch-v1 all-to-all (PCIe Gen6 switched fabric) [S7] [S11] |
| LNC (logical NeuronCore) | not applicable | not applicable | 1 or 2, default 2 (4 logical cores per chip) [S14] | 1 or 2 [S14]; Trn3 default UNCONFIRMED (the NxDI Trn3 GPT-OSS tutorial uses LNC=2) |

Sources: [S5] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium.html ,
[S6] .../trainium2.html , [S7] .../trainium3.html , [S8] .../inferentia2.html , [S9] .../trn1-arch.html ,
[S10] .../trn2-arch.html , [S11] .../trn3-arch.html , [S12] .../inf2-arch.html (all under
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/ ),
[S14] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html .
Note: the Trainium2 page says 16 CC-Cores per chip [S6]; the NKI Trn2 guide says 20 [S16] (inconsistent).

Instances and scale-up domains:

- **trn1.2xlarge** (1 chip), **trn1.32xlarge** (16 chips, 512 GiB HBM, 800 Gbps EFA), **trn1n.32xlarge** (16 chips,
  1,600 Gbps EFA) [S9].
- **inf2.xlarge / inf2.8xlarge** (1 chip), **inf2.24xlarge** (6), **inf2.48xlarge** (12 chips, 384 GiB) [S12].
- **trn2.48xlarge / trn2u.48xlarge**: 16 Trainium2, 1,536 GiB HBM, 46.4 TB/s, 3,200 Gbps EFAv3; at LNC=2 a
  trn2.48xlarge presents 64 NeuronCores, at LNC=1 128 [S10] [S14].
- **Trn2 UltraServer**: 4 x trn2u.48xlarge = 64 Trainium2, 6,144 GiB HBM, inter-instance NeuronLink-v3 256 GB/s/chip [S10].
- **Trn3 Gen1 UltraServer**: 4 servers x 16 Trainium3 = 64 chips, 9,216 GiB HBM, 161 PFLOPS dense MXFP8 [S11].
- **Trn3 Gen2 UltraServer**: 36 servers x 4 Trainium3 = 144 chips, 20,736 GiB HBM, 362 PFLOPS dense MXFP8, intra-sled
  256 GB/s/chip, intra-rack 320 GB/s/chip, inter-rack 128 GB/s/chip [S11]. The EC2 page says "scale up to 144
  Trainium3 chips", "144 GB of HBM3e" per chip, and lists no instance sizes or pricing
  ([S66] https://aws.amazon.com/ec2/instance-types/trn3/ ).
- **Trn3 instance-type strings are UNCONFIRMED publicly.** The open driver source enumerates `trn3pd98.3xlarge`,
  `trn3sn.3xlarge`, `trn3s-es.3xlarge`, `trn3sn-es.3xlarge` as "3xl" Trn3 types
  ([S57] https://github.com/aws-neuron/aws-neuron-driver/blob/master/v4/neuron_dhal_v4.c ), and `nrt.h` has
  `NRT_INSTANCE_TRN3 = 14`, `NRT_INSTANCE_TRN3PDS98 = 15` [S29]; the EKS UltraServer Operator doc says "Trn3pds"
  instance types ([S68] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/deploy/eks/ultraserver-operator.html ).
  Driver strings also include `trn3.48xlarge`, `trn3p.48xlarge`, `trn3e.24xlarge`; whether these are customer-launchable
  is UNCONFIRMED.

### 1.3 Supported data types

- **NeuronCore-v2 (Trn1/Inf2)**: FP32, FP16, TF32, BF16, cFP8 (configurable-range FP8), UINT8, INT32/UINT32
  ([S69] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/data-types.html );
  TensorE runs BF16/FP16/TF32/cFP8 at the same 92 TFLOPS/core, FP32 at 23, always accumulating in FP32 [S15].
- **NeuronCore-v3 (Trn2)**: TensorE FP8_E4/FP8_E5 at **double** BF16 throughput ("double row" mode, contraction 256);
  FP8_E3 still supported at BF16 rate; M:N structured sparsity (4:16, 4:12, 4:8, 2:8, 2:4, 1:4, 1:2)
  ([S16] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html ,
  [S13v3] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v3.html ).
- **NeuronCore-v4 (Trn3)**: OCP **MXFP8 and MXFP4** inputs at 4x BF16 rate (contraction 512, group size 32, E8M0
  scales); MXFP4 is converted to MXFP8 before the array; output FP32 or **BF16 into PSUM**; VectorE can quantize
  BF16/FP16 to MXFP8 natively ([S17] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium3_arch.html ,
  [S13v4] .../neuron-core-v4.html ).
- **NKI dtypes (nki 0.6.0 type stubs)**: `bool_, int8, int16, int32, uint8, uint16, uint32, float16, float32, bfloat16,
  tfloat32, float8_e4m3, float8_e4m3fn, float8_e5m2, float8_e5m2_x4, float8_e4m3fn_x4, float8_e8m0fnu,
  float4_e2m1fn_x4` (the `_x4` types are packed operands for `nc_matmul_mx` on NeuronCore-v4) [S19] (file
  `nki/language/__init__.pyi` in the `nki` wheel at https://pip.repos.neuron.amazonaws.com/nki/ ).
- **libnrt tensor dtype enum**: `FP8_E3, FP8_E4, FP8_E5, FLOAT16, BFLOAT16, FLOAT32, FP32R, (U)INT8/16/32/64` [S29].
- **Compiler**: 2.32 adds `--native-int64` and `--implicit-integer-downcast`; next release flips the defaults to native
  int64 + hard error on unsupported ops [S3] [S33].
- **What vLLM Neuron exposes**: BF16, FP8 weights, FP8 KV cache, MXFP4 and MXFP8 weights (Trn3 only) [S43].

---

## 2. Layers, open vs closed

| Layer | Artifact | Open? | License (evidence) | Source / notes |
|---|---|---|---|---|
| Kernel driver | `aws-neuronx-dkms` 2.30.2.0 | **Yes, source public** | GPL-2.0 (GitHub API) | https://github.com/aws-neuron/aws-neuron-driver [S57] |
| Runtime | `aws-neuronx-runtime-lib` (`libnrt.so.2.34.10.0`, `libnrtucode_extisa.so`, `libncfw.so`, `libnds.a`) | **Binary only; headers public** | "AWS Neuron License Agreement" in `/opt/aws/neuron/share/doc/aws-neuronx-runtime-lib/LICENSE.txt` (inspected in the deb [S55]): no reverse engineering, no redistribution, use "in connection with AWS Services" | headers mirrored at https://github.com/aws-neuron/aws-neuron-sdk/tree/master/src/libnrt [S29] |
| Collectives | `aws-neuronx-collectives` (`libnccom.so.2.34.10`, `libnccom-net.so`) | Binary only | Bundled LICENSE.txt is NVIDIA's NCCL BSD-style copyright text (inspected [S56]), i.e. NCCL-derived; no public source repo found (UNCONFIRMED whether source is obtainable) | [S56] |
| Graph compiler | `neuronx-cc` 2.27.5334.0 (571 MB wheel, 783 `.so`, 117 binaries) | **Closed** | METADATA `License: AWS Neuron Software License Agreement` [S53] | takes XLA HLO (`hlo.pb`, `--framework XLA`) ([S34] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/index.html ); 2.31 added StableHLO composite mapping [S33] |
| NKI compiler / frontend | `nki` 0.6.0 wheel | **Partially**: Apache-2.0 label, Python frontend + `nki-stdlib` visible as `.py`, but the MLIR compiler core ships as binaries (`libNkiPythonCAPI.so` 23 MB, `_nki`, `_frontend`, `_nisaDialectsNanobind`, `_nki_simulator` `.so`) | METADATA `License-Expression: Apache-2.0`, Summary "MLIR-based compiler for Neuron devices" [S19] | No public repo for the NKI compiler exists in the aws-neuron org listing (2026-10-01) [S58]; the C++ sources are UNCONFIRMED as published. It also requires `neuronx-cc` for BIR -> NEFF (section 3). |
| NKI Library | `nki-library` (also bundled in `neuronx-cc` as `nkilib`) | **Yes** | Apache-2.0 | https://github.com/aws-neuron/nki-library [S24] |
| NKI samples | `nki-samples` | Yes | MIT-0 | https://github.com/aws-neuron/nki-samples |
| PyTorch (XLA) | `torch-neuronx` 2.9 + `torch-xla` + `libneuronxla` | Closed (torch-neuronx); `libneuronxla` is proprietary | `libneuronxla/LICENSE.txt` = "AWS Neuron Software License Agreement" [S54] | PyTorch 2.9 is the last PyTorch/XLA version [S27] |
| PyTorch native ("TorchNeuron") | out-of-tree PrivateUse1 backend, eager + `torch.compile(backend="neuron")` | Announced as open source (Apache-2.0) at re:Invent 2025, **but closed Beta**: `github.com/aws-neuron/torch-neuronx` returns 404 to the GitHub API on 2026-10-01 | [S25] [S26] | "TorchNeuron is currently only available as part of a closed Beta program" ([S26] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html ); PyTorch 2.10 transition still "planned for a future Neuron release" [S27]; initial `torch.compile` does not use Inductor and has no CUDA-graph equivalent [S26] |
| JAX | `jax-neuronx` + `libneuronxla` (PJRT C-API plugin) | Plugin is proprietary | [S54] | JAX 0.7 to 0.10 with `libneuronxla 3.*` ([S35] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/jax/index.html , https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/jax.html ) |
| vLLM runtime shim | `libtorch-neuronx-lite` 2.12/2.13 (FX -> torch-mlir/StableHLO -> `neuronx-cc` subprocess) | **Closed** | `Classifier: License :: Other/Proprietary License`, Summary "Custom torch-neuronx runtime for vLLM inference with DI on Neuron"; requires `torch==2.13.*`, `torch-xla==2.13.*` (2.13 wheel) [S52] | used by vLLM Neuron [S51] |
| NxD Inference | `neuronx-distributed-inference` 0.10.18399 | Yes | Apache-2.0 | https://github.com/aws-neuron/neuronx-distributed-inference [S42]; maintenance mode |
| NxD Core | `neuronx-distributed` | Yes | MIT-0 | https://github.com/aws-neuron/neuronx-distributed |
| vLLM Neuron plugin | `vllm-neuron` 0.24.0.1.1.0 | Yes | Apache-2.0 | https://github.com/vllm-project/vllm-neuron [S43]; the legacy 0.5.x NxDI-based branch lives in a private repo (`aws-neuron/private-vllm-neuron`, linked from the README) [S43] |
| Legacy vLLM fork | `aws-neuron/upstreaming-to-vllm` | Yes, **archived** | Apache-2.0 | https://github.com/aws-neuron/upstreaming-to-vllm [S65] |
| Runtime Python bindings | `nrtpy` 2.34.10 (Beta, nanobind) | Binary wheel; source not in a public repo (UNCONFIRMED) | wheel METADATA has no license field [S31] | ([S31] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrtpy/index.html ) |
| Experimental | `nkipy` (NumPy-like front end), **Spike** (C++ libnrt bindings), **nkigen** (NumPy trace -> linalg MLIR -> NISA dialect, 26-stage pipeline) | Yes | Apache-2.0 | https://github.com/aws-neuron/nkipy [S59]; explicitly "not an official part of the Neuron SDK", prototype/alpha |

### 2.1 NKI status and the NKI Library

- NKI 0.3.0 (Neuron 2.29, 2026-04-09) "is now out of Beta and Stable", introduced `nki-stdlib`, a CPU simulator
  (`NKI_SIMULATOR=1` / `nki.simulate`), and `nki.language` high-level wrappers [S3]. The new NKI compiler is "built on
  MLIR" (re:Invent 2025 post [S3]); kernels are compiled by a dedicated NKI Compiler and spliced into the graph "very
  late in the compilation process" ([S22] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki-compiler.html ).
- NKI 0.6.0 (2.32): `nl.fori_loop` / `nl.while_loop` runtime loops, `all_gather_v`, `nisa.topk`, relaxed DMA transpose
  ([S23] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/nki.html ).
- NKI Library core kernels (README of [S24]): Attention CTE, Attention KV-Parallel Segmented CTE, Attention TKG, MLP,
  MoE CTE, MoE TKG, Output Projection CTE/TKG, QKV, RMSNorm-Quant, RMSNorm MX Prefill, RoPE, Router Top-K, Cumsum.
  Experimental: Attention Block TKG (fused RMSNorm+QKV+RoPE+attn+oproj), Transformer TKG megakernel, ring attention
  fwd/bwd, fine-grained all-gather and FGCC (all-gather + matmul overlap), SBUF-to-SBUF all-gather, MXFP8 matmul / MLP /
  MoE bwd / quantize, MXFP8 attention TKG (flash decode), DeepSeek sparse-attention indexer, DeepSeek-V3.2 MX MLP,
  GpSIMD top-K, SSD / selective scan / linear scan, gather, scatter-add, NeuroTile tile-iterator library, and more.
- **Paged KV in kernels**: `attention_tkg` supports a block KV cache via `active_blocks_table [B, num_blocks]` and
  `block_len`, requires `qk_in_sb=True` ([S70] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/attention-tkg.html ).
  There is **no "all-reduce + matmul" fused kernel for decode TP named as such**; the closest are FGCC and SBUF-to-SBUF
  all-gather (experimental) [S24].
- The 2.29 release notes say "NKI 0.3.0 kernels are not supported on Trn1/Inf2", which is why NxDI dropped them [S36];
  the NKI component itself still lists Inf2/Trn1 support [S2]. Treat NKI on NeuronCore-v2 as second-class.

---

## 3. Minimal component set for a custom engine

### 3.1 What a custom engine must keep

ANALYSIS backed by the cited evidence:

| Component | Why it cannot be dropped | Evidence |
|---|---|---|
| `aws-neuronx-dkms` (driver) | `/dev/neuron*`, DMA rings, HBM mapping; the runtime "requires the Neuron Driver" | [S30]; GPL-2.0 source [S57] (patchable, but runtime/driver versions are coupled: "Neuron Driver 2.29 requires Neuron Runtime Library 2.33 or later" [S3]) |
| `aws-neuronx-runtime-lib` (`libnrt`) | the only documented way to load and run a NEFF; C API in `nrt.h`, `nrt_async.h` | [S29] [S30] |
| `aws-neuronx-collectives` (`libnccom`) | required "for applications that use distributed training or distributed inferences" (any TP across cores/chips); plus EFA installer/libfabric for multi-instance | [S30] |
| `neuronx-cc` | only producer of NEFF from HLO; also the backend for NKI kernels | [S34]; the `nki` wheel's internal `nki/compiler/ncc_driver.py` header: "interface for driving neuronx-cc backend compilation. It handles MLIR -> BIR -> NEFF pipeline"; `_baremetal_driver.py` builds neuronx-cc command lines [S19]; `neuronx-cc` itself `Requires-Dist: nki~=0.6.0b1` [S53] |
| `nki` | to author custom kernels (attention, MoE, sampling) | [S19] [S22] |

Optional: `libneuronxla` (PJRT C-API plugin, if the engine emits StableHLO and wants XLA/PJRT semantics; this is what
ZML uses: `libneuronxla>=3.0`, `neuronx-cc>=2.25`, `nki>=0.4`, debs `aws-neuronx-runtime-lib`,
`aws-neuronx-collectives`, `aws-neuronx-tools` [S60]); `aws-neuronx-tools` (neuron-ls, profiler/Neuron Explorer);
`nrtpy` or Spike for Python-side harnesses.

### 3.2 What can be replaced

NxD Inference (model code, bucketing, KV management, spec decoding) [S42]; NxD Core; transformers-neuronx; the vLLM
Neuron scheduler/model runner [S43]; `libtorch-neuronx-lite` and `torch-neuronx`/`torch-xla` (pure framework glue)
[S51] [S52]; the NKI Library (open source, can be forked) [S24].

### 3.3 Calling NRT directly: documented, yes

- The developer guide "focuses on the information you need to know when building custom frameworks that call `libnrt`
  APIs directly from C/C++ apps" and includes a "first application ... which runs without the aid of an ML framework" [S30].
  Install: `aws-neuronx-runtime-lib` puts `libnrt.so` in `/opt/aws/neuron/lib` and headers in `/opt/aws/neuron/include` [S30]
  (confirmed in the deb file list: `include/nrt/{nrt.h,nrt_async.h,nrt_async_sendrecv.h,nrt_experimental.h,nrt_profile.h,
  nrt_sys_trace.h,nrt_status.h,nrt_version.h,nec.h,ndebug_stream.h,nds/neuron_ds.h}`, `include/ndl/ndl.h` [S55]).
- Core synchronous API (`nrt.h`, `NRT_MAJOR_VERSION 2`) [S29]: `nrt_init(framework, fw_version, fal_version)` (with
  `NRT_FRAMEWORK_TYPE_NO_FW` for framework-less use), `nrt_load(neff_bytes, size, vnc, vnc_count, &model)`,
  `nrt_load_collectives(...)`, `nrt_unload`, `nrt_allocate_tensor_set` / `nrt_add_tensor_to_tensor_set(set, name, tensor)`,
  `nrt_execute(model, inputs, outputs)`, `nrt_execute_repeat`, `nrt_tensor_allocate(placement, vnc, size, name, &t)`,
  `nrt_tensor_read/write(_batch)`, `nrt_tensor_copy`, `nrt_tensor_allocate_slice`, `nrt_tensor_get_va`,
  `nrt_get_dmabuf_fd` (dmabuf export, e.g. for EFA/NIXL), `nrt_get_hbm_mmap_va`, `nrt_get_vnc_memory_stats`,
  `nrt_build_global_comm` / `nrt_cc_global_comm_init`, `nrt_get_attached_efa_bdf`.
- Explicit async API (`nrt_async.h`, marked experimental): execution units `TENSOR_READ/WRITE/OP, COMPUTE, COLLECTIVES`,
  `nrta_execute_schedule`, `nrta_tensor_write/read/copy`, `nrta_cc_prepare/schedule`, `nrta_is_completed`,
  `nrta_get_completion_handle(seq, &fd)` (pollable fd), error trackers [S29]. The implicit async mode
  (`NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS`) was removed in 2.32; "migrate to the explicit async APIs"
  ([S28] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/runtime.html ). `nrt_execute()`
  "is synchronous by default. It's typically required to enable asynchronous mode to achieve high on-device utilization"
  ([S71] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/explore/runtime-performance-tips.html ).
- Point-to-point (`nrt_async_sendrecv.h`): connect/accept by peer IP + LNC, `send_tensor/recv_tensor(offset, length)`
  (usable for KV transfer in disaggregated serving) [S29].
- NEFF container: a 1024-byte header + tarball, unpackable with `neuron-packager` (struct documented:
  `neff_version_major/minor`, `num_tpb`, `uuid`, `lnc_size`, ...)
  ([S32] https://github.com/aws-neuron/aws-neuron-sdk/blob/master/neuron-runtime/explore/work-with-neff-files.rst ;
  note: this page is in the repo but returns 404 on readthedocs, i.e. unpublished). The instruction streams inside are
  UNCONFIRMED as documented.
- NKI kernels compile to standalone NEFFs: set `NKI_ARTIFACTS_DIR`, call the `@nki.jit` kernel with NumPy arrays, get
  `kernel.neff`, then load it with `nrtpy` ([S72] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrtpy/tutorial-debug-neff-nrtpy.html ).
- Precedents: Spike (AWS, experimental) calls `nrt_execute`, `nrta_execute_schedule`, `nrta_tensor_write/read`,
  `nrta_event_register_xu_completion`, `nrt_sys_trace_*` (grep of `spike/src` in [S59]); ZML loads `libneuronpjrt.so`
  from `libneuronxla` behind a proxy PJRT plugin and also links `libnrt` directly [S60].
- Constraints worth designing around: the compiler `-lnc` must equal the runtime `NEURON_LOGICAL_NC_CONFIG` [S14];
  max NCCL communicators (replica groups) per NEFF is 16 [S28]; NEFFs are fixed-shape (section 5).

### 3.4 Three viable engine architectures (ANALYSIS)

1. **HLO/StableHLO graphs + NKI custom calls -> `neuronx-cc` -> NEFF, executed via `libnrt` async API.** Closest to
   what vLLM Neuron does internally (FX -> HLO via `libtorch_neuronx_lite.compile.capture_backend`, `neuronx-cc`
   subprocess [S51] [S52]) but without torch.
2. **PJRT path via `libneuronxla`** (ZML pattern [S60]): least code, but `libneuronxla` is proprietary and adds an
   XLA-shaped abstraction (buffers, executables) between the engine and NRT.
3. **NKI-first megakernels** (e.g. Transformer TKG megakernel in NKI Library [S24]) compiled to NEFF and driven by a
   thin C++/Rust scheduler on `nrta_*`. Highest control; depends on NKI 0.6 dynamic loops for ragged batches.

### 3.5 Licensing constraint (not legal advice)

The runtime and compiler licenses grant use "in connection with AWS Services" and prohibit redistribution, modification
and reverse engineering [S55] [S53] [S54]. A product that bundles these binaries in its own images needs a legal read;
the engine itself can be open source if it links/dlopens the user-installed Neuron packages (as ZML and vLLM do).

---

## 4. NxD Inference and vLLM Neuron, current state

### 4.1 NxD Inference (frozen at 0.10.18399)

- Status: maintenance mode from 2.30, "No new releases are planned" (2.31 announcement dated August 12, 2026), not in
  2.32 DLAMIs/DLCs; Trn2+ only since 2.29 [S36] [S37] [S3]. An "NxDI v2 on torch.compile" plan was announced
  2025-12-19 ([S73] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/announcements/neuron2.x/announce-nxdi-changes.html )
  and superseded by the maintenance decision.
- Feature guide sections ([S39] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/feature-guide.html ):
  HF checkpoint compatibility, persistent compile cache, serialization, LNC, TP, sequence parallelism, on-device
  sampling (greedy, top-k, top-p, temperature, dynamic), QKV fusion, **bucketing (automatic or explicit)**, weight
  quantization (INT8, FP8 `f8e4m3`, per-tensor/per-channel), **KV-cache quantization**, **speculative decoding: draft
  model, Medusa, EAGLE (v1 and v3), fused speculation**, MoE support, GQA, async runtime, **prefix caching** (block KV
  layout), multi-LoRA, disaggregated inference (Beta).
- Continuous batching yes (via vLLM); **chunked prefill: "not supported on Neuron"**; KV cache is contiguous unless
  prefix caching (then blockwise) ([S40] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide-v1.html ).
- Model list (production-ready) ([S41] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/model-reference.html ):
  Llama (text), Llama 4, Mixtral, DBRX, Qwen2.5, Qwen3, Qwen3 MoE (Qwen3-235B-A22B, Beta since 2.27), FLUX.1 (Beta),
  Pixtral-Large-Instruct-2411, Qwen2-VL-7B, Qwen3-VL-8B. Source tree also has `deepseek`, `gpt_oss`, `gemma3`,
  `mllama`, `mistral`, `whisper` model dirs; a Trn3 GPT-OSS-120B tutorial exists; DeepSeek-V3-0324 (671B) is a
  community **contrib** model at TP=64 on trn2.48xlarge [S42] (`contrib/models/DeepSeek-V3/README.md`).
- Known regressions recorded in 2.29: removed BWMM shard-on-hidden prefill kernel (Llama 4, Mixtral, DBRX told to pin
  2.28); Qwen3-MoE 235B decode throughput degraded vs earlier releases [S36].

### 4.2 vLLM Neuron plugin (Beta)

- Repo `vllm-project/vllm-neuron`, Apache-2.0; releases: `v0.24.0.1.1.0` (2026-08-17, Beta), `v0.21.0.1.0.0`
  (2026-07-20, first NxDI-free Beta), legacy NxDI-based `0.5.3` (2026-07-16), `0.5.1` (2026-05-28), `0.5.0`
  (2026-03-19), `0.4.1` (2026-02-26), `0.3.0` (2026-01-23), `0.2.x+lts` (2025-12), `0.2.0` (2025-11-15)
  ([S44] https://github.com/vllm-project/vllm-neuron/releases ). Version string encodes the vLLM version: `0.24.0.*`
  pins `vllm==0.24.0` (`requirements/core.txt`) [S43]. Legacy 0.4.1 tracked vLLM 0.13.0 [S3]; 0.5.1 tracked vLLM 0.16.0
  ([S61] https://github.com/aws-neuron/aws-neuron-sdk/issues/1346 ).
- Instances: Trn2 and Trn3 only [S3] [S64].
- Supported models (tested end-to-end): Llama 3 1B/8B/70B, GPT-OSS 20B/120B, Qwen3-VL 32B, Qwen3-Embedding 8B [S43].
  Not in the list: DeepSeek V3, Qwen3 MoE, Llama 4 (they exist only in frozen NxDI).
- Feature table (README [S43]): continuous batching yes; OpenAI APIs, embeddings, streaming, structured outputs, tool
  calling yes; **LoRA no**; weight reload no; sleep mode no; TP/SP/DP/EP/CP/vision-encoder parallelism yes; **PP no**;
  **paged KV yes; prefix caching yes; segmented prefill yes; chunked prefill (mixed batching) no**; disaggregated
  inference 1P1D/xPyD yes (NIXL); async scheduling yes; on-device sampling yes; **KV offloading no**; disaggregated
  encoder (EPD) yes; `torch.compile` via XLA backend yes, **native PyTorch backend no**; compile cache yes; CPU
  compilation yes; **EAGLE3 yes, MTP no** (vanilla draft-model speculation not supported, migrate to EAGLE3
  ([S74] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/getting-started/migration-nxdi-to-vllm-neuron.html ));
  BF16, FP8, FP8 KV, MXFP4/MXFP8 weights (Trn3); images and video yes, audio no.
- Implementation: the plugin registers `torch.compile` backends from `libtorch_neuronx_lite` (`neuron_libtorch`, and an
  FX-to-HLO `neuron_libtorch_graph_capture`); an opt-in `VLLM_NEURON_BACKEND=neuron_native` route exists; it forces
  vLLM's V1 model runner (`VLLM_USE_V2_MODEL_RUNNER=0`) because its own `NeuronModelRunner` breaks on the V2 path
  ([S51] https://github.com/vllm-project/vllm-neuron/blob/release-0.24.0.1.1.0/vllm_neuron/__init__.py ). Kernels are
  NKI Library kernels wrapped with `libtorch_neuronx_lite.nki.nki_hop.wrap_nki` (attention CTE/decode/segmented,
  QKV, o_proj, MLP, MoE CTE/TKG/blockwise, all_to_all_v, all_gather_v, reduce_scatter_v, cumsum, argsort, topk) [S43].
- 2.32 gains: disaggregated vision encoder (NIXL device-to-device), context parallelism, `/v1/embeddings`, up to 50%
  GPT-OSS and 25% Qwen3-VL-32B throughput over the previous release, accuracy debugger [S3].

---

## 5. Pain points of the managed path a replacement could beat

All quotes are from AWS's own docs unless noted.

| # | Limitation (vendor statement) | Source | Opportunity (ANALYSIS) |
|---|---|---|---|
| 1 | "The Neuron compiler does not yet fully support dynamic shapes. All compiled graphs have fixed tensor dimensions"; bucketing pre-compiles fixed sizes and pads; more buckets = longer startup + more memory | [S46] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/design/vllm/padding-batching-block-kv.html | Use NKI 0.6 runtime loops (`fori_loop`/`while_loop`) and DGE `scalar_offset`/`vector_offset` access patterns so one NEFF handles ragged token counts ([S23], [S63] https://github.com/aws-neuron/aws-neuron-sdk/issues/1328 ) |
| 2 | "Neuron requires prefill and decode operations to run in separate batches. Mixing them in the same batch is not supported"; scheduler raises if violated | [S45] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/design/vllm/neuron-scheduler.html | A ragged mixed-batch attention kernel (the ZML contract in [S61]) enables real chunked prefill / Sarathi-style stall-free batching |
| 3 | "The scheduler currently processes only one prefill request per iteration. This is hardcoded" (`max_prefills_per_batch = 1`); "Decode requests are hidden during each prefill iteration" | [S45] | Batched multi-request prefill; decode never starves (TTFT/ITL tail) |
| 4 | "Preemption is not supported on Neuron", so admission assumes every running request may grow to `max_model_len` ("deliberately conservative ... some KV capacity may go unused") | [S45] | Preemption/swap + optimistic admission raise concurrency |
| 5 | Default KV budget cap `VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION = 0.30` of the GMU budget | ([S50] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/guides/reference-configuration.html ) | Precise memory accounting from `nrt_get_vnc_memory_stats` [S29] |
| 6 | Decode NEFF reads `ceil(max_model_len / block_size)` KV blocks per request "regardless of the request's actual position"; "~32x DMA overhead amplifier" for 4K live KV at 131072 max; mitigation is an opt-in second bucket dimension | [S47] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/design/vllm/decode-context-length-bucketing.html | Length-bounded paged decode (stream only live pages): decode is HBM-bound, so this is the single biggest throughput lever |
| 7 | NxDI prefix caching "does not use paged attention": gather block KV -> flat -> attention -> scatter back, overhead grows with `max_model_len` | [S39] | True paged attention over the block table |
| 8 | Prefix cache is vLLM APC ("Prefix caching (APC)"), block-aligned, `kv_segment_size` in {512, 1024, 2048, 4096}; no radix-tree cache is documented in any Neuron path (absence UNCONFIRMED beyond docs) | [S43] model recipe; ([S49] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/design/vllm/prefix-caching.html ) | Radix prefix cache + cache-aware routing |
| 9 | Chunked prefill "not supported on Neuron" (NxDI) / "Chunked prefill (mixed batching)" not supported (vLLM Neuron) | [S40] [S43] | as #2 |
| 10 | Async overlap (scheduling step N+1 while N runs) "breaks" and falls back to synchronous whenever the batch composition, padded shape/dtype, or request order changes | ([S48] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/design/vllm/async-scheduling-and-async-execution.html ) | Fixed-slot batch layouts + on-device input construction keep overlap on through churn |
| 11 | Compile time: "a small-ish model of 15GB might take around 15min to compile" (NxDI); "cold compile of a 32B engine can exceed 10 minutes" (vLLM Neuron); fixes listed are fewer buckets, cache, parallel compile workers | [S40]; ([S75] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/tutorials/tutorial-epd-1e-1pd-xeypd.html ); ([S50b] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/guides/features-guide.html ) | Fewer, shape-polymorphic NEFFs; per-layer NEFF reuse; AOT artifact cache |
| 12 | Large NKI kernels: "long compile times (~hours) and high memory usage (40GB+)" in the compiler; AWS root-caused to a default-on pass, fix "will land in the upcoming release of NKI" | ([S62] https://github.com/aws-neuron/aws-neuron-sdk/issues/1374 ) | Keep kernels modest or pin compiler options; budget CI compile time |
| 13 | Nested runtime loops (dynamic Q-tile loop around dynamic KV loop) failed with `NCC_ICFG018 ... do not support ... nested loops` through NKI 0.5.0; compiles on NKI 0.6.0 | [S61] | Ragged paged attention is now expressible but freshly so; expect compiler edge cases |
| 14 | No PP, no LoRA, no MTP, no KV offload in vLLM Neuron; draft-model speculation removed | [S43] [S74] | Feature parity gaps to fill |
| 15 | Trn1/Inf2: no maintained LLM serving path; legacy plugin silently corrupted all but one sequence with on-device sampling at `max_num_seqs > 1` (HTTP 200, garbage text) | [S64] [S36] [S38] | A custom engine is the only forward path on Inf2/Trn1 |
| 16 | `torch.compile` on TorchNeuron: no CUDA-graph equivalent "in the initial release"; compilation requires Trn2/Trn3 hardware; still closed beta | [S26] | Engine-owned NEFF graphs avoid waiting on TorchNeuron |
| 17 | Compiler and runtime LNC must match; "AWS Neuron currently doesn't support setting the compiler flag to a different LNC configuration than the Neuron Runtime environment variable" | [S14] | Compile per-LNC artifact sets |

---

## 6. NKI programming model essentials (for a TileLang-style DSL)

### 6.1 Memory hierarchy and tile limits

- Per NeuronCore: **SBUF** (state buffer) 24 / 28 / 32 MiB on NCv2 / NCv3 / NCv4, organized as **128 partitions**
  (192 / 224 / 256 KiB per partition); usable per partition 180,224 / 212,984 / 245,752 bytes (16 KiB reserved, plus 8
  bytes on gen3+) [S19] (`nki/language/_tile_size.py`), [S15] [S16] [S17].
- **PSUM** (partial-sum buffer, TensorE output + near-memory accumulation): 2 MiB per core on all generations = 128
  partitions x 16 KiB, **8 banks** per partition of 2 KiB = 512 FP32 each (7 usable if DMA transpose is lowered to PE
  transpose) [S19] [S15].
- `nl.tile_size` constants ([S18] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/nki.language.tile_size.html , values from [S19]):
  `pmax = 128` (partition dim), `gemm_stationary_fmax = 128`, `gemm_moving_fmax = 512`, `psum_bank_fmax = 512` FP32
  (2,048 bytes), `bn_stats_fmax = 512`, `psum_min_align = sbuf_min_align = 4` bytes, max PSUM free dim 4,096.
- Matmul contract: `nc_matmul(stationary[K, M], moving[K, N]) = stationary.T @ moving -> PSUM[M, N]`, contraction K on
  the **partition** axis (<= 128), M <= 128 (PE columns), N <= 512 (one PSUM bank) [S15]. Tiling over K accumulates
  into the same PSUM tile.
  NCv3 double-row FP8: stationary `[128, 2, 128]`, moving `[128, 2, 512]` (K = 256) [S16]. NCv4 MX: K = 512, max
  stationary `[128,128]` x4-packed (= `[128,512]` values), moving `[128,512]` x4 (= `[128,2048]` values), plus two
  scale tensors (one E8M0 per 32 elements along K, laid out across SBUF quadrants) via `nc_matmul_mx` [S17].
- Vector/Scalar instructions: partition dim <= 128, free dim up to 64K elements from SBUF or 4K from PSUM [S15].

### 6.2 Engines (each has its own instruction stream; synchronization by semaphores)

"Each compute engine has its own sequencer ... The four compute engines execute four independent instruction streams
asynchronously in parallel", plus a Sync Engine commonly used to trigger DMAs [S15].

| Engine | Function | Width / clock |
|---|---|---|
| Tensor (PE) | 128x128 systolic array, GEMM/conv/transpose; LoadStationary + MultiplyMoving | NCv2 2x128 in / 1x128 out @ 2.8 GHz [S15]; NCv3 4x128 (FP8), 2x128 (BF16) @ 2.4 GHz [S16]; NCv4 8x128 (MXFP8), 2x128 (non-MX) @ 2.4 GHz [S17] |
| Vector (DVE) | multi-input ops, reductions, BN stats, cross-partition shuffles within 32-partition groups, 32x32 transpose; NCv4 adds QuantizeMX and fast `exponential` (4x ScalarE `exp`) | NCv2 128 lanes @ 1.12 GHz [S15]; NCv3 512 BF16/cycle @ 0.96 GHz [S16]; NCv4 512 BF16/FP8 @ 1.2 GHz [S17] |
| Scalar (ACT) | per-element nonlinearities with fused `func(x*scale + bias)` and pipelined add-reduce (`activation_reduce`) | NCv2 128 @ 1.4 GHz [S15]; NCv3 @ 1.2 [S16]; NCv4 256 BF16/FP8 @ 1.2 [S17] |
| GpSimd (POOL) | 8 programmable 512-bit vector cores running C/C++, 64 KB local RAM each; software DGE; iota, masks, `topk` (NKI 0.6) | NCv2 @ 1.4 GHz; NCv3/v4 @ 1.2 GHz [S15] [S16] [S17] |
| DMA | 16 engines per core; per-core aggregate ~272 / 368 / 528 GB/s (TRN1/2/3); payloads >= 2 KiB per partition minimum, >= 4 KiB to saturate | ([S20] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki-dma-bandwidth-guide.html ); the older Trn1 guide says 27 GiB/s per engine [S15] (inconsistent with 17 B/ns in [S20]) |

Cost-model rules of thumb [S15]: back-to-back MM issue interval ~`max(N, 64)` TensorE cycles on NCv2 (N = moving free
size, typically 512); FP32 matmul ~4x slower than BF16/FP8; LoadStationary up to 4x faster than MultiplyMoving and
overlaps in the background, so put the larger-free-dim operand as stationary for GEMV-like shapes; VectorE ~N cycles
(one input) or ~2N (two inputs) plus ~100-cycle fixed overhead for tiny dependent instructions; VectorE and GpSimd
cannot access SBUF in parallel.

DMA descriptor generation ([S21] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki-dge.html ):
`none` (host-precomputed descriptors in HBM; static only), `swdge` (GpSimd generates at runtime; **the only mode with
indirect gather/scatter**, needed for paged-KV block-table loads), `hwdge` (dedicated block, two per TRN2 core, ~600 ns
per DMA instruction, triggered from Scalar or Sync engine, no indirect).

Multi-core: kernels launch on 1 or 2 cores with `kernel[2](...)` and use `nl.num_programs()` / `nl.program_id()`; LNC=2
cores share HBM; `nki.collectives` provides all_reduce/all_gather/reduce_scatter/all_to_all(_v)/permute/all_gather_v
([S76] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/lnc.html , [S3] [S23]).

Control flow: `affine_range`, `sequential_range`, `static_range` (compile-time bounds); runtime loops are hardware
loop instructions (body not unrolled), `step` must be a compile-time constant
([S77] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki-dynamic-loops.html ); as of 0.6.0 use
`fori_loop`/`while_loop` [S23].

### 6.3 How a TileLang-style DSL maps (ANALYSIS)

- Good fits: `alloc_shared` -> SBUF tile `[P<=128, F]`; `alloc_fragment`/accumulator -> PSUM bank(s) (2 KiB/partition
  per bank, 8 banks, so an accumulator is at most 128 x 4,096 FP32 per core); `T.gemm` -> `nc_matmul` /
  `nc_matmul_mx` with an explicit stationary/moving choice; `T.copy` -> `dma_copy` with a DGE-mode decision;
  `T.Pipelined` -> double/triple buffering across DMA queues and engines (the hardware is natively decoupled
  producer/consumer, like warp-specialized Hopper kernels but with 4 heterogeneous engines instead of warps).
- Mismatches a lowering must handle: no threads/warps or shared-memory banks; layout is partition-major and the
  contraction dim must sit on partitions (so transposes are explicit, via TensorE identity matmul, NCv3+ transpose mode,
  VectorE 32x32, or DMA transpose); elementwise ops run on 128 lanes, so small-partition ops need "partition
  vectorization"; cross-partition reductions go through TensorE or 32-partition VectorE groups; softmax exp should go
  to VectorE `exponential` on Trn3 and ScalarE `activation(_reduce)` elsewhere; engine selection is a scheduling
  decision the DSL must expose or infer.
- Lowering targets: (a) emit NKI Python (`nki.isa` level), the stable documented API; (b) emit MLIR in the NISA
  dialect, as AWS's experimental `nkigen` does (NumPy trace -> linalg -> 26 passes -> NISA, with `knob` tiling/layout
  annotations) [S59]. The NISA dialect bindings live in the `nki` wheel under `nki/compiler/_internal` (marked
  internal) [S19], so (b) is unsupported API surface.

---

## 7. Inconsistencies found between AWS sources (worth knowing)

1. Compiler version: index table 2.27.5334.0 vs compiler page header "2.27.4747.0" [S1] [S33].
2. JAX: "supports up to JAX 0.9.0" (index) vs max 0.10.0 (component page) [S1] [S35].
3. Trn2 CC-cores 16 [S6] vs 20 [S16]; Trn2 HBM BW 2.9 TB/s [S6] vs 3 TB/s [S16]; Trn3 HBM BW 4.9 TB/s [S7] vs 4.7 TB/s [S17].
4. DMA engine bandwidth on NCv2: 27 GiB/s per engine [S15] vs 17 B/ns (TRN1) [S20].
5. NKI on Trn1/Inf2: NKI component "Supports: Inf2, Trn1, Trn1n" [S2] vs NxDI 2.29 note "NKI 0.3.0 kernels are not supported on Trn1/Inf2" [S36].
6. The NEFF file-format page exists in the docs repo but is not published (readthedocs 404) [S32].

## 8. UNCONFIRMED items

- Public Trn3 instance-type names, pricing, regions, GA date (only driver-source strings and "Trn3pds" in the EKS doc).
- Default LNC on Trn3.
- Whether NKI compiler C++ sources, `nrtpy` sources, `libnccom` sources, or a NEFF/ISA encoding spec are public.
- TorchNeuron public availability date (still closed beta; repo 404).
- Whether any Neuron path implements a radix-tree prefix cache (none documented).
- vLLM Neuron absolute throughput vs NxDI or GPUs (only relative release-over-release numbers are published [S3]).

## 9. Sources

- [S1] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/index.html
- [S2] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/2.32.0.html
- [S3] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/whats-new.html (source: https://github.com/aws-neuron/aws-neuron-sdk/blob/master/about-neuron/whats-new.rst , modified 09/30/2026)
- [S4] https://github.com/aws-neuron/aws-neuron-sdk (HEAD a6be96661f3d, 2026-09-30)
- [S5]-[S12] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/{trainium,trainium2,trainium3,inferentia2,trn1-arch,trn2-arch,trn3-arch,inf2-arch}.html
- [S13v2/v3/v4] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v{2,3,4}.html
- [S14] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/logical-neuroncore-config.html
- [S15] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium_inferentia2_arch.html
- [S16] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html
- [S17] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium3_arch.html
- [S18] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/nki.language.tile_size.html
- [S19] https://pip.repos.neuron.amazonaws.com/nki/ (wheel `nki-0.6.0+31049202112.g85070674-cp312`, files `nki/language/_tile_size.py`, `nki/language/__init__.pyi`, `nki/compiler/ncc_driver.py`, `nki/compiler/_baremetal_driver.py`, METADATA)
- [S20] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki-dma-bandwidth-guide.html
- [S21] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki-dge.html
- [S22] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki-compiler.html
- [S23] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/nki.html
- [S24] https://github.com/aws-neuron/nki-library (README; Apache-2.0)
- [S25] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/oss/index.html
- [S26] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html
- [S27] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/pytorch.html
- [S28] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/runtime.html
- [S29] https://github.com/aws-neuron/aws-neuron-sdk/tree/master/src/libnrt (nrt.h, nrt_async.h, nrt_experimental.h, nrt_async_sendrecv.h, nec.h); rendered: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/api/nrt.html
- [S30] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/guides/nrt-developer-guide.html
- [S31] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrtpy/index.html ; wheel https://pip.repos.neuron.amazonaws.com/nrtpy/
- [S32] https://github.com/aws-neuron/aws-neuron-sdk/blob/master/neuron-runtime/explore/work-with-neff-files.rst
- [S33] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/compiler.html
- [S34] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/index.html
- [S35] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/jax/index.html ; https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/jax.html
- [S36] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/components/nxd-inference.html
- [S37] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/announcements/neuron2.x/announce-nxdi-maintenance-support-2-31.html ; https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/announcements/neuron2.x/announce-maintenance-nxdi-nxd-core-inference.html
- [S38] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/announcements/neuron2.x/announce-maintenance-pytorch-xla-inference.html
- [S39] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/feature-guide.html
- [S40] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide-v1.html
- [S41] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/model-reference.html
- [S42] https://github.com/aws-neuron/neuronx-distributed-inference (src/neuronx_distributed_inference/models, contrib/models/DeepSeek-V3/README.md)
- [S43] https://github.com/vllm-project/vllm-neuron/tree/release-0.24.0.1.1.0 (README, requirements/core.txt, docs/model-recipes/gpt-oss.md)
- [S44] https://github.com/vllm-project/vllm-neuron/releases
- [S45] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/design/vllm/neuron-scheduler.html
- [S46] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/design/vllm/padding-batching-block-kv.html
- [S47] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/design/vllm/decode-context-length-bucketing.html
- [S48] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/design/vllm/async-scheduling-and-async-execution.html
- [S49] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/design/vllm/prefix-caching.html
- [S50] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/guides/reference-configuration.html ; [S50b] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/guides/features-guide.html
- [S51] https://github.com/vllm-project/vllm-neuron/blob/release-0.24.0.1.1.0/vllm_neuron/__init__.py ; .../vllm_neuron/envs.py
- [S52] https://pip.repos.neuron.amazonaws.com/libtorch-neuronx-lite/ (wheel 2.13.0.1.0.2651+723ba691 METADATA; `_compiler/kernels/neuronx_cc_wrapper.py`)
- [S53] https://pip.repos.neuron.amazonaws.com/neuronx-cc/ (wheel 2.27.5334.0+f702b353 METADATA, read via HTTP range)
- [S54] https://pip.repos.neuron.amazonaws.com/libneuronxla/ (wheel 3.0.5356.0+c743c3ec, `libneuronxla/LICENSE.txt`)
- [S55] https://apt.repos.neuron.amazonaws.com/pool/main/a/aws-neuronx-runtime-lib/aws-neuronx-runtime-lib_2.34.10.0-ac18d186d_amd64.deb
- [S56] https://apt.repos.neuron.amazonaws.com/pool/main/a/aws-neuronx-collectives/aws-neuronx-collectives_2.34.10.0-74eaafac6_amd64.deb
- [S57] https://github.com/aws-neuron/aws-neuron-driver (GPL-2.0; v4/neuron_dhal_v4.c)
- [S58] https://github.com/orgs/aws-neuron/repositories (listed via `gh api orgs/aws-neuron/repos` on 2026-10-01)
- [S59] https://github.com/aws-neuron/nkipy (README, spike/, nkigen/README.md)
- [S60] https://github.com/zml/zml/tree/master/platforms/neuron (BUILD.bazel, requirements.in, packages.yaml, neuron.zig)
- [S61] https://github.com/aws-neuron/aws-neuron-sdk/issues/1346
- [S62] https://github.com/aws-neuron/aws-neuron-sdk/issues/1374
- [S63] https://github.com/aws-neuron/aws-neuron-sdk/issues/1328
- [S64] https://github.com/vllm-project/vllm-neuron/issues/57
- [S65] https://github.com/aws-neuron/upstreaming-to-vllm (archived)
- [S66] https://aws.amazon.com/ec2/instance-types/trn3/
- [S67] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/announcements/neuron2.x/announce-intent-eos-tnx.html
- [S68] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/deploy/eks/ultraserver-operator.html
- [S69] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-features/data-types.html
- [S70] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/attention-tkg.html
- [S71] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/explore/runtime-performance-tips.html
- [S72] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrtpy/tutorial-debug-neff-nrtpy.html
- [S73] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/announcements/neuron2.x/announce-nxdi-changes.html
- [S74] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/getting-started/migration-nxdi-to-vllm-neuron.html
- [S75] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/vllm-neuron/docs/tutorials/tutorial-epd-1e-1pd-xeypd.html
- [S76] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/about/lnc.html
- [S77] https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/deep-dives/nki-dynamic-loops.html

## 10. How to re-verify (commands used)

```sh
R=<local-dir>/research
git clone --depth 1 https://github.com/aws-neuron/aws-neuron-sdk.git $R/aws-neuron-sdk      # docs + libnrt headers
git clone --depth 1 -b release-0.24.0.1.1.0 https://github.com/vllm-project/vllm-neuron.git  # plugin
gh api --paginate "orgs/aws-neuron/repos?per_page=100" -q '.[]|[.name,.license.spdx_id,.pushed_at,.archived]|@tsv'
gh api repos/aws-neuron/torch-neuronx            # 404 on 2026-10-01 (closed beta)
curl -s https://pip.repos.neuron.amazonaws.com/nki/ | grep -o 'nki-[^"]*whl' | tail -2
python3 $R/remote_zip.py "https://pip.repos.neuron.amazonaws.com/neuronx-cc/neuronx_cc-2.27.5334.0%2Bf702b353-cp312-cp312-linux_x86_64.whl"   # License line via HTTP range reads
curl -s https://apt.repos.neuron.amazonaws.com/dists/jammy/main/binary-amd64/Packages | grep -A3 '^Package: aws-neuronx-runtime-lib'
```

Local copies of everything inspected are under `<local-dir>/research/` (repos, `wheels/`
with the extracted `nki`, `libtorch-neuronx-lite`, `nrtpy` wheels and the runtime/collectives debs).
