# Open-weight model targets for a Trainium/Inferentia inference engine

Reading date: **2026-10-01**. All numbers fetched live on that date. Raw captures (HTML, config.json,
safetensors headers, API JSON) are in `<local-dir>/research/mw/`.

Conventions:
- `cfg:` = `https://huggingface.co/<repo>/raw/main/config.json`
- `card:` = `https://huggingface.co/<repo>/raw/main/README.md`
- `api:` = `https://huggingface.co/api/models/<repo>?expand[]=safetensors` (param counts, license tag)
- `tree:` = `https://huggingface.co/api/models/<repo>/tree/main?recursive=true` (checkpoint bytes = sum of `*.safetensors`, decimal GB)
- `hdr:` = safetensors header of one shard, read by HTTP range request (`mw/sthdr.py`), used to confirm on-disk dtypes
- **UNCONFIRMED** = not stated by a primary source; my inference or estimate.

---

## 1. Artificial Analysis Intelligence Index - open weights, current

Source: `https://artificialanalysis.ai/leaderboards/models` - the page embeds the full model table as JSON
(689 models; fields `intelligenceIndex`, `isOpenWeights`, `deprecated`, `intelligenceIndexIsEstimated`).
Filtered to `isOpenWeights=true`, `deprecated=false`, best reasoning-effort variant per model.
The page labels the metric **"Artificial Analysis Intelligence Index v4.3"**. For scale: the top closed
model on the same page is Claude Opus 5.5 (Max) at 57.6; the best open model ranks 25th overall.

| # | Model (AA name) | AA II v4.3 | Creator | AA slug | HF repo used below |
|---|---|---|---|---|---|
| 1 | MiMo-V2.6-Pro | 46.3 | Xiaomi | mimo-v2-6-pro | XiaomiMiMo/MiMo-V2.6-Pro-RL (also -MOPD) |
| 2 | GLM-5.3 (Max) | 44.8 | Z AI | glm-5-3 | zai-org/GLM-5.3 |
| 3 | Kimi K3 (Max) | 43.6 | Moonshot | kimi-k3 | moonshotai/Kimi-K3 |
| 4 | GLM 5.3 Flash | 41.8 | Z AI | glm-5-3-flash | zai-org/GLM-5.3-Flash |
| 5 | Qwen3.8 2.4T A95B | 39.9 | Alibaba | qwen3-8-2-4t-a95b | Qwen/Qwen3.8-2.4T-A95B |
| 6 | Qwen3.8-Flash-Next | 39.8 | Alibaba | qwen3-8-flash-next | Qwen/Qwen3.8-Flash-Next |
| 7 | DeepSeek V4.1 Flash (Max) | 39.5 | DeepSeek | deepseek-v4-1-flash | deepseek-ai/DeepSeek-V4.1-Flash |
| 8 | MiMo-V2.6-Flash | 37.9 | Xiaomi | mimo-v2-6-flash | XiaomiMiMo/MiMo-V2.6-Flash-RL |
| 9 | DeepSeek V4 Pro 0813 (Max) | 36.0 | DeepSeek | deepseek-v4-pro | deepseek-ai/DeepSeek-V4-Pro-0813 |
| 10 | Qwen3.8 27B (Xhigh) | 33.7 | Alibaba | qwen3-8-27b | Qwen/Qwen3.8-27B |
| 11 | Motif 3 | 33.6 (AA-estimated) | Motif Technologies | motif-3 | Motif-Technologies/Motif-3 |
| 12 | K2 Horizon 375B A23B | 30.5 | Inst. of Foundation Models | k2-horizon-375b-a23b | IFM/K2-Horizon-375B-A23B |
| 13 | MiniMax-M3 | 29.2 | MiniMax | minimax-m3 | MiniMaxAI/MiniMax-M3 |
| - | Nex-N2-Pro (fine-tune of Qwen3.5-397B) | 28.2 (est) | Nex AGI | nex-n2-pro | excluded: derivative |
| 14 | Kimi K2.7 Code | 25.8 | Moonshot | kimi-k2-7-code | moonshotai/Kimi-K2.7-Code |
| 15 | Inkling Small | 25.7 | Thinking Machines | inkling-small | thinkingmachines/Inkling-Small |
| 16 | Hy3 | 25.3 | Tencent | hy3 | tencent/Hy3 |
| 17 | Inkling (Xhigh) | 25.0 | Thinking Machines | inkling | thinkingmachines/Inkling |
| 18 | Nemotron 3 Ultra 550B A55B | 22.9 | NVIDIA | nvidia-nemotron-3-ultra-550b-a55b | nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 |

The families named in the brief, for the record (same source): Mistral Medium 3.5 = 14.2,
gpt-oss-120b (High) = 11.6, Llama 4 Maverick = 10.0 (AA-estimated). None of them is near the top 12
(the cut-off is about 29).

AA does not say which HF checkpoint it evaluated (it calls APIs). For MiMo-V2.6-Pro, HF has `-RL`
(2026-09-21) and `-MOPD` (2026-09-27, card: "the MOPD upgrade of the MiMo-V2.6-Pro-RL checkpoint").
Both have identical config and size. Which one matches the AA score is **UNCONFIRMED**.

### 1.1 Architecture cards (top 13)

Each line carries its source. "HF params" = HF safetensors parameter count (it un-packs 4-bit storage, so
it equals the logical parameter count; verified on gpt-oss-120b: 116.8B vs the card's 117B).

**1. MiMo-V2.6-Pro** - repo `XiaomiMiMo/MiMo-V2.6-Pro-RL`, license **MIT** (api)
- Params: 1.02T total / 42B active (card L80); HF params 1,024,216,603,392 (api)
- 70 layers = 60 SWA + 10 global attention, first block global and dense FFN (card L130, L140; cfg `hybrid_layer_pattern`)
- hidden 6144; dense intermediate 16384; MoE intermediate 2048 (cfg)
- Attention: GQA, 128 Q heads / 8 KV heads, K head_dim 192 / V 128, sliding window 128 with attention-sink bias, partial RoPE 0.334 (cfg)
- MoE: 384 routed experts, top-8, **no shared expert**, sigmoid noaux_tc routing (cfg; card L136)
- Context 1,048,576 (cfg `max_position_embeddings`; card "1M")
- MTP: 5-layer SWA DFlash drafter, predicts 7 tokens per pass (card L85, L161)
- Native precision: **experts MXFP4** (U8 packed + U8 E8M0 scale per 32), dense MLP and fused QKV **FP8 E4M3 block 128x128**, o_proj / embeddings / lm_head BF16 (hdr shard `model_pp0_ep0_shard0.safetensors`; cfg `quantization_config.store_dtype=mxfp4`)
- Checkpoint 573.5 GB = 566.0 backbone + 5.5 `dflash/` + 1.9 `audio_tokenizer/` (tree)
- Omni-modal (681M ViT, audio encoder) - text path can ignore them (card L83-84)
- Same architecture class (`MiMoV2ForCausalLM`, identical layer/head/expert counts) as MiMo-V2.5-Pro, which shipped as FP8 at 1,033 GB (cfg + tree of `XiaomiMiMo/MiMo-V2.5-Pro`)

**2. GLM-5.3** - `zai-org/GLM-5.3`, license **glm-5.3** (custom, MIT-style; MaaS operators with >US$10B revenue must pass a Z.AI security review: `https://huggingface.co/zai-org/GLM-5.3/raw/main/LICENSE`)
- Params: 744B total / 40B active (GLM-5 card `https://huggingface.co/zai-org/GLM-5/raw/main/README.md` L32; GLM-5.3 card L13: "uses the same base model as GLM-5.2"); HF params 753.3B incl. MTP (api)
- 78 layers, first 3 dense (cfg `first_k_dense_replace=3`)
- hidden 6144; dense intermediate 12288; MoE intermediate 2048 (cfg)
- Attention: **MLA + DeepSeek Sparse Attention (DSA)** - 64 heads, q_lora 2048, kv_lora 512, qk_nope 192, qk_rope 64, v 256; indexer 32 heads x 128, top-2048; IndexShare: 21 `full` + 57 `shared` indexers (cfg)
- MoE: 256 routed, top-8, + 1 shared (cfg)
- Context 1,048,576 (cfg)
- Native precision: **FP8 E4M3, block 128x128** (F32 scale_inv), embeddings/lm_head BF16 (cfg; hdr)
- Checkpoint 755.6 GB (tree). Community NVFP4: `RadixArk/GLM-5.3-NVFP4` 464.8 GB (tree)

**3. Kimi K3** - `moonshotai/Kimi-K3`, license **Kimi K3** (`.../Kimi-K3/raw/main/LICENSE`: a MaaS business with >US$20M revenue over 12 months must sign a separate agreement; >100M MAU or >US$20M/month revenue must display "Kimi K3")
- Params: 2.8T total / 104B active (card table L58-62); HF params 2.78T (api)
- 93 layers, 1 dense; **69 KDA (Kimi Delta Attention, linear) + 24 Gated MLA** (card L75; cfg `linear_attn_config.kda_layers` 69 entries, `full_attn_layers` 24)
- hidden 7168; 96 heads; MLA q_lora 1536, kv_lora 512, qk_nope 128, qk_rope 64, v 128; KDA head_dim 128, short conv 4 (cfg)
- MoE: **896 experts, 16 active + 2 shared**, LatentMoE dim 3584, expert hidden 3072; Attention Residuals (card L43, table; cfg `num_experts_per_token=16`)
- Context 1,048,576 (card table; cfg)
- Native precision: **MXFP4 weights / MXFP8 activations, QAT** (card L131, L611-613); hdr: routed experts U8 `weight_packed` + U8 scale (compressed-tensors `mxfp4-pack-quantized`, group 32), attention and shared experts BF16
- Checkpoint 1,560.9 GB (tree)

**4. GLM-5.3-Flash** - `zai-org/GLM-5.3-Flash`, license **MIT** (api)
- Params: 320B total / 18B active (card L25); HF params 321.3B (api)
- 45 layers = **34 KDA linear-attention + 11 DSA** (MLA-based, NoPE: `qk_rope_head_dim=0`, kv_lora 512, q_lora 1536, 64 heads, qk/v head 256; indexer 32x128 top-2048 with kpool 4) (cfg `layer_types`, `linear_attn_config`)
- hidden 4096; 3 dense layers (12288); MoE 288 routed, top-8, + 1 shared, intermediate 2048 (cfg)
- Manifold-constrained hyper-connections, `hc_mult=4` (card L27; cfg)
- Context 1,048,576 (cfg)
- Native precision: FP8 E4M3 block 128 (cfg); BF16 also published as `zai-org/GLM-5.3-Flash-BF16`
- Checkpoint FP8 328.3 GB; BF16 642.7 GB; `nvidia/GLM-5.3-Flash-NVFP4` 204.4 GB (tree)

**5. Qwen3.8-2.4T-A95B** - `Qwen/Qwen3.8-2.4T-A95B`, license **qwen3.8-max** (`.../LICENSE`: a MaaS / AI-work-assistant business with >US$50M revenue over 12 months needs a separate license)
- Params: 2.4T total / 95B active (card L46); HF params 2.446T (api)
- 92 layers = 23 x (3 x Gated DeltaNet + 1 x Gated Attention) (card L49-50; cfg `layer_types` 69/23)
- hidden 8192; Gated Attention 64 Q / 4 KV heads, head_dim 256; GDN 16 QK heads / 128 V heads, dims 128 (cfg; card L55)
- MoE: 512 experts, 10 routed + 1 shared, intermediate 2048 (card L59-60; cfg)
- Context 262,144 native, extensible to 1,010,000 (card L64)
- Native precision BF16, 4,892.4 GB; official FP8 (block 128) `Qwen/Qwen3.8-2.4T-A95B-FP8` 2,496.1 GB; `amd/...-Quark-MXFP4` 1,372.4 GB; `nvidia/...-NVFP4` 1,444.5 GB (tree)

**6. Qwen3.8-Flash-Next** - `Qwen/Qwen3.8-Flash-Next`, license **qwen-community-1.0** (`.../LICENSE` clause 2: **any** MaaS or AI-work-assistant business must obtain a separate license before commercial use - no revenue threshold)
- Params: 125B + 51B n-gram embedding + 4B MTP, 6B active (card L46); HF params 180.0B (api)
- 48 layers = 12 x (3 x Gated DeltaNet + 1 x **Qwen Sparse Attention** (block-level sparse)) (card L32, L51; cfg)
- hidden 2560; attention 24 Q / 2 KV heads, head 256; GDN 16 QK / 48 V heads x 128; QSA indexer 4 heads x 128, budget 2048, compress 4; hyper-connections `hc_count=4`; n-gram embedding (ngram 3) (cfg)
- MoE: 512 experts, 10 routed + 1 shared, intermediate 640 (card L63-64; cfg)
- Context 262,144 native, extensible to 1,000,000 (card L71)
- Native BF16 360.0 GB; official FP8 (block 128) 185.5 GB (tree)

**7. DeepSeek-V4.1-Flash** - `deepseek-ai/DeepSeek-V4.1-Flash`, license **MIT** (api)
- Params: 552B backbone + 196B Engram conditional memory (lookup-only); **8B active in prefill / 16B in decode** (card L45-51, L77-78); HF params 763.2B (api)
- 40 layers as a **Causal Encoder-Decoder** (20 encoder + 20 decoder); decoder global KV projected from encoder output (card L47)
- hidden 5120; 64 heads, **1 KV head, head_dim 512**, rope 64, q_lora 1280, o_lora 1024 (8 groups) (cfg)
- Attention: **Compressed Sparse Attention 2** (Full / Reindex / Reuse modes, hierarchical sparse indexer 32x128 top-512), sliding window 128, FP4 main KV cache = **890 bytes/token** (card L49; cfg)
- Single-Pass mHC (`hc_mult=4`), DSpark speculative decoding, 3 MTP layers (card L51; cfg)
- MoE: 384 routed, top-6, + 1 shared, intermediate 2304 (card L51; cfg)
- Context 1,048,576 (cfg; card)
- Native precision: **experts FP4 E2M1** (packed in I8, UE8M0 scale per 32), attention/shared experts **FP8 E4M3 block 32 UE8M0**, Engram tables FP8 (hdr; cfg `expert_dtype=fp4`)
- Checkpoint 510.3 GB (tree), of which ~204 GB is the FP8 Engram table (api dtype breakdown F8_E4M3 204.0B params)

**8. MiMo-V2.6-Flash** - `XiaomiMiMo/MiMo-V2.6-Flash-RL`, license **MIT**
- Params: 309B total / 15B active (card L80); HF 310.8B (api)
- 48 layers = 39 SWA + 9 global (card L130; cfg pattern 39/9); hidden 4096; 64 Q heads, global 4 KV / SWA 8 KV, head 192/128, window 128 (cfg)
- MoE: 256 routed, top-8, no shared, intermediate 2048 (card L136; cfg)
- Context 1M (card L81; cfg 1,048,576)
- Native: MXFP4 experts + FP8 (cfg `store_dtype=mxfp4`); checkpoint 177.7 GB = 172.9 backbone + 2.9 dflash + 1.9 audio (tree)

**9. DeepSeek-V4-Pro-0813** - `deepseek-ai/DeepSeek-V4-Pro-0813`, license **MIT**
- Params: 1.6T / 49B active (V4.1-Flash card base-model table L77-78, "DeepSeek-V4-Pro-Base"); HF 1.650T (api)
- 61 layers; hidden 7168; 128 heads, 1 KV head, head_dim 512, rope 64, q_lora 1536, o_lora 1024 (16 groups); compressed attention `compress_ratios` = {128: 31, 4: 30, 0: 3}, indexer 64x128 top-1024, sliding window 128; mHC `hc_mult=4`; 1 MTP layer (cfg)
- MoE: 384 routed, top-6, + 1 shared, intermediate 3072 (cfg)
- Context 1,048,576 (cfg, YaRN x16 from 65,536)
- Native: FP4 experts (I8-packed, UE8M0) + FP8 E4M3 block 128 UE8M0 (hdr; cfg); checkpoint 892.7 GB (tree)

**10. Qwen3.8-27B** - `Qwen/Qwen3.8-27B`, license **Apache-2.0**
- 27B dense (card L37); HF 27.78B incl. vision tower (api)
- 64 layers = 16 x (3 x Gated DeltaNet + 1 x Gated Attention) (card L40-41); hidden 5120; FFN 17408; attention 24 Q / 4 KV, head 256; GDN 16 QK / 48 V heads x 128 (cfg)
- Context 262,144 native, extensible to 1,000,000 (card L53)
- BF16 55.6 GB; official FP8 `Qwen/Qwen3.8-27B-FP8` 30.9 GB (tree)

**11. Motif-3** - `Motif-Technologies/Motif-3`, license **MIT** (api). AA score is AA-estimated.
- ~314B total / 13.2B active (card L34, L53-54)
- 53 layers (2 dense + 51 MoE) (card L55); hidden 4096 (cfg)
- Attention: **Grouped Differential Latent Attention** (differential attention over an MLA latent): 80 heads, 16 KV + 16 noise heads, head 192/128, q_lora 1024, kv_lora 512, rope 64; interleaved SWA window 128, period 4 (card L36, L58; cfg)
- MoE: 384 routed, top-8, + 1 shared, intermediate 1280; modified mHC (card L41, L60-63; cfg)
- Context 262,144 (card L65)
- BF16 629.7 GB; official `Motif-3-NVFP4` 186.9 GB; community FP8 321.7 GB (tree)

**12. K2-Horizon-375B-A23B** - `IFM/K2-Horizon-375B-A23B`, license **Apache-2.0**, data and training code also released (card L21)
- 375B stored / 23B active (card L21); HF 379.2B (api)
- 61 layers, first 3 dense; hidden 6144; plain **GQA 48 Q / 8 KV, head 128, no sliding window**; 192 experts top-8 + 1 shared, intermediate 1792 (cfg)
- Context 524,288 (card L30; cfg)
- BF16 758.3 GB (tree). No official FP8.

**13. MiniMax-M3** - `MiniMaxAI/MiniMax-M3`, license **minimax-community** (`.../LICENSE`: commercial use must display "Built with MiniMax M3"; >US$20M yearly revenue needs written authorization)
- ~428B / ~23B active (card L34); HF 427.0B (api)
- 60 layers, 3 dense; hidden 6144; GQA 64 Q / 4 KV, head 128, partial RoPE 64; **MiniMax Sparse Attention** (lightning indexer: 4 index heads x 128, block 128, top-16 blocks + 1 local) on 57 layers (card L38-48; cfg `sparse_attention_config`)
- MoE: 128 experts, top-4, + 1 shared, intermediate 3072 (cfg)
- Context 1,048,576 (cfg; card "1M")
- BF16 854.2 GB; official `MiniMax-M3-MXFP8` 443.7 GB; `amd/MiniMax-M3-MXFP4` 242.7 GB (tree)

### 1.2 Reference rows (lower AA, but relevant)

| Model | Total / active | Arch notes | Context | Native ckpt | License | Sources |
|---|---|---|---|---|---|---|
| Kimi K2.7 Code (25.8) | 1T / 32B | DeepSeek-V3-style MLA, 61 L, 384e top-8 +1 | 256K | INT4 QAT (W4, g32) 595.2 GB | modified-MIT | card L44-57, L146; cfg; tree |
| Inkling-Small (25.7) | 276B / 12B | 42 L, local SWA-512 (35 L) + global GQA 32/8, 256e top-6 +2 shared | 1,048,576 | BF16 531.9 GB; NVFP4 170.7 | Apache-2.0 | card L53-57; cfg; tree |
| Hy3 (25.3) | 295B / 21B (+3.8B MTP) | 80 L GQA 64/8 h128, 192e top-8 +1 | 256K | BF16 597.6 GB; FP8 299.9 | Apache-2.0 | card L63-80; tree |
| Inkling (25.0) | 975B / 41B | 66 L, 55 local SWA-512 + 11 global GQA 64/8 | 1,048,576 | BF16 1,904.8 GB; NVFP4 592.0 | Apache-2.0 | card L53-61; cfg; tree |
| Nemotron 3 Ultra (22.9) | 550B / 55B | 108 blocks: 48 Mamba-2 + 48 LatentMoE + 12 attention (GQA 64/2); 512e top-22 +1 | card "up to 1M"; cfg 262,144 | BF16 1,121.1 GB; NVFP4 352.3 | openmdw-1.1 | card L70-72, L186; cfg; tree |
| gpt-oss-120b (11.6) | 117B / 5.1B | 36 L alternating SWA-128 / full, GQA 64/8 h64, 128e top-4 | 131,072 | MXFP4 experts, 65.2 GB | Apache-2.0 | card L25, L41; cfg; tree |
| Qwen3-0.6B | 0.75B dense (tied emb) | 28 L, hidden 1024, GQA 16/8 h128, QK-norm | 40,960 | BF16 1.5 GB | Apache-2.0 | cfg; api; tree |
| Qwen3-1.7B | 2.03B dense (tied emb) | 28 L, hidden 2048, GQA 16/8 h128 | 40,960 | BF16 4.1 GB | Apache-2.0 | cfg; api; tree |
| Qwen3.5-0.8B / 2B | 0.87B / 2.27B dense | 24 L = 18 GDN + 6 gated attention (GQA 8/2 h256) | 262,144 | BF16 | Apache-2.0 | cfg; api |
| Qwen3-32B | 32.8B dense | 64 L, GQA 64/8 h128 | 40,960 | BF16 65.5 GB | Apache-2.0 | cfg; api; tree |

### 1.3 KV-cache per token (derived from cfg, BF16 KV, global-attention layers only)

Formula: layers_global x 2 (K,V) x kv_heads x head_dim x 2 bytes, or MLA latent (kv_lora + rope) x 2 bytes.
Linear-attention / SWA layers add a constant per-sequence state (fp32 recurrent state assumed). Computed
in `mw/` with the cfg numbers above.

| Model | KV / token | 128K ctx / seq | constant state / seq |
|---|---|---|---|
| MiMo-V2.6-Pro | 50.0 KiB | 6.7 GB | 39 MB (SWA) |
| GLM-5.3 | 87.8 KiB (MLA latent, excl. indexer keys) | 11.8 GB | - |
| Kimi K3 | 27.0 KiB | 3.6 GB | 434 MB (KDA) |
| GLM-5.3-Flash | 11.0 KiB | 1.5 GB | 143 MB (KDA) |
| Qwen3.8-2.4T-A95B | 92.0 KiB | 12.3 GB | 579 MB (GDN) |
| Qwen3.8-Flash-Next | 24.0 KiB | 3.2 GB | 113 MB (GDN) |
| DeepSeek-V4.1-Flash | 890 B (card, FP4 KV) | 0.12 GB | SWA replay |
| MiMo-V2.6-Flash | 22.5 KiB | 3.0 GB | 26 MB |
| DeepSeek-V4-Pro-0813 | ~5 KiB **UNCONFIRMED estimate** | ~0.7 GB | - |
| Qwen3.8-27B | 64.0 KiB | 8.6 GB | 151 MB |
| Motif-3 | ~15 KiB **UNCONFIRMED** (SWA pattern) | ~2 GB | - |
| K2-Horizon-375B | 244.0 KiB | 32.8 GB | - |
| MiniMax-M3 | 120 KiB + indexer keys (~57 KiB) | 16-24 GB | - |
| gpt-oss-120b | 36.0 KiB | 4.8 GB | 4.7 MB |
| Qwen3-0.6B / 1.7B | 112.0 KiB | 15.0 GB | - |

---

## 2. Hardware, confirmed from AWS

| Instance | Accelerators | Memory | Bandwidth | Low precision | Source |
|---|---|---|---|---|---|
| **g7e.48xlarge** | 8x NVIDIA RTX PRO 6000 Blackwell Server Edition | 96 GB GDDR7 each, **768 GB total**; 2,048 GiB host; 4 x 3.8 TB NVMe | 1,597 GB/s per GPU; 1,600 Gbps EFA | 5th-gen Tensor Cores with FP4 | `https://aws.amazon.com/ec2/instance-types/g7e/` (L197, L209, size table); EC2 doc lists "768 GiB (8 x 96 GiB)" `https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html` |
| **trn1.32xlarge** | 16x Trainium (2 NeuronCore-v2 each = 32 cores) | 32 GiB/chip, **512 GiB**; 512 GiB host | 820 GiB/s/chip; NeuronLink-v2 2D torus, 384 GB/s/chip | **190 TFLOPS BF16/FP16/cFP8/TF32 per chip - FP8 runs at the BF16 rate**; no MX, no FP4 | `https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium.html`; `.../trainium2.html` comparison table; EC2 doc |
| **inf2.48xlarge** | 12x Inferentia2 (24 NeuronCore-v2) | 32 GiB/chip, **384 GiB**; 768 GiB host | 9,840 GiB/s total; NeuronLink-v2 192 GiB/s/chip | 2,280 TFLOPS FP8/FP16/BF16/TF32 total (same rate) | `.../neuron-hardware/inf2-arch.html`; `.../inferentia2.html` |
| **trn2.48xlarge** | 16x Trainium2 (8 NeuronCore-v3 each = 128 physical, 64 logical at LNC=2) | 96 GiB/chip, **1,536 GiB**; 2,048 GiB host | 2.9 TB/s/chip, 46.4 TB/s total; NeuronLink-v3 4x4 torus 1,024 GB/s/chip | **1,299 FP8 / 667 BF16 TFLOPS per chip** (20.8 / 10.7 PFLOPS instance); cFP8 only - MXFP8/MXFP4 matmul (`nc_matmul_mx`) is "Available only on NeuronCore-v4 and newer" | `.../neuron-hardware/trainium2.html`; `.../trn2-arch.html`; `https://aws.amazon.com/ec2/instance-types/trn2/`; `.../nki/api/generated/nki.isa.nc_matmul_mx.html` |
| **Trn3** | Trainium3 (8 NeuronCore-v4) | **144 GiB HBM3e/chip**; Gen1 UltraServer 64 chips (4 servers x 16) = 9,216 GiB; Gen2 144 chips (36 servers x 4) = 20,736 GiB | 4.9 TB/s/chip; NeuronLink-v4 2.56 TB/s/chip; NeuronSwitch all-to-all | **2,517 MXFP8/MXFP4 TFLOPS**, 671 BF16 per chip; OCP `float8_e4m3fn` accepted by `nc_matmul` on trn3 | `.../neuron-hardware/trainium3.html`; `.../trn3-arch.html`; `https://aws.amazon.com/ec2/instance-types/trn3/`; `.../release-notes/components/nki.html` L163 |

Discrepancies and gaps, all worth knowing before quoting a spec:
- **trn2.48xlarge accelerator memory**: the EC2 instance-types doc says "8192 GiB (16 x 512 GiB)" (`docs.aws.amazon.com/ec2/latest/instancetypes/ac.html`), while the Neuron docs and the trn2 product page say 96 GiB/chip, 1,536 GiB / "1.5 TB". The EC2 doc looks wrong. I used 1,536 GiB.
- **trn1.32xlarge bandwidth**: the product page says "512 GB ... 9.8 TB/s total" (`aws.amazon.com/ec2/instance-types/trn1/` L212). That equals inf2's 12 x 820 GiB/s; 16 x 820 GiB/s would be about 13.1 TiB/s. Probably a copy error. **UNCONFIRMED** which number is right.
- **Trn3 instance size**: Trn3 is sold as UltraServers. The EC2 instance-types doc lists no `trn3.*` sizes. The NxDI GPT-OSS-on-Trn3 tutorial runs "8 copies per Trn3 instance, TP=8, LNC=2", which works out to a 16-chip instance (`.../nxd-inference/tutorials/trn3-gpt-oss-120b-tutorial.html`). Below I use a 16-chip Trn3 server (2,304 GiB) as the unit. **UNCONFIRMED** as a rentable size.
- **g7e NVLink**: the AWS page does not mention NVLink; the GPUs talk over PCIe, with "up to 4x the GPU-to-GPU bandwidth compared to G6e" (L204). Treat 8-way TP as PCIe-bound: **UNCONFIRMED**.
- **Neuron FP8 is not OCP FP8 on trn1/trn2**: NKI calls trn3's `float8_e4m3fn` "distinct from the legacy float8_e4m3" (nki release notes L163). NxDI contrib PR #137 says the OCP E4M3 range is ±448 and Neuron's IEEE-style E4M3 is ±240, so it preprocesses ("OCP ±448 -> Neuron ±240") `https://github.com/aws-neuron/neuronx-distributed-inference/pull/137`. Every HF FP8 checkpoint therefore needs re-scaling on trn1/trn2. The ±240 figure comes from the contrib, not from the data-types doc.
- Trn1/Inf2 data types: FP32, TF32, BF16, FP16, cFP8 (E5M2, E4M3, E3M4), UINT8, INT32/UINT32 (`.../neuron-features/data-types.html`).

---

## 3. Fit analysis

Capacities in decimal GB: g7e 768 (conservative marketing figure; the EC2 doc says GiB, which is 824),
trn1.32xl 512 GiB = 550, inf2.48xl 384 GiB = 412, trn2.48xl 1,536 GiB = 1,649, Trn3 16-chip server
2,304 GiB = 2,474. **Usable = 90%** (runtime, scratchpad, activations: an assumption) -> 691 / 495 / 371
/ 1,484 / 2,227. "Headroom" = usable - weights. Token capacity = headroom / KV-per-token from 1.3
(excludes the constant linear-attention state).

### 3.1 g7e.48xlarge (Blackwell, native FP8 and FP4 compute)

| Model | Precision used | Weights GB | Headroom | KV tokens | Verdict |
|---|---|---|---|---|---|
| MiMo-V2.6-Pro | native MXFP4+FP8 | 573.5 | +117 GB | 2.3M | **1 node** |
| GLM-5.3 | native FP8 | 755.6 | -64 | - | **2 nodes** at native |
| GLM-5.3 | NVFP4 (community) | 464.8 | +226 | 2.5M | 1 node, re-quantized |
| Kimi K3 | native MXFP4 | 1,560.9 | -869 | - | **3 nodes** (2 x 691 = 1,382 < 1,561) |
| GLM-5.3-Flash | native FP8 | 328.3 | +362 | 32M | **1 node** |
| Qwen3.8-2.4T-A95B | FP8 / MXFP4 | 2,496 / 1,372 | - | - | **4 nodes FP8; 3 nodes MXFP4** (2 nodes leaves 10 GB) |
| Qwen3.8-Flash-Next | BF16 / FP8 | 360 / 185.5 | +331 / +505 | 13M / 21M | 1 node |
| DeepSeek-V4.1-Flash | native FP4+FP8 | 510.3 | +180 | 196M | 1 node |
| MiMo-V2.6-Flash | native | 177.7 | +513 | 22M | 1 node (fits on 2-4 GPUs) |
| DeepSeek-V4-Pro-0813 | native FP4+FP8 | 892.7 | -201 | - | **2 nodes** (NVFP4 is 913 GB, still 2) |
| Qwen3.8-27B | BF16 | 55.6 | +635 | 9.7M | 1 GPU |
| Motif-3 | BF16 / NVFP4 | 629.7 / 186.9 | +61 / +504 | 4M / 33M | 1 node (BF16 tight) |
| K2-Horizon-375B | BF16 / FP8 (self-quantized, ~380 est.) | 758.3 / 380 | -67 / +311 | - / 1.25M | 1 node only after FP8 |
| MiniMax-M3 | BF16 / MXFP8 official | 854.2 / 443.7 | -163 / +247 | - / 2.0M | 1 node at MXFP8 |
| Kimi K2.7 Code | native INT4 | 595.2 | +96 | - | 1 node |
| Nemotron 3 Ultra | NVFP4 official | 352.3 | +339 | - | 1 node |

### 3.2 Neuron targets

Neuron constraint that changes every row: **no FP4/MX matmul before Trn3**. On trn1/inf2/trn2 an MXFP4
or NVFP4 checkpoint must either be re-quantized to FP8, which roughly doubles the expert bytes, or be
served by a custom 4-bit weight-only dequant kernel. AWS ships no such kernel for those chips; vLLM
Neuron's MXFP4 path is "Trn3" only (whats-new, 2026-07-20). On trn1/inf2 FP8 also computes at the BF16
rate, so FP8 buys capacity and bandwidth, not FLOPs. FP8 re-quantized sizes are computed from the api
dtype breakdown (4-bit params x 1 byte + FP8 + 2 x BF16).

| Model | trn1.32xl (495 usable) | inf2.48xl (371) | trn2.48xl (1,484) | Trn3 16-chip (2,227) |
|---|---|---|---|---|
| MiMo-V2.6-Pro | NO (FP8 ~1,040; 4-bit 574) | NO | **YES FP8 re-quant ~1,040 GB, +444 GB = 8.7M tok**; 4-bit kernel: +910 | YES native MXFP4, +1,653 |
| GLM-5.3 | NO | NO | **YES native FP8 755.6, +728 = 8.1M tok**; BF16 does NOT fit (contrib verified) | YES |
| Kimi K3 | NO | NO | **NO** (1,561 > 1,484, no FP4 compute) -> Trn2 UltraServer / 2 nodes | YES native MXFP4, +665 |
| GLM-5.3-Flash | **YES FP8 328, +166 = 14.8M tok** | marginal FP8, +42 | YES BF16 642.7, +841 | YES |
| Qwen3.8-2.4T-A95B | NO | NO | NO at FP8 (2,496); 4-bit 1,372 leaves +111 (custom kernel) -> 2 nodes | FP8 NO (-269); MXFP4 YES (+854) |
| Qwen3.8-Flash-Next | **YES BF16 360, +134**; FP8 +309 | YES FP8 +185; BF16 no (+11) | YES | YES |
| DeepSeek-V4.1-Flash | NO as shipped (510 > 495); YES only with the 196B Engram table in host DRAM **and** 4-bit kernels (~306 GB, +188) | NO (same trick leaves +65) | YES FP8 re-quant ~770 GB, +714 | YES native |
| MiMo-V2.6-Flash | **YES FP8 re-quant ~315, +179 = 7.8M tok** | marginal FP8 +56; 4-bit +193 (custom) | YES | YES |
| DeepSeek-V4-Pro-0813 | NO | NO | 4-bit storage 892.7, +591 (custom kernel); FP8 re-quant ~1.65 TB does NOT fit | YES native |
| Qwen3.8-27B | YES BF16 (1 node, 2-4 chips) | YES BF16 | YES on 1 chip (96 GiB) | YES |
| Motif-3 | YES FP8 322, +173 | marginal FP8 +49 | YES BF16 +854 | YES |
| K2-Horizon-375B | YES FP8 ~380, +114 (only 0.46M KV tok at 244 KiB/tok) | NO | YES BF16 +726 | YES |
| MiniMax-M3 | marginal FP8 444, +51 | NO | YES BF16 +630 | YES |
| gpt-oss-120b | YES (FP8 ~120 / BF16 ~234) | YES | YES (officially supported) | YES (official tutorial) |

Per-rank limits sit on top of the totals: a trn1 NeuronCore-v2 has 16 GiB; a trn2 logical core at LNC=2
has 24 GiB. The GLM-5.2 contrib reports that BF16 failed because the token-generation graph "needs ~26
GB per core vs the 24 GB per-core bank". The MiMo-V2.5-Pro PR reports per-rank tensors of about 20 GB of
the 24 GB.

---

## 4. Official Neuron support, and where we would be first

Lifecycle facts (all `awsdocs-neuron.readthedocs-hosted.com/en/latest/...`):
- Neuron **2.32.0** (2026-08-17) is the latest release (`release-notes/index.html`).
- **NxD Inference is in maintenance mode from 2.32.0**: "no new feature releases are planned"; migrate to vLLM Neuron (`release-notes/components/nxd-inference.html`; `about-neuron/whats-new.html`). NxDI and NxD Core are no longer in the 2.32 DLAMIs or DLCs.
- Since 2.29 (2026-04-09): **"NxD Inference models are now only supported on Trn2 and newer hardware"**; Trn1/Inf2 users must pin 2.28 (`nxd-inference.html`, 2.29 Breaking Changes).
- **Torch/XLA inference on Inf2 and Trn1 is in maintenance mode from 2.31** (announced 2026-06-28; `about-neuron/announcements/index.html`).
- **vLLM Neuron (Beta) v0.24.0.1.1.0** (2026-08-17) is Trn2/Trn3 only. Its model registry (`vllm_neuron/model/registry.py` at tag `release-0.24.0.1.1.0`, `https://github.com/vllm-project/vllm-neuron`) has exactly: `LlamaForCausalLM`, `GptOssForCausalLM`, `Eagle3LlamaForCausalLM`, `Qwen3ForCausalLM`, `Qwen3VLForConditionalGeneration`. Recipes: Llama 3 (1B/8B/70B), GPT-OSS 20B/120B, Qwen3-VL-32B, Qwen3-Embedding-8B (`vllm-neuron/docs/model-recipes/index.html`).
- NxDI "production ready" architectures (`libraries/nxd-inference/developer_guides/model-reference.html`): Llama 2-3.3 (and Mistral via Llama), Llama 4, Mixtral, DBRX, Qwen2.5, Qwen3 (0.6B-32B), Qwen3 MoE (235B-A22B), FLUX.1, Pixtral, Qwen2-VL, Qwen3-VL. The source tree also has `gpt_oss`, `deepseek` (V3), `gemma3`, `mistral` (`src/neuronx_distributed_inference/models/`, GitHub).
- Signals of where AWS is heading: 2.32 NKI-Lib adds "DeepSeek-V3.2 sparse-MLA context encoding" kernels, MXFP8 flash-decode, and a fused GPT-OSS sliding-window block (whats-new 2026-08-17).

Community contribs (`contrib/models/` in `aws-neuron/neuronx-distributed-inference`, plus open PRs). They are not official, but they are the de-facto baselines:
- **GLM-5.2** (`glm_moe_dsa`, the same architecture as GLM-5.3): trn2.48xlarge, FP8, TP=64, BS=1, seq 2048: **TTFT 6,377 ms, ITL 241.5 ms/tok, 2.96 tok/s**; "DSA indexer disabled", so long context (>2048) is "future work" (`contrib/models/GLM-5.2/README.md`).
- **MiniMax-M3** text backbone: trn2, usable at 2K context or less (2K, BS=8: TTFT 7.3 s, ITL 54 ms); 8K and 16K fail on per-rank HBM (`contrib/models/MiniMax-M3/README.md`).
- **MiMo-V2.5-Pro** (the same `MiMoV2ForCausalLM` shape as V2.6-Pro), **open PR #150**: trn2.48xlarge, FP8 1,033 GB, TP=64/EP=64: **TPOT 220 ms at concurrency 1 (4.3 output tok/s); 55 output tok/s at concurrency 48**.
- Open PRs also cover MiMo-V2-Flash (#137), MiMo-V2.5 (#148), GLM-5 (#143), Kimi-K2.5 (#145: 21.4 ms TPOT at seq 512), Kimi-K2-0905 (#131), MiniMax-M2 (#138), Qwen3.5 / 3.6 / Qwen3-Coder-Next GDN models (#128, #140, #152, #170, #173). Merged contribs: Qwen3.5-2B, Qwen3.5-35B-A3B ("first DeltaNet + MoE integration on Neuron", trn2, TP=8, BF16), DeepSeek-V3, Qwen3-0.6B (profiled on trn1).

| Model | NxDI official | vLLM Neuron | Community baseline | Our position |
|---|---|---|---|---|
| MiMo-V2.6-Pro / Flash | no | no | PR #150 (V2.5-Pro), PR #137 (V2-Flash), FP8 on trn2 | **Near-first**: same arch has unmerged contribs; nothing for MXFP4 weights or trn1 |
| GLM-5.3 | no | no | GLM-5.2 contrib (2.96 tok/s, 2K context, DSA off) | **Head-to-head vs a weak contrib**; first with DSA long context |
| Kimi K3 | no | no | none (K2.x only) | **First** (KDA + LatentMoE + MXFP4) |
| GLM-5.3-Flash | no | no | none (`glm5_next`, KDA) | **First** |
| Qwen3.8-2.4T / Flash-Next / 27B | no | no (`Qwen3ForCausalLM` is the old Qwen3) | Qwen3.5/3.6 GDN contribs (2B, 35B-A3B merged; 27B open) | **First** for QSA, n-gram embedding, HC; near-first for GDN dense/MoE |
| DeepSeek-V4.1-Flash / V4-Pro | no (V3 code only) | no | DeepSeek-V3 contrib | **First** (CSA/CSA2, mHC, Engram, FP4 experts, CED) |
| Motif-3, K2-Horizon, Hy3, Inkling, Nemotron 3 | no | no | none | **First** |
| MiniMax-M3 | no | no | contrib, short context only | Head-to-head vs a weak contrib |
| gpt-oss-20b/120b | yes (+ Trn3 tutorial) | **yes** (Trn2/Trn3, MXFP4 on Trn3, EAGLE3) | - | **Head-to-head vs AWS's best-optimized path** |
| Qwen3 dense 0.6B-32B | yes (tested on Trn1, 2.25) | **yes** | - | Head-to-head; a free reference oracle |
| Qwen3-30B-A3B / 235B-A22B | yes (Qwen3 MoE) | no | - | Head-to-head vs NxDI (maintenance) |
| Llama 3.x / 4 | yes | Llama 3 yes | - | Head-to-head |

**On trn1 and inf2 nothing 2026-era is supported, and nothing will be.** NxDI dropped those chips at
2.29, Torch/XLA inference on them is in maintenance, and vLLM Neuron is Trn2/Trn3 only. Any 2026 model on
trn1 is greenfield for us. The flip side is that the AWS platform under it is frozen.

---

## 5. Recommended model ladder

**(a) Bring-up: `Qwen/Qwen3-0.6B`, then `Qwen/Qwen3-1.7B`** (+ `Qwen/Qwen3.5-0.8B` for linear attention)
- 0.75B params, 1.5 GB BF16, so one 16 GiB trn1 NeuronCore holds it. 28 layers, hidden 1024, GQA 16/8 x 128, QK-norm, tied embeddings, vocab 151,936; KV 112 KiB/token (cfg; section 1.3).
- It is the only tiny model with an official reference on **every** Neuron target: NxDI Qwen3 (tested on Trn1 since 2.25; pinned 2.28 for trn1/inf2), vLLM Neuron `Qwen3ForCausalLM` on trn2/trn3, and a contrib profile on trn1.32xlarge (TP=2, BS=1, seq 512: 70.74 tok/s; TP=8: 196 tok/s; `contrib/models/Qwen3-0.6B/README.md`). That gives a numerics and performance oracle on day one, and the same blocks scale to Qwen3-32B / 235B.
- Add Qwen3.5-0.8B (0.87B; 18 Gated DeltaNet + 6 gated-attention layers, head 256, 262,144 ctx; cfg) early. Four of the top six models are linear-attention hybrids (Kimi K3 and GLM-5.3-Flash use KDA; Qwen3.8-2.4T and Qwen3.8-Flash-Next use GDN), and so is the (b) pick, Qwen3.8-27B. The Qwen3.5-2B contrib can serve as its oracle.

**(b) Mid dense: `Qwen/Qwen3.8-27B`** (AA 33.7, #10 open, the highest non-deprecated dense open model; the next dense is K2 Horizon 7B at 20.6, `num_experts=0` in `https://huggingface.co/IFM/K2-Horizon-7B/raw/main/config.json`, then Gemma 4 31B at 14.7)
- 27.8B params; 55.6 GB BF16 / 30.9 GB official FP8. One trn2 chip (96 GiB) holds BF16 plus about 30 GB of KV; also fits trn1 (TP 4-16) and inf2.
- KV 64 KiB/token + 151 MB/seq of GDN state, so a 128K sequence is 8.6 GB.
- Its layer types (Gated DeltaNet + gated attention, head 256) are exactly those of Qwen3.8-Flash-Next and Qwen3.8-2.4T, so the kernels carry upward.
- Apache-2.0, whereas Flash-Next and 2.4T carry MaaS clauses.
- No official Neuron support; contribs for the 3.5/3.6-27B shape are still open PRs, so we would be near-first.
- Keep Qwen3-32B (official in both AWS stacks) as the control for a head-to-head throughput comparison.

**(c) Mid MoE on one trn1.32xlarge: `XiaomiMiMo/MiMo-V2.6-Flash-RL`** (AA 37.9; 309B / 15B active; MIT)
- Native checkpoint is 177.7 GB (MXFP4 + FP8). trn1 has no MX/FP4 compute, so we re-quantize to Neuron FP8: about 315 GB. PR #137 produced "~310 GB" Neuron-FP8 for the same-shape V2-Flash. That leaves 179 GB of headroom, about 7.8M tokens at 22.5 KiB/token (only 9 global layers).
- Bandwidth roofline: 15 GB of active weights per token at FP8 against 9.8-13 TB/s aggregate HBM (see the section 2 discrepancy) is about 650-870 tok/s at BS=1 (theoretical).
- Attention is GQA + SWA-128 + sinks, the same primitives AWS already optimizes for gpt-oss. It is the **same `MiMoV2ForCausalLM` code path as (d)**, so trn1 bring-up scales straight to the flagship.
- Higher-scoring alternative: **GLM-5.3-Flash** (AA 41.8, MIT). Native FP8 328 GB fits with +166 GB (14.8M tokens at 11 KiB/token), but it needs KDA chunked delta-rule kernels, a DSA indexer and mHC on NeuronCore-v2, where AWS no longer ships kernel support.
- Excluded: **Qwen3.8-Flash-Next** (AA 39.8). It even fits at BF16 (360 GB, +134), but its license requires every MaaS operator to obtain a separate license from Qwen.

**(d) Flagship on trn2.48xlarge: `XiaomiMiMo/MiMo-V2.6-Pro-RL`** (AA #1 open, 46.3; 1.02T / 42B active; MIT; 1M context)
- FP8 re-quantized it is about 1,040 GB, leaving +444 GB = 8.7M KV tokens at 50 KiB/token, or about 64 concurrent 128K sequences. With a 4-bit weight-only kernel the native 573.5 GB leaves +910 GB.
- The fit is proven by the community for the same architecture at the same size: MiMo-V2.5-Pro's 1,033 GB FP8 ran on one trn2.48xlarge (PR #150).
- PR #150 is also the bar to beat: 220 ms TPOT at concurrency 1, 55 output tok/s at concurrency 48. The bandwidth roofline for 42 GB/token at 46.4 TB/s is about 1,100 tok/s at BS=1 (theoretical).
- On Trn3 it runs native MXFP4 (2,517 TFLOPS/chip) with 1.65 TB headroom on a 16-chip server. On g7e it fits one node natively (+117 GB), which sets the GPU cost comparison.
- Alternative: **GLM-5.3** (AA 44.8). Native FP8 755.6 GB fits trn2 (+728 GB); BF16 does not. The incumbent is the GLM-5.2 contrib at 2.96 tok/s with only 2K context, so long-context DSA is the differentiator.
- Not eligible: **Kimi K3** (AA 43.6). Its 1,560.9 GB native MXFP4 exceeds trn2's 1,484 GB usable, and trn2 has no FP4 compute; it needs a Trn2 UltraServer or Trn3.

---

## 6. UNCONFIRMED items
- Which MiMo-V2.6-Pro checkpoint (RL vs MOPD) AA scored.
- DeepSeek-V4-Pro and Motif-3 KV-per-token (my estimates; the cards give no number).
- MXFP4 -> Neuron FP8 (±240) re-quantization accuracy for MiMo-V2.6: it should be near-lossless (E2M1 x 2^k is exact in E4M3 inside its range), but it has not been measured.
- K2-Horizon FP8 size (~380 GB) is my estimate; there is no official FP8 checkpoint.
- Trn3 rentable instance size (16-chip vs 4-chip server).
- g7e PCIe-only GPU interconnect.
- trn1.32xlarge total HBM bandwidth (9.8 TB/s per the page vs 16 x 820 GiB/s per the chip spec).
- The 90% usable-memory assumption and the token capacities that follow from it.
- Rooflines are bandwidth ceilings, not predictions.
