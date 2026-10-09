# 2026-10-07, engine start: capture -> parallel compile -> warmup (kiln/precompile.py) against serial compiles and vllm-neuron 0.24

vllm-neuron starts an engine by extracting every bucket's graph, compiling them all at once, and only then warming up
(vllm_neuron/vllm/worker/neuron_worker.py:1207-1240, release-0.24.0.1.1.0). Kiln compiled during its warmup, one
graph at a time behind LNL's compile lock. `KILN_PRECOMPILE_WORKERS=N` (EngineConfig.precompile_workers) does this:
- a process started before the ranks spawn captures every warmup graph on the meta device (kiln/capture.py);
- it compiles the missing ones N at once while the weights load;
- the engine waits for it before any graph runs;
- a per-configuration manifest lets a warm start skip the capture.

Harness: `bench/ttft_compare.py --engine kiln` (engine up = LLMEngine + one generate + warmup), driven by
start_bench_v2.sh. Arms, each a fresh process a minute apart:
- A: cold, serial (KILN_PRECOMPILE_WORKERS=0, empty cache);
- B: cold, precompile;
- C: warm, precompile (B's cache);
- D: warm, serial.

Arms B, C and D ran with NEURON_LIBTORCH_ASSERT_CACHE_HIT=1. SDK 2.32: neuronx-cc 2.27.5334, libtorch-neuronx-lite
2.11.0.1.0.1284, runtime 2.34.10. Tree: feat/vn-parity c5fbf34 / 78b55bc.

## trn2.3xlarge (one Trainium2 chip, 4 logical cores, 12 vCPU), Qwen3-8B bf16 TP=4, 6 workers

The same configuration as bench/results/2026-10-07-trn2.3xlarge-qwen3-ttft.md: max_model_len 32768, page 32,
page buckets 64,256,512,1024, prefill buckets 2048,4096, KV 2 GiB. Box kiln-vn-t2b (ap-southeast-4c, Capacity Block
cr-0debbd9acabcf3ffc). Logs: s3://<your-bucket>/logs/kiln-vnp/start-q8/.

| arm | engine up | of which | vllm-neuron 0.24, same box type (results file above) |
|---|---|---|---|
| A cold, serial | 503.8 s | | 491.4 s cold (chunk 8192, 2 graphs) |
| **B cold, precompile** | **278.9 s** | 27 graphs captured in 27.2 s, compiled with 6 workers in 220.3 s; 248.9 s from engine start | |
| **C warm, precompile** | **49.4 s** | manifest: 27 graphs all cached, checked in 0.01 s | 106.9 s warm |
| D warm, serial | 48.5 s | | |

- Cold: Kiln 278.9 s against vllm-neuron's 491.4 s, 1.76x faster, and 1.81x faster than Kiln's serial 503.8 s.
  Kiln compiles 27 graphs here against vllm-neuron's 2 at chunk 8192, so this is the parallel compile, not a
  smaller bucket set. With vllm-neuron at chunk 2048 (135.0 s cold) the graph set differs too much to compare.
- Warm: 49.4 s against vllm-neuron's 106.9 s, 2.2x faster.

## trn1.2xlarge (one Trainium1 chip, 2 cores, 8 vCPU), Qwen3-1.7B bf16 TP=2, 4 workers

max_model_len 8192, page buckets 64,256, prefill buckets 2048,4096, KV 1 GiB; 15 graphs. Box kiln-vnp-k1 (us-east-2c
spot). Logs: s3://.../logs/kiln-vnp/start-q17-ABCD-tokenizer-fix/ (A-D before the manifest) and
s3://.../logs/kiln-vnp/start-q17/ (B, C with it).

| arm | engine up | of which |
|---|---|---|
| A cold, serial | 270.5 s | warmup 200.0 s |
| B cold, precompile | 137.2-137.9 s | 15 graphs captured in 15.1-15.4 s, compiled in 103.3-104.0 s |
| C warm, precompile, before the manifest | 51.1 s | the capture re-ran (14.2 s) and slowed the weight load 7.9 -> 24.3 s on 8 vCPU |
| C warm, precompile, manifest | 33.3 s | skipped in 0.01 s |
| D warm, serial | 31.8 s | |

The run also surfaced an engine bug. LLMEngine loaded the tokenizer between init_rank and build_shard, and with that
order rank 0 traced torch.topk / argmax in its 3 sampling post graphs un-normalized (`kwargs={dim: -1}`). So it
compiled its own copies: 18 cache entries against the 15 every spawned rank and the capture compute, and
ASSERT_CACHE_HIT failed on a precompiled cache. The tokenizer now loads after build_shard (commit 78b55bc), and the
count is 15.
