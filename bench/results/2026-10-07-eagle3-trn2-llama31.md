# 2026-10-07, EAGLE-3 on one Trainium2 chip: Kiln against vllm-neuron 0.24 on vllm-neuron's own tutorial pair

Target meta-llama/Llama-3.1-8B-Instruct (the unsloth/Llama-3.1-8B-Instruct mirror: the LFS sha256 of all 4 shards
match), bf16, TP 4. Draft RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3 (snapshot f4fa34a8), the pair of vllm-neuron's
tutorial (`docs/tutorials/tutorial-eagle3-speculative-decoding-llama-3-1.md`, release-0.24.0.1.1.0). Box kiln-vn-t2b,
trn2.3xlarge (4 logical cores at LNC 2), ap-southeast-4c Capacity Block. SDK 2.32: neuronx-cc 2.27.5334,
libtorch-neuronx-lite 2.11.0.1.0.1284.

Harness: `tools/check_eagle3.py`.
- Prompts: 6 chat prompts through the model's chat template, batch 1, greedy, 128 new tokens, `ignore_eos`.
- Passes: each prompt runs twice in one engine.
- tok/s: 127 / (request wall - TTFT).
- max_model_len 2048.
- Kiln: page 32, one page bucket (64 pages), prefill bucket 256.
- vllm-neuron: `LLM(...)` offline, `max_num_seqs=1`, no prefix caching,
  `speculative_config={"method": "eagle3", "model": <draft>, "num_speculative_tokens": 3}`, acceptance from its
  spec-decode counters.
- k is 3 unless stated.

Logs: s3://<your-bucket>/logs/kiln-vnp/winB/ (vllm-neuron, Kiln piecewise), winC/ (Kiln whole-model graphs,
k 4), winF/ (the final tree 7c91df0: plain and k 1, 2, 3, the board probe on trn2), t2b-outs/ (the step scripts' output).

## Head-to-head (final tree, feat/vn-parity 7c91df0; Kiln whole-model graphs, overlap scheduling, overlapped speculation)

| prompt | vllm-neuron plain | vllm-neuron EAGLE-3 | its acceptance length | Kiln plain | Kiln EAGLE-3 | its acceptance length | Kiln / vllm-neuron EAGLE-3 |
|---|---|---|---|---|---|---|---|
| Python Fibonacci function | 99.20 | 192.97 | 3.17 | 100.19 | 189.64 | 3.12 | 0.983 |
| hash map collisions | 99.18 | 170.87 | 2.80 | 100.18 | 165.42 | 2.83 | 0.968 |
| train timetable | 99.17 | 164.18 | 2.70 | 100.17 | 155.49 | 2.68 | 0.947 |
| Romeo and Juliet | 99.16 | 135.06 | 2.19 | 100.19 | 131.81 | 2.19 | 0.976 |
| documentation tips | 99.17 | 160.13 | 2.61 | 100.18 | 168.99 | 2.87 | 1.055 |
| French translation | 99.18 | 113.43 | 1.87 | 100.19 | 108.00 | 1.78 | 0.952 |
| **mean** | **99.18** | **156.11** | **2.48** (accept 0.495) | **100.18** | **153.22** | **2.49** (accept 0.496) | **0.981** |

- Without speculation Kiln decodes 1.0% faster than vllm-neuron (9.98 ms against 10.08 ms per token).
- The drafts are as good as vllm-neuron's: the acceptance lengths match within 0.26 per prompt, and over all prompts
  they are 2.49 against 2.48.
- With EAGLE-3, Kiln reaches 0.981x vllm-neuron's tokens per second. A speculative step costs Kiln 16.3 ms against
  vllm-neuron's 15.9 ms (acceptance length / tok/s): 1.63 plain steps against 1.58.
- vllm-neuron's published figure for this pair (trn2.48xlarge, TP 8, concurrency 2, sonnet 512 in / 128 out, tutorial
  lines 238-239 and 278) is acceptance rate 74.14%, acceptance length 3.22, median TPOT 10.52 -> 6.41 ms. Those are
  sonnet prompts at TP 8. On these chat prompts both engines land at 2.48-2.49.

## Kiln configurations (mean tok/s over the 6 prompts)

| configuration | plain | EAGLE-3 | acceptance length |
|---|---|---|---|
| piecewise graphs, synchronous steps (Kiln's defaults) | 64.8 | 95.2 | 2.45 |
| piecewise, overlap scheduling (EAGLE-3 steps still synchronous: spec_async was k 1 only) | 91.8 | (95.2) | |
| piecewise, overlap, overlapped speculation at k 3 (3f92035) | 91.8 | 137.7 | 2.44 |
| piecewise + whole-graph decode (`KILN_DECODE_WHOLE=1`), overlap | 100.16 | | |
| whole-model graphs (`piecewise=False`), overlap, overlapped speculation k 3 | 100.18 | **153.2** | 2.49 |
| the same at k 1 | | 136.6 | 1.71 |
| the same at k 2 | | 146.0 | 2.19 |
| the same at k 4 (3f92035) | | 143.1 | 2.62 |

- k 3 is the best draft length here, as vllm-neuron's tutorial chose.
- At batch 1 a piecewise decode step (about 14 graph launches) costs 10.9 ms against 10.0 ms for one graph. Each
  synchronous step adds about 4.5 ms of host round trip (15.4 ms per token). A synchronous speculative step does two
  round trips (verify, then the draft), which is why overlapped speculation at k > 1 (3f92035) mattered most.
- Overlapped speculation at k 1 and k 2 first accepted nothing (0 of 128 drafts per prompt at k 1, accept rate 0.008
  at k 2). That was a neuronx-cc misread of the one-row board graphs, fixed in 0b854db (next section).

## The one-row board misread (spec_async, fixed in 0b854db)

`tools/probe_spec_board.py` runs the board graphs on the device against the CPU on hand-built boards.

| where | result |
|---|---|
| trn1.2xlarge, before the fix | at one row and k 1, spec_prep's draft column and mtp_prep's acc and base came back 0. Two and sixteen rows, k 3, and board widths 7, 8 and 16 were right |
| trn1.2xlarge and trn2.3xlarge, after the fix | every shape equal (k 1, 2, 3 at one and two rows) |
| trn2 at k 2 and two rows | neuronx-cc fails with `NCC_IIIV902 InferInitValue error` (winF/probe_spec_board.log). It is not an engine shape in these runs, but a k 2 overlapped engine with a decode bucket of 2 would fail at warmup, loudly |

The fix runs a one-row call to the read-only board graphs (spec_prep, mtp_prep) on the row twice and returns the first
copy. Every other shape traces as before.

## Greedy agreement

- Run-to-run jitter is zero: in every configuration of both engines, pass 0 and pass 1 give identical outputs (plain
  against plain, EAGLE-3 against EAGLE-3), on all 6 prompts.
- Plain and EAGLE-3 part at the first difference shown below, because the verify graph (4 rows) and the decode graph
  (1 row) round bf16 differently. The margin is the plain run's top-1 minus top-2 logprob at that position
  (`--logprobs`). Every one is an exact tie or one bf16 step, so these are near ties, not a verify-path defect.

| engine | prompt: first difference (plain margin there) |
|---|---|
| vllm-neuron | documentation tips: 71; French: 27 |
| Kiln piecewise | hash map: 65 (0.0); Romeo: 5 (0.0); tips: 54 (0.125); French: 27 (0.0) |
| Kiln whole-model | train: 76 (0.0); tips: 56 (0.25); French: 27 (0.0) |

## trn1

trn1.2xlarge, Qwen3-8B TP 2, RedHatAI/Qwen3-8B-speculator.eagle3, the same prompts, piecewise and synchronous: accept
rate 0.416, 1.39x mean speedup over plain (s3 logs/kiln-vnp/ceagle-t1/).
