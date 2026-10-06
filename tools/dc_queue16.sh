#!/bin/bash
# q/dc1-t1: the trn1 decode box with REAL 8K KV (t1-STL9r2): tools/hbm_estimate.py puts the STL config at 14.04 GiB per rank
# with 0.5 GB of KV (tensors 12.76 GiB: experts 10.01 GB, other weights 2.23 GB), and a page of every layer's fp8 latent
# KV + pool keys is ~293 KB per rank (replicated across an attention group's 8 ranks), a sequence of 8448 tokens ~77 MB
# plus ~19 MB of KDA state: 28 sequences per DP group fit under ~15.2 GiB, 48 do not (17 GiB). Decode bucket 28 rows
# per group (112 per step), v9 for every call (KILN_MOE_DEDUPE_V9=1), KV 2.1 GB (~7,770 pages: 28 x 264 + 1 fit).
set -uo pipefail
cd ${KILN_BOX_SRC:-/opt/kiln/src}
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1 KILN_MOE_DEDUPE_MAX_TOKENS=256 KILN_MOE_DEDUPE_V9=1"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024"
G="--max-num-seqs 112 --kv-cache-gb 2.1 --kv-cache-dtype fp8 --no-prefix-caching --decode-buckets 28"
L=/opt/kiln/logs
bash tools/dc_capture.sh dc1-t1 t1-STL9r2 trn1 "$E1" $A $G > $L/dcq-t1-STL9r2.out 2>&1
echo "t1-STL9r2 rc=$? $(tail -1 $L/dcq-t1-STL9r2.out)" >> $L/dcq-enqueued.txt
