#!/bin/bash
# q/dc1-t1: a trn1 decode-only box at large batch under ST (TP experts, DENSE_FP8=0, PREFIX=mm, WHOLE): decode buckets 32
# and 48 rows per DP group (128 / 192 rows per step; ~48 is what a rank's HBM holds at 8K context with no prefill graphs),
# every KV row the null page (tools/time_decode.py), fp8 KV, no prefix cache (no checkpoint rows).
set -uo pipefail
cd ${KILN_BOX_SRC:-/opt/kiln/src}
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024"
G="--max-num-seqs 192 --kv-cache-gb 0.5 --kv-cache-dtype fp8 --no-prefix-caching --decode-buckets 32,48"
L=/opt/kiln/logs
bash tools/dc_capture.sh dc1-t1 t1-STL trn1 "$E1" $A $G > $L/dcq-t1-STL.out 2>&1
echo "t1-STL rc=$? $(tail -1 $L/dcq-t1-STL.out)" >> $L/dcq-enqueued.txt
