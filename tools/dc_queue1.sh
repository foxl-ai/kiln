#!/bin/bash
# Decode-scale farm queues on engine-v0 70ddc1b (feat/decode-scale): captures, then workers.
#   q/dc1-t1: trn1, the three serving decode shapes (G16 4 / F0 8 / G64 16 rows per DP group, plus 1 row on G16)
#             with 12-layer (the default), 24-layer and one 45-layer decode graph (KILN_PIECEWISE_MOE_GROUP).
#   q/dc1-t2: trn2 (LNC=2), one tp=32 engine, decode buckets 16 / 32 / 64 rows per group, the trn2 defaults
#             (XLA decode paths) and the trn1 decode kernels + SP decode streams turned on.
set -uo pipefail
cd /opt/kiln/src
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024"
G16="--max-num-seqs 16 --concurrency 16 --decode-buckets 1,4 --kv-cache-gb 0.65"
F0="--max-num-seqs 32 --concurrency 32 --decode-buckets 8 --kv-cache-gb 1.5"
G64="--max-num-seqs 64 --concurrency 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8"
E2="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=0 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=6 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1"
T2="--max-num-seqs 512 --kv-cache-gb 0.5 --kv-cache-dtype fp8 --no-prefix-caching --decode-buckets 16,32,64"
K="KILN_KDA_DECODE_KERNEL=nki KILN_DSA_DECODE_KERNEL=nki KILN_DECODE_SP=1"
L=/opt/kiln/logs
run() { bash tools/dc_capture.sh "$@" > $L/dcq-$2.out 2>&1; echo "$2 rc=$? $(tail -1 $L/dcq-$2.out)" >> $L/dcq-enqueued.txt; }
run dc1-t2 t2-TD-X trn2 "$E2" $A $T2 &
run dc1-t2 t2-TD-K trn2 "$E2 $K" $A $T2 &
for G in 48 24 12; do  # the longest compiles first
  run dc1-t1 t1-G16-G$G trn1 "$E1 KILN_PIECEWISE_MOE_GROUP=$G" $A $G16 &
  run dc1-t1 t1-F0-G$G trn1 "$E1 KILN_PIECEWISE_MOE_GROUP=$G" $A $F0 &
  run dc1-t1 t1-G64-G$G trn1 "$E1 KILN_PIECEWISE_MOE_GROUP=$G" $A $G64 &
  wait -n; wait -n; wait -n
done
wait
echo CAPTURES_DONE >> $L/dcq-enqueued.txt
