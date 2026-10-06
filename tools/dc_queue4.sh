#!/bin/bash
# q/dc1-t1: KILN_DENSE_FP8=0 (the FP8 weights outside the routed experts dequantized once at load) on the G16 / G64 decode
# shapes, 12-layer decode graphs.
set -uo pipefail
cd ${KILN_BOX_SRC:-/opt/kiln/src}
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_DENSE_FP8=0"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024"
G16="--max-num-seqs 16 --concurrency 16 --decode-buckets 1,4 --kv-cache-gb 0.65"
G64="--max-num-seqs 64 --concurrency 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8"
L=/opt/kiln/logs
run() { bash tools/dc_capture.sh "$@" > $L/dcq-$2.out 2>&1; echo "$2 rc=$? $(tail -1 $L/dcq-$2.out)" >> $L/dcq-enqueued.txt; }
run dc1-t1 t1-G16-DF0 trn1 "$E1" $A $G16 &
run dc1-t1 t1-G64-DF0 trn1 "$E1" $A $G64 &
wait
