#!/bin/bash
# q/dc1-t1: STL9 at 1f966d4 (v9 tuned, its slot-tail segment): STL9d the 192-row calls on v9, STL9e with
# KILN_MOE_DEDUPE_V9=1 (the 128-row calls on v9 too).
set -uo pipefail
cd ${KILN_BOX_SRC:-/opt/kiln/src}
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1 KILN_MOE_DEDUPE_MAX_TOKENS=256"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024"
G="--max-num-seqs 192 --kv-cache-gb 0.5 --kv-cache-dtype fp8 --no-prefix-caching --decode-buckets 32,48"
L=/opt/kiln/logs
run() { bash tools/dc_capture.sh "$@" > $L/dcq-$2.out 2>&1; echo "$2 rc=$? $(tail -1 $L/dcq-$2.out)" >> $L/dcq-enqueued.txt; }
run dc1-t1 t1-STL9d trn1 "$E1" $A $G &
run dc1-t1 t1-STL9e trn1 "$E1 KILN_MOE_DEDUPE_V9=1" $A $G &
wait
