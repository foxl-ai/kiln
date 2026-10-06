#!/bin/bash
# q/dc1-t1: KILN_DSA_PREFIX=mm (decode_slots' prefix counts as two triangular matmuls) on the G16 / G64 decode shapes.
# q/dc1-t2: the trn2 decode-kernel config (t2-TD-K's env) at 1 and 4 rows per group, and with KILN_DECODE_WHOLE=1.
set -uo pipefail
cd ${KILN_BOX_SRC:-/opt/kiln/src}
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024"
G16="--max-num-seqs 16 --concurrency 16 --decode-buckets 1,4 --kv-cache-gb 0.65"
G64="--max-num-seqs 64 --concurrency 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8"
E2="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=0 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=6 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_KDA_DECODE_KERNEL=nki KILN_DSA_DECODE_KERNEL=nki KILN_DECODE_SP=1"
T2S="--max-num-seqs 512 --kv-cache-gb 0.5 --kv-cache-dtype fp8 --no-prefix-caching --decode-buckets 1,4"
T2="--max-num-seqs 512 --kv-cache-gb 0.5 --kv-cache-dtype fp8 --no-prefix-caching --decode-buckets 16,32,64"
L=/opt/kiln/logs
run() { bash tools/dc_capture.sh "$@" > $L/dcq-$2.out 2>&1; echo "$2 rc=$? $(tail -1 $L/dcq-$2.out)" >> $L/dcq-enqueued.txt; }
run dc1-t2 t2-TD-K1 trn2 "$E2" $A $T2S &
run dc1-t2 t2-TD-KW trn2 "$E2 KILN_DECODE_WHOLE=1" $A $T2 &
run dc1-t1 t1-G16-PM trn1 "$E1 KILN_DSA_PREFIX=mm" $A $G16 &
run dc1-t1 t1-G64-PM trn1 "$E1 KILN_DSA_PREFIX=mm" $A $G64 &
wait
