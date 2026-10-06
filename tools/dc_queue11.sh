#!/bin/bash
# q/dc1-t1: ST without DP attention (--dp-attention 1, attention TP 32): at small batch the mixer weights a rank reads per
# step are a quarter of DP attention 4's (attention TP 8: 1.36 GB of KDA / DSA projections per rank), at the price of
# every rank holding every sequence's KV. Decode buckets 4 and 16 rows per step (= DP 4's 1 and 4 rows per group).
set -uo pipefail
cd ${KILN_BOX_SRC:-/opt/kiln/src}
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 1 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024"
G="--max-num-seqs 16 --concurrency 16 --decode-buckets 4,16 --kv-cache-gb 0.65"
L=/opt/kiln/logs
bash tools/dc_capture.sh dc1-t1 t1-D1-ST trn1 "$E1" $A $G > $L/dcq-t1-D1-ST.out 2>&1
echo "t1-D1-ST rc=$? $(tail -1 $L/dcq-t1-D1-ST.out)" >> $L/dcq-enqueued.txt
