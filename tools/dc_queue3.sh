#!/bin/bash
# q/dc1-t1, the synthetic W1M decode shape on the current tree: page bucket 32768 (1,048,576 keys), decode bucket
# 1 / 4 rows per DP group, KV rows the null page (tools/time_decode.py), fp8 KV.
set -uo pipefail
cd ${KILN_BOX_SRC:-/opt/kiln/src}
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 1044480 --output-len 4096 --page-buckets 32768 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 16 --kv-cache-gb 0.5 --kv-cache-dtype fp8 --no-prefix-caching --decode-buckets 1,4"
L=/opt/kiln/logs
bash tools/dc_capture.sh dc1-t1 t1-W1M-G12 trn1 "$E1" $A > $L/dcq-t1-W1M-G12.out 2>&1
echo "t1-W1M-G12 rc=$? $(tail -1 $L/dcq-t1-W1M-G12.out)" >> $L/dcq-enqueued.txt
