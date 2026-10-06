#!/bin/bash
# q/dc1-t2: the trn2 analogue of trn1's fixed-cost stack on top of the decode kernels: K (KDA / DSA decode kernels, SP
# decode streams) + the moe_dedupe LNC split (the trn2 agent's D2) + KILN_DENSE_FP8=0 + KILN_DSA_PREFIX=mm +
# KILN_DECODE_WHOLE=1 (TP experts are the trn2 default), decode buckets 1 / 4 / 16 / 32 / 64 rows per group.
set -uo pipefail
cd ${KILN_BOX_SRC:-/opt/kiln/src}
E2="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=0 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=6 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_KDA_DECODE_KERNEL=nki KILN_DSA_DECODE_KERNEL=nki KILN_DECODE_SP=1"
X="KILN_LNC_SPLIT=delta_rule,dsa_topk,moe_dedupe KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 512 --kv-cache-gb 0.5 --kv-cache-dtype fp8 --no-prefix-caching --decode-buckets 1,4,16,32,64"
L=/opt/kiln/logs
bash tools/dc_capture.sh dc1-t2 t2-TD-KST trn2 "$E2 $X" $A > $L/dcq-t2-TD-KST.out 2>&1
echo "t2-TD-KST $(tail -1 $L/dcq-t2-TD-KST.out)" >> $L/dcq-enqueued.txt
