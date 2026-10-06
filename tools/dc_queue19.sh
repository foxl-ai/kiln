#!/bin/bash
# q/dc1-t1: the CP-64 decode box (tools/dc_queue18.sh's cp-R64-rb) as PIECEWISE decode graphs (KILN_DECODE_WHOLE=0: prep,
# the layer groups, post), so that a captured decode call can be replayed piece by piece for a per-op profile (one whole
# decode graph's inputs, every rank's weights and KV, did not fit the host's memory for the replay).
set -uo pipefail
export KILN_BOX_SRC=${KILN_BOX_SRC:-/opt/kiln/src}
cd $KILN_BOX_SRC
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=0 KILN_MOE_DEDUPE_V9=1 KILN_DSA_CP=1 KILN_MOE_DEDUPE_MAX_TOKENS=256"
A0="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --kv-cache-dtype fp8 --no-prefix-caching --page-size 256 --page-buckets 64"
L=/opt/kiln/logs
bash tools/dc_capture.sh dc1-t1 cp-R64p trn1 "$E1" $A0 --max-num-seqs 256 --kv-cache-gb 0.62 --decode-buckets 64 > $L/dcq-cp-R64p.out 2>&1
echo "cp-R64p rc=$? $(tail -1 $L/dcq-cp-R64p.out)" >> $L/dcq-enqueued.txt
