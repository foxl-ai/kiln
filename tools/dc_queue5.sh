#!/bin/bash
# q/dc1-mtp: async MTP (KILN_SPEC_ASYNC=1, MTP k=1) on engine-v0 70ddc1b's G64 serving shape, KV 1.5 fp8, 16 checkpoint
# rows (so the KDA state pool keeps the default config's 49 rows per group and every target graph is q/final-f70c14b's
# G64 config's); a serving capture (prefill graphs too).
set -uo pipefail
cd ${KILN_BOX_SRC:-/opt/kiln/src}
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 3000 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64 --concurrency 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8"
L=/opt/kiln/logs
KILN_CAP_PREFILL=1 bash tools/dc_capture.sh dc1-mtp t1-G64-MA trn1 "$E1 KILN_SPEC_ASYNC=1" $A --spec-method mtp --spec-k 1 --state-checkpoints 16 > $L/dcq-t1-G64-MA.out 2>&1
echo "t1-G64-MA rc=$? $(tail -1 $L/dcq-t1-G64-MA.out)" >> $L/dcq-enqueued.txt
