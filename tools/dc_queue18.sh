#!/bin/bash
# q/dc1-t1, from feat/decode-scale rebased onto engine-v0 f997d35 (feat/long-context merged): the CP decode box again
# (cp-R64-rb, keys to compare with the scratch merge's cp-R64) and, with kiln_moe_dedupe_v10 for the 320 / 384-row MoE
# calls (KILN_MOE_DEDUPE_MAX_TOKENS=512) and the merge in 64-row pieces, cp-R80-v10 / cp-R96-v10.
set -uo pipefail
export KILN_BOX_SRC=${KILN_BOX_SRC:-/opt/kiln/src}
cd $KILN_BOX_SRC
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1 KILN_MOE_DEDUPE_V9=1 KILN_DSA_CP=1"
A0="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --kv-cache-dtype fp8 --no-prefix-caching --page-size 256 --page-buckets 64"
L=/opt/kiln/logs
mkdir -p $L /opt/kiln/work
run() { bash tools/dc_capture.sh "$@" > $L/dcq-$2.out 2>&1; echo "$2 rc=$? $(tail -1 $L/dcq-$2.out)" >> $L/dcq-enqueued.txt; }
run dc1-t1 cp-R64-rb trn1 "$E1 KILN_MOE_DEDUPE_MAX_TOKENS=256" $A0 --max-num-seqs 256 --kv-cache-gb 0.62 --decode-buckets 64 &
[ "${DC_V10:-0}" = 1 ] && run dc1-t1 cp-R80-v10 trn1 "$E1 KILN_MOE_DEDUPE_MAX_TOKENS=512" $A0 --max-num-seqs 320 --kv-cache-gb 0.77 --decode-buckets 80 &
wait
[ "${DC_V10:-0}" = 1 ] && run dc1-t1 cp-R96-v10 trn1 "$E1 KILN_MOE_DEDUPE_MAX_TOKENS=512" $A0 --max-num-seqs 384 --kv-cache-gb 0.92 --decode-buckets 96
python3 - <<PY >> $L/dcq-enqueued.txt
import json
a=json.load(open('/opt/kiln/work/cap-cp-R64-rb/keys.json'))
b=['851bf972f8cba8f7ddd7331f86ced517', 'b9673eb6a1930cbfffa1e8d2a7b4067c', '8a8019e73cd15ff6e6891c87e6ac92ab', '95fde958ff4d7d8f82890a0a6ef6e180']
print('cp-R64-rb keys', sorted(a), 'same as the scratch cp-R64:', sorted(a) == sorted(b))
PY
