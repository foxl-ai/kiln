#!/bin/bash
# trn1 decode-step A/B: tensor-parallel experts for decode (TP: KILN_MOE_EP=0) and the fixed-cost stack (ST: TP +
# KILN_DENSE_FP8=0 + KILN_DSA_PREFIX=mm + KILN_DECODE_WHOLE=1), q/dc1-t1 graphs.   bash tools/dc_td6.sh <box> <cfg> <variant...>
set -uo pipefail
BOX=$1; CFG=$2; shift 2
cd /opt/kiln/src
E0="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/dc1-t1/ KILN_MOE_EP=0"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024"
declare -A C
C[G16]="--max-num-seqs 16 --concurrency 16 --decode-buckets 1,4 --kv-cache-gb 0.65"
C[G64]="--max-num-seqs 64 --concurrency 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8"
TD="--all-buckets --steps 64 --skip 8 --pieces --price trn1.32xlarge-spot=2.15"
for v in "$@"; do
  case $v in TP) X="" ;; ST) X="KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1" ;; esac
  bash tools/dc_run.sh $BOX td-$CFG-$v "$E0 $X" tools/time_decode.py $TD -- $A ${C[$CFG]}
  echo "td-$CFG-$v rc=$(cat /opt/kiln/logs/td-$CFG-$v.log.rc)" >> /opt/kiln/logs/dc-td6.txt
done
echo TD6_DONE >> /opt/kiln/logs/dc-td6.txt
