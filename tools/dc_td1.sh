#!/bin/bash
# Decode-step A/B on trn1 (tools/time_decode.py, real weights, serving decode shapes, every graph from the farm):
#   td-<cfg>-base   the engine-v0 70ddc1b defaults (graphs of q/final-f70c14b, key-identical)
#   td-<cfg>-nb     the same with NEURON_RT_DISABLE_EXECUTION_BARRIER=1 (runtime only, no key change)
# cfg: G64 (16 rows per DP group), G16 (4), F0 (8).
set -uo pipefail
cd /opt/kiln/src
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1"
FARM="KILN_COMPILE_FARM=${DC_FARM:-s3://<your-bucket>/compile-farm/q/final-f70c14b/}"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024"
declare -A C
C[G16]="--max-num-seqs 16 --concurrency 16 --decode-buckets 4 --kv-cache-gb 0.65"
C[F0]="--max-num-seqs 32 --concurrency 32 --decode-buckets 8 --kv-cache-gb 1.5"
C[G64]="--max-num-seqs 64 --concurrency 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8"
TD="--steps 64 --skip 8 --pieces --price trn1.32xlarge-spot=2.15"
for cfg in ${DC_CFGS:-G64 G16 F0}; do
  for v in ${DC_VARIANTS:-base nb}; do
    X=""; [ $v = nb ] && X="NEURON_RT_DISABLE_EXECUTION_BARRIER=1"
    bash tools/dc_run.sh kiln-dc-32 td-$cfg-$v "$E1 $FARM ${DC_EXTRA_ENV:-} $X" tools/time_decode.py $TD -- $A ${C[$cfg]}
    echo "td-$cfg-$v rc=$(cat /opt/kiln/logs/td-$cfg-$v.log.rc)" >> /opt/kiln/logs/dc-td1.txt
  done
done
echo TD1_DONE >> /opt/kiln/logs/dc-td1.txt
