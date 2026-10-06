#!/bin/bash
# Decode-step cost curve on trn2 (tools/time_decode.py, real weights, one tp=32 engine on half the box, DP attention 4,
# page bucket 264, decode buckets 16 / 32 / 64 rows per DP group), every graph from q/dc1-t2:
#   td2-X   the trn2 defaults (XLA decode paths: KDA / DSA decode kernels and SP decode streams are trn1-only defaults)
#   td2-K   the trn1 decode kernels and SP decode streams turned on (KILN_KDA_DECODE_KERNEL / KILN_DSA_DECODE_KERNEL=nki,
#           KILN_DECODE_SP=1)
#   bash tools/dc_td2.sh <core_base> <variant...>
set -uo pipefail
CB="$1"; shift
SRC=${KILN_BOX_SRC:-/opt/kiln/src-decode}
cd $SRC
E2="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=0 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=6 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 NEURON_LOGICAL_NC_CONFIG=2 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/dc1-t2/"
K="KILN_KDA_DECODE_KERNEL=nki KILN_DSA_DECODE_KERNEL=nki KILN_DECODE_SP=1"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 512 --kv-cache-gb 0.5 --kv-cache-dtype fp8 --no-prefix-caching --decode-buckets 16,32,64"
TD="--all-buckets --steps 48 --skip 8 --pieces --price trn2.48xlarge-spot-half-box=7.545"
for v in "$@"; do
  X=""; AA="$A"
  case $v in
    K) X="$K" ;;
    K1) X="$K"; AA="${A/--decode-buckets 16,32,64/--decode-buckets 1,4}" ;;  # the fixed cost: 1 and 4 rows per group
    KW) X="$K KILN_DECODE_WHOLE=1" ;;  # the decode call as one graph
    KST) X="$K KILN_LNC_SPLIT=delta_rule,dsa_topk,moe_dedupe KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1"
         AA="${A/--decode-buckets 16,32,64/--decode-buckets 1,4,16,32,64}" ;;  # the fixed-cost stack on trn2
  esac
  KILN_BOX_SRC=$SRC bash tools/dc_run.sh kiln-t2-cb td2-$v "$E2 ${DC_EXTRA_ENV:-} $X" tools/time_decode.py $TD -- $AA --core-base $CB
  echo "td2-$v rc=$(cat /opt/kiln/logs/td2-$v.log.rc)" >> /opt/kiln/logs/dc-td2.txt
done
echo TD2_DONE >> /opt/kiln/logs/dc-td2.txt
