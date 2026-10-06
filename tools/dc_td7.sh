#!/bin/bash
# trn1 decode-step A/Bs under ST (TP experts, DENSE_FP8=0, PREFIX=mm, WHOLE), q/dc1-t1 graphs, on engine-v0 8229c3d
# (the hardware execution barrier on by default on trn1):
#   ST    the reference, G16 shape (1 / 4 rows per group)
#   STk   KDA decode kernel off (KILN_KDA_DECODE_KERNEL=xla)      STd   DSA decode kernel off (KILN_DSA_DECODE_KERNEL=xla)
#   D1    DP attention 1 (attention TP 32), 4 / 16 rows per step  STL   32 / 48 rows per group, decode-only HBM
set -uo pipefail
BOX=${DC_BOX:-kiln-dc-32}
cd /opt/kiln/src
E0="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/dc1-t1/ KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1"
A4="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024"
G16="--max-num-seqs 16 --concurrency 16 --decode-buckets 1,4 --kv-cache-gb 0.65"
TD="--all-buckets --steps 64 --skip 8 --pieces --price trn1.32xlarge-spot=2.15"
for v in "$@"; do
  X=""; ARGS="$A4 $G16"
  case $v in
    ST) ;;
    STk) X="KILN_KDA_DECODE_KERNEL=xla" ;;
    STd) X="KILN_DSA_DECODE_KERNEL=xla" ;;
    D1) ARGS="${A4/--dp-attention 4/--dp-attention 1} --max-num-seqs 16 --concurrency 16 --decode-buckets 4,16 --kv-cache-gb 0.65" ;;
    STL) ARGS="$A4 --max-num-seqs 192 --kv-cache-gb 0.5 --kv-cache-dtype fp8 --no-prefix-caching --decode-buckets 32,48" ;;
  esac
  bash tools/dc_run.sh $BOX td7-$v "$E0 $X" tools/time_decode.py $TD -- $ARGS
  echo "td7-$v rc=$(cat /opt/kiln/logs/td7-$v.log.rc)" >> /opt/kiln/logs/dc-td7.txt
done
echo TD7_DONE >> /opt/kiln/logs/dc-td7.txt
