#!/bin/bash
# The trn1 8K decode box with real KV, one tree, q/dc1-t1 graphs, time_decode --real-kv: R28 (no CP) and the
# context-parallel DSA configs (KILN_DSA_CP=1, 256-token pages, page bucket 64): R48 / R64 / R80 (tools/dc_queue17.sh on the
# scratch merge, tools/dc_queue18.sh on feat/decode-scale-f997), R80v10 / R96v10 with kiln_moe_dedupe_v10
# (KILN_MOE_DEDUPE_MAX_TOKENS=512). R28n / R48n: the same graphs on the null page.
#     KILN_BOX_SRC=/opt/kiln/src-f997 bash tools/dc_tdcp.sh R64 R80v10 R96v10
set -uo pipefail
BOX=${DC_BOX:-kiln-dc-32}
export KILN_BOX_SRC=${KILN_BOX_SRC:-/opt/kiln/src-f997}
cd $KILN_BOX_SRC
E0="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/dc1-t1/ KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1 KILN_MOE_DEDUPE_MAX_TOKENS=256 KILN_MOE_DEDUPE_V9=1"
A0="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --kv-cache-dtype fp8 --no-prefix-caching"
TD="--real-kv --all-buckets --steps 64 --skip 8 --price trn1.32xlarge-spot=2.15"
i=0
for v in "$@"; do
  i=$((i + 1)); X=""
  case $v in
    R28) G="--page-buckets 264 --max-num-seqs 112 --kv-cache-gb 2.1 --decode-buckets 28" ;;
    R48) X="KILN_DSA_CP=1"; G="--page-size 256 --page-buckets 64 --max-num-seqs 192 --kv-cache-gb 0.47 --decode-buckets 48" ;;
    R28n) G="--page-buckets 264 --max-num-seqs 112 --kv-cache-gb 2.1 --decode-buckets 28" ;;
    R48n) X="KILN_DSA_CP=1"; G="--page-size 256 --page-buckets 64 --max-num-seqs 192 --kv-cache-gb 0.47 --decode-buckets 48" ;;
    R80) X="KILN_DSA_CP=1"; G="--page-size 256 --page-buckets 64 --max-num-seqs 320 --kv-cache-gb 0.77 --decode-buckets 80" ;;
    R80v10) X="KILN_DSA_CP=1 KILN_MOE_DEDUPE_MAX_TOKENS=512"; G="--page-size 256 --page-buckets 64 --max-num-seqs 320 --kv-cache-gb 0.77 --decode-buckets 80" ;;
    R96v10) X="KILN_DSA_CP=1 KILN_MOE_DEDUPE_MAX_TOKENS=512"; G="--page-size 256 --page-buckets 64 --max-num-seqs 384 --kv-cache-gb 0.92 --decode-buckets 96" ;;
    R64) X="KILN_DSA_CP=1"; G="--page-size 256 --page-buckets 64 --max-num-seqs 256 --kv-cache-gb 0.62 --decode-buckets 64" ;;
    *) echo "unknown $v" >&2; continue ;;
  esac
  N=tdcp-$v-$i
  T2="$TD"; case $v in R48n|R28n) T2="${TD/--real-kv /}" ;; esac  # the same graphs on the null page
  bash tools/dc_run.sh $BOX $N "$E0 $X" tools/time_decode.py $T2 -- $A0 $G
  echo "$N rc=$(cat /opt/kiln/logs/$N.log.rc) $(grep -h -E 'curve' /opt/kiln/logs/$N.log | tail -2 | tr '\n' ' ' | cut -c1-300)" >> /opt/kiln/logs/dc-tdcp.txt
done
echo TDCP_DONE >> /opt/kiln/logs/dc-tdcp.txt
