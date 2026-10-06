#!/bin/bash
# trn1 decode-only A/B at large batch, one tree (KILN_BOX_SRC), q/dc1-t1 graphs: STL (ST, 32 / 48 rows per DP group,
# 128 / 192 rows per step, every MoE call of 192 rows as moe_dedupe v8's 128 + 64) against STL9 (the same with
# KILN_MOE_DEDUPE_MAX_TOKENS=256: one kiln_moe_dedupe_v9 call per 192-row MoE).
#     KILN_BOX_SRC=/opt/kiln/src-v9a bash tools/dc_td8.sh STL STL9 STL ...   (STL9b / STL9c: tools/dc_queue14.sh's)
# Logs td8-<v>-<i>.log, or td<DC_TDN>-... with DC_TDN set.
set -uo pipefail
BOX=${DC_BOX:-kiln-dc-32}
cd ${KILN_BOX_SRC:-/opt/kiln/src}
E0="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/dc1-t1/ KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 192 --kv-cache-gb 0.5 --kv-cache-dtype fp8 --no-prefix-caching --decode-buckets 32,48"
TD="--all-buckets --steps 64 --skip 8 --pieces --price trn1.32xlarge-spot=2.15"
i=0
for v in "$@"; do
  i=$((i + 1)); X=""
  AA="$A"; T2="$TD"
  case $v in
    STL9|STL9b|STL9d) X="KILN_MOE_DEDUPE_MAX_TOKENS=256" ;;
    STL9c|STL9e) X="KILN_MOE_DEDUPE_MAX_TOKENS=256 KILN_MOE_DEDUPE_V9=1" ;;  # the 128-row calls on v9 too
    STL9r2)  # tools/dc_queue16.sh: 28 rows per group with real 8K KV, timed with distinct real pages
      X="KILN_MOE_DEDUPE_MAX_TOKENS=256 KILN_MOE_DEDUPE_V9=1"
      AA="${A/--max-num-seqs 192 --kv-cache-gb 0.5/--max-num-seqs 112 --kv-cache-gb 2.1}"; AA="${AA/--decode-buckets 32,48/--decode-buckets 28}"
      T2="$TD --real-kv" ;;
  esac
  N=td${DC_TDN:-8}-$v-$i
  bash tools/dc_run.sh $BOX $N "$E0 $X" tools/time_decode.py $T2 -- $AA
  echo "$N rc=$(cat /opt/kiln/logs/$N.log.rc) $(grep -h -E 'rows|tok/s' /opt/kiln/logs/$N.log | tail -3 | tr '\n' ' ' | cut -c1-400)" >> /opt/kiln/logs/dc-td8.txt
done
echo TD8_DONE >> /opt/kiln/logs/dc-td8.txt
