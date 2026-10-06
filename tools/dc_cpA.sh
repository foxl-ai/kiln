#!/bin/bash
# The CP decode box A/B on one tree (feat/decode-scale-f997 >= 0079633, engine-v0 2143e0b merged in): CP-64 base,
# + KILN_DSA_CP_ALL_LOCAL=1 (A: the local list without a selection), + KILN_DSA_CP_PAGE_KEYS=1 (Ak: page-row pool keys);
# CP-96 (v10) the same. Every variant is captured and compiled on this box (q/dc1-t1), then timed with real KV
# (time_decode --real-kv), the variants in turn and twice, so drift shows as a pass-to-pass difference.
#     KILN_BOX_SRC=/opt/kiln/src-m2143 bash tools/dc_cpA.sh R64 R64A R64Ak
set -uo pipefail
BOX=${DC_BOX:-kiln-dc-32}
export KILN_BOX_SRC=${KILN_BOX_SRC:-/opt/kiln/src-m2143}
cd $KILN_BOX_SRC
L=/opt/kiln/logs
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1 KILN_MOE_DEDUPE_V9=1 KILN_DSA_CP=1"
A0="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --kv-cache-dtype fp8 --no-prefix-caching --page-size 256 --page-buckets 64"
TD="--real-kv --all-buckets --steps 64 --skip 8 --price trn1.32xlarge-spot=2.15"
cfg() {
  case $1 in
    R64*) X="KILN_MOE_DEDUPE_MAX_TOKENS=256"; G="--max-num-seqs 256 --kv-cache-gb 0.62 --decode-buckets 64" ;;
    R96*) X="KILN_MOE_DEDUPE_MAX_TOKENS=512"; G="--max-num-seqs 384 --kv-cache-gb 0.92 --decode-buckets 96" ;;
    *) return 1 ;;
  esac
  case $1 in
    R64|R96) ;;
    *Ak) X="$X KILN_DSA_CP_ALL_LOCAL=1 KILN_DSA_CP_PAGE_KEYS=1" ;;
    *A) X="$X KILN_DSA_CP_ALL_LOCAL=1" ;;
    *k) X="$X KILN_DSA_CP_PAGE_KEYS=1" ;;
    *) return 1 ;;
  esac
}
for v in "$@"; do
  cfg $v || { echo "unknown $v" >&2; exit 2; }
  N=cpA-$v
  bash tools/dc_capture.sh dc1-t1 $N trn1 "$E1 $X" $A0 $G > $L/dcq-$N.out 2>&1
  echo "$N rc=$? $(tail -1 $L/dcq-$N.out)" >> $L/dc-cpA.txt
done
(export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root PYTHONPATH=$KILN_BOX_SRC
 cd /opt/kiln/work && python $KILN_BOX_SRC/tools/compile_farm.py work --queue s3://<your-bucket>/compile-farm/q/dc1-t1/ \
   --mem-budget-gb 300 --linger 30 --out /opt/kiln/work/farm-cpA-$1.jsonl > $L/farm-cpA-$1.log 2>&1)
echo "compiled $(grep -h summary /opt/kiln/work/farm-cpA-$1.jsonl | tail -1 | cut -c1-200)" >> $L/dc-cpA.txt
E0="$E1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/dc1-t1/"
for pass in 1 2; do
  for v in "$@"; do
    cfg $v
    N=tdA-$v-$pass
    bash tools/dc_run.sh $BOX $N "$E0 $X" tools/time_decode.py $TD -- $A0 $G
    echo "$N rc=$(cat $L/$N.log.rc) $(grep -h curve $L/$N.log | tail -1 | cut -c1-300)" >> $L/dc-cpA.txt
  done
done
echo CPA_DONE "$@" >> $L/dc-cpA.txt
