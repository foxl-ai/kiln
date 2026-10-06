#!/bin/bash
# Quality gates for an upstream-harvest variant, the commands of docs/neuron-notes.md "The final combined measurement"
# (kiln-ak-32's cm-def-long / cm-def-wiki / ppl-wt-def .log.cmd), with the variant's extra environment on top:
#   long  tools/check_mixed.py on LONG_TEXT through the G64 serving graphs (32 prompts, 64 greedy tokens, top-2)
#   wiki  the same on wikitext-2 prompts
#   ppl   tools/check_ppl.py wikitext-2 at DP attention 4
#
#   bash tools/hv_quality.sh <long|wiki|ppl> <tag> [VAR=value ...]
set -u
what="$1"; tag="$2"; shift 2
SRC="${KILN_HV_SRC:-/opt/kiln/src-hv}"
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=$SRC
L=/opt/kiln/logs; mkdir -p $L /opt/kiln/work; cd /opt/kiln/work
BOX="${KILN_HV_BOX:-kiln-hv-32}"
FARM="${KILN_HV_FARM:-s3://<your-bucket>/compile-farm/q/final-f70c14b/}"
[ -f /opt/kiln/work/wikitext2_test.txt ] || aws s3 cp --quiet s3://<your-bucket>/data/wikitext2_test.txt /opt/kiln/work/ --region us-east-2
BASE_ENV="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=$FARM"
CM="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8 --concurrency 32 --requests 32"
log=$L/hv-q-$what-$tag.log
case "$what" in
  long) cmd="cd $SRC && env $BASE_ENV $* PYTHONPATH=$SRC python tools/check_mixed.py $CM --out $L/hv-q-long-$tag.json" ;;
  wiki) cmd="cd $SRC && env $BASE_ENV $* PYTHONPATH=$SRC python tools/check_mixed.py $CM --text-file /opt/kiln/work/wikitext2_test.txt --out $L/hv-q-wiki-$tag.json" ;;
  ppl) cmd="env $BASE_ENV KILN_MOE_PREFILL_MIN_TOKENS=1 $* PYTHONPATH=$SRC python $SRC/tools/check_ppl.py --model zai-org/GLM-5.3-Flash --tp 32 --dp-attention 4 --piecewise --kv-cache-gb 1.0 --text-file /opt/kiln/work/wikitext2_test.txt --out-json $L/hv-q-ppl-$tag.json" ;;
  *) echo "what must be long, wiki or ppl" >&2; exit 2 ;;
esac
echo "$cmd" > $log.cmd
(eval "$cmd") > $log 2>&1; echo $? > $log.rc
for f in $log $log.cmd $log.rc $L/hv-q-$what-$tag.json; do [ -f $f ] && aws s3 cp --quiet $f s3://<your-bucket>/logs/$BOX/ --region us-east-2; done
echo "HV-Q-DONE $what $tag rc=$(cat $log.rc)"
