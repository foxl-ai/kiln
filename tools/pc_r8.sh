#!/bin/bash
# feat/prefill-compute: one arm of the 1M R8 A/B on a trn1.32xlarge. A lone 1,044,480-token real-text request
# (tools/lc_ttft.py, the q/lc2-r8 engine verbatim: lc2-ttft-R8-text's env and arguments) gives TTFT and the device
# split's prefill call over its 255 calls. Then optionally lc2's long-context gates on the same engine config:
# check_long needle (128k, 1M; 6/6 at baseline) and the 128k NLL band with KILN_PLP_VP=1 (tools/nll_compare.py against
# s3 logs/kiln-lc2-32/lc2-nllvp-R8.json).
#     KILN_BOX_SRC=/opt/kiln/src-pc bash tools/pc_r8.sh <arm> "<extra env>" <farm queue> [ttft] [needle] [nll]
#     e.g. bash tools/pc_r8.sh fnu "KILN_KDA_FUSED_NORM=1 KILN_DELTA_RULE_UNITS=6" pcx-fn ttft needle nll
# Results: s3 logs/<box>/pcr8-<arm>-{ttft,needle,nll}.{log,json} (+ .cmd, .rc) and /opt/kiln/logs/pc-r8.txt.
set -uo pipefail
BOX=${PC_BOX:-kiln-pcf-32}
export KILN_BOX_SRC=${KILN_BOX_SRC:-/opt/kiln/src-pc}
cd $KILN_BOX_SRC
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=$KILN_BOX_SRC
B3=s3://<your-bucket>
L=/opt/kiln/logs
arm=$1; extra=$2; farm=$3; shift 3
BASE="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_DSA_CP=1 KILN_MOE_EP=1 KILN_DSA_CP_SLOT_CLASSES=1 KILN_DSA_CP_DEGREE=8 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=$B3/compile-farm/q/$farm/ $extra"
A1="--model zai-org/GLM-5.3-Flash --tp 32 --dp-attention 1 --piecewise --overlap --page-size 256 --kv-cache-gb 1.5 --kv-cache-dtype fp8 --prefill-tokens 4096 --max-num-seqs 4 --page-buckets 1024 4096 --max-model-len 1048576 --decode-buckets 1 --state-checkpoints 0 --text-file /opt/kiln/data/long.txt"
SA="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 1 --piecewise --overlap --page-size 256 --max-model-len 1048576 --page-buckets 1024,4096 --warmup --max-seconds 7200 --prefill-tokens 4096 --prefill-buckets 4096 --max-num-seqs 4 --decode-buckets 1 --kv-cache-gb 1.5 --kv-cache-dtype fp8 --state-checkpoints 0 --output-len 128"
[ -s /opt/kiln/data/long.txt ] || { mkdir -p /opt/kiln/data; aws --region us-east-2 s3 cp --quiet $B3/data/lc-long.txt /opt/kiln/data/long.txt; }
for what in "$@"; do
  t=pcr8-$arm-$what
  case $what in
    ttft) bash tools/dc_run.sh $BOX $t "$BASE" tools/lc_ttft.py --lengths 1044480 --warm-lengths 8192 \
            --text-file /opt/kiln/data/long.txt --out-json $L/$t.json -- $SA ;;
    needle) bash tools/dc_run.sh $BOX $t "$BASE" tools/check_long.py needle $A1 --lengths 131072 1044480 \
              --depths 0.1 0.5 0.9 --out-json $L/$t.json ;;
    nll) bash tools/dc_run.sh $BOX $t "$BASE KILN_PLP_VP=1" tools/check_long.py nll $A1 --max-tokens 131072 \
           --out-json $L/$t.json
         aws --region us-east-2 s3 cp --quiet $B3/logs/kiln-lc2-32/lc2-nllvp-R8.json $L/lc2-nllvp-R8.json
         python tools/nll_compare.py $L/$t.json $L/lc2-nllvp-R8.json > $L/$t.compare.txt 2>&1 ;;
    *) echo "unknown $what" >&2; continue ;;
  esac
  for f in $L/$t.json $L/$t.compare.txt; do [ -f $f ] && aws --region us-east-2 s3 cp --quiet $f $B3/logs/$BOX/; done
  echo "$(date -u +%H:%M:%SZ) $t rc=$(cat $L/$t.log.rc) $(grep -h 'RESULT' $L/$t.log | tail -1 | cut -c1-300)" >> $L/pc-r8.txt
  aws --region us-east-2 s3 cp --quiet $L/pc-r8.txt $B3/logs/$BOX/
done
