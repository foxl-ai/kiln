#!/bin/bash
# Upstream-harvest A/B on one trn1.32xlarge: the final G64 / F0 / G16 serving commands of docs/price-performance.md
# (engine-v0 f70c14b graphs, farm queue q/final-f70c14b unless KILN_HV_FARM names another), each run under
# NEURON_LIBTORCH_ASSERT_CACHE_HIT=1, with the variant's extra environment on top.
#
#   bash tools/hv_ab.sh <G64|F0|G16> <tag> [VAR=value ...]
#
# Log /opt/kiln/logs/hv-<config>-<tag>.log with .cmd (the exact command) and .rc, copied to
# s3://<your-bucket>/logs/<box>/ (KILN_HV_BOX, default kiln-hv-32).
set -u
cfg="$1"; tag="$2"; shift 2
SRC="${KILN_HV_SRC:-/opt/kiln/src-hv}"
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=$SRC
L=/opt/kiln/logs; mkdir -p $L /opt/kiln/work; cd /opt/kiln/work
BOX="${KILN_HV_BOX:-kiln-hv-32}"
FARM="${KILN_HV_FARM:-s3://<your-bucket>/compile-farm/q/final-f70c14b/}"
BASE_ENV="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=$FARM"
COMMON="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024"
case "$cfg" in
  G64) SHAPE="--max-num-seqs 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8 --requests 128 --max-seconds 3000 --price trn1.32xlarge-spot=2.15 --concurrency 64" ;;
  F0) SHAPE="--max-num-seqs 32 --decode-buckets 8 --kv-cache-gb 1.5 --requests 128 --max-seconds 3000 --price trn1.32xlarge-spot=2.15 --concurrency 32" ;;
  G16) SHAPE="--max-num-seqs 16 --decode-buckets 4 --kv-cache-gb 0.65 --requests 128 --max-seconds 3000 --price trn1.32xlarge-spot=2.15 --concurrency 16" ;;
  # G64 + EPLB (ab-fin3-G64-EPLB.log.cmd): with KILN_HV_FARM=.../q/final-25a45c9/ and the variant env
  # KILN_EP_REDUNDANT=1 KILN_EPLB_INIT=/opt/kiln/eplb-init-random.pt (s3 logs/kiln-tq-cpu/eplb-init-random.pt)
  G64E) SHAPE="--max-num-seqs 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8 --requests 128 --max-seconds 3000 --price trn1.32xlarge-spot=2.15 --concurrency 64 64 --eplb-rebalance" ;;
  *) echo "config must be G64, F0, G16 or G64E" >&2; exit 2 ;;
esac
log=$L/hv-$cfg-$tag.log
cmd="env $BASE_ENV $* PYTHONPATH=$SRC python $SRC/bench/serve_sweep.py $COMMON $SHAPE"
echo "$cmd" > $log.cmd
eval "$cmd" > $log 2>&1; echo $? > $log.rc
for f in $log $log.cmd $log.rc; do aws s3 cp --quiet $f s3://<your-bucket>/logs/$BOX/ --region us-east-2; done
echo "HV-DONE $cfg $tag rc=$(cat $log.rc)"
