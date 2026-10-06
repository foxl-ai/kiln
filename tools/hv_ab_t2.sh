#!/bin/bash
# The execution-barrier serving A/B on trn2.48xlarge (one tp=32 engine on logical cores 32-63 of an LNC=2 box): the
# trn2 agent's single-engine baseline (engine-v0 70ddc1b, farm queue q/t2f-base, its log
# s3://<your-bucket>/logs/kiln-t2-cb/20261005T163035Z-t2-e1-base.log.cmd), at one concurrency, with the
# variant's extra environment on top.
#   bash tools/hv_ab_t2.sh <tag> <concurrency> [VAR=value ...]
set -u
tag="$1"; conc="$2"; shift 2
SRC="${KILN_HV_SRC:-/opt/kiln/src-hv}"
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=$SRC
L=/opt/kiln/logs; mkdir -p $L /opt/kiln/work; cd /opt/kiln/work
BOX="${KILN_HV_BOX:-kiln-hv-t2}"  # logs under logs/kiln-hv-t2/, apart from the box owner's
ENV="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=0 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=6 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/t2f-base/ NEURON_LIBTORCH_ASSERT_CACHE_HIT=1"
ARGS="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 128 --concurrency $conc --requests 128 --decode-buckets 4,8,16,32 --kv-cache-gb 2.75 --kv-cache-dtype fp8 --core-base 32 --price trn2.48xlarge-spot=15.09"
log=$L/hv-t2-$tag-c$conc.log
cmd="env $ENV $* PYTHONPATH=$SRC python $SRC/bench/serve_sweep.py $ARGS"
echo "$cmd" > $log.cmd
eval "$cmd" > $log 2>&1; echo $? > $log.rc
for f in $log $log.cmd $log.rc; do aws s3 cp --quiet $f s3://<your-bucket>/logs/$BOX/ --region us-east-2; done
echo "HV-T2-DONE $tag c$conc rc=$(cat $log.rc)"
