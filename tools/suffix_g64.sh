#!/bin/bash
# Suffix decoding (vLLM v0.30.0 v1/spec_decode/suffix_decoding.py; Kiln engine/spec_suffix.py) at G1 conc 64, k=1: plain EP, KV 1.2 CK4.
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=/opt/kiln/src
cd /opt/kiln/src
L=/opt/kiln/logs
export KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1
export KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/eplb-a2b1414/
ARGS="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64 --decode-buckets 16 --kv-cache-gb 1.2 --kv-cache-dtype fp8 --state-checkpoints 4 --requests 128 --spec-method suffix --spec-k 1"
which="${1:-A}"
if [[ $which == *A* ]]; then
  echo "python bench/serve_sweep.py $ARGS --concurrency 64" > $L/suf-A.log.cmd
  python bench/serve_sweep.py $ARGS --concurrency 64 > $L/suf-A.log 2>&1; echo $? > $L/suf-A.log.rc
  aws s3 cp --quiet $L/suf-A.log s3://<your-bucket>/logs/kiln-tq-32/; aws s3 cp --quiet $L/suf-A.log.cmd s3://<your-bucket>/logs/kiln-tq-32/
  sleep 120
fi
echo AB_DONE
