#!/bin/bash
# EPLB A/B on one trn1.32xlarge: A (G64 CK4 KV1.2 fp8) then B (A + one redundant slot), 128 requests per level.
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=/opt/kiln/src
cd /opt/kiln/src
L=/opt/kiln/logs
export KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1
export KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/eplb-a2b1414/
ARGS="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64 --decode-buckets 16 --kv-cache-gb 1.2 --kv-cache-dtype fp8 --state-checkpoints 4 --requests 128"
which="${1:-AB}"
if [[ $which == *A* ]]; then
  echo "env KILN_EP_REDUNDANT unset; python bench/serve_sweep.py $ARGS --concurrency 64" > $L/ab-A.log.cmd
  python bench/serve_sweep.py $ARGS --concurrency 64 > $L/ab-A.log 2>&1; echo $? > $L/ab-A.log.rc
  aws s3 cp --quiet $L/ab-A.log s3://<your-bucket>/logs/kiln-tq-32/; aws s3 cp --quiet $L/ab-A.log.cmd s3://<your-bucket>/logs/kiln-tq-32/
  sleep 120
fi
if [[ $which == *B* ]]; then
  export KILN_EP_REDUNDANT=1 KILN_EPLB_INIT=/opt/kiln/eplb-init-random.pt
  echo "env KILN_EP_REDUNDANT=1 KILN_EPLB_INIT=/opt/kiln/eplb-init-random.pt python bench/serve_sweep.py $ARGS --concurrency 64 64 --eplb-rebalance --eplb-dump $L/eplb-B" > $L/ab-B.log.cmd
  python bench/serve_sweep.py $ARGS --concurrency 64 64 --eplb-rebalance --eplb-dump $L/eplb-B > $L/ab-B.log 2>&1; echo $? > $L/ab-B.log.rc
  aws s3 cp --quiet $L/ab-B.log s3://<your-bucket>/logs/kiln-tq-32/; aws s3 cp --quiet $L/ab-B.log.cmd s3://<your-bucket>/logs/kiln-tq-32/
  for f in $L/eplb-B.L*.pt; do aws s3 cp --quiet $f s3://<your-bucket>/logs/kiln-tq-32/; done
fi
echo AB_DONE
