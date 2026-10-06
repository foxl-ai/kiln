#!/bin/bash
# G1b (cost per request with prompt caching): conc 64, 75% of every 8192-token prompt one of 4 shared 6144-token
# prefixes, a cold level then a warm one (--keep-cache), 256 requests per level; A plain EP, B one redundant slot
# (copies from KILN_EPLB_INIT). The G64 KV1.2 CK4 graphs of tools/eplb_ab.sh (the 4 junction checkpoints fit CK4).
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=/opt/kiln/src
cd /opt/kiln/src
L=/opt/kiln/logs
export KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1
export KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/eplb-a2b1414/
ARGS="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 3000 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64 --decode-buckets 16 --kv-cache-gb 1.2 --kv-cache-dtype fp8 --state-checkpoints 4 --concurrency 64 --requests 256 --shared-prefix-len 6144 6144 --num-prefixes 4 --keep-cache"
for which in ${@:-A B}; do
  E=; [ $which = B ] && E="KILN_EP_REDUNDANT=1 KILN_EPLB_INIT=/opt/kiln/eplb-init-random.pt"
  echo "env $E python bench/serve_sweep.py $ARGS" > $L/g1b-$which.log.cmd
  env $E python bench/serve_sweep.py $ARGS > $L/g1b-$which.log 2>&1; echo $? > $L/g1b-$which.log.rc
  aws s3 cp --quiet $L/g1b-$which.log s3://<your-bucket>/logs/kiln-tq-32/; aws s3 cp --quiet $L/g1b-$which.log.cmd s3://<your-bucket>/logs/kiln-tq-32/
  sleep 120
done
echo G1B_DONE
