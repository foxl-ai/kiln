#!/bin/bash
# Greedy equality of asynchronous MTP drafting (KILN_SPEC_ASYNC=1) with the synchronous MTP engine on the device, the
# G64 graphs of tools/amtp_g1b.sh: tools/check_mtp.py, 16 prompts x 256 tokens per set (random, wikitext, chat), MS
# writes the reference, MA compares token for token (and reports acceptance by position).
#     bash tools/amtp_check.sh [MS MA]
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf
SRC=${KILN_SRC:-/opt/kiln/src-amtp}
export PYTHONPATH=$SRC
cd $SRC
L=/opt/kiln/logs
W=/opt/kiln/work
export KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1
export KILN_COMPILE_FARM=${KILN_COMPILE_FARM:-s3://<your-bucket>/compile-farm/q/amtp-9692e97/}
ARGS="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 3000 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64 --concurrency 64 --decode-buckets 16 --kv-cache-gb 1.2 --kv-cache-dtype fp8 --spec-method mtp --spec-k 1 --state-checkpoints 16"
for which in ${@:-MS MA}; do
  case $which in
    MS) E="KILN_SPEC_ASYNC=0"; J="--out-json $W/amtp-check-MS.json";;
    MA) E="KILN_SPEC_ASYNC=1"; J="--out-json $W/amtp-check-MA.json --reference-json $W/amtp-check-MS.json";;
  esac
  echo "env $E python tools/check_mtp.py --sets random,wikitext,chat --n 16 --gen 256 $J -- $ARGS  (tree $(cat $SRC/.tree 2>/dev/null))" > $L/amtp-check-$which.log.cmd
  env $E python tools/check_mtp.py --sets random,wikitext,chat --n 16 --gen 256 $J -- $ARGS > $L/amtp-check-$which.log 2>&1
  echo $? > $L/amtp-check-$which.log.rc
  for f in $L/amtp-check-$which.log $L/amtp-check-$which.log.cmd $W/amtp-check-$which.json; do [ -f $f ] && aws s3 cp --quiet $f s3://<your-bucket>/logs/kiln-tq-32/; done
  sleep 120
done
echo AMTP_CHECK_DONE
