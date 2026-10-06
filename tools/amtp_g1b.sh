#!/bin/bash
# Asynchronous MTP drafting (KILN_SPEC_ASYNC=1, engine/spec_async.py) on the decode-heavy G1b workload, one
# trn1.32xlarge: conc 64, 75% of every 8192-token prompt one of 4 shared 6144-token prefixes, a cold level then a warm
# one (--keep-cache), 256 requests per level, EP, KV 1.2 fp8. MB = no MTP (the default 32 checkpoint rows), MS = MTP
# k=1 stepped synchronously, MA = MTP k=1 under overlap scheduling (16 checkpoint rows hold the KDA state pool at
# MB's 49 rows per group, so every target graph is MB's). Graphs: q/amtp-9692e97 (configs
# G64-4096-KV1.2-S20-P12-K-EPT-MB / -MA; MS runs MA's graphs without the board ones).
#     bash tools/amtp_g1b.sh [MB MS MA]
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf
SRC=${KILN_SRC:-/opt/kiln/src-amtp}
export PYTHONPATH=$SRC
cd $SRC
L=/opt/kiln/logs
export KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1
export KILN_COMPILE_FARM=${KILN_COMPILE_FARM:-s3://<your-bucket>/compile-farm/q/amtp-9692e97/}
ARGS="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 3000 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64 --concurrency 64 --decode-buckets 16 --kv-cache-gb 1.2 --kv-cache-dtype fp8 --requests 256 --shared-prefix-len 6144 6144 --num-prefixes 4 --keep-cache"
MTP="--spec-method mtp --spec-k 1 --state-checkpoints 16"
for which in ${@:-MB MS MA}; do
  case $which in MB) E=""; X="";; MS) E="KILN_SPEC_ASYNC=0"; X="$MTP";; MA) E="KILN_SPEC_ASYNC=1"; X="$MTP";; esac
  echo "env $E python bench/serve_sweep.py $ARGS $X  (tree $(cat $SRC/.tree 2>/dev/null))" > $L/amtp-$which.log.cmd
  env $E python bench/serve_sweep.py $ARGS $X > $L/amtp-$which.log 2>&1; echo $? > $L/amtp-$which.log.rc
  for f in $L/amtp-$which.log $L/amtp-$which.log.cmd; do aws s3 cp --quiet $f s3://<your-bucket>/logs/kiln-tq-32/; done
  sleep 120
done
echo AMTP_G1B_DONE
