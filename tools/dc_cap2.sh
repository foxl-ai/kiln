#!/bin/bash
# Where a real-KV decode call goes on trn1 (tools/dc_cap.sh with time_decode --real-kv): one run of the t1-STL9r2 decode
# box (28 rows per group, real 8K KV, v9 everywhere) with every rank's inputs of one warm decode call captured, each NEFF
# replayed on 32 cores with those inputs, rank 0 profiled (util_report replay / bins / report).
set -uo pipefail
cd ${KILN_BOX_SRC:-/opt/kiln/src}
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=$PWD
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/dc1-t1/ KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1 KILN_MOE_DEDUPE_MAX_TOKENS=256 KILN_MOE_DEDUPE_V9=1"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 112 --kv-cache-gb 2.1 --kv-cache-dtype fp8 --no-prefix-caching --decode-buckets 28"
cfg=${DC_CFG:-R28}
N=${DC_CALL:-40}
CAP=/opt/kiln/nvme/cap-$cfg PROF=/opt/kiln/nvme/prof-$cfg
rm -rf $CAP $PROF
[ -f /opt/kiln/shapes/glm53/config.json ] || { mkdir -p /opt/kiln/shapes/glm53; aws --region us-east-2 s3 sync --quiet s3://<your-bucket>/compile-farm/shapes/glm53/ /opt/kiln/shapes/glm53/; }
bash tools/dc_run.sh ${DC_BOX:-kiln-dc-32} cap2-$cfg "$E1 KILN_CAPTURE_INPUTS=$CAP KILN_CAPTURE_AT=decode:$N" \
  tools/time_decode.py --real-kv --steps 48 --skip 8 -- $A
python tools/util_report.py replay $CAP --call decode:$N --out $PROF --keep-ntff > /opt/kiln/logs/rep2-$cfg.log 2>&1
python tools/util_report.py bins $PROF >> /opt/kiln/logs/rep2-$cfg.log 2>&1
python tools/util_report.py report $PROF --kind decode --rows 28 --shape-dir /opt/kiln/shapes/glm53 > /opt/kiln/logs/rep2-$cfg.report.txt 2>&1
for f in /opt/kiln/logs/rep2-$cfg.log /opt/kiln/logs/rep2-$cfg.report.txt; do
  aws --region us-east-2 s3 cp --quiet $f s3://<your-bucket>/logs/${DC_BOX:-kiln-dc-32}/; done
rm -rf $CAP
echo "cap2-$cfg done" >> /opt/kiln/logs/dc-cap.txt
