#!/bin/bash
# Where a decode call goes on trn1: one time_decode run per serving decode shape with every rank's inputs of one warm
# decode call captured (KILN_CAPTURE_INPUTS, kiln/profiling.py), then each NEFF of that call replayed on 32 cores with
# those inputs, rank 0 profiled (tools/util_report.py replay / bins / report).
#   bash tools/dc_cap.sh <cfg...>      cfg: G16 F0 G64 (the serving decode shapes of tools/dc_td1.sh)
set -uo pipefail
cd /opt/kiln/src
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=/opt/kiln/src
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1"
FARM="KILN_COMPILE_FARM=${DC_FARM:-s3://<your-bucket>/compile-farm/q/dc1-t1/}"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024"
declare -A C R
C[G16]="--max-num-seqs 16 --concurrency 16 --decode-buckets 4 --kv-cache-gb 0.65"; R[G16]=16
C[G1]="--max-num-seqs 16 --concurrency 16 --decode-buckets 1,4 --kv-cache-gb 0.65"; R[G1]=4  # time_decode times bucket 1
C[F0]="--max-num-seqs 32 --concurrency 32 --decode-buckets 8 --kv-cache-gb 1.5"; R[F0]=32
C[G64]="--max-num-seqs 64 --concurrency 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8"; R[G64]=64
N=${DC_CALL:-40}
[ -f /opt/kiln/shapes/glm53/config.json ] || { mkdir -p /opt/kiln/shapes/glm53; aws --region us-east-2 s3 sync --quiet s3://<your-bucket>/compile-farm/shapes/glm53/ /opt/kiln/shapes/glm53/; }
for cfg in "$@"; do
  CAP=/opt/kiln/nvme/cap-$cfg PROF=/opt/kiln/nvme/prof-$cfg
  rm -rf $CAP $PROF
  bash tools/dc_run.sh ${DC_BOX:-kiln-dc-32} cap-$cfg "$E1 $FARM ${DC_EXTRA_ENV:-} KILN_CAPTURE_INPUTS=$CAP KILN_CAPTURE_AT=decode:$N" \
    tools/time_decode.py --steps 48 --skip 8 -- $A ${C[$cfg]}
  python tools/util_report.py replay $CAP --call decode:$N --out $PROF --keep-ntff ${DC_REPLAY_ARGS:-} > /opt/kiln/logs/rep-$cfg.log 2>&1
  python tools/util_report.py bins $PROF >> /opt/kiln/logs/rep-$cfg.log 2>&1
  python tools/util_report.py report $PROF --kind decode --rows ${R[$cfg]} --shape-dir /opt/kiln/shapes/glm53 \
    > /opt/kiln/logs/rep-$cfg.report.txt 2>&1
  for f in /opt/kiln/logs/rep-$cfg.log /opt/kiln/logs/rep-$cfg.report.txt; do
    aws --region us-east-2 s3 cp --quiet $f s3://<your-bucket>/logs/${DC_BOX:-kiln-dc-32}/; done
  aws --region us-east-2 s3 sync --quiet --exclude "*.ntff" --exclude "*.inputs" $PROF s3://<your-bucket>/logs/${DC_BOX:-kiln-dc-32}/prof-$cfg/
  [ "${DC_KEEP_CAP:-0}" = 1 ] || rm -rf $CAP
  echo "cap-$cfg done" >> /opt/kiln/logs/dc-cap.txt
done
