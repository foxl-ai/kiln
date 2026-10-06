#!/bin/bash
# Where the CP-64 decode box's step goes on trn1: tools/dc_cap2.sh on q/dc1-t1 cp-R64p (tools/dc_queue19.sh: CP-64 with
# piecewise decode graphs, so each piece's replay loads only its layers' inputs), real 8K KV via time_decode --real-kv,
# every rank's inputs of one warm decode call captured, each piece replayed on 32 cores, rank 0 profiled.
set -uo pipefail
cd ${KILN_BOX_SRC:-/opt/kiln/src}
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=$PWD
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/dc1-t1/ KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=0 KILN_MOE_DEDUPE_V9=1 KILN_DSA_CP=1 KILN_MOE_DEDUPE_MAX_TOKENS=256"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --kv-cache-dtype fp8 --no-prefix-caching --page-size 256 --page-buckets 64 --max-num-seqs 256 --kv-cache-gb 0.62 --decode-buckets 64"
cfg=${DC_CFG:-R64p}
N=${DC_CALL:-40}
CAP=/opt/kiln/nvme/cap-$cfg PROF=/opt/kiln/nvme/prof-$cfg
rm -rf $CAP $PROF
[ -f /opt/kiln/shapes/glm53/config.json ] || { mkdir -p /opt/kiln/shapes/glm53; aws --region us-east-2 s3 sync --quiet s3://<your-bucket>/compile-farm/shapes/glm53/ /opt/kiln/shapes/glm53/; }
bash tools/dc_run.sh ${DC_BOX:-kiln-dc-32} cap2-$cfg "$E1 KILN_CAPTURE_INPUTS=$CAP KILN_CAPTURE_AT=decode:$N" \
  tools/time_decode.py --real-kv --steps 48 --skip 8 -- $A
python tools/util_report.py replay $CAP --call decode:$N --out $PROF --keep-ntff ${DC_REPLAY_ARGS:-} > /opt/kiln/logs/rep2-$cfg.log 2>&1
python tools/util_report.py bins $PROF >> /opt/kiln/logs/rep2-$cfg.log 2>&1
python tools/util_report.py report $PROF --kind decode --rows 256 --shape-dir /opt/kiln/shapes/glm53 > /opt/kiln/logs/rep2-$cfg.report.txt 2>&1
for f in /opt/kiln/logs/rep2-$cfg.log /opt/kiln/logs/rep2-$cfg.report.txt; do
  aws --region us-east-2 s3 cp --quiet $f s3://<your-bucket>/logs/${DC_BOX:-kiln-dc-32}/; done
[ "${DC_KEEP_CAP:-0}" = 1 ] || rm -rf $CAP
if [ -n "${DC_SKEW:-}" ]; then  # every rank profiled (DC_REPLAY_ARGS=--profile-all): the slowest rank per segment
  for f in $PROF/*.summary.json; do t=$(basename $f .summary.json); python tools/prof_skew.py $PROF $t > /opt/kiln/logs/skew-$cfg-$t.txt 2>&1
    aws --region us-east-2 s3 cp --quiet /opt/kiln/logs/skew-$cfg-$t.txt s3://<your-bucket>/logs/${DC_BOX:-kiln-dc-32}/; done
fi
echo "cap2-$cfg done" >> /opt/kiln/logs/dc-cap.txt
