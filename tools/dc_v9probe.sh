#!/bin/bash
# moe_dedupe at 128-256 rows on one NeuronCore (trn1.2xlarge): 128-token v8 calls against the one-call v9
# (KILN_MOE_DEDUPE_MAX_TOKENS=256), lanes 8 and 16, GLM-5.3-Flash's tp=32 rank shapes.
set -uo pipefail
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root PYTHONPATH=/opt/kiln/src
cd /opt/kiln/work
P="python /opt/kiln/src/tools/probe_moe_kernel.py --experts 288 --scales block128 --act silu_clamp --limit 10 --kernels dedupe --no-xla --iters 20"
L=/opt/kiln/logs
KILN_MOE_DEDUPE_MAX_TOKENS=128 $P --batch 128 192 256 > $L/v9-mt128.log 2>&1; echo "mt128 rc=$?" >> $L/v9.txt
KILN_MOE_DEDUPE_MAX_TOKENS=256 $P --batch 192 256 > $L/v9-mt256.log 2>&1; echo "mt256 rc=$?" >> $L/v9.txt
KILN_MOE_DEDUPE_MAX_TOKENS=256 $P --batch 192 256 --lanes 16 > $L/v9-mt256-l16.log 2>&1; echo "mt256-l16 rc=$?" >> $L/v9.txt
for f in $L/v9*.log $L/v9.txt; do aws --region us-east-2 s3 cp --quiet $f s3://<your-bucket>/logs/kiln-dc-k1/; done
echo V9_DONE >> $L/v9.txt
