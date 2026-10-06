#!/bin/bash
# moe_dedupe v8 against v9 (KILN_MOE_DEDUPE_V9=1) at 16-128 rows, one trn1.2xlarge core, GLM-5.3-Flash's rank shapes.
set -uo pipefail
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root PYTHONPATH=/opt/kiln/src
cd /opt/kiln/work
P="python /opt/kiln/src/tools/probe_moe_kernel.py --experts 288 --scales block128 --act silu_clamp --limit 10 --kernels dedupe --no-xla --iters 20 --batch 16 32 64 128"
L=/opt/kiln/logs
$P > $L/v9s-v8.log 2>&1; echo "v8 $(grep -h 'NKI dedupe kernel' $L/v9s-v8.log | tr -s ' ' | cut -c1-60 | tr '\n' ';')" >> $L/v9s.txt
KILN_MOE_DEDUPE_V9=1 $P > $L/v9s-v9.log 2>&1; echo "v9 $(grep -h 'NKI dedupe kernel' $L/v9s-v9.log | tr -s ' ' | cut -c1-60 | tr '\n' ';')" >> $L/v9s.txt
for f in $L/v9s*; do aws --region us-east-2 s3 cp --quiet $f s3://<your-bucket>/logs/kiln-dc-k1/; done
echo V9S_DONE >> $L/v9s.txt
