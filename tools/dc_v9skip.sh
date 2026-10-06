#!/bin/bash
# kiln_moe_dedupe_v9's slot tail in device-loop segments, one trn1.2xlarge core, 192 / 256 rows, uniform routing
# (tools/probe_moe_kernel.py's distinct random experts per token). DC_COMBOS: "skip:head:ring ..." (KILN_MOE_DEDUPE_SKIP9
# segment blocks, KILN_MOE_DEDUPE_HEAD9 static head percent, KILN_MOE_DEDUPE_RING9).
set -uo pipefail
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root PYTHONPATH=/opt/kiln/src
cd /opt/kiln/work
P="python /opt/kiln/src/tools/probe_moe_kernel.py --experts 288 --scales block128 --act silu_clamp --limit 10 --kernels dedupe --no-xla --iters 20 --batch ${DC_ROWS:-192 256}"
L=/opt/kiln/logs
for c in ${DC_COMBOS:-0:0:4 2:0:4 4:0:4}; do
  IFS=: read k h r <<< "$c"
  KILN_MOE_DEDUPE_MAX_TOKENS=256 KILN_MOE_DEDUPE_SKIP9=$k KILN_MOE_DEDUPE_HEAD9=$h KILN_MOE_DEDUPE_RING9=$r $P > $L/v9k-$c.log 2>&1
  echo "$c rc=$? $(grep -h -E 'NKI dedupe kernel|first call' $L/v9k-$c.log | tr -s ' ' | sed 's/pairs=[0-9]* distinct=[0-9]*: dedupe kernel graph //' | cut -c1-70 | tr '\n' ';')" >> $L/v9k.txt
done
for f in $L/v9k*; do aws --region us-east-2 s3 cp --quiet $f s3://<your-bucket>/logs/kiln-dc-k1/; done
echo V9K_DONE >> $L/v9k.txt
