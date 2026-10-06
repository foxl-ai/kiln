#!/bin/bash
set -uo pipefail
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root PYTHONPATH=/opt/kiln/src
mkdir -p /opt/kiln/prof && cd /opt/kiln/work
L=/opt/kiln/logs
for cfg in "256 192" "128 128"; do set -- $cfg
  KILN_MOE_DEDUPE_MAX_TOKENS=$1 python /opt/kiln/src/tools/probe_dedupe_prof.py --rows $2 --out /opt/kiln/prof/dd$2.pt > $L/ddprof-$2.log 2>&1
  k=$(grep -o "keys \[.*\]" $L/ddprof-$2.log | grep -o "[0-9a-f]\{32\}" | tail -1)
  python /opt/kiln/src/tools/prof_engines.py $k /opt/kiln/prof/dd$2.pt --window 0.5 0.52 --max-lines 150 >> $L/ddprof-$2.log 2>&1
  aws --region us-east-2 s3 cp --quiet $L/ddprof-$2.log s3://<your-bucket>/logs/kiln-dc-k1/
done
echo DDPROF_DONE >> $L/v9.txt
