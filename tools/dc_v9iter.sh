#!/bin/bash
# One v9 iteration on kiln-dc-k1: the dedupe simulator tests, the 192 / 256-row probe, the 192-row engine profile.
#   bash tools/dc_v9iter.sh <tag>
set -uo pipefail
TAG=$1
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root PYTHONPATH=/opt/kiln/src
L=/opt/kiln/logs
cd /opt/kiln/src
timeout 3000 python -m pytest -q -x tests/test_moe_dedupe.py tests/test_lnc_split.py -k "v9 or dedupe" > $L/v9it-$TAG-tests.log 2>&1
echo "$TAG tests rc=$? $(tail -1 $L/v9it-$TAG-tests.log)" >> $L/v9it.txt
cd /opt/kiln/work
KILN_MOE_DEDUPE_MAX_TOKENS=256 python /opt/kiln/src/tools/probe_moe_kernel.py --experts 288 --scales block128 --act silu_clamp \
  --limit 10 --kernels dedupe --no-xla --iters 20 --batch 192 256 > $L/v9it-$TAG-probe.log 2>&1
echo "$TAG probe rc=$? $(grep -h 'NKI dedupe kernel' $L/v9it-$TAG-probe.log | tr -s ' ' | cut -c1-80 | tr '\n' ';')" >> $L/v9it.txt
KILN_MOE_DEDUPE_MAX_TOKENS=256 python /opt/kiln/src/tools/probe_dedupe_prof.py --rows 192 --out /opt/kiln/prof/dd192-$TAG.pt > $L/v9it-$TAG-prof.log 2>&1
k=$(grep -o "keys \[.*\]" $L/v9it-$TAG-prof.log | grep -o "[0-9a-f]\{32\}" | tail -1)
python /opt/kiln/src/tools/prof_engines.py $k /opt/kiln/prof/dd192-$TAG.pt >> $L/v9it-$TAG-prof.log 2>&1
for e in Vector Scalar GpSimd; do python /opt/kiln/src/tools/prof_ops.py /opt/kiln/prof/eng/$k/profile.json --engine $e --top 12 >> $L/v9it-$TAG-prof.log 2>&1; done
for f in $L/v9it-$TAG-*.log $L/v9it.txt; do aws --region us-east-2 s3 cp --quiet $f s3://<your-bucket>/logs/kiln-dc-k1/; done
echo "$TAG DONE" >> $L/v9it.txt
