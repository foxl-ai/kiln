#!/bin/bash
# moe_dedupe at 256-512 rows on one trn1.2xlarge core: v9's 256-token calls (MAX_TOKENS=256) against one
# kiln_moe_dedupe_v10 call (MAX_TOKENS=512), and v10 at 256 (KILN_MOE_DEDUPE_V10=1) against v9.
set -uo pipefail
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root PYTHONPATH=/opt/kiln/src
cd /opt/kiln/work
P="python /opt/kiln/src/tools/probe_moe_kernel.py --experts 288 --scales block128 --act silu_clamp --limit 10 --kernels dedupe --no-xla --iters 20"
L=/opt/kiln/logs
run() { n=$1; shift; env "$@" $P --batch ${DC_ROWS:-256 320 384 512} > $L/v10p-$n.log 2>&1
  echo "$n rc=$? $(grep -h -E 'NKI dedupe kernel|first call|rel ' $L/v10p-$n.log | tr -s ' ' | sed 's/pairs=[0-9]* distinct=[0-9]*: dedupe kernel graph //' | cut -c1-70 | tr '\n' ';')" >> $L/v10p.txt; }
run v9x KILN_MOE_DEDUPE_MAX_TOKENS=256 KILN_MOE_DEDUPE_V9=1
run v10 KILN_MOE_DEDUPE_MAX_TOKENS=512 KILN_MOE_DEDUPE_V9=1
DC_ROWS="192 256" run v10all KILN_MOE_DEDUPE_MAX_TOKENS=512 KILN_MOE_DEDUPE_V10=1
for f in $L/v10p*; do aws --region us-east-2 s3 cp --quiet $f s3://<your-bucket>/logs/kiln-dc-k1/; done
echo V10P_DONE >> $L/v10p.txt
