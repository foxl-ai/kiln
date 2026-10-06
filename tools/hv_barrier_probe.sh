#!/bin/bash
# The fixed per-execution cost of a graph holding a collective, with the Neuron runtime's per-execution barrier on
# (default), off (NEURON_RT_DISABLE_EXECUTION_BARRIER=1: vllm-neuron 0.24 neuron_worker.py:715-718 and nkipy
# 1089b54 set it) and in its hardware form (NEURON_RT_ENABLE_HW_EXECUTION_BARRIER=1; both names are strings in
# libnrt.so.1 of aws-neuronx-runtime-lib 2.34.10, SDK 2.32). tools/profile_layer.py --allreduce at 32 ranks.
#
#   bash tools/hv_barrier_probe.sh [variants...]   # default: base off hw
set -u
SRC="${KILN_HV_SRC:-/opt/kiln/src-hv}"
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=$SRC
L=/opt/kiln/logs; mkdir -p $L /opt/kiln/work; cd /opt/kiln/work
BOX="${KILN_HV_BOX:-kiln-hv-32}"
variants="${*:-base off hw}"
for v in $variants; do
  case "$v" in
    base) env="NEURON_RT_DISABLE_EXECUTION_BARRIER=0" ;;  # explicit: the engine and profile_layer now default to 1
    off) env="NEURON_RT_DISABLE_EXECUTION_BARRIER=1" ;;
    hw) env="NEURON_RT_DISABLE_EXECUTION_BARRIER=0 NEURON_RT_ENABLE_HW_EXECUTION_BARRIER=1" ;;
    *) echo "unknown variant $v"; continue ;;
  esac
  log=$L/hv-barrier-$v.log
  cmd="env $env KILN_PROBE_COLLECTIVES=1 python $SRC/tools/profile_layer.py --allreduce --ranks 32 --batch 4"
  echo "$cmd" > $log.cmd
  eval "$cmd" > $log 2>&1; echo $? > $log.rc
  for f in $log $log.cmd $log.rc; do aws s3 cp --quiet $f s3://<your-bucket>/logs/$BOX/ --region us-east-2; done
  sleep 60  # a 32-rank job right after another can fail nrt_init (CLAUDE.md)
done
echo HV-BARRIER-DONE
