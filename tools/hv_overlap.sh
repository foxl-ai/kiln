#!/bin/bash
# NxD Inference's collective/compute overlap compiler options against Kiln's serving flags, through
# tools/probe_overlap.py at 32 ranks (the per-execution barrier left ON, the runtime default). The options are what
# neuronx-distributed-inference models/model_wrapper.py:85-107 (4bcdc54) passes for every context-encoding graph
# (--cc-pipeline-tiling-factor 2) and token-generation graph (1):
#   --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=N --vectorize-strided-dma'
#   bash tools/hv_overlap.sh [variants...]   # default: base ov2 ov2v ov1v
set -u
SRC="${KILN_HV_SRC:-/opt/kiln/src-hv}"
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=$SRC
L=/opt/kiln/logs; mkdir -p $L /opt/kiln/work; cd /opt/kiln/work
BOX="${KILN_HV_BOX:-kiln-hv-32b}"
for v in ${*:-base ov2 ov2v ov1v}; do
  case "$v" in
    base) cc="--model-type=transformer" ;;
    ov2) cc="--model-type=transformer '--tensorizer-options=--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'" ;;
    ov2v) cc="--model-type=transformer '--tensorizer-options=--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 --vectorize-strided-dma'" ;;
    ov1v) cc="--model-type=transformer '--tensorizer-options=--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=1 --vectorize-strided-dma'" ;;
    ov4) cc="--model-type=transformer '--tensorizer-options=--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=4'" ;;
    *) echo "unknown variant $v"; continue ;;
  esac
  log=$L/hv-ov-$v.log
  cmd="env NEURON_RT_DISABLE_EXECUTION_BARRIER=0 KILN_CC_ARGS=\"$cc\" python $SRC/tools/probe_overlap.py --ranks 32"
  echo "$cmd" > $log.cmd
  # In its own process group (job control: the job's group id is its pid), so that what is left of it (its spawned
  # ranks) can be killed without touching any other job on a shared box, which a pkill by pattern would.
  set -m
  bash -c "$cmd" > $log 2>&1 &
  pid=$!; wait $pid; echo $? > $log.rc
  kill -9 -- -$pid 2>/dev/null
  set +m
  for f in $log $log.cmd $log.rc; do aws s3 cp --quiet $f s3://<your-bucket>/logs/$BOX/ --region us-east-2; done
  echo "OV-DONE $v rc=$(cat $log.rc)"
  sleep 75
done
echo HV-OV-DONE
