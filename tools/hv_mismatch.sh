#!/bin/bash
# tools/probe_mismatch.py over its cases at 32 ranks: barrier off with a short runtime execution timeout, barrier on,
# and barrier off with the runtime's default timeout. Logs /opt/kiln/logs/hv-mm-<tag>.log (+ .cmd), copied to S3.
#   bash tools/hv_mismatch.sh "<tag>:<case>:<wait>:<env> ..."   (env: VAR=v,VAR2=v2 or -)
set -u
SRC="${KILN_HV_SRC:-/opt/kiln/src-hv}"
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=$SRC
L=/opt/kiln/logs; mkdir -p $L /opt/kiln/work; cd /opt/kiln/work
BOX="${KILN_HV_BOX:-kiln-hv-32b}"
for spec in $1; do
  IFS=: read -r tag case wait env <<< "$spec"
  [ "$env" = "-" ] && env="" || env="${env//,/ }"
  log=$L/hv-mm-$tag.log
  cmd="env $env python $SRC/tools/probe_mismatch.py --case $case --ranks 32 --wait $wait --iters ${KILN_HV_ITERS:-3000}"
  echo "$cmd" > $log.cmd
  # In its own process group (job control: the job's group id is its pid), so that what is left of it (its spawned
  # ranks) can be killed without touching any other job on a shared box, which a pkill by pattern would.
  set -m
  bash -c "$cmd" > $log 2>&1 &
  pid=$!; wait $pid; echo $? > $log.rc
  kill -9 -- -$pid 2>/dev/null
  set +m
  for f in $log $log.cmd $log.rc; do aws s3 cp --quiet $f s3://<your-bucket>/logs/$BOX/ --region us-east-2; done
  echo "MM-DONE $tag $(grep -c ' ok ' $log) ok $(grep -c ' raised ' $log) raised $(grep -c ' hung ' $log) hung"
  sleep 75
done
echo HV-MM-DONE
