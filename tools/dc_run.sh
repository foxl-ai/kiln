#!/bin/bash
# Run one measurement on a device box and keep its exact command beside the log (s3 logs/<box>/<tag>.log + .cmd + .rc).
#
#   bash tools/dc_run.sh <box> <tag> "<env assignments>" <script.py> [args...]
#
# The env string is applied with `env` before python, so it lands in the .cmd verbatim. Waits a minute after the previous
# 32-rank job on the box ended (a 32-rank start right after one exited can fail nrt_init: CLAUDE.md).
set -uo pipefail
BOX="$1"; TAG="$2"; ENVS="$3"; SCRIPT="$4"; shift 4
SRC="${KILN_BOX_SRC:-/opt/kiln/src}"
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=$SRC
L=/opt/kiln/logs
mkdir -p $L /opt/kiln/work && cd /opt/kiln/work
if [ -f $L/.last-end ]; then
  dt=$(( $(date +%s) - $(cat $L/.last-end) )); [ $dt -lt 70 ] && sleep $(( 70 - dt ))
fi
ARGS=$(printf ' %q' "$@")
echo "env $ENVS python $SCRIPT$ARGS" > $L/$TAG.log.cmd
date -u +%Y-%m-%dT%H:%M:%SZ >> $L/$TAG.log.cmd
env $ENVS python $SRC/$SCRIPT "$@" > $L/$TAG.log 2>&1
echo $? > $L/$TAG.log.rc
date +%s > $L/.last-end
for f in $L/$TAG.log $L/$TAG.log.cmd $L/$TAG.log.rc; do aws --region us-east-2 s3 cp --quiet $f s3://<your-bucket>/logs/$BOX/; done
