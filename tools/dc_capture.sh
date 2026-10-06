#!/bin/bash
# Capture one decode configuration on a CPU farm host and put its graphs on a farm queue.
#
#   bash tools/dc_capture.sh <queue> <config> <target> "<env assignments>" <serve_sweep args...>
#
# <queue> is a name under s3://.../compile-farm/q/, <config> names the capture (its config.json and
# keys.json land in <queue>/configs/), <target> is trn1 or trn2 (the cache prefix follows it). Decode
# graphs only (--skip-prefill: what tools/time_decode.py builds), one rank per DP-attention group
# (ranks 0,8,16,24 at tp=32 / DP attention 4), as the earlier decode queues were captured.
# KILN_CAP_PREFILL=1 captures the prefill graphs too (a serving run), KILN_CAP_RANKS overrides the ranks.
set -euo pipefail
Q="$1"; NAME="$2"; TARGET="$3"; ENVS="$4"; shift 4
B=s3://<your-bucket>
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=${KILN_BOX_SRC:-/opt/kiln/src}
case "$TARGET" in
  trn1) CACHE=$B/compile-cache/trn1-sdk2.32/lnl/ ;;
  trn2) CACHE=$B/compile-cache/trn2-sdk2.32/lnl/; export NEURON_LOGICAL_NC_CONFIG=2 ;;
  *) echo "target trn1 or trn2" >&2; exit 2 ;;
esac
SHAPES=/opt/kiln/shapes/glm53
if [ ! -f $SHAPES/kiln-load-decisions.json ]; then
  mkdir -p $SHAPES && aws --region us-east-2 s3 sync --quiet $B/compile-farm/shapes/glm53/ $SHAPES/
fi
OUT=/opt/kiln/work/cap-$NAME
mkdir -p /opt/kiln/logs /opt/kiln/work && cd /opt/kiln/work
SKIP=--skip-prefill; [ "${KILN_CAP_PREFILL:-0}" = 1 ] && SKIP=
env $ENVS python ${KILN_BOX_SRC:-/opt/kiln/src}/tools/compile_farm.py capture --shape-dir $SHAPES --ranks "${KILN_CAP_RANKS:-0,8,16,24}" \
  --target "$TARGET" --out-dir "$OUT" $SKIP -- "$@" > /opt/kiln/logs/cap-$NAME.log 2>&1
tail -2 /opt/kiln/logs/cap-$NAME.log
aws --region us-east-2 s3 cp --quiet "$OUT/config.json" "$B/compile-farm/q/$Q/configs/$NAME.config.json"
aws --region us-east-2 s3 cp --quiet "$OUT/keys.json" "$B/compile-farm/q/$Q/configs/$NAME.keys.json"
python ${KILN_BOX_SRC:-/opt/kiln/src}/tools/compile_farm.py enqueue --queue "$B/compile-farm/q/$Q/" --cache "$CACHE" --keys-file "$OUT/keys.json"
