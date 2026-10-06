#!/bin/bash
# The ST gate on one trn1.32xlarge (engine-v0 8229c3d + feat/decode-scale, q/dc2-st graphs, ASSERT_CACHE_HIT):
#   serve-BASE / serve-STb   bench/serve_sweep.py at the G64 shape, conc 64, 128 requests
#   cm-<text>-<cfg>          tools/check_mixed.py (greedy, top-2 logprobs, 32 prompts at conc 32), LONG_TEXT and wikitext-2
# STb = ST (KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1) + KILN_SP_GATHER=xla: with TP experts
# the NKI world gather's prefill pieces fail neuronx-cc 2.27 (NCC_ISCH719), with the zero-padded all-reduce they compile.
#     bash tools/dc_gate.sh [runs...]   (default: serve-BASE serve-STb cm-long-BASE cm-long-BASE2 cm-long-STb cm-wiki-BASE cm-wiki-STb)
set -uo pipefail
BOX=${DC_BOX:-kiln-dc-32}
cd /opt/kiln/src
L=/opt/kiln/logs
E0="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/dc2-st/"
ST="KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1 KILN_SP_GATHER=xla"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 3000 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8"
[ -f /opt/kiln/work/wikitext2_test.txt ] || { mkdir -p /opt/kiln/work; aws s3 cp --quiet s3://<your-bucket>/data/wikitext2_test.txt /opt/kiln/work/wikitext2_test.txt --region us-east-2; }
for run in ${@:-serve-BASE serve-STb cm-long-BASE cm-long-BASE2 cm-long-STb cm-wiki-BASE cm-wiki-STb}; do
  kind=${run%%-*}; rest=${run#*-}
  if [ $kind = serve ]; then
    cfg=$rest; X=""; [ $cfg = STb ] && X="$ST"
    bash tools/dc_run.sh $BOX gate-$run "$E0 $X" bench/serve_sweep.py $A --concurrency 64 --requests 128 --price trn1.32xlarge-spot=2.15
  else
    text=${rest%%-*}; cfg=${rest#*-}; X=""; [[ $cfg == STb* ]] && X="$ST"
    T=""; [ $text = wiki ] && T="--text-file /opt/kiln/work/wikitext2_test.txt"
    bash tools/dc_run.sh $BOX gate-$run "$E0 $X" tools/check_mixed.py $A --concurrency 32 --requests 32 $T --out $L/gate-$run.json
    aws s3 cp --quiet $L/gate-$run.json s3://<your-bucket>/logs/$BOX/ --region us-east-2
  fi
  echo "gate-$run rc=$(cat $L/gate-$run.log.rc)" >> $L/dc-gate.txt
done
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:$PATH PYTHONPATH=/opt/kiln/src
for p in "long-BASE long-BASE2" "long-BASE long-STb" "wiki-BASE wiki-STb"; do set -- $p
  [ -f $L/gate-cm-$1.json ] && [ -f $L/gate-cm-$2.json ] || continue
  python tools/check_mixed.py --compare $L/gate-cm-$1.json $L/gate-cm-$2.json > $L/gate-cmp-$1-$2.log 2>&1
  aws s3 cp --quiet $L/gate-cmp-$1-$2.log s3://<your-bucket>/logs/$BOX/ --region us-east-2
done
echo GATE_DONE >> $L/dc-gate.txt
