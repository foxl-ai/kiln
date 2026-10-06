#!/bin/bash
# EPLB correctness on the serving graphs (tools/check_mixed.py: greedy text, top-2 logprobs, teacher-forced compare):
# A (plain EP) twice for the run-to-run floor, B (one redundant slot, copies from KILN_EPLB_INIT) once, on
# check_ppl's LONG_TEXT and on wikitext-2. Usage: eplb_check.sh [runs...] (default: A1 A2 B on long, A1 B on wiki).
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=/opt/kiln/src
cd /opt/kiln/src
L=/opt/kiln/logs
export KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1
export KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/eplb-a2b1414/
ARGS="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64 --decode-buckets 16 --kv-cache-gb 1.2 --kv-cache-dtype fp8 --state-checkpoints 4 --concurrency 32 --requests 32"
[ -f /opt/kiln/work/wikitext2_test.txt ] || { mkdir -p /opt/kiln/work; aws s3 cp --quiet s3://<your-bucket>/data/wikitext2_test.txt /opt/kiln/work/wikitext2_test.txt; }
for run in ${@:-long-A1 long-A2 long-B wiki-A1 wiki-B}; do
  text=${run%%-*}; cfg=${run#*-}
  T=; [ $text = wiki ] && T="--text-file /opt/kiln/work/wikitext2_test.txt"
  if [[ $cfg == B* ]]; then E="KILN_EP_REDUNDANT=1 KILN_EPLB_INIT=/opt/kiln/eplb-init-random.pt"; else E=""; fi
  echo "env $E python tools/check_mixed.py $ARGS $T --out $L/cm-$run.json" > $L/cm-$run.log.cmd
  env $E python tools/check_mixed.py $ARGS $T --out $L/cm-$run.json > $L/cm-$run.log 2>&1; echo $? > $L/cm-$run.log.rc
  aws s3 cp --quiet $L/cm-$run.json s3://<your-bucket>/logs/kiln-tq-32/; aws s3 cp --quiet $L/cm-$run.log.cmd s3://<your-bucket>/logs/kiln-tq-32/
  sleep 120
done
for p in "long-A1 long-A2" "long-A1 long-B" "wiki-A1 wiki-B"; do set -- $p
  python tools/check_mixed.py --compare $L/cm-$1.json $L/cm-$2.json > $L/cm-cmp-$1-$2.log 2>&1
  aws s3 cp --quiet $L/cm-cmp-$1-$2.log s3://<your-bucket>/logs/kiln-tq-32/
done
echo CHECK_DONE
