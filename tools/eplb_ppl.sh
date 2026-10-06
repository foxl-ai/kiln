#!/bin/bash
# Real-weight ppl with expert parallelism forced on (check_ppl's 4 sequences per call leave the automatic default
# off): A plain EP, B one redundant slot (copies from KILN_EPLB_INIT), on the 4 sentences and the wikitext-2 slice.
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=/opt/kiln/src
cd /opt/kiln/src
L=/opt/kiln/logs
export KILN_MOE_EP=1 KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_MIN_TOKENS=1 KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1
export KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/eplb-a2b1414/
BASE="--model zai-org/GLM-5.3-Flash --tp 32 --dp-attention 4 --piecewise --kv-cache-gb 1.0"
for run in ${@:-ppl4-A ppl4-B wt-A wt-B}; do
  kind=${run%%-*}; cfg=${run#*-}
  T=; [ $kind = wt ] && T="--text-file /opt/kiln/work/wikitext2_test.txt"
  E=; [[ $cfg == B* ]] && E="KILN_EP_REDUNDANT=1 KILN_EPLB_INIT=/opt/kiln/eplb-init-random.pt"
  echo "env $E python tools/check_ppl.py $BASE $T" > $L/ppl-$run.log.cmd
  env $E python tools/check_ppl.py $BASE $T > $L/ppl-$run.log 2>&1; echo $? > $L/ppl-$run.log.rc
  aws s3 cp --quiet $L/ppl-$run.log s3://<your-bucket>/logs/kiln-tq-32/; aws s3 cp --quiet $L/ppl-$run.log.cmd s3://<your-bucket>/logs/kiln-tq-32/
  sleep 120
done
echo PPL_DONE
