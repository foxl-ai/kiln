#!/bin/bash
# CP-64 at page bucket 33 (the context's own 8448 tokens) instead of 64: under CP each rank then holds 264 local pools,
# fewer than keep (512), which f997d35's attention_cp takes as "every local pool a candidate" (77d716f) instead of a
# top-512 over 512 local pools of which 264 are real. Captured and compiled on this box, then timed with real KV.
set -uo pipefail
export KILN_BOX_SRC=${KILN_BOX_SRC:-/opt/kiln/src-f997}
cd $KILN_BOX_SRC
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1 KILN_MOE_DEDUPE_V9=1 KILN_DSA_CP=1 KILN_MOE_DEDUPE_MAX_TOKENS=${DC_MT:-256}"
A0="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --kv-cache-dtype fp8 --no-prefix-caching --page-size 256 --page-buckets 33"
R=${DC_R:-64}
G="--max-num-seqs $((4 * R)) --kv-cache-gb ${DC_KV:-0.62} --decode-buckets $R"
N=cp-R${R}b33
L=/opt/kiln/logs
bash tools/dc_capture.sh dc1-t1 $N trn1 "$E1" $A0 $G > $L/dcq-$N.out 2>&1
echo "$N rc=$? $(tail -1 $L/dcq-$N.out)" >> $L/dcq-enqueued.txt
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root PYTHONPATH=$KILN_BOX_SRC
cd /opt/kiln/work && python $KILN_BOX_SRC/tools/compile_farm.py work --queue s3://<your-bucket>/compile-farm/q/dc1-t1/ \
  --mem-budget-gb 300 --linger 30 --out /opt/kiln/work/farm-$N.jsonl > $L/farm-$N.log 2>&1
cd $KILN_BOX_SRC
E0="$E1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/dc1-t1/"
bash tools/dc_run.sh kiln-dc-32 td-$N "$E0" tools/time_decode.py --real-kv --all-buckets --steps 64 --skip 8 \
  --price trn1.32xlarge-spot=2.15 -- $A0 $G
echo "td-$N rc=$(cat $L/td-$N.log.rc) $(grep -h curve $L/td-$N.log | tail -1 | cut -c1-300)" >> $L/dc-tdcp.txt
