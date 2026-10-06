#!/bin/bash
# HBM per rank of the STL decode box (ST, decode buckets 32 / 48, fp8 KV) at a few KV pool sizes: tools/tensor_bytes.py on
# the meta device (rank 0) plus tools/hbm_estimate.py over the compiled graphs of q/dc1-t1 t1-STL9.
set -uo pipefail
cd ${KILN_BOX_SRC:-/opt/kiln/src}
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root PYTHONPATH=$PWD
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1 KILN_MOE_DEDUPE_MAX_TOKENS=256"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 192 --kv-cache-dtype fp8 --no-prefix-caching --decode-buckets 32,48"
B=s3://<your-bucket>
L=/opt/kiln/logs
aws --region us-east-2 s3 cp --quiet $B/compile-farm/q/dc1-t1/configs/t1-STL9.keys.json /opt/kiln/work/t1-STL9.keys.json
for kv in ${DC_KVS:-0.5 3.5}; do
  env $E1 python tools/tensor_bytes.py /opt/kiln/shapes/glm53 trn1 0 -- $A --kv-cache-gb $kv > $L/hbm-tb-$kv.log 2>&1
  tb=$(grep -o '"total_gb": [0-9.]*' $L/hbm-tb-$kv.log | tail -1 | grep -o '[0-9.]*$')
  python tools/hbm_estimate.py --keys-file /opt/kiln/work/t1-STL9.keys.json --cache $B/compile-cache/trn1-sdk2.32/lnl/ \
    --tensors-gb ${tb:-0} > $L/hbm-est-$kv.log 2>&1
  echo "kv $kv tensors_gb ${tb:-?} $(tail -3 $L/hbm-est-$kv.log | tr '\n' ' ')" >> $L/hbm.txt
done
for f in $L/hbm*; do aws --region us-east-2 s3 cp --quiet $f $B/logs/kiln-dc-cf/; done
echo HBM_DONE >> $L/hbm.txt
