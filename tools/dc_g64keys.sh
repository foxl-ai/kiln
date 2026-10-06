#!/bin/bash
# The colocated G64 default graph keys on two trees (no flags set: the serving engine's defaults), for "no default
# changes": compile_farm capture with the prefill graphs (a serving run), ranks 0, 8, 16, 24, the G64 env and argv of
# tools/hv_ab.sh; then each rank's key set compared. Nothing is enqueued or compiled.
# DC_KEYCFG=CP64: the same for KILN_DSA_CP=1 alone (no flag of this branch) at the CP-64 decode box's shapes, prefill
# chunks included: context parallelism without the new flags traces as on the other tree.
#     bash tools/dc_g64keys.sh /opt/kiln/src-ev3df /opt/kiln/src-m3df
set -uo pipefail
B=s3://<your-bucket>
L=/opt/kiln/logs
SHAPES=/opt/kiln/shapes/glm53
E="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 3000 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8 --requests 128 --concurrency 64"
CFG=${DC_KEYCFG:-G64}
if [ "$CFG" = CP64 ]; then
  E="$E KILN_DSA_CP=1"
  A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --kv-cache-dtype fp8 --no-prefix-caching --page-size 256 --page-buckets 64 --max-num-seqs 256 --kv-cache-gb 0.62 --decode-buckets 64 --requests 256 --concurrency 256"
fi
outs=()
for src in "$@"; do
  n=$(basename $src)
  out=/opt/kiln/work/keys-$CFG-$n
  rm -rf $out
  (export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=$src
   cd /opt/kiln/work && env $E python $src/tools/compile_farm.py capture --shape-dir $SHAPES --ranks 0,8,16,24 --target trn1 \
     --out-dir $out -- $A > $L/keys-$CFG-$n.log 2>&1)
  rc=$?
  echo "$CFG $n $(cat $src/.kiln-head 2>/dev/null) rc=$rc $(tail -1 $L/keys-$CFG-$n.log | cut -c1-160)" >> $L/dc-g64keys.txt
  outs+=($out/keys.json)
done
python3 - "${outs[@]}" >> $L/dc-g64keys.txt <<'PY'
import json, sys
ks = [json.load(open(p)) for p in sys.argv[1:]]
def per_rank(d):
    out = {}
    for k, ranks in d.items():
        for r in ranks:
            out.setdefault(r, set()).add(k)
    return out
a, b = per_rank(ks[0]), per_rank(ks[1])
for r in sorted(set(a) | set(b)):
    print(f"rank {r}: {len(a.get(r, ()))} / {len(b.get(r, ()))} keys, equal={a.get(r) == b.get(r)}")
print(sys.argv[1].split("/")[-2].split("-")[1] + "KEYS", "EQUAL" if a == b else "DIFFER", f"{len(ks[0])} / {len(ks[1])} distinct")
PY
for f in $L/dc-g64keys.txt "${outs[@]}"; do aws --region us-east-2 s3 cp --quiet $f $B/logs/kiln-dc-32/$(basename $(dirname $f))-$(basename $f); done
