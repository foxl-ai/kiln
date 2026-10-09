#!/bin/bash
# feat/prefill-compute: one warm prefill call of a serving configuration captured on every rank (KILN_CAPTURE_INPUTS at
# prefill:<n>), each piece replayed on 32 cores with rank 0 profiled (tools/util_report.py replay --keep-ntff), then bins
# and report, and the per-segment op split of the layer pieces (neuron-explorer's JSON view + tools/prof_segments.py).
#     KILN_BOX_SRC=/opt/kiln/src-pc bash tools/pc_cap.sh G64           # the colocated G64 default (4096-token calls)
# Configs: G64 = tools/pd_box.sh G64P's env and arguments (q/pf-spw2-c0c8074 graphs), with 64 requests at concurrency 64.
# PC_ENV_EXTRA adds env, PC_ARGS_EXTRA serve_sweep arguments (G64: --prompt-ids <ids.npy> for real text), PC_FARM
# another farm queue, PC_TAG a suffix for the arm's logs and S3 prefix. R8: see the case below.
# Results: s3 logs/<box>/pcap-<cfg>/ (report, split per piece, NTFFs of rank 0) and /opt/kiln/logs/pc-cap.txt.
set -uo pipefail
BOX=${PC_BOX:-kiln-pcm-32}
export KILN_BOX_SRC=${KILN_BOX_SRC:-/opt/kiln/src-pc}
cd $KILN_BOX_SRC
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=$KILN_BOX_SRC
B3=s3://<your-bucket>
L=/opt/kiln/logs
mkdir -p $L /opt/kiln/work
COMMON="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1"
G1="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800"
cfg=$1
case $cfg in
  G64) E="$COMMON KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_COMPILE_FARM=$B3/compile-farm/q/${PC_FARM:-pf-spw2-c0c8074}/"
       A="$G1 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64 --concurrency 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8 --requests 64 ${PC_ARGS_EXTRA:-}"
       S=bench/serve_sweep.py; n=${PC_CALL:-20} ;;
  # R8: the 1M CP-8 engine (q/lc2-r8, tools/pc_r8.sh's), one lone 1,044,480-token request of lc2's long.txt
  # (tools/lc_ttft.py), call 200 captured (lc2's p200).
  R8) E="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_DSA_CP=1 KILN_MOE_EP=1 KILN_DSA_CP_SLOT_CLASSES=1 KILN_DSA_CP_DEGREE=8 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=$B3/compile-farm/q/${PC_FARM:-lc2-r8}/"
      A="--lengths 1044480 --warm-lengths 8192 --text-file /opt/kiln/data/long.txt -- --model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 1 --piecewise --overlap --page-size 256 --max-model-len 1048576 --page-buckets 1024,4096 --warmup --max-seconds 7200 --prefill-tokens 4096 --prefill-buckets 4096 --max-num-seqs 4 --decode-buckets 1 --kv-cache-gb 1.5 --kv-cache-dtype fp8 --state-checkpoints 0 --output-len 4"
      S=tools/lc_ttft.py; n=${PC_CALL:-200} ;;
  *) echo "unknown $cfg" >&2; exit 2 ;;
esac
arm=$cfg${PC_TAG:-}  # PC_TAG names an arm (e.g. -fn with PC_ENV_EXTRA=KILN_KDA_FUSED_NORM=1 PC_FARM=pcx-fn)
CAP=/opt/kiln/nvme/capp-$arm PROF=/opt/kiln/nvme/profp-$arm
rm -rf $CAP $PROF
bash tools/dc_run.sh $BOX capp-$arm "$E ${PC_ENV_EXTRA:-} KILN_CAPTURE_INPUTS=$CAP KILN_CAPTURE_AT=prefill:$n" $S $A
echo "$(date -u +%H:%M:%SZ) capture $arm rc=$(cat $L/capp-$arm.log.rc) $(grep -h 'out tok/s' $L/capp-$arm.log | tail -1 | cut -c1-160)" >> $L/pc-cap.txt
python tools/util_report.py replay $CAP --call prefill:$n --out $PROF --keep-ntff ${PC_REPLAY_ARGS:-} > $L/repp-$arm.log 2>&1
python tools/util_report.py bins $PROF >> $L/repp-$arm.log 2>&1
python tools/util_report.py report $PROF --kind prefill --shape-dir /opt/kiln/shapes/glm53 > $L/repp-$arm.report.txt 2>&1
[ "${PC_KEEP_CAP:-0}" = 1 ] || rm -rf $CAP
echo "$(date -u +%H:%M:%SZ) replay $arm $(head -12 $L/repp-$arm.report.txt | tail -4 | tr '\n' ' ' | cut -c1-300)" >> $L/pc-cap.txt
# the op split of every layer piece (the pieces with more than 20 ms in the report's table)
for nt in $PROF/*_rank_0_exec_2.ntff; do
  [ "${PC_VIEW:-1}" = 1 ] || break  # PC_VIEW=0: the replay report only
  tag=$(basename $nt _rank_0_exec_2.ntff); key=${tag#*-}
  neff=$(ls /root/.cache/neuron_libtorch/neuron/compile_cache/$key/graph_$key.neff 2>/dev/null | head -1)
  [ -n "$neff" ] || continue
  [ $(stat -c %s $nt) -gt 20000000 ] || continue
  neuron-explorer view -n $neff -s $nt --output-format json --output-file /opt/kiln/nvme/view-$arm-$tag.json \
    --ignore-dma-trace --ignore-nc-buf-usage > /dev/null 2>&1
  python tools/prof_segments.py /opt/kiln/nvme/view-$arm-$tag.json > $L/split-$arm-$tag.txt 2>&1
  rm -f /opt/kiln/nvme/view-$arm-$tag.json
done
for f in $L/repp-$arm.log $L/repp-$arm.report.txt $L/split-$arm-*.txt $L/capp-$arm.log.cmd $L/pc-cap.txt; do
  aws --region us-east-2 s3 cp --quiet $f $B3/logs/$BOX/pcap-$arm/; done
for f in $PROF/*.summary.json $PROF/*.bins.json $PROF/*_rank_0_exec_2.ntff; do
  aws --region us-east-2 s3 cp --quiet $f $B3/logs/$BOX/pcap-$arm/prof/; done
echo "$(date -u +%H:%M:%SZ) PCAP_DONE $arm" >> $L/pc-cap.txt
aws --region us-east-2 s3 cp --quiet $L/pc-cap.txt $B3/logs/$BOX/
