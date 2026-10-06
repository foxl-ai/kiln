#!/bin/bash
# Async MTP re-measured on engine-v0 70ddc1b (tools/amtp_g1b.sh's G1b workload: conc 64, 75% of every 8192-token prompt
# one of 4 shared 6144-token prefixes, a cold level then a warm one, 256 requests per level), the final G64 shape with
# KV 1.5 fp8 (amtp_g1b.sh used 1.2 on f54fc18): MB = no MTP (q/final-f70c14b's G64 graphs), MS = MTP k=1 stepped
# synchronously, MA = MTP k=1 under overlap scheduling (KILN_SPEC_ASYNC=1); MS / MA hold 16 checkpoint rows so the KDA
# state pool keeps MB's 49 rows per group and every target graph is MB's (q/dc1-mtp).
#     bash tools/dc_mtp.sh [MB MS MA]
set -uo pipefail
cd /opt/kiln/src
E0="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1"
ARGS="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 3000 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64 --concurrency 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8 --requests 256 --shared-prefix-len 6144 6144 --num-prefixes 4 --keep-cache --price trn1.32xlarge-spot=2.15"
MTP="--spec-method mtp --spec-k 1 --state-checkpoints 16"
for which in ${@:-MB MS MA}; do
  case $which in
    MB) E="KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/final-f70c14b/"; X="" ;;
    MS) E="KILN_SPEC_ASYNC=0 KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/dc1-mtp/"; X="$MTP" ;;
    MA) E="KILN_SPEC_ASYNC=1 KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/dc1-mtp/"; X="$MTP" ;;
  esac
  bash tools/dc_run.sh kiln-dc-32 amtp-$which "$E0 $E" bench/serve_sweep.py $ARGS $X
  echo "amtp-$which rc=$(cat /opt/kiln/logs/amtp-$which.log.rc)" >> /opt/kiln/logs/dc-mtp.txt
done
echo MTP_DONE >> /opt/kiln/logs/dc-mtp.txt
