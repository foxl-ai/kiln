#!/bin/bash
# What one MTP speculative step costs on trn1 against a decode step (tools/time_decode.py --spec-method mtp: the decode
# step, the verify step at Q = 2 rows per sequence and the MTP draft graph), at the G64 shape with the async-MTP serving
# config's graphs (q/dc1-mtp).
set -uo pipefail
cd /opt/kiln/src
E0="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=s3://<your-bucket>/compile-farm/q/dc1-mtp/ KILN_SPEC_ASYNC=1"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 3000 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64 --concurrency 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8 --spec-method mtp --spec-k 1 --state-checkpoints 16"
bash tools/dc_run.sh kiln-dc-32 td-G64-MTP "$E0" tools/time_decode.py --steps 48 --skip 8 --pieces -- $A
echo "td-G64-MTP rc=$(cat /opt/kiln/logs/td-G64-MTP.log.rc)" >> /opt/kiln/logs/dc-td5.txt
