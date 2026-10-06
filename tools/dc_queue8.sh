#!/bin/bash
# q/dc2-st (engine-v0 b8814ab + this branch): the fixed-cost stack (KILN_MOE_EP=0, KILN_DENSE_FP8=0, KILN_DSA_PREFIX=mm, KILN_DECODE_WHOLE=1) as a serving
# configuration with its prefill graphs (the G64 shape), for its serving A/B and tools/check_mixed.py.
set -uo pipefail
cd ${KILN_BOX_SRC:-/opt/kiln/src}
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1"
A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 64 --concurrency 64 --decode-buckets 16 --kv-cache-gb 1.5 --kv-cache-dtype fp8"
L=/opt/kiln/logs
E0="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1"
# ST's 12-layer prefill pieces fail neuronx-cc 2.27 with NCC_ISCH719 "topological order violations" (4 of 12 graphs,
# q/dc2-st failed/, 2026-10-05): ST6 takes 6-layer prefill pieces (decode graphs unchanged)
# ST6 fails the same way (4 of its 16 new pieces): two variants to isolate it, both with 12-layer prefill pieces:
# STa without KILN_DENSE_FP8=0, STb with KILN_SP_GATHER=xla (b8814ab's prefill default on trn1 is the NKI gather).
if [ "${DC_STAB:-0}" = 1 ]; then
  KILN_CAP_PREFILL=1 bash tools/dc_capture.sh dc2-st G64-STa trn1 "${E1/KILN_DENSE_FP8=0 /}" $A > $L/dcq-G64-STa.out 2>&1 &
  KILN_CAP_PREFILL=1 bash tools/dc_capture.sh dc2-st G64-STb trn1 "$E1 KILN_SP_GATHER=xla" $A > $L/dcq-G64-STb.out 2>&1 &
  wait
  echo "G64-STa $(tail -1 $L/dcq-G64-STa.out)" >> $L/dcq-enqueued.txt
  echo "G64-STb $(tail -1 $L/dcq-G64-STb.out)" >> $L/dcq-enqueued.txt
  exit 0
fi
if [ "${DC_ST6:-0}" = 1 ]; then
  KILN_CAP_PREFILL=1 bash tools/dc_capture.sh dc2-st G64-ST6 trn1 "${E1/KILN_PIECEWISE_PREFILL_MOE_GROUP=12/KILN_PIECEWISE_PREFILL_MOE_GROUP=6}" $A > $L/dcq-G64-ST6.out 2>&1
  echo "G64-ST6 $(tail -1 $L/dcq-G64-ST6.out)" >> $L/dcq-enqueued.txt
  exit 0
fi
KILN_CAP_PREFILL=1 bash tools/dc_capture.sh dc2-st G64-ST trn1 "$E1" $A > $L/dcq-G64-ST.out 2>&1 &
KILN_CAP_PREFILL=1 bash tools/dc_capture.sh dc2-st G64-BASE trn1 "$E0" $A > $L/dcq-G64-BASE.out 2>&1 &
wait
echo "G64-ST $(tail -1 $L/dcq-G64-ST.out)" >> $L/dcq-enqueued.txt
echo "G64-BASE $(tail -1 $L/dcq-G64-BASE.out)" >> $L/dcq-enqueued.txt
