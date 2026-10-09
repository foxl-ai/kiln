#!/bin/bash
# The trn1 CP decode box (KILN_DSA_CP=1, 256-token pages, page bucket 64, TP experts, real 8K KV) on ONE tree, every
# variant captured, compiled and timed on this box (feat/decode-next, q/dc2-t1). A variant is R<rows per group>
# followed by letters, each one opt-in flag:
#   A  KILN_DSA_CP_ALL_LOCAL=1       the local list without a selection (8K: every local pool fits keep)
#   k  KILN_DSA_CP_PAGE_KEYS=1       pool keys read as page rows
#   B  KILN_DSA_CP_MERGE_BOUND=1     the merge's tie search over ceil(log2(context pools)) bits instead of 21
#   C  KILN_DSA_CP_DECODE_COMPACT=1  partial attention over each row's live slots, compacted in the kernel (8 rows per
#                                    kernel iteration: KILN_DSA_SLOTS_C_RPI=8, the same graph as before RPI was a knob)
#   D  the same at 16 rows per iteration (KILN_DSA_SLOTS_C_RPI=16, the kernel's default)
#   P  (control) the decode partial attention x (1 + 2^-20), KILN_DSA_CP_DEBUG_EPS: how far a numerically neutral
#      change moves this box's outputs
#   p  KILN_DECODE_WHOLE=0           piecewise decode graphs (for profiles; the timed box is one graph)
#   n  a 0.25 GB KV pool, timed on the null page (MoE / KDA profiles: R96p with real KV does not load, Allocation
#      Failure, and neither MoE nor the token mixer reads the paged KV)
# KV is sized for rows x 33 pages + 1 per group (the 8K context plus the spread), max-num-seqs = 4 rows, MoE dedupe
# v9 (256 tokens) at 64 rows and v10 (512) above.
#
#   KILN_BOX_SRC=/opt/kiln/src-dc2 bash tools/dc_next.sh cap R96Ak R96AkB        # capture + enqueue (no NeuronCore)
#   KILN_BOX_SRC=/opt/kiln/src-dc2 bash tools/dc_next.sh compile <tag>           # compile the queue on this box's CPUs
#   KILN_BOX_SRC=/opt/kiln/src-dc2 bash tools/dc_next.sh time <passes> R96Ak R96AkB   # time_decode --real-kv, in turn
#                                                    (DC_DUMP=1: each run's outputs too, tools/compare_decode_dumps.py)
#   KILN_BOX_SRC=/opt/kiln/src-dc2 bash tools/dc_next.sh size R96 R112 R128      # tensors per rank (meta device)
# Results: /opt/kiln/logs/dc-next.txt (one line per step) and s3 logs/<box>/.
set -uo pipefail
BOX=${DC_BOX:-kiln-dc2-32}
export KILN_BOX_SRC=${KILN_BOX_SRC:-/opt/kiln/src-dc2}
cd $KILN_BOX_SRC
B3=s3://<your-bucket>
Q=${DC_QUEUE:-dc2-t1}
L=/opt/kiln/logs
mkdir -p $L /opt/kiln/work
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=$KILN_BOX_SRC
E1="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_MOE_DEDUPE_V9=1 KILN_DSA_CP=1"
A0="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --kv-cache-dtype fp8 --no-prefix-caching --page-size 256 --page-buckets 64"
TD="--real-kv --all-buckets --steps 64 --skip 8 --price trn1.32xlarge-spot=2.15 ${DC_TD_EXTRA:-}"

cfg() {  # sets X (env) and G (sweep args) for variant $1
  local v=$1 R flags
  [[ $v =~ ^R([0-9]+)([A-Za-z]*)$ ]] || return 1
  R=${BASH_REMATCH[1]}; flags=${BASH_REMATCH[2]}
  local kv; kv=$(python3 -c "print(round(($R * 33 + 1) * 0.285e6 / 1e9 + 0.02, 2))")
  local mt=512; [ "$R" -le 64 ] && mt=256
  X="KILN_MOE_DEDUPE_MAX_TOKENS=$mt"
  local whole=1
  KVSMALL=0
  local i c
  for ((i = 0; i < ${#flags}; i++)); do
    c=${flags:$i:1}
    case $c in
      A) X="$X KILN_DSA_CP_ALL_LOCAL=1" ;;
      k) X="$X KILN_DSA_CP_PAGE_KEYS=1" ;;
      B) X="$X KILN_DSA_CP_MERGE_BOUND=1" ;;
      C) X="$X KILN_DSA_CP_DECODE_COMPACT=1 KILN_DSA_SLOTS_C_RPI=8" ;;
      D) X="$X KILN_DSA_CP_DECODE_COMPACT=1 KILN_DSA_SLOTS_C_RPI=16" ;;
      P) X="$X KILN_DSA_CP_DEBUG_EPS=9.5367431640625e-07" ;;  # (control: the partial attention x (1 + 2^-20))
      p) whole=0 ;;
      n) KVSMALL=1; kv=0.25 ;;
      *) return 1 ;;
    esac
  done
  X="KILN_DECODE_WHOLE=$whole $X"
  G="--max-num-seqs $((4 * R)) --kv-cache-gb $kv --decode-buckets $R"
}

log() { echo "$(date -u +%H:%M:%SZ) $*" >> $L/dc-next.txt; aws --region us-east-2 s3 cp --quiet $L/dc-next.txt $B3/logs/$BOX/; }

case "${1:-}" in
  cap)
    shift
    for v in "$@"; do
      cfg $v || { echo "unknown $v" >&2; exit 2; }
      N=dn-$v${DC_TAG:-}  # DC_TAG / DC_ENV_EXTRA: a tagged capture with extra env (debug forms)
      bash tools/dc_capture.sh $Q $N trn1 "$E1 $X ${DC_ENV_EXTRA:-}" $A0 $G > $L/dcq-$N.out 2>&1
      log "cap $N rc=$? keys=$(python3 -c "import json; print(' '.join(k[:8] for k in json.load(open('/opt/kiln/work/cap-$N/keys.json'))))" 2>/dev/null) $(tail -1 $L/dcq-$N.out | cut -c1-160)"
    done
    log CAP_DONE "$@"
    ;;
  compile)
    tag=${2:-x}
    (cd /opt/kiln/work && python $KILN_BOX_SRC/tools/compile_farm.py work --queue $B3/compile-farm/q/$Q/ \
       --mem-budget-gb ${DC_MEM_GB:-300} --linger 30 --out /opt/kiln/work/farm-dn-$tag.jsonl > $L/farm-dn-$tag.log 2>&1)
    log "compiled $tag $(grep -h summary /opt/kiln/work/farm-dn-$tag.jsonl | tail -1 | cut -c1-240)"
    log COMPILE_DONE $tag
    ;;
  time)
    passes=$2; shift 2
    E0="$E1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=$B3/compile-farm/q/$Q/"
    for pass in $(seq 1 $passes); do
      for v in "$@"; do
        cfg $v || { echo "unknown $v" >&2; exit 2; }
        N=tdn-$v${DC_TAG:-}-$pass
        D=""; [ "${DC_DUMP:-0}" = 1 ] && D="--dump /opt/kiln/work/dump-$N"  # outputs, for compare_decode_dumps.py
        bash tools/dc_run.sh $BOX $N "$E0 $X ${DC_ENV_EXTRA:-}" tools/time_decode.py $TD $D -- $A0 $G
        log "$N rc=$(cat $L/$N.log.rc) $(grep -h curve $L/$N.log | tail -1 | cut -c1-300)"
      done
    done
    log TIME_DONE "$@"
    ;;
  size)
    shift
    for v in "$@"; do
      cfg $v || { echo "unknown $v" >&2; exit 2; }
      env $E1 $X python tools/tensor_bytes.py /opt/kiln/shapes/glm53 trn1 0 -- $A0 $G > $L/dn-size-$v.log 2>&1
      log "size $v $(grep -o '{"num_pages.*' $L/dn-size-$v.log | tail -1 | cut -c1-400) $(grep -i -E 'error|Traceback' $L/dn-size-$v.log | head -2)"
    done
    log SIZE_DONE "$@"
    ;;
  prof)  # tools/dc_cap3.sh for a variant (piecewise, `p`): one warm real-KV decode call captured on every rank, each
         # piece replayed on 32 cores with rank 0 profiled (DC_REPLAY_ARGS=--profile-all: every rank), util_report
    v=$2
    cfg $v || { echo "unknown $v" >&2; exit 2; }
    E0="$E1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=$B3/compile-farm/q/$Q/"
    n=${DC_CALL:-40}
    CAP=/opt/kiln/nvme/cap-$v PROF=/opt/kiln/nvme/prof-$v
    rm -rf $CAP $PROF
    RK=--real-kv; [ "$KVSMALL" = 1 ] && RK=
    bash tools/dc_run.sh $BOX capn-$v "$E0 $X KILN_CAPTURE_INPUTS=$CAP KILN_CAPTURE_AT=decode:$n" \
      tools/time_decode.py $RK --steps 48 --skip 8 -- $A0 $G
    python tools/util_report.py replay $CAP --call decode:$n --out $PROF --keep-ntff ${DC_REPLAY_ARGS:-} > $L/repn-$v.log 2>&1
    python tools/util_report.py bins $PROF >> $L/repn-$v.log 2>&1
    rows=$(echo $G | grep -o 'decode-buckets [0-9]*' | grep -o '[0-9]*$')
    python tools/util_report.py report $PROF --kind decode --rows $((4 * rows)) --shape-dir /opt/kiln/shapes/glm53 \
      > $L/repn-$v.report.txt 2>&1
    for f in $L/repn-$v.log $L/repn-$v.report.txt; do aws --region us-east-2 s3 cp --quiet $f $B3/logs/$BOX/; done
    [ "${DC_KEEP_CAP:-0}" = 1 ] || rm -rf $CAP
    log "prof $v done: $(grep -i -E 'call|replay' $L/repn-$v.report.txt | head -2 | tr '\n' ' ' | cut -c1-300)"
    ;;
  hbm)  # after size and compile: tools/hbm_estimate.py over the variant's captured keys plus its tensors
    shift
    for v in "$@"; do
      cfg $v || { echo "unknown $v" >&2; exit 2; }
      [[ $v =~ ^(R[0-9]+) ]] && base=${BASH_REMATCH[1]}  # the flags change no tensor: size R<rows> once
      tb=$(cat $L/dn-size-$v.log $L/dn-size-$base.log 2>/dev/null | grep -o '"total_gb": [0-9.]*' | tail -1 | grep -o '[0-9.]*$')
      python tools/hbm_estimate.py --keys-file /opt/kiln/work/cap-dn-$v/keys.json \
        --cache $B3/compile-cache/trn1-sdk2.32/lnl/ --tensors-gb ${tb:-0} > $L/dn-hbm-$v.log 2>&1
      log "hbm $v tensors_gb ${tb:-?} $(grep summary $L/dn-hbm-$v.log | tail -1 | cut -c1-400)"
    done
    log HBM_DONE "$@"
    ;;
  *) sed -n 2,20p "$0"; exit 2 ;;
esac
