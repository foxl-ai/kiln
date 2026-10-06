#!/bin/bash
# One box of a disaggregated deployment (engine/disagg.py), run ON the box (infra/fleet.sh run ... 'bash tools/pd_box.sh ...').
#
#   bash tools/pd_box.sh prep [prefill]           # nvme RAID0, GLM-5.3-Flash into the HF cache, (prefill) the EPLB init file
#   bash tools/pd_box.sh start <role> <name>      # start bench/pd_serve.py detached: role prefill | decode, config <name>
#   bash tools/pd_box.sh router <prefill urls> <decode urls> [threshold]   # start the PD router on :8000 (PD_ROUTER_ARGS:
#                                                 # --latency-prefill-urls, --prefill-depth, --latency-depth, ...)
#   bash tools/pd_box.sh stop                     # stop every server this script started (by process group)
#   bash tools/pd_box.sh status                   # health of the local server, last log lines
#
# Configs (the exact env and serve_sweep arguments of a compile-farm capture: the KV pool, max_num_seqs and the buckets
# are graph input shapes, so a server runs the graphs only with these):
#   P8K-EPLB  prefill: engine-v0 c7c43e9's G64 + EPLB + one-piece 8192 prefill, q/pf-p8-7ce6a12 G64-EPLB-P8K-KV12CK4
#             (the same-box best, 191.3 out tok/s); EPLB rebalances once after 200 prefill calls (KILN_EPLB_INTERVAL).
#   STL9R2    decode: feat/decode-scale 4fd7469's ST + v9, 28 rows per DP group, real 8K KV (2.1 GB), q/dc1-t1 t1-STL9r2.
set -uo pipefail
SRC="${KILN_BOX_SRC:-/opt/kiln/src-pd}"
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=$SRC
L=/opt/kiln/logs
mkdir -p $L /opt/kiln/work
B=s3://<your-bucket>
COMMON="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1"
G1="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800"
declare -A ENV ARGS
ENV[P8K-EPLB]="$COMMON KILN_EPLB_INIT=/opt/kiln/eplb-init-random.pt KILN_EP_REDUNDANT=1 KILN_PIECEWISE_PREFILL_MOE_GROUP=45 KILN_EPLB_INTERVAL=${PD_EPLB_INTERVAL:-200} KILN_EPLB_MAX_REBALANCES=1 KILN_COMPILE_FARM=$B/compile-farm/q/pf-p8-7ce6a12/"
ARGS[P8K-EPLB]="$G1 --prefill-tokens 8192 --prefill-buckets 2048 --max-num-seqs 64 --concurrency 64 64 --decode-buckets 16 --kv-cache-gb 1.2 --kv-cache-dtype fp8 --state-checkpoints 4 --eplb-rebalance"
ENV[STL9R]="$COMMON KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_MOE_EP=0 KILN_DENSE_FP8=0 KILN_DSA_PREFIX=mm KILN_DECODE_WHOLE=1 KILN_MOE_DEDUPE_MAX_TOKENS=256 KILN_MOE_DEDUPE_V9=1 KILN_COMPILE_FARM=$B/compile-farm/q/dc1-t1/"
ARGS[STL9R]="$G1 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 112 --kv-cache-gb 1.95 --kv-cache-dtype fp8 --no-prefix-caching --decode-buckets 28"
# t1-STL9r2: the same with KV 2.1 GB (~7,777 pages per group; 1.95 GB is ~7,222, under 28 x 264 + 1 = 7,393).
# CP-R64 / CP-R48: feat/decode-scale-cp 969c0ba (decode-scale + long-context 2c99613), KILN_DSA_CP=1 over the 8 ranks of a
# group, 256-token pages, 64 / 48 rows per group, real 8K KV (decode agent: 237.3 ms / 195.5 ms per step decode-only),
# q/dc1-t1 cp-R64 / cp-R48.
ENV[CP-R64]="${ENV[STL9R]} KILN_DSA_CP=1"
ARGS[CP-R64]="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --kv-cache-dtype fp8 --no-prefix-caching --page-size 256 --page-buckets 64 --max-num-seqs 256 --kv-cache-gb 0.62 --decode-buckets 64"
ENV[CP-R48]="${ENV[CP-R64]}"
ARGS[CP-R48]="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --kv-cache-dtype fp8 --no-prefix-caching --page-size 256 --page-buckets 64 --max-num-seqs 192 --kv-cache-gb 0.47 --decode-buckets 48"
# T2-BEST1: trn2.48xlarge, the trn2 agent's best1 with 8192-row prefill calls (feat/trn2-fast 8ba166d, q/t2f-best1-p4@pf8k,
# rank-0 keys of scratch/pd-trn2 checked equal), for BOTH roles of a single-box PD pair (prefill cores 0-31, decode
# 32-63: PD_CORE_BASE=32). TP experts (EP hangs on trn2 in serving).
ENV[T2-BEST1]="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DECODE_SP=1 KILN_DSA_DECODE_KERNEL=nki KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_KDA_DECODE_KERNEL=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_LNC_SPLIT=delta_rule,dsa_topk,moe_dedupe,kda_decode,dsa_decode KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_GROUP=1 KILN_SP_ROUTE=1 KILN_PIECEWISE_PREFILL_MOE_GROUP=4 KILN_MOE_EP=0 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=$B/compile-farm/q/t2f-best1-p4@pf8k/"
ARGS[T2-BEST1]="$G1 --prefill-tokens 8192 --prefill-buckets 2048 --max-num-seqs 128 --decode-buckets 4,8,16,32 --kv-cache-gb 2.75 --kv-cache-dtype fp8"
# CP-R96 / CP-R80: feat/decode-scale-f997 (engine-v0 f997d35 lineage) with kiln_moe_dedupe_v10, 96 / 80 rows per group, q/dc1-t1
# cp-R96-v10 / cp-R80-v10 (decode agent: 335.7 / 295.4 ms per step decode-only; CP-96 ~14.9 GiB per rank of 16).
ENV[CP-R96]="${ENV[CP-R64]} KILN_MOE_DEDUPE_MAX_TOKENS=512"
ARGS[CP-R96]="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --kv-cache-dtype fp8 --no-prefix-caching --page-size 256 --page-buckets 64 --max-num-seqs 384 --kv-cache-gb 0.92 --decode-buckets 96"
# CPA-R96: CP-R96 plus the decode agent's lever A and page keys (feat/decode-scale-f997 e34272d, q/dc1-t1 cpA-R96Ak):
# neither flag changes a paged cache or state row, so the handoff layout is CP-R96's.
ENV[CPA-R96]="${ENV[CP-R96]} KILN_DSA_CP_ALL_LOCAL=1 KILN_DSA_CP_PAGE_KEYS=1"
ARGS[CPA-R96]="${ARGS[CP-R96]}"
ENV[CP-R80]="${ENV[CP-R96]}"
ARGS[CP-R80]="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 4 --piecewise --overlap --input-len 8192 --output-len 256 --warmup --max-seconds 1800 --prefill-tokens 4096 --prefill-buckets 1024 --kv-cache-dtype fp8 --no-prefix-caching --page-size 256 --page-buckets 64 --max-num-seqs 320 --kv-cache-gb 0.77 --decode-buckets 80"
# TP1-P4K / TP1-P2K: the latency prefill box, P8K-EPLB's env at DP attention 1 (attention TP 32: one request per call over
# all 32 ranks, the MLA latent and indexer replicated on each) in 4096- / 2048-row calls, q/pf-tp1-pdpre TP1-P4096 /
# TP1-P2048 (11 graphs each, ranks share every key). It hands off into a DP-attention-4 decode engine by regrouping
# (ModelRunner.pd_regroupable). Per-rank HBM (tools/hbm_estimate.py, P8K-EPLB at 15.85 GiB loads): 14.48 / 13.62 GiB; the
# one-piece 8192-row call (TP1-P8192) is 19.80 GiB, its prefill graph alone 6.58 GiB of spill rings, so it is not a config.
TP1="$COMMON KILN_EPLB_INIT=/opt/kiln/eplb-init-random.pt KILN_EP_REDUNDANT=1 KILN_PIECEWISE_PREFILL_MOE_GROUP=45 KILN_EPLB_INTERVAL=${PD_EPLB_INTERVAL:-200} KILN_EPLB_MAX_REBALANCES=1 KILN_COMPILE_FARM=$B/compile-farm/q/pf-tp1-pdpre/"
TP1A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 1 --piecewise --overlap --input-len 8192 --output-len 256 --page-buckets 264 --warmup --max-seconds 1800 --max-num-seqs 16 --concurrency 16 --decode-buckets 16 --kv-cache-gb 1.2 --kv-cache-dtype fp8 --state-checkpoints 4 --eplb-rebalance"
ENV[TP1-P4K]="$TP1"
ARGS[TP1-P4K]="$TP1A --prefill-tokens 4096 --prefill-buckets 4096"
ENV[TP1-P2K]="$TP1"
ARGS[TP1-P2K]="$TP1A --prefill-tokens 2048 --prefill-buckets 2048"
ENV[STL9R2]="${ENV[STL9R]}"
ARGS[STL9R2]="$G1 --prefill-tokens 4096 --prefill-buckets 1024 --max-num-seqs 112 --kv-cache-gb 2.1 --kv-cache-dtype fp8 --no-prefix-caching --decode-buckets 28"

ip() { hostname -I | awk '{print $1}'; }

case "${1:-}" in
  prep)
    if ! mountpoint -q /opt/kiln/nvme; then
      devs=$(lsblk -dpno NAME,MODEL | awk '/Instance Storage/{print $1}'); n=$(echo $devs | wc -w)
      mkdir -p /opt/kiln/nvme
      mdadm --create /dev/md0 --level=0 --raid-devices=$n $devs --run >/dev/null 2>&1 || mdadm --assemble /dev/md0 $devs
      mkfs.ext4 -F -q -E lazy_itable_init=1,lazy_journal_init=1,nodiscard /dev/md0 2>/dev/null; mount /dev/md0 /opt/kiln/nvme
      mkdir -p /opt/kiln/nvme/hf; rm -rf /opt/kiln/hf; ln -sfn /opt/kiln/nvme/hf /opt/kiln/hf
    fi
    python $SRC/tools/fetch_models.py zai-org/GLM-5.3-Flash 2>&1 | grep FETCHED
    [ "${2:-}" = prefill ] && aws --region us-east-2 s3 cp --quiet $B/logs/kiln-tq-cpu/eplb-init-random.pt /opt/kiln/eplb-init-random.pt
    echo "PREP-DONE $(df -h /opt/kiln/nvme | tail -1 | awk '{print $3}') used, $(ls /opt/kiln/*.pt 2>/dev/null)"
    ;;
  start)
    # PD_PORT (default 8100), PD_CORE_BASE (default 0: --core-base for a second engine on the same box) and
    # PD_LISTEN_PORT (default 7400) let two engines share a box (trn2.48xlarge: cores 0-31 and 32-63).
    role="$2"; name="$3"; port="${PD_PORT:-8100}"
    [ -n "${ENV[$name]:-}" ] || { echo "unknown config $name"; exit 2; }
    extra=""
    [ "$role" = both ] && role=none
    [ "$role" = decode ] && extra="--pd-listen 0.0.0.0:${PD_LISTEN_PORT:-7400} --pd-advertise $(ip):${PD_LISTEN_PORT:-7400} --pd-buffer-gb ${PD_BUFFER_GB:-48}"
    cb=""; [ -n "${PD_CORE_BASE:-}" ] && cb="--core-base $PD_CORE_BASE"
    cd /opt/kiln/work
    echo "env ${ENV[$name]} ${PD_EXTRA_ENV:-} python $SRC/bench/pd_serve.py --pd-role $role $extra --port $port -- ${ARGS[$name]} $cb" > $L/pd-$role-$port.cmd
    setsid nohup env ${ENV[$name]} ${PD_EXTRA_ENV:-} python $SRC/bench/pd_serve.py --pd-role $role $extra --port $port -- ${ARGS[$name]} $cb \
      > $L/pd-$role-$port.log 2>&1 < /dev/null &
    echo $! > $L/pd-server-$port.pid
    ln -sfn $L/pd-$role-$port.log $L/pd-$role.log
    echo "started $role $name pid $! at $(ip):$port"
    ;;
  router)
    cd /opt/kiln/work
    setsid nohup python -m kiln.server.pd_router --prefill-urls "$2" --decode-urls "$3" --threshold "${4:-0}" \
      ${PD_ROUTER_ARGS:-} --port 8000 > $L/pd-router.log 2>&1 < /dev/null &
    echo $! > $L/pd-router.pid
    echo "router pid $!"
    ;;
  stop)
    for f in $L/pd-router.pid $L/pd-server*.pid; do
      [ -f $f ] || continue
      pid=$(cat $f); kill -TERM -- -$pid 2>/dev/null || kill -TERM $pid 2>/dev/null
      for _ in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 1; done
      kill -KILL -- -$pid 2>/dev/null; rm -f $f
    done
    echo stopped
    ;;
  status)
    curl -s -m 3 localhost:8100/health; echo; curl -s -m 3 localhost:8000/health; echo
    for f in $L/pd-prefill.log $L/pd-decode.log $L/pd-router.log; do
      [ -f $f ] && { echo "== $f"; grep -vE "Warning|warn|INFO" $f | tail -n ${2:-6} | cut -c1-300; }
    done
    ;;
  *) sed -n 2,14p "$0"; exit 2 ;;
esac
