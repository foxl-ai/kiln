#!/bin/bash
# One stage of a multi-box layer pipeline (engine/pp.py), run ON the box (infra/fleet.sh run ... 'bash tools/pp_box.sh ...').
#
#   bash tools/pp_box.sh ttft <tag> <stage> <stages> <split> <next ip|-> <barrier ip> [config]
#       start tools/lc_ttft.py detached for this stage: lone requests at PP_LENGTHS (default 32768 131072 307200 1044480)
#       after PP_WARM (default 8192 307200); logs /opt/kiln/logs/<tag>-S<stage>.{log,cmd,json,timeline.jsonl}, copied
#       to s3 logs/<box>/ when the run ends. <split>: comma list (pp_split), <next ip>: the next stage's private IP
#       ("-" on the last stage), <barrier ip>: the LAST stage's private IP (it listens; lc_ttft --pp-barrier).
#   bash tools/pp_box.sh needle <tag> <stage> <stages> <split> <next ip|-> [config]
#       tools/check_long.py needle through the pipeline, teacher-forced (--forced: every answer token must rank 1 in
#       the last stage's prompt logprobs); PP_NEEDLE_ARGS replaces the default lengths 131072 1044480 x depths.
#   bash tools/pp_box.sh follow <tag> <stage> <stages> <split> <next ip|-> <barrier ip> [config]
#       a following stage (tools/pp_follow.py) of a pipeline whose stage 0 runs `ttft` with
#       PP_SWEEP_ARGS=--pp-follow: stage 0 alone takes the requests, the others follow its plan frames.
#   bash tools/pp_box.sh nll <tag> <stage> <stages> <split> <next ip|-> [config]
#       tools/check_long.py nll through the pipeline (PP_NLL_ARGS, default --max-tokens 131072; run it with
#       PP_EXTRA_ENV=KILN_PLP_VP=1, whose vocabulary-shard post graph fits beside the long-context KV): per-position
#       prompt logprobs to compare with one engine's on the same text (s3 logs/kiln-lc2-32/lc2-nllvp-R8.json).
#   bash tools/pp_box.sh warm <tag> [config]
#       tools/lc_warm.py on this box alone (default config R8W): the second-turn TTFT on a long document.
#   bash tools/pp_box.sh prompts | serve0 | servef | decode | router | client ...
#       a disaggregated deployment of a following pipeline (stage 0 serve0, the others servef), one decode engine
#       (decode; PP_ROLE=none for the one-engine reference), the PD router with the pipeline as a prefill unit (router),
#       and tools/pd_client.py's lone requests through it (client); see each command's comment.
#   bash tools/pp_box.sh stop | status <tag> <stage>
#
# Configs: the exact env and serve_sweep arguments of a compile-farm queue (every shape argument is in the graph keys).
#   R8   lever 1 (feat/long-context-next, docs/neuron-notes.md "Lever 1"), q/lc2-r8: a stage split on the 12-layer
#        prefill piece boundaries (12, 24, 36) runs only the single engine's graphs.
# A <stages> of 1 runs the plain single engine (no pipeline arguments): the reference for the same seeded prompts.
# Env passed through: PP_EXTRA_ENV (e.g. KILN_PP_ASYNC=1), PP_TTFT_ARGS (e.g. --record-tokens), PP_PORT (default 29600: the stage link, + rank),
# PP_BARRIER_PORT (default 29590).
set -uo pipefail
SRC="${KILN_BOX_SRC:-/opt/kiln/src-sc}"
export PATH=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin:/opt/aws/neuron/bin:$PATH HOME=/root HF_HOME=/opt/kiln/hf PYTHONPATH=$SRC
L=/opt/kiln/logs
mkdir -p $L /opt/kiln/work
B=s3://<your-bucket>
declare -A ENV ARGS
ENV[R8]="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PIECEWISE_PREFILL_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_DSA_CP=1 KILN_MOE_EP=1 KILN_DSA_CP_SLOT_CLASSES=1 KILN_DSA_CP_DEGREE=8 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=$B/compile-farm/q/lc2-r8/"
ARGS[R8]="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 1 --piecewise --overlap --page-size 256 --max-model-len 1048576 --page-buckets 1024,4096 --warmup --max-seconds 7200 --prefill-tokens 4096 --prefill-buckets 4096 --max-num-seqs 4 --decode-buckets 1 --kv-cache-gb 1.5 --kv-cache-dtype fp8 --state-checkpoints 0"
# R8W: R8 with one state-checkpoint row traded for one sequence (the state pool keeps R8's 5 rows per group, so 50 of
# R8's 54 graphs are shared; the 5 that differ are in q/sc-warm, which lists all 55): the warm second-turn run.
ENV[R8W]="${ENV[R8]%KILN_COMPILE_FARM=*}KILN_COMPILE_FARM=$B/compile-farm/q/sc-warm/"
ARGS[R8W]="${ARGS[R8]/--max-num-seqs 4/--max-num-seqs 3}"
ARGS[R8W]="${ARGS[R8W]/--state-checkpoints 0/--state-checkpoints 1}"
# R8N: R8 for the teacher-forced needle (tools/check_long.py --forced): q/lc2-r8p holds the prompt-logprob post graphs
# the 2-stage forced needle ran from (s3 logs/kiln-lc2-32/lc2-pp-needle-S*.log.cmd).
ENV[R8N]="${ENV[R8]%KILN_COMPILE_FARM=*}KILN_COMPILE_FARM=$B/compile-farm/q/lc2-r8p/"
ARGS[R8N]="${ARGS[R8]}"
# R8P8 / R8P8N: R8 for the 8-stage split 7,12,16,23,28,35,40 (pieces cut at the stage bounds: 64 new graphs, q/sc-pp8; a
# graph a queue does not list but the shared cache holds is fetched too, compile_cache.FarmWait, so the forced needle's
# prompt-logprob post graph of q/lc2-r8p loads from the same cache).
ENV[R8P8]="${ENV[R8]%KILN_COMPILE_FARM=*}KILN_COMPILE_FARM=$B/compile-farm/q/sc-pp8/"
ARGS[R8P8]="${ARGS[R8]}"
ENV[R8P8N]="${ENV[R8P8]}"
ARGS[R8P8N]="${ARGS[R8]}"
# The prefill-comm agent's 8K latency configs (q/pc-8k-1bae0bd, 2026-10-07: every stage key of the 12-layer-boundary
# splits checked a subset of the single engine's): 4 x 2048 / 3 x 2048 / 1 x 2048 / 1 x 4096 rows, and R8 with
# redundant experts rebalanced online on one stage (the S2 EP A/B).
L8E="KILN_ADMISSION=reserve KILN_CC_ARGS=--model-type=transformer KILN_DSA_POOL_CACHE=auto KILN_DSA_SELECT=nki KILN_LINEAR_ATTN_KERNEL=nki KILN_MOE_KERNEL=nki KILN_MOE_PREFILL_KERNEL=nki KILN_MOE_PREFILL_SKIP=20 KILN_PIECEWISE_MOE_GROUP=12 KILN_PREFILL_SP=1 KILN_SP_ROUTE=1 KILN_MOE_EP=1 NEURON_LIBTORCH_ASSERT_CACHE_HIT=1 KILN_COMPILE_FARM=$B/compile-farm/q/pc-8k-1bae0bd/"
L8A="--model zai-org/GLM-5.3-Flash --device neuron --tp 32 --dp-attention 1 --piecewise --overlap --input-len 8192 --output-len 256 --max-model-len 8448 --page-buckets 264 --warmup --max-seconds 1800 --max-num-seqs 16 --concurrency 16 --decode-buckets 16 --kv-cache-gb 1.2 --kv-cache-dtype fp8 --state-checkpoints 4"
ENV[L8]="$L8E KILN_PIECEWISE_PREFILL_MOE_GROUP=12"; ARGS[L8]="$L8A --prefill-tokens 2048 --prefill-buckets 2048"
ENV[L8-4K]="${ENV[L8]}"; ARGS[L8-4K]="$L8A --prefill-tokens 4096 --prefill-buckets 4096"
ENV[L8-G15]="$L8E KILN_PIECEWISE_PREFILL_MOE_GROUP=15"; ARGS[L8-G15]="${ARGS[L8]}"
ENV[R8-S2EP]="${ENV[R8]%KILN_COMPILE_FARM=*}KILN_EP_REDUNDANT=1 KILN_EPLB_INTERVAL=16 KILN_COMPILE_FARM=$B/compile-farm/q/pc-8k-1bae0bd/"
ARGS[R8-S2EP]="${ARGS[R8]} --eplb-rebalance"
# R8LK: the final 1M config (2026-10-07): lc2's R8 + LOCAL_K 120 (+ LONG_PIPE, MERGE_BOUND), with the prefill-compute
# flip (KDA gated-norm kernel, delta-rule units 6) as glm5_next defaults; stage graphs in q/sc-pp-final, the single
# engine's (decode + the 4-stage pieces) in q/pcx-fn and q/sc-pp-final. Gate every run on the local-K counters
# ("past F" 0, lc_ttft / check_long print them per DSA stage).
ENV[R8LK]="${ENV[R8]%KILN_COMPILE_FARM=*}KILN_DSA_CP_LOCAL_K=120 KILN_DSA_LONG_PIPE=1 KILN_DSA_CP_MERGE_BOUND=1 KILN_COMPILE_FARM=$B/compile-farm/q/sc-pp-final/"
ARGS[R8LK]="${ARGS[R8]}"
# Bigger prefill calls (feat/prefill-fewer-graphs; code only, nothing captured: q/pfg-* is empty until the farm runs
# them, so ASSERT_CACHE_HIT refuses every call). R8LK at 8192 / 16384-row chunks (C8, C16) to spread the per-call fixed
# costs (each expert's dequantize, the graph launches and their barriers) over 2x / 4x the rows, and each with
# KILN_PREFILL_WHOLE=1 (W: one graph per call instead of prep + 4 pieces + post). A stage holding a quarter of the
# layers has the HBM a bigger chunk's activations want (stage-only loading freed 8.75 GiB per core at 4 stages); one
# engine at 16384 may not. At 16384 an R8 rank's quarter of the chunk is 32 query tiles, past the pipelined DSA
# selection's MAX_TILES, which now runs per block of 16 (models/mla.py _select_tiles). The 8K path: L8 with one
# 8192-row chunk per prompt (L8-C8) and as one graph (L8-W8).
PFG=$B/compile-farm/q/pfg-505b7c9/
ENV[R8LK-C8]="${ENV[R8LK]%KILN_COMPILE_FARM=*}KILN_COMPILE_FARM=$PFG"
ARGS[R8LK-C8]="${ARGS[R8LK]/--prefill-tokens 4096 --prefill-buckets 4096/--prefill-tokens 8192 --prefill-buckets 8192}"
ENV[R8LK-C16]="${ENV[R8LK-C8]}"
ARGS[R8LK-C16]="${ARGS[R8LK]/--prefill-tokens 4096 --prefill-buckets 4096/--prefill-tokens 16384 --prefill-buckets 16384}"
ENV[R8LK-W]="${ENV[R8LK-C8]} KILN_PREFILL_WHOLE=1"; ARGS[R8LK-W]="${ARGS[R8LK]}"
ENV[R8LK-WC8]="${ENV[R8LK-W]}"; ARGS[R8LK-WC8]="${ARGS[R8LK-C8]}"
ENV[R8LK-WC16]="${ENV[R8LK-W]}"; ARGS[R8LK-WC16]="${ARGS[R8LK-C16]}"
ENV[L8-C8]="${L8E%KILN_COMPILE_FARM=*}KILN_COMPILE_FARM=$PFG KILN_PIECEWISE_PREFILL_MOE_GROUP=12"
ARGS[L8-C8]="$L8A --prefill-tokens 8192 --prefill-buckets 8192"
ENV[L8-W8]="${ENV[L8-C8]} KILN_PREFILL_WHOLE=1"; ARGS[L8-W8]="${ARGS[L8-C8]}"
PORT="${PP_PORT:-29600}"
BPORT="${PP_BARRIER_PORT:-29590}"
box=$(cat /opt/kiln/box-name 2>/dev/null || hostname)

pp_args() {  # <stage> <stages> <split> <next ip|-> [sep]: serve_sweep takes the split as one comma list, check_long as
  # separate integers (sep " ")
  local s="$1" n="$2" split="$3" nxt="$4" sep="${5:-,}" a
  a="--pp-stages $n --pp-stage $s --pp-split ${split//,/$sep}"
  [ "$s" -gt 0 ] && a="$a --pp-listen 0.0.0.0:$PORT"
  [ "$nxt" != "-" ] && a="$a --pp-next $nxt:$PORT"
  echo "$a"
}

launch() {  # <log base> <command...>: detached, logs copied to s3 at the end
  local base="$1"; shift
  echo "$*" > $base.log.cmd
  setsid nohup bash -c "cd /opt/kiln/work && $* > $base.log 2>&1; echo \"rc=\$?\" >> $base.log; \
    for f in $base.log $base.log.cmd $base.json $base.timeline.jsonl; do [ -f \$f ] && aws --region us-east-2 s3 cp --quiet \$f $B/logs/$box/; done; \
    echo STATE-DONE >> $base.log" < /dev/null > /dev/null 2>&1 &
  echo $! > $base.pid
  echo "started $(basename $base) pid $!"
}

case "${1:-}" in
  ttft)
    tag="$2"; s="$3"; n="$4"; split="$5"; nxt="$6"; bar="$7"; cfg="${8:-R8}"
    [ -n "${ENV[$cfg]:-}" ] || { echo "unknown config $cfg"; exit 2; }
    base=$L/$tag-S$s
    [ -f /opt/kiln/data/wt8k.npy ] || { mkdir -p /opt/kiln/data && aws --region us-east-2 s3 cp --quiet --recursive \
      $B/logs/pcomm/windows/ /opt/kiln/data/; }
    launch $base env ${ENV[$cfg]} ${PP_EXTRA_ENV:-} KILN_TIMELINE=$base.timeline.jsonl \
      python $SRC/tools/lc_ttft.py --lengths ${PP_LENGTHS:-32768 131072 307200 1044480} \
      --warm-lengths ${PP_WARM:-8192 307200} ${PP_TTFT_ARGS:-} \
      $([ "$n" -gt 1 ] && [ "$bar" != "-" ] && echo "--pp-barrier $bar:$BPORT") \
      --out-json $base.json -- \
      ${ARGS[$cfg]} --output-len 1 ${PP_SWEEP_ARGS:-} $([ "$n" -gt 1 ] && pp_args $s $n $split $nxt)
    ;;
  follow)
    # A following stage (stages 1 .. S - 1 of a --pp-follow pipeline): tools/pp_follow.py until stage 0 closes the run.
    # Stage 0 runs `ttft` with PP_SWEEP_ARGS=--pp-follow and PP_TTFT_ARGS="--gap-s <s>": every stage meets once at the
    # last stage's barrier after its warmup, then stage 0 alone paces the requests.
    tag="$2"; s="$3"; n="$4"; split="$5"; nxt="$6"; bar="$7"; cfg="${8:-R8}"
    base=$L/$tag-S$s
    launch $base env ${ENV[$cfg]} ${PP_EXTRA_ENV:-} KILN_TIMELINE=$base.timeline.jsonl \
      python $SRC/tools/pp_follow.py --out-json $base.json --pp-barrier $bar:$BPORT -- ${ARGS[$cfg]} --output-len 1 \
      --pp-follow ${PP_SWEEP_ARGS:-} \
      $(pp_args $s $n $split $nxt)
    ;;
  warm)
    # tools/lc_warm.py on one engine (no pipeline): turn 0 cold over PP_DOC_LEN tokens, then turns on the same
    # document (prompt + answer + PP_NEW_LEN new tokens) served from the prefix cache and a state checkpoint.
    tag="$2"; cfg="${3:-R8W}"
    base=$L/$tag-S0
    launch $base env ${ENV[$cfg]} ${PP_EXTRA_ENV:-} KILN_TIMELINE=$base.timeline.jsonl \
      python $SRC/tools/lc_warm.py --doc-len ${PP_DOC_LEN:-1040384} --answer-len 16 --new-len ${PP_NEW_LEN:-4096} \
      --out-json $base.json -- ${ARGS[$cfg]}
    ;;
  needle|nll)
    # check_long.py takes its own engine flags (not serve_sweep's): the R8 engine's, as lc2-needle-R8 ran them.
    tag="$2"; s="$3"; n="$4"; split="$5"; nxt="$6"; cfg="${7:-R8}"
    base=$L/$tag-S$s
    [ -f /opt/kiln/data/long.txt ] || { mkdir -p /opt/kiln/data && aws --region us-east-2 s3 cp --quiet $B/data/lc-long.txt /opt/kiln/data/long.txt; }
    launch $base env ${ENV[$cfg]} ${PP_EXTRA_ENV:-} \
      python $SRC/tools/check_long.py $1 --model zai-org/GLM-5.3-Flash --tp 32 --dp-attention 1 --piecewise --overlap \
      --page-size 256 --kv-cache-gb 1.5 --kv-cache-dtype fp8 --prefill-tokens 4096 --max-num-seqs 4 --page-buckets 1024 4096 \
      --max-model-len 1048576 --decode-buckets 1 --state-checkpoints 0 --text-file /opt/kiln/data/long.txt \
      $([ "$1" = nll ] && echo "${PP_NLL_ARGS:---max-tokens 131072}" || echo "${PP_NEEDLE_ARGS:---forced --lengths 131072 1044480 --depths 0.1 0.5 0.9}") \
      $(pp_args $s $n $split $nxt " ") --out-json $base.json
    ;;
  prompts)
    # /opt/kiln/data/pp-prompts.npy: [3, 1044480] int32 seeded random ids in [1000, 100000) (row 0 a warm-up, rows 1
    # and 2 the 128k / 1M prompts); the same file on every box, for pd_client.py and the one-engine reference.
    mkdir -p /opt/kiln/data
    python -c "import numpy as np; np.save('/opt/kiln/data/pp-prompts.npy', np.random.default_rng(0).integers(1000, 100000, (3, 1044480), dtype=np.int32))"
    python -c "import numpy as np; a = np.load('/opt/kiln/data/pp-prompts.npy'); print(a.shape, a[:, :4].tolist(), int(a.sum()))"
    ;;
  serve0)
    # Stage 0 of a disaggregated, following pipeline: the prefill server the PD router posts to.
    tag="$2"; n="$3"; split="$4"; nxt="$5"; cfg="${6:-R8}"
    base=$L/$tag-S0
    launch $base env ${ENV[$cfg]} ${PP_EXTRA_ENV:-} KILN_TIMELINE=$base.timeline.jsonl \
      python $SRC/bench/pd_serve.py --pd-role prefill --port ${PP_HTTP_PORT:-8100} -- ${ARGS[$cfg]} --pp-follow \
      $(pp_args 0 $n $split $nxt)
    ;;
  servef)
    # A following stage of that pipeline, with the /health the router checks.
    tag="$2"; s="$3"; n="$4"; split="$5"; nxt="$6"; cfg="${7:-R8}"
    base=$L/$tag-S$s
    launch $base env ${ENV[$cfg]} ${PP_EXTRA_ENV:-} KILN_TIMELINE=$base.timeline.jsonl \
      python $SRC/tools/pp_follow.py --pd-role prefill --health-port ${PP_HTTP_PORT:-8100} --out-json $base.json -- \
      ${ARGS[$cfg]} --pp-follow $(pp_args $s $n $split $nxt)
    ;;
  decode)
    # The decode engine the pipeline's stages hand off to (and, with PP_ROLE=none, the one-engine reference server).
    tag="$2"; cfg="${3:-R8}"; role="${PP_ROLE:-decode}"
    base=$L/$tag-D
    extra=""; [ "$role" = decode ] && extra="--pd-listen 0.0.0.0:7400 --pd-advertise $(hostname -I | awk '{print $1}'):7400 --pd-buffer-gb ${PP_PD_BUFFER_GB:-64}"
    launch $base env ${ENV[$cfg]} ${PP_EXTRA_ENV:-} KILN_TIMELINE=$base.timeline.jsonl \
      python $SRC/bench/pd_serve.py --pd-role $role $extra --port ${PP_HTTP_PORT:-8100} -- ${ARGS[$cfg]}
    ;;
  router)
    tag="$2"; units="$3"; dec="$4"
    base=$L/$tag-R
    launch $base python -m kiln.server.pd_router --prefill-units "$units" --decode-urls "$dec" --threshold 0 \
      --port ${PP_ROUTER_PORT:-8000}
    ;;
  client)
    # tools/pd_client.py against a server (the router, or the reference): rows 1 and 2 of pp-prompts.npy.
    tag="$2"; url="$3"
    base=$L/$tag-C
    [ -f /opt/kiln/data/wt8k.npy ] || { mkdir -p /opt/kiln/data && aws --region us-east-2 s3 cp --quiet --recursive \
      $B/logs/pcomm/windows/ /opt/kiln/data/; }
    launch $base python $SRC/tools/pd_client.py --url $url --ids-npy ${PP_NPY:-/opt/kiln/data/pp-prompts.npy} \
      --warm-rows ${PP_WARM_ROWS:-0} --warm-lengths ${PP_WARM_LENGTHS:-8192} --rows ${PP_ROWS:-1 2} \
      --lengths ${PP_LENGTHS:-131072 1044480} \
      --max-tokens ${PP_MAX_TOKENS:-64} --gap-s ${PP_GAP:-15} --out-json $base.json
    ;;
  stop)
    # TERM the python under each launched wrapper first, so the server exits cleanly (its KILN_TIMELINE is written at
    # exit) and the wrapper still copies the logs to s3; the whole group goes only if it has not ended within 120 s.
    for f in $L/*.pid; do
      [ -f $f ] || continue
      pid=$(cat $f); pkill -TERM -P $pid 2>/dev/null
      for _ in $(seq 1 120); do kill -0 $pid 2>/dev/null || break; sleep 1; done
      kill -TERM -- -$pid 2>/dev/null; sleep 2; kill -KILL -- -$pid 2>/dev/null; rm -f $f
    done
    pkill -f "$SRC/tools/(lc_ttft|check_long|pp_follow|pd_client).py" 2>/dev/null
    pkill -f "$SRC/bench/pd_serve.py" 2>/dev/null
    pkill -f "kiln.server.pd_router" 2>/dev/null
    echo stopped
    ;;
  status)
    f=$L/$2-S$3.log
    [ -f $f ] && grep -vE "Warning|warn|INFO" $f | grep -E "lone request|warm request|RESULT|rc=|STATE-DONE|Error|error|Traceback|needle" | tail -n ${4:-12} | cut -c1-400
    ;;
  *) sed -n 2,20p "$0"; exit 2 ;;
esac
