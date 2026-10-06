#!/bin/bash
# Run a list of harvest steps back to back on one box, a pause between 32-rank jobs (CLAUDE.md: a 32-rank job right
# after another can fail nrt_init). Each step is "ab <cfg> <tag> [VAR=v ...]" or "q <what> <tag> [VAR=v ...]",
# separated by ";" in one argument:
#   bash tools/hv_chain.sh "ab G64 base; ab G64 off NEURON_RT_DISABLE_EXECUTION_BARRIER=1; q long off NEURON_RT_DISABLE_EXECUTION_BARRIER=1"
set -u
SRC="${KILN_HV_SRC:-/opt/kiln/src-hv}"
IFS=';' read -ra steps <<< "$1"
for s in "${steps[@]}"; do
  set -- $s
  [ $# -eq 0 ] && continue
  kind="$1"; shift
  case "$kind" in
    ab) bash $SRC/tools/hv_ab.sh "$@" ;;
    q) bash $SRC/tools/hv_quality.sh "$@" ;;
    *) echo "unknown step kind $kind" ;;
  esac
  sleep 90
done
echo HV-CHAIN-DONE
