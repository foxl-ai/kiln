#!/bin/bash
# The execution-barrier measurements on trn2.48xlarge, on logical cores 32-63 only (KILN_CORE_BASE=32), in the order
# tools/hv_barrier_probe.sh (default / off / hardware barrier), tools/hv_mismatch.sh (race with the default and the
# hardware barrier, shape mismatch with the hardware barrier; the barrier-off race is not run on a shared box), then
# tools/hv_ab_t2.sh serving at concurrency 64 with the default and the hardware barrier. Logs under
# s3://<your-bucket>/logs/kiln-hv-t2/.
set -u
S="${KILN_HV_SRC:-/opt/kiln/src-hv}/tools"
export KILN_CORE_BASE=32 KILN_HV_BOX=kiln-hv-t2
HW=NEURON_RT_DISABLE_EXECUTION_BARRIER=0,NEURON_RT_ENABLE_HW_EXECUTION_BARRIER=1
bash $S/hv_barrier_probe.sh base off hw
bash $S/hv_mismatch.sh "t2-base-race:race:900:NEURON_RT_DISABLE_EXECUTION_BARRIER=0,NEURON_RT_ENABLE_HW_EXECUTION_BARRIER=0 t2-hw-race:race:900:$HW t2-hw-shape:shape:120:$HW"
bash $S/hv_ab_t2.sh base 64 NEURON_RT_DISABLE_EXECUTION_BARRIER=0 NEURON_RT_ENABLE_HW_EXECUTION_BARRIER=0
sleep 90
bash $S/hv_ab_t2.sh hw 64 NEURON_RT_DISABLE_EXECUTION_BARRIER=0 NEURON_RT_ENABLE_HW_EXECUTION_BARRIER=1
echo HV-T2-ALL-DONE
