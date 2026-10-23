#!/bin/bash
# T2 long run: chained gpu_shared_interactive windows (m5386) for one resumable lane.
# One GPU per window. EXTRA_CMD (e.g. the test-set eval of the scan winner) runs once, at the start of the first window.
# Usage: long_driver.sh TAG MAX_LR MAX_STEPS VAL_INTERVAL ["EXTRA_CMD"]     Stop file: $OUT/STOP_DRIVER
set -uo pipefail
TAG=$1; LR=$2; STEPS=$3; VCI=$4; EXTRA=${5:-}
MF=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer
OUT=/pscratch/sd/d/danieltm/quirk_tracking/maskformer
MAINT=$(date -d 2026-10-21T05:30 +%s)
log() { echo "[long] $(date -Is) $*"; }
NOENV() { env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /') "$@"; }
while true; do
  [ -f $OUT/STOP_DRIVER ] && { log "stop file"; exit 0; }
  [ -f $OUT/runs/$TAG/TRAIN_DONE ] && [ -f $OUT/runs/$TAG/EVAL_DONE ] && [ -z "$EXTRA" ] && { log "done"; exit 0; }
  NOW=$(date +%s); LIMIT=240
  if [ $(( NOW + 4*3600 )) -gt $MAINT ]; then LIMIT=$(( (MAINT - NOW) / 60 - 10 )); fi
  [ $LIMIT -lt 30 ] && { log "too close to maintenance; stopping"; exit 0; }
  G=1
  JOB=$(salloc -A m5386 -C gpu -q shared_interactive --gpus $G -c $((32*G)) -t $LIMIT --no-shell -J qtrk_mf 2>&1 | grep -o 'Granted job allocation [0-9]*' | grep -o '[0-9]*$')
  [ -z "$JOB" ] && { log "salloc failed; retry in 120 s"; sleep 120; continue; }
  log "window $JOB granted ($G GPU, $LIMIT min)"
  if [ -n "$EXTRA" ]; then
    NOENV srun --jobid=$JOB --exact -n1 -c32 --gpus=1 bash -c "$EXTRA" >> $OUT/logs/long_extra.log 2>&1
    EXTRA=""
  fi
  PIDS=()
  if [ ! -f $OUT/runs/$TAG/EVAL_DONE ]; then
    NOENV srun --jobid=$JOB --exact -n1 -c32 --gpus=1 env MAX_STEPS=$STEPS VAL_INTERVAL=$VCI bash $MF/scripts/lane.sh $TAG $LR >> $OUT/logs/lanes_$TAG.log 2>&1 &
    PIDS+=($!)
  fi
  for p in "${PIDS[@]}"; do wait $p; done
  scancel $JOB; sleep 5
  sacct -j $JOB -X -n -o Start,End,ElapsedRaw | awk -v j=$JOB -v g=$G -v t=$TAG '{printf "%s %s %s %d shared_interactive(m5386) T2 long %s gpu_hours=%.3f\n", j, $1, $2, g, t, g*$3/3600}' >> $OUT/gpu_hours.txt
  log "window $JOB closed"
done
