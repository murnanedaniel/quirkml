#!/bin/bash
# Chained gpu_shared_interactive windows (m5386, 2 GPUs, 4 h): two lanes per window, one per GPU,
# until every lane has TRAIN_DONE and EVAL_DONE. Launch with setsid nohup from a login shell.
# Usage: window_driver.sh TAG1:LR1 TAG2:LR2 ...      Stop file: $OUT/STOP_DRIVER
set -uo pipefail
MF=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer
OUT=/pscratch/sd/d/danieltm/quirk_tracking/maskformer
HOURS=$OUT/gpu_hours.txt
MAINT=$(date -d "${MAINT_AT:-2099-01-01T00:00}" +%s)  # next maintenance start (MAINT_AT env)
log() { echo "[driver] $(date -Is) $*"; }
done_lane() { [ -f $OUT/runs/$1/TRAIN_DONE ] && [ -f $OUT/runs/$1/EVAL_DONE ]; }
while true; do
  [ -f $OUT/STOP_DRIVER ] && { log "stop file"; exit 0; }
  TODO=()
  for spec in "$@"; do done_lane ${spec%%:*} || TODO+=("$spec"); done
  [ ${#TODO[@]} -eq 0 ] && { log "all lanes done"; exit 0; }
  if [ $(( $(date +%s) + 4*3600 )) -gt $MAINT ]; then log "window would cross maintenance; stopping"; exit 0; fi
  LANES=("${TODO[@]:0:2}")
  log "requesting window for ${LANES[*]}"
  JOB=$(salloc -A m5386 -C gpu -q shared_interactive --gpus 2 -c 64 -t 04:00:00 --no-shell -J qtrk_mf 2>&1 | grep -o 'Granted job allocation [0-9]*' | grep -o '[0-9]*$')
  [ -z "$JOB" ] && { log "salloc failed; retry in 120 s"; sleep 120; continue; }
  log "window $JOB granted"
  PIDS=()
  for spec in "${LANES[@]}"; do
    env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /') srun --jobid=$JOB --exact -n1 -c32 --gpus=1 \
      bash $MF/scripts/lane.sh ${spec%%:*} ${spec##*:} >> $OUT/logs/lanes_${spec%%:*}.log 2>&1 &
    PIDS+=($!)
  done
  for p in "${PIDS[@]}"; do wait $p; done
  scancel $JOB; sleep 5
  sacct -j $JOB -X -n -o Start,End,ElapsedRaw | awk -v j=$JOB -v l="${LANES[*]}" '{printf "%s %s %s 2 shared_interactive(m5386) T2 lanes: %s gpu_hours=%.3f\n", j, $1, $2, l, 2*$3/3600}' >> $HOURS
  log "window $JOB closed"
  EL=$(sacct -j $JOB -X -n -o ElapsedRaw | head -1 | tr -d ' ')
  if [ "${EL:-0}" -lt 240 ]; then log "window lasted ${EL}s (<4 min): lanes are failing; stopping"; exit 1; fi
done
