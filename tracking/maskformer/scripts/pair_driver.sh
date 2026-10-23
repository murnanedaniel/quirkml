#!/bin/bash
# Chained 2-GPU gpu_shared_interactive windows (m5386). GPU A: the long run (one resumable lane).
# GPU B: a queue of 30k-step lanes run one after another (each resumable; done lanes are skipped).
# A lane spec is TAG|MAX_LR|EXTRA_ARGS (EXTRA_ARGS e.g. "--config .../quirk_weight10.yaml").
# Usage: pair_driver.sh "LONGSPEC" LONG_STEPS LONG_VAL_INTERVAL "SPEC_B1" "SPEC_B2" ...   Stop file: $OUT/STOP_DRIVER
set -uo pipefail
LONG=$1; LSTEPS=$2; LVCI=$3; shift 3
MF=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer
OUT=/pscratch/sd/d/danieltm/quirk_tracking/maskformer
MAINT=$(date -d 2026-10-21T05:30 +%s)
log() { echo "[pair] $(date -Is) $*"; }
NOENV() { env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /') "$@"; }
done_lane() { [ -f $OUT/runs/$1/TRAIN_DONE ] && [ -f $OUT/runs/$1/EVAL_DONE ]; }
run_spec() {  # TAG|LR|EXTRA [STEPS VCI]
  local spec=$1 steps=${2:-30000} vci=${3:-1000}
  local tag=${spec%%|*}; local rest=${spec#*|}; local lr=${rest%%|*}; local extra=${rest#*|}
  [ "$extra" = "$rest" ] && extra=""
  done_lane $tag && return 0
  MAX_STEPS=$steps VAL_INTERVAL=$vci bash $MF/scripts/lane.sh $tag $lr $extra >> $OUT/logs/lanes_$tag.log 2>&1
}
export -f run_spec done_lane; export MF OUT
while true; do
  [ -f $OUT/STOP_DRIVER ] && { log "stop file"; exit 0; }
  LT=${LONG%%|*}; BTODO=0
  for s in "$@"; do done_lane ${s%%|*} || BTODO=1; done
  if done_lane $LT && [ $BTODO = 0 ]; then log "all done"; exit 0; fi
  NOW=$(date +%s); LIMIT=240
  [ $(( NOW + 4*3600 )) -gt $MAINT ] && LIMIT=$(( (MAINT - NOW) / 60 - 10 ))
  [ $LIMIT -lt 30 ] && { log "too close to maintenance; stopping"; exit 0; }
  JOB=$(salloc -A m5386 -C gpu -q shared_interactive --gpus 2 -c 64 -t $LIMIT --no-shell -J qtrk_mf 2>&1 | grep -o 'Granted job allocation [0-9]*' | grep -o '[0-9]*$')
  [ -z "$JOB" ] && { log "salloc failed; retry in 120 s"; sleep 120; continue; }
  log "window $JOB granted ($LIMIT min)"
  T0=$(date +%s); PIDS=()
  if ! done_lane $LT; then
    NOENV srun --jobid=$JOB --exact -n1 -c32 --gpus=1 bash -c "run_spec '$LONG' $LSTEPS $LVCI" &
    PIDS+=($!)
  fi
  if [ $BTODO = 1 ]; then
    Q=""; for s in "$@"; do Q="$Q run_spec '$s';"; done
    NOENV srun --jobid=$JOB --exact -n1 -c32 --gpus=1 bash -c "$Q" &
    PIDS+=($!)
  fi
  for p in "${PIDS[@]}"; do wait $p; done
  scancel $JOB; sleep 5
  sacct -j $JOB -X -n -o Start,End,ElapsedRaw | awk -v j=$JOB -v l="$LT + $*" '{printf "%s %s %s 2 shared_interactive(m5386) T2 pair: %s gpu_hours=%.3f\n", j, $1, $2, l, 2*$3/3600}' >> $OUT/gpu_hours.txt
  log "window $JOB closed"
  [ $(( $(date +%s) - T0 )) -lt 240 ] && { log "window under 4 min: lanes failing; stopping"; exit 1; }
done
