#!/bin/bash
# Runs inside one 2-GPU gpu_shared_interactive window: two lanes, each takes the next arm of arm_queue.txt that is not
# done, not taken in this window, not FAILED and whose graph shards exist; arms resume across windows.
S=/pscratch/sd/d/danieltm/quirk_tracking/scaling
R=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/scaling
END=$(( $(date +%s) + ${WIN_S:-14400} ))
log() { echo "$(date '+%F %T') [win $SLURM_JOB_ID lane $1] $2" >> $S/logs/windows.log; }
ready() {  # kind need
  local GR=$S/graphs/$1; [ -f $GR/.complete ] || [ -f $GR/events$(( ($2+1)*1000 ))-$(( ($2+1)*1000+999 )).parquet ]
}
lane() {
  local L=$1
  while true; do
    [ $(( END - $(date +%s) )) -lt 1800 ] && { log $L "less than 30 min left, lane stops"; return; }
    local picked="" waiting=0
    while read TAG KIND STEPS SEED NEED RANGES; do
      [ -z "$TAG" ] || [ "${TAG:0:1}" = "#" ] && continue
      M=$S/runs/$TAG
      [ -f $M/driver.log ] && grep -q finished $M/driver.log && continue
      [ -f $M/FAILED ] && continue
      ready $KIND $NEED || { waiting=1; continue; }
      if [ -f $M/RUNNING ] && [ "$(cat $M/RUNNING)" != "$SLURM_JOB_ID" ] && squeue -h -j $(cat $M/RUNNING) 2>/dev/null | grep -q .; then continue; fi
      mkdir -p $M; mkdir $M/.claim_$SLURM_JOB_ID 2>/dev/null || continue
      picked="$TAG $KIND $STEPS $SEED $NEED $RANGES"; break
    done < $S/arm_queue.txt
    if [ -z "$picked" ]; then
      [ $waiting = 1 ] && { sleep 60; continue; }
      log $L "nothing left to run, lane stops"; return
    fi
    read TAG KIND STEPS SEED NEED RANGES <<< "$picked"
    log $L "run $TAG"
    TAG=$TAG KIND=$KIND STEPS=$STEPS SEED=$SEED RANGES="$RANGES" DEADLINE=$(( END - 900 )) SRUN_OPTS="--exact" \
      bash $R/arm_run.sh
    rc=$?
    log $L "$TAG exit $rc"
    if [ $rc -eq 3 ]; then return; fi
    if [ $rc -ne 0 ]; then
      echo "$(date '+%F %T') rc $rc job $SLURM_JOB_ID" >> $S/runs/$TAG/failures
      [ $(wc -l < $S/runs/$TAG/failures) -ge 2 ] && { touch $S/runs/$TAG/FAILED; log $L "$TAG FAILED twice, skipped"; }
    fi
  done
}
log - "window start"
for l in ${LANES:-0 1}; do lane $l & sleep 5; done; wait
log - "window end"
