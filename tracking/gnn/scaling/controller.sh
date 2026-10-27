#!/bin/bash
# Renews 2-GPU gpu_shared_interactive windows (m5386) while arms in arm_queue.txt remain. Waits for a free slot (per-user
# limit 2 jobs on that QOS), starts a window only when an arm is runnable, logs GPU-hours per window, stops after two
# windows in a row without progress (sum of trained steps + finished arms) or 3 windows that never started.
S=/pscratch/sd/d/danieltm/quirk_tracking/scaling
R=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/scaling
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
log() { echo "$(date '+%F %T') [controller] $*" >> $S/logs/windows.log; }
progress() {
  local tot=0
  for d in $S/runs/*/; do
    [ -f $d/driver.log ] && grep -q finished $d/driver.log && tot=$(( tot + 1000000 ))
    [ -f $d/gnn/state.pt ] && tot=$(( tot + $($PY -c "import torch;print(torch.load('$d/gnn/state.pt',weights_only=False,map_location='cpu')['step'])" 2>/dev/null || echo 0) ))
  done; echo $tot
}
remaining() { local n=0; while read TAG rest; do [ -z "$TAG" ] || [ "${TAG:0:1}" = "#" ] && continue
  [ -f $S/runs/$TAG/FAILED ] && continue; grep -qs finished $S/runs/$TAG/driver.log || n=$((n+1)); done < $S/arm_queue.txt; echo $n; }
runnable() { while read TAG KIND STEPS SEED NEED RANGES; do [ -z "$TAG" ] || [ "${TAG:0:1}" = "#" ] && continue
  [ -f $S/runs/$TAG/FAILED ] && continue; grep -qs finished $S/runs/$TAG/driver.log && continue
  GR=$S/graphs/$KIND; { [ -f $GR/.complete ] || [ -f $GR/events$(( (NEED+1)*1000 ))-$(( (NEED+1)*1000+999 )).parquet ]; } && return 0
  done < $S/arm_queue.txt; return 1; }
noprog=0; nostart=0
log "start pid $$"
while [ $(remaining) -gt 0 ]; do
  [ -f $S/STOP ] && { log "STOP file, exit"; exit 0; }
  WS=window.sh; { [ -f $S/graphs/v3/.complete ] && [ -f $S/graphs/pair/.complete ]; } || WS=window0.sh
  if [ $WS = window.sh ] && ! runnable; then sleep 120; continue; fi
  # quirk work holds ONE gpu_shared_interactive slot (coordinator, 2026-10-27): request only when no other qtrk_* job
  # holds one (e.g. the splitter's qtrk_split2 has ended and is not being renewed) and the per-user limit (2) allows it
  if [ $(squeue -h -u danieltm -q gpu_shared_interactive -o %j | grep -c "^qtrk_") -ge 1 ] || \
     [ $(squeue -h -u danieltm -q gpu_shared_interactive | wc -l) -ge 2 ]; then sleep 120; continue; fi
  p0=$(progress)
  log "requesting window $WS (progress $p0)"
  env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /') salloc -A m5386 -C gpu -q shared_interactive -N 1 -n 2 -c 32 \
    --gpus=2 -t 04:00:00 -J qtrk_scal_win bash $R/$WS >> $S/logs/windows_salloc.log 2>&1
  rc=$?
  J=$(grep -o "Granted job allocation [0-9]*" $S/logs/windows_salloc.log | tail -1 | awk '{print $4}')
  log "window ended rc $rc job $J"
  if [ -n "$J" ] && ! grep -qs "^$J " $S/gpu_hours.txt; then
    line=$(sacct -j $J -X -n -P -o Start,End,ElapsedRaw)
    IFS='|' read st en el <<< "$line"
    [ -n "$el" ] && echo "$J gpu_shared_interactive(m5386) 2 $st $en $(python3 -c "print(round(2*$el/3600,2))") scaling arms window (streamed GNN training + inference)" >> $S/gpu_hours.txt
  fi
  p1=$(progress)
  if [ -z "$J" ] || [ "$J" = "$LASTJ" ]; then nostart=$((nostart+1)); [ $nostart -ge 3 ] && { log "3 windows never started, exit"; exit 1; }; sleep 300; continue; fi
  LASTJ=$J; nostart=0
  if [ "$p1" -le "$p0" ]; then noprog=$((noprog+1)); log "no progress ($p0 -> $p1), count $noprog"; [ $noprog -ge 2 ] && { log "two windows without progress, exit"; exit 1; }
  else noprog=0; fi
done
log "all arms done or failed, exit"
