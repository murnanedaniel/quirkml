#!/bin/bash
# Score grid sets as predictions land: debug-qos 8-node jobs (m4958) of up to 40 sets each, at most MAXJOBS at once.
# A set is claimed by writing its job id to grid_eval/.../<ds>/.job. When that job has left the queue and the set has no
# complete summaries, the attempt counts as a failure (.fails); a set is given up after 2 failures (listed in the log).
G=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/grid
DATA=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data
EV=$DATA/grid_eval/mf_pairdir_long
H=/pscratch/sd/d/danieltm/quirk_tracking/maskformer/gpu_hours.txt
MAXJOBS=${MAXJOBS:-2}; MAXFAIL=2; BATCH=24
complete() { for f in pair_ot0.1 pair_ot0.2 truth_ot0.1 truth_ot0.2; do [ -f $EV/$1/$f/summary.json ] || return 1; done; [ -f $EV/$1/python_ot0.2.json ]; }
declare -A logged
while true; do
  free=() ; nscored=0; npred=0; failed=()
  for ds in $(cat $G/grid_sets_mu0.txt); do
    mkdir -p $EV/$ds
    complete $ds && { nscored=$((nscored+1)); rm -f $EV/$ds/.job; continue; }
    [ -f $DATA/grid_predictions/mf_pairdir_long/$ds/ot0.2/predictions.parquet ] || continue
    npred=$((npred+1))
    if [ -f $EV/$ds/.job ]; then
      j=$(cat $EV/$ds/.job)
      squeue -h -j $j 2>/dev/null | grep -q . && continue
      echo $(( $(cat $EV/$ds/.fails 2>/dev/null || echo 0) + 1 )) > $EV/$ds/.fails; rm -f $EV/$ds/.job
      echo "$(date -Is) $ds: job $j ended without complete summaries (failure $(cat $EV/$ds/.fails))"
    fi
    [ "$(cat $EV/$ds/.fails 2>/dev/null || echo 0)" -ge $MAXFAIL ] && { failed+=($ds); continue; }
    free+=($ds)
  done
  njobs=$(squeue -h -u danieltm -n qtrk_F_evalr | wc -l)
  echo "$(date -Is) scored $nscored/110, predicted-unscored $npred (unclaimed ${#free[@]}), jobs $njobs, given up: ${failed[*]:-none}"
  while [ ${#free[@]} -gt 0 ] && [ $njobs -lt $MAXJOBS ]; do
    batch=("${free[@]:0:$BATCH}"); free=("${free[@]:$BATCH}")
    j=$(sbatch --parsable $G/sbatch_eval_sets.sh "${batch[@]}") || break
    for ds in "${batch[@]}"; do echo $j > $EV/$ds/.job; done
    echo "$(date -Is) job $j: ${batch[*]}"; njobs=$((njobs+1))
  done
  [ $nscored -eq 110 ] && { echo "all done"; exit 0; }
  [ $((nscored + ${#failed[@]})) -eq 110 ] && { echo "done with failures: ${failed[*]}"; exit 1; }
  for j in $(sacct -u danieltm -X -n -S 2026-10-26T14:40 -o JobID,JobName%20,State | awk '$2=="qtrk_F_evalr" && $3!="RUNNING" && $3!="PENDING" {print $1}'); do
    [ -n "${logged[$j]:-}" ] && continue; logged[$j]=1
    sacct -j $j -X -n -o JobID,Start,End,ElapsedRaw | awk '{printf "%s %s %s 0 debug(m4958,CPU 8 nodes) Stage F eval (cpu_node_hours=%.3f)\n", $1, $2, $3, 8*$4/3600}' >> $H
  done
  sleep 120
done
