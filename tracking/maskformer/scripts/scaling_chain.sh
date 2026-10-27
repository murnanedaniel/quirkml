#!/bin/bash
# Keep one gpu_shared batch job per data-scaling arm until its lane has TRAIN_DONE (or TRAIN_FAILED); resubmit with
# resume after a time limit, at most MAXSUB submissions per arm. Logs GPU-hours of every finished job.
set -f  # no globbing of the [a,b] CLI values
MF=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/maskformer
OUT=/pscratch/sd/d/danieltm/quirk_tracking/maskformer
C=$MF/packages/ml_tracking/src/ml_tracking/experiments/quirk/configs
H=$OUT/gpu_hours.txt; MAXSUB=4
# Budget rule (2026-10-27, common with the GNN study): plateau of val/coupled_strict, patience 3 validation points every
# 2000 steps, cap 120k steps (4x the 16k control); configs/plateau.yaml.
P="--config $C/quirk_pair.yaml --config $C/scaling.yaml --config $C/plateau.yaml"
ARMS=("s2_pair_n16k|5e-5|120000|2000|10:00:00|$P"
      "s2_pair_n50k|5e-5|120000|2000|10:00:00|$P --data.train_extra_events [20000,53999]"
      "s2_pair_n150k|5e-5|120000|2000|10:00:00|$P --data.train_extra_events [20000,153999]")
declare -A job nsub
while true; do
  alldone=1
  for a in "${ARMS[@]}"; do
    IFS='|' read -r tag lr steps vci tl extra <<< "$a"
    R=$OUT/runs/$tag
    { [ -f $R/TRAIN_DONE ] || [ -f $R/TRAIN_FAILED ]; } && continue
    alldone=0
    j=${job[$tag]:-}
    if [ -n "$j" ] && squeue -h -j $j 2>/dev/null | grep -q .; then continue; fi
    if [ -n "$j" ]; then
      sacct -j $j -X -n -o Start,End,ElapsedRaw,State | awk -v j=$j -v t=$tag '{printf "%s %s %s 1 gpu_shared(m5386) data scaling %s (%s) gpu_hours=%.3f\n", j, $1, $2, t, $4, $3/3600}' >> $H
    fi
    n=${nsub[$tag]:-0}
    [ $n -ge $MAXSUB ] && { echo "$(date -Is) $tag: $n submissions without finishing; giving up"; touch $R/TRAIN_FAILED; continue; }
    j=$(sbatch --parsable -t $tl -J qtrk_$tag $MF/scripts/sbatch_lane.sh $tag $lr $steps $vci $extra) || { sleep 120; continue; }
    job[$tag]=$j; nsub[$tag]=$((n+1)); echo "$(date -Is) $tag: submitted $j (submission $((n+1)))"
  done
  [ $alldone = 1 ] && { echo "$(date -Is) all arms trained"; for t in "${!job[@]}"; do j=${job[$t]}; sacct -j $j -X -n -o Start,End,ElapsedRaw,State | awk -v j=$j -v t=$t '{printf "%s %s %s 1 gpu_shared(m5386) data scaling %s (%s) gpu_hours=%.3f\n", j, $1, $2, t, $4, $3/3600}' >> $H; done; exit 0; }
  sleep 300
done
