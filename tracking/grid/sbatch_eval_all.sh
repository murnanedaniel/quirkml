#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 8 -t 00:30:00 -J qtrk_F_eval
#SBATCH -o /pscratch/sd/d/danieltm/quirk_stageF/logs/eval_all_%j.out
# grid_eval.sh on every set with predictions and not yet scored: node k takes every 8th such set, 2 at a time (60 workers each).
G=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/grid
DATA=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data
L=/pscratch/sd/d/danieltm/quirk_stageF/logs/eval_list_$SLURM_JOB_ID.txt
for ds in $(cat $G/grid_sets_mu0.txt); do
  [ -f $DATA/grid_predictions/mf_pairdir_long/$ds/ot0.2/predictions.parquet ] || continue
  [ -f $DATA/grid_eval/mf_pairdir_long/$ds/python_ot0.2.json ] && [ -f $DATA/grid_eval/mf_pairdir_long/$ds/pair_ot0.2/summary.json ] && continue
  echo $ds
done > $L
echo "$(wc -l < $L) sets to score"
for k in 0 1 2 3 4 5 6 7; do
  srun -N1 -n1 -r $k --exact -c 256 bash -c "awk 'NR % 8 == $k' $L | xargs -P 2 -I{} bash $G/grid_eval.sh {} 60" &
done
wait
