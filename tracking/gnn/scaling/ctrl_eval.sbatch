#!/bin/bash
#SBATCH -A m4958 -C cpu -q regular -N 1 -t 01:30:00 -J qtrk_scal_ctrl
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/scaling/logs/ctrl_eval_%j.out
# 16k controls (existing models) through the scaling eval script: val curve points + test reproduction.
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
S=/pscratch/sd/d/danieltm/quirk_tracking/scaling
R=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/scaling
for spec in g2_q50_dir_s0:v3 g2_q50_dir_s1:v3 e2_pair_dir_s0:pair e2_pair_dir_s1:pair e2_pair_dir_s2:pair; do
  IFS=: read T K <<< "$spec"
  [ -f $S/eval/ctrl_$T/test_eval/summary.json ] && continue
  TAG=ctrl_$T KIND=$K SCORED=$O/main/$T/gnn/scored OUTD=$S/eval/ctrl_$T bash $R/eval.sbatch
  echo "$T rc=$?"
done
