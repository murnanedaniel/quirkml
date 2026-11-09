#!/bin/bash
#SBATCH -A m4958 -C cpu -q regular -N 1 -c 256 -t 02:00:00 -J qtrk_pu60_roc
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/roc_%j.out
R=${R:-/pscratch/sd/d/danieltm/quirk_pu60/gnn/pair_16k_s0}; N=${N:-pair60_s0_cc0.6}; CC=${CC:-0.6}
/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/roc60.py \
  $N $R/val/pred_cc${CC}_mh4 $R/test/pred_cc${CC}_mh4 $R/scored /pscratch/sd/d/danieltm/quirk_pu60/roc
