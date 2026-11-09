#!/bin/bash
#SBATCH -A m4958 -C cpu -q regular -N 1 -c 256 -t 01:30:00 -J qtrk_pu60_roc
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/roc_%j.out
R=/pscratch/sd/d/danieltm/quirk_pu60/gnn/pair_16k_s0
/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/roc60.py \
  pair60_s0_cc0.6 $R/val/pred_cc0.6_mh4 $R/test/pred_cc0.6_mh4 $R/scored /pscratch/sd/d/danieltm/quirk_pu60/roc
