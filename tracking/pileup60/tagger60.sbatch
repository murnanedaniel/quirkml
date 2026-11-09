#!/bin/bash
#SBATCH -A m4958 -C cpu -q regular -N 1 -c 256 -t 02:00:00 -J qtrk_pu60_tagger
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/tagger_%j.out
# usage: sbatch tagger60.sbatch RUN_DIR CUT NAME
R=$1; CC=$2; N=$3
/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/tagger60.py \
  $N $R/val/pred_cc${CC}_mh4 $R/test/pred_cc${CC}_mh4 $R/scored /pscratch/sd/d/danieltm/quirk_pu60/tagger \
  /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/models/tagger/b3_$N
