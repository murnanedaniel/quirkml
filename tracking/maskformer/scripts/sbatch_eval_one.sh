#!/bin/bash
#SBATCH -A m4958 -C cpu -N 1
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/maskformer/logs/slurm-%x-%j.out
# One evaluate.py run (CPU, whole node). Usage: sbatch -q debug|regular -t T -J NAME sbatch_eval_one.sh PRED_DIR DATASET A B_EXCL OUT JOBS STAGING
set -uo pipefail
umask 007
P=$1; DS=$2; A=$3; B=$4; O=$5; J=$6; STG=$7
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml
/usr/bin/time -v /global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python tracking/evaluate.py \
  --predictions-dir $P --dataset $DS --events $A $B --out $O --jobs $J --staging-dir $STG
