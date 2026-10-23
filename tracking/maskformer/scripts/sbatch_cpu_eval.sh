#!/bin/bash
#SBATCH -A m4958
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J qtrk_mf_ceval
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/maskformer/logs/slurm-%x-%j.out
# CPU evaluate.py on existing prediction dirs. Usage: sbatch sbatch_cpu_eval.sh FIRST LAST_EXCL PRED_NAME...
set -uo pipefail
A=$1; B=$2; shift 2
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml
OUT=/pscratch/sd/d/danieltm/quirk_tracking/maskformer
ROOT=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1
EVPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
cd $Q
for N in "$@"; do
  $EVPY tracking/evaluate.py --predictions-dir $OUT/predictions/$N --dataset $ROOT --events $A $B \
    --out $OUT/eval/$N --jobs 60 --staging-dir $OUT/staging_cpu_$SLURM_JOB_ID > $OUT/logs/eval_$N.log 2>&1
  echo "[$N] rc=$? $(grep '^very_loose' $OUT/logs/eval_$N.log)"
done
