#!/bin/bash
#SBATCH -A m4958 -C cpu -q regular -N 1 -t 04:00:00
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/eval_%x_%j.out
# One strict ACTS evaluation at mu 60 (whole CPU node, 16 workers: ~27 GB per worker). Usage: sbatch -J NAME eval_one.sbatch PRED DATASET A B_EXCL OUT STAGING
set -uo pipefail
umask 007
[ -f $5/summary.json ] && { echo "exists"; exit 0; }
/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/evaluate.py \
  --predictions-dir $1 --dataset $2 --events $3 $4 --out $5 --jobs ${JOBS:-16} --staging-dir $6 > $5.log 2>&1
grep -v WARNING $5.log | grep "_strict"
[ -f $5/summary.json ]
