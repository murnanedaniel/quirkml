#!/bin/bash
#SBATCH -A m4958
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J qtrk_mf_reval
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/maskformer/logs/slurm-%x-%j.out
# Usage: sbatch sbatch_round_eval.sh PRED_NAME:FIRST:STOP_EXCL ...  (PRED_NAME under predictions/)
# For each: evaluate.py vs the normal R1 truth (skipped if done) and vs the pair truth, and coupled_eval.py.
# Summary lines go to logs/round_eval.txt.
set -uo pipefail
umask 007
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml
SRC=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1
PT=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/pair_truth/m200_L774_mu0
OUT=/pscratch/sd/d/danieltm/quirk_tracking/maskformer
EVPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
cd $Q
for spec in "$@"; do
  N=${spec%%:*}; rest=${spec#*:}; A=${rest%%:*}; B=${rest#*:}; P=$OUT/predictions/$N
  [ -f $P/predictions.parquet ] || { echo "[$N] missing predictions" | tee -a $OUT/logs/round_eval.txt; continue; }
  if [ ! -f $OUT/eval/$N/summary.json ]; then
    $EVPY tracking/evaluate.py --predictions-dir $P --dataset $SRC --events $A $B --out $OUT/eval/$N --jobs 60 \
      --staging-dir $OUT/staging_round > $OUT/logs/eval_$N.log 2>&1
  fi
  $EVPY tracking/evaluate.py --predictions-dir $P --dataset $PT --events $A $B --out $OUT/eval/pairtruth_$N --jobs 60 \
    --staging-dir $OUT/staging_round_pt > $OUT/logs/eval_pairtruth_$N.log 2>&1
  $EVPY $Q/tracking/maskformer/scripts/coupled_eval.py $P $SRC $A $B $OUT/eval/pairtruth_${N}_python.json > /dev/null
  {
    echo "[$N] truth:      $(grep '^very_loose' $OUT/logs/eval_$N.log)"
    echo "[$N] pair truth: $(grep -e '^very_loose' $OUT/logs/eval_pairtruth_$N.log) || $(grep -e '^tight' $OUT/logs/eval_pairtruth_$N.log)"
    echo "[$N] python:     $(cat $OUT/eval/pairtruth_${N}_python.json | tr -d '\n ')"
  } | tee -a $OUT/logs/round_eval.txt
done
