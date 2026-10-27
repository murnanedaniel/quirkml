#!/bin/bash
# ACTS (evaluate.py) scoring of one grid set's predictions at ot 0.1 and 0.2: normal truth (per-quirk, technical_sm,
# fake) and pair truth (coupled very_loose / tight), plus the Python coupled metric. Usage: grid_eval.sh <ds> [jobs]
set -uo pipefail
ds=$1; JOBS=${2:-30}
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml
DATA=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data
P=$DATA/grid_predictions/mf_pairdir_long/$ds
EV=$DATA/grid_eval/mf_pairdir_long/$ds
EVPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
STG=/pscratch/sd/d/danieltm/quirk_stageF/staging
umask 007
mkdir -p $EV
cd $Q
for ot in 0.1 0.2; do
  [ -f $P/ot$ot/predictions.parquet ] || { echo "[$ds] no predictions ot$ot"; exit 1; }
  for tr in truth pair; do
    out=$EV/${tr}_ot$ot; [ -f $out/summary.json ] && continue
    root=$DATA/testruns/$ds/parquet_r1; [ $tr = pair ] && root=$DATA/pair_truth/$ds
    $EVPY tracking/evaluate.py --predictions-dir $P/ot$ot --dataset $root --events 0 2000 --out $out --jobs $JOBS \
      --staging-dir $STG/$tr/$ds > $EV/${tr}_ot$ot.log 2>&1 || echo "[$ds] evaluate $tr ot$ot FAILED"
  done
  [ -f $EV/python_ot$ot.json ] || $EVPY tracking/maskformer/scripts/coupled_eval.py $P/ot$ot $DATA/testruns/$ds/parquet_r1 0 2000 $EV/python_ot$ot.json > /dev/null
done
echo "[$ds] $(date -Is) eval done"
