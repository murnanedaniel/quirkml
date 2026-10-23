#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00 -J qtrk_eval_v2
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/logs/eval_v2_%x_%j.out
# usage: sbatch -J qtrk_eval_v2_<mode> --export=ALL,MODE=<mode> eval_v2.sbatch
# Track building + ACTS eval of the v2 GNN on test 17000-19999 (and val 16000-16999 for the quick check).
G=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
EPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
C=$O/cache/m200_L774_mu0
M=$O/main/v2
DATA=/pscratch/sd/d/danieltm/quirk_pilots/r1/m200_L774_mu0
cd $G
run() {  # name, args...
  n=$1; shift
  $PY build_tracks.py --cache $C --scored "$M/gnn_$MODE/scored/*.parquet" --events 17000 19999 --out $M/test_$MODE/pred_$n --run-id v2_${MODE}_$n --workers 64 "$@"
  echo "$n quick $($PY quick_eval.py --cache $C --pred $M/test_$MODE/pred_$n/predictions.parquet --events 17000 19999)"
  $EPY /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/evaluate.py --predictions-dir $M/test_$MODE/pred_$n \
    --dataset $DATA --events 17000 20000 --out $M/test_$MODE/eval_$n --jobs 60 > $O/logs/v2_${MODE}_eval_$n.log 2>&1
  echo "$n"; grep -v WARNING $O/logs/v2_${MODE}_eval_$n.log | grep "very_loose\|tight \|proto"
}
$PY score_dists.py --cache $C --scored "$M/gnn_$MODE/scored/events16000-16999.parquet" --events 16000 16999 > $M/gnn_$MODE/score_dists_val.json
run ccwalk --method ccwalk
run ccwalk_mh5 --method ccwalk --min-hits 5
for mh in 3 4 5 6; do run stitch0.7_mh$mh --method ccwalk_stitch --th-stitch 0.7 --min-hits $mh; done
# same 200 events as the v1 numbers, for a like-for-like v1/v2 comparison
for n in ccwalk stitch0.7_mh5; do
  $EPY /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/evaluate.py --predictions-dir $M/test_$MODE/pred_$n \
    --dataset $DATA --events 17000 17200 --out $M/test_$MODE/eval200_$n --jobs 32 > $O/logs/v2_${MODE}_eval200_$n.log 2>&1
  echo "$n (17000-17199)"; grep -v WARNING $O/logs/v2_${MODE}_eval200_$n.log | grep "very_loose\|tight "
done
