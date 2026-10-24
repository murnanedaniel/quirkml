#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/logs/e1_eval_%x_%j.out
# usage: sbatch -J qtrk_e1eval --dependency=singleton --export=ALL,TAG=e1_dir_s0 e1_eval.sbatch
# Stage E arm on test 17000-19999: ccwalk + stitch 0.7 at min hits 3-6 (ACTS each), quick_eval pair metrics, dig.py.
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
EPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
C=$O/cache/m200_L774_mu0
M=$O/main/$TAG
DATA=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
$PY score_dists.py --cache $C --scored "$M/gnn/scored/events16000-16999.parquet" --events 16000 16999 > $M/gnn/score_dists_val.json
for mh in ${MH:-3 4 5 6}; do
  n=stitch0.7_mh$mh
  $PY build_tracks.py --cache $C --scored "$M/gnn/scored/*.parquet" --events 17000 19999 --out $M/test/pred_$n \
    --run-id ${TAG}_$n --workers 64 --method ccwalk_stitch --th-stitch 0.7 --min-hits $mh
  echo "$n quick $($PY quick_eval.py --cache $C --pred $M/test/pred_$n/predictions.parquet --events 17000 19999)"
  $EPY /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/evaluate.py --predictions-dir $M/test/pred_$n \
    --dataset $DATA --events 17000 20000 --out $M/test/eval_$n --jobs 60 > $O/logs/${TAG}_eval_$n.log 2>&1
  echo "$n"; grep -v WARNING $O/logs/${TAG}_eval_$n.log | grep "very_loose\|tight \|proto"
done
$PY dig.py --pred $M/test/pred_stitch0.7_mh4/predictions.parquet --out $M/dig --workers 6 > $M/dig.log 2>&1
echo "dig done: $M/dig/dig_summary.json"
