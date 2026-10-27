#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/logs/perfect_eval_%x_%j.out
# usage: sbatch -J qtrk_perfect --export=ALL,P=<perfect dir> perfect_eval.sbatch
# Perfect-score ceiling on a candidate graph: stitch 0.7 tracks from truth scores, ACTS on test 17000-19999.
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
EPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
DATA=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
cat $P/graph_ceiling.json
for mh in ${MH:-3 4 5}; do
  n=stitch0.7_mh$mh
  $PY build_tracks.py --cache $O/cache/m200_L774_mu0 --scored "$P/scored/*.parquet" --events 17000 19999 --out $P/pred_$n \
    --workers 64 --method ccwalk_stitch --th-stitch 0.7 --min-hits $mh
  $EPY /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/evaluate.py --predictions-dir $P/pred_$n \
    --dataset $DATA --events 17000 20000 --out $P/eval_$n --jobs 60 > $P/eval_$n.log 2>&1
  echo "$n"; grep -v WARNING $P/eval_$n.log | grep "very_loose\|tight \|proto"
done
