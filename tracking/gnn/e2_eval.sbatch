#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/logs/e2_eval_%x_%j.out
# usage: sbatch -J qtrk_e1eval --dependency=singleton --export=ALL,TAG=e2_pair_dir_s0 e2_eval.sbatch
# Coupled-task GNN: stitch 0.7 tracks at min hits 3-6. Val (16000-16999): quick_eval on the pair cache (choice of min hits).
# Test 17000-19999: ACTS against the pair truth copy (quirk = pair object) and against the normal truth (technical, fake).
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
EPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
CP=$O/cache/m200_L774_mu0_dir_pair
M=$O/main/$TAG
DATA=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1
PAIR=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/pair_truth/m200_L774_mu0
EV=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/evaluate.py
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
for mh in ${MH:-3 4 5 6}; do
  n=stitch0.7_mh$mh
  $PY build_tracks.py --cache $CP --scored "$M/gnn/scored/events16000-16999.parquet" --events 16000 16999 --out $M/val/pred_$n \
    --workers 64 --method ccwalk_stitch --th-stitch 0.7 --min-hits $mh > /dev/null
  echo "$n val quick (pair cache) $($PY quick_eval.py --cache $CP --pred $M/val/pred_$n/predictions.parquet --events 16000 16999)"
done
for mh in ${MH:-3 4 5 6}; do
  n=stitch0.7_mh$mh
  $PY build_tracks.py --cache $CP --scored "$M/gnn/scored/*.parquet" --events 17000 19999 --out $M/test/pred_$n \
    --run-id ${TAG}_$n --workers 64 --method ccwalk_stitch --th-stitch 0.7 --min-hits $mh
  $EPY $EV --predictions-dir $M/test/pred_$n --dataset $PAIR --events 17000 20000 --out $M/test/evalpair_$n --jobs 60 \
    --staging-dir $O/staging_pair > $O/logs/${TAG}_evalpair_$n.log 2>&1
  echo "$n PAIR TRUTH"; grep -v WARNING $O/logs/${TAG}_evalpair_$n.log | grep "very_loose\|tight \|proto"
  $EPY $EV --predictions-dir $M/test/pred_$n --dataset $DATA --events 17000 20000 --out $M/test/eval_$n --jobs 60 > $O/logs/${TAG}_eval_$n.log 2>&1
  echo "$n NORMAL TRUTH"; grep -v WARNING $O/logs/${TAG}_eval_$n.log | grep "very_loose\|tight \|proto"
done
