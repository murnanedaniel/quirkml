#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00 -J qtrk_grideval
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/grid/logs/eval_%j.out
# GNN grid (Stage F), CPU step: per test set with GNN scores, >= 4 hits:
#  per-quirk: graph v3 + GNN g2_q50_dir_s0, walk + stitch 0.7 -> ACTS on the normal truth
#  coupled:   pair-arm graph + pair GNN e2_pair_dir_s0, components at 0.5 -> ACTS on the pair-truth copy
# Stops after 25 min (debug wall); skips finished sets.
D=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data
OUT=/pscratch/sd/d/danieltm/quirk_tracking/gnn/grid
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
EPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
EV=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/evaluate.py
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
T0=$(date +%s)
for s in $(ls $OUT | grep _mu); do
  R=$OUT/$s
  [ -f $R/gpu_done ] || continue
  [ -f $R/eval_done ] && continue
  [ $(( $(date +%s) - T0 )) -gt 1500 ] && break
  C=$OUT/cache/${s}_dir
  $PY build_tracks.py --cache $C --scored "$R/scored_v3_g2s0/*.parquet" --events 0 1999 --out $R/pred_v3_walk_mh4 --workers 64 \
    --method ccwalk_stitch --th-stitch 0.7 --min-hits 4 --run-id ${s}_v3_walk > /dev/null 2>&1
  $EPY $EV --predictions-dir $R/pred_v3_walk_mh4 --dataset $D/testruns/$s/parquet_r1 --events 0 2000 --out $R/eval_v3_walk_mh4 \
    --jobs 60 --staging-dir $OUT/staging > $R/eval_v3_walk_mh4.log 2>&1
  $PY build_tracks.py --cache $C --scored "$R/scored_pair_s0/*.parquet" --events 0 1999 --out $R/pred_pair_cc05_mh4 --workers 64 \
    --method cc --cc-cut 0.5 --min-hits 4 --run-id ${s}_pair_cc > /dev/null 2>&1
  $EPY $EV --predictions-dir $R/pred_pair_cc05_mh4 --dataset $D/pair_truth/$s --events 0 2000 --out $R/evalpair_pair_cc05_mh4 \
    --jobs 60 --staging-dir $OUT/staging_pair > $R/evalpair_pair_cc05_mh4.log 2>&1
  [ -f $R/eval_v3_walk_mh4/summary.json ] && [ -f $R/evalpair_pair_cc05_mh4/summary.json ] && touch $R/eval_done && echo "eval done $s"
done
