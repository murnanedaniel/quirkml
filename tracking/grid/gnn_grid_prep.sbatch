#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00 -J qtrk_gridprep
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/grid/logs/prep_%j.out
# GNN grid (Stage F), CPU step: per test set with direction features available, build the GNN cache (make_cache.py)
# and add the direction features (make_dir_cache.py). Skips finished sets. Reads quirk_tracking_data/{testruns,direction}.
D=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data
OUT=/pscratch/sd/d/danieltm/quirk_tracking/gnn/grid
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
for s in $(ls $D/direction | grep _mu); do
  [ "$s" = "m200_L774_mu0" ] && continue
  [ -f $OUT/cache/${s}_dir/done ] && continue
  [ $(ls $D/direction/$s/*.parquet 2>/dev/null | wc -l) -ge 2 ] || continue
  mkdir -p $OUT/cache
  $PY make_cache.py --root $D/testruns/$s/parquet_r1 --out $OUT/cache/$s --workers 2 > $OUT/logs/prep_$s.log 2>&1 &&
  $PY make_dir_cache.py --cache $OUT/cache/$s --direction $D/direction/$s --out $OUT/cache/${s}_dir --workers 2 >> $OUT/logs/prep_$s.log 2>&1 &&
  touch $OUT/cache/${s}_dir/done && echo "prepared $s" &
  while [ $(jobs -r | wc -l) -ge 32 ]; do sleep 2; done
done
wait
