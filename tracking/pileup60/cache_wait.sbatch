#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -c 256 -t 00:30:00 -J qtrk_pu60_cachew
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/cachew_%j.out
# dir + pair caches for blocks B0..B1 as their direction features appear (at most 4 at a time)
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data
DS=${DS:-m200_L774_mu60}
for b in $(seq ${B0:-3} ${B1:-12}); do
  f=$Q/direction/$DS/quirks.$DS.v1.direction.events$((1000*b))-$((1000*b+999)).parquet
  until [ -f $f ]; do sleep 20; done
  while [ $(jobs -r | wc -l) -ge 4 ]; do sleep 5; done
  $PY /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/make_cache60.py --r1 $Q/parquet_r1_mu60/$DS --direction $Q/direction/$DS \
    --out-dir /pscratch/sd/d/danieltm/quirk_pu60/cache/${DS}_dir --out-pair /pscratch/sd/d/danieltm/quirk_pu60/cache/${DS}_dir_pair --blocks $b --workers 1 &
done
wait; date
