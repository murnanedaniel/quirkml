#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -c 256 -t 00:30:00 -J qtrk_pu60_cache
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/cache_%j.out
# dir + pair caches of mu 60 blocks (env BLOCKS, default 0 1 2) from the shared R1 copy
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data
DS=${DS:-m200_L774_mu60}
date
$PY /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/make_cache60.py --r1 $Q/parquet_r1_mu60/$DS \
  --direction $Q/direction/$DS --out-dir /pscratch/sd/d/danieltm/quirk_pu60/cache/${DS}_dir \
  --out-pair /pscratch/sd/d/danieltm/quirk_pu60/cache/${DS}_dir_pair --blocks ${BLOCKS:-0 1 2} --workers ${W:-3}
date
