#!/bin/bash
#SBATCH -A m4958 -C cpu -q regular -N 1 -t 01:30:00 -J qtrk_scal_cache
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/scaling/logs/cache_%j.out
# GNN caches for training events 20000-153999: base (make_cache.py on the R1 conversion), + direction features
# (make_dir_cache.py), pair-truth copy (make_pair_cache.py); then combined symlink dirs with the existing 0-19999 caches.
umask 007
set -o pipefail
S=/pscratch/sd/d/danieltm/quirk_tracking/scaling
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
G=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
cd $G
date
srun -n1 -c 256 $PY make_cache.py --root $S/r1_extra --out $S/cache/base_new --workers 48 | tail -3 || { echo "make_cache failed"; exit 1; }
date
srun -n1 -c 256 $PY make_dir_cache.py --cache $S/cache/base_new --direction $S/direction/features --out $S/cache/dir_new --workers 48 | tail -3 || { echo "dir cache failed"; exit 1; }
date
srun -n1 -c 256 $PY make_pair_cache.py --cache $S/cache/dir_new --r1 $S/r1_extra --out $S/cache/dir_pair_new --workers 48 | tail -3 || { echo "pair cache failed"; exit 1; }
date
mkdir -p $S/cache/dir $S/cache/dir_pair
for f in $O/cache/m200_L774_mu0_dir/events*.pkl $S/cache/dir_new/events*.pkl; do ln -sf $f $S/cache/dir/; done
for f in $O/cache/m200_L774_mu0_dir_pair/events*.pkl $S/cache/dir_pair_new/events*.pkl; do ln -sf $f $S/cache/dir_pair/; done
ls $S/cache/dir | wc -l; ls $S/cache/dir_pair | wc -l
