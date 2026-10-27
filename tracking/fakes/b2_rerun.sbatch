#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00 -J qtrk_fakes_b2r
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/fakes/logs/b2rerun_%j.out
# quality.py (with quality_qw / quality_or_tag) for models whose track tables are cached.
umask 007
export OMP_NUM_THREADS=64 POLARS_MAX_THREADS=128
EPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/fakes
for m in "$@"; do srun -n1 -c256 $EPY quality.py $m /pscratch/sd/d/danieltm/quirk_tracking/fakes/b2; done
