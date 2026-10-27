#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00 -J qtrk_fakes_b2
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/fakes/logs/b2_%x_%j.out
# B2/B3 for one model: features + strict labels on val and test, quality score and quirk tagger (quality.py).
# usage: sbatch --job-name=qtrk_b2_<model> b2.sbatch <model>
umask 007
export OMP_NUM_THREADS=64 POLARS_MAX_THREADS=128
EPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/fakes
srun -n1 -c256 $EPY quality.py $1 /pscratch/sd/d/danieltm/quirk_tracking/fakes/b2
