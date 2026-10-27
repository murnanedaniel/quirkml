#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00 -J qtrk_fakes_b1
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/fakes/logs/b1_%j.out
# B1: strict fake decomposition of 4 test prediction sets (decompose.py), one CPU node.
umask 007
EPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/fakes
srun -n1 -c256 $EPY decompose.py /pscratch/sd/d/danieltm/quirk_tracking/fakes/b1
