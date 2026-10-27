#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00 -J qtrk_a0
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/split/logs/a0_%j.out
set -o pipefail
/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python -u /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/split/a0_crossing.py --out /pscratch/sd/d/danieltm/quirk_tracking/split/a0
