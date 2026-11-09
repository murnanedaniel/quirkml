#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -c 64 -t 00:30:00 -J qtrk_pu60_crop
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/crop_%j.out
/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/crop_audit.py /pscratch/sd/d/danieltm/quirk_pu60/crop_audit.json
