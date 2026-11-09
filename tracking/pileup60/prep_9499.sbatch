#!/bin/bash
#SBATCH -A m4958 -C cpu -q regular -N 1 -c 256 -t 01:30:00 -J qtrk_pu60_dir9499
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/prep_%j.out
bash /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/mu60_prep.sh ${DS:-m200_L774_mu60} ${B0:-0} ${B1:-2} ${P:-6}
