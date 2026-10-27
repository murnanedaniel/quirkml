#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00 -J qtrk_F_eval1
#SBATCH -o /pscratch/sd/d/danieltm/quirk_stageF/logs/eval1_%j.out
/usr/bin/time -v bash /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/grid/grid_eval.sh m100_L10000_mu0 60
