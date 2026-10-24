#!/bin/bash
#SBATCH -A m4958
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH -J qtrk_mf_mkpair
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/maskformer/logs/slurm-%x-%j.out
# Usage: sbatch sbatch_make_pair_truth.sh FIRST STOP_EXCL
umask 007
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml
/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python tracking/make_pair_truth.py \
  --dataset /global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1 --events $1 $2 \
  --out /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/pair_truth/m200_L774_mu0
