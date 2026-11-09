#!/bin/bash
#SBATCH -A m5386 -C gpu -q shared -N 1 --gpus 1 -c 32 -t 01:00:00 -J qtrk_pu60_infer
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/infer_%j.out
# GNN inference (final.pt) on test + val blocks 0-2. usage: sbatch infer_run.sbatch RUN_DIR CACHE
R=$1; C=$2
[ -f $R/final.pt ] || { echo "no final.pt"; exit 1; }
srun /pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/infer60.py \
  --run $R --ckpt final.pt --cache $C --graphs /pscratch/sd/d/danieltm/quirk_pu60/graphs/v3_mu0f_cut0.1 --blocks 0 1 2 --out $R/scored >> $R/infer.log 2>&1
echo "exit $?"
