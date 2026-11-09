#!/bin/bash
#SBATCH -A m5386 -C gpu -q shared -N 1 --gpus 1 -c 32 -t 06:00:00 -J qtrk_pu60_graphs
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/graphs_batch_%j.out
# compact graph v3 (mu 0 filter, cut 0.1) for the blocks in $BLOCKS (skips blocks already written)
M=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/models/gnn
srun /pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/graph_apply60.py \
  --cache /pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir --emb $M/g1_qw10/emb --radius 0.075 --filter $M/g1f_qw10r075_q50/filter \
  --cut 0.1 --out /pscratch/sd/d/danieltm/quirk_pu60/graphs/v3_mu0f_cut0.1 --blocks $BLOCKS
