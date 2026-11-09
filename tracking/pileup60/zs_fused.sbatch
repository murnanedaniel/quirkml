#!/bin/bash
#SBATCH -A m5386 -C gpu&hbm80g -q shared -N 1 --gpus 1 -c 32 -t 04:00:00
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/zs_fused_%x_%j.out
# Zero-shot combined pass (mu 0 models and cuts) on mu 60 test events 0-1999. env WHICH=v3|pair
DS=${DS:-m200_L774_mu60}
M=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/models/gnn
C=/pscratch/sd/d/danieltm/quirk_pu60/cache/${DS}_dir
O=/pscratch/sd/d/danieltm/quirk_pu60/zs/$DS
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
S=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/zs_fused.py
if [ "$WHICH" = v3 ]; then
  srun $PY $S --cache $C --emb $M/g1_qw10/emb --radius 0.075 --filter $M/g1f_qw10r075_q50/filter --cut 0.001 \
    --gnn $M/g2_q50_dir_s0/gnn --ranges 0-999 1000-1999 --out $O/v3/scored
else
  srun $PY $S --cache ${C}_pair --emb $M/v1/emb --radius 0.07 --filter $M/e2_pair_dir_s0/filter --cut 0.005 \
    --gnn $M/e2_pair_dir_s0/gnn --ranges 0-999 1000-1999 --out $O/pair/scored
fi
