#!/bin/bash
#SBATCH -A m4958 -C gpu -q shared --gpus 1 -c 32 -t 05:00:00 -J qtrk_v2_filter
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/logs/v2_filter_%j.out
# v2 filter on all 16k training events (graphs on the fly from the v1 embedding, r=0.07), then filtered graphs (cut 0.005)
# for train/val/test. These graphs are the shared input of the GNN cluster-feature A/B.
set -e
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
C=$O/cache/m200_L774_mu0
M=$O/main/v2
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
mkdir -p $M
if [ ! -f $M/filter/cut_scan.json ]; then
srun -n 1 $PY -u train_filter.py train --cache $C --emb-dir $O/main/v1/emb --radius 0.07 --out $M/filter --name v2_filter \
  --train 0 15999 --val 16000 16199 --n-val 100 --epochs 6 --batch 2 --pos-weight 3 --quirk-weight 10 --warmup 500 \
  --cuts 0.005 0.01 0.02 > $O/logs/v2_filt.log 2>&1
fi
srun -n 1 $PY -u train_filter.py apply --cache $C --emb-dir $O/main/v1/emb --radius 0.07 --out $M/filter --cut 0.005 \
  --ranges 0-15999 16000-16999 17000-19999 > $O/logs/v2_filt_apply.log 2>&1
