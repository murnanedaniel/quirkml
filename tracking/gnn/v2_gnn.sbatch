#!/bin/bash
#SBATCH -A m4958 -C gpu -q shared --gpus 1 -c 32 -t 08:00:00
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/logs/v2_gnn_%x_%j.out
# usage: sbatch -J qtrk_v2_gnn_<mode> --export=ALL,MODE=<none|node|node_edge> v2_gnn.sbatch
set -e
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
C=$O/cache/m200_L774_mu0
M=$O/main/v2
FG="$M/filter/graphs_cut0.005/*.parquet"
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
srun -n 1 $PY -u train_gnn.py train --cache $C --graphs "$FG" --out $M/gnn_$MODE --name v2_gnn_$MODE --cluster $MODE \
  --train 0 15999 --val 16000 16199 --n-val 200 --epochs 16 --batch 4 --quirk-weight 10 --warmup 1000 > $O/logs/v2_gnn_$MODE.log 2>&1
srun -n 1 $PY -u train_gnn.py infer --cache $C --graphs "$FG" --out $M/gnn_$MODE --ranges 16000-16999 17000-19999 > $O/logs/v2_gnn_${MODE}_infer.log 2>&1
