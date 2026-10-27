#!/bin/bash
# GNN grid (Stage F), GPU step on one GPU of an allocation: for each prepared test set (no retraining):
#  1. graph v3 (embedding g1_qw10 r=0.075 + filter g1f_qw10r075_q50 cut 0.001) -> per-quirk GNN g2_q50_dir_s0 scores
#  2. pair-arm graph (v1 embedding r=0.07 + filter e2_pair_dir_s0 cut 0.005) -> pair GNN e2_pair_dir_s0 scores
# usage: gnn_grid_gpu.sh <jobid>   (loops until no prepared set is left to do)
J=$1
export NO_OVERLAP=1
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
OUT=$O/grid
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
for s in $(ls $OUT/cache | grep "_dir$" | sed 's/_dir$//'); do
  C=$OUT/cache/${s}_dir; R=$OUT/$s; mkdir -p $R
  [ -f $C/done ] || continue
  [ -f $R/gpu_done ] && continue
  ( mkdir $R/lock 2>/dev/null ) || continue   # one GPU per set
  ./run.sh $J grid_${s}_v3graph train_filter.py apply --cache $C --emb-dir $O/main/g1_qw10/emb --radius 0.075 --k-max 512 \
    --out $O/main/g1f_qw10r075_q50/filter --cut 0.001 --graph-dir $R/graph_v3 --ranges 0-1999 &&
  ./run.sh $J grid_${s}_v3gnn train_gnn.py infer --cache $C --graphs "$R/graph_v3/*.parquet" --out $O/main/g2_q50_dir_s0/gnn \
    --ckpt best.pt --scored-dir $R/scored_v3_g2s0 --ranges 0-1999 &&
  ./run.sh $J grid_${s}_pairgraph train_filter.py apply --cache $C --emb-dir $O/main/v1/emb --radius 0.07 --k-max 512 \
    --out $O/main/e2_pair_dir_s0/filter --cut 0.005 --graph-dir $R/graph_pair --ranges 0-1999 &&
  ./run.sh $J grid_${s}_pairgnn train_gnn.py infer --cache $C --graphs "$R/graph_pair/*.parquet" --out $O/main/e2_pair_dir_s0/gnn \
    --ckpt best.pt --scored-dir $R/scored_pair_s0 --ranges 0-1999 &&
  touch $R/gpu_done && echo "$(date '+%F %T') $s gpu done" >> $OUT/logs/gpu.log
  rmdir $R/lock 2>/dev/null
done
