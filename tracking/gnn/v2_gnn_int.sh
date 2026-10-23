#!/bin/bash
# GNN A/B run on an interactive allocation (resumable): usage v2_gnn_int.sh <jobid> <none|node|node_edge>
J=$1; MODE=$2
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
C=$O/cache/m200_L774_mu0
M=$O/main/v2
FG="$M/filter/graphs_cut0.005/*.parquet"
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
L=$O/logs/v2_gnn_${MODE}_int.log
echo "$(date +%T) start job $J" >> $L
if [ ! -f $M/gnn_$MODE/cut_scan.json ]; then
  n=$(ls $O/logs/v2_gnn_${MODE}_part*.log 2>/dev/null | wc -l)
  ./run.sh $J v2_gnn_${MODE}_part$n train_gnn.py train --cache $C --graphs "$FG" --out $M/gnn_$MODE --name v2_gnn_$MODE \
    --cluster $MODE --train 0 15999 --val 16000 16199 --n-val 200 --epochs 16 --batch 4 --quirk-weight 10 --warmup 1000 || exit 1
fi
echo "$(date +%T) train done" >> $L
./run.sh $J v2_gnn_${MODE}_infer train_gnn.py infer --cache $C --graphs "$FG" --out $M/gnn_$MODE --ranges 16000-16999 17000-19999 || exit 1
echo "$(date +%T) infer done" >> $L
