#!/bin/bash
# Pair-target GNN on quirk graph v3: pair cache, every same-pair edge true (--pair-all-true), v2 recipe + direction
# features. No filter retraining (graph v3 as is). Resumable. usage: p3_arm.sh <jobid> <tag> <seed>
J=$1; TAG=$2; SEED=$3
export NO_OVERLAP=1
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
C=$O/cache/m200_L774_mu0_dir_pair
FGD=$O/main/g1f_qw10r075_q50/filter/graphs_cut0.001
M=$O/main/$TAG
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn; mkdir -p $M
L=$O/logs/${TAG}_driver.log
stamp() { echo "$(date '+%F %T') $*" >> $L; }
stamp "start job $J seed $SEED"
if [ ! -f $M/gnn/cut_scan.json ]; then
  n=$(ls $O/logs/${TAG}_gnn_part*.log 2>/dev/null | wc -l)
  ./run.sh $J ${TAG}_gnn_part$n train_gnn.py train --pair-all-true --direction --seed $SEED --cluster node --cache $C \
    --graphs "$FGD/*.parquet" --out $M/gnn --name ${TAG}_gnn --train 0 15999 --val 16000 16199 --n-val 200 --epochs 16 \
    --batch 4 --quirk-weight 10 --warmup 1000 || { stamp "gnn stopped (resume next window)"; exit 1; }
  stamp "gnn done"
fi
if [ "$(ls $M/gnn/scored 2>/dev/null | wc -l)" -lt 4 ]; then
  ./run.sh $J ${TAG}_gnn_infer train_gnn.py infer --cache $C --graphs "$FGD/*.parquet" --out $M/gnn --ckpt best.pt \
    --ranges 16000-16999 17000-19999 || { stamp "infer failed"; exit 1; }
  stamp "infer done"
fi
stamp finished
