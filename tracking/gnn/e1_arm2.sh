#!/bin/bash
# Stage E A/B arm: v2 recipe (all tracks, quirk-edge weight 10, cluster shape on nodes, 16k events, v1 embedding r=0.07),
# filter + GNN retrained, with (DIR=1) or without (DIR=0) the cluster direction features. Resumable across windows.
# usage: [CACHE=<cache dir>] e1_arm2.sh <jobid> <tag> <DIR 0|1> <seed>
J=$1; TAG=$2; DIR=$3; SEED=$4
export NO_OVERLAP=1
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
C=${CACHE:-$O/cache/m200_L774_mu0_dir}
M=$O/main/$TAG
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn; mkdir -p $M
L=$O/logs/${TAG}_driver.log
stamp() { echo "$(date '+%F %T') $*" >> $L; }
DF=""; [ "$DIR" = "1" ] && DF="--direction"
stamp "start job $J dir $DIR seed $SEED"
if [ ! -f $M/filter/cut_scan.json ]; then
  ./run.sh $J ${TAG}_filt train_filter.py train $DF --seed $SEED --cache $C --emb-dir $O/main/v1/emb --radius 0.07 \
    --out $M/filter --name ${TAG}_filter --train 0 15999 --val 16000 16199 --n-val 100 --epochs 6 --batch 2 \
    --pos-weight 3 --quirk-weight 10 --warmup 500 --cuts 0.005 0.01 0.02 || { stamp "filter failed"; exit 1; }
  stamp "filter done"
fi
FGD=$M/filter/graphs_cut0.005
if [ "$(ls $FGD 2>/dev/null | wc -l)" -lt 20 ]; then
  ./run.sh $J ${TAG}_filt_apply train_filter.py apply --cache $C --emb-dir $O/main/v1/emb --radius 0.07 --out $M/filter \
    --cut 0.005 --ranges 0-15999 16000-16999 17000-19999 || { stamp "apply failed"; exit 1; }
  stamp "apply done"
fi
if [ ! -f $M/gnn/cut_scan.json ]; then
  n=$(ls $O/logs/${TAG}_gnn_part*.log 2>/dev/null | wc -l)
  ./run.sh $J ${TAG}_gnn_part$n train_gnn.py train $DF --seed $SEED --cluster node --cache $C --graphs "$FGD/*.parquet" \
    --out $M/gnn --name ${TAG}_gnn --train 0 15999 --val 16000 16199 --n-val 200 --epochs 16 --batch 4 \
    --quirk-weight 10 --warmup 1000 || { stamp "gnn stopped (resume next window)"; exit 1; }
  stamp "gnn done"
fi
if [ "$(ls $M/gnn/scored 2>/dev/null | wc -l)" -lt 4 ]; then
  ./run.sh $J ${TAG}_gnn_infer train_gnn.py infer --cache $C --graphs "$FGD/*.parquet" --out $M/gnn --ckpt best.pt \
    --ranges 16000-16999 17000-19999 || { stamp "infer failed"; exit 1; }
  stamp "infer done"
fi
stamp finished
