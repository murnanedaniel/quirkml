#!/bin/bash
# Quirk-aware graph v3 (embedding g1_qw10, r=0.075; filter g1f_qw10r075_q50 at cut 0.001) -> perfect-score ceiling ->
# GNN (v2 recipe + direction features, seed S) -> inference. Resumable across windows.
# usage: g2_arm.sh <jobid> <tag> <seed>
J=$1; TAG=$2; SEED=$3
export NO_OVERLAP=1
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
C=$O/cache/m200_L774_mu0_dir
F=$O/main/g1f_qw10r075_q50/filter
FGD=$F/graphs_cut0.001
M=$O/main/$TAG
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn; mkdir -p $M
L=$O/logs/${TAG}_driver.log
stamp() { echo "$(date '+%F %T') $*" >> $L; }
stamp "start job $J seed $SEED"
if [ "$(ls $FGD 2>/dev/null | wc -l)" -lt 20 ]; then
  ./run.sh $J g1f_q50_apply train_filter.py apply --cache $C --emb-dir $O/main/g1_qw10/emb --radius 0.075 --k-max 512 \
    --out $F --cut 0.001 --ranges 0-15999 16000-16999 17000-19999 || { stamp "apply failed"; exit 1; }
  stamp "apply done"
fi
if [ ! -f $O/main/g1f_qw10r075_q50/perfect/done ]; then
  env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /') srun --jobid=$J --overlap -n 1 -c 8 \
    /pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python ../separability/perfect_scores.py --graphs "$FGD/events1[7-9]*.parquet" \
    --cache $O/cache/m200_L774_mu0 --events 17000 19999 --out $O/main/g1f_qw10r075_q50/perfect > $O/logs/g1f_q50_perfect.log 2>&1 \
    && touch $O/main/g1f_qw10r075_q50/perfect/done
  stamp "perfect scores written"
fi
if [ ! -f $M/gnn/cut_scan.json ]; then
  n=$(ls $O/logs/${TAG}_gnn_part*.log 2>/dev/null | wc -l)
  ./run.sh $J ${TAG}_gnn_part$n train_gnn.py train --direction --seed $SEED --cluster node --cache $C --graphs "$FGD/*.parquet" \
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
