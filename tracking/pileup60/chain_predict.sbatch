#!/bin/bash
#SBATCH -A m5386 -C gpu -q shared -N 1 --gpus 1 -c 32 -t 00:30:00 -J qtrk_pu60_chainpred
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/chainpred_%j.out
# 1M splitter (d256 l6, as trained at mu 0) on the gated tracks; split tracks replace their sources in the pair-GNN predictions
R=$1; CC=$2; TAGS=$3; GATE=$4; O=$5
srun /pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/split/splitter2.py predict \
  --data $O/ccnodir_test.pkl --out /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/models/splitter/a1s_1m_nodir_aug_d256l6 --base $R/test/pred_cc${CC}_mh4/predictions.parquet \
  --pred-out $O/pred_split --decode knn_refine --gate-file $TAGS --gate-cut $GATE --keep-neither
