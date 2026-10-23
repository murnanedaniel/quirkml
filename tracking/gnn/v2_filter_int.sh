#!/bin/bash
# v2 filter on an interactive allocation: usage v2_filter_int.sh <jobid>
J=$1
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
C=$O/cache/m200_L774_mu0
M=$O/main/v2
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
mkdir -p $M
echo "$(date +%T) start" >> $O/logs/v2_filter_int.log
./run.sh $J v2_filt train_filter.py train --cache $C --emb-dir $O/main/v1/emb --radius 0.07 --out $M/filter --name v2_filter \
  --train 0 15999 --val 16000 16199 --n-val 100 --epochs 6 --batch 2 --pos-weight 3 --quirk-weight 10 --warmup 500 \
  --cuts 0.005 0.01 0.02 || exit 1
echo "$(date +%T) filter done" >> $O/logs/v2_filter_int.log
./run.sh $J v2_filt_apply train_filter.py apply --cache $C --emb-dir $O/main/v1/emb --radius 0.07 --out $M/filter --cut 0.005 \
  --ranges 0-15999 16000-16999 17000-19999 || exit 1
echo "$(date +%T) apply done" >> $O/logs/v2_filter_int.log
