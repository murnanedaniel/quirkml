#!/bin/bash
# Graph construction for quirks, step 2: filter on a quirk-weighted embedding graph, quirk-edge weight QW, direction features.
# usage: g1_filter.sh <jobid> <tag> <emb_tag> <radius> <QW>
J=$1; TAG=$2; EMB=$3; R=$4; QW=$5
export NO_OVERLAP=1
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
echo "$(date '+%F %T') start $TAG emb $EMB r $R qw $QW job $J" >> $O/logs/${TAG}_driver.log
./run.sh $J ${TAG}_filt train_filter.py train --direction --seed 0 --cache $O/cache/m200_L774_mu0_dir --emb-dir $O/main/$EMB/emb \
  --radius $R --k-max 512 --out $O/main/$TAG/filter --name ${TAG}_filter --train 0 15999 --val 16000 16199 --n-val 100 \
  --epochs 6 --batch 2 --pos-weight 3 --quirk-weight $QW --warmup 500 --cuts 0.001 0.005 0.02 \
  --scan 0.0001 0.0003 0.001 0.002 0.005 0.01 0.02
echo "$(date '+%F %T') filter exit $?" >> $O/logs/${TAG}_driver.log
