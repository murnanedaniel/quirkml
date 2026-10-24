#!/bin/bash
# Graph construction for quirks, step 1: metric learning with quirk chain pairs up-weighted among the positives.
# usage: g1_emb.sh <jobid> <tag> <quirk_pos_weight> [wait_for_driver_log]
J=$1; TAG=$2; W=$3; WAITLOG=$4
export NO_OVERLAP=1
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
if [ -n "$WAITLOG" ]; then until grep -q "finished\|failed" $WAITLOG; do sleep 30; done; fi
echo "$(date '+%F %T') start $TAG w=$W job $J" >> $O/logs/${TAG}_driver.log
./run.sh $J ${TAG}_emb train_embedding.py train --quirk-pos-weight $W --cache $O/cache/m200_L774_mu0 --out $O/main/$TAG/emb \
  --name ${TAG}_emb --train 0 15999 --val 16000 16999 --n-val 200 --epochs 20 --warmup 500 \
  --scan 0.5 0.6 0.7 0.8 0.9 1.0 1.2 --k-max 512
echo "$(date '+%F %T') emb exit $?" >> $O/logs/${TAG}_driver.log
