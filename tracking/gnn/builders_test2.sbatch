#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00 -J qtrk_btest2
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/logs/builders_test_%j.out
# Test 17000-19999 with the builders chosen on val (tune_builders.py): pair GNN s0 cc 0.5 and hybrid 0.5 (pair + normal
# truth); v3 GNN s0 cc 0.5 and hybrid 0.7 (normal truth). >= 4 hits.
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
EPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
DATA=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1
PAIR=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/pair_truth/m200_L774_mu0
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
ev() { $EPY ../evaluate.py --predictions-dir $1 --dataset $2 --events 17000 20000 --out $3 --jobs 60 $4 > $3.log 2>&1; grep -v WARNING $3.log | grep "very_loose\|tight "; }
for spec in $SPECS; do
  IFS=: read T METH CUT <<< "$spec"; M=$O/main/$T; n=${METH}${CUT}_mh4
  $PY build_tracks.py --cache $O/cache/m200_L774_mu0 --scored "$M/gnn/scored/events1[7-9]*.parquet" --events 17000 19999 \
    --out $M/test/pred_$n --workers 64 --method $METH --cc-cut $CUT --min-hits 4 > /dev/null 2>&1
  if [ "${T#e2_pair}" != "$T" ]; then echo "== $T $n PAIR TRUTH"; ev $M/test/pred_$n $PAIR $M/test/evalpair_$n "--staging-dir $O/staging_pair"; fi
  echo "== $T $n NORMAL TRUTH"; ev $M/test/pred_$n $DATA $M/test/eval_$n
done
