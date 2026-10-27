#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00 -J qtrk_e1eval
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/logs/p3_eval_%j.out
# Pair-target GNN on graph v3: builder cut chosen on val (components: highest coupled eff with fake <= walk's and
# technical within 1 point of walk's), then test once: components at that cut and walk + stitch, >= 4 hits; ACTS on
# the pair truth (coupled) and the normal truth (fake, technical_sm). usage: sbatch --export=ALL,TAG=p3_pairv3_s0 p3_eval.sbatch
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
EPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
DATA=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1
PAIR=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/pair_truth/m200_L774_mu0
M=$O/main/$TAG
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
$PY tune_builders.py --cache $O/cache/m200_L774_mu0_dir_pair --scored "$M/gnn/scored/events16000-16999.parquet" --events 16000 16499 > $M/val_builders.jsonl
cat $M/val_builders.jsonl
CUT=$($PY -c "
import json; r=[json.loads(l) for l in open('$M/val_builders.jsonl')]
w=[x for x in r if x['method']=='ccwalk_stitch'][0]
ok=[x for x in r if x['method']=='cc' and x['fake']<=w['fake'] and x['tech']>=w['tech']-0.01]
print(max(ok,key=lambda x:x['quirk_eff'])['cut'] if ok else 0.5)")
echo "chosen components cut $CUT"
ev() { $EPY ../evaluate.py --predictions-dir $1 --dataset $2 --events 17000 20000 --out $3 --jobs 60 $4 > $3.log 2>&1; grep -v WARNING $3.log | grep "very_loose\|tight "; }
for spec in "cc:$CUT" "ccwalk_stitch:0.5"; do
  IFS=: read METH C2 <<< "$spec"; n=${METH}${C2}_mh4
  $PY build_tracks.py --cache $O/cache/m200_L774_mu0 --scored "$M/gnn/scored/events1[7-9]*.parquet" --events 17000 19999 \
    --out $M/test/pred_$n --workers 64 --method $METH --cc-cut $C2 --th-stitch 0.7 --min-hits 4 > /dev/null 2>&1
  echo "== $n PAIR TRUTH"; ev $M/test/pred_$n $PAIR $M/test/evalpair_$n "--staging-dir $O/staging_pair"
  echo "== $n NORMAL TRUTH"; ev $M/test/pred_$n $DATA $M/test/eval_$n
done
