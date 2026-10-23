#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/logs/eval_t1_%x_%j.out
# usage: sbatch -J qtrk_eval_t1 --dependency=singleton --export=ALL,TAG=t1_qq10,GNN=gnn,SC=scored_sel,TH_MIN=..,TH_ADD=..,TH_STITCH=..,OPTAG=..,MH="3 5 8" eval_t1.sbatch
# T1 arm: ccwalk_stitch (0.7) tracks from the selected checkpoint (scored_sel) on test 17000-19999, ACTS per min-hits cut.
G=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
EPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
C=$O/cache/m200_L774_mu0
M=$O/main/$TAG
SC=${SC:-scored_sel}
DATA=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1
cd $G
$PY score_dists.py --cache $C --scored "$M/${GNN:-gnn}/$SC/events16000-16999.parquet" --events 16000 16999 > $M/${GNN:-gnn}/score_dists_val.json
for mh in ${MH:-3 5 8 10 12 15}; do
  n=op${OPTAG:-default}_mh$mh
  $PY build_tracks.py --cache $C --scored "$M/${GNN:-gnn}/$SC/*.parquet" --events 17000 19999 --out $M/test_${GNN:-gnn}/pred_$n \
    --run-id ${TAG}_$n --workers 64 --method ccwalk_stitch --th-min ${TH_MIN:-0.1} --th-add ${TH_ADD:-0.6} \
    --th-stitch ${TH_STITCH:-0.7} --min-hits $mh
  echo "$n quick $($PY quick_eval.py --cache $C --pred $M/test_${GNN:-gnn}/pred_$n/predictions.parquet --events 17000 19999)"
  $EPY /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/evaluate.py --predictions-dir $M/test_${GNN:-gnn}/pred_$n \
    --dataset $DATA --events 17000 20000 --out $M/test_${GNN:-gnn}/eval_$n --jobs 60 > $O/logs/${TAG}_${GNN:-gnn}_eval_$n.log 2>&1
  echo "$n"; grep -v WARNING $O/logs/${TAG}_${GNN:-gnn}_eval_$n.log | grep "very_loose\|tight \|proto"
done
