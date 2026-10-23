#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/logs/t1_eval_full_%x_%j.out
# usage: sbatch -J qtrk_dbg --dependency=singleton --export=ALL,TAG=t1_qq10 t1_eval_full.sbatch
# Operating point on val (tune_op.py, events 16200-16999), then ACTS on test 17000-19999 per min-hits cut.
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
M=$O/main/$TAG
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
if [ ! -f $M/gnn/tune_op.json ]; then
  $PY tune_op.py --cache $O/cache/m200_L774_mu0 --scored "$M/gnn/scored_sel/events16000-16999.parquet" --events 16200 16999 \
    --out $M/gnn/tune_op.json --workers 24 | tail -1
fi
read TH_MIN TH_ADD TH_STITCH <<< $($PY -c "
import json; b=json.load(open('$M/gnn/tune_op.json'))['best']; print(b['th_min'], b['th_add'], b['th_stitch'])")
echo "operating point th_min $TH_MIN th_add $TH_ADD th_stitch $TH_STITCH"
export TAG TH_MIN TH_ADD TH_STITCH OPTAG=tuned MH="${MH:-3 5 8 10 12}" SC=scored_sel GNN=gnn
bash eval_t1.sbatch
