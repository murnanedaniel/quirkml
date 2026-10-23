#!/bin/bash
# ROADMAP T1 arm: quirk-as-signal (LRT) recipe, full chain, resumable across allocation windows.
# usage: t1_arm.sh <jobid> <tag> <w_qq>     e.g. t1_arm.sh 123 t1_qq10 10
# Stages are skipped when their outputs exist; the GNN resumes from state.pt.
J=$1; TAG=$2; WQQ=$3
export NO_OVERLAP=1
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
C=$O/cache/m200_L774_mu0
M=$O/main/$TAG
G=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
cd $G; mkdir -p $M
L=$O/logs/${TAG}_driver.log
stamp() { echo "$(date '+%F %T') $*" >> $L; }
stamp "start job $J w_qq $WQQ"
if [ ! -f $M/emb/radius_scan.json ]; then
  ./run.sh $J ${TAG}_emb train_embedding.py train --recipe lrt --w-qq $WQQ --cache $C --out $M/emb --name ${TAG}_emb \
    --train 0 15999 --val 16000 16999 --n-val 200 --epochs 12 --warmup 500 --scan 0.3 0.4 0.5 0.6 0.7 0.8 1.0 || { stamp "emb failed"; exit 1; }
  stamp "emb done"
fi
R=$($PY -c "
import json; s=json.load(open('$M/emb/radius_scan.json'))
ok=[d for d in s if d['eff_quirk']>=0.97 and d['edges_per_event']<=400000]
print((ok[0] if ok else max(s, key=lambda d: d['eff_quirk'] if d['edges_per_event']<=400000 else -1))['radius'])")
stamp "radius $R"
if [ ! -f $M/filter/cut_scan.json ]; then
  ./run.sh $J ${TAG}_filt train_filter.py train --recipe lrt --w-qq $WQQ --cache $C --emb-dir $M/emb --radius $R \
    --out $M/filter --name ${TAG}_filter --train 0 15999 --val 16000 16199 --n-val 100 --epochs 6 --batch 2 \
    --pos-weight 1 --warmup 500 --cuts 0.005 0.01 0.02 0.05 \
    --scan 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 || { stamp "filter failed"; exit 1; }
  stamp "filter done"
fi
CUT=$($PY -c "
import json; s=json.load(open('$M/filter/cut_scan.json'))
ok=[d for d in s if d['eff_quirk']>=0.95 and d['edges_per_event']<=150000]
print((ok[-1] if ok else s[0])['cut'])")
CUT=$($PY -c "print(f'{$CUT:g}')")
stamp "cut $CUT"
FGD=$M/filter/graphs_cut$CUT
if [ "$(ls $FGD 2>/dev/null | wc -l)" -lt 20 ]; then
  ./run.sh $J ${TAG}_filt_apply train_filter.py apply --cache $C --emb-dir $M/emb --radius $R --out $M/filter --cut $CUT \
    --ranges 0-15999 16000-16999 17000-19999 || { stamp "apply failed"; exit 1; }
  stamp "apply done"
fi
if [ ! -f $M/gnn/cut_scan.json ]; then
  n=$(ls $O/logs/${TAG}_gnn_part*.log 2>/dev/null | wc -l)
  ./run.sh $J ${TAG}_gnn_part$n train_gnn.py train --recipe lrt --w-qq $WQQ --track-val --n-track-val 200 \
    --fake-target 0.10 --cluster node --cache $C --graphs "$FGD/*.parquet" --out $M/gnn --name ${TAG}_gnn \
    --train 0 15999 --val 16000 16199 --n-val 200 --epochs 16 --batch 4 --warmup 1000 || { stamp "gnn stopped (resume next window)"; exit 1; }
  stamp "gnn done"
fi
if [ "$(ls $M/gnn/scored_best_trk 2>/dev/null | wc -l)" -lt 4 ]; then
  ./run.sh $J ${TAG}_gnn_infer train_gnn.py infer --cache $C --graphs "$FGD/*.parquet" --out $M/gnn --ckpt best_trk.pt \
    --scored-dir scored_best_trk --ranges 16000-16999 17000-19999 || { stamp "infer failed"; exit 1; }
  stamp "infer done"
fi
stamp finished
