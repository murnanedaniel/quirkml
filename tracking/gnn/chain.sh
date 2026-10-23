#!/bin/bash
# Full GNN chain on one GPU allocation: embedding -> graphs -> filter -> filtered graphs -> GNN -> tracks -> eval.
# usage: chain.sh <jobid> <tag>     (main set m200_L774_mu0; train 0-15999 / val 16000-16999 / test 17000-19999)
set -e
JOB=$1; TAG=$2
G=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
C=$O/cache/m200_L774_mu0
M=$O/main/$TAG
DATA=/pscratch/sd/d/danieltm/quirk_pilots/r1/m200_L774_mu0
cd $G
stamp() { echo "$(date +%T) $*" >> $O/logs/chain_$TAG.log; }
stamp start
[ -z "$SKIP_EMB" ] && ./run.sh $JOB ${TAG}_emb train_embedding.py train --cache $C --out $M/emb --name ${TAG}_emb --train 0 15999 \
  --val 16000 16999 --n-val 200 --epochs ${EMB_EPOCHS:-12} --warmup 500 --scan 0.5 0.6 0.7 0.8 1.0
stamp emb done
R=$($PY -c "
import json; s=json.load(open('$M/emb/radius_scan.json'))
ok=[d for d in s if d['eff_quirk']>=0.97 and d['edges_per_event']<=400000]
print((ok[0] if ok else s[-1])['radius'])")
R=${RADIUS:-$R}
stamp radius $R
./run.sh $JOB ${TAG}_graph train_embedding.py graph --cache $C --out $M/emb --radius $R --k-max 512 \
  --ranges 16000-16199 17000-17199 0-3999
stamp graph done
GR="$M/emb/graphs_r$($PY -c "print(f'{$R:g}')")/*.parquet"
./run.sh $JOB ${TAG}_filt train_filter.py train --cache $C --graphs "$GR" --out $M/filter --name ${TAG}_filter \
  --train 0 3999 --val 16000 16199 --n-val 100 --epochs ${FILT_EPOCHS:-3} --batch 2 --pos-weight 3 --quirk-weight 10 \
  --cuts 0.02 0.05 0.1
stamp filter done
CUT=$($PY -c "
import json; s=json.load(open('$M/filter/cut_scan.json'))
ok=[d for d in s if d['eff_quirk']>=0.95 and d['edges_per_event']<=60000]
print((ok[-1] if ok else s[0])['cut'])")
stamp cut $CUT
./run.sh $JOB ${TAG}_filt_apply train_filter.py apply --cache $C --graphs "$GR" --out $M/filter --cut $CUT \
  --ranges 16000-16199 17000-17199 0-3999
stamp apply done
FG="$M/filter/graphs_cut$($PY -c "print(f'{$CUT:g}')")/*.parquet"
./run.sh $JOB ${TAG}_gnn train_gnn.py train --cache $C --graphs "$FG" --out $M/gnn --name ${TAG}_gnn --train 0 3999 \
  --val 16000 16199 --n-val 200 --epochs ${GNN_EPOCHS:-12} --batch 4 --quirk-weight 10
stamp gnn done
./run.sh $JOB ${TAG}_gnn_infer train_gnn.py infer --cache $C --graphs "$FG" --out $M/gnn --ranges 17000-17199 16000-16199
stamp infer done
for METH in ccwalk cc; do
  ./run.sh $JOB ${TAG}_tb_$METH build_tracks.py --cache $C --scored "$M/gnn/scored/*.parquet" --events 17000 17199 \
    --method $METH --out $M/pred_$METH --run-id ${TAG}_$METH --workers 32
  env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /') srun --jobid=$JOB --overlap -n 1 -c 32 \
    /global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python \
    /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/evaluate.py --predictions-dir $M/pred_$METH \
    --dataset $DATA --events 17000 17200 --out $M/eval_$METH --jobs 32 > $O/logs/${TAG}_eval_$METH.log 2>&1
  stamp eval $METH done
done
stamp finished
