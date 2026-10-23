#!/bin/bash
#SBATCH -A m4958 -C gpu -q shared --gpus 1 -c 32 -t 11:00:00 -J qtrk_v2
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/logs/chain_v2_%j.out
# v2: filter + GNN retrained on all 16k training events (embedding from v1; graphs built on the fly).
set -e
G=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
C=$O/cache/m200_L774_mu0
M=$O/main/v2
cd $G
mkdir -p $M
stamp() { echo "$(date +%T) $*" >> $O/logs/chain_v2.log; }
L=$O/logs
stamp start job $SLURM_JOB_ID
if [ ! -f $M/filter/cut_scan.json ]; then
srun -n 1 $PY -u train_filter.py train --cache $C --emb-dir $O/main/v1/emb --radius 0.07 --out $M/filter --name v2_filter \
  --train 0 15999 --val 16000 16199 --n-val 100 --epochs 6 --batch 2 --pos-weight 3 --quirk-weight 10 --warmup 500 \
  --cuts 0.005 0.01 0.02 > $L/v2_filt.log 2>&1
fi
stamp filter done
CUT=0.005
if [ ! -d $M/filter/graphs_cut$CUT ] || [ $(ls $M/filter/graphs_cut$CUT | wc -l) -lt 20 ]; then
srun -n 1 $PY -u train_filter.py apply --cache $C --emb-dir $O/main/v1/emb --radius 0.07 --out $M/filter --cut $CUT \
  --ranges 0-15999 16000-16999 17000-19999 > $L/v2_filt_apply.log 2>&1
fi
stamp apply done
FG="$M/filter/graphs_cut$CUT/*.parquet"
srun -n 1 $PY -u train_gnn.py train --cache $C --graphs "$FG" --out $M/gnn --name v2_gnn --train 0 15999 \
  --val 16000 16199 --n-val 200 --epochs 16 --batch 4 --quirk-weight 10 --warmup 1000 > $L/v2_gnn.log 2>&1
stamp gnn done
srun -n 1 $PY -u train_gnn.py infer --cache $C --graphs "$FG" --out $M/gnn --ranges 16000-16999 17000-19999 > $L/v2_gnn_infer.log 2>&1
stamp infer done
