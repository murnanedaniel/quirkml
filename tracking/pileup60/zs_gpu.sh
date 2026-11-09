#!/bin/bash
# Step 2 zero-shot, GPU part, mu 60 events 0-2999 (test 0-1999, val 2000-2999), mu 0 models unchanged:
#  graph v3 (g1_qw10 emb r 0.075 k 512 + g1f_qw10r075_q50 filter cut 0.001) -> per-quirk GNN v3 (g2_q50_dir_s0) ->
#  walk + stitch 0.7, >= 4 hits;  pair graph (v1 emb r 0.07 + e2_pair_dir_s0 filter cut 0.005) -> pair GNN
#  (e2_pair_dir_s0) -> components 0.5, >= 4 hits. Resumable; stops at the first failure.
# usage: zs_gpu.sh  (inside a 1-GPU allocation, or srun'd)
set -uo pipefail
DS=${DS:-m200_L774_mu60}
O=/pscratch/sd/d/danieltm/quirk_pu60/zs/$DS
M=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/models/gnn
C=/pscratch/sd/d/danieltm/quirk_pu60/cache/${DS}_dir; CP=${C}_pair
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
G=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
RANGES="0-999 1000-1999 2000-2999"
mkdir -p $O/logs; cd $G
L=$O/logs/zs_gpu.log; st() { echo "$(date '+%F %T') $*" | tee -a $L; }
n() { ls $1 2>/dev/null | wc -l; }
if [ $(n $O/v3/graphs) -lt 3 ]; then
  $PY train_filter.py apply --cache $C --emb-dir $M/g1_qw10/emb --radius 0.075 --k-max 512 --out $M/g1f_qw10r075_q50/filter \
    --cut 0.001 --graph-dir $O/v3/graphs --ranges $RANGES > $O/logs/v3_graphs.log 2>&1 || { st "v3 graphs FAILED"; exit 1; }
  st "v3 graphs done"
fi
if [ $(n $O/pair/graphs) -lt 3 ]; then
  $PY train_filter.py apply --cache $CP --emb-dir $M/v1/emb --radius 0.07 --k-max 512 --out $M/e2_pair_dir_s0/filter \
    --cut 0.005 --graph-dir $O/pair/graphs --ranges $RANGES > $O/logs/pair_graphs.log 2>&1 || { st "pair graphs FAILED"; exit 1; }
  st "pair graphs done"
fi
if [ $(n $O/v3/scored) -lt 3 ]; then
  $PY train_gnn.py infer --cache $C --graphs "$O/v3/graphs/*.parquet" --out $M/g2_q50_dir_s0/gnn --ckpt best.pt \
    --ranges $RANGES --scored-dir $O/v3/scored > $O/logs/v3_infer.log 2>&1 || { st "v3 infer FAILED"; exit 1; }
  st "v3 infer done"
fi
if [ $(n $O/pair/scored) -lt 3 ]; then
  $PY train_gnn.py infer --cache $CP --graphs "$O/pair/graphs/*.parquet" --out $M/e2_pair_dir_s0/gnn --ckpt best.pt \
    --ranges $RANGES --scored-dir $O/pair/scored > $O/logs/pair_infer.log 2>&1 || { st "pair infer FAILED"; exit 1; }
  st "pair infer done"
fi
for sp in "test 0 1999" "val 2000 2999"; do
  set -- $sp
  [ -s $O/v3/$1/pred_stitch0.7_mh4/predictions.parquet ] || $PY build_tracks.py --cache $C --scored "$O/v3/scored/*.parquet" --events $2 $3 \
    --out $O/v3/$1/pred_stitch0.7_mh4 --run-id zs_v3_$1 --workers 30 --method ccwalk_stitch --th-stitch 0.7 --min-hits 4 \
    > $O/logs/v3_build_$1.log 2>&1 || { st "v3 build $1 FAILED"; exit 1; }
  [ -s $O/pair/$1/pred_cc0.5_mh4/predictions.parquet ] || $PY build_tracks.py --cache $CP --scored "$O/pair/scored/*.parquet" --events $2 $3 \
    --out $O/pair/$1/pred_cc0.5_mh4 --run-id zs_pair_$1 --workers 30 --method cc --cc-cut 0.5 --min-hits 4 \
    > $O/logs/pair_build_$1.log 2>&1 || { st "pair build $1 FAILED"; exit 1; }
done
st "tracks built"
