#!/bin/bash
#SBATCH -A m4958
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J qtrk_mf_pairtruth
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/maskformer/logs/slurm-%x-%j.out
# Build the pair-truth copy of the test range (if missing) and score pair-model and GNN v2 predictions against it.
set -uo pipefail
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml
SRC=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1
PT=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/pair_truth/m200_L774_mu0
OUT=/pscratch/sd/d/danieltm/quirk_tracking/maskformer
EVPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
cd $Q
umask 007
[ -f $PT/pair_truth_manifest.json ] || $EVPY tracking/make_pair_truth.py --dataset $SRC --events 17000 20000 --out $PT
for spec in "mf_pair5e-5_ot0.1:$OUT/predictions/t2d_pair_lr5e-5_test_ot0.1" "mf_pair5e-5_ot0.2:$OUT/predictions/t2d_pair_lr5e-5_test_ot0.2" \
            "mf_t2fix_ot0.1:$OUT/predictions/t2fix_lr5e-5_test_ot0.1" \
            "gnn_v2_stitch0.7_mh4:/pscratch/sd/d/danieltm/quirk_tracking/gnn/main/v2/test_node/pred_stitch0.7_mh4"; do
  N=${spec%%:*}; P=${spec#*:}
  [ -f $P/predictions.parquet ] || { echo "[$N] missing $P"; continue; }
  $EVPY tracking/evaluate.py --predictions-dir $P --dataset $PT --events 17000 20000 --out $OUT/eval/pairtruth_$N \
     --jobs 60 --staging-dir $OUT/staging_pairtruth > $OUT/logs/eval_pairtruth_$N.log 2>&1
  echo "[$N] rc=$?"; grep -e '^very_loose' -e '^tight' $OUT/logs/eval_pairtruth_$N.log
  $EVPY $Q/tracking/maskformer/scripts/coupled_eval.py $P $SRC 17000 20000 $OUT/eval/pairtruth_${N}_python.json
done
