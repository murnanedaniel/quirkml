#!/bin/bash
#SBATCH -A m4958 -C cpu -q regular -N 1 -c 256 -t 01:30:00 -J qtrk_pu60_lam_score
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/lam_score_%j.out
# Other string scales at mu 60, test 0-1999 of set DS: pair-GNN tracks (components 0.5), Python strict (pair + normal
# truth), the L774 tagger + quality cut (lam_apply60.py), gated splitter-chain inputs. usage: sbatch lam_score.sbatch DS
set -uo pipefail
DS=$1; CC=0.5
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
T=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data
S=/pscratch/sd/d/danieltm/quirk_pu60; W=$S/lambda/$DS; P=$W/test/pred_cc${CC}_mh4
cd $T/gnn
[ -s $P/predictions.parquet ] || $PY build_tracks.py --cache $S/cache/${DS}_dir_pair --scored "$W/scored/events*.parquet" --events 0 1999 \
  --out $P --run-id qw40_${DS}_cc$CC --workers 120 --method cc --cc-cut $CC --min-hits 4 || exit 1
$PY $T/mu60_eval/py_score.py $P $Q/pair_truth_mu60/$DS 0 2000 $W/test/pypair_cc${CC}_mh4.json &
$PY $T/mu60_eval/py_score.py $P $Q/parquet_r1_mu60/$DS 0 2000 $W/test/py_cc${CC}_mh4.json &
$PY $T/pileup60/lam_apply60.py $DS qw40_cc$CC $W/test/pred_cc${CC}_mh4 $W/scored $W/apply \
  $Q/models/tagger/b3_pair60_50k_qw40_cc0.5 $Q/models/quality_pu60/pair60_50k_qw40_cc0.5 > $W/apply.log 2>&1 &
wait
[ -s $W/apply/qw40_cc$CC/b3_scores_test.parquet ] || { echo "apply FAILED"; exit 1; }
GATE=$($PY -c "import json;print(json.load(open('$Q/models/tagger/b3_pair60_50k_qw40_cc0.5/tagger.json'))['pair_good']['cut_from_valB'])")
SPLIT_CACHE_DIR=$S/cache/${DS}_dir bash $T/pileup60/chain_split.sbatch $W $CC $W/apply/qw40_cc$CC/b3_scores_test.parquet $GATE $W/chain
ls -la $W/test/*.json $W/chain/ccnodir_test.pkl
