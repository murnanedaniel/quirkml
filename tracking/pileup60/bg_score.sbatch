#!/bin/bash
#SBATCH -A m4958 -C cpu -q regular -N 1 -c 256 -t 01:30:00 -J qtrk_pu60_bg_score
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/bg_score_%j.out
# background tag rate on quirk-free mu 60 events 60000-61999 (m200_L774_mu60_noquirk): pair_50k_qw40_s0 tracks
# (components 0.5), L774 tagger + quality cut (bg_tag60.py)
set -uo pipefail
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
T=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data
S=/pscratch/sd/d/danieltm/quirk_pu60; DS=m200_L774_mu60_noquirk; W=$S/lambda/$DS; P=$W/test/pred_cc0.5_mh4
cd $T/gnn
[ -s $P/predictions.parquet ] || $PY build_tracks.py --cache $S/cache/${DS}_dir_pair --scored "$W/scored/events*.parquet" --events 60000 61999 \
  --out $P --run-id qw40_${DS}_cc0.5 --workers 120 --method cc --cc-cut 0.5 --min-hits 4 || exit 1
$PY $T/pileup60/bg_tag60.py $DS 60000 62000 qw40_cc0.5 $P $W/scored $W/bg $Q/models/tagger/b3_pair60_50k_qw40_cc0.5 \
  $Q/models/quality_pu60/pair60_50k_qw40_cc0.5 || exit 1
mkdir -p $Q/results/pu60/background; cp $W/bg/qw40_cc0.5/bg_tag.json $Q/results/pu60/background/
