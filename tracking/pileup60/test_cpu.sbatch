#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -c 256 -t 00:30:00 -J qtrk_pu60_test
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/test_%x_%j.out
# Track building + Python strict scoring on test 0-1999 for one trained mu 60 GNN.
# usage: sbatch test_cpu.sbatch <run dir> <pair|quirk>
set -uo pipefail
RUN=$1; W=$2; CC=${3:-0.5}
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
T=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking
C=/pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data
R1=$Q/parquet_r1_mu60/m200_L774_mu60; PT=$Q/pair_truth_mu60/m200_L774_mu60
cd $T/gnn
if [ $W = pair ]; then
  P=$RUN/test/pred_cc${CC}_mh4
  [ -s $P/predictions.parquet ] || $PY build_tracks.py --cache ${C}_pair --scored "$RUN/scored/events[01]*.parquet" --events 0 1999 \
    --out $P --run-id $(basename $RUN)_cc$CC --workers 120 --method cc --cc-cut $CC --min-hits 4 || exit 1
  $PY $T/mu60_eval/py_score.py $P $PT 0 2000 $RUN/test/pypair_cc${CC}_mh4.json &
  $PY $T/mu60_eval/py_score.py $P $R1 0 2000 $RUN/test/py_cc${CC}_mh4.json &
  wait
  J="$RUN/test/pypair_cc${CC}_mh4.json $RUN/test/py_cc${CC}_mh4.json"
else
  P=$RUN/test/pred_stitch0.7_mh4
  [ -s $P/predictions.parquet ] || $PY build_tracks.py --cache $C --scored "$RUN/scored/events[01]*.parquet" --events 0 1999 \
    --out $P --run-id $(basename $RUN)_stitch0.7 --workers 120 --method ccwalk_stitch --th-stitch 0.7 --min-hits 4 || exit 1
  $PY $T/mu60_eval/py_score.py $P $R1 0 2000 $RUN/test/py_stitch0.7_mh4.json
  J=$RUN/test/py_stitch0.7_mh4.json
fi
for f in $J; do $PY -c "
import json; r=json.load(open('$f'))
for p in ['very_loose_strict','tight_strict']:
    x=r[p]; print('$f'.split('/')[-1],p,'quirk %.3f'%x['quirk']['eff'],'event',x['quirk_event']['eff'],'tech_sm %.3f'%x['technical_sm']['eff'],'fake %.3f'%x['tracks']['fake_rate'])"; done
