#!/bin/bash
#SBATCH -A m4958 -C cpu -q regular -N 1 -c 256 -t 01:30:00 -J qtrk_pu60_zscpu
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/zs_cpu_%j.out
# Step 2 zero-shot, CPU part (test 0-1999): strict ACTS of the per-quirk GNN v3 (normal truth) and the pair GNN
# components (pair truth: coupled; normal truth: fake, technical_sm); B3 tagger (mu 0 fit) on the pair tracks;
# splitter input tracks (cc mode, direction columns zeroed as in the 1M splitter's training). Resumable.
set -uo pipefail
DS=${DS:-m200_L774_mu60}
O=/pscratch/sd/d/danieltm/quirk_pu60/zs/$DS
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data
R1=$Q/parquet_r1_mu60/$DS; PT=$Q/pair_truth_mu60/$DS
T=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
EPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
ST=/pscratch/sd/d/danieltm/quirk_pu60/staging
C=/pscratch/sd/d/danieltm/quirk_pu60/cache/${DS}_dir
G=$T/gnn
[ -s $O/v3/test/pred_stitch0.7_mh4/predictions.parquet ] || $PY $G/build_tracks.py --cache $C --scored "$O/v3/scored/events*.parquet" --events 0 1999 \
  --out $O/v3/test/pred_stitch0.7_mh4 --run-id zs_v3_test --workers 120 --method ccwalk_stitch --th-stitch 0.7 --min-hits 4 > $O/logs/v3_build_test.log 2>&1 &
[ -s $O/pair/test/pred_cc0.5_mh4/predictions.parquet ] || $PY $G/build_tracks.py --cache ${C}_pair --scored "$O/pair/scored/events*.parquet" --events 0 1999 \
  --out $O/pair/test/pred_cc0.5_mh4 --run-id zs_pair_test --workers 120 --method cc --cc-cut 0.5 --min-hits 4 > $O/logs/pair_build_test.log 2>&1 &
wait
for f in $O/v3/test/pred_stitch0.7_mh4 $O/pair/test/pred_cc0.5_mh4; do [ -s $f/predictions.parquet ] || { echo "build FAILED $f"; exit 1; }; done
E=$T/pileup60/eval_one.sbatch
sub() { env -u SLURM_MEM_PER_CPU -u SLURM_JOB_ID -u SLURM_JOBID sbatch -J $1 $E $2 $3 0 2000 $4 $5; }
sub qtrk_pu60_ev_v3 $O/v3/test/pred_stitch0.7_mh4 $R1 $O/v3/test/eval_stitch0.7_mh4 $ST/normal
sub qtrk_pu60_ev_pairp $O/pair/test/pred_cc0.5_mh4 $PT $O/pair/test/evalpair_cc0.5_mh4 $ST/pair
sub qtrk_pu60_ev_pairn $O/pair/test/pred_cc0.5_mh4 $R1 $O/pair/test/eval_cc0.5_mh4 $ST/normal2
[ -s $O/pair/test/b3_scores.parquet ] || $PY $T/pileup60/tagger_b3.py apply $Q/models/tagger/b3_pair_gnn_s0_cc_mu0 $O/pair/test/pred_cc0.5_mh4 \
  "$O/pair/scored/events[01]*.parquet" $R1 0 2000 $O/pair/test/b3_scores.parquet > $O/logs/b3_test.log 2>&1 &
wait  # the gate needs the tagger scores
if [ ! -s $O/split/ccnodir_test.pkl ]; then
  mkdir -p $O/split
  SPLIT_NO_TIME=1 SPLIT_CACHE=/pscratch/sd/d/danieltm/quirk_pu60/cache/${DS}_dir $PY $T/split/split_data.py --mode cc --pred $O/pair/test/pred_cc0.5_mh4/predictions.parquet \
    --events 0 1999 --gate-file $O/pair/test/b3_scores.parquet --gate-cut 0.0018037541753598385 --out $O/split/cc_test.pkl > $O/logs/split_data_test.log 2>&1 && $PY -c "
import pickle; d=pickle.load(open('$O/split/cc_test.pkl','rb'))
for t in d: t['x']=t['x'].copy(); t['x'][:,28:38]=0
pickle.dump(d,open('$O/split/ccnodir_test.pkl','wb'),protocol=5); print('cc test tracks',len(d))" >> $O/logs/split_data_test.log 2>&1
fi
wait
tail -n 2 $O/logs/b3_test.log $O/logs/split_data_test.log
