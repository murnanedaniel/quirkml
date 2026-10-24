#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 2 -t 00:30:00 -J qtrk_sep
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/separability/logs/run_%j.out
# Separability ceilings, m200_L774_mu0 test 17000-19999.
#   node 0: perfect edge scores on the GNN v2 filter graph -> ccwalk / ccwalk+stitch 0.7 mh4 -> ACTS (normal truth)
#   node 1: waits for the predictions, ACTS on the pair truth copy; timing study (timing.py) meanwhile
# usage: env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /') sbatch run.sbatch
set -u
umask 007
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking
PY=/global/homes/d/danieltm/.conda/envs/collider-env/bin/python
EPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
O=/pscratch/sd/d/danieltm/quirk_tracking/separability
C=/pscratch/sd/d/danieltm/quirk_tracking/gnn/cache/m200_L774_mu0
G=/pscratch/sd/d/danieltm/quirk_tracking/gnn/main/v2/filter/graphs_cut0.005
DATA=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet_r1
PAIR=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/pair_truth/m200_L774_mu0
V2PRED=/pscratch/sd/d/danieltm/quirk_tracking/gnn/main/v2/test_node/pred_stitch0.7_mh4/predictions.parquet
P=$O/perfect
cd $Q/gnn
node0() {
  $PY $Q/separability/perfect_scores.py --graphs "$G/events1[789]000-*.parquet" --cache $C --events 17000 19999 --out $P
  for spec in "ccwalk|--method ccwalk" "stitch0.7_mh4|--method ccwalk_stitch --th-stitch 0.7 --min-hits 4"; do
    n=${spec%%|*}; args=${spec#*|}
    $PY build_tracks.py --cache $C --scored "$P/scored/*.parquet" --events 17000 19999 --out $P/pred_$n \
      --run-id perfect_$n --workers 64 $args
    echo "$n quick $($PY quick_eval.py --cache $C --pred $P/pred_$n/predictions.parquet --events 17000 19999)"
    touch $P/pred_$n/DONE
  done
  for n in ccwalk stitch0.7_mh4; do
    $EPY $Q/evaluate.py --predictions-dir $P/pred_$n --dataset $DATA --events 17000 20000 --out $P/eval_$n --jobs 60 \
      --staging-dir $O/staging > $O/logs/perfect_eval_$n.log 2>&1
    echo "$n exit $?"; grep -v WARNING $O/logs/perfect_eval_$n.log | grep "very_loose\|tight \|proto"
  done
}
node1() {
  $PY $Q/separability/timing.py --pred $V2PRED --events 17000 19999 --out $O/timing --fig-dir $O > $O/logs/timing.log 2>&1
  echo "timing exit $?"
  for n in ccwalk stitch0.7_mh4; do
    while [ ! -f $P/pred_$n/DONE ]; do sleep 10; done
    $EPY $Q/evaluate.py --predictions-dir $P/pred_$n --dataset $PAIR --events 17000 20000 --out $P/evalpair_$n --jobs 60 \
      --staging-dir $O/staging_pair > $O/logs/perfect_evalpair_$n.log 2>&1
    echo "$n PAIR exit $?"; grep -v WARNING $O/logs/perfect_evalpair_$n.log | grep "very_loose\|tight \|proto"
  done
}
if [ "${1:-}" = node0 ]; then node0; exit; fi
if [ "${1:-}" = node1 ]; then node1; exit; fi
srun -N1 -n1 -c 256 -r 0 --overlap bash $Q/separability/run.sbatch node0 > $O/logs/node0_$SLURM_JOB_ID.out 2>&1 &
srun -N1 -n1 -c 256 -r 1 --overlap bash $Q/separability/run.sbatch node1 > $O/logs/node1_$SLURM_JOB_ID.out 2>&1 &
wait
echo ALL DONE
