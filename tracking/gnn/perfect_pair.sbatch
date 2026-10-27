#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -t 00:30:00 -J qtrk_perfpair
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/gnn/logs/perfect_pair_%j.out
# Coupled ceiling: perfect pair-truth edge scores on graph v3 and on the pair arm's graph; stitch 0.7, >= 4 hits; ACTS on pair truth.
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
EPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
CP=$O/cache/m200_L774_mu0_dir_pair
PAIR=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/pair_truth/m200_L774_mu0
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
for spec in "v3:$O/main/g1f_qw10r075_q50/filter/graphs_cut0.001" "pairarm:$O/main/e2_pair_dir_s0/filter/graphs_cut0.005"; do
  nm=${spec%%:*}; G=${spec#*:}; P=$O/perfect_pair/$nm
  $PY ../separability/perfect_scores.py --graphs "$G/events1[7-9]*.parquet" --cache $CP --events 17000 19999 --out $P > $P.log 2>&1
  echo "== $nm"; cat $P/graph_ceiling.json
  $PY build_tracks.py --cache $CP --scored "$P/scored/*.parquet" --events 17000 19999 --out $P/pred_mh4 --workers 64 \
    --method ccwalk_stitch --th-stitch 0.7 --min-hits 4 > /dev/null 2>&1
  $EPY ../evaluate.py --predictions-dir $P/pred_mh4 --dataset $PAIR --events 17000 20000 --out $P/evalpair_mh4 --jobs 60 \
    --staging-dir $O/staging_pair > $P/evalpair_mh4.log 2>&1
  grep -v WARNING $P/evalpair_mh4.log | grep "very_loose\|tight \|proto"
done
