#!/bin/bash
#SBATCH -A m5386 -C gpu -q shared -n 1 -c 32 --gpus=1 -t 04:30:00 -J qtrk_scal_graphs
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/scaling/logs/graphs_%j.out
# Graph construction (inference only, 16k-trained models) on training events 20000-153999, 1000-event blocks,
# blocks 20-53 first, one GPU, v3 then pair: graph v3 (embedding g1_qw10 r 0.075, filter g1f_qw10r075_q50 cut 0.001), then the pair
# arm's graph (embedding v1 r 0.07, filter e2_pair_dir_s0 cut 0.005, pair cache). Finished blocks are skipped.
umask 007
S=/pscratch/sd/d/danieltm/quirk_tracking/scaling
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
todo() { for b in $(seq ${B0:-20} ${B1:-153}); do [ -f $1/events$((b*1000))-$((b*1000+999)).parquet ] || echo "$((b*1000))-$((b*1000+999))"; done; }
run_v3() {
  srun -n1 -c 32 --gpus=1 $PY -u train_filter.py apply --cache $S/cache/dir --emb-dir $O/main/g1_qw10/emb \
    --radius 0.075 --k-max 512 --out $O/main/g1f_qw10r075_q50/filter --cut 0.001 --graph-dir $S/graphs/v3 \
    --ranges $(todo $S/graphs/v3) > $S/logs/graphs_v3_$SLURM_JOB_ID.log 2>&1
  rc=$?; echo "v3 exit $rc"; [ $rc -eq 0 ] && [ -z "$(B0=20 B1=153 todo $S/graphs/v3)" ] && touch $S/graphs/v3/.complete
}
run_pair() {
  srun -n1 -c 32 --gpus=1 $PY -u train_filter.py apply --cache $S/cache/dir_pair --emb-dir $O/main/v1/emb \
    --radius 0.07 --k-max 512 --out $O/main/e2_pair_dir_s0/filter --cut 0.005 --graph-dir $S/graphs/pair \
    --ranges $(todo $S/graphs/pair) > $S/logs/graphs_pair_$SLURM_JOB_ID.log 2>&1
  rc=$?; echo "pair exit $rc"; [ $rc -eq 0 ] && [ -z "$(B0=20 B1=153 todo $S/graphs/pair)" ] && touch $S/graphs/pair/.complete
}
date
B0=20 B1=53 run_v3; B0=20 B1=53 run_pair; run_v3; run_pair
date
ls $S/graphs/v3 | wc -l; ls $S/graphs/pair | wc -l
