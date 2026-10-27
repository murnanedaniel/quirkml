#!/bin/bash
# Window chain: pair-target GNN on graph v3 (p3_pairv3_s0) on one GPU, the GNN grid GPU step (Stage F) on the other.
# At each new window: log GPU-hours, clear stale grid locks, submit the CPU grid prep (new test sets) and relaunch.
# Stops when p3 is finished and no prepared grid set is waiting. usage: chain_windows4.sh <current jobid>
J=$1
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
G=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking
L=$O/logs/chain_windows4.log
U="env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /' | tr '\n' ' ')"
grid_left() { for d in $O/grid/cache/*_dir; do s=$(basename $d _dir); [ -f $d/done ] && [ ! -f $O/grid/$s/gpu_done ] && return 0; done; return 1; }
while true; do
  while squeue -j $J -h 2>/dev/null | grep -q .; do sleep 60; done
  S=$(sacct -j $J -X -n --format=ElapsedRaw 2>/dev/null | head -1 | tr -d ' ')
  [ "$J" != "0" ] && [ -n "$S" ] && echo "$J gpu_shared_interactive(m5386) 2 $(sacct -j $J -X -n --format=Start | head -1 | tr -d ' ') $(sacct -j $J -X -n --format=End | head -1 | tr -d ' ') $(python3 -c "print(round(2*$S/3600,2))") pair-target GNN on graph v3 (p3_pairv3_s0) + GNN grid inference (Stage F)" >> $O/gpu_hours.txt
  cd $G/grid && $U sbatch -J qtrk_gridprep --dependency=singleton gnn_grid_prep.sbatch >> $L 2>&1
  if grep -q finished $O/logs/p3_pairv3_s0_driver.log && ! grid_left; then echo "$(date '+%F %T') done; stop" >> $L; exit 0; fi
  J=$($U salloc -A m5386 -C gpu -q shared_interactive --gpus 2 -c 64 -t 04:00:00 --no-shell -J qtrk_g1 2>&1 | grep -o "Granted job allocation [0-9]*" | grep -o "[0-9]*$")
  if [ -z "$J" ]; then echo "$(date '+%F %T') salloc failed; retry in 5 min" >> $L; sleep 300; J=0; continue; fi
  echo "$(date '+%F %T') new window $J" >> $L
  rmdir $O/grid/*/lock 2>/dev/null
  cd $G/gnn
  grep -q finished $O/logs/p3_pairv3_s0_driver.log || setsid nohup ./p3_arm.sh $J p3_pairv3_s0 0 < /dev/null >> $O/logs/p3_pairv3_s0_driver.out 2>&1 &
  sleep 15
  setsid nohup $G/grid/gnn_grid_gpu.sh $J < /dev/null >> $O/grid/logs/gpu_driver.out 2>&1 &
  if grep -q finished $O/logs/p3_pairv3_s0_driver.log; then sleep 15; setsid nohup $G/grid/gnn_grid_gpu.sh $J < /dev/null >> $O/grid/logs/gpu_driver.out 2>&1 & fi
done
