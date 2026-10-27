#!/bin/bash
# Window chain for the g2 seed-1 (quirk graph v3 GNN) and pair seed-2 arms: when the current shared_interactive allocation ends,
# log its GPU-hours, request a new 2-GPU window (m5386) and relaunch both drivers (they skip finished stages and resume
# the GNN from state.pt). Stops when both drivers have logged "finished". usage: chain_windows.sh <current jobid>
J=$1
O=/pscratch/sd/d/danieltm/quirk_tracking/gnn
G=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn
L=$O/logs/chain_windows3.log
cd $G
done_both() { grep -q finished $O/logs/g2_q50_dir_s1_driver.log && grep -q finished $O/logs/p3_pairv3_s0_driver.log; }
while true; do
  while squeue -j $J -h 2>/dev/null | grep -q .; do sleep 60; done
  S=$(sacct -j $J -X -n --format=ElapsedRaw 2>/dev/null | head -1 | tr -d ' ')
  [ "$J" != "0" ] && [ -n "$S" ] && echo "$J gpu_shared_interactive(m5386) 2 $(sacct -j $J -X -n --format=Start | head -1 | tr -d ' ') $(sacct -j $J -X -n --format=End | head -1 | tr -d ' ') $(python3 -c "print(round(2*$S/3600,2))") graph v3 GNN seed 1 (g2_q50_dir_s1) + pair-target on graph v3 (p3_pairv3_s0)" >> $O/gpu_hours.txt
  if done_both; then echo "$(date '+%F %T') both finished; stop" >> $L; exit 0; fi
  J=$(env $(env | grep -o '^SLURM_[A-Z_]*' | sed 's/^/-u /') salloc -A m5386 -C gpu -q shared_interactive --gpus 2 -c 64 -t 04:00:00 --no-shell -J qtrk_g1 2>&1 | grep -o "Granted job allocation [0-9]*" | grep -o "[0-9]*$")
  if [ -z "$J" ]; then echo "$(date '+%F %T') salloc failed; retry in 5 min" >> $L; sleep 300; J=0; continue; fi
  echo "$(date '+%F %T') new window $J" >> $L
  grep -q finished $O/logs/g2_q50_dir_s1_driver.log || setsid nohup ./g2_arm.sh $J g2_q50_dir_s1 1 < /dev/null >> $O/logs/g2_q50_dir_s1_driver.out 2>&1 &
  sleep 15
  grep -q finished $O/logs/p3_pairv3_s0_driver.log || setsid nohup ./p3_arm.sh $J p3_pairv3_s0 0 < /dev/null >> $O/logs/p3_pairv3_s0_driver.out 2>&1 &
done
