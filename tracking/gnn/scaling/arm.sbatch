#!/bin/bash
#SBATCH -A m5386 -C gpu -q shared -n 1 -c 32 --gpus=1 -t 10:00:00
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/scaling/logs/arm_%x_%j.out
# Batch fallback for one scaling arm (1 GPU): waits for its graph shards, then arm_run.sh. Resubmit to resume.
# env: TAG KIND(v3|pair) RANGES STEPS SEED WALL_S NEED (last training block)
umask 007
T_START=$(date +%s)
S=/pscratch/sd/d/danieltm/quirk_tracking/scaling
GR=$S/graphs/$KIND
until [ -f $GR/.complete ] || [ -f $GR/events$(( (NEED+1)*1000 ))-$(( (NEED+1)*1000+999 )).parquet ]; do
  [ $(( $(date +%s) - T_START )) -gt 5400 ] && { echo "graphs not ready"; exit 4; }; sleep 30; done
M=$S/runs/$TAG
if [ -f $M/RUNNING ] && [ "$(cat $M/RUNNING)" != "$SLURM_JOB_ID" ] && squeue -h -j $(cat $M/RUNNING) 2>/dev/null | grep -q .; then
  echo "$TAG already running in job $(cat $M/RUNNING)"; exit 0; fi
grep -qs finished $M/driver.log && { echo "$TAG already finished"; exit 0; }
DEADLINE=$(( T_START + ${WALL_S:-36000} - 1500 )) bash /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/scaling/arm_run.sh
