#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 8 -t 00:30:00 -J qtrk_F_evalr
#SBATCH -o /pscratch/sd/d/danieltm/quirk_stageF/logs/evalr_%j.out
# Score the given grid sets: set i goes to node i % 8; each node runs 1 set at a time with 60 evaluate.py workers (2 at a time used up to 500 GB and was OOM-killed).
# srun -c 256: without it each step got 1 CPU (cause of the 30x slowdown of the earlier debug eval/prep jobs).
G=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/grid
L=/pscratch/sd/d/danieltm/quirk_stageF/logs/evalr_list_$SLURM_JOB_ID.txt
printf '%s\n' "$@" > $L
for k in 0 1 2 3 4 5 6 7; do
  srun -N1 -n1 -r $k --exact -c 256 bash -c "awk 'NR % 8 == $k' $L | xargs -P 1 -I{} /usr/bin/time -f '{} %e s %M kB' bash $G/grid_eval.sh {} 60" &
done
wait
