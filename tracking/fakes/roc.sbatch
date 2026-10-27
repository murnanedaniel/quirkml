#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 2 -t 00:30:00 -J qtrk_fakes_roc
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/fakes/logs/roc_%j.out
# coupled_roc.py for every model given, split over the job's nodes (one model at a time per node, explicit host).
umask 007
export OMP_NUM_THREADS=64 POLARS_MAX_THREADS=128
EPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
OUT=/pscratch/sd/d/danieltm/quirk_tracking/fakes/coupled
cd /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/fakes
HOSTS=($(scontrol show hostnames $SLURM_JOB_NODELIST)); N=${#HOSTS[@]}
MODELS=("$@")
for k in $(seq 0 $((N-1))); do
  ( for i in $(seq $k $N $((${#MODELS[@]}-1))); do m=${MODELS[$i]}
      [ -f $OUT/$m/roc.json ] && continue
      srun -N1 -n1 -c256 -w ${HOSTS[$k]} --overlap $EPY coupled_roc.py $m $OUT > $OUT/../logs/roc_$m.log 2>&1
      echo "$m rc=$?"; done ) &
done
wait
