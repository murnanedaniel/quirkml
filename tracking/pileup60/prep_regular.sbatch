#!/bin/bash
#SBATCH -A m4958 -C cpu -q regular -N 4 -t 03:00:00 -J qtrk_pu60_dir_train
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/prep_train_%j.out
# direction features for the mu 60 training blocks 3-99 (runs 6-199): 4 nodes, ~24 blocks each
S=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/mu60_prep.sh
DS=${DS:-m200_L774_mu60}
for i in 0 1 2 3; do
  b0=$((3 + 25 * i)); b1=$((b0 + 24)); [ $b1 -gt 99 ] && b1=99
  srun -N1 -n1 -r $i -c 256 --exact bash $S $DS $b0 $b1 24 > /pscratch/sd/d/danieltm/quirk_pu60/logs/prep_train_${SLURM_JOB_ID}_node$i.log 2>&1 &
done
wait
ls /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/direction/$DS/*.parquet | wc -l
