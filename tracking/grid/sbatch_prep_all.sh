#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 8 -t 00:30:00 -J qtrk_F_prep
#SBATCH -o /pscratch/sd/d/danieltm/quirk_stageF/logs/prep_all_%j.out
# grid_prep.sh for every set in grid_sets_mu0.txt: node k takes every 8th set, 3 sets at a time (32 at once was 40x slower) (sets done are skipped).
G=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/grid
for k in 0 1 2 3 4 5 6 7; do
  srun -N1 -n1 -r $k --exact -c 256 bash -c "awk 'NR % 8 == $k' $G/grid_sets_mu0.txt | xargs -P 3 -I{} bash $G/grid_prep.sh {}" &
done
wait
