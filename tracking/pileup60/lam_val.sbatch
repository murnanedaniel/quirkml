#!/bin/bash
#SBATCH -A m4958 -C cpu -q regular -N 1 -c 256 -t 01:30:00 -J qtrk_pu60_lam_val
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/lam_val_%j.out
# val builder study (builders60.py, val 2000-2999) per string scale for env JOBS="SET:SCORED_RUN_DIR:OUT_JSON ..."
for j in $JOBS; do IFS=: read DS R O <<< "$j"
  B60_CACHE=/pscratch/sd/d/danieltm/quirk_pu60/cache/${DS}_dir_pair /pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python \
    /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/pileup60/builders60.py $R $O cc:0.5 cc:0.4 cc:0.6 cc:0.7 \
    || { echo "$j FAILED"; exit 1; }
done
