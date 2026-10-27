#!/bin/bash
#SBATCH -A m4958 -C cpu -q regular -N 1 -t 01:00:00 -J qtrk_oraclepkg
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/split/logs/oraclepkg_%j.out
# Oracle pair tracks from packaged events 20000-999999 (one pickle per 1000-event file), 64 workers.
P=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/parquet/reco/tracker_hits
OUT=/pscratch/sd/d/danieltm/quirk_tracking/split/data/pkg_oracle
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
S=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/split/oracle_from_package.py
ls $P/*.parquet | awk -F'events' '{split($2,a,"-"); if (a[1]+0 >= 20000) print}' > $OUT.list
N=$(wc -l < $OUT.list); echo "$N files"
split -n l/64 -d $OUT.list $OUT.part.
for f in $OUT.part.*; do srun -n 1 -c 2 --exact $PY $S --files $(cat $f) --out-dir $OUT & done
wait
ls $OUT/*.pkl | wc -l
