#!/bin/bash
#SBATCH -A m4958 -C cpu -q regular -N 1 -t 01:30:00 -J qtrk_scal_dir
#SBATCH -o /pscratch/sd/d/danieltm/quirk_tracking/scaling/logs/direction_%j.out
# Direction features for training events 20000-153999 (runs 40-307, blocks 20-153): re-digitise, then build features.
umask 007
R=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking/gnn/scaling
date
seq ${R0:-40} ${R1:-307} | xargs -P 24 -I{} $R/redigi_one.sh {}
date
nfail=$(seq ${R0:-40} ${R1:-307} | while read r; do [ -f /pscratch/sd/d/danieltm/quirk_tracking/scaling/direction/redigi/$r/.done ] || echo $r; done | wc -l)
echo "runs without .done: $nfail"
seq $(( ${R0:-40} / 2 )) $(( ${R1:-307} / 2 )) | xargs -P 16 -I{} /pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python $R/build_dir_features.py {}
date
ls /pscratch/sd/d/danieltm/quirk_tracking/scaling/direction/features | wc -l
