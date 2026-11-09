#!/bin/bash
#SBATCH -A m4958 -C cpu -q debug -N 1 -c 256 -t 00:30:00 -J qtrk_pu60_cache0
#SBATCH -o /pscratch/sd/d/danieltm/quirk_pu60/logs/cache0_%j.out
# Scratch R1 copy of block 0 only (convert_to_r1.convert_file, for the size check until the shared R1 copy lands),
# then dir + pair caches of block 0.
set -uo pipefail
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
T=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/tracking
PKG=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu60/v1/parquet
R=/pscratch/sd/d/danieltm/quirk_pu60/scratch_r1/m200_L774_mu60
mkdir -p $R/reco/tracker_hits $R/truth/particles
f=quirks.m200_L774_mu60.v1
date
[ -s $R/reco/tracker_hits/$f.reco.tracker_hits.events0-999.parquet ] || cd $T && $PY -c "
import sys; sys.path.insert(0,'$T'); import convert_to_r1 as c
c.convert_file('$PKG/reco/tracker_hits/$f.reco.tracker_hits.events0-999.parquet','$PKG/truth/tracker_simhits/$f.truth.tracker_simhits.events0-999.parquet','$R/reco/tracker_hits/$f.reco.tracker_hits.events0-999.parquet')" || exit 1
cp -n $PKG/truth/particles/$f.truth.particles.events0-999.parquet $R/truth/particles/
date
$PY $T/pileup60/make_cache60.py --r1 $R --direction /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/direction/m200_L774_mu60 \
  --out-dir /pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir --out-pair /pscratch/sd/d/danieltm/quirk_pu60/cache/m200_L774_mu60_dir_pair --blocks 0 --workers 1
date
