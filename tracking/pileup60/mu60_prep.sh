#!/bin/bash
# mu 60 direction features: re-digitise runs (production digitisation + measurements.root, seeds as production), then
# features per 1000-event block (mu60_direction.py, hit-by-hit check). Resumable (.done per run, file per block).
# Usage: mu60_prep.sh <dataset> <first block> <last block> <parallel runs>
set -uo pipefail
ds=$1; B0=$2; B1=$3; P=${4:-6}
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml
SIM=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/$ds/v1/runs
PKG=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/$ds/v1/parquet/reco/tracker_hits
OUT=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/direction/$ds
SCR=/pscratch/sd/d/danieltm/quirk_pu60/direction/$ds
PY=/pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python
umask 007
mkdir -p $SCR/configs $SCR/logs $SCR/redigi $OUT
sed "s/^dataset: .*/dataset: \"$ds\"/" /pscratch/sd/d/danieltm/quirk_stageA/direction/configs/3_digi_measroot.yaml > $SCR/configs/3_digi_measroot.yaml
export SCR SIM Q ds
redigi() {
  r=$1; [ -f $SCR/redigi/$r/.done ] && return 0
  mkdir -p $SCR/redigi/$r; cp $SIM/$r/edm4hep.root $SCR/redigi/$r/edm4hep.root || { echo "run $r copy FAILED"; return 0; }
  t0=$(date +%s)
  ( source $Q/colliderml/container.sh /pscratch/sd/d/danieltm/quirk_pilots/quirks $SCR
    qml_run "cd /workspace/scripts/simulation && python3 digi_and_reco.py --config /data/configs/3_digi_measroot.yaml --output /data/redigi --output-subdir $r --seed ${ds}_v1_run$r" ) > $SCR/logs/redigi_run$r.log 2>&1
  rm -f $SCR/redigi/$r/edm4hep.root
  if [ -s $SCR/redigi/$r/measurements.root ]; then touch $SCR/redigi/$r/.done; echo "run $r OK $(( $(date +%s) - t0 )) s"; else echo "run $r FAILED"; fi
}
export -f redigi
date
seq $((2*B0)) $((2*B1+1)) | xargs -P $P -I{} bash -c 'redigi {}'
for b in $(seq $B0 $B1); do
  if [ -f $SCR/redigi/$((2*b))/.done ] && [ -f $SCR/redigi/$((2*b+1))/.done ]; then
    $PY $Q/tracking/pileup60/mu60_direction.py $ds $SCR $PKG $OUT $b && rm -f $SCR/redigi/$((2*b))/measurements.root $SCR/redigi/$((2*b+1))/measurements.root
  else echo "block $b: runs missing"; fi
done
date
