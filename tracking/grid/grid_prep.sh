#!/bin/bash
# Stage F data prep for one grid test set (runs 0-3, events 0-1999), no retraining:
#  1. re-digitise runs 0-3 from their kept edm4hep.root with the production digitisation settings plus
#     output_measurements_root (as tracking/direction/redigi_one.sh does for m200_L774_mu0), seeds as production;
#  2. cluster-direction features per 1000-event block (grid_direction.py, checked hit by hit against production);
#  3. pair-truth copy of events 0-1999 (tracking/make_pair_truth.py).
# Scratch: $SCR/<ds> (measurements.root deleted afterwards). Outputs on CFS under quirk_tracking_data/{direction,pair_truth}/<ds>.
# Usage: grid_prep.sh <dataset>
set -uo pipefail
ds=$1
Q=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml
SIM=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/$ds/v1/runs
TR=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/testruns/$ds
DIR_OUT=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/direction/$ds
PT_OUT=/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirk_tracking_data/pair_truth/$ds
SCR=/pscratch/sd/d/danieltm/quirk_stageF/direction/$ds
EVPY=/global/cfs/cdirs/m4958/usr/danieltm/ml_tracking_pilot_2026-09/envs/eval48/bin/python
umask 007
mkdir -p $SCR/configs $SCR/logs $SCR/redigi
log() { echo "[$ds] $(date -Is) $*"; }
if [ ! -f $DIR_OUT/.done ]; then
  sed "s/^dataset: .*/dataset: \"$ds\"/" /pscratch/sd/d/danieltm/quirk_stageA/direction/configs/3_digi_measroot.yaml > $SCR/configs/3_digi_measroot.yaml
  redigi() {
    r=$1; [ -f $SCR/redigi/$r/.done ] && return 0
    mkdir -p $SCR/redigi/$r; cp $SIM/$r/edm4hep.root $SCR/redigi/$r/edm4hep.root || return 1
    ( source $Q/colliderml/container.sh /pscratch/sd/d/danieltm/quirk_pilots/quirks $SCR
      qml_run "cd /workspace/scripts/simulation && python3 digi_and_reco.py --config /data/configs/3_digi_measroot.yaml --output /data/redigi --output-subdir $r --seed ${ds}_v1_run$r" ) > $SCR/logs/redigi_run$r.log 2>&1
    [ -s $SCR/redigi/$r/measurements.root ] && touch $SCR/redigi/$r/.done && rm -f $SCR/redigi/$r/edm4hep.root
  }
  for r in 0 1 2 3; do redigi $r & done; wait
  for r in 0 1 2 3; do [ -f $SCR/redigi/$r/.done ] || { log "redigi run $r FAILED"; exit 1; }; done
  log "redigi ok"
  /pscratch/sd/d/danieltm/envs/ml-tracking-review/bin/python $Q/tracking/grid/grid_direction.py $ds $SCR $TR/parquet/reco/tracker_hits $DIR_OUT || { log "direction FAILED"; exit 1; }
  touch $DIR_OUT/.done
  for r in 0 1 2 3; do rm -f $SCR/redigi/$r/measurements.root; done
  log "direction ok"
fi
if [ ! -f $PT_OUT/pair_truth_manifest_events0-1999.json ]; then
  $EVPY $Q/tracking/make_pair_truth.py --dataset $TR/parquet_r1 --events 0 2000 --out $PT_OUT --name quirks.$ds.v1.pairtruth > $SCR/logs/pair_truth.log 2>&1 || { log "pair truth FAILED"; exit 1; }
  log "pair truth ok"
fi
log done
