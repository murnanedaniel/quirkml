#!/bin/bash
# Re-digitise one run of m200_L774_mu0 v1 (production settings + measurements.root) into the scaling dir, for the
# cluster-direction features of the extra training events (copy of tracking/direction/redigi_one.sh, other output dir).
r=$1
D=/pscratch/sd/d/danieltm/quirk_tracking/scaling/direction
S=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/runs/$r
mkdir -p $D/redigi/$r
[ -f $D/redigi/$r/.done ] && { echo "run $r already done"; exit 0; }
cp $S/edm4hep.root $D/redigi/$r/edm4hep.root || { echo "run $r FAILED copy"; exit 0; }
source /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/colliderml/container.sh /pscratch/sd/d/danieltm/quirk_pilots/quirks $D
qml_run "cd /workspace/scripts/simulation && python3 digi_and_reco.py --config /data/configs/3_digi_measroot.yaml --output /data/redigi --output-subdir $r --seed m200_L774_mu0_v1_run$r" > $D/logs/redigi_run$r.log 2>&1
rc=$?
rm -f $D/redigi/$r/edm4hep.root
if [ $rc -eq 0 ] && [ -s $D/redigi/$r/measurements.root ]; then touch $D/redigi/$r/.done; echo "run $r OK"; else echo "run $r FAILED rc=$rc"; fi
