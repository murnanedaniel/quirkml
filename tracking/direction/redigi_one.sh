#!/bin/bash
# Re-digitise one run of m200_L774_mu0 v1 (production settings, reco off, + measurements.root) into redigi/<run>/.
r=$1
D=/pscratch/sd/d/danieltm/quirk_stageA/direction
S=/global/cfs/cdirs/m4958/data/ColliderML/simulation/quirks/m200_L774_mu0/v1/runs/$r
mkdir -p $D/redigi/$r
[ -f $D/redigi/$r/.done ] && { echo "run $r already done"; exit 0; }
cp $S/edm4hep.root $D/redigi/$r/edm4hep.root || exit 1
source /global/cfs/cdirs/m4958/usr/danieltm/ColliderML/quirkml/colliderml/container.sh /pscratch/sd/d/danieltm/quirk_pilots/quirks $D
qml_run "cd /workspace/scripts/simulation && python3 digi_and_reco.py --config /data/configs/3_digi_measroot.yaml --output /data/redigi --output-subdir $r --seed m200_L774_mu0_v1_run$r" > $D/logs/redigi_run$r.log 2>&1
rc=$?
if [ $rc -eq 0 ] && [ -s $D/redigi/$r/measurements.root ]; then touch $D/redigi/$r/.done; rm -f $D/redigi/$r/edm4hep.root; echo "run $r OK"; else echo "run $r FAILED rc=$rc"; fi
