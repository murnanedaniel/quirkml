#!/bin/bash
# Run the quirk ddsim driver inside the ColliderML el9 Shifter container, with
# the QuirkDDG4 plugin on DD4HEP_LIBRARY_PATH.  All arguments are forwarded to
# run_quirk_ddsim.py.
#
#   bash run_in_shifter.sh --compact ../geometry/QuirkBarrel.xml \
#        --input inputs/quirk_ev4.hepmc3 --output out/quirk_ev4_edm4hep.root
CONTAINER=registry.cern.ch/atlasadc/atlas-grid-almalinux9:latest
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUIRK_LIB="${QUIRK_LIB:-$HERE/QuirkDDG4/build/lib}"

if ! grep -qi "almalinux\|red hat\|centos" /etc/os-release 2>/dev/null; then
  echo "[run] re-exec inside shifter container $CONTAINER"
  exec shifter --image=$CONTAINER --module=cvmfs /bin/bash "$0" "$@"
fi
echo "[run] inside container: $(grep -o 'AlmaLinux[^\"]*' /etc/os-release | head -1)"

SW=${COLLIDERML_SW:-/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software}
DD4=$SW/OtherLibraries/dd4hep_versions/dd4hep-colliderml-new
ACTS=$SW/OtherLibraries/acts_versions/acts_v4/build/this_acts.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh || true
source $DD4/bin/thisdd4hep.sh || true
# acts setup provides the OpenDataDetector plugin libs (needed for the ODD run)
[ -f "$ACTS" ] && source "$ACTS" || true

# make the plugin discoverable
export DD4HEP_LIBRARY_PATH=$QUIRK_LIB:$DD4HEP_LIBRARY_PATH
export LD_LIBRARY_PATH=$QUIRK_LIB:$LD_LIBRARY_PATH
echo "[run] DD4HEP_LIBRARY_PATH includes $QUIRK_LIB"

cd "$HERE"
python run_quirk_ddsim.py "$@"
