#!/bin/bash
# Build the QuirkDDG4 plugin against the ColliderML key4hep stack
# (Geant4 11.3 + dd4hep 1.32, LCG_107).  CPU only.
#
# MUST be built inside the ColliderML el9 Shifter container: the cvmfs gcc
# toolchain is built for el9, but Perlmutter login/compute nodes run SLES, so
# gcc's fixincludes pthread.h mismatches the host glibc and even <mutex> fails
# to compile on the bare node.  This script re-execs itself inside shifter.
# NB: no `set -e` -- the LCG setup.sh emits a benign non-zero R/libreadline
# warning that would otherwise abort the script.
CONTAINER=registry.cern.ch/atlasadc/atlas-grid-almalinux9:latest
if ! grep -qi "almalinux\|red hat\|centos" /etc/os-release 2>/dev/null; then
  echo "[build.sh] re-exec inside shifter container $CONTAINER"
  exec shifter --image=$CONTAINER --module=cvmfs /bin/bash "$0" "$@"
fi
echo "[build.sh] inside container: $(grep PRETTY_NAME /etc/os-release)"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# DD4hep install to build against (override with DD4HEP_DIR)
SW=${COLLIDERML_SW:-/global/cfs/cdirs/m4958/usr/danieltm/ColliderML/software}
DD4=${DD4HEP_DIR:-$SW/OtherLibraries/dd4hep_versions/dd4hep-colliderml-new}

source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh || true
source $DD4/bin/thisdd4hep.sh || true

: "${DD4hepINSTALL:=$DD4}"
echo "DD4hepINSTALL = $DD4hepINSTALL"
echo "Geant4         = $(which geant4-config 2>/dev/null)"

cmake -S "$HERE" -B "$HERE/build" \
      -DDD4hep_DIR="$DD4hepINSTALL" \
      -DCMAKE_BUILD_TYPE=Release
cmake --build "$HERE/build" -j8

echo
echo "=== built libraries ==="
ls -la "$HERE/build/lib"/libQuirkDDG4.* 2>/dev/null
echo
echo "=== factory check (expect 'QuirkPhysics') ==="
listcomponents_dd4hep "$HERE/build/lib/libQuirkDDG4.so" 2>/dev/null | grep -i quirk || \
  grep -i quirk "$HERE/build/lib"/*.components 2>/dev/null || echo "(factory not found)"
