#!/bin/bash
# Make quirk-gen output readable by the ColliderML sim image (sw:pr-8).
# quirk-gen writes HepMC3 3.03.01 ASCII (LCG_109); the image's HepMC3 3.3.0 reader rejects some
# events in those files ("too few vertices were parsed") and silently returns empty events.
# This rewrites the file as HepMC2 ASCII using LCG_109's own reader, which reads it correctly.
#   bash convert_quirkgen_hepmc.sh <quirk-gen events.hepmc> <output .hepmc>
if ! grep -qi "almalinux" /etc/os-release 2>/dev/null; then
  exec shifter --image=registry.cern.ch/atlasadc/atlas-grid-almalinux9:latest --module=cvmfs /bin/bash "$0" "$@"
fi
source ${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt}/setup.sh >/dev/null 2>&1
python3 "$(dirname "$0")/convert_to_hepmc2.py" "$1" "$2"
