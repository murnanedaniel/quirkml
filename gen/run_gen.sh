#!/bin/bash
# run gen_quirk_pythia.py inside the el9 container with LCG_107 (Pythia 8.313)
if ! grep -qi "almalinux" /etc/os-release 2>/dev/null; then
  exec shifter --image=registry.cern.ch/atlasadc/atlas-grid-almalinux9:latest --module=cvmfs /bin/bash "$0" "$@"
fi
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh >/dev/null 2>&1
python "$(dirname "$0")/gen_quirk_pythia.py" "$@"
