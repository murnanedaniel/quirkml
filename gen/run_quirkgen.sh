#!/bin/bash
# Generate the quirk events of one run with quirk-gen, into the run's directory.
#   bash gen/run_quirkgen.sh <config_dir> init     build the MadGraph process (once per sample)
#   bash gen/run_quirkgen.sh <config_dir> <run>    generate run <run>
# Usually called by colliderml/run_stage.sh <config_dir> generation, which does init and then
# every run. quirk-gen (burzynski-lab/quirk-gen at the commit below, unmodified) is cloned into
# <work_dir>/quirkgen/; gen/quirkgen_run.sh calls its two steps (MadGraph, then Pythia8).
# The seed of run N is the integer ColliderML production makes of <dataset>_<version>_run<N>.
# Environment, first that works:
#   1. the ODD image itself, if it has Pythia8's and HepMC3's python bindings (sw images built with
#      +python, e.g. ghcr.io/opendatadetector/sw:feat-python-pythia8-hepmc3); MadGraph is already in it;
#   2. quirk-gen's LCG_109 view from cvmfs: directly on an el9 host, or through shifter (NERSC) or
#      Apptainer with an el9 image. Set QUIRKGEN_ENV=image|lcg to choose the route, and
#      QUIRKGEN_LCG_VIA=direct|shifter|apptainer to choose how the LCG view is entered.
# Writes to the run directory:
#   events.hepmc, events.json    quirk-gen output
#   events_hepmc2.hepmc          the same events as HepMC2, numbered from 0 (the pileup merge needs 0).
#                                With the LCG view it is also needed because the ODD image's HepMC3
#                                3.3.0 can't read LCG_109's 3.03 files.
# The MadGraph process built by one route doesn't work with the other: delete
# <work_dir>/quirkgen/procdir_<route> to rebuild.
set -eo pipefail   # not -u: the LCG setup script uses unset variables
QUIRKGEN_REPO=https://github.com/burzynski-lab/quirk-gen.git
QUIRKGEN_COMMIT=9980db3
LCG_VIEW=${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt}
EL9_IMAGE=registry.cern.ch/atlasadc/atlas-grid-almalinux9:latest
CFG=$(cd "${1:?usage: bash gen/run_quirkgen.sh <config_dir> init|<run>}" && pwd)
RUN=${2:?usage: bash gen/run_quirkgen.sh <config_dir> init|<run>}
source "$CFG/sample.env"
QUIRKML=$(cd "$(dirname "$0")/.." && pwd)
GEN=$WORK_DIR/quirkgen
CORES=${QUIRKGEN_CORES:-$SIM_THREADS}

seed_of() {   # production's hash_seed_string (scripts/simulation/utils/config.py) of the run's seed string
  python3 -c 'import hashlib, sys
s = abs(int.from_bytes(hashlib.md5(sys.argv[1].encode()).digest()[:4], "big", signed=True)) % 900000000
print(s or 1)' "${DATASET}_${VERSION}_run$1"
}

# args for gen/quirkgen_run.sh, with paths under the given work-dir mount point
qg_args() {   # qg_args <work_dir as seen by the environment> <quirkml dir as seen> <route>
  local w=$1 q=$2 proc=$1/quirkgen/procdir_$3
  if [ "$RUN" = init ]; then
    echo "init $w/quirkgen/quirk-gen $proc $MASS_GEV"
  else
    echo "run $w/quirkgen/quirk-gen $proc $w/$RUNS_DIR/$RUN $MASS_GEV $LAMBDA_EV $EVENTS_PER_RUN $(seed_of "$RUN") $CORES $q/colliderml/convert_to_hepmc2.py"
  fi
}

# Inside the LCG environment (second pass of route 2).
if [ "${3:-}" = "--inside-lcg" ]; then
  source "$LCG_VIEW/setup.sh" >/dev/null
  bash "$QUIRKML/gen/quirkgen_run.sh" $(qg_args "$WORK_DIR" "$QUIRKML" lcg)
  exit 0
fi

mkdir -p "$GEN"
if [ ! -d "$GEN/quirk-gen/.git" ]; then
  git clone -q "$QUIRKGEN_REPO" "$GEN/quirk-gen" && git -C "$GEN/quirk-gen" checkout -q "$QUIRKGEN_COMMIT"
fi
[ "$RUN" = init ] || echo "run $RUN: seed $(seed_of "$RUN") (${DATASET}_${VERSION}_run$RUN), $EVENTS_PER_RUN events"

# Route 1: the ODD image. Its spack tree has MadGraph; when pythia8 and hepmc3 are built with +python,
# their modules are in the spack tree but not on the image's PYTHONPATH, so add them.
source "$QUIRKML/colliderml/container.sh" "$WORK_DIR"
SPACK_PY='S=/spack/opt/spack/linux-x86_64; export PYTHONPATH=$(ls -d $S/pythia8-*/lib $S/hepmc3-*/lib/python3*/site-packages 2>/dev/null | tr "\n" :)$PYTHONPATH;'
if [ "${QUIRKGEN_ENV:-}" != "lcg" ] && qml_run "$SPACK_PY python3 -c 'import pythia8, pyHepMC3'" >/dev/null 2>&1; then
  qml_run "$SPACK_PY bash /quirkml/gen/quirkgen_run.sh $(qg_args /work /quirkml image)"
  exit 0
fi
[ "${QUIRKGEN_ENV:-}" = "image" ] && { echo "$QUIRKML_IMAGE has no Pythia8/HepMC3 python bindings" >&2; exit 1; }

# Route 2: LCG_109 from cvmfs.
[ -d "$LCG_VIEW" ] || { echo "no $LCG_VIEW: quirk-gen needs cvmfs (sft.cern.ch), or an image with Pythia8 python (the default QUIRKML_IMAGE has it)" >&2; exit 1; }
VIA=${QUIRKGEN_LCG_VIA:-}
if [ -z "$VIA" ]; then
  if grep -qi 'platform:el9' /etc/os-release 2>/dev/null; then VIA=direct
  elif command -v shifter >/dev/null; then VIA=shifter
  elif qml_is_apptainer; then VIA=apptainer; fi
fi
if [ "$VIA" = direct ]; then
  bash "$0" "$CFG" "$RUN" --inside-lcg
elif [ "$VIA" = shifter ]; then
  shifter --image=$EL9_IMAGE --module=cvmfs /bin/bash "$0" "$CFG" "$RUN" --inside-lcg
elif [ "$VIA" = apptainer ]; then
  "$QML_RT" exec -B /cvmfs -B "$WORK_DIR" -B "$QUIRKML" "docker://$EL9_IMAGE" /bin/bash "$0" "$CFG" "$RUN" --inside-lcg
else
  echo "the LCG view needs an el9 host, shifter or Apptainer" >&2; exit 1
fi
