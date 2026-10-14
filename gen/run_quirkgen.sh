#!/bin/bash
# Generate the quirk events of one sample with quirk-gen and put them, converted, in its run directory.
#   bash gen/run_quirkgen.sh <config_dir>          (config_dir from colliderml/make_sample.py)
# Runs quirk-gen (burzynski-lab/quirk-gen at the commit below, unmodified) in <work_dir>/quirkgen/.
# Environment, first that works:
#   1. the ODD image itself, if it has Pythia8's python bindings (sw images built with +python);
#   2. quirk-gen's LCG_109 view from cvmfs: directly on an el9 host, or through shifter (NERSC) or
#      Apptainer with an el9 image. Set QUIRKGEN_ENV=image|lcg to choose.
# Writes to the run directory:
#   events.hepmc, events.json    quirk-gen output
#   events_hepmc2.hepmc          the same events as HepMC2, numbered from 0. With the LCG view this is
#                                needed: the ODD image's HepMC3 3.3.0 can't read LCG_109's 3.03 files.
set -eo pipefail   # not -u: the LCG setup script uses unset variables
QUIRKGEN_REPO=https://github.com/burzynski-lab/quirk-gen.git
QUIRKGEN_COMMIT=9980db3
LCG_VIEW=${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt}
EL9_IMAGE=registry.cern.ch/atlasadc/atlas-grid-almalinux9:latest
CFG=$(cd "${1:?usage: bash gen/run_quirkgen.sh <config_dir>}" && pwd)
source "$CFG/sample.env"
QUIRKML=$(cd "$(dirname "$0")/.." && pwd)
GEN=$WORK_DIR/quirkgen
ARGS="--mass $MASS_GEV --events $EVENTS --lambda $LAMBDA_EV --seed $SEED"

# Inside the LCG environment (second pass of route 2).
if [ "${1:-}" != "" ] && [ "${2:-}" = "--inside-lcg" ]; then
  source "$LCG_VIEW/setup.sh" >/dev/null
  cd "$GEN/quirk-gen" && source setup.sh >/dev/null
  scripts/run_chain.sh $ARGS --out "$GEN/$DATASET"
  cp "$GEN/$DATASET/events.hepmc" "$GEN/$DATASET/events.json" "$WORK_DIR/runs/$DATASET/"
  python3 "$QUIRKML/colliderml/convert_to_hepmc2.py" "$WORK_DIR/runs/$DATASET/events.hepmc" \
    "$WORK_DIR/runs/$DATASET/events_hepmc2.hepmc"
  exit 0
fi

mkdir -p "$GEN" "$WORK_DIR/runs/$DATASET"
if [ ! -d "$GEN/quirk-gen/.git" ]; then
  git clone -q "$QUIRKGEN_REPO" "$GEN/quirk-gen" && git -C "$GEN/quirk-gen" checkout -q "$QUIRKGEN_COMMIT"
fi

# Route 1: the ODD image.
source "$QUIRKML/colliderml/container.sh" "$WORK_DIR"
if [ "${QUIRKGEN_ENV:-}" != "lcg" ] && qml_run "python3 -c 'import pythia8, pyHepMC3'" >/dev/null 2>&1; then
  echo "quirk-gen in $QUIRKML_IMAGE"
  qml_run "export QG_ROOT=/work/quirkgen/quirk-gen PATH=/work/quirkgen/quirk-gen/scripts:\$PATH &&
           cd \$QG_ROOT && scripts/run_chain.sh $ARGS --out /work/quirkgen/$DATASET &&
           cp /work/quirkgen/$DATASET/events.hepmc /work/quirkgen/$DATASET/events.json /work/runs/$DATASET/ &&
           python3 /quirkml/colliderml/convert_to_hepmc2.py /work/runs/$DATASET/events.hepmc /work/runs/$DATASET/events_hepmc2.hepmc"
  echo "quirk events ready in $WORK_DIR/runs/$DATASET"; exit 0
fi
[ "${QUIRKGEN_ENV:-}" = "image" ] && { echo "the image has no Pythia8 python bindings" >&2; exit 1; }

# Route 2: LCG_109 from cvmfs.
[ -d "$LCG_VIEW" ] || { echo "no $LCG_VIEW: quirk-gen needs cvmfs (sft.cern.ch), or an image with Pythia8 python" >&2; exit 1; }
echo "quirk-gen in $LCG_VIEW"
if grep -qi 'platform:el9' /etc/os-release 2>/dev/null; then
  bash "$0" "$CFG" --inside-lcg
elif command -v shifter >/dev/null; then
  shifter --image=$EL9_IMAGE --module=cvmfs /bin/bash "$0" "$CFG" --inside-lcg
elif qml_is_apptainer; then
  "$QML_RT" exec -B /cvmfs -B "$WORK_DIR" -B "$QUIRKML" "docker://$EL9_IMAGE" /bin/bash "$0" "$CFG" --inside-lcg
else
  echo "the LCG view needs an el9 host, shifter or Apptainer" >&2; exit 1
fi
echo "quirk events ready in $WORK_DIR/runs/$DATASET"
