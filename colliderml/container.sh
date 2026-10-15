# Run a command in the ODD software image, the way the ColliderML library's colliderml.simulate()
# does: the production scripts at /workspace, outputs at /work, caches at /cache. Sourced by the
# other scripts, not run directly.
#
#   source colliderml/container.sh <work_dir>
#   qml_run "<bash command>"
#
# Mounts: <work_dir> -> /work, <work_dir>/colliderml-production -> /workspace,
#         <work_dir>/cache -> /cache, this repo -> /quirkml.
# Runtime: $QUIRKML_RUNTIME, else the first found of podman-hpc (NERSC), docker, podman, apptainer.

# The default image has Pythia8 and HepMC3 python bindings (OpenDataDetector/sw branch
# feat/python-pythia8-hepmc3), so quirk-gen also runs in it. sw:pr-8 works for the stages alone.
QUIRKML_IMAGE=${QUIRKML_IMAGE:-ghcr.io/opendatadetector/sw:feat-python-pythia8-hepmc3}
QML_WORK=$(cd "${1:?usage: source container.sh <work_dir>}" && pwd)
QML_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
QML_APPTAINER_FALLBACK=/cvmfs/oasis.opensciencegrid.org/mis/apptainer/current/bin/apptainer

qml_runtime() {
  if [ -n "${QUIRKML_RUNTIME:-}" ]; then echo "$QUIRKML_RUNTIME"; return; fi
  for rt in podman-hpc docker podman apptainer singularity; do
    command -v "$rt" >/dev/null 2>&1 && { echo "$rt"; return; }
  done
  [ -x "$QML_APPTAINER_FALLBACK" ] && { echo "$QML_APPTAINER_FALLBACK"; return; }
  echo "no container runtime found: install Docker, Podman or Apptainer, or set QUIRKML_RUNTIME" >&2
  return 1
}
QML_RT=$(qml_runtime) || return 1 2>/dev/null || exit 1

qml_is_apptainer() { case "$(basename "$QML_RT")" in apptainer|singularity) return 0;; *) return 1;; esac; }
# Apptainer keeps its download cache in $HOME by default; the image needs about 6 GB there.
if qml_is_apptainer; then
  export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$QML_WORK/cache/apptainer}
  export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-$QML_WORK/cache/apptainer-tmp}
  mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
fi
QML_SIF=$QML_WORK/sw-$(echo "$QUIRKML_IMAGE" | sed 's|.*:||').sif

# What the plugin in <work_dir>/plugin must have been built from: this image and the current
# QuirkDDG4 source. setup.sh writes it to plugin/BUILD_STAMP; run_stage.sh checks it.
qml_plugin_stamp() {
  echo "$QUIRKML_IMAGE $(cat "$QML_REPO"/sim/QuirkDDG4/CMakeLists.txt "$QML_REPO"/sim/QuirkDDG4/src/* | md5sum | cut -c1-12)"
}

# Make the image available locally (pull once).
qml_pull() {
  if qml_is_apptainer; then
    [ -f "$QML_SIF" ] || "$QML_RT" pull "$QML_SIF" "docker://$QUIRKML_IMAGE"
  else
    "$QML_RT" image exists "$QUIRKML_IMAGE" 2>/dev/null || "$QML_RT" image inspect "$QUIRKML_IMAGE" >/dev/null 2>&1 \
      || "$QML_RT" pull "$QUIRKML_IMAGE"
  fi
}

# Run one bash command inside the image, with the production environment set up.
qml_run() {
  local setup="source /workspace/scripts/cli/setup_container_env.sh >/dev/null 2>&1; \
[ -f /opt/acts-arrow/setup.sh ] && source /opt/acts-arrow/setup.sh; \
[ -f /opt/odd-install/bin/this_odd.sh ] && source /opt/odd-install/bin/this_odd.sh; "
  mkdir -p "$QML_WORK/cache"
  if qml_is_apptainer; then
    # The setup script downloads the Geant4 data (about 2 GB) into /g4data, which under Apptainer
    # is inside a small in-memory overlay. Mount <work_dir>/cache/g4data there instead. --home
    # keeps the container off the user's home directory (and its quota).
    mkdir -p "$QML_WORK/cache/g4data" "$QML_WORK/cache/home"
    "$QML_RT" exec --cleanenv --writable-tmpfs --home "$QML_WORK/cache/home" \
      -B "$QML_WORK:/work,$QML_WORK/colliderml-production:/workspace,$QML_WORK/cache:/cache,$QML_REPO:/quirkml" \
      -B "$QML_WORK/cache/g4data:/g4data" \
      --env COLLIDERML_CACHE=/cache "$QML_SIF" bash -c "$setup $1"
  else
    "$QML_RT" run --rm \
      -v "$QML_WORK:/work" -v "$QML_WORK/colliderml-production:/workspace" \
      -v "$QML_WORK/cache:/cache" -v "$QML_REPO:/quirkml" \
      -e COLLIDERML_CACHE=/cache --entrypoint /bin/bash "$QUIRKML_IMAGE" -c "$setup $1"
  fi
}
