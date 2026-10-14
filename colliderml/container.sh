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

QUIRKML_IMAGE=${QUIRKML_IMAGE:-ghcr.io/opendatadetector/sw:pr-8}
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
    # --writable-tmpfs: the setup script links the Geant4 data into the image's own data directory.
    "$QML_RT" exec --cleanenv --writable-tmpfs \
      -B "$QML_WORK:/work,$QML_WORK/colliderml-production:/workspace,$QML_WORK/cache:/cache,$QML_REPO:/quirkml" \
      --env COLLIDERML_CACHE=/cache "$QML_SIF" bash -c "$setup $1"
  else
    "$QML_RT" run --rm \
      -v "$QML_WORK:/work" -v "$QML_WORK/colliderml-production:/workspace" \
      -v "$QML_WORK/cache:/cache" -v "$QML_REPO:/quirkml" \
      -e COLLIDERML_CACHE=/cache --entrypoint /bin/bash "$QUIRKML_IMAGE" -c "$setup $1"
  fi
}
