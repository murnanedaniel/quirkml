#!/bin/bash
# Example: one Slurm job that runs one stage over many runs, one task per run, the way ColliderML
# production's multi_node_slurm mode does (run id = first run + SLURM_PROCID). Edit the #SBATCH
# lines for your site; submit one stage at a time, each after the previous has finished:
#   sbatch --export=ALL,CFG=<config_dir>,STAGE=simulation,FIRST=0 quirkml/colliderml/batch/slurm_runs.sh
# (CFG as an absolute path). Tested on NERSC with -q debug: 8 runs of 500 events, one node.
# Sizing: --ntasks = number of runs, --cpus-per-task = the stage's threads (sim_threads,
# digi_threads; 1 for stage1). Generation: run "run_stage.sh <config_dir> generation --run-list 0"
# once first, so that the MadGraph process is built before the tasks start.
# Afterwards: bash colliderml/run_stage.sh <config_dir> status, and rerun what failed with --run-list.
#SBATCH --job-name=quirks
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=quirks_%j.log
set -u
# sbatch runs a copy of this script, so the repo is found from the submit directory (the one
# holding quirkml/, as in the tutorial) unless QUIRKML is set.
QUIRKML=${QUIRKML:-$SLURM_SUBMIT_DIR/quirkml}
[ -f "$QUIRKML/colliderml/run_stage.sh" ] || { echo "no quirkml at $QUIRKML: set QUIRKML" >&2; exit 1; }
srun --kill-on-bad-exit=0 bash -c "bash $QUIRKML/colliderml/run_stage.sh $CFG $STAGE --run-list \$((${FIRST:-0} + SLURM_PROCID))"
