#!/bin/bash
# Example: one Slurm job that runs one stage over many runs, one task per run, the way ColliderML
# production's multi_node_slurm mode does (run id = first run + SLURM_PROCID). Edit the #SBATCH
# lines for your site; submit one stage at a time, each after the previous has finished:
#   sbatch --export=ALL,CFG=<config_dir>,STAGE=simulation,FIRST=0 colliderml/batch/slurm_runs.sh
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
QUIRKML=${QUIRKML:-$(cd "$(dirname "$0")/../.." && pwd)}
srun --kill-on-bad-exit=0 bash -c "bash $QUIRKML/colliderml/run_stage.sh $CFG $STAGE --run-list \$((${FIRST:-0} + SLURM_PROCID))"
