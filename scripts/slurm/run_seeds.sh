#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mamoham3@ualberta.ca
#SBATCH --account=aip-machado
#SBATCH --job-name=empathy_seeds
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=03:00:00
#SBATCH --cpu-freq=Performance
#
# One configuration, many seeds: array task N runs train.py with --seed N (same pattern as
# `seed=$SLURM_ARRAY_TASK_ID` in the AgarLE scripts).  Everything after the script name is
# passed to train.py unchanged.
#
#   mkdir -p slurm_logs
#   sbatch --array=1-10 scripts/slurm/run_seeds.sh --env-id pd --formulation ei --signal value --alpha 20
#   sbatch --array=1-5 --job-name=harvest_sia --gres=gpu:1 --mem=32GB --time=12:00:00 \
#       scripts/slurm/run_seeds.sh --env-id meltingpot:commons_harvest__open --formulation sia --alpha 0.1
#
# Runs go to $RUN_ROOT/<job name>/ (override with RUN_DIR); paths / modules come from cluster.env.
set -euo pipefail

# SLURM copies this script to /var/spool/slurmd/..., so the repo cannot be found from BASH_SOURCE.
# Order: EMPATHY_REPO (exported by submit.sh / pd.sh) -> the directory sbatch was run from -> script location (dry runs).
for candidate in "${EMPATHY_REPO:-}" "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"; do
  if [ -n "$candidate" ] && [ -f "$candidate/scripts/slurm/cluster.env" ]; then REPO=$candidate; break; fi
done
[ -n "${REPO:-}" ] || { echo "cannot locate the repository (scripts/slurm/cluster.env); export EMPATHY_REPO=<repo path>"; exit 1; }
source "$REPO/scripts/slurm/cluster.env"
RUN_DIR=${RUN_DIR:-$RUN_ROOT/${SLURM_JOB_NAME:-seeds}}

if command -v module >/dev/null 2>&1; then module load "$STDENV" "$PY_MODULE"; fi
[ -f "$VENV/bin/activate" ] || { echo "virtualenv not found at $VENV -- run scripts/slurm/setup_env.sh (or set VENV)"; exit 1; }
source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1   # never pick up ~/.local packages instead of the venv
cd "$PROJECT_DIR"
mkdir -p "$RUN_DIR"

seed=${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID (run as a job array)}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local} seed=$seed"
echo "python train.py $* --seed $seed --run-dir $RUN_DIR"
python train.py "$@" --seed "$seed" --run-dir "$RUN_DIR"
echo "[$(date)] done"
