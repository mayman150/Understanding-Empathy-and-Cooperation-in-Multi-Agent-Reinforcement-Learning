#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mamoham3@ualberta.ca
#SBATCH --account=aip-machado
#SBATCH --job-name=empathy_grid
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --time=03:00:00
#SBATCH --cpu-freq=Performance
#
# SLURM job-array worker (Compute Canada / Alliance): array task N runs line N of a grid file
# produced by scripts/make_grid.py.
#
# Submit through scripts/slurm/submit.sh (sets --array, the concurrency cap and the job name):
#   scripts/slurm/submit.sh grids/pd_report.txt 50
# or directly (mkdir -p slurm_logs first):
#   sbatch --array=1-455%50 scripts/slurm/run_grid.sh grids/pd_report.txt
#
# The header is sized for the CPU-only Prisoner's Dilemma (5M steps: roughly 20-60 min on 2 cluster CPUs).  For
# Melting Pot pass GPU resources at submit time:
#   scripts/slurm/submit.sh grids/harvest.txt 8 --gres=gpu:1 --cpus-per-task=4 --mem=32GB --time=12:00:00
#
# Paths / modules come from scripts/slurm/cluster.env (VENV, PROJECT_DIR, RUN_ROOT, PY_MODULE, ...).
# Dry run without SLURM:
#   SLURM_ARRAY_TASK_ID=3 VENV=.venv PROJECT_DIR=$PWD RUN_ROOT=/tmp/runs scripts/slurm/run_grid.sh grids/pd_report.txt
set -euo pipefail

GRID=${1:?usage: sbatch --array=1-N scripts/slurm/run_grid.sh <grid file>}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/cluster.env"
RUN_DIR=${RUN_DIR:-$RUN_ROOT/$(basename "$GRID" .txt)}

if command -v module >/dev/null 2>&1; then module load "$STDENV" "$PY_MODULE"; fi
[ -f "$VENV/bin/activate" ] || { echo "virtualenv not found at $VENV -- run scripts/slurm/setup_env.sh (or set VENV)"; exit 1; }
source "$VENV/bin/activate"
export PYTHONNOUSERSITE=1   # never pick up ~/.local packages instead of the venv
cd "$PROJECT_DIR"
mkdir -p "$RUN_DIR"

TASK=${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID (run as a job array)}
LINE=$(sed -n "${TASK}p" "$GRID")
if [ -z "$LINE" ]; then echo "no line $TASK in $GRID"; exit 1; fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local} task=$TASK grid=$GRID"
echo "python train.py $LINE --run-dir $RUN_DIR"
# shellcheck disable=SC2086   (the grid line is a pre-split argument list)
python train.py $LINE --run-dir "$RUN_DIR"
echo "[$(date)] done"
