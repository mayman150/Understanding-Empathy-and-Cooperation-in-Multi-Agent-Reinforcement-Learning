#!/bin/bash
# SLURM job-array worker for Compute Canada / Alliance clusters (Narval, Beluga, Cedar, Graham).
# Array task N runs line N of a grid file produced by scripts/make_grid.py.
#
# Submit through scripts/slurm/submit.sh (it sets --array and the concurrency cap), or directly:
#   sbatch --account=def-XXXX --array=1-455%50 scripts/slurm/run_grid.sh grids/pd_report.txt
#
# Defaults below are for the CPU-only Prisoner's Dilemma runs (~2-5 min per 1M steps).  For
# Melting Pot pass GPU resources at submit time, e.g.
#   scripts/slurm/submit.sh grids/harvest.txt 8 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=12:00:00
#
# Environment variables (all optional):
#   VENV         virtualenv built by scripts/slurm/setup_env.sh   (default ~/envs/empathy)
#   RUN_DIR      where TensorBoard logs / checkpoints go          (default $SCRATCH/empathy_runs/<grid name>)
#   PROJECT_DIR  repository root                                    (default: the directory sbatch was run from)
#   PY_MODULE / STDENV   modules to load                            (default python/3.11, StdEnv/2023)
#SBATCH --job-name=empathy
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=slurm_logs/%x_%A_%a.out
set -euo pipefail

GRID=${1:?usage: sbatch --array=1-N scripts/slurm/run_grid.sh <grid file>}
PROJECT_DIR=${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
VENV=${VENV:-$HOME/envs/empathy}
RUN_DIR=${RUN_DIR:-${SCRATCH:-$PROJECT_DIR}/empathy_runs/$(basename "$GRID" .txt)}
PY_MODULE=${PY_MODULE:-python/3.11}
STDENV=${STDENV:-StdEnv/2023}

if command -v module >/dev/null 2>&1; then module load "$STDENV" "$PY_MODULE"; fi
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; fi
cd "$PROJECT_DIR"
mkdir -p "$RUN_DIR"

# dry run outside SLURM:  SLURM_ARRAY_TASK_ID=3 VENV=.venv scripts/slurm/run_grid.sh grids/pd_report.txt
TASK=${SLURM_ARRAY_TASK_ID:?set SLURM_ARRAY_TASK_ID (run as a job array)}
LINE=$(sed -n "${TASK}p" "$GRID")
if [ -z "$LINE" ]; then echo "no line $TASK in $GRID"; exit 1; fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
echo "[$(date)] host=$(hostname) task=$TASK"
echo "python train.py $LINE --run-dir $RUN_DIR"
# shellcheck disable=SC2086   (the grid line is a pre-split argument list)
python train.py $LINE --run-dir "$RUN_DIR"
echo "[$(date)] done"
