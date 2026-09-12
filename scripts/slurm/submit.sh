#!/bin/bash
# Submit a grid file as a SLURM job array with a concurrency cap.
#
#   scripts/slurm/submit.sh <grid file> [max concurrent jobs=50] [extra sbatch options...]
#
#   export SLURM_ACCOUNT=def-XXXX            # or pass --account=def-XXXX as an extra option
#   scripts/slurm/submit.sh grids/pd_report.txt 50
#   scripts/slurm/submit.sh grids/harvest.txt 8 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=12:00:00
#
# Job arrays are capped at MaxArraySize (1000 on most Alliance clusters); larger grids are split
# into consecutive arrays automatically.
set -euo pipefail
cd "$(dirname "$0")/../.."

GRID=${1:?usage: scripts/slurm/submit.sh <grid file> [max concurrent] [sbatch options...]}
CONCURRENT=${2:-50}
shift $(( $# >= 2 ? 2 : 1 ))
N=$(grep -c . "$GRID")
[ "$N" -gt 0 ] || { echo "empty grid $GRID"; exit 1; }
MAX_ARRAY=${MAX_ARRAY:-1000}
mkdir -p slurm_logs

ACCOUNT_OPT=()
if [ -n "${SLURM_ACCOUNT:-}" ]; then ACCOUNT_OPT=(--account="$SLURM_ACCOUNT"); fi

start=1
while [ "$start" -le "$N" ]; do
  end=$(( start + MAX_ARRAY - 1 )); [ "$end" -gt "$N" ] && end=$N
  echo "sbatch ${ACCOUNT_OPT[*]} --array=${start}-${end}%${CONCURRENT} $* scripts/slurm/run_grid.sh $GRID"
  sbatch "${ACCOUNT_OPT[@]}" --array="${start}-${end}%${CONCURRENT}" "$@" scripts/slurm/run_grid.sh "$GRID"
  start=$(( end + 1 ))
done
echo "submitted $N jobs from $GRID; logs in slurm_logs/, runs in \$RUN_DIR (default \$SCRATCH/empathy_runs/$(basename "$GRID" .txt))"
