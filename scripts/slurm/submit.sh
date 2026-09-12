#!/bin/bash
# Submit a grid file as a SLURM job array with a concurrency cap.
#
#   scripts/slurm/submit.sh <grid file> [max concurrent jobs=50] [extra sbatch options...]
#
#   scripts/slurm/submit.sh grids/pd_report.txt 50
#   scripts/slurm/submit.sh grids/harvest.txt 8 --gres=gpu:1 --cpus-per-task=4 --mem=32GB --time=12:00:00
#
# Account, mail and paths come from scripts/slurm/cluster.env (override with environment variables).
# The job name is the grid file name, so logs are slurm_logs/<grid>_<array id>_<task>.out|err.
# Job arrays are capped at MaxArraySize (1000 on most Alliance clusters); larger grids are split
# into consecutive arrays automatically.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/cluster.env"
cd "$SCRIPT_DIR/../.."

GRID=${1:?usage: scripts/slurm/submit.sh <grid file> [max concurrent] [sbatch options...]}
CONCURRENT=${2:-50}
shift $(( $# >= 2 ? 2 : 1 ))
N=$(grep -c . "$GRID")
[ "$N" -gt 0 ] || { echo "empty grid $GRID"; exit 1; }
MAX_ARRAY=${MAX_ARRAY:-1000}
NAME=$(basename "$GRID" .txt)
mkdir -p slurm_logs

start=1
while [ "$start" -le "$N" ]; do
  end=$(( start + MAX_ARRAY - 1 )); [ "$end" -gt "$N" ] && end=$N
  set -x
  sbatch --account="$SLURM_ACCOUNT" --mail-user="$MAIL_USER" --mail-type="$MAIL_TYPE" --job-name="$NAME" \
    --array="${start}-${end}%${CONCURRENT}" "$@" "$SCRIPT_DIR/run_grid.sh" "$GRID"
  { set +x; } 2>/dev/null
  start=$(( end + 1 ))
done
echo "submitted $N jobs from $GRID -> runs in ${RUN_DIR:-$RUN_ROOT/$NAME}, logs in slurm_logs/${NAME}_*"
