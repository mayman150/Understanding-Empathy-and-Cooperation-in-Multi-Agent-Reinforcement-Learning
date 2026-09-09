#!/usr/bin/env bash
# Melting Pot sweep (Commons Harvest / Cleanup): formulations x alpha x seeds, feed-forward or recurrent.
#
#   scripts/sweep_meltingpot.sh                                    # commons_harvest__open, ff
#   SUBSTRATE=clean_up FORMULATIONS="none sia reward_ia" scripts/sweep_meltingpot.sh
#   EXTRA="--recurrent --num-envs 4 --num-minibatches 4" scripts/sweep_meltingpot.sh   # partial-obs study
#
# Runs are sequential by default (each one is GPU heavy); set PARALLEL>1 if the GPU has room.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}

SUBSTRATE=${SUBSTRATE:-commons_harvest__open}
RUN_DIR=${RUN_DIR:-runs/$SUBSTRATE}
SEEDS=${SEEDS:-"1 2 3"}
FORMULATIONS=${FORMULATIONS:-"none ei svo sia ia reward_ia"}
ALPHAS=${ALPHAS:-"0.01 0.1 0.3"}
BETA=${BETA:-0.05}
PHI=${PHI:-0.7854}
REWARD_IA_ALPHA=${REWARD_IA_ALPHA:-5}    # Hughes et al. 2018 disadvantageous-inequity coefficient
PARALLEL=${PARALLEL:-1}
COMMON=${COMMON:-"--env-id meltingpot:$SUBSTRATE --total-timesteps 5000000 --num-steps 512 --num-minibatches 4"}
EXTRA=${EXTRA:-""}

run() { echo "+ $*"; $PY train.py $COMMON $EXTRA --run-dir "$RUN_DIR" "$@"; }
jobs_running() { jobs -rp | wc -l | tr -d ' '; }
launch() {
  while [ "$(jobs_running)" -ge "$PARALLEL" ]; do sleep 5; done
  run "$@" > "$RUN_DIR.$(date +%s%N).log" 2>&1 &
}
mkdir -p "$RUN_DIR"

for seed in $SEEDS; do
  for f in $FORMULATIONS; do
    case "$f" in
      none)      launch --formulation none --seed "$seed" ;;
      reward_ia) launch --formulation reward_ia --alpha "$REWARD_IA_ALPHA" --beta "$BETA" --seed "$seed" ;;
      *) for a in $ALPHAS; do launch --formulation "$f" --alpha "$a" --beta "$BETA" --phi "$PHI" --seed "$seed"; done ;;
    esac
  done
done
wait
echo "done -> $RUN_DIR"
