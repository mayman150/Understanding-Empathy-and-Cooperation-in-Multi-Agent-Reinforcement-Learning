#!/usr/bin/env bash
# Prisoner's Dilemma sanity-check sweep: formulations x alpha x seeds (+ mixed selfish/empathetic pairs).
#
#   scripts/sweep_pd.sh                      # defaults below
#   SEEDS="1 2 3 4 5" ALPHAS="0 0.1 0.5 1" PARALLEL=4 scripts/sweep_pd.sh
#   FORMULATIONS="sia" MIXED="0,0.5 0.5,0" scripts/sweep_pd.sh
#
# Every run writes TensorBoard logs + agents.pt under $RUN_DIR; summarise with
#   python scripts/summarize_runs.py $RUN_DIR
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}

RUN_DIR=${RUN_DIR:-runs/pd}
SEEDS=${SEEDS:-"1 2 3"}
FORMULATIONS=${FORMULATIONS:-"none ei svo sia ia reward_ia"}
ALPHAS=${ALPHAS:-"0.1 0.5 1.0"}
BETA=${BETA:-0.05}                 # ia / reward_ia advantageous-inequity coefficient
PHI=${PHI:-0.7854}                 # svo angle (pi/4)
MIXED=${MIXED:-"0,0.5 0.5,0"}      # per-agent alphas: selfish vs empathetic and vice versa
PARALLEL=${PARALLEL:-2}
COMMON=${COMMON:-"--env-id pd --max-cycles 100 --num-envs 8 --num-steps 128 --num-minibatches 4 --total-timesteps 1000000 --no-cuda"}
EXTRA=${EXTRA:-""}                 # e.g. "--recurrent"

run() { echo "+ $*"; $PY train.py $COMMON $EXTRA --run-dir "$RUN_DIR" "$@"; }

jobs_running() { jobs -rp | wc -l | tr -d ' '; }
launch() {
  while [ "$(jobs_running)" -ge "$PARALLEL" ]; do sleep 2; done
  run "$@" > /dev/null 2>&1 &
}

for seed in $SEEDS; do
  for f in $FORMULATIONS; do
    if [ "$f" = "none" ]; then
      launch --formulation none --seed "$seed"
      continue
    fi
    for a in $ALPHAS; do
      launch --formulation "$f" --alpha "$a" --beta "$BETA" --phi "$PHI" --seed "$seed"
    done
    for m in $MIXED; do
      launch --formulation "$f" --alpha "$m" --beta "$BETA" --phi "$PHI" --seed "$seed"
    done
  done
done
wait
echo "done -> $RUN_DIR"
