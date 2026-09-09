#!/usr/bin/env bash
# Prisoner's Dilemma sanity-check sweep: formulations x signal (value vs reward) x alpha x seeds,
# plus mixed selfish/empathetic pairs.
#
#   scripts/sweep_pd.sh                                   # defaults below (2 players)
#   NUM_AGENTS=4 SEEDS="1 2 3 4 5" scripts/sweep_pd.sh    # 4-player PD
#   SIGNALS="value" FORMULATIONS="ei svo" ALPHAS="5 20 100" PARALLEL=8 scripts/sweep_pd.sh
#   MIXED="0,20 20,0" scripts/sweep_pd.sh                 # selfish vs empathetic and vice versa
#
# The value signal (X_i added to the advantage) needs alphas comparable to the advantage scale
# (~10-25 in this game), the reward signal (intrinsic reward) alphas comparable to the reward
# scale (~1); hence the two alpha grids.  Every run writes TensorBoard logs + agents.pt under
# $RUN_DIR; summarise with:  python scripts/summarize_runs.py $RUN_DIR
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}

NUM_AGENTS=${NUM_AGENTS:-2}
RUN_DIR=${RUN_DIR:-runs/pd_n$NUM_AGENTS}
SEEDS=${SEEDS:-"1 2 3"}
FORMULATIONS=${FORMULATIONS:-"none ei svo sia ia"}
SIGNALS=${SIGNALS:-"value reward"}
VALUE_ALPHAS=${VALUE_ALPHAS:-"5 20 100"}       # --signal value
REWARD_ALPHAS=${REWARD_ALPHAS:-"0.5 1 5"}      # --signal reward
ALPHAS=${ALPHAS:-""}                           # if set, overrides both grids
BETA=${BETA:-"same"}                           # ia: advantageous coefficient; "same" = equal to alpha
PHI=${PHI:-0.7854}                             # svo angle (pi/4)
MIXED=${MIXED:-""}                             # per-agent alpha lists, e.g. "0,20 20,0" (2 players)
PARALLEL=${PARALLEL:-2}
COMMON=${COMMON:-"--env-id pd --max-cycles 100 --num-envs 8 --num-steps 128 --num-minibatches 4 --total-timesteps 1000000 --no-cuda"}
EXTRA=${EXTRA:-""}                             # e.g. "--recurrent"

run() { echo "+ $*"; $PY train.py $COMMON $EXTRA --num-agents "$NUM_AGENTS" --run-dir "$RUN_DIR" "$@"; }
jobs_running() { jobs -rp | wc -l | tr -d ' '; }
launch() {
  while [ "$(jobs_running)" -ge "$PARALLEL" ]; do sleep 2; done
  run "$@" > /dev/null 2>&1 &
}
beta_for() { if [ "$BETA" = "same" ]; then echo "$1"; else echo "$BETA"; fi; }

for seed in $SEEDS; do
  for f in $FORMULATIONS; do
    if [ "$f" = "none" ]; then
      launch --formulation none --seed "$seed"
      continue
    fi
    for sig in $SIGNALS; do
      if [ -n "$ALPHAS" ]; then grid=$ALPHAS; elif [ "$sig" = "value" ]; then grid=$VALUE_ALPHAS; else grid=$REWARD_ALPHAS; fi
      for a in $grid; do
        launch --formulation "$f" --signal "$sig" --alpha "$a" --beta "$(beta_for "$a")" --phi "$PHI" --seed "$seed"
      done
      for m in $MIXED; do
        launch --formulation "$f" --signal "$sig" --alpha "$m" --beta "$m" --phi "$PHI" --seed "$seed"
      done
    done
  done
done
wait
echo "done -> $RUN_DIR"
