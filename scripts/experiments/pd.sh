#!/usr/bin/env bash
# Experiment 1 - repeated Prisoner's Dilemma, two grids with the same functional forms (EI, SIA, SVO, IA):
#
#   grids/pd_value.txt    our idea:  --signal value   (own critic on the others' observations; no reward access)
#   grids/pd_reward.txt   baseline:  --signal reward  (intrinsic reward from the others' smoothed rewards)
#
# Stage 1 (tuning, 2 seeds, 5M steps each):
#   scripts/experiments/pd.sh grids      # write the grid files + LaTeX tables only
#   scripts/experiments/pd.sh tune       # write the grids and submit both job arrays
#   scripts/experiments/pd.sh summary    # mean +/- std over seeds per configuration, both grids (+ CSV)
#
# Stage 2 (best configuration of each signal, 8 fresh seeds 3..10):
#   scripts/experiments/pd.sh final --formulation ei  --signal value  --alpha 20
#   scripts/experiments/pd.sh final --formulation ia  --signal reward --alpha 0.1 --beta 0.05
#
# Alpha grids: the value signal competes with the advantage (order 10-25 in this game), so the
# report's alphas are extended upwards; the reward signal competes with the smoothed reward
# (e ~ r / (1 - gamma*lambda) ~ 30-100 here), so it keeps the report's range plus 1 and 3.
# Override anything through the environment, e.g.  SEEDS="1 2 3" NUM_AGENTS=4 scripts/experiments/pd.sh tune
#   FORMULATIONS="sia" scripts/experiments/pd.sh tune     # plain PPO + SIA (value) + SIA (reward) only: 36 jobs
#
# PPO is held fixed across the grid (CleanRL defaults: lr 2.5e-4, clip 0.2, ent 0.01, gamma 0.99, gae 0.95,
# 8 envs x 128 steps, 4 minibatches x 4 epochs) so that differences are attributable to the social term.
# For a PPO sensitivity check of a chosen configuration give those runs their own TAG, e.g.
#   TAG=pd_n2_ent0.05 TRAIN_ARGS="--num-envs 8 --num-steps 128 --num-minibatches 4 --ent-coef 0.05" \
#       scripts/experiments/pd.sh final --formulation sia --signal value --alpha 20
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/slurm/cluster.env
PY=${PY:-python}

SEEDS=${SEEDS:-"1 2"}
FINAL_SEEDS=${FINAL_SEEDS:-"3-10"}
NUM_AGENTS=${NUM_AGENTS:-2}
MAX_CYCLES=${MAX_CYCLES:-100}
STEPS=${STEPS:-5000000}
CONCURRENT=${CONCURRENT:-50}
FORMULATIONS=${FORMULATIONS:-"ei sia svo ia"}   # social-term formulations to tune (plain PPO is always included)
VALUE_ALPHAS=${VALUE_ALPHAS:-"0 0.003 0.01 0.03 0.1 0.3 1 3 10 30 100"}
REWARD_ALPHAS=${REWARD_ALPHAS:-"0 0.003 0.01 0.03 0.1 0.3 1 3"}
TRAIN_ARGS=${TRAIN_ARGS:-"--num-envs 8 --num-steps 128 --num-minibatches 4"}

PD_ARGS="--env-id pd --num-agents $NUM_AGENTS --max-cycles $MAX_CYCLES --total-timesteps $STEPS"
TAG=${TAG:-pd_n${NUM_AGENTS}}   # names of the grid files / result folders
VALUE_GRID="grids/${TAG}_value.txt"
REWARD_GRID="grids/${TAG}_reward.txt"

make_grids() {
  mkdir -p grids
  # the plain-PPO baseline (alpha = 0) is the same run for both signals: emitted once, in the value grid
  $PY scripts/make_grid.py $PD_ARGS --signals value --alphas $VALUE_ALPHAS --seeds $SEEDS \
      --formulations none $FORMULATIONS --extra "$TRAIN_ARGS" -o "$VALUE_GRID"
  $PY scripts/make_grid.py $PD_ARGS --signals reward --alphas $REWARD_ALPHAS --seeds $SEEDS \
      --formulations $FORMULATIONS --extra "$TRAIN_ARGS" -o "$REWARD_GRID"
  $PY scripts/make_grid.py --signals value --alphas $VALUE_ALPHAS --seeds $SEEDS --formulations $FORMULATIONS --latex > "grids/${TAG}_value_table.tex"
  $PY scripts/make_grid.py --signals reward --alphas $REWARD_ALPHAS --seeds $SEEDS --formulations $FORMULATIONS --latex > "grids/${TAG}_reward_table.tex"
  echo "LaTeX tables: grids/${TAG}_value_table.tex grids/${TAG}_reward_table.tex"
}

case "${1:-}" in
  grids)
    make_grids ;;
  tune)
    make_grids
    scripts/slurm/submit.sh "$VALUE_GRID" "$CONCURRENT"
    scripts/slurm/submit.sh "$REWARD_GRID" "$CONCURRENT" ;;
  final)
    shift
    [ $# -gt 0 ] || { echo "usage: $0 final <train.py args, e.g. --formulation ei --signal value --alpha 20>"; exit 1; }
    NAME="${TAG}_final_$(echo "$*" | sed -e 's/--//g' -e 's/[ ,]/_/g')"
    mkdir -p slurm_logs
    set -x
    sbatch --account="$SLURM_ACCOUNT" --mail-user="$MAIL_USER" --mail-type="$MAIL_TYPE" --job-name="$NAME" \
      --array="$FINAL_SEEDS" scripts/slurm/run_seeds.sh $PD_ARGS $TRAIN_ARGS "$@"
    { set +x; } 2>/dev/null
    echo "runs -> $RUN_ROOT/$NAME" ;;
  summary)
    for g in value reward; do
      echo "=== $g signal ($RUN_ROOT/${TAG}_$g)"
      $PY scripts/summarize_runs.py "$RUN_ROOT/${TAG}_$g" --last 20 --csv "${TAG}_${g}.csv" \
        --tags charts/collective_return charts/equality charts/cooperation_rate/player_0 charts/cooperation_rate/player_1
    done
    ls -d "$RUN_ROOT/${TAG}_final_"* 2>/dev/null | while read -r d; do
      echo "=== $(basename "$d")"; $PY scripts/summarize_runs.py "$d" --last 20 \
        --tags charts/collective_return charts/equality charts/cooperation_rate/player_0 charts/cooperation_rate/player_1
    done ;;
  *)
    sed -n '2,20p' "$0"; exit 1 ;;
esac
