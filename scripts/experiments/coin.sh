#!/usr/bin/env bash
# Experiment 1b - Coin Game (Lerer & Peysakhovich 2017): the Prisoner's Dilemma on a 3x3 grid.
#
# The PD turned out to be the wrong sanity check for the value signal: its state (last round's
# actions) carries no information about how well off either agent is, so V_i(o_i) = V_i(o_j) and every
# gap-based term vanishes.  In the Coin Game the state (positions, coin colour) does: the agent next to
# its own coin is genuinely better off.  Same two grids with the same functional forms (EI, SIA, SVO, IA):
#
#   grids/coin_value.txt    our idea:  --signal value   (own critic on the other's observation; no reward access)
#   grids/coin_reward.txt   baseline:  --signal reward  (intrinsic reward from the other's smoothed rewards)
#
# Stage 1 (tuning, 3 seeds, 1M steps each = about 5-10 min on 2 cluster CPUs; jobs are capped at 30 min):
#   scripts/experiments/coin.sh grids      # write the grid files + LaTeX tables only
#   scripts/experiments/coin.sh tune       # write the grids and submit both job arrays
#   scripts/experiments/coin.sh summary    # mean +/- std over seeds per configuration, both grids (+ CSV)
#
# Stage 2 (winners of stage 1 on 8 fresh seeds 4..11, PPO defaults; use 2M steps, the value signal was still
# improving at 1M):
#   STEPS=2000000 TIME=01:00:00 scripts/experiments/coin.sh final --formulation svo --signal value --alpha 30 --phi 0.523599
#   STEPS=2000000 TIME=01:00:00 scripts/experiments/coin.sh final --formulation svo --signal value --alpha 30 --phi 0   # own-value-only control
#   STEPS=2000000 TIME=01:00:00 scripts/experiments/coin.sh final --formulation ia  --signal value --alpha 10 --beta 1
#   STEPS=2000000 TIME=01:00:00 scripts/experiments/coin.sh final --formulation ei  --signal reward --alpha 0.1        # reference with reward access
#   STEPS=2000000 TIME=01:00:00 scripts/experiments/coin.sh final --formulation none                                   # plain PPO
#
# Stage 2b (PPO sensitivity of the same configurations: PPO_LRS x PPO_ENTS, PPO_SEEDS each; the baseline and the
# reward-signal reference get the same grid so the comparison stays fair):
#   scripts/experiments/coin.sh ppo --formulation svo --signal value --alpha 30 --phi 0.523599
#   scripts/experiments/coin.sh ppo --formulation none
#   PPO_LRS="1e-4 1e-3" PPO_ENTS="0.01 0.05 0.1" scripts/experiments/coin.sh ppo --formulation ia --signal value --alpha 10 --beta 1
#   scripts/experiments/coin.sh summary    # also prints the final table and one PPO table per configuration
#
# Alpha grids (from local 1M-step probes, seed 1).  Value signal: the two families need different scales.  Gap
# terms (SIA, IA) see |V_i(o_i) - V_i(o_j)| ~ 0.8-1 against an advantage of ~1, and SIA only moved behaviour at
# alpha >= 3 (alpha 10: cooperation 0.6-0.7, collective return 8 vs 0 for plain PPO); level terms (EI, SVO) see
# V itself, which is ~0 once both agents grab everything, so alpha 0.03-0.3 was indistinguishable from plain PPO
# while alpha 3 made the term 50x the advantage and broke learning.  The shared grid therefore spans 0.03..100.
# Reward signal: the term competes with the smoothed reward e ~ r / (1 - gamma*lambda); EI at alpha 0.1 already
# reaches full cooperation, SIA at 0.1 does nothing, so it keeps the report's range plus 1 and 3.
# Override anything through the environment, e.g.  SEEDS="1 2 3 4 5" scripts/experiments/coin.sh tune
#   FORMULATIONS="ei sia" scripts/experiments/coin.sh tune           # plain PPO + EI + SIA only
#   TAG=coin_ent0.05 TRAIN_ARGS="--num-envs 8 --num-steps 128 --num-minibatches 4 --ent-coef 0.05" \
#       scripts/experiments/coin.sh tune                             # PPO sensitivity check under its own tag
#
# Metrics to read: charts/cooperation_rate/player_i = fraction of the coins agent i picked up that had its own
# colour (independent PPO: about 0.5 = grabs everything; full cooperation: 1.0), charts/collective_return
# (about 0 when both grab everything, +1 per coin under cooperation) and charts/equality.
set -euo pipefail
cd "$(dirname "$0")/../.."
export EMPATHY_REPO=$PWD   # lets the job scripts find cluster.env (they run from /var/spool/slurmd)
source scripts/slurm/cluster.env
# use the cluster virtualenv (tensorboard, numpy, ...) when it exists, so `summary` works from a login shell
if [ -z "${PY:-}" ] && [ -x "$VENV/bin/python" ]; then
  if command -v module >/dev/null 2>&1; then module load "$STDENV" "$PY_MODULE" 2>/dev/null || true; fi
  PY="$VENV/bin/python"
fi
PY=${PY:-python}

SEEDS=${SEEDS:-"1 2 3"}
FINAL_SEEDS=${FINAL_SEEDS:-"4-11"}
MAX_CYCLES=${MAX_CYCLES:-50}
STEPS=${STEPS:-1000000}
TIME=${TIME:-00:30:00}   # SLURM time limit per job
CONCURRENT=${CONCURRENT:-50}
FORMULATIONS=${FORMULATIONS:-"ei sia svo ia"}   # social-term formulations to tune (plain PPO is always included)
VALUE_ALPHAS=${VALUE_ALPHAS:-"0 0.03 0.1 0.3 1 3 10 30 100"}
REWARD_ALPHAS=${REWARD_ALPHAS:-"0 0.003 0.01 0.03 0.1 0.3 1 3"}
TRAIN_ARGS=${TRAIN_ARGS:-"--num-envs 8 --num-steps 128 --num-minibatches 4"}
PPO_LRS=${PPO_LRS:-"1e-4 2.5e-4 1e-3"}   # `ppo` subcommand: learning rates x entropy coefficients x PPO_SEEDS
PPO_ENTS=${PPO_ENTS:-"0.01 0.05"}
PPO_SEEDS=${PPO_SEEDS:-"4-6"}

COIN_ARGS="--env-id coin --max-cycles $MAX_CYCLES --total-timesteps $STEPS"
TAG=${TAG:-coin}   # names of the grid files / result folders
VALUE_GRID="grids/${TAG}_value.txt"
REWARD_GRID="grids/${TAG}_reward.txt"
SUMMARY_TAGS="charts/collective_return charts/equality charts/cooperation_rate/player_0 charts/cooperation_rate/player_1"
LAST=${LAST:-20}   # summary: mean of the last K logged points; episode metrics are logged once per batch of 8 finished episodes (20 = 8k steps)

make_grids() {
  mkdir -p grids
  # the plain-PPO baseline (alpha = 0) is the same run for both signals: emitted once, in the value grid
  $PY scripts/make_grid.py $COIN_ARGS --signals value --alphas $VALUE_ALPHAS --seeds $SEEDS \
      --formulations none $FORMULATIONS --extra "$TRAIN_ARGS" -o "$VALUE_GRID"
  $PY scripts/make_grid.py $COIN_ARGS --signals reward --alphas $REWARD_ALPHAS --seeds $SEEDS \
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
    scripts/slurm/submit.sh "$VALUE_GRID" "$CONCURRENT" --time="$TIME"
    scripts/slurm/submit.sh "$REWARD_GRID" "$CONCURRENT" --time="$TIME" ;;
  final)
    shift
    [ $# -gt 0 ] || { echo "usage: $0 final <train.py args, e.g. --formulation ei --signal value --alpha 3>"; exit 1; }
    NAME="${TAG}_final_$(echo "$*" | sed -e 's/--//g' -e 's/[ ,]/_/g')"
    mkdir -p slurm_logs
    set -x
    sbatch --account="$SLURM_ACCOUNT" --mail-user="$MAIL_USER" --mail-type="$MAIL_TYPE" --job-name="$NAME" \
      --time="$TIME" --array="$FINAL_SEEDS" scripts/slurm/run_seeds.sh $COIN_ARGS $TRAIN_ARGS "$@"
    { set +x; } 2>/dev/null
    echo "runs -> $RUN_ROOT/$NAME" ;;
  ppo)
    # PPO sensitivity of ONE configuration: one job array per (learning rate, entropy coefficient), PPO_SEEDS each.
    # Result folders: $RUN_ROOT/${TAG}_ppo_<configuration>_lr<lr>_ent<ent>; `summary` puts them in one table.
    shift
    [ $# -gt 0 ] || { echo "usage: $0 ppo <train.py args of one configuration, e.g. --formulation svo --signal value --alpha 30 --phi 0.523599>"; exit 1; }
    CONF="$(echo "$*" | sed -e 's/--//g' -e 's/[ ,]/_/g')"
    mkdir -p slurm_logs
    for lr in $PPO_LRS; do
      for ent in $PPO_ENTS; do
        NAME="${TAG}_ppo_${CONF}_lr${lr}_ent${ent}"
        set -x
        sbatch --account="$SLURM_ACCOUNT" --mail-user="$MAIL_USER" --mail-type="$MAIL_TYPE" --job-name="$NAME" \
          --time="$TIME" --array="$PPO_SEEDS" scripts/slurm/run_seeds.sh $COIN_ARGS $TRAIN_ARGS \
          --learning-rate "$lr" --ent-coef "$ent" "$@"
        { set +x; } 2>/dev/null
      done
    done
    echo "runs -> $RUN_ROOT/${TAG}_ppo_${CONF}_lr*_ent*" ;;
  summary)
    for g in value reward; do
      echo "=== $g signal ($RUN_ROOT/${TAG}_$g)"
      $PY scripts/summarize_runs.py "$RUN_ROOT/${TAG}_$g" --last "$LAST" --csv "${TAG}_${g}.csv" --tags $SUMMARY_TAGS
    done
    FINAL_DIRS=$(ls -d "$RUN_ROOT/${TAG}_final_"* 2>/dev/null || true)
    if [ -n "$FINAL_DIRS" ]; then
      echo "=== final (fresh seeds), one row per configuration"
      # shellcheck disable=SC2086
      $PY scripts/summarize_runs.py $FINAL_DIRS --last "$LAST" --csv "${TAG}_final.csv" --tags $SUMMARY_TAGS
    fi
    # PPO sensitivity: one table per configuration, rows = (lr, ent) settings
    ls -d "$RUN_ROOT/${TAG}_ppo_"*_lr*_ent* 2>/dev/null | sed -E 's/_lr[^_]+_ent[^_]+$//' | sort -u | while read -r conf; do
      echo "=== PPO sensitivity: $(basename "$conf" | sed "s/^${TAG}_ppo_//")"
      # shellcheck disable=SC2086
      $PY scripts/summarize_runs.py "$conf"_lr*_ent* --last "$LAST" --tags $SUMMARY_TAGS
    done
    # ... and the ranking per method across all of them, with the stage-B command for every winner
    PPO_DIRS=$(ls -d "$RUN_ROOT/${TAG}_ppo_"* 2>/dev/null || true)
    if [ -n "$PPO_DIRS" ]; then
      echo "=== best (alpha, PPO setting) per method across the PPO grid"
      # shellcheck disable=SC2086
      $PY scripts/best_configs.py $PPO_DIRS --last "$LAST" --top "${TOP:-5}"
    fi ;;
  *)
    sed -n '2,30p' "$0"; exit 1 ;;
esac
