#!/usr/bin/env bash
# Experiment 2 - Clean Up (Hughes et al. 2018) in the small fully observable configuration of Yang et al. 2020 (LIO):
# 10x10 map, 3 agents, 50-step episodes, egocentric plane observations that contain the whole map.
#
# The Coin Game was the sanity check (immediate harm to one other agent).  Clean Up changes what it could not test:
# 3 agents, a public good (cleaning pays the cleaner nothing and raises everyone's apples later), so the consequences
# of an action for the others are delayed and diffuse.  Stage A (this script) tunes EI and IA:
#
#   plain PPO                                   baseline
#   EI  imagined / other  (+ --social-scale)    the main method: my critic on the other's observation as the value
#                                               function of the other's imagined return (Coin Game winner)
#   EI  imagined / none   (+ --social-scale)    ablation of `other` WITHOUT the critic: does the value function matter
#                                               when the benefit of cleaning arrives many steps later?
#   EI  imagined / shaped                       the same imagined rewards as an intrinsic reward (literature path)
#   EI  value (+ --social-scale)                the paper's original term alpha * V_i(o_j'), scaled
#   SVO value, phi = 0 (+ --social-scale)       own-value control: alpha * V_i(o_i'), nobody else's state
#   EI  reward                                  upper reference WITH access to the others' true rewards
#   IA  imagined / shaped                       inequity aversion on imagined smoothed rewards (Hughes et al. with
#                                               imagined rewards): envy-heavy (Hughes' 5 / 0.05), guilt-heavy, symmetric
#   IA  reward                                  Hughes et al. 2018 with true rewards (the published Clean Up method)
#   IA  value (+ --social-scale)                the paper's IA on values
#
# Every imagined configuration uses --imagined-warmup 50 (about 100k steps of plain PPO while the reward model learns;
# apples are rare at the start) and --reward-model-replay 2048 (own rewarded transitions are replayed so that an agent
# that stops eating does not forget what eating looks like).  One PPO setting per grid file (PPO_SETTINGS crosses).
#
# Usage (from the repository root, on the cluster):
#   scripts/experiments/cleanup.sh grids                 # write grids/cleanup_stageA[_lr..].txt (inspect, count jobs)
#   scripts/experiments/cleanup.sh tune                  # write + submit (3 seeds x 29 configurations = 87 jobs per PPO setting)
#   PPO_SETTINGS="2.5e-4:0.01 1e-3:0.01" scripts/experiments/cleanup.sh tune      # crossed with PPO settings
#   scripts/experiments/cleanup.sh baseline              # plain PPO + EI/IA with true rewards only (cheap PPO check first)
#   IA_PAIRS="" IA_VALUE_PAIRS="" scripts/experiments/cleanup.sh tune   # EI only (17 configurations x 3 seeds = 51 jobs)
#   scripts/experiments/cleanup.sh summary               # tables per folder + ranking per method + stage-B `final` lines
#   STEPS=3000000 TIME=06:00:00 FINAL_SEEDS=7-14 scripts/experiments/cleanup.sh final --formulation ei --signal imagined \
#       --imagined-critic other --social-scale --alpha 1 --imagined-warmup 50 --reward-model-replay 2048
#   scripts/experiments/cleanup.sh sweep "--imagined-lambda:0.95,0.99" "--reward-model-replay:0,2048" -- <configuration>
#
# Budget: about 260 steps/s for imagined/other and 730 for plain PPO on 2 laptop cores (8 envs x 256 steps, small CNN);
# 3M steps is therefore 1-3.5 h per job on the laptop and roughly twice that on the cluster.  TIME defaults to 10 h.
#
# Metrics to read (charts/<key>/<agent> from the environment's episode statistics, plus the usual ones):
#   collective_return   apples eaten per episode by all agents (the public good works iff this goes up)
#   equality            1 - Gini of the per-agent returns (EI may accept unequal role splits; IA should resist them)
#   clean_actions       cleaning beams fired per agent per episode; waste_cleaned: waste cells removed
#   waste_density       mean polluted fraction of the river over the episode (0.5 at the start, < 0.4 for apples to grow)
#   apple_prob          mean apple spawn probability per cell and step (0.3 = clean river)
#   imagined/corr       correlation of imagined and true rewards of the others, per agent (own-experience bias shows here)
set -euo pipefail
cd "$(dirname "$0")/../.."
export EMPATHY_REPO=$PWD
source scripts/slurm/cluster.env
if [ -z "${PY:-}" ] && [ -x "$VENV/bin/python" ]; then
  if command -v module >/dev/null 2>&1; then module load "$STDENV" "$PY_MODULE" 2>/dev/null || true; fi
  PY="$VENV/bin/python"
fi
PY=${PY:-python}

SEEDS=${SEEDS:-"1 2 3"}
FINAL_SEEDS=${FINAL_SEEDS:-"4-11"}
NUM_AGENTS=${NUM_AGENTS:-3}
MAX_CYCLES=${MAX_CYCLES:-50}
STEPS=${STEPS:-3000000}
TIME=${TIME:-10:00:00}   # imagined/other runs at ~260 steps/s on 2 laptop cores (3M steps = 3.2 h); cluster cores are slower
CONCURRENT=${CONCURRENT:-50}
PPO_SETTINGS=${PPO_SETTINGS:-"1e-3:0.01"}   # "lr:ent" pairs; one grid file / result folder per pair
TRAIN_ARGS=${TRAIN_ARGS:-"--num-envs 8 --num-steps 256 --num-minibatches 4"}
IMAGINED_ARGS=${IMAGINED_ARGS:-"--imagined-warmup 50 --reward-model-replay 2048"}
# alpha grids (relative weights where --social-scale is on; reward units for shaped / reward)
OTHER_ALPHAS=${OTHER_ALPHAS:-"0.5 1 2"}
NONE_ALPHAS=${NONE_ALPHAS:-"1 2"}
SHAPED_ALPHAS=${SHAPED_ALPHAS:-"0.03 0.1 0.3"}
VALUE_ALPHAS=${VALUE_ALPHAS:-"0.5 1 2"}
CONTROL_ALPHAS=${CONTROL_ALPHAS:-"1 2"}
REWARD_ALPHAS=${REWARD_ALPHAS:-"0.03 0.1 0.3"}
IA_PAIRS=${IA_PAIRS:-"5:0.05 1:0.05 0.05:5 0.05:1 1:1"}     # alpha(envy):beta(guilt); 5:0.05 = Hughes et al. 2018
IA_VALUE_PAIRS=${IA_VALUE_PAIRS:-"1:0.1 2:0.2"}
PPO_SEEDS=${PPO_SEEDS:-"4-6"}
TAG=${TAG:-cleanup}
LAST=${LAST:-20}
SUMMARY_TAGS="charts/collective_return charts/equality charts/waste_density/player_0 charts/apple_prob/player_0 charts/clean_actions/player_0 charts/clean_actions/player_1 charts/clean_actions/player_2"
COMMON="--env-id cleanup --num-agents $NUM_AGENTS --max-cycles $MAX_CYCLES --total-timesteps $STEPS"

GRIDS=()
# make_grids [baseline-only]: one grid file per PPO setting
make_grids() {
  mkdir -p grids
  GRIDS=()
  local only_baseline=${1:-}
  local n_settings; n_settings=$(echo $PPO_SETTINGS | wc -w)
  for setting in $PPO_SETTINGS; do
    local lr="${setting%%:*}" ent="${setting#*:}" sfx=""
    [ "$n_settings" -gt 1 ] && sfx="_lr${lr}_ent${ent}"
    local extra="$TRAIN_ARGS --learning-rate $lr --ent-coef $ent"
    local g="grids/${TAG}_$([ -n "$only_baseline" ] && echo baseline || echo stageA)${sfx}.txt"
    : > "$g"
    local mg="$PY scripts/make_grid.py $COMMON --seeds $SEEDS"
    # plain PPO
    $mg --signals value --alphas 0 --formulations none --extra "$extra" >> "$g"
    # references with access to the true rewards
    $mg --signals reward --alphas $REWARD_ALPHAS --formulations ei --extra "$extra" >> "$g"
    [ -n "$IA_PAIRS" ] && $mg --signals reward --formulations ia --ia-pairs $IA_PAIRS --extra "$extra" >> "$g"
    if [ -z "$only_baseline" ]; then
      # EI: imagined rewards, three ways
      $mg --signals imagined --alphas $OTHER_ALPHAS --formulations ei --extra "$extra $IMAGINED_ARGS --imagined-critic other --social-scale" >> "$g"
      $mg --signals imagined --alphas $NONE_ALPHAS --formulations ei --extra "$extra $IMAGINED_ARGS --imagined-critic none --social-scale" >> "$g"
      $mg --signals imagined --alphas $SHAPED_ALPHAS --formulations ei --extra "$extra $IMAGINED_ARGS --imagined-critic shaped" >> "$g"
      # the paper's value term (scaled) and its own-value control
      $mg --signals value --alphas $VALUE_ALPHAS --formulations ei --extra "$extra --social-scale" >> "$g"
      $mg --signals value --alphas $CONTROL_ALPHAS --formulations svo --phis 0 --extra "$extra --social-scale" >> "$g"
      # IA: imagined (shaped) and on values (IA_PAIRS="" IA_VALUE_PAIRS="" -> EI only)
      [ -n "$IA_PAIRS" ] && $mg --signals imagined --formulations ia --ia-pairs $IA_PAIRS --extra "$extra $IMAGINED_ARGS --imagined-critic shaped" >> "$g"
      [ -n "$IA_VALUE_PAIRS" ] && $mg --signals value --formulations ia --ia-pairs $IA_VALUE_PAIRS --extra "$extra --social-scale" >> "$g"
    fi
    GRIDS+=("$g")
    echo "$g: $(grep -c . "$g") jobs ($(grep -c -- '--seed 1$' "$g") configurations x $(echo $SEEDS | wc -w) seeds)"
  done
}

case "${1:-}" in
  grids)
    make_grids ;;
  tune)
    make_grids
    for g in "${GRIDS[@]}"; do scripts/slurm/submit.sh "$g" "$CONCURRENT" --time="$TIME"; done ;;
  baseline)
    make_grids baseline
    for g in "${GRIDS[@]}"; do scripts/slurm/submit.sh "$g" "$CONCURRENT" --time="$TIME"; done ;;
  final)
    shift
    [ $# -gt 0 ] || { echo "usage: $0 final <train.py args, e.g. --formulation ei --signal imagined --imagined-critic other --social-scale --alpha 1>"; exit 1; }
    NAME="${TAG}_final_$(echo "$*" | sed -e 's/--//g' -e 's/[ ,]/_/g')"
    mkdir -p slurm_logs
    set -x
    sbatch --account="$SLURM_ACCOUNT" --mail-user="$MAIL_USER" --mail-type="$MAIL_TYPE" --job-name="$NAME" \
      --time="$TIME" --array="$FINAL_SEEDS" scripts/slurm/run_seeds.sh $COMMON $TRAIN_ARGS "$@"
    { set +x; } 2>/dev/null
    echo "runs -> $RUN_ROOT/$NAME" ;;
  sweep)
    # one-at-a-time sensitivity of train.py flags around ONE configuration (PPO_SEEDS each):
    #   cleanup.sh sweep "--imagined-lambda:0.95,0.99" "--reward-model-replay:0,2048" -- <configuration args>
    shift
    SPECS=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do SPECS+=("$1"); shift; done
    [ "${1:-}" = "--" ] && shift
    [ ${#SPECS[@]} -gt 0 ] && [ $# -gt 0 ] || { echo "usage: $0 sweep \"--flag:v1,v2\" [...] -- <train.py args of one configuration>"; exit 1; }
    CONF="$(echo "$*" | sed -e 's/--//g' -e 's/[ ,]/_/g')"
    mkdir -p slurm_logs
    for spec in "${SPECS[@]}"; do
      flag="${spec%%:*}"; vals="${spec#*:}"
      for val in ${vals//,/ }; do
        NAME="${TAG}_sweep_${CONF}_$(echo "$flag" | sed 's/^--//')${val}"
        set -x
        sbatch --account="$SLURM_ACCOUNT" --mail-user="$MAIL_USER" --mail-type="$MAIL_TYPE" --job-name="$NAME" \
          --time="$TIME" --array="$PPO_SEEDS" scripts/slurm/run_seeds.sh $COMMON $TRAIN_ARGS "$@" "$flag" "$val"
        { set +x; } 2>/dev/null
      done
    done
    echo "runs -> $RUN_ROOT/${TAG}_sweep_${CONF}_*" ;;
  summary)
    for d in "$RUN_ROOT/${TAG}_baseline"* "$RUN_ROOT/${TAG}_stageA"*; do
      [ -d "$d" ] || continue
      echo "=== $(basename "$d")"
      $PY scripts/summarize_runs.py "$d" --last "$LAST" --csv "$(basename "$d").csv" --tags $SUMMARY_TAGS
    done
    DIRS=$(ls -d "$RUN_ROOT/${TAG}_baseline"* "$RUN_ROOT/${TAG}_stageA"* "$RUN_ROOT/${TAG}_sweep_"* 2>/dev/null || true)
    if [ -n "$DIRS" ]; then
      echo "=== best configuration per method (all folders), ranked by final collective return (mean - std over seeds)"
      # shellcheck disable=SC2086
      $PY scripts/best_configs.py $DIRS --last "$LAST" --top "${TOP:-4}" --final-time "$TIME" \
          --tags charts/collective_return charts/equality charts/clean_actions/player_0 charts/waste_density/player_0
    fi
    FINAL_DIRS=$(ls -d "$RUN_ROOT/${TAG}_final_"* 2>/dev/null || true)
    if [ -n "$FINAL_DIRS" ]; then
      echo "=== final (fresh seeds), one row per configuration"
      # shellcheck disable=SC2086
      $PY scripts/summarize_runs.py $FINAL_DIRS --last "$LAST" --csv "${TAG}_final.csv" --tags $SUMMARY_TAGS
    fi ;;
  *)
    sed -n '2,45p' "$0"; exit 1 ;;
esac
