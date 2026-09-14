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
#   IA  imagined / other  (+ --social-scale)    inequity aversion through the advantage construction: the Fehr-Schmidt
#                                               derivative at the current levels weighs my own and the others' imagined
#                                               advantages (guilt-only rows 0:1, 0:0.5 are the ones with a mechanism)
#   SIA imagined / other  (+ --social-scale)    the symmetric control
#   IA  imagined / shaped                       inequity aversion on imagined smoothed rewards (Hughes et al. with
#                                               imagined rewards): envy-heavy (Hughes' 5 / 0.05), guilt-heavy, symmetric
#   IA  reward                                  Hughes et al. 2018 with true rewards (the published Clean Up method)
#   IA  value (+ --social-scale)                the paper's IA on values
#
# Every imagined configuration uses a warm-up of about WARMUP_STEPS (100k) environment steps of plain PPO while the reward
# model learns (apples are rare at the start; the number of iterations is derived from the rollout size) and
# --reward-model-replay 2048 (own rewarded transitions are replayed so that an agent that stops eating does not forget
# what eating looks like).  One PPO setting per grid file (PPO_SETTINGS crosses).
#
# Usage (from the repository root, on the cluster).  Protocol, one stage at a time:
#   A1  PPO tuning:  learning rate x entropy x rollout length (PPO_LRS x PPO_ENTS x PPO_ROLLOUTS, 12 settings) on a few
#       anchor configurations (PPO_ANCHORS: plain PPO, EI imagined/other alpha 1, EI value alpha 1), PPO_SEEDS each:
#         scripts/experiments/cleanup.sh ppo                 # 12 settings x 3 anchors x 3 seeds = 108 jobs, one folder per setting
#       -> `summary` ranks every anchor over the settings; pick ONE setting for everything that follows (fair comparison),
#          and put it into TRAIN_ARGS, e.g.  export TRAIN_ARGS="--num-envs 8 --num-steps 512 --num-minibatches 4"
#                                             PPO_SETTINGS="1e-3:0.01"
#   A2  method hyper-parameters at that setting (alpha grids of all EI modes, the phi = 0 control, the true-reward reference;
#       IA rows unless IA_PAIRS="" IA_VALUE_PAIRS=""):
#         METHODS=ei scripts/experiments/cleanup.sh tune                          # EI only: 17 configurations x 3 seeds = 51 jobs
#         METHODS=ia TAG=cleanup_ia scripts/experiments/cleanup.sh tune           # IA / SIA only (own folder): 26 x 3 = 78 jobs
#         METHODS=svo TAG=cleanup_svo scripts/experiments/cleanup.sh tune         # SVO only: alpha x angle in every path, 17 x 3 = 51 jobs
#         scripts/experiments/cleanup.sh tune                                    # EI + IA: 42 configurations x 3 seeds = 126 jobs
#         LEVEL_VALUE=1 scripts/experiments/cleanup.sh tune                      # + the shaped rows on the "trace + value" level
#                                                                                #   (EI, IA and SVO groups alike)
#       IA can also join the PPO stage as an extra anchor (same folders as the running one):
#         PPO_ANCHORS="--formulation ia --signal imagined --imagined-critic other --social-scale --alpha 0 --beta 1" \
#             scripts/experiments/cleanup.sh ppo                                 # 12 settings x 3 seeds = 36 jobs
#       and so can the shaped + "trace + value" level of the three formulations (reward-slot weights, hence small):
#         PPO_ANCHORS="--formulation ei --signal imagined --imagined-critic shaped --imagined-level trace_value --alpha 0.1|\
#           --formulation ia --signal imagined --imagined-critic shaped --imagined-level trace_value --alpha 0.03 --beta 0.3|\
#           --formulation svo --signal imagined --imagined-critic shaped --imagined-level trace_value --alpha 0.1 --phi 0.785398" \
#             scripts/experiments/cleanup.sh ppo                                 # 12 settings x 3 anchors x 3 seeds = 108 jobs
#       then the imagined-specific knobs around the winner:
#         scripts/experiments/cleanup.sh sweep "--imagined-lambda:0.95,0.99" "--reward-model-replay:0,2048" -- <configuration>
#   C   winners on fresh seeds:  FINAL_SEEDS=7-14 scripts/experiments/cleanup.sh final <configuration>   (lines printed by `summary`)
#   Other subcommands:
#   scripts/experiments/cleanup.sh grids                 # write the A2 grid file(s) only (inspect, count jobs)
#   scripts/experiments/cleanup.sh baseline              # plain PPO + EI/IA with true rewards only (cheap check)
#   scripts/experiments/cleanup.sh summary               # tables per folder + ranking per method + `final` lines for the winners
#   STEPS=3000000 TIME=06:00:00 FINAL_SEEDS=7-14 scripts/experiments/cleanup.sh final --formulation ei --signal imagined \
#       --imagined-critic other --social-scale --alpha 1 --imagined-warmup 50 --reward-model-replay 2048
#   scripts/experiments/cleanup.sh sweep "--imagined-lambda:0.95,0.99" "--reward-model-replay:0,2048" -- <configuration>
#
# Budget: about 260 steps/s for imagined/other and 730 for plain PPO on 2 laptop cores (8 envs x 256 steps, small CNN),
# roughly half that on cluster cores: 5M steps is 4 h (plain PPO) to 11 h (imagined/other) per job.  TIME defaults to 14 h.
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
STEPS=${STEPS:-5000000}
TIME=${TIME:-14:00:00}   # imagined/other runs at ~260 steps/s on 2 laptop cores (5M steps = 5.3 h); cluster cores are slower
CONCURRENT=${CONCURRENT:-50}
SBATCH_EXTRA=${SBATCH_EXTRA:-}   # extra sbatch options for every submission, e.g. "--gres=gpu:1 --cpus-per-task=4 --mem=16GB" (default: 2 CPUs, no GPU)
PPO_SETTINGS=${PPO_SETTINGS:-"1e-3:0.01"}   # "lr:ent" pairs; one grid file / result folder per pair
TRAIN_ARGS=${TRAIN_ARGS:-"--num-envs 8 --num-steps 256 --num-minibatches 4"}
WARMUP_STEPS=${WARMUP_STEPS:-102400}   # imagined: plain-PPO warm-up while the reward model learns, in environment steps
REPLAY=${REPLAY:-2048}                 # imagined: --reward-model-replay capacity (0 = off)
# stage A1 (`ppo`): PPO grid on a few anchor configurations
PPO_LRS=${PPO_LRS:-"2.5e-4 1e-3"}
PPO_ENTS=${PPO_ENTS:-"0.01 0.05"}
PPO_ROLLOUTS=${PPO_ROLLOUTS:-"128 256 512"}   # --num-steps per env copy (8 envs: batches of 1024 / 2048 / 4096)
PPO_ANCHORS=${PPO_ANCHORS-"--formulation none|--formulation ei --signal imagined --imagined-critic other --social-scale --alpha 1|--formulation ei --signal value --social-scale --alpha 1"}
PPO_SEEDS=${PPO_SEEDS:-"1-3"}
# alpha grids (relative weights where --social-scale is on; reward units for shaped / reward)
OTHER_ALPHAS=${OTHER_ALPHAS:-"0.5 1 2"}
NONE_ALPHAS=${NONE_ALPHAS:-"1 2"}
SHAPED_ALPHAS=${SHAPED_ALPHAS:-"0.03 0.1 0.3"}
VALUE_ALPHAS=${VALUE_ALPHAS:-"0.5 1 2"}
CONTROL_ALPHAS=${CONTROL_ALPHAS:-"1 2"}
REWARD_ALPHAS=${REWARD_ALPHAS:-"0.03 0.1 0.3"}
IA_PAIRS=${IA_PAIRS-"5:0.05 1:0.05 0.05:5 0.05:1 1:1 0:1 0:0.5"}   # alpha(envy):beta(guilt); 5:0.05 = Hughes et al. 2018; 0:x = guilt only
SIA_ALPHAS=${SIA_ALPHAS-"0.5 1"}                                 # symmetric inequity aversion, the control row
IA_VALUE_PAIRS=${IA_VALUE_PAIRS-"1:0.1 2:0.2"}
# LEVEL_VALUE=1 adds the `shaped + value level` rows (--imagined-level trace_value: the formulation is applied to the
# smoothed imagined rewards PLUS my critic's forecast for the other, "what you earned lately + what you are about to earn")
LEVEL_VALUE=${LEVEL_VALUE:-0}
METHODS=${METHODS:-"ei ia"}   # which method groups the A2 grid contains (plain PPO is always in): any of "ei", "ia", "svo"
                              # (not GROUPS: that is bash's own read-only array of the user's group ids on Linux)
SVO_ALPHAS=${SVO_ALPHAS:-"1 2 3"}        # svo group: --imagined-critic other / value, crossed with the angles below
SVO_PHIS=${SVO_PHIS:-"pi/4 pi/3"}        # pi/2 is EI, 0 is the own-value control (already in the ei group)
SVO_SHAPED_ALPHAS=${SVO_SHAPED_ALPHAS:-"0.1"}
TAG=${TAG:-cleanup}
LAST=${LAST:-20}
LAST_STEPS=${LAST_STEPS:-20000}   # summary / ranking window: mean over the last 20k environment steps (about 50 logged points here)
SUMMARY_TAGS="charts/collective_return charts/equality charts/waste_density/player_0 charts/apple_prob/player_0 charts/clean_actions/player_0 charts/clean_actions/player_1 charts/clean_actions/player_2"
COMMON="--env-id cleanup --num-agents $NUM_AGENTS --max-cycles $MAX_CYCLES --total-timesteps $STEPS"

# imagined_args "<train args>": --imagined-warmup as iterations covering WARMUP_STEPS at that rollout size, plus the replay
imagined_args() {
  local envs steps
  envs=$(echo "$1" | sed -n 's/.*--num-envs \([0-9]*\).*/\1/p'); steps=$(echo "$1" | sed -n 's/.*--num-steps \([0-9]*\).*/\1/p')
  envs=${envs:-1}; steps=${steps:-512}
  echo "--imagined-warmup $(( (WARMUP_STEPS + envs * steps - 1) / (envs * steps) )) --reward-model-replay $REPLAY"
}

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
    local IMAGINED_ARGS; IMAGINED_ARGS=$(imagined_args "$extra")
    local g="grids/${TAG}_$([ -n "$only_baseline" ] && echo baseline || echo stageA)${sfx}.txt"
    : > "$g"
    local mg="$PY scripts/make_grid.py $COMMON --seeds $SEEDS"
    local want_ei=0 want_ia=0 want_svo=0
    [[ " $METHODS " == *" ei "* ]] && want_ei=1
    [[ " $METHODS " == *" ia "* ]] && [ -n "$IA_PAIRS" ] && want_ia=1
    [[ " $METHODS " == *" svo "* ]] && want_svo=1
    # plain PPO
    $mg --signals value --alphas 0 --formulations none --extra "$extra" >> "$g"
    # references with access to the true rewards
    [ $want_ei = 1 ] && $mg --signals reward --alphas $REWARD_ALPHAS --formulations ei --extra "$extra" >> "$g"
    [ $want_ia = 1 ] && $mg --signals reward --formulations ia --ia-pairs $IA_PAIRS --extra "$extra" >> "$g"
    if [ -z "$only_baseline" ] && [ $want_ei = 1 ]; then
      # EI: imagined rewards, three ways
      $mg --signals imagined --alphas $OTHER_ALPHAS --formulations ei --extra "$extra $IMAGINED_ARGS --imagined-critic other --social-scale" >> "$g"
      $mg --signals imagined --alphas $NONE_ALPHAS --formulations ei --extra "$extra $IMAGINED_ARGS --imagined-critic none --social-scale" >> "$g"
      $mg --signals imagined --alphas $SHAPED_ALPHAS --formulations ei --extra "$extra $IMAGINED_ARGS --imagined-critic shaped" >> "$g"
      [ "$LEVEL_VALUE" = 1 ] && $mg --signals imagined --alphas $SHAPED_ALPHAS --formulations ei --extra "$extra $IMAGINED_ARGS --imagined-critic shaped --imagined-level trace_value" >> "$g"
      # the paper's value term (scaled) and its own-value control
      $mg --signals value --alphas $VALUE_ALPHAS --formulations ei --extra "$extra --social-scale" >> "$g"
      $mg --signals value --alphas $CONTROL_ALPHAS --formulations svo --phis 0 --extra "$extra --social-scale" >> "$g"
    fi
    if [ -z "$only_baseline" ] && [ $want_svo = 1 ]; then
      # SVO: the same linear family as EI with the weight split by the angle; tuned over alpha x phi in every path
      $mg --signals imagined --alphas $SVO_ALPHAS --formulations svo --phis $SVO_PHIS --extra "$extra $IMAGINED_ARGS --imagined-critic other --social-scale" >> "$g"
      $mg --signals value --alphas $SVO_ALPHAS --formulations svo --phis $SVO_PHIS --extra "$extra --social-scale" >> "$g"
      $mg --signals imagined --alphas $SVO_SHAPED_ALPHAS --formulations svo --phis $SVO_PHIS --extra "$extra $IMAGINED_ARGS --imagined-critic shaped" >> "$g"
      [ "$LEVEL_VALUE" = 1 ] && $mg --signals imagined --alphas $SVO_SHAPED_ALPHAS --formulations svo --phis $SVO_PHIS --extra "$extra $IMAGINED_ARGS --imagined-critic shaped --imagined-level trace_value" >> "$g"
      $mg --signals reward --alphas $SVO_SHAPED_ALPHAS --formulations svo --phis $SVO_PHIS --extra "$extra" >> "$g"
    fi
    if [ -z "$only_baseline" ] && [ $want_ia = 1 ]; then
      # IA / SIA through the advantage construction (inequity-dependent weights on the imagined advantages, my critic as
      # the others' value function); SIA is the symmetric control
      $mg --signals imagined --formulations ia --ia-pairs $IA_PAIRS --extra "$extra $IMAGINED_ARGS --imagined-critic other --social-scale" >> "$g"
      $mg --signals imagined --alphas $SIA_ALPHAS --formulations sia --extra "$extra $IMAGINED_ARGS --imagined-critic other --social-scale" >> "$g"
      # IA: imagined (shaped, Hughes et al. with imagined rewards) and on values (the report's IA)
      $mg --signals imagined --formulations ia --ia-pairs $IA_PAIRS --extra "$extra $IMAGINED_ARGS --imagined-critic shaped" >> "$g"
      [ "$LEVEL_VALUE" = 1 ] && $mg --signals imagined --formulations ia --ia-pairs $IA_PAIRS --extra "$extra $IMAGINED_ARGS --imagined-critic shaped --imagined-level trace_value" >> "$g"
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
    for g in "${GRIDS[@]}"; do scripts/slurm/submit.sh "$g" "$CONCURRENT" --time="$TIME" $SBATCH_EXTRA; done ;;
  baseline)
    make_grids baseline
    for g in "${GRIDS[@]}"; do scripts/slurm/submit.sh "$g" "$CONCURRENT" --time="$TIME" $SBATCH_EXTRA; done ;;
  ppo)
    # stage A1: PPO_LRS x PPO_ENTS x PPO_ROLLOUTS, each on every anchor configuration (PPO_ANCHORS, '|'-separated) and
    # PPO_SEEDS; one grid file / result folder per PPO setting: $RUN_ROOT/${TAG}_ppo_lr<lr>_ent<ent>_T<rollout>
    mkdir -p grids
    IFS='|' read -r -a anchors <<< "$PPO_ANCHORS"
    seeds_list=$(seq "${PPO_SEEDS%-*}" "${PPO_SEEDS#*-}" | tr '\n' ' ')
    total=0
    for T in $PPO_ROLLOUTS; do
      for lr in $PPO_LRS; do
        for ent in $PPO_ENTS; do
          g="grids/${TAG}_ppo_lr${lr}_ent${ent}_T${T}.txt"
          extra="--num-envs 8 --num-steps $T --num-minibatches 4 --learning-rate $lr --ent-coef $ent"
          im=$(imagined_args "$extra")
          : > "$g"
          for conf in "${anchors[@]}"; do
            line_extra="$extra"
            [[ "$conf" == *"--signal imagined"* ]] && line_extra="$extra $im"
            for seed in $seeds_list; do echo "$COMMON $line_extra $conf --seed $seed" >> "$g"; done
          done
          n=$(grep -c . "$g"); total=$(( total + n ))
          echo "$g: $n jobs"
          [ "${DRY_RUN:-}" = 1 ] || scripts/slurm/submit.sh "$g" "$CONCURRENT" --time="$TIME" $SBATCH_EXTRA
        done
      done
    done
    echo "stage A1: $total jobs -> $RUN_ROOT/${TAG}_ppo_*" ;;
  final)
    shift
    [ $# -gt 0 ] || { echo "usage: $0 final <train.py args, e.g. --formulation ei --signal imagined --imagined-critic other --social-scale --alpha 1>"; exit 1; }
    NAME="${TAG}_final_$(echo "$*" | sed -e 's/--//g' -e 's/[ ,]/_/g')"
    mkdir -p slurm_logs
    set -x
    sbatch --account="$SLURM_ACCOUNT" --mail-user="$MAIL_USER" --mail-type="$MAIL_TYPE" --job-name="$NAME" \
      --time="$TIME" --array="$FINAL_SEEDS" $SBATCH_EXTRA scripts/slurm/run_seeds.sh $COMMON $TRAIN_ARGS "$@"
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
          --time="$TIME" --array="$PPO_SEEDS" $SBATCH_EXTRA scripts/slurm/run_seeds.sh $COMMON $TRAIN_ARGS "$@" "$flag" "$val"
        { set +x; } 2>/dev/null
      done
    done
    echo "runs -> $RUN_ROOT/${TAG}_sweep_${CONF}_*" ;;
  summary)
    for d in "$RUN_ROOT/${TAG}_baseline"* "$RUN_ROOT/${TAG}_stageA"*; do
      [ -d "$d" ] || continue
      echo "=== $(basename "$d")"
      $PY scripts/summarize_runs.py "$d" --last "$LAST" --last-steps "$LAST_STEPS" --csv "$(basename "$d").csv" --tags $SUMMARY_TAGS
    done
    PPO_DIRS=$(ls -d "$RUN_ROOT/${TAG}_ppo_"* 2>/dev/null || true)
    if [ -n "$PPO_DIRS" ]; then
      echo "=== stage A1: PPO settings (rows = lr / ent / rollout folders) per anchor configuration"
      # shellcheck disable=SC2086
      $PY scripts/summarize_runs.py $PPO_DIRS --last "$LAST" --last-steps "$LAST_STEPS" --csv "${TAG}_ppo.csv" --tags $SUMMARY_TAGS
      echo "=== stage A1: best PPO setting per anchor (mean - std of the final collective return)"
      # shellcheck disable=SC2086
      $PY scripts/best_configs.py $PPO_DIRS --last "$LAST" --last-steps "$LAST_STEPS" --top "${TOP:-6}" --final-time "$TIME" \
          --tags charts/collective_return charts/equality charts/clean_actions/player_0 charts/waste_density/player_0
    fi
    DIRS=$(ls -d "$RUN_ROOT/${TAG}_baseline"* "$RUN_ROOT/${TAG}_stageA"* "$RUN_ROOT/${TAG}_sweep_"* 2>/dev/null || true)
    if [ -n "$DIRS" ]; then
      echo "=== best configuration per method (all folders), ranked by final collective return (mean - std over seeds)"
      # shellcheck disable=SC2086
      $PY scripts/best_configs.py $DIRS --last "$LAST" --last-steps "$LAST_STEPS" --top "${TOP:-4}" --final-time "$TIME" \
          --tags charts/collective_return charts/equality charts/clean_actions/player_0 charts/waste_density/player_0
    fi
    FINAL_DIRS=$(ls -d "$RUN_ROOT/${TAG}_final_"* 2>/dev/null || true)
    if [ -n "$FINAL_DIRS" ]; then
      echo "=== final (fresh seeds), one row per configuration"
      # shellcheck disable=SC2086
      $PY scripts/summarize_runs.py $FINAL_DIRS --last "$LAST" --last-steps "$LAST_STEPS" --csv "${TAG}_final.csv" --tags $SUMMARY_TAGS
    fi ;;
  *)
    sed -n '2,45p' "$0"; exit 1 ;;
esac
