"""Experiment configuration (tyro dataclass, CleanRL style)."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional

from empathy_marl.empathy import FORMULATIONS, IMAGINED_CRITICS, SIGNALS

Formulation = Literal["none", "ei", "svo", "sia", "ia"]
Signal = Literal["value", "reward", "imagined"]
ImaginedCritic = Literal["other", "shaped", "none", "level"]
SocialAggregate = Literal["mean", "sum"]
ImaginedLevel = Literal["trace", "trace_value"]
IAWeighting = Literal["indicator", "relu"]


@dataclass
class Args:
    exp_name: str = field(default_factory=lambda: os.path.basename(sys.argv[0]).removesuffix(".py"))
    """name of this experiment (used in the run directory)"""
    seed: int = 1
    """seed of the experiment (python, numpy, torch and, where supported, the environment)"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """if toggled, cuda will be used when available"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "empathy-marl"
    wandb_entity: Optional[str] = None
    run_dir: str = "runs"
    """root directory for TensorBoard logs and checkpoints"""
    save_model: bool = True
    """save a checkpoint (`agents.pt`) at the end of training"""
    checkpoint_every: int = 0
    """additionally save a checkpoint every N iterations (0 = only at the end)"""

    # --- environment ---------------------------------------------------------------------
    env_id: str = "meltingpot:commons_harvest__open"
    """`pd` (repeated Prisoner's Dilemma), `coin` (Coin Game, 3x3 grid), `cleanup` (Clean Up, LIO's small fully
    observable version, `--num-agents` players) or `meltingpot:<substrate>` (e.g. clean_up)"""
    num_envs: int = 1
    """number of parallel copies of the game (each copy contains all agents)"""
    num_cpus: int = 0
    """subprocesses used by supersuit to step the env copies (0 = in-process)"""
    max_cycles: int = 1000
    """episode length: Melting Pot cycles, Prisoner's Dilemma rounds or Coin Game steps (50 in the literature)"""
    num_agents: int = 2
    """number of players for `pd` / `cleanup` / `debug:image` (Melting Pot substrates fix their own player count)"""
    pd_payoffs: tuple[float, float, float, float] = (3.0, 0.0, 4.0, 1.0)
    """Prisoner's Dilemma payoffs (R, S, T, P); with N > 2 players payoffs are averaged over all pairings"""
    cleanup_map: str = "10x10"
    """cleanup: `10x10` (LIO, 3 spawn points) | `7x7` (LIO, 2) | `original` (Hughes et al., 5+ players)"""
    cleanup_obs: Literal["planes", "rgb"] = "planes"
    """cleanup: egocentric full-map window as one-hot planes [self, others, apple, waste, river, wall, beam] or as RGB
    (observer drawn in a fixed self colour)"""
    cleanup_fixed_spawn: bool = False
    """cleanup: LIO's fixed spawn assignment (agent 0 on the river side) instead of shuffling spawn points per episode"""

    # --- policy --------------------------------------------------------------------------
    recurrent: bool = False
    """use an LSTM between trunk and heads (CleanRL ppo_atari_lstm.py); needs num_envs % num_minibatches == 0"""
    lstm_hidden_size: int = 128

    # --- social preference -----------------------------------------------------------------
    formulation: Formulation = "none"
    """functional form of the social term: none (plain PPO) | ei | svo | sia | ia"""
    signal: Signal = "value"
    """value: X_i from the agent's OWN critic on others' observations, added to the advantage (our proposal);
    reward: intrinsic reward from the others' smoothed rewards (literature baseline, needs reward access);
    imagined: the others' rewards imagined with the agent's OWN reward model f_i(o_j, a_j, o_j') (no reward access)"""
    imagined_critic: ImaginedCritic = "other"
    """signal=imagined: `other` = EI/SVO on the GAE advantage of the other's imagined rewards, with the agent's own
    critic on the other's observations as the other's value function (feed-forward only); `shaped` = imagined rewards
    through the intrinsic-reward path of signal=reward (all formulations, recurrent ok); `none` = ablation of `other`
    without any critic: the other's lambda-discounted sum of imagined rewards (no baseline, no bootstrap), EI/SVO;
    `level` = the report's coefficient term on the level "smoothed imagined rewards + weight * gamma * V_i(o_j')"
    (all formulations, feed-forward only; --level-value-weight as for --imagined-level trace_value)"""
    imagined_level: ImaginedLevel = "trace"
    """signal=imagined: the level "how well off is agent j" used by shaped (the formulation is applied to it) and by the
    inequity weights of other / none with sia / ia. `trace` = the smoothed imagined rewards (Hughes et al. with imagined
    rewards); `trace_value` = the trace plus the agent's own critic on the other's next observation, weight * gamma *
    V_i(o_j') ("what you earned lately plus what you are about to earn"; feed-forward only, not with critic none)"""
    ia_weighting: IAWeighting = "indicator"
    """signal=imagined, critic other / none, formulation sia / ia: how the current inequity sets the weights of the imagined
    advantages. `indicator` = Fehr-Schmidt's piecewise-linear utility (who is ahead); `relu` = quadratic utility (the
    standardised gap: the further ahead, the larger the weight)"""
    level_value_weight: Optional[float] = None
    """imagined_level=trace_value: weight of the value part; default (1 - gamma) / (1 - gamma * reward_lambda), which puts a
    constant reward stream on the same scale in the trace and in the value"""
    imagined_warmup: int = 20
    """signal=imagined: iterations during which the reward model trains but the social term is off"""
    imagined_lambda: Optional[float] = None
    """signal=imagined/other: GAE lambda of the other's imagined advantage (default: --gae-lambda). 1.0 = the other's
    imagined return with my critic used only at the rollout boundary, i.e. the reward model without the value idea"""
    reward_model_coef: float = 1.0
    """signal=imagined: weight of the reward-model regression loss (own transitions -> own reward)"""
    reward_model_replay: int = 0
    """signal=imagined: per-agent replay buffer (capacity) of the agent's own transitions with non-zero reward, replayed
    into the reward-model loss so that rare reward events are not forgotten when the agent's behaviour specialises
    (e.g. an agent that only cleans never eats an apple).  0 = off (recent rollout only)"""
    social_scale: bool = False
    """standardise the own advantage and every z[i, j] over the batch before the social term is formed, so that
    alpha is a weight relative to the agent's own advantage (signal=value and signal=imagined/other)"""
    social_aggregate: SocialAggregate = "mean"
    """how the others' terms are combined for N > 2 agents: `mean` over the N-1 others (alpha = weight of the average
    other; the formulations as written) or `sum` (alpha = weight of each other agent, utilitarian at alpha = 1)"""
    alpha: str = "0.0"
    """strength of the social term; one value for all agents or N comma-separated per-agent values"""
    beta: str = "0.0"
    """advantageous-inequity coefficient for ia; scalar or per-agent list"""
    phi: str = "0.0"
    """SVO angle in radians (0 = selfish, pi/2 = fully prosocial); scalar or per-agent list"""
    reward_lambda: float = 0.975
    """signal=reward: temporal smoothing e_t = gamma * lambda * e_{t-1} + r_t (0.975 as in Hughes et al.; 0 = raw rewards)"""
    cross_value_chunk: int = 64
    """time steps per chunk when evaluating V_i(s_j) for all pairs (memory / speed trade-off)"""
    bootstrap_truncation: bool = True
    """at pure time-limit truncations, bootstrap from V(terminal observation) instead of 0"""

    # --- PPO (CleanRL Atari defaults) --------------------------------------------------------
    total_timesteps: int = 5_000_000
    """total environment steps (one step of every env copy counts num_envs)"""
    learning_rate: float = 2.5e-4
    num_steps: int = 512
    """steps per env copy per rollout"""
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    """normalise (advantage + social term) per agent within each minibatch"""
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    """gradient clipping, applied per agent"""
    target_kl: Optional[float] = None
    """early stop the epoch loop when the largest per-agent approx KL exceeds this"""

    # --- filled in at runtime --------------------------------------------------------------
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def resolve(args: Args) -> Args:
    """Derive batch sizes and validate the combination of options."""
    if args.formulation not in FORMULATIONS:
        raise ValueError(f"--formulation must be one of {FORMULATIONS}")
    if args.signal not in SIGNALS:
        raise ValueError(f"--signal must be one of {SIGNALS}")
    if args.imagined_critic not in IMAGINED_CRITICS:
        raise ValueError(f"--imagined-critic must be one of {IMAGINED_CRITICS}")
    if args.signal == "imagined" and args.imagined_critic in ("other", "none"):
        if args.recurrent and args.imagined_critic == "other":
            raise ValueError("--imagined-critic other is implemented for feed-forward agents; use --imagined-critic shaped")
        if args.recurrent and args.formulation in ("sia", "ia"):
            raise ValueError("inequity weights for --imagined-critic none are implemented for feed-forward agents")
    if args.imagined_level not in ("trace", "trace_value"):
        raise ValueError("--imagined-level must be trace or trace_value")
    if args.imagined_level == "trace_value":
        inequity_weights_use = args.imagined_critic in ("other", "none") and args.formulation in ("sia", "ia")
        if not (args.signal == "imagined" and args.formulation != "none"
                and (args.imagined_critic == "shaped" or inequity_weights_use)):
            raise ValueError("--imagined-level trace_value applies to --imagined-critic shaped, or to other / none with sia / ia")
        if args.imagined_critic == "none":
            raise ValueError("--imagined-level trace_value needs the critic; use --imagined-critic other")
        if args.recurrent:
            raise ValueError("--imagined-level trace_value is implemented for feed-forward agents")
    if args.signal == "imagined" and args.imagined_critic == "level" and args.recurrent:
        raise ValueError("--imagined-critic level is implemented for feed-forward agents")
    if args.ia_weighting not in ("indicator", "relu"):
        raise ValueError("--ia-weighting must be indicator or relu")
    if args.level_value_weight is None and (
        args.imagined_level == "trace_value" or (args.signal == "imagined" and args.imagined_critic == "level")
    ):
        args.level_value_weight = (1.0 - args.gamma) / (1.0 - args.gamma * args.reward_lambda)
    if args.imagined_warmup < 0 or args.reward_model_coef < 0 or args.reward_model_replay < 0:
        raise ValueError("--imagined-warmup, --reward-model-coef and --reward-model-replay must be >= 0")
    if args.social_aggregate not in ("mean", "sum"):
        raise ValueError("--social-aggregate must be mean or sum")
    if args.cleanup_obs not in ("planes", "rgb"):
        raise ValueError("--cleanup-obs must be planes or rgb")
    if args.imagined_lambda is not None and not 0.0 <= args.imagined_lambda <= 1.0:
        raise ValueError("--imagined-lambda must be in [0, 1]")
    if args.num_envs < 1 or args.num_steps < 1 or args.num_minibatches < 1:
        raise ValueError("num_envs, num_steps and num_minibatches must be >= 1")
    if args.num_agents < 2:
        raise ValueError("num_agents must be >= 2")
    args.batch_size = int(args.num_envs * args.num_steps)
    if args.recurrent:
        if args.num_envs % args.num_minibatches != 0:
            raise ValueError(
                "recurrent policies form minibatches from whole env sequences: "
                f"num_envs ({args.num_envs}) must be divisible by num_minibatches ({args.num_minibatches})"
            )
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
    else:
        if args.batch_size % args.num_minibatches != 0:
            raise ValueError("batch_size (num_envs * num_steps) must be divisible by num_minibatches")
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.num_iterations < 1:
        raise ValueError("total_timesteps is smaller than one batch (num_envs * num_steps)")
    return args
