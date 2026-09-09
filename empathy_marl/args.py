"""Experiment configuration (tyro dataclass, CleanRL style)."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional

from empathy_marl.empathy import FORMULATIONS

Formulation = Literal["none", "ei", "svo", "sia", "ia", "reward_ia"]


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
    """`pd` (repeated Prisoner's Dilemma) or `meltingpot:<substrate>` (e.g. clean_up)"""
    num_envs: int = 1
    """number of parallel copies of the game (each copy contains all agents)"""
    num_cpus: int = 0
    """subprocesses used by supersuit to step the env copies (0 = in-process)"""
    max_cycles: int = 1000
    """episode length: Melting Pot cycles or Prisoner's Dilemma rounds"""
    pd_payoffs: tuple[float, float, float, float] = (3.0, 0.0, 4.0, 1.0)
    """Prisoner's Dilemma payoffs (R, S, T, P)"""

    # --- policy --------------------------------------------------------------------------
    recurrent: bool = False
    """use an LSTM between trunk and heads (CleanRL ppo_atari_lstm.py); needs num_envs % num_minibatches == 0"""
    lstm_hidden_size: int = 128

    # --- social preference -----------------------------------------------------------------
    formulation: Formulation = "none"
    """none | ei | svo | sia | ia (value based) | reward_ia (Hughes et al. 2018, uses others' rewards)"""
    alpha: str = "0.0"
    """strength of the social term; one value for all agents or N comma-separated per-agent values"""
    beta: str = "0.0"
    """advantageous-inequity coefficient for ia / reward_ia; scalar or per-agent list"""
    phi: str = "0.0"
    """SVO angle in radians (0 = selfish, pi/2 = fully prosocial); scalar or per-agent list"""
    ia_lambda: float = 0.975
    """temporal smoothing of rewards in reward_ia (e_t = gamma * lambda * e_{t-1} + r_t)"""
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
    if args.num_envs < 1 or args.num_steps < 1 or args.num_minibatches < 1:
        raise ValueError("num_envs, num_steps and num_minibatches must be >= 1")
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
