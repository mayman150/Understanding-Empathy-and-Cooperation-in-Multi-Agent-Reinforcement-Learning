"""Social terms added to the policy-gradient coefficient, and the reward-based baseline.

Value-based formulations (report, Sec. "Empathy formulations")
---------------------------------------------------------------
All of them are functions of ``v_next[..., i, j] = V_i(S_j^{t+1})``: agent ``i``'s *own*
critic evaluated on agent ``j``'s next observation.  No agent ever sees another agent's
reward.  With ``Vbar_i = mean_{j != i} V_i(S_j^{t+1})``:

* ``none``  X_i = 0                                                    (plain PPO)
* ``ei``    X_i =  alpha_i * Vbar_i                                    (Empathetic Influence)
* ``svo``   X_i =  alpha_i * (cos(phi_i) V_i(S_i^{t+1}) + sin(phi_i) Vbar_i)   (Social Value Orientation)
* ``sia``   X_i = -alpha_i * | V_i(S_i^{t+1}) - Vbar_i |               (Simple Inequity Aversion)
* ``ia``    X_i = -alpha_i/(N-1) sum_{j!=i} max(V_i(S_j') - V_i(S_i'), 0)
                  -beta_i /(N-1) sum_{j!=i} max(V_i(S_i') - V_i(S_j'), 0)     (Inequity Aversion)

``X_i`` is treated as a *constant* coefficient in the PPO surrogate, exactly like the
advantage: it must be computed under ``torch.no_grad()`` (or detached).  The legacy
scripts let gradients flow from the policy loss through ``X_i`` into the critic, which
for SIA/IA rewards the critic for predicting the same value everywhere.

Reward-based baseline
---------------------
``reward_ia`` is the inequity-aversion reward of Hughes et al. (2018), which *does*
require access to the other agents' rewards.  It is the "PPO + inequity aversion with
rewards" comparison point from the plan:

    e_i^t = gamma * lambda * e_i^{t-1} + r_i^t                (temporally smoothed reward)
    r_i^total = r_i^t - alpha_i/(N-1) sum_{j!=i} max(e_j - e_i, 0)
                      - beta_i /(N-1) sum_{j!=i} max(e_i - e_j, 0)

Hughes et al. use lambda = 0.975 and, for the advantageous-inequity-averse agents,
alpha = 0 / beta = 0.05; for disadvantageous-averse agents alpha = 5 / beta = 0.
"""
from __future__ import annotations

import torch

VALUE_BASED = ("ei", "svo", "sia", "ia")
REWARD_BASED = ("reward_ia",)
FORMULATIONS = ("none",) + VALUE_BASED + REWARD_BASED


def parse_per_agent(spec: str | float, num_agents: int, name: str) -> torch.Tensor:
    """``"0.1"`` -> all agents 0.1; ``"0,0.1"`` -> per-agent values (length must be ``num_agents``)."""
    if isinstance(spec, (int, float)):
        values = [float(spec)]
    else:
        values = [float(v) for v in str(spec).split(",") if v.strip() != ""]
    if len(values) == 1:
        values = values * num_agents
    if len(values) != num_agents:
        raise ValueError(f"--{name} must be a single value or {num_agents} comma-separated values, got {spec!r}")
    return torch.tensor(values, dtype=torch.float32)


def _split_self_others(v_next: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(v_self (..., N), v_others (..., N, N) with the diagonal zeroed, others_mask (N, N))``."""
    N = v_next.shape[-1]
    if v_next.shape[-2] != N:
        raise ValueError(f"v_next must be (..., N, N), got {tuple(v_next.shape)}")
    eye = torch.eye(N, dtype=torch.bool, device=v_next.device)
    v_self = v_next.diagonal(dim1=-2, dim2=-1)
    v_others = v_next.masked_fill(eye, 0.0)
    return v_self, v_others, ~eye


def social_term(
    formulation: str,
    v_next: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Compute ``X`` of shape ``(..., N)`` from ``v_next`` of shape ``(..., N, N)``.

    ``alpha, beta, phi`` are per-agent tensors of shape ``(N,)`` (broadcast over leading dims).
    The result is detached: it is a coefficient, not something to back-propagate through.
    """
    if formulation not in FORMULATIONS:
        raise ValueError(f"unknown formulation {formulation!r}; choose from {FORMULATIONS}")
    v_next = v_next.detach()
    N = v_next.shape[-1]
    if formulation in ("none",) + REWARD_BASED:
        return torch.zeros(v_next.shape[:-1], device=v_next.device, dtype=v_next.dtype)
    if N < 2:
        raise ValueError("value-based social terms need at least two agents")

    alpha = alpha.to(v_next.device)
    beta = beta.to(v_next.device)
    phi = phi.to(v_next.device)
    v_self, v_others, _ = _split_self_others(v_next)
    v_others_mean = v_others.sum(dim=-1) / (N - 1)

    if formulation == "ei":
        return alpha * v_others_mean
    if formulation == "svo":
        return alpha * (torch.cos(phi) * v_self + torch.sin(phi) * v_others_mean)
    if formulation == "sia":
        return -alpha * torch.abs(v_self - v_others_mean)
    if formulation == "ia":
        diff = v_next - v_self.unsqueeze(-1)  # [..., i, j] = V_i(s_j') - V_i(s_i'); zero on the diagonal
        disadvantageous = torch.relu(diff).sum(dim=-1) / (N - 1)  # others better off than me
        advantageous = torch.relu(-diff).sum(dim=-1) / (N - 1)  # me better off than others
        return -alpha * disadvantageous - beta * advantageous
    raise AssertionError("unreachable")


class InequityAversionRewardShaper:
    """Hughes et al. (2018) inequity-aversion intrinsic reward (needs other agents' rewards).

    Keeps the temporally smoothed rewards ``e`` of shape ``(num_envs, N)`` across steps.
    """

    def __init__(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: float,
        lam: float,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        if num_agents < 2:
            raise ValueError("reward_ia needs at least two agents")
        self.alpha = alpha.to(device)
        self.beta = beta.to(device)
        self.decay = gamma * lam
        self.num_agents = num_agents
        self.e = torch.zeros(num_envs, num_agents, device=device)

    def __call__(self, rewards: torch.Tensor, episode_start: torch.Tensor) -> torch.Tensor:
        """``rewards, episode_start: (num_envs, N)`` -> shaped rewards ``(num_envs, N)``.

        ``episode_start[e, i] = 1`` when the observation that produced ``rewards[e, i]`` was the
        first of a new episode, so the smoothed reward restarts from zero.
        """
        N = self.num_agents
        self.e = self.decay * self.e * (1.0 - episode_start) + rewards
        diff = self.e.unsqueeze(-1) - self.e.unsqueeze(-2)  # [e, i, j] = e_i - e_j
        disadvantageous = torch.relu(-diff).sum(dim=-1) / (N - 1)  # sum_j max(e_j - e_i, 0)
        advantageous = torch.relu(diff).sum(dim=-1) / (N - 1)  # sum_j max(e_i - e_j, 0)
        return rewards - self.alpha * disadvantageous - self.beta * advantageous
