"""Social preference terms: one functional form, two signals.

Formulations (report, Sec. "Empathy formulations")
--------------------------------------------------
Every formulation is a function ``F_i(z)`` of a matrix ``z[..., i, j]`` = "agent *i*'s
estimate of how well off agent *j* is".  With ``zbar_i = mean_{j != i} z[i, j]``:

* ``none``  F_i = 0                                                    (plain PPO)
* ``ei``    F_i =  alpha_i * zbar_i                                    (Empathetic Influence)
* ``svo``   F_i =  alpha_i * (cos(phi_i) z[i, i] + sin(phi_i) zbar_i)  (Social Value Orientation, Schwarting et al. 2019)
* ``sia``   F_i = -alpha_i * | z[i, i] - zbar_i |                      (Simple Inequity Aversion)
* ``ia``    F_i = -alpha_i/(N-1) sum_{j!=i} max(z[i, j] - z[i, i], 0)
                  -beta_i /(N-1) sum_{j!=i} max(z[i, i] - z[i, j], 0)  (Inequity Aversion, Fehr & Schmidt / Hughes et al. 2018)

Signals
-------
``value`` (our proposal, no access to other agents' rewards):
    ``z[i, j] = V_i(S_j^{t+1})`` -- agent *i*'s **own critic** evaluated on agent *j*'s next
    observation.  ``X_i = F_i(z)`` is added to the advantage as a *constant* coefficient of
    the PPO surrogate (computed under ``no_grad``; the legacy scripts let gradients flow
    through it into the critic, which for SIA/IA rewards a critic that predicts the same
    value everywhere).

``reward`` (the literature's baseline, needs other agents' rewards):
    ``z[i, j] = e_j``, the temporally smoothed reward of agent *j*,
    ``e_j^t = gamma * lambda * e_j^{t-1} + r_j^t``, and ``F_i(z)`` is an **intrinsic reward**
    added to ``r_i`` before GAE.  With ``ia`` and lambda = 0.975 this is exactly the
    inequity-aversion reward of Hughes et al. (2018) (alpha = 5 / beta = 0.05 in their
    experiments); with ``svo`` and lambda = 0 it is the reward of Schwarting et al. (2019).

Because both signals share :func:`social_term`, a value-vs-reward comparison differs only
in what ``z`` is and where the term enters (policy-gradient coefficient vs. reward).
"""
from __future__ import annotations

import torch

FORMULATIONS = ("none", "ei", "svo", "sia", "ia")
SIGNALS = ("value", "reward")


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


def social_term(
    formulation: str,
    z: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """``F`` of shape ``(..., N)`` from ``z`` of shape ``(..., N, N)`` (``z[..., i, j]`` = i's estimate of j).

    ``alpha, beta, phi`` are per-agent tensors of shape ``(N,)`` (broadcast over leading dims).
    The result is detached: it is a coefficient / reward, not something to back-propagate through.
    """
    if formulation not in FORMULATIONS:
        raise ValueError(f"unknown formulation {formulation!r}; choose from {FORMULATIONS}")
    z = z.detach()
    N = z.shape[-1]
    if z.shape[-2] != N:
        raise ValueError(f"z must be (..., N, N), got {tuple(z.shape)}")
    if formulation == "none":
        return torch.zeros(z.shape[:-1], device=z.device, dtype=z.dtype)
    if N < 2:
        raise ValueError("social terms need at least two agents")

    alpha = alpha.to(z.device)
    beta = beta.to(z.device)
    phi = phi.to(z.device)
    eye = torch.eye(N, dtype=torch.bool, device=z.device)
    z_self = z.diagonal(dim1=-2, dim2=-1)  # (..., N)
    z_others_mean = z.masked_fill(eye, 0.0).sum(dim=-1) / (N - 1)

    if formulation == "ei":
        return alpha * z_others_mean
    if formulation == "svo":
        return alpha * (torch.cos(phi) * z_self + torch.sin(phi) * z_others_mean)
    if formulation == "sia":
        return -alpha * torch.abs(z_self - z_others_mean)
    if formulation == "ia":
        diff = z - z_self.unsqueeze(-1)  # [..., i, j] = z_ij - z_ii; zero on the diagonal
        disadvantageous = torch.relu(diff).sum(dim=-1) / (N - 1)  # others better off than me
        advantageous = torch.relu(-diff).sum(dim=-1) / (N - 1)  # me better off than others
        return -alpha * disadvantageous - beta * advantageous
    raise AssertionError("unreachable")


class RewardSocialShaper:
    """``signal=reward``: intrinsic reward ``F_i(e)`` from the other agents' smoothed rewards.

    Keeps ``e`` of shape ``(num_envs, N)`` across steps and restarts it at episode starts.
    """

    def __init__(
        self,
        formulation: str,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        phi: torch.Tensor,
        gamma: float,
        lam: float,
        num_envs: int,
        num_agents: int,
        device: torch.device,
    ):
        if formulation not in FORMULATIONS:
            raise ValueError(f"unknown formulation {formulation!r}")
        if formulation != "none" and num_agents < 2:
            raise ValueError("reward-based social terms need at least two agents")
        self.formulation = formulation
        self.alpha, self.beta, self.phi = alpha.to(device), beta.to(device), phi.to(device)
        self.decay = gamma * lam
        self.num_agents = num_agents
        self.e = torch.zeros(num_envs, num_agents, device=device)

    def __call__(self, rewards: torch.Tensor, episode_start: torch.Tensor) -> torch.Tensor:
        """``rewards, episode_start: (num_envs, N)`` -> intrinsic reward ``F: (num_envs, N)`` (add it to ``rewards``).

        ``episode_start[e, i] = 1`` when the observation that produced ``rewards[e, i]`` was the
        first of a new episode, so the smoothed reward restarts from zero.
        """
        self.e = self.decay * self.e * (1.0 - episode_start) + rewards
        z = self.e.unsqueeze(-2).expand(-1, self.num_agents, -1)  # z[e, i, j] = e_j (everyone sees true rewards)
        return social_term(self.formulation, z, self.alpha, self.beta, self.phi)
