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

``imagined`` (no access to other agents' rewards; outcome-based):
    agent *i* learns a reward model ``f_i(o, a, o')`` of its **own** reward from its own
    transitions and imagines agent *j*'s reward as ``rhat_ij = f_i(o_j, a_j, o_j')`` (its observation, action and next observation).
    Two ways to use it (``--imagined-critic``):

    * ``shaped``: ``rhat`` replaces ``r_j`` in the reward signal above
      (:class:`ImaginedRewardShaper`), so ``F_i`` is an intrinsic reward and the agent's critic
      learns the shaped return.  Works for every formulation.
    * ``other``: for EI / SVO, ``z[i, j] = Ahat_ij``, the GAE advantage of *j*'s imagined
      rewards with agent *i*'s **own critic on j's observations** as *j*'s value function
      (``delta = rhat_ij + gamma V_i(o_j') - V_i(o_j)``), and ``z[i, i] = A_i``.  ``F_i(z)`` is added
      to the advantage.  With lambda = 1 this is the policy gradient of ``J_i + alpha * J_j^imagined``;
      the value function plays its usual role (bootstrap / baseline of a return made of
      outcomes) instead of standing in for the reward.
    * ``none``: the ablation of ``other`` without a critic: ``z[i, j]`` is the lambda-discounted sum of
      *j*'s imagined rewards from ``t`` on (``delta = rhat_ij``, no baseline, no bootstrap), so the
      consequences of my action that arrive after the rollout window, or that the critic would carry
      through the bootstrap, are not credited.  Comparing ``none`` with ``other`` measures what the
      value function on the other's observation contributes.

Because all signals share :func:`social_term`, comparisons differ only in what ``z`` is and
where the term enters (policy-gradient coefficient vs. reward).  :func:`standardize` implements
the optional relative scaling (``--social-scale``): every ``z[i, j]`` and the own advantage are
standardised over the batch before ``F_i`` is applied, so ``alpha`` is a weight relative to the
agent's own advantage.
"""
from __future__ import annotations

import torch

FORMULATIONS = ("none", "ei", "svo", "sia", "ia")
SIGNALS = ("value", "reward", "imagined")
IMAGINED_CRITICS = ("other", "shaped", "none")
AGGREGATES = ("mean", "sum")


def standardize(x: torch.Tensor, dims: tuple[int, ...], eps: float = 1e-6) -> torch.Tensor:
    """``(x - mean) / max(std, eps)`` over ``dims`` (kept per remaining index); constants map to 0."""
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True, unbiased=False)
    return (x - mean) / torch.clamp(std, min=eps)


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
    aggregate: str = "mean",
) -> torch.Tensor:
    """``F`` of shape ``(..., N)`` from ``z`` of shape ``(..., N, N)`` (``z[..., i, j]`` = i's estimate of j).

    ``alpha, beta, phi`` are per-agent tensors of shape ``(N,)`` (broadcast over leading dims).  ``aggregate``
    says how the ``N - 1`` others are combined: ``"mean"`` (the formulas above, ``zbar_i`` and the ``1/(N-1)`` of IA)
    or ``"sum"`` (every other agent enters with weight ``alpha`` / ``beta``; identical for ``N = 2``).
    The result is detached: it is a coefficient / reward, not something to back-propagate through.
    """
    if formulation not in FORMULATIONS:
        raise ValueError(f"unknown formulation {formulation!r}; choose from {FORMULATIONS}")
    if aggregate not in AGGREGATES:
        raise ValueError(f"unknown aggregate {aggregate!r}; choose from {AGGREGATES}")
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
    denom = (N - 1) if aggregate == "mean" else 1
    z_others = z.masked_fill(eye, 0.0).sum(dim=-1) / denom  # mean (or sum) over the others

    if formulation == "ei":
        return alpha * z_others
    if formulation == "svo":
        return alpha * (torch.cos(phi) * z_self + torch.sin(phi) * z_others)
    if formulation == "sia":
        return -alpha * torch.abs(z_self - z_others)
    if formulation == "ia":
        diff = z - z_self.unsqueeze(-1)  # [..., i, j] = z_ij - z_ii; zero on the diagonal
        disadvantageous = torch.relu(diff).sum(dim=-1) / denom  # others better off than me
        advantageous = torch.relu(-diff).sum(dim=-1) / denom  # me better off than others
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
        aggregate: str = "mean",
    ):
        if formulation not in FORMULATIONS:
            raise ValueError(f"unknown formulation {formulation!r}")
        if formulation != "none" and num_agents < 2:
            raise ValueError("reward-based social terms need at least two agents")
        self.formulation = formulation
        self.alpha, self.beta, self.phi = alpha.to(device), beta.to(device), phi.to(device)
        self.aggregate = aggregate
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
        return social_term(self.formulation, z, self.alpha, self.beta, self.phi, self.aggregate)


class ImaginedRewardShaper:
    """``signal=imagined, critic=shaped``: the reward signal with imagined rewards instead of observed ones.

    Keeps one smoothed trace per (observer *i*, observed *j*): ``ehat[e, i, j]`` with
    ``ehat_ij^t = gamma * lambda * ehat_ij^{t-1} + rhat_ij^t``, where ``rhat_ij`` is *i*'s imagined reward
    of *j* and ``rhat_ii`` is *i*'s own true reward.  ``z[e, i, j] = ehat_ij`` differs across observers,
    unlike :class:`RewardSocialShaper` where everyone reads the same true ``e_j``.
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
        aggregate: str = "mean",
    ):
        if formulation not in FORMULATIONS:
            raise ValueError(f"unknown formulation {formulation!r}")
        if formulation != "none" and num_agents < 2:
            raise ValueError("social terms need at least two agents")
        self.formulation = formulation
        self.alpha, self.beta, self.phi = alpha.to(device), beta.to(device), phi.to(device)
        self.aggregate = aggregate
        self.decay = gamma * lam
        self.num_agents = num_agents
        self.e = torch.zeros(num_envs, num_agents, num_agents, device=device)

    def __call__(
        self, imagined: torch.Tensor, episode_start: torch.Tensor, value_level: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``imagined: (num_envs, N, N)`` (``[e, i, j]`` = i's estimate of j's reward this step, diagonal = own true
        reward), ``episode_start: (num_envs, N)`` for the observed agents -> intrinsic reward ``(num_envs, N)``.

        ``value_level`` (optional, ``(num_envs, N, N)``): a forecast to add to the smoothed trace before the
        formulation is applied, ``z[e, i, j] = ehat_ij + value_level[e, i, j]`` ("what j earned lately plus what j is
        about to earn", the ``trace_value`` level of ``--imagined-level``; the caller supplies
        ``weight * gamma * V_i(o_j')``).  The trace itself is unaffected.
        """
        reset = (1.0 - episode_start).unsqueeze(1)  # (E, 1, N): the trace of observed agent j restarts with j's episode
        self.e = self.decay * self.e * reset + imagined
        level = self.e if value_level is None else self.e + value_level
        return social_term(self.formulation, level, self.alpha, self.beta, self.phi, self.aggregate)
