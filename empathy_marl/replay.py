"""Replay of an agent's own *rewarded* transitions for the reward model (``--reward-model-replay``).

The reward model f_i(o, a, o') is trained on the agent's own recent transitions only.  When behaviour
specialises (in Clean Up an agent that only cleans never eats an apple) the recent data no longer contains
the reward event, and a model that keeps training on recent data forgets what it looks like; its imagined
rewards for the *other* agents, who do eat, then degrade.  This buffer keeps the agent's last ``capacity``
transitions with non-zero reward and replays a sample of them into every reward-model update, so the
event stays in the training distribution.  Still the agent's own experience only: nothing about the other
agents' rewards enters.
"""
from __future__ import annotations

import torch


class RewardReplay:
    def __init__(self, capacity: int, num_agents: int, obs_shape: tuple[int, ...], obs_dtype: torch.dtype, device: torch.device):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self.num_agents = num_agents
        self.obs = torch.zeros((num_agents, self.capacity) + obs_shape, dtype=obs_dtype, device=device)
        self.next_obs = torch.zeros_like(self.obs)
        self.actions = torch.zeros((num_agents, self.capacity), dtype=torch.long, device=device)
        self.rewards = torch.zeros((num_agents, self.capacity), device=device)
        self.size = [0] * num_agents
        self.ptr = [0] * num_agents

    def add_rollout(self, obs: torch.Tensor, actions: torch.Tensor, next_obs: torch.Tensor, env_rewards: torch.Tensor) -> None:
        """``obs, next_obs: (T, E, N, *obs)``, ``actions, env_rewards: (T, E, N)``: store every transition with reward != 0."""
        T, E, N = env_rewards.shape
        for i in range(N):
            idx = torch.nonzero(env_rewards[:, :, i].reshape(-1) != 0.0, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            if idx.numel() > self.capacity:  # keep the most recent ones
                idx = idx[-self.capacity:]
            n = idx.numel()
            slots = (self.ptr[i] + torch.arange(n, device=idx.device)) % self.capacity
            self.obs[i, slots] = obs[:, :, i].reshape((T * E,) + obs.shape[3:])[idx]
            self.next_obs[i, slots] = next_obs[:, :, i].reshape((T * E,) + next_obs.shape[3:])[idx]
            self.actions[i, slots] = actions[:, :, i].reshape(-1)[idx]
            self.rewards[i, slots] = env_rewards[:, :, i].reshape(-1)[idx]
            self.ptr[i] = int((self.ptr[i] + n) % self.capacity)
            self.size[i] = min(self.capacity, self.size[i] + n)

    def loss(self, agents, sample_size: int) -> torch.Tensor:
        """Per-agent ``0.5 * MSE`` of the reward model on a random replay sample; ``0`` for agents with an empty buffer.

        Returns a tensor of shape ``(N,)`` that is differentiable w.r.t. the reward-model parameters.
        """
        losses = []
        for i, agent in enumerate(agents.agents):
            if self.size[i] == 0:
                losses.append(torch.zeros((), device=self.rewards.device))
                continue
            n = min(self.size[i], max(1, sample_size))
            sel = torch.randint(0, self.size[i], (n,), device=self.rewards.device)
            pred = agent.predict_reward(self.obs[i, sel], self.actions[i, sel], self.next_obs[i, sel])
            losses.append(0.5 * ((pred - self.rewards[i, sel]) ** 2).mean())
        return torch.stack(losses)
