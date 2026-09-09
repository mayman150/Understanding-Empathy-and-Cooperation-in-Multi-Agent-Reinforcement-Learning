"""A tiny image-observation environment for smoke-testing the CNN / LSTM pipeline.

It mimics the *interface* of a Melting Pot substrate through shimmy: a PettingZoo
``ParallelEnv`` whose per-agent observation is a ``Dict`` with an ``"RGB"`` key holding an
``(88, 88, 3)`` uint8 image, ``num_agents`` players, a discrete action space and a fixed
episode length.  It is *not* a social dilemma; it exists so the training code can be run
on machines without ``dm-meltingpot`` (e.g. macOS laptops) and in CI.

Task: a bright square is drawn in one of ``num_actions`` slots along the top of the image;
choosing the matching action yields reward 1 (otherwise 0).  The target changes every step,
so a working CNN policy should learn it quickly.
"""
from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

IMG = 88


class DebugImageEnv(ParallelEnv):
    metadata = {"render_modes": ["rgb_array"], "name": "debug_image_v0"}

    def __init__(self, num_agents: int = 3, num_actions: int = 4, max_cycles: int = 50, render_mode=None):
        super().__init__()
        self._n = num_agents
        self.num_actions = num_actions
        self.max_cycles = max_cycles
        self.render_mode = render_mode
        self.possible_agents = [f"player_{i}" for i in range(num_agents)]
        self.agents: list[str] = []
        self._rng = np.random.default_rng()
        self._targets = np.zeros(num_agents, dtype=np.int64)
        self.num_cycles = 0

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Dict(
            {
                "RGB": spaces.Box(0, 255, shape=(IMG, IMG, 3), dtype=np.uint8),
                "READY_TO_SHOOT": spaces.Box(0.0, 1.0, shape=(), dtype=np.float64),
            }
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(self.num_actions)

    def _render_obs(self, target: int) -> dict:
        img = np.full((IMG, IMG, 3), 30, dtype=np.uint8)
        slot_w = IMG // self.num_actions
        x0 = target * slot_w
        img[4:20, x0 + 2 : x0 + slot_w - 2] = np.array([255, 220, 40], dtype=np.uint8)
        return {"RGB": img, "READY_TO_SHOOT": np.float64(1.0)}

    def _observe_all(self) -> dict:
        return {agent: self._render_obs(int(self._targets[i])) for i, agent in enumerate(self.possible_agents)}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.num_cycles = 0
        self._targets = self._rng.integers(0, self.num_actions, size=self._n)
        return self._observe_all(), {agent: {} for agent in self.agents}

    def step(self, actions):
        rewards = {
            agent: float(int(actions[agent]) == int(self._targets[i])) for i, agent in enumerate(self.agents)
        }
        self.num_cycles += 1
        self._targets = self._rng.integers(0, self.num_actions, size=self._n)
        done = self.num_cycles >= self.max_cycles
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        observations = self._observe_all()
        if done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self._render_obs(int(self._targets[0]))["RGB"]

    def close(self):
        pass
