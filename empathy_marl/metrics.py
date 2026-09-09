"""Per-episode social-outcome metrics for a vectorised multi-agent environment.

Metrics follow Perolat et al. (2017), "A multi-agent reinforcement learning model of
common-pool resource appropriation":

* ``r``               per-agent undiscounted episodic return, shape ``(num_agents,)``
* ``collective``      sum of ``r``
* ``l``               episode length ``T``
* ``efficiency``  U = sum_i R_i / T
* ``equality``    E = 1 - Gini(R)
* ``sustainability`` S = mean_i t_i, with t_i the average time step at which agent ``i``
                      receives a positive reward.  Agents that never receive a positive
                      reward are excluded; if no agent does, S = 0.

"Peace" (untagged agent steps) is not reported because the zap matrix is a ``WORLD.*``
observation that the shimmy wrapper strips.

The wrapper sits on top of the auto-resetting supersuit ``ConcatVecEnv`` and only *reads*
rewards, so reward shaping applied inside the training loop (``reward_ia``) does not
affect the metrics.  When environment ``e`` finishes an episode, its statistics are placed
in ``infos[e * num_agents]["ma_episode"]``.
"""
from __future__ import annotations

import time

import numpy as np


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    total = x.sum()
    if x.size == 0 or total <= 0:
        return 0.0
    return float(np.abs(x[:, None] - x[None, :]).sum() / (2.0 * x.size * total))


class MultiAgentEpisodeStatistics:
    def __init__(self, env, num_envs: int, num_agents: int):
        self.env = env
        self.num_envs = num_envs
        self.num_agents = num_agents
        if env.num_envs != num_envs * num_agents:
            raise ValueError(f"vec env has {env.num_envs} slots, expected {num_envs} x {num_agents}")
        self.t0 = time.perf_counter()
        self._reset_stats(np.ones(num_envs, dtype=bool))

    def __getattr__(self, name):  # delegate everything else to the wrapped vec env
        return getattr(self.env, name)

    def _reset_stats(self, mask: np.ndarray) -> None:
        if not hasattr(self, "episode_returns"):
            E, N = self.num_envs, self.num_agents
            self.episode_returns = np.zeros((E, N), dtype=np.float64)
            self.episode_lengths = np.zeros(E, dtype=np.int64)
            self.positive_time_sum = np.zeros((E, N), dtype=np.float64)
            self.positive_count = np.zeros((E, N), dtype=np.int64)
        self.episode_returns[mask] = 0.0
        self.episode_lengths[mask] = 0
        self.positive_time_sum[mask] = 0.0
        self.positive_count[mask] = 0

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self._reset_stats(np.ones(self.num_envs, dtype=bool))
        return obs, infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        E, N = self.num_envs, self.num_agents
        r = np.asarray(rewards, dtype=np.float64).reshape(E, N)
        done = (np.asarray(terminations, dtype=bool) | np.asarray(truncations, dtype=bool)).reshape(E, N)

        self.episode_lengths += 1
        self.episode_returns += r
        positive = r > 0.0
        self.positive_time_sum += positive * self.episode_lengths[:, None]
        self.positive_count += positive

        env_done = done.all(axis=1)
        partial = done.any(axis=1) & ~env_done
        if partial.any():
            print(f"[metrics] warning: agents of env(s) {np.flatnonzero(partial)} finished at different times")

        for e in np.flatnonzero(env_done):
            T = int(self.episode_lengths[e])
            returns = self.episode_returns[e].copy()
            with np.errstate(invalid="ignore", divide="ignore"):
                t_i = self.positive_time_sum[e] / self.positive_count[e]
            has_reward = self.positive_count[e] > 0
            sustainability = float(t_i[has_reward].mean()) if has_reward.any() else 0.0
            infos[e * N]["ma_episode"] = {
                "r": returns,
                "collective": float(returns.sum()),
                "l": T,
                "efficiency": float(returns.sum() / max(T, 1)),
                "equality": 1.0 - gini(returns),
                "sustainability": sustainability,
                "t": round(time.perf_counter() - self.t0, 6),
            }
        if env_done.any():
            self._reset_stats(env_done)
        return obs, rewards, terminations, truncations, infos
