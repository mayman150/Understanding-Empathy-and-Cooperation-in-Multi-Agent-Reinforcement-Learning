"""N-player repeated Prisoner's Dilemma as a PettingZoo ``ParallelEnv``.

This is the "too easy" sanity-check environment: it lets us verify that the social
terms change behaviour in the expected direction and lets agents with different
social preferences (selfish vs. empathetic) interact in a controlled setting.

Actions: 0 = Cooperate, 1 = Defect.

Payoffs are given as ``(R, S, T, P)`` (reward, sucker, temptation, punishment):

                 other: C        other: D
    me: C        (R, R)          (S, T)
    me: D        (T, S)          (P, P)

The default ``(3, 0, 4, 1)`` is the matrix from the report and satisfies the usual
PD conditions ``T > R > P > S`` and ``2R > T + S``.

With ``num_players > 2`` every agent plays the stage game against each of the others
with its single action and receives the **average** pairwise payoff.  With ``k`` of the
``N - 1`` others cooperating, a cooperator earns ``(k R + (N-1-k) S) / (N-1)`` and a
defector ``(k T + (N-1-k) P) / (N-1)``: defecting still dominates and all-cooperate
still beats all-defect, and for ``N = 2`` this is exactly the matrix above.

Observation (per agent, float32, shape ``(5,)``):
    ``[own_last_C, own_last_D, frac_others_C, frac_others_D, first_round]``
i.e. a one-hot of the agent's own previous action, the fraction of the *other* players
that cooperated / defected in the previous round (a one-hot of the opponent's action when
``N = 2``) and a flag for the first round.  The observation is agent specific, which is
what the value-based social term needs (``V_i(s_j)`` must say something about *j*).
The round index is intentionally *not* observed so that policies cannot condition on
the finite horizon (see the discussion of end-game defection in the report).
"""
from __future__ import annotations

import functools
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

COOPERATE = 0
DEFECT = 1
OBS_DIM = 5


class RepeatedPrisonersDilemma(ParallelEnv):
    metadata = {"render_modes": ["ansi"], "name": "repeated_prisoners_dilemma_v0"}

    cooperate_action = COOPERATE

    def __init__(
        self,
        num_rounds: int = 100,
        payoffs: tuple[float, float, float, float] = (3.0, 0.0, 4.0, 1.0),
        num_players: int = 2,
        render_mode: str | None = None,
    ):
        super().__init__()
        if num_rounds < 1:
            raise ValueError("num_rounds must be >= 1")
        if num_players < 2:
            raise ValueError("num_players must be >= 2")
        reward, sucker, temptation, punishment = payoffs
        if not (temptation > reward > punishment > sucker):
            raise ValueError(f"payoffs {payoffs} do not satisfy T > R > P > S")
        self.num_rounds = int(num_rounds)
        self.payoffs = tuple(float(p) for p in payoffs)
        # payoff_matrix[my_action, other_action] -> my reward from that pairing
        self.payoff_matrix = np.array([[reward, sucker], [temptation, punishment]], dtype=np.float32)
        self.render_mode = render_mode
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents: list[str] = []
        self.round = 0
        self.last_actions: dict[str, int] | None = None
        self._rng = np.random.default_rng()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return spaces.Discrete(2)

    def _others(self, agent: str) -> list[str]:
        return [a for a in self.possible_agents if a != agent]

    def _observe(self, agent: str) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        if self.last_actions is None:
            obs[4] = 1.0
        else:
            obs[self.last_actions[agent]] = 1.0
            others = np.array([self.last_actions[o] for o in self._others(agent)])
            frac_cooperate = float((others == COOPERATE).mean())
            obs[2] = frac_cooperate
            obs[3] = 1.0 - frac_cooperate
        return obs

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.round = 0
        self.last_actions = None
        observations = {agent: self._observe(agent) for agent in self.agents}
        infos: dict[str, dict[str, Any]] = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, int]):
        if not self.agents:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        acts = {agent: int(actions[agent]) for agent in self.agents}
        rewards = {
            agent: float(np.mean([self.payoff_matrix[acts[agent], acts[o]] for o in self._others(agent)]))
            for agent in self.agents
        }
        self.last_actions = acts
        self.round += 1
        truncated = self.round >= self.num_rounds
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: {"actions": dict(acts)} for agent in self.agents}
        observations = {agent: self._observe(agent) for agent in self.agents}
        if truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.last_actions is None:
            return "round 0: no actions yet"
        names = {COOPERATE: "C", DEFECT: "D"}
        return f"round {self.round}: " + " ".join(f"{a}={names[self.last_actions[a]]}" for a in self.possible_agents)

    def close(self):
        pass
