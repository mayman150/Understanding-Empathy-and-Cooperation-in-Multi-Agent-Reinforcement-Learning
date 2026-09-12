"""Coin Game (Lerer & Peysakhovich 2017; Foerster et al. 2018, LOLA) as a PettingZoo ``ParallelEnv``.

The "one step above matrix games" social dilemma: the Prisoner's Dilemma played out on a
grid, so that (unlike the repeated PD) the *state* says something about how well off each
agent is.  It replaces the PD as the sanity check for the value-based social term.

Two agents, red (``player_0``) and blue (``player_1``), move on a small toroidal grid on
which a single coin of one colour is present.

* An agent that steps onto the coin receives ``+1`` (whatever the coin's colour).
* If the coin has the *other* agent's colour, that other agent receives ``-2``.
* A new coin of the other colour then appears on a cell not occupied by either agent
  (the first coin's colour is random, after that colours alternate, as in LOLA).

Per coin: taking your own colour is ``(+1, 0)`` for (you, other) -- collectively ``+1``;
taking the other's colour is ``(+1, -2)`` -- collectively ``-1``.  Grabbing every coin is the
individually dominant strategy but, when both agents do it, the ``-2``s cancel the ``+1``s
and both end up near zero.  *Cooperation* means taking only your own colour (Temptation >
Reward > Punishment > Sucker, exactly the PD ordering).  Independent learners are known to
grab everything (about 50% own-colour pickups); prosocial / opponent-shaping agents move
towards 100%.

Actions: ``0`` = up, ``1`` = down, ``2`` = left, ``3`` = right (moves wrap around the edges).
Both agents move simultaneously; if both land on the coin, both receive ``+1`` and the owner
additionally receives ``-2`` (LOLA convention).

Observation (per agent, float32, shape ``(4 * grid * grid,)``): four ``grid x grid`` planes,
flattened, in an **agent-relative** order

    [own position, other agent's position, coin of own colour, coin of the other's colour]

so handing agent *i* the observation of agent *j* means "the board as if I were where you
are, with your colour" -- which is what the value-based social term ``V_i(o_j)`` needs.
By default the planes are also **egocentric**: the torus is rolled so that the observing
agent sits in the centre cell, and the other planes hold *relative* offsets (the coin
"one step to my right" always looks the same).  This does not change the game, only the
encoding; it removes the position-dependent navigation problem from the policy so that
learning is about the social decision (``egocentric=False`` gives LOLA's absolute planes).

Episode statistics (in ``infos[agent]["episode_stats"]`` on the last step of an episode):
``own_coins``, ``other_coins`` (coins of each colour the agent picked up) and
``cooperation_rate`` = ``own_coins / (own_coins + other_coins)`` (``nan`` if the agent picked
up nothing).  :class:`empathy_marl.metrics.MultiAgentEpisodeStatistics` forwards them and
``train.py`` logs them as ``charts/<key>/<agent>``.  Read these rather than ``charts/equality``:
returns can be negative here, and ``1 - Gini`` is only meaningful for non-negative returns.

Reference points (independent PPO, 1M steps, 50-step episodes): plain PPO converges within
~200k steps to grabbing everything -- cooperation rate 0.50, ~16 coins of each colour per agent
per episode, collective return ~0; full cooperation is ~11 own coins per agent per episode and
a collective return of ~22 (+1 per coin).
"""
from __future__ import annotations

import functools
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)  # (row, col) deltas
NUM_PLAYERS = 2
RED, BLUE = 0, 1  # player_0 is red, player_1 is blue


class CoinGame(ParallelEnv):
    metadata = {"render_modes": ["ansi"], "name": "coin_game_v0"}

    cooperate_action = None  # cooperation is "leave the other's coins alone", not an action

    def __init__(
        self, num_rounds: int = 50, grid_size: int = 3, egocentric: bool = True, render_mode: str | None = None
    ):
        super().__init__()
        if num_rounds < 1:
            raise ValueError("num_rounds must be >= 1")
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2 (the coin needs a free cell)")
        self.num_rounds = int(num_rounds)
        self.grid_size = int(grid_size)
        self.egocentric = bool(egocentric)
        self.render_mode = render_mode
        self.possible_agents = [f"player_{i}" for i in range(NUM_PLAYERS)]
        self.agents: list[str] = []
        self.round = 0
        self.positions = np.zeros((NUM_PLAYERS, 2), dtype=np.int64)  # [player, (row, col)]
        self.coin_pos = np.zeros(2, dtype=np.int64)
        self.coin_owner = RED
        self.coins_taken = np.zeros((NUM_PLAYERS, 2), dtype=np.int64)  # [player, (own colour, other colour)]
        self._rng = np.random.default_rng()

    # ---- spaces ---------------------------------------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Box(0.0, 1.0, shape=(4 * self.grid_size * self.grid_size,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return spaces.Discrete(len(MOVES))

    # ---- helpers --------------------------------------------------------------------------------
    def _spawn_coin(self) -> None:
        """New coin of the other colour on a cell that neither agent occupies."""
        self.coin_owner = 1 - self.coin_owner
        occupied = {tuple(p) for p in self.positions}
        free = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if (r, c) not in occupied]
        self.coin_pos = np.array(free[self._rng.integers(len(free))], dtype=np.int64)

    def _observe(self, player: int) -> np.ndarray:
        g = self.grid_size
        planes = np.zeros((4, g, g), dtype=np.float32)
        other = 1 - player
        planes[0, self.positions[player, 0], self.positions[player, 1]] = 1.0
        planes[1, self.positions[other, 0], self.positions[other, 1]] = 1.0
        planes[2 if self.coin_owner == player else 3, self.coin_pos[0], self.coin_pos[1]] = 1.0
        if self.egocentric:  # roll the torus so that the observing agent is in the centre cell
            shift = (g // 2 - self.positions[player]) % g
            planes = np.roll(planes, (int(shift[0]), int(shift[1])), axis=(1, 2))
        return planes.reshape(-1)

    # ---- PettingZoo API -------------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = self.possible_agents[:]
        self.round = 0
        self.positions = self._rng.integers(self.grid_size, size=(NUM_PLAYERS, 2))
        self.coin_owner = int(self._rng.integers(2))  # _spawn_coin flips it: the first colour is random
        self._spawn_coin()
        self.coins_taken[:] = 0
        observations = {agent: self._observe(i) for i, agent in enumerate(self.agents)}
        infos: dict[str, dict[str, Any]] = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, int]):
        if not self.agents:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        for i, agent in enumerate(self.agents):
            self.positions[i] = (self.positions[i] + MOVES[int(actions[agent])]) % self.grid_size

        rewards = np.zeros(NUM_PLAYERS, dtype=np.float64)
        picked = [i for i in range(NUM_PLAYERS) if np.array_equal(self.positions[i], self.coin_pos)]
        for i in picked:
            rewards[i] += 1.0
            if i == self.coin_owner:
                self.coins_taken[i, 0] += 1
            else:
                rewards[self.coin_owner] -= 2.0
                self.coins_taken[i, 1] += 1
        if picked:
            self._spawn_coin()

        self.round += 1
        truncated = self.round >= self.num_rounds
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos: dict[str, dict[str, Any]] = {agent: {} for agent in self.agents}
        if truncated:
            for i, agent in enumerate(self.agents):
                own, other = int(self.coins_taken[i, 0]), int(self.coins_taken[i, 1])
                infos[agent]["episode_stats"] = {
                    "own_coins": float(own),
                    "other_coins": float(other),
                    "cooperation_rate": own / (own + other) if own + other > 0 else float("nan"),
                }
        observations = {agent: self._observe(i) for i, agent in enumerate(self.agents)}
        reward_dict = {agent: float(rewards[i]) for i, agent in enumerate(self.agents)}
        if truncated:
            self.agents = []
        return observations, reward_dict, terminations, truncations, infos

    def render(self):
        g = self.grid_size
        board = [["." for _ in range(g)] for _ in range(g)]
        board[self.coin_pos[0]][self.coin_pos[1]] = "r" if self.coin_owner == RED else "b"
        for i, mark in ((RED, "R"), (BLUE, "B")):
            r, c = self.positions[i]
            board[r][c] = "X" if board[r][c] in ("R", "B") else mark  # X = both agents on one cell
        return f"round {self.round}:\n" + "\n".join(" ".join(row) for row in board)

    def close(self):
        pass
