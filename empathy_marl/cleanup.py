"""Clean Up (Hughes et al. 2018) in the small, fully observable configuration of Yang et al. (2020, LIO), as a
PettingZoo ``ParallelEnv`` with image-shaped observations.

The game
--------
A 10x10 grid (walls included).  Apples spawn on the orchard cells on the right (+1 to the agent that steps on
one, nothing else is rewarded); waste spawns in the river on the left.  The apple spawn probability falls
linearly from ``appleRespawnProbability`` (0.3) at zero waste to 0 at the depletion threshold (waste on 40% of
the 16 river cells); every episode starts *above* the threshold (8 of 16 cells polluted) with no apples, so
nothing grows until someone cleans.  A cleaning beam (action ``clean``) travels up to 5 cells upward from the
agent, 3 cells wide, and turns the first waste cell it meets in each column back into river.  Waste respawns
with probability 0.5 per step (one cell) while below the threshold, so keeping the orchard productive needs
*continuous* cleaning, which pays nothing to the cleaner: a public-good dilemma with free-riding.  Agents
always face up (no rotation, as in LIO), so cleaning means walking to the river columns; the LIO punishment
("fining") beam is disabled.  50-step episodes.

Dynamics are the vendored SSD code (:mod:`empathy_marl.ssd`, unchanged); this module only maps actions and
renders observations.  Deviations from LIO's configuration, all optional: ``view_size`` defaults to the value
that puts *every* cell of the map inside the window from every position (8 for 10x10; LIO used 7, which
cuts the far wall row off for agents on the edge rows), observations are planes instead of RGB
(``obs="rgb"`` restores colours, with the observer drawn in a fixed "self" colour), and spawn points are
shuffled every episode (``shuffle_spawn=True``; LIO's fixed assignment puts agent 0 on the river side, which
builds an asymmetry into the roles).

Observation (per agent): an egocentric window of ``(2 * view_size + 1)`` cells a side centred on the agent
(cells outside the map are empty), as ``view x view x 7`` float32 planes in HWC layout

    [self, other agents, apple, waste, river, wall, cleaning beam]

so handing agent *i* the observation of agent *j* means "the world with me standing where you stand", which is
what the value-based and imagined-reward social terms (``V_i(o_j)``, ``f_i(o_j, a_j, o_j')``) need.

Actions: 0 left, 1 right, 2 up, 3 down, 4 stay, 5 clean.

Episode statistics (``infos[agent]["episode_stats"]`` on the last step, forwarded by
:class:`empathy_marl.metrics.MultiAgentEpisodeStatistics` and logged as ``charts/<key>/<agent>``):
``apples`` (eaten by the agent), ``clean_actions`` (cleaning beams fired), ``waste_cleaned`` (waste cells the
agent's beams removed), ``waste_density`` (mean over the episode of the polluted fraction of the river; the
same value for every agent) and ``apple_prob`` (mean apple spawn probability per cell and step, same for all).
"""
from __future__ import annotations

import functools
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from empathy_marl.ssd import CLEANUP_PARAMS, MAPS
from empathy_marl.ssd.cleanup import CLEANUP_COLORS
from empathy_marl.ssd.cleanup import CleanupEnv as SSDCleanup
from empathy_marl.ssd.map_env import DEFAULT_COLOURS

LEFT, RIGHT, UP, DOWN, STAY, CLEAN = range(6)
ACTION_NAMES = ("left", "right", "up", "down", "stay", "clean")
SSD_ACTIONS = (0, 1, 2, 3, 4, 8)  # MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN, STAY, CLEAN (LIO: no rotation, no fining beam)

PLANE_NAMES = ("self", "others", "apple", "waste", "river", "wall", "beam")
SELF, OTHERS = 0, 1
CELL_PLANES = {"A": 2, "H": 3, "R": 4, "S": 4, "@": 5, "C": 6}  # ' ' floor and '0' (outside the map) are all-zero
SELF_COLOUR = np.array([159, 67, 255], dtype=np.uint8)  # SSD's purple (agent 1)
OTHERS_COLOUR = np.array([2, 81, 154], dtype=np.uint8)  # SSD's blue (agent 2)


def full_view_size(ascii_map: list[str]) -> int:
    """Smallest view radius for which the whole map is inside the window from every non-wall cell.

    Agents cannot stand on the outer wall, so from row r in [1, H-2] a radius v covers rows r-v..r+v, which
    contains 0..H-1 for every such r iff v >= H-2 (and likewise for columns).
    """
    return max(len(ascii_map), len(ascii_map[0])) - 2


class CleanupEnv(ParallelEnv):
    metadata = {"render_modes": ["ansi", "rgb_array"], "name": "cleanup_v0"}

    cooperate_action = None  # cooperation is cleaning, measured by the episode statistics, not a single action

    def __init__(
        self,
        num_agents: int = 3,
        num_rounds: int = 50,
        map_name: str = "10x10",
        view_size: int | None = None,
        obs: str = "planes",
        shuffle_spawn: bool = True,
        render_mode: str | None = None,
    ):
        super().__init__()
        if map_name not in MAPS:
            raise ValueError(f"unknown map {map_name!r}; choose from {sorted(MAPS)}")
        if obs not in ("planes", "rgb"):
            raise ValueError("obs must be 'planes' or 'rgb'")
        if num_rounds < 1:
            raise ValueError("num_rounds must be >= 1")
        ascii_map = MAPS[map_name]
        n_spawn = sum(row.count("P") for row in ascii_map)
        if not 1 <= num_agents <= n_spawn:
            raise ValueError(f"map {map_name!r} has {n_spawn} spawn points; num_agents must be in [1, {n_spawn}]")
        self.map_name = map_name
        self.num_rounds = int(num_rounds)
        self.view_size = full_view_size(ascii_map) if view_size is None else int(view_size)
        self.obs_kind = obs
        self.render_mode = render_mode
        self.possible_agents = [f"player_{i}" for i in range(num_agents)]
        self.agents: list[str] = []
        self._ssd_ids = [f"agent-{i}" for i in range(num_agents)]
        self._agent_chars = [str(i + 1) for i in range(num_agents)]  # how the SSD map marks each agent
        self.ssd = SSDCleanup(
            ascii_map=ascii_map,
            num_agents=num_agents,
            render=False,
            shuffle_spawn=shuffle_spawn,
            global_ref_point=None,  # egocentric window (LIO); the map is fully inside it, see view_size
            view_size=self.view_size,
            random_orientation=False,  # everyone faces up: the beam fires upward, cleaning needs the river columns
            cleanup_params=CLEANUP_PARAMS[map_name],
            beam_width=3,
        )
        self.round = 0
        self._reset_episode_stats()
        # colour table for obs="rgb": character code -> RGB (agents are recoloured per observer)
        self._palette = np.zeros((256, 3), dtype=np.uint8)
        for ch, col in {**DEFAULT_COLOURS, **CLEANUP_COLORS}.items():
            if len(ch) == 1:
                self._palette[ord(ch)] = col

    # ---- spaces ---------------------------------------------------------------------------------
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        side = 2 * self.view_size + 1
        if self.obs_kind == "rgb":
            return spaces.Box(0, 255, shape=(side, side, 3), dtype=np.uint8)
        return spaces.Box(0.0, 1.0, shape=(side, side, len(PLANE_NAMES)), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return spaces.Discrete(len(SSD_ACTIONS))

    # ---- helpers --------------------------------------------------------------------------------
    def _reset_episode_stats(self) -> None:
        n = len(self.possible_agents)
        self.apples = np.zeros(n, dtype=np.int64)
        self.clean_actions = np.zeros(n, dtype=np.int64)
        self.waste_cleaned = np.zeros(n, dtype=np.int64)
        self.waste_density_sum = 0.0
        self.apple_prob_sum = 0.0

    def waste_density(self) -> float:
        """Polluted fraction of the river cells (the quantity the spawn probabilities depend on)."""
        return 1.0 - self.ssd.compute_permitted_area() / self.ssd.potential_waste_area

    def _observe(self, view: np.ndarray, player: int) -> np.ndarray:
        """Character window (from the SSD env, centred on the agent) -> planes or recoloured RGB."""
        others = [ch for j, ch in enumerate(self._agent_chars) if j != player]
        if self.obs_kind == "rgb":
            codes = np.frombuffer(view.astype("U1").tobytes(), dtype=np.uint32).reshape(view.shape)  # code points
            rgb = self._palette[np.minimum(codes, 255)].copy()
            rgb[view == self._agent_chars[player]] = SELF_COLOUR
            rgb[np.isin(view, others)] = OTHERS_COLOUR
            return rgb
        planes = np.zeros(view.shape + (len(PLANE_NAMES),), dtype=np.float32)
        planes[..., SELF] = view == self._agent_chars[player]
        planes[..., OTHERS] = np.isin(view, others)
        for ch, k in CELL_PLANES.items():
            planes[..., k] += view == ch  # 'R' and 'S' share the river plane
        return planes

    # ---- PettingZoo API -------------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.ssd.seed(seed)
        self.agents = self.possible_agents[:]
        self.round = 0
        self._reset_episode_stats()
        views = self.ssd.reset()
        observations = {agent: self._observe(views[self._ssd_ids[i]], i) for i, agent in enumerate(self.agents)}
        infos: dict[str, dict[str, Any]] = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, int]):
        if not self.agents:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        ssd_actions = {self._ssd_ids[i]: SSD_ACTIONS[int(actions[agent])] for i, agent in enumerate(self.agents)}
        views, rewards, _, info = self.ssd.step(ssd_actions)
        for i, agent in enumerate(self.agents):
            if int(actions[agent]) == CLEAN:
                self.clean_actions[i] += 1
        # n_cleaned_each_agent follows the order of the action dict we passed (player order)
        self.waste_cleaned += np.asarray(info.get("n_cleaned_each_agent", [0] * len(self.agents)), dtype=np.int64)
        r = np.array([float(rewards[self._ssd_ids[i]]) for i in range(len(self.agents))])
        self.apples += (r > 0).astype(np.int64)
        self.waste_density_sum += self.waste_density()
        self.apple_prob_sum += float(self.ssd.current_apple_spawn_prob)

        self.round += 1
        truncated = self.round >= self.num_rounds
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos: dict[str, dict[str, Any]] = {agent: {} for agent in self.agents}
        if truncated:
            for i, agent in enumerate(self.agents):
                infos[agent]["episode_stats"] = {
                    "apples": float(self.apples[i]),
                    "clean_actions": float(self.clean_actions[i]),
                    "waste_cleaned": float(self.waste_cleaned[i]),
                    "waste_density": self.waste_density_sum / self.round,
                    "apple_prob": self.apple_prob_sum / self.round,
                }
        observations = {agent: self._observe(views[self._ssd_ids[i]], i) for i, agent in enumerate(self.agents)}
        reward_dict = {agent: float(r[i]) for i, agent in enumerate(self.agents)}
        if truncated:
            self.agents = []
        return observations, reward_dict, terminations, truncations, infos

    def render(self):
        if self.render_mode == "rgb_array":
            return self.ssd.render()
        grid = self.ssd.get_map_with_agents()
        return f"round {self.round}:\n" + "\n".join("".join(row) for row in grid)

    def close(self):
        pass
