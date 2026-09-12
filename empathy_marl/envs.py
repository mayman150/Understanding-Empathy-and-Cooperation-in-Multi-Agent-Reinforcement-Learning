"""Environment factory.

Every environment is exposed through the same interface so that ``train.py`` does not
need to know which game it is running:

* a supersuit ``ConcatVecEnv`` whose ``num_envs`` equals ``num_envs * num_agents``.
  Entries are **env-major**: ``[env0: player_0 .. player_{N-1}, env1: player_0 ..]``,
  so a flat array of shape ``(num_envs * num_agents, ...)`` reshapes to
  ``(num_envs, num_agents, ...)``.
* ``extract_obs`` turns the raw (possibly dict) observation returned by the vec env
  into the single array the policy consumes.

Supported ``env_id`` values
---------------------------
* ``pd`` / ``prisoners_dilemma``          repeated Prisoner's Dilemma (N players, default 2)
* ``coin`` / ``coin_game``                Coin Game (Lerer & Peysakhovich 2017), 2 players on a 3x3 grid
* ``meltingpot:<substrate>``              any Melting Pot substrate through shimmy, e.g.
                                          ``meltingpot:commons_harvest__open``,
                                          ``meltingpot:clean_up``
* ``<substrate>``                         bare Melting Pot substrate name (legacy)
* ``debug:image``                         tiny 88x88 RGB toy task with the Melting Pot
                                          observation layout, for smoke tests without dmlab2d

Melting Pot agents observe their egocentric ``RGB`` window (88x88x3), i.e. the game is
partially observable from the agent's point of view.  ``WORLD.*`` observations are
stripped by shimmy.  A fully observable variant needs an agent-specific observation
(see the README) and is left as an explicit extension point (``MELTINGPOT_OBS_KEY``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import supersuit as ss
from gymnasium import spaces

from empathy_marl.coin_game import CoinGame
from empathy_marl.prisoners_dilemma import RepeatedPrisonersDilemma

PD_IDS = ("pd", "prisoners_dilemma", "repeated_prisoners_dilemma")
COIN_IDS = ("coin", "coin_game", "coingame")
DEBUG_IMAGE_ID = "debug:image"
MELTINGPOT_PREFIX = "meltingpot:"
MELTINGPOT_OBS_KEY = "RGB"


@dataclass
class EnvBundle:
    envs: Any  # supersuit ConcatVecEnv (gymnasium VectorEnv API), auto-resetting
    env_id: str
    num_envs: int
    num_agents: int
    agent_names: list[str]
    single_observation_space: spaces.Box  # what one agent's policy sees
    single_action_space: spaces.Discrete
    obs_type: str  # "image" (uint8 HWC) or "vector" (float32)
    extract_obs: Callable[[Any], np.ndarray]  # raw vec-env obs -> (num_envs*num_agents, *obs_shape)
    cooperate_action: Optional[int] = None  # for cooperation-rate logging (PD)

    def extract_terminal_obs(self, infos: list[dict]) -> Optional[np.ndarray]:
        """Stack the ``terminal_observation`` entries supersuit puts in ``infos`` at episode end.

        Returns ``None`` when no environment finished on this step, otherwise an array of
        shape ``(num_envs, num_agents, *obs_shape)`` where rows of environments that did
        not finish are zero.
        """
        if not any("terminal_observation" in info for info in infos):
            return None
        out = np.zeros(
            (self.num_envs, self.num_agents) + self.single_observation_space.shape,
            dtype=self.single_observation_space.dtype,
        )
        for e in range(self.num_envs):
            for a in range(self.num_agents):
                info = infos[e * self.num_agents + a]
                if "terminal_observation" in info:
                    out[e, a] = self._extract_single(info["terminal_observation"])
        return out

    def _extract_single(self, raw: Any) -> np.ndarray:
        if isinstance(raw, dict):
            return np.asarray(raw[MELTINGPOT_OBS_KEY])
        return np.asarray(raw)


def is_prisoners_dilemma(env_id: str) -> bool:
    return env_id.lower() in PD_IDS


def is_coin_game(env_id: str) -> bool:
    return env_id.lower() in COIN_IDS


def _vector_bundle(par_env, env_id: str, num_envs: int, num_cpus: int, cooperate_action: Optional[int]) -> EnvBundle:
    """Bundle for a ParallelEnv with flat float32 observations (PD, Coin Game)."""
    agent_names = list(par_env.possible_agents)
    envs = _vectorize(par_env, num_envs, num_cpus)
    return EnvBundle(
        envs=envs,
        env_id=env_id,
        num_envs=num_envs,
        num_agents=len(agent_names),
        agent_names=agent_names,
        single_observation_space=envs.observation_space,
        single_action_space=envs.action_space,
        obs_type="vector",
        extract_obs=lambda raw: np.asarray(raw, dtype=np.float32),
        cooperate_action=cooperate_action,
    )


def _vectorize(par_env, num_envs: int, num_cpus: int):
    venv = ss.pettingzoo_env_to_vec_env_v1(par_env)
    return ss.concat_vec_envs_v1(venv, num_vec_envs=num_envs, num_cpus=num_cpus, base_class="gymnasium")


def make_envs(
    env_id: str,
    num_envs: int = 1,
    num_cpus: int = 0,
    max_cycles: int = 1000,
    num_agents: int = 2,
    pd_payoffs: tuple[float, float, float, float] = (3.0, 0.0, 4.0, 1.0),
    render_mode: Optional[str] = None,
) -> EnvBundle:
    """``num_agents`` applies to ``pd`` and ``debug:image``; the Coin Game and Melting Pot substrates fix their own player count."""
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")

    if is_prisoners_dilemma(env_id):
        par_env = RepeatedPrisonersDilemma(
            num_rounds=max_cycles, payoffs=pd_payoffs, num_players=num_agents, render_mode=render_mode
        )
        return _vector_bundle(par_env, env_id, num_envs, num_cpus, RepeatedPrisonersDilemma.cooperate_action)

    if is_coin_game(env_id):
        par_env = CoinGame(num_rounds=max_cycles, render_mode=render_mode)
        return _vector_bundle(par_env, env_id, num_envs, num_cpus, CoinGame.cooperate_action)

    if env_id == DEBUG_IMAGE_ID:
        from empathy_marl.debug_env import DebugImageEnv

        par_env = DebugImageEnv(num_agents=num_agents, max_cycles=max_cycles, render_mode=render_mode)
    else:
        substrate = env_id[len(MELTINGPOT_PREFIX):] if env_id.startswith(MELTINGPOT_PREFIX) else env_id
        from shimmy import MeltingPotCompatibilityV0  # requires dm-meltingpot (Linux)

        par_env = MeltingPotCompatibilityV0(substrate_name=substrate, max_cycles=max_cycles, render_mode=render_mode)
    agent_names = list(par_env.possible_agents)
    envs = _vectorize(par_env, num_envs, num_cpus)
    dict_space = envs.observation_space
    if not isinstance(dict_space, spaces.Dict) or MELTINGPOT_OBS_KEY not in dict_space.spaces:
        raise RuntimeError(f"expected a Dict observation space containing '{MELTINGPOT_OBS_KEY}', got {dict_space}")
    obs_space = dict_space[MELTINGPOT_OBS_KEY]
    return EnvBundle(
        envs=envs,
        env_id=env_id,
        num_envs=num_envs,
        num_agents=len(agent_names),
        agent_names=agent_names,
        single_observation_space=obs_space,
        single_action_space=envs.action_space,
        obs_type="image",
        extract_obs=lambda raw: np.asarray(raw[MELTINGPOT_OBS_KEY]),
        cooperate_action=None,
    )
