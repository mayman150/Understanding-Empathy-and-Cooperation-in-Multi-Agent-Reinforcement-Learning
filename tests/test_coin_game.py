import numpy as np
import pytest

from empathy_marl.coin_game import BLUE, DOWN, LEFT, RED, RIGHT, UP, CoinGame
from empathy_marl.envs import make_envs
from empathy_marl.metrics import MultiAgentEpisodeStatistics


def _place(env: CoinGame, red, blue, coin, owner) -> None:
    env.positions = np.array([red, blue], dtype=np.int64)
    env.coin_pos = np.array(coin, dtype=np.int64)
    env.coin_owner = owner


def _planes(obs: np.ndarray, grid: int = 3) -> np.ndarray:
    return obs.reshape(4, grid, grid)


def test_taking_own_coin_is_plus_one_for_me_and_nothing_for_the_other():
    env = CoinGame(num_rounds=10)
    env.reset(seed=0)
    _place(env, red=(0, 0), blue=(2, 2), coin=(0, 1), owner=RED)
    _, rew, _, _, _ = env.step({"player_0": RIGHT, "player_1": DOWN})
    assert (rew["player_0"], rew["player_1"]) == (1.0, 0.0)
    assert env.coins_taken.tolist() == [[1, 0], [0, 0]]
    assert env.coin_owner == BLUE  # colours alternate
    assert not any(np.array_equal(env.coin_pos, p) for p in env.positions)  # new coin on a free cell


def test_taking_the_others_coin_costs_them_two():
    env = CoinGame(num_rounds=10)
    env.reset(seed=0)
    _place(env, red=(0, 0), blue=(2, 2), coin=(0, 1), owner=BLUE)
    _, rew, _, _, _ = env.step({"player_0": RIGHT, "player_1": DOWN})
    assert (rew["player_0"], rew["player_1"]) == (1.0, -2.0)
    assert env.coins_taken.tolist() == [[0, 1], [0, 0]]
    assert env.coin_owner == RED


def test_simultaneous_pickup_follows_lola():
    env = CoinGame(num_rounds=10)
    env.reset(seed=0)
    _place(env, red=(0, 0), blue=(0, 2), coin=(0, 1), owner=RED)
    _, rew, _, _, _ = env.step({"player_0": RIGHT, "player_1": LEFT})
    assert (rew["player_0"], rew["player_1"]) == (-1.0, 1.0)  # owner: +1 - 2 ; other: +1
    assert env.coins_taken.tolist() == [[1, 0], [0, 1]]


def test_moves_wrap_around_and_missing_the_coin_gives_nothing():
    env = CoinGame(num_rounds=10)
    env.reset(seed=0)
    _place(env, red=(0, 0), blue=(1, 1), coin=(2, 2), owner=RED)
    _, rew, _, _, _ = env.step({"player_0": UP, "player_1": LEFT})
    assert env.positions[0].tolist() == [2, 0] and env.positions[1].tolist() == [1, 0]
    assert rew == {"player_0": 0.0, "player_1": 0.0}
    assert env.coin_pos.tolist() == [2, 2] and env.coin_owner == RED  # untouched coin stays
    env.step({"player_0": LEFT, "player_1": DOWN})
    assert env.positions[0].tolist() == [2, 2]  # (2, 0) + left -> wraps to column 2 -> picks up the coin
    assert env.coins_taken[0].tolist() == [1, 0]


def test_absolute_observation_is_agent_relative():
    env = CoinGame(num_rounds=10, egocentric=False)
    obs, _ = env.reset(seed=0)
    assert obs["player_0"].shape == (36,) and obs["player_0"].dtype == np.float32
    _place(env, red=(0, 0), blue=(2, 1), coin=(1, 2), owner=BLUE)
    obs = {agent: env._observe(i) for i, agent in enumerate(env.possible_agents)}
    p0, p1 = _planes(obs["player_0"]), _planes(obs["player_1"])
    # player_0 (red): own, other, own-colour coin (none: the coin is blue), other-colour coin
    assert p0[0].tolist() == [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert p0[1].tolist() == [[0, 0, 0], [0, 0, 0], [0, 1, 0]]
    assert p0[2].sum() == 0 and p0[3][1, 2] == 1.0
    # player_1 (blue) sees the same board from its side: planes 0<->1 and 2<->3 swapped
    assert np.array_equal(p1[0], p0[1]) and np.array_equal(p1[1], p0[0])
    assert np.array_equal(p1[2], p0[3]) and np.array_equal(p1[3], p0[2])
    assert p0.sum() == 3.0  # exactly three marks: two agents and one coin


def test_egocentric_observation_puts_the_observer_in_the_centre():
    env = CoinGame(num_rounds=10)  # egocentric by default
    env.reset(seed=0)
    _place(env, red=(0, 0), blue=(2, 1), coin=(1, 2), owner=BLUE)
    p0 = _planes(env._observe(0))
    p1 = _planes(env._observe(1))
    for p in (p0, p1):
        assert p[0].tolist() == [[0, 0, 0], [0, 1, 0], [0, 0, 0]]  # own position: always the centre
        assert p.sum() == 3.0
    # red: everything shifted by (+1, +1) -> blue at (0, 2), blue's coin at (2, 0) in the other-colour plane
    assert p0[1][0, 2] == 1.0 and p0[2].sum() == 0 and p0[3][2, 0] == 1.0
    # blue: shifted by (-1, 0) -> red at (2, 0), the coin (blue's own colour) at (0, 2)
    assert p1[1][2, 0] == 1.0 and p1[2][0, 2] == 1.0 and p1[3].sum() == 0
    # the same relative layout is the same observation wherever it happens on the torus
    _place(env, red=(2, 2), blue=(1, 0), coin=(0, 1), owner=BLUE)  # everything translated by (+2, +2)
    assert np.array_equal(_planes(env._observe(0)), p0) and np.array_equal(_planes(env._observe(1)), p1)


def test_episode_ends_by_truncation_with_stats():
    env = CoinGame(num_rounds=3)
    env.reset(seed=1)
    _place(env, red=(0, 0), blue=(2, 2), coin=(0, 1), owner=BLUE)
    _, _, term, trunc, infos = env.step({"player_0": RIGHT, "player_1": DOWN})  # red takes blue's coin
    assert not any(trunc.values()) and infos["player_0"] == {} and infos["player_1"] == {}
    _place(env, red=(0, 0), blue=(2, 2), coin=(0, 1), owner=RED)
    env.step({"player_0": RIGHT, "player_1": DOWN})  # red takes its own coin
    _place(env, red=(0, 0), blue=(2, 2), coin=(1, 1), owner=RED)
    _, _, term, trunc, infos = env.step({"player_0": DOWN, "player_1": DOWN})  # nobody scores
    assert all(trunc.values()) and not any(term.values()) and env.agents == []
    s0, s1 = infos["player_0"]["episode_stats"], infos["player_1"]["episode_stats"]
    assert (s0["own_coins"], s0["other_coins"]) == (1.0, 1.0) and s0["cooperation_rate"] == 0.5
    assert (s1["own_coins"], s1["other_coins"]) == (0.0, 0.0) and np.isnan(s1["cooperation_rate"])
    with pytest.raises(RuntimeError):
        env.step({"player_0": UP, "player_1": UP})
    obs, infos = env.reset(seed=2)  # counters restart
    assert env.coins_taken.sum() == 0 and infos == {"player_0": {}, "player_1": {}}
    assert env.round == 0 and len(env.agents) == 2


def test_random_play_invariants():
    env = CoinGame(num_rounds=200)
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    owners, pickups = [], 0
    for _ in range(200):
        owner_before, pos_before = env.coin_owner, env.coin_pos.copy()
        _, rew, _, trunc, _ = env.step({"player_0": int(rng.integers(4)), "player_1": int(rng.integers(4))})
        assert all(0 <= p < 3 for p in env.positions.reshape(-1))
        landed = sum(np.array_equal(p, pos_before) for p in env.positions)  # 0, 1 or 2 agents on the coin
        if landed:
            pickups += landed
            assert env.coin_owner == 1 - owner_before  # alternating colours
            assert not any(np.array_equal(env.coin_pos, p) for p in env.positions)
            owners.append(owner_before)
        else:
            assert env.coin_owner == owner_before and np.array_equal(env.coin_pos, pos_before)
            assert rew == {"player_0": 0.0, "player_1": 0.0}
        assert sum(rew.values()) in (-1.0, 0.0, 1.0)  # per coin: own colour +1, other's colour -1
    assert pickups > 10 and env.coins_taken.sum() == pickups
    assert all(trunc.values())
    with pytest.raises(ValueError):
        CoinGame(grid_size=1)
    with pytest.raises(ValueError):
        CoinGame(num_rounds=0)


def test_vectorised_coin_game_layout_and_stats_pass_through_the_metrics_wrapper():
    b = make_envs("coin", num_envs=2, max_cycles=4)
    assert b.num_agents == 2 and b.envs.num_envs == 4 and b.obs_type == "vector"
    assert b.single_observation_space.shape == (36,) and b.single_action_space.n == 4
    assert b.cooperate_action is None
    envs = MultiAgentEpisodeStatistics(b.envs, num_envs=2, num_agents=2)
    obs, _ = envs.reset(seed=0)
    assert b.extract_obs(obs).shape == (4, 36)
    for t in range(4):
        obs, rew, _, trunc, infos = envs.step(np.array([RIGHT, LEFT, UP, DOWN]))
        if t < 3:
            assert not any("ma_episode" in i for i in infos)
    assert trunc.all()
    ep0, ep1 = infos[0]["ma_episode"], infos[2]["ma_episode"]
    for ep in (ep0, ep1):
        assert set(ep["stats"]) == {"own_coins", "other_coins", "cooperation_rate"}
        assert ep["stats"]["own_coins"].shape == (2,)
        assert ep["l"] == 4 and ep["collective"] == pytest.approx(ep["stats"]["own_coins"].sum() - ep["stats"]["other_coins"].sum())
    assert b.extract_terminal_obs(infos).shape == (2, 2, 36)
    assert b.extract_obs(obs).shape == (4, 36)  # auto-reset produced fresh observations
