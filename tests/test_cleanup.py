import numpy as np
import pytest
import torch
from gymnasium import spaces

from empathy_marl.agents import GridCNN, GridRewardModel, MultiAgents
from empathy_marl.cleanup import CLEAN, DOWN, LEFT, RIGHT, STAY, UP, CleanupEnv, PLANE_NAMES, SELF, OTHERS, full_view_size
from empathy_marl.empathy import social_term
from empathy_marl.envs import make_envs
from empathy_marl.metrics import MultiAgentEpisodeStatistics
from empathy_marl.replay import RewardReplay
from empathy_marl.ssd import CLEANUP_10x10_SYM

APPLE, WASTE, RIVER, WALL, BEAM = 2, 3, 4, 5, 6
V = full_view_size(CLEANUP_10x10_SYM)  # 8 -> 17x17 window


def _env(**kw) -> CleanupEnv:
    env = CleanupEnv(num_agents=3, num_rounds=50, **kw)
    env.reset(seed=0)
    return env


def _place(env: CleanupEnv, positions) -> None:
    """Put the agents on the given (row, col) cells (any non-wall cell is legal in the SSD games)."""
    for i, pos in enumerate(positions):
        env.ssd.agents[f"agent-{i}"].set_pos(np.array(pos))


def _obs(env: CleanupEnv):
    views = {aid: env.ssd.rotate_view(agent.orientation, agent.get_state()) for aid, agent in env.ssd.agents.items()}
    for agent in env.ssd.agents.values():
        agent.grid = env.ssd.get_map_with_agents()
    return {f"player_{i}": env._observe(env.ssd.agents[f"agent-{i}"].get_state(), i) for i in range(3)}


def test_lio_configuration():
    env = _env()
    assert env.view_size == V == 8 and env.observation_space("player_0").shape == (17, 17, 7)
    assert env.action_space("player_0").n == 6
    ssd = env.ssd
    assert ssd.appleRespawnProbability == 0.3 and ssd.thresholdDepletion == 0.4 and ssd.wasteSpawnProbability == 0.5
    assert all(agent.orientation == "UP" for agent in ssd.agents.values())
    # the episode starts polluted above the threshold: nothing grows until someone cleans
    assert env.waste_density() == 0.5 and ssd.current_apple_spawn_prob == 0
    grid = ssd.get_map_with_agents()
    assert (grid == "A").sum() == 0 and (grid == "H").sum() == 8 and (grid == "R").sum() == 8


def test_whole_map_visible_from_every_position_and_self_at_the_centre():
    env = _env()
    # every non-wall cell can be stood on (the beam / river columns included); from each the window holds all 100 cells
    for r in range(1, 9):
        for c in range(1, 9):
            _place(env, [(r, c), (1, 6) if (r, c) != (1, 6) else (1, 5), (8, 3) if (r, c) != (8, 3) else (8, 4)])
            obs = _obs(env)["player_0"]
            assert obs[V, V, SELF] == 1.0 and obs[..., SELF].sum() == 1.0
            assert obs[..., WALL].sum() == 36  # 10x10 map: the full ring of walls is inside the window
            assert obs[..., OTHERS].sum() == 2.0
            hidden = 1 if env.ssd.world_map[r, c] in ("H", "R") else 0  # an agent standing on a river cell covers it
            assert obs[..., WASTE].sum() + obs[..., RIVER].sum() == 16 - hidden
    # LIO's view size 7 loses the far wall row for agents on the edge rows
    small = CleanupEnv(num_agents=3, view_size=7)
    small.reset(seed=0)
    _place(small, [(1, 4), (5, 4), (8, 4)])
    assert _obs(small)["player_0"][..., WALL].sum() < 36
    assert full_view_size(["@@@@@", "@   @", "@@@@@"]) == 3


def test_observation_swap_is_the_same_world_from_the_others_seat():
    env = _env()
    _place(env, [(3, 5), (6, 2), (2, 8)])
    obs = _obs(env)
    a, b = obs["player_0"], obs["player_1"]
    # both windows are centred on their observer ...
    assert a[V, V, SELF] == 1.0 and b[V, V, SELF] == 1.0
    # ... and contain the same world, shifted by the difference of positions: (3,5) -> (6,2) is (+3, -3)
    dr, dc = 6 - 3, 2 - 5
    for plane in (APPLE, WASTE, RIVER, WALL):
        assert np.array_equal(a[dr:, : 17 + dc, plane], b[: 17 - dr, -dc:, plane])
    # in player_1's window, player_0 is one of the "others", at the mirrored offset
    assert b[V - dr, V - dc, OTHERS] == 1.0 and a[V + dr, V + dc, OTHERS] == 1.0


def test_cleaning_beam_removes_waste_and_apples_then_spawn():
    env = _env()
    _place(env, [(4, 2), (5, 6), (7, 6)])  # agent 0 on a river cell below the waste cells of row 3
    before = env.waste_density()
    _, rew, _, _, _ = env.step({"player_0": CLEAN, "player_1": STAY, "player_2": STAY})
    # the 3-wide upward beam cleans the first waste cell it meets in columns 1 and 2 (column 3 has none)
    assert env.waste_cleaned.tolist() == [2, 0, 0] and env.clean_actions.tolist() == [1, 0, 0]
    assert env.waste_density() < before and rew == {"player_0": 0.0, "player_1": 0.0, "player_2": 0.0}
    grid = env.ssd.get_map_with_agents()
    assert grid[3, 1] != "H" and grid[3, 2] != "H"
    # cleaning below the threshold switches apple spawning on (density 6/16 = 0.375 < 0.4)
    assert env.ssd.current_apple_spawn_prob == pytest.approx(0.3 * (1 - 0.375 / 0.4))
    obs = _obs(env)["player_0"]
    assert obs[..., BEAM].sum() > 0  # the beam is drawn in the observation of the step it was fired


def test_eating_an_apple_is_plus_one():
    env = _env()
    _place(env, [(2, 6), (5, 4), (8, 3)])
    env.ssd.world_map[2, 7] = "A"
    _, rew, _, _, _ = env.step({"player_0": RIGHT, "player_1": STAY, "player_2": STAY})
    assert rew["player_0"] == 1.0 and env.apples.tolist() == [1, 0, 0]
    assert env.ssd.agents["agent-0"].get_pos().tolist() == [2, 7] and env.ssd.world_map[2, 7] != "A"


def test_moves_walls_and_action_mapping():
    env = _env()
    _place(env, [(1, 3), (5, 4), (8, 3)])
    env.step({"player_0": UP, "player_1": STAY, "player_2": STAY})  # wall above: no move
    assert env.ssd.agents["agent-0"].get_pos().tolist() == [1, 3]
    env.step({"player_0": DOWN, "player_1": LEFT, "player_2": RIGHT})
    assert env.ssd.agents["agent-0"].get_pos().tolist() == [2, 3]
    assert env.ssd.agents["agent-1"].get_pos().tolist() == [5, 3]
    assert env.ssd.agents["agent-2"].get_pos().tolist() == [8, 4]


def test_episode_length_stats_and_seeding():
    env = _env()
    rng = np.random.default_rng(0)
    for t in range(50):
        acts = {a: int(rng.integers(6)) for a in env.possible_agents}
        obs, rew, term, trunc, infos = env.step(acts)
        assert set(obs) == set(env.possible_agents) if t < 49 else True
    assert all(trunc.values()) and env.agents == []
    stats = infos["player_0"]["episode_stats"]
    assert set(stats) == {"apples", "clean_actions", "waste_cleaned", "waste_density", "apple_prob"}
    assert 0.0 <= stats["waste_density"] <= 1.0 and 0.0 <= stats["apple_prob"] <= 0.3
    with pytest.raises(RuntimeError):
        env.step(acts)
    # same seed + same actions -> same trajectory
    a, b = CleanupEnv(num_agents=3), CleanupEnv(num_agents=3)
    oa, _ = a.reset(seed=7)
    ob, _ = b.reset(seed=7)
    assert all(np.array_equal(oa[k], ob[k]) for k in oa)
    for _ in range(10):
        acts = {k: int(rng.integers(6)) for k in a.possible_agents}
        oa, ra, *_ = a.step(acts)
        ob, rb, *_ = b.step(acts)
        assert ra == rb and all(np.array_equal(oa[k], ob[k]) for k in oa)


def test_rgb_observation_and_fixed_spawn():
    env = CleanupEnv(num_agents=3, obs="rgb")
    obs, _ = env.reset(seed=0)
    o = obs["player_2"]
    assert o.shape == (17, 17, 3) and o.dtype == np.uint8
    assert o[V, V].tolist() == [159, 67, 255]  # the observer is always drawn in the self colour ...
    others = [tuple(px) for px in o.reshape(-1, 3) if tuple(px) == (2, 81, 154)]
    assert len(others) == 2  # ... and the two others in the fixed "others" colour
    fixed = CleanupEnv(num_agents=3, shuffle_spawn=False)
    fixed.reset(seed=3)
    assert fixed.ssd.agent_pos == [[8, 3], [5, 4], [1, 6]]  # LIO: agent 0 on the river side


def test_bad_configurations_are_rejected():
    with pytest.raises(ValueError):
        CleanupEnv(num_agents=4, map_name="10x10")  # three spawn points
    with pytest.raises(ValueError):
        CleanupEnv(map_name="nope")
    with pytest.raises(ValueError):
        CleanupEnv(obs="pixels")


def test_vectorised_bundle_and_episode_statistics():
    bundle = make_envs("cleanup", num_envs=2, max_cycles=5, num_agents=3)
    assert bundle.obs_type == "grid" and bundle.num_agents == 3
    assert bundle.single_observation_space.shape == (17, 17, 7) and bundle.single_action_space.n == 6
    envs = MultiAgentEpisodeStatistics(bundle.envs, 2, 3)
    obs, _ = envs.reset(seed=0)
    assert bundle.extract_obs(obs).shape == (6, 17, 17, 7)
    for t in range(5):
        obs, rewards, term, trunc, infos = envs.step(np.full(6, STAY))
    assert "ma_episode" in infos[0] and "ma_episode" in infos[3]
    ep = infos[0]["ma_episode"]
    assert ep["l"] == 5 and set(ep["stats"]) == {"apples", "clean_actions", "waste_cleaned", "waste_density", "apple_prob"}
    assert ep["stats"]["waste_density"].shape == (3,)


def test_grid_networks_shapes_and_scaling():
    torch.manual_seed(0)
    planes = spaces.Box(0.0, 1.0, (17, 17, 7), np.float32)
    rgb = spaces.Box(0, 255, (17, 17, 3), np.uint8)
    for space in (planes, rgb):
        agents = MultiAgents(3, space, spaces.Discrete(6), "grid", reward_model=True)
        x = torch.as_tensor(np.stack([space.sample() for _ in range(3 * 4)])).reshape(4, 3, *space.shape)
        a, lp, ent, v, st = agents.get_action_and_value(x)
        assert a.shape == (4, 3) and v.shape == (4, 3) and st is None
        V_, _ = agents.cross_values(x)
        assert V_.shape == (4, 3, 3)
        assert torch.allclose(V_.diagonal(dim1=1, dim2=2), v, atol=1e-5)
        assert agents.cross_rewards(x, a, x).shape == (4, 3, 3)
        assert isinstance(agents.agents[0].trunk, GridCNN) and isinstance(agents.agents[0].reward_model, GridRewardModel)
    assert MultiAgents(2, rgb, spaces.Discrete(6), "grid").agents[0].trunk.scale == pytest.approx(1 / 255)
    assert MultiAgents(2, planes, spaces.Discrete(6), "grid").agents[0].trunk.scale == 1.0


def test_grid_reward_model_learns_a_local_event():
    """The model must pick up 'the apple under my cell disappeared' from the (o, o') pair."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = GridRewardModel((9, 9, 3), num_actions=2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    def batch(n):
        x = (rng.random((n, 9, 9, 3)) < 0.1).astype(np.float32)
        x[:, 4, 4, 0] = 1.0  # self at the centre
        x[:, 4, 4, 1] = 0.0
        eat = rng.random(n) < 0.5
        x[eat, 4, 4, 1] = 1.0  # an apple under me
        xn = x.copy()
        xn[:, 4, 4, 1] = 0.0  # gone afterwards
        a = rng.integers(2, size=n)
        return torch.as_tensor(x), torch.as_tensor(a), torch.as_tensor(xn), torch.as_tensor(eat.astype(np.float32))

    for _ in range(300):
        x, a, xn, r = batch(64)
        loss = ((model(x, a, xn) - r) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    x, a, xn, r = batch(256)
    assert ((model(x, a, xn) - r) ** 2).mean().item() < 0.02


def test_social_term_aggregate_sum_vs_mean():
    z = torch.tensor([[[1.0, 2.0, 4.0], [0.0, 1.0, 0.0], [3.0, 0.0, 2.0]]])  # z[0, i, j]
    alpha, beta, phi = torch.ones(3), torch.full((3,), 0.5), torch.zeros(3)
    mean = social_term("ei", z, alpha, beta, phi, "mean")
    total = social_term("ei", z, alpha, beta, phi, "sum")
    assert torch.allclose(total, 2 * mean)  # N - 1 = 2 others
    assert mean[0, 0].item() == pytest.approx((2 + 4) / 2)
    ia_mean = social_term("ia", z, alpha, beta, phi, "mean")
    ia_sum = social_term("ia", z, alpha, beta, phi, "sum")
    assert torch.allclose(ia_sum, 2 * ia_mean)
    # agent 0: others 2 and 4 vs own 1 -> disadvantageous (1 + 3) / 2 = 2, no advantageous term
    assert ia_mean[0, 0].item() == pytest.approx(-2.0)
    z2 = z[:, :2, :2]
    assert torch.allclose(social_term("svo", z2, alpha[:2], beta[:2], phi[:2], "sum"),
                          social_term("svo", z2, alpha[:2], beta[:2], phi[:2], "mean"))  # identical for N = 2
    with pytest.raises(ValueError):
        social_term("ei", z, alpha, beta, phi, "median")


def test_reward_replay_keeps_rewarded_transitions_only():
    torch.manual_seed(0)
    T, E, N = 4, 2, 2
    obs = torch.rand(T, E, N, 5)
    nxt = torch.rand(T, E, N, 5)
    acts = torch.randint(0, 3, (T, E, N))
    rew = torch.zeros(T, E, N)
    rew[0, 0, 0] = 1.0
    rew[2, 1, 0] = -2.0
    rew[3, 0, 1] = 1.0
    replay = RewardReplay(capacity=3, num_agents=N, obs_shape=(5,), obs_dtype=torch.float32, device=torch.device("cpu"))
    replay.add_rollout(obs, acts, nxt, rew)
    assert replay.size == [2, 1]
    assert set(replay.rewards[0, :2].tolist()) == {1.0, -2.0} and replay.rewards[1, 0].item() == 1.0
    assert torch.equal(replay.obs[1, 0], obs[3, 0, 1])
    for _ in range(3):  # FIFO with capacity 3: the buffer wraps around and stays bounded
        replay.add_rollout(obs, acts, nxt, rew)
    assert replay.size == [3, 3]
    agents = MultiAgents(N, spaces.Box(0.0, 1.0, (5,), np.float32), spaces.Discrete(3), "vector", reward_model=True)
    loss = replay.loss(agents, sample_size=8)
    assert loss.shape == (N,) and (loss >= 0).all() and loss.requires_grad
    empty = RewardReplay(4, N, (5,), torch.float32, torch.device("cpu"))
    assert torch.equal(empty.loss(agents, 8), torch.zeros(N))
