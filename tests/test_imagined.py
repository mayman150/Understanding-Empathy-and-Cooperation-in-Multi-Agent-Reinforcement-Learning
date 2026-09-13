"""``--signal imagined``: reward model, cross rewards, cross GAE, imagined shaper, standardisation."""
import numpy as np
import pytest
import torch
from gymnasium import spaces

from empathy_marl.agents import Agent, MultiAgents
from empathy_marl.coin_game import BLUE, RED, RIGHT, DOWN, CoinGame
from empathy_marl.empathy import ImaginedRewardShaper, RewardSocialShaper, standardize
from train import cross_gae

OBS = spaces.Box(0.0, 1.0, shape=(36,), dtype=np.float32)
ACT = spaces.Discrete(4)


def _random_play(steps: int, seed: int):
    """Transitions of both Coin Game agents under random play: obs, action, next_obs, reward -> (steps, 2, ...)."""
    env = CoinGame(num_rounds=50)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    o, a, o2, r = [], [], [], []
    for _ in range(steps):
        acts = {p: int(rng.integers(4)) for p in env.possible_agents}
        nobs, rew, _, trunc, _ = env.step(acts)
        o.append([obs["player_0"], obs["player_1"]])
        a.append([acts["player_0"], acts["player_1"]])
        o2.append([nobs["player_0"], nobs["player_1"]])
        r.append([rew["player_0"], rew["player_1"]])
        obs = nobs
        if all(trunc.values()):
            obs, _ = env.reset()
    f = lambda x: torch.as_tensor(np.array(x), dtype=torch.float32)  # noqa: E731
    return f(o), torch.as_tensor(np.array(a)), f(o2), f(r)


def test_reward_model_trained_on_own_rewards_predicts_the_others_rewards():
    """The premise of --signal imagined: a model of MY reward, trained on MY transitions, applied to the OTHER agent's
    (observation, action, next observation) reproduces the other's rewards, which the model has never seen."""
    torch.manual_seed(0)
    o, a, o2, r = _random_play(8000, seed=1)
    ho, ha, ho2, hr = _random_play(3000, seed=7)  # held out
    agent = Agent(OBS, ACT, "vector", reward_model=True)
    opt = torch.optim.Adam(agent.reward_model.parameters(), lr=3e-3)
    for _ in range(1500):  # regression on player_0's OWN transitions only
        idx = torch.randint(0, o.shape[0], (512,))
        loss = ((agent.predict_reward(o[idx, 0], a[idx, 0], o2[idx, 0]) - r[idx, 0]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        own = agent.predict_reward(ho[:, 0], ha[:, 0], ho2[:, 0])
        imagined = agent.predict_reward(ho[:, 1], ha[:, 1], ho2[:, 1])  # player_0's model on player_1's transitions
    assert (own - hr[:, 0]).abs().mean() < 0.1
    corr = torch.corrcoef(torch.stack([imagined, hr[:, 1]]))[0, 1]
    assert corr > 0.97, corr
    assert (imagined - hr[:, 1]).abs().mean() < 0.1
    # the two events that matter: player_1 collected (+1) and player_0 stole player_1's coin (-2)
    assert imagined[hr[:, 1] == 1.0].mean() > 0.85
    assert imagined[hr[:, 1] == -2.0].mean() < -1.7
    assert imagined[hr[:, 1] == 0.0].abs().mean() < 0.1


def test_cross_rewards_layout_and_own_rewards():
    torch.manual_seed(0)
    agents = MultiAgents(3, OBS, ACT, "vector", reward_model=True)
    x, x2 = torch.rand(5, 3, 36), torch.rand(5, 3, 36)
    act = torch.randint(0, 4, (5, 3))
    cross = agents.cross_rewards(x, act, x2)
    assert cross.shape == (5, 3, 3)
    for i in range(3):
        for j in range(3):
            ref = agents.agents[i].predict_reward(x[:, j], act[:, j], x2[:, j])
            assert torch.allclose(cross[:, i, j], ref, atol=1e-6)
    own = agents.predict_own_rewards(x, act, x2)
    assert torch.allclose(own, torch.diagonal(cross, dim1=1, dim2=2), atol=1e-6)
    assert own.requires_grad  # trainable


def test_reward_model_is_optional_and_does_not_duplicate_the_trunk():
    plain = MultiAgents(2, OBS, ACT, "vector")
    with_rm = MultiAgents(2, OBS, ACT, "vector", reward_model=True)
    assert set(plain.state_dict()) < set(with_rm.state_dict())
    assert all("reward_model" in k for k in set(with_rm.state_dict()) - set(plain.state_dict()))
    with pytest.raises(RuntimeError):
        plain.agents[0].predict_reward(torch.zeros(1, 36), torch.zeros(1, dtype=torch.long), torch.zeros(1, 36))
    # image observations: the head uses the agent's trunk features but the trunk is not registered twice
    img_space = spaces.Box(0, 255, shape=(88, 88, 3), dtype=np.uint8)
    img_plain = Agent(img_space, ACT, "image")
    img_rm = Agent(img_space, ACT, "image", reward_model=True)
    head = sum(p.numel() for p in img_rm.reward_model.parameters())
    assert sum(p.numel() for p in img_rm.parameters()) == sum(p.numel() for p in img_plain.parameters()) + head
    assert not any(k.startswith("reward_model._features") for k in img_rm.state_dict())
    x = torch.randint(0, 255, (2, 88, 88, 3), dtype=torch.uint8)
    out = img_rm.predict_reward(x, torch.tensor([0, 3]), x)
    out.sum().backward()
    assert all(p.grad is None for p in img_rm.trunk.parameters())  # detached features: the trunk is not trained by it


def test_cross_gae_matches_a_per_stream_reference_and_handles_episode_ends():
    torch.manual_seed(0)
    T, E, N = 7, 2, 3
    r = torch.randn(T, E, N, N)
    v = torch.randn(T, E, N, N)
    v_next = torch.randn(T, E, N, N)
    term = torch.zeros(T, E, N)
    end = torch.zeros(T, E, N)
    end[3, 0, 1] = 1.0  # observed agent 1 in env 0 ends an episode (truncation) at t = 3
    term[5, 1, 2] = 1.0  # observed agent 2 in env 1 hits a real terminal at t = 5
    end[5, 1, 2] = 1.0
    gamma, lam = 0.9, 0.8
    adv = cross_gae(r, v, v_next, term, end, gamma, lam)
    assert adv.shape == (T, E, N, N)
    for e in range(E):
        for i in range(N):
            for j in range(N):
                ref, last = torch.zeros(T), 0.0
                for t in reversed(range(T)):
                    delta = r[t, e, i, j] + gamma * v_next[t, e, i, j] * (1 - term[t, e, j]) - v[t, e, i, j]
                    last = delta + gamma * lam * (1 - end[t, e, j]) * last
                    ref[t] = last
                assert torch.allclose(adv[:, e, i, j], ref, atol=1e-5)
    # at a truncation the bootstrap is kept and the recursion is cut; at a terminal both are cut
    assert torch.allclose(adv[3, 0, :, 1], r[3, 0, :, 1] + gamma * v_next[3, 0, :, 1] - v[3, 0, :, 1])
    assert torch.allclose(adv[5, 1, :, 2], r[5, 1, :, 2] - v[5, 1, :, 2])


def test_imagined_shaper_equals_reward_shaper_when_imagination_is_exact():
    torch.manual_seed(0)
    E, N = 3, 2
    alpha, beta, phi = torch.tensor([0.5, 0.5]), torch.tensor([0.1, 0.1]), torch.tensor([0.3, 0.3])
    for formulation in ("ei", "svo", "sia", "ia"):
        ref = RewardSocialShaper(formulation, alpha, beta, phi, 0.99, 0.975, E, N, torch.device("cpu"))
        imag = ImaginedRewardShaper(formulation, alpha, beta, phi, 0.99, 0.975, E, N, torch.device("cpu"))
        for t in range(12):
            rewards = torch.randn(E, N)
            start = (torch.rand(E, N) < 0.2).float()
            start = start.max(dim=1, keepdim=True).values.expand(E, N)  # env copies end for all agents at once
            exact = rewards.unsqueeze(1).expand(E, N, N)  # every observer imagines the true rewards
            assert torch.allclose(imag(exact, start), ref(rewards, start), atol=1e-6), (formulation, t)


def test_standardize_per_index_over_batch_dims():
    x = torch.randn(50, 4, 3, 3) * torch.tensor([1.0, 5.0, 0.1]) + 7.0
    x[..., 2, 2] = 3.0  # a constant entry
    z = standardize(x, dims=(0, 1))
    assert torch.allclose(z.mean(dim=(0, 1)), torch.zeros(3, 3), atol=1e-5)
    std = z.std(dim=(0, 1), unbiased=False)
    assert torch.allclose(std[std > 0.5], torch.ones_like(std[std > 0.5]), atol=1e-4)
    assert torch.all(z[..., 2, 2] == 0.0)  # constants map to zero, not to noise


def test_coin_game_steal_gives_the_victim_minus_two_from_the_observers_seat():
    """The observation transition of the victim contains the event the reward model needs."""
    env = CoinGame(num_rounds=10)
    env.reset(seed=0)
    env.positions = np.array([[0, 0], [2, 2]])
    env.coin_pos = np.array([0, 1])
    env.coin_owner = BLUE
    before = env._observe(1)
    _, rew, _, _, _ = env.step({"player_0": RIGHT, "player_1": DOWN})
    after = env._observe(1)
    assert rew["player_1"] == -2.0 and rew["player_0"] == 1.0
    # from blue's seat: its own-colour coin (plane 2) disappeared and the other agent (plane 1) moved onto its cell
    assert before.reshape(4, 3, 3)[2].sum() == 1.0 and after.reshape(4, 3, 3)[2].sum() == 0.0


def test_imagined_shaper_value_level_adds_the_forecast_to_the_trace():
    """`trace_value`: the formulation is applied to ehat + value_level; the trace itself is unchanged."""
    torch.manual_seed(0)
    E, N = 2, 3
    alpha, beta, phi = torch.ones(N), torch.full((N,), 0.5), torch.zeros(N)
    plain = ImaginedRewardShaper("ia", alpha, beta, phi, 0.99, 0.975, E, N, torch.device("cpu"))
    with_v = ImaginedRewardShaper("ia", alpha, beta, phi, 0.99, 0.975, E, N, torch.device("cpu"))
    start = torch.zeros(E, N)
    for _ in range(5):
        imagined = torch.randn(E, N, N)
        level = torch.randn(E, N, N)
        f_plain = plain(imagined, start)
        f_v = with_v(imagined, start, level)
        assert torch.equal(plain.e, with_v.e)  # same trace
        assert not torch.allclose(f_plain, f_v)
        # equals applying IA to (trace + level) directly
        from empathy_marl.empathy import social_term

        assert torch.allclose(f_v, social_term("ia", with_v.e + level, alpha, beta, phi), atol=1e-6)
    # a zero level reproduces the plain shaper
    assert torch.allclose(with_v(imagined, start, torch.zeros(E, N, N)), plain(imagined, start), atol=1e-6)
