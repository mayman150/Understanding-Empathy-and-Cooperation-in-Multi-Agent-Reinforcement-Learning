import numpy as np
import pytest
import torch

from empathy_marl.envs import make_envs
from empathy_marl.metrics import MultiAgentEpisodeStatistics, gini
from empathy_marl.prisoners_dilemma import COOPERATE, DEFECT, RepeatedPrisonersDilemma


def test_pd_payoff_matrix():
    env = RepeatedPrisonersDilemma(num_rounds=3, payoffs=(3.0, 0.0, 4.0, 1.0))
    obs, _ = env.reset(seed=0)
    assert obs["player_0"].tolist() == [0, 0, 0, 0, 1]  # first-round flag
    _, rew, _, _, _ = env.step({"player_0": COOPERATE, "player_1": DEFECT})
    assert (rew["player_0"], rew["player_1"]) == (0.0, 4.0)
    obs, rew, _, _, _ = env.step({"player_0": DEFECT, "player_1": DEFECT})
    assert (rew["player_0"], rew["player_1"]) == (1.0, 1.0)
    assert obs["player_0"].tolist() == [0, 1, 0, 1, 0]  # own D, other D
    _, rew, term, trunc, _ = env.step({"player_0": COOPERATE, "player_1": COOPERATE})
    assert (rew["player_0"], rew["player_1"]) == (3.0, 3.0)
    assert not any(term.values()) and all(trunc.values())
    with pytest.raises(ValueError):
        RepeatedPrisonersDilemma(payoffs=(1.0, 0.0, 4.0, 3.0))  # violates T > R > P > S


def test_n_player_pd_averages_pairwise_payoffs_and_observes_fraction_of_others():
    env = RepeatedPrisonersDilemma(num_rounds=5, payoffs=(3.0, 0.0, 4.0, 1.0), num_players=4)
    env.reset(seed=0)
    # players 0,1 cooperate; players 2,3 defect
    obs, rew, _, _, _ = env.step({"player_0": COOPERATE, "player_1": COOPERATE, "player_2": DEFECT, "player_3": DEFECT})
    # cooperator: others = (C, D, D) -> (3 + 0 + 0) / 3 = 1 ; defector: others = (C, C, D) -> (4 + 4 + 1) / 3 = 3
    assert rew["player_0"] == pytest.approx(1.0) and rew["player_1"] == pytest.approx(1.0)
    assert rew["player_2"] == pytest.approx(3.0) and rew["player_3"] == pytest.approx(3.0)
    # obs: own one-hot, fraction of the *others* that cooperated / defected, first-round flag
    assert obs["player_0"].tolist() == pytest.approx([1, 0, 1 / 3, 2 / 3, 0])
    assert obs["player_2"].tolist() == pytest.approx([0, 1, 2 / 3, 1 / 3, 0])
    # all cooperate beats all defect, defecting still dominates
    _, rew_c, _, _, _ = env.step({a: COOPERATE for a in env.possible_agents})
    _, rew_d, _, _, _ = env.step({a: DEFECT for a in env.possible_agents})
    assert all(r == 3.0 for r in rew_c.values()) and all(r == 1.0 for r in rew_d.values())
    with pytest.raises(ValueError):
        RepeatedPrisonersDilemma(num_players=1)
    assert make_envs("pd", num_envs=1, num_agents=3).num_agents == 3


def test_vectorised_pd_layout_is_env_major_and_terminal_obs_is_extracted():
    b = make_envs("pd", num_envs=2, max_cycles=2)
    assert b.num_agents == 2 and b.envs.num_envs == 4
    b.envs.reset(seed=3)
    # env0: (C, D)  env1: (D, D)
    _, rew, _, trunc, infos = b.envs.step(np.array([COOPERATE, DEFECT, DEFECT, DEFECT]))
    assert rew.reshape(2, 2).tolist() == [[0.0, 4.0], [1.0, 1.0]]
    assert b.extract_terminal_obs(infos) is None
    obs, rew, _, trunc, infos = b.envs.step(np.array([COOPERATE, COOPERATE, DEFECT, DEFECT]))
    assert trunc.all()
    tobs = b.extract_terminal_obs(infos)
    assert tobs.shape == (2, 2, 5)
    assert tobs[0, 0].tolist() == [1, 0, 1, 0, 0]  # terminal obs: both cooperated in env 0
    assert tobs[1, 1].tolist() == [0, 1, 0, 1, 0]
    assert b.extract_obs(obs)[0].tolist() == [0, 0, 0, 0, 1]  # auto-reset -> fresh episode


def test_gini():
    assert gini(np.array([1.0, 1.0, 1.0])) == 0.0
    assert gini(np.array([0.0, 0.0, 4.0])) == pytest.approx(2 / 3)
    assert gini(np.array([0.0, 0.0])) == 0.0


def test_metrics_wrapper_per_env_bookkeeping_and_reset():
    b = make_envs("pd", num_envs=2, max_cycles=3)
    envs = MultiAgentEpisodeStatistics(b.envs, num_envs=2, num_agents=2)
    envs.reset(seed=0)
    # env0 always (C, D): returns (0, 12); positive reward only for player_1 at t=1,2,3 -> S = 2
    # env1: (D, D), (C, C), (C, C): returns (7, 7); positive at all t -> S = 2
    plan = [
        [COOPERATE, DEFECT, DEFECT, DEFECT],
        [COOPERATE, DEFECT, COOPERATE, COOPERATE],
        [COOPERATE, DEFECT, COOPERATE, COOPERATE],
    ]
    for t, acts in enumerate(plan):
        _, _, _, _, infos = envs.step(np.array(acts))
        if t < 2:
            assert not any("ma_episode" in i for i in infos)
    ep0, ep1 = infos[0]["ma_episode"], infos[2]["ma_episode"]
    assert "ma_episode" not in infos[1] and "ma_episode" not in infos[3]
    assert ep0["r"].tolist() == [0.0, 12.0] and ep0["l"] == 3
    assert ep0["collective"] == 12.0 and ep0["efficiency"] == pytest.approx(4.0)
    assert ep0["equality"] == pytest.approx(1 - 0.5)  # gini of (0, 12) is 0.5
    assert ep0["sustainability"] == pytest.approx(2.0)
    assert ep1["r"].tolist() == [7.0, 7.0] and ep1["equality"] == pytest.approx(1.0)
    # counters were reset: a second episode starts from zero
    for acts in plan:
        _, _, _, _, infos = envs.step(np.array(acts))
    assert infos[0]["ma_episode"]["r"].tolist() == [0.0, 12.0]
    assert infos[0]["ma_episode"]["l"] == 3


def test_debug_image_env_has_meltingpot_like_layout():
    b = make_envs("debug:image", num_envs=1, max_cycles=4)
    assert b.obs_type == "image" and b.single_observation_space.shape == (88, 88, 3)
    obs, _ = b.envs.reset(seed=0)
    x = b.extract_obs(obs)
    assert x.shape == (b.num_agents, 88, 88, 3) and x.dtype == np.uint8
    torch.as_tensor(x)  # uint8 -> tensor without copy issues
