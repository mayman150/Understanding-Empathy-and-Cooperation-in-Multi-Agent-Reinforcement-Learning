import math

import pytest
import torch

from empathy_marl.empathy import (
    FORMULATIONS,
    SIGNALS,
    RewardSocialShaper,
    parse_per_agent,
    social_term,
)

# v_next[i, j] = V_i(s_j'): three agents, hand-picked numbers
V = torch.tensor(
    [
        [1.0, 3.0, 5.0],  # agent 0: self 1, others 3, 5  -> mean 4
        [2.0, 2.0, 2.0],  # agent 1: self 2, others 2, 2  -> mean 2 (no inequity)
        [6.0, 0.0, 3.0],  # agent 2: self 3, others 6, 0  -> mean 3
    ]
)
ONES = torch.ones(3)
ZEROS = torch.zeros(3)


def test_parse_per_agent_broadcast_and_list():
    assert parse_per_agent("0.1", 3, "alpha").tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert parse_per_agent("0,0.5", 2, "alpha").tolist() == pytest.approx([0.0, 0.5])
    assert parse_per_agent(0.25, 2, "alpha").tolist() == pytest.approx([0.25, 0.25])
    with pytest.raises(ValueError):
        parse_per_agent("0,0.5,1", 2, "alpha")


def test_none_has_zero_social_term_and_constants_are_complete():
    assert torch.equal(social_term("none", V, ONES, ONES, ONES), ZEROS)
    assert FORMULATIONS == ("none", "ei", "svo", "sia", "ia")
    assert SIGNALS == ("value", "reward", "imagined")
    with pytest.raises(ValueError):
        social_term("reward_ia", V, ONES, ONES, ONES)


def test_ei_is_alpha_times_mean_of_others():
    alpha = torch.tensor([1.0, 2.0, 0.5])
    x = social_term("ei", V, alpha, ZEROS, ZEROS)
    assert x.tolist() == pytest.approx([1.0 * 4.0, 2.0 * 2.0, 0.5 * 3.0])


def test_svo_mixes_self_and_others_by_angle():
    phi = torch.tensor([0.0, math.pi / 2, math.pi / 4])
    x = social_term("svo", V, ONES, ZEROS, phi)
    expected = [1.0 * 1.0 + 0.0, 0.0 + 1.0 * 2.0, math.cos(math.pi / 4) * 3.0 + math.sin(math.pi / 4) * 3.0]
    assert x.tolist() == pytest.approx(expected, abs=1e-6)


def test_sia_penalises_absolute_gap_to_mean():
    x = social_term("sia", V, torch.tensor([1.0, 1.0, 2.0]), ZEROS, ZEROS)
    assert x.tolist() == pytest.approx([-abs(1 - 4), -abs(2 - 2), -2 * abs(3 - 3)])


def test_ia_splits_disadvantageous_and_advantageous_inequity():
    alpha = torch.tensor([1.0, 1.0, 1.0])
    beta = torch.tensor([0.5, 0.5, 0.5])
    x = social_term("ia", V, alpha, beta, ZEROS)
    # agent 0: others 3,5 vs self 1 -> disadvantageous (2+4)/2 = 3, advantageous 0
    # agent 1: no inequity
    # agent 2: others 6,0 vs self 3 -> disadvantageous 3/2 = 1.5, advantageous 3/2 = 1.5
    assert x.tolist() == pytest.approx([-3.0, 0.0, -1.0 * 1.5 - 0.5 * 1.5])


def test_social_term_broadcasts_over_leading_dims_and_is_detached():
    v = V.unsqueeze(0).unsqueeze(0).repeat(4, 2, 1, 1).requires_grad_(True)  # (T=4, E=2, N, N)
    x = social_term("sia", v, ONES, ZEROS, ZEROS)
    assert x.shape == (4, 2, 3)
    assert not x.requires_grad
    assert torch.allclose(x[3, 1], social_term("sia", V, ONES, ZEROS, ZEROS))


@pytest.mark.parametrize("f", [f for f in FORMULATIONS if f != "none"])
def test_zero_alpha_gives_zero_term(f):
    assert torch.equal(social_term(f, V, ZEROS, ZEROS, ONES), ZEROS)


def _shaper(formulation, alpha, beta, phi=(0.0, 0.0), lam=0.5, num_agents=2):
    return RewardSocialShaper(
        formulation,
        torch.tensor(alpha),
        torch.tensor(beta),
        torch.tensor(phi),
        gamma=0.5,
        lam=lam,
        num_envs=1,
        num_agents=num_agents,
        device=torch.device("cpu"),
    )


def test_reward_signal_ia_matches_hughes_formula_and_resets():
    shaper = _shaper("ia", alpha=[1.0, 1.0], beta=[0.5, 0.5])
    no_start = torch.zeros(1, 2)
    # step 1: rewards (4, 0): e = (4, 0). agent0 advantageous by 4 -> -0.5*4 ; agent1 disadvantageous by 4 -> -1*4
    f1 = shaper(torch.tensor([[4.0, 0.0]]), no_start)
    assert f1.tolist()[0] == pytest.approx([-2.0, -4.0])
    # step 2: decay gamma*lambda = 0.25 -> e = (1 + 0, 0 + 2) = (1, 2). agent0 disadvantaged by 1 -> -1 ; agent1 advantaged by 1 -> -0.5
    f2 = shaper(torch.tensor([[0.0, 2.0]]), no_start)
    assert f2.tolist()[0] == pytest.approx([-1.0, -0.5])
    # new episode: smoothed rewards restart from zero -> equal rewards, no inequity
    f3 = shaper(torch.tensor([[1.0, 1.0]]), torch.ones(1, 2))
    assert f3.tolist()[0] == pytest.approx([0.0, 0.0])
    assert shaper.e.tolist()[0] == pytest.approx([1.0, 1.0])


def test_reward_signal_uses_the_same_functional_form_as_the_value_signal():
    """With lambda = 0 (raw rewards) every formulation on the reward signal equals social_term on z[i, j] = r_j."""
    r = torch.tensor([[1.0, 3.0, 5.0]])
    z = r.unsqueeze(-2).expand(1, 3, 3)
    alpha, beta, phi = torch.tensor([1.0, 2.0, 0.5]), torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.0, math.pi / 2, math.pi / 4])
    for f in ("ei", "svo", "sia", "ia"):
        shaper = _shaper(f, alpha=alpha.tolist(), beta=beta.tolist(), phi=phi.tolist(), lam=0.0, num_agents=3)
        got = shaper(r, torch.zeros(1, 3))
        assert torch.allclose(got, social_term(f, z, alpha, beta, phi)), f
    # ei on raw rewards is alpha * mean of the others' rewards
    assert _shaper("ei", [1.0, 1.0, 1.0], [0.0] * 3, lam=0.0, num_agents=3)(r, torch.zeros(1, 3)).tolist()[0] == pytest.approx(
        [4.0, 3.0, 2.0]
    )
    # none never adds anything
    assert torch.equal(_shaper("none", [1.0, 1.0], [0.0, 0.0])(torch.tensor([[4.0, 0.0]]), torch.zeros(1, 2)), torch.zeros(1, 2))
