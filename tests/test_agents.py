"""Alignment tests: every (batch row, agent) cell must come from the right network and the
right observation.  This is the class of bug that broke the legacy scripts
(``torch.cat(...).reshape(-1, num_agents)``)."""
import numpy as np
import pytest
import torch
from gymnasium import spaces

from empathy_marl.agents import MultiAgents

torch.manual_seed(0)
OBS = spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32)
ACT = spaces.Discrete(3)


def make(num_agents=4, recurrent=False):
    torch.manual_seed(0)
    return MultiAgents(num_agents, OBS, ACT, "vector", recurrent=recurrent, lstm_hidden_size=8)


def test_feedforward_outputs_are_aligned_per_agent_and_row():
    N, M = 4, 6
    agents = make(N)
    x = torch.rand(M, N, 5)
    action = torch.randint(0, 3, (M, N))
    _, logprob, entropy, value, state = agents.get_action_and_value(x, None, None, action)
    assert logprob.shape == entropy.shape == value.shape == (M, N)
    assert state is None
    for m in range(M):
        for i in range(N):
            _, lp, ent, v, _ = agents.agents[i].get_action_and_value(x[m : m + 1, i], action=action[m : m + 1, i])
            assert torch.allclose(logprob[m, i], lp[0])
            assert torch.allclose(entropy[m, i], ent[0])
            assert torch.allclose(value[m, i], v[0])


def test_cross_values_is_agent_i_critic_on_agent_j_observation():
    N, M = 3, 5
    agents = make(N)
    x = torch.rand(M, N, 5)
    v, state = agents.cross_values(x)
    assert v.shape == (M, N, N) and state is None
    own, _ = agents.get_values(x)
    assert torch.allclose(v.diagonal(dim1=1, dim2=2), own)
    for m in range(M):
        for i in range(N):
            for j in range(N):
                vij, _ = agents.agents[i].get_value(x[m : m + 1, j])
                assert torch.allclose(v[m, i, j], vij[0])


def test_agents_have_independent_parameters():
    agents = make(3)
    p0 = agents.agents[0].critic.weight
    p1 = agents.agents[1].critic.weight
    assert not torch.equal(p0, p1)
    loss = agents.agents[0].critic(torch.rand(2, 64)).sum()
    loss.backward()
    assert agents.agents[0].critic.weight.grad is not None
    assert agents.agents[1].critic.weight.grad is None


def _rollout_lstm(agents, x_seq, dones_seq, E):
    """Step-by-step recurrent evaluation (as in the rollout phase)."""
    state = agents.initial_lstm_state(E, torch.device("cpu"))
    values, logprobs = [], []
    T = x_seq.shape[0]
    for t in range(T):
        _, lp, _, v, state = agents.get_action_and_value(x_seq[t], state, dones_seq[t], torch.zeros(E, agents.num_agents, dtype=torch.long))
        values.append(v)
        logprobs.append(lp)
    return torch.stack(values), torch.stack(logprobs), state


def test_lstm_batched_sequence_matches_step_by_step_with_resets():
    N, E, T = 3, 2, 7
    agents = make(N, recurrent=True)
    x_seq = torch.rand(T, E, N, 5)
    dones_seq = torch.zeros(T, E, N)
    dones_seq[3, 0] = 1.0  # env 0 starts a new episode at t=3
    dones_seq[5, 1] = 1.0
    step_values, step_logprobs, _ = _rollout_lstm(agents, x_seq, dones_seq, E)

    # batched, time-major flattening exactly as in train.py / CleanRL
    x_flat = x_seq.reshape(T * E, N, 5)
    d_flat = dones_seq.reshape(T * E, N)
    init = agents.initial_lstm_state(E, torch.device("cpu"))
    _, lp_flat, _, v_flat, _ = agents.get_action_and_value(x_flat, init, d_flat, torch.zeros(T * E, N, dtype=torch.long))
    assert torch.allclose(v_flat.reshape(T, E, N), step_values, atol=1e-6)
    assert torch.allclose(lp_flat.reshape(T, E, N), step_logprobs, atol=1e-6)

    # a done flag really resets the memory: value at t=3 for env 0 must equal a fresh-start value
    fresh, _ = agents.get_values(x_seq[3, 0:1], agents.initial_lstm_state(1, torch.device("cpu")), torch.zeros(1, N))
    assert torch.allclose(step_values[3, 0], fresh[0], atol=1e-6)


def test_lstm_cross_values_diagonal_matches_own_values_over_a_sequence():
    N, E, T = 3, 2, 6
    agents = make(N, recurrent=True)
    x_seq = torch.rand(T, E, N, 5)
    dones_seq = torch.zeros(T, E, N)
    dones_seq[2, 1] = 1.0
    own, _ = agents.get_values(x_seq.reshape(T * E, N, 5), agents.initial_lstm_state(E, torch.device("cpu")), dones_seq.reshape(T * E, N))
    cross, cross_state = agents.cross_values(
        x_seq.reshape(T * E, N, 5), agents.initial_cross_lstm_state(E, torch.device("cpu")), dones_seq.reshape(T * E, N)
    )
    assert cross.shape == (T * E, N, N)
    assert torch.allclose(cross.diagonal(dim1=1, dim2=2), own, atol=1e-6)
    assert cross_state[0].shape == (N, 1, E * N, 8)

    # off-diagonal: agent i's LSTM run over agent j's stream from scratch
    for i in range(N):
        for j in range(N):
            vij, _ = agents.agents[i].get_value(
                x_seq[:, :, j].reshape(T * E, 5),
                (torch.zeros(1, E, 8), torch.zeros(1, E, 8)),
                dones_seq[:, :, j].reshape(T * E),
            )
            assert torch.allclose(cross[:, i, j], vij, atol=1e-6)


@pytest.mark.parametrize("recurrent", [False, True])
def test_image_trunk_accepts_uint8_hwc(recurrent):
    obs = spaces.Box(0, 255, shape=(88, 88, 3), dtype=np.uint8)
    agents = MultiAgents(2, obs, spaces.Discrete(5), "image", recurrent=recurrent, lstm_hidden_size=16)
    x = torch.randint(0, 256, (3, 2, 88, 88, 3), dtype=torch.uint8)
    state = agents.initial_lstm_state(3, torch.device("cpu"))
    a, lp, ent, v, _ = agents.get_action_and_value(x, state, torch.zeros(3, 2))
    assert a.shape == lp.shape == ent.shape == v.shape == (3, 2)
    assert torch.isfinite(v).all()
