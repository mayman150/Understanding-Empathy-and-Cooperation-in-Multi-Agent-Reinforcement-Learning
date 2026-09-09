"""End-to-end smoke tests of ``train.py`` plus a unit test of the next-state cross values."""
import glob
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from gymnasium import spaces

from empathy_marl.agents import MultiAgents
from train import compute_next_cross_values

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS = spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32)


@pytest.mark.parametrize("recurrent", [False, True])
def test_next_cross_values_are_shifted_by_one_step(recurrent):
    torch.manual_seed(0)
    N, E, T = 3, 2, 9
    agents = MultiAgents(N, OBS, spaces.Discrete(2), "vector", recurrent=recurrent, lstm_hidden_size=8)
    obs = torch.rand(T, E, N, 5)
    next_obs = torch.rand(E, N, 5)
    dones = torch.zeros(T, E, N)
    dones[4, 1] = 1.0
    next_done = torch.zeros(E, N)
    state0 = agents.initial_cross_lstm_state(E, torch.device("cpu")) if recurrent else None

    v_next, new_state = compute_next_cross_values(agents, obs, next_obs, dones, next_done, state0, chunk_steps=4)
    assert v_next.shape == (T, E, N, N)

    # reference: evaluate the whole sequence (plus the final next_obs) in one go
    seq = torch.cat([obs, next_obs.unsqueeze(0)])
    dseq = torch.cat([dones, next_done.unsqueeze(0)])
    ref, _ = agents.cross_values(seq.reshape((T + 1) * E, N, 5), state0, dseq.reshape((T + 1) * E, N))
    ref = ref.reshape(T + 1, E, N, N)
    assert torch.allclose(v_next, ref[1:], atol=1e-6)  # v_next[t] = V(o^{t+1})

    if recurrent:
        # the returned state is the state after obs[T-1] (before next_obs), so continuing from it
        # with next_obs must reproduce the last entry
        cont, _ = agents.cross_values(next_obs, new_state, next_done)
        assert torch.allclose(cont, v_next[-1], atol=1e-6)
    else:
        assert new_state is None


def _run(args, tmp_path):
    cmd = [sys.executable, os.path.join(ROOT, "train.py"), "--run-dir", str(tmp_path), *args]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]
    return res.stdout


PD_FAST = ["--env-id", "pd", "--max-cycles", "8", "--num-envs", "4", "--num-steps", "16", "--num-minibatches", "2",
           "--total-timesteps", "128", "--no-cuda"]


@pytest.mark.parametrize("signal", ["value", "reward"])
@pytest.mark.parametrize("formulation", ["none", "ei", "svo", "sia", "ia"])
def test_train_pd_feedforward_all_formulations_and_signals(formulation, signal, tmp_path):
    out = _run(
        PD_FAST + ["--formulation", formulation, "--signal", signal, "--alpha", "0,0.3", "--beta", "0.1", "--phi", "0.5"],
        tmp_path,
    )
    assert "iteration=2/2" in out
    ckpt = glob.glob(str(tmp_path / "*" / "agents.pt"))
    assert len(ckpt) == 1
    method = "none" if formulation == "none" else f"{formulation}_{signal}"
    assert os.path.basename(os.path.dirname(ckpt[0])).startswith(f"pd_n2__{method}__ff__")
    data = torch.load(ckpt[0], map_location="cpu", weights_only=False)
    assert data["num_agents"] == 2 and data["args"]["formulation"] == formulation and data["args"]["signal"] == signal
    agents = MultiAgents(2, OBS, spaces.Discrete(2), "vector")
    agents.load_state_dict(data["agents"])


@pytest.mark.parametrize("signal", ["value", "reward"])
def test_train_pd_recurrent(signal, tmp_path):
    out = _run(PD_FAST + ["--recurrent", "--formulation", "sia", "--signal", signal, "--alpha", "0.2"], tmp_path)
    assert "iteration=2/2" in out


def test_train_four_player_pd_with_per_agent_alphas(tmp_path):
    out = _run(
        PD_FAST + ["--num-agents", "4", "--formulation", "ia", "--signal", "value", "--alpha", "0,1,2,3", "--beta", "0.5"],
        tmp_path,
    )
    assert "agents=4" in out and "iteration=2/2" in out
    ckpt = glob.glob(str(tmp_path / "pd_n4__ia_value__ff__*" / "agents.pt"))
    assert len(ckpt) == 1
    assert torch.load(ckpt[0], map_location="cpu", weights_only=False)["num_agents"] == 4


def test_alpha_list_of_wrong_length_is_rejected(tmp_path):
    cmd = [sys.executable, os.path.join(ROOT, "train.py"), "--run-dir", str(tmp_path), *PD_FAST, "--num-agents", "3",
           "--formulation", "ei", "--alpha", "0,1"]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert res.returncode != 0 and "--alpha must be a single value or 3" in res.stderr


def test_train_image_env_cnn(tmp_path):
    out = _run(
        ["--env-id", "debug:image", "--max-cycles", "8", "--num-envs", "1", "--num-steps", "16", "--num-minibatches", "1",
         "--total-timesteps", "32", "--formulation", "ia", "--alpha", "0.1", "--beta", "0.05", "--recurrent",
         "--no-cuda", "--cross-value-chunk", "5"],
        tmp_path,
    )
    assert "iteration=2/2" in out


def test_recurrent_requires_divisible_minibatches(tmp_path):
    cmd = [sys.executable, os.path.join(ROOT, "train.py"), "--run-dir", str(tmp_path), *PD_FAST, "--recurrent",
           "--num-envs", "3", "--num-minibatches", "2"]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert res.returncode != 0 and "divisible" in res.stderr
