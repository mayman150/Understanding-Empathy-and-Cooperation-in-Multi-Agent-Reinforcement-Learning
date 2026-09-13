"""End-to-end smoke tests of ``train.py`` plus a unit test of the next-state cross values."""
import glob
import json
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


@pytest.mark.parametrize("signal", ["value", "reward"])
def test_train_coin_game_logs_episode_cooperation_rate(signal, tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = _run(
        ["--env-id", "coin", "--max-cycles", "8", "--num-envs", "4", "--num-steps", "16", "--num-minibatches", "2",
         "--total-timesteps", "128", "--no-cuda", "--formulation", "sia", "--signal", signal, "--alpha", "1"],
        tmp_path,
    )
    assert "actions=4" in out and "iteration=2/2" in out
    run_dirs = glob.glob(str(tmp_path / f"coin__sia_{signal}__ff__a1_b0.0_p0.0__s1__*"))
    assert len(run_dirs) == 1
    acc = EventAccumulator(run_dirs[0], size_guidance={"scalars": 0})
    acc.Reload()
    tags = set(acc.Tags()["scalars"])
    # per-episode statistics from the environment (8-step episodes -> 16 episodes in 128 steps)
    assert {"charts/own_coins/player_0", "charts/other_coins/player_1", "charts/collective_return"} <= tags
    coop = [t for t in tags if t.startswith("charts/cooperation_rate/")]
    assert coop, "cooperation_rate is logged whenever an agent picked up at least one coin"
    for tag in coop:
        assert all(0.0 <= s.value <= 1.0 for s in acc.Scalars(tag))
    assert "social/term_abs_mean/player_0" in tags


COIN_FAST = ["--env-id", "coin", "--max-cycles", "8", "--num-envs", "4", "--num-steps", "16", "--num-minibatches", "2",
             "--total-timesteps", "192", "--no-cuda"]


@pytest.mark.parametrize(
    "extra, method",
    [
        (["--formulation", "ei", "--imagined-critic", "other", "--social-scale", "--alpha", "1"], "ei_imagined_other_sc"),
        (["--formulation", "svo", "--imagined-critic", "other", "--alpha", "1", "--phi", "0.5236"], "svo_imagined_other"),
        (["--formulation", "sia", "--imagined-critic", "shaped", "--alpha", "0.1"], "sia_imagined_shaped"),
        (["--formulation", "ia", "--imagined-critic", "shaped", "--alpha", "0.1", "--beta", "0.05", "--recurrent"], "ia_imagined_shaped"),
        (["--formulation", "none"], "none"),
    ],
)
def test_train_imagined_signal(extra, method, tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = _run(COIN_FAST + ["--signal", "imagined", "--imagined-warmup", "1", *extra], tmp_path)
    assert "iteration=3/3" in out
    run_dirs = glob.glob(str(tmp_path / f"coin__{method}__*"))
    assert len(run_dirs) == 1, glob.glob(str(tmp_path / "*"))
    acc = EventAccumulator(run_dirs[0], size_guidance={"scalars": 0})
    acc.Reload()
    tags = set(acc.Tags()["scalars"])
    assert {"losses/reward_model/player_0", "imagined/corr/player_0", "imagined/mae/player_1"} <= tags
    if method != "none":
        term = [s.value for s in acc.Scalars("social/term_abs_mean/player_0")]
        assert term[0] == 0.0  # warm-up iteration: social term off
        assert any(v > 0.0 for v in term[1:])  # ... then on
    data = torch.load(os.path.join(run_dirs[0], "agents.pt"), map_location="cpu", weights_only=False)
    assert any("reward_model" in k for k in data["agents"])
    agents = MultiAgents(2, spaces.Box(0.0, 1.0, (36,), np.float32), spaces.Discrete(4), "vector",
                         recurrent="--recurrent" in extra, reward_model=True)
    agents.load_state_dict(data["agents"])


def test_train_value_signal_with_social_scale(tmp_path):
    out = _run(COIN_FAST + ["--formulation", "ei", "--signal", "value", "--social-scale", "--alpha", "1"], tmp_path)
    assert "iteration=3/3" in out
    assert len(glob.glob(str(tmp_path / "coin__ei_value_sc__*"))) == 1


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--formulation", "sia", "--imagined-critic", "none", "--imagined-level", "trace_value"], "needs the critic"),
        (["--formulation", "ei", "--imagined-critic", "other", "--recurrent", "--num-envs", "4", "--num-minibatches", "2"], "feed-forward"),
    ],
)
def test_imagined_other_rejects_unsupported_combinations(extra, message, tmp_path):
    cmd = [sys.executable, os.path.join(ROOT, "train.py"), "--run-dir", str(tmp_path), *COIN_FAST, "--signal", "imagined",
           "--alpha", "1", *extra]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert res.returncode != 0 and message in res.stderr


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


CLEANUP_FAST = ["--env-id", "cleanup", "--num-agents", "3", "--max-cycles", "6", "--num-envs", "2", "--num-steps", "8",
                "--num-minibatches", "2", "--total-timesteps", "48", "--no-cuda"]


@pytest.mark.parametrize(
    "extra, method",
    [
        (["--formulation", "none"], "none"),
        (["--formulation", "ei", "--signal", "imagined", "--imagined-critic", "other", "--social-scale", "--alpha", "1",
          "--imagined-warmup", "1", "--reward-model-replay", "64", "--social-aggregate", "sum"], "ei_imagined_other_sc_sum"),
        (["--formulation", "ia", "--signal", "imagined", "--imagined-critic", "shaped", "--alpha", "1", "--beta", "0.1",
          "--imagined-warmup", "1"], "ia_imagined_shaped"),
        (["--formulation", "ei", "--signal", "value", "--social-scale", "--alpha", "1"], "ei_value_sc"),
        (["--formulation", "ei", "--signal", "reward", "--alpha", "0.1"], "ei_reward"),
    ],
)
def test_train_cleanup_grid_observations(extra, method, tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = _run(CLEANUP_FAST + extra, tmp_path)
    assert "obs=(17, 17, 7) (grid) actions=6" in out and "iteration=3/3" in out
    run_dirs = glob.glob(str(tmp_path / f"cleanup__{method}__ff__*"))
    assert len(run_dirs) == 1, glob.glob(str(tmp_path / "*"))
    acc = EventAccumulator(run_dirs[0], size_guidance={"scalars": 0})
    acc.Reload()
    tags = set(acc.Tags()["scalars"])
    # 6-step episodes -> per-episode Clean Up statistics are logged for every agent
    assert {"charts/apples/player_0", "charts/clean_actions/player_2", "charts/waste_density/player_1",
            "charts/collective_return", "charts/equality"} <= tags
    if "imagined" in method:
        assert {"imagined/corr/player_0", "losses/reward_model/player_1"} <= tags
    data = torch.load(os.path.join(run_dirs[0], "agents.pt"), map_location="cpu", weights_only=False)
    assert data["obs_type"] == "grid"
    agents = MultiAgents(3, spaces.Box(0.0, 1.0, (17, 17, 7), np.float32), spaces.Discrete(6), "grid",
                         reward_model="imagined" in method)
    agents.load_state_dict(data["agents"])


def test_train_cleanup_rgb_observations(tmp_path):
    out = _run(CLEANUP_FAST + ["--cleanup-obs", "rgb", "--formulation", "none"], tmp_path)
    assert "obs=(17, 17, 3) (grid)" in out and "iteration=3/3" in out


def test_train_imagined_without_critic_ablation(tmp_path):
    """--imagined-critic none: the other's discounted imagined rewards enter the coefficient, no critic involved."""
    out = _run(COIN_FAST + ["--signal", "imagined", "--imagined-critic", "none", "--social-scale", "--formulation", "ei",
                            "--alpha", "1", "--imagined-warmup", "1", "--imagined-lambda", "0"], tmp_path)
    assert "iteration=3/3" in out
    assert len(glob.glob(str(tmp_path / "coin__ei_imagined_none_sc__*"))) == 1


@pytest.mark.parametrize(
    "extra, method",
    [
        (["--imagined-critic", "other", "--formulation", "ia", "--alpha", "0.1", "--beta", "1", "--social-scale"], "ia_imagined_other_sc"),
        (["--imagined-critic", "other", "--formulation", "ia", "--alpha", "0.1", "--beta", "1", "--imagined-level", "trace_value",
          "--ia-weighting", "relu"], "ia_imagined_other_lv_relu"),
        (["--imagined-critic", "other", "--formulation", "sia", "--alpha", "1", "--social-scale"], "sia_imagined_other_sc"),
        (["--imagined-critic", "none", "--formulation", "sia", "--alpha", "1"], "sia_imagined_none"),
    ],
)
def test_train_inequity_through_the_advantage_construction(extra, method, tmp_path):
    """sia / ia with imagined advantages: inequity-dependent weights on my own and the others' imagined advantages."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = _run(COIN_FAST + ["--signal", "imagined", "--imagined-warmup", "1", *extra], tmp_path)
    assert "iteration=3/3" in out
    run_dirs = glob.glob(str(tmp_path / f"coin__{method}__*"))
    assert len(run_dirs) == 1, glob.glob(str(tmp_path / "*"))
    acc = EventAccumulator(run_dirs[0], size_guidance={"scalars": 0})
    acc.Reload()
    term = [s.value for s in acc.Scalars("social/term_abs_mean/player_0")]
    assert term[0] == 0.0 and any(v > 0.0 for v in term[1:])


def test_cross_next_values_matches_full_pass_at_episode_boundaries():
    from train import cross_next_values

    torch.manual_seed(0)
    T, E, N = 6, 2, 3
    agents = MultiAgents(N, OBS, spaces.Discrete(2), "vector")
    obs = torch.rand(T, E, N, 5)
    next_obs_buf = torch.empty_like(obs)
    next_obs_buf[:-1] = obs[1:]
    next_obs_buf[-1] = torch.rand(E, N, 5)
    episode_end = torch.zeros(T, E, N)
    episode_end[2, 1] = 1.0  # env 1: all agents' episodes end at t = 2 -> next_obs_buf holds terminal observations
    episode_end[4, 0, 1] = 1.0  # env 0: only agent 1 (asymmetric end, as the flags are per observed agent)
    for t, e, j in ((2, 1, 0), (2, 1, 1), (2, 1, 2), (4, 0, 1)):
        next_obs_buf[t, e, j] = torch.rand(5)
    with torch.no_grad():
        v_cross, _ = agents.cross_values(obs.reshape(T * E, N, 5))
        v_cross = v_cross.reshape(T, E, N, N)
        fast = cross_next_values(agents, v_cross, next_obs_buf, episode_end)
        full, _ = agents.cross_values(next_obs_buf.reshape(T * E, N, 5))
    assert torch.allclose(fast, full.reshape(T, E, N, N), atol=1e-6)


def test_train_shaped_with_value_level(tmp_path):
    """--imagined-level trace_value: IA / EI on 'recent imagined earnings + my critic on the other's next observation'."""
    out = _run(COIN_FAST + ["--signal", "imagined", "--imagined-critic", "shaped", "--imagined-level", "trace_value",
                            "--formulation", "ia", "--alpha", "1", "--beta", "0.1", "--imagined-warmup", "1"], tmp_path)
    assert "iteration=3/3" in out
    run_dirs = glob.glob(str(tmp_path / "coin__ia_imagined_shaped_lv__*"))
    assert len(run_dirs) == 1
    with open(os.path.join(run_dirs[0], "args.json")) as f:
        args = json.load(f)
    assert args["imagined_level"] == "trace_value"
    assert args["level_value_weight"] == pytest.approx((1 - 0.99) / (1 - 0.99 * 0.975))
    for extra, message in [
        (["--signal", "imagined", "--imagined-critic", "other", "--formulation", "ei", "--alpha", "1"], "applies to"),
        (["--signal", "value", "--formulation", "ei", "--alpha", "1"], "applies to"),
        (["--signal", "imagined", "--imagined-critic", "shaped", "--formulation", "ei", "--alpha", "1", "--recurrent",
          "--num-envs", "4", "--num-minibatches", "2"], "feed-forward"),
    ]:
        cmd = [sys.executable, os.path.join(ROOT, "train.py"), "--run-dir", str(tmp_path), *COIN_FAST,
               "--imagined-level", "trace_value", *extra]
        res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert res.returncode != 0 and message in res.stderr


@pytest.mark.parametrize(
    "extra, method",
    [
        (["--formulation", "ei", "--alpha", "1", "--social-scale"], "ei_imagined_level_sc"),
        (["--formulation", "ia", "--alpha", "1", "--beta", "0.1"], "ia_imagined_level"),
    ],
)
def test_train_imagined_level_in_the_coefficient(extra, method, tmp_path):
    """--imagined-critic level: F(smoothed imagined rewards + w * gamma * V_i(o_j')) added to the advantage."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = _run(COIN_FAST + ["--signal", "imagined", "--imagined-critic", "level", "--imagined-warmup", "1", *extra], tmp_path)
    assert "iteration=3/3" in out
    run_dirs = glob.glob(str(tmp_path / f"coin__{method}__*"))
    assert len(run_dirs) == 1, glob.glob(str(tmp_path / "*"))
    acc = EventAccumulator(run_dirs[0], size_guidance={"scalars": 0})
    acc.Reload()
    term = [s.value for s in acc.Scalars("social/term_abs_mean/player_0")]
    assert term[0] == 0.0 and any(v > 0.0 for v in term[1:])  # off during the warm-up, on afterwards
    with open(os.path.join(run_dirs[0], "args.json")) as f:
        assert json.load(f)["level_value_weight"] == pytest.approx((1 - 0.99) / (1 - 0.99 * 0.975))
