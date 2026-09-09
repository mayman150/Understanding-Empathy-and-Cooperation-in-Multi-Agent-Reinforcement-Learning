"""Evaluate a trained checkpoint: run episodes, report social-outcome metrics, optionally record a video.

    python evaluate.py --checkpoint runs/<run>/agents.pt --episodes 10
    python evaluate.py --checkpoint runs/<run>/agents.pt --episodes 1 --video out.mp4   # Melting Pot / debug env

The environment is rebuilt from the arguments stored in the checkpoint, so the same
formulation / observation layout / episode length are used.  Metrics are printed and, with
``--out``, written as JSON.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import tyro

from empathy_marl.agents import MultiAgents
from empathy_marl.envs import make_envs
from empathy_marl.metrics import MultiAgentEpisodeStatistics


@dataclass
class EvalArgs:
    checkpoint: str
    episodes: int = 5
    seed: int = 0
    deterministic: bool = False
    """take argmax actions instead of sampling"""
    cuda: bool = False
    max_cycles: Optional[int] = None
    """override the episode length stored in the checkpoint"""
    video: Optional[str] = None
    """path of an .mp4/.gif to write (first episode only; needs imageio and an env that renders images)"""
    out: Optional[str] = None
    """write the aggregated metrics to this JSON file"""


def main(args: EvalArgs) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    data = torch.load(args.checkpoint, map_location=device, weights_only=False)
    targs = data["args"]
    bundle = make_envs(
        targs["env_id"],
        num_envs=1,
        max_cycles=args.max_cycles or targs["max_cycles"],
        num_agents=targs.get("num_agents", data["num_agents"]),
        pd_payoffs=tuple(targs["pd_payoffs"]),
        render_mode="rgb_array" if args.video else None,
    )
    N = bundle.num_agents
    envs = MultiAgentEpisodeStatistics(bundle.envs, 1, N)
    agents = MultiAgents(
        N,
        bundle.single_observation_space,
        bundle.single_action_space,
        bundle.obs_type,
        recurrent=targs["recurrent"],
        lstm_hidden_size=targs["lstm_hidden_size"],
    ).to(device)
    agents.load_state_dict(data["agents"])
    agents.eval()
    obs_dtype = torch.uint8 if bundle.obs_type == "image" else torch.float32
    obs_shape = bundle.single_observation_space.shape

    def to_obs(raw) -> torch.Tensor:
        return torch.as_tensor(bundle.extract_obs(raw), dtype=obs_dtype, device=device).reshape((1, N) + obs_shape)

    episodes: list[dict] = []
    frames: list[np.ndarray] = []
    coop_counts = np.zeros(N)
    steps = 0
    raw, _ = envs.reset(seed=args.seed)
    next_obs = to_obs(raw)
    next_done = torch.zeros((1, N), device=device)
    lstm_state = agents.initial_lstm_state(1, device)
    with torch.no_grad():
        while len(episodes) < args.episodes:
            if args.video and not episodes:
                frame = envs.render()
                if isinstance(frame, np.ndarray):
                    frames.append(frame)
            action, _, _, _, lstm_state = agents.get_action_and_value(
                next_obs, lstm_state, next_done, deterministic=args.deterministic
            )
            raw, _, term, trunc, infos = envs.step(action.reshape(-1).cpu().numpy())
            if bundle.cooperate_action is not None:
                coop_counts += (action.reshape(-1).cpu().numpy() == bundle.cooperate_action)
            steps += 1
            next_obs = to_obs(raw)
            next_done = torch.as_tensor(np.maximum(term, trunc), dtype=torch.float32, device=device).reshape(1, N)
            if "ma_episode" in infos[0]:
                ep = infos[0]["ma_episode"]
                episodes.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ep.items()})
                print(
                    f"episode {len(episodes)}: collective={ep['collective']:.2f} per_agent={np.round(ep['r'], 2).tolist()} "
                    f"equality={ep['equality']:.3f} sustainability={ep['sustainability']:.1f}"
                )
    envs.close()

    summary = {
        "checkpoint": args.checkpoint,
        "env_id": targs["env_id"],
        "formulation": targs["formulation"],
        "signal": targs.get("signal", "value"),
        "num_agents": N,
        "alpha": targs["alpha"],
        "beta": targs["beta"],
        "phi": targs["phi"],
        "episodes": len(episodes),
        "collective_return_mean": float(np.mean([e["collective"] for e in episodes])),
        "collective_return_std": float(np.std([e["collective"] for e in episodes])),
        "per_agent_return_mean": np.mean([e["r"] for e in episodes], axis=0).tolist(),
        "equality_mean": float(np.mean([e["equality"] for e in episodes])),
        "sustainability_mean": float(np.mean([e["sustainability"] for e in episodes])),
    }
    if bundle.cooperate_action is not None:
        summary["cooperation_rate"] = (coop_counts / steps).tolist()
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "episodes": episodes}, f, indent=2)
    if args.video:
        if frames:
            import imageio

            imageio.mimsave(args.video, frames, fps=8)
            print(f"wrote {len(frames)} frames to {args.video}")
        else:
            print("no image frames were produced by this environment; video skipped")
    return summary


if __name__ == "__main__":
    main(tyro.cli(EvalArgs))
