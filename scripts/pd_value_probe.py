"""Probe a trained Prisoner's Dilemma checkpoint: what does each agent's critic think of every state?

For every canonical observation (first round, and the four (own, other) last-action combinations)
print V_i(s) for every agent, then the value-based social term X_i each formulation would produce
after an asymmetric round (one cooperates, one defects), where the two agents see *different*
observations.  This is the direct check of the mechanism behind the value signal: if the critic
assigns (almost) the same value to every state, gap-based terms (SIA, IA) vanish.

    python scripts/pd_value_probe.py runs/<run>/agents.pt
    python scripts/pd_value_probe.py runs/<run>/agents.pt --alpha 30
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from empathy_marl.agents import MultiAgents  # noqa: E402
from empathy_marl.empathy import FORMULATIONS, social_term
from empathy_marl.prisoners_dilemma import OBS_DIM

C, D = "C", "D"


def obs(own: str | None, other: str | None) -> np.ndarray:
    """Observation of an agent whose last action was `own` while the other played `other` (None = first round)."""
    x = np.zeros(OBS_DIM, dtype=np.float32)
    if own is None:
        x[4] = 1.0
    else:
        x[0 if own == C else 1] = 1.0
        x[2 if other == C else 3] = 1.0
    return x


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--alpha", type=float, default=None, help="alpha used for the printed X_i (default: the run's alpha)")
    a = p.parse_args()

    data = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    targs = data["args"]
    N = data["num_agents"]
    if N != 2:
        raise SystemExit("this probe is written for the 2-player Prisoner's Dilemma")
    agents = MultiAgents(
        N, spaces.Box(0.0, 1.0, (OBS_DIM,), np.float32), spaces.Discrete(2), "vector",
        recurrent=targs["recurrent"], lstm_hidden_size=targs["lstm_hidden_size"],
        reward_model=targs.get("signal") == "imagined",
    )
    agents.load_state_dict(data["agents"])
    agents.eval()
    if targs["recurrent"]:
        print("note: recurrent critic probed from a zero hidden state (single-step values)")
    alpha_run = [float(v) for v in str(targs["alpha"]).split(",")]
    alpha_run = alpha_run * N if len(alpha_run) == 1 else alpha_run
    alpha = torch.tensor([a.alpha] * N if a.alpha is not None else alpha_run)
    print(f"run: formulation={targs['formulation']} signal={targs['signal']} alpha={targs['alpha']} seed={targs['seed']}")

    states = {"first round": obs(None, None), "(C,C)": obs(C, C), "(C,D) me C, other D": obs(C, D),
              "(D,C) me D, other C": obs(D, C), "(D,D)": obs(D, D)}
    x = torch.as_tensor(np.stack(list(states.values())))  # (5, obs)
    device = torch.device("cpu")

    def state_for(i: int, batch: int):
        st = agents.initial_lstm_state(batch, device)
        return None if st is None else (st[0][i], st[1][i])

    with torch.no_grad():
        values, probs = [], []
        for i, ag in enumerate(agents.agents):
            hidden, _ = ag.get_states(x, state_for(i, 5), torch.zeros(5))
            values.append(ag.critic(hidden).squeeze(-1))
            probs.append(torch.softmax(ag.actor(hidden), dim=-1)[:, 0])  # P(cooperate)
        v = torch.stack(values, dim=1)  # (5, N)
        p_coop = torch.stack(probs, dim=1)

    print("\ncritic value V_i(s) and P_i(cooperate | s) for every state:")
    print(f"{'state':24s}" + "".join(f"  V_{i}      P{i}(C)" for i in range(N)))
    for k, name in enumerate(states):
        print(f"{name:24s}" + "".join(f"  {v[k, i]:7.3f}  {p_coop[k, i]:5.2f}" for i in range(N)))
    spread = (v.max(dim=0).values - v.min(dim=0).values)
    print("value spread across states per agent (max - min): " + ", ".join(f"{s:.4f}" for s in spread))

    # after an asymmetric round: agent 0 played C, agent 1 played D.
    # agent 0 sees (C,D), agent 1 sees (D,C); z[i, j] = V_i(s_j)
    s0, s1 = obs(C, D), obs(D, C)
    with torch.no_grad():
        z = torch.zeros(N, N)
        for i, ag in enumerate(agents.agents):
            for j, s in enumerate((s0, s1)):
                z[i, j] = ag.get_value(torch.as_tensor(s)[None], state_for(i, 1), torch.zeros(1))[0][0]
    print("\nafter an asymmetric round (agent 0 cooperated, agent 1 defected):")
    print(f"  V_0(own state)={z[0,0]:.4f}  V_0(other's state)={z[0,1]:.4f}   V_1(own)={z[1,1]:.4f}  V_1(other's)={z[1,0]:.4f}")
    beta = alpha / 2
    phi = torch.full((N,), np.pi / 4)
    print(f"  value-based social term X_i at alpha={alpha.tolist()} (beta=alpha/2, phi=pi/4):")
    for f in FORMULATIONS:
        if f == "none":
            continue
        xv = social_term(f, z, alpha, beta, phi)
        print(f"    {f:4s}: X_0={xv[0]:9.4f}  X_1={xv[1]:9.4f}")
    print("\n  same round with the REWARD signal on raw rewards (r_0=0, r_1=4):")
    zr = torch.tensor([[0.0, 4.0], [0.0, 4.0]])
    for f in FORMULATIONS:
        if f == "none":
            continue
        xr = social_term(f, zr, alpha, beta, phi)
        print(f"    {f:4s}: X_0={xr[0]:9.4f}  X_1={xr[1]:9.4f}")


if __name__ == "__main__":
    main()
