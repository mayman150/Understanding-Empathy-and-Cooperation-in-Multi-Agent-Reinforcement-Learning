"""Actor-critic networks (CleanRL ``ppo_atari.py`` / ``ppo_atari_lstm.py`` style).

``Agent`` is one independent learner: trunk (Nature-CNN for images, small MLP for
vectors) -> optional LSTM -> actor / critic heads.  ``MultiAgents`` holds one ``Agent``
per player (no parameter sharing) and presents batched ``(batch, num_agents, ...)``
tensors to the training loop.

Tensor layout conventions
-------------------------
* ``x``      observations,            ``(M, N, *obs_shape)``
* ``done``   episode-start flags,     ``(M, N)``   (1.0 => ``x`` is the first obs of a new episode)
* ``action`` ``(M, N)``
* outputs    ``(M, N)``               built with ``torch.stack(..., dim=1)`` so that
                                       ``out[m, i]`` belongs to agent ``i`` at batch row ``m``.
* LSTM state ``(h, c)`` with          ``h, c: (N, num_layers, num_sequences, hidden)``

For recurrent agents the batch rows must be **time-major**, ``m = t * num_sequences + s``
(exactly like CleanRL), because ``Agent.get_states`` re-folds ``M`` into
``(T, num_sequences)`` using ``num_sequences = lstm_state.shape[2]``.

``MultiAgents.cross_values`` evaluates agent ``i``'s critic on agent ``j``'s observation
stream for all ``(i, j)``: ``V[m, i, j] = V_i(x[m, j])``.  This is the quantity the social
terms in :mod:`empathy_marl.empathy` are built from.  For recurrent critics, agent ``i``
imagines *having seen agent j's observation history*, so a separate hidden state is
kept for every ``(i, j)`` pair: ``cross_state[h|c]: (N, num_layers, num_sequences * N, hidden)``
laid out as ``sequence-major, observed-agent-minor``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions.categorical import Categorical

LSTMState = tuple[torch.Tensor, torch.Tensor]


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """CleanRL Atari trunk, input ``(B, H, W, C)`` uint8/float in [0, 255]."""

    def __init__(self, obs_shape: tuple[int, ...], out_features: int = 512):
        super().__init__()
        h, w, c = obs_shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.fc = nn.Sequential(layer_init(nn.Linear(n_flat, out_features)), nn.ReLU())
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        return self.fc(self.conv(x.permute(0, 3, 1, 2)))


class MLP(nn.Module):
    """CleanRL classic-control trunk for flat observations."""

    def __init__(self, obs_shape: tuple[int, ...], hidden: int = 64):
        super().__init__()
        n_in = int(np.prod(obs_shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(n_in, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        self.out_features = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class Agent(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        obs_type: str,
        recurrent: bool = False,
        lstm_hidden_size: int = 128,
    ):
        super().__init__()
        if obs_type == "image":
            self.trunk: nn.Module = NatureCNN(observation_space.shape)
        elif obs_type == "vector":
            self.trunk = MLP(observation_space.shape)
        else:
            raise ValueError(f"unknown obs_type {obs_type!r}")
        head_in = self.trunk.out_features
        self.recurrent = recurrent
        if recurrent:
            self.lstm = nn.LSTM(head_in, lstm_hidden_size)
            for name, param in self.lstm.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param, 1.0)
            head_in = lstm_hidden_size
        self.actor = layer_init(nn.Linear(head_in, int(action_space.n)), std=0.01)
        self.critic = layer_init(nn.Linear(head_in, 1), std=1)

    def get_states(
        self, x: torch.Tensor, lstm_state: Optional[LSTMState], done: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, Optional[LSTMState]]:
        hidden = self.trunk(x)
        if not self.recurrent:
            return hidden, None
        assert lstm_state is not None and done is not None, "recurrent agent needs lstm_state and done"
        # CleanRL ppo_atari_lstm.py: fold the time-major batch back into (T, num_sequences)
        num_sequences = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, num_sequences, self.lstm.input_size))
        done = done.reshape((-1, num_sequences))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(
        self, x: torch.Tensor, lstm_state: Optional[LSTMState] = None, done: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[LSTMState]]:
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        return self.critic(hidden).squeeze(-1), lstm_state

    def get_action_and_value(
        self,
        x: torch.Tensor,
        lstm_state: Optional[LSTMState] = None,
        done: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden).squeeze(-1), lstm_state


class MultiAgents(nn.Module):
    """One independent ``Agent`` per player."""

    def __init__(
        self,
        num_agents: int,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        obs_type: str,
        recurrent: bool = False,
        lstm_hidden_size: int = 128,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.recurrent = recurrent
        self.lstm_hidden_size = lstm_hidden_size
        self.agents = nn.ModuleList(
            Agent(observation_space, action_space, obs_type, recurrent, lstm_hidden_size) for _ in range(num_agents)
        )

    @property
    def num_lstm_layers(self) -> int:
        return self.agents[0].lstm.num_layers if self.recurrent else 0

    # ----------------------------------------------------------------- LSTM state helpers
    def initial_lstm_state(self, num_sequences: int, device: torch.device) -> Optional[LSTMState]:
        if not self.recurrent:
            return None
        shape = (self.num_agents, self.num_lstm_layers, num_sequences, self.lstm_hidden_size)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    def initial_cross_lstm_state(self, num_sequences: int, device: torch.device) -> Optional[LSTMState]:
        """State for ``cross_values``: one hidden state per (evaluating agent i, observed agent j)."""
        return self.initial_lstm_state(num_sequences * self.num_agents, device)

    @staticmethod
    def _agent_state(lstm_state: Optional[LSTMState], i: int) -> Optional[LSTMState]:
        return None if lstm_state is None else (lstm_state[0][i], lstm_state[1][i])

    @staticmethod
    def _stack_states(states: list[Optional[LSTMState]]) -> Optional[LSTMState]:
        if states[0] is None:
            return None
        return torch.stack([s[0] for s in states]), torch.stack([s[1] for s in states])

    # ----------------------------------------------------------------- batched forward passes
    def get_action_and_value(
        self,
        x: torch.Tensor,
        lstm_state: Optional[LSTMState] = None,
        done: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        """``x: (M, N, *obs)``, ``done: (M, N)``, ``action: (M, N)`` -> all outputs ``(M, N)``."""
        actions, logprobs, entropies, values, states = [], [], [], [], []
        for i, agent in enumerate(self.agents):
            a, lp, ent, v, st = agent.get_action_and_value(
                x[:, i],
                self._agent_state(lstm_state, i),
                None if done is None else done[:, i],
                None if action is None else action[:, i],
                deterministic=deterministic,
            )
            actions.append(a)
            logprobs.append(lp)
            entropies.append(ent)
            values.append(v)
            states.append(st)
        return (
            torch.stack(actions, dim=1),
            torch.stack(logprobs, dim=1),
            torch.stack(entropies, dim=1),
            torch.stack(values, dim=1),
            self._stack_states(states),
        )

    def get_values(
        self, x: torch.Tensor, lstm_state: Optional[LSTMState] = None, done: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[LSTMState]]:
        values, states = [], []
        for i, agent in enumerate(self.agents):
            v, st = agent.get_value(x[:, i], self._agent_state(lstm_state, i), None if done is None else done[:, i])
            values.append(v)
            states.append(st)
        return torch.stack(values, dim=1), self._stack_states(states)

    def cross_values(
        self,
        x: torch.Tensor,
        cross_state: Optional[LSTMState] = None,
        done: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[LSTMState]]:
        """Evaluate every agent's critic on every agent's observation.

        ``x: (M, N, *obs)`` (time-major rows for recurrent agents) -> ``V: (M, N, N)`` with
        ``V[m, i, j] = V_i(x[m, j])``.  ``done: (M, N)`` are the observed agents' episode-start
        flags (they reset the imagined hidden state of stream ``j``).
        """
        M, N = x.shape[0], x.shape[1]
        flat_x = x.reshape((M * N,) + x.shape[2:])  # row m*N + j  -> (m, j)
        flat_done = None if done is None else done.reshape(M * N)
        values, states = [], []
        for i, agent in enumerate(self.agents):
            v, st = agent.get_value(flat_x, self._agent_state(cross_state, i), flat_done)
            values.append(v.reshape(M, N))
            states.append(st)
        return torch.stack(values, dim=1), self._stack_states(states)
