"""Actor-critic networks (CleanRL ``ppo_atari.py`` / ``ppo_atari_lstm.py`` style).

``Agent`` is one independent learner: trunk (Nature-CNN for images, ``GridCNN`` for small
grid-world windows of planes / colours, small MLP for vectors) -> optional LSTM -> actor / critic heads.  ``MultiAgents`` holds one ``Agent``
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

``--signal imagined`` adds a per-agent **reward model** ``f_i(o, a, o')`` (``reward_model=True``):
a regression of the agent's *own* reward on its own observation transition.  Applied to
another agent's transition, ``f_i(o_j, a_j, o_j')`` is agent ``i``'s imagined reward of agent ``j``
("what I would have received in your shoes"); ``MultiAgents.cross_rewards`` computes it for
all pairs.  For vector observations the model is a small MLP on the raw ``(o, o')`` pair; for
grid windows a tiny convolution of its own over the stacked pair (``GridRewardModel``); for
images a head on the (detached) trunk features of both frames, so it never trains the trunk.
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


class GridCNN(nn.Module):
    """Small convolutional trunk for grid-world windows (``obs_type="grid"``): input ``(B, H, W, C)``.

    Meant for one-hot planes (or per-cell colours) of a few dozen cells a side, e.g. the 17x17x7 Clean Up
    window, where the Atari trunk's 8x8 / stride-4 filters do not fit.  Two 3x3 convolutions (the second with
    stride 2 to keep the flattened size small) and one linear layer; ``scale`` divides the input (1/255 for
    uint8 colours, 1 for 0/1 planes).  Sized for a 10x10 map (about 0.4M parameters; LIO's Clean Up network was
    a single 6-filter convolution): the social terms evaluate every critic on every agent's observation, so the
    trunk's cost is paid N^2 times per step.  No normalisation layers (on-policy RL convention).
    """

    def __init__(self, obs_shape: tuple[int, ...], out_features: int = 128, scale: float = 1.0, channels: tuple[int, int] = (16, 32)):
        super().__init__()
        h, w, c = obs_shape
        self.scale = float(scale)
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(c, channels[0], 3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.fc = nn.Sequential(layer_init(nn.Linear(n_flat, out_features)), nn.ReLU())
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() * self.scale
        return self.fc(self.conv(x.permute(0, 3, 1, 2)))


class GridRewardModel(nn.Module):
    """``rhat = f(o, a, o')`` for grid windows: a tiny convolution over the stacked pair ``[o, o']`` plus the action.

    The reward events of these games are local ("the apple under my cell disappeared"), so 3x3 filters over
    the two frames see them directly; global max- and mean-pooling make the output independent of where in
    the window the event happens (the observer is at the centre of its own window, but the model is also
    evaluated on *other* agents' windows).  Own network, trained on the agent's own transitions only; it does
    not share or train the policy trunk.
    """

    def __init__(self, obs_shape: tuple[int, ...], num_actions: int, scale: float = 1.0, channels: int = 16, hidden: int = 64):
        super().__init__()
        h, w, c = obs_shape
        self.scale = float(scale)
        self.num_actions = int(num_actions)
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(2 * c, channels, 3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(channels, channels, 3, stride=1, padding=1)),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            layer_init(nn.Linear(2 * channels + self.num_actions, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def forward(self, x: torch.Tensor, action: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        """``x, x_next: (B, H, W, C)``, ``action: (B,)`` integer -> ``(B,)``."""
        pair = torch.cat([x.float(), x_next.float()], dim=-1).permute(0, 3, 1, 2) * self.scale
        feats = self.conv(pair)
        pooled = torch.cat([feats.amax(dim=(2, 3)), feats.mean(dim=(2, 3))], dim=-1)
        a = torch.nn.functional.one_hot(action.long(), self.num_actions).float()
        return self.head(torch.cat([pooled, a], dim=-1)).squeeze(-1)


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


class RewardModel(nn.Module):
    """``rhat = f(o, a, o')``: the reward an agent receives for taking ``a`` in ``o`` and arriving in ``o'``.

    Trained on the agent's own transitions only; evaluated on other agents' transitions (their
    observation, their action, their next observation) to imagine their rewards.  The action is
    part of the input because an egocentric frame does not always identify the agent's own move
    (on the Coin Game's 3x3 torus "I stepped onto my coin" and "the other stepped onto my coin
    while I stepped away" can produce the same before/after pictures).  ``features`` is ``None``
    for vector observations (MLP on the raw pair) or the agent's trunk for images (head on the
    *detached* features of both frames).
    """

    def __init__(self, obs_shape: tuple[int, ...], num_actions: int, features: Optional[nn.Module], hidden: int = 64):
        super().__init__()
        # plain attribute on purpose (bypasses nn.Module registration): the trunk belongs to the Agent, so it must
        # not appear a second time in this module's parameters / state_dict
        object.__setattr__(self, "_features", features)
        self.num_actions = int(num_actions)
        n_in = 2 * (features.out_features if features is not None else int(np.prod(obs_shape))) + self.num_actions
        layers = [layer_init(nn.Linear(n_in, hidden)), nn.Tanh()]
        if features is None:
            layers += [layer_init(nn.Linear(hidden, hidden)), nn.Tanh()]
        layers += [layer_init(nn.Linear(hidden, 1), std=1.0)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, action: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        """``x, x_next: (B, *obs)``, ``action: (B,)`` integer -> ``(B,)``."""
        a = torch.nn.functional.one_hot(action.long(), self.num_actions).float()
        if self._features is None:
            z = torch.cat([x.flatten(1).float(), a, x_next.flatten(1).float()], dim=-1)
        else:
            with torch.no_grad():
                h, h_next = self._features(x), self._features(x_next)
            z = torch.cat([h, a, h_next], dim=-1)
        return self.net(z).squeeze(-1)


class Agent(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        obs_type: str,
        recurrent: bool = False,
        lstm_hidden_size: int = 128,
        reward_model: bool = False,
    ):
        super().__init__()
        grid_scale = 1.0 / 255.0 if np.issubdtype(observation_space.dtype, np.integer) else 1.0
        if obs_type == "image":
            self.trunk: nn.Module = NatureCNN(observation_space.shape)
        elif obs_type == "grid":
            self.trunk = GridCNN(observation_space.shape, scale=grid_scale)
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
        self.reward_model: Optional[nn.Module] = None
        if reward_model:
            if obs_type == "grid":
                self.reward_model = GridRewardModel(observation_space.shape, int(action_space.n), scale=grid_scale)
            else:
                self.reward_model = RewardModel(observation_space.shape, int(action_space.n), self.trunk if obs_type == "image" else None)

    def predict_reward(self, x: torch.Tensor, action: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        """Imagined reward for the transition ``(x, action) -> x_next`` (``x, x_next: (B, *obs)``, ``action: (B,)``)."""
        if self.reward_model is None:
            raise RuntimeError("this agent has no reward model (construct with reward_model=True)")
        return self.reward_model(x, action, x_next)

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
        reward_model: bool = False,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.recurrent = recurrent
        self.lstm_hidden_size = lstm_hidden_size
        self.agents = nn.ModuleList(
            Agent(observation_space, action_space, obs_type, recurrent, lstm_hidden_size, reward_model)
            for _ in range(num_agents)
        )

    # ----------------------------------------------------------------- imagined rewards
    def predict_own_rewards(self, x: torch.Tensor, action: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        """``x, x_next: (M, N, *obs)``, ``action: (M, N)`` -> ``(M, N)`` with ``[m, i] = f_i(x[m, i], a[m, i], x_next[m, i])``
        (differentiable: this is the reward-model training target)."""
        return torch.stack(
            [agent.predict_reward(x[:, i], action[:, i], x_next[:, i]) for i, agent in enumerate(self.agents)], dim=1
        )

    def cross_rewards(self, x: torch.Tensor, action: torch.Tensor, x_next: torch.Tensor) -> torch.Tensor:
        """Every agent's reward model on every agent's transition.

        ``x, x_next: (M, N, *obs)``, ``action: (M, N)`` -> ``(M, N, N)`` with
        ``[m, i, j] = f_i(x[m, j], a[m, j], x_next[m, j])``: agent ``i``'s imagined reward of agent ``j``.
        """
        M, N = x.shape[0], x.shape[1]
        flat_x = x.reshape((M * N,) + x.shape[2:])
        flat_a = action.reshape(M * N)
        flat_next = x_next.reshape((M * N,) + x_next.shape[2:])
        return torch.stack([agent.predict_reward(flat_x, flat_a, flat_next).reshape(M, N) for agent in self.agents], dim=1)

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
