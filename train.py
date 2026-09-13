"""Independent PPO learners with social preferences (CleanRL style).

``--formulation`` picks the functional form (none | ei | svo | sia | ia) and ``--signal`` picks
what it is computed from: ``value`` (our proposal: the agent's own critic evaluated on the
other agents' observations, added to the advantage) or ``reward`` (the literature baseline:
an intrinsic reward built from the other agents' rewards).  One script covers every
experiment in the plan:

    # sanity check: repeated Prisoner's Dilemma, selfish vs. empathetic (per-agent alpha), value signal
    python train.py --env-id pd --num-agents 2 --max-cycles 100 --formulation sia --signal value \
        --alpha 0,20 --seed 1 --num-envs 8 --num-steps 128 --total-timesteps 500000

    # the same functional form driven by the other agent's rewards (baseline with reward access)
    python train.py --env-id pd --max-cycles 100 --formulation sia --signal reward --alpha 0,1 --seed 1

    # Coin Game (Lerer & Peysakhovich 2017): the PD on a 3x3 grid, so the state carries information
    # about how well off each agent is (the sanity check for the value signal); 50-step episodes
    python train.py --env-id coin --max-cycles 50 --formulation ei --signal value --alpha 3 --seed 1 \
        --num-envs 8 --num-steps 128 --total-timesteps 1000000

    # 4-player Prisoner's Dilemma
    python train.py --env-id pd --num-agents 4 --formulation ei --signal value --alpha 20

    # Commons Harvest, feed-forward CNN, SIA
    python train.py --env-id meltingpot:commons_harvest__open --formulation sia --alpha 0.1 --seed 1

    # Commons Harvest, recurrent policy (partial observability), 4 env copies / 4 minibatches
    python train.py --env-id meltingpot:commons_harvest__open --formulation ei --alpha 0.01 \
        --recurrent --num-envs 4 --num-minibatches 4

    # Hughes et al. (2018) inequity-aversion reward baseline
    python train.py --env-id meltingpot:clean_up --formulation ia --signal reward --alpha 5 --beta 0.05

    # imagined rewards (no reward access): my own reward model on the other's transitions.
    #   other:  EI/SVO on the GAE of the other's imagined rewards, bootstrapped with MY critic on the other's
    #           observations; --social-scale makes alpha a weight relative to my own advantage
    #   shaped: imagined rewards through the intrinsic-reward path (every formulation)
    python train.py --env-id coin --max-cycles 50 --formulation ei --signal imagined --imagined-critic other \
        --social-scale --alpha 1 --imagined-warmup 20
    python train.py --env-id coin --max-cycles 50 --formulation sia --signal imagined --imagined-critic shaped --alpha 0.1

The rollout / GAE / clipped-surrogate code follows CleanRL's ``ppo_atari.py`` and
``ppo_atari_lstm.py``; the additions are (i) a ``num_agents`` axis with one network per
agent, (ii) the social term ``X_i`` added to the advantage (``--signal value``, ``--signal
imagined --imagined-critic other``), (iii) optional intrinsic reward (``--signal reward``,
``--signal imagined --imagined-critic shaped``) and (iv) a per-agent reward model trained
on the agent's own transitions (``--signal imagined``).  TensorBoard tags ``imagined/corr``
and ``imagined/mae`` compare the imagined rewards of the others with their true rewards
(diagnostic only; the true rewards never enter learning).
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from empathy_marl.agents import LSTMState, MultiAgents
from empathy_marl.args import Args, resolve
from empathy_marl.empathy import ImaginedRewardShaper, RewardSocialShaper, parse_per_agent, social_term, standardize
from empathy_marl.envs import is_prisoners_dilemma, make_envs
from empathy_marl.metrics import MultiAgentEpisodeStatistics


def make_run_name(args: Args) -> str:
    env_tag = args.env_id.split(":")[-1]
    if is_prisoners_dilemma(args.env_id) or args.env_id.startswith("debug"):
        env_tag = f"{env_tag}_n{args.num_agents}"
    if args.formulation == "none":
        method = "none"
    elif args.signal == "imagined":
        method = f"{args.formulation}_imagined_{args.imagined_critic}"
    else:
        method = f"{args.formulation}_{args.signal}"
    if args.social_scale and args.formulation != "none" and args.signal in ("value", "imagined"):
        method += "_sc"
    policy = "lstm" if args.recurrent else "ff"
    params = f"a{args.alpha}_b{args.beta}_p{args.phi}".replace(",", "-")
    return f"{env_tag}__{method}__{policy}__{params}__s{args.seed}__{int(time.time())}"


def cross_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminated: torch.Tensor,
    episode_end: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """GAE of every observed agent's (imagined) reward stream under every observer's critic.

    ``rewards, values, next_values: (T, E, N, N)`` with ``[t, e, i, j]`` = observer ``i``, observed ``j``:
    ``rewards`` = ``i``'s imagined reward of ``j`` at ``t`` (diagonal: own true reward), ``values`` = ``V_i(o_j^t)``,
    ``next_values`` = ``V_i`` of ``j``'s *true* next observation (the terminal observation at episode ends, so
    time-limit truncations are bootstrapped like ``--bootstrap-truncation``).  ``terminated, episode_end: (T, E, N)``
    are the observed agents' flags: ``terminated`` = real terminal (no bootstrap), ``episode_end`` = the episode of
    ``j`` ended at ``t`` (the lambda-recursion is cut there).  Returns ``(T, E, N, N)``.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        boot = (1.0 - terminated[t]).unsqueeze(1)  # (E, 1, N): broadcast over observers
        cont = (1.0 - episode_end[t]).unsqueeze(1)
        delta = rewards[t] + gamma * next_values[t] * boot - values[t]
        adv[t] = last = delta + gamma * lam * cont * last
    return adv


@torch.no_grad()
def imagined_diagnostics(rhat: torch.Tensor, env_rewards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per observer ``i``: correlation and mean absolute error between its imagined rewards of the others,
    ``rhat[..., i, j]`` (``j != i``), and their true rewards ``env_rewards[..., j]`` (diagnostic only, never used
    for learning).  ``rhat: (T, E, N, N)``, ``env_rewards: (T, E, N)`` -> two ``(N,)`` tensors."""
    N = rhat.shape[-1]
    corr = torch.zeros(N, device=rhat.device)
    mae = torch.zeros(N, device=rhat.device)
    for i in range(N):
        others = [j for j in range(N) if j != i]
        pred = rhat[..., i, others].reshape(-1)
        true = env_rewards[..., others].reshape(-1)
        mae[i] = (pred - true).abs().mean()
        pc, tc = pred - pred.mean(), true - true.mean()
        denom = pc.norm() * tc.norm()
        corr[i] = (pc * tc).sum() / denom if denom > 0 else torch.tensor(float("nan"), device=rhat.device)
    return corr, mae


@torch.no_grad()
def compute_next_cross_values(
    agents: MultiAgents,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    dones: torch.Tensor,
    next_done: torch.Tensor,
    cross_state: LSTMState | None,
    chunk_steps: int,
) -> tuple[torch.Tensor, LSTMState | None]:
    """``v_next[t, e, i, j] = V_i(o_j^{t+1})`` for the whole rollout.

    ``obs: (T, E, N, *o)``, ``next_obs: (E, N, *o)``, ``dones: (T, E, N)``, ``next_done: (E, N)``.
    Returns ``v_next: (T, E, N, N)`` (not yet masked at episode ends) and, for recurrent
    critics, the cross hidden state after consuming ``obs[T-1]`` (input for the next rollout).
    """
    T, E, N = obs.shape[:3]
    obs_shape = obs.shape[3:]
    v_all = torch.zeros((T, E, N, N), device=obs.device)
    state = cross_state
    for t0 in range(0, T, chunk_steps):
        t1 = min(T, t0 + chunk_steps)
        x = obs[t0:t1].reshape(((t1 - t0) * E, N) + obs_shape)  # time-major rows
        d = dones[t0:t1].reshape((t1 - t0) * E, N)
        v, state = agents.cross_values(x, state, d)
        v_all[t0:t1] = v.reshape(t1 - t0, E, N, N)
    v_final, _ = agents.cross_values(next_obs, state, next_done)  # (E, N, N)
    v_next = torch.cat([v_all[1:], v_final.unsqueeze(0)], dim=0)
    return v_next, state


def save_checkpoint(path: str, args: Args, agents: MultiAgents, bundle, global_step: int) -> None:
    torch.save(
        {
            "args": asdict(args),
            "agents": agents.state_dict(),
            "global_step": global_step,
            "num_agents": bundle.num_agents,
            "agent_names": bundle.agent_names,
            "obs_type": bundle.obs_type,
        },
        path,
    )


if __name__ == "__main__":
    args = resolve(tyro.cli(Args))
    run_name = make_run_name(args)
    run_path = os.path.join(args.run_dir, run_name)
    os.makedirs(run_path, exist_ok=True)
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    writer = SummaryWriter(run_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    with open(os.path.join(run_path, "args.json"), "w") as f:
        json.dump(asdict(args), f, indent=2)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup ------------------------------------------------------------------------------
    bundle = make_envs(
        args.env_id,
        num_envs=args.num_envs,
        num_cpus=args.num_cpus,
        max_cycles=args.max_cycles,
        num_agents=args.num_agents,
        pd_payoffs=args.pd_payoffs,
    )
    envs = MultiAgentEpisodeStatistics(bundle.envs, args.num_envs, bundle.num_agents)
    E, N, T = args.num_envs, bundle.num_agents, args.num_steps
    obs_shape = bundle.single_observation_space.shape
    obs_dtype = torch.uint8 if bundle.obs_type == "image" else torch.float32
    alpha = parse_per_agent(args.alpha, N, "alpha")
    beta = parse_per_agent(args.beta, N, "beta")
    phi = parse_per_agent(args.phi, N, "phi")
    use_value_signal = args.formulation != "none" and args.signal == "value"
    use_reward_signal = args.formulation != "none" and args.signal == "reward"
    use_imagined = args.signal == "imagined"  # the reward model is trained even for formulation=none (diagnostics)
    imagined_other = use_imagined and args.formulation != "none" and args.imagined_critic == "other"
    imagined_shaped = use_imagined and args.formulation != "none" and args.imagined_critic == "shaped"
    print(f"env={args.env_id} agents={N} envs={E} obs={obs_shape} ({bundle.obs_type}) actions={bundle.single_action_space.n}")
    print(
        f"formulation={args.formulation} signal={args.signal} alpha={alpha.tolist()} beta={beta.tolist()} "
        f"phi={phi.tolist()} recurrent={args.recurrent}"
        + (f" imagined_critic={args.imagined_critic} warmup={args.imagined_warmup}" if use_imagined else "")
        + (" social_scale" if args.social_scale else "")
    )

    agents = MultiAgents(
        N,
        bundle.single_observation_space,
        bundle.single_action_space,
        bundle.obs_type,
        recurrent=args.recurrent,
        lstm_hidden_size=args.lstm_hidden_size,
        reward_model=use_imagined,
    ).to(device)
    optimizer = optim.Adam(agents.parameters(), lr=args.learning_rate, eps=1e-5)
    reward_shaper = (
        RewardSocialShaper(args.formulation, alpha, beta, phi, args.gamma, args.reward_lambda, E, N, device)
        if use_reward_signal
        else None
    )
    imagined_shaper = (
        ImaginedRewardShaper(args.formulation, alpha, beta, phi, args.gamma, args.reward_lambda, E, N, device)
        if imagined_shaped
        else None
    )
    eye = torch.eye(N, dtype=torch.bool, device=device)

    def to_obs_tensor(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=obs_dtype, device=device).reshape((E, N) + obs_shape)

    # ALGO Logic: Storage setup  (layout: time, env copy, agent) -----------------------------------
    obs = torch.zeros((T, E, N) + obs_shape, dtype=obs_dtype, device=device)
    actions = torch.zeros((T, E, N), dtype=torch.long, device=device)
    logprobs = torch.zeros((T, E, N), device=device)
    rewards = torch.zeros((T, E, N), device=device)  # what PPO learns from (includes the intrinsic term for signal=reward)
    env_rewards = torch.zeros((T, E, N), device=device)  # raw environment rewards, for logging
    social = torch.zeros((T, E, N), device=device)  # the social term: X_i (signal=value) or intrinsic reward (signal=reward)
    dones = torch.zeros((T, E, N), device=device)
    values = torch.zeros((T, E, N), device=device)
    # signal=imagined: the TRUE next observation of every step (the terminal observation at episode ends, where
    # `obs[t+1]` is already the reset observation) and the real-termination flags, for the reward model / cross GAE
    next_obs_buf = torch.zeros_like(obs) if use_imagined else None
    term_buf = torch.zeros((T, E, N), device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    raw_obs, infos = envs.reset(seed=args.seed)
    next_obs = to_obs_tensor(bundle.extract_obs(raw_obs))
    next_done = torch.zeros((E, N), device=device)
    next_lstm_state = agents.initial_lstm_state(E, device)
    cross_lstm_state = agents.initial_cross_lstm_state(E, device) if use_value_signal else None
    episodes_done = 0

    for iteration in range(1, args.num_iterations + 1):
        initial_lstm_state = (
            None if next_lstm_state is None else (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        )
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(T):
            global_step += E
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agents.get_action_and_value(
                    next_obs, next_lstm_state, next_done
                )
            values[step] = value
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            raw_obs, reward, terminations, truncations, infos = envs.step(action.reshape(-1).cpu().numpy())
            reward_t = torch.as_tensor(np.asarray(reward), dtype=torch.float32, device=device).reshape(E, N)
            term_t = torch.as_tensor(np.asarray(terminations), dtype=torch.float32, device=device).reshape(E, N)
            trunc_t = torch.as_tensor(np.asarray(truncations), dtype=torch.float32, device=device).reshape(E, N)
            env_rewards[step] = reward_t
            term_buf[step] = term_t
            done_t = torch.maximum(term_t, trunc_t)
            terminal_obs_t = None
            if bool(done_t.any()):
                raw_terminal = bundle.extract_terminal_obs(infos)
                terminal_obs_t = None if raw_terminal is None else to_obs_tensor(raw_terminal)

            # Time-limit truncation is not a real terminal: bootstrap from V(terminal observation)
            # by folding gamma * V into the reward (the GAE below then treats the step as terminal).
            if args.bootstrap_truncation:
                trunc_only = trunc_t * (1.0 - term_t)
                if bool(trunc_only.any()) and terminal_obs_t is not None:
                    with torch.no_grad():
                        terminal_value, _ = agents.get_values(
                            terminal_obs_t, next_lstm_state, torch.zeros((E, N), device=device)
                        )
                    reward_t = reward_t + args.gamma * terminal_value * trunc_only

            if reward_shaper is not None:  # signal=reward: intrinsic reward built from the others' (smoothed) rewards
                intrinsic = reward_shaper(env_rewards[step], dones[step])
                social[step] = intrinsic
                reward_t = reward_t + intrinsic
            rewards[step] = reward_t

            next_done = done_t
            next_obs = to_obs_tensor(bundle.extract_obs(raw_obs))
            if next_obs_buf is not None:  # true next observation of this step (terminal obs where an episode ended)
                next_obs_buf[step] = next_obs
                if terminal_obs_t is not None:
                    ended = done_t.bool()
                    next_obs_buf[step][ended] = terminal_obs_t[ended]

            # Episode statistics.  Env copies run in lockstep (fixed episode length), so all E of them usually
            # finish on the same step; log the mean over the finished copies once per step rather than E points
            # with the same global_step (identical curves after binning, E times fewer events to parse).
            finished = [infos[e * N]["ma_episode"] for e in range(E) if "ma_episode" in infos[e * N]]
            if finished:
                episodes_done += len(finished)
                r_mean = np.mean([ep["r"] for ep in finished], axis=0)
                scalars = {
                    key: float(np.mean([ep[src] for ep in finished]))
                    for key, src in (
                        ("collective_return", "collective"),
                        ("efficiency", "efficiency"),
                        ("equality", "equality"),
                        ("sustainability", "sustainability"),
                        ("episode_length", "l"),
                    )
                }
                print(
                    f"global_step={global_step}, episodes={episodes_done}, collective_return={scalars['collective_return']:.2f}, "
                    f"per_agent={np.round(r_mean, 2).tolist()}, equality={scalars['equality']:.3f}, "
                    f"sustainability={scalars['sustainability']:.1f} (mean of {len(finished)} episodes)"
                )
                for key, value in scalars.items():
                    writer.add_scalar(f"charts/{key}", value, global_step)
                for i, name in enumerate(bundle.agent_names):
                    writer.add_scalar(f"charts/episodic_return/{name}", r_mean[i], global_step)
                # environment-specific per-episode statistics (e.g. Coin Game: charts/cooperation_rate/<agent>);
                # nan = "undefined for this agent in this episode" and is left out of the mean
                for key in sorted({k for ep in finished for k in ep.get("stats", {})}):
                    vals = np.array([ep["stats"][key] for ep in finished if key in ep.get("stats", {})])  # (n, N)
                    defined = ~np.isnan(vals)
                    for i, name in enumerate(bundle.agent_names):
                        if defined[:, i].any():
                            writer.add_scalar(f"charts/{key}/{name}", vals[defined[:, i], i].mean(), global_step)

        # signal=imagined: imagine every other agent's reward with my own reward model -----------------
        social_on = iteration > args.imagined_warmup
        episode_end = torch.cat([dones[1:], next_done.unsqueeze(0)], dim=0)  # (T, E, N): the episode of agent j ended at t
        if use_imagined:
            with torch.no_grad():
                rhat = agents.cross_rewards(
                    obs.reshape((T * E, N) + obs_shape), actions.reshape(T * E, N),
                    next_obs_buf.reshape((T * E, N) + obs_shape),
                ).reshape(T, E, N, N)  # [t, e, i, j] = f_i(o_j^t, a_j^t, o_j^{t+1})
                imagined_corr, imagined_mae = imagined_diagnostics(rhat, env_rewards)
                rhat = torch.where(eye, env_rewards.unsqueeze(-1).expand(-1, -1, -1, N), rhat)  # diagonal: own true reward
            if imagined_shaped:  # the reward signal with imagined rewards: intrinsic reward, enters the return
                with torch.no_grad():
                    for t in range(T):
                        intrinsic = imagined_shaper(rhat[t], dones[t])
                        if social_on:
                            social[t] = intrinsic
                            rewards[t] = rewards[t] + intrinsic
                        else:
                            social[t] = 0.0

        # bootstrap value if not done ---------------------------------------------------------
        with torch.no_grad():
            next_value, _ = agents.get_values(next_obs, next_lstm_state, next_done)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(T)):
                if t == T - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

            scaled_advantages = standardize(advantages, dims=(0, 1)) if args.social_scale else advantages
            # signal=value: X_i(t) = F_i(V_i(s_j^{t+1})), a constant coefficient computed with the rollout-time critic
            if use_value_signal:
                v_next, cross_lstm_state = compute_next_cross_values(
                    agents, obs, next_obs, dones, next_done, cross_lstm_state, args.cross_value_chunk
                )
                z = standardize(v_next, dims=(0, 1)) if args.social_scale else v_next  # per (i, j) over the batch
                social[:] = social_term(args.formulation, z, alpha, beta, phi) * (1.0 - episode_end)
            # signal=imagined, critic=other: z_ij = GAE of j's imagined rewards under MY critic on j's observations
            # (delta = rhat_ij + gamma V_i(o_j') - V_i(o_j)), z_ii = my own advantage; F_i(z) is added to the advantage
            if imagined_other:
                flat_obs = obs.reshape((T * E, N) + obs_shape)
                flat_next = next_obs_buf.reshape((T * E, N) + obs_shape)
                v_cross, _ = agents.cross_values(flat_obs)  # feed-forward only (resolve() enforces it)
                v_cross_next, _ = agents.cross_values(flat_next)
                adv_cross = cross_gae(
                    rhat, v_cross.reshape(T, E, N, N), v_cross_next.reshape(T, E, N, N),
                    term_buf, episode_end, args.gamma,
                    args.gae_lambda if args.imagined_lambda is None else args.imagined_lambda,
                )
                z = torch.where(eye, scaled_advantages.unsqueeze(-1).expand(-1, -1, -1, N), adv_cross)
                if args.social_scale:
                    z = torch.where(eye, z, standardize(adv_cross, dims=(0, 1)))
                social[:] = social_term(args.formulation, z, alpha, beta, phi) if social_on else 0.0
            # signal=reward / imagined-shaped: the intrinsic term already entered `rewards` (and GAE)
            advantage_coef = scaled_advantages + social if (use_value_signal or imagined_other) else advantages

        # flatten the batch: row = t * E + e (time-major, as required by the LSTM) ------------------
        b_obs = obs.reshape((T * E, N) + obs_shape)
        b_logprobs = logprobs.reshape(T * E, N)
        b_actions = actions.reshape(T * E, N)
        b_dones = dones.reshape(T * E, N)
        b_advantages = advantage_coef.reshape(T * E, N)  # GAE advantage (+ X_i for signal=value)
        b_returns = returns.reshape(T * E, N)
        b_values = values.reshape(T * E, N)
        if use_imagined:  # reward-model regression targets: own transitions -> own environment reward
            b_next_obs = next_obs_buf.reshape((T * E, N) + obs_shape)
            b_env_rewards = env_rewards.reshape(T * E, N)

        # Optimizing the policy and value network ---------------------------------------------------
        if args.recurrent:
            envsperbatch = E // args.num_minibatches
            envinds = np.arange(E)
            flatinds = np.arange(T * E).reshape(T, E)
        else:
            b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            if args.recurrent:
                np.random.shuffle(envinds)
                minibatches = [
                    (flatinds[:, envinds[s : s + envsperbatch]].ravel(), envinds[s : s + envsperbatch])
                    for s in range(0, E, envsperbatch)
                ]
            else:
                np.random.shuffle(b_inds)
                minibatches = [
                    (b_inds[s : s + args.minibatch_size], None) for s in range(0, args.batch_size, args.minibatch_size)
                ]
            for mb_inds, mbenvinds in minibatches:
                mb_lstm_state = (
                    None
                    if initial_lstm_state is None
                    else (initial_lstm_state[0][:, :, mbenvinds], initial_lstm_state[1][:, :, mbenvinds])
                )
                _, newlogprob, entropy, newvalue, _ = agents.get_action_and_value(
                    b_obs[mb_inds], mb_lstm_state, b_dones[mb_inds], b_actions[mb_inds]
                )  # all (minibatch, N)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html   (per agent)
                    old_approx_kl = (-logratio).mean(dim=0)
                    approx_kl = ((ratio - 1) - logratio).mean(dim=0)
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean(dim=0)]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean(dim=0)) / (mb_advantages.std(dim=0) + 1e-8)

                # Policy loss (per agent)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean(dim=0)

                # Value loss (per agent)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean(dim=0)
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean(dim=0)

                entropy_loss = entropy.mean(dim=0)
                loss_per_agent = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                if use_imagined:  # reward model: regress my own reward on my own observation transition
                    pred_reward = agents.predict_own_rewards(b_obs[mb_inds], b_actions[mb_inds], b_next_obs[mb_inds])
                    rm_loss = 0.5 * ((pred_reward - b_env_rewards[mb_inds]) ** 2).mean(dim=0)
                    loss_per_agent = loss_per_agent + args.reward_model_coef * rm_loss
                loss = loss_per_agent.sum()  # networks are independent, so the sum decouples

                optimizer.zero_grad()
                loss.backward()
                for agent in agents.agents:
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl.max().item() > args.target_kl:
                break

        # TRY NOT TO MODIFY: record rewards for plotting purposes ------------------------------------
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            explained_var = np.where(var_y == 0, np.nan, 1 - np.var(y_true - y_pred, axis=0) / var_y)
        mean_clipfrac = torch.stack(clipfracs).mean(dim=0)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/rollout_collective_env_reward", env_rewards.sum(dim=(0, 2)).mean().item(), global_step)
        if bundle.cooperate_action is not None:
            coop = (actions == bundle.cooperate_action).float().mean(dim=(0, 1))
        for i, name in enumerate(bundle.agent_names):
            writer.add_scalar(f"losses/value_loss/{name}", v_loss[i].item(), global_step)
            writer.add_scalar(f"losses/policy_loss/{name}", pg_loss[i].item(), global_step)
            writer.add_scalar(f"losses/entropy/{name}", entropy_loss[i].item(), global_step)
            writer.add_scalar(f"losses/old_approx_kl/{name}", old_approx_kl[i].item(), global_step)
            writer.add_scalar(f"losses/approx_kl/{name}", approx_kl[i].item(), global_step)
            writer.add_scalar(f"losses/clipfrac/{name}", mean_clipfrac[i].item(), global_step)
            writer.add_scalar(f"losses/explained_variance/{name}", explained_var[i], global_step)
            writer.add_scalar(f"charts/rollout_env_reward/{name}", env_rewards[:, :, i].sum(dim=0).mean().item(), global_step)
            # social term: X_i (signal=value, compare with the advantage scale) or intrinsic reward
            # (signal=reward, compare with the environment reward scale)
            writer.add_scalar(f"social/term_mean/{name}", social[:, :, i].mean().item(), global_step)
            writer.add_scalar(f"social/term_abs_mean/{name}", social[:, :, i].abs().mean().item(), global_step)
            writer.add_scalar(f"social/advantage_abs_mean/{name}", advantages[:, :, i].abs().mean().item(), global_step)
            writer.add_scalar(f"social/env_reward_abs_mean/{name}", env_rewards[:, :, i].abs().mean().item(), global_step)
            if use_imagined:
                writer.add_scalar(f"losses/reward_model/{name}", rm_loss[i].item(), global_step)
                # how well agent i's reward model, built from its OWN rewards, predicts the OTHERS' true rewards
                writer.add_scalar(f"imagined/corr/{name}", imagined_corr[i].item(), global_step)
                writer.add_scalar(f"imagined/mae/{name}", imagined_mae[i].item(), global_step)
            if bundle.cooperate_action is not None:
                writer.add_scalar(f"charts/cooperation_rate/{name}", coop[i].item(), global_step)
        print(f"iteration={iteration}/{args.num_iterations} global_step={global_step} SPS={sps}")

        if args.checkpoint_every and iteration % args.checkpoint_every == 0:
            save_checkpoint(os.path.join(run_path, f"agents_{iteration}.pt"), args, agents, bundle, global_step)

    if args.save_model:
        save_checkpoint(os.path.join(run_path, "agents.pt"), args, agents, bundle, global_step)
        print(f"saved checkpoint to {os.path.join(run_path, 'agents.pt')}")

    envs.close()
    writer.close()
