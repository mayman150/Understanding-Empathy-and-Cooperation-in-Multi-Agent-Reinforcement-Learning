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

The rollout / GAE / clipped-surrogate code follows CleanRL's ``ppo_atari.py`` and
``ppo_atari_lstm.py``; the only additions are (i) a ``num_agents`` axis with one network
per agent, (ii) the social term ``X_i`` added to the advantage (``--signal value``) and
(iii) optional intrinsic reward (``--signal reward``).
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
from empathy_marl.empathy import RewardSocialShaper, parse_per_agent, social_term
from empathy_marl.envs import is_prisoners_dilemma, make_envs
from empathy_marl.metrics import MultiAgentEpisodeStatistics


def make_run_name(args: Args) -> str:
    env_tag = args.env_id.split(":")[-1]
    if is_prisoners_dilemma(args.env_id) or args.env_id.startswith("debug"):
        env_tag = f"{env_tag}_n{args.num_agents}"
    method = args.formulation if args.formulation == "none" else f"{args.formulation}_{args.signal}"
    policy = "lstm" if args.recurrent else "ff"
    params = f"a{args.alpha}_b{args.beta}_p{args.phi}".replace(",", "-")
    return f"{env_tag}__{method}__{policy}__{params}__s{args.seed}__{int(time.time())}"


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
    print(f"env={args.env_id} agents={N} envs={E} obs={obs_shape} ({bundle.obs_type}) actions={bundle.single_action_space.n}")
    print(
        f"formulation={args.formulation} signal={args.signal} alpha={alpha.tolist()} beta={beta.tolist()} "
        f"phi={phi.tolist()} recurrent={args.recurrent}"
    )

    agents = MultiAgents(
        N,
        bundle.single_observation_space,
        bundle.single_action_space,
        bundle.obs_type,
        recurrent=args.recurrent,
        lstm_hidden_size=args.lstm_hidden_size,
    ).to(device)
    optimizer = optim.Adam(agents.parameters(), lr=args.learning_rate, eps=1e-5)
    reward_shaper = (
        RewardSocialShaper(args.formulation, alpha, beta, phi, args.gamma, args.reward_lambda, E, N, device)
        if use_reward_signal
        else None
    )

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

            # Time-limit truncation is not a real terminal: bootstrap from V(terminal observation)
            # by folding gamma * V into the reward (the GAE below then treats the step as terminal).
            if args.bootstrap_truncation:
                trunc_only = trunc_t * (1.0 - term_t)
                if bool(trunc_only.any()):
                    terminal_obs = bundle.extract_terminal_obs(infos)
                    if terminal_obs is not None:
                        with torch.no_grad():
                            terminal_value, _ = agents.get_values(
                                to_obs_tensor(terminal_obs), next_lstm_state, torch.zeros((E, N), device=device)
                            )
                        reward_t = reward_t + args.gamma * terminal_value * trunc_only

            if reward_shaper is not None:  # signal=reward: intrinsic reward built from the others' (smoothed) rewards
                intrinsic = reward_shaper(env_rewards[step], dones[step])
                social[step] = intrinsic
                reward_t = reward_t + intrinsic
            rewards[step] = reward_t

            next_done = torch.maximum(term_t, trunc_t)
            next_obs = to_obs_tensor(bundle.extract_obs(raw_obs))

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

            # signal=value: X_i(t) = F_i(V_i(s_j^{t+1})), a constant coefficient computed with the rollout-time critic
            if use_value_signal:
                v_next, cross_lstm_state = compute_next_cross_values(
                    agents, obs, next_obs, dones, next_done, cross_lstm_state, args.cross_value_chunk
                )
                next_nonterminal_all = 1.0 - torch.cat([dones[1:], next_done.unsqueeze(0)], dim=0)  # (T, E, N)
                social[:] = social_term(args.formulation, v_next, alpha, beta, phi) * next_nonterminal_all
            # signal=reward: the intrinsic term already entered `rewards` (and GAE) during the rollout
            advantage_coef = advantages + social if use_value_signal else advantages

        # flatten the batch: row = t * E + e (time-major, as required by the LSTM) ------------------
        b_obs = obs.reshape((T * E, N) + obs_shape)
        b_logprobs = logprobs.reshape(T * E, N)
        b_actions = actions.reshape(T * E, N)
        b_dones = dones.reshape(T * E, N)
        b_advantages = advantage_coef.reshape(T * E, N)  # GAE advantage (+ X_i for signal=value)
        b_returns = returns.reshape(T * E, N)
        b_values = values.reshape(T * E, N)

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
