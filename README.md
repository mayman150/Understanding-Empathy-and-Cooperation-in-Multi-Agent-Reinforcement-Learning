# Understanding Empathy and Cooperation in Multi-Agent Reinforcement Learning

## Introduction
In navigating complex social dilemmas, human groups often seek avenues for cooperation. However, traditional models rooted in behavioral economics face limitations, as they can only elucidate cooperative behavior within simplified, static scenarios such as matrix games, neglecting temporal dynamics. Recent advancements in multi-agent reinforcement learning offer a more comprehensive approach. Research indicates that while not universally exhibited, many human individuals display a preference against inequity, shaping their behavior in social dilemmas. This inclination tends to promote pro-social tendencies and feel negative sentiments if they are the sole defectors, influencing the resolution of matrix game social dilemmas. Nevertheless, assuming access to rewards is a strong assumption to make, as in real-world scenarios humans cannot accurately perceive others’ feelings. In our project, we aim to enforce cooperation among agents without granting access to rewards. From our four proposed formulations, Simple Inequity Aversion (SIA) shows promising results that agents may potentially learn without access to rewards.


## Research Questions
In this work, we expect to understand more about how the expectation of future rewards that other agents receive can change empathetic social interactions. More empathetic agents will probably expect all the others to be as well, and the same goes for more egotistical ones. We therefore ask:

- How does this impact the equilibrium?
- Do empathetic agents become even more unselfish since they expect everyone to behave similarly?

Moreover, how does this translate to human interactions? If someone is interacting with someone that they do not know, they will likely model the other agents similarly to how they behave themselves and we expect to gather insights from such social interactions.

We argue that individuals typically rely solely on their own perception of rewards, value function, to generate empathetic behavior for two main reasons: firstly, because one cannot directly experience others' rewards, and secondly, because it is natural to assume that others will behave in a manner similar to oneself, since the value function incorporates the policy for future rewards.

## Environment
In our project, agents will interact and learn in the **Common Harvest Open** environment from Agapiou et al. (2022). In the game, apples are scattered across the environment and can be eaten for a reward of 1. The rate at which consumed apples grow back depends on how many other apples are nearby within a 2-unit radius. If there are three or more apples nearby, the probability of regrowth is 0.025. If there are exactly two apples, it's 0.005. With just one apple nearby, the probability drops to 0.001, and if there are no nearby apples, regrowth probability is 0. This means if all apples in a group are eaten and there are no others nearby, the group cannot recover. So, agents must be cautious when eating apples from a group. In a single-agent situation, there's usually no reason to eat the last apple in a group, except maybe at the end of a task. However, in a multi-agent scenario, each agent has an incentive to eat the last apple before another agent does. This leads to a common problem known as the tragedy of the commons, which is why the environment is named as such. 

![The Common Harvest Open environment implementation](img/harvest.png)
*Figure: The Common Harvest Open environment implementation.*

## Results
We graphed the sum of the rewards for each episode as a measure of "learning". 

![Sum of the undiscounted rewards per episode for 3 of the 7 agents](img/charts_rewards_player_3.svg)
![Sum of the undiscounted rewards per episode for 3 of the 7 agents](img/charts_rewards_player_4.svg)
![Sum of the undiscounted rewards per episode for 3 of the 7 agents](img/charts_rewards_player_6.svg)
*Figure: Sum of the undiscounted rewards per episode for 3 of the 7 agents.*

Figure above shows the graphs for 3 of the 7 agents where each curve corresponds to a variation of the algorithms tested. We can see that most of the curves stabilize at around 10 rewards per episode, which is consistent with what we would expect if a tragedy of the commons occurred. They also seem to follow our baseline (the green curve).

However, there is a single blue curve that seems to achieve a much higher reward per episode consistently for all agents. This corresponds to one of our SIA agents, which seemed to learn to cooperate to some degree. While this result is not conclusive, this indicates a good path of investigation moving forward. We hope to do a more thorough parameter sweep of the SIA variation using several seeds in the future.

> **Note on the preliminary results above.** They were produced with the scripts now in
> `legacy/`, which contained a tensor-alignment bug that made PPO's ratio, value targets and
> the social term compare unrelated (agent, timestep) pairs (see `legacy/README.md`).
> They should be regenerated with `train.py` before being used in the paper.

## Code

```
train.py                 single entrypoint: independent PPO learners (CleanRL style) + social term
evaluate.py              roll out a checkpoint, report metrics, optional video
empathy_marl/
  envs.py                env factory: `pd`, `meltingpot:<substrate>`, `debug:image`
  prisoners_dilemma.py   repeated Prisoner's Dilemma (PettingZoo ParallelEnv)
  agents.py              CNN / MLP trunk, optional LSTM (ppo_atari_lstm.py), one network per agent
  empathy.py             X_i formulations (ei, svo, sia, ia) + reward-based inequity aversion baseline
  metrics.py             efficiency / equality / sustainability per episode
scripts/sweep_pd.sh, sweep_meltingpot.sh, summarize_runs.py
tests/                   pytest suite (alignment, formulas, envs, end-to-end smoke tests)
legacy/                  original scripts, kept for reference only
```

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # Melting Pot (dm-meltingpot / dmlab2d) is Linux only
python -m pytest tests -q                # 36 tests, ~20 s, no Melting Pot needed
```

### Train

```bash
# 1. sanity check: repeated Prisoner's Dilemma, per-agent social preference (selfish vs. empathetic)
python train.py --env-id pd --max-cycles 100 --formulation sia --alpha 0,0.5 --seed 1 \
    --num-envs 8 --num-steps 128 --total-timesteps 1000000

# 2. Commons Harvest, feed-forward CNN, one formulation / seed
python train.py --env-id meltingpot:commons_harvest__open --formulation ei --alpha 0.01 --seed 1

# 3. partial observability: recurrent policy (needs num_envs % num_minibatches == 0)
python train.py --env-id meltingpot:commons_harvest__open --formulation sia --alpha 0.1 \
    --recurrent --num-envs 4 --num-minibatches 4

# 4. reward-based baseline (Hughes et al. 2018 inequity aversion; uses other agents' rewards)
python train.py --env-id meltingpot:clean_up --formulation reward_ia --alpha 5 --beta 0.05

# sweeps over formulations x alpha x seeds
SEEDS="1 2 3 4 5" scripts/sweep_pd.sh
SUBSTRATE=clean_up EXTRA="--recurrent --num-envs 4" scripts/sweep_meltingpot.sh
python scripts/summarize_runs.py runs/pd            # mean +/- std over seeds per configuration
```

`--formulation` is one of `none` (plain PPO), `ei`, `svo`, `sia`, `ia` (value based, no access
to others' rewards) or `reward_ia` (reward based).  `--alpha/--beta/--phi` accept one value or
a comma-separated per-agent list.  Every run writes TensorBoard logs, `args.json` and
`agents.pt` to `runs/<env>__<formulation>__<ff|lstm>__<params>__s<seed>__<time>/`.
Useful TensorBoard tags: `charts/collective_return`, `charts/episodic_return/<agent>`,
`charts/equality`, `charts/sustainability`, `charts/cooperation_rate/<agent>` (PD only),
`social/X_abs_mean/<agent>` vs `social/advantage_abs_mean/<agent>` (scale of the social term).

### Evaluate

```bash
python evaluate.py --checkpoint runs/<run>/agents.pt --episodes 10 --out metrics.json
python evaluate.py --checkpoint runs/<run>/agents.pt --episodes 1 --video harvest.mp4
```

### How the social term enters PPO

With `v_next[t, i, j] = V_i(s_j^{t+1})` (agent *i*'s own critic on agent *j*'s next observation,
computed once per rollout under `no_grad`), the minibatch coefficient is
`A_GAE + X_i`, which is then normalised per agent and used in the clipped surrogate exactly as
in CleanRL.  `X_i` is a constant with respect to the parameters (like the advantage); it is
masked to zero at episode ends.  For recurrent critics agent *i* keeps a separate hidden state
per observed agent *j* ("what if I had seen what *j* saw").

### Observations and the value-based mechanism

For `V_i(s_j)` to say anything about *j*'s situation, `s_j` has to be **agent specific**.  The
Melting Pot egocentric `RGB` view is.  A "fully observable" variant that gives every agent the
same global `WORLD.RGB` frame would make `V_i(s_j) = V_i(s_i)` and switch the social term off;
a full-information setting therefore needs an agent-centred observation (e.g. global frame plus
the agent's own egocentric view, or the agent's position marked in an extra channel).  See
`empathy_marl/envs.py` (`MELTINGPOT_OBS_KEY`) for the extension point.


## Citation
Agapiou, J., et al. (2022). *Melting Pot 2.0: Measuring and Understanding Multi-Agent Learning Complexity*. Proceedings of the 39th International Conference on Machine Learning.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
