# Legacy scripts (superseded by `train.py`)

These are the original per-formulation CleanRL-derived scripts used for the course
report.  They are kept for reference only and **should not be used for new experiments**:
the results they produced are not trustworthy because of the following defects, all fixed
in `train.py` / `empathy_marl/`:

| Issue | Where | Effect |
|---|---|---|
| `torch.cat(per_agent, dim=0).reshape(-1, num_agents)` used to build `(batch, agent)` tensors | all `MultiAgents.get_actions_and_values` callers | `newlogprob`, `newvalue`, `entropy` and the social term are paired with the **wrong** `(agent, timestep)` cells (only 0.2% aligned): the PPO ratio, value regression and social term are noise |
| Social term computed with gradients enabled and not detached | `ppo_ei/sia/svo/ia.py` | policy loss back-propagates into the critic and CNN trunk; SIA/IA reward the critic for predicting the same value everywhere |
| `for i in range(1, num_steps - 1)` when building next observations | same | `b_next_obs[0]` is a black frame; episode boundaries ignored; ~330 MB re-allocated per minibatch |
| `next_values` referenced but never defined | `ppo_svo.py` | `NameError` on the first update: SVO cannot have run as committed |
| `mb_advantages = b_advantages + sum_j V_i(s_j')` (alpha = 1, sum not mean) | `ppo.py` ("baseline") | the baseline curve is not plain PPO; `ppo_old.py` is |
| `run_name`/pickle path use `ppo_ei_*` | `ppo_ia.py` | IA runs overwrite EI logs and checkpoints |
| loaded pickle immediately overwritten by a fresh `MultiAgents(...)` | `generate_simulation.py` | "simulations" used untrained agents |
| live `pdb.set_trace()` | `main.py` | script halts |
| sustainability counts reward events, counters never reset after auto-reset, `peace` printed from `'s'` | `record_ma_episode_statistics.py` | social metrics wrong after the first episode |

`sb3_train.py`, `utils.py`, `utils_gym.py` are the Melting Pot Stable-Baselines3 example
and its helpers (unused by the PPO scripts).
