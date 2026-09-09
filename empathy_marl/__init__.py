"""Value-based social preferences for independent PPO learners in sequential social dilemmas.

Modules
-------
- ``envs``               environment factory (Melting Pot substrates, repeated Prisoner's Dilemma)
- ``prisoners_dilemma``  two-player repeated Prisoner's Dilemma as a PettingZoo ParallelEnv
- ``agents``             CleanRL-style actor-critic (CNN / MLP trunk, optional LSTM) + multi-agent container
- ``empathy``            the social terms X_i (EI, SVO, SIA, IA) and the reward-based inequity-aversion baseline
- ``metrics``            per-episode social-outcome metrics (efficiency, equality, sustainability)
"""
