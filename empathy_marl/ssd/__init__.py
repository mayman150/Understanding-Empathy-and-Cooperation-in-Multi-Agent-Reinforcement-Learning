"""Vendored Sequential Social Dilemma (SSD) grid-world code: the Clean Up game of Hughes et al. (2018).

Source: https://github.com/011235813/sequential_social_dilemma_games (Jiachen Yang's fork, used for the
Clean Up experiments of "Learning to Incentivize Other Learning Agents", NeurIPS 2020) of
https://github.com/eugenevinitsky/sequential_social_dilemma_games (Eugene Vinitsky et al.), MIT licence.
See LICENSE and NOTICE in this directory for the licence text and the list of local changes.

The dynamics (movement and conflict resolution, cleaning beam, apple / waste spawning, rewards) are the
upstream ones.  :mod:`empathy_marl.cleanup` wraps :class:`CleanupEnv` as a PettingZoo ``ParallelEnv``
with plane (or RGB) observations for this repository's training code.
"""
from empathy_marl.ssd.cleanup import CleanupEnv
from empathy_marl.ssd.maps import CLEANUP_10x10_SYM, CLEANUP_MAP, CLEANUP_PARAMS, CLEANUP_SMALL_SYM, MAPS

__all__ = ["CleanupEnv", "CLEANUP_10x10_SYM", "CLEANUP_MAP", "CLEANUP_PARAMS", "CLEANUP_SMALL_SYM", "MAPS"]
