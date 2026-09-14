"""Reference collective returns for Clean Up from hand-scripted policies.

Puts numbers on "what is a good collective return" for the 10x10, 3-agent, 50-step configuration used in the
experiments: what nobody-cleans gives (the free-riding equilibrium), what a fixed division of labour gives
(1 or 2 dedicated cleaners, the rest eating), what everybody-cleans-then-eats gives, and what random actions
give.  The scripted agents see the true map, so these are upper references for policies of that *shape*, not
optimal play.

    python scripts/cleanup_reference.py --episodes 200
"""
from __future__ import annotations

import argparse
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from empathy_marl.cleanup import CLEAN, DOWN, LEFT, RIGHT, STAY, UP, CleanupEnv  # noqa: E402

BEAM_LEN = 5


def agent_positions(env: CleanupEnv) -> list[tuple[int, int]]:
    return [tuple(int(x) for x in env.ssd.agents[sid].get_pos()) for sid in env._ssd_ids]


def reach(world: np.ndarray, pos: tuple[int, int], occupied: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Waste cells a clean beam fired upward from ``pos`` would remove (3 lanes, first waste per lane; agents and
    walls stop a lane), mirroring ``MapEnv.update_map_fire`` with ``blocking_cells=['H']``."""
    r, c = pos
    lanes = [((r, c), BEAM_LEN), ((r + 1, c - 1), BEAM_LEN), ((r + 1, c + 1), BEAM_LEN)]  # side lanes start one row behind
    hit = set()
    for (r0, c0), length in lanes:
        rr, cc = r0 - 1, c0
        for _ in range(length):
            if not (0 <= rr < world.shape[0] and 0 <= cc < world.shape[1]) or world[rr, cc] == "@":
                break
            if (rr, cc) in occupied and (rr, cc) != pos:
                if world[rr, cc] == "H":
                    hit.add((rr, cc))
                break
            if world[rr, cc] == "H":
                hit.add((rr, cc))
                break
            rr -= 1
    return hit


def step_towards(pos, target, occupied, world) -> int:
    """One Manhattan step from ``pos`` to ``target`` avoiding walls and other agents; STAY if boxed in."""
    dr, dc = target[0] - pos[0], target[1] - pos[1]
    moves = []
    if abs(dr) >= abs(dc):
        if dr:
            moves.append(DOWN if dr > 0 else UP)
        if dc:
            moves.append(RIGHT if dc > 0 else LEFT)
    else:
        if dc:
            moves.append(RIGHT if dc > 0 else LEFT)
        if dr:
            moves.append(DOWN if dr > 0 else UP)
    for m in moves:
        nxt = next_cell(pos, m)
        if world[nxt] != "@" and nxt not in occupied:
            return m
    return STAY


def next_cell(pos, move) -> tuple[int, int]:
    d = {LEFT: (0, -1), RIGHT: (0, 1), UP: (-1, 0), DOWN: (1, 0), STAY: (0, 0), CLEAN: (0, 0)}[move]
    return (pos[0] + d[0], pos[1] + d[1])


def cleaner_action(world, pos, others, river_cells) -> int:
    occupied = set(others)
    if reach(world, pos, occupied):
        return CLEAN
    waste = list(zip(*np.where(world == "H")))
    if not waste:
        return STAY
    # go to the river cell whose beam removes the most waste (nearest such cell on ties)
    best, best_key = None, None
    for cell in river_cells:
        if cell in occupied:
            continue
        n = len(reach(world, cell, occupied))
        if n == 0:
            continue
        key = (-n, abs(cell[0] - pos[0]) + abs(cell[1] - pos[1]))
        if best_key is None or key < best_key:
            best, best_key = cell, key
    if best is None or best == pos:
        return STAY
    return step_towards(pos, best, occupied, world)


def eater_action(world, pos, others, wait_at) -> int:
    occupied = set(others)
    apples = list(zip(*np.where(world == "A")))
    if apples:
        target = min(apples, key=lambda a: abs(a[0] - pos[0]) + abs(a[1] - pos[1]))
        return step_towards(pos, target, occupied, world)
    if pos == wait_at or wait_at in occupied:
        return STAY
    return step_towards(pos, wait_at, occupied, world)


def run(policy: str, episodes: int, seed: int, num_rounds: int = 50):
    env = CleanupEnv(num_agents=3, num_rounds=num_rounds)
    rng = np.random.default_rng(seed)
    base = env.ssd.base_map
    river_cells = [(r, c) for r in range(base.shape[0]) for c in range(base.shape[1]) if base[r, c] in ("R", "H")]
    wait_spots = [(2, 6), (5, 6), (7, 6)]  # floor cells next to the orchard column
    totals, per_agent, density, prob = [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        roles = {"none": ["eat"] * 3, "one_cleaner": ["clean", "eat", "eat"], "two_cleaners": ["clean", "clean", "eat"],
                 "random": ["random"] * 3, "all_switch": ["switch"] * 3}[policy]
        ret = np.zeros(3)
        phase_clean = True  # all_switch: everybody cleans until the river is nearly clean, then everybody eats
        while env.agents:
            world = env.ssd.world_map
            pos = agent_positions(env)
            n_waste = int((world == "H").sum())
            if policy == "all_switch":
                if phase_clean and n_waste <= 1:
                    phase_clean = False
                elif not phase_clean and n_waste >= 7 and not (world == "A").any():
                    phase_clean = True  # back to the river once the orchard has stopped and is empty
            actions = {}
            for i, agent in enumerate(env.possible_agents):
                others = [p for j, p in enumerate(pos) if j != i]
                role = roles[i]
                if role == "switch":
                    role = "clean" if phase_clean else "eat"
                if role == "random":
                    a = int(rng.integers(6))
                elif role == "clean":
                    a = cleaner_action(world, pos[i], others, river_cells)
                else:
                    a = eater_action(world, pos[i], others, wait_spots[i])
                actions[agent] = a
            obs, rewards, terms, truncs, infos = env.step(actions)
            ret += np.array([rewards[a] for a in env.possible_agents])
            if not env.agents:
                st = infos[env.possible_agents[0]]["episode_stats"]
                density.append(st["waste_density"])
                prob.append(st["apple_prob"])
        totals.append(ret.sum())
        per_agent.append(ret)
    totals = np.array(totals)
    per_agent = np.array(per_agent)
    return {
        "policy": policy,
        "collective": (totals.mean(), totals.std()),
        "per_agent": per_agent.mean(axis=0),
        "waste_density": float(np.mean(density)),
        "apple_prob": float(np.mean(prob)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policies", nargs="+", default=["none", "random", "one_cleaner", "two_cleaners", "all_switch"])
    args = p.parse_args()
    print(f"{'policy':14s} {'collective return':>22s}  {'per agent':>20s}  {'waste dens.':>11s}  {'apple prob':>10s}")
    for pol in args.policies:
        r = run(pol, args.episodes, args.seed)
        m, s = r["collective"]
        pa = " ".join(f"{x:5.1f}" for x in r["per_agent"])
        print(f"{pol:14s} {m:12.1f} +- {s:5.1f}  {pa:>20s}  {r['waste_density']:11.3f}  {r['apple_prob']:10.3f}")


if __name__ == "__main__":
    main()
