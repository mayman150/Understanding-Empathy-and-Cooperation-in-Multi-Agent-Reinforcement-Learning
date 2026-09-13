"""Rank configurations per method across many result folders and print the stage-B command for each winner.

    python scripts/best_configs.py ~/scratch/MARL/empathy_runs/coin_ppo_*                 # PPO sensitivity folders
    python scripts/best_configs.py ~/scratch/MARL/empathy_runs/coin_value ~/scratch/MARL/empathy_runs/coin_reward --top 3
    python scripts/best_configs.py ~/scratch/MARL/empathy_runs/coin_ppo_* --metric charts/cooperation_rate/player_0 --min-coop 0.6

Unlike summarize_runs.py this does not rely on run or folder names: every run's ``args.json`` is read, and runs
are grouped by *all* training arguments except the seed (formulation, signal, alpha, beta, phi, learning rate,
entropy coefficient, ...).  For every method (formulation + signal) the groups are ranked by the final value of
``--metric`` (mean of the last K logged points per run, then over seeds; by default the lower bound mean - std,
so that a configuration has to be good on every seed, ``--rank mean`` for the plain mean) and the top rows are
printed with the arguments that differ between them, followed by the ``coin.sh final`` line that reruns the
winner on fresh seeds with exactly its PPO settings.

Selection protocol: every method gets the same grid (same alpha neighbourhood size, same PPO settings, same
seeds and step budget), the metric and the ranking rule are fixed before looking at the results, and the
selected configurations are re-run on *fresh* seeds; the numbers printed here are selection statistics (biased
upwards by the selection itself) and are not the ones to report.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_TAGS = ["charts/collective_return", "charts/cooperation_rate/player_0", "charts/cooperation_rate/player_1"]
# arguments that do not define a configuration
IGNORED = {
    "seed", "exp_name", "torch_deterministic", "cuda", "track", "wandb_project_name", "wandb_entity", "run_dir",
    "save_model", "checkpoint_every", "batch_size", "minibatch_size", "num_iterations", "num_cpus", "cross_value_chunk",
}
TRAIN_ARG_KEYS = ["num_envs", "num_steps", "num_minibatches", "learning_rate", "ent_coef"]  # -> TRAIN_ARGS of coin.sh


def final_window(scalars, last: int, last_steps: int | None) -> list[float]:
    """The last ``last_steps`` environment steps (at least one point) if given, else the last ``last`` logged points."""
    if not scalars:
        return []
    if last_steps:
        end = scalars[-1].step
        window = [s.value for s in scalars if s.step > end - last_steps]
        return window or [scalars[-1].value]
    return [s.value for s in scalars[-last:]]


def load_run(run_dir: str, tags: list[str], last: int, last_steps: int | None = None) -> dict:
    with open(os.path.join(run_dir, "args.json")) as f:
        args = json.load(f)
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    available = set(acc.Tags().get("scalars", []))
    metrics = {}
    for tag in tags:
        if tag in available:
            values = final_window(acc.Scalars(tag), last, last_steps)
            if values:
                metrics[tag] = float(np.mean(values))
    steps = [s.step for s in acc.Scalars(tags[0])] if tags[0] in available else [0]
    return {"args": args, "metrics": metrics, "last_step": max(steps)}


def config_key(args: dict) -> tuple:
    return tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in args.items() if k not in IGNORED))


def method_of(args: dict) -> str:
    if args["formulation"] == "none":
        return "none"
    method = f"{args['formulation']}_{args['signal']}"
    if args["signal"] == "imagined":
        method += f"_{args.get('imagined_critic', 'other')}"
        if args.get("imagined_level", "trace") != "trace":
            method += "_lv"
    if args.get("social_scale") and args["signal"] in ("value", "imagined"):
        method += "_sc"
    # SVO with phi = 0 has no other-regarding part (alpha * V_i(o_i) for the value signal, a rescaled own advantage
    # for imagined/other) -> the control, ranked separately
    if args["formulation"] == "svo" and all(float(v) == 0.0 for v in str(args["phi"]).split(",")):
        method += "_phi0_control"
    return method


def cli_flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, list):
        return ",".join(fmt(x) for x in v)
    return str(v)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="result folders (each containing run folders with args.json + events)")
    p.add_argument("--metric", default="charts/collective_return", help="ranking metric (final value)")
    p.add_argument("--tags", nargs="*", default=DEFAULT_TAGS, help="metrics to show (the ranking metric is added)")
    p.add_argument("--last", type=int, default=20, help="average over the last K logged points")
    p.add_argument("--last-steps", type=int, default=None,
                   help="instead of --last: average over the points of the last N environment steps (e.g. 20000)")
    p.add_argument("--top", type=int, default=5, help="rows per method")
    p.add_argument("--rank", choices=["mean", "lcb"], default="lcb",
                   help="rank by the seed mean, or by mean - std (lcb: prefers configurations that are good on every seed)")
    p.add_argument("--min-coop", type=float, default=None, help="only rank groups whose mean cooperation rate (player_0) exceeds this")
    p.add_argument("--final-script", default=None, help="experiment script for the printed stage-B lines (default: from env_id)")
    p.add_argument("--final-time", default="01:00:00", help="SLURM time limit in the printed stage-B lines")
    p.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    a = p.parse_args()
    tags = list(dict.fromkeys([a.metric, *a.tags]))

    run_dirs = [
        d for root in a.roots for d in sorted(glob.glob(os.path.join(os.path.expanduser(root), "*")))
        if os.path.isfile(os.path.join(d, "args.json")) and glob.glob(os.path.join(d, "events.out.tfevents.*"))
    ]
    if not run_dirs:
        raise SystemExit("no runs found")
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        loaded = list(pool.map(load_run, run_dirs, [tags] * len(run_dirs), [a.last] * len(run_dirs), [a.last_steps] * len(run_dirs)))

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for run in loaded:
        if a.metric in run["metrics"]:
            groups[config_key(run["args"])].append(run)

    # per method: rank groups, work out which arguments vary within the method
    by_method: dict[str, list[dict]] = defaultdict(list)
    for key, runs in groups.items():
        args = runs[0]["args"]
        row = {
            "args": args, "n": len(runs), "steps": max(r["last_step"] for r in runs),
            "seeds": sorted(r["args"]["seed"] for r in runs),
        }
        for tag in tags:
            vals = [r["metrics"][tag] for r in runs if tag in r["metrics"]]
            row[tag] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))
        by_method[method_of(args)].append(row)

    winners = []
    for method in sorted(by_method):
        rows = by_method[method]
        if a.min_coop is not None:
            rows = [r for r in rows if r.get("charts/cooperation_rate/player_0", (float("nan"),))[0] >= a.min_coop] or rows
        def score(r: dict) -> float:
            mean, std = r[a.metric]
            return mean - std if a.rank == "lcb" else mean

        rows.sort(key=lambda r: -score(r))
        varying = sorted(
            k for k in rows[0]["args"] if k not in IGNORED and len({json.dumps(r["args"][k]) for r in rows}) > 1
        )
        window = f"last {a.last_steps} env steps" if a.last_steps else f"last {a.last} logged points"
        rule = ("mean - std over seeds" if a.rank == "lcb" else "mean over seeds") + f", {window}"
        print(f"\n=== {method}: {len(rows)} configurations, ranked by final {a.metric} ({rule})  (varying: {', '.join(varying) or 'nothing'})")
        short = {t: "/".join(t.split("/")[1:]) for t in tags}  # charts/cooperation_rate/player_0 -> cooperation_rate/player_0
        header = [f"{k:>12s}" for k in varying] + [f"{'seeds':>5s}", f"{'steps':>7s}", f"{'score':>7s}"] + [f"{short[t]:>28s}" for t in tags]
        print(" ".join(header))
        for r in rows[: a.top]:
            cells = [f"{fmt(r['args'][k]):>12s}" for k in varying] + [f"{r['n']:>5d}", f"{r['steps']/1e6:6.2f}M", f"{score(r):7.2f}"]
            cells += [f"{r[t][0]:14.3f} +/- {r[t][1]:<9.3f}" for t in tags]
            print(" ".join(cells))
        winners.append((method, rows[0]))

    print("\n=== stage B: rerun every winner on fresh seeds with its own PPO settings ===")
    for method, r in winners:
        args = r["args"]
        train_args = " ".join(f"{cli_flag(k)} {fmt(args[k])}" for k in TRAIN_ARG_KEYS)
        conf = f"--formulation {args['formulation']}"
        if args["formulation"] != "none":
            conf += f" --signal {args['signal']} --alpha {args['alpha']}"
            if args["formulation"] == "ia":
                conf += f" --beta {args['beta']}"
            if args["formulation"] == "svo":
                conf += f" --phi {args['phi']}"
            if args["signal"] == "imagined":
                conf += f" --imagined-critic {args.get('imagined_critic', 'other')} --imagined-warmup {args.get('imagined_warmup', 20)}"
                if args.get("reward_model_replay"):
                    conf += f" --reward-model-replay {args['reward_model_replay']}"
                if args.get("imagined_lambda") is not None:
                    conf += f" --imagined-lambda {fmt(args['imagined_lambda'])}"
                if args.get("imagined_level", "trace") != "trace":
                    conf += f" --imagined-level {args['imagined_level']}"
            if args.get("social_scale"):
                conf += " --social-scale"
            if args.get("social_aggregate", "mean") != "mean":
                conf += f" --social-aggregate {args['social_aggregate']}"
        env = str(args.get("env_id", "coin")).lower()
        script = a.final_script or ("cleanup" if env in ("cleanup", "clean_up") else "coin")
        print(f"# {method}: {a.metric} = {r[a.metric][0]:.2f} +/- {r[a.metric][1]:.2f} (n={r['n']})")
        print(f'TAG={script}_best FINAL_SEEDS=7-14 STEPS={args["total_timesteps"]} TIME={a.final_time} TRAIN_ARGS="{train_args}" \\\n'
              f"    scripts/experiments/{script}.sh final {conf}")


if __name__ == "__main__":
    main()
