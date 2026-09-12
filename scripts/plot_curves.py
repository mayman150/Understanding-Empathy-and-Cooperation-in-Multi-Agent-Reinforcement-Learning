"""Learning curves from TensorBoard logs: mean +/- std over seeds, one line per configuration.

    python scripts/plot_curves.py ~/scratch/MARL/empathy_runs/coin_value ~/scratch/MARL/empathy_runs/coin_reward -o curves_coin
    python scripts/plot_curves.py runs/pd_n2_value --tags charts/cooperation_rate/player_0 social/term_abs_mean/player_0 --bins 100
    # PPO sensitivity of one configuration: lines = PPO settings (folder-name suffix lr.._ent..)
    python scripts/plot_curves.py ~/scratch/MARL/empathy_runs/coin_ppo_formulation_svo_signal_value_alpha_30_phi_0.523599_lr*_ent* \
        -o curves_ppo_svo30 --tags charts/collective_return charts/cooperation_rate/player_0

Output directory:
    curves.csv                    long format: env, method, params, runs, seed, tag, step, value   (binned by step)
    <tag>__<method>.png           one figure per (tag, method); lines = configurations (alpha, beta, phi) x result
                                  folder, shaded band = +/- 1 std over seeds
    <tag>__all_methods.png        best configuration of every method on one figure (by final mean of the first tag)

Runs are grouped like scripts/summarize_runs.py (folder name <env>__<method>__<ff|lstm>__<params>__s<seed>__<time>).
The PPO setting is not part of a run name, so when several result folders are given, runs are additionally
keyed by the part of the folder name that differs between them (the `runs` column / the [label] in legends).
Curves are averaged inside `--bins` equal-width step bins so that per-episode tags become smooth and the CSV
stays small.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

RUN_RE = re.compile(r"^(?P<env>.+?)__(?P<method>[a-z]+(?:_[a-z]+)*)__(?P<policy>ff|lstm)__(?P<params>.+?)__s(?P<seed>\d+)__\d+$")
DEFAULT_TAGS = [
    "charts/cooperation_rate/player_0",
    "charts/cooperation_rate/player_1",
    "charts/collective_return",
    "charts/equality",
    "charts/episodic_return/player_0",
    "charts/episodic_return/player_1",
    "social/term_abs_mean/player_0",
    "social/advantage_abs_mean/player_0",
    "losses/entropy/player_0",
]


def load_scalars(run_dir: str, tags: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    available = set(acc.Tags().get("scalars", []))
    out = {}
    for tag in tags:
        if tag in available:
            events = acc.Scalars(tag)
            out[tag] = (np.array([e.step for e in events], dtype=np.float64), np.array([e.value for e in events], dtype=np.float64))
    return out


def bin_curve(steps: np.ndarray, values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean value per step bin (nan where a bin has no points)."""
    idx = np.clip(np.searchsorted(edges, steps, side="right") - 1, 0, len(edges) - 2)
    sums = np.bincount(idx, weights=values, minlength=len(edges) - 1)
    counts = np.bincount(idx, minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def root_labels(roots: list[str]) -> dict[str, str]:
    """Short label per result folder: the folder name minus the prefix shared by all of them (e.g. 'lr1e-3_ent0.05')."""
    names = {r: os.path.basename(os.path.normpath(r)) for r in roots}
    prefix = os.path.commonprefix(list(names.values()))
    prefix = prefix[: prefix.rfind("_") + 1] if "_" in prefix else ""  # cut at a separator, not mid-token
    return {r: (n[len(prefix):] or n) for r, n in names.items()}


def pretty_params(params: str) -> str:
    # a20_b0.0_p0.0 -> alpha=20 ; a20_b10_p0.0 -> alpha=20, beta=10 ; a20_b0.0_p0.785 -> alpha=20, phi=0.785
    m = re.match(r"a(?P<a>[^_]+)_b(?P<b>[^_]+)_p(?P<p>.+)$", params)
    if not m:
        return params
    parts = [f"alpha={m['a']}"]
    if m["b"] not in ("0", "0.0"):
        parts.append(f"beta={m['b']}")
    if m["p"] not in ("0", "0.0"):
        parts.append(f"phi={m['p']}")
    return ", ".join(parts)


def sort_key(params: str):
    m = re.match(r"a(?P<a>[^_,]+)", params)
    try:
        return (float(m["a"]) if m else 0.0, params)
    except ValueError:
        return (float("inf"), params)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="result folders (each containing run folders)")
    p.add_argument("-o", "--out", default="curves", help="output directory")
    p.add_argument("--tags", nargs="*", default=DEFAULT_TAGS)
    p.add_argument("--bins", type=int, default=200, help="number of step bins per curve")
    p.add_argument("--methods", nargs="*", default=None, help="only these methods (e.g. sia_value none)")
    p.add_argument("--no-plots", action="store_true", help="only write curves.csv")
    p.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="parallel workers for reading event files")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- load -------------------------------------------------------------------------------
    # The PPO setting of a run is not in its name (only env/method/params/seed are), so runs of the same
    # configuration from different result folders (e.g. coin_ppo_<conf>_lr1e-3_ent0.05) are told apart by a
    # folder label: the part of the folder name that differs between the given roots.
    roots = [os.path.expanduser(r) for r in a.roots]
    labels = root_labels(roots) if len(roots) > 1 else {roots[0]: ""}
    run_dirs, run_labels = [], []
    for root in roots:
        for run_dir in sorted(glob.glob(os.path.join(root, "*"))):
            m = RUN_RE.match(os.path.basename(run_dir))
            if not m or not glob.glob(os.path.join(run_dir, "events.out.tfevents.*")):
                continue
            if a.methods and m["method"] not in a.methods:
                continue
            run_dirs.append(run_dir)
            run_labels.append(labels[root])
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        loaded = list(pool.map(load_scalars, run_dirs, [a.tags] * len(run_dirs)))
    runs: dict[tuple, dict[int, dict]] = defaultdict(dict)  # (env, method, params, label) -> seed -> {tag: (steps, values)}
    max_step = 0.0
    for run_dir, label, data in zip(run_dirs, run_labels, loaded):
        if not data:
            continue
        m = RUN_RE.match(os.path.basename(run_dir))
        runs[(m["env"], m["method"], m["params"], label)][int(m["seed"])] = data
        max_step = max(max_step, max(s[-1] for s, _ in data.values()))
    if not runs:
        raise SystemExit("no runs found")
    edges = np.linspace(0.0, max_step, a.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    print(f"loaded {sum(len(v) for v in runs.values())} runs in {len(runs)} configurations; step range 0..{int(max_step)}")

    # ---- bin + aggregate ------------------------------------------------------------------------
    agg: dict[tuple, dict[str, tuple[np.ndarray, np.ndarray, int]]] = {}  # config -> tag -> (mean, std, n)
    with open(os.path.join(a.out, "curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "method", "params", "runs", "seed", "tag", "step", "value"])
        for key, seeds in sorted(runs.items()):
            agg[key] = {}
            for tag in a.tags:
                per_seed = []
                for seed, data in sorted(seeds.items()):
                    if tag not in data:
                        continue
                    curve = bin_curve(*data[tag], edges)
                    per_seed.append(curve)
                    for step, val in zip(centers, curve):
                        if not np.isnan(val):
                            w.writerow([key[0], key[1], key[2], key[3], seed, tag, int(step), f"{val:.6g}"])
                if per_seed:
                    stack = np.vstack(per_seed)
                    with np.errstate(invalid="ignore"):
                        agg[key][tag] = (np.nanmean(stack, axis=0), np.nanstd(stack, axis=0), len(per_seed))
    print(f"wrote {os.path.join(a.out, 'curves.csv')}")
    if a.no_plots:
        return

    # ---- plots --------------------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(ax, mean, std, label):
        ok = ~np.isnan(mean)
        ax.plot(centers[ok], mean[ok], label=label, linewidth=1.4)
        ax.fill_between(centers[ok], (mean - std)[ok], (mean + std)[ok], alpha=0.2)

    def legend_name(key: tuple) -> str:
        return pretty_params(key[2]) + (f" [{key[3]}]" if key[3] else "")

    methods = sorted({k[1] for k in agg})
    for tag in a.tags:
        safe_tag = tag.replace("/", "_")
        # one figure per method: lines = configurations (x folder label, e.g. PPO setting)
        for method in methods:
            configs = sorted([k for k in agg if k[1] == method and tag in agg[k]], key=lambda k: (sort_key(k[2]), k[3]))
            if not configs:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.5))
            cmap = plt.get_cmap("viridis", max(len(configs), 2))
            for i, key in enumerate(configs):
                mean, std, n = agg[key][tag]
                ax.plot([], [])  # keep colour order stable
                ok = ~np.isnan(mean)
                ax.plot(centers[ok], mean[ok], color=cmap(i), linewidth=1.4, label=f"{legend_name(key)} (n={n})")
                ax.fill_between(centers[ok], (mean - std)[ok], (mean + std)[ok], color=cmap(i), alpha=0.18)
            ax.set_title(f"{tag}   |   {method}", fontsize=10)
            ax.set_xlabel("environment steps")
            ax.set_ylabel(tag.split("/")[-2] if tag.count("/") > 1 else tag.split("/")[-1])
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, ncol=2 if len(configs) > 6 else 1)
            fig.tight_layout()
            fig.savefig(os.path.join(a.out, f"{safe_tag}__{method}.png"), dpi=130)
            plt.close(fig)

        # overview: best configuration per method (by final mean of the FIRST requested tag)
        ref_tag = a.tags[0]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for method in methods:
            configs = [k for k in agg if k[1] == method and tag in agg[k] and ref_tag in agg[k]]
            if not configs:
                continue
            best = max(configs, key=lambda k: np.nanmean(agg[k][ref_tag][0][-max(1, a.bins // 10):]))
            mean, std, n = agg[best][tag]
            draw(ax, mean, std, f"{method}: {legend_name(best)} (n={n})")
        ax.set_title(f"{tag}   |   best configuration per method", fontsize=10)
        ax.text(0.01, 0.01, f"selection: final {ref_tag}", transform=ax.transAxes, fontsize=7, alpha=0.7)
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"{safe_tag}__all_methods.png"), dpi=130)
        plt.close(fig)
    print(f"wrote plots to {a.out}/  ({len(a.tags)} tags x {len(methods)} methods + overviews)")


if __name__ == "__main__":
    main()
