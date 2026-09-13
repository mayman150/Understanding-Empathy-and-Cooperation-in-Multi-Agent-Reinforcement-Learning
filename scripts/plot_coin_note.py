"""Figures for the Coin Game section of docs/method_update.tex.

    python scripts/plot_coin_note.py --final ~/Desktop/MARL/final_curves.csv --curves ~/Desktop/MARL/curves -o docs/figures

Inputs
------
* ``final_curves.csv``: stage-1 finals (8 seeds, default PPO, 2M steps) as written by scripts/plot_final.py
  (columns method, label, metric, step, mean, ci_low, ci_high, n): plain PPO, the own-value control, EI value,
  EI reward.
* ``curves/<folder>/curves.csv``: stage-A imagined grids as written by scripts/plot_curves.py (columns env, method,
  params, runs, seed, tag, step, value; one row per seed and step bin).

Figure 1  methods over training at the *default* PPO setting (the stage-1 finals plus the stage-A imagined runs
          from the lr 2.5e-4 / ent 0.01 folders): collective return and cooperation rate, mean with a 95% CI.
Figure 2  alpha sweeps at lr 1e-3 / ent 0.01: EI imagined/other (flat plateau from alpha = 1) next to EI
          imagined/shaped (narrow band, collapse for alpha >= 1); collective return.
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

import numpy as np

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


def load_final(path: str) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[(r["method"], r["metric"])].append((int(r["step"]), float(r["mean"]), float(r["ci_low"]), float(r["ci_high"]), int(r["n"])))
    out = {}
    for key, vals in rows.items():
        vals.sort()
        a = np.array(vals, dtype=float)
        out[key] = (a[:, 0], a[:, 1], a[:, 2], a[:, 3], int(a[0, 4]))
    return out


def load_curves(folder: str, method: str, params: str, tag: str):
    """seed curves -> (steps, mean, ci_low, ci_high, n) with a 95% t-interval over seeds"""
    per_seed: dict[int, list[tuple[int, float]]] = defaultdict(list)
    with open(os.path.join(folder, "curves.csv")) as f:
        for r in csv.DictReader(f):
            if r["method"] == method and r["params"] == params and r["tag"] == tag:
                per_seed[int(r["seed"])].append((int(r["step"]), float(r["value"])))
    if not per_seed:
        raise SystemExit(f"no rows for {method} {params} {tag} in {folder}")
    steps = sorted({s for v in per_seed.values() for s, _ in v})
    grid = np.full((len(per_seed), len(steps)), np.nan)
    index = {s: k for k, s in enumerate(steps)}
    for i, v in enumerate(per_seed.values()):
        for s, val in v:
            grid[i, index[s]] = val
    n = len(per_seed)
    mean = np.nanmean(grid, axis=0)
    sd = np.nanstd(grid, axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    half = T95.get(n - 1, 1.96) * sd / np.sqrt(n)
    return np.array(steps, dtype=float), mean, mean - half, mean + half, n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--final", required=True, help="final_curves.csv of the stage-1 finals")
    p.add_argument("--curves", required=True, help="folder with the stage-A curve folders (imagined_other_lr..., ...)")
    p.add_argument("-o", "--out", default="docs/figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8})
    final = load_final(os.path.expanduser(a.final))
    C = os.path.expanduser(a.curves)
    default_other = os.path.join(C, "imagined_other_lr2.5e-4_ent0.01")
    default_shaped = os.path.join(C, "imagined_shaped_lr2.5e-4_ent0.01")
    fast_other = os.path.join(C, "imagined_other_lr1e-3_ent0.01")
    fast_shaped = os.path.join(C, "imagined_shaped_lr1e-3_ent0.01")

    # ---- figure 1: methods over training, default PPO -------------------------------------------
    series = [  # label, colour, style, source
        ("plain PPO", "#555555", ":", ("final", "none")),
        ("own-value control ($\\alpha V_i(o_i')$)", "#999999", "-.", ("final", "control")),
        ("EI, value only (report's term)", "#1f77b4", "-", ("final", "ei_value")),
        ("EI, imagined reward, shaped ($\\alpha=0.1$)", "#2ca02c", ":", ("curves", default_shaped, "ei_imagined_shaped", "a0.1_b0.0_p0.0")),
        ("EI, imagined reward, other ($\\alpha=1$)", "#d62728", "-.", ("curves", default_other, "ei_imagined_other_sc", "a1_b0.0_p0.0")),
        ("EI, true rewards (reference)", "#1f77b4", "--", ("final", "ei_reward")),
    ]
    metrics = [("charts/collective_return", "collective return per episode"), ("charts/cooperation_rate", "cooperation rate")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for ax, (metric, ylabel) in zip(axes, metrics):
        for label, colour, style, src in series:
            if src[0] == "final":
                steps, mean, lo, hi, n = final[(src[1], metric)]
            else:
                tag = metric if metric != "charts/cooperation_rate" else "charts/cooperation_rate/player_0"
                steps, mean, lo, hi, n = load_curves(src[1], src[2], src[3], tag)
            ax.plot(steps, mean, color=colour, linestyle=style, linewidth=1.5, label=f"{label} (n={n})")
            ax.fill_between(steps, lo, hi, color=colour, alpha=0.15, linewidth=0)
        ax.set_xlabel("environment steps")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 2.0e6)
    axes[0].axhline(35, color="black", linewidth=0.6, linestyle=(0, (2, 4)))
    axes[0].text(1.95e6, 35.6, "ceiling (~35 coins)", ha="right", fontsize=7)
    axes[1].axhline(0.5, color="black", linewidth=0.6, linestyle=(0, (2, 4)))
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "coin_methods.pdf"))
    fig.savefig(os.path.join(a.out, "coin_methods.png"), dpi=160)
    plt.close(fig)

    # ---- figure 2: alpha sweeps at lr 1e-3 --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    sweeps = [
        (axes[0], fast_other, "ei_imagined_other_sc", ["0.3", "0.5", "1", "2", "3"], "EI, imagined reward, other (advantage path)"),
        (axes[1], fast_shaped, "ei_imagined_shaped", ["0.01", "0.03", "0.1", "0.3", "1", "3"], "EI, imagined reward, shaped (reward path)"),
    ]
    for ax, folder, method, alphas, title in sweeps:
        cmap = plt.get_cmap("viridis", len(alphas))
        for k, alpha in enumerate(alphas):
            steps, mean, lo, hi, n = load_curves(folder, method, f"a{alpha}_b0.0_p0.0", "charts/collective_return")
            ax.plot(steps, mean, color=cmap(k), linewidth=1.5, label=f"$\\alpha={alpha}$")
            ax.fill_between(steps, lo, hi, color=cmap(k), alpha=0.15, linewidth=0)
        ax.axhline(35, color="black", linewidth=0.6, linestyle=(0, (2, 4)))
        ax.set_title(title)
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 2.0e6)
        ax.legend(frameon=False, ncol=2)
    axes[0].set_ylabel("collective return per episode")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "coin_alpha.pdf"))
    fig.savefig(os.path.join(a.out, "coin_alpha.png"), dpi=160)
    plt.close(fig)
    print(f"wrote {a.out}/coin_methods.{{pdf,png}} and coin_alpha.{{pdf,png}}")


if __name__ == "__main__":
    main()
