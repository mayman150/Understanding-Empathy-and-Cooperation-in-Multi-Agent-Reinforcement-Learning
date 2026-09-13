"""Final figures: one configuration per method, mean over seeds with a 95% confidence interval.

    python scripts/plot_final.py ~/scratch/MARL/empathy_runs/coin_best_final_* -o figures_coin
    python scripts/plot_final.py ~/scratch/MARL/empathy_runs/coin_best_final_* -o figures_coin --bins 50 --ci 0.9 \
        --metrics charts/collective_return charts/cooperation_rate charts/equality

Each result folder holds the runs (seeds) of ONE configuration; the method is read from every run's
``args.json`` (formulation + signal; SVO with phi = 0 is the own-value control).  Curves are binned by
environment step, averaged over seeds, and drawn with a t-based confidence interval of the seed mean
(``mean +/- t_{(1+ci)/2, n-1} * std / sqrt(n)``); individual seeds are never drawn.

Per metric the script writes
    <metric>__<formulation>.png   value vs. reward signal of one formulation (EI, SIA, SVO, IA) + references
    <metric>__value.png           every formulation with the value signal (+ references)
    <metric>__reward.png          every formulation with the reward signal (+ references)
    <metric>__all.png             everything: colour = formulation, solid = value signal, dashed = reward signal
and two tables: ``final_curves.csv`` (method, metric, step, mean, ci_low, ci_high, n) and ``final_table.csv``
(final value of every metric per method: mean over the last 10% of training, +/- half-width of the CI, n).

A metric without an agent suffix (e.g. ``charts/cooperation_rate``) that is not logged itself is built from
the per-agent tags ``charts/cooperation_rate/player_*`` by averaging the agents within each seed.
References (``--reference``, default: plain PPO and the own-value control) are drawn in grey.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_curves import RUN_RE, bin_curve  # noqa: E402

FORMULATION_NAMES = {"ei": "EI", "sia": "SIA", "svo": "SVO", "ia": "IA"}
FORMULATION_COLOURS = {"ei": "#1f77b4", "sia": "#ff7f0e", "svo": "#2ca02c", "ia": "#d62728"}
SIGNAL_STYLES = {"value": "-", "reward": "--", "imagined_other": "-.", "imagined_shaped": ":"}
REFERENCE_STYLES = {"none": ("#555555", ":"), "control": ("#999999", "-.")}
DEFAULT_METRICS = ["charts/collective_return", "charts/cooperation_rate"]
# two-sided Student t critical values for 95% (index = degrees of freedom); beyond 30 the normal value is used
T95 = [float("nan"), 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
       2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048,
       2.045, 2.042]
T90 = [float("nan"), 6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782, 1.771,
       1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706, 1.703, 1.701,
       1.699, 1.697]


def t_critical(ci: float, n: int) -> float:
    """Two-sided Student t critical value for a `ci` interval of a mean over n samples."""
    if n < 2:
        return float("nan")
    df = n - 1
    if math.isclose(ci, 0.95):
        table, z = T95, 1.960
    elif math.isclose(ci, 0.90):
        table, z = T90, 1.645
    else:
        try:
            from scipy import stats

            return float(stats.t.ppf(0.5 + ci / 2, df))
        except ImportError as e:  # pragma: no cover
            raise SystemExit("--ci other than 0.95 / 0.9 needs scipy") from e
    return table[df] if df < len(table) else z


def method_of(args: dict) -> tuple[str, str, str]:
    """(key, label, kind) with kind in {'none', 'control', 'method'}."""
    f, s = args["formulation"], args["signal"]
    if f == "none":
        return "none", "plain PPO", "none"
    signal = s if s != "imagined" else f"imagined/{args.get('imagined_critic', 'other')}"
    suffix = " (scaled)" if args.get("social_scale") and s in ("value", "imagined") else ""
    if f == "svo" and all(float(v) == 0.0 for v in str(args["phi"]).split(",")):
        return f"control_{signal.replace('/', '_')}", f"own-value control (SVO phi=0, {signal} signal{suffix})", "control"
    return f"{f}_{signal.replace('/', '_')}", f"{FORMULATION_NAMES[f]}, {signal} signal{suffix}", "method"


def load_run(run_dir: str, metrics: list[str]) -> dict:
    """{'args': ..., 'curves': {metric: (steps, values)}} with agent-averaged metrics resolved."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    with open(os.path.join(run_dir, "args.json")) as f:
        args = json.load(f)
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    available = set(acc.Tags().get("scalars", []))

    def series(tag):
        ev = acc.Scalars(tag)
        return np.array([e.step for e in ev], dtype=np.float64), np.array([e.value for e in ev], dtype=np.float64)

    curves = {}
    for metric in metrics:
        if metric in available:
            curves[metric] = [series(metric)]
        else:  # e.g. charts/cooperation_rate -> mean over charts/cooperation_rate/player_*
            parts = sorted(t for t in available if t.startswith(metric + "/"))
            if parts:
                curves[metric] = [series(t) for t in parts]
    return {"args": args, "curves": curves}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="result folders, one configuration (several seeds) each")
    p.add_argument("-o", "--out", default="figures", help="output directory")
    p.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    p.add_argument("--bins", type=int, default=100, help="step bins per curve")
    p.add_argument("--ci", type=float, default=0.95, help="confidence level of the interval around the seed mean")
    p.add_argument("--reference", nargs="*", default=["none", "control"], choices=["none", "control"],
                   help="reference curves drawn in grey on every figure")
    p.add_argument("--final-frac", type=float, default=0.1, help="fraction of training averaged for final_table.csv")
    p.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---- load ---------------------------------------------------------------------------------
    run_dirs = [
        d for root in a.roots for d in sorted(glob.glob(os.path.join(os.path.expanduser(root), "*")))
        if RUN_RE.match(os.path.basename(d)) and os.path.isfile(os.path.join(d, "args.json"))
        and glob.glob(os.path.join(d, "events.out.tfevents.*"))
    ]
    if not run_dirs:
        raise SystemExit("no runs found")
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        loaded = list(pool.map(load_run, run_dirs, [a.metrics] * len(run_dirs)))

    methods: dict[str, dict] = {}  # key -> {"label", "kind", "formulation", "signal", "runs": [...], "configs": set()}
    for run in loaded:
        key, label, kind = method_of(run["args"])
        signal_key = key.split("_", 1)[1] if kind == "method" else key  # e.g. "value", "reward", "imagined_other"
        m = methods.setdefault(key, {"label": label, "kind": kind, "formulation": run["args"]["formulation"],
                                     "signal": run["args"]["signal"], "signal_key": signal_key, "runs": [], "configs": set()})
        m["runs"].append(run)
        m["configs"].add(json.dumps({k: v for k, v in run["args"].items() if k not in ("seed", "run_dir", "exp_name")}, sort_keys=True))
    for key, m in methods.items():
        if len(m["configs"]) > 1:
            print(f"warning: {key} mixes {len(m['configs'])} different configurations; pass one folder per method")
    max_step = max(s[-1] for run in loaded for parts in run["curves"].values() for s, _ in parts)
    edges = np.linspace(0.0, max_step, a.bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    print(f"loaded {len(loaded)} runs, {len(methods)} methods: " + ", ".join(f"{k} (n={len(m['runs'])})" for k, m in sorted(methods.items())))

    # ---- aggregate: mean + CI over seeds -------------------------------------------------------
    stats: dict[tuple, tuple[np.ndarray, np.ndarray, int]] = {}  # (key, metric) -> (mean, half-width, n)
    for key, m in methods.items():
        for metric in a.metrics:
            per_seed = []
            for run in m["runs"]:
                if metric not in run["curves"]:
                    continue
                agent_curves = [bin_curve(s, v, edges) for s, v in run["curves"][metric]]
                with np.errstate(invalid="ignore"):
                    per_seed.append(np.nanmean(np.vstack(agent_curves), axis=0))  # mean over agents within the seed
            if not per_seed:
                continue
            stack = np.vstack(per_seed)
            n = len(per_seed)
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(stack, axis=0)
                std = np.nanstd(stack, axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
            half = t_critical(a.ci, n) * std / math.sqrt(n) if n > 1 else np.zeros_like(mean)
            stats[(key, metric)] = (mean, half, n)

    with open(os.path.join(a.out, "final_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "label", "metric", "step", "mean", "ci_low", "ci_high", "n"])
        for (key, metric), (mean, half, n) in sorted(stats.items()):
            for step, mu, h in zip(centers, mean, half):
                if not np.isnan(mu):
                    w.writerow([key, methods[key]["label"], metric, int(step), f"{mu:.6g}", f"{mu - h:.6g}", f"{mu + h:.6g}", n])
    k_final = max(1, int(round(a.final_frac * a.bins)))
    with open(os.path.join(a.out, "final_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "label", "n"] + [x for metric in a.metrics for x in (metric, metric + "_ci")])
        print(f"\nfinal values (mean over the last {int(a.final_frac * 100)}% of training; +/- = half-width of the {int(a.ci * 100)}% CI over seeds)")
        for key in sorted(methods, key=lambda k: (methods[k]["kind"] != "method", k)):
            row, cells = [key, methods[key]["label"], len(methods[key]["runs"])], []
            for metric in a.metrics:
                if (key, metric) not in stats:
                    row += ["", ""]
                    cells.append(f"{metric.split('/')[-1]}: n/a")
                    continue
                mean, half, n = stats[(key, metric)]
                # final value: mean of the per-seed final means, CI over seeds of those
                finals = []
                for run in methods[key]["runs"]:
                    if metric in run["curves"]:
                        agent = [bin_curve(s, v, edges)[-k_final:] for s, v in run["curves"][metric]]
                        with np.errstate(invalid="ignore"):
                            finals.append(float(np.nanmean(np.vstack(agent))))
                fin = np.array(finals)
                h = t_critical(a.ci, len(fin)) * fin.std(ddof=1) / math.sqrt(len(fin)) if len(fin) > 1 else 0.0
                row += [f"{fin.mean():.4g}", f"{h:.3g}"]
                cells.append(f"{metric.split('/')[-1]}: {fin.mean():7.3f} +/- {h:.3f}")
            w.writerow(row)
            print(f"  {methods[key]['label']:34s} n={len(methods[key]['runs'])}  " + "   ".join(cells))
    print(f"wrote {a.out}/final_curves.csv and {a.out}/final_table.csv")

    # ---- figures ------------------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(ax, key, metric, colour, style, label=None):
        if (key, metric) not in stats:
            return False
        mean, half, n = stats[(key, metric)]
        ok = ~np.isnan(mean)
        ax.plot(centers[ok], mean[ok], color=colour, linestyle=style, linewidth=1.6, label=f"{label or methods[key]['label']} (n={n})")
        ax.fill_between(centers[ok], (mean - half)[ok], (mean + half)[ok], color=colour, alpha=0.15, linewidth=0)
        return True

    control_keys = sorted(k for k, m in methods.items() if m["kind"] == "control")
    control_greys = ["#999999", "#777777", "#bbbbbb", "#555555"]

    def draw_references(ax, metric):
        for ref in a.reference:
            if ref == "none" and "none" in methods:
                draw(ax, "none", metric, *REFERENCE_STYLES["none"])
            if ref == "control":
                for k, key in enumerate(control_keys):
                    draw(ax, key, metric, control_greys[k % len(control_greys)], REFERENCE_STYLES["control"][1])

    # line style per signal (the same formulation keeps its colour across signals)
    signals = sorted({m["signal_key"] for m in methods.values() if m["kind"] == "method"})
    style_cycle = ["-", "--", "-.", ":"]
    signal_styles = {s: SIGNAL_STYLES.get(s, style_cycle[k % len(style_cycle)]) for k, s in enumerate(signals)}

    def finish(fig, ax, metric, title, name):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("environment steps")
        ax.set_ylabel(metric.split("/")[-1].replace("_", " "))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        ax.text(0.99, 0.01, f"mean over seeds, shaded = {int(a.ci * 100)}% CI", transform=ax.transAxes,
                fontsize=7, alpha=0.7, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, name), dpi=150)
        plt.close(fig)

    formulations = [f for f in FORMULATION_NAMES if any(m["formulation"] == f and m["kind"] == "method" for m in methods.values())]
    n_fig = 0
    for metric in a.metrics:
        safe = metric.replace("/", "_")
        pretty = metric.split("/")[-1].replace("_", " ")
        # one formulation, all signals (value vs. reward vs. imagined)
        for f in formulations:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            drawn = 0
            for signal, style in signal_styles.items():
                drawn += draw(ax, f"{f}_{signal}", metric, FORMULATION_COLOURS[f], style)
            if drawn:
                draw_references(ax, metric)
                finish(fig, ax, metric, f"{pretty}: {FORMULATION_NAMES[f]} across signals", f"{safe}__{f}.png")
                n_fig += 1
            else:
                plt.close(fig)
        # one signal, all formulations
        for signal in signal_styles:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            drawn = sum(draw(ax, f"{f}_{signal}", metric, FORMULATION_COLOURS[f], "-") for f in formulations)
            if drawn:
                draw_references(ax, metric)
                finish(fig, ax, metric, f"{pretty}: all formulations, {signal.replace('_', '/')} signal", f"{safe}__{signal}.png")
                n_fig += 1
            else:
                plt.close(fig)
        # everything
        fig, ax = plt.subplots(figsize=(8, 4.8))
        drawn = 0
        for f in formulations:
            for signal, style in signal_styles.items():
                drawn += draw(ax, f"{f}_{signal}", metric, FORMULATION_COLOURS[f], style)
        if drawn:
            draw_references(ax, metric)
            legend = ", ".join(f"{sty} = {sig.replace('_', '/')}" for sig, sty in signal_styles.items())
            finish(fig, ax, metric, f"{pretty}: all formulations ({legend})", f"{safe}__all.png")
            n_fig += 1
        else:
            plt.close(fig)
    print(f"wrote {n_fig} figures to {a.out}/")


if __name__ == "__main__":
    main()
