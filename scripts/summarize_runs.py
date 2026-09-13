"""Summarise TensorBoard runs into one table (mean of the last K logged values per tag).

    python scripts/summarize_runs.py runs/ --tags charts/collective_return charts/cooperation_rate/player_0 --last 20
    python scripts/summarize_runs.py runs/ --csv results.csv
    python scripts/summarize_runs.py results/coin_ppo_*_lr*_ent*        # several folders -> extra 'runs' column

Groups runs by (env, method = formulation_signal, policy, params) and reports mean +/- std across seeds,
which is what the report's plots and tables need.  With several result folders (e.g. the same configuration
under different PPO settings, one folder each) the rows are additionally labelled by folder name.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_TAGS = [
    "charts/collective_return",
    "charts/equality",
    "charts/sustainability",
    "charts/cooperation_rate/player_0",
    "charts/cooperation_rate/player_1",
]
RUN_RE = re.compile(r"^(?P<env>.+?)__(?P<method>[a-z]+(?:_[a-z]+)*)__(?P<policy>ff|lstm)__(?P<params>.+?)__s(?P<seed>\d+)__\d+$")


def load_run(run_dir: str, tags: list[str], last: int) -> dict[str, float]:
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    available = set(acc.Tags().get("scalars", []))
    for tag in tags:
        if tag in available:
            values = [s.value for s in acc.Scalars(tag)]
            if values:
                out[tag] = float(np.mean(values[-last:]))
    return out


def root_labels(roots: list[str]) -> dict[str, str]:
    """Short label per result folder: the folder name minus the prefix shared by all of them (e.g. 'lr1e-3_ent0.05')."""
    names = {r: os.path.basename(os.path.normpath(r)) for r in roots}
    prefix = os.path.commonprefix(list(names.values())) if len(roots) > 1 else ""
    prefix = prefix[: prefix.rfind("_") + 1] if "_" in prefix else ""  # cut at a separator, not mid-token
    return {r: (n[len(prefix):] or n) for r, n in names.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="+", metavar="root", help="result folder(s); with several, a 'runs' column tells them apart")
    p.add_argument("--tags", nargs="*", default=DEFAULT_TAGS)
    p.add_argument("--last", type=int, default=20, help="average over the last K logged points")
    p.add_argument("--csv", default=None)
    p.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1), help="parallel workers for reading event files")
    a = p.parse_args()

    labels = root_labels(a.roots)
    run_dirs, run_roots = [], []
    for root in a.roots:
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            if RUN_RE.match(os.path.basename(d)) and glob.glob(os.path.join(d, "events.out.tfevents.*")):
                run_dirs.append(d)
                run_roots.append(root)
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        loaded = list(pool.map(load_run, run_dirs, [a.tags] * len(run_dirs), [a.last] * len(run_dirs)))
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for run_dir, root, metrics in zip(run_dirs, run_roots, loaded):
        m = RUN_RE.match(os.path.basename(run_dir))
        if metrics:
            key = (labels[root], m["env"], m["method"], m["policy"], m["params"])
            metrics["seed"] = int(m["seed"])
            groups[key].append(metrics)

    rows = []
    for key, runs in sorted(groups.items()):
        row = {"runs": key[0], "env": key[1], "method": key[2], "policy": key[3], "params": key[4], "seeds": len(runs)}
        for tag in a.tags:
            vals = [r[tag] for r in runs if tag in r]
            if vals:
                row[tag] = f"{np.mean(vals):.3f} +/- {np.std(vals):.3f}"
        rows.append(row)

    if not rows:
        print("no runs found")
        return
    cols = (["runs"] if len(a.roots) > 1 else []) + ["env", "method", "policy", "params", "seeds"]
    cols += [t for t in a.tags if any(t in r for r in rows)]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    if a.csv:
        import csv

        with open(a.csv, "w", newline="") as f:
            # rows always carry the folder label ('runs'); the column is only shown for several folders
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
