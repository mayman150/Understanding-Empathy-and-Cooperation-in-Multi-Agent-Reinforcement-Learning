"""Summarise TensorBoard runs into one table (mean of the last K logged values per tag).

    python scripts/summarize_runs.py runs/ --tags charts/collective_return charts/cooperation_rate/player_0 --last 20
    python scripts/summarize_runs.py runs/ --csv results.csv

Groups runs by (env, method = formulation_signal, policy, params) and reports mean +/- std across seeds,
which is what the report's plots and tables need.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--tags", nargs="*", default=DEFAULT_TAGS)
    p.add_argument("--last", type=int, default=20, help="average over the last K logged points")
    p.add_argument("--csv", default=None)
    a = p.parse_args()

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for run_dir in sorted(glob.glob(os.path.join(a.root, "*"))):
        name = os.path.basename(run_dir)
        m = RUN_RE.match(name)
        if not m or not glob.glob(os.path.join(run_dir, "events.out.tfevents.*")):
            continue
        metrics = load_run(run_dir, a.tags, a.last)
        if metrics:
            key = (m["env"], m["method"], m["policy"], m["params"])
            metrics["seed"] = int(m["seed"])
            groups[key].append(metrics)

    rows = []
    for key, runs in sorted(groups.items()):
        row = {"env": key[0], "method": key[1], "policy": key[2], "params": key[3], "seeds": len(runs)}
        for tag in a.tags:
            vals = [r[tag] for r in runs if tag in r]
            if vals:
                row[tag] = f"{np.mean(vals):.3f} +/- {np.std(vals):.3f}"
        rows.append(row)

    if not rows:
        print("no runs found")
        return
    cols = ["env", "method", "policy", "params", "seeds"] + [t for t in a.tags if any(t in r for r in rows)]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    if a.csv:
        import csv

        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()
