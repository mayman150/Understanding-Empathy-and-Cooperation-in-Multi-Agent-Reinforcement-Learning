"""Generate a hyper-parameter grid: one ``train.py`` argument line per job (for SLURM job arrays).

The default grid is the parameter table of the report ("Parameter Configurations"):

    EI   alpha in A
    SIA  alpha in A
    SVO  alpha in A,  phi  in {pi/2, pi/3, pi/4, pi/6}
    IA   alpha in A,  beta in {alpha/2, alpha/3, alpha/10}

with A = {0, 0.003, 0.01, 0.03, 0.1, 0.3}.  ``alpha = 0`` is the plain-PPO baseline for every
formulation, so it is emitted once per seed as ``--formulation none`` instead of once per
formulation.  Each configuration is crossed with ``--signals`` and ``--seeds``.

Examples
--------
    # the report's grid on the 2-player PD, value and reward signals, 5 seeds
    python scripts/make_grid.py --env-id pd --seeds 1 2 3 4 5 -o grids/pd_report.txt

    # extended alpha range (the value signal needs alphas comparable to the advantage scale)
    python scripts/make_grid.py --env-id pd --alphas 0 0.01 0.1 1 3 10 30 100 --seeds 1 2 3 4 5 -o grids/pd_wide.txt

    # mixed populations: player_0 selfish, player_1 with the grid alpha
    python scripts/make_grid.py --env-id pd --mixed "0,{a}" -o grids/pd_mixed.txt

    # print the grid as a LaTeX table (same layout as the report)
    python scripts/make_grid.py --alphas 0 0.1 1 10 100 --latex

Then submit with ``scripts/slurm/submit.sh grids/pd_report.txt``.
"""
from __future__ import annotations

import argparse
import math
import os
from fractions import Fraction

PHIS = {"pi/2": math.pi / 2, "pi/3": math.pi / 3, "pi/4": math.pi / 4, "pi/6": math.pi / 6}  # the report's SVO angles
ALL_PHIS = {**PHIS, "0": 0.0}  # "0": SVO with no other-regarding term = the own-value control
BETA_RATIOS = [Fraction(1, 2), Fraction(1, 3), Fraction(1, 10)]
REPORT_ALPHAS = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3]


def fmt(x: float) -> str:
    """Compact float formatting: 0.1 -> '0.1', 1.0 -> '1', 1.5708 -> '1.5708'."""
    s = f"{x:.6g}"
    return s


def build_grid(args: argparse.Namespace) -> list[str]:
    common = f"--env-id {args.env_id} --max-cycles {args.max_cycles} --total-timesteps {args.total_timesteps}"
    if (args.env_id.lower() in ("pd", "prisoners_dilemma", "repeated_prisoners_dilemma", "cleanup", "clean_up")
            or args.env_id.startswith("debug")):
        common += f" --num-agents {args.num_agents}"
    if args.extra:
        common += " " + args.extra.strip()

    ia_pairs = getattr(args, "ia_pairs", None)

    def alpha_arg(a: float) -> str:
        # --mixed "0,{a}" -> per-agent list with the grid alpha substituted
        return args.mixed.replace("{a}", fmt(a)) if args.mixed else fmt(a)

    lines: list[str] = []
    for seed in args.seeds:
        if 0.0 in args.alphas and "none" in args.formulations:
            lines.append(f"{common} --formulation none --seed {seed}")
        for signal in args.signals:
            for a in args.alphas:
                if a == 0.0:
                    continue
                if "ei" in args.formulations:
                    lines.append(f"{common} --formulation ei --signal {signal} --alpha {alpha_arg(a)} --seed {seed}")
                if "sia" in args.formulations:
                    lines.append(f"{common} --formulation sia --signal {signal} --alpha {alpha_arg(a)} --seed {seed}")
                if "svo" in args.formulations:
                    for name in args.phis:
                        lines.append(
                            f"{common} --formulation svo --signal {signal} --alpha {alpha_arg(a)} "
                            f"--phi {fmt(ALL_PHIS[name])} --seed {seed}"
                        )
                if "ia" in args.formulations and not ia_pairs:
                    for ratio in args.beta_ratios:
                        beta = a * float(ratio)
                        beta_arg = args.mixed.replace("{a}", fmt(beta)) if args.mixed else fmt(beta)
                        lines.append(
                            f"{common} --formulation ia --signal {signal} --alpha {alpha_arg(a)} "
                            f"--beta {beta_arg} --seed {seed}"
                        )
            if "ia" in args.formulations and ia_pairs:  # explicit (alpha, beta) pairs instead of alphas x ratios
                for pair in ia_pairs:
                    a_str, b_str = pair.split(":")
                    lines.append(
                        f"{common} --formulation ia --signal {signal} --alpha {alpha_arg(float(a_str))} "
                        f"--beta {fmt(float(b_str))} --seed {seed}"
                    )
    return lines


def latex_table(args: argparse.Namespace) -> str:
    alphas = ", ".join(fmt(a) for a in args.alphas)
    phis = ", ".join("0" if name == "0" else rf"\frac{{\pi}}{{{name.split('/')[1]}}}" for name in args.phis)
    betas = ", ".join(rf"\frac{{\alpha}}{{{r.denominator}}}" if r.numerator == 1 else rf"{r}\alpha" for r in args.beta_ratios)
    signals = " / ".join(args.signals)
    rows = []
    if "ei" in args.formulations:
        rows.append(rf"            EI & $\{{{alphas}\}}$ & N/A & N/A\\")
    if "sia" in args.formulations:
        rows.append(rf"            SIA & $\{{{alphas}\}}$ & N/A & N/A\\")
    if "svo" in args.formulations:
        rows.append(rf"            SVO & $\{{{alphas}\}}$ & N/A & $\{{{phis}\}}$\\")
    if "ia" in args.formulations:
        rows.append(rf"            IA & $\{{{alphas}\}}$ & $\{{{betas}\}}$ & N/A\\")
    body = "\n            \\hline\n".join(rows)
    return rf"""\begin{{figure}}[htbp]
        \centering
        \begin{{tabular}}{{|c|c|c|c|}}
            \hline
            \textbf{{Variation}} & \textbf{{$\alpha$}} & \textbf{{$\beta$}} &\textbf{{$\phi$}} \\
            \hline
{body}
            \hline
        \end{{tabular}}
    \caption{{Parameter configurations (signal: {signals}; {len(args.seeds)} seeds each).}}
    \label{{fig:paramConfigs}}
\end{{figure}}"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-id", default="pd")
    p.add_argument("--num-agents", type=int, default=2)
    p.add_argument("--max-cycles", type=int, default=100)
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--formulations", nargs="+", default=["none", "ei", "sia", "svo", "ia"])
    p.add_argument("--signals", nargs="+", default=["value", "reward"], choices=["value", "reward", "imagined"])
    p.add_argument("--alphas", nargs="+", type=float, default=REPORT_ALPHAS)
    p.add_argument("--phis", nargs="+", default=list(PHIS), choices=list(ALL_PHIS),
                   help="SVO angles; \"0\" is the own-value control (no other-regarding term)")
    p.add_argument("--beta-ratios", nargs="+", type=Fraction, default=BETA_RATIOS, help="beta = ratio * alpha")
    p.add_argument("--ia-pairs", nargs="*", default=None,
                   help='IA as explicit "alpha:beta" pairs (e.g. 5:0.05 0.05:5), replacing --alphas x --beta-ratios for ia')
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--mixed", default="", help='per-agent alpha template, e.g. "0,{a}" ({a} = grid alpha)')
    p.add_argument("--extra", default="", help="extra train.py arguments appended to every line")
    p.add_argument("-o", "--output", default=None, help="grid file to write (default: print to stdout)")
    p.add_argument("--latex", action="store_true", help="print the grid as a LaTeX table and exit")
    args = p.parse_args()

    if args.latex:
        print(latex_table(args))
        return
    lines = build_grid(args)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {len(lines)} jobs to {args.output}")
    else:
        print("\n".join(lines))


if __name__ == "__main__":
    main()
