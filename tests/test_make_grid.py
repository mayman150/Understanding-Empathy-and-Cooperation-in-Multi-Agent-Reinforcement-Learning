"""The grid generator must reproduce the report's parameter table and emit valid train.py lines."""
import importlib.util
import os
import shlex
import sys

import tyro

from empathy_marl.args import Args, resolve
from empathy_marl.empathy import parse_per_agent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("make_grid", os.path.join(ROOT, "scripts", "make_grid.py"))
make_grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_grid)


def _grid(**overrides):
    argv = sys.argv
    sys.argv = ["make_grid.py"]
    try:
        parser_args = make_grid.argparse.Namespace(
            env_id="pd", num_agents=2, max_cycles=100, total_timesteps=1_000_000,
            formulations=["none", "ei", "sia", "svo", "ia"], signals=["value", "reward"],
            alphas=make_grid.REPORT_ALPHAS, phis=list(make_grid.PHIS), beta_ratios=make_grid.BETA_RATIOS,
            seeds=[1, 2, 3, 4, 5], mixed="", extra="",
        )
        for k, v in overrides.items():
            setattr(parser_args, k, v)
        return make_grid.build_grid(parser_args), parser_args
    finally:
        sys.argv = argv


def test_report_table_counts():
    lines, _ = _grid()
    # per seed: 1 baseline + 2 signals x (EI 5 + SIA 5 + SVO 5*4 + IA 5*3) = 1 + 2 * 45 = 91
    assert len(lines) == 5 * 91
    seed1_value = [l for l in lines if l.endswith("--seed 1") and ("--signal value" in l or "--formulation none" in l)]
    counts = {f: sum(f"--formulation {f} " in l for l in seed1_value) for f in ("none", "ei", "sia", "svo", "ia")}
    assert counts == {"none": 1, "ei": 5, "sia": 5, "svo": 20, "ia": 15}
    assert not any("--alpha 0 " in l for l in lines)  # alpha = 0 collapses into the baseline


def test_every_line_is_a_valid_train_invocation():
    lines, _ = _grid(seeds=[7])
    for line in lines:
        a = resolve(tyro.cli(Args, args=shlex.split(line)))
        for name in ("alpha", "beta", "phi"):
            parse_per_agent(getattr(a, name), a.num_agents, name)
        assert a.seed == 7 and a.env_id == "pd"


def test_ia_beta_is_ratio_of_alpha_and_svo_phi_values():
    lines, _ = _grid(seeds=[1], signals=["value"], alphas=[0.3], formulations=["ia", "svo"])
    ia = sorted(l.split("--beta ")[1].split()[0] for l in lines if "--formulation ia" in l)
    assert ia == sorted(["0.15", "0.1", "0.03"])
    svo = sorted(float(l.split("--phi ")[1].split()[0]) for l in lines if "--formulation svo" in l)
    assert svo == sorted([1.5708, 1.0472, 0.785398, 0.523599])


def test_mixed_template_substitutes_grid_alpha_per_agent():
    lines, _ = _grid(seeds=[1], signals=["value"], alphas=[0.0, 20.0], formulations=["none", "ei", "ia"], mixed="0,{a}")
    ei = [l for l in lines if "--formulation ei" in l]
    assert len(ei) == 1 and "--alpha 0,20 " in ei[0] + " "
    ia = [l for l in lines if "--formulation ia" in l]
    assert any("--alpha 0,20 --beta 0,10 " in l + " " for l in ia)
    a = resolve(tyro.cli(Args, args=shlex.split(ei[0])))
    assert parse_per_agent(a.alpha, 2, "alpha").tolist() == [0.0, 20.0]


def test_latex_table_contains_all_rows():
    _, ns = _grid()
    tex = make_grid.latex_table(ns)
    for row in ("EI &", "SIA &", "SVO &", "IA &", r"\frac{\pi}{6}", r"\frac{\alpha}{10}", "0.003, 0.01, 0.03, 0.1, 0.3"):
        assert row in tex
