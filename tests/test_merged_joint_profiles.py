"""Camera-ready merged joint-profile figure: headless render + same series as
the per-figure versions (best run = lowest position RMSE, first minimum)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bimo_ik.plotting import fig_joint_profiles_merged, merged_joint_profile_runs
from bimo_ik.plotting.figures import _best_run

REPO = Path(__file__).resolve().parents[1]


def _fake_cell(rng, rmses):
    return {
        "runs": [{"seed": np.int64(42 + k), "q_path": rng.uniform(-3, 3, (30, 6))}
                 for k in range(len(rmses))],
        "accuracy": [{"rmse_pos_mm": r} for r in rmses],
    }


def _fake_inputs():
    rng = np.random.default_rng(0)
    per_cell = {
        "pso-cold-static": _fake_cell(rng, [5.0, 2.0, 3.0, 2.0]),
        "pso-injection-static": _fake_cell(rng, [1.0, 4.0, 0.5]),
        "pso-persistence-static": _fake_cell(rng, [0.1, 0.2]),
    }
    lm = {"cold": _fake_cell(rng, [9.0, 7.0, 8.0]), "warm": _fake_cell(rng, [0.0, 0.0, 1.0])}
    return per_cell, lm


def test_panels_use_the_existing_best_run_selection():
    per_cell, lm = _fake_inputs()
    panels = merged_joint_profile_runs(per_cell, lm)
    assert [t for t, _ in panels] == [
        "(a) PSO cold", "(b) PSO injection", "(c) TRF cold", "(d) TRF warm"]
    expected = [
        _best_run(per_cell["pso-cold-static"])["q_path"],       # fig_joint_profiles
        _best_run(per_cell["pso-injection-static"])["q_path"],
    ]
    for cond in ("cold", "warm"):                               # run_lm_baseline.py rule
        rows = lm[cond]["accuracy"]
        best = min(range(len(rows)), key=lambda k: rows[k]["rmse_pos_mm"])
        expected.append(lm[cond]["runs"][best]["q_path"])
    for (_, run), q in zip(panels, expected):
        np.testing.assert_array_equal(run["q_path"], q)
    # ties resolve to the first minimum, as in both existing figures
    assert [int(run["seed"]) for _, run in panels] == [43, 44, 43, 42]


def test_renders_headless_at_column_size(tmp_path):
    per_cell, lm = _fake_inputs()
    path = fig_joint_profiles_merged(per_cell, lm, "line", tmp_path)
    assert path.name == "joint_profiles_merged_line.png"
    assert path.with_suffix(".pdf").exists()
    assert Image.open(path).size == (1050, 1020)   # 3.5 x 3.4 in at 300 dpi


@pytest.mark.skipif(
    not (REPO / "results" / "lars" / "manifest.json").exists()
    or not (REPO / "results" / "lm_baseline" / "manifest.json").exists()
    or not any((REPO / "results" / "lars").rglob("*.npz")),
    reason="real results (.npz runs, not versioned) not present",
)
def test_real_data_seeds_match_existing_figures():
    from bimo_ik.config import config_from_manifest
    from bimo_ik.plotting import load_lm_condition
    from bimo_ik.stats import collect_results
    from bimo_ik.trajectories import build_trajectory

    lars = REPO / "results" / "lars"
    config = config_from_manifest(lars)
    per_cell = collect_results(lars, config)["line"]
    trajectory = build_trajectory(next(s for s in config.trajectories if s.name == "line"))
    lm = {c: load_lm_condition(REPO / "results" / "lm_baseline", trajectory, c)
          for c in ("cold", "warm")}
    seeds = [int(run["seed"]) for _, run in merged_joint_profile_runs(per_cell, lm)]
    assert seeds == [53, 67, 51, 45]
