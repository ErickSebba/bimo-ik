"""Load a results directory into per-run metrics ready for stats/plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config.schema import ExperimentConfig
from ..experiment.engine import load_result, result_path
from ..metrics.accuracy import accuracy_summary
from ..metrics.effort import EffortTable, effort_table
from ..metrics.kinematic import kinematic_summary
from ..trajectories.generators import Trajectory, build_trajectory


def collect_results(
    results_dir: str | Path, config: ExperimentConfig
) -> dict[str, dict[str, dict]]:
    """{trajectory: {cell_id: {"runs": [raw dicts], "accuracy": [...],
    "kinematic": [...], "effort": [EffortTable], "wall_time_s": [...]}}}.

    Missing runs raise: statistics over a partial experiment would silently
    bias every comparison.
    """
    results_dir = Path(results_dir)
    eps = np.asarray(config.epsilon_values, dtype=float)
    collected: dict[str, dict[str, dict]] = {}

    for spec in config.trajectories:
        trajectory: Trajectory = build_trajectory(spec)
        per_cell: dict[str, dict] = {}
        for cell in config.cells:
            runs, acc, kin, eff, wall = [], [], [], [], []
            for seed in config.run_seeds:
                path = result_path(results_dir, spec.name, cell.cell_id, seed)
                if not path.exists():
                    raise FileNotFoundError(
                        f"missing run {path}; refusing to analyze a partial experiment"
                    )
                data = load_result(path)
                runs.append(data)
                acc.append(accuracy_summary(data["q_path"], trajectory))
                kin.append(kinematic_summary(data["q_path"]))
                eff.append(
                    effort_table(data["pose_error_hist"], data["evals_hist"], eps)
                )
                wall.append(float(data["wall_time_s"]))
            per_cell[cell.cell_id] = {
                "runs": runs,
                "accuracy": acc,
                "kinematic": kin,
                "effort": eff,
                "wall_time_s": wall,
            }
        collected[spec.name] = per_cell
    return collected


def metric_matrix(
    per_cell: dict[str, dict], family: str, metric: str
) -> dict[str, np.ndarray]:
    """values_by_cell for one scalar metric, ready for run_planned_comparisons.

    family: "accuracy" | "kinematic" | "wall_time_s" | "effort_iterations".
    For effort, `metric` is the epsilon value as float and the per-run scalar
    is the median first-crossing iteration count over trajectory points
    (censored points at the cap).
    """
    out: dict[str, np.ndarray] = {}
    for cell_id, data in per_cell.items():
        if family in ("accuracy", "kinematic"):
            out[cell_id] = np.array([run[metric] for run in data[family]])
        elif family == "wall_time_s":
            out[cell_id] = np.array(data["wall_time_s"])
        elif family == "effort_iterations":
            eps = float(metric)
            values = []
            for table in data["effort"]:
                k = int(np.argmin(np.abs(table.epsilon_values - eps)))
                values.append(float(np.median(table.iterations[k])))
            out[cell_id] = np.array(values)
        else:
            raise ValueError(f"unknown metric family {family!r}")
    return out
