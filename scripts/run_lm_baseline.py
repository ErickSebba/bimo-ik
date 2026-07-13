"""Numerical-IK (LM) baseline: cold vs warm initial guesses.

    python scripts/run_lm_baseline.py [--reps 10] [--output results/lm_baseline]

Complementary context, NOT a factorial cell: descriptive statistics only
(mean +- sd and median/IQR) — no inferential tests against the stochastic
metaheuristic cells. See src/bimo_ik/experiment/lm_baseline.py.
"""

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np

METRIC_ORDER = (
    "rmse_pos_mm", "rmse_ori_rad", "roughness_rad", "jerk_rad",
    "peak_joint_velocity_rad", "peak_joint_acceleration_rad", "wall_time_s",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results") / "lm_baseline")
    args = parser.parse_args()

    from bimo_ik.experiment.lm_baseline import CONDITIONS, run_lm_condition
    from bimo_ik.metrics import accuracy_summary, kinematic_summary
    from bimo_ik.plotting import fig_lm_joint_profiles
    from bimo_ik.trajectories import build_trajectory, default_trajectory_specs

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_base, args.seed_base + args.reps))

    specs = default_trajectory_specs(30)
    trajectories = [build_trajectory(specs[name]) for name in ("line", "corner")]

    manifest = {
        "description": "LM (bounded least-squares) numerical-IK baseline, "
                       "cold vs warm initial guesses; descriptive only",
        "reps": args.reps,
        "seeds": seeds,
        "conditions": list(CONDITIONS),
        "trajectories": [t.name for t in trajectories],
        "solver": "scipy.optimize.least_squares (trf, bounded), single start, "
                  "no midpoint start",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = ["# Numerical-IK (LM) baseline — cold vs warm initial guess", "",
             f"{args.reps} repetitions per condition, seeds {seeds[0]}-{seeds[-1]}. "
             "Descriptive only (no inferential tests vs the stochastic cells).", ""]

    for trajectory in trajectories:
        per_condition_metrics: dict[str, list[dict]] = {}
        best_q_paths: dict[str, np.ndarray] = {}
        for condition in CONDITIONS:
            rows = []
            runs = []
            for seed in seeds:
                run = run_lm_condition(trajectory, condition, seed)
                runs.append(run)
                metrics = {
                    **accuracy_summary(run.q_path, trajectory),
                    **kinematic_summary(run.q_path),
                    "wall_time_s": run.wall_time_s,
                }
                rows.append(metrics)
                np.savez_compressed(
                    out / f"{trajectory.name}_{condition}_seed{seed:04d}.npz",
                    q_path=run.q_path,
                    pose_errors=run.pose_errors,
                    wall_time_s=np.float64(run.wall_time_s),
                    seed=np.int64(seed),
                )
            per_condition_metrics[condition] = rows
            best = min(range(len(rows)), key=lambda k: rows[k]["rmse_pos_mm"])
            best_q_paths[condition] = runs[best].q_path

        lines.append(f"## Trajectory: {trajectory.name}")
        lines.append("")
        lines.append("| metric | LM-cold mean ± sd | LM-cold median (IQR) "
                     "| LM-warm mean ± sd | LM-warm median (IQR) |")
        lines.append("|---|---|---|---|---|")
        for metric in METRIC_ORDER:
            cells = []
            for condition in CONDITIONS:
                v = np.array([r[metric] for r in per_condition_metrics[condition]])
                iqr = np.percentile(v, 75) - np.percentile(v, 25)
                cells.append(f"{v.mean():.4g} ± {v.std(ddof=1):.3g}")
                cells.append(f"{np.median(v):.4g} ({iqr:.3g})")
            lines.append(f"| {metric} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
        lines.append("")

        fig_path = fig_lm_joint_profiles(best_q_paths, trajectory.name, out)
        lines.append(f"Joint profiles (best run per condition): `{fig_path.name}`")
        lines.append("")
        print(f"[lm] {trajectory.name} done; figure: {fig_path}")

    report = out / "lm_baseline.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[lm] report: {report}")


if __name__ == "__main__":
    main()
