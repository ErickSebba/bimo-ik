"""Planned-comparison statistics from a results directory.

    python scripts/run_stats.py --results results/lars [--output DIR]

Writes one markdown report per trajectory (descriptives + Welch t with one
BH-FDR pass per metric family + Cohen's d) and a descriptives JSON.
"""

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np

ACCURACY_METRICS = (
    "rmse_pos_mm", "mean_pos_mm", "max_pos_mm",
    "rmse_ori_rad", "mean_ori_rad", "max_ori_rad",
)
KINEMATIC_METRICS = (
    "smoothness_rad", "smoothness_index", "jerk_rad",
    "peak_joint_velocity_rad", "peak_joint_acceleration_rad",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from bimo_ik.config import config_from_manifest
    from bimo_ik.stats import (
        adjust_family,
        collect_results,
        format_report,
        metric_matrix,
        run_planned_comparisons,
    )

    config = config_from_manifest(args.results)
    collected = collect_results(args.results, config)
    out_dir = args.output or args.results / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in config.trajectories:
        per_cell = collected[spec.name]
        families = {}
        descriptives = {}

        def family(name, family_key, metrics):
            comparisons = []
            for metric in metrics:
                values = metric_matrix(per_cell, family_key, metric)
                descriptives.setdefault(name, {})[str(metric)] = {
                    cell: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                           "n": int(len(v))}
                    for cell, v in values.items()
                }
                comparisons += run_planned_comparisons(values, str(metric))
            families[name] = adjust_family(comparisons)  # one FDR pass / family

        family("accuracy", "accuracy", ACCURACY_METRICS)
        family("kinematic-executability", "kinematic", KINEMATIC_METRICS)
        family("computation", "wall_time_s", ("wall_time_s",))
        family("effort", "effort_iterations",
               tuple(float(e) for e in config.report_epsilons))

        report = format_report(
            families,
            f"Planned comparisons — trajectory {spec.name!r} "
            f"(N={config.n_runs} runs/cell)",
        )
        report_path = out_dir / f"stats_{spec.name}.md"
        report_path.write_text(report, encoding="utf-8")
        (out_dir / f"descriptives_{spec.name}.json").write_text(
            json.dumps(descriptives, indent=2), encoding="utf-8"
        )
        print(f"[stats] {report_path}")


if __name__ == "__main__":
    main()
