"""Planned-comparison statistics from a results directory.

    python scripts/run_stats.py --results results/lars [--output DIR]

Writes one markdown report per trajectory and a descriptives JSON.

Inference policy per metric family:
- accuracy, effort (heavy-tailed: sd frequently exceeds the mean) —
  PRIMARY: Mann-Whitney U + Cliff's delta on the planned pairs, with a
  Kruskal-Wallis omnibus over the PSO memory triple; the Welch t table is
  retained beneath as a secondary check. Descriptives carry median/IQR.
- kinematic-executability, computation (well-behaved) — Welch t + Cohen's d.
One Benjamini-Hochberg FDR pass per family per track.
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
    "roughness_rad", "roughness_index", "jerk_rad",
    "peak_joint_velocity_rad", "peak_joint_acceleration_rad",
)
HEAVY_FAMILIES = {"accuracy", "effort"}
Q1_TRIPLE = ["pso-cold-static", "pso-injection-static", "pso-persistence-static"]

FOOTER = (
    "Effect-size guides: Cohen's d — |d| < 0.2 trivial, 0.2-0.5 small, "
    "0.5-0.8 medium, > 0.8 large. Cliff's delta — |delta| < 0.147 negligible, "
    "0.147-0.33 small, 0.33-0.474 medium, > 0.474 large. "
    "`*` = significant at FDR-adjusted p < 0.05 (one BH pass per family per track)."
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
        equivalence_analysis,
        equivalence_section_lines,
        kruskal_check,
        metric_matrix,
        rank_table_lines,
        run_planned_comparisons,
        run_rank_comparisons,
        welch_table_lines,
    )

    config = config_from_manifest(args.results)
    collected = collect_results(args.results, config)
    out_dir = args.output or args.results / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in config.trajectories:
        per_cell = collected[spec.name]
        descriptives: dict = {}
        lines = [
            f"# Planned comparisons — trajectory {spec.name!r} "
            f"(N={config.n_runs} runs/cell)",
            "",
        ]

        def family(name: str, family_key: str, metrics) -> None:
            welch, rank, kw_lines = [], [], []
            for metric in metrics:
                values = metric_matrix(per_cell, family_key, metric)
                descriptives.setdefault(name, {})[str(metric)] = {
                    cell: {
                        "mean": float(np.mean(v)),
                        "sd": float(np.std(v, ddof=1)),
                        "median": float(np.median(v)),
                        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
                        "n": int(len(v)),
                    }
                    for cell, v in values.items()
                }
                welch += run_planned_comparisons(values, str(metric))
                if name in HEAVY_FAMILIES:
                    rank += run_rank_comparisons(values, str(metric))
                    kw = kruskal_check(values, Q1_TRIPLE)
                    if kw is not None:
                        kw_lines.append(
                            f"- {metric}: H = {kw[0]:.3f}, p = {kw[1]:.2e}"
                        )

            lines.append(f"## Metric family: {name}")
            lines.append("")
            if name in HEAVY_FAMILIES:
                lines.append(
                    "### Rank-based planned comparisons (primary — "
                    "heavy-tailed family)"
                )
                lines.append("")
                lines.extend(rank_table_lines(adjust_family(rank)))
                lines.append("")
                if kw_lines:
                    lines.append(
                        "Kruskal-Wallis omnibus across the PSO memory triple "
                        "(cold / injection / persistence, static):"
                    )
                    lines.append("")
                    lines.extend(kw_lines)
                    lines.append("")
                lines.append("### Welch t (secondary check)")
                lines.append("")
            lines.extend(welch_table_lines(adjust_family(welch)))
            lines.append("")

        family("accuracy", "accuracy", ACCURACY_METRICS)
        family("kinematic-executability", "kinematic", KINEMATIC_METRICS)
        family("computation", "wall_time_s", ("wall_time_s",))
        family("effort", "effort_iterations",
               tuple(float(e) for e in config.report_epsilons))

        # TOST equivalence for the keystone memory-matched algorithm pair.
        equivalence_results = []
        for family_key, metric in (
            ("kinematic", "roughness_rad"),
            ("kinematic", "jerk_rad"),
            ("accuracy", "rmse_pos_mm"),
            ("accuracy", "rmse_ori_rad"),
        ):
            values = metric_matrix(per_cell, family_key, metric)
            result = equivalence_analysis(values, metric)
            if result is not None:
                equivalence_results.append(result)
        lines += equivalence_section_lines(equivalence_results)

        lines.append(FOOTER)
        lines.append("")

        report_path = out_dir / f"stats_{spec.name}.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        (out_dir / f"descriptives_{spec.name}.json").write_text(
            json.dumps(descriptives, indent=2), encoding="utf-8"
        )
        print(f"[stats] {report_path}")


if __name__ == "__main__":
    main()
