"""Reproduction anchor against the prior study (regression check, not equality).

Reruns the old paper's qualitative setup through the new engine: line
trajectory, 30 points, 80 iterations, 10 runs, position-dominant fitness
(w_ori = 0 — the old study was position-only), with three cells that map to
the old algorithms:

  pso-cold-static          ~ old PSO
  pso-persistence-adaptive ~ old PSO-DFL (persistent swarm + weight schedule)
  ga-cold-static           ~ old GA

PASS criteria (the old paper's qualitative result):
  1. GA-cold has the lowest mean positional RMSE of the three.
  2. PSO-persistence roughness (sum |dq|) is at least 20x lower (old: ~50x)
     than both cold baselines.

Exact numbers differ by construction (normalized fitness, different RNG);
this anchors the *ordering*, which is what the rebuilt engine must preserve.
"""

import _bootstrap  # noqa: F401

import argparse
import multiprocessing
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--output", type=Path, default=Path("results") / "sanity")
    args = parser.parse_args()

    from bimo_ik.config import ExperimentConfig
    from bimo_ik.experiment import Cell
    from bimo_ik.experiment.engine import run_experiment
    from bimo_ik.fitness import FitnessWeights
    from bimo_ik.stats import collect_results, metric_matrix
    from bimo_ik.trajectories import default_trajectory_specs

    config = ExperimentConfig(
        name="sanity",
        cells=(
            Cell("pso", "cold", "static"),
            Cell("pso", "persistence", "adaptive"),
            Cell("ga", "cold", "static"),
        ),
        trajectories=(default_trajectory_specs(30)["line"],),
        n_runs=10,
        seed_base=42,
        max_iter=80,
        weights=FitnessWeights(w_pos=1.0, w_ori=0.0, w_joint=0.1),
    )

    run_experiment(config, args.output, n_jobs=args.jobs)
    collected = collect_results(args.output, config)
    per_cell = collected["line"]

    rmse = metric_matrix(per_cell, "accuracy", "rmse_pos_mm")
    smooth = metric_matrix(per_cell, "kinematic", "roughness_rad")

    print("\n--- reproduction anchor (mean over 10 runs) ---")
    for cell in rmse:
        print(f"{cell:26s} RMSE={np.mean(rmse[cell]):8.3f} mm   "
              f"roughness={np.mean(smooth[cell]):8.2f} rad")

    ga_best_rmse = (
        np.mean(rmse["ga-cold-static"]) < np.mean(rmse["pso-cold-static"])
    )
    ratio_pso = np.mean(smooth["pso-cold-static"]) / np.mean(
        smooth["pso-persistence-adaptive"]
    )
    ratio_ga = np.mean(smooth["ga-cold-static"]) / np.mean(
        smooth["pso-persistence-adaptive"]
    )
    smooth_ok = min(ratio_pso, ratio_ga) >= 20.0

    print(f"\nGA-cold beats PSO-cold on mean positional RMSE: {ga_best_rmse}")
    print(f"persistence roughness advantage: {ratio_pso:.1f}x vs PSO-cold, "
          f"{ratio_ga:.1f}x vs GA-cold (need >= 20x)")
    verdict = "PASS" if (ga_best_rmse and smooth_ok) else "FAIL"
    print(f"\n[{verdict}] reproduction anchor")
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
