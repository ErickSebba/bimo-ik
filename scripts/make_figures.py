"""Generate the standard figure set from a results directory.

    python scripts/make_figures.py --results results/lars [--output DIR]

The config is rebuilt from the directory's manifest.json (hash-verified).

    python scripts/make_figures.py --results results/lars --merged-joint-profiles \
        [--lm-results results/lm_baseline] [--output figures]

writes ONLY the camera-ready 2x2 joint-profile figure (PSO cold/injection +
LM cold/warm, line trajectory) as PNG + PDF; no other figure is regenerated.
Its default output is figures/ (outside results/).
"""

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--merged-joint-profiles", action="store_true",
                        help="write only joint_profiles_merged_line.png/.pdf")
    parser.add_argument("--lm-results", type=Path, default=Path("results") / "lm_baseline")
    args = parser.parse_args()

    from bimo_ik.config import config_from_manifest
    from bimo_ik.plotting import make_all_figures
    from bimo_ik.stats import collect_results

    config = config_from_manifest(args.results)
    collected = collect_results(args.results, config)

    if args.merged_joint_profiles:
        from bimo_ik.plotting import (
            fig_joint_profiles_merged,
            load_lm_condition,
            merged_joint_profile_runs,
        )
        from bimo_ik.trajectories import build_trajectory

        spec = next(s for s in config.trajectories if s.name == "line")
        trajectory = build_trajectory(spec)
        lm = {c: load_lm_condition(args.lm_results, trajectory, c) for c in ("cold", "warm")}
        per_cell = collected["line"]
        for title, run in merged_joint_profile_runs(per_cell, lm):
            print(f"[fig] {title}: best seed {int(run['seed'])}")
        path = fig_joint_profiles_merged(per_cell, lm, "line", args.output or Path("figures"))
        print(f"[fig] {path}\n[fig] {path.with_suffix('.pdf')}")
        return

    out_dir = args.output or args.results / "figures"
    paths = make_all_figures(collected, config, out_dir)
    for p in paths:
        print(f"[fig] {p}")


if __name__ == "__main__":
    main()
