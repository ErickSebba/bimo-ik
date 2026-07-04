"""Run an experiment preset.

    python scripts/run_experiment.py --config lars [--smoke] [--jobs N] [--output DIR]

--smoke shrinks the preset (2 runs, 10 points, 20 iterations) for a fast
end-to-end check; it writes to its own results directory and never mixes
with real data (different config name => different manifest hash).
"""

import _bootstrap  # noqa: F401

import argparse
import multiprocessing
from dataclasses import replace
from pathlib import Path


def load_config(name: str, smoke: bool):
    if name == "lars":
        from bimo_ik.config.config_lars import get_config
    elif name == "full":
        from bimo_ik.config.config_full import get_config
    else:
        raise SystemExit(f"unknown config preset {name!r}")
    config = get_config()
    if smoke:
        config = replace(
            config,
            name=f"{config.name}-smoke",
            n_runs=2,
            max_iter=20,
            trajectories=tuple(
                replace(spec, n_points=10) for spec in config.trajectories
            ),
        )
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("lars", "full"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--jobs", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.smoke)
    output = args.output or Path("results") / config.name

    from bimo_ik.experiment.engine import run_experiment

    written = run_experiment(config, output, n_jobs=args.jobs)
    print(f"\n[done] {len(written)} run(s) written under {output}")
    print("next: scripts/make_figures.py and scripts/run_stats.py with "
          f"--results {output}")


if __name__ == "__main__":
    main()
