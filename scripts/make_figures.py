"""Generate the standard figure set from a results directory.

    python scripts/make_figures.py --results results/lars [--output DIR]

The config is rebuilt from the directory's manifest.json (hash-verified).
"""

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from bimo_ik.config import config_from_manifest
    from bimo_ik.plotting import make_all_figures
    from bimo_ik.stats import collect_results

    config = config_from_manifest(args.results)
    collected = collect_results(args.results, config)
    out_dir = args.output or args.results / "figures"
    paths = make_all_figures(collected, config, out_dir)
    for p in paths:
        print(f"[fig] {p}")


if __name__ == "__main__":
    main()
