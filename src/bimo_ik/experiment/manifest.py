"""Run manifest: everything needed to reproduce a results directory."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..config.schema import ExperimentConfig, config_hash, config_to_dict


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except OSError:
        return None


def _library_versions() -> dict[str, str]:
    import matplotlib
    import numpy
    import scipy
    import statsmodels

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": matplotlib.__version__,
    }


def write_manifest(config: ExperimentConfig, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash(config),
        "config": config_to_dict(config),
        "seed_scheme": (
            "per-run rng = default_rng(SeedSequence([run_seed, crc32(cell_id), "
            "crc32(trajectory_name)])); run_seed in "
            f"[{config.seed_base}, {config.seed_base + config.n_runs - 1}]"
        ),
        "run_seeds": list(config.run_seeds),
        "library_versions": _library_versions(),
        "platform": platform.platform(),
        "git_revision": _git_revision(),
    }
    path = output_dir / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path
