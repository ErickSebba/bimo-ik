"""Declarative experiment configuration.

Everything an experiment depends on — cells, trajectories, run counts,
budget, epsilon sweep, weights, hyperparameters — lives in one frozen,
JSON-serializable dataclass. `config_hash` gives a stable fingerprint that
the run manifest records, so any results directory can be traced to the
exact configuration that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from ..algorithms.ga import GAParams
from ..algorithms.pso import PSOParams
from ..factors import Cell
from ..fitness.fitness import FitnessWeights
from ..trajectories.generators import TrajectorySpec


@dataclass(frozen=True)
class InjectionParams:
    fraction: float = 0.25
    sigma: float = 0.1  # rad


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    cells: tuple[Cell, ...]
    trajectories: tuple[TrajectorySpec, ...]
    n_runs: int
    seed_base: int
    max_iter: int
    weights: FitnessWeights = FitnessWeights()
    pso: PSOParams = PSOParams()
    ga: GAParams = GAParams()
    injection: InjectionParams = InjectionParams()
    # epsilon sweep for the effort-vs-epsilon analysis (dimensionless pose error)
    epsilon_values: tuple[float, ...] = tuple(
        float(f"{v:.6g}")
        for v in (
            1e-4, 1.7783e-4, 3.1623e-4, 5.6234e-4,
            1e-3, 1.7783e-3, 3.1623e-3, 5.6234e-3,
            1e-2, 1.7783e-2, 3.1623e-2, 5.6234e-2, 1e-1,
        )
    )
    # thresholds highlighted in tables (read off the same curve)
    report_epsilons: tuple[float, ...] = (1e-3, 1e-2)
    reach_tolerance: float = 1e-3
    initial_posture: tuple[float, ...] = (0.0,) * 6

    def __post_init__(self):
        ids = [c.cell_id for c in self.cells]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate cells in config")
        names = [t.name for t in self.trajectories]
        if len(set(names)) != len(names):
            raise ValueError("duplicate trajectory names in config")
        if not all(e in self.epsilon_values for e in self.report_epsilons):
            raise ValueError("report_epsilons must be part of epsilon_values")

    @property
    def run_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.seed_base, self.seed_base + self.n_runs))


def _jsonify(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def config_to_dict(config: ExperimentConfig) -> dict:
    return _jsonify(asdict(config))


def config_hash(config: ExperimentConfig) -> str:
    canonical = json.dumps(config_to_dict(config), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def config_from_dict(d: dict) -> ExperimentConfig:
    """Inverse of config_to_dict (used to rebuild a config from manifest.json)."""
    from ..algorithms.ga import GAParams
    from ..algorithms.pso import PSOParams
    from ..fitness.fitness import FitnessWeights

    cells = tuple(
        Cell(c["algorithm"], c["memory"], c["landscape"]) for c in d["cells"]
    )
    trajectories = tuple(
        TrajectorySpec(
            name=t["name"],
            kind=t["kind"],
            n_points=t["n_points"],
            start=tuple(t["start"]),
            end=tuple(t["end"]),
            quat_start=tuple(t["quat_start"]),
            quat_end=tuple(t["quat_end"]),
            corner=tuple(t["corner"]) if t["corner"] is not None else None,
        )
        for t in d["trajectories"]
    )
    return ExperimentConfig(
        name=d["name"],
        cells=cells,
        trajectories=trajectories,
        n_runs=d["n_runs"],
        seed_base=d["seed_base"],
        max_iter=d["max_iter"],
        weights=FitnessWeights(**d["weights"]),
        pso=PSOParams(**d["pso"]),
        ga=GAParams(**d["ga"]),
        injection=InjectionParams(**d["injection"]),
        epsilon_values=tuple(d["epsilon_values"]),
        report_epsilons=tuple(d["report_epsilons"]),
        reach_tolerance=d["reach_tolerance"],
        initial_posture=tuple(d["initial_posture"]),
    )


def config_from_manifest(results_dir) -> ExperimentConfig:
    """Load the config recorded in a results directory's manifest.json,
    verifying the stored hash."""
    from pathlib import Path

    manifest = json.loads((Path(results_dir) / "manifest.json").read_text())
    config = config_from_dict(manifest["config"])
    if config_hash(config) != manifest["config_hash"]:
        raise ValueError(
            "manifest config_hash mismatch: results were produced by a "
            "different code/config state"
        )
    return config
