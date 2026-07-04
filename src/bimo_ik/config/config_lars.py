"""LARS (conference) preset: the memory-isolation story, static landscape.

Line + corner trajectories; the five static-landscape cells that answer
"what does population memory change, algorithm held fixed?" and
"which algorithm wins, memory policy held fixed?". Two epsilon thresholds
are highlighted in tables; the underlying data still carries the full sweep.
"""

from __future__ import annotations

from ..factors import Cell
from ..trajectories.generators import default_trajectory_specs
from .schema import ExperimentConfig

_CELL_IDS = (
    ("pso", "cold", "static"),
    ("pso", "injection", "static"),
    ("pso", "persistence", "static"),
    ("ga", "cold", "static"),
    ("ga", "injection", "static"),
)


def get_config() -> ExperimentConfig:
    specs = default_trajectory_specs(n_points=30)
    return ExperimentConfig(
        name="lars",
        cells=tuple(Cell(*ids) for ids in _CELL_IDS),
        trajectories=(specs["line"], specs["corner"]),
        n_runs=30,
        seed_base=42,
        max_iter=160,
    )
