"""Full (journal) preset: all 10 cells, 3 trajectories, full epsilon sweep.

The weight-sensitivity slice (w_ori/w_pos in {0.5, 1, 2}) is exposed as
separate configs run over a reduced cell subset; the continuity ranking is
checked for stability across the sweep.
"""

from __future__ import annotations

from ..factors import full_matrix
from ..fitness.fitness import FitnessWeights
from ..trajectories.generators import default_trajectory_specs
from .schema import ExperimentConfig


def get_config() -> ExperimentConfig:
    specs = default_trajectory_specs(n_points=30)
    return ExperimentConfig(
        name="full",
        cells=full_matrix(),
        trajectories=(specs["line"], specs["arc"], specs["corner"]),
        n_runs=30,
        seed_base=42,
        max_iter=160,
    )


# Sensitivity slice: does the continuity ranking depend on w_ori/w_pos?
_SENSITIVITY_CELLS = (
    ("pso", "cold", "static"),
    ("pso", "persistence", "static"),
    ("ga", "cold", "static"),
)


def get_sensitivity_configs() -> tuple[ExperimentConfig, ...]:
    from ..factors import Cell

    specs = default_trajectory_specs(n_points=30)
    configs = []
    for ratio in (0.5, 2.0):  # ratio 1.0 is the main experiment itself
        configs.append(
            ExperimentConfig(
                name=f"sensitivity-wo{ratio:g}",
                cells=tuple(Cell(*ids) for ids in _SENSITIVITY_CELLS),
                trajectories=(specs["line"],),
                n_runs=30,
                seed_base=42,
                max_iter=160,
                weights=FitnessWeights(w_pos=1.0, w_ori=ratio, w_joint=0.1),
            )
        )
    return tuple(configs)
