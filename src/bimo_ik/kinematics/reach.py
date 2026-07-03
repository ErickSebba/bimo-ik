"""Characteristic scales of the robot, computed from the model — never hardcoded.

These scales make the fitness dimensionless: position errors are divided by
`max_reach()`, joint-space distances by `joint_range_norm()`.
"""

from __future__ import annotations

import numpy as np

from .dh import DH_PARAMS, JOINT_LOWER, JOINT_UPPER


def max_reach(dh_params: np.ndarray | None = None) -> float:
    """Upper bound on the end-effector distance from the base, from the DH table.

    Each DH row translates the frame origin by sqrt(a^2 + d^2); by the triangle
    inequality the end-effector can never lie farther from the base than the
    sum of those step lengths. For the BIMO table (one nonzero length per row)
    the bound is attained by the fully stretched configuration.
    """
    dh = DH_PARAMS if dh_params is None else np.asarray(dh_params, dtype=float)
    return float(np.sum(np.hypot(dh[:, 1], dh[:, 2])))


def joint_range_norm(
    lower: np.ndarray | None = None, upper: np.ndarray | None = None
) -> float:
    """Euclidean norm of the joint-range vector — the joint-space scale."""
    lo = JOINT_LOWER if lower is None else np.asarray(lower, dtype=float)
    up = JOINT_UPPER if upper is None else np.asarray(upper, dtype=float)
    return float(np.linalg.norm(up - lo))
