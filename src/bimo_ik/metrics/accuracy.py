"""Accuracy metrics, position and orientation reported separately.

All values are recomputed from the executed joint path via forward
kinematics — never taken from optimizer-internal fitness values — so they
are comparable across cells regardless of landscape mode.
"""

from __future__ import annotations

import numpy as np

from ..kinematics.dh import fk_pose
from ..kinematics.orientation import rotation_error
from ..trajectories.generators import Trajectory


def pose_errors(q_path: np.ndarray, trajectory: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    """Per-point (position error [mm], orientation error [rad]) of a run."""
    n = len(trajectory)
    pos_err = np.empty(n)
    ori_err = np.empty(n)
    for i in range(n):
        position, rotation = fk_pose(q_path[i])
        pos_err[i] = np.linalg.norm(position - trajectory.positions[i])
        ori_err[i] = rotation_error(rotation, trajectory.rotations[i])
    return pos_err, ori_err


def accuracy_summary(q_path: np.ndarray, trajectory: Trajectory) -> dict[str, float]:
    pos_err, ori_err = pose_errors(q_path, trajectory)
    return {
        "rmse_pos_mm": float(np.sqrt(np.mean(pos_err**2))),
        "mean_pos_mm": float(np.mean(pos_err)),
        "max_pos_mm": float(np.max(pos_err)),
        "rmse_ori_rad": float(np.sqrt(np.mean(ori_err**2))),
        "mean_ori_rad": float(np.mean(ori_err)),
        "max_ori_rad": float(np.max(ori_err)),
    }
