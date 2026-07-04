"""Kinematic-executability metrics from the executed joint path.

Per-step finite differences over the trajectory index (units rad/step):
these are kinematic proxies for executability — NOT dynamic quantities;
no torque, mass, or friction is modeled anywhere in this study.
"""

from __future__ import annotations

import numpy as np


def kinematic_summary(q_path: np.ndarray) -> dict[str, float]:
    dq = np.diff(q_path, axis=0)     # joint "velocity" per step
    ddq = np.diff(dq, axis=0)        # joint "acceleration" per step
    n_steps = len(q_path) - 1
    smoothness = float(np.sum(np.abs(dq)))
    return {
        "smoothness_rad": smoothness,
        "smoothness_index": smoothness / n_steps,
        "jerk_rad": float(np.sum(np.abs(ddq))),
        "peak_joint_velocity_rad": float(np.max(np.abs(dq))),
        "peak_joint_acceleration_rad": float(np.max(np.abs(ddq))),
    }
