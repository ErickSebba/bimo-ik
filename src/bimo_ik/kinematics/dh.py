"""Denavit-Hartenberg model of the BIMO 6-DoF arm.

The DH table and joint limits are the canonical values from the prior study's
reference implementation (reference/main.py). Angles are in radians, lengths
in millimetres. Unlike the reference code, the forward kinematics here keeps
the full rotation matrix: targets in this study are full poses.
"""

from __future__ import annotations

import numpy as np

N_JOINTS = 6

# Rows: (theta_offset, d, a, alpha) per joint.
DH_PARAMS = np.array(
    [
        [0.0, 184.0, 0.0, np.pi / 2],   # joint 1
        [0.0, 0.0, 222.0, 0.0],          # joint 2
        [0.0, 0.0, 205.0, 0.0],          # joint 3
        [0.0, 0.0, 0.0, -np.pi / 2],     # joint 4
        [0.0, 0.0, 0.0, np.pi / 2],      # joint 5
        [0.0, 75.0, 0.0, 0.0],           # joint 6
    ]
)

JOINT_LOWER = np.array(
    [-np.pi, -np.pi * 110 / 180, -np.pi * 80 / 180,
     -np.pi, -np.pi * 50 / 180, -np.pi]
)
JOINT_UPPER = np.array(
    [np.pi, np.pi * 155 / 180, np.pi * 210 / 180,
     np.pi, np.pi * 230 / 180, np.pi]
)


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Homogeneous transform of a single DH row."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def fk_transform(q: np.ndarray) -> np.ndarray:
    """Full homogeneous transform base -> tool for joint vector q (6,)."""
    T = np.eye(4)
    for i in range(N_JOINTS):
        offset, d, a, alpha = DH_PARAMS[i]
        T = T @ dh_transform(q[i] + offset, d, a, alpha)
    return T


def fk_pose(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """End-effector pose: (position (3,), rotation matrix (3, 3))."""
    T = fk_transform(q)
    return T[:3, 3], T[:3, :3]


def fk_position(q: np.ndarray) -> np.ndarray:
    """End-effector position only (3,)."""
    return fk_transform(q)[:3, 3]


def link_positions(q: np.ndarray) -> np.ndarray:
    """Origins of the base frame and every link frame, shape (7, 3)."""
    T = np.eye(4)
    points = [T[:3, 3].copy()]
    for i in range(N_JOINTS):
        offset, d, a, alpha = DH_PARAMS[i]
        T = T @ dh_transform(q[i] + offset, d, a, alpha)
        points.append(T[:3, 3].copy())
    return np.array(points)
