"""SO(3) utilities: geodesic orientation error, quaternions, SLERP.

Quaternions are (w, x, y, z), always unit-normalized.
"""

from __future__ import annotations

import numpy as np


def rotation_error(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices, in [0, pi].

    The arccos argument is clipped to [-1, 1]: floating-point products
    routinely push the trace fractionally outside the valid range, which
    would otherwise return NaN.
    """
    cos_angle = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Unit quaternion from a rotation matrix (Shepperd's branch selection)."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0.0:
        s = 2.0 * np.sqrt(tr + 1.0)
        quat = np.array(
            [0.25 * s,
             (R[2, 1] - R[1, 2]) / s,
             (R[0, 2] - R[2, 0]) / s,
             (R[1, 0] - R[0, 1]) / s]
        )
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        quat = np.array(
            [(R[2, 1] - R[1, 2]) / s,
             0.25 * s,
             (R[0, 1] + R[1, 0]) / s,
             (R[0, 2] + R[2, 0]) / s]
        )
    elif R[1, 1] >= R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        quat = np.array(
            [(R[0, 2] - R[2, 0]) / s,
             (R[0, 1] + R[1, 0]) / s,
             0.25 * s,
             (R[1, 2] + R[2, 1]) / s]
        )
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        quat = np.array(
            [(R[1, 0] - R[0, 1]) / s,
             (R[0, 2] + R[2, 0]) / s,
             (R[1, 2] + R[2, 1]) / s,
             0.25 * s]
        )
    return quat / np.linalg.norm(quat)


def matrix_from_quat(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix from a (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between unit quaternions (shortest arc)."""
    q0 = np.asarray(q0, dtype=float) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=float) / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # antipodal representative: take the short way around
        q1, dot = -q1, -dot
    if dot > 0.9995:  # nearly parallel: lerp is numerically stable
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s0 = np.sin((1.0 - t) * theta) / np.sin(theta)
    s1 = np.sin(t * theta) / np.sin(theta)
    return s0 * q0 + s1 * q1


def rotation_vector(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector of R (SO(3) log map), robust near 0 and pi."""
    w, x, y, z = quat_from_matrix(R)
    v = np.array([x, y, z])
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(n, abs(w))
    axis = v / n if w >= 0.0 else -v / n
    return angle * axis
