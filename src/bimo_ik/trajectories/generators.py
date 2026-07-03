"""Trajectory generators: line, arc, and sharp corner — all with full poses.

Positions are in millimetres. Orientations vary smoothly along the path:
quaternion SLERP between a start and an end orientation, parametrized by
normalized cumulative arc length (so the corner trajectory has no orientation
discontinuity even where the position path is sharp).

The default endpoint orientations were derived from the manipulator itself:
the tool orientation of a comfortable (regularized elbow-up) configuration
solved at the line-trajectory midpoint (225, 0, 200) mm, pre-rotated by
+15 deg / -15 deg about the world z axis. Every generated pose is verified
reachable by `reachability.validate_trajectory` before an experiment runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..kinematics.orientation import matrix_from_quat, slerp

# Provenance: see module docstring; validated by the reachability guard.
DEFAULT_QUAT_START = (0.49944473, 0.39597488, -0.11377559, 0.76211152)
DEFAULT_QUAT_END = (0.67967554, 0.35303507, -0.21238462, 0.60687739)

_KINDS = ("line", "arc", "corner")


@dataclass(frozen=True)
class TrajectorySpec:
    """Declarative trajectory description (JSON-serializable, hashable)."""

    name: str
    kind: str  # "line" | "arc" | "corner"
    n_points: int
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    quat_start: tuple[float, float, float, float] = DEFAULT_QUAT_START
    quat_end: tuple[float, float, float, float] = DEFAULT_QUAT_END
    corner: tuple[float, float, float] | None = None  # required for kind="corner"

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f"unknown trajectory kind {self.kind!r}")
        if self.kind == "corner" and self.corner is None:
            raise ValueError("corner trajectories require a corner waypoint")


@dataclass(frozen=True)
class Trajectory:
    """Materialized trajectory: positions (N, 3) and rotations (N, 3, 3)."""

    name: str
    positions: np.ndarray = field(repr=False)
    rotations: np.ndarray = field(repr=False)

    def __len__(self) -> int:
        return len(self.positions)


def _line_positions(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    return np.linspace(start, end, n)


def _arc_positions(start: np.ndarray, end: np.ndarray, n: int) -> np.ndarray:
    """Semicircular arc over the start-end chord, in the horizontal plane,
    bulging away from the base column (toward +cross(u, z))."""
    if abs(start[2] - end[2]) > 1e-9:
        raise ValueError("arc trajectories require start and end at the same height")
    center = (start + end) / 2.0
    radius = float(np.linalg.norm(end - start)) / 2.0
    u = (start - center) / radius
    v = np.cross(u, np.array([0.0, 0.0, 1.0]))
    v /= np.linalg.norm(v)
    theta = np.linspace(0.0, np.pi, n)
    return center + radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)


def _corner_positions(
    start: np.ndarray, corner: np.ndarray, end: np.ndarray, n: int
) -> np.ndarray:
    """Two straight legs with a sharp corner, points split proportionally to
    leg length; the corner waypoint appears exactly once."""
    len1 = float(np.linalg.norm(corner - start))
    len2 = float(np.linalg.norm(end - corner))
    n1 = max(2, round(n * len1 / (len1 + len2)))
    n1 = min(n1, n - 2)  # keep at least 2 points on the second leg
    leg1 = np.linspace(start, corner, n1, endpoint=False)
    leg2 = np.linspace(corner, end, n - n1)
    return np.vstack([leg1, leg2])


def _arc_length_parameter(positions: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(steps)])
    return s / s[-1]


def build_trajectory(spec: TrajectorySpec) -> Trajectory:
    start = np.asarray(spec.start, dtype=float)
    end = np.asarray(spec.end, dtype=float)

    if spec.kind == "line":
        positions = _line_positions(start, end, spec.n_points)
    elif spec.kind == "arc":
        positions = _arc_positions(start, end, spec.n_points)
    else:  # corner (validated in __post_init__)
        positions = _corner_positions(
            start, np.asarray(spec.corner, dtype=float), end, spec.n_points
        )

    t = _arc_length_parameter(positions)
    rotations = np.array(
        [matrix_from_quat(slerp(spec.quat_start, spec.quat_end, ti)) for ti in t]
    )
    return Trajectory(name=spec.name, positions=positions, rotations=rotations)


def default_trajectory_specs(n_points: int = 30) -> dict[str, TrajectorySpec]:
    """The three study trajectories. The line reuses the prior paper's
    endpoints (150, 100, 200) -> (300, -100, 200) mm."""
    start = (150.0, 100.0, 200.0)
    end = (300.0, -100.0, 200.0)
    return {
        "line": TrajectorySpec("line", "line", n_points, start, end),
        "arc": TrajectorySpec("arc", "arc", n_points, start, end),
        "corner": TrajectorySpec(
            "corner", "corner", n_points, start, end, corner=(300.0, 100.0, 200.0)
        ),
    }
