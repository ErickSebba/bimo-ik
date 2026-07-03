"""Reachability guard: every target pose must be provably reachable.

A pose can be unreachable even when its position lies inside the workspace
(the orientation may be infeasible there under the joint limits). Before an
experiment runs, every pose of every trajectory is checked by solving a
bounded numerical IK (scipy least_squares, multi-start). A failure aborts
with `UnreachablePoseError` — the study never reports "error" that is
actually kinematic impossibility.

The guard's pose-error measure is the same dimensionless combination used by
the epsilon criterion with unit weights: e_pos / L_reach + e_ori / pi.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ..kinematics.dh import JOINT_LOWER, JOINT_UPPER, fk_pose
from ..kinematics.orientation import rotation_error, rotation_vector
from ..kinematics.reach import max_reach
from .generators import Trajectory


class UnreachablePoseError(RuntimeError):
    """Raised when a trajectory contains a pose the arm provably cannot meet
    within tolerance (or the solver cannot certify within its budget)."""


@dataclass(frozen=True)
class IKSolution:
    q: np.ndarray
    pose_error: float  # dimensionless: e_pos / L_reach + e_ori / pi


def _residuals(q, target_position, target_rotation, l_reach):
    position, rotation = fk_pose(q)
    pos_res = (position - target_position) / l_reach
    ori_res = rotation_vector(target_rotation.T @ rotation) / np.pi
    return np.concatenate([pos_res, ori_res])


def _pose_error(q, target_position, target_rotation, l_reach) -> float:
    position, rotation = fk_pose(q)
    return float(
        np.linalg.norm(position - target_position) / l_reach
        + rotation_error(rotation, target_rotation) / np.pi
    )


def solve_ik(
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    q_seed: np.ndarray | None = None,
    n_starts: int = 5,
    rng: np.random.Generator | None = None,
    tol: float = 1e-3,
) -> IKSolution:
    """Bounded numerical IK. Returns the best solution over multi-start."""
    rng = np.random.default_rng(0) if rng is None else rng
    l_reach = max_reach()
    target_position = np.asarray(target_position, dtype=float)
    target_rotation = np.asarray(target_rotation, dtype=float)

    starts: list[np.ndarray] = []
    if q_seed is not None:
        starts.append(np.clip(np.asarray(q_seed, dtype=float), JOINT_LOWER, JOINT_UPPER))
    starts.append((JOINT_LOWER + JOINT_UPPER) / 2.0)
    while len(starts) < n_starts:
        starts.append(rng.uniform(JOINT_LOWER, JOINT_UPPER))

    best: IKSolution | None = None
    for x0 in starts:
        result = least_squares(
            _residuals,
            x0,
            bounds=(JOINT_LOWER, JOINT_UPPER),
            args=(target_position, target_rotation, l_reach),
            method="trf",
        )
        err = _pose_error(result.x, target_position, target_rotation, l_reach)
        if best is None or err < best.pose_error:
            best = IKSolution(q=result.x.copy(), pose_error=err)
        if best.pose_error < tol / 10.0:  # already certified with margin
            break
    assert best is not None
    return best


def validate_trajectory(
    trajectory: Trajectory,
    tol: float = 1e-3,
    n_starts: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Certify every pose of a trajectory reachable within `tol`.

    Chains the previous point's solution as the first IK seed (fast and
    representative of trajectory tracking), plus fresh random restarts.
    Returns the per-point certified errors; raises UnreachablePoseError
    listing every failing point index.
    """
    rng = np.random.default_rng(seed)
    errors = np.empty(len(trajectory))
    q_seed: np.ndarray | None = None
    failures: list[tuple[int, float]] = []

    for i in range(len(trajectory)):
        solution = solve_ik(
            trajectory.positions[i],
            trajectory.rotations[i],
            q_seed=q_seed,
            n_starts=n_starts,
            rng=rng,
            tol=tol,
        )
        errors[i] = solution.pose_error
        q_seed = solution.q
        if solution.pose_error > tol:
            failures.append((i, solution.pose_error))

    if failures:
        detail = ", ".join(f"point {i} (err={e:.2e})" for i, e in failures)
        raise UnreachablePoseError(
            f"trajectory {trajectory.name!r}: {len(failures)} pose(s) not "
            f"certified reachable within tol={tol}: {detail}"
        )
    return errors
