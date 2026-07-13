"""Numerical-IK (Levenberg-Marquardt-style least-squares) baseline, cold vs warm.

Complementary context for the metaheuristic study, NOT a factorial cell: the
same bounded least-squares solver the reachability guard uses is run under
the same two memory conditions the metaheuristics face —

- ``cold``: a fresh random initial guess (within joint limits) at every
  trajectory point;
- ``warm``: the previous point's solution as the initial guess (the analogue
  of injection; the first point degenerates to a random guess).

The hypothesis this makes testable: a memoryless classical solver should
branch-switch between neighbouring IK solutions (jagged joint profiles),
while the warm-started one should be smooth — which would generalize the
memory finding beyond population-based search. The experiment is reported
descriptively only (the solver is near-deterministic once seeded; inferential
tests against the stochastic cells would be apples-to-oranges).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..kinematics.dh import JOINT_LOWER, JOINT_UPPER
from ..trajectories.generators import Trajectory
from ..trajectories.reachability import solve_ik

CONDITIONS = ("cold", "warm")


@dataclass
class LMRunResult:
    condition: str
    trajectory_name: str
    seed: int
    q_path: np.ndarray        # (N, 6) solved joint path
    pose_errors: np.ndarray   # (N,) dimensionless certified pose errors
    initial_guesses: np.ndarray  # (N, 6) the actual starts used (for tests)
    wall_time_s: float


def run_lm_condition(trajectory: Trajectory, condition: str, seed: int) -> LMRunResult:
    """One repetition of the LM baseline over a trajectory."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown LM condition {condition!r}")
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, hash_condition(condition, trajectory.name)])
    )
    n_points = len(trajectory)
    q_path = np.empty((n_points, len(JOINT_LOWER)))
    pose_errors = np.empty(n_points)
    guesses = np.empty_like(q_path)

    q_prev: np.ndarray | None = None
    t0 = time.perf_counter()
    for i in range(n_points):
        if condition == "cold" or q_prev is None:
            guess = rng.uniform(JOINT_LOWER, JOINT_UPPER)
        else:
            guess = q_prev
        solution = solve_ik(
            trajectory.positions[i],
            trajectory.rotations[i],
            q_seed=guess,
            n_starts=1,
            rng=rng,
            include_midpoint=False,  # the seeded start is the ONLY start
        )
        q_path[i] = solution.q
        pose_errors[i] = solution.pose_error
        guesses[i] = guess
        q_prev = solution.q
    wall_time = time.perf_counter() - t0

    return LMRunResult(
        condition=condition,
        trajectory_name=trajectory.name,
        seed=seed,
        q_path=q_path,
        pose_errors=pose_errors,
        initial_guesses=guesses,
        wall_time_s=wall_time,
    )


def hash_condition(condition: str, trajectory_name: str) -> int:
    """Stable integer for seed decorrelation across (condition, trajectory)."""
    import zlib

    return zlib.crc32(f"lm-{condition}-{trajectory_name}".encode())
