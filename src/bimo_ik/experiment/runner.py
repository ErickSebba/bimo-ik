"""Execute one (cell, trajectory, seed) run and collect full histories.

Budget policy: every trajectory point runs to the fixed `max_iter` cap; the
executed joint solution is the incumbent at the cap, so kinematic metrics
compare identical budgets across cells. Per-iteration histories (fitness,
pose error, diversity, cumulative evaluations) are stored so the
effort-vs-epsilon analysis can be derived post hoc for any epsilon.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass

import numpy as np

from ..algorithms.ga import GA
from ..algorithms.pso import PSO
from ..config.schema import ExperimentConfig
from ..fitness.fitness import make_evaluator, scheduled_weights
from ..kinematics.dh import JOINT_LOWER, JOINT_UPPER
from ..kinematics.reach import joint_range_norm, max_reach
from ..factors import AlgorithmKind, Cell, LandscapeMode
from ..trajectories.generators import Trajectory, TrajectorySpec, build_trajectory


@dataclass
class RunResult:
    cell_id: str
    trajectory_name: str
    seed: int
    q_path: np.ndarray            # (N, 6) executed joint solutions
    fitness_hist: np.ndarray      # (N, max_iter) incumbent fitness per iteration
    pose_error_hist: np.ndarray   # (N, max_iter) incumbent pose error per iteration
    diversity_hist: np.ndarray    # (N, max_iter)
    evals_hist: np.ndarray        # (N, max_iter) cumulative evals within each point
    best_fitness: np.ndarray      # (N,)
    best_pose_error: np.ndarray   # (N,)
    wall_time_s: float


def run_rng(seed: int, cell_id: str, trajectory_name: str) -> np.random.Generator:
    """Deterministic, decorrelated generator for one run.

    The scheme (documented in the manifest) mixes the run seed with stable
    CRC32 digests of the cell and trajectory identifiers, so the same seed
    never reuses a stream across cells and results are identical no matter
    how runs are distributed over processes.
    """
    entropy = [seed, zlib.crc32(cell_id.encode()), zlib.crc32(trajectory_name.encode())]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def build_optimizer(cell: Cell, config: ExperimentConfig):
    if cell.algorithm is AlgorithmKind.PSO:
        return PSO(
            config.pso, cell.memory, JOINT_LOWER, JOINT_UPPER,
            injection_fraction=config.injection.fraction,
            injection_sigma=config.injection.sigma,
        )
    return GA(
        config.ga, cell.memory, JOINT_LOWER, JOINT_UPPER,
        injection_fraction=config.injection.fraction,
        injection_sigma=config.injection.sigma,
    )


def run_single(
    config: ExperimentConfig,
    cell: Cell,
    spec: TrajectorySpec,
    seed: int,
    trajectory: Trajectory | None = None,
) -> RunResult:
    trajectory = build_trajectory(spec) if trajectory is None else trajectory
    n_points = len(trajectory)
    rng = run_rng(seed, cell.cell_id, trajectory.name)
    optimizer = build_optimizer(cell, config)
    optimizer.reset()

    l_reach = max_reach()
    q_norm = joint_range_norm()
    q_ref = np.asarray(config.initial_posture, dtype=float)

    q_path = np.empty((n_points, len(q_ref)))
    fitness_hist = np.empty((n_points, config.max_iter))
    pose_error_hist = np.empty((n_points, config.max_iter))
    diversity_hist = np.empty((n_points, config.max_iter))
    evals_hist = np.empty((n_points, config.max_iter), dtype=np.int64)
    best_fitness = np.empty(n_points)
    best_pose_error = np.empty(n_points)

    t0 = time.perf_counter()
    for i in range(n_points):
        progress = i / (n_points - 1) if n_points > 1 else 0.0
        if cell.landscape is LandscapeMode.ADAPTIVE:
            weights = scheduled_weights(config.weights, progress)
        else:
            weights = config.weights
        evaluate = make_evaluator(
            trajectory.positions[i],
            trajectory.rotations[i],
            q_ref,
            weights,
            config.weights,
            l_reach,
            q_norm,
        )
        prev_best = None if i == 0 else q_ref
        point = optimizer.solve_point(evaluate, config.max_iter, rng, prev_best=prev_best)

        q_path[i] = point.best_q
        fitness_hist[i] = point.fitness_history
        pose_error_hist[i] = point.pose_error_history
        diversity_hist[i] = point.diversity_history
        evals_hist[i] = point.evals_history
        best_fitness[i] = point.best_fitness
        best_pose_error[i] = point.best_pose_error
        q_ref = point.best_q

    return RunResult(
        cell_id=cell.cell_id,
        trajectory_name=trajectory.name,
        seed=seed,
        q_path=q_path,
        fitness_hist=fitness_hist,
        pose_error_hist=pose_error_hist,
        diversity_hist=diversity_hist,
        evals_hist=evals_hist,
        best_fitness=best_fitness,
        best_pose_error=best_pose_error,
        wall_time_s=time.perf_counter() - t0,
    )
