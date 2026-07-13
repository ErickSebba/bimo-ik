"""LM baseline: solver start control, cold/warm seeding semantics, determinism."""

import numpy as np
import pytest

from bimo_ik.experiment.lm_baseline import run_lm_condition
from bimo_ik.kinematics import JOINT_LOWER, JOINT_UPPER, fk_pose
from bimo_ik.trajectories import TrajectorySpec, build_trajectory, solve_ik

TINY = build_trajectory(
    TrajectorySpec("line", "line", 5, (150.0, 100.0, 200.0), (300.0, -100.0, 200.0))
)


def test_solve_ik_without_midpoint_still_certifies_from_a_seed():
    q_true = np.array([0.2, 0.4, 0.9, -0.3, 0.8, 0.5])
    pos, rot = fk_pose(q_true)
    sol = solve_ik(pos, rot, q_seed=q_true + 0.15, n_starts=1, include_midpoint=False)
    assert sol.pose_error < 1e-6


def test_solve_ik_default_behavior_unchanged():
    # guard path: midpoint start still present by default (n_starts respected)
    q_true = np.array([0.2, 0.4, 0.9, -0.3, 0.8, 0.5])
    pos, rot = fk_pose(q_true)
    assert solve_ik(pos, rot).pose_error < 1e-3


def test_warm_condition_chains_previous_solution_as_guess():
    run = run_lm_condition(TINY, "warm", seed=42)
    for i in range(1, len(TINY)):
        np.testing.assert_array_equal(run.initial_guesses[i], run.q_path[i - 1])


def test_cold_condition_uses_fresh_random_guesses():
    run = run_lm_condition(TINY, "cold", seed=42)
    for i in range(1, len(TINY)):
        # a fresh uniform draw, not the previous solution
        assert not np.allclose(run.initial_guesses[i], run.q_path[i - 1])
        assert np.all(run.initial_guesses[i] >= JOINT_LOWER)
        assert np.all(run.initial_guesses[i] <= JOINT_UPPER)


def test_lm_pose_errors_recorded_and_finite():
    # NOTE deliberately weak: single-start LM is NOT guaranteed to converge
    # (a random start frequently lands in a local minimum — that is itself a
    # finding of the baseline). What the runner guarantees is bookkeeping.
    for condition in ("cold", "warm"):
        run = run_lm_condition(TINY, condition, seed=43)
        assert run.q_path.shape == (len(TINY), 6)
        assert np.all(np.isfinite(run.pose_errors))
        assert np.all(run.pose_errors >= 0.0)
        assert np.all(run.q_path >= JOINT_LOWER) and np.all(run.q_path <= JOINT_UPPER)


def test_warm_path_is_smoother_than_cold_for_same_seed():
    # deterministic regression of the mechanism: warm follows one solution
    # branch (smooth), cold branch-switches between random basins (jagged)
    cold = run_lm_condition(TINY, "cold", seed=43)
    warm = run_lm_condition(TINY, "warm", seed=43)
    roughness = lambda q: float(np.sum(np.abs(np.diff(q, axis=0))))  # noqa: E731
    assert roughness(warm.q_path) < roughness(cold.q_path)


def test_lm_is_deterministic_per_seed():
    a = run_lm_condition(TINY, "cold", seed=44)
    b = run_lm_condition(TINY, "cold", seed=44)
    np.testing.assert_array_equal(a.q_path, b.q_path)
    c = run_lm_condition(TINY, "cold", seed=45)
    assert not np.array_equal(a.q_path, c.q_path)


def test_unknown_condition_rejected():
    with pytest.raises(ValueError):
        run_lm_condition(TINY, "lukewarm", seed=1)
